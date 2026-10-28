use std::collections::HashMap;

use crate::sparse_error::SparseError;
use crate::Scalar;

/// A sparse matrix of `T` values (`f64` by default; any [`Scalar`](crate::Scalar) type works).
///
/// Entries live in a dictionary of keys (a `HashMap` from `(row, col)` to value), which
/// makes single-element reads and writes cheap. A zero is never stored: inserting a zero
/// removes any existing entry, so `num_nonzero` is the number of stored entries.
///
/// A compressed sparse row (CSR) copy is built on demand by `explicitly_compress` and
/// exposed through `compressed`. Any mutation makes that copy stale. To support efficient
/// operations the storage should eventually move to CSR/CSC proper.
#[derive(Clone)]
pub struct SparseMatrix<T = f64> {
    /// The matrix dimensions as `(rows, cols)`.
    pub shape: (usize, usize),
    /// The stored (non-zero) entries, keyed by `(row, col)`.
    values: HashMap<(usize, usize), T>,

    /// Whether the `compressed_*` arrays reflect the current `values`.
    compressed_updated: bool,
    /// Cached CSR row pointers (`rows + 1` entries once built).
    compressed_rowarray: Vec<usize>,
    /// Cached CSR column indices, sorted within each row.
    compressed_colarray: Vec<usize>,
    /// Cached CSR values, parallel to `compressed_colarray`.
    compressed_dataarray: Vec<T>,
}

/// Borrowed view of the matrix's cached CSR arrays, as returned by
/// `SparseMatrix::compressed`.
pub struct CsrView<'a, T = f64> {
    /// Row pointers: row `i`'s entries are `cols[rows[i]..rows[i + 1]]`.
    pub rows: &'a [usize],
    /// Column index of each stored entry, sorted within each row.
    pub cols: &'a [usize],
    /// Value of each stored entry, parallel to `cols`.
    pub data: &'a [T],
}

/// Standalone CSR arrays, built without touching the matrix's cached copy.
struct Csr<T> {
    /// Row pointers (`rows + 1` entries).
    rows: Vec<usize>,
    /// Column index of each stored entry, sorted within each row.
    cols: Vec<usize>,
    /// Value of each stored entry, parallel to `cols`.
    data: Vec<T>,
}

/// Iterator over a matrix's rows, created by `SparseMatrix::row_iter`.
///
/// Each item is one dense row of length `shape.1`.
pub struct RowIterator<'a, T = f64> {
    matrix: &'a SparseMatrix<T>,
    /// Only set when the matrix's cached CSR arrays are stale.
    owned: Option<Csr<T>>,
    row_iter_idx: usize,
}

impl<T: Scalar> Iterator for RowIterator<'_, T> {
    /// One dense row.
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_iter_idx < self.matrix.shape.0 {
            let (rows, cols, data) = match &self.owned {
                Some(csr) => (&csr.rows, &csr.cols, &csr.data),
                None => (
                    &self.matrix.compressed_rowarray,
                    &self.matrix.compressed_colarray,
                    &self.matrix.compressed_dataarray,
                ),
            };
            let start = rows[self.row_iter_idx];
            let end = rows[self.row_iter_idx + 1];

            let colslice = &cols[start..end];
            let dataslice = &data[start..end];

            let mut retvec: Vec<T> = vec![T::zero(); self.matrix.shape.1];

            for (col, val) in std::iter::zip(colslice, dataslice) {
                retvec[*col] = *val;
            }

            self.row_iter_idx += 1;

            Some(retvec)
        } else {
            None
        }
    }
}

impl<T: Scalar> Default for SparseMatrix<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> SparseMatrix<T> {
    /// Builds CSR arrays from `values` without modifying the cache.
    fn build_csr(&self) -> Csr<T> {
        // Create row vecs that we'll sort by col
        let mut row_vecs: Vec<Vec<(usize, T)>> = vec![];
        for _ in 0..self.shape.0 {
            row_vecs.push(vec![]);
        }

        for ((row, col), val) in self.values.iter() {
            row_vecs[*row].push((*col, *val));
        }
        for row in row_vecs.iter_mut() {
            row.sort_by_key(|a| a.0);
        }

        let mut csr = Csr {
            rows: vec![0],
            cols: vec![],
            data: vec![],
        };
        for row in row_vecs {
            for (col, val) in row {
                csr.cols.push(col);
                csr.data.push(val);
            }
            csr.rows.push(csr.data.len());
        }
        csr
    }

    /// Rebuilds the cached CSR arrays and marks them current.
    fn _update_compressed(&mut self) {
        let csr = self.build_csr();
        self.compressed_rowarray = csr.rows;
        self.compressed_colarray = csr.cols;
        self.compressed_dataarray = csr.data;

        self.compressed_updated = true
    }

    /// Rebuilds the cached CSR arrays so that `compressed` returns `Some` again.
    ///
    /// Worthwhile before iterating repeatedly, since `row_iter` otherwise rebuilds the
    /// arrays for every iterator while the cache is stale.
    pub fn explicitly_compress(&mut self) {
        self._update_compressed();
    }

    /// The cached CSR arrays, or `None` if they are stale (or were never built).
    /// Call `explicitly_compress` to refresh them.
    pub fn compressed(&self) -> Option<CsrView<'_, T>> {
        if self.compressed_updated {
            Some(CsrView {
                rows: &self.compressed_rowarray,
                cols: &self.compressed_colarray,
                data: &self.compressed_dataarray,
            })
        } else {
            None
        }
    }

    /// Iterates over the rows of the matrix, yielding each as a dense `Vec<T>`.
    ///
    /// Always reflects the current contents. Call `explicitly_compress` first when
    /// iterating repeatedly; otherwise each iterator rebuilds the CSR arrays itself.
    #[allow(dead_code)]
    pub fn row_iter(&self) -> RowIterator<'_, T> {
        RowIterator {
            matrix: self,
            owned: if self.compressed_updated {
                None
            } else {
                Some(self.build_csr())
            },
            row_iter_idx: 0,
        }
    }

    /// Creates an empty `0 x 0` matrix.
    #[allow(dead_code)]
    pub fn new() -> SparseMatrix<T> {
        Self {
            shape: (0, 0),
            values: HashMap::new(),
            compressed_updated: false,
            compressed_rowarray: vec![],
            compressed_colarray: vec![],
            compressed_dataarray: vec![],
        }
    }

    /// Creates an `n x m` matrix with no stored entries.
    #[allow(dead_code)]
    pub fn empty_with_shape(n: usize, m: usize) -> SparseMatrix<T> {
        // No up-front reservation: n * m says nothing about how many entries a sparse
        //   matrix will hold, and can overflow or be enormous. The map grows as needed.
        Self::from_values((n, m), HashMap::new())
    }

    /// Wraps an existing entry map in a matrix of the given shape, with a stale cache.
    /// The caller guarantees every key is in bounds and no value is zero.
    fn from_values(shape: (usize, usize), values: HashMap<(usize, usize), T>) -> SparseMatrix<T> {
        Self {
            shape,
            values,
            compressed_updated: false,
            compressed_rowarray: vec![],
            compressed_colarray: vec![],
            compressed_dataarray: vec![],
        }
    }

    /// Builds a matrix from `(row, col, value)` triplets using **COO (coordinate list)
    /// semantics**: triplets that share a `(row, col)` are *summed*, not overwritten, and
    /// entries that sum to exactly zero are not stored. Compare `insert_triplets`, which
    /// overwrites duplicates (last write wins).
    ///
    /// Returns `SparseError::OutOfBounds` if any triplet lies outside `rows x cols`.
    #[allow(dead_code)]
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        // COO semantics: duplicate (row, col) entries are summed
        triplets: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> Result<SparseMatrix<T>, SparseError> {
        let triplets = triplets.into_iter();
        // Sized by the number of triplets, not rows * cols, which can be enormous
        let mut values = HashMap::with_capacity(triplets.size_hint().0);

        for (row, col, val) in triplets {
            if row >= rows || col >= cols {
                return Err(SparseError::OutOfBounds {
                    row,
                    col,
                    shape: (rows, cols),
                });
            }
            let entry = values.entry((row, col)).or_insert(T::zero());
            *entry = *entry + val;
        }

        let mut local = Self::from_values((rows, cols), values);
        // Zero inputs and exact cancellations leave stored zeros; drop them
        local.prune_zeroes();
        Ok(local)
    }

    /// Builds a matrix from dense rows, e.g. `Vec<Vec<T>>` or `[[T; 3]; 2]`. Zeros are
    /// not stored. The shape is inferred: the number of rows, by the length of the first row.
    /// Empty input gives a 0 x 0 matrix.
    ///
    /// Returns `SparseError::RaggedRows` if any row's length differs from the first row's.
    /// Accepts `row_iter()` output, so `from_dense(m.row_iter())` rebuilds `m` (except a
    /// `0 x m` matrix, which has no rows to carry its column count and rebuilds as `0 x 0`).
    #[allow(dead_code)]
    pub fn from_dense<R: AsRef<[T]>>(
        rows: impl IntoIterator<Item = R>,
    ) -> Result<SparseMatrix<T>, SparseError> {
        let mut values = HashMap::new();
        let mut ncols: Option<usize> = None;
        let mut nrows: usize = 0;

        for (row_idx, row) in rows.into_iter().enumerate() {
            let row = row.as_ref();
            let expected = *ncols.get_or_insert(row.len());
            if row.len() != expected {
                return Err(SparseError::RaggedRows {
                    row: row_idx,
                    expected,
                    found: row.len(),
                });
            }
            for (col_idx, &val) in row.iter().enumerate() {
                // Zeros (including -0.0 for floats) are never stored
                if val != T::zero() {
                    values.insert((row_idx, col_idx), val);
                }
            }
            nrows += 1;
        }

        Ok(Self::from_values((nrows, ncols.unwrap_or(0)), values))
    }

    /// Returns the matrix as dense rows. Allocates `rows * cols` elements, so this is only
    /// suitable for shapes that fit in memory. The inverse of `from_dense`, except that a
    /// `0 x m` matrix has no rows to carry its column count, so it round-trips as `0 x 0`.
    #[allow(dead_code)]
    pub fn to_dense(&self) -> Vec<Vec<T>> {
        let mut dense = vec![vec![T::zero(); self.shape.1]; self.shape.0];
        for ((row, col), val) in self.values.iter() {
            dense[*row][*col] = *val;
        }
        dense
    }

    /// Creates the `n x n` identity matrix.
    #[allow(dead_code)]
    pub fn identity(n: usize) -> SparseMatrix<T> {
        let mut local = Self::empty_with_shape(n, n);
        local.values.reserve(n);
        for diag_idx in 0..n {
            // Unwrap safe by construction of diag_idx
            local.insert(diag_idx, diag_idx, T::one()).unwrap();
        }
        local
    }

    /// Returns the transpose as a new matrix, leaving `self` unchanged. See
    /// `transpose_inplace` to transpose without allocating a second matrix.
    #[allow(dead_code)]
    pub fn create_transpose(&self) -> SparseMatrix<T> {
        let mut local = Self::empty_with_shape(self.shape.1, self.shape.0);
        local.values.reserve(self.values.len());
        for ((row, col), val) in self.values.iter() {
            // Unwrap safe by construction of col, row
            local.insert(*col, *row, *val).unwrap(); // Deref okay due to elementary r, c, v types
        }
        local
    }

    /// Sets the entry at `(row, col)` to `value`, replacing any existing value.
    ///
    /// Inserting a zero removes the entry instead, since zeros are never stored.
    /// Returns `SparseError::OutOfBounds` if the position is outside the matrix.
    #[allow(dead_code)]
    pub fn insert(&mut self, row: usize, col: usize, value: T) -> Result<(), SparseError> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return Err(SparseError::OutOfBounds {
                row,
                col,
                shape: self.shape,
            });
        }

        // If new value is exactly zero, clear any existing entry
        if value == T::zero() {
            // UNWRAP: already validated row, col above
            self.clear_at(row, col).unwrap();
        } else {
            self.values.insert((row, col), value);
        }
        self.compressed_updated = false;
        Ok(())
    }

    /// Inserts several `(row, col, value)` triplets, as if by repeated `insert`.
    ///
    /// Duplicate `(row, col)` triplets overwrite (last write wins). See `from_triplets`
    /// for COO semantics, where duplicates are summed.
    ///
    /// Every triplet is bounds-checked before any is applied, so on
    /// `SparseError::OutOfBounds` the matrix is left unchanged.
    #[allow(dead_code)]
    pub fn insert_triplets(&mut self, triplets: Vec<(usize, usize, T)>) -> Result<(), SparseError> {
        // Iterate through triplets first to validate no oob errors before changing
        //   underlying matrix state
        for (row, col, _) in triplets.iter() {
            if *row >= self.shape.0 || *col >= self.shape.1 {
                return Err(SparseError::OutOfBounds {
                    row: *row,
                    col: *col,
                    shape: self.shape,
                });
            }
        }

        for (row, col, val) in triplets.iter() {
            // If new value is exactly zero, clear any existing entry
            if *val == T::zero() {
                // UNWRAP: Already validated row, col above
                self.clear_at(*row, *col).unwrap();
            } else {
                self.values.insert((*row, *col), *val);
            }
        }
        self.compressed_updated = false;
        Ok(())
    }

    /// Removes the entry at `(row, col)` and returns it, or `None` if nothing was stored
    /// there.
    ///
    /// Returns `SparseError::OutOfBounds` if the position is outside the matrix.
    #[allow(dead_code)]
    pub fn clear_at(&mut self, row: usize, col: usize) -> Result<Option<T>, SparseError> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return Err(SparseError::OutOfBounds {
                row,
                col,
                shape: self.shape,
            });
        }

        self.compressed_updated = false;
        Ok(self.values.remove(&(row, col)))
    }

    /// Returns the entry at `(row, col)`, or `None` if nothing is stored there (an implicit
    /// zero).
    ///
    /// Returns `SparseError::OutOfBounds` if the position is outside the matrix.
    #[allow(dead_code)]
    pub fn peek_at(&self, row: usize, col: usize) -> Result<Option<T>, SparseError> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return Err(SparseError::OutOfBounds {
                row,
                col,
                shape: self.shape,
            });
        }

        Ok(self.values.get(&(row, col)).copied())
    }

    /// Returns the number of stored entries.
    #[allow(dead_code)]
    pub fn num_nonzero(&self) -> usize {
        self.values.len()
    }

    /// Transposes the matrix in place, swapping `shape` and every entry's row and column.
    /// See `create_transpose` for a version that returns a new matrix.
    #[allow(dead_code)]
    pub fn transpose_inplace(&mut self) {
        // Naive impl, could do better
        self.shape = (self.shape.1, self.shape.0);

        let triplets: Vec<((usize, usize), T)> = self.values.drain().collect();

        for ((row, col), val) in triplets {
            self.values.insert((col, row), val);
        }
        self.compressed_updated = false;
    }

    /// Removes any stored entries equal to zero.
    ///
    /// Every public method already avoids storing zeros, so this is a safety net that
    /// leaves the cache untouched when there is nothing to remove.
    #[allow(dead_code)]
    pub fn prune_zeroes(&mut self) {
        let init_len = self.values.len();
        self.values.retain(|_, v| *v != T::zero());
        let final_len = self.values.len();
        if init_len != final_len {
            self.compressed_updated = false;
        }
    }
}

use std::fmt;
/// Formats the matrix as a bracketed grid of dense rows, one per line.
impl<T: Scalar + fmt::Display> fmt::Display for SparseMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let inner_line_width = 8 * self.shape.1; // 6 chars per col + comma + space + leading space
        write!(f, "\t/")?;
        for _ in 0..inner_line_width {
            write!(f, " ")?;
        }
        writeln!(f, "\\")?;
        // TODO: Account for variable number of digits in cols
        for row in self.row_iter() {
            write!(f, "\t| ")?;
            for (idx, elem) in row.iter().enumerate() {
                if idx != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:>6.2}", elem)?; // TODO: dynamic precision based on longest values
            }
            writeln!(f, " |")?;
        }
        write!(f, "\t\\")?;
        for _ in 0..inner_line_width {
            write!(f, " ")?;
        }
        write!(f, "/")?;
        writeln!(f)
    }
}

use std::ops::Add;

/// Element-wise sum of two matrices, returned as a new matrix.
///
/// Entries that cancel to exactly zero are not stored in the result.
///
/// # Panics
///
/// Panics if the two matrices have different shapes.
impl<T: Scalar> Add for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn add(self, other: &SparseMatrix<T>) -> SparseMatrix<T> {
        assert!(self.shape == other.shape);
        let mut local = self.clone();

        for ((rother, cother), elemother) in other.values.iter() {
            // RESULT UNWRAP: shape validated above, so rother, cother in bounds
            let existingval = local
                .peek_at(*rother, *cother)
                .unwrap()
                .unwrap_or(T::zero());
            // See if new value for r, c would be exactly zero
            let newval = existingval + *elemother;
            // TODO: zero check here is redundant with zero check in local.insert(). Evaluate removal
            if newval == T::zero() {
                // Don't insert, and also clear whatever value already exists at local
                // UNWRAP: validated shapes above, rother cother safe by construction
                local.clear_at(*rother, *cother).unwrap();
            } else {
                // Unwrap safe by construction of rother, cother
                local.insert(*rother, *cother, newval).unwrap();
            }
        }
        local
    }
}
