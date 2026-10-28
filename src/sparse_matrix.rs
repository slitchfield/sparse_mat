use std::collections::HashMap;

use crate::sparse_error::SparseError;

/* Starting with Dictionary of Keys impl. To support efficient operations,
     should eventually move to compressed sparse row/col
*/
#[derive(Clone)]
pub struct SparseMatrix {
    pub shape: (u64, u64),
    values: HashMap<(u64, u64), f64>,

    compressed_updated: bool,
    compressed_rowarray: Vec<u64>,
    compressed_colarray: Vec<u64>,
    compressed_dataarray: Vec<f64>,
}

/// Borrowed view of the matrix's cached CSR arrays.
pub struct CsrView<'a> {
    pub rows: &'a [u64],
    pub cols: &'a [u64],
    pub data: &'a [f64],
}

// Standalone CSR arrays, built without touching the matrix's cached copy
struct Csr {
    rows: Vec<u64>,
    cols: Vec<u64>,
    data: Vec<f64>,
}

pub struct RowIterator<'a> {
    matrix: &'a SparseMatrix,
    // Only set when the matrix's cached CSR arrays are stale
    owned: Option<Csr>,
    row_iter_idx: usize,
}

impl Iterator for RowIterator<'_> {
    // Iterate by rows
    type Item = Vec<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_iter_idx < self.matrix.shape.0 as usize {
            let (rows, cols, data) = match &self.owned {
                Some(csr) => (&csr.rows, &csr.cols, &csr.data),
                None => (
                    &self.matrix.compressed_rowarray,
                    &self.matrix.compressed_colarray,
                    &self.matrix.compressed_dataarray,
                ),
            };
            let start = rows[self.row_iter_idx] as usize;
            let end = rows[self.row_iter_idx + 1] as usize;

            let colslice = &cols[start..end];
            let dataslice = &data[start..end];

            let mut retvec: Vec<f64> = vec![0.0; self.matrix.shape.1 as usize];

            for (col, val) in std::iter::zip(colslice, dataslice) {
                retvec[*col as usize] = *val;
            }

            self.row_iter_idx += 1;

            Some(retvec)
        } else {
            None
        }
    }
}

impl Default for SparseMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseMatrix {
    fn build_csr(&self) -> Csr {
        // Create row vecs that we'll sort by col
        let mut row_vecs: Vec<Vec<(u64, f64)>> = vec![];
        for _ in 0..self.shape.0 {
            row_vecs.push(vec![]);
        }

        for ((row, col), val) in self.values.iter() {
            row_vecs[*row as usize].push((*col, *val));
        }
        for rowidx in 0..self.shape.0 {
            row_vecs[rowidx as usize].sort_by_key(|a| a.0);
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
            csr.rows.push(csr.data.len() as u64);
        }
        csr
    }

    fn _update_compressed(&mut self) {
        let csr = self.build_csr();
        self.compressed_rowarray = csr.rows;
        self.compressed_colarray = csr.cols;
        self.compressed_dataarray = csr.data;

        self.compressed_updated = true
    }

    pub fn explicitly_compress(&mut self) {
        self._update_compressed();
    }

    /// The cached CSR arrays, or `None` if they are stale (or were never built).
    /// Call `explicitly_compress` to refresh them.
    pub fn compressed(&self) -> Option<CsrView<'_>> {
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

    // Call `explicitly_compress` first when iterating repeatedly; otherwise each
    // iterator rebuilds the CSR arrays itself.
    #[allow(dead_code)]
    pub fn row_iter(&self) -> RowIterator<'_> {
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

    #[allow(dead_code)]
    pub fn new() -> SparseMatrix {
        SparseMatrix {
            shape: (0, 0),
            values: HashMap::new(),
            compressed_updated: false,
            compressed_rowarray: vec![],
            compressed_colarray: vec![],
            compressed_dataarray: vec![],
        }
    }

    #[allow(dead_code)]
    pub fn empty_with_shape(n: u64, m: u64) -> SparseMatrix {
        // No up-front reservation: n * m says nothing about how many entries a sparse
        //   matrix will hold, and can overflow or be enormous. The map grows as needed.
        SparseMatrix::from_values((n, m), HashMap::new())
    }

    fn from_values(shape: (u64, u64), values: HashMap<(u64, u64), f64>) -> SparseMatrix {
        SparseMatrix {
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
        rows: u64,
        cols: u64,
        // COO semantics: duplicate (row, col) entries are summed
        triplets: impl IntoIterator<Item = (u64, u64, f64)>,
    ) -> Result<SparseMatrix, SparseError> {
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
            *values.entry((row, col)).or_insert(0.0) += val;
        }

        let mut local = SparseMatrix::from_values((rows, cols), values);
        // Zero inputs and exact cancellations leave stored zeros; drop them
        local.prune_zeroes();
        Ok(local)
    }

    /// Builds a matrix from dense rows, e.g. `Vec<Vec<f64>>` or `[[f64; 3]; 2]`. Zeros are
    /// not stored. The shape is inferred: the number of rows, by the length of the first row.
    /// Empty input gives a 0 x 0 matrix.
    ///
    /// Returns `SparseError::RaggedRows` if any row's length differs from the first row's.
    /// Accepts `row_iter()` output, so `from_dense(m.row_iter())` rebuilds `m` (except a
    /// `0 x m` matrix, which has no rows to carry its column count and rebuilds as `0 x 0`).
    #[allow(dead_code)]
    pub fn from_dense<R: AsRef<[f64]>>(
        rows: impl IntoIterator<Item = R>,
    ) -> Result<SparseMatrix, SparseError> {
        let mut values = HashMap::new();
        let mut ncols: Option<usize> = None;
        let mut nrows: u64 = 0;

        for (row_idx, row) in rows.into_iter().enumerate() {
            let row = row.as_ref();
            let expected = *ncols.get_or_insert(row.len());
            if row.len() != expected {
                return Err(SparseError::RaggedRows {
                    row: row_idx as u64,
                    expected: expected as u64,
                    found: row.len() as u64,
                });
            }
            for (col_idx, &val) in row.iter().enumerate() {
                // Zeros (including -0.0) are never stored
                if val != 0.0f64 {
                    values.insert((row_idx as u64, col_idx as u64), val);
                }
            }
            nrows += 1;
        }

        Ok(SparseMatrix::from_values(
            (nrows, ncols.unwrap_or(0) as u64),
            values,
        ))
    }

    /// Returns the matrix as dense rows. Allocates `rows * cols` elements, so this is only
    /// suitable for shapes that fit in memory. The inverse of `from_dense`, except that a
    /// `0 x m` matrix has no rows to carry its column count, so it round-trips as `0 x 0`.
    #[allow(dead_code)]
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.shape.1 as usize]; self.shape.0 as usize];
        for ((row, col), val) in self.values.iter() {
            dense[*row as usize][*col as usize] = *val;
        }
        dense
    }

    #[allow(dead_code)]
    pub fn identity(n: u64) -> SparseMatrix {
        let mut local = SparseMatrix::empty_with_shape(n, n);
        local.values.reserve(n as usize);
        for diag_idx in 0..n {
            // Unwrap safe by construction of diag_idx
            local.insert(diag_idx, diag_idx, 1.0).unwrap();
        }
        local
    }

    #[allow(dead_code)]
    pub fn create_transpose(&self) -> SparseMatrix {
        let mut local = SparseMatrix::empty_with_shape(self.shape.1, self.shape.0);
        local.values.reserve(self.values.len());
        for ((row, col), val) in self.values.iter() {
            // Unwrap safe by construction of col, row
            local.insert(*col, *row, *val).unwrap(); // Deref okay due to elementary r, c, v types
        }
        local
    }

    #[allow(dead_code)]
    pub fn insert(&mut self, row: u64, col: u64, value: f64) -> Result<(), SparseError> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return Err(SparseError::OutOfBounds {
                row,
                col,
                shape: self.shape,
            });
        }

        // If new value is exactly zero, clear any existing entry
        if value == 0.0f64 {
            // UNWRAP: already validated row, col above
            self.clear_at(row, col).unwrap();
        } else {
            self.values.insert((row, col), value);
        }
        self.compressed_updated = false;
        Ok(())
    }

    /// Duplicate `(row, col)` triplets overwrite (last write wins). See `from_triplets`
    /// for COO semantics, where duplicates are summed.
    #[allow(dead_code)]
    pub fn insert_triplets(&mut self, triplets: Vec<(u64, u64, f64)>) -> Result<(), SparseError> {
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
            if *val == 0.0f64 {
                // UNWRAP: Already validated row, col above
                self.clear_at(*row, *col).unwrap();
            } else {
                self.values.insert((*row, *col), *val);
            }
        }
        self.compressed_updated = false;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn clear_at(&mut self, row: u64, col: u64) -> Result<Option<f64>, SparseError> {
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

    #[allow(dead_code)]
    pub fn peek_at(&self, row: u64, col: u64) -> Result<Option<f64>, SparseError> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return Err(SparseError::OutOfBounds {
                row,
                col,
                shape: self.shape,
            });
        }

        Ok(self.values.get(&(row, col)).copied())
    }

    #[allow(dead_code)]
    pub fn num_nonzero(&self) -> u64 {
        self.values.len() as u64
    }

    #[allow(dead_code)]
    pub fn transpose_inplace(&mut self) {
        // Naive impl, could do better
        self.shape = (self.shape.1, self.shape.0);

        let triplets: Vec<((u64, u64), f64)> = self.values.drain().collect();

        for ((row, col), val) in triplets {
            self.values.insert((col, row), val);
        }
        self.compressed_updated = false;
    }

    #[allow(dead_code)]
    pub fn prune_zeroes(&mut self) {
        let init_len = self.values.len();
        self.values.retain(|_, v| *v != 0.0f64);
        let final_len = self.values.len();
        if init_len != final_len {
            self.compressed_updated = false;
        }
    }
}

use std::fmt;
impl fmt::Display for SparseMatrix {
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

impl Add for &SparseMatrix {
    type Output = SparseMatrix;

    fn add(self, other: &SparseMatrix) -> SparseMatrix {
        assert!(self.shape == other.shape);
        let mut local = self.clone();

        for ((rother, cother), elemother) in other.values.iter() {
            // RESULT UNWRAP: shape validated above, so rother, cother in bounds
            let existingval = local.peek_at(*rother, *cother).unwrap().unwrap_or(0.0);
            // See if new value for r, c would be exactly 0.0
            let newval = existingval + *elemother;
            // TODO: zero check here is redundant with zero check in local.insert(). Evaluate removal
            if newval == 0.0f64 {
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
