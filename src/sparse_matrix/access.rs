//! Read access to a matrix's stored entries, rows and columns.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::{Index, Range};

use super::{Compressed, SparseMatrix};
use crate::sparse_error::SparseError;
use crate::Scalar;

/// One row or column of a matrix, holding only its stored entries.
///
/// Returned by `SparseMatrix::row`, `SparseMatrix::col` and `SparseMatrix::diagonal`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseVector<T = f64> {
    len: usize,
    indices: Vec<usize>,
    values: Vec<T>,
}

impl<T: Scalar> SparseVector<T> {
    /// The length of the vector, counting implicit zeros.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector has length zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of stored (non-zero) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// The positions of the stored entries, in increasing order.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// The stored values, parallel to `indices`.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// The element at `index`, or `None` if it is an implicit zero or out of range.
    pub fn get(&self, index: usize) -> Option<T> {
        self.indices
            .binary_search(&index)
            .ok()
            .map(|pos| self.values[pos])
    }

    /// Iterates over the stored entries as `(index, value)`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, T)> + '_ {
        self.indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }

    /// Expands to a dense vector of length `len`.
    pub fn to_dense(&self) -> Vec<T> {
        let mut dense = vec![T::zero(); self.len];
        for (index, value) in self.iter() {
            dense[index] = value;
        }
        dense
    }
}

/// Iterator over a matrix's columns, created by `SparseMatrix::col_iter`.
///
/// Each item is one dense column of length `shape.0`.
pub struct ColIterator<'a, T: Scalar = f64> {
    shape: (usize, usize),
    /// Borrowed from the matrix's CSC cache when fresh, otherwise built for this iterator.
    csc: Cow<'a, Compressed<T>>,
    col_iter_idx: usize,
}

impl<T: Scalar> Iterator for ColIterator<'_, T> {
    /// One dense column.
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.col_iter_idx >= self.shape.1 {
            return None;
        }
        let start = self.csc.ptr[self.col_iter_idx];
        let end = self.csc.ptr[self.col_iter_idx + 1];

        let mut column = vec![T::zero(); self.shape.0];
        for (row, val) in std::iter::zip(&self.csc.idx[start..end], &self.csc.data[start..end]) {
            column[*row] = *val;
        }

        self.col_iter_idx += 1;
        Some(column)
    }
}

/// Iterator over a matrix's stored entries as `(row, col, value)`, created by
/// `SparseMatrix::iter`.
///
/// Entries come in row-major order: by row, then by column within each row.
pub struct Entries<'a, T: Scalar = f64> {
    /// Borrowed from the matrix's cache when fresh, otherwise built for this iterator.
    csr: Cow<'a, Compressed<T>>,
    row: usize,
    pos: usize,
}

impl<T: Scalar> Iterator for Entries<'_, T> {
    type Item = (usize, usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.csr.idx.len() {
            return None;
        }
        // Skip past rows with no entries left; ptr's last element equals the entry count,
        //   so this stops before running off the end.
        while self.csr.ptr[self.row + 1] <= self.pos {
            self.row += 1;
        }
        let item = (self.row, self.csr.idx[self.pos], self.csr.data[self.pos]);
        self.pos += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.csr.idx.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<T: Scalar> ExactSizeIterator for Entries<'_, T> {}

impl<T: Scalar> SparseMatrix<T> {
    /// Iterates over the stored entries as `(row, col, value)` triplets in row-major order.
    ///
    /// Unlike `row_iter`, this only visits stored entries, so it costs time proportional to
    /// `num_nonzero` rather than `rows * cols`. Call `explicitly_compress` first when
    /// iterating repeatedly; otherwise each iterator rebuilds the CSR arrays itself.
    pub fn iter(&self) -> Entries<'_, T> {
        Entries {
            csr: self.csr_cow(),
            row: 0,
            pos: 0,
        }
    }
}

impl<T: Scalar> SparseMatrix<T> {
    /// Extracts the compressed segment `major` as a sparse vector of length `len`.
    fn segment(compressed: &Compressed<T>, major: usize, len: usize) -> SparseVector<T> {
        let (start, end) = (compressed.ptr[major], compressed.ptr[major + 1]);
        SparseVector {
            len,
            indices: compressed.idx[start..end].to_vec(),
            values: compressed.data[start..end].to_vec(),
        }
    }

    /// Returns row `i` as a sparse vector of length `shape.1`.
    ///
    /// Returns `SparseError::OutOfRange` if `i` is not a row of the matrix.
    pub fn row(&self, i: usize) -> Result<SparseVector<T>, SparseError> {
        if i >= self.shape.0 {
            return Err(SparseError::OutOfRange {
                index: i,
                len: self.shape.0,
            });
        }
        Ok(Self::segment(&self.csr_cow(), i, self.shape.1))
    }

    /// Returns column `j` as a sparse vector of length `shape.0`.
    ///
    /// Reads the CSC form, so call `explicitly_compress_csc` first when taking many columns.
    /// Returns `SparseError::OutOfRange` if `j` is not a column of the matrix.
    pub fn col(&self, j: usize) -> Result<SparseVector<T>, SparseError> {
        if j >= self.shape.1 {
            return Err(SparseError::OutOfRange {
                index: j,
                len: self.shape.1,
            });
        }
        Ok(Self::segment(&self.csc_cow(), j, self.shape.0))
    }

    /// Iterates over the columns of the matrix, yielding each as a dense `Vec<T>`.
    ///
    /// The column counterpart of `row_iter`; call `explicitly_compress_csc` first when
    /// iterating repeatedly.
    pub fn col_iter(&self) -> ColIterator<'_, T> {
        ColIterator {
            shape: self.shape,
            csc: self.csc_cow(),
            col_iter_idx: 0,
        }
    }

    /// Returns the main diagonal as a sparse vector of length `min(rows, cols)`.
    pub fn diagonal(&self) -> SparseVector<T> {
        let len = self.shape.0.min(self.shape.1);
        let mut indices = vec![];
        let mut values = vec![];
        for i in 0..len {
            if let Some(&val) = self.values.get(&(i, i)) {
                indices.push(i);
                values.push(val);
            }
        }
        SparseVector {
            len,
            indices,
            values,
        }
    }

    /// Extracts the block `rows x cols` as a new matrix of shape
    /// `(rows.len(), cols.len())`, with indices shifted so the block starts at `(0, 0)`.
    ///
    /// Returns `SparseError::InvalidRange` if either range is reversed or ends past the
    /// matrix.
    pub fn submatrix(
        &self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<SparseMatrix<T>, SparseError> {
        for (range, len) in [(&rows, self.shape.0), (&cols, self.shape.1)] {
            if range.start > range.end || range.end > len {
                return Err(SparseError::InvalidRange {
                    start: range.start,
                    end: range.end,
                    len,
                });
            }
        }

        let values = self
            .values
            .iter()
            .filter(|((row, col), _)| rows.contains(row) && cols.contains(col))
            .map(|((row, col), val)| ((row - rows.start, col - cols.start), *val))
            .collect();
        Ok(SparseMatrix::from_values((rows.len(), cols.len()), values))
    }

    /// Extracts the rows and columns named by two index lists, as a new matrix of shape
    /// `(rows.len(), cols.len())`: entry `(i, j)` of the result is entry `(rows[i], cols[j])`
    /// of `self`.
    ///
    /// Unlike `submatrix`, the indices need not be contiguous or sorted, so this can pull out
    /// any subset (such as the PV and PQ buses of a power system) in any order. An index may
    /// appear more than once, which repeats that row or column, as multiplying by a selection
    /// matrix would. Makes one pass over the stored entries. Returns `SparseError::OutOfRange`
    /// for the first index that is not a row or column of the matrix.
    pub fn select(&self, rows: &[usize], cols: &[usize]) -> Result<SparseMatrix<T>, SparseError> {
        let row_positions = Self::positions_by_index(rows, self.shape.0)?;
        let col_positions = Self::positions_by_index(cols, self.shape.1)?;

        let mut values = HashMap::new();
        for (&(row, col), &val) in self.values.iter() {
            let (Some(out_rows), Some(out_cols)) =
                (row_positions.get(&row), col_positions.get(&col))
            else {
                continue;
            };
            for &out_row in out_rows {
                for &out_col in out_cols {
                    values.insert((out_row, out_col), val);
                }
            }
        }
        Ok(SparseMatrix::from_values((rows.len(), cols.len()), values))
    }

    /// Maps each index in `indices` to the output positions it appears at, checking that every
    /// index is below `len`.
    fn positions_by_index(
        indices: &[usize],
        len: usize,
    ) -> Result<HashMap<usize, Vec<usize>>, SparseError> {
        let mut positions: HashMap<usize, Vec<usize>> = HashMap::with_capacity(indices.len());
        for (position, &index) in indices.iter().enumerate() {
            if index >= len {
                return Err(SparseError::OutOfRange { index, len });
            }
            positions.entry(index).or_default().push(position);
        }
        Ok(positions)
    }
}

/// Reads the element at `(row, col)`, yielding zero for entries that are not stored.
///
/// # Panics
///
/// Panics if the position is outside the matrix. Use `peek_at` for a checked read.
impl<T: Scalar> Index<(usize, usize)> for SparseMatrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &T {
        assert!(
            row < self.shape.0 && col < self.shape.1,
            "index ({row}, {col}) out of bounds for shape {:?}",
            self.shape
        );
        self.values.get(&(row, col)).unwrap_or(&self.zero)
    }
}
