//! Read access to a matrix's stored entries, rows and columns.

use std::borrow::Cow;

use super::{Compressed, SparseMatrix};
use crate::Scalar;

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
