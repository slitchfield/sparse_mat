//! Matrix products: matrix-vector, vector-matrix, matrix-dense and matrix-matrix (SpGEMM).
//!
//! Every product walks the compressed form, whose entries come in sorted order, so results
//! are deterministic even for floating point sums.

use std::collections::HashMap;
use std::ops::Mul;

use super::{Compressed, SparseMatrix};
use crate::sparse_error::SparseError;
use crate::Scalar;

impl<T: Scalar> SparseMatrix<T> {
    /// `A x` for a matrix in CSR form; the caller has checked that `x` has one entry per
    /// column. Split out so that iterative solvers can build the CSR form once.
    pub(super) fn mul_vec_csr(csr: &Compressed<T>, x: &[T]) -> Vec<T> {
        csr.ptr
            .windows(2)
            .map(|bounds| {
                let (cols, vals) = (
                    &csr.idx[bounds[0]..bounds[1]],
                    &csr.data[bounds[0]..bounds[1]],
                );
                cols.iter()
                    .zip(vals)
                    .fold(T::zero(), |sum, (&col, &val)| sum + val * x[col])
            })
            .collect()
    }

    /// Computes the matrix-vector product `A x` for a column vector `x`, giving a vector of
    /// length `shape.0`.
    ///
    /// Returns `SparseError::ShapeMismatch` (with the vector treated as `len x 1`) unless
    /// `x.len() == shape.1`. Call `explicitly_compress` first when multiplying repeatedly.
    pub fn mul_vec(&self, x: &[T]) -> Result<Vec<T>, SparseError> {
        if x.len() != self.shape.1 {
            return Err(SparseError::ShapeMismatch {
                left: self.shape,
                right: (x.len(), 1),
            });
        }

        Ok(Self::mul_vec_csr(&self.csr_cow(), x))
    }

    /// Computes the vector-matrix product `x A` for a row vector `x`, giving a vector of
    /// length `shape.1`. This is `Aᵀ x` without forming the transpose.
    ///
    /// Reads the CSC form, so call `explicitly_compress_csc` first when multiplying
    /// repeatedly. Returns `SparseError::ShapeMismatch` (with the vector treated as
    /// `1 x len`) unless `x.len() == shape.0`.
    pub fn vec_mul(&self, x: &[T]) -> Result<Vec<T>, SparseError> {
        if x.len() != self.shape.0 {
            return Err(SparseError::ShapeMismatch {
                left: (1, x.len()),
                right: self.shape,
            });
        }

        let csc = self.csc_cow();
        Ok(csc
            .ptr
            .windows(2)
            .map(|bounds| {
                let (rows, vals) = (
                    &csc.idx[bounds[0]..bounds[1]],
                    &csc.data[bounds[0]..bounds[1]],
                );
                rows.iter()
                    .zip(vals)
                    .fold(T::zero(), |sum, (&row, &val)| sum + x[row] * val)
            })
            .collect())
    }

    /// Computes the product `A B` with a dense matrix `B` given as rows, giving dense rows.
    ///
    /// Returns `SparseError::RaggedRows` if `B`'s rows differ in length, and
    /// `SparseError::ShapeMismatch` unless `B` has `shape.1` rows.
    pub fn mul_dense(&self, b: &[Vec<T>]) -> Result<Vec<Vec<T>>, SparseError> {
        let b_cols = b.first().map_or(0, Vec::len);
        if let Some((row, bad)) = b.iter().enumerate().find(|(_, row)| row.len() != b_cols) {
            return Err(SparseError::RaggedRows {
                row,
                expected: b_cols,
                found: bad.len(),
            });
        }
        if b.len() != self.shape.1 {
            return Err(SparseError::ShapeMismatch {
                left: self.shape,
                right: (b.len(), b_cols),
            });
        }

        let csr = self.csr_cow();
        Ok(csr
            .ptr
            .windows(2)
            .map(|bounds| {
                let mut out_row = vec![T::zero(); b_cols];
                for pos in bounds[0]..bounds[1] {
                    let (k, val) = (csr.idx[pos], csr.data[pos]);
                    for (out, &b_kj) in out_row.iter_mut().zip(&b[k]) {
                        *out = *out + val * b_kj;
                    }
                }
                out_row
            })
            .collect())
    }

    /// Computes the sparse matrix product `A B` (SpGEMM), using Gustavson's row-by-row
    /// algorithm: row `i` of the result is the sum of the rows of `B` selected and scaled by
    /// the entries of row `i` of `A`. Work is proportional to the number of multiplications
    /// between stored entries, not to the dimensions.
    ///
    /// Entries that cancel to exactly zero are not stored. Returns
    /// `SparseError::ShapeMismatch` unless `shape.1 == other.shape.0`; `*` panics instead.
    pub fn matmul(&self, other: &SparseMatrix<T>) -> Result<SparseMatrix<T>, SparseError> {
        if self.shape.1 != other.shape.0 {
            return Err(SparseError::ShapeMismatch {
                left: self.shape,
                right: other.shape,
            });
        }

        let a = self.csr_cow();
        let b = other.csr_cow();
        let mut values = HashMap::new();
        // Accumulator for the row being built, reused across rows
        let mut row_sum: HashMap<usize, T> = HashMap::new();

        for (row, a_bounds) in a.ptr.windows(2).enumerate() {
            if a_bounds[0] == a_bounds[1] {
                continue;
            }
            row_sum.clear();
            for a_pos in a_bounds[0]..a_bounds[1] {
                let (k, a_val) = (a.idx[a_pos], a.data[a_pos]);
                for b_pos in b.ptr[k]..b.ptr[k + 1] {
                    let entry = row_sum.entry(b.idx[b_pos]).or_insert(T::zero());
                    *entry = *entry + a_val * b.data[b_pos];
                }
            }
            for (&col, &val) in row_sum.iter() {
                if val != T::zero() {
                    values.insert((row, col), val);
                }
            }
        }
        Ok(SparseMatrix::from_values(
            (self.shape.0, other.shape.1),
            values,
        ))
    }
}

/// Matrix product; see `SparseMatrix::matmul`.
///
/// # Panics
///
/// Panics unless the left matrix has as many columns as the right has rows.
impl<T: Scalar> Mul<&SparseMatrix<T>> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn mul(self, other: &SparseMatrix<T>) -> SparseMatrix<T> {
        self.matmul(other).unwrap_or_else(|err| panic!("{err}"))
    }
}

/// Matrix product; see `SparseMatrix::matmul`.
///
/// # Panics
///
/// Panics unless the left matrix has as many columns as the right has rows.
impl<T: Scalar> Mul<SparseMatrix<T>> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn mul(self, other: SparseMatrix<T>) -> SparseMatrix<T> {
        &self * &other
    }
}

/// Matrix product; see `SparseMatrix::matmul`.
///
/// # Panics
///
/// Panics unless the left matrix has as many columns as the right has rows.
impl<T: Scalar> Mul<&SparseMatrix<T>> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn mul(self, other: &SparseMatrix<T>) -> SparseMatrix<T> {
        &self * other
    }
}

/// Matrix product; see `SparseMatrix::matmul`.
///
/// # Panics
///
/// Panics unless the left matrix has as many columns as the right has rows.
impl<T: Scalar> Mul<SparseMatrix<T>> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn mul(self, other: SparseMatrix<T>) -> SparseMatrix<T> {
        self * &other
    }
}

/// Matrix-vector product `A x`; see `SparseMatrix::mul_vec`.
///
/// # Panics
///
/// Panics unless the vector's length equals the matrix's column count.
impl<T: Scalar> Mul<&[T]> for &SparseMatrix<T> {
    type Output = Vec<T>;

    fn mul(self, x: &[T]) -> Vec<T> {
        self.mul_vec(x).unwrap_or_else(|err| panic!("{err}"))
    }
}

/// Matrix-vector product `A x`; see `SparseMatrix::mul_vec`.
///
/// # Panics
///
/// Panics unless the vector's length equals the matrix's column count.
impl<T: Scalar> Mul<&Vec<T>> for &SparseMatrix<T> {
    type Output = Vec<T>;

    fn mul(self, x: &Vec<T>) -> Vec<T> {
        self.mul_vec(x).unwrap_or_else(|err| panic!("{err}"))
    }
}

/// Vector-matrix product `x A`; see `SparseMatrix::vec_mul`.
///
/// # Panics
///
/// Panics unless the vector's length equals the matrix's row count.
impl<T: Scalar> Mul<&SparseMatrix<T>> for &[T] {
    type Output = Vec<T>;

    fn mul(self, matrix: &SparseMatrix<T>) -> Vec<T> {
        matrix.vec_mul(self).unwrap_or_else(|err| panic!("{err}"))
    }
}

/// Vector-matrix product `x A`; see `SparseMatrix::vec_mul`.
///
/// # Panics
///
/// Panics unless the vector's length equals the matrix's row count.
impl<T: Scalar> Mul<&SparseMatrix<T>> for &Vec<T> {
    type Output = Vec<T>;

    fn mul(self, matrix: &SparseMatrix<T>) -> Vec<T> {
        matrix.vec_mul(self).unwrap_or_else(|err| panic!("{err}"))
    }
}
