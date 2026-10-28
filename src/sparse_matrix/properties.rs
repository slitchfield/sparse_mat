//! Properties and reductions of a matrix: shape predicates, sums and norms.

use num_traits::Float;

use super::SparseMatrix;
use crate::sparse_error::SparseError;
use crate::Scalar;

impl<T: Scalar> SparseMatrix<T> {
    /// Whether the matrix has as many rows as columns.
    pub fn is_square(&self) -> bool {
        self.shape.0 == self.shape.1
    }

    /// The fraction of positions that hold a stored entry: `num_nonzero / (rows * cols)`.
    /// An empty matrix (zero rows or columns) has density `0.0`.
    pub fn density(&self) -> f64 {
        let area = self.shape.0 as f64 * self.shape.1 as f64;
        if area == 0.0 {
            0.0
        } else {
            self.values.len() as f64 / area
        }
    }

    /// The sum of all entries.
    ///
    /// Adds in row-major order, so the result is deterministic for floating point types.
    pub fn sum(&self) -> T {
        self.iter()
            .fold(T::zero(), |total, (_, _, val)| total + val)
    }

    /// The sum of the main diagonal. Returns `SparseError::NotSquare` for a non-square matrix.
    pub fn trace(&self) -> Result<T, SparseError> {
        if !self.is_square() {
            return Err(SparseError::NotSquare { shape: self.shape });
        }
        Ok((0..self.shape.0)
            .filter_map(|i| self.values.get(&(i, i)))
            .fold(T::zero(), |total, &val| total + val))
    }

    /// Whether the matrix equals its transpose. Non-square matrices are never symmetric.
    ///
    /// Compares with `==`, so floating point entries must match exactly.
    pub fn is_symmetric(&self) -> bool {
        self.is_square()
            && self
                .values
                .iter()
                .all(|(&(row, col), &val)| self.values.get(&(col, row)) == Some(&val))
    }
}

impl<T: Scalar + Float> SparseMatrix<T> {
    /// The Frobenius norm: the square root of the sum of squared magnitudes.
    pub fn norm_frobenius(&self) -> T {
        self.iter()
            .fold(T::zero(), |total, (_, _, val)| total + val * val)
            .sqrt()
    }

    /// The 1-norm: the largest column sum of absolute values.
    pub fn norm_1(&self) -> T {
        let csc = self.csc_cow();
        csc.ptr.windows(2).fold(T::zero(), |largest, bounds| {
            let column_sum = csc.data[bounds[0]..bounds[1]]
                .iter()
                .fold(T::zero(), |total, val| total + val.abs());
            largest.max(column_sum)
        })
    }

    /// The infinity norm: the largest row sum of absolute values.
    pub fn norm_inf(&self) -> T {
        let csr = self.csr_cow();
        csr.ptr.windows(2).fold(T::zero(), |largest, bounds| {
            let row_sum = csr.data[bounds[0]..bounds[1]]
                .iter()
                .fold(T::zero(), |total, val| total + val.abs());
            largest.max(row_sum)
        })
    }
}
