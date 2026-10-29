//! Parallel products using `rayon` (feature `rayon`).
//!
//! Each result row is computed by exactly one thread with the same accumulation order as
//! the sequential version, so `par_*` results are bit-for-bit equal to their sequential
//! counterparts, not merely close.

use std::collections::HashMap;

use rayon::prelude::*;

use super::SparseMatrix;
use crate::sparse_error::SparseError;
use crate::Scalar;

impl<T: Scalar + Send + Sync> SparseMatrix<T> {
    /// Parallel version of `mul_vec`: rows of the result are computed on separate threads.
    pub fn par_mul_vec(&self, x: &[T]) -> Result<Vec<T>, SparseError> {
        if x.len() != self.shape.1 {
            return Err(SparseError::ShapeMismatch {
                left: self.shape,
                right: (x.len(), 1),
            });
        }

        let csr = self.csr_cow();
        let csr = &*csr;
        Ok(csr
            .ptr
            .par_windows(2)
            .map(|bounds| {
                let range = bounds[0]..bounds[1];
                csr.idx[range.clone()]
                    .iter()
                    .zip(&csr.data[range])
                    .fold(T::zero(), |sum, (&col, &val)| sum + val * x[col])
            })
            .collect())
    }

    /// Parallel version of `vec_mul`: entries of the result are computed on separate
    /// threads from the CSC form.
    pub fn par_vec_mul(&self, x: &[T]) -> Result<Vec<T>, SparseError> {
        if x.len() != self.shape.0 {
            return Err(SparseError::ShapeMismatch {
                left: (1, x.len()),
                right: self.shape,
            });
        }

        let csc = self.csc_cow();
        let csc = &*csc;
        Ok(csc
            .ptr
            .par_windows(2)
            .map(|bounds| {
                let range = bounds[0]..bounds[1];
                csc.idx[range.clone()]
                    .iter()
                    .zip(&csc.data[range])
                    .fold(T::zero(), |sum, (&row, &val)| sum + x[row] * val)
            })
            .collect())
    }

    /// Parallel version of `matmul`: rows of the product are built on separate threads and
    /// merged at the end.
    pub fn par_matmul(&self, other: &SparseMatrix<T>) -> Result<SparseMatrix<T>, SparseError> {
        if self.shape.1 != other.shape.0 {
            return Err(SparseError::ShapeMismatch {
                left: self.shape,
                right: other.shape,
            });
        }

        let a = self.csr_cow();
        let b = other.csr_cow();
        let (a, b) = (&*a, &*b);
        let rows: Vec<Vec<(usize, T)>> = a
            .ptr
            .par_windows(2)
            .map(|a_bounds| {
                let mut row_sum: HashMap<usize, T> = HashMap::new();
                for a_pos in a_bounds[0]..a_bounds[1] {
                    let (k, a_val) = (a.idx[a_pos], a.data[a_pos]);
                    for b_pos in b.ptr[k]..b.ptr[k + 1] {
                        let entry = row_sum.entry(b.idx[b_pos]).or_insert(T::zero());
                        *entry = *entry + a_val * b.data[b_pos];
                    }
                }
                row_sum
                    .into_iter()
                    .filter(|&(_, val)| val != T::zero())
                    .collect()
            })
            .collect();

        let mut values = HashMap::new();
        for (row, entries) in rows.into_iter().enumerate() {
            for (col, val) in entries {
                values.insert((row, col), val);
            }
        }
        Ok(SparseMatrix::from_values(
            (self.shape.0, other.shape.1),
            values,
        ))
    }
}
