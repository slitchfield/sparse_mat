//! Arithmetic on `SparseMatrix`: checked methods returning `Result`, and the operator impls
//! that panic on incompatible shapes.

use std::collections::hash_map::Entry;
use std::ops::Add;

use super::SparseMatrix;
use crate::sparse_error::SparseError;
use crate::Scalar;

impl<T: Scalar> SparseMatrix<T> {
    /// Element-wise sum of two matrices, returned as a new matrix.
    ///
    /// Entries that cancel to exactly zero are not stored in the result. Returns
    /// `SparseError::ShapeMismatch` if the shapes differ; `+` panics in that case instead.
    pub fn try_add(&self, other: &SparseMatrix<T>) -> Result<SparseMatrix<T>, SparseError> {
        if self.shape != other.shape {
            return Err(SparseError::ShapeMismatch {
                left: self.shape,
                right: other.shape,
            });
        }

        let mut values = self.values.clone();
        for (&key, &val) in other.values.iter() {
            match values.entry(key) {
                Entry::Occupied(mut existing) => {
                    let sum = *existing.get() + val;
                    if sum == T::zero() {
                        existing.remove();
                    } else {
                        *existing.get_mut() = sum;
                    }
                }
                // Stored values are never zero, so nothing to check here
                Entry::Vacant(slot) => {
                    slot.insert(val);
                }
            }
        }
        Ok(SparseMatrix::from_values(self.shape, values))
    }
}

/// Element-wise sum of two matrices, returned as a new matrix.
///
/// Entries that cancel to exactly zero are not stored in the result.
///
/// # Panics
///
/// Panics if the two matrices have different shapes. Use `try_add` to get an error instead.
impl<T: Scalar> Add for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn add(self, other: &SparseMatrix<T>) -> SparseMatrix<T> {
        self.try_add(other).unwrap_or_else(|err| panic!("{err}"))
    }
}
