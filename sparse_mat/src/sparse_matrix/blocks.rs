//! Building larger matrices from smaller ones: Kronecker product and stacking.

use std::borrow::Borrow;
use std::collections::HashMap;

use super::SparseMatrix;
use crate::sparse_error::SparseError;
use crate::Scalar;

impl<T: Scalar> SparseMatrix<T> {
    /// The Kronecker product `A ⊗ B`: an `(m·p) x (n·q)` matrix, for `A` of shape `m x n` and
    /// `B` of shape `p x q`, made of a copy of `B` scaled by each entry of `A`.
    ///
    /// Only stored entries are visited, so cost is `A.num_nonzero() * B.num_nonzero()`.
    /// Returns `SparseError::ShapeOverflow` if a dimension of the result does not fit in a
    /// `usize`.
    pub fn kron(&self, other: &SparseMatrix<T>) -> Result<SparseMatrix<T>, SparseError> {
        let (rows, cols) = (
            self.shape
                .0
                .checked_mul(other.shape.0)
                .ok_or(SparseError::ShapeOverflow)?,
            self.shape
                .1
                .checked_mul(other.shape.1)
                .ok_or(SparseError::ShapeOverflow)?,
        );

        let mut values = HashMap::with_capacity(self.values.len() * other.values.len());
        for (&(a_row, a_col), &a_val) in self.values.iter() {
            for (&(b_row, b_col), &b_val) in other.values.iter() {
                let product = a_val * b_val;
                if product != T::zero() {
                    values.insert(
                        (a_row * other.shape.0 + b_row, a_col * other.shape.1 + b_col),
                        product,
                    );
                }
            }
        }
        Ok(SparseMatrix::from_values((rows, cols), values))
    }

    /// Joins matrices side by side into one matrix with the same number of rows.
    ///
    /// Accepts anything that iterates over matrices or references to them. No blocks gives a
    /// `0 x 0` matrix. Returns `SparseError::ShapeMismatch` (naming the first block and the
    /// first one that differs) unless every block has the same number of rows, and
    /// `SparseError::ShapeOverflow` if the columns do not fit in a `usize`.
    pub fn hstack<I>(blocks: I) -> Result<SparseMatrix<T>, SparseError>
    where
        I: IntoIterator,
        I::Item: Borrow<SparseMatrix<T>>,
    {
        Self::stack(blocks, true)
    }

    /// Joins matrices top to bottom into one matrix with the same number of columns.
    ///
    /// The vertical counterpart of `hstack`, with the same errors for column counts.
    pub fn vstack<I>(blocks: I) -> Result<SparseMatrix<T>, SparseError>
    where
        I: IntoIterator,
        I::Item: Borrow<SparseMatrix<T>>,
    {
        Self::stack(blocks, false)
    }

    /// Shared implementation of `hstack` (`horizontal`) and `vstack`.
    fn stack<I>(blocks: I, horizontal: bool) -> Result<SparseMatrix<T>, SparseError>
    where
        I: IntoIterator,
        I::Item: Borrow<SparseMatrix<T>>,
    {
        let mut first: Option<(usize, usize)> = None;
        // Length along the axis being stacked, and the shared length of the other axis
        let mut stacked_len = 0usize;
        let mut values = HashMap::new();

        for block in blocks {
            let block: &SparseMatrix<T> = block.borrow();
            let (along, across) = if horizontal {
                (block.shape.1, block.shape.0)
            } else {
                (block.shape.0, block.shape.1)
            };
            match first {
                None => first = Some(block.shape),
                Some(first_shape) => {
                    let first_across = if horizontal {
                        first_shape.0
                    } else {
                        first_shape.1
                    };
                    if across != first_across {
                        return Err(SparseError::ShapeMismatch {
                            left: first_shape,
                            right: block.shape,
                        });
                    }
                }
            }

            for (&(row, col), &val) in block.values.iter() {
                let key = if horizontal {
                    (row, col + stacked_len)
                } else {
                    (row + stacked_len, col)
                };
                values.insert(key, val);
            }
            stacked_len = stacked_len
                .checked_add(along)
                .ok_or(SparseError::ShapeOverflow)?;
        }

        let shape = match first {
            None => (0, 0),
            Some(first_shape) if horizontal => (first_shape.0, stacked_len),
            Some(first_shape) => (stacked_len, first_shape.1),
        };
        Ok(SparseMatrix::from_values(shape, values))
    }
}
