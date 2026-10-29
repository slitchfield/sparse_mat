//! Reordering rows and columns by permutations.
//!
//! Permutations use the same "gather" convention as `select` and `Lu::permutation`: row `i` of
//! the result is row `perm[i]` of the input.

use super::SparseMatrix;
use crate::sparse_error::SparseError;
use crate::Scalar;

impl<T: Scalar> SparseMatrix<T> {
    /// Reorders rows and columns: entry `(i, j)` of the result is entry
    /// `(row_perm[i], col_perm[j])` of `self`. As matrices, this is `P A Qᵀ` where `P` and `Q`
    /// are the permutation matrices whose row `i` selects `row_perm[i]` and `col_perm[i]`.
    ///
    /// The same result as `select(row_perm, col_perm)`, but both lists are checked to be true
    /// permutations, so no row or column is lost or repeated. Runs in O(nnz + rows + cols).
    /// Returns `SparseError::InvalidPermutation` if either list has the wrong length, an
    /// index out of range, or a repeated index.
    pub fn permute(
        &self,
        row_perm: &[usize],
        col_perm: &[usize],
    ) -> Result<SparseMatrix<T>, SparseError> {
        let row_destination = inverse_permutation(row_perm, self.shape.0)?;
        let col_destination = inverse_permutation(col_perm, self.shape.1)?;
        let values = self
            .values
            .iter()
            .map(|(&(row, col), &val)| ((row_destination[row], col_destination[col]), val))
            .collect();
        Ok(SparseMatrix::from_values(self.shape, values))
    }

    /// Applies the same permutation to rows and columns, `P A Pᵀ`, as fill-reducing orderings
    /// do. This keeps the diagonal on the diagonal, so it preserves symmetry, the trace and
    /// the set of diagonal values.
    ///
    /// Returns `SparseError::NotSquare` for a non-square matrix, and
    /// `SparseError::InvalidPermutation` as for `permute`.
    pub fn permute_symmetric(&self, perm: &[usize]) -> Result<SparseMatrix<T>, SparseError> {
        if !self.is_square() {
            return Err(SparseError::NotSquare { shape: self.shape });
        }
        self.permute(perm, perm)
    }
}

/// Checks that `perm` is a permutation of `0..len` and returns its inverse: for each original
/// index, the position it moves to.
fn inverse_permutation(perm: &[usize], len: usize) -> Result<Vec<usize>, SparseError> {
    let invalid = |reason| Err(SparseError::InvalidPermutation { reason });
    if perm.len() != len {
        return invalid("length must equal the dimension being permuted");
    }
    // usize::MAX marks "not yet seen": no real position can be that large, as len <= perm.len()
    let mut inverse = vec![usize::MAX; len];
    for (position, &index) in perm.iter().enumerate() {
        if index >= len {
            return invalid("index out of range");
        }
        if inverse[index] != usize::MAX {
            return invalid("index repeated");
        }
        inverse[index] = position;
    }
    Ok(inverse)
}
