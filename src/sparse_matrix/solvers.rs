//! Solvers for linear systems `A x = b`: triangular solves, LU with partial pivoting,
//! Cholesky for symmetric positive definite matrices, and conjugate gradient.
//!
//! All of them need a float element type, since they divide.

use std::collections::{BTreeMap, HashMap};

use num_traits::Float;

use super::SparseMatrix;
use crate::sparse_error::SparseError;
use crate::Scalar;

/// An LU factorization with partial pivoting, `P A = L U`, from `SparseMatrix::lu`.
///
/// `L` is unit lower triangular and `U` upper triangular. Keep it around to solve many
/// right-hand sides against the same matrix.
#[derive(Clone, Debug)]
pub struct Lu<T: Scalar = f64> {
    l: SparseMatrix<T>,
    u: SparseMatrix<T>,
    perm: Vec<usize>,
    swaps: usize,
}

impl<T: Scalar + Float> Lu<T> {
    /// The unit lower triangular factor `L`, including its unit diagonal.
    pub fn l(&self) -> &SparseMatrix<T> {
        &self.l
    }

    /// The upper triangular factor `U`.
    pub fn u(&self) -> &SparseMatrix<T> {
        &self.u
    }

    /// The row permutation: row `i` of `P A` is row `permutation()[i]` of `A`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// The determinant of the original matrix: the product of `U`'s diagonal, negated once
    /// per row swap.
    pub fn determinant(&self) -> T {
        let sign = if self.swaps.is_multiple_of(2) {
            T::one()
        } else {
            -T::one()
        };
        (0..self.perm.len()).fold(sign, |det, i| {
            det * self.u.values.get(&(i, i)).copied().unwrap_or(T::zero())
        })
    }

    /// Solves `A x = b` for `x`. Returns `SparseError::ShapeMismatch` unless `b` has one
    /// entry per row of `A`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SparseError> {
        if b.len() != self.perm.len() {
            return Err(SparseError::ShapeMismatch {
                left: self.l.shape,
                right: (b.len(), 1),
            });
        }
        let permuted: Vec<T> = self.perm.iter().map(|&row| b[row]).collect();
        let y = self.l.solve_lower_triangular(&permuted)?;
        self.u.solve_upper_triangular(&y)
    }
}

impl<T: Scalar + Float> SparseMatrix<T> {
    /// Checks that `self` is square and `b_len` matches, returning the size.
    fn check_system(&self, b_len: usize) -> Result<usize, SparseError> {
        if !self.is_square() {
            return Err(SparseError::NotSquare { shape: self.shape });
        }
        if b_len != self.shape.0 {
            return Err(SparseError::ShapeMismatch {
                left: self.shape,
                right: (b_len, 1),
            });
        }
        Ok(self.shape.0)
    }

    /// Solves `L x = b` by forward substitution, reading only the lower triangle of `self`
    /// (entries above the diagonal are ignored).
    ///
    /// Returns `SparseError::SingularMatrix` if a diagonal entry is zero or not stored.
    pub fn solve_lower_triangular(&self, b: &[T]) -> Result<Vec<T>, SparseError> {
        let n = self.check_system(b.len())?;
        let csr = self.csr_cow();
        let mut x = vec![T::zero(); n];
        for i in 0..n {
            let mut remainder = b[i];
            let mut diagonal = T::zero();
            for pos in csr.ptr[i]..csr.ptr[i + 1] {
                let col = csr.idx[pos];
                if col < i {
                    remainder = remainder - csr.data[pos] * x[col];
                } else {
                    if col == i {
                        diagonal = csr.data[pos];
                    }
                    break;
                }
            }
            if diagonal == T::zero() {
                return Err(SparseError::SingularMatrix { pivot: i });
            }
            x[i] = remainder / diagonal;
        }
        Ok(x)
    }

    /// Solves `U x = b` by back substitution, reading only the upper triangle of `self`
    /// (entries below the diagonal are ignored).
    ///
    /// Returns `SparseError::SingularMatrix` if a diagonal entry is zero or not stored.
    pub fn solve_upper_triangular(&self, b: &[T]) -> Result<Vec<T>, SparseError> {
        let n = self.check_system(b.len())?;
        let csr = self.csr_cow();
        let mut x = vec![T::zero(); n];
        for i in (0..n).rev() {
            let mut remainder = b[i];
            let mut diagonal = T::zero();
            for pos in csr.ptr[i]..csr.ptr[i + 1] {
                let col = csr.idx[pos];
                if col == i {
                    diagonal = csr.data[pos];
                } else if col > i {
                    remainder = remainder - csr.data[pos] * x[col];
                }
            }
            if diagonal == T::zero() {
                return Err(SparseError::SingularMatrix { pivot: i });
            }
            x[i] = remainder / diagonal;
        }
        Ok(x)
    }

    /// Factors a square matrix as `P A = L U` with partial (row) pivoting.
    ///
    /// Rows are kept as ordered sparse maps, so only the entries that fill in are ever
    /// touched; the pivot search still looks at every remaining row for each column, so
    /// the cost is at least quadratic in the dimension.
    ///
    /// Returns `SparseError::NotSquare`, or `SparseError::SingularMatrix` if a column has no
    /// usable pivot.
    pub fn lu(&self) -> Result<Lu<T>, SparseError> {
        if !self.is_square() {
            return Err(SparseError::NotSquare { shape: self.shape });
        }
        let n = self.shape.0;
        let csr = self.csr_cow();
        let mut rows: Vec<BTreeMap<usize, T>> = csr
            .ptr
            .windows(2)
            .map(|bounds| {
                let range = bounds[0]..bounds[1];
                csr.idx[range.clone()]
                    .iter()
                    .copied()
                    .zip(csr.data[range].iter().copied())
                    .collect()
            })
            .collect();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut swaps = 0;

        for k in 0..n {
            let magnitude = |row: &BTreeMap<usize, T>| row.get(&k).map_or(T::zero(), |v| v.abs());
            let mut best = k;
            let mut best_magnitude = magnitude(&rows[k]);
            for (i, row) in rows.iter().enumerate().skip(k + 1) {
                if magnitude(row) > best_magnitude {
                    best = i;
                    best_magnitude = magnitude(row);
                }
            }
            if best_magnitude == T::zero() {
                return Err(SparseError::SingularMatrix { pivot: k });
            }
            if best != k {
                rows.swap(k, best);
                perm.swap(k, best);
                swaps += 1;
            }

            let pivot = rows[k][&k];
            let pivot_row: Vec<(usize, T)> =
                rows[k].range(k + 1..).map(|(&j, &v)| (j, v)).collect();
            for row in rows.iter_mut().skip(k + 1) {
                let Some(&below) = row.get(&k) else {
                    continue;
                };
                let multiplier = below / pivot;
                if multiplier == T::zero() {
                    row.remove(&k);
                    continue;
                }
                row.insert(k, multiplier);
                for &(j, u_kj) in &pivot_row {
                    let current = row.get(&j).copied().unwrap_or(T::zero());
                    let updated = current - multiplier * u_kj;
                    if updated == T::zero() {
                        row.remove(&j);
                    } else {
                        row.insert(j, updated);
                    }
                }
            }
        }

        let mut l_values = HashMap::new();
        let mut u_values = HashMap::new();
        for (i, row) in rows.iter().enumerate() {
            l_values.insert((i, i), T::one());
            for (&j, &val) in row {
                if j < i {
                    l_values.insert((i, j), val);
                } else {
                    u_values.insert((i, j), val);
                }
            }
        }
        Ok(Lu {
            l: SparseMatrix::from_values((n, n), l_values),
            u: SparseMatrix::from_values((n, n), u_values),
            perm,
            swaps,
        })
    }

    /// Solves `A x = b` for a general square matrix by LU factorization. Factor once with
    /// `lu` instead when solving several right-hand sides.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SparseError> {
        self.check_system(b.len())?;
        self.lu()?.solve(b)
    }

    /// Computes the Cholesky factor `L` (lower triangular) of a symmetric positive definite
    /// matrix, so that `A = L Lᵀ`.
    ///
    /// Works row by row, solving a sparse triangular system for each, so cost follows the
    /// fill-in of `L`. Returns `SparseError::NotSquare`, `SparseError::NotSymmetric`, or
    /// `SparseError::NotPositiveDefinite` if a pivot is not strictly positive.
    pub fn cholesky(&self) -> Result<SparseMatrix<T>, SparseError> {
        if !self.is_square() {
            return Err(SparseError::NotSquare { shape: self.shape });
        }
        if !self.is_symmetric() {
            return Err(SparseError::NotSymmetric);
        }
        let n = self.shape.0;
        let csr = self.csr_cow();
        let mut diagonal = vec![T::zero(); n];
        // Entries of L strictly below the diagonal, grouped by column: (row, value)
        let mut columns: Vec<Vec<(usize, T)>> = vec![vec![]; n];
        let mut l_values = HashMap::new();

        for i in 0..n {
            // Row i of A (lower part), which is turned into row i of L one column at a time,
            //   smallest column first, since each entry depends on the smaller ones.
            let mut work: BTreeMap<usize, T> = BTreeMap::new();
            let mut a_ii = T::zero();
            for pos in csr.ptr[i]..csr.ptr[i + 1] {
                let col = csr.idx[pos];
                if col < i {
                    work.insert(col, csr.data[pos]);
                } else {
                    if col == i {
                        a_ii = csr.data[pos];
                    }
                    break;
                }
            }

            let mut remaining = a_ii;
            while let Some((j, w_j)) = work.pop_first() {
                let l_ij = w_j / diagonal[j];
                if l_ij == T::zero() {
                    continue;
                }
                for &(m, l_mj) in &columns[j] {
                    let current = work.get(&m).copied().unwrap_or(T::zero());
                    work.insert(m, current - l_mj * l_ij);
                }
                columns[j].push((i, l_ij));
                l_values.insert((i, j), l_ij);
                remaining = remaining - l_ij * l_ij;
            }

            if remaining.is_nan() || remaining <= T::zero() {
                return Err(SparseError::NotPositiveDefinite { pivot: i });
            }
            diagonal[i] = remaining.sqrt();
            l_values.insert((i, i), diagonal[i]);
        }
        Ok(SparseMatrix::from_values((n, n), l_values))
    }

    /// Solves `A x = b` for a symmetric positive definite matrix by Cholesky factorization.
    /// Fails with the errors of `cholesky`.
    pub fn solve_spd(&self, b: &[T]) -> Result<Vec<T>, SparseError> {
        self.check_system(b.len())?;
        let l = self.cholesky()?;
        let y = l.solve_lower_triangular(b)?;
        l.create_transpose().solve_upper_triangular(&y)
    }

    /// Solves `A x = b` for a symmetric positive definite matrix by the conjugate gradient
    /// method, starting from zero.
    ///
    /// Stops when the residual norm is at most `tolerance * ||b||`. Each iteration costs one
    /// matrix-vector product, so this suits large systems where factoring would fill in.
    /// Symmetry is assumed, not checked. Returns `SparseError::NoConvergence` if the
    /// tolerance is not reached within `max_iterations`, or if the search direction shows the
    /// matrix is not positive definite.
    pub fn conjugate_gradient(
        &self,
        b: &[T],
        tolerance: T,
        max_iterations: usize,
    ) -> Result<Vec<T>, SparseError> {
        let n = self.check_system(b.len())?;
        let csr = self.csr_cow();
        let dot = |u: &[T], v: &[T]| u.iter().zip(v).fold(T::zero(), |sum, (&a, &b)| sum + a * b);

        let b_norm = dot(b, b).sqrt();
        let mut x = vec![T::zero(); n];
        if b_norm == T::zero() {
            return Ok(x);
        }
        let mut residual = b.to_vec();
        let mut direction = residual.clone();
        let mut residual_sq = dot(&residual, &residual);

        for iteration in 0..max_iterations {
            if residual_sq.sqrt() <= tolerance * b_norm {
                return Ok(x);
            }
            let a_direction = Self::mul_vec_csr(&csr, &direction);
            let curvature = dot(&direction, &a_direction);
            if curvature.is_nan() || curvature <= T::zero() {
                return Err(SparseError::NoConvergence {
                    iterations: iteration,
                });
            }
            let step = residual_sq / curvature;
            for i in 0..n {
                x[i] = x[i] + step * direction[i];
                residual[i] = residual[i] - step * a_direction[i];
            }
            let next_residual_sq = dot(&residual, &residual);
            let beta = next_residual_sq / residual_sq;
            for i in 0..n {
                direction[i] = residual[i] + beta * direction[i];
            }
            residual_sq = next_residual_sq;
        }

        if residual_sq.sqrt() <= tolerance * b_norm {
            Ok(x)
        } else {
            Err(SparseError::NoConvergence {
                iterations: max_iterations,
            })
        }
    }
}
