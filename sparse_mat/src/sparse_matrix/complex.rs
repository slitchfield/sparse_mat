//! Complex-specific operations: conjugates, the adjoint, and splitting a matrix into or
//! building it from real and imaginary parts.
//!
//! Everything except `from_parts` is generic over `num_complex::ComplexFloat`, which covers
//! `f32` and `f64` as well as their complex types, so the same code path works for real
//! matrices: `conj` is then the identity, `adjoint` the transpose, and `imag` an empty matrix.

use num_complex::{Complex, ComplexFloat};

use super::SparseMatrix;
use crate::sparse_error::SparseError;
use crate::Scalar;

impl<T: Scalar + ComplexFloat> SparseMatrix<T> {
    /// The element-wise complex conjugate, as a new matrix. For a real element type this is a
    /// copy.
    pub fn conj(&self) -> SparseMatrix<T> {
        self.map(|val| val.conj())
    }

    /// The conjugate transpose `Aᴴ`, as a new matrix. For a real element type this is the
    /// transpose.
    pub fn adjoint(&self) -> SparseMatrix<T> {
        let values = self
            .values
            .iter()
            .map(|(&(row, col), &val)| ((col, row), val.conj()))
            .collect();
        SparseMatrix::from_values((self.shape.1, self.shape.0), values)
    }

    /// The real part of each entry, as a matrix of the underlying real type (`f64` for
    /// `Complex<f64>`). Entries that are purely imaginary are not stored.
    pub fn real(&self) -> SparseMatrix<T::Real> {
        self.map(|val| val.re())
    }

    /// The imaginary part of each entry, as a matrix of the underlying real type. Entries that
    /// are purely real are not stored, so for a real element type the result is empty.
    pub fn imag(&self) -> SparseMatrix<T::Real> {
        self.map(|val| val.im())
    }
}

impl<R: Scalar> SparseMatrix<Complex<R>> {
    /// Builds a complex matrix from its real and imaginary parts: entry `(i, j)` is
    /// `re[(i, j)] + i·im[(i, j)]`. The inverse of `real` and `imag`, so
    /// `from_parts(&a.real(), &a.imag())` gives back `a`.
    ///
    /// Returns `SparseError::ShapeMismatch` unless the two parts have the same shape.
    pub fn from_parts(
        re: &SparseMatrix<R>,
        im: &SparseMatrix<R>,
    ) -> Result<SparseMatrix<Complex<R>>, SparseError> {
        if re.shape != im.shape {
            return Err(SparseError::ShapeMismatch {
                left: re.shape,
                right: im.shape,
            });
        }

        let mut values: std::collections::HashMap<(usize, usize), Complex<R>> = re
            .values
            .iter()
            .map(|(&key, &re_val)| (key, Complex::new(re_val, R::zero())))
            .collect();
        for (&key, &im_val) in im.values.iter() {
            values
                .entry(key)
                .and_modify(|val| val.im = im_val)
                .or_insert(Complex::new(R::zero(), im_val));
        }
        // Both parts store only non-zeros, so every combined entry is non-zero
        Ok(SparseMatrix::from_values(re.shape, values))
    }
}
