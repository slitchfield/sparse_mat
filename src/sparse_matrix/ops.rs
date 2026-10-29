//! Arithmetic on `SparseMatrix`: checked methods returning `Result`, and the operator impls
//! that panic on incompatible shapes.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::SparseMatrix;
use crate::sparse_error::SparseError;
use crate::Scalar;

impl<T: Scalar> SparseMatrix<T> {
    /// A copy of the shape and entries without the compressed caches, which the copy is about
    /// to invalidate anyway.
    fn clone_entries(&self) -> SparseMatrix<T> {
        SparseMatrix::from_values(self.shape, self.values.clone())
    }

    /// Replaces each entry `a` of `self` that also appears in `other` (as `b`) with
    /// `combine(a, b)`, and entries only in `other` with `combine(0, b)`. Results that are
    /// exactly zero are removed. Entries only in `self` are left alone, which is correct for
    /// `+` and `-` because `combine(a, 0) == a`.
    fn merge_in_place(
        &mut self,
        other: &SparseMatrix<T>,
        combine: impl Fn(T, T) -> T,
    ) -> Result<(), SparseError> {
        if self.shape != other.shape {
            return Err(SparseError::ShapeMismatch {
                left: self.shape,
                right: other.shape,
            });
        }

        for (&key, &val) in other.values.iter() {
            let current = self.values.get(&key).copied().unwrap_or(T::zero());
            let merged = combine(current, val);
            if merged == T::zero() {
                self.values.remove(&key);
            } else {
                self.values.insert(key, merged);
            }
        }
        self.invalidate();
        Ok(())
    }

    /// Element-wise sum of two matrices, returned as a new matrix.
    ///
    /// Entries that cancel to exactly zero are not stored in the result. Returns
    /// `SparseError::ShapeMismatch` if the shapes differ; `+` panics in that case instead.
    pub fn try_add(&self, other: &SparseMatrix<T>) -> Result<SparseMatrix<T>, SparseError> {
        let mut sum = self.clone_entries();
        sum.merge_in_place(other, |a, b| a + b)?;
        Ok(sum)
    }

    /// Element-wise difference `self - other`, returned as a new matrix.
    ///
    /// Entries that cancel to exactly zero are not stored in the result. Returns
    /// `SparseError::ShapeMismatch` if the shapes differ; `-` panics in that case instead.
    /// Arithmetic follows `T`'s own rules, so for unsigned integers subtracting a larger
    /// value from a smaller one overflows.
    pub fn try_sub(&self, other: &SparseMatrix<T>) -> Result<SparseMatrix<T>, SparseError> {
        let mut difference = self.clone_entries();
        difference.merge_in_place(other, |a, b| a - b)?;
        Ok(difference)
    }

    /// Multiplies every entry by `factor`, returned as a new matrix.
    ///
    /// A zero factor (or any product that is exactly zero) leaves no stored entry.
    pub fn scale(&self, factor: T) -> SparseMatrix<T> {
        let mut scaled = self.clone_entries();
        scaled.scale_in_place(factor);
        scaled
    }

    /// Multiplies every entry by `factor` in place.
    fn scale_in_place(&mut self, factor: T) {
        self.values.retain(|_, val| {
            *val = *val * factor;
            *val != T::zero()
        });
        self.invalidate();
    }
}

/// Implements a binary operator for every combination of owned and borrowed operands. An
/// owned left operand is updated in place instead of copied.
macro_rules! impl_matrix_binary_op {
    ($trait:ident, $method:ident, $try_method:ident, $verb:literal, $combine:expr) => {
        #[doc = concat!("Element-wise ", $verb, " of two matrices, returned as a new matrix.")]
        ///
        /// Entries that cancel to exactly zero are not stored in the result.
        ///
        /// # Panics
        ///
        /// Panics if the two matrices have different shapes. Use the matching `try_` method to
        /// get an error instead.
        impl<T: Scalar> $trait<&SparseMatrix<T>> for &SparseMatrix<T> {
            type Output = SparseMatrix<T>;

            fn $method(self, other: &SparseMatrix<T>) -> SparseMatrix<T> {
                self.$try_method(other)
                    .unwrap_or_else(|err| panic!("{err}"))
            }
        }

        impl<T: Scalar> $trait<SparseMatrix<T>> for &SparseMatrix<T> {
            type Output = SparseMatrix<T>;

            fn $method(self, other: SparseMatrix<T>) -> SparseMatrix<T> {
                self.$method(&other)
            }
        }

        impl<T: Scalar> $trait<&SparseMatrix<T>> for SparseMatrix<T> {
            type Output = SparseMatrix<T>;

            fn $method(mut self, other: &SparseMatrix<T>) -> SparseMatrix<T> {
                self.merge_in_place(other, $combine)
                    .unwrap_or_else(|err| panic!("{err}"));
                self
            }
        }

        impl<T: Scalar> $trait<SparseMatrix<T>> for SparseMatrix<T> {
            type Output = SparseMatrix<T>;

            fn $method(self, other: SparseMatrix<T>) -> SparseMatrix<T> {
                self.$method(&other)
            }
        }
    };
}

impl_matrix_binary_op!(Add, add, try_add, "sum", |a, b| a + b);
impl_matrix_binary_op!(Sub, sub, try_sub, "difference", |a, b| a - b);

/// Adds `other` into `self` element-wise.
///
/// # Panics
///
/// Panics if the two matrices have different shapes.
impl<T: Scalar> AddAssign<&SparseMatrix<T>> for SparseMatrix<T> {
    fn add_assign(&mut self, other: &SparseMatrix<T>) {
        self.merge_in_place(other, |a, b| a + b)
            .unwrap_or_else(|err| panic!("{err}"));
    }
}

/// Adds `other` into `self` element-wise.
///
/// # Panics
///
/// Panics if the two matrices have different shapes.
impl<T: Scalar> AddAssign<SparseMatrix<T>> for SparseMatrix<T> {
    fn add_assign(&mut self, other: SparseMatrix<T>) {
        *self += &other;
    }
}

/// Subtracts `other` from `self` element-wise.
///
/// # Panics
///
/// Panics if the two matrices have different shapes.
impl<T: Scalar> SubAssign<&SparseMatrix<T>> for SparseMatrix<T> {
    fn sub_assign(&mut self, other: &SparseMatrix<T>) {
        self.merge_in_place(other, |a, b| a - b)
            .unwrap_or_else(|err| panic!("{err}"));
    }
}

/// Subtracts `other` from `self` element-wise.
///
/// # Panics
///
/// Panics if the two matrices have different shapes.
impl<T: Scalar> SubAssign<SparseMatrix<T>> for SparseMatrix<T> {
    fn sub_assign(&mut self, other: SparseMatrix<T>) {
        *self -= &other;
    }
}

/// Negates every entry, returned as a new matrix.
impl<T: Scalar + Neg<Output = T>> Neg for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn neg(self) -> SparseMatrix<T> {
        let values = self.values.iter().map(|(&key, &val)| (key, -val)).collect();
        SparseMatrix::from_values(self.shape, values)
    }
}

/// Negates every entry.
impl<T: Scalar + Neg<Output = T>> Neg for SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn neg(mut self) -> SparseMatrix<T> {
        for val in self.values.values_mut() {
            *val = -*val;
        }
        self.invalidate();
        self
    }
}

/// Multiplies every entry by a scalar; see `SparseMatrix::scale`.
impl<T: Scalar> Mul<T> for &SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn mul(self, factor: T) -> SparseMatrix<T> {
        self.scale(factor)
    }
}

/// Multiplies every entry by a scalar in place.
impl<T: Scalar> Mul<T> for SparseMatrix<T> {
    type Output = SparseMatrix<T>;

    fn mul(mut self, factor: T) -> SparseMatrix<T> {
        self.scale_in_place(factor);
        self
    }
}

/// Multiplies every entry by a scalar in place.
impl<T: Scalar> MulAssign<T> for SparseMatrix<T> {
    fn mul_assign(&mut self, factor: T) {
        self.scale_in_place(factor);
    }
}

/// Lets a scalar stand on the left of `*`, as in `2.0 * &matrix`. A blanket impl over `T` is
/// not allowed for a foreign type, so each primitive scalar gets its own.
macro_rules! impl_scalar_on_left {
    ($($scalar:ty),*) => {$(
        impl Mul<&SparseMatrix<$scalar>> for $scalar {
            type Output = SparseMatrix<$scalar>;

            fn mul(self, matrix: &SparseMatrix<$scalar>) -> SparseMatrix<$scalar> {
                matrix.scale(self)
            }
        }

        impl Mul<SparseMatrix<$scalar>> for $scalar {
            type Output = SparseMatrix<$scalar>;

            fn mul(self, matrix: SparseMatrix<$scalar>) -> SparseMatrix<$scalar> {
                matrix * self
            }
        }
    )*};
}

impl_scalar_on_left!(f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl<T: Scalar> SparseMatrix<T> {
    /// Element-wise (Hadamard) product of two matrices, returned as a new matrix.
    ///
    /// Only positions stored in both operands can be non-zero, so this visits the entries of
    /// the sparser operand. Returns `SparseError::ShapeMismatch` if the shapes differ.
    pub fn hadamard(&self, other: &SparseMatrix<T>) -> Result<SparseMatrix<T>, SparseError> {
        if self.shape != other.shape {
            return Err(SparseError::ShapeMismatch {
                left: self.shape,
                right: other.shape,
            });
        }

        let (smaller, larger) = if self.values.len() <= other.values.len() {
            (self, other)
        } else {
            (other, self)
        };
        let values = smaller
            .values
            .iter()
            .filter_map(|(&key, &a)| {
                let product = a * *larger.values.get(&key)?;
                (product != T::zero()).then_some((key, product))
            })
            .collect();
        Ok(SparseMatrix::from_values(self.shape, values))
    }

    /// Applies `f` to every stored entry, returned as a new matrix of the same shape. The
    /// result may have a different element type.
    ///
    /// `f` sees stored entries only, never the implicit zeros, so `map(|x| x + 1)` does not
    /// turn the zeros into ones. Results that are exactly zero are not stored.
    pub fn map<U: Scalar>(&self, f: impl Fn(T) -> U) -> SparseMatrix<U> {
        let values = self
            .values
            .iter()
            .filter_map(|(&key, &val)| {
                let mapped = f(val);
                (mapped != U::zero()).then_some((key, mapped))
            })
            .collect();
        SparseMatrix::from_values(self.shape, values)
    }

    /// Applies `f` to every stored entry in place; see `map` for what `f` is applied to.
    pub fn apply(&mut self, f: impl Fn(T) -> T) {
        self.values.retain(|_, val| {
            *val = f(*val);
            *val != T::zero()
        });
        self.invalidate();
    }
}
