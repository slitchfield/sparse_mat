use std::error::Error;
use std::fmt;

/// Errors returned by `SparseMatrix` operations.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum SparseError {
    /// An index lies outside the matrix.
    OutOfBounds {
        /// The offending row index.
        row: usize,
        /// The offending column index.
        col: usize,
        /// The shape `(rows, cols)` of the matrix the index was checked against.
        shape: (usize, usize),
    },
    /// Dense input rows do not all have the same length.
    RaggedRows {
        /// The first row whose length differs from the first row's.
        row: usize,
        /// The length of the first row, which every row must match.
        expected: usize,
        /// The length of the offending row.
        found: usize,
    },
    /// The operands' shapes are incompatible for the operation.
    ShapeMismatch {
        /// The shape `(rows, cols)` of the left operand.
        left: (usize, usize),
        /// The shape `(rows, cols)` of the right operand.
        right: (usize, usize),
    },
}

/// Formats a short, lowercase description of the error.
impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparseError::OutOfBounds { row, col, shape } => {
                write!(f, "index ({row}, {col}) out of bounds for shape {shape:?}")
            }
            SparseError::RaggedRows {
                row,
                expected,
                found,
            } => {
                write!(f, "row {row} has length {found}, expected {expected}")
            }
            SparseError::ShapeMismatch { left, right } => {
                write!(f, "incompatible shapes {left:?} and {right:?}")
            }
        }
    }
}

/// Lets `SparseError` be used with `?` into `Box<dyn Error>` and similar.
impl Error for SparseError {}
