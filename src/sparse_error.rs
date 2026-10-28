use std::error::Error;
use std::fmt;

/// Errors returned by `SparseMatrix` operations.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum SparseError {
    /// An index lies outside the matrix.
    OutOfBounds {
        /// The offending row index.
        row: u64,
        /// The offending column index.
        col: u64,
        /// The shape `(rows, cols)` of the matrix the index was checked against.
        shape: (u64, u64),
    },
    /// Dense input rows do not all have the same length.
    RaggedRows {
        /// The first row whose length differs from the first row's.
        row: u64,
        /// The length of the first row, which every row must match.
        expected: u64,
        /// The length of the offending row.
        found: u64,
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
        }
    }
}

/// Lets `SparseError` be used with `?` into `Box<dyn Error>` and similar.
impl Error for SparseError {}
