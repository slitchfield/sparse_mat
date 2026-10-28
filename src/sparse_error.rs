use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum SparseError {
    OutOfBounds {
        row: u64,
        col: u64,
        shape: (u64, u64),
    },
    RaggedRows {
        row: u64,
        expected: u64,
        found: u64,
    },
}

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

impl Error for SparseError {}
