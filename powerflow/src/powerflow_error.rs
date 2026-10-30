//! The error type for reading cases and running power flows.

use std::{fmt, path::PathBuf};

/// Errors returned by `powerflow` operations.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum PowerflowError {
    /// A case file couldn't be read.
    NoCaseFile {
        /// The path that was tried.
        bad_path: PathBuf,
        /// The underlying I/O error's message.
        reason: String,
    },
    /// Case file text is malformed, or uses syntax the reader doesn't support.
    Parse {
        /// The 1-based line the problem was found on.
        line: usize,
        /// What is wrong.
        message: String,
    },
    /// The case file lacks a field every case needs (`version`, `baseMVA`, `bus`, `gen` or
    /// `branch`).
    MissingField {
        /// The field's name in the file, without `mpc.`.
        field: &'static str,
    },
    /// The case file isn't in MATPOWER's version 2 format, the only one supported.
    UnsupportedVersion {
        /// The version the file states.
        found: String,
    },
    /// A bus row has a type code other than 1 (PQ), 2 (PV), 3 (reference) or 4 (isolated).
    UnknownBusType {
        /// The 1-based line of the bus row.
        line: usize,
        /// The code found.
        code: f64,
    },
    /// The system MVA base isn't a positive, finite number.
    InvalidBaseMva {
        /// The value found.
        base_mva: f64,
    },
    /// Two buses have the same label.
    DuplicateBus {
        /// The repeated label.
        label: usize,
        /// The 1-based row in the bus table of its first use.
        first_row: usize,
        /// The 1-based row in the bus table of its second use.
        second_row: usize,
    },
    /// A generator or branch refers to a bus label that no bus has.
    UnknownBus {
        /// The table the reference is in (`gen` or `branch`).
        table: &'static str,
        /// The 1-based row in that table.
        row: usize,
        /// The label referred to.
        label: usize,
    },
    /// The case has no buses.
    NoBuses,
    /// An optional table's length doesn't match the table it describes.
    CountMismatch {
        /// The field's name in the file (`bus_name` or `gencost`).
        field: &'static str,
        /// The length or lengths allowed.
        expected: String,
        /// The length found.
        found: usize,
    },
}

impl fmt::Display for PowerflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PowerflowError::NoCaseFile { bad_path, reason } => {
                write!(f, "can't read case file {}: {reason}", bad_path.display())
            }
            PowerflowError::Parse { line, message } => write!(f, "line {line}: {message}"),
            PowerflowError::MissingField { field } => {
                write!(f, "case file has no mpc.{field}")
            }
            PowerflowError::UnsupportedVersion { found } => write!(
                f,
                "case file is version {found:?}; only MATPOWER case format version 2 is supported"
            ),
            PowerflowError::UnknownBusType { line, code } => write!(
                f,
                "line {line}: bus type {code} is not 1 (PQ), 2 (PV), 3 (reference) or 4 (isolated)"
            ),
            PowerflowError::InvalidBaseMva { base_mva } => {
                write!(f, "baseMVA must be positive and finite, not {base_mva}")
            }
            PowerflowError::DuplicateBus {
                label,
                first_row,
                second_row,
            } => write!(
                f,
                "bus {label} appears twice in the bus table, at rows {first_row} and {second_row}"
            ),
            PowerflowError::UnknownBus { table, row, label } => {
                write!(
                    f,
                    "{table} row {row} refers to bus {label}, which doesn't exist"
                )
            }
            PowerflowError::NoBuses => write!(f, "case has no buses"),
            PowerflowError::CountMismatch {
                field,
                expected,
                found,
            } => write!(f, "{field} has {found} entries, but should have {expected}"),
        }
    }
}

impl std::error::Error for PowerflowError {}
