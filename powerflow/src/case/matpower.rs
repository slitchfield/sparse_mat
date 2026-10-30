//! Reading MATPOWER case files (version 2 format).
//!
//! A case file is MATLAB code that assigns fields of a struct `mpc`. Rather than interpret
//! MATLAB, this reads the subset that pure data files use and rejects everything else:
//!
//! - comments (`%` to the end of the line, outside strings), blank lines, a leading
//!   `function mpc = name` line, and `end`;
//! - `mpc.<field> = <literal>;` where the literal is a matrix of plain numbers (`[ … ]`, rows
//!   ended by `;` or a new line, values separated by spaces, tabs or commas, `...` continuing a
//!   row onto the next line), a cell array of strings (`{ 'a'; 'b' }`), a string (`'2'`) or
//!   a number.
//!
//! Some MATPOWER cases rescale their data with code after the literals (for example
//! `mpc.bus(:, [PD, QD]) = mpc.bus(:, [PD, QD]) / 1e3;`). Ignoring that code would silently
//! give the wrong values, so any statement outside the subset is an error, as are numeric
//! expressions such as `50/3` inside a matrix.

use std::collections::HashMap;

use super::{Branch, Bus, BusType, Case, Gen, GenCost};
use crate::powerflow_error::PowerflowError;

/// The minimum columns of each table in the version 2 format. Extra columns, such as the
/// results MATPOWER appends to solved cases, are ignored.
const BUS_COLUMNS: usize = 13;
const GEN_COLUMNS: usize = 21;
const BRANCH_COLUMNS: usize = 13;

/// A matrix row, with the line it starts on for error messages.
struct Row {
    line: usize,
    values: Vec<f64>,
}

/// The literal assigned to an `mpc` field.
enum Value {
    Matrix(Vec<Row>),
    Cell(Vec<String>),
    Str(String),
    Number(f64),
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Value::Matrix(_) => "a matrix",
            Value::Cell(_) => "a cell array",
            Value::Str(_) => "a string",
            Value::Number(_) => "a number",
        }
    }
}

/// An `mpc` field's value and the line its assignment starts on.
struct Field {
    line: usize,
    value: Value,
}

fn parse_error(line: usize, message: impl Into<String>) -> PowerflowError {
    PowerflowError::Parse {
        line,
        message: message.into(),
    }
}

/// Parses case file text into a [`Case`], without the cross-table checks of
/// [`Case::validate`].
pub(super) fn parse(text: &str) -> Result<Case, PowerflowError> {
    build_case(read_fields(text)?)
}

/// One line with its comment removed, and whether it ends in a `...` continuation.
struct CodeLine<'a> {
    number: usize,
    code: &'a str,
    continued: bool,
}

/// Removes a `%` comment and a `...` continuation from `line`, outside single-quoted
/// strings (where `''` is an escaped quote and keeps the string open).
fn strip_line(number: usize, line: &str) -> CodeLine<'_> {
    let bytes = line.as_bytes();
    let mut in_string = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' => in_string = !in_string,
            b'%' if !in_string => {
                return CodeLine {
                    number,
                    code: &line[..i],
                    continued: false,
                }
            }
            b'.' if !in_string && line[i..].starts_with("...") => {
                return CodeLine {
                    number,
                    code: &line[..i],
                    continued: true,
                }
            }
            _ => {}
        }
        i += 1;
    }
    CodeLine {
        number,
        code: line,
        continued: false,
    }
}

/// Reads every `mpc.<field> = <literal>;` assignment, rejecting any other statement.
fn read_fields(text: &str) -> Result<HashMap<String, Field>, PowerflowError> {
    let lines: Vec<CodeLine> = text
        .lines()
        .enumerate()
        .map(|(index, line)| strip_line(index + 1, line))
        .collect();

    let mut fields: HashMap<String, Field> = HashMap::new();
    let mut seen_statement = false;
    let mut index = 0;
    while index < lines.len() {
        let line = &lines[index];
        let statement = line.code.trim();
        index += 1;
        if statement.is_empty() {
            continue;
        }

        if statement.starts_with("function ") && !seen_statement {
            seen_statement = true;
            continue;
        }
        seen_statement = true;
        if matches!(
            statement.trim_end_matches(';').trim(),
            "end" | "endfunction"
        ) {
            continue;
        }

        let Some(assignment) = statement.strip_prefix("mpc.") else {
            return Err(unsupported_statement(line.number, statement));
        };
        let name_len = assignment
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(assignment.len());
        let (name, rest) = assignment.split_at(name_len);
        let Some(rhs) = rest.trim_start().strip_prefix('=') else {
            // Such as an indexed assignment, `mpc.bus(:, PD) = …`
            return Err(unsupported_statement(line.number, statement));
        };
        let rhs = rhs.trim_start();

        let (value, next_index) = read_literal(&lines, index - 1, rhs)?;
        index = next_index;
        if let Some(previous) = fields.insert(
            name.to_string(),
            Field {
                line: line.number,
                value,
            },
        ) {
            return Err(parse_error(
                line.number,
                format!(
                    "mpc.{name} is assigned twice (first at line {})",
                    previous.line
                ),
            ));
        }
    }
    Ok(fields)
}

fn unsupported_statement(line: usize, statement: &str) -> PowerflowError {
    parse_error(
        line,
        format!(
            "unsupported statement `{statement}`: only `mpc.<field> = <literal>;` assignments are \
             supported, and this case file may compute its data with code"
        ),
    )
}

/// Reads the literal starting at `rhs` on line `start`, which may continue over later lines.
/// Returns the value and the index of the first line after it.
fn read_literal(
    lines: &[CodeLine],
    start: usize,
    rhs: &str,
) -> Result<(Value, usize), PowerflowError> {
    let number = lines[start].number;
    if let Some(body) = rhs.strip_prefix('[') {
        let (segments, rest, next) = collect_until(lines, start, body, ']')?;
        expect_statement_end(lines[next - 1].number, rest)?;
        return Ok((Value::Matrix(parse_matrix(&segments)?), next));
    }
    if let Some(body) = rhs.strip_prefix('{') {
        let (segments, rest, next) = collect_until(lines, start, body, '}')?;
        expect_statement_end(lines[next - 1].number, rest)?;
        return Ok((Value::Cell(parse_cell(&segments)?), next));
    }
    if rhs.starts_with('\'') {
        let (string, rest) = read_string(number, rhs)?;
        expect_statement_end(number, rest)?;
        return Ok((Value::Str(string), start + 1));
    }

    let (text, rest) = rhs.split_once(';').unwrap_or((rhs, ""));
    expect_statement_end(number, rest)?;
    let value = parse_number(number, text.trim(), "a scalar")?;
    Ok((Value::Number(value), start + 1))
}

/// A piece of a multi-line literal: its line, its text, and whether the line was continued
/// with `...` (which doesn't end a matrix row).
struct Segment<'a> {
    line: usize,
    text: &'a str,
    continued: bool,
}

/// Gathers a literal's body from `first` (the text after the opening bracket on line
/// `start`) until the closing `close` outside quotes. Returns the body's segments, the text
/// after the closing bracket, and the index of the line after the one it's on.
fn collect_until<'a>(
    lines: &'a [CodeLine<'a>],
    start: usize,
    first: &'a str,
    close: char,
) -> Result<(Vec<Segment<'a>>, &'a str, usize), PowerflowError> {
    let mut segments = Vec::new();
    let mut text = first;
    let mut index = start;
    loop {
        let line = &lines[index];
        if let Some(end) = find_outside_quotes(text, close) {
            segments.push(Segment {
                line: line.number,
                text: &text[..end],
                continued: false,
            });
            return Ok((segments, &text[end + 1..], index + 1));
        }
        segments.push(Segment {
            line: line.number,
            text,
            continued: line.continued,
        });
        index += 1;
        if index == lines.len() {
            return Err(parse_error(
                lines[start].number,
                format!("`{close}` never closes this literal"),
            ));
        }
        text = lines[index].code;
    }
}

fn find_outside_quotes(text: &str, target: char) -> Option<usize> {
    let mut in_string = false;
    for (i, c) in text.char_indices() {
        match c {
            '\'' => in_string = !in_string,
            c if c == target && !in_string => return Some(i),
            _ => {}
        }
    }
    None
}

/// Checks that only an optional `;` and whitespace follow a literal.
fn expect_statement_end(line: usize, rest: &str) -> Result<(), PowerflowError> {
    let rest = rest.trim();
    if rest.is_empty() || rest == ";" {
        Ok(())
    } else {
        Err(parse_error(
            line,
            format!("unexpected `{rest}` after the literal"),
        ))
    }
}

/// Parses a plain number, including MATLAB's `Inf` and `NaN`. Expressions are rejected.
fn parse_number(line: usize, token: &str, what: &str) -> Result<f64, PowerflowError> {
    token.parse::<f64>().map_err(|_| {
        parse_error(
            line,
            format!("`{token}` in {what} is not a plain number (expressions aren't supported)"),
        )
    })
}

/// Splits a matrix body into rows: `;` always ends a row, and so does the end of a line
/// unless it was continued with `...`. Every row must have the same number of columns.
fn parse_matrix(segments: &[Segment]) -> Result<Vec<Row>, PowerflowError> {
    let mut rows: Vec<Row> = Vec::new();
    let mut current: Option<Row> = None;
    let finish = |row: Option<Row>, rows: &mut Vec<Row>| -> Result<(), PowerflowError> {
        if let Some(row) = row {
            if let Some(first) = rows.first() {
                if row.values.len() != first.values.len() {
                    return Err(parse_error(
                        row.line,
                        format!(
                            "matrix row has {} columns, but the first row (line {}) has {}",
                            row.values.len(),
                            first.line,
                            first.values.len()
                        ),
                    ));
                }
            }
            rows.push(row);
        }
        Ok(())
    };

    for segment in segments {
        let mut parts = segment.text.split(';').peekable();
        while let Some(part) = parts.next() {
            for token in part
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|token| !token.is_empty())
            {
                let value = parse_number(segment.line, token, "a matrix")?;
                current
                    .get_or_insert_with(|| Row {
                        line: segment.line,
                        values: Vec::new(),
                    })
                    .values
                    .push(value);
            }
            // A `;` ends the row; so does the end of a line that isn't continued
            if parts.peek().is_some() || !segment.continued {
                finish(current.take(), &mut rows)?;
            }
        }
    }
    finish(current.take(), &mut rows)?;
    Ok(rows)
}

/// Splits a cell array body into its strings, separated by spaces, commas, `;` or new lines.
fn parse_cell(segments: &[Segment]) -> Result<Vec<String>, PowerflowError> {
    let mut strings = Vec::new();
    for segment in segments {
        let mut rest = segment.text;
        loop {
            rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',' || c == ';');
            if rest.is_empty() {
                break;
            }
            if !rest.starts_with('\'') {
                return Err(parse_error(
                    segment.line,
                    format!(
                        "cell arrays may only hold quoted strings, not `{}`",
                        rest.trim()
                    ),
                ));
            }
            let (string, after) = read_string(segment.line, rest)?;
            strings.push(string);
            rest = after;
        }
    }
    Ok(strings)
}

/// Reads a single-quoted string at the start of `text`, where `''` is an escaped quote.
/// Returns the string and the text after its closing quote.
fn read_string(line: usize, text: &str) -> Result<(String, &str), PowerflowError> {
    let body = &text[1..];
    let mut string = String::new();
    let mut chars = body.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c == '\'' {
            if matches!(chars.peek(), Some((_, '\''))) {
                string.push('\'');
                chars.next();
            } else {
                return Ok((string, &body[i + 1..]));
            }
        } else {
            string.push(c);
        }
    }
    Err(parse_error(line, "string is never closed"))
}

/// Builds a [`Case`] from the assigned fields, converting each table's rows.
fn build_case(mut fields: HashMap<String, Field>) -> Result<Case, PowerflowError> {
    let version = fields
        .remove("version")
        .ok_or(PowerflowError::MissingField { field: "version" })?;
    match version.value {
        Value::Str(found) if found == "2" => {}
        Value::Str(found) => return Err(PowerflowError::UnsupportedVersion { found }),
        other => {
            return Err(parse_error(
                version.line,
                format!("mpc.version must be a string, not {}", other.kind()),
            ))
        }
    }

    let base_mva = match take(&mut fields, "baseMVA")? {
        Field {
            value: Value::Number(value),
            ..
        } => value,
        Field { line, value } => {
            return Err(parse_error(
                line,
                format!("mpc.baseMVA must be a number, not {}", value.kind()),
            ))
        }
    };

    let bus = matrix(take(&mut fields, "bus")?, "bus")?
        .iter()
        .map(bus_from_row)
        .collect::<Result<_, _>>()?;
    let generator = matrix(take(&mut fields, "gen")?, "gen")?
        .iter()
        .map(gen_from_row)
        .collect::<Result<_, _>>()?;
    let branch = matrix(take(&mut fields, "branch")?, "branch")?
        .iter()
        .map(branch_from_row)
        .collect::<Result<_, _>>()?;
    let gencost = match fields.remove("gencost") {
        Some(field) => Some(
            matrix(field, "gencost")?
                .iter()
                .map(gencost_from_row)
                .collect::<Result<_, _>>()?,
        ),
        None => None,
    };
    let busnames = match fields.remove("bus_name") {
        Some(Field {
            value: Value::Cell(names),
            ..
        }) => Some(names),
        Some(Field { line, value }) => {
            return Err(parse_error(
                line,
                format!("mpc.bus_name must be a cell array, not {}", value.kind()),
            ))
        }
        None => None,
    };
    // Any other field (gentype, genfuel, dcline, areas, user fields) isn't used

    Ok(Case {
        base_mva,
        bus,
        generator,
        branch,
        gencost,
        busnames,
    })
}

fn take(fields: &mut HashMap<String, Field>, name: &'static str) -> Result<Field, PowerflowError> {
    fields
        .remove(name)
        .ok_or(PowerflowError::MissingField { field: name })
}

fn matrix(field: Field, name: &str) -> Result<Vec<Row>, PowerflowError> {
    match field.value {
        Value::Matrix(rows) => Ok(rows),
        other => Err(parse_error(
            field.line,
            format!("mpc.{name} must be a matrix, not {}", other.kind()),
        )),
    }
}

/// Checks a row has at least `min` columns.
fn require_columns(row: &Row, table: &str, min: usize) -> Result<(), PowerflowError> {
    if row.values.len() < min {
        Err(parse_error(
            row.line,
            format!(
                "mpc.{table} row has {} columns, but version 2 needs at least {min}",
                row.values.len()
            ),
        ))
    } else {
        Ok(())
    }
}

/// Converts a value that must be a whole number of at least `min`, such as a bus label.
fn whole(row: &Row, column: usize, what: &str, min: usize) -> Result<usize, PowerflowError> {
    let value = row.values[column];
    if value.is_finite() && value.fract() == 0.0 && value >= min as f64 {
        Ok(value as usize)
    } else {
        Err(parse_error(
            row.line,
            format!("{what} must be a whole number of at least {min}, not {value}"),
        ))
    }
}

fn bus_from_row(row: &Row) -> Result<Bus, PowerflowError> {
    require_columns(row, "bus", BUS_COLUMNS)?;
    let v = &row.values;
    let bus_type = BusType::from_code(v[1]).ok_or(PowerflowError::UnknownBusType {
        line: row.line,
        code: v[1],
    })?;
    Ok(Bus {
        bus_label: whole(row, 0, "bus number", 1)?,
        bus_type,
        pd_mw: v[2],
        qd_mvar: v[3],
        gs_mw: v[4],
        bs_mvar: v[5],
        // The format calls area and zone positive, but real cases use 0
        area: whole(row, 6, "bus area", 0)?,
        vm_pu: v[7],
        va_deg: v[8],
        base_kv: v[9],
        zone: whole(row, 10, "bus zone", 0)?,
        vmax_pu: v[11],
        vmin_pu: v[12],
    })
}

fn gen_from_row(row: &Row) -> Result<Gen, PowerflowError> {
    require_columns(row, "gen", GEN_COLUMNS)?;
    let v = &row.values;
    Ok(Gen {
        bus_label: whole(row, 0, "generator bus number", 1)?,
        pg_mw: v[1],
        qg_mvar: v[2],
        qmax_mvar: v[3],
        qmin_mvar: v[4],
        vg_pu: v[5],
        mbase_mva: v[6],
        status: v[7] > 0.0,
        pmax_mw: v[8],
        pmin_mw: v[9],
        pc1_mw: v[10],
        pc2_mw: v[11],
        qc1min_mvar: v[12],
        qc1max_mvar: v[13],
        qc2min_mvar: v[14],
        qc2max_mvar: v[15],
        ramp_agc_mw_per_min: v[16],
        ramp_10_mw: v[17],
        ramp_30_mw: v[18],
        ramp_q_mvar_per_min: v[19],
        apf: v[20],
    })
}

fn branch_from_row(row: &Row) -> Result<Branch, PowerflowError> {
    require_columns(row, "branch", BRANCH_COLUMNS)?;
    let v = &row.values;
    Ok(Branch {
        from_label: whole(row, 0, "branch from-bus number", 1)?,
        to_label: whole(row, 1, "branch to-bus number", 1)?,
        r: v[2],
        x: v[3],
        b: v[4],
        rate_a_mva: v[5],
        rate_b_mva: v[6],
        rate_c_mva: v[7],
        tap_ratio: v[8],
        shift_deg: v[9],
        status: v[10] != 0.0,
        angmin_deg: v[11],
        angmax_deg: v[12],
    })
}

/// Converts a `gencost` row: model, startup, shutdown, `N`, then `N` polynomial coefficients
/// (model 2) or `N` `(p, cost)` points (model 1). Rows can be longer, padded with zeros so
/// the matrix is rectangular.
fn gencost_from_row(row: &Row) -> Result<GenCost, PowerflowError> {
    require_columns(row, "gencost", 4)?;
    let v = &row.values;
    let n = whole(row, 3, "gencost N", 1)?;
    let (startup, shutdown) = (v[1], v[2]);
    match v[0] {
        2.0 => {
            require_columns(row, "gencost", 4 + n)?;
            Ok(GenCost::Polynomial {
                startup,
                shutdown,
                coeffs: v[4..4 + n].to_vec(),
            })
        }
        1.0 => {
            require_columns(row, "gencost", 4 + 2 * n)?;
            Ok(GenCost::PiecewiseLinear {
                startup,
                shutdown,
                points: v[4..4 + 2 * n]
                    .chunks_exact(2)
                    .map(|pair| (pair[0], pair[1]))
                    .collect(),
            })
        }
        model => Err(parse_error(
            row.line,
            format!("gencost model {model} is not 1 (piecewise linear) or 2 (polynomial)"),
        )),
    }
}
