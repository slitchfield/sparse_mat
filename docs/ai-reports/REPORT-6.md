# Implementation report: `dev/case-model`

This branch completes two Phase 1.5 items in `docs/PowerFlowChecklist.md`: the power system
**case model** and a **MATPOWER case reader**. The maintainer started the data model (`Case`,
`Bus`, `BusType`, `Gen`, `Branch` and a first `PowerflowError`); Claude completed it and wrote
the reader, validation and tests. It has three commits (the code with its tests, the roadmap
and README updates, and this report) on top of `main` at `a66227f`.

| | `main` | `dev/case-model` |
|---|---|---|
| `powerflow` | empty crate | case model, MATPOWER reader, validation |
| `powerflow` tests | 18 (locator) | 34 (18 locator, 16 data model) |
| MATPOWER 8.1 data files that parse | none | 52 of 84; every other one errors cleanly |
| Diff | | 8 files, +1,690 / −7, about a third of it tests and this report |

## Commits

| Commit | Change |
|---|---|
| `4567a95` | The case model, the MATPOWER reader, the error type and 16 tests |
| `550672b` | Roadmap and README updates |

## The case model (`powerflow/src/case.rs`)

- **Types:** `Case` holds `base_mva` and `Vec`s of `Bus`, `Gen` and `Branch`, plus optional
  `gencost` and `busnames`.
  - `BusType` covers MATPOWER's codes 1–4: `PQ`, `PV`, `Ref`, `Isolated`.
  - `GenCost` is either `Polynomial { startup, shutdown, coeffs }` (highest order first) or
    `PiecewiseLinear { startup, shutdown, points }`.
  - All types and fields are public and documented, since this is plain data that tests and
    the coming DC model need to read.
- **File units, not per unit:** values are stored as the file gives them (MW, MVAr, degrees,
  p.u. impedances), with units in the field names (`pd_mw`, `va_deg`, `shift_deg`, `rate_a_mva`
  and so on). Solvers will convert, as MATPOWER does. This differs from the checklist's "in per
  unit", and the ticked item records why: the case stays a faithful copy of the file, so
  values can't be converted twice and comparisons with the reference results are direct.
- **Bus labels:** buses are referred to by label (the bus number in the file), not position.
  `Case::bus_indices()` maps labels to positions.
- **Statuses:** become `bool` by MATPOWER's rules. A generator is in service when its status is
  above 0, so `-1` is legal and out of service; a branch is in service when its status isn't 0.
- **`Branch::effective_tap()`:** turns the file's tap of 0 (meaning a line) into 1.
- **Construction:** `Case::from_path(impl AsRef<Path>)` reads a file, and `Case::parse(&str)`
  (also `FromStr`) parses text. Both validate.
- **`Case::validate()`:** checks a positive, finite base MVA, unique bus labels, that generators
  and branches refer to existing buses, and that `bus_name` and `gencost` have the right
  lengths. It can be called again after editing a case.

## The reader (`powerflow/src/case/matpower.rs`)

**A survey first.** Before writing the parser, all 84 MATPOWER 8.1 data files were surveyed.
The main finding: **24 of them aren't pure data**. After their matrix literals they run MATLAB
code that rescales the values, for example converting loads from kW to MW with `/ 1e3`, or R and
X from ohms to per unit. A parser that read the literals and skipped everything else would load
those cases with values wrong by a factor of 1000, without any error.

**So the reader accepts a strict subset and rejects everything else:**
- comments (`%`, outside strings), blank lines, the `function` line and `end`
- `mpc.<field> = <literal>;`, where the literal is one of:
  - a matrix of plain numbers (rows ended by `;` or a new line, commas or whitespace between
    values, `...` continuing a row)
  - a cell array of quoted strings (with `''` escapes)
  - a string or a number
- `Inf` and scientific notation
- extra trailing columns, such as the result columns of solved cases
- unknown fields such as `gentype` and `dcline`, which are read and ignored

Indexed assignments, helper variables and arithmetic such as `50/3` are errors that name the
reason.

**Values are converted with checks.** Labels must be whole numbers of at least 1. Area and zone
must be whole numbers of at least 0: the format says "positive", but real files use 0. Bus type
codes must be 1–4. `gencost` rows are checked against their `N`, and padded rows are allowed.
Errors carry line numbers.

**Survey result:** 52 of the 84 files parse, and the other 32 error cleanly with no panics:
- 24 compute their data with code
- 6 are contingency and scenario tables, not cases
- 2 use arithmetic in their literals

## Errors (`powerflow/src/powerflow_error.rs`)

`NoCaseFile` keeps the underlying I/O error's message. New variants:
- `Parse { line, message }`
- `MissingField`, `UnsupportedVersion`, `UnknownBusType`, `InvalidBaseMva`
- `DuplicateBus`, `UnknownBus`, `CountMismatch`

`PowerflowError` implements `std::error::Error`.

## Tests (`powerflow/tests/data_model_test.rs`, 16)

- **case14 in detail:** sizes, bus-type counts, spot values (bus 9's `bs_mvar`, the 4–7
  transformer's tap, generator 1's output and limits), the first cost function and the bus
  names.
- **All four test cases:** their sizes.
- **Every MATPOWER data file:** parses or errors cleanly. At least 50 must parse, and the known
  code and expression files must give the expected errors.
- **Inline-text cases:** syntax variations (commas, `...`, extra columns, `Inf`, `%` inside a
  name, `''` escapes, ignored fields), status rules, both cost models, and every error variant
  with its line number or identity.

Two expectations in the tests were wrong at first and were fixed: a miscounted line number, and
an "unclosed bracket" test that wasn't actually unclosed. The reader was right in both cases. The
second now has its own test for a run-on literal.

## Verification

- `./check.sh` passes, including with `YBUS_REQUIRE_CASES=1`.
- The Rust 1.87 MSRV build passes.
- `cargo doc` with warnings as errors is clean.

## Known limitations and follow-ups

- **Code-computing cases** are rejected, by design. The next branch, `dev/normalized-cases`,
  makes them usable by having MATPOWER evaluate them.
- **Strict column counts:** this branch requires all 21 generator and 13 branch columns of the
  version 2 format. MATPOWER itself accepts version 2 files without the optional ones
  (case533mt has 18-column generator rows), so the next branch relaxes this to MATPOWER's own
  defaults.
- **Test time:** `data_model_test` takes about 3 s in debug builds, because it parses every
  MATPOWER data file, including the 70,000-bus case.
- **Next:** Bbus assembly and DC power flow, building on `bus_indices()` and `effective_tap()`.
