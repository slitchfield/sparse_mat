# Implementation report: `dev/parser-hardening`

This branch hardens the MATPOWER case reader for files users write themselves, before the DC
power flow work builds on it. It has six commits on top of `main` at `7007590`: one per item,
the documentation, and this report. Every change came from evidence: a fuzz test written
before any fixes, and a probe of plausible hand-written files against the existing reader.

| | `main` | `dev/parser-hardening` |
|---|---|---|
| `powerflow` tests | 44 | 62 (23 data model, 18 locator, 4 fuzz, 8 syntax, 9 normalized cases) |
| Known panics in the reader | 1 (found by the new fuzzer) | 0 |
| Probe files handled correctly | 0 of 11 | 11 of 11 |
| MATPOWER 8.1 files that parse directly | 52 | 52 (the same ones) |
| Diff | | 8 files, +1,209 / −171, about half of it tests and this report |

## Commits

| Commit | Change |
|---|---|
| `c2dd5e4` | Encoding: `Case::from_bytes`, with byte order mark stripping and a Latin-1 fallback |
| `d890955` | Fuzz tests; fixes for a `gencost` overflow panic and saturating whole numbers |
| `dd38a53` | MATLAB and Octave syntax that hand-written files use |
| `e463f20` | Rejects `NaN` in columns the model reads, and a case with no buses |
| `540e7ff` | Module docs, `Case::parse` docs and the checklist |

## How the work was driven

1. **A fuzz test first** (`matpower_fuzz_test.rs`), modelled on `sparse_mat`'s Matrix Market
   fuzz test: truncations, 20,000 seeded mutations and oversized inputs, which must give an
   `Err`, never a panic. Its seeds are synthetic, because MATPOWER's case files can't be
   committed.
2. **A probe of 11 plausible user files** against the existing reader. Nine legitimate files
   were rejected, two invalid ones accepted, and one was misdiagnosed.

| Probe | Before | After |
|---|---|---|
| Two statements on one line | rejected | accepted |
| `"2"` (double-quoted string) | rejected | accepted |
| `%{ … %}` block comment | rejected | accepted |
| `#` comment (Octave) | rejected | accepted |
| `function s = …` with `s.bus = …` | rejected | accepted |
| `mpc.if.lims = …` (nested, unused field) | rejected | accepted, skipped |
| `mpc.note = sprintf(…)` (unused field) | rejected | accepted, skipped |
| `[1 2]'` transpose, then a comment containing `'` | rejected | accepted, skipped |
| Version 1 file | rejected as "code" | rejected as version 1 |
| `NaN` reactance | accepted | rejected |
| Empty bus table | accepted | rejected |

## Details

### Encoding (`c2dd5e4`)
- **The problem:** `Case::from_path` required UTF-8. MATLAB has long saved files in the system's
  native encoding, often Windows-1252 or Latin-1, so an accented bus name made the read fail.
  A UTF-8 byte order mark, which some editors write, made the first line an unsupported
  statement.
- **The fix:** the new `Case::from_bytes`, which `from_path` now uses, strips a byte order mark
  and decodes valid UTF-8 as UTF-8. Otherwise it decodes the bytes as Latin-1, where every byte
  is a character. The syntax is ASCII either way, so only comments and bus names can differ.

### Fuzzing and two fixes (`d890955`)
- **A panic, found by the fuzzer on its first run:** a `gencost` row's needed length,
  `4 + N` or `4 + 2N`, was computed unchecked from the file's `N`, so a huge `N` overflowed.
  The length is now checked, with an error naming `N` and the values the row actually has.
- **Silent saturation, which a fuzzer can't see:** whole numbers such as bus labels were cast
  with `as usize`, so `1e30` silently became `usize::MAX`. They're now limited to 2⁵³ (or
  `usize::MAX` if smaller), past which an `f64` can't represent every whole number.

### Syntax (`dd38a53`)
- **Accepted now:**
  - several statements on a line, separated by `;` or `,`
  - double-quoted strings, with `""` escapes
  - `%{ … %}` and Octave `#{ … #}` block comments (nestable, markers alone on their lines),
    and Octave `#` line comments
  - the struct named by the function's output (`function s = name`, `function [out] = name`),
    or `mpc` when there's no function line
- **Unused fields are skipped, whatever they hold.** That includes nested ones such as
  `mpc.if.lims`, and arbitrary expressions. It's safe because of MATLAB's value semantics:
  `mpc.other = f(mpc)` can only ever change `mpc.other`. Their brackets and strings must still
  balance, so the reader can find where each statement ends.
- **One shared string scanner** now serves comments, literals and statement ends. It follows
  MATLAB's rule for `'`: a string, unless it directly follows a name, number, closing bracket,
  `.` or `'`, where it's the transpose operator. The old scanner treated every `'` as a
  quote, which caused several of the probe failures.
- **Fields the model reads stay strictly literal.** Indexed or nested assignments to them, and
  a transposed literal, are still errors.
- **Version 1 files** (bare `baseMVA`/`bus`/… variables, or a function returning them) are
  reported as `UnsupportedVersion { found: "1" }`. A comparison like `bus == 1` isn't mistaken
  for one.
- **Safety check:** exactly the same 52 MATPOWER files parse directly as before, and the other
  32 are rejected for the same reasons. So the new skipping let no code-computing file through.

### Invalid values (`e463f20`)
- **`NaN`:** rejected in every column the model reads, with an error naming the table and
  column. No MATPOWER case uses `NaN`; I checked all 78 normalized cases. `Inf` stays legal,
  since limits use it, and `NaN` in columns the model ignores is ignored.
- **No buses:** `Case::validate` reports the new `PowerflowError::NoBuses`.

## Tests

- **`matpower_fuzz_test.rs` (4):**
  - the three seeds parse; the third uses the new syntax, so mutations reach the new scanner
  - every truncation of each seed
  - 20,000 mutations
  - oversized inputs: huge `N` and labels, 100,000-character lines, 10,000 rows, unclosed
    literals
- **`matpower_syntax_test.rs` (8):** each syntax feature, error cases for unused fields
  (unbalanced, unclosed), the strictness of used fields, and version 1 detection.
- **`data_model_test.rs` (+6):** byte order mark, Latin-1 fallback, rejected and largest
  accepted whole numbers, the `gencost` length check, `NaN`, and no buses.

## Verification

- **Locally:** `./check.sh` passes, as do the Rust 1.87 build, `cargo doc` with warnings as
  errors, and all 62 `powerflow` tests with `YBUS_REQUIRE_CASES=1`.
- **Not yet verified:** CI runs on this PR.

## Mistakes caught

- **Two test bugs:** one test renamed bus 1 without updating the branch that referred to it. I'd
  also used `Box::leak` as a shortcut in another test, and replaced it with plain string
  building.
- **A miscount:** I told the maintainer there were 63 `powerflow` tests; there are 62.
- **Branch cleanup:** a stale local copy of `dev/normalized-cases` had different commit hashes
  from `main`, because GitHub rewrote the commits when the PR was retargeted. Its contents were
  confirmed identical to `main` before it was deleted.

## Still unsupported, by design

- **Data computed with code** in fields the model reads; `scripts/normalize-cases.sh` covers
  MATPOWER's own cases.
- **Version 1 files:** a possible follow-up, since MATPOWER's `mpc_1to2` defines the
  conversion, and the reader already applies its column defaults.
- **Expressions inside literals,** such as `50/3`.

## Next

Bbus assembly and the DC power flow.
