# Implementation report: `dev/normalized-cases`

This branch makes MATPOWER's code-computing case files usable. The reader from
`dev/case-model` (REPORT-6) rejects them, because ignoring their code would load wrong values.
Instead, MATPOWER itself evaluates them and the result is written back out as literals. It has
four commits on top of `dev/case-model` (stacked: merge that one first): a reader fix found
along the way, the normalization, its CI and documentation, and this report.

| | `dev/case-model` | `dev/normalized-cases` |
|---|---|---|
| MATPOWER 8.1 cases usable in tests | 52 (read directly) | all 78, normalized (the other 6 data files aren't cases) |
| `powerflow` tests | 34 | 44 (18 locator, 17 data model, 9 normalized cases) |
| CI | fetches MATPOWER | also installs Octave and normalizes, on a cache miss only |
| Diff | | 10 files, +649 / −20, including this report |

## Commits

| Commit | Change |
|---|---|
| `4874652` | The reader accepts version 2 rows without the optional columns, as MATPOWER does |
| `2183b5d` | `scripts/normalize-cases.sh` and `normalize_cases.m`, the `normalized_case` locator, and 9 tests |
| `6daf06e` | CI caching and Octave install; README, module docs and checklist |

## The options considered

The 24 code-computing files were surveyed first. 23 of them share one template: they import
MATPOWER's column-name constants, convert loads from kW to MW, and in 21 cases convert R and X
from ohms to per unit. Only case8387pegase has real control flow (`if fixed … end`).

The options, discussed with the maintainer:
1. **Evaluate offline with MATPOWER (chosen).** Correct by definition, covers all 24, and needs
   no new Rust code. It needs Octave, a standard apt package, at conversion time.
2. **Recognize the template in the reader.** Covers 23 of 24 with no external tool, but it's a
   partial MATLAB interpreter whose mistakes could be silent.
3. **A MATLAB-subset interpreter.** Covers everything, at the cost of implementing a language.
4. **Call Octave at runtime.** A poor fit for a library.
5. **Keep rejecting.** Legitimate, since most of the 24 are small distribution feeders.

Options 1 and 2 also combine well: option 1's output would be a differential test oracle for any
future option 2.

## Normalization (`scripts/normalize-cases.sh`, `scripts/normalize_cases.m`)

- **How it works:** every file in MATPOWER's `data/` goes through MATPOWER's `loadcase` under
  Octave. Each case is written as `mpc.<field> = [ … ];` literals: version, baseMVA, bus, gen,
  branch, and `gencost` and `bus_name` when present. Files `loadcase` can't load as a case are
  listed and skipped.
- **Full precision:** values are written with `%.17g`, enough to round-trip any double.
  MATPOWER's `savecase` writes `%.9g`, which would round what `loadcase` computed; in case10ba,
  for example, 1840 kW becomes `1.8400000000000001` MW.
- **Never partial or stale:** output is built in a `.part` directory and moved into place only
  when complete, with a `.ybus-normalized` marker. A `SOURCE` file records the physical path of
  the MATPOWER directory and the versions used. The script reuses complete output made from the
  same directory, so a rerun is instant; a fresh run takes about 37 s.
- **Location:** `testdata/matpower/<matpower-dir-name>-normalized/`, gitignored like the
  download, so no MATPOWER data is committed.

## The locator (`powerflow/tests/common/mod.rs`)

`normalized_case(name)` follows the same rules as `matpower_case`:
- **Missing:** skip, or fail under `YBUS_REQUIRE_CASES=1`. That covers no complete output, and
  output made from a different MATPOWER directory, such as after a version bump.
- **Broken:** always fail, when a case is absent from complete output.
- **Version warning:** the same one as for a `MATPOWER_DIR` of another version.

## A reader fix found along the way (`4874652`)

Normalized case533mt still has **18-column generator rows**: MATPOWER doesn't pad version 2
files, and its power flow reads generator columns only up to `Pmin`. The reader had required all
21, which is stricter than MATPOWER.

Generator rows now need 10 columns and branch rows 11. Missing optional columns get the values
from MATPOWER's own version 1 to 2 conversion (`mpc_1to2` in `loadcase.m`):
- 0 for generator columns 11–21
- −360 and 360, meaning unconstrained, for the branch angle limits

## Tests (`powerflow/tests/normalized_cases_test.rs`, 9; plus 1 in `data_model_test.rs`)

- **Every normalized case parses,** including the formerly rejected case10ba, case69,
  case8387pegase and case533mt_hi.
- **Normalizing a literal case changes nothing:** for each of the 52 directly readable cases,
  the normalized copy parses to an *identical* `Case`. This checks both that `%.17g` loses no
  precision and that `loadcase` adds nothing the reader would read differently.
- **case10ba matches its own code:** the original is cut just before its conversion code, its
  raw literals are parsed (in kW and ohms), and the file's arithmetic is redone in Rust. The
  normalized values must match **exactly**. The test also pins the factor of 1000 that ignoring
  the code would cost: 1840 versus 1.84.
- **case8387pegase's code is evaluated, not assumed:** its block only runs when `fixed` is set,
  which it isn't by default, so 615 generators keep infinite limits. Guessing at the code's
  effect would get this wrong.
- **case533mt's `50/3` is evaluated:** `baseMVA` is `50/3` exactly.
- **Locator unit tests:** found; incomplete output; output from another source; a missing case.
- **Short-row defaults** (in `data_model_test.rs`): generator and branch rows without the
  optional columns, and a row that is too short.

## CI (`.github/workflows/rust.yml`)

- **Two separate caches:**
  - the MATPOWER download, now excluding the normalized output so the two can't mix under one
    key
  - the normalized cases, keyed on the MATPOWER pin and both normalization scripts
- **Octave on a cache miss only:** Octave is installed with `apt-get install
  --no-install-recommends octave`, then the script runs; on a hit, the script exits at once.
- **Required in CI:** `./check.sh` runs with `YBUS_REQUIRE_CASES=1`, so the normalized-case
  tests can't skip there. The MSRV job skips them.

## Verification

- **Locally:** `./check.sh` passes, with and without `YBUS_REQUIRE_CASES=1`, as do the Rust 1.87
  build and `cargo doc` with warnings as errors.
- **Not yet verified:** the CI steps, including the Octave install on a cache miss. This branch's
  PR is their first run.

## Mistakes caught

- **A stricter-than-MATPOWER reader:** the column-count rule described above.
- **A rebase that went wrong:** while stacking the branches, amending `REPORT-6` and then plainly
  rebasing replayed the old report commit, which conflicted. The rebase was aborted and redone
  with `--onto`. No work was lost.

## Follow-ups

- **Octave versions:** CI uses Ubuntu's Octave, currently 8.4, and the reference results were
  generated with 6.4. Normalization output depends only on MATPOWER's `loadcase` and `printf`,
  so this shouldn't matter; the `SOURCE` file records the Octave version in case it ever does.
- **Next:** Bbus assembly and DC power flow. Its tests can now draw on all 78 cases, which is
  useful, because the original four have no isolated buses, no `Gs` and only case2383wp has
  phase shifters.
