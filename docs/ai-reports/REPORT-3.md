# Implementation report: `dev/claude/housekeeping`

This branch clears the Housekeeping section of the power flow plan
(`docs/PowerFlowChecklist.md`), the items left over from the PR #1 pre-push review. It has six
commits, one per checklist item plus this report, on top of `main` at `b862b98`. Fuzzing the
Matrix Market reader found a real overflow bug, which is fixed here. It also found a second
overflow that needs a decision from you; see [Open decision](#open-decision).

| | `main` | `dev/claude/housekeeping` |
|---|---|---|
| Housekeeping items done | 1 of 5 | 5 of 5, plus one new **Decide** item |
| Tests (all features) | 186 (176 core, 6 `rayon`, 4 `serde`), 0 doctests | 191 (176 core, 5 fuzz, 6 `rayon`, 4 `serde`), 1 doctest |
| CI jobs | Check, MSRV | Check, MSRV, Dependencies |
| `cargo package --list` | warns: no description or repository | no warnings |
| Diff | | 8 files, +484 / −15, mostly the fuzz tests and this report |

## Commits

| Commit | Change |
|---|---|
| `9425827` | Remove the leftover template `add` function from `lib.rs` |
| `bea7b79` | Package metadata: description, repository, keywords, categories |
| `7087f13` | Fuzz tests for the Matrix Market reader, and a fix for a skew-symmetric overflow |
| `f89573f` | `cargo deny` configuration and a Dependencies CI job |
| `63f79ca` | Doctest the README's examples |

Each commit ticks its item in the checklist.

## Details

### Remove `add` (`9425827`, `lib.rs`)
- Deleted the `add(left, right)` placeholder from the `cargo new` template. Nothing used it, and
  it had no tests.
- **Breaking:** strictly speaking this removes a public function. The crate is unreleased
  (0.1.0, git dependency only), so no version bump.

### Package metadata (`bea7b79`, `Cargo.toml`)
- Added `description`, `repository`, `readme`, `keywords` and `categories`. Checked against a
  copy of `main`'s manifest: `cargo package --list` warned "manifest has no description,
  documentation, homepage or repository" before, and gives no warnings now.
- **Decided:** `exclude` keeps `.github/` and `check.sh` out of the package, because they're
  only for CI. The docs, including these reports, stay in alongside `AI-DISCLAIMER.md`, so the
  AI disclosure ships with the code. Easy to change if you'd rather ship a leaner package.
- `cargo package --no-verify` builds the `.crate` cleanly. Nothing is published.

### Fuzzing the Matrix Market reader (`7087f13`, new `tests/matrix_market_fuzz_test.rs`)
- **Approach:** deterministic fuzz-style tests instead of `cargo-fuzz`. `cargo-fuzz` needs
  nightly and a separate `fuzz/` crate, and its runs aren't reproducible in CI. These tests use
  the same fixed-seed LCG as the other test files, need only `std`, and run in under a second
  in `./check.sh`. A coverage-guided `cargo-fuzz` target would still be a worthwhile extra
  later.
- **Inputs:**
  - every byte-prefix truncation of five valid seed files, which cover all three supported
    fields, all three symmetries, comments, blank lines, duplicates and an empty matrix
  - 20,000 random mutations of those seeds: byte flips, inserts and deletes, inserted
    "interesting" tokens (integer limits, `1e400`, `nan`, `0x10`, NUL, a BOM, non-ASCII), and
    duplicated, deleted and truncated lines
  - oversized input: entry counts of `usize::MAX` and 10¹², shapes of `usize::MAX`, sizes and
    indices past `usize`, a 100,000-digit value, a 200,000-token line, and 10,000 surplus
    entries
  - invalid UTF-8
- **Element types:** every input is parsed as `f64`, `f32`, `i64`, `i32`, `u32` and `u8`.
- **Checks:** `catch_unwind` around each parse, failing with the input that panicked. A
  successful parse must also survive a write and re-read unchanged (skipped when the matrix
  holds NaN, since NaN ≠ NaN).
- **Bug found and fixed:** skew-symmetric files computed each mirrored entry as `0 - value`.
  That overflows for any nonzero unsigned value and for `i32::MIN`: a panic in debug builds and
  a silently wrong entry in release builds. The first failing input was
  `2 1 3` in a skew-symmetric file read as `u32`.
  - **Fix:** the reader now parses the negated token (it toggles a leading `-`, and turns `+`
    into `-`) instead of doing the arithmetic. A value with no negation in the element type
    fails the parse, so the error is a `Parse` error naming the line: "value '3' has no
    negation in this element type". This needed no new trait bounds on `Scalar`.
  - **Regression test:** `matrix_market_skew_symmetric_values_without_a_negation_error`
    covers `u32`, `i32::MIN`, a `pattern` file read as `u8`, and the cases that must still work:
    `i32::MAX`, `+2.5` and `i8` `-127`.
- **Edge case to know about:** a skew-symmetric file with an explicit off-diagonal `0` is now
  rejected for unsigned types, because `"-0"` doesn't parse as unsigned. Such entries are
  meaningless and get dropped anyway, so I left it.
- **Longer run:** before committing, a one-off run of about 780,000 mutated inputs (39 seeds ×
  20,000 inputs, two rounds of mutations each, in debug so overflow checks are on) across all
  six types found nothing more. That run isn't kept in the suite, to keep `check.sh` fast.
- **Not a parser bug:** a `usize::MAX` × `usize::MAX` matrix parses fine, but *writing* it
  panics, because `iter()` builds the CSR form and compressed storage refuses dimensions that
  large (the deliberate Phase 0 guard). The fuzz test only round-trips shapes under 2²⁰. See the
  follow-ups.

### `cargo deny` (`f89573f`, new `deny.toml`, `rust.yml`)
- **Decided:** `cargo deny` rather than `cargo audit`. It checks RustSec advisories as
  `cargo audit` does, and also licenses, banned crates and sources. The license check enforces
  the claims in `AI-DISCLAIMER.md` automatically.
- **Configuration:**
  - licenses: only `MIT`, `Apache-2.0` and `Unicode-3.0`
  - advisories: any RustSec advisory fails
  - versions: wildcard versions are denied, and duplicate versions only warn
  - sources: only crates.io
  - all features are checked
- **`Unlicense`:** `AI-DISCLAIMER.md` lists `Unlicense`, but it isn't in the allow list.
  The only crate that offers it is `memchr`, a dev-dependency pulled in by `serde_json`, which
  is licensed "Unlicense OR MIT", and MIT already covers it. cargo-deny warns about allowances
  that nothing uses. `deny.toml` explains this in a comment. The disclaimer is still accurate,
  so I didn't change it.
- **Result:** `advisories ok, bans ok, licenses ok, sources ok` with cargo-deny 0.20.2.
- **CI:** a new Dependencies job runs `EmbarkStudios/cargo-deny-action@v2`. New advisories
  can make this job fail on a branch nobody has touched. That's intended, but it isn't a
  regression in the code.

### README doctest (`63f79ca`, `lib.rs`)
- **Decided:** doctest the README through a `#[cfg(doctest)] pub struct ReadmeDoctests`
  carrying `#[doc = include_str!("../README.md")]`, rather than making the README the crate
  docs as the checklist suggested. The README's relative links (`docs/PowerFlowChecklist.md`,
  the licenses) would be broken in rustdoc, and the README and crate docs have different
  audiences. The struct only exists while doctests are compiled, so it isn't public API.
- `cargo test --doc` (already the last step of `check.sh`) now runs one doctest instead of
  none. **Checked:** changing the README's `assert_eq!(a[(1, 1)], 2.0)` to `3.0` makes the
  doctest fail, so drift is caught.

## Open decision

Integer overflow when duplicate entries are summed. `from_triplets` sums duplicates with `+`,
and so does the Matrix Market reader through it. For integer element types, two `i32::MAX`
entries at one position panic in debug builds and wrap in release builds. The fuzzer's random
inputs didn't hit this; a targeted probe did. Two options:

1. **Keep integer `+` semantics (current behaviour, now documented** on `from_matrix_market`).
   This is consistent with the matrix `Add`/`Mul` operators and with Rust integers generally.
2. **Return an error.** That needs a checked-add capability, which `num_traits::Num` doesn't
   provide and floats don't implement as `CheckedAdd`. It would mean a new trait in `Scalar`'s
   bounds, which affects every generic signature.

I recommend option 1: integer matrices are a side use of a crate aimed at `f64` and
`Complex<f64>` power flow. It's recorded as a **Decide** item in the checklist's Housekeeping
section, so the section isn't fully ticked until you choose.

## Verification

- `./check.sh` passed after the final commit. It covers fmt, clippy with `-D warnings` (all
  features and each feature alone), and tests with no features, all features and each feature
  alone, plus doc tests.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` is clean.
- `cargo deny check` passes, as above.
- **MSRV:** not checked locally, because the Rust 1.87 toolchain isn't installed here. The new
  code uses only long-stable APIs (`str::strip_prefix`, `Option::filter`, `catch_unwind`), so
  the MSRV job in CI should pass. Please confirm it does.
- **CI:** not run yet. The branch hasn't been pushed and no PR has been opened.

## Notes and follow-ups

- **Writing huge shapes:** `write_matrix_market` panics for dimensions near `usize::MAX`,
  because it goes through `iter()` and the CSR form. It could iterate the hash map in sorted
  order instead, and then it would always succeed. This is low priority, since no real matrix
  has those dimensions, but it's the only panic left that a parsed matrix can trigger.
- **Coverage-guided fuzzing:** a `cargo-fuzz` target on nightly would explore input space
  more systematically than these fixed-seed tests. It's worth adding if the case reader in
  Phase 1.5 grows its own text parser, since a MATPOWER `.m` reader has the same risks.
- With Housekeeping done (apart from the decision above), next is Phase 1.5, starting with its
  two **Decide** items: workspace or separate repo, and the MATPOWER license terms.
