# Implementation report: `dev/claude/hardening`

Phase 0 hardening, merged as PR #2 on 2026-10-29. This was the PR description; it is archived
here with the CI result filled in.

This branch is Phase 0 of the power flow plan: fixing the risks found while reviewing PR #1,
before more code is built on top of them. It contains five commits, one per item, and changes
6 files (+114 / −7). Three of the commits fix bugs, and each of those was reproduced with a
failing test before the fix. The other two add CI coverage.

| Commit | Change | Kind |
|---|---|---|
| `156a137` | `from_csr` rejects `rows == usize::MAX` instead of overflowing | Bug fix |
| `d9f80fc` | Compressed-storage allocation guarded against huge dimensions | Bug fix |
| `9f115ef` | Matrix Market reader no longer trusts the header's entry count as an allocation size | Bug fix |
| `6894b00` | `check.sh` lints and tests each optional feature on its own | CI coverage |
| `f5bfbad` | `rust-version = "1.87"`, MSRV-aware resolver, and an MSRV CI job | CI coverage |

## Bug fixes

### `from_csr` overflow (`156a137`)
- **Bug:** `from_csr` compared `row_ptr.len()` with `rows + 1`. For `rows == usize::MAX` that
  addition overflows: debug builds panic with "attempt to add with overflow", and release builds
  wrap to 0 silently.
- **Fix:** a checked add, so the call returns the usual
  `InvalidCsr { reason: "row_ptr must have rows + 1 entries" }`.
- **Test:** `sparsemat_from_csr_rows_overflow_is_an_error`.

### Compressed-storage allocation (`d9f80fc`)
- **Bug:** building a CSR or CSC form allocated `dim + 1` pointers with `vec!`. For a dimension
  of `usize::MAX` the `+ 1` overflowed (wrapping in release builds), and for a merely huge
  dimension (say `2^61`) it failed inside the allocator with only "capacity overflow".
- **Fix:** a `zeroed_pointers` helper that uses `checked_add` and `try_reserve_exact`, and
  panics with a message naming the dimension ("compressed storage for dimension … needs …").
- **Why a panic and not a `Result`:** these paths back infallible APIs (`iter`, `row_iter`,
  `row`, `col`, the products and the solvers). Changing all of their signatures to report a
  condition that only occurs for absurd shapes would cost every caller. The struct docs now
  state that compressed forms cost O(dim) memory however sparse the matrix is.
- **Also in this commit:** the scatter step no longer clones the pointer array as a write cursor.
  It advances the pointers in place and shifts them back, which halves pointer memory for tall,
  mostly-empty matrices. The existing CSR/CSC layout tests and the product and solver tests
  cover the rewrite.
- **Tests:**
  - `sparsemat_csr_for_usize_max_rows_panics_clearly` (CSR path)
  - `sparsemat_csc_for_usize_max_cols_panics_clearly` (CSC path)
  - `sparsemat_csr_too_large_to_allocate_panics_clearly` (`2^61` rows: no overflow, but too
    large to allocate)

### Matrix Market pre-allocation (`9f115ef`)
- **Bug:** the reader reserved the size line's claimed entry count up front. A file claiming
  `usize::MAX` entries panicked with "capacity overflow". A smaller but still dishonest count
  (10¹⁰, for example) would try to reserve about 240 GB before reading a single entry.
- **Fix:** reserve at most `2^20` entries (`MAX_PREALLOCATED_ENTRIES`) and let the list grow as
  entries are actually parsed. The existing count check still reports the mismatch with a line
  number.
- **Test:** `sparsemat_matrix_market_huge_entry_count_is_not_preallocated` checks that the parser
  returns `Parse { line: 3, message: "expected 18446744073709551615 entries, found 1" }`
  instead of panicking.

## CI coverage

### Single-feature runs (`6894b00`)
`check.sh` ran with no features and with all features, so code that only builds or lints cleanly
when `serde` and `rayon` are enabled together could slip through. It now runs clippy and the
tests with each feature alone. Both pass today, so nothing needed fixing. The feature list in
`check.sh` has to be kept in sync with `[features]` in `Cargo.toml`, and a comment says so.

### MSRV (`f5bfbad`)
- **`rust-version = "1.87"`.** I measured this rather than taking it from memory, and it
  corrects the "1.89+" I gave earlier in planning:
  - `is_multiple_of` was stabilized in 1.87 and is the newest API in use.
  - On 1.86 the build fails with `E0658: unstable library feature unsigned_is_multiple_of`.
  - On 1.87 all 164 tests pass with all features.
- **`resolver = "3"`.** `Cargo.lock` is not committed, so CI resolves dependencies on every run,
  and a future dependency release could raise the effective MSRV without anyone noticing. With
  the MSRV-aware resolver, a lockfile-free checkout resolves "to latest Rust 1.87 compatible
  versions". I verified that by building and testing a clean copy of the repo on 1.87 with no
  `Cargo.lock`.
- **New `MSRV` CI job:** it installs 1.87 and runs `cargo test --workspace --all-targets
  --all-features`. Lints and formatting stay on stable in the existing `Check` job, since clippy
  and rustfmt output varies between releases.
- **Side effect:** clippy reads `rust-version` and will no longer suggest APIs newer than 1.87.

## Verification

- `./check.sh` passes on stable after every commit. It now covers fmt, clippy (all features and
  each feature alone, `-D warnings`), and tests with no features, all features and each feature
  alone (154 core + 6 `rayon` + 4 `serde`), plus doc tests.
- The new overflow and allocation tests also pass in `--release`. That matters because release
  builds are where the old code wrapped silently instead of panicking.
- The MSRV was checked with two installed toolchains: 1.86 fails as expected and 1.87 passes.
- **CI:** on PR #2 the `Check` job and the new `MSRV` job (in its first run on GitHub) both
  passed.

## Behaviour changes

- `from_csr(usize::MAX, …)` now returns `Err(InvalidCsr)`. Before, it panicked in debug builds
  and misbehaved in release builds.
- Operations that need compressed storage on an impossibly large dimension now panic with a
  descriptive message. Before, they panicked with "attempt to add with overflow" or "capacity
  overflow", or, in release builds, wrapped and then failed with an unrelated index panic.
- Building on Rust older than 1.87 is now refused up front by Cargo, with a clear message.
  Before, it failed partway through compilation.

## Known gaps and follow-ups (not in this PR)

- **`kron` pre-allocation:** it reserves `nnz(A) · nnz(B)` entries. The result really does need
  that many, so this isn't a trust issue like the Matrix Market one, but the multiplication isn't
  checked.
- **Dense allocation:** `to_dense` and `mul_dense` allocate `rows × cols` and will abort for
  shapes that don't fit in memory. This is documented, not guarded.
- **Lowering the MSRV:** replacing the single `is_multiple_of` call with `% 2 == 0` would allow
  something lower. Clippy only suggests `is_multiple_of` when the MSRV allows it, so it wouldn't
  object. I left it at 1.87 because that release is over a year old.
- **Next:** Phase 1 of the power flow plan (`scale_rows`/`scale_cols`, index-set `select`,
  permutations and the complex helpers).
