# Implementation report: `dev/claude/matpower-harness`

This branch settles the second **Decide** item of Phase 1.5 in `docs/PowerFlowChecklist.md`,
MATPOWER's license terms. It adds a repeatable harness that fetches MATPOWER instead of
committing its case files, and it commits DC power flow reference results for the three
public-source cases. It has four commits (the fetch script, the generator and results, the
documentation, and this report) on top of `main` at `c9facf9`. No Rust code changes.

| | `main` | `dev/claude/matpower-harness` |
|---|---|---|
| Phase 1.5 **Decide** items open | 1 | 0 |
| MATPOWER case files in the repo | none | none (fetched on demand) |
| Committed reference results | none | case14, case30, case118 (9 CSVs) |
| Diff | | 19 files, +828 / −4, mostly generated CSV, two scripts and this report |

## Commits

| Commit | Change |
|---|---|
| `66b7e2b` | `scripts/fetch-matpower.sh`, a gitignored `testdata/`, and the license decision |
| `87ef294` | `scripts/generate-reference.sh` and `generate_reference.m`, and the generated results with their README |
| `4846ea1` | Root README, `AI-DISCLAIMER.md` and checklist updates |

## The license review

The decision recorded in the checklist rests on these findings, from MATPOWER 8.1's `LICENSE`
and the case files' headers:
- **Code:** MATPOWER's code is BSD 3-clause.
- **Case files:** the `LICENSE` says outright that they are not covered by the BSD license.
  The data "has either been included with permission or has been converted from data
  available from a public source."
- **Where each case comes from:**
  - case14 and case118 are the IEEE test cases from the University of Washington archive.
  - case30 was compiled by MATPOWER's authors from two published papers.
  - case2383wp is "distributed with the permission of" its provider, with no stated scope for
    passing it on.
- **Conclusion:** fetch every case file rather than commit it. Commit results only for the
  three public-source cases, and nothing derived from case2383wp. This is a reading of the
  documents, not legal advice.

## Details

### Fetch script (`66b7e2b`)
- **What it fetches:** the official release asset `matpower8.1.zip` (47 MB), not GitHub's
  auto-generated tag tarball, whose bytes aren't guaranteed to stay stable. The committed
  SHA-256 matches the digest GitHub publishes for the asset in its releases API.
- **Behaviour:** it unpacks into `testdata/matpower/matpower8.1/` and prints that path. A
  marker file is written only after a verified, complete extraction, so a rerun reuses it
  without network access. A partial download is written to `.part` and renamed only when
  complete.
- **`MATPOWER_DIR`:** points the script at an existing installation instead, after checking it
  contains the four case files used.
- **Tested by hand, every path:**
  - a fresh download (about 5 seconds)
  - a cached rerun
  - a bad `MATPOWER_DIR`, which exits 1 with a clear message
  - a corrupted archive, which is re-downloaded
  - a checksum mismatch, which refuses to extract and exits 1, tested with a copy of the
    script holding a wrong hash
- **Dependencies:** `curl`, `unzip`, and `sha256sum` or `shasum`.

### Reference results (`87ef294`)
- **`generate_reference.m`:** runs `rundcpf` quietly on each case and writes three CSVs per
  case, plus a `VERSIONS` file (MATPOWER 8.1, Octave 6.4.0):
  - bus number and angle in degrees
  - branch row, endpoints, status and from-end flow in MW
  - generator row, bus and output in MW

  Values have 17 significant digits, so they parse back to exactly the doubles MATPOWER
  computed. That leaves the tolerance choice to the tests, rather than baking rounding into
  the data.
- **Octave detail:** a script's functions have to be defined before they are used, so the
  file starts with `1;` and the CSV helper comes first. That's unusual compared with MATLAB, so
  there's a comment explaining it.
- **`generate-reference.sh`:** runs the fetch script, then the Octave script, writing into
  `powerflow/tests/data/matpower-reference/`.
- **Checks:**
  - A second generation is byte-identical to the first.
  - Total generation equals total load (including shunt conductance) to 10 decimal places in
    every case, as a lossless DC power flow requires: 259, 189.2 and 4242 MW.
  - The slack bus angle is 0.
- **Attribution:** the results directory's README documents the columns and how to
  regenerate them. It cites MATPOWER and each case's origin. The two case30 papers were taken
  from the case file's header and then checked against online records, as `AI-DISCLAIMER.md`
  promises. Both matched, and their DOIs were added: 10.1109/TPAS.1974.293972 and
  10.1109/59.630479.

### Documentation (`4846ea1`)
- **Root README:** a Development bullet for the two scripts and `MATPOWER_DIR`. The MATPOWER
  note under "Licenses of related software" now describes what the repository does instead of
  what to check.
- **`AI-DISCLAIMER.md`:** a "Third-party test data" line next to "No third-party code". It says
  the case files aren't included, and that the results MATPOWER computed from three public
  cases are, with a link to their credits.
- **Checklist:** the **Decide** item is ticked with the reasoning. The reference-results item
  stays open, with a note: its second half, tests comparing within a tolerance, needs the DC
  power flow.

## Verification

- `./check.sh` passes, and `cargo deny check` passes. Neither exercises the new scripts, which
  were tested by hand as described above.
- **Not done:**
  - There's no CI step for the harness yet, because nothing consumes the fetched cases.
  - The `YBUS_REQUIRE_CASES` switch, which turns a skipped test into a failure in CI, isn't
    implemented. Both should come with the first test that reads a case file.
  - The scripts haven't been run through `shellcheck`.
  - The results haven't been regenerated with MATLAB to compare against Octave.

## Notes and follow-ups

- **Next:** the case model and MATPOWER `.m` reader in `powerflow/`. The reader will be the
  first consumer of `testdata/`. It needs a helper that finds the cases (`MATPOWER_DIR`, then
  `testdata/`), skips with a pointer to the fetch script when they're missing, and fails
  instead when `YBUS_REQUIRE_CASES=1`. CI then needs a fetch step with a cache keyed on the
  checksum.
- **Bumping MATPOWER:** change `VERSION` and `SHA256` together, then rerun
  `generate-reference.sh`. Any change to the results then shows up in review as a CSV diff, and
  `VERSIONS` records the new versions.
- **case2383wp:** reference results for it, wanted later for AC validation in Phase 4.5, would
  have to be generated locally by the harness, not committed.
