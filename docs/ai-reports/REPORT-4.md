# Implementation report: `dev/claude/workspace`

This branch turns the repository, now renamed `ybus`, into a Cargo workspace. That settles the
first **Decide** item of Phase 1.5 in `docs/PowerFlowChecklist.md`. It has four commits (the
move, an empty `powerflow` crate, the documentation, and this report) on top of `main` at
`31708e4`. No library code changes: every file under `sparse_mat/src` and `sparse_mat/tests` is
a pure rename.

| | `main` | `dev/claude/workspace` |
|---|---|---|
| Layout | one crate at the root | virtual workspace: `sparse_mat/`, `powerflow/` |
| Repository URL | `slitchfield/sparse_mat` | `slitchfield/ybus` |
| Tests (all features) | 191, 1 doctest | 191, 1 doctest (unchanged) |
| Diff | | 33 files, +281 / −209, almost all renames and the README split |

## Commits

| Commit | Change |
|---|---|
| `8f4d895` | Move `sparse_mat` into `sparse_mat/` under a workspace manifest |
| `2ee2188` | Add an empty `powerflow` crate |
| `c11aca9` | Split the README into a workspace overview and a crate README, and update the URLs |

## Details

### The move (`8f4d895`)
- **Renames:** `src/`, `tests/`, `Cargo.toml` and `README.md` move into `sparse_mat/` with
  `git mv`, so `git log --follow` and blame keep working. `git diff -M` shows them as 100%
  renames.
- **Root `Cargo.toml`** is a virtual manifest. `[workspace.package]` holds the fields both
  crates share (edition, license, repository, rust-version), and each crate opts in with
  `<field>.workspace = true`. The resolver setting moves here too, since Cargo ignores it in
  member crates. The repository URL is now `https://github.com/slitchfield/ybus`.
- **Crate-specific fields** stay in `sparse_mat/Cargo.toml`: version, description, readme,
  keywords and categories.
- **`check.sh`:** a virtual workspace has no features of its own, so the feature-by-feature
  loop names them with their crate (`sparse_mat/serde`, `sparse_mat/rayon`). Everything else
  already used `--workspace`.
- **CI:** no functional change. The MSRV job's comment now points at the root `Cargo.toml`.
- **Packaging:**
  - `LICENSE-MIT`, `LICENSE-APACHE` and `AI-DISCLAIMER.md` are symlinked into `sparse_mat/`, so
    `cargo package -p sparse_mat` still ships them (Cargo copies the target of a symlink).
    The old `exclude` list is gone, since CI files are now outside the crate directory.
  - **Changed from the last branch's decision:** `docs/`, including the AI reports, no longer
    ships in the package, because it's outside the crate directory. The disclaimer still
    ships, and it links to the reports on GitHub. Say if you want the reports symlinked in
    too.
- The doctest's `include_str!("../README.md")` path is unchanged, because the README moved
  together with `src/`.

### `powerflow` (`2ee2188`)
- **Contents:** only `Cargo.toml` and a `lib.rs` with a crate doc comment. No placeholder code,
  given that the last one (`add`) took a checklist item to remove.
- It depends on `sparse_mat` by path, and is `publish = false` until it has an API.
- **`deny.toml`:** cargo-deny counts a path dependency without a version as a wildcard and
  failed the bans check. `allow-wildcard-paths = true` is its documented option for this: it
  allows such dependencies in unpublished crates only, so publishing `powerflow` later will
  require adding a version to the dependency.

### Documentation (`c11aca9`)
- **Root `README.md`:** now the workspace overview, with:
  - a table of the two crates and their status
  - a line on the name (Y<sub>bus</sub>)
  - Development: `check.sh`, MSRV and `cargo deny`
  - the documentation index, and the license
  - the References section. It moved here from the crate README because it covers the whole
    roadmap, and `AI-DISCLAIMER.md` links to `README.md#references`. Its algorithm table's
    first column now says "in `sparse_mat` unless noted".
- **`sparse_mat/README.md`:** keeps the quick start and the API overview through Optional
  features, so the doctested example is unchanged. It ends with short Development, License and
  References sections that link back to the root. Its links into `docs/` are now `../docs/…`.
- **URLs:** the quick start's git dependency is now
  `sparse_mat = { git = "https://github.com/slitchfield/ybus" }`, and the disclaimer's issue link
  points at `ybus`. Cargo looks through the repository for the package by name, so the git
  dependency still resolves to the crate in its subdirectory.
- **Checklist:** the workspace **Decide** item is ticked, with the decision and the new repo
  name.
- **Left alone:** past AI reports still mention `sparse_mat` URLs and root-level paths. They're
  records of what was true at the time.

## Verification

- `./check.sh` passed after every commit, over both crates. That includes the README doctest,
  so its new location is covered.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace --all-features` is clean.
- `cargo deny check`: advisories, bans, licenses and sources all ok.
- `cargo package -p sparse_mat --list` lists the sources, tests, README, both licenses and the
  disclaimer, with no warnings.
- **Not checked:** the git dependency URL against GitHub, which can't resolve until this branch
  is merged; MSRV locally (no 1.87 toolchain); CI, since the branch isn't pushed.

## Notes and follow-ups

- **Local directory:** the working copy is still at `~/projects/sparse_mat`. Renaming it to
  `ybus` is harmless for git and Cargo. Claude Code's per-project memory is keyed on the path,
  though, so a rename starts that memory afresh.
- **GitHub redirects:** GitHub redirects the old `slitchfield/sparse_mat` URLs to `ybus` as long
  as no new repository takes the old name. So existing clones and links keep working, but
  they shouldn't be relied on.
- **Next:** the second Phase 1.5 **Decide** item, the MATPOWER case files' license terms, and
  then the case model and reader in `powerflow/`. Octave is installed now, so the reference
  results can come from MATPOWER's own `rundcpf`.
