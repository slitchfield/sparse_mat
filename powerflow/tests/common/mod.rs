//! Shared test support: locating MATPOWER case files.
//!
//! MATPOWER's case files are not committed (see `docs/PowerFlowChecklist.md`, Phase 1.5).
//! `scripts/fetch-matpower.sh` downloads them, so tests that need a case look it up here:
//!
//! ```ignore
//! mod common;
//!
//! #[test]
//! fn reads_case14() {
//!     let Some(path) = common::matpower_case("case14") else {
//!         return; // not fetched; the skip was reported on stderr
//!     };
//!     // ... use path ...
//! }
//! ```
//!
//! Lookup order:
//! 1. `MATPOWER_DIR`, if set, names an existing MATPOWER installation, as for the fetch script.
//!    A case missing from it is an error, not a skip, since the directory was chosen on purpose.
//! 2. Otherwise `testdata/matpower/matpower<VERSION>/` from the fetch script, at the version the
//!    script pins. It only counts once the script's `.ybus-verified` marker shows a complete,
//!    checksum-verified extraction.
//!
//! When the cases aren't there, tests skip by returning early. Set `YBUS_REQUIRE_CASES=1` to
//! make a missing case fail instead. CI does this, so a broken fetch can't turn into silently
//! passing tests.

// Each integration test file compiles this module separately and uses only part of it
#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Names an existing MATPOWER installation to use instead of `testdata/`.
pub const MATPOWER_DIR_VAR: &str = "MATPOWER_DIR";
/// When `1`, a missing case fails the test instead of skipping it.
pub const REQUIRE_VAR: &str = "YBUS_REQUIRE_CASES";
/// Written by `scripts/fetch-matpower.sh` after a complete, verified extraction.
pub const VERIFIED_MARKER: &str = ".ybus-verified";

/// The outcome of looking for a case file.
#[derive(Debug, PartialEq)]
pub enum Located {
    /// The case file exists at this path.
    Found(PathBuf),
    /// MATPOWER hasn't been fetched (or not completely). Tests skip, unless cases are required.
    Missing(String),
    /// The installation exists but is wrong. Tests always fail, so a misconfiguration is fixed
    /// rather than skipped.
    Broken(String),
}

/// The workspace root, the parent of this crate's directory.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the powerflow crate is inside the workspace")
        .to_path_buf()
}

/// The MATPOWER version pinned by `scripts/fetch-matpower.sh`, read from its `VERSION="…"`
/// line so the version is written down in one place.
pub fn pinned_version(script: &str) -> Option<String> {
    script.lines().find_map(|line| {
        line.trim()
            .strip_prefix("VERSION=\"")?
            .strip_suffix('"')
            .map(str::to_string)
    })
}

/// The directory the fetch script extracts the pinned release into.
pub fn fetched_dir(root: &Path) -> PathBuf {
    let script = root.join("scripts").join("fetch-matpower.sh");
    let text = fs::read_to_string(&script)
        .unwrap_or_else(|err| panic!("can't read {}: {err}", script.display()));
    let version = pinned_version(&text)
        .unwrap_or_else(|| panic!("no VERSION=\"...\" line in {}", script.display()));
    root.join("testdata")
        .join("matpower")
        .join(format!("matpower{version}"))
}

/// Looks for `data/<case>.m`, in `env_dir` if given (the value of `MATPOWER_DIR`) and
/// otherwise in `fetched_dir`. Reads no environment variables, so it can be tested directly.
pub fn locate(case: &str, env_dir: Option<&Path>, fetched_dir: &Path) -> Located {
    let file = format!("{case}.m");
    if let Some(dir) = env_dir {
        let path = dir.join("data").join(&file);
        return if path.is_file() {
            Located::Found(path)
        } else {
            Located::Broken(format!(
                "{MATPOWER_DIR_VAR}={} has no data/{file}; is it a MATPOWER directory?",
                dir.display()
            ))
        };
    }

    if !fetched_dir.join(VERIFIED_MARKER).is_file() {
        return Located::Missing(format!(
            "MATPOWER is not fetched (no complete download at {}); run scripts/fetch-matpower.sh \
             or set {MATPOWER_DIR_VAR}",
            fetched_dir.display()
        ));
    }
    let path = fetched_dir.join("data").join(&file);
    if path.is_file() {
        Located::Found(path)
    } else {
        Located::Broken(format!(
            "the verified MATPOWER download at {} has no data/{file}",
            fetched_dir.display()
        ))
    }
}

/// Parses `YBUS_REQUIRE_CASES`: unset, empty or `0` means skip, `1` means require. Anything
/// else is rejected, so a typo such as `yes` can't silently mean "skip".
pub fn parse_require(value: Option<&str>) -> bool {
    match value {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        Some(other) => panic!("{REQUIRE_VAR} must be 0 or 1, not {other:?}"),
    }
}

/// Turns a lookup into the path to use, `None` to skip, or a panic.
pub fn resolve(located: Located, require: bool) -> Option<PathBuf> {
    match located {
        Located::Found(path) => Some(path),
        Located::Missing(reason) if require => {
            panic!("{reason} ({REQUIRE_VAR}=1, so missing cases are an error)")
        }
        Located::Missing(reason) => {
            eprintln!("skipping: {reason}");
            None
        }
        Located::Broken(reason) => panic!("{reason}"),
    }
}

/// The path of MATPOWER's `data/<case>.m`, or `None` (after reporting why on stderr) when
/// MATPOWER hasn't been fetched and cases aren't required. Panics if cases are required but
/// missing, or if the installation is broken.
pub fn matpower_case(case: &str) -> Option<PathBuf> {
    let env_dir = env::var_os(MATPOWER_DIR_VAR)
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from);
    let require = parse_require(env::var(REQUIRE_VAR).ok().as_deref());
    resolve(
        locate(case, env_dir.as_deref(), &fetched_dir(&workspace_root())),
        require,
    )
}
