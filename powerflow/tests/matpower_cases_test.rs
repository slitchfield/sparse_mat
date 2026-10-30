//! Tests for the MATPOWER case locator in `common`, plus a check that the cases the power flow
//! tests use are available whenever `YBUS_REQUIRE_CASES=1` (as in CI).

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::{locate, parse_require, pinned_version, resolve, Located, VERIFIED_MARKER};

/// A fresh, empty directory under Cargo's per-target scratch space, unique to `name`.
fn scratch_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("matpower_cases_test")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A fake MATPOWER installation holding `cases`, optionally with the verified marker.
fn fake_install(name: &str, cases: &[&str], verified: bool) -> PathBuf {
    let dir = scratch_dir(name);
    fs::create_dir_all(dir.join("data")).unwrap();
    for case in cases {
        fs::write(
            dir.join("data").join(format!("{case}.m")),
            "function mpc = x\n",
        )
        .unwrap();
    }
    if verified {
        fs::write(dir.join(VERIFIED_MARKER), "").unwrap();
    }
    dir
}

#[test]
fn finds_a_case_in_a_verified_download() {
    let dir = fake_install("verified", &["case14"], true);
    assert_eq!(
        locate("case14", None, &dir),
        Located::Found(dir.join("data").join("case14.m"))
    );
}

#[test]
fn an_unverified_download_is_missing_not_found() {
    // An interrupted extraction leaves files but no marker; it must not be used
    let dir = fake_install("unverified", &["case14"], false);
    assert!(matches!(locate("case14", None, &dir), Located::Missing(_)));
}

#[test]
fn no_download_is_missing_with_a_pointer_to_the_fetch_script() {
    let dir = scratch_dir("absent").join("matpower8.1");
    match locate("case14", None, &dir) {
        Located::Missing(reason) => {
            assert!(reason.contains("scripts/fetch-matpower.sh"), "{reason}");
            assert!(reason.contains("MATPOWER_DIR"), "{reason}");
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

#[test]
fn a_verified_download_without_the_case_is_broken() {
    let dir = fake_install("verified-incomplete", &["case14"], true);
    assert!(matches!(locate("case30", None, &dir), Located::Broken(_)));
}

#[test]
fn matpower_dir_takes_precedence_and_needs_no_marker() {
    let env_dir = fake_install("env-dir", &["case30"], false);
    let fetched = fake_install("env-dir-fetched", &["case30"], true);
    assert_eq!(
        locate("case30", Some(&env_dir), &fetched),
        Located::Found(env_dir.join("data").join("case30.m"))
    );
}

#[test]
fn a_matpower_dir_without_the_case_is_broken_even_if_fetched() {
    // An explicitly chosen directory that's wrong is a misconfiguration, so no fallback
    let env_dir = scratch_dir("env-dir-empty");
    let fetched = fake_install("env-dir-empty-fetched", &["case118"], true);
    match locate("case118", Some(&env_dir), &fetched) {
        Located::Broken(reason) => assert!(reason.contains("MATPOWER_DIR="), "{reason}"),
        other => panic!("expected Broken, got {other:?}"),
    }
}

#[test]
fn parse_require_accepts_only_0_and_1() {
    assert!(!parse_require(None));
    assert!(!parse_require(Some("")));
    assert!(!parse_require(Some("0")));
    assert!(parse_require(Some("1")));
}

#[test]
#[should_panic(expected = "must be 0 or 1")]
fn parse_require_rejects_other_values() {
    parse_require(Some("yes"));
}

#[test]
fn resolve_skips_missing_cases_unless_required() {
    let path = PathBuf::from("data/case14.m");
    assert_eq!(resolve(Located::Found(path.clone()), true), Some(path));
    assert_eq!(resolve(Located::Missing("not fetched".into()), false), None);
}

#[test]
#[should_panic(expected = "YBUS_REQUIRE_CASES=1")]
fn resolve_fails_on_missing_cases_when_required() {
    resolve(Located::Missing("not fetched".into()), true);
}

#[test]
#[should_panic(expected = "wrong install")]
fn resolve_always_fails_on_a_broken_install() {
    resolve(Located::Broken("wrong install".into()), false);
}

#[test]
fn pinned_version_reads_the_fetch_script() {
    assert_eq!(
        pinned_version("set -e\nVERSION=\"8.1\"\nSHA256=\"abc\"\n").as_deref(),
        Some("8.1")
    );
    assert_eq!(pinned_version("no version here"), None);
}

#[test]
fn reference_results_match_the_pinned_matpower_version() {
    // The committed reference results must come from the release the tests fetch, or the
    // power flow tests would compare against results from different case data
    let root = common::workspace_root();
    let script = fs::read_to_string(root.join("scripts/fetch-matpower.sh")).unwrap();
    let pinned = pinned_version(&script).expect("the fetch script pins a version");
    let versions =
        fs::read_to_string(root.join("powerflow/tests/data/matpower-reference/VERSIONS")).unwrap();
    assert!(
        versions
            .lines()
            .any(|line| line.trim() == format!("MATPOWER {pinned}")),
        "fetch-matpower.sh pins MATPOWER {pinned}, but the reference results' VERSIONS file \
         says:\n{versions}\nRegenerate them with scripts/generate-reference.sh"
    );
}

#[test]
fn power_flow_cases_are_available() {
    // Skips locally when MATPOWER isn't fetched; with YBUS_REQUIRE_CASES=1 it checks the
    // fetch worked, so the data-driven tests can't all skip silently
    for case in ["case14", "case30", "case118", "case2383wp"] {
        let Some(path) = common::matpower_case(case) else {
            return;
        };
        let text = fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("mpc.bus") && text.contains("mpc.branch"),
            "{} doesn't look like a MATPOWER case file",
            path.display()
        );
    }
}
