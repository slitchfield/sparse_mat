//! Tests for MATLAB and Octave syntax that hand-written case files use: comments in every
//! form, double-quoted strings, several statements on a line, other struct names, fields the
//! model doesn't use (which are skipped whatever they hold), and the version 1 format.

use powerflow::case::Case;
use powerflow::powerflow_error::PowerflowError;

/// The three required tables of a minimal case, assigned to fields of `variable`.
fn tables(variable: &str) -> String {
    format!(
        "{variable}.bus = [ 1 3 0 0 0 0 1 1 0 135 1 1.05 0.95; 2 1 50 10 0 0 1 1 0 135 1 1.05 0.95 ];\n\
         {variable}.gen = [ 1 0 0 1 -1 1 100 1 200 0 ];\n\
         {variable}.branch = [ 1 2 0.01 0.1 0.02 0 0 0 0 0 1 ];\n"
    )
}

/// A minimal valid case with `body` inserted after the version and base MVA.
fn case_with(body: &str) -> String {
    format!(
        "function mpc = syntax\nmpc.version = '2';\nmpc.baseMVA = 100;\n{body}{}",
        tables("mpc")
    )
}

fn parse_message(result: Result<Case, PowerflowError>) -> (usize, String) {
    match result {
        Err(PowerflowError::Parse { line, message }) => (line, message),
        other => panic!("expected a Parse error, got {other:?}"),
    }
}

#[test]
fn several_statements_on_a_line() {
    let text = format!(
        "function mpc = syntax\nmpc.version = '2'; mpc.baseMVA = 100,\n{}",
        tables("mpc")
    );
    let case = Case::parse(&text).unwrap();
    assert_eq!(case.base_mva, 100.0);

    // A skipped field followed by a used one on the same line
    let text = case_with("mpc.note = 'made by hand'; mpc.bus_name = { 'a'; 'b' };\n");
    let case = Case::parse(&text).unwrap();
    assert_eq!(case.busnames.unwrap(), vec!["a", "b"]);
}

#[test]
fn double_quoted_strings() {
    let text = format!(
        "function mpc = syntax\nmpc.version = \"2\";\nmpc.baseMVA = 100;\n{}\
         mpc.bus_name = {{ \"Bus \"\"1\"\"\"; 'it''s 50% done' }};\n",
        tables("mpc")
    );
    let case = Case::parse(&text).unwrap();
    assert_eq!(case.busnames.unwrap(), vec!["Bus \"1\"", "it's 50% done"]);
}

#[test]
fn block_and_octave_comments() {
    let body = "%{\n\
                Anything at all, even mpc.bus(:, 3) = 0; or unbalanced [ brackets\n\
                %{\n\
                nested blocks are allowed\n\
                %}\n\
                still inside the outer block\n\
                %}\n\
                #{\n\
                an Octave block comment\n\
                #}\n\
                # an Octave line comment\n\
                %{ is only a block marker when alone on its line, so this is a line comment\n\
                mpc.note = 1; # trailing Octave comment\n";
    assert!(Case::parse(&case_with(body)).is_ok());
}

#[test]
fn other_struct_names() {
    // MATPOWER takes whatever the function returns
    let text = format!(
        "function s = syntax\ns.version = '2';\ns.baseMVA = 100;\n{}",
        tables("s")
    );
    assert_eq!(Case::parse(&text).unwrap().bus.len(), 2);

    let bracketed = format!(
        "function [out] = syntax\nout.version = '2';\nout.baseMVA = 100;\n{}",
        tables("out")
    );
    assert_eq!(Case::parse(&bracketed).unwrap().bus.len(), 2);

    // Without a function line the struct must be `mpc`
    let script = format!("mpc.version = '2';\nmpc.baseMVA = 100;\n{}", tables("mpc"));
    assert_eq!(Case::parse(&script).unwrap().bus.len(), 2);

    // Fields of a different variable than the one returned aren't part of the case
    let wrong = format!(
        "function s = syntax\nmpc.version = '2';\nmpc.baseMVA = 100;\n{}",
        tables("mpc")
    );
    assert!(parse_message(Case::parse(&wrong))
        .1
        .contains("unsupported statement"));

    let no_output = case_with("").replace("function mpc = syntax", "function syntax");
    assert!(parse_message(Case::parse(&no_output))
        .1
        .contains("must return the case struct"));
}

#[test]
fn unused_fields_are_skipped_whatever_they_hold() {
    // Assignments to fields the model doesn't read can't change the ones it does (MATLAB has
    //   value semantics), so their right-hand sides needn't be understood
    let body = "mpc.if.map = [ 1 -1; 2 3 ];\n\
                mpc.if.lims = [ 1 -100 100 ];\n\
                mpc.note = sprintf('made %d, 100%% by hand', 2026);\n\
                mpc.transposed = [1 2]'; % a comment with 'quotes'\n\
                mpc.nested = {1, 'a', [2 3]', {\"x\"}};\n\
                mpc.multiline = [ 1 2\n    3 4 ];\n\
                mpc.continued = 1 + ...\n    2;\n\
                mpc.call = max(mpc.baseMVA, 2);\n";
    let case = Case::parse(&case_with(body)).unwrap();
    assert_eq!((case.bus.len(), case.base_mva), (2, 100.0));
}

#[test]
fn unused_field_syntax_errors_are_still_errors() {
    let (line, message) = parse_message(Case::parse(&case_with("mpc.x = [1 2));\n")));
    assert_eq!(line, 4);
    assert!(message.contains("closing bracket"), "{message}");

    let (line, message) = parse_message(Case::parse(&case_with("mpc.x = ones(2,\n")));
    assert_eq!(line, 4);
    assert!(message.contains("never close"), "{message}");

    let (_, message) = parse_message(Case::parse(&case_with("mpc.x = 'unclosed;\n")));
    assert!(message.contains("never closed"), "{message}");
}

#[test]
fn used_fields_must_still_be_literals() {
    // A transposed matrix would be read untransposed, so it's rejected
    let text = case_with("").replace(
        "mpc.gen = [ 1 0 0 1 -1 1 100 1 200 0 ];",
        "mpc.gen = [ 1 0 0 1 -1 1 100 1 200 0 ]';",
    );
    assert!(parse_message(Case::parse(&text))
        .1
        .contains("unexpected `';`"));

    // Nested fields of a used one, and indexed assignments, are code
    let nested = case_with("mpc.bus.extra = 1;\n");
    assert!(parse_message(Case::parse(&nested))
        .1
        .contains("unsupported statement"));
    let indexed = format!("{}mpc.bus(:, 3) = 0;\n", case_with(""));
    assert!(parse_message(Case::parse(&indexed))
        .1
        .contains("compute its data with code"));
}

#[test]
fn version_1_files_are_named_as_such() {
    let version_1 = Err(PowerflowError::UnsupportedVersion {
        found: "1".to_string(),
    });
    let signature = "function [baseMVA, bus, gen, branch, areas, gencost] = old\n\
                     baseMVA = 100;\n";
    assert_eq!(Case::parse(signature), version_1);
    // A version 1 script, without the function line
    assert_eq!(
        Case::parse("baseMVA = 100;\nbus = [ 1 3 0 0 0 0 1 1 0 135 1 1.05 0.95 ];\n"),
        version_1
    );
    // But a comparison isn't an assignment
    assert!(parse_message(Case::parse("bus == 1;\n"))
        .1
        .contains("unsupported statement"));
}
