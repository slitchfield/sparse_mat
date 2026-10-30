//! Small hand-built cases for the DC power flow tests. Each table lists only the columns DC
//! power flow reads; the rest are filled with neutral values, and the result is parsed from
//! MATPOWER text, so fixtures go through the same reader as real cases.

use powerflow::case::Case;

/// A bus: label, MATPOWER type code (1 PQ, 2 PV, 3 REF, 4 isolated), Pd (MW), Gs (MW at
/// 1 p.u.) and Va (degrees).
pub struct B(pub usize, pub u8, pub f64, pub f64, pub f64);

/// A generator: bus label, Pg (MW), in service.
pub struct G(pub usize, pub f64, pub bool);

/// A branch: from label, to label, x (p.u.), tap ratio (0 for a line), shift (degrees), in
/// service.
pub struct L(pub usize, pub usize, pub f64, pub f64, pub f64, pub bool);

/// A case on a 100 MVA base from the given tables. Panics if it doesn't parse, since a fixture
/// that doesn't is a bug in the test.
pub fn fixture(buses: &[B], gens: &[G], branches: &[L]) -> Case {
    let mut text = String::from("function mpc = fixture\nmpc.version = '2';\nmpc.baseMVA = 100;\n");
    text.push_str("mpc.bus = [\n");
    for B(label, kind, pd, gs, va) in buses {
        text.push_str(&format!(
            "{label} {kind} {pd:?} 0 {gs:?} 0 1 1 {va:?} 135 1 1.1 0.9;\n"
        ));
    }
    text.push_str("];\nmpc.gen = [\n");
    for G(bus, pg, status) in gens {
        let status = u8::from(*status);
        text.push_str(&format!(
            "{bus} {pg:?} 0 100 -100 1 100 {status} 1000 0 0 0 0 0 0 0 0 0 0 0 0;\n"
        ));
    }
    text.push_str("];\nmpc.branch = [\n");
    for L(from, to, x, tap, shift, status) in branches {
        let status = u8::from(*status);
        text.push_str(&format!(
            "{from} {to} 0 {x:?} 0 0 0 0 {tap:?} {shift:?} {status} -360 360;\n"
        ));
    }
    text.push_str("];\n");
    Case::parse(&text).unwrap_or_else(|err| panic!("fixture doesn't parse: {err}\n{text}"))
}
