//! DC power flow, computed exactly as MATPOWER's `rundcpf` does.
//!
//! The DC approximation keeps only real power and voltage angles: every voltage magnitude is
//! 1 p.u., resistances and line charging are ignored, and angle differences are small enough
//! that sin θ ≈ θ. Branch flows are then linear in the bus angles, `P = B θ`, and one sparse
//! solve gives the angles.
//!
//! [`DcModel::new`] builds the matrices once, following MATPOWER's `ext2int`, `bustypes` and
//! `makeBdc`.
//!
//! Elements are dropped from the model as MATPOWER's `ext2int` drops them: isolated buses
//! (type 4), out-of-service generators and branches, and generators and branches connected to
//! an isolated bus.

use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

use sparse_mat::sparse_matrix::SparseMatrix;

use crate::case::{BusType, Case};
use crate::powerflow_error::PowerflowError;

/// The DC model of a case: the B matrices and phase-shifter injections in per unit, over the
/// connected buses and in-service branches, with each bus's role in the solve.
///
/// Internal bus `i` is case bus row `bus_rows()[i]`, and internal branch `k` is case branch row
/// `branch_rows()[k]`; both keep the case's order.
#[derive(Debug, Clone)]
pub struct DcModel {
    base_mva: f64,
    bbus: SparseMatrix<f64>,
    bf: SparseMatrix<f64>,
    pbusinj_pu: Vec<f64>,
    pfinj_pu: Vec<f64>,
    bus_rows: Vec<usize>,
    branch_rows: Vec<usize>,
    gen_rows: Vec<usize>,
    ref_buses: Vec<usize>,
    pv_buses: Vec<usize>,
    pq_buses: Vec<usize>,
}

impl DcModel {
    /// Builds the DC model of `case`, as MATPOWER's `makeBdc` does after `ext2int`.
    ///
    /// Each in-service branch has series susceptance `b = 1 / (x · tap)`, using
    /// [`Branch::effective_tap`](crate::case::Branch::effective_tap), and stamps
    /// `Bbus = Cftᵀ · diag(b) · Cft` and `Bf = diag(b) · Cft`, where `Cft` has +1 at the
    /// branch's from bus and −1 at its to bus. Parallel branches sum. A phase shift of `φ`
    /// degrees adds the injection `−b · φ · π/180` at the from end (`pfinj_pu`), and its
    /// effect on each bus (`pbusinj_pu = Cftᵀ · pfinj_pu`).
    ///
    /// Bus roles follow MATPOWER's `bustypes`: a REF or PV bus counts as one only if it has an
    /// in-service generator, and is otherwise treated as PQ. With no REF bus, the first PV bus
    /// becomes the reference. A case may have several reference buses.
    ///
    /// Returns `PowerflowError::ZeroReactance` for an in-service branch with `x == 0`,
    /// `NoReferenceBus` if no bus can be the reference, and `IslandWithoutReference` if some
    /// connected bus can't reach a reference bus through in-service branches, which would make
    /// the solve singular.
    pub fn new(case: &Case) -> Result<DcModel, PowerflowError> {
        // ext2int: connected buses, in case order, and the label → internal index map
        let bus_rows: Vec<usize> = (0..case.bus.len())
            .filter(|&row| case.bus[row].bus_type != BusType::Isolated)
            .collect();
        let mut internal = HashMap::with_capacity(bus_rows.len());
        for (index, &row) in bus_rows.iter().enumerate() {
            internal.insert(case.bus[row].bus_label, index);
        }
        let nb = bus_rows.len();

        // Generators and branches count only when in service and fully connected
        let (mut gen_rows, mut gen_buses) = (vec![], vec![]);
        for (row, generator) in case.generator.iter().enumerate() {
            if let (true, Some(&bus)) = (generator.status, internal.get(&generator.bus_label)) {
                gen_rows.push(row);
                gen_buses.push(bus);
            }
        }
        let mut branch_rows = vec![];
        let mut ends = vec![];
        for (row, branch) in case.branch.iter().enumerate() {
            let from = internal.get(&branch.from_label);
            let to = internal.get(&branch.to_label);
            if let (true, Some(&from), Some(&to)) = (branch.status, from, to) {
                if branch.x == 0.0 {
                    return Err(PowerflowError::ZeroReactance { row: row + 1 });
                }
                branch_rows.push(row);
                ends.push((from, to));
            }
        }

        // makeBdc. Each branch's (from, to) and (to, from) stamps are pushed together, so both
        //   positions sum the same values in the same order and Bbus is exactly symmetric,
        //   which cholesky requires.
        let nl = branch_rows.len();
        let mut bbus_triplets = Vec::with_capacity(4 * nl);
        let mut bf_triplets = Vec::with_capacity(2 * nl);
        let mut pfinj_pu = Vec::with_capacity(nl);
        let mut pbusinj_pu = vec![0.0; nb];
        for (k, (&row, &(from, to))) in branch_rows.iter().zip(&ends).enumerate() {
            let branch = &case.branch[row];
            let b = 1.0 / branch.x / branch.effective_tap();
            bbus_triplets.extend([(from, from, b), (from, to, -b), (to, from, -b), (to, to, b)]);
            bf_triplets.extend([(k, from, b), (k, to, -b)]);
            let pfinj = b * (-branch.shift_deg * PI / 180.0);
            pfinj_pu.push(pfinj);
            pbusinj_pu[from] += pfinj;
            pbusinj_pu[to] -= pfinj;
        }
        let bbus = SparseMatrix::from_triplets(nb, nb, bbus_triplets)
            .expect("internal bus indices are below nb");
        let bf = SparseMatrix::from_triplets(nl, nb, bf_triplets)
            .expect("internal branch and bus indices are in range");

        // bustypes
        let mut has_gen = vec![false; nb];
        for &bus in &gen_buses {
            has_gen[bus] = true;
        }
        let (mut ref_buses, mut pv_buses, mut pq_buses) = (vec![], vec![], vec![]);
        for (index, &row) in bus_rows.iter().enumerate() {
            match case.bus[row].bus_type {
                BusType::Ref if has_gen[index] => ref_buses.push(index),
                BusType::PV if has_gen[index] => pv_buses.push(index),
                _ => pq_buses.push(index),
            }
        }
        if ref_buses.is_empty() {
            if pv_buses.is_empty() {
                return Err(PowerflowError::NoReferenceBus);
            }
            ref_buses.push(pv_buses.remove(0));
        }

        // Every connected bus must reach a reference bus, or B′ is singular: an island with
        //   no reference has no fixed angle. Checked here rather than inferred from a failed
        //   factorization, which can't name the island and may not fail at all, since
        //   rounding can leave a tiny positive pivot.
        let mut neighbours = vec![vec![]; nb];
        for &(from, to) in &ends {
            neighbours[from].push(to);
            neighbours[to].push(from);
        }
        let mut reached = vec![false; nb];
        let mut queue: VecDeque<usize> = ref_buses.iter().copied().collect();
        for &bus in &ref_buses {
            reached[bus] = true;
        }
        while let Some(bus) = queue.pop_front() {
            for &next in &neighbours[bus] {
                if !reached[next] {
                    reached[next] = true;
                    queue.push_back(next);
                }
            }
        }
        if let Some(stranded) = reached.iter().position(|&reached| !reached) {
            return Err(PowerflowError::IslandWithoutReference {
                bus_label: case.bus[bus_rows[stranded]].bus_label,
            });
        }

        Ok(DcModel {
            base_mva: case.base_mva,
            bbus,
            bf,
            pbusinj_pu,
            pfinj_pu,
            bus_rows,
            branch_rows,
            gen_rows,
            ref_buses,
            pv_buses,
            pq_buses,
        })
    }

    /// The system MVA base the per-unit values are relative to.
    pub fn base_mva(&self) -> f64 {
        self.base_mva
    }

    /// The bus susceptance matrix, connected buses × connected buses (p.u.).
    pub fn bbus(&self) -> &SparseMatrix<f64> {
        &self.bbus
    }

    /// The branch susceptance matrix, in-service branches × connected buses (p.u.), which maps
    /// bus angles to from-end branch flows.
    pub fn bf(&self) -> &SparseMatrix<f64> {
        &self.bf
    }

    /// The phase shifters' equivalent injection at each connected bus (p.u.).
    pub fn pbusinj_pu(&self) -> &[f64] {
        &self.pbusinj_pu
    }

    /// The phase shifters' contribution to each in-service branch's from-end flow (p.u.).
    pub fn pfinj_pu(&self) -> &[f64] {
        &self.pfinj_pu
    }

    /// The case bus row of each internal bus.
    pub fn bus_rows(&self) -> &[usize] {
        &self.bus_rows
    }

    /// The case branch row of each internal branch.
    pub fn branch_rows(&self) -> &[usize] {
        &self.branch_rows
    }

    /// The case generator rows of the in-service generators at connected buses.
    pub fn gen_rows(&self) -> &[usize] {
        &self.gen_rows
    }

    /// The reference buses (internal indices), in case order.
    pub fn ref_buses(&self) -> &[usize] {
        &self.ref_buses
    }

    /// The PV buses (internal indices), in case order.
    pub fn pv_buses(&self) -> &[usize] {
        &self.pv_buses
    }

    /// The PQ buses (internal indices), in case order. Includes REF and PV buses that have no
    /// in-service generator.
    pub fn pq_buses(&self) -> &[usize] {
        &self.pq_buses
    }
}
