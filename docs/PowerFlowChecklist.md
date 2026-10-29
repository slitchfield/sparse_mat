# Power flow checklist

The roadmap for using `sparse_mat` as the numerical base of a DC and AC power flow library. Each
unchecked item is sized to be one commit. Items marked **Decide** need an answer before the work
that depends on them.

## Review notes: changes from the original plan

The plan was first written in conversation on 2026-10-28. Reviewing it against the code after
Phase 0 merged led to these changes:

1. **`num-complex` becomes a dependency in Phase 1, not Phase 4.** The complex helpers (`conj`,
   `real`, `imag`) need it, and so does the solver generalization, because `ComplexFloat` lives
   in `num_complex`, not in `num-traits`. That location was a correction from the original plan.
2. **The new LU is generic over `ComplexFloat` from the start.** Writing it for `Float` in
   Phase 3 and generalizing it in Phase 4 would mean writing the numeric core twice. Phase 4 now
   only migrates the existing solvers.
3. **Complex Cholesky and CG need Hermitian, not symmetric, semantics.** Complex Cholesky factors
   `A = L Lᴴ`, so the symmetry check must become a Hermitian check. CG's inner products need a
   conjugate. The original plan said only "rerun the tests for `Complex<f64>`".
4. **A DC power flow prototype moves up to right after Phase 1.** It only needs `select` (to drop
   the slack bus) and the existing `cholesky`. Building it early creates the case reader, the
   MATPOWER reference results and the benchmark harness before the LU rewrite, so Phase 3 is
   measured against real cases instead of synthetic ones.
5. **Open items from the PR #1 pre-push review are tracked here:** parser fuzzing, `cargo audit`,
   package metadata and the leftover template `add` function in `lib.rs`.

## Phase 0: Hardening

Merged in PR #2.

- [x] `from_csr` rejects `rows == usize::MAX` instead of overflowing.
- [x] Compressed-storage allocation guarded against huge dimensions (clear panic, and half the
      pointer memory).
- [x] The Matrix Market reader caps pre-allocation from the untrusted size line.
- [x] `check.sh` lints and tests each optional feature on its own.
- [x] MSRV declared as 1.87 (measured), with the MSRV-aware resolver and a CI job.

## Phase 1: Cheap primitives

- [x] **Decide:** whether `num-complex` is always a dependency or behind a `complex` feature.
      Decided: always a dependency, re-exported as `sparse_mat::num_complex` and
      `sparse_mat::Complex`.
- [x] `scale_rows(&[T])` and `scale_cols(&[T])` (diag(v)·A and A·diag(v)), with
      `ShapeMismatch` on a wrong length. Test against `from_diagonal(v) * A`.
- [x] `select(rows: &[usize], cols: &[usize])` for non-contiguous index sets. Rows and columns
      come out in the order given; `OutOfRange` on a bad index; repeated indices are allowed and
      repeat the row or column (decided and documented). Test that `select` with ranges matches
      `submatrix`.
- [x] `permute(row_perm, col_perm)` and `permute_symmetric(perm)`, validating that each is a
      permutation. Test that a permutation followed by its inverse is the identity.
- [x] Complex helpers: `conj()` (identity for real types via `ComplexFloat`), `adjoint()`,
      `real()`, `imag()`, and `from_parts(re, im)`. Test that `from_parts(a.real(), a.imag())`
      gives back `a`.

## Phase 1.5: DC power flow prototype (power flow crate)

- [ ] **Decide:** a Cargo workspace in this repo (`sparse_mat/` and `powerflow/`) or a separate
      repo. Recommended: a workspace, so one CI run covers both. Converting means moving the
      crate into a subdirectory; `check.sh` already uses `--workspace`.
- [ ] **Decide:** confirm the MATPOWER license terms before vendoring its case files as test data.
- [ ] Case model: buses (type, demand, shunts, voltage limits), branches (r, x, b, tap, shift,
      status) and generators, in per unit.
- [ ] MATPOWER case reader for the `mpc.bus`, `mpc.branch`, `mpc.gen` and `mpc.baseMVA` matrices
      in `.m` files.
- [ ] Bbus assembly with `from_triplets` (summing branch stamps), honoring branch status and
      phase shifters.
- [ ] DC power flow: drop the slack row and column with `select`, solve B′θ = P with
      `cholesky`, then compute branch flows.
- [ ] Reference results: MATPOWER `rundcpf` outputs for case14, case30 and case118 checked into
      test data. Tests compare angles and flows within a stated tolerance.
- [ ] Benchmark harness (criterion) running DC power flow on the largest available case, so
      later phases have a baseline.

## Phase 2: Fixed-pattern `CsrMatrix`

- [ ] **Decide:** whether `CsrMatrix` becomes the main type for numerical work, with
      `SparseMatrix` kept for editing and assembly. Recommended: yes.
- [ ] Promote the internal `Compressed<T>` to a public `CsrMatrix<T>` with an immutable pattern
      and `values()`/`values_mut()`. Build it from `SparseMatrix` and from validated raw arrays
      (reusing `from_csr`'s checks), and convert back.
- [ ] Zero-copy row access (`row(i)` returning column and value slices), `position(i, j)` by
      binary search, and cached diagonal positions.
- [ ] `mul_vec` on `CsrMatrix`, reusing `mul_vec_csr`, plus `par_mul_vec` under `rayon`.
- [ ] An assembly map built once from a triplet list's positions that records each triplet's
      slot in `values`, so a refill with new values is an O(nnz) write with no hashing. Test that
      a refill matches a fresh `from_triplets`.
- [ ] Benchmark: refilling Ybus and Jacobian-sized patterns through the assembly map versus
      `from_triplets`.

## Phase 3: Factorization rewrite (the largest piece)

- [ ] Reverse Cuthill–McKee ordering, with a test that bandwidth doesn't increase on banded and
      grid matrices.
- [ ] Symbolic analysis, `analyze(pattern)`: ordering, plus the elimination tree for Cholesky.
- [ ] Gilbert–Peierls left-looking LU with threshold partial pivoting, generic over
      `ComplexFloat` from the start (review note 2). Keep the current `Lu` API (`solve`,
      `determinant`, `l`, `u`, `permutation`) as the front end.
- [ ] `refactor(values)`, which reuses the analysis and pivot sequence when only values change.
      If a pivot becomes too small, fall back to a full factorization and report that.
- [ ] Keep the current `BTreeMap` LU as a test-only reference implementation, and compare the
      two on pseudo-random and power flow matrices.
- [ ] `solve_in_place` and `solve_many` (several right-hand sides).
- [ ] AMD ordering for Cholesky, and COLAMD (or AMD on the pattern of AᵀA) for LU. Benchmark fill
      and time against RCM on the large case.
- [ ] Cholesky after ordering, reusing the existing up-looking algorithm.

## Phase 4: Complex-capable solvers (migrating what Phase 3 didn't rewrite)

- [ ] Change the bounds of the triangular solves, `solve` and the norms from `Float` to
      `Scalar + ComplexFloat`. Pivot choice compares `abs()` (a real value), and norms return
      `Real`.
- [ ] Hermitian Cholesky (`A = L Lᴴ`): replace the symmetry check with a Hermitian check, and
      take real square roots of the diagonal (review note 3).
- [ ] Hermitian conjugate gradient: use conjugated inner products.
- [ ] Run every solver test for both `f64` and `Complex<f64>`.
- [ ] Note in the docs that user-defined `Float` types lose solver support, since `ComplexFloat`
      is sealed.

## Phase 4.5: AC power flow (power flow crate)

- [ ] Ybus assembly with `from_triplets` (complex branch π-models, taps, shifts, shunts), then
      converted to `CsrMatrix` with an assembly map for re-stamping.
- [ ] Newton–Raphson in polar form: mismatch, then the dS/dV Jacobian built with
      `scale_rows`/`scale_cols`, `conj` and `select` over the PV/PQ index sets, then a solve
      using `refactor` from the second iteration on.
- [ ] Convergence control: mismatch tolerance, iteration limit, and reported iteration count and
      final mismatch.
- [ ] Fast-decoupled power flow (XB and BX variants), which factors B′ and B″ once.
- [ ] Validation against MATPOWER `runpf` for case14, case30, case118 and a large case such as
      case2383wp: voltage magnitudes, angles and branch flows within a stated tolerance.
- [ ] Benchmarks against the Phase 1.5 baseline.

## Phase 5: DC studies

- [ ] Sparse right-hand-side solves, using the Gilbert–Peierls reach to touch only the needed
      part of the factor.
- [ ] PTDF and LODF computation using `solve_many` or sparse right-hand sides.
- [ ] Rank-k updates (Woodbury) for N-1 contingencies without refactoring. Test against a full
      refactorization.
- [ ] Island detection: connected components on the pattern, with an error naming the islands
      instead of `SingularMatrix`.
- [ ] Iterative refinement and a condition estimate.
- [ ] Parallel batch solves over a shared factorization (`rayon`). Check that `Lu` is
      `Send + Sync` and that results are deterministic.

## Housekeeping (carried over from the PR #1 review)

- [ ] Fuzz the Matrix Market parser with truncated, malformed and oversized input, and confirm
      it always errors instead of panicking.
- [ ] Run `cargo audit` or `cargo deny` over the dependency tree, and consider adding it to CI.
- [ ] Add package metadata (license, description, repository) and check `cargo package --list`.
      There is no license file yet.
- [ ] Remove the leftover `add(left, right)` template function from `src/lib.rs`.
- [ ] Decide whether to doctest the README by including it as crate docs
      (`#![doc = include_str!("../README.md")]`), so its example can't drift.
