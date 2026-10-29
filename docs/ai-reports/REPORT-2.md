# Implementation report: `dev/claude/phase1`

This branch is Phase 1 of the power flow plan (`docs/PowerFlowChecklist.md`). It adds the
cheap primitives that Jacobian assembly and bus partitioning need, and makes `num-complex` a
regular dependency, as decided. It has six commits, one per checklist item plus this report, on
top of `main` at `af78a64`. All the code changes are additions; no existing behaviour changes.

| | `main` | `dev/claude/phase1` |
|---|---|---|
| Phase 1 checklist items done | 0 of 5 | 5 of 5 |
| Tests (all features) | 164 (154 core, 6 `rayon`, 4 `serde`) | 186 (176 core, 6 `rayon`, 4 `serde`) |
| Required dependencies | `num-traits` | `num-traits`, `num-complex` |
| Diff | | 11 files, +575 / −15, about half of it tests |

## Commits

| Commit | Change |
|---|---|
| `5693a90` | `num-complex` becomes a regular dependency and is re-exported |
| `7e67c84` | `scale_rows` and `scale_cols` |
| `1b4a0cc` | `select` for non-contiguous index lists |
| `a5414be` | `permute` and `permute_symmetric`, plus the `InvalidPermutation` error |
| `69e62ce` | Complex helpers: `conj`, `adjoint`, `real`, `imag` and `from_parts` |

Each commit also updates the README's feature list and ticks its item in the checklist.

## Details

### `num-complex` as a dependency (`5693a90`)
- Moved from `[dev-dependencies]` to `[dependencies]`. It provides `ComplexFloat`, which the
  complex helpers use now and the Phase 4 solvers will use later.
- Re-exported as `sparse_mat::num_complex` and `sparse_mat::Complex`, so callers get the same
  version as this crate without adding their own dependency.
- Records the decision in the checklist.

### `scale_rows` and `scale_cols` (`7e67c84`, `ops.rs`)
- `scale_rows(&v)` returns diag(v)·A and `scale_cols(&v)` returns A·diag(v). Each is one pass
  over the stored entries, without forming the diagonal matrix, and products that are exactly
  zero are dropped.
- A wrong length returns `ShapeMismatch`, with `v` reported as the square diagonal matrix it
  stands for, for example `left: (3, 3), right: (2, 3)` for `scale_rows`.
- **Tests:** exact equality against `from_diagonal(v).matmul(A)` and `A.matmul(from_diagonal(v))`
  (each entry is a single multiplication, so the results match bit for bit), zero factors,
  complex factors, the error cases and an empty matrix.

### `select` (`1b4a0cc`, `access.rs`)
- `select(&rows, &cols)` returns the matrix whose entry `(i, j)` is `(rows[i], cols[j])` of the
  original. Indices can be non-contiguous and in any order, which is what pulling the PV/PQ
  partitions out of a Jacobian needs.
- **Decided:** repeated indices are allowed and repeat that row or column, as multiplying by a
  selection matrix would. The checklist left this open; it is now documented on the method.
- **Implementation:** one pass over the stored entries, with maps from each source index to its
  output positions. It never builds a CSR form, so a one-off selection on a stale matrix costs
  O(nnz).
- An out-of-range index returns `OutOfRange` for the first bad one.
- **Tests:** matches `submatrix` for contiguous ranges, matches a dense selection for scrambled
  index sets, reordering, repeated indices, empty selections and both error cases.

### `permute` and `permute_symmetric` (`a5414be`, new `reorder.rs`)
- `permute(&row_perm, &col_perm)` computes P A Qᵀ in O(nnz + rows + cols).
  `permute_symmetric(&perm)` computes P A Pᵀ, the form fill-reducing orderings need. It returns
  `NotSquare` for a non-square matrix.
- **Convention:** gather, so row `i` of the result is row `perm[i]` of the input. That's the
  same as `select` and `Lu::permutation`, so the three compose.
- **New error:** `SparseError::InvalidPermutation { reason }` covers a wrong length, an index out
  of range and a repeated index.
- **Tests:**
  - equality with `select`
  - permuting and then applying the inverse gives back the original
  - the identity permutation changes nothing
  - `permute_symmetric` preserves symmetry, the trace and the multiset of diagonal values
  - equality with the explicit product `P·A·Pᵀ`, which confirms the doc's convention
  - permuting rows by `Lu::permutation` reproduces `L·U`
  - every error reason
  - empty matrices

### Complex helpers (`69e62ce`, new `complex.rs`)
- **`conj`, `adjoint`, `real`, `imag`:** generic over `ComplexFloat` (`f32`, `f64` and their
  complex types). `adjoint` builds the conjugate transpose in one pass instead of transposing a
  conjugated copy. `real` and `imag` return matrices of the underlying real type and drop parts
  that are zero, so a purely imaginary entry is absent from `real()`. On real matrices, `conj`
  is a copy, `adjoint` is the transpose and `imag` is empty, so later code can share one path
  for real and complex.
- **`from_parts(&re, &im)`:** builds a `SparseMatrix<Complex<R>>` for any `Scalar` `R`, as the
  inverse of `real` and `imag`. Unequal shapes return `ShapeMismatch`.
- **Tests:**
  - `from_parts(a.real(), a.imag()) == a`
  - zero parts are dropped
  - conjugating twice gives back the original, and likewise for the adjoint
  - (AB)ᴴ = BᴴAᴴ on a rectangular pair
  - a Hermitian matrix equals its adjoint and has a real diagonal
  - real-matrix behaviour for `f64` and `f32`
  - the shape error

## Verification

- `./check.sh` passed after every commit. It covers fmt, clippy with `-D warnings` (all features
  and each feature alone), and tests with no features, all features and each feature alone,
  plus doc tests.
- **MSRV:** all 186 tests pass on Rust 1.87 with all features. No new API raised the minimum.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` is clean.
- The README quick-start example still compiles and runs.
- **CI:** not run yet. The branch is pushed, but no PR has been opened.

## Notes and follow-ups

- The checklist's Phase 1 is complete. Next is Phase 1.5, the DC power flow prototype, which
  now has what it needs from this crate: `select` to drop the slack bus, and the existing
  `cholesky`. It starts with two **Decide** items: workspace or separate repo, and the MATPOWER
  license terms.
- `select` and `permute` overlap on purpose: `permute` adds validation and a cheaper inverse-map
  implementation. Phase 3's orderings will produce permutations for `permute_symmetric`.
- There is no `is_hermitian` yet. Phase 4's Hermitian Cholesky will need one, and
  `a.adjoint() == a` works in the meantime.
- **API surface:** the new methods are inherent methods on `SparseMatrix`. The fixed-pattern
  `CsrMatrix` in Phase 2 will need its own row and column scaling for Jacobian refills, so
  `scale_rows` and `scale_cols` here are the assembly-time versions.
