# Todo

Missing functionality, compared with mature sparse libraries (SciPy `sparse`, Eigen, `sprs`, `nalgebra-sparse`).

## Core arithmetic (the biggest gap)
- [ ] **Multiplication:** matrix × vector (the main reason to have CSR), matrix × dense matrix, and sparse × sparse (SpGEMM).
- [ ] **Scalar and other operators:** `Mul<f64>`, `Sub`, `Neg`, and `AddAssign`/`SubAssign`. `Add` is only implemented for `&SparseMatrix`, so owned operands don't work.
- [ ] **Element-wise ops:** Hadamard product, and `map`/`apply` over stored values.

## Construction and conversion
- [ ] **Bulk constructors:**
    - [x] `from_triplets` that sums duplicates (COO semantics, where `insert_triplets` silently overwrites them)
    - [ ] `from_dense`
    - [ ] `from_csr`
    - [ ] `diagonal(vec)`
    - [ ] `to_dense`
- [x] **`empty_with_shape` over-reserves:** it reserves `n * m / 4` entries, which can overflow or attempt a huge allocation for large sparse shapes.
- [x] **Explicit zeros:** nothing stops `insert(r, c, 0.0)` or `Add` from storing zeros. There's no `prune()` or `drop_zeros()`, and `num_nonzero` over-counts as a result.
- [ ] **Result-based errors:** the TODOs already note that the `assert!`s should become `Result`.
    - [x] **OOB Errors:** Translate `assert!`s into results for OOB errors
    - [ ] **Shape Mismatch Errors:** The same applies to shape mismatches in `Add`.

## Access and iteration
- [ ] **Stored-entry iterator:** an iterator over `(row, col, val)` triplets. Today `row_iter` only yields dense rows, which defeats the purpose for large matrices.
- [ ] **Sparse rows and columns:** `row(i)`/`col(j)` slices, a column iterator, `diagonal()`, and submatrix extraction.
- [ ] **Indexing:** `Index<(u64, u64)>` (`peek_at` returns an `Option` instead).

## Standard traits
- [ ] `Debug` (only `Clone` is derived), `PartialEq`, `IntoIterator`, and `From` conversions.

## Storage
- [ ] **Real compressed storage:** the file's header comment already plans a move to CSR and CSC. CSC would make column access and `Aᵀ·x` cheap.
- [ ] **Dimension type:** `u64` dimensions are cast to `usize` throughout, and `empty_with_shape` computes `n * m / 4`, which can overflow. `usize` would be more idiomatic.

## Properties and reductions
- [ ] `trace`, `sum`, norms (Frobenius, 1, ∞), `density`, `is_symmetric`, and `is_square`.

## Further out
- [ ] **Generic scalar type:** `f32`, integers and complex numbers, via `num-traits`.
- [ ] **Block operations:** `kron`, `hstack` and `vstack`.
- [ ] **Solvers:** triangular solves, then LU/Cholesky and iterative methods like CG.
- [ ] **File I/O:** Matrix Market (`.mtx`) read and write.
- [ ] **Optional features:** `serde` and `rayon`.

## Suggested order
The matrix-vector product, then the triplet iterator, then `from_triplets` with duplicate summing, then `Sub` and `Mul<f64>`. That set makes the type usable for real work and builds on the CSR path already in place. The `Result` error handling is worth doing early too, since it changes every public signature.
