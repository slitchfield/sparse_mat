# sparse_mat

A sparse matrix library for Rust: construction, arithmetic, products, factorizations and
iterative solvers, Matrix Market I/O, and optional `serde` and `rayon` support. It is generic
over the element type (`f64` by default) and has two required dependencies, `num-traits` and `num-complex`.

It is being developed as the numerical foundation for a DC and AC power flow library; see
[`docs/PowerFlowChecklist.md`](docs/PowerFlowChecklist.md) for that roadmap.

## Quick start

```toml
[dependencies]
sparse_mat = { git = "https://github.com/slitchfield/sparse_mat" }
```

```rust
use sparse_mat::sparse_error::SparseError;
use sparse_mat::sparse_matrix::SparseMatrix;

fn main() -> Result<(), SparseError> {
    // Assemble from (row, col, value) triplets. from_triplets has COO semantics: entries
    // that share a position are summed, which is how stamped/assembled matrices build up.
    let a = SparseMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, 2.0), (0, 1, -1.0),
            (1, 0, -1.0), (1, 1, 1.0), (1, 1, 1.0), (1, 2, -1.0),
            (2, 1, -1.0), (2, 2, 2.0),
        ],
    )?;
    assert_eq!(a[(1, 1)], 2.0); // the two (1, 1) entries were summed

    // Products and solves return Result; the operators panic on shape mismatches instead.
    let b: [f64; 3] = [1.0, 0.0, 1.0];
    let x = a.solve(&b)?; // LU with partial pivoting; x is [1, 1, 1] up to rounding
    let residual = a.mul_vec(&x)?;
    assert!(residual.iter().zip(&b).all(|(r, b)| (r - b).abs() < 1e-12));

    let l = a.cholesky()?; // a is symmetric positive definite, so a = L Lᵀ
    assert!((&(&l * &l.create_transpose()) - &a).norm_frobenius() < 1e-12);

    let squared = &a * &a; // sparse x sparse
    let shifted = &a + &SparseMatrix::identity(3);
    println!("{} stored entries, trace {}", squared.num_nonzero(), shifted.trace()?);
    Ok(())
}
```

## What it provides

**Construction and conversion**
- `new`, `empty_with_shape`, `identity`, `from_diagonal`.
- `from_triplets`, which sums duplicates (COO semantics).
- `from_dense`, `from_csr`, `From<[[T; C]; R]>` and `TryFrom<Vec<Vec<T>>>`.
- `to_dense`.
- Element edits with `insert`, `insert_triplets` (duplicates overwrite), `clear_at` and
  `prune_zeroes`.
- A zero is never stored, so `num_nonzero` counts real entries.

**Access and iteration**
- Checked reads with `peek_at`, and `matrix[(row, col)]` indexing, which panics out of bounds.
- `iter()` over stored `(row, col, value)` entries in row-major order. It costs O(nnz), not
  O(rows × cols).
- `row(i)` and `col(j)` as `SparseVector`s, plus `row_iter`/`col_iter` over dense rows and
  columns.
- `diagonal()`, `submatrix(rows, cols)` for contiguous ranges, and `select(&rows, &cols)` for
  arbitrary index lists in any order.
- `permute(&row_perm, &col_perm)` and `permute_symmetric(&perm)` (P A Pᵀ), using the same
  convention as `select`: row `i` of the result is row `perm[i]` of the input.

**Arithmetic**
- `+`, `-`, unary `-`, `+=`, `-=`, `*=` and scalar `*` (on either side for primitive types), for
  owned and borrowed operands.
- `try_add` and `try_sub` return `Result` instead of panicking.
- `scale`, `hadamard`, and `map`/`apply` over stored entries.
- `scale_rows(&v)` and `scale_cols(&v)` compute diag(v)·A and A·diag(v) without forming the
  diagonal matrix.

**Complex matrices**
- `conj()`, `adjoint()` (conjugate transpose), and `real()`/`imag()` for the parts. These are
  generic over `ComplexFloat`, so on `f32`/`f64` matrices `conj` is a copy, `adjoint` the
  transpose and `imag` empty.
- `SparseMatrix::from_parts(&re, &im)` builds a complex matrix and inverts `real`/`imag`.

**Products**
- Matrix × vector (`mul_vec`, `&a * &x`) and vector × matrix (`vec_mul`, `&x * &a`).
- Matrix × dense matrix (`mul_dense`) and sparse × sparse (`matmul`, `&a * &b`, using Gustavson's
  algorithm).
- `kron`, `hstack` and `vstack`.
- Sums walk sorted storage, so floating-point results are deterministic.

**Properties**
- `is_square`, `is_symmetric`, `density`, `sum` and `trace`.
- `norm_frobenius`, `norm_1` and `norm_inf` (float element types).

**Linear solvers** (float element types)
- `solve_lower_triangular` and `solve_upper_triangular`.
- `lu()` returns an `Lu` with `solve`, `determinant`, `l`, `u` and `permutation`; `solve` is the
  one-shot form.
- `cholesky()` and `solve_spd` for symmetric positive definite matrices.
- `conjugate_gradient(b, tolerance, max_iterations)`.

**I/O**
- Matrix Market coordinate files through `from_matrix_market`, `read_matrix_market`,
  `write_matrix_market` and `to_matrix_market`. Reading supports `real`, `integer` and
  `pattern` fields with `general`, `symmetric` or `skew-symmetric` symmetry.

**Standard traits**
- `Clone`, `Debug`, `PartialEq`, `Display` and `Default`.
- `IntoIterator` for owned and borrowed matrices.

## Element types

`SparseMatrix<T>` accepts any `Scalar`, meaning any `Copy + PartialEq + num_traits::Num`
type. That covers `f32`, `f64`, every integer type and `num_complex::Complex` (re-exported as
`sparse_mat::Complex`). Arithmetic, products and conversions work for all of them. Norms and
solvers need `num_traits::Float`, so today they cover `f32` and `f64` but not complex numbers.
Complex-capable solvers are on the roadmap.

Rust does not apply a type parameter's default in expressions, so a call with nothing to infer
`T` from needs it spelled out: `SparseMatrix::<f64>::new()`.

## Storage model and performance

Entries live in a hash map keyed by `(row, col)`, so single-element edits are O(1). Compressed
sparse row (CSR) and column (CSC) copies are built on demand and cached:

- `explicitly_compress()` builds the CSR cache and `explicitly_compress_csc()` the CSC cache.
  `compressed()` and `compressed_csc()` return views of them.
- Any mutation drops both caches.
- Operations that need compressed storage, such as `iter`, `row`, `col`, the products and the
  solvers, borrow a fresh cache when one exists. Otherwise they build a temporary copy. Results
  are always correct, but calling something like `row(i)` in a loop on a stale matrix rebuilds
  the CSR form every time, so compress first.
- A compressed copy needs one pointer per row or column however sparse the matrix is. Shapes too
  large for that still support element access, but operations needing compressed storage panic
  with a message saying so.

Known performance limits:
- `lu()` searches for pivots in O(n²) of the dimension, which suits systems up to thousands of
  rows, not hundreds of thousands.
- `cholesky()` does no fill-reducing ordering.
- `to_dense` and `mul_dense` allocate `rows × cols`.

## Errors

Fallible operations return `Result<_, SparseError>`. Its variants name the problem and carry
the offending values, for example `OutOfBounds { row, col, shape }`,
`ShapeMismatch { left, right }`, `SingularMatrix { pivot }` and `Parse { line, message }`. The
enum is `#[non_exhaustive]`.

Operators (`+`, `*`, `[]`, …) cannot return `Result`, so they panic on invalid input. Each has a
checked method alongside it (`try_add`, `matmul`, `mul_vec`, `peek_at`, …).

## Optional features

| Feature | Adds |
|---|---|
| `serde` | `Serialize`/`Deserialize` as `{"shape": [r, c], "entries": [[row, col, value], ...]}`, with entries in row-major order; deserializing validates like `from_triplets` |
| `rayon` | `par_mul_vec`, `par_vec_mul` and `par_matmul`, which return results bit-for-bit equal to the sequential versions |

## Requirements and development

- **MSRV:** Rust 1.87, set in `Cargo.toml` and checked by a CI job.
- **`./check.sh`:** runs everything CI runs: fmt, clippy with `-D warnings`, and the tests with no
  features, all features and each feature alone, plus doc tests.
- **Tests:** in `tests/`. The products, factorizations and Kronecker products are checked
  against dense reference implementations and algebraic identities.

## Documentation

- [`docs/BaseFunctionalityChecklist.md`](docs/BaseFunctionalityChecklist.md) is the completed
  checklist of core functionality.
- [`docs/PowerFlowChecklist.md`](docs/PowerFlowChecklist.md) is the roadmap toward DC/AC power
  flow.
- [`docs/ai-reports/`](docs/ai-reports/) holds implementation reports for past branches.
- API documentation: `cargo doc --open --all-features`.
