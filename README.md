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

## License

Licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([`LICENSE-MIT`](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.

## References

The algorithms in this crate were written from the standard published descriptions below. No
source code from other libraries was copied or ported.

### Implemented

| Where | Algorithm | Reference |
|---|---|---|
| `solve_lower_triangular`, `solve_upper_triangular`, `lu`, `Lu::determinant` | Forward and back substitution; Gaussian elimination with partial pivoting | G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013, ch. 3. |
| `cholesky` | Up-looking (row-by-row) sparse Cholesky. This crate's version does no elimination-tree symbolic analysis. | T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006, ch. 4. |
| `conjugate_gradient` | Conjugate gradient | M. R. Hestenes and E. Stiefel, "Methods of conjugate gradients for solving linear systems," *J. Res. Natl. Bur. Stand.* 49(6):409–436, 1952. [doi:10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044) |
| | | Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003. |
| | | J. R. Shewchuk, "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain," Tech. Rep. CMU-CS-94-125, Carnegie Mellon University, 1994. |
| `matmul`, `par_matmul`, CSR/CSC construction | Row-by-row sparse matrix product; counting-sort transposition into compressed form | F. G. Gustavson, "Two fast algorithms for sparse matrices: multiplication and permuted transposition," *ACM Trans. Math. Softw.* 4(3):250–269, 1978. [doi:10.1145/355791.355796](https://doi.org/10.1145/355791.355796) |
| Matrix Market I/O | Coordinate exchange format | R. F. Boisvert, R. Pozo and K. A. Remington, "The Matrix Market Exchange Formats: Initial Design," NISTIR 5935, NIST, 1996. [doi:10.6028/NIST.IR.5935](https://doi.org/10.6028/NIST.IR.5935) |

### Planned (see [`docs/PowerFlowChecklist.md`](docs/PowerFlowChecklist.md))

| Planned work | Reference |
|---|---|
| Left-looking sparse LU (Phase 3) | J. R. Gilbert and T. Peierls, "Sparse partial pivoting in time proportional to arithmetic operations," *SIAM J. Sci. Stat. Comput.* 9(5):862–874, 1988. [doi:10.1137/0909058](https://doi.org/10.1137/0909058) |
| Refactoring circuit-like matrices (Phase 3) | T. A. Davis and E. Palamadai Natarajan, "Algorithm 907: KLU, a direct sparse solver for circuit simulation problems," *ACM Trans. Math. Softw.* 37(3), article 36, 2010. [doi:10.1145/1824801.1824814](https://doi.org/10.1145/1824801.1824814) |
| Reverse Cuthill–McKee ordering (Phase 3) | E. Cuthill and J. McKee, "Reducing the bandwidth of sparse symmetric matrices," *Proc. 24th ACM National Conference*, pp. 157–172, 1969. |
| | A. George and J. W. H. Liu, *Computer Solution of Large Sparse Positive Definite Systems*, Prentice-Hall, 1981. |
| AMD ordering (Phase 3) | P. R. Amestoy, T. A. Davis and I. S. Duff, "An approximate minimum degree ordering algorithm," *SIAM J. Matrix Anal. Appl.* 17(4):886–905, 1996. [doi:10.1137/S0895479894278952](https://doi.org/10.1137/S0895479894278952) |
| COLAMD ordering (Phase 3) | T. A. Davis, J. R. Gilbert, S. I. Larimore and E. G. Ng, "A column approximate minimum degree ordering algorithm," *ACM Trans. Math. Softw.* 30(3):353–376, 2004. |
| Low-rank updates for contingencies (Phase 5) | J. Sherman and W. J. Morrison, "Adjustment of an inverse matrix corresponding to a change in one element of a given matrix," *Ann. Math. Statist.* 21(1):124–127, 1950. [doi:10.1214/aoms/1177729893](https://doi.org/10.1214/aoms/1177729893) |
| | M. A. Woodbury, "Inverting modified matrices," Memorandum Report 42, Statistical Research Group, Princeton University, 1950. |
| Newton–Raphson power flow and its Jacobian; reference results (Phases 1.5 and 4.5) | R. D. Zimmerman, C. E. Murillo-Sánchez and R. J. Thomas, "MATPOWER: Steady-state operations, planning, and analysis tools for power systems research and education," *IEEE Trans. Power Syst.* 26(1):12–19, 2011. [doi:10.1109/TPWRS.2010.2051168](https://doi.org/10.1109/TPWRS.2010.2051168) |
| Fast-decoupled power flow (Phase 4.5) | B. Stott and O. Alsaç, "Fast decoupled load flow," *IEEE Trans. Power App. Syst.* PAS-93(3):859–869, 1974. |

### Licenses of related software

Algorithms are not covered by copyright, so implementing them from the publications above
carries no license obligations. Porting the source code of a reference implementation would.
The current terms of the most relevant implementations:

- **SuiteSparse AMD, CAMD, COLAMD and CCOLAMD:** BSD 3-clause. Porting is possible if their
  copyright notices are kept.
- **SuiteSparse CSparse, CXSparse, KLU and BTF:** LGPL 2.1 or later. Porting their code would
  bring LGPL obligations, so implement from the papers instead.
- **MATPOWER:** the code is BSD 3-clause. Its case files have their own provenance, so check
  each file's header before vendoring them as test data.

Before relying on these, check them against each project's current `LICENSE` file.
