# Implementation report: `dev/claude`

This branch implements every remaining item in `Todo.md` on top of `main` (`72aefee`
"Add doc comments to all items" was the starting point). Each Todo item is its own commit, and
I ran `./check.sh` before each commit (see [Verification](#verification)).

| | `main` | `dev/claude` |
|---|---|---|
| Commits on the branch | – | 19 (17 Todo items, 1 doc fix-up, this report) |
| Source files | `lib.rs`, `sparse_matrix.rs`, `sparse_error.rs` | + 10 modules under `src/sparse_matrix/` |
| Tests (all features) | 46 | 159 (149 core, 6 `rayon`, 4 `serde`) |
| Dependencies | none | `num-traits`; optional `serde`, `rayon`; dev: `num-complex`, `serde_json` |

## Read this first: interpretations and deviations

A few Todo items were ambiguous or collided with each other. These are the decisions I made.
Anything you would rather have gone the other way is a small, isolated follow-up.

1. **Branch name is `dev/claude`, not `/dev/claude`.** Git ref names cannot start with `/`.
2. **`diagonal(vec)` is named `from_diagonal`.** The same Todo list also asks for a
   `diagonal()` accessor, and an associated function and a method cannot share a name in one
   `impl`.
3. **"Real compressed storage" is cached CSR *and* CSC, not a CSR-only store.** The hash map
   remains the mutable source of truth, because making CSR primary would turn `insert` from
   O(1) into O(nnz). The item's stated goals are met: a CSC form now exists, column access and
   `Aᵀx` read it, and one shared `build_compressed` routine builds both forms in
   O(nnz + dim). Operations that need compressed storage borrow a fresh cache when there is
   one and otherwise build a temporary copy, so they are always correct and merely faster after
   `explicitly_compress` / `explicitly_compress_csc`.
4. **`Mul<f64>` is `Mul<T>`.** Scalar multiplication is generic over the element type, with
   `scalar * matrix` provided for each primitive numeric type (a blanket impl is not allowed
   for a foreign left-hand type).
5. **Generic scalar type is a `Scalar` trait** (`Copy + PartialEq + num_traits::Num`, blanket
   implemented), so `f32`, `f64`, all integer types and `num_complex::Complex` work. Operations
   that need `abs`/`sqrt`/ordering (norms, solvers) are limited to `T: Float`, so they exclude
   complex numbers.
6. **`SparseMatrix` is now `SparseMatrix<T = f64>`.** The default keeps type annotations
   unchanged, but Rust does not apply defaults to expression paths, so calls like
   `SparseMatrix::new()` with nothing else to infer `T` from need `::<f64>`. Nine existing test
   lines were annotated for this.

## Breaking API changes relative to `main`

- Dimensions and indices are `usize`, not `u64`.
- `SparseMatrix` gained a type parameter (see above); `SparseError` is `#[non_exhaustive]`
  and has many new variants.
- The four `compressed_*` fields and the `compressed_updated` flag are replaced by `Option`
  caches; `compressed()` is unchanged, and `compressed_csc()` is new.
- `Add` was moved into `ops.rs` and now builds its result from the entry maps instead of cloning
  the whole matrix; behaviour is unchanged except that a shape mismatch panics with the
  `SparseError` message ("incompatible shapes ...").

## Per-item summary

### 1. Dimension type → `usize` — `25d0e77`
`u64` → `usize` everywhere, removing every `as usize` cast. Also removed a clippy-flagged index
loop in the CSR builder.

### 2. Generic scalar type — `bab8d9e`
`SparseMatrix<T>`, `CsrView<T>`, `RowIterator<T>`, plus the `Scalar` trait in `lib.rs`. New tests
for `i32`, `f32` and `Complex<f64>` matrices.

### 3. Real compressed storage — `195cd21`
`Compressed<T> { ptr, idx, data }` shared by CSR and CSC; `explicitly_compress_csc`,
`compressed_csc` and `CscView`; `RowIterator` holds a `Cow` (borrowed cache or owned temporary).
Tests check the CSC arrays directly and that CSC equals the CSR of the transpose.

### 4. Result-based errors: shape mismatch — `b66e702`
`SparseError::ShapeMismatch`, `try_add`. Together with the earlier OOB work this completes
"Result-based errors".

### 5. Stored-entry iterator — `d55b2fd`
`iter()` → `Entries`, an `ExactSizeIterator` over `(row, col, value)` in row-major order, with
cost proportional to `num_nonzero`.

### 6. Sparse rows and columns — `40becbd`
`row(i)`, `col(j)` (→ `SparseVector`), `col_iter()`, `diagonal()`, `submatrix(rows, cols)`;
`OutOfRange` and `InvalidRange` errors.

### 7. Indexing — `19096eb`
`Index<(usize, usize)>`. Unstored entries read as zero via a `zero: T` field kept on the matrix
so a reference can be returned; out-of-bounds panics with a message (`peek_at` stays the checked
read).

### 8. Standard traits — `c7c3cc2`
Custom `Debug` (shape + sorted entries, no caches), `PartialEq` (shape and entries, independent
of cache state), `IntoIterator` for `&SparseMatrix` and `SparseMatrix`, `From<[[T; C]; R]>`,
`TryFrom<Vec<Vec<T>>>`, and `From<SparseMatrix>`/`From<&SparseMatrix>` for `Vec<Vec<T>>`.

### 9. Bulk constructors (remainder) — `38f1e42`
`from_csr` (validated; adopts the arrays as the CSR cache when there are no zeros) and
`from_diagonal`. `InvalidCsr` error.

### 10. Scalar and other operators — `3c5884e`
`Sub`, `Neg`, `Mul<T>`, `MulAssign<T>`, `AddAssign`/`SubAssign`, `try_sub`, `scale`. Binary
operators exist for every owned/borrowed operand combination; an owned left operand is updated in
place. All share one `merge_in_place` routine that never stores zeros.

### 11. Element-wise ops — `387afa7`
`hadamard` (visits the sparser operand), `map` (may change element type) and `apply` (in place).
`map`/`apply` see stored entries only, never the implicit zeros; this is documented.

### 12. Multiplication — `92aec27`
`mul_vec`, `vec_mul` (`xA` without forming the transpose, via CSC), `mul_dense`, `matmul`
(Gustavson SpGEMM), plus `Mul` operators for matrix×matrix (all operand forms), matrix×vector and
vector×matrix. Products walk sorted compressed storage so floating-point sums are deterministic.

### 13. Properties and reductions — `8bcefa6`
`is_square`, `density`, `sum`, `trace` (`NotSquare` error), `is_symmetric`, and
`norm_frobenius` / `norm_1` / `norm_inf` for float types.

### 14. Block operations — `f730ce2`
`kron` (visits stored entries only; `ShapeOverflow` error), `hstack`, `vstack` (accept owned or
borrowed blocks).

### 15. Solvers — `6ac50f9`
- `solve_lower_triangular` / `solve_upper_triangular` (read one triangle only).
- `lu()` → `Lu` (sparse row maps, partial pivoting; `solve`, `determinant`, `l`, `u`,
  `permutation`) and `solve()`.
- `cholesky()` (up-looking sparse algorithm) and `solve_spd()`.
- `conjugate_gradient(b, tolerance, max_iterations)`, which builds the CSR form once.
- Errors: `SingularMatrix`, `NotSymmetric`, `NotPositiveDefinite`, `NoConvergence`.

### 16. File I/O — `231e0c0`
Matrix Market coordinate format: `from_matrix_market`, `read_matrix_market` (any `BufRead`),
`write_matrix_market` (any `Write`), `to_matrix_market`. Reads `real`/`integer`/`pattern` fields
and `general`/`symmetric`/`skew-symmetric` symmetry, with line-numbered `Parse` errors and `Io`
errors. Float round trips are exact.

### 17. Optional features — `3a4a279`
- `serde`: `{"shape":[r,c],"entries":[[row,col,value],...]}` in row-major order; deserialization
  validates like `from_triplets`.
- `rayon`: `par_mul_vec`, `par_vec_mul`, `par_matmul`. Each result row is computed by one thread
  in the sequential accumulation order, so results are bit-for-bit equal to the sequential ones.
- `check.sh` now also runs the tests without features.

### Fix-up — `b5f345d`
Removed a redundant intra-doc link flagged by `cargo doc -D warnings`.

## Verification

- `./check.sh` passes at the branch tip, and I ran it before each commit: `cargo check` (default
  and all features), `cargo fmt --check`, `cargo clippy --all-targets --all-features -D warnings`,
  tests with default features and with `--all-features`, and doc tests.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` is clean.
- Products, solvers and Kronecker products are checked against dense reference implementations
  on deterministic pseudo-random matrices, and against algebraic identities: `A·I = A`,
  associativity, `(AB)ᵀ = BᵀAᵀ`, `(A⊗B)(C⊗D) = AC⊗BD`, `PA = LU`, `A = LLᵀ`, `x·A = Aᵀx`.
- Huge sparse shapes (`1 << 20` and `1 << 32` squares) are exercised for construction, iteration
  and SpGEMM to guard against `rows × cols` costs.
- Mistakes caught while building: two wrong expected values in my own tests (a diagonal sample and
  an upper-triangular right-hand side) and four clippy findings in the solvers; all fixed before
  committing.

## Known limitations and things not verified

- **CI is unverified at the time of writing.** The workflow now runs `./check.sh`, which fetches
  crates and runs the suite with and without features; check the PR's status checks for the
  first real result.
- **No benchmarks.** `rayon` correctness is tested, but I make no performance claims about it.
- **LU pivot search is O(n²)** in the dimension (it examines every remaining row per column), and
  `matmul` iterates every row of the left operand, so very tall mostly-empty matrices pay O(rows).
- **`to_dense` and `mul_dense` allocate `rows × cols`** and will panic/abort for shapes that do
  not fit in memory.
- **Unsigned integer subtraction follows Rust's overflow rules** (documented on `try_sub`).
- **`cholesky` requires exact symmetry** (it checks with `==`); `conjugate_gradient` assumes
  symmetry without checking.
- **Matrix Market:** `array` format, `complex` values and `hermitian` symmetry are rejected;
  writing always emits `real general`, including for integer matrices.
- **Floating point:** `PartialEq` is not reflexive for matrices containing `NaN`.
- `Todo.md`'s closing "Suggested order" section is now historical.
