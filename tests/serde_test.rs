#![cfg(feature = "serde")]

use sparse_mat::sparse_matrix::SparseMatrix;

#[test]
fn serde_json_format_and_round_trip() {
    let local = SparseMatrix::from_triplets(2, 3, vec![(1, 0, 2.0), (0, 2, 1.5)]).unwrap();
    let json = serde_json::to_string(&local).unwrap();
    // Row-major entries, whatever order they were inserted in
    assert_eq!(json, r#"{"shape":[2,3],"entries":[[0,2,1.5],[1,0,2.0]]}"#);

    let back: SparseMatrix<f64> = serde_json::from_str(&json).unwrap();
    assert!(back == local);
}

#[test]
fn serde_ignores_cache_state() {
    let mut local = SparseMatrix::from_triplets(2, 2, vec![(0, 0, 1), (1, 1, 2)]).unwrap();
    let before = serde_json::to_string(&local).unwrap();
    local.explicitly_compress();
    local.explicitly_compress_csc();
    assert_eq!(serde_json::to_string(&local).unwrap(), before);
}

#[test]
fn serde_empty_and_integer_matrices() {
    let empty = SparseMatrix::<f64>::empty_with_shape(4, 0);
    let json = serde_json::to_string(&empty).unwrap();
    assert_eq!(json, r#"{"shape":[4,0],"entries":[]}"#);
    let back: SparseMatrix<f64> = serde_json::from_str(&json).unwrap();
    assert!(back.shape == (4, 0));

    let ints = SparseMatrix::from([[1i64, 0], [0, -5]]);
    let back: SparseMatrix<i64> =
        serde_json::from_str(&serde_json::to_string(&ints).unwrap()).unwrap();
    assert!(back == ints);
}

#[test]
fn serde_deserialize_validates_entries() {
    let out_of_bounds = r#"{"shape":[2,2],"entries":[[0,5,1.0]]}"#;
    let err = serde_json::from_str::<SparseMatrix<f64>>(out_of_bounds).unwrap_err();
    assert!(err.to_string().contains("out of bounds"), "{err}");

    // Duplicates are summed and zeros dropped, as in from_triplets
    let messy = r#"{"shape":[2,2],"entries":[[0,0,1.0],[0,0,2.0],[1,1,0.0]]}"#;
    let local: SparseMatrix<f64> = serde_json::from_str(messy).unwrap();
    assert!(local[(0, 0)] == 3.0 && local.num_nonzero() == 1);

    assert!(serde_json::from_str::<SparseMatrix<f64>>(r#"{"shape":[2,2]}"#).is_err());
    assert!(serde_json::from_str::<SparseMatrix<f64>>("[]").is_err());
}
