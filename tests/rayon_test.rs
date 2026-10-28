#![cfg(feature = "rayon")]

use sparse_mat::sparse_error::SparseError;
use sparse_mat::sparse_matrix::SparseMatrix;

/// Deterministic pseudo-random sparse matrix with small integer-valued entries.
fn pseudo_random(rows: usize, cols: usize, seed: u64, fill_percent: u64) -> SparseMatrix {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    let mut triplets = vec![];
    for row in 0..rows {
        for col in 0..cols {
            if next() % 100 < fill_percent {
                triplets.push((row, col, (next() % 9) as f64 - 4.5));
            }
        }
    }
    SparseMatrix::from_triplets(rows, cols, triplets).unwrap()
}

#[test]
fn par_mul_vec_equals_sequential_exactly() {
    for seed in 1..=5 {
        let a = pseudo_random(60, 45, seed, 20);
        let x: Vec<f64> = (0..45).map(|i| (i as f64).sin()).collect();
        assert_eq!(a.par_mul_vec(&x).unwrap(), a.mul_vec(&x).unwrap());
    }
}

#[test]
fn par_vec_mul_equals_sequential_exactly() {
    for seed in 1..=5 {
        let a = pseudo_random(45, 60, seed, 20);
        let x: Vec<f64> = (0..45).map(|i| (i as f64).cos()).collect();
        assert_eq!(a.par_vec_mul(&x).unwrap(), a.vec_mul(&x).unwrap());
    }
}

#[test]
fn par_matmul_equals_sequential() {
    for seed in 1..=4 {
        let a = pseudo_random(40, 30, seed, 15);
        let b = pseudo_random(30, 50, seed + 50, 15);
        assert!(a.par_matmul(&b).unwrap() == a.matmul(&b).unwrap());
    }
}

#[test]
fn par_matmul_drops_cancellations() {
    let a = SparseMatrix::from([[1, -1], [0, 2]]);
    let b = SparseMatrix::from([[1, 0], [1, 5]]);
    let product = a.par_matmul(&b).unwrap();
    assert!(product == a.matmul(&b).unwrap());
    assert_eq!(product.num_nonzero(), 3);
}

#[test]
fn par_products_report_shape_mismatches() {
    let a = SparseMatrix::<f64>::empty_with_shape(2, 3);
    assert_eq!(
        a.par_mul_vec(&[1.0]).err(),
        Some(SparseError::ShapeMismatch {
            left: (2, 3),
            right: (1, 1)
        })
    );
    assert_eq!(
        a.par_vec_mul(&[1.0, 2.0, 3.0]).err(),
        Some(SparseError::ShapeMismatch {
            left: (1, 3),
            right: (2, 3)
        })
    );
    assert_eq!(
        a.par_matmul(&a).err(),
        Some(SparseError::ShapeMismatch {
            left: (2, 3),
            right: (2, 3)
        })
    );
}

#[test]
fn par_products_on_empty_matrices() {
    let a = SparseMatrix::<f64>::new();
    assert_eq!(a.par_mul_vec(&[]).unwrap(), Vec::<f64>::new());
    assert_eq!(a.par_vec_mul(&[]).unwrap(), Vec::<f64>::new());
    assert_eq!(a.par_matmul(&a).unwrap().shape, (0, 0));
}
