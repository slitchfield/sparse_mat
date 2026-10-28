use sparse_mat::{sparse_error::SparseError, sparse_matrix::SparseMatrix};

#[test]
fn sparsemat_creation() {
    let _local = SparseMatrix::<f64>::new();
    let _local2 = SparseMatrix::<f64>::empty_with_shape(3, 3);
}

#[test]
fn sparsemat_creation_huge_shape() {
    // Must not reserve space proportional to rows * cols
    let mut local = SparseMatrix::empty_with_shape(1 << 32, 1 << 32);
    local.insert(5, 7, 1.0).unwrap();
    assert!(local.num_nonzero() == 1);
}

#[test]
fn sparsemat_identity_creation() {
    let local = SparseMatrix::identity(3);
    assert!(local.num_nonzero() == 3);
    assert!(local.peek_at(0, 0).unwrap() == Some(1.0));
    assert!(local.peek_at(1, 1).unwrap() == Some(1.0));
    assert!(local.peek_at(2, 2).unwrap() == Some(1.0));
}

#[test]
fn sparsemat_transpose_creation() {
    let mut local = SparseMatrix::empty_with_shape(4, 6);
    local
        .insert_triplets(vec![
            (0, 0, 10.0),
            (0, 1, 20.0),
            (1, 1, 30.0),
            (2, 2, 50.0),
            (1, 3, 40.0),
            (2, 3, 60.0),
            (2, 4, 70.0),
            (3, 5, 80.0),
        ])
        .unwrap();

    let local2 = local.create_transpose();
    assert!(local2.shape == (6, 4));
    assert!(local2.peek_at(0, 0).unwrap() == Some(10.0));
    assert!(local2.peek_at(0, 1).unwrap().is_none());
    assert!(local2.peek_at(1, 0).unwrap() == Some(20.0));
    assert!(local2.peek_at(1, 1).unwrap() == Some(30.0));
    assert!(local2.peek_at(2, 2).unwrap() == Some(50.0));
    assert!(local2.peek_at(3, 1).unwrap() == Some(40.0));
    assert!(local2.peek_at(1, 3).unwrap().is_none());
    assert!(local2.peek_at(3, 2).unwrap() == Some(60.0));
    assert!(local2.peek_at(2, 3).unwrap().is_none());
    assert!(local2.peek_at(4, 2).unwrap() == Some(70.0));
    assert!(local2.peek_at(5, 3).unwrap() == Some(80.0));
}

#[test]
fn sparsemat_rowiter() {
    let mut local = SparseMatrix::empty_with_shape(4, 6);
    local
        .insert_triplets(vec![
            (0, 0, 10.0),
            (0, 1, 20.0),
            (1, 1, 30.0),
            (2, 2, 50.0),
            (1, 3, 40.0),
            (2, 3, 60.0),
            (2, 4, 70.0),
            (3, 5, 80.0),
        ])
        .unwrap();

    local.explicitly_compress();

    for row in local.row_iter() {
        dbg!(row);
    }
}

#[test]
fn sparsemat_rowiter_uncompressed() {
    let mut local = SparseMatrix::empty_with_shape(2, 3);
    local
        .insert_triplets(vec![(0, 2, 1.0), (1, 0, 2.0)])
        .unwrap();

    let rows: Vec<Vec<f64>> = local.row_iter().collect();
    assert!(rows == vec![vec![0.0, 0.0, 1.0], vec![2.0, 0.0, 0.0]]);
}

#[test]
fn sparsemat_rowiter_stale_after_compress() {
    let mut local = SparseMatrix::empty_with_shape(2, 2);
    local.insert(0, 0, 1.0).unwrap();
    local.explicitly_compress();
    local.insert(1, 1, 4.0).unwrap();

    let rows: Vec<Vec<f64>> = local.row_iter().collect();
    assert!(rows == vec![vec![1.0, 0.0], vec![0.0, 4.0]]);
}

#[test]
fn sparsemat_insert() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    local.insert(0, 0, 1.0).unwrap();
}

#[test]
fn sparsemat_insert_oob() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    assert_eq!(
        local.insert(4, 4, 1.0),
        Err(SparseError::OutOfBounds {
            row: 4,
            col: 4,
            shape: (3, 3)
        })
    );
}

#[test]
fn sparsemat_insert_triplets() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    local
        .insert_triplets(vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)])
        .unwrap();
    assert!(local.peek_at(0, 0) == Ok(Some(1.0)));
    assert!(local.peek_at(1, 1) == Ok(Some(2.0)));
    assert!(local.peek_at(2, 2) == Ok(Some(3.0)));
}

#[test]
fn sparsemat_insert_triplets_oob() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    assert!(local.num_nonzero() == 0);
    assert!(
        local.insert_triplets(vec![(0, 0, 2.0), (4, 4, 4.0)])
            == Err(SparseError::OutOfBounds {
                row: 4,
                col: 4,
                shape: (3, 3)
            })
    );
    // Check to see we didn't partially insert the first triplet
    assert!(local.num_nonzero() == 0);
}

#[test]
fn sparsemat_remove() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    local.insert(0, 0, 1.0).unwrap();

    let output = local.clear_at(0, 0);
    assert!(output == Ok(Some(1.0)));

    let output = local.clear_at(0, 0);
    assert!(output.is_ok());
    assert!(output.unwrap().is_none());

    let output = local.clear_at(1, 1);
    assert!(output.is_ok());
    assert!(output.unwrap().is_none());
}

#[test]
fn sparsemat_remove_oob() {
    let mut local = SparseMatrix::<f64>::empty_with_shape(3, 3);

    assert!(
        local.clear_at(4, 4)
            == Err(SparseError::OutOfBounds {
                row: 4,
                col: 4,
                shape: (3, 3)
            })
    );
}

#[test]
fn sparsemat_peek() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);

    local
        .insert_triplets(vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)])
        .unwrap();
    assert!(local.peek_at(0, 0).unwrap() == Some(1.0));
    assert!(local.peek_at(0, 1).unwrap().is_none());
}

#[test]
fn sparsemat_peek_oob() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);

    local
        .insert_triplets(vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)])
        .unwrap();
    assert!(
        local.peek_at(4, 4)
            == Err(SparseError::OutOfBounds {
                row: 4,
                col: 4,
                shape: (3, 3)
            })
    );
}

#[test]
fn sparsemat_transposeinplace() {
    let mut local = SparseMatrix::empty_with_shape(4, 4);

    local
        .insert_triplets(vec![(0, 0, 1.0), (1, 0, 2.0), (2, 2, 3.0), (2, 3, 4.0)])
        .unwrap();
    local.transpose_inplace();

    assert!(local.peek_at(0, 0).unwrap() == Some(1.0));
    assert!(local.peek_at(0, 1).unwrap() == Some(2.0));
    assert!(local.peek_at(1, 0).unwrap().is_none());
    assert!(local.peek_at(2, 2).unwrap() == Some(3.0));
    assert!(local.peek_at(3, 2).unwrap() == Some(4.0));
    assert!(local.peek_at(2, 3).unwrap().is_none());
}

#[test]
fn sparsemat_compressedrepr() {
    let mut local = SparseMatrix::empty_with_shape(4, 4);
    local
        .insert_triplets(vec![(0, 0, 5.0), (1, 1, 8.0), (3, 1, 6.0), (2, 2, 3.0)])
        .unwrap();
    assert!(local.compressed().is_none());
    local.explicitly_compress();
    assert!(local.compressed().unwrap().data == vec![5.0, 8.0, 3.0, 6.0]);
    assert!(local.compressed().unwrap().cols == vec![0, 1, 2, 1]);
    assert!(local.compressed().unwrap().rows == vec![0, 1, 2, 3, 4]);

    let mut local2 = SparseMatrix::empty_with_shape(4, 6);
    local2
        .insert_triplets(vec![
            (0, 0, 10.0),
            (0, 1, 20.0),
            (1, 1, 30.0),
            (2, 2, 50.0),
            (1, 3, 40.0),
            (2, 3, 60.0),
            (2, 4, 70.0),
            (3, 5, 80.0),
        ])
        .unwrap();
    local2.explicitly_compress();
    assert!(
        local2.compressed().unwrap().data == vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    );
    assert!(local2.compressed().unwrap().cols == vec![0, 1, 1, 3, 2, 3, 4, 5]);
    assert!(local2.compressed().unwrap().rows == vec![0, 2, 4, 7, 8]);

    // Unwrap as OOB not a part of this test
    local2.insert(0, 2, 1.0).unwrap();
    assert!(local2.compressed().is_none());
}

#[test]
fn sparsemat_display() {
    let mut local = SparseMatrix::empty_with_shape(4, 6);
    local
        .insert_triplets(vec![
            (0, 0, 10.0),
            (0, 1, 20.0),
            (1, 1, 30.0),
            (2, 2, 50.0),
            (1, 3, 40.0),
            (2, 3, 60.0),
            (2, 4, 70.0),
            (3, 5, 80.0),
        ])
        .unwrap();
    local.explicitly_compress();
    println!("{}", local)
}

#[test]
fn sparsemat_display_uncompressed() {
    let mut local = SparseMatrix::empty_with_shape(2, 2);
    local
        .insert_triplets(vec![(0, 0, 1.0), (1, 1, 2.0)])
        .unwrap();
    let before = format!("{}", local);

    local.explicitly_compress();
    assert!(before == format!("{}", local));
    assert!(before.contains("1.00"));
}

#[test]
#[should_panic]
fn sparsemat_bad_addition() {
    let local = SparseMatrix::<f64>::empty_with_shape(3, 3);
    let local2 = SparseMatrix::empty_with_shape(2, 2);

    let _local3 = &local + &local2;
}

#[test]
fn sparsemat_good_addition() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    local
        .insert_triplets(vec![(0, 0, 10.0), (0, 1, 20.0), (1, 1, 30.0), (2, 2, 50.0)])
        .unwrap();
    let local2 = local.create_transpose();

    let local3 = &local + &local2;
    assert!(local3.peek_at(0, 0).unwrap() == Some(20.0));
    assert!(local3.peek_at(0, 1).unwrap() == Some(20.0));
    assert!(local3.peek_at(1, 0).unwrap() == Some(20.0));
    assert!(local3.peek_at(1, 1).unwrap() == Some(60.0));
    assert!(local3.peek_at(2, 2).unwrap() == Some(100.0));
}

#[test]
fn sparsemat_insert_zero_into_empty() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    // Unwrap as OOB not a part of this test
    local.insert(0, 0, 0.0).unwrap();
    assert!(local.peek_at(0, 0).unwrap().is_none());
    assert!(local.num_nonzero() == 0);
}

#[test]
fn sparsemat_insert_zero_over_existing() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    // Unwrap as OOB not a part of this test
    local.insert(0, 0, 5.0).unwrap();
    local.insert(0, 0, 0.0).unwrap();
    assert!(local.peek_at(0, 0).unwrap().is_none());
    assert!(local.num_nonzero() == 0);
}

#[test]
fn sparsemat_insert_triplets_zeros() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    local
        .insert_triplets(vec![(1, 1, 2.0), (2, 2, 3.0)])
        .unwrap();
    local
        .insert_triplets(vec![(1, 1, 0.0), (0, 2, 0.0)])
        .unwrap();
    assert!(local.peek_at(1, 1).unwrap().is_none());
    assert!(local.peek_at(0, 2).unwrap().is_none());
    assert!(local.peek_at(2, 2).unwrap() == Some(3.0));
    assert!(local.num_nonzero() == 1);
}

#[test]
fn sparsemat_insert_zero_invalidates_compressed() {
    let mut local = SparseMatrix::empty_with_shape(2, 2);
    local
        .insert_triplets(vec![(0, 1, 4.0), (1, 1, 3.0)])
        .unwrap();
    local.explicitly_compress();
    // Unwrap as OOB not a part of this test
    local.insert(0, 1, 0.0).unwrap();
    assert!(local.compressed().is_none());

    let rows: Vec<Vec<f64>> = local.row_iter().collect();
    assert!(rows == vec![vec![0.0, 0.0], vec![0.0, 3.0]]);
}

#[test]
fn sparsemat_addition_cancellation() {
    let mut local = SparseMatrix::empty_with_shape(2, 2);
    local
        .insert_triplets(vec![(0, 0, 1.5), (1, 1, 2.0)])
        .unwrap();
    let mut local2 = SparseMatrix::empty_with_shape(2, 2);
    local2
        .insert_triplets(vec![(0, 0, -1.5), (0, 1, 1.0)])
        .unwrap();

    let local3 = &local + &local2;
    assert!(local3.peek_at(0, 0).unwrap().is_none());
    assert!(local3.peek_at(0, 1).unwrap() == Some(1.0));
    assert!(local3.peek_at(1, 1).unwrap() == Some(2.0));
    assert!(local3.num_nonzero() == 2);
}

#[test]
fn sparsemat_prune_zeroes_noop() {
    let mut local = SparseMatrix::empty_with_shape(2, 2);
    local
        .insert_triplets(vec![(0, 0, 1.0), (1, 1, 2.0)])
        .unwrap();
    local.explicitly_compress();

    local.prune_zeroes();
    assert!(local.num_nonzero() == 2);
    assert!(local.compressed().is_some());
}

#[test]
fn sparsemat_from_triplets_basic() {
    let local = SparseMatrix::from_triplets(3, 4, vec![(0, 0, 1.0), (2, 3, 7.0)]).unwrap();
    assert!(local.shape == (3, 4));
    assert!(local.peek_at(0, 0) == Ok(Some(1.0)));
    assert!(local.peek_at(2, 3) == Ok(Some(7.0)));
    assert!(local.peek_at(1, 1) == Ok(None));
    assert!(local.num_nonzero() == 2);
    assert!(local.compressed().is_none());
}

#[test]
fn sparsemat_from_triplets_sums_duplicates() {
    let local =
        SparseMatrix::from_triplets(2, 2, vec![(0, 0, 1.0), (1, 1, 4.0), (0, 0, 2.5)]).unwrap();
    assert!(local.peek_at(0, 0) == Ok(Some(3.5)));
    assert!(local.num_nonzero() == 2);
}

#[test]
fn sparsemat_from_triplets_cancellation_removed() {
    let local = SparseMatrix::from_triplets(2, 2, vec![(1, 1, 5.0), (1, 1, -5.0)]).unwrap();
    assert!(local.peek_at(1, 1) == Ok(None));
    assert!(local.num_nonzero() == 0);
}

#[test]
fn sparsemat_from_triplets_skips_zero_inputs() {
    let local = SparseMatrix::from_triplets(2, 2, vec![(0, 0, 0.0), (1, 0, 2.0)]).unwrap();
    assert!(local.peek_at(0, 0) == Ok(None));
    assert!(local.num_nonzero() == 1);
}

#[test]
fn sparsemat_from_triplets_oob() {
    let result = SparseMatrix::from_triplets(3, 3, vec![(0, 0, 1.0), (4, 4, 2.0)]);
    assert!(
        result.err()
            == Some(SparseError::OutOfBounds {
                row: 4,
                col: 4,
                shape: (3, 3)
            })
    );
}

#[test]
fn sparsemat_from_triplets_empty() {
    let local = SparseMatrix::<f64>::from_triplets(3, 5, vec![]).unwrap();
    assert!(local.shape == (3, 5));
    assert!(local.num_nonzero() == 0);
}

#[test]
fn sparsemat_from_triplets_huge_shape() {
    // Capacity must come from the triplet count, not rows * cols
    let local = SparseMatrix::from_triplets(1 << 32, 1 << 32, [(0, 0, 1.0)]).unwrap();
    assert!(local.num_nonzero() == 1);
}

#[test]
fn sparsemat_from_triplets_row_iter() {
    let local =
        SparseMatrix::from_triplets(2, 3, vec![(0, 2, 1.0), (1, 0, 2.0), (0, 2, 1.0)]).unwrap();
    let rows: Vec<Vec<f64>> = local.row_iter().collect();
    assert!(rows == vec![vec![0.0, 0.0, 2.0], vec![2.0, 0.0, 0.0]]);
}

#[test]
fn sparsemat_from_dense_basic() {
    let local = SparseMatrix::from_dense(vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 0.0]]).unwrap();
    assert!(local.shape == (2, 3));
    assert!(local.peek_at(0, 0) == Ok(Some(1.0)));
    assert!(local.peek_at(0, 2) == Ok(Some(2.0)));
    assert!(local.peek_at(1, 1) == Ok(Some(3.0)));
    assert!(local.num_nonzero() == 3);
    assert!(local.compressed().is_none());
}

#[test]
fn sparsemat_from_dense_skips_zeros() {
    let local = SparseMatrix::from_dense(vec![vec![0.0, -0.0], vec![0.0, 5.0]]).unwrap();
    assert!(local.peek_at(0, 0) == Ok(None));
    assert!(local.peek_at(0, 1) == Ok(None));
    assert!(local.num_nonzero() == 1);
}

#[test]
fn sparsemat_from_dense_ragged() {
    let result = SparseMatrix::from_dense(vec![vec![1.0, 2.0], vec![3.0], vec![4.0, 5.0]]);
    assert!(
        result.err()
            == Some(SparseError::RaggedRows {
                row: 1,
                expected: 2,
                found: 1
            })
    );
}

#[test]
fn sparsemat_from_dense_empty() {
    let local = SparseMatrix::from_dense(Vec::<Vec<f64>>::new()).unwrap();
    assert!(local.shape == (0, 0));
    assert!(local.num_nonzero() == 0);

    // Rows with no columns are a valid n x 0 matrix
    let local = SparseMatrix::<f64>::from_dense(vec![vec![], vec![]]).unwrap();
    assert!(local.shape == (2, 0));
}

#[test]
fn sparsemat_from_dense_round_trip() {
    let mut original = SparseMatrix::empty_with_shape(3, 4);
    original
        .insert_triplets(vec![(0, 0, 1.0), (1, 3, -2.0), (2, 1, 4.5)])
        .unwrap();

    let rebuilt = SparseMatrix::from_dense(original.row_iter()).unwrap();
    assert!(rebuilt.shape == original.shape);
    assert!(rebuilt.num_nonzero() == original.num_nonzero());
    for (r, c, v) in [(0, 0, 1.0), (1, 3, -2.0), (2, 1, 4.5)] {
        assert!(rebuilt.peek_at(r, c) == Ok(Some(v)));
    }
}

#[test]
fn sparsemat_from_dense_array_input() {
    let local = SparseMatrix::from_dense([[1.0, 0.0], [0.0, 2.0]]).unwrap();
    assert!(local.shape == (2, 2));
    assert!(local.num_nonzero() == 2);
}

#[test]
fn sparsemat_from_dense_nan_is_stored() {
    let local = SparseMatrix::from_dense(vec![vec![f64::NAN, 0.0]]).unwrap();
    assert!(local.num_nonzero() == 1);
    assert!(local.peek_at(0, 0).unwrap().unwrap().is_nan());
}

#[test]
fn sparsemat_to_dense_basic() {
    let mut local = SparseMatrix::empty_with_shape(2, 3);
    local
        .insert_triplets(vec![(0, 2, 1.0), (1, 0, 2.0)])
        .unwrap();
    assert!(local.to_dense() == vec![vec![0.0, 0.0, 1.0], vec![2.0, 0.0, 0.0]]);
}

#[test]
fn sparsemat_to_dense_matches_row_iter() {
    let local =
        SparseMatrix::from_triplets(3, 4, vec![(0, 0, 1.0), (2, 3, 4.0), (1, 1, -2.0)]).unwrap();
    let rows: Vec<Vec<f64>> = local.row_iter().collect();
    assert!(local.to_dense() == rows);
}

#[test]
fn sparsemat_to_dense_empty_shapes() {
    assert!(SparseMatrix::<f64>::new().to_dense().is_empty());
    assert!(SparseMatrix::<f64>::empty_with_shape(2, 0).to_dense() == vec![Vec::<f64>::new(); 2]);
    // No rows means the column count is lost
    assert!(SparseMatrix::<f64>::empty_with_shape(0, 3)
        .to_dense()
        .is_empty());
}

#[test]
fn sparsemat_to_dense_round_trip() {
    let dense = vec![
        vec![1.0, 0.0, 2.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 3.5, 0.0],
    ];
    let local = SparseMatrix::from_dense(dense.clone()).unwrap();
    assert!(local.to_dense() == dense);

    let rebuilt = SparseMatrix::from_dense(local.to_dense()).unwrap();
    assert!(rebuilt.shape == local.shape);
    assert!(rebuilt.num_nonzero() == local.num_nonzero());
}

#[test]
fn sparsemat_integer_elements() {
    let mut local = SparseMatrix::<i32>::empty_with_shape(2, 2);
    local.insert(0, 0, 3).unwrap();
    local.insert(1, 1, 4).unwrap();
    local.insert(1, 1, 0).unwrap();
    assert!(local.peek_at(0, 0) == Ok(Some(3)));
    assert!(local.peek_at(1, 1) == Ok(None));

    let other = SparseMatrix::from_triplets(2, 2, vec![(0, 0, 4), (0, 1, 1)]).unwrap();
    let sum = &local + &other;
    assert!(sum.to_dense() == vec![vec![7, 1], vec![0, 0]]);
    assert!(format!("{}", sum).contains('7'));
}

#[test]
fn sparsemat_f32_elements() {
    let local = SparseMatrix::<f32>::identity(3);
    assert!(local.num_nonzero() == 3);
    let doubled = &local + &local;
    assert!(doubled.peek_at(2, 2) == Ok(Some(2.0f32)));
}

#[test]
fn sparsemat_complex_elements() {
    use num_complex::Complex;
    let i = Complex::new(0.0, 1.0);
    let mut local = SparseMatrix::empty_with_shape(2, 2);
    local.insert(0, 1, i).unwrap();
    local.insert(1, 0, Complex::new(1.0, 0.0)).unwrap();
    let sum = &local + &local;
    assert!(sum.peek_at(0, 1) == Ok(Some(Complex::new(0.0, 2.0))));
    assert!(local.create_transpose().peek_at(1, 0) == Ok(Some(i)));
}

#[test]
fn sparsemat_compressed_csc() {
    let mut local = SparseMatrix::from_triplets(
        4,
        6,
        vec![
            (0, 0, 10.0),
            (0, 1, 20.0),
            (1, 1, 30.0),
            (2, 2, 50.0),
            (1, 3, 40.0),
            (2, 3, 60.0),
            (2, 4, 70.0),
            (3, 5, 80.0),
        ],
    )
    .unwrap();
    assert!(local.compressed_csc().is_none());
    local.explicitly_compress_csc();

    let csc = local.compressed_csc().unwrap();
    assert!(csc.cols == vec![0, 1, 3, 4, 6, 7, 8]);
    assert!(csc.rows == vec![0, 0, 1, 2, 1, 2, 2, 3]);
    assert!(csc.data == vec![10.0, 20.0, 30.0, 50.0, 40.0, 60.0, 70.0, 80.0]);
    // CSR was not built as a side effect
    assert!(local.compressed().is_none());

    local.insert(3, 0, 1.0).unwrap();
    assert!(local.compressed_csc().is_none());
}

#[test]
fn sparsemat_csc_matches_transpose_csr() {
    let mut local = SparseMatrix::from_triplets(
        3,
        5,
        vec![
            (0, 4, 1.0),
            (2, 0, 2.0),
            (1, 1, 3.0),
            (2, 4, 4.0),
            (0, 0, 5.0),
        ],
    )
    .unwrap();
    local.explicitly_compress_csc();
    let mut transposed = local.create_transpose();
    transposed.explicitly_compress();

    let csc = local.compressed_csc().unwrap();
    let csr = transposed.compressed().unwrap();
    assert!(csc.cols == csr.rows);
    assert!(csc.rows == csr.cols);
    assert!(csc.data == csr.data);
}

#[test]
fn sparsemat_try_add_shape_mismatch() {
    let local = SparseMatrix::<f64>::empty_with_shape(3, 3);
    let local2 = SparseMatrix::<f64>::empty_with_shape(2, 4);
    assert!(
        local.try_add(&local2).err()
            == Some(SparseError::ShapeMismatch {
                left: (3, 3),
                right: (2, 4)
            })
    );
}

#[test]
fn sparsemat_try_add_matches_operator() {
    let local = SparseMatrix::from_triplets(2, 2, vec![(0, 0, 1.5), (1, 1, 2.0)]).unwrap();
    let local2 = SparseMatrix::from_triplets(2, 2, vec![(0, 0, -1.5), (0, 1, 1.0)]).unwrap();
    let sum = local.try_add(&local2).unwrap();
    assert!(sum.to_dense() == (&local + &local2).to_dense());
    assert!(sum.num_nonzero() == 2);
    // Operands are untouched
    assert!(local.peek_at(0, 0) == Ok(Some(1.5)));
}

#[test]
#[should_panic(expected = "incompatible shapes")]
fn sparsemat_add_shape_mismatch_message() {
    let local = SparseMatrix::<f64>::empty_with_shape(3, 3);
    let local2 = SparseMatrix::<f64>::empty_with_shape(2, 2);
    let _ = &local + &local2;
}

#[test]
fn sparsemat_iter_row_major_order() {
    let local = SparseMatrix::from_triplets(
        4,
        4,
        vec![
            (3, 1, 6.0),
            (0, 3, 2.0),
            (2, 2, 5.0),
            (0, 0, 1.0),
            (3, 0, 9.0),
        ],
    )
    .unwrap();
    let entries: Vec<(usize, usize, f64)> = local.iter().collect();
    assert!(
        entries
            == vec![
                (0, 0, 1.0),
                (0, 3, 2.0),
                (2, 2, 5.0),
                (3, 0, 9.0),
                (3, 1, 6.0)
            ]
    );
}

#[test]
fn sparsemat_iter_len_and_empty() {
    let local = SparseMatrix::from_triplets(3, 3, vec![(0, 0, 1.0), (2, 2, 2.0)]).unwrap();
    let mut entries = local.iter();
    assert!(entries.len() == 2);
    entries.next();
    assert!(entries.len() == 1);

    assert!(SparseMatrix::<f64>::empty_with_shape(5, 5)
        .iter()
        .next()
        .is_none());
    assert!(SparseMatrix::<f64>::new().iter().count() == 0);
}

#[test]
fn sparsemat_iter_compressed_or_not() {
    let mut local = SparseMatrix::from_triplets(3, 3, vec![(1, 2, 4.0), (0, 1, 3.0)]).unwrap();
    let before: Vec<_> = local.iter().collect();
    local.explicitly_compress();
    let after: Vec<_> = local.iter().collect();
    assert!(before == after);
    assert!(before == vec![(0, 1, 3.0), (1, 2, 4.0)]);
}

#[test]
fn sparsemat_iter_sparse_cost_on_huge_shape() {
    let mut local = SparseMatrix::<f64>::empty_with_shape(1 << 20, 1 << 20);
    local.insert(5, 7, 1.0).unwrap();
    local.insert(1 << 19, 3, 2.0).unwrap();
    let entries: Vec<_> = local.iter().collect();
    assert!(entries == vec![(5, 7, 1.0), (1 << 19, 3, 2.0)]);
}

fn sample_4x6() -> SparseMatrix {
    SparseMatrix::from_triplets(
        4,
        6,
        vec![
            (0, 0, 10.0),
            (0, 1, 20.0),
            (1, 1, 30.0),
            (2, 2, 50.0),
            (1, 3, 40.0),
            (2, 3, 60.0),
            (2, 4, 70.0),
            (3, 5, 80.0),
        ],
    )
    .unwrap()
}

#[test]
fn sparsemat_row_extraction() {
    let local = sample_4x6();
    let row = local.row(2).unwrap();
    assert!(row.len() == 6);
    assert!(row.nnz() == 3);
    assert!(row.indices() == [2, 3, 4]);
    assert!(row.values() == [50.0, 60.0, 70.0]);
    assert!(row.get(3) == Some(60.0));
    assert!(row.get(0).is_none());
    assert!(row.to_dense() == vec![0.0, 0.0, 50.0, 60.0, 70.0, 0.0]);
    // An empty row
    let empty = SparseMatrix::<f64>::empty_with_shape(2, 3).row(1).unwrap();
    assert!(empty.nnz() == 0 && empty.len() == 3);
}

#[test]
fn sparsemat_col_extraction() {
    let mut local = sample_4x6();
    let col = local.col(3).unwrap();
    assert!(col.len() == 4);
    assert!(col.indices() == [1, 2]);
    assert!(col.values() == [40.0, 60.0]);

    // Same answer from a fresh CSC cache
    local.explicitly_compress_csc();
    assert!(local.col(3).unwrap() == col);
}

#[test]
fn sparsemat_row_col_out_of_range() {
    let local = sample_4x6();
    assert!(local.row(4).err() == Some(SparseError::OutOfRange { index: 4, len: 4 }));
    assert!(local.col(6).err() == Some(SparseError::OutOfRange { index: 6, len: 6 }));
}

#[test]
fn sparsemat_col_iter() {
    let local = sample_4x6();
    let cols: Vec<Vec<f64>> = local.col_iter().collect();
    assert!(cols.len() == 6);
    assert!(cols[1] == vec![20.0, 30.0, 0.0, 0.0]);
    assert!(cols[5] == vec![0.0, 0.0, 0.0, 80.0]);
    // Columns of a matrix are the rows of its transpose
    let transposed_rows: Vec<Vec<f64>> = local.create_transpose().row_iter().collect();
    assert!(cols == transposed_rows);
}

#[test]
fn sparsemat_diagonal() {
    let local = sample_4x6();
    let diag = local.diagonal();
    assert!(diag.len() == 4);
    assert!(diag.indices() == [0, 1, 2]);
    assert!(diag.values() == [10.0, 30.0, 50.0]);
    assert!(diag.to_dense() == vec![10.0, 30.0, 50.0, 0.0]);

    let identity = SparseMatrix::<f64>::identity(3);
    assert!(identity.diagonal().to_dense() == vec![1.0, 1.0, 1.0]);
    // Tall matrix: diagonal length is the smaller dimension
    assert!(SparseMatrix::<f64>::empty_with_shape(5, 2).diagonal().len() == 2);
}

#[test]
fn sparsemat_submatrix() {
    let local = sample_4x6();
    let block = local.submatrix(1..3, 1..4).unwrap();
    assert!(block.shape == (2, 3));
    assert!(block.to_dense() == vec![vec![30.0, 0.0, 40.0], vec![0.0, 50.0, 60.0]]);

    let whole = local.submatrix(0..4, 0..6).unwrap();
    assert!(whole.to_dense() == local.to_dense());

    let empty = local.submatrix(2..2, 0..6).unwrap();
    assert!(empty.shape == (0, 6));
}

#[test]
fn sparsemat_submatrix_invalid_range() {
    let local = sample_4x6();
    assert!(
        local.submatrix(0..5, 0..6).err()
            == Some(SparseError::InvalidRange {
                start: 0,
                end: 5,
                len: 4
            })
    );
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = local.submatrix(0..4, 4..2);
    assert!(
        reversed.err()
            == Some(SparseError::InvalidRange {
                start: 4,
                end: 2,
                len: 6
            })
    );
}

#[test]
fn sparsemat_index() {
    let local = sample_4x6();
    assert!(local[(0, 1)] == 20.0);
    assert!(local[(3, 5)] == 80.0);
    // Unstored entries read as zero
    assert!(local[(1, 0)] == 0.0);
    assert!(local[(3, 0)] == 0.0);
}

#[test]
fn sparsemat_index_integers() {
    let local = SparseMatrix::from_triplets(2, 2, vec![(0, 1, 7)]).unwrap();
    assert!(local[(0, 1)] == 7);
    assert!(local[(1, 1)] == 0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn sparsemat_index_oob() {
    let local = sample_4x6();
    let _ = local[(4, 0)];
}

#[test]
#[should_panic(expected = "out of bounds")]
fn sparsemat_index_oob_col() {
    let local = sample_4x6();
    let _ = local[(0, 6)];
}
