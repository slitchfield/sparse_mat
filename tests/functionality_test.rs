use sparse_mat::{sparse_error::SparseError, sparse_matrix::SparseMatrix};

#[test]
fn sparsemat_creation() {
    let _local = SparseMatrix::new();
    let _local2 = SparseMatrix::empty_with_shape(3, 3);
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
    let mut local = SparseMatrix::empty_with_shape(3, 3);

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
    let local = SparseMatrix::empty_with_shape(3, 3);
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
    let local = SparseMatrix::from_triplets(3, 5, vec![]).unwrap();
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
