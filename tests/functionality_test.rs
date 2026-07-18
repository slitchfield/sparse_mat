use sparse_mat::sparse_matrix::SparseMatrix;

#[test]
fn sparsemat_creation() {
    let _local = SparseMatrix::new();
    let _local2 = SparseMatrix::empty_with_shape(3, 3);
}

#[test]
fn sparsemat_identity_creation() {
    let local = SparseMatrix::identity(3);
    assert!(local.num_nonzero() == 3);
    assert!(local.peek_at(0, 0) == Some(1.0));
    assert!(local.peek_at(1, 1) == Some(1.0));
    assert!(local.peek_at(2, 2) == Some(1.0));
}

#[test]
fn sparsemat_transpose_creation() {
    let mut local = SparseMatrix::empty_with_shape(4, 6);
    local.insert_triplets(vec![
        (0, 0, 10.0),
        (0, 1, 20.0),
        (1, 1, 30.0),
        (2, 2, 50.0),
        (1, 3, 40.0),
        (2, 3, 60.0),
        (2, 4, 70.0),
        (3, 5, 80.0),
    ]);

    let local2 = local.create_transpose();
    assert!(local2.shape == (6, 4));
    assert!(local2.peek_at(0, 0) == Some(10.0));
    assert!(local2.peek_at(0, 1).is_none());
    assert!(local2.peek_at(1, 0) == Some(20.0));
    assert!(local2.peek_at(1, 1) == Some(30.0));
    assert!(local2.peek_at(2, 2) == Some(50.0));
    assert!(local2.peek_at(3, 1) == Some(40.0));
    assert!(local2.peek_at(1, 3).is_none());
    assert!(local2.peek_at(3, 2) == Some(60.0));
    assert!(local2.peek_at(2, 3).is_none());
    assert!(local2.peek_at(4, 2) == Some(70.0));
    assert!(local2.peek_at(5, 3) == Some(80.0));
}

#[test]
fn sparsemat_rowiter() {
    let mut local = SparseMatrix::empty_with_shape(4, 6);
    local.insert_triplets(vec![
        (0, 0, 10.0),
        (0, 1, 20.0),
        (1, 1, 30.0),
        (2, 2, 50.0),
        (1, 3, 40.0),
        (2, 3, 60.0),
        (2, 4, 70.0),
        (3, 5, 80.0),
    ]);

    local.explicitly_compress();

    for row in local.row_iter() {
        dbg!(row);
    }
}

#[test]
fn sparsemat_insert() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    local.insert(0, 0, 1.0);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn sparsemat_insert_oob() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    local.insert(4, 4, 1.0);
}

#[test]
fn sparsemat_insert_triplets() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    local.insert_triplets(vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
    assert!(local.peek_at(0, 0) == Some(1.0));
    assert!(local.peek_at(1, 1) == Some(2.0));
    assert!(local.peek_at(2, 2) == Some(3.0));
}

#[test]
fn sparsemat_remove() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);
    local.insert(0, 0, 1.0);

    let output = local.clear_at(0, 0);
    assert!(output == Some(1.0));

    let output = local.clear_at(0, 0);
    assert!(output.is_none());

    let output = local.clear_at(1, 1);
    assert!(output.is_none());
}

#[test]
#[should_panic(expected = "assertion failed")]
fn sparsemat_remove_oob() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);

    let _ = local.clear_at(4, 4);
}

#[test]
fn sparsemat_peek() {
    let mut local = SparseMatrix::empty_with_shape(3, 3);

    local.insert_triplets(vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);

    assert!(local.peek_at(0, 0) == Some(1.0));
    assert!(local.peek_at(0, 1).is_none());
}

#[test]
fn sparsemat_transposeinplace() {
    let mut local = SparseMatrix::empty_with_shape(4, 4);

    local.insert_triplets(vec![(0, 0, 1.0), (1, 0, 2.0), (2, 2, 3.0), (2, 3, 4.0)]);
    local.transpose_inplace();

    assert!(local.peek_at(0, 0) == Some(1.0));
    assert!(local.peek_at(0, 1) == Some(2.0));
    assert!(local.peek_at(1, 0).is_none());
    assert!(local.peek_at(2, 2) == Some(3.0));
    assert!(local.peek_at(3, 2) == Some(4.0));
    assert!(local.peek_at(2, 3).is_none());
}

#[test]
fn sparsemat_compressedrepr() {
    let mut local = SparseMatrix::empty_with_shape(4, 4);
    local.insert_triplets(vec![(0, 0, 5.0), (1, 1, 8.0), (3, 1, 6.0), (2, 2, 3.0)]);
    local.explicitly_compress();
    assert!(local.compressed_dataarray == vec![5.0, 8.0, 3.0, 6.0]);
    assert!(local.compressed_colarray == vec![0, 1, 2, 1]);
    assert!(local.compressed_rowarray == vec![0, 1, 2, 3, 4]);

    let mut local2 = SparseMatrix::empty_with_shape(4, 6);
    local2.insert_triplets(vec![
        (0, 0, 10.0),
        (0, 1, 20.0),
        (1, 1, 30.0),
        (2, 2, 50.0),
        (1, 3, 40.0),
        (2, 3, 60.0),
        (2, 4, 70.0),
        (3, 5, 80.0),
    ]);
    local2.explicitly_compress();
    assert!(local2.compressed_dataarray == vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    assert!(local2.compressed_colarray == vec![0, 1, 1, 3, 2, 3, 4, 5]);
    assert!(local2.compressed_rowarray == vec![0, 2, 4, 7, 8]);
}

#[test]
fn sparsemat_display() {
    let mut local = SparseMatrix::empty_with_shape(4, 6);
    local.insert_triplets(vec![
        (0, 0, 10.0),
        (0, 1, 20.0),
        (1, 1, 30.0),
        (2, 2, 50.0),
        (1, 3, 40.0),
        (2, 3, 60.0),
        (2, 4, 70.0),
        (3, 5, 80.0),
    ]);
    local.explicitly_compress();
    println!("{}", local)
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
    local.insert_triplets(vec![(0, 0, 10.0), (0, 1, 20.0), (1, 1, 30.0), (2, 2, 50.0)]);
    let local2 = local.create_transpose();

    let local3 = &local + &local2;
    assert!(local3.peek_at(0, 0) == Some(20.0));
    assert!(local3.peek_at(0, 1) == Some(20.0));
    assert!(local3.peek_at(1, 0) == Some(20.0));
    assert!(local3.peek_at(1, 1) == Some(60.0));
    assert!(local3.peek_at(2, 2) == Some(100.0));
}
