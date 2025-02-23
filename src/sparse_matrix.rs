use std::collections::HashMap;

/* Starting with Dictionary of Keys impl. To support efficient operations,
     should eventually move to compressed sparse row/col
*/
struct SparseMatrix {
    shape: (u64, u64),
    values: HashMap<(u64, u64), f64>,
}

impl SparseMatrix {
    #[allow(dead_code)]
    fn new() -> SparseMatrix {
        SparseMatrix {
            shape: (0, 0),
            values: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    fn empty_with_shape(n: u64, m: u64) -> SparseMatrix {
        let mut value_map = HashMap::new();
        value_map.reserve((n * m / 4) as usize);
        SparseMatrix {
            shape: (n, m),
            values: value_map,
        }
    }

    #[allow(dead_code)]
    fn identity(n: u64) -> SparseMatrix {
        let mut local = SparseMatrix::empty_with_shape(n, n);
        for diag_idx in 0..n {
            local.insert(diag_idx, diag_idx, 1.0);
        }
        local
    }

    #[allow(dead_code)]
    fn insert(&mut self, row: u64, col: u64, value: f64) {
        // TODO: return result with oob error instead
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.values.insert((row, col), value);
    }

    #[allow(dead_code)]
    fn insert_triplets(&mut self, triplets: Vec<(u64, u64, f64)>) {
        for (row, col, val) in triplets.iter() {
            assert!(*row < self.shape.0);
            assert!(*col < self.shape.1);

            self.values.insert((*row, *col), *val);
        }
    }

    #[allow(dead_code)]
    fn clear_at(&mut self, row: u64, col: u64) -> Option<f64> {
        // TODO: return result with oob error instead
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.values.remove(&(row, col))
    }

    #[allow(dead_code)]
    fn peek_at(&self, row: u64, col: u64) -> Option<f64> {
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.values.get(&(row, col)).copied()
    }

    #[allow(dead_code)]
    fn num_nonzero(&self) -> u64 {
        self.values.len() as u64
    }

    #[allow(dead_code)]
    fn transpose(&mut self) {
        // Naive impl, could do better
        self.shape = (self.shape.1, self.shape.0);

        let triplets: Vec<((u64, u64), f64)> = self.values.drain().collect();

        for ((row, col), val) in triplets {
            self.values.insert((col, row), val);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn sparsemat_creation() {
        let _local = sparse_matrix::SparseMatrix::new();
        let _local2 = sparse_matrix::SparseMatrix::empty_with_shape(3, 3);
    }

    #[test]
    fn sparsemat_identity_creation() {
        let local = sparse_matrix::SparseMatrix::identity(3);
        assert!(local.num_nonzero() == 3);
        assert!(local.peek_at(0, 0) == Some(1.0));
        assert!(local.peek_at(1, 1) == Some(1.0));
        assert!(local.peek_at(2, 2) == Some(1.0));
    }

    #[test]
    fn sparsemat_insert() {
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(3, 3);
        local.insert(0, 0, 1.0);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn sparsemat_insert_oob() {
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(3, 3);
        local.insert(4, 4, 1.0);
    }

    #[test]
    fn sparsemat_insert_triplets() {
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(3, 3);
        local.insert_triplets(vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);
        assert!(local.peek_at(0, 0) == Some(1.0));
        assert!(local.peek_at(1, 1) == Some(2.0));
        assert!(local.peek_at(2, 2) == Some(3.0));
    }

    #[test]
    fn sparsemat_remove() {
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(3, 3);
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
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(3, 3);

        let _ = local.clear_at(4, 4);
    }

    #[test]
    fn sparsemat_peek() {
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(3, 3);

        local.insert_triplets(vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);

        assert!(local.peek_at(0, 0) == Some(1.0));
        assert!(local.peek_at(0, 1).is_none());
    }

    #[test]
    fn sparsemat_transpose() {
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(4, 4);

        local.insert_triplets(vec![(0, 0, 1.0), (1, 0, 2.0), (2, 2, 3.0), (2, 3, 4.0)]);
        local.transpose();

        assert!(local.peek_at(0, 0) == Some(1.0));
        assert!(local.peek_at(0, 1) == Some(2.0));
        assert!(local.peek_at(1, 0).is_none());
        assert!(local.peek_at(2, 2) == Some(3.0));
        assert!(local.peek_at(3, 2) == Some(4.0));
        assert!(local.peek_at(2, 3).is_none());
    }
}
