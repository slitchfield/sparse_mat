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
    fn insert(&mut self, row: u64, col: u64, value: f64) {
        // TODO: return result with oob error instead
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.values.insert((row, col), value);
    }

    #[allow(dead_code)]
    fn clear_at(&mut self, row: u64, col: u64) -> Option<f64> {
        // TODO: return result with oob error instead
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.values.remove(&(row, col))
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
}
