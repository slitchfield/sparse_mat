use std::collections::HashMap;

/* Starting with Dictionary of Keys impl. To support efficient operations,
     should eventually move to compressed sparse row/col
*/
#[derive(Clone)]
struct SparseMatrix {
    shape: (u64, u64),
    values: HashMap<(u64, u64), f64>,

    compressed_updated: bool,
    compressed_rowarray: Vec<u64>,
    compressed_colarray: Vec<u64>,
    compressed_dataarray: Vec<f64>,

    #[allow(dead_code)]
    row_iter_idx: usize,
}

struct RowIterator<'a> {
    matrix: &'a SparseMatrix,
    row_iter_idx: usize,
}

impl Iterator for RowIterator<'_> {
    // Iterate by rows
    type Item = Vec<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check for compressed updates here?
        if self.row_iter_idx < self.matrix.shape.0 as usize {
            let start = self.matrix.compressed_rowarray[self.row_iter_idx] as usize;
            let end = self.matrix.compressed_rowarray[self.row_iter_idx + 1] as usize;

            let colslice = &self.matrix.compressed_colarray[start..end];
            let dataslice = &self.matrix.compressed_dataarray[start..end];

            let mut retvec: Vec<f64> = vec![0.0; self.matrix.shape.1 as usize];

            for (col, val) in std::iter::zip(colslice, dataslice) {
                retvec[*col as usize] = *val;
            }

            self.row_iter_idx += 1;

            Some(retvec)
        } else {
            None
        }
    }
}

impl SparseMatrix {
    fn _update_compressed(&mut self) {
        self.compressed_rowarray.clear();
        self.compressed_colarray.clear();
        self.compressed_dataarray.clear();

        // Create row vecs that we'll sort by col
        let mut row_vecs: Vec<Vec<(u64, f64)>> = vec![];
        for _ in 0..self.shape.0 {
            row_vecs.push(vec![]);
        }

        for ((row, col), val) in self.values.iter() {
            row_vecs[*row as usize].push((*col, *val));
        }
        for rowidx in 0..self.shape.0 {
            row_vecs[rowidx as usize].sort_by(|a, b| a.0.cmp(&b.0));
        }

        self.compressed_rowarray.push(0);
        for row in row_vecs {
            for (col, val) in row {
                self.compressed_colarray.push(col);
                self.compressed_dataarray.push(val);
            }
            self.compressed_rowarray
                .push(self.compressed_dataarray.len() as u64);
        }

        self.compressed_updated = true
    }

    #[allow(dead_code)]
    fn row_iter(&self) -> RowIterator {
        RowIterator {
            matrix: self,
            row_iter_idx: 0,
        }
    }

    #[allow(dead_code)]
    fn new() -> SparseMatrix {
        SparseMatrix {
            shape: (0, 0),
            values: HashMap::new(),
            compressed_updated: false,
            compressed_rowarray: vec![],
            compressed_colarray: vec![],
            compressed_dataarray: vec![],
            row_iter_idx: 0,
        }
    }

    #[allow(dead_code)]
    fn empty_with_shape(n: u64, m: u64) -> SparseMatrix {
        let mut value_map = HashMap::new();
        // TODO: evaluate expected sparsity, add reservation for compressed reps
        value_map.reserve((n * m / 4) as usize);
        SparseMatrix {
            shape: (n, m),
            values: value_map,
            compressed_updated: false,
            compressed_rowarray: vec![],
            compressed_colarray: vec![],
            compressed_dataarray: vec![],
            row_iter_idx: 0,
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
    fn create_transpose(&self) -> SparseMatrix {
        let mut local = SparseMatrix::empty_with_shape(self.shape.1, self.shape.0);
        for ((row, col), val) in self.values.iter() {
            local.insert(*col, *row, *val); // Deref okay due to elementary r, c, v types
        }
        local
    }

    #[allow(dead_code)]
    fn insert(&mut self, row: u64, col: u64, value: f64) {
        // TODO: return result with oob error instead
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.values.insert((row, col), value);
        self.compressed_updated = false;
    }

    #[allow(dead_code)]
    fn insert_triplets(&mut self, triplets: Vec<(u64, u64, f64)>) {
        for (row, col, val) in triplets.iter() {
            assert!(*row < self.shape.0);
            assert!(*col < self.shape.1);

            self.values.insert((*row, *col), *val);
        }
        self.compressed_updated = false;
    }

    #[allow(dead_code)]
    fn clear_at(&mut self, row: u64, col: u64) -> Option<f64> {
        // TODO: return result with oob error instead
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.compressed_updated = false;
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
    fn transpose_inplace(&mut self) {
        // Naive impl, could do better
        self.shape = (self.shape.1, self.shape.0);

        let triplets: Vec<((u64, u64), f64)> = self.values.drain().collect();

        for ((row, col), val) in triplets {
            self.values.insert((col, row), val);
        }
        self.compressed_updated = false;
    }
}

use std::fmt;
impl fmt::Display for SparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let inner_line_width = 8 * self.shape.1; // 6 chars per col + comma + space + leading space
        write!(f, "\t/")?;
        for _ in 0..inner_line_width {
            write!(f, " ")?;
        }
        writeln!(f, "\\")?;
        // TODO: Account for variable number of digits in cols
        for row in self.row_iter() {
            write!(f, "\t| ")?;
            for (idx, elem) in row.iter().enumerate() {
                if idx != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:>6.2}", elem)?; // TODO: dynamic precision based on longest values
            }
            writeln!(f, " |")?;
        }
        write!(f, "\t\\")?;
        for _ in 0..inner_line_width {
            write!(f, " ")?;
        }
        write!(f, "/")?;
        writeln!(f)
    }
}

use std::ops::Add;

impl Add for &SparseMatrix {
    type Output = SparseMatrix;

    fn add(self, other: &SparseMatrix) -> SparseMatrix {
        assert!(self.shape == other.shape);
        let mut local = self.clone();

        for ((rother, cother), elemother) in other.values.iter() {
            let existingval = local.peek_at(*rother, *cother).unwrap_or(0.0);
            local.insert(*rother, *cother, existingval + *elemother);
        }
        local
    }
}

#[cfg(test)]
mod tests {
    use sparse_matrix::SparseMatrix;

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
    fn sparsemat_transpose_creation() {
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(4, 6);
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
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(4, 6);
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

        local._update_compressed();

        for row in local.row_iter() {
            dbg!(row);
        }
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
    fn sparsemat_transposeinplace() {
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(4, 4);

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
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(4, 4);
        local.insert_triplets(vec![(0, 0, 5.0), (1, 1, 8.0), (3, 1, 6.0), (2, 2, 3.0)]);
        local._update_compressed();
        assert!(local.compressed_dataarray == vec![5.0, 8.0, 3.0, 6.0]);
        assert!(local.compressed_colarray == vec![0, 1, 2, 1]);
        assert!(local.compressed_rowarray == vec![0, 1, 2, 3, 4]);

        let mut local2 = sparse_matrix::SparseMatrix::empty_with_shape(4, 6);
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
        local2._update_compressed();
        assert!(
            local2.compressed_dataarray == vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        );
        assert!(local2.compressed_colarray == vec![0, 1, 1, 3, 2, 3, 4, 5]);
        assert!(local2.compressed_rowarray == vec![0, 2, 4, 7, 8]);
    }

    #[test]
    fn sparsemat_display() {
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(4, 6);
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
        local._update_compressed();
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
        let mut local = sparse_matrix::SparseMatrix::empty_with_shape(3, 3);
        local.insert_triplets(vec![(0, 0, 10.0), (0, 1, 20.0), (1, 1, 30.0), (2, 2, 50.0)]);
        let local2 = local.create_transpose();

        let local3 = &local + &local2;
        assert!(local3.peek_at(0, 0) == Some(20.0));
        assert!(local3.peek_at(0, 1) == Some(20.0));
        assert!(local3.peek_at(1, 0) == Some(20.0));
        assert!(local3.peek_at(1, 1) == Some(60.0));
        assert!(local3.peek_at(2, 2) == Some(100.0));
    }
}
