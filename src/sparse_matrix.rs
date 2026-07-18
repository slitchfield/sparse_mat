use std::collections::HashMap;

/* Starting with Dictionary of Keys impl. To support efficient operations,
     should eventually move to compressed sparse row/col
*/
#[derive(Clone)]
pub struct SparseMatrix {
    pub shape: (u64, u64),
    values: HashMap<(u64, u64), f64>,

    compressed_updated: bool,
    pub compressed_rowarray: Vec<u64>,
    pub compressed_colarray: Vec<u64>,
    pub compressed_dataarray: Vec<f64>,

    #[allow(dead_code)]
    row_iter_idx: usize,
}

pub struct RowIterator<'a> {
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

impl Default for SparseMatrix {
    fn default() -> Self {
        Self::new()
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
            row_vecs[rowidx as usize].sort_by_key(|a| a.0);
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

    pub fn explicitly_compress(&mut self) {
        self._update_compressed();
    }

    #[allow(dead_code)]
    pub fn row_iter(&self) -> RowIterator<'_> {
        RowIterator {
            matrix: self,
            row_iter_idx: 0,
        }
    }

    #[allow(dead_code)]
    pub fn new() -> SparseMatrix {
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
    pub fn empty_with_shape(n: u64, m: u64) -> SparseMatrix {
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
    pub fn identity(n: u64) -> SparseMatrix {
        let mut local = SparseMatrix::empty_with_shape(n, n);
        for diag_idx in 0..n {
            local.insert(diag_idx, diag_idx, 1.0);
        }
        local
    }

    #[allow(dead_code)]
    pub fn create_transpose(&self) -> SparseMatrix {
        let mut local = SparseMatrix::empty_with_shape(self.shape.1, self.shape.0);
        for ((row, col), val) in self.values.iter() {
            local.insert(*col, *row, *val); // Deref okay due to elementary r, c, v types
        }
        local
    }

    #[allow(dead_code)]
    pub fn insert(&mut self, row: u64, col: u64, value: f64) {
        // TODO: return result with oob error instead
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.values.insert((row, col), value);
        self.compressed_updated = false;
    }

    #[allow(dead_code)]
    pub fn insert_triplets(&mut self, triplets: Vec<(u64, u64, f64)>) {
        for (row, col, val) in triplets.iter() {
            assert!(*row < self.shape.0);
            assert!(*col < self.shape.1);

            self.values.insert((*row, *col), *val);
        }
        self.compressed_updated = false;
    }

    #[allow(dead_code)]
    pub fn clear_at(&mut self, row: u64, col: u64) -> Option<f64> {
        // TODO: return result with oob error instead
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.compressed_updated = false;
        self.values.remove(&(row, col))
    }

    #[allow(dead_code)]
    pub fn peek_at(&self, row: u64, col: u64) -> Option<f64> {
        assert!(row < self.shape.0);
        assert!(col < self.shape.1);

        self.values.get(&(row, col)).copied()
    }

    #[allow(dead_code)]
    pub fn num_nonzero(&self) -> u64 {
        self.values.len() as u64
    }

    #[allow(dead_code)]
    pub fn transpose_inplace(&mut self) {
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
