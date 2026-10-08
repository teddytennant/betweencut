//! BetweenCut (arXiv:2610.10075), Algorithms 1, 2, 3, and 6, plus the
//! Theorem 5.5 parameter map.
//!
//! Height is the number of levels: `h = max_depth + 1`, so a chain of `h`
//! nodes has height `h` and BetweenCut walks `DepO(0, h - 1)`. Nodes at one
//! depth are visited in increasing index order. The child list is the public
//! child order.

/// Stored label. Every node is Heavy or Light when BetweenCut returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Label {
    Light,
    Heavy,
}

/// One answer from GuardedBT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    Light,
    Heavy,
    Between,
}

/// A node at which GuardedBT was invoked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Query {
    pub node: usize,
    pub count: f64,
    pub answer: Answer,
    /// False when a deterministic guard fired. The noise closure is not called.
    pub drew_noise: bool,
}

/// Theorem 5.5 parameters for a tree of height `h` and privacy `(epsilon, delta)`.
///
/// The paper assumes `h >= 1`, `0 < epsilon <= 1`, and `0 < delta < 1/10`.
/// The formulas are evaluated as written for any positive `epsilon` and `delta`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    /// `L = ceil(log2(h + 1))`.
    pub l: u32,
    /// `r0 = ceil(ln(2/delta) / ln(5/4))`.
    pub r0: u32,
    /// `r = min(L, r0)`.
    pub r: u32,
    /// `lambda = epsilon / r`.
    pub lambda: f64,
    /// `w = 1/lambda = r / epsilon`.
    pub w: f64,
    /// Guard margin `R = 1 + (r/epsilon) * ln(L/delta)`.
    pub big_r: f64,
    /// Additive margin `E = w + R`.
    pub e: f64,
}

/// Result of one BetweenCut run.
#[derive(Clone, Debug, PartialEq)]
pub struct Cut {
    /// Label of every node, index-aligned with the tree.
    pub labels: Vec<Label>,
    /// Nodes labeled Heavy, in increasing index order.
    pub heavy: Vec<usize>,
    /// GuardedBT invocations, in the order they happened.
    pub queries: Vec<Query>,
    /// Opposite-label overwrites inside MarkUp / MarkDown. Zero on the
    /// monotone instances the paper analyzes.
    pub conflicts: u32,
    /// Times the noise closure was called.
    pub noise_draws: u32,
    /// Nodes in the order they first received a label. MarkDown appends children
    /// in public child order.
    pub first_label: Vec<usize>,
    /// `w = 1/lambda`, `a = tau - w`, `b = tau + w`.
    pub w: f64,
    pub a: f64,
    pub b: f64,
}

/// Rooted tree. Node 0 is the root. `children[u]` is the public child order.
#[derive(Clone, Debug)]
pub struct Tree {
    parent: Vec<Option<usize>>,
    children: Vec<Vec<usize>>,
    depth: Vec<usize>,
    height: usize,
}

/// Seeded xorshift64. Used only to draw the uniform that the Laplace inverse
/// CDF consumes. Not a named statistical RNG.
#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// `seed == 0` is replaced so the shift register is never stuck at zero.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform on `(0, 1)`, never 0 or 1, so `ln` in the Laplace inverse CDF is finite.
    pub fn uniform_open(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    /// One draw of `Lap(1/lambda)`, scale `1/lambda`.
    pub fn laplace(&mut self, lambda: f64) -> f64 {
        laplace_from_uniform(self.uniform_open(), lambda)
    }
}

/// Inverse CDF of `Lap(1/lambda)` at `u` in `(0, 1)`.
///
/// `Z = ln(2U) / lambda` when `U < 1/2`, else `-ln(2(1-U)) / lambda`.
pub fn laplace_from_uniform(u: f64, lambda: f64) -> f64 {
    assert!(u > 0.0 && u < 1.0, "uniform must lie in (0, 1)");
    assert!(lambda > 0.0, "lambda must be positive");
    if u < 0.5 {
        (2.0 * u).ln() / lambda
    } else {
        -(2.0 * (1.0 - u)).ln() / lambda
    }
}

/// `ceil(log2(n))` for `n >= 1`. `n == 1` returns 0.
pub fn ceil_log2(n: u32) -> u32 {
    assert!(n >= 1);
    if n == 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

fn ceil_pos(x: f64) -> u32 {
    assert!(
        x.is_finite() && x > 0.0,
        "ceil input must be finite and positive"
    );
    let nearest = x.round();
    if (x - nearest).abs() < 1e-9 {
        nearest as u32
    } else {
        x.ceil() as u32
    }
}

/// Theorem 5.5 calibration.
pub fn calibrate(height: u32, epsilon: f64, delta: f64) -> Calibration {
    assert!(height >= 1, "height must be at least 1");
    assert!(epsilon > 0.0, "epsilon must be positive");
    assert!(delta > 0.0 && delta < 1.0, "delta must lie in (0, 1)");
    let l = ceil_log2(height + 1);
    let ratio = (2.0 / delta).ln() / (1.25_f64).ln();
    let r0 = ceil_pos(ratio);
    let r = l.min(r0);
    assert!(r >= 1, "r must be at least 1");
    let lambda = epsilon / f64::from(r);
    let w = f64::from(r) / epsilon;
    let big_r = 1.0 + (f64::from(r) / epsilon) * (f64::from(l) / delta).ln();
    let e = w + big_r;
    Calibration {
        l,
        r0,
        r,
        lambda,
        w,
        big_r,
        e,
    }
}

/// Depth traversal order, Algorithm 2.
///
/// `m = i + floor((j - i + 1) / 2)`, then `[m] ++ DepO(m+1, j) ++ DepO(i, m-1)`.
/// `i` and `j` are depths and should be non-negative. Integer division is
/// toward zero, which matches floor on that range.
pub fn depo(i: i64, j: i64) -> Vec<i64> {
    if i > j {
        return Vec::new();
    }
    let m = i + (j - i + 1) / 2;
    let mut out = Vec::with_capacity((j - i + 1) as usize);
    out.push(m);
    out.extend(depo(m + 1, j));
    out.extend(depo(i, m - 1));
    out
}

/// Algorithm 1. `noise` returns a fresh `Z`, already on the Laplace scale.
/// It is called only when `c` is strictly inside `(a - R, b + R)`.
///
/// Guards are non-strict (`c <= a - R`, `c >= b + R`). The noisy comparisons
/// are strict (`c + Z < a`, `c + Z > b`). Equality with `a` or `b` is Between.
pub fn guarded_bt<F>(c: f64, a: f64, b: f64, big_r: f64, noise: &mut F) -> (Answer, bool)
where
    F: FnMut() -> f64,
{
    if c <= a - big_r {
        (Answer::Light, false)
    } else if c >= b + big_r {
        (Answer::Heavy, false)
    } else {
        let z = noise();
        let noisy = c + z;
        let answer = if noisy < a {
            Answer::Light
        } else if noisy > b {
            Answer::Heavy
        } else {
            Answer::Between
        };
        (answer, true)
    }
}

impl Tree {
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// `max_depth + 1`.
    pub fn height(&self) -> usize {
        self.height
    }

    pub fn parent(&self, u: usize) -> Option<usize> {
        self.parent[u]
    }

    pub fn children(&self, u: usize) -> &[usize] {
        &self.children[u]
    }

    pub fn depth(&self, u: usize) -> usize {
        self.depth[u]
    }

    /// Nodes at depth `d`, increasing index order. That is the public order
    /// BetweenCut uses inside a depth.
    pub fn nodes_at_depth(&self, d: usize) -> Vec<usize> {
        (0..self.len()).filter(|&u| self.depth[u] == d).collect()
    }

    /// Chain of `height` nodes. Node `i` is the parent of `i + 1`.
    pub fn chain(height: usize) -> Self {
        assert!(height >= 1);
        let mut children = vec![Vec::new(); height];
        for i in 0..height - 1 {
            children[i].push(i + 1);
        }
        Self::from_children(children)
    }

    /// Complete binary tree with `height` levels (depth `0..height-1`).
    /// Children are left then right, which is also increasing index order.
    pub fn complete_binary(height: usize) -> Self {
        assert!(height >= 1);
        assert!(height <= 24, "complete binary tree would be enormous");
        let n = (1usize << height) - 1;
        let mut children = vec![Vec::new(); n];
        for i in 0..n {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < n {
                children[i].push(left);
            }
            if right < n {
                children[i].push(right);
            }
        }
        Self::from_children(children)
    }

    /// Spine of `height` nodes. Each non-leaf spine node also has one leaf child.
    /// Public child order is spine child, then the extra leaf.
    pub fn skewed(height: usize) -> Self {
        assert!(height >= 1);
        let n = if height == 1 { 1 } else { 2 * height - 1 };
        let mut children = vec![Vec::new(); n];
        for i in 0..height - 1 {
            children[i].push(i + 1);
            children[i].push(height + i);
        }
        Self::from_children(children)
    }

    /// `children[u]` is the public child order. Node 0 is the root. Every other
    /// node appears as a child exactly once.
    pub fn from_children(children: Vec<Vec<usize>>) -> Self {
        let n = children.len();
        assert!(n >= 1, "tree must have a root");
        let mut parent = vec![None; n];
        let mut seen = vec![false; n];
        seen[0] = true;
        for (u, kids) in children.iter().enumerate() {
            for &v in kids {
                assert!(v < n && v != u, "child index out of range");
                assert!(parent[v].is_none(), "node has two parents");
                parent[v] = Some(u);
                seen[v] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "unreachable node");
        let mut depth = vec![0usize; n];
        let mut stack = vec![0usize];
        let mut visited = 0usize;
        let mut guard = 0usize;
        while let Some(u) = stack.pop() {
            visited += 1;
            guard += 1;
            assert!(guard <= n, "cycle in child lists");
            for &v in children[u].iter().rev() {
                depth[v] = depth[u] + 1;
                stack.push(v);
            }
        }
        assert_eq!(visited, n, "cycle or disconnected child lists");
        let height = depth.iter().copied().max().unwrap() + 1;
        Self {
            parent,
            children,
            depth,
            height,
        }
    }

    /// Parent of node 0 is `None`. Every other parent is `Some`. Children are
    /// stored in increasing index order.
    pub fn from_parents(parent: Vec<Option<usize>>) -> Self {
        let n = parent.len();
        assert!(n >= 1, "tree must have a root");
        assert!(parent[0].is_none(), "node 0 is the root");
        let mut children = vec![Vec::new(); n];
        for (u, p) in parent.iter().enumerate().skip(1) {
            let p = p.expect("non-root has a parent");
            assert!(p < n && p != u, "parent index out of range");
            children[p].push(u);
        }
        Self::from_children(children)
    }

    /// Subtree counts. Each record is a node id (a leaf, in the paper). One
    /// occurrence increments that node and every ancestor.
    pub fn subtree_counts(&self, records: &[usize]) -> Vec<f64> {
        let mut counts = vec![0.0; self.len()];
        for &rec in records {
            assert!(rec < self.len(), "record is not a node id");
            let mut u = Some(rec);
            while let Some(x) = u {
                counts[x] += 1.0;
                u = self.parent[x];
            }
        }
        counts
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Slot {
    Unknown,
    Light,
    Heavy,
}

fn mark_up(
    tree: &Tree,
    label: &mut [Slot],
    mut u: Option<usize>,
    conflicts: &mut u32,
    first_label: &mut Vec<usize>,
) {
    while let Some(x) = u {
        if label[x] == Slot::Heavy {
            break;
        }
        if label[x] == Slot::Light {
            *conflicts += 1;
        }
        if label[x] == Slot::Unknown {
            first_label.push(x);
        }
        label[x] = Slot::Heavy;
        u = tree.parent[x];
    }
}

fn mark_down(
    tree: &Tree,
    label: &mut [Slot],
    u: usize,
    conflicts: &mut u32,
    first_label: &mut Vec<usize>,
) {
    if label[u] == Slot::Light {
        return;
    }
    if label[u] == Slot::Heavy {
        *conflicts += 1;
    }
    if label[u] == Slot::Unknown {
        first_label.push(u);
    }
    label[u] = Slot::Light;
    for &v in &tree.children[u] {
        mark_down(tree, label, v, conflicts, first_label);
    }
}

/// Algorithm 3. `records` is the private leaf dataset (node ids, with
/// multiplicity). `noise` returns `Z`, not a uniform. Fresh noise on every
/// randomized GuardedBT call.
pub fn between_cut<F>(
    tree: &Tree,
    records: &[usize],
    tau: f64,
    lambda: f64,
    big_r: f64,
    noise: &mut F,
) -> Cut
where
    F: FnMut() -> f64,
{
    let counts = tree.subtree_counts(records);
    between_cut_counts(tree, &counts, tau, lambda, big_r, noise)
}

/// Algorithm 3 on precomputed node counts. Counts should be subtree sums for
/// the paper's monotonicity claims. The routine itself does not check that.
pub fn between_cut_counts<F>(
    tree: &Tree,
    counts: &[f64],
    tau: f64,
    lambda: f64,
    big_r: f64,
    noise: &mut F,
) -> Cut
where
    F: FnMut() -> f64,
{
    assert_eq!(counts.len(), tree.len(), "one count per node");
    assert!(lambda > 0.0, "lambda must be positive");
    assert!(tau.is_finite() && big_r.is_finite());
    let w = 1.0 / lambda;
    let a = tau - w;
    let b = tau + w;
    let mut label = vec![Slot::Unknown; tree.len()];
    let mut queries = Vec::new();
    let mut conflicts = 0u32;
    let mut noise_draws = 0u32;
    let mut first_label = Vec::new();
    let mut draws = || {
        noise_draws += 1;
        noise()
    };

    for d in depo(0, tree.height() as i64 - 1) {
        let depth = d as usize;
        for u in tree.nodes_at_depth(depth) {
            if label[u] != Slot::Unknown {
                continue;
            }
            let c = counts[u];
            let (answer, drew) = guarded_bt(c, a, b, big_r, &mut draws);
            queries.push(Query {
                node: u,
                count: c,
                answer,
                drew_noise: drew,
            });
            match answer {
                Answer::Light => mark_down(tree, &mut label, u, &mut conflicts, &mut first_label),
                Answer::Heavy => {
                    mark_up(tree, &mut label, Some(u), &mut conflicts, &mut first_label)
                }
                Answer::Between => {
                    mark_up(
                        tree,
                        &mut label,
                        tree.parent[u],
                        &mut conflicts,
                        &mut first_label,
                    );
                    mark_down(tree, &mut label, u, &mut conflicts, &mut first_label);
                }
            }
        }
    }

    let labels: Vec<Label> = label
        .into_iter()
        .enumerate()
        .map(|(u, slot)| match slot {
            Slot::Heavy => Label::Heavy,
            Slot::Light => Label::Light,
            Slot::Unknown => panic!("node {u} was never labeled"),
        })
        .collect();
    let heavy = labels
        .iter()
        .enumerate()
        .filter(|(_, lab)| **lab == Label::Heavy)
        .map(|(u, _)| u)
        .collect();
    Cut {
        labels,
        heavy,
        queries,
        conflicts,
        noise_draws,
        first_label,
        w,
        a,
        b,
    }
}
