use betweencut::{
    between_cut, between_cut_counts, calibrate, ceil_log2, depo, guarded_bt, laplace_from_uniform,
    Answer, Label, Tree, XorShift64,
};

fn leaves(tree: &Tree) -> Vec<usize> {
    (0..tree.len())
        .filter(|&u| tree.children(u).is_empty())
        .collect()
}

fn paths(tree: &Tree) -> Vec<Vec<usize>> {
    fn walk(tree: &Tree, u: usize, acc: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        acc.push(u);
        let kids = tree.children(u);
        if kids.is_empty() {
            out.push(acc.clone());
        } else {
            for &v in kids {
                walk(tree, v, acc, out);
            }
        }
        acc.pop();
    }
    let mut out = Vec::new();
    walk(tree, 0, &mut Vec::new(), &mut out);
    out
}

fn ancestor_closed(tree: &Tree, labels: &[Label]) {
    for u in 0..tree.len() {
        if labels[u] != Label::Heavy {
            continue;
        }
        let mut p = tree.parent(u);
        while let Some(x) = p {
            assert_eq!(
                labels[x],
                Label::Heavy,
                "heavy {u} has non-heavy ancestor {x}"
            );
            p = tree.parent(x);
        }
    }
}

fn descendant_closed(tree: &Tree, labels: &[Label]) {
    fn walk(tree: &Tree, u: usize, labels: &[Label]) {
        assert_eq!(
            labels[u],
            Label::Light,
            "light ancestor of {u} is not light"
        );
        for &v in tree.children(u) {
            walk(tree, v, labels);
        }
    }
    for u in 0..tree.len() {
        if labels[u] == Label::Light {
            walk(tree, u, labels);
        }
    }
}

fn is_ancestor(tree: &Tree, anc: usize, node: usize) -> bool {
    let mut p = tree.parent(node);
    while let Some(x) = p {
        if x == anc {
            return true;
        }
        p = tree.parent(x);
    }
    false
}

#[test]
fn depo_paper_example_and_cover() {
    assert_eq!(
        depo(0, 14),
        vec![7, 11, 13, 14, 12, 9, 10, 8, 3, 5, 6, 4, 1, 2, 0]
    );
    assert!(depo(3, 2).is_empty());
    assert_eq!(depo(4, 4), vec![4]);
    for h in 1..40 {
        let order = depo(0, h - 1);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        let expect: Vec<i64> = (0..h).collect();
        assert_eq!(sorted, expect, "DepO misses a depth at h={h}");
        assert_eq!(order.len(), h as usize);
    }
}

#[test]
fn guarded_bt_guards_and_ties() {
    let mut calls = 0u32;
    let mut noise = || {
        calls += 1;
        0.0
    };
    // c == a - R is Light, and the sampler is not called.
    let (ans, drew) = guarded_bt(0.0, 1.0, 3.0, 1.0, &mut noise);
    assert_eq!(ans, Answer::Light);
    assert!(!drew);
    // c == b + R is Heavy, still no sample.
    let (ans, drew) = guarded_bt(4.0, 1.0, 3.0, 1.0, &mut noise);
    assert_eq!(ans, Answer::Heavy);
    assert!(!drew);
    assert_eq!(calls, 0);

    // Strictly inside. c + Z == a and c + Z == b are Between.
    let cases = [
        (-0.5, Answer::Between),
        (1.5, Answer::Between),
        (-0.5000001, Answer::Light),
        (1.5000001, Answer::Heavy),
    ];
    for (z, expect) in cases {
        let mut noise = || z;
        let (ans, drew) = guarded_bt(1.5, 1.0, 3.0, 1.0, &mut noise);
        assert!(drew);
        assert_eq!(ans, expect);
    }
}

#[test]
fn laplace_inverse_cdf_and_seed() {
    let ln2 = 2.0_f64.ln();
    assert!((laplace_from_uniform(0.25, 2.0) - (-ln2 / 2.0)).abs() < 1e-12);
    assert!((laplace_from_uniform(0.75, 2.0) - (ln2 / 2.0)).abs() < 1e-12);
    assert_eq!(laplace_from_uniform(0.5, 4.0), 0.0);

    let mut a = XorShift64::new(1);
    let mut b = XorShift64::new(1);
    for _ in 0..20 {
        let u = a.uniform_open();
        assert!(u > 0.0 && u < 1.0);
        assert_eq!(u, b.uniform_open());
    }
    let mut rng = XorShift64::new(7);
    let z = rng.laplace(0.5);
    assert!(z.is_finite());
    assert_eq!(z, XorShift64::new(7).laplace(0.5));
    // Distinct seeds diverge.
    let mut c = XorShift64::new(2);
    assert_ne!(XorShift64::new(1).next_u64(), c.next_u64());
}

#[test]
fn calibration_identities() {
    assert_eq!(ceil_log2(16), 4);
    let cal = calibrate(15, 1.0, 0.05);
    assert_eq!(cal.l, 4);
    let ratio = (2.0 / 0.05_f64).ln() / (1.25_f64).ln();
    assert_eq!(cal.r0, ratio.ceil() as u32);
    assert_eq!(cal.r, cal.l.min(cal.r0));
    assert!((cal.lambda - 1.0 / f64::from(cal.r)).abs() < 1e-12);
    assert!((cal.w - 1.0 / cal.lambda).abs() < 1e-12);
    assert!((cal.w - f64::from(cal.r) / 1.0).abs() < 1e-12);
    let big_r = 1.0 + (f64::from(cal.r) / 1.0) * (f64::from(cal.l) / 0.05).ln();
    assert!((cal.big_r - big_r).abs() < 1e-12);
    assert!((cal.e - (cal.w + cal.big_r)).abs() < 1e-12);
    let e_closed = 1.0 + (f64::from(cal.r) / 1.0) * (1.0 + (f64::from(cal.l) / 0.05).ln());
    assert!((cal.e - e_closed).abs() < 1e-9);
    // r0 is the larger of the two at this height, so r = L = 4.
    assert_eq!(cal.r, 4);
    assert!(cal.r0 > cal.l);
}

#[test]
fn zero_noise_chain_is_a_cut() {
    let tree = Tree::chain(8);
    assert_eq!(tree.height(), 8);
    // One record at the leaf: every node has count 1.
    let mut noise = || 0.0;
    let cut = between_cut(&tree, &[7], 1.0, 1.0, 1.0, &mut noise);
    assert_eq!(cut.labels.len(), 8);
    assert!(cut
        .labels
        .iter()
        .all(|l| *l == Label::Heavy || *l == Label::Light));
    ancestor_closed(&tree, &cut.labels);
    descendant_closed(&tree, &cut.labels);
    assert_eq!(cut.conflicts, 0);
    assert_eq!(cut.heavy, heavy_of(&cut.labels));
}

fn heavy_of(labels: &[Label]) -> Vec<usize> {
    labels
        .iter()
        .enumerate()
        .filter(|(_, l)| **l == Label::Heavy)
        .map(|(u, _)| u)
        .collect()
}

#[test]
fn explicit_boundary_on_a_chain() {
    // counts 5,4,3,2,1. tau=3, lambda=1, R=0.5, noise 0.
    // DepO hits depth 2 first (count 3), which is Between, and that labels the cut.
    let tree = Tree::chain(5);
    let counts = [5.0, 4.0, 3.0, 2.0, 1.0];
    let mut noise = || 0.0;
    let cut = between_cut_counts(&tree, &counts, 3.0, 1.0, 0.5, &mut noise);
    assert_eq!(cut.conflicts, 0);
    assert_eq!(cut.queries.len(), 1);
    assert_eq!(cut.queries[0].node, 2);
    assert_eq!(cut.queries[0].answer, Answer::Between);
    assert!(cut.queries[0].drew_noise);
    assert_eq!(
        cut.labels,
        vec![
            Label::Heavy,
            Label::Heavy,
            Label::Light,
            Label::Light,
            Label::Light
        ]
    );
    ancestor_closed(&tree, &cut.labels);
    descendant_closed(&tree, &cut.labels);
}

#[test]
fn guards_do_not_draw_and_labeled_nodes_are_skipped() {
    let tree = Tree::complete_binary(4);
    // Spread counts so some nodes sit outside the band and some inside.
    let mut counts = vec![0.0; tree.len()];
    for u in 0..tree.len() {
        counts[u] = (tree.len() - u) as f64;
    }
    let tau = 8.0;
    let lambda = 1.0;
    let big_r = 1.5;
    let mut calls = 0u32;
    let mut noise = || {
        calls += 1;
        0.0
    };
    let cut = between_cut_counts(&tree, &counts, tau, lambda, big_r, &mut noise);
    assert_eq!(cut.noise_draws, calls);
    let inside = cut
        .queries
        .iter()
        .filter(|q| q.count > cut.a - big_r && q.count < cut.b + big_r)
        .count() as u32;
    assert_eq!(cut.noise_draws, inside);
    for q in &cut.queries {
        let outside = q.count <= cut.a - big_r || q.count >= cut.b + big_r;
        assert_eq!(q.drew_noise, !outside);
    }
    let mut seen = vec![false; tree.len()];
    for q in &cut.queries {
        assert!(!seen[q.node], "node queried twice");
        seen[q.node] = true;
    }
    // Replay the marks. A later query must still have been unlabeled.
    let mut marked = vec![false; tree.len()];
    for q in &cut.queries {
        assert!(
            !marked[q.node],
            "queried a node that propagation already labeled"
        );
        match q.answer {
            Answer::Light => mark_down_flag(&tree, &mut marked, q.node),
            Answer::Heavy => mark_up_flag(&tree, &mut marked, Some(q.node)),
            Answer::Between => {
                mark_up_flag(&tree, &mut marked, tree.parent(q.node));
                mark_down_flag(&tree, &mut marked, q.node);
            }
        }
    }
    assert!(cut
        .labels
        .iter()
        .all(|l| matches!(*l, Label::Heavy | Label::Light)));
    assert_eq!(cut.labels.len(), tree.len());
}

fn mark_up_flag(tree: &Tree, marked: &mut [bool], mut u: Option<usize>) {
    while let Some(x) = u {
        if marked[x] {
            break;
        }
        marked[x] = true;
        u = tree.parent(x);
    }
}

fn mark_down_flag(tree: &Tree, marked: &mut [bool], u: usize) {
    if marked[u] {
        return;
    }
    marked[u] = true;
    for &v in tree.children(u) {
        mark_down_flag(tree, marked, v);
    }
}

fn random_tree(seed: u64, height: usize) -> Tree {
    assert!(height >= 2);
    let mut rng = XorShift64::new(seed);
    // Spine fixes the height. Extra nodes attach to a random existing node
    // whose depth is less than height - 1, so the height does not grow.
    let extra = 4 + (rng.next_u64() % 8) as usize;
    let n = height + extra;
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..height - 1 {
        children[i].push(i + 1);
    }
    for v in height..n {
        let mut parent = (rng.next_u64() as usize) % (height - 1);
        // Prefer a shallow parent so the shape is not another chain.
        if rng.next_u64() % 3 == 0 {
            parent = 0;
        }
        children[parent].push(v);
    }
    Tree::from_children(children)
}

fn random_records(seed: u64, tree: &Tree) -> Vec<usize> {
    let mut rng = XorShift64::new(seed);
    let leafs = leaves(tree);
    let nrec = (rng.next_u64() % 12) as usize;
    (0..nrec)
        .map(|_| leafs[(rng.next_u64() as usize) % leafs.len()])
        .collect()
}

fn check_lemmas(
    tree: &Tree,
    records: &[usize],
    tau: f64,
    lambda: f64,
    big_r: f64,
    noise: &mut impl FnMut() -> f64,
) {
    let cut = between_cut(tree, records, tau, lambda, big_r, noise);
    assert_eq!(cut.labels.len(), tree.len());
    assert_eq!(cut.conflicts, 0, "monotone counts overwrote a label");
    ancestor_closed(tree, &cut.labels);
    descendant_closed(tree, &cut.labels);
    let bound = ceil_log2(tree.height() as u32 + 1) as usize;
    let qs: Vec<usize> = cut.queries.iter().map(|q| q.node).collect();
    for path in paths(tree) {
        let hits = qs.iter().filter(|&&u| path.contains(&u)).count();
        assert!(hits <= bound, "|Q ∩ P| = {hits} > {bound}");
        let bees = cut
            .queries
            .iter()
            .filter(|q| q.answer == Answer::Between && path.contains(&q.node))
            .count();
        assert!(bees <= 1, "two Between answers on one root-to-leaf path");
    }
    // A Between at u blocks every later query on a path through u.
    for (i, q) in cut.queries.iter().enumerate() {
        if q.answer != Answer::Between {
            continue;
        }
        for later in &cut.queries[i + 1..] {
            let related = later.node == q.node
                || is_ancestor(tree, q.node, later.node)
                || is_ancestor(tree, later.node, q.node);
            assert!(!related, "query after Between on a path through {}", q.node);
        }
    }
}

#[test]
fn lemmas_on_fixed_and_random_instances() {
    let fixed = [Tree::chain(9), Tree::complete_binary(6), Tree::skewed(7)];
    for tree in &fixed {
        let leafs = leaves(tree);
        let databases: Vec<Vec<usize>> = vec![
            vec![],
            leafs.clone(),
            leafs.iter().flat_map(|&u| [u, u, u]).collect(),
            vec![*leafs.last().unwrap()],
        ];
        for (i, records) in databases.iter().enumerate() {
            for seed in [1u64, 2, 7, 11, 42] {
                let mut rng = XorShift64::new(seed.wrapping_mul(100) + i as u64);
                let mut noise = || rng.laplace(1.0);
                check_lemmas(tree, records, 2.0, 1.0, 1.0, &mut noise);
                let mut zero = || 0.0;
                check_lemmas(tree, records, 2.0, 1.0, 1.0, &mut zero);
            }
        }
    }
    for seed in [3u64, 9, 21, 100, 256] {
        let tree = random_tree(seed, 5 + (seed % 4) as usize);
        assert!(tree.height() >= 5);
        let records = random_records(seed + 17, &tree);
        let mut rng = XorShift64::new(seed + 99);
        let mut noise = || rng.laplace(0.5);
        check_lemmas(&tree, &records, 1.5, 0.5, 2.0, &mut noise);
    }
}

#[test]
fn theorem_5_6_holds_for_every_draw() {
    let trees = [
        Tree::chain(8),
        Tree::complete_binary(5),
        Tree::skewed(6),
        random_tree(8, 6),
        random_tree(19, 7),
    ];
    // Extreme noise and Laplace noise. The guard, not the draw, enforces the margin.
    for tree in &trees {
        let cal = calibrate(tree.height() as u32, 1.0, 0.05);
        let leafs = leaves(tree);
        let mut heavy_records = Vec::new();
        // tau sits above E, so empty subtrees must be Light. The fat leaf's
        // ancestors have count above tau + E, so they must be Heavy.
        let tau = cal.e + 3.0;
        let copies = (tau + cal.e).ceil() as usize + 2;
        let fat = *leafs.last().unwrap();
        for _ in 0..copies {
            heavy_records.push(fat);
        }
        let counts = tree.subtree_counts(&heavy_records);
        let noises: Vec<Box<dyn FnMut() -> f64>> =
            vec![Box::new(|| 0.0), Box::new(|| 1.0e9), Box::new(|| -1.0e9)];
        for mut noise in noises {
            let cut = between_cut(tree, &heavy_records, tau, cal.lambda, cal.big_r, &mut noise);
            assert_eq!(cut.conflicts, 0);
            for u in 0..tree.len() {
                if counts[u] >= tau + cal.e {
                    assert_eq!(
                        cut.labels[u],
                        Label::Heavy,
                        "node {u} count {} >= tau+E",
                        counts[u]
                    );
                }
                if counts[u] <= tau - cal.e {
                    assert_eq!(
                        cut.labels[u],
                        Label::Light,
                        "node {u} count {} <= tau-E",
                        counts[u]
                    );
                }
            }
        }
        for seed in [1u64, 4, 8, 15, 16, 23, 42, 100, 2024, 9001] {
            let mut rng = XorShift64::new(seed);
            let mut noise = || rng.laplace(cal.lambda);
            let cut = between_cut(tree, &heavy_records, tau, cal.lambda, cal.big_r, &mut noise);
            assert_eq!(cut.conflicts, 0);
            for u in 0..tree.len() {
                if counts[u] + 1e-9 >= tau + cal.e {
                    assert_eq!(cut.labels[u], Label::Heavy);
                }
                if counts[u] <= tau - cal.e + 1e-9 && counts[u] + 1e-6 < tau + cal.e {
                    // The +1e-9 on the Light side only fires for counts that are
                    // not also above the heavy threshold. Exact integers here.
                    if counts[u] <= tau - cal.e {
                        assert_eq!(cut.labels[u], Label::Light);
                    }
                }
            }
        }
    }
}

#[test]
fn markdown_follows_public_child_order() {
    // Height 3 so DepO visits depth 1 before depth 2. Node 1's children are
    // stored as [3, 2], not index order. A Between there runs MarkDown, which
    // must label 3 before 2.
    let tree = Tree::from_children(vec![vec![1], vec![3, 2], vec![], vec![]]);
    assert_eq!(tree.children(1), &[3, 2]);
    assert_eq!(tree.height(), 3);
    let mut noise = || 0.0;
    let cut = between_cut_counts(&tree, &[0.0, 0.0, 0.0, 0.0], 0.0, 1.0, 1.0, &mut noise);
    assert_eq!(cut.queries[0].node, 1);
    assert_eq!(cut.queries[0].answer, Answer::Between);
    assert_eq!(&cut.first_label[..4], &[0, 1, 3, 2]);
    assert_eq!(cut.conflicts, 0);
}

#[test]
fn single_node_and_empty_database() {
    let tree = Tree::chain(1);
    let mut noise = || 0.0;
    let cut = between_cut(&tree, &[], 0.0, 1.0, 1.0, &mut noise);
    assert_eq!(cut.labels.len(), 1);
    assert_eq!(cut.conflicts, 0);
}
