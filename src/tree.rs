//! Hierarchical Leiden over the graph: root -> domains -> subdomains -> tables.
//!
//! Leiden (Traag et al. 2019, modularity objective) is implemented here rather than
//! pulled from a crate: it's ~200 lines, and owning it keeps runs reproducible from
//! `SEED` (no HashMap iteration order anywhere in the hot path).
//! The hierarchy is built the GraphRAG way: cluster the whole graph, then re-run
//! Leiden inside every community larger than `MAX_CLUSTER_SIZE`.

use crate::graph::{Graph, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// Leiden isn't deterministic. A fixed seed keeps the tree stable between runs.
pub const SEED: u64 = 42;
/// Communities with more tables than this get split again. Property nodes don't count:
/// most properties sit on a single table and would otherwise force needless splits.
pub const MAX_CLUSTER_SIZE: usize = 12;
/// Root is level 0; nothing is split below this level.
pub const MAX_DEPTH: usize = 4;
/// Modularity resolution. Higher -> more, smaller communities.
pub const RESOLUTION: f64 = 1.0;

pub type CommunityId = usize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub id: CommunityId,
    /// 0 = root.
    pub level: usize,
    pub parent: Option<CommunityId>,
    pub children: Vec<CommunityId>,
    /// Graph nodes placed directly in this community (leaf communities).
    pub members: Vec<NodeId>,
    /// Filled bottom-up by `summarize`.
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub root: CommunityId,
    pub communities: Vec<Community>,
}

impl Tree {
    fn add(&mut self, level: usize, parent: Option<CommunityId>) -> CommunityId {
        let id = self.communities.len();
        self.communities.push(Community {
            id,
            level,
            parent,
            children: vec![],
            members: vec![],
            summary: String::new(),
        });
        if let Some(p) = parent {
            self.communities[p].children.push(id);
        }
        id
    }

    /// Every graph node in the subtree under `c`.
    pub fn nodes_under(&self, c: CommunityId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut stack = vec![c];
        while let Some(c) = stack.pop() {
            let comm = &self.communities[c];
            out.extend(&comm.members);
            stack.extend(comm.children.iter().rev());
        }
        out
    }

    /// node -> community it sits in directly.
    pub fn placement(&self) -> HashMap<NodeId, CommunityId> {
        self.communities
            .iter()
            .flat_map(|c| c.members.iter().map(move |&m| (m, c.id)))
            .collect()
    }
}

pub fn cluster(graph: &Graph, seed: u64) -> Tree {
    let adj = merged_adjacency(graph);
    let is_table: Vec<bool> = graph
        .nodes
        .iter()
        .map(|n| n.table_name().is_some())
        .collect();
    let mut tree = Tree {
        root: 0,
        communities: vec![],
    };
    let root = tree.add(0, None);
    let mut rng = Rng(seed);
    split(
        &mut tree,
        &adj,
        &is_table,
        (0..graph.nodes.len()).collect(),
        root,
        &mut rng,
    );
    tree
}

/// Run Leiden on `nodes` and hang the resulting communities under `cid`,
/// recursing into the big ones.
fn split(
    tree: &mut Tree,
    adj: &[Vec<(usize, f64)>],
    is_table: &[bool],
    nodes: Vec<NodeId>,
    cid: CommunityId,
    rng: &mut Rng,
) {
    let level = tree.communities[cid].level;
    let tables = nodes.iter().filter(|&&n| is_table[n]).count();
    if tables <= MAX_CLUSTER_SIZE || level >= MAX_DEPTH {
        tree.communities[cid].members = nodes;
        return;
    }

    let labels = leiden(&WGraph::induced(adj, &nodes), RESOLUTION, rng);
    let mut groups: Vec<Vec<NodeId>> = vec![vec![]; labels.iter().max().map_or(0, |m| m + 1)];
    for (i, &l) in labels.iter().enumerate() {
        groups[l].push(nodes[i]);
    }
    // Nodes with no edges inside this subgraph come back as singletons; pool them
    // instead of making one community each.
    let (loose, mut groups): (Vec<_>, Vec<_>) = groups.into_iter().partition(|g| g.len() == 1);
    if !loose.is_empty() {
        groups.push(loose.into_iter().flatten().collect());
    }

    if groups.len() <= 1 {
        tree.communities[cid].members = nodes;
        return;
    }
    for group in groups {
        let child = tree.add(level + 1, Some(cid));
        split(tree, adj, is_table, group, child, rng);
    }
}

/// On a schema change: attach the new node to its best-fitting community
/// instead of reclustering, so the tree doesn't reshuffle under the model.
/// Best fit = the community holding the most edge weight to `node`.
pub fn attach(tree: &mut Tree, graph: &Graph, node: NodeId) {
    let placement = tree.placement();
    if placement.contains_key(&node) {
        return;
    }
    let mut score: BTreeMap<CommunityId, f64> = BTreeMap::new();
    for e in &graph.edges {
        let other = match (e.from == node, e.to == node) {
            (true, false) => e.to,
            (false, true) => e.from,
            _ => continue,
        };
        if let Some(&c) = placement.get(&other) {
            *score.entry(c).or_default() += e.weight;
        }
    }
    let target = score
        .into_iter()
        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        .map_or(tree.root, |(c, _)| c);
    tree.communities[target].members.push(node);
}

// ---------------------------------------------------------------------------
// Leiden

const EPS: f64 = 1e-12;
const MAX_ITERS: usize = 32;

/// Undirected weighted graph, no self loops. `k[v]` is v's degree including weight
/// folded inside it by aggregation, so modularity stays correct across levels.
#[derive(Debug, Clone)]
struct WGraph {
    adj: Vec<Vec<(usize, f64)>>,
    k: Vec<f64>,
}

impl WGraph {
    fn n(&self) -> usize {
        self.k.len()
    }

    fn two_m(&self) -> f64 {
        self.k.iter().sum()
    }

    /// Subgraph on `nodes` (indices into `adj`), renumbered 0..nodes.len().
    fn induced(adj: &[Vec<(usize, f64)>], nodes: &[NodeId]) -> WGraph {
        let local: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut g = WGraph {
            adj: vec![vec![]; nodes.len()],
            k: vec![0.0; nodes.len()],
        };
        for (i, &v) in nodes.iter().enumerate() {
            for &(u, w) in &adj[v] {
                if let Some(&j) = local.get(&u) {
                    g.adj[i].push((j, w));
                    g.k[i] += w;
                }
            }
        }
        g
    }
}

/// Parallel edges summed, self loops dropped, neighbours sorted.
fn merged_adjacency(graph: &Graph) -> Vec<Vec<(usize, f64)>> {
    let mut maps: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); graph.nodes.len()];
    for e in &graph.edges {
        if e.from != e.to && e.weight > 0.0 {
            *maps[e.from].entry(e.to).or_default() += e.weight;
            *maps[e.to].entry(e.from).or_default() += e.weight;
        }
    }
    maps.into_iter().map(|m| m.into_iter().collect()).collect()
}

/// Community label per node, numbered 0.. in order of first appearance.
fn leiden(g: &WGraph, gamma: f64, rng: &mut Rng) -> Vec<usize> {
    let n = g.n();
    if n == 0 {
        return vec![];
    }
    let mut cur = g.clone();
    let mut node_of: Vec<usize> = (0..n).collect(); // original node -> node of `cur`
    let mut part: Vec<usize> = (0..n).collect();

    for iter in 0..MAX_ITERS {
        let changed = move_nodes(&cur, &mut part, gamma, rng);
        if !changed && iter > 0 {
            break;
        }
        if renumber(&part).1 == cur.n() {
            break; // every node is its own community
        }
        let refined = refine(&cur, &part, gamma, rng);
        let (next, map) = aggregate(&cur, &refined);
        if next.n() == cur.n() {
            break; // refinement merged nothing; nothing left to aggregate
        }
        let mut next_part = vec![0; next.n()];
        for v in 0..cur.n() {
            next_part[map[v]] = part[v];
        }
        for o in node_of.iter_mut() {
            *o = map[*o];
        }
        cur = next;
        part = renumber(&next_part).0;
    }
    renumber(&(0..n).map(|o| part[node_of[o]]).collect::<Vec<_>>()).0
}

/// Fast local moving: visit nodes from a queue, move each to the neighbouring
/// community with the best modularity gain, re-queue neighbours of moved nodes.
fn move_nodes(g: &WGraph, part: &mut [usize], gamma: f64, rng: &mut Rng) -> bool {
    let n = g.n();
    let two_m = g.two_m();
    if two_m <= 0.0 {
        return false;
    }
    let mut tot = vec![0.0; n];
    let mut size = vec![0usize; n];
    for v in 0..n {
        tot[part[v]] += g.k[v];
        size[part[v]] += 1;
    }
    let mut empty: Vec<usize> = (0..n).filter(|&c| size[c] == 0).collect();

    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let mut queue: VecDeque<usize> = order.into();
    let mut queued = vec![true; n];
    let mut w_to = vec![0.0; n];
    let mut seen = vec![false; n];
    let mut touched = Vec::new();
    let mut changed = false;

    while let Some(v) = queue.pop_front() {
        queued[v] = false;
        let cur = part[v];
        let kv = g.k[v];
        for &(u, w) in &g.adj[v] {
            let c = part[u];
            if !seen[c] {
                seen[c] = true;
                touched.push(c);
            }
            w_to[c] += w;
        }

        tot[cur] -= kv;
        size[cur] -= 1;
        if size[cur] == 0 {
            empty.push(cur);
        }

        let gain = |c: usize| w_to[c] - gamma * kv * tot[c] / two_m;
        let mut best = cur;
        let mut best_gain = gain(cur);
        for &c in &touched {
            let gc = gain(c);
            if gc > best_gain + EPS {
                best = c;
                best_gain = gc;
            }
        }
        if best_gain < -EPS {
            best = *empty
                .last()
                .expect("v's removal leaves at least one community free");
        }
        if size[best] == 0 {
            let pos = empty
                .iter()
                .position(|&c| c == best)
                .expect("tracked as empty");
            empty.swap_remove(pos);
        }
        tot[best] += kv;
        size[best] += 1;
        part[v] = best;

        if best != cur {
            changed = true;
            for &(u, _) in &g.adj[v] {
                if part[u] != best && !queued[u] {
                    queued[u] = true;
                    queue.push_back(u);
                }
            }
        }
        for c in touched.drain(..) {
            w_to[c] = 0.0;
            seen[c] = false;
        }
    }
    changed
}

/// Refinement: inside each community, start from singletons and merge only
/// well-connected nodes into well-connected subsets. Guarantees every community
/// Leiden returns is internally connected (Louvain doesn't).
fn refine(g: &WGraph, part: &[usize], gamma: f64, rng: &mut Rng) -> Vec<usize> {
    let n = g.n();
    let two_m = g.two_m();
    let mut refined: Vec<usize> = (0..n).collect();
    let mut tot_r = g.k.clone();
    let mut size_r = vec![1usize; n];
    let mut tot_s = vec![0.0; n];
    for v in 0..n {
        tot_s[part[v]] += g.k[v];
    }
    // weight from v to the rest of its community
    let w_in: Vec<f64> = (0..n)
        .map(|v| {
            g.adj[v]
                .iter()
                .filter(|&&(u, _)| part[u] == part[v])
                .map(|&(_, w)| w)
                .sum()
        })
        .collect();
    // weight from refined subset c to the rest of its community
    let mut ext = w_in.clone();

    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let mut w_to = vec![0.0; n];
    let mut seen = vec![false; n];
    let mut touched = Vec::new();

    for v in order {
        let own = refined[v];
        if size_r[own] != 1 {
            continue; // already absorbed others; only singletons move
        }
        let s = part[v];
        let kv = g.k[v];
        if w_in[v] < gamma * kv * (tot_s[s] - kv) / two_m {
            continue; // v not well connected to its community
        }
        for &(u, w) in &g.adj[v] {
            if part[u] != s {
                continue;
            }
            let c = refined[u];
            if !seen[c] {
                seen[c] = true;
                touched.push(c);
            }
            w_to[c] += w;
        }
        let mut best = own;
        let mut best_gain = 0.0;
        for &c in &touched {
            if c == own {
                continue;
            }
            let well_connected = ext[c] >= gamma * tot_r[c] * (tot_s[s] - tot_r[c]) / two_m - EPS;
            let gain = w_to[c] - gamma * kv * tot_r[c] / two_m;
            if well_connected && gain > best_gain + EPS {
                best = c;
                best_gain = gain;
            }
        }
        if best != own {
            size_r[own] = 0;
            tot_r[own] = 0.0;
            refined[v] = best;
            size_r[best] += 1;
            tot_r[best] += kv;
            ext[best] += w_in[v] - 2.0 * w_to[best];
        }
        for c in touched.drain(..) {
            w_to[c] = 0.0;
            seen[c] = false;
        }
    }
    refined
}

/// Collapse each refined community into one node. Returns the new graph and
/// old node -> new node.
fn aggregate(g: &WGraph, refined: &[usize]) -> (WGraph, Vec<usize>) {
    let (map, r) = renumber(refined);
    let mut k = vec![0.0; r];
    let mut maps: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); r];
    for v in 0..g.n() {
        k[map[v]] += g.k[v];
        for &(u, w) in &g.adj[v] {
            if map[u] != map[v] {
                *maps[map[v]].entry(map[u]).or_default() += w;
            }
        }
    }
    let adj = maps.into_iter().map(|m| m.into_iter().collect()).collect();
    (WGraph { adj, k }, map)
}

/// Relabel to 0.. in order of first appearance; returns (labels, count).
fn renumber(labels: &[usize]) -> (Vec<usize>, usize) {
    let mut ids: HashMap<usize, usize> = HashMap::new();
    let out = labels
        .iter()
        .map(|&l| {
            let next = ids.len();
            *ids.entry(l).or_insert(next)
        })
        .collect();
    (out, ids.len())
}

/// splitmix64: tiny, seedable, good enough for shuffling.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, EdgeKind, Node};

    /// `cliques` fully connected groups of `size`, chained by single weak bridges.
    fn cliques(count: usize, size: usize) -> Graph {
        let mut g = Graph::default();
        for i in 0..count * size {
            g.nodes.push(Node::Table(format!("t{i}")));
        }
        let mut edge = |from, to, weight| {
            g.edges.push(Edge {
                from,
                to,
                kind: EdgeKind::Join,
                weight,
            })
        };
        for c in 0..count {
            for a in 0..size {
                for b in a + 1..size {
                    edge(c * size + a, c * size + b, 1.0);
                }
            }
            if c > 0 {
                edge((c - 1) * size, c * size, 0.1);
            }
        }
        g
    }

    #[test]
    fn leiden_finds_cliques() {
        let g = cliques(4, 6);
        let w = WGraph::induced(&merged_adjacency(&g), &(0..24).collect::<Vec<_>>());
        let labels = leiden(&w, 1.0, &mut Rng(SEED));
        for c in 0..4 {
            let block = &labels[c * 6..(c + 1) * 6];
            assert!(block.iter().all(|&l| l == block[0]), "{labels:?}");
        }
        assert_eq!(labels.iter().max(), Some(&3));
    }

    #[test]
    fn hierarchy_splits_and_is_deterministic() {
        let g = cliques(4, 6); // 24 nodes > MAX_CLUSTER_SIZE
        let a = cluster(&g, SEED);
        let b = cluster(&g, SEED);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
        assert_eq!(a.communities[a.root].children.len(), 4);
        let mut all = a.nodes_under(a.root);
        all.sort();
        assert_eq!(all, (0..24).collect::<Vec<_>>());
    }

    #[test]
    fn attach_goes_to_best_neighbour_community() {
        let mut g = cliques(4, 6);
        let mut tree = cluster(&g, SEED);
        let new = g.nodes.len();
        g.nodes.push(Node::Table("new".into()));
        g.edges.push(Edge {
            from: new,
            to: 13,
            kind: EdgeKind::Join,
            weight: 1.0,
        });
        g.edges.push(Edge {
            from: new,
            to: 14,
            kind: EdgeKind::Join,
            weight: 1.0,
        });
        g.edges.push(Edge {
            from: new,
            to: 0,
            kind: EdgeKind::Join,
            weight: 1.0,
        });
        attach(&mut tree, &g, new);
        let placement = tree.placement();
        assert_eq!(placement[&new], placement[&13]);
    }
}

#[cfg(test)]
mod adventureworks {
    use super::*;
    use crate::{graph, schema};

    #[test]
    fn every_table_lands_in_exactly_one_community() {
        let db = concat!(env!("CARGO_MANIFEST_DIR"), "/data/adventureworks.db");
        let schema = schema::from_sqlite(db).expect("run scripts/fetch-adventureworks.sh");
        let g = graph::build(&schema, &[]); // join edges only
        let tree = cluster(&g, SEED);
        let mut placed = tree.nodes_under(tree.root);
        placed.sort();
        assert_eq!(placed, (0..g.nodes.len()).collect::<Vec<_>>());
        assert!(tree.communities[tree.root].children.len() > 1);
        for c in &tree.communities {
            if c.children.is_empty() {
                eprintln!(
                    "L{} {:?}",
                    c.level,
                    c.members
                        .iter()
                        .map(|&m| g.nodes[m].label())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}
