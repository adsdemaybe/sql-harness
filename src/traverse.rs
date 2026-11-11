//! Retrieval over the tree and the graph, then (optionally) a natural-language answer.
//!
//! Pipeline per question:
//!   1. candidates: `descend` (vague) or `land_and_expand` (specific), ~CANDIDATES tables
//!   2. select:     the AI reads the candidates' columns and keeps the ones needed
//!   3. connect:    join paths between the kept tables (bridge tables added)
//! `ask --nl` continues in answer.rs: write SQL, run it, answer from the rows.

use crate::ai::Ai;
use crate::graph::{EdgeKind, Graph, NodeId, cosine};
use crate::schema::{Schema, Table};
use crate::summarize::describe;
use crate::tree::{CommunityId, Tree};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};

/// Tables shown to the AI in the select step.
pub const CANDIDATES: usize = 25;
/// Most tables kept after selection (before bridge tables are added).
pub const MAX_TABLES: usize = 12;
/// Kept when the select step returns nothing usable.
pub const FALLBACK_TABLES: usize = 6;
/// Nodes the question lands on before expanding.
pub const LAND_K: usize = 8;
/// Expansion depth. 2 = table -> property -> table sharing it.
pub const HOPS: usize = 2;
/// Longest join path (in joins) worth reporting.
pub const MAX_JOIN_HOPS: usize = 4;

#[derive(Debug, Serialize)]
pub struct Retrieved {
    pub mode: &'static str,
    pub tables: Vec<String>,
    /// Each path is a list of table names connected by join edges.
    pub joins: Vec<Vec<String>>,
    /// SQL join conditions for every hop on those paths.
    pub join_on: Vec<String>,
}

/// Entry point: vague questions go to `descend`, specific ones to `land_and_expand`.
pub fn retrieve(
    question: &str,
    graph: &Graph,
    tree: &Tree,
    schema: &Schema,
    ai: &dyn Ai,
) -> Result<Retrieved> {
    ensure!(
        graph.embeddings.len() == graph.nodes.len(),
        "graph.json has no node embeddings; rerun `build`"
    );
    let tables = by_name(schema);
    let query = ai.embed(question)?;
    let sims: Vec<f64> = graph
        .embeddings
        .iter()
        .map(|e| cosine(&query, e) as f64)
        .collect();

    let vague = is_vague(question, ai)?;
    let mut ranked: Vec<NodeId> = if vague {
        let path = descend(question, tree, ai)?;
        let here = *path.last().unwrap_or(&tree.root);
        let mut nodes = tree.nodes_under(here);
        // Hub tables (people, products) may sit in a neighbouring community:
        // cross the boundary along join edges.
        let inside: HashSet<NodeId> = nodes.iter().copied().collect();
        let adj = graph.adjacency();
        let mut extra: Vec<NodeId> = nodes
            .iter()
            .flat_map(|&n| adj[n].iter())
            .filter(|&&(u, kind, _)| kind == EdgeKind::Join && !inside.contains(&u))
            .map(|&(u, ..)| u)
            .collect();
        extra.sort();
        extra.dedup();
        nodes.extend(extra);
        nodes.sort_by(|&a, &b| sims[b].total_cmp(&sims[a]));
        nodes
    } else {
        land_and_expand(&sims, graph, HOPS)
    };
    ranked.retain(|&n| graph.nodes[n].table_name().is_some());
    ranked.truncate(CANDIDATES);

    let selected = select(question, &ranked, graph, &tables, ai)?;
    let (mut picked, paths) = connect(graph, &selected);
    picked.dedup();

    let name = |n: NodeId| graph.nodes[n].label();
    let mut join_on = Vec::new();
    for path in &paths {
        for pair in path.windows(2) {
            for cond in join_conditions(&tables, &name(pair[0]), &name(pair[1])) {
                if !join_on.contains(&cond) {
                    join_on.push(cond);
                }
            }
        }
    }
    Ok(Retrieved {
        mode: if vague { "descend" } else { "land_and_expand" },
        tables: picked.into_iter().map(name).collect(),
        joins: paths
            .into_iter()
            .map(|p| p.into_iter().map(name).collect())
            .collect(),
        join_on,
    })
}

pub fn by_name(schema: &Schema) -> HashMap<&str, &Table> {
    schema.tables.iter().map(|t| (t.name.as_str(), t)).collect()
}

/// Graph search for the agent's `search_schema` tool: embed `text`, land and expand,
/// return the best `k` tables.
pub fn search(text: &str, graph: &Graph, ai: &dyn Ai, k: usize) -> Result<Vec<String>> {
    ensure!(
        graph.embeddings.len() == graph.nodes.len(),
        "graph.json has no node embeddings; rerun `build`"
    );
    let query = ai.embed(text)?;
    let sims: Vec<f64> = graph
        .embeddings
        .iter()
        .map(|e| cosine(&query, e) as f64)
        .collect();
    Ok(land_and_expand(&sims, graph, HOPS)
        .into_iter()
        .filter_map(|n| graph.nodes[n].table_name().map(String::from))
        .take(k)
        .collect())
}

fn is_vague(question: &str, ai: &dyn Ai) -> Result<bool> {
    let reply = ai.complete(
        "Classify a question about a database. Reply with one word: \"specific\" if it names \
         concrete entities, columns or values (e.g. \"order totals per territory in 2014\"), \
         \"vague\" if it is broad or exploratory (e.g. \"what do we know about our staff?\").",
        question,
    )?;
    Ok(reply.to_lowercase().contains("vague"))
}

/// Vague question: start at the root, read child summaries, descend into the best match.
/// Returns the path from the root; stops where the question spans several children.
fn descend(question: &str, tree: &Tree, ai: &dyn Ai) -> Result<Vec<CommunityId>> {
    let mut path = vec![tree.root];
    let mut cur = tree.root;
    loop {
        let children = &tree.communities[cur].children;
        if children.is_empty() {
            break;
        }
        let mut prompt = format!("Question: {question}\n\nGroups of tables:\n");
        for (i, &c) in children.iter().enumerate() {
            let summary = tree.communities[c].summary.replace('\n', " / ");
            prompt += &format!("{}. {summary}\n", i + 1);
        }
        prompt += "\nReply with only the number of the group most likely to hold the tables \
                   needed, or 0 if the question needs several of these groups.";
        let reply = ai.complete("You navigate a database schema top-down.", &prompt)?;
        let pick: usize = reply
            .split(|c: char| !c.is_ascii_digit())
            .find(|s| !s.is_empty())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if pick == 0 || pick > children.len() {
            break;
        }
        cur = children[pick - 1];
        path.push(cur);
    }
    Ok(path)
}

/// Specific question: land on the nodes nearest the question embedding, expand along
/// edges. Expansion crosses community boundaries, so hub tables like `users` don't get
/// stranded. Score = half direct similarity, half similarity propagated through edge
/// weights from the landing nodes. Returns nodes best first.
fn land_and_expand(sims: &[f64], graph: &Graph, hops: usize) -> Vec<NodeId> {
    let n = graph.nodes.len();
    let adj = graph.adjacency();
    let mut by_sim: Vec<NodeId> = (0..n).collect();
    by_sim.sort_by(|&a, &b| sims[b].total_cmp(&sims[a]));

    let mut spread = vec![f64::NEG_INFINITY; n];
    let mut frontier: Vec<NodeId> = by_sim.into_iter().take(LAND_K).collect();
    for &l in &frontier {
        spread[l] = sims[l];
    }
    for _ in 0..hops {
        let mut next = Vec::new();
        for &v in &frontier {
            for &(u, _, w) in &adj[v] {
                let s = spread[v] * w;
                if s > spread[u] {
                    if spread[u] == f64::NEG_INFINITY {
                        next.push(u);
                    }
                    spread[u] = s;
                }
            }
        }
        frontier = next;
    }

    let mut reached: Vec<(NodeId, f64)> = (0..n)
        .filter(|&v| spread[v] > f64::NEG_INFINITY)
        .map(|v| (v, 0.5 * spread[v] + 0.5 * sims[v]))
        .collect();
    reached.sort_by(|a, b| b.1.total_cmp(&a.1));
    reached.into_iter().map(|(v, _)| v).collect()
}

/// The AI reads the candidates' columns and keeps the tables the question needs.
/// Returned in candidate rank order; falls back to the top few if the reply is unusable.
fn select(
    question: &str,
    candidates: &[NodeId],
    graph: &Graph,
    tables: &HashMap<&str, &Table>,
    ai: &dyn Ai,
) -> Result<Vec<NodeId>> {
    let mut prompt = format!("Question: {question}\n\nCandidate tables:\n");
    for &c in candidates {
        if let Some(t) = graph.nodes[c].table_name().and_then(|n| tables.get(n)) {
            prompt += &describe(t);
            prompt += "\n";
        }
    }
    prompt += "\nWhich of these tables are needed to write SQL that answers the question? \
               Pick the smallest set that covers every entity, filter and measure in the \
               question. Tables needed only to join others are added automatically, so you \
               can leave them out. Reply with a JSON array of exact table names.";
    let reply = ai.complete(
        "You pick database tables for a question. Reply with a JSON array only.",
        &prompt,
    )?;

    let names: Vec<String> = match (reply.find('['), reply.rfind(']')) {
        (Some(s), Some(e)) if e > s => serde_json::from_str(&reply[s..=e]).unwrap_or_default(),
        _ => vec![],
    };
    let wanted: HashSet<&str> = names.iter().map(String::as_str).collect();
    let mut picked: Vec<NodeId> = candidates
        .iter()
        .copied()
        .filter(|&c| {
            graph.nodes[c]
                .table_name()
                .is_some_and(|n| wanted.contains(n))
        })
        .take(MAX_TABLES)
        .collect();
    if picked.is_empty() {
        eprintln!("select step returned no usable tables; using the top {FALLBACK_TABLES}");
        picked = candidates.iter().copied().take(FALLBACK_TABLES).collect();
    }
    Ok(picked)
}

/// Link the retrieved tables with join paths: each table connects by the shortest path
/// to any table already placed. Bridge tables on those paths are added to the result.
fn connect(graph: &Graph, ranked: &[NodeId]) -> (Vec<NodeId>, Vec<Vec<NodeId>>) {
    let mut tables: Vec<NodeId> = Vec::new();
    let mut placed: HashSet<NodeId> = HashSet::new();
    let mut paths = Vec::new();
    for &t in ranked {
        if placed.contains(&t) {
            continue;
        }
        let best = tables
            .iter()
            .filter_map(|&p| join_path(graph, p, t))
            .filter(|p| p.len() <= MAX_JOIN_HOPS + 1)
            .min_by_key(Vec::len);
        tables.push(t);
        placed.insert(t);
        if let Some(path) = best {
            for &b in &path {
                if placed.insert(b) {
                    tables.push(b);
                }
            }
            paths.push(path);
        }
    }
    (tables, paths)
}

/// `"A"."x" = "B"."y"` for every foreign key between the two tables, either direction.
pub fn join_conditions(tables: &HashMap<&str, &Table>, a: &str, b: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (from, to) in [(a, b), (b, a)] {
        let Some(t) = tables.get(from) else { continue };
        for fk in t.foreign_keys.iter().filter(|fk| fk.ref_table == to) {
            let cond = fk
                .columns
                .iter()
                .zip(&fk.ref_columns)
                .map(|(c, r)| format!("\"{from}\".{c} = \"{to}\".{r}"))
                .collect::<Vec<_>>()
                .join(" AND ");
            out.push(cond);
        }
    }
    out
}

/// Join path between two tables, following Join edges only.
/// Same community means "related", not "joinable".
fn join_path(graph: &Graph, from: NodeId, to: NodeId) -> Option<Vec<NodeId>> {
    if from == to {
        return Some(vec![from]);
    }
    let adj = graph.adjacency();
    let mut prev: Vec<Option<NodeId>> = vec![None; graph.nodes.len()];
    let mut seen = vec![false; graph.nodes.len()];
    let mut queue = VecDeque::from([from]);
    seen[from] = true;
    while let Some(v) = queue.pop_front() {
        for &(u, kind, _) in &adj[v] {
            if kind != EdgeKind::Join || seen[u] {
                continue;
            }
            seen[u] = true;
            prev[u] = Some(v);
            if u == to {
                let mut path = vec![to];
                let mut cur = to;
                while let Some(p) = prev[cur] {
                    path.push(p);
                    cur = p;
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(u);
        }
    }
    None
}
