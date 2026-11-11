//! The graph Leiden clusters. Leiden only sees edges and weights, so properties are
//! their own nodes: two tables sharing "audit" are two hops apart even with no join.

use crate::ai::Ai;
use crate::ingest::{Property, PropertyKind, TableProperties};
use crate::schema::Schema;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub type NodeId = usize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    Table(String),
    Property(Property),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Table to table, from a foreign key. Strongest.
    Join,
    /// Table to property node, weighted by the property's rarity.
    Property,
    /// Table to table, by description similarity. Weakest, optional.
    Similarity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
    pub weight: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// One vector per node (same index), filled by `embed`. Used to land questions.
    #[serde(default)]
    pub embeddings: Vec<Vec<f32>>,
}

pub const JOIN_WEIGHT: f64 = 1.0;
/// Weight of the rarest property (on one table); commoner ones scale down from here.
pub const PROPERTY_WEIGHT: f64 = 0.5;
pub const SIMILARITY_WEIGHT: f64 = 0.1;
/// Properties linked to more than this share of tables are dropped, or Leiden returns one blob.
pub const MAX_PROPERTY_SHARE: f64 = 0.5;
/// Similarity edges: each table links to its `SIMILARITY_K` nearest tables (mutual only).
pub const SIMILARITY_K: usize = 3;

/// Table nodes + property nodes, then join and property edges.
/// Similarity edges need embeddings, so they're added later by `add_similarity_edges`.
pub fn build(schema: &Schema, props: &[TableProperties]) -> Graph {
    let mut g = Graph::default();
    let mut table_id: HashMap<&str, NodeId> = HashMap::new();
    for t in &schema.tables {
        table_id.insert(&t.name, g.nodes.len());
        g.nodes.push(Node::Table(t.name.clone()));
    }

    // Property -> the tables carrying it (BTree keeps node ids stable between runs).
    let mut carriers: BTreeMap<&Property, BTreeSet<NodeId>> = BTreeMap::new();
    for tp in props {
        if let Some(&tid) = table_id.get(tp.table.as_str()) {
            for p in &tp.properties {
                carriers.entry(p).or_default().insert(tid);
            }
        }
    }
    let total = schema.tables.len();
    for (prop, tables) in carriers {
        let Some(weight) = rarity_weight(tables.len(), total) else {
            continue;
        };
        let pid = g.nodes.len();
        g.nodes.push(Node::Property(prop.clone()));
        for t in tables {
            g.edges.push(Edge {
                from: t,
                to: pid,
                kind: EdgeKind::Property,
                weight,
            });
        }
    }

    for t in &schema.tables {
        let from = table_id[t.name.as_str()];
        for fk in &t.foreign_keys {
            match table_id.get(fk.ref_table.as_str()) {
                Some(&to) if to != from => g.edges.push(Edge {
                    from,
                    to,
                    kind: EdgeKind::Join,
                    weight: JOIN_WEIGHT,
                }),
                _ => {} // self-reference or dangling FK
            }
        }
    }
    g
}

/// Weight for a property found on `tables_with` of `total` tables; None if near-universal.
/// Normalized IDF: a property on one table gets PROPERTY_WEIGHT, one on half of them ~0.
fn rarity_weight(tables_with: usize, total: usize) -> Option<f64> {
    if tables_with == 0 || total == 0 {
        return None;
    }
    let share = tables_with as f64 / total as f64;
    if share > MAX_PROPERTY_SHARE {
        return None;
    }
    if total == 1 {
        return Some(PROPERTY_WEIGHT);
    }
    let idf = (total as f64 / tables_with as f64).ln() / (total as f64).ln();
    Some(PROPERTY_WEIGHT * idf)
}

/// Embed every node (tables as name + columns, properties as "kind: value").
pub fn embed(graph: &mut Graph, schema: &Schema, ai: &dyn Ai) -> Result<()> {
    let columns: HashMap<&str, Vec<&str>> = schema
        .tables
        .iter()
        .map(|t| {
            (
                t.name.as_str(),
                t.columns.iter().map(|c| c.name.as_str()).collect(),
            )
        })
        .collect();
    let texts: Vec<String> = graph
        .nodes
        .iter()
        .map(|n| match n {
            Node::Table(name) => match columns.get(name.as_str()) {
                Some(cols) => format!("table {name}: {}", cols.join(", ")),
                None => format!("table {name}"),
            },
            Node::Property(_) => n.label(),
        })
        .collect();
    eprintln!("embedding {} nodes", texts.len());
    let vecs = ai.embed_many(&texts)?;
    ensure!(
        vecs.len() == texts.len(),
        "got {} embeddings for {} nodes",
        vecs.len(),
        texts.len()
    );
    graph.embeddings = vecs;
    Ok(())
}

/// Table-to-table edges between mutual nearest neighbours by embedding.
pub fn add_similarity_edges(graph: &mut Graph) {
    let tables: Vec<NodeId> = graph.table_ids().collect();
    if graph.embeddings.len() != graph.nodes.len() {
        return;
    }
    let nearest: HashMap<NodeId, Vec<(NodeId, f32)>> = tables
        .iter()
        .map(|&a| {
            let mut sims: Vec<(NodeId, f32)> = tables
                .iter()
                .filter(|&&b| b != a)
                .map(|&b| (b, cosine(&graph.embeddings[a], &graph.embeddings[b])))
                .collect();
            sims.sort_by(|x, y| y.1.total_cmp(&x.1));
            sims.truncate(SIMILARITY_K);
            (a, sims)
        })
        .collect();
    for &a in &tables {
        for &(b, sim) in &nearest[&a] {
            let mutual = nearest[&b].iter().any(|&(x, _)| x == a);
            if a < b && mutual && sim > 0.0 {
                let weight = SIMILARITY_WEIGHT * sim as f64;
                graph.edges.push(Edge {
                    from: a,
                    to: b,
                    kind: EdgeKind::Similarity,
                    weight,
                });
            }
        }
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

impl Node {
    pub fn label(&self) -> String {
        match self {
            Node::Table(name) => name.clone(),
            Node::Property(p) => {
                let kind = match p.kind {
                    PropertyKind::Entity => "entity",
                    PropertyKind::Role => "role",
                    PropertyKind::ColumnConcept => "column concept",
                    PropertyKind::Stage => "process stage",
                };
                format!("{kind}: {}", p.value)
            }
        }
    }

    pub fn table_name(&self) -> Option<&str> {
        match self {
            Node::Table(name) => Some(name),
            Node::Property(_) => None,
        }
    }
}

impl Graph {
    pub fn table_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        (0..self.nodes.len()).filter(|&i| matches!(self.nodes[i], Node::Table(_)))
    }

    /// Undirected adjacency: node -> (neighbour, kind, weight), one entry per edge.
    pub fn adjacency(&self) -> Vec<Vec<(NodeId, EdgeKind, f64)>> {
        let mut adj = vec![Vec::new(); self.nodes.len()];
        for e in &self.edges {
            adj[e.from].push((e.to, e.kind, e.weight));
            adj[e.to].push((e.from, e.kind, e.weight));
        }
        adj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rarity() {
        assert_eq!(rarity_weight(1, 100), Some(PROPERTY_WEIGHT));
        assert!(rarity_weight(80, 100).is_none());
        let common = rarity_weight(40, 100).unwrap();
        let rare = rarity_weight(3, 100).unwrap();
        assert!(rare > common && common > 0.0);
    }
}
