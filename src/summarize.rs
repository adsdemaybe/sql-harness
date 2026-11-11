//! Bottom-up summaries: leaves first, then each parent from its children's summaries,
//! so every level of the tree has a description the model can read.

use crate::ai::Ai;
use crate::graph::{Graph, Node};
use crate::schema::{Schema, Table};
use crate::tree::Tree;
use anyhow::{Context, Result};
use std::cmp::Reverse;
use std::collections::HashMap;

const SYSTEM: &str = "You summarize part of a database schema for a model that navigates \
the schema top-down to find tables for a question. Reply with a short title on the first \
line, then 1-3 sentences on what this group covers and which questions it can answer. \
Name the key tables. No preamble, no markdown.";

/// Columns listed per table in a leaf prompt.
const MAX_COLUMNS: usize = 20;

pub fn run(tree: &mut Tree, graph: &Graph, schema: &Schema, ai: &dyn Ai) -> Result<()> {
    let tables: HashMap<&str, &Table> =
        schema.tables.iter().map(|t| (t.name.as_str(), t)).collect();

    // Deepest first, so children are always summarized before their parent.
    let mut order: Vec<usize> = (0..tree.communities.len()).collect();
    order.sort_by_key(|&c| Reverse(tree.communities[c].level));

    for (i, c) in order.iter().copied().enumerate() {
        let comm = &tree.communities[c];
        if comm.members.is_empty() && comm.children.is_empty() {
            continue;
        }
        let prompt = prompt(tree, c, graph, &tables);
        eprintln!(
            "[{}/{}] summarizing community {c} (level {})",
            i + 1,
            order.len(),
            comm.level
        );
        let summary = ai
            .complete(SYSTEM, &prompt)
            .with_context(|| format!("summarizing community {c}"))?;
        tree.communities[c].summary = summary.trim().to_string();
    }
    Ok(())
}

fn prompt(tree: &Tree, c: usize, graph: &Graph, tables: &HashMap<&str, &Table>) -> String {
    let comm = &tree.communities[c];
    let mut table_lines = Vec::new();
    let mut themes = Vec::new();
    for &m in &comm.members {
        match &graph.nodes[m] {
            Node::Table(name) => table_lines.push(match tables.get(name.as_str()) {
                Some(t) => describe(t),
                None => format!("- {name}"),
            }),
            prop @ Node::Property(_) => themes.push(prop.label()),
        }
    }

    let mut p = String::new();
    if !comm.children.is_empty() {
        // Without this the model tends to summarize just one subgroup.
        p += &format!(
            "This group is made of the {} subgroups below. Summarize the group as a whole, \
             covering every subgroup, not just one.\nSubgroups:\n",
            comm.children.len()
        );
        for &ch in &comm.children {
            let s = &tree.communities[ch].summary;
            if !s.is_empty() {
                p += &format!("- {}\n", s.replace('\n', " / "));
            }
        }
    }
    if !table_lines.is_empty() {
        p += "Tables:\n";
        p += &table_lines.join("\n");
        p += "\n";
    }
    if !themes.is_empty() {
        p += &format!("Shared properties: {}\n", themes.join("; "));
    }
    p
}

/// One line per table: name, columns (PKs marked), description, join targets.
pub fn describe(t: &Table) -> String {
    let mut cols: Vec<String> = t
        .columns
        .iter()
        .take(MAX_COLUMNS)
        .map(|c| {
            if c.primary_key {
                format!("{} (PK)", c.name)
            } else {
                c.name.clone()
            }
        })
        .collect();
    if t.columns.len() > MAX_COLUMNS {
        cols.push(format!("... {} more", t.columns.len() - MAX_COLUMNS));
    }
    let mut line = format!("- {}: {}", t.name, cols.join(", "));
    if let Some(d) = &t.description {
        line += &format!(" | {d}");
    }
    if !t.foreign_keys.is_empty() {
        let refs: Vec<&str> = t
            .foreign_keys
            .iter()
            .map(|fk| fk.ref_table.as_str())
            .collect();
        line += &format!(" | joins to {}", refs.join(", "));
    }
    line
}
