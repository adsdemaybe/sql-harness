//! Ingestion: the AI reads each table's description and column names and extracts
//! properties; then properties are canonicalized so variants collapse into one node.

use crate::ai::Ai;
use crate::schema::{Schema, Table};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyKind {
    /// What the table is about: audit, customer, invoice.
    Entity,
    /// fact, lookup, junction, log, history.
    Role,
    /// Normalized column meaning: "audit identifier", "filing date" (not raw names).
    ColumnConcept,
    /// Where it sits in a workflow: filed, reviewed, approved.
    Stage,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Property {
    pub kind: PropertyKind,
    /// Canonical value after `canonicalize`.
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableProperties {
    pub table: String,
    pub properties: Vec<Property>,
}

const ROLES: [&str; 5] = ["fact", "lookup", "junction", "log", "history"];

const SYSTEM: &str = "You extract properties from one database table's schema for a knowledge \
graph of the database. Reply with a single JSON object and nothing else.";

pub fn run(schema: &Schema, ai: &dyn Ai) -> Result<Vec<TableProperties>> {
    let total = schema.tables.len();
    let mut out = schema
        .tables
        .iter()
        .enumerate()
        .map(|(i, t)| {
            eprintln!("[{}/{total}] extracting {}", i + 1, t.name);
            Ok(TableProperties {
                table: t.name.clone(),
                properties: extract(t, ai).with_context(|| format!("extracting {}", t.name))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    canonicalize(&mut out);
    Ok(out)
}

#[derive(Debug, Default, Deserialize)]
struct Extracted {
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    column_concepts: Vec<String>,
    #[serde(default)]
    stages: Vec<String>,
}

/// Ask the AI for this table's properties (prompt asks for JSON, parse into `Property`).
/// Input is the table name, columns (type, PK) and foreign keys, plus the description
/// when the source has one (SQLite doesn't).
fn extract(table: &Table, ai: &dyn Ai) -> Result<Vec<Property>> {
    let reply = ai.complete(SYSTEM, &prompt(table))?;
    let parsed: Extracted = serde_json::from_str(json_object(&reply))
        .with_context(|| format!("AI reply was not the expected JSON:\n{reply}"))?;

    let mut props = Vec::new();
    let mut push = |kind: PropertyKind, value: &str| {
        let value = normalize(value);
        if !value.is_empty() {
            props.push(Property { kind, value });
        }
    };
    parsed
        .entities
        .iter()
        .for_each(|v| push(PropertyKind::Entity, v));
    parsed
        .column_concepts
        .iter()
        .for_each(|v| push(PropertyKind::ColumnConcept, v));
    parsed
        .stages
        .iter()
        .for_each(|v| push(PropertyKind::Stage, v));
    if let Some(role) = parsed.role.as_deref().map(normalize) {
        if ROLES.contains(&role.as_str()) {
            push(PropertyKind::Role, &role);
        }
    }
    Ok(props)
}

fn prompt(table: &Table) -> String {
    let mut p = format!("Table: {}\n", table.name);
    if let Some(desc) = &table.description {
        p += &format!("Description: {desc}\n");
    }
    p += "Columns:\n";
    for c in &table.columns {
        let pk = if c.primary_key { " PRIMARY KEY" } else { "" };
        p += &format!("- {} {}{pk}\n", c.name, c.ty);
    }
    if !table.foreign_keys.is_empty() {
        p += "Foreign keys:\n";
        for fk in &table.foreign_keys {
            p += &format!(
                "- ({}) -> {}({})\n",
                fk.columns.join(", "),
                fk.ref_table,
                fk.ref_columns.join(", ")
            );
        }
    }
    p += r#"
Return exactly this JSON shape:
{"entities": [...], "role": "...", "column_concepts": [...], "stages": [...]}

- entities: 1-3 business entities the table is about, as singular lowercase nouns
  ("sales order", "customer", "product").
- role: exactly one of "fact", "lookup", "junction", "log", "history".
- column_concepts: the normalized meaning of each meaningful column, lowercase, 1-3 words,
  not the raw column name ("order date", "product identifier", "unit price").
- stages: workflow stages this table records, if any ("ordered", "shipped", "approved").
  Usually empty."#;
    p
}

/// The model may wrap JSON in prose or ``` fences; take the outermost {...}.
fn json_object(reply: &str) -> &str {
    match (reply.find('{'), reply.rfind('}')) {
        (Some(start), Some(end)) if end > start => &reply[start..=end],
        _ => reply,
    }
}

/// Lowercase, punctuation and underscores to spaces, collapse whitespace.
fn normalize(s: &str) -> String {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Crude stemmer used only as a grouping key: "audits", "auditing" -> "audit".
fn stem(word: &str) -> String {
    let mut w = word.to_string();
    let n = w.len();
    if n > 5 && w.ends_with("ing") {
        w.truncate(n - 3);
    } else if n > 4 && w.ends_with("ies") {
        w.truncate(n - 3);
        w.push('y');
    } else if w.ends_with("sses") {
        w.truncate(n - 2);
    } else if n > 4 && w.ends_with("ed") {
        w.truncate(n - 2);
    } else if n > 3
        && w.ends_with('s')
        && !w.ends_with("ss")
        && !w.ends_with("us")
        && !w.ends_with("is")
    {
        w.truncate(n - 1);
    }
    // approve/approved, file/filing, ship/shipping, bill/billing
    if w.len() > 3 && w.ends_with('e') {
        w.pop();
    }
    let b = w.as_bytes();
    if b.len() > 3 && b[b.len() - 1] == b[b.len() - 2] && !b"aeiou".contains(&b[b.len() - 1]) {
        w.pop();
    }
    w
}

fn stem_key(value: &str) -> String {
    value.split(' ').map(stem).collect::<Vec<_>>().join(" ")
}

/// "audit", "audits", "auditing" must merge into one value, or the links never form.
/// Values of the same kind with the same stem key collapse to the group's most common
/// spelling (ties: shortest, then alphabetical).
fn canonicalize(tables: &mut [TableProperties]) {
    // (kind, stem key) -> spelling -> count
    let mut groups: BTreeMap<(PropertyKind, String), BTreeMap<String, usize>> = BTreeMap::new();
    for p in tables.iter_mut().flat_map(|t| t.properties.iter_mut()) {
        p.value = normalize(&p.value);
        let key = (p.kind.clone(), stem_key(&p.value));
        *groups
            .entry(key)
            .or_default()
            .entry(p.value.clone())
            .or_default() += 1;
    }

    let canonical: BTreeMap<(PropertyKind, String), String> = groups
        .into_iter()
        .map(|(key, spellings)| {
            let best = spellings
                .into_iter()
                .min_by(|(a, ca), (b, cb)| cb.cmp(ca).then(a.len().cmp(&b.len())).then(a.cmp(b)))
                .map(|(s, _)| s)
                .expect("group has at least one spelling");
            (key, best)
        })
        .collect();

    for t in tables.iter_mut() {
        for p in &mut t.properties {
            p.value = canonical[&(p.kind.clone(), stem_key(&p.value))].clone();
        }
        t.properties.sort();
        t.properties.dedup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tp(table: &str, values: &[&str]) -> TableProperties {
        TableProperties {
            table: table.into(),
            properties: values
                .iter()
                .map(|v| Property {
                    kind: PropertyKind::Entity,
                    value: v.to_string(),
                })
                .collect(),
        }
    }

    #[test]
    fn variants_merge() {
        let mut t = vec![
            tp("a", &["audit", "Shipping"]),
            tp("b", &["audits", "ship"]),
            tp("c", &["auditing", "sales_orders"]),
            tp("d", &["audit", "sales order", "Audit"]),
        ];
        canonicalize(&mut t);
        let vals = |i: usize| {
            t[i].properties
                .iter()
                .map(|p| p.value.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(vals(0), ["audit", "ship"]);
        assert_eq!(vals(1), ["audit", "ship"]);
        assert_eq!(vals(2), ["audit", "sales order"]);
        assert_eq!(vals(3), ["audit", "sales order"]); // deduped
    }

    #[test]
    fn json_in_fences() {
        let reply = "```json\n{\"entities\": [\"customer\"], \"role\": \"lookup\"}\n```";
        let e: Extracted = serde_json::from_str(json_object(reply)).unwrap();
        assert_eq!(e.entities, ["customer"]);
        assert_eq!(e.role.as_deref(), Some("lookup"));
    }
}
