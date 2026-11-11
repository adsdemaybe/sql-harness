//! Input: the database schema being indexed. Tables play the role of documents.
//!
//! Source for the PoC is a SQLite file. Another source (e.g. a REST endpoint over
//! Azure SQL) would be another `from_*` function producing the same `Schema`.

use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Where the schema was read from; `ask --nl` runs its queries there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub tables: Vec<Table>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub name: String,
    /// SQLite has no table comments, so this is None for `from_sqlite`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub columns: Vec<Column>,
    #[serde(default)]
    pub foreign_keys: Vec<ForeignKey>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub primary_key: bool,
}

/// A real join (possibly composite). Becomes the strongest edge, and the only kind
/// joins may follow. `columns[i]` references `ref_columns[i]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKey {
    pub columns: Vec<String>,
    pub ref_table: String,
    pub ref_columns: Vec<String>,
}

impl Schema {
    pub fn column_count(&self) -> usize {
        self.tables.iter().map(|t| t.columns.len()).sum()
    }

    pub fn foreign_key_count(&self) -> usize {
        self.tables.iter().map(|t| t.foreign_keys.len()).sum()
    }
}

/// Read tables, columns and foreign keys from a SQLite file, opened read-only.
pub fn from_sqlite(path: &str) -> Result<Schema> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("opening {path}"))?;

    let names: Vec<String> = conn
        .prepare(
            "SELECT name FROM sqlite_master
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
             ORDER BY name",
        )?
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;

    let tables = names
        .into_iter()
        .map(|name| {
            Ok(Table {
                columns: columns(&conn, &name)?,
                foreign_keys: foreign_keys(&conn, &name)?,
                description: None,
                name,
            })
        })
        .collect::<Result<_>>()?;

    Ok(Schema {
        source: Some(path.to_string()),
        tables,
    })
}

fn columns(conn: &Connection, table: &str) -> Result<Vec<Column>> {
    let mut stmt = conn.prepare("SELECT name, type, pk FROM pragma_table_info(?1) ORDER BY cid")?;
    let cols = stmt
        .query_map([table], |row| {
            Ok(Column {
                name: row.get(0)?,
                ty: row.get(1)?,
                primary_key: row.get::<_, i64>(2)? > 0,
            })
        })?
        .collect::<rusqlite::Result<_>>()
        .with_context(|| format!("reading columns of {table}"))?;
    Ok(cols)
}

/// One pragma row per column pair; rows sharing an `id` form one (composite) key.
fn foreign_keys(conn: &Connection, table: &str) -> Result<Vec<ForeignKey>> {
    let mut stmt = conn.prepare(
        r#"SELECT id, "table", "from", "to" FROM pragma_foreign_key_list(?1) ORDER BY id, seq"#,
    )?;
    let rows = stmt.query_map([table], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<String>>(3)?,
        ))
    })?;

    let mut keys: Vec<(i64, ForeignKey, bool)> = Vec::new(); // (id, key, has implicit target)
    for row in rows {
        let (id, ref_table, from, to) = row?;
        if keys.last().map(|(last, ..)| *last) != Some(id) {
            let fk = ForeignKey {
                columns: vec![],
                ref_table,
                ref_columns: vec![],
            };
            keys.push((id, fk, false));
        }
        let (_, fk, implicit) = keys.last_mut().expect("pushed above");
        fk.columns.push(from);
        match to {
            Some(col) => fk.ref_columns.push(col),
            None => *implicit = true,
        }
    }

    // `REFERENCES parent` without columns means the parent's primary key.
    keys.into_iter()
        .map(|(_, mut fk, implicit)| {
            if implicit {
                fk.ref_columns = primary_key(conn, &fk.ref_table)?;
            }
            Ok(fk)
        })
        .collect()
}

fn primary_key(conn: &Connection, table: &str) -> Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT name FROM pragma_table_info(?1) WHERE pk > 0 ORDER BY pk")?;
    let cols = stmt
        .query_map([table], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/adventureworks.db");

    fn adventureworks() -> Schema {
        from_sqlite(AW)
            .expect("data/adventureworks.db missing: run scripts/fetch-adventureworks.sh")
    }

    fn table<'a>(schema: &'a Schema, name: &str) -> &'a Table {
        schema.tables.iter().find(|t| t.name == name).unwrap()
    }

    #[test]
    fn loads_all_tables_and_keys() {
        let schema = adventureworks();
        assert_eq!(schema.tables.len(), 71);
        assert_eq!(schema.foreign_key_count(), 90);
    }

    #[test]
    fn composite_primary_and_foreign_keys() {
        let schema = adventureworks();
        let detail = table(&schema, "Sales.SalesOrderDetail");

        let pk: Vec<_> = detail
            .columns
            .iter()
            .filter(|c| c.primary_key)
            .map(|c| &c.name)
            .collect();
        assert_eq!(pk, ["SalesOrderID", "SalesOrderDetailID"]);

        let fk = detail
            .foreign_keys
            .iter()
            .find(|fk| fk.ref_table == "Sales.SpecialOfferProduct")
            .unwrap();
        assert_eq!(fk.columns, ["SpecialOfferID", "ProductID"]);
        assert_eq!(fk.ref_columns, ["SpecialOfferID", "ProductID"]);
    }

    #[test]
    fn every_foreign_key_targets_a_real_table_and_columns() {
        let schema = adventureworks();
        for t in &schema.tables {
            for fk in &t.foreign_keys {
                let target = table(&schema, &fk.ref_table);
                assert_eq!(
                    fk.columns.len(),
                    fk.ref_columns.len(),
                    "{} -> {}",
                    t.name,
                    fk.ref_table
                );
                for col in &fk.ref_columns {
                    assert!(
                        target.columns.iter().any(|c| &c.name == col),
                        "{}.{col}",
                        fk.ref_table
                    );
                }
            }
        }
    }
}
