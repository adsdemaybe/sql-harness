//! Run model-written SQL against the source database.
//!
//! Read queries run straight away on a read-only connection. Anything SQLite reports
//! as not read-only (INSERT, UPDATE, DELETE, CREATE, DROP, ATTACH, ...) is a write:
//! it only runs after the human confirms it, on a separate read-write connection.
//! Always: one statement per call, cancelled after `TIMEOUT`, at most `MAX_ROWS` rows.

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OpenFlags, Statement, types::ValueRef};
use serde::Serialize;
use std::time::{Duration, Instant};

pub const MAX_ROWS: usize = 50;
pub const TIMEOUT: Duration = Duration::from_secs(10);
/// Long text cells are cut to this many characters.
const MAX_CELL: usize = 200;

#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    /// More rows existed than `MAX_ROWS`.
    pub truncated: bool,
}

#[derive(Debug)]
pub enum Outcome {
    Rows(QueryResult),
    /// A confirmed write ran; this many rows changed (plus any RETURNING rows).
    Written {
        changed: usize,
        returned: Option<QueryResult>,
    },
    /// The human said no.
    Declined,
}

/// Read-only query; writes are rejected outright.
pub fn run(db: &str, sql: &str) -> Result<QueryResult> {
    match execute(db, sql, &mut |_| false)? {
        Outcome::Rows(r) => Ok(r),
        _ => bail!("only read-only queries are allowed"),
    }
}

/// Run `sql`. Reads run directly; writes call `confirm(sql)` first and only run on `true`.
pub fn execute(db: &str, sql: &str, confirm: &mut dyn FnMut(&str) -> bool) -> Result<Outcome> {
    let start = Instant::now();
    let conn = open(db, false, start)?;
    // `prepare` rejects multiple statements, so "SELECT 1; DROP ..." never runs.
    let mut stmt = conn.prepare(sql)?;
    if stmt.readonly() {
        return Ok(Outcome::Rows(collect(&mut stmt, start)?));
    }
    drop(stmt);
    drop(conn);

    if !confirm(sql) {
        return Ok(Outcome::Declined);
    }
    let start = Instant::now();
    let conn = open(db, true, start)?;
    let mut stmt = conn.prepare(sql)?;
    if stmt.column_count() > 0 {
        // e.g. INSERT ... RETURNING id
        let returned = collect(&mut stmt, start)?;
        drop(stmt);
        Ok(Outcome::Written {
            changed: conn.changes() as usize,
            returned: Some(returned),
        })
    } else {
        let changed = stmt.execute([]).map_err(|e| timeout_or(e, start))?;
        Ok(Outcome::Written {
            changed,
            returned: None,
        })
    }
}

fn open(db: &str, writable: bool, start: Instant) -> Result<Connection> {
    let mode = if writable {
        OpenFlags::SQLITE_OPEN_READ_WRITE
    } else {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    };
    let mut conn = Connection::open_with_flags(db, mode | OpenFlags::SQLITE_OPEN_NO_MUTEX)
        .with_context(|| format!("opening {db}"))?;
    // Returning true from the handler interrupts the running statement.
    conn.progress_handler(10_000, Some(move || start.elapsed() > TIMEOUT))?;
    Ok(conn)
}

fn collect(stmt: &mut Statement, start: Instant) -> Result<QueryResult> {
    let columns: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();
    let mut rows = Vec::new();
    let mut truncated = false;
    let mut cursor = stmt.query([])?;
    while let Some(row) = cursor.next().map_err(|e| timeout_or(e, start))? {
        if rows.len() == MAX_ROWS {
            truncated = true;
            break;
        }
        rows.push(
            (0..columns.len())
                .map(|i| row.get_ref(i).map(cell))
                .collect::<rusqlite::Result<_>>()?,
        );
    }
    Ok(QueryResult {
        columns,
        rows,
        truncated,
    })
}

fn timeout_or(e: rusqlite::Error, start: Instant) -> anyhow::Error {
    if start.elapsed() > TIMEOUT {
        anyhow::anyhow!("query cancelled after {}s", TIMEOUT.as_secs())
    } else {
        e.into()
    }
}

fn cell(v: ValueRef) -> String {
    match v {
        ValueRef::Null => "NULL".into(),
        ValueRef::Integer(i) => i.to_string(),
        ValueRef::Real(f) => {
            let s = format!("{f:.4}");
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        }
        ValueRef::Text(t) => {
            let s = String::from_utf8_lossy(t);
            match s.char_indices().nth(MAX_CELL) {
                Some((cut, _)) => format!("{}...", &s[..cut]),
                None => s.into_owned(),
            }
        }
        ValueRef::Blob(b) => format!("<blob {} bytes>", b.len()),
    }
}

impl QueryResult {
    /// Pipe-separated table, used both in prompts and on the terminal.
    pub fn to_table(&self, max_rows: usize) -> String {
        let mut out = self.columns.join(" | ");
        out.push('\n');
        for row in self.rows.iter().take(max_rows) {
            out += &row.join(" | ");
            out.push('\n');
        }
        let hidden = self.rows.len().saturating_sub(max_rows);
        if hidden > 0 || self.truncated {
            let more = if self.truncated { "+" } else { "" };
            out += &format!("... {hidden}{more} more rows\n");
        }
        if self.rows.is_empty() {
            out += "(no rows)\n";
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/adventureworks.db");

    #[test]
    fn runs_select() {
        let r = run(
            AW,
            r#"SELECT Name, TerritoryID FROM "Sales.SalesTerritory" ORDER BY TerritoryID"#,
        )
        .unwrap();
        assert_eq!(r.columns, ["Name", "TerritoryID"]);
        assert_eq!(r.rows.len(), 10);
        assert_eq!(r.rows[0], ["Northwest", "1"]);
        assert!(!r.truncated);
    }

    #[test]
    fn caps_rows() {
        let r = run(AW, r#"SELECT ProductID FROM "Production.Product""#).unwrap();
        assert_eq!(r.rows.len(), MAX_ROWS);
        assert!(r.truncated);
    }

    #[test]
    fn rejects_writes_and_multiple_statements() {
        assert!(run(AW, r#"DELETE FROM "Sales.SalesTerritory""#).is_err());
        assert!(run(AW, r#"SELECT 1; DELETE FROM "Sales.SalesTerritory""#).is_err());
        assert!(run(AW, "ATTACH DATABASE 'x.db' AS x").is_err());
    }

    #[test]
    fn writes_need_confirmation() {
        let path = std::env::temp_dir().join(format!("sql-harness-test-{}.db", std::process::id()));
        let db = path.to_str().unwrap();
        Connection::open(db)
            .unwrap()
            .execute_batch("CREATE TABLE t (x INTEGER)")
            .unwrap();
        let count = || run(db, "SELECT COUNT(*) FROM t").unwrap().rows[0][0].clone();

        let mut asked = Vec::new();
        let out = execute(db, "INSERT INTO t VALUES (1)", &mut |sql| {
            asked.push(sql.to_string());
            false
        })
        .unwrap();
        assert!(matches!(out, Outcome::Declined));
        assert_eq!(asked, ["INSERT INTO t VALUES (1)"]);
        assert_eq!(count(), "0");

        let out = execute(db, "INSERT INTO t VALUES (1), (2)", &mut |_| true).unwrap();
        assert!(matches!(
            out,
            Outcome::Written {
                changed: 2,
                returned: None
            }
        ));
        assert_eq!(count(), "2");

        // reads never ask
        let out = execute(db, "SELECT x FROM t", &mut |_| panic!("asked for a read")).unwrap();
        assert!(matches!(out, Outcome::Rows(_)));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn times_out() {
        // Unbounded recursive CTE never finishes on its own.
        let sql = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) \
                   SELECT count(*) FROM n";
        let err = run(AW, sql).unwrap_err().to_string();
        assert!(
            err.contains("cancelled") || err.contains("interrupt"),
            "{err}"
        );
    }
}
