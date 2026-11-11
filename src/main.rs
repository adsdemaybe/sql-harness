//! Schema GraphRAG PoC.
//!
//! Pipeline (each stage writes JSON to `out/` so the next one can pick it up):
//!   ingest [db]  SQLite file -> out/schema.json, out/properties.json
//!   build        schema + properties -> out/graph.json, out/tree.json
//!   ask          question -> relevant tables, join paths, join conditions (JSON)
//!   ask --nl     agent: the AI runs SQL (as many queries as it needs) and streams an
//!                answer, the tables/joins it used and the final SQL
//!   chat         the same agent, interactive, with follow-up questions
//!
//! Write queries proposed by the agent only run after a y/N confirmation.
//!
//! `ingest` reads the schema; the agent queries the database (query.rs). Everything
//! else works from the snapshot in out/.

#![allow(dead_code)] // tree::attach is for incremental schema changes; not on the CLI yet

mod agent;
mod ai;
mod graph;
mod ingest;
mod query;
mod schema;
mod summarize;
mod traverse;
mod tree;

use anyhow::{Context, Result, bail};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs,
    io::{self, IsTerminal, Write},
    path::Path,
};

const DEFAULT_DB: &str = "data/adventureworks.db";
const OUT_DIR: &str = "out";

fn main() -> Result<()> {
    dotenvy::dotenv().ok(); // load .env if present; real env vars take precedence
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("ingest") => {
            let db = args.get(2).map(String::as_str).unwrap_or(DEFAULT_DB);
            let schema = schema::from_sqlite(db)?;
            eprintln!(
                "{db}: {} tables, {} columns, {} foreign keys",
                schema.tables.len(),
                schema.column_count(),
                schema.foreign_key_count(),
            );
            write_json("schema.json", &schema)?;

            let ai = ai::Google::from_env()?;
            let props = ingest::run(&schema, &ai)?;
            write_json("properties.json", &props)?;
        }
        Some("build") => {
            let ai = ai::Google::from_env()?;
            let schema: schema::Schema = read_json("schema.json")?;
            let props: Vec<ingest::TableProperties> = read_json("properties.json")?;
            let mut graph = graph::build(&schema, &props);
            graph::embed(&mut graph, &schema, &ai)?;
            graph::add_similarity_edges(&mut graph);
            let mut tree = tree::cluster(&graph, tree::SEED);
            eprintln!(
                "graph: {} nodes, {} edges; tree: {} communities",
                graph.nodes.len(),
                graph.edges.len(),
                tree.communities.len(),
            );
            summarize::run(&mut tree, &graph, &schema, &ai)?;
            write_json("graph.json", &graph)?;
            write_json("tree.json", &tree)?;
        }
        Some(cmd @ ("ask" | "chat")) => {
            let chat = cmd == "chat";
            let opts = AskOpts::parse(&args[2..], !chat)?;
            let ai = ai::Google::from_env()?;
            let schema: schema::Schema = read_json("schema.json")?;
            let graph: graph::Graph = read_json("graph.json")?;
            let tree: tree::Tree = read_json("tree.json")?;

            if !chat && !opts.nl {
                let hits = traverse::retrieve(&opts.question, &graph, &tree, &schema, &ai)?;
                println!("{}", serde_json::to_string_pretty(&hits)?);
                return Ok(());
            }

            let db = opts
                .db
                .or(schema.source.clone())
                .unwrap_or_else(|| DEFAULT_DB.into());
            let mut agent = agent::Agent::new(&ai, &graph, &schema, db);
            let mut ui = TerminalUi::default();
            let mut ask = |agent: &mut agent::Agent, question: &str, first: bool| -> Result<()> {
                // The graph retrieval seeds the first question; follow-ups can use the
                // agent's search_schema tool instead.
                let hits = if first {
                    let hits = traverse::retrieve(question, &graph, &tree, &schema, &ai)?;
                    eprintln!("tables ({}): {}\n", hits.mode, hits.tables.join(", "));
                    Some(hits)
                } else {
                    None
                };
                agent.ask(question, hits.as_ref(), &mut ui)?;
                ui.end_turn();
                Ok(())
            };

            if !chat {
                return ask(&mut agent, &opts.question, true);
            }
            eprintln!("chat with the database. Ctrl-D or `exit` to quit.");
            let mut first = true;
            if !opts.question.is_empty() {
                ask(&mut agent, &opts.question, true)?;
                first = false;
            }
            while let Some(line) = prompt_line("\n> ") {
                let q = line.trim();
                if matches!(q, "exit" | "quit") {
                    break;
                }
                if q.is_empty() {
                    continue;
                }
                match ask(&mut agent, q, first) {
                    Ok(()) => first = false,
                    Err(e) => eprintln!("error: {e:#}"),
                }
            }
        }
        _ => bail!(
            "usage: sql-harness <ingest [path.db] | build | ask [--nl] [--db path] <question>>"
        ),
    }
    Ok(())
}

struct AskOpts {
    question: String,
    nl: bool,
    db: Option<String>,
}

impl AskOpts {
    const USAGE: &str = "usage: sql-harness ask [--nl] [--db path.db] <question>";

    fn parse(args: &[String], need_question: bool) -> Result<Self> {
        let mut opts = AskOpts {
            question: String::new(),
            nl: false,
            db: None,
        };
        let mut words = Vec::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--nl" | "--natural" => opts.nl = true,
                "--db" => opts.db = Some(it.next().context(Self::USAGE)?.clone()),
                flag if flag.starts_with("--") => bail!("unknown flag {flag}; {}", Self::USAGE),
                word => words.push(word),
            }
        }
        opts.question = words.join(" ");
        if need_question && opts.question.is_empty() {
            bail!(Self::USAGE);
        }
        Ok(opts)
    }
}

/// Agent output on the terminal: answer text streams to stdout, tool activity and the
/// write confirmation go to stderr.
#[derive(Default)]
struct TerminalUi {
    /// stdout has text without a trailing newline.
    mid_line: bool,
}

impl TerminalUi {
    fn break_line(&mut self) {
        if self.mid_line {
            println!();
            self.mid_line = false;
        }
    }

    fn end_turn(&mut self) {
        self.break_line();
    }
}

impl agent::Ui for TerminalUi {
    fn text(&mut self, chunk: &str) {
        let mut out = io::stdout();
        let _ = write!(out, "{chunk}");
        let _ = out.flush();
        self.mid_line = !chunk.ends_with('\n');
    }

    fn tool_call(&mut self, call: &ai::ToolCall) {
        self.break_line();
        match call.name.as_str() {
            "run_sql" => {
                let sql = call.args["sql"].as_str().unwrap_or_default();
                eprintln!("  > run_sql");
                for line in sql.lines() {
                    eprintln!("    {line}");
                }
            }
            name => eprintln!("  > {name} {}", call.args),
        }
    }

    fn tool_result(&mut self, summary: &str) {
        eprintln!("    = {summary}");
    }

    fn confirm_write(&mut self, sql: &str) -> bool {
        self.break_line();
        if !io::stdin().is_terminal() {
            eprintln!("  ! write query refused: no terminal to confirm it");
            return false;
        }
        eprintln!("\n  ! The agent wants to run a WRITE query on the database:\n");
        for line in sql.lines() {
            eprintln!("      {line}");
        }
        let answer = prompt_line("\n  Run it? [y/N] ").unwrap_or_default();
        matches!(answer.trim().to_lowercase().as_str(), "y" | "yes")
    }
}

/// Print `prompt` to stderr and read one line from stdin; None on EOF.
fn prompt_line(prompt: &str) -> Option<String> {
    eprint!("{prompt}");
    let _ = io::stderr().flush();
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line),
    }
}

fn write_json<T: Serialize>(name: &str, value: &T) -> Result<()> {
    fs::create_dir_all(OUT_DIR)?;
    let path = Path::new(OUT_DIR).join(name);
    fs::write(&path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

fn read_json<T: DeserializeOwned>(name: &str) -> Result<T> {
    let path = Path::new(OUT_DIR).join(name);
    let text = fs::read_to_string(&path)
        .with_context(|| format!("reading {} (run the earlier stage first)", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// Whole pipeline offline: a scripted AI stands in for Gemini, everything else is real
/// (AdventureWorks schema, Leiden, retrieval, the agent loop, SQL execution).
#[cfg(test)]
mod e2e {
    use super::*;
    use ai::{Message, Tool, ToolCall};
    use anyhow::Result;
    use serde_json::json;

    struct FakeAi;

    impl ai::Ai for FakeAi {
        fn complete(&self, system: &str, prompt: &str) -> Result<String> {
            Ok(if system.contains("extract properties") {
                // entity = schema prefix ("Sales", "Production", ...)
                let table = prompt.lines().next().unwrap().trim_start_matches("Table: ");
                let entity = table.split('.').next().unwrap().to_lowercase();
                format!(r#"{{"entities": ["{entity}"], "role": "lookup"}}"#)
            } else if system.contains("summarize") {
                "Some group\nTables about something.".into()
            } else if system.contains("Classify") {
                "specific".into()
            } else if system.contains("pick database tables") {
                r#"["Sales.SalesTerritory"]"#.into()
            } else {
                panic!("unexpected prompt: {system}")
            })
        }

        /// Script, by how many tool rounds have happened so far:
        /// broken query -> write (declined) -> good query -> answer.
        fn chat(
            &self,
            _system: &str,
            messages: &[Message],
            tools: &[Tool],
            on_text: &mut dyn FnMut(&str),
        ) -> Result<Message> {
            assert_eq!(tools.len(), 4);
            let results: Vec<&Vec<(ToolCall, String)>> = messages
                .iter()
                .filter_map(|m| match m {
                    Message::ToolResults(r) => Some(r),
                    _ => None,
                })
                .collect();
            let call = |sql: &str| {
                Ok(Message::Assistant {
                    text: String::new(),
                    calls: vec![ToolCall {
                        id: None,
                        name: "run_sql".into(),
                        args: json!({ "sql": sql }),
                    }],
                    raw: None,
                })
            };
            match results.len() {
                0 => {
                    let Message::User(first) = &messages[0] else {
                        panic!()
                    };
                    assert!(
                        first.contains("Sales.SalesTerritory: TerritoryID (PK)"),
                        "{first}"
                    );
                    call(r#"SELECT nope FROM "Sales.SalesTerritory""#)
                }
                1 => {
                    assert!(results[0][0].1.starts_with("Error"), "{}", results[0][0].1);
                    call(r#"DELETE FROM "Sales.SalesTerritory""#)
                }
                2 => {
                    assert!(results[1][0].1.contains("declined"));
                    call(r#"SELECT COUNT(*) AS territories FROM "Sales.SalesTerritory""#)
                }
                _ => {
                    assert!(
                        results[2][0].1.contains("territories\n10\n"),
                        "{}",
                        results[2][0].1
                    );
                    on_text("There are ");
                    on_text("10 territories.");
                    Ok(Message::Assistant {
                        text: "There are 10 territories.".into(),
                        calls: vec![],
                        raw: None,
                    })
                }
            }
        }

        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            // bag of character trigrams hashed into 64 dims
            let mut v = vec![0.0; 64];
            let b = text.to_lowercase().into_bytes();
            for w in b.windows(3) {
                let h = w
                    .iter()
                    .fold(7u32, |h, &c| h.wrapping_mul(31).wrapping_add(c as u32));
                v[(h % 64) as usize] += 1.0;
            }
            Ok(v)
        }
    }

    #[derive(Default)]
    struct Recorder {
        text: String,
        summaries: Vec<String>,
        asked: Vec<String>,
    }

    impl agent::Ui for Recorder {
        fn text(&mut self, chunk: &str) {
            self.text.push_str(chunk);
        }
        fn tool_call(&mut self, _call: &ToolCall) {}
        fn tool_result(&mut self, summary: &str) {
            self.summaries.push(summary.to_string());
        }
        fn confirm_write(&mut self, sql: &str) -> bool {
            self.asked.push(sql.to_string());
            false // human says no
        }
    }

    #[test]
    fn ingest_build_ask_agent() {
        let ai = FakeAi;
        let db = concat!(env!("CARGO_MANIFEST_DIR"), "/data/adventureworks.db");
        let schema = schema::from_sqlite(db).unwrap();

        let props = ingest::run(&schema, &ai).unwrap();
        assert_eq!(props.len(), 71);

        let mut graph = graph::build(&schema, &props);
        graph::embed(&mut graph, &schema, &ai).unwrap();
        graph::add_similarity_edges(&mut graph);
        let mut tree = tree::cluster(&graph, tree::SEED);
        summarize::run(&mut tree, &graph, &schema, &ai).unwrap();
        assert!(tree.communities.iter().all(|c| !c.summary.is_empty()));

        let q = "how many sales territories are there?";
        let hits = traverse::retrieve(q, &graph, &tree, &schema, &ai).unwrap();
        assert_eq!(hits.tables, ["Sales.SalesTerritory"]);

        let mut agent = agent::Agent::new(&ai, &graph, &schema, db.into());
        let mut ui = Recorder::default();
        let answer = agent.ask(q, Some(&hits), &mut ui).unwrap();
        assert_eq!(answer, "There are 10 territories.");
        assert_eq!(ui.text, answer, "answer was streamed");
        assert_eq!(
            ui.asked,
            [r#"DELETE FROM "Sales.SalesTerritory""#],
            "write needed a yes"
        );
        assert_eq!(ui.summaries.len(), 3);
        assert!(ui.summaries[0].starts_with("error"));
        assert_eq!(ui.summaries[1], "declined");
        assert_eq!(ui.summaries[2], "1 rows");
        // the declined DELETE really didn't run
        let left = query::run(db, r#"SELECT COUNT(*) FROM "Sales.SalesTerritory""#).unwrap();
        assert_eq!(left.rows, [["10"]]);
    }
}
