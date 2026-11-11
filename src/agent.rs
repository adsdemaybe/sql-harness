//! The agent behind `ask --nl` and `chat`: the AI answers by calling tools in a loop,
//! running as many SQL queries as the analysis needs.
//!
//! Tools:
//!   run_sql          reads run immediately; writes run only after the human confirms
//!   search_schema    graph search (embeddings + expansion) for more tables
//!   describe_tables  columns, keys and join conditions for named tables
//!   list_tables      every table name
//!
//! Provider-agnostic: it only talks to `Ai::chat`. Terminal I/O goes through `Ui`.

use crate::ai::{Ai, Message, Tool, ToolCall};
use crate::graph::Graph;
use crate::query::{self, Outcome};
use crate::schema::Schema;
use crate::summarize::describe;
use crate::traverse::{self, Retrieved};
use anyhow::Result;
use serde_json::{Value, json};

/// Model turns per question. When reached, the model is told to answer with what it has.
pub const MAX_STEPS: usize = 15;
/// Tables returned by one `search_schema` call.
const SEARCH_K: usize = 8;

const SYSTEM: &str = r#"You are a data analyst agent working directly on a SQLite database.
Answer the user's question by running SQL with the run_sql tool. Run as many queries as the
analysis needs: look at the data, check your assumptions, and follow up on what you find.

Rules:
- Table names contain a dot, so always double-quote them: "Sales.SalesOrderHeader".
- Results are capped at 50 rows. Aggregate, ORDER BY and LIMIT instead of dumping tables.
- Dates are TEXT like '2013-07-31 00:00:00'; use strftime() or string comparison.
- If you need tables beyond the ones suggested, use search_schema, describe_tables or list_tables.
- If a query errors, read the error, fix the query and try again.
- Only run write queries (INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, ...) when the user
  explicitly asks to change data. The user must confirm every write before it runs; if they
  decline, do not retry it.
- Never invent data. Every number and name in your answer must come from a query result.
- If the data doesn't match the question's assumptions (different years, missing values,
  ambiguous names), say so plainly instead of silently substituting something else.

When you are done, reply without calling a tool, in this shape:
1. The direct answer, using the actual numbers and names from the results.
2. How you got it: which tables you used and how they join.
3. The final SQL, in a ```sql block."#;

/// Where the agent's activity goes (the terminal in main.rs, a recorder in tests).
pub trait Ui {
    /// Streamed model text.
    fn text(&mut self, chunk: &str);
    fn tool_call(&mut self, call: &ToolCall);
    /// One-line outcome of the call just announced.
    fn tool_result(&mut self, summary: &str);
    /// A write query wants to run. Return true to allow it.
    fn confirm_write(&mut self, sql: &str) -> bool;
}

pub struct Agent<'a> {
    ai: &'a dyn Ai,
    graph: &'a Graph,
    schema: &'a Schema,
    db: String,
    /// The whole conversation, so follow-up questions keep their context.
    messages: Vec<Message>,
}

impl<'a> Agent<'a> {
    pub fn new(ai: &'a dyn Ai, graph: &'a Graph, schema: &'a Schema, db: String) -> Self {
        Agent {
            ai,
            graph,
            schema,
            db,
            messages: Vec::new(),
        }
    }

    /// Ask a question (first or follow-up) and run the tool loop until the model
    /// answers. `hits` is the graph retrieval for this question, given as a head start.
    /// Returns the final answer text (it was also streamed through `ui.text`).
    pub fn ask(
        &mut self,
        question: &str,
        hits: Option<&Retrieved>,
        ui: &mut dyn Ui,
    ) -> Result<String> {
        let before = self.messages.len();
        let result = self.ask_inner(question, hits, ui);
        if result.is_err() {
            // Drop the half-finished exchange so the conversation stays valid.
            self.messages.truncate(before);
        }
        result
    }

    fn ask_inner(
        &mut self,
        question: &str,
        hits: Option<&Retrieved>,
        ui: &mut dyn Ui,
    ) -> Result<String> {
        self.messages
            .push(Message::User(self.opening(question, hits)));
        let tools = tools();
        for step in 0.. {
            let reply = self
                .ai
                .chat(SYSTEM, &self.messages, &tools, &mut |t| ui.text(t))?;
            let Message::Assistant { text, calls, .. } = &reply else {
                unreachable!("chat returns an assistant message")
            };
            let (text, calls) = (text.clone(), calls.clone());
            self.messages.push(reply);
            if calls.is_empty() {
                return Ok(text);
            }

            let out_of_steps = step + 1 >= MAX_STEPS;
            let results = calls
                .into_iter()
                .map(|call| {
                    let output = if out_of_steps {
                        "Not run: the tool-call limit was reached. Answer now with what you have."
                            .to_string()
                    } else {
                        ui.tool_call(&call);
                        let (output, summary) = self.run_tool(&call, ui);
                        ui.tool_result(&summary);
                        output
                    };
                    (call, output)
                })
                .collect();
            self.messages.push(Message::ToolResults(results));
            if step + 1 > MAX_STEPS {
                // Model kept calling tools after being told to stop.
                return Ok(text);
            }
        }
        unreachable!()
    }

    /// The user message: the question, plus the retrieved tables on the first turn.
    fn opening(&self, question: &str, hits: Option<&Retrieved>) -> String {
        let Some(hits) = hits else {
            return question.to_string();
        };
        let tables = traverse::by_name(self.schema);
        let mut p = format!("Question: {question}\n\nThe schema graph suggests these tables:\n");
        for name in &hits.tables {
            if let Some(t) = tables.get(name.as_str()) {
                p += &describe(t);
                p += "\n";
            }
        }
        if !hits.join_on.is_empty() {
            p += "\nJoin conditions between them (from foreign keys):\n";
            for j in &hits.join_on {
                p += &format!("- {j}\n");
            }
        }
        p
    }

    /// Returns (output for the model, one-line summary for the human).
    fn run_tool(&self, call: &ToolCall, ui: &mut dyn Ui) -> (String, String) {
        let arg = |k: &str| call.args[k].as_str().unwrap_or_default().to_string();
        match call.name.as_str() {
            "run_sql" => self.run_sql(&arg("sql"), ui),
            "search_schema" => match traverse::search(&arg("query"), self.graph, self.ai, SEARCH_K)
            {
                Ok(names) => {
                    let summary = format!("{} tables", names.len());
                    (self.describe_named(&names), summary)
                }
                Err(e) => (format!("Error: {e:#}"), format!("error: {e}")),
            },
            "describe_tables" => {
                let names: Vec<String> = call.args["tables"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                (
                    self.describe_named(&names),
                    format!("{} tables", names.len()),
                )
            }
            "list_tables" => {
                let names: Vec<&str> = self.schema.tables.iter().map(|t| t.name.as_str()).collect();
                (names.join("\n"), format!("{} tables", names.len()))
            }
            other => (
                format!("Error: unknown tool {other}"),
                format!("unknown tool {other}"),
            ),
        }
    }

    fn run_sql(&self, sql: &str, ui: &mut dyn Ui) -> (String, String) {
        match query::execute(&self.db, sql, &mut |s| ui.confirm_write(s)) {
            Ok(Outcome::Rows(r)) => {
                let mut out = r.to_table(query::MAX_ROWS);
                if r.truncated {
                    out += &format!("(capped at {} rows; more exist)\n", query::MAX_ROWS);
                }
                let more = if r.truncated { "+" } else { "" };
                (out, format!("{}{more} rows", r.rows.len()))
            }
            Ok(Outcome::Written { changed, returned }) => {
                let mut out = format!("OK: {changed} rows changed.\n");
                if let Some(r) = returned {
                    out += &r.to_table(query::MAX_ROWS);
                }
                let first = sql
                    .trim_start()
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_uppercase();
                if matches!(first.as_str(), "CREATE" | "DROP" | "ALTER") {
                    out += "Note: the schema changed. The schema graph is stale until `ingest` and \
                            `build` are rerun.\n";
                }
                (out, format!("write ran: {changed} rows changed"))
            }
            Ok(Outcome::Declined) => (
                "The user declined to run this write query. Do not retry it; tell the user what \
                 it would have done."
                    .into(),
                "declined".into(),
            ),
            Err(e) => (format!("Error: {e:#}"), format!("error: {e}")),
        }
    }

    fn describe_named(&self, names: &[String]) -> String {
        let tables = traverse::by_name(self.schema);
        let mut out = String::new();
        for n in names {
            match tables.get(n.as_str()) {
                Some(t) => out += &describe(t),
                None => out += &format!("- {n}: unknown table"),
            }
            out.push('\n');
        }
        let mut joins = Vec::new();
        for (i, a) in names.iter().enumerate() {
            for b in &names[i + 1..] {
                joins.extend(traverse::join_conditions(&tables, a, b));
            }
        }
        if !joins.is_empty() {
            out += "Join conditions:\n";
            for j in joins {
                out += &format!("- {j}\n");
            }
        }
        out
    }
}

fn tools() -> Vec<Tool> {
    let object = |props: Value, required: &[&str]| json!({ "type": "object", "properties": props, "required": required });
    vec![
        Tool {
            name: "run_sql",
            description: "Run one SQLite statement on the database and get the result rows \
                          (max 50). Read queries run immediately. Write queries (INSERT, \
                          UPDATE, DELETE, CREATE, ...) are shown to the user, who must confirm \
                          before they run.",
            parameters: object(
                json!({ "sql": { "type": "string", "description": "A single SQLite statement." } }),
                &["sql"],
            ),
        },
        Tool {
            name: "search_schema",
            description: "Find tables relevant to a topic using the schema graph. Returns \
                          tables with their columns and join conditions.",
            parameters: object(
                json!({ "query": { "type": "string", "description": "What to look for, e.g. 'product categories'." } }),
                &["query"],
            ),
        },
        Tool {
            name: "describe_tables",
            description: "Columns, primary keys, foreign keys and join conditions for the \
                          named tables.",
            parameters: object(
                json!({ "tables": { "type": "array", "items": { "type": "string" }, "description": "Exact table names." } }),
                &["tables"],
            ),
        },
        Tool {
            name: "list_tables",
            description: "Names of every table in the database.",
            parameters: json!({ "type": "object", "properties": {} }),
        },
    ]
}
