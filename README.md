# sql-harness

PoC: schema GraphRAG with Leiden. Tables are the documents, extracted properties are
the entities, and hierarchical Leiden over both gives a tree the model can traverse.

```
data/adventureworks.db   test database (gitignored; fetch with the script below)
scripts/
  fetch-adventureworks.sh  downloads + checksums the AdventureWorks SQLite file
out/                     generated artifacts (gitignored)
src/
  main.rs            CLI: ingest | build | ask | chat (+ terminal UI for the agent)
  schema.rs          schema types + SQLite loader (tables, columns, PKs, FKs)
  ai/                provider-agnostic AI interface
    mod.rs           `Ai` trait: complete(), chat() with tools, embed()
    google.rs        Google AI (Gemini API) backend, the one main.rs uses
    stub.rs          offline placeholder backend
  ingest.rs          extract properties per table, canonicalize
  graph.rs           table + property nodes; join / property / similarity edges
  tree.rs            hierarchical Leiden (fixed seed), incremental attach
  summarize.rs       bottom-up community summaries
  traverse.rs        descend (vague), land + expand (specific), select, join paths
  agent.rs           ask --nl / chat: tool-calling loop (run_sql, search_schema, ...)
  query.rs           SQL execution: reads run directly, writes need a y/N confirmation
```

## Run

```sh
./scripts/fetch-adventureworks.sh        # once: -> data/adventureworks.db (~127 MB)
# put GEMINI_API_KEY in .env (optional: GEMINI_MODEL, GEMINI_EMBED_MODEL,
#   GEMINI_SECS_PER_CALL = gap between model calls, default 5 for the 15 RPM free tier)

cargo run -- ingest [path.db]            # -> out/schema.json, out/properties.json
cargo run -- build                       # -> out/graph.json, out/tree.json
cargo run -- ask "which products sold best in each territory?"       # JSON: tables + joins
cargo run -- ask --nl "which products sold best in each territory?"  # agent: runs SQL, streams the answer
cargo run -- chat                                                     # same agent, with follow-ups

cargo test                               # schema loader against AdventureWorks
```

`ingest` reads the schema; the agent queries the same file (or `--db path`). Read queries run
immediately; any write (INSERT/UPDATE/DELETE/DDL) is shown first and only runs if you answer `y`.
Everything else works from the snapshot in `out/`.
AdventureWorks: [demo-db/adventureworks](https://github.com/demo-db/adventureworks)
(Microsoft's OLTP sample converted to SQLite, MIT).
