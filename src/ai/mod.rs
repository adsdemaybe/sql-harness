//! Everything the harness needs from an AI, behind one trait.
//! Provider-agnostic: nothing outside this folder knows which model is plugged in.

mod google;
mod stub;

pub use google::Google;
#[allow(unused_imports)] // offline placeholder; swap it in at `let ai = ...` in main.rs
pub use stub::Stub;

use anyhow::{Result, bail};
use serde_json::Value;

pub trait Ai {
    /// Text in, text out. Used for property extraction (ingest) and community summaries.
    fn complete(&self, system: &str, prompt: &str) -> Result<String>;

    /// Like `complete`, but hands text to `on_chunk` as it is generated.
    /// Returns the full text. Backends without streaming send one chunk.
    fn complete_stream(
        &self,
        system: &str,
        prompt: &str,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let text = self.complete(system, prompt)?;
        on_chunk(&text);
        Ok(text)
    }

    /// One model turn of a tool-using conversation (the agent loop in agent.rs).
    /// Streams the turn's text through `on_text` and returns a `Message::Assistant`
    /// holding the text and any tool calls the model made.
    fn chat(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Message> {
        let _ = (system, messages, tools, on_text);
        bail!("this AI backend does not support tool use")
    }

    /// Text to vector. Used to land specific questions on graph nodes
    /// (and for the optional description-similarity edges).
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Many texts at once (graph build embeds every node). Backends with a batch
    /// endpoint should override this.
    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// A function the model may call. `parameters` is a JSON Schema object.
#[derive(Debug, Clone)]
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Provider-assigned call id, echoed back with the result when present.
    pub id: Option<String>,
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone)]
pub enum Message {
    User(String),
    Assistant {
        text: String,
        calls: Vec<ToolCall>,
        /// The provider's own encoding of this turn, replayed verbatim on the next
        /// request (Gemini requires its thought signatures back). None = rebuild it.
        raw: Option<Value>,
    },
    /// Results for the previous turn's calls, in the same order.
    ToolResults(Vec<(ToolCall, String)>),
}
