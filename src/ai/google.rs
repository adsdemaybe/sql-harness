//! Google AI (Gemini API) backend over plain REST.
//!
//! Env: GEMINI_API_KEY (or GOOGLE_API_KEY), optional GEMINI_MODEL / GEMINI_EMBED_MODEL,
//! optional GEMINI_SECS_PER_CALL (spacing between generation calls, default 5).
//! Streaming uses streamGenerateContent with server-sent events (`alt=sse`).

use super::{Ai, Message, Tool, ToolCall};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    cell::Cell,
    io::{BufRead, BufReader},
    thread,
    time::{Duration, Instant},
};
use ureq::{Body, http::Response};

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const DEFAULT_MODEL: &str = "gemini-3.6-flash";
const DEFAULT_EMBED_MODEL: &str = "gemini-embedding-2";
/// Truncated (Matryoshka) embedding size; keeps graph.json small. Cosine doesn't need
/// the vectors normalized.
const EMBED_DIM: u32 = 768;
/// Texts per batchEmbedContents call. The free tier allows 100 embed requests per
/// minute per model, and every text in a batch counts as one request, so one full
/// batch per `EMBED_WINDOW` stays under it.
const EMBED_BATCH: usize = 100;
const EMBED_WINDOW: Duration = Duration::from_secs(61);
/// Retries on 429 / 5xx: wait as long as the server's RetryInfo says, else 2s, 4s, 8s, ...
const RETRIES: u32 = 5;
/// Free tier allows 15 requests/min for generateContent; one every 5s (12/min) stays under it.
const DEFAULT_SECS_PER_CALL: f64 = 5.0;

pub struct Google {
    api_key: String,
    model: String,
    embed_model: String,
    /// Minimum gap between generateContent requests.
    min_interval: Duration,
    last_generate: Cell<Option<Instant>>,
}

impl Google {
    pub fn from_env() -> Result<Self> {
        // Blank counts as unset (the .env template ships with `GEMINI_API_KEY=`).
        let api_key = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
            .iter()
            .filter_map(|name| std::env::var(name).ok())
            .find(|key| !key.trim().is_empty())
            .context("set GEMINI_API_KEY in .env to use the Google AI backend")?;
        Ok(Self {
            api_key,
            model: std::env::var("GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into()),
            embed_model: std::env::var("GEMINI_EMBED_MODEL")
                .unwrap_or_else(|_| DEFAULT_EMBED_MODEL.into()),
            min_interval: Duration::from_secs_f64(
                std::env::var("GEMINI_SECS_PER_CALL")
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(DEFAULT_SECS_PER_CALL),
            ),
            last_generate: Cell::new(None),
        })
    }

    /// Sleep until `min_interval` has passed since the previous generateContent request.
    fn pace(&self) {
        if let Some(last) = self.last_generate.get() {
            let elapsed = last.elapsed();
            if elapsed < self.min_interval {
                thread::sleep(self.min_interval - elapsed);
            }
        }
        self.last_generate.set(Some(Instant::now()));
    }

    /// POST {BASE_URL}/{model}:{method} and return the successful response.
    /// Rate limits and 5xx are retried (honouring Google's retryDelay); other non-2xx
    /// responses surface Google's error body. Generation calls are paced.
    fn send(&self, model: &str, method: &str, body: &Value) -> Result<Response<Body>> {
        let url = format!("{BASE_URL}/{model}:{method}");
        let generation =
            method.starts_with("generateContent") || method.starts_with("streamGenerateContent");
        for attempt in 0.. {
            if generation {
                self.pace(); // retries count against the limit too
            }
            let mut resp = ureq::post(&url)
                .header("x-goog-api-key", &self.api_key)
                .config()
                .http_status_as_error(false)
                .build()
                .send_json(body)
                .with_context(|| format!("calling {model}:{method}"))?;

            let status = resp.status();
            if status.is_success() {
                return Ok(resp);
            }
            let text = resp.body_mut().read_to_string().unwrap_or_default();
            let retryable = status.as_u16() == 429 || status.is_server_error();
            if !retryable || attempt >= RETRIES {
                bail!("{model}:{method} returned {status}: {text}");
            }
            let wait = retry_delay(&text).unwrap_or(Duration::from_secs(2u64 << attempt));
            eprintln!(
                "{model}:{method} returned {status}, retrying in {}s",
                wait.as_secs()
            );
            thread::sleep(wait);
        }
        unreachable!()
    }

    /// `send` + parse the JSON body.
    fn call(&self, model: &str, method: &str, body: &Value) -> Result<Value> {
        let mut resp = self.send(model, method, body)?;
        let text = resp.body_mut().read_to_string()?;
        Ok(serde_json::from_str(&text)?)
    }
}

fn generate_body(system: &str, prompt: &str) -> Value {
    let mut body = json!({
        "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
    });
    if !system.is_empty() {
        body["systemInstruction"] = json!({ "parts": [{ "text": system }] });
    }
    body
}

/// Text of one generateContent response (or one streamed chunk), skipping thought parts.
/// Errors if the prompt was blocked or generation stopped for a reason other than STOP.
fn response_text(resp: &Value) -> Result<String> {
    if let Some(reason) = resp["promptFeedback"]["blockReason"].as_str() {
        bail!("prompt blocked: {reason}");
    }
    let candidate = &resp["candidates"][0];
    match candidate["finishReason"].as_str() {
        None | Some("STOP") | Some("FINISH_REASON_UNSPECIFIED") => {}
        Some(reason) => bail!("generation stopped early: {reason}"),
    }
    Ok(candidate["content"]["parts"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|part| part["thought"] != true)
        .filter_map(|part| part["text"].as_str())
        .collect())
}

/// Provider-neutral messages -> Gemini `contents`.
fn contents(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| match m {
            Message::User(text) => json!({ "role": "user", "parts": [{ "text": text }] }),
            Message::Assistant { raw: Some(raw), .. } => raw.clone(),
            Message::Assistant {
                text,
                calls,
                raw: None,
            } => {
                let mut parts = Vec::new();
                if !text.is_empty() {
                    parts.push(json!({ "text": text }));
                }
                for c in calls {
                    parts.push(json!({ "functionCall": { "name": c.name, "args": c.args } }));
                }
                json!({ "role": "model", "parts": parts })
            }
            Message::ToolResults(results) => {
                let parts: Vec<Value> = results
                    .iter()
                    .map(|(call, output)| {
                        let mut resp = json!({
                            "name": call.name,
                            "response": { "result": output },
                        });
                        if let Some(id) = &call.id {
                            resp["id"] = json!(id);
                        }
                        json!({ "functionResponse": resp })
                    })
                    .collect();
                json!({ "role": "user", "parts": parts })
            }
        })
        .collect()
}

/// Server-sent events: `data: <json>` lines, events separated by blank lines.
/// Calls `on_event` with each event's JSON.
fn read_sse(reader: impl BufRead, mut on_event: impl FnMut(Value) -> Result<()>) -> Result<()> {
    let mut data = String::new();
    let mut flush = |data: &mut String| -> Result<()> {
        if !data.trim().is_empty() {
            let event = serde_json::from_str(data).context("bad stream event")?;
            on_event(event)?;
        }
        data.clear();
        Ok(())
    };
    for line in reader.lines() {
        let line = line.context("reading stream")?;
        if let Some(rest) = line.strip_prefix("data:") {
            data.push_str(rest.trim_start());
        } else if line.trim().is_empty() {
            flush(&mut data)?;
        }
    }
    flush(&mut data)
}

/// Google's 429 body carries `"retryDelay": "10s"` in a RetryInfo detail. Add a second
/// of slack so we don't land right on the boundary.
fn retry_delay(body: &str) -> Option<Duration> {
    let v: Value = serde_json::from_str(body).ok()?;
    let delay = v["error"]["details"]
        .as_array()?
        .iter()
        .find_map(|d| d["retryDelay"].as_str())?;
    let secs: f64 = delay.trim_end_matches('s').parse().ok()?;
    Some(Duration::from_secs_f64(secs + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_retry_delay() {
        let body = r#"{"error":{"code":429,"details":[{"@type":"x.Help"},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"10s"}]}}"#;
        assert_eq!(retry_delay(body), Some(Duration::from_secs(11)));
        assert_eq!(retry_delay("not json"), None);
    }

    #[test]
    fn parses_sse_stream() {
        let stream = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\r\n\r\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"thinking\",\"thought\":true},{\"text\":\"lo\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"!\"}]},\"finishReason\":\"STOP\"}]}\n",
        );
        let mut out = String::new();
        read_sse(stream.as_bytes(), |e| {
            out += &response_text(&e)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(out, "Hello!");
    }

    #[test]
    fn stream_stopping_early_is_an_error() {
        let stream = "data: {\"candidates\":[{\"finishReason\":\"MAX_TOKENS\"}]}\n\n";
        assert!(read_sse(stream.as_bytes(), |e| response_text(&e).map(drop)).is_err());
    }
}

impl Ai for Google {
    fn complete(&self, system: &str, prompt: &str) -> Result<String> {
        let resp = self.call(
            &self.model,
            "generateContent",
            &generate_body(system, prompt),
        )?;
        let text = response_text(&resp)?;
        if text.is_empty() {
            bail!("empty response from {}", self.model);
        }
        Ok(text)
    }

    fn complete_stream(
        &self,
        system: &str,
        prompt: &str,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let body = generate_body(system, prompt);
        let mut resp = self.send(&self.model, "streamGenerateContent?alt=sse", &body)?;
        let mut full = String::new();
        let reader = BufReader::new(resp.body_mut().as_reader());
        read_sse(reader, |event| {
            let chunk = response_text(&event)?;
            if !chunk.is_empty() {
                on_chunk(&chunk);
                full.push_str(&chunk);
            }
            Ok(())
        })?;
        if full.is_empty() {
            bail!("empty response from {}", self.model);
        }
        Ok(full)
    }

    fn chat(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Message> {
        let mut body = json!({ "contents": contents(messages) });
        if !system.is_empty() {
            body["systemInstruction"] = json!({ "parts": [{ "text": system }] });
        }
        if !tools.is_empty() {
            let decls: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    })
                })
                .collect();
            body["tools"] = json!([{ "functionDeclarations": decls }]);
        }

        let mut resp = self.send(&self.model, "streamGenerateContent?alt=sse", &body)?;
        let mut parts: Vec<Value> = Vec::new(); // every part, kept for replay
        let mut text = String::new();
        let mut calls = Vec::new();
        let reader = BufReader::new(resp.body_mut().as_reader());
        read_sse(reader, |event| {
            if let Some(reason) = event["promptFeedback"]["blockReason"].as_str() {
                bail!("prompt blocked: {reason}");
            }
            let candidate = &event["candidates"][0];
            for part in candidate["content"]["parts"]
                .as_array()
                .into_iter()
                .flatten()
            {
                if let Some(call) = part.get("functionCall") {
                    calls.push(ToolCall {
                        id: call["id"].as_str().map(String::from),
                        name: call["name"].as_str().unwrap_or_default().to_string(),
                        args: call["args"].clone(),
                    });
                } else if part["thought"] != true {
                    if let Some(t) = part["text"].as_str() {
                        on_text(t);
                        text.push_str(t);
                    }
                }
                parts.push(part.clone());
            }
            match candidate["finishReason"].as_str() {
                None | Some("STOP") | Some("FINISH_REASON_UNSPECIFIED") => Ok(()),
                Some(reason) => bail!("generation stopped early: {reason}"),
            }
        })?;
        if parts.is_empty() {
            bail!("empty response from {}", self.model);
        }
        Ok(Message::Assistant {
            text,
            calls,
            raw: Some(json!({ "role": "model", "parts": parts })),
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let body = json!({
            "content": { "parts": [{ "text": text }] },
            "outputDimensionality": EMBED_DIM,
        });
        let mut resp = self.call(&self.embed_model, "embedContent", &body)?;
        serde_json::from_value(resp["embedding"]["values"].take())
            .context("no embedding values in response")
    }

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let model = format!("models/{}", self.embed_model);
        let mut out = Vec::with_capacity(texts.len());
        for (i, chunk) in texts.chunks(EMBED_BATCH).enumerate() {
            if i > 0 {
                eprintln!(
                    "embedded {}/{}; waiting {}s for the per-minute quota",
                    out.len(),
                    texts.len(),
                    EMBED_WINDOW.as_secs()
                );
                thread::sleep(EMBED_WINDOW);
            }
            let requests: Vec<Value> = chunk
                .iter()
                .map(|t| {
                    json!({
                        "model": model,
                        "content": { "parts": [{ "text": t }] },
                        "outputDimensionality": EMBED_DIM,
                    })
                })
                .collect();
            let body = json!({ "requests": requests });
            let resp = self.call(&self.embed_model, "batchEmbedContents", &body)?;
            let embeddings = resp["embeddings"]
                .as_array()
                .context("no embeddings in response")?;
            for e in embeddings {
                out.push(serde_json::from_value(e["values"].clone()).context("bad embedding")?);
            }
        }
        Ok(out)
    }
}
