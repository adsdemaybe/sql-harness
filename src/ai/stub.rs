//! Placeholder backend so the pipeline compiles end to end. Swap for a real one.

use super::Ai;
use anyhow::{Result, bail};

pub struct Stub;

impl Ai for Stub {
    fn complete(&self, _system: &str, _prompt: &str) -> Result<String> {
        bail!("no AI backend wired up yet (see src/ai/stub.rs)")
    }

    fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        bail!("no AI backend wired up yet (see src/ai/stub.rs)")
    }
}
