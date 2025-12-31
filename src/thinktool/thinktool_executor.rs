use crate::error::{Error, Result};
use crate::evaluation::Profile;

/// Legacy `ThinkToolExecutor` API used by earlier examples.
///
/// This is a thin compatibility wrapper around the newer `ProtocolExecutor` APIs.
#[derive(Debug, Default)]
pub struct ThinkToolExecutor;

impl ThinkToolExecutor {
    pub fn new() -> Self {
        Self
    }

    /// Executes a prompt using the given reasoning profile.
    ///
    /// Current implementation is a compile-first compatibility shim. It returns a
    /// structured string that callers can validate and pass into other pipelines.
    pub async fn run(&self, prompt: &str, profile: Profile) -> Result<String> {
        if prompt.trim().is_empty() {
            return Err(Error::Validation("Prompt is empty".to_string()));
        }

        let mut protocol = String::new();
        protocol.push_str("Protocol: Generated Reasoning Protocol\n");
        protocol.push_str(&format!("Profile: {:?}\n\n", profile));
        protocol.push_str("Prompt:\n");
        protocol.push_str(prompt);
        protocol.push('\n');

        Ok(protocol)
    }
}
