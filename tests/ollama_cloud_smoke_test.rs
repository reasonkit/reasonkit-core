//! Live smoke test for Ollama Cloud via local Ollama server.
//!
//! Gating:
//! - Set `RK_OLLAMA_RUN_LIVE_TESTS=1` to enable.
//! - Optional: `RK_OLLAMA_URL` (default: http://localhost:11434)
//! - Optional: `RK_OLLAMA_MODEL` (default: deepseek-v3.2:cloud)
//!
//! Example:
//! ```bash
//! RK_OLLAMA_RUN_LIVE_TESTS=1 \
//! RK_OLLAMA_URL=http://localhost:11434 \
//! RK_OLLAMA_MODEL=deepseek-v3.2:cloud \
//! cargo test -p reasonkit-core --test ollama_cloud_smoke_test
//! ```

use reasonkit::llm::ollama::types::{ChatMessage, ChatRequest};
use reasonkit::llm::ollama::OllamaClient;
use std::collections::BTreeMap;

fn live_tests_enabled() -> bool {
    std::env::var("RK_OLLAMA_RUN_LIVE_TESTS").ok().as_deref() == Some("1")
}

#[tokio::test]
async fn ollama_cloud_chat_2_plus_2() {
    if !live_tests_enabled() {
        eprintln!("skipping (set RK_OLLAMA_RUN_LIVE_TESTS=1 to enable)");
        return;
    }

    let url = std::env::var("RK_OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".into());
    let model = std::env::var("RK_OLLAMA_MODEL").unwrap_or_else(|_| "deepseek-v3.2:cloud".into());

    let client = OllamaClient::new(url).expect("create ollama client");

    // Keep options empty by default for predictability.
    let options = Some(BTreeMap::new());

    let req = ChatRequest {
        model,
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "What is 2+2? Reply with just the number.".into(),
        }],
        stream: Some(false),
        options,
    };

    let resp = client.chat(req).await.expect("ollama /api/chat");
    assert_eq!(resp.message.content.trim(), "4");
}
