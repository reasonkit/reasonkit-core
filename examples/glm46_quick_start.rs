//! # GLM-4.6 Quick Start Example
//!
//! This example demonstrates how to quickly get started with GLM-4.6 integration
//! in ReasonKit. It covers basic client setup and chat completion.
//!
//! ## Prerequisites
//!
//! 1. Set `GLM46_API_KEY` environment variable:
//!    ```bash
//!    export GLM46_API_KEY="your-api-key-here"
//!    ```
//!
//! 2. Run the example:
//!    ```bash
//!    cargo run --example glm46_quick_start --features glm46
//!    ```
//!
//! ## Features Demonstrated
//!
//! - Basic client initialization
//! - Chat completion with structured output
//! - Cost tracking
//! - Error handling

use reasonkit::glm46::types::{ChatMessage, ChatRequest, ResponseFormat};
use reasonkit::glm46::{GLM46Client, GLM46Config};
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🚀 GLM-4.6 Quick Start Example\n");

    // Example 1: Basic Client Setup
    println!("📋 Example 1: Basic Client Setup");
    example_basic_client().await?;

    // Example 2: Chat Completion
    println!("\n📋 Example 2: Chat Completion");
    example_chat_completion().await?;

    // Example 3: Structured Output
    println!("\n📋 Example 3: Structured Output");
    example_structured_output().await?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Example 1: Basic client setup and configuration
async fn example_basic_client() -> anyhow::Result<()> {
    let config = GLM46Config {
        api_key: std::env::var("GLM46_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        base_url: "https://openrouter.ai/api/v1".to_string(),
        model: "glm-4.6".to_string(),
        timeout: Duration::from_secs(30),
        context_budget: 198_000, // Full 198K context window
        cost_tracking: true,
        local_fallback: true,
    };

    let client = GLM46Client::new(config)?;
    println!("  ✅ Client created successfully");
    println!(
        "  📊 Context budget: {} tokens",
        client.config().context_budget
    );
    println!("  💰 Cost tracking: {}", client.config().cost_tracking);

    Ok(())
}

/// Example 2: Basic chat completion
async fn example_chat_completion() -> anyhow::Result<()> {
    let config = GLM46Config {
        api_key: std::env::var("GLM46_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        base_url: "https://openrouter.ai/api/v1".to_string(),
        model: "glm-4.6".to_string(),
        timeout: Duration::from_secs(30),
        context_budget: 198_000,
        cost_tracking: true,
        local_fallback: false,
    };

    let client = GLM46Client::new(config)?;

    let request = ChatRequest {
        messages: vec![
            ChatMessage::system("You are a helpful assistant specialized in agent coordination."),
            ChatMessage::user(
                "Explain the key principles of multi-agent coordination in 2-3 sentences.",
            ),
        ],
        temperature: 0.7,
        max_tokens: 200,
        response_format: None,
        tools: None,
        tool_choice: None,
        stop: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
    };

    // Note: This will fail without a valid API key, but demonstrates the API
    match client.chat_completion(request).await {
        Ok(response) => {
            println!("  ✅ Chat completion successful");
            println!("  📝 Response: {}", response.choices[0].message.content);
            println!(
                "  📊 Tokens used: {} input, {} output",
                response.usage.prompt_tokens, response.usage.completion_tokens
            );
        }
        Err(e) => {
            println!(
                "  ⚠️  Chat completion failed (expected without valid API key): {}",
                e
            );
            println!("  💡 Set GLM46_API_KEY environment variable for real API calls");
        }
    }

    Ok(())
}

/// Example 3: Structured output for agent coordination
async fn example_structured_output() -> anyhow::Result<()> {
    let config = GLM46Config {
        api_key: std::env::var("GLM46_API_KEY").unwrap_or_else(|_| "demo-key".to_string()),
        base_url: "https://openrouter.ai/api/v1".to_string(),
        model: "glm-4.6".to_string(),
        timeout: Duration::from_secs(30),
        context_budget: 198_000,
        cost_tracking: true,
        local_fallback: false,
    };

    let _client = GLM46Client::new(config)?;

    let _request = ChatRequest {
        messages: vec![ChatMessage::user(
            "Create a coordination plan for 3 agents working on a software project.",
        )],
        temperature: 0.7,
        max_tokens: 500,
        response_format: Some(ResponseFormat::JsonSchema {
            name: "coordination_plan".to_string(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_assignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent_id": {"type": "string"},
                                "tasks": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "estimated_completion": {"type": "number"}
                },
                "required": ["agent_assignments", "estimated_completion"]
            }),
        }),
        tools: None,
        tool_choice: None,
        stop: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
    };

    println!("  ✅ Structured output request prepared");
    println!("  📋 Response format: JSON Schema");
    println!("  💡 Use client.chat_completion(request) with valid API key to execute");

    Ok(())
}
