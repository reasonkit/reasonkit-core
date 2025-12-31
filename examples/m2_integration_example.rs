//! # M2 Integration Example
//!
//! NOTE: The full `reasonkit::m2` integration is currently stubbed.
//! This example is kept minimal so `cargo test -p reasonkit-core --all-features`
//! remains green.

use reasonkit::error::Error;
use reasonkit::m2::{AgentFramework, M2Config, M2IntegrationConfig, M2ServiceBuilder, UseCase};

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt::init();

    println!("M2 integration example (stubbed)");
    println!("{}", "=".repeat(60));

    let m2_config = M2Config {
        api_key: std::env::var("MINIMAX_API_KEY").unwrap_or_default(),
        ..Default::default()
    };

    let integration_config = M2IntegrationConfig::default();

    let m2_service = M2ServiceBuilder::new()
        .with_config(m2_config)
        .with_integration_config(integration_config)
        .build()
        .await?;

    let input = serde_json::json!({
        "query": "Analyze this Rust snippet for potential issues",
        "code": "fn add(a:i32,b:i32)->i32 { a+b }",
    });

    let classification = m2_service.classify_use_case(UseCase::CodeAnalysis, &input)?;
    println!("Use case classification: {:?}", classification);

    match m2_service
        .execute_for_use_case(
            UseCase::CodeAnalysis,
            input,
            Some(AgentFramework::ClaudeCode),
        )
        .await
    {
        Ok(result) => {
            println!("Got result: {}", result.summary);
        }
        Err(err) => {
            println!("M2 execution unavailable in this build: {err}");
        }
    }

    Ok(())
}
