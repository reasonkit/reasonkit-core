use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize a mock ThinkTool executor (no network calls)
    let engine = ProtocolExecutor::mock()?;

    // 2. Define a query
    let query =
        "Explain the relationship between quantum entanglement and information transfer speed.";

    // 3. Execute reasoning chain (illustrative: uses the built-in gigathink protocol)
    let output = engine
        .execute("gigathink", ProtocolInput::query(query))
        .await?;

    // 4. Output results
    println!("Success: {}", output.success);
    println!("Confidence: {:.2}", output.confidence);
    println!(
        "Summary: {}",
        output
            .data
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
    );

    Ok(())
}
