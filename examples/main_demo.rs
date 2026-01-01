use reasonkit_core::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize the ReasonKit engine
    let engine = ReasonEngine::new()
        .with_profile(ReasoningProfile::Balanced)
        .build();

    // 2. Define a query
    let query =
        "Explain the relationship between quantum entanglement and information transfer speed.";

    // 3. Execute reasoning chain
    let analysis = engine.analyze(query).await?;

    // 4. Output results
    println!("Analysis: {}", analysis.summary);
    println!("Confidence: {:.2}", analysis.confidence_score);

    Ok(())
}
