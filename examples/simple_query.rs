//! # Simple Query Example - Basic ThinkTool Usage
//!
//! This example demonstrates the fundamental usage patterns for ReasonKit's
//! ThinkTool system: structured reasoning protocols that turn prompts into
//! reproducible, auditable thinking chains.
//!
//! ## What You'll Learn
//!
//! - Creating a ProtocolExecutor
//! - Running individual ThinkTools (GigaThink, LaserLogic, etc.)
//! - Understanding ProtocolInput and ProtocolOutput
//! - Extracting results and confidence scores
//!
//! ## Running This Example
//!
//! ```bash
//! # With mock LLM (no API key required)
//! cargo run --example simple_query
//!
//! # With real LLM (requires ANTHROPIC_API_KEY or similar)
//! ANTHROPIC_API_KEY=your-key cargo run --example simple_query -- --real
//! ```
//!
//! ## ThinkTools Overview
//!
//! | Tool         | Code | Purpose                              |
//! |--------------|------|--------------------------------------|
//! | GigaThink    | gt   | Multi-perspective expansion (10+)    |
//! | LaserLogic   | ll   | Deductive reasoning, fallacy detect  |
//! | BedRock      | br   | First principles decomposition       |
//! | ProofGuard   | pg   | Multi-source verification            |
//! | BrutalHonesty| bh   | Adversarial self-critique            |

use reasonkit::thinktool::{ExecutorConfig, ProtocolExecutor, ProtocolInput, ProtocolOutput};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging (optional but helpful for debugging)
    tracing_subscriber::fmt()
        .with_env_filter("reasonkit=info")
        .init();

    println!("======================================================");
    println!("  ReasonKit - Simple Query Example");
    println!("  Basic ThinkTool Usage Patterns");
    println!("======================================================\n");

    // Check for --real flag to use actual LLM instead of mock
    let use_mock = !env::args().any(|arg| arg == "--real");

    if use_mock {
        println!("Running with MOCK LLM (no API key required)");
        println!("Use --real flag to run with actual LLM\n");
    } else {
        println!("Running with REAL LLM (requires API key)\n");
    }

    // =========================================================================
    // STEP 1: Create the Protocol Executor
    // =========================================================================

    println!("Step 1: Creating ProtocolExecutor...\n");

    let config = if use_mock {
        ExecutorConfig::mock()
    } else {
        ExecutorConfig::default()
    };

    let executor = ProtocolExecutor::with_config(config)?;

    // List available protocols
    println!("Available ThinkTools:");
    for protocol_id in executor.list_protocols() {
        if let Some(protocol) = executor.get_protocol(protocol_id) {
            println!(
                "  - {} (v{}): {}",
                protocol_id, protocol.version, protocol.description
            );
        }
    }
    println!();

    // =========================================================================
    // STEP 2: Execute GigaThink - Multi-Perspective Analysis
    // =========================================================================

    println!("------------------------------------------------------");
    println!("Step 2: Running GigaThink (Multi-Perspective Analysis)");
    println!("------------------------------------------------------\n");

    // GigaThink generates 10+ diverse perspectives on a topic
    let gigathink_input =
        ProtocolInput::query("What are the key factors that determine startup success in 2025?");

    println!("Query: {}\n", gigathink_input.get_str("query").unwrap());

    let gigathink_result = executor.execute("gigathink", gigathink_input).await?;

    print_result("GigaThink", &gigathink_result);

    // Extract perspectives from GigaThink output
    let perspectives = gigathink_result.perspectives();
    if !perspectives.is_empty() {
        println!("\nGenerated Perspectives ({} total):", perspectives.len());
        for (i, perspective) in perspectives.iter().take(5).enumerate() {
            println!("  {}. {}", i + 1, perspective);
        }
        if perspectives.len() > 5 {
            println!("  ... and {} more", perspectives.len() - 5);
        }
    }
    println!();

    // =========================================================================
    // STEP 3: Execute LaserLogic - Deductive Reasoning
    // =========================================================================

    println!("------------------------------------------------------");
    println!("Step 3: Running LaserLogic (Deductive Reasoning)");
    println!("------------------------------------------------------\n");

    // LaserLogic analyzes logical arguments and detects fallacies
    let laserlogic_input = ProtocolInput::argument(
        "All successful startups have product-market fit. \
         Company X has rapid user growth. \
         Therefore, Company X has product-market fit.",
    );

    println!(
        "Argument: {}\n",
        laserlogic_input.get_str("argument").unwrap()
    );

    let laserlogic_result = executor.execute("laserlogic", laserlogic_input).await?;

    print_result("LaserLogic", &laserlogic_result);
    println!();

    // =========================================================================
    // STEP 4: Execute BedRock - First Principles Analysis
    // =========================================================================

    println!("------------------------------------------------------");
    println!("Step 4: Running BedRock (First Principles Analysis)");
    println!("------------------------------------------------------\n");

    // BedRock breaks down complex topics into foundational axioms
    let bedrock_input = ProtocolInput::statement(
        "Microservices architecture is the best approach for all modern applications.",
    );

    println!(
        "Statement: {}\n",
        bedrock_input.get_str("statement").unwrap()
    );

    let bedrock_result = executor.execute("bedrock", bedrock_input).await?;

    print_result("BedRock", &bedrock_result);
    println!();

    // =========================================================================
    // STEP 5: Execute ProofGuard - Multi-Source Verification
    // =========================================================================

    println!("------------------------------------------------------");
    println!("Step 5: Running ProofGuard (Multi-Source Verification)");
    println!("------------------------------------------------------\n");

    // ProofGuard verifies claims using multiple sources
    let proofguard_input = ProtocolInput::claim(
        "GPT-4 was released in March 2023 and improved on GPT-3.5 in reasoning tasks.",
    );

    println!("Claim: {}\n", proofguard_input.get_str("claim").unwrap());

    let proofguard_result = executor.execute("proofguard", proofguard_input).await?;

    print_result("ProofGuard", &proofguard_result);
    println!();

    // =========================================================================
    // STEP 6: Execute BrutalHonesty - Adversarial Self-Critique
    // =========================================================================

    println!("------------------------------------------------------");
    println!("Step 6: Running BrutalHonesty (Adversarial Critique)");
    println!("------------------------------------------------------\n");

    // BrutalHonesty provides ruthless critique to find weaknesses
    let brutalhonesty_input = ProtocolInput::work(
        "Our startup will succeed because we have the best team and \
         our product is 10x better than anything else on the market.",
    );

    println!("Work: {}\n", brutalhonesty_input.get_str("work").unwrap());

    let brutalhonesty_result = executor
        .execute("brutalhonesty", brutalhonesty_input)
        .await?;

    print_result("BrutalHonesty", &brutalhonesty_result);

    // Check verdict
    if let Some(verdict) = brutalhonesty_result.verdict() {
        println!("\nVerdict: {}", verdict);
    }
    println!();

    // =========================================================================
    // STEP 7: Using Custom Input Fields
    // =========================================================================

    println!("------------------------------------------------------");
    println!("Step 7: Custom Input Fields");
    println!("------------------------------------------------------\n");

    // You can add custom fields to your input
    let custom_input = ProtocolInput::query("Should we adopt Rust for our backend?")
        .with_field(
            "context",
            "We currently use Python and need better performance",
        )
        .with_field("constraints", "Team has limited Rust experience")
        .with_field("timeline", "6 months to migration");

    println!("Query with context:");
    println!("  Query: {}", custom_input.get_str("query").unwrap());
    println!("  Context: {}", custom_input.get_str("context").unwrap());
    println!(
        "  Constraints: {}",
        custom_input.get_str("constraints").unwrap()
    );
    println!("  Timeline: {}", custom_input.get_str("timeline").unwrap());
    println!();

    let custom_result = executor.execute("gigathink", custom_input).await?;
    print_result("Custom Query", &custom_result);
    println!();

    // =========================================================================
    // SUMMARY
    // =========================================================================

    println!("======================================================");
    println!("  Example Complete!");
    println!("======================================================\n");

    println!("Key Takeaways:");
    println!("  1. ProtocolExecutor orchestrates ThinkTool execution");
    println!("  2. ProtocolInput.query/argument/statement/claim/work for different tools");
    println!("  3. ProtocolOutput contains confidence, steps, and structured data");
    println!("  4. Each ThinkTool has a specific purpose and output format");
    println!();
    println!("Next Steps:");
    println!("  - Try powercombo.rs for chained reasoning");
    println!("  - See rag_pipeline.rs for RAG integration");
    println!("  - Check mcp_server.rs for MCP server setup");

    Ok(())
}

/// Helper function to print protocol output in a readable format
fn print_result(name: &str, output: &ProtocolOutput) {
    println!("{} Result:", name);
    println!("  Success: {}", output.success);
    println!("  Confidence: {:.1}%", output.confidence * 100.0);
    println!("  Duration: {}ms", output.duration_ms);
    println!(
        "  Token Usage: {} input, {} output",
        output.tokens.input_tokens, output.tokens.output_tokens
    );
    println!("  Steps Executed: {}", output.steps.len());

    // Show step details
    for step in &output.steps {
        println!(
            "    - {} (confidence: {:.1}%, {}ms)",
            step.step_id,
            step.confidence * 100.0,
            step.duration_ms
        );
    }

    // Show any errors
    if let Some(error) = &output.error {
        println!("  Error: {}", error);
    }

    // Show budget summary if available
    if let Some(budget) = &output.budget_summary {
        println!("  Budget: {} tokens used", budget.tokens_used);
    }
}
