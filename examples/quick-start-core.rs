//! # ReasonKit Core - Quick Start Example
//!
//! This example demonstrates basic usage of ReasonKit Core's ThinkTool protocol system.
//!
//! Run with: `cargo run --example quick-start-core`

use reasonkit::thinktool::protocol::StepOutputFormat;
use reasonkit::thinktool::{Protocol, ProtocolStep, ReasoningStrategy, StepAction};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ReasonKit Core - Quick Start                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create a simple protocol for problem-solving using the builder pattern
    let protocol =
        Protocol::new("problem-solver", "Problem Solver")
            .with_strategy(ReasoningStrategy::Analytical)
            .with_step(ProtocolStep {
                id: "understand".to_string(),
                action: StepAction::Analyze { criteria: vec![] },
                prompt_template: "What is the core problem we need to solve? Problem: {{problem}}"
                    .to_string(),
                output_format: StepOutputFormat::Text,
                min_confidence: 0.7,
                depends_on: vec![],
                branch: None,
            })
            .with_step(ProtocolStep {
                id: "analyze".to_string(),
                action: StepAction::Generate {
                    min_count: 3,
                    max_count: 10,
                },
                prompt_template:
                    "What are the possible solutions? Problem understanding: {{understand}}"
                        .to_string(),
                output_format: StepOutputFormat::List,
                min_confidence: 0.7,
                depends_on: vec!["understand".to_string()],
                branch: None,
            })
            .with_step(ProtocolStep {
                id: "decide".to_string(),
                action: StepAction::Decide {
                    method: reasonkit::thinktool::protocol::DecisionMethod::ProsCons,
                },
                prompt_template: "Which solution is best and why? Options: {{analyze}}".to_string(),
                output_format: StepOutputFormat::Text,
                min_confidence: 0.8,
                depends_on: vec!["analyze".to_string()],
                branch: None,
            });

    println!("📋 Protocol: {} ({})", protocol.name, protocol.id);
    println!("   Version: {}", protocol.version);
    println!("   Strategy: {:?}", protocol.strategy);
    println!("   Steps: {}", protocol.steps.len());
    println!();

    // Display protocol structure
    for (i, step) in protocol.steps.iter().enumerate() {
        println!("   {}. Step: {}", i + 1, step.id);
        println!("      Action: {:?}", step.action);
        let prompt_preview = step.prompt_template.chars().take(50).collect::<String>();
        println!("      Prompt: {}...", prompt_preview);
        println!("      Output Format: {:?}", step.output_format);
        if !step.depends_on.is_empty() {
            println!("      Depends on: {:?}", step.depends_on);
        }
    }

    println!();
    println!("✅ Protocol created successfully!");
    println!();
    println!("💡 Next steps:");
    println!("   - Use the ProtocolExecutor to run this protocol");
    println!("   - Configure LLM clients for execution");
    println!("   - Add more complex reasoning steps");
    println!();

    Ok(())
}
