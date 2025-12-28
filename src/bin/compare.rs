//! A/B Comparison Tool for ReasonKit
//!
//! Compare raw LLM responses vs ThinkTool-enhanced responses side-by-side.
//! Let users SEE the difference (or lack thereof).
//!
//! Usage:
//!   rk-core compare "Should we use microservices?"
//!   rk-core compare "What causes inflation?" --profile balanced
//!   rk-core compare "Solve: 2x + 5 = 15" --profile scientific

use clap::Parser;
use std::time::Instant;

/// A/B Comparison: Raw vs ThinkTool-Enhanced
#[derive(Parser)]
#[command(name = "compare")]
#[command(about = "Compare raw LLM response vs ThinkTool-enhanced response")]
struct Args {
    /// The question or query to analyze
    query: String,

    /// ThinkTool profile to use (quick, balanced, deep, paranoid, powercombo)
    #[arg(short, long, default_value = "balanced")]
    profile: String,

    /// Output format (text, json, markdown)
    #[arg(short, long, default_value = "text")]
    format: String,

    /// Use mock LLM (for testing without API key)
    #[arg(long)]
    mock: bool,
}

fn main() {
    let args = Args::parse();

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    ReasonKit A/B Comparison");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("Query: \"{}\"", args.query);
    println!("Profile: {}", args.profile);
    println!();

    if args.mock {
        run_mock_comparison(&args);
    } else {
        println!("Error: Live comparison requires ANTHROPIC_API_KEY");
        println!("Use --mock for demonstration");
        std::process::exit(1);
    }
}

fn run_mock_comparison(args: &Args) {
    println!("───────────────────────────────────────────────────────────────────────");
    println!("                         [A] RAW PROMPT");
    println!("───────────────────────────────────────────────────────────────────────");
    println!();

    let raw_start = Instant::now();

    // Mock raw response
    let raw_response = format!(
        "Based on my analysis, here are my thoughts on \"{}\":\n\n\
         This is a complex question that depends on many factors. \
         Generally speaking, the answer involves considering multiple \
         perspectives and trade-offs. Without more specific context, \
         I would recommend evaluating your specific situation and \
         requirements before making a decision.\n\n\
         Key considerations include scalability, maintainability, \
         team expertise, and long-term goals.",
        args.query
    );

    let raw_duration = raw_start.elapsed();

    println!("{}", raw_response);
    println!();
    println!("  ⏱️  Duration: {:?}", raw_duration);
    println!("  📊 Tokens: ~150 (estimated)");
    println!();

    println!("───────────────────────────────────────────────────────────────────────");
    println!(
        "                    [B] THINKTOOL ENHANCED ({})",
        args.profile
    );
    println!("───────────────────────────────────────────────────────────────────────");
    println!();

    let enhanced_start = Instant::now();

    // Mock enhanced response (structured)
    let enhanced_response = format!(
        "## Analysis: \"{}\"\n\n\
         ### 💡 Perspectives Explored (GigaThink)\n\
         1. **Technical**: Architecture complexity, deployment overhead\n\
         2. **Business**: Time-to-market, maintenance costs\n\
         3. **Team**: Learning curve, hiring implications\n\
         4. **Scale**: Current vs future requirements\n\
         5. **Risk**: Failure modes, rollback strategies\n\n\
         ### ⚡ Logical Analysis (LaserLogic)\n\
         - Premise: Microservices solve scaling problems\n\
         - Hidden assumption: You HAVE scaling problems\n\
         - Fallacy risk: Appeal to novelty (\"everyone uses microservices\")\n\n\
         ### 🪨 First Principles (BedRock)\n\
         - Core need: Serve users reliably\n\
         - Monolith CAN scale (see: Stack Overflow, Shopify)\n\
         - Microservices add operational complexity\n\n\
         ### 🛡️ Evidence Check (ProofGuard)\n\
         - AWS: 70% of enterprises use hybrid approach\n\
         - Thoughtworks: Start monolith, extract when needed\n\
         - Martin Fowler: \"Monolith First\" pattern\n\n\
         ### 🔥 Honest Assessment (BrutalHonesty)\n\
         - If you're asking, you probably don't need microservices yet\n\
         - Microservices solve organizational problems, not technical ones\n\
         - Premature distribution is the root of much suffering\n\n\
         ### Recommendation\n\
         **Start with a modular monolith.** Extract services only when:\n\
         - Team size exceeds 8-10 per service boundary\n\
         - Scale requirements are PROVEN, not projected\n\
         - You have DevOps maturity for distributed systems\n\n\
         **Confidence: 85%**",
        args.query
    );

    let enhanced_duration = enhanced_start.elapsed();

    println!("{}", enhanced_response);
    println!();
    println!("  ⏱️  Duration: {:?}", enhanced_duration);
    println!("  📊 Tokens: ~800 (estimated)");
    println!("  💰 Cost: ~5x raw");
    println!();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                         COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("  | Metric          | Raw      | Enhanced  | Delta      |");
    println!("  |-----------------|----------|-----------|------------|");
    println!("  | Structure       | Low      | High      | +5 sections|");
    println!("  | Perspectives    | 1        | 5+        | +4         |");
    println!("  | Evidence cited  | 0        | 3         | +3         |");
    println!("  | Actionable      | Vague    | Specific  | ✓          |");
    println!("  | Self-critique   | None     | Present   | ✓          |");
    println!("  | Token cost      | ~150     | ~800      | 5.3x       |");
    println!();
    println!("  📋 YOUR JUDGMENT: Which response is more useful?");
    println!();
    println!("───────────────────────────────────────────────────────────────────────");
    println!();
    println!("  The ThinkTool process:");
    println!("    1. 💡 Divergent thinking (multiple perspectives)");
    println!("    2. ⚡ Convergent analysis (logical validation)");
    println!("    3. 🪨 Grounding (first principles)");
    println!("    4. 🛡️ Validation (evidence check)");
    println!("    5. 🔥 Ruthless cutting (honest assessment)");
    println!();
}
