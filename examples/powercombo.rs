//! # PowerCombo Example - Chained Reasoning Profiles
//!
//! This example demonstrates ReasonKit's **reasoning profiles**: pre-configured
//! chains of ThinkTools that combine multiple analytical strategies for
//! comprehensive, multi-stage reasoning.
//!
//! ## What You'll Learn
//!
//! - Using reasoning profiles (quick, balanced, deep, paranoid)
//! - Executing chained protocols with execute_profile()
//! - Understanding profile composition and flow
//! - Self-Consistency voting for improved accuracy
//! - Parallel execution for performance
//!
//! ## Running This Example
//!
//! ```bash
//! # With mock LLM (no API key required)
//! cargo run --example powercombo
//!
//! # With real LLM (requires ANTHROPIC_API_KEY or similar)
//! ANTHROPIC_API_KEY=your-key cargo run --example powercombo -- --real
//! ```
//!
//! ## Profile Overview
//!
//! | Profile  | Tools Chain          | Confidence | Use Case           |
//! |----------|---------------------|------------|--------------------|
//! | quick    | GT -> LL            | 70%        | Fast exploration   |
//! | balanced | GT -> LL -> BR -> PG| 80%        | Standard analysis  |
//! | deep     | All 5 tools         | 85%        | Thorough research  |
//! | paranoid | All 5 + validation  | 95%        | High-stakes decide |

use reasonkit::thinktool::{
    ExecutorConfig, ProtocolExecutor, ProtocolInput, SelfConsistencyConfig, VotingMethod,
};
use std::env;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("reasonkit=info")
        .init();

    println!("================================================================");
    println!("  ReasonKit PowerCombo - Chained Reasoning Profiles");
    println!("  Multi-Stage Analysis for Complex Decisions");
    println!("================================================================\n");

    // Check for --real flag
    let use_mock = !env::args().any(|arg| arg == "--real");

    if use_mock {
        println!("Running with MOCK LLM (no API key required)");
        println!("Use --real flag to run with actual LLM\n");
    }

    // Create executor with appropriate config
    let config = if use_mock {
        ExecutorConfig::mock()
    } else {
        ExecutorConfig::default()
    };

    let executor = ProtocolExecutor::with_config(config)?;

    // List available profiles
    println!("Available Reasoning Profiles:");
    for profile_id in executor.list_profiles() {
        if let Some(profile) = executor.get_profile(profile_id) {
            let tool_chain: Vec<&str> = profile
                .chain
                .iter()
                .map(|s| s.protocol_id.as_str())
                .collect();
            println!(
                "  - {} (min confidence: {:.0}%): {} -> {}",
                profile_id,
                profile.min_confidence * 100.0,
                profile.description,
                tool_chain.join(" -> ")
            );
        }
    }
    println!();

    // =========================================================================
    // EXAMPLE 1: Quick Profile - Fast Exploration
    // =========================================================================

    println!("----------------------------------------------------------------");
    println!("Example 1: QUICK Profile (Fast Exploration)");
    println!("Chain: GigaThink -> LaserLogic");
    println!("----------------------------------------------------------------\n");

    let query = "Should we use Kubernetes for our startup's infrastructure?";
    println!("Query: {}\n", query);

    let start = Instant::now();
    let quick_result = executor
        .execute_profile("quick", ProtocolInput::query(query))
        .await?;
    let quick_duration = start.elapsed();

    println!("Quick Profile Results:");
    println!("  Success: {}", quick_result.success);
    println!("  Confidence: {:.1}%", quick_result.confidence * 100.0);
    println!("  Duration: {:.2}s", quick_duration.as_secs_f64());
    println!("  Steps: {}", quick_result.steps.len());
    println!("  Tokens: {} total", quick_result.tokens.total_tokens);
    println!();

    // =========================================================================
    // EXAMPLE 2: Balanced Profile - Standard Analysis
    // =========================================================================

    println!("----------------------------------------------------------------");
    println!("Example 2: BALANCED Profile (Standard Analysis)");
    println!("Chain: GigaThink -> LaserLogic -> BedRock -> ProofGuard");
    println!("----------------------------------------------------------------\n");

    let query = "What is the optimal team size for an early-stage AI startup?";
    println!("Query: {}\n", query);

    let start = Instant::now();
    let balanced_result = executor
        .execute_profile("balanced", ProtocolInput::query(query))
        .await?;
    let balanced_duration = start.elapsed();

    println!("Balanced Profile Results:");
    println!("  Success: {}", balanced_result.success);
    println!("  Confidence: {:.1}%", balanced_result.confidence * 100.0);
    println!("  Duration: {:.2}s", balanced_duration.as_secs_f64());
    println!("  Steps: {}", balanced_result.steps.len());

    // Show step breakdown
    println!("\n  Step Breakdown:");
    for step in &balanced_result.steps {
        println!(
            "    {} - {:.1}% confidence, {}ms",
            step.step_id,
            step.confidence * 100.0,
            step.duration_ms
        );
    }
    println!();

    // =========================================================================
    // EXAMPLE 3: Deep Profile - Thorough Research
    // =========================================================================

    println!("----------------------------------------------------------------");
    println!("Example 3: DEEP Profile (Thorough Research)");
    println!("Chain: All 5 ThinkTools");
    println!("----------------------------------------------------------------\n");

    let query = "Should our startup pivot from B2C to B2B? We have 10K free users \
                 but low conversion rates. Our enterprise trial has 3 paying customers.";
    println!("Query: {}\n", query);

    let start = Instant::now();
    let deep_result = executor
        .execute_profile("deep", ProtocolInput::query(query))
        .await?;
    let deep_duration = start.elapsed();

    println!("Deep Profile Results:");
    println!("  Success: {}", deep_result.success);
    println!("  Confidence: {:.1}%", deep_result.confidence * 100.0);
    println!("  Duration: {:.2}s", deep_duration.as_secs_f64());
    println!("  Steps: {}", deep_result.steps.len());
    println!("  Token Cost: ${:.4}", deep_result.tokens.cost_usd);

    // Show aggregated insights
    if let Some(conclusion) = deep_result.data.get("conclusion") {
        println!("\n  Synthesized Conclusion:");
        println!("    {}", conclusion);
    }
    println!();

    // =========================================================================
    // EXAMPLE 4: Paranoid Profile - High-Stakes Decisions
    // =========================================================================

    println!("----------------------------------------------------------------");
    println!("Example 4: PARANOID Profile (High-Stakes Decisions)");
    println!("Chain: All 5 ThinkTools + Extra Validation");
    println!("----------------------------------------------------------------\n");

    let query = "Should we accept a $2M acquisition offer from BigTech Corp? \
                 We have 18 months runway, growing 20% MoM, but face competition \
                 from well-funded competitors.";
    println!("Query: {}\n", query);

    let start = Instant::now();
    let paranoid_result = executor
        .execute_profile("paranoid", ProtocolInput::query(query))
        .await?;
    let paranoid_duration = start.elapsed();

    println!("Paranoid Profile Results:");
    println!("  Success: {}", paranoid_result.success);
    println!("  Confidence: {:.1}%", paranoid_result.confidence * 100.0);
    println!("  Duration: {:.2}s", paranoid_duration.as_secs_f64());
    println!("  Steps: {}", paranoid_result.steps.len());
    println!("  Token Cost: ${:.4}", paranoid_result.tokens.cost_usd);

    // Show devil's advocate perspective
    if let Some(critique) = paranoid_result.data.get("critique") {
        println!("\n  Devil's Advocate Perspective:");
        println!("    {}", critique);
    }
    println!();

    // =========================================================================
    // EXAMPLE 5: Self-Consistency Voting
    // =========================================================================

    println!("----------------------------------------------------------------");
    println!("Example 5: Self-Consistency Voting (Improved Accuracy)");
    println!("Research: Wang et al. 2023 - +17.9% on GSM8K benchmark");
    println!("----------------------------------------------------------------\n");

    // Create executor with Self-Consistency enabled
    let sc_config = if use_mock {
        ExecutorConfig::mock()
    } else {
        ExecutorConfig::default().with_self_consistency_config(SelfConsistencyConfig {
            num_samples: 5,
            voting_method: VotingMethod::MajorityVote,
            temperature_base: 0.5,
            temperature_variance: 0.1,
            min_sample_confidence: 0.5,
            use_cisc: true,
            early_stopping: true,
            consensus_threshold: 0.8,
        })
    };

    let sc_executor = ProtocolExecutor::with_config(sc_config)?;

    let query = "Is it better to raise VC funding or bootstrap a SaaS startup in 2025?";
    println!("Query: {}\n", query);

    let start = Instant::now();

    // Use Self-Consistency configuration
    let sc_config = SelfConsistencyConfig::default();
    let (sc_result, consistency_info) = sc_executor
        .execute_with_self_consistency("balanced", ProtocolInput::query(query), &sc_config)
        .await?;
    let sc_duration = start.elapsed();

    println!("Self-Consistency Results:");
    println!("  Voted Answer: {}", consistency_info.answer);
    println!(
        "  Agreement Ratio: {:.1}%",
        consistency_info.agreement_ratio * 100.0
    );
    println!(
        "  Vote Count: {}/{}",
        consistency_info.vote_count, consistency_info.total_samples
    );
    println!("  Early Stopped: {}", consistency_info.early_stopped);
    println!("  Final Confidence: {:.1}%", sc_result.confidence * 100.0);
    println!("  Duration: {:.2}s", sc_duration.as_secs_f64());
    println!();

    // =========================================================================
    // EXAMPLE 6: Parallel Execution
    // =========================================================================

    println!("----------------------------------------------------------------");
    println!("Example 6: Parallel Execution (Performance Optimization)");
    println!("Performance: Up to (N-1)/N latency reduction");
    println!("----------------------------------------------------------------\n");

    // Create executor with parallel execution enabled
    let parallel_config = if use_mock {
        ExecutorConfig::mock().with_parallel_limit(4)
    } else {
        ExecutorConfig::default().with_parallel_limit(4)
    };

    let parallel_executor = ProtocolExecutor::with_config(parallel_config)?;

    let query = "What are the trade-offs between SQL and NoSQL for a real-time \
                 analytics platform processing 1M events/second?";
    println!("Query: {}\n", query);

    // Sequential execution
    let seq_start = Instant::now();
    let seq_result = executor
        .execute("gigathink", ProtocolInput::query(query))
        .await?;
    let seq_duration = seq_start.elapsed();

    // Parallel execution
    let par_start = Instant::now();
    let par_result = parallel_executor
        .execute("gigathink", ProtocolInput::query(query))
        .await?;
    let par_duration = par_start.elapsed();

    println!("Execution Comparison:");
    println!("  Sequential: {:.2}s", seq_duration.as_secs_f64());
    println!("  Parallel:   {:.2}s", par_duration.as_secs_f64());

    let speedup = seq_duration.as_secs_f64() / par_duration.as_secs_f64();
    if speedup > 1.0 {
        println!("  Speedup:    {:.2}x faster", speedup);
    }
    println!(
        "  Both Successful: {}",
        seq_result.success && par_result.success
    );
    println!();

    // =========================================================================
    // PROFILE COMPARISON SUMMARY
    // =========================================================================

    println!("================================================================");
    println!("  Profile Comparison Summary");
    println!("================================================================\n");

    println!(
        "| {:<10} | {:<10} | {:<12} | {:<12} |",
        "Profile", "Duration", "Confidence", "Tokens"
    );
    println!("|------------|------------|--------------|--------------|");
    println!(
        "| {:<10} | {:>8.2}s | {:>10.1}% | {:>12} |",
        "quick",
        quick_duration.as_secs_f64(),
        quick_result.confidence * 100.0,
        quick_result.tokens.total_tokens
    );
    println!(
        "| {:<10} | {:>8.2}s | {:>10.1}% | {:>12} |",
        "balanced",
        balanced_duration.as_secs_f64(),
        balanced_result.confidence * 100.0,
        balanced_result.tokens.total_tokens
    );
    println!(
        "| {:<10} | {:>8.2}s | {:>10.1}% | {:>12} |",
        "deep",
        deep_duration.as_secs_f64(),
        deep_result.confidence * 100.0,
        deep_result.tokens.total_tokens
    );
    println!(
        "| {:<10} | {:>8.2}s | {:>10.1}% | {:>12} |",
        "paranoid",
        paranoid_duration.as_secs_f64(),
        paranoid_result.confidence * 100.0,
        paranoid_result.tokens.total_tokens
    );
    println!();

    println!("Key Insights:");
    println!("  1. Quick profile is fastest but lower confidence");
    println!("  2. Paranoid profile is slowest but highest confidence");
    println!("  3. Self-Consistency improves accuracy through voting");
    println!("  4. Parallel execution reduces latency for independent steps");
    println!();
    println!("When to use each profile:");
    println!("  - quick:    Brainstorming, initial exploration");
    println!("  - balanced: Most business decisions, research");
    println!("  - deep:     Complex analysis, strategic planning");
    println!("  - paranoid: High-stakes decisions, compliance, legal");

    Ok(())
}
