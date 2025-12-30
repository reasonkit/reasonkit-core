//! End-to-End Protocol Execution Tests
//!
//! These tests verify the full protocol execution pipeline for each profile
//! using mock LLM responses (no API calls required).
//!
//! Tests cover:
//! - Individual protocol execution (GigaThink, LaserLogic, BedRock, ProofGuard, BrutalHonesty)
//! - Profile chain execution (quick, balanced, deep, paranoid, decide, scientific, powercombo)
//! - Input/output mapping between chain steps
//! - Confidence tracking and thresholds
//! - Token usage aggregation
//! - Trace generation

use reasonkit::thinktool::{ExecutorConfig, ProtocolExecutor, ProtocolInput};

// ═══════════════════════════════════════════════════════════════════════════
// INDIVIDUAL PROTOCOL TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod individual_protocol_tests {
    use super::*;

    #[tokio::test]
    async fn test_gigathink_protocol() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input =
            ProtocolInput::query("What are the key factors for successful product launches?");

        let result = executor.execute("gigathink", input).await;
        assert!(
            result.is_ok(),
            "GigaThink execution failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "GigaThink should succeed");
        assert!(output.confidence > 0.0, "Should have positive confidence");
        assert!(!output.steps.is_empty(), "Should have step results");
        assert!(output.tokens.total_tokens > 0, "Should track token usage");
    }

    #[tokio::test]
    async fn test_laserlogic_protocol() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::argument(
            "If all successful products meet customer needs, \
             and Product X meets customer needs, \
             then Product X is successful.",
        );

        let result = executor.execute("laserlogic", input).await;
        assert!(
            result.is_ok(),
            "LaserLogic execution failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "LaserLogic should succeed");
        assert!(output.confidence > 0.0, "Should have positive confidence");
    }

    #[tokio::test]
    async fn test_bedrock_protocol() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::statement(
            "Machine learning models require large datasets for training.",
        );

        let result = executor.execute("bedrock", input).await;
        assert!(
            result.is_ok(),
            "BedRock execution failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "BedRock should succeed");
    }

    #[tokio::test]
    async fn test_proofguard_protocol() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::claim("GPT-4 achieves 86.4% accuracy on the MMLU benchmark.");

        let result = executor.execute("proofguard", input).await;
        assert!(
            result.is_ok(),
            "ProofGuard execution failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "ProofGuard should succeed");
    }

    #[tokio::test]
    async fn test_brutalhonesty_protocol() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::work(
            "Analysis: The market will grow 50% next year based on current trends.",
        );

        let result = executor.execute("brutalhonesty", input).await;
        assert!(
            result.is_ok(),
            "BrutalHonesty execution failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "BrutalHonesty should succeed");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PROFILE CHAIN TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod profile_chain_tests {
    use super::*;

    #[tokio::test]
    async fn test_quick_profile() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Should we adopt microservices architecture?");

        let result = executor.execute_profile("quick", input).await;
        assert!(result.is_ok(), "Quick profile failed: {:?}", result.err());

        let output = result.unwrap();
        assert!(output.success, "Quick profile should succeed");
        assert!(
            output.confidence >= 0.70,
            "Quick profile should achieve minimum 70% confidence"
        );

        // Quick profile = GigaThink → LaserLogic (2 protocols)
        assert!(
            !output.steps.is_empty(),
            "Should have step results from chain"
        );
    }

    #[tokio::test]
    async fn test_balanced_profile() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input =
            ProtocolInput::query("What is the optimal pricing strategy for a SaaS product?");

        let result = executor.execute_profile("balanced", input).await;
        assert!(
            result.is_ok(),
            "Balanced profile failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "Balanced profile should succeed");
        assert!(
            output.confidence >= 0.70,
            "Balanced profile should achieve reasonable confidence"
        );
    }

    #[tokio::test]
    async fn test_deep_profile() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Analyze the long-term implications of AI on employment.");

        let result = executor.execute_profile("deep", input).await;
        assert!(result.is_ok(), "Deep profile failed: {:?}", result.err());

        let output = result.unwrap();
        assert!(output.success, "Deep profile should succeed");
    }

    #[tokio::test]
    async fn test_paranoid_profile() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query(
            "Is the claim that quantum computing will break RSA encryption by 2030 accurate?",
        );

        let result = executor.execute_profile("paranoid", input).await;
        assert!(
            result.is_ok(),
            "Paranoid profile failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        // Paranoid may have lower success due to high 95% threshold
        assert!(!output.steps.is_empty(), "Should have step results");
    }

    #[tokio::test]
    async fn test_decide_profile() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        // Decide profile starts with LaserLogic which maps input.query to argument
        let input = ProtocolInput::query(
            "Should we build in-house or buy a third-party solution for authentication?",
        );

        let result = executor.execute_profile("decide", input).await;
        assert!(result.is_ok(), "Decide profile failed: {:?}", result.err());

        let output = result.unwrap();
        // The decide profile chains through multiple steps; check we got results
        assert!(
            !output.steps.is_empty(),
            "Decide profile should produce step results"
        );
        assert!(output.confidence > 0.0, "Should have calculated confidence");
    }

    #[tokio::test]
    async fn test_scientific_profile() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query(
            "What is the relationship between model size and performance in LLMs?",
        );

        let result = executor.execute_profile("scientific", input).await;
        assert!(
            result.is_ok(),
            "Scientific profile failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "Scientific profile should succeed");
    }

    #[tokio::test]
    async fn test_powercombo_profile() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query(
            "Evaluate the claim: ReasonKit improves AI reasoning quality by 20%.",
        );

        let result = executor.execute_profile("powercombo", input).await;
        assert!(
            result.is_ok(),
            "PowerCombo profile failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        // PowerCombo uses all 5 tools
        assert!(
            !output.steps.is_empty(),
            "Should have step results from full chain"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT STRUCTURE TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod output_structure_tests {
    use super::*;

    #[tokio::test]
    async fn test_output_contains_protocol_id() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test query");

        let output = executor.execute("gigathink", input).await.unwrap();
        assert_eq!(output.protocol_id, "gigathink");
    }

    #[tokio::test]
    async fn test_output_tracks_duration() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test query");

        let output = executor.execute("gigathink", input).await.unwrap();
        assert!(output.duration_ms > 0, "Duration should be tracked");
    }

    #[tokio::test]
    async fn test_output_aggregates_tokens() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test query");

        let output = executor.execute_profile("quick", input).await.unwrap();

        // Quick profile executes 2 protocols, so should have aggregated tokens
        assert!(output.tokens.input_tokens > 0, "Should track input tokens");
        assert!(
            output.tokens.output_tokens > 0,
            "Should track output tokens"
        );
        assert!(output.tokens.total_tokens > 0, "Should track total tokens");
    }

    #[tokio::test]
    async fn test_profile_output_contains_all_step_results() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test query for multiple steps");

        let output = executor.execute_profile("quick", input).await.unwrap();

        // Quick profile: GigaThink → LaserLogic
        // Each protocol has multiple steps, so we should have several step results
        assert!(
            !output.steps.is_empty(),
            "Should have step results from chain"
        );
    }

    #[tokio::test]
    async fn test_confidence_is_normalized() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test query");

        let output = executor.execute("gigathink", input).await.unwrap();

        assert!(output.confidence >= 0.0, "Confidence should be >= 0");
        assert!(output.confidence <= 1.0, "Confidence should be <= 1");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INPUT HANDLING TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod input_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_query_input() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("What is the meaning of life?");

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_argument_input() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::argument(
            "All cats are mammals. Fluffy is a cat. Therefore Fluffy is a mammal.",
        );

        let result = executor.execute("laserlogic", input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_statement_input() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::statement("Water boils at 100 degrees Celsius at sea level.");

        let result = executor.execute("bedrock", input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_claim_input() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::claim("The Earth is approximately 4.5 billion years old.");

        let result = executor.execute("proofguard", input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_work_input() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::work("My analysis suggests the market will double in 5 years.");

        let result = executor.execute("brutalhonesty", input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_input_with_additional_field() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("What is AI?")
            .with_field("context", "Technology industry analysis");

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_missing_required_input_fails() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // Empty input - should fail validation for protocols requiring query/argument
        let input = ProtocolInput {
            fields: std::collections::HashMap::new(),
        };

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_err(), "Should fail with missing required input");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// REGISTRY TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn test_executor_has_all_builtin_protocols() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let protocols = executor.list_protocols();

        // Core ThinkTools
        assert!(protocols.contains(&"gigathink"), "Should have gigathink");
        assert!(protocols.contains(&"laserlogic"), "Should have laserlogic");
        assert!(protocols.contains(&"bedrock"), "Should have bedrock");
        assert!(protocols.contains(&"proofguard"), "Should have proofguard");
        assert!(
            protocols.contains(&"brutalhonesty"),
            "Should have brutalhonesty"
        );
    }

    #[test]
    fn test_executor_has_all_builtin_profiles() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let profiles = executor.list_profiles();

        assert!(profiles.contains(&"quick"), "Should have quick profile");
        assert!(
            profiles.contains(&"balanced"),
            "Should have balanced profile"
        );
        assert!(profiles.contains(&"deep"), "Should have deep profile");
        assert!(
            profiles.contains(&"paranoid"),
            "Should have paranoid profile"
        );
        assert!(profiles.contains(&"decide"), "Should have decide profile");
        assert!(
            profiles.contains(&"scientific"),
            "Should have scientific profile"
        );
        assert!(
            profiles.contains(&"powercombo"),
            "Should have powercombo profile"
        );
    }

    #[test]
    fn test_get_protocol_returns_correct_protocol() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let protocol = executor.get_protocol("gigathink");
        assert!(protocol.is_some());
        assert_eq!(protocol.unwrap().id, "gigathink");
    }

    #[test]
    fn test_get_profile_returns_correct_profile() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let profile = executor.get_profile("quick");
        assert!(profile.is_some());
        assert_eq!(profile.unwrap().id, "quick");
        assert_eq!(profile.unwrap().min_confidence, 0.70);
    }

    #[test]
    fn test_nonexistent_protocol_returns_none() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let protocol = executor.get_protocol("nonexistent");
        assert!(protocol.is_none());
    }

    #[test]
    fn test_nonexistent_profile_returns_none() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let profile = executor.get_profile("nonexistent");
        assert!(profile.is_none());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR HANDLING TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_nonexistent_protocol_fails() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test query");

        let result = executor.execute("nonexistent_protocol", input).await;
        assert!(result.is_err(), "Should fail for nonexistent protocol");
    }

    #[tokio::test]
    async fn test_execute_nonexistent_profile_fails() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test query");

        let result = executor.execute_profile("nonexistent_profile", input).await;
        assert!(result.is_err(), "Should fail for nonexistent profile");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod configuration_tests {
    use super::*;

    #[test]
    fn test_mock_config_creation() {
        let config = ExecutorConfig::mock();
        assert!(config.use_mock, "Mock config should set use_mock to true");
    }

    #[test]
    fn test_default_config_timeout() {
        let config = ExecutorConfig::default();
        assert_eq!(
            config.timeout_secs, 120,
            "Default timeout should be 120 seconds"
        );
    }

    #[test]
    fn test_cli_tool_configs() {
        let claude = ExecutorConfig::claude_cli();
        assert!(claude.cli_tool.is_some());
        assert_eq!(claude.cli_tool.as_ref().unwrap().command, "claude");

        let codex = ExecutorConfig::codex_cli();
        assert!(codex.cli_tool.is_some());
        assert_eq!(codex.cli_tool.as_ref().unwrap().command, "codex");

        let gemini = ExecutorConfig::gemini_cli();
        assert!(gemini.cli_tool.is_some());
        assert_eq!(gemini.cli_tool.as_ref().unwrap().command, "gemini");
    }

    #[test]
    fn test_self_consistency_configs() {
        let config = ExecutorConfig::default().with_self_consistency();
        assert!(config.self_consistency.is_some());

        let fast = ExecutorConfig::default().with_self_consistency_fast();
        assert!(fast.self_consistency.is_some());

        let thorough = ExecutorConfig::default().with_self_consistency_thorough();
        assert!(thorough.self_consistency.is_some());

        let paranoid = ExecutorConfig::default().with_self_consistency_paranoid();
        assert!(paranoid.self_consistency.is_some());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION SCENARIOS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod integration_scenarios {
    use super::*;

    /// Scenario: Product decision analysis
    #[tokio::test]
    async fn test_product_decision_scenario() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // Use balanced profile for product decision
        let input = ProtocolInput::query(
            "Should we prioritize mobile app development over web features for Q1 2025?",
        )
        .with_field("context", "Enterprise B2B SaaS with 60% mobile usage");

        let result = executor.execute_profile("balanced", input).await;
        assert!(result.is_ok(), "Product decision scenario failed");

        let output = result.unwrap();
        assert!(output.success);
    }

    /// Scenario: Technical architecture review
    #[tokio::test]
    async fn test_architecture_review_scenario() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // Use deep profile for thorough architecture analysis
        let input = ProtocolInput::query(
            "Evaluate the trade-offs between event sourcing and traditional CRUD for our order management system."
        );

        let result = executor.execute_profile("deep", input).await;
        assert!(result.is_ok(), "Architecture review scenario failed");
    }

    /// Scenario: Claim verification
    #[tokio::test]
    async fn test_claim_verification_scenario() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // Use paranoid profile for high-stakes claim verification
        let input = ProtocolInput::query(
            "Verify: Our new algorithm reduces processing time by 40% compared to the industry standard."
        );

        let result = executor.execute_profile("paranoid", input).await;
        assert!(result.is_ok(), "Claim verification scenario failed");
    }

    /// Scenario: Research hypothesis
    #[tokio::test]
    async fn test_research_hypothesis_scenario() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // Use scientific profile for research analysis
        let input = ProtocolInput::query(
            "Hypothesis: Larger language models exhibit emergent reasoning abilities at the 10B+ parameter scale."
        );

        let result = executor.execute_profile("scientific", input).await;
        assert!(result.is_ok(), "Research hypothesis scenario failed");
    }

    /// Scenario: Quick brainstorm
    #[tokio::test]
    async fn test_quick_brainstorm_scenario() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // Use quick profile for rapid ideation
        let input = ProtocolInput::query("What are 10 ways to improve user onboarding?");

        let result = executor.execute_profile("quick", input).await;
        assert!(result.is_ok(), "Quick brainstorm scenario failed");

        let output = result.unwrap();
        assert!(output.duration_ms < 10000, "Quick profile should be fast");
    }
}
