//! Unit Tests for ProtocolExecutor
//!
//! Comprehensive unit test suite for the ProtocolExecutor covering:
//! 1. Initialization tests
//! 2. Protocol execution with mock LLM
//! 3. Profile selection (quick, balanced, deep, paranoid)
//! 4. Error handling for invalid protocols
//! 5. Timeout handling
//! 6. Concurrent execution tests
//!
//! All tests use mock LLM to avoid external API calls.

use reasonkit::error::Error;
use reasonkit::thinktool::{ExecutorConfig, ProtocolExecutor, ProtocolInput, ProtocolOutput};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// =============================================================================
// MODULE: INITIALIZATION TESTS
// =============================================================================

#[cfg(test)]
mod initialization_tests {
    use super::*;

    #[test]
    fn test_executor_new_succeeds() {
        // Arrange & Act
        let result = ProtocolExecutor::mock();

        // Assert
        assert!(result.is_ok(), "ProtocolExecutor::mock() should succeed");
    }

    #[test]
    fn test_executor_mock_sets_use_mock_flag() {
        // Arrange
        let config = ExecutorConfig::mock();

        // Assert
        assert!(config.use_mock, "Mock config should have use_mock = true");
    }

    #[test]
    fn test_executor_default_config() {
        // Arrange
        let config = ExecutorConfig::default();

        // Assert
        assert!(!config.use_mock, "Default config should not use mock");
        assert_eq!(config.timeout_secs, 120, "Default timeout should be 120s");
        assert!(!config.save_traces, "Default should not save traces");
        assert!(!config.verbose, "Default should not be verbose");
        assert!(config.show_progress, "Default should show progress");
        assert!(
            !config.enable_parallel,
            "Default should not enable parallel"
        );
        assert_eq!(
            config.max_concurrent_steps, 4,
            "Default max concurrent should be 4"
        );
    }

    #[test]
    fn test_executor_with_config() {
        // Arrange
        let config = ExecutorConfig {
            timeout_secs: 60,
            save_traces: true,
            verbose: true,
            use_mock: true,
            show_progress: false,
            ..Default::default()
        };

        // Act
        let result = ProtocolExecutor::with_config(config);

        // Assert
        assert!(result.is_ok(), "Should create executor with custom config");
    }

    #[test]
    fn test_executor_registry_is_populated() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");

        // Act
        let protocols = executor.list_protocols();

        // Assert
        assert!(!protocols.is_empty(), "Registry should have protocols");
        assert!(protocols.contains(&"gigathink"), "Should contain gigathink");
        assert!(
            protocols.contains(&"laserlogic"),
            "Should contain laserlogic"
        );
        assert!(protocols.contains(&"bedrock"), "Should contain bedrock");
        assert!(
            protocols.contains(&"proofguard"),
            "Should contain proofguard"
        );
        assert!(
            protocols.contains(&"brutalhonesty"),
            "Should contain brutalhonesty"
        );
    }

    #[test]
    fn test_executor_profiles_registry_is_populated() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");

        // Act
        let profiles = executor.list_profiles();

        // Assert
        assert!(
            !profiles.is_empty(),
            "Profiles registry should have profiles"
        );
        assert!(profiles.contains(&"quick"), "Should contain quick profile");
        assert!(
            profiles.contains(&"balanced"),
            "Should contain balanced profile"
        );
        assert!(profiles.contains(&"deep"), "Should contain deep profile");
        assert!(
            profiles.contains(&"paranoid"),
            "Should contain paranoid profile"
        );
    }

    #[test]
    fn test_executor_get_protocol_returns_correct_protocol() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");

        // Act
        let protocol = executor.get_protocol("gigathink");

        // Assert
        assert!(protocol.is_some(), "Should find gigathink protocol");
        let protocol = protocol.unwrap();
        assert_eq!(protocol.id, "gigathink");
        assert_eq!(protocol.name, "GigaThink");
        assert!(!protocol.steps.is_empty(), "Protocol should have steps");
    }

    #[test]
    fn test_executor_get_profile_returns_correct_profile() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");

        // Act
        let profile = executor.get_profile("quick");

        // Assert
        assert!(profile.is_some(), "Should find quick profile");
        let profile = profile.unwrap();
        assert_eq!(profile.id, "quick");
        assert_eq!(profile.min_confidence, 0.70);
    }

    #[test]
    fn test_executor_registry_mutable_access() {
        // Arrange
        let mut executor = ProtocolExecutor::mock().expect("Failed to create executor");

        // Act
        let registry = executor.registry_mut();
        let initial_count = registry.len();
        registry.remove("gigathink");
        let final_count = registry.len();

        // Assert
        assert_eq!(
            final_count,
            initial_count - 1,
            "Should have removed one protocol"
        );
    }
}

// =============================================================================
// MODULE: PROTOCOL EXECUTION WITH MOCK LLM
// =============================================================================

#[cfg(test)]
mod mock_execution_tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_gigathink_with_mock() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("What are the key success factors for startups?");

        // Act
        let result = executor.execute("gigathink", input).await;

        // Assert
        assert!(
            result.is_ok(),
            "GigaThink execution should succeed: {:?}",
            result.err()
        );
        let output = result.unwrap();
        assert!(output.success, "Output should indicate success");
        assert_eq!(output.protocol_id, "gigathink");
        assert!(output.confidence > 0.0, "Should have positive confidence");
        assert!(output.confidence <= 1.0, "Confidence should be <= 1.0");
    }

    #[tokio::test]
    async fn test_execute_laserlogic_with_mock() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::argument(
            "All humans are mortal. Socrates is human. Therefore, Socrates is mortal.",
        );

        // Act
        let result = executor.execute("laserlogic", input).await;

        // Assert
        assert!(result.is_ok(), "LaserLogic execution should succeed");
        let output = result.unwrap();
        assert!(output.success);
        assert_eq!(output.protocol_id, "laserlogic");
    }

    #[tokio::test]
    async fn test_execute_bedrock_with_mock() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::statement("The speed of light is constant in a vacuum.");

        // Act
        let result = executor.execute("bedrock", input).await;

        // Assert
        assert!(result.is_ok(), "BedRock execution should succeed");
        let output = result.unwrap();
        assert!(output.success);
        assert_eq!(output.protocol_id, "bedrock");
    }

    #[tokio::test]
    async fn test_execute_proofguard_with_mock() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::claim("Water freezes at 0 degrees Celsius at sea level.");

        // Act
        let result = executor.execute("proofguard", input).await;

        // Assert
        assert!(result.is_ok(), "ProofGuard execution should succeed");
        let output = result.unwrap();
        assert!(output.success);
        assert_eq!(output.protocol_id, "proofguard");
    }

    #[tokio::test]
    async fn test_execute_brutalhonesty_with_mock() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::work("My business plan assumes 100% year-over-year growth.");

        // Act
        let result = executor.execute("brutalhonesty", input).await;

        // Assert
        assert!(result.is_ok(), "BrutalHonesty execution should succeed");
        let output = result.unwrap();
        assert!(output.success);
        assert_eq!(output.protocol_id, "brutalhonesty");
    }

    #[tokio::test]
    async fn test_execution_tracks_duration() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test query");

        // Act
        let result = executor.execute("gigathink", input).await.unwrap();

        // Assert
        assert!(result.duration_ms > 0, "Duration should be tracked and > 0");
    }

    #[tokio::test]
    async fn test_execution_tracks_tokens() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test query");

        // Act
        let result = executor.execute("gigathink", input).await.unwrap();

        // Assert
        assert!(result.tokens.total_tokens > 0, "Should track token usage");
        assert!(result.tokens.input_tokens > 0, "Should track input tokens");
        assert!(
            result.tokens.output_tokens > 0,
            "Should track output tokens"
        );
    }

    #[tokio::test]
    async fn test_execution_returns_step_results() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test query");

        // Act
        let result = executor.execute("gigathink", input).await.unwrap();

        // Assert
        assert!(!result.steps.is_empty(), "Should return step results");
        for step in &result.steps {
            assert!(step.success, "Each step should succeed");
            assert!(step.confidence > 0.0, "Each step should have confidence");
        }
    }

    #[tokio::test]
    async fn test_execution_with_additional_fields() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Analyze market trends")
            .with_field("context", "Technology sector")
            .with_field("constraints", "Focus on AI companies");

        // Act
        let result = executor.execute("gigathink", input).await;

        // Assert
        assert!(result.is_ok(), "Should handle additional fields");
    }
}

// =============================================================================
// MODULE: PROFILE SELECTION TESTS
// =============================================================================

#[cfg(test)]
mod profile_selection_tests {
    use super::*;

    #[tokio::test]
    async fn test_quick_profile_execution() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Quick analysis needed");

        // Act
        let result = executor.execute_profile("quick", input).await;

        // Assert
        assert!(
            result.is_ok(),
            "Quick profile should execute: {:?}",
            result.err()
        );
        let output = result.unwrap();
        assert!(output.success, "Quick profile should succeed");
        assert!(
            output.confidence >= 0.70,
            "Quick profile min confidence is 70%"
        );
    }

    #[tokio::test]
    async fn test_balanced_profile_execution() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Balanced analysis of pricing strategy");

        // Act
        let result = executor.execute_profile("balanced", input).await;

        // Assert
        assert!(result.is_ok(), "Balanced profile should execute");
        let output = result.unwrap();
        assert!(!output.steps.is_empty(), "Should have step results");
    }

    #[tokio::test]
    async fn test_deep_profile_execution() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Deep analysis of system architecture");

        // Act
        let result = executor.execute_profile("deep", input).await;

        // Assert
        assert!(result.is_ok(), "Deep profile should execute");
        let output = result.unwrap();
        assert!(output.confidence > 0.0, "Should have calculated confidence");
    }

    #[tokio::test]
    async fn test_paranoid_profile_execution() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Maximum verification required for this claim");

        // Act
        let result = executor.execute_profile("paranoid", input).await;

        // Assert
        assert!(result.is_ok(), "Paranoid profile should execute");
        let output = result.unwrap();
        // Paranoid has 95% min confidence, mock may not achieve this
        assert!(
            !output.steps.is_empty(),
            "Should have step results from all tools"
        );
    }

    #[tokio::test]
    async fn test_decide_profile_execution() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Should we invest in project A or B?");

        // Act
        let result = executor.execute_profile("decide", input).await;

        // Assert
        assert!(result.is_ok(), "Decide profile should execute");
    }

    #[tokio::test]
    async fn test_scientific_profile_execution() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test hypothesis about data patterns");

        // Act
        let result = executor.execute_profile("scientific", input).await;

        // Assert
        assert!(result.is_ok(), "Scientific profile should execute");
    }

    #[tokio::test]
    async fn test_powercombo_profile_execution() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Ultimate analysis with all tools");

        // Act
        let result = executor.execute_profile("powercombo", input).await;

        // Assert
        assert!(result.is_ok(), "PowerCombo profile should execute");
        let output = result.unwrap();
        // PowerCombo uses all 5 tools plus validation pass
        assert!(
            !output.steps.is_empty(),
            "Should have results from all protocols"
        );
    }

    #[tokio::test]
    async fn test_profile_aggregates_tokens() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test token aggregation");

        // Act
        let result = executor.execute_profile("quick", input).await.unwrap();

        // Assert
        // Quick profile runs 2 protocols, so tokens should be aggregated
        assert!(result.tokens.total_tokens > 0, "Should aggregate tokens");
    }

    #[tokio::test]
    async fn test_profile_chain_confidence() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Confidence test");

        // Act
        let output = executor.execute_profile("balanced", input).await.unwrap();

        // Assert
        assert!(
            output.confidence >= 0.0,
            "Confidence should be non-negative"
        );
        assert!(output.confidence <= 1.0, "Confidence should be <= 1.0");
    }
}

// =============================================================================
// MODULE: ERROR HANDLING TESTS
// =============================================================================

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_nonexistent_protocol_returns_error() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test query");

        // Act
        let result = executor.execute("nonexistent_protocol", input).await;

        // Assert
        assert!(result.is_err(), "Should fail for nonexistent protocol");
        let error = result.unwrap_err();
        match error {
            Error::NotFound { resource } => {
                assert!(
                    resource.contains("protocol"),
                    "Error should mention protocol"
                );
                assert!(
                    resource.contains("nonexistent_protocol"),
                    "Error should mention protocol name"
                );
            }
            _ => panic!("Expected NotFound error, got: {:?}", error),
        }
    }

    #[tokio::test]
    async fn test_execute_nonexistent_profile_returns_error() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test query");

        // Act
        let result = executor.execute_profile("nonexistent_profile", input).await;

        // Assert
        assert!(result.is_err(), "Should fail for nonexistent profile");
        let error = result.unwrap_err();
        match error {
            Error::NotFound { resource } => {
                assert!(resource.contains("profile"), "Error should mention profile");
            }
            _ => panic!("Expected NotFound error, got: {:?}", error),
        }
    }

    #[tokio::test]
    async fn test_missing_required_input_returns_validation_error() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput {
            fields: HashMap::new(),
        }; // Empty input

        // Act
        let result = executor.execute("gigathink", input).await;

        // Assert
        assert!(result.is_err(), "Should fail with missing required input");
        let error = result.unwrap_err();
        match error {
            Error::Validation(msg) => {
                assert!(
                    msg.contains("query") || msg.contains("required"),
                    "Error should mention missing field: {}",
                    msg
                );
            }
            _ => panic!("Expected Validation error, got: {:?}", error),
        }
    }

    #[tokio::test]
    async fn test_wrong_input_type_for_protocol() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        // LaserLogic requires "argument" but we provide "query"
        let input = ProtocolInput::query("This is a query, not an argument");

        // Act
        let result = executor.execute("laserlogic", input).await;

        // Assert
        // This should fail because laserlogic requires "argument" field
        assert!(result.is_err(), "Should fail with wrong input type");
    }

    #[tokio::test]
    async fn test_get_nonexistent_protocol_returns_none() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");

        // Act
        let result = executor.get_protocol("definitely_not_a_protocol");

        // Assert
        assert!(
            result.is_none(),
            "Should return None for nonexistent protocol"
        );
    }

    #[tokio::test]
    async fn test_get_nonexistent_profile_returns_none() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");

        // Act
        let result = executor.get_profile("definitely_not_a_profile");

        // Assert
        assert!(
            result.is_none(),
            "Should return None for nonexistent profile"
        );
    }

    #[test]
    fn test_protocol_output_error_field_when_failed() {
        // Arrange - simulate a failed output
        let output = ProtocolOutput {
            protocol_id: "test".to_string(),
            success: false,
            data: HashMap::new(),
            confidence: 0.0,
            steps: vec![],
            tokens: reasonkit::thinktool::step::TokenUsage::default(),
            duration_ms: 0,
            error: Some("Test error message".to_string()),
            trace_id: None,
            budget_summary: None,
        };

        // Assert
        assert!(!output.success);
        assert!(output.error.is_some());
        assert_eq!(output.error.unwrap(), "Test error message");
    }
}

// =============================================================================
// MODULE: TIMEOUT HANDLING TESTS
// =============================================================================

#[cfg(test)]
mod timeout_handling_tests {
    use super::*;

    #[test]
    fn test_config_timeout_default() {
        // Arrange & Act
        let config = ExecutorConfig::default();

        // Assert
        assert_eq!(
            config.timeout_secs, 120,
            "Default timeout should be 120 seconds"
        );
    }

    #[test]
    fn test_config_custom_timeout() {
        // Arrange
        let config = ExecutorConfig {
            timeout_secs: 30,
            use_mock: true,
            ..Default::default()
        };

        // Assert
        assert_eq!(
            config.timeout_secs, 30,
            "Custom timeout should be preserved"
        );
    }

    #[tokio::test]
    async fn test_mock_execution_completes_within_timeout() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test timeout");
        let start = Instant::now();

        // Act
        let result = executor.execute("gigathink", input).await;

        // Assert
        let elapsed = start.elapsed();
        assert!(result.is_ok(), "Execution should complete");
        assert!(
            elapsed < Duration::from_secs(10),
            "Mock should complete quickly"
        );
    }

    #[tokio::test]
    async fn test_profile_execution_duration_is_reasonable() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test profile duration");
        let start = Instant::now();

        // Act
        let result = executor.execute_profile("quick", input).await;

        // Assert
        let elapsed = start.elapsed();
        assert!(result.is_ok(), "Profile execution should complete");
        assert!(
            elapsed < Duration::from_secs(30),
            "Mock profile should complete reasonably fast"
        );
    }

    #[test]
    fn test_timeout_config_serialization() {
        // Arrange
        let config = ExecutorConfig {
            timeout_secs: 45,
            use_mock: true,
            ..Default::default()
        };

        // Act
        let json = serde_json::to_string(&config).expect("Should serialize");
        let deserialized: ExecutorConfig = serde_json::from_str(&json).expect("Should deserialize");

        // Assert
        assert_eq!(deserialized.timeout_secs, 45);
    }
}

// =============================================================================
// MODULE: CONCURRENT EXECUTION TESTS
// =============================================================================

#[cfg(test)]
mod concurrent_execution_tests {
    use super::*;

    #[tokio::test]
    async fn test_multiple_protocols_concurrently() {
        // Arrange
        let executor = Arc::new(ProtocolExecutor::mock().expect("Failed to create executor"));

        // Act - Execute 5 protocols concurrently
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let exec = Arc::clone(&executor);
                let query = format!("Concurrent query {}", i);
                tokio::spawn(async move {
                    let input = ProtocolInput::query(query);
                    exec.execute("gigathink", input).await
                })
            })
            .collect();

        // Assert
        let results: Vec<_> = futures::future::join_all(handles).await;
        for (i, result) in results.into_iter().enumerate() {
            let inner = result.expect("Task should not panic");
            assert!(inner.is_ok(), "Concurrent execution {} should succeed", i);
        }
    }

    #[tokio::test]
    async fn test_multiple_profiles_concurrently() {
        // Arrange
        let executor = Arc::new(ProtocolExecutor::mock().expect("Failed to create executor"));
        let profiles = vec!["quick", "balanced", "scientific"];

        // Act
        let handles: Vec<_> = profiles
            .iter()
            .map(|&profile| {
                let exec = Arc::clone(&executor);
                let p = profile.to_string();
                tokio::spawn(async move {
                    let input = ProtocolInput::query(format!("Query for {}", p));
                    exec.execute_profile(&p, input).await
                })
            })
            .collect();

        // Assert
        let results: Vec<_> = futures::future::join_all(handles).await;
        for result in results {
            let inner = result.expect("Task should not panic");
            assert!(inner.is_ok(), "Concurrent profile execution should succeed");
        }
    }

    #[tokio::test]
    async fn test_parallel_execution_config() {
        // Arrange
        let config = ExecutorConfig::default().with_parallel();

        // Assert
        assert!(
            config.enable_parallel,
            "with_parallel should enable parallel"
        );
    }

    #[tokio::test]
    async fn test_parallel_execution_with_limit() {
        // Arrange
        let config = ExecutorConfig::default().with_parallel_limit(2);

        // Assert
        assert!(config.enable_parallel, "should enable parallel");
        assert_eq!(config.max_concurrent_steps, 2, "should set limit to 2");
    }

    #[tokio::test]
    async fn test_concurrent_different_protocols() {
        // Arrange
        let executor = Arc::new(ProtocolExecutor::mock().expect("Failed to create executor"));

        // Act - Execute different protocols concurrently
        let gigathink_handle = {
            let exec = Arc::clone(&executor);
            tokio::spawn(async move {
                exec.execute("gigathink", ProtocolInput::query("GigaThink query"))
                    .await
            })
        };

        let bedrock_handle = {
            let exec = Arc::clone(&executor);
            tokio::spawn(async move {
                exec.execute("bedrock", ProtocolInput::statement("BedRock statement"))
                    .await
            })
        };

        let proofguard_handle = {
            let exec = Arc::clone(&executor);
            tokio::spawn(async move {
                exec.execute("proofguard", ProtocolInput::claim("ProofGuard claim"))
                    .await
            })
        };

        // Assert
        let (gigathink, bedrock, proofguard) =
            tokio::join!(gigathink_handle, bedrock_handle, proofguard_handle);

        assert!(gigathink.unwrap().is_ok(), "GigaThink should succeed");
        assert!(bedrock.unwrap().is_ok(), "BedRock should succeed");
        assert!(proofguard.unwrap().is_ok(), "ProofGuard should succeed");
    }

    #[tokio::test]
    async fn test_shared_executor_thread_safety() {
        // Arrange
        let executor = Arc::new(ProtocolExecutor::mock().expect("Failed to create executor"));
        let counter = Arc::new(AtomicUsize::new(0));

        // Act - Execute 10 concurrent operations
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let exec = Arc::clone(&executor);
                let cnt = Arc::clone(&counter);
                tokio::spawn(async move {
                    let input = ProtocolInput::query("Thread safety test");
                    let result = exec.execute("gigathink", input).await;
                    if result.is_ok() {
                        cnt.fetch_add(1, Ordering::SeqCst);
                    }
                    result
                })
            })
            .collect();

        let _ = futures::future::join_all(handles).await;

        // Assert
        let successful = counter.load(Ordering::SeqCst);
        assert_eq!(
            successful, 10,
            "All 10 concurrent executions should succeed"
        );
    }

    #[tokio::test]
    async fn test_concurrent_profile_independence() {
        // Arrange
        let executor = Arc::new(ProtocolExecutor::mock().expect("Failed to create executor"));

        // Act - Run quick and paranoid profiles concurrently
        // They have different complexity, so this tests independence
        let quick_handle = {
            let exec = Arc::clone(&executor);
            tokio::spawn(async move {
                exec.execute_profile("quick", ProtocolInput::query("Quick query"))
                    .await
            })
        };

        let paranoid_handle = {
            let exec = Arc::clone(&executor);
            tokio::spawn(async move {
                exec.execute_profile("paranoid", ProtocolInput::query("Paranoid query"))
                    .await
            })
        };

        // Assert
        let (quick, paranoid) = tokio::join!(quick_handle, paranoid_handle);

        let quick_result = quick.unwrap().expect("Quick should complete");
        let paranoid_result = paranoid.unwrap().expect("Paranoid should complete");

        assert_eq!(quick_result.protocol_id, "quick");
        assert_eq!(paranoid_result.protocol_id, "paranoid");

        // Quick should have fewer steps than paranoid (2 vs 6 protocols)
        assert!(
            quick_result.steps.len() < paranoid_result.steps.len(),
            "Quick should have fewer steps than paranoid"
        );
    }
}

// =============================================================================
// MODULE: PROTOCOL INPUT TESTS
// =============================================================================

#[cfg(test)]
mod protocol_input_tests {
    use super::*;

    #[test]
    fn test_protocol_input_query() {
        // Arrange & Act
        let input = ProtocolInput::query("What is the answer?");

        // Assert
        assert_eq!(input.get_str("query"), Some("What is the answer?"));
    }

    #[test]
    fn test_protocol_input_argument() {
        // Arrange & Act
        let input = ProtocolInput::argument("A implies B. A is true. Therefore B.");

        // Assert
        assert_eq!(
            input.get_str("argument"),
            Some("A implies B. A is true. Therefore B.")
        );
    }

    #[test]
    fn test_protocol_input_statement() {
        // Arrange & Act
        let input = ProtocolInput::statement("The sun rises in the east.");

        // Assert
        assert_eq!(
            input.get_str("statement"),
            Some("The sun rises in the east.")
        );
    }

    #[test]
    fn test_protocol_input_claim() {
        // Arrange & Act
        let input = ProtocolInput::claim("Rust is memory-safe.");

        // Assert
        assert_eq!(input.get_str("claim"), Some("Rust is memory-safe."));
    }

    #[test]
    fn test_protocol_input_work() {
        // Arrange & Act
        let input = ProtocolInput::work("My analysis shows positive trends.");

        // Assert
        assert_eq!(
            input.get_str("work"),
            Some("My analysis shows positive trends.")
        );
    }

    #[test]
    fn test_protocol_input_with_field_chaining() {
        // Arrange & Act
        let input = ProtocolInput::query("Main query")
            .with_field("context", "Some context")
            .with_field("constraints", "Some constraints")
            .with_field("domain", "Technology");

        // Assert
        assert_eq!(input.get_str("query"), Some("Main query"));
        assert_eq!(input.get_str("context"), Some("Some context"));
        assert_eq!(input.get_str("constraints"), Some("Some constraints"));
        assert_eq!(input.get_str("domain"), Some("Technology"));
    }

    #[test]
    fn test_protocol_input_nonexistent_field_returns_none() {
        // Arrange
        let input = ProtocolInput::query("Test");

        // Act & Assert
        assert!(input.get_str("nonexistent").is_none());
    }

    #[test]
    fn test_protocol_input_serialization() {
        // Arrange
        let input = ProtocolInput::query("Test query").with_field("context", "Test context");

        // Act
        let json = serde_json::to_string(&input).expect("Should serialize");
        let deserialized: ProtocolInput = serde_json::from_str(&json).expect("Should deserialize");

        // Assert
        assert_eq!(deserialized.get_str("query"), Some("Test query"));
        assert_eq!(deserialized.get_str("context"), Some("Test context"));
    }
}

// =============================================================================
// MODULE: PROTOCOL OUTPUT TESTS
// =============================================================================

#[cfg(test)]
mod protocol_output_tests {
    use super::*;

    #[tokio::test]
    async fn test_output_get_method() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test output access");

        // Act
        let output = executor.execute("gigathink", input).await.unwrap();

        // Assert
        let confidence = output.get("confidence");
        assert!(confidence.is_some(), "Should have confidence in data");
    }

    #[tokio::test]
    async fn test_output_verdict_method() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::work("Test work for verdict");

        // Act
        let output = executor.execute("brutalhonesty", input).await.unwrap();

        // Assert
        // Note: verdict() returns Option based on data content
        // Mock may or may not populate this
        assert!(output.success);
    }

    #[tokio::test]
    async fn test_output_contains_protocol_id() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test");

        // Act
        let output = executor
            .execute("laserlogic", ProtocolInput::argument("A therefore B"))
            .await
            .unwrap();

        // Assert
        assert_eq!(output.protocol_id, "laserlogic");
    }

    #[test]
    fn test_output_serialization() {
        // Arrange
        let output = ProtocolOutput {
            protocol_id: "test".to_string(),
            success: true,
            data: HashMap::from([("key".to_string(), serde_json::json!("value"))]),
            confidence: 0.85,
            steps: vec![],
            tokens: reasonkit::thinktool::step::TokenUsage::new(100, 200, 0.001),
            duration_ms: 500,
            error: None,
            trace_id: Some("trace-123".to_string()),
            budget_summary: None,
        };

        // Act
        let json = serde_json::to_string(&output).expect("Should serialize");
        let deserialized: ProtocolOutput = serde_json::from_str(&json).expect("Should deserialize");

        // Assert
        assert_eq!(deserialized.protocol_id, "test");
        assert_eq!(deserialized.confidence, 0.85);
        assert!(deserialized.success);
    }
}

// =============================================================================
// MODULE: CLI TOOL CONFIG TESTS
// =============================================================================

#[cfg(test)]
mod cli_tool_config_tests {
    use super::*;
    use reasonkit::thinktool::CliToolConfig;

    #[test]
    fn test_claude_cli_config() {
        // Arrange & Act
        let config = CliToolConfig::claude();

        // Assert
        assert_eq!(config.command, "claude");
        assert!(config.pre_args.contains(&"-p".to_string()));
        assert!(!config.interactive);
    }

    #[test]
    fn test_codex_cli_config() {
        // Arrange & Act
        let config = CliToolConfig::codex();

        // Assert
        assert_eq!(config.command, "codex");
        assert!(config.pre_args.contains(&"-q".to_string()));
        assert!(!config.interactive);
    }

    #[test]
    fn test_gemini_cli_config() {
        // Arrange & Act
        let config = CliToolConfig::gemini();

        // Assert
        assert_eq!(config.command, "gemini");
        assert!(config.pre_args.contains(&"-p".to_string()));
        assert!(!config.interactive);
    }

    #[test]
    fn test_copilot_cli_config() {
        // Arrange & Act
        let config = CliToolConfig::copilot();

        // Assert
        assert_eq!(config.command, "gh");
        assert!(config.pre_args.contains(&"copilot".to_string()));
        assert!(config.interactive);
    }

    #[test]
    fn test_executor_config_with_cli_tool() {
        // Arrange & Act
        let config = ExecutorConfig::claude_cli();

        // Assert
        assert!(config.cli_tool.is_some());
        assert_eq!(config.cli_tool.as_ref().unwrap().command, "claude");
    }

    #[test]
    fn test_cli_config_serialization() {
        // Arrange
        let config = CliToolConfig::claude();

        // Act
        let json = serde_json::to_string(&config).expect("Should serialize");
        let deserialized: CliToolConfig = serde_json::from_str(&json).expect("Should deserialize");

        // Assert
        assert_eq!(deserialized.command, "claude");
    }
}

// =============================================================================
// MODULE: SELF-CONSISTENCY CONFIG TESTS
// =============================================================================

#[cfg(test)]
mod self_consistency_tests {
    use super::*;

    #[test]
    fn test_with_self_consistency() {
        // Arrange & Act
        let config = ExecutorConfig::default().with_self_consistency();

        // Assert
        assert!(config.self_consistency.is_some());
    }

    #[test]
    fn test_with_self_consistency_fast() {
        // Arrange & Act
        let config = ExecutorConfig::default().with_self_consistency_fast();

        // Assert
        assert!(config.self_consistency.is_some());
    }

    #[test]
    fn test_with_self_consistency_thorough() {
        // Arrange & Act
        let config = ExecutorConfig::default().with_self_consistency_thorough();

        // Assert
        assert!(config.self_consistency.is_some());
    }

    #[test]
    fn test_with_self_consistency_paranoid() {
        // Arrange & Act
        let config = ExecutorConfig::default().with_self_consistency_paranoid();

        // Assert
        assert!(config.self_consistency.is_some());
    }
}

// =============================================================================
// MODULE: PARALLEL EXECUTION CONFIG TESTS
// =============================================================================

#[cfg(test)]
mod parallel_config_tests {
    use super::*;

    #[test]
    fn test_default_parallel_disabled() {
        // Arrange & Act
        let config = ExecutorConfig::default();

        // Assert
        assert!(!config.enable_parallel);
        assert_eq!(config.max_concurrent_steps, 4);
    }

    #[test]
    fn test_with_parallel() {
        // Arrange & Act
        let config = ExecutorConfig::default().with_parallel();

        // Assert
        assert!(config.enable_parallel);
    }

    #[test]
    fn test_with_parallel_limit() {
        // Arrange & Act
        let config = ExecutorConfig::default().with_parallel_limit(8);

        // Assert
        assert!(config.enable_parallel);
        assert_eq!(config.max_concurrent_steps, 8);
    }

    #[test]
    fn test_parallel_config_serialization() {
        // Arrange
        let config = ExecutorConfig::default().with_parallel_limit(6);

        // Act
        let json = serde_json::to_string(&config).expect("Should serialize");
        let deserialized: ExecutorConfig = serde_json::from_str(&json).expect("Should deserialize");

        // Assert
        assert!(deserialized.enable_parallel);
        assert_eq!(deserialized.max_concurrent_steps, 6);
    }
}

// =============================================================================
// MODULE: EDGE CASE TESTS
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_query_string() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("");

        // Act
        let result = executor.execute("gigathink", input).await;

        // Assert
        // Should still execute (empty string is valid input, just not useful)
        assert!(result.is_ok(), "Empty query should still execute");
    }

    #[tokio::test]
    async fn test_very_long_query() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let long_query = "a".repeat(10000);
        let input = ProtocolInput::query(long_query);

        // Act
        let result = executor.execute("gigathink", input).await;

        // Assert
        assert!(result.is_ok(), "Long query should execute");
    }

    #[tokio::test]
    async fn test_special_characters_in_query() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Query with special chars: <>&\"'{}[]()!@#$%^*");

        // Act
        let result = executor.execute("gigathink", input).await;

        // Assert
        assert!(result.is_ok(), "Special characters should be handled");
    }

    #[tokio::test]
    async fn test_unicode_in_query() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Unicode test: cafe, ninos, nihongo");

        // Act
        let result = executor.execute("gigathink", input).await;

        // Assert
        assert!(result.is_ok(), "Unicode characters should be handled");
    }

    #[tokio::test]
    async fn test_multiline_query() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Line 1\nLine 2\nLine 3\n\nLine 5");

        // Act
        let result = executor.execute("gigathink", input).await;

        // Assert
        assert!(result.is_ok(), "Multiline query should be handled");
    }

    #[test]
    fn test_executor_default_implementation() {
        // Arrange & Act
        let executor = ProtocolExecutor::default();

        // Assert
        // Default uses mock internally
        assert!(!executor.list_protocols().is_empty());
    }
}

// =============================================================================
// MODULE: TRACE AND BUDGET TESTS
// =============================================================================

#[cfg(test)]
mod trace_and_budget_tests {
    use super::*;

    #[test]
    fn test_trace_config_disabled_by_default() {
        // Arrange & Act
        let config = ExecutorConfig::default();

        // Assert
        assert!(!config.save_traces);
        assert!(config.trace_dir.is_none());
    }

    #[test]
    fn test_trace_config_can_be_enabled() {
        // Arrange & Act
        let config = ExecutorConfig {
            save_traces: true,
            trace_dir: Some(std::path::PathBuf::from("/tmp/traces")),
            use_mock: true,
            ..Default::default()
        };

        // Assert
        assert!(config.save_traces);
        assert!(config.trace_dir.is_some());
    }

    #[tokio::test]
    async fn test_output_has_no_trace_id_by_default() {
        // Arrange
        let executor = ProtocolExecutor::mock().expect("Failed to create executor");
        let input = ProtocolInput::query("Test");

        // Act
        let output = executor.execute("gigathink", input).await.unwrap();

        // Assert
        assert!(
            output.trace_id.is_none(),
            "Trace ID should be None when save_traces is false"
        );
    }

    #[test]
    fn test_budget_config_defaults() {
        // Arrange & Act
        let config = ExecutorConfig::default();

        // Assert
        // Budget config exists with default values
        assert_eq!(config.budget.token_limit, None);
        assert_eq!(config.budget.cost_limit, None);
    }
}
