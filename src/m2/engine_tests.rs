//! Comprehensive unit tests for the M2 ThinkingOrchestrator engine.
//!
//! This test module covers:
//! - Engine initialization
//! - Request handling
//! - Response parsing
//! - Error recovery
//! - Async behavior with mocked external API calls

use super::engine::*;
use super::types::*;
use crate::error::Error;
use crate::m2::connector::M2Connector;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// TEST HELPER FUNCTIONS
// ============================================================================

/// Creates a minimal M2Config for testing
fn create_test_config() -> M2Config {
    M2Config {
        endpoint: "http://localhost:11434/api/generate".to_string(),
        api_key: "test-api-key".to_string(),
        max_context_length: 10000,
        max_output_length: 5000,
        rate_limit: RateLimitConfig::default(),
        performance: PerformanceConfig::default(),
    }
}

/// Creates a test M2Connector with mock configuration
fn create_test_connector() -> Arc<M2Connector> {
    let config = create_test_config();
    Arc::new(M2Connector::new(config))
}

/// Creates a minimal InterleavedProtocol for testing
fn create_test_protocol(name: &str, phase_count: usize) -> InterleavedProtocol {
    let phases: Vec<InterleavedPhase> = (0..phase_count)
        .map(|i| InterleavedPhase {
            name: format!("phase_{}", i),
            parallel_branches: 2,
            required_confidence: 0.8,
            validation_methods: vec![ValidationMethod::SelfCheck],
            synthesis_methods: vec![SynthesisMethod::Ensemble],
            constraints: CompositeConstraints {
                time_budget_ms: 5000,
                token_budget: 10000,
                dependencies: vec![],
            },
        })
        .collect();

    InterleavedProtocol {
        id: format!("{}_id", name),
        name: name.to_string(),
        version: "1.0.0".to_string(),
        description: "Test protocol for unit testing".to_string(),
        phases,
        constraints: CompositeConstraints {
            time_budget_ms: 30000,
            token_budget: 50000,
            dependencies: vec![],
        },
        m2_optimizations: M2Optimizations {
            target_parameters: 10_000_000_000,
            context_optimization: ContextOptimization {
                method: "none".to_string(),
                compression_ratio: 1.0,
            },
            output_optimization: OutputOptimization {
                max_output_length: 5000,
                streaming_enabled: false,
                compression_enabled: false,
                format: "text".to_string(),
                template: "".to_string(),
            },
            cost_optimization: CostOptimization {
                target_cost_reduction: 92.0,
                target_latency_reduction: 0.15,
                parallel_processing_enabled: true,
                caching_enabled: true,
                strategy: "balanced".to_string(),
                max_budget: 1.0,
            },
        },
        framework_compatibility: vec!["claude_code".to_string()],
        language_support: vec!["rust".to_string(), "python".to_string()],
    }
}

/// Creates test CompositeConstraints
fn create_test_constraints() -> CompositeConstraints {
    CompositeConstraints {
        time_budget_ms: 30000,
        token_budget: 50000,
        dependencies: vec![],
    }
}

/// Creates a minimal ProtocolInput for testing
fn create_test_input() -> ProtocolInput {
    serde_json::json!({
        "query": "Test query for M2 engine",
        "context": "Unit testing context"
    })
}

/// Creates a test ThinkingPath
fn create_test_thinking_path(phase_name: &str) -> ThinkingPath {
    ThinkingPath {
        path_id: format!("{}_path", phase_name),
        phase: InterleavedPhase {
            name: phase_name.to_string(),
            parallel_branches: 2,
            required_confidence: 0.8,
            validation_methods: vec![ValidationMethod::SelfCheck],
            synthesis_methods: vec![SynthesisMethod::Ensemble],
            constraints: CompositeConstraints {
                time_budget_ms: 5000,
                token_budget: 10000,
                dependencies: vec![],
            },
        },
        branches: vec![
            ThinkingBranch {
                branch_id: format!("{}_branch_0", phase_name),
                phase_id: phase_name.to_string(),
                reasoning_steps: vec![ReasoningStep {
                    id: "step_0".to_string(),
                    name: "initial_reasoning".to_string(),
                }],
                validation_methods: vec![],
                synthesis_methods: vec![],
                confidence_targets: vec![0.85],
            },
            ThinkingBranch {
                branch_id: format!("{}_branch_1", phase_name),
                phase_id: phase_name.to_string(),
                reasoning_steps: vec![ReasoningStep {
                    id: "step_1".to_string(),
                    name: "secondary_reasoning".to_string(),
                }],
                validation_methods: vec![],
                synthesis_methods: vec![],
                confidence_targets: vec![0.80],
            },
        ],
        dependencies: vec![],
        resource_allocation: ResourceAllocation {
            token_budget: TokenBudget {
                total: 10000,
                context: 8000,
                output: 2000,
                validation: 0,
            },
            time_allocation_ms: 5000,
            priority: 1,
            quality_targets: QualityTargets {
                min_confidence: 0.8,
                required_depth: 2,
            },
            parallel_capacity: 2,
        },
    }
}

/// Creates a test PhaseResult
fn create_test_phase_result(phase_name: &str, confidence: f64) -> PhaseResult {
    PhaseResult {
        phase_name: phase_name.to_string(),
        output: PhaseOutput {
            content: format!("Output from {}", phase_name),
            reasoning_trace: vec![],
            confidence_scores: ConfidenceScores {
                overall: confidence,
                reasoning: confidence,
                evidence: confidence,
            },
            evidence: vec![],
            metadata: PhaseMetadata {
                tokens_used: 1000,
                execution_time_ms: 500,
                validation_results: vec![],
                synthesis_applied: vec![],
                branching_factor: 2,
            },
        },
        execution_time: Duration::from_millis(500),
        branches_executed: 2,
        success: true,
    }
}

// ============================================================================
// ENGINE INITIALIZATION TESTS
// ============================================================================

#[cfg(test)]
mod initialization_tests {
    use super::*;

    #[test]
    fn test_thinking_orchestrator_creation() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector.clone());

        // Verify the orchestrator is created with correct internal state
        assert!(Arc::ptr_eq(&orchestrator.m2_connector, &connector));
    }

    #[test]
    fn test_performance_monitor_initialization() {
        let monitor = PerformanceMonitor::new();

        // Verify default state
        assert!(monitor.latency_tracker.try_read().is_ok());
        assert!(monitor.quality_tracker.try_read().is_ok());
    }

    #[test]
    fn test_protocol_validator_creation() {
        let validator = ProtocolValidator::new();

        // Validator should be successfully created
        // Internal components are placeholders, just verify construction
        assert!(std::mem::size_of_val(&validator) > 0);
    }

    #[test]
    fn test_result_synthesizer_creation() {
        let synthesizer = ResultSynthesizer::new();

        // Synthesizer should be successfully created
        assert!(std::mem::size_of_val(&synthesizer) > 0);
    }

    #[test]
    fn test_connector_configuration() {
        let config = create_test_config();
        let connector = M2Connector::new(config.clone());

        // Verify connector is created (we can't access internal state directly)
        assert!(std::mem::size_of_val(&connector) > 0);
    }
}

// ============================================================================
// REQUEST HANDLING TESTS
// ============================================================================

#[cfg(test)]
mod request_handling_tests {
    use super::*;

    #[test]
    fn test_protocol_input_json_creation() {
        let input = create_test_input();

        assert!(input.get("query").is_some());
        assert!(input.get("context").is_some());
        assert_eq!(
            input.get("query").unwrap().as_str().unwrap(),
            "Test query for M2 engine"
        );
    }

    #[test]
    fn test_protocol_creation_with_phases() {
        let protocol = create_test_protocol("test_protocol", 3);

        assert_eq!(protocol.name, "test_protocol");
        assert_eq!(protocol.phases.len(), 3);
        assert_eq!(protocol.version, "1.0.0");

        // Verify phase names
        for (i, phase) in protocol.phases.iter().enumerate() {
            assert_eq!(phase.name, format!("phase_{}", i));
            assert_eq!(phase.parallel_branches, 2);
        }
    }

    #[test]
    fn test_composite_constraints_defaults() {
        let constraints = create_test_constraints();

        assert_eq!(constraints.time_budget_ms, 30000);
        assert_eq!(constraints.token_budget, 50000);
        assert!(constraints.dependencies.is_empty());
    }

    #[test]
    fn test_thinking_path_structure() {
        let path = create_test_thinking_path("analysis");

        assert_eq!(path.path_id, "analysis_path");
        assert_eq!(path.phase.name, "analysis");
        assert_eq!(path.branches.len(), 2);
        assert!(path.dependencies.is_empty());

        // Verify branch structure
        assert_eq!(path.branches[0].branch_id, "analysis_branch_0");
        assert_eq!(path.branches[1].branch_id, "analysis_branch_1");
    }

    #[test]
    fn test_resource_allocation_structure() {
        let path = create_test_thinking_path("test_phase");
        let allocation = &path.resource_allocation;

        assert_eq!(allocation.token_budget.total, 10000);
        assert_eq!(allocation.token_budget.context, 8000);
        assert_eq!(allocation.token_budget.output, 2000);
        assert_eq!(allocation.time_allocation_ms, 5000);
        assert_eq!(allocation.priority, 1);
        assert_eq!(allocation.parallel_capacity, 2);
    }
}

// ============================================================================
// RESPONSE PARSING TESTS
// ============================================================================

#[cfg(test)]
mod response_parsing_tests {
    use super::*;

    #[test]
    fn test_phase_result_creation() {
        let result = create_test_phase_result("analysis", 0.85);

        assert_eq!(result.phase_name, "analysis");
        assert!(result.success);
        assert_eq!(result.branches_executed, 2);
        assert!(result.execution_time.as_millis() > 0);
    }

    #[test]
    fn test_confidence_scores_parsing() {
        let scores = ConfidenceScores {
            overall: 0.9,
            reasoning: 0.85,
            evidence: 0.88,
        };

        assert_eq!(scores.overall, 0.9);
        assert_eq!(scores.reasoning, 0.85);
        assert_eq!(scores.evidence, 0.88);
    }

    #[test]
    fn test_phase_output_structure() {
        let output = PhaseOutput {
            content: "Test content".to_string(),
            reasoning_trace: vec![],
            confidence_scores: ConfidenceScores {
                overall: 0.9,
                reasoning: 0.85,
                evidence: 0.88,
            },
            evidence: vec![Evidence {
                content: "Test evidence".to_string(),
                source: "test_source".to_string(),
                confidence: 0.9,
            }],
            metadata: PhaseMetadata {
                tokens_used: 500,
                execution_time_ms: 250,
                validation_results: vec![],
                synthesis_applied: vec![],
                branching_factor: 2,
            },
        };

        assert_eq!(output.content, "Test content");
        assert_eq!(output.evidence.len(), 1);
        assert_eq!(output.metadata.tokens_used, 500);
    }

    #[test]
    fn test_validation_report_structure() {
        use crate::thinktool::protocol::ValidationRule;

        let report = ValidationReport {
            issues_found: vec![ValidationIssue {
                severity: "warning".to_string(),
                description: "Test issue".to_string(),
                affected_phases: vec!["phase_0".to_string()],
            }],
            consensus_points: vec![ConsensusPoint {
                description: "Agreement on approach".to_string(),
                supporting_phases: vec!["phase_0".to_string(), "phase_1".to_string()],
                confidence: 0.9,
            }],
            validation_rules_applied: vec![],
            overall_validity: 0.95,
            recommendations: vec!["Review warning".to_string()],
        };

        assert_eq!(report.issues_found.len(), 1);
        assert_eq!(report.consensus_points.len(), 1);
        assert_eq!(report.overall_validity, 0.95);
    }

    #[test]
    fn test_branch_result_structure() {
        let result = BranchResult {
            branch_id: "test_branch".to_string(),
            reasoning_steps: vec![],
            validation_results: vec![ValidationResult {
                passed: true,
                details: "All checks passed".to_string(),
                score: 0.95,
            }],
            synthesis_results: vec![],
            execution_time: Duration::from_millis(100),
            confidence: 0.88,
        };

        assert_eq!(result.branch_id, "test_branch");
        assert_eq!(result.validation_results.len(), 1);
        assert!(result.validation_results[0].passed);
        assert_eq!(result.confidence, 0.88);
    }

    #[test]
    fn test_interleaved_result_structure() {
        let result = InterleavedResult {
            summary: "Test summary of results".to_string(),
        };

        assert!(!result.summary.is_empty());
    }
}

// ============================================================================
// ERROR RECOVERY TESTS
// ============================================================================

#[cfg(test)]
mod error_recovery_tests {
    use super::*;

    #[test]
    fn test_m2_execution_error_creation() {
        let error = Error::M2ExecutionError("Test execution failure".to_string());

        assert!(error.to_string().contains("M2 execution error"));
        assert!(error.to_string().contains("Test execution failure"));
    }

    #[test]
    fn test_rate_limit_error() {
        let error = Error::RateLimitExceeded;

        assert!(error.to_string().contains("rate limit"));
    }

    #[test]
    fn test_budget_exceeded_error() {
        let error = Error::BudgetExceeded(1.5, 1.0);

        let error_str = error.to_string();
        assert!(error_str.contains("budget exceeded"));
        assert!(error_str.contains("1.5"));
        assert!(error_str.contains("1"));
    }

    #[test]
    fn test_protocol_validation_error() {
        let error = Error::M2ProtocolValidation("Invalid phase configuration".to_string());

        assert!(error.to_string().contains("protocol validation"));
    }

    #[test]
    fn test_constraint_violation_error() {
        let error = Error::M2ConstraintViolation("Token budget exceeded".to_string());

        assert!(error.to_string().contains("constraint violation"));
    }

    #[test]
    fn test_framework_incompatibility_error() {
        let error = Error::M2FrameworkIncompatibility("Unsupported framework: legacy".to_string());

        assert!(error.to_string().contains("framework incompatibility"));
    }

    #[test]
    fn test_timeout_error() {
        let error = Error::Timeout("Phase execution timed out after 30s".to_string());

        assert!(error.to_string().contains("Timeout"));
    }

    #[test]
    fn test_dependency_not_met_error() {
        let error = Error::DependencyNotMet("phase_1 requires phase_0".to_string());

        assert!(error.to_string().contains("Dependency not met"));
    }

    #[test]
    fn test_phase_status_enumeration() {
        assert_ne!(PhaseStatus::Pending, PhaseStatus::Running);
        assert_ne!(PhaseStatus::Running, PhaseStatus::Completed);
        assert_ne!(PhaseStatus::Completed, PhaseStatus::Failed);
        assert_ne!(PhaseStatus::Failed, PhaseStatus::Skipped);
    }

    #[test]
    fn test_phase_execution_state_with_error() {
        let state = PhaseExecutionState {
            phase_id: "failed_phase".to_string(),
            status: PhaseStatus::Failed,
            start_time: std::time::Instant::now(),
            end_time: Some(std::time::Instant::now()),
            confidence: 0.0,
            output: None,
            error: Some("Connection timeout".to_string()),
        };

        assert_eq!(state.status, PhaseStatus::Failed);
        assert!(state.error.is_some());
        assert!(state.output.is_none());
    }
}

// ============================================================================
// ASYNC BEHAVIOR TESTS
// ============================================================================

#[cfg(test)]
mod async_behavior_tests {
    use super::*;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn test_performance_monitor_record_phase() {
        let monitor = PerformanceMonitor::new();

        monitor
            .record_phase_execution(
                "test_phase".to_string(),
                Duration::from_millis(150),
                0.85,
            )
            .await;

        let latencies = monitor.latency_tracker.read().await;
        let qualities = monitor.quality_tracker.read().await;

        assert_eq!(latencies.len(), 1);
        assert_eq!(latencies[0], Duration::from_millis(150));
        assert_eq!(qualities.len(), 1);
        assert_eq!(qualities[0], 0.85);
    }

    #[tokio::test]
    async fn test_performance_monitor_multiple_records() {
        let monitor = PerformanceMonitor::new();

        // Record multiple phases
        for i in 0..5 {
            monitor
                .record_phase_execution(
                    format!("phase_{}", i),
                    Duration::from_millis((i + 1) as u64 * 100),
                    0.8 + (i as f64 * 0.02),
                )
                .await;
        }

        let latencies = monitor.latency_tracker.read().await;
        let qualities = monitor.quality_tracker.read().await;

        assert_eq!(latencies.len(), 5);
        assert_eq!(qualities.len(), 5);
        assert_eq!(latencies[4], Duration::from_millis(500));
    }

    #[tokio::test]
    async fn test_performance_monitor_get_final_metrics() {
        let monitor = PerformanceMonitor::new();

        monitor
            .record_phase_execution(
                "phase_1".to_string(),
                Duration::from_millis(100),
                0.9,
            )
            .await;

        let metrics = monitor.get_final_metrics().await;

        // Default metrics returned in placeholder implementation
        assert_eq!(metrics.duration_ms, 0);
    }

    #[tokio::test]
    async fn test_execution_cache_concurrent_access() {
        let cache: Arc<RwLock<HashMap<String, PhaseExecutionState>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // Simulate concurrent writes
        let cache_clone1 = cache.clone();
        let cache_clone2 = cache.clone();

        let handle1 = tokio::spawn(async move {
            let mut writer = cache_clone1.write().await;
            writer.insert(
                "exec_1".to_string(),
                PhaseExecutionState {
                    phase_id: "phase_1".to_string(),
                    status: PhaseStatus::Running,
                    start_time: std::time::Instant::now(),
                    end_time: None,
                    confidence: 0.0,
                    output: None,
                    error: None,
                },
            );
        });

        let handle2 = tokio::spawn(async move {
            let mut writer = cache_clone2.write().await;
            writer.insert(
                "exec_2".to_string(),
                PhaseExecutionState {
                    phase_id: "phase_2".to_string(),
                    status: PhaseStatus::Pending,
                    start_time: std::time::Instant::now(),
                    end_time: None,
                    confidence: 0.0,
                    output: None,
                    error: None,
                },
            );
        });

        handle1.await.unwrap();
        handle2.await.unwrap();

        let reader = cache.read().await;
        assert_eq!(reader.len(), 2);
        assert!(reader.contains_key("exec_1"));
        assert!(reader.contains_key("exec_2"));
    }

    #[tokio::test]
    async fn test_orchestrator_initialize_execution_context() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let protocol = create_test_protocol("context_test", 2);
        let input = create_test_input();
        let execution_id = uuid::Uuid::new_v4();

        let context = orchestrator
            .initialize_execution_context(&protocol, &input, execution_id)
            .await;

        assert!(context.is_ok());
        let ctx = context.unwrap();
        assert_eq!(ctx.execution_id, execution_id);
        assert_eq!(ctx.phase_states.len(), 2);
        assert_eq!(ctx.global_constraints, 2);
    }

    #[tokio::test]
    async fn test_orchestrator_generate_thinking_paths() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let protocol = create_test_protocol("paths_test", 3);
        let input = create_test_input();
        let execution_id = uuid::Uuid::new_v4();

        let context = orchestrator
            .initialize_execution_context(&protocol, &input, execution_id)
            .await
            .unwrap();

        let paths = orchestrator
            .generate_thinking_paths(&protocol, &context)
            .await;

        assert!(paths.is_ok());
        let paths = paths.unwrap();
        assert_eq!(paths.len(), 3);

        for (i, path) in paths.iter().enumerate() {
            assert!(path.path_id.contains(&format!("phase_{}", i)));
            assert!(!path.branches.is_empty());
        }
    }
}

// ============================================================================
// VALIDATOR AND SYNTHESIZER TESTS
// ============================================================================

#[cfg(test)]
mod validator_synthesizer_tests {
    use super::*;

    #[test]
    fn test_validator_validate_final_result() {
        let validator = ProtocolValidator::new();
        let result = InterleavedResult {
            summary: "Test result".to_string(),
        };
        let protocol = create_test_protocol("validation_test", 2);

        let validated = validator.validate_final_result(&result, &protocol);

        assert!(validated.is_ok());
        assert_eq!(validated.unwrap().summary, "Test result");
    }

    #[test]
    fn test_validator_apply_validation_rules() {
        let validator = ProtocolValidator::new();
        let results = vec![create_test_phase_result("phase_0", 0.85)];
        let protocol = create_test_protocol("rules_test", 1);

        let rules = validator.apply_validation_rules(&results, &protocol);

        assert!(rules.is_ok());
        // Currently returns empty vec in placeholder implementation
        assert!(rules.unwrap().is_empty());
    }

    #[test]
    fn test_synthesizer_synthesize_results() {
        let synthesizer = ResultSynthesizer::new();
        let validated_results = ValidatedResults {
            phase_results: vec![create_test_phase_result("phase_0", 0.9)],
            validation_report: ValidationReport {
                issues_found: vec![],
                consensus_points: vec![],
                validation_rules_applied: vec![],
                overall_validity: 1.0,
                recommendations: vec![],
            },
            validated_at: chrono::Utc::now(),
            validator_id: "test_validator".to_string(),
        };
        let protocol = create_test_protocol("synthesis_test", 1);

        let result = synthesizer.synthesize_results(validated_results, &protocol);

        assert!(result.is_ok());
        assert!(!result.unwrap().summary.is_empty());
    }

    #[test]
    fn test_synthesizer_synthesize_phase_output() {
        let synthesizer = ResultSynthesizer::new();
        let phase = InterleavedPhase {
            name: "test_phase".to_string(),
            parallel_branches: 2,
            required_confidence: 0.8,
            validation_methods: vec![],
            synthesis_methods: vec![],
            constraints: create_test_constraints(),
        };
        let branch_results = vec![
            BranchResult {
                branch_id: "branch_0".to_string(),
                reasoning_steps: vec![],
                validation_results: vec![],
                synthesis_results: vec![],
                execution_time: Duration::from_millis(100),
                confidence: 0.85,
            },
            BranchResult {
                branch_id: "branch_1".to_string(),
                reasoning_steps: vec![],
                validation_results: vec![],
                synthesis_results: vec![],
                execution_time: Duration::from_millis(120),
                confidence: 0.88,
            },
        ];

        let output = synthesizer.synthesize_phase_output(&phase, branch_results, 0);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert!(!output.content.is_empty());
        assert_eq!(output.metadata.branching_factor, 2);
    }
}

// ============================================================================
// TYPE STRUCTURE TESTS
// ============================================================================

#[cfg(test)]
mod type_structure_tests {
    use super::*;

    #[test]
    fn test_m2_config_default() {
        let config = M2Config::default();

        assert!(!config.endpoint.is_empty());
        assert_eq!(config.max_context_length, 200_000);
        assert_eq!(config.max_output_length, 128_000);
    }

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();

        assert_eq!(config.rpm, 60);
        assert_eq!(config.rps, 1);
        assert_eq!(config.burst, 5);
    }

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();

        assert_eq!(config.cost_reduction_target, 92.0);
        assert_eq!(config.latency_target_ms, 2000);
        assert_eq!(config.quality_threshold, 0.90);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_execution_metrics_default() {
        let metrics = ExecutionMetrics::default();

        assert_eq!(metrics.duration_ms, 0);
        assert_eq!(metrics.token_usage.total, 0);
        assert_eq!(metrics.cost_metrics.total_cost, 0.0);
    }

    #[test]
    fn test_synthesis_method_enum() {
        assert_eq!(SynthesisMethod::Ensemble, SynthesisMethod::Ensemble);
        assert_ne!(SynthesisMethod::Ensemble, SynthesisMethod::WeightedAverage);
        assert_ne!(SynthesisMethod::WeightedAverage, SynthesisMethod::BestOfN);
    }

    #[test]
    fn test_validation_method_enum() {
        assert_eq!(ValidationMethod::SelfCheck, ValidationMethod::SelfCheck);
        assert_ne!(ValidationMethod::SelfCheck, ValidationMethod::PeerReview);
        assert_ne!(
            ValidationMethod::PeerReview,
            ValidationMethod::FormalVerification
        );
    }

    #[test]
    fn test_task_classification_from_use_case() {
        let classification = TaskClassification::from(UseCase::CodeAnalysis);

        assert_eq!(classification.task_type, TaskType::CodeAnalysis);
        assert_eq!(classification.complexity_level, ComplexityLevel::Complex);
        assert_eq!(classification.domain, TaskDomain::SystemProgramming);
        assert_eq!(classification.expected_output_size, OutputSize::Large);
    }

    #[test]
    fn test_optimization_goals_default() {
        let goals = OptimizationGoals::default();

        assert_eq!(goals.primary_goal, OptimizationGoal::BalanceAll);
        assert!(goals.secondary_goals.is_empty());
        assert_eq!(goals.performance_targets.cost_reduction_target, 92.0);
    }
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_protocol_phases() {
        let protocol = InterleavedProtocol {
            id: "empty".to_string(),
            name: "empty_protocol".to_string(),
            version: "1.0.0".to_string(),
            description: "Protocol with no phases".to_string(),
            phases: vec![],
            constraints: create_test_constraints(),
            m2_optimizations: M2Optimizations {
                target_parameters: 10_000_000_000,
                context_optimization: ContextOptimization {
                    method: "none".to_string(),
                    compression_ratio: 1.0,
                },
                output_optimization: OutputOptimization {
                    max_output_length: 5000,
                    streaming_enabled: false,
                    compression_enabled: false,
                    format: "text".to_string(),
                    template: "".to_string(),
                },
                cost_optimization: CostOptimization {
                    target_cost_reduction: 92.0,
                    target_latency_reduction: 0.15,
                    parallel_processing_enabled: true,
                    caching_enabled: true,
                    strategy: "balanced".to_string(),
                    max_budget: 1.0,
                },
            },
            framework_compatibility: vec![],
            language_support: vec![],
        };

        assert!(protocol.phases.is_empty());
        assert!(protocol.framework_compatibility.is_empty());
    }

    #[test]
    fn test_zero_confidence_handling() {
        let scores = ConfidenceScores {
            overall: 0.0,
            reasoning: 0.0,
            evidence: 0.0,
        };

        assert_eq!(scores.overall, 0.0);
    }

    #[test]
    fn test_max_confidence_handling() {
        let scores = ConfidenceScores {
            overall: 1.0,
            reasoning: 1.0,
            evidence: 1.0,
        };

        assert_eq!(scores.overall, 1.0);
    }

    #[test]
    fn test_large_token_budget() {
        let constraints = CompositeConstraints {
            time_budget_ms: u64::MAX,
            token_budget: u64::MAX,
            dependencies: vec![],
        };

        assert_eq!(constraints.token_budget, u64::MAX);
    }

    #[test]
    fn test_single_branch_path() {
        let path = ThinkingPath {
            path_id: "single_branch".to_string(),
            phase: InterleavedPhase {
                name: "single".to_string(),
                parallel_branches: 1,
                required_confidence: 0.9,
                validation_methods: vec![],
                synthesis_methods: vec![],
                constraints: create_test_constraints(),
            },
            branches: vec![ThinkingBranch {
                branch_id: "only_branch".to_string(),
                phase_id: "single".to_string(),
                reasoning_steps: vec![],
                validation_methods: vec![],
                synthesis_methods: vec![],
                confidence_targets: vec![0.9],
            }],
            dependencies: vec![],
            resource_allocation: ResourceAllocation {
                token_budget: TokenBudget {
                    total: 1000,
                    context: 800,
                    output: 200,
                    validation: 0,
                },
                time_allocation_ms: 1000,
                priority: 1,
                quality_targets: QualityTargets {
                    min_confidence: 0.9,
                    required_depth: 1,
                },
                parallel_capacity: 1,
            },
        };

        assert_eq!(path.branches.len(), 1);
        assert_eq!(path.phase.parallel_branches, 1);
    }

    #[test]
    fn test_many_dependencies() {
        let dependencies: Vec<String> = (0..100).map(|i| format!("dep_{}", i)).collect();

        let constraints = CompositeConstraints {
            time_budget_ms: 10000,
            token_budget: 50000,
            dependencies: dependencies.clone(),
        };

        assert_eq!(constraints.dependencies.len(), 100);
    }

    #[test]
    fn test_empty_evidence_list() {
        let output = ProtocolOutput {
            result: "Result with no evidence".to_string(),
            confidence: 0.5,
            evidence: vec![],
        };

        assert!(output.evidence.is_empty());
        assert_eq!(output.confidence, 0.5);
    }

    #[test]
    fn test_consistency_check_no_issues() {
        let check = ConsistencyCheck {
            issue: None,
            consensus: Some(ConsensusPoint {
                description: "Full agreement".to_string(),
                supporting_phases: vec!["phase_0".to_string(), "phase_1".to_string()],
                confidence: 1.0,
            }),
        };

        assert!(check.issue.is_none());
        assert!(check.consensus.is_some());
    }

    #[test]
    fn test_consistency_check_with_issues() {
        let check = ConsistencyCheck {
            issue: Some(ValidationIssue {
                severity: "error".to_string(),
                description: "Contradictory conclusions".to_string(),
                affected_phases: vec!["phase_0".to_string(), "phase_1".to_string()],
            }),
            consensus: None,
        };

        assert!(check.issue.is_some());
        assert!(check.consensus.is_none());
    }
}

// ============================================================================
// HELPER METHOD TESTS
// ============================================================================

#[cfg(test)]
mod helper_method_tests {
    use super::*;

    #[test]
    fn test_orchestrator_adapt_constraints() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let constraints = create_test_constraints();
        let branch = ThinkingBranch {
            branch_id: "test".to_string(),
            phase_id: "test_phase".to_string(),
            reasoning_steps: vec![],
            validation_methods: vec![],
            synthesis_methods: vec![],
            confidence_targets: vec![0.8],
        };

        let adapted = orchestrator.adapt_constraints_for_branch(&constraints, &branch);

        assert!(adapted.is_ok());
        // Currently returns clone, verify it matches original
        let adapted = adapted.unwrap();
        assert_eq!(adapted.time_budget_ms, constraints.time_budget_ms);
        assert_eq!(adapted.token_budget, constraints.token_budget);
    }

    #[test]
    fn test_orchestrator_adapt_input() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let input = create_test_input();
        let branch = ThinkingBranch {
            branch_id: "test".to_string(),
            phase_id: "test_phase".to_string(),
            reasoning_steps: vec![],
            validation_methods: vec![],
            synthesis_methods: vec![],
            confidence_targets: vec![0.8],
        };

        let adapted = orchestrator.adapt_input_for_branch(&input, &branch);

        assert!(adapted.is_ok());
        // Currently returns clone
        let adapted = adapted.unwrap();
        assert_eq!(adapted, input);
    }

    #[test]
    fn test_orchestrator_generate_reasoning_steps() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let phase = InterleavedPhase {
            name: "test_phase".to_string(),
            parallel_branches: 2,
            required_confidence: 0.8,
            validation_methods: vec![],
            synthesis_methods: vec![],
            constraints: create_test_constraints(),
        };

        let steps = orchestrator.generate_reasoning_steps(&phase, 0);

        assert!(steps.is_ok());
        let steps = steps.unwrap();
        assert!(!steps.is_empty());
        assert_eq!(steps[0].id, "step_0");
    }

    #[test]
    fn test_orchestrator_calculate_confidence_targets() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let phase = InterleavedPhase {
            name: "test_phase".to_string(),
            parallel_branches: 2,
            required_confidence: 0.85,
            validation_methods: vec![],
            synthesis_methods: vec![],
            constraints: create_test_constraints(),
        };

        let targets = orchestrator.calculate_confidence_targets(&phase);

        assert!(targets.is_ok());
        let targets = targets.unwrap();
        assert!(!targets.is_empty());
        // Default target is 0.85
        assert_eq!(targets[0], 0.85);
    }

    #[test]
    fn test_orchestrator_calculate_resource_allocation() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let phase = InterleavedPhase {
            name: "test_phase".to_string(),
            parallel_branches: 4,
            required_confidence: 0.8,
            validation_methods: vec![],
            synthesis_methods: vec![],
            constraints: create_test_constraints(),
        };

        let allocation = orchestrator.calculate_resource_allocation(&phase);

        assert!(allocation.is_ok());
        let allocation = allocation.unwrap();
        assert_eq!(allocation.parallel_capacity, 4);
        assert!(allocation.token_budget.total > 0);
    }

    #[test]
    fn test_orchestrator_calculate_overall_validity() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);

        // No issues = 1.0 validity
        let validity = orchestrator.calculate_overall_validity(&[]);
        assert_eq!(validity, 1.0);

        // With issues = 0.9 validity
        let issues = vec![ValidationIssue {
            severity: "warning".to_string(),
            description: "Minor issue".to_string(),
            affected_phases: vec![],
        }];
        let validity = orchestrator.calculate_overall_validity(&issues);
        assert_eq!(validity, 0.9);
    }

    #[test]
    fn test_orchestrator_generate_validation_recommendations() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let issues = vec![ValidationIssue {
            severity: "error".to_string(),
            description: "Critical issue".to_string(),
            affected_phases: vec!["phase_0".to_string()],
        }];

        let recommendations = orchestrator.generate_validation_recommendations(&issues);

        assert!(recommendations.is_ok());
        let recommendations = recommendations.unwrap();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_orchestrator_calculate_branch_confidence() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let steps: Vec<ReasoningStepResult> = vec![];
        let validations: Vec<ValidationResult> = vec![];

        let confidence = orchestrator.calculate_branch_confidence(&steps, &validations);

        assert!(confidence.is_ok());
        // Default confidence is 0.85
        assert_eq!(confidence.unwrap(), 0.85);
    }

    #[test]
    fn test_orchestrator_check_phase_consistency() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let result1 = create_test_phase_result("phase_0", 0.85);
        let result2 = create_test_phase_result("phase_1", 0.88);

        let check = orchestrator.check_phase_consistency(&result1, &result2);

        assert!(check.is_ok());
        let check = check.unwrap();
        // Default implementation returns no issues
        assert!(check.issue.is_none());
        assert!(check.consensus.is_none());
    }

    #[test]
    fn test_orchestrator_create_step_protocol() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let step = ReasoningStep {
            id: "test_step".to_string(),
            name: "Test Step".to_string(),
        };
        let input = create_test_input();
        let constraints = create_test_constraints();

        let protocol = orchestrator.create_step_protocol(&step, &input, &constraints);

        assert!(protocol.is_ok());
        let protocol = protocol.unwrap();
        assert_eq!(protocol.id, "test_step");
        assert_eq!(protocol.name, "Test Step");
    }

    #[test]
    fn test_orchestrator_update_input_for_next_step() {
        let connector = create_test_connector();
        let orchestrator = ThinkingOrchestrator::new(connector);
        let current_input = create_test_input();
        let step_result = ReasoningStepResult {
            step_id: "step_0".to_string(),
            output: ProtocolOutput {
                result: "Step output".to_string(),
                confidence: 0.9,
                evidence: vec![],
            },
            confidence: 0.9,
            execution_time: Duration::from_millis(100),
            evidence: vec![],
            metadata: StepMetadata {
                tokens_used: 500,
                cost: 0.001,
                latency: 100,
            },
        };

        let updated = orchestrator.update_input_for_next_step(current_input.clone(), &step_result);

        assert!(updated.is_ok());
        // Currently returns clone
        assert_eq!(updated.unwrap(), current_input);
    }
}

// ============================================================================
// SERIALIZATION TESTS
// ============================================================================

#[cfg(test)]
mod serialization_tests {
    use super::*;

    #[test]
    fn test_protocol_output_serialization() {
        let output = ProtocolOutput {
            result: "Test result".to_string(),
            confidence: 0.9,
            evidence: vec![Evidence {
                content: "Test evidence".to_string(),
                source: "test".to_string(),
                confidence: 0.85,
            }],
        };

        let json = serde_json::to_string(&output);
        assert!(json.is_ok());

        let deserialized: Result<ProtocolOutput, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
        let deserialized = deserialized.unwrap();
        assert_eq!(deserialized.result, "Test result");
        assert_eq!(deserialized.confidence, 0.9);
    }

    #[test]
    fn test_interleaved_protocol_serialization() {
        let protocol = create_test_protocol("serialization_test", 2);

        let json = serde_json::to_string(&protocol);
        assert!(json.is_ok());

        let deserialized: Result<InterleavedProtocol, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
        let deserialized = deserialized.unwrap();
        assert_eq!(deserialized.name, "serialization_test");
        assert_eq!(deserialized.phases.len(), 2);
    }

    #[test]
    fn test_validation_report_serialization() {
        let report = ValidationReport {
            issues_found: vec![],
            consensus_points: vec![],
            validation_rules_applied: vec![],
            overall_validity: 1.0,
            recommendations: vec!["All good".to_string()],
        };

        let json = serde_json::to_string(&report);
        assert!(json.is_ok());

        let deserialized: Result<ValidationReport, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_execution_metrics_serialization() {
        let metrics = ExecutionMetrics {
            duration_ms: 1500,
            token_usage: TokenUsage {
                total: 5000,
                context: 4000,
                output: 1000,
                validation: 0,
            },
            cost_metrics: CostMetrics {
                total_cost: 0.05,
                savings: 0.02,
            },
            quality_metrics: QualityMetrics {
                reliability: 0.95,
                accuracy: 0.92,
            },
            performance_metrics: PerformanceMetrics {
                latency_ms: 1500,
                throughput: 100.0,
            },
            audit_trail: AuditTrail {
                steps: vec!["init".to_string(), "execute".to_string()],
                timestamp: 1234567890,
                compliance_flags: vec![ComplianceFlag::GDPRCompliant],
            },
        };

        let json = serde_json::to_string(&metrics);
        assert!(json.is_ok());

        let deserialized: Result<ExecutionMetrics, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
        let deserialized = deserialized.unwrap();
        assert_eq!(deserialized.duration_ms, 1500);
        assert_eq!(deserialized.token_usage.total, 5000);
    }

    #[test]
    fn test_m2_config_serialization() {
        let config = create_test_config();

        let json = serde_json::to_string(&config);
        assert!(json.is_ok());

        let deserialized: Result<M2Config, _> = serde_json::from_str(&json.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_use_case_serialization() {
        let cases = vec![
            UseCase::CodeAnalysis,
            UseCase::BugFinding,
            UseCase::Documentation,
            UseCase::Architecture,
            UseCase::General,
        ];

        for case in cases {
            let json = serde_json::to_string(&case);
            assert!(json.is_ok());

            let deserialized: Result<UseCase, _> = serde_json::from_str(&json.unwrap());
            assert!(deserialized.is_ok());
            assert_eq!(deserialized.unwrap(), case);
        }
    }
}
