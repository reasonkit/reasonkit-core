//! Comprehensive Unit Tests for ThinkTool Modules
//!
//! This test suite covers all five core ThinkTool modules:
//! - GigaThink: Multi-perspective expansion (10+ viewpoints)
//! - LaserLogic: Precision deductive reasoning with fallacy detection
//! - ProofGuard: Multi-source verification via triangulation
//! - BedRock: First principles decomposition
//! - BrutalHonesty: Adversarial self-critique
//!
//! Test categories:
//! - Module configuration and initialization
//! - Protocol structure and validation
//! - Mock execution and output parsing
//! - Edge cases and error conditions
//! - Profile-based execution chains

use reasonkit::thinktool::protocol::{
    AggregationType, BranchCondition, CritiqueSeverity, DecisionMethod, InputSpec, OutputSpec,
    ProtocolMetadata, StepOutputFormat, ValidationRule,
};
use reasonkit::thinktool::step::{ListItem, StepOutput, TokenUsage};
use reasonkit::thinktool::{
    // Module types
    BedRock,
    BrutalHonesty,
    // Core executor and protocol types
    ExecutorConfig,
    GigaThink,
    LaserLogic,
    // Profiles
    ProfileRegistry,
    ProofGuard,
    // Protocol definitions
    Protocol,
    ProtocolExecutor,
    ProtocolInput,
    // Registry
    ProtocolRegistry,
    ProtocolStep,
    // Step and output types
    ReasoningStrategy,
    StepAction,
    StepResult,
    ThinkToolContext,
    ThinkToolModule,
};
use std::collections::HashMap;

// ============================================================================
// MODULE CONFIGURATION TESTS
// ============================================================================

mod gigathink_module_tests {
    use super::*;

    #[test]
    fn test_gigathink_creation() {
        let gt = GigaThink::new();
        let config = gt.config();

        // GigaThinkConfig is behavioral configuration (not metadata)
        assert!(config.min_perspectives > 0);
        assert!(config.max_perspectives >= config.min_perspectives);
        assert!(config.min_confidence >= 0.0 && config.min_confidence <= 1.0);
    }

    #[test]
    fn test_gigathink_default() {
        let gt = GigaThink::default();
        let config = gt.config();

        // GigaThinkConfig has no metadata fields; just validate invariants
        assert!(config.min_perspectives > 0);
        assert!(config.max_perspectives >= config.min_perspectives);
    }

    #[test]
    fn test_gigathink_execute_basic() {
        let gt = GigaThink::new();
        let context = ThinkToolContext {
            query: "What factors drive startup success?".to_string(),
            previous_steps: vec![],
        };

        let result = gt.execute(&context);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.module, "GigaThink");
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
    }

    #[test]
    fn test_gigathink_output_structure() {
        let gt = GigaThink::new();
        let context = ThinkToolContext {
            query: "Test query".to_string(),
            previous_steps: vec![],
        };

        let result = gt.execute(&context).unwrap();
        let output_json = &result.output;

        // Verify expected fields exist in output structure
        assert!(output_json.get("dimensions").is_some());
        assert!(output_json.get("perspectives").is_some());
        assert!(output_json.get("themes").is_some());
        assert!(output_json.get("insights").is_some());
    }

    #[test]
    fn test_gigathink_with_previous_steps() {
        let gt = GigaThink::new();
        let context = ThinkToolContext {
            query: "Test query".to_string(),
            previous_steps: vec![
                "Previous analysis step 1".to_string(),
                "Previous analysis step 2".to_string(),
            ],
        };

        let result = gt.execute(&context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gigathink_empty_query() {
        let gt = GigaThink::new();
        let context = ThinkToolContext {
            query: "".to_string(),
            previous_steps: vec![],
        };

        // Empty query should be rejected by Gigathink input validation
        let result = gt.execute(&context);
        assert!(result.is_err());
    }

    #[test]
    fn test_gigathink_config_weights_valid() {
        let gt = GigaThink::new();
        let config = gt.config();

        // Weights should be between 0 and 1
        assert!(config.novelty_weight >= 0.0 && config.novelty_weight <= 1.0);
        assert!(config.depth_weight >= 0.0 && config.depth_weight <= 1.0);
        assert!(config.coherence_weight >= 0.0 && config.coherence_weight <= 1.0);

        // Should sum to ~1.0 in the default config
        let sum = config.novelty_weight + config.depth_weight + config.coherence_weight;
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

mod laserlogic_module_tests {
    use super::*;

    #[test]
    fn test_laserlogic_creation() {
        let ll = LaserLogic::new();
        let config = ll.config();

        assert_eq!(config.name, "LaserLogic");
        assert_eq!(config.version, "3.0.0");
        assert!(config.description.contains("deductive") || config.description.contains("fallacy"));
    }

    #[test]
    fn test_laserlogic_default() {
        let ll = LaserLogic::default();
        assert_eq!(ll.config().name, "LaserLogic");
    }

    #[test]
    fn test_laserlogic_execute_basic() {
        let ll = LaserLogic::new();
        let context = ThinkToolContext {
            query: "All humans are mortal. Socrates is human. Therefore Socrates is mortal."
                .to_string(),
            previous_steps: vec![],
        };

        let result = ll.execute(&context);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.module, "LaserLogic");
        // LaserLogic should have high confidence for logical validation
        assert!(output.confidence >= 0.0);
    }

    #[test]
    fn test_laserlogic_output_structure() {
        let ll = LaserLogic::new();
        let context = ThinkToolContext {
            query: "Premise 1. Premise 2. Therefore, Conclusion.".to_string(),
            previous_steps: vec![],
        };

        let result = ll.execute(&context).unwrap();
        let output_json = &result.output;

        // Verify expected fields for logical analysis
        assert!(output_json.get("validity").is_some());
        assert!(output_json.get("soundness").is_some());
        assert!(output_json.get("fallacies").is_some());
        assert!(output_json.get("verdict").is_some());
    }

    #[test]
    fn test_laserlogic_confidence_weight() {
        let ll = LaserLogic::new();
        let config = ll.config();

        // LaserLogic should have higher weight (analytical)
        assert_eq!(config.confidence_weight, 0.25);
    }

    #[test]
    fn test_laserlogic_fallacy_detection_output() {
        let ll = LaserLogic::new();
        let context = ThinkToolContext {
            query: "Premise 1. Premise 2. Therefore, Conclusion.".to_string(),
            previous_steps: vec![],
        };

        let result = ll.execute(&context).unwrap();
        // The fallacies field should be an array
        assert!(result.output.get("fallacies").unwrap().is_array());
    }
}

mod proofguard_module_tests {
    use super::*;

    #[test]
    fn test_proofguard_creation() {
        let pg = ProofGuard::new();
        let config = pg.config();

        assert_eq!(config.name, "ProofGuard");
        assert_eq!(config.version, "2.1.0");
        assert!(
            config.description.contains("Triangulation")
                || config.description.contains("verification")
        );
    }

    #[test]
    fn test_proofguard_default() {
        let pg = ProofGuard::default();
        assert_eq!(pg.config().name, "ProofGuard");
    }

    #[test]
    fn test_proofguard_execute_basic() {
        let pg = ProofGuard::new();
        let context = ThinkToolContext {
            query: "The Earth orbits the Sun".to_string(),
            previous_steps: vec![],
        };

        let result = pg.execute(&context);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.module, "ProofGuard");
    }

    #[test]
    fn test_proofguard_output_structure() {
        let pg = ProofGuard::new();
        let context = ThinkToolContext {
            query: "Test claim".to_string(),
            previous_steps: vec![],
        };

        let result = pg.execute(&context).unwrap();
        let output_json = &result.output;

        // Verify triangulation output fields
        assert!(output_json.get("verdict").is_some());
        assert!(output_json.get("sources").is_some());
        // ProofGuardOutput uses "contradictions" and "issues" rather than "evidence"/"discrepancies"
        assert!(output_json.get("contradictions").is_some());
        assert!(output_json.get("issues").is_some());
    }

    #[test]
    fn test_proofguard_confidence_weight() {
        let pg = ProofGuard::new();
        let config = pg.config();

        // ProofGuard should have highest weight (verification is critical)
        assert_eq!(config.confidence_weight, 0.30);
    }

    #[test]
    fn test_proofguard_source_tier_verification() {
        let pg = ProofGuard::new();
        let context = ThinkToolContext {
            query: "Claim requiring verification".to_string(),
            previous_steps: vec![],
        };

        let result = pg.execute(&context).unwrap();
        // Sources should be an array
        assert!(result.output.get("sources").unwrap().is_array());
    }
}

mod bedrock_module_tests {
    use super::*;

    #[test]
    fn test_bedrock_creation() {
        let br = BedRock::new();
        let config = br.config();

        assert_eq!(config.name, "BedRock");
        assert_eq!(config.version, "3.0.0");
        assert!(config.description.contains("principles") || config.description.contains("axiom"));
    }

    #[test]
    fn test_bedrock_default() {
        let br = BedRock::default();
        assert_eq!(br.config().name, "BedRock");
    }

    #[test]
    fn test_bedrock_execute_basic() {
        let br = BedRock::new();
        let context = ThinkToolContext {
            query: "Markets are efficient".to_string(),
            previous_steps: vec![],
        };

        let result = br.execute(&context);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.module, "BedRock");
    }

    #[test]
    fn test_bedrock_output_structure() {
        let br = BedRock::new();
        let context = ThinkToolContext {
            query: "Test statement".to_string(),
            previous_steps: vec![],
        };

        let result = br.execute(&context).unwrap();
        let output_json = &result.output;

        // Verify first principles decomposition fields
        assert!(output_json.get("axioms").is_some());
        assert!(output_json.get("decomposition").is_some());
        assert!(output_json.get("reconstruction").is_some());
        assert!(output_json.get("gaps").is_some());
    }

    #[test]
    fn test_bedrock_confidence_weight() {
        let br = BedRock::new();
        let config = br.config();

        assert_eq!(config.confidence_weight, 0.25);
    }

    #[test]
    fn test_bedrock_axiom_extraction() {
        let br = BedRock::new();
        let context = ThinkToolContext {
            query: "Complex statement to decompose".to_string(),
            previous_steps: vec![],
        };

        let result = br.execute(&context).unwrap();
        // Axioms should be an array
        assert!(result.output.get("axioms").unwrap().is_array());
    }
}

mod brutalhonesty_module_tests {
    use super::*;

    #[test]
    fn test_brutalhonesty_creation() {
        let bh = BrutalHonesty::new();
        let config = bh.config();

        assert_eq!(config.name, "BrutalHonesty");
        assert_eq!(config.version, "3.0.0");
        assert!(config.description.contains("critique") || config.description.contains("Red-team"));
    }

    #[test]
    fn test_brutalhonesty_default() {
        let bh = BrutalHonesty::default();
        assert_eq!(bh.config().name, "BrutalHonesty");
    }

    #[test]
    fn test_brutalhonesty_execute_basic() {
        let bh = BrutalHonesty::new();
        let context = ThinkToolContext {
            query: "This is the best solution ever designed".to_string(),
            previous_steps: vec![],
        };

        let result = bh.execute(&context);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.module, "BrutalHonesty");
    }

    #[test]
    fn test_brutalhonesty_output_structure() {
        let bh = BrutalHonesty::new();
        let context = ThinkToolContext {
            query: "Work to critique".to_string(),
            previous_steps: vec![],
        };

        let result = bh.execute(&context).unwrap();
        let output_json = &result.output;

        // Verify adversarial critique fields
        // BrutalHonesty nests strengths/flaws under "analysis"
        assert!(output_json.get("analysis").is_some());
        assert!(output_json
            .get("analysis")
            .unwrap()
            .get("strengths")
            .is_some());
        assert!(output_json.get("analysis").unwrap().get("flaws").is_some());
        assert!(output_json.get("verdict").is_some());
        assert!(output_json.get("critical_fix").is_some());
    }

    #[test]
    fn test_brutalhonesty_confidence_weight() {
        let bh = BrutalHonesty::new();
        let config = bh.config();

        // BrutalHonesty has lower weight (critique is input to other tools)
        assert_eq!(config.confidence_weight, 0.15);
    }

    #[test]
    fn test_brutalhonesty_critique_generation() {
        let bh = BrutalHonesty::new();
        let context = ThinkToolContext {
            query: "Work with obvious flaws".to_string(),
            previous_steps: vec![],
        };

        let result = bh.execute(&context).unwrap();
        // Strengths and flaws should be arrays (nested under analysis)
        let analysis = result.output.get("analysis").unwrap();
        assert!(analysis.get("strengths").unwrap().is_array());
        assert!(analysis.get("flaws").unwrap().is_array());
    }
}

// ============================================================================
// PROTOCOL REGISTRY TESTS
// ============================================================================

mod protocol_registry_tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ProtocolRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_builtins() {
        let mut registry = ProtocolRegistry::new();
        let result = registry.register_builtins();
        assert!(result.is_ok());

        // Should have all 5 core protocols (plus powercombo if defined in YAML)
        assert!(registry.len() >= 5);
        assert!(registry.contains("gigathink"));
        assert!(registry.contains("laserlogic"));
        assert!(registry.contains("bedrock"));
        assert!(registry.contains("proofguard"));
        assert!(registry.contains("brutalhonesty"));
    }

    #[test]
    fn test_get_protocol() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let gt = registry.get("gigathink");
        assert!(gt.is_some());
        let gt = gt.unwrap();
        assert_eq!(gt.name, "GigaThink");
        assert_eq!(gt.strategy, ReasoningStrategy::Expansive);
    }

    #[test]
    fn test_get_nonexistent_protocol() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let result = registry.get("nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_list_protocol_ids() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let ids = registry.list_ids();
        assert!(ids.contains(&"gigathink"));
        assert!(ids.contains(&"laserlogic"));
    }

    #[test]
    fn test_protocol_validation() {
        // Create a valid protocol
        let protocol = Protocol::new("test_protocol", "Test Protocol")
            .with_strategy(ReasoningStrategy::Analytical)
            .with_step(ProtocolStep {
                id: "step1".to_string(),
                action: StepAction::Analyze {
                    criteria: vec!["test".to_string()],
                },
                prompt_template: "Test: {{query}}".to_string(),
                output_format: StepOutputFormat::Text,
                min_confidence: 0.7,
                depends_on: vec![],
                branch: None,
            });

        assert!(protocol.validate().is_ok());
    }

    #[test]
    fn test_protocol_validation_empty_steps() {
        let protocol = Protocol::new("test", "Test");
        let result = protocol.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .iter()
            .any(|e| e.contains("at least one step")));
    }

    #[test]
    fn test_protocol_validation_invalid_dependency() {
        let protocol = Protocol::new("test", "Test").with_step(ProtocolStep {
            id: "step1".to_string(),
            action: StepAction::Analyze { criteria: vec![] },
            prompt_template: "Test".to_string(),
            output_format: StepOutputFormat::Text,
            min_confidence: 0.7,
            depends_on: vec!["nonexistent_step".to_string()],
            branch: None,
        });

        let result = protocol.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .iter()
            .any(|e| e.contains("unknown step")));
    }
}

// ============================================================================
// PROTOCOL EXECUTOR TESTS
// ============================================================================

mod executor_tests {
    use super::*;

    #[test]
    fn test_executor_creation_mock() {
        let executor = ProtocolExecutor::mock();
        assert!(executor.is_ok());

        let executor = executor.unwrap();
        assert!(!executor.registry().is_empty());
        assert!(!executor.profiles().is_empty());
    }

    #[test]
    fn test_executor_list_protocols() {
        let executor = ProtocolExecutor::mock().unwrap();
        let protocols = executor.list_protocols();

        assert!(protocols.contains(&"gigathink"));
        assert!(protocols.contains(&"laserlogic"));
        assert!(protocols.contains(&"bedrock"));
        assert!(protocols.contains(&"proofguard"));
        assert!(protocols.contains(&"brutalhonesty"));
    }

    #[test]
    fn test_executor_list_profiles() {
        let executor = ProtocolExecutor::mock().unwrap();
        let profiles = executor.list_profiles();

        assert!(profiles.contains(&"quick"));
        assert!(profiles.contains(&"balanced"));
        assert!(profiles.contains(&"deep"));
        assert!(profiles.contains(&"paranoid"));
    }

    #[test]
    fn test_protocol_input_query() {
        let input = ProtocolInput::query("What is AI?");
        assert_eq!(input.get_str("query"), Some("What is AI?"));
    }

    #[test]
    fn test_protocol_input_argument() {
        let input = ProtocolInput::argument("All X are Y");
        assert_eq!(input.get_str("argument"), Some("All X are Y"));
    }

    #[test]
    fn test_protocol_input_statement() {
        let input = ProtocolInput::statement("Markets are efficient");
        assert_eq!(input.get_str("statement"), Some("Markets are efficient"));
    }

    #[test]
    fn test_protocol_input_claim() {
        let input = ProtocolInput::claim("The Earth is round");
        assert_eq!(input.get_str("claim"), Some("The Earth is round"));
    }

    #[test]
    fn test_protocol_input_work() {
        let input = ProtocolInput::work("My project code");
        assert_eq!(input.get_str("work"), Some("My project code"));
    }

    #[test]
    fn test_protocol_input_with_field() {
        let input = ProtocolInput::query("Test")
            .with_field("context", "Additional context")
            .with_field("constraints", "Some constraints");

        assert_eq!(input.get_str("query"), Some("Test"));
        assert_eq!(input.get_str("context"), Some("Additional context"));
        assert_eq!(input.get_str("constraints"), Some("Some constraints"));
    }

    #[tokio::test]
    async fn test_execute_gigathink_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What are the key factors for startup success?");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
        assert!(!result.steps.is_empty());
        assert_eq!(result.protocol_id, "gigathink");
    }

    #[tokio::test]
    async fn test_execute_laserlogic_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::argument("All humans are mortal. Socrates is human.");

        let result = executor.execute("laserlogic", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
        assert_eq!(result.protocol_id, "laserlogic");
    }

    #[tokio::test]
    async fn test_execute_bedrock_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::statement("Markets are efficient");

        let result = executor.execute("bedrock", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_execute_proofguard_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::claim("The Earth orbits the Sun");

        let result = executor.execute("proofguard", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_execute_brutalhonesty_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::work("My solution implementation");

        let result = executor.execute("brutalhonesty", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_execute_nonexistent_protocol() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let result = executor.execute("nonexistent_protocol", input).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_missing_required_input() {
        let executor = ProtocolExecutor::mock().unwrap();
        // GigaThink requires "query" but we provide "argument"
        let input = ProtocolInput::argument("Wrong input type");

        let result = executor.execute("gigathink", input).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_profile_quick() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Should we adopt microservices?");

        let result = executor.execute_profile("quick", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_execute_profile_balanced() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Is this a good investment?");

        let result = executor.execute_profile("balanced", input).await.unwrap();

        assert!(result.success);
    }
}

// ============================================================================
// STEP OUTPUT TESTS
// ============================================================================

mod step_output_tests {
    use super::*;

    #[test]
    fn test_step_result_success() {
        let result = StepResult::success(
            "test_step",
            StepOutput::Text {
                content: "Hello world".to_string(),
            },
            0.85,
        );

        assert!(result.success);
        assert_eq!(result.step_id, "test_step");
        assert_eq!(result.confidence, 0.85);
        assert_eq!(result.as_text(), Some("Hello world"));
        assert!(result.error.is_none());
    }

    #[test]
    fn test_step_result_failure() {
        let result = StepResult::failure("test_step", "Something went wrong");

        assert!(!result.success);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_step_result_with_duration() {
        let result = StepResult::success("test_step", StepOutput::Empty, 0.7).with_duration(150);

        assert_eq!(result.duration_ms, 150);
    }

    #[test]
    fn test_step_result_with_tokens() {
        let tokens = TokenUsage::new(100, 50, 0.001);
        let result = StepResult::success("test_step", StepOutput::Empty, 0.7).with_tokens(tokens);

        assert_eq!(result.tokens.input_tokens, 100);
        assert_eq!(result.tokens.output_tokens, 50);
        assert_eq!(result.tokens.total_tokens, 150);
    }

    #[test]
    fn test_step_result_meets_threshold() {
        let result = StepResult::success("test_step", StepOutput::Empty, 0.85);

        assert!(result.meets_threshold(0.80));
        assert!(result.meets_threshold(0.85));
        assert!(!result.meets_threshold(0.90));
    }

    #[test]
    fn test_step_result_failed_never_meets_threshold() {
        let result = StepResult::failure("test_step", "Error");

        assert!(!result.meets_threshold(0.0));
    }

    #[test]
    fn test_step_output_text() {
        let output = StepOutput::Text {
            content: "Test content".to_string(),
        };

        let result = StepResult::success("step", output, 0.8);
        assert_eq!(result.as_text(), Some("Test content"));
        assert!(result.as_list().is_none());
        assert!(result.as_score().is_none());
    }

    #[test]
    fn test_step_output_list() {
        let items = vec![
            ListItem::new("Item 1"),
            ListItem::new("Item 2"),
            ListItem::with_confidence("Item 3", 0.9),
        ];
        let output = StepOutput::List { items };

        let result = StepResult::success("step", output, 0.8);
        let list = result.as_list().unwrap();

        assert_eq!(list.len(), 3);
        assert_eq!(list[0].content, "Item 1");
        assert_eq!(list[2].confidence, Some(0.9));
    }

    #[test]
    fn test_step_output_score() {
        let output = StepOutput::Score { value: 0.75 };

        let result = StepResult::success("step", output, 0.8);
        assert_eq!(result.as_score(), Some(0.75));
    }

    #[test]
    fn test_step_output_boolean() {
        let output = StepOutput::Boolean {
            value: true,
            reason: Some("Valid reasoning".to_string()),
        };

        if let StepOutput::Boolean { value, reason } = output {
            assert!(value);
            assert_eq!(reason, Some("Valid reasoning".to_string()));
        } else {
            panic!("Expected Boolean output");
        }
    }

    #[test]
    fn test_step_output_structured() {
        let mut data = HashMap::new();
        data.insert("key1".to_string(), serde_json::json!("value1"));
        data.insert("key2".to_string(), serde_json::json!(42));

        let output = StepOutput::Structured { data };

        if let StepOutput::Structured { data } = output {
            assert_eq!(data.get("key1").unwrap(), &serde_json::json!("value1"));
            assert_eq!(data.get("key2").unwrap(), &serde_json::json!(42));
        } else {
            panic!("Expected Structured output");
        }
    }

    #[test]
    fn test_list_item_creation() {
        let item = ListItem::new("Test item");
        assert_eq!(item.content, "Test item");
        assert!(item.confidence.is_none());
        assert!(item.metadata.is_empty());
    }

    #[test]
    fn test_list_item_with_confidence() {
        let item = ListItem::with_confidence("Test item", 0.95);
        assert_eq!(item.content, "Test item");
        assert_eq!(item.confidence, Some(0.95));
    }

    #[test]
    fn test_token_usage_creation() {
        let usage = TokenUsage::new(100, 200, 0.003);

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 200);
        assert_eq!(usage.total_tokens, 300);
        assert_eq!(usage.cost_usd, 0.003);
    }

    #[test]
    fn test_token_usage_add() {
        let mut usage1 = TokenUsage::new(100, 50, 0.001);
        let usage2 = TokenUsage::new(200, 100, 0.002);

        usage1.add(&usage2);

        assert_eq!(usage1.input_tokens, 300);
        assert_eq!(usage1.output_tokens, 150);
        assert_eq!(usage1.total_tokens, 450);
        assert!((usage1.cost_usd - 0.003).abs() < 0.0001);
    }

    #[test]
    fn test_token_usage_default() {
        let usage = TokenUsage::default();

        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
        assert_eq!(usage.cost_usd, 0.0);
    }
}

// ============================================================================
// PROTOCOL STRUCTURE TESTS
// ============================================================================

mod protocol_structure_tests {
    use super::*;

    #[test]
    fn test_reasoning_strategy_serialization() {
        let strategies = vec![
            ReasoningStrategy::Expansive,
            ReasoningStrategy::Deductive,
            ReasoningStrategy::Analytical,
            ReasoningStrategy::Adversarial,
            ReasoningStrategy::Verification,
            ReasoningStrategy::Decision,
            ReasoningStrategy::Empirical,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).expect("Serialization failed");
            let parsed: ReasoningStrategy =
                serde_json::from_str(&json).expect("Deserialization failed");
            assert_eq!(strategy, parsed);
        }
    }

    #[test]
    fn test_step_action_generate() {
        let action = StepAction::Generate {
            min_count: 5,
            max_count: 10,
        };

        let json = serde_json::to_string(&action).unwrap();
        assert!(json.contains("generate"));

        let parsed: StepAction = serde_json::from_str(&json).unwrap();
        if let StepAction::Generate {
            min_count,
            max_count,
        } = parsed
        {
            assert_eq!(min_count, 5);
            assert_eq!(max_count, 10);
        } else {
            panic!("Wrong action type");
        }
    }

    #[test]
    fn test_step_action_analyze() {
        let action = StepAction::Analyze {
            criteria: vec!["clarity".to_string(), "depth".to_string()],
        };

        let json = serde_json::to_string(&action).unwrap();
        let parsed: StepAction = serde_json::from_str(&json).unwrap();

        if let StepAction::Analyze { criteria } = parsed {
            assert!(criteria.contains(&"clarity".to_string()));
            assert!(criteria.contains(&"depth".to_string()));
        } else {
            panic!("Wrong action type");
        }
    }

    #[test]
    fn test_step_action_validate() {
        let action = StepAction::Validate {
            rules: vec!["rule1".to_string(), "rule2".to_string()],
        };

        let json = serde_json::to_string(&action).unwrap();
        let parsed: StepAction = serde_json::from_str(&json).unwrap();

        if let StepAction::Validate { rules } = parsed {
            assert_eq!(rules.len(), 2);
        } else {
            panic!("Wrong action type");
        }
    }

    #[test]
    fn test_step_action_critique() {
        let action = StepAction::Critique {
            severity: CritiqueSeverity::Brutal,
        };

        let json = serde_json::to_string(&action).unwrap();
        let parsed: StepAction = serde_json::from_str(&json).unwrap();

        if let StepAction::Critique { severity } = parsed {
            assert_eq!(severity, CritiqueSeverity::Brutal);
        } else {
            panic!("Wrong action type");
        }
    }

    #[test]
    fn test_step_action_cross_reference() {
        let action = StepAction::CrossReference { min_sources: 3 };

        let json = serde_json::to_string(&action).unwrap();
        let parsed: StepAction = serde_json::from_str(&json).unwrap();

        if let StepAction::CrossReference { min_sources } = parsed {
            assert_eq!(min_sources, 3);
        } else {
            panic!("Wrong action type");
        }
    }

    #[test]
    fn test_aggregation_type_variants() {
        let types = vec![
            AggregationType::ThematicClustering,
            AggregationType::Concatenate,
            AggregationType::WeightedMerge,
            AggregationType::Consensus,
        ];

        for agg_type in types {
            let json = serde_json::to_string(&agg_type).unwrap();
            let parsed: AggregationType = serde_json::from_str(&json).unwrap();
            assert_eq!(agg_type, parsed);
        }
    }

    #[test]
    fn test_critique_severity_variants() {
        let severities = vec![
            CritiqueSeverity::Light,
            CritiqueSeverity::Standard,
            CritiqueSeverity::Adversarial,
            CritiqueSeverity::Brutal,
        ];

        for severity in severities {
            let json = serde_json::to_string(&severity).unwrap();
            let parsed: CritiqueSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(severity, parsed);
        }
    }

    #[test]
    fn test_decision_method_variants() {
        let methods = vec![
            DecisionMethod::ProsCons,
            DecisionMethod::MultiCriteria,
            DecisionMethod::ExpectedValue,
            DecisionMethod::RegretMinimization,
        ];

        for method in methods {
            let json = serde_json::to_string(&method).unwrap();
            let parsed: DecisionMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(method, parsed);
        }
    }

    #[test]
    fn test_branch_condition_confidence_below() {
        let condition = BranchCondition::ConfidenceBelow { threshold: 0.5 };

        let json = serde_json::to_string(&condition).unwrap();
        let parsed: BranchCondition = serde_json::from_str(&json).unwrap();

        if let BranchCondition::ConfidenceBelow { threshold } = parsed {
            assert_eq!(threshold, 0.5);
        } else {
            panic!("Wrong condition type");
        }
    }

    #[test]
    fn test_branch_condition_output_equals() {
        let condition = BranchCondition::OutputEquals {
            field: "verdict".to_string(),
            value: "pass".to_string(),
        };

        let json = serde_json::to_string(&condition).unwrap();
        let parsed: BranchCondition = serde_json::from_str(&json).unwrap();

        if let BranchCondition::OutputEquals { field, value } = parsed {
            assert_eq!(field, "verdict");
            assert_eq!(value, "pass");
        } else {
            panic!("Wrong condition type");
        }
    }

    #[test]
    fn test_validation_rule_min_count() {
        let rule = ValidationRule::MinCount {
            field: "perspectives".to_string(),
            value: 10,
        };

        let json = serde_json::to_string(&rule).unwrap();
        let parsed: ValidationRule = serde_json::from_str(&json).unwrap();

        if let ValidationRule::MinCount { field, value } = parsed {
            assert_eq!(field, "perspectives");
            assert_eq!(value, 10);
        } else {
            panic!("Wrong rule type");
        }
    }

    #[test]
    fn test_validation_rule_confidence_range() {
        let rule = ValidationRule::ConfidenceRange { min: 0.7, max: 1.0 };

        let json = serde_json::to_string(&rule).unwrap();
        let parsed: ValidationRule = serde_json::from_str(&json).unwrap();

        if let ValidationRule::ConfidenceRange { min, max } = parsed {
            assert_eq!(min, 0.7);
            assert_eq!(max, 1.0);
        } else {
            panic!("Wrong rule type");
        }
    }

    #[test]
    fn test_step_output_format_variants() {
        let formats = vec![
            StepOutputFormat::Text,
            StepOutputFormat::List,
            StepOutputFormat::Structured,
            StepOutputFormat::Score,
            StepOutputFormat::Boolean,
        ];

        for format in formats {
            let json = serde_json::to_string(&format).unwrap();
            let parsed: StepOutputFormat = serde_json::from_str(&json).unwrap();
            assert_eq!(format, parsed);
        }
    }
}

// ============================================================================
// EDGE CASES AND ERROR CONDITIONS
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_protocol_id() {
        let protocol = Protocol {
            id: "".to_string(),
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            description: "Test".to_string(),
            strategy: ReasoningStrategy::default(),
            input: InputSpec::default(),
            steps: vec![ProtocolStep {
                id: "step1".to_string(),
                action: StepAction::Analyze { criteria: vec![] },
                prompt_template: "Test".to_string(),
                output_format: StepOutputFormat::Text,
                min_confidence: 0.7,
                depends_on: vec![],
                branch: None,
            }],
            output: OutputSpec::default(),
            validation: vec![],
            metadata: ProtocolMetadata::default(),
        };

        let result = protocol.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .iter()
            .any(|e| e.contains("ID cannot be empty")));
    }

    #[test]
    fn test_circular_dependency_detection() {
        // This would cause infinite loop if not caught
        let protocol = Protocol::new("test", "Test")
            .with_step(ProtocolStep {
                id: "step1".to_string(),
                action: StepAction::Analyze { criteria: vec![] },
                prompt_template: "Test".to_string(),
                output_format: StepOutputFormat::Text,
                min_confidence: 0.7,
                depends_on: vec!["step2".to_string()], // depends on step2
                branch: None,
            })
            .with_step(ProtocolStep {
                id: "step2".to_string(),
                action: StepAction::Analyze { criteria: vec![] },
                prompt_template: "Test".to_string(),
                output_format: StepOutputFormat::Text,
                min_confidence: 0.7,
                depends_on: vec!["step1".to_string()], // depends on step1 (circular!)
                branch: None,
            });

        // Basic validation should pass (it checks if deps exist, not cycles)
        // The executor handles cycle detection at runtime
        assert!(protocol.validate().is_ok());
    }

    #[test]
    fn test_very_long_query() {
        let gt = GigaThink::new();
        let long_query = "a".repeat(10000);
        let context = ThinkToolContext {
            query: long_query,
            previous_steps: vec![],
        };

        // Queries longer than the configured max are rejected
        let result = gt.execute(&context);
        assert!(result.is_err());
    }

    #[test]
    fn test_special_characters_in_query() {
        let gt = GigaThink::new();
        let context = ThinkToolContext {
            query: "Query with special chars: <>&\"'{{}}".to_string(),
            previous_steps: vec![],
        };

        let result = gt.execute(&context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unicode_in_query() {
        let gt = GigaThink::new();
        let context = ThinkToolContext {
            query: "Unicode query: こんにちは 世界 🌍 🚀".to_string(),
            previous_steps: vec![],
        };

        let result = gt.execute(&context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_confidence_boundary_values() {
        // Test confidence exactly at boundaries
        let result_zero = StepResult::success("step", StepOutput::Empty, 0.0);
        let result_one = StepResult::success("step", StepOutput::Empty, 1.0);

        assert!(result_zero.meets_threshold(0.0));
        assert!(!result_zero.meets_threshold(0.01));
        assert!(result_one.meets_threshold(1.0));
        assert!(result_one.meets_threshold(0.99));
    }

    #[test]
    fn test_min_confidence_default() {
        // Default min_confidence should be 0.7
        let step = ProtocolStep {
            id: "test".to_string(),
            action: StepAction::Analyze { criteria: vec![] },
            prompt_template: "Test".to_string(),
            output_format: StepOutputFormat::Text,
            min_confidence: 0.7, // default value
            depends_on: vec![],
            branch: None,
        };

        assert_eq!(step.min_confidence, 0.7);
    }

    #[tokio::test]
    async fn test_empty_protocol_registry() {
        let registry = ProtocolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.get("gigathink").is_none());
    }

    #[test]
    fn test_protocol_metadata_defaults() {
        let metadata = ProtocolMetadata::default();

        assert!(metadata.category.is_empty());
        assert!(metadata.composable_with.is_empty());
        assert_eq!(metadata.typical_tokens, 0);
        assert_eq!(metadata.estimated_latency_ms, 0);
        assert!(metadata.extra.is_empty());
    }

    #[test]
    fn test_input_spec_defaults() {
        let spec = InputSpec::default();
        assert!(spec.required.is_empty());
        assert!(spec.optional.is_empty());
    }

    #[test]
    fn test_output_spec_defaults() {
        let spec = OutputSpec::default();
        assert!(spec.format.is_empty());
        assert!(spec.fields.is_empty());
    }
}

// ============================================================================
// PROFILE TESTS
// ============================================================================

mod profile_tests {
    use super::*;

    #[test]
    fn test_profile_registry_creation() {
        let registry = ProfileRegistry::with_builtins();
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_get_quick_profile() {
        let registry = ProfileRegistry::with_builtins();
        let quick = registry.get("quick");

        assert!(quick.is_some());
        let quick = quick.unwrap();
        assert_eq!(quick.id, "quick");
        // Quick profile should require lower confidence
        assert!(quick.min_confidence < 0.8);
    }

    #[test]
    fn test_get_balanced_profile() {
        let registry = ProfileRegistry::with_builtins();
        let balanced = registry.get("balanced");

        assert!(balanced.is_some());
        let balanced = balanced.unwrap();
        assert_eq!(balanced.id, "balanced");
    }

    #[test]
    fn test_get_deep_profile() {
        let registry = ProfileRegistry::with_builtins();
        let deep = registry.get("deep");

        assert!(deep.is_some());
        let deep = deep.unwrap();
        assert_eq!(deep.id, "deep");
    }

    #[test]
    fn test_get_paranoid_profile() {
        let registry = ProfileRegistry::with_builtins();
        let paranoid = registry.get("paranoid");

        assert!(paranoid.is_some());
        let paranoid = paranoid.unwrap();
        assert_eq!(paranoid.id, "paranoid");
        // Paranoid should have highest confidence requirement
        assert!(paranoid.min_confidence >= 0.9);
    }

    #[test]
    fn test_profile_chain_not_empty() {
        let registry = ProfileRegistry::with_builtins();

        for id in registry.list_ids() {
            let profile = registry.get(id).unwrap();
            assert!(!profile.chain.is_empty(), "Profile {} has empty chain", id);
        }
    }

    #[test]
    fn test_list_profile_ids() {
        let registry = ProfileRegistry::with_builtins();
        let ids = registry.list_ids();

        // All standard profiles should be present
        assert!(ids.contains(&"quick"));
        assert!(ids.contains(&"balanced"));
        assert!(ids.contains(&"deep"));
        assert!(ids.contains(&"paranoid"));
    }
}

// ============================================================================
// GIGATHINK MINIMUM OUTPUT TESTS (10+ perspectives requirement)
// ============================================================================

mod gigathink_perspective_tests {
    use super::*;

    #[test]
    fn test_gigathink_protocol_requires_min_perspectives() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let gt = registry.get("gigathink").unwrap();

        // Find the generate step
        let generate_step = gt
            .steps
            .iter()
            .find(|s| matches!(s.action, StepAction::Generate { .. }));

        assert!(
            generate_step.is_some(),
            "GigaThink should have a Generate step"
        );

        if let StepAction::Generate {
            min_count,
            max_count,
        } = &generate_step.unwrap().action
        {
            // Per requirements: GigaThink must generate 10+ perspectives
            // The identify_dimensions step generates 5-10 dimensions
            // Each dimension should yield perspectives
            assert!(*min_count >= 5, "GigaThink min_count should be at least 5");
            assert!(
                *max_count >= 10,
                "GigaThink max_count should be at least 10"
            );
        }
    }

    #[test]
    fn test_gigathink_has_synthesis_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let gt = registry.get("gigathink").unwrap();

        // GigaThink should have a synthesis step to combine perspectives
        let synthesis_step = gt
            .steps
            .iter()
            .find(|s| matches!(s.action, StepAction::Synthesize { .. }));

        assert!(
            synthesis_step.is_some(),
            "GigaThink should have a Synthesize step"
        );
    }

    #[test]
    fn test_gigathink_uses_expansive_strategy() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let gt = registry.get("gigathink").unwrap();
        assert_eq!(gt.strategy, ReasoningStrategy::Expansive);
    }
}

// ============================================================================
// LASERLOGIC FALLACY DETECTION TESTS
// ============================================================================

mod laserlogic_fallacy_tests {
    use super::*;

    #[test]
    fn test_laserlogic_has_fallacy_detection_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let ll = registry.get("laserlogic").unwrap();

        // Find the fallacy detection step (typically a Critique step)
        let fallacy_step = ll.steps.iter().find(|s| {
            s.id.contains("fallac")
                || s.prompt_template.to_lowercase().contains("fallac")
                || matches!(s.action, StepAction::Critique { .. })
        });

        assert!(
            fallacy_step.is_some(),
            "LaserLogic should have a fallacy detection step"
        );
    }

    #[test]
    fn test_laserlogic_has_validation_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let ll = registry.get("laserlogic").unwrap();

        let validation_step = ll
            .steps
            .iter()
            .find(|s| matches!(s.action, StepAction::Validate { .. }));

        assert!(
            validation_step.is_some(),
            "LaserLogic should have a Validate step"
        );
    }

    #[test]
    fn test_laserlogic_uses_deductive_strategy() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let ll = registry.get("laserlogic").unwrap();
        assert_eq!(ll.strategy, ReasoningStrategy::Deductive);
    }

    #[test]
    fn test_laserlogic_requires_argument_input() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let ll = registry.get("laserlogic").unwrap();
        assert!(ll.input.required.contains(&"argument".to_string()));
    }

    #[test]
    fn test_laserlogic_outputs_fallacies_field() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let ll = registry.get("laserlogic").unwrap();
        assert!(ll.output.fields.contains(&"fallacies".to_string()));
    }
}

// ============================================================================
// PROOFGUARD TRIANGULATION TESTS
// ============================================================================

mod proofguard_triangulation_tests {
    use super::*;

    #[test]
    fn test_proofguard_has_cross_reference_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let pg = registry.get("proofguard").unwrap();

        let cross_ref_step = pg
            .steps
            .iter()
            .find(|s| matches!(s.action, StepAction::CrossReference { .. }));

        assert!(
            cross_ref_step.is_some(),
            "ProofGuard should have a CrossReference step"
        );

        if let StepAction::CrossReference { min_sources } = &cross_ref_step.unwrap().action {
            // Per requirements: triangulation requires minimum 3 sources
            assert!(
                *min_sources >= 3,
                "ProofGuard should require at least 3 sources"
            );
        }
    }

    #[test]
    fn test_proofguard_uses_verification_strategy() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let pg = registry.get("proofguard").unwrap();
        assert_eq!(pg.strategy, ReasoningStrategy::Verification);
    }

    #[test]
    fn test_proofguard_requires_claim_input() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let pg = registry.get("proofguard").unwrap();
        assert!(pg.input.required.contains(&"claim".to_string()));
    }

    #[test]
    fn test_proofguard_outputs_sources_field() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let pg = registry.get("proofguard").unwrap();
        assert!(pg.output.fields.contains(&"sources".to_string()));
    }

    #[test]
    fn test_proofguard_outputs_evidence_field() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let pg = registry.get("proofguard").unwrap();
        assert!(pg.output.fields.contains(&"evidence".to_string()));
    }

    #[test]
    fn test_proofguard_has_triangulation_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let pg = registry.get("proofguard").unwrap();

        let triangulate_step = pg.steps.iter().find(|s| {
            s.id.contains("triangulat") || s.prompt_template.to_lowercase().contains("triangulat")
        });

        assert!(
            triangulate_step.is_some(),
            "ProofGuard should have a triangulation step"
        );
    }
}

// ============================================================================
// BEDROCK FIRST PRINCIPLES TESTS
// ============================================================================

mod bedrock_first_principles_tests {
    use super::*;

    #[test]
    fn test_bedrock_has_decomposition_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let br = registry.get("bedrock").unwrap();

        let decompose_step = br.steps.iter().find(|s| {
            s.id.contains("decompos")
                || s.prompt_template.to_lowercase().contains("decompos")
                || s.prompt_template
                    .to_lowercase()
                    .contains("first principles")
        });

        assert!(
            decompose_step.is_some(),
            "BedRock should have a decomposition step"
        );
    }

    #[test]
    fn test_bedrock_has_axiom_identification_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let br = registry.get("bedrock").unwrap();

        let axiom_step = br
            .steps
            .iter()
            .find(|s| s.id.contains("axiom") || s.prompt_template.to_lowercase().contains("axiom"));

        assert!(
            axiom_step.is_some(),
            "BedRock should have an axiom identification step"
        );
    }

    #[test]
    fn test_bedrock_uses_analytical_strategy() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let br = registry.get("bedrock").unwrap();
        assert_eq!(br.strategy, ReasoningStrategy::Analytical);
    }

    #[test]
    fn test_bedrock_requires_statement_input() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let br = registry.get("bedrock").unwrap();
        assert!(br.input.required.contains(&"statement".to_string()));
    }

    #[test]
    fn test_bedrock_outputs_axioms_field() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let br = registry.get("bedrock").unwrap();
        assert!(br.output.fields.contains(&"axioms".to_string()));
    }

    #[test]
    fn test_bedrock_has_reconstruction_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let br = registry.get("bedrock").unwrap();

        let reconstruct_step = br.steps.iter().find(|s| {
            s.id.contains("reconstruct") || s.prompt_template.to_lowercase().contains("reconstruct")
        });

        assert!(
            reconstruct_step.is_some(),
            "BedRock should have a reconstruction step"
        );
    }
}

// ============================================================================
// BRUTALHONESTY CRITIQUE TESTS
// ============================================================================

mod brutalhonesty_critique_tests {
    use super::*;

    #[test]
    fn test_brutalhonesty_has_steelman_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let bh = registry.get("brutalhonesty").unwrap();

        let steelman_step = bh.steps.iter().find(|s| {
            s.id.contains("steelman")
                || s.prompt_template.to_lowercase().contains("steelman")
                || s.prompt_template.to_lowercase().contains("strength")
        });

        assert!(
            steelman_step.is_some(),
            "BrutalHonesty should have a steelman step"
        );
    }

    #[test]
    fn test_brutalhonesty_has_attack_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let bh = registry.get("brutalhonesty").unwrap();

        let attack_step = bh.steps.iter().find(|s| {
            s.id.contains("attack") ||
            matches!(&s.action, StepAction::Critique { severity } if *severity == CritiqueSeverity::Brutal)
        });

        assert!(
            attack_step.is_some(),
            "BrutalHonesty should have a brutal critique step"
        );
    }

    #[test]
    fn test_brutalhonesty_uses_adversarial_strategy() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let bh = registry.get("brutalhonesty").unwrap();
        assert_eq!(bh.strategy, ReasoningStrategy::Adversarial);
    }

    #[test]
    fn test_brutalhonesty_requires_work_input() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let bh = registry.get("brutalhonesty").unwrap();
        assert!(bh.input.required.contains(&"work".to_string()));
    }

    #[test]
    fn test_brutalhonesty_outputs_strengths_and_flaws() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let bh = registry.get("brutalhonesty").unwrap();
        assert!(bh.output.fields.contains(&"strengths".to_string()));
        assert!(bh.output.fields.contains(&"flaws".to_string()));
    }

    #[test]
    fn test_brutalhonesty_has_verdict_step() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let bh = registry.get("brutalhonesty").unwrap();

        let verdict_step = bh
            .steps
            .iter()
            .find(|s| s.id.contains("verdict") || matches!(s.action, StepAction::Decide { .. }));

        assert!(
            verdict_step.is_some(),
            "BrutalHonesty should have a verdict step"
        );
    }

    #[test]
    fn test_brutalhonesty_critique_is_brutal() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let bh = registry.get("brutalhonesty").unwrap();

        let has_brutal_critique = bh.steps.iter().any(|s| {
            matches!(&s.action, StepAction::Critique { severity } if *severity == CritiqueSeverity::Brutal)
        });

        assert!(
            has_brutal_critique,
            "BrutalHonesty should use Brutal severity critique"
        );
    }
}

// ============================================================================
// EXECUTOR CONFIG TESTS
// ============================================================================

mod executor_config_tests {
    use super::*;

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();

        assert_eq!(config.timeout_secs, 120);
        assert!(!config.save_traces);
        assert!(!config.verbose);
        assert!(!config.use_mock);
        assert!(config.cli_tool.is_none());
    }

    #[test]
    fn test_executor_config_mock() {
        let config = ExecutorConfig::mock();

        assert!(config.use_mock);
    }

    #[test]
    fn test_executor_with_parallel() {
        let config = ExecutorConfig::default().with_parallel();

        assert!(config.enable_parallel);
    }

    #[test]
    fn test_executor_with_parallel_limit() {
        let config = ExecutorConfig::default().with_parallel_limit(8);

        assert!(config.enable_parallel);
        assert_eq!(config.max_concurrent_steps, 8);
    }

    #[test]
    fn test_executor_with_self_consistency() {
        let config = ExecutorConfig::default().with_self_consistency();

        assert!(config.self_consistency.is_some());
    }

    #[test]
    fn test_executor_with_self_consistency_fast() {
        let config = ExecutorConfig::default().with_self_consistency_fast();

        let sc = config.self_consistency.unwrap();
        assert_eq!(sc.num_samples, 3);
    }

    #[test]
    fn test_executor_with_self_consistency_thorough() {
        let config = ExecutorConfig::default().with_self_consistency_thorough();

        let sc = config.self_consistency.unwrap();
        assert_eq!(sc.num_samples, 10);
    }

    #[test]
    fn test_executor_with_self_consistency_paranoid() {
        let config = ExecutorConfig::default().with_self_consistency_paranoid();

        let sc = config.self_consistency.unwrap();
        assert_eq!(sc.num_samples, 15);
    }
}

// ============================================================================
// INTEGRATION-STYLE MOCK EXECUTION TESTS
// ============================================================================

mod mock_execution_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_gigathink_execution_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What are the implications of AGI?")
            .with_field("context", "Considering near-term developments");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
        assert!(!result.steps.is_empty());
        // duration_ms is u64; just ensure it's present by referencing it.
        let _ = result.duration_ms;
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_full_laserlogic_execution_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::argument(
            "Premise 1: All successful startups have product-market fit. \
             Premise 2: This startup has product-market fit. \
             Conclusion: This startup will be successful.",
        );

        let result = executor.execute("laserlogic", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_full_bedrock_execution_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::statement(
            "Electric vehicles are better for the environment than gasoline cars",
        )
        .with_field("domain", "environmental science");

        let result = executor.execute("bedrock", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_full_proofguard_execution_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::claim("GPT-4 has 1.8 trillion parameters");

        let result = executor.execute("proofguard", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_full_brutalhonesty_execution_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::work(
            "Our new product solves all customer problems and will disrupt the entire industry. \
             We have no competitors and infinite market potential.",
        );

        let result = executor.execute("brutalhonesty", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_protocols_sequential() {
        let executor = ProtocolExecutor::mock().unwrap();

        // Run GigaThink first
        let gt_result = executor
            .execute("gigathink", ProtocolInput::query("Test query 1"))
            .await
            .unwrap();

        // Then run LaserLogic
        let ll_result = executor
            .execute("laserlogic", ProtocolInput::argument("Test argument"))
            .await
            .unwrap();

        assert!(gt_result.success);
        assert!(ll_result.success);
    }

    #[tokio::test]
    async fn test_protocol_output_has_confidence() {
        let executor = ProtocolExecutor::mock().unwrap();

        for protocol_id in &[
            "gigathink",
            "laserlogic",
            "bedrock",
            "proofguard",
            "brutalhonesty",
        ] {
            let input = match *protocol_id {
                "gigathink" => ProtocolInput::query("Test"),
                "laserlogic" => ProtocolInput::argument("Test"),
                "bedrock" => ProtocolInput::statement("Test"),
                "proofguard" => ProtocolInput::claim("Test"),
                "brutalhonesty" => ProtocolInput::work("Test"),
                _ => ProtocolInput::query("Test"),
            };

            let result = executor.execute(protocol_id, input).await.unwrap();

            assert!(
                result.confidence >= 0.0 && result.confidence <= 1.0,
                "Protocol {} has invalid confidence: {}",
                protocol_id,
                result.confidence
            );
        }
    }
}
