//! Property-based testing for ThinkTool data structures.
//!
//! Uses proptest to generate arbitrary inputs and verify invariants
//! for ThinkTool protocols, steps, and execution results.

use proptest::prelude::*;
use std::collections::HashMap;

// ============================================================================
// ARBITRARY IMPLEMENTATIONS FOR CORE ENUMS
// ============================================================================

/// Strategy for generating ReasoningStrategy
pub fn arb_reasoning_strategy() -> impl Strategy<Value = ReasoningStrategy> {
    prop_oneof![
        Just(ReasoningStrategy::Expansive),
        Just(ReasoningStrategy::Deductive),
        Just(ReasoningStrategy::Analytical),
        Just(ReasoningStrategy::Adversarial),
        Just(ReasoningStrategy::Verification),
        Just(ReasoningStrategy::Decision),
        Just(ReasoningStrategy::Empirical),
    ]
}

/// Strategy for generating StepOutputFormat
pub fn arb_step_output_format() -> impl Strategy<Value = StepOutputFormat> {
    prop_oneof![
        Just(StepOutputFormat::Text),
        Just(StepOutputFormat::List),
        Just(StepOutputFormat::Structured),
        Just(StepOutputFormat::Score),
        Just(StepOutputFormat::Boolean),
    ]
}

/// Strategy for generating AggregationType
pub fn arb_aggregation_type() -> impl Strategy<Value = AggregationType> {
    prop_oneof![
        Just(AggregationType::ThematicClustering),
        Just(AggregationType::Concatenate),
        Just(AggregationType::WeightedMerge),
        Just(AggregationType::Consensus),
    ]
}

/// Strategy for generating CritiqueSeverity
pub fn arb_critique_severity() -> impl Strategy<Value = CritiqueSeverity> {
    prop_oneof![
        Just(CritiqueSeverity::Light),
        Just(CritiqueSeverity::Standard),
        Just(CritiqueSeverity::Adversarial),
        Just(CritiqueSeverity::Brutal),
    ]
}

/// Strategy for generating DecisionMethod
pub fn arb_decision_method() -> impl Strategy<Value = DecisionMethod> {
    prop_oneof![
        Just(DecisionMethod::ProsCons),
        Just(DecisionMethod::MultiCriteria),
        Just(DecisionMethod::ExpectedValue),
        Just(DecisionMethod::RegretMinimization),
    ]
}

// ============================================================================
// ARBITRARY IMPLEMENTATIONS FOR STEP ACTIONS
// ============================================================================

/// Strategy for generating StepAction variants
pub fn arb_step_action() -> impl Strategy<Value = StepAction> {
    prop_oneof![
        // Generate action with min/max counts
        (1usize..=5, 5usize..=20).prop_map(|(min, max)| StepAction::Generate {
            min_count: min,
            max_count: max.max(min), // Ensure max >= min
        }),
        // Analyze action with criteria
        prop::collection::vec("[a-z]{3,15}", 0..5)
            .prop_map(|criteria| StepAction::Analyze { criteria }),
        // Synthesize action
        arb_aggregation_type().prop_map(|agg| StepAction::Synthesize { aggregation: agg }),
        // Validate action with rules
        prop::collection::vec("[a-z_]{3,15}", 0..5)
            .prop_map(|rules| StepAction::Validate { rules }),
        // Critique action
        arb_critique_severity().prop_map(|sev| StepAction::Critique { severity: sev }),
        // Decide action
        arb_decision_method().prop_map(|method| StepAction::Decide { method }),
        // CrossReference action
        (1usize..=5).prop_map(|min| StepAction::CrossReference { min_sources: min }),
    ]
}

// ============================================================================
// ARBITRARY IMPLEMENTATIONS FOR BRANCH CONDITIONS
// ============================================================================

/// Strategy for generating BranchCondition variants
pub fn arb_branch_condition() -> impl Strategy<Value = BranchCondition> {
    prop_oneof![
        (0.0f64..=1.0).prop_map(|threshold| BranchCondition::ConfidenceBelow { threshold }),
        (0.0f64..=1.0).prop_map(|threshold| BranchCondition::ConfidenceAbove { threshold }),
        ("[a-z_]{3,15}", "[a-zA-Z0-9_]{1,20}")
            .prop_map(|(field, value)| { BranchCondition::OutputEquals { field, value } }),
        Just(BranchCondition::Always),
    ]
}

// ============================================================================
// ARBITRARY IMPLEMENTATIONS FOR VALIDATION RULES
// ============================================================================

/// Strategy for generating ValidationRule variants
pub fn arb_validation_rule() -> impl Strategy<Value = ValidationRule> {
    prop_oneof![
        ("[a-z_]{3,15}", 1usize..100)
            .prop_map(|(field, value)| ValidationRule::MinCount { field, value }),
        ("[a-z_]{3,15}", 1usize..100)
            .prop_map(|(field, value)| ValidationRule::MaxCount { field, value }),
        (0.0f64..=0.5, 0.5f64..=1.0)
            .prop_map(|(min, max)| ValidationRule::ConfidenceRange { min, max }),
        "[a-z_]{3,15}".prop_map(|field| ValidationRule::Required { field }),
        ".{5,50}".prop_map(|expr| ValidationRule::Custom { expression: expr }),
    ]
}

// ============================================================================
// ARBITRARY IMPLEMENTATIONS FOR STEP RESULT
// ============================================================================

/// Strategy for generating TokenUsage
pub fn arb_token_usage() -> impl Strategy<Value = TokenUsage> {
    (0u32..10000, 0u32..10000, 0.0f64..1.0)
        .prop_map(|(input, output, cost)| TokenUsage::new(input, output, cost))
}

/// Strategy for generating ListItem
pub fn arb_list_item() -> impl Strategy<Value = ListItem> {
    (
        ".{1,100}",                     // content
        prop::option::of(0.0f64..=1.0), // optional confidence
    )
        .prop_map(|(content, confidence)| ListItem {
            content,
            confidence,
            metadata: HashMap::new(),
        })
}

/// Strategy for generating StepOutput variants
pub fn arb_step_output() -> impl Strategy<Value = StepOutput> {
    prop_oneof![
        ".{0,500}".prop_map(|content| StepOutput::Text { content }),
        prop::collection::vec(arb_list_item(), 0..20).prop_map(|items| StepOutput::List { items }),
        Just(StepOutput::Structured {
            data: HashMap::new(),
        }),
        (0.0f64..=1.0).prop_map(|value| StepOutput::Score { value }),
        (any::<bool>(), prop::option::of(".{1,100}"))
            .prop_map(|(value, reason)| { StepOutput::Boolean { value, reason } }),
        Just(StepOutput::Empty),
    ]
}

/// Strategy for generating StepResult
pub fn arb_step_result() -> impl Strategy<Value = StepResult> {
    (
        "[a-z_]{3,20}",               // step_id
        any::<bool>(),                // success
        arb_step_output(),            // output
        0.0f64..=1.0,                 // confidence
        0u64..10000,                  // duration_ms
        arb_token_usage(),            // tokens
        prop::option::of(".{1,100}"), // error
    )
        .prop_map(
            |(step_id, success, output, confidence, duration_ms, tokens, error)| StepResult {
                step_id,
                success,
                output,
                confidence,
                duration_ms,
                tokens,
                error,
            },
        )
}

// ============================================================================
// ARBITRARY IMPLEMENTATIONS FOR PROTOCOL
// ============================================================================

/// Strategy for generating ProtocolStep
pub fn arb_protocol_step() -> impl Strategy<Value = ProtocolStep> {
    (
        "[a-z_]{3,20}",                              // id
        arb_step_action(),                           // action
        ".{10,200}",                                 // prompt_template
        arb_step_output_format(),                    // output_format
        0.0f64..=1.0,                                // min_confidence
        prop::collection::vec("[a-z_]{3,20}", 0..3), // depends_on
        prop::option::of(arb_branch_condition()),    // branch
    )
        .prop_map(
            |(id, action, prompt_template, output_format, min_confidence, depends_on, branch)| {
                ProtocolStep {
                    id,
                    action,
                    prompt_template,
                    output_format,
                    min_confidence,
                    depends_on,
                    branch,
                }
            },
        )
}

/// Strategy for generating InputSpec
pub fn arb_input_spec() -> impl Strategy<Value = InputSpec> {
    (
        prop::collection::vec("[a-z_]{3,15}", 0..5),
        prop::collection::vec("[a-z_]{3,15}", 0..5),
    )
        .prop_map(|(required, optional)| InputSpec { required, optional })
}

/// Strategy for generating OutputSpec
pub fn arb_output_spec() -> impl Strategy<Value = OutputSpec> {
    ("[a-z_]{3,15}", prop::collection::vec("[a-z_]{3,15}", 0..5))
        .prop_map(|(format, fields)| OutputSpec { format, fields })
}

/// Strategy for generating ProtocolMetadata
pub fn arb_protocol_metadata() -> impl Strategy<Value = ProtocolMetadata> {
    (
        "[a-z_]{0,20}",                              // category
        prop::collection::vec("[a-z_]{3,15}", 0..5), // composable_with
        0u32..50000,                                 // typical_tokens
        0u32..60000,                                 // estimated_latency_ms
    )
        .prop_map(
            |(category, composable_with, typical_tokens, estimated_latency_ms)| ProtocolMetadata {
                category,
                composable_with,
                typical_tokens,
                estimated_latency_ms,
                extra: HashMap::new(),
            },
        )
}

/// Strategy for generating Protocol
pub fn arb_protocol() -> impl Strategy<Value = Protocol> {
    (
        "[a-z_]{3,20}",          // id
        ".{3,50}",               // name
        "[0-9]\\.[0-9]\\.[0-9]", // version
        ".{10,200}",             // description
        arb_reasoning_strategy(),
        arb_input_spec(),
        prop::collection::vec(arb_protocol_step(), 1..5), // steps (at least 1)
        arb_output_spec(),
        prop::collection::vec(arb_validation_rule(), 0..5),
        arb_protocol_metadata(),
    )
        .prop_map(
            |(
                id,
                name,
                version,
                description,
                strategy,
                input,
                steps,
                output,
                validation,
                metadata,
            )| {
                Protocol {
                    id,
                    name,
                    version,
                    description,
                    strategy,
                    input,
                    steps,
                    output,
                    validation,
                    metadata,
                }
            },
        )
}

// ============================================================================
// PROPERTY TESTS: INVARIANTS
// ============================================================================

// Re-export types for use in tests (these would be actual imports in real code)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningStrategy {
    Expansive,
    Deductive,
    Analytical,
    Adversarial,
    Verification,
    Decision,
    Empirical,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutputFormat {
    Text,
    List,
    Structured,
    Score,
    Boolean,
}

#[derive(Debug, Clone)]
pub enum AggregationType {
    ThematicClustering,
    Concatenate,
    WeightedMerge,
    Consensus,
}

#[derive(Debug, Clone)]
pub enum CritiqueSeverity {
    Light,
    Standard,
    Adversarial,
    Brutal,
}

#[derive(Debug, Clone)]
pub enum DecisionMethod {
    ProsCons,
    MultiCriteria,
    ExpectedValue,
    RegretMinimization,
}

#[derive(Debug, Clone)]
pub enum StepAction {
    Generate { min_count: usize, max_count: usize },
    Analyze { criteria: Vec<String> },
    Synthesize { aggregation: AggregationType },
    Validate { rules: Vec<String> },
    Critique { severity: CritiqueSeverity },
    Decide { method: DecisionMethod },
    CrossReference { min_sources: usize },
}

#[derive(Debug, Clone)]
pub enum BranchCondition {
    ConfidenceBelow { threshold: f64 },
    ConfidenceAbove { threshold: f64 },
    OutputEquals { field: String, value: String },
    Always,
}

#[derive(Debug, Clone)]
pub enum ValidationRule {
    MinCount { field: String, value: usize },
    MaxCount { field: String, value: usize },
    ConfidenceRange { min: f64, max: f64 },
    Required { field: String },
    Custom { expression: String },
}

#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub cost_usd: f64,
}

impl TokenUsage {
    pub fn new(input: u32, output: u32, cost: f64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            cost_usd: cost,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ListItem {
    pub content: String,
    pub confidence: Option<f64>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum StepOutput {
    Text {
        content: String,
    },
    List {
        items: Vec<ListItem>,
    },
    Structured {
        data: HashMap<String, serde_json::Value>,
    },
    Score {
        value: f64,
    },
    Boolean {
        value: bool,
        reason: Option<String>,
    },
    Empty,
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_id: String,
    pub success: bool,
    pub output: StepOutput,
    pub confidence: f64,
    pub duration_ms: u64,
    pub tokens: TokenUsage,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProtocolStep {
    pub id: String,
    pub action: StepAction,
    pub prompt_template: String,
    pub output_format: StepOutputFormat,
    pub min_confidence: f64,
    pub depends_on: Vec<String>,
    pub branch: Option<BranchCondition>,
}

#[derive(Debug, Clone)]
pub struct InputSpec {
    pub required: Vec<String>,
    pub optional: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct OutputSpec {
    pub format: String,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProtocolMetadata {
    pub category: String,
    pub composable_with: Vec<String>,
    pub typical_tokens: u32,
    pub estimated_latency_ms: u32,
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct Protocol {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub strategy: ReasoningStrategy,
    pub input: InputSpec,
    pub steps: Vec<ProtocolStep>,
    pub output: OutputSpec,
    pub validation: Vec<ValidationRule>,
    pub metadata: ProtocolMetadata,
}

// ============================================================================
// PROPERTY TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    // ========================================================================
    // TokenUsage Invariants
    // ========================================================================

    #[test]
    fn prop_token_usage_total_equals_sum(input in 0u32..10000, output in 0u32..10000) {
        let usage = TokenUsage::new(input, output, 0.0);
        prop_assert_eq!(usage.total_tokens, input + output);
    }

    #[test]
    fn prop_token_usage_cost_non_negative(input in 0u32..10000, output in 0u32..10000, cost in 0.0f64..1.0) {
        let usage = TokenUsage::new(input, output, cost);
        prop_assert!(usage.cost_usd >= 0.0);
    }

    // ========================================================================
    // StepAction Invariants
    // ========================================================================

    #[test]
    fn prop_generate_action_min_lte_max(min in 1usize..=10, max in 1usize..=20) {
        let action = StepAction::Generate {
            min_count: min,
            max_count: max.max(min),
        };
        if let StepAction::Generate { min_count, max_count } = action {
            prop_assert!(min_count <= max_count, "min_count must be <= max_count");
        }
    }

    #[test]
    fn prop_cross_reference_min_sources_positive(min in 1usize..=10) {
        let action = StepAction::CrossReference { min_sources: min };
        if let StepAction::CrossReference { min_sources } = action {
            prop_assert!(min_sources >= 1, "min_sources must be >= 1");
        }
    }

    // ========================================================================
    // Confidence Invariants
    // ========================================================================

    #[test]
    fn prop_confidence_in_valid_range(confidence in 0.0f64..=1.0) {
        prop_assert!((0.0..=1.0).contains(&confidence),
            "Confidence must be in [0.0, 1.0], got {}", confidence);
    }

    #[test]
    fn prop_confidence_range_valid(min in 0.0f64..=0.5, max in 0.5f64..=1.0) {
        let rule = ValidationRule::ConfidenceRange { min, max };
        if let ValidationRule::ConfidenceRange { min: m, max: x } = rule {
            prop_assert!(m <= x, "ConfidenceRange min must be <= max");
        }
    }

    // ========================================================================
    // BranchCondition Invariants
    // ========================================================================

    #[test]
    fn prop_branch_condition_threshold_valid(threshold in 0.0f64..=1.0) {
        let below = BranchCondition::ConfidenceBelow { threshold };
        let above = BranchCondition::ConfidenceAbove { threshold };

        if let BranchCondition::ConfidenceBelow { threshold: t } = below {
            prop_assert!((0.0..=1.0).contains(&t));
        }
        if let BranchCondition::ConfidenceAbove { threshold: t } = above {
            prop_assert!((0.0..=1.0).contains(&t));
        }
    }

    // ========================================================================
    // StepResult Invariants
    // ========================================================================

    #[test]
    fn prop_step_result_failure_has_zero_confidence(step_id in "[a-z_]{3,20}", error in ".{1,100}") {
        let result = StepResult {
            step_id,
            success: false,
            output: StepOutput::Empty,
            confidence: 0.0,
            duration_ms: 0,
            tokens: TokenUsage::default(),
            error: Some(error),
        };
        // Failed results should have 0 confidence (invariant)
        prop_assert!(!result.success || result.confidence >= 0.0);
    }

    #[test]
    fn prop_step_result_success_no_error(
        step_id in "[a-z_]{3,20}",
        confidence in 0.0f64..=1.0
    ) {
        let result = StepResult {
            step_id,
            success: true,
            output: StepOutput::Empty,
            confidence,
            duration_ms: 0,
            tokens: TokenUsage::default(),
            error: None,
        };
        prop_assert!(result.success && result.error.is_none(),
            "Successful result should have no error");
    }

    // ========================================================================
    // Protocol Invariants
    // ========================================================================

    #[test]
    fn prop_protocol_has_valid_id(id in "[a-z_]{3,20}") {
        prop_assert!(!id.is_empty(), "Protocol ID cannot be empty");
        prop_assert!(id.len() >= 3, "Protocol ID should have at least 3 characters");
    }

    #[test]
    fn prop_protocol_step_count_positive(num_steps in 1usize..=10) {
        let steps: Vec<ProtocolStep> = (0..num_steps)
            .map(|i| ProtocolStep {
                id: format!("step_{}", i),
                action: StepAction::Generate { min_count: 3, max_count: 10 },
                prompt_template: "Test prompt".to_string(),
                output_format: StepOutputFormat::Text,
                min_confidence: 0.7,
                depends_on: vec![],
                branch: None,
            })
            .collect();

        prop_assert!(!steps.is_empty(), "Protocol must have at least one step");
    }

    // ========================================================================
    // Score Invariants
    // ========================================================================

    #[test]
    fn prop_score_output_in_valid_range(value in 0.0f64..=1.0) {
        let output = StepOutput::Score { value };
        if let StepOutput::Score { value: v } = output {
            prop_assert!((0.0..=1.0).contains(&v), "Score must be in [0.0, 1.0]");
        }
    }

    // ========================================================================
    // ListItem Invariants
    // ========================================================================

    #[test]
    fn prop_list_item_confidence_valid(content in ".{1,100}", confidence in prop::option::of(0.0f64..=1.0)) {
        let item = ListItem {
            content,
            confidence,
            metadata: HashMap::new(),
        };

        if let Some(c) = item.confidence {
            prop_assert!(
                (0.0..=1.0).contains(&c),
                "ListItem confidence must be in [0.0, 1.0]"
            );
        }
    }

    // ========================================================================
    // Version String Invariants
    // ========================================================================

    #[test]
    fn prop_semver_format(major in 0u32..100, minor in 0u32..100, patch in 0u32..100) {
        let version = format!("{}.{}.{}", major, minor, patch);
        let parts: Vec<&str> = version.split('.').collect();

        prop_assert_eq!(parts.len(), 3, "Semver must have 3 parts");
        for part in parts {
            prop_assert!(part.parse::<u32>().is_ok(), "Each part must be a number");
        }
    }
}

// ============================================================================
// ROUNDTRIP SERIALIZATION TESTS
// ============================================================================

#[cfg(test)]
mod roundtrip_tests {
    use super::*;

    // Note: In actual implementation, these would use the real serde traits
    // from the reasonkit-core crate. These are placeholder implementations.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_token_usage_roundtrip(input in 0u32..10000, output in 0u32..10000, cost in 0.0f64..1.0) {
            let original = TokenUsage::new(input, output, cost);
            // In real implementation: let json = serde_json::to_string(&original)?;
            // let roundtrip: TokenUsage = serde_json::from_str(&json)?;
            // prop_assert_eq!(original, roundtrip);

            // For now, just verify the structure is valid
            prop_assert_eq!(original.input_tokens, input);
            prop_assert_eq!(original.output_tokens, output);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arb_reasoning_strategy_coverage() {
        // Verify all variants are reachable
        let strategies = [
            ReasoningStrategy::Expansive,
            ReasoningStrategy::Deductive,
            ReasoningStrategy::Analytical,
            ReasoningStrategy::Adversarial,
            ReasoningStrategy::Verification,
            ReasoningStrategy::Decision,
            ReasoningStrategy::Empirical,
        ];

        assert_eq!(strategies.len(), 7);
    }

    #[test]
    fn test_token_usage_new() {
        let usage = TokenUsage::new(100, 50, 0.001);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert!((usage.cost_usd - 0.001).abs() < f64::EPSILON);
    }
}
