//! Protocol definition types
//!
//! Defines the schema for ThinkTool protocols.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A ThinkTool Protocol definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Protocol {
    /// Unique protocol identifier (e.g., "gigathink", "laserlogic")
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Protocol version (semver)
    pub version: String,

    /// Brief description
    pub description: String,

    /// Reasoning strategy category
    pub strategy: ReasoningStrategy,

    /// Input specification
    pub input: InputSpec,

    /// Protocol steps (ordered)
    pub steps: Vec<ProtocolStep>,

    /// Output specification
    pub output: OutputSpec,

    /// Validation rules
    #[serde(default)]
    pub validation: Vec<ValidationRule>,

    /// Metadata for composition
    #[serde(default)]
    pub metadata: ProtocolMetadata,
}

/// Reasoning strategy categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ReasoningStrategy {
    /// Divergent thinking - maximize perspectives
    Expansive,
    /// Convergent thinking - deduce conclusions
    Deductive,
    /// Break down to fundamentals
    #[default]
    Analytical,
    /// Challenge and critique
    Adversarial,
    /// Cross-reference and confirm
    Verification,
    /// Weigh options systematically
    Decision,
    /// Scientific method
    Empirical,
}

/// Input specification for a protocol
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputSpec {
    /// Required input fields
    #[serde(default)]
    pub required: Vec<String>,

    /// Optional input fields
    #[serde(default)]
    pub optional: Vec<String>,
}

/// Output specification for a protocol
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputSpec {
    /// Output format name
    pub format: String,

    /// Output fields
    #[serde(default)]
    pub fields: Vec<String>,
}

/// A single step in a protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolStep {
    /// Step identifier within protocol
    pub id: String,

    /// What this step does
    pub action: StepAction,

    /// Prompt template (with {{placeholders}})
    pub prompt_template: String,

    /// Expected output format
    pub output_format: StepOutputFormat,

    /// Minimum confidence to proceed (0.0 - 1.0)
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,

    /// Dependencies on previous steps
    #[serde(default)]
    pub depends_on: Vec<String>,

    /// Optional branching conditions
    #[serde(default)]
    pub branch: Option<BranchCondition>,
}

fn default_min_confidence() -> f64 {
    0.7
}

/// Step action types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StepAction {
    /// Generate perspectives/ideas
    Generate {
        /// Minimum number of items to generate
        #[serde(default = "default_min_count")]
        min_count: usize,
        /// Maximum number of items to generate
        #[serde(default = "default_max_count")]
        max_count: usize,
    },

    /// Analyze/evaluate input
    Analyze {
        /// Criteria for analysis
        #[serde(default)]
        criteria: Vec<String>,
    },

    /// Synthesize multiple inputs
    Synthesize {
        /// Aggregation method to use
        #[serde(default)]
        aggregation: AggregationType,
    },

    /// Validate against rules
    Validate {
        /// Validation rules to apply
        #[serde(default)]
        rules: Vec<String>,
    },

    /// Challenge/critique
    Critique {
        /// Severity level for critique
        #[serde(default)]
        severity: CritiqueSeverity,
    },

    /// Make decision
    Decide {
        /// Decision method to use
        #[serde(default)]
        method: DecisionMethod,
    },

    /// Cross-reference sources
    CrossReference {
        /// Minimum number of sources required
        #[serde(default = "default_min_sources")]
        min_sources: usize,
    },
}

fn default_min_count() -> usize {
    3
}

fn default_max_count() -> usize {
    10
}

fn default_min_sources() -> usize {
    3
}

/// Output format for a step
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepOutputFormat {
    /// Free-form text
    #[default]
    Text,
    /// Numbered/bulleted list
    List,
    /// Key-value structured data
    Structured,
    /// Numeric score (0.0 - 1.0)
    Score,
    /// Boolean decision
    Boolean,
}

/// Aggregation types for synthesis
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationType {
    /// Group by themes
    #[default]
    ThematicClustering,
    /// Simple concatenation
    Concatenate,
    /// Weighted by confidence
    WeightedMerge,
    /// Majority voting
    Consensus,
}

/// Severity levels for critique
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CritiqueSeverity {
    /// Light review
    Light,
    /// Standard critique
    #[default]
    Standard,
    /// Adversarial challenge
    Adversarial,
    /// Maximum scrutiny
    Brutal,
}

/// Methods for decision making
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMethod {
    /// Simple pros/cons
    #[default]
    ProsCons,
    /// Multi-criteria analysis
    MultiCriteria,
    /// Expected value calculation
    ExpectedValue,
    /// Regret minimization
    RegretMinimization,
}

/// Conditional branching
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BranchCondition {
    /// Branch if confidence below threshold
    ConfidenceBelow {
        /// Confidence threshold value
        threshold: f64,
    },
    /// Branch if confidence above threshold
    ConfidenceAbove {
        /// Confidence threshold value
        threshold: f64,
    },
    /// Branch based on output value
    OutputEquals {
        /// Field name to check
        field: String,
        /// Expected value
        value: String,
    },
    /// Always execute (unconditional)
    Always,
}

/// Validation rule for protocol output
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum ValidationRule {
    /// Minimum number of items
    MinCount {
        /// Field name to validate
        field: String,
        /// Minimum count value
        value: usize,
    },
    /// Maximum number of items
    MaxCount {
        /// Field name to validate
        field: String,
        /// Maximum count value
        value: usize,
    },
    /// Confidence must be in range
    ConfidenceRange {
        /// Minimum confidence value
        min: f64,
        /// Maximum confidence value
        max: f64,
    },
    /// Field must be present
    Required {
        /// Required field name
        field: String,
    },
    /// Custom validation (expression)
    Custom {
        /// Validation expression
        expression: String,
    },
}

/// Protocol metadata for composition and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProtocolMetadata {
    /// Category tag
    #[serde(default)]
    pub category: String,

    /// Protocols this can be composed with
    #[serde(default)]
    pub composable_with: Vec<String>,

    /// Typical token usage
    #[serde(default)]
    pub typical_tokens: u32,

    /// Estimated latency in milliseconds
    #[serde(default)]
    pub estimated_latency_ms: u32,

    /// Additional key-value metadata
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Protocol {
    /// Create a new protocol with required fields
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            version: "1.0.0".to_string(),
            description: String::new(),
            strategy: ReasoningStrategy::default(),
            input: InputSpec::default(),
            steps: Vec::new(),
            output: OutputSpec::default(),
            validation: Vec::new(),
            metadata: ProtocolMetadata::default(),
        }
    }

    /// Add a step to the protocol
    pub fn with_step(mut self, step: ProtocolStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Set the reasoning strategy
    pub fn with_strategy(mut self, strategy: ReasoningStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Validate protocol definition
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.id.is_empty() {
            errors.push("Protocol ID cannot be empty".to_string());
        }

        if self.steps.is_empty() {
            errors.push("Protocol must have at least one step".to_string());
        }

        // Check step dependencies
        let step_ids: Vec<&str> = self.steps.iter().map(|s| s.id.as_str()).collect();
        for step in &self.steps {
            for dep in &step.depends_on {
                if !step_ids.contains(&dep.as_str()) {
                    errors.push(format!(
                        "Step '{}' depends on unknown step '{}'",
                        step.id, dep
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_creation() {
        let protocol = Protocol::new("test", "Test Protocol")
            .with_strategy(ReasoningStrategy::Expansive)
            .with_step(ProtocolStep {
                id: "step1".to_string(),
                action: StepAction::Generate {
                    min_count: 5,
                    max_count: 10,
                },
                prompt_template: "Generate ideas for: {{query}}".to_string(),
                output_format: StepOutputFormat::List,
                min_confidence: 0.7,
                depends_on: Vec::new(),
                branch: None,
            });

        assert_eq!(protocol.id, "test");
        assert_eq!(protocol.steps.len(), 1);
        assert!(protocol.validate().is_ok());
    }

    #[test]
    fn test_protocol_validation_empty_steps() {
        let protocol = Protocol::new("test", "Test Protocol");
        let result = protocol.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .iter()
            .any(|e| e.contains("at least one step")));
    }

    #[test]
    fn test_step_action_serialization() {
        let action = StepAction::Generate {
            min_count: 5,
            max_count: 10,
        };
        let json = serde_json::to_string(&action).unwrap();
        assert!(json.contains("generate"));

        let parsed: StepAction = serde_json::from_str(&json).unwrap();
        match parsed {
            StepAction::Generate {
                min_count,
                max_count,
            } => {
                assert_eq!(min_count, 5);
                assert_eq!(max_count, 10);
            }
            _ => panic!("Wrong action type"),
        }
    }
}
