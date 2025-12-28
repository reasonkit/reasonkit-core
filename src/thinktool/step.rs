//! Step execution types and result handling

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of executing a single protocol step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Step identifier
    pub step_id: String,

    /// Whether step succeeded
    pub success: bool,

    /// Output data (format depends on step type)
    pub output: StepOutput,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,

    /// Execution time in milliseconds
    pub duration_ms: u64,

    /// Token usage
    pub tokens: TokenUsage,

    /// Error message if failed
    pub error: Option<String>,
}

/// Step output variants
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[derive(Default)]
pub enum StepOutput {
    /// Free-form text output
    Text {
        /// The text content
        content: String,
    },

    /// List of items
    List {
        /// The list items
        items: Vec<ListItem>,
    },

    /// Structured key-value output
    Structured {
        /// Key-value data map
        data: HashMap<String, serde_json::Value>,
    },

    /// Numeric score
    Score {
        /// The score value
        value: f64,
    },

    /// Boolean result
    Boolean {
        /// The boolean value
        value: bool,
        /// Optional reason for the boolean
        reason: Option<String>,
    },

    /// Empty (no output yet)
    #[default]
    Empty,
}

/// A single item in a list output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListItem {
    /// Item content
    pub content: String,

    /// Item-level confidence
    #[serde(default)]
    pub confidence: Option<f64>,

    /// Optional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ListItem {
    /// Create a simple list item
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            confidence: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a list item with confidence
    pub fn with_confidence(content: impl Into<String>, confidence: f64) -> Self {
        Self {
            content: content.into(),
            confidence: Some(confidence),
            metadata: HashMap::new(),
        }
    }
}

/// Token usage tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input/prompt tokens
    pub input_tokens: u32,

    /// Output/completion tokens
    pub output_tokens: u32,

    /// Total tokens
    pub total_tokens: u32,

    /// Estimated cost in USD
    pub cost_usd: f64,
}

impl TokenUsage {
    /// Create new token usage
    pub fn new(input: u32, output: u32, cost: f64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            cost_usd: cost,
        }
    }

    /// Add another token usage
    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.cost_usd += other.cost_usd;
    }
}

/// Output format hint for parsing
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// Raw text, no parsing
    #[default]
    Raw,
    /// Parse as JSON
    Json,
    /// Parse as numbered/bulleted list
    List,
    /// Parse as key: value pairs
    KeyValue,
    /// Parse as single numeric value
    Numeric,
}

impl StepResult {
    /// Create a successful step result
    pub fn success(step_id: impl Into<String>, output: StepOutput, confidence: f64) -> Self {
        Self {
            step_id: step_id.into(),
            success: true,
            output,
            confidence,
            duration_ms: 0,
            tokens: TokenUsage::default(),
            error: None,
        }
    }

    /// Create a failed step result
    pub fn failure(step_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            step_id: step_id.into(),
            success: false,
            output: StepOutput::Empty,
            confidence: 0.0,
            duration_ms: 0,
            tokens: TokenUsage::default(),
            error: Some(error.into()),
        }
    }

    /// Set duration
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Set token usage
    pub fn with_tokens(mut self, tokens: TokenUsage) -> Self {
        self.tokens = tokens;
        self
    }

    /// Check if confidence meets threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.success && self.confidence >= threshold
    }

    /// Extract text content if available
    pub fn as_text(&self) -> Option<&str> {
        match &self.output {
            StepOutput::Text { content } => Some(content),
            _ => None,
        }
    }

    /// Extract list items if available
    pub fn as_list(&self) -> Option<&[ListItem]> {
        match &self.output {
            StepOutput::List { items } => Some(items),
            _ => None,
        }
    }

    /// Extract score if available
    pub fn as_score(&self) -> Option<f64> {
        match &self.output {
            StepOutput::Score { value } => Some(*value),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(result.confidence, 0.85);
        assert_eq!(result.as_text(), Some("Hello world"));
    }

    #[test]
    fn test_step_result_failure() {
        let result = StepResult::failure("test_step", "Something went wrong");

        assert!(!result.success);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_token_usage_add() {
        let mut usage1 = TokenUsage::new(100, 50, 0.001);
        let usage2 = TokenUsage::new(200, 100, 0.002);

        usage1.add(&usage2);

        assert_eq!(usage1.input_tokens, 300);
        assert_eq!(usage1.output_tokens, 150);
        assert_eq!(usage1.total_tokens, 450);
    }

    #[test]
    fn test_list_item() {
        let item = ListItem::with_confidence("Test item", 0.9);
        assert_eq!(item.content, "Test item");
        assert_eq!(item.confidence, Some(0.9));
    }
}
