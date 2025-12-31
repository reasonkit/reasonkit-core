//! Execution trace types for auditability
//!
//! Every protocol execution is traced step-by-step for:
//! - Debugging: Identify where reasoning went wrong
//! - Auditability: Know exactly how conclusions were reached
//! - Reproducibility: Re-run with same inputs
//! - Learning: Improve protocols based on traces

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::step::{StepOutput, TokenUsage};

/// A complete execution trace
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionTrace {
    /// Unique trace identifier
    pub id: Uuid,

    /// Protocol that was executed
    pub protocol_id: String,

    /// Protocol version
    pub protocol_version: String,

    /// Input provided to the protocol
    pub input: serde_json::Value,

    /// Step-by-step execution record
    pub steps: Vec<StepTrace>,

    /// Final output (if completed)
    pub output: Option<serde_json::Value>,

    /// Overall execution status
    pub status: ExecutionStatus,

    /// Timing information
    pub timing: TimingInfo,

    /// Total token usage
    pub tokens: TokenUsage,

    /// Overall confidence
    pub confidence: f64,

    /// Execution metadata
    pub metadata: TraceMetadata,
}

/// Trace of a single step execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepTrace {
    /// Step identifier
    pub step_id: String,

    /// Step index (0-based)
    pub index: usize,

    /// Actual prompt sent to LLM
    pub prompt: String,

    /// Raw LLM response
    pub raw_response: String,

    /// Parsed/structured output
    pub parsed_output: StepOutput,

    /// Step confidence score
    pub confidence: f64,

    /// Step execution time in milliseconds
    pub duration_ms: u64,

    /// Tokens used for this step
    pub tokens: TokenUsage,

    /// Step status
    pub status: StepStatus,

    /// Error message (if failed)
    pub error: Option<String>,

    /// Timestamp when step started
    pub started_at: DateTime<Utc>,

    /// Timestamp when step completed
    pub completed_at: Option<DateTime<Utc>>,
}

/// Overall execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ExecutionStatus {
    /// Execution is running
    #[default]
    Running,
    /// All steps completed successfully
    Completed,
    /// Execution failed
    Failed,
    /// Execution was cancelled
    Cancelled,
    /// Execution timed out
    TimedOut,
    /// Paused (can be resumed)
    Paused,
}

/// Individual step status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum StepStatus {
    /// Waiting to execute
    #[default]
    Pending,
    /// Currently executing
    Running,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed,
    /// Skipped (condition not met)
    Skipped,
}

/// Timing information for execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingInfo {
    /// When execution started
    pub started_at: Option<DateTime<Utc>>,

    /// When execution completed
    pub completed_at: Option<DateTime<Utc>>,

    /// Total duration in milliseconds
    pub total_duration_ms: u64,

    /// Time spent in LLM calls
    pub llm_duration_ms: u64,

    /// Time spent in local processing
    pub processing_duration_ms: u64,
}

impl TimingInfo {
    /// Start timing
    pub fn start(&mut self) {
        self.started_at = Some(Utc::now());
    }

    /// Complete timing
    pub fn complete(&mut self) {
        self.completed_at = Some(Utc::now());
        if let Some(start) = self.started_at {
            self.total_duration_ms = (Utc::now() - start).num_milliseconds() as u64;
        }
    }
}

/// Execution metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceMetadata {
    /// LLM model used
    pub model: Option<String>,

    /// LLM provider (openai, anthropic, etc.)
    pub provider: Option<String>,

    /// Temperature setting
    pub temperature: Option<f64>,

    /// Profile used (if any)
    pub profile: Option<String>,

    /// User-provided tags
    #[serde(default)]
    pub tags: Vec<String>,

    /// Environment info
    pub environment: Option<String>,
}

impl ExecutionTrace {
    /// Create a new execution trace
    pub fn new(protocol_id: impl Into<String>, protocol_version: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            protocol_id: protocol_id.into(),
            protocol_version: protocol_version.into(),
            input: serde_json::Value::Null,
            steps: Vec::new(),
            output: None,
            status: ExecutionStatus::Running,
            timing: TimingInfo::default(),
            tokens: TokenUsage::default(),
            confidence: 0.0,
            metadata: TraceMetadata::default(),
        }
    }

    /// Set input
    pub fn with_input(mut self, input: serde_json::Value) -> Self {
        self.input = input;
        self
    }

    /// Add a step trace
    pub fn add_step(&mut self, step: StepTrace) {
        self.tokens.add(&step.tokens);
        self.steps.push(step);
    }

    /// Mark as completed
    pub fn complete(&mut self, output: serde_json::Value, confidence: f64) {
        self.output = Some(output);
        self.confidence = confidence;
        self.status = ExecutionStatus::Completed;
        self.timing.complete();
    }

    /// Mark as failed
    pub fn fail(&mut self, error: &str) {
        self.status = ExecutionStatus::Failed;
        self.timing.complete();
        // Add error to last step if exists
        if let Some(last) = self.steps.last_mut() {
            last.error = Some(error.to_string());
            last.status = StepStatus::Failed;
        }
    }

    /// Get completed step count
    pub fn completed_steps(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.status == StepStatus::Completed)
            .count()
    }

    /// Get average step confidence
    pub fn average_confidence(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.steps.iter().map(|s| s.confidence).sum();
        sum / self.steps.len() as f64
    }

    /// Export trace to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export trace to compact JSON
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

impl StepTrace {
    /// Create a new step trace
    pub fn new(step_id: impl Into<String>, index: usize) -> Self {
        Self {
            step_id: step_id.into(),
            index,
            prompt: String::new(),
            raw_response: String::new(),
            parsed_output: StepOutput::Empty,
            confidence: 0.0,
            duration_ms: 0,
            tokens: TokenUsage::default(),
            status: StepStatus::Pending,
            error: None,
            started_at: Utc::now(),
            completed_at: None,
        }
    }

    /// Mark step as running
    pub fn start(&mut self) {
        self.status = StepStatus::Running;
        self.started_at = Utc::now();
    }

    /// Mark step as completed
    pub fn complete(&mut self, output: StepOutput, confidence: f64) {
        self.status = StepStatus::Completed;
        self.parsed_output = output;
        self.confidence = confidence;
        self.completed_at = Some(Utc::now());
        self.duration_ms = (Utc::now() - self.started_at).num_milliseconds() as u64;
    }

    /// Mark step as failed
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = StepStatus::Failed;
        self.error = Some(error.into());
        self.completed_at = Some(Utc::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_trace_creation() {
        let trace = ExecutionTrace::new("gigathink", "1.0.0");

        assert_eq!(trace.protocol_id, "gigathink");
        assert_eq!(trace.status, ExecutionStatus::Running);
        assert!(trace.steps.is_empty());
    }

    #[test]
    fn test_trace_with_steps() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        let mut step = StepTrace::new("step1", 0);
        step.tokens = TokenUsage::new(100, 50, 0.001);
        step.complete(
            StepOutput::Text {
                content: "Hello".to_string(),
            },
            0.9,
        );

        trace.add_step(step);

        assert_eq!(trace.steps.len(), 1);
        assert_eq!(trace.tokens.total_tokens, 150);
        assert_eq!(trace.completed_steps(), 1);
    }

    #[test]
    fn test_average_confidence() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        for (i, conf) in [0.8, 0.9, 0.7].iter().enumerate() {
            let mut step = StepTrace::new(format!("step{}", i), i);
            step.confidence = *conf;
            step.status = StepStatus::Completed;
            trace.add_step(step);
        }

        assert!((trace.average_confidence() - 0.8).abs() < 0.001);
    }
}
