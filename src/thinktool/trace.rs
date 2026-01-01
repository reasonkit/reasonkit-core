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
    use serde_json::json;
    use std::collections::HashMap;

    // =========================================================================
    // SECTION 1: ExecutionTrace Creation Tests
    // =========================================================================

    #[test]
    fn test_execution_trace_creation() {
        let trace = ExecutionTrace::new("gigathink", "1.0.0");

        assert_eq!(trace.protocol_id, "gigathink");
        assert_eq!(trace.protocol_version, "1.0.0");
        assert_eq!(trace.status, ExecutionStatus::Running);
        assert!(trace.steps.is_empty());
        assert!(trace.output.is_none());
        assert_eq!(trace.confidence, 0.0);
        assert_eq!(trace.tokens.total_tokens, 0);
    }

    #[test]
    fn test_execution_trace_unique_id() {
        let trace1 = ExecutionTrace::new("test", "1.0.0");
        let trace2 = ExecutionTrace::new("test", "1.0.0");

        // Each trace should have a unique UUID
        assert_ne!(trace1.id, trace2.id);
    }

    #[test]
    fn test_execution_trace_default() {
        let trace = ExecutionTrace::default();

        assert_eq!(trace.protocol_id, "");
        assert_eq!(trace.protocol_version, "");
        assert_eq!(trace.status, ExecutionStatus::Running);
        assert!(trace.steps.is_empty());
    }

    #[test]
    fn test_execution_trace_with_input() {
        let input = json!({
            "query": "What is Rust?",
            "context": ["systems programming", "memory safety"]
        });

        let trace = ExecutionTrace::new("laserlogic", "2.0.0").with_input(input.clone());

        assert_eq!(trace.input, input);
        assert_eq!(trace.input["query"], "What is Rust?");
    }

    #[test]
    fn test_execution_trace_with_complex_input() {
        let input = json!({
            "nested": {
                "deeply": {
                    "value": 42,
                    "array": [1, 2, 3]
                }
            },
            "boolean": true,
            "null_value": null
        });

        let trace = ExecutionTrace::new("test", "1.0.0").with_input(input.clone());

        assert_eq!(trace.input["nested"]["deeply"]["value"], 42);
        assert_eq!(trace.input["boolean"], true);
        assert!(trace.input["null_value"].is_null());
    }

    // =========================================================================
    // SECTION 2: StepTrace Creation and Lifecycle Tests
    // =========================================================================

    #[test]
    fn test_step_trace_creation() {
        let step = StepTrace::new("analyze", 0);

        assert_eq!(step.step_id, "analyze");
        assert_eq!(step.index, 0);
        assert_eq!(step.status, StepStatus::Pending);
        assert!(step.prompt.is_empty());
        assert!(step.raw_response.is_empty());
        assert!(step.error.is_none());
        assert!(step.completed_at.is_none());
    }

    #[test]
    fn test_step_trace_start() {
        let mut step = StepTrace::new("step1", 0);
        let before_start = Utc::now();

        step.start();

        assert_eq!(step.status, StepStatus::Running);
        assert!(step.started_at >= before_start);
    }

    #[test]
    fn test_step_trace_complete() {
        let mut step = StepTrace::new("step1", 0);
        step.start();

        let output = StepOutput::Text {
            content: "Analysis complete".to_string(),
        };
        step.complete(output.clone(), 0.95);

        assert_eq!(step.status, StepStatus::Completed);
        assert_eq!(step.confidence, 0.95);
        assert!(step.completed_at.is_some());
        // Duration should be recorded
        assert!(step.duration_ms >= 0);

        // Verify output was stored
        if let StepOutput::Text { content } = &step.parsed_output {
            assert_eq!(content, "Analysis complete");
        } else {
            panic!("Expected Text output");
        }
    }

    #[test]
    fn test_step_trace_fail() {
        let mut step = StepTrace::new("step1", 0);
        step.start();

        step.fail("LLM timeout occurred");

        assert_eq!(step.status, StepStatus::Failed);
        assert_eq!(step.error, Some("LLM timeout occurred".to_string()));
        assert!(step.completed_at.is_some());
    }

    #[test]
    fn test_step_trace_with_prompt_and_response() {
        let mut step = StepTrace::new("reasoning", 1);
        step.prompt = "Analyze the following code for bugs...".to_string();
        step.raw_response = "I found 3 potential issues: 1) null pointer...".to_string();

        assert!(step.prompt.contains("Analyze"));
        assert!(step.raw_response.contains("3 potential issues"));
    }

    #[test]
    fn test_step_trace_with_tokens() {
        let mut step = StepTrace::new("step1", 0);
        step.tokens = TokenUsage::new(500, 200, 0.0035);

        assert_eq!(step.tokens.input_tokens, 500);
        assert_eq!(step.tokens.output_tokens, 200);
        assert_eq!(step.tokens.total_tokens, 700);
        assert!((step.tokens.cost_usd - 0.0035).abs() < 0.0001);
    }

    // =========================================================================
    // SECTION 3: Nested Trace Spans (Steps within Traces)
    // =========================================================================

    #[test]
    fn test_trace_with_single_step() {
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
    fn test_trace_with_multiple_steps() {
        let mut trace = ExecutionTrace::new("powercombo", "1.0.0");

        // Add 5 steps simulating a full ThinkTool pipeline
        let step_configs = vec![
            ("gigathink", 200, 150, 0.85),
            ("laserlogic", 180, 120, 0.90),
            ("bedrock", 250, 200, 0.88),
            ("proofguard", 300, 250, 0.92),
            ("brutalhonesty", 150, 100, 0.95),
        ];

        for (i, (name, input_tok, output_tok, conf)) in step_configs.iter().enumerate() {
            let mut step = StepTrace::new(*name, i);
            step.tokens = TokenUsage::new(*input_tok, *output_tok, 0.001);
            step.complete(
                StepOutput::Text {
                    content: format!("{} output", name),
                },
                *conf,
            );
            trace.add_step(step);
        }

        assert_eq!(trace.steps.len(), 5);
        assert_eq!(trace.completed_steps(), 5);

        // Verify token aggregation
        let expected_total: u32 = step_configs.iter().map(|(_, i, o, _)| i + o).sum();
        assert_eq!(trace.tokens.total_tokens, expected_total);
    }

    #[test]
    fn test_trace_with_mixed_step_statuses() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        // Step 1: Completed
        let mut step1 = StepTrace::new("step1", 0);
        step1.complete(StepOutput::Empty, 0.9);
        trace.add_step(step1);

        // Step 2: Failed
        let mut step2 = StepTrace::new("step2", 1);
        step2.fail("Validation error");
        trace.add_step(step2);

        // Step 3: Still pending
        let step3 = StepTrace::new("step3", 2);
        trace.add_step(step3);

        assert_eq!(trace.steps.len(), 3);
        assert_eq!(trace.completed_steps(), 1);
        assert_eq!(trace.steps[1].status, StepStatus::Failed);
        assert_eq!(trace.steps[2].status, StepStatus::Pending);
    }

    #[test]
    fn test_trace_step_ordering() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        for i in 0..10 {
            let step = StepTrace::new(format!("step_{}", i), i);
            trace.add_step(step);
        }

        // Verify steps maintain insertion order
        for (i, step) in trace.steps.iter().enumerate() {
            assert_eq!(step.index, i);
            assert_eq!(step.step_id, format!("step_{}", i));
        }
    }

    // =========================================================================
    // SECTION 4: Trace Metadata Tests
    // =========================================================================

    #[test]
    fn test_trace_metadata_default() {
        let metadata = TraceMetadata::default();

        assert!(metadata.model.is_none());
        assert!(metadata.provider.is_none());
        assert!(metadata.temperature.is_none());
        assert!(metadata.profile.is_none());
        assert!(metadata.tags.is_empty());
        assert!(metadata.environment.is_none());
    }

    #[test]
    fn test_trace_metadata_full() {
        let mut metadata = TraceMetadata::default();
        metadata.model = Some("claude-3-opus".to_string());
        metadata.provider = Some("anthropic".to_string());
        metadata.temperature = Some(0.7);
        metadata.profile = Some("paranoid".to_string());
        metadata.tags = vec!["production".to_string(), "critical".to_string()];
        metadata.environment = Some("aws-us-east-1".to_string());

        assert_eq!(metadata.model, Some("claude-3-opus".to_string()));
        assert_eq!(metadata.provider, Some("anthropic".to_string()));
        assert_eq!(metadata.temperature, Some(0.7));
        assert_eq!(metadata.profile, Some("paranoid".to_string()));
        assert_eq!(metadata.tags.len(), 2);
        assert!(metadata.tags.contains(&"production".to_string()));
    }

    #[test]
    fn test_trace_with_metadata() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");
        trace.metadata.model = Some("gpt-4".to_string());
        trace.metadata.provider = Some("openai".to_string());
        trace.metadata.temperature = Some(0.5);

        assert_eq!(trace.metadata.model, Some("gpt-4".to_string()));
        assert_eq!(trace.metadata.provider, Some("openai".to_string()));
    }

    // =========================================================================
    // SECTION 5: Timing Information Tests
    // =========================================================================

    #[test]
    fn test_timing_info_default() {
        let timing = TimingInfo::default();

        assert!(timing.started_at.is_none());
        assert!(timing.completed_at.is_none());
        assert_eq!(timing.total_duration_ms, 0);
        assert_eq!(timing.llm_duration_ms, 0);
        assert_eq!(timing.processing_duration_ms, 0);
    }

    #[test]
    fn test_timing_info_start() {
        let mut timing = TimingInfo::default();
        let before = Utc::now();

        timing.start();

        assert!(timing.started_at.is_some());
        assert!(timing.started_at.unwrap() >= before);
        assert!(timing.completed_at.is_none());
    }

    #[test]
    fn test_timing_info_complete() {
        let mut timing = TimingInfo::default();
        timing.start();

        // Small delay to ensure measurable duration
        std::thread::sleep(std::time::Duration::from_millis(10));

        timing.complete();

        assert!(timing.completed_at.is_some());
        assert!(timing.total_duration_ms >= 10);
        assert!(timing.completed_at.unwrap() > timing.started_at.unwrap());
    }

    #[test]
    fn test_timing_info_complete_without_start() {
        let mut timing = TimingInfo::default();

        // Complete without starting - should handle gracefully
        timing.complete();

        assert!(timing.completed_at.is_some());
        // Duration should remain 0 since there's no start time
        assert_eq!(timing.total_duration_ms, 0);
    }

    #[test]
    fn test_step_timing_captures_duration() {
        let mut step = StepTrace::new("timed_step", 0);
        step.start();

        std::thread::sleep(std::time::Duration::from_millis(15));

        step.complete(StepOutput::Empty, 0.9);

        // Duration should be at least 15ms
        assert!(step.duration_ms >= 15);
        assert!(step.completed_at.is_some());
    }

    // =========================================================================
    // SECTION 6: Average Confidence Tests
    // =========================================================================

    #[test]
    fn test_average_confidence_empty() {
        let trace = ExecutionTrace::new("test", "1.0.0");
        assert_eq!(trace.average_confidence(), 0.0);
    }

    #[test]
    fn test_average_confidence_single_step() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        let mut step = StepTrace::new("step1", 0);
        step.confidence = 0.85;
        trace.add_step(step);

        assert!((trace.average_confidence() - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_average_confidence_multiple_steps() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        for (i, conf) in [0.8, 0.9, 0.7].iter().enumerate() {
            let mut step = StepTrace::new(format!("step{}", i), i);
            step.confidence = *conf;
            step.status = StepStatus::Completed;
            trace.add_step(step);
        }

        // (0.8 + 0.9 + 0.7) / 3 = 0.8
        assert!((trace.average_confidence() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_average_confidence_includes_failed_steps() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        let mut step1 = StepTrace::new("step1", 0);
        step1.confidence = 0.9;
        step1.status = StepStatus::Completed;
        trace.add_step(step1);

        let mut step2 = StepTrace::new("step2", 1);
        step2.confidence = 0.0; // Failed step with 0 confidence
        step2.status = StepStatus::Failed;
        trace.add_step(step2);

        // Average includes both: (0.9 + 0.0) / 2 = 0.45
        assert!((trace.average_confidence() - 0.45).abs() < 0.001);
    }

    // =========================================================================
    // SECTION 7: Trace Completion and Failure Tests
    // =========================================================================

    #[test]
    fn test_trace_complete() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");
        trace.timing.start();

        let output = json!({"result": "success", "data": [1, 2, 3]});
        trace.complete(output.clone(), 0.92);

        assert_eq!(trace.status, ExecutionStatus::Completed);
        assert_eq!(trace.confidence, 0.92);
        assert!(trace.output.is_some());
        assert_eq!(trace.output.as_ref().unwrap()["result"], "success");
        assert!(trace.timing.completed_at.is_some());
    }

    #[test]
    fn test_trace_fail_with_steps() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");
        trace.timing.start();

        // Add a step
        let mut step = StepTrace::new("step1", 0);
        step.start();
        trace.add_step(step);

        // Fail the trace
        trace.fail("Connection timeout");

        assert_eq!(trace.status, ExecutionStatus::Failed);
        assert!(trace.timing.completed_at.is_some());

        // Last step should be marked as failed
        let last_step = trace.steps.last().unwrap();
        assert_eq!(last_step.status, StepStatus::Failed);
        assert_eq!(last_step.error, Some("Connection timeout".to_string()));
    }

    #[test]
    fn test_trace_fail_without_steps() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");
        trace.timing.start();

        // Fail without any steps - should not panic
        trace.fail("Early failure");

        assert_eq!(trace.status, ExecutionStatus::Failed);
        assert!(trace.steps.is_empty());
    }

    // =========================================================================
    // SECTION 8: JSON Export Tests
    // =========================================================================

    #[test]
    fn test_trace_to_json() {
        let trace =
            ExecutionTrace::new("test_protocol", "1.0.0").with_input(json!({"query": "test"}));

        let json_str = trace.to_json().expect("JSON serialization should succeed");

        assert!(json_str.contains("test_protocol"));
        assert!(json_str.contains("1.0.0"));
        assert!(json_str.contains("query"));
        // Pretty format should have newlines
        assert!(json_str.contains('\n'));
    }

    #[test]
    fn test_trace_to_json_compact() {
        let trace = ExecutionTrace::new("test", "1.0.0");

        let json_str = trace
            .to_json_compact()
            .expect("Compact JSON should succeed");

        // Compact format should not have pretty-print newlines (except in strings)
        let lines: Vec<&str> = json_str.lines().collect();
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn test_trace_json_roundtrip() {
        let mut original = ExecutionTrace::new("roundtrip", "2.0.0")
            .with_input(json!({"key": "value", "number": 42}));

        original.metadata.model = Some("test-model".to_string());
        original.metadata.tags = vec!["tag1".to_string(), "tag2".to_string()];

        let mut step = StepTrace::new("step1", 0);
        step.prompt = "Test prompt".to_string();
        step.raw_response = "Test response".to_string();
        step.tokens = TokenUsage::new(100, 50, 0.001);
        step.complete(
            StepOutput::Text {
                content: "Result".to_string(),
            },
            0.88,
        );
        original.add_step(step);

        original.complete(json!({"final": "output"}), 0.9);

        // Serialize
        let json_str = original.to_json().expect("Serialization should succeed");

        // Deserialize
        let deserialized: ExecutionTrace =
            serde_json::from_str(&json_str).expect("Deserialization should succeed");

        // Verify key fields
        assert_eq!(deserialized.protocol_id, original.protocol_id);
        assert_eq!(deserialized.protocol_version, original.protocol_version);
        assert_eq!(deserialized.id, original.id);
        assert_eq!(deserialized.status, ExecutionStatus::Completed);
        assert_eq!(deserialized.confidence, 0.9);
        assert_eq!(deserialized.steps.len(), 1);
        assert_eq!(deserialized.metadata.model, Some("test-model".to_string()));
        assert_eq!(deserialized.metadata.tags.len(), 2);
    }

    #[test]
    fn test_trace_json_with_all_step_outputs() {
        let mut trace = ExecutionTrace::new("output_types", "1.0.0");

        // Text output
        let mut step1 = StepTrace::new("text_step", 0);
        step1.complete(
            StepOutput::Text {
                content: "Hello world".to_string(),
            },
            0.9,
        );
        trace.add_step(step1);

        // List output
        let mut step2 = StepTrace::new("list_step", 1);
        step2.complete(
            StepOutput::List {
                items: vec![
                    super::super::step::ListItem::new("Item 1"),
                    super::super::step::ListItem::with_confidence("Item 2", 0.95),
                ],
            },
            0.85,
        );
        trace.add_step(step2);

        // Structured output
        let mut step3 = StepTrace::new("struct_step", 2);
        let mut data = HashMap::new();
        data.insert("key1".to_string(), json!("value1"));
        data.insert("key2".to_string(), json!(123));
        step3.complete(StepOutput::Structured { data }, 0.88);
        trace.add_step(step3);

        // Score output
        let mut step4 = StepTrace::new("score_step", 3);
        step4.complete(StepOutput::Score { value: 0.75 }, 0.92);
        trace.add_step(step4);

        // Boolean output
        let mut step5 = StepTrace::new("bool_step", 4);
        step5.complete(
            StepOutput::Boolean {
                value: true,
                reason: Some("Validation passed".to_string()),
            },
            0.99,
        );
        trace.add_step(step5);

        let json_str = trace.to_json().expect("Should serialize all output types");

        // Verify all types are present
        assert!(json_str.contains("Hello world"));
        assert!(json_str.contains("Item 1"));
        assert!(json_str.contains("key1"));
        assert!(json_str.contains("0.75"));
        assert!(json_str.contains("Validation passed"));

        // Verify roundtrip
        let deserialized: ExecutionTrace =
            serde_json::from_str(&json_str).expect("Should deserialize");
        assert_eq!(deserialized.steps.len(), 5);
    }

    // =========================================================================
    // SECTION 9: Execution Status Tests
    // =========================================================================

    #[test]
    fn test_execution_status_default() {
        assert_eq!(ExecutionStatus::default(), ExecutionStatus::Running);
    }

    #[test]
    fn test_execution_status_serialization() {
        let statuses = vec![
            (ExecutionStatus::Running, "\"running\""),
            (ExecutionStatus::Completed, "\"completed\""),
            (ExecutionStatus::Failed, "\"failed\""),
            (ExecutionStatus::Cancelled, "\"cancelled\""),
            (ExecutionStatus::TimedOut, "\"timed_out\""),
            (ExecutionStatus::Paused, "\"paused\""),
        ];

        for (status, expected) in statuses {
            let json = serde_json::to_string(&status).expect("Should serialize");
            assert_eq!(json, expected);

            let deserialized: ExecutionStatus =
                serde_json::from_str(&json).expect("Should deserialize");
            assert_eq!(deserialized, status);
        }
    }

    #[test]
    fn test_step_status_default() {
        assert_eq!(StepStatus::default(), StepStatus::Pending);
    }

    #[test]
    fn test_step_status_serialization() {
        let statuses = vec![
            (StepStatus::Pending, "\"pending\""),
            (StepStatus::Running, "\"running\""),
            (StepStatus::Completed, "\"completed\""),
            (StepStatus::Failed, "\"failed\""),
            (StepStatus::Skipped, "\"skipped\""),
        ];

        for (status, expected) in statuses {
            let json = serde_json::to_string(&status).expect("Should serialize");
            assert_eq!(json, expected);
        }
    }

    // =========================================================================
    // SECTION 10: Token Aggregation Tests
    // =========================================================================

    #[test]
    fn test_token_aggregation_across_steps() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        let step1 = {
            let mut s = StepTrace::new("s1", 0);
            s.tokens = TokenUsage::new(100, 50, 0.001);
            s
        };

        let step2 = {
            let mut s = StepTrace::new("s2", 1);
            s.tokens = TokenUsage::new(200, 100, 0.002);
            s
        };

        let step3 = {
            let mut s = StepTrace::new("s3", 2);
            s.tokens = TokenUsage::new(150, 75, 0.0015);
            s
        };

        trace.add_step(step1);
        trace.add_step(step2);
        trace.add_step(step3);

        assert_eq!(trace.tokens.input_tokens, 450);
        assert_eq!(trace.tokens.output_tokens, 225);
        assert_eq!(trace.tokens.total_tokens, 675);
        assert!((trace.tokens.cost_usd - 0.0045).abs() < 0.0001);
    }

    // =========================================================================
    // SECTION 11: Edge Cases and Boundary Conditions
    // =========================================================================

    #[test]
    fn test_empty_string_protocol_id() {
        let trace = ExecutionTrace::new("", "");

        assert_eq!(trace.protocol_id, "");
        assert_eq!(trace.protocol_version, "");

        // Should still serialize
        let json = trace.to_json().expect("Should serialize empty strings");
        assert!(json.contains("\"protocol_id\": \"\""));
    }

    #[test]
    fn test_unicode_in_trace() {
        let trace = ExecutionTrace::new("test", "1.0.0")
            .with_input(json!({"query": "What is the meaning of life? "}));

        let mut step = StepTrace::new("step1", 0);
        step.prompt = "Analyze: ...".to_string();
        step.raw_response = "The answer involves philosophical concepts".to_string();
        step.complete(
            StepOutput::Text {
                content: "Deep philosophical analysis complete".to_string(),
            },
            0.9,
        );

        let mut trace = trace;
        trace.add_step(step);

        let json = trace.to_json().expect("Should handle unicode");
        let deserialized: ExecutionTrace =
            serde_json::from_str(&json).expect("Should deserialize unicode");

        assert!(deserialized.input["query"].as_str().unwrap().contains(""));
    }

    #[test]
    fn test_very_long_response() {
        let mut step = StepTrace::new("long_response", 0);

        // Create a very long response (100KB)
        let long_response: String = "x".repeat(100_000);
        step.raw_response = long_response.clone();

        step.complete(
            StepOutput::Text {
                content: long_response.clone(),
            },
            0.9,
        );

        let mut trace = ExecutionTrace::new("test", "1.0.0");
        trace.add_step(step);

        let json = trace.to_json().expect("Should handle long strings");
        assert!(json.len() > 100_000);

        let deserialized: ExecutionTrace = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.steps[0].raw_response.len(), 100_000);
    }

    #[test]
    fn test_many_steps() {
        let mut trace = ExecutionTrace::new("stress_test", "1.0.0");

        // Add 1000 steps
        for i in 0..1000 {
            let mut step = StepTrace::new(format!("step_{}", i), i);
            step.tokens = TokenUsage::new(10, 5, 0.0001);
            step.complete(StepOutput::Empty, 0.9);
            trace.add_step(step);
        }

        assert_eq!(trace.steps.len(), 1000);
        assert_eq!(trace.completed_steps(), 1000);
        assert_eq!(trace.tokens.total_tokens, 15_000);

        // Should still serialize efficiently
        let json = trace
            .to_json_compact()
            .expect("Should serialize many steps");
        let deserialized: ExecutionTrace = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.steps.len(), 1000);
    }

    #[test]
    fn test_zero_confidence_steps() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        for i in 0..3 {
            let mut step = StepTrace::new(format!("step_{}", i), i);
            step.confidence = 0.0;
            step.status = StepStatus::Completed;
            trace.add_step(step);
        }

        assert_eq!(trace.average_confidence(), 0.0);
    }

    #[test]
    fn test_maximum_confidence() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        let mut step = StepTrace::new("perfect", 0);
        step.confidence = 1.0;
        step.status = StepStatus::Completed;
        trace.add_step(step);

        assert_eq!(trace.average_confidence(), 1.0);
    }

    // =========================================================================
    // SECTION 12: Clone and Debug Trait Tests
    // =========================================================================

    #[test]
    fn test_execution_trace_clone() {
        let mut original = ExecutionTrace::new("test", "1.0.0");
        original.metadata.model = Some("gpt-4".to_string());

        let mut step = StepTrace::new("step1", 0);
        step.complete(StepOutput::Empty, 0.9);
        original.add_step(step);

        let cloned = original.clone();

        // Clones should be equal
        assert_eq!(cloned.id, original.id);
        assert_eq!(cloned.protocol_id, original.protocol_id);
        assert_eq!(cloned.steps.len(), original.steps.len());
        assert_eq!(cloned.metadata.model, original.metadata.model);
    }

    #[test]
    fn test_step_trace_clone() {
        let mut original = StepTrace::new("step1", 0);
        original.prompt = "Test prompt".to_string();
        original.tokens = TokenUsage::new(100, 50, 0.001);
        original.complete(
            StepOutput::Text {
                content: "Result".to_string(),
            },
            0.9,
        );

        let cloned = original.clone();

        assert_eq!(cloned.step_id, original.step_id);
        assert_eq!(cloned.prompt, original.prompt);
        assert_eq!(cloned.confidence, original.confidence);
        assert_eq!(cloned.tokens.total_tokens, original.tokens.total_tokens);
    }

    #[test]
    fn test_debug_formatting() {
        let trace = ExecutionTrace::new("debug_test", "1.0.0");

        // Debug should not panic and should contain key info
        let debug_str = format!("{:?}", trace);

        assert!(debug_str.contains("ExecutionTrace"));
        assert!(debug_str.contains("debug_test"));
    }

    // =========================================================================
    // SECTION 13: Completed Steps Counter Tests
    // =========================================================================

    #[test]
    fn test_completed_steps_empty() {
        let trace = ExecutionTrace::new("test", "1.0.0");
        assert_eq!(trace.completed_steps(), 0);
    }

    #[test]
    fn test_completed_steps_all_completed() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        for i in 0..5 {
            let mut step = StepTrace::new(format!("step_{}", i), i);
            step.complete(StepOutput::Empty, 0.9);
            trace.add_step(step);
        }

        assert_eq!(trace.completed_steps(), 5);
    }

    #[test]
    fn test_completed_steps_none_completed() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        for i in 0..3 {
            let step = StepTrace::new(format!("step_{}", i), i);
            // Status remains Pending
            trace.add_step(step);
        }

        assert_eq!(trace.completed_steps(), 0);
    }

    #[test]
    fn test_completed_steps_mixed() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        // 2 completed
        for i in 0..2 {
            let mut step = StepTrace::new(format!("completed_{}", i), i);
            step.complete(StepOutput::Empty, 0.9);
            trace.add_step(step);
        }

        // 1 failed
        let mut failed = StepTrace::new("failed", 2);
        failed.fail("Error");
        trace.add_step(failed);

        // 1 pending
        let pending = StepTrace::new("pending", 3);
        trace.add_step(pending);

        // 1 running
        let mut running = StepTrace::new("running", 4);
        running.start();
        trace.add_step(running);

        assert_eq!(trace.steps.len(), 5);
        assert_eq!(trace.completed_steps(), 2);
    }
}
