//! Protocol Executor
//!
//! Executes ThinkTool protocols by orchestrating LLM calls
//! and managing step execution flow.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use super::llm::{LlmClient, LlmConfig, LlmProvider, LlmRequest, UnifiedLlmClient};
use super::protocol::{Protocol, ProtocolStep, BranchCondition, StepAction};
use super::registry::ProtocolRegistry;
use super::profiles::{ProfileRegistry, ReasoningProfile, ChainCondition};
use super::step::{StepResult, StepOutput, TokenUsage, ListItem};
use super::trace::{ExecutionTrace, StepTrace, StepStatus, TraceMetadata};

/// Configuration for protocol execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// LLM configuration
    pub llm: LlmConfig,

    /// Global timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Whether to save traces
    #[serde(default)]
    pub save_traces: bool,

    /// Trace output directory
    pub trace_dir: Option<PathBuf>,

    /// Verbose output
    #[serde(default)]
    pub verbose: bool,

    /// Use mock LLM (for testing)
    #[serde(default)]
    pub use_mock: bool,
}

fn default_timeout() -> u64 {
    120
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            llm: LlmConfig::default(),
            timeout_secs: default_timeout(),
            save_traces: false,
            trace_dir: None,
            verbose: false,
            use_mock: false,
        }
    }
}

impl ExecutorConfig {
    /// Create config for testing with mock LLM
    pub fn mock() -> Self {
        Self {
            use_mock: true,
            ..Default::default()
        }
    }
}

/// Input for protocol execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolInput {
    /// Input fields
    pub fields: HashMap<String, serde_json::Value>,
}

impl ProtocolInput {
    /// Create input with a query
    pub fn query(query: impl Into<String>) -> Self {
        let mut fields = HashMap::new();
        fields.insert("query".to_string(), serde_json::Value::String(query.into()));
        Self { fields }
    }

    /// Create input with an argument (for LaserLogic)
    pub fn argument(argument: impl Into<String>) -> Self {
        let mut fields = HashMap::new();
        fields.insert("argument".to_string(), serde_json::Value::String(argument.into()));
        Self { fields }
    }

    /// Create input with a statement (for BedRock)
    pub fn statement(statement: impl Into<String>) -> Self {
        let mut fields = HashMap::new();
        fields.insert("statement".to_string(), serde_json::Value::String(statement.into()));
        Self { fields }
    }

    /// Create input with a claim (for ProofGuard)
    pub fn claim(claim: impl Into<String>) -> Self {
        let mut fields = HashMap::new();
        fields.insert("claim".to_string(), serde_json::Value::String(claim.into()));
        Self { fields }
    }

    /// Create input with work to critique (for BrutalHonesty)
    pub fn work(work: impl Into<String>) -> Self {
        let mut fields = HashMap::new();
        fields.insert("work".to_string(), serde_json::Value::String(work.into()));
        Self { fields }
    }

    /// Add additional field
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), serde_json::Value::String(value.into()));
        self
    }

    /// Get a field as string
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(|v| v.as_str())
    }
}

/// Output from protocol execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolOutput {
    /// Protocol that was executed
    pub protocol_id: String,

    /// Whether execution succeeded
    pub success: bool,

    /// Output data
    pub data: HashMap<String, serde_json::Value>,

    /// Overall confidence score
    pub confidence: f64,

    /// Step results
    pub steps: Vec<StepResult>,

    /// Total token usage
    pub tokens: TokenUsage,

    /// Execution time in milliseconds
    pub duration_ms: u64,

    /// Error message if failed
    pub error: Option<String>,

    /// Trace ID (if saved)
    pub trace_id: Option<String>,
}

impl ProtocolOutput {
    /// Get a field from output data
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }

    /// Get perspectives (for GigaThink output)
    pub fn perspectives(&self) -> Vec<&str> {
        self.data
            .get("perspectives")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get verdict (for various outputs)
    pub fn verdict(&self) -> Option<&str> {
        self.data.get("verdict").and_then(|v| v.as_str())
    }
}

/// Protocol executor
pub struct ProtocolExecutor {
    /// Protocol registry
    registry: ProtocolRegistry,

    /// Profile registry
    profiles: ProfileRegistry,

    /// Configuration
    config: ExecutorConfig,

    /// LLM client
    llm_client: Option<UnifiedLlmClient>,
}

impl ProtocolExecutor {
    /// Create a new executor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ExecutorConfig::default())
    }

    /// Create executor for testing (mock LLM)
    pub fn mock() -> Result<Self> {
        Self::with_config(ExecutorConfig::mock())
    }

    /// Create executor with custom configuration
    pub fn with_config(config: ExecutorConfig) -> Result<Self> {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins()?;

        let profiles = ProfileRegistry::with_builtins();

        // Only create LLM client if not using mock
        let llm_client = if config.use_mock {
            None
        } else {
            Some(UnifiedLlmClient::new(config.llm.clone())?)
        };

        Ok(Self {
            registry,
            profiles,
            config,
            llm_client,
        })
    }

    /// Get the protocol registry
    pub fn registry(&self) -> &ProtocolRegistry {
        &self.registry
    }

    /// Get mutable registry
    pub fn registry_mut(&mut self) -> &mut ProtocolRegistry {
        &mut self.registry
    }

    /// Get profile registry
    pub fn profiles(&self) -> &ProfileRegistry {
        &self.profiles
    }

    /// Execute a protocol
    pub async fn execute(
        &self,
        protocol_id: &str,
        input: ProtocolInput,
    ) -> Result<ProtocolOutput> {
        let protocol = self.registry.get(protocol_id).ok_or_else(|| Error::NotFound {
            resource: format!("protocol:{}", protocol_id),
        })?.clone();

        // Validate input
        self.validate_input(&protocol, &input)?;

        let start = Instant::now();
        let mut trace = ExecutionTrace::new(&protocol.id, &protocol.version)
            .with_input(serde_json::to_value(&input.fields).unwrap_or_default());

        trace.timing.start();
        trace.metadata = TraceMetadata {
            model: Some(self.config.llm.model.clone()),
            provider: Some(format!("{:?}", self.config.llm.provider)),
            temperature: Some(self.config.llm.temperature),
            ..Default::default()
        };

        // Execute steps
        let mut step_results: Vec<StepResult> = Vec::new();
        let mut step_outputs: HashMap<String, StepOutput> = HashMap::new();
        let mut total_tokens = TokenUsage::default();

        for (index, step) in protocol.steps.iter().enumerate() {
            // Check dependencies
            if !self.dependencies_met(&step.depends_on, &step_results) {
                continue;
            }

            // Check branch condition
            if let Some(condition) = &step.branch {
                if !self.evaluate_branch_condition(condition, &step_results) {
                    let mut skipped = StepTrace::new(&step.id, index);
                    skipped.status = StepStatus::Skipped;
                    trace.add_step(skipped);
                    continue;
                }
            }

            // Execute step
            let step_result = self
                .execute_step(step, &input, &step_outputs, index)
                .await?;

            // Record in trace
            let mut step_trace = StepTrace::new(&step.id, index);
            step_trace.confidence = step_result.confidence;
            step_trace.tokens = step_result.tokens.clone();
            step_trace.duration_ms = step_result.duration_ms;

            if step_result.success {
                step_trace.complete(step_result.output.clone(), step_result.confidence);
            } else {
                step_trace.fail(step_result.error.clone().unwrap_or_default());
            }

            trace.add_step(step_trace);
            total_tokens.add(&step_result.tokens);
            step_outputs.insert(step.id.clone(), step_result.output.clone());
            step_results.push(step_result);
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // Calculate overall confidence
        let confidence = if step_results.is_empty() {
            0.0
        } else {
            step_results.iter().map(|r| r.confidence).sum::<f64>() / step_results.len() as f64
        };

        // Build output data
        let mut data = HashMap::new();
        for (key, output) in &step_outputs {
            data.insert(key.clone(), serde_json::to_value(output).unwrap_or_default());
        }
        data.insert("confidence".to_string(), serde_json::json!(confidence));

        // Check if all steps succeeded
        let success = step_results.iter().all(|r| r.success);
        let error = if success {
            None
        } else {
            step_results
                .iter()
                .find(|r| !r.success)
                .and_then(|r| r.error.clone())
        };

        // Complete trace
        if success {
            trace.complete(serde_json::to_value(&data).unwrap_or_default(), confidence);
        } else {
            trace.fail(&error.clone().unwrap_or_else(|| "Unknown error".to_string()));
        }

        // Save trace if configured
        let trace_id = if self.config.save_traces {
            self.save_trace(&trace)?;
            Some(trace.id.to_string())
        } else {
            None
        };

        Ok(ProtocolOutput {
            protocol_id: protocol_id.to_string(),
            success,
            data,
            confidence,
            steps: step_results,
            tokens: total_tokens,
            duration_ms,
            error,
            trace_id,
        })
    }

    /// Execute a reasoning profile (chain of protocols)
    pub async fn execute_profile(
        &self,
        profile_id: &str,
        input: ProtocolInput,
    ) -> Result<ProtocolOutput> {
        let profile = self.profiles.get(profile_id).ok_or_else(|| Error::NotFound {
            resource: format!("profile:{}", profile_id),
        })?.clone();

        let start = Instant::now();
        let mut all_step_results: Vec<StepResult> = Vec::new();
        let mut all_outputs: HashMap<String, serde_json::Value> = HashMap::new();
        let mut total_tokens = TokenUsage::default();
        let mut current_input = input.clone();

        // Track outputs by step ID for input mapping
        let mut step_outputs: HashMap<String, serde_json::Value> = HashMap::new();
        step_outputs.insert("input".to_string(), serde_json::to_value(&input.fields).unwrap_or_default());

        for chain_step in &profile.chain {
            // Check condition
            if let Some(condition) = &chain_step.condition {
                if !self.evaluate_chain_condition(condition, &all_step_results) {
                    continue;
                }
            }

            // Build input from mapping
            let mut mapped_input = ProtocolInput { fields: HashMap::new() };
            for (target_field, source_expr) in &chain_step.input_mapping {
                if let Some(value) = self.resolve_mapping(source_expr, &step_outputs, &input) {
                    mapped_input.fields.insert(target_field.clone(), value);
                }
            }

            // Fall back to original input if no mapping
            if mapped_input.fields.is_empty() {
                mapped_input = current_input.clone();
            }

            // Execute protocol
            let result = self.execute(&chain_step.protocol_id, mapped_input).await?;

            // Store outputs for next step
            step_outputs.insert(
                format!("steps.{}", chain_step.protocol_id),
                serde_json::to_value(&result.data).unwrap_or_default()
            );

            total_tokens.add(&result.tokens);
            all_step_results.extend(result.steps);
            all_outputs.extend(result.data);
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // Calculate overall confidence
        let confidence = if all_step_results.is_empty() {
            0.0
        } else {
            all_step_results.iter().map(|r| r.confidence).sum::<f64>() / all_step_results.len() as f64
        };

        let success = all_step_results.iter().all(|r| r.success) && confidence >= profile.min_confidence;

        Ok(ProtocolOutput {
            protocol_id: profile_id.to_string(),
            success,
            data: all_outputs,
            confidence,
            steps: all_step_results,
            tokens: total_tokens,
            duration_ms,
            error: None,
            trace_id: None,
        })
    }

    /// Validate input against protocol requirements
    fn validate_input(&self, protocol: &Protocol, input: &ProtocolInput) -> Result<()> {
        for required in &protocol.input.required {
            if !input.fields.contains_key(required) {
                return Err(Error::Validation(
                    format!("Missing required input field: {}", required)
                ));
            }
        }
        Ok(())
    }

    /// Check if step dependencies are met
    fn dependencies_met(&self, deps: &[String], results: &[StepResult]) -> bool {
        deps.iter().all(|dep| {
            results
                .iter()
                .any(|r| r.step_id == *dep && r.success)
        })
    }

    /// Evaluate a branch condition
    fn evaluate_branch_condition(&self, condition: &BranchCondition, results: &[StepResult]) -> bool {
        match condition {
            BranchCondition::Always => true,
            BranchCondition::ConfidenceBelow { threshold } => {
                results.last().map(|r| r.confidence < *threshold).unwrap_or(true)
            }
            BranchCondition::ConfidenceAbove { threshold } => {
                results.last().map(|r| r.confidence >= *threshold).unwrap_or(false)
            }
            BranchCondition::OutputEquals { field: _, value } => {
                results.last().map(|r| {
                    if let Some(text) = r.as_text() {
                        text.contains(value)
                    } else {
                        false
                    }
                }).unwrap_or(false)
            }
        }
    }

    /// Evaluate a chain condition
    fn evaluate_chain_condition(&self, condition: &ChainCondition, results: &[StepResult]) -> bool {
        match condition {
            ChainCondition::Always => true,
            ChainCondition::ConfidenceBelow { threshold } => {
                results.last().map(|r| r.confidence < *threshold).unwrap_or(true)
            }
            ChainCondition::ConfidenceAbove { threshold } => {
                results.last().map(|r| r.confidence >= *threshold).unwrap_or(false)
            }
            ChainCondition::OutputExists { step_id, field } => {
                results.iter().any(|r| r.step_id == *step_id && r.as_text().is_some())
            }
        }
    }

    /// Resolve a mapping expression to a value
    fn resolve_mapping(
        &self,
        expr: &str,
        step_outputs: &HashMap<String, serde_json::Value>,
        input: &ProtocolInput,
    ) -> Option<serde_json::Value> {
        // Handle input.* expressions
        if let Some(field) = expr.strip_prefix("input.") {
            return input.fields.get(field).cloned();
        }

        // Handle steps.*.* expressions
        if let Some(rest) = expr.strip_prefix("steps.") {
            let key = format!("steps.{}", rest.split('.').next().unwrap_or(""));
            if let Some(step_data) = step_outputs.get(&key) {
                // Try to get nested field
                let field = rest.split('.').skip(1).collect::<Vec<_>>().join(".");
                if !field.is_empty() {
                    return step_data.get(&field).cloned();
                }
                return Some(step_data.clone());
            }
        }

        None
    }

    /// Execute a single step
    async fn execute_step(
        &self,
        step: &ProtocolStep,
        input: &ProtocolInput,
        previous_outputs: &HashMap<String, StepOutput>,
        index: usize,
    ) -> Result<StepResult> {
        let start = Instant::now();

        // Build prompt from template
        let prompt = self.render_template(&step.prompt_template, input, previous_outputs);

        // Build system prompt based on step action
        let system = self.build_system_prompt(step);

        // Call LLM or use mock
        let (content, tokens) = if self.config.use_mock {
            self.mock_llm_call(&prompt, step).await
        } else {
            self.real_llm_call(&prompt, &system).await?
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        // Parse response into structured output
        let (output, confidence) = self.parse_step_output(&content, step);

        Ok(StepResult::success(&step.id, output, confidence)
            .with_duration(duration_ms)
            .with_tokens(tokens))
    }

    /// Build system prompt for a step
    fn build_system_prompt(&self, step: &ProtocolStep) -> String {
        let base = "You are a structured reasoning assistant. Follow the instructions precisely and provide clear, well-organized responses.";

        let action_guidance = match &step.action {
            StepAction::Generate { min_count, max_count } => {
                format!("Generate {}-{} distinct items. Number each item clearly.", min_count, max_count)
            }
            StepAction::Analyze { criteria } => {
                format!("Analyze based on these criteria: {}. Be thorough and specific.", criteria.join(", "))
            }
            StepAction::Synthesize { .. } => {
                "Synthesize the information into a coherent summary. Identify patterns and themes.".to_string()
            }
            StepAction::Validate { rules } => {
                format!("Validate against these rules: {}. Be explicit about pass/fail for each.", rules.join(", "))
            }
            StepAction::Critique { severity } => {
                format!("Provide {:?}-level critique. Be honest and specific about weaknesses.", severity)
            }
            StepAction::Decide { method } => {
                format!("Make a decision using {:?} method. Explain your reasoning clearly.", method)
            }
            StepAction::CrossReference { min_sources } => {
                format!("Cross-reference with at least {} sources. Cite each source.", min_sources)
            }
        };

        format!("{}\n\n{}\n\nProvide a confidence score (0.0-1.0) for your response.", base, action_guidance)
    }

    /// Render template with input values
    fn render_template(
        &self,
        template: &str,
        input: &ProtocolInput,
        previous_outputs: &HashMap<String, StepOutput>,
    ) -> String {
        let mut result = template.to_string();

        // Replace input placeholders
        for (key, value) in &input.fields {
            let placeholder = format!("{{{{{}}}}}", key);
            let value_str = match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            result = result.replace(&placeholder, &value_str);
        }

        // Replace previous output placeholders
        for (key, output) in previous_outputs {
            let placeholder = format!("{{{{{}}}}}", key);
            let value_str = match output {
                StepOutput::Text { content } => content.clone(),
                StepOutput::List { items } => {
                    items.iter().map(|i| i.content.clone()).collect::<Vec<_>>().join("\n")
                }
                other => serde_json::to_string(other).unwrap_or_default(),
            };
            result = result.replace(&placeholder, &value_str);
        }

        // Clean up unfilled placeholders
        let re = regex::Regex::new(r"\{\{#if \w+\}\}.*?\{\{/if\}\}").unwrap();
        result = re.replace_all(&result, "").to_string();

        result
    }

    /// Real LLM call
    async fn real_llm_call(&self, prompt: &str, system: &str) -> Result<(String, TokenUsage)> {
        let client = self.llm_client.as_ref().ok_or_else(|| {
            Error::Config("LLM client not initialized".to_string())
        })?;

        let request = LlmRequest::new(prompt)
            .with_system(system)
            .with_temperature(self.config.llm.temperature)
            .with_max_tokens(self.config.llm.max_tokens);

        let response = client.complete(request).await?;

        let tokens = TokenUsage::new(
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cost_usd(&self.config.llm.model),
        );

        Ok((response.content, tokens))
    }

    /// Mock LLM call (for testing)
    async fn mock_llm_call(&self, _prompt: &str, step: &ProtocolStep) -> (String, TokenUsage) {
        let content = match &step.action {
            StepAction::Generate { min_count, .. } => {
                let items: Vec<String> = (1..=*min_count)
                    .map(|i| format!("{}. Generated perspective {}", i, i))
                    .collect();
                format!("{}\n\nConfidence: 0.85", items.join("\n"))
            }
            StepAction::Analyze { .. } => {
                "Analysis:\n- Key finding 1\n- Key finding 2\n- Key finding 3\n\nConfidence: 0.82".to_string()
            }
            StepAction::Synthesize { .. } => {
                "Synthesis: The main themes identified are X, Y, and Z. Key insight: ...\n\nConfidence: 0.88".to_string()
            }
            StepAction::Validate { .. } => {
                "Validation result: PASS\n- Rule 1: Pass\n- Rule 2: Pass\n\nConfidence: 0.90".to_string()
            }
            StepAction::Critique { .. } => {
                "Critique:\n1. Strength: Good structure\n2. Weakness: Needs more evidence\n3. Suggestion: Add sources\n\nConfidence: 0.78".to_string()
            }
            StepAction::Decide { .. } => {
                "Decision: Option A recommended\nRationale: Best balance of factors\n\nConfidence: 0.85".to_string()
            }
            StepAction::CrossReference { .. } => {
                "Sources verified:\n1. Source A: Confirms\n2. Source B: Confirms\n3. Source C: Partially confirms\n\nConfidence: 0.87".to_string()
            }
        };

        let tokens = TokenUsage::new(100, 150, 0.001);
        (content, tokens)
    }

    /// Parse step output from LLM response
    fn parse_step_output(&self, content: &str, step: &ProtocolStep) -> (StepOutput, f64) {
        // Extract confidence from response
        let confidence = self.extract_confidence(content).unwrap_or(0.75);

        let output = match &step.action {
            StepAction::Generate { .. } => {
                let items = self.extract_list_items(content);
                StepOutput::List { items }
            }
            StepAction::Analyze { .. } | StepAction::Synthesize { .. } => {
                StepOutput::Text { content: content.to_string() }
            }
            StepAction::Validate { .. } => {
                let passed = content.to_lowercase().contains("pass");
                StepOutput::Boolean {
                    value: passed,
                    reason: Some(content.to_string()),
                }
            }
            StepAction::Critique { .. } => {
                let items = self.extract_list_items(content);
                StepOutput::List { items }
            }
            StepAction::Decide { .. } => {
                let mut data = HashMap::new();
                data.insert("decision".to_string(), serde_json::json!(content));
                StepOutput::Structured { data }
            }
            StepAction::CrossReference { .. } => {
                let items = self.extract_list_items(content);
                StepOutput::List { items }
            }
        };

        (output, confidence)
    }

    /// Extract confidence score from response text
    fn extract_confidence(&self, content: &str) -> Option<f64> {
        // Look for "Confidence: X.XX" pattern
        let re = regex::Regex::new(r"[Cc]onfidence:?\s*(\d+\.?\d*)").ok()?;
        if let Some(caps) = re.captures(content) {
            if let Some(m) = caps.get(1) {
                return m.as_str().parse::<f64>().ok().map(|v| v.min(1.0));
            }
        }
        None
    }

    /// Extract list items from response text
    fn extract_list_items(&self, content: &str) -> Vec<ListItem> {
        let mut items = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            // Match numbered items (1. 2. etc) or bullet points (- *)
            if let Some(text) = trimmed.strip_prefix(|c: char| c.is_ascii_digit())
                .and_then(|s| s.strip_prefix('.').or(s.strip_prefix(')')))
                .map(|s| s.trim())
            {
                if !text.is_empty() && !text.to_lowercase().starts_with("confidence") {
                    items.push(ListItem::new(text));
                }
            } else if let Some(text) = trimmed.strip_prefix('-').or(trimmed.strip_prefix('*')) {
                let text = text.trim();
                if !text.is_empty() && !text.to_lowercase().starts_with("confidence") {
                    items.push(ListItem::new(text));
                }
            }
        }

        items
    }

    /// Save trace to file
    fn save_trace(&self, trace: &ExecutionTrace) -> Result<()> {
        let dir = self.config.trace_dir.as_ref().ok_or_else(|| {
            Error::Config("Trace directory not configured".to_string())
        })?;

        std::fs::create_dir_all(dir).map_err(|e| {
            Error::IoMessage { message: format!("Failed to create trace directory: {}", e) }
        })?;

        let filename = format!("{}_{}.json", trace.protocol_id, trace.id);
        let path = dir.join(filename);

        let json = trace.to_json().map_err(|e| {
            Error::Parse { message: format!("Failed to serialize trace: {}", e) }
        })?;

        std::fs::write(&path, json).map_err(|e| {
            Error::IoMessage { message: format!("Failed to write trace: {}", e) }
        })?;

        Ok(())
    }

    /// List available protocols
    pub fn list_protocols(&self) -> Vec<&str> {
        self.registry.list_ids()
    }

    /// List available profiles
    pub fn list_profiles(&self) -> Vec<&str> {
        self.profiles.list_ids()
    }

    /// Get protocol info
    pub fn get_protocol(&self, id: &str) -> Option<&Protocol> {
        self.registry.get(id)
    }

    /// Get profile info
    pub fn get_profile(&self, id: &str) -> Option<&ReasoningProfile> {
        self.profiles.get(id)
    }
}

impl Default for ProtocolExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default executor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = ProtocolExecutor::mock().unwrap();
        assert!(!executor.registry().is_empty());
        assert!(!executor.profiles().is_empty());
    }

    #[test]
    fn test_list_protocols() {
        let executor = ProtocolExecutor::mock().unwrap();
        let protocols = executor.list_protocols();
        assert!(protocols.contains(&"gigathink"));
        assert!(protocols.contains(&"laserlogic"));
    }

    #[test]
    fn test_list_profiles() {
        let executor = ProtocolExecutor::mock().unwrap();
        let profiles = executor.list_profiles();
        assert!(profiles.contains(&"quick"));
        assert!(profiles.contains(&"balanced"));
        assert!(profiles.contains(&"paranoid"));
    }

    #[test]
    fn test_protocol_input() {
        let input = ProtocolInput::query("Test query")
            .with_field("context", "Some context");

        assert_eq!(input.get_str("query"), Some("Test query"));
        assert_eq!(input.get_str("context"), Some("Some context"));
    }

    #[test]
    fn test_template_rendering() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What is AI?");

        let template = "Question: {{query}}";
        let rendered = executor.render_template(template, &input, &HashMap::new());

        assert_eq!(rendered, "Question: What is AI?");
    }

    #[test]
    fn test_extract_confidence() {
        let executor = ProtocolExecutor::mock().unwrap();

        assert_eq!(executor.extract_confidence("Confidence: 0.85"), Some(0.85));
        assert_eq!(executor.extract_confidence("confidence: 0.9"), Some(0.9));
        assert_eq!(executor.extract_confidence("Some text\nConfidence: 0.75"), Some(0.75));
    }

    #[test]
    fn test_extract_list_items() {
        let executor = ProtocolExecutor::mock().unwrap();

        let content = "1. First item\n2. Second item\n- Third item\nConfidence: 0.8";
        let items = executor.extract_list_items(content);

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].content, "First item");
        assert_eq!(items[1].content, "Second item");
        assert_eq!(items[2].content, "Third item");
    }

    #[tokio::test]
    async fn test_execute_gigathink_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What are the key factors for startup success?");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_execute_profile_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Should we adopt microservices?");

        let result = executor.execute_profile("quick", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
    }
}
