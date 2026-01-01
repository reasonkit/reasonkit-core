//! Protocol Executor
//!
//! Executes ThinkTool protocols by orchestrating LLM calls
//! and managing step execution flow.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use futures::stream::{FuturesUnordered, StreamExt};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock as TokioRwLock;

// ═══════════════════════════════════════════════════════════════════════════
// STATIC REGEX PATTERNS (PERFORMANCE OPTIMIZATION)
// ═══════════════════════════════════════════════════════════════════════════

/// Static regex for conditional blocks: {{#if ...}}...{{/if}}
static CONDITIONAL_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\{\{#if \w+\}\}.*?\{\{/if\}\}").expect("Invalid static regex pattern")
});

/// Static regex for unfilled placeholders: {{...}}
static UNFILLED_PLACEHOLDER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\{\{[^}]+\}\}").expect("Invalid static regex pattern"));

// Thread-local cache for dynamic nested regex patterns ({{step_id.field}})
// This avoids recompiling the same patterns within a single execution
thread_local! {
    static NESTED_REGEX_CACHE: RefCell<HashMap<String, Regex>> = RefCell::new(HashMap::new());
}

/// Get or create a cached nested field regex pattern
fn get_nested_regex(key: &str) -> Regex {
    NESTED_REGEX_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(re) = cache.get(key) {
            return re.clone();
        }
        let pattern = format!(r"\{{\{{{}\\.(\w+)\}}\}}", regex::escape(key));
        let re = Regex::new(&pattern).expect("Failed to compile nested regex pattern");
        cache.insert(key.to_string(), re.clone());
        re
    })
}

use super::budget::{BudgetConfig, BudgetSummary, BudgetTracker};
use super::consistency::{ConsistencyResult, SelfConsistencyConfig, SelfConsistencyEngine};
use super::llm::{LlmClient, LlmConfig, LlmRequest, UnifiedLlmClient};
use super::profiles::{ChainCondition, ProfileRegistry, ReasoningProfile};
use super::protocol::{BranchCondition, Protocol, ProtocolStep, StepAction};
use super::registry::ProtocolRegistry;
use super::step::{ListItem, StepOutput, StepResult, TokenUsage};
use super::trace::{ExecutionTrace, StepStatus, StepTrace, TraceMetadata};
use crate::error::{Error, Result};

/// CLI tool configuration for shell-out execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliToolConfig {
    /// CLI command to use (e.g., "claude", "codex", "gemini")
    pub command: String,
    /// Arguments to pass before the prompt (e.g., ["-p"] for claude/gemini)
    pub pre_args: Vec<String>,
    /// Arguments to pass after the prompt
    pub post_args: Vec<String>,
    /// Whether this tool requires interactive input handling
    pub interactive: bool,
}

impl CliToolConfig {
    /// Create config for Claude CLI (claude -p "...")
    pub fn claude() -> Self {
        Self {
            command: "claude".to_string(),
            pre_args: vec!["-p".to_string()],
            post_args: vec!["--output-format".to_string(), "text".to_string()],
            interactive: false,
        }
    }

    /// Create config for Codex CLI (codex "...")
    pub fn codex() -> Self {
        Self {
            command: "codex".to_string(),
            pre_args: vec!["-q".to_string()],
            post_args: vec![],
            interactive: false,
        }
    }

    /// Create config for Gemini CLI (gemini -p "...")
    pub fn gemini() -> Self {
        Self {
            command: "gemini".to_string(),
            pre_args: vec!["-p".to_string()],
            post_args: vec![],
            interactive: false,
        }
    }

    /// Create config for OpenCode CLI (opencode "...")
    pub fn opencode() -> Self {
        let command = std::env::var("RK_OPENCODE_CMD").unwrap_or_else(|_| "opencode".to_string());
        Self {
            command,
            pre_args: vec!["--no-rk".to_string(), "run".to_string()],
            post_args: vec![],
            interactive: false,
        }
    }

    /// Create config for GitHub Copilot CLI (gh copilot suggest)
    pub fn copilot() -> Self {
        Self {
            command: "gh".to_string(),
            pre_args: vec!["copilot".to_string(), "suggest".to_string()],
            post_args: vec![],
            interactive: true,
        }
    }
}

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

    /// Budget constraints for adaptive execution
    #[serde(default)]
    pub budget: BudgetConfig,

    /// CLI tool configuration (for shell-out execution to claude/codex/gemini/etc.)
    #[serde(default)]
    pub cli_tool: Option<CliToolConfig>,

    /// Self-Consistency configuration (Wang et al. 2023)
    /// Enables voting across multiple reasoning paths for improved accuracy
    /// Reference: arXiv:2203.11171 (+17.9% GSM8K improvement)
    #[serde(default)]
    pub self_consistency: Option<SelfConsistencyConfig>,

    /// Show progress indicators during execution
    /// Outputs step progress to stderr so it doesn't interfere with JSON output
    #[serde(default = "default_show_progress")]
    pub show_progress: bool,

    /// Enable parallel execution of independent steps
    /// When enabled, steps without dependencies or with satisfied dependencies
    /// will be executed concurrently, significantly reducing total latency.
    /// PERFORMANCE: Can reduce (N-1)/N latency for N independent steps.
    #[serde(default)]
    pub enable_parallel: bool,

    /// Maximum concurrent steps when parallel execution is enabled
    /// Set to 0 for unlimited concurrency (default: 4)
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_steps: usize,
}

fn default_max_concurrent() -> usize {
    4
}

fn default_show_progress() -> bool {
    true
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
            budget: BudgetConfig::default(),
            cli_tool: None,
            self_consistency: None,
            show_progress: default_show_progress(),
            enable_parallel: false,
            max_concurrent_steps: default_max_concurrent(),
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

    /// Create config for CLI tool execution
    pub fn with_cli_tool(tool: CliToolConfig) -> Self {
        Self {
            cli_tool: Some(tool),
            ..Default::default()
        }
    }

    /// Create config for Claude CLI
    pub fn claude_cli() -> Self {
        Self::with_cli_tool(CliToolConfig::claude())
    }

    /// Create config for Codex CLI
    pub fn codex_cli() -> Self {
        Self::with_cli_tool(CliToolConfig::codex())
    }

    /// Create config for Gemini CLI
    pub fn gemini_cli() -> Self {
        Self::with_cli_tool(CliToolConfig::gemini())
    }

    /// Create config for OpenCode CLI
    pub fn opencode_cli() -> Self {
        Self::with_cli_tool(CliToolConfig::opencode())
    }

    /// Create config for Copilot CLI
    pub fn copilot_cli() -> Self {
        Self::with_cli_tool(CliToolConfig::copilot())
    }

    /// Enable Self-Consistency with default config (5 samples, majority vote)
    /// Research: Wang et al. 2023 (arXiv:2203.11171) +17.9% GSM8K
    pub fn with_self_consistency(mut self) -> Self {
        self.self_consistency = Some(SelfConsistencyConfig::default());
        self
    }

    /// Enable Self-Consistency with custom config
    pub fn with_self_consistency_config(mut self, config: SelfConsistencyConfig) -> Self {
        self.self_consistency = Some(config);
        self
    }

    /// Enable fast Self-Consistency (3 samples, 70% threshold)
    pub fn with_self_consistency_fast(mut self) -> Self {
        self.self_consistency = Some(SelfConsistencyConfig::fast());
        self
    }

    /// Enable thorough Self-Consistency (10 samples, no early stopping)
    pub fn with_self_consistency_thorough(mut self) -> Self {
        self.self_consistency = Some(SelfConsistencyConfig::thorough());
        self
    }

    /// Enable paranoid Self-Consistency (15 samples, max accuracy)
    pub fn with_self_consistency_paranoid(mut self) -> Self {
        self.self_consistency = Some(SelfConsistencyConfig::paranoid());
        self
    }

    /// Enable parallel execution of independent steps
    /// PERFORMANCE: Can reduce (N-1)/N latency for N independent steps
    pub fn with_parallel(mut self) -> Self {
        self.enable_parallel = true;
        self
    }

    /// Enable parallel execution with custom concurrency limit
    pub fn with_parallel_limit(mut self, max_concurrent: usize) -> Self {
        self.enable_parallel = true;
        self.max_concurrent_steps = max_concurrent;
        self
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
        fields.insert(
            "argument".to_string(),
            serde_json::Value::String(argument.into()),
        );
        Self { fields }
    }

    /// Create input with a statement (for BedRock)
    pub fn statement(statement: impl Into<String>) -> Self {
        let mut fields = HashMap::new();
        fields.insert(
            "statement".to_string(),
            serde_json::Value::String(statement.into()),
        );
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
        self.fields
            .insert(key.into(), serde_json::Value::String(value.into()));
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

    /// Budget usage summary (if budget was set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_summary: Option<BudgetSummary>,
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
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
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
    pub async fn execute(&self, protocol_id: &str, input: ProtocolInput) -> Result<ProtocolOutput> {
        let protocol = self
            .registry
            .get(protocol_id)
            .ok_or_else(|| Error::NotFound {
                resource: format!("protocol:{}", protocol_id),
            })?
            .clone();

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

        // Execute steps - choose sequential or parallel based on config
        let (step_results, step_outputs, total_tokens, step_traces) = if self.config.enable_parallel
        {
            self.execute_steps_parallel(&protocol.steps, &input, &start)
                .await?
        } else {
            self.execute_steps_sequential(&protocol.steps, &input, &start)
                .await?
        };

        // Add traces
        for step_trace in step_traces {
            trace.add_step(step_trace);
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
            data.insert(
                key.clone(),
                serde_json::to_value(output).unwrap_or_default(),
            );
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

        // Build budget summary if budget is constrained
        let budget_summary = if self.config.budget.is_constrained() {
            let mut tracker = BudgetTracker::new(self.config.budget.clone());
            // Record cumulative usage from all steps
            for result in &step_results {
                tracker.record_usage(result.tokens.total_tokens, result.tokens.cost_usd);
            }
            Some(tracker.summary())
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
            budget_summary,
        })
    }

    /// Execute steps sequentially (original behavior)
    async fn execute_steps_sequential(
        &self,
        steps: &[ProtocolStep],
        input: &ProtocolInput,
        start: &Instant,
    ) -> Result<(
        Vec<StepResult>,
        HashMap<String, StepOutput>,
        TokenUsage,
        Vec<StepTrace>,
    )> {
        let mut step_results: Vec<StepResult> = Vec::with_capacity(steps.len());
        let mut step_outputs: HashMap<String, StepOutput> = HashMap::with_capacity(steps.len());
        let mut total_tokens = TokenUsage::default();
        let mut traces: Vec<StepTrace> = Vec::with_capacity(steps.len());

        let total_steps = steps.len();
        for (index, step) in steps.iter().enumerate() {
            // Check dependencies
            if !self.dependencies_met(&step.depends_on, &step_results) {
                continue;
            }

            // Check branch condition
            if let Some(condition) = &step.branch {
                if !self.evaluate_branch_condition(condition, &step_results) {
                    let mut skipped = StepTrace::new(&step.id, index);
                    skipped.status = StepStatus::Skipped;
                    traces.push(skipped);
                    continue;
                }
            }

            // Progress indicator
            if self.config.show_progress {
                let elapsed = start.elapsed().as_secs();
                eprintln!(
                    "\x1b[2m[{}/{}]\x1b[0m \x1b[36m⏳\x1b[0m Executing step: \x1b[1m{}\x1b[0m ({}s elapsed)...",
                    index + 1,
                    total_steps,
                    step.id,
                    elapsed
                );
            }

            // Execute step
            let step_result = self.execute_step(step, input, &step_outputs, index).await?;

            // Progress update
            if self.config.show_progress {
                let status_icon = if step_result.success { "✓" } else { "✗" };
                let status_color = if step_result.success {
                    "\x1b[32m"
                } else {
                    "\x1b[31m"
                };
                eprintln!(
                    "\x1b[2m[{}/{}]\x1b[0m {}{}\x1b[0m {} completed ({:.1}% confidence, {}ms)",
                    index + 1,
                    total_steps,
                    status_color,
                    status_icon,
                    step.id,
                    step_result.confidence * 100.0,
                    step_result.duration_ms
                );
            }

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

            traces.push(step_trace);
            total_tokens.add(&step_result.tokens);
            step_outputs.insert(step.id.clone(), step_result.output.clone());
            step_results.push(step_result);
        }

        Ok((step_results, step_outputs, total_tokens, traces))
    }

    /// Execute steps in parallel when dependencies allow
    /// PERFORMANCE: Reduces latency by (N-1)/N for N independent steps
    async fn execute_steps_parallel(
        &self,
        steps: &[ProtocolStep],
        input: &ProtocolInput,
        start: &Instant,
    ) -> Result<(
        Vec<StepResult>,
        HashMap<String, StepOutput>,
        TokenUsage,
        Vec<StepTrace>,
    )> {
        let total_steps = steps.len();

        // Shared state protected by async RwLock
        let completed_ids: Arc<TokioRwLock<HashSet<String>>> =
            Arc::new(TokioRwLock::new(HashSet::with_capacity(total_steps)));
        let step_outputs: Arc<TokioRwLock<HashMap<String, StepOutput>>> =
            Arc::new(TokioRwLock::new(HashMap::with_capacity(total_steps)));
        let step_results: Arc<TokioRwLock<Vec<(usize, StepResult)>>> =
            Arc::new(TokioRwLock::new(Vec::with_capacity(total_steps)));
        let traces: Arc<TokioRwLock<Vec<StepTrace>>> =
            Arc::new(TokioRwLock::new(Vec::with_capacity(total_steps)));

        // Track pending steps
        let mut pending: HashSet<usize> = (0..total_steps).collect();
        let mut completed_count = 0;

        while completed_count < total_steps && !pending.is_empty() {
            // Find steps that are ready to execute (dependencies satisfied)
            let completed_ids_guard = completed_ids.read().await;
            let ready_indices: Vec<usize> = pending
                .iter()
                .filter(|&&idx| {
                    let step = &steps[idx];
                    step.depends_on
                        .iter()
                        .all(|dep| completed_ids_guard.contains(dep))
                })
                .copied()
                .collect();
            drop(completed_ids_guard);

            if ready_indices.is_empty() && completed_count < total_steps {
                // No ready steps but not all complete - this shouldn't happen with valid deps
                break;
            }

            // Limit concurrency if configured
            let max_concurrent = if self.config.max_concurrent_steps > 0 {
                self.config.max_concurrent_steps.min(ready_indices.len())
            } else {
                ready_indices.len()
            };

            // Execute ready steps concurrently
            let mut futures = FuturesUnordered::new();

            for idx in ready_indices.into_iter().take(max_concurrent) {
                pending.remove(&idx);
                let step = steps[idx].clone();
                let input = input.clone();
                let step_outputs_clone = Arc::clone(&step_outputs);
                let completed_ids_clone = Arc::clone(&completed_ids);
                let step_results_clone = Arc::clone(&step_results);
                let traces_clone = Arc::clone(&traces);
                let show_progress = self.config.show_progress;
                let start_clone = *start;

                // Clone self fields needed for execution
                let config = self.config.clone();
                let llm_client = self.llm_client.as_ref().map(|_| {
                    // For parallel execution, create new clients that share the HTTP pool
                    UnifiedLlmClient::new(config.llm.clone()).ok()
                });

                futures.push(async move {
                    // Progress indicator
                    if show_progress {
                        let elapsed = start_clone.elapsed().as_secs();
                        eprintln!(
                            "\x1b[2m[{}/{}]\x1b[0m \x1b[36m⏳\x1b[0m Executing step: \x1b[1m{}\x1b[0m ({}s elapsed, parallel)...",
                            idx + 1,
                            total_steps,
                            step.id,
                            elapsed
                        );
                    }

                    // Get current outputs for template rendering
                    let outputs = step_outputs_clone.read().await.clone();

                    // Execute the step
                    let step_start = Instant::now();
                    let (response, tokens) = if config.use_mock {
                        // Mock execution
                        let mock_response = format!("Mock response for step: {}", step.id);
                        (mock_response, TokenUsage::default())
                    } else if let Some(Some(client)) = llm_client {
                        // Real LLM call
                        let prompt = Self::render_template_static(&step.prompt_template, &input, &outputs);
                        let system = Self::build_system_prompt_static(&step);
                        let request = super::llm::LlmRequest::new(&prompt)
                            .with_system(&system)
                            .with_temperature(config.llm.temperature)
                            .with_max_tokens(config.llm.max_tokens);

                        match client.complete(request).await {
                            Ok(resp) => {
                                let tokens = TokenUsage {
                                    input_tokens: resp.usage.input_tokens,
                                    output_tokens: resp.usage.output_tokens,
                                    total_tokens: resp.usage.total_tokens,
                                    cost_usd: 0.0,
                                };
                                (resp.content, tokens)
                            }
                            Err(e) => {
                                return (idx, step.id.clone(), Err(e));
                            }
                        }
                    } else {
                        // Fallback mock
                        let mock_response = format!("Mock response for step: {}", step.id);
                        (mock_response, TokenUsage::default())
                    };

                    let duration_ms = step_start.elapsed().as_millis() as u64;
                    let confidence = Self::extract_confidence_static(&response).unwrap_or(0.7);
                    let output = StepOutput::Text {
                        content: response.clone(),
                    };

                    let result = StepResult {
                        step_id: step.id.clone(),
                        success: true,
                        output: output.clone(),
                        confidence,
                        tokens: tokens.clone(),
                        duration_ms,
                        error: None,
                    };

                    // Update shared state
                    {
                        let mut outputs = step_outputs_clone.write().await;
                        outputs.insert(step.id.clone(), output.clone());
                    }
                    {
                        let mut completed = completed_ids_clone.write().await;
                        completed.insert(step.id.clone());
                    }
                    {
                        let mut results = step_results_clone.write().await;
                        results.push((idx, result.clone()));
                    }

                    // Create trace
                    let mut step_trace = StepTrace::new(&step.id, idx);
                    step_trace.confidence = confidence;
                    step_trace.tokens = tokens;
                    step_trace.duration_ms = duration_ms;
                    step_trace.complete(output, confidence);

                    {
                        let mut traces = traces_clone.write().await;
                        traces.push(step_trace);
                    }

                    // Progress update
                    if show_progress {
                        eprintln!(
                            "\x1b[2m[{}/{}]\x1b[0m \x1b[32m✓\x1b[0m {} completed ({:.1}% confidence, {}ms, parallel)",
                            idx + 1,
                            total_steps,
                            step.id,
                            confidence * 100.0,
                            duration_ms
                        );
                    }

                    (idx, step.id.clone(), Ok(result))
                });
            }

            // Wait for this batch to complete
            while let Some((_idx, step_id, result)) = futures.next().await {
                match result {
                    Ok(_) => {
                        completed_count += 1;
                    }
                    Err(e) => {
                        return Err(Error::Validation(format!(
                            "Step '{}': Parallel step execution failed: {}",
                            step_id, e
                        )));
                    }
                }
            }
        }

        // Collect results
        let step_outputs = Arc::try_unwrap(step_outputs)
            .map_err(|_| {
                Error::Config(
                    "Failed to unwrap step_outputs: Arc still has multiple references".to_string(),
                )
            })?
            .into_inner();
        let mut step_results_vec = Arc::try_unwrap(step_results)
            .map_err(|_| {
                Error::Config(
                    "Failed to unwrap step_results: Arc still has multiple references".to_string(),
                )
            })?
            .into_inner();
        let traces = Arc::try_unwrap(traces)
            .map_err(|_| {
                Error::Config(
                    "Failed to unwrap traces: Arc still has multiple references".to_string(),
                )
            })?
            .into_inner();

        // Sort results by index to maintain order
        step_results_vec.sort_by_key(|(idx, _)| *idx);
        let step_results: Vec<StepResult> = step_results_vec.into_iter().map(|(_, r)| r).collect();

        // Calculate total tokens
        let mut total_tokens = TokenUsage::default();
        for result in &step_results {
            total_tokens.add(&result.tokens);
        }

        Ok((step_results, step_outputs, total_tokens, traces))
    }

    /// Static version of render_template for use in parallel execution
    fn render_template_static(
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
                StepOutput::List { items } => items
                    .iter()
                    .map(|i| i.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
                other => serde_json::to_string(other).unwrap_or_default(),
            };
            result = result.replace(&placeholder, &value_str);
        }

        // Clean up using static regexes
        result = CONDITIONAL_BLOCK_RE.replace_all(&result, "").to_string();
        result = UNFILLED_PLACEHOLDER_RE.replace_all(&result, "").to_string();

        result
    }

    /// Static version of build_system_prompt for use in parallel execution
    fn build_system_prompt_static(step: &ProtocolStep) -> String {
        let base = "You are a structured reasoning assistant following the ReasonKit protocol.";
        let action_guidance = match &step.action {
            StepAction::Analyze { .. } => {
                "Analyze the given input thoroughly. Break down components and relationships."
            }
            StepAction::Synthesize { .. } => {
                "Synthesize information from previous steps into a coherent whole."
            }
            StepAction::Validate { .. } => "Validate claims and check for logical consistency.",
            StepAction::Generate { .. } => "Generate new ideas or content based on the context.",
            StepAction::Critique { .. } => {
                "Critically evaluate the reasoning and identify weaknesses."
            }
            StepAction::Decide { .. } => {
                "Make a decision based on the available evidence and reasoning."
            }
            StepAction::CrossReference { .. } => {
                "Cross-reference information from multiple sources to verify claims."
            }
        };

        format!(
            "{}\n\n{}\n\nProvide a confidence score (0.0-1.0) for your response.",
            base, action_guidance
        )
    }

    /// Static version of extract_confidence for use in parallel execution
    fn extract_confidence_static(content: &str) -> Option<f64> {
        use once_cell::sync::Lazy;
        static CONFIDENCE_RE: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"(?i)confidence[:\s]+(\d+\.?\d*)").expect("Invalid regex pattern")
        });

        if let Some(caps) = CONFIDENCE_RE.captures(content) {
            if let Some(m) = caps.get(1) {
                return m.as_str().parse::<f64>().ok().map(|v| v.min(1.0));
            }
        }
        None
    }

    /// Execute a reasoning profile (chain of protocols)
    pub async fn execute_profile(
        &self,
        profile_id: &str,
        input: ProtocolInput,
    ) -> Result<ProtocolOutput> {
        let profile = self
            .profiles
            .get(profile_id)
            .ok_or_else(|| Error::NotFound {
                resource: format!("profile:{}", profile_id),
            })?
            .clone();

        let start = Instant::now();
        let chain_len = profile.chain.len();
        // PERFORMANCE: Pre-allocate based on expected chain size
        let mut all_step_results: Vec<StepResult> = Vec::with_capacity(chain_len * 3); // ~3 steps per protocol
        let mut all_outputs: HashMap<String, serde_json::Value> = HashMap::with_capacity(chain_len);
        let mut total_tokens = TokenUsage::default();
        let current_input = input.clone();

        // Track outputs by step ID for input mapping
        let mut step_outputs: HashMap<String, serde_json::Value> =
            HashMap::with_capacity(chain_len + 1);
        step_outputs.insert(
            "input".to_string(),
            serde_json::to_value(&input.fields).unwrap_or_default(),
        );

        for chain_step in &profile.chain {
            // Check condition
            if let Some(condition) = &chain_step.condition {
                if !self.evaluate_chain_condition(condition, &all_step_results) {
                    continue;
                }
            }

            // Build input from mapping
            let mut mapped_input = ProtocolInput {
                fields: HashMap::with_capacity(chain_step.input_mapping.len()),
            };
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
                serde_json::to_value(&result.data).unwrap_or_default(),
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
            all_step_results.iter().map(|r| r.confidence).sum::<f64>()
                / all_step_results.len() as f64
        };

        let success =
            all_step_results.iter().all(|r| r.success) && confidence >= profile.min_confidence;

        // Build budget summary if budget is constrained
        let budget_summary = if self.config.budget.is_constrained() {
            let mut tracker = BudgetTracker::new(self.config.budget.clone());
            for result in &all_step_results {
                tracker.record_usage(result.tokens.total_tokens, result.tokens.cost_usd);
            }
            Some(tracker.summary())
        } else {
            None
        };

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
            budget_summary,
        })
    }

    /// Execute a profile with Self-Consistency voting
    ///
    /// This runs multiple independent reasoning paths and aggregates them via voting.
    /// Based on Wang et al. 2023 (arXiv:2203.11171) which showed +17.9% on GSM8K.
    ///
    /// # Arguments
    /// * `profile_id` - Profile to execute (e.g., "balanced", "deep")
    /// * `input` - Input data for the protocol
    /// * `sc_config` - Self-Consistency configuration
    ///
    /// # Returns
    /// ProtocolOutput with voted answer and ConsistencyResult details
    pub async fn execute_with_self_consistency(
        &self,
        profile_id: &str,
        input: ProtocolInput,
        sc_config: &SelfConsistencyConfig,
    ) -> Result<(ProtocolOutput, ConsistencyResult)> {
        let engine = SelfConsistencyEngine::new(sc_config.clone());
        let start = Instant::now();
        // PERFORMANCE: Pre-allocate based on configured sample count
        let mut all_results: Vec<StepResult> = Vec::with_capacity(sc_config.num_samples);
        let mut all_outputs: Vec<ProtocolOutput> = Vec::with_capacity(sc_config.num_samples);
        let mut total_tokens = TokenUsage::default();

        // Run multiple samples
        for sample_idx in 0..sc_config.num_samples {
            // Execute profile for this sample
            let output = self.execute_profile(profile_id, input.clone()).await?;

            // Create step result for voting
            let step_result = StepResult::success(
                format!("sample_{}", sample_idx),
                StepOutput::Text {
                    content: self.extract_voting_text(&output),
                },
                output.confidence,
            );

            all_results.push(step_result);
            total_tokens.add(&output.tokens);
            all_outputs.push(output);

            // Check for early stopping
            if sc_config.early_stopping && engine.should_early_stop(&all_results) {
                break;
            }
        }

        // Vote using Self-Consistency engine
        let consistency_result = engine.vote(all_results.clone());

        // Build the final output using the voted answer
        let duration_ms = start.elapsed().as_millis() as u64;

        // Use the output from the sample that matches the voted answer, or first sample
        let best_output = all_outputs
            .iter()
            .find(|o| {
                self.extract_voting_text(o)
                    .contains(&consistency_result.answer)
            })
            .cloned()
            .or_else(|| all_outputs.first().cloned())
            .ok_or_else(|| Error::Config("No outputs generated during self-consistency".into()))?;

        let mut final_output = best_output;
        final_output.confidence = consistency_result.confidence;
        final_output.tokens = total_tokens;
        final_output.duration_ms = duration_ms;

        // Add self-consistency metadata to output data
        final_output.data.insert(
            "self_consistency".to_string(),
            serde_json::json!({
                "voted_answer": consistency_result.answer,
                "agreement_ratio": consistency_result.agreement_ratio,
                "vote_count": consistency_result.vote_count,
                "total_samples": consistency_result.total_samples,
                "early_stopped": consistency_result.early_stopped,
            }),
        );

        Ok((final_output, consistency_result))
    }

    /// Extract text suitable for voting from protocol output
    fn extract_voting_text(&self, output: &ProtocolOutput) -> String {
        // Try to get conclusion from data
        if let Some(conclusion) = output.data.get("conclusion") {
            if let Some(s) = conclusion.as_str() {
                return s.to_string();
            }
        }

        // Try to get from last step
        if let Some(last_step) = output.steps.last() {
            if let Some(text) = last_step.as_text() {
                return text.to_string();
            }
        }

        // Concatenate all step outputs
        output
            .steps
            .iter()
            .filter_map(|s| s.as_text())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Validate input against protocol requirements
    fn validate_input(&self, protocol: &Protocol, input: &ProtocolInput) -> Result<()> {
        for required in &protocol.input.required {
            if !input.fields.contains_key(required) {
                return Err(Error::Validation(format!(
                    "Missing required input field: {}",
                    required
                )));
            }
        }
        Ok(())
    }

    /// Check if step dependencies are met
    fn dependencies_met(&self, deps: &[String], results: &[StepResult]) -> bool {
        deps.iter()
            .all(|dep| results.iter().any(|r| r.step_id == *dep && r.success))
    }

    /// Evaluate a branch condition
    fn evaluate_branch_condition(
        &self,
        condition: &BranchCondition,
        results: &[StepResult],
    ) -> bool {
        match condition {
            BranchCondition::Always => true,
            BranchCondition::ConfidenceBelow { threshold } => results
                .last()
                .map(|r| r.confidence < *threshold)
                .unwrap_or(true),
            BranchCondition::ConfidenceAbove { threshold } => results
                .last()
                .map(|r| r.confidence >= *threshold)
                .unwrap_or(false),
            BranchCondition::OutputEquals { field: _, value } => results
                .last()
                .map(|r| {
                    if let Some(text) = r.as_text() {
                        text.contains(value)
                    } else {
                        false
                    }
                })
                .unwrap_or(false),
        }
    }

    /// Evaluate a chain condition
    fn evaluate_chain_condition(&self, condition: &ChainCondition, results: &[StepResult]) -> bool {
        match condition {
            ChainCondition::Always => true,
            ChainCondition::ConfidenceBelow { threshold } => results
                .last()
                .map(|r| r.confidence < *threshold)
                .unwrap_or(true),
            ChainCondition::ConfidenceAbove { threshold } => results
                .last()
                .map(|r| r.confidence >= *threshold)
                .unwrap_or(false),
            ChainCondition::OutputExists { step_id, field: _ } => results
                .iter()
                .any(|r| r.step_id == *step_id && r.as_text().is_some()),
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
        _index: usize,
    ) -> Result<StepResult> {
        let start = Instant::now();

        // Build prompt from template
        let prompt = self.render_template(&step.prompt_template, input, previous_outputs);

        // Build system prompt based on step action
        let system = self.build_system_prompt(step);

        // Call LLM, CLI tool, or mock
        let (content, tokens) = if self.config.use_mock {
            self.mock_llm_call(&prompt, step).await
        } else if self.config.cli_tool.is_some() {
            self.cli_tool_call(&prompt, &system).await?
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
            StepAction::Generate {
                min_count,
                max_count,
            } => {
                format!(
                    "Generate {}-{} distinct items. Number each item clearly.",
                    min_count, max_count
                )
            }
            StepAction::Analyze { criteria } => {
                format!(
                    "Analyze based on these criteria: {}. Be thorough and specific.",
                    criteria.join(", ")
                )
            }
            StepAction::Synthesize { .. } => {
                "Synthesize the information into a coherent summary. Identify patterns and themes."
                    .to_string()
            }
            StepAction::Validate { rules } => {
                format!(
                    "Validate against these rules: {}. Be explicit about pass/fail for each.",
                    rules.join(", ")
                )
            }
            StepAction::Critique { severity } => {
                format!(
                    "Provide {:?}-level critique. Be honest and specific about weaknesses.",
                    severity
                )
            }
            StepAction::Decide { method } => {
                format!(
                    "Make a decision using {:?} method. Explain your reasoning clearly.",
                    method
                )
            }
            StepAction::CrossReference { min_sources } => {
                format!(
                    "Cross-reference with at least {} sources. Cite each source.",
                    min_sources
                )
            }
        };

        format!(
            "{}\n\n{}\n\nProvide a confidence score (0.0-1.0) for your response.",
            base, action_guidance
        )
    }

    /// Render template with input values
    ///
    /// PERFORMANCE: Uses static and cached regex patterns to avoid
    /// recompilation overhead (20-40% faster template rendering)
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

        // Replace previous output placeholders - handle nested field access
        for (key, output) in previous_outputs {
            // First, handle nested field access like {{step_id.field}}
            // Convert output to JSON for field extraction
            if let Ok(json_value) = serde_json::to_value(output) {
                // Use cached regex pattern instead of compiling each time
                let nested_re = get_nested_regex(key);
                result = nested_re
                    .replace_all(&result, |caps: &regex::Captures| {
                        let field = &caps[1];
                        // Try to extract the field from the JSON
                        if let Some(field_value) = json_value.get(field) {
                            match field_value {
                                serde_json::Value::String(s) => s.clone(),
                                serde_json::Value::Array(arr) => arr
                                    .iter()
                                    .filter_map(|v| v.as_str())
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                                other => other.to_string().trim_matches('"').to_string(),
                            }
                        } else {
                            // Field not found - return a helpful placeholder
                            format!("[{}.{}: not found]", key, field)
                        }
                    })
                    .to_string();
            }

            // Then handle full step output replacement ({{step_id}})
            let placeholder = format!("{{{{{}}}}}", key);
            let value_str = match output {
                StepOutput::Text { content } => content.clone(),
                StepOutput::List { items } => items
                    .iter()
                    .map(|i| i.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
                other => serde_json::to_string(other).unwrap_or_default(),
            };
            result = result.replace(&placeholder, &value_str);
        }

        // Clean up conditional blocks {{#if ...}}...{{/if}}
        // PERFORMANCE: Use static compiled regex instead of compiling each time
        result = CONDITIONAL_BLOCK_RE.replace_all(&result, "").to_string();

        // Clean up any remaining unfilled placeholders {{...}}
        // PERFORMANCE: Use static compiled regex instead of compiling each time
        if UNFILLED_PLACEHOLDER_RE.is_match(&result) {
            tracing::warn!(
                "Template has unfilled placeholders: {:?}",
                UNFILLED_PLACEHOLDER_RE
                    .find_iter(&result)
                    .map(|m| m.as_str())
                    .collect::<Vec<_>>()
            );
        }
        result = UNFILLED_PLACEHOLDER_RE.replace_all(&result, "").to_string();

        result
    }

    /// Real LLM call
    async fn real_llm_call(&self, prompt: &str, system: &str) -> Result<(String, TokenUsage)> {
        let client = self
            .llm_client
            .as_ref()
            .ok_or_else(|| Error::Config("LLM client not initialized".to_string()))?;

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

    /// CLI tool call (shells out to claude/codex/gemini/etc.)
    async fn cli_tool_call(&self, prompt: &str, system: &str) -> Result<(String, TokenUsage)> {
        let cli_config = self
            .config
            .cli_tool
            .as_ref()
            .ok_or_else(|| Error::Config("CLI tool not configured".to_string()))?;

        // Build the full prompt with system message
        let full_prompt = if system.is_empty() {
            prompt.to_string()
        } else {
            format!("{}\n\n{}", system, prompt)
        };

        // Build command with arguments
        let mut cmd = Command::new(&cli_config.command);

        // Add pre-args
        for arg in &cli_config.pre_args {
            cmd.arg(arg);
        }

        // Add the prompt
        cmd.arg(&full_prompt);

        // Add post-args
        for arg in &cli_config.post_args {
            cmd.arg(arg);
        }

        if self.config.verbose {
            tracing::info!(
                "CLI tool call: {} {} \"{}\"",
                cli_config.command,
                cli_config.pre_args.join(" "),
                if full_prompt.len() > 50 {
                    format!("{}...", &full_prompt[..50])
                } else {
                    full_prompt.clone()
                }
            );
        }

        // Execute the command
        let output = cmd.output().map_err(|e| {
            Error::Network(format!(
                "Failed to execute CLI tool '{}': {}",
                cli_config.command, e
            ))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Network(format!(
                "CLI tool '{}' failed with status {}: {}",
                cli_config.command, output.status, stderr
            )));
        }

        let content = String::from_utf8_lossy(&output.stdout).to_string();

        // Estimate tokens (rough approximation: ~4 chars per token for English)
        let input_tokens = (full_prompt.len() / 4) as u32;
        let output_tokens = (content.len() / 4) as u32;

        // Estimate cost based on CLI tool type
        // Prices as of Dec 2025 (per 1M tokens):
        // - Claude Sonnet 4: $3/1M input, $15/1M output
        // - Gemini 2.0 Flash: $0.10/1M input, $0.40/1M output
        // - GPT-4: $30/1M input, $60/1M output
        let (input_price, output_price) = match cli_config.command.as_str() {
            "claude" => (3.0, 15.0),              // Claude Sonnet 4 default
            "gemini" => (0.10, 0.40),             // Gemini 2.0 Flash
            "codex" | "opencode" => (30.0, 60.0), // GPT-4 level
            _ => (5.0, 15.0),                     // Conservative default
        };

        // Calculate cost in USD (prices are per 1M tokens)
        let cost_usd = (input_tokens as f64 * input_price / 1_000_000.0)
            + (output_tokens as f64 * output_price / 1_000_000.0);

        let tokens = TokenUsage::new(input_tokens, output_tokens, cost_usd);

        // Log estimated cost for transparency
        if self.config.verbose {
            tracing::info!(
                "CLI tool estimated: {} input + {} output tokens ≈ ${:.6}",
                input_tokens,
                output_tokens,
                cost_usd
            );
        }

        Ok((content, tokens))
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
            StepAction::Analyze { .. } | StepAction::Synthesize { .. } => StepOutput::Text {
                content: content.to_string(),
            },
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
    /// Handles multiple formats: numbered (1. 2. 10.), bullets (- *), bold headers (**item**)
    fn extract_list_items(&self, content: &str) -> Vec<ListItem> {
        use once_cell::sync::Lazy;

        // Compile regexes once (static lifetime)
        static NUMBERED_RE: Lazy<regex::Regex> =
            Lazy::new(|| regex::Regex::new(r"^\d+[\.\)]\s*(.+)$").expect("Invalid regex pattern"));
        static BOLD_RE: Lazy<regex::Regex> = Lazy::new(|| {
            regex::Regex::new(r"^\*\*([^*]+)\*\*[:\s-]*(.*)$").expect("Invalid regex pattern")
        });

        let mut items = Vec::new();
        let mut current_item: Option<String> = None;

        for line in content.lines() {
            let trimmed = line.trim();

            // Skip empty lines and confidence markers
            if trimmed.is_empty() || trimmed.to_lowercase().starts_with("confidence") {
                // If we have a current item being built, save it
                if let Some(item) = current_item.take() {
                    if !item.is_empty() {
                        items.push(ListItem::new(item));
                    }
                }
                continue;
            }

            // Try to match numbered items (1. 2. ... 10. 11. etc)
            if let Some(caps) = NUMBERED_RE.captures(trimmed) {
                // Save previous item if exists
                if let Some(item) = current_item.take() {
                    if !item.is_empty() {
                        items.push(ListItem::new(item));
                    }
                }
                current_item = Some(caps[1].to_string());
                continue;
            }

            // Match bullet points (- * •)
            if let Some(text) = trimmed
                .strip_prefix('-')
                .or(trimmed.strip_prefix('*'))
                .or(trimmed.strip_prefix('•'))
            {
                let text = text.trim();
                if !text.is_empty() {
                    // Save previous item if exists
                    if let Some(item) = current_item.take() {
                        if !item.is_empty() {
                            items.push(ListItem::new(item));
                        }
                    }
                    current_item = Some(text.to_string());
                    continue;
                }
            }

            // Match bold headers: **Title**: Description or **Title** - Description
            if let Some(caps) = BOLD_RE.captures(trimmed) {
                // Save previous item if exists
                if let Some(item) = current_item.take() {
                    if !item.is_empty() {
                        items.push(ListItem::new(item));
                    }
                }
                let title = caps[1].trim();
                let desc = caps[2].trim();
                if desc.is_empty() {
                    current_item = Some(title.to_string());
                } else {
                    current_item = Some(format!("{}: {}", title, desc));
                }
                continue;
            }

            // If line starts with indentation and we have a current item, append to it
            if line.starts_with("  ") || line.starts_with("\t") {
                if let Some(ref mut item) = current_item {
                    item.push(' ');
                    item.push_str(trimmed);
                    continue;
                }
            }

            // Otherwise, if it's a continuation of the previous item or standalone text
            if let Some(ref mut item) = current_item {
                // Continuation of multi-line item
                item.push(' ');
                item.push_str(trimmed);
            }
            // Skip standalone lines that aren't list items
        }

        // Don't forget the last item
        if let Some(item) = current_item {
            if !item.is_empty() {
                items.push(ListItem::new(item));
            }
        }

        items
    }

    /// Save trace to file
    fn save_trace(&self, trace: &ExecutionTrace) -> Result<()> {
        let dir = self
            .config
            .trace_dir
            .as_ref()
            .ok_or_else(|| Error::Config("Trace directory not configured".to_string()))?;

        std::fs::create_dir_all(dir).map_err(|e| Error::IoMessage {
            message: format!("Failed to create trace directory: {}", e),
        })?;

        let filename = format!("{}_{}.json", trace.protocol_id, trace.id);
        let path = dir.join(filename);

        let json = trace.to_json().map_err(|e| Error::Parse {
            message: format!("Failed to serialize trace: {}", e),
        })?;

        std::fs::write(&path, json).map_err(|e| Error::IoMessage {
            message: format!("Failed to write trace: {}", e),
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

// ═══════════════════════════════════════════════════════════════════════════════
// COMPREHENSIVE TEST SUITE
// ═══════════════════════════════════════════════════════════════════════════════
//
// Test Coverage:
// 1. Executor Creation and Configuration
// 2. Single Module Execution (GigaThink, LaserLogic, BedRock, ProofGuard, BrutalHonesty)
// 3. PowerCombo Chain Execution (all 5 ThinkTools)
// 4. Timeout Handling
// 5. Error Recovery
// 6. Trace Generation
// 7. Template Rendering
// 8. Confidence Extraction
// 9. List Item Parsing
// 10. Branch Condition Evaluation
// 11. Dependency Management
// 12. Budget Tracking
// 13. Parallel Execution
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. EXECUTOR CREATION AND CONFIGURATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_executor_creation() {
        let executor = ProtocolExecutor::mock().unwrap();
        assert!(!executor.registry().is_empty());
        assert!(!executor.profiles().is_empty());
    }

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert_eq!(config.timeout_secs, 120);
        assert!(!config.use_mock);
        assert!(!config.save_traces);
        assert!(config.show_progress);
        assert!(!config.enable_parallel);
        assert_eq!(config.max_concurrent_steps, 4);
    }

    #[test]
    fn test_executor_config_mock() {
        let config = ExecutorConfig::mock();
        assert!(config.use_mock);
    }

    #[test]
    fn test_executor_config_parallel() {
        let config = ExecutorConfig::default().with_parallel();
        assert!(config.enable_parallel);
        assert_eq!(config.max_concurrent_steps, 4);

        let config_limited = ExecutorConfig::default().with_parallel_limit(2);
        assert!(config_limited.enable_parallel);
        assert_eq!(config_limited.max_concurrent_steps, 2);
    }

    #[test]
    fn test_executor_config_self_consistency() {
        let config = ExecutorConfig::default().with_self_consistency();
        assert!(config.self_consistency.is_some());

        let config_fast = ExecutorConfig::default().with_self_consistency_fast();
        assert!(config_fast.self_consistency.is_some());
        assert_eq!(config_fast.self_consistency.unwrap().num_samples, 3);

        let config_thorough = ExecutorConfig::default().with_self_consistency_thorough();
        assert!(config_thorough.self_consistency.is_some());
        assert_eq!(config_thorough.self_consistency.unwrap().num_samples, 10);
    }

    #[test]
    fn test_list_protocols() {
        let executor = ProtocolExecutor::mock().unwrap();
        let protocols = executor.list_protocols();
        assert!(protocols.contains(&"gigathink"));
        assert!(protocols.contains(&"laserlogic"));
        assert!(protocols.contains(&"bedrock"));
        assert!(protocols.contains(&"proofguard"));
        assert!(protocols.contains(&"brutalhonesty"));
    }

    #[test]
    fn test_list_profiles() {
        let executor = ProtocolExecutor::mock().unwrap();
        let profiles = executor.list_profiles();
        assert!(profiles.contains(&"quick"));
        assert!(profiles.contains(&"balanced"));
        assert!(profiles.contains(&"deep"));
        assert!(profiles.contains(&"paranoid"));
        assert!(profiles.contains(&"powercombo"));
    }

    #[test]
    fn test_get_protocol() {
        let executor = ProtocolExecutor::mock().unwrap();
        let gigathink = executor.get_protocol("gigathink");
        assert!(gigathink.is_some());
        assert_eq!(gigathink.unwrap().id, "gigathink");

        let nonexistent = executor.get_protocol("nonexistent_protocol");
        assert!(nonexistent.is_none());
    }

    #[test]
    fn test_get_profile() {
        let executor = ProtocolExecutor::mock().unwrap();
        let quick = executor.get_profile("quick");
        assert!(quick.is_some());
        assert_eq!(quick.unwrap().id, "quick");

        let nonexistent = executor.get_profile("nonexistent_profile");
        assert!(nonexistent.is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. PROTOCOL INPUT TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_protocol_input_query() {
        let input = ProtocolInput::query("Test query");
        assert_eq!(input.get_str("query"), Some("Test query"));
    }

    #[test]
    fn test_protocol_input_argument() {
        let input = ProtocolInput::argument("Test argument");
        assert_eq!(input.get_str("argument"), Some("Test argument"));
    }

    #[test]
    fn test_protocol_input_statement() {
        let input = ProtocolInput::statement("Test statement");
        assert_eq!(input.get_str("statement"), Some("Test statement"));
    }

    #[test]
    fn test_protocol_input_claim() {
        let input = ProtocolInput::claim("Test claim");
        assert_eq!(input.get_str("claim"), Some("Test claim"));
    }

    #[test]
    fn test_protocol_input_work() {
        let input = ProtocolInput::work("Test work");
        assert_eq!(input.get_str("work"), Some("Test work"));
    }

    #[test]
    fn test_protocol_input_with_field() {
        let input = ProtocolInput::query("Test query")
            .with_field("context", "Some context")
            .with_field("domain", "AI");

        assert_eq!(input.get_str("query"), Some("Test query"));
        assert_eq!(input.get_str("context"), Some("Some context"));
        assert_eq!(input.get_str("domain"), Some("AI"));
    }

    #[test]
    fn test_protocol_input_missing_field() {
        let input = ProtocolInput::query("Test query");
        assert_eq!(input.get_str("nonexistent"), None);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. TEMPLATE RENDERING TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_template_rendering_simple() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What is AI?");

        let template = "Question: {{query}}";
        let rendered = executor.render_template(template, &input, &HashMap::new());

        assert_eq!(rendered, "Question: What is AI?");
    }

    #[test]
    fn test_template_rendering_multiple_fields() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What is AI?").with_field("context", "machine learning");

        let template = "Question: {{query}}\nContext: {{context}}";
        let rendered = executor.render_template(template, &input, &HashMap::new());

        assert_eq!(rendered, "Question: What is AI?\nContext: machine learning");
    }

    #[test]
    fn test_template_rendering_with_previous_outputs() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let mut previous_outputs = HashMap::new();
        previous_outputs.insert(
            "step1".to_string(),
            StepOutput::Text {
                content: "Previous output".to_string(),
            },
        );

        let template = "Input: {{query}}\nPrevious: {{step1}}";
        let rendered = executor.render_template(template, &input, &previous_outputs);

        assert_eq!(rendered, "Input: Test\nPrevious: Previous output");
    }

    #[test]
    fn test_template_rendering_list_output() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let mut previous_outputs = HashMap::new();
        previous_outputs.insert(
            "ideas".to_string(),
            StepOutput::List {
                items: vec![
                    ListItem::new("Idea 1"),
                    ListItem::new("Idea 2"),
                    ListItem::new("Idea 3"),
                ],
            },
        );

        let template = "Ideas:\n{{ideas}}";
        let rendered = executor.render_template(template, &input, &previous_outputs);

        assert!(rendered.contains("Idea 1"));
        assert!(rendered.contains("Idea 2"));
        assert!(rendered.contains("Idea 3"));
    }

    #[test]
    fn test_template_rendering_unfilled_placeholders_removed() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let template = "Question: {{query}}\nOptional: {{optional_field}}";
        let rendered = executor.render_template(template, &input, &HashMap::new());

        assert_eq!(rendered, "Question: Test\nOptional: ");
    }

    #[test]
    fn test_template_static_rendering() {
        let input = ProtocolInput::query("Test query");
        let previous_outputs = HashMap::new();

        let template = "Question: {{query}}";
        let rendered = ProtocolExecutor::render_template_static(template, &input, &previous_outputs);

        assert_eq!(rendered, "Question: Test query");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. CONFIDENCE EXTRACTION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_extract_confidence_standard_format() {
        let executor = ProtocolExecutor::mock().unwrap();

        assert_eq!(executor.extract_confidence("Confidence: 0.85"), Some(0.85));
        assert_eq!(executor.extract_confidence("confidence: 0.9"), Some(0.9));
        assert_eq!(executor.extract_confidence("Confidence 0.75"), Some(0.75));
    }

    #[test]
    fn test_extract_confidence_multiline() {
        let executor = ProtocolExecutor::mock().unwrap();

        let content = "Some analysis text\nMore text\nConfidence: 0.75";
        assert_eq!(executor.extract_confidence(content), Some(0.75));
    }

    #[test]
    fn test_extract_confidence_integer() {
        let executor = ProtocolExecutor::mock().unwrap();

        // Integer values should be capped at 1.0
        assert_eq!(executor.extract_confidence("Confidence: 95"), Some(1.0));
    }

    #[test]
    fn test_extract_confidence_missing() {
        let executor = ProtocolExecutor::mock().unwrap();

        assert_eq!(executor.extract_confidence("No confidence here"), None);
        assert_eq!(executor.extract_confidence(""), None);
    }

    #[test]
    fn test_extract_confidence_static() {
        assert_eq!(
            ProtocolExecutor::extract_confidence_static("Confidence: 0.88"),
            Some(0.88)
        );
        assert_eq!(
            ProtocolExecutor::extract_confidence_static("confidence 0.72"),
            Some(0.72)
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 5. LIST ITEM EXTRACTION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_extract_list_items_numbered() {
        let executor = ProtocolExecutor::mock().unwrap();

        let content = "1. First item\n2. Second item\n3. Third item\nConfidence: 0.8";
        let items = executor.extract_list_items(content);

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].content, "First item");
        assert_eq!(items[1].content, "Second item");
        assert_eq!(items[2].content, "Third item");
    }

    #[test]
    fn test_extract_list_items_bulleted() {
        let executor = ProtocolExecutor::mock().unwrap();

        let content = "- First item\n- Second item\n- Third item";
        let items = executor.extract_list_items(content);

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].content, "First item");
    }

    #[test]
    fn test_extract_list_items_mixed() {
        let executor = ProtocolExecutor::mock().unwrap();

        let content = "1. First item\n2. Second item\n- Third item\nConfidence: 0.8";
        let items = executor.extract_list_items(content);

        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_extract_list_items_with_bold() {
        let executor = ProtocolExecutor::mock().unwrap();

        let content = "**Title**: Description here\n**Another**: More text";
        let items = executor.extract_list_items(content);

        assert_eq!(items.len(), 2);
        assert!(items[0].content.contains("Title"));
    }

    #[test]
    fn test_extract_list_items_multiline() {
        let executor = ProtocolExecutor::mock().unwrap();

        let content = "1. First item with\n   continuation on next line\n2. Second item";
        let items = executor.extract_list_items(content);

        assert_eq!(items.len(), 2);
        assert!(items[0].content.contains("continuation"));
    }

    #[test]
    fn test_extract_list_items_empty() {
        let executor = ProtocolExecutor::mock().unwrap();

        let content = "No list items here\nJust plain text";
        let items = executor.extract_list_items(content);

        assert!(items.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 6. SINGLE MODULE EXECUTION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_execute_gigathink_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What are the key factors for startup success?");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
        assert!(!result.steps.is_empty());
        assert_eq!(result.protocol_id, "gigathink");
        assert!(result.duration_ms > 0);
    }

    #[tokio::test]
    async fn test_execute_laserlogic_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::argument("All humans are mortal. Socrates is human. Therefore, Socrates is mortal.");

        let result = executor.execute("laserlogic", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_execute_bedrock_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::statement("The Earth revolves around the Sun.");

        let result = executor.execute("bedrock", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_execute_proofguard_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::claim("Climate change is caused by human activities.");

        let result = executor.execute("proofguard", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_execute_brutalhonesty_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::work("My analysis concludes that AI will solve all problems.");

        let result = executor.execute("brutalhonesty", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 7. POWERCOMBO CHAIN EXECUTION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_execute_profile_quick_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Should we adopt microservices?");

        let result = executor.execute_profile("quick", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_execute_profile_balanced_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What is the future of AI in healthcare?");

        let result = executor.execute_profile("balanced", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_execute_profile_powercombo_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Analyze the impact of quantum computing on cryptography.");

        let result = executor.execute_profile("powercombo", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
        // PowerCombo should produce many steps from all 5 ThinkTools
        assert!(result.steps.len() >= 5);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 8. ERROR HANDLING AND RECOVERY TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_execute_nonexistent_protocol() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let result = executor.execute("nonexistent_protocol", input).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::NotFound { .. }));
    }

    #[tokio::test]
    async fn test_execute_nonexistent_profile() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let result = executor.execute_profile("nonexistent_profile", input).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_missing_required_input() {
        let executor = ProtocolExecutor::mock().unwrap();
        // GigaThink requires "query" field, but we provide "argument"
        let input = ProtocolInput::argument("Wrong field type");

        let result = executor.execute("gigathink", input).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 9. TRACE GENERATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_trace_generation_basic() {
        // Create executor with trace saving enabled (but to temp dir)
        let config = ExecutorConfig {
            use_mock: true,
            save_traces: true,
            trace_dir: Some(std::env::temp_dir().join("reasonkit_test_traces")),
            show_progress: false,
            ..Default::default()
        };
        let executor = ProtocolExecutor::with_config(config).unwrap();
        let input = ProtocolInput::query("Test trace generation");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.trace_id.is_some());
    }

    #[test]
    fn test_execution_trace_creation() {
        let trace = ExecutionTrace::new("test_protocol", "1.0.0");

        assert_eq!(trace.protocol_id, "test_protocol");
        assert_eq!(trace.protocol_version, "1.0.0");
        assert!(trace.steps.is_empty());
        assert_eq!(trace.status, crate::thinktool::trace::ExecutionStatus::Running);
    }

    #[test]
    fn test_step_trace_creation() {
        let mut step_trace = StepTrace::new("step1", 0);

        assert_eq!(step_trace.step_id, "step1");
        assert_eq!(step_trace.index, 0);
        assert_eq!(step_trace.status, StepStatus::Pending);

        step_trace.complete(
            StepOutput::Text {
                content: "Output".to_string(),
            },
            0.85,
        );

        assert_eq!(step_trace.status, StepStatus::Completed);
        assert_eq!(step_trace.confidence, 0.85);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 10. BRANCH CONDITION EVALUATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_branch_condition_always() {
        let executor = ProtocolExecutor::mock().unwrap();
        let condition = BranchCondition::Always;
        let results: Vec<StepResult> = vec![];

        assert!(executor.evaluate_branch_condition(&condition, &results));
    }

    #[test]
    fn test_branch_condition_confidence_below() {
        let executor = ProtocolExecutor::mock().unwrap();
        let condition = BranchCondition::ConfidenceBelow { threshold: 0.8 };

        // With empty results, should return true (no confidence to check)
        let empty_results: Vec<StepResult> = vec![];
        assert!(executor.evaluate_branch_condition(&condition, &empty_results));

        // With low confidence result, should return true
        let low_conf_results = vec![StepResult::success(
            "step1",
            StepOutput::Text {
                content: "test".to_string(),
            },
            0.5,
        )];
        assert!(executor.evaluate_branch_condition(&condition, &low_conf_results));

        // With high confidence result, should return false
        let high_conf_results = vec![StepResult::success(
            "step1",
            StepOutput::Text {
                content: "test".to_string(),
            },
            0.9,
        )];
        assert!(!executor.evaluate_branch_condition(&condition, &high_conf_results));
    }

    #[test]
    fn test_branch_condition_confidence_above() {
        let executor = ProtocolExecutor::mock().unwrap();
        let condition = BranchCondition::ConfidenceAbove { threshold: 0.8 };

        // With high confidence result, should return true
        let high_conf_results = vec![StepResult::success(
            "step1",
            StepOutput::Text {
                content: "test".to_string(),
            },
            0.9,
        )];
        assert!(executor.evaluate_branch_condition(&condition, &high_conf_results));

        // With low confidence result, should return false
        let low_conf_results = vec![StepResult::success(
            "step1",
            StepOutput::Text {
                content: "test".to_string(),
            },
            0.5,
        )];
        assert!(!executor.evaluate_branch_condition(&condition, &low_conf_results));
    }

    #[test]
    fn test_branch_condition_output_equals() {
        let executor = ProtocolExecutor::mock().unwrap();
        let condition = BranchCondition::OutputEquals {
            field: "result".to_string(),
            value: "PASS".to_string(),
        };

        // With matching output
        let matching_results = vec![StepResult::success(
            "step1",
            StepOutput::Text {
                content: "Result: PASS".to_string(),
            },
            0.9,
        )];
        assert!(executor.evaluate_branch_condition(&condition, &matching_results));

        // With non-matching output
        let non_matching_results = vec![StepResult::success(
            "step1",
            StepOutput::Text {
                content: "Result: FAIL".to_string(),
            },
            0.9,
        )];
        assert!(!executor.evaluate_branch_condition(&condition, &non_matching_results));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 11. DEPENDENCY MANAGEMENT TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dependencies_met_empty() {
        let executor = ProtocolExecutor::mock().unwrap();
        let deps: Vec<String> = vec![];
        let results: Vec<StepResult> = vec![];

        assert!(executor.dependencies_met(&deps, &results));
    }

    #[test]
    fn test_dependencies_met_satisfied() {
        let executor = ProtocolExecutor::mock().unwrap();
        let deps = vec!["step1".to_string(), "step2".to_string()];
        let results = vec![
            StepResult::success(
                "step1",
                StepOutput::Text {
                    content: "".to_string(),
                },
                0.9,
            ),
            StepResult::success(
                "step2",
                StepOutput::Text {
                    content: "".to_string(),
                },
                0.8,
            ),
        ];

        assert!(executor.dependencies_met(&deps, &results));
    }

    #[test]
    fn test_dependencies_met_unsatisfied() {
        let executor = ProtocolExecutor::mock().unwrap();
        let deps = vec!["step1".to_string(), "step2".to_string()];
        let results = vec![StepResult::success(
            "step1",
            StepOutput::Text {
                content: "".to_string(),
            },
            0.9,
        )];

        assert!(!executor.dependencies_met(&deps, &results));
    }

    #[test]
    fn test_dependencies_met_failed_step() {
        let executor = ProtocolExecutor::mock().unwrap();
        let deps = vec!["step1".to_string()];
        let results = vec![StepResult::failure("step1", "Some error")];

        assert!(!executor.dependencies_met(&deps, &results));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 12. BUDGET TRACKING TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_execution_with_token_budget() {
        let config = ExecutorConfig {
            use_mock: true,
            budget: BudgetConfig::with_tokens(10000),
            show_progress: false,
            ..Default::default()
        };
        let executor = ProtocolExecutor::with_config(config).unwrap();
        let input = ProtocolInput::query("Test budget tracking");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.success);
        assert!(result.budget_summary.is_some());
        let summary = result.budget_summary.unwrap();
        assert!(summary.tokens_used > 0);
    }

    #[tokio::test]
    async fn test_execution_with_cost_budget() {
        let config = ExecutorConfig {
            use_mock: true,
            budget: BudgetConfig::with_cost(1.0),
            show_progress: false,
            ..Default::default()
        };
        let executor = ProtocolExecutor::with_config(config).unwrap();
        let input = ProtocolInput::query("Test cost budget");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.success);
        assert!(result.budget_summary.is_some());
    }

    #[test]
    fn test_budget_config_parsing() {
        // Time budget
        let time_budget = BudgetConfig::parse("30s").unwrap();
        assert_eq!(time_budget.time_limit, Some(Duration::from_secs(30)));

        let min_budget = BudgetConfig::parse("5m").unwrap();
        assert_eq!(min_budget.time_limit, Some(Duration::from_secs(300)));

        // Token budget
        let token_budget = BudgetConfig::parse("1000t").unwrap();
        assert_eq!(token_budget.token_limit, Some(1000));

        // Cost budget
        let cost_budget = BudgetConfig::parse("$0.50").unwrap();
        assert_eq!(cost_budget.cost_limit, Some(0.50));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 13. PARALLEL EXECUTION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn test_parallel_execution_mock() {
        let config = ExecutorConfig {
            use_mock: true,
            enable_parallel: true,
            max_concurrent_steps: 4,
            show_progress: false,
            ..Default::default()
        };
        let executor = ProtocolExecutor::with_config(config).unwrap();
        let input = ProtocolInput::query("Test parallel execution");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.success);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_parallel_execution_with_limit() {
        let config = ExecutorConfig {
            use_mock: true,
            enable_parallel: true,
            max_concurrent_steps: 2,
            show_progress: false,
            ..Default::default()
        };
        let executor = ProtocolExecutor::with_config(config).unwrap();
        let input = ProtocolInput::query("Test parallel with limit");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.success);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 14. CLI TOOL CONFIG TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cli_tool_config_claude() {
        let config = CliToolConfig::claude();
        assert_eq!(config.command, "claude");
        assert!(config.pre_args.contains(&"-p".to_string()));
        assert!(!config.interactive);
    }

    #[test]
    fn test_cli_tool_config_codex() {
        let config = CliToolConfig::codex();
        assert_eq!(config.command, "codex");
        assert!(config.pre_args.contains(&"-q".to_string()));
    }

    #[test]
    fn test_cli_tool_config_gemini() {
        let config = CliToolConfig::gemini();
        assert_eq!(config.command, "gemini");
        assert!(config.pre_args.contains(&"-p".to_string()));
    }

    #[test]
    fn test_cli_tool_config_copilot() {
        let config = CliToolConfig::copilot();
        assert_eq!(config.command, "gh");
        assert!(config.pre_args.contains(&"copilot".to_string()));
        assert!(config.interactive);
    }

    #[test]
    fn test_executor_config_cli() {
        let config = ExecutorConfig::claude_cli();
        assert!(config.cli_tool.is_some());
        assert_eq!(config.cli_tool.unwrap().command, "claude");

        let config = ExecutorConfig::gemini_cli();
        assert!(config.cli_tool.is_some());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 15. PROTOCOL OUTPUT TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_protocol_output_get() {
        let mut data = HashMap::new();
        data.insert("key1".to_string(), serde_json::json!("value1"));
        data.insert("key2".to_string(), serde_json::json!(42));

        let output = ProtocolOutput {
            protocol_id: "test".to_string(),
            success: true,
            data,
            confidence: 0.85,
            steps: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 100,
            error: None,
            trace_id: None,
            budget_summary: None,
        };

        assert_eq!(output.get("key1"), Some(&serde_json::json!("value1")));
        assert_eq!(output.get("key2"), Some(&serde_json::json!(42)));
        assert_eq!(output.get("nonexistent"), None);
    }

    #[test]
    fn test_protocol_output_verdict() {
        let mut data = HashMap::new();
        data.insert("verdict".to_string(), serde_json::json!("VALID"));

        let output = ProtocolOutput {
            protocol_id: "test".to_string(),
            success: true,
            data,
            confidence: 0.85,
            steps: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 100,
            error: None,
            trace_id: None,
            budget_summary: None,
        };

        assert_eq!(output.verdict(), Some("VALID"));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 16. CHAIN CONDITION EVALUATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_chain_condition_always() {
        let executor = ProtocolExecutor::mock().unwrap();
        let condition = ChainCondition::Always;
        let results: Vec<StepResult> = vec![];

        assert!(executor.evaluate_chain_condition(&condition, &results));
    }

    #[test]
    fn test_chain_condition_confidence_below() {
        let executor = ProtocolExecutor::mock().unwrap();
        let condition = ChainCondition::ConfidenceBelow { threshold: 0.8 };

        let low_conf_results = vec![StepResult::success(
            "step1",
            StepOutput::Text {
                content: "test".to_string(),
            },
            0.5,
        )];
        assert!(executor.evaluate_chain_condition(&condition, &low_conf_results));
    }

    #[test]
    fn test_chain_condition_output_exists() {
        let executor = ProtocolExecutor::mock().unwrap();
        let condition = ChainCondition::OutputExists {
            step_id: "step1".to_string(),
            field: "result".to_string(),
        };

        let results = vec![StepResult::success(
            "step1",
            StepOutput::Text {
                content: "output".to_string(),
            },
            0.9,
        )];
        assert!(executor.evaluate_chain_condition(&condition, &results));

        let empty_results: Vec<StepResult> = vec![];
        assert!(!executor.evaluate_chain_condition(&condition, &empty_results));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 17. MAPPING RESOLUTION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_resolve_mapping_input() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test query");
        let step_outputs: HashMap<String, serde_json::Value> = HashMap::new();

        let result = executor.resolve_mapping("input.query", &step_outputs, &input);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), serde_json::json!("Test query"));
    }

    #[test]
    fn test_resolve_mapping_missing_input() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test query");
        let step_outputs: HashMap<String, serde_json::Value> = HashMap::new();

        let result = executor.resolve_mapping("input.nonexistent", &step_outputs, &input);
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_mapping_step_output() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");
        let mut step_outputs: HashMap<String, serde_json::Value> = HashMap::new();
        step_outputs.insert(
            "steps.gigathink".to_string(),
            serde_json::json!({
                "result": "some output",
                "confidence": 0.85
            }),
        );

        let result = executor.resolve_mapping("steps.gigathink", &step_outputs, &input);
        assert!(result.is_some());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 18. TOKEN USAGE TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_token_usage_creation() {
        let usage = TokenUsage::new(100, 50, 0.001);

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.cost_usd, 0.001);
    }

    #[test]
    fn test_token_usage_add() {
        let mut usage1 = TokenUsage::new(100, 50, 0.001);
        let usage2 = TokenUsage::new(200, 100, 0.002);

        usage1.add(&usage2);

        assert_eq!(usage1.input_tokens, 300);
        assert_eq!(usage1.output_tokens, 150);
        assert_eq!(usage1.total_tokens, 450);
        assert_eq!(usage1.cost_usd, 0.003);
    }

    #[tokio::test]
    async fn test_execution_accumulates_tokens() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test token accumulation");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.tokens.total_tokens > 0);
        assert!(result.tokens.input_tokens > 0);
        assert!(result.tokens.output_tokens > 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 19. SYSTEM PROMPT GENERATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_build_system_prompt_generate() {
        let step = ProtocolStep {
            id: "test".to_string(),
            action: StepAction::Generate {
                min_count: 5,
                max_count: 10,
            },
            prompt_template: "".to_string(),
            output_format: crate::thinktool::protocol::StepOutputFormat::List,
            min_confidence: 0.7,
            depends_on: vec![],
            branch: None,
        };

        let prompt = ProtocolExecutor::build_system_prompt_static(&step);
        assert!(prompt.contains("Generate"));
        assert!(prompt.contains("confidence"));
    }

    #[test]
    fn test_build_system_prompt_analyze() {
        let step = ProtocolStep {
            id: "test".to_string(),
            action: StepAction::Analyze {
                criteria: vec!["accuracy".to_string()],
            },
            prompt_template: "".to_string(),
            output_format: crate::thinktool::protocol::StepOutputFormat::Text,
            min_confidence: 0.7,
            depends_on: vec![],
            branch: None,
        };

        let prompt = ProtocolExecutor::build_system_prompt_static(&step);
        assert!(prompt.contains("Analyze"));
    }

    #[test]
    fn test_build_system_prompt_validate() {
        let step = ProtocolStep {
            id: "test".to_string(),
            action: StepAction::Validate {
                rules: vec!["rule1".to_string()],
            },
            prompt_template: "".to_string(),
            output_format: crate::thinktool::protocol::StepOutputFormat::Boolean,
            min_confidence: 0.7,
            depends_on: vec![],
            branch: None,
        };

        let prompt = ProtocolExecutor::build_system_prompt_static(&step);
        assert!(prompt.contains("Validate"));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 20. EDGE CASES AND STRESS TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_empty_template() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let rendered = executor.render_template("", &input, &HashMap::new());
        assert_eq!(rendered, "");
    }

    #[test]
    fn test_template_with_special_characters() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test with \"quotes\" and 'apostrophes'");

        let template = "Query: {{query}}";
        let rendered = executor.render_template(template, &input, &HashMap::new());
        assert!(rendered.contains("\"quotes\""));
        assert!(rendered.contains("'apostrophes'"));
    }

    #[test]
    fn test_confidence_extraction_edge_cases() {
        let executor = ProtocolExecutor::mock().unwrap();

        // Very small confidence
        assert_eq!(
            executor.extract_confidence("Confidence: 0.001"),
            Some(0.001)
        );

        // Confidence at boundary
        assert_eq!(executor.extract_confidence("Confidence: 1.0"), Some(1.0));
        assert_eq!(executor.extract_confidence("Confidence: 0.0"), Some(0.0));
    }

    #[tokio::test]
    async fn test_execution_with_very_long_input() {
        let executor = ProtocolExecutor::mock().unwrap();
        let long_query = "A".repeat(10000);
        let input = ProtocolInput::query(long_query);

        let result = executor.execute("gigathink", input).await.unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_list_extraction_with_many_items() {
        let executor = ProtocolExecutor::mock().unwrap();

        let mut content = String::new();
        for i in 1..=50 {
            content.push_str(&format!("{}. Item number {}\n", i, i));
        }
        content.push_str("Confidence: 0.85");

        let items = executor.extract_list_items(&content);
        assert_eq!(items.len(), 50);
    }
}
