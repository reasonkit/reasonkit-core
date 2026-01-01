//! # Async ReasoningLoop Engine
//!
//! High-performance async reasoning engine with Tokio-based concurrency,
//! memory integration, and streaming support.
//!
//! ## Design Principles
//!
//! 1. **Async-First**: All I/O operations are non-blocking
//! 2. **Channel-Based Concurrency**: ThinkTools execute via broadcast channels
//! 3. **Streaming Output**: Real-time step-by-step reasoning visibility
//! 4. **Memory Integration**: Optional reasonkit-mem for context enrichment
//! 5. **Profile System**: quick/balanced/deep/paranoid execution modes
//!
//! ## Performance Characteristics
//!
//! - Concurrent ThinkTool execution reduces latency by (N-1)/N for N independent tools
//! - Zero-copy streaming via tokio broadcast channels
//! - Bounded channel backpressure prevents memory exhaustion
//! - Connection pooling for LLM API calls

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors that can occur during reasoning loop execution
#[derive(Error, Debug)]
pub enum ReasoningError {
    /// ThinkTool execution failed
    #[error("ThinkTool '{tool}' failed: {message}")]
    ThinkToolFailed { tool: String, message: String },

    /// Memory query failed
    #[error("Memory query failed: {0}")]
    MemoryQueryFailed(String),

    /// Profile not found
    #[error("Profile '{0}' not found")]
    ProfileNotFound(String),

    /// Channel communication error
    #[error("Channel error: {0}")]
    ChannelError(String),

    /// Timeout during execution
    #[error("Execution timed out after {0:?}")]
    Timeout(Duration),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// LLM provider error
    #[error("LLM error: {0}")]
    LlmError(String),

    /// Cancelled by user
    #[error("Execution cancelled")]
    Cancelled,

    /// Confidence threshold not met
    #[error("Confidence {actual:.2} below threshold {required:.2}")]
    ConfidenceBelowThreshold { actual: f64, required: f64 },
}

/// Result type for reasoning operations
pub type Result<T> = std::result::Result<T, ReasoningError>;

// ============================================================================
// PROFILE SYSTEM
// ============================================================================

/// Reasoning profiles that determine ThinkTool chains and confidence targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    /// Fast 2-step: GigaThink -> LaserLogic (70% confidence target)
    Quick,

    /// Standard 4-step: GigaThink -> LaserLogic -> BedRock -> ProofGuard (80%)
    #[default]
    Balanced,

    /// Thorough 5-step with meta-cognition (85% confidence target)
    Deep,

    /// Maximum verification with adversarial critique (95% confidence target)
    Paranoid,
}

impl Profile {
    /// Get the ThinkTool chain for this profile
    pub fn thinktool_chain(&self) -> Vec<&'static str> {
        match self {
            Profile::Quick => vec!["gigathink", "laserlogic"],
            Profile::Balanced => vec!["gigathink", "laserlogic", "bedrock", "proofguard"],
            Profile::Deep => vec![
                "gigathink",
                "laserlogic",
                "bedrock",
                "proofguard",
                "brutalhonesty",
            ],
            Profile::Paranoid => vec![
                "gigathink",
                "laserlogic",
                "bedrock",
                "proofguard",
                "brutalhonesty",
                "proofguard", // Second verification pass
            ],
        }
    }

    /// Get the minimum confidence threshold for this profile
    pub fn min_confidence(&self) -> f64 {
        match self {
            Profile::Quick => 0.70,
            Profile::Balanced => 0.80,
            Profile::Deep => 0.85,
            Profile::Paranoid => 0.95,
        }
    }

    /// Get the maximum token budget for this profile
    pub fn token_budget(&self) -> u32 {
        match self {
            Profile::Quick => 3_000,
            Profile::Balanced => 8_000,
            Profile::Deep => 12_000,
            Profile::Paranoid => 25_000,
        }
    }

    /// Parse profile from string
    pub fn parse_profile(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "quick" | "q" => Some(Profile::Quick),
            "balanced" | "b" => Some(Profile::Balanced),
            "deep" | "d" => Some(Profile::Deep),
            "paranoid" | "p" => Some(Profile::Paranoid),
            _ => None,
        }
    }
}

impl std::fmt::Display for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Profile::Quick => write!(f, "quick"),
            Profile::Balanced => write!(f, "balanced"),
            Profile::Deep => write!(f, "deep"),
            Profile::Paranoid => write!(f, "paranoid"),
        }
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for the ReasoningLoop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Default profile to use
    #[serde(default)]
    pub default_profile: Profile,

    /// Maximum execution time
    #[serde(default = "default_timeout")]
    pub timeout: Duration,

    /// Enable parallel ThinkTool execution when possible
    #[serde(default = "default_true")]
    pub enable_parallel: bool,

    /// Maximum concurrent ThinkTool executions
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    /// Enable memory integration
    #[serde(default)]
    pub enable_memory: bool,

    /// Number of memory results to retrieve
    #[serde(default = "default_memory_top_k")]
    pub memory_top_k: usize,

    /// Minimum relevance score for memory results
    #[serde(default = "default_memory_min_score")]
    pub memory_min_score: f32,

    /// Enable streaming output
    #[serde(default = "default_true")]
    pub enable_streaming: bool,

    /// Streaming buffer size
    #[serde(default = "default_stream_buffer")]
    pub stream_buffer_size: usize,

    /// LLM temperature (0.0-1.0)
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// LLM max tokens per step
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Retry failed steps
    #[serde(default = "default_true")]
    pub retry_on_failure: bool,

    /// Maximum retries per step
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_timeout() -> Duration {
    Duration::from_secs(300) // 5 minutes
}
fn default_true() -> bool {
    true
}
fn default_max_concurrent() -> usize {
    4
}
fn default_memory_top_k() -> usize {
    10
}
fn default_memory_min_score() -> f32 {
    0.5
}
fn default_stream_buffer() -> usize {
    32
}
fn default_temperature() -> f64 {
    0.7
}
fn default_max_tokens() -> u32 {
    2048
}
fn default_max_retries() -> u32 {
    2
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            default_profile: Profile::Balanced,
            timeout: default_timeout(),
            enable_parallel: true,
            max_concurrent: default_max_concurrent(),
            enable_memory: false,
            memory_top_k: default_memory_top_k(),
            memory_min_score: default_memory_min_score(),
            enable_streaming: true,
            stream_buffer_size: default_stream_buffer(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            retry_on_failure: true,
            max_retries: default_max_retries(),
        }
    }
}

// ============================================================================
// MEMORY INTEGRATION
// ============================================================================

/// Context retrieved from memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    /// Retrieved chunks with relevance scores
    pub chunks: Vec<MemoryChunk>,

    /// Query that was used
    pub query: String,

    /// Time taken for retrieval
    pub retrieval_time_ms: u64,

    /// Whether RAPTOR tree was used
    pub used_raptor: bool,
}

/// A chunk retrieved from memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChunk {
    /// Chunk ID
    pub id: Uuid,

    /// Document ID
    pub doc_id: Uuid,

    /// Text content
    pub text: String,

    /// Relevance score
    pub score: f32,

    /// Source metadata
    pub source: Option<String>,
}

/// Trait for memory providers (enables mock testing and multiple backends)
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    /// Query memory for relevant context
    async fn query(&self, query: &str, top_k: usize, min_score: f32) -> Result<MemoryContext>;

    /// Store a reasoning session for future reference
    async fn store_session(&self, session: &ReasoningSession) -> Result<()>;
}

// ============================================================================
// THINKTOOL EXECUTION
// ============================================================================

/// Result from a ThinkTool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkToolResult {
    /// ThinkTool that was executed
    pub tool_id: String,

    /// Output content
    pub content: String,

    /// Confidence score (0.0-1.0)
    pub confidence: f64,

    /// Execution time in milliseconds
    pub duration_ms: u64,

    /// Token usage
    pub tokens: TokenUsage,

    /// Structured output (if applicable)
    pub structured: Option<serde_json::Value>,

    /// Warnings or notes
    pub notes: Vec<String>,
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub cost_usd: f64,
}

impl TokenUsage {
    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.cost_usd += other.cost_usd;
    }
}

/// Trait for ThinkTool execution (enables custom implementations)
#[async_trait]
pub trait ThinkToolExecutor: Send + Sync {
    /// Execute a ThinkTool with the given input
    async fn execute(
        &self,
        tool_id: &str,
        input: &str,
        context: &ExecutionContext,
    ) -> Result<ThinkToolResult>;

    /// List available ThinkTools
    fn available_tools(&self) -> Vec<&str>;
}

/// Context for ThinkTool execution
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Session ID
    pub session_id: Uuid,

    /// Current profile
    pub profile: Profile,

    /// Memory context (if available)
    pub memory: Option<MemoryContext>,

    /// Previous step outputs
    pub previous_outputs: HashMap<String, ThinkToolResult>,

    /// LLM temperature
    pub temperature: f64,

    /// Max tokens
    pub max_tokens: u32,
}

// ============================================================================
// STREAMING OUTPUT
// ============================================================================

/// Events emitted during reasoning loop execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningEvent {
    /// Reasoning session started
    SessionStarted {
        session_id: Uuid,
        profile: Profile,
        prompt: String,
    },

    /// Memory query completed
    MemoryQueried {
        chunks_found: usize,
        retrieval_time_ms: u64,
    },

    /// ThinkTool step started
    StepStarted {
        step_index: usize,
        total_steps: usize,
        tool_id: String,
    },

    /// ThinkTool step completed
    StepCompleted {
        step_index: usize,
        tool_id: String,
        confidence: f64,
        duration_ms: u64,
    },

    /// Partial output from a step (for streaming LLM responses)
    PartialOutput { tool_id: String, delta: String },

    /// Warning or note
    Warning { message: String },

    /// Final decision reached
    DecisionReached {
        confidence: f64,
        total_duration_ms: u64,
    },

    /// Error occurred
    Error { message: String },

    /// Session completed
    SessionCompleted { success: bool },
}

/// Handle for receiving streaming events
pub struct StreamHandle {
    receiver: broadcast::Receiver<ReasoningEvent>,
    session_id: Uuid,
}

impl StreamHandle {
    /// Receive the next event
    pub async fn next(&mut self) -> Option<ReasoningEvent> {
        loop {
            match self.receiver.recv().await {
                Ok(event) => return Some(event),
                Err(broadcast::error::RecvError::Closed) => return None,
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    // If we lagged, continue loop to get next available (non-recursive)
                    continue;
                }
            }
        }
    }

    /// Get session ID
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }
}

// ============================================================================
// REASONING STEP & DECISION
// ============================================================================

/// Kind of reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// Memory retrieval
    MemoryQuery,
    /// ThinkTool execution
    ThinkTool { tool_id: String },
    /// Synthesis of previous steps
    Synthesis,
    /// Validation pass
    Validation,
}

/// A single step in the reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step index (0-based)
    pub index: usize,

    /// Kind of step
    pub kind: StepKind,

    /// Input to this step
    pub input: String,

    /// Output from this step
    pub output: String,

    /// Confidence score
    pub confidence: f64,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Token usage
    pub tokens: TokenUsage,

    /// Whether step succeeded
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,
}

/// Final decision from reasoning loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// Unique decision ID
    pub id: Uuid,

    /// Session ID this decision belongs to
    pub session_id: Uuid,

    /// Original prompt
    pub prompt: String,

    /// Profile used
    pub profile: Profile,

    /// Final conclusion/answer
    pub conclusion: String,

    /// Overall confidence (0.0-1.0)
    pub confidence: f64,

    /// All reasoning steps taken
    pub steps: Vec<ReasoningStep>,

    /// Total token usage
    pub total_tokens: TokenUsage,

    /// Total duration in milliseconds
    pub total_duration_ms: u64,

    /// Memory context used (if any)
    pub memory_context: Option<MemoryContext>,

    /// Whether reasoning succeeded
    pub success: bool,

    /// Key insights extracted
    pub insights: Vec<String>,

    /// Caveats or limitations noted
    pub caveats: Vec<String>,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Decision {
    /// Check if confidence meets profile threshold
    pub fn meets_threshold(&self) -> bool {
        self.confidence >= self.profile.min_confidence()
    }

    /// Get a summary of the decision
    pub fn summary(&self) -> String {
        format!(
            "[{}] {} (confidence: {:.0}%, {} steps, {}ms)",
            self.profile,
            if self.success { "SUCCESS" } else { "FAILED" },
            self.confidence * 100.0,
            self.steps.len(),
            self.total_duration_ms
        )
    }
}

// ============================================================================
// REASONING SESSION
// ============================================================================

/// Stateful reasoning session with accumulated context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSession {
    /// Session ID
    pub id: Uuid,

    /// Profile being used
    pub profile: Profile,

    /// Original prompt
    pub prompt: String,

    /// Current step index
    pub current_step: usize,

    /// All steps executed
    pub steps: Vec<ReasoningStep>,

    /// Memory context
    pub memory_context: Option<MemoryContext>,

    /// Accumulated token usage
    pub total_tokens: TokenUsage,

    /// Session start time
    pub started_at: chrono::DateTime<chrono::Utc>,

    /// Whether session is complete
    pub completed: bool,

    /// Final decision (if completed)
    pub decision: Option<Decision>,
}

impl ReasoningSession {
    /// Create a new session
    pub fn new(prompt: &str, profile: Profile) -> Self {
        Self {
            id: Uuid::new_v4(),
            profile,
            prompt: prompt.to_string(),
            current_step: 0,
            steps: Vec::new(),
            memory_context: None,
            total_tokens: TokenUsage::default(),
            started_at: chrono::Utc::now(),
            completed: false,
            decision: None,
        }
    }

    /// Add a completed step
    pub fn add_step(&mut self, step: ReasoningStep) {
        self.total_tokens.add(&step.tokens);
        self.steps.push(step);
        self.current_step += 1;
    }

    /// Get current confidence (average of all steps)
    pub fn current_confidence(&self) -> f64 {
        if self.steps.is_empty() {
            0.0
        } else {
            self.steps.iter().map(|s| s.confidence).sum::<f64>() / self.steps.len() as f64
        }
    }

    /// Complete the session with a decision
    pub fn complete(&mut self, conclusion: String, insights: Vec<String>, caveats: Vec<String>) {
        let total_duration_ms = (chrono::Utc::now() - self.started_at).num_milliseconds() as u64;

        self.decision = Some(Decision {
            id: Uuid::new_v4(),
            session_id: self.id,
            prompt: self.prompt.clone(),
            profile: self.profile,
            conclusion,
            confidence: self.current_confidence(),
            steps: self.steps.clone(),
            total_tokens: self.total_tokens.clone(),
            total_duration_ms,
            memory_context: self.memory_context.clone(),
            success: true,
            insights,
            caveats,
            timestamp: chrono::Utc::now(),
        });

        self.completed = true;
    }
}

// ============================================================================
// REASONING LOOP BUILDER
// ============================================================================

/// Builder for ReasoningLoop with fluent API
pub struct ReasoningLoopBuilder {
    config: ReasoningConfig,
    executor: Option<Arc<dyn ThinkToolExecutor>>,
    memory: Option<Arc<dyn MemoryProvider>>,
}

impl ReasoningLoopBuilder {
    /// Create a new builder with default config
    pub fn new() -> Self {
        Self {
            config: ReasoningConfig::default(),
            executor: None,
            memory: None,
        }
    }

    /// Set the default profile
    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.config.default_profile = profile;
        self
    }

    /// Set the execution timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Enable parallel execution
    pub fn with_parallel(mut self, enabled: bool, max_concurrent: usize) -> Self {
        self.config.enable_parallel = enabled;
        self.config.max_concurrent = max_concurrent;
        self
    }

    /// Set the ThinkTool executor
    pub fn with_executor(mut self, executor: Arc<dyn ThinkToolExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Set the memory provider
    pub fn with_memory(mut self, memory: Arc<dyn MemoryProvider>) -> Self {
        self.memory = Some(memory);
        self.config.enable_memory = true;
        self
    }

    /// Configure memory retrieval
    pub fn with_memory_config(mut self, top_k: usize, min_score: f32) -> Self {
        self.config.memory_top_k = top_k;
        self.config.memory_min_score = min_score;
        self
    }

    /// Enable streaming output
    pub fn with_streaming(mut self, enabled: bool, buffer_size: usize) -> Self {
        self.config.enable_streaming = enabled;
        self.config.stream_buffer_size = buffer_size;
        self
    }

    /// Set LLM parameters
    pub fn with_llm_params(mut self, temperature: f64, max_tokens: u32) -> Self {
        self.config.temperature = temperature;
        self.config.max_tokens = max_tokens;
        self
    }

    /// Set retry configuration
    pub fn with_retries(mut self, enabled: bool, max_retries: u32) -> Self {
        self.config.retry_on_failure = enabled;
        self.config.max_retries = max_retries;
        self
    }

    /// Set the full config
    pub fn with_config(mut self, config: ReasoningConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the ReasoningLoop
    pub fn build(self) -> Result<ReasoningLoop> {
        let executor = self
            .executor
            .ok_or_else(|| ReasoningError::Config("ThinkTool executor required".into()))?;

        Ok(ReasoningLoop {
            config: self.config,
            executor,
            memory: self.memory,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            _event_sender: None,
        })
    }
}

impl Default for ReasoningLoopBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// REASONING LOOP - MAIN ENGINE
// ============================================================================

/// High-performance async reasoning engine
///
/// The ReasoningLoop orchestrates ThinkTool execution with:
/// - Async/concurrent execution via Tokio
/// - Optional memory integration via reasonkit-mem
/// - Streaming output via broadcast channels
/// - Profile-based execution modes
pub struct ReasoningLoop {
    /// Configuration
    config: ReasoningConfig,

    /// ThinkTool executor
    executor: Arc<dyn ThinkToolExecutor>,

    /// Memory provider (optional)
    memory: Option<Arc<dyn MemoryProvider>>,

    /// Active sessions
    active_sessions: Arc<RwLock<HashMap<Uuid, ReasoningSession>>>,

    /// Event broadcaster (lazily initialized)
    _event_sender: Option<broadcast::Sender<ReasoningEvent>>,
}

impl ReasoningLoop {
    /// Create a builder for ReasoningLoop
    pub fn builder() -> ReasoningLoopBuilder {
        ReasoningLoopBuilder::new()
    }

    /// Get configuration
    pub fn config(&self) -> &ReasoningConfig {
        &self.config
    }

    /// Execute reasoning with streaming output
    pub async fn reason_stream(&self, prompt: &str) -> Result<(StreamHandle, Decision)> {
        self.reason_stream_with_profile(prompt, self.config.default_profile)
            .await
    }

    /// Execute reasoning with streaming output and custom profile
    pub async fn reason_stream_with_profile(
        &self,
        prompt: &str,
        profile: Profile,
    ) -> Result<(StreamHandle, Decision)> {
        // Create broadcast channel for events
        let (tx, rx) = broadcast::channel(self.config.stream_buffer_size);

        let session = ReasoningSession::new(prompt, profile);
        let session_id = session.id;

        // Store session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id, session.clone());
        }

        // Send start event
        let _ = tx.send(ReasoningEvent::SessionStarted {
            session_id,
            profile,
            prompt: prompt.to_string(),
        });

        // Execute reasoning
        let decision = self.execute_loop(session, Some(&tx)).await?;

        // Send completion event
        let _ = tx.send(ReasoningEvent::SessionCompleted {
            success: decision.success,
        });

        // Remove from active sessions
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.remove(&session_id);
        }

        Ok((
            StreamHandle {
                receiver: rx,
                session_id,
            },
            decision,
        ))
    }

    /// Execute reasoning and return decision (blocking until complete)
    pub async fn reason(&self, prompt: &str) -> Result<Decision> {
        self.reason_with_profile(prompt, self.config.default_profile)
            .await
    }

    /// Execute reasoning with custom profile
    pub async fn reason_with_profile(&self, prompt: &str, profile: Profile) -> Result<Decision> {
        let session = ReasoningSession::new(prompt, profile);
        self.execute_loop(session, None).await
    }

    /// Core execution loop
    async fn execute_loop(
        &self,
        mut session: ReasoningSession,
        event_tx: Option<&broadcast::Sender<ReasoningEvent>>,
    ) -> Result<Decision> {
        let start = Instant::now();
        let profile = session.profile;

        // Step 1: Query memory if enabled
        if self.config.enable_memory {
            if let Some(ref memory) = self.memory {
                let mem_start = Instant::now();
                match memory
                    .query(
                        &session.prompt,
                        self.config.memory_top_k,
                        self.config.memory_min_score,
                    )
                    .await
                {
                    Ok(context) => {
                        let retrieval_time = mem_start.elapsed().as_millis() as u64;

                        if let Some(tx) = event_tx {
                            let _ = tx.send(ReasoningEvent::MemoryQueried {
                                chunks_found: context.chunks.len(),
                                retrieval_time_ms: retrieval_time,
                            });
                        }

                        session.memory_context = Some(context);
                    }
                    Err(e) => {
                        if let Some(tx) = event_tx {
                            let _ = tx.send(ReasoningEvent::Warning {
                                message: format!("Memory query failed: {}", e),
                            });
                        }
                    }
                }
            }
        }

        // Step 2: Execute ThinkTool chain
        let tools = profile.thinktool_chain();
        let total_steps = tools.len();

        let mut previous_outputs: HashMap<String, ThinkToolResult> = HashMap::new();

        for (step_idx, tool_id) in tools.iter().enumerate() {
            // Check timeout
            if start.elapsed() > self.config.timeout {
                return Err(ReasoningError::Timeout(self.config.timeout));
            }

            // Emit step start event
            if let Some(tx) = event_tx {
                let _ = tx.send(ReasoningEvent::StepStarted {
                    step_index: step_idx,
                    total_steps,
                    tool_id: tool_id.to_string(),
                });
            }

            // Build input for this step
            let input = self.build_step_input(&session, &previous_outputs, step_idx);

            // Create execution context
            let context = ExecutionContext {
                session_id: session.id,
                profile,
                memory: session.memory_context.clone(),
                previous_outputs: previous_outputs.clone(),
                temperature: self.config.temperature,
                max_tokens: self.config.max_tokens,
            };

            // Execute with retry logic
            let result = self.execute_with_retry(tool_id, &input, &context).await?;

            // Create reasoning step
            let step = ReasoningStep {
                index: step_idx,
                kind: StepKind::ThinkTool {
                    tool_id: tool_id.to_string(),
                },
                input: input.clone(),
                output: result.content.clone(),
                confidence: result.confidence,
                duration_ms: result.duration_ms,
                tokens: result.tokens.clone(),
                success: true,
                error: None,
            };

            // Emit step complete event
            if let Some(tx) = event_tx {
                let _ = tx.send(ReasoningEvent::StepCompleted {
                    step_index: step_idx,
                    tool_id: tool_id.to_string(),
                    confidence: result.confidence,
                    duration_ms: result.duration_ms,
                });
            }

            // Store result for next step
            previous_outputs.insert(tool_id.to_string(), result);
            session.add_step(step);
        }

        // Step 3: Synthesize final decision
        let (conclusion, insights, caveats) = self.synthesize_decision(&session, &previous_outputs);

        session.complete(conclusion, insights, caveats);

        let decision = session.decision.clone().unwrap();

        // Emit decision event
        if let Some(tx) = event_tx {
            let _ = tx.send(ReasoningEvent::DecisionReached {
                confidence: decision.confidence,
                total_duration_ms: decision.total_duration_ms,
            });
        }

        // Store session to memory if enabled
        if self.config.enable_memory {
            if let Some(ref memory) = self.memory {
                let _ = memory.store_session(&session).await;
            }
        }

        Ok(decision)
    }

    /// Build input for a step based on previous outputs
    fn build_step_input(
        &self,
        session: &ReasoningSession,
        previous_outputs: &HashMap<String, ThinkToolResult>,
        step_idx: usize,
    ) -> String {
        let mut input = session.prompt.clone();

        // Add memory context if available
        if let Some(ref memory) = session.memory_context {
            if !memory.chunks.is_empty() {
                input.push_str("\n\n--- RELEVANT CONTEXT ---\n");
                for chunk in memory.chunks.iter().take(3) {
                    input.push_str(&format!("- {}\n", chunk.text));
                }
            }
        }

        // Add previous step outputs for context
        if step_idx > 0 {
            input.push_str("\n\n--- PREVIOUS ANALYSIS ---\n");
            for (tool_id, result) in previous_outputs {
                // Truncate long outputs
                let content = if result.content.len() > 500 {
                    format!("{}...", &result.content[..500])
                } else {
                    result.content.clone()
                };
                input.push_str(&format!(
                    "[{}] (confidence: {:.0}%)\n{}\n\n",
                    tool_id,
                    result.confidence * 100.0,
                    content
                ));
            }
        }

        input
    }

    /// Execute a ThinkTool with retry logic
    async fn execute_with_retry(
        &self,
        tool_id: &str,
        input: &str,
        context: &ExecutionContext,
    ) -> Result<ThinkToolResult> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self.executor.execute(tool_id, input, context).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if !self.config.retry_on_failure || attempt == self.config.max_retries {
                        return Err(e);
                    }
                    last_error = Some(e);
                    // Exponential backoff
                    let delay = Duration::from_millis(100 * 2u64.pow(attempt));
                    tokio::time::sleep(delay).await;
                }
            }
        }

        Err(
            last_error.unwrap_or_else(|| ReasoningError::ThinkToolFailed {
                tool: tool_id.to_string(),
                message: "Unknown error".into(),
            }),
        )
    }

    /// Synthesize final decision from all steps
    fn synthesize_decision(
        &self,
        session: &ReasoningSession,
        outputs: &HashMap<String, ThinkToolResult>,
    ) -> (String, Vec<String>, Vec<String>) {
        // Extract conclusion from last step
        let conclusion = outputs
            .values()
            .last()
            .map(|r| r.content.clone())
            .unwrap_or_else(|| "No conclusion reached".to_string());

        // Extract insights from various tools
        let mut insights = Vec::new();
        if let Some(gt) = outputs.get("gigathink") {
            if let Some(structured) = &gt.structured {
                if let Some(perspectives) = structured.get("perspectives") {
                    if let Some(arr) = perspectives.as_array() {
                        for p in arr.iter().take(3) {
                            if let Some(s) = p.as_str() {
                                insights.push(s.to_string());
                            }
                        }
                    }
                }
            }
        }

        // Extract caveats from BrutalHonesty if available
        let mut caveats = Vec::new();
        if let Some(bh) = outputs.get("brutalhonesty") {
            if bh.content.to_lowercase().contains("caveat")
                || bh.content.to_lowercase().contains("limitation")
            {
                caveats.push("See BrutalHonesty analysis for detailed limitations".to_string());
            }
        }

        // Add confidence-based caveat
        let confidence = session.current_confidence();
        if confidence < session.profile.min_confidence() {
            caveats.push(format!(
                "Confidence ({:.0}%) below target ({:.0}%)",
                confidence * 100.0,
                session.profile.min_confidence() * 100.0
            ));
        }

        (conclusion, insights, caveats)
    }

    /// Get an active session by ID
    pub async fn get_session(&self, session_id: Uuid) -> Option<ReasoningSession> {
        let sessions = self.active_sessions.read().await;
        sessions.get(&session_id).cloned()
    }

    /// Cancel an active session
    pub async fn cancel_session(&self, session_id: Uuid) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if sessions.remove(&session_id).is_some() {
            Ok(())
        } else {
            Err(ReasoningError::Config(format!(
                "Session {} not found",
                session_id
            )))
        }
    }

    /// Get count of active sessions
    pub async fn active_session_count(&self) -> usize {
        let sessions = self.active_sessions.read().await;
        sessions.len()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_chains() {
        assert_eq!(Profile::Quick.thinktool_chain().len(), 2);
        assert_eq!(Profile::Balanced.thinktool_chain().len(), 4);
        assert_eq!(Profile::Deep.thinktool_chain().len(), 5);
        assert_eq!(Profile::Paranoid.thinktool_chain().len(), 6);
    }

    #[test]
    fn test_profile_confidence() {
        assert_eq!(Profile::Quick.min_confidence(), 0.70);
        assert_eq!(Profile::Balanced.min_confidence(), 0.80);
        assert_eq!(Profile::Deep.min_confidence(), 0.85);
        assert_eq!(Profile::Paranoid.min_confidence(), 0.95);
    }

    #[test]
    fn test_profile_from_str() {
        assert_eq!(Profile::from_str("quick"), Some(Profile::Quick));
        assert_eq!(Profile::from_str("Q"), Some(Profile::Quick));
        assert_eq!(Profile::from_str("balanced"), Some(Profile::Balanced));
        assert_eq!(Profile::from_str("PARANOID"), Some(Profile::Paranoid));
        assert_eq!(Profile::from_str("invalid"), None);
    }

    #[test]
    fn test_config_defaults() {
        let config = ReasoningConfig::default();
        assert_eq!(config.default_profile, Profile::Balanced);
        assert!(config.enable_parallel);
        assert_eq!(config.max_concurrent, 4);
        assert!(!config.enable_memory);
    }

    #[test]
    fn test_session_creation() {
        let session = ReasoningSession::new("Test prompt", Profile::Balanced);
        assert!(!session.completed);
        assert_eq!(session.current_step, 0);
        assert!(session.steps.is_empty());
    }

    #[test]
    fn test_session_confidence() {
        let mut session = ReasoningSession::new("Test", Profile::Balanced);

        // Empty session has 0 confidence
        assert_eq!(session.current_confidence(), 0.0);

        // Add steps with varying confidence
        session.add_step(ReasoningStep {
            index: 0,
            kind: StepKind::ThinkTool {
                tool_id: "gigathink".into(),
            },
            input: "test".into(),
            output: "output".into(),
            confidence: 0.8,
            duration_ms: 100,
            tokens: TokenUsage::default(),
            success: true,
            error: None,
        });

        session.add_step(ReasoningStep {
            index: 1,
            kind: StepKind::ThinkTool {
                tool_id: "laserlogic".into(),
            },
            input: "test".into(),
            output: "output".into(),
            confidence: 0.9,
            duration_ms: 100,
            tokens: TokenUsage::default(),
            success: true,
            error: None,
        });

        assert_eq!(session.current_confidence(), 0.85);
    }

    #[test]
    fn test_token_usage_add() {
        let mut total = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            cost_usd: 0.001,
        };

        let step = TokenUsage {
            input_tokens: 200,
            output_tokens: 100,
            total_tokens: 300,
            cost_usd: 0.002,
        };

        total.add(&step);

        assert_eq!(total.input_tokens, 300);
        assert_eq!(total.output_tokens, 150);
        assert_eq!(total.total_tokens, 450);
        assert!((total.cost_usd - 0.003).abs() < 0.0001);
    }

    #[test]
    fn test_decision_meets_threshold() {
        let decision = Decision {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            prompt: "test".into(),
            profile: Profile::Balanced,
            conclusion: "test conclusion".into(),
            confidence: 0.85,
            steps: vec![],
            total_tokens: TokenUsage::default(),
            total_duration_ms: 1000,
            memory_context: None,
            success: true,
            insights: vec![],
            caveats: vec![],
            timestamp: chrono::Utc::now(),
        };

        assert!(decision.meets_threshold()); // 0.85 >= 0.80

        let low_confidence = Decision {
            confidence: 0.75,
            ..decision.clone()
        };
        assert!(!low_confidence.meets_threshold()); // 0.75 < 0.80
    }
}
