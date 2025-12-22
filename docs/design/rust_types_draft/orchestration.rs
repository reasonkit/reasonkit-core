//! Multi-Agent Orchestration Type Definitions
//!
//! Types for agent coordination, task routing, and distributed execution.
//! Reference: docs/design/MULTI_AGENT_ORCHESTRATION.md

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::common::{Confidence, Id, Metadata, TokenUsage};

// ═══════════════════════════════════════════════════════════════════════════
// AGENT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Agent tier in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum AgentTier {
    /// Tier 1: Governance - Final authority, architecture decisions
    Governance = 1,
    /// Tier 2: Executive - Strategy, large refactors, reviews
    Executive = 2,
    /// Tier 3: Engineering - Implementation, testing
    Engineering = 3,
    /// Tier 4: Specialist - Domain-specific validation
    Specialist = 4,
}

impl AgentTier {
    /// Check if this tier can delegate to another tier
    pub fn can_delegate_to(&self, other: AgentTier) -> bool {
        (*self as u8) < (other as u8)
    }

    /// Check if this tier can escalate to another tier
    pub fn can_escalate_to(&self, other: AgentTier) -> bool {
        (*self as u8) > (other as u8)
    }
}

/// Agent capability categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Mathematics,
    Security,
    Architecture,
    Documentation,
    Performance,
    Testing,
    CodeReview,
    UnsafeRust,
    Concurrency,
    DataAnalysis,
    Research,
    Writing,
}

/// Capability proficiency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Proficiency {
    Basic = 1,
    Proficient = 2,
    Expert = 3,
}

impl Proficiency {
    /// Convert to numeric score for routing
    pub fn score(&self) -> f64 {
        match self {
            Self::Basic => 0.4,
            Self::Proficient => 0.7,
            Self::Expert => 1.0,
        }
    }
}

/// Agent operational status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStatus {
    /// Ready to accept tasks
    Idle,
    /// Currently executing a task
    Busy,
    /// High load, can still accept critical tasks
    Overloaded,
    /// Cannot accept any tasks
    Unavailable,
    /// API rate limit reached
    RateLimited,
    /// Warming up / initializing
    Starting,
    /// Shutting down gracefully
    Stopping,
}

/// Agent identity and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Unique agent identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Agent tier in hierarchy
    pub tier: AgentTier,
    /// AI model identifier
    pub model: String,
    /// Model provider (anthropic, openai, google, etc.)
    pub provider: String,
    /// Capability map with proficiency levels
    pub capabilities: HashMap<Capability, Proficiency>,
    /// Maximum context window in tokens
    pub context_window: u32,
    /// Cost per input token (USD)
    pub cost_per_input_token: f64,
    /// Cost per output token (USD)
    pub cost_per_output_token: f64,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: u32,
    /// Default task timeout in milliseconds
    pub timeout_default_ms: u64,
    /// Rate limit (requests per minute)
    pub rate_limit_rpm: u32,
}

impl AgentConfig {
    /// Calculate average cost per 1K tokens
    pub fn avg_cost_per_1k(&self) -> f64 {
        (self.cost_per_input_token + self.cost_per_output_token) / 2.0 * 1000.0
    }

    /// Check if agent has required capabilities at minimum proficiency
    pub fn has_capabilities(&self, required: &[Capability], min_proficiency: Proficiency) -> bool {
        required.iter().all(|cap| {
            self.capabilities
                .get(cap)
                .map(|p| *p >= min_proficiency)
                .unwrap_or(false)
        })
    }
}

/// Runtime agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub agent_id: String,
    pub status: AgentStatus,
    pub current_task: Option<TaskId>,
    pub queue: Vec<TaskId>,
    pub load: f64,
    pub last_heartbeat: DateTime<Utc>,
    pub rate_limit_remaining: u32,
    pub error_count_1h: u32,
    pub success_count_1h: u32,
    pub avg_latency_ms: f64,
    pub total_tokens_1h: u64,
    pub total_cost_1h: f64,
}

impl AgentState {
    /// Check if agent is available for new tasks
    pub fn is_available(&self) -> bool {
        matches!(self.status, AgentStatus::Idle | AgentStatus::Busy)
            && self.load < 0.9
            && self.rate_limit_remaining > 0
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count_1h + self.error_count_1h;
        if total == 0 {
            0.5 // Neutral for no history
        } else {
            self.success_count_1h as f64 / total as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TASK TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Unique task identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

impl TaskId {
    pub fn new() -> Self {
        Self(Id::new("task").to_string())
    }

    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Task type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    CodeGeneration,
    CodeReview,
    Refactoring,
    BugFix,
    Testing,
    Documentation,
    Architecture,
    Security,
    Performance,
    Research,
    Validation,
    DataAnalysis,
    Translation,
    Summarization,
}

impl TaskType {
    /// Get default timeout for this task type (ms)
    pub fn default_timeout(&self) -> u64 {
        match self {
            Self::CodeGeneration => 60_000,
            Self::CodeReview => 45_000,
            Self::Refactoring => 90_000,
            Self::BugFix => 60_000,
            Self::Testing => 45_000,
            Self::Documentation => 30_000,
            Self::Architecture => 120_000,
            Self::Security => 90_000,
            Self::Performance => 60_000,
            Self::Research => 180_000,
            Self::Validation => 30_000,
            Self::DataAnalysis => 90_000,
            Self::Translation => 30_000,
            Self::Summarization => 30_000,
        }
    }

    /// Get required capabilities for this task type
    pub fn required_capabilities(&self) -> Vec<Capability> {
        match self {
            Self::CodeGeneration => vec![Capability::Rust],
            Self::CodeReview => vec![Capability::CodeReview],
            Self::Refactoring => vec![Capability::Architecture],
            Self::BugFix => vec![Capability::Rust],
            Self::Testing => vec![Capability::Testing],
            Self::Documentation => vec![Capability::Documentation],
            Self::Architecture => vec![Capability::Architecture],
            Self::Security => vec![Capability::Security],
            Self::Performance => vec![Capability::Performance],
            Self::Research => vec![Capability::Research],
            Self::Validation => vec![],
            Self::DataAnalysis => vec![Capability::DataAnalysis],
            Self::Translation => vec![Capability::Writing],
            Self::Summarization => vec![Capability::Writing],
        }
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum TaskPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Task execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    /// Task created but not yet routed
    Created,
    /// Task assigned and waiting in queue
    Queued,
    /// Task currently executing
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed (may be retried)
    Failed,
    /// Task deadline passed
    Expired,
    /// Task cancelled by user/system
    Cancelled,
    /// Task escalated to higher tier
    Escalated,
}

impl TaskState {
    /// Check if task is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Expired | Self::Cancelled
        )
    }

    /// Check if task can be retried
    pub fn can_retry(&self) -> bool {
        matches!(self, Self::Failed | Self::Expired)
    }
}

/// Task context for execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskContext {
    /// Project identifier
    pub project: String,
    /// Component within project
    pub component: Option<String>,
    /// Relevant files
    pub files: Vec<String>,
    /// Task dependencies (must complete first)
    pub dependencies: Vec<TaskId>,
    /// Accumulated context from previous tasks
    pub accumulated_context: Option<String>,
    /// Session identifier for context sharing
    pub session_id: Option<String>,
}

/// Routing hints for task assignment
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingHints {
    /// Required capabilities (all must be present)
    pub required_capabilities: Vec<Capability>,
    /// Preferred agent tier
    pub preferred_tier: Option<AgentTier>,
    /// Maximum cost per token (USD)
    pub max_cost_per_token: Option<f64>,
    /// Preferred agent IDs (first available)
    pub preferred_agents: Vec<String>,
    /// Excluded agent IDs (never use)
    pub excluded_agents: Vec<String>,
    /// Minimum context window required
    pub min_context_window: Option<u32>,
}

/// Task payload with instructions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPayload {
    /// Main instruction/prompt
    pub instruction: String,
    /// Constraints to follow
    pub constraints: Vec<String>,
    /// Acceptance criteria
    pub acceptance_criteria: Vec<String>,
    /// File contents for context
    pub context_files: HashMap<String, String>,
    /// Additional metadata
    pub metadata: Metadata,
}

/// Complete task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub deadline_ms: u64,
    pub created_at: DateTime<Utc>,
    pub context: TaskContext,
    pub payload: TaskPayload,
    pub routing_hints: RoutingHints,
    pub state: TaskState,
    pub assigned_agent: Option<String>,
    pub attempt_count: u32,
    pub max_attempts: u32,
    pub parent_task: Option<TaskId>,
    pub child_tasks: Vec<TaskId>,
}

impl Task {
    /// Create a new task
    pub fn new(task_type: TaskType, instruction: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: TaskId::new(),
            task_type,
            priority: TaskPriority::Medium,
            deadline_ms: task_type.default_timeout(),
            created_at: now,
            context: TaskContext::default(),
            payload: TaskPayload {
                instruction: instruction.into(),
                constraints: Vec::new(),
                acceptance_criteria: Vec::new(),
                context_files: HashMap::new(),
                metadata: HashMap::new(),
            },
            routing_hints: RoutingHints {
                required_capabilities: task_type.required_capabilities(),
                ..Default::default()
            },
            state: TaskState::Created,
            assigned_agent: None,
            attempt_count: 0,
            max_attempts: 3,
            parent_task: None,
            child_tasks: Vec::new(),
        }
    }

    /// Check if task has exceeded deadline
    pub fn is_expired(&self) -> bool {
        let elapsed = Utc::now()
            .signed_duration_since(self.created_at)
            .num_milliseconds();
        elapsed as u64 > self.deadline_ms
    }

    /// Check if task can be retried
    pub fn can_retry(&self) -> bool {
        self.state.can_retry() && self.attempt_count < self.max_attempts
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Task execution result status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskResultStatus {
    Completed,
    Failed,
    PartialSuccess,
    Escalated,
}

/// Generated artifact from task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub artifact_type: ArtifactType,
    pub path: String,
    pub content: Option<String>,
    pub diff: Option<String>,
    pub checksum: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactType {
    Code,
    Test,
    Documentation,
    Config,
    Schema,
    Report,
    Data,
}

/// Validation results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationResult {
    pub clippy: ValidationStatus,
    pub tests: ValidationStatus,
    pub format: ValidationStatus,
    pub benchmark: Option<BenchmarkResult>,
    pub custom_checks: HashMap<String, ValidationStatus>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    Pass,
    Fail,
    #[default]
    Skip,
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub latency_p50_ms: f64,
    pub latency_p99_ms: f64,
    pub throughput_ops: Option<f64>,
    pub memory_mb: Option<f64>,
}

/// Task execution error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskError {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
    pub suggested_action: Option<String>,
    pub stack_trace: Option<String>,
}

/// Complete task result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub status: TaskResultStatus,
    pub agent_id: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub execution_time_ms: u64,
    pub token_usage: TokenUsage,
    pub artifacts: Vec<Artifact>,
    pub validation: Option<ValidationResult>,
    pub confidence: Confidence,
    pub notes: Option<String>,
    pub error: Option<TaskError>,
}

impl TaskResult {
    /// Check if result represents success
    pub fn is_success(&self) -> bool {
        matches!(
            self.status,
            TaskResultStatus::Completed | TaskResultStatus::PartialSuccess
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ESCALATION TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Reason for escalation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationReason {
    /// Task exceeded timeout
    Timeout,
    /// Agent lacks required capability
    CapabilityExceeded,
    /// Multiple retry failures
    RepeatedFailure,
    /// Agents disagree on approach
    Conflict,
    /// Would exceed cost budget
    CostExceeded,
    /// Security-critical decision needed
    SecurityCritical,
    /// Requires human judgment
    HumanReviewRequired,
    /// Architectural decision needed
    ArchitecturalDecision,
}

/// Escalation request from agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRequest {
    pub id: String,
    pub original_task_id: TaskId,
    pub escalating_agent: String,
    pub reason: EscalationReason,
    pub details: String,
    pub attempted_solutions: Vec<String>,
    pub suggested_agent_tier: AgentTier,
    pub context_snapshot: TaskContext,
    pub partial_progress: f64,
    pub created_at: DateTime<Utc>,
}

// ═══════════════════════════════════════════════════════════════════════════
// MESSAGE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Agent health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealth {
    pub api_latency_ms: f64,
    pub error_rate_1h: f64,
    pub memory_usage_mb: Option<f64>,
    pub cpu_usage_percent: Option<f64>,
}

/// Heartbeat message from agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    pub agent_id: String,
    pub timestamp: DateTime<Utc>,
    pub status: AgentStatus,
    pub current_load: f64,
    pub queue_depth: u32,
    pub capabilities_available: Vec<Capability>,
    pub rate_limit_remaining: u32,
    pub health: AgentHealth,
}

/// Message types for the message bus
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum Message {
    TaskRequest(Task),
    TaskResponse(TaskResult),
    Heartbeat(Heartbeat),
    Escalation(EscalationRequest),
    Cancellation { task_id: TaskId },
    StateSync { agent_id: String, state: AgentState },
}

// ═══════════════════════════════════════════════════════════════════════════
// AUDIT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Audit event types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventType {
    TaskCreated,
    TaskAssigned,
    TaskStarted,
    TaskCompleted,
    TaskFailed,
    TaskEscalated,
    TaskCancelled,
    AgentRegistered,
    AgentUnregistered,
    AgentStateChanged,
    EscalationHandled,
    SystemStart,
    SystemStop,
}

/// Audit event for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub task_id: Option<TaskId>,
    pub agent_id: Option<String>,
    pub details: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_tier_ordering() {
        assert!(AgentTier::Governance < AgentTier::Executive);
        assert!(AgentTier::Executive < AgentTier::Engineering);
        assert!(AgentTier::Engineering < AgentTier::Specialist);
    }

    #[test]
    fn test_tier_delegation() {
        assert!(AgentTier::Governance.can_delegate_to(AgentTier::Executive));
        assert!(!AgentTier::Specialist.can_delegate_to(AgentTier::Engineering));
    }

    #[test]
    fn test_tier_escalation() {
        assert!(AgentTier::Specialist.can_escalate_to(AgentTier::Engineering));
        assert!(!AgentTier::Governance.can_escalate_to(AgentTier::Executive));
    }

    #[test]
    fn test_task_creation() {
        let task = Task::new(TaskType::CodeGeneration, "Implement RRF fusion");
        assert_eq!(task.state, TaskState::Created);
        assert!(task.routing_hints.required_capabilities.contains(&Capability::Rust));
    }

    #[test]
    fn test_proficiency_scoring() {
        assert_eq!(Proficiency::Basic.score(), 0.4);
        assert_eq!(Proficiency::Proficient.score(), 0.7);
        assert_eq!(Proficiency::Expert.score(), 1.0);
    }

    #[test]
    fn test_agent_capability_check() {
        let mut capabilities = HashMap::new();
        capabilities.insert(Capability::Rust, Proficiency::Expert);
        capabilities.insert(Capability::Testing, Proficiency::Proficient);

        let config = AgentConfig {
            id: "test".to_string(),
            name: "Test Agent".to_string(),
            tier: AgentTier::Engineering,
            model: "test-model".to_string(),
            provider: "test".to_string(),
            capabilities,
            context_window: 100000,
            cost_per_input_token: 0.001,
            cost_per_output_token: 0.002,
            max_concurrent_tasks: 5,
            timeout_default_ms: 60000,
            rate_limit_rpm: 100,
        };

        assert!(config.has_capabilities(&[Capability::Rust], Proficiency::Expert));
        assert!(config.has_capabilities(&[Capability::Testing], Proficiency::Basic));
        assert!(!config.has_capabilities(&[Capability::Security], Proficiency::Basic));
    }
}
