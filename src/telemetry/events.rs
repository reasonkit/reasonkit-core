//! Telemetry Event Types
//!
//! Defines all event types that can be recorded by the telemetry system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Query event - recorded for each user query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEvent {
    /// Event ID
    pub id: Uuid,
    /// Session ID
    pub session_id: Uuid,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Original query text (will be hashed, not stored)
    pub query_text: String,
    /// Query type classification
    pub query_type: QueryType,
    /// Execution latency in milliseconds
    pub latency_ms: u64,
    /// Number of tool calls made
    pub tool_calls: u32,
    /// Number of documents retrieved
    pub retrieval_count: u32,
    /// Result count
    pub result_count: u32,
    /// Quality score (0.0 - 1.0)
    pub quality_score: Option<f64>,
    /// Error occurred
    pub error: Option<QueryError>,
    /// Reasoning profile used
    pub profile: Option<String>,
    /// Tools used (list of tool names)
    pub tools_used: Vec<String>,
}

impl QueryEvent {
    /// Create a new query event
    pub fn new(session_id: Uuid, query_text: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id,
            timestamp: Utc::now(),
            query_text,
            query_type: QueryType::General,
            latency_ms: 0,
            tool_calls: 0,
            retrieval_count: 0,
            result_count: 0,
            quality_score: None,
            error: None,
            profile: None,
            tools_used: Vec::new(),
        }
    }

    /// Set query type
    pub fn with_type(mut self, query_type: QueryType) -> Self {
        self.query_type = query_type;
        self
    }

    /// Set latency
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Set tools used
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tool_calls = tools.len() as u32;
        self.tools_used = tools;
        self
    }
}

/// Query type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum QueryType {
    /// Search/retrieval query
    Search,
    /// Reasoning/analysis query
    Reason,
    /// Code generation/editing
    Code,
    /// General conversation
    #[default]
    General,
    /// File operations
    File,
    /// System commands
    System,
}

/// Query error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryError {
    /// Error category
    pub category: ErrorCategory,
    /// Error code (if applicable)
    pub code: Option<String>,
    /// Is recoverable
    pub recoverable: bool,
}

/// Error category classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    /// Network/connectivity error
    Network,
    /// API error (rate limit, auth, etc.)
    Api,
    /// Parsing error
    Parse,
    /// Timeout
    Timeout,
    /// Resource not found
    NotFound,
    /// Permission denied
    Permission,
    /// Internal error
    Internal,
    /// Unknown error
    Unknown,
}

/// Feedback event - user feedback on results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEvent {
    /// Event ID
    pub id: Uuid,
    /// Session ID
    pub session_id: Uuid,
    /// Related query ID (optional)
    pub query_id: Option<Uuid>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Feedback type
    pub feedback_type: FeedbackType,
    /// Rating (1-5, if explicit)
    pub rating: Option<u8>,
    /// Category of feedback
    pub category: Option<FeedbackCategory>,
    /// Context hash (for dedup)
    pub context_hash: Option<String>,
}

impl FeedbackEvent {
    /// Create thumbs up feedback
    pub fn thumbs_up(session_id: Uuid, query_id: Option<Uuid>) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id,
            query_id,
            timestamp: Utc::now(),
            feedback_type: FeedbackType::ThumbsUp,
            rating: None,
            category: None,
            context_hash: None,
        }
    }

    /// Create thumbs down feedback
    pub fn thumbs_down(session_id: Uuid, query_id: Option<Uuid>) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id,
            query_id,
            timestamp: Utc::now(),
            feedback_type: FeedbackType::ThumbsDown,
            rating: None,
            category: None,
            context_hash: None,
        }
    }

    /// Create explicit rating feedback
    pub fn rating(session_id: Uuid, query_id: Option<Uuid>, rating: u8) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id,
            query_id,
            timestamp: Utc::now(),
            feedback_type: FeedbackType::Explicit,
            rating: Some(rating.clamp(1, 5)),
            category: None,
            context_hash: None,
        }
    }

    /// Set feedback category
    pub fn with_category(mut self, category: FeedbackCategory) -> Self {
        self.category = Some(category);
        self
    }
}

/// Feedback type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackType {
    /// Positive quick feedback
    ThumbsUp,
    /// Negative quick feedback
    ThumbsDown,
    /// Explicit rating (1-5)
    Explicit,
    /// Implicit (inferred from behavior)
    Implicit,
}

/// Feedback category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackCategory {
    /// Accuracy of response
    Accuracy,
    /// Relevance to query
    Relevance,
    /// Speed of response
    Speed,
    /// Format/presentation
    Format,
    /// Completeness
    Completeness,
    /// Other
    Other,
}

/// Reasoning trace event - ThinkTool execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    /// Event ID
    pub id: Uuid,
    /// Session ID
    pub session_id: Uuid,
    /// Related query ID (optional)
    pub query_id: Option<Uuid>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// ThinkTool name
    pub thinktool_name: String,
    /// Number of reasoning steps
    pub step_count: u32,
    /// Total execution time in milliseconds
    pub total_ms: u64,
    /// Average step time in milliseconds
    pub avg_step_ms: Option<f64>,
    /// Coherence score (0.0 - 1.0)
    pub coherence_score: Option<f64>,
    /// Depth score (0.0 - 1.0)
    pub depth_score: Option<f64>,
    /// Step types (for analysis)
    pub step_types: Vec<String>,
}

impl TraceEvent {
    /// Create a new trace event
    pub fn new(session_id: Uuid, thinktool_name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id,
            query_id: None,
            timestamp: Utc::now(),
            thinktool_name,
            step_count: 0,
            total_ms: 0,
            avg_step_ms: None,
            coherence_score: None,
            depth_score: None,
            step_types: Vec::new(),
        }
    }

    /// Set execution metrics
    pub fn with_execution(mut self, step_count: u32, total_ms: u64) -> Self {
        self.step_count = step_count;
        self.total_ms = total_ms;
        if step_count > 0 {
            self.avg_step_ms = Some(total_ms as f64 / step_count as f64);
        }
        self
    }

    /// Set quality metrics
    pub fn with_quality(mut self, coherence: f64, depth: f64) -> Self {
        self.coherence_score = Some(coherence.clamp(0.0, 1.0));
        self.depth_score = Some(depth.clamp(0.0, 1.0));
        self
    }

    /// Set step types
    pub fn with_steps(mut self, step_types: Vec<String>) -> Self {
        self.step_types = step_types;
        self
    }
}

/// Tool usage event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsageEvent {
    /// Event ID
    pub id: Uuid,
    /// Session ID
    pub session_id: Uuid,
    /// Related query ID (optional)
    pub query_id: Option<Uuid>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Tool name
    pub tool_name: String,
    /// Tool category
    pub tool_category: ToolCategory,
    /// Execution time in milliseconds
    pub execution_ms: u64,
    /// Success flag
    pub success: bool,
    /// Error type (if failed)
    pub error_type: Option<String>,
    /// Input size in bytes
    pub input_size: Option<u64>,
    /// Output size in bytes
    pub output_size: Option<u64>,
}

/// Tool category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ToolCategory {
    /// Search/retrieval tools
    Search,
    /// File system operations
    File,
    /// Shell/command execution
    Shell,
    /// MCP server tools
    Mcp,
    /// Reasoning tools
    Reasoning,
    /// Web/network tools
    Web,
    /// Other
    #[default]
    Other,
}

/// Session event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    /// Session ID
    pub id: Uuid,
    /// Start timestamp
    pub started_at: DateTime<Utc>,
    /// End timestamp (None if still active)
    pub ended_at: Option<DateTime<Utc>>,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Reasoning profile used
    pub profile: Option<String>,
    /// Client version
    pub client_version: String,
    /// OS family (sanitized)
    pub os_family: String,
}

impl SessionEvent {
    /// Create a new session
    pub fn start(client_version: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            duration_ms: None,
            profile: None,
            client_version,
            os_family: std::env::consts::OS.to_string(),
        }
    }

    /// End the session
    pub fn end(mut self) -> Self {
        let now = Utc::now();
        let duration = now.signed_duration_since(self.started_at);
        self.ended_at = Some(now);
        self.duration_ms = Some(duration.num_milliseconds().max(0) as u64);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_event_creation() {
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, "test query".to_string())
            .with_type(QueryType::Search)
            .with_latency(100);

        assert_eq!(event.session_id, session_id);
        assert_eq!(event.query_type, QueryType::Search);
        assert_eq!(event.latency_ms, 100);
    }

    #[test]
    fn test_feedback_rating_clamp() {
        let session_id = Uuid::new_v4();
        let event = FeedbackEvent::rating(session_id, None, 10);
        assert_eq!(event.rating, Some(5)); // Clamped to max

        let event = FeedbackEvent::rating(session_id, None, 0);
        assert_eq!(event.rating, Some(1)); // Clamped to min
    }

    #[test]
    fn test_session_lifecycle() {
        let session = SessionEvent::start(crate::VERSION.to_string());
        assert!(session.ended_at.is_none());

        let ended = session.end();
        assert!(ended.ended_at.is_some());
        assert!(ended.duration_ms.is_some());
    }
}
