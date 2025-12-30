//! # ReasonKit Telemetry Module
//!
//! Privacy-first telemetry for the ReasonKit Adaptive Learning Loop (RALL).
//!
//! ## Design Principles
//!
//! 1. **Local-First**: All data stored in SQLite, never sent externally by default
//! 2. **Privacy-First**: PII stripping, differential privacy, opt-in only
//! 3. **Lightweight**: Minimal overhead, async batch processing
//! 4. **Auditable**: Full schema versioning, data lineage
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    RALL TELEMETRY SYSTEM                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │ LAYER 4: AGGREGATION                                        │
//! │   Local Clustering | Pattern Detection | Model Feedback     │
//! ├─────────────────────────────────────────────────────────────┤
//! │ LAYER 3: PRIVACY FIREWALL                                   │
//! │   PII Stripping | Differential Privacy | Redaction          │
//! ├─────────────────────────────────────────────────────────────┤
//! │ LAYER 2: COLLECTION                                         │
//! │   Event Queue | Batch Writer | Schema Validation            │
//! ├─────────────────────────────────────────────────────────────┤
//! │ LAYER 1: EVENTS                                             │
//! │   Query Events | Feedback Events | Session Events           │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod config;
mod events;
mod privacy;
mod schema;
mod storage;

pub use config::*;
pub use events::*;
pub use privacy::*;
pub use schema::*;
pub use storage::*;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Telemetry system version for schema migrations
pub const TELEMETRY_SCHEMA_VERSION: u32 = 1;

/// Default telemetry database filename
pub const DEFAULT_TELEMETRY_DB: &str = ".rk_telemetry.db";

/// Result type for telemetry operations
pub type TelemetryResult<T> = Result<T, TelemetryError>;

/// Telemetry error types
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    /// Database error
    #[error("Database error: {0}")]
    Database(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Privacy violation detected
    #[error("Privacy violation: {0}")]
    PrivacyViolation(String),

    /// Schema validation error
    #[error("Schema validation error: {0}")]
    SchemaValidation(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Telemetry disabled
    #[error("Telemetry is disabled")]
    Disabled,
}

/// Main telemetry collector
pub struct TelemetryCollector {
    /// Configuration
    config: TelemetryConfig,
    /// Storage backend
    storage: Arc<RwLock<TelemetryStorage>>,
    /// Privacy filter
    privacy: PrivacyFilter,
    /// Session ID
    session_id: Uuid,
    /// Whether telemetry is enabled
    enabled: bool,
}

impl TelemetryCollector {
    /// Create a new telemetry collector
    pub async fn new(config: TelemetryConfig) -> TelemetryResult<Self> {
        if !config.enabled {
            return Ok(Self {
                config: config.clone(),
                storage: Arc::new(RwLock::new(TelemetryStorage::noop())),
                privacy: PrivacyFilter::new(config.privacy.clone()),
                session_id: Uuid::new_v4(),
                enabled: false,
            });
        }

        let mut storage = TelemetryStorage::new(&config.db_path).await?;
        let privacy = PrivacyFilter::new(config.privacy.clone());
        let session_id = Uuid::new_v4();

        // Insert session record to satisfy foreign key constraints
        storage.insert_session(session_id).await?;

        Ok(Self {
            config: config.clone(),
            storage: Arc::new(RwLock::new(storage)),
            privacy,
            session_id,
            enabled: true,
        })
    }

    /// Record a query event
    pub async fn record_query(&self, event: QueryEvent) -> TelemetryResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // Apply privacy filter
        let sanitized = self.privacy.sanitize_query_event(event)?;

        // Store event
        let mut storage = self.storage.write().await;
        storage.insert_query_event(&sanitized).await
    }

    /// Record user feedback
    pub async fn record_feedback(&self, event: FeedbackEvent) -> TelemetryResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let sanitized = self.privacy.sanitize_feedback_event(event)?;

        let mut storage = self.storage.write().await;
        storage.insert_feedback_event(&sanitized).await
    }

    /// Record a reasoning trace
    pub async fn record_trace(&self, event: TraceEvent) -> TelemetryResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let sanitized = self.privacy.sanitize_trace_event(event)?;

        let mut storage = self.storage.write().await;
        storage.insert_trace_event(&sanitized).await
    }

    /// Get aggregated metrics for local ML training
    pub async fn get_aggregated_metrics(&self) -> TelemetryResult<AggregatedMetrics> {
        if !self.enabled {
            return Err(TelemetryError::Disabled);
        }

        let storage = self.storage.read().await;
        storage.get_aggregated_metrics().await
    }

    /// Export anonymized data for community model training (opt-in)
    pub async fn export_for_community(&self) -> TelemetryResult<CommunityExport> {
        if !self.enabled || !self.config.community_contribution {
            return Err(TelemetryError::Disabled);
        }

        let storage = self.storage.read().await;
        storage.export_anonymized().await
    }

    /// Get current session ID
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Check if telemetry is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Aggregated metrics for local ML optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Total query count
    pub total_queries: u64,
    /// Average query latency (ms)
    pub avg_latency_ms: f64,
    /// Tool usage distribution
    pub tool_usage: Vec<ToolUsageMetric>,
    /// Query pattern clusters
    pub query_clusters: Vec<QueryCluster>,
    /// Feedback summary
    pub feedback_summary: FeedbackSummary,
    /// Time range
    pub time_range: TimeRange,
}

/// Tool usage metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsageMetric {
    /// Tool name
    pub tool: String,
    /// Usage count
    pub count: u64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Average execution time (ms)
    pub avg_execution_ms: f64,
}

/// Query cluster for pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCluster {
    /// Cluster ID
    pub id: u32,
    /// Centroid embedding (if available)
    pub centroid: Option<Vec<f32>>,
    /// Number of queries in cluster
    pub count: u64,
    /// Representative query (anonymized)
    pub representative: String,
    /// Common tools used
    pub common_tools: Vec<String>,
}

/// Feedback summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSummary {
    /// Total feedback count
    pub total_feedback: u64,
    /// Positive feedback ratio
    pub positive_ratio: f64,
    /// Categories with most negative feedback
    pub improvement_areas: Vec<String>,
}

/// Time range for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time
    pub start: DateTime<Utc>,
    /// End time
    pub end: DateTime<Utc>,
}

/// Community export format (fully anonymized)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityExport {
    /// Schema version
    pub schema_version: u32,
    /// Export timestamp
    pub exported_at: DateTime<Utc>,
    /// Anonymized aggregates only
    pub aggregates: AggregatedMetrics,
    /// Differential privacy epsilon used
    pub dp_epsilon: f64,
    /// Hash of contributing user (for dedup)
    pub contributor_hash: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telemetry_disabled() {
        let config = TelemetryConfig {
            enabled: false,
            ..Default::default()
        };

        let collector = TelemetryCollector::new(config).await.unwrap();
        assert!(!collector.is_enabled());
    }

    #[tokio::test]
    async fn test_session_id_generation() {
        let config = TelemetryConfig::default();
        let collector = TelemetryCollector::new(config).await.unwrap();

        // Session ID should be a valid UUID
        let _ = collector.session_id();
    }
}
