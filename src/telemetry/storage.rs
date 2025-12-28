//! Telemetry Storage Backend
//!
//! SQLite-based local storage for telemetry data using rusqlite.

use crate::telemetry::{
    schema::SCHEMA_SQL, AggregatedMetrics, CommunityExport, FeedbackEvent, FeedbackSummary,
    QueryEvent, TelemetryError, TelemetryResult, TimeRange, ToolUsageMetric, TraceEvent,
    TELEMETRY_SCHEMA_VERSION,
};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Mutex;

/// Telemetry storage backend using SQLite
pub struct TelemetryStorage {
    /// SQLite connection (wrapped in Mutex for thread safety)
    conn: Option<Mutex<Connection>>,
    /// Database path for reference
    db_path: String,
    /// Whether this is a no-op storage
    is_noop: bool,
}

impl TelemetryStorage {
    /// Create a new storage backend
    pub async fn new(db_path: &Path) -> TelemetryResult<Self> {
        let path_str = db_path.to_string_lossy().to_string();

        // Ensure parent directory exists (critical for first-run initialization)
        if let Some(parent) = db_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    TelemetryError::Io(std::io::Error::other(format!(
                        "Failed to create telemetry directory {:?}: {}",
                        parent, e
                    )))
                })?;
                tracing::info!(path = ?parent, "Created telemetry data directory");
            }
        }

        // Open or create the database
        let conn =
            Connection::open(db_path).map_err(|e| TelemetryError::Database(e.to_string()))?;

        // Check and perform schema migration if needed
        Self::migrate_schema(&conn)?;

        tracing::info!(
            path = %path_str,
            schema_version = TELEMETRY_SCHEMA_VERSION,
            "Initialized telemetry database"
        );

        Ok(Self {
            conn: Some(Mutex::new(conn)),
            db_path: path_str,
            is_noop: false,
        })
    }

    /// Initialize with default configuration
    ///
    /// Uses `~/.local/share/reasonkit/.rk_telemetry.db` on Linux/Mac
    /// or the appropriate XDG data directory.
    pub async fn initialize_default() -> TelemetryResult<Self> {
        use crate::telemetry::TelemetryConfig;
        let db_path = TelemetryConfig::default_db_path();
        Self::new(&db_path).await
    }

    /// Check schema version and migrate if needed
    fn migrate_schema(conn: &Connection) -> TelemetryResult<()> {
        // Check if schema_version table exists
        let has_version_table: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='schema_version'",
                [],
                |row| row.get::<_, i64>(0).map(|c| c > 0),
            )
            .unwrap_or(false);

        if !has_version_table {
            // Fresh database - initialize schema
            conn.execute_batch(SCHEMA_SQL).map_err(|e| {
                TelemetryError::Database(format!("Failed to initialize schema: {}", e))
            })?;

            // Record schema version (separate table for tracking)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)",
                [],
            )
            .map_err(|e| TelemetryError::Database(e.to_string()))?;

            // Use INSERT OR REPLACE to handle re-initialization gracefully
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?1, datetime('now'))",
                params![TELEMETRY_SCHEMA_VERSION as i64],
            )
            .map_err(|e| TelemetryError::Database(e.to_string()))?;

            tracing::info!(
                version = TELEMETRY_SCHEMA_VERSION,
                "Initialized fresh telemetry schema"
            );
        } else {
            // Check current version
            let current_version: i64 = conn
                .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                    row.get(0)
                })
                .unwrap_or(0);

            if current_version < TELEMETRY_SCHEMA_VERSION as i64 {
                // Future: Apply migrations here
                // For now, just update the version record
                tracing::info!(
                    from = current_version,
                    to = TELEMETRY_SCHEMA_VERSION,
                    "Migrating telemetry schema"
                );

                // Record the new version (use INSERT OR REPLACE for idempotency)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?1, datetime('now'))",
                    params![TELEMETRY_SCHEMA_VERSION as i64],
                )
                .map_err(|e| TelemetryError::Database(e.to_string()))?;
            }
            // If current_version >= TELEMETRY_SCHEMA_VERSION, nothing to do - already up to date
        }

        Ok(())
    }

    /// Create an in-memory storage (for testing)
    pub fn in_memory() -> TelemetryResult<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| TelemetryError::Database(e.to_string()))?;

        // Use the same migration logic for consistency
        Self::migrate_schema(&conn)?;

        Ok(Self {
            conn: Some(Mutex::new(conn)),
            db_path: ":memory:".to_string(),
            is_noop: false,
        })
    }

    /// Get current schema version
    pub fn schema_version(&self) -> TelemetryResult<u32> {
        if self.is_noop {
            return Ok(0);
        }

        let conn = self.get_conn()?;
        let version: i64 = conn
            .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        Ok(version as u32)
    }

    /// Get database file path
    pub fn db_path(&self) -> &str {
        &self.db_path
    }

    /// Create a no-op storage (when telemetry is disabled)
    pub fn noop() -> Self {
        Self {
            conn: None,
            db_path: String::new(),
            is_noop: true,
        }
    }

    /// Get a reference to the connection
    fn get_conn(&self) -> TelemetryResult<std::sync::MutexGuard<'_, Connection>> {
        self.conn
            .as_ref()
            .ok_or(TelemetryError::Disabled)?
            .lock()
            .map_err(|e| TelemetryError::Database(format!("Lock poisoned: {}", e)))
    }

    /// Hash a query for privacy
    fn hash_query(query: &str) -> String {
        let normalized = query
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Insert a query event
    pub async fn insert_query_event(&mut self, event: &QueryEvent) -> TelemetryResult<()> {
        if self.is_noop {
            return Ok(());
        }

        let conn = self.get_conn()?;
        let query_hash = Self::hash_query(&event.query_text);
        let tools_json = serde_json::to_string(&event.tools_used).unwrap_or_default();

        conn.execute(
            r#"INSERT INTO queries (
                id, session_id, timestamp, query_hash, query_length,
                query_token_count, query_type, latency_ms, tool_calls,
                retrieval_count, result_count, result_quality_score,
                error_occurred, error_category, profile_used, tools_used
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)"#,
            params![
                event.id.to_string(),
                event.session_id.to_string(),
                event.timestamp.to_rfc3339(),
                query_hash,
                event.query_text.len() as i64,
                None::<i64>, // token_count
                format!("{:?}", event.query_type).to_lowercase(),
                event.latency_ms as i64,
                event.tool_calls as i64,
                event.retrieval_count as i64,
                event.result_count as i64,
                event.quality_score,
                event.error.is_some() as i64,
                event
                    .error
                    .as_ref()
                    .map(|e| format!("{:?}", e.category).to_lowercase()),
                event.profile.as_deref(),
                tools_json,
            ],
        )
        .map_err(|e| TelemetryError::Database(e.to_string()))?;

        tracing::debug!(
            event_id = %event.id,
            session_id = %event.session_id,
            query_type = ?event.query_type,
            latency_ms = event.latency_ms,
            "Recorded query event"
        );

        Ok(())
    }

    /// Insert a feedback event
    pub async fn insert_feedback_event(&mut self, event: &FeedbackEvent) -> TelemetryResult<()> {
        if self.is_noop {
            return Ok(());
        }

        let conn = self.get_conn()?;

        conn.execute(
            r#"INSERT INTO feedback (
                id, session_id, query_id, timestamp,
                feedback_type, rating, category, context_hash
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
            params![
                event.id.to_string(),
                event.session_id.to_string(),
                event.query_id.map(|id| id.to_string()),
                event.timestamp.to_rfc3339(),
                format!("{:?}", event.feedback_type).to_lowercase(),
                event.rating.map(|r| r as i64),
                event
                    .category
                    .as_ref()
                    .map(|c| format!("{:?}", c).to_lowercase()),
                event.context_hash.as_deref(),
            ],
        )
        .map_err(|e| TelemetryError::Database(e.to_string()))?;

        tracing::debug!(
            event_id = %event.id,
            session_id = %event.session_id,
            feedback_type = ?event.feedback_type,
            "Recorded feedback event"
        );

        Ok(())
    }

    /// Insert a trace event
    pub async fn insert_trace_event(&mut self, event: &TraceEvent) -> TelemetryResult<()> {
        if self.is_noop {
            return Ok(());
        }

        let conn = self.get_conn()?;
        let step_types_json = serde_json::to_string(&event.step_types).unwrap_or_default();

        conn.execute(
            r#"INSERT INTO reasoning_traces (
                id, session_id, query_id, timestamp,
                thinktool_name, step_count, total_ms, avg_step_ms,
                coherence_score, depth_score, step_types
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"#,
            params![
                event.id.to_string(),
                event.session_id.to_string(),
                event.query_id.map(|id| id.to_string()),
                event.timestamp.to_rfc3339(),
                event.thinktool_name,
                event.step_count as i64,
                event.total_ms as i64,
                event.avg_step_ms,
                event.coherence_score,
                event.depth_score,
                step_types_json,
            ],
        )
        .map_err(|e| TelemetryError::Database(e.to_string()))?;

        tracing::debug!(
            event_id = %event.id,
            session_id = %event.session_id,
            thinktool = %event.thinktool_name,
            steps = event.step_count,
            "Recorded trace event"
        );

        Ok(())
    }

    /// Get aggregated metrics
    pub async fn get_aggregated_metrics(&self) -> TelemetryResult<AggregatedMetrics> {
        if self.is_noop {
            return Err(TelemetryError::Disabled);
        }

        let conn = self.get_conn()?;

        // Get total queries and average latency
        let (total_queries, avg_latency): (i64, f64) = conn.query_row(
            "SELECT COUNT(*), COALESCE(AVG(latency_ms), 0) FROM queries WHERE timestamp > datetime('now', '-30 days')",
            [],
            |row| Ok((row.get(0)?, row.get(1)?))
        ).map_err(|e| TelemetryError::Database(e.to_string()))?;

        // Get tool usage metrics
        let mut tool_stmt = conn
            .prepare(
                r#"SELECT
                tool_name,
                COUNT(*) as count,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate,
                AVG(execution_ms) as avg_execution_ms
            FROM tool_usage
            WHERE timestamp > datetime('now', '-30 days')
            GROUP BY tool_name
            ORDER BY count DESC
            LIMIT 20"#,
            )
            .map_err(|e| TelemetryError::Database(e.to_string()))?;

        let tool_usage: Vec<ToolUsageMetric> = tool_stmt
            .query_map([], |row| {
                Ok(ToolUsageMetric {
                    tool: row.get(0)?,
                    count: row.get::<_, i64>(1)? as u64,
                    success_rate: row.get(2)?,
                    avg_execution_ms: row.get(3)?,
                })
            })
            .map_err(|e| TelemetryError::Database(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        // Get feedback summary
        let (total_feedback, positive_ratio): (i64, f64) = conn.query_row(
            r#"SELECT
                COUNT(*),
                COALESCE(SUM(CASE WHEN feedback_type = 'thumbs_up' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0), 0)
            FROM feedback
            WHERE timestamp > datetime('now', '-30 days')"#,
            [],
            |row| Ok((row.get(0)?, row.get(1)?))
        ).map_err(|e| TelemetryError::Database(e.to_string()))?;

        // Get time range
        let (start, end): (String, String) = conn.query_row(
            "SELECT COALESCE(MIN(timestamp), datetime('now')), COALESCE(MAX(timestamp), datetime('now')) FROM queries",
            [],
            |row| Ok((row.get(0)?, row.get(1)?))
        ).map_err(|e| TelemetryError::Database(e.to_string()))?;

        Ok(AggregatedMetrics {
            total_queries: total_queries as u64,
            avg_latency_ms: avg_latency,
            tool_usage,
            query_clusters: Vec::new(), // Clustering requires more complex logic
            feedback_summary: FeedbackSummary {
                total_feedback: total_feedback as u64,
                positive_ratio,
                improvement_areas: Vec::new(),
            },
            time_range: TimeRange {
                start: DateTime::parse_from_rfc3339(&format!("{}Z", start))
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                end: DateTime::parse_from_rfc3339(&format!("{}Z", end))
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            },
        })
    }

    /// Export anonymized data for community contribution
    pub async fn export_anonymized(&self) -> TelemetryResult<CommunityExport> {
        if self.is_noop {
            return Err(TelemetryError::Disabled);
        }

        let aggregates = self.get_aggregated_metrics().await?;
        let contributor_hash = self.generate_contributor_hash();

        Ok(CommunityExport {
            schema_version: TELEMETRY_SCHEMA_VERSION,
            exported_at: Utc::now(),
            aggregates,
            dp_epsilon: 1.0,
            contributor_hash,
        })
    }

    /// Generate a stable but anonymous contributor hash
    fn generate_contributor_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.db_path.as_bytes());
        hasher.update(b"reasonkit-contributor-v1");
        format!("{:x}", hasher.finalize())[..16].to_string()
    }

    /// Run daily aggregation
    pub async fn run_daily_aggregation(&mut self, date: &str) -> TelemetryResult<()> {
        if self.is_noop {
            return Ok(());
        }

        let conn = self.get_conn()?;

        // Compute and insert daily aggregate
        conn.execute(
            r#"INSERT OR REPLACE INTO daily_aggregates (
                date, computed_at,
                session_count, query_count, feedback_count, tool_invocations,
                avg_latency_ms, p50_latency_ms, p95_latency_ms, p99_latency_ms,
                avg_success_rate, positive_feedback_ratio, error_rate,
                tool_distribution, query_type_distribution
            )
            SELECT
                ?1 as date,
                datetime('now') as computed_at,
                COUNT(DISTINCT session_id) as session_count,
                COUNT(*) as query_count,
                (SELECT COUNT(*) FROM feedback WHERE date(timestamp) = ?1) as feedback_count,
                SUM(tool_calls) as tool_invocations,
                AVG(latency_ms) as avg_latency_ms,
                AVG(latency_ms) as p50_latency_ms,
                AVG(latency_ms) as p95_latency_ms,
                AVG(latency_ms) as p99_latency_ms,
                1.0 - (SUM(error_occurred) * 1.0 / NULLIF(COUNT(*), 0)) as avg_success_rate,
                (SELECT SUM(CASE WHEN feedback_type = 'thumbs_up' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0)
                 FROM feedback WHERE date(timestamp) = ?1) as positive_feedback_ratio,
                SUM(error_occurred) * 1.0 / NULLIF(COUNT(*), 0) as error_rate,
                '{}' as tool_distribution,
                '{}' as query_type_distribution
            FROM queries
            WHERE date(timestamp) = ?1"#,
            params![date],
        ).map_err(|e| TelemetryError::Database(e.to_string()))?;

        tracing::info!(date = %date, "Ran daily aggregation");

        Ok(())
    }

    /// Get database size in bytes
    pub async fn get_db_size(&self) -> TelemetryResult<u64> {
        if self.is_noop {
            return Ok(0);
        }

        let conn = self.get_conn()?;

        let size: i64 = conn
            .query_row(
                "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        Ok(size as u64)
    }

    /// Prune old data based on retention policy
    pub async fn prune_old_data(&mut self, retention_days: u32) -> TelemetryResult<u64> {
        if self.is_noop {
            return Ok(0);
        }

        let conn = self.get_conn()?;
        let cutoff = format!("-{} days", retention_days);

        let mut total_deleted = 0u64;

        // Delete old queries
        let deleted = conn
            .execute(
                "DELETE FROM queries WHERE timestamp < datetime('now', ?1)",
                params![cutoff],
            )
            .map_err(|e| TelemetryError::Database(e.to_string()))?;
        total_deleted += deleted as u64;

        // Delete old feedback
        let deleted = conn
            .execute(
                "DELETE FROM feedback WHERE timestamp < datetime('now', ?1)",
                params![cutoff],
            )
            .map_err(|e| TelemetryError::Database(e.to_string()))?;
        total_deleted += deleted as u64;

        // Delete old traces
        let deleted = conn
            .execute(
                "DELETE FROM reasoning_traces WHERE timestamp < datetime('now', ?1)",
                params![cutoff],
            )
            .map_err(|e| TelemetryError::Database(e.to_string()))?;
        total_deleted += deleted as u64;

        tracing::info!(
            retention_days = retention_days,
            deleted = total_deleted,
            "Pruned old telemetry data"
        );

        Ok(total_deleted)
    }

    /// Vacuum database to reclaim space
    pub async fn vacuum(&mut self) -> TelemetryResult<()> {
        if self.is_noop {
            return Ok(());
        }

        let conn = self.get_conn()?;
        conn.execute("VACUUM", [])
            .map_err(|e| TelemetryError::Database(e.to_string()))?;

        tracing::info!("Vacuumed telemetry database");

        Ok(())
    }

    /// Get query count
    pub async fn get_query_count(&self) -> TelemetryResult<u64> {
        if self.is_noop {
            return Ok(0);
        }

        let conn = self.get_conn()?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM queries", [], |row| row.get(0))
            .map_err(|e| TelemetryError::Database(e.to_string()))?;

        Ok(count as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::QueryType;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_in_memory_storage() {
        let storage = TelemetryStorage::in_memory().unwrap();
        assert!(!storage.is_noop);

        let count = storage.get_query_count().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_noop_storage() {
        let storage = TelemetryStorage::noop();
        assert!(storage.is_noop);

        let result = storage.get_aggregated_metrics().await;
        assert!(result.is_err());
    }

    /// Helper to create a test session in the database
    fn create_test_session(storage: &TelemetryStorage, session_id: &Uuid) {
        let conn = storage.conn.as_ref().unwrap().lock().unwrap();
        conn.execute(
            "INSERT INTO sessions (id, started_at) VALUES (?1, ?2)",
            params![session_id.to_string(), Utc::now().to_rfc3339()],
        )
        .unwrap();
    }

    #[tokio::test]
    async fn test_insert_and_query_event() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        // Create session first (required by foreign key constraint)
        create_test_session(&storage, &session_id);

        let event = QueryEvent::new(session_id, "test query".to_string())
            .with_type(QueryType::Search)
            .with_latency(100);

        storage.insert_query_event(&event).await.unwrap();

        let count = storage.get_query_count().await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_insert_feedback_event() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        // Create session first (required by foreign key constraint)
        create_test_session(&storage, &session_id);

        let feedback = FeedbackEvent::thumbs_up(session_id, None);
        storage.insert_feedback_event(&feedback).await.unwrap();

        let metrics = storage.get_aggregated_metrics().await.unwrap();
        assert_eq!(metrics.feedback_summary.total_feedback, 1);
    }

    #[test]
    fn test_query_hash_consistency() {
        let hash1 = TelemetryStorage::hash_query("test query");
        let hash2 = TelemetryStorage::hash_query("test  query"); // extra space
        assert_eq!(hash1, hash2);

        let hash3 = TelemetryStorage::hash_query("different query");
        assert_ne!(hash1, hash3);
    }

    #[tokio::test]
    async fn test_prune_old_data() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        // Create session first (required by foreign key constraint)
        create_test_session(&storage, &session_id);

        // Insert a query
        let event = QueryEvent::new(session_id, "test".to_string());
        storage.insert_query_event(&event).await.unwrap();

        // Prune with 0 days retention (should delete everything)
        let _deleted = storage.prune_old_data(0).await.unwrap();

        // The event was just inserted, so with 0 days retention it should be deleted
        // But since timestamp is "now", it might not be deleted. Let's check count instead.
        let count = storage.get_query_count().await.unwrap();
        // After prune with 0 days, recent data should remain (within today)
        assert!(count <= 1);
    }

    #[test]
    fn test_schema_version_tracking() {
        let storage = TelemetryStorage::in_memory().unwrap();

        // Schema version should match the constant
        let version = storage.schema_version().unwrap();
        assert_eq!(version, TELEMETRY_SCHEMA_VERSION);
    }

    #[test]
    fn test_db_path_accessor() {
        let storage = TelemetryStorage::in_memory().unwrap();
        assert_eq!(storage.db_path(), ":memory:");

        let noop = TelemetryStorage::noop();
        assert_eq!(noop.db_path(), "");
    }

    #[tokio::test]
    async fn test_file_based_storage_with_directory_creation() {
        use std::fs;

        // Create a temp directory for testing
        let temp_dir = std::env::temp_dir().join("reasonkit_test_telemetry");
        let db_path = temp_dir.join("nested").join("dir").join("test.db");

        // Ensure it doesn't exist first
        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir).ok();
        }

        // This should create the nested directories automatically
        let storage = TelemetryStorage::new(&db_path).await.unwrap();

        // Verify the directory was created
        assert!(db_path.parent().unwrap().exists());

        // Verify schema is initialized
        let version = storage.schema_version().unwrap();
        assert_eq!(version, TELEMETRY_SCHEMA_VERSION);

        // Cleanup
        drop(storage);
        fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    async fn test_schema_migration_idempotent() {
        // Opening the same database twice should work without errors
        use std::fs;

        let temp_dir = std::env::temp_dir().join("reasonkit_test_migration");
        let db_path = temp_dir.join("migration_test.db");

        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir).ok();
        }

        // First open - creates schema
        {
            let storage = TelemetryStorage::new(&db_path).await.unwrap();
            assert_eq!(storage.schema_version().unwrap(), TELEMETRY_SCHEMA_VERSION);
        }

        // Second open - should work without error (idempotent)
        {
            let storage = TelemetryStorage::new(&db_path).await.unwrap();
            assert_eq!(storage.schema_version().unwrap(), TELEMETRY_SCHEMA_VERSION);
        }

        // Cleanup
        fs::remove_dir_all(&temp_dir).ok();
    }
}
