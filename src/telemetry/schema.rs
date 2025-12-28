//! SQLite Schema for RALL Telemetry
//!
//! Local-first telemetry storage with privacy-preserving design.

use crate::telemetry::TELEMETRY_SCHEMA_VERSION;

/// SQLite schema for telemetry database
pub const SCHEMA_SQL: &str = r#"
-- ============================================================================
-- RALL TELEMETRY SCHEMA v1
-- ReasonKit Adaptive Learning Loop - Local Telemetry Storage
-- ============================================================================

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

-- Insert current version if not exists
INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial RALL telemetry schema');

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Sessions table: One entry per CLI session
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,                    -- UUID
    started_at TEXT NOT NULL,               -- ISO 8601 timestamp
    ended_at TEXT,                          -- ISO 8601 timestamp (NULL if active)
    duration_ms INTEGER,                    -- Session duration in milliseconds
    tool_count INTEGER DEFAULT 0,           -- Number of tools used
    query_count INTEGER DEFAULT 0,          -- Number of queries
    feedback_count INTEGER DEFAULT 0,       -- Number of feedback events
    profile TEXT,                           -- Reasoning profile used (quick/balanced/deep/etc)
    success_rate REAL,                      -- Overall success rate (0.0-1.0)
    client_version TEXT,                    -- ReasonKit CLI version
    os_family TEXT                          -- Operating system (sanitized)
);

-- Queries table: Individual query events (PII-stripped)
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,                    -- UUID
    session_id TEXT NOT NULL,               -- FK to sessions
    timestamp TEXT NOT NULL,                -- ISO 8601 timestamp

    -- Query metadata (NO raw query text - privacy)
    query_hash TEXT NOT NULL,               -- SHA-256 hash of normalized query
    query_length INTEGER NOT NULL,          -- Character count
    query_token_count INTEGER,              -- Token count (if available)
    query_type TEXT,                        -- Classification: search/reason/code/general

    -- Execution metrics
    latency_ms INTEGER NOT NULL,            -- Total execution time
    tool_calls INTEGER DEFAULT 0,           -- Number of tool calls
    retrieval_count INTEGER DEFAULT 0,      -- Number of documents retrieved

    -- Results (anonymized)
    result_count INTEGER,                   -- Number of results returned
    result_quality_score REAL,              -- Self-assessed quality (0.0-1.0)
    error_occurred INTEGER DEFAULT 0,       -- Boolean: did an error occur?
    error_category TEXT,                    -- Error classification (if error)

    -- Context
    profile_used TEXT,                      -- Reasoning profile
    tools_used TEXT,                        -- JSON array of tool names used

    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_queries_session ON queries(session_id);
CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);
CREATE INDEX IF NOT EXISTS idx_queries_type ON queries(query_type);
CREATE INDEX IF NOT EXISTS idx_queries_hash ON queries(query_hash);

-- Feedback table: User feedback events
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,                    -- UUID
    session_id TEXT NOT NULL,               -- FK to sessions
    query_id TEXT,                          -- FK to queries (optional)
    timestamp TEXT NOT NULL,                -- ISO 8601 timestamp

    -- Feedback data
    feedback_type TEXT NOT NULL,            -- thumbs_up/thumbs_down/explicit/implicit
    rating INTEGER,                         -- 1-5 star rating (if explicit)
    category TEXT,                          -- accuracy/relevance/speed/format/other

    -- Context (anonymized)
    context_hash TEXT,                      -- Hash of surrounding context

    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (query_id) REFERENCES queries(id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);

-- Tool usage table: Track tool invocations
CREATE TABLE IF NOT EXISTS tool_usage (
    id TEXT PRIMARY KEY,                    -- UUID
    session_id TEXT NOT NULL,               -- FK to sessions
    query_id TEXT,                          -- FK to queries
    timestamp TEXT NOT NULL,                -- ISO 8601 timestamp

    -- Tool metadata
    tool_name TEXT NOT NULL,                -- Tool identifier
    tool_category TEXT,                     -- search/file/shell/mcp/reasoning

    -- Execution metrics
    execution_ms INTEGER NOT NULL,          -- Execution time
    success INTEGER NOT NULL,               -- Boolean: did it succeed?
    error_type TEXT,                        -- Error classification (if failed)

    -- Input/output stats (NO content - privacy)
    input_size_bytes INTEGER,               -- Size of input
    output_size_bytes INTEGER,              -- Size of output

    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (query_id) REFERENCES queries(id)
);

CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON tool_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON tool_usage(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_usage_category ON tool_usage(tool_category);

-- Reasoning traces table: ThinkTool execution traces
CREATE TABLE IF NOT EXISTS reasoning_traces (
    id TEXT PRIMARY KEY,                    -- UUID
    session_id TEXT NOT NULL,               -- FK to sessions
    query_id TEXT,                          -- FK to queries
    timestamp TEXT NOT NULL,                -- ISO 8601 timestamp

    -- Trace metadata
    thinktool_name TEXT NOT NULL,           -- GigaThink/LaserLogic/etc
    step_count INTEGER NOT NULL,            -- Number of reasoning steps

    -- Execution metrics
    total_ms INTEGER NOT NULL,              -- Total execution time
    avg_step_ms REAL,                       -- Average time per step

    -- Quality metrics (computed, not user-provided)
    coherence_score REAL,                   -- Self-consistency check (0.0-1.0)
    depth_score REAL,                       -- Reasoning depth metric

    -- Anonymized structure
    step_types TEXT,                        -- JSON array of step type names

    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (query_id) REFERENCES queries(id)
);

CREATE INDEX IF NOT EXISTS idx_traces_session ON reasoning_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_thinktool ON reasoning_traces(thinktool_name);

-- ============================================================================
-- AGGREGATION TABLES (for ML training data)
-- ============================================================================

-- Daily aggregates: Pre-computed daily statistics
CREATE TABLE IF NOT EXISTS daily_aggregates (
    date TEXT PRIMARY KEY,                  -- YYYY-MM-DD
    computed_at TEXT NOT NULL,              -- When this was computed

    -- Volume metrics
    session_count INTEGER DEFAULT 0,
    query_count INTEGER DEFAULT 0,
    feedback_count INTEGER DEFAULT 0,
    tool_invocations INTEGER DEFAULT 0,

    -- Performance metrics
    avg_latency_ms REAL,
    p50_latency_ms REAL,
    p95_latency_ms REAL,
    p99_latency_ms REAL,

    -- Quality metrics
    avg_success_rate REAL,
    positive_feedback_ratio REAL,
    error_rate REAL,

    -- Tool distribution (JSON)
    tool_distribution TEXT,

    -- Query type distribution (JSON)
    query_type_distribution TEXT
);

-- Query clusters: K-means clustering results for pattern detection
CREATE TABLE IF NOT EXISTS query_clusters (
    id INTEGER PRIMARY KEY,
    computed_at TEXT NOT NULL,

    -- Cluster metadata
    cluster_count INTEGER NOT NULL,         -- K value used
    silhouette_score REAL,                  -- Clustering quality

    -- Cluster data (JSON)
    centroids TEXT,                         -- JSON array of centroid embeddings
    cluster_sizes TEXT,                     -- JSON array of cluster sizes
    representative_hashes TEXT              -- JSON array of representative query hashes
);

-- ============================================================================
-- PRIVACY CONTROLS
-- ============================================================================

-- Privacy consent tracking
CREATE TABLE IF NOT EXISTS privacy_consent (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,

    -- Consent levels
    local_telemetry INTEGER NOT NULL,       -- Boolean: allow local storage
    aggregated_sharing INTEGER NOT NULL,    -- Boolean: allow aggregated sharing
    community_contribution INTEGER NOT NULL, -- Boolean: contribute to community model

    -- Metadata
    consent_version INTEGER NOT NULL,       -- Consent form version
    ip_hash TEXT                            -- Hashed IP for legal compliance
);

-- Redaction log: Track what was redacted (for audit)
CREATE TABLE IF NOT EXISTS redaction_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,

    -- Redaction metadata
    source_table TEXT NOT NULL,
    source_id TEXT NOT NULL,
    redaction_type TEXT NOT NULL,           -- pii/sensitive/custom
    pattern_matched TEXT                    -- Regex pattern that triggered redaction
);

CREATE INDEX IF NOT EXISTS idx_redaction_source ON redaction_log(source_table, source_id);

-- ============================================================================
-- VIEWS for common queries
-- ============================================================================

-- Recent session summary
CREATE VIEW IF NOT EXISTS v_recent_sessions AS
SELECT
    s.id,
    s.started_at,
    s.duration_ms,
    s.query_count,
    s.tool_count,
    s.success_rate,
    COUNT(DISTINCT f.id) as feedback_items,
    AVG(CASE WHEN f.feedback_type = 'thumbs_up' THEN 1.0
             WHEN f.feedback_type = 'thumbs_down' THEN 0.0
             ELSE NULL END) as feedback_score
FROM sessions s
LEFT JOIN feedback f ON s.id = f.session_id
WHERE s.started_at > datetime('now', '-30 days')
GROUP BY s.id
ORDER BY s.started_at DESC;

-- Tool performance summary
CREATE VIEW IF NOT EXISTS v_tool_performance AS
SELECT
    tool_name,
    tool_category,
    COUNT(*) as invocation_count,
    AVG(execution_ms) as avg_execution_ms,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate,
    SUM(input_size_bytes) as total_input_bytes,
    SUM(output_size_bytes) as total_output_bytes
FROM tool_usage
WHERE timestamp > datetime('now', '-30 days')
GROUP BY tool_name, tool_category
ORDER BY invocation_count DESC;

-- ThinkTool effectiveness
CREATE VIEW IF NOT EXISTS v_thinktool_stats AS
SELECT
    thinktool_name,
    COUNT(*) as usage_count,
    AVG(step_count) as avg_steps,
    AVG(total_ms) as avg_execution_ms,
    AVG(coherence_score) as avg_coherence,
    AVG(depth_score) as avg_depth
FROM reasoning_traces
WHERE timestamp > datetime('now', '-30 days')
GROUP BY thinktool_name
ORDER BY usage_count DESC;
"#;

/// Get the current schema version
pub fn current_version() -> u32 {
    TELEMETRY_SCHEMA_VERSION
}

/// Migration SQL for future schema updates
pub fn get_migration_sql(_from_version: u32, _to_version: u32) -> Option<&'static str> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version() {
        assert_eq!(current_version(), 1);
    }

    #[test]
    fn test_schema_sql_not_empty() {
        assert!(!SCHEMA_SQL.is_empty());
        assert!(SCHEMA_SQL.contains("CREATE TABLE"));
    }
}
