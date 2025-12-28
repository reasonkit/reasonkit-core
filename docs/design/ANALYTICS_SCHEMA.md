# ReasonKit Analytics & Telemetry Schema Design

> Privacy-First Analytics for Structured Reasoning
> "What gets measured gets managed - but what gets shared must be consented"

**Version**: 2.0.0
**Status**: Design Document
**Author**: ReasonKit Team
**Date**: 2025-12-28
**Related**: `src/telemetry/` module implementation

---

## Executive Summary

This document defines the complete analytics and telemetry schema for ReasonKit, covering:

1. **Local SQLite storage** for OSS users (privacy-first, opt-in)
2. **Cloud event structure** for Pro tier (usage metering, billing)
3. **Aggregation schemas** for dashboards and insights
4. **Billing event format** for metered pricing
5. **GDPR/CCPA compliance** mechanisms

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Local-First** | All data stored in SQLite by default, never sent externally without consent |
| **Privacy-First** | PII stripping, differential privacy, opt-in only |
| **Tiered Collection** | OSS collects less, Pro collects more (with consent) |
| **Auditable** | Full schema versioning, data lineage tracking |
| **GDPR by Default** | Right to deletion, data portability, consent management |

---

## 1. Architecture Overview

```
                              REASONKIT TELEMETRY ARCHITECTURE

    +---------------------------------------------------------------------------+
    |                              USER DEVICE                                   |
    +---------------------------------------------------------------------------+
    |                                                                           |
    |  +-------------------+    +-------------------+    +-------------------+  |
    |  |   reasonkit-core  |    |   reasonkit-pro   |    |   reasonkit-web   |  |
    |  |   (OSS Events)    |    |   (Pro Events)    |    |   (Web Events)    |  |
    |  +--------+----------+    +--------+----------+    +--------+----------+  |
    |           |                        |                        |             |
    |           v                        v                        v             |
    |  +-----------------------------------------------------------------------+|
    |  |                        EVENT COLLECTION LAYER                         ||
    |  |   +------------------+  +------------------+  +------------------+    ||
    |  |   | Privacy Filter   |  | Rate Limiter     |  | Schema Validator |    ||
    |  |   +------------------+  +------------------+  +------------------+    ||
    |  +-----------------------------------------------------------------------+|
    |                                    |                                      |
    |                                    v                                      |
    |  +-----------------------------------------------------------------------+|
    |  |                         LOCAL STORAGE (SQLite)                        ||
    |  |   ~/.local/share/reasonkit/.rk_telemetry.db                          ||
    |  |                                                                       ||
    |  |   +-------------+  +-------------+  +-------------+  +-------------+ ||
    |  |   | sessions    |  | queries     |  | feedback    |  | traces      | ||
    |  |   +-------------+  +-------------+  +-------------+  +-------------+ ||
    |  |   +-------------+  +-------------+  +-------------+  +-------------+ ||
    |  |   | tool_usage  |  | billing_*   |  | aggregates  |  | consent     | ||
    |  |   +-------------+  +-------------+  +-------------+  +-------------+ ||
    |  +-----------------------------------------------------------------------+|
    |                                    |                                      |
    +------------------------------------|--------------------------------------+
                                         | (OPT-IN SYNC)
                                         v
    +---------------------------------------------------------------------------+
    |                           CLOUD SYNC (Pro Only)                           |
    +---------------------------------------------------------------------------+
    |                                                                           |
    |  +-------------------+    +-------------------+    +-------------------+  |
    |  |  Event Ingestion  |--->|  Stream Processing|--->|  Data Warehouse   |  |
    |  |  (HTTPS/gRPC)     |    |  (Kafka/Kinesis)  |    |  (ClickHouse)     |  |
    |  +-------------------+    +-------------------+    +-------------------+  |
    |                                                           |               |
    |                                                           v               |
    |                                              +-------------------+        |
    |                                              |  Billing Service  |        |
    |                                              |  (Stripe/Orb)     |        |
    |                                              +-------------------+        |
    |                                                                           |
    +---------------------------------------------------------------------------+
```

---

## 2. SQLite Schema (Local Storage)

### 2.1 Core Tables

The following schema is implemented in `src/telemetry/schema.rs`:

```sql
-- ============================================================================
-- REASONKIT TELEMETRY SCHEMA v2
-- Local-First Analytics with Privacy Controls
-- ============================================================================

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT,
    migration_hash TEXT  -- SHA-256 of migration SQL for verification
);

-- ============================================================================
-- SESSION TRACKING
-- ============================================================================

-- Sessions: One entry per CLI/API session
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,                    -- UUID v4
    started_at TEXT NOT NULL,               -- ISO 8601 timestamp
    ended_at TEXT,                          -- ISO 8601 (NULL if active)
    duration_ms INTEGER,                    -- Total session duration

    -- Usage counters
    query_count INTEGER DEFAULT 0,
    tool_count INTEGER DEFAULT 0,
    feedback_count INTEGER DEFAULT 0,
    token_count_input INTEGER DEFAULT 0,    -- Total input tokens
    token_count_output INTEGER DEFAULT 0,   -- Total output tokens

    -- Context
    profile TEXT,                           -- Reasoning profile (quick/balanced/deep/paranoid)
    success_rate REAL,                      -- 0.0-1.0 success ratio
    client_version TEXT NOT NULL,           -- ReasonKit version (semver)
    os_family TEXT,                         -- linux/macos/windows (sanitized)
    tier TEXT DEFAULT 'oss',                -- oss/pro/team/enterprise

    -- Pro-only fields (NULL for OSS)
    user_id_hash TEXT,                      -- Hashed user identifier
    organization_id TEXT,                   -- Organization UUID (enterprise)

    -- Indexes
    CONSTRAINT chk_tier CHECK (tier IN ('oss', 'pro', 'team', 'enterprise'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_tier ON sessions(tier);
CREATE INDEX IF NOT EXISTS idx_sessions_org ON sessions(organization_id);

-- ============================================================================
-- QUERY EVENTS
-- ============================================================================

-- Queries: Individual reasoning requests (PII-stripped)
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,                    -- UUID v4
    session_id TEXT NOT NULL,               -- FK to sessions
    timestamp TEXT NOT NULL,                -- ISO 8601

    -- Query metadata (NO raw query text stored - privacy)
    query_hash TEXT NOT NULL,               -- SHA-256 of normalized query
    query_length INTEGER NOT NULL,          -- Character count
    query_token_count INTEGER,              -- Token count (estimated)
    query_type TEXT NOT NULL,               -- search/reason/code/general/file/system

    -- Execution metrics
    latency_ms INTEGER NOT NULL,            -- Total wall-clock time
    llm_latency_ms INTEGER,                 -- Time waiting for LLM
    processing_latency_ms INTEGER,          -- Local processing time

    -- Tool usage
    tool_calls INTEGER DEFAULT 0,
    retrieval_count INTEGER DEFAULT 0,      -- Documents retrieved (RAG)

    -- Results
    result_count INTEGER,
    result_quality_score REAL,              -- Self-assessed 0.0-1.0
    confidence_score REAL,                  -- Final confidence from ThinkTools

    -- Error tracking
    error_occurred INTEGER DEFAULT 0,       -- Boolean
    error_category TEXT,                    -- network/api/parse/timeout/permission/internal
    error_code TEXT,                        -- Specific error code

    -- Context
    profile_used TEXT,                      -- Reasoning profile
    thinktool_chain TEXT,                   -- JSON array: ["gigathink", "laserlogic", ...]
    tools_used TEXT,                        -- JSON array of tool names

    -- Token usage (for billing)
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    tokens_cached INTEGER DEFAULT 0,        -- Cache hits
    model_id TEXT,                          -- Model used (claude-3-5-sonnet, etc.)

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_queries_session ON queries(session_id);
CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);
CREATE INDEX IF NOT EXISTS idx_queries_type ON queries(query_type);
CREATE INDEX IF NOT EXISTS idx_queries_hash ON queries(query_hash);
CREATE INDEX IF NOT EXISTS idx_queries_model ON queries(model_id);

-- ============================================================================
-- USER FEEDBACK
-- ============================================================================

-- Feedback: User ratings and reactions
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,                    -- UUID v4
    session_id TEXT NOT NULL,
    query_id TEXT,                          -- FK to queries (optional)
    timestamp TEXT NOT NULL,

    -- Feedback data
    feedback_type TEXT NOT NULL,            -- thumbs_up/thumbs_down/explicit/implicit
    rating INTEGER,                         -- 1-5 for explicit ratings
    category TEXT,                          -- accuracy/relevance/speed/format/completeness/other

    -- Anonymized context
    context_hash TEXT,                      -- Hash of surrounding context

    -- Optional structured feedback (Pro)
    improvement_areas TEXT,                 -- JSON array of improvement suggestions

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE SET NULL,

    CONSTRAINT chk_rating CHECK (rating IS NULL OR (rating >= 1 AND rating <= 5)),
    CONSTRAINT chk_feedback_type CHECK (feedback_type IN ('thumbs_up', 'thumbs_down', 'explicit', 'implicit'))
);

CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);

-- ============================================================================
-- TOOL USAGE TRACKING
-- ============================================================================

-- Tool usage: Individual tool invocations
CREATE TABLE IF NOT EXISTS tool_usage (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query_id TEXT,
    timestamp TEXT NOT NULL,

    -- Tool identification
    tool_name TEXT NOT NULL,                -- Exact tool name
    tool_category TEXT NOT NULL,            -- search/file/shell/mcp/reasoning/web/other
    mcp_server TEXT,                        -- MCP server name (if applicable)

    -- Execution metrics
    execution_ms INTEGER NOT NULL,
    success INTEGER NOT NULL,               -- Boolean: 1 = success, 0 = failure
    error_type TEXT,                        -- Error classification if failed
    retry_count INTEGER DEFAULT 0,          -- Number of retries

    -- I/O stats (NO content - privacy)
    input_size_bytes INTEGER,
    output_size_bytes INTEGER,

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE SET NULL,

    CONSTRAINT chk_tool_category CHECK (tool_category IN
        ('search', 'file', 'shell', 'mcp', 'reasoning', 'web', 'other'))
);

CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON tool_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON tool_usage(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_usage_category ON tool_usage(tool_category);
CREATE INDEX IF NOT EXISTS idx_tool_usage_mcp ON tool_usage(mcp_server);

-- ============================================================================
-- REASONING TRACES (ThinkTool Execution)
-- ============================================================================

-- Reasoning traces: ThinkTool execution records
CREATE TABLE IF NOT EXISTS reasoning_traces (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query_id TEXT,
    timestamp TEXT NOT NULL,

    -- ThinkTool identification
    thinktool_name TEXT NOT NULL,           -- gigathink/laserlogic/bedrock/proofguard/brutalhonesty
    profile_name TEXT,                      -- Profile used (balanced, deep, etc.)

    -- Execution metrics
    step_count INTEGER NOT NULL,
    total_ms INTEGER NOT NULL,
    avg_step_ms REAL,

    -- Quality metrics (computed)
    coherence_score REAL,                   -- Self-consistency (0.0-1.0)
    depth_score REAL,                       -- Reasoning depth (0.0-1.0)
    confidence_final REAL,                  -- Final confidence output

    -- Token usage per trace
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,

    -- Anonymized structure
    step_types TEXT,                        -- JSON array of step type names

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_session ON reasoning_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_thinktool ON reasoning_traces(thinktool_name);
CREATE INDEX IF NOT EXISTS idx_traces_profile ON reasoning_traces(profile_name);

-- ============================================================================
-- BILLING EVENTS (Pro Tier)
-- ============================================================================

-- Billing events: Usage for metered pricing
CREATE TABLE IF NOT EXISTS billing_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,

    -- Billing dimensions
    event_type TEXT NOT NULL,               -- query/tool_call/trace/storage/api_call
    quantity INTEGER NOT NULL DEFAULT 1,
    unit TEXT NOT NULL,                     -- tokens/calls/mb/minutes

    -- Pricing context
    tier TEXT NOT NULL,                     -- pro/team/enterprise
    feature_flag TEXT,                      -- atomicbreak/highreflect/etc.
    model_id TEXT,                          -- claude-3-5-sonnet, gpt-4, etc.

    -- Cost tracking (in micro-cents for precision)
    estimated_cost_microcents INTEGER,      -- 1 cent = 100 microcents

    -- Sync status
    synced_at TEXT,                         -- NULL if not synced to cloud
    sync_batch_id TEXT,                     -- Batch ID for grouped sync

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,

    CONSTRAINT chk_billing_type CHECK (event_type IN
        ('query', 'tool_call', 'trace', 'storage', 'api_call', 'thinktool'))
);

CREATE INDEX IF NOT EXISTS idx_billing_session ON billing_events(session_id);
CREATE INDEX IF NOT EXISTS idx_billing_type ON billing_events(event_type);
CREATE INDEX IF NOT EXISTS idx_billing_synced ON billing_events(synced_at);
CREATE INDEX IF NOT EXISTS idx_billing_tier ON billing_events(tier);

-- Billing periods: Aggregated billing by period
CREATE TABLE IF NOT EXISTS billing_periods (
    id TEXT PRIMARY KEY,
    period_start TEXT NOT NULL,             -- ISO 8601 (start of billing period)
    period_end TEXT NOT NULL,               -- ISO 8601 (end of billing period)

    -- Aggregated usage
    total_queries INTEGER DEFAULT 0,
    total_tokens_input INTEGER DEFAULT 0,
    total_tokens_output INTEGER DEFAULT 0,
    total_tool_calls INTEGER DEFAULT 0,
    total_traces INTEGER DEFAULT 0,

    -- Cost breakdown (micro-cents)
    cost_queries_microcents INTEGER DEFAULT 0,
    cost_tokens_microcents INTEGER DEFAULT 0,
    cost_features_microcents INTEGER DEFAULT 0,
    total_cost_microcents INTEGER DEFAULT 0,

    -- Sync status
    synced_at TEXT,
    invoice_id TEXT,                        -- Stripe/Orb invoice ID

    CONSTRAINT chk_period CHECK (period_start < period_end)
);

CREATE INDEX IF NOT EXISTS idx_billing_periods_start ON billing_periods(period_start);

-- ============================================================================
-- AGGREGATION TABLES (ML Training Data)
-- ============================================================================

-- Daily aggregates: Pre-computed daily statistics
CREATE TABLE IF NOT EXISTS daily_aggregates (
    date TEXT PRIMARY KEY,                  -- YYYY-MM-DD
    computed_at TEXT NOT NULL,

    -- Volume metrics
    session_count INTEGER DEFAULT 0,
    query_count INTEGER DEFAULT 0,
    feedback_count INTEGER DEFAULT 0,
    tool_invocations INTEGER DEFAULT 0,
    trace_count INTEGER DEFAULT 0,

    -- Performance metrics
    avg_latency_ms REAL,
    p50_latency_ms REAL,
    p95_latency_ms REAL,
    p99_latency_ms REAL,

    -- Quality metrics
    avg_success_rate REAL,
    avg_confidence REAL,
    positive_feedback_ratio REAL,
    error_rate REAL,

    -- Token metrics
    total_tokens_input INTEGER DEFAULT 0,
    total_tokens_output INTEGER DEFAULT 0,

    -- Distribution data (JSON)
    tool_distribution TEXT,                 -- {"Read": 100, "Bash": 50, ...}
    query_type_distribution TEXT,           -- {"search": 30, "reason": 50, ...}
    thinktool_distribution TEXT,            -- {"gigathink": 20, "laserlogic": 15, ...}
    model_distribution TEXT,                -- {"claude-3-5-sonnet": 80, ...}
    error_distribution TEXT                 -- {"network": 5, "timeout": 2, ...}
);

-- Hourly aggregates: For real-time dashboards (Pro)
CREATE TABLE IF NOT EXISTS hourly_aggregates (
    id TEXT PRIMARY KEY,                    -- YYYY-MM-DD-HH
    hour_start TEXT NOT NULL,
    computed_at TEXT NOT NULL,

    -- Volume
    query_count INTEGER DEFAULT 0,
    tool_calls INTEGER DEFAULT 0,

    -- Performance
    avg_latency_ms REAL,
    error_count INTEGER DEFAULT 0,

    -- Tokens
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_hourly_start ON hourly_aggregates(hour_start);

-- Query clusters: K-means clustering for pattern detection
CREATE TABLE IF NOT EXISTS query_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at TEXT NOT NULL,
    valid_until TEXT,                       -- Expiration for stale clusters

    -- Clustering metadata
    algorithm TEXT DEFAULT 'kmeans',        -- kmeans/dbscan/hierarchical
    cluster_count INTEGER NOT NULL,         -- K value
    silhouette_score REAL,                  -- Clustering quality metric

    -- Cluster data (JSON)
    centroids TEXT,                         -- JSON array of centroid embeddings
    cluster_sizes TEXT,                     -- JSON array: [120, 89, 45, ...]
    representative_hashes TEXT,             -- JSON array of representative query hashes
    cluster_labels TEXT                     -- JSON array of human-readable labels
);

-- ============================================================================
-- PRIVACY & CONSENT
-- ============================================================================

-- Privacy consent tracking
CREATE TABLE IF NOT EXISTS privacy_consent (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,

    -- Consent levels (granular)
    local_telemetry INTEGER NOT NULL,       -- Allow local SQLite storage
    aggregated_sharing INTEGER NOT NULL,    -- Allow aggregated (not individual) sharing
    community_contribution INTEGER NOT NULL, -- Contribute to community model training
    cloud_sync INTEGER NOT NULL DEFAULT 0,  -- Allow sync to ReasonKit cloud (Pro)
    billing_sync INTEGER NOT NULL DEFAULT 0, -- Allow billing data sync (Pro)

    -- Metadata
    consent_version INTEGER NOT NULL,       -- Form version for re-consent
    consent_method TEXT,                    -- cli/web/api
    ip_hash TEXT,                           -- Hashed IP for legal compliance

    -- GDPR fields
    data_subject_id TEXT,                   -- Anonymized user identifier
    purpose_ids TEXT,                       -- JSON array of consented purposes

    CONSTRAINT chk_consent_version CHECK (consent_version > 0)
);

CREATE INDEX IF NOT EXISTS idx_consent_timestamp ON privacy_consent(timestamp);

-- Data deletion requests (GDPR Right to Erasure)
CREATE TABLE IF NOT EXISTS deletion_requests (
    id TEXT PRIMARY KEY,
    requested_at TEXT NOT NULL,
    completed_at TEXT,

    -- Request details
    data_subject_id TEXT,
    scope TEXT NOT NULL,                    -- all/sessions/queries/feedback

    -- Execution
    rows_deleted INTEGER,
    status TEXT DEFAULT 'pending',          -- pending/processing/completed/failed
    error_message TEXT
);

-- Redaction log: Audit trail for redacted data
CREATE TABLE IF NOT EXISTS redaction_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,

    -- Redaction details
    source_table TEXT NOT NULL,
    source_id TEXT NOT NULL,
    redaction_type TEXT NOT NULL,           -- pii/sensitive/gdpr_request/retention
    pattern_matched TEXT,                   -- Regex pattern (if PII detection)

    -- Audit
    automated INTEGER DEFAULT 1             -- 1 = automatic, 0 = manual
);

CREATE INDEX IF NOT EXISTS idx_redaction_source ON redaction_log(source_table, source_id);
CREATE INDEX IF NOT EXISTS idx_redaction_type ON redaction_log(redaction_type);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Recent session summary (last 30 days)
CREATE VIEW IF NOT EXISTS v_recent_sessions AS
SELECT
    s.id,
    s.started_at,
    s.duration_ms,
    s.query_count,
    s.tool_count,
    s.token_count_input + s.token_count_output as total_tokens,
    s.success_rate,
    s.profile,
    s.tier,
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
    SUM(output_size_bytes) as total_output_bytes,
    MAX(timestamp) as last_used
FROM tool_usage
WHERE timestamp > datetime('now', '-30 days')
GROUP BY tool_name, tool_category
ORDER BY invocation_count DESC;

-- ThinkTool effectiveness
CREATE VIEW IF NOT EXISTS v_thinktool_stats AS
SELECT
    thinktool_name,
    profile_name,
    COUNT(*) as usage_count,
    AVG(step_count) as avg_steps,
    AVG(total_ms) as avg_execution_ms,
    AVG(coherence_score) as avg_coherence,
    AVG(depth_score) as avg_depth,
    AVG(confidence_final) as avg_confidence,
    AVG(tokens_input + tokens_output) as avg_tokens
FROM reasoning_traces
WHERE timestamp > datetime('now', '-30 days')
GROUP BY thinktool_name, profile_name
ORDER BY usage_count DESC;

-- Daily usage trend
CREATE VIEW IF NOT EXISTS v_daily_trend AS
SELECT
    date(timestamp) as day,
    COUNT(*) as queries,
    AVG(latency_ms) as avg_latency,
    SUM(tokens_input + tokens_output) as total_tokens,
    SUM(CASE WHEN error_occurred = 1 THEN 1 ELSE 0 END) as errors
FROM queries
WHERE timestamp > datetime('now', '-30 days')
GROUP BY date(timestamp)
ORDER BY day DESC;

-- Billing summary (Pro)
CREATE VIEW IF NOT EXISTS v_billing_summary AS
SELECT
    date(timestamp) as day,
    tier,
    event_type,
    SUM(quantity) as total_quantity,
    SUM(estimated_cost_microcents) / 100.0 as estimated_cost_cents
FROM billing_events
WHERE timestamp > datetime('now', '-30 days')
GROUP BY date(timestamp), tier, event_type
ORDER BY day DESC, estimated_cost_cents DESC;
```

### 2.2 Schema Migration Strategy

```sql
-- Migration framework (future migrations go here)
-- Each migration is idempotent and versioned

-- Example: Migration from v1 to v2
-- PRAGMA user_version = 2;
-- ALTER TABLE queries ADD COLUMN tokens_cached INTEGER DEFAULT 0;
-- ALTER TABLE sessions ADD COLUMN tier TEXT DEFAULT 'oss';
-- UPDATE schema_version SET version = 2, applied_at = datetime('now');
```

---

## 3. Event Structures (Cloud Sync Format)

### 3.1 Base Event Envelope

All events share a common envelope for cloud sync:

```json
{
  "schema_version": "2.0.0",
  "event_id": "uuid-v4",
  "event_type": "query|feedback|trace|tool|billing",
  "timestamp": "2025-12-28T10:30:00.000Z",
  "source": {
    "client_version": "0.1.0",
    "client_id_hash": "sha256-hash-of-installation-id",
    "tier": "pro",
    "os_family": "linux"
  },
  "session_id": "uuid-v4",
  "payload": { ... }
}
```

### 3.2 Query Event Payload

```json
{
  "query_hash": "sha256-of-normalized-query",
  "query_length": 150,
  "query_type": "reason",
  "execution": {
    "latency_ms": 2500,
    "llm_latency_ms": 2100,
    "processing_ms": 400
  },
  "tools": {
    "count": 3,
    "names": ["Read", "Grep", "Bash"]
  },
  "retrieval": {
    "count": 5,
    "sources": ["local", "web"]
  },
  "results": {
    "count": 1,
    "quality_score": 0.85,
    "confidence": 0.82
  },
  "tokens": {
    "input": 1500,
    "output": 800,
    "cached": 200,
    "model": "claude-3-5-sonnet-20241022"
  },
  "error": null,
  "profile": "balanced",
  "thinktool_chain": ["gigathink", "laserlogic", "bedrock", "proofguard"]
}
```

### 3.3 Trace Event Payload

```json
{
  "query_id": "uuid-v4",
  "thinktool": {
    "name": "gigathink",
    "version": "1.0.0"
  },
  "profile": "balanced",
  "execution": {
    "step_count": 4,
    "total_ms": 3200,
    "avg_step_ms": 800
  },
  "quality": {
    "coherence": 0.88,
    "depth": 0.75,
    "confidence": 0.82
  },
  "tokens": {
    "input": 2000,
    "output": 1200
  },
  "steps": ["identify_dimensions", "explore_each", "synthesize", "calibrate"]
}
```

### 3.4 Billing Event Payload

```json
{
  "event_type": "thinktool",
  "quantity": 1,
  "unit": "invocation",
  "dimensions": {
    "tier": "pro",
    "feature": "atomicbreak",
    "model": "claude-3-5-sonnet-20241022"
  },
  "tokens": {
    "input": 1500,
    "output": 800
  },
  "cost": {
    "estimated_microcents": 450,
    "pricing_version": "2025-01"
  }
}
```

### 3.5 Feedback Event Payload

```json
{
  "query_id": "uuid-v4",
  "feedback_type": "thumbs_up",
  "rating": null,
  "category": "accuracy",
  "context_hash": "sha256-of-context",
  "improvement_areas": []
}
```

---

## 4. Metrics Aggregation Schema

### 4.1 Dashboard Metrics (Grafana/Prometheus Compatible)

```json
{
  "timestamp": "2025-12-28T10:00:00Z",
  "window": "1h",
  "metrics": {
    "queries": {
      "total": 1250,
      "by_type": {
        "search": 300,
        "reason": 650,
        "code": 200,
        "general": 100
      },
      "by_profile": {
        "quick": 400,
        "balanced": 600,
        "deep": 200,
        "paranoid": 50
      }
    },
    "performance": {
      "latency_p50_ms": 850,
      "latency_p95_ms": 2500,
      "latency_p99_ms": 4200,
      "success_rate": 0.982,
      "error_rate": 0.018
    },
    "tokens": {
      "input_total": 1250000,
      "output_total": 625000,
      "cached_total": 125000,
      "by_model": {
        "claude-3-5-sonnet": 1500000,
        "gpt-4": 250000,
        "gemini-2.0-flash": 125000
      }
    },
    "thinktools": {
      "invocations": 2100,
      "by_tool": {
        "gigathink": 800,
        "laserlogic": 600,
        "bedrock": 400,
        "proofguard": 200,
        "brutalhonesty": 100
      },
      "avg_confidence": 0.82,
      "avg_coherence": 0.85
    },
    "feedback": {
      "total": 180,
      "positive_ratio": 0.78,
      "by_category": {
        "accuracy": 50,
        "relevance": 45,
        "speed": 35,
        "format": 30,
        "completeness": 20
      }
    }
  }
}
```

### 4.2 Billing Aggregates

```json
{
  "period": {
    "start": "2025-12-01T00:00:00Z",
    "end": "2025-12-31T23:59:59Z"
  },
  "organization_id": "org-uuid",
  "tier": "team",
  "usage": {
    "queries": {
      "count": 45000,
      "tokens_input": 67500000,
      "tokens_output": 33750000
    },
    "thinktools": {
      "gigathink": 12000,
      "laserlogic": 9000,
      "bedrock": 7000,
      "proofguard": 5000,
      "brutalhonesty": 3000,
      "atomicbreak": 2000,
      "highreflect": 1500,
      "riskradar": 1000,
      "decidomatic": 800,
      "sciengine": 500
    },
    "storage_mb": 250
  },
  "cost_breakdown": {
    "base_subscription_cents": 9900,
    "token_usage_cents": 4500,
    "pro_features_cents": 2000,
    "overage_cents": 0,
    "total_cents": 16400
  },
  "pricing_version": "2025-01"
}
```

---

## 5. Retention Policies

### 5.1 Default Retention (OSS)

| Data Type | Retention | Rationale |
|-----------|-----------|-----------|
| Sessions | 90 days | Balance between history and storage |
| Queries | 90 days | Privacy-first |
| Feedback | 180 days | Valuable for improvement |
| Tool usage | 90 days | Operational data |
| Traces | 30 days | High volume, quickly stale |
| Daily aggregates | 365 days | Low volume, valuable trends |
| Hourly aggregates | 7 days | High volume, real-time only |

### 5.2 Pro Tier Retention

| Data Type | Retention | Configurable |
|-----------|-----------|--------------|
| Sessions | 365 days | Yes, up to 3 years |
| Queries | 365 days | Yes |
| Billing events | 7 years | No (legal requirement) |
| Aggregates | Unlimited | Yes |

### 5.3 Retention Implementation

```sql
-- Automated cleanup job (run daily)
DELETE FROM queries
WHERE timestamp < datetime('now', '-90 days');

DELETE FROM tool_usage
WHERE timestamp < datetime('now', '-90 days');

DELETE FROM reasoning_traces
WHERE timestamp < datetime('now', '-30 days');

DELETE FROM hourly_aggregates
WHERE hour_start < datetime('now', '-7 days');

-- Vacuum after large deletes
VACUUM;
```

---

## 6. GDPR Compliance

### 6.1 Data Subject Rights Implementation

| Right | Implementation |
|-------|----------------|
| **Right to Access** | `rk telemetry export --format json` |
| **Right to Erasure** | `rk telemetry delete --all` |
| **Right to Portability** | Export in JSON/CSV formats |
| **Right to Rectification** | Feedback correction via API |
| **Right to Restrict** | Pause telemetry collection |
| **Right to Object** | Full opt-out mechanism |

### 6.2 Consent Management

```rust
/// Consent levels (granular)
pub struct ConsentSettings {
    /// Collect local telemetry (stored only on device)
    pub local_telemetry: bool,

    /// Share aggregated (not individual) statistics
    pub aggregated_sharing: bool,

    /// Contribute to community model training
    pub community_contribution: bool,

    /// Sync to ReasonKit cloud (Pro only)
    pub cloud_sync: bool,

    /// Sync billing data (Pro only)
    pub billing_sync: bool,
}

impl Default for ConsentSettings {
    fn default() -> Self {
        Self {
            local_telemetry: false,  // Opt-in only
            aggregated_sharing: false,
            community_contribution: false,
            cloud_sync: false,
            billing_sync: false,
        }
    }
}
```

### 6.3 Data Minimization

1. **Query text is NEVER stored** - only SHA-256 hash
2. **File paths are redacted** to `[USER_PATH]`
3. **PII patterns are detected and stripped**:
   - Email addresses
   - Phone numbers
   - SSN patterns
   - Credit card numbers
   - IP addresses
   - API keys
   - AWS access keys
   - URLs with authentication
4. **Differential privacy** applied to aggregates (configurable epsilon)

### 6.4 CLI Commands

```bash
# View current consent settings
rk telemetry status

# Enable/disable telemetry
rk telemetry enable
rk telemetry disable

# Export all personal data (GDPR access)
rk telemetry export --output my-data.json

# Delete all personal data (GDPR erasure)
rk telemetry delete --confirm

# View what would be deleted
rk telemetry delete --dry-run

# Configure consent levels
rk telemetry consent --local-only
rk telemetry consent --allow-aggregates
rk telemetry consent --community
```

---

## 7. Privacy Controls Summary

### 7.1 What We Collect (OSS, Opt-In)

| Data | Collected | Stored | Shared |
|------|-----------|--------|--------|
| Query text | No | Hash only | Never |
| File paths | Redacted | Redacted | Never |
| Tool names | Yes | Yes | Aggregated only |
| Latency metrics | Yes | Yes | Aggregated only |
| Token counts | Yes | Yes | Aggregated only |
| Error types | Yes | Yes | Aggregated only |
| OS family | Yes | Yes | Aggregated only |
| User identity | No | No | Never |

### 7.2 What We Collect (Pro, With Consent)

Additional data with explicit consent:

| Data | Purpose | Sharing |
|------|---------|---------|
| Billing events | Metered pricing | Stripe/Orb only |
| Organization ID | Team features | Internal only |
| Model usage | Cost allocation | Internal only |
| Feature flags | Feature access | Internal only |

### 7.3 Opt-Out Mechanisms

1. **Environment variable**: `RK_TELEMETRY_ENABLED=false`
2. **Config file**: `~/.config/reasonkit/config.toml` with `telemetry.enabled = false`
3. **CLI command**: `rk telemetry disable`
4. **Build flag**: `--no-telemetry` for air-gapped environments

---

## 8. Implementation Checklist

### 8.1 Current Status (Implemented in `src/telemetry/`)

- [x] SQLite storage backend (`storage.rs`)
- [x] Core schema v1 (`schema.rs`)
- [x] Privacy filter with PII detection (`privacy.rs`)
- [x] Event types (Query, Feedback, Trace) (`events.rs`)
- [x] Configuration with env vars (`config.rs`)
- [x] Consent tracking (`config.rs`)
- [x] Daily aggregation (`storage.rs`)
- [x] Data pruning with retention (`storage.rs`)

### 8.2 To Be Implemented

- [ ] Schema v2 migration (billing tables, hourly aggregates)
- [ ] Cloud sync transport (HTTPS/gRPC)
- [ ] Billing event emission
- [ ] Billing period aggregation
- [ ] Stripe/Orb integration
- [ ] GDPR export command
- [ ] GDPR delete command
- [ ] Real-time dashboard metrics
- [ ] Query clustering (k-means)
- [ ] Differential privacy noise injection

---

## 9. Performance Considerations

### 9.1 Write Performance

- Batch writes (default: 100 events per batch)
- Async flush (configurable interval: 60s default)
- WAL mode for SQLite (concurrent reads during writes)

### 9.2 Storage Limits

- Default max size: 100MB (OSS), 500MB (Pro)
- Auto-prune when approaching limit
- VACUUM after large deletes

### 9.3 Query Performance

- Indexes on all foreign keys
- Indexes on timestamp columns
- Pre-computed aggregates for dashboards

---

## 10. References

- `src/telemetry/mod.rs` - Main telemetry module
- `src/telemetry/schema.rs` - SQLite schema definition
- `src/telemetry/storage.rs` - Storage backend
- `src/telemetry/events.rs` - Event type definitions
- `src/telemetry/privacy.rs` - Privacy filter
- `src/telemetry/config.rs` - Configuration
- GDPR Article 17 - Right to Erasure
- CCPA Section 1798.105 - Right to Delete

---

*ReasonKit Analytics Schema v2.0 | "Privacy-First Telemetry"*
*https://reasonkit.sh*
