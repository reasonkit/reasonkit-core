# ReasonKit Telemetry & Usage Analytics System Design

> Comprehensive Privacy-First Telemetry for Product Intelligence
> "Measure What Matters, Protect What's Private"

**Version**: 1.0.0
**Status**: Design Document
**Author**: ReasonKit Team
**Date**: 2025-12-28
**Related**:

- `src/telemetry/` - Implementation modules
- `docs/design/ANALYTICS_SCHEMA.md` - Database schema
- `docs/SECURITY_AUDIT_CHECKLIST.md` - Security considerations

---

## Executive Summary

This document defines the complete telemetry and usage analytics system for ReasonKit, a CLI tool and library for structured AI reasoning. The system is designed with privacy as the foundational principle, enabling product improvement while respecting user autonomy and data protection regulations.

### Key Design Decisions

| Decision               | Choice                  | Rationale                           |
| ---------------------- | ----------------------- | ----------------------------------- |
| **Default State**      | Opt-in (disabled)       | Privacy-first, builds trust         |
| **Storage**            | Local SQLite first      | User controls their data            |
| **Content Collection** | Never                   | Query/response content never stored |
| **Identifiers**        | Anonymous session-based | No persistent user tracking         |
| **Cloud Sync**         | Opt-in, Pro tier only   | Explicit consent required           |

---

## 1. Telemetry Philosophy

### 1.1 Core Principles

```
+------------------------------------------------------------------+
|                    TELEMETRY PRINCIPLES                           |
+------------------------------------------------------------------+
|                                                                   |
|  1. PRIVACY BY DEFAULT                                            |
|     - Telemetry is OFF until explicitly enabled                   |
|     - Local storage only, no external transmission by default     |
|     - No PII collection under any circumstance                    |
|                                                                   |
|  2. TRANSPARENCY                                                  |
|     - Users can see exactly what data is collected                |
|     - Clear documentation of all event types                      |
|     - Audit trail of what has been sent (if sync enabled)         |
|                                                                   |
|  3. USER CONTROL                                                  |
|     - Granular opt-in/out for different data categories          |
|     - Data export in standard formats (JSON, CSV)                 |
|     - Complete data deletion on request                           |
|                                                                   |
|  4. MINIMAL DATA                                                  |
|     - Collect only what's needed for specific improvements        |
|     - Aggregate rather than individual where possible             |
|     - Automatic data expiration and cleanup                       |
|                                                                   |
|  5. VALUE EXCHANGE                                                |
|     - Users who enable telemetry get better product               |
|     - Local analytics provide personal insights                   |
|     - Community contribution improves everyone's experience       |
|                                                                   |
+------------------------------------------------------------------+
```

### 1.2 What We DO NOT Collect (Ever)

These are hard constraints that can never be violated:

| Category                 | Examples                            | Reason                 |
| ------------------------ | ----------------------------------- | ---------------------- |
| **Query Content**        | User prompts, questions, input text | Privacy, IP protection |
| **Response Content**     | AI responses, generated text        | Privacy, IP protection |
| **API Keys**             | ANTHROPIC_API_KEY, OPENAI_API_KEY   | Security critical      |
| **Personal Information** | Name, email, phone, address         | PII regulations        |
| **File Contents**        | Code, documents, data files         | IP protection          |
| **Network Data**         | IP addresses, hostnames             | Privacy                |
| **System Identifiers**   | MAC address, hardware IDs           | Device fingerprinting  |
| **Credentials**          | Passwords, tokens, certificates     | Security               |
| **Geolocation**          | GPS, precise location               | Privacy                |

### 1.3 What We MAY Collect (With Consent)

| Category                | Examples                   | Purpose                | Consent Level      |
| ----------------------- | -------------------------- | ---------------------- | ------------------ |
| **Command Usage**       | `think --profile balanced` | Feature adoption       | Local              |
| **Performance Metrics** | Latency, error rates       | Quality improvement    | Local              |
| **Tool Usage**          | Which ThinkTools are used  | Feature prioritization | Local              |
| **Error Types**         | `timeout`, `api_error`     | Bug fixing             | Local              |
| **Session Metadata**    | Duration, query count      | Usage patterns         | Local              |
| **Aggregates**          | Daily totals, averages     | Product analytics      | Aggregated Sharing |

---

## 2. Event Taxonomy

### 2.1 Event Categories Overview

```
+------------------------------------------------------------------+
|                      EVENT TAXONOMY                               |
+------------------------------------------------------------------+
|                                                                   |
|  LIFECYCLE EVENTS                                                 |
|  +-- install_started                                              |
|  +-- install_completed                                            |
|  +-- install_failed                                               |
|  +-- install_method (cargo/curl/npm)                              |
|  +-- update_available                                             |
|  +-- update_completed                                             |
|  +-- uninstall                                                    |
|                                                                   |
|  SESSION EVENTS                                                   |
|  +-- session_started                                              |
|  +-- session_ended                                                |
|  +-- session_resumed                                              |
|                                                                   |
|  COMMAND EVENTS                                                   |
|  +-- command_executed                                             |
|  +-- command_failed                                               |
|  +-- command_cancelled                                            |
|                                                                   |
|  THINKTOOL EVENTS                                                 |
|  +-- thinktool_started                                            |
|  +-- thinktool_completed                                          |
|  +-- thinktool_step_executed                                      |
|  +-- profile_used                                                 |
|                                                                   |
|  ERROR EVENTS                                                     |
|  +-- error_occurred                                               |
|  +-- error_recovered                                              |
|  +-- crash_detected                                               |
|                                                                   |
|  FEATURE EVENTS                                                   |
|  +-- feature_discovered                                           |
|  +-- feature_first_use                                            |
|  +-- feature_adoption                                             |
|  +-- help_viewed                                                  |
|                                                                   |
|  UPGRADE EVENTS (Pro)                                             |
|  +-- upgrade_prompt_shown                                         |
|  +-- upgrade_link_clicked                                         |
|  +-- upgrade_completed                                            |
|  +-- trial_started                                                |
|  +-- trial_ended                                                  |
|                                                                   |
+------------------------------------------------------------------+
```

### 2.2 Installation Events

```rust
/// Installation event types
pub enum InstallEvent {
    /// Installation process started
    Started {
        method: InstallMethod,      // cargo/curl/npm/manual
        version: String,            // Target version
        os: String,                 // linux/macos/windows
        arch: String,               // x86_64/aarch64
    },

    /// Installation completed successfully
    Completed {
        method: InstallMethod,
        version: String,
        duration_ms: u64,
        binary_size_bytes: u64,
    },

    /// Installation failed
    Failed {
        method: InstallMethod,
        error_category: InstallErrorCategory,
        // Note: No error messages - may contain paths
    },

    /// Update available notification
    UpdateAvailable {
        current_version: String,
        available_version: String,
        days_since_release: u32,
    },

    /// Uninstall
    Uninstall {
        version: String,
        days_since_install: u32,
        total_sessions: u32,
    },
}

pub enum InstallMethod {
    Cargo,
    Curl,
    Npm,
    Homebrew,
    Manual,
}

pub enum InstallErrorCategory {
    NetworkError,
    PermissionDenied,
    DiskSpace,
    DependencyMissing,
    ArchitectureMismatch,
    Unknown,
}
```

### 2.3 Usage Events

```rust
/// Command execution events
pub struct CommandEvent {
    /// Command name (e.g., "think", "compare", "metrics")
    pub command: String,

    /// Subcommand if applicable (e.g., "think --profile")
    pub subcommand: Option<String>,

    /// Profile used (quick/balanced/deep/paranoid)
    pub profile: Option<String>,

    /// Number of ThinkTools in chain
    pub thinktool_count: u8,

    /// Total execution duration
    pub duration_ms: u64,

    /// Success/failure
    pub success: bool,

    /// Error category (if failed)
    pub error_category: Option<ErrorCategory>,

    /// Token usage (for metering)
    pub tokens: TokenUsage,

    /// Was result from cache
    pub cache_hit: bool,
}

pub struct TokenUsage {
    pub input: u32,
    pub output: u32,
    pub cached: u32,
    pub model: String,  // e.g., "claude-3-5-sonnet"
}

/// ThinkTool execution events
pub struct ThinkToolEvent {
    /// Tool name (gigathink/laserlogic/bedrock/proofguard/brutalhonesty)
    pub tool_name: String,

    /// Number of steps executed
    pub step_count: u8,

    /// Execution time
    pub duration_ms: u64,

    /// Quality metrics (computed, not content-based)
    pub coherence_score: Option<f64>,
    pub depth_score: Option<f64>,
    pub confidence_score: Option<f64>,

    /// Token consumption
    pub tokens_used: u32,

    /// Was this a partial execution (cancelled)
    pub completed: bool,
}
```

### 2.4 Feature Events

```rust
/// Feature discovery and adoption tracking
pub struct FeatureEvent {
    /// Feature identifier
    pub feature_id: String,

    /// Event type
    pub event_type: FeatureEventType,

    /// How feature was discovered
    pub discovery_source: Option<DiscoverySource>,
}

pub enum FeatureEventType {
    /// Feature documentation/help viewed
    Discovered,

    /// First time feature was used
    FirstUse,

    /// Repeated use (milestone: 5, 10, 25, 50, 100 uses)
    Milestone { count: u32 },

    /// Feature was disabled/turned off
    Disabled,
}

pub enum DiscoverySource {
    Help,           // --help output
    Documentation,  // Docs site
    Error,          // Suggested in error message
    Upgrade,        // Upgrade prompt
    Organic,        // User found it themselves
}
```

---

## 3. Data Schema

### 3.1 Event Envelope (Base Structure)

All telemetry events share a common envelope:

```json
{
  "schema_version": "1.0.0",
  "event_id": "uuid-v4",
  "event_type": "command_executed",
  "timestamp": "2025-12-28T10:30:00.000Z",

  "source": {
    "product": "reasonkit-core",
    "version": "1.0.0",
    "tier": "oss",
    "os": "linux",
    "arch": "x86_64",
    "rust_version": "1.74.0"
  },

  "session": {
    "id": "uuid-v4-rotated-daily",
    "sequence": 42,
    "started_at": "2025-12-28T09:00:00.000Z"
  },

  "payload": {
    // Event-specific data
  }
}
```

### 3.2 Anonymization Strategy

```
+------------------------------------------------------------------+
|                    ANONYMIZATION LAYERS                           |
+------------------------------------------------------------------+
|                                                                   |
|  LAYER 1: NO USER IDENTIFIERS                                     |
|  - No user IDs, account IDs, or email addresses                   |
|  - No persistent device identifiers                               |
|  - Session IDs rotate daily                                       |
|                                                                   |
|  LAYER 2: CONTENT HASHING                                         |
|  - Query text -> SHA-256 hash (for dedup only)                    |
|  - File paths -> [USER_PATH] redaction                            |
|  - URLs -> domain only (no path/query params)                     |
|                                                                   |
|  LAYER 3: IP NOT STORED                                           |
|  - Server logs do not retain IP addresses                         |
|  - No geographic precision beyond country (if any)                |
|                                                                   |
|  LAYER 4: DIFFERENTIAL PRIVACY (Aggregates)                       |
|  - Noise injection for aggregate queries                          |
|  - k-anonymity for small groups                                   |
|  - Configurable epsilon (default: 1.0)                            |
|                                                                   |
+------------------------------------------------------------------+
```

### 3.3 Command Event Schema

```json
{
  "event_type": "command_executed",
  "payload": {
    "command": "think",
    "profile": "balanced",
    "thinktool_chain": ["gigathink", "laserlogic", "bedrock", "proofguard"],
    "thinktool_count": 4,

    "execution": {
      "duration_ms": 2500,
      "llm_latency_ms": 2100,
      "processing_ms": 400,
      "success": true
    },

    "tokens": {
      "input": 1500,
      "output": 800,
      "cached": 200,
      "model": "claude-3-5-sonnet-20241022"
    },

    "quality": {
      "confidence_score": 0.82,
      "coherence_score": 0.88
    },

    "error": null
  }
}
```

### 3.4 Error Event Schema

```json
{
  "event_type": "error_occurred",
  "payload": {
    "error_category": "api_error",
    "error_code": "rate_limit_exceeded",
    "recoverable": true,
    "retry_count": 2,

    "context": {
      "command": "think",
      "profile": "deep",
      "step": 3
    }
  }
}
```

**Note**: Error messages are NOT included as they may contain sensitive information (file paths, query content, etc.).

---

## 4. Implementation Architecture

### 4.1 System Architecture

```
+------------------------------------------------------------------+
|                    TELEMETRY ARCHITECTURE                         |
+------------------------------------------------------------------+
|                                                                   |
|  USER DEVICE                                                      |
|  +--------------------------------------------------------------+ |
|  |                                                              | |
|  |  +------------------+    +------------------+                 | |
|  |  | reasonkit-core   |    | reasonkit-pro    |                 | |
|  |  | (OSS Events)     |    | (Pro Events)     |                 | |
|  |  +--------+---------+    +--------+---------+                 | |
|  |           |                       |                           | |
|  |           v                       v                           | |
|  |  +----------------------------------------------+             | |
|  |  |           EVENT EMISSION LAYER               |             | |
|  |  |  +-------------+  +-------------+            |             | |
|  |  |  | Event Queue |  | Batch Buffer|            |             | |
|  |  |  +-------------+  +-------------+            |             | |
|  |  +----------------------------------------------+             | |
|  |                          |                                    | |
|  |                          v                                    | |
|  |  +----------------------------------------------+             | |
|  |  |           PRIVACY FIREWALL                   |             | |
|  |  |  +-------------+  +-------------+            |             | |
|  |  |  | PII Scanner |  | Content     |            |             | |
|  |  |  |             |  | Redactor    |            |             | |
|  |  |  +-------------+  +-------------+            |             | |
|  |  +----------------------------------------------+             | |
|  |                          |                                    | |
|  |                          v                                    | |
|  |  +----------------------------------------------+             | |
|  |  |           LOCAL STORAGE (SQLite)             |             | |
|  |  |  ~/.local/share/reasonkit/.rk_telemetry.db   |             | |
|  |  +----------------------------------------------+             | |
|  |                          |                                    | |
|  +--------------------------|-----------------------------------+ |
|                             | (OPT-IN SYNC ONLY)                  |
|                             v                                     |
|  +--------------------------------------------------------------+ |
|  |                    CLOUD SYNC (Pro Only)                      | |
|  |  +------------------+  +------------------+  +-------------+  | |
|  |  | Event Ingestion  |->| Stream Process  |->| Data        |  | |
|  |  | (HTTPS)          |  | (Aggregation)   |  | Warehouse   |  | |
|  |  +------------------+  +------------------+  +-------------+  | |
|  +--------------------------------------------------------------+ |
+------------------------------------------------------------------+
```

### 4.2 Rust Implementation

```rust
//! Core telemetry module structure (src/telemetry/)

/// Main telemetry collector - thread-safe, async
pub struct TelemetryCollector {
    /// Configuration (loaded from env/file)
    config: TelemetryConfig,

    /// Event buffer for batching
    buffer: Arc<RwLock<EventBuffer>>,

    /// Local SQLite storage
    storage: Arc<RwLock<TelemetryStorage>>,

    /// Privacy filter for sanitization
    privacy: PrivacyFilter,

    /// Current session info
    session: SessionInfo,

    /// Background flush task handle
    flush_handle: Option<JoinHandle<()>>,
}

impl TelemetryCollector {
    /// Create a new telemetry collector
    pub async fn new(config: TelemetryConfig) -> TelemetryResult<Self> {
        // Early return if disabled
        if !config.enabled {
            return Ok(Self::noop());
        }

        // Initialize storage
        let storage = TelemetryStorage::new(&config.db_path).await?;

        // Initialize privacy filter
        let privacy = PrivacyFilter::new(&config.privacy);

        // Create session
        let session = SessionInfo::new();

        // Start background flush task
        let flush_handle = Self::start_flush_task(
            config.flush_interval_secs,
            storage.clone(),
        );

        Ok(Self {
            config,
            buffer: Arc::new(RwLock::new(EventBuffer::new(100))),
            storage: Arc::new(RwLock::new(storage)),
            privacy,
            session,
            flush_handle: Some(flush_handle),
        })
    }

    /// Record a command execution event
    pub async fn record_command(&self, event: CommandEvent) -> TelemetryResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Apply privacy filter
        let sanitized = self.privacy.sanitize_command(event)?;

        // Create envelope
        let envelope = EventEnvelope::new(
            "command_executed",
            &self.session,
            &self.config,
            sanitized,
        );

        // Add to buffer
        self.buffer.write().await.push(envelope);

        // Flush if buffer is full
        if self.buffer.read().await.should_flush() {
            self.flush().await?;
        }

        Ok(())
    }

    /// Record a ThinkTool execution event
    pub async fn record_thinktool(&self, event: ThinkToolEvent) -> TelemetryResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let sanitized = self.privacy.sanitize_thinktool(event)?;
        let envelope = EventEnvelope::new(
            "thinktool_executed",
            &self.session,
            &self.config,
            sanitized,
        );

        self.buffer.write().await.push(envelope);

        if self.buffer.read().await.should_flush() {
            self.flush().await?;
        }

        Ok(())
    }

    /// Record an error event
    pub async fn record_error(&self, event: ErrorEvent) -> TelemetryResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Errors are stripped of all content
        let sanitized = self.privacy.sanitize_error(event)?;
        let envelope = EventEnvelope::new(
            "error_occurred",
            &self.session,
            &self.config,
            sanitized,
        );

        // Errors flush immediately (important for crash detection)
        self.storage.write().await.insert(envelope).await?;

        Ok(())
    }

    /// Flush buffered events to storage
    pub async fn flush(&self) -> TelemetryResult<()> {
        let events = self.buffer.write().await.drain();
        if events.is_empty() {
            return Ok(());
        }

        self.storage.write().await.insert_batch(events).await
    }

    /// Get aggregated metrics for local display
    pub async fn get_metrics(&self) -> TelemetryResult<LocalMetrics> {
        self.storage.read().await.aggregate_local().await
    }

    /// Export all data (GDPR right to access)
    pub async fn export_all(&self) -> TelemetryResult<Vec<u8>> {
        self.storage.read().await.export_json().await
    }

    /// Delete all data (GDPR right to erasure)
    pub async fn delete_all(&self) -> TelemetryResult<()> {
        self.storage.write().await.delete_all().await
    }
}

/// Event buffer for batching
struct EventBuffer {
    events: Vec<EventEnvelope>,
    capacity: usize,
}

impl EventBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, event: EventEnvelope) {
        self.events.push(event);
    }

    fn should_flush(&self) -> bool {
        self.events.len() >= self.capacity
    }

    fn drain(&mut self) -> Vec<EventEnvelope> {
        std::mem::take(&mut self.events)
    }
}
```

### 4.3 Python SDK Integration

```python
"""
ReasonKit Python SDK Telemetry Integration

Example usage:
    from reasonkit import Reasoner

    # Telemetry is opt-in
    reasoner = Reasoner(telemetry=True)

    # Or via environment
    # export RK_TELEMETRY_ENABLED=true
"""

import os
import json
import uuid
import sqlite3
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, List
from pathlib import Path
import threading


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""
    enabled: bool = False
    db_path: Optional[Path] = None
    batch_size: int = 100
    flush_interval_secs: int = 60
    strip_pii: bool = True

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        """Load config from environment variables."""
        enabled = os.getenv("RK_TELEMETRY_ENABLED", "false").lower() in ("true", "1")
        db_path = os.getenv("RK_TELEMETRY_PATH")

        return cls(
            enabled=enabled,
            db_path=Path(db_path) if db_path else cls.default_db_path(),
        )

    @staticmethod
    def default_db_path() -> Path:
        """Get default database path."""
        data_dir = Path.home() / ".local" / "share" / "reasonkit"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / ".rk_telemetry.db"


@dataclass
class CommandEvent:
    """Command execution event."""
    command: str
    profile: Optional[str] = None
    thinktool_count: int = 0
    duration_ms: int = 0
    success: bool = True
    error_category: Optional[str] = None
    tokens_input: int = 0
    tokens_output: int = 0
    model: str = ""


class TelemetryCollector:
    """Python telemetry collector."""

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig.from_env()
        self._session_id = str(uuid.uuid4())
        self._buffer: List[dict] = []
        self._lock = threading.Lock()

        if self.config.enabled:
            self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        if self.config.db_path is None:
            return

        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.config.db_path), check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """Create telemetry tables."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def record_command(self, event: CommandEvent) -> None:
        """Record a command execution event."""
        if not self.config.enabled:
            return

        envelope = {
            "id": str(uuid.uuid4()),
            "event_type": "command_executed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "payload": asdict(event),
        }

        with self._lock:
            self._buffer.append(envelope)
            if len(self._buffer) >= self.config.batch_size:
                self._flush()

    def _flush(self) -> None:
        """Flush buffered events to storage."""
        if not self._buffer:
            return

        events = self._buffer
        self._buffer = []

        for event in events:
            self._conn.execute(
                "INSERT INTO events (id, event_type, timestamp, session_id, payload) VALUES (?, ?, ?, ?, ?)",
                (event["id"], event["event_type"], event["timestamp"], event["session_id"], json.dumps(event["payload"]))
            )
        self._conn.commit()

    def get_status(self) -> dict:
        """Get telemetry status for display."""
        if not self.config.enabled:
            return {"enabled": False}

        cursor = self._conn.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]

        return {
            "enabled": True,
            "session_id": self._session_id[:8] + "...",
            "events_stored": count,
            "db_path": str(self.config.db_path),
        }

    def export(self) -> str:
        """Export all telemetry data as JSON."""
        if not self.config.enabled:
            return "[]"

        cursor = self._conn.execute("SELECT * FROM events ORDER BY timestamp DESC")
        events = [
            {
                "id": row[0],
                "event_type": row[1],
                "timestamp": row[2],
                "session_id": row[3],
                "payload": json.loads(row[4]),
            }
            for row in cursor.fetchall()
        ]

        return json.dumps(events, indent=2)

    def delete_all(self) -> int:
        """Delete all telemetry data. Returns count of deleted events."""
        if not self.config.enabled:
            return 0

        cursor = self._conn.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]

        self._conn.execute("DELETE FROM events")
        self._conn.commit()

        return count

    def __del__(self):
        """Cleanup on destruction."""
        if self.config.enabled and hasattr(self, '_conn'):
            self._flush()
            self._conn.close()
```

### 4.4 Transport Layer

```rust
/// Cloud sync transport (Pro tier only)
pub struct TelemetryTransport {
    /// HTTPS client
    client: reqwest::Client,

    /// Endpoint URL
    endpoint: String,

    /// Retry configuration
    retry_config: RetryConfig,

    /// Offline queue
    offline_queue: Arc<RwLock<Vec<EventEnvelope>>>,
}

impl TelemetryTransport {
    pub fn new(endpoint: &str) -> Self {
        let client = reqwest::Client::builder()
            // TLS 1.3 minimum
            .min_tls_version(reqwest::tls::Version::TLS_1_3)
            // Connection timeout
            .connect_timeout(Duration::from_secs(10))
            // Request timeout
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            endpoint: endpoint.to_string(),
            retry_config: RetryConfig::default(),
            offline_queue: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Send events to cloud endpoint
    pub async fn send(&self, events: Vec<EventEnvelope>) -> Result<(), TransportError> {
        let payload = serde_json::to_vec(&events)?;

        // Retry with exponential backoff
        let mut attempts = 0;
        let mut delay = self.retry_config.initial_delay;

        loop {
            match self.client
                .post(&self.endpoint)
                .header("Content-Type", "application/json")
                .body(payload.clone())
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    return Ok(());
                }
                Ok(response) if response.status().as_u16() == 429 => {
                    // Rate limited - queue for later
                    self.offline_queue.write().await.extend(events.clone());
                    return Err(TransportError::RateLimited);
                }
                Ok(response) => {
                    attempts += 1;
                    if attempts >= self.retry_config.max_retries {
                        self.offline_queue.write().await.extend(events.clone());
                        return Err(TransportError::ServerError(response.status().as_u16()));
                    }
                }
                Err(_) => {
                    attempts += 1;
                    if attempts >= self.retry_config.max_retries {
                        // Network error - queue for later
                        self.offline_queue.write().await.extend(events.clone());
                        return Err(TransportError::NetworkError);
                    }
                }
            }

            // Exponential backoff
            tokio::time::sleep(delay).await;
            delay = std::cmp::min(delay * 2, self.retry_config.max_delay);
        }
    }

    /// Flush offline queue when connectivity restored
    pub async fn flush_offline_queue(&self) -> Result<usize, TransportError> {
        let events = self.offline_queue.write().await.drain(..).collect::<Vec<_>>();
        if events.is_empty() {
            return Ok(0);
        }

        let count = events.len();
        self.send(events).await?;
        Ok(count)
    }
}

struct RetryConfig {
    max_retries: u32,
    initial_delay: Duration,
    max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
        }
    }
}
```

---

## 5. User Control Interface

### 5.1 CLI Commands

```bash
# =============================================================================
# TELEMETRY STATUS AND CONTROL
# =============================================================================

# Show current telemetry status
rk-core telemetry status

# Example output:
# Telemetry Status
# ================
# Enabled:        false (opt-in required)
# Database:       ~/.local/share/reasonkit/.rk_telemetry.db
# Events Stored:  0
# Last Flush:     never
# Cloud Sync:     disabled (Pro feature)

# Enable telemetry (interactive consent)
rk-core telemetry enable

# Example output:
# ReasonKit Telemetry Consent
# ===========================
#
# We collect anonymous usage data to improve ReasonKit.
#
# What we collect:
#   - Command usage (which commands, not content)
#   - Performance metrics (latency, error rates)
#   - ThinkTool usage patterns
#   - Error types (not messages)
#
# What we NEVER collect:
#   - Query/response content
#   - API keys or credentials
#   - Personal information
#   - File contents
#
# You can disable telemetry at any time with:
#   rk-core telemetry disable
#
# Do you consent to telemetry collection? [y/N]: y
#
# Telemetry enabled. Thank you for helping improve ReasonKit!

# Disable telemetry
rk-core telemetry disable

# Example output:
# Telemetry disabled.
# Existing data retained. Use 'rk-core telemetry delete' to remove.

# =============================================================================
# DATA TRANSPARENCY
# =============================================================================

# Show what data would be collected for a command (dry run)
rk-core telemetry show

# Example output:
# Pending Telemetry Events (not yet flushed)
# ==========================================
#
# Event 1: command_executed
#   command: think
#   profile: balanced
#   duration_ms: 2543
#   success: true
#   thinktool_count: 4
#   tokens_input: 1500
#   tokens_output: 800
#
# Event 2: thinktool_executed
#   tool_name: gigathink
#   step_count: 4
#   duration_ms: 650
#   confidence_score: 0.82
#
# Total: 2 events pending

# Explain what telemetry collects and why
rk-core telemetry explain

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

# Export all telemetry data (GDPR right to access)
rk-core telemetry export --output my-data.json

# Export as CSV
rk-core telemetry export --output my-data.csv --format csv

# Delete all telemetry data (GDPR right to erasure)
rk-core telemetry delete

# Example output:
# This will permanently delete all telemetry data.
#
# Events to be deleted: 1,234
# Database size: 2.4 MB
#
# Are you sure? [y/N]: y
#
# Deleted 1,234 events.
# Database vacuumed, freed 2.4 MB.

# Delete with confirmation skip (for automation)
rk-core telemetry delete --confirm

# =============================================================================
# LOCAL ANALYTICS
# =============================================================================

# View your personal usage statistics
rk-core telemetry metrics

# Example output:
# Personal Usage Metrics (Last 30 Days)
# =====================================
#
# Sessions:        42
# Total Queries:   1,234
# Total Duration:  4h 23m
#
# Command Usage:
#   think:         856 (69.4%)
#   compare:       234 (19.0%)
#   metrics:       89 (7.2%)
#   other:         55 (4.4%)
#
# Profile Usage:
#   balanced:      612 (49.6%)
#   quick:         345 (28.0%)
#   deep:          189 (15.3%)
#   paranoid:      88 (7.1%)
#
# ThinkTool Usage:
#   gigathink:     1,234 calls
#   laserlogic:    987 calls
#   bedrock:       756 calls
#   proofguard:    623 calls
#   brutalhonesty: 412 calls
#
# Performance:
#   Avg Latency:   1,234ms
#   Success Rate:  98.2%
#   Cache Hit:     34.5%
#
# Tokens Used:
#   Input:         1.2M
#   Output:        0.6M
#   Est. Cost:     $4.50
```

### 5.2 Environment Variables

```bash
# Completely disable telemetry (overrides all other settings)
export RK_TELEMETRY_ENABLED=false
# or
export REASONKIT_TELEMETRY=false

# Custom database location
export RK_TELEMETRY_PATH=/custom/path/telemetry.db

# Enable community contribution (opt-in)
export RK_TELEMETRY_COMMUNITY=true

# Pro: Enable cloud sync (requires authentication)
export RK_TELEMETRY_CLOUD_SYNC=true
```

### 5.3 Configuration File

```toml
# ~/.config/reasonkit/config.toml

[telemetry]
# Master switch (default: false)
enabled = true

# Database location (default: ~/.local/share/reasonkit/.rk_telemetry.db)
# db_path = "/custom/path/telemetry.db"

# Contribute anonymized data to community model (default: false)
community_contribution = false

# Batch size before flush (default: 100)
batch_size = 100

# Auto-flush interval in seconds (default: 60)
flush_interval_secs = 60

# Maximum database size in MB (default: 100)
max_db_size_mb = 100

# Data retention in days (default: 90)
retention_days = 90

[telemetry.privacy]
# Strip PII patterns from all data (default: true)
strip_pii = true

# Block events containing sensitive keywords (default: false)
block_sensitive = false

# Redact file paths to [USER_PATH] (default: true)
redact_file_paths = true

# Apply differential privacy to aggregates (default: false)
differential_privacy = false

# Epsilon for differential privacy (default: 1.0, lower = more private)
dp_epsilon = 1.0
```

---

## 6. Data Pipeline (Cloud Sync)

### 6.1 Collection Layer

```
+------------------------------------------------------------------+
|                    DATA COLLECTION PIPELINE                       |
+------------------------------------------------------------------+
|                                                                   |
|  CLIENT SIDE                                                      |
|  +--------------------------+                                     |
|  |  Event Buffer            |                                     |
|  |  (In-memory, max 100)    |                                     |
|  +-----------+--------------+                                     |
|              |                                                    |
|              v                                                    |
|  +--------------------------+                                     |
|  |  Privacy Firewall        |                                     |
|  |  - PII scanning          |                                     |
|  |  - Content redaction     |                                     |
|  |  - Path sanitization     |                                     |
|  +-----------+--------------+                                     |
|              |                                                    |
|              v                                                    |
|  +--------------------------+                                     |
|  |  Local SQLite            |                                     |
|  |  (Persistent storage)    |                                     |
|  +-----------+--------------+                                     |
|              |                                                    |
|              | (If cloud sync enabled)                            |
|              v                                                    |
|  +--------------------------+                                     |
|  |  Transport Layer         |                                     |
|  |  - HTTPS POST            |                                     |
|  |  - Batched (max 1000)    |                                     |
|  |  - Retry with backoff    |                                     |
|  |  - Offline queue         |                                     |
|  +-----------+--------------+                                     |
|              |                                                    |
+--------------|------------------------------------------------+   |
               |                                                    |
|  SERVER SIDE |                                                    |
|              v                                                    |
|  +--------------------------+                                     |
|  |  Event Ingestion API     |                                     |
|  |  api.reasonkit.sh/v1/    |                                     |
|  |  telemetry/events        |                                     |
|  +-----------+--------------+                                     |
|              |                                                    |
|              v                                                    |
|  +--------------------------+                                     |
|  |  Schema Validation       |                                     |
|  |  - Version check         |                                     |
|  |  - Required fields       |                                     |
|  |  - Type validation       |                                     |
|  +-----------+--------------+                                     |
|              |                                                    |
|              v                                                    |
|  +--------------------------+                                     |
|  |  Server Privacy Filter   |                                     |
|  |  - Double-check PII      |                                     |
|  |  - IP stripping          |                                     |
|  |  - Anomaly detection     |                                     |
|  +-----------+--------------+                                     |
|              |                                                    |
|              v                                                    |
|  +--------------------------+                                     |
|  |  Stream Processing       |                                     |
|  |  (Kafka/Kinesis)         |                                     |
|  +-----------+--------------+                                     |
|              |                                                    |
|              v                                                    |
|  +--------------------------+                                     |
|  |  Data Warehouse          |                                     |
|  |  (ClickHouse)            |                                     |
|  +--------------------------+                                     |
|                                                                   |
+------------------------------------------------------------------+
```

### 6.2 Storage & Retention

| Data Tier             | Storage    | Retention | Purpose              |
| --------------------- | ---------- | --------- | -------------------- |
| **Raw Events**        | ClickHouse | 90 days   | Debugging, analysis  |
| **Hourly Aggregates** | ClickHouse | 1 year    | Real-time dashboards |
| **Daily Aggregates**  | PostgreSQL | 3 years   | Trend analysis       |
| **Monthly Summaries** | PostgreSQL | Forever   | Historical reports   |

### 6.3 Data Lifecycle

```
Event Lifecycle:
================

Day 0:      Event generated on client
            -> Stored in local SQLite
            -> Synced to cloud (if enabled)

Day 1-90:   Raw event available for analysis
            -> Included in hourly/daily aggregates

Day 91:     Raw event deleted from data warehouse
            -> Aggregates retained

Year 1+:    Daily aggregates downsampled to monthly
            -> Hourly aggregates deleted

Year 3+:    Monthly summaries only
            -> Daily aggregates deleted
```

---

## 7. Analytics & Dashboards

### 7.1 Product Analytics Metrics

```
+------------------------------------------------------------------+
|                    KEY METRICS FRAMEWORK                          |
+------------------------------------------------------------------+
|                                                                   |
|  ENGAGEMENT METRICS                                               |
|  +--------------------------+                                     |
|  | DAU (Daily Active Users) |  Unique sessions per day            |
|  | WAU (Weekly Active)      |  Unique sessions per week           |
|  | MAU (Monthly Active)     |  Unique sessions per month          |
|  | DAU/MAU Ratio            |  Stickiness indicator               |
|  | Session Duration         |  Time per session                   |
|  | Queries per Session      |  Depth of engagement                |
|  +--------------------------+                                     |
|                                                                   |
|  FEATURE ADOPTION                                                 |
|  +--------------------------+                                     |
|  | ThinkTool Usage          |  Which tools are used most          |
|  | Profile Distribution     |  quick/balanced/deep/paranoid       |
|  | Feature Discovery        |  How users find features            |
|  | Feature Stickiness       |  Repeated use after discovery       |
|  | Pro Feature Interest     |  Upgrade prompt interactions        |
|  +--------------------------+                                     |
|                                                                   |
|  QUALITY METRICS                                                  |
|  +--------------------------+                                     |
|  | Success Rate             |  Commands completed successfully    |
|  | Error Rate by Type       |  Network, API, timeout, etc.        |
|  | Latency Percentiles      |  P50, P95, P99                      |
|  | Confidence Scores        |  ThinkTool output quality           |
|  +--------------------------+                                     |
|                                                                   |
|  GROWTH METRICS                                                   |
|  +--------------------------+                                     |
|  | Install Rate             |  New installations per day          |
|  | Install Method           |  cargo/curl/npm distribution        |
|  | Retention (D1, D7, D30)  |  Return rate after install          |
|  | Churn Rate               |  Users who stop using               |
|  +--------------------------+                                     |
|                                                                   |
+------------------------------------------------------------------+
```

### 7.2 Funnel Analysis

```
+------------------------------------------------------------------+
|                    CONVERSION FUNNELS                             |
+------------------------------------------------------------------+
|                                                                   |
|  INSTALL -> FIRST RUN                                             |
|  +--------------------------+                                     |
|  | install_completed        |  100%  (baseline)                   |
|  | session_started          |   85%  (15% never run)              |
|  | first_command_executed   |   72%  (13% don't complete)         |
|  | command_success          |   68%  (4% hit errors)              |
|  +--------------------------+                                     |
|                                                                   |
|  FIRST RUN -> ACTIVATION                                          |
|  +--------------------------+                                     |
|  | first_command_success    |  100%  (baseline)                   |
|  | second_session           |   45%  (D1 retention)               |
|  | thinktool_discovery      |   38%  (used non-default profile)   |
|  | week_active              |   28%  (D7 retention)               |
|  +--------------------------+                                     |
|                                                                   |
|  FREE -> PAID (Pro)                                               |
|  +--------------------------+                                     |
|  | upgrade_prompt_shown     |  100%  (baseline)                   |
|  | upgrade_link_clicked     |   12%                               |
|  | pricing_page_viewed      |   10%                               |
|  | trial_started            |    4%                               |
|  | upgrade_completed        |    2%                               |
|  +--------------------------+                                     |
|                                                                   |
+------------------------------------------------------------------+
```

### 7.3 Cohort Analysis

```sql
-- Example: Weekly cohort retention analysis
WITH cohorts AS (
    SELECT
        date_trunc('week', first_session_at) AS cohort_week,
        session_hash,
        first_session_at
    FROM sessions
    WHERE first_session_at >= current_date - interval '12 weeks'
),
activity AS (
    SELECT
        c.cohort_week,
        c.session_hash,
        date_trunc('week', s.started_at) AS activity_week,
        EXTRACT(week FROM s.started_at - c.first_session_at) AS weeks_since_join
    FROM cohorts c
    JOIN sessions s ON c.session_hash = s.session_hash
)
SELECT
    cohort_week,
    weeks_since_join,
    COUNT(DISTINCT session_hash) AS active_users,
    COUNT(DISTINCT session_hash)::float /
        FIRST_VALUE(COUNT(DISTINCT session_hash))
        OVER (PARTITION BY cohort_week ORDER BY weeks_since_join) AS retention_rate
FROM activity
GROUP BY cohort_week, weeks_since_join
ORDER BY cohort_week, weeks_since_join;
```

---

## 8. Privacy Compliance

### 8.1 GDPR Compliance

| Article     | Requirement       | Implementation                                         |
| ----------- | ----------------- | ------------------------------------------------------ |
| **Art. 6**  | Lawful Basis      | Legitimate interest (product improvement) with opt-out |
| **Art. 7**  | Consent           | Clear opt-in, documented consent record                |
| **Art. 12** | Transparency      | `rk-core telemetry explain` command                    |
| **Art. 15** | Right to Access   | `rk-core telemetry export` command                     |
| **Art. 17** | Right to Erasure  | `rk-core telemetry delete` command                     |
| **Art. 20** | Data Portability  | Export in JSON/CSV formats                             |
| **Art. 21** | Right to Object   | `rk-core telemetry disable` command                    |
| **Art. 25** | Privacy by Design | Opt-in default, minimal collection                     |
| **Art. 32** | Security          | TLS 1.3, encryption at rest, access controls           |

### 8.2 CCPA Compliance

| Requirement              | Implementation                           |
| ------------------------ | ---------------------------------------- |
| **Right to Know**        | Full event list in `telemetry explain`   |
| **Right to Delete**      | Complete deletion via `telemetry delete` |
| **Right to Opt-Out**     | Telemetry disabled by default            |
| **Non-Discrimination**   | No feature reduction for opt-out         |
| **Notice at Collection** | Consent prompt on first enable           |

### 8.3 Privacy Documentation

Required documentation for privacy compliance:

1. **Privacy Policy Section**

   ```markdown
   ## Telemetry Data Collection

   ReasonKit collects anonymous usage data to improve the product.

   ### What We Collect

   - Command usage (which commands, not content)
   - Performance metrics (latency, error rates)
   - ThinkTool usage patterns
   - Error types (not error messages)
   - Platform information (OS, architecture)

   ### What We Never Collect

   - Query or response content
   - API keys or credentials
   - Personal information (name, email, IP)
   - File contents or paths

   ### Your Rights

   - **View**: `rk-core telemetry status`
   - **Export**: `rk-core telemetry export`
   - **Delete**: `rk-core telemetry delete`
   - **Opt-out**: `rk-core telemetry disable`

   ### Data Retention

   - Local: Configurable, default 90 days
   - Cloud (Pro): Raw events 90 days, aggregates 3 years
   ```

2. **Data Processing Agreement** (for Pro cloud sync)

3. **Security Measures Documentation**

---

## 9. Security

### 9.1 Transport Security

```rust
/// Security configuration for telemetry transport
pub struct TransportSecurity {
    /// Minimum TLS version (1.3 required)
    pub min_tls_version: TlsVersion,

    /// Certificate pinning (optional, for high-security deployments)
    pub certificate_pins: Option<Vec<String>>,

    /// Request signing (HMAC-SHA256)
    pub sign_requests: bool,
}

impl Default for TransportSecurity {
    fn default() -> Self {
        Self {
            min_tls_version: TlsVersion::Tls13,
            certificate_pins: None,  // Optional
            sign_requests: true,
        }
    }
}
```

### 9.2 Data Security

| Layer               | Protection        | Implementation           |
| ------------------- | ----------------- | ------------------------ |
| **In Transit**      | TLS 1.3           | reqwest with rustls      |
| **At Rest (Local)** | OS permissions    | SQLite file 0600         |
| **At Rest (Cloud)** | AES-256-GCM       | ClickHouse encryption    |
| **Access Control**  | Role-based        | Internal dashboards only |
| **Audit Logging**   | All access logged | Security audit trail     |

### 9.3 Incident Response

```
+------------------------------------------------------------------+
|                    SECURITY INCIDENT RESPONSE                     |
+------------------------------------------------------------------+
|                                                                   |
|  DETECTION                                                        |
|  - Automated anomaly detection on event patterns                  |
|  - Rate limit monitoring (DoS protection)                         |
|  - PII leak detection in server-side filter                       |
|                                                                   |
|  RESPONSE                                                         |
|  1. Isolate affected data                                         |
|  2. Notify affected users (if identifiable)                       |
|  3. Root cause analysis                                           |
|  4. Remediation and prevention                                    |
|                                                                   |
|  CONTACT                                                          |
|  security@reasonkit.sh                                            |
|                                                                   |
+------------------------------------------------------------------+
```

---

## 10. Reporting

### 10.1 Internal Reports

| Report                | Frequency | Audience    | Content                                 |
| --------------------- | --------- | ----------- | --------------------------------------- |
| **Weekly Metrics**    | Weekly    | Engineering | Feature usage, error rates, performance |
| **Monthly Deep Dive** | Monthly   | Product     | Funnel analysis, cohorts, growth        |
| **Quarterly Review**  | Quarterly | Leadership  | Strategic metrics, YoY trends           |

### 10.2 Public Transparency Report

Published annually, includes:

- Total telemetry events processed
- Data deletion requests fulfilled
- Security incidents (if any)
- Privacy improvements made
- Aggregated usage statistics

**Note**: No individual user data ever published.

---

## 11. Implementation Checklist

### 11.1 Phase 1: Local Telemetry (Current)

- [x] SQLite storage backend (`src/telemetry/storage.rs`)
- [x] Core schema v1 (`src/telemetry/schema.rs`)
- [x] Privacy filter with PII detection (`src/telemetry/privacy.rs`)
- [x] Event types (Query, Feedback, Trace) (`src/telemetry/events.rs`)
- [x] Configuration with env vars (`src/telemetry/config.rs`)
- [x] Consent tracking
- [x] Daily aggregation
- [x] Data pruning with retention

### 11.2 Phase 2: User Control (Next)

- [ ] `rk-core telemetry status` command
- [ ] `rk-core telemetry enable` with consent flow
- [ ] `rk-core telemetry disable` command
- [ ] `rk-core telemetry show` command
- [ ] `rk-core telemetry explain` command
- [ ] `rk-core telemetry export` (JSON/CSV)
- [ ] `rk-core telemetry delete` with confirmation
- [ ] `rk-core telemetry metrics` local analytics

### 11.3 Phase 3: Cloud Sync (Pro)

- [ ] Transport layer with retry/backoff
- [ ] Offline queue for connectivity issues
- [ ] Server-side ingestion API
- [ ] Stream processing pipeline
- [ ] ClickHouse schema deployment
- [ ] Aggregation ETL jobs
- [ ] Dashboard creation

### 11.4 Phase 4: Analytics (Future)

- [ ] Product analytics dashboard
- [ ] Funnel visualization
- [ ] Cohort analysis tooling
- [ ] Alerting on anomalies
- [ ] A/B testing framework integration

---

## 12. References

### Internal Documents

- `src/telemetry/mod.rs` - Main telemetry module
- `src/telemetry/schema.rs` - SQLite schema definition
- `src/telemetry/storage.rs` - Storage backend
- `src/telemetry/events.rs` - Event type definitions
- `src/telemetry/privacy.rs` - Privacy filter
- `src/telemetry/config.rs` - Configuration
- `docs/design/ANALYTICS_SCHEMA.md` - Full schema specification
- `docs/SECURITY_AUDIT_CHECKLIST.md` - Security requirements

### External References

- GDPR Text: https://gdpr-info.eu/
- CCPA Text: https://oag.ca.gov/privacy/ccpa
- ClickHouse Docs: https://clickhouse.com/docs
- Differential Privacy: https://desfontain.es/privacy/differential-privacy-awesomeness.html

---

## Version History

| Version | Date       | Changes                 |
| ------- | ---------- | ----------------------- |
| 1.0.0   | 2025-12-28 | Initial design document |

---

_ReasonKit Telemetry Design v1.0 | "Measure What Matters, Protect What's Private"_
*https://reasonkit.sh*
