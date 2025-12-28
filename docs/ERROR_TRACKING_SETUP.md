# Error Tracking and Monitoring Setup Guide

> Production-Grade Observability for ReasonKit
> "See Every Error, Understand Every Failure"

**Version:** 1.0.0
**Last Updated:** 2025-12-28
**Audience:** Developers integrating ReasonKit, DevOps Engineers

---

## Table of Contents

1. [Overview](#1-overview)
2. [Error Tracking Architecture](#2-error-tracking-architecture)
3. [Rust Error Handling](#3-rust-error-handling)
4. [Python Bindings Error Handling](#4-python-bindings-error-handling)
5. [Error Reporting Service Setup](#5-error-reporting-service-setup)
6. [Error Context and Enrichment](#6-error-context-and-enrichment)
7. [Privacy Considerations](#7-privacy-considerations)
8. [Alerting and Triage](#8-alerting-and-triage)
9. [Dashboard and Reporting](#9-dashboard-and-reporting)
10. [User Error Reporting](#10-user-error-reporting)
11. [Implementation Checklist](#11-implementation-checklist)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

### Purpose

This guide provides a complete strategy for implementing production-grade error tracking and monitoring for ReasonKit. It covers both the Rust CLI (`rk-core`) and Python bindings (`reasonkit`), ensuring comprehensive observability across all integration points.

### Architecture Summary

```
                           Error Sources
                                |
        +------------+----------+----------+------------+
        |            |          |          |            |
   CLI Errors   Library    API Errors  Integration  User-Facing
                Errors                   Errors       Errors
        |            |          |          |            |
        +------------+----------+----------+------------+
                                |
                    +------------------------+
                    |   Error Classification  |
                    |   (Fatal/Error/Warn/Info)|
                    +------------------------+
                                |
        +------------+----------+----------+------------+
        |            |          |          |            |
   Local         Telemetry    Sentry/    Alerting    Dashboard
   Logging       System       Self-Host   System      Metrics
        |            |          |          |            |
        +------------+----------+----------+------------+
                                |
                    +------------------------+
                    |   Privacy Firewall     |
                    |   (PII Scrubbing)      |
                    +------------------------+
```

### Key Principles

| Principle | Description |
|-----------|-------------|
| **Privacy First** | PII scrubbing before any external transmission |
| **Local First** | All data stored locally by default, external opt-in |
| **Fail Loud** | Errors surface immediately with actionable context |
| **Structured** | Consistent error formats across Rust and Python |
| **Traceable** | Full stack traces and execution context |

---

## 2. Error Tracking Architecture

### 2.1 Components to Monitor

#### CLI Application Errors

```rust
// Component: src/main.rs, src/bin/*.rs
// Error types: Command parsing, file I/O, configuration
// Severity: Fatal (crash), Error (command failure)

// Examples:
// - Invalid command-line arguments
// - Missing configuration files
// - Permission denied on file operations
// - Resource exhaustion (memory, file handles)
```

#### Library/SDK Errors

```rust
// Component: src/lib.rs, src/thinktool/*.rs
// Error types: Processing errors, validation failures
// Severity: Error (operation failed), Warning (degraded)

// Examples:
// - ThinkTool execution failures
// - Protocol validation errors
// - LLM API failures
// - Embedding generation errors
```

#### API Errors (LLM Provider Integration)

```rust
// Component: src/thinktool/llm.rs
// Error types: Network, rate limiting, authentication
// Severity: Error (recoverable), Warning (degraded)

// Examples:
// - Rate limit exceeded (429)
// - Authentication failure (401/403)
// - Timeout (504)
// - Invalid response format
```

#### Integration Errors

```rust
// Component: src/mcp/*.rs, src/bindings.rs
// Error types: Protocol errors, serialization failures
// Severity: Error (integration broken), Warning (partial failure)

// Examples:
// - MCP server connection failures
// - JSON-RPC parsing errors
// - Python binding exceptions
// - External tool execution failures
```

#### User-Facing Errors

```rust
// Component: All user-visible output paths
// Error types: Formatted error messages for humans
// Severity: All levels with user-friendly messages

// Examples:
// - "Configuration file not found: ~/.config/reasonkit/config.toml"
// - "LLM API rate limit exceeded. Retry in 60 seconds."
// - "Invalid query format. See 'rk-core help query' for usage."
```

### 2.2 Error Classification

| Level | Code | Description | Action Required | Example |
|-------|------|-------------|-----------------|---------|
| **Fatal** | F001-F999 | Application crash, unrecoverable | Immediate restart, incident | Panic, OOM |
| **Error** | E001-E999 | Operation failed, recoverable | Retry or user action | API timeout |
| **Warning** | W001-W999 | Degraded functionality | Monitoring, no action | Cache miss |
| **Info** | I001-I999 | Informational events | Logging only | Config loaded |

### 2.3 Error Code Registry

```rust
// src/error_codes.rs (proposed)

/// ReasonKit Error Code Registry
pub mod codes {
    // Fatal Errors (F001-F099: Core)
    pub const F001_PANIC: &str = "F001";           // Unhandled panic
    pub const F002_OOM: &str = "F002";             // Out of memory
    pub const F003_CONFIG_CORRUPT: &str = "F003"; // Corrupted configuration

    // Error: Configuration (E100-E199)
    pub const E100_CONFIG_NOT_FOUND: &str = "E100";
    pub const E101_CONFIG_PARSE: &str = "E101";
    pub const E102_CONFIG_INVALID: &str = "E102";

    // Error: Network/API (E200-E299)
    pub const E200_NETWORK: &str = "E200";
    pub const E201_TIMEOUT: &str = "E201";
    pub const E202_RATE_LIMIT: &str = "E202";
    pub const E203_AUTH: &str = "E203";
    pub const E204_API_ERROR: &str = "E204";

    // Error: Processing (E300-E399)
    pub const E300_PARSE: &str = "E300";
    pub const E301_VALIDATION: &str = "E301";
    pub const E302_EMBEDDING: &str = "E302";
    pub const E303_THINKTOOL: &str = "E303";

    // Error: Storage (E400-E499)
    pub const E400_IO: &str = "E400";
    pub const E401_PERMISSION: &str = "E401";
    pub const E402_NOT_FOUND: &str = "E402";
    pub const E403_STORAGE: &str = "E403";

    // Error: Integration (E500-E599)
    pub const E500_MCP: &str = "E500";
    pub const E501_PYTHON: &str = "E501";
    pub const E502_EXTERNAL_TOOL: &str = "E502";

    // Warning: Performance (W100-W199)
    pub const W100_SLOW_QUERY: &str = "W100";
    pub const W101_CACHE_MISS: &str = "W101";
    pub const W102_FALLBACK: &str = "W102";

    // Warning: Deprecation (W200-W299)
    pub const W200_DEPRECATED_API: &str = "W200";
    pub const W201_DEPRECATED_CONFIG: &str = "W201";
}
```

---

## 3. Rust Error Handling

### 3.1 Existing Error Types

ReasonKit uses `thiserror` for error definitions. The current error hierarchy in `src/error.rs`:

```rust
use thiserror::Error;

/// Result type alias for ReasonKit operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for ReasonKit Core
#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("PDF processing error: {0}")]
    Pdf(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Retrieval error: {0}")]
    Retrieval(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Schema validation error: {0}")]
    Validation(String),

    #[error("{context}: {source}")]
    WithContext {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
```

### 3.2 Enhanced Error Types (Recommended)

```rust
// src/error_enhanced.rs

use thiserror::Error;
use std::backtrace::Backtrace;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Enhanced error with full context for production monitoring
#[derive(Error, Debug)]
pub struct TrackedError {
    /// Unique error ID for tracking
    pub id: Uuid,

    /// Error code from registry
    pub code: &'static str,

    /// Human-readable message
    pub message: String,

    /// Error severity level
    pub severity: ErrorSeverity,

    /// When the error occurred
    pub timestamp: DateTime<Utc>,

    /// Stack trace (if available)
    pub backtrace: Option<Backtrace>,

    /// Additional context for debugging
    pub context: ErrorContext,

    /// The underlying error
    #[source]
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl std::fmt::Display for TrackedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.code, self.severity, self.message)
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Fatal,
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fatal => write!(f, "FATAL"),
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARN"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// Additional error context
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    /// ReasonKit version
    pub version: Option<String>,

    /// Operating system
    pub os: Option<String>,

    /// Rust version
    pub rust_version: Option<String>,

    /// ThinkTool being used
    pub thinktool: Option<String>,

    /// Reasoning profile
    pub profile: Option<String>,

    /// Session ID
    pub session_id: Option<Uuid>,

    /// Request/trace ID
    pub trace_id: Option<Uuid>,

    /// Custom key-value metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create context with system information
    pub fn with_system_info() -> Self {
        Self {
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            os: Some(std::env::consts::OS.to_string()),
            rust_version: Some(rustc_version_runtime::version().to_string()),
            ..Default::default()
        }
    }

    /// Add ThinkTool context
    pub fn with_thinktool(mut self, name: &str) -> Self {
        self.thinktool = Some(name.to_string());
        self
    }

    /// Add profile context
    pub fn with_profile(mut self, profile: &str) -> Self {
        self.profile = Some(profile.to_string());
        self
    }

    /// Add session context
    pub fn with_session(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

impl TrackedError {
    /// Create a new tracked error
    pub fn new(code: &'static str, message: impl Into<String>, severity: ErrorSeverity) -> Self {
        Self {
            id: Uuid::new_v4(),
            code,
            message: message.into(),
            severity,
            timestamp: Utc::now(),
            backtrace: Some(Backtrace::capture()),
            context: ErrorContext::with_system_info(),
            source: None,
        }
    }

    /// Create from an existing error
    pub fn from_error<E: std::error::Error + Send + Sync + 'static>(
        code: &'static str,
        severity: ErrorSeverity,
        error: E,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            code,
            message: error.to_string(),
            severity,
            timestamp: Utc::now(),
            backtrace: Some(Backtrace::capture()),
            context: ErrorContext::with_system_info(),
            source: Some(Box::new(error)),
        }
    }

    /// Add context to the error
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = context;
        self
    }

    /// Check if error is fatal
    pub fn is_fatal(&self) -> bool {
        matches!(self.severity, ErrorSeverity::Fatal)
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        !self.is_fatal()
    }

    /// Serialize for external reporting (PII-safe)
    pub fn to_safe_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id.to_string(),
            "code": self.code,
            "message": self.message,
            "severity": format!("{}", self.severity),
            "timestamp": self.timestamp.to_rfc3339(),
            "context": {
                "version": self.context.version,
                "os": self.context.os,
                "thinktool": self.context.thinktool,
                "profile": self.context.profile,
            }
        })
    }
}

/// Extension trait for adding tracking to Results
pub trait TrackableResult<T> {
    fn track(self, code: &'static str, severity: ErrorSeverity) -> Result<T, TrackedError>;
    fn track_with_context(
        self,
        code: &'static str,
        severity: ErrorSeverity,
        context: ErrorContext,
    ) -> Result<T, TrackedError>;
}

impl<T, E: std::error::Error + Send + Sync + 'static> TrackableResult<T> for Result<T, E> {
    fn track(self, code: &'static str, severity: ErrorSeverity) -> Result<T, TrackedError> {
        self.map_err(|e| TrackedError::from_error(code, severity, e))
    }

    fn track_with_context(
        self,
        code: &'static str,
        severity: ErrorSeverity,
        context: ErrorContext,
    ) -> Result<T, TrackedError> {
        self.map_err(|e| TrackedError::from_error(code, severity, e).with_context(context))
    }
}
```

### 3.3 Integration with Error Reporting Services

#### Sentry Integration for Rust

Add to `Cargo.toml`:

```toml
[dependencies]
sentry = { version = "0.34", default-features = false, features = [
    "backtrace",
    "contexts",
    "panic",
    "reqwest",
    "rustls",
], optional = true }
sentry-tracing = { version = "0.34", optional = true }

[features]
error-tracking = ["sentry", "sentry-tracing"]
```

Implementation:

```rust
// src/error_tracking/sentry.rs

use sentry::{ClientOptions, IntoDsn};
use std::sync::OnceLock;

static SENTRY_GUARD: OnceLock<sentry::ClientInitGuard> = OnceLock::new();

/// Initialize Sentry error tracking
pub fn init_sentry(dsn: Option<&str>, environment: &str) -> Result<(), String> {
    let dsn = match dsn {
        Some(d) => d.into_dsn().map_err(|e| e.to_string())?,
        None => {
            // Check environment variable
            std::env::var("SENTRY_DSN")
                .ok()
                .and_then(|s| s.into_dsn().ok().flatten())
        }
    };

    if dsn.is_none() {
        tracing::info!("Sentry DSN not configured, error tracking disabled");
        return Ok(());
    }

    let guard = sentry::init(ClientOptions {
        dsn,
        environment: Some(environment.into()),
        release: Some(format!("reasonkit@{}", env!("CARGO_PKG_VERSION")).into()),
        sample_rate: 1.0,
        traces_sample_rate: 0.1, // 10% of transactions
        attach_stacktrace: true,
        send_default_pii: false, // CRITICAL: Never send PII
        before_send: Some(std::sync::Arc::new(|mut event| {
            // Strip any PII that might have leaked through
            if let Some(ref mut extra) = event.extra {
                extra.retain(|k, _| !is_pii_key(k));
            }
            Some(event)
        })),
        ..Default::default()
    });

    SENTRY_GUARD.set(guard).map_err(|_| "Sentry already initialized")?;

    tracing::info!("Sentry error tracking initialized for environment: {}", environment);
    Ok(())
}

/// Check if a key might contain PII
fn is_pii_key(key: &str) -> bool {
    let key_lower = key.to_lowercase();
    [
        "email", "password", "token", "secret", "key", "auth",
        "ssn", "phone", "address", "name", "user", "ip",
    ]
    .iter()
    .any(|&pii| key_lower.contains(pii))
}

/// Capture a TrackedError to Sentry
pub fn capture_error(error: &TrackedError) {
    use sentry::protocol::{Event, Level};

    let level = match error.severity {
        ErrorSeverity::Fatal => Level::Fatal,
        ErrorSeverity::Error => Level::Error,
        ErrorSeverity::Warning => Level::Warning,
        ErrorSeverity::Info => Level::Info,
    };

    sentry::with_scope(
        |scope| {
            scope.set_tag("error.code", error.code);
            scope.set_tag("error.severity", format!("{}", error.severity));

            if let Some(ref tool) = error.context.thinktool {
                scope.set_tag("thinktool", tool);
            }
            if let Some(ref profile) = error.context.profile {
                scope.set_tag("profile", profile);
            }
            if let Some(ref version) = error.context.version {
                scope.set_tag("reasonkit.version", version);
            }

            // Add safe context as extra data
            scope.set_extra("error_id", error.id.to_string().into());
            scope.set_extra("timestamp", error.timestamp.to_rfc3339().into());
        },
        || {
            sentry::capture_message(&error.message, level);
        },
    );
}

/// Capture a panic to Sentry
pub fn setup_panic_handler() {
    let default_hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |panic_info| {
        // Capture to Sentry
        sentry::capture_message(
            &format!("Panic: {}", panic_info),
            sentry::Level::Fatal,
        );

        // Flush Sentry before potentially crashing
        sentry::flush(Some(std::time::Duration::from_secs(2)));

        // Call the default handler
        default_hook(panic_info);
    }));
}
```

#### OpenTelemetry Integration

```rust
// src/error_tracking/otel.rs

use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use tracing_subscriber::layer::SubscriberExt;

/// Initialize OpenTelemetry tracing
pub fn init_otel(service_name: &str, endpoint: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(endpoint),
        )
        .with_trace_config(
            opentelemetry_sdk::trace::config()
                .with_resource(opentelemetry_sdk::Resource::new(vec![
                    KeyValue::new("service.name", service_name.to_string()),
                    KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                ])),
        )
        .install_batch(opentelemetry_sdk::runtime::Tokio)?;

    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    let subscriber = tracing_subscriber::registry()
        .with(telemetry)
        .with(tracing_subscriber::fmt::layer());

    tracing::subscriber::set_global_default(subscriber)?;

    Ok(())
}

/// Record an error as a span event
pub fn record_error_event(error: &TrackedError) {
    use tracing::Span;

    let span = Span::current();
    span.record("error", true);
    span.record("error.code", error.code);
    span.record("error.message", &error.message);
    span.record("error.severity", format!("{}", error.severity).as_str());

    tracing::event!(
        tracing::Level::ERROR,
        error.code = error.code,
        error.message = %error.message,
        error.id = %error.id,
        "Error occurred"
    );
}
```

---

## 4. Python Bindings Error Handling

### 4.1 Exception Hierarchy

```python
# reasonkit/exceptions.py

"""
ReasonKit Python Exception Hierarchy

Exception Hierarchy:
    ReasonKitError (base)
    ├── ConfigurationError
    │   ├── ConfigNotFoundError
    │   └── ConfigParseError
    ├── NetworkError
    │   ├── TimeoutError
    │   ├── RateLimitError
    │   └── AuthenticationError
    ├── ProcessingError
    │   ├── ValidationError
    │   ├── EmbeddingError
    │   └── ThinkToolError
    ├── StorageError
    │   ├── IOError
    │   ├── PermissionError
    │   └── NotFoundError
    └── IntegrationError
        ├── MCPError
        └── ExternalToolError
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import traceback


@dataclass
class ErrorContext:
    """Context information for error tracking."""
    version: Optional[str] = None
    os: Optional[str] = None
    python_version: Optional[str] = None
    thinktool: Optional[str] = None
    profile: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def with_system_info(cls) -> "ErrorContext":
        import platform
        import sys
        try:
            from reasonkit import __version__
            version = __version__
        except ImportError:
            version = "unknown"

        return cls(
            version=version,
            os=platform.system(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )


class ReasonKitError(Exception):
    """
    Base exception for all ReasonKit errors.

    Attributes:
        code: Error code from the registry (e.g., "E100")
        message: Human-readable error message
        severity: Error severity level
        error_id: Unique identifier for this error instance
        timestamp: When the error occurred
        context: Additional context for debugging
        cause: The underlying exception, if any
    """

    def __init__(
        self,
        message: str,
        code: str = "E000",
        severity: str = "error",
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.severity = severity
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.context = context or ErrorContext.with_system_info()
        self.cause = cause
        self._traceback = traceback.format_exc()

    def __str__(self) -> str:
        return f"[{self.code}] {self.severity.upper()}: {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_id": self.error_id,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "version": self.context.version,
                "os": self.context.os,
                "python_version": self.context.python_version,
                "thinktool": self.context.thinktool,
                "profile": self.context.profile,
            },
            "cause": str(self.cause) if self.cause else None,
        }

    def with_context(self, **kwargs) -> "ReasonKitError":
        """Add context to the error."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.metadata[key] = str(value)
        return self


# Configuration Errors (E100-E199)

class ConfigurationError(ReasonKitError):
    """Base class for configuration-related errors."""

    def __init__(self, message: str, code: str = "E100", **kwargs):
        super().__init__(message, code=code, **kwargs)


class ConfigNotFoundError(ConfigurationError):
    """Configuration file not found."""

    def __init__(self, path: str, **kwargs):
        super().__init__(
            f"Configuration file not found: {path}",
            code="E100",
            **kwargs,
        )
        self.path = path


class ConfigParseError(ConfigurationError):
    """Configuration file could not be parsed."""

    def __init__(self, path: str, detail: str, **kwargs):
        super().__init__(
            f"Failed to parse configuration at {path}: {detail}",
            code="E101",
            **kwargs,
        )
        self.path = path
        self.detail = detail


# Network Errors (E200-E299)

class NetworkError(ReasonKitError):
    """Base class for network-related errors."""

    def __init__(self, message: str, code: str = "E200", **kwargs):
        super().__init__(message, code=code, **kwargs)


class TimeoutError(NetworkError):
    """Request timed out."""

    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds}s",
            code="E201",
            **kwargs,
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class RateLimitError(NetworkError):
    """API rate limit exceeded."""

    def __init__(self, provider: str, retry_after: Optional[int] = None, **kwargs):
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, code="E202", **kwargs)
        self.provider = provider
        self.retry_after = retry_after


class AuthenticationError(NetworkError):
    """Authentication failed."""

    def __init__(self, provider: str, detail: str = "", **kwargs):
        super().__init__(
            f"Authentication failed for {provider}: {detail}",
            code="E203",
            **kwargs,
        )
        self.provider = provider


# Processing Errors (E300-E399)

class ProcessingError(ReasonKitError):
    """Base class for processing-related errors."""

    def __init__(self, message: str, code: str = "E300", **kwargs):
        super().__init__(message, code=code, **kwargs)


class ValidationError(ProcessingError):
    """Input validation failed."""

    def __init__(self, field: str, reason: str, **kwargs):
        super().__init__(
            f"Validation failed for '{field}': {reason}",
            code="E301",
            **kwargs,
        )
        self.field = field
        self.reason = reason


class EmbeddingError(ProcessingError):
    """Embedding generation failed."""

    def __init__(self, model: str, detail: str, **kwargs):
        super().__init__(
            f"Embedding generation failed with {model}: {detail}",
            code="E302",
            **kwargs,
        )
        self.model = model


class ThinkToolError(ProcessingError):
    """ThinkTool execution failed."""

    def __init__(self, tool_name: str, step: Optional[str] = None, detail: str = "", **kwargs):
        message = f"ThinkTool '{tool_name}' failed"
        if step:
            message += f" at step '{step}'"
        if detail:
            message += f": {detail}"
        super().__init__(message, code="E303", **kwargs)
        self.tool_name = tool_name
        self.step = step


# Storage Errors (E400-E499)

class StorageError(ReasonKitError):
    """Base class for storage-related errors."""

    def __init__(self, message: str, code: str = "E400", **kwargs):
        super().__init__(message, code=code, **kwargs)


class NotFoundError(StorageError):
    """Resource not found."""

    def __init__(self, resource_type: str, identifier: str, **kwargs):
        super().__init__(
            f"{resource_type} not found: {identifier}",
            code="E402",
            **kwargs,
        )
        self.resource_type = resource_type
        self.identifier = identifier


# Integration Errors (E500-E599)

class IntegrationError(ReasonKitError):
    """Base class for integration-related errors."""

    def __init__(self, message: str, code: str = "E500", **kwargs):
        super().__init__(message, code=code, **kwargs)


class MCPError(IntegrationError):
    """MCP server communication error."""

    def __init__(self, server: str, detail: str, **kwargs):
        super().__init__(
            f"MCP server '{server}' error: {detail}",
            code="E500",
            **kwargs,
        )
        self.server = server


# Error Code Mapping from Rust

RUST_ERROR_MAP = {
    "E100": ConfigNotFoundError,
    "E101": ConfigParseError,
    "E200": NetworkError,
    "E201": TimeoutError,
    "E202": RateLimitError,
    "E203": AuthenticationError,
    "E300": ProcessingError,
    "E301": ValidationError,
    "E302": EmbeddingError,
    "E303": ThinkToolError,
    "E400": StorageError,
    "E402": NotFoundError,
    "E500": MCPError,
}


def from_rust_error(code: str, message: str, **kwargs) -> ReasonKitError:
    """Create Python exception from Rust error code and message."""
    exception_class = RUST_ERROR_MAP.get(code, ReasonKitError)
    return exception_class(message, code=code, **kwargs)
```

### 4.2 PyO3 Binding Error Conversion

```rust
// src/bindings.rs (error handling section)

use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError, PyIOError};
use pyo3::create_exception;

// Create Python exception types
create_exception!(reasonkit, ReasonKitError, PyException);
create_exception!(reasonkit, ConfigurationError, ReasonKitError);
create_exception!(reasonkit, NetworkError, ReasonKitError);
create_exception!(reasonkit, ProcessingError, ReasonKitError);
create_exception!(reasonkit, StorageError, ReasonKitError);
create_exception!(reasonkit, IntegrationError, ReasonKitError);

/// Convert Rust Error to Python exception
impl From<crate::Error> for PyErr {
    fn from(error: crate::Error) -> PyErr {
        match &error {
            crate::Error::Config(_) => ConfigurationError::new_err(error.to_string()),
            crate::Error::Network(_) => NetworkError::new_err(error.to_string()),
            crate::Error::Validation(_) => ProcessingError::new_err(error.to_string()),
            crate::Error::Embedding(_) => ProcessingError::new_err(error.to_string()),
            crate::Error::Storage(_) => StorageError::new_err(error.to_string()),
            crate::Error::Io(_) => PyIOError::new_err(error.to_string()),
            crate::Error::Json(_) => PyValueError::new_err(error.to_string()),
            _ => ReasonKitError::new_err(error.to_string()),
        }
    }
}

/// Convert TrackedError to Python exception with full context
impl From<TrackedError> for PyErr {
    fn from(error: TrackedError) -> PyErr {
        let py_err = match error.code {
            c if c.starts_with("E1") => ConfigurationError::new_err(error.message.clone()),
            c if c.starts_with("E2") => NetworkError::new_err(error.message.clone()),
            c if c.starts_with("E3") => ProcessingError::new_err(error.message.clone()),
            c if c.starts_with("E4") => StorageError::new_err(error.message.clone()),
            c if c.starts_with("E5") => IntegrationError::new_err(error.message.clone()),
            _ => ReasonKitError::new_err(error.message.clone()),
        };

        // Attach context as cause (Python 3.11+)
        py_err
    }
}

/// Python module registration
#[pymodule]
fn reasonkit(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Register exception types
    m.add("ReasonKitError", py.get_type::<ReasonKitError>())?;
    m.add("ConfigurationError", py.get_type::<ConfigurationError>())?;
    m.add("NetworkError", py.get_type::<NetworkError>())?;
    m.add("ProcessingError", py.get_type::<ProcessingError>())?;
    m.add("StorageError", py.get_type::<StorageError>())?;
    m.add("IntegrationError", py.get_type::<IntegrationError>())?;

    Ok(())
}
```

### 4.3 Python Sentry Integration

```python
# reasonkit/error_tracking.py

"""
Error tracking integration for ReasonKit Python bindings.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger("reasonkit.errors")


class ErrorTracker:
    """
    Central error tracking manager.

    Supports:
    - Sentry (recommended)
    - Structured logging (default)
    - Custom handlers
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ErrorTracker._initialized:
            return

        self._sentry_enabled = False
        self._handlers = []
        ErrorTracker._initialized = True

    def init_sentry(
        self,
        dsn: Optional[str] = None,
        environment: str = "development",
        release: Optional[str] = None,
        sample_rate: float = 1.0,
        traces_sample_rate: float = 0.1,
    ) -> bool:
        """
        Initialize Sentry error tracking.

        Args:
            dsn: Sentry DSN (or use SENTRY_DSN env var)
            environment: Deployment environment
            release: Release version (defaults to reasonkit version)
            sample_rate: Error sample rate (0.0 to 1.0)
            traces_sample_rate: Transaction trace rate (0.0 to 1.0)

        Returns:
            True if Sentry was initialized successfully
        """
        dsn = dsn or os.environ.get("SENTRY_DSN")
        if not dsn:
            logger.info("Sentry DSN not configured, error tracking disabled")
            return False

        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration
        except ImportError:
            logger.warning("sentry-sdk not installed. Install with: pip install sentry-sdk")
            return False

        try:
            from reasonkit import __version__
            version = __version__
        except ImportError:
            version = "0.0.0"

        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release or f"reasonkit@{version}",
            sample_rate=sample_rate,
            traces_sample_rate=traces_sample_rate,
            send_default_pii=False,  # CRITICAL: Never send PII
            integrations=[
                LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR,
                ),
            ],
            before_send=self._before_send,
        )

        self._sentry_enabled = True
        logger.info(f"Sentry initialized for environment: {environment}")
        return True

    def _before_send(self, event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Pre-process event before sending to Sentry.
        Strips any potential PII that leaked through.
        """
        # Strip PII from extra data
        if "extra" in event:
            event["extra"] = {
                k: v for k, v in event["extra"].items()
                if not self._is_pii_key(k)
            }

        # Strip PII from breadcrumbs
        if "breadcrumbs" in event:
            for crumb in event.get("breadcrumbs", {}).get("values", []):
                if "data" in crumb:
                    crumb["data"] = {
                        k: v for k, v in crumb["data"].items()
                        if not self._is_pii_key(k)
                    }

        return event

    def _is_pii_key(self, key: str) -> bool:
        """Check if a key might contain PII."""
        pii_patterns = [
            "email", "password", "token", "secret", "key", "auth",
            "ssn", "phone", "address", "name", "user", "ip",
        ]
        key_lower = key.lower()
        return any(pii in key_lower for pii in pii_patterns)

    def capture_error(self, error: "ReasonKitError") -> Optional[str]:
        """
        Capture a ReasonKitError to configured tracking services.

        Args:
            error: The error to capture

        Returns:
            Event ID if captured, None otherwise
        """
        # Always log
        logger.error(
            f"[{error.code}] {error.message}",
            exc_info=error.cause,
            extra=error.to_dict(),
        )

        # Send to Sentry if enabled
        if self._sentry_enabled:
            import sentry_sdk

            with sentry_sdk.push_scope() as scope:
                scope.set_tag("error.code", error.code)
                scope.set_tag("error.severity", error.severity)

                if error.context.thinktool:
                    scope.set_tag("thinktool", error.context.thinktool)
                if error.context.profile:
                    scope.set_tag("profile", error.context.profile)

                scope.set_extra("error_id", error.error_id)
                scope.set_extra("timestamp", error.timestamp.isoformat())

                level = {
                    "fatal": "fatal",
                    "error": "error",
                    "warning": "warning",
                    "info": "info",
                }.get(error.severity, "error")

                return sentry_sdk.capture_message(error.message, level=level)

        # Call custom handlers
        for handler in self._handlers:
            try:
                handler(error)
            except Exception as e:
                logger.warning(f"Error handler failed: {e}")

        return None

    def add_handler(self, handler):
        """Add a custom error handler function."""
        self._handlers.append(handler)

    @contextmanager
    def capture_exceptions(self, reraise: bool = True):
        """
        Context manager to capture any ReasonKitError.

        Usage:
            with error_tracker.capture_exceptions():
                # Your code here
                ...
        """
        try:
            yield
        except ReasonKitError as e:
            self.capture_error(e)
            if reraise:
                raise
        except Exception as e:
            # Wrap unexpected exceptions
            wrapped = ReasonKitError(str(e), cause=e)
            self.capture_error(wrapped)
            if reraise:
                raise


# Global instance
error_tracker = ErrorTracker()


def init_error_tracking(
    dsn: Optional[str] = None,
    environment: str = "development",
) -> bool:
    """
    Convenience function to initialize error tracking.

    Args:
        dsn: Sentry DSN (or use SENTRY_DSN env var)
        environment: Deployment environment

    Returns:
        True if error tracking was initialized
    """
    return error_tracker.init_sentry(dsn=dsn, environment=environment)


def capture_error(error: "ReasonKitError") -> Optional[str]:
    """
    Capture an error to the configured tracking service.

    Args:
        error: The ReasonKitError to capture

    Returns:
        Event ID if captured, None otherwise
    """
    return error_tracker.capture_error(error)
```

---

## 5. Error Reporting Service Setup

### 5.1 Sentry Setup (Recommended)

#### Project Configuration

1. **Create Sentry Project**
   ```
   Organization: reasonkit
   Project: reasonkit-core (Rust)
   Project: reasonkit-python (Python bindings)
   ```

2. **Configure DSN Management**
   ```bash
   # Development (local .env)
   SENTRY_DSN=https://xxx@xxx.ingest.sentry.io/xxx
   SENTRY_ENVIRONMENT=development

   # Production (secrets manager)
   # Store in AWS Secrets Manager, Vault, or similar
   ```

3. **Environment Separation**
   ```yaml
   environments:
     - development  # Local development
     - staging      # CI/CD testing
     - production   # Live users
   ```

4. **Release Tracking**
   ```bash
   # In CI/CD pipeline
   export SENTRY_RELEASE="reasonkit@$(cargo pkgid | cut -d# -f2)"
   sentry-cli releases new "$SENTRY_RELEASE"
   sentry-cli releases set-commits "$SENTRY_RELEASE" --auto
   sentry-cli releases finalize "$SENTRY_RELEASE"
   ```

#### Sentry Dashboard Setup

```yaml
# .sentry/config.yml
project: reasonkit-core
org: reasonkit

# Alert rules
alerts:
  - name: "High Error Rate"
    conditions:
      - type: event_frequency
        value: 100
        interval: 1h
    actions:
      - type: email
        targetIdentifier: team@reasonkit.sh

  - name: "New Error Type"
    conditions:
      - type: first_seen_event
    actions:
      - type: slack
        workspace: reasonkit
        channel: "#errors"

# Issue grouping
fingerprinting:
  - match:
      type: "error"
      logger: "reasonkit.*"
    fingerprint:
      - "{{ default }}"
      - "{{ tags.error.code }}"
```

### 5.2 Self-Hosted Alternative: GlitchTip

For organizations requiring self-hosted solutions:

```yaml
# docker-compose.glitchtip.yml
version: '3.8'

services:
  glitchtip:
    image: glitchtip/glitchtip:latest
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgres://glitchtip:password@postgres:5432/glitchtip
      SECRET_KEY: ${GLITCHTIP_SECRET_KEY}
      PORT: 8000
      EMAIL_URL: smtp://mailhost:587
      DEFAULT_FROM_EMAIL: errors@reasonkit.sh
      GLITCHTIP_DOMAIN: https://errors.reasonkit.sh
      ENABLE_ORGANIZATION_CREATION: "false"
      CELERY_WORKER_CONCURRENCY: 2

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: glitchtip
      POSTGRES_USER: glitchtip
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - glitchtip-postgres:/var/lib/postgresql/data

  redis:
    image: redis:7

  worker:
    image: glitchtip/glitchtip:latest
    command: celery -A glitchtip worker -B -l info
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgres://glitchtip:password@postgres:5432/glitchtip
      SECRET_KEY: ${GLITCHTIP_SECRET_KEY}

volumes:
  glitchtip-postgres:
```

### 5.3 Comparison: Sentry vs Self-Hosted

| Feature | Sentry Cloud | GlitchTip | Highlight.io |
|---------|--------------|-----------|--------------|
| **Setup Complexity** | Low | Medium | Medium |
| **Cost** | $26+/month | Infrastructure only | $0-150/month |
| **Data Residency** | US/EU | Your servers | Your servers |
| **Features** | Full | 80% | 90% |
| **Retention** | 90 days | Unlimited | Configurable |
| **GDPR Compliance** | DPA available | Full control | DPA available |
| **Rust SDK** | Official | Compatible | Official |
| **Python SDK** | Official | Compatible | Official |

**Recommendation:** Use Sentry Cloud for teams <10 developers, self-hosted GlitchTip for enterprise/compliance requirements.

---

## 6. Error Context and Enrichment

### 6.1 Required Context

Every error report MUST include:

```rust
/// Minimum required error context
pub struct RequiredContext {
    // Application Info
    pub reasonkit_version: String,      // e.g., "0.1.0"
    pub build_timestamp: String,        // Build time
    pub git_commit: String,             // Short SHA

    // Environment Info
    pub os_family: String,              // e.g., "linux", "macos", "windows"
    pub os_version: String,             // e.g., "Ubuntu 22.04"
    pub rust_version: String,           // If Rust CLI
    pub python_version: Option<String>, // If Python bindings

    // ReasonKit Context
    pub thinktool: Option<String>,      // ThinkTool being used
    pub profile: Option<String>,        // Reasoning profile
    pub session_id: Uuid,               // Current session
    pub trace_id: Uuid,                 // Request trace

    // Error Classification
    pub error_code: String,             // Error registry code
    pub error_category: String,         // Category (config/network/etc)
    pub is_recoverable: bool,           // Can retry?
}
```

### 6.2 Optional Context (Sanitized)

```rust
/// Optional context (MUST be sanitized before inclusion)
pub struct OptionalContext {
    // Sanitized Input Context
    pub query_hash: String,             // SHA256 of query (never raw text)
    pub query_length: usize,            // Character count
    pub query_type: String,             // Classified type

    // Execution Context
    pub step_count: usize,              // ThinkTool steps executed
    pub duration_ms: u64,               // Execution time
    pub memory_usage_mb: Option<f64>,   // Memory at error time

    // Integration Context
    pub llm_provider: Option<String>,   // e.g., "anthropic"
    pub llm_model: Option<String>,      // e.g., "claude-sonnet-4"
    pub mcp_servers: Vec<String>,       // Active MCP servers

    // Custom Tags
    pub tags: HashMap<String, String>,  // Arbitrary tags
}
```

### 6.3 Context Collection Implementation

```rust
// src/error_tracking/context.rs

use sysinfo::{System, SystemExt};
use std::collections::HashMap;

/// Collect system context for error reporting
pub fn collect_system_context() -> HashMap<String, String> {
    let mut ctx = HashMap::new();

    // Static info (from build)
    ctx.insert("reasonkit.version".into(), env!("CARGO_PKG_VERSION").into());
    ctx.insert("reasonkit.build.timestamp".into(), env!("BUILD_TIMESTAMP", "unknown").into());

    // OS info
    ctx.insert("os.family".into(), std::env::consts::OS.into());
    ctx.insert("os.arch".into(), std::env::consts::ARCH.into());

    // Runtime info
    ctx.insert("rust.version".into(), rustc_version_runtime::version().to_string());

    // Optional: System info (may be slow)
    if let Some(system) = System::new_all().get_system() {
        ctx.insert("system.memory_mb".into(), format!("{}", system.total_memory() / 1024 / 1024));
        ctx.insert("system.cpu_count".into(), format!("{}", system.cpus().len()));
    }

    ctx
}

/// Collect ThinkTool execution context
pub fn collect_thinktool_context(
    tool_name: &str,
    profile: &str,
    step_count: usize,
    duration_ms: u64,
) -> HashMap<String, String> {
    let mut ctx = HashMap::new();

    ctx.insert("thinktool.name".into(), tool_name.into());
    ctx.insert("thinktool.profile".into(), profile.into());
    ctx.insert("thinktool.step_count".into(), format!("{}", step_count));
    ctx.insert("thinktool.duration_ms".into(), format!("{}", duration_ms));

    ctx
}
```

---

## 7. Privacy Considerations

### 7.1 PII Scrubbing

ReasonKit includes built-in PII scrubbing in the telemetry module (`src/telemetry/privacy.rs`):

```rust
// Already implemented patterns:
// - Email addresses
// - Phone numbers
// - SSN
// - Credit card numbers
// - IP addresses
// - API keys (various formats)
// - AWS access keys
// - URLs with credentials
// - File paths with usernames
```

**Additional patterns to add for error tracking:**

```rust
// src/error_tracking/pii.rs

use regex::Regex;
use once_cell::sync::Lazy;

/// Additional PII patterns for error context
static ADDITIONAL_PII_PATTERNS: Lazy<Vec<(&str, Regex, &str)>> = Lazy::new(|| vec![
    // JWT tokens
    ("jwt", Regex::new(r"eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+").unwrap(), "[JWT]"),

    // Bearer tokens
    ("bearer", Regex::new(r"(?i)bearer\s+[a-zA-Z0-9\-._~+/]+=*").unwrap(), "Bearer [TOKEN]"),

    // GitHub tokens
    ("github_token", Regex::new(r"gh[pousr]_[A-Za-z0-9_]{36,}").unwrap(), "[GITHUB_TOKEN]"),

    // OpenAI API keys
    ("openai_key", Regex::new(r"sk-[A-Za-z0-9]{48}").unwrap(), "[OPENAI_KEY]"),

    // Anthropic API keys
    ("anthropic_key", Regex::new(r"sk-ant-[A-Za-z0-9\-]{95}").unwrap(), "[ANTHROPIC_KEY]"),

    // Private keys (PEM format)
    ("private_key", Regex::new(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----").unwrap(), "[PRIVATE_KEY]"),

    // Connection strings
    ("connection_string", Regex::new(r"(?i)(?:mongodb|postgres|mysql|redis)://[^\s]+").unwrap(), "[CONNECTION_STRING]"),

    // Environment variable values that look sensitive
    ("env_secret", Regex::new(r#"(?i)(?:password|secret|token|key)\s*[=:]\s*['"]?[^\s'"]{8,}['"]?"#).unwrap(), "[ENV_SECRET]"),
]);

/// Scrub all PII from a string
pub fn scrub_pii(text: &str) -> String {
    let mut result = text.to_string();

    for (_, pattern, replacement) in ADDITIONAL_PII_PATTERNS.iter() {
        result = pattern.replace_all(&result, *replacement).to_string();
    }

    result
}

/// Scrub PII from error context HashMap
pub fn scrub_context(context: &mut HashMap<String, String>) {
    for value in context.values_mut() {
        *value = scrub_pii(value);
    }
}
```

### 7.2 Data Retention

```yaml
# Error data retention policy
retention:
  # Local SQLite telemetry
  local_telemetry:
    default: 90 days
    configurable: true
    min: 7 days
    max: 365 days

  # Sentry (if enabled)
  sentry:
    events: 90 days  # Sentry default
    transactions: 90 days

  # Self-hosted
  glitchtip:
    events: configurable
    recommendation: 180 days
```

### 7.3 User Consent

```rust
// src/consent.rs

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// User consent record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrackingConsent {
    /// Consent version
    pub version: u32,

    /// When consent was given
    pub timestamp: DateTime<Utc>,

    /// Allow local error logging
    pub local_logging: bool,

    /// Allow sending errors to external service
    pub external_reporting: bool,

    /// Allow sending stack traces
    pub include_stacktrace: bool,

    /// Allow sending execution context
    pub include_context: bool,
}

impl Default for ErrorTrackingConsent {
    fn default() -> Self {
        Self {
            version: 1,
            timestamp: Utc::now(),
            local_logging: true,          // Default: Yes
            external_reporting: false,    // Default: Opt-in
            include_stacktrace: false,    // Default: Opt-in
            include_context: true,        // Default: Yes (sanitized)
        }
    }
}

/// Check consent before sending
pub fn check_consent_for_external(consent: &ErrorTrackingConsent) -> bool {
    consent.external_reporting
}
```

### 7.4 Opt-Out Mechanism

```bash
# Environment variable opt-out
export RK_ERROR_TRACKING=false

# CLI flag
rk-core --no-error-tracking think "query"

# Configuration file (~/.config/reasonkit/config.toml)
[error_tracking]
enabled = false
external_reporting = false
```

```rust
// Check before any error reporting
fn should_report_error() -> bool {
    // Environment variable takes precedence
    if let Ok(val) = std::env::var("RK_ERROR_TRACKING") {
        if val.to_lowercase() == "false" || val == "0" {
            return false;
        }
    }

    // Check config
    // ...

    true
}
```

---

## 8. Alerting and Triage

### 8.1 Alert Rules

#### Critical Alerts (Immediate)

```yaml
# PagerDuty / Opsgenie integration
critical_alerts:
  - name: "Fatal Crash Rate Spike"
    condition: "count(severity=fatal) > 10 in 5m"
    action: page_oncall

  - name: "New Fatal Error Type"
    condition: "first_seen AND severity=fatal"
    action: page_oncall

  - name: "100% Error Rate"
    condition: "error_rate = 100% for 5m"
    action: page_oncall
```

#### Warning Alerts (Business Hours)

```yaml
warning_alerts:
  - name: "Error Rate Above Threshold"
    condition: "error_rate > 5% for 15m"
    action: slack_channel

  - name: "New Error Type"
    condition: "first_seen AND severity=error"
    action: slack_channel

  - name: "Regression Detected"
    condition: "previously_resolved AND reoccurred"
    action: slack_channel
```

#### Informational Alerts (Daily Digest)

```yaml
info_alerts:
  - name: "Daily Error Summary"
    schedule: "0 9 * * *"  # 9 AM daily
    action: email_digest

  - name: "Weekly Trend Report"
    schedule: "0 9 * * 1"  # Monday 9 AM
    action: email_report
```

### 8.2 Severity Classification

```python
# alerting/classifier.py

def classify_severity(error: dict) -> str:
    """
    Classify error severity for alerting purposes.
    """
    code = error.get("code", "")
    message = error.get("message", "").lower()

    # Fatal indicators
    if code.startswith("F"):
        return "critical"
    if "panic" in message or "crash" in message:
        return "critical"
    if "out of memory" in message or "oom" in message:
        return "critical"

    # High priority
    if code.startswith("E2"):  # Network errors
        if "authentication" in message or "401" in message:
            return "high"
    if "data loss" in message or "corruption" in message:
        return "high"

    # Medium priority
    if code.startswith("E"):
        return "medium"

    # Low priority
    if code.startswith("W"):
        return "low"

    return "medium"
```

### 8.3 Triage Process

```mermaid
graph TD
    A[Error Received] --> B{Severity?}
    B -->|Critical| C[Page On-Call]
    B -->|High| D[Slack Alert]
    B -->|Medium| E[Ticket Created]
    B -->|Low| F[Log Only]

    C --> G[Acknowledge]
    G --> H{Root Cause Known?}
    H -->|Yes| I[Apply Fix]
    H -->|No| J[Investigate]
    J --> K[Document Finding]
    K --> I
    I --> L[Verify Fix]
    L --> M[Close Incident]
```

### 8.4 Assignment Rules

```yaml
# Assignment based on error category
assignment_rules:
  - match: "code starts with E1"  # Config errors
    assign: "platform-team"

  - match: "code starts with E2"  # Network errors
    assign: "infrastructure-team"

  - match: "code starts with E3"  # Processing errors
    assign: "core-team"

  - match: "thinktool contains PowerCombo"
    assign: "reasoning-team"

  - match: "default"
    assign: "triage-rotation"
```

### 8.5 Communication Templates

#### Incident Alert

```markdown
## [SEVERITY] Error in ReasonKit: [CODE]

**Error ID:** [error_id]
**First Seen:** [timestamp]
**Affected:** [affected_count] users

### Summary
[error_message]

### Context
- ThinkTool: [thinktool]
- Profile: [profile]
- Environment: [environment]

### Quick Links
- [View in Sentry](sentry_link)
- [Runbook](runbook_link)
- [Recent Deploys](deploy_link)

### Actions Required
1. Acknowledge this alert
2. Check runbook for known fixes
3. Escalate if not resolved in 30m
```

#### Resolution Notification

```markdown
## Resolved: [CODE] Error

**Resolution Time:** [duration]
**Root Cause:** [root_cause]
**Fix Applied:** [fix_description]

### Impact
- Affected users: [count]
- Duration: [duration]
- Data loss: [yes/no]

### Prevention
[prevention_steps]
```

---

## 9. Dashboard and Reporting

### 9.1 Key Metrics

```yaml
# Dashboard metrics
metrics:
  # Error Rate
  error_rate:
    formula: "errors / total_requests * 100"
    unit: "%"
    target: "< 1%"
    alert_threshold: "5%"

  # Error Volume
  error_count:
    formula: "count(errors)"
    unit: "count"
    group_by: ["code", "severity", "thinktool"]

  # Affected Users
  affected_users:
    formula: "unique(session_id) where error"
    unit: "users"

  # Mean Time to Detection
  mttd:
    formula: "avg(alert_time - error_time)"
    unit: "minutes"
    target: "< 5 min"

  # Mean Time to Resolution
  mttr:
    formula: "avg(resolved_time - alert_time)"
    unit: "minutes"
    target: "< 60 min"

  # Error Trend
  error_trend:
    formula: "error_count this_week / error_count last_week"
    unit: "ratio"
    target: "< 1.0"
```

### 9.2 Sentry Dashboard Configuration

```json
{
  "dashboard": {
    "name": "ReasonKit Error Monitoring",
    "widgets": [
      {
        "type": "line-chart",
        "title": "Error Rate Over Time",
        "queries": [
          {
            "name": "Error Rate",
            "fields": ["count()"],
            "conditions": "event.type:error",
            "groupBy": ["hour"]
          }
        ]
      },
      {
        "type": "table",
        "title": "Top Errors by Volume",
        "queries": [
          {
            "name": "Error Volume",
            "fields": ["count()", "tags[error.code]"],
            "conditions": "event.type:error",
            "groupBy": ["tags[error.code]"],
            "orderBy": "-count()",
            "limit": 10
          }
        ]
      },
      {
        "type": "pie-chart",
        "title": "Errors by ThinkTool",
        "queries": [
          {
            "name": "ThinkTool Distribution",
            "fields": ["count()"],
            "conditions": "event.type:error",
            "groupBy": ["tags[thinktool]"]
          }
        ]
      },
      {
        "type": "big-number",
        "title": "Affected Users (24h)",
        "queries": [
          {
            "name": "Unique Users",
            "fields": ["count_unique(user)"],
            "conditions": "event.type:error"
          }
        ]
      }
    ]
  }
}
```

### 9.3 Grafana Dashboard (Custom)

```json
{
  "dashboard": {
    "title": "ReasonKit Error Tracking",
    "panels": [
      {
        "title": "Error Rate by Severity",
        "type": "timeseries",
        "datasource": "prometheus",
        "targets": [
          {
            "expr": "sum(rate(reasonkit_errors_total[5m])) by (severity)",
            "legendFormat": "{{severity}}"
          }
        ]
      },
      {
        "title": "Error Heatmap by Code",
        "type": "heatmap",
        "datasource": "prometheus",
        "targets": [
          {
            "expr": "sum(increase(reasonkit_errors_total[1h])) by (code)",
            "legendFormat": "{{code}}"
          }
        ]
      },
      {
        "title": "MTTR Trend",
        "type": "gauge",
        "datasource": "prometheus",
        "targets": [
          {
            "expr": "avg(reasonkit_error_mttr_seconds) / 60"
          }
        ],
        "thresholds": [
          { "value": 30, "color": "green" },
          { "value": 60, "color": "yellow" },
          { "value": 120, "color": "red" }
        ]
      }
    ]
  }
}
```

### 9.4 Weekly Report Template

```markdown
# ReasonKit Error Report: Week of [DATE]

## Executive Summary
- **Total Errors:** [count] ([trend]% vs last week)
- **Error Rate:** [rate]%
- **Affected Users:** [count]
- **Critical Issues:** [count]

## Top Issues This Week

| Rank | Error Code | Count | Trend | Status |
|------|------------|-------|-------|--------|
| 1 | E303 | 1,234 | +15% | Investigating |
| 2 | E201 | 567 | -20% | Fixed |
| 3 | E100 | 234 | New | Triaging |

## Resolution Metrics
- **MTTD:** [X] minutes (Target: <5 min)
- **MTTR:** [X] minutes (Target: <60 min)
- **First Response:** [X] minutes

## Notable Incidents
1. **[INC-001]** - [Brief description] - Resolved in [X] min
2. **[INC-002]** - [Brief description] - Ongoing

## Action Items
- [ ] Investigate E303 spike
- [ ] Deploy fix for E201
- [ ] Update runbook for E100

## Trends
[Include trend chart image]
```

---

## 10. User Error Reporting

### 10.1 In-CLI Automatic Crash Reports

```rust
// src/crash_reporter.rs

use std::panic;
use std::fs::File;
use std::io::Write;

/// Install crash reporter
pub fn install_crash_reporter() {
    let default_hook = panic::take_hook();

    panic::set_hook(Box::new(move |panic_info| {
        // Generate crash report
        let report = generate_crash_report(panic_info);

        // Save locally
        if let Ok(path) = save_crash_report(&report) {
            eprintln!("\n========================================");
            eprintln!("ReasonKit crashed unexpectedly.");
            eprintln!("A crash report has been saved to:");
            eprintln!("  {}", path.display());
            eprintln!("");
            eprintln!("To help us fix this issue, please:");
            eprintln!("1. Run: rk-core report-crash {}", path.display());
            eprintln!("   (This will strip personal data before sending)");
            eprintln!("");
            eprintln!("Or manually file an issue at:");
            eprintln!("  https://github.com/reasonkit/reasonkit-core/issues");
            eprintln!("========================================\n");
        }

        // Call default handler
        default_hook(panic_info);
    }));
}

fn generate_crash_report(panic_info: &panic::PanicInfo) -> String {
    let mut report = String::new();

    // Header
    report.push_str("# ReasonKit Crash Report\n\n");
    report.push_str(&format!("Generated: {}\n", chrono::Utc::now().to_rfc3339()));
    report.push_str(&format!("Version: {}\n", env!("CARGO_PKG_VERSION")));
    report.push_str(&format!("OS: {} {}\n", std::env::consts::OS, std::env::consts::ARCH));

    // Panic info
    report.push_str("\n## Panic Information\n\n");
    if let Some(location) = panic_info.location() {
        report.push_str(&format!("Location: {}:{}:{}\n",
            location.file(), location.line(), location.column()));
    }
    if let Some(msg) = panic_info.payload().downcast_ref::<&str>() {
        report.push_str(&format!("Message: {}\n", msg));
    }

    // Backtrace
    report.push_str("\n## Backtrace\n\n```\n");
    report.push_str(&format!("{:?}", std::backtrace::Backtrace::capture()));
    report.push_str("\n```\n");

    report
}

fn save_crash_report(report: &str) -> std::io::Result<std::path::PathBuf> {
    let crash_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("reasonkit")
        .join("crashes");

    std::fs::create_dir_all(&crash_dir)?;

    let filename = format!("crash_{}.md", chrono::Utc::now().format("%Y%m%d_%H%M%S"));
    let path = crash_dir.join(filename);

    let mut file = File::create(&path)?;
    file.write_all(report.as_bytes())?;

    Ok(path)
}
```

### 10.2 Manual Bug Report Command

```rust
// src/commands/report.rs

use clap::Args;

#[derive(Args)]
pub struct ReportCrashArgs {
    /// Path to crash report file
    crash_file: std::path::PathBuf,

    /// Dry run - show what would be sent without sending
    #[arg(long)]
    dry_run: bool,

    /// Include additional context
    #[arg(long)]
    include_config: bool,
}

pub async fn report_crash(args: ReportCrashArgs) -> anyhow::Result<()> {
    // Read crash report
    let report = std::fs::read_to_string(&args.crash_file)?;

    // Sanitize PII
    let sanitized = sanitize_crash_report(&report);

    if args.dry_run {
        println!("=== Sanitized Report (Dry Run) ===\n");
        println!("{}", sanitized);
        println!("\n=== End Report ===");
        return Ok(());
    }

    // Ask for confirmation
    println!("The following information will be sent to ReasonKit:\n");
    println!("{}\n", summarize_report(&sanitized));

    print!("Send this report? [y/N] ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim().to_lowercase() != "y" {
        println!("Report cancelled.");
        return Ok(());
    }

    // Send report
    send_crash_report(&sanitized).await?;

    println!("Report sent. Thank you for helping improve ReasonKit!");

    Ok(())
}

fn sanitize_crash_report(report: &str) -> String {
    use crate::telemetry::PrivacyFilter;

    let filter = PrivacyFilter::new(Default::default());
    filter.strip_pii(report)
}
```

### 10.3 Feedback Collection

```rust
// src/commands/feedback.rs

#[derive(Args)]
pub struct FeedbackArgs {
    /// Type of feedback
    #[arg(value_enum)]
    feedback_type: FeedbackType,

    /// Brief description
    message: String,

    /// Include last command's trace
    #[arg(long)]
    include_trace: bool,
}

#[derive(Clone, ValueEnum)]
pub enum FeedbackType {
    Bug,
    Feature,
    Question,
    Other,
}

pub async fn submit_feedback(args: FeedbackArgs) -> anyhow::Result<()> {
    println!("Thank you for your feedback!\n");

    // Create GitHub issue URL with pre-filled template
    let template = match args.feedback_type {
        FeedbackType::Bug => "bug_report.md",
        FeedbackType::Feature => "feature_request.md",
        _ => "blank.md",
    };

    let url = format!(
        "https://github.com/reasonkit/reasonkit-core/issues/new?template={}&title={}&body={}",
        template,
        urlencoding::encode(&args.message),
        urlencoding::encode(&format!(
            "**Version:** {}\n**OS:** {}\n\n",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS
        ))
    );

    println!("Opening browser to submit feedback...");
    open::that(&url)?;

    Ok(())
}
```

### 10.4 User-Facing Error Messages

```rust
// src/error_display.rs

/// Format error for user display
pub fn format_user_error(error: &Error) -> String {
    let mut output = String::new();

    // Error header with code
    output.push_str(&format!("Error [{}]: ", error_code(error)));
    output.push_str(&user_message(error));
    output.push('\n');

    // Suggestion
    if let Some(suggestion) = suggest_fix(error) {
        output.push_str(&format!("\nSuggestion: {}\n", suggestion));
    }

    // Help link
    output.push_str(&format!("\nFor more help: https://reasonkit.sh/errors/{}\n", error_code(error)));

    output
}

fn user_message(error: &Error) -> String {
    match error {
        Error::Config(msg) => {
            format!("Configuration error: {}. Check your config file.", msg)
        }
        Error::Network(msg) => {
            format!("Network error: {}. Check your internet connection.", msg)
        }
        Error::Embedding(msg) => {
            format!("Failed to generate embeddings: {}. The LLM API may be unavailable.", msg)
        }
        _ => error.to_string(),
    }
}

fn suggest_fix(error: &Error) -> Option<String> {
    match error {
        Error::Config(_) => Some("Run 'rk-core config validate' to check your configuration".into()),
        Error::Network(_) => Some("Try again in a few moments, or check the status page".into()),
        _ => None,
    }
}

fn error_code(error: &Error) -> &'static str {
    match error {
        Error::Config(_) => "E100",
        Error::Network(_) => "E200",
        Error::Embedding(_) => "E302",
        Error::Storage(_) => "E400",
        _ => "E000",
    }
}
```

---

## 11. Implementation Checklist

### 11.1 Setup Phase

- [ ] **Choose error tracking service**
  - [ ] Evaluate Sentry vs self-hosted options
  - [ ] Create organization and projects
  - [ ] Configure DSN management

- [ ] **Configure projects**
  - [ ] Create `reasonkit-core` project (Rust)
  - [ ] Create `reasonkit-python` project (Python bindings)
  - [ ] Set up environments (dev/staging/prod)

- [ ] **Add Rust integration**
  - [ ] Add `sentry` crate with appropriate features
  - [ ] Implement `TrackedError` type
  - [ ] Add panic handler integration
  - [ ] Implement PII scrubbing layer

- [ ] **Add Python integration**
  - [ ] Add `sentry-sdk` dependency
  - [ ] Implement exception hierarchy
  - [ ] Add error tracking initialization
  - [ ] Implement PyO3 error conversion

- [ ] **Set up alerts**
  - [ ] Configure critical alerts (fatal errors)
  - [ ] Configure warning alerts (error rate)
  - [ ] Set up daily/weekly digests
  - [ ] Connect to Slack/PagerDuty

- [ ] **Create dashboards**
  - [ ] Sentry default dashboard configuration
  - [ ] Custom Grafana dashboard (if applicable)
  - [ ] Weekly report automation

### 11.2 Testing Phase

- [ ] **Test error capture**
  - [ ] Trigger test errors in development
  - [ ] Verify errors appear in dashboard
  - [ ] Check error grouping is correct

- [ ] **Verify context collection**
  - [ ] Confirm system info is captured
  - [ ] Confirm ThinkTool context is captured
  - [ ] Confirm session/trace IDs are correct

- [ ] **Check PII scrubbing**
  - [ ] Test with known PII patterns
  - [ ] Verify no PII reaches external service
  - [ ] Test edge cases (Unicode, etc.)

- [ ] **Validate alerts**
  - [ ] Trigger test alerts
  - [ ] Verify notification delivery
  - [ ] Test escalation paths

### 11.3 Production Phase

- [ ] **Monitor error rates**
  - [ ] Set up error rate baseline
  - [ ] Configure anomaly detection
  - [ ] Review first week of data

- [ ] **Activate triage process**
  - [ ] Document on-call rotation
  - [ ] Create runbooks for common errors
  - [ ] Train team on triage procedures

- [ ] **Weekly reviews**
  - [ ] Schedule weekly error review meeting
  - [ ] Review top errors and trends
  - [ ] Prioritize fixes

- [ ] **User feedback loop**
  - [ ] Enable crash reporter
  - [ ] Set up feedback collection
  - [ ] Monitor user-reported issues

### 11.4 Ongoing Maintenance

- [ ] **Monthly tasks**
  - [ ] Review and update error codes
  - [ ] Update PII patterns
  - [ ] Review alert thresholds

- [ ] **Quarterly tasks**
  - [ ] Audit data retention
  - [ ] Review consent mechanisms
  - [ ] Update documentation

---

## 12. Troubleshooting

### 12.1 Common Issues

#### Sentry Not Receiving Events

```bash
# Check DSN is set
echo $SENTRY_DSN

# Test connectivity
curl -X POST "https://sentry.io/api/0/" \
  -H "Content-Type: application/json" \
  -d '{"test": true}'

# Enable debug mode
SENTRY_DEBUG=1 rk-core think "test"
```

#### PII Leaking Through

```rust
// Add comprehensive test
#[test]
fn test_pii_scrubbing_comprehensive() {
    let test_cases = vec![
        ("email: test@example.com", false),
        ("api_key=sk-1234567890abcdef", false),
        ("/home/username/file.txt", false),
        ("SELECT * FROM users WHERE id=1", true), // SQL is OK
    ];

    for (input, should_contain_original) in test_cases {
        let scrubbed = scrub_pii(input);
        if should_contain_original {
            assert!(scrubbed.contains(input), "Should preserve: {}", input);
        } else {
            assert!(!scrubbed.contains(input), "Should scrub: {}", input);
        }
    }
}
```

#### High Error Volume

```yaml
# Reduce sample rate temporarily
sentry:
  sample_rate: 0.1  # 10% of errors

# Add rate limiting
rate_limits:
  - path: /api/errors
    limit: 1000/minute
```

### 12.2 Debug Mode

```bash
# Enable verbose error logging
RK_LOG=debug rk-core think "test query"

# Dump error context to file
RK_ERROR_DUMP=/tmp/errors.jsonl rk-core think "test query"

# Test error tracking without external sending
RK_ERROR_TRACKING_DRY_RUN=true rk-core think "test query"
```

### 12.3 Support Contacts

- **Documentation:** https://reasonkit.sh/docs/error-tracking
- **Issues:** https://github.com/reasonkit/reasonkit-core/issues
- **Status Page:** https://status.reasonkit.sh
- **Security Issues:** security@reasonkit.sh

---

## Appendix A: Error Code Reference

| Code Range | Category | Description |
|------------|----------|-------------|
| F001-F099 | Fatal | Unrecoverable crashes |
| E100-E199 | Configuration | Config file errors |
| E200-E299 | Network | API/connectivity errors |
| E300-E399 | Processing | Validation/parsing errors |
| E400-E499 | Storage | File/database errors |
| E500-E599 | Integration | MCP/external tool errors |
| W100-W199 | Performance | Slow operations |
| W200-W299 | Deprecation | Deprecated API usage |
| I100-I199 | Informational | Notable events |

## Appendix B: Related Documents

- [Telemetry Module](/home/zyxsys/RK-PROJECT/reasonkit-core/src/telemetry/mod.rs) - Privacy-first telemetry implementation
- [Error Types](/home/zyxsys/RK-PROJECT/reasonkit-core/src/error.rs) - Core error definitions
- [Privacy Filter](/home/zyxsys/RK-PROJECT/reasonkit-core/src/telemetry/privacy.rs) - PII scrubbing implementation
- [SECURITY_AUDIT_CHECKLIST](/home/zyxsys/RK-PROJECT/reasonkit-core/docs/SECURITY_AUDIT_CHECKLIST.md) - Security audit procedures

---

*Document Version: 1.0.0*
*Last Updated: 2025-12-28*
*Maintainer: ReasonKit Team*
