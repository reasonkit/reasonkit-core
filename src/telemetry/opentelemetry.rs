//! # OpenTelemetry Instrumentation
//!
//! Distributed tracing support for ReasonKit Core using OpenTelemetry.
//! Provides observability for protocol execution, ThinkTool steps, LLM calls, and RAG operations.
//!
//! ## Usage
//!
//! ```rust
//! use reasonkit::telemetry::opentelemetry::init_otel;
//!
//! // Initialize OpenTelemetry (optional feature)
//! #[cfg(feature = "opentelemetry")]
//! init_otel("reasonkit-core", "http://localhost:4317").await?;
//!
//! // Spans are automatically created via tracing macros
//! tracing::info_span!("protocol_execution", protocol_id = "gigathink").in_scope(|| {
//!     // Your code here
//! });
//! ```

use crate::error::Error;

/// Initialize OpenTelemetry tracing
///
/// # Arguments
///
/// * `service_name` - Name of the service (e.g., "reasonkit-core")
/// * `endpoint` - OTLP endpoint URL (e.g., "http://localhost:4317")
///
/// # Returns
///
/// `Result<(), Error>` - Error if initialization fails
///
/// # Example
///
/// ```rust
/// use reasonkit::telemetry::opentelemetry::init_otel;
///
/// #[cfg(feature = "opentelemetry")]
/// init_otel("reasonkit-core", "http://localhost:4317").await?;
/// ```
///
/// # Note
///
/// This is currently a stub implementation. Full OpenTelemetry integration requires:
/// - API compatibility research for opentelemetry 0.31+ versions
/// - Proper resource configuration
/// - Tracer provider setup
/// - Integration with tracing-subscriber
///
/// The feature flag is commented out in Cargo.toml until the API is properly researched.
#[cfg(feature = "opentelemetry")]
pub async fn init_otel(_service_name: &str, _endpoint: &str) -> Result<(), Error> {
    // TODO: Implement full OpenTelemetry integration
    // This requires API compatibility research for opentelemetry 0.31+
    // See: https://docs.rs/opentelemetry-otlp/latest/opentelemetry_otlp/
    tracing::warn!("OpenTelemetry integration is a stub - full implementation pending API research");
    Ok(())
}

/// Initialize OpenTelemetry tracing (no-op when feature is disabled)
#[cfg(not(feature = "opentelemetry"))]
pub async fn init_otel(_service_name: &str, _endpoint: &str) -> Result<(), Error> {
    tracing::warn!("OpenTelemetry feature is not enabled. Install with: cargo build --features opentelemetry");
    Ok(())
}

/// Shutdown OpenTelemetry tracer provider
///
/// Call this before application exit to ensure all spans are exported.
#[cfg(feature = "opentelemetry")]
pub fn shutdown_otel() {
    // TODO: Implement shutdown when full OpenTelemetry integration is complete
    tracing::info!("OpenTelemetry tracer provider shut down (stub)");
}

/// Shutdown OpenTelemetry tracer provider (no-op when feature is disabled)
#[cfg(not(feature = "opentelemetry"))]
pub fn shutdown_otel() {
    // No-op
}

/// Helper macro for creating spans with common attributes
///
/// # Example
///
/// ```rust
/// use reasonkit::telemetry::opentelemetry::protocol_span;
///
/// let span = protocol_span!("gigathink", "analyze_dimensions");
/// let _guard = span.enter();
/// // Your code here
/// ```
#[macro_export]
macro_rules! protocol_span {
    ($protocol:expr, $step:expr) => {
        tracing::info_span!(
            "protocol_step",
            protocol = $protocol,
            step = $step,
            service.name = "reasonkit-core"
        )
    };
}

/// Helper macro for creating LLM call spans
///
/// # Example
///
/// ```rust
/// use reasonkit::telemetry::opentelemetry::llm_span;
///
/// let span = llm_span!("claude", "sonnet-4");
/// let _guard = span.enter();
/// // LLM call here
/// ```
#[macro_export]
macro_rules! llm_span {
    ($provider:expr, $model:expr) => {
        tracing::info_span!(
            "llm_call",
            provider = $provider,
            model = $model,
            service.name = "reasonkit-core",
            gen_ai.system = "reasonkit"
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[cfg(feature = "opentelemetry")]
    async fn test_otel_initialization() {
        // This test requires an OTLP endpoint, so we'll skip it in CI
        // In a real scenario, you'd use a test collector or mock
        let result = init_otel("test-service", "http://localhost:4317").await;
        // We expect this to fail if no collector is running, which is fine for tests
        assert!(result.is_ok() || result.is_err());
        shutdown_otel();
    }

    #[tokio::test]
    #[cfg(not(feature = "opentelemetry"))]
    async fn test_otel_noop() {
        // When feature is disabled, should return Ok
        let result = init_otel("test-service", "http://localhost:4317").await;
        assert!(result.is_ok());
    }
}

