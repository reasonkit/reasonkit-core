//! # Framework Types and Core Types
//!
//! Core type definitions for the Universal Agent Integration Framework

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported AI Agent Frameworks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameworkType {
    /// Claude Code: JSON-formatted outputs with confidence scoring
    ClaudeCode,
    /// Cline: Structured logical analysis with fallacy detection
    Cline,
    /// Kilo Code: Comprehensive critique with flaw categorization
    KiloCode,
    /// Droid (Factory AI): Android development with mobile-specific optimizations
    Droid,
    /// Roo Code: Multi-agent collaboration with protocol delegation
    RooCode,
    /// BlackBox AI: High-throughput operations with speed optimization
    BlackBoxAI,
}

impl FrameworkType {
    /// Get all supported framework types
    pub fn all() -> &'static [Self] {
        &[
            Self::ClaudeCode,
            Self::Cline,
            Self::KiloCode,
            Self::Droid,
            Self::RooCode,
            Self::BlackBoxAI,
        ]
    }

    /// Get framework display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::ClaudeCode => "Claude Code",
            Self::Cline => "Cline",
            Self::KiloCode => "Kilo Code",
            Self::Droid => "Droid (Factory AI)",
            Self::RooCode => "Roo Code",
            Self::BlackBoxAI => "BlackBox AI",
        }
    }

    /// Get framework priority for auto-detection (1 = highest priority)
    pub fn priority(&self) -> u8 {
        match self {
            Self::ClaudeCode => 1,
            Self::Cline => 2,
            Self::KiloCode => 3,
            Self::Droid => 4,
            Self::RooCode => 5,
            Self::BlackBoxAI => 6,
        }
    }

    /// Get maximum supported context length for this framework
    pub fn max_context_length(&self) -> usize {
        match self {
            Self::ClaudeCode => 200_000,
            Self::Cline => 150_000,
            Self::KiloCode => 180_000,
            Self::Droid => 100_000,
            Self::RooCode => 120_000,
            Self::BlackBoxAI => 250_000,
        }
    }

    /// Check if framework supports real-time processing
    pub fn supports_realtime(&self) -> bool {
        match self {
            Self::ClaudeCode => true,
            Self::Cline => true,
            Self::KiloCode => false,
            Self::Droid => true,
            Self::RooCode => false,
            Self::BlackBoxAI => true,
        }
    }
}

impl fmt::Display for FrameworkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Framework capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkCapability {
    pub framework_type: FrameworkType,
    pub name: String,
    pub version: String,
    pub supported_protocols: Vec<String>,
    pub max_context_length: usize,
    pub supports_realtime: bool,
    pub performance_rating: f64,
    pub optimization_features: Vec<String>,
    pub security_features: Vec<String>,
}

/// Agent framework registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRegistration {
    pub capability: FrameworkCapability,
    pub registered_at: chrono::DateTime<chrono::Utc>,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
    pub usage_count: u64,
    pub average_performance: f64,
}

/// Framework output format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    StructuredText,
    LogicalAnalysis,
    MobileOptimized,
    MultiAgentProtocol,
    HighThroughput,
}

/// Processing result from a framework adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedProtocol {
    pub content: ProtocolContent,
    pub confidence_score: f64,
    pub processing_time_ms: u64,
    pub framework_used: FrameworkType,
    pub format: OutputFormat,
    pub optimizations_applied: Vec<String>,
    pub metadata: ProcessingMetadata,
}

impl ProcessedProtocol {
    /// Create a new processed protocol result
    pub fn new(
        content: ProtocolContent,
        confidence_score: f64,
        processing_time_ms: u64,
        framework_used: FrameworkType,
        format: OutputFormat,
    ) -> Self {
        Self {
            content,
            confidence_score,
            processing_time_ms,
            framework_used,
            format,
            optimizations_applied: Vec::new(),
            metadata: ProcessingMetadata::default(),
        }
    }

    /// Check if processing was successful
    pub fn is_success(&self) -> bool {
        self.confidence_score >= 0.8 && self.processing_time_ms < 1000
    }
}

/// Processing metadata for tracking and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub protocol_version: String,
    pub optimization_level: OptimizationLevel,
    pub cache_hit: bool,
    pub parallel_processing_used: bool,
    pub memory_usage_mb: Option<f64>,
    pub cpu_usage_percent: Option<f64>,
}

/// Optimization level for protocol processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization, fastest processing
    None,
    /// Basic optimization, balanced performance
    Basic,
    /// Medium optimization, better results
    Medium,
    /// High optimization, best results
    High,
    /// Maximum optimization, slowest but best
    Maximum,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        Self::Medium
    }
}

impl fmt::Display for OptimizationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Basic => write!(f, "basic"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Maximum => write!(f, "maximum"),
        }
    }
}

/// Framework adapter trait
/// All framework adapters must implement this trait
#[async_trait::async_trait]
pub trait FrameworkAdapter: Send + Sync {
    /// Get the framework type this adapter supports
    fn framework_type(&self) -> FrameworkType;

    /// Process a protocol through this framework
    async fn process_protocol(&self, protocol: &Protocol) -> Result<ProcessedProtocol>;

    /// Get framework capabilities
    async fn get_capabilities(&self) -> Result<FrameworkCapability>;

    /// Benchmark performance for this framework
    async fn benchmark_performance(&self) -> Result<BenchmarkResult>;

    /// Validate if protocol is compatible with this framework
    async fn validate_compatibility(&self, protocol: &Protocol) -> Result<CompatibilityResult>;

    /// Get adapter health status
    async fn health_check(&self) -> Result<HealthStatus>;
}

/// Compatibility check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    pub is_compatible: bool,
    pub compatibility_score: f64,
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Health status of a framework adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub response_time_ms: u64,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub issues: Vec<String>,
    pub performance_metrics: Option<PerformanceMetrics>,
}

/// Benchmark result for a framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub framework_type: FrameworkType,
    pub success_rate: f64,
    pub average_latency_ms: f64,
    pub throughput_rps: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub confidence_score: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl BenchmarkResult {
    /// Check if benchmark meets M2 performance standards
    pub fn meets_m2_standards(&self) -> bool {
        self.success_rate >= 0.95
            && self.average_latency_ms <= 50.0
            && self.confidence_score >= 0.90
    }
}

/// Performance metrics for a framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub framework_type: FrameworkType,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_rps: f64,
    pub error_rate: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl PerformanceMetrics {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }

    /// Check if metrics meet performance targets
    pub fn meets_targets(&self, targets: &PerformanceTargets) -> bool {
        self.success_rate() >= targets.min_success_rate
            && self.average_latency_ms <= targets.max_latency_ms
            && self.error_rate <= targets.max_error_rate
    }
}

/// Performance targets for frameworks
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    pub min_success_rate: f64,
    pub max_latency_ms: f64,
    pub max_error_rate: f64,
    pub min_throughput_rps: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            min_success_rate: 0.95,
            max_latency_ms: 50.0,
            max_error_rate: 0.05,
            min_throughput_rps: 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_type_all() {
        let all_frameworks = FrameworkType::all();
        assert_eq!(all_frameworks.len(), 6);
        assert!(all_frameworks.contains(&FrameworkType::ClaudeCode));
        assert!(all_frameworks.contains(&FrameworkType::Cline));
    }

    #[test]
    fn test_framework_priority() {
        assert_eq!(FrameworkType::ClaudeCode.priority(), 1);
        assert_eq!(FrameworkType::BlackBoxAI.priority(), 6);
    }

    #[test]
    fn test_processed_protocol_creation() {
        let content = ProtocolContent::Text("test content".to_string());
        let result = ProcessedProtocol::new(
            content.clone(),
            0.95,
            45,
            FrameworkType::ClaudeCode,
            OutputFormat::Json,
        );

        assert!(result.is_success());
        assert_eq!(result.framework_used, FrameworkType::ClaudeCode);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            framework_type: FrameworkType::ClaudeCode,
            total_requests: 1000,
            successful_requests: 950,
            failed_requests: 50,
            average_latency_ms: 45.0,
            p95_latency_ms: 65.0,
            p99_latency_ms: 85.0,
            throughput_rps: 120.0,
            error_rate: 0.05,
            last_updated: chrono::Utc::now(),
        };

        assert_eq!(metrics.success_rate(), 0.95);
    }
}
