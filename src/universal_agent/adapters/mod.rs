//! # Framework-Specific Adapters
//!
//! Individual adapters for each supported AI agent framework

pub mod claude;
pub mod cline;
pub mod kilo;
pub mod droid;
pub mod roo;
pub mod blackbox;

pub use claude::ClaudeCodeAdapter;
pub use cline::ClineAdapter;
pub use kilo::KiloCodeAdapter;
pub use droid::DroidAdapter;
pub use roo::RooCodeAdapter;
pub use blackbox::BlackBoxAIAdapter;

use crate::error::Result;
use crate::universal_agent::types::*;
use crate::thinktool::{Protocol, ProtocolContent};
use async_trait::async_trait;

/// Common adapter functionality shared across all frameworks
pub struct BaseAdapter {
    pub framework_type: FrameworkType,
    pub performance_metrics: PerformanceMetrics,
    pub health_status: HealthStatus,
}

impl BaseAdapter {
    pub fn new(framework_type: FrameworkType) -> Self {
        Self {
            framework_type,
            performance_metrics: PerformanceMetrics {
                framework_type,
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                average_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                throughput_rps: 0.0,
                error_rate: 0.0,
                last_updated: chrono::Utc::now(),
            },
            health_status: HealthStatus {
                is_healthy: true,
                response_time_ms: 0,
                last_check: chrono::Utc::now(),
                issues: Vec::new(),
                performance_metrics: None,
            },
        }
    }

    /// Update performance metrics after processing
    pub fn update_performance(&mut self, success: bool, latency_ms: u64) {
        self.performance_metrics.total_requests += 1;

        if success {
            self.performance_metrics.successful_requests += 1;
        } else {
            self.performance_metrics.failed_requests += 1;
        }

        // Update latency metrics
        self.performance_metrics.average_latency_ms =
            (self.performance_metrics.average_latency_ms * 0.9) + (latency_ms as f64 * 0.1);

        if latency_ms > self.performance_metrics.p95_latency_ms as u64 {
            self.performance_metrics.p95_latency_ms = latency_ms as f64;
        }

        // Update throughput (requests per second)
        self.performance_metrics.throughput_rps =
            self.performance_metrics.successful_requests as f64 /
            (chrono::Utc::now().signed_duration_since(self.performance_metrics.last_updated).num_seconds() as f64 + 1.0);

        // Update error rate
        self.performance_metrics.error_rate =
            self.performance_metrics.failed_requests as f64 / self.performance_metrics.total_requests as f64;

        self.performance_metrics.last_updated = chrono::Utc::now();
    }

    /// Check if adapter meets M2 performance standards
    pub fn meets_m2_standards(&self) -> bool {
        let metrics = &self.performance_metrics;
        metrics.success_rate() >= 0.95
            && metrics.average_latency_ms <= 50.0
            && metrics.error_rate <= 0.05
    }

    /// Get framework-specific optimization features
    pub fn get_optimization_features(&self) -> Vec<String> {
        match self.framework_type {
            FrameworkType::ClaudeCode => vec![
                "json_optimization".to_string(),
                "confidence_scoring".to_string(),
                "structured_output".to_string(),
                "priority_processing".to_string(),
            ],
            FrameworkType::Cline => vec![
                "logical_analysis".to_string(),
                "fallacy_detection".to_string(),
                "deductive_reasoning".to_string(),
                "argument_validation".to_string(),
            ],
            FrameworkType::KiloCode => vec![
                "comprehensive_critique".to_string(),
                "flaw_categorization".to_string(),
                "deep_analysis".to_string(),
                "quality_assessment".to_string(),
            ],
            FrameworkType::Droid => vec![
                "mobile_optimization".to_string(),
                "android_specific".to_string(),
                "apk_integration".to_string(),
                "resource_efficiency".to_string(),
            ],
            FrameworkType::RooCode => vec![
                "multi_agent_collaboration".to_string(),
                "protocol_delegation".to_string(),
                "agent_coordination".to_string(),
                "workflow_orchestration".to_string(),
            ],
            FrameworkType::BlackBoxAI => vec![
                "high_throughput".to_string(),
                "speed_optimization".to_string(),
                "batch_processing".to_string(),
                "parallel_execution".to_string(),
            ],
        }
    }

    /// Get framework-specific security features
    pub fn get_security_features(&self) -> Vec<String> {
        match self.framework_type {
            FrameworkType::ClaudeCode => vec![
                "input_validation".to_string(),
                "output_sanitization".to_string(),
                "secure_communication".to_string(),
            ],
            FrameworkType::Cline => vec![
                "logical_validation".to_string(),
                "fallacy_detection".to_string(),
                "reasoning_integrity".to_string(),
            ],
            FrameworkType::KiloCode => vec![
                "comprehensive_validation".to_string(),
                "quality_assurance".to_string(),
                "thorough_checking".to_string(),
            ],
            FrameworkType::Droid => vec![
                "mobile_security".to_string(),
                "android_permissions".to_string(),
                "secure_storage".to_string(),
            ],
            FrameworkType::RooCode => vec![
                "agent_authentication".to_string(),
                "secure_delegation".to_string(),
                "collaboration_security".to_string(),
            ],
            FrameworkType::BlackBoxAI => vec![
                "high_speed_validation".to_string(),
                "throughput_security".to_string(),
                "batch_validation".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_adapter_creation() {
        let adapter = BaseAdapter::new(FrameworkType::ClaudeCode);
        assert_eq!(adapter.framework_type, FrameworkType::ClaudeCode);
        assert!(adapter.meets_m2_standards());
    }

    #[test]
    fn test_performance_update() {
        let mut adapter = BaseAdapter::new(FrameworkType::ClaudeCode);

        adapter.update_performance(true, 45);
        assert_eq!(adapter.performance_metrics.successful_requests, 1);
        assert_eq!(adapter.performance_metrics.total_requests, 1);

        adapter.update_performance(false, 60);
        assert_eq!(adapter.performance_metrics.failed_requests, 1);
        assert_eq!(adapter.performance_metrics.total_requests, 2);
    }

    #[test]
    fn test_optimization_features() {
        let adapter = BaseAdapter::new(FrameworkType::ClaudeCode);
        let features = adapter.get_optimization_features();
        assert!(features.contains(&"json_optimization".to_string()));
        assert!(features.contains(&"confidence_scoring".to_string()));
    }

    #[test]
    fn test_security_features() {
        let adapter = BaseAdapter::new(FrameworkType::Cline);
        let features = adapter.get_security_features();
        assert!(features.contains(&"logical_validation".to_string()));
        assert!(features.contains(&"fallacy_detection".to_string()));
    }
}
