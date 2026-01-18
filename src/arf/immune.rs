//! # Immune System Module
//!
//! Self-defense and stability mechanisms for the ARF platform.
//! This module monitors system health, prevents resource abuse, and maintains stability.

use crate::error::Result;
use std::collections::HashMap;
use std::sync::Arc;
use sysinfo::{CpuExt, DiskExt, ProcessExt, System, SystemExt};
use tokio::sync::RwLock;
use tokio::time::{self, Duration};

/// System health metrics
#[derive(Debug, Clone)]
pub struct HealthMetrics {
    pub cpu_usage: f32,
    pub memory_usage_mb: u64,
    pub disk_usage_percent: f32,
    pub network_connections: usize,
    pub active_processes: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Resource limits and thresholds
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_cpu_percent: f32,
    pub max_memory_mb: u64,
    pub max_disk_percent: f32,
    pub max_network_connections: usize,
    pub max_process_age_seconds: u64,
}

/// Immune response actions
#[derive(Debug, Clone)]
pub enum ImmuneAction {
    Warning(String),
    Throttle(String),
    Terminate(String),
    Quarantine(String),
    Log(String),
}

/// Immune system for self-defense and stability
pub struct ImmuneSystem {
    system_monitor: Arc<RwLock<System>>,
    health_history: Arc<RwLock<Vec<HealthMetrics>>>,
    resource_limits: ResourceLimits,
    active_responses: Arc<RwLock<HashMap<String, ImmuneAction>>>,
    threat_patterns: Arc<RwLock<Vec<ThreatPattern>>>,
}

#[derive(Debug, Clone)]
pub struct ThreatPattern {
    pattern_type: ThreatType,
    signature: String,
    _severity: ThreatSeverity,
    response: ImmuneAction,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ThreatType {
    ResourceAbuse,
    UnauthorizedAccess,
    MaliciousCode,
    SystemInstability,
    PluginMisbehavior,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl ImmuneSystem {
    /// Create a new immune system
    pub fn new(resource_limits: ResourceLimits) -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self {
            system_monitor: Arc::new(RwLock::new(system)),
            health_history: Arc::new(RwLock::new(Vec::new())),
            resource_limits,
            active_responses: Arc::new(RwLock::new(HashMap::new())),
            threat_patterns: Arc::new(RwLock::new(Self::load_default_patterns())),
        }
    }

    /// Start the immune system monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        let system_monitor = Arc::clone(&self.system_monitor);
        let health_history = Arc::clone(&self.health_history);
        let resource_limits = self.resource_limits.clone();
        let active_responses = Arc::clone(&self.active_responses);
        let threat_patterns = Arc::clone(&self.threat_patterns);

        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                if let Err(e) = Self::monitor_system_health(
                    &system_monitor,
                    &health_history,
                    &resource_limits,
                    &active_responses,
                    &threat_patterns,
                )
                .await
                {
                    tracing::error!("Immune system monitoring error: {}", e);
                }
            }
        });

        tracing::info!("Immune system monitoring started");
        Ok(())
    }

    /// Monitor system health and respond to threats
    async fn monitor_system_health(
        system_monitor: &Arc<RwLock<System>>,
        health_history: &Arc<RwLock<Vec<HealthMetrics>>>,
        resource_limits: &ResourceLimits,
        active_responses: &Arc<RwLock<HashMap<String, ImmuneAction>>>,
        threat_patterns: &Arc<RwLock<Vec<ThreatPattern>>>,
    ) -> Result<()> {
        // Refresh system information
        {
            let mut system = system_monitor.write().await;
            system.refresh_all();
        }

        let system = system_monitor.read().await;

        // Collect current health metrics
        let metrics = HealthMetrics {
            cpu_usage: system.global_cpu_info().cpu_usage(),
            memory_usage_mb: system.used_memory() / 1024 / 1024,
            disk_usage_percent: Self::calculate_disk_usage(),
            network_connections: 0, // Would need network monitoring
            active_processes: system.processes().len(),
            timestamp: chrono::Utc::now(),
        };

        // Store metrics history
        {
            let mut history = health_history.write().await;
            history.push(metrics.clone());

            // Keep only last 100 readings
            if history.len() > 100 {
                history.remove(0);
            }
        }

        // Check resource limits
        let violations = Self::check_resource_violations(&metrics, resource_limits);

        // Respond to violations
        for violation in violations {
            Self::respond_to_threat(violation, active_responses, threat_patterns).await?;
        }

        // Check for threat patterns
        Self::scan_for_threats(&system, threat_patterns, active_responses).await?;

        Ok(())
    }

    /// Check for resource limit violations
    fn check_resource_violations(metrics: &HealthMetrics, limits: &ResourceLimits) -> Vec<String> {
        let mut violations = Vec::new();

        if metrics.cpu_usage > limits.max_cpu_percent {
            violations.push(format!(
                "CPU usage {:.1}% exceeds limit {:.1}%",
                metrics.cpu_usage, limits.max_cpu_percent
            ));
        }

        if metrics.memory_usage_mb > limits.max_memory_mb {
            violations.push(format!(
                "Memory usage {}MB exceeds limit {}MB",
                metrics.memory_usage_mb, limits.max_memory_mb
            ));
        }

        if metrics.disk_usage_percent > limits.max_disk_percent {
            violations.push(format!(
                "Disk usage {:.1}% exceeds limit {:.1}%",
                metrics.disk_usage_percent, limits.max_disk_percent
            ));
        }

        violations
    }

    /// Calculate disk usage percentage
    fn calculate_disk_usage() -> f32 {
        let mut system = System::new();
        system.refresh_disks_list();
        system.refresh_disks();

        let mut total_percent = 0.0;
        let mut count = 0u32;

        for disk in system.disks() {
            let total = disk.total_space() as f32;
            let available = disk.available_space() as f32;
            if total > 0.0 {
                total_percent += ((total - available) / total) * 100.0;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total_percent / count as f32
        }
    }

    /// Respond to detected threats
    async fn respond_to_threat(
        violation: String,
        active_responses: &Arc<RwLock<HashMap<String, ImmuneAction>>>,
        threat_patterns: &Arc<RwLock<Vec<ThreatPattern>>>,
    ) -> Result<()> {
        let patterns = threat_patterns.read().await;

        // Find matching threat pattern
        for pattern in patterns.iter() {
            if violation.contains(&pattern.signature) {
                let response_key = format!("threat_{}", chrono::Utc::now().timestamp());

                let mut responses = active_responses.write().await;
                responses.insert(response_key.clone(), pattern.response.clone());

                // Execute response
                Self::execute_immune_response(&pattern.response, &violation).await?;

                tracing::warn!(
                    "Immune response activated: {} for violation: {}",
                    response_key,
                    violation
                );
                break;
            }
        }

        Ok(())
    }

    /// Execute an immune response
    async fn execute_immune_response(response: &ImmuneAction, context: &str) -> Result<()> {
        match response {
            ImmuneAction::Warning(message) => {
                tracing::warn!("IMMUNE WARNING: {} - Context: {}", message, context);
            }
            ImmuneAction::Throttle(resource) => {
                tracing::warn!(
                    "IMMUNE THROTTLE: Throttling {} - Context: {}",
                    resource,
                    context
                );
                // Would implement actual throttling logic
            }
            ImmuneAction::Terminate(process) => {
                tracing::error!(
                    "IMMUNE TERMINATION: Terminating {} - Context: {}",
                    process,
                    context
                );
                // Would implement process termination
            }
            ImmuneAction::Quarantine(component) => {
                tracing::error!(
                    "IMMUNE QUARANTINE: Isolating {} - Context: {}",
                    component,
                    context
                );
                // Would implement component isolation
            }
            ImmuneAction::Log(entry) => {
                tracing::info!("IMMUNE LOG: {} - Context: {}", entry, context);
            }
        }

        Ok(())
    }

    /// Scan for threat patterns in the system
    async fn scan_for_threats(
        system: &System,
        threat_patterns: &Arc<RwLock<Vec<ThreatPattern>>>,
        active_responses: &Arc<RwLock<HashMap<String, ImmuneAction>>>,
    ) -> Result<()> {
        let patterns = threat_patterns.read().await;

        // Check processes for suspicious activity
        for (pid, process) in system.processes() {
            for pattern in patterns.iter() {
                if Self::matches_threat_pattern(process, pattern) {
                    let violation = format!(
                        "Threat pattern detected: {} in process {}",
                        pattern.signature, pid
                    );
                    Self::respond_to_threat(violation, active_responses, threat_patterns).await?;
                }
            }
        }

        Ok(())
    }

    /// Check if a process matches a threat pattern
    fn matches_threat_pattern(process: &sysinfo::Process, pattern: &ThreatPattern) -> bool {
        let process_name = process.name().to_lowercase();
        let pattern_sig = pattern.signature.to_lowercase();

        match pattern.pattern_type {
            ThreatType::ResourceAbuse => {
                process.cpu_usage() > 90.0 || process.memory() > 1_000_000_000 // 1GB
            }
            ThreatType::UnauthorizedAccess => {
                process_name.contains("ssh") && process.cpu_usage() > 50.0
            }
            ThreatType::MaliciousCode => {
                process_name.contains(&pattern_sig)
                    || process.cmd().iter().any(|arg| arg.contains(&pattern_sig))
            }
            ThreatType::SystemInstability => process.status() == sysinfo::ProcessStatus::Zombie,
            ThreatType::PluginMisbehavior => {
                process_name.contains("plugin") && process.cpu_usage() > 80.0
            }
        }
    }

    /// Load default threat patterns
    fn load_default_patterns() -> Vec<ThreatPattern> {
        vec![
            ThreatPattern {
                pattern_type: ThreatType::ResourceAbuse,
                signature: "high_cpu".to_string(),
                _severity: ThreatSeverity::Medium,
                response: ImmuneAction::Throttle("CPU intensive processes".to_string()),
            },
            ThreatPattern {
                pattern_type: ThreatType::ResourceAbuse,
                signature: "high_memory".to_string(),
                _severity: ThreatSeverity::High,
                response: ImmuneAction::Terminate("Memory hog processes".to_string()),
            },
            ThreatPattern {
                pattern_type: ThreatType::MaliciousCode,
                signature: "exploit".to_string(),
                _severity: ThreatSeverity::Critical,
                response: ImmuneAction::Quarantine("Suspicious processes".to_string()),
            },
            ThreatPattern {
                pattern_type: ThreatType::SystemInstability,
                signature: "zombie".to_string(),
                _severity: ThreatSeverity::High,
                response: ImmuneAction::Log("Zombie process detected".to_string()),
            },
        ]
    }

    /// Add a custom threat pattern
    pub async fn add_threat_pattern(&self, pattern: ThreatPattern) -> Result<()> {
        let mut patterns = self.threat_patterns.write().await;
        patterns.push(pattern);
        Ok(())
    }

    /// Get current health status
    pub async fn get_health_status(&self) -> Result<SystemHealth> {
        let history = self.health_history.read().await;
        let responses = self.active_responses.read().await;

        let latest_metrics = history.last().cloned().unwrap_or_else(|| HealthMetrics {
            cpu_usage: 0.0,
            memory_usage_mb: 0,
            disk_usage_percent: 0.0,
            network_connections: 0,
            active_processes: 0,
            timestamp: chrono::Utc::now(),
        });

        Ok(SystemHealth {
            current_metrics: latest_metrics,
            active_threats: responses.len(),
            threat_response_history: responses.values().cloned().collect(),
            overall_status: if responses.is_empty() {
                HealthStatus::Healthy
            } else {
                HealthStatus::UnderThreat
            },
        })
    }

    /// Perform system self-diagnosis
    pub async fn self_diagnose(&self) -> Result<DiagnosticReport> {
        let health = self.get_health_status().await?;
        let history = self.health_history.read().await;

        // Analyze trends
        let cpu_trend = Self::analyze_trend(history.iter().map(|m| m.cpu_usage).collect());
        let memory_trend =
            Self::analyze_trend(history.iter().map(|m| m.memory_usage_mb as f32).collect());

        let recommendations = Self::generate_recommendations(&health, &cpu_trend, &memory_trend);

        Ok(DiagnosticReport {
            health_status: health,
            system_trends: SystemTrends {
                cpu_trend,
                memory_trend,
                stability_score: Self::calculate_stability_score(history.as_slice()),
            },
            recommendations,
        })
    }

    /// Analyze metric trends
    fn analyze_trend(values: Vec<f32>) -> MetricTrend {
        if values.len() < 2 {
            return MetricTrend::Stable;
        }

        let recent_avg = values.iter().rev().take(5).sum::<f32>() / 5.0;
        let older_avg = values
            .iter()
            .take(values.len().saturating_sub(5))
            .sum::<f32>()
            / values.len().saturating_sub(5).max(1) as f32;

        let change_percent = if older_avg > 0.0 {
            ((recent_avg - older_avg) / older_avg) * 100.0
        } else {
            0.0
        };

        if change_percent > 10.0 {
            MetricTrend::Increasing
        } else if change_percent < -10.0 {
            MetricTrend::Decreasing
        } else {
            MetricTrend::Stable
        }
    }

    /// Calculate system stability score
    fn calculate_stability_score(history: &[HealthMetrics]) -> f64 {
        if history.is_empty() {
            return 1.0;
        }

        // Calculate variance in key metrics
        let cpu_values: Vec<f32> = history.iter().map(|m| m.cpu_usage).collect();
        let memory_values: Vec<f32> = history.iter().map(|m| m.memory_usage_mb as f32).collect();

        let cpu_variance = Self::calculate_variance(&cpu_values);
        let memory_variance = Self::calculate_variance(&memory_values);

        // Lower variance = higher stability
        let stability = 1.0 / (1.0 + cpu_variance + memory_variance);
        stability.clamp(0.0, 1.0)
    }

    /// Calculate variance of a dataset
    fn calculate_variance(values: &[f32]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance as f64
    }

    /// Generate health recommendations
    fn generate_recommendations(
        health: &SystemHealth,
        cpu_trend: &MetricTrend,
        memory_trend: &MetricTrend,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if health.current_metrics.cpu_usage > 80.0 {
            recommendations.push(
                "High CPU usage detected. Consider optimizing compute-intensive operations."
                    .to_string(),
            );
        }

        if matches!(cpu_trend, MetricTrend::Increasing) {
            recommendations.push(
                "CPU usage is trending upward. Monitor for potential performance issues."
                    .to_string(),
            );
        }

        if health.current_metrics.memory_usage_mb > 8000 {
            // 8GB
            recommendations.push(
                "High memory usage detected. Consider implementing memory optimization strategies."
                    .to_string(),
            );
        }

        if matches!(memory_trend, MetricTrend::Increasing) {
            recommendations.push(
                "Memory usage is trending upward. Check for potential memory leaks.".to_string(),
            );
        }

        if health.active_threats > 0 {
            recommendations.push(format!(
                "{} active threats detected. Review immune system responses.",
                health.active_threats
            ));
        }

        if recommendations.is_empty() {
            recommendations.push("System health is good. Continue monitoring.".to_string());
        }

        recommendations
    }
}

/// System health status
#[derive(Debug, Clone)]
pub struct SystemHealth {
    pub current_metrics: HealthMetrics,
    pub active_threats: usize,
    pub threat_response_history: Vec<ImmuneAction>,
    pub overall_status: HealthStatus,
}

/// System health status enum
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Warning,
    UnderThreat,
    Critical,
}

/// Metric trend analysis
#[derive(Debug, Clone)]
pub enum MetricTrend {
    Increasing,
    Decreasing,
    Stable,
}

/// System trends analysis
#[derive(Debug, Clone)]
pub struct SystemTrends {
    pub cpu_trend: MetricTrend,
    pub memory_trend: MetricTrend,
    pub stability_score: f64,
}

/// Diagnostic report
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    pub health_status: SystemHealth,
    pub system_trends: SystemTrends,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_immune_system_creation() {
        let limits = ResourceLimits {
            max_cpu_percent: 80.0,
            max_memory_mb: 8192,
            max_disk_percent: 90.0,
            max_network_connections: 1000,
            max_process_age_seconds: 3600,
        };

        let immune = ImmuneSystem::new(limits);
        let health = immune.get_health_status().await.unwrap();
        assert!(matches!(health.overall_status, HealthStatus::Healthy));
    }

    #[tokio::test]
    async fn test_resource_violation_detection() {
        let limits = ResourceLimits {
            max_cpu_percent: 50.0,
            max_memory_mb: 4096,
            max_disk_percent: 90.0,
            max_network_connections: 1000,
            max_process_age_seconds: 3600,
        };

        let metrics = HealthMetrics {
            cpu_usage: 75.0,       // Above limit
            memory_usage_mb: 3000, // Below limit
            disk_usage_percent: 85.0,
            network_connections: 500,
            active_processes: 100,
            timestamp: chrono::Utc::now(),
        };

        let violations = ImmuneSystem::check_resource_violations_static(&metrics, &limits);
        assert!(!violations.is_empty());
        assert!(violations[0].contains("CPU usage"));
    }

    impl ImmuneSystem {
        fn check_resource_violations_static(
            metrics: &HealthMetrics,
            limits: &ResourceLimits,
        ) -> Vec<String> {
            Self::check_resource_violations(metrics, limits)
        }
    }
}
