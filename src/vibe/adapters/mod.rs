//! # VIBE Validation Adapters
//!
//! Cross-platform validation adapters and integration components for seamless
//! VIBE protocol validation across different environments and frameworks.

use super::*;
use crate::vibe::platforms::{ProxyConfiguration, ResourceRequirements, SslConfiguration};
use crate::vibe::validation::{VIBEError, ValidationIssue, ValidationStatus};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;

// NOTE: Most adapter functionality currently lives in this module.
// Declarations for split-out modules were removed because the corresponding
// files were never added.

// Main adapter types are defined in this module.

/// Protocol validation suite combining multiple validation approaches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSuite {
    /// Suite identifier
    pub suite_id: Uuid,

    /// Suite metadata
    pub name: String,
    pub description: String,

    /// Validation components
    pub components: Vec<ValidationComponent>,

    /// Suite configuration
    pub config: SuiteConfig,

    /// Execution results
    pub results: Vec<SuiteResult>,
}

impl ComponentPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_val = match self {
            ComponentPriority::Low => 0,
            ComponentPriority::Normal => 1,
            ComponentPriority::High => 2,
            ComponentPriority::Critical => 3,
        };
        let other_val = match other {
            ComponentPriority::Low => 0,
            ComponentPriority::Normal => 1,
            ComponentPriority::High => 2,
            ComponentPriority::Critical => 3,
        };
        self_val.cmp(&other_val)
    }
}

/// Individual validation component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationComponent {
    pub component_id: Uuid,
    pub name: String,
    pub component_type: ValidationComponentType,
    pub configuration: ComponentConfiguration,
}

/// Types of validation components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationComponentType {
    VIBEValidation,
    ThinkToolValidation,
    ProtocolDeltaValidation,
    CustomValidator,
    ExternalIntegration,
}

/// Component-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConfiguration {
    pub enabled: bool,
    pub priority: ComponentPriority,
    pub timeout_ms: Option<u64>,
    pub custom_parameters: HashMap<String, serde_json::Value>,
}

/// Component execution priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteConfig {
    pub parallel_execution: bool,
    pub fail_fast: bool,
    pub continue_on_warning: bool,
    pub aggregation_method: AggregationMethod,
    pub custom_rules: Vec<SuiteRule>,
}

/// Methods for aggregating multiple validation results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationMethod {
    WeightedAverage,
    BestScore,
    WorstScore,
    MajorityVote,
    Custom,
}

/// Custom rules for suite execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteRule {
    pub rule_name: String,
    pub condition: RuleCondition,
    pub action: RuleAction,
}

/// Rule condition for triggering actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    pub component_type: Option<ValidationComponentType>,
    pub score_threshold: Option<f32>,
    pub issue_count_threshold: Option<usize>,
    pub custom_condition: Option<String>,
}

/// Rule action to execute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleAction {
    pub action_type: RuleActionType,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Types of rule actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleActionType {
    SkipComponent,
    IncreaseTimeout,
    AdjustScore,
    AddCustomIssue,
    RetryValidation,
}

/// Suite execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteResult {
    pub result_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub overall_result: OverallSuiteResult,
    pub component_results: Vec<ComponentResult>,
    pub execution_metrics: ExecutionMetrics,
}

/// Overall suite result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallSuiteResult {
    pub overall_score: f32,
    pub status: SuiteStatus,
    pub confidence_level: f32,
    pub execution_time_ms: u64,
    pub components_executed: usize,
    pub components_passed: usize,
    pub components_failed: usize,
}

/// Suite execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuiteStatus {
    Passed,
    Failed,
    Warning,
    Partial,
    Error,
}

/// Individual component result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentResult {
    pub component_id: Uuid,
    pub component_type: ValidationComponentType,
    pub execution_result: ComponentExecutionResult,
    pub performance_metrics: ComponentPerformanceMetrics,
}

/// Component execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentExecutionResult {
    pub success: bool,
    pub score: Option<f32>,
    pub issues: Vec<ValidationIssue>,
    pub recommendations: Vec<String>,
    pub data: HashMap<String, serde_json::Value>,
}

/// Performance metrics for component execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPerformanceMetrics {
    pub execution_time_ms: u64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f32,
    pub network_requests: u32,
    pub cache_hits: u32,
    pub cache_misses: u32,
}

/// Suite execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub total_execution_time_ms: u64,
    pub parallel_efficiency: f32,
    pub resource_utilization: ResourceUtilization,
    pub bottleneck_analysis: BottleneckAnalysis,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub peak_memory_mb: u64,
    pub peak_cpu_percent: f32,
    pub network_bandwidth_mbps: f32,
    pub disk_io_mb_per_second: f32,
}

/// Bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub slowest_component: Option<(String, u64)>,
    pub most_memory_intensive: Option<(String, u64)>,
    pub highest_cpu_usage: Option<(String, f32)>,
    pub optimization_suggestions: Vec<String>,
}

/// Validation adapter trait for cross-platform integration
#[async_trait]
pub trait ValidationAdapter: Send + Sync {
    /// Initialize the adapter
    async fn initialize(&self, config: &AdapterConfig) -> Result<(), VIBEError>;

    /// Validate protocol using this adapter
    async fn validate(
        &self,
        protocol: &str,
        config: &ValidationConfig,
    ) -> Result<AdapterValidationResult, VIBEError>;

    /// Get adapter capabilities
    fn get_capabilities(&self) -> AdapterCapabilities;

    /// Get adapter health status
    async fn health_check(&self) -> Result<AdapterHealthStatus, VIBEError>;

    /// Cleanup adapter resources
    async fn cleanup(&self) -> Result<(), VIBEError>;
}

/// Adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub adapter_id: String,
    pub adapter_type: AdapterType,
    pub connection_settings: ConnectionSettings,
    pub timeout_settings: TimeoutSettings,
    pub retry_settings: RetrySettings,
}

/// Types of adapters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterType {
    VIBE,
    ThinkTool,
    ProtocolDelta,
    ExternalAPI,
    Custom,
}

/// Connection settings for adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSettings {
    pub endpoint: Option<String>,
    pub authentication: Option<AuthenticationConfig>,
    pub ssl_config: Option<SslConfiguration>,
    pub proxy_config: Option<ProxyConfiguration>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    pub auth_type: AuthType,
    pub credentials: HashMap<String, String>,
    pub token: Option<String>,
}

/// Authentication types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthType {
    None,
    Basic,
    Bearer,
    OAuth2,
    ApiKey,
}

/// Timeout settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutSettings {
    pub connection_timeout_ms: u64,
    pub read_timeout_ms: u64,
    pub write_timeout_ms: u64,
}

/// Retry settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrySettings {
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub backoff_multiplier: f32,
    pub max_retry_delay_ms: u64,
}

/// Adapter capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterCapabilities {
    pub supported_protocols: Vec<String>,
    pub supported_platforms: Vec<Platform>,
    pub features: Vec<String>,
    pub limitations: Vec<String>,
    pub performance_characteristics: AdapterPerformanceCharacteristics,
}

/// Adapter performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterPerformanceCharacteristics {
    pub typical_latency_ms: u64,
    pub throughput_capacity: f32,
    pub resource_requirements: ResourceRequirements,
    pub scalability_limits: ScalabilityLimits,
}

/// Scalability limits for adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityLimits {
    pub max_concurrent_validations: u32,
    pub max_protocol_size_bytes: u64,
    pub max_queue_size: u32,
}

/// Adapter health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterHealthStatus {
    pub healthy: bool,
    pub status_message: String,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
    pub metrics: AdapterMetrics,
}

/// Adapter metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetrics {
    pub total_validations: u64,
    pub successful_validations: u64,
    pub failed_validations: u64,
    pub average_latency_ms: f32,
    pub error_rate_percent: f32,
}

/// Result from adapter validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterValidationResult {
    pub adapter_id: String,
    pub validation_id: Uuid,
    pub success: bool,
    pub score: Option<f32>,
    pub platform_scores: HashMap<Platform, f32>,
    pub issues: Vec<ValidationIssue>,
    pub recommendations: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub execution_time_ms: u64,
}

#[async_trait]
impl ValidationAdapter for Box<dyn ValidationAdapter> {
    async fn initialize(&self, config: &AdapterConfig) -> Result<(), VIBEError> {
        self.as_ref().initialize(config).await
    }

    async fn validate(
        &self,
        protocol: &str,
        config: &ValidationConfig,
    ) -> Result<AdapterValidationResult, VIBEError> {
        self.as_ref().validate(protocol, config).await
    }

    fn get_capabilities(&self) -> AdapterCapabilities {
        self.as_ref().get_capabilities()
    }

    async fn health_check(&self) -> Result<AdapterHealthStatus, VIBEError> {
        self.as_ref().health_check().await
    }

    async fn cleanup(&self) -> Result<(), VIBEError> {
        self.as_ref().cleanup().await
    }
}

/// Cross-platform validator combining multiple validation approaches
pub struct CrossPlatformValidator {
    /// Core VIBE engine
    vibe_engine: super::validation::VIBEEngine,

    /// Registered adapters
    adapters: HashMap<String, Box<dyn ValidationAdapter>>,

    /// Execution configuration
    config: CrossPlatformConfig,

    /// Performance metrics
    metrics: Arc<RwLock<CrossPlatformMetrics>>,
}

/// Cross-platform validation configuration
#[derive(Debug, Clone)]
pub struct CrossPlatformConfig {
    pub default_timeout_ms: u64,
    pub max_concurrent_adapters: usize,
    pub enable_failover: bool,
    pub aggregation_strategy: AggregationStrategy,
    pub health_check_interval_ms: u64,
}

/// Aggregation strategies for multiple adapters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    VIBEWeighted,
    BestOfBreed,
    Consensus,
    Custom,
}

/// Cross-platform performance metrics
#[derive(Debug, Clone)]
pub struct CrossPlatformMetrics {
    pub total_validations: u64,
    pub successful_validations: u64,
    pub failed_validations: u64,
    pub adapter_utilization: HashMap<String, f32>,
    pub average_execution_time_ms: f32,
    pub success_rate: f32,
    pub error_breakdown: HashMap<String, u32>,
}

impl Default for CrossPlatformMetrics {
    fn default() -> Self {
        Self {
            total_validations: 0,
            successful_validations: 0,
            failed_validations: 0,
            adapter_utilization: HashMap::new(),
            average_execution_time_ms: 0.0,
            success_rate: 0.0,
            error_breakdown: HashMap::new(),
        }
    }
}

impl CrossPlatformValidator {
    /// Create new cross-platform validator
    pub fn new(vibe_engine: super::validation::VIBEEngine) -> Self {
        Self {
            vibe_engine,
            adapters: HashMap::new(),
            config: CrossPlatformConfig {
                default_timeout_ms: 30000,
                max_concurrent_adapters: 5,
                enable_failover: true,
                aggregation_strategy: AggregationStrategy::VIBEWeighted,
                health_check_interval_ms: 30000,
            },
            metrics: Arc::new(RwLock::new(CrossPlatformMetrics::default())),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        vibe_engine: super::validation::VIBEEngine,
        config: CrossPlatformConfig,
    ) -> Self {
        Self {
            vibe_engine,
            adapters: HashMap::new(),
            config,
            metrics: Arc::new(RwLock::new(CrossPlatformMetrics::default())),
        }
    }

    /// Register validation adapter
    pub fn register_adapter(&mut self, adapter_id: String, adapter: Box<dyn ValidationAdapter>) {
        self.adapters.insert(adapter_id, adapter);
    }

    /// Unregister validation adapter
    pub fn unregister_adapter(&mut self, adapter_id: &str) -> Option<Box<dyn ValidationAdapter>> {
        self.adapters.remove(adapter_id)
    }

    /// Execute cross-platform validation using multiple adapters
    pub async fn validate_cross_platform(
        &self,
        protocol: &str,
        config: &ValidationConfig,
    ) -> Result<CrossPlatformValidationResult, VIBEError> {
        let start_time = std::time::Instant::now();

        // Execute validation using all healthy adapters
        let mut adapter_results = Vec::new();

        for (adapter_id, adapter) in &self.adapters {
            let health = adapter.health_check().await?;
            if !health.healthy {
                continue;
            }

            let adapter_result = self
                .execute_adapter_validation(adapter_id, adapter.as_ref(), protocol, config)
                .await?;

            adapter_results.push(adapter_result);
        }

        if adapter_results.is_empty() {
            return Err(VIBEError::AdapterError("No available adapters".to_string()));
        }

        // Aggregate results using configured strategy
        let aggregated_result = self.aggregate_results(&adapter_results, config)?;

        // Update metrics
        self.update_metrics(&adapter_results).await?;

        let execution_time = start_time.elapsed().as_millis() as u64;

        let confidence_level = self.calculate_confidence_level(&adapter_results)?;

        Ok(CrossPlatformValidationResult {
            validation_id: Uuid::new_v4(),
            overall_score: aggregated_result.score,
            status: aggregated_result.status,
            adapter_results,
            aggregated_score: aggregated_result,
            execution_time_ms: execution_time,
            confidence_level,
        })
    }

    /// Execute validation suite
    pub async fn execute_suite(
        &self,
        suite: &ValidationSuite,
        protocol: &str,
    ) -> Result<SuiteResult, VIBEError> {
        let start_time = std::time::Instant::now();

        // Validate suite configuration
        self.validate_suite_config(suite)?;

        // Sort components by priority
        let mut sorted_components = suite.components.clone();
        sorted_components.sort_by(|a, b| b.configuration.priority.cmp(&a.configuration.priority));

        // Execute components
        let mut component_results = Vec::new();
        let mut overall_status = SuiteStatus::Passed;
        let mut total_score = 0.0;
        let mut score_count = 0;

        for component in &sorted_components {
            if !component.configuration.enabled {
                continue;
            }

            // Check if we should continue based on previous results
            if !self.should_continue_execution(
                &overall_status,
                &component_results,
                &suite.config,
            )? {
                break;
            }

            let result = self.execute_component(component, protocol).await?;

            // Update overall status
            overall_status =
                self.determine_overall_status(&overall_status, &result, &suite.config)?;

            // Accumulate scores
            if let Some(score) = result.execution_result.score {
                total_score += score;
                score_count += 1;
            }

            component_results.push(result);
        }

        let execution_time = start_time.elapsed().as_millis() as u64;
        let final_score = if score_count > 0 {
            total_score / score_count as f32
        } else {
            0.0
        };

        let confidence_level = self.calculate_suite_confidence(&component_results)?;
        let parallel_efficiency =
            self.calculate_parallel_efficiency(&component_results, &suite.config)?;
        let resource_utilization = self.analyze_resource_utilization(&component_results)?;
        let bottleneck_analysis = self.analyze_bottlenecks(&component_results)?;

        Ok(SuiteResult {
            result_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            overall_result: OverallSuiteResult {
                overall_score: final_score,
                status: overall_status,
                confidence_level,
                execution_time_ms: execution_time,
                components_executed: component_results.len(),
                components_passed: component_results
                    .iter()
                    .filter(|r| r.execution_result.success)
                    .count(),
                components_failed: component_results
                    .iter()
                    .filter(|r| !r.execution_result.success)
                    .count(),
            },
            component_results,
            execution_metrics: ExecutionMetrics {
                total_execution_time_ms: execution_time,
                parallel_efficiency,
                resource_utilization,
                bottleneck_analysis,
            },
        })
    }

    /// Get health status of all adapters
    pub async fn get_adapters_health(
        &self,
    ) -> Result<HashMap<String, AdapterHealthStatus>, VIBEError> {
        let mut health_status = HashMap::new();

        for (adapter_id, adapter) in &self.adapters {
            let status = adapter.health_check().await?;
            health_status.insert(adapter_id.clone(), status);
        }

        Ok(health_status)
    }

    /// Get performance metrics
    pub async fn get_metrics(&self) -> Result<CrossPlatformMetrics, VIBEError> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }

    // Helper methods

    async fn execute_adapter_validation(
        &self,
        adapter_id: &str,
        adapter: &dyn ValidationAdapter,
        protocol: &str,
        config: &ValidationConfig,
    ) -> Result<AdapterValidationResult, VIBEError> {
        let start_time = std::time::Instant::now();

        let result = adapter.validate(protocol, config).await?;

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(AdapterValidationResult {
            adapter_id: adapter_id.to_string(),
            validation_id: Uuid::new_v4(),
            success: result.success,
            score: result.score,
            platform_scores: result.platform_scores,
            issues: result.issues,
            recommendations: result.recommendations,
            metadata: result.metadata,
            execution_time_ms: execution_time,
        })
    }

    fn aggregate_results(
        &self,
        adapter_results: &[AdapterValidationResult],
        config: &ValidationConfig,
    ) -> Result<AggregatedScore, VIBEError> {
        let successful_results: Vec<&AdapterValidationResult> =
            adapter_results.iter().filter(|r| r.success).collect();

        if successful_results.is_empty() {
            return Err(VIBEError::AdapterError(
                "No successful validation results".to_string(),
            ));
        }

        let score = match self.config.aggregation_strategy {
            AggregationStrategy::VIBEWeighted => {
                // Weight VIBE results higher
                let vibe_score = successful_results
                    .iter()
                    .find(|r| r.adapter_id == "vibe")
                    .map(|r| r.score.unwrap_or(0.0))
                    .unwrap_or(0.0);

                let other_scores: Vec<f32> = successful_results
                    .iter()
                    .filter(|r| r.adapter_id != "vibe")
                    .filter_map(|r| r.score)
                    .collect();

                if other_scores.is_empty() {
                    vibe_score
                } else {
                    let other_avg = other_scores.iter().sum::<f32>() / other_scores.len() as f32;
                    (vibe_score * 0.6 + other_avg * 0.4).clamp(0.0, 100.0)
                }
            }
            AggregationStrategy::BestOfBreed => successful_results
                .iter()
                .filter_map(|r| r.score)
                .fold(0.0f32, f32::max),
            AggregationStrategy::Consensus => {
                // Simple consensus - average of all scores
                let scores: Vec<f32> = successful_results.iter().filter_map(|r| r.score).collect();
                scores.iter().sum::<f32>() / scores.len() as f32
            }
            _ => {
                // Default to VIBE weighted
                self.aggregate_results(adapter_results, config)?.score
            }
        };

        let status = if score >= config.minimum_score {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        };

        Ok(AggregatedScore {
            score,
            status,
            contributing_adapters: successful_results.len(),
            confidence_factors: self.calculate_confidence_factors(&successful_results)?,
        })
    }

    fn calculate_confidence_level(
        &self,
        adapter_results: &[AdapterValidationResult],
    ) -> Result<f32, VIBEError> {
        let successful_count = adapter_results.iter().filter(|r| r.success).count();
        let total_count = adapter_results.len();

        if total_count == 0 {
            return Ok(0.0);
        }

        let agreement_rate = successful_count as f32 / total_count as f32;
        let refs: Vec<_> = adapter_results.iter().collect();
        let score_variance = self.calculate_score_variance(&refs)?;

        // Higher agreement and lower variance = higher confidence
        let confidence = agreement_rate * (1.0 - (score_variance / 100.0)).max(0.0f32);
        Ok(confidence.clamp(0.0, 1.0))
    }

    fn calculate_score_variance(
        &self,
        results: &[&AdapterValidationResult],
    ) -> Result<f32, VIBEError> {
        let scores: Vec<f32> = results.iter().filter_map(|r| r.score).collect();

        if scores.len() < 2 {
            return Ok(0.0);
        }

        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores
            .iter()
            .map(|&score| (score - mean).powi(2))
            .sum::<f32>()
            / scores.len() as f32;

        Ok(variance)
    }

    fn calculate_confidence_factors(
        &self,
        results: &[&AdapterValidationResult],
    ) -> Result<HashMap<String, f32>, VIBEError> {
        let mut factors = HashMap::new();

        // Adapter agreement factor
        let scores: Vec<f32> = results.iter().filter_map(|r| r.score).collect();
        if scores.len() > 1 {
            let variance = self.calculate_score_variance(results)?;
            factors.insert(
                "agreement_factor".to_string(),
                (1.0 - variance / 100.0).max(0.0f32),
            );
        }

        // Execution time factor (faster = more confident)
        let avg_time = results
            .iter()
            .map(|r| r.execution_time_ms as f32)
            .sum::<f32>()
            / results.len() as f32;
        let time_factor = (30000.0 / avg_time).min(1.0); // Normalize to 30s baseline
        factors.insert("execution_time_factor".to_string(), time_factor);

        Ok(factors)
    }

    async fn update_metrics(&self, results: &[AdapterValidationResult]) -> Result<(), VIBEError> {
        let mut metrics = self.metrics.write().await;

        metrics.total_validations += 1;

        for result in results {
            if result.success {
                metrics.successful_validations += 1;
            } else {
                metrics.failed_validations += 1;
            }

            // Update adapter utilization
            let utilization = metrics
                .adapter_utilization
                .entry(result.adapter_id.clone())
                .or_insert(0.0);
            *utilization += 1.0;

            // Update error breakdown
            if !result.success {
                let error_count = metrics
                    .error_breakdown
                    .entry(result.adapter_id.clone())
                    .or_insert(0);
                *error_count += 1;
            }
        }

        // Calculate success rate
        if metrics.total_validations > 0 {
            metrics.success_rate =
                metrics.successful_validations as f32 / metrics.total_validations as f32;
        }

        // Normalize utilization percentages
        let total_validations = metrics.total_validations as f32;
        for utilization in metrics.adapter_utilization.values_mut() {
            *utilization = (*utilization / total_validations) * 100.0;
        }

        Ok(())
    }

    fn validate_suite_config(&self, suite: &ValidationSuite) -> Result<(), VIBEError> {
        if suite.components.is_empty() {
            return Err(VIBEError::AdapterError(
                "Validation suite has no components".to_string(),
            ));
        }

        // Check for duplicate component IDs
        let mut component_ids = HashSet::new();
        for component in &suite.components {
            if !component_ids.insert(component.component_id) {
                return Err(VIBEError::AdapterError(format!(
                    "Duplicate component ID: {:?}",
                    component.component_id
                )));
            }
        }

        Ok(())
    }

    fn should_continue_execution(
        &self,
        current_status: &SuiteStatus,
        _results: &[ComponentResult],
        config: &SuiteConfig,
    ) -> Result<bool, VIBEError> {
        if config.fail_fast && current_status == &SuiteStatus::Failed {
            return Ok(false);
        }

        if !config.continue_on_warning && current_status == &SuiteStatus::Warning {
            return Ok(false);
        }

        Ok(true)
    }

    async fn execute_component(
        &self,
        component: &ValidationComponent,
        protocol: &str,
    ) -> Result<ComponentResult, VIBEError> {
        let start_time = std::time::Instant::now();

        let (success, score, issues, recommendations, data) = match component.component_type {
            ValidationComponentType::VIBEValidation => {
                let vibe_config = ValidationConfig::default();
                match self
                    .vibe_engine
                    .validate_protocol(protocol, vibe_config)
                    .await
                {
                    Ok(result) => {
                        let super::validation::ValidationResult {
                            overall_score,
                            issues,
                            recommendations,
                            platform_scores,
                            ..
                        } = result;

                        let recommendations =
                            recommendations.into_iter().map(|r| r.title).collect();

                        (true, Some(overall_score), issues, recommendations, {
                            let mut data = HashMap::new();
                            data.insert(
                                "platform_scores".to_string(),
                                serde_json::to_value(&platform_scores).unwrap(),
                            );
                            data
                        })
                    }
                    Err(e) => (false, None, vec![], vec![], {
                        let mut data = HashMap::new();
                        data.insert(
                            "error".to_string(),
                            serde_json::to_value(e.to_string()).unwrap(),
                        );
                        data
                    }),
                }
            }
            // Add other component types as needed
            _ => {
                return Err(VIBEError::AdapterError(format!(
                    "Unsupported component type: {:?}",
                    component.component_type
                )));
            }
        };

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(ComponentResult {
            component_id: component.component_id,
            component_type: component.component_type,
            execution_result: ComponentExecutionResult {
                success,
                score,
                issues,
                recommendations,
                data,
            },
            performance_metrics: ComponentPerformanceMetrics {
                execution_time_ms: execution_time,
                memory_usage_mb: 100,    // Simplified
                cpu_usage_percent: 25.0, // Simplified
                network_requests: 0,
                cache_hits: 0,
                cache_misses: 0,
            },
        })
    }

    fn determine_overall_status(
        &self,
        current_status: &SuiteStatus,
        result: &ComponentResult,
        _config: &SuiteConfig,
    ) -> Result<SuiteStatus, VIBEError> {
        match (current_status, result.execution_result.success) {
            (SuiteStatus::Passed, false) => Ok(SuiteStatus::Failed),
            (SuiteStatus::Passed, true) => {
                if result.execution_result.score.unwrap_or(0.0) < 70.0 {
                    Ok(SuiteStatus::Warning)
                } else {
                    Ok(SuiteStatus::Passed)
                }
            }
            (SuiteStatus::Warning, false) => Ok(SuiteStatus::Failed),
            (SuiteStatus::Warning, true) => Ok(SuiteStatus::Warning),
            (SuiteStatus::Failed, _) => Ok(SuiteStatus::Failed),
            _ => Ok(*current_status),
        }
    }

    fn calculate_suite_confidence(&self, results: &[ComponentResult]) -> Result<f32, VIBEError> {
        let successful_results: Vec<&ComponentResult> = results
            .iter()
            .filter(|r| r.execution_result.success)
            .collect();
        let total_results = results.len();

        if total_results == 0 {
            return Ok(0.0);
        }

        let success_rate = successful_results.len() as f32 / total_results as f32;

        // Factor in score consistency
        let scores: Vec<f32> = results
            .iter()
            .filter_map(|r| r.execution_result.score)
            .collect();

        let score_consistency = if scores.len() > 1 {
            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            let variance = scores
                .iter()
                .map(|&score| (score - mean).powi(2))
                .sum::<f32>()
                / scores.len() as f32;
            (1.0 - (variance / 100.0)).max(0.0f32)
        } else {
            1.0
        };

        Ok(success_rate * score_consistency)
    }

    fn calculate_parallel_efficiency(
        &self,
        results: &[ComponentResult],
        config: &SuiteConfig,
    ) -> Result<f32, VIBEError> {
        if !config.parallel_execution || results.len() < 2 {
            return Ok(1.0);
        }

        let total_time: u64 = results
            .iter()
            .map(|r| r.performance_metrics.execution_time_ms)
            .sum();

        let max_time = results
            .iter()
            .map(|r| r.performance_metrics.execution_time_ms)
            .max()
            .unwrap_or(0);

        if max_time == 0 {
            return Ok(1.0);
        }

        Ok(total_time as f32 / (max_time as f32 * results.len() as f32))
    }

    fn analyze_resource_utilization(
        &self,
        results: &[ComponentResult],
    ) -> Result<ResourceUtilization, VIBEError> {
        let peak_memory = results
            .iter()
            .map(|r| r.performance_metrics.memory_usage_mb)
            .max()
            .unwrap_or(0);

        let peak_cpu = results
            .iter()
            .map(|r| r.performance_metrics.cpu_usage_percent)
            .fold(0.0f32, f32::max);

        Ok(ResourceUtilization {
            peak_memory_mb: peak_memory,
            peak_cpu_percent: peak_cpu,
            network_bandwidth_mbps: 10.0, // Simplified
            disk_io_mb_per_second: 5.0,   // Simplified
        })
    }

    fn analyze_bottlenecks(
        &self,
        results: &[ComponentResult],
    ) -> Result<BottleneckAnalysis, VIBEError> {
        let slowest = results
            .iter()
            .max_by_key(|r| r.performance_metrics.execution_time_ms)
            .map(|r| {
                (
                    format!("{:?}", r.component_type),
                    r.performance_metrics.execution_time_ms,
                )
            });

        let most_memory = results
            .iter()
            .max_by_key(|r| r.performance_metrics.memory_usage_mb)
            .map(|r| {
                (
                    format!("{:?}", r.component_type),
                    r.performance_metrics.memory_usage_mb,
                )
            });

        let highest_cpu = results
            .iter()
            .max_by(|a, b| {
                a.performance_metrics
                    .cpu_usage_percent
                    .partial_cmp(&b.performance_metrics.cpu_usage_percent)
                    .unwrap()
            })
            .map(|r| {
                (
                    format!("{:?}", r.component_type),
                    r.performance_metrics.cpu_usage_percent,
                )
            });

        Ok(BottleneckAnalysis {
            slowest_component: slowest,
            most_memory_intensive: most_memory,
            highest_cpu_usage: highest_cpu,
            optimization_suggestions: vec![
                "Consider parallel execution for independent components".to_string(),
                "Optimize memory-intensive operations".to_string(),
            ],
        })
    }
}

/// Result of aggregating multiple adapter results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedScore {
    score: f32,
    status: ValidationStatus,
    contributing_adapters: usize,
    confidence_factors: HashMap<String, f32>,
}

/// Cross-platform validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformValidationResult {
    pub validation_id: Uuid,
    pub overall_score: f32,
    pub status: ValidationStatus,
    pub adapter_results: Vec<AdapterValidationResult>,
    pub aggregated_score: AggregatedScore,
    pub execution_time_ms: u64,
    pub confidence_level: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_suite_creation() {
        let suite = ValidationSuite {
            suite_id: Uuid::new_v4(),
            name: "Test Suite".to_string(),
            description: "A test validation suite".to_string(),
            components: vec![ValidationComponent {
                component_id: Uuid::new_v4(),
                name: "VIBE Validation".to_string(),
                component_type: ValidationComponentType::VIBEValidation,
                configuration: ComponentConfiguration {
                    enabled: true,
                    priority: ComponentPriority::Normal,
                    timeout_ms: Some(30000),
                    custom_parameters: HashMap::new(),
                },
            }],
            config: SuiteConfig {
                parallel_execution: true,
                fail_fast: false,
                continue_on_warning: true,
                aggregation_method: AggregationMethod::WeightedAverage,
                custom_rules: Vec::new(),
            },
            results: Vec::new(),
        };

        assert_eq!(suite.components.len(), 1);
        assert_eq!(
            suite.components[0].component_type,
            ValidationComponentType::VIBEValidation
        );
    }

    #[test]
    fn test_adapter_config_creation() {
        let config = AdapterConfig {
            adapter_id: "test_adapter".to_string(),
            adapter_type: AdapterType::VIBE,
            connection_settings: ConnectionSettings {
                endpoint: Some("http://localhost:9100".to_string()),
                authentication: None,
                ssl_config: None,
                proxy_config: None,
            },
            timeout_settings: TimeoutSettings {
                connection_timeout_ms: 5000,
                read_timeout_ms: 10000,
                write_timeout_ms: 5000,
            },
            retry_settings: RetrySettings {
                max_retries: 3,
                retry_delay_ms: 1000,
                backoff_multiplier: 2.0,
                max_retry_delay_ms: 10000,
            },
        };

        assert_eq!(config.adapter_id, "test_adapter");
        assert_eq!(config.adapter_type, AdapterType::VIBE);
    }
}
