//! # VIBE Validation Engine
//!
//! Core validation engine implementing MiniMax M2's "Agent-as-a-Verifier" paradigm
//! for comprehensive protocol validation across multiple platforms.

use super::validation_config::ValidationConfig;
use super::*;
use futures::future::join_all;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Semaphore};

/// The main VIBE validation engine that orchestrates protocol validation
/// across multiple platforms using the "Agent-as-a-Verifier" paradigm.
pub struct VIBEEngine {
    /// Internal state protected by RwLock for concurrent access
    state: Arc<RwLock<EngineState>>,

    /// Validation adapters for different platforms
    adapters: HashMap<Platform, Arc<dyn PlatformValidator>>,

    /// Performance tracking and metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,

    /// Concurrent validation limit
    concurrent_limit: usize,

    /// Validation cache for performance optimization
    cache: Arc<RwLock<ValidationCache>>,
}

/// Internal engine state
#[derive(Debug, Clone)]
struct EngineState {
    /// Total protocols validated
    protocols_validated: u64,

    /// Platform usage statistics
    platform_stats: HashMap<Platform, u64>,

    /// Recent validation results for trend analysis
    recent_results: VecDeque<ValidationResult>,

    /// System health metrics
    health_metrics: HealthMetrics,
}

/// Combined validation result across platforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub overall_score: f32,
    pub platform_scores: HashMap<Platform, f32>,
    pub confidence_interval: Option<ConfidenceInterval>,
    pub status: ValidationStatus,
    pub detailed_results: HashMap<Platform, PlatformValidationResult>,
    pub validation_time_ms: u64,
    pub issues: Vec<ValidationIssue>,
    pub recommendations: Vec<Recommendation>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub protocol_id: Uuid,
}

/// Health metrics for monitoring system performance
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct HealthMetrics {
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: u64,
    pub disk_usage_mb: u64,
    pub network_requests_per_minute: u32,
    pub validation_success_rate: f32,
}

/// Performance metrics tracking
#[derive(Debug, Default, Clone)]
pub struct PerformanceMetrics {
    pub average_validation_time_ms: u64,
    pub fastest_validation_ms: u64,
    pub slowest_validation_ms: u64,
    pub protocols_per_minute: f32,
    pub platform_distribution: HashMap<Platform, f32>,
    pub error_rate_percent: f32,
}

/// Validation cache for performance optimization
#[derive(Debug, Default)]
struct ValidationCache {
    /// Cache of recent validation results
    results: HashMap<Uuid, CachedValidation>,

    /// Platform-specific performance data
    platform_data: HashMap<Platform, PlatformPerformance>,

    /// Cache statistics
    hit_rate_percent: f32,
    total_requests: u64,
    cache_hits: u64,
}

/// Cached validation result
#[derive(Debug, Clone)]
struct CachedValidation {
    result: ValidationResult,
    timestamp: chrono::DateTime<chrono::Utc>,
    ttl_seconds: u64,
}

/// Platform-specific performance data
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PlatformPerformance {
    pub average_response_time_ms: u64,
    pub success_rate: f32,
    pub error_patterns: HashMap<String, u32>,
    pub optimization_suggestions: Vec<String>,
}

impl VIBEEngine {
    /// Create a new VIBE validation engine
    pub fn new() -> Self {
        let mut engine = Self {
            state: Arc::new(RwLock::new(EngineState {
                protocols_validated: 0,
                platform_stats: HashMap::new(),
                recent_results: VecDeque::new(),
                health_metrics: HealthMetrics {
                    cpu_usage_percent: 0.0,
                    memory_usage_mb: 0,
                    disk_usage_mb: 0,
                    network_requests_per_minute: 0,
                    validation_success_rate: 1.0,
                },
            })),
            adapters: HashMap::new(),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            concurrent_limit: 10,
            cache: Arc::new(RwLock::new(ValidationCache::default())),
        };

        // Initialize default platform adapters
        engine.initialize_default_adapters();

        engine
    }

    /// Create VIBE engine with custom configuration
    pub fn with_config(
        concurrent_limit: usize,
        custom_adapters: HashMap<Platform, Arc<dyn PlatformValidator>>,
    ) -> Self {
        let mut engine = Self {
            state: Arc::new(RwLock::new(EngineState {
                protocols_validated: 0,
                platform_stats: HashMap::new(),
                recent_results: VecDeque::new(),
                health_metrics: HealthMetrics {
                    cpu_usage_percent: 0.0,
                    memory_usage_mb: 0,
                    disk_usage_mb: 0,
                    network_requests_per_minute: 0,
                    validation_success_rate: 1.0,
                },
            })),
            adapters: custom_adapters,
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            concurrent_limit,
            cache: Arc::new(RwLock::new(ValidationCache::default())),
        };

        // Add any missing default adapters
        engine.initialize_default_adapters();

        engine
    }

    /// Initialize default platform validators
    fn initialize_default_adapters(&mut self) {
        self.adapters.entry(Platform::Web).or_insert_with(|| {
            Arc::new(super::platforms::WebValidator::new()) as Arc<dyn PlatformValidator>
        });

        self.adapters
            .entry(Platform::Simulation)
            .or_insert_with(|| {
                Arc::new(super::platforms::SimulationValidator::new()) as Arc<dyn PlatformValidator>
            });

        self.adapters.entry(Platform::Android).or_insert_with(|| {
            Arc::new(super::platforms::AndroidValidator::new()) as Arc<dyn PlatformValidator>
        });

        self.adapters.entry(Platform::IOS).or_insert_with(|| {
            Arc::new(super::platforms::IOSValidator::new()) as Arc<dyn PlatformValidator>
        });

        self.adapters.entry(Platform::Backend).or_insert_with(|| {
            Arc::new(super::platforms::BackendValidator::new()) as Arc<dyn PlatformValidator>
        });
    }

    /// Validate a protocol across specified platforms using Agent-as-Verifier paradigm
    pub async fn validate_protocol(
        &self,
        protocol_content: &str,
        config: ValidationConfig,
    ) -> Result<ValidationResult> {
        let start_time = Instant::now();

        // Generate protocol ID for caching and tracking
        let protocol_id = self.generate_protocol_id(protocol_content);

        // Check cache for recent validation
        if let Some(cached_result) = self.get_cached_validation(&protocol_id).await? {
            tracing::info!(
                "Using cached validation result for protocol {}",
                protocol_id
            );
            return Ok(cached_result);
        }

        // Validate protocol content
        self.validate_protocol_content(protocol_content)?;

        // Create validation semaphore for concurrent platform validation
        let semaphore = Arc::new(Semaphore::new(self.concurrent_limit));

        // Spawn validation tasks for each platform
        let mut validation_tasks = Vec::new();

        for platform in &config.target_platforms {
            if let Some(adapter) = self.adapters.get(platform) {
                let adapter = Arc::clone(adapter);
                let semaphore_clone = semaphore.clone();
                let config_clone = config.clone();
                let protocol_content = protocol_content.to_string();
                let platform = *platform;

                let task = tokio::spawn(async move {
                    let _permit = semaphore_clone.acquire().await.unwrap();
                    adapter
                        .validate_protocol(&protocol_content, &config_clone, platform)
                        .await
                });

                validation_tasks.push(task);
            } else {
                tracing::warn!("No validator adapter found for platform {:?}", platform);
            }
        }

        // Wait for all platform validations to complete
        let join_results = join_all(validation_tasks).await;
        let mut platform_results = Vec::with_capacity(join_results.len());
        for join_result in join_results {
            match join_result {
                Ok(Ok(result)) => platform_results.push(result),
                Ok(Err(e)) => return Err(e),
                Err(e) => {
                    return Err(VIBEError::ValidationError(format!(
                        "Platform worker task failed: {e}"
                    )))
                }
            }
        }

        // Aggregate results using Agent-as-Verifier logic
        let aggregated_result = self
            .aggregate_platform_results(platform_results, &config, start_time)
            .await?;

        // Update statistics and cache
        self.update_statistics(&aggregated_result).await?;
        self.cache_validation_result(&protocol_id, &aggregated_result)
            .await?;

        Ok(aggregated_result)
    }

    /// Validate protocol content for basic requirements
    fn validate_protocol_content(&self, content: &str) -> Result<()> {
        if content.trim().is_empty() {
            return Err(VIBEError::InvalidProtocol(
                "Protocol content is empty".to_string(),
            ));
        }

        if content.len() > 100_000 {
            return Err(VIBEError::InvalidProtocol(
                "Protocol content exceeds maximum size".to_string(),
            ));
        }

        // Check for basic protocol structure
        if !content.contains("protocol") && !content.contains("Protocol") {
            tracing::warn!("Protocol content may not follow standard structure");
        }

        Ok(())
    }

    /// Aggregate results from multiple platform validators
    async fn aggregate_platform_results(
        &self,
        platform_results: Vec<PlatformValidationResult>,
        config: &ValidationConfig,
        start_time: Instant,
    ) -> Result<ValidationResult> {
        let validation_duration = start_time.elapsed();

        // Collect scores from all platforms
        let mut score_values = Vec::new();
        let mut platform_scores: HashMap<Platform, f32> = HashMap::new();
        let mut detailed_results = HashMap::new();
        let mut overall_issues = Vec::new();

        for result in platform_results {
            score_values.push(result.score);
            platform_scores.insert(result.platform, result.score);
            detailed_results.insert(result.platform, result.clone());

            // Collect issues for overall analysis
            if !result.issues.is_empty() {
                overall_issues.extend(result.issues);
            }
        }

        // Calculate overall VIBE score (weighted average with confidence intervals)
        let overall_score =
            self.calculate_overall_score(&score_values, &config.validation_criteria)?;

        // Generate confidence interval based on platform consensus
        let confidence_interval = Some(self.calculate_confidence_interval(&score_values)?);

        // Determine overall validation status
        let status = if overall_score >= config.minimum_score {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        };

        // Generate recommendations based on validation results
        let recommendations = self.generate_recommendations(&detailed_results, overall_score)?;

        // Create validation result
        let result = ValidationResult {
            overall_score,
            platform_scores,
            confidence_interval,
            status,
            detailed_results,
            validation_time_ms: validation_duration.as_millis() as u64,
            issues: overall_issues,
            recommendations,
            timestamp: chrono::Utc::now(),
            protocol_id: self.generate_protocol_id("aggregated"),
        };

        Ok(result)
    }

    /// Calculate overall VIBE score using weighted aggregation
    fn calculate_overall_score(
        &self,
        scores: &[f32],
        criteria: &ValidationCriteria,
    ) -> Result<f32> {
        if scores.is_empty() {
            return Err(VIBEError::NoValidations(
                "No platform scores available".to_string(),
            ));
        }

        // Base weights for different platforms
        let platform_weights = match scores.len() {
            1 => vec![1.0],
            2 => vec![0.4, 0.6],
            3 => vec![0.3, 0.4, 0.3],
            4 => vec![0.25, 0.25, 0.25, 0.25],
            _ => vec![0.2, 0.2, 0.2, 0.2, 0.2], // 5 platforms
        };

        // Calculate weighted average
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, &score) in scores.iter().enumerate() {
            if i < platform_weights.len() {
                let weight = platform_weights[i];
                weighted_sum += score * weight;
                total_weight += weight;
            }
        }

        let base_score = weighted_sum / total_weight;

        // Apply criteria adjustments
        let mut adjusted_score = base_score;

        // Adjust for logical consistency requirement
        if criteria.logical_consistency && base_score < 70.0 {
            adjusted_score *= 0.9; // Penalty for poor logical consistency
        }

        // Adjust for practical applicability requirement
        if criteria.practical_applicability && base_score < 60.0 {
            adjusted_score *= 0.85; // Strong penalty for impractical protocols
        }

        // Adjust for platform compatibility
        if criteria.platform_compatibility && scores.len() < 3 {
            adjusted_score *= 0.95; // Slight penalty for limited platform coverage
        }

        // Ensure score stays within valid range
        Ok(adjusted_score.clamp(0.0, 100.0))
    }

    /// Calculate confidence interval based on platform score variance
    fn calculate_confidence_interval(&self, platform_scores: &[f32]) -> Result<ConfidenceInterval> {
        if platform_scores.is_empty() {
            return Err(VIBEError::NoValidations(
                "No platform scores for confidence calculation".to_string(),
            ));
        }

        let mean: f32 = platform_scores.iter().sum::<f32>() / platform_scores.len() as f32;
        let variance: f32 = platform_scores
            .iter()
            .map(|&score| (score - mean).powi(2))
            .sum::<f32>()
            / platform_scores.len() as f32;
        let std_dev = variance.sqrt();

        // 95% confidence interval
        let margin_of_error = 1.96 * std_dev / (platform_scores.len() as f32).sqrt();

        Ok(ConfidenceInterval {
            lower: (mean - margin_of_error).max(0.0),
            upper: (mean + margin_of_error).min(100.0),
            confidence_level: 0.95,
            sample_size: platform_scores.len(),
        })
    }

    /// Generate actionable recommendations based on validation results
    fn generate_recommendations(
        &self,
        detailed_results: &HashMap<Platform, PlatformValidationResult>,
        overall_score: f32,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Overall performance recommendations
        if overall_score < 60.0 {
            recommendations.push(Recommendation {
                priority: Priority::High,
                category: RecommendationCategory::Overall,
                title: "Major Protocol Revision Needed".to_string(),
                description: "Protocol requires significant improvements across multiple areas."
                    .to_string(),
                actionable_steps: vec![
                    "Review protocol logic flow".to_string(),
                    "Strengthen validation criteria".to_string(),
                    "Improve error handling".to_string(),
                ],
            });
        } else if overall_score < 80.0 {
            recommendations.push(Recommendation {
                priority: Priority::Medium,
                category: RecommendationCategory::Overall,
                title: "Protocol Enhancement Recommended".to_string(),
                description: "Protocol is functional but could benefit from targeted improvements."
                    .to_string(),
                actionable_steps: vec![
                    "Optimize performance-critical sections".to_string(),
                    "Add comprehensive error handling".to_string(),
                    "Enhance documentation".to_string(),
                ],
            });
        }

        // Platform-specific recommendations
        for (platform, result) in detailed_results {
            if result.score < 60.0 {
                recommendations.push(Recommendation {
                    priority: Priority::High,
                    category: RecommendationCategory::PlatformSpecific,
                    title: format!("{} Platform Issues Detected", platform),
                    description: format!(
                        "Protocol scored {} on {} platform with {} issues detected.",
                        result.score,
                        platform,
                        result.issues.len()
                    ),
                    actionable_steps: self
                        .generate_platform_specific_steps(platform, &result.issues)?,
                });
            }
        }

        // Performance recommendations
        let avg_response_time = detailed_results
            .values()
            .map(|r| r.performance_metrics.average_response_time_ms)
            .sum::<u64>()
            / detailed_results.len() as u64;

        if avg_response_time > 2000 {
            recommendations.push(Recommendation {
                priority: Priority::Medium,
                category: RecommendationCategory::Performance,
                title: "Performance Optimization Needed".to_string(),
                description: format!(
                    "Average response time ({})ms exceeds recommended threshold (2000ms).",
                    avg_response_time
                ),
                actionable_steps: vec![
                    "Optimize algorithm complexity".to_string(),
                    "Implement caching strategies".to_string(),
                    "Reduce network requests".to_string(),
                ],
            });
        }

        Ok(recommendations)
    }

    /// Generate platform-specific improvement steps
    fn generate_platform_specific_steps(
        &self,
        platform: &Platform,
        issues: &[ValidationIssue],
    ) -> Result<Vec<String>> {
        let mut steps = Vec::new();

        match platform {
            Platform::Web => {
                steps.push("Optimize UI/UX design patterns".to_string());
                steps.push("Implement responsive design".to_string());
                steps.push("Add accessibility features".to_string());
            }
            Platform::Simulation => {
                steps.push("Review logic flow completeness".to_string());
                steps.push("Validate state management".to_string());
                steps.push("Test edge cases".to_string());
            }
            Platform::Android => {
                steps.push("Optimize for Android Material Design".to_string());
                steps.push("Implement touch gesture handling".to_string());
                steps.push("Test on multiple screen densities".to_string());
            }
            Platform::IOS => {
                steps.push("Follow iOS Human Interface Guidelines".to_string());
                steps.push("Implement native iOS patterns".to_string());
                steps.push("Optimize for different iOS versions".to_string());
            }
            Platform::Backend => {
                steps.push("Strengthen API security".to_string());
                steps.push("Implement proper error handling".to_string());
                steps.push("Optimize database queries".to_string());
            }
        }

        // Add issue-specific steps
        for issue in issues {
            match issue.severity {
                Severity::Critical => steps.push(format!(
                    "URGENT: Address critical issue - {}",
                    issue.description
                )),
                Severity::High => steps.push(format!(
                    "HIGH: Fix high-priority issue - {}",
                    issue.description
                )),
                Severity::Medium => steps.push(format!(
                    "MEDIUM: Address medium-priority issue - {}",
                    issue.description
                )),
                Severity::Low => {
                    steps.push(format!("LOW: Consider improving - {}", issue.description))
                }
            }
        }

        Ok(steps)
    }

    /// Update internal statistics after validation
    async fn update_statistics(&self, result: &ValidationResult) -> Result<()> {
        let mut state = self.state.write().await;

        // Update protocol count
        state.protocols_validated += 1;

        // Update platform statistics
        for platform in result.detailed_results.keys() {
            *state.platform_stats.entry(*platform).or_insert(0) += 1;
        }

        // Add to recent results (keep last 100)
        state.recent_results.push_back(result.clone());
        if state.recent_results.len() > 100 {
            state.recent_results.pop_front();
        }

        // Update health metrics
        let success = result.status == ValidationStatus::Passed;
        let current_success_rate = state.health_metrics.validation_success_rate;
        let _total_validations = state.protocols_validated as f32;

        // Calculate new success rate with exponential moving average
        let alpha = 0.1; // Smoothing factor
        let new_success_rate = if success {
            current_success_rate * (1.0 - alpha) + alpha * 1.0
        } else {
            current_success_rate * (1.0 - alpha) + alpha * 0.0
        };

        state.health_metrics.validation_success_rate = new_success_rate;

        // Update performance metrics
        let mut metrics = self.metrics.write().await;
        let validation_time = result.validation_time_ms;

        // Update timing metrics
        if metrics.fastest_validation_ms == 0 || validation_time < metrics.fastest_validation_ms {
            metrics.fastest_validation_ms = validation_time;
        }
        if validation_time > metrics.slowest_validation_ms {
            metrics.slowest_validation_ms = validation_time;
        }

        // Update average (exponential moving average)
        let alpha = 0.2;
        if metrics.average_validation_time_ms == 0 {
            metrics.average_validation_time_ms = validation_time;
        } else {
            metrics.average_validation_time_ms =
                (metrics.average_validation_time_ms as f32 * (1.0 - alpha)
                    + validation_time as f32 * alpha) as u64;
        }

        Ok(())
    }

    /// Cache validation result for performance
    async fn cache_validation_result(
        &self,
        protocol_id: &Uuid,
        result: &ValidationResult,
    ) -> Result<()> {
        let mut cache = self.cache.write().await;

        let cached = CachedValidation {
            result: result.clone(),
            timestamp: chrono::Utc::now(),
            ttl_seconds: 3600, // 1 hour TTL
        };

        cache.results.insert(*protocol_id, cached);

        // Clean expired entries
        let now = chrono::Utc::now();
        cache.results.retain(|_, cached_result| {
            now.signed_duration_since(cached_result.timestamp)
                .num_seconds()
                < cached_result.ttl_seconds as i64
        });

        Ok(())
    }

    /// Retrieve cached validation result
    async fn get_cached_validation(&self, protocol_id: &Uuid) -> Result<Option<ValidationResult>> {
        let cache = self.cache.read().await;

        if let Some(cached) = cache.results.get(protocol_id) {
            let now = chrono::Utc::now();
            let age_seconds = now.signed_duration_since(cached.timestamp).num_seconds();

            if age_seconds < cached.ttl_seconds as i64 {
                // Update cache statistics
                let mut cache_mut = self.cache.write().await;
                cache_mut.cache_hits += 1;
                cache_mut.total_requests += 1;
                cache_mut.hit_rate_percent =
                    (cache_mut.cache_hits as f32 / cache_mut.total_requests as f32) * 100.0;

                return Ok(Some(cached.result.clone()));
            }
        }

        Ok(None)
    }

    /// Generate protocol ID for caching and tracking
    fn generate_protocol_id(&self, content: &str) -> Uuid {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        Uuid::from_u128(hasher.finish() as u128)
    }

    /// Get current engine statistics
    pub async fn get_statistics(&self) -> Result<VIBEStats> {
        let state = self.state.read().await;
        let _metrics = self.metrics.read().await;
        let _cache = self.cache.read().await;

        // Calculate validation trends from recent results
        let validation_trends: Vec<ScoreTrend> = state
            .recent_results
            .iter()
            .map(|result| {
                // Determine platform distribution from detailed results
                let platform = if !result.detailed_results.is_empty() {
                    result
                        .detailed_results
                        .keys()
                        .next()
                        .copied()
                        .unwrap_or(Platform::Web)
                } else {
                    Platform::Web
                };

                ScoreTrend {
                    timestamp: result.timestamp,
                    score: result.overall_score,
                    platform,
                }
            })
            .collect();

        Ok(VIBEStats {
            total_validations: state.protocols_validated,
            average_score: if !validation_trends.is_empty() {
                validation_trends.iter().map(|t| t.score).sum::<f32>()
                    / validation_trends.len() as f32
            } else {
                0.0
            },
            success_rate: state.health_metrics.validation_success_rate,
            platform_distribution: state
                .platform_stats
                .iter()
                .map(|(platform, count)| (*platform, *count as u32))
                .collect(),
            validation_trends,
        })
    }

    /// Add custom platform validator
    pub fn add_platform_validator(
        &mut self,
        platform: Platform,
        validator: Arc<dyn PlatformValidator>,
    ) {
        self.adapters.insert(platform, validator);
    }

    /// Remove platform validator
    pub fn remove_platform_validator(
        &mut self,
        platform: &Platform,
    ) -> Option<Arc<dyn PlatformValidator>> {
        self.adapters.remove(platform)
    }

    /// Clear validation cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.results.clear();
        cache.platform_data.clear();
        Ok(())
    }
}

impl Default for VIBEEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// VIBE-specific errors
/// Result type for VIBE operations.
pub type Result<T> = std::result::Result<T, VIBEError>;

#[derive(Debug, thiserror::Error)]
pub enum VIBEError {
    #[error("Invalid protocol: {0}")]
    InvalidProtocol(String),

    #[error("Validation failed: {0}")]
    ValidationError(String),

    #[error("No validations performed: {0}")]
    NoValidations(String),

    #[error("Platform error: {0}")]
    PlatformError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Performance error: {0}")]
    PerformanceError(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Adapter error: {0}")]
    AdapterError(String),

    #[error("Benchmark error: {0}")]
    BenchmarkError(String),

    #[error("Scoring error: {0}")]
    ScoringError(String),

    #[error("Proof ledger error: {0}")]
    ProofLedgerError(#[from] crate::verification::proof_ledger::ProofLedgerError),

    #[error("ReasonKit error: {0}")]
    ReasonKitError(#[from] crate::error::Error),
}

/// Validation status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    Passed,
    Failed,
    Warning,
    Pending,
}

/// Confidence interval for validation scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: f32,
    pub upper: f32,
    pub confidence_level: f32,
    pub sample_size: usize,
}

/// Recommendation for protocol improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub priority: Priority,
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub actionable_steps: Vec<String>,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Recommendation categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationCategory {
    Overall,
    PlatformSpecific,
    Performance,
    Security,
    UserExperience,
    Logic,
}

/// Validation issue detected during platform validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub platform: Platform,
    pub severity: Severity,
    pub category: IssueCategory,
    pub description: String,
    pub location: Option<String>,
    pub suggestion: Option<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Issue categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueCategory {
    LogicError,
    PerformanceIssue,
    SecurityVulnerability,
    UIUXIssue,
    CompatibilityProblem,
    ResourceUsage,
    ErrorHandling,
    Documentation,
}

/// Result from individual platform validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformValidationResult {
    pub platform: Platform,
    pub score: f32,
    pub status: ValidationStatus,
    pub issues: Vec<ValidationIssue>,
    pub performance_metrics: PlatformPerformanceMetrics,
    pub recommendations: Vec<String>,
}

/// Platform-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformPerformanceMetrics {
    pub average_response_time_ms: u64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f32,
    pub error_rate_percent: f32,
    pub throughput_requests_per_second: f32,
}

use std::collections::VecDeque;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vibe_engine_creation() {
        let engine = VIBEEngine::new();
        let stats = engine.get_statistics().await.unwrap();
        assert_eq!(stats.total_validations, 0);
    }

    #[test]
    fn test_protocol_id_generation() {
        let engine = VIBEEngine::new();
        let id1 = engine.generate_protocol_id("test protocol");
        let id2 = engine.generate_protocol_id("test protocol");
        assert_eq!(id1, id2); // Same content should generate same ID

        let id3 = engine.generate_protocol_id("different protocol");
        assert_ne!(id1, id3); // Different content should generate different ID
    }

    #[test]
    fn test_overall_score_calculation() {
        let engine = VIBEEngine::new();
        let criteria = ValidationCriteria {
            logical_consistency: true,
            practical_applicability: true,
            platform_compatibility: true,
            performance_requirements: false,
            security_considerations: false,
            user_experience: false,
            custom_metrics: HashMap::new(),
        };

        let scores = vec![80.0, 75.0, 90.0];
        let score = engine.calculate_overall_score(&scores, &criteria).unwrap();

        assert!(score >= 0.0);
        assert!(score <= 100.0);
        assert!(score < 90.0); // Should be adjusted by criteria
    }
}
