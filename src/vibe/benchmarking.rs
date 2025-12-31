//! # VIBE Benchmarking System
//!
//! Comprehensive benchmarking suite for VIBE protocol validation performance
//! and quality assessment across different platforms and scenarios.

use super::*;
use crate::vibe::scoring::TrendDirection;
use crate::vibe::validation::{Severity, VIBEError};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Benchmark suite for VIBE validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    /// Suite identifier
    pub suite_id: Uuid,

    /// Suite name and description
    pub name: String,
    pub description: String,

    /// Benchmark scenarios
    pub scenarios: Vec<BenchmarkScenario>,

    /// Suite configuration
    pub config: BenchmarkConfig,

    /// Historical results
    pub results: Vec<BenchmarkResult>,
}

/// Individual benchmark scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScenario {
    /// Scenario identifier
    pub scenario_id: Uuid,

    /// Scenario metadata
    pub name: String,
    pub description: String,
    pub category: BenchmarkCategory,

    /// Protocol to benchmark
    pub protocol: BenchmarkProtocol,

    /// Target platforms for validation
    pub target_platforms: Vec<Platform>,

    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,

    /// Expected outcomes
    pub expected_outcomes: ExpectedOutcomes,
}

/// Protocol specifically designed for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkProtocol {
    /// Protocol content
    pub content: String,

    /// Protocol type
    pub protocol_type: ProtocolType,

    /// Complexity level
    pub complexity: ProtocolComplexity,

    /// Known characteristics
    pub characteristics: ProtocolCharacteristics,
}

/// Complexity levels for benchmark protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolComplexity {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Protocol characteristics for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolCharacteristics {
    pub has_multiple_platforms: bool,
    pub has_security_requirements: bool,
    pub has_performance_requirements: bool,
    pub has_accessibility_requirements: bool,
    pub has_integration_requirements: bool,
    pub estimated_validation_time_ms: u64,
}

/// Benchmark categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkCategory {
    Performance,
    Accuracy,
    CrossPlatform,
    Regression,
    Stress,
}

/// Performance thresholds for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub max_validation_time_ms: u64,
    pub max_memory_usage_mb: u64,
    pub min_score_threshold: f32,
    pub max_error_rate_percent: f32,
}

/// Expected outcomes for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcomes {
    pub expected_score_range: (f32, f32),
    pub expected_issues_count: (usize, usize),
    pub expected_platform_scores: HashMap<Platform, f32>,
    pub required_validations: Vec<Platform>,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of iterations for each scenario
    pub iterations: usize,

    /// Parallel execution settings
    pub parallel_execution: bool,
    pub max_concurrent_validations: usize,

    /// Warm-up iterations (discarded from results)
    pub warmup_iterations: usize,

    /// Statistical confidence level
    pub confidence_level: f32,

    /// Enable detailed profiling
    pub enable_profiling: bool,

    /// Custom scoring criteria
    pub scoring_criteria: Option<ValidationCriteria>,
}

/// Benchmark execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Result identifier
    pub result_id: Uuid,

    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Scenario that was executed
    pub scenario_id: Uuid,

    /// Platform results
    pub platform_results: HashMap<Platform, PlatformBenchmarkResult>,

    /// Overall metrics
    pub overall_metrics: OverallBenchmarkMetrics,

    /// Statistical analysis
    pub statistics: BenchmarkStatistics,

    /// Pass/fail status
    pub passed: bool,
    pub failure_reason: Option<String>,
}

/// Platform-specific benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformBenchmarkResult {
    /// Platform validation score
    pub score: f32,

    /// Validation time
    pub validation_time_ms: u64,

    /// Memory usage
    pub memory_usage_mb: u64,

    /// CPU usage
    pub cpu_usage_percent: f32,

    /// Issues found
    pub issues_count: usize,

    /// Recommendations count
    pub recommendations_count: usize,
}

/// Overall benchmark metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallBenchmarkMetrics {
    pub total_validation_time_ms: u64,
    pub average_score: f32,
    pub score_variance: f32,
    pub total_issues_found: usize,
    pub platforms_passed: usize,
    pub platforms_failed: usize,
}

/// Benchmark statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
    pub mean_validation_time_ms: f32,
    pub std_dev_validation_time_ms: f32,
    pub min_validation_time_ms: u64,
    pub max_validation_time_ms: u64,
    pub percentile_95_ms: u64,
    pub percentile_99_ms: u64,
    pub throughput_validations_per_second: f32,
}

/// Performance metrics for the entire benchmarking system
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total benchmarks executed
    pub total_benchmarks: u64,

    /// Average benchmark duration
    pub average_duration_ms: f32,

    /// Fastest benchmark
    pub fastest_benchmark_ms: u64,

    /// Slowest benchmark
    pub slowest_benchmark_ms: u64,

    /// Success rate
    pub success_rate_percent: f32,

    /// Platform performance distribution
    pub platform_distribution: HashMap<Platform, f32>,

    /// Error patterns
    pub error_patterns: HashMap<String, u32>,

    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
}

/// Benchmark execution engine
pub struct BenchmarkEngine {
    /// VIBE engine for validation
    vibe_engine: super::validation::VIBEEngine,

    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,

    /// Historical benchmark data
    benchmark_history: Arc<RwLock<BenchmarkHistory>>,
}

/// Historical benchmark data for trend analysis
#[derive(Debug, Default)]
pub struct BenchmarkHistory {
    pub results: Vec<BenchmarkResult>,
    pub trend_analysis: Option<BenchmarkTrendAnalysis>,
}

/// Benchmark trend analysis
#[derive(Debug, Clone)]
pub struct BenchmarkTrendAnalysis {
    pub performance_trend: PerformanceTrend,
    pub accuracy_trend: AccuracyTrend,
    pub regression_detection: RegressionDetection,
}

/// Performance trend over time
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub direction: TrendDirection,
    pub slope: f32,
    pub correlation: f32,
}

/// Accuracy trend analysis
#[derive(Debug, Clone)]
pub struct AccuracyTrend {
    pub accuracy_improvement: f32,
    pub false_positive_rate: f32,
    pub false_negative_rate: f32,
}

/// Regression detection results
#[derive(Debug, Clone)]
pub struct RegressionDetection {
    pub regressions_detected: u32,
    pub regression_threshold: f32,
    pub regression_details: Vec<RegressionDetail>,
}

/// Individual regression detail
#[derive(Debug, Clone)]
pub struct RegressionDetail {
    pub scenario_id: Uuid,
    pub platform: Platform,
    pub regression_severity: Severity,
    pub performance_drop_percent: f32,
}

impl BenchmarkEngine {
    /// Create new benchmark engine
    pub fn new(vibe_engine: super::validation::VIBEEngine) -> Self {
        Self {
            vibe_engine,
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            benchmark_history: Arc::new(RwLock::new(BenchmarkHistory::default())),
        }
    }

    /// Execute a complete benchmark suite
    pub async fn execute_suite(
        &self,
        suite: &BenchmarkSuite,
        config: &ValidationConfig,
    ) -> Result<BenchmarkResult, VIBEError> {
        let start_time = Instant::now();

        // Validate suite configuration
        self.validate_suite_config(suite)?;

        // Execute warm-up iterations
        if suite.config.warmup_iterations > 0 {
            self.execute_warmup(suite, config).await?;
        }

        // Execute benchmark iterations
        let mut all_results = Vec::new();

        for iteration in 0..suite.config.iterations {
            tracing::info!(
                "Executing benchmark iteration {}/{}",
                iteration + 1,
                suite.config.iterations
            );

            let iteration_results = self.execute_iteration(suite, config).await?;
            all_results.extend(iteration_results);
        }

        // Aggregate results
        let result = self.aggregate_benchmark_results(suite, all_results, start_time)?;

        // Update metrics and history
        self.update_benchmark_metrics(&result).await?;
        self.store_benchmark_result(&result).await?;

        // Perform regression analysis
        if let Some(regression) = self.detect_regressions(&result).await? {
            tracing::warn!("Regression detected: {:?}", regression);
        }

        Ok(result)
    }

    /// Execute a single benchmark scenario
    pub async fn execute_scenario(
        &self,
        scenario: &BenchmarkScenario,
        config: &ValidationConfig,
    ) -> Result<BenchmarkResult, VIBEError> {
        let _start_time = Instant::now();

        // Validate scenario
        self.validate_scenario(scenario)?;

        // Execute validation for target platforms
        let mut platform_results = HashMap::new();

        for platform in &scenario.target_platforms {
            let platform_start = Instant::now();

            // Create platform-specific validation config
            let mut platform_config = config.clone();
            platform_config.target_platforms = vec![*platform];

            // Execute validation
            let validation_result = self
                .vibe_engine
                .validate_protocol(&scenario.protocol.content, platform_config)
                .await?;

            let validation_time = platform_start.elapsed();

            platform_results.insert(
                *platform,
                PlatformBenchmarkResult {
                    score: validation_result.overall_score,
                    validation_time_ms: validation_time.as_millis() as u64,
                    memory_usage_mb: self.estimate_memory_usage(&validation_result),
                    cpu_usage_percent: self.estimate_cpu_usage(&validation_result),
                    issues_count: validation_result.issues.len(),
                    recommendations_count: validation_result.recommendations.len(),
                },
            );
        }

        // Calculate overall metrics
        let overall_metrics = self.calculate_overall_metrics(&platform_results)?;

        // Calculate statistics
        let statistics = self.calculate_statistics(&platform_results)?;

        // Determine pass/fail status
        let (passed, failure_reason) = self.evaluate_scenario_outcome(
            &scenario.expected_outcomes,
            &overall_metrics,
            &platform_results,
        )?;

        let result = BenchmarkResult {
            result_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            scenario_id: scenario.scenario_id,
            platform_results,
            overall_metrics,
            statistics,
            passed,
            failure_reason,
        };

        Ok(result)
    }

    /// Execute warm-up iterations to stabilize performance
    async fn execute_warmup(
        &self,
        suite: &BenchmarkSuite,
        config: &ValidationConfig,
    ) -> Result<(), VIBEError> {
        for _ in 0..suite.config.warmup_iterations {
            for scenario in &suite.scenarios {
                // Quick validation without storing results
                let mut warmup_config = config.clone();
                warmup_config.target_platforms = scenario.target_platforms.clone();

                let _ = self
                    .vibe_engine
                    .validate_protocol(&scenario.protocol.content, warmup_config)
                    .await;
            }
        }

        Ok(())
    }

    /// Execute a single iteration of all scenarios
    async fn execute_iteration(
        &self,
        suite: &BenchmarkSuite,
        config: &ValidationConfig,
    ) -> Result<Vec<BenchmarkResult>, VIBEError> {
        let mut results = Vec::new();

        for scenario in &suite.scenarios {
            let result = self.execute_scenario(scenario, config).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Aggregate multiple benchmark results
    fn aggregate_benchmark_results(
        &self,
        suite: &BenchmarkSuite,
        all_results: Vec<BenchmarkResult>,
        start_time: Instant,
    ) -> Result<BenchmarkResult, VIBEError> {
        let total_time = start_time.elapsed();

        // Combine all platform results
        let mut combined_platform_results = HashMap::new();
        let mut all_overall_metrics = Vec::new();

        for result in &all_results {
            for (platform, platform_result) in &result.platform_results {
                combined_platform_results
                    .entry(*platform)
                    .or_insert_with(Vec::new)
                    .push(platform_result.clone());
            }
            all_overall_metrics.push(result.overall_metrics.clone());
        }

        // Calculate average metrics across all iterations
        let average_score = all_results
            .iter()
            .map(|r| r.overall_metrics.average_score)
            .sum::<f32>()
            / all_results.len() as f32;

        let total_issues = all_results
            .iter()
            .map(|r| r.overall_metrics.total_issues_found)
            .sum::<usize>();

        let passed_scenarios = all_results.iter().filter(|r| r.passed).count();
        let failed_scenarios = all_results.len() - passed_scenarios;

        // Combine statistics
        let combined_statistics = self.combine_statistics(&all_results)?;

        // Determine overall pass/fail
        let overall_passed = if suite.scenarios.is_empty() {
            false
        } else {
            (passed_scenarios as f32 / suite.scenarios.len() as f32) >= 0.8 // 80% pass rate
        };

        let overall_failure_reason = if !overall_passed {
            Some(format!(
                "Only {}/{} scenarios passed",
                passed_scenarios,
                suite.scenarios.len()
            ))
        } else {
            None
        };

        Ok(BenchmarkResult {
            result_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            scenario_id: suite.suite_id, // Use suite ID for aggregated result
            platform_results: combined_platform_results
                .into_iter()
                .map(|(platform, results)| {
                    let avg_score =
                        results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;
                    let avg_time = results.iter().map(|r| r.validation_time_ms).sum::<u64>()
                        / results.len() as u64;

                    (
                        platform,
                        PlatformBenchmarkResult {
                            score: avg_score,
                            validation_time_ms: avg_time,
                            memory_usage_mb: results.iter().map(|r| r.memory_usage_mb).sum::<u64>()
                                / results.len() as u64,
                            cpu_usage_percent: results
                                .iter()
                                .map(|r| r.cpu_usage_percent)
                                .sum::<f32>()
                                / results.len() as f32,
                            issues_count: results.iter().map(|r| r.issues_count).sum::<usize>()
                                / results.len(),
                            recommendations_count: results
                                .iter()
                                .map(|r| r.recommendations_count)
                                .sum::<usize>()
                                / results.len(),
                        },
                    )
                })
                .collect(),
            overall_metrics: OverallBenchmarkMetrics {
                total_validation_time_ms: total_time.as_millis() as u64,
                average_score,
                score_variance: self.calculate_score_variance(&all_results),
                total_issues_found: total_issues / all_results.len(),
                platforms_passed: passed_scenarios,
                platforms_failed: failed_scenarios,
            },
            statistics: combined_statistics,
            passed: overall_passed,
            failure_reason: overall_failure_reason,
        })
    }

    /// Validate suite configuration
    fn validate_suite_config(&self, suite: &BenchmarkSuite) -> Result<(), VIBEError> {
        if suite.scenarios.is_empty() {
            return Err(VIBEError::BenchmarkError(
                "Benchmark suite has no scenarios".to_string(),
            ));
        }

        if suite.config.iterations == 0 {
            return Err(VIBEError::BenchmarkError(
                "Benchmark iterations must be greater than 0".to_string(),
            ));
        }

        if suite.config.warmup_iterations >= suite.config.iterations {
            return Err(VIBEError::BenchmarkError(
                "Warm-up iterations must be less than total iterations".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate individual scenario
    fn validate_scenario(&self, scenario: &BenchmarkScenario) -> Result<(), VIBEError> {
        if scenario.protocol.content.trim().is_empty() {
            return Err(VIBEError::BenchmarkError(
                "Scenario protocol content is empty".to_string(),
            ));
        }

        if scenario.target_platforms.is_empty() {
            return Err(VIBEError::BenchmarkError(
                "Scenario has no target platforms".to_string(),
            ));
        }

        // Validate expected outcomes consistency
        for platform in &scenario.expected_outcomes.required_validations {
            if !scenario.target_platforms.contains(platform) {
                return Err(VIBEError::BenchmarkError(format!(
                    "Platform {:?} required in outcomes but not in target platforms",
                    platform
                )));
            }
        }

        Ok(())
    }

    /// Calculate overall metrics from platform results
    fn calculate_overall_metrics(
        &self,
        platform_results: &HashMap<Platform, PlatformBenchmarkResult>,
    ) -> Result<OverallBenchmarkMetrics, VIBEError> {
        let scores: Vec<f32> = platform_results.values().map(|r| r.score).collect();
        let average_score = scores.iter().sum::<f32>() / scores.len() as f32;

        let score_variance = self.calculate_variance(&scores);

        let total_issues = platform_results.values().map(|r| r.issues_count).sum();

        let validation_times: Vec<u64> = platform_results
            .values()
            .map(|r| r.validation_time_ms)
            .collect();
        let total_time = validation_times.iter().sum::<u64>();

        let passed_platforms = platform_results
            .values()
            .filter(|r| r.score >= 70.0)
            .count();

        Ok(OverallBenchmarkMetrics {
            total_validation_time_ms: total_time,
            average_score,
            score_variance,
            total_issues_found: total_issues,
            platforms_passed: passed_platforms,
            platforms_failed: platform_results.len() - passed_platforms,
        })
    }

    /// Calculate statistics from platform results
    fn calculate_statistics(
        &self,
        platform_results: &HashMap<Platform, PlatformBenchmarkResult>,
    ) -> Result<BenchmarkStatistics, VIBEError> {
        let times: Vec<u64> = platform_results
            .values()
            .map(|r| r.validation_time_ms)
            .collect();

        if times.is_empty() {
            return Err(VIBEError::BenchmarkError(
                "No platform results for statistics".to_string(),
            ));
        }

        let mean_time = times.iter().sum::<u64>() as f32 / times.len() as f32;
        let variance =
            self.calculate_variance(&times.iter().map(|&t| t as f32).collect::<Vec<_>>());
        let std_dev = variance.sqrt();

        let mut sorted_times = times.clone();
        sorted_times.sort();

        let min_time = sorted_times.first().copied().unwrap_or(0);
        let max_time = sorted_times.last().copied().unwrap_or(0);

        let percentile_95 = self.calculate_percentile(&sorted_times, 95.0);
        let percentile_99 = self.calculate_percentile(&sorted_times, 99.0);

        let throughput = if mean_time > 0.0 {
            1000.0 / mean_time
        } else {
            0.0
        };

        Ok(BenchmarkStatistics {
            mean_validation_time_ms: mean_time,
            std_dev_validation_time_ms: std_dev,
            min_validation_time_ms: min_time,
            max_validation_time_ms: max_time,
            percentile_95_ms: percentile_95,
            percentile_99_ms: percentile_99,
            throughput_validations_per_second: throughput,
        })
    }

    /// Combine statistics from multiple results
    fn combine_statistics(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<BenchmarkStatistics, VIBEError> {
        if results.is_empty() {
            return Err(VIBEError::BenchmarkError(
                "No results to combine".to_string(),
            ));
        }

        // Combine all validation times
        let all_times: Vec<f32> = results
            .iter()
            .flat_map(|r| {
                r.platform_results
                    .values()
                    .map(|pr| pr.validation_time_ms as f32)
                    .collect::<Vec<_>>()
            })
            .collect();

        if all_times.is_empty() {
            return Err(VIBEError::BenchmarkError(
                "No validation times found".to_string(),
            ));
        }

        let mean_time = all_times.iter().sum::<f32>() / all_times.len() as f32;
        let variance = self.calculate_variance(&all_times);
        let std_dev = variance.sqrt();

        let min_time = all_times.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as u64;
        let max_time = all_times.iter().fold(0.0f32, |a, &b| a.max(b)) as u64;

        let mut sorted_times = all_times.iter().map(|&t| t as u64).collect::<Vec<_>>();
        sorted_times.sort();

        let percentile_95 = self.calculate_percentile(&sorted_times, 95.0);
        let percentile_99 = self.calculate_percentile(&sorted_times, 99.0);

        let throughput = if mean_time > 0.0 {
            1000.0 / mean_time
        } else {
            0.0
        };

        Ok(BenchmarkStatistics {
            mean_validation_time_ms: mean_time,
            std_dev_validation_time_ms: std_dev,
            min_validation_time_ms: min_time,
            max_validation_time_ms: max_time,
            percentile_95_ms: percentile_95,
            percentile_99_ms: percentile_99,
            throughput_validations_per_second: throughput,
        })
    }

    /// Evaluate if scenario meets expected outcomes
    fn evaluate_scenario_outcome(
        &self,
        expected: &ExpectedOutcomes,
        metrics: &OverallBenchmarkMetrics,
        platform_results: &HashMap<Platform, PlatformBenchmarkResult>,
    ) -> Result<(bool, Option<String>), VIBEError> {
        let (min_expected, max_expected) = expected.expected_score_range;
        let (min_issues, max_issues) = expected.expected_issues_count;

        // Check score range
        if metrics.average_score < min_expected || metrics.average_score > max_expected {
            return Ok((
                false,
                Some(format!(
                    "Score {:.1} outside expected range {:.1}-{:.1}",
                    metrics.average_score, min_expected, max_expected
                )),
            ));
        }

        // Check issues count
        if metrics.total_issues_found < min_issues || metrics.total_issues_found > max_issues {
            return Ok((
                false,
                Some(format!(
                    "Issues count {} outside expected range {}-{}",
                    metrics.total_issues_found, min_issues, max_issues
                )),
            ));
        }

        // Check platform scores
        for (platform, expected_score) in &expected.expected_platform_scores {
            if let Some(result) = platform_results.get(platform) {
                let score_diff = (result.score - expected_score).abs();
                if score_diff > 10.0 {
                    // 10 point tolerance
                    return Ok((
                        false,
                        Some(format!(
                            "Platform {:?} score {:.1} differs from expected {:.1}",
                            platform, result.score, expected_score
                        )),
                    ));
                }
            }
        }

        // Check required validations
        for platform in &expected.required_validations {
            if !platform_results.contains_key(platform) {
                return Ok((
                    false,
                    Some(format!(
                        "Required platform {:?} validation missing",
                        platform
                    )),
                ));
            }
        }

        Ok((true, None))
    }

    /// Calculate variance for a set of values
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        values
            .iter()
            .map(|&value| (value - mean).powi(2))
            .sum::<f32>()
            / (values.len() - 1) as f32
    }

    /// Calculate score variance across results
    fn calculate_score_variance(&self, results: &[BenchmarkResult]) -> f32 {
        let scores: Vec<f32> = results
            .iter()
            .map(|r| r.overall_metrics.average_score)
            .collect();
        self.calculate_variance(&scores)
    }

    /// Calculate percentile from sorted data
    fn calculate_percentile(&self, sorted_data: &[u64], percentile: f32) -> u64 {
        if sorted_data.is_empty() {
            return 0;
        }

        let index = ((percentile / 100.0) * (sorted_data.len() - 1) as f32).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }

    /// Estimate memory usage from validation result
    fn estimate_memory_usage(&self, _result: &super::validation::ValidationResult) -> u64 {
        // Simplified memory estimation based on result complexity
        150 // Base memory usage in MB
    }

    /// Estimate CPU usage from validation result
    fn estimate_cpu_usage(&self, _result: &super::validation::ValidationResult) -> f32 {
        // Simplified CPU estimation
        30.0 // Base CPU usage percentage
    }

    /// Update benchmark metrics
    async fn update_benchmark_metrics(&self, result: &BenchmarkResult) -> Result<(), VIBEError> {
        let mut metrics = self.metrics.write().await;

        metrics.total_benchmarks += 1;

        // Update average duration
        let current_avg = metrics.average_duration_ms;
        let new_avg = if metrics.total_benchmarks == 1 {
            result.overall_metrics.total_validation_time_ms as f32
        } else {
            (current_avg * (metrics.total_benchmarks - 1) as f32
                + result.overall_metrics.total_validation_time_ms as f32)
                / metrics.total_benchmarks as f32
        };
        metrics.average_duration_ms = new_avg;

        // Update min/max times
        if metrics.fastest_benchmark_ms == 0
            || result.overall_metrics.total_validation_time_ms < metrics.fastest_benchmark_ms
        {
            metrics.fastest_benchmark_ms = result.overall_metrics.total_validation_time_ms;
        }

        if result.overall_metrics.total_validation_time_ms > metrics.slowest_benchmark_ms {
            metrics.slowest_benchmark_ms = result.overall_metrics.total_validation_time_ms;
        }

        // Update success rate
        let _success_count = if result.passed { 1 } else { 0 };
        let current_success_rate = metrics.success_rate_percent;
        let new_success_rate = if metrics.total_benchmarks == 1 {
            if result.passed {
                100.0
            } else {
                0.0
            }
        } else {
            (current_success_rate * (metrics.total_benchmarks - 1) as f32
                + if result.passed { 100.0 } else { 0.0 })
                / metrics.total_benchmarks as f32
        };
        metrics.success_rate_percent = new_success_rate;

        Ok(())
    }

    /// Store benchmark result in history
    async fn store_benchmark_result(&self, result: &BenchmarkResult) -> Result<(), VIBEError> {
        let mut history = self.benchmark_history.write().await;
        history.results.push(result.clone());

        // Keep only last 1000 results
        if history.results.len() > 1000 {
            let overflow = history.results.len() - 1000;
            history.results.drain(0..overflow);
        }

        Ok(())
    }

    /// Detect regressions in benchmark results
    async fn detect_regressions(
        &self,
        _current_result: &BenchmarkResult,
    ) -> Result<Option<RegressionDetection>, VIBEError> {
        // Simplified regression detection
        // In a real implementation, this would compare with historical baselines

        let history = self.benchmark_history.read().await;
        if history.results.len() < 10 {
            return Ok(None); // Need sufficient history
        }

        // Compare with recent average (last 5 results)
        let recent_results = &history.results[history.results.len().saturating_sub(5)..];
        let recent_avg_time: f32 = recent_results
            .iter()
            .map(|r| r.overall_metrics.total_validation_time_ms as f32)
            .sum::<f32>()
            / recent_results.len() as f32;

        let current_time = _current_result.overall_metrics.total_validation_time_ms as f32;

        // Check for significant regression (20% slower)
        if current_time > recent_avg_time * 1.2 {
            return Ok(Some(RegressionDetection {
                regressions_detected: 1,
                regression_threshold: 20.0,
                regression_details: vec![RegressionDetail {
                    scenario_id: _current_result.scenario_id,
                    platform: Platform::Web, // Simplified
                    regression_severity: Severity::High,
                    performance_drop_percent: ((current_time - recent_avg_time) / recent_avg_time)
                        * 100.0,
                }],
            }));
        }

        Ok(None)
    }
}

impl Default for BenchmarkEngine {
    fn default() -> Self {
        let vibe_engine = super::validation::VIBEEngine::new();
        Self::new(vibe_engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_benchmark_engine_creation() {
        let vibe_engine = super::validation::VIBEEngine::new();
        let engine = BenchmarkEngine::new(vibe_engine);
        assert!(engine.vibe_engine.get_statistics().await.is_ok());
    }

    #[test]
    fn test_scenario_validation() {
        let vibe_engine = super::validation::VIBEEngine::new();
        let engine = BenchmarkEngine::new(vibe_engine);

        let scenario = BenchmarkScenario {
            scenario_id: Uuid::new_v4(),
            name: "Test Scenario".to_string(),
            description: "A test benchmark scenario".to_string(),
            category: BenchmarkCategory::Performance,
            protocol: BenchmarkProtocol {
                content: "Test protocol content".to_string(),
                protocol_type: ProtocolType::ThinkToolChain,
                complexity: ProtocolComplexity::Simple,
                characteristics: ProtocolCharacteristics {
                    has_multiple_platforms: false,
                    has_security_requirements: false,
                    has_performance_requirements: false,
                    has_accessibility_requirements: false,
                    has_integration_requirements: false,
                    estimated_validation_time_ms: 1000,
                },
            },
            target_platforms: vec![Platform::Web],
            performance_thresholds: PerformanceThresholds {
                max_validation_time_ms: 5000,
                max_memory_usage_mb: 1000,
                min_score_threshold: 70.0,
                max_error_rate_percent: 5.0,
            },
            expected_outcomes: ExpectedOutcomes {
                expected_score_range: (60.0, 90.0),
                expected_issues_count: (0, 5),
                expected_platform_scores: HashMap::new(),
                required_validations: vec![Platform::Web],
            },
        };

        assert!(engine.validate_scenario(&scenario).is_ok());

        // Test empty protocol
        let mut invalid_scenario = scenario.clone();
        invalid_scenario.protocol.content = "".to_string();
        assert!(engine.validate_scenario(&invalid_scenario).is_err());
    }

    #[test]
    fn test_statistics_calculation() {
        let vibe_engine = super::validation::VIBEEngine::new();
        let engine = BenchmarkEngine::new(vibe_engine);

        let mut platform_results = HashMap::new();
        platform_results.insert(
            Platform::Web,
            PlatformBenchmarkResult {
                score: 80.0,
                validation_time_ms: 1000,
                memory_usage_mb: 100,
                cpu_usage_percent: 25.0,
                issues_count: 2,
                recommendations_count: 3,
            },
        );
        platform_results.insert(
            Platform::Backend,
            PlatformBenchmarkResult {
                score: 75.0,
                validation_time_ms: 1200,
                memory_usage_mb: 150,
                cpu_usage_percent: 30.0,
                issues_count: 3,
                recommendations_count: 2,
            },
        );

        let metrics = engine.calculate_overall_metrics(&platform_results).unwrap();
        assert!(metrics.average_score > 0.0);
        assert!(metrics.total_validation_time_ms > 0);

        let stats = engine.calculate_statistics(&platform_results).unwrap();
        assert!(stats.mean_validation_time_ms > 0.0);
        assert!(stats.throughput_validations_per_second > 0.0);
    }
}
