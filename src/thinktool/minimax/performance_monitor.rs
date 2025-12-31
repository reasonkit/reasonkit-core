//! Performance Monitor for MiniMax M2 ThinkTools
//!
//! Provides real-time performance tracking, optimization, and monitoring
//! for M2-enhanced ThinkTools with comprehensive metrics collection.

use super::M2ThinkToolResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Performance metrics for M2 ThinkTools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub target_confidence: f64,
    pub max_processing_time_ms: u64,
    pub max_token_count: u32,
    pub target_cost_efficiency: f64,
    pub target_cross_validation_score: f64,
    pub achieved_confidence: f64,
    pub achieved_processing_time_ms: u64,
    pub achieved_token_count: u32,
    pub achieved_cost_efficiency: f64,
    pub achieved_cross_validation_score: f64,
    pub constraint_adherence_rate: f64,
    pub validation_success_rate: f64,
    pub overall_performance_score: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            target_confidence: 0.9,
            max_processing_time_ms: 5000,
            max_token_count: 2000,
            target_cost_efficiency: 1.0,
            target_cross_validation_score: 0.85,
            achieved_confidence: 0.0,
            achieved_processing_time_ms: 0,
            achieved_token_count: 0,
            achieved_cost_efficiency: 0.0,
            achieved_cross_validation_score: 0.0,
            constraint_adherence_rate: 0.0,
            validation_success_rate: 0.0,
            overall_performance_score: 0.0,
        }
    }
}

/// Monitoring result with analysis and recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringResult {
    pub monitoring_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub execution_profile: String,
    pub thinktool_module: String,
    pub performance_metrics: PerformanceMetrics,
    pub performance_analysis: PerformanceAnalysis,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    pub alerts: Vec<PerformanceAlert>,
    pub trend_analysis: TrendAnalysis,
}

/// Performance analysis of ThinkTool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub overall_score: f64,
    pub confidence_score: f64,
    pub efficiency_score: f64,
    pub quality_score: f64,
    pub speed_score: f64,
    pub reliability_score: f64,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub improvement_opportunities: Vec<String>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: String,
    pub priority: RecommendationPriority,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_difficulty: String,
    pub estimated_effort_hours: f64,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Performance alert for threshold violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub alert_id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub threshold_violated: String,
    pub actual_value: f64,
    pub threshold_value: f64,
    pub recommendation: Option<String>,
}

/// Alert types for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    ConfidenceBelowTarget,
    ProcessingTimeExceeded,
    TokenLimitExceeded,
    CostEfficiencyBelowTarget,
    ValidationFailure,
    ConstraintViolation,
    SystemError,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Trend analysis for performance over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub time_period_hours: u32,
    pub confidence_trend: TrendDirection,
    pub efficiency_trend: TrendDirection,
    pub quality_trend: TrendDirection,
    pub speed_trend: TrendDirection,
    pub reliability_trend: TrendDirection,
    pub overall_performance_trend: TrendDirection,
    pub data_points: u32,
}

/// Trend direction indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    InsufficientData,
}

/// Performance monitor for M2 ThinkTools
pub struct PerformanceMonitor {
    pub monitoring_id: String,
    pub start_time: Instant,
    pub execution_history: VecDeque<MonitoringResult>,
    pub performance_baselines: HashMap<String, PerformanceBaseline>,
    pub alert_thresholds: AlertThresholds,
    pub optimization_cache: OptimizationCache,
    pub monitoring_config: MonitoringConfig,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        let mut baselines = HashMap::new();
        baselines.insert(
            "enhanced_gigathink".to_string(),
            PerformanceBaseline::gigathink_baseline(),
        );
        baselines.insert(
            "enhanced_laserlogic".to_string(),
            PerformanceBaseline::laserlogic_baseline(),
        );
        baselines.insert(
            "enhanced_bedrock".to_string(),
            PerformanceBaseline::bedrock_baseline(),
        );
        baselines.insert(
            "enhanced_proofguard".to_string(),
            PerformanceBaseline::proofguard_baseline(),
        );
        baselines.insert(
            "enhanced_brutalhonesty".to_string(),
            PerformanceBaseline::brutalhonesty_baseline(),
        );

        Self {
            monitoring_id: uuid::Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            execution_history: VecDeque::with_capacity(1000), // Keep last 1000 executions
            performance_baselines: baselines,
            alert_thresholds: AlertThresholds::default(),
            optimization_cache: OptimizationCache::new(),
            monitoring_config: MonitoringConfig::default(),
        }
    }

    /// Monitor ThinkTool execution and collect performance metrics
    pub fn monitor_execution(
        &mut self,
        result: &M2ThinkToolResult,
        profile: &super::ProfileType,
        module: &str,
    ) -> MonitoringResult {
        let timestamp = chrono::Utc::now();

        // Calculate performance metrics
        let metrics = self.calculate_performance_metrics(result, module);

        // Perform performance analysis
        let analysis = self.analyze_performance(&metrics, module);

        // Generate optimization recommendations
        let recommendations = self.generate_recommendations(&metrics, &analysis);

        // Check for alerts
        let alerts = self.check_alerts(&metrics, module);

        // Update trend analysis
        self.update_trend_analysis(&metrics);

        // Create monitoring result
        let monitoring_result = MonitoringResult {
            monitoring_id: uuid::Uuid::new_v4().to_string(),
            timestamp,
            execution_profile: format!("{:?}", profile),
            thinktool_module: module.to_string(),
            performance_metrics: metrics,
            performance_analysis: analysis,
            optimization_recommendations: recommendations,
            alerts,
            trend_analysis: self.calculate_trend_analysis(),
        };

        // Add to execution history
        if self.execution_history.len() >= self.monitoring_config.history_limit {
            self.execution_history.pop_front();
        }
        self.execution_history.push_back(monitoring_result.clone());

        monitoring_result
    }

    /// Calculate comprehensive performance metrics
    fn calculate_performance_metrics(
        &self,
        result: &M2ThinkToolResult,
        module: &str,
    ) -> PerformanceMetrics {
        let default_baseline = PerformanceBaseline::default();
        let baseline = self
            .performance_baselines
            .get(module)
            .unwrap_or(&default_baseline);

        // Calculate achievement scores
        let achieved_confidence = result.confidence;
        let achieved_processing_time = result.processing_time_ms;
        let achieved_token_count = result.token_count;

        // Calculate cost efficiency
        let baseline_cost = baseline.average_cost_per_execution;
        let current_cost = self.calculate_current_cost(result);
        let achieved_cost_efficiency = if baseline_cost > 0.0 {
            baseline_cost / current_cost
        } else {
            1.0
        };

        // Calculate cross-validation score
        let achieved_cross_validation_score = if !result.interleaved_steps.is_empty() {
            result
                .interleaved_steps
                .iter()
                .filter(|step| step.cross_validation_passed)
                .count() as f64
                / result.interleaved_steps.len() as f64
        } else {
            0.7
        };

        // Calculate constraint adherence rate
        let constraint_adherence_rate = match &result.constraint_adherence {
            super::ConstraintResult::Passed(score) => *score,
            super::ConstraintResult::Failed(_) => 0.0,
            super::ConstraintResult::Pending => 0.5,
        };

        // Calculate validation success rate
        let validation_success_rate = if !result.interleaved_steps.is_empty() {
            let total_validations = result
                .interleaved_steps
                .iter()
                .map(|step| step.validation_results.len())
                .sum::<usize>();
            let successful_validations = result
                .interleaved_steps
                .iter()
                .flat_map(|step| &step.validation_results)
                .filter(|result| result.passed)
                .count();

            if total_validations > 0 {
                successful_validations as f64 / total_validations as f64
            } else {
                0.7
            }
        } else {
            0.7
        };

        // Calculate overall performance score
        let overall_performance_score = self.calculate_overall_performance_score(
            achieved_confidence,
            achieved_processing_time,
            achieved_token_count,
            achieved_cost_efficiency,
            achieved_cross_validation_score,
            constraint_adherence_rate,
            validation_success_rate,
        );

        PerformanceMetrics {
            target_confidence: baseline.target_confidence,
            max_processing_time_ms: baseline.max_processing_time_ms,
            max_token_count: baseline.max_token_count,
            target_cost_efficiency: baseline.target_cost_efficiency,
            target_cross_validation_score: baseline.target_cross_validation_score,
            achieved_confidence,
            achieved_processing_time_ms: achieved_processing_time,
            achieved_token_count,
            achieved_cost_efficiency,
            achieved_cross_validation_score,
            constraint_adherence_rate,
            validation_success_rate,
            overall_performance_score,
        }
    }

    /// Calculate current cost of execution
    fn calculate_current_cost(&self, result: &M2ThinkToolResult) -> f64 {
        // Simplified cost calculation based on token count and processing time
        let token_cost = result.token_count as f64 * 0.0001; // $0.0001 per token
        let time_cost = result.processing_time_ms as f64 * 0.00001; // $0.00001 per millisecond

        token_cost + time_cost
    }

    /// Calculate overall performance score
    #[allow(clippy::too_many_arguments)]
    fn calculate_overall_performance_score(
        &self,
        confidence: f64,
        processing_time: u64,
        _token_count: u32,
        cost_efficiency: f64,
        cross_validation_score: f64,
        constraint_adherence: f64,
        validation_success: f64,
    ) -> f64 {
        // Weighted performance calculation
        let confidence_weight = 0.25;
        let speed_weight = 0.20;
        let efficiency_weight = 0.15;
        let quality_weight = 0.20;
        let reliability_weight = 0.20;

        // Speed score (inverse of processing time)
        let baseline_time = 5000.0; // 5 seconds baseline
        let speed_score = (baseline_time / processing_time as f64).clamp(0.1, 2.0);

        // Efficiency score
        let efficiency_score = cost_efficiency.clamp(0.1, 2.0);

        // Quality score
        let quality_score = (cross_validation_score + validation_success) / 2.0;

        // Reliability score
        let reliability_score = constraint_adherence;

        (confidence * confidence_weight
            + speed_score * speed_weight
            + efficiency_score * efficiency_weight
            + quality_score * quality_weight
            + reliability_score * reliability_weight)
            .clamp(0.0, 1.0)
    }

    /// Analyze performance metrics
    fn analyze_performance(
        &self,
        metrics: &PerformanceMetrics,
        module: &str,
    ) -> PerformanceAnalysis {
        let default_baseline = PerformanceBaseline::default();
        let baseline = self
            .performance_baselines
            .get(module)
            .unwrap_or(&default_baseline);

        let overall_score = metrics.overall_performance_score;
        let confidence_score = metrics.achieved_confidence / metrics.target_confidence;
        let efficiency_score = (metrics.achieved_cost_efficiency
            + (baseline.max_processing_time_ms as f64
                / metrics.achieved_processing_time_ms as f64))
            / 2.0;
        let quality_score =
            (metrics.achieved_cross_validation_score + metrics.validation_success_rate) / 2.0;
        let speed_score = (baseline.max_processing_time_ms as f64
            / metrics.achieved_processing_time_ms as f64)
            .min(2.0);
        let reliability_score = metrics.constraint_adherence_rate;

        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();
        let mut improvement_opportunities = Vec::new();

        // Identify strengths
        if confidence_score >= 1.0 {
            strengths.push("Confidence target met or exceeded".to_string());
        }
        if speed_score >= 1.2 {
            strengths.push("Excellent processing speed".to_string());
        }
        if metrics.achieved_cost_efficiency >= 1.1 {
            strengths.push("High cost efficiency".to_string());
        }
        if reliability_score >= 0.9 {
            strengths.push("Strong constraint adherence".to_string());
        }

        // Identify weaknesses
        if confidence_score < 0.8 {
            weaknesses.push("Confidence below acceptable threshold".to_string());
        }
        if metrics.achieved_processing_time_ms > baseline.max_processing_time_ms {
            weaknesses.push("Processing time exceeds target".to_string());
        }
        if metrics.achieved_token_count > baseline.max_token_count {
            weaknesses.push("Token usage exceeds budget".to_string());
        }
        if metrics.constraint_adherence_rate < 0.7 {
            weaknesses.push("Frequent constraint violations".to_string());
        }

        // Identify improvement opportunities
        if speed_score < 1.0 {
            improvement_opportunities.push("Optimize processing speed".to_string());
        }
        if metrics.achieved_cost_efficiency < 1.0 {
            improvement_opportunities.push("Reduce execution cost".to_string());
        }
        if metrics.achieved_cross_validation_score < 0.8 {
            improvement_opportunities.push("Improve validation rigor".to_string());
        }

        PerformanceAnalysis {
            overall_score,
            confidence_score,
            efficiency_score,
            quality_score,
            speed_score,
            reliability_score,
            strengths,
            weaknesses,
            improvement_opportunities,
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        metrics: &PerformanceMetrics,
        analysis: &PerformanceAnalysis,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Confidence-based recommendations
        if analysis.confidence_score < 0.8 {
            recommendations.push(OptimizationRecommendation {
                category: "Confidence".to_string(),
                priority: RecommendationPriority::Critical,
                description: "Increase validation rounds and cross-checking".to_string(),
                expected_improvement: 0.15,
                implementation_difficulty: "Medium".to_string(),
                estimated_effort_hours: 8.0,
            });
        }

        // Speed-based recommendations
        if analysis.speed_score < 0.8 {
            recommendations.push(OptimizationRecommendation {
                category: "Speed".to_string(),
                priority: RecommendationPriority::High,
                description: "Implement parallel processing and caching".to_string(),
                expected_improvement: 0.25,
                implementation_difficulty: "High".to_string(),
                estimated_effort_hours: 16.0,
            });
        }

        // Cost-based recommendations
        if metrics.achieved_cost_efficiency < 1.0 {
            recommendations.push(OptimizationRecommendation {
                category: "Cost".to_string(),
                priority: RecommendationPriority::Medium,
                description: "Optimize token usage and reduce unnecessary processing".to_string(),
                expected_improvement: 0.20,
                implementation_difficulty: "Medium".to_string(),
                estimated_effort_hours: 12.0,
            });
        }

        // Quality-based recommendations
        if analysis.quality_score < 0.8 {
            recommendations.push(OptimizationRecommendation {
                category: "Quality".to_string(),
                priority: RecommendationPriority::High,
                description: "Enhance validation logic and cross-reference mechanisms".to_string(),
                expected_improvement: 0.18,
                implementation_difficulty: "High".to_string(),
                estimated_effort_hours: 20.0,
            });
        }

        recommendations
    }

    /// Check for performance alerts
    fn check_alerts(&self, metrics: &PerformanceMetrics, module: &str) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();
        let _baseline = self
            .performance_baselines
            .get(module)
            .unwrap_or(&PerformanceBaseline::default());

        // Confidence alert
        if metrics.achieved_confidence < self.alert_thresholds.confidence_threshold {
            alerts.push(PerformanceAlert {
                alert_id: uuid::Uuid::new_v4().to_string(),
                alert_type: AlertType::ConfidenceBelowTarget,
                severity: if metrics.achieved_confidence
                    < self.alert_thresholds.confidence_threshold * 0.8
                {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                message: format!(
                    "Confidence {} below threshold {}",
                    metrics.achieved_confidence, self.alert_thresholds.confidence_threshold
                ),
                threshold_violated: "confidence_threshold".to_string(),
                actual_value: metrics.achieved_confidence,
                threshold_value: self.alert_thresholds.confidence_threshold,
                recommendation: Some(
                    "Increase validation rigor and constraint checking".to_string(),
                ),
            });
        }

        // Processing time alert
        if metrics.achieved_processing_time_ms > self.alert_thresholds.processing_time_threshold {
            alerts.push(PerformanceAlert {
                alert_id: uuid::Uuid::new_v4().to_string(),
                alert_type: AlertType::ProcessingTimeExceeded,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Processing time {}ms exceeds threshold {}ms",
                    metrics.achieved_processing_time_ms,
                    self.alert_thresholds.processing_time_threshold
                ),
                threshold_violated: "processing_time_threshold".to_string(),
                actual_value: metrics.achieved_processing_time_ms as f64,
                threshold_value: self.alert_thresholds.processing_time_threshold as f64,
                recommendation: Some(
                    "Optimize algorithm efficiency and implement caching".to_string(),
                ),
            });
        }

        // Cost efficiency alert
        if metrics.achieved_cost_efficiency < self.alert_thresholds.cost_efficiency_threshold {
            alerts.push(PerformanceAlert {
                alert_id: uuid::Uuid::new_v4().to_string(),
                alert_type: AlertType::CostEfficiencyBelowTarget,
                severity: AlertSeverity::Info,
                message: format!(
                    "Cost efficiency {} below target {}",
                    metrics.achieved_cost_efficiency,
                    self.alert_thresholds.cost_efficiency_threshold
                ),
                threshold_violated: "cost_efficiency_threshold".to_string(),
                actual_value: metrics.achieved_cost_efficiency,
                threshold_value: self.alert_thresholds.cost_efficiency_threshold,
                recommendation: Some(
                    "Reduce token usage and optimize prompt efficiency".to_string(),
                ),
            });
        }

        alerts
    }

    /// Update trend analysis
    fn update_trend_analysis(&mut self, metrics: &PerformanceMetrics) {
        // This would update internal trend tracking
        // For simplicity, we'll just note that updates happen
        let _ = metrics; // Prevent unused variable warning
    }

    /// Calculate trend analysis
    fn calculate_trend_analysis(&self) -> TrendAnalysis {
        let recent_executions: Vec<_> = self
            .execution_history
            .iter()
            .rev()
            .take(24) // Last 24 executions
            .collect();

        let data_points = recent_executions.len() as u32;

        if data_points < 3 {
            return TrendAnalysis {
                time_period_hours: 1,
                confidence_trend: TrendDirection::InsufficientData,
                efficiency_trend: TrendDirection::InsufficientData,
                quality_trend: TrendDirection::InsufficientData,
                speed_trend: TrendDirection::InsufficientData,
                reliability_trend: TrendDirection::InsufficientData,
                overall_performance_trend: TrendDirection::InsufficientData,
                data_points,
            };
        }

        // Calculate trends (simplified)
        let confidence_trend = self.calculate_trend_direction(
            recent_executions
                .iter()
                .map(|r| r.performance_metrics.achieved_confidence)
                .collect(),
        );
        let efficiency_trend = self.calculate_trend_direction(
            recent_executions
                .iter()
                .map(|r| r.performance_metrics.achieved_cost_efficiency)
                .collect(),
        );
        let quality_trend = self.calculate_trend_direction(
            recent_executions
                .iter()
                .map(|r| {
                    (r.performance_metrics.achieved_cross_validation_score
                        + r.performance_metrics.validation_success_rate)
                        / 2.0
                })
                .collect(),
        );
        let speed_trend = self.calculate_trend_direction(
            recent_executions
                .iter()
                .map(|r| 1.0 / r.performance_metrics.achieved_processing_time_ms as f64)
                .collect(),
        );
        let reliability_trend = self.calculate_trend_direction(
            recent_executions
                .iter()
                .map(|r| r.performance_metrics.constraint_adherence_rate)
                .collect(),
        );
        let overall_trend = self.calculate_trend_direction(
            recent_executions
                .iter()
                .map(|r| r.performance_metrics.overall_performance_score)
                .collect(),
        );

        TrendAnalysis {
            time_period_hours: 1,
            confidence_trend,
            efficiency_trend,
            quality_trend,
            speed_trend,
            reliability_trend,
            overall_performance_trend: overall_trend,
            data_points,
        }
    }

    /// Calculate trend direction from data points
    fn calculate_trend_direction(&self, values: Vec<f64>) -> TrendDirection {
        if values.len() < 3 {
            return TrendDirection::InsufficientData;
        }

        let first_third = &values[0..values.len() / 3];
        let last_third = &values[values.len() * 2 / 3..];

        let first_avg: f64 = first_third.iter().sum::<f64>() / first_third.len() as f64;
        let last_avg: f64 = last_third.iter().sum::<f64>() / last_third.len() as f64;

        let threshold = 0.05; // 5% change threshold
        let change_ratio = (last_avg - first_avg) / first_avg;

        if change_ratio > threshold {
            TrendDirection::Improving
        } else if change_ratio < -threshold {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        }
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let recent_executions = &self.execution_history;
        let total_executions = recent_executions.len();

        if total_executions == 0 {
            return PerformanceSummary::default();
        }

        let avg_confidence = recent_executions
            .iter()
            .map(|r| r.performance_metrics.achieved_confidence)
            .sum::<f64>()
            / total_executions as f64;

        let avg_processing_time = recent_executions
            .iter()
            .map(|r| r.performance_metrics.achieved_processing_time_ms)
            .sum::<u64>()
            / total_executions as u64;

        let avg_performance_score = recent_executions
            .iter()
            .map(|r| r.performance_metrics.overall_performance_score)
            .sum::<f64>()
            / total_executions as f64;

        let total_alerts = recent_executions
            .iter()
            .map(|r| r.alerts.len())
            .sum::<usize>();

        PerformanceSummary {
            total_executions,
            average_confidence: avg_confidence,
            average_processing_time_ms: avg_processing_time,
            average_performance_score: avg_performance_score,
            total_alerts_generated: total_alerts,
            monitoring_uptime_hours: self.start_time.elapsed().as_secs() / 3600,
        }
    }
}

/// Performance baseline for different modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub target_confidence: f64,
    pub max_processing_time_ms: u64,
    pub max_token_count: u32,
    pub target_cost_efficiency: f64,
    pub target_cross_validation_score: f64,
    pub average_cost_per_execution: f64,
}

impl PerformanceBaseline {
    pub fn gigathink_baseline() -> Self {
        Self {
            target_confidence: 0.92,
            max_processing_time_ms: 5000,
            max_token_count: 2500,
            target_cost_efficiency: 1.08,
            target_cross_validation_score: 0.85,
            average_cost_per_execution: 0.05,
        }
    }

    pub fn laserlogic_baseline() -> Self {
        Self {
            target_confidence: 0.95,
            max_processing_time_ms: 4000,
            max_token_count: 1800,
            target_cost_efficiency: 1.08,
            target_cross_validation_score: 0.90,
            average_cost_per_execution: 0.04,
        }
    }

    pub fn bedrock_baseline() -> Self {
        Self {
            target_confidence: 0.90,
            max_processing_time_ms: 4500,
            max_token_count: 2000,
            target_cost_efficiency: 1.08,
            target_cross_validation_score: 0.88,
            average_cost_per_execution: 0.045,
        }
    }

    pub fn proofguard_baseline() -> Self {
        Self {
            target_confidence: 0.95,
            max_processing_time_ms: 5000,
            max_token_count: 2200,
            target_cost_efficiency: 1.08,
            target_cross_validation_score: 0.92,
            average_cost_per_execution: 0.055,
        }
    }

    pub fn brutalhonesty_baseline() -> Self {
        Self {
            target_confidence: 0.95,
            max_processing_time_ms: 4500,
            max_token_count: 2000,
            target_cost_efficiency: 1.08,
            target_cross_validation_score: 0.90,
            average_cost_per_execution: 0.05,
        }
    }
}

impl Default for PerformanceBaseline {
    fn default() -> Self {
        Self::gigathink_baseline()
    }
}

/// Alert thresholds configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub confidence_threshold: f64,
    pub processing_time_threshold: u64,
    pub cost_efficiency_threshold: f64,
    pub validation_failure_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.8,
            processing_time_threshold: 8000,
            cost_efficiency_threshold: 0.9,
            validation_failure_threshold: 0.7,
        }
    }
}

/// Optimization cache for performance improvements
#[derive(Debug)]
pub struct OptimizationCache {
    pub cached_optimizations: HashMap<String, f64>,
    pub cache_size_limit: usize,
}

impl Default for OptimizationCache {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationCache {
    pub fn new() -> Self {
        Self {
            cached_optimizations: HashMap::new(),
            cache_size_limit: 100,
        }
    }

    pub fn get_optimization(&self, key: &str) -> Option<f64> {
        self.cached_optimizations.get(key).copied()
    }

    pub fn cache_optimization(&mut self, key: String, value: f64) {
        if self.cached_optimizations.len() >= self.cache_size_limit {
            // Remove oldest entry (simple FIFO)
            if let Some(first_key) = self.cached_optimizations.keys().next().cloned() {
                self.cached_optimizations.remove(&first_key);
            }
        }
        self.cached_optimizations.insert(key, value);
    }
}

/// Monitoring configuration
#[derive(Debug)]
pub struct MonitoringConfig {
    pub history_limit: usize,
    pub alert_cooldown_minutes: u32,
    pub trend_analysis_window: usize,
    pub performance_check_interval: Duration,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            history_limit: 1000,
            alert_cooldown_minutes: 15,
            trend_analysis_window: 24,
            performance_check_interval: Duration::from_secs(60),
        }
    }
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_executions: usize,
    pub average_confidence: f64,
    pub average_processing_time_ms: u64,
    pub average_performance_score: f64,
    pub total_alerts_generated: usize,
    pub monitoring_uptime_hours: u64,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            total_executions: 0,
            average_confidence: 0.0,
            average_processing_time_ms: 0,
            average_performance_score: 0.0,
            total_alerts_generated: 0,
            monitoring_uptime_hours: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_calculation() {
        let mut monitor = PerformanceMonitor::new();
        let result = super::super::M2ThinkToolResult::new(
            "test_module".to_string(),
            serde_json::json!({"test": true}),
        );

        let monitoring_result = monitor.monitor_execution(
            &result,
            &super::super::ProfileType::Balanced,
            "enhanced_gigathink",
        );

        assert!(monitoring_result.performance_metrics.achieved_confidence >= 0.0);
        assert!(
            monitoring_result
                .performance_metrics
                .overall_performance_score
                >= 0.0
        );
    }

    #[test]
    fn test_alert_generation() {
        let monitor = PerformanceMonitor::new();
        let metrics = PerformanceMetrics {
            target_confidence: 0.8,
            max_processing_time_ms: 5000,
            max_token_count: 2000,
            target_cost_efficiency: 1.0,
            target_cross_validation_score: 0.8,
            achieved_confidence: 0.6, // Below threshold
            achieved_processing_time_ms: 3000,
            achieved_token_count: 1500,
            achieved_cost_efficiency: 1.2,
            achieved_cross_validation_score: 0.9,
            constraint_adherence_rate: 0.9,
            validation_success_rate: 0.85,
            overall_performance_score: 0.8,
        };

        let alerts = monitor.check_alerts(&metrics, "enhanced_gigathink");
        assert!(!alerts.is_empty());
        assert!(alerts
            .iter()
            .any(|a| matches!(a.alert_type, AlertType::ConfidenceBelowTarget)));
    }

    #[test]
    fn test_performance_baselines() {
        let baseline = PerformanceBaseline::gigathink_baseline();
        assert_eq!(baseline.target_confidence, 0.92);
        assert_eq!(baseline.max_processing_time_ms, 5000);
    }
}
