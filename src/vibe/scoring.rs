//! # VIBE Scoring System
//!
//! Comprehensive scoring system for VIBE protocol validation with detailed
//! breakdowns and intelligent scoring algorithms.

use super::*;
use crate::vibe::validation::{
    ConfidenceInterval, IssueCategory, PlatformValidationResult, Severity, VIBEError,
    ValidationIssue, ValidationStatus,
};
use crate::vibe::validation_config::{
    ConditionOperator, ConditionValue, PlatformConfig, ScoringWeights, ValidationDepth,
};
use serde::{Deserialize, Serialize};

/// VIBE Score representation with detailed breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VIBEScore {
    /// Overall score (0-100)
    pub overall_score: f32,

    /// Platform-specific scores
    pub platform_scores: HashMap<Platform, f32>,

    /// Component scores breakdown
    pub component_scores: ComponentScores,

    /// Confidence interval for the score
    pub confidence_interval: Option<ConfidenceInterval>,

    /// Score metadata
    pub metadata: ScoreMetadata,
}

/// Component scores breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentScores {
    /// Logical consistency score
    pub logical_consistency: f32,

    /// Practical applicability score
    pub practical_applicability: f32,

    /// Platform compatibility score
    pub platform_compatibility: f32,

    /// Performance requirements score
    pub performance_requirements: f32,

    /// Security considerations score
    pub security_considerations: f32,

    /// User experience score
    pub user_experience: f32,

    /// Code quality score
    pub code_quality: f32,

    /// Custom component scores
    pub custom_scores: HashMap<String, f32>,
}

/// Score metadata for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreMetadata {
    /// Timestamp of scoring
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Scoring algorithm version
    pub algorithm_version: String,

    /// Platform validation depth
    pub validation_depth: ValidationDepth,

    /// Quality gates passed
    pub quality_gates: Vec<QualityGate>,

    /// Scoring factors applied
    pub scoring_factors: HashMap<String, f32>,

    /// Normalization applied
    pub normalization_applied: bool,
}

/// Quality gates for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
    pub gate_name: String,
    pub passed: bool,
    pub score_impact: f32,
    pub description: String,
}

/// Detailed score breakdown by category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Category name
    pub category: String,

    /// Category score
    pub score: f32,

    /// Weight in overall score
    pub weight: f32,

    /// Sub-scores breakdown
    pub sub_scores: Vec<SubScore>,

    /// Issues affecting this category
    pub issues: Vec<ValidationIssue>,

    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Individual sub-score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubScore {
    pub name: String,
    pub score: f32,
    pub weight: f32,
    pub description: String,
}

/// Scoring criteria configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringCriteria {
    /// Base scoring weights
    pub base_weights: ScoringWeights,

    /// Platform-specific weight adjustments
    pub platform_adjustments: HashMap<Platform, PlatformScoreAdjustments>,

    /// Dynamic scoring factors
    pub dynamic_factors: HashMap<String, DynamicFactor>,

    /// Penalty rules
    pub penalty_rules: Vec<PenaltyRule>,

    /// Bonus rules
    pub bonus_rules: Vec<BonusRule>,
}

/// Platform-specific score adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformScoreAdjustments {
    pub weight_multipliers: HashMap<String, f32>,
    pub threshold_adjustments: HashMap<String, f32>,
    pub custom_penalties: Vec<CustomPenalty>,
}

/// Dynamic scoring factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicFactor {
    pub factor_name: String,
    pub calculation_method: CalculationMethod,
    pub weight: f32,
    pub conditions: Vec<ScoringCondition>,
}

/// Penalty rule for score deduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyRule {
    pub rule_name: String,
    pub trigger_conditions: Vec<ScoringCondition>,
    pub penalty_amount: f32,
    pub severity_level: Severity,
}

/// Bonus rule for score addition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonusRule {
    pub rule_name: String,
    pub trigger_conditions: Vec<ScoringCondition>,
    pub bonus_amount: f32,
    pub max_bonus: f32,
}

/// Scoring condition for rule triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringCondition {
    pub condition_type: ConditionType,
    pub target: String,
    pub operator: ConditionOperator,
    pub value: ConditionValue,
}

/// Custom penalty definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPenalty {
    pub penalty_name: String,
    pub description: String,
    pub amount: f32,
    pub applicability: PenaltyApplicability,
}

/// Penalty applicability conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PenaltyApplicability {
    Always,
    PlatformSpecific(Platform),
    ConditionBased(Vec<ScoringCondition>),
    ThresholdBased { metric: String, threshold: f32 },
}

/// Calculation methods for dynamic factors
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalculationMethod {
    Linear,
    Exponential,
    Logarithmic,
    StepFunction,
    Custom,
}

/// Condition types for scoring rules
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionType {
    Score,
    IssueCount,
    IssueSeverity,
    PlatformScore,
    ComponentScore,
    Custom,
}

/// Scoring engine for VIBE validation
pub struct ScoringEngine {
    /// Default scoring criteria
    default_criteria: ScoringCriteria,

    /// Platform-specific scoring configurations
    platform_configs: HashMap<Platform, PlatformScoringConfig>,

    /// Historical score database for comparison
    score_history: ScoreHistory,
}

/// Platform-specific scoring configuration
#[derive(Debug, Clone)]
pub struct PlatformScoringConfig {
    pub platform: Platform,
    pub base_weights: ScoringWeights,
    pub penalty_thresholds: HashMap<String, f32>,
    pub bonus_multipliers: HashMap<String, f32>,
    pub normalization_method: NormalizationMethod,
}

/// Score history for trend analysis
#[derive(Debug, Default)]
pub struct ScoreHistory {
    pub scores: Vec<HistoricalScore>,
    pub trend_analysis: Option<ScoreTrendAnalysis>,
}

/// Historical score entry
#[derive(Debug, Clone)]
pub struct HistoricalScore {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub score: f32,
    pub platform: Platform,
    pub protocol_id: Uuid,
}

/// Score trend analysis
#[derive(Debug, Clone)]
pub struct ScoreTrendAnalysis {
    pub trend_direction: TrendDirection,
    pub trend_strength: f32,
    pub seasonal_patterns: Vec<SeasonalPattern>,
    pub outlier_analysis: OutlierAnalysis,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Seasonal pattern in scores
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    pub period: String,
    pub amplitude: f32,
    pub phase: f32,
}

/// Outlier analysis
#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    pub outlier_count: usize,
    pub outlier_threshold: f32,
    pub unusual_patterns: Vec<String>,
}

/// Normalization methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    Percentile,
    Custom { min: f32, max: f32 },
}

impl ScoringEngine {
    /// Create new scoring engine with default configuration
    pub fn new() -> Self {
        Self {
            default_criteria: Self::create_default_criteria(),
            platform_configs: Self::create_platform_configs(),
            score_history: ScoreHistory::default(),
        }
    }

    /// Calculate comprehensive VIBE score
    pub fn calculate_vibe_score(
        &mut self,
        validation_results: &HashMap<Platform, PlatformValidationResult>,
        criteria: &ValidationCriteria,
        config: &ValidationConfig,
    ) -> Result<VIBEScore, VIBEError> {
        let _start_time = chrono::Utc::now();

        // Calculate component scores
        let component_scores = self.calculate_component_scores(validation_results, criteria)?;

        // Calculate platform scores
        let platform_scores = self.calculate_platform_scores(validation_results, config)?;

        // Apply scoring criteria and weights
        let weighted_score = self.apply_scoring_weights(&component_scores, criteria)?;

        // Apply dynamic factors and rules
        let adjusted_score =
            self.apply_dynamic_factors(weighted_score, validation_results, criteria)?;

        // Calculate confidence interval
        let confidence_interval = self.calculate_confidence_interval(&platform_scores)?;

        // Generate score metadata
        let metadata = self.generate_score_metadata(&adjusted_score, &platform_scores, config)?;

        // Apply normalization if needed
        let final_score = self.apply_normalization(adjusted_score, &metadata)?;

        // Store in history
        self.update_score_history(&final_score, &platform_scores, validation_results)?;

        Ok(VIBEScore {
            overall_score: final_score,
            platform_scores,
            component_scores,
            confidence_interval,
            metadata,
        })
    }

    /// Calculate component-level scores
    fn calculate_component_scores(
        &self,
        validation_results: &HashMap<Platform, PlatformValidationResult>,
        criteria: &ValidationCriteria,
    ) -> Result<ComponentScores, VIBEError> {
        let mut logical_consistency = 0.0;
        let mut practical_applicability = 0.0;
        let mut platform_compatibility = 0.0;
        let mut performance_requirements = 0.0;
        let mut security_considerations = 0.0;
        let mut user_experience = 0.0;
        let mut code_quality = 0.0;
        let mut total_platforms = 0.0;

        // Aggregate scores from all platforms
        for result in validation_results.values() {
            total_platforms += 1.0;

            // Calculate platform-specific scores based on validation results
            let platform_logic = self.extract_logical_consistency_score(result);
            let platform_practical = self.extract_practical_score(result);
            let platform_compat = self.extract_compatibility_score(result);
            let platform_perf = result.performance_metrics.average_response_time_ms as f32 / 100.0;
            let platform_security = self.extract_security_score(result);
            let platform_ux = self.extract_ux_score(result);
            let platform_quality = self.extract_quality_score(result);

            logical_consistency += platform_logic;
            practical_applicability += platform_practical;
            platform_compatibility += platform_compat;
            performance_requirements += platform_perf;
            security_considerations += platform_security;
            user_experience += platform_ux;
            code_quality += platform_quality;
        }

        // Average across platforms
        if total_platforms > 0.0 {
            logical_consistency /= total_platforms;
            practical_applicability /= total_platforms;
            platform_compatibility /= total_platforms;
            performance_requirements /= total_platforms;
            security_considerations /= total_platforms;
            user_experience /= total_platforms;
            code_quality /= total_platforms;
        }

        // Apply criteria adjustments
        if !criteria.logical_consistency {
            logical_consistency *= 0.5;
        }
        if !criteria.practical_applicability {
            practical_applicability *= 0.5;
        }
        if !criteria.platform_compatibility {
            platform_compatibility *= 0.5;
        }
        if !criteria.performance_requirements {
            performance_requirements *= 0.5;
        }
        if !criteria.security_considerations {
            security_considerations *= 0.5;
        }
        if !criteria.user_experience {
            user_experience *= 0.5;
        }

        Ok(ComponentScores {
            logical_consistency: logical_consistency.clamp(0.0, 100.0),
            practical_applicability: practical_applicability.clamp(0.0, 100.0),
            platform_compatibility: platform_compatibility.clamp(0.0, 100.0),
            performance_requirements: performance_requirements.clamp(0.0, 100.0),
            security_considerations: security_considerations.clamp(0.0, 100.0),
            user_experience: user_experience.clamp(0.0, 100.0),
            code_quality: code_quality.clamp(0.0, 100.0),
            custom_scores: HashMap::new(),
        })
    }

    /// Calculate platform-specific scores
    fn calculate_platform_scores(
        &self,
        validation_results: &HashMap<Platform, PlatformValidationResult>,
        config: &ValidationConfig,
    ) -> Result<HashMap<Platform, f32>, VIBEError> {
        let mut platform_scores = HashMap::new();

        for (platform, result) in validation_results {
            // Base score from validation result
            let mut score = result.score;

            // Apply platform-specific adjustments
            if let Some(platform_config) = self.platform_configs.get(platform) {
                score = self.apply_platform_adjustments(score, result, platform_config)?;
            }

            // Apply custom platform configuration from validation config
            if let Some(custom_config) = config.platform_configs.get(platform) {
                score = self.apply_custom_platform_config(score, custom_config)?;
            }

            platform_scores.insert(*platform, score.clamp(0.0, 100.0));
        }

        Ok(platform_scores)
    }

    /// Apply scoring weights to component scores
    fn apply_scoring_weights(
        &self,
        component_scores: &ComponentScores,
        criteria: &ValidationCriteria,
    ) -> Result<f32, VIBEError> {
        let weights = &self.default_criteria.base_weights;

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        // Apply weights based on enabled criteria
        if criteria.logical_consistency {
            weighted_sum += component_scores.logical_consistency * weights.logical_consistency;
            total_weight += weights.logical_consistency;
        }

        if criteria.practical_applicability {
            weighted_sum +=
                component_scores.practical_applicability * weights.practical_applicability;
            total_weight += weights.practical_applicability;
        }

        if criteria.platform_compatibility {
            weighted_sum +=
                component_scores.platform_compatibility * weights.platform_compatibility;
            total_weight += weights.platform_compatibility;
        }

        if criteria.performance_requirements {
            weighted_sum +=
                component_scores.performance_requirements * weights.performance_requirements;
            total_weight += weights.performance_requirements;
        }

        if criteria.security_considerations {
            weighted_sum +=
                component_scores.security_considerations * weights.security_considerations;
            total_weight += weights.security_considerations;
        }

        if criteria.user_experience {
            weighted_sum += component_scores.user_experience * weights.user_experience;
            total_weight += weights.user_experience;
        }

        // Add code quality with default weight if no other criteria
        weighted_sum += component_scores.code_quality * weights.code_quality;
        total_weight += weights.code_quality;

        if total_weight == 0.0 {
            return Err(VIBEError::ScoringError(
                "No valid scoring criteria enabled".to_string(),
            ));
        }

        Ok(weighted_sum / total_weight)
    }

    /// Apply dynamic factors and scoring rules
    fn apply_dynamic_factors(
        &self,
        base_score: f32,
        validation_results: &HashMap<Platform, PlatformValidationResult>,
        criteria: &ValidationCriteria,
    ) -> Result<f32, VIBEError> {
        let mut adjusted_score = base_score;

        // Apply penalty rules
        for penalty_rule in &self.default_criteria.penalty_rules {
            if self.evaluate_penalty_conditions(penalty_rule, validation_results, criteria) {
                adjusted_score -= penalty_rule.penalty_amount;
            }
        }

        // Apply bonus rules
        for bonus_rule in &self.default_criteria.bonus_rules {
            if self.evaluate_bonus_conditions(bonus_rule, validation_results, criteria) {
                adjusted_score += bonus_rule.bonus_amount.min(bonus_rule.max_bonus);
            }
        }

        // Apply dynamic factors
        for dynamic_factor in self.default_criteria.dynamic_factors.values() {
            if self.evaluate_dynamic_factor_conditions(dynamic_factor, validation_results, criteria)
            {
                let factor_value =
                    self.calculate_dynamic_factor_value(dynamic_factor, validation_results)?;
                adjusted_score *= factor_value;
            }
        }

        Ok(adjusted_score.clamp(0.0, 100.0))
    }

    /// Calculate confidence interval for the score
    fn calculate_confidence_interval(
        &self,
        platform_scores: &HashMap<Platform, f32>,
    ) -> Result<Option<ConfidenceInterval>, VIBEError> {
        if platform_scores.len() < 2 {
            return Ok(None); // Need at least 2 data points for confidence interval
        }

        let scores: Vec<f32> = platform_scores.values().cloned().collect();
        let mean: f32 = scores.iter().sum::<f32>() / scores.len() as f32;

        let variance: f32 = scores
            .iter()
            .map(|&score| (score - mean).powi(2))
            .sum::<f32>()
            / scores.len() as f32;
        let std_dev = variance.sqrt();

        // 95% confidence interval
        let margin_of_error = 1.96 * std_dev / (scores.len() as f32).sqrt();

        Ok(Some(ConfidenceInterval {
            lower: (mean - margin_of_error).max(0.0),
            upper: (mean + margin_of_error).min(100.0),
            confidence_level: 0.95,
            sample_size: scores.len(),
        }))
    }

    /// Generate score metadata
    fn generate_score_metadata(
        &self,
        score: &f32,
        platform_scores: &HashMap<Platform, f32>,
        _config: &ValidationConfig,
    ) -> Result<ScoreMetadata, VIBEError> {
        let quality_gates = self.generate_quality_gates(score, platform_scores)?;

        let scoring_factors = self.calculate_scoring_factors(platform_scores)?;

        Ok(ScoreMetadata {
            timestamp: chrono::Utc::now(),
            algorithm_version: "VIBE-Score-v1.0".to_string(),
            validation_depth: self.determine_validation_depth(platform_scores),
            quality_gates,
            scoring_factors,
            normalization_applied: false,
        })
    }

    /// Apply score normalization if needed
    fn apply_normalization(&self, score: f32, metadata: &ScoreMetadata) -> Result<f32, VIBEError> {
        // Simple min-max normalization for now
        // In a real implementation, this could use historical data for more sophisticated normalization
        let normalized_score = score.clamp(0.0, 100.0);

        // Mark metadata as normalized if adjustment was made
        let mut metadata = metadata.clone();
        if (normalized_score - score).abs() > 0.01 {
            metadata.normalization_applied = true;
        }

        Ok(normalized_score)
    }

    // Helper methods for score extraction
    fn extract_logical_consistency_score(&self, result: &PlatformValidationResult) -> f32 {
        let logic_issues = result
            .issues
            .iter()
            .filter(|i| i.category == IssueCategory::LogicError)
            .count();

        let total_issues = result.issues.len();
        if total_issues == 0 {
            100.0
        } else {
            (1.0 - logic_issues as f32 / total_issues as f32) * 100.0
        }
    }

    fn extract_practical_score(&self, result: &PlatformValidationResult) -> f32 {
        // Simple heuristic based on validation status and issues
        match result.status {
            ValidationStatus::Passed => 85.0 + (result.score - 70.0),
            ValidationStatus::Warning => 70.0 + (result.score - 60.0),
            ValidationStatus::Failed => result.score * 0.8,
            ValidationStatus::Pending => 50.0,
        }
    }

    fn extract_compatibility_score(&self, result: &PlatformValidationResult) -> f32 {
        let compat_issues = result
            .issues
            .iter()
            .filter(|i| i.category == IssueCategory::CompatibilityProblem)
            .count();

        100.0 - (compat_issues as f32 * 15.0)
    }

    fn extract_security_score(&self, result: &PlatformValidationResult) -> f32 {
        let security_issues = result
            .issues
            .iter()
            .filter(|i| i.category == IssueCategory::SecurityVulnerability)
            .count();

        100.0 - (security_issues as f32 * 20.0)
    }

    fn extract_ux_score(&self, result: &PlatformValidationResult) -> f32 {
        let ux_issues = result
            .issues
            .iter()
            .filter(|i| i.category == IssueCategory::UIUXIssue)
            .count();

        100.0 - (ux_issues as f32 * 12.0)
    }

    fn extract_quality_score(&self, result: &PlatformValidationResult) -> f32 {
        let quality_issues = result
            .issues
            .iter()
            .filter(|i| i.category == IssueCategory::ErrorHandling)
            .count();

        100.0 - (quality_issues as f32 * 10.0)
    }

    /// Apply platform-specific adjustments
    fn apply_platform_adjustments(
        &self,
        score: f32,
        result: &PlatformValidationResult,
        config: &PlatformScoringConfig,
    ) -> Result<f32, VIBEError> {
        let mut adjusted_score = score;

        // Apply threshold-based adjustments
        for (metric, threshold) in &config.penalty_thresholds {
            let metric_value = match metric.as_str() {
                "error_rate" => result.performance_metrics.error_rate_percent,
                "response_time" => {
                    result.performance_metrics.average_response_time_ms as f32 / 1000.0
                }
                "memory_usage" => result.performance_metrics.memory_usage_mb as f32 / 1000.0,
                _ => 0.0,
            };

            if metric_value > *threshold {
                adjusted_score *= 0.9; // 10% penalty
            }
        }

        Ok(adjusted_score)
    }

    /// Apply custom platform configuration
    fn apply_custom_platform_config(
        &self,
        score: f32,
        config: &PlatformConfig,
    ) -> Result<f32, VIBEError> {
        let mut adjusted_score = score;

        // Apply custom criteria weights if available
        if let Some(criteria_weights) = &config.criteria_weights {
            // Simple adjustment based on available weights
            if let Some(logical_weight) = criteria_weights.logical_consistency {
                adjusted_score *= logical_weight;
            }
        }

        Ok(adjusted_score.clamp(0.0, 100.0))
    }

    /// Evaluate penalty rule conditions
    fn evaluate_penalty_conditions(
        &self,
        penalty_rule: &PenaltyRule,
        validation_results: &HashMap<Platform, PlatformValidationResult>,
        criteria: &ValidationCriteria,
    ) -> bool {
        // Simplified condition evaluation
        for condition in &penalty_rule.trigger_conditions {
            if self.evaluate_single_condition(condition, validation_results, criteria) {
                return true;
            }
        }
        false
    }

    /// Evaluate bonus rule conditions
    fn evaluate_bonus_conditions(
        &self,
        bonus_rule: &BonusRule,
        validation_results: &HashMap<Platform, PlatformValidationResult>,
        criteria: &ValidationCriteria,
    ) -> bool {
        // Simplified condition evaluation
        for condition in &bonus_rule.trigger_conditions {
            if self.evaluate_single_condition(condition, validation_results, criteria) {
                return true;
            }
        }
        false
    }

    /// Evaluate dynamic factor conditions
    fn evaluate_dynamic_factor_conditions(
        &self,
        dynamic_factor: &DynamicFactor,
        validation_results: &HashMap<Platform, PlatformValidationResult>,
        criteria: &ValidationCriteria,
    ) -> bool {
        // Simplified condition evaluation
        for condition in &dynamic_factor.conditions {
            if self.evaluate_single_condition(condition, validation_results, criteria) {
                return true;
            }
        }
        false
    }

    /// Evaluate a single scoring condition
    fn evaluate_single_condition(
        &self,
        condition: &ScoringCondition,
        validation_results: &HashMap<Platform, PlatformValidationResult>,
        _criteria: &ValidationCriteria,
    ) -> bool {
        // Simplified condition evaluation logic
        match condition.condition_type {
            ConditionType::Score => {
                let avg_score: f32 = validation_results.values().map(|r| r.score).sum::<f32>()
                    / validation_results.len() as f32;
                self.compare_values(avg_score, condition.operator, &condition.value)
            }
            ConditionType::IssueCount => {
                let total_issues: usize = validation_results.values().map(|r| r.issues.len()).sum();
                self.compare_values(total_issues as f32, condition.operator, &condition.value)
            }
            _ => false,
        }
    }

    /// Compare values using operator
    fn compare_values(
        &self,
        left: f32,
        operator: ConditionOperator,
        right: &ConditionValue,
    ) -> bool {
        match right {
            ConditionValue::Float(right_val) => match operator {
                ConditionOperator::Equals => (left - right_val).abs() < 0.01,
                ConditionOperator::GreaterThan => left > *right_val,
                ConditionOperator::LessThan => left < *right_val,
                ConditionOperator::GreaterEqual => left >= *right_val,
                ConditionOperator::LessEqual => left <= *right_val,
                _ => false,
            },
            _ => false,
        }
    }

    /// Calculate dynamic factor value
    fn calculate_dynamic_factor_value(
        &self,
        dynamic_factor: &DynamicFactor,
        _validation_results: &HashMap<Platform, PlatformValidationResult>,
    ) -> Result<f32, VIBEError> {
        let base_value = 1.0;

        match dynamic_factor.calculation_method {
            CalculationMethod::Linear => Ok(base_value + dynamic_factor.weight * 0.1),
            CalculationMethod::Exponential => Ok((base_value + dynamic_factor.weight).powf(1.1)),
            CalculationMethod::Logarithmic => Ok(1.0 + (dynamic_factor.weight * 0.1).ln()),
            _ => Ok(base_value),
        }
    }

    /// Generate quality gates for the score
    fn generate_quality_gates(
        &self,
        score: &f32,
        platform_scores: &HashMap<Platform, f32>,
    ) -> Result<Vec<QualityGate>, VIBEError> {
        let mut gates = Vec::new();

        // Overall score gate
        gates.push(QualityGate {
            gate_name: "Overall Score".to_string(),
            passed: *score >= 70.0,
            score_impact: if *score >= 70.0 { 0.0 } else { 20.0 },
            description: "Overall VIBE score must be at least 70".to_string(),
        });

        // Platform consistency gate
        if platform_scores.len() > 1 {
            let scores: Vec<f32> = platform_scores.values().cloned().collect();
            let variance = self.calculate_variance(&scores);
            gates.push(QualityGate {
                gate_name: "Platform Consistency".to_string(),
                passed: variance < 100.0, // Low variance threshold
                score_impact: if variance < 100.0 { 0.0 } else { 10.0 },
                description: "Platform scores should be consistent".to_string(),
            });
        }

        Ok(gates)
    }

    /// Calculate variance for a set of values
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        values
            .iter()
            .map(|&value| (value - mean).powi(2))
            .sum::<f32>()
            / values.len() as f32
    }

    /// Calculate scoring factors
    fn calculate_scoring_factors(
        &self,
        platform_scores: &HashMap<Platform, f32>,
    ) -> Result<HashMap<String, f32>, VIBEError> {
        let mut factors = HashMap::new();

        if !platform_scores.is_empty() {
            let avg_score: f32 =
                platform_scores.values().sum::<f32>() / platform_scores.len() as f32;
            factors.insert("average_platform_score".to_string(), avg_score);

            let score_range = platform_scores.values().cloned().fold(0.0, f32::max)
                - platform_scores.values().cloned().fold(100.0, f32::min);
            factors.insert("score_range".to_string(), score_range);
        }

        Ok(factors)
    }

    /// Determine validation depth based on platform scores
    fn determine_validation_depth(
        &self,
        platform_scores: &HashMap<Platform, f32>,
    ) -> ValidationDepth {
        let avg_score: f32 = platform_scores.values().sum::<f32>() / platform_scores.len() as f32;

        match avg_score {
            score if score >= 85.0 => ValidationDepth::Exhaustive,
            score if score >= 75.0 => ValidationDepth::Comprehensive,
            score if score >= 65.0 => ValidationDepth::Standard,
            _ => ValidationDepth::Basic,
        }
    }

    /// Update score history
    fn update_score_history(
        &mut self,
        _score: &f32,
        platform_scores: &HashMap<Platform, f32>,
        _validation_results: &HashMap<Platform, PlatformValidationResult>,
    ) -> Result<(), VIBEError> {
        let protocol_id = Uuid::new_v4(); // Generate unique ID for this validation

        for (platform, platform_score) in platform_scores {
            self.score_history.scores.push(HistoricalScore {
                timestamp: chrono::Utc::now(),
                score: *platform_score,
                platform: *platform,
                protocol_id,
            });
        }

        // Keep only last 1000 scores to prevent memory bloat
        let excess_scores = self.score_history.scores.len().saturating_sub(1000);
        if excess_scores > 0 {
            self.score_history.scores.drain(0..excess_scores);
        }

        Ok(())
    }

    /// Create default scoring criteria
    fn create_default_criteria() -> ScoringCriteria {
        ScoringCriteria {
            base_weights: ScoringWeights::default(),
            platform_adjustments: HashMap::new(),
            dynamic_factors: HashMap::new(),
            penalty_rules: Vec::new(),
            bonus_rules: Vec::new(),
        }
    }

    /// Create platform-specific configurations
    fn create_platform_configs() -> HashMap<Platform, PlatformScoringConfig> {
        let mut configs = HashMap::new();

        // Web platform configuration
        configs.insert(
            Platform::Web,
            PlatformScoringConfig {
                platform: Platform::Web,
                base_weights: ScoringWeights {
                    logical_consistency: 0.20,
                    practical_applicability: 0.20,
                    platform_compatibility: 0.15,
                    performance_requirements: 0.20,
                    security_considerations: 0.15,
                    user_experience: 0.10,
                    code_quality: 0.00,
                    custom_weights: HashMap::new(),
                },
                penalty_thresholds: HashMap::from([
                    ("response_time".to_string(), 3.0),
                    ("error_rate".to_string(), 5.0),
                ]),
                bonus_multipliers: HashMap::new(),
                normalization_method: NormalizationMethod::MinMax,
            },
        );

        // Add similar configs for other platforms...

        configs
    }
}

impl Default for ScoringEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vibe::validation::PlatformPerformanceMetrics;

    #[test]
    fn test_scoring_engine_creation() {
        let engine = ScoringEngine::new();
        assert!(!engine
            .default_criteria
            .base_weights
            .logical_consistency
            .is_nan());
    }

    #[test]
    fn test_vibe_score_calculation() {
        let mut engine = ScoringEngine::new();
        let mut validation_results = HashMap::new();

        // Add a mock validation result
        validation_results.insert(
            Platform::Web,
            PlatformValidationResult {
                platform: Platform::Web,
                score: 80.0,
                status: ValidationStatus::Passed,
                issues: vec![],
                performance_metrics: PlatformPerformanceMetrics {
                    average_response_time_ms: 1500,
                    memory_usage_mb: 100,
                    cpu_usage_percent: 25.0,
                    error_rate_percent: 2.0,
                    throughput_requests_per_second: 10.0,
                },
                recommendations: vec![],
            },
        );

        let criteria = ValidationCriteria {
            logical_consistency: true,
            practical_applicability: true,
            platform_compatibility: true,
            performance_requirements: false,
            security_considerations: false,
            user_experience: false,
            custom_metrics: HashMap::new(),
        };

        let config = ValidationConfig::default();

        let score = engine
            .calculate_vibe_score(&validation_results, &criteria, &config)
            .unwrap();
        assert!(score.overall_score >= 0.0);
        assert!(score.overall_score <= 100.0);
        assert!(score.platform_scores.contains_key(&Platform::Web));
    }

    #[test]
    fn test_component_scores_calculation() {
        let engine = ScoringEngine::new();
        let mut validation_results = HashMap::new();

        validation_results.insert(
            Platform::Web,
            PlatformValidationResult {
                platform: Platform::Web,
                score: 75.0,
                status: ValidationStatus::Passed,
                issues: vec![ValidationIssue {
                    platform: Platform::Web,
                    severity: Severity::Medium,
                    category: IssueCategory::LogicError,
                    description: "Test logic issue".to_string(),
                    location: None,
                    suggestion: None,
                }],
                performance_metrics: PlatformPerformanceMetrics {
                    average_response_time_ms: 1000,
                    memory_usage_mb: 80,
                    cpu_usage_percent: 20.0,
                    error_rate_percent: 1.0,
                    throughput_requests_per_second: 15.0,
                },
                recommendations: vec![],
            },
        );

        let criteria = ValidationCriteria {
            logical_consistency: true,
            practical_applicability: true,
            platform_compatibility: true,
            performance_requirements: false,
            security_considerations: false,
            user_experience: false,
            custom_metrics: HashMap::new(),
        };

        let component_scores = engine
            .calculate_component_scores(&validation_results, &criteria)
            .unwrap();
        assert!(component_scores.logical_consistency <= 100.0);
        assert!(component_scores.practical_applicability >= 0.0);
    }
}
