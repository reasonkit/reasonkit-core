//! Profile Optimizer for MiniMax M2 ThinkTools
//!
//! Optimizes ThinkTool execution based on confidence profiles:
//! - Quick (70% confidence)
//! - Balanced (80% confidence)
//! - Deep (85% confidence)
//! - Paranoid (95% confidence)

use super::M2ThinkToolResult;
use serde::{Deserialize, Serialize};

/// Profile types for ThinkTool execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ProfileType {
    Quick,
    Balanced,
    Deep,
    Paranoid,
}

impl ProfileType {
    /// Get target confidence level for profile
    pub fn target_confidence(&self) -> f64 {
        match self {
            ProfileType::Quick => 0.70,
            ProfileType::Balanced => 0.80,
            ProfileType::Deep => 0.85,
            ProfileType::Paranoid => 0.95,
        }
    }

    /// Get optimization strategy for profile
    pub fn optimization_strategy(&self) -> OptimizationStrategy {
        match self {
            ProfileType::Quick => OptimizationStrategy::SpeedFirst,
            ProfileType::Balanced => OptimizationStrategy::BalancedSpeedQuality,
            ProfileType::Deep => OptimizationStrategy::QualityFirst,
            ProfileType::Paranoid => OptimizationStrategy::MaximumQuality,
        }
    }

    /// Get resource allocation for profile
    pub fn resource_allocation(&self) -> ResourceAllocation {
        match self {
            ProfileType::Quick => ResourceAllocation {
                max_time_ms: 3000,
                max_tokens: 1500,
                validation_rounds: 1,
                parallel_tasks: 1,
            },
            ProfileType::Balanced => ResourceAllocation {
                max_time_ms: 4500,
                max_tokens: 2000,
                validation_rounds: 2,
                parallel_tasks: 2,
            },
            ProfileType::Deep => ResourceAllocation {
                max_time_ms: 6000,
                max_tokens: 2500,
                validation_rounds: 3,
                parallel_tasks: 3,
            },
            ProfileType::Paranoid => ResourceAllocation {
                max_time_ms: 8000,
                max_tokens: 3000,
                validation_rounds: 5,
                parallel_tasks: 4,
            },
        }
    }
}

/// Confidence target for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceTarget {
    pub profile: ProfileType,
    pub target_confidence: f64,
    pub minimum_confidence: f64,
    pub confidence_variance: f64,
    pub optimization_priority: OptimizationPriority,
}

impl ConfidenceTarget {
    pub fn new(profile: ProfileType) -> Self {
        let target_confidence = profile.target_confidence();

        Self {
            profile: profile.clone(),
            target_confidence,
            minimum_confidence: target_confidence - 0.1,
            confidence_variance: 0.05,
            optimization_priority: match profile {
                ProfileType::Quick => OptimizationPriority::Speed,
                ProfileType::Balanced => OptimizationPriority::Balanced,
                ProfileType::Deep => OptimizationPriority::Quality,
                ProfileType::Paranoid => OptimizationPriority::MaximumQuality,
            },
        }
    }
}

/// Optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    SpeedFirst,
    BalancedSpeedQuality,
    QualityFirst,
    MaximumQuality,
}

/// Resource allocation for profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub max_time_ms: u32,
    pub max_tokens: u32,
    pub validation_rounds: u32,
    pub parallel_tasks: u32,
}

/// Optimization priorities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Speed,
    Balanced,
    Quality,
    MaximumQuality,
}

/// Profile optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub profile: ProfileType,
    pub confidence_multiplier: f64,
    pub performance_improvement: f64,
    pub resource_efficiency: f64,
    pub validation_coverage: f64,
    pub optimization_applied: Vec<OptimizationApplied>,
    pub quality_metrics: QualityMetrics,
}

impl OptimizationResult {
    pub fn new(profile: ProfileType) -> Self {
        let (confidence_multiplier, performance_improvement) = match profile {
            ProfileType::Quick => (0.9, 0.3), // Faster but slightly less confident
            ProfileType::Balanced => (1.0, 0.15), // Balanced improvement
            ProfileType::Deep => (1.1, 0.05), // Higher confidence with minor speed cost
            ProfileType::Paranoid => (1.2, -0.1), // Highest confidence with speed cost
        };

        Self {
            profile: profile.clone(),
            confidence_multiplier,
            performance_improvement,
            resource_efficiency: 1.0,
            validation_coverage: 0.8,
            optimization_applied: Vec::new(),
            quality_metrics: QualityMetrics::new(profile),
        }
    }
}

/// Optimization applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationApplied {
    pub optimization_type: String,
    pub description: String,
    pub impact_score: f64,
    pub applied: bool,
}

/// Quality metrics for profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub consistency: f64,
    pub robustness: f64,
}

impl QualityMetrics {
    pub fn new(profile: ProfileType) -> Self {
        let (accuracy, precision, recall, f1, consistency, robustness) = match profile {
            ProfileType::Quick => (0.74, 0.70, 0.80, 0.76, 0.70, 0.65),
            ProfileType::Balanced => (0.85, 0.80, 0.85, 0.82, 0.80, 0.75),
            ProfileType::Deep => (0.92, 0.88, 0.90, 0.89, 0.88, 0.85),
            ProfileType::Paranoid => (0.97, 0.95, 0.96, 0.95, 0.95, 0.92),
        };

        Self {
            accuracy,
            precision,
            recall,
            f1_score: f1,
            consistency,
            robustness,
        }
    }
}

/// Profile optimizer for M2 ThinkTools
pub struct ProfileOptimizer {
    pub optimization_history: Vec<OptimizationResult>,
    pub performance_baselines: std::collections::HashMap<ProfileType, PerformanceBaseline>,
}

impl Default for ProfileOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileOptimizer {
    pub fn new() -> Self {
        let mut baselines = std::collections::HashMap::new();
        baselines.insert(ProfileType::Quick, PerformanceBaseline::quick_baseline());
        baselines.insert(
            ProfileType::Balanced,
            PerformanceBaseline::balanced_baseline(),
        );
        baselines.insert(ProfileType::Deep, PerformanceBaseline::deep_baseline());
        baselines.insert(
            ProfileType::Paranoid,
            PerformanceBaseline::paranoid_baseline(),
        );

        Self {
            optimization_history: Vec::new(),
            performance_baselines: baselines,
        }
    }

    /// Optimize ThinkTool result for specific profile
    pub fn optimize_for_profile(
        &mut self,
        result: &M2ThinkToolResult,
        profile: ProfileType,
    ) -> Option<OptimizationResult> {
        let mut optimization = OptimizationResult::new(profile.clone());

        // Apply profile-specific optimizations
        self.apply_profile_optimizations(&mut optimization, result);

        // Calculate performance improvement
        optimization.performance_improvement =
            self.calculate_performance_improvement(&optimization, result);

        // Calculate resource efficiency
        optimization.resource_efficiency = self.calculate_resource_efficiency(result, &profile);

        // Store in history
        self.optimization_history.push(optimization.clone());

        Some(optimization)
    }

    /// Apply profile-specific optimizations
    fn apply_profile_optimizations(
        &self,
        optimization: &mut OptimizationResult,
        result: &M2ThinkToolResult,
    ) {
        match optimization.profile {
            ProfileType::Quick => {
                optimization.optimization_applied.push(OptimizationApplied {
                    optimization_type: "FastProcessing".to_string(),
                    description: "Skip detailed validation for speed".to_string(),
                    impact_score: 0.3,
                    applied: true,
                });
                optimization.validation_coverage = 0.6;
            }
            ProfileType::Balanced => {
                optimization.optimization_applied.push(OptimizationApplied {
                    optimization_type: "BalancedValidation".to_string(),
                    description: "Standard validation with quality checks".to_string(),
                    impact_score: 0.2,
                    applied: true,
                });
                optimization.optimization_applied.push(OptimizationApplied {
                    optimization_type: "ParallelProcessing".to_string(),
                    description: "Parallel execution for moderate speed improvement".to_string(),
                    impact_score: 0.15,
                    applied: true,
                });
                optimization.validation_coverage = 0.8;
            }
            ProfileType::Deep => {
                optimization.optimization_applied.push(OptimizationApplied {
                    optimization_type: "DeepValidation".to_string(),
                    description: "Comprehensive validation and cross-checking".to_string(),
                    impact_score: 0.25,
                    applied: true,
                });
                optimization.optimization_applied.push(OptimizationApplied {
                    optimization_type: "QualityEnhancement".to_string(),
                    description: "Enhanced reasoning depth and analysis".to_string(),
                    impact_score: 0.2,
                    applied: true,
                });
                optimization.validation_coverage = 0.9;
            }
            ProfileType::Paranoid => {
                optimization.optimization_applied.push(OptimizationApplied {
                    optimization_type: "MaximumValidation".to_string(),
                    description: "Multiple validation rounds and exhaustive checking".to_string(),
                    impact_score: 0.4,
                    applied: true,
                });
                optimization.optimization_applied.push(OptimizationApplied {
                    optimization_type: "AdversarialTesting".to_string(),
                    description: "Adversarial analysis to identify hidden flaws".to_string(),
                    impact_score: 0.3,
                    applied: true,
                });
                optimization.optimization_applied.push(OptimizationApplied {
                    optimization_type: "ConsensusBuilding".to_string(),
                    description: "Multiple perspective consensus for maximum reliability"
                        .to_string(),
                    impact_score: 0.25,
                    applied: true,
                });
                optimization.validation_coverage = 0.95;
            }
        }

        // Adjust confidence multiplier based on constraint adherence
        if let super::ConstraintResult::Passed(score) = &result.constraint_adherence {
            optimization.confidence_multiplier *= score;
        }
    }

    /// Calculate performance improvement
    fn calculate_performance_improvement(
        &self,
        optimization: &OptimizationResult,
        result: &M2ThinkToolResult,
    ) -> f64 {
        let baseline = self
            .performance_baselines
            .get(&optimization.profile)
            .cloned()
            .unwrap_or_default();

        let current_performance = self.calculate_current_performance(result);
        let baseline_performance = baseline.average_confidence;

        if baseline_performance > 0.0 {
            (current_performance - baseline_performance) / baseline_performance
        } else {
            0.0
        }
    }

    /// Calculate resource efficiency
    fn calculate_resource_efficiency(
        &self,
        result: &M2ThinkToolResult,
        profile: &ProfileType,
    ) -> f64 {
        let resource_allocation = profile.resource_allocation();
        let time_efficiency =
            resource_allocation.max_time_ms as f64 / result.processing_time_ms as f64;
        let token_efficiency = resource_allocation.max_tokens as f64 / result.token_count as f64;

        (time_efficiency + token_efficiency) / 2.0
    }

    /// Calculate current performance score
    fn calculate_current_performance(&self, result: &M2ThinkToolResult) -> f64 {
        let confidence_score = result.confidence;
        let constraint_score = match &result.constraint_adherence {
            super::ConstraintResult::Passed(score) => *score,
            _ => 0.5,
        };

        let validation_score = if !result.interleaved_steps.is_empty() {
            result
                .interleaved_steps
                .iter()
                .filter(|step| step.cross_validation_passed)
                .count() as f64
                / result.interleaved_steps.len() as f64
        } else {
            0.7
        };

        confidence_score * 0.4 + constraint_score * 0.3 + validation_score * 0.3
    }

    /// Get optimization recommendations for profile
    pub fn get_recommendations(&self, profile: ProfileType) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        match profile {
            ProfileType::Quick => {
                recommendations.push(OptimizationRecommendation {
                    category: "Speed".to_string(),
                    recommendation: "Consider caching common patterns to reduce processing time"
                        .to_string(),
                    impact: "High".to_string(),
                    difficulty: "Low".to_string(),
                });
            }
            ProfileType::Balanced => {
                recommendations.push(OptimizationRecommendation {
                    category: "Quality".to_string(),
                    recommendation: "Implement adaptive validation based on content complexity"
                        .to_string(),
                    impact: "Medium".to_string(),
                    difficulty: "Medium".to_string(),
                });
            }
            ProfileType::Deep => {
                recommendations.push(OptimizationRecommendation {
                    category: "Depth".to_string(),
                    recommendation: "Add specialized reasoning modules for domain-specific analysis".to_string(),
                    impact: "High".to_string(),
                    difficulty: "High".to_string(),
                });
            }
            ProfileType::Paranoid => {
                recommendations.push(OptimizationRecommendation {
                    category: "Reliability".to_string(),
                    recommendation: "Implement consensus mechanisms across multiple reasoning paths".to_string(),
                    impact: "Very High".to_string(),
                    difficulty: "Very High".to_string(),
                });
            }
        }

        recommendations
    }

    /// Analyze optimization trends
    pub fn analyze_optimization_trends(&self) -> OptimizationTrends {
        let recent_optimizations: Vec<_> =
            self.optimization_history.iter().rev().take(10).collect();

        let avg_confidence_improvement = if !recent_optimizations.is_empty() {
            recent_optimizations
                .iter()
                .map(|opt| opt.performance_improvement)
                .sum::<f64>()
                / recent_optimizations.len() as f64
        } else {
            0.0
        };

        let most_effective_profile = self.find_most_effective_profile();

        OptimizationTrends {
            total_optimizations: self.optimization_history.len(),
            average_confidence_improvement: avg_confidence_improvement,
            most_effective_profile,
            optimization_velocity: self.calculate_optimization_velocity(),
            recommendations_count: self
                .optimization_history
                .iter()
                .map(|opt| opt.optimization_applied.len())
                .sum(),
        }
    }

    /// Find most effective profile based on historical data
    fn find_most_effective_profile(&self) -> ProfileType {
        let mut profile_scores = std::collections::HashMap::new();

        for optimization in &self.optimization_history {
            let score =
                optimization.performance_improvement + optimization.resource_efficiency * 0.5;
            *profile_scores
                .entry(optimization.profile.clone())
                .or_insert(0.0) += score;
        }

        profile_scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(profile, _)| profile)
            .unwrap_or(ProfileType::Balanced)
    }

    /// Calculate optimization velocity (improvements per time period)
    fn calculate_optimization_velocity(&self) -> f64 {
        if self.optimization_history.len() < 2 {
            return 0.0;
        }

        let recent_count = self.optimization_history.len().min(5);
        let recent: Vec<_> = self
            .optimization_history
            .iter()
            .rev()
            .take(recent_count)
            .collect();

        recent
            .windows(2)
            .map(|window| {
                let current = &window[0];
                let previous = &window[1];
                current.performance_improvement - previous.performance_improvement
            })
            .sum::<f64>()
            / (recent_count - 1) as f64
    }
}

/// Performance baseline for profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub average_confidence: f64,
    pub average_processing_time_ms: u32,
    pub average_token_count: u32,
    pub success_rate: f64,
    pub validation_coverage: f64,
}

impl PerformanceBaseline {
    pub fn quick_baseline() -> Self {
        Self {
            average_confidence: 0.70,
            average_processing_time_ms: 3000,
            average_token_count: 1500,
            success_rate: 0.85,
            validation_coverage: 0.6,
        }
    }

    pub fn balanced_baseline() -> Self {
        Self {
            average_confidence: 0.80,
            average_processing_time_ms: 4500,
            average_token_count: 2000,
            success_rate: 0.90,
            validation_coverage: 0.8,
        }
    }

    pub fn deep_baseline() -> Self {
        Self {
            average_confidence: 0.85,
            average_processing_time_ms: 6000,
            average_token_count: 2500,
            success_rate: 0.95,
            validation_coverage: 0.9,
        }
    }

    pub fn paranoid_baseline() -> Self {
        Self {
            average_confidence: 0.95,
            average_processing_time_ms: 8000,
            average_token_count: 3000,
            success_rate: 0.98,
            validation_coverage: 0.95,
        }
    }
}

impl Default for PerformanceBaseline {
    fn default() -> Self {
        Self::balanced_baseline()
    }
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub category: String,
    pub recommendation: String,
    pub impact: String,
    pub difficulty: String,
}

/// Optimization trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTrends {
    pub total_optimizations: usize,
    pub average_confidence_improvement: f64,
    pub most_effective_profile: ProfileType,
    pub optimization_velocity: f64,
    pub recommendations_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_confidence_levels() {
        assert_eq!(ProfileType::Quick.target_confidence(), 0.70);
        assert_eq!(ProfileType::Balanced.target_confidence(), 0.80);
        assert_eq!(ProfileType::Deep.target_confidence(), 0.85);
        assert_eq!(ProfileType::Paranoid.target_confidence(), 0.95);
    }

    #[test]
    fn test_resource_allocation() {
        let quick_resources = ProfileType::Quick.resource_allocation();
        assert_eq!(quick_resources.max_time_ms, 3000);
        assert_eq!(quick_resources.validation_rounds, 1);

        let paranoid_resources = ProfileType::Paranoid.resource_allocation();
        assert_eq!(paranoid_resources.max_time_ms, 8000);
        assert_eq!(paranoid_resources.validation_rounds, 5);
    }

    #[test]
    fn test_optimization_result_creation() {
        let optimization = OptimizationResult::new(ProfileType::Balanced);
        assert_eq!(optimization.profile, ProfileType::Balanced);
        assert_eq!(optimization.confidence_multiplier, 1.0);
        assert_eq!(optimization.performance_improvement, 0.15);
    }

    #[test]
    fn test_quality_metrics() {
        let quick_metrics = QualityMetrics::new(ProfileType::Quick);
        assert!(quick_metrics.accuracy < quick_metrics.f1_score);

        let paranoid_metrics = QualityMetrics::new(ProfileType::Paranoid);
        assert_eq!(paranoid_metrics.accuracy, 0.97);
    }

    #[test]
    fn test_performance_baselines() {
        let baseline = PerformanceBaseline::quick_baseline();
        assert_eq!(baseline.average_confidence, 0.70);
        assert_eq!(baseline.average_processing_time_ms, 3000);
    }

    #[test]
    fn test_profile_optimizer() {
        let optimizer = ProfileOptimizer::new();
        assert_eq!(optimizer.performance_baselines.len(), 4);
    }
}
