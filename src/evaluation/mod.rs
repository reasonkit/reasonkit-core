//! # Evaluation Module
//!
//! Provides quality metrics for **AI REASONING EVALUATION**.
//!
//! ReasonKit measures whether AI THINKS BETTER with ThinkTool protocols.
//!
//! ## Core Metrics (Tier 1 - Reasoning Quality)
//!
//! - **Accuracy**: Correct answers on reasoning benchmarks (GSM8K, MATH, ARC-C)
//! - **Improvement Delta**: Performance with vs without ThinkTools
//! - **Self-Consistency**: Same answer across multiple runs
//! - **Calibration**: Confidence matches accuracy
//!
//! ## ThinkTool Metrics (Tier 2)
//!
//! - **GigaThink**: Perspective count, coverage, novelty
//! - **LaserLogic**: Validity rate, fallacy detection
//! - **BedRock**: Decomposition depth, axiom validity
//! - **ProofGuard**: Triangulation rate, contradiction detection
//! - **BrutalHonesty**: Flaw detection rate, improvement suggestions
//!
//! ## Supporting Metrics (Tier 5)
//!
//! - **Recall@K**: For source retrieval (ProofGuard support)
//! - **Latency**: Performance measurements
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::evaluation::{ReasoningMetrics, BenchmarkResult};
//!
//! // Run GSM8K benchmark with --balanced profile
//! let baseline = run_benchmark("gsm8k", None);
//! let treatment = run_benchmark("gsm8k", Some("balanced"));
//!
//! let improvement = treatment.accuracy - baseline.accuracy;
//! println!("Improvement: {:.1}%", improvement * 100.0);
//! ```

pub mod metrics;
pub mod reasoning;

pub use metrics::{
    average_precision, mean_average_precision, mean_reciprocal_rank, ndcg_at_k, precision_at_k,
    recall_at_k, EvaluationResult, QueryResult, RetrievalMetrics,
};

pub use reasoning::{
    BedRockMetrics, BenchmarkResult, BrutalHonestyMetrics, CalibrationMetrics, ConsistencyMetrics,
    GigaThinkMetrics, LaserLogicMetrics, Profile, ProofGuardMetrics, ReasoningMetrics,
    ThinkToolMetrics,
};

/// Reasoning evaluation configuration
#[derive(Debug, Clone)]
pub struct ReasoningEvalConfig {
    /// Number of runs for self-consistency measurement
    pub consistency_runs: usize,
    /// Profile to evaluate
    pub profile: Profile,
    /// Benchmark datasets to run
    pub benchmarks: Vec<String>,
    /// Whether to measure calibration
    pub measure_calibration: bool,
    /// Whether to measure per-ThinkTool effectiveness
    pub measure_thinktool_effectiveness: bool,
}

impl Default for ReasoningEvalConfig {
    fn default() -> Self {
        Self {
            consistency_runs: 5,
            profile: Profile::Balanced,
            benchmarks: vec!["gsm8k".into(), "arc_challenge".into()],
            measure_calibration: true,
            measure_thinktool_effectiveness: true,
        }
    }
}

/// Retrieval evaluation configuration (Tier 5 - Supporting)
#[derive(Debug, Clone)]
pub struct RetrievalEvalConfig {
    /// K values for Recall@K, Precision@K, NDCG@K
    pub k_values: Vec<usize>,
    /// Whether to compute MRR
    pub compute_mrr: bool,
    /// Whether to compute MAP
    pub compute_map: bool,
}

impl Default for RetrievalEvalConfig {
    fn default() -> Self {
        Self {
            k_values: vec![5, 10, 20],
            compute_mrr: true,
            compute_map: true,
        }
    }
}

/// Run full reasoning evaluation
pub fn evaluate_reasoning(
    results: &[BenchmarkResult],
    config: &ReasoningEvalConfig,
) -> ReasoningEvalSummary {
    let mut summary = ReasoningEvalSummary::new(&config.benchmarks);

    for result in results {
        summary.add_result(result);
    }

    summary.finalize()
}

/// Summary of reasoning evaluation
#[derive(Debug, Clone)]
pub struct ReasoningEvalSummary {
    pub num_benchmarks: usize,
    /// Accuracy per benchmark
    pub accuracy: std::collections::HashMap<String, f64>,
    /// Improvement over baseline per benchmark
    pub improvement: std::collections::HashMap<String, f64>,
    /// Self-consistency rate
    pub self_consistency: f64,
    /// Expected Calibration Error
    pub calibration_ece: f64,
    /// ThinkTool effectiveness scores
    pub thinktool_scores: std::collections::HashMap<String, f64>,

    // Internal
    accuracy_sums: std::collections::HashMap<String, (f64, usize)>,
}

impl ReasoningEvalSummary {
    pub fn new(benchmarks: &[String]) -> Self {
        let mut accuracy = std::collections::HashMap::new();
        let mut improvement = std::collections::HashMap::new();
        let mut accuracy_sums = std::collections::HashMap::new();

        for b in benchmarks {
            accuracy.insert(b.clone(), 0.0);
            improvement.insert(b.clone(), 0.0);
            accuracy_sums.insert(b.clone(), (0.0, 0));
        }

        Self {
            num_benchmarks: benchmarks.len(),
            accuracy,
            improvement,
            self_consistency: 0.0,
            calibration_ece: 0.0,
            thinktool_scores: std::collections::HashMap::new(),
            accuracy_sums,
        }
    }

    fn add_result(&mut self, result: &BenchmarkResult) {
        if let Some((sum, count)) = self.accuracy_sums.get_mut(&result.benchmark) {
            *sum += result.accuracy;
            *count += 1;
        }
    }

    fn finalize(mut self) -> Self {
        for (benchmark, (sum, count)) in &self.accuracy_sums {
            if *count > 0 {
                self.accuracy
                    .insert(benchmark.clone(), sum / (*count as f64));
            }
        }
        self
    }

    /// Check if reasoning targets are met
    pub fn check_targets(&self, targets: &ReasoningTargets) -> TargetResult {
        let mut passed = true;
        let mut failures = Vec::new();

        // Check GSM8K improvement
        if let Some(&target) = targets.gsm8k_improvement.as_ref() {
            if let Some(&actual) = self.improvement.get("gsm8k") {
                if actual < target {
                    passed = false;
                    failures.push(format!(
                        "GSM8K improvement: {:.1}% < {:.1}%",
                        actual * 100.0,
                        target * 100.0
                    ));
                }
            }
        }

        // Check self-consistency
        if let Some(target) = targets.self_consistency {
            if self.self_consistency < target {
                passed = false;
                failures.push(format!(
                    "Self-consistency: {:.1}% < {:.1}%",
                    self.self_consistency * 100.0,
                    target * 100.0
                ));
            }
        }

        // Check calibration
        if let Some(target) = targets.calibration_ece_max {
            if self.calibration_ece > target {
                passed = false;
                failures.push(format!(
                    "Calibration ECE: {:.3} > {:.3}",
                    self.calibration_ece, target
                ));
            }
        }

        TargetResult { passed, failures }
    }
}

/// Reasoning quality targets for release gates
#[derive(Debug, Clone, Default)]
pub struct ReasoningTargets {
    pub gsm8k_improvement: Option<f64>,
    pub arc_c_improvement: Option<f64>,
    pub logiqa_improvement: Option<f64>,
    pub self_consistency: Option<f64>,
    pub calibration_ece_max: Option<f64>,
}

impl ReasoningTargets {
    /// V1.0 release targets (January 2026)
    pub fn v1_targets() -> Self {
        Self {
            gsm8k_improvement: Some(0.15),   // +15% improvement
            arc_c_improvement: Some(0.08),   // +8% improvement
            logiqa_improvement: None,        // v1.5
            self_consistency: Some(0.85),    // 85% consistency
            calibration_ece_max: Some(0.10), // ECE < 0.10
        }
    }

    /// V1.5 stretch targets
    pub fn v1_5_targets() -> Self {
        Self {
            gsm8k_improvement: Some(0.20),
            arc_c_improvement: Some(0.10),
            logiqa_improvement: Some(0.20),
            self_consistency: Some(0.90),
            calibration_ece_max: Some(0.08),
        }
    }
}

/// Result of target check
#[derive(Debug, Clone)]
pub struct TargetResult {
    pub passed: bool,
    pub failures: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_config_default() {
        let config = ReasoningEvalConfig::default();
        assert_eq!(config.consistency_runs, 5);
        assert!(config.measure_calibration);
    }

    #[test]
    fn test_v1_targets() {
        let targets = ReasoningTargets::v1_targets();
        assert_eq!(targets.gsm8k_improvement, Some(0.15));
        assert_eq!(targets.self_consistency, Some(0.85));
    }

    #[test]
    fn test_target_check_pass() {
        let mut summary = ReasoningEvalSummary::new(&["gsm8k".into()]);
        summary.improvement.insert("gsm8k".into(), 0.20);
        summary.self_consistency = 0.90;
        summary.calibration_ece = 0.05;

        let targets = ReasoningTargets::v1_targets();
        let result = summary.check_targets(&targets);

        assert!(result.passed);
        assert!(result.failures.is_empty());
    }

    #[test]
    fn test_target_check_fail() {
        let mut summary = ReasoningEvalSummary::new(&["gsm8k".into()]);
        summary.improvement.insert("gsm8k".into(), 0.10); // Below 0.15 target
        summary.self_consistency = 0.75; // Below 0.85 target
        summary.calibration_ece = 0.15; // Above 0.10 target

        let targets = ReasoningTargets::v1_targets();
        let result = summary.check_targets(&targets);

        assert!(!result.passed);
        assert_eq!(result.failures.len(), 3);
    }
}
