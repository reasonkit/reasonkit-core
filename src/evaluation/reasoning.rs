//! # Reasoning Quality Metrics
//!
//! Core metrics for evaluating AI THINKING improvement with ReasonKit.
//!
//! This is the HEART of ReasonKit evaluation - measuring whether
//! ThinkTool protocols actually improve AI reasoning.

use std::collections::HashMap;

/// ReasonKit profiles (ThinkTool chains)
///
/// Profiles define which ThinkTools are used and in what order.
/// See `thinktool::profiles::ReasoningProfile` for the full chain configuration
/// including conditional execution and validation passes.
///
/// # Confidence Thresholds (per ORCHESTRATOR.md spec)
///
/// | Profile  | Min Confidence | Modules |
/// |----------|----------------|---------|
/// | Quick    | 70%            | gt, ll  |
/// | Balanced | 80%            | gt, ll, br, pg |
/// | Deep     | 85%            | All 5   |
/// | Paranoid | 95%            | All 5 + validation |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// No ThinkTools (baseline for A/B comparison)
    None,
    /// Quick: GigaThink -> LaserLogic (70% confidence target)
    /// Fast 2-step analysis for rapid insights
    Quick,
    /// Balanced: GigaThink -> LaserLogic -> BedRock -> ProofGuard (80% confidence target)
    /// Standard 4-module chain for thorough but efficient analysis
    Balanced,
    /// Deep: All 5 ThinkTools (85% confidence target)
    /// GigaThink -> LaserLogic -> BedRock -> ProofGuard -> BrutalHonesty (conditional)
    /// BrutalHonesty runs if confidence < 85%
    Deep,
    /// Paranoid: All 5 ThinkTools + validation pass (95% confidence target)
    /// GigaThink -> LaserLogic -> BedRock -> ProofGuard -> BrutalHonesty -> ProofGuard
    /// Maximum rigor with adversarial critique and second verification pass
    Paranoid,
}

impl Profile {
    /// Get the list of ThinkTools for this profile
    ///
    /// Note: This returns the *unique* tools used, not the full execution chain.
    /// For the actual execution chain (including conditional steps and validation passes),
    /// see `thinktool::profiles::ReasoningProfile`.
    pub fn thinktools(&self) -> Vec<ThinkTool> {
        match self {
            Profile::None => vec![],
            Profile::Quick => vec![ThinkTool::GigaThink, ThinkTool::LaserLogic],
            Profile::Balanced => vec![
                ThinkTool::GigaThink,
                ThinkTool::LaserLogic,
                ThinkTool::BedRock,
                ThinkTool::ProofGuard,
            ],
            Profile::Deep => vec![
                ThinkTool::GigaThink,
                ThinkTool::LaserLogic,
                ThinkTool::BedRock,
                ThinkTool::ProofGuard,
                ThinkTool::BrutalHonesty,
            ],
            Profile::Paranoid => vec![
                ThinkTool::GigaThink,
                ThinkTool::LaserLogic,
                ThinkTool::BedRock,
                ThinkTool::ProofGuard,
                ThinkTool::BrutalHonesty,
            ],
        }
    }

    /// Get the minimum confidence threshold for this profile
    ///
    /// Returns the confidence level required for the profile to be considered successful.
    /// Per ORCHESTRATOR.md specification:
    /// - Quick: 70%
    /// - Balanced: 80%
    /// - Deep: 85%
    /// - Paranoid: 95%
    pub fn min_confidence(&self) -> f64 {
        match self {
            Profile::None => 0.0,
            Profile::Quick => 0.70,
            Profile::Balanced => 0.80,
            Profile::Deep => 0.85,
            Profile::Paranoid => 0.95,
        }
    }

    /// Get the number of steps in the execution chain
    ///
    /// Note: Paranoid has 6 steps (includes 2nd ProofGuard validation pass)
    pub fn chain_length(&self) -> usize {
        match self {
            Profile::None => 0,
            Profile::Quick => 2,
            Profile::Balanced => 4,
            Profile::Deep => 5,
            Profile::Paranoid => 6, // Includes 2nd ProofGuard pass
        }
    }

    /// Convert from profile ID string
    pub fn from_id(id: &str) -> Option<Self> {
        match id.to_lowercase().as_str() {
            "none" | "baseline" => Some(Profile::None),
            "quick" => Some(Profile::Quick),
            "balanced" => Some(Profile::Balanced),
            "deep" => Some(Profile::Deep),
            "paranoid" => Some(Profile::Paranoid),
            _ => None,
        }
    }

    /// Get the profile ID string
    pub fn id(&self) -> &'static str {
        match self {
            Profile::None => "none",
            Profile::Quick => "quick",
            Profile::Balanced => "balanced",
            Profile::Deep => "deep",
            Profile::Paranoid => "paranoid",
        }
    }
}

/// Individual ThinkTools
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThinkTool {
    /// Multi-perspective expansion (10+ viewpoints)
    GigaThink,
    /// Precision deductive reasoning
    LaserLogic,
    /// First principles decomposition
    BedRock,
    /// Multi-source verification
    ProofGuard,
    /// Adversarial self-critique
    BrutalHonesty,
}

/// Result from running a benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name (e.g., "gsm8k", "arc_challenge")
    pub benchmark: String,
    /// Profile used
    pub profile: Profile,
    /// Accuracy (0.0-1.0)
    pub accuracy: f64,
    /// Number of correct answers
    pub correct: usize,
    /// Total questions
    pub total: usize,
    /// Per-question results
    pub question_results: Vec<QuestionResult>,
}

impl BenchmarkResult {
    /// Calculate improvement over a baseline
    pub fn improvement_over(&self, baseline: &BenchmarkResult) -> f64 {
        self.accuracy - baseline.accuracy
    }
}

/// Result for a single question
#[derive(Debug, Clone)]
pub struct QuestionResult {
    /// Question ID
    pub id: String,
    /// Whether answer was correct
    pub correct: bool,
    /// Confidence (0.0-1.0) if available
    pub confidence: Option<f64>,
    /// Answer given
    pub answer: String,
    /// Correct answer
    pub expected: String,
    /// Reasoning chain if captured
    pub reasoning: Option<String>,
}

/// Aggregated reasoning metrics
#[derive(Debug, Clone)]
pub struct ReasoningMetrics {
    /// Accuracy on benchmark
    pub accuracy: f64,
    /// Improvement over no-protocol baseline
    pub improvement: f64,
    /// Self-consistency metrics
    pub consistency: ConsistencyMetrics,
    /// Calibration metrics
    pub calibration: CalibrationMetrics,
    /// Per-ThinkTool effectiveness
    pub thinktool_metrics: HashMap<ThinkTool, ThinkToolMetrics>,
}

/// Self-consistency metrics
#[derive(Debug, Clone, Default)]
pub struct ConsistencyMetrics {
    /// Same answer across multiple runs (0.0-1.0)
    pub answer_agreement: f64,
    /// Same reasoning path across runs (0.0-1.0)
    pub reasoning_agreement: f64,
    /// Confidence variance (lower is better)
    pub confidence_variance: f64,
    /// Number of runs used to calculate
    pub num_runs: usize,
}

impl ConsistencyMetrics {
    /// Calculate from multiple runs of same questions
    pub fn from_runs(runs: &[Vec<QuestionResult>]) -> Self {
        if runs.is_empty() || runs[0].is_empty() {
            return Self::default();
        }

        let num_runs = runs.len();
        let num_questions = runs[0].len();
        let mut answer_agreements = 0;
        let mut confidence_sum = 0.0;
        let mut confidence_sq_sum = 0.0;
        let mut confidence_count = 0;

        for q_idx in 0..num_questions {
            // Check if all runs agree on this question
            let first_answer = &runs[0][q_idx].answer;
            let all_agree = runs.iter().all(|run| &run[q_idx].answer == first_answer);
            if all_agree {
                answer_agreements += 1;
            }

            // Collect confidences
            for run in runs {
                if let Some(conf) = run[q_idx].confidence {
                    confidence_sum += conf;
                    confidence_sq_sum += conf * conf;
                    confidence_count += 1;
                }
            }
        }

        let answer_agreement = answer_agreements as f64 / num_questions as f64;

        let confidence_variance = if confidence_count > 1 {
            let mean = confidence_sum / confidence_count as f64;
            (confidence_sq_sum / confidence_count as f64) - (mean * mean)
        } else {
            0.0
        };

        Self {
            answer_agreement,
            reasoning_agreement: 0.0, // Requires semantic comparison
            confidence_variance,
            num_runs,
        }
    }
}

/// Calibration metrics (confidence vs accuracy)
#[derive(Debug, Clone, Default)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error (lower is better)
    pub ece: f64,
    /// Overconfidence rate (high confidence + wrong)
    pub overconfidence_rate: f64,
    /// Underconfidence rate (low confidence + right)
    pub underconfidence_rate: f64,
    /// Brier score (lower is better)
    pub brier_score: f64,
}

impl CalibrationMetrics {
    /// Calculate calibration from results with confidence scores
    pub fn from_results(results: &[QuestionResult]) -> Self {
        let with_confidence: Vec<_> = results.iter().filter(|r| r.confidence.is_some()).collect();

        if with_confidence.is_empty() {
            return Self::default();
        }

        // Bin results by confidence for ECE
        let num_bins = 10;
        let mut bins: Vec<Vec<(f64, bool)>> = vec![vec![]; num_bins];

        for result in &with_confidence {
            let conf = result.confidence.unwrap();
            let bin_idx = ((conf * num_bins as f64) as usize).min(num_bins - 1);
            bins[bin_idx].push((conf, result.correct));
        }

        // Calculate ECE
        let n = with_confidence.len() as f64;
        let mut ece = 0.0;
        for bin in &bins {
            if !bin.is_empty() {
                let bin_size = bin.len() as f64;
                let avg_confidence: f64 = bin.iter().map(|(c, _)| c).sum::<f64>() / bin_size;
                let accuracy: f64 =
                    bin.iter().filter(|(_, correct)| *correct).count() as f64 / bin_size;
                ece += (bin_size / n) * (avg_confidence - accuracy).abs();
            }
        }

        // Overconfidence: confidence > 0.8 but wrong
        let overconfident = with_confidence
            .iter()
            .filter(|r| r.confidence.unwrap() > 0.8 && !r.correct)
            .count();
        let overconfidence_rate = overconfident as f64 / with_confidence.len() as f64;

        // Underconfidence: confidence < 0.5 but correct
        let underconfident = with_confidence
            .iter()
            .filter(|r| r.confidence.unwrap() < 0.5 && r.correct)
            .count();
        let underconfidence_rate = underconfident as f64 / with_confidence.len() as f64;

        // Brier score
        let brier_score: f64 = with_confidence
            .iter()
            .map(|r| {
                let conf = r.confidence.unwrap();
                let outcome = if r.correct { 1.0 } else { 0.0 };
                (conf - outcome).powi(2)
            })
            .sum::<f64>()
            / with_confidence.len() as f64;

        Self {
            ece,
            overconfidence_rate,
            underconfidence_rate,
            brier_score,
        }
    }
}

/// Generic ThinkTool effectiveness metrics
#[derive(Debug, Clone, Default)]
pub struct ThinkToolMetrics {
    /// Improvement delta when this tool is added
    pub improvement_delta: f64,
    /// Is this tool worth the latency cost?
    pub cost_effective: bool,
    /// Latency added (ms)
    pub latency_ms: f64,
}

/// GigaThink-specific metrics
#[derive(Debug, Clone, Default)]
pub struct GigaThinkMetrics {
    /// Number of distinct perspectives generated
    pub perspective_count: usize,
    /// Coverage of relevant angles (0.0-1.0)
    pub coverage_score: f64,
    /// Proportion of non-obvious perspectives (0.0-1.0)
    pub novelty_rate: f64,
    /// How well perspectives integrate (0.0-1.0)
    pub integration_quality: f64,
}

/// LaserLogic-specific metrics
#[derive(Debug, Clone, Default)]
pub struct LaserLogicMetrics {
    /// Proportion of valid deductions (0.0-1.0)
    pub validity_rate: f64,
    /// Rate of detecting inserted fallacies (0.0-1.0)
    pub fallacy_detection_rate: f64,
    /// Avoidance of irrelevant premises (0.0-1.0)
    pub precision: f64,
    /// Valid deductions from true premises (0.0-1.0)
    pub soundness: f64,
}

/// BedRock-specific metrics
#[derive(Debug, Clone, Default)]
pub struct BedRockMetrics {
    /// Levels of first-principles breakdown
    pub decomposition_depth: usize,
    /// Proportion of truly fundamental axioms (0.0-1.0)
    pub axiom_validity: f64,
    /// Can rebuild conclusion from axioms? (0.0-1.0)
    pub reconstruction_rate: f64,
    /// Hidden assumptions made explicit (0.0-1.0)
    pub assumption_surfacing: f64,
}

/// ProofGuard-specific metrics
#[derive(Debug, Clone, Default)]
pub struct ProofGuardMetrics {
    /// Proportion of claims with 3+ sources (0.0-1.0)
    pub triangulation_rate: f64,
    /// Rate of detecting conflicting sources (0.0-1.0)
    pub contradiction_detection: f64,
    /// Tier 1 source priority adherence (0.0-1.0)
    pub source_quality_score: f64,
    /// Correct attribution rate (0.0-1.0)
    pub citation_accuracy: f64,
}

/// BrutalHonesty-specific metrics
#[derive(Debug, Clone, Default)]
pub struct BrutalHonestyMetrics {
    /// Proportion of real flaws identified (0.0-1.0)
    pub flaw_detection_rate: f64,
    /// Proportion of non-flaws flagged (lower is better)
    pub false_positive_rate: f64,
    /// Average actionable suggestions per flaw
    pub suggestions_per_flaw: f64,
    /// Correct severity prioritization (0.0-1.0)
    pub severity_calibration: f64,
}

/// Calculate improvement delta for a ThinkTool
pub fn calculate_thinktool_delta(without: &BenchmarkResult, with: &BenchmarkResult) -> f64 {
    with.accuracy - without.accuracy
}

/// Statistical significance test (simplified)
pub fn is_significant(delta: f64, n: usize, alpha: f64) -> bool {
    // Approximate significance test
    // For proper testing, would use bootstrap or permutation test
    let se = (0.25 / n as f64).sqrt(); // Worst-case SE for proportions
    let z = delta / se;
    let critical = if alpha <= 0.01 {
        2.576
    } else if alpha <= 0.05 {
        1.96
    } else {
        1.645
    };
    z.abs() > critical
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_thinktools() {
        assert!(Profile::None.thinktools().is_empty());
        assert_eq!(Profile::Quick.thinktools().len(), 2);
        assert_eq!(Profile::Balanced.thinktools().len(), 4);
        assert_eq!(Profile::Deep.thinktools().len(), 5);
        assert_eq!(Profile::Paranoid.thinktools().len(), 5); // Unique tools (not chain length)
    }

    #[test]
    fn test_profile_min_confidence() {
        assert_eq!(Profile::None.min_confidence(), 0.0);
        assert_eq!(Profile::Quick.min_confidence(), 0.70);
        assert_eq!(Profile::Balanced.min_confidence(), 0.80);
        assert_eq!(Profile::Deep.min_confidence(), 0.85);
        assert_eq!(Profile::Paranoid.min_confidence(), 0.95);
    }

    #[test]
    fn test_profile_chain_length() {
        assert_eq!(Profile::None.chain_length(), 0);
        assert_eq!(Profile::Quick.chain_length(), 2);
        assert_eq!(Profile::Balanced.chain_length(), 4);
        assert_eq!(Profile::Deep.chain_length(), 5);
        assert_eq!(Profile::Paranoid.chain_length(), 6); // Includes 2nd ProofGuard pass
    }

    #[test]
    fn test_profile_from_id() {
        assert_eq!(Profile::from_id("quick"), Some(Profile::Quick));
        assert_eq!(Profile::from_id("BALANCED"), Some(Profile::Balanced));
        assert_eq!(Profile::from_id("paranoid"), Some(Profile::Paranoid));
        assert_eq!(Profile::from_id("baseline"), Some(Profile::None));
        assert_eq!(Profile::from_id("invalid"), None);
    }

    #[test]
    fn test_profile_id() {
        assert_eq!(Profile::Quick.id(), "quick");
        assert_eq!(Profile::Balanced.id(), "balanced");
        assert_eq!(Profile::Deep.id(), "deep");
        assert_eq!(Profile::Paranoid.id(), "paranoid");
    }

    #[test]
    fn test_improvement_calculation() {
        let baseline = BenchmarkResult {
            benchmark: "gsm8k".into(),
            profile: Profile::None,
            accuracy: 0.57,
            correct: 57,
            total: 100,
            question_results: vec![],
        };

        let treatment = BenchmarkResult {
            benchmark: "gsm8k".into(),
            profile: Profile::Balanced,
            accuracy: 0.78,
            correct: 78,
            total: 100,
            question_results: vec![],
        };

        let improvement = treatment.improvement_over(&baseline);
        assert!((improvement - 0.21).abs() < 0.001);
    }

    #[test]
    fn test_consistency_from_runs() {
        let runs = vec![
            vec![QuestionResult {
                id: "q1".into(),
                correct: true,
                confidence: Some(0.9),
                answer: "42".into(),
                expected: "42".into(),
                reasoning: None,
            }],
            vec![QuestionResult {
                id: "q1".into(),
                correct: true,
                confidence: Some(0.85),
                answer: "42".into(),
                expected: "42".into(),
                reasoning: None,
            }],
        ];

        let consistency = ConsistencyMetrics::from_runs(&runs);
        assert_eq!(consistency.answer_agreement, 1.0);
        assert_eq!(consistency.num_runs, 2);
    }

    #[test]
    fn test_calibration_ece() {
        // Perfect calibration: 80% confident, 80% correct
        let results: Vec<QuestionResult> = (0..100)
            .map(|i| QuestionResult {
                id: format!("q{}", i),
                correct: i < 80, // 80 correct
                confidence: Some(0.8),
                answer: "x".into(),
                expected: if i < 80 { "x" } else { "y" }.into(),
                reasoning: None,
            })
            .collect();

        let calibration = CalibrationMetrics::from_results(&results);
        // ECE should be low for well-calibrated predictions
        assert!(calibration.ece < 0.1);
    }

    #[test]
    fn test_significance() {
        // Large improvement with large N should be significant
        assert!(is_significant(0.10, 1000, 0.05));
        // Small improvement with small N should not be
        assert!(!is_significant(0.02, 50, 0.05));
    }
}
