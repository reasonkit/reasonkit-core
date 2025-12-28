//! # Process Reward Model (PRM) for Step-by-Step Verification
//!
//! Implements step-level verification based on Math-Shepherd research
//! achieving +6.2% GSM8K improvement through granular reasoning validation.
//!
//! ## Scientific Foundation
//!
//! Based on:
//! - Math-Shepherd (Wang et al., 2024): Process reward models for math reasoning
//! - Let's Verify Step by Step (Lightman et al., 2023): Step-level human verification
//!
//! ## Key Concepts
//!
//! - **Outcome Reward Model (ORM)**: Scores only final answer correctness
//! - **Process Reward Model (PRM)**: Scores each reasoning step independently
//!
//! PRM advantages:
//! 1. Better credit assignment - identifies WHERE reasoning went wrong
//! 2. More training signal - learns from partial success
//! 3. Improved calibration - confidence per step
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::prm::{ProcessRewardModel, StepScore};
//!
//! let prm = ProcessRewardModel::new();
//! let steps = vec!["Step 1: Given x + 2 = 5", "Step 2: x = 5 - 2 = 3"];
//! let scores = prm.score_steps(&steps).await?;
//! ```

use serde::{Deserialize, Serialize};

/// Individual step score from PRM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepScore {
    /// Step index (0-based)
    pub step_index: usize,
    /// Step content
    pub step_content: String,
    /// Correctness probability (0.0 - 1.0)
    pub correctness: f32,
    /// Logical validity score
    pub logical_validity: f32,
    /// Relevance to problem score
    pub relevance: f32,
    /// Identified issues (if any)
    pub issues: Vec<StepIssue>,
    /// Whether this step should be revised
    pub needs_revision: bool,
}

/// Issue identified in a reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepIssue {
    pub issue_type: IssueType,
    pub description: String,
    pub severity: Severity,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueType {
    ArithmeticError,
    LogicalFallacy,
    MissingJustification,
    InvalidAssumption,
    Irrelevant,
    SkippedStep,
    CircularReasoning,
    Contradiction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Low,      // Minor issue, doesn't affect correctness
    Medium,   // Could lead to errors downstream
    High,     // Likely causes incorrect final answer
    Critical, // Definitely invalidates the reasoning
}

/// Result of PRM evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrmResult {
    /// Scores for each step
    pub step_scores: Vec<StepScore>,
    /// Overall process score (product of step scores)
    pub overall_score: f32,
    /// First problematic step index (if any)
    pub first_error_step: Option<usize>,
    /// Confidence in the final answer
    pub final_answer_confidence: f32,
    /// Whether the reasoning chain is sound
    pub is_sound: bool,
    /// Aggregated metrics
    pub metrics: PrmMetrics,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrmMetrics {
    pub total_steps: usize,
    pub correct_steps: usize,
    pub avg_correctness: f32,
    pub avg_logical_validity: f32,
    pub avg_relevance: f32,
    pub critical_issues: usize,
}

impl PrmResult {
    pub fn compute(step_scores: Vec<StepScore>) -> Self {
        if step_scores.is_empty() {
            return Self {
                step_scores: vec![],
                overall_score: 0.0,
                first_error_step: None,
                final_answer_confidence: 0.0,
                is_sound: false,
                metrics: PrmMetrics::default(),
            };
        }

        // Find first problematic step
        let first_error_step = step_scores
            .iter()
            .position(|s| s.needs_revision || s.correctness < 0.5);

        // Overall score = product of correctness (with floor)
        let overall_score = step_scores
            .iter()
            .map(|s| s.correctness.max(0.01))
            .product::<f32>();

        // Is sound if no critical issues and all steps >= 0.6 correctness
        let critical_issues = step_scores
            .iter()
            .flat_map(|s| s.issues.iter())
            .filter(|i| i.severity == Severity::Critical)
            .count();

        let is_sound = critical_issues == 0 && step_scores.iter().all(|s| s.correctness >= 0.6);

        // Final answer confidence considers path dependency
        let final_answer_confidence = if is_sound {
            step_scores.last().map(|s| s.correctness).unwrap_or(0.0) * overall_score.sqrt()
        } else {
            overall_score * 0.5 // Penalize unsound chains
        };

        let total_steps = step_scores.len();
        let correct_steps = step_scores.iter().filter(|s| s.correctness >= 0.7).count();

        let metrics = PrmMetrics {
            total_steps,
            correct_steps,
            avg_correctness: step_scores.iter().map(|s| s.correctness).sum::<f32>()
                / total_steps as f32,
            avg_logical_validity: step_scores.iter().map(|s| s.logical_validity).sum::<f32>()
                / total_steps as f32,
            avg_relevance: step_scores.iter().map(|s| s.relevance).sum::<f32>()
                / total_steps as f32,
            critical_issues,
        };

        Self {
            step_scores,
            overall_score,
            first_error_step,
            final_answer_confidence,
            is_sound,
            metrics,
        }
    }
}

/// Process Reward Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrmConfig {
    /// Minimum step correctness to continue
    pub min_step_correctness: f32,
    /// Whether to halt on critical issues
    pub halt_on_critical: bool,
    /// Maximum steps to evaluate
    pub max_steps: usize,
    /// Verification strategy
    pub strategy: VerificationStrategy,
}

impl Default for PrmConfig {
    fn default() -> Self {
        Self {
            min_step_correctness: 0.5,
            halt_on_critical: true,
            max_steps: 50,
            strategy: VerificationStrategy::Sequential,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStrategy {
    /// Verify steps one by one
    Sequential,
    /// Verify all steps in parallel
    Parallel,
    /// Verify in batches
    Batched { batch_size: usize },
    /// Only verify final step (ORM fallback)
    FinalOnly,
}

/// Verification prompt templates for different step types
pub struct VerificationPrompts;

impl VerificationPrompts {
    /// Generate verification prompt for a math step
    pub fn math_step(step: &str, context: &str, problem: &str) -> String {
        format!(
            r#"You are a mathematical reasoning verifier. Evaluate the following reasoning step.

PROBLEM: {problem}

PREVIOUS CONTEXT:
{context}

STEP TO VERIFY:
{step}

Evaluate this step on three dimensions (0.0-1.0):
1. CORRECTNESS: Is the mathematical operation/statement correct?
2. LOGICAL_VALIDITY: Does it follow logically from the previous steps?
3. RELEVANCE: Does it contribute to solving the problem?

Identify any issues:
- Arithmetic errors
- Invalid assumptions
- Missing justifications
- Logical fallacies

Respond in JSON:
{{
    "correctness": 0.0-1.0,
    "logical_validity": 0.0-1.0,
    "relevance": 0.0-1.0,
    "issues": [
        {{
            "issue_type": "ArithmeticError|LogicalFallacy|MissingJustification|InvalidAssumption|Irrelevant|SkippedStep|CircularReasoning|Contradiction",
            "description": "...",
            "severity": "Low|Medium|High|Critical",
            "suggested_fix": "..." or null
        }}
    ],
    "needs_revision": true/false
}}"#,
            problem = problem,
            context = context,
            step = step
        )
    }

    /// Generate verification prompt for a logical reasoning step
    pub fn logic_step(step: &str, context: &str, claim: &str) -> String {
        format!(
            r#"You are a logical reasoning verifier using formal logic principles.

CLAIM BEING ANALYZED: {claim}

PRIOR REASONING:
{context}

STEP TO VERIFY:
{step}

Evaluate using Toulmin model components:
- Does it provide valid GROUNDS (evidence)?
- Does it provide valid WARRANT (logical connection)?
- Are there unstated but necessary BACKING assumptions?
- What REBUTTALS might apply?

Rate on three dimensions (0.0-1.0):
1. CORRECTNESS: Is the logical step valid?
2. LOGICAL_VALIDITY: Is the inference sound?
3. RELEVANCE: Does it support or refute the claim?

Respond in JSON:
{{
    "correctness": 0.0-1.0,
    "logical_validity": 0.0-1.0,
    "relevance": 0.0-1.0,
    "issues": [...],
    "needs_revision": true/false
}}"#,
            claim = claim,
            context = context,
            step = step
        )
    }
}

/// Step parser to extract reasoning steps from LLM output
pub struct StepParser;

impl StepParser {
    /// Parse numbered steps (1. 2. 3. or Step 1: Step 2:)
    pub fn parse_numbered(text: &str) -> Vec<String> {
        let mut steps = Vec::new();
        let mut current_step = String::new();

        for line in text.lines() {
            let trimmed = line.trim();

            // Check for step markers
            let is_new_step = trimmed.starts_with(|c: char| c.is_ascii_digit())
                || trimmed.to_lowercase().starts_with("step ")
                || trimmed.starts_with("- ")
                || trimmed.starts_with("* ");

            if is_new_step && !current_step.is_empty() {
                steps.push(current_step.trim().to_string());
                current_step = String::new();
            }

            if !trimmed.is_empty() {
                if !current_step.is_empty() {
                    current_step.push(' ');
                }
                current_step.push_str(trimmed);
            }
        }

        if !current_step.is_empty() {
            steps.push(current_step.trim().to_string());
        }

        steps
    }

    /// Parse steps by sentence boundaries
    pub fn parse_sentences(text: &str) -> Vec<String> {
        let mut steps = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            current.push(c);

            // Sentence end markers
            if c == '.' || c == '!' || c == '?' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() && trimmed.len() > 10 {
                    steps.push(trimmed);
                }
                current.clear();
            }
        }

        if !current.trim().is_empty() && current.trim().len() > 10 {
            steps.push(current.trim().to_string());
        }

        steps
    }

    /// Smart parsing that detects format
    pub fn parse_auto(text: &str) -> Vec<String> {
        // First try numbered
        let numbered = Self::parse_numbered(text);
        if numbered.len() >= 2 {
            return numbered;
        }

        // Fall back to sentences
        Self::parse_sentences(text)
    }
}

/// Best-of-N with PRM reranking
#[derive(Debug, Clone)]
pub struct PrmReranker {
    /// Number of candidate solutions to generate
    pub n_candidates: usize,
    /// How to aggregate step scores
    pub aggregation: ScoreAggregation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreAggregation {
    /// Product of step scores
    Product,
    /// Minimum step score
    Minimum,
    /// Weighted average (later steps count more)
    WeightedAverage,
    /// Geometric mean
    GeometricMean,
}

impl Default for PrmReranker {
    fn default() -> Self {
        Self {
            n_candidates: 5,
            aggregation: ScoreAggregation::Product,
        }
    }
}

impl PrmReranker {
    pub fn new(n_candidates: usize) -> Self {
        Self {
            n_candidates,
            ..Default::default()
        }
    }

    /// Calculate aggregate score for a reasoning chain
    pub fn aggregate_score(&self, step_scores: &[f32]) -> f32 {
        if step_scores.is_empty() {
            return 0.0;
        }

        match self.aggregation {
            ScoreAggregation::Product => step_scores.iter().product(),
            ScoreAggregation::Minimum => step_scores
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0),
            ScoreAggregation::WeightedAverage => {
                let weights: Vec<f32> = (1..=step_scores.len()).map(|i| i as f32).collect();
                let weight_sum: f32 = weights.iter().sum();
                step_scores
                    .iter()
                    .zip(weights.iter())
                    .map(|(s, w)| s * w)
                    .sum::<f32>()
                    / weight_sum
            }
            ScoreAggregation::GeometricMean => {
                let n = step_scores.len() as f32;
                step_scores
                    .iter()
                    .map(|s| s.max(0.001))
                    .product::<f32>()
                    .powf(1.0 / n)
            }
        }
    }

    /// Rerank solutions by PRM score
    pub fn rerank<T>(&self, solutions: &mut [(T, f32)])
    where
        T: Clone,
    {
        solutions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_parser_numbered() {
        let text = r#"
1. First, identify the given information
2. Next, set up the equation
3. Solve for x
4. Verify the answer
"#;

        let steps = StepParser::parse_numbered(text);
        assert_eq!(steps.len(), 4);
        assert!(steps[0].contains("identify"));
        assert!(steps[2].contains("Solve"));
    }

    #[test]
    fn test_prm_result_computation() {
        let scores = vec![
            StepScore {
                step_index: 0,
                step_content: "Step 1".into(),
                correctness: 0.9,
                logical_validity: 0.95,
                relevance: 0.9,
                issues: vec![],
                needs_revision: false,
            },
            StepScore {
                step_index: 1,
                step_content: "Step 2".into(),
                correctness: 0.85,
                logical_validity: 0.9,
                relevance: 0.85,
                issues: vec![],
                needs_revision: false,
            },
            StepScore {
                step_index: 2,
                step_content: "Step 3".into(),
                correctness: 0.8,
                logical_validity: 0.85,
                relevance: 0.9,
                issues: vec![],
                needs_revision: false,
            },
        ];

        let result = PrmResult::compute(scores);

        assert!(result.is_sound);
        assert!(result.first_error_step.is_none());
        assert!(result.overall_score > 0.5);
        assert_eq!(result.metrics.total_steps, 3);
        assert_eq!(result.metrics.correct_steps, 3);
    }

    #[test]
    fn test_prm_detects_errors() {
        let scores = vec![
            StepScore {
                step_index: 0,
                step_content: "Good step".into(),
                correctness: 0.9,
                logical_validity: 0.9,
                relevance: 0.9,
                issues: vec![],
                needs_revision: false,
            },
            StepScore {
                step_index: 1,
                step_content: "Bad step".into(),
                correctness: 0.3,
                logical_validity: 0.4,
                relevance: 0.5,
                issues: vec![StepIssue {
                    issue_type: IssueType::ArithmeticError,
                    description: "2 + 2 != 5".into(),
                    severity: Severity::Critical,
                    suggested_fix: Some("2 + 2 = 4".into()),
                }],
                needs_revision: true,
            },
        ];

        let result = PrmResult::compute(scores);

        assert!(!result.is_sound);
        assert_eq!(result.first_error_step, Some(1));
        assert_eq!(result.metrics.critical_issues, 1);
    }

    #[test]
    fn test_prm_reranker() {
        let reranker = PrmReranker::default();

        let mut solutions = vec![
            ("Solution A", 0.7),
            ("Solution B", 0.9),
            ("Solution C", 0.5),
        ];

        reranker.rerank(&mut solutions);

        assert_eq!(solutions[0].0, "Solution B");
        assert_eq!(solutions[1].0, "Solution A");
        assert_eq!(solutions[2].0, "Solution C");
    }

    #[test]
    fn test_score_aggregation() {
        let reranker = PrmReranker {
            n_candidates: 5,
            aggregation: ScoreAggregation::GeometricMean,
        };

        let scores = vec![0.9, 0.8, 0.7];
        let agg = reranker.aggregate_score(&scores);

        assert!((agg - 0.797).abs() < 0.01);
    }
}
