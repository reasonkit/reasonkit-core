//! # Self-Refine Module
//!
//! Implements iterative self-refinement based on Madaan et al. (2023)
//! "Self-Refine: Iterative Refinement with Self-Feedback"
//!
//! ## Scientific Foundation
//!
//! - arXiv:2303.17651: Self-Refine achieves +20% on math, code, and reasoning
//! - Works through GENERATE → FEEDBACK → REFINE loop
//! - No additional training required (prompt-based)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::self_refine::{SelfRefineEngine, RefineConfig};
//!
//! let engine = SelfRefineEngine::new(RefineConfig::default());
//! let result = engine.refine(initial_output, problem).await?;
//! ```

use serde::{Deserialize, Serialize};

/// Self-Refine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineConfig {
    /// Maximum refinement iterations
    pub max_iterations: usize,
    /// Stop if quality improvement is below this threshold
    pub min_improvement_threshold: f32,
    /// Stop if quality reaches this level
    pub target_quality: f32,
    /// Feedback dimensions to evaluate
    pub feedback_dimensions: Vec<FeedbackDimension>,
    /// Whether to preserve reasoning chain
    pub preserve_reasoning: bool,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            min_improvement_threshold: 0.05,
            target_quality: 0.90,
            feedback_dimensions: vec![
                FeedbackDimension::Correctness,
                FeedbackDimension::Completeness,
                FeedbackDimension::Clarity,
                FeedbackDimension::Coherence,
            ],
            preserve_reasoning: true,
        }
    }
}

impl RefineConfig {
    /// Config for BrutalHonesty (adversarial critique)
    pub fn brutal_honesty() -> Self {
        Self {
            max_iterations: 5,
            min_improvement_threshold: 0.03,
            target_quality: 0.95,
            feedback_dimensions: vec![
                FeedbackDimension::Correctness,
                FeedbackDimension::Honesty,
                FeedbackDimension::Completeness,
                FeedbackDimension::BiasDetection,
                FeedbackDimension::WeaknessIdentification,
            ],
            preserve_reasoning: true,
        }
    }

    /// Config for code refinement
    pub fn code() -> Self {
        Self {
            max_iterations: 4,
            feedback_dimensions: vec![
                FeedbackDimension::Correctness,
                FeedbackDimension::Efficiency,
                FeedbackDimension::Readability,
                FeedbackDimension::EdgeCases,
            ],
            ..Default::default()
        }
    }
}

/// Dimensions to evaluate for feedback
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackDimension {
    /// Is the content factually correct?
    Correctness,
    /// Is all relevant information included?
    Completeness,
    /// Is it clearly written/explained?
    Clarity,
    /// Does it flow logically?
    Coherence,
    /// Is it truthful without exaggeration?
    Honesty,
    /// Are biases identified and addressed?
    BiasDetection,
    /// Are weaknesses/limitations acknowledged?
    WeaknessIdentification,
    /// Are edge cases handled?
    EdgeCases,
    /// Is it efficient (for code)?
    Efficiency,
    /// Is it readable (for code)?
    Readability,
    /// Custom dimension
    Custom,
}

impl FeedbackDimension {
    pub fn prompt_question(&self) -> &'static str {
        match self {
            Self::Correctness => "Is the content factually correct? Identify any errors.",
            Self::Completeness => "Is all relevant information included? What's missing?",
            Self::Clarity => "Is the explanation clear? What's confusing?",
            Self::Coherence => "Does the reasoning flow logically? Any gaps?",
            Self::Honesty => "Is the assessment honest without exaggeration or false modesty?",
            Self::BiasDetection => "Are there any hidden biases or assumptions? Identify them.",
            Self::WeaknessIdentification => "What weaknesses or limitations exist? Be specific.",
            Self::EdgeCases => "Are edge cases and exceptions handled properly?",
            Self::Efficiency => "Is the solution efficient? How can it be optimized?",
            Self::Readability => "Is the code/text readable? How can it be improved?",
            Self::Custom => "Evaluate the overall quality and suggest improvements.",
        }
    }
}

/// Feedback for a single dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionFeedback {
    pub dimension: FeedbackDimension,
    /// Quality score (0.0 - 1.0)
    pub score: f32,
    /// Specific issues found
    pub issues: Vec<String>,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

/// Complete feedback from one iteration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationFeedback {
    pub iteration: usize,
    pub dimension_feedback: Vec<DimensionFeedback>,
    /// Overall quality score
    pub overall_score: f32,
    /// Combined improvement suggestions
    pub improvement_plan: String,
    /// Whether refinement should continue
    pub should_continue: bool,
}

impl IterationFeedback {
    pub fn compute_overall_score(&mut self) {
        if self.dimension_feedback.is_empty() {
            self.overall_score = 0.0;
            return;
        }

        self.overall_score = self.dimension_feedback.iter().map(|f| f.score).sum::<f32>()
            / self.dimension_feedback.len() as f32;
    }

    pub fn has_critical_issues(&self) -> bool {
        self.dimension_feedback.iter().any(|f| f.score < 0.5)
    }

    pub fn worst_dimension(&self) -> Option<&DimensionFeedback> {
        self.dimension_feedback.iter().min_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// A single refinement iteration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineIteration {
    pub iteration: usize,
    /// Content before this iteration
    pub input: String,
    /// Content after this iteration
    pub output: String,
    /// Feedback that guided this iteration
    pub feedback: IterationFeedback,
    /// Improvement from previous iteration
    pub quality_delta: f32,
}

/// Result of the self-refine process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineResult {
    /// Original input
    pub original: String,
    /// Final refined output
    pub refined: String,
    /// Quality of original (0.0 - 1.0)
    pub original_quality: f32,
    /// Quality of refined (0.0 - 1.0)
    pub refined_quality: f32,
    /// Total improvement
    pub improvement: f32,
    /// All iterations
    pub iterations: Vec<RefineIteration>,
    /// Why refinement stopped
    pub stop_reason: StopReason,
    /// Total tokens used
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Reached target quality
    TargetReached,
    /// Max iterations hit
    MaxIterations,
    /// Improvement below threshold
    DiminishingReturns,
    /// No issues found
    NoIssuesFound,
    /// Error during refinement
    Error,
}

impl RefineResult {
    pub fn improvement_percentage(&self) -> f32 {
        if self.original_quality > 0.0 {
            ((self.refined_quality - self.original_quality) / self.original_quality) * 100.0
        } else {
            0.0
        }
    }

    pub fn format_summary(&self) -> String {
        format!(
            "Self-Refine: {} iterations, {:.1}% → {:.1}% (+{:.1}%), stopped: {:?}",
            self.iterations.len(),
            self.original_quality * 100.0,
            self.refined_quality * 100.0,
            self.improvement * 100.0,
            self.stop_reason
        )
    }
}

/// Prompt templates for self-refine
pub struct RefinePrompts;

impl RefinePrompts {
    /// Generate feedback prompt
    pub fn feedback(content: &str, problem: &str, dimensions: &[FeedbackDimension]) -> String {
        let dimension_prompts: String = dimensions
            .iter()
            .enumerate()
            .map(|(i, d)| format!("{}. {}", i + 1, d.prompt_question()))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"You are a critical reviewer evaluating the following content.

ORIGINAL PROBLEM/TASK:
{problem}

CONTENT TO REVIEW:
{content}

Evaluate on these dimensions:
{dimension_prompts}

For each dimension, provide:
1. Score (0.0 - 1.0, where 1.0 is perfect)
2. Specific issues found
3. Concrete suggestions for improvement

Then provide an overall improvement plan.

Respond in JSON:
{{
    "dimensions": [
        {{
            "dimension": "dimension_name",
            "score": 0.0-1.0,
            "issues": ["issue1", "issue2"],
            "suggestions": ["suggestion1", "suggestion2"]
        }}
    ],
    "overall_score": 0.0-1.0,
    "improvement_plan": "Detailed plan to address the issues..."
}}"#,
            problem = problem,
            content = content,
            dimension_prompts = dimension_prompts
        )
    }

    /// Generate refinement prompt
    pub fn refine(content: &str, problem: &str, feedback: &IterationFeedback) -> String {
        let issues: Vec<String> = feedback
            .dimension_feedback
            .iter()
            .flat_map(|f| f.issues.clone())
            .collect();

        let suggestions: Vec<String> = feedback
            .dimension_feedback
            .iter()
            .flat_map(|f| f.suggestions.clone())
            .collect();

        format!(
            r#"Refine the following content based on the feedback provided.

ORIGINAL PROBLEM/TASK:
{problem}

CONTENT TO REFINE:
{content}

ISSUES IDENTIFIED:
{issues}

SUGGESTIONS FOR IMPROVEMENT:
{suggestions}

IMPROVEMENT PLAN:
{plan}

Provide the refined version that addresses ALL the issues and incorporates ALL the suggestions.
Maintain the same format and structure, but improve the quality."#,
            problem = problem,
            content = content,
            issues = issues
                .iter()
                .map(|i| format!("- {}", i))
                .collect::<Vec<_>>()
                .join("\n"),
            suggestions = suggestions
                .iter()
                .map(|s| format!("- {}", s))
                .collect::<Vec<_>>()
                .join("\n"),
            plan = feedback.improvement_plan
        )
    }

    /// BrutalHonesty-specific feedback prompt
    pub fn brutal_honesty_feedback(content: &str, claim: &str) -> String {
        format!(
            r#"You are the BRUTAL HONESTY reviewer. Your job is to find EVERY flaw.

CLAIM/ARGUMENT BEING ANALYZED:
{claim}

CURRENT ANALYSIS:
{content}

Be RUTHLESSLY CRITICAL. Evaluate:

1. HONESTY: Is the analysis truthful without exaggeration or false modesty?
   - Are strengths overstated?
   - Are weaknesses downplayed?
   - Is uncertainty properly communicated?

2. COMPLETENESS: What is MISSING from the analysis?
   - What perspectives weren't considered?
   - What evidence was overlooked?
   - What counterarguments weren't addressed?

3. BIAS DETECTION: What BIASES are present?
   - Confirmation bias (only seeing supporting evidence)?
   - Authority bias (over-relying on sources)?
   - Recency bias (ignoring historical context)?

4. WEAKNESS IDENTIFICATION: What are the WEAKNESSES?
   - In the reasoning?
   - In the evidence?
   - In the conclusions?

5. DEVIL'S ADVOCATE: Argue the OPPOSITE position
   - What would a critic say?
   - How could this be wrong?

Score each dimension 0.0-1.0 and provide specific improvements.
Respond in JSON format."#,
            claim = claim,
            content = content
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refine_config_default() {
        let config = RefineConfig::default();
        assert_eq!(config.max_iterations, 3);
        assert!(!config.feedback_dimensions.is_empty());
    }

    #[test]
    fn test_brutal_honesty_config() {
        let config = RefineConfig::brutal_honesty();
        assert_eq!(config.max_iterations, 5);
        assert!(config
            .feedback_dimensions
            .contains(&FeedbackDimension::Honesty));
        assert!(config
            .feedback_dimensions
            .contains(&FeedbackDimension::BiasDetection));
    }

    #[test]
    fn test_iteration_feedback() {
        let mut feedback = IterationFeedback {
            iteration: 1,
            dimension_feedback: vec![
                DimensionFeedback {
                    dimension: FeedbackDimension::Correctness,
                    score: 0.8,
                    issues: vec!["Minor error".into()],
                    suggestions: vec!["Fix error".into()],
                },
                DimensionFeedback {
                    dimension: FeedbackDimension::Clarity,
                    score: 0.6,
                    issues: vec!["Unclear section".into()],
                    suggestions: vec!["Rewrite section".into()],
                },
            ],
            overall_score: 0.0,
            improvement_plan: "Fix issues".into(),
            should_continue: true,
        };

        feedback.compute_overall_score();
        assert!((feedback.overall_score - 0.7).abs() < 0.01);
        assert!(!feedback.has_critical_issues());
    }

    #[test]
    fn test_dimension_prompts() {
        let q = FeedbackDimension::Honesty.prompt_question();
        assert!(q.contains("honest"));

        let q = FeedbackDimension::BiasDetection.prompt_question();
        assert!(q.contains("bias"));
    }

    #[test]
    fn test_refine_result_summary() {
        let result = RefineResult {
            original: "Original".into(),
            refined: "Refined".into(),
            original_quality: 0.6,
            refined_quality: 0.85,
            improvement: 0.25,
            iterations: vec![],
            stop_reason: StopReason::TargetReached,
            total_tokens: 1000,
        };

        let summary = result.format_summary();
        assert!(summary.contains("TargetReached"));
        assert!(result.improvement_percentage() > 40.0);
    }
}
