//! # Tripartite Mind Processing
//!
//! Implements the three-level cognitive processing model for structured reasoning.
//!
//! ## Scientific Foundation
//!
//! Based on Stanovich's tripartite model of mind:
//! - Autonomous Mind: Fast, automatic, intuitive (System 1)
//! - Algorithmic Mind: Deliberate, rule-following (System 2)
//! - Reflective Mind: Meta-cognitive, evaluative (System 3)
//!
//! ## The Processing Model
//!
//! ```text
//! STIMULUS → AUTONOMOUS → ALGORITHMIC → REFLECTIVE → OUTPUT
//!               ↓            ↓             ↓
//!            Intuition    Reasoning    Meta-check
//!            (fast)       (slow)       (evaluative)
//! ```
//!
//! ## Why This Matters for AI
//!
//! LLMs often blend all three modes. Explicit separation:
//! - Makes reasoning more transparent
//! - Enables targeted intervention
//! - Improves reliability through meta-cognition
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::tripartite::{TripartiteProcessor, TripartiteConfig};
//!
//! let processor = TripartiteProcessor::new(TripartiteConfig::default());
//! let result = processor.process(problem).await?;
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for tripartite processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripartiteConfig {
    /// Enable autonomous (intuitive) phase
    pub enable_autonomous: bool,
    /// Enable algorithmic (deliberate) phase
    pub enable_algorithmic: bool,
    /// Enable reflective (meta-cognitive) phase
    pub enable_reflective: bool,
    /// Number of autonomous responses to generate
    pub autonomous_samples: usize,
    /// Algorithmic reasoning depth
    pub algorithmic_depth: AlgorithmicDepth,
    /// Reflective checks to perform
    pub reflective_checks: Vec<ReflectiveCheck>,
    /// Allow reflective phase to override earlier phases
    pub allow_override: bool,
}

impl Default for TripartiteConfig {
    fn default() -> Self {
        Self {
            enable_autonomous: true,
            enable_algorithmic: true,
            enable_reflective: true,
            autonomous_samples: 3,
            algorithmic_depth: AlgorithmicDepth::Standard,
            reflective_checks: vec![
                ReflectiveCheck::BiasDetection,
                ReflectiveCheck::ConsistencyCheck,
                ReflectiveCheck::ConfidenceCalibration,
                ReflectiveCheck::AlternativeConsideration,
            ],
            allow_override: true,
        }
    }
}

impl TripartiteConfig {
    /// Full PowerCombo configuration (maximum rigor)
    pub fn powercombo() -> Self {
        Self {
            enable_autonomous: true,
            enable_algorithmic: true,
            enable_reflective: true,
            autonomous_samples: 5,
            algorithmic_depth: AlgorithmicDepth::Deep,
            reflective_checks: vec![
                ReflectiveCheck::BiasDetection,
                ReflectiveCheck::ConsistencyCheck,
                ReflectiveCheck::ConfidenceCalibration,
                ReflectiveCheck::AlternativeConsideration,
                ReflectiveCheck::MetaCognition,
                ReflectiveCheck::DevilsAdvocate,
            ],
            allow_override: true,
        }
    }

    /// Quick intuition + check mode
    pub fn quick() -> Self {
        Self {
            enable_autonomous: true,
            enable_algorithmic: false,
            enable_reflective: true,
            autonomous_samples: 1,
            algorithmic_depth: AlgorithmicDepth::None,
            reflective_checks: vec![ReflectiveCheck::ConfidenceCalibration],
            allow_override: false,
        }
    }

    /// Pure algorithmic mode (no intuition)
    pub fn algorithmic_only() -> Self {
        Self {
            enable_autonomous: false,
            enable_algorithmic: true,
            enable_reflective: true,
            autonomous_samples: 0,
            algorithmic_depth: AlgorithmicDepth::Deep,
            reflective_checks: vec![
                ReflectiveCheck::ConsistencyCheck,
                ReflectiveCheck::ConfidenceCalibration,
            ],
            allow_override: false,
        }
    }
}

/// Depth of algorithmic processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlgorithmicDepth {
    /// No algorithmic processing
    None,
    /// Light processing (quick checks)
    Light,
    /// Standard processing (full reasoning)
    Standard,
    /// Deep processing (exhaustive analysis)
    Deep,
}

/// Types of reflective checks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReflectiveCheck {
    /// Detect cognitive biases
    BiasDetection,
    /// Check internal consistency
    ConsistencyCheck,
    /// Calibrate confidence levels
    ConfidenceCalibration,
    /// Consider alternatives
    AlternativeConsideration,
    /// Meta-cognitive awareness
    MetaCognition,
    /// Devil's advocate challenge
    DevilsAdvocate,
    /// Check for epistemic humility
    EpistemicHumility,
    /// Verify logical soundness
    LogicalSoundness,
}

impl ReflectiveCheck {
    /// Get the question this check asks
    pub fn question(&self) -> &'static str {
        match self {
            Self::BiasDetection => "What cognitive biases might be affecting this reasoning?",
            Self::ConsistencyCheck => "Is this answer internally consistent?",
            Self::ConfidenceCalibration => "How confident should we be, and why?",
            Self::AlternativeConsideration => "What alternatives haven't been considered?",
            Self::MetaCognition => "How do I know what I know here?",
            Self::DevilsAdvocate => "What's the strongest case against this conclusion?",
            Self::EpistemicHumility => "What might I be wrong about?",
            Self::LogicalSoundness => "Is the logical structure of this argument valid?",
        }
    }
}

/// Result from autonomous processing (System 1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousResponse {
    /// The intuitive response
    pub response: String,
    /// Confidence in this intuition
    pub confidence: f32,
    /// Reaction time indicator (fast = more intuitive)
    pub intuition_strength: f32,
    /// Any immediate flags or concerns
    pub flags: Vec<String>,
}

/// Result from algorithmic processing (System 2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmicResponse {
    /// The reasoned response
    pub response: String,
    /// Reasoning steps taken
    pub reasoning_steps: Vec<ReasoningStep>,
    /// Confidence after deliberation
    pub confidence: f32,
    /// Agreed with autonomous response?
    pub agrees_with_autonomous: bool,
    /// Disagreement explanation (if applicable)
    pub disagreement_reason: Option<String>,
}

/// A single reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step number
    pub step: usize,
    /// Step description
    pub description: String,
    /// Type of step
    pub step_type: StepType,
    /// Confidence at this step
    pub confidence: f32,
}

/// Types of reasoning steps
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepType {
    /// Identify relevant information
    Identify,
    /// Apply a rule or principle
    Apply,
    /// Make a deduction
    Deduce,
    /// Verify a claim
    Verify,
    /// Conclude
    Conclude,
    /// Backtrack and reconsider
    Backtrack,
}

/// Result from reflective processing (System 3)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectiveResponse {
    /// The final, reflected response
    pub response: String,
    /// Reflective check results
    pub check_results: Vec<CheckResult>,
    /// Final confidence after reflection
    pub confidence: f32,
    /// Did reflection change the answer?
    pub answer_changed: bool,
    /// Explanation of any changes
    pub change_explanation: Option<String>,
    /// Remaining concerns
    pub remaining_concerns: Vec<String>,
}

/// Result of a single reflective check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub check: ReflectiveCheck,
    /// Passed/failed/partial
    pub status: CheckStatus,
    /// Issues found
    pub issues: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Confidence adjustment
    pub confidence_adjustment: f32,
}

/// Status of a reflective check
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckStatus {
    /// Check passed
    Passed,
    /// Check failed
    Failed,
    /// Partial pass
    Partial,
    /// Could not determine
    Undetermined,
}

/// Complete tripartite processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripartiteResult {
    /// Original problem
    pub problem: String,
    /// Autonomous phase results
    pub autonomous: Option<Vec<AutonomousResponse>>,
    /// Algorithmic phase result
    pub algorithmic: Option<AlgorithmicResponse>,
    /// Reflective phase result
    pub reflective: Option<ReflectiveResponse>,
    /// Final answer
    pub final_answer: String,
    /// Final confidence
    pub final_confidence: f32,
    /// Processing metrics
    pub metrics: TripartiteMetrics,
}

/// Metrics for tripartite processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripartiteMetrics {
    /// Did phases agree?
    pub phase_agreement: bool,
    /// Confidence progression (auto → algo → reflective)
    pub confidence_progression: Vec<f32>,
    /// Number of reasoning steps
    pub reasoning_steps: usize,
    /// Number of checks passed
    pub checks_passed: usize,
    /// Number of checks failed
    pub checks_failed: usize,
    /// Was answer overridden by later phase?
    pub was_overridden: bool,
}

impl TripartiteResult {
    /// Get the phase that determined the final answer
    pub fn determining_phase(&self) -> &'static str {
        if self.reflective.as_ref().is_some_and(|r| r.answer_changed) {
            "reflective"
        } else if self
            .algorithmic
            .as_ref()
            .is_some_and(|a| !a.agrees_with_autonomous)
        {
            "algorithmic"
        } else {
            "autonomous"
        }
    }

    /// Did all phases agree?
    pub fn all_phases_agree(&self) -> bool {
        self.metrics.phase_agreement
    }

    /// Format a summary
    pub fn format_summary(&self) -> String {
        let phase = self.determining_phase();
        let progression: String = self
            .metrics
            .confidence_progression
            .iter()
            .map(|c| format!("{:.0}%", c * 100.0))
            .collect::<Vec<_>>()
            .join(" → ");

        format!(
            "Tripartite: {} phase determined answer, confidence: {}, {} checks passed/{} failed",
            phase, progression, self.metrics.checks_passed, self.metrics.checks_failed
        )
    }
}

/// Prompt templates for tripartite processing
pub struct TripartitePrompts;

impl TripartitePrompts {
    /// Autonomous phase prompt (intuitive)
    pub fn autonomous(problem: &str) -> String {
        format!(
            r#"AUTONOMOUS MIND: Generate immediate intuitive response.

PROBLEM: {problem}

Respond QUICKLY with your first intuition.
Don't overthink. Trust your pattern recognition.

What is your immediate, intuitive answer?

Format:
INTUITIVE_ANSWER: [your immediate response]
CONFIDENCE: [0.0-1.0]
INTUITION_STRENGTH: [0.0-1.0, how strongly this "feels" right]
FLAGS: [any immediate concerns or uncertainties]"#,
            problem = problem
        )
    }

    /// Algorithmic phase prompt (deliberate)
    pub fn algorithmic(
        problem: &str,
        autonomous_answer: Option<&str>,
        depth: AlgorithmicDepth,
    ) -> String {
        let depth_instruction = match depth {
            AlgorithmicDepth::None => "Skip detailed reasoning.",
            AlgorithmicDepth::Light => "Quick logical check of key points.",
            AlgorithmicDepth::Standard => "Systematic step-by-step reasoning.",
            AlgorithmicDepth::Deep => "Exhaustive analysis of all aspects.",
        };

        let autonomous_section = autonomous_answer
            .map_or("No autonomous response available.".to_string(), |a| {
                format!("AUTONOMOUS RESPONSE: {}", a)
            });

        format!(
            r#"ALGORITHMIC MIND: Apply deliberate, systematic reasoning.

PROBLEM: {problem}

{autonomous_section}

DEPTH: {depth_instruction}

Reason through this step by step:
1. Identify key information and constraints
2. Apply relevant principles and rules
3. Make logical deductions
4. Verify each step
5. Reach a conclusion

Format:
STEP 1: [type: identify/apply/deduce/verify/conclude] [description]
STEP 2: ...
...

REASONED_ANSWER: [your answer after deliberation]
CONFIDENCE: [0.0-1.0]
AGREES_WITH_AUTONOMOUS: [true/false]
DISAGREEMENT_REASON: [if false, explain why]"#,
            problem = problem,
            autonomous_section = autonomous_section,
            depth_instruction = depth_instruction
        )
    }

    /// Reflective phase prompt (meta-cognitive)
    pub fn reflective(problem: &str, current_answer: &str, checks: &[ReflectiveCheck]) -> String {
        let checks_formatted: String = checks
            .iter()
            .map(|c| format!("- {:?}: {}", c, c.question()))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"REFLECTIVE MIND: Meta-cognitive evaluation.

PROBLEM: {problem}

CURRENT ANSWER: {current_answer}

Perform these reflective checks:
{checks_formatted}

For each check:
1. Evaluate the current answer
2. Identify any issues
3. Suggest improvements if needed
4. Adjust confidence accordingly

After all checks:
- Should the answer change?
- What's the final confidence?
- What concerns remain?

Format:
CHECK_1:
  - Status: [passed/failed/partial]
  - Issues: [list any issues]
  - Recommendations: [suggestions]
  - Confidence_adjustment: [+/-0.X]

...

FINAL_ANSWER: [unchanged or revised answer]
ANSWER_CHANGED: [true/false]
CHANGE_EXPLANATION: [if changed, why]
FINAL_CONFIDENCE: [0.0-1.0]
REMAINING_CONCERNS: [list any unresolved issues]"#,
            problem = problem,
            current_answer = current_answer,
            checks_formatted = checks_formatted
        )
    }

    /// Integration prompt (combine all phases)
    pub fn integrate(
        problem: &str,
        autonomous: &str,
        algorithmic: &str,
        reflective: &str,
    ) -> String {
        format!(
            r#"INTEGRATION: Synthesize all three processing phases.

PROBLEM: {problem}

AUTONOMOUS (Intuitive): {autonomous}

ALGORITHMIC (Deliberate): {algorithmic}

REFLECTIVE (Meta-cognitive): {reflective}

Synthesize:
1. Where do the phases agree?
2. Where do they disagree?
3. Which phase should we trust most for this problem?
4. What is the final integrated answer?

PHASE_AGREEMENT: [full/partial/none]
TRUSTED_PHASE: [autonomous/algorithmic/reflective]
TRUST_REASON: [why this phase is most reliable for this problem]
INTEGRATED_ANSWER: [final synthesized answer]
INTEGRATED_CONFIDENCE: [0.0-1.0]"#,
            problem = problem,
            autonomous = autonomous,
            algorithmic = algorithmic,
            reflective = reflective
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TripartiteConfig::default();
        assert!(config.enable_autonomous);
        assert!(config.enable_algorithmic);
        assert!(config.enable_reflective);
    }

    #[test]
    fn test_powercombo_config() {
        let config = TripartiteConfig::powercombo();
        assert_eq!(config.autonomous_samples, 5);
        assert_eq!(config.algorithmic_depth, AlgorithmicDepth::Deep);
        assert!(config
            .reflective_checks
            .contains(&ReflectiveCheck::MetaCognition));
    }

    #[test]
    fn test_reflective_checks() {
        let check = ReflectiveCheck::BiasDetection;
        assert!(check.question().contains("bias"));

        let check = ReflectiveCheck::ConfidenceCalibration;
        assert!(check.question().contains("confident"));
    }

    #[test]
    fn test_check_status() {
        let result = CheckResult {
            check: ReflectiveCheck::ConsistencyCheck,
            status: CheckStatus::Passed,
            issues: vec![],
            recommendations: vec![],
            confidence_adjustment: 0.1,
        };

        assert_eq!(result.status, CheckStatus::Passed);
        assert!(result.confidence_adjustment > 0.0);
    }

    #[test]
    fn test_tripartite_result() {
        let result = TripartiteResult {
            problem: "What is 2+2?".into(),
            autonomous: Some(vec![AutonomousResponse {
                response: "4".into(),
                confidence: 0.95,
                intuition_strength: 0.99,
                flags: vec![],
            }]),
            algorithmic: Some(AlgorithmicResponse {
                response: "4".into(),
                reasoning_steps: vec![ReasoningStep {
                    step: 1,
                    description: "Addition of single digits".into(),
                    step_type: StepType::Apply,
                    confidence: 1.0,
                }],
                confidence: 1.0,
                agrees_with_autonomous: true,
                disagreement_reason: None,
            }),
            reflective: Some(ReflectiveResponse {
                response: "4".into(),
                check_results: vec![CheckResult {
                    check: ReflectiveCheck::ConsistencyCheck,
                    status: CheckStatus::Passed,
                    issues: vec![],
                    recommendations: vec![],
                    confidence_adjustment: 0.0,
                }],
                confidence: 1.0,
                answer_changed: false,
                change_explanation: None,
                remaining_concerns: vec![],
            }),
            final_answer: "4".into(),
            final_confidence: 1.0,
            metrics: TripartiteMetrics {
                phase_agreement: true,
                confidence_progression: vec![0.95, 1.0, 1.0],
                reasoning_steps: 1,
                checks_passed: 1,
                checks_failed: 0,
                was_overridden: false,
            },
        };

        assert!(result.all_phases_agree());
        assert_eq!(result.determining_phase(), "autonomous");
        assert!(result.format_summary().contains("autonomous"));
    }

    #[test]
    fn test_override_detection() {
        let result = TripartiteResult {
            problem: "Complex problem".into(),
            autonomous: Some(vec![AutonomousResponse {
                response: "Wrong answer".into(),
                confidence: 0.6,
                intuition_strength: 0.5,
                flags: vec!["Uncertain".into()],
            }]),
            algorithmic: Some(AlgorithmicResponse {
                response: "Better answer".into(),
                reasoning_steps: vec![],
                confidence: 0.8,
                agrees_with_autonomous: false,
                disagreement_reason: Some("Intuition missed key factor".into()),
            }),
            reflective: Some(ReflectiveResponse {
                response: "Better answer".into(),
                check_results: vec![],
                confidence: 0.85,
                answer_changed: false,
                change_explanation: None,
                remaining_concerns: vec![],
            }),
            final_answer: "Better answer".into(),
            final_confidence: 0.85,
            metrics: TripartiteMetrics {
                phase_agreement: false,
                confidence_progression: vec![0.6, 0.8, 0.85],
                reasoning_steps: 0,
                checks_passed: 0,
                checks_failed: 0,
                was_overridden: true,
            },
        };

        assert!(!result.all_phases_agree());
        assert_eq!(result.determining_phase(), "algorithmic");
    }
}
