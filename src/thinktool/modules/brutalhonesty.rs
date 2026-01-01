//! BrutalHonesty Module - Adversarial Self-Critique
//!
//! Red-team analysis that finds flaws before others do, challenges
//! assumptions aggressively, and scores confidence with skepticism.
//!
//! ## Design Philosophy
//!
//! BrutalHonesty applies adversarial thinking to identify weaknesses:
//! - **Assumption Hunter**: Extracts implicit assumptions and questions them
//! - **Flaw Finder**: Categorizes weaknesses by severity and type
//! - **Skeptical Scorer**: Adjusts confidence downward based on issues found
//! - **Devil's Advocate**: Argues against the position to stress-test it
//!
//! ## Usage
//!
//! ```ignore
//! use reasonkit::thinktool::modules::{BrutalHonesty, ThinkToolContext};
//!
//! let module = BrutalHonesty::builder()
//!     .severity(CritiqueSeverity::Ruthless)
//!     .enable_devil_advocate(true)
//!     .build();
//!
//! let context = ThinkToolContext {
//!     query: "Our startup will succeed because we have the best team".to_string(),
//!     previous_steps: vec![],
//! };
//!
//! let result = module.execute(&context)?;
//! ```

use super::{ThinkToolContext, ThinkToolModule, ThinkToolModuleConfig, ThinkToolOutput};
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Severity level for critique analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CritiqueSeverity {
    /// Gentle critique - focus on constructive feedback
    Gentle,
    /// Standard critique - balanced flaw detection
    #[default]
    Standard,
    /// Harsh critique - aggressive assumption challenging
    Harsh,
    /// Ruthless critique - no mercy, find every possible flaw
    Ruthless,
}

impl CritiqueSeverity {
    /// Get the skepticism multiplier for confidence scoring.
    /// Higher skepticism = lower confidence adjustments.
    fn skepticism_multiplier(&self) -> f64 {
        match self {
            Self::Gentle => 0.90,   // 10% skepticism reduction
            Self::Standard => 0.80, // 20% skepticism reduction
            Self::Harsh => 0.65,    // 35% skepticism reduction
            Self::Ruthless => 0.50, // 50% skepticism reduction
        }
    }
}

/// Category of detected flaw.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlawCategory {
    /// Logical flaw (fallacy, contradiction, non-sequitur)
    Logical,
    /// Evidential flaw (missing data, weak sources, cherry-picking)
    Evidential,
    /// Assumption flaw (unexamined premises, hidden biases)
    Assumption,
    /// Scope flaw (overgeneralization, false dichotomy)
    Scope,
    /// Temporal flaw (recency bias, ignoring history)
    Temporal,
    /// Adversarial flaw (vulnerability to counter-arguments)
    Adversarial,
    /// Completeness flaw (missing considerations, blind spots)
    Completeness,
}

/// Severity of a detected flaw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlawSeverity {
    /// Minor issue - worth noting but not critical
    Minor,
    /// Moderate issue - should be addressed
    Moderate,
    /// Major issue - significantly weakens the argument
    Major,
    /// Critical issue - fundamentally undermines the position
    Critical,
}

impl FlawSeverity {
    /// Get the confidence penalty for this severity level.
    fn confidence_penalty(&self) -> f64 {
        match self {
            Self::Minor => 0.02,
            Self::Moderate => 0.08,
            Self::Major => 0.15,
            Self::Critical => 0.30,
        }
    }
}

/// A detected flaw in the reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedFlaw {
    /// Category of the flaw
    pub category: FlawCategory,
    /// Severity of the flaw
    pub severity: FlawSeverity,
    /// Description of the flaw
    pub description: String,
    /// The specific text or aspect that triggered this flaw
    pub trigger: Option<String>,
    /// Suggested remediation
    pub remediation: Option<String>,
}

impl DetectedFlaw {
    /// Create a new detected flaw.
    pub fn new(
        category: FlawCategory,
        severity: FlawSeverity,
        description: impl Into<String>,
    ) -> Self {
        Self {
            category,
            severity,
            description: description.into(),
            trigger: None,
            remediation: None,
        }
    }

    /// Add a trigger to the flaw.
    pub fn with_trigger(mut self, trigger: impl Into<String>) -> Self {
        self.trigger = Some(trigger.into());
        self
    }

    /// Add a remediation suggestion.
    pub fn with_remediation(mut self, remediation: impl Into<String>) -> Self {
        self.remediation = Some(remediation.into());
        self
    }
}

/// An implicit assumption detected in the reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplicitAssumption {
    /// The assumption itself
    pub assumption: String,
    /// How confident we are this is an implicit assumption (0.0-1.0)
    pub confidence: f64,
    /// Why this assumption may be problematic
    pub risk: String,
    /// Whether this assumption is likely valid
    pub likely_valid: bool,
}

/// A strength identified in the reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentifiedStrength {
    /// Description of the strength
    pub description: String,
    /// How significant this strength is (0.0-1.0)
    pub significance: f64,
}

/// Overall verdict from the brutal honesty analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CritiqueVerdict {
    /// The reasoning is solid and defensible
    Solid,
    /// The reasoning has merit but needs improvement
    Promising,
    /// The reasoning has significant issues
    Weak,
    /// The reasoning is fundamentally flawed
    Flawed,
    /// Cannot assess due to insufficient information
    Indeterminate,
}

/// Configuration for the BrutalHonesty module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrutalHonestyConfig {
    /// Severity level for critique
    pub severity: CritiqueSeverity,
    /// Whether to enable devil's advocate mode
    pub enable_devil_advocate: bool,
    /// Whether to check for confirmation bias
    pub check_confirmation_bias: bool,
    /// Minimum confidence threshold (below this triggers warning)
    pub min_confidence_threshold: f64,
    /// Maximum number of flaws to report
    pub max_flaws_reported: usize,
    /// Focus areas for critique (empty = all areas)
    pub focus_areas: Vec<FlawCategory>,
}

impl Default for BrutalHonestyConfig {
    fn default() -> Self {
        Self {
            severity: CritiqueSeverity::Standard,
            enable_devil_advocate: true,
            check_confirmation_bias: true,
            min_confidence_threshold: 0.50,
            max_flaws_reported: 10,
            focus_areas: vec![],
        }
    }
}

/// Builder for BrutalHonesty module configuration.
#[derive(Debug, Default)]
pub struct BrutalHonestyBuilder {
    config: BrutalHonestyConfig,
}

impl BrutalHonestyBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the critique severity level.
    pub fn severity(mut self, severity: CritiqueSeverity) -> Self {
        self.config.severity = severity;
        self
    }

    /// Enable or disable devil's advocate mode.
    pub fn enable_devil_advocate(mut self, enable: bool) -> Self {
        self.config.enable_devil_advocate = enable;
        self
    }

    /// Enable or disable confirmation bias checking.
    pub fn check_confirmation_bias(mut self, enable: bool) -> Self {
        self.config.check_confirmation_bias = enable;
        self
    }

    /// Set the minimum confidence threshold.
    pub fn min_confidence_threshold(mut self, threshold: f64) -> Self {
        self.config.min_confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum number of flaws to report.
    pub fn max_flaws_reported(mut self, max: usize) -> Self {
        self.config.max_flaws_reported = max;
        self
    }

    /// Set focus areas for critique.
    pub fn focus_areas(mut self, areas: Vec<FlawCategory>) -> Self {
        self.config.focus_areas = areas;
        self
    }

    /// Build the BrutalHonesty module.
    pub fn build(self) -> BrutalHonesty {
        BrutalHonesty::with_config(self.config)
    }
}

/// BrutalHonesty reasoning module for adversarial critique.
///
/// Attacks ideas to identify weaknesses, challenges assumptions,
/// and scores confidence with appropriate skepticism.
pub struct BrutalHonesty {
    /// Module base configuration
    module_config: ThinkToolModuleConfig,
    /// BrutalHonesty-specific configuration
    config: BrutalHonestyConfig,
}

impl Default for BrutalHonesty {
    fn default() -> Self {
        Self::new()
    }
}

impl BrutalHonesty {
    /// Create a new BrutalHonesty module with default configuration.
    pub fn new() -> Self {
        Self::with_config(BrutalHonestyConfig::default())
    }

    /// Create a BrutalHonesty module with custom configuration.
    pub fn with_config(config: BrutalHonestyConfig) -> Self {
        Self {
            module_config: ThinkToolModuleConfig {
                name: "BrutalHonesty".to_string(),
                version: "3.0.0".to_string(),
                description: "Adversarial self-critique with skeptical confidence scoring"
                    .to_string(),
                confidence_weight: 0.15,
            },
            config,
        }
    }

    /// Create a builder for customizing the module.
    pub fn builder() -> BrutalHonestyBuilder {
        BrutalHonestyBuilder::new()
    }

    /// Get the current configuration.
    pub fn brutal_config(&self) -> &BrutalHonestyConfig {
        &self.config
    }

    /// Analyze the input for implicit assumptions.
    fn extract_assumptions(&self, query: &str) -> Vec<ImplicitAssumption> {
        let mut assumptions = Vec::new();

        // Pattern-based assumption detection
        let assumption_patterns = [
            ("will", "Assumes future outcome is certain", 0.75),
            ("always", "Assumes universal applicability", 0.80),
            ("never", "Assumes absolute exclusion", 0.80),
            ("everyone", "Assumes universal agreement", 0.85),
            ("obvious", "Assumes shared understanding", 0.70),
            ("clearly", "Assumes self-evidence", 0.65),
            ("best", "Assumes optimal status without comparison", 0.60),
            ("only", "Assumes exclusivity", 0.70),
            ("must", "Assumes necessity without justification", 0.65),
            ("should", "Assumes normative position", 0.55),
            ("need", "Assumes requirement without evidence", 0.60),
            ("because", "May assume causation from correlation", 0.50),
        ];

        let query_lower = query.to_lowercase();

        for (pattern, risk, confidence) in assumption_patterns {
            if query_lower.contains(pattern) {
                assumptions.push(ImplicitAssumption {
                    assumption: format!(
                        "Use of '{}' implies unstated certainty or universality",
                        pattern
                    ),
                    confidence,
                    risk: risk.to_string(),
                    likely_valid: confidence < 0.65,
                });
            }
        }

        // Check for causal language without evidence
        if query_lower.contains("because")
            && !query_lower.contains("data")
            && !query_lower.contains("evidence")
            && !query_lower.contains("study")
            && !query_lower.contains("research")
        {
            assumptions.push(ImplicitAssumption {
                assumption: "Causal claim made without citing evidence".to_string(),
                confidence: 0.70,
                risk: "Causation may be assumed from correlation".to_string(),
                likely_valid: false,
            });
        }

        // Check for value judgments
        let value_words = ["good", "bad", "right", "wrong", "better", "worse"];
        for word in value_words {
            if query_lower.contains(word) {
                assumptions.push(ImplicitAssumption {
                    assumption: format!("Value judgment '{}' assumes shared moral framework", word),
                    confidence: 0.55,
                    risk: "Value judgments may not be universally shared".to_string(),
                    likely_valid: true, // Values can be valid but should be acknowledged
                });
                break; // Only report once
            }
        }

        assumptions
    }

    /// Detect flaws in the reasoning.
    fn detect_flaws(&self, query: &str, previous_steps: &[String]) -> Vec<DetectedFlaw> {
        let mut flaws = Vec::new();
        let query_lower = query.to_lowercase();
        let query_len = query.len();

        // Check for overgeneralization
        let universal_quantifiers = ["all", "every", "always", "never", "none", "no one"];
        for quantifier in universal_quantifiers {
            if query_lower.contains(quantifier) {
                flaws.push(
                    DetectedFlaw::new(
                        FlawCategory::Scope,
                        FlawSeverity::Moderate,
                        format!(
                            "Universal quantifier '{}' may indicate overgeneralization",
                            quantifier
                        ),
                    )
                    .with_remediation("Consider whether there are exceptions or edge cases"),
                );
            }
        }

        // Check for appeal to authority without specifics
        if (query_lower.contains("expert")
            || query_lower.contains("studies show")
            || query_lower.contains("research shows"))
            && !query_lower.contains("according to")
            && !query_lower.contains("published")
        {
            flaws.push(
                DetectedFlaw::new(
                    FlawCategory::Evidential,
                    FlawSeverity::Moderate,
                    "Vague appeal to authority without specific citation",
                )
                .with_remediation("Cite specific sources, authors, or publications"),
            );
        }

        // Check for false dichotomy
        if query_lower.contains("either") && query_lower.contains("or") {
            flaws.push(
                DetectedFlaw::new(
                    FlawCategory::Logical,
                    FlawSeverity::Moderate,
                    "Either/or construction may present false dichotomy",
                )
                .with_trigger("either...or")
                .with_remediation("Consider whether other alternatives exist"),
            );
        }

        // Check for recency bias
        if query_lower.contains("nowadays")
            || query_lower.contains("these days")
            || query_lower.contains("modern")
        {
            flaws.push(
                DetectedFlaw::new(
                    FlawCategory::Temporal,
                    FlawSeverity::Minor,
                    "Temporal framing may indicate recency bias",
                )
                .with_remediation("Consider historical patterns and whether 'new' means 'better'"),
            );
        }

        // Check for emotional language that may cloud reasoning
        let emotional_words = [
            "amazing",
            "terrible",
            "disaster",
            "revolutionary",
            "incredible",
            "horrible",
            "catastrophic",
            "miraculous",
            "devastating",
        ];
        for word in emotional_words {
            if query_lower.contains(word) {
                flaws.push(
                    DetectedFlaw::new(
                        FlawCategory::Logical,
                        FlawSeverity::Minor,
                        format!("Emotional language '{}' may indicate bias", word),
                    )
                    .with_trigger(word)
                    .with_remediation("Replace emotional terms with factual descriptions"),
                );
                break;
            }
        }

        // Check for lack of counter-arguments
        let counter_arg_indicators = [
            "however",
            "although",
            "but",
            "on the other hand",
            "conversely",
        ];
        let has_counter = counter_arg_indicators
            .iter()
            .any(|ind| query_lower.contains(ind));

        if !has_counter && query_len > 100 {
            flaws.push(
                DetectedFlaw::new(
                    FlawCategory::Completeness,
                    FlawSeverity::Major,
                    "No counter-arguments or alternative viewpoints presented",
                )
                .with_remediation("Steel-man opposing positions before dismissing them"),
            );
        }

        // Check for consistency with previous steps
        if !previous_steps.is_empty() {
            let prev_combined = previous_steps.join(" ").to_lowercase();

            // Look for potential contradictions
            if (query_lower.contains("not") && !prev_combined.contains("not"))
                || (!query_lower.contains("not") && prev_combined.contains("not "))
            {
                flaws.push(
                    DetectedFlaw::new(
                        FlawCategory::Logical,
                        FlawSeverity::Moderate,
                        "Potential inconsistency detected with previous reasoning steps",
                    )
                    .with_remediation("Review previous steps for logical consistency"),
                );
            }
        }

        // Check for vague claims
        let vague_indicators = ["somewhat", "kind of", "sort of", "basically", "essentially"];
        for vague in vague_indicators {
            if query_lower.contains(vague) {
                flaws.push(
                    DetectedFlaw::new(
                        FlawCategory::Evidential,
                        FlawSeverity::Minor,
                        format!("Vague qualifier '{}' reduces precision", vague),
                    )
                    .with_trigger(vague)
                    .with_remediation("Be more specific and precise in claims"),
                );
            }
        }

        // Check for confirmation bias indicators
        if self.config.check_confirmation_bias
            && query_lower.contains("proves")
            && !query_lower.contains("disproves")
        {
            flaws.push(
                DetectedFlaw::new(
                    FlawCategory::Assumption,
                    FlawSeverity::Moderate,
                    "One-sided evidence presentation may indicate confirmation bias",
                )
                .with_remediation("Actively seek disconfirming evidence"),
            );
        }

        // Limit flaws to configured maximum
        if flaws.len() > self.config.max_flaws_reported {
            // Sort by severity (most severe first) and take top N
            flaws.sort_by(|a, b| b.severity.cmp(&a.severity));
            flaws.truncate(self.config.max_flaws_reported);
        }

        flaws
    }

    /// Identify strengths in the reasoning.
    fn identify_strengths(&self, query: &str) -> Vec<IdentifiedStrength> {
        let mut strengths = Vec::new();
        let query_lower = query.to_lowercase();

        // Check for evidence-based reasoning
        if query_lower.contains("data")
            || query_lower.contains("evidence")
            || query_lower.contains("study")
            || query_lower.contains("research")
        {
            strengths.push(IdentifiedStrength {
                description: "References to data or evidence support claims".to_string(),
                significance: 0.75,
            });
        }

        // Check for nuanced language
        if query_lower.contains("however")
            || query_lower.contains("although")
            || query_lower.contains("on the other hand")
        {
            strengths.push(IdentifiedStrength {
                description: "Acknowledges counter-arguments or nuance".to_string(),
                significance: 0.70,
            });
        }

        // Check for specific examples
        if query_lower.contains("for example")
            || query_lower.contains("for instance")
            || query_lower.contains("specifically")
        {
            strengths.push(IdentifiedStrength {
                description: "Uses specific examples to support arguments".to_string(),
                significance: 0.65,
            });
        }

        // Check for qualified claims
        if query_lower.contains("likely")
            || query_lower.contains("probably")
            || query_lower.contains("may")
            || query_lower.contains("might")
        {
            strengths.push(IdentifiedStrength {
                description: "Uses appropriate epistemic qualifiers".to_string(),
                significance: 0.60,
            });
        }

        // Check for structured reasoning
        if query_lower.contains("first")
            || query_lower.contains("second")
            || query_lower.contains("finally")
            || query_lower.contains("therefore")
        {
            strengths.push(IdentifiedStrength {
                description: "Demonstrates structured reasoning approach".to_string(),
                significance: 0.55,
            });
        }

        strengths
    }

    /// Generate a devil's advocate counter-argument.
    fn devils_advocate(&self, query: &str) -> Option<String> {
        if !self.config.enable_devil_advocate {
            return None;
        }

        let query_lower = query.to_lowercase();

        // Generate contextual counter-arguments
        if query_lower.contains("will succeed") || query_lower.contains("will work") {
            Some("What if the underlying assumptions about market conditions, timing, or execution are wrong? What specific failure modes have been considered?".to_string())
        } else if query_lower.contains("best") {
            Some("By what criteria is 'best' defined? Have alternatives been fairly evaluated? Could 'best' be contingent on circumstances?".to_string())
        } else if query_lower.contains("everyone") || query_lower.contains("all") {
            Some("Are there exceptions or edge cases being overlooked? Is this universality actually validated by data?".to_string())
        } else if query_lower.contains("obvious") || query_lower.contains("clearly") {
            Some("What appears obvious from one perspective may not be from another. Have blind spots been systematically checked?".to_string())
        } else {
            Some("What is the strongest argument against this position? What would make this claim false?".to_string())
        }
    }

    /// Calculate skeptical confidence score.
    fn calculate_skeptical_confidence(
        &self,
        flaws: &[DetectedFlaw],
        strengths: &[IdentifiedStrength],
    ) -> f64 {
        // Start with base confidence
        let mut confidence = 0.75;

        // Apply flaw penalties
        for flaw in flaws {
            confidence -= flaw.severity.confidence_penalty();
        }

        // Apply strength bonuses (but with diminishing returns)
        let mut strength_bonus = 0.0;
        for (i, strength) in strengths.iter().enumerate() {
            // Diminishing returns: each subsequent strength adds less
            let diminish_factor = 1.0 / (1.0 + i as f64 * 0.5);
            strength_bonus += strength.significance * 0.1 * diminish_factor;
        }
        confidence += strength_bonus;

        // Apply severity-based skepticism multiplier
        confidence *= self.config.severity.skepticism_multiplier();

        // Clamp to valid range
        confidence.clamp(0.0, 0.95) // Never return 1.0 - always leave room for doubt
    }

    /// Determine the overall verdict.
    fn determine_verdict(&self, confidence: f64, flaws: &[DetectedFlaw]) -> CritiqueVerdict {
        // Count critical and major flaws
        let critical_count = flaws
            .iter()
            .filter(|f| f.severity == FlawSeverity::Critical)
            .count();
        let major_count = flaws
            .iter()
            .filter(|f| f.severity == FlawSeverity::Major)
            .count();

        if critical_count > 0 {
            return CritiqueVerdict::Flawed;
        }

        if major_count >= 3 || confidence < 0.30 {
            return CritiqueVerdict::Flawed;
        }

        if major_count >= 1 || confidence < 0.50 {
            return CritiqueVerdict::Weak;
        }

        if flaws.len() >= 3 || confidence < 0.70 {
            return CritiqueVerdict::Promising;
        }

        if confidence >= 0.70 && flaws.len() <= 2 {
            return CritiqueVerdict::Solid;
        }

        CritiqueVerdict::Promising
    }

    /// Generate the most critical fix recommendation.
    fn critical_fix(&self, flaws: &[DetectedFlaw]) -> Option<String> {
        // Find the most severe flaw
        flaws.iter().max_by_key(|f| &f.severity).and_then(|f| {
            f.remediation.clone().or_else(|| {
                Some(format!(
                    "Address {} issue: {}",
                    match f.category {
                        FlawCategory::Logical => "logical",
                        FlawCategory::Evidential => "evidential",
                        FlawCategory::Assumption => "assumption",
                        FlawCategory::Scope => "scope",
                        FlawCategory::Temporal => "temporal",
                        FlawCategory::Adversarial => "adversarial",
                        FlawCategory::Completeness => "completeness",
                    },
                    f.description
                ))
            })
        })
    }
}

impl ThinkToolModule for BrutalHonesty {
    fn config(&self) -> &ThinkToolModuleConfig {
        &self.module_config
    }

    fn execute(&self, context: &ThinkToolContext) -> Result<ThinkToolOutput> {
        // Validate input
        if context.query.trim().is_empty() {
            return Err(Error::validation(
                "BrutalHonesty requires non-empty query input",
            ));
        }

        // Perform adversarial analysis
        let assumptions = self.extract_assumptions(&context.query);
        let flaws = self.detect_flaws(&context.query, &context.previous_steps);
        let strengths = self.identify_strengths(&context.query);
        let devils_advocate = self.devils_advocate(&context.query);

        // Calculate skeptical confidence
        let confidence = self.calculate_skeptical_confidence(&flaws, &strengths);

        // Determine verdict
        let verdict = self.determine_verdict(confidence, &flaws);

        // Get critical fix
        let critical_fix = self.critical_fix(&flaws);

        // Build confidence warning
        let confidence_warning = if confidence < self.config.min_confidence_threshold {
            Some(format!(
                "Confidence {:.0}% is below threshold {:.0}%",
                confidence * 100.0,
                self.config.min_confidence_threshold * 100.0
            ))
        } else {
            None
        };

        // Construct output
        let output = json!({
            "verdict": verdict,
            "confidence": confidence,
            "confidence_warning": confidence_warning,
            "severity_applied": format!("{:?}", self.config.severity),
            "analysis": {
                "assumptions": assumptions,
                "flaws": flaws,
                "strengths": strengths,
                "flaw_count": flaws.len(),
                "strength_count": strengths.len(),
            },
            "devils_advocate": devils_advocate,
            "critical_fix": critical_fix,
            "metadata": {
                "input_length": context.query.len(),
                "previous_steps_count": context.previous_steps.len(),
                "skepticism_multiplier": self.config.severity.skepticism_multiplier(),
            }
        });

        Ok(ThinkToolOutput {
            module: self.module_config.name.clone(),
            confidence,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_module() {
        let module = BrutalHonesty::new();
        assert_eq!(module.config().name, "BrutalHonesty");
        assert_eq!(module.config().version, "3.0.0");
        assert_eq!(module.brutal_config().severity, CritiqueSeverity::Standard);
    }

    #[test]
    fn test_builder_pattern() {
        let module = BrutalHonesty::builder()
            .severity(CritiqueSeverity::Ruthless)
            .enable_devil_advocate(false)
            .min_confidence_threshold(0.60)
            .build();

        assert_eq!(module.brutal_config().severity, CritiqueSeverity::Ruthless);
        assert!(!module.brutal_config().enable_devil_advocate);
        assert!((module.brutal_config().min_confidence_threshold - 0.60).abs() < 0.001);
    }

    #[test]
    fn test_assumption_extraction() {
        let module = BrutalHonesty::new();
        let assumptions = module.extract_assumptions("Our product will always be the best");

        assert!(!assumptions.is_empty());
        let has_will = assumptions.iter().any(|a| a.assumption.contains("will"));
        let has_always = assumptions.iter().any(|a| a.assumption.contains("always"));
        let has_best = assumptions.iter().any(|a| a.assumption.contains("best"));

        assert!(has_will || has_always || has_best);
    }

    #[test]
    fn test_flaw_detection() {
        let module = BrutalHonesty::new();
        let flaws = module.detect_flaws(
            "Either we succeed or we fail completely. All experts agree this is amazing.",
            &[],
        );

        assert!(!flaws.is_empty());

        // Should detect false dichotomy
        let has_dichotomy = flaws.iter().any(|f| f.category == FlawCategory::Logical);
        assert!(has_dichotomy);

        // Should detect overgeneralization
        let has_scope = flaws.iter().any(|f| f.category == FlawCategory::Scope);
        assert!(has_scope);
    }

    #[test]
    fn test_strength_identification() {
        let module = BrutalHonesty::new();
        let strengths = module.identify_strengths(
            "The data shows, for example, that our approach is likely effective. However, there are limitations.",
        );

        assert!(!strengths.is_empty());
        assert!(strengths.len() >= 2); // data + for example + however + likely
    }

    #[test]
    fn test_skeptical_confidence() {
        let module = BrutalHonesty::new();

        // With flaws, confidence should be reduced
        let flaws = vec![DetectedFlaw::new(
            FlawCategory::Logical,
            FlawSeverity::Major,
            "Test flaw",
        )];
        let confidence = module.calculate_skeptical_confidence(&flaws, &[]);
        assert!(confidence < 0.60); // Major flaw + skepticism should reduce significantly

        // No flaws with strengths should be higher
        let strengths = vec![IdentifiedStrength {
            description: "Test strength".to_string(),
            significance: 0.8,
        }];
        let confidence_with_strength = module.calculate_skeptical_confidence(&[], &strengths);
        assert!(confidence_with_strength > confidence);
    }

    #[test]
    fn test_verdict_determination() {
        let module = BrutalHonesty::new();

        // Critical flaw should result in Flawed verdict
        let flaws = vec![DetectedFlaw::new(
            FlawCategory::Logical,
            FlawSeverity::Critical,
            "Critical issue",
        )];
        let verdict = module.determine_verdict(0.5, &flaws);
        assert_eq!(verdict, CritiqueVerdict::Flawed);

        // No flaws with high confidence should be Solid
        let verdict = module.determine_verdict(0.85, &[]);
        assert_eq!(verdict, CritiqueVerdict::Solid);
    }

    #[test]
    fn test_execute_empty_input() {
        let module = BrutalHonesty::new();
        let context = ThinkToolContext {
            query: "".to_string(),
            previous_steps: vec![],
        };

        let result = module.execute(&context);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_valid_input() {
        let module = BrutalHonesty::new();
        let context = ThinkToolContext {
            query: "Our startup will succeed because we have the best team".to_string(),
            previous_steps: vec![],
        };

        let result = module.execute(&context).unwrap();
        assert_eq!(result.module, "BrutalHonesty");
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 0.95);

        // Check output structure
        let output = &result.output;
        assert!(output.get("verdict").is_some());
        assert!(output.get("analysis").is_some());
        assert!(output.get("devils_advocate").is_some());
    }

    #[test]
    fn test_severity_affects_confidence() {
        let gentle = BrutalHonesty::builder()
            .severity(CritiqueSeverity::Gentle)
            .build();
        let ruthless = BrutalHonesty::builder()
            .severity(CritiqueSeverity::Ruthless)
            .build();

        let context = ThinkToolContext {
            query: "This approach will work well".to_string(),
            previous_steps: vec![],
        };

        let gentle_result = gentle.execute(&context).unwrap();
        let ruthless_result = ruthless.execute(&context).unwrap();

        // Ruthless should have lower confidence
        assert!(ruthless_result.confidence < gentle_result.confidence);
    }

    #[test]
    fn test_devils_advocate() {
        let module = BrutalHonesty::new();

        // Should generate counter-argument
        let counter = module.devils_advocate("We will succeed");
        assert!(counter.is_some());
        assert!(counter.unwrap().contains("?"));

        // With devil's advocate disabled
        let no_devil = BrutalHonesty::builder()
            .enable_devil_advocate(false)
            .build();
        let counter = no_devil.devils_advocate("We will succeed");
        assert!(counter.is_none());
    }

    #[test]
    fn test_flaw_severity_ordering() {
        assert!(FlawSeverity::Critical > FlawSeverity::Major);
        assert!(FlawSeverity::Major > FlawSeverity::Moderate);
        assert!(FlawSeverity::Moderate > FlawSeverity::Minor);
    }

    #[test]
    fn test_max_flaws_limit() {
        let module = BrutalHonesty::builder().max_flaws_reported(2).build();

        // Input with many potential flaws
        let flaws = module.detect_flaws(
            "All experts always agree that this amazing product will never fail because it's obviously the best",
            &[],
        );

        assert!(flaws.len() <= 2);
    }
}
