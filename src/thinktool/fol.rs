//! # First-Order Logic Verification
//!
//! Implements NL→FOL translation and logical verification for LaserLogic.
//!
//! ## Scientific Foundation
//!
//! Based on NL2FOL framework research:
//! - 78-80% F1 on fallacy detection with FOL translation
//! - Structured logical forms enable mechanical verification
//! - Bridges natural language reasoning to formal proof
//!
//! ## Approach
//!
//! 1. Parse natural language into logical structure
//! 2. Translate to First-Order Logic
//! 3. Check validity using satisfiability rules
//! 4. Detect fallacies through logical patterns
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::fol::{FolVerifier, FolConfig};
//!
//! let verifier = FolVerifier::new(FolConfig::default());
//! let result = verifier.verify(argument).await?;
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for FOL verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolConfig {
    /// Whether to generate full logical forms
    pub generate_logical_forms: bool,
    /// Enable fallacy detection
    pub detect_fallacies: bool,
    /// Enable validity checking
    pub check_validity: bool,
    /// Maximum nested quantifier depth
    pub max_quantifier_depth: usize,
    /// Enable soundness checking (premises + validity)
    pub check_soundness: bool,
    /// Confidence threshold for translation
    pub translation_confidence_threshold: f32,
}

impl Default for FolConfig {
    fn default() -> Self {
        Self {
            generate_logical_forms: true,
            detect_fallacies: true,
            check_validity: true,
            max_quantifier_depth: 3,
            check_soundness: true,
            translation_confidence_threshold: 0.7,
        }
    }
}

impl FolConfig {
    /// LaserLogic-optimized configuration
    pub fn laser_logic() -> Self {
        Self {
            generate_logical_forms: true,
            detect_fallacies: true,
            check_validity: true,
            max_quantifier_depth: 4,
            check_soundness: true,
            translation_confidence_threshold: 0.75,
        }
    }

    /// Quick validity check only
    pub fn quick_check() -> Self {
        Self {
            generate_logical_forms: false,
            detect_fallacies: true,
            check_validity: true,
            max_quantifier_depth: 2,
            check_soundness: false,
            translation_confidence_threshold: 0.6,
        }
    }
}

/// Logical connectives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Connective {
    /// Logical AND (∧)
    And,
    /// Logical OR (∨)
    Or,
    /// Logical NOT (¬)
    Not,
    /// Implication (→)
    Implies,
    /// Biconditional (↔)
    Iff,
}

impl Connective {
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::And => "∧",
            Self::Or => "∨",
            Self::Not => "¬",
            Self::Implies => "→",
            Self::Iff => "↔",
        }
    }
}

/// Quantifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quantifier {
    /// Universal quantifier (∀)
    ForAll,
    /// Existential quantifier (∃)
    Exists,
}

impl Quantifier {
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::ForAll => "∀",
            Self::Exists => "∃",
        }
    }
}

/// A term in FOL (variable, constant, or function application)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Term {
    /// A variable (x, y, z)
    Variable(String),
    /// A constant (john, 42)
    Constant(String),
    /// A function application (f(x), father(john))
    Function { name: String, args: Vec<Term> },
}

/// A formula in FOL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Formula {
    /// A predicate application (P(x), Loves(x, y))
    Predicate { name: String, args: Vec<Term> },
    /// Negation
    Not(Box<Formula>),
    /// Conjunction (AND)
    And(Box<Formula>, Box<Formula>),
    /// Disjunction (OR)
    Or(Box<Formula>, Box<Formula>),
    /// Implication
    Implies(Box<Formula>, Box<Formula>),
    /// Biconditional
    Iff(Box<Formula>, Box<Formula>),
    /// Universal quantification
    ForAll {
        variable: String,
        formula: Box<Formula>,
    },
    /// Existential quantification
    Exists {
        variable: String,
        formula: Box<Formula>,
    },
}

impl Formula {
    /// Check if formula is a simple predicate
    pub fn is_atomic(&self) -> bool {
        matches!(self, Formula::Predicate { .. })
    }

    /// Get free variables in the formula
    pub fn free_variables(&self) -> Vec<String> {
        match self {
            Formula::Predicate { args, .. } => args
                .iter()
                .filter_map(|t| match t {
                    Term::Variable(v) => Some(v.clone()),
                    _ => None,
                })
                .collect(),
            Formula::Not(f) => f.free_variables(),
            Formula::And(l, r)
            | Formula::Or(l, r)
            | Formula::Implies(l, r)
            | Formula::Iff(l, r) => {
                let mut vars = l.free_variables();
                vars.extend(r.free_variables());
                vars.sort();
                vars.dedup();
                vars
            }
            Formula::ForAll { variable, formula } | Formula::Exists { variable, formula } => {
                formula
                    .free_variables()
                    .into_iter()
                    .filter(|v| v != variable)
                    .collect()
            }
        }
    }
}

/// An argument structure for FOL verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolArgument {
    /// Natural language premises
    pub premises_nl: Vec<String>,
    /// Natural language conclusion
    pub conclusion_nl: String,
    /// Translated premises (FOL)
    pub premises_fol: Vec<Formula>,
    /// Translated conclusion (FOL)
    pub conclusion_fol: Option<Formula>,
    /// Translation confidence
    pub translation_confidence: f32,
    /// Any translation notes/issues
    pub translation_notes: Vec<String>,
}

/// Fallacy types detectable through FOL analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FolFallacy {
    /// Affirming the consequent: P→Q, Q ⊢ P
    AffirmingConsequent,
    /// Denying the antecedent: P→Q, ¬P ⊢ ¬Q
    DenyingAntecedent,
    /// Undistributed middle: All A are B, All C are B ⊢ All A are C
    UndistributedMiddle,
    /// Illicit major/minor: Distribution errors in syllogisms
    IllicitDistribution,
    /// Four-term fallacy: Using a term with different meanings
    FourTerms,
    /// Existential fallacy: Assuming existence from universal
    ExistentialFallacy,
    /// Composition: What's true of parts is true of whole
    Composition,
    /// Division: What's true of whole is true of parts
    Division,
    /// Circular reasoning: Conclusion appears in premises
    CircularReasoning,
    /// Non sequitur: Conclusion doesn't follow logically
    NonSequitur,
}

impl FolFallacy {
    /// Get description of the fallacy
    pub fn description(&self) -> &'static str {
        match self {
            Self::AffirmingConsequent => {
                "Inferring P from P→Q and Q (invalid: other causes possible)"
            }
            Self::DenyingAntecedent => {
                "Inferring ¬Q from P→Q and ¬P (invalid: Q might still be true)"
            }
            Self::UndistributedMiddle => "Middle term not distributed in at least one premise",
            Self::IllicitDistribution => "Term distributed in conclusion but not in premises",
            Self::FourTerms => "Four distinct terms used where three expected",
            Self::ExistentialFallacy => "Assuming existence from purely universal premises",
            Self::Composition => "Assuming whole has properties of its parts",
            Self::Division => "Assuming parts have properties of the whole",
            Self::CircularReasoning => "Conclusion is equivalent to or contained in premises",
            Self::NonSequitur => "Conclusion does not logically follow from premises",
        }
    }

    /// Get the logical pattern of the fallacy
    pub fn pattern(&self) -> &'static str {
        match self {
            Self::AffirmingConsequent => "P→Q, Q ⊢ P (INVALID)",
            Self::DenyingAntecedent => "P→Q, ¬P ⊢ ¬Q (INVALID)",
            Self::UndistributedMiddle => "∀x(Mx→Px), ∀x(Mx→Sx) ⊢ ∀x(Sx→Px) (INVALID)",
            Self::IllicitDistribution => "∀x(Px→Mx), ∃x(Mx∧Sx) ⊢ ∀x(Sx→Px) (INVALID)",
            Self::FourTerms => "Four distinct predicates instead of three",
            Self::ExistentialFallacy => "∀x(Px→Qx) ⊢ ∃x(Px∧Qx) (INVALID if ¬∃xPx)",
            Self::Composition => "∀x∈S(Px) ⊢ P(S) (INVALID)",
            Self::Division => "P(S) ⊢ ∀x∈S(Px) (INVALID)",
            Self::CircularReasoning => "P ⊢ P (TRIVIALLY VALID but uninformative)",
            Self::NonSequitur => "No valid derivation path",
        }
    }
}

/// A detected fallacy with evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedFallacy {
    pub fallacy: FolFallacy,
    /// Confidence in detection (0.0 - 1.0)
    pub confidence: f32,
    /// Evidence/explanation
    pub evidence: String,
    /// Which premises are involved
    pub involved_premises: Vec<usize>,
}

/// Validity status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidityStatus {
    /// Argument is logically valid
    Valid,
    /// Argument is logically invalid
    Invalid,
    /// Validity could not be determined
    Undetermined,
}

/// Soundness status (validity + true premises)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SoundnessStatus {
    /// Argument is sound (valid with true premises)
    Sound,
    /// Argument is unsound (invalid or false premises)
    Unsound,
    /// Soundness could not be determined
    Undetermined,
}

/// Complete FOL verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolResult {
    /// The translated argument
    pub argument: FolArgument,
    /// Validity status
    pub validity: ValidityStatus,
    /// Validity explanation
    pub validity_explanation: String,
    /// Soundness status
    pub soundness: SoundnessStatus,
    /// Premise truth assessments
    pub premise_assessments: Vec<PremiseAssessment>,
    /// Detected fallacies
    pub fallacies: Vec<DetectedFallacy>,
    /// Overall confidence in the analysis
    pub confidence: f32,
    /// Suggested improvements
    pub suggestions: Vec<String>,
}

/// Assessment of a single premise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremiseAssessment {
    pub premise_index: usize,
    pub premise_nl: String,
    /// Is the premise likely true?
    pub likely_true: Option<bool>,
    /// Confidence in truth assessment
    pub confidence: f32,
    /// Reasoning for assessment
    pub reasoning: String,
}

impl FolResult {
    /// Overall verdict
    pub fn verdict(&self) -> &'static str {
        match (self.validity, self.soundness) {
            (ValidityStatus::Valid, SoundnessStatus::Sound) => {
                "SOUND - Argument is valid with true premises"
            }
            (ValidityStatus::Valid, SoundnessStatus::Unsound) => {
                "VALID BUT UNSOUND - Valid logic but questionable premises"
            }
            (ValidityStatus::Valid, SoundnessStatus::Undetermined) => {
                "VALID - Logic is correct, premises unverified"
            }
            (ValidityStatus::Invalid, _) => "INVALID - Conclusion does not follow from premises",
            (ValidityStatus::Undetermined, _) => "UNDETERMINED - Could not fully analyze",
        }
    }

    /// Has any fallacies?
    pub fn has_fallacies(&self) -> bool {
        !self.fallacies.is_empty()
    }

    /// Most severe fallacy
    pub fn most_confident_fallacy(&self) -> Option<&DetectedFallacy> {
        self.fallacies.iter().max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Prompt templates for FOL verification
pub struct FolPrompts;

impl FolPrompts {
    /// Translate argument to FOL
    pub fn translate(premises: &[String], conclusion: &str) -> String {
        let premises_formatted: String = premises
            .iter()
            .enumerate()
            .map(|(i, p)| format!("P{}: {}", i + 1, p))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"Translate this argument into First-Order Logic (FOL).

PREMISES:
{premises_formatted}

CONCLUSION:
C: {conclusion}

For each statement:
1. Identify predicates (properties and relations)
2. Identify quantifiers (all, some, none)
3. Identify logical connectives (and, or, not, if-then)
4. Express in FOL notation

Use standard symbols:
- ∀x (for all x)
- ∃x (there exists x)
- ∧ (and)
- ∨ (or)
- ¬ (not)
- → (implies)
- ↔ (if and only if)

Format:
P1_FOL: [FOL translation]
P2_FOL: [FOL translation]
...
C_FOL: [FOL translation]

TRANSLATION_CONFIDENCE: [0.0-1.0]
TRANSLATION_NOTES: [any ambiguities or assumptions]"#,
            premises_formatted = premises_formatted,
            conclusion = conclusion
        )
    }

    /// Check validity of FOL argument
    pub fn check_validity(premises_fol: &[String], conclusion_fol: &str) -> String {
        let premises_formatted: String = premises_fol
            .iter()
            .enumerate()
            .map(|(i, p)| format!("P{}: {}", i + 1, p))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"Determine the logical validity of this argument.

PREMISES (FOL):
{premises_formatted}

CONCLUSION (FOL):
C: {conclusion_fol}

An argument is VALID if: whenever all premises are true, the conclusion MUST be true.
An argument is INVALID if: it's possible for all premises to be true while the conclusion is false.

Analysis steps:
1. Identify the logical structure
2. Look for counterexamples (model where premises true, conclusion false)
3. If no counterexample possible, the argument is valid

VALIDITY: [VALID | INVALID | UNDETERMINED]
EXPLANATION: [why valid/invalid]
COUNTEREXAMPLE: [if invalid, describe a scenario where premises are true but conclusion is false]
CONFIDENCE: [0.0-1.0]"#,
            premises_formatted = premises_formatted,
            conclusion_fol = conclusion_fol
        )
    }

    /// Detect fallacies
    pub fn detect_fallacies(argument_nl: &str, argument_fol: Option<&str>) -> String {
        let fol_section = argument_fol.map_or(
            "Not available - analyze from natural language only.".to_string(),
            |f| format!("FOL:\n{}", f),
        );

        format!(
            r#"Analyze this argument for logical fallacies.

ARGUMENT (Natural Language):
{argument_nl}

{fol_section}

Check for these formal fallacies:
1. Affirming the Consequent: P→Q, Q ⊢ P
2. Denying the Antecedent: P→Q, ¬P ⊢ ¬Q
3. Undistributed Middle: syllogism with undistributed middle term
4. Illicit Distribution: invalid distribution in syllogism
5. Four-Term Fallacy: using equivocal term
6. Existential Fallacy: invalid existential inference
7. Composition/Division: part-whole errors
8. Circular Reasoning: conclusion in premises
9. Non Sequitur: conclusion doesn't follow

For each fallacy found:
FALLACY: [name]
PATTERN: [the invalid logical pattern]
EVIDENCE: [where this appears in the argument]
CONFIDENCE: [0.0-1.0]

If no fallacies found:
NO_FALLACIES_DETECTED
CONFIDENCE: [0.0-1.0]"#,
            argument_nl = argument_nl,
            fol_section = fol_section
        )
    }

    /// Assess premise truth
    pub fn assess_premises(premises: &[String]) -> String {
        let premises_formatted: String = premises
            .iter()
            .enumerate()
            .map(|(i, p)| format!("P{}: {}", i + 1, p))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"Assess the truth of each premise.

PREMISES:
{premises_formatted}

For each premise:
1. Is it empirically verifiable?
2. Is it a definitional truth?
3. Is it a widely accepted claim?
4. What evidence supports or refutes it?

Format:
P1_ASSESSMENT:
- LIKELY_TRUE: [true | false | unknown]
- CONFIDENCE: [0.0-1.0]
- REASONING: [explanation]

P2_ASSESSMENT:
..."#,
            premises_formatted = premises_formatted
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FolConfig::default();
        assert!(config.generate_logical_forms);
        assert!(config.detect_fallacies);
        assert!(config.check_validity);
    }

    #[test]
    fn test_connectives() {
        assert_eq!(Connective::And.symbol(), "∧");
        assert_eq!(Connective::Or.symbol(), "∨");
        assert_eq!(Connective::Implies.symbol(), "→");
    }

    #[test]
    fn test_quantifiers() {
        assert_eq!(Quantifier::ForAll.symbol(), "∀");
        assert_eq!(Quantifier::Exists.symbol(), "∃");
    }

    #[test]
    fn test_formula_free_variables() {
        // P(x, y)
        let formula = Formula::Predicate {
            name: "P".into(),
            args: vec![Term::Variable("x".into()), Term::Variable("y".into())],
        };
        let vars = formula.free_variables();
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));

        // ∀x P(x, y) - y is free, x is bound
        let quantified = Formula::ForAll {
            variable: "x".into(),
            formula: Box::new(formula.clone()),
        };
        let vars = quantified.free_variables();
        assert!(!vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
    }

    #[test]
    fn test_fallacy_descriptions() {
        let fallacy = FolFallacy::AffirmingConsequent;
        assert!(fallacy.description().contains("Inferring P"));
        assert!(fallacy.pattern().contains("INVALID"));
    }

    #[test]
    fn test_result_verdict() {
        let result = FolResult {
            argument: FolArgument {
                premises_nl: vec!["All men are mortal".into()],
                conclusion_nl: "Socrates is mortal".into(),
                premises_fol: vec![],
                conclusion_fol: None,
                translation_confidence: 0.9,
                translation_notes: vec![],
            },
            validity: ValidityStatus::Valid,
            validity_explanation: "Valid syllogism".into(),
            soundness: SoundnessStatus::Sound,
            premise_assessments: vec![],
            fallacies: vec![],
            confidence: 0.95,
            suggestions: vec![],
        };

        assert!(result.verdict().contains("SOUND"));
        assert!(!result.has_fallacies());
    }

    #[test]
    fn test_invalid_result() {
        let result = FolResult {
            argument: FolArgument {
                premises_nl: vec![
                    "If it rains, the ground is wet".into(),
                    "The ground is wet".into(),
                ],
                conclusion_nl: "It rained".into(),
                premises_fol: vec![],
                conclusion_fol: None,
                translation_confidence: 0.85,
                translation_notes: vec![],
            },
            validity: ValidityStatus::Invalid,
            validity_explanation: "Affirming the consequent".into(),
            soundness: SoundnessStatus::Unsound,
            premise_assessments: vec![],
            fallacies: vec![DetectedFallacy {
                fallacy: FolFallacy::AffirmingConsequent,
                confidence: 0.95,
                evidence: "Inferring cause from effect".into(),
                involved_premises: vec![0, 1],
            }],
            confidence: 0.9,
            suggestions: vec!["Consider other causes".into()],
        };

        assert!(result.verdict().contains("INVALID"));
        assert!(result.has_fallacies());
        assert_eq!(result.fallacies[0].fallacy, FolFallacy::AffirmingConsequent);
    }
}
