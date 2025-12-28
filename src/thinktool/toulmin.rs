//! # Toulmin Argumentation Model
//!
//! Implements the Toulmin model of argumentation for structured reasoning.
//! Based on Stephen Toulmin's "The Uses of Argument" (1958).
//!
//! ## Scientific Foundation
//!
//! - Toulmin (1958): The Uses of Argument - foundational argumentation theory
//! - NL2FOL (arXiv:2405.02318): Achieves 78-80% F1 on fallacy detection
//!
//! ## The Six Components
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    TOULMIN ARGUMENT MODEL                          │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │   GROUNDS ──────────► WARRANT ──────────► CLAIM                    │
//! │   (Evidence)           (Connection)        (Conclusion)            │
//! │       ▲                    │                   ▲                   │
//! │       │                    ▼                   │                   │
//! │       │               BACKING             QUALIFIER                │
//! │       │           (Support for           (Strength)                │
//! │       │             warrant)                 │                     │
//! │       │                                      ▼                     │
//! │       └──────────────────────────────── REBUTTAL                  │
//! │                                       (Exceptions)                 │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::toulmin::{ToulminArgument, ArgumentBuilder};
//!
//! let argument = ArgumentBuilder::new()
//!     .claim("AI will transform education")
//!     .grounds("Studies show 40% improvement in learning outcomes")
//!     .warrant("Personalized learning leads to better outcomes")
//!     .backing("Meta-analysis of 200 educational AI studies")
//!     .qualifier(Qualifier::Probably)
//!     .rebuttal("Unless access inequality persists")
//!     .build()?;
//!
//! let evaluation = argument.evaluate()?;
//! ```

use serde::{Deserialize, Serialize};

/// The six components of a Toulmin argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToulminArgument {
    /// The assertion being made
    pub claim: Claim,
    /// Evidence supporting the claim
    pub grounds: Vec<Ground>,
    /// Logical bridge from grounds to claim
    pub warrant: Option<Warrant>,
    /// Support for the warrant itself
    pub backing: Vec<Backing>,
    /// Strength/certainty modifier
    pub qualifier: Qualifier,
    /// Conditions that would invalidate the claim
    pub rebuttals: Vec<Rebuttal>,
}

/// The main claim/conclusion of the argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    /// The assertion text
    pub statement: String,
    /// Type of claim
    pub claim_type: ClaimType,
    /// Scope of the claim
    pub scope: Scope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimType {
    /// Statement of fact
    Fact,
    /// Value judgment
    Value,
    /// Proposed action/policy
    Policy,
    /// Predicted outcome
    Prediction,
    /// Causal relationship
    Causal,
    /// Definition or classification
    Definition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scope {
    /// Applies to all cases
    Universal,
    /// Applies to most cases
    General,
    /// Applies to some cases
    Particular,
    /// Applies to specific case
    Singular,
}

/// Evidence/data supporting the claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ground {
    /// The evidence statement
    pub evidence: String,
    /// Type of evidence
    pub evidence_type: EvidenceType,
    /// Source of the evidence
    pub source: Option<String>,
    /// Credibility score (0.0-1.0)
    pub credibility: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Statistical data
    Statistical,
    /// Expert testimony
    Testimonial,
    /// Concrete example
    Example,
    /// Documentary evidence
    Documentary,
    /// Physical/empirical evidence
    Empirical,
    /// Common knowledge
    CommonGround,
    /// Analogical reasoning
    Analogical,
}

/// Logical principle connecting grounds to claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Warrant {
    /// The principle/rule being invoked
    pub principle: String,
    /// Type of warrant
    pub warrant_type: WarrantType,
    /// Whether this is explicit or implicit
    pub is_explicit: bool,
    /// Strength of the warrant
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarrantType {
    /// Based on authority
    Authority,
    /// Based on cause-effect
    Causal,
    /// Based on classification
    Classification,
    /// Based on signs/indicators
    Sign,
    /// Based on comparison
    Comparison,
    /// Based on generalization
    Generalization,
    /// Based on principle/rule
    Principle,
}

/// Support for the warrant itself
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Backing {
    /// The supporting statement
    pub support: String,
    /// Type of backing
    pub backing_type: BackingType,
    /// Source if applicable
    pub source: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackingType {
    /// Legal or regulatory
    Legal,
    /// Scientific research
    Scientific,
    /// Historical precedent
    Historical,
    /// Cultural norm
    Cultural,
    /// Expert consensus
    Consensus,
}

/// Qualifier - degree of certainty
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Qualifier {
    /// Certainly true (100%)
    Certainly,
    /// Very likely (90%+)
    Presumably,
    /// Probably (70%+)
    Probably,
    /// Possibly (50%+)
    Possibly,
    /// Unlikely (<50%)
    Unlikely,
    /// Qualified by specific condition
    Conditionally,
}

impl Qualifier {
    pub fn confidence(&self) -> f32 {
        match self {
            Qualifier::Certainly => 0.99,
            Qualifier::Presumably => 0.90,
            Qualifier::Probably => 0.75,
            Qualifier::Possibly => 0.50,
            Qualifier::Unlikely => 0.25,
            Qualifier::Conditionally => 0.60,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Qualifier::Certainly => "certainly",
            Qualifier::Presumably => "presumably",
            Qualifier::Probably => "probably",
            Qualifier::Possibly => "possibly",
            Qualifier::Unlikely => "unlikely",
            Qualifier::Conditionally => "if conditions hold",
        }
    }
}

/// Conditions that would invalidate the claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rebuttal {
    /// The exception/condition
    pub exception: String,
    /// How likely is this exception
    pub likelihood: f32,
    /// Severity if exception occurs
    pub severity: RebuttalSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebuttalSeverity {
    /// Minor qualification
    Minor,
    /// Moderate impact
    Moderate,
    /// Major impact on claim
    Major,
    /// Completely defeats claim
    Fatal,
}

/// Builder for constructing Toulmin arguments
#[derive(Debug, Default)]
pub struct ArgumentBuilder {
    claim: Option<Claim>,
    grounds: Vec<Ground>,
    warrant: Option<Warrant>,
    backing: Vec<Backing>,
    qualifier: Option<Qualifier>,
    rebuttals: Vec<Rebuttal>,
}

impl ArgumentBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn claim(mut self, statement: impl Into<String>) -> Self {
        self.claim = Some(Claim {
            statement: statement.into(),
            claim_type: ClaimType::Fact,
            scope: Scope::General,
        });
        self
    }

    pub fn claim_full(
        mut self,
        statement: impl Into<String>,
        claim_type: ClaimType,
        scope: Scope,
    ) -> Self {
        self.claim = Some(Claim {
            statement: statement.into(),
            claim_type,
            scope,
        });
        self
    }

    pub fn grounds(mut self, evidence: impl Into<String>) -> Self {
        self.grounds.push(Ground {
            evidence: evidence.into(),
            evidence_type: EvidenceType::Empirical,
            source: None,
            credibility: 0.7,
        });
        self
    }

    pub fn grounds_full(
        mut self,
        evidence: impl Into<String>,
        evidence_type: EvidenceType,
        source: Option<String>,
        credibility: f32,
    ) -> Self {
        self.grounds.push(Ground {
            evidence: evidence.into(),
            evidence_type,
            source,
            credibility,
        });
        self
    }

    pub fn warrant(mut self, principle: impl Into<String>) -> Self {
        self.warrant = Some(Warrant {
            principle: principle.into(),
            warrant_type: WarrantType::Principle,
            is_explicit: true,
            strength: 0.8,
        });
        self
    }

    pub fn warrant_full(
        mut self,
        principle: impl Into<String>,
        warrant_type: WarrantType,
        strength: f32,
    ) -> Self {
        self.warrant = Some(Warrant {
            principle: principle.into(),
            warrant_type,
            is_explicit: true,
            strength,
        });
        self
    }

    pub fn backing(mut self, support: impl Into<String>) -> Self {
        self.backing.push(Backing {
            support: support.into(),
            backing_type: BackingType::Scientific,
            source: None,
        });
        self
    }

    pub fn qualifier(mut self, qualifier: Qualifier) -> Self {
        self.qualifier = Some(qualifier);
        self
    }

    pub fn rebuttal(mut self, exception: impl Into<String>) -> Self {
        self.rebuttals.push(Rebuttal {
            exception: exception.into(),
            likelihood: 0.3,
            severity: RebuttalSeverity::Moderate,
        });
        self
    }

    pub fn rebuttal_full(
        mut self,
        exception: impl Into<String>,
        likelihood: f32,
        severity: RebuttalSeverity,
    ) -> Self {
        self.rebuttals.push(Rebuttal {
            exception: exception.into(),
            likelihood,
            severity,
        });
        self
    }

    pub fn build(self) -> Result<ToulminArgument, ArgumentError> {
        let claim = self.claim.ok_or(ArgumentError::MissingClaim)?;

        if self.grounds.is_empty() {
            return Err(ArgumentError::MissingGrounds);
        }

        Ok(ToulminArgument {
            claim,
            grounds: self.grounds,
            warrant: self.warrant,
            backing: self.backing,
            qualifier: self.qualifier.unwrap_or(Qualifier::Probably),
            rebuttals: self.rebuttals,
        })
    }
}

#[derive(Debug, Clone)]
pub enum ArgumentError {
    MissingClaim,
    MissingGrounds,
    InvalidWarrant(String),
}

impl std::fmt::Display for ArgumentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArgumentError::MissingClaim => write!(f, "Argument requires a claim"),
            ArgumentError::MissingGrounds => write!(f, "Argument requires at least one ground"),
            ArgumentError::InvalidWarrant(msg) => write!(f, "Invalid warrant: {}", msg),
        }
    }
}

impl std::error::Error for ArgumentError {}

/// Evaluation of an argument's strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentEvaluation {
    /// Overall argument strength (0.0-1.0)
    pub overall_strength: f32,
    /// Grounds quality
    pub grounds_score: f32,
    /// Warrant validity
    pub warrant_score: f32,
    /// Backing support
    pub backing_score: f32,
    /// Impact of rebuttals
    pub rebuttal_impact: f32,
    /// Issues found
    pub issues: Vec<ArgumentIssue>,
    /// Is the argument valid?
    pub is_valid: bool,
    /// Is the argument sound?
    pub is_sound: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentIssue {
    pub component: ToulminComponent,
    pub issue: String,
    pub severity: IssueSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToulminComponent {
    Claim,
    Grounds,
    Warrant,
    Backing,
    Qualifier,
    Rebuttal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Minor,
    Moderate,
    Serious,
    Critical,
}

impl ToulminArgument {
    /// Evaluate the argument's strength
    pub fn evaluate(&self) -> ArgumentEvaluation {
        let mut issues = Vec::new();

        // Score grounds
        let grounds_score = if self.grounds.is_empty() {
            issues.push(ArgumentIssue {
                component: ToulminComponent::Grounds,
                issue: "No supporting evidence provided".into(),
                severity: IssueSeverity::Critical,
            });
            0.0
        } else {
            self.grounds.iter().map(|g| g.credibility).sum::<f32>() / self.grounds.len() as f32
        };

        // Score warrant
        let warrant_score = if let Some(ref w) = self.warrant {
            if !w.is_explicit {
                issues.push(ArgumentIssue {
                    component: ToulminComponent::Warrant,
                    issue: "Warrant is implicit - should be made explicit".into(),
                    severity: IssueSeverity::Minor,
                });
            }
            w.strength
        } else {
            issues.push(ArgumentIssue {
                component: ToulminComponent::Warrant,
                issue: "Missing warrant connecting evidence to claim".into(),
                severity: IssueSeverity::Serious,
            });
            0.3 // Implicit warrant assumed
        };

        // Score backing
        let backing_score = if self.backing.is_empty() {
            0.5 // Neutral - backing not always needed
        } else {
            0.7 + 0.1 * self.backing.len().min(3) as f32
        };

        // Calculate rebuttal impact
        let rebuttal_impact: f32 = self
            .rebuttals
            .iter()
            .map(|r| {
                let severity_weight = match r.severity {
                    RebuttalSeverity::Minor => 0.1,
                    RebuttalSeverity::Moderate => 0.25,
                    RebuttalSeverity::Major => 0.5,
                    RebuttalSeverity::Fatal => 1.0,
                };
                r.likelihood * severity_weight
            })
            .sum::<f32>()
            .min(0.8); // Cap at 80% reduction

        // Add issues for high-impact rebuttals
        for rebuttal in &self.rebuttals {
            if rebuttal.likelihood > 0.5 && rebuttal.severity == RebuttalSeverity::Fatal {
                issues.push(ArgumentIssue {
                    component: ToulminComponent::Rebuttal,
                    issue: format!("High likelihood fatal rebuttal: {}", rebuttal.exception),
                    severity: IssueSeverity::Critical,
                });
            }
        }

        // Overall strength calculation
        let base_strength =
            grounds_score * 0.35 + warrant_score * 0.30 + backing_score * 0.20 + 0.15;

        let qualified_strength = base_strength * self.qualifier.confidence();
        let overall_strength = (qualified_strength * (1.0 - rebuttal_impact)).max(0.0);

        // Determine validity and soundness
        let is_valid = warrant_score >= 0.5 && grounds_score > 0.0;
        let is_sound = is_valid
            && grounds_score >= 0.6
            && !issues.iter().any(|i| i.severity == IssueSeverity::Critical);

        ArgumentEvaluation {
            overall_strength,
            grounds_score,
            warrant_score,
            backing_score,
            rebuttal_impact,
            issues,
            is_valid,
            is_sound,
        }
    }

    /// Format as structured text
    pub fn format(&self) -> String {
        let mut output = String::new();

        output
            .push_str("┌─────────────────────────────────────────────────────────────────────┐\n");
        output
            .push_str("│                    TOULMIN ARGUMENT STRUCTURE                       │\n");
        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");

        // Claim
        output.push_str(&format!(
            "│ CLAIM ({:?}, {:?}):                                                   \n",
            self.claim.claim_type, self.claim.scope
        ));
        output.push_str(&format!(
            "│   {} {}\n",
            self.qualifier.label(),
            self.claim.statement
        ));

        // Grounds
        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");
        output.push_str("│ GROUNDS (Evidence):                                                 \n");
        for ground in &self.grounds {
            output.push_str(&format!(
                "│   • [{}] {} (credibility: {:.0}%)\n",
                format!("{:?}", ground.evidence_type).to_uppercase(),
                ground.evidence,
                ground.credibility * 100.0
            ));
        }

        // Warrant
        if let Some(ref warrant) = self.warrant {
            output.push_str(
                "├─────────────────────────────────────────────────────────────────────┤\n",
            );
            output.push_str(&format!(
                "│ WARRANT ({:?}, strength: {:.0}%):                                   \n",
                warrant.warrant_type,
                warrant.strength * 100.0
            ));
            output.push_str(&format!("│   {}\n", warrant.principle));
        }

        // Backing
        if !self.backing.is_empty() {
            output.push_str(
                "├─────────────────────────────────────────────────────────────────────┤\n",
            );
            output.push_str(
                "│ BACKING:                                                            \n",
            );
            for backing in &self.backing {
                output.push_str(&format!("│   • {}\n", backing.support));
            }
        }

        // Rebuttals
        if !self.rebuttals.is_empty() {
            output.push_str(
                "├─────────────────────────────────────────────────────────────────────┤\n",
            );
            output.push_str(
                "│ REBUTTALS (Exceptions):                                             \n",
            );
            for rebuttal in &self.rebuttals {
                output.push_str(&format!(
                    "│   • UNLESS: {} ({:?}, {:.0}% likely)\n",
                    rebuttal.exception,
                    rebuttal.severity,
                    rebuttal.likelihood * 100.0
                ));
            }
        }

        output
            .push_str("└─────────────────────────────────────────────────────────────────────┘\n");

        output
    }
}

/// Prompt templates for generating Toulmin arguments
pub struct ToulminPrompts;

impl ToulminPrompts {
    /// Generate a Toulmin analysis of a claim
    pub fn analyze_claim(claim: &str) -> String {
        format!(
            r#"Analyze this claim using the Toulmin model of argumentation.

CLAIM: {claim}

Provide a structured analysis with:

1. CLAIM CLASSIFICATION
   - Type: Fact/Value/Policy/Prediction/Causal/Definition
   - Scope: Universal/General/Particular/Singular

2. GROUNDS (Evidence needed)
   - What evidence would support this claim?
   - What type of evidence (Statistical/Testimonial/Example/Documentary/Empirical)?
   - Rate credibility (0-100%)

3. WARRANT (Logical bridge)
   - What principle connects the evidence to the claim?
   - Type: Authority/Causal/Classification/Sign/Comparison/Generalization/Principle
   - Is it explicit or assumed?

4. BACKING (Warrant support)
   - What supports the warrant itself?
   - Source type: Legal/Scientific/Historical/Cultural/Consensus

5. QUALIFIER (Certainty level)
   - How certain is this claim? (Certainly/Presumably/Probably/Possibly/Unlikely)

6. REBUTTALS (Exceptions)
   - What conditions would invalidate this claim?
   - How likely are these exceptions?
   - Severity: Minor/Moderate/Major/Fatal

Respond in JSON format."#,
            claim = claim
        )
    }

    /// Evaluate argument strength
    pub fn evaluate_argument(argument: &str) -> String {
        format!(
            r#"Evaluate this argument for logical strength and soundness.

ARGUMENT:
{argument}

Identify:
1. The main CLAIM
2. The supporting GROUNDS (evidence)
3. The WARRANT (logical connection)
4. Any BACKING for the warrant
5. The QUALIFIER (certainty level)
6. Potential REBUTTALS

Then evaluate:
- Grounds quality (0-100%)
- Warrant validity (0-100%)
- Overall argument strength (0-100%)
- Is it VALID? (logical structure correct)
- Is it SOUND? (valid AND true premises)

List any logical fallacies or weaknesses found.

Respond in JSON format."#,
            argument = argument
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argument_builder() {
        let argument = ArgumentBuilder::new()
            .claim("Climate change is caused by human activity")
            .grounds("CO2 levels have risen 50% since industrialization")
            .warrant("CO2 is a greenhouse gas that traps heat")
            .backing("Established physics of radiative forcing")
            .qualifier(Qualifier::Presumably)
            .rebuttal("Unless natural cycles are the primary driver")
            .build()
            .unwrap();

        assert_eq!(argument.grounds.len(), 1);
        assert!(argument.warrant.is_some());
        assert_eq!(argument.rebuttals.len(), 1);
    }

    #[test]
    fn test_argument_evaluation() {
        let argument = ArgumentBuilder::new()
            .claim("Regular exercise improves health")
            .grounds_full(
                "Meta-analysis of 100 studies shows 30% reduction in mortality",
                EvidenceType::Statistical,
                Some("Lancet 2023".into()),
                0.9,
            )
            .warrant_full(
                "Physical activity strengthens cardiovascular system",
                WarrantType::Causal,
                0.95,
            )
            .backing("Established medical consensus")
            .qualifier(Qualifier::Presumably)
            .build()
            .unwrap();

        let eval = argument.evaluate();

        assert!(eval.is_valid);
        assert!(eval.is_sound);
        assert!(eval.overall_strength > 0.7);
    }

    #[test]
    fn test_weak_argument() {
        let argument = ArgumentBuilder::new()
            .claim("All swans are white")
            .grounds_full(
                "I've only seen white swans",
                EvidenceType::Example,
                None,
                0.3,
            )
            .qualifier(Qualifier::Certainly)
            .rebuttal_full(
                "Black swans exist in Australia",
                0.9,
                RebuttalSeverity::Fatal,
            )
            .build()
            .unwrap();

        let eval = argument.evaluate();

        assert!(!eval.is_sound);
        assert!(eval.overall_strength < 0.5);
        assert!(!eval.issues.is_empty());
    }

    #[test]
    fn test_qualifier_confidence() {
        assert!(Qualifier::Certainly.confidence() > Qualifier::Probably.confidence());
        assert!(Qualifier::Probably.confidence() > Qualifier::Possibly.confidence());
    }
}
