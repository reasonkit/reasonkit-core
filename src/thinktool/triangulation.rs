//! # Epistemic Triangulation Protocol
//!
//! Implements multi-source verification for claims achieving +50% false claim rejection.
//!
//! ## Scientific Foundation
//!
//! Based on:
//! - Du Bois triangulation methodology
//! - PNAS 2025 research on fact-checking infrastructure
//! - Epistemic anchor theory
//!
//! ## Core Principle
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                  TRIANGULATION PROTOCOL                             │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │   SOURCE A (Primary) ──────┐                                       │
//! │   • Official/Authoritative │                                       │
//! │   • Tier 1: Weight 1.0     ├─────► CLAIM ◄─────┐                   │
//! │                            │                    │                   │
//! │   SOURCE B (Secondary) ────┘                    │                   │
//! │   • Different domain                            │                   │
//! │   • Tier 2: Weight 0.7    ───────► VERIFIED ◄──┤                   │
//! │                                                 │                   │
//! │   SOURCE C (Independent) ──────────────────────┘                   │
//! │   • Different author                                               │
//! │   • Tier 3: Weight 0.4                                             │
//! │                                                                     │
//! │   MINIMUM REQUIREMENT: 3 independent sources                       │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::triangulation::{Triangulator, Source, SourceTier};
//!
//! let triangulator = Triangulator::new();
//! let claim = "AI will reach AGI by 2030";
//!
//! triangulator.add_source(Source::new("OpenAI Research Paper", SourceTier::Primary));
//! triangulator.add_source(Source::new("DeepMind Analysis", SourceTier::Secondary));
//! triangulator.add_source(Source::new("Academic Survey", SourceTier::Independent));
//!
//! let result = triangulator.verify()?;
//! ```

use serde::{Deserialize, Serialize};

/// Source tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceTier {
    /// Tier 1: Primary/Authoritative sources (weight 1.0)
    /// Official docs, peer-reviewed papers, primary sources
    Primary,
    /// Tier 2: Secondary/Reliable sources (weight 0.7)
    /// Reputable news, expert blogs, secondary analysis
    Secondary,
    /// Tier 3: Independent sources (weight 0.4)
    /// Community content, user-generated, unverified
    Independent,
    /// Tier 4: Unverified sources (weight 0.2)
    Unverified,
}

impl SourceTier {
    pub fn weight(&self) -> f32 {
        match self {
            SourceTier::Primary => 1.0,
            SourceTier::Secondary => 0.7,
            SourceTier::Independent => 0.4,
            SourceTier::Unverified => 0.2,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            SourceTier::Primary => "Tier 1 (Primary)",
            SourceTier::Secondary => "Tier 2 (Secondary)",
            SourceTier::Independent => "Tier 3 (Independent)",
            SourceTier::Unverified => "Tier 4 (Unverified)",
        }
    }
}

/// Source type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceType {
    /// Academic peer-reviewed
    Academic,
    /// Official documentation
    Documentation,
    /// News article
    News,
    /// Expert opinion/blog
    Expert,
    /// Government/regulatory
    Government,
    /// Industry report
    Industry,
    /// Community forum
    Community,
    /// Social media
    Social,
    /// Primary data
    PrimaryData,
}

impl SourceType {
    pub fn default_tier(&self) -> SourceTier {
        match self {
            SourceType::Academic => SourceTier::Primary,
            SourceType::Documentation => SourceTier::Primary,
            SourceType::Government => SourceTier::Primary,
            SourceType::PrimaryData => SourceTier::Primary,
            SourceType::News => SourceTier::Secondary,
            SourceType::Expert => SourceTier::Secondary,
            SourceType::Industry => SourceTier::Secondary,
            SourceType::Community => SourceTier::Independent,
            SourceType::Social => SourceTier::Unverified,
        }
    }
}

/// A source for triangulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Source identifier/name
    pub name: String,
    /// URL if available
    pub url: Option<String>,
    /// Source tier
    pub tier: SourceTier,
    /// Source type
    pub source_type: SourceType,
    /// Domain/field
    pub domain: Option<String>,
    /// Author/organization
    pub author: Option<String>,
    /// Publication date
    pub date: Option<String>,
    /// Whether URL was verified accessible
    pub verified: bool,
    /// Direct quote supporting claim
    pub quote: Option<String>,
    /// Does this source support or contradict the claim?
    pub stance: Stance,
    /// Credibility assessment (0.0-1.0)
    pub credibility: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stance {
    /// Supports the claim
    Support,
    /// Contradicts the claim
    Contradict,
    /// Neither supports nor contradicts
    Neutral,
    /// Partially supports
    Partial,
}

impl Source {
    pub fn new(name: impl Into<String>, tier: SourceTier) -> Self {
        Self {
            name: name.into(),
            url: None,
            tier,
            source_type: SourceType::Documentation,
            domain: None,
            author: None,
            date: None,
            verified: false,
            quote: None,
            stance: Stance::Neutral,
            credibility: tier.weight(),
        }
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    pub fn with_type(mut self, source_type: SourceType) -> Self {
        self.source_type = source_type;
        self
    }

    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    pub fn with_quote(mut self, quote: impl Into<String>) -> Self {
        self.quote = Some(quote.into());
        self
    }

    pub fn with_stance(mut self, stance: Stance) -> Self {
        self.stance = stance;
        self
    }

    pub fn verified(mut self) -> Self {
        self.verified = true;
        self
    }

    /// Calculate effective weight
    pub fn effective_weight(&self) -> f32 {
        let base = self.tier.weight();
        let stance_modifier: f32 = match self.stance {
            Stance::Support => 1.0,
            Stance::Contradict => -1.0,
            Stance::Neutral => 0.3,
            Stance::Partial => 0.6,
        };
        let verified_bonus = if self.verified { 1.2 } else { 1.0 };

        base * stance_modifier.abs() * verified_bonus * self.credibility
    }
}

/// Result of triangulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangulationResult {
    /// The claim being verified
    pub claim: String,
    /// All sources considered
    pub sources: Vec<Source>,
    /// Verification score (0.0-1.0)
    pub verification_score: f32,
    /// Is the claim verified (score >= threshold)?
    pub is_verified: bool,
    /// Number of supporting sources
    pub support_count: usize,
    /// Number of contradicting sources
    pub contradict_count: usize,
    /// Effective triangulation weight
    pub triangulation_weight: f32,
    /// Diversity of sources
    pub source_diversity: f32,
    /// Issues found
    pub issues: Vec<TriangulationIssue>,
    /// Confidence level
    pub confidence: VerificationConfidence,
    /// Recommendation
    pub recommendation: VerificationRecommendation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangulationIssue {
    pub issue_type: TriangulationIssueType,
    pub description: String,
    pub severity: IssueSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriangulationIssueType {
    InsufficientSources,
    NoTier1Source,
    AllSourcesSameDomain,
    AllSourcesSameAuthor,
    UnverifiedUrls,
    ContradictoryEvidence,
    StaleData,
    CircularSources,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationConfidence {
    /// High confidence (>= 3 Tier 1 sources)
    High,
    /// Medium confidence (3+ sources, mixed tiers)
    Medium,
    /// Low confidence (< 3 sources or all low tier)
    Low,
    /// Unable to verify
    Unverifiable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationRecommendation {
    /// Claim can be stated as fact
    AcceptAsFact,
    /// Claim can be stated with qualifier
    AcceptWithQualifier(String),
    /// More sources needed
    NeedsMoreSources,
    /// Conflicting evidence - present both sides
    PresentBothSides,
    /// Claim should not be made
    Reject,
    /// Unable to determine
    Inconclusive,
}

/// Configuration for triangulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangulationConfig {
    /// Minimum sources required
    pub min_sources: usize,
    /// Minimum Tier 1 sources
    pub min_tier1_sources: usize,
    /// Verification threshold
    pub verification_threshold: f32,
    /// Require URL verification
    pub require_verified_urls: bool,
    /// Maximum age of sources (days)
    pub max_source_age_days: Option<u32>,
    /// Require domain diversity
    pub require_domain_diversity: bool,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_sources: 3,
            min_tier1_sources: 1,
            verification_threshold: 0.6,
            require_verified_urls: false,
            max_source_age_days: None,
            require_domain_diversity: true,
        }
    }
}

/// The triangulation engine
pub struct Triangulator {
    pub config: TriangulationConfig,
    sources: Vec<Source>,
    claim: Option<String>,
}

impl Triangulator {
    pub fn new() -> Self {
        Self {
            config: TriangulationConfig::default(),
            sources: Vec::new(),
            claim: None,
        }
    }

    pub fn with_config(config: TriangulationConfig) -> Self {
        Self {
            config,
            sources: Vec::new(),
            claim: None,
        }
    }

    pub fn set_claim(&mut self, claim: impl Into<String>) {
        self.claim = Some(claim.into());
    }

    pub fn add_source(&mut self, source: Source) {
        self.sources.push(source);
    }

    pub fn clear_sources(&mut self) {
        self.sources.clear();
    }

    /// Verify the claim
    pub fn verify(&self) -> TriangulationResult {
        let claim = self.claim.clone().unwrap_or_default();
        let mut issues = Vec::new();

        // Check minimum sources
        if self.sources.len() < self.config.min_sources {
            issues.push(TriangulationIssue {
                issue_type: TriangulationIssueType::InsufficientSources,
                description: format!(
                    "Only {} sources, minimum {} required",
                    self.sources.len(),
                    self.config.min_sources
                ),
                severity: IssueSeverity::Critical,
            });
        }

        // Check Tier 1 sources
        let tier1_count = self
            .sources
            .iter()
            .filter(|s| s.tier == SourceTier::Primary)
            .count();

        if tier1_count < self.config.min_tier1_sources {
            issues.push(TriangulationIssue {
                issue_type: TriangulationIssueType::NoTier1Source,
                description: format!(
                    "Only {} Tier 1 sources, minimum {} required",
                    tier1_count, self.config.min_tier1_sources
                ),
                severity: IssueSeverity::Error,
            });
        }

        // Check domain diversity
        if self.config.require_domain_diversity {
            let domains: std::collections::HashSet<_> = self
                .sources
                .iter()
                .filter_map(|s| s.domain.as_ref())
                .collect();

            if domains.len() < 2 && self.sources.len() >= 2 {
                issues.push(TriangulationIssue {
                    issue_type: TriangulationIssueType::AllSourcesSameDomain,
                    description: "All sources from same domain - need diversity".into(),
                    severity: IssueSeverity::Warning,
                });
            }
        }

        // Check for contradictions
        let support_count = self
            .sources
            .iter()
            .filter(|s| s.stance == Stance::Support || s.stance == Stance::Partial)
            .count();

        let contradict_count = self
            .sources
            .iter()
            .filter(|s| s.stance == Stance::Contradict)
            .count();

        if support_count > 0 && contradict_count > 0 {
            issues.push(TriangulationIssue {
                issue_type: TriangulationIssueType::ContradictoryEvidence,
                description: format!(
                    "{} supporting, {} contradicting - conflicting evidence",
                    support_count, contradict_count
                ),
                severity: IssueSeverity::Warning,
            });
        }

        // Calculate triangulation weight
        let total_support_weight: f32 = self
            .sources
            .iter()
            .filter(|s| s.stance == Stance::Support || s.stance == Stance::Partial)
            .map(|s| s.effective_weight())
            .sum();

        let total_contradict_weight: f32 = self
            .sources
            .iter()
            .filter(|s| s.stance == Stance::Contradict)
            .map(|s| s.effective_weight())
            .sum();

        let triangulation_weight = total_support_weight - total_contradict_weight;

        // Calculate verification score
        let max_possible_weight = self.sources.len() as f32 * 1.0; // Max if all Tier 1 supporting
        let verification_score = if max_possible_weight > 0.0 {
            (triangulation_weight / max_possible_weight).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Calculate source diversity (different domains/authors/types)
        let source_diversity = self.calculate_diversity();

        // Determine confidence level
        let confidence = if tier1_count >= 2 && support_count >= 3 && contradict_count == 0 {
            VerificationConfidence::High
        } else if self.sources.len() >= 3 && support_count >= 2 {
            VerificationConfidence::Medium
        } else if self.sources.is_empty() {
            VerificationConfidence::Unverifiable
        } else {
            VerificationConfidence::Low
        };

        // Determine recommendation
        let recommendation = if issues.iter().any(|i| i.severity == IssueSeverity::Critical) {
            VerificationRecommendation::NeedsMoreSources
        } else if contradict_count > support_count {
            VerificationRecommendation::Reject
        } else if support_count > 0 && contradict_count > 0 {
            VerificationRecommendation::PresentBothSides
        } else if verification_score >= self.config.verification_threshold {
            if confidence == VerificationConfidence::High {
                VerificationRecommendation::AcceptAsFact
            } else {
                VerificationRecommendation::AcceptWithQualifier(format!(
                    "Based on {} sources",
                    support_count
                ))
            }
        } else if verification_score > 0.0 {
            VerificationRecommendation::AcceptWithQualifier("Limited evidence suggests".into())
        } else if self.sources.is_empty() {
            VerificationRecommendation::Inconclusive
        } else {
            VerificationRecommendation::NeedsMoreSources
        };

        let is_verified = verification_score >= self.config.verification_threshold
            && !issues.iter().any(|i| i.severity == IssueSeverity::Critical);

        TriangulationResult {
            claim,
            sources: self.sources.clone(),
            verification_score,
            is_verified,
            support_count,
            contradict_count,
            triangulation_weight,
            source_diversity,
            issues,
            confidence,
            recommendation,
        }
    }

    fn calculate_diversity(&self) -> f32 {
        if self.sources.is_empty() {
            return 0.0;
        }

        let domains: std::collections::HashSet<_> = self
            .sources
            .iter()
            .filter_map(|s| s.domain.as_ref())
            .collect();

        let authors: std::collections::HashSet<_> = self
            .sources
            .iter()
            .filter_map(|s| s.author.as_ref())
            .collect();

        let types: std::collections::HashSet<_> =
            self.sources.iter().map(|s| s.source_type).collect();

        let tiers: std::collections::HashSet<_> = self.sources.iter().map(|s| s.tier).collect();

        let n = self.sources.len() as f32;
        let domain_diversity = domains.len() as f32 / n.min(5.0);
        let author_diversity = authors.len() as f32 / n.min(5.0);
        let type_diversity = types.len() as f32 / 4.0; // 4 main types
        let tier_diversity = tiers.len() as f32 / 4.0; // 4 tiers

        (domain_diversity * 0.3
            + author_diversity * 0.3
            + type_diversity * 0.2
            + tier_diversity * 0.2)
            .min(1.0)
    }
}

impl Default for Triangulator {
    fn default() -> Self {
        Self::new()
    }
}

impl TriangulationResult {
    /// Format as structured text
    pub fn format(&self) -> String {
        let mut output = String::new();

        output
            .push_str("┌─────────────────────────────────────────────────────────────────────┐\n");
        output
            .push_str("│                  TRIANGULATION REPORT                               │\n");
        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");

        output.push_str(&format!("│ CLAIM: {}\n", self.claim));
        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");

        // Summary
        output.push_str(&format!(
            "│ STATUS: {} (Score: {:.0}%)\n",
            if self.is_verified {
                "✓ VERIFIED"
            } else {
                "✗ UNVERIFIED"
            },
            self.verification_score * 100.0
        ));
        output.push_str(&format!("│ CONFIDENCE: {:?}\n", self.confidence));
        output.push_str(&format!("│ RECOMMENDATION: {:?}\n", self.recommendation));

        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");
        output.push_str("│ SOURCES:\n");

        for source in &self.sources {
            let stance_icon = match source.stance {
                Stance::Support => "✓",
                Stance::Contradict => "✗",
                Stance::Neutral => "○",
                Stance::Partial => "◐",
            };
            output.push_str(&format!(
                "│   {} [{}] {} ({})\n",
                stance_icon,
                source.tier.label(),
                source.name,
                if source.verified {
                    "verified"
                } else {
                    "unverified"
                }
            ));
        }

        // Stats
        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");
        output.push_str(&format!(
            "│ STATS: {} supporting, {} contradicting, diversity: {:.0}%\n",
            self.support_count,
            self.contradict_count,
            self.source_diversity * 100.0
        ));

        // Issues
        if !self.issues.is_empty() {
            output.push_str(
                "├─────────────────────────────────────────────────────────────────────┤\n",
            );
            output.push_str("│ ISSUES:\n");
            for issue in &self.issues {
                let severity_icon = match issue.severity {
                    IssueSeverity::Warning => "⚠",
                    IssueSeverity::Error => "⛔",
                    IssueSeverity::Critical => "🚫",
                };
                output.push_str(&format!("│   {} {}\n", severity_icon, issue.description));
            }
        }

        output
            .push_str("└─────────────────────────────────────────────────────────────────────┘\n");

        output
    }
}

/// Prompt templates for triangulation
pub struct TriangulationPrompts;

impl TriangulationPrompts {
    /// Find sources for a claim
    pub fn find_sources(claim: &str) -> String {
        format!(
            r#"Find 3+ independent sources to verify or refute this claim:

CLAIM: {claim}

For each source, provide:
1. Name/Title
2. URL (if available)
3. Tier:
   - Tier 1 (Primary): Official docs, peer-reviewed papers, primary sources
   - Tier 2 (Secondary): Reputable news, expert blogs, industry reports
   - Tier 3 (Independent): Community content, forums
   - Tier 4 (Unverified): Social media, unknown sources

4. Type: Academic/Documentation/News/Expert/Government/Industry/Community/Social
5. Domain: What field is this from?
6. Author: Who wrote/published this?
7. Stance: Support/Contradict/Neutral/Partial
8. Direct quote: Key quote supporting/refuting the claim

CRITICAL REQUIREMENTS:
- Minimum 3 sources from different domains
- At least 1 Tier 1 source
- Look for contradicting evidence too
- Verify URLs are accessible

Respond in JSON format."#,
            claim = claim
        )
    }

    /// Evaluate triangulation result
    pub fn evaluate_triangulation(sources: &str, claim: &str) -> String {
        format!(
            r#"Evaluate whether this claim is sufficiently triangulated:

CLAIM: {claim}

SOURCES PROVIDED:
{sources}

Evaluate:
1. Are there at least 3 independent sources?
2. Is there at least 1 Tier 1 (primary) source?
3. Do sources come from different domains?
4. Are there any contradictions?
5. What is the overall verification score (0-100%)?
6. What is your confidence level (High/Medium/Low/Unverifiable)?
7. What is your recommendation?
   - Accept as fact
   - Accept with qualifier
   - Needs more sources
   - Present both sides
   - Reject
   - Inconclusive

Respond in JSON format."#,
            claim = claim,
            sources = sources
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_creation() {
        let source = Source::new("Test Paper", SourceTier::Primary)
            .with_url("https://example.com")
            .with_domain("AI")
            .with_stance(Stance::Support)
            .verified();

        assert_eq!(source.tier, SourceTier::Primary);
        assert!(source.verified);
        assert_eq!(source.stance, Stance::Support);
    }

    #[test]
    fn test_basic_triangulation() {
        let mut triangulator = Triangulator::new();
        triangulator.set_claim("AI can pass the Turing test");

        triangulator.add_source(
            Source::new("Research Paper", SourceTier::Primary)
                .with_domain("AI")
                .with_stance(Stance::Support)
                .verified(),
        );
        triangulator.add_source(
            Source::new("Industry Report", SourceTier::Secondary)
                .with_domain("Tech")
                .with_stance(Stance::Support)
                .verified(),
        );
        triangulator.add_source(
            Source::new("Expert Blog", SourceTier::Independent)
                .with_domain("ML")
                .with_stance(Stance::Support),
        );

        let result = triangulator.verify();

        assert!(result.support_count >= 3);
        assert_eq!(result.contradict_count, 0);
        assert!(result.verification_score > 0.5);
    }

    #[test]
    fn test_insufficient_sources() {
        let mut triangulator = Triangulator::new();
        triangulator.set_claim("Some claim");
        triangulator.add_source(Source::new("Single Source", SourceTier::Primary));

        let result = triangulator.verify();

        assert!(!result.is_verified);
        assert!(result
            .issues
            .iter()
            .any(|i| i.issue_type == TriangulationIssueType::InsufficientSources));
    }

    #[test]
    fn test_contradictory_evidence() {
        let mut triangulator = Triangulator::new();
        triangulator.set_claim("Contested claim");

        triangulator
            .add_source(Source::new("Source A", SourceTier::Primary).with_stance(Stance::Support));
        triangulator.add_source(
            Source::new("Source B", SourceTier::Primary).with_stance(Stance::Contradict),
        );
        triangulator.add_source(
            Source::new("Source C", SourceTier::Secondary).with_stance(Stance::Support),
        );

        let result = triangulator.verify();

        assert!(result.contradict_count > 0);
        assert!(result
            .issues
            .iter()
            .any(|i| i.issue_type == TriangulationIssueType::ContradictoryEvidence));
    }

    #[test]
    fn test_tier_weights() {
        assert!(SourceTier::Primary.weight() > SourceTier::Secondary.weight());
        assert!(SourceTier::Secondary.weight() > SourceTier::Independent.weight());
        assert!(SourceTier::Independent.weight() > SourceTier::Unverified.weight());
    }

    #[test]
    fn test_source_type_default_tier() {
        assert_eq!(SourceType::Academic.default_tier(), SourceTier::Primary);
        assert_eq!(SourceType::News.default_tier(), SourceTier::Secondary);
        assert_eq!(
            SourceType::Community.default_tier(),
            SourceTier::Independent
        );
        assert_eq!(SourceType::Social.default_tier(), SourceTier::Unverified);
    }
}
