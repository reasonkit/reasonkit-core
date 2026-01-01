//! ProofGuard Module - Multi-Source Verification
//!
//! Triangulates claims across 3+ independent sources to verify factual accuracy.
//!
//! ## Core Features
//!
//! - **3+ Source Requirement**: Enforces triangulation protocol (CONS-006)
//! - **Contradiction Detection**: Identifies conflicting evidence
//! - **Source Tier Ranking**: Weights evidence by source quality
//! - **Confidence Scoring**: Produces calibrated verification scores
//!
//! ## Source Tiers
//!
//! | Tier | Weight | Examples |
//! |------|--------|----------|
//! | Tier 1 (Primary) | 1.0 | Official docs, peer-reviewed papers, primary sources |
//! | Tier 2 (Secondary) | 0.7 | Reputable news, expert blogs, industry reports |
//! | Tier 3 (Independent) | 0.4 | Community content, forums |
//! | Tier 4 (Unverified) | 0.2 | Social media, unknown sources |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::modules::{ProofGuard, ThinkToolContext, ThinkToolModule};
//!
//! let proofguard = ProofGuard::new();
//!
//! // Context with claim and sources (JSON format)
//! let context = ThinkToolContext {
//!     query: r#"{
//!         "claim": "Rust is memory-safe without a garbage collector",
//!         "sources": [
//!             {"name": "Rust Book", "tier": "Primary", "stance": "Support"},
//!             {"name": "ACM Paper", "tier": "Primary", "stance": "Support"},
//!             {"name": "Tech Blog", "tier": "Secondary", "stance": "Support"}
//!         ]
//!     }"#.to_string(),
//!     previous_steps: vec![],
//! };
//!
//! let result = proofguard.execute(&context)?;
//! ```

use super::{ThinkToolContext, ThinkToolModule, ThinkToolModuleConfig, ThinkToolOutput};
use crate::error::Error;
use crate::thinktool::triangulation::{
    IssueSeverity, Source, SourceTier, SourceType, Stance, TriangulationConfig,
    TriangulationIssueType, TriangulationResult, Triangulator, VerificationConfidence,
    VerificationRecommendation,
};
use serde::{Deserialize, Serialize};

/// ProofGuard reasoning module for fact verification.
///
/// Verifies claims using triangulated evidence from multiple sources.
/// Implements the three-source rule (CONS-006) and provides structured
/// verification output with confidence scoring.
pub struct ProofGuard {
    /// Module configuration
    config: ThinkToolModuleConfig,
    /// Triangulation configuration
    triangulation_config: TriangulationConfig,
}

impl Default for ProofGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Input format for ProofGuard verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofGuardInput {
    /// The claim to verify
    pub claim: String,
    /// Sources supporting or contradicting the claim
    #[serde(default)]
    pub sources: Vec<ProofGuardSource>,
    /// Optional: Minimum number of sources required (default: 3)
    #[serde(default)]
    pub min_sources: Option<usize>,
    /// Optional: Require at least one Tier 1 source (default: true)
    #[serde(default)]
    pub require_tier1: Option<bool>,
}

/// Source input for ProofGuard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofGuardSource {
    /// Source name/title
    pub name: String,
    /// Source tier (Primary, Secondary, Independent, Unverified)
    #[serde(default = "default_tier")]
    pub tier: String,
    /// Source type (Academic, Documentation, News, Expert, Government, Industry, Community, Social, PrimaryData)
    #[serde(default = "default_source_type")]
    pub source_type: String,
    /// Source stance (Support, Contradict, Neutral, Partial)
    #[serde(default = "default_stance")]
    pub stance: String,
    /// URL if available
    #[serde(default)]
    pub url: Option<String>,
    /// Domain/field
    #[serde(default)]
    pub domain: Option<String>,
    /// Author/organization
    #[serde(default)]
    pub author: Option<String>,
    /// Direct quote supporting/refuting the claim
    #[serde(default)]
    pub quote: Option<String>,
    /// Whether URL has been verified accessible
    #[serde(default)]
    pub verified: bool,
}

fn default_tier() -> String {
    "Unverified".to_string()
}

fn default_source_type() -> String {
    "Documentation".to_string()
}

fn default_stance() -> String {
    "Neutral".to_string()
}

/// Output from ProofGuard verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofGuardOutput {
    /// Overall verification verdict
    pub verdict: ProofGuardVerdict,
    /// Claim being verified
    pub claim: String,
    /// Verification score (0.0-1.0)
    pub verification_score: f64,
    /// Whether the claim is verified
    pub is_verified: bool,
    /// Confidence level
    pub confidence_level: String,
    /// Recommendation for how to treat this claim
    pub recommendation: String,
    /// Sources analyzed
    pub sources: Vec<SourceSummary>,
    /// Detected contradictions
    pub contradictions: Vec<ContradictionInfo>,
    /// Issues found during verification
    pub issues: Vec<IssueInfo>,
    /// Statistics
    pub stats: VerificationStats,
}

/// Verification verdict
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofGuardVerdict {
    /// Claim verified with high confidence
    Verified,
    /// Claim partially verified, needs qualifier
    PartiallyVerified,
    /// Conflicting evidence found
    Contested,
    /// Insufficient sources
    InsufficientSources,
    /// Claim contradicted by evidence
    Refuted,
    /// Unable to determine
    Inconclusive,
}

impl std::fmt::Display for ProofGuardVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofGuardVerdict::Verified => write!(f, "Verified"),
            ProofGuardVerdict::PartiallyVerified => write!(f, "Partially Verified"),
            ProofGuardVerdict::Contested => write!(f, "Contested"),
            ProofGuardVerdict::InsufficientSources => write!(f, "Insufficient Sources"),
            ProofGuardVerdict::Refuted => write!(f, "Refuted"),
            ProofGuardVerdict::Inconclusive => write!(f, "Inconclusive"),
        }
    }
}

/// Summary of a source used in verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSummary {
    /// Source name
    pub name: String,
    /// Source tier label
    pub tier: String,
    /// Tier weight (0.0-1.0)
    pub weight: f64,
    /// Stance on the claim
    pub stance: String,
    /// Whether verified
    pub verified: bool,
    /// Effective weight after modifiers
    pub effective_weight: f64,
}

/// Information about detected contradictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionInfo {
    /// Sources that support the claim
    pub supporting_sources: Vec<String>,
    /// Sources that contradict the claim
    pub contradicting_sources: Vec<String>,
    /// Severity of the contradiction
    pub severity: String,
    /// Description
    pub description: String,
}

/// Information about verification issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueInfo {
    /// Issue type
    pub issue_type: String,
    /// Severity (Warning, Error, Critical)
    pub severity: String,
    /// Description
    pub description: String,
}

/// Statistics from verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStats {
    /// Total sources analyzed
    pub total_sources: usize,
    /// Number of supporting sources
    pub supporting_count: usize,
    /// Number of contradicting sources
    pub contradicting_count: usize,
    /// Number of neutral sources
    pub neutral_count: usize,
    /// Number of Tier 1 sources
    pub tier1_count: usize,
    /// Number of Tier 2 sources
    pub tier2_count: usize,
    /// Number of Tier 3 sources
    pub tier3_count: usize,
    /// Number of Tier 4 sources
    pub tier4_count: usize,
    /// Source diversity score (0.0-1.0)
    pub source_diversity: f64,
    /// Triangulation weight (support - contradict)
    pub triangulation_weight: f64,
}

impl ProofGuard {
    /// Create a new ProofGuard module instance.
    pub fn new() -> Self {
        Self {
            config: ThinkToolModuleConfig {
                name: "ProofGuard".to_string(),
                version: "2.1.0".to_string(),
                description: "Triangulation-based fact verification with 3+ source requirement"
                    .to_string(),
                confidence_weight: 0.30,
            },
            triangulation_config: TriangulationConfig::default(),
        }
    }

    /// Create with custom triangulation configuration.
    pub fn with_config(triangulation_config: TriangulationConfig) -> Self {
        Self {
            config: ThinkToolModuleConfig {
                name: "ProofGuard".to_string(),
                version: "2.1.0".to_string(),
                description: "Triangulation-based fact verification with 3+ source requirement"
                    .to_string(),
                confidence_weight: 0.30,
            },
            triangulation_config,
        }
    }

    /// Create with strict configuration (requires 2 Tier 1 sources).
    pub fn strict() -> Self {
        let config = TriangulationConfig {
            min_sources: 3,
            min_tier1_sources: 2,
            verification_threshold: 0.7,
            require_verified_urls: true,
            require_domain_diversity: true,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create with relaxed configuration (1 Tier 1 source sufficient).
    pub fn relaxed() -> Self {
        let config = TriangulationConfig {
            min_sources: 2,
            min_tier1_sources: 0,
            verification_threshold: 0.5,
            require_verified_urls: false,
            require_domain_diversity: false,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Parse the input from context.
    fn parse_input(&self, context: &ThinkToolContext) -> Result<ProofGuardInput, Error> {
        // First try to parse as JSON
        if let Ok(input) = serde_json::from_str::<ProofGuardInput>(&context.query) {
            return Ok(input);
        }

        // If not JSON, treat as plain text claim with no sources
        Ok(ProofGuardInput {
            claim: context.query.clone(),
            sources: Vec::new(),
            min_sources: None,
            require_tier1: None,
        })
    }

    /// Convert ProofGuardSource to triangulation Source.
    fn convert_source(&self, src: &ProofGuardSource) -> Source {
        let tier = match src.tier.to_lowercase().as_str() {
            "primary" | "tier1" | "tier 1" => SourceTier::Primary,
            "secondary" | "tier2" | "tier 2" => SourceTier::Secondary,
            "independent" | "tier3" | "tier 3" => SourceTier::Independent,
            _ => SourceTier::Unverified,
        };

        let source_type = match src.source_type.to_lowercase().as_str() {
            "academic" => SourceType::Academic,
            "documentation" | "docs" => SourceType::Documentation,
            "news" => SourceType::News,
            "expert" | "blog" => SourceType::Expert,
            "government" | "gov" => SourceType::Government,
            "industry" => SourceType::Industry,
            "community" | "forum" => SourceType::Community,
            "social" | "socialmedia" => SourceType::Social,
            "primarydata" | "data" => SourceType::PrimaryData,
            _ => SourceType::Documentation,
        };

        let stance = match src.stance.to_lowercase().as_str() {
            "support" | "supports" | "supporting" => Stance::Support,
            "contradict" | "contradicts" | "contradicting" | "oppose" | "against" => {
                Stance::Contradict
            }
            "partial" | "partially" => Stance::Partial,
            _ => Stance::Neutral,
        };

        let mut source = Source::new(&src.name, tier)
            .with_type(source_type)
            .with_stance(stance);

        if let Some(url) = &src.url {
            source = source.with_url(url);
        }
        if let Some(domain) = &src.domain {
            source = source.with_domain(domain);
        }
        if let Some(author) = &src.author {
            source = source.with_author(author);
        }
        if let Some(quote) = &src.quote {
            source = source.with_quote(quote);
        }
        if src.verified {
            source = source.verified();
        }

        source
    }

    /// Convert triangulation result to ProofGuard output.
    fn convert_result(&self, result: &TriangulationResult) -> ProofGuardOutput {
        // Determine verdict
        let verdict = self.determine_verdict(result);

        // Build source summaries
        let sources: Vec<SourceSummary> = result
            .sources
            .iter()
            .map(|s| SourceSummary {
                name: s.name.clone(),
                tier: s.tier.label().to_string(),
                weight: s.tier.weight() as f64,
                stance: format!("{:?}", s.stance),
                verified: s.verified,
                effective_weight: s.effective_weight() as f64,
            })
            .collect();

        // Build contradiction info if present
        let contradictions = if result.contradict_count > 0 && result.support_count > 0 {
            let supporting: Vec<String> = result
                .sources
                .iter()
                .filter(|s| matches!(s.stance, Stance::Support | Stance::Partial))
                .map(|s| s.name.clone())
                .collect();
            let contradicting: Vec<String> = result
                .sources
                .iter()
                .filter(|s| matches!(s.stance, Stance::Contradict))
                .map(|s| s.name.clone())
                .collect();

            vec![ContradictionInfo {
                supporting_sources: supporting,
                contradicting_sources: contradicting,
                severity: if result.contradict_count >= result.support_count {
                    "High".to_string()
                } else {
                    "Medium".to_string()
                },
                description: format!(
                    "{} sources support while {} sources contradict the claim",
                    result.support_count, result.contradict_count
                ),
            }]
        } else {
            vec![]
        };

        // Build issue info
        let issues: Vec<IssueInfo> = result
            .issues
            .iter()
            .map(|i| IssueInfo {
                issue_type: format!("{:?}", i.issue_type),
                severity: match i.severity {
                    IssueSeverity::Warning => "Warning".to_string(),
                    IssueSeverity::Error => "Error".to_string(),
                    IssueSeverity::Critical => "Critical".to_string(),
                },
                description: i.description.clone(),
            })
            .collect();

        // Calculate tier counts
        let tier1_count = result
            .sources
            .iter()
            .filter(|s| s.tier == SourceTier::Primary)
            .count();
        let tier2_count = result
            .sources
            .iter()
            .filter(|s| s.tier == SourceTier::Secondary)
            .count();
        let tier3_count = result
            .sources
            .iter()
            .filter(|s| s.tier == SourceTier::Independent)
            .count();
        let tier4_count = result
            .sources
            .iter()
            .filter(|s| s.tier == SourceTier::Unverified)
            .count();
        let neutral_count = result
            .sources
            .iter()
            .filter(|s| matches!(s.stance, Stance::Neutral))
            .count();

        // Build stats
        let stats = VerificationStats {
            total_sources: result.sources.len(),
            supporting_count: result.support_count,
            contradicting_count: result.contradict_count,
            neutral_count,
            tier1_count,
            tier2_count,
            tier3_count,
            tier4_count,
            source_diversity: result.source_diversity as f64,
            triangulation_weight: result.triangulation_weight as f64,
        };

        // Format recommendation
        let recommendation = match &result.recommendation {
            VerificationRecommendation::AcceptAsFact => "Accept as fact".to_string(),
            VerificationRecommendation::AcceptWithQualifier(q) => {
                format!("Accept with qualifier: {}", q)
            }
            VerificationRecommendation::NeedsMoreSources => {
                "Need more sources before making this claim".to_string()
            }
            VerificationRecommendation::PresentBothSides => {
                "Present both supporting and contradicting evidence".to_string()
            }
            VerificationRecommendation::Reject => {
                "Evidence does not support this claim".to_string()
            }
            VerificationRecommendation::Inconclusive => {
                "Unable to determine - more research needed".to_string()
            }
        };

        ProofGuardOutput {
            verdict,
            claim: result.claim.clone(),
            verification_score: result.verification_score as f64,
            is_verified: result.is_verified,
            confidence_level: format!("{:?}", result.confidence),
            recommendation,
            sources,
            contradictions,
            issues,
            stats,
        }
    }

    /// Determine the verdict based on triangulation result.
    fn determine_verdict(&self, result: &TriangulationResult) -> ProofGuardVerdict {
        // Check for critical issues first
        if result
            .issues
            .iter()
            .any(|i| i.issue_type == TriangulationIssueType::InsufficientSources)
        {
            return ProofGuardVerdict::InsufficientSources;
        }

        // Check for contradictions
        if result.contradict_count > 0 && result.support_count > 0 {
            if result.contradict_count > result.support_count {
                return ProofGuardVerdict::Refuted;
            }
            return ProofGuardVerdict::Contested;
        }

        // Check if refuted (only contradictions, no support)
        if result.contradict_count > 0 && result.support_count == 0 {
            return ProofGuardVerdict::Refuted;
        }

        // Check verification status
        match result.confidence {
            VerificationConfidence::High => {
                if result.is_verified {
                    ProofGuardVerdict::Verified
                } else {
                    ProofGuardVerdict::PartiallyVerified
                }
            }
            VerificationConfidence::Medium => {
                if result.is_verified {
                    ProofGuardVerdict::PartiallyVerified
                } else {
                    ProofGuardVerdict::Inconclusive
                }
            }
            VerificationConfidence::Low => ProofGuardVerdict::Inconclusive,
            VerificationConfidence::Unverifiable => ProofGuardVerdict::Inconclusive,
        }
    }

    /// Calculate overall confidence for ThinkToolOutput.
    fn calculate_confidence(&self, result: &TriangulationResult) -> f64 {
        // Base confidence from verification score
        let base_confidence = result.verification_score as f64;

        // Adjust based on confidence level
        let level_modifier = match result.confidence {
            VerificationConfidence::High => 1.0,
            VerificationConfidence::Medium => 0.85,
            VerificationConfidence::Low => 0.6,
            VerificationConfidence::Unverifiable => 0.3,
        };

        // Penalize for issues
        let issue_penalty: f64 = result
            .issues
            .iter()
            .map(|i| match i.severity {
                IssueSeverity::Critical => 0.3,
                IssueSeverity::Error => 0.15,
                IssueSeverity::Warning => 0.05,
            })
            .sum();

        // Boost for source diversity
        let diversity_boost = result.source_diversity as f64 * 0.1;

        // Calculate final confidence - return directly
        (base_confidence * level_modifier - issue_penalty + diversity_boost).clamp(0.0, 1.0)
    }
}

impl ThinkToolModule for ProofGuard {
    fn config(&self) -> &ThinkToolModuleConfig {
        &self.config
    }

    fn execute(&self, context: &ThinkToolContext) -> Result<ThinkToolOutput, Error> {
        // Parse input
        let input = self.parse_input(context)?;

        // Apply any input-level configuration overrides
        let mut config = self.triangulation_config.clone();
        if let Some(min) = input.min_sources {
            config.min_sources = min;
        }
        if let Some(req) = input.require_tier1 {
            config.min_tier1_sources = if req { 1 } else { 0 };
        }

        // Create triangulator
        let mut triangulator = Triangulator::with_config(config);
        triangulator.set_claim(&input.claim);

        // Convert and add sources
        for src in &input.sources {
            let source = self.convert_source(src);
            triangulator.add_source(source);
        }

        // Execute verification
        let result = triangulator.verify();

        // Convert to ProofGuard output
        let output = self.convert_result(&result);

        // Calculate confidence
        let confidence = self.calculate_confidence(&result);

        // Return structured output
        Ok(ThinkToolOutput {
            module: self.config.name.clone(),
            confidence,
            output: serde_json::to_value(&output).map_err(|e| {
                Error::ThinkToolExecutionError(format!("Failed to serialize output: {}", e))
            })?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proofguard_new() {
        let pg = ProofGuard::new();
        assert_eq!(pg.config.name, "ProofGuard");
        assert_eq!(pg.config.version, "2.1.0");
    }

    #[test]
    fn test_proofguard_default() {
        let pg = ProofGuard::default();
        assert_eq!(pg.config.name, "ProofGuard");
    }

    #[test]
    fn test_proofguard_strict() {
        let pg = ProofGuard::strict();
        assert_eq!(pg.triangulation_config.min_tier1_sources, 2);
        assert!(pg.triangulation_config.require_verified_urls);
    }

    #[test]
    fn test_proofguard_relaxed() {
        let pg = ProofGuard::relaxed();
        assert_eq!(pg.triangulation_config.min_sources, 2);
        assert_eq!(pg.triangulation_config.min_tier1_sources, 0);
    }

    #[test]
    fn test_execute_with_json_input() {
        let pg = ProofGuard::new();
        let input_json = r#"{
            "claim": "Rust is memory-safe without garbage collection",
            "sources": [
                {"name": "Rust Book", "tier": "Primary", "stance": "Support", "verified": true},
                {"name": "ACM Paper", "tier": "Primary", "stance": "Support", "domain": "PL"},
                {"name": "Tech Blog", "tier": "Secondary", "stance": "Support"}
            ]
        }"#;

        let context = ThinkToolContext {
            query: input_json.to_string(),
            previous_steps: vec![],
        };

        let result = pg.execute(&context).unwrap();

        assert_eq!(result.module, "ProofGuard");
        assert!(result.confidence > 0.0);

        // Parse the output
        let output: ProofGuardOutput = serde_json::from_value(result.output).unwrap();
        assert_eq!(
            output.claim,
            "Rust is memory-safe without garbage collection"
        );
        assert_eq!(output.stats.total_sources, 3);
        assert_eq!(output.stats.tier1_count, 2);
        assert!(output.is_verified);
    }

    #[test]
    fn test_execute_with_plain_text_claim() {
        let pg = ProofGuard::new();
        let context = ThinkToolContext {
            query: "This is a plain text claim".to_string(),
            previous_steps: vec![],
        };

        let result = pg.execute(&context).unwrap();

        // Should return insufficient sources
        let output: ProofGuardOutput = serde_json::from_value(result.output).unwrap();
        assert_eq!(output.verdict, ProofGuardVerdict::InsufficientSources);
        assert!(!output.is_verified);
    }

    #[test]
    fn test_execute_with_contradictions() {
        let pg = ProofGuard::new();
        let input_json = r#"{
            "claim": "AI will achieve AGI by 2030",
            "sources": [
                {"name": "Optimist Paper", "tier": "Primary", "stance": "Support"},
                {"name": "Skeptic Paper", "tier": "Primary", "stance": "Contradict"},
                {"name": "Neutral Review", "tier": "Secondary", "stance": "Partial"}
            ]
        }"#;

        let context = ThinkToolContext {
            query: input_json.to_string(),
            previous_steps: vec![],
        };

        let result = pg.execute(&context).unwrap();
        let output: ProofGuardOutput = serde_json::from_value(result.output).unwrap();

        assert_eq!(output.verdict, ProofGuardVerdict::Contested);
        assert!(!output.contradictions.is_empty());
        assert!(output.stats.contradicting_count > 0);
    }

    #[test]
    fn test_execute_refuted_claim() {
        let pg = ProofGuard::new();
        let input_json = r#"{
            "claim": "The Earth is flat",
            "sources": [
                {"name": "NASA", "tier": "Primary", "stance": "Contradict"},
                {"name": "ESA", "tier": "Primary", "stance": "Contradict"},
                {"name": "Physics Journal", "tier": "Primary", "stance": "Contradict"}
            ]
        }"#;

        let context = ThinkToolContext {
            query: input_json.to_string(),
            previous_steps: vec![],
        };

        let result = pg.execute(&context).unwrap();
        let output: ProofGuardOutput = serde_json::from_value(result.output).unwrap();

        assert_eq!(output.verdict, ProofGuardVerdict::Refuted);
        assert!(!output.is_verified);
    }

    #[test]
    fn test_source_tier_parsing() {
        let pg = ProofGuard::new();

        // Test various tier formats
        let test_cases = vec![
            ("Primary", SourceTier::Primary),
            ("primary", SourceTier::Primary),
            ("Tier1", SourceTier::Primary),
            ("tier 1", SourceTier::Primary),
            ("Secondary", SourceTier::Secondary),
            ("tier2", SourceTier::Secondary),
            ("Independent", SourceTier::Independent),
            ("tier 3", SourceTier::Independent),
            ("Unverified", SourceTier::Unverified),
            ("unknown", SourceTier::Unverified),
        ];

        for (input, expected) in test_cases {
            let src = ProofGuardSource {
                name: "Test".to_string(),
                tier: input.to_string(),
                source_type: default_source_type(),
                stance: default_stance(),
                url: None,
                domain: None,
                author: None,
                quote: None,
                verified: false,
            };
            let converted = pg.convert_source(&src);
            assert_eq!(converted.tier, expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_stance_parsing() {
        let pg = ProofGuard::new();

        let test_cases = vec![
            ("Support", Stance::Support),
            ("supports", Stance::Support),
            ("Contradict", Stance::Contradict),
            ("against", Stance::Contradict),
            ("Partial", Stance::Partial),
            ("Neutral", Stance::Neutral),
            ("unknown", Stance::Neutral),
        ];

        for (input, expected) in test_cases {
            let src = ProofGuardSource {
                name: "Test".to_string(),
                tier: default_tier(),
                source_type: default_source_type(),
                stance: input.to_string(),
                url: None,
                domain: None,
                author: None,
                quote: None,
                verified: false,
            };
            let converted = pg.convert_source(&src);
            assert_eq!(converted.stance, expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", ProofGuardVerdict::Verified), "Verified");
        assert_eq!(
            format!("{}", ProofGuardVerdict::PartiallyVerified),
            "Partially Verified"
        );
        assert_eq!(format!("{}", ProofGuardVerdict::Contested), "Contested");
        assert_eq!(
            format!("{}", ProofGuardVerdict::InsufficientSources),
            "Insufficient Sources"
        );
        assert_eq!(format!("{}", ProofGuardVerdict::Refuted), "Refuted");
        assert_eq!(
            format!("{}", ProofGuardVerdict::Inconclusive),
            "Inconclusive"
        );
    }

    #[test]
    fn test_min_sources_override() {
        let pg = ProofGuard::new();
        let input_json = r#"{
            "claim": "Some claim",
            "sources": [
                {"name": "Source A", "tier": "Primary", "stance": "Support"}
            ],
            "min_sources": 1
        }"#;

        let context = ThinkToolContext {
            query: input_json.to_string(),
            previous_steps: vec![],
        };

        let result = pg.execute(&context).unwrap();
        let output: ProofGuardOutput = serde_json::from_value(result.output).unwrap();

        // With min_sources=1, should not be InsufficientSources
        assert_ne!(output.verdict, ProofGuardVerdict::InsufficientSources);
    }
}
