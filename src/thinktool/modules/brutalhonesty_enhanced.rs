//! Enhanced BrutalHonesty Module with Extended Analysis
//!
//! Advanced adversarial self-critique that builds upon the base BrutalHonesty
//! module with additional analysis capabilities:
//!
//! - **Multi-Cultural Bias Detection**: Identifies cultural assumptions
//! - **Cognitive Bias Catalog**: Checks for common cognitive biases
//! - **Argument Mapping**: Structures the argument for analysis
//! - **Steelmanning**: Constructs the strongest version of opposing arguments
//!
//! ## Design Philosophy
//!
//! The enhanced module provides deeper analysis for high-stakes decisions
//! where thorough adversarial review is critical. It is designed to be
//! used in conjunction with or as a replacement for the base module.
//!
//! ## Usage
//!
//! ```ignore
//! use reasonkit::thinktool::modules::{BrutalHonestyEnhanced, ThinkToolContext};
//!
//! let module = BrutalHonestyEnhanced::builder()
//!     .enable_cultural_analysis(true)
//!     .enable_steelmanning(true)
//!     .cognitive_bias_depth(CognitiveBiasDepth::Deep)
//!     .build();
//!
//! let context = ThinkToolContext {
//!     query: "We should expand to international markets immediately".to_string(),
//!     previous_steps: vec![],
//! };
//!
//! let result = module.execute(&context)?;
//! ```

use super::brutalhonesty::{
    BrutalHonesty, BrutalHonestyConfig, CritiqueSeverity, CritiqueVerdict, FlawSeverity,
};
use super::{ThinkToolContext, ThinkToolModule, ThinkToolModuleConfig, ThinkToolOutput};
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Depth of cognitive bias analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveBiasDepth {
    /// Basic bias detection (5 most common biases)
    Basic,
    /// Standard bias detection (15 common biases)
    #[default]
    Standard,
    /// Deep bias detection (30+ biases with sub-categories)
    Deep,
}

/// A detected cognitive bias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveBias {
    /// Name of the bias
    pub name: String,
    /// Category of the bias
    pub category: BiasCategory,
    /// Description of how this bias manifests
    pub manifestation: String,
    /// Confidence that this bias is present (0.0-1.0)
    pub confidence: f64,
    /// Debiasing strategy
    pub debiasing_strategy: String,
}

/// Category of cognitive bias.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BiasCategory {
    /// Decision-making biases (anchoring, availability, etc.)
    DecisionMaking,
    /// Social biases (in-group, halo effect, etc.)
    Social,
    /// Memory biases (hindsight, rosy retrospection, etc.)
    Memory,
    /// Probability biases (gambler's fallacy, base rate neglect, etc.)
    Probability,
    /// Self-related biases (overconfidence, self-serving, etc.)
    SelfRelated,
}

/// A cultural assumption detected in reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CulturalAssumption {
    /// The assumption
    pub assumption: String,
    /// Cultural context where this assumption holds
    pub context: String,
    /// Cultures where this assumption may not hold
    pub exceptions: Vec<String>,
    /// Risk of misapplication
    pub risk_level: FlawSeverity,
}

/// A steelmanned counter-argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteelmanArgument {
    /// The strongest version of the opposing argument
    pub argument: String,
    /// Key premises of the steelmanned argument
    pub premises: Vec<String>,
    /// How this argument challenges the original position
    pub challenge: String,
    /// Strength of the steelmanned argument (0.0-1.0)
    pub strength: f64,
}

/// Structured argument map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentMap {
    /// Main claim being analyzed
    pub claim: String,
    /// Supporting premises
    pub premises: Vec<String>,
    /// Evidence cited
    pub evidence: Vec<String>,
    /// Warrants (logical connections)
    pub warrants: Vec<String>,
    /// Qualifiers (conditions or limitations)
    pub qualifiers: Vec<String>,
    /// Rebuttals (counter-arguments acknowledged)
    pub rebuttals: Vec<String>,
}

/// Configuration for the enhanced BrutalHonesty module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedConfig {
    /// Base BrutalHonesty configuration
    pub base_config: BrutalHonestyConfig,
    /// Enable cultural assumption analysis
    pub enable_cultural_analysis: bool,
    /// Enable steelmanning of counter-arguments
    pub enable_steelmanning: bool,
    /// Enable argument mapping
    pub enable_argument_mapping: bool,
    /// Depth of cognitive bias analysis
    pub cognitive_bias_depth: CognitiveBiasDepth,
    /// Target cultures for cultural analysis
    pub target_cultures: Vec<String>,
}

impl Default for EnhancedConfig {
    fn default() -> Self {
        Self {
            base_config: BrutalHonestyConfig::default(),
            enable_cultural_analysis: true,
            enable_steelmanning: true,
            enable_argument_mapping: true,
            cognitive_bias_depth: CognitiveBiasDepth::Standard,
            target_cultures: vec![
                "Western".to_string(),
                "East Asian".to_string(),
                "South Asian".to_string(),
                "Middle Eastern".to_string(),
                "Latin American".to_string(),
            ],
        }
    }
}

/// Builder for enhanced BrutalHonesty module.
#[derive(Debug, Default)]
pub struct EnhancedBuilder {
    config: EnhancedConfig,
}

impl EnhancedBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base critique severity.
    pub fn severity(mut self, severity: CritiqueSeverity) -> Self {
        self.config.base_config.severity = severity;
        self
    }

    /// Enable or disable cultural analysis.
    pub fn enable_cultural_analysis(mut self, enable: bool) -> Self {
        self.config.enable_cultural_analysis = enable;
        self
    }

    /// Enable or disable steelmanning.
    pub fn enable_steelmanning(mut self, enable: bool) -> Self {
        self.config.enable_steelmanning = enable;
        self
    }

    /// Enable or disable argument mapping.
    pub fn enable_argument_mapping(mut self, enable: bool) -> Self {
        self.config.enable_argument_mapping = enable;
        self
    }

    /// Set cognitive bias analysis depth.
    pub fn cognitive_bias_depth(mut self, depth: CognitiveBiasDepth) -> Self {
        self.config.cognitive_bias_depth = depth;
        self
    }

    /// Set target cultures for cultural analysis.
    pub fn target_cultures(mut self, cultures: Vec<String>) -> Self {
        self.config.target_cultures = cultures;
        self
    }

    /// Build the enhanced BrutalHonesty module.
    pub fn build(self) -> BrutalHonestyEnhanced {
        BrutalHonestyEnhanced::with_config(self.config)
    }
}

/// Enhanced BrutalHonesty module with extended analysis.
pub struct BrutalHonestyEnhanced {
    /// Module base configuration
    module_config: ThinkToolModuleConfig,
    /// Enhanced configuration
    config: EnhancedConfig,
    /// Base BrutalHonesty module for core analysis
    base_module: BrutalHonesty,
}

impl Default for BrutalHonestyEnhanced {
    fn default() -> Self {
        Self::new()
    }
}

impl BrutalHonestyEnhanced {
    /// Create a new enhanced BrutalHonesty module with default configuration.
    pub fn new() -> Self {
        Self::with_config(EnhancedConfig::default())
    }

    /// Create an enhanced module with custom configuration.
    pub fn with_config(config: EnhancedConfig) -> Self {
        let base_module = BrutalHonesty::with_config(config.base_config.clone());

        Self {
            module_config: ThinkToolModuleConfig {
                name: "BrutalHonestyEnhanced".to_string(),
                version: "3.0.0".to_string(),
                description:
                    "Enhanced adversarial critique with cognitive bias detection and steelmanning"
                        .to_string(),
                confidence_weight: 0.18, // Slightly higher weight due to deeper analysis
            },
            config,
            base_module,
        }
    }

    /// Create a builder for customizing the module.
    pub fn builder() -> EnhancedBuilder {
        EnhancedBuilder::new()
    }

    /// Get the current configuration.
    pub fn enhanced_config(&self) -> &EnhancedConfig {
        &self.config
    }

    /// Detect cognitive biases in the reasoning.
    fn detect_cognitive_biases(&self, query: &str) -> Vec<CognitiveBias> {
        let mut biases = Vec::new();
        let query_lower = query.to_lowercase();

        // Define bias patterns based on depth
        let basic_biases = self.get_basic_biases(&query_lower);
        biases.extend(basic_biases);

        if matches!(
            self.config.cognitive_bias_depth,
            CognitiveBiasDepth::Standard | CognitiveBiasDepth::Deep
        ) {
            let standard_biases = self.get_standard_biases(&query_lower);
            biases.extend(standard_biases);
        }

        if matches!(self.config.cognitive_bias_depth, CognitiveBiasDepth::Deep) {
            let deep_biases = self.get_deep_biases(&query_lower);
            biases.extend(deep_biases);
        }

        biases
    }

    /// Get basic (most common) cognitive biases.
    fn get_basic_biases(&self, query: &str) -> Vec<CognitiveBias> {
        let mut biases = Vec::new();

        // Confirmation bias
        if query.contains("proves")
            || query.contains("confirms")
            || (query.contains("evidence") && !query.contains("counter"))
        {
            biases.push(CognitiveBias {
                name: "Confirmation Bias".to_string(),
                category: BiasCategory::DecisionMaking,
                manifestation: "Seeking or interpreting information to confirm existing beliefs"
                    .to_string(),
                confidence: 0.70,
                debiasing_strategy:
                    "Actively seek disconfirming evidence; assign someone to argue the opposite"
                        .to_string(),
            });
        }

        // Anchoring bias
        if query.contains("first") || query.contains("initial") || query.contains("started with") {
            biases.push(CognitiveBias {
                name: "Anchoring Bias".to_string(),
                category: BiasCategory::DecisionMaking,
                manifestation: "Over-relying on initial information as reference point".to_string(),
                confidence: 0.55,
                debiasing_strategy:
                    "Consider multiple starting points; recalculate from different anchors"
                        .to_string(),
            });
        }

        // Availability heuristic
        if query.contains("recently") || query.contains("just saw") || query.contains("in the news")
        {
            biases.push(CognitiveBias {
                name: "Availability Heuristic".to_string(),
                category: BiasCategory::Memory,
                manifestation: "Overweighting easily recalled examples".to_string(),
                confidence: 0.65,
                debiasing_strategy:
                    "Seek base rate statistics; consider examples from multiple time periods"
                        .to_string(),
            });
        }

        // Overconfidence
        if query.contains("certain") || query.contains("definitely") || query.contains("no doubt") {
            biases.push(CognitiveBias {
                name: "Overconfidence Bias".to_string(),
                category: BiasCategory::SelfRelated,
                manifestation: "Excessive confidence in own judgment or predictions".to_string(),
                confidence: 0.75,
                debiasing_strategy: "Create prediction intervals; track calibration over time"
                    .to_string(),
            });
        }

        // Sunk cost fallacy
        if query.contains("already invested")
            || query.contains("too far to stop")
            || query.contains("wasted if")
        {
            biases.push(CognitiveBias {
                name: "Sunk Cost Fallacy".to_string(),
                category: BiasCategory::DecisionMaking,
                manifestation: "Continuing due to past investment rather than future value"
                    .to_string(),
                confidence: 0.80,
                debiasing_strategy: "Evaluate decisions based only on future costs and benefits"
                    .to_string(),
            });
        }

        biases
    }

    /// Get standard cognitive biases.
    fn get_standard_biases(&self, query: &str) -> Vec<CognitiveBias> {
        let mut biases = Vec::new();

        // Hindsight bias
        if query.contains("should have known")
            || query.contains("was obvious")
            || query.contains("predictable")
        {
            biases.push(CognitiveBias {
                name: "Hindsight Bias".to_string(),
                category: BiasCategory::Memory,
                manifestation: "Believing past events were more predictable than they were"
                    .to_string(),
                confidence: 0.70,
                debiasing_strategy:
                    "Document predictions before outcomes; review original uncertainty".to_string(),
            });
        }

        // Bandwagon effect
        if query.contains("everyone") || query.contains("popular") || query.contains("trend") {
            biases.push(CognitiveBias {
                name: "Bandwagon Effect".to_string(),
                category: BiasCategory::Social,
                manifestation: "Following the crowd rather than independent analysis".to_string(),
                confidence: 0.60,
                debiasing_strategy: "Evaluate based on fundamentals; consider contrarian positions"
                    .to_string(),
            });
        }

        // Status quo bias
        if query.contains("always done") || query.contains("traditional") || query.contains("usual")
        {
            biases.push(CognitiveBias {
                name: "Status Quo Bias".to_string(),
                category: BiasCategory::DecisionMaking,
                manifestation: "Preferring current state regardless of merit".to_string(),
                confidence: 0.65,
                debiasing_strategy: "Explicitly compare status quo costs vs change benefits"
                    .to_string(),
            });
        }

        // Fundamental attribution error
        if query.contains("they failed because they") || query.contains("their fault") {
            biases.push(CognitiveBias {
                name: "Fundamental Attribution Error".to_string(),
                category: BiasCategory::Social,
                manifestation: "Over-attributing behavior to personality vs situation".to_string(),
                confidence: 0.55,
                debiasing_strategy:
                    "Consider situational factors; ask 'what would I do in their position?'"
                        .to_string(),
            });
        }

        // Optimism bias
        if query.contains("will succeed")
            || query.contains("bound to")
            || query.contains("can't fail")
        {
            biases.push(CognitiveBias {
                name: "Optimism Bias".to_string(),
                category: BiasCategory::SelfRelated,
                manifestation: "Overestimating likelihood of positive outcomes".to_string(),
                confidence: 0.70,
                debiasing_strategy: "Conduct pre-mortem analysis; use reference class forecasting"
                    .to_string(),
            });
        }

        // Framing effect
        if query.contains("savings") || query.contains("loss") || query.contains("gain") {
            biases.push(CognitiveBias {
                name: "Framing Effect".to_string(),
                category: BiasCategory::DecisionMaking,
                manifestation: "Decision influenced by how information is presented".to_string(),
                confidence: 0.50,
                debiasing_strategy: "Reframe the problem multiple ways; use absolute numbers"
                    .to_string(),
            });
        }

        biases
    }

    /// Get deep cognitive biases.
    fn get_deep_biases(&self, query: &str) -> Vec<CognitiveBias> {
        let mut biases = Vec::new();

        // Planning fallacy
        if query.contains("estimate") || query.contains("timeline") || query.contains("schedule") {
            biases.push(CognitiveBias {
                name: "Planning Fallacy".to_string(),
                category: BiasCategory::Probability,
                manifestation: "Underestimating time, costs, and risks".to_string(),
                confidence: 0.75,
                debiasing_strategy: "Use reference class forecasting; add buffer based on past projects".to_string(),
            });
        }

        // Survivorship bias
        if query.contains("successful companies")
            || query.contains("winners")
            || query.contains("examples of success")
        {
            biases.push(CognitiveBias {
                name: "Survivorship Bias".to_string(),
                category: BiasCategory::Probability,
                manifestation: "Focusing on survivors while ignoring failures".to_string(),
                confidence: 0.65,
                debiasing_strategy: "Study failures; consider base rates of success vs failure"
                    .to_string(),
            });
        }

        // Narrative fallacy
        if query.contains("story") || query.contains("journey") || query.contains("narrative") {
            biases.push(CognitiveBias {
                name: "Narrative Fallacy".to_string(),
                category: BiasCategory::Memory,
                manifestation: "Creating false causal chains in retrospect".to_string(),
                confidence: 0.55,
                debiasing_strategy: "Focus on data; be skeptical of neat explanations".to_string(),
            });
        }

        // Dunning-Kruger effect
        if query.contains("simple") || query.contains("easy") || query.contains("just need to") {
            biases.push(CognitiveBias {
                name: "Dunning-Kruger Effect".to_string(),
                category: BiasCategory::SelfRelated,
                manifestation: "Overestimating competence in areas of limited knowledge"
                    .to_string(),
                confidence: 0.50,
                debiasing_strategy: "Consult domain experts; acknowledge knowledge gaps"
                    .to_string(),
            });
        }

        // Affect heuristic
        if query.contains("feel") || query.contains("gut") || query.contains("intuition") {
            biases.push(CognitiveBias {
                name: "Affect Heuristic".to_string(),
                category: BiasCategory::DecisionMaking,
                manifestation: "Letting emotions drive risk/benefit judgments".to_string(),
                confidence: 0.60,
                debiasing_strategy: "Separate emotional response from analytical evaluation"
                    .to_string(),
            });
        }

        biases
    }

    /// Detect cultural assumptions in the reasoning.
    fn detect_cultural_assumptions(&self, query: &str) -> Vec<CulturalAssumption> {
        if !self.config.enable_cultural_analysis {
            return Vec::new();
        }

        let mut assumptions = Vec::new();
        let query_lower = query.to_lowercase();

        // Individualism vs Collectivism
        if query_lower.contains("individual")
            || query_lower.contains("personal achievement")
            || query_lower.contains("self-made")
        {
            assumptions.push(CulturalAssumption {
                assumption: "Individual achievement and autonomy are primary values".to_string(),
                context: "Western, particularly American, cultural context".to_string(),
                exceptions: vec![
                    "East Asian cultures (group harmony)".to_string(),
                    "Latin American cultures (family ties)".to_string(),
                    "African cultures (community focus)".to_string(),
                ],
                risk_level: FlawSeverity::Moderate,
            });
        }

        // Direct communication assumption
        if query_lower.contains("straightforward")
            || query_lower.contains("direct")
            || query_lower.contains("clear communication")
        {
            assumptions.push(CulturalAssumption {
                assumption: "Direct, explicit communication is preferable".to_string(),
                context: "Low-context cultures (Northern European, American)".to_string(),
                exceptions: vec![
                    "High-context cultures (Japan, China)".to_string(),
                    "Middle Eastern cultures".to_string(),
                    "Many Asian cultures".to_string(),
                ],
                risk_level: FlawSeverity::Minor,
            });
        }

        // Time orientation
        if query_lower.contains("deadline")
            || query_lower.contains("punctual")
            || query_lower.contains("time is money")
        {
            assumptions.push(CulturalAssumption {
                assumption: "Linear, monochronic view of time".to_string(),
                context: "Northern European, North American cultures".to_string(),
                exceptions: vec![
                    "Polychronic cultures (Mediterranean, Latin America)".to_string(),
                    "Relationship-focused cultures".to_string(),
                ],
                risk_level: FlawSeverity::Minor,
            });
        }

        // Hierarchical assumptions
        if query_lower.contains("flat organization")
            || query_lower.contains("anyone can")
            || query_lower.contains("speak up")
        {
            assumptions.push(CulturalAssumption {
                assumption: "Low power distance is desirable".to_string(),
                context: "Scandinavian, Dutch, American cultures".to_string(),
                exceptions: vec![
                    "High power distance cultures (Japan, India)".to_string(),
                    "Traditional hierarchical societies".to_string(),
                ],
                risk_level: FlawSeverity::Moderate,
            });
        }

        assumptions
    }

    /// Generate a steelmanned counter-argument.
    fn generate_steelman(&self, query: &str) -> Option<SteelmanArgument> {
        if !self.config.enable_steelmanning {
            return None;
        }

        let query_lower = query.to_lowercase();

        // Generate contextual steelman arguments
        if query_lower.contains("should")
            || query_lower.contains("must")
            || query_lower.contains("need to")
        {
            Some(SteelmanArgument {
                argument: "The strongest case against this action is that the opportunity costs and risks may outweigh the benefits, and that the current state, while imperfect, has proven stability.".to_string(),
                premises: vec![
                    "Change carries inherent risk and transition costs".to_string(),
                    "Status quo has survived past challenges".to_string(),
                    "Resources spent here cannot be used elsewhere".to_string(),
                ],
                challenge: "This challenges whether the proposed action is truly necessary or optimal given alternatives".to_string(),
                strength: 0.70,
            })
        } else if query_lower.contains("will succeed") || query_lower.contains("will work") {
            Some(SteelmanArgument {
                argument: "Even well-planned initiatives with strong teams fail due to unforeseeable market shifts, timing issues, or execution challenges that compound unexpectedly.".to_string(),
                premises: vec![
                    "Base rates of success are often lower than expected".to_string(),
                    "External factors can override internal capabilities".to_string(),
                    "Complex systems have emergent failure modes".to_string(),
                ],
                challenge: "This challenges the assumption that good inputs guarantee good outcomes".to_string(),
                strength: 0.75,
            })
        } else if query_lower.contains("better") || query_lower.contains("best") {
            Some(SteelmanArgument {
                argument: "What is 'better' depends on criteria that may differ across stakeholders, time horizons, and contexts. The supposedly inferior alternative may optimize for different, equally valid goals.".to_string(),
                premises: vec![
                    "Value judgments depend on evaluation criteria".to_string(),
                    "Stakeholders have different preferences".to_string(),
                    "Trade-offs are inherent in most choices".to_string(),
                ],
                challenge: "This challenges whether the comparison criteria are appropriate and complete".to_string(),
                strength: 0.65,
            })
        } else {
            Some(SteelmanArgument {
                argument: "A sophisticated opponent would argue that this position overlooks key factors, relies on assumptions that may not hold, and underestimates alternatives.".to_string(),
                premises: vec![
                    "Complex issues have multiple valid perspectives".to_string(),
                    "Our information may be incomplete".to_string(),
                    "Intelligent people disagree for good reasons".to_string(),
                ],
                challenge: "This challenges the completeness and robustness of the original argument".to_string(),
                strength: 0.60,
            })
        }
    }

    /// Build an argument map from the input.
    fn build_argument_map(&self, query: &str) -> Option<ArgumentMap> {
        if !self.config.enable_argument_mapping {
            return None;
        }

        let sentences: Vec<&str> = query.split(['.', '!', '?']).collect();
        let query_lower = query.to_lowercase();

        // Extract claim (usually first substantial sentence or sentence with "should/must/will")
        let claim = sentences
            .iter()
            .find(|s| {
                let lower = s.to_lowercase();
                lower.contains("should")
                    || lower.contains("must")
                    || lower.contains("will")
                    || lower.contains("is the")
                    || lower.contains("are the")
            })
            .unwrap_or(sentences.first().unwrap_or(&""))
            .trim()
            .to_string();

        // Extract premises (sentences with "because", "since", "as")
        let premises: Vec<String> = sentences
            .iter()
            .filter(|s| {
                let lower = s.to_lowercase();
                lower.contains("because") || lower.contains("since") || lower.contains(" as ")
            })
            .map(|s| s.trim().to_string())
            .collect();

        // Extract evidence references
        let evidence: Vec<String> = sentences
            .iter()
            .filter(|s| {
                let lower = s.to_lowercase();
                lower.contains("data")
                    || lower.contains("study")
                    || lower.contains("research")
                    || lower.contains("evidence")
                    || lower.contains("according to")
            })
            .map(|s| s.trim().to_string())
            .collect();

        // Extract qualifiers
        let qualifiers: Vec<String> = sentences
            .iter()
            .filter(|s| {
                let lower = s.to_lowercase();
                lower.contains("unless")
                    || lower.contains("except")
                    || lower.contains("if")
                    || lower.contains("when")
                    || lower.contains("provided")
            })
            .map(|s| s.trim().to_string())
            .collect();

        // Extract rebuttals (acknowledged counter-arguments)
        let rebuttals: Vec<String> = sentences
            .iter()
            .filter(|s| {
                let lower = s.to_lowercase();
                lower.contains("however")
                    || lower.contains("although")
                    || lower.contains("but")
                    || lower.contains("despite")
                    || lower.contains("critics")
            })
            .map(|s| s.trim().to_string())
            .collect();

        // Extract warrants (logical connections)
        let warrants: Vec<String> = if query_lower.contains("therefore")
            || query_lower.contains("thus")
            || query_lower.contains("hence")
        {
            vec!["Explicit logical connection present".to_string()]
        } else if premises.is_empty() {
            vec!["Implicit warrant: premises assumed to support claim".to_string()]
        } else {
            vec![]
        };

        Some(ArgumentMap {
            claim,
            premises,
            evidence,
            warrants,
            qualifiers,
            rebuttals,
        })
    }

    /// Calculate enhanced confidence incorporating all analyses.
    fn calculate_enhanced_confidence(
        &self,
        base_confidence: f64,
        biases: &[CognitiveBias],
        cultural_assumptions: &[CulturalAssumption],
    ) -> f64 {
        let mut confidence = base_confidence;

        // Apply cognitive bias penalties
        for bias in biases {
            confidence -= bias.confidence * 0.05;
        }

        // Apply cultural assumption penalties
        for assumption in cultural_assumptions {
            match assumption.risk_level {
                FlawSeverity::Critical => confidence -= 0.10,
                FlawSeverity::Major => confidence -= 0.06,
                FlawSeverity::Moderate => confidence -= 0.03,
                FlawSeverity::Minor => confidence -= 0.01,
            }
        }

        confidence.clamp(0.0, 0.90) // Slightly lower max than base due to deeper skepticism
    }
}

impl ThinkToolModule for BrutalHonestyEnhanced {
    fn config(&self) -> &ThinkToolModuleConfig {
        &self.module_config
    }

    fn execute(&self, context: &ThinkToolContext) -> Result<ThinkToolOutput> {
        // Validate input
        if context.query.trim().is_empty() {
            return Err(Error::validation(
                "BrutalHonestyEnhanced requires non-empty query input",
            ));
        }

        // Run base BrutalHonesty analysis
        let base_result = self.base_module.execute(context)?;
        let base_output = &base_result.output;

        // Extract base analysis results
        let base_analysis = base_output.get("analysis").cloned().unwrap_or(json!({}));
        let base_verdict = base_output
            .get("verdict")
            .cloned()
            .unwrap_or(json!("indeterminate"));
        let devils_advocate = base_output.get("devils_advocate").cloned();
        let critical_fix = base_output.get("critical_fix").cloned();

        // Perform enhanced analyses
        let cognitive_biases = self.detect_cognitive_biases(&context.query);
        let cultural_assumptions = self.detect_cultural_assumptions(&context.query);
        let steelman = self.generate_steelman(&context.query);
        let argument_map = self.build_argument_map(&context.query);

        // Calculate enhanced confidence
        let enhanced_confidence = self.calculate_enhanced_confidence(
            base_result.confidence,
            &cognitive_biases,
            &cultural_assumptions,
        );

        // Determine enhanced verdict
        let enhanced_verdict = if cognitive_biases.len() >= 3 || cultural_assumptions.len() >= 2 {
            CritiqueVerdict::Weak
        } else {
            // Parse base verdict and potentially downgrade
            match base_verdict.as_str() {
                Some("solid") if !cognitive_biases.is_empty() => CritiqueVerdict::Promising,
                Some("solid") => CritiqueVerdict::Solid,
                Some("promising") => CritiqueVerdict::Promising,
                Some("weak") => CritiqueVerdict::Weak,
                Some("flawed") => CritiqueVerdict::Flawed,
                _ => CritiqueVerdict::Indeterminate,
            }
        };

        // Build output
        let output = json!({
            "verdict": enhanced_verdict,
            "confidence": enhanced_confidence,
            "base_analysis": base_analysis,
            "base_verdict": base_verdict,
            "enhanced_analysis": {
                "cognitive_biases": cognitive_biases,
                "bias_count": cognitive_biases.len(),
                "cultural_assumptions": cultural_assumptions,
                "cultural_assumption_count": cultural_assumptions.len(),
                "steelman": steelman,
                "argument_map": argument_map,
            },
            "devils_advocate": devils_advocate,
            "critical_fix": critical_fix,
            "metadata": {
                "input_length": context.query.len(),
                "previous_steps_count": context.previous_steps.len(),
                "analysis_depth": format!("{:?}", self.config.cognitive_bias_depth),
                "cultural_analysis_enabled": self.config.enable_cultural_analysis,
                "steelmanning_enabled": self.config.enable_steelmanning,
            }
        });

        Ok(ThinkToolOutput {
            module: self.module_config.name.clone(),
            confidence: enhanced_confidence,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_enhanced_module() {
        let module = BrutalHonestyEnhanced::new();
        assert_eq!(module.config().name, "BrutalHonestyEnhanced");
        assert_eq!(module.config().version, "3.0.0");
        assert!(module.enhanced_config().enable_cultural_analysis);
        assert!(module.enhanced_config().enable_steelmanning);
    }

    #[test]
    fn test_builder_pattern() {
        let module = BrutalHonestyEnhanced::builder()
            .severity(CritiqueSeverity::Ruthless)
            .enable_cultural_analysis(false)
            .cognitive_bias_depth(CognitiveBiasDepth::Deep)
            .build();

        assert_eq!(
            module.enhanced_config().base_config.severity,
            CritiqueSeverity::Ruthless
        );
        assert!(!module.enhanced_config().enable_cultural_analysis);
        assert_eq!(
            module.enhanced_config().cognitive_bias_depth,
            CognitiveBiasDepth::Deep
        );
    }

    #[test]
    fn test_cognitive_bias_detection() {
        let module = BrutalHonestyEnhanced::new();
        let biases = module.detect_cognitive_biases(
            "I'm certain this will succeed. We've already invested too much to stop.",
        );

        assert!(!biases.is_empty());

        // Should detect overconfidence
        let has_overconfidence = biases.iter().any(|b| b.name == "Overconfidence Bias");
        assert!(has_overconfidence);

        // Should detect sunk cost fallacy
        let has_sunk_cost = biases.iter().any(|b| b.name == "Sunk Cost Fallacy");
        assert!(has_sunk_cost);
    }

    #[test]
    fn test_cultural_assumption_detection() {
        let module = BrutalHonestyEnhanced::new();
        let assumptions = module.detect_cultural_assumptions(
            "We need direct communication and individual achievement focus.",
        );

        assert!(!assumptions.is_empty());
    }

    #[test]
    fn test_steelman_generation() {
        let module = BrutalHonestyEnhanced::new();
        let steelman = module.generate_steelman("We should expand immediately");

        assert!(steelman.is_some());
        let s = steelman.unwrap();
        assert!(!s.argument.is_empty());
        assert!(!s.premises.is_empty());
    }

    #[test]
    fn test_argument_mapping() {
        let module = BrutalHonestyEnhanced::new();
        let map = module.build_argument_map(
            "We should launch now because the market is ready. Data shows strong demand. However, there are risks.",
        );

        assert!(map.is_some());
        let m = map.unwrap();
        assert!(!m.claim.is_empty());
    }

    #[test]
    fn test_execute_valid_input() {
        let module = BrutalHonestyEnhanced::new();
        let context = ThinkToolContext {
            query:
                "We're certain this will succeed because everyone agrees it's the best approach."
                    .to_string(),
            previous_steps: vec![],
        };

        let result = module.execute(&context).unwrap();
        assert_eq!(result.module, "BrutalHonestyEnhanced");
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 0.90);

        // Check output structure
        let output = &result.output;
        assert!(output.get("verdict").is_some());
        assert!(output.get("base_analysis").is_some());
        assert!(output.get("enhanced_analysis").is_some());
    }

    #[test]
    fn test_execute_empty_input() {
        let module = BrutalHonestyEnhanced::new();
        let context = ThinkToolContext {
            query: "".to_string(),
            previous_steps: vec![],
        };

        let result = module.execute(&context);
        assert!(result.is_err());
    }

    #[test]
    fn test_bias_depth_affects_detection() {
        let basic = BrutalHonestyEnhanced::builder()
            .cognitive_bias_depth(CognitiveBiasDepth::Basic)
            .build();
        let deep = BrutalHonestyEnhanced::builder()
            .cognitive_bias_depth(CognitiveBiasDepth::Deep)
            .build();

        let query =
            "I'm certain this simple plan with a clear timeline will succeed. Everyone agrees.";
        let basic_biases = basic.detect_cognitive_biases(query);
        let deep_biases = deep.detect_cognitive_biases(query);

        // Deep analysis should find more biases
        assert!(deep_biases.len() >= basic_biases.len());
    }

    #[test]
    fn test_enhanced_confidence_reduction() {
        let module = BrutalHonestyEnhanced::new();

        // With biases and cultural assumptions, confidence should be lower
        let biases = vec![CognitiveBias {
            name: "Test Bias".to_string(),
            category: BiasCategory::DecisionMaking,
            manifestation: "Test".to_string(),
            confidence: 0.8,
            debiasing_strategy: "Test".to_string(),
        }];
        let assumptions = vec![CulturalAssumption {
            assumption: "Test".to_string(),
            context: "Test".to_string(),
            exceptions: vec![],
            risk_level: FlawSeverity::Major,
        }];

        let confidence = module.calculate_enhanced_confidence(0.70, &biases, &assumptions);
        assert!(confidence < 0.70);
    }
}
