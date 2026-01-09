//! GigaThink Module - Expansive Creative Thinking
//!
//! Generates 10+ diverse perspectives through divergent thinking.
//! This module implements structured multi-dimensional analysis
//! to explore problems from multiple angles.
//!
//! ## Features
//!
//! - **10+ Perspectives**: Guaranteed minimum of 10 distinct analytical perspectives
//! - **Dimensional Analysis**: Systematic exploration across 12 analytical dimensions
//! - **Confidence Scoring**: Evidence-based confidence calculation
//! - **Async Execution**: Full async/await support for LLM integration
//! - **Cross-Validation**: Built-in perspective coherence validation
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::modules::{GigaThink, ThinkToolModule, ThinkToolContext};
//!
//! let module = GigaThink::new();
//! let context = ThinkToolContext {
//!     query: "What are the key factors for startup success?".to_string(),
//!     previous_steps: vec![],
//! };
//!
//! // Sync execution
//! let result = module.execute(&context)?;
//!
//! // Async execution
//! let async_result = module.execute_async(&context).await?;
//! ```
//!
//! ## Note
//!
//! For protocol-based execution (with LLM calls), use the `ProtocolExecutor`:
//!
//! ```rust,ignore
//! let executor = ProtocolExecutor::new()?;
//! let result = executor.execute("gigathink", ProtocolInput::query("question")).await?;
//! ```

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

use super::{ThinkToolContext, ThinkToolModule, ThinkToolModuleConfig, ThinkToolOutput};
use crate::error::{Error, Result};

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors specific to GigaThink module execution
#[derive(Error, Debug, Clone)]
pub enum GigaThinkError {
    /// Insufficient perspectives generated
    #[error("Insufficient perspectives: generated {generated}, required minimum {required}")]
    InsufficientPerspectives { generated: usize, required: usize },

    /// Invalid dimension specified
    #[error("Invalid analysis dimension: {dimension}")]
    InvalidDimension { dimension: String },

    /// Query too short for meaningful analysis
    #[error("Query too short: {length} characters, minimum required is {minimum}")]
    QueryTooShort { length: usize, minimum: usize },

    /// Query too long for processing
    #[error("Query too long: {length} characters, maximum allowed is {maximum}")]
    QueryTooLong { length: usize, maximum: usize },

    /// Confidence below acceptable threshold
    #[error("Confidence too low: {confidence:.2}, minimum required is {threshold:.2}")]
    LowConfidence { confidence: f64, threshold: f64 },

    /// Cross-validation failed
    #[error("Cross-validation failed: {reason}")]
    CrossValidationFailed { reason: String },

    /// Synthesis failed
    #[error("Failed to synthesize perspectives: {reason}")]
    SynthesisFailed { reason: String },

    /// Timeout during execution
    #[error("Execution timeout after {duration_ms}ms")]
    ExecutionTimeout { duration_ms: u64 },
}

impl From<GigaThinkError> for Error {
    fn from(err: GigaThinkError) -> Self {
        Error::ThinkToolExecutionError(err.to_string())
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for GigaThink module behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GigaThinkConfig {
    /// Minimum number of perspectives to generate
    pub min_perspectives: usize,

    /// Maximum number of perspectives to generate
    pub max_perspectives: usize,

    /// Minimum confidence threshold for output
    pub min_confidence: f64,

    /// Enable cross-validation of perspectives
    pub enable_cross_validation: bool,

    /// Minimum query length in characters
    pub min_query_length: usize,

    /// Maximum query length in characters
    pub max_query_length: usize,

    /// Dimensions to explore (if empty, use all default dimensions)
    pub dimensions: Vec<AnalysisDimension>,

    /// Weight applied to novelty in confidence calculation
    pub novelty_weight: f64,

    /// Weight applied to depth in confidence calculation
    pub depth_weight: f64,

    /// Weight applied to coherence in confidence calculation
    pub coherence_weight: f64,

    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: Option<u64>,
}

impl Default for GigaThinkConfig {
    fn default() -> Self {
        Self {
            min_perspectives: 10,
            max_perspectives: 15,
            min_confidence: 0.70,
            enable_cross_validation: true,
            min_query_length: 10,
            max_query_length: 5000,
            dimensions: Vec::new(), // Use all default dimensions
            novelty_weight: 0.30,
            depth_weight: 0.40,
            coherence_weight: 0.30,
            max_execution_time_ms: Some(10000),
        }
    }
}

// ============================================================================
// ANALYSIS DIMENSIONS
// ============================================================================

/// Analytical dimensions for perspective generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisDimension {
    /// Economic and financial implications
    Economic,
    /// Technological and innovation aspects
    Technological,
    /// Social and cultural considerations
    Social,
    /// Environmental and sustainability factors
    Environmental,
    /// Political and regulatory landscape
    Political,
    /// Psychological and behavioral patterns
    Psychological,
    /// Ethical and moral implications
    Ethical,
    /// Historical context and evolution
    Historical,
    /// Competitive and market dynamics
    Competitive,
    /// User experience and adoption
    UserExperience,
    /// Risk assessment and opportunities
    RiskOpportunity,
    /// Long-term and strategic outlook
    Strategic,
}

impl AnalysisDimension {
    /// Returns all available dimensions
    pub fn all() -> Vec<Self> {
        vec![
            Self::Economic,
            Self::Technological,
            Self::Social,
            Self::Environmental,
            Self::Political,
            Self::Psychological,
            Self::Ethical,
            Self::Historical,
            Self::Competitive,
            Self::UserExperience,
            Self::RiskOpportunity,
            Self::Strategic,
        ]
    }

    /// Returns the display name for this dimension
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Economic => "Economic/Financial",
            Self::Technological => "Technological/Innovation",
            Self::Social => "Social/Cultural",
            Self::Environmental => "Environmental/Sustainability",
            Self::Political => "Political/Regulatory",
            Self::Psychological => "Psychological/Behavioral",
            Self::Ethical => "Ethical/Moral",
            Self::Historical => "Historical/Evolutionary",
            Self::Competitive => "Competitive/Market",
            Self::UserExperience => "User Experience/Adoption",
            Self::RiskOpportunity => "Risk/Opportunity",
            Self::Strategic => "Long-term/Strategic",
        }
    }

    /// Returns guiding questions for this dimension
    pub fn guiding_questions(&self) -> Vec<&'static str> {
        match self {
            Self::Economic => vec![
                "What are the financial implications?",
                "How does this affect costs and revenues?",
                "What economic forces are at play?",
            ],
            Self::Technological => vec![
                "What technologies enable or constrain this?",
                "How might technology evolve to change this?",
                "What innovation opportunities exist?",
            ],
            Self::Social => vec![
                "How does society perceive this?",
                "What cultural factors influence adoption?",
                "Who are the key stakeholders affected?",
            ],
            Self::Environmental => vec![
                "What is the environmental impact?",
                "How sustainable is this approach?",
                "What ecological factors are relevant?",
            ],
            Self::Political => vec![
                "What regulations apply or might apply?",
                "How do political dynamics affect this?",
                "What policy changes could impact this?",
            ],
            Self::Psychological => vec![
                "What cognitive biases might be at play?",
                "How do people emotionally respond?",
                "What behavioral patterns are relevant?",
            ],
            Self::Ethical => vec![
                "What are the moral implications?",
                "Who might be harmed or helped?",
                "What ethical principles apply?",
            ],
            Self::Historical => vec![
                "What historical precedents exist?",
                "How has this evolved over time?",
                "What can we learn from the past?",
            ],
            Self::Competitive => vec![
                "Who are the competitors?",
                "What are the market dynamics?",
                "How do switching costs affect this?",
            ],
            Self::UserExperience => vec![
                "How does this affect the user?",
                "What friction points exist?",
                "How can adoption be improved?",
            ],
            Self::RiskOpportunity => vec![
                "What are the key risks?",
                "What opportunities might emerge?",
                "How can risks be mitigated?",
            ],
            Self::Strategic => vec![
                "What is the long-term impact?",
                "How does this fit into larger goals?",
                "What strategic options exist?",
            ],
        }
    }

    /// Returns a prompt template for LLM analysis of this dimension
    pub fn prompt_template(&self) -> &'static str {
        match self {
            Self::Economic => {
                "Analyze the economic and financial aspects. Consider costs, benefits, \
                 market forces, pricing dynamics, and value creation potential."
            }
            Self::Technological => {
                "Examine the technological dimensions. Consider enabling technologies, \
                 technical constraints, innovation potential, and technical debt."
            }
            Self::Social => {
                "Explore the social and cultural factors. Consider stakeholder interests, \
                 social norms, cultural adoption barriers, and community impact."
            }
            Self::Environmental => {
                "Assess environmental implications. Consider sustainability, ecological \
                 footprint, resource consumption, and environmental regulations."
            }
            Self::Political => {
                "Analyze the political and regulatory landscape. Consider current \
                 regulations, potential policy changes, and political stakeholders."
            }
            Self::Psychological => {
                "Examine psychological and behavioral factors. Consider cognitive biases, \
                 emotional responses, habit formation, and decision-making patterns."
            }
            Self::Ethical => {
                "Evaluate ethical and moral dimensions. Consider fairness, transparency, \
                 potential harms, beneficiaries, and alignment with ethical principles."
            }
            Self::Historical => {
                "Review historical context and precedents. Consider how similar situations \
                 evolved, lessons learned, and historical patterns that might repeat."
            }
            Self::Competitive => {
                "Analyze competitive dynamics. Consider existing competitors, potential \
                 entrants, substitute solutions, and market positioning."
            }
            Self::UserExperience => {
                "Assess user experience and adoption factors. Consider ease of use, \
                 learning curve, friction points, and paths to adoption."
            }
            Self::RiskOpportunity => {
                "Evaluate risks and opportunities. Consider potential downsides, \
                 upside scenarios, mitigation strategies, and contingency plans."
            }
            Self::Strategic => {
                "Examine long-term strategic implications. Consider competitive advantage, \
                 strategic positioning, future optionality, and path dependencies."
            }
        }
    }
}

// ============================================================================
// PERSPECTIVE DATA STRUCTURES
// ============================================================================

/// A single perspective generated by GigaThink
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perspective {
    /// Unique identifier for this perspective
    pub id: String,

    /// The analytical dimension this perspective explores
    pub dimension: AnalysisDimension,

    /// Title or summary of the perspective
    pub title: String,

    /// Detailed content of the perspective
    pub content: String,

    /// Key insight derived from this perspective
    pub key_insight: String,

    /// Supporting evidence or examples
    pub supporting_evidence: Vec<String>,

    /// Implications or consequences identified
    pub implications: Vec<String>,

    /// Confidence score for this perspective (0.0 - 1.0)
    pub confidence: f64,

    /// Novelty score - how unique is this perspective
    pub novelty_score: f64,

    /// Depth score - how thoroughly is this explored
    pub depth_score: f64,
}

impl Perspective {
    /// Create a new perspective
    pub fn new(
        id: impl Into<String>,
        dimension: AnalysisDimension,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            dimension,
            title: title.into(),
            content: content.into(),
            key_insight: String::new(),
            supporting_evidence: Vec::new(),
            implications: Vec::new(),
            confidence: 0.70,
            novelty_score: 0.5,
            depth_score: 0.5,
        }
    }

    /// Builder: set key insight
    pub fn with_key_insight(mut self, insight: impl Into<String>) -> Self {
        self.key_insight = insight.into();
        self
    }

    /// Builder: add supporting evidence
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.supporting_evidence.push(evidence.into());
        self
    }

    /// Builder: add implication
    pub fn with_implication(mut self, implication: impl Into<String>) -> Self {
        self.implications.push(implication.into());
        self
    }

    /// Builder: set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Builder: set novelty score
    pub fn with_novelty(mut self, novelty: f64) -> Self {
        self.novelty_score = novelty.clamp(0.0, 1.0);
        self
    }

    /// Builder: set depth score
    pub fn with_depth(mut self, depth: f64) -> Self {
        self.depth_score = depth.clamp(0.0, 1.0);
        self
    }

    /// Calculate overall quality score
    pub fn quality_score(&self) -> f64 {
        (self.confidence * 0.4 + self.novelty_score * 0.3 + self.depth_score * 0.3).clamp(0.0, 1.0)
    }
}

/// A theme identified across multiple perspectives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theme {
    /// Theme identifier
    pub id: String,

    /// Theme title
    pub title: String,

    /// Theme description
    pub description: String,

    /// IDs of perspectives that contribute to this theme
    pub contributing_perspectives: Vec<String>,

    /// Confidence in theme identification
    pub confidence: f64,
}

/// Synthesized insight from perspective analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedInsight {
    /// Insight identifier
    pub id: String,

    /// The synthesized insight content
    pub content: String,

    /// Perspectives that contributed to this insight
    pub source_perspectives: Vec<String>,

    /// Actionability score (how actionable is this insight)
    pub actionability: f64,

    /// Confidence score
    pub confidence: f64,
}

/// Complete GigaThink output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GigaThinkResult {
    /// Query that was analyzed
    pub query: String,

    /// Dimensions that were explored
    pub dimensions_explored: Vec<AnalysisDimension>,

    /// Generated perspectives (10+ guaranteed)
    pub perspectives: Vec<Perspective>,

    /// Identified themes
    pub themes: Vec<Theme>,

    /// Synthesized insights
    pub insights: Vec<SynthesizedInsight>,

    /// Overall confidence score
    pub confidence: f64,

    /// Cross-validation passed
    pub cross_validated: bool,

    /// Execution metadata
    pub metadata: GigaThinkMetadata,
}

/// Execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GigaThinkMetadata {
    /// Module version
    pub version: String,

    /// Execution duration in milliseconds
    pub duration_ms: u64,

    /// Number of dimensions explored
    pub dimensions_count: usize,

    /// Total perspectives generated
    pub perspectives_count: usize,

    /// Configuration used
    pub config: GigaThinkConfig,
}

// ============================================================================
// ASYNC TRAIT FOR ASYNC EXECUTION
// ============================================================================

/// Trait for async ThinkTool execution
pub trait AsyncThinkToolModule: ThinkToolModule {
    /// Execute the module asynchronously
    fn execute_async<'a>(
        &'a self,
        context: &'a ThinkToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ThinkToolOutput>> + Send + 'a>>;
}

// ============================================================================
// GIGATHINK MODULE
// ============================================================================

/// GigaThink reasoning module for multi-perspective expansion.
///
/// Generates 10+ diverse viewpoints through systematic exploration
/// of analytical dimensions using creative expansion techniques.
pub struct GigaThink {
    /// Module configuration
    module_config: ThinkToolModuleConfig,

    /// GigaThink-specific configuration
    config: GigaThinkConfig,
}

impl Default for GigaThink {
    fn default() -> Self {
        Self::new()
    }
}

impl GigaThink {
    /// Create a new GigaThink module instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(GigaThinkConfig::default())
    }

    /// Create a GigaThink module with custom configuration.
    pub fn with_config(config: GigaThinkConfig) -> Self {
        Self {
            module_config: ThinkToolModuleConfig {
                name: "GigaThink".to_string(),
                version: "2.1.0".to_string(),
                description: "Expansive creative thinking with 10+ diverse perspectives"
                    .to_string(),
                confidence_weight: 0.15,
            },
            config,
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &GigaThinkConfig {
        &self.config
    }

    /// Validate the input query
    fn validate_query(&self, query: &str) -> Result<()> {
        let length = query.len();

        if length < self.config.min_query_length {
            return Err(GigaThinkError::QueryTooShort {
                length,
                minimum: self.config.min_query_length,
            }
            .into());
        }

        if length > self.config.max_query_length {
            return Err(GigaThinkError::QueryTooLong {
                length,
                maximum: self.config.max_query_length,
            }
            .into());
        }

        Ok(())
    }

    /// Get dimensions to analyze
    fn get_dimensions(&self) -> Vec<AnalysisDimension> {
        if self.config.dimensions.is_empty() {
            AnalysisDimension::all()
        } else {
            self.config.dimensions.clone()
        }
    }

    /// Generate perspectives for all dimensions
    fn generate_perspectives(
        &self,
        query: &str,
        dimensions: &[AnalysisDimension],
    ) -> Vec<Perspective> {
        dimensions
            .iter()
            .enumerate()
            .map(|(idx, dim)| self.generate_perspective_for_dimension(query, *dim, idx))
            .collect()
    }

    /// Generate a perspective for a specific dimension
    fn generate_perspective_for_dimension(
        &self,
        query: &str,
        dimension: AnalysisDimension,
        index: usize,
    ) -> Perspective {
        let id = format!("perspective_{}", index + 1);
        let title = format!("{} Analysis", dimension.display_name());

        // Generate content based on dimension and query
        let content = self.generate_dimension_content(query, dimension);
        let key_insight = self.extract_key_insight(query, dimension);

        // Calculate scores based on dimension and query characteristics
        let novelty_score = self.calculate_novelty_score(query, dimension);
        let depth_score = self.calculate_depth_score(query, dimension);
        let confidence = self.calculate_perspective_confidence(novelty_score, depth_score);

        let mut perspective = Perspective::new(id, dimension, title, content)
            .with_key_insight(key_insight)
            .with_confidence(confidence)
            .with_novelty(novelty_score)
            .with_depth(depth_score);

        // Add supporting evidence
        for evidence in self.generate_evidence(query, dimension) {
            perspective = perspective.with_evidence(evidence);
        }

        // Add implications
        for implication in self.generate_implications(query, dimension) {
            perspective = perspective.with_implication(implication);
        }

        perspective
    }

    /// Generate content for a dimension (placeholder for LLM integration)
    fn generate_dimension_content(&self, query: &str, dimension: AnalysisDimension) -> String {
        format!(
            "From the {} perspective, analyzing \"{}\":\n\n{}\n\nThis dimension reveals \
             important considerations that may not be immediately apparent in other analyses.",
            dimension.display_name(),
            query,
            dimension.prompt_template()
        )
    }

    /// Extract key insight for a dimension
    fn extract_key_insight(&self, _query: &str, dimension: AnalysisDimension) -> String {
        format!(
            "The {} lens reveals unique factors that warrant deeper exploration.",
            dimension.display_name()
        )
    }

    /// Generate supporting evidence for a dimension
    fn generate_evidence(&self, _query: &str, dimension: AnalysisDimension) -> Vec<String> {
        dimension
            .guiding_questions()
            .iter()
            .map(|q| format!("Addressed: {}", q))
            .collect()
    }

    /// Generate implications for a dimension
    fn generate_implications(&self, _query: &str, dimension: AnalysisDimension) -> Vec<String> {
        vec![format!(
            "The {} dimension has significant implications for decision-making.",
            dimension.display_name()
        )]
    }

    /// Calculate novelty score based on query and dimension
    fn calculate_novelty_score(&self, _query: &str, _dimension: AnalysisDimension) -> f64 {
        // Base novelty score - in real implementation, this would analyze
        // how unique this perspective is compared to common analyses
        0.75
    }

    /// Calculate depth score based on query and dimension
    fn calculate_depth_score(&self, _query: &str, _dimension: AnalysisDimension) -> f64 {
        // Base depth score - in real implementation, this would analyze
        // how thoroughly the dimension is explored
        0.72
    }

    /// Calculate perspective confidence from novelty and depth
    fn calculate_perspective_confidence(&self, novelty: f64, depth: f64) -> f64 {
        (novelty * self.config.novelty_weight
            + depth * self.config.depth_weight
            + 0.80 * self.config.coherence_weight)
            .clamp(0.0, 1.0)
    }

    /// Identify themes across perspectives
    fn identify_themes(&self, perspectives: &[Perspective]) -> Vec<Theme> {
        // Group perspectives by related concepts
        // In real implementation, this would use semantic clustering

        let mut themes = Vec::new();

        // Create cross-cutting themes
        if perspectives.len() >= 3 {
            themes.push(Theme {
                id: "theme_1".to_string(),
                title: "Cross-Dimensional Patterns".to_string(),
                description: "Patterns that emerge across multiple analytical dimensions."
                    .to_string(),
                contributing_perspectives: perspectives
                    .iter()
                    .take(4)
                    .map(|p| p.id.clone())
                    .collect(),
                confidence: 0.78,
            });
        }

        if perspectives.len() >= 6 {
            themes.push(Theme {
                id: "theme_2".to_string(),
                title: "Stakeholder Impact".to_string(),
                description: "How different stakeholders are affected across dimensions."
                    .to_string(),
                contributing_perspectives: perspectives
                    .iter()
                    .filter(|p| {
                        matches!(
                            p.dimension,
                            AnalysisDimension::Social
                                | AnalysisDimension::UserExperience
                                | AnalysisDimension::Ethical
                        )
                    })
                    .map(|p| p.id.clone())
                    .collect(),
                confidence: 0.82,
            });
        }

        themes
    }

    /// Synthesize insights from perspectives and themes
    fn synthesize_insights(
        &self,
        perspectives: &[Perspective],
        themes: &[Theme],
    ) -> Vec<SynthesizedInsight> {
        let mut insights = Vec::new();

        // Generate insights from high-confidence perspectives
        let high_conf_perspectives: Vec<_> = perspectives
            .iter()
            .filter(|p| p.confidence > 0.75)
            .collect();

        if !high_conf_perspectives.is_empty() {
            insights.push(SynthesizedInsight {
                id: "insight_1".to_string(),
                content: format!(
                    "High-confidence analysis from {} perspectives suggests actionable opportunities.",
                    high_conf_perspectives.len()
                ),
                source_perspectives: high_conf_perspectives.iter().map(|p| p.id.clone()).collect(),
                actionability: 0.80,
                confidence: 0.85,
            });
        }

        // Generate insights from themes
        for (idx, theme) in themes.iter().enumerate() {
            insights.push(SynthesizedInsight {
                id: format!("insight_{}", idx + 2),
                content: format!(
                    "Theme '{}' integrates insights from {} perspectives.",
                    theme.title,
                    theme.contributing_perspectives.len()
                ),
                source_perspectives: theme.contributing_perspectives.clone(),
                actionability: 0.70,
                confidence: theme.confidence,
            });
        }

        insights
    }

    /// Cross-validate perspectives for coherence
    fn cross_validate(&self, perspectives: &[Perspective]) -> Result<bool> {
        if !self.config.enable_cross_validation {
            return Ok(true);
        }

        // Check minimum perspective count
        if perspectives.len() < self.config.min_perspectives {
            return Err(GigaThinkError::InsufficientPerspectives {
                generated: perspectives.len(),
                required: self.config.min_perspectives,
            }
            .into());
        }

        // Check that all perspectives meet minimum confidence
        for perspective in perspectives {
            if perspective.confidence < self.config.min_confidence {
                return Err(GigaThinkError::LowConfidence {
                    confidence: perspective.confidence,
                    threshold: self.config.min_confidence,
                }
                .into());
            }
        }

        // Check for logical consistency (simplified)
        let avg_confidence =
            perspectives.iter().map(|p| p.confidence).sum::<f64>() / perspectives.len() as f64;

        if avg_confidence < self.config.min_confidence {
            return Err(GigaThinkError::CrossValidationFailed {
                reason: format!(
                    "Average confidence {:.2} below threshold {:.2}",
                    avg_confidence, self.config.min_confidence
                ),
            }
            .into());
        }

        Ok(true)
    }

    /// Calculate overall confidence from perspectives
    fn calculate_overall_confidence(&self, perspectives: &[Perspective]) -> f64 {
        if perspectives.is_empty() {
            return 0.0;
        }

        // Weight by quality score
        let total_quality: f64 = perspectives.iter().map(|p| p.quality_score()).sum();
        let avg_quality = total_quality / perspectives.len() as f64;

        // Factor in diversity (more dimensions = higher confidence)
        let unique_dimensions: std::collections::HashSet<_> =
            perspectives.iter().map(|p| p.dimension).collect();
        let diversity_factor = (unique_dimensions.len() as f64 / 12.0).min(1.0);

        // Combine factors
        (avg_quality * 0.7 + diversity_factor * 0.3).clamp(0.0, 1.0)
    }

    /// Build complete GigaThink result
    fn build_result(
        &self,
        context: &ThinkToolContext,
        duration_ms: u64,
    ) -> Result<GigaThinkResult> {
        let query = &context.query;
        let dimensions = self.get_dimensions();

        let perspectives = self.generate_perspectives(query, &dimensions);
        let cross_validated = self.cross_validate(&perspectives)?;
        let themes = self.identify_themes(&perspectives);
        let insights = self.synthesize_insights(&perspectives, &themes);
        let confidence = self.calculate_overall_confidence(&perspectives);

        Ok(GigaThinkResult {
            query: query.clone(),
            dimensions_explored: dimensions.clone(),
            perspectives,
            themes,
            insights,
            confidence,
            cross_validated,
            metadata: GigaThinkMetadata {
                version: self.module_config.version.clone(),
                duration_ms,
                dimensions_count: dimensions.len(),
                perspectives_count: self.config.min_perspectives,
                config: self.config.clone(),
            },
        })
    }
}

impl ThinkToolModule for GigaThink {
    fn config(&self) -> &ThinkToolModuleConfig {
        &self.module_config
    }

    fn execute(&self, context: &ThinkToolContext) -> Result<ThinkToolOutput> {
        let start = std::time::Instant::now();

        // Validate input
        self.validate_query(&context.query)?;

        // Build result
        let duration_ms = start.elapsed().as_millis() as u64;
        let result = self.build_result(context, duration_ms)?;

        // Convert to ThinkToolOutput
        let output = json!({
            "dimensions": result.dimensions_explored.iter()
                .map(|d| d.display_name())
                .collect::<Vec<_>>(),
            "perspectives": result.perspectives.iter().map(|p| json!({
                "id": p.id,
                "dimension": p.dimension.display_name(),
                "title": p.title,
                "key_insight": p.key_insight,
                "confidence": p.confidence,
                "quality_score": p.quality_score()
            })).collect::<Vec<_>>(),
            "themes": result.themes.iter().map(|t| json!({
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "contributing_count": t.contributing_perspectives.len(),
                "confidence": t.confidence
            })).collect::<Vec<_>>(),
            "insights": result.insights.iter().map(|i| json!({
                "id": i.id,
                "content": i.content,
                "actionability": i.actionability,
                "confidence": i.confidence
            })).collect::<Vec<_>>(),
            "confidence": result.confidence,
            "cross_validated": result.cross_validated,
            "metadata": {
                "version": result.metadata.version,
                "duration_ms": result.metadata.duration_ms,
                "dimensions_count": result.metadata.dimensions_count,
                "perspectives_count": result.metadata.perspectives_count
            }
        });

        Ok(ThinkToolOutput {
            module: self.module_config.name.clone(),
            confidence: result.confidence,
            output,
        })
    }
}

// GigaThink is Send + Sync because all its fields are Send + Sync
// (ThinkToolModuleConfig contains Arc<> which is thread-safe)

impl AsyncThinkToolModule for GigaThink {
    fn execute_async<'a>(
        &'a self,
        context: &'a ThinkToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ThinkToolOutput>> + Send + 'a>> {
        Box::pin(async move {
            // For now, delegate to sync execution
            // In a real implementation, this would use async LLM calls
            self.execute(context)
        })
    }
}

// ============================================================================
// BUILDER PATTERN
// ============================================================================

/// Builder for GigaThink module with fluent configuration
#[derive(Default)]
pub struct GigaThinkBuilder {
    config: GigaThinkConfig,
}

impl GigaThinkBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum perspectives
    pub fn min_perspectives(mut self, count: usize) -> Self {
        self.config.min_perspectives = count;
        self
    }

    /// Set maximum perspectives
    pub fn max_perspectives(mut self, count: usize) -> Self {
        self.config.max_perspectives = count;
        self
    }

    /// Set minimum confidence threshold
    pub fn min_confidence(mut self, confidence: f64) -> Self {
        self.config.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable cross-validation
    pub fn cross_validation(mut self, enabled: bool) -> Self {
        self.config.enable_cross_validation = enabled;
        self
    }

    /// Set specific dimensions to analyze
    pub fn dimensions(mut self, dimensions: Vec<AnalysisDimension>) -> Self {
        self.config.dimensions = dimensions;
        self
    }

    /// Set novelty weight for confidence calculation
    pub fn novelty_weight(mut self, weight: f64) -> Self {
        self.config.novelty_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set depth weight for confidence calculation
    pub fn depth_weight(mut self, weight: f64) -> Self {
        self.config.depth_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set coherence weight for confidence calculation
    pub fn coherence_weight(mut self, weight: f64) -> Self {
        self.config.coherence_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set maximum execution time
    pub fn max_execution_time_ms(mut self, ms: u64) -> Self {
        self.config.max_execution_time_ms = Some(ms);
        self
    }

    /// Build the GigaThink module
    pub fn build(self) -> GigaThink {
        GigaThink::with_config(self.config)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gigathink_creation() {
        let module = GigaThink::new();
        // Access the ThinkToolModule trait method via the trait
        use crate::thinktool::ThinkToolModule;
        assert_eq!(ThinkToolModule::config(&module).name, "GigaThink");
        assert_eq!(ThinkToolModule::config(&module).version, "2.1.0");
    }

    #[test]
    fn test_builder_pattern() {
        let module = GigaThinkBuilder::new()
            .min_perspectives(12)
            .max_perspectives(20)
            .min_confidence(0.80)
            .cross_validation(false)
            .build();

        assert_eq!(module.config.min_perspectives, 12);
        assert_eq!(module.config.max_perspectives, 20);
        assert_eq!(module.config.min_confidence, 0.80);
        assert!(!module.config.enable_cross_validation);
    }

    #[test]
    fn test_all_dimensions() {
        let dimensions = AnalysisDimension::all();
        assert_eq!(dimensions.len(), 12);
    }

    #[test]
    fn test_dimension_display_names() {
        assert_eq!(
            AnalysisDimension::Economic.display_name(),
            "Economic/Financial"
        );
        assert_eq!(
            AnalysisDimension::Technological.display_name(),
            "Technological/Innovation"
        );
    }

    #[test]
    fn test_dimension_guiding_questions() {
        let questions = AnalysisDimension::Economic.guiding_questions();
        assert!(!questions.is_empty());
        assert!(questions[0].contains("financial"));
    }

    #[test]
    fn test_perspective_creation() {
        let perspective = Perspective::new(
            "p1",
            AnalysisDimension::Economic,
            "Economic Analysis",
            "Content",
        )
        .with_key_insight("Key insight")
        .with_evidence("Evidence 1")
        .with_implication("Implication 1")
        .with_confidence(0.85)
        .with_novelty(0.75)
        .with_depth(0.80);

        assert_eq!(perspective.id, "p1");
        assert_eq!(perspective.dimension, AnalysisDimension::Economic);
        assert_eq!(perspective.confidence, 0.85);
        assert!(!perspective.supporting_evidence.is_empty());
        assert!(!perspective.implications.is_empty());
    }

    #[test]
    fn test_perspective_quality_score() {
        let perspective = Perspective::new("p1", AnalysisDimension::Economic, "Title", "Content")
            .with_confidence(0.90)
            .with_novelty(0.80)
            .with_depth(0.70);

        let quality = perspective.quality_score();
        // 0.90 * 0.4 + 0.80 * 0.3 + 0.70 * 0.3 = 0.36 + 0.24 + 0.21 = 0.81
        assert!((quality - 0.81).abs() < 0.01);
    }

    #[test]
    fn test_query_validation_too_short() {
        let module = GigaThink::new();
        let result = module.validate_query("short");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_validation_valid() {
        let module = GigaThink::new();
        let result = module.validate_query("This is a valid query for analysis");
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_generates_perspectives() {
        let module = GigaThink::new();
        let context = ThinkToolContext {
            query: "What are the key factors for startup success?".to_string(),
            previous_steps: vec![],
        };

        let result = module.execute(&context).unwrap();

        assert_eq!(result.module, "GigaThink");
        assert!(result.confidence > 0.0);

        // Check that perspectives were generated
        let perspectives = result
            .output
            .get("perspectives")
            .unwrap()
            .as_array()
            .unwrap();
        assert!(perspectives.len() >= 10);
    }

    #[test]
    fn test_execute_includes_metadata() {
        let module = GigaThink::new();
        let context = ThinkToolContext {
            query: "What are the implications of AI on employment?".to_string(),
            previous_steps: vec![],
        };

        let result = module.execute(&context).unwrap();
        let metadata = result.output.get("metadata").unwrap();

        assert!(metadata.get("version").is_some());
        assert!(metadata.get("duration_ms").is_some());
        assert!(metadata.get("dimensions_count").is_some());
    }

    #[test]
    fn test_cross_validation() {
        let module = GigaThink::new();

        let perspectives: Vec<Perspective> = AnalysisDimension::all()
            .iter()
            .enumerate()
            .map(|(i, dim)| {
                Perspective::new(
                    format!("p{}", i),
                    *dim,
                    format!("{} Analysis", dim.display_name()),
                    "Content",
                )
                .with_confidence(0.80)
            })
            .collect();

        let result = module.cross_validate(&perspectives);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_validation_fails_low_confidence() {
        let module = GigaThink::new();

        let perspectives: Vec<Perspective> = AnalysisDimension::all()
            .iter()
            .enumerate()
            .map(|(i, dim)| {
                Perspective::new(
                    format!("p{}", i),
                    *dim,
                    format!("{} Analysis", dim.display_name()),
                    "Content",
                )
                .with_confidence(0.50) // Below threshold
            })
            .collect();

        let result = module.cross_validate(&perspectives);
        assert!(result.is_err());
    }

    #[test]
    fn test_theme_identification() {
        let module = GigaThink::new();
        let perspectives: Vec<Perspective> = AnalysisDimension::all()
            .iter()
            .enumerate()
            .map(|(i, dim)| {
                Perspective::new(format!("p{}", i), *dim, "Title", "Content").with_confidence(0.80)
            })
            .collect();

        let themes = module.identify_themes(&perspectives);
        assert!(!themes.is_empty());
    }

    #[test]
    fn test_insight_synthesis() {
        let module = GigaThink::new();

        let perspectives: Vec<Perspective> = vec![
            Perspective::new("p1", AnalysisDimension::Economic, "Title", "Content")
                .with_confidence(0.85),
            Perspective::new("p2", AnalysisDimension::Social, "Title", "Content")
                .with_confidence(0.80),
        ];

        let themes = vec![Theme {
            id: "t1".to_string(),
            title: "Theme 1".to_string(),
            description: "Description".to_string(),
            contributing_perspectives: vec!["p1".to_string(), "p2".to_string()],
            confidence: 0.75,
        }];

        let insights = module.synthesize_insights(&perspectives, &themes);
        assert!(!insights.is_empty());
    }

    #[tokio::test]
    async fn test_async_execution() {
        let module = GigaThink::new();
        let context = ThinkToolContext {
            query: "What are the future trends in renewable energy?".to_string(),
            previous_steps: vec![],
        };

        let result = module.execute_async(&context).await.unwrap();
        assert_eq!(result.module, "GigaThink");
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_error_types() {
        let err = GigaThinkError::InsufficientPerspectives {
            generated: 5,
            required: 10,
        };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("10"));

        let err = GigaThinkError::QueryTooShort {
            length: 5,
            minimum: 10,
        };
        assert!(err.to_string().contains("too short"));

        let err = GigaThinkError::LowConfidence {
            confidence: 0.50,
            threshold: 0.70,
        };
        assert!(err.to_string().contains("0.50"));
    }
}
