//! # Aesthetic System Types
//!
//! Core types for the Aesthetic Expression Mastery System.
//! Designed to support M2's VIBE Benchmark excellence across platforms.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// CONFIGURATION TYPES
// =============================================================================

/// Configuration for the Aesthetic Mastery Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AestheticConfig {
    /// Enable visual analysis
    pub enable_visual: bool,
    /// Enable usability assessment
    pub enable_usability: bool,
    /// Enable accessibility checks
    pub enable_accessibility: bool,
    /// Enable 3D rendering analysis
    pub enable_3d: bool,
    /// Enable cross-platform validation
    pub enable_cross_platform: bool,
    /// Enable performance analysis
    pub enable_performance: bool,
    /// WCAG compliance level target
    pub wcag_level: WcagLevel,
    /// Quality threshold (0.0-1.0)
    pub quality_threshold: f64,
    /// Maximum analysis time in milliseconds
    pub max_analysis_time_ms: u64,
}

impl Default for AestheticConfig {
    fn default() -> Self {
        Self {
            enable_visual: true,
            enable_usability: true,
            enable_accessibility: true,
            enable_3d: true,
            enable_cross_platform: true,
            enable_performance: true,
            wcag_level: WcagLevel::AA,
            quality_threshold: 0.85,
            max_analysis_time_ms: 30_000,
        }
    }
}

/// WCAG Compliance Level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WcagLevel {
    A,
    #[default]
    AA,
    AAA,
}

// =============================================================================
// PLATFORM & INPUT TYPES
// =============================================================================

/// Target platform for design assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Platform {
    #[default]
    Web,
    Android,
    IOS,
    Desktop,
}

/// Design input for assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignInput {
    /// Raw design data (screenshot, DOM, or design file)
    pub data: DesignData,
    /// Target platform
    pub platform: Platform,
    /// Optional context/description
    pub context: Option<String>,
    /// Component type being assessed
    pub component_type: ComponentType,
    /// Design tokens/theme if available
    pub design_tokens: Option<DesignTokens>,
}

/// Types of design data that can be assessed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DesignData {
    /// Screenshot or image as base64
    Screenshot(String),
    /// DOM structure as HTML
    Html(String),
    /// CSS styles
    Css(String),
    /// Combined HTML + CSS
    HtmlCss { html: String, css: String },
    /// Figma JSON export
    FigmaExport(serde_json::Value),
    /// React component code
    ReactComponent(String),
    /// URL to fetch and analyze
    Url(String),
    /// Raw pixel data
    RawPixels {
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
}

/// Component type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ComponentType {
    /// Full page/screen
    Page,
    /// Navigation component
    Navigation,
    /// Button or CTA
    Button,
    /// Form elements
    Form,
    /// Card or container
    Card,
    /// Modal/dialog
    Modal,
    /// List or table
    List,
    /// Hero section
    Hero,
    /// Footer
    Footer,
    /// Custom component
    #[default]
    Custom,
}

/// Design tokens/theme configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DesignTokens {
    pub colors: ColorTokens,
    pub typography: TypographyTokens,
    pub spacing: SpacingTokens,
    pub borders: BorderTokens,
    pub shadows: ShadowTokens,
    pub animations: AnimationTokens,
}

/// Color design tokens
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColorTokens {
    pub primary: Option<String>,
    pub secondary: Option<String>,
    pub background: Option<String>,
    pub surface: Option<String>,
    pub text_primary: Option<String>,
    pub text_secondary: Option<String>,
    pub success: Option<String>,
    pub warning: Option<String>,
    pub error: Option<String>,
    pub custom: HashMap<String, String>,
}

/// Typography design tokens
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypographyTokens {
    pub font_family_primary: Option<String>,
    pub font_family_secondary: Option<String>,
    pub font_family_mono: Option<String>,
    pub font_sizes: HashMap<String, String>,
    pub line_heights: HashMap<String, f64>,
    pub font_weights: HashMap<String, u16>,
}

/// Spacing design tokens
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpacingTokens {
    pub base_unit: Option<f64>,
    pub scale: Vec<f64>,
    pub custom: HashMap<String, f64>,
}

/// Border design tokens
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BorderTokens {
    pub radius: HashMap<String, String>,
    pub widths: HashMap<String, String>,
    pub colors: HashMap<String, String>,
}

/// Shadow design tokens
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShadowTokens {
    pub shadows: HashMap<String, String>,
}

/// Animation design tokens
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnimationTokens {
    pub durations: HashMap<String, String>,
    pub easings: HashMap<String, String>,
}

// =============================================================================
// 3D DESIGN TYPES
// =============================================================================

/// Input for 3D design assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeDDesignInput {
    /// 3D framework used
    pub framework: ThreeDFramework,
    /// Scene data or code
    pub scene_data: ThreeDSceneData,
    /// Target performance metrics
    pub performance_targets: ThreeDPerformanceTargets,
    /// Platform target
    pub platform: Platform,
}

/// Supported 3D frameworks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreeDFramework {
    ReactThreeFiber,
    ThreeJs,
    BabylonJs,
    WebGL,
    WebGPU,
}

/// 3D scene data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreeDSceneData {
    /// React Three Fiber JSX code
    R3FCode(String),
    /// Three.js JavaScript code
    ThreeJsCode(String),
    /// GLTF/GLB file data
    GltfData(Vec<u8>),
    /// Scene graph JSON
    SceneGraph(serde_json::Value),
}

/// Performance targets for 3D rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeDPerformanceTargets {
    /// Target FPS
    pub target_fps: u32,
    /// Max polygon count
    pub max_polygons: u64,
    /// Max texture memory MB
    pub max_texture_memory_mb: u32,
    /// Max draw calls
    pub max_draw_calls: u32,
}

impl Default for ThreeDPerformanceTargets {
    fn default() -> Self {
        Self {
            target_fps: 60,
            max_polygons: 1_000_000,
            max_texture_memory_mb: 512,
            max_draw_calls: 100,
        }
    }
}

// =============================================================================
// ASSESSMENT RESULT TYPES
// =============================================================================

/// Comprehensive design assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignAssessmentResult {
    /// Overall aesthetic score (0.0-1.0)
    pub overall_score: f64,
    /// VIBE benchmark compliance
    pub vibe_compliance: VibeComplianceResult,
    /// Visual assessment results
    pub visual: VisualAssessmentResult,
    /// Usability assessment results
    pub usability: UsabilityAssessmentResult,
    /// Accessibility assessment results
    pub accessibility: AccessibilityResult,
    /// Cross-platform validation results
    pub cross_platform: CrossPlatformResult,
    /// Performance impact analysis
    pub performance: PerformanceImpactResult,
    /// Improvement recommendations
    pub recommendations: Vec<DesignRecommendation>,
    /// Assessment metadata
    pub metadata: AssessmentMetadata,
}

/// VIBE Benchmark compliance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeComplianceResult {
    /// Platform-specific scores
    pub platform_scores: HashMap<Platform, f64>,
    /// Overall VIBE score
    pub overall_score: f64,
    /// Passes VIBE threshold
    pub passes_threshold: bool,
    /// Details per criterion
    pub criteria_results: Vec<VibeCriterionResult>,
}

/// Individual VIBE criterion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeCriterionResult {
    pub criterion: String,
    pub score: f64,
    pub weight: f64,
    pub details: String,
}

/// Visual assessment result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VisualAssessmentResult {
    /// Overall visual score
    pub score: f64,
    /// Color harmony analysis
    pub color_harmony: ColorHarmonyResult,
    /// Typography assessment
    pub typography: TypographyResult,
    /// Layout analysis
    pub layout: LayoutResult,
    /// Visual hierarchy evaluation
    pub hierarchy: HierarchyResult,
    /// Consistency check
    pub consistency: ConsistencyResult,
    /// White space analysis
    pub white_space: WhiteSpaceResult,
}

/// Color harmony analysis result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColorHarmonyResult {
    pub score: f64,
    pub harmony_type: Option<ColorHarmonyType>,
    pub contrast_ratios: Vec<ContrastRatio>,
    pub palette_coherence: f64,
    pub issues: Vec<ColorIssue>,
}

/// Color harmony types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorHarmonyType {
    Complementary,
    Analogous,
    Triadic,
    SplitComplementary,
    Tetradic,
    Monochromatic,
    Custom,
}

/// Contrast ratio between two colors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastRatio {
    pub foreground: String,
    pub background: String,
    pub ratio: f64,
    pub passes_aa: bool,
    pub passes_aaa: bool,
    pub passes_aa_large: bool,
}

/// Color-related issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorIssue {
    pub severity: IssueSeverity,
    pub description: String,
    pub element: Option<String>,
    pub suggestion: String,
}

/// Typography assessment result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypographyResult {
    pub score: f64,
    pub font_pairing_score: f64,
    pub readability_score: f64,
    pub hierarchy_score: f64,
    pub line_height_score: f64,
    pub issues: Vec<TypographyIssue>,
}

/// Typography issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypographyIssue {
    pub severity: IssueSeverity,
    pub issue_type: TypographyIssueType,
    pub description: String,
    pub suggestion: String,
}

/// Types of typography issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypographyIssueType {
    TooManyFonts,
    PoorContrast,
    LineLengthTooLong,
    LineLengthTooShort,
    LineHeightTooTight,
    LineHeightTooLoose,
    FontSizeTooSmall,
    InconsistentScale,
}

/// Layout analysis result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayoutResult {
    pub score: f64,
    pub grid_adherence: f64,
    pub alignment_score: f64,
    pub balance_score: f64,
    pub responsive_score: f64,
    pub issues: Vec<LayoutIssue>,
}

/// Layout issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutIssue {
    pub severity: IssueSeverity,
    pub description: String,
    pub affected_area: String,
    pub suggestion: String,
}

/// Visual hierarchy result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HierarchyResult {
    pub score: f64,
    pub focal_point_clarity: f64,
    pub information_flow: f64,
    pub cta_prominence: f64,
    pub issues: Vec<HierarchyIssue>,
}

/// Hierarchy issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyIssue {
    pub severity: IssueSeverity,
    pub description: String,
    pub suggestion: String,
}

/// Consistency check result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsistencyResult {
    pub score: f64,
    pub style_consistency: f64,
    pub spacing_consistency: f64,
    pub component_consistency: f64,
    pub issues: Vec<ConsistencyIssue>,
}

/// Consistency issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyIssue {
    pub severity: IssueSeverity,
    pub issue_type: String,
    pub description: String,
    pub elements: Vec<String>,
}

/// White space analysis result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhiteSpaceResult {
    pub score: f64,
    pub breathing_room: f64,
    pub density_balance: f64,
    pub margin_consistency: f64,
}

/// Usability assessment result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsabilityAssessmentResult {
    pub score: f64,
    pub interaction_clarity: f64,
    pub navigation_ease: f64,
    pub feedback_quality: f64,
    pub error_prevention: f64,
    pub cognitive_load: f64,
    pub issues: Vec<UsabilityIssue>,
}

/// Usability issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsabilityIssue {
    pub severity: IssueSeverity,
    pub heuristic: UsabilityHeuristic,
    pub description: String,
    pub suggestion: String,
}

/// Nielsen's usability heuristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsabilityHeuristic {
    VisibilityOfSystemStatus,
    MatchSystemRealWorld,
    UserControlFreedom,
    ConsistencyStandards,
    ErrorPrevention,
    RecognitionNotRecall,
    FlexibilityEfficiency,
    AestheticMinimalist,
    RecoverFromErrors,
    HelpDocumentation,
}

/// Accessibility assessment result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccessibilityResult {
    pub score: f64,
    pub wcag_level_achieved: Option<WcagLevel>,
    pub pass_criteria: Vec<WcagCriterion>,
    pub fail_criteria: Vec<WcagCriterion>,
    pub issues: Vec<AccessibilityIssue>,
}

/// WCAG criterion evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WcagCriterion {
    pub id: String,
    pub name: String,
    pub level: WcagLevel,
    pub passed: bool,
    pub details: String,
}

/// Accessibility issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityIssue {
    pub severity: IssueSeverity,
    pub wcag_criterion: String,
    pub description: String,
    pub element: Option<String>,
    pub suggestion: String,
    pub impact: AccessibilityImpact,
}

/// Accessibility impact level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessibilityImpact {
    Critical,
    Serious,
    Moderate,
    Minor,
}

/// Cross-platform validation result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossPlatformResult {
    pub score: f64,
    pub platform_results: HashMap<Platform, PlatformSpecificResult>,
    pub consistency_score: f64,
    pub issues: Vec<CrossPlatformIssue>,
}

/// Platform-specific result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformSpecificResult {
    pub platform: Platform,
    pub compliance_score: f64,
    pub design_guideline_adherence: f64,
    pub platform_conventions_score: f64,
    pub issues: Vec<String>,
}

/// Cross-platform issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformIssue {
    pub severity: IssueSeverity,
    pub platforms_affected: Vec<Platform>,
    pub description: String,
    pub suggestion: String,
}

/// 3D assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeDAssessmentResult {
    pub score: f64,
    pub visual_quality: f64,
    pub performance_score: f64,
    pub interaction_quality: f64,
    pub r3f_instance_count: Option<u64>,
    pub polygon_count: u64,
    pub texture_memory_mb: f64,
    pub draw_calls: u32,
    pub estimated_fps: u32,
    pub issues: Vec<ThreeDIssue>,
    pub optimizations: Vec<ThreeDOptimization>,
}

/// 3D-specific issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeDIssue {
    pub severity: IssueSeverity,
    pub category: ThreeDIssueCategory,
    pub description: String,
    pub suggestion: String,
}

/// 3D issue categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreeDIssueCategory {
    Performance,
    Visual,
    Interaction,
    Memory,
    Compatibility,
}

/// 3D optimization suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeDOptimization {
    pub category: String,
    pub current_value: String,
    pub suggested_value: String,
    pub expected_improvement: String,
    pub priority: OptimizationPriority,
}

/// Optimization priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Performance impact analysis result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceImpactResult {
    pub score: f64,
    pub estimated_load_time_ms: u64,
    pub estimated_interaction_delay_ms: u64,
    pub render_complexity: RenderComplexity,
    pub resource_usage: ResourceUsage,
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Render complexity assessment
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderComplexity {
    pub dom_depth: u32,
    pub dom_node_count: u32,
    pub css_complexity_score: f64,
    pub animation_count: u32,
    pub reflow_risk: f64,
}

/// Resource usage assessment
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub estimated_css_size_kb: f64,
    pub estimated_js_size_kb: f64,
    pub image_optimization_score: f64,
    pub font_loading_strategy: String,
}

/// Performance recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    pub category: String,
    pub description: String,
    pub impact: String,
    pub priority: OptimizationPriority,
}

/// Design improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignRecommendation {
    pub category: RecommendationCategory,
    pub priority: OptimizationPriority,
    pub title: String,
    pub description: String,
    pub expected_impact: String,
    pub implementation_difficulty: DifficultyLevel,
    pub before_after: Option<BeforeAfter>,
}

/// Recommendation categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Visual,
    Usability,
    Accessibility,
    Performance,
    CrossPlatform,
    ThreeD,
}

/// Difficulty level for implementation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Trivial,
    Easy,
    Medium,
    Hard,
    Complex,
}

/// Before/after comparison for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeforeAfter {
    pub before: String,
    pub after: String,
    pub diff_type: String,
}

/// Assessment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentMetadata {
    pub assessment_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub duration_ms: u64,
    pub engine_version: String,
    pub config_hash: String,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum IssueSeverity {
    Critical,
    Major,
    Minor,
    #[default]
    Info,
}

// =============================================================================
// BRAND-ALIGNED DEFAULTS
// =============================================================================

/// ReasonKit brand-aligned color tokens
/// Reference: reasonkit-core/src/constants/brand.rs
impl ColorTokens {
    /// Create ReasonKit brand-aligned color tokens
    pub fn reasonkit_brand() -> Self {
        use crate::constants::brand::colors;

        let mut custom = HashMap::new();
        custom.insert("cyan".to_string(), colors::CYAN.to_string());
        custom.insert("purple".to_string(), colors::PURPLE.to_string());
        custom.insert("pink".to_string(), colors::PINK.to_string());
        custom.insert("orange".to_string(), colors::ORANGE.to_string());
        custom.insert("yellow".to_string(), colors::YELLOW.to_string());

        Self {
            primary: Some(colors::PRIMARY.to_string()),
            secondary: Some(colors::SECONDARY.to_string()),
            background: Some(colors::BACKGROUND.to_string()),
            surface: Some(colors::SURFACE.to_string()),
            text_primary: Some(colors::TEXT_PRIMARY.to_string()),
            text_secondary: Some(colors::TEXT_SECONDARY.to_string()),
            success: Some(colors::SUCCESS.to_string()),
            warning: Some(colors::WARNING.to_string()),
            error: Some(colors::ERROR.to_string()),
            custom,
        }
    }
}

impl TypographyTokens {
    /// Create ReasonKit brand-aligned typography tokens
    pub fn reasonkit_brand() -> Self {
        use crate::constants::brand::typography;

        let mut font_sizes = HashMap::new();
        font_sizes.insert("xs".to_string(), "0.75rem".to_string());
        font_sizes.insert("sm".to_string(), "0.875rem".to_string());
        font_sizes.insert("base".to_string(), "1rem".to_string());
        font_sizes.insert("lg".to_string(), "1.125rem".to_string());
        font_sizes.insert("xl".to_string(), "1.25rem".to_string());
        font_sizes.insert("2xl".to_string(), "1.5rem".to_string());
        font_sizes.insert("3xl".to_string(), "1.875rem".to_string());
        font_sizes.insert("4xl".to_string(), "2.25rem".to_string());

        let mut line_heights = HashMap::new();
        line_heights.insert("tight".to_string(), 1.25);
        line_heights.insert("normal".to_string(), 1.5);
        line_heights.insert("relaxed".to_string(), 1.75);

        let mut font_weights = HashMap::new();
        font_weights.insert("normal".to_string(), 400);
        font_weights.insert("medium".to_string(), 500);
        font_weights.insert("semibold".to_string(), 600);
        font_weights.insert("bold".to_string(), 700);

        Self {
            font_family_primary: Some(typography::FONT_BODY.to_string()),
            font_family_secondary: Some(typography::FONT_DISPLAY.to_string()),
            font_family_mono: Some(typography::FONT_CODE.to_string()),
            font_sizes,
            line_heights,
            font_weights,
        }
    }
}
