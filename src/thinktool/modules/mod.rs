//! ThinkTool Module Type Definitions
//!
//! This module provides the core type definitions and traits for ThinkTool modules,
//! which are structured reasoning components that implement specific analytical strategies.
//!
//! ## Architecture Note
//!
//! ThinkTool modules can be used in two ways:
//!
//! 1. **Direct Module Execution** - Use the module structs directly for local processing
//! 2. **Protocol Execution** - Use ProtocolExecutor with LLM integration
//!
//! ## Direct Module Usage
//!
//! ```rust,ignore
//! use reasonkit_core::thinktool::modules::{GigaThink, ThinkToolModule, ThinkToolContext};
//!
//! let module = GigaThink::new();
//! let context = ThinkToolContext {
//!     query: "What are the key factors for success?".to_string(),
//!     previous_steps: vec![],
//! };
//!
//! // Synchronous execution
//! let result = module.execute(&context)?;
//!
//! // Asynchronous execution (requires AsyncThinkToolModule)
//! let async_result = module.execute_async(&context).await?;
//! ```
//!
//! ## Protocol Execution (with LLM)
//!
//! ```rust,ignore
//! let executor = ProtocolExecutor::new()?;
//! let result = executor.execute("gigathink", ProtocolInput::query("question")).await?;
//! ```
//!
//! ## Available Modules
//!
//! | Module | Code | Purpose | Key Feature |
//! |--------|------|---------|-------------|
//! | `GigaThink` | `gt` | Expansive creative thinking | 10+ perspectives |
//! | `LaserLogic` | `ll` | Precision deductive reasoning | Fallacy detection |
//! | `BedRock` | `br` | First principles decomposition | Core axiom extraction |
//! | `ProofGuard` | `pg` | Multi-source verification | 3+ sources required |
//! | `BrutalHonesty` | `bh` | Adversarial self-critique | Skeptical scoring |
//! | `BrutalHonestyEnhanced` | `bhe` | Deep adversarial critique | Cognitive bias detection |
//!
//! See `registry.rs` for full protocol definitions.

use crate::error::Result;
use serde::{Deserialize, Serialize};

// ============================================================================
// MODULE RE-EXPORTS
// ============================================================================

pub mod bedrock;
pub mod brutalhonesty;
pub mod brutalhonesty_enhanced;
pub mod gigathink;
pub mod laserlogic;
pub mod proofguard;

// Re-export module structs
pub use bedrock::BedRock;
pub use brutalhonesty::BrutalHonesty;
pub use brutalhonesty_enhanced::BrutalHonestyEnhanced;
pub use gigathink::GigaThink;
pub use laserlogic::LaserLogic;
pub use proofguard::ProofGuard;

// Re-export GigaThink types for comprehensive access
pub use gigathink::{
    AnalysisDimension, AsyncThinkToolModule, GigaThinkBuilder, GigaThinkConfig, GigaThinkError,
    GigaThinkMetadata, GigaThinkResult, Perspective, SynthesizedInsight, Theme,
};

// Re-export LaserLogic types for comprehensive access
pub use laserlogic::{
    Argument, ArgumentForm, Contradiction, ContradictionType, DetectedFallacy, Fallacy,
    LaserLogicConfig, LaserLogicResult, Premise, PremiseType, SoundnessStatus, ValidityStatus,
};

// Re-export BrutalHonesty types for comprehensive access
pub use brutalhonesty::{
    BrutalHonestyBuilder, BrutalHonestyConfig, CritiqueSeverity, CritiqueVerdict, DetectedFlaw,
    FlawCategory, FlawSeverity, IdentifiedStrength, ImplicitAssumption,
};

// Re-export BrutalHonestyEnhanced types
pub use brutalhonesty_enhanced::{
    ArgumentMap, BiasCategory, CognitiveBias, CognitiveBiasDepth, CulturalAssumption,
    EnhancedBuilder, EnhancedConfig, SteelmanArgument,
};

// ============================================================================
// CORE TYPE DEFINITIONS
// ============================================================================

/// Configuration for a ThinkTool module
///
/// Defines the metadata and behavior parameters for a reasoning module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkToolModuleConfig {
    /// Module name (e.g., "GigaThink", "LaserLogic")
    pub name: String,

    /// Semantic version (e.g., "2.1.0")
    pub version: String,

    /// Human-readable description of module purpose
    pub description: String,

    /// Weight applied to this module's confidence in composite calculations
    /// Range: 0.0 - 1.0, typical: 0.10 - 0.30
    pub confidence_weight: f64,
}

impl ThinkToolModuleConfig {
    /// Create a new module configuration
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: description.into(),
            confidence_weight: 0.20, // Default weight
        }
    }

    /// Builder: set confidence weight
    pub fn with_confidence_weight(mut self, weight: f64) -> Self {
        self.confidence_weight = weight.clamp(0.0, 1.0);
        self
    }
}

/// Context provided to a ThinkTool module for execution
///
/// Contains the query to analyze and any context from previous reasoning steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkToolContext {
    /// The primary query or problem to analyze
    pub query: String,

    /// Results from previous reasoning steps (for chained execution)
    pub previous_steps: Vec<String>,
}

impl ThinkToolContext {
    /// Create a new context with just a query
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            previous_steps: Vec::new(),
        }
    }

    /// Create a context with previous step results
    pub fn with_previous_steps(query: impl Into<String>, steps: Vec<String>) -> Self {
        Self {
            query: query.into(),
            previous_steps: steps,
        }
    }

    /// Add a previous step result
    pub fn add_previous_step(&mut self, step: impl Into<String>) {
        self.previous_steps.push(step.into());
    }

    /// Check if this context has previous steps
    pub fn has_previous_steps(&self) -> bool {
        !self.previous_steps.is_empty()
    }

    /// Get the number of previous steps
    pub fn previous_step_count(&self) -> usize {
        self.previous_steps.len()
    }
}

/// Output produced by a ThinkTool module
///
/// Contains the module identification, confidence score, and structured output data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkToolOutput {
    /// Name of the module that produced this output
    pub module: String,

    /// Confidence score for the output (0.0 - 1.0)
    pub confidence: f64,

    /// Structured output data (module-specific format)
    pub output: serde_json::Value,
}

impl ThinkToolOutput {
    /// Create a new output
    pub fn new(module: impl Into<String>, confidence: f64, output: serde_json::Value) -> Self {
        Self {
            module: module.into(),
            confidence: confidence.clamp(0.0, 1.0),
            output,
        }
    }

    /// Get a field from the output
    pub fn get(&self, field: &str) -> Option<&serde_json::Value> {
        self.output.get(field)
    }

    /// Get a string field from the output
    pub fn get_str(&self, field: &str) -> Option<&str> {
        self.output.get(field).and_then(|v| v.as_str())
    }

    /// Get an array field from the output
    pub fn get_array(&self, field: &str) -> Option<&Vec<serde_json::Value>> {
        self.output.get(field).and_then(|v| v.as_array())
    }

    /// Check if the output has high confidence (>= 0.80)
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.80
    }

    /// Check if the output meets a confidence threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

// ============================================================================
// CORE TRAIT DEFINITIONS
// ============================================================================

/// Core trait for ThinkTool modules
///
/// All ThinkTool modules must implement this trait to provide
/// synchronous execution capability and configuration access.
///
/// For async execution, also implement `AsyncThinkToolModule`.
pub trait ThinkToolModule: Send + Sync {
    /// Get the module configuration
    fn config(&self) -> &ThinkToolModuleConfig;

    /// Execute the module synchronously
    ///
    /// # Arguments
    /// * `context` - The execution context containing query and previous steps
    ///
    /// # Returns
    /// * `Ok(ThinkToolOutput)` - Successful execution with output and confidence
    /// * `Err(Error)` - Execution failed with error details
    fn execute(&self, context: &ThinkToolContext) -> Result<ThinkToolOutput>;

    /// Get the module name (convenience method)
    fn name(&self) -> &str {
        &self.config().name
    }

    /// Get the module version (convenience method)
    fn version(&self) -> &str {
        &self.config().version
    }

    /// Get the module description (convenience method)
    fn description(&self) -> &str {
        &self.config().description
    }

    /// Get the confidence weight for this module
    fn confidence_weight(&self) -> f64 {
        self.config().confidence_weight
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_config_creation() {
        let config = ThinkToolModuleConfig::new("TestModule", "1.0.0", "A test module");
        assert_eq!(config.name, "TestModule");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.confidence_weight, 0.20);
    }

    #[test]
    fn test_module_config_with_weight() {
        let config = ThinkToolModuleConfig::new("TestModule", "1.0.0", "A test module")
            .with_confidence_weight(0.35);
        assert_eq!(config.confidence_weight, 0.35);
    }

    #[test]
    fn test_context_creation() {
        let context = ThinkToolContext::new("Test query");
        assert_eq!(context.query, "Test query");
        assert!(context.previous_steps.is_empty());
    }

    #[test]
    fn test_context_with_previous_steps() {
        let context =
            ThinkToolContext::with_previous_steps("Query", vec!["Step 1".into(), "Step 2".into()]);
        assert!(context.has_previous_steps());
        assert_eq!(context.previous_step_count(), 2);
    }

    #[test]
    fn test_output_creation() {
        let output =
            ThinkToolOutput::new("TestModule", 0.85, serde_json::json!({"result": "success"}));
        assert_eq!(output.module, "TestModule");
        assert_eq!(output.confidence, 0.85);
        assert!(output.is_high_confidence());
    }

    #[test]
    fn test_output_threshold() {
        let output =
            ThinkToolOutput::new("TestModule", 0.75, serde_json::json!({"result": "success"}));
        assert!(output.meets_threshold(0.70));
        assert!(!output.meets_threshold(0.80));
    }

    #[test]
    fn test_output_field_access() {
        let output = ThinkToolOutput::new(
            "TestModule",
            0.85,
            serde_json::json!({
                "name": "test",
                "values": [1, 2, 3]
            }),
        );

        assert_eq!(output.get_str("name"), Some("test"));
        assert!(output.get_array("values").is_some());
    }

    #[test]
    fn test_gigathink_module() {
        let module = GigaThink::new();
        assert_eq!(module.name(), "GigaThink");
        assert_eq!(module.version(), "2.1.0");
    }

    #[test]
    fn test_gigathink_execution() {
        let module = GigaThink::new();
        let context = ThinkToolContext::new("What are the implications of AI adoption?");

        let result = module.execute(&context).unwrap();
        assert_eq!(result.module, "GigaThink");
        assert!(result.confidence > 0.0);

        // Verify 10+ perspectives
        let perspectives = result.get_array("perspectives").unwrap();
        assert!(perspectives.len() >= 10);
    }

    #[test]
    fn test_laserlogic_module() {
        let module = LaserLogic::new();
        assert_eq!(module.name(), "LaserLogic");
        assert_eq!(module.version(), "3.0.0");
    }

    #[test]
    fn test_laserlogic_analyze_argument() {
        let module = LaserLogic::new();
        let result = module
            .analyze_argument(
                &["All humans are mortal", "Socrates is human"],
                "Socrates is mortal",
            )
            .unwrap();

        // Should detect categorical syllogism
        assert_eq!(
            result.argument_form,
            Some(ArgumentForm::CategoricalSyllogism)
        );
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_laserlogic_fallacy_detection() {
        let module = LaserLogic::new();
        let result = module
            .analyze_argument(
                &["If it rains, then the ground is wet", "The ground is wet"],
                "It rained",
            )
            .unwrap();

        // Should detect affirming the consequent
        assert!(result.has_fallacies());
        assert!(result
            .fallacies
            .iter()
            .any(|f| f.fallacy == Fallacy::AffirmingConsequent));
    }

    #[test]
    fn test_brutalhonesty_module() {
        let module = BrutalHonesty::new();
        assert_eq!(module.name(), "BrutalHonesty");
        assert_eq!(module.version(), "3.0.0");
    }

    #[test]
    fn test_brutalhonesty_execution() {
        let module = BrutalHonesty::new();
        let context =
            ThinkToolContext::new("Our startup will succeed because we have the best team");

        let result = module.execute(&context).unwrap();
        assert_eq!(result.module, "BrutalHonesty");
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 0.95);

        // Verify output contains required fields
        assert!(result.get("verdict").is_some());
        assert!(result.get("analysis").is_some());
        assert!(result.get("devils_advocate").is_some());
    }

    #[test]
    fn test_brutalhonesty_enhanced_module() {
        let module = BrutalHonestyEnhanced::new();
        assert_eq!(module.name(), "BrutalHonestyEnhanced");
        assert_eq!(module.version(), "3.0.0");
    }

    #[test]
    fn test_brutalhonesty_enhanced_execution() {
        let module = BrutalHonestyEnhanced::new();
        let context = ThinkToolContext::new(
            "We're certain this will succeed because everyone agrees it's the best approach.",
        );

        let result = module.execute(&context).unwrap();
        assert_eq!(result.module, "BrutalHonestyEnhanced");
        assert!(result.confidence > 0.0);
        assert!(result.confidence <= 0.90);

        // Verify output contains enhanced analysis
        assert!(result.get("enhanced_analysis").is_some());
        assert!(result.get("base_analysis").is_some());
    }

    #[test]
    fn test_brutalhonesty_builder() {
        let module = BrutalHonesty::builder()
            .severity(CritiqueSeverity::Ruthless)
            .enable_devil_advocate(true)
            .build();

        assert_eq!(module.brutal_config().severity, CritiqueSeverity::Ruthless);
        assert!(module.brutal_config().enable_devil_advocate);
    }

    #[test]
    fn test_brutalhonesty_enhanced_builder() {
        let module = BrutalHonestyEnhanced::builder()
            .severity(CritiqueSeverity::Harsh)
            .cognitive_bias_depth(CognitiveBiasDepth::Deep)
            .enable_cultural_analysis(true)
            .build();

        assert_eq!(
            module.enhanced_config().base_config.severity,
            CritiqueSeverity::Harsh
        );
        assert_eq!(
            module.enhanced_config().cognitive_bias_depth,
            CognitiveBiasDepth::Deep
        );
    }
}
