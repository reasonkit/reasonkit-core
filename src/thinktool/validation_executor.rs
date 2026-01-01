//! DeepSeek Protocol Validation Engine - Protocol Integration
//!
//! **Built-in protocol integration for enterprise-grade reasoning chain validation**
//!
//! This module extends the existing ProtocolExecutor with DeepSeek-powered
//! validation capabilities that validate complete reasoning chains with 671B scale expertise.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::{
    executor::{ExecutorConfig, ProtocolExecutor, ProtocolInput, ProtocolOutput},
    protocol::Protocol,
    validation::{DeepSeekValidationEngine, DeepSeekValidationResult, ValidationVerdict},
};

/// Enhanced protocol executor with DeepSeek validation
pub struct ValidatingProtocolExecutor {
    base_executor: ProtocolExecutor,
    validation_engine: Option<DeepSeekValidationEngine>,
    validation_config: ValidationExecutorConfig,
}

/// Configuration for validation-enhanced execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationExecutorConfig {
    /// Enable DeepSeek validation
    #[serde(default)]
    pub enable_validation: bool,

    /// Validation level
    #[serde(default)]
    pub validation_level: ValidationLevel,

    /// Minimum chain confidence threshold for validation
    #[serde(default = "default_min_confidence_threshold")]
    pub min_confidence_threshold: f64,

    /// Enable validation for specific protocols only
    #[serde(default)]
    pub validate_protocols: Vec<String>,

    /// Skip validation for specific protocols
    #[serde(default)]
    pub skip_protocols: Vec<String>,
}

fn default_min_confidence_threshold() -> f64 {
    0.70
}

/// Validation levels
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationLevel {
    None,
    Quick,
    #[default]
    Standard,
    Rigorous,
    Paranoid,
}

impl Default for ValidationExecutorConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            min_confidence_threshold: default_min_confidence_threshold(),
            validate_protocols: Vec::new(),
            skip_protocols: Vec::new(),
        }
    }
}

impl ValidatingProtocolExecutor {
    /// Create a new validating executor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_configs(
            ExecutorConfig::default(),
            ValidationExecutorConfig::default(),
        )
    }

    /// Create with custom configurations
    pub fn with_configs(
        executor_config: ExecutorConfig,
        validation_config: ValidationExecutorConfig,
    ) -> Result<Self> {
        let base_executor = ProtocolExecutor::with_config(executor_config)?;

        let validation_engine = if validation_config.enable_validation {
            Some(DeepSeekValidationEngine::new()?)
        } else {
            None
        };

        Ok(Self {
            base_executor,
            validation_engine,
            validation_config,
        })
    }

    /// Execute protocol with DeepSeek validation
    pub async fn execute_with_validation(
        &self,
        protocol_id: &str,
        input: ProtocolInput,
    ) -> Result<ProtocolOutput> {
        // First, execute the protocol normally
        let base_result = self
            .base_executor
            .execute(protocol_id, input.clone())
            .await?;

        // Check if validation should be applied
        if !self.should_validate(protocol_id, &base_result) {
            return Ok(base_result);
        }

        // Apply DeepSeek validation
        let validation_result = self
            .apply_validation(protocol_id, &input, &base_result)
            .await?;

        // Integrate validation results
        self.merge_validation_results(base_result, validation_result)
    }

    /// Execute profile with DeepSeek validation
    pub async fn execute_profile_with_validation(
        &self,
        profile_id: &str,
        input: ProtocolInput,
    ) -> Result<ProtocolOutput> {
        // Execute profile normally
        let base_result = self
            .base_executor
            .execute_profile(profile_id, input.clone())
            .await?;

        // Check if validation should be applied
        if !self.should_validate_profile(profile_id, &base_result) {
            return Ok(base_result);
        }

        // Apply DeepSeek validation
        let validation_result = self
            .apply_validation(profile_id, &input, &base_result)
            .await?;

        // Integrate validation results
        self.merge_validation_results(base_result, validation_result)
    }

    /// Check if validation should be applied to a protocol
    fn should_validate(&self, protocol_id: &str, result: &ProtocolOutput) -> bool {
        if !self.validation_config.enable_validation {
            return false;
        }

        if self.validation_config.validation_level == ValidationLevel::None {
            return false;
        }

        // Check protocol-specific settings
        if !self.validation_config.validate_protocols.is_empty()
            && !self
                .validation_config
                .validate_protocols
                .contains(&protocol_id.to_string())
        {
            return false;
        }

        if self
            .validation_config
            .skip_protocols
            .contains(&protocol_id.to_string())
        {
            return false;
        }

        // Check confidence threshold
        if result.confidence < self.validation_config.min_confidence_threshold {
            return false; // Too low confidence - focus on improvement
        }

        true
    }

    /// Check if validation should be applied to a profile
    fn should_validate_profile(&self, profile_id: &str, result: &ProtocolOutput) -> bool {
        if !self.validation_config.enable_validation {
            return false;
        }

        // Profiles might have different validation criteria
        match profile_id {
            "paranoid" | "deep" => true, // Always validate high-stakes profiles
            "quick" => self.validation_config.validation_level != ValidationLevel::None,
            _ => result.confidence >= self.validation_config.min_confidence_threshold,
        }
    }

    /// Apply DeepSeek validation to protocol results
    async fn apply_validation(
        &self,
        _target_id: &str,
        original_input: &ProtocolInput,
        protocol_result: &ProtocolOutput,
    ) -> Result<DeepSeekValidationResult> {
        let validation_engine = self
            .validation_engine
            .as_ref()
            .ok_or_else(|| Error::Config("Validation engine not available".into()))?;

        match self.validation_config.validation_level {
            ValidationLevel::Quick => {
                validation_engine
                    .validate_quick(protocol_result, original_input)
                    .await
            }
            ValidationLevel::Standard => {
                // Use default trace (empty for now - could be enhanced)
                validation_engine
                    .validate_chain(protocol_result, original_input, &Default::default())
                    .await
            }
            ValidationLevel::Rigorous => {
                validation_engine
                    .validate_rigorous(protocol_result, original_input, &Default::default())
                    .await
            }
            ValidationLevel::Paranoid => {
                // Paranoid level applies all validation techniques
                validation_engine
                    .validate_with_statistical_significance(
                        protocol_result,
                        original_input,
                        &Default::default(),
                    )
                    .await
            }
            ValidationLevel::None => {
                unreachable!("ValidationLevel::None should be filtered earlier")
            }
        }
    }

    /// Merge validation results into protocol output
    fn merge_validation_results(
        &self,
        mut base_result: ProtocolOutput,
        validation_result: DeepSeekValidationResult,
    ) -> Result<ProtocolOutput> {
        // Add validation metadata to output
        base_result.data.insert(
            "deepseek_validation".into(),
            serde_json::to_value(&validation_result)?,
        );

        // Adjust overall confidence based on validation results
        let validation_impact = match validation_result.verdict {
            ValidationVerdict::Validated => 1.10, // Boost confidence for validated results
            ValidationVerdict::PartiallyValidated => 1.00, // Neutral
            ValidationVerdict::NeedsImprovement => 0.85, // Moderate penalty
            ValidationVerdict::Invalid => 0.60,   // Significant penalty
            ValidationVerdict::CriticalIssues => 0.30, // Severe penalty
        };

        base_result.confidence = (base_result.confidence * validation_impact).clamp(0.0, 1.0);

        // Add validation performance metrics
        let tokens_used = super::step::TokenUsage {
            input_tokens: validation_result.tokens_used.input_tokens,
            output_tokens: validation_result.tokens_used.output_tokens,
            total_tokens: validation_result.tokens_used.total_tokens,
            cost_usd: validation_result.tokens_used.cost_usd,
        };
        base_result.tokens.add(&tokens_used);
        base_result.duration_ms += validation_result.performance.duration_ms;

        Ok(base_result)
    }

    /// Delegate to base executor methods
    pub fn list_protocols(&self) -> Vec<&str> {
        self.base_executor.list_protocols()
    }

    pub fn list_profiles(&self) -> Vec<&str> {
        self.base_executor.list_profiles()
    }

    pub fn get_protocol(&self, id: &str) -> Option<&Protocol> {
        self.base_executor.get_protocol(id)
    }

    pub fn get_profile(&self, id: &str) -> Option<super::profiles::ReasoningProfile> {
        self.base_executor.get_profile(id).cloned()
    }
}

impl Default for ValidatingProtocolExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default validating executor")
    }
}

/// Configuration helpers for common validation scenarios
impl ValidationExecutorConfig {
    /// Create configuration for enterprise deployment
    pub fn enterprise() -> Self {
        Self {
            enable_validation: true,
            validation_level: ValidationLevel::Rigorous,
            min_confidence_threshold: 0.80,
            validate_protocols: vec![
                "proofguard".into(),
                "brutalhonesty".into(),
                "laserlogic".into(),
            ],
            skip_protocols: Vec::new(),
        }
    }

    /// Create configuration for research applications
    pub fn research() -> Self {
        Self {
            enable_validation: true,
            validation_level: ValidationLevel::Paranoid,
            min_confidence_threshold: 0.90,
            validate_protocols: vec!["gigathink".into(), "scientific".into()],
            skip_protocols: Vec::new(),
        }
    }

    /// Create configuration for production applications
    pub fn production() -> Self {
        Self {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            min_confidence_threshold: 0.70,
            validate_protocols: Vec::new(),
            skip_protocols: vec!["quick".into()],
        }
    }

    /// Create configuration for compliance applications
    pub fn compliance() -> Self {
        Self {
            enable_validation: true,
            validation_level: ValidationLevel::Rigorous,
            min_confidence_threshold: 0.85,
            validate_protocols: vec!["proofguard".into(), "brutalhonesty".into()],
            skip_protocols: vec!["gigathink".into(), "quick".into()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thinktool::step::{StepOutput, StepResult, TokenUsage};
    use crate::thinktool::validation::{
        ChainIntegrityResult, DependencyStatus, LogicalFlowStatus, ProgressionStatus,
        ValidationPerformance,
    };
    use std::collections::HashMap;

    // =========================================================================
    // CONFIGURATION TESTS
    // =========================================================================

    #[test]
    fn test_default_validation_executor_config() {
        let config = ValidationExecutorConfig::default();

        assert!(config.enable_validation);
        assert_eq!(config.validation_level, ValidationLevel::Standard);
        assert!((config.min_confidence_threshold - 0.70).abs() < f64::EPSILON);
        assert!(config.validate_protocols.is_empty());
        assert!(config.skip_protocols.is_empty());
    }

    #[test]
    fn test_enterprise_configuration() {
        let config = ValidationExecutorConfig::enterprise();

        assert!(config.enable_validation);
        assert_eq!(config.validation_level, ValidationLevel::Rigorous);
        assert!((config.min_confidence_threshold - 0.80).abs() < f64::EPSILON);
        assert!(config.validate_protocols.contains(&"proofguard".to_string()));
        assert!(config.validate_protocols.contains(&"brutalhonesty".to_string()));
        assert!(config.validate_protocols.contains(&"laserlogic".to_string()));
        assert!(config.skip_protocols.is_empty());
    }

    #[test]
    fn test_research_configuration() {
        let config = ValidationExecutorConfig::research();

        assert!(config.enable_validation);
        assert_eq!(config.validation_level, ValidationLevel::Paranoid);
        assert!((config.min_confidence_threshold - 0.90).abs() < f64::EPSILON);
        assert!(config.validate_protocols.contains(&"gigathink".to_string()));
        assert!(config.validate_protocols.contains(&"scientific".to_string()));
    }

    #[test]
    fn test_production_configuration() {
        let config = ValidationExecutorConfig::production();

        assert!(config.enable_validation);
        assert_eq!(config.validation_level, ValidationLevel::Standard);
        assert!((config.min_confidence_threshold - 0.70).abs() < f64::EPSILON);
        assert!(config.validate_protocols.is_empty());
        assert!(config.skip_protocols.contains(&"quick".to_string()));
    }

    #[test]
    fn test_compliance_configuration() {
        let config = ValidationExecutorConfig::compliance();

        assert!(config.enable_validation);
        assert_eq!(config.validation_level, ValidationLevel::Rigorous);
        assert!((config.min_confidence_threshold - 0.85).abs() < f64::EPSILON);
        assert!(config.validate_protocols.contains(&"proofguard".to_string()));
        assert!(config.validate_protocols.contains(&"brutalhonesty".to_string()));
        assert!(config.skip_protocols.contains(&"gigathink".to_string()));
        assert!(config.skip_protocols.contains(&"quick".to_string()));
    }

    #[test]
    fn test_validation_level_default() {
        let level = ValidationLevel::default();
        assert_eq!(level, ValidationLevel::Standard);
    }

    #[test]
    fn test_validation_level_equality() {
        assert_eq!(ValidationLevel::None, ValidationLevel::None);
        assert_eq!(ValidationLevel::Quick, ValidationLevel::Quick);
        assert_eq!(ValidationLevel::Standard, ValidationLevel::Standard);
        assert_eq!(ValidationLevel::Rigorous, ValidationLevel::Rigorous);
        assert_eq!(ValidationLevel::Paranoid, ValidationLevel::Paranoid);

        assert_ne!(ValidationLevel::None, ValidationLevel::Quick);
        assert_ne!(ValidationLevel::Quick, ValidationLevel::Standard);
        assert_ne!(ValidationLevel::Standard, ValidationLevel::Rigorous);
        assert_ne!(ValidationLevel::Rigorous, ValidationLevel::Paranoid);
    }

    #[test]
    fn test_config_serialization() {
        let config = ValidationExecutorConfig::enterprise();
        let json = serde_json::to_string(&config).expect("Failed to serialize config");

        assert!(json.contains("enable_validation"));
        assert!(json.contains("rigorous"));
        assert!(json.contains("proofguard"));

        let deserialized: ValidationExecutorConfig =
            serde_json::from_str(&json).expect("Failed to deserialize config");

        assert_eq!(deserialized.validation_level, config.validation_level);
        assert_eq!(
            deserialized.min_confidence_threshold,
            config.min_confidence_threshold
        );
    }

    // =========================================================================
    // EXECUTOR CREATION TESTS
    // =========================================================================

    #[tokio::test]
    async fn test_executor_creation_with_default_config() {
        let executor = ValidatingProtocolExecutor::new().unwrap();

        // Verify executor has protocols loaded
        let protocols = executor.list_protocols();
        assert!(protocols.contains(&"gigathink"));
        assert!(protocols.contains(&"laserlogic"));
        assert!(protocols.contains(&"bedrock"));
    }

    #[tokio::test]
    async fn test_executor_creation_with_mock_config() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        assert!(executor.list_protocols().len() > 0);
        assert!(executor.list_profiles().len() > 0);
    }

    #[tokio::test]
    async fn test_executor_creation_with_disabled_validation() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: false,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Executor should still work, just without validation engine
        assert!(executor.list_protocols().len() > 0);
    }

    #[test]
    fn test_executor_list_protocols() {
        let executor = ValidatingProtocolExecutor::new().unwrap();
        let protocols = executor.list_protocols();

        // Should contain all built-in protocols
        assert!(protocols.contains(&"gigathink"));
        assert!(protocols.contains(&"laserlogic"));
        assert!(protocols.contains(&"bedrock"));
        assert!(protocols.contains(&"proofguard"));
        assert!(protocols.contains(&"brutalhonesty"));
    }

    #[test]
    fn test_executor_list_profiles() {
        let executor = ValidatingProtocolExecutor::new().unwrap();
        let profiles = executor.list_profiles();

        // Should contain all built-in profiles
        assert!(profiles.contains(&"quick"));
        assert!(profiles.contains(&"balanced"));
        assert!(profiles.contains(&"deep"));
        assert!(profiles.contains(&"paranoid"));
    }

    #[test]
    fn test_executor_get_protocol() {
        let executor = ValidatingProtocolExecutor::new().unwrap();

        let protocol = executor.get_protocol("gigathink");
        assert!(protocol.is_some());
        assert_eq!(protocol.unwrap().id, "gigathink");

        let missing = executor.get_protocol("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_executor_get_profile() {
        let executor = ValidatingProtocolExecutor::new().unwrap();

        let profile = executor.get_profile("balanced");
        assert!(profile.is_some());

        let missing = executor.get_profile("nonexistent");
        assert!(missing.is_none());
    }

    // =========================================================================
    // VALIDATION RULES TESTS (should_validate)
    // =========================================================================

    /// Helper function to create a mock ProtocolOutput for testing
    fn create_mock_output(protocol_id: &str, confidence: f64) -> ProtocolOutput {
        ProtocolOutput {
            protocol_id: protocol_id.to_string(),
            success: true,
            data: HashMap::new(),
            confidence,
            steps: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 100,
            error: None,
            trace_id: None,
            budget_summary: None,
        }
    }

    #[test]
    fn test_should_validate_disabled_validation() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: false,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let output = create_mock_output("gigathink", 0.85);
        assert!(!executor.should_validate("gigathink", &output));
    }

    #[test]
    fn test_should_validate_with_none_level() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::None,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let output = create_mock_output("gigathink", 0.85);
        assert!(!executor.should_validate("gigathink", &output));
    }

    #[test]
    fn test_should_validate_with_specific_protocols() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            validate_protocols: vec!["proofguard".to_string(), "laserlogic".to_string()],
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Should validate protocols in the list
        let output_proofguard = create_mock_output("proofguard", 0.85);
        assert!(executor.should_validate("proofguard", &output_proofguard));

        // Should NOT validate protocols not in the list
        let output_gigathink = create_mock_output("gigathink", 0.85);
        assert!(!executor.should_validate("gigathink", &output_gigathink));
    }

    #[test]
    fn test_should_validate_with_skip_protocols() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            skip_protocols: vec!["quick".to_string()],
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Should NOT validate skipped protocols
        let output_quick = create_mock_output("quick", 0.85);
        assert!(!executor.should_validate("quick", &output_quick));

        // Should validate non-skipped protocols
        let output_gigathink = create_mock_output("gigathink", 0.85);
        assert!(executor.should_validate("gigathink", &output_gigathink));
    }

    #[test]
    fn test_should_validate_below_confidence_threshold() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            min_confidence_threshold: 0.80,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Below threshold - should NOT validate
        let output_low_confidence = create_mock_output("gigathink", 0.75);
        assert!(!executor.should_validate("gigathink", &output_low_confidence));

        // At threshold - should validate
        let output_at_threshold = create_mock_output("gigathink", 0.80);
        assert!(executor.should_validate("gigathink", &output_at_threshold));

        // Above threshold - should validate
        let output_high_confidence = create_mock_output("gigathink", 0.90);
        assert!(executor.should_validate("gigathink", &output_high_confidence));
    }

    #[test]
    fn test_should_validate_all_conditions_met() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            min_confidence_threshold: 0.70,
            validate_protocols: Vec::new(), // Empty = all protocols
            skip_protocols: Vec::new(),
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let output = create_mock_output("gigathink", 0.85);
        assert!(executor.should_validate("gigathink", &output));
    }

    // =========================================================================
    // VALIDATION RULES TESTS (should_validate_profile)
    // =========================================================================

    #[test]
    fn test_should_validate_profile_disabled() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: false,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let output = create_mock_output("paranoid", 0.95);
        assert!(!executor.should_validate_profile("paranoid", &output));
    }

    #[test]
    fn test_should_validate_profile_paranoid_always() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            min_confidence_threshold: 0.99, // Very high threshold
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Paranoid profile should ALWAYS validate regardless of confidence
        let output = create_mock_output("paranoid", 0.50);
        assert!(executor.should_validate_profile("paranoid", &output));
    }

    #[test]
    fn test_should_validate_profile_deep_always() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            min_confidence_threshold: 0.99,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Deep profile should ALWAYS validate regardless of confidence
        let output = create_mock_output("deep", 0.50);
        assert!(executor.should_validate_profile("deep", &output));
    }

    #[test]
    fn test_should_validate_profile_quick_with_level() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Quick profile should validate when level is not None
        let output = create_mock_output("quick", 0.50);
        assert!(executor.should_validate_profile("quick", &output));
    }

    #[test]
    fn test_should_validate_profile_quick_with_none_level() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::None,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Quick profile should NOT validate when level is None
        let output = create_mock_output("quick", 0.90);
        assert!(!executor.should_validate_profile("quick", &output));
    }

    #[test]
    fn test_should_validate_profile_other_profiles_use_threshold() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            min_confidence_threshold: 0.80,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Other profiles should use confidence threshold
        let output_low = create_mock_output("balanced", 0.75);
        assert!(!executor.should_validate_profile("balanced", &output_low));

        let output_high = create_mock_output("balanced", 0.85);
        assert!(executor.should_validate_profile("balanced", &output_high));
    }

    // =========================================================================
    // CONFIDENCE ADJUSTMENT TESTS (merge_validation_results)
    // =========================================================================

    /// Helper function to create a mock DeepSeekValidationResult
    fn create_mock_validation_result(verdict: ValidationVerdict) -> DeepSeekValidationResult {
        use crate::thinktool::validation::TokenUsage as ValidationTokenUsage;

        DeepSeekValidationResult {
            verdict,
            chain_integrity: ChainIntegrityResult {
                logical_flow: LogicalFlowStatus::Good,
                step_dependencies: DependencyStatus::FullySatisfied,
                confidence_progression: ProgressionStatus::Monotonic,
                gaps_detected: vec![],
                continuity_score: 0.85,
            },
            statistical_results: None,
            compliance_results: None,
            meta_cognitive_results: None,
            validation_confidence: 0.90,
            findings: vec![],
            tokens_used: ValidationTokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                total_tokens: 150,
                cost_usd: 0.002,
            },
            performance: ValidationPerformance {
                duration_ms: 500,
                tokens_per_second: 300.0,
                memory_usage_mb: 50.0,
            },
        }
    }

    #[test]
    fn test_merge_validation_results_validated_boosts_confidence() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let base_result = create_mock_output("gigathink", 0.80);
        let validation_result = create_mock_validation_result(ValidationVerdict::Validated);

        let merged = executor
            .merge_validation_results(base_result, validation_result)
            .unwrap();

        // Validated verdict should boost confidence by 10% (clamped to 1.0)
        // 0.80 * 1.10 = 0.88
        assert!((merged.confidence - 0.88).abs() < 0.001);
    }

    #[test]
    fn test_merge_validation_results_validated_clamps_to_one() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let base_result = create_mock_output("gigathink", 0.95);
        let validation_result = create_mock_validation_result(ValidationVerdict::Validated);

        let merged = executor
            .merge_validation_results(base_result, validation_result)
            .unwrap();

        // 0.95 * 1.10 = 1.045 -> clamped to 1.0
        assert!((merged.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merge_validation_results_partially_validated_neutral() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let base_result = create_mock_output("gigathink", 0.80);
        let validation_result =
            create_mock_validation_result(ValidationVerdict::PartiallyValidated);

        let merged = executor
            .merge_validation_results(base_result, validation_result)
            .unwrap();

        // PartiallyValidated verdict is neutral (1.0 multiplier)
        // 0.80 * 1.00 = 0.80
        assert!((merged.confidence - 0.80).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merge_validation_results_needs_improvement_reduces() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let base_result = create_mock_output("gigathink", 0.80);
        let validation_result = create_mock_validation_result(ValidationVerdict::NeedsImprovement);

        let merged = executor
            .merge_validation_results(base_result, validation_result)
            .unwrap();

        // NeedsImprovement reduces by 15%
        // 0.80 * 0.85 = 0.68
        assert!((merged.confidence - 0.68).abs() < 0.001);
    }

    #[test]
    fn test_merge_validation_results_invalid_significantly_reduces() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let base_result = create_mock_output("gigathink", 0.80);
        let validation_result = create_mock_validation_result(ValidationVerdict::Invalid);

        let merged = executor
            .merge_validation_results(base_result, validation_result)
            .unwrap();

        // Invalid reduces by 40%
        // 0.80 * 0.60 = 0.48
        assert!((merged.confidence - 0.48).abs() < 0.001);
    }

    #[test]
    fn test_merge_validation_results_critical_issues_severely_reduces() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let base_result = create_mock_output("gigathink", 0.80);
        let validation_result = create_mock_validation_result(ValidationVerdict::CriticalIssues);

        let merged = executor
            .merge_validation_results(base_result, validation_result)
            .unwrap();

        // CriticalIssues reduces by 70%
        // 0.80 * 0.30 = 0.24
        assert!((merged.confidence - 0.24).abs() < 0.001);
    }

    #[test]
    fn test_merge_validation_results_adds_metadata() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let base_result = create_mock_output("gigathink", 0.80);
        let validation_result = create_mock_validation_result(ValidationVerdict::Validated);

        let merged = executor
            .merge_validation_results(base_result, validation_result)
            .unwrap();

        // Should contain validation metadata
        assert!(merged.data.contains_key("deepseek_validation"));
    }

    #[test]
    fn test_merge_validation_results_adds_tokens() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let mut base_result = create_mock_output("gigathink", 0.80);
        base_result.tokens = TokenUsage::new(200, 100, 0.003);

        let validation_result = create_mock_validation_result(ValidationVerdict::Validated);

        let merged = executor
            .merge_validation_results(base_result, validation_result)
            .unwrap();

        // Tokens should be added: base + validation
        // Input: 200 + 100 = 300
        // Output: 100 + 50 = 150
        assert_eq!(merged.tokens.input_tokens, 300);
        assert_eq!(merged.tokens.output_tokens, 150);
        assert_eq!(merged.tokens.total_tokens, 450);
    }

    #[test]
    fn test_merge_validation_results_adds_duration() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let mut base_result = create_mock_output("gigathink", 0.80);
        base_result.duration_ms = 1000;

        let validation_result = create_mock_validation_result(ValidationVerdict::Validated);

        let merged = executor
            .merge_validation_results(base_result, validation_result)
            .unwrap();

        // Duration should be added: 1000 + 500 = 1500
        assert_eq!(merged.duration_ms, 1500);
    }

    // =========================================================================
    // EXECUTION FLOW TESTS (with mock)
    // =========================================================================

    #[tokio::test]
    async fn test_execute_with_mock_returns_success() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: false, // Disable to avoid DeepSeek API call
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let input = ProtocolInput::query("What is machine learning?");
        let result = executor
            .execute_with_validation("gigathink", input)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
        assert!(!result.steps.is_empty());
    }

    #[tokio::test]
    async fn test_execute_profile_with_mock_returns_success() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: false, // Disable to avoid DeepSeek API call
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let input = ProtocolInput::query("Should we use microservices?");
        let result = executor
            .execute_profile_with_validation("quick", input)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_execute_nonexistent_protocol_returns_error() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: false,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let input = ProtocolInput::query("Test query");
        let result = executor
            .execute_with_validation("nonexistent_protocol", input)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_nonexistent_profile_returns_error() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: false,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let input = ProtocolInput::query("Test query");
        let result = executor
            .execute_profile_with_validation("nonexistent_profile", input)
            .await;

        assert!(result.is_err());
    }

    // =========================================================================
    // ERROR REPORTING TESTS
    // =========================================================================

    #[test]
    fn test_validation_engine_not_available_error() {
        // Create executor with validation disabled (no engine)
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: false,
            ..Default::default()
        };

        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Try to access validation engine directly (should be None)
        assert!(executor.validation_engine.is_none());
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_zero_confidence_output() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig::default();
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let output = create_mock_output("gigathink", 0.0);

        // Should NOT validate when confidence is 0 (below threshold)
        assert!(!executor.should_validate("gigathink", &output));
    }

    #[test]
    fn test_exact_threshold_confidence() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            min_confidence_threshold: 0.75,
            ..Default::default()
        };
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Exactly at threshold should validate
        let output = create_mock_output("gigathink", 0.75);
        assert!(executor.should_validate("gigathink", &output));
    }

    #[test]
    fn test_just_below_threshold() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            min_confidence_threshold: 0.75,
            ..Default::default()
        };
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        // Just below threshold should NOT validate
        let output = create_mock_output("gigathink", 0.7499999);
        assert!(!executor.should_validate("gigathink", &output));
    }

    #[test]
    fn test_empty_validate_protocols_means_all() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            validate_protocols: Vec::new(), // Empty = validate all
            skip_protocols: Vec::new(),
            min_confidence_threshold: 0.0,
        };
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let output = create_mock_output("any_protocol", 0.50);
        assert!(executor.should_validate("any_protocol", &output));
    }

    #[test]
    fn test_skip_takes_precedence_over_validate() {
        let executor_config = ExecutorConfig::mock();
        let validation_config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Standard,
            validate_protocols: vec!["gigathink".to_string()],
            skip_protocols: vec!["gigathink".to_string()], // Both contain gigathink
            min_confidence_threshold: 0.0,
        };
        let executor =
            ValidatingProtocolExecutor::with_configs(executor_config, validation_config).unwrap();

        let output = create_mock_output("gigathink", 0.90);

        // Skip should take precedence - should NOT validate
        assert!(!executor.should_validate("gigathink", &output));
    }

    #[test]
    fn test_default_executor_via_default_trait() {
        // Test that Default trait implementation works
        let executor = ValidatingProtocolExecutor::default();
        assert!(executor.list_protocols().len() > 0);
    }

    #[test]
    fn test_validation_level_serde_roundtrip() {
        for level in [
            ValidationLevel::None,
            ValidationLevel::Quick,
            ValidationLevel::Standard,
            ValidationLevel::Rigorous,
            ValidationLevel::Paranoid,
        ] {
            let json = serde_json::to_string(&level).unwrap();
            let deserialized: ValidationLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, level);
        }
    }

    #[test]
    fn test_config_with_all_fields_set() {
        let config = ValidationExecutorConfig {
            enable_validation: true,
            validation_level: ValidationLevel::Rigorous,
            min_confidence_threshold: 0.95,
            validate_protocols: vec!["a".to_string(), "b".to_string()],
            skip_protocols: vec!["c".to_string()],
        };

        assert!(config.enable_validation);
        assert_eq!(config.validation_level, ValidationLevel::Rigorous);
        assert!((config.min_confidence_threshold - 0.95).abs() < f64::EPSILON);
        assert_eq!(config.validate_protocols.len(), 2);
        assert_eq!(config.skip_protocols.len(), 1);
    }
}
