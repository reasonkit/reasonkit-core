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

    #[test]
    fn test_configuration_options() {
        let enterprise_config = ValidationExecutorConfig::enterprise();
        assert!(enterprise_config.enable_validation);
        assert_eq!(
            enterprise_config.validation_level,
            ValidationLevel::Rigorous
        );
        assert!(enterprise_config
            .validate_protocols
            .contains(&"proofguard".to_string()));

        let research_config = ValidationExecutorConfig::research();
        assert_eq!(research_config.validation_level, ValidationLevel::Paranoid);
    }

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = ValidatingProtocolExecutor::new().unwrap();
        assert!(executor.list_protocols().contains(&"gigathink"));
    }
}
