//! Enhanced VIBE Validation Engine with DeepSeek Integration
//!
//! Extends the existing VIBE engine with DeepSeek-V3.1 validation capabilities
//! and multi-model triangulation for comprehensive protocol assessment.

use crate::thinktool::{
    DeepSeekValidationConfig, DeepSeekValidationEngine, DeepSeekValidationResult,
    ValidationVerdict,
};
use crate::vibe::{
    cross_cultural::{CulturalConfig, CulturalValidationEngine, CulturalValidationResult},
    multi_model_validator::{MultiModelConfig, MultiModelValidationResult, MultiModelValidator},
    statistical_testing::{StatisticalConfig, StatisticalEngine, StatisticalResult},
    ValidationConfig, ValidationResult, VIBEEngine, VIBEError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced validation configuration with DeepSeek integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekVIBEConfig {
    /// Base VIBE configuration
    pub vibe_config: ValidationConfig,

    /// DeepSeek validation configuration
    pub deepseek_config: DeepSeekValidationConfig,

    /// Multi-model configuration
    pub multi_model_config: MultiModelConfig,

    /// Statistical testing configuration
    pub statistical_config: StatisticalConfig,

    /// Cultural validation configuration
    pub cultural_config: CulturalConfig,

    /// Enable DeepSeek validation
    pub enable_deepseek: bool,

    /// Enable multi-model triangulation
    pub enable_triangulation: bool,

    /// Enable statistical significance testing
    pub enable_statistical: bool,

    /// Enable cross-cultural validation
    pub enable_cultural: bool,
}

impl Default for DeepSeekVIBEConfig {
    fn default() -> Self {
        Self {
            vibe_config: ValidationConfig::default(),
            deepseek_config: DeepSeekValidationConfig::default(),
            multi_model_config: MultiModelConfig::default(),
            statistical_config: StatisticalConfig::default(),
            cultural_config: CulturalConfig::default(),
            enable_deepseek: true,
            enable_triangulation: true,
            enable_statistical: true,
            enable_cultural: true,
        }
    }
}

impl DeepSeekVIBEConfig {
    /// Create enterprise configuration
    pub fn enterprise() -> Self {
        Self {
            vibe_config: ValidationConfig::enterprise(),
            deepseek_config: DeepSeekValidationConfig::enterprise(),
            multi_model_config: MultiModelConfig::enterprise(),
            statistical_config: StatisticalConfig::default(),
            cultural_config: CulturalConfig::default(),
            enable_deepseek: true,
            enable_triangulation: true,
            enable_statistical: true,
            enable_cultural: true,
        }
    }

    /// Create research-grade configuration
    pub fn research() -> Self {
        Self {
            vibe_config: ValidationConfig::research(),
            deepseek_config: DeepSeekValidationConfig::rigorous(),
            multi_model_config: MultiModelConfig::research(),
            statistical_config: StatisticalConfig::default(),
            cultural_config: CulturalConfig::default(),
            enable_deepseek: true,
            enable_triangulation: true,
            enable_statistical: true,
            enable_cultural: true,
        }
    }
}

/// Enhanced validation result with DeepSeek integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekVIBEResult {
    /// Base VIBE validation result
    pub vibe_result: ValidationResult,

    /// DeepSeek validation result
    pub deepseek_result: Option<DeepSeekValidationResult>,

    /// Multi-model triangulation result
    pub multi_model_result: Option<MultiModelValidationResult>,

    /// Statistical testing result
    pub statistical_result: Option<StatisticalResult>,

    /// Cross-cultural validation result
    pub cultural_result: Option<CulturalValidationResult>,

    /// Final aggregated score
    pub final_score: f32,

    /// Overall confidence level
    pub overall_confidence: f32,

    /// Validation verdict
    pub verdict: ValidationVerdict,
}

/// Enhanced VIBE engine with DeepSeek integration
pub struct DeepSeekVIBEEngine {
    /// Base VIBE engine
    vibe_engine: VIBEEngine,

    /// DeepSeek validation engine
    deepseek_engine: DeepSeekValidationEngine,

    /// Multi-model validator
    multi_model_validator: MultiModelValidator,

    /// Statistical engine
    statistical_engine: StatisticalEngine,

    /// Cultural validation engine
    cultural_engine: CulturalValidationEngine,

    /// Configuration
    config: DeepSeekVIBEConfig,
}

impl DeepSeekVIBEEngine {
    /// Create new enhanced VIBE engine
    pub fn new(config: DeepSeekVIBEConfig) -> Result<Self, VIBEError> {
        Ok(Self {
            vibe_engine: VIBEEngine::new(),
            deepseek_engine: DeepSeekValidationEngine::new()?,
            multi_model_validator: MultiModelValidator::new()?,
            statistical_engine: StatisticalEngine::default(),
            cultural_engine: CulturalValidationEngine::new(config.cultural_config.clone())?,
            config,
        })
    }

    /// Create with default configuration
    pub fn default() -> Result<Self, VIBEError> {
        Self::new(DeepSeekVIBEConfig::default())
    }

    /// Create enterprise engine
    pub fn enterprise() -> Result<Self, VIBEError> {
        Self::new(DeepSeekVIBEConfig::enterprise())
    }

    /// Create research-grade engine
    pub fn research() -> Result<Self, VIBEError> {
        Self::new(DeepSeekVIBEConfig::research())
    }

    /// Validate protocol with full DeepSeek integration
    pub async fn validate_with_deepseek(
        &self,
        protocol: &str,
    ) -> Result<DeepSeekVIBEResult, VIBEError> {
        let start_time = std::time::Instant::now();

        // Step 1: Base VIBE validation
        let vibe_result = self
            .vibe_engine
            .validate_protocol(protocol, self.config.vibe_config.clone())
            .await?;

        // Step 2: DeepSeek validation (if enabled)
        let deepseek_result = if self.config.enable_deepseek {
            Some(
                self.deepseek_engine
                    .validate_reasoning_chain(protocol)
                    .await?,
            )
        } else {
            None
        };

        // Step 3: Multi-model triangulation (if enabled)
        let multi_model_result = if self.config.enable_triangulation {
            Some(
                self.multi_model_validator
                    .validate_triangulation(protocol)
                    .await?,
            )
        } else {
            None
        };

        // Step 4: Statistical significance testing (if enabled)
        let statistical_result = if self.config.enable_statistical {
            Some(self.perform_statistical_analysis(&vibe_result, &deepseek_result)?)
        } else {
            None
        };

        // Step 5: Cross-cultural validation (if enabled)
        let cultural_result = if self.config.enable_cultural {
            Some(
                self.cultural_engine
                    .validate_cultural(protocol, vibe_result.clone())
                    .await?,
            )
        } else {
            None
        };

        // Step 6: Aggregate all results
        let (final_score, overall_confidence, verdict) =
            self.aggregate_results(
                &vibe_result,
                &deepseek_result,
                &multi_model_result,
                &statistical_result,
                &cultural_result,
            );

        let validation_time = start_time.elapsed().as_millis() as u64;

        Ok(DeepSeekVIBEResult {
            vibe_result: ValidationResult {
                validation_time_ms: validation_time,
                ..vibe_result
            },
            deepseek_result,
            multi_model_result,
            statistical_result,
            cultural_result,
            final_score,
            overall_confidence,
            verdict,
        })
    }

    /// Validate with specific DeepSeek model
    pub async fn validate_with_specific_deepseek(
        &self,
        protocol: &str,
        model_name: &str,
    ) -> Result<DeepSeekVIBEResult, VIBEError> {
        // Create custom DeepSeek configuration
        let mut custom_config = self.config.deepseek_config.clone();
        custom_config.model = model_name.to_string();

        let custom_engine = DeepSeekValidationEngine::with_config(custom_config)?;

        // Perform validation with custom engine
        let vibe_result = self
            .vibe_engine
            .validate_protocol(protocol, self.config.vibe_config.clone())
            .await?;

        let deepseek_result = custom_engine.validate_reasoning_chain(protocol).await?;

        // Aggregate results
        let (final_score, overall_confidence, verdict) = self.aggregate_basic_results(
            &vibe_result,
            &deepseek_result,
            None,
            None,
            None,
        );

        Ok(DeepSeekVIBEResult {
            vibe_result,
            deepseek_result: Some(deepseek_result),
            multi_model_result: None,
            statistical_result: None,
            cultural_result: None,
            final_score,
            overall_confidence,
            verdict,
        })
    }

    /// Perform quick validation (DeepSeek only)
    pub async fn validate_quick(
        &self,
        protocol: &str,
    ) -> Result<DeepSeekVIBEResult, VIBEError> {
        let start_time = std::time::Instant::now();

        // Quick validation: DeepSeek only
        let deepseek_result = self
            .deepseek_engine
            .validate_reasoning_chain(protocol)
            .await?;

        let vibe_result = ValidationResult {
            overall_score: deepseek_result.confidence * 100.0,
            platform_scores: HashMap::new(),
            confidence_interval: None,
            status: crate::vibe::validation::ValidationStatus::Validated,
            detailed_results: HashMap::new(),
            validation_time_ms: start_time.elapsed().as_millis() as u64,
            issues: Vec::new(),
            recommendations: Vec::new(),
            timestamp: chrono::Utc::now(),
            protocol_id: uuid::Uuid::new_v4(),
        };

        let verdict = if deepseek_result.confidence > 0.8 {
            ValidationVerdict::Validated
        } else {
            ValidationVerdict::PartiallyValidated
        };

        Ok(DeepSeekVIBEResult {
            vibe_result,
            deepseek_result: Some(deepseek_result),
            multi_model_result: None,
            statistical_result: None,
            cultural_result: None,
            final_score: deepseek_result.confidence * 100.0,
            overall_confidence: deepseek_result.confidence,
            verdict,
        })
    }

    /// Perform statistical analysis
    fn perform_statistical_analysis(
        &self,
        vibe_result: &ValidationResult,
        deepseek_result: &Option<DeepSeekValidationResult>,
    ) -> Result<StatisticalResult, VIBEError> {
        // Create validation result vector for statistical testing
        let mut validation_results = vec![vibe_result.clone()];

        if let Some(ds_result) = deepseek_result {
            // Convert DeepSeek result to VIBE result format
            let ds_vibe_result = ValidationResult {
                overall_score: ds_result.confidence * 100.0,
                platform_scores: HashMap::new(),
                confidence_interval: None,
                status: crate::vibe::validation::ValidationStatus::Validated,
                detailed_results: HashMap::new(),
                validation_time_ms: 0,
                issues: Vec::new(),
                recommendations: Vec::new(),
                timestamp: chrono::Utc::now(),
                protocol_id: uuid::Uuid::new_v4(),
            };
            validation_results.push(ds_vibe_result);
        }

        // Perform bootstrap significance test
        let mut statistical_engine = StatisticalEngine::new(self.config.statistical_config.clone());
        statistical_engine.bootstrap_significance_test(&validation_results, 0.7)
    }

    /// Aggregate all validation results
    fn aggregate_results(
        &self,
        vibe_result: &ValidationResult,
        deepseek_result: &Option<DeepSeekValidationResult>,
        multi_model_result: &Option<MultiModelValidationResult>,
        statistical_result: &Option<StatisticalResult>,
        cultural_result: &Option<CulturalValidationResult>,
    ) -> (f32, f32, ValidationVerdict) {
        // Start with base VIBE score
        let mut total_score = vibe_result.overall_score;
        let mut total_weight = 1.0;

        // Add DeepSeek score
        if let Some(ds_result) = deepseek_result {
            total_score += ds_result.confidence * 100.0 * 0.6; // DeepSeek gets 60% weight
            total_weight += 0.6;
        }

        // Add multi-model score
        if let Some(mm_result) = multi_model_result {
            total_score += mm_result.overall_score * 0.4; // Multi-model gets 40% weight
            total_weight += 0.4;
        }

        // Add cultural score
        if let Some(cultural_result) = cultural_result {
            total_score += cultural_result.cultural_score * 0.3; // Cultural gets 30% weight
            total_weight += 0.3;
        }

        // Calculate final score
        let final_score = total_score / total_weight;

        // Calculate overall confidence
        let mut confidence = vibe_result.overall_score / 100.0;

        if let Some(ds_result) = deepseek_result {
            confidence = (confidence + ds_result.confidence) / 2.0;
        }

        if let Some(mm_result) = multi_model_result {
            confidence = (confidence + mm_result.triangulation_score) / 2.0;
        }

        // Determine verdict
        let verdict = if final_score > 90.0 {
            ValidationVerdict::StronglyValidated
        } else if final_score > 75.0 {
            ValidationVerdict::Validated
        } else if final_score > 60.0 {
            ValidationVerdict::PartiallyValidated
        } else {
            ValidationVerdict::Rejected
        };

        (final_score, confidence, verdict)
    }

    /// Basic result aggregation for simplified validation
    fn aggregate_basic_results(
        &self,
        vibe_result: &ValidationResult,
        deepseek_result: &DeepSeekValidationResult,
        multi_model_result: &Option<MultiModelValidationResult>,
        statistical_result: &Option<StatisticalResult>,
        cultural_result: &Option<CulturalValidationResult>,
    ) -> (f32, f32, ValidationVerdict) {
        self.aggregate_results(
            vibe_result,
            &Some(deepseek_result.clone()),
            multi_model_result,
            statistical_result,
            cultural_result,
        )
    }

    /// Get configuration reference
    pub fn config(&self) -> &DeepSeekVIBEConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: DeepSeekVIBEConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_deepseek_vibe_engine_creation() {
        let engine = DeepSeekVIBEEngine::default().unwrap();
        assert!(engine.config.enable_deepseek);
        assert!(engine.config.enable_triangulation);
    }

    #[tokio::test]
    async fn test_quick_validation() {
        let engine = DeepSeekVIBEEngine::default().unwrap();
        let result = engine
            .validate_quick("Sample protocol for quick validation")
            .await
            .unwrap();

        assert!(result.final_score > 0.0);
        assert!(result.overall_confidence > 0.0);
    }

    #[tokio::test]
    async fn test_full_validation() {
        let engine = DeepSeekVIBEEngine::enterprise().unwrap();
        let result = engine
            .validate_with_deepseek("Comprehensive protocol validation test")
            .await
            .unwrap();

        assert!(result.final_score > 0.0);
        assert!(result.deepseek_result.is_some());
    }
}
