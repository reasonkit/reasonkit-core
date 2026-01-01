//! # Aesthetic Expression Mastery System
//!
//! **M2-Enhanced UI/UX Assessment leveraging VIBE Benchmark Excellence**
//!
//! This module implements a comprehensive design evaluation framework that combines
//! MiniMax M2's proven aesthetic expression mastery with ReasonKit's structured
//! reasoning protocols.
//!
//! ## VIBE Benchmark Targets
//! - **VIBE-Web**: 91.5% (M2's proven benchmark)
//! - **VIBE-Android**: 89.7%
//! - **VIBE-iOS**: 88.0%
//!
//! ## Core Capabilities
//! - Visual Design Assessment (color, typography, layout)
//! - Usability Evaluation (UX patterns, interaction design)
//! - Accessibility Assessment (WCAG 2.1 AA/AAA compliance)
//! - 3D Rendering Evaluation (React Three Fiber, WebGL)
//! - Cross-Platform Design Validation (Web, iOS, Android)
//! - Performance Impact Analysis

pub mod accessibility;
pub mod cross_platform;
pub mod engine;
pub mod performance;
pub mod three_d;
pub mod types;
pub mod usability;
pub mod vibe_benchmark;
pub mod visual_analysis;

// Re-export the engine
pub use engine::AestheticMasteryEngine;
pub use engine::QuickAssessment;

// Re-export all types from types.rs
pub use types::*;

// Re-export types from accessibility.rs
pub use accessibility::{
    ContrastAnalyzer, KeyboardAccessibilityAnalyzer, KeyboardAccessibilityResult,
    ScreenReaderAnalyzer, ScreenReaderResult, WcagChecker,
};

// Re-export types from cross_platform.rs
pub use cross_platform::{CrossPlatformValidator, ResponsiveValidationResult, ResponsiveValidator};

// Re-export types from performance.rs
pub use performance::{
    CoreWebVitalsAnalyzer, CoreWebVitalsEstimate, PerformanceAnalyzer, VitalRating,
};

// Re-export types from three_d.rs
pub use three_d::{
    R3FAnalysisResult, ReactThreeFiberAnalyzer, ThreeDEvaluator, M2_R3F_INSTANCE_CAPABILITY,
};

// Re-export types from usability.rs
pub use usability::{
    FlowAnalysisResult, FlowStep, FrictionLevel, FrictionPoint, HeuristicEvaluator, HeuristicScore,
    UsabilityEvaluator, UserFlowAnalyzer,
};

// Re-export types from vibe_benchmark.rs
pub use vibe_benchmark::{
    M2BenchmarkComparison, M2ComparisonResult, M2ComparisonStatus, MultiPlatformVibeEvaluator,
    VibeBenchmarkEvaluator, VibeInterpretation, VibeRating, VibeScoreInterpreter,
    VIBE_ANDROID_BENCHMARK, VIBE_IOS_BENCHMARK, VIBE_WEB_BENCHMARK,
};

// Re-export types from visual_analysis.rs
pub use visual_analysis::{
    ColorHarmonyAnalyzer, ConsistencyAnalyzer, HierarchyAnalyzer, LayoutAnalyzer,
    TypographyAnalyzer, WhiteSpaceAnalyzer,
};

use crate::error::Error;
use serde::{Deserialize, Serialize};

/// VIBE Benchmark Excellence Targets (proven M2 benchmarks)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VibeBenchmarkTargets {
    pub web: f64,     // 91.5%
    pub android: f64, // 89.7%
    pub ios: f64,     // 88.0%
}

impl Default for VibeBenchmarkTargets {
    fn default() -> Self {
        Self {
            web: 0.915,
            android: 0.897,
            ios: 0.880,
        }
    }
}

/// Aesthetic Expression Mastery Service
///
/// Primary entry point for UI/UX assessment combining M2's aesthetic
/// capabilities with ReasonKit's structured evaluation protocols.
#[derive(Debug, Clone)]
pub struct AestheticExpressionService {
    #[allow(dead_code)]
    config: AestheticConfig,
    vibe_targets: VibeBenchmarkTargets,
    engine: AestheticMasteryEngine,
}

impl AestheticExpressionService {
    /// Create a new Aesthetic Expression Service
    pub async fn new(config: AestheticConfig) -> Result<Self, Error> {
        let vibe_targets = VibeBenchmarkTargets::default();
        let engine = AestheticMasteryEngine::new(config.clone())?;

        Ok(Self {
            config,
            vibe_targets,
            engine,
        })
    }

    /// Perform comprehensive UI/UX assessment
    pub async fn assess_design(&self, input: DesignInput) -> Result<DesignAssessmentResult, Error> {
        self.engine.comprehensive_assessment(input).await
    }

    /// Quick visual-only assessment
    pub async fn quick_visual_check(
        &self,
        input: DesignInput,
    ) -> Result<VisualAssessmentResult, Error> {
        self.engine.visual_assessment(input).await
    }

    /// Accessibility compliance check
    pub async fn check_accessibility(
        &self,
        input: DesignInput,
    ) -> Result<AccessibilityResult, Error> {
        self.engine.accessibility_assessment(input).await
    }

    /// Cross-platform validation
    pub async fn validate_cross_platform(
        &self,
        input: DesignInput,
    ) -> Result<CrossPlatformResult, Error> {
        self.engine.cross_platform_validation(input).await
    }

    /// 3D rendering assessment
    pub async fn assess_3d_design(
        &self,
        input: ThreeDDesignInput,
    ) -> Result<ThreeDAssessmentResult, Error> {
        self.engine.three_d_assessment(input).await
    }

    /// Get VIBE benchmark compliance score
    pub fn get_vibe_compliance(&self, platform: Platform) -> f64 {
        match platform {
            Platform::Web => self.vibe_targets.web,
            Platform::Android => self.vibe_targets.android,
            Platform::IOS => self.vibe_targets.ios,
            Platform::Desktop => (self.vibe_targets.web + self.vibe_targets.ios) / 2.0,
        }
    }
}

/// Service builder for customization
#[derive(Debug, Clone, Default)]
pub struct AestheticServiceBuilder {
    config: Option<AestheticConfig>,
    custom_vibe_targets: Option<VibeBenchmarkTargets>,
}

impl AestheticServiceBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(mut self, config: AestheticConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn with_vibe_targets(mut self, targets: VibeBenchmarkTargets) -> Self {
        self.custom_vibe_targets = Some(targets);
        self
    }

    pub async fn build(self) -> Result<AestheticExpressionService, Error> {
        let config = self.config.unwrap_or_default();
        let mut service = AestheticExpressionService::new(config).await?;

        if let Some(targets) = self.custom_vibe_targets {
            service.vibe_targets = targets;
        }

        Ok(service)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vibe_defaults() {
        let targets = VibeBenchmarkTargets::default();
        assert!((targets.web - 0.915).abs() < 0.001);
        assert!((targets.android - 0.897).abs() < 0.001);
        assert!((targets.ios - 0.880).abs() < 0.001);
    }
}
