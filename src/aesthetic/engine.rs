//! # Aesthetic Mastery Engine
//!
//! Main orchestration engine for the Aesthetic Expression Mastery System.
//! Combines all analysis modules for comprehensive UI/UX assessment.

use super::accessibility::WcagChecker;
use super::cross_platform::CrossPlatformValidator;
use super::performance::PerformanceAnalyzer;
use super::three_d::ThreeDEvaluator;
use super::types::*;
use super::usability::UsabilityEvaluator;
use super::vibe_benchmark::{M2BenchmarkComparison, VibeBenchmarkEvaluator, VibeScoreInterpreter};
use super::visual_analysis::*;
use crate::error::Error;
use chrono::Utc;

/// Aesthetic Mastery Engine
///
/// Central orchestrator for all UI/UX assessment capabilities,
/// leveraging M2's proven aesthetic expression mastery.
#[derive(Debug, Clone)]
pub struct AestheticMasteryEngine {
    config: AestheticConfig,
}

impl AestheticMasteryEngine {
    /// Create a new Aesthetic Mastery Engine
    pub fn new(config: AestheticConfig) -> Result<Self, Error> {
        Ok(Self { config })
    }

    /// Perform comprehensive design assessment
    pub async fn comprehensive_assessment(
        &self,
        input: DesignInput,
    ) -> Result<DesignAssessmentResult, Error> {
        let start_time = std::time::Instant::now();

        // Visual assessment
        let visual = if self.config.enable_visual {
            self.visual_assessment(input.clone()).await?
        } else {
            VisualAssessmentResult::default()
        };

        // Usability assessment
        let usability = if self.config.enable_usability {
            UsabilityEvaluator::evaluate(&input)
        } else {
            UsabilityAssessmentResult::default()
        };

        // Accessibility assessment
        let accessibility = if self.config.enable_accessibility {
            self.accessibility_assessment(input.clone()).await?
        } else {
            AccessibilityResult::default()
        };

        // Cross-platform validation
        let cross_platform = if self.config.enable_cross_platform {
            self.cross_platform_validation(input.clone()).await?
        } else {
            CrossPlatformResult::default()
        };

        // Performance impact
        let performance = if self.config.enable_performance {
            PerformanceAnalyzer::analyze(&input)
        } else {
            PerformanceImpactResult::default()
        };

        // VIBE benchmark evaluation
        let vibe_evaluator = VibeBenchmarkEvaluator::new(input.platform);
        let vibe_compliance = vibe_evaluator.evaluate(&input);

        // Calculate overall score (weighted average)
        let overall_score = Self::calculate_overall_score(
            &visual,
            &usability,
            &accessibility,
            &cross_platform,
            &performance,
            &vibe_compliance,
        );

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &visual,
            &usability,
            &accessibility,
            &cross_platform,
            &performance,
            &vibe_compliance,
        );

        // Build metadata
        let metadata = AssessmentMetadata {
            assessment_id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            duration_ms: start_time.elapsed().as_millis() as u64,
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            config_hash: format!("{:x}", md5::compute(format!("{:?}", self.config))),
        };

        Ok(DesignAssessmentResult {
            overall_score,
            vibe_compliance,
            visual,
            usability,
            accessibility,
            cross_platform,
            performance,
            recommendations,
            metadata,
        })
    }

    /// Visual-only assessment
    pub async fn visual_assessment(
        &self,
        input: DesignInput,
    ) -> Result<VisualAssessmentResult, Error> {
        let tokens = input.design_tokens.clone().unwrap_or_default();

        // Color harmony analysis
        let colors: Vec<String> = [
            tokens.colors.primary.clone(),
            tokens.colors.secondary.clone(),
            tokens.colors.background.clone(),
            tokens.colors.text_primary.clone(),
        ]
        .iter()
        .filter_map(|c| c.clone())
        .collect();

        let color_harmony = if !colors.is_empty() {
            ColorHarmonyAnalyzer::analyze(&colors)
        } else {
            ColorHarmonyResult::default()
        };

        // Typography analysis
        let typography = TypographyAnalyzer::analyze(&tokens.typography);

        // Layout analysis
        let layout = LayoutAnalyzer::analyze(&tokens.spacing);

        // Hierarchy analysis
        let hierarchy = HierarchyAnalyzer::analyze();

        // Consistency analysis
        let consistency = ConsistencyAnalyzer::analyze(&tokens);

        // White space analysis
        let white_space = WhiteSpaceAnalyzer::analyze();

        // Calculate overall visual score
        let score = color_harmony.score * 0.20
            + typography.score * 0.20
            + layout.score * 0.20
            + hierarchy.score * 0.15
            + consistency.score * 0.15
            + white_space.score * 0.10;

        Ok(VisualAssessmentResult {
            score,
            color_harmony,
            typography,
            layout,
            hierarchy,
            consistency,
            white_space,
        })
    }

    /// Accessibility assessment
    pub async fn accessibility_assessment(
        &self,
        input: DesignInput,
    ) -> Result<AccessibilityResult, Error> {
        let checker = WcagChecker::new(self.config.wcag_level);
        Ok(checker.assess(&input))
    }

    /// Cross-platform validation
    pub async fn cross_platform_validation(
        &self,
        input: DesignInput,
    ) -> Result<CrossPlatformResult, Error> {
        let platforms = vec![Platform::Web, Platform::Android, Platform::IOS];

        Ok(CrossPlatformValidator::validate(&input, &platforms))
    }

    /// 3D design assessment
    pub async fn three_d_assessment(
        &self,
        input: ThreeDDesignInput,
    ) -> Result<ThreeDAssessmentResult, Error> {
        let evaluator = ThreeDEvaluator::new(input.performance_targets.clone());
        Ok(evaluator.evaluate(&input))
    }

    /// Calculate overall score
    fn calculate_overall_score(
        visual: &VisualAssessmentResult,
        usability: &UsabilityAssessmentResult,
        accessibility: &AccessibilityResult,
        cross_platform: &CrossPlatformResult,
        performance: &PerformanceImpactResult,
        vibe: &VibeComplianceResult,
    ) -> f64 {
        // Weighted scoring
        let weights = [
            (visual.score, 0.20),         // Visual design
            (usability.score, 0.20),      // User experience
            (accessibility.score, 0.20),  // Accessibility
            (cross_platform.score, 0.10), // Cross-platform
            (performance.score, 0.10),    // Performance
            (vibe.overall_score, 0.20),   // VIBE benchmark
        ];

        weights.iter().map(|(score, weight)| score * weight).sum()
    }

    /// Generate improvement recommendations
    fn generate_recommendations(
        &self,
        visual: &VisualAssessmentResult,
        usability: &UsabilityAssessmentResult,
        accessibility: &AccessibilityResult,
        _cross_platform: &CrossPlatformResult,
        performance: &PerformanceImpactResult,
        vibe: &VibeComplianceResult,
    ) -> Vec<DesignRecommendation> {
        let mut recommendations = Vec::new();

        // Visual recommendations
        if visual.score < 0.85 {
            if visual.color_harmony.score < 0.80 {
                recommendations.push(DesignRecommendation {
                    category: RecommendationCategory::Visual,
                    priority: OptimizationPriority::High,
                    title: "Improve Color Harmony".to_string(),
                    description: "Review color palette for better harmony and contrast".to_string(),
                    expected_impact: "10-15% improvement in visual assessment".to_string(),
                    implementation_difficulty: DifficultyLevel::Medium,
                    before_after: None,
                });
            }

            if visual.typography.score < 0.80 {
                recommendations.push(DesignRecommendation {
                    category: RecommendationCategory::Visual,
                    priority: OptimizationPriority::High,
                    title: "Strengthen Typography System".to_string(),
                    description: "Establish clear typographic hierarchy with consistent scale"
                        .to_string(),
                    expected_impact: "8-12% improvement in readability scores".to_string(),
                    implementation_difficulty: DifficultyLevel::Easy,
                    before_after: None,
                });
            }
        }

        // Usability recommendations
        for issue in &usability.issues {
            recommendations.push(DesignRecommendation {
                category: RecommendationCategory::Usability,
                priority: match issue.severity {
                    IssueSeverity::Critical => OptimizationPriority::Critical,
                    IssueSeverity::Major => OptimizationPriority::High,
                    _ => OptimizationPriority::Medium,
                },
                title: format!("Address {:?} Issue", issue.heuristic),
                description: issue.description.clone(),
                expected_impact: "Improved user experience".to_string(),
                implementation_difficulty: DifficultyLevel::Medium,
                before_after: None,
            });
        }

        // Accessibility recommendations
        for issue in &accessibility.issues {
            recommendations.push(DesignRecommendation {
                category: RecommendationCategory::Accessibility,
                priority: match issue.impact {
                    AccessibilityImpact::Critical => OptimizationPriority::Critical,
                    AccessibilityImpact::Serious => OptimizationPriority::High,
                    _ => OptimizationPriority::Medium,
                },
                title: format!("Fix WCAG {}", issue.wcag_criterion),
                description: issue.description.clone(),
                expected_impact: "WCAG compliance improvement".to_string(),
                implementation_difficulty: DifficultyLevel::Medium,
                before_after: None,
            });
        }

        // Performance recommendations
        for rec in &performance.recommendations {
            recommendations.push(DesignRecommendation {
                category: RecommendationCategory::Performance,
                priority: rec.priority,
                title: rec.category.clone(),
                description: rec.description.clone(),
                expected_impact: rec.impact.clone(),
                implementation_difficulty: DifficultyLevel::Medium,
                before_after: None,
            });
        }

        // VIBE benchmark recommendations
        if !vibe.passes_threshold {
            let interpretation = VibeScoreInterpreter::interpret(vibe.overall_score);
            for area in interpretation.improvement_areas {
                recommendations.push(DesignRecommendation {
                    category: RecommendationCategory::Visual,
                    priority: OptimizationPriority::High,
                    title: "VIBE Benchmark Improvement".to_string(),
                    description: area,
                    expected_impact: "Progress toward M2 benchmark targets".to_string(),
                    implementation_difficulty: DifficultyLevel::Medium,
                    before_after: None,
                });
            }
        }

        // Sort by priority
        recommendations.sort_by(|a, b| {
            let priority_order = |p: &OptimizationPriority| match p {
                OptimizationPriority::Critical => 0,
                OptimizationPriority::High => 1,
                OptimizationPriority::Medium => 2,
                OptimizationPriority::Low => 3,
            };
            priority_order(&a.priority).cmp(&priority_order(&b.priority))
        });

        recommendations
    }
}

/// Quick assessment for rapid design evaluation
pub struct QuickAssessment;

impl QuickAssessment {
    /// Perform a quick design assessment (visual + VIBE only)
    pub fn assess(input: &DesignInput) -> QuickAssessmentResult {
        let tokens = input.design_tokens.clone().unwrap_or_default();

        // Quick visual check
        let colors: Vec<String> = [
            tokens.colors.primary.clone(),
            tokens.colors.secondary.clone(),
        ]
        .iter()
        .filter_map(|c| c.clone())
        .collect();

        let color_score = if !colors.is_empty() {
            ColorHarmonyAnalyzer::analyze(&colors).score
        } else {
            0.7
        };

        let typography_score = TypographyAnalyzer::analyze(&tokens.typography).score;

        // Quick VIBE check
        let vibe_evaluator = VibeBenchmarkEvaluator::new(input.platform);
        let vibe = vibe_evaluator.evaluate(input);

        let visual_score = (color_score + typography_score) / 2.0;
        let overall_score = (visual_score + vibe.overall_score) / 2.0;

        QuickAssessmentResult {
            overall_score,
            visual_score,
            vibe_score: vibe.overall_score,
            passes_vibe: vibe.passes_threshold,
            m2_comparison: M2BenchmarkComparison::compare(vibe.overall_score, input.platform),
        }
    }
}

/// Quick assessment result
#[derive(Debug, Clone)]
pub struct QuickAssessmentResult {
    pub overall_score: f64,
    pub visual_score: f64,
    pub vibe_score: f64,
    pub passes_vibe: bool,
    pub m2_comparison: super::vibe_benchmark::M2ComparisonResult,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_comprehensive_assessment() {
        let engine = AestheticMasteryEngine::new(AestheticConfig::default()).unwrap();

        let input = DesignInput {
            data: DesignData::Html("<button class='btn'>Click Me</button>".to_string()),
            platform: Platform::Web,
            context: Some("Primary CTA button".to_string()),
            component_type: ComponentType::Button,
            design_tokens: Some(DesignTokens {
                colors: ColorTokens::reasonkit_brand(),
                typography: TypographyTokens::reasonkit_brand(),
                ..Default::default()
            }),
        };

        let result = engine.comprehensive_assessment(input).await.unwrap();

        assert!(result.overall_score > 0.0);
        assert!(!result.recommendations.is_empty() || result.overall_score > 0.95);
        assert!(result.metadata.duration_ms < 5000); // Should be fast
    }

    #[test]
    fn test_quick_assessment() {
        let input = DesignInput {
            data: DesignData::Html("<nav>Navigation</nav>".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Navigation,
            design_tokens: Some(DesignTokens {
                colors: ColorTokens::reasonkit_brand(),
                typography: TypographyTokens::reasonkit_brand(),
                ..Default::default()
            }),
        };

        let result = QuickAssessment::assess(&input);

        assert!(result.overall_score > 0.0);
        assert!(result.vibe_score > 0.0);
    }

    #[tokio::test]
    async fn test_3d_assessment() {
        let engine = AestheticMasteryEngine::new(AestheticConfig::default()).unwrap();

        let input = ThreeDDesignInput {
            framework: ThreeDFramework::ReactThreeFiber,
            scene_data: ThreeDSceneData::R3FCode(
                r#"
                <Canvas>
                    <Suspense fallback={null}>
                        <mesh position={[0, 0, 0]}>
                            <boxGeometry args={[1, 1, 1]} />
                            <meshStandardMaterial color="cyan" />
                        </mesh>
                    </Suspense>
                </Canvas>
            "#
                .to_string(),
            ),
            performance_targets: ThreeDPerformanceTargets::default(),
            platform: Platform::Web,
        };

        let result = engine.three_d_assessment(input).await.unwrap();

        assert!(result.score > 0.0);
        assert!(result.r3f_instance_count.is_some());
    }
}
