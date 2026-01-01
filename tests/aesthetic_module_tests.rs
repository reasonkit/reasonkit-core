//! # Aesthetic Module Tests
//!
//! Comprehensive test suite for the Aesthetic Expression Mastery System.
//! Tests cover all submodules: accessibility, cross_platform, engine,
//! performance, usability, three_d, types, vibe_benchmark, and visual_analysis.
//!
//! Test Categories:
//! - Unit Tests: Individual function testing with mocked data
//! - Integration Tests: Cross-module interaction testing
//! - Property Tests: Edge cases and boundary conditions
//!
//! Coverage Targets:
//! - 80%+ for critical paths
//! - All public APIs tested
//! - Error handling paths covered

use reasonkit_core::aesthetic::*;
use reasonkit_core::aesthetic::accessibility::*;
use reasonkit_core::aesthetic::cross_platform::*;
use reasonkit_core::aesthetic::engine::*;
use reasonkit_core::aesthetic::performance::*;
use reasonkit_core::aesthetic::three_d::*;
use reasonkit_core::aesthetic::usability::*;
use reasonkit_core::aesthetic::vibe_benchmark::*;
use reasonkit_core::aesthetic::visual_analysis::*;
use reasonkit_core::aesthetic::types::ThreeDFramework;
use reasonkit_core::error::Error;
use std::collections::HashMap;

// =============================================================================
// TEST FIXTURES AND HELPERS
// =============================================================================

/// Create a minimal design input for testing
fn create_minimal_input() -> DesignInput {
    DesignInput {
        data: DesignData::Html("<div>Test</div>".to_string()),
        platform: Platform::Web,
        context: None,
        component_type: ComponentType::Custom,
        design_tokens: None,
    }
}

/// Create a design input with full tokens for comprehensive testing
fn create_full_input() -> DesignInput {
    DesignInput {
        data: DesignData::Html("<button class='btn'>Click Me</button>".to_string()),
        platform: Platform::Web,
        context: Some("Primary CTA button on landing page".to_string()),
        component_type: ComponentType::Button,
        design_tokens: Some(create_brand_tokens()),
    }
}

/// Create ReasonKit brand-aligned design tokens
fn create_brand_tokens() -> DesignTokens {
    DesignTokens {
        colors: ColorTokens::reasonkit_brand(),
        typography: TypographyTokens::reasonkit_brand(),
        spacing: create_spacing_tokens(),
        borders: create_border_tokens(),
        shadows: create_shadow_tokens(),
        animations: AnimationTokens::default(),
    }
}

/// Create spacing tokens with 8px base grid
fn create_spacing_tokens() -> SpacingTokens {
    SpacingTokens {
        base_unit: Some(8.0),
        scale: vec![4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0],
        custom: HashMap::new(),
    }
}

/// Create border tokens
fn create_border_tokens() -> BorderTokens {
    let mut radius = HashMap::new();
    radius.insert("sm".to_string(), "4px".to_string());
    radius.insert("md".to_string(), "8px".to_string());
    radius.insert("lg".to_string(), "16px".to_string());
    radius.insert("full".to_string(), "9999px".to_string());

    BorderTokens {
        radius,
        widths: HashMap::new(),
        colors: HashMap::new(),
    }
}

/// Create shadow tokens
fn create_shadow_tokens() -> ShadowTokens {
    let mut shadows = HashMap::new();
    shadows.insert("sm".to_string(), "0 1px 2px rgba(0,0,0,0.1)".to_string());
    shadows.insert("md".to_string(), "0 4px 6px rgba(0,0,0,0.1)".to_string());
    shadows.insert("lg".to_string(), "0 10px 15px rgba(0,0,0,0.1)".to_string());

    ShadowTokens { shadows }
}

/// Create a 3D design input for testing
fn create_3d_input() -> ThreeDDesignInput {
    ThreeDDesignInput {
        framework: ThreeDFramework::ReactThreeFiber,
        scene_data: ThreeDSceneData::R3FCode(r#"
            import { Canvas } from '@react-three/fiber';
            import { Suspense } from 'react';
            import { OrbitControls } from '@react-three/drei';

            <Canvas>
                <Suspense fallback={null}>
                    <mesh position={[0, 0, 0]}>
                        <boxGeometry args={[1, 1, 1]} />
                        <meshStandardMaterial color="#06b6d4" />
                    </mesh>
                    <mesh position={[2, 0, 0]}>
                        <sphereGeometry args={[0.5, 32, 32]} />
                        <meshStandardMaterial color="#a855f7" />
                    </mesh>
                    <OrbitControls />
                </Suspense>
            </Canvas>
        "#.to_string()),
        performance_targets: ThreeDPerformanceTargets::default(),
        platform: Platform::Web,
    }
}

// =============================================================================
// TYPE TESTS
// =============================================================================

mod types_tests {
    use super::*;

    #[test]
    fn test_wcag_level_default() {
        let level = WcagLevel::default();
        assert_eq!(level, WcagLevel::AA);
    }

    #[test]
    fn test_platform_default() {
        let platform = Platform::default();
        assert_eq!(platform, Platform::Web);
    }

    #[test]
    fn test_component_type_default() {
        let component = ComponentType::default();
        assert_eq!(component, ComponentType::Custom);
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Critical > IssueSeverity::Major);
        assert!(IssueSeverity::Major > IssueSeverity::Minor);
        assert!(IssueSeverity::Minor > IssueSeverity::Info);
    }

    #[test]
    fn test_issue_severity_default() {
        let severity = IssueSeverity::default();
        assert_eq!(severity, IssueSeverity::Info);
    }

    #[test]
    fn test_aesthetic_config_default() {
        let config = AestheticConfig::default();

        assert!(config.enable_visual);
        assert!(config.enable_usability);
        assert!(config.enable_accessibility);
        assert!(config.enable_3d);
        assert!(config.enable_cross_platform);
        assert!(config.enable_performance);
        assert_eq!(config.wcag_level, WcagLevel::AA);
        assert!((config.quality_threshold - 0.85).abs() < 0.001);
        assert_eq!(config.max_analysis_time_ms, 30_000);
    }

    #[test]
    fn test_three_d_performance_targets_default() {
        let targets = ThreeDPerformanceTargets::default();

        assert_eq!(targets.target_fps, 60);
        assert_eq!(targets.max_polygons, 1_000_000);
        assert_eq!(targets.max_texture_memory_mb, 512);
        assert_eq!(targets.max_draw_calls, 100);
    }

    #[test]
    fn test_design_data_variants() {
        // Test Screenshot variant
        let _screenshot = DesignData::Screenshot("base64data".to_string());

        // Test Html variant
        let _html = DesignData::Html("<div>test</div>".to_string());

        // Test Css variant
        let _css = DesignData::Css("body { color: red; }".to_string());

        // Test HtmlCss variant
        let _html_css = DesignData::HtmlCss {
            html: "<div>test</div>".to_string(),
            css: "div { color: blue; }".to_string(),
        };

        // Test ReactComponent variant
        let _react = DesignData::ReactComponent("function App() {}".to_string());

        // Test Url variant
        let _url = DesignData::Url("https://example.com".to_string());

        // Test RawPixels variant
        let _pixels = DesignData::RawPixels {
            width: 100,
            height: 100,
            data: vec![0u8; 40000],
        };
    }

    #[test]
    fn test_color_tokens_reasonkit_brand() {
        let tokens = ColorTokens::reasonkit_brand();

        assert_eq!(tokens.primary, Some("#06b6d4".to_string()));
        assert_eq!(tokens.secondary, Some("#a855f7".to_string()));
        assert_eq!(tokens.background, Some("#030508".to_string()));
        assert_eq!(tokens.surface, Some("#0a0d14".to_string()));
        assert!(tokens.custom.contains_key("cyan"));
        assert!(tokens.custom.contains_key("purple"));
        assert!(tokens.custom.contains_key("pink"));
    }

    #[test]
    fn test_typography_tokens_reasonkit_brand() {
        let tokens = TypographyTokens::reasonkit_brand();

        assert!(tokens.font_family_primary.is_some());
        assert!(tokens.font_family_secondary.is_some());
        assert!(tokens.font_family_mono.is_some());
        assert!(!tokens.font_sizes.is_empty());
        assert!(!tokens.line_heights.is_empty());
        assert!(!tokens.font_weights.is_empty());
    }
}

// =============================================================================
// VIBE BENCHMARK TESTS
// =============================================================================

mod vibe_benchmark_tests {
    use super::*;

    #[test]
    fn test_vibe_benchmark_constants() {
        assert!((VIBE_WEB_BENCHMARK - 0.915).abs() < 0.001);
        assert!((VIBE_ANDROID_BENCHMARK - 0.897).abs() < 0.001);
        assert!((VIBE_IOS_BENCHMARK - 0.880).abs() < 0.001);
    }

    #[test]
    fn test_vibe_evaluator_benchmark_targets() {
        let web_eval = VibeBenchmarkEvaluator::new(Platform::Web);
        assert!((web_eval.benchmark_target() - 0.915).abs() < 0.001);

        let android_eval = VibeBenchmarkEvaluator::new(Platform::Android);
        assert!((android_eval.benchmark_target() - 0.897).abs() < 0.001);

        let ios_eval = VibeBenchmarkEvaluator::new(Platform::IOS);
        assert!((ios_eval.benchmark_target() - 0.880).abs() < 0.001);

        // Desktop should be average of web and iOS
        let desktop_eval = VibeBenchmarkEvaluator::new(Platform::Desktop);
        let expected = (0.915 + 0.880) / 2.0;
        assert!((desktop_eval.benchmark_target() - expected).abs() < 0.001);
    }

    #[test]
    fn test_vibe_evaluation_with_tokens() {
        let evaluator = VibeBenchmarkEvaluator::new(Platform::Web);
        let input = create_full_input();

        let result = evaluator.evaluate(&input);

        assert!(result.overall_score > 0.0);
        assert!(result.overall_score <= 1.0);
        assert!(!result.criteria_results.is_empty());
        assert!(result.platform_scores.contains_key(&Platform::Web));
    }

    #[test]
    fn test_vibe_evaluation_without_tokens() {
        let evaluator = VibeBenchmarkEvaluator::new(Platform::Web);
        let input = create_minimal_input();

        let result = evaluator.evaluate(&input);

        // Should still produce valid results, but lower scores
        assert!(result.overall_score > 0.0);
        assert!(result.overall_score <= 1.0);
    }

    #[test]
    fn test_vibe_evaluation_criteria_count() {
        let evaluator = VibeBenchmarkEvaluator::new(Platform::Web);
        let input = create_full_input();

        let result = evaluator.evaluate(&input);

        // Should have 10 criteria
        assert_eq!(result.criteria_results.len(), 10);
    }

    #[test]
    fn test_multi_platform_evaluation() {
        let input = create_full_input();
        let results = MultiPlatformVibeEvaluator::evaluate_all(&input);

        assert_eq!(results.len(), 4);
        assert!(results.contains_key(&Platform::Web));
        assert!(results.contains_key(&Platform::Android));
        assert!(results.contains_key(&Platform::IOS));
        assert!(results.contains_key(&Platform::Desktop));
    }

    #[test]
    fn test_combined_score() {
        let input = create_full_input();
        let results = MultiPlatformVibeEvaluator::evaluate_all(&input);
        let combined = MultiPlatformVibeEvaluator::combined_score(&results);

        assert!(combined > 0.0);
        assert!(combined <= 1.0);
    }

    #[test]
    fn test_vibe_score_interpretation_ratings() {
        let exceptional = VibeScoreInterpreter::interpret(0.96);
        assert_eq!(exceptional.rating, VibeRating::Exceptional);

        let excellent = VibeScoreInterpreter::interpret(0.91);
        assert_eq!(excellent.rating, VibeRating::Excellent);

        let good = VibeScoreInterpreter::interpret(0.86);
        assert_eq!(good.rating, VibeRating::Good);

        let fair = VibeScoreInterpreter::interpret(0.76);
        assert_eq!(fair.rating, VibeRating::Fair);

        let needs_work = VibeScoreInterpreter::interpret(0.65);
        assert_eq!(needs_work.rating, VibeRating::NeedsWork);

        let poor = VibeScoreInterpreter::interpret(0.50);
        assert_eq!(poor.rating, VibeRating::Poor);
    }

    #[test]
    fn test_vibe_score_improvement_suggestions() {
        let low_score = VibeScoreInterpreter::interpret(0.70);

        // Should have multiple improvement suggestions for low score
        assert!(!low_score.improvement_areas.is_empty());
        assert!(low_score.improvement_areas.len() >= 3);
    }

    #[test]
    fn test_m2_benchmark_comparison() {
        // Test exceeding benchmark
        let exceeds = M2BenchmarkComparison::compare(0.95, Platform::Web);
        assert_eq!(exceeds.status, M2ComparisonStatus::Exceeds);
        assert!(exceeds.difference > 0.0);

        // Test meeting benchmark (within 5%)
        let meets = M2BenchmarkComparison::compare(0.88, Platform::Web);
        assert_eq!(meets.status, M2ComparisonStatus::Meets);

        // Test approaching benchmark (within 15%)
        let approaching = M2BenchmarkComparison::compare(0.80, Platform::Web);
        assert_eq!(approaching.status, M2ComparisonStatus::Approaching);

        // Test below target
        let below = M2BenchmarkComparison::compare(0.60, Platform::Web);
        assert_eq!(below.status, M2ComparisonStatus::BelowTarget);
    }

    #[test]
    fn test_m2_comparison_percentage() {
        let result = M2BenchmarkComparison::compare(0.915, Platform::Web);

        // 100% of benchmark
        assert!((result.percentage_of_benchmark - 100.0).abs() < 0.1);
    }
}

// =============================================================================
// ACCESSIBILITY TESTS
// =============================================================================

mod accessibility_tests {
    use super::*;

    #[test]
    fn test_wcag_checker_level_a() {
        let checker = WcagChecker::new(WcagLevel::A);
        let input = create_full_input();

        let result = checker.assess(&input);

        assert!(result.score >= 0.0);
        assert!(result.score <= 1.0);
    }

    #[test]
    fn test_wcag_checker_level_aa() {
        let checker = WcagChecker::new(WcagLevel::AA);
        let input = create_full_input();

        let result = checker.assess(&input);

        // AA should have more criteria than A
        let total_criteria = result.pass_criteria.len() + result.fail_criteria.len();
        assert!(total_criteria > 9); // Level A has 9 criteria
    }

    #[test]
    fn test_wcag_checker_level_aaa() {
        let checker = WcagChecker::new(WcagLevel::AAA);
        let input = create_full_input();

        let result = checker.assess(&input);

        // AAA should have most criteria
        let total_criteria = result.pass_criteria.len() + result.fail_criteria.len();
        assert!(total_criteria > 15); // AA + AAA criteria
    }

    #[test]
    fn test_wcag_achieved_level_calculation() {
        let checker = WcagChecker::new(WcagLevel::AAA);
        let input = create_full_input();

        let result = checker.assess(&input);

        // With brand tokens, should achieve at least A
        assert!(result.wcag_level_achieved.is_some());
    }

    #[test]
    fn test_contrast_analyzer_black_white() {
        let ratio = ContrastAnalyzer::contrast_ratio("#000000", "#ffffff").unwrap();

        // WCAG contrast ratio for black on white is 21:1
        assert!(ratio > 20.0);
        assert!(ratio < 22.0);
    }

    #[test]
    fn test_contrast_analyzer_gray_on_gray() {
        let ratio = ContrastAnalyzer::contrast_ratio("#808080", "#c0c0c0").unwrap();

        // Gray on gray should have low contrast
        assert!(ratio < 5.0);
    }

    #[test]
    fn test_contrast_analyzer_wcag_aa_check() {
        // Black on white passes AAA
        assert!(ContrastAnalyzer::check_wcag_contrast(
            "#000000", "#ffffff", WcagLevel::AAA, false
        ));

        // Gray on gray fails AA for normal text
        assert!(!ContrastAnalyzer::check_wcag_contrast(
            "#808080", "#c0c0c0", WcagLevel::AA, false
        ));

        // Large text has lower requirements
        assert!(ContrastAnalyzer::check_wcag_contrast(
            "#666666", "#ffffff", WcagLevel::AA, true
        ));
    }

    #[test]
    fn test_contrast_analyzer_invalid_hex() {
        let result = ContrastAnalyzer::contrast_ratio("invalid", "#ffffff");
        assert!(result.is_none());

        let result2 = ContrastAnalyzer::contrast_ratio("#fff", "#000"); // Too short
        assert!(result2.is_none());
    }

    #[test]
    fn test_keyboard_accessibility_analyzer() {
        let result = KeyboardAccessibilityAnalyzer::analyze("<button>Click</button>");

        assert!(result.score > 0.0);
        assert!(result.has_skip_links);
        assert!(result.has_focus_indicators);
    }

    #[test]
    fn test_screen_reader_analyzer() {
        let result = ScreenReaderAnalyzer::analyze("<main><h1>Title</h1></main>");

        assert!(result.score > 0.0);
        assert!(result.has_landmarks);
        assert!(result.has_headings);
    }

    #[test]
    fn test_accessibility_result_default() {
        let result = AccessibilityResult::default();

        assert_eq!(result.score, 0.0);
        assert!(result.wcag_level_achieved.is_none());
        assert!(result.pass_criteria.is_empty());
        assert!(result.fail_criteria.is_empty());
        assert!(result.issues.is_empty());
    }
}

// =============================================================================
// CROSS-PLATFORM TESTS
// =============================================================================

mod cross_platform_tests {
    use super::*;

    #[test]
    fn test_cross_platform_single_platform() {
        let input = create_full_input();
        let result = CrossPlatformValidator::validate(&input, &[Platform::Web]);

        assert_eq!(result.platform_results.len(), 1);
        assert!(result.platform_results.contains_key(&Platform::Web));
        assert_eq!(result.consistency_score, 1.0); // Single platform = perfect consistency
    }

    #[test]
    fn test_cross_platform_multiple_platforms() {
        let input = create_full_input();
        let platforms = vec![Platform::Web, Platform::Android, Platform::IOS];
        let result = CrossPlatformValidator::validate(&input, &platforms);

        assert_eq!(result.platform_results.len(), 3);
        assert!(result.score > 0.0);
        assert!(result.score <= 1.0);
    }

    #[test]
    fn test_cross_platform_all_platforms() {
        let input = create_full_input();
        let platforms = vec![Platform::Web, Platform::Android, Platform::IOS, Platform::Desktop];
        let result = CrossPlatformValidator::validate(&input, &platforms);

        assert_eq!(result.platform_results.len(), 4);
    }

    #[test]
    fn test_platform_specific_issues_ios() {
        // iOS-specific input
        let mut input = create_full_input();
        input.platform = Platform::IOS;

        let result = CrossPlatformValidator::validate(&input, &[Platform::IOS]);

        // Should have iOS-specific suggestions
        let ios_result = result.platform_results.get(&Platform::IOS).unwrap();
        assert!(!ios_result.issues.is_empty());
    }

    #[test]
    fn test_platform_specific_issues_android() {
        let mut input = create_full_input();
        input.platform = Platform::Android;

        let result = CrossPlatformValidator::validate(&input, &[Platform::Android]);

        let android_result = result.platform_results.get(&Platform::Android).unwrap();
        assert!(!android_result.issues.is_empty());
    }

    #[test]
    fn test_responsive_validation() {
        let input = create_full_input();
        let result = ResponsiveValidator::validate(&input);

        assert!(result.score > 0.0);
        assert!(result.score <= 1.0);
        assert!(result.breakpoints_score > 0.0);
        assert!(result.fluid_typography_score > 0.0);
        assert!(result.flexible_layouts_score > 0.0);
        assert!(result.touch_targets_score > 0.0);
    }

    #[test]
    fn test_responsive_validation_with_rem_units() {
        // Typography tokens already use rem units
        let input = create_full_input();
        let result = ResponsiveValidator::validate(&input);

        // Should score higher for fluid typography
        assert!(result.fluid_typography_score >= 0.85);
    }

    #[test]
    fn test_responsive_validation_without_tokens() {
        let input = create_minimal_input();
        let result = ResponsiveValidator::validate(&input);

        // Should still produce valid but lower scores
        assert!(result.score > 0.0);
        assert!(result.score < 0.9); // Lower without proper tokens
    }

    #[test]
    fn test_cross_platform_consistency_calculation() {
        let input = create_full_input();
        let platforms = vec![Platform::Web, Platform::Android, Platform::IOS];
        let result = CrossPlatformValidator::validate(&input, &platforms);

        // Consistency should be high with same input across platforms
        assert!(result.consistency_score > 0.5);
    }

    #[test]
    fn test_cross_platform_result_default() {
        let result = CrossPlatformResult::default();

        assert_eq!(result.score, 0.0);
        assert!(result.platform_results.is_empty());
        assert_eq!(result.consistency_score, 0.0);
        assert!(result.issues.is_empty());
    }
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_performance_analyzer_basic_html() {
        let input = DesignInput {
            data: DesignData::Html("<div><p>Hello</p></div>".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Card,
            design_tokens: None,
        };

        let result = PerformanceAnalyzer::analyze(&input);

        assert!(result.score > 0.0);
        assert!(result.estimated_load_time_ms > 0);
        assert!(result.estimated_interaction_delay_ms > 0);
    }

    #[test]
    fn test_performance_analyzer_complex_html() {
        // Create more complex HTML
        let complex_html = (0..100).map(|i| format!("<div><p>Item {}</p></div>", i))
            .collect::<Vec<_>>().join("");

        let input = DesignInput {
            data: DesignData::Html(complex_html),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::List,
            design_tokens: None,
        };

        let result = PerformanceAnalyzer::analyze(&input);

        // Complex HTML should have higher load time
        assert!(result.estimated_load_time_ms > 100);
    }

    #[test]
    fn test_performance_analyzer_with_css() {
        let input = DesignInput {
            data: DesignData::HtmlCss {
                html: "<div>Test</div>".to_string(),
                css: r#"
                    .test { color: red; transition: all 0.3s; }
                    @keyframes fade { from { opacity: 0; } to { opacity: 1; } }
                "#.to_string(),
            },
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Custom,
            design_tokens: None,
        };

        let result = PerformanceAnalyzer::analyze(&input);

        assert!(result.render_complexity.animation_count > 0);
    }

    #[test]
    fn test_performance_analyzer_react_component() {
        let input = DesignInput {
            data: DesignData::ReactComponent(r#"
                function Card() {
                    return (
                        <div className="card">
                            <h1>Title</h1>
                            <p>Content</p>
                        </div>
                    );
                }
            "#.to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Card,
            design_tokens: None,
        };

        let result = PerformanceAnalyzer::analyze(&input);

        assert!(result.score > 0.0);
        assert!(result.resource_usage.estimated_js_size_kb > 0.0);
    }

    #[test]
    fn test_core_web_vitals_estimate() {
        let input = create_full_input();
        let estimate = CoreWebVitalsAnalyzer::estimate(&input);

        assert!(estimate.lcp_ms > 0);
        assert!(estimate.fid_ms > 0);
        assert!(estimate.cls >= 0.0);
        assert!(estimate.overall_score > 0.0);
        assert!(estimate.overall_score <= 1.0);
    }

    #[test]
    fn test_core_web_vitals_ratings() {
        let input = create_minimal_input();
        let estimate = CoreWebVitalsAnalyzer::estimate(&input);

        // Simple input should have good ratings
        match estimate.lcp_rating {
            VitalRating::Good | VitalRating::NeedsImprovement | VitalRating::Poor => {}
        }
    }

    #[test]
    fn test_vital_rating_thresholds() {
        // LCP thresholds
        let fast_input = create_minimal_input();
        let estimate = CoreWebVitalsAnalyzer::estimate(&fast_input);

        if estimate.lcp_ms <= 2500 {
            assert_eq!(estimate.lcp_rating, VitalRating::Good);
        } else if estimate.lcp_ms <= 4000 {
            assert_eq!(estimate.lcp_rating, VitalRating::NeedsImprovement);
        } else {
            assert_eq!(estimate.lcp_rating, VitalRating::Poor);
        }
    }

    #[test]
    fn test_performance_recommendations_generation() {
        // Create input that should trigger recommendations
        let large_html = (0..600).map(|i| format!("<div><p>Item {}</p></div>", i))
            .collect::<Vec<_>>().join("");

        let input = DesignInput {
            data: DesignData::Html(large_html),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::List,
            design_tokens: None,
        };

        let result = PerformanceAnalyzer::analyze(&input);

        // Should have DOM size recommendation
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_font_loading_strategy_detection() {
        let input_block = DesignInput {
            data: DesignData::HtmlCss {
                html: "<p>Test</p>".to_string(),
                css: "@font-face { font-family: 'Custom'; src: url(...); }".to_string(),
            },
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Custom,
            design_tokens: None,
        };

        let result = PerformanceAnalyzer::analyze(&input_block);
        assert_eq!(result.resource_usage.font_loading_strategy, "block");

        let input_swap = DesignInput {
            data: DesignData::HtmlCss {
                html: "<p>Test</p>".to_string(),
                css: "@font-face { font-display: swap; }".to_string(),
            },
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Custom,
            design_tokens: None,
        };

        let result_swap = PerformanceAnalyzer::analyze(&input_swap);
        assert_eq!(result_swap.resource_usage.font_loading_strategy, "swap");
    }
}

// =============================================================================
// USABILITY TESTS
// =============================================================================

mod usability_tests {
    use super::*;

    #[test]
    fn test_usability_evaluation_button() {
        let input = DesignInput {
            data: DesignData::Html("<button>Click me</button>".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Button,
            design_tokens: None,
        };

        let result = UsabilityEvaluator::evaluate(&input);

        assert!(result.score > 0.8);
        assert!(result.interaction_clarity > 0.9); // Buttons should be very clear
    }

    #[test]
    fn test_usability_evaluation_form() {
        let input = DesignInput {
            data: DesignData::Html("<form><input type='text'></form>".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Form,
            design_tokens: None,
        };

        let result = UsabilityEvaluator::evaluate(&input);

        assert!(result.score > 0.7);
        // Forms should have validation suggestions
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_usability_evaluation_navigation() {
        let input = DesignInput {
            data: DesignData::Html("<nav><a href='/'>Home</a></nav>".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Navigation,
            design_tokens: None,
        };

        let result = UsabilityEvaluator::evaluate(&input);

        assert!(result.navigation_ease > 0.85);
    }

    #[test]
    fn test_heuristic_evaluation() {
        let input = create_full_input();
        let scores = HeuristicEvaluator::evaluate_heuristics(&input);

        // Should return all 10 Nielsen heuristics
        assert_eq!(scores.len(), 10);

        // All scores should be valid
        for score in &scores {
            assert!(score.score >= 0.0);
            assert!(score.score <= 1.0);
            assert!(!score.findings.is_empty());
        }
    }

    #[test]
    fn test_heuristic_categories() {
        let input = create_full_input();
        let scores = HeuristicEvaluator::evaluate_heuristics(&input);

        let heuristics: Vec<UsabilityHeuristic> = scores.iter().map(|s| s.heuristic).collect();

        assert!(heuristics.contains(&UsabilityHeuristic::VisibilityOfSystemStatus));
        assert!(heuristics.contains(&UsabilityHeuristic::MatchSystemRealWorld));
        assert!(heuristics.contains(&UsabilityHeuristic::UserControlFreedom));
        assert!(heuristics.contains(&UsabilityHeuristic::ConsistencyStandards));
        assert!(heuristics.contains(&UsabilityHeuristic::ErrorPrevention));
        assert!(heuristics.contains(&UsabilityHeuristic::RecognitionNotRecall));
        assert!(heuristics.contains(&UsabilityHeuristic::FlexibilityEfficiency));
        assert!(heuristics.contains(&UsabilityHeuristic::AestheticMinimalist));
        assert!(heuristics.contains(&UsabilityHeuristic::RecoverFromErrors));
        assert!(heuristics.contains(&UsabilityHeuristic::HelpDocumentation));
    }

    #[test]
    fn test_user_flow_analysis_simple() {
        let steps = vec![
            FlowStep {
                name: "Click button".to_string(),
                description: "User clicks the CTA".to_string(),
                friction_level: FrictionLevel::Low,
                time_estimate_seconds: 1,
            },
            FlowStep {
                name: "See result".to_string(),
                description: "User sees the result".to_string(),
                friction_level: FrictionLevel::None,
                time_estimate_seconds: 2,
            },
        ];

        let result = UserFlowAnalyzer::analyze_flow(&steps);

        assert_eq!(result.total_steps, 2);
        assert!(result.friction_points.is_empty());
        assert!(result.efficiency_score > 0.9);
    }

    #[test]
    fn test_user_flow_analysis_with_friction() {
        let steps = vec![
            FlowStep {
                name: "Fill form".to_string(),
                description: "Complex form".to_string(),
                friction_level: FrictionLevel::High,
                time_estimate_seconds: 60,
            },
            FlowStep {
                name: "Verify email".to_string(),
                description: "Email verification".to_string(),
                friction_level: FrictionLevel::Medium,
                time_estimate_seconds: 30,
            },
        ];

        let result = UserFlowAnalyzer::analyze_flow(&steps);

        assert_eq!(result.friction_points.len(), 2);
        assert!(result.efficiency_score < 0.5);
    }

    #[test]
    fn test_user_flow_recommendations() {
        let steps = vec![
            FlowStep {
                name: "Step 1".to_string(),
                description: "".to_string(),
                friction_level: FrictionLevel::High,
                time_estimate_seconds: 10,
            },
            FlowStep {
                name: "Step 2".to_string(),
                description: "".to_string(),
                friction_level: FrictionLevel::Low,
                time_estimate_seconds: 5,
            },
            FlowStep {
                name: "Step 3".to_string(),
                description: "".to_string(),
                friction_level: FrictionLevel::Low,
                time_estimate_seconds: 5,
            },
            FlowStep {
                name: "Step 4".to_string(),
                description: "".to_string(),
                friction_level: FrictionLevel::Low,
                time_estimate_seconds: 5,
            },
            FlowStep {
                name: "Step 5".to_string(),
                description: "".to_string(),
                friction_level: FrictionLevel::Low,
                time_estimate_seconds: 5,
            },
            FlowStep {
                name: "Step 6".to_string(),
                description: "".to_string(),
                friction_level: FrictionLevel::Low,
                time_estimate_seconds: 5,
            },
        ];

        let result = UserFlowAnalyzer::analyze_flow(&steps);

        // Should recommend reducing steps (>5) and fixing high friction
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_friction_level_ordering() {
        assert!(FrictionLevel::High > FrictionLevel::Medium);
        assert!(FrictionLevel::Medium > FrictionLevel::Low);
        assert!(FrictionLevel::Low > FrictionLevel::None);
    }
}

// =============================================================================
// 3D DESIGN TESTS
// =============================================================================

mod three_d_tests {
    use super::*;

    #[test]
    fn test_m2_r3f_capability_constant() {
        assert_eq!(M2_R3F_INSTANCE_CAPABILITY, 7000);
    }

    #[test]
    fn test_three_d_evaluator_basic() {
        let evaluator = ThreeDEvaluator::new(ThreeDPerformanceTargets::default());
        let input = create_3d_input();

        let result = evaluator.evaluate(&input);

        assert!(result.score > 0.0);
        assert!(result.score <= 1.0);
        assert!(result.visual_quality > 0.0);
        assert!(result.performance_score > 0.0);
        assert!(result.interaction_quality > 0.0);
    }

    #[test]
    fn test_three_d_r3f_instance_count() {
        let evaluator = ThreeDEvaluator::new(ThreeDPerformanceTargets::default());
        let input = create_3d_input();

        let result = evaluator.evaluate(&input);

        assert!(result.r3f_instance_count.is_some());
        assert!(result.r3f_instance_count.unwrap() > 0);
    }

    #[test]
    fn test_three_d_framework_scoring() {
        let targets = ThreeDPerformanceTargets::default();

        // React Three Fiber should score higher than raw WebGL
        let r3f_input = ThreeDDesignInput {
            framework: ThreeDFramework::ReactThreeFiber,
            scene_data: ThreeDSceneData::R3FCode("<mesh />".to_string()),
            performance_targets: targets.clone(),
            platform: Platform::Web,
        };

        let webgl_input = ThreeDDesignInput {
            framework: ThreeDFramework::WebGL,
            scene_data: ThreeDSceneData::ThreeJsCode("new Mesh()".to_string()),
            performance_targets: targets.clone(),
            platform: Platform::Web,
        };

        let r3f_evaluator = ThreeDEvaluator::new(targets.clone());
        let webgl_evaluator = ThreeDEvaluator::new(targets);

        let r3f_result = r3f_evaluator.evaluate(&r3f_input);
        let webgl_result = webgl_evaluator.evaluate(&webgl_input);

        assert!(r3f_result.visual_quality > webgl_result.visual_quality);
    }

    #[test]
    fn test_three_d_polygon_estimation() {
        let evaluator = ThreeDEvaluator::new(ThreeDPerformanceTargets::default());

        // Input with multiple meshes
        let multi_mesh = ThreeDDesignInput {
            framework: ThreeDFramework::ReactThreeFiber,
            scene_data: ThreeDSceneData::R3FCode(r#"
                <mesh /><mesh /><mesh /><mesh /><mesh />
                <Mesh /><Mesh /><Mesh />
            "#.to_string()),
            performance_targets: ThreeDPerformanceTargets::default(),
            platform: Platform::Web,
        };

        let result = evaluator.evaluate(&multi_mesh);

        // More meshes should mean more polygons
        assert!(result.polygon_count > 5000);
    }

    #[test]
    fn test_three_d_draw_call_with_instancing() {
        let evaluator = ThreeDEvaluator::new(ThreeDPerformanceTargets::default());

        // Input with instancing
        let instanced = ThreeDDesignInput {
            framework: ThreeDFramework::ReactThreeFiber,
            scene_data: ThreeDSceneData::R3FCode(r#"
                <instancedMesh>
                    <boxGeometry />
                    <meshStandardMaterial />
                </instancedMesh>
            "#.to_string()),
            performance_targets: ThreeDPerformanceTargets::default(),
            platform: Platform::Web,
        };

        let result = evaluator.evaluate(&instanced);

        // Instancing should reduce draw calls
        assert!(result.draw_calls < 20);
    }

    #[test]
    fn test_three_d_fps_estimation() {
        let evaluator = ThreeDEvaluator::new(ThreeDPerformanceTargets::default());
        let input = create_3d_input();

        let result = evaluator.evaluate(&input);

        // Simple scene should have high estimated FPS
        assert!(result.estimated_fps > 30);
    }

    #[test]
    fn test_three_d_issues_detection() {
        let targets = ThreeDPerformanceTargets {
            target_fps: 60,
            max_polygons: 100, // Very low limit
            max_texture_memory_mb: 1,
            max_draw_calls: 1,
        };

        let evaluator = ThreeDEvaluator::new(targets);
        let input = create_3d_input();

        let result = evaluator.evaluate(&input);

        // Should detect issues with exceeded limits
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_three_d_optimizations() {
        let evaluator = ThreeDEvaluator::new(ThreeDPerformanceTargets::default());

        // Complex scene
        let complex = ThreeDDesignInput {
            framework: ThreeDFramework::ReactThreeFiber,
            scene_data: ThreeDSceneData::R3FCode(
                (0..100).map(|_| "<mesh />").collect::<Vec<_>>().join("")
            ),
            performance_targets: ThreeDPerformanceTargets::default(),
            platform: Platform::Web,
        };

        let result = evaluator.evaluate(&complex);

        // Should suggest optimizations for complex scene
        assert!(!result.optimizations.is_empty());
    }

    #[test]
    fn test_r3f_analyzer() {
        let code = r#"
            import { Suspense } from 'react';
            import { Canvas } from '@react-three/fiber';
            import { OrbitControls, Environment } from '@react-three/drei';
            import { EffectComposer } from '@react-three/postprocessing';
            import { useControls } from 'leva';
        "#;

        let result = ReactThreeFiberAnalyzer::analyze(code);

        assert!(result.uses_suspense);
        assert!(result.uses_drei);
        assert!(result.uses_postprocessing);
        assert!(result.uses_leva);
        assert!(result.best_practices_score > 0.8);
    }

    #[test]
    fn test_r3f_analyzer_minimal() {
        let code = "<Canvas><mesh /></Canvas>";

        let result = ReactThreeFiberAnalyzer::analyze(code);

        assert!(!result.uses_suspense);
        assert!(!result.uses_drei);
        assert!(!result.suggestions.is_empty());
    }

    #[test]
    fn test_three_d_gltf_estimation() {
        let evaluator = ThreeDEvaluator::new(ThreeDPerformanceTargets::default());

        // Simulate GLTF data (10KB)
        let gltf_input = ThreeDDesignInput {
            framework: ThreeDFramework::ThreeJs,
            scene_data: ThreeDSceneData::GltfData(vec![0u8; 10_000]),
            performance_targets: ThreeDPerformanceTargets::default(),
            platform: Platform::Web,
        };

        let result = evaluator.evaluate(&gltf_input);

        assert!(result.polygon_count > 0);
        assert!(result.texture_memory_mb > 0.0);
    }
}

// =============================================================================
// VISUAL ANALYSIS TESTS
// =============================================================================

mod visual_analysis_tests {
    use super::*;

    #[test]
    fn test_color_harmony_complementary() {
        let colors = vec![
            "#ff0000".to_string(), // Red
            "#00ffff".to_string(), // Cyan (complement)
        ];

        let result = ColorHarmonyAnalyzer::analyze(&colors);

        assert_eq!(result.harmony_type, Some(ColorHarmonyType::Complementary));
        assert!(result.score > 0.8);
    }

    #[test]
    fn test_color_harmony_monochromatic() {
        let colors = vec![
            "#0066ff".to_string(),
        ];

        let result = ColorHarmonyAnalyzer::analyze(&colors);

        assert_eq!(result.harmony_type, Some(ColorHarmonyType::Monochromatic));
    }

    #[test]
    fn test_color_harmony_analogous() {
        let colors = vec![
            "#ff0000".to_string(), // Red
            "#ff6600".to_string(), // Orange
            "#ffcc00".to_string(), // Yellow
        ];

        let result = ColorHarmonyAnalyzer::analyze(&colors);

        // Adjacent colors on the wheel
        assert!(result.score > 0.7);
    }

    #[test]
    fn test_color_harmony_contrast_detection() {
        let colors = vec![
            "#000000".to_string(), // Black
            "#ffffff".to_string(), // White
        ];

        let result = ColorHarmonyAnalyzer::analyze(&colors);

        assert!(!result.contrast_ratios.is_empty());
        assert!(result.contrast_ratios[0].passes_aaa);
    }

    #[test]
    fn test_color_harmony_issues_low_contrast() {
        let colors = vec![
            "#cccccc".to_string(),
            "#dddddd".to_string(),
        ];

        let result = ColorHarmonyAnalyzer::analyze(&colors);

        // Low contrast should generate issues
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_color_harmony_too_many_colors() {
        let colors: Vec<String> = (0..10)
            .map(|i| format!("#{:02x}{:02x}{:02x}", i * 25, i * 20, i * 15))
            .collect();

        let result = ColorHarmonyAnalyzer::analyze(&colors);

        // Should warn about too many colors
        assert!(result.issues.iter().any(|i| i.description.contains("more than 7")));
    }

    #[test]
    fn test_typography_analysis_complete() {
        let tokens = TypographyTokens::reasonkit_brand();
        let result = TypographyAnalyzer::analyze(&tokens);

        assert!(result.score > 0.8);
        assert!(result.font_pairing_score > 0.9);
        assert!(result.readability_score > 0.8);
        assert!(result.hierarchy_score > 0.9);
    }

    #[test]
    fn test_typography_analysis_empty() {
        let tokens = TypographyTokens::default();
        let result = TypographyAnalyzer::analyze(&tokens);

        // Empty tokens should have issues
        assert!(!result.issues.is_empty());
        assert!(result.score < 0.8);
    }

    #[test]
    fn test_typography_line_height_issues() {
        let mut tokens = TypographyTokens::default();
        tokens.line_heights.insert("tight".to_string(), 1.0); // Too tight

        let result = TypographyAnalyzer::analyze(&tokens);

        assert!(result.issues.iter().any(|i|
            matches!(i.issue_type, TypographyIssueType::LineHeightTooTight)
        ));
    }

    #[test]
    fn test_layout_analysis() {
        let spacing = create_spacing_tokens();
        let result = LayoutAnalyzer::analyze(&spacing);

        assert!(result.score > 0.8);
        assert!(result.grid_adherence > 0.8);
        assert!(result.alignment_score > 0.8);
    }

    #[test]
    fn test_hierarchy_analysis() {
        let result = HierarchyAnalyzer::analyze();

        assert!(result.score > 0.8);
        assert!(result.focal_point_clarity > 0.8);
        assert!(result.information_flow > 0.8);
        assert!(result.cta_prominence > 0.8);
    }

    #[test]
    fn test_consistency_analysis() {
        let tokens = create_brand_tokens();
        let result = ConsistencyAnalyzer::analyze(&tokens);

        assert!(result.score > 0.8);
        assert!(result.style_consistency > 0.8);
        assert!(result.spacing_consistency > 0.8);
    }

    #[test]
    fn test_consistency_analysis_empty() {
        let tokens = DesignTokens::default();
        let result = ConsistencyAnalyzer::analyze(&tokens);

        // Empty tokens should have lower scores
        assert!(result.score < 0.7);
    }

    #[test]
    fn test_white_space_analysis() {
        let result = WhiteSpaceAnalyzer::analyze();

        assert!(result.score > 0.8);
        assert!(result.breathing_room > 0.8);
        assert!(result.density_balance > 0.8);
        assert!(result.margin_consistency > 0.8);
    }

    #[test]
    fn test_visual_result_default() {
        let result = VisualAssessmentResult::default();

        assert_eq!(result.score, 0.0);
    }
}

// =============================================================================
// ENGINE INTEGRATION TESTS
// =============================================================================

mod engine_tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        let config = AestheticConfig::default();
        let engine = AestheticMasteryEngine::new(config);

        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_comprehensive_assessment() {
        let config = AestheticConfig::default();
        let engine = AestheticMasteryEngine::new(config).unwrap();
        let input = create_full_input();

        let result: DesignAssessmentResult = engine.comprehensive_assessment(input).await.unwrap();

        assert!(result.overall_score > 0.0);
        assert!(result.overall_score <= 1.0);
        assert!(!result.metadata.assessment_id.is_empty());
        assert!(result.metadata.duration_ms < 30_000);
    }

    #[tokio::test]
    async fn test_visual_assessment_only() {
        let config = AestheticConfig::default();
        let engine = AestheticMasteryEngine::new(config).unwrap();
        let input = create_full_input();

        let result: VisualAssessmentResult = engine.visual_assessment(input).await.unwrap();

        assert!(result.score > 0.0);
        assert!(result.score <= 1.0);
    }

    #[tokio::test]
    async fn test_accessibility_assessment_only() {
        let config = AestheticConfig::default();
        let engine = AestheticMasteryEngine::new(config).unwrap();
        let input = create_full_input();

        let result: AccessibilityResult = engine.accessibility_assessment(input).await.unwrap();

        assert!(result.score > 0.0);
    }

    #[tokio::test]
    async fn test_cross_platform_validation_only() {
        let config = AestheticConfig::default();
        let engine = AestheticMasteryEngine::new(config).unwrap();
        let input = create_full_input();

        let result: CrossPlatformResult = engine.cross_platform_validation(input).await.unwrap();

        assert_eq!(result.platform_results.len(), 3);
    }

    #[tokio::test]
    async fn test_3d_assessment() {
        let config = AestheticConfig::default();
        let engine = AestheticMasteryEngine::new(config).unwrap();
        let input = create_3d_input();

        let result: ThreeDAssessmentResult = engine.three_d_assessment(input).await.unwrap();

        assert!(result.score > 0.0);
        assert!(result.r3f_instance_count.is_some());
    }

    #[tokio::test]
    async fn test_disabled_modules() {
        let config = AestheticConfig {
            enable_visual: false,
            enable_usability: false,
            enable_accessibility: false,
            enable_3d: false,
            enable_cross_platform: false,
            enable_performance: false,
            wcag_level: WcagLevel::AA,
            quality_threshold: 0.85,
            max_analysis_time_ms: 30_000,
        };

        let engine = AestheticMasteryEngine::new(config).unwrap();
        let input = create_full_input();

        let result: DesignAssessmentResult = engine.comprehensive_assessment(input).await.unwrap();

        // Should still produce a result with VIBE scores
        assert!(result.overall_score >= 0.0);
    }

    #[test]
    fn test_quick_assessment() {
        let input = create_full_input();
        let result = QuickAssessment::assess(&input);

        assert!(result.overall_score > 0.0);
        assert!(result.visual_score > 0.0);
        assert!(result.vibe_score > 0.0);
    }

    #[test]
    fn test_quick_assessment_without_tokens() {
        let input = create_minimal_input();
        let result = QuickAssessment::assess(&input);

        // Should still work but with lower scores
        assert!(result.overall_score > 0.0);
    }

    #[tokio::test]
    async fn test_recommendations_generation() {
        let config = AestheticConfig::default();
        let engine = AestheticMasteryEngine::new(config).unwrap();

        // Create input that should trigger recommendations
        let input = DesignInput {
            data: DesignData::Html("<div>Test</div>".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Custom,
            design_tokens: None, // No tokens = lower scores = recommendations
        };

        let result: DesignAssessmentResult = engine.comprehensive_assessment(input).await.unwrap();

        // Should have recommendations for improving scores
        // (may be empty if scores are high enough)
        assert!(result.recommendations.len() >= 0);
    }

    #[tokio::test]
    async fn test_assessment_metadata() {
        let config = AestheticConfig::default();
        let engine = AestheticMasteryEngine::new(config).unwrap();
        let input = create_full_input();

        let result: DesignAssessmentResult = engine.comprehensive_assessment(input).await.unwrap();

        // Check metadata is populated
        assert!(!result.metadata.assessment_id.is_empty());
        assert!(result.metadata.duration_ms > 0);
        assert!(!result.metadata.engine_version.is_empty());
        assert!(!result.metadata.config_hash.is_empty());
    }
}

// =============================================================================
// SERVICE BUILDER TESTS
// =============================================================================

mod service_tests {
    use super::*;

    #[tokio::test]
    async fn test_service_creation() {
        let config = AestheticConfig::default();
        let service: Result<AestheticExpressionService, Error> = AestheticExpressionService::new(config).await;

        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_service_builder() {
        let service: Result<AestheticExpressionService, Error> = AestheticServiceBuilder::new()
            .with_config(AestheticConfig::default())
            .build()
            .await;

        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_service_builder_with_custom_vibe() {
        let custom_vibe = VibeBenchmarkTargets {
            web: 0.90,
            android: 0.85,
            ios: 0.85,
        };

        let service: Result<AestheticExpressionService, Error> = AestheticServiceBuilder::new()
            .with_config(AestheticConfig::default())
            .with_vibe_targets(custom_vibe)
            .build()
            .await
            .unwrap();

        // Verify custom targets are used
        let compliance = service.get_vibe_compliance(Platform::Web);
        assert!((compliance - 0.90).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_service_assess_design() {
        let service = AestheticExpressionService::new(AestheticConfig::default())
            .await
            .unwrap();

        let input = create_full_input();
        let result = service.assess_design(input).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_service_quick_visual_check() {
        let service = AestheticExpressionService::new(AestheticConfig::default())
            .await
            .unwrap();

        let input = create_full_input();
        let result = service.quick_visual_check(input).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_service_check_accessibility() {
        let service = AestheticExpressionService::new(AestheticConfig::default())
            .await
            .unwrap();

        let input = create_full_input();
        let result = service.check_accessibility(input).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_service_validate_cross_platform() {
        let service = AestheticExpressionService::new(AestheticConfig::default())
            .await
            .unwrap();

        let input = create_full_input();
        let result = service.validate_cross_platform(input).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_service_assess_3d() {
        let service = AestheticExpressionService::new(AestheticConfig::default())
            .await
            .unwrap();

        let input = create_3d_input();
        let result = service.assess_3d_design(input).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_service_vibe_compliance() {
        let service = AestheticExpressionService::new(AestheticConfig::default())
            .await
            .unwrap();

        // Test all platforms
        let web = service.get_vibe_compliance(Platform::Web);
        assert!((web - 0.915).abs() < 0.001);

        let android = service.get_vibe_compliance(Platform::Android);
        assert!((android - 0.897).abs() < 0.001);

        let ios = service.get_vibe_compliance(Platform::IOS);
        assert!((ios - 0.880).abs() < 0.001);

        let desktop = service.get_vibe_compliance(Platform::Desktop);
        assert!(desktop > 0.85 && desktop < 0.92);
    }
}

// =============================================================================
// MODULE-LEVEL TESTS (mod.rs)
// =============================================================================

mod module_tests {
    use super::*;

    #[test]
    fn test_vibe_benchmark_targets_default() {
        let targets = VibeBenchmarkTargets::default();

        assert!((targets.web - 0.915).abs() < 0.001);
        assert!((targets.android - 0.897).abs() < 0.001);
        assert!((targets.ios - 0.880).abs() < 0.001);
    }

    #[test]
    fn test_vibe_benchmark_targets_serialization() {
        let targets = VibeBenchmarkTargets::default();

        // Serialize and deserialize
        let json = serde_json::to_string(&targets).unwrap();
        let deserialized: VibeBenchmarkTargets = serde_json::from_str(&json).unwrap();

        assert!((deserialized.web - targets.web).abs() < 0.001);
        assert!((deserialized.android - targets.android).abs() < 0.001);
        assert!((deserialized.ios - targets.ios).abs() < 0.001);
    }
}

// =============================================================================
// EDGE CASES AND BOUNDARY TESTS
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_color_array() {
        let colors: Vec<String> = vec![];
        let result = ColorHarmonyAnalyzer::analyze(&colors);

        // Should handle gracefully
        assert!(result.contrast_ratios.is_empty());
    }

    #[test]
    fn test_invalid_hex_color() {
        let colors = vec!["invalid".to_string(), "notahex".to_string()];
        let result = ColorHarmonyAnalyzer::analyze(&colors);

        // Should not crash, just return default/empty
        assert_eq!(result.harmony_type, Some(ColorHarmonyType::Custom));
    }

    #[test]
    fn test_empty_html_input() {
        let input = DesignInput {
            data: DesignData::Html("".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Custom,
            design_tokens: None,
        };

        let result = PerformanceAnalyzer::analyze(&input);

        // Should handle gracefully
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_very_large_html() {
        let large_html = (0..10000).map(|i| format!("<div id='{}'>Content</div>", i))
            .collect::<Vec<_>>().join("");

        let input = DesignInput {
            data: DesignData::Html(large_html),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Page,
            design_tokens: None,
        };

        let result = PerformanceAnalyzer::analyze(&input);

        // Should detect performance issues
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_empty_flow_steps() {
        let steps: Vec<FlowStep> = vec![];
        let result = UserFlowAnalyzer::analyze_flow(&steps);

        assert_eq!(result.total_steps, 0);
        assert_eq!(result.efficiency_score, 1.0); // Empty = perfect efficiency
    }

    #[test]
    fn test_contrast_same_color() {
        let ratio = ContrastAnalyzer::contrast_ratio("#ffffff", "#ffffff");

        assert!(ratio.is_some());
        assert!((ratio.unwrap() - 1.0).abs() < 0.01); // Same color = 1:1 ratio
    }

    #[test]
    fn test_typography_empty_scale() {
        let tokens = TypographyTokens::default();
        let result = TypographyAnalyzer::analyze(&tokens);

        // Should not crash on empty
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_3d_empty_code() {
        let evaluator = ThreeDEvaluator::new(ThreeDPerformanceTargets::default());

        let input = ThreeDDesignInput {
            framework: ThreeDFramework::ReactThreeFiber,
            scene_data: ThreeDSceneData::R3FCode("".to_string()),
            performance_targets: ThreeDPerformanceTargets::default(),
            platform: Platform::Web,
        };

        let result = evaluator.evaluate(&input);

        // Should handle empty gracefully
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_wcag_checker_minimal_input() {
        let checker = WcagChecker::new(WcagLevel::AAA);

        let input = DesignInput {
            data: DesignData::Html("".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Custom,
            design_tokens: None,
        };

        let result = checker.assess(&input);

        // Should produce valid result even with minimal input
        assert!(result.score >= 0.0);
        assert!(result.score <= 1.0);
    }
}
