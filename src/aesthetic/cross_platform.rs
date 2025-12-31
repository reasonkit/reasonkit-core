//! # Cross-Platform Design Validation
//!
//! Validates designs against platform-specific guidelines for Web, iOS, and Android.

use super::types::*;
use std::collections::HashMap;

/// Cross-platform design validator
pub struct CrossPlatformValidator;

impl CrossPlatformValidator {
    /// Validate design across multiple platforms
    pub fn validate(input: &DesignInput, platforms: &[Platform]) -> CrossPlatformResult {
        let mut platform_results = HashMap::new();
        let mut all_issues = Vec::new();

        for platform in platforms {
            let result = Self::validate_for_platform(input, *platform);
            all_issues.extend(result.issues.iter().map(|i| CrossPlatformIssue {
                severity: IssueSeverity::Minor,
                platforms_affected: vec![*platform],
                description: i.clone(),
                suggestion: format!(
                    "Review {} design guidelines",
                    Self::platform_name(*platform)
                ),
            }));
            platform_results.insert(*platform, result);
        }

        // Calculate consistency across platforms
        let consistency_score = Self::calculate_consistency(&platform_results);

        // Overall score
        let avg_compliance: f64 = platform_results
            .values()
            .map(|r| r.compliance_score)
            .sum::<f64>()
            / platform_results.len() as f64;

        let score = (avg_compliance + consistency_score) / 2.0;

        CrossPlatformResult {
            score,
            platform_results,
            consistency_score,
            issues: all_issues,
        }
    }

    /// Validate for a specific platform
    fn validate_for_platform(input: &DesignInput, platform: Platform) -> PlatformSpecificResult {
        let validator: Box<dyn PlatformGuidelines> = match platform {
            Platform::Web => Box::new(WebGuidelines),
            Platform::IOS => Box::new(IOSGuidelines),
            Platform::Android => Box::new(AndroidGuidelines),
            Platform::Desktop => Box::new(DesktopGuidelines),
        };

        validator.validate(input)
    }

    /// Calculate consistency score across platforms
    fn calculate_consistency(results: &HashMap<Platform, PlatformSpecificResult>) -> f64 {
        if results.len() < 2 {
            return 1.0;
        }

        let scores: Vec<f64> = results.values().map(|r| r.compliance_score).collect();

        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance: f64 =
            scores.iter().map(|&s| (s - avg).powi(2)).sum::<f64>() / scores.len() as f64;

        // Lower variance = higher consistency
        (1.0 - variance.sqrt()).max(0.0)
    }

    fn platform_name(platform: Platform) -> &'static str {
        match platform {
            Platform::Web => "Web",
            Platform::IOS => "iOS HIG",
            Platform::Android => "Material Design",
            Platform::Desktop => "Desktop",
        }
    }
}

/// Platform guidelines trait
trait PlatformGuidelines {
    fn validate(&self, input: &DesignInput) -> PlatformSpecificResult;
}

/// Web platform guidelines
struct WebGuidelines;

impl PlatformGuidelines for WebGuidelines {
    fn validate(&self, input: &DesignInput) -> PlatformSpecificResult {
        let mut issues = Vec::new();
        let mut compliance_score = 0.85;
        let mut guideline_adherence = 0.88;
        let conventions_score = 0.90;

        // Check for responsive design patterns
        if let Some(tokens) = &input.design_tokens {
            if tokens.spacing.scale.is_empty() {
                issues.push("No spacing scale defined - may affect responsiveness".to_string());
                compliance_score -= 0.05;
            }
        }

        // Component-specific checks
        match input.component_type {
            ComponentType::Navigation => {
                guideline_adherence = 0.92; // Navigation is well-defined for web
            }
            ComponentType::Form => {
                guideline_adherence = 0.85;
                issues.push("Ensure form labels are visible (not placeholders only)".to_string());
            }
            _ => {}
        }

        PlatformSpecificResult {
            platform: Platform::Web,
            compliance_score,
            design_guideline_adherence: guideline_adherence,
            platform_conventions_score: conventions_score,
            issues,
        }
    }
}

/// iOS Human Interface Guidelines validator
struct IOSGuidelines;

impl PlatformGuidelines for IOSGuidelines {
    fn validate(&self, input: &DesignInput) -> PlatformSpecificResult {
        let mut issues = Vec::new();
        let mut compliance_score = 0.85;

        // iOS-specific checks
        if let Some(tokens) = &input.design_tokens {
            // Check for iOS system colors
            if let Some(primary) = &tokens.colors.primary {
                // iOS prefers specific blue: #007AFF
                if primary != "#007AFF" && primary != "#0A84FF" {
                    issues.push(
                        "Consider using iOS system blue (#007AFF) for primary actions".to_string(),
                    );
                }
            }

            // Check font
            if let Some(font) = &tokens.typography.font_family_primary {
                if !font.contains("SF Pro") && !font.contains("system") {
                    issues.push("iOS recommends SF Pro or system font".to_string());
                    compliance_score -= 0.05;
                }
            }
        }

        // Touch target sizes
        match input.component_type {
            ComponentType::Button => {
                issues.push("Ensure minimum 44x44pt touch target".to_string());
            }
            ComponentType::Navigation => {
                issues.push("Tab bar should have 49pt height on iPhone".to_string());
            }
            _ => {}
        }

        // iOS guideline adherence scoring
        let guideline_adherence = match input.component_type {
            ComponentType::Navigation => 0.90,
            ComponentType::Button => 0.88,
            ComponentType::Modal => 0.92, // iOS modals are well-defined
            _ => 0.85,
        };

        PlatformSpecificResult {
            platform: Platform::IOS,
            compliance_score,
            design_guideline_adherence: guideline_adherence,
            platform_conventions_score: 0.87,
            issues,
        }
    }
}

/// Material Design (Android) guidelines validator
struct AndroidGuidelines;

impl PlatformGuidelines for AndroidGuidelines {
    fn validate(&self, input: &DesignInput) -> PlatformSpecificResult {
        let mut issues = Vec::new();
        let mut compliance_score = 0.85;

        // Material Design checks
        if let Some(tokens) = &input.design_tokens {
            // Check for Material color scheme
            let has_elevation = !tokens.shadows.shadows.is_empty();
            if !has_elevation {
                issues.push("Material Design uses elevation/shadows for depth".to_string());
                compliance_score -= 0.05;
            }

            // Check spacing follows 8dp grid
            if let Some(base) = tokens.spacing.base_unit {
                if (base % 8.0).abs() > 0.01 {
                    issues.push("Material Design uses 8dp grid system".to_string());
                }
            }

            // Check for Roboto font
            if let Some(font) = &tokens.typography.font_family_primary {
                if !font.contains("Roboto") {
                    issues.push("Material Design recommends Roboto font".to_string());
                }
            }
        }

        // Touch targets (48dp minimum)
        match input.component_type {
            ComponentType::Button => {
                issues.push("Ensure minimum 48dp touch target".to_string());
            }
            ComponentType::Form => {
                issues.push("Text fields should use outlined or filled style".to_string());
            }
            _ => {}
        }

        let guideline_adherence = match input.component_type {
            ComponentType::Button => 0.90, // FAB is well-defined
            ComponentType::Card => 0.92,   // Cards are core to Material
            ComponentType::Navigation => 0.88,
            _ => 0.85,
        };

        PlatformSpecificResult {
            platform: Platform::Android,
            compliance_score,
            design_guideline_adherence: guideline_adherence,
            platform_conventions_score: 0.86,
            issues,
        }
    }
}

/// Desktop platform guidelines
struct DesktopGuidelines;

impl PlatformGuidelines for DesktopGuidelines {
    fn validate(&self, input: &DesignInput) -> PlatformSpecificResult {
        let mut issues = Vec::new();

        // Desktop-specific considerations
        match input.component_type {
            ComponentType::Navigation => {
                issues.push("Consider keyboard shortcuts for navigation".to_string());
            }
            ComponentType::Form => {
                issues.push("Tab order should be logical for keyboard users".to_string());
            }
            _ => {}
        }

        // Hover states are important on desktop
        issues.push("Ensure all interactive elements have hover states".to_string());

        PlatformSpecificResult {
            platform: Platform::Desktop,
            compliance_score: 0.88,
            design_guideline_adherence: 0.85,
            platform_conventions_score: 0.87,
            issues,
        }
    }
}

/// Responsive design validator
pub struct ResponsiveValidator;

impl ResponsiveValidator {
    /// Validate responsive design patterns
    pub fn validate(input: &DesignInput) -> ResponsiveValidationResult {
        let breakpoints = Self::check_breakpoints(input);
        let fluid_typography = Self::check_fluid_typography(input);
        let flexible_layouts = Self::check_flexible_layouts(input);
        let touch_targets = Self::check_touch_targets(input);

        let score = (breakpoints + fluid_typography + flexible_layouts + touch_targets) / 4.0;

        ResponsiveValidationResult {
            score,
            breakpoints_score: breakpoints,
            fluid_typography_score: fluid_typography,
            flexible_layouts_score: flexible_layouts,
            touch_targets_score: touch_targets,
            recommendations: Self::generate_recommendations(
                breakpoints,
                fluid_typography,
                flexible_layouts,
                touch_targets,
            ),
        }
    }

    fn check_breakpoints(_input: &DesignInput) -> f64 {
        // Would check CSS for breakpoints
        0.85
    }

    fn check_fluid_typography(input: &DesignInput) -> f64 {
        if let Some(tokens) = &input.design_tokens {
            // Check for relative units in font sizes
            let has_relative = tokens
                .typography
                .font_sizes
                .values()
                .any(|s| s.contains("rem") || s.contains("em") || s.contains("vw"));

            if has_relative {
                0.90
            } else {
                0.70
            }
        } else {
            0.75
        }
    }

    fn check_flexible_layouts(input: &DesignInput) -> f64 {
        if let Some(tokens) = &input.design_tokens {
            if tokens.spacing.base_unit.is_some() || !tokens.spacing.scale.is_empty() {
                0.88
            } else {
                0.70
            }
        } else {
            0.75
        }
    }

    fn check_touch_targets(_input: &DesignInput) -> f64 {
        0.85 // Would check min-height/width of interactive elements
    }

    fn generate_recommendations(
        breakpoints: f64,
        typography: f64,
        layouts: f64,
        touch: f64,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        if breakpoints < 0.80 {
            recs.push("Define breakpoints for sm, md, lg, xl viewports".to_string());
        }
        if typography < 0.80 {
            recs.push("Use relative units (rem) for font sizes".to_string());
        }
        if layouts < 0.80 {
            recs.push("Implement a consistent spacing scale".to_string());
        }
        if touch < 0.80 {
            recs.push("Ensure 44px minimum touch targets on mobile".to_string());
        }

        recs
    }
}

/// Responsive validation result
#[derive(Debug, Clone)]
pub struct ResponsiveValidationResult {
    pub score: f64,
    pub breakpoints_score: f64,
    pub fluid_typography_score: f64,
    pub flexible_layouts_score: f64,
    pub touch_targets_score: f64,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_platform_validation() {
        let input = DesignInput {
            data: DesignData::Html("<button>Test</button>".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Button,
            design_tokens: Some(DesignTokens {
                colors: ColorTokens::reasonkit_brand(),
                typography: TypographyTokens::reasonkit_brand(),
                ..Default::default()
            }),
        };

        let result = CrossPlatformValidator::validate(
            &input,
            &[Platform::Web, Platform::IOS, Platform::Android],
        );

        assert!(result.score > 0.0);
        assert_eq!(result.platform_results.len(), 3);
    }

    #[test]
    fn test_responsive_validation() {
        let input = DesignInput {
            data: DesignData::Css("body { font-size: 1rem; }".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Page,
            design_tokens: Some(DesignTokens::default()),
        };

        let result = ResponsiveValidator::validate(&input);
        assert!(result.score > 0.0);
    }
}
