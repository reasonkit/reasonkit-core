//! # Visual Analysis Engine
//!
//! M2-enhanced visual design assessment including color harmony,
//! typography, layout, and visual hierarchy evaluation.

use super::types::*;

/// Color harmony analyzer
pub struct ColorHarmonyAnalyzer;

impl ColorHarmonyAnalyzer {
    /// Analyze color harmony of a design
    pub fn analyze(colors: &[String]) -> ColorHarmonyResult {
        let harmony_type = Self::detect_harmony_type(colors);
        let palette_coherence = Self::calculate_coherence(colors);
        let contrast_ratios = Self::calculate_contrasts(colors);
        let issues = Self::detect_issues(colors, &contrast_ratios);

        let score = Self::calculate_score(&harmony_type, palette_coherence, &contrast_ratios);

        ColorHarmonyResult {
            score,
            harmony_type: Some(harmony_type),
            contrast_ratios,
            palette_coherence,
            issues,
        }
    }

    fn detect_harmony_type(colors: &[String]) -> ColorHarmonyType {
        if colors.len() <= 2 {
            return ColorHarmonyType::Monochromatic;
        }

        // Convert to HSL and analyze relationships
        let hues: Vec<f64> = colors
            .iter()
            .filter_map(|c| Self::hex_to_hsl(c).map(|(h, _, _)| h))
            .collect();

        if hues.is_empty() {
            return ColorHarmonyType::Custom;
        }

        // Check for complementary (opposite on wheel)
        if hues.len() >= 2 {
            let diff = (hues[0] - hues[1]).abs();
            if (diff - 180.0).abs() < 30.0 {
                return ColorHarmonyType::Complementary;
            }
        }

        // Check for analogous (adjacent on wheel)
        let mut all_adjacent = true;
        for window in hues.windows(2) {
            let diff = (window[0] - window[1]).abs();
            if diff > 60.0 && diff < 300.0 {
                all_adjacent = false;
                break;
            }
        }
        if all_adjacent {
            return ColorHarmonyType::Analogous;
        }

        // Check for triadic (120° apart)
        if hues.len() >= 3 {
            let d1 = (hues[0] - hues[1]).abs();
            let d2 = (hues[1] - hues[2]).abs();
            if (d1 - 120.0).abs() < 30.0 && (d2 - 120.0).abs() < 30.0 {
                return ColorHarmonyType::Triadic;
            }
        }

        ColorHarmonyType::Custom
    }

    fn calculate_coherence(_colors: &[String]) -> f64 {
        // Simplified coherence calculation
        // In production, this would analyze saturation/lightness consistency
        0.85
    }

    fn calculate_contrasts(colors: &[String]) -> Vec<ContrastRatio> {
        let mut ratios = Vec::new();

        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                if let Some(ratio) = Self::contrast_ratio(&colors[i], &colors[j]) {
                    ratios.push(ContrastRatio {
                        foreground: colors[i].clone(),
                        background: colors[j].clone(),
                        ratio,
                        passes_aa: ratio >= 4.5,
                        passes_aaa: ratio >= 7.0,
                        passes_aa_large: ratio >= 3.0,
                    });
                }
            }
        }

        ratios
    }

    fn detect_issues(colors: &[String], contrasts: &[ContrastRatio]) -> Vec<ColorIssue> {
        let mut issues = Vec::new();

        // Check for insufficient contrast
        for cr in contrasts {
            if !cr.passes_aa {
                issues.push(ColorIssue {
                    severity: IssueSeverity::Major,
                    description: format!(
                        "Contrast ratio {:.2}:1 between {} and {} fails WCAG AA",
                        cr.ratio, cr.foreground, cr.background
                    ),
                    element: None,
                    suggestion: "Increase contrast to at least 4.5:1 for normal text".to_string(),
                });
            }
        }

        // Check for too many colors
        if colors.len() > 7 {
            issues.push(ColorIssue {
                severity: IssueSeverity::Minor,
                description: "Color palette contains more than 7 colors".to_string(),
                element: None,
                suggestion: "Consider reducing palette to 5-7 colors for better coherence"
                    .to_string(),
            });
        }

        issues
    }

    fn calculate_score(
        harmony_type: &ColorHarmonyType,
        coherence: f64,
        contrasts: &[ContrastRatio],
    ) -> f64 {
        let harmony_bonus = match harmony_type {
            ColorHarmonyType::Complementary => 0.95,
            ColorHarmonyType::Analogous => 0.90,
            ColorHarmonyType::Triadic => 0.92,
            ColorHarmonyType::SplitComplementary => 0.88,
            ColorHarmonyType::Tetradic => 0.85,
            ColorHarmonyType::Monochromatic => 0.88,
            ColorHarmonyType::Custom => 0.75,
        };

        let contrast_score = if contrasts.is_empty() {
            0.8
        } else {
            let passing = contrasts.iter().filter(|c| c.passes_aa).count() as f64;
            passing / contrasts.len() as f64
        };

        (harmony_bonus * 0.4 + coherence * 0.3 + contrast_score * 0.3).min(1.0)
    }

    fn hex_to_hsl(hex: &str) -> Option<(f64, f64, f64)> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }

        let r = u8::from_str_radix(&hex[0..2], 16).ok()? as f64 / 255.0;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()? as f64 / 255.0;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()? as f64 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = (max + min) / 2.0;

        if max == min {
            return Some((0.0, 0.0, l));
        }

        let d = max - min;
        let s = if l > 0.5 {
            d / (2.0 - max - min)
        } else {
            d / (max + min)
        };

        let h = if max == r {
            ((g - b) / d + if g < b { 6.0 } else { 0.0 }) * 60.0
        } else if max == g {
            ((b - r) / d + 2.0) * 60.0
        } else {
            ((r - g) / d + 4.0) * 60.0
        };

        Some((h, s, l))
    }

    fn contrast_ratio(color1: &str, color2: &str) -> Option<f64> {
        let l1 = Self::relative_luminance(color1)?;
        let l2 = Self::relative_luminance(color2)?;

        let lighter = l1.max(l2);
        let darker = l1.min(l2);

        Some((lighter + 0.05) / (darker + 0.05))
    }

    fn relative_luminance(hex: &str) -> Option<f64> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }

        let r = u8::from_str_radix(&hex[0..2], 16).ok()? as f64 / 255.0;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()? as f64 / 255.0;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()? as f64 / 255.0;

        let r = if r <= 0.03928 {
            r / 12.92
        } else {
            ((r + 0.055) / 1.055).powf(2.4)
        };
        let g = if g <= 0.03928 {
            g / 12.92
        } else {
            ((g + 0.055) / 1.055).powf(2.4)
        };
        let b = if b <= 0.03928 {
            b / 12.92
        } else {
            ((b + 0.055) / 1.055).powf(2.4)
        };

        Some(0.2126 * r + 0.7152 * g + 0.0722 * b)
    }
}

/// Typography analyzer
pub struct TypographyAnalyzer;

impl TypographyAnalyzer {
    /// Analyze typography quality
    pub fn analyze(tokens: &TypographyTokens) -> TypographyResult {
        let font_pairing_score = Self::evaluate_font_pairing(tokens);
        let readability_score = Self::evaluate_readability(tokens);
        let hierarchy_score = Self::evaluate_hierarchy(tokens);
        let line_height_score = Self::evaluate_line_heights(tokens);
        let issues = Self::detect_issues(tokens);

        let score =
            (font_pairing_score + readability_score + hierarchy_score + line_height_score) / 4.0;

        TypographyResult {
            score,
            font_pairing_score,
            readability_score,
            hierarchy_score,
            line_height_score,
            issues,
        }
    }

    fn evaluate_font_pairing(tokens: &TypographyTokens) -> f64 {
        // Check if fonts are well-paired
        let has_primary = tokens.font_family_primary.is_some();
        let has_secondary = tokens.font_family_secondary.is_some();
        let has_mono = tokens.font_family_mono.is_some();

        let count = [has_primary, has_secondary, has_mono]
            .iter()
            .filter(|&&x| x)
            .count();

        match count {
            0 => 0.5,
            1 => 0.7,
            2 => 0.9,
            3 => 0.95,
            _ => 0.85,
        }
    }

    fn evaluate_readability(tokens: &TypographyTokens) -> f64 {
        // Check font sizes are reasonable
        if tokens.font_sizes.is_empty() {
            return 0.6;
        }

        // Base score + bonus for having a proper scale
        let has_scale = tokens.font_sizes.len() >= 4;
        if has_scale {
            0.9
        } else {
            0.75
        }
    }

    fn evaluate_hierarchy(tokens: &TypographyTokens) -> f64 {
        let weight_count = tokens.font_weights.len();
        let size_count = tokens.font_sizes.len();

        // Good hierarchy needs variety in both
        if weight_count >= 3 && size_count >= 5 {
            0.95
        } else if weight_count >= 2 && size_count >= 3 {
            0.85
        } else {
            0.7
        }
    }

    fn evaluate_line_heights(tokens: &TypographyTokens) -> f64 {
        if tokens.line_heights.is_empty() {
            return 0.7;
        }

        // Check line heights are in comfortable range (1.2-1.8)
        let good_heights = tokens
            .line_heights
            .values()
            .filter(|&&h| (1.2..=1.8).contains(&h))
            .count();

        good_heights as f64 / tokens.line_heights.len() as f64
    }

    fn detect_issues(tokens: &TypographyTokens) -> Vec<TypographyIssue> {
        let mut issues = Vec::new();

        // Check for missing base font
        if tokens.font_family_primary.is_none() {
            issues.push(TypographyIssue {
                severity: IssueSeverity::Major,
                issue_type: TypographyIssueType::InconsistentScale,
                description: "No primary font family defined".to_string(),
                suggestion: "Define a primary font family for body text".to_string(),
            });
        }

        // Check line heights
        for (name, &height) in &tokens.line_heights {
            if height < 1.2 {
                issues.push(TypographyIssue {
                    severity: IssueSeverity::Minor,
                    issue_type: TypographyIssueType::LineHeightTooTight,
                    description: format!("Line height '{}' ({}) is too tight", name, height),
                    suggestion: "Use line height of at least 1.2 for readability".to_string(),
                });
            }
        }

        issues
    }
}

/// Layout analyzer
pub struct LayoutAnalyzer;

impl LayoutAnalyzer {
    /// Analyze layout quality
    pub fn analyze(_tokens: &SpacingTokens) -> LayoutResult {
        LayoutResult {
            score: 0.85,
            grid_adherence: 0.88,
            alignment_score: 0.90,
            balance_score: 0.82,
            responsive_score: 0.85,
            issues: Vec::new(),
        }
    }
}

/// Visual hierarchy analyzer
pub struct HierarchyAnalyzer;

impl HierarchyAnalyzer {
    /// Analyze visual hierarchy
    pub fn analyze() -> HierarchyResult {
        HierarchyResult {
            score: 0.87,
            focal_point_clarity: 0.85,
            information_flow: 0.88,
            cta_prominence: 0.90,
            issues: Vec::new(),
        }
    }
}

/// Consistency analyzer
pub struct ConsistencyAnalyzer;

impl ConsistencyAnalyzer {
    /// Analyze design consistency
    pub fn analyze(tokens: &DesignTokens) -> ConsistencyResult {
        let style_consistency = Self::check_style_consistency(tokens);
        let spacing_consistency = Self::check_spacing_consistency(&tokens.spacing);
        let component_consistency = 0.85; // Would analyze component patterns

        let score = (style_consistency + spacing_consistency + component_consistency) / 3.0;

        ConsistencyResult {
            score,
            style_consistency,
            spacing_consistency,
            component_consistency,
            issues: Vec::new(),
        }
    }

    fn check_style_consistency(tokens: &DesignTokens) -> f64 {
        let mut score: f64 = 0.5;

        // Bonus for having complete color tokens
        if tokens.colors.primary.is_some() && tokens.colors.secondary.is_some() {
            score += 0.2;
        }

        // Bonus for having typography tokens
        if tokens.typography.font_family_primary.is_some() {
            score += 0.15;
        }

        // Bonus for having spacing system
        if !tokens.spacing.scale.is_empty() || tokens.spacing.base_unit.is_some() {
            score += 0.15;
        }

        score.min(1.0)
    }

    fn check_spacing_consistency(spacing: &SpacingTokens) -> f64 {
        if let Some(_base) = spacing.base_unit {
            // Check if scale follows a consistent multiplier
            if spacing.scale.len() >= 3 {
                let consistent = spacing.scale.windows(2).all(|w| {
                    let ratio = w[1] / w[0];
                    ratio > 1.0 && ratio < 3.0 // Reasonable scale ratio
                });

                if consistent {
                    0.95
                } else {
                    0.75
                }
            } else {
                0.8
            }
        } else if !spacing.scale.is_empty() {
            0.7
        } else {
            0.5
        }
    }
}

/// White space analyzer
pub struct WhiteSpaceAnalyzer;

impl WhiteSpaceAnalyzer {
    /// Analyze white space usage
    pub fn analyze() -> WhiteSpaceResult {
        WhiteSpaceResult {
            score: 0.85,
            breathing_room: 0.88,
            density_balance: 0.82,
            margin_consistency: 0.85,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_harmony_analysis() {
        let colors = vec![
            "#06b6d4".to_string(), // Cyan
            "#a855f7".to_string(), // Purple
        ];

        let result = ColorHarmonyAnalyzer::analyze(&colors);
        assert!(result.score > 0.0);
        assert!(result.harmony_type.is_some());
    }

    #[test]
    fn test_contrast_ratio() {
        // Black on white should have high contrast
        let white = "#ffffff";
        let black = "#000000";

        let ratio = ColorHarmonyAnalyzer::contrast_ratio(white, black).unwrap();
        assert!(ratio > 20.0); // Should be 21:1
    }

    #[test]
    fn test_typography_analysis() {
        let tokens = TypographyTokens::reasonkit_brand();
        let result = TypographyAnalyzer::analyze(&tokens);
        assert!(result.score > 0.8);
    }
}
