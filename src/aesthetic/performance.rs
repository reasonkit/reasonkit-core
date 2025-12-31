//! # Performance Impact Analysis
//!
//! Analyzes design decisions impact on performance.

use super::types::*;

/// Performance impact analyzer
pub struct PerformanceAnalyzer;

impl PerformanceAnalyzer {
    /// Analyze performance impact of design
    pub fn analyze(input: &DesignInput) -> PerformanceImpactResult {
        let render_complexity = Self::analyze_render_complexity(input);
        let resource_usage = Self::analyze_resource_usage(input);

        let estimated_load_time_ms = Self::estimate_load_time(&render_complexity, &resource_usage);
        let estimated_interaction_delay_ms = Self::estimate_interaction_delay(&render_complexity);

        let recommendations = Self::generate_recommendations(
            &render_complexity,
            &resource_usage,
            estimated_load_time_ms,
        );

        // Calculate overall score
        let load_score = (2000.0 - estimated_load_time_ms as f64).max(0.0) / 2000.0;
        let interaction_score = (100.0 - estimated_interaction_delay_ms as f64).max(0.0) / 100.0;
        let complexity_score = 1.0 - (render_complexity.dom_depth as f64 / 30.0).min(1.0);

        let score =
            (load_score * 0.4 + interaction_score * 0.35 + complexity_score * 0.25).min(1.0);

        PerformanceImpactResult {
            score,
            estimated_load_time_ms,
            estimated_interaction_delay_ms,
            render_complexity,
            resource_usage,
            recommendations,
        }
    }

    /// Analyze render complexity
    fn analyze_render_complexity(input: &DesignInput) -> RenderComplexity {
        match &input.data {
            DesignData::Html(html) => {
                let dom_depth = Self::estimate_dom_depth(html);
                let dom_node_count = Self::count_dom_nodes(html);
                let animation_count = Self::count_animations(html);

                RenderComplexity {
                    dom_depth,
                    dom_node_count,
                    css_complexity_score: 0.7, // Would analyze actual CSS
                    animation_count,
                    reflow_risk: Self::calculate_reflow_risk(dom_depth, animation_count),
                }
            }
            DesignData::HtmlCss { html, css } => {
                let dom_depth = Self::estimate_dom_depth(html);
                let dom_node_count = Self::count_dom_nodes(html);
                let animation_count = Self::count_animations(css);
                let css_complexity = Self::analyze_css_complexity(css);

                RenderComplexity {
                    dom_depth,
                    dom_node_count,
                    css_complexity_score: css_complexity,
                    animation_count,
                    reflow_risk: Self::calculate_reflow_risk(dom_depth, animation_count),
                }
            }
            DesignData::ReactComponent(code) => {
                let dom_depth = Self::estimate_react_depth(code);
                let dom_node_count = Self::count_react_nodes(code);

                RenderComplexity {
                    dom_depth,
                    dom_node_count,
                    css_complexity_score: 0.75,
                    animation_count: 0,
                    reflow_risk: 0.3,
                }
            }
            _ => RenderComplexity::default(),
        }
    }

    /// Analyze resource usage
    fn analyze_resource_usage(input: &DesignInput) -> ResourceUsage {
        match &input.data {
            DesignData::Css(css) => ResourceUsage {
                estimated_css_size_kb: css.len() as f64 / 1024.0,
                estimated_js_size_kb: 0.0,
                image_optimization_score: 0.8,
                font_loading_strategy: "system".to_string(),
            },
            DesignData::HtmlCss { css, .. } => ResourceUsage {
                estimated_css_size_kb: css.len() as f64 / 1024.0,
                estimated_js_size_kb: 0.0,
                image_optimization_score: 0.8,
                font_loading_strategy: Self::detect_font_strategy(css),
            },
            DesignData::ReactComponent(code) => {
                ResourceUsage {
                    estimated_css_size_kb: 5.0, // Estimate for styled components
                    estimated_js_size_kb: code.len() as f64 / 1024.0,
                    image_optimization_score: 0.75,
                    font_loading_strategy: "next/font".to_string(),
                }
            }
            _ => ResourceUsage::default(),
        }
    }

    fn estimate_dom_depth(html: &str) -> u32 {
        // Simplified: count nested tags
        let open_tags = html.matches('<').count() - html.matches("</").count();
        (open_tags as u32 / 3).min(30)
    }

    fn count_dom_nodes(html: &str) -> u32 {
        // Count opening tags
        html.matches('<').count() as u32 - html.matches("</").count() as u32
    }

    fn count_animations(content: &str) -> u32 {
        let transitions = content.matches("transition").count();
        let animations = content.matches("animation").count();
        let keyframes = content.matches("@keyframes").count();

        (transitions + animations + keyframes) as u32
    }

    fn analyze_css_complexity(css: &str) -> f64 {
        let selectors = css.matches('{').count();
        let rules = css.matches(';').count();
        let complexity = (selectors + rules) as f64;

        // Higher complexity = lower score
        (1.0 - (complexity / 1000.0)).max(0.1)
    }

    fn calculate_reflow_risk(dom_depth: u32, animation_count: u32) -> f64 {
        let depth_risk = (dom_depth as f64 / 20.0).min(1.0);
        let animation_risk = (animation_count as f64 / 10.0).min(1.0);

        (depth_risk * 0.6 + animation_risk * 0.4).min(1.0)
    }

    fn estimate_react_depth(code: &str) -> u32 {
        // Count JSX nesting by looking for patterns
        let jsx_opens =
            code.matches("<").count() - code.matches("</").count() - code.matches("/>").count();
        (jsx_opens as u32 / 2).min(25)
    }

    fn count_react_nodes(code: &str) -> u32 {
        // Count JSX elements
        let elements = code.matches("<").count() - code.matches("</").count();
        elements as u32
    }

    fn detect_font_strategy(css: &str) -> String {
        if css.contains("font-display: swap") {
            "swap".to_string()
        } else if css.contains("font-display: optional") {
            "optional".to_string()
        } else if css.contains("@font-face") {
            "block".to_string()
        } else {
            "system".to_string()
        }
    }

    fn estimate_load_time(complexity: &RenderComplexity, resources: &ResourceUsage) -> u64 {
        // Base time
        let mut time_ms = 100.0;

        // Add for DOM complexity
        time_ms += complexity.dom_node_count as f64 * 0.5;
        time_ms += complexity.dom_depth as f64 * 10.0;

        // Add for resources
        time_ms += resources.estimated_css_size_kb * 5.0;
        time_ms += resources.estimated_js_size_kb * 10.0;

        // Font loading penalty
        if resources.font_loading_strategy == "block" {
            time_ms += 200.0;
        }

        time_ms as u64
    }

    fn estimate_interaction_delay(complexity: &RenderComplexity) -> u64 {
        let base_delay = 16.0; // 60fps frame time

        let reflow_penalty = complexity.reflow_risk * 50.0;
        let animation_penalty = complexity.animation_count as f64 * 2.0;

        (base_delay + reflow_penalty + animation_penalty) as u64
    }

    fn generate_recommendations(
        complexity: &RenderComplexity,
        resources: &ResourceUsage,
        load_time: u64,
    ) -> Vec<PerformanceRecommendation> {
        let mut recs = Vec::new();

        // DOM depth
        if complexity.dom_depth > 15 {
            recs.push(PerformanceRecommendation {
                category: "DOM Structure".to_string(),
                description: format!(
                    "DOM depth ({}) is high. Flatten component hierarchy.",
                    complexity.dom_depth
                ),
                impact: "Reduces reflow time and improves interaction responsiveness".to_string(),
                priority: OptimizationPriority::High,
            });
        }

        // Node count
        if complexity.dom_node_count > 500 {
            recs.push(PerformanceRecommendation {
                category: "DOM Size".to_string(),
                description: format!(
                    "DOM has {} nodes. Consider virtualization for large lists.",
                    complexity.dom_node_count
                ),
                impact: "Reduces memory usage and improves scroll performance".to_string(),
                priority: OptimizationPriority::High,
            });
        }

        // CSS size
        if resources.estimated_css_size_kb > 50.0 {
            recs.push(PerformanceRecommendation {
                category: "CSS Optimization".to_string(),
                description: format!(
                    "CSS size ({:.1}KB) is large. Consider code splitting.",
                    resources.estimated_css_size_kb
                ),
                impact: "Faster initial render and reduced bandwidth".to_string(),
                priority: OptimizationPriority::Medium,
            });
        }

        // Font loading
        if resources.font_loading_strategy == "block" {
            recs.push(PerformanceRecommendation {
                category: "Font Loading".to_string(),
                description: "Font loading blocks render. Use font-display: swap.".to_string(),
                impact: "Eliminates Flash of Invisible Text (FOIT)".to_string(),
                priority: OptimizationPriority::High,
            });
        }

        // Animations
        if complexity.animation_count > 5 {
            recs.push(PerformanceRecommendation {
                category: "Animation Optimization".to_string(),
                description: format!(
                    "{} animations detected. Prefer transform/opacity only.",
                    complexity.animation_count
                ),
                impact: "Enables GPU acceleration, smoother animations".to_string(),
                priority: OptimizationPriority::Medium,
            });
        }

        // Overall load time
        if load_time > 1500 {
            recs.push(PerformanceRecommendation {
                category: "Critical Path".to_string(),
                description: format!("Estimated load time ({}ms) exceeds 1.5s target.", load_time),
                impact: "Improves Core Web Vitals (LCP, FID)".to_string(),
                priority: OptimizationPriority::Critical,
            });
        }

        recs
    }
}

/// Core Web Vitals analyzer
pub struct CoreWebVitalsAnalyzer;

impl CoreWebVitalsAnalyzer {
    /// Estimate Core Web Vitals from design
    pub fn estimate(input: &DesignInput) -> CoreWebVitalsEstimate {
        let perf = PerformanceAnalyzer::analyze(input);

        // Largest Contentful Paint estimate
        let lcp_ms = perf.estimated_load_time_ms
            + match input.component_type {
                ComponentType::Hero => 300, // Hero images are often LCP
                ComponentType::Page => 200,
                _ => 100,
            };

        // First Input Delay estimate
        let fid_ms = perf.estimated_interaction_delay_ms;

        // Cumulative Layout Shift estimate
        let cls = if perf.render_complexity.animation_count > 0 {
            0.15
        } else {
            0.05
        };

        CoreWebVitalsEstimate {
            lcp_ms,
            lcp_rating: Self::rate_lcp(lcp_ms),
            fid_ms,
            fid_rating: Self::rate_fid(fid_ms),
            cls,
            cls_rating: Self::rate_cls(cls),
            overall_score: Self::calculate_overall(lcp_ms, fid_ms, cls),
        }
    }

    fn rate_lcp(ms: u64) -> VitalRating {
        if ms <= 2500 {
            VitalRating::Good
        } else if ms <= 4000 {
            VitalRating::NeedsImprovement
        } else {
            VitalRating::Poor
        }
    }

    fn rate_fid(ms: u64) -> VitalRating {
        if ms <= 100 {
            VitalRating::Good
        } else if ms <= 300 {
            VitalRating::NeedsImprovement
        } else {
            VitalRating::Poor
        }
    }

    fn rate_cls(score: f64) -> VitalRating {
        if score <= 0.1 {
            VitalRating::Good
        } else if score <= 0.25 {
            VitalRating::NeedsImprovement
        } else {
            VitalRating::Poor
        }
    }

    fn calculate_overall(lcp: u64, fid: u64, cls: f64) -> f64 {
        let lcp_score = (2500.0 - lcp as f64).max(0.0) / 2500.0;
        let fid_score = (100.0 - fid as f64).max(0.0) / 100.0;
        let cls_score = (0.1 - cls).max(0.0) / 0.1;

        (lcp_score * 0.4 + fid_score * 0.3 + cls_score * 0.3).min(1.0)
    }
}

/// Core Web Vitals estimate
#[derive(Debug, Clone)]
pub struct CoreWebVitalsEstimate {
    pub lcp_ms: u64,
    pub lcp_rating: VitalRating,
    pub fid_ms: u64,
    pub fid_rating: VitalRating,
    pub cls: f64,
    pub cls_rating: VitalRating,
    pub overall_score: f64,
}

/// Web Vital rating
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VitalRating {
    Good,
    NeedsImprovement,
    Poor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_analysis() {
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
    }

    #[test]
    fn test_core_web_vitals() {
        let input = DesignInput {
            data: DesignData::Html("<main><h1>Title</h1></main>".to_string()),
            platform: Platform::Web,
            context: None,
            component_type: ComponentType::Page,
            design_tokens: None,
        };

        let estimate = CoreWebVitalsAnalyzer::estimate(&input);
        assert!(estimate.overall_score > 0.0);
    }
}
