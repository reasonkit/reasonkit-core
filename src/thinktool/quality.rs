//! # Quality Metrics Collection System
//!
//! Comprehensive quality tracking across all ThinkTools modules.
//! Provides dashboards, trends, and improvement recommendations.
//!
//! ## Metrics Collected
//!
//! | Category | Metrics |
//! |----------|---------|
//! | Accuracy | GSM8K, MATH, ARC-C benchmark scores |
//! | Calibration | Brier score, ECE, overconfidence ratio |
//! | Reasoning | PRM scores, ToT success rate, step validity |
//! | Verification | Triangulation score, fact-check accuracy |
//! | Debate | Win rate, argument strength, verdict confidence |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::quality::{QualityDashboard, QualityMetric};
//!
//! let mut dashboard = QualityDashboard::new();
//! dashboard.record_metric(QualityMetric::Accuracy { benchmark: "GSM8K", score: 0.859 });
//! dashboard.record_metric(QualityMetric::Calibration { brier: 0.15, ece: 0.08 });
//!
//! let report = dashboard.generate_report();
//! println!("{}", report.format());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Individual quality metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityMetric {
    /// Benchmark accuracy scores
    Accuracy {
        benchmark: String,
        score: f32,
        samples: usize,
    },
    /// Calibration metrics
    Calibration {
        brier_score: f32,
        ece: f32,
        overconfidence_ratio: f32,
    },
    /// Process Reward Model metrics
    PrmScore {
        avg_step_correctness: f32,
        critical_issues: usize,
        sound_chains: f32,
    },
    /// Tree-of-Thoughts metrics
    TotMetrics {
        success_rate: f32,
        avg_depth: f32,
        nodes_explored: usize,
        pruning_rate: f32,
    },
    /// Triangulation metrics
    Triangulation {
        verification_rate: f32,
        avg_sources: f32,
        contradiction_rate: f32,
    },
    /// Debate metrics
    Debate {
        advocate_win_rate: f32,
        avg_argument_strength: f32,
        consensus_rate: f32,
    },
    /// Toulmin argument quality
    Argumentation {
        soundness_rate: f32,
        avg_grounds_score: f32,
        avg_warrant_score: f32,
    },
    /// Latency metrics
    Latency {
        avg_ms: f64,
        p95_ms: f64,
        p99_ms: f64,
    },
    /// Token usage
    TokenUsage {
        avg_tokens: usize,
        total_tokens: usize,
        efficiency: f32,
    },
    /// Custom metric
    Custom {
        name: String,
        value: f32,
        unit: Option<String>,
    },
}

impl QualityMetric {
    pub fn category(&self) -> &'static str {
        match self {
            QualityMetric::Accuracy { .. } => "accuracy",
            QualityMetric::Calibration { .. } => "calibration",
            QualityMetric::PrmScore { .. } => "reasoning",
            QualityMetric::TotMetrics { .. } => "exploration",
            QualityMetric::Triangulation { .. } => "verification",
            QualityMetric::Debate { .. } => "debate",
            QualityMetric::Argumentation { .. } => "argumentation",
            QualityMetric::Latency { .. } => "performance",
            QualityMetric::TokenUsage { .. } => "efficiency",
            QualityMetric::Custom { .. } => "custom",
        }
    }

    pub fn primary_value(&self) -> f32 {
        match self {
            QualityMetric::Accuracy { score, .. } => *score,
            QualityMetric::Calibration { brier_score, .. } => 1.0 - *brier_score, // Invert (lower is better)
            QualityMetric::PrmScore {
                avg_step_correctness,
                ..
            } => *avg_step_correctness,
            QualityMetric::TotMetrics { success_rate, .. } => *success_rate,
            QualityMetric::Triangulation {
                verification_rate, ..
            } => *verification_rate,
            QualityMetric::Debate {
                avg_argument_strength,
                ..
            } => *avg_argument_strength,
            QualityMetric::Argumentation { soundness_rate, .. } => *soundness_rate,
            QualityMetric::Latency { avg_ms, .. } => (1000.0 / *avg_ms as f32).min(1.0), // Faster = better
            QualityMetric::TokenUsage { efficiency, .. } => *efficiency,
            QualityMetric::Custom { value, .. } => *value,
        }
    }
}

/// Timestamped metric record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    pub metric: QualityMetric,
    pub timestamp: u64,
    pub profile: Option<String>,
    pub session_id: Option<String>,
}

impl MetricRecord {
    pub fn new(metric: QualityMetric) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            metric,
            timestamp,
            profile: None,
            session_id: None,
        }
    }

    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }
}

/// Quality targets for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTargets {
    /// Accuracy targets per benchmark
    pub accuracy: HashMap<String, f32>,
    /// Maximum acceptable Brier score
    pub max_brier_score: f32,
    /// Maximum acceptable ECE
    pub max_ece: f32,
    /// Minimum PRM step correctness
    pub min_prm_correctness: f32,
    /// Minimum ToT success rate
    pub min_tot_success: f32,
    /// Minimum triangulation verification rate
    pub min_triangulation: f32,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f64,
}

impl Default for QualityTargets {
    fn default() -> Self {
        let mut accuracy = HashMap::new();
        accuracy.insert("GSM8K".into(), 0.859);
        accuracy.insert("MATH".into(), 0.365);
        accuracy.insert("ARC-C".into(), 0.90);
        accuracy.insert("TruthfulQA".into(), 0.72);

        Self {
            accuracy,
            max_brier_score: 0.20,
            max_ece: 0.10,
            min_prm_correctness: 0.80,
            min_tot_success: 0.60,
            min_triangulation: 0.70,
            max_latency_ms: 5000.0,
        }
    }
}

/// Quality score aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScore {
    /// Overall quality score (0-100)
    pub overall: f32,
    /// Per-category scores
    pub categories: HashMap<String, f32>,
    /// Grade (A-F)
    pub grade: QualityGrade,
    /// Trend (improving/declining/stable)
    pub trend: Trend,
    /// Areas needing improvement
    pub improvement_areas: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityGrade {
    A, // 90-100
    B, // 80-89
    C, // 70-79
    D, // 60-69
    F, // < 60
}

impl QualityGrade {
    pub fn from_score(score: f32) -> Self {
        match (score * 100.0) as u32 {
            90..=100 => QualityGrade::A,
            80..=89 => QualityGrade::B,
            70..=79 => QualityGrade::C,
            60..=69 => QualityGrade::D,
            _ => QualityGrade::F,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            QualityGrade::A => "Excellent",
            QualityGrade::B => "Good",
            QualityGrade::C => "Acceptable",
            QualityGrade::D => "Needs Improvement",
            QualityGrade::F => "Failing",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Improving,
    Stable,
    Declining,
    Unknown,
}

/// Quality dashboard for tracking and reporting
pub struct QualityDashboard {
    pub targets: QualityTargets,
    records: Vec<MetricRecord>,
    /// Category weights for overall score
    weights: HashMap<String, f32>,
}

impl QualityDashboard {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("accuracy".into(), 0.25);
        weights.insert("calibration".into(), 0.15);
        weights.insert("reasoning".into(), 0.15);
        weights.insert("verification".into(), 0.15);
        weights.insert("argumentation".into(), 0.10);
        weights.insert("exploration".into(), 0.10);
        weights.insert("performance".into(), 0.05);
        weights.insert("efficiency".into(), 0.05);

        Self {
            targets: QualityTargets::default(),
            records: Vec::new(),
            weights,
        }
    }

    pub fn with_targets(mut self, targets: QualityTargets) -> Self {
        self.targets = targets;
        self
    }

    /// Record a metric
    pub fn record_metric(&mut self, metric: QualityMetric) {
        self.records.push(MetricRecord::new(metric));
    }

    /// Record a metric with profile
    pub fn record_with_profile(&mut self, metric: QualityMetric, profile: &str) {
        self.records
            .push(MetricRecord::new(metric).with_profile(profile));
    }

    /// Get records by category
    pub fn get_by_category(&self, category: &str) -> Vec<&MetricRecord> {
        self.records
            .iter()
            .filter(|r| r.metric.category() == category)
            .collect()
    }

    /// Get latest record for each category
    pub fn get_latest_by_category(&self) -> HashMap<String, &MetricRecord> {
        let mut latest: HashMap<String, &MetricRecord> = HashMap::new();

        for record in &self.records {
            let cat = record.metric.category().to_string();
            match latest.get(&cat) {
                None => {
                    latest.insert(cat, record);
                }
                Some(existing) if record.timestamp > existing.timestamp => {
                    latest.insert(cat, record);
                }
                _ => {}
            }
        }

        latest
    }

    /// Compute category score
    fn compute_category_score(&self, category: &str) -> Option<f32> {
        let records: Vec<_> = self.get_by_category(category);
        if records.is_empty() {
            return None;
        }

        // Use latest N records for averaging
        let recent: Vec<_> = records.into_iter().rev().take(10).collect();
        let avg =
            recent.iter().map(|r| r.metric.primary_value()).sum::<f32>() / recent.len() as f32;

        Some(avg)
    }

    /// Compute overall quality score
    pub fn compute_score(&self) -> QualityScore {
        let mut categories = HashMap::new();
        let mut weighted_sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for (cat, weight) in &self.weights {
            if let Some(score) = self.compute_category_score(cat) {
                categories.insert(cat.clone(), score);
                weighted_sum += score * weight;
                weight_sum += weight;
            }
        }

        let overall = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };

        let grade = QualityGrade::from_score(overall);

        // Find improvement areas
        let mut improvement_areas = Vec::new();
        for (cat, score) in &categories {
            if *score < 0.7 {
                improvement_areas.push(format!("{} ({:.0}%)", cat, score * 100.0));
            }
        }

        // Compute trend (compare last 10 vs previous 10)
        let trend = self.compute_trend();

        QualityScore {
            overall,
            categories,
            grade,
            trend,
            improvement_areas,
        }
    }

    fn compute_trend(&self) -> Trend {
        if self.records.len() < 20 {
            return Trend::Unknown;
        }

        let mid = self.records.len() / 2;
        let first_half_avg = self.records[..mid]
            .iter()
            .map(|r| r.metric.primary_value())
            .sum::<f32>()
            / mid as f32;

        let second_half_avg = self.records[mid..]
            .iter()
            .map(|r| r.metric.primary_value())
            .sum::<f32>()
            / (self.records.len() - mid) as f32;

        let diff = second_half_avg - first_half_avg;

        if diff > 0.05 {
            Trend::Improving
        } else if diff < -0.05 {
            Trend::Declining
        } else {
            Trend::Stable
        }
    }

    /// Check metrics against targets
    pub fn check_targets(&self) -> Vec<TargetViolation> {
        let mut violations = Vec::new();

        for record in self.get_latest_by_category().values() {
            match &record.metric {
                QualityMetric::Accuracy {
                    benchmark, score, ..
                } => {
                    if let Some(&target) = self.targets.accuracy.get(benchmark) {
                        if *score < target {
                            violations.push(TargetViolation {
                                metric: format!("{} accuracy", benchmark),
                                target,
                                actual: *score,
                                gap: target - score,
                            });
                        }
                    }
                }
                QualityMetric::Calibration {
                    brier_score, ece, ..
                } => {
                    if *brier_score > self.targets.max_brier_score {
                        violations.push(TargetViolation {
                            metric: "Brier score".into(),
                            target: self.targets.max_brier_score,
                            actual: *brier_score,
                            gap: *brier_score - self.targets.max_brier_score,
                        });
                    }
                    if *ece > self.targets.max_ece {
                        violations.push(TargetViolation {
                            metric: "ECE".into(),
                            target: self.targets.max_ece,
                            actual: *ece,
                            gap: *ece - self.targets.max_ece,
                        });
                    }
                }
                QualityMetric::PrmScore {
                    avg_step_correctness,
                    ..
                } => {
                    if *avg_step_correctness < self.targets.min_prm_correctness {
                        violations.push(TargetViolation {
                            metric: "PRM step correctness".into(),
                            target: self.targets.min_prm_correctness,
                            actual: *avg_step_correctness,
                            gap: self.targets.min_prm_correctness - avg_step_correctness,
                        });
                    }
                }
                QualityMetric::Latency { avg_ms, .. } => {
                    if *avg_ms > self.targets.max_latency_ms {
                        violations.push(TargetViolation {
                            metric: "Latency".into(),
                            target: self.targets.max_latency_ms as f32,
                            actual: *avg_ms as f32,
                            gap: (*avg_ms - self.targets.max_latency_ms) as f32,
                        });
                    }
                }
                _ => {}
            }
        }

        violations
    }

    /// Generate quality report
    pub fn generate_report(&self) -> QualityReport {
        let score = self.compute_score();
        let violations = self.check_targets();

        let recommendations = self.generate_recommendations(&score, &violations);

        QualityReport {
            score,
            violations,
            total_records: self.records.len(),
            recommendations,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    fn generate_recommendations(
        &self,
        score: &QualityScore,
        violations: &[TargetViolation],
    ) -> Vec<String> {
        let mut recs = Vec::new();

        // Based on grade
        match score.grade {
            QualityGrade::F | QualityGrade::D => {
                recs.push("Use --paranoid profile for maximum verification".into());
                recs.push("Enable PRM for step-by-step validation".into());
            }
            QualityGrade::C => {
                recs.push("Consider using --deep profile for thorough analysis".into());
            }
            _ => {}
        }

        // Based on violations
        for violation in violations {
            if violation.metric.contains("accuracy") {
                recs.push(format!(
                    "Improve {} - currently {:.1}% below target",
                    violation.metric,
                    violation.gap * 100.0
                ));
            }
            if violation.metric.contains("Brier") || violation.metric.contains("ECE") {
                recs.push("Recalibrate confidence levels - currently overconfident".into());
            }
            if violation.metric.contains("Latency") {
                recs.push("Consider using lighter models or caching".into());
            }
        }

        // Based on trend
        if score.trend == Trend::Declining {
            recs.push("Quality is declining - review recent changes".into());
        }

        recs
    }

    /// Clear all records
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Export records as JSON
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&self.records).unwrap_or_default()
    }
}

impl Default for QualityDashboard {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetViolation {
    pub metric: String,
    pub target: f32,
    pub actual: f32,
    pub gap: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub score: QualityScore,
    pub violations: Vec<TargetViolation>,
    pub total_records: usize,
    pub recommendations: Vec<String>,
    pub timestamp: u64,
}

impl QualityReport {
    pub fn format(&self) -> String {
        let mut output = String::new();

        output
            .push_str("┌─────────────────────────────────────────────────────────────────────┐\n");
        output
            .push_str("│                    QUALITY METRICS REPORT                           │\n");
        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");

        // Overall score
        let grade_icon = match self.score.grade {
            QualityGrade::A => "⭐",
            QualityGrade::B => "✓",
            QualityGrade::C => "○",
            QualityGrade::D => "⚠",
            QualityGrade::F => "✗",
        };

        output.push_str(&format!(
            "│ OVERALL SCORE: {:.0}/100 {} {:?} ({})            \n",
            self.score.overall * 100.0,
            grade_icon,
            self.score.grade,
            self.score.grade.label()
        ));

        let trend_icon = match self.score.trend {
            Trend::Improving => "📈",
            Trend::Stable => "➡️",
            Trend::Declining => "📉",
            Trend::Unknown => "❓",
        };
        output.push_str(&format!(
            "│ TREND: {:?} {}                                              \n",
            self.score.trend, trend_icon
        ));

        // Category breakdown
        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");
        output
            .push_str("│ CATEGORY SCORES:                                                    │\n");

        let mut cats: Vec<_> = self.score.categories.iter().collect();
        cats.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (cat, score) in cats {
            let bar_len = (*score * 30.0) as usize;
            let bar = "█".repeat(bar_len);
            let gap = " ".repeat(30 - bar_len);
            let icon = if *score >= 0.8 {
                "✓"
            } else if *score >= 0.6 {
                "○"
            } else {
                "✗"
            };
            output.push_str(&format!(
                "│   {:<15} {} |{}{}| {:.0}%\n",
                cat,
                icon,
                bar,
                gap,
                score * 100.0
            ));
        }

        // Violations
        if !self.violations.is_empty() {
            output.push_str(
                "├─────────────────────────────────────────────────────────────────────┤\n",
            );
            output.push_str(
                "│ TARGET VIOLATIONS:                                                  │\n",
            );
            for v in &self.violations {
                output.push_str(&format!(
                    "│   ⚠ {}: {:.1} (target: {:.1}, gap: {:.1})\n",
                    v.metric, v.actual, v.target, v.gap
                ));
            }
        }

        // Improvement areas
        if !self.score.improvement_areas.is_empty() {
            output.push_str(
                "├─────────────────────────────────────────────────────────────────────┤\n",
            );
            output.push_str(
                "│ NEEDS IMPROVEMENT:                                                  │\n",
            );
            for area in &self.score.improvement_areas {
                output.push_str(&format!("│   • {}\n", area));
            }
        }

        // Recommendations
        if !self.recommendations.is_empty() {
            output.push_str(
                "├─────────────────────────────────────────────────────────────────────┤\n",
            );
            output.push_str(
                "│ RECOMMENDATIONS:                                                    │\n",
            );
            for rec in &self.recommendations {
                output.push_str(&format!("│   → {}\n", rec));
            }
        }

        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");
        output.push_str(&format!(
            "│ Total metrics recorded: {}                                          \n",
            self.total_records
        ));
        output
            .push_str("└─────────────────────────────────────────────────────────────────────┘\n");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_dashboard() {
        let mut dashboard = QualityDashboard::new();

        dashboard.record_metric(QualityMetric::Accuracy {
            benchmark: "GSM8K".into(),
            score: 0.85,
            samples: 100,
        });

        dashboard.record_metric(QualityMetric::Calibration {
            brier_score: 0.15,
            ece: 0.08,
            overconfidence_ratio: 0.2,
        });

        let score = dashboard.compute_score();
        assert!(score.overall > 0.0);
    }

    #[test]
    fn test_grade_from_score() {
        assert_eq!(QualityGrade::from_score(0.95), QualityGrade::A);
        assert_eq!(QualityGrade::from_score(0.85), QualityGrade::B);
        assert_eq!(QualityGrade::from_score(0.75), QualityGrade::C);
        assert_eq!(QualityGrade::from_score(0.65), QualityGrade::D);
        assert_eq!(QualityGrade::from_score(0.50), QualityGrade::F);
    }

    #[test]
    fn test_target_violations() {
        let mut dashboard = QualityDashboard::new();

        // Record below-target accuracy
        dashboard.record_metric(QualityMetric::Accuracy {
            benchmark: "GSM8K".into(),
            score: 0.70, // Target is 0.859
            samples: 100,
        });

        let violations = dashboard.check_targets();
        assert!(!violations.is_empty());
        assert!(violations[0].metric.contains("GSM8K"));
    }

    #[test]
    fn test_metric_categories() {
        assert_eq!(
            QualityMetric::Accuracy {
                benchmark: "test".into(),
                score: 0.9,
                samples: 10
            }
            .category(),
            "accuracy"
        );

        assert_eq!(
            QualityMetric::PrmScore {
                avg_step_correctness: 0.8,
                critical_issues: 0,
                sound_chains: 0.9
            }
            .category(),
            "reasoning"
        );
    }

    #[test]
    fn test_report_generation() {
        let mut dashboard = QualityDashboard::new();

        // Add various metrics
        dashboard.record_metric(QualityMetric::Accuracy {
            benchmark: "GSM8K".into(),
            score: 0.88,
            samples: 100,
        });
        dashboard.record_metric(QualityMetric::PrmScore {
            avg_step_correctness: 0.85,
            critical_issues: 2,
            sound_chains: 0.90,
        });
        dashboard.record_metric(QualityMetric::Triangulation {
            verification_rate: 0.75,
            avg_sources: 3.2,
            contradiction_rate: 0.05,
        });

        let report = dashboard.generate_report();
        assert!(report.score.overall > 0.0);
        assert!(!report.score.categories.is_empty());
    }
}
