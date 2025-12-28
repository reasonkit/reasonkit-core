//! # Metacognitive Calibration System
//!
//! Implements confidence calibration tracking and improvement.
//! Ensures AI confidence matches actual accuracy.
//!
//! ## Scientific Foundation
//!
//! Based on:
//! - Brier score for probabilistic forecasting
//! - Expected Calibration Error (ECE)
//! - Metacognitive sensitivity research
//!
//! ## Core Concept
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                 CALIBRATION = CONFIDENCE ≈ ACCURACY                 │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │   Perfect Calibration:                                              │
//! │   • 90% confident claims are correct 90% of the time               │
//! │   • 70% confident claims are correct 70% of the time               │
//! │   • 50% confident claims are correct 50% of the time               │
//! │                                                                     │
//! │   Overconfidence (common):                                          │
//! │   • 90% confident but only 60% accurate → needs recalibration      │
//! │                                                                     │
//! │   Underconfidence (rare):                                           │
//! │   • 50% confident but 80% accurate → can trust more                │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Metrics
//!
//! - **Brier Score**: Mean squared error of probabilistic predictions (0 = perfect)
//! - **ECE**: Expected calibration error across confidence bins
//! - **MCE**: Maximum calibration error (worst bin)
//! - **meta-d'**: Metacognitive sensitivity measure
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::calibration::{CalibrationTracker, Prediction};
//!
//! let mut tracker = CalibrationTracker::new();
//!
//! tracker.record(Prediction::new(0.9, true)); // 90% confident, was correct
//! tracker.record(Prediction::new(0.8, false)); // 80% confident, was wrong
//!
//! let report = tracker.generate_report();
//! println!("Brier Score: {:.3}", report.brier_score);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single prediction with confidence and outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,
    /// Was the prediction correct?
    pub correct: bool,
    /// Category/domain of prediction
    pub category: Option<String>,
    /// Timestamp if tracking over time
    pub timestamp: Option<u64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Prediction {
    pub fn new(confidence: f32, correct: bool) -> Self {
        Self {
            confidence: confidence.clamp(0.0, 1.0),
            correct,
            category: None,
            timestamp: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }
}

/// Confidence bin for calibration analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceBin {
    /// Bin lower bound
    pub lower: f32,
    /// Bin upper bound
    pub upper: f32,
    /// Number of predictions in bin
    pub count: usize,
    /// Average confidence in bin
    pub avg_confidence: f32,
    /// Actual accuracy in bin
    pub accuracy: f32,
    /// Calibration error for this bin (|confidence - accuracy|)
    pub calibration_error: f32,
}

/// Calibration report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    /// Total predictions analyzed
    pub total_predictions: usize,
    /// Overall accuracy
    pub overall_accuracy: f32,
    /// Average confidence
    pub avg_confidence: f32,
    /// Brier score (lower is better, 0 = perfect)
    pub brier_score: f32,
    /// Expected Calibration Error
    pub ece: f32,
    /// Maximum Calibration Error
    pub mce: f32,
    /// Confidence bins
    pub bins: Vec<ConfidenceBin>,
    /// Calibration diagnosis
    pub diagnosis: CalibrationDiagnosis,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Per-category stats
    pub category_stats: HashMap<String, CategoryCalibration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryCalibration {
    pub count: usize,
    pub accuracy: f32,
    pub avg_confidence: f32,
    pub brier_score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationDiagnosis {
    /// Well calibrated (ECE < 0.05)
    WellCalibrated,
    /// Slightly overconfident (ECE < 0.10)
    SlightlyOverconfident,
    /// Significantly overconfident (ECE < 0.20)
    Overconfident,
    /// Severely overconfident (ECE >= 0.20)
    SeverelyOverconfident,
    /// Underconfident
    Underconfident,
    /// Mixed calibration issues
    Mixed,
    /// Not enough data
    InsufficientData,
}

impl CalibrationDiagnosis {
    pub fn from_metrics(ece: f32, avg_confidence: f32, accuracy: f32) -> Self {
        if avg_confidence > accuracy + 0.15 {
            if ece >= 0.20 {
                Self::SeverelyOverconfident
            } else if ece >= 0.10 {
                Self::Overconfident
            } else {
                Self::SlightlyOverconfident
            }
        } else if avg_confidence < accuracy - 0.15 {
            Self::Underconfident
        } else if ece < 0.05 {
            Self::WellCalibrated
        } else if ece < 0.10 {
            Self::SlightlyOverconfident
        } else {
            Self::Mixed
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::WellCalibrated => "Confidence matches accuracy well",
            Self::SlightlyOverconfident => "Slightly too confident in predictions",
            Self::Overconfident => "Significantly overconfident - reduce certainty",
            Self::SeverelyOverconfident => "Severely overconfident - major recalibration needed",
            Self::Underconfident => "Too cautious - can trust predictions more",
            Self::Mixed => "Calibration varies by confidence level",
            Self::InsufficientData => "Not enough data to assess calibration",
        }
    }
}

/// Configuration for calibration tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Number of confidence bins
    pub num_bins: usize,
    /// Minimum predictions for valid analysis
    pub min_predictions: usize,
    /// ECE threshold for "well calibrated"
    pub well_calibrated_threshold: f32,
    /// Track per-category stats
    pub track_categories: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_bins: 10,
            min_predictions: 20,
            well_calibrated_threshold: 0.05,
            track_categories: true,
        }
    }
}

/// Calibration tracker
pub struct CalibrationTracker {
    pub config: CalibrationConfig,
    predictions: Vec<Prediction>,
}

impl CalibrationTracker {
    pub fn new() -> Self {
        Self {
            config: CalibrationConfig::default(),
            predictions: Vec::new(),
        }
    }

    pub fn with_config(config: CalibrationConfig) -> Self {
        Self {
            config,
            predictions: Vec::new(),
        }
    }

    /// Record a prediction
    pub fn record(&mut self, prediction: Prediction) {
        self.predictions.push(prediction);
    }

    /// Record multiple predictions
    pub fn record_batch(&mut self, predictions: Vec<Prediction>) {
        self.predictions.extend(predictions);
    }

    /// Get number of predictions
    pub fn count(&self) -> usize {
        self.predictions.len()
    }

    /// Clear all predictions
    pub fn clear(&mut self) {
        self.predictions.clear();
    }

    /// Compute Brier score
    pub fn brier_score(&self) -> f32 {
        if self.predictions.is_empty() {
            return 0.0;
        }

        self.predictions
            .iter()
            .map(|p| {
                let outcome = if p.correct { 1.0 } else { 0.0 };
                (p.confidence - outcome).powi(2)
            })
            .sum::<f32>()
            / self.predictions.len() as f32
    }

    /// Compute binned calibration
    fn compute_bins(&self) -> Vec<ConfidenceBin> {
        let num_bins = self.config.num_bins;
        let bin_width = 1.0 / num_bins as f32;

        (0..num_bins)
            .map(|i| {
                let lower = i as f32 * bin_width;
                let upper = (i + 1) as f32 * bin_width;

                let in_bin: Vec<_> = self
                    .predictions
                    .iter()
                    .filter(|p| p.confidence >= lower && p.confidence < upper.min(1.001))
                    .collect();

                let count = in_bin.len();

                if count == 0 {
                    return ConfidenceBin {
                        lower,
                        upper,
                        count: 0,
                        avg_confidence: (lower + upper) / 2.0,
                        accuracy: 0.0,
                        calibration_error: 0.0,
                    };
                }

                let avg_confidence =
                    in_bin.iter().map(|p| p.confidence).sum::<f32>() / count as f32;
                let accuracy = in_bin.iter().filter(|p| p.correct).count() as f32 / count as f32;
                let calibration_error = (avg_confidence - accuracy).abs();

                ConfidenceBin {
                    lower,
                    upper,
                    count,
                    avg_confidence,
                    accuracy,
                    calibration_error,
                }
            })
            .collect()
    }

    /// Compute Expected Calibration Error
    pub fn ece(&self) -> f32 {
        if self.predictions.is_empty() {
            return 0.0;
        }

        let bins = self.compute_bins();
        let total = self.predictions.len() as f32;

        bins.iter()
            .map(|bin| (bin.count as f32 / total) * bin.calibration_error)
            .sum()
    }

    /// Compute Maximum Calibration Error
    pub fn mce(&self) -> f32 {
        self.compute_bins()
            .iter()
            .filter(|bin| bin.count > 0)
            .map(|bin| bin.calibration_error)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Overall accuracy
    pub fn accuracy(&self) -> f32 {
        if self.predictions.is_empty() {
            return 0.0;
        }

        self.predictions.iter().filter(|p| p.correct).count() as f32 / self.predictions.len() as f32
    }

    /// Average confidence
    pub fn avg_confidence(&self) -> f32 {
        if self.predictions.is_empty() {
            return 0.0;
        }

        self.predictions.iter().map(|p| p.confidence).sum::<f32>() / self.predictions.len() as f32
    }

    /// Compute per-category stats
    fn compute_category_stats(&self) -> HashMap<String, CategoryCalibration> {
        let mut categories: HashMap<String, Vec<&Prediction>> = HashMap::new();

        for pred in &self.predictions {
            if let Some(ref cat) = pred.category {
                categories.entry(cat.clone()).or_default().push(pred);
            }
        }

        categories
            .into_iter()
            .map(|(cat, preds)| {
                let count = preds.len();
                let accuracy = preds.iter().filter(|p| p.correct).count() as f32 / count as f32;
                let avg_confidence = preds.iter().map(|p| p.confidence).sum::<f32>() / count as f32;
                let brier_score = preds
                    .iter()
                    .map(|p| {
                        let outcome = if p.correct { 1.0 } else { 0.0 };
                        (p.confidence - outcome).powi(2)
                    })
                    .sum::<f32>()
                    / count as f32;

                (
                    cat,
                    CategoryCalibration {
                        count,
                        accuracy,
                        avg_confidence,
                        brier_score,
                    },
                )
            })
            .collect()
    }

    /// Generate recommendations
    fn generate_recommendations(
        &self,
        diagnosis: CalibrationDiagnosis,
        bins: &[ConfidenceBin],
    ) -> Vec<String> {
        let mut recs = Vec::new();

        match diagnosis {
            CalibrationDiagnosis::SeverelyOverconfident => {
                recs.push("Reduce confidence by 20-30% across all predictions".into());
                recs.push("Add explicit uncertainty language (\"possibly\", \"likely\")".into());
                recs.push("Consider using --paranoid profile for verification".into());
            }
            CalibrationDiagnosis::Overconfident => {
                recs.push("Reduce confidence by 10-20%".into());
                recs.push("Add qualifiers to high-confidence claims".into());
            }
            CalibrationDiagnosis::SlightlyOverconfident => {
                recs.push("Minor confidence adjustment recommended".into());
                recs.push("Focus on claims in 80-100% confidence range".into());
            }
            CalibrationDiagnosis::Underconfident => {
                recs.push("Can trust predictions more".into());
                recs.push("Consider increasing confidence by 10-15%".into());
            }
            CalibrationDiagnosis::Mixed => {
                // Find problematic bins
                for bin in bins {
                    if bin.count >= 5
                        && bin.calibration_error > 0.15
                        && bin.avg_confidence > bin.accuracy
                    {
                        recs.push(format!(
                            "For {:.0}%-{:.0}% confidence: reduce by {:.0}%",
                            bin.lower * 100.0,
                            bin.upper * 100.0,
                            bin.calibration_error * 100.0
                        ));
                    }
                }
            }
            CalibrationDiagnosis::WellCalibrated => {
                recs.push("Calibration is good - maintain current approach".into());
            }
            CalibrationDiagnosis::InsufficientData => {
                recs.push("Need more predictions to assess calibration".into());
            }
        }

        recs
    }

    /// Generate full calibration report
    pub fn generate_report(&self) -> CalibrationReport {
        let bins = self.compute_bins();
        let brier_score = self.brier_score();
        let ece = self.ece();
        let mce = self.mce();
        let overall_accuracy = self.accuracy();
        let avg_confidence = self.avg_confidence();

        let diagnosis = if self.predictions.len() < self.config.min_predictions {
            CalibrationDiagnosis::InsufficientData
        } else {
            CalibrationDiagnosis::from_metrics(ece, avg_confidence, overall_accuracy)
        };

        let recommendations = self.generate_recommendations(diagnosis, &bins);

        let category_stats = if self.config.track_categories {
            self.compute_category_stats()
        } else {
            HashMap::new()
        };

        CalibrationReport {
            total_predictions: self.predictions.len(),
            overall_accuracy,
            avg_confidence,
            brier_score,
            ece,
            mce,
            bins,
            diagnosis,
            recommendations,
            category_stats,
        }
    }
}

impl Default for CalibrationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationReport {
    /// Format as a readable report
    pub fn format(&self) -> String {
        let mut output = String::new();

        output
            .push_str("┌─────────────────────────────────────────────────────────────────────┐\n");
        output
            .push_str("│                    CALIBRATION REPORT                               │\n");
        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");

        output.push_str(&format!(
            "│ Total Predictions: {:<50}│\n",
            self.total_predictions
        ));
        output.push_str(&format!(
            "│ Overall Accuracy:  {:.1}%{:>45}│\n",
            self.overall_accuracy * 100.0,
            ""
        ));
        output.push_str(&format!(
            "│ Avg Confidence:    {:.1}%{:>45}│\n",
            self.avg_confidence * 100.0,
            ""
        ));

        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");
        output
            .push_str("│ CALIBRATION METRICS                                                 │\n");
        output.push_str(&format!(
            "│   Brier Score: {:.3} (0=perfect, <0.25 good){:>21}│\n",
            self.brier_score, ""
        ));
        output.push_str(&format!(
            "│   ECE:         {:.3} (<0.05 well-calibrated){:>21}│\n",
            self.ece, ""
        ));
        output.push_str(&format!(
            "│   MCE:         {:.3} (worst bin){:>33}│\n",
            self.mce, ""
        ));

        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");
        output.push_str(&format!("│ DIAGNOSIS: {:?} {:>42}│\n", self.diagnosis, ""));
        output.push_str(&format!(
            "│   {}{:>52}│\n",
            self.diagnosis.description(),
            ""
        ));

        // Confidence bins visualization
        output
            .push_str("├─────────────────────────────────────────────────────────────────────┤\n");
        output
            .push_str("│ CALIBRATION CURVE                                                   │\n");
        output
            .push_str("│   Confidence → Accuracy                                             │\n");

        for bin in &self.bins {
            if bin.count > 0 {
                let bar_len = (bin.accuracy * 30.0) as usize;
                let bar = "█".repeat(bar_len);
                let gap = " ".repeat(30 - bar_len);

                let indicator = if bin.calibration_error > 0.15 {
                    "⚠"
                } else if bin.calibration_error > 0.05 {
                    "○"
                } else {
                    "✓"
                };

                output.push_str(&format!(
                    "│   {:.0}-{:.0}%: {} |{}{}| {:.0}% (n={}){}│\n",
                    bin.lower * 100.0,
                    bin.upper * 100.0,
                    indicator,
                    bar,
                    gap,
                    bin.accuracy * 100.0,
                    bin.count,
                    " ".repeat(10)
                ));
            }
        }

        // Recommendations
        if !self.recommendations.is_empty() {
            output.push_str(
                "├─────────────────────────────────────────────────────────────────────┤\n",
            );
            output.push_str(
                "│ RECOMMENDATIONS                                                     │\n",
            );
            for rec in &self.recommendations {
                output.push_str(&format!("│   • {:<62}│\n", rec));
            }
        }

        output
            .push_str("└─────────────────────────────────────────────────────────────────────┘\n");

        output
    }
}

/// Recalibration function using Platt scaling
pub fn platt_scale(confidence: f32, a: f32, b: f32) -> f32 {
    1.0 / (1.0 + (-a * confidence + b).exp())
}

/// Temperature scaling for recalibration
pub fn temperature_scale(logit: f32, temperature: f32) -> f32 {
    1.0 / (1.0 + (-logit / temperature).exp())
}

/// Confidence adjustment recommendations
pub struct ConfidenceAdjuster;

impl ConfidenceAdjuster {
    /// Adjust confidence based on calibration data
    pub fn adjust(raw_confidence: f32, diagnosis: CalibrationDiagnosis) -> f32 {
        match diagnosis {
            CalibrationDiagnosis::SeverelyOverconfident => {
                // Reduce by 25%
                raw_confidence * 0.75
            }
            CalibrationDiagnosis::Overconfident => {
                // Reduce by 15%
                raw_confidence * 0.85
            }
            CalibrationDiagnosis::SlightlyOverconfident => {
                // Reduce by 5%
                raw_confidence * 0.95
            }
            CalibrationDiagnosis::Underconfident => {
                // Increase by 10% (but cap at 0.95)
                (raw_confidence * 1.1).min(0.95)
            }
            _ => raw_confidence,
        }
    }

    /// Convert confidence to appropriate qualifier
    pub fn confidence_to_qualifier(confidence: f32) -> &'static str {
        if confidence >= 0.95 {
            "certainly"
        } else if confidence >= 0.85 {
            "very likely"
        } else if confidence >= 0.70 {
            "probably"
        } else if confidence >= 0.50 {
            "possibly"
        } else if confidence >= 0.30 {
            "unlikely"
        } else {
            "very unlikely"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_calibration() {
        let mut tracker = CalibrationTracker::new();

        // Perfectly calibrated: 90% confident, 90% correct
        for _ in 0..9 {
            tracker.record(Prediction::new(0.9, true));
        }
        tracker.record(Prediction::new(0.9, false));

        // 50% confident, 50% correct
        for _ in 0..5 {
            tracker.record(Prediction::new(0.5, true));
        }
        for _ in 0..5 {
            tracker.record(Prediction::new(0.5, false));
        }

        let report = tracker.generate_report();
        assert!(report.ece < 0.15); // Should be reasonably calibrated
    }

    #[test]
    fn test_overconfident() {
        let mut tracker = CalibrationTracker::new();

        // 90% confident but only 50% correct
        for _ in 0..25 {
            tracker.record(Prediction::new(0.9, true));
            tracker.record(Prediction::new(0.9, false));
        }

        let report = tracker.generate_report();
        assert!(matches!(
            report.diagnosis,
            CalibrationDiagnosis::Overconfident | CalibrationDiagnosis::SeverelyOverconfident
        ));
    }

    #[test]
    fn test_brier_score() {
        let mut tracker = CalibrationTracker::new();

        // Perfect predictions
        tracker.record(Prediction::new(1.0, true));
        tracker.record(Prediction::new(0.0, false));

        let brier = tracker.brier_score();
        assert!((brier - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_category_tracking() {
        let mut tracker = CalibrationTracker::with_config(CalibrationConfig {
            track_categories: true,
            ..Default::default()
        });

        tracker.record(Prediction::new(0.8, true).with_category("math"));
        tracker.record(Prediction::new(0.7, true).with_category("math"));
        tracker.record(Prediction::new(0.9, false).with_category("logic"));

        let report = tracker.generate_report();
        assert!(report.category_stats.contains_key("math"));
        assert_eq!(report.category_stats["math"].count, 2);
    }

    #[test]
    fn test_confidence_adjuster() {
        let adjusted = ConfidenceAdjuster::adjust(0.9, CalibrationDiagnosis::SeverelyOverconfident);
        assert!((adjusted - 0.675).abs() < 0.01);

        let qualifier = ConfidenceAdjuster::confidence_to_qualifier(0.85);
        assert_eq!(qualifier, "very likely");
    }
}
