//! ThinkTools Metrics Tracking System
//!
//! Continuous measurement system for tracking reasoning quality across executions.
//! Provides grades, scores, reviews, and feedback for each ThinkTool and profile.
//!
//! Key metrics tracked:
//! - Execution time (latency)
//! - Token usage (cost)
//! - Confidence scores (quality)
//! - Step completion rates (reliability)
//! - Error rates (robustness)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Individual execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Unique execution ID
    pub id: String,

    /// Protocol or profile executed
    pub protocol_or_profile: String,

    /// Whether this was a profile chain vs single protocol
    pub is_profile: bool,

    /// Execution timestamp
    pub timestamp: DateTime<Utc>,

    /// Total execution time in milliseconds
    pub duration_ms: u64,

    /// Token counts
    pub tokens_input: u32,
    pub tokens_output: u32,

    /// Final confidence score (0.0 - 1.0)
    pub confidence: f64,

    /// Number of steps completed
    pub steps_completed: usize,

    /// Number of steps total
    pub steps_total: usize,

    /// Was execution successful?
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,

    /// Per-step metrics
    pub step_metrics: Vec<StepMetric>,

    /// LLM provider used
    pub provider: String,

    /// Model used
    pub model: String,
}

/// Metrics for a single protocol step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetric {
    /// Step ID
    pub step_id: String,

    /// Protocol this step belongs to
    pub protocol_id: String,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Token count
    pub tokens: u32,

    /// Confidence achieved
    pub confidence: f64,

    /// Was step successful?
    pub success: bool,
}

/// Aggregate statistics for a protocol or profile
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregateStats {
    /// Number of executions
    pub execution_count: usize,

    /// Average duration (ms)
    pub avg_duration_ms: f64,

    /// Min/Max duration
    pub min_duration_ms: u64,
    pub max_duration_ms: u64,

    /// Average token usage
    pub avg_tokens: f64,

    /// Average confidence
    pub avg_confidence: f64,

    /// Min/Max confidence
    pub min_confidence: f64,
    pub max_confidence: f64,

    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,

    /// Standard deviation of confidence
    pub confidence_std_dev: f64,

    /// Grade (A-F based on metrics)
    pub grade: String,

    /// Score (0-100)
    pub score: u8,
}

/// Quality grade thresholds
#[derive(Debug, Clone)]
pub struct GradeThresholds {
    /// Confidence threshold for A grade
    pub a_confidence: f64,
    /// Success rate threshold for A grade
    pub a_success_rate: f64,

    /// B grade thresholds
    pub b_confidence: f64,
    pub b_success_rate: f64,

    /// C grade thresholds
    pub c_confidence: f64,
    pub c_success_rate: f64,

    /// D grade thresholds
    pub d_confidence: f64,
    pub d_success_rate: f64,
}

impl Default for GradeThresholds {
    fn default() -> Self {
        Self {
            a_confidence: 0.90,
            a_success_rate: 0.95,
            b_confidence: 0.80,
            b_success_rate: 0.85,
            c_confidence: 0.70,
            c_success_rate: 0.75,
            d_confidence: 0.60,
            d_success_rate: 0.60,
        }
    }
}

/// Metrics tracker for continuous measurement
#[derive(Debug)]
pub struct MetricsTracker {
    /// Storage path for metrics data
    storage_path: PathBuf,

    /// In-memory cache of recent records
    recent_records: Vec<ExecutionRecord>,

    /// Maximum records to keep in memory
    max_cache_size: usize,

    /// Grade thresholds
    thresholds: GradeThresholds,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new(storage_path: impl Into<PathBuf>) -> Self {
        let storage_path = storage_path.into();

        // Ensure storage directory exists
        if let Some(parent) = storage_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        Self {
            storage_path,
            recent_records: Vec::new(),
            max_cache_size: 1000,
            thresholds: GradeThresholds::default(),
        }
    }

    /// Create with default storage path
    pub fn with_default_path() -> Self {
        let path = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("reasonkit")
            .join("metrics.jsonl");
        Self::new(path)
    }

    /// Record an execution
    pub fn record(&mut self, record: ExecutionRecord) -> crate::error::Result<()> {
        // Add to cache
        self.recent_records.push(record.clone());

        // Trim cache if too large
        if self.recent_records.len() > self.max_cache_size {
            self.recent_records.remove(0);
        }

        // Persist to file (append)
        self.persist_record(&record)
    }

    /// Persist a single record to storage
    fn persist_record(&self, record: &ExecutionRecord) -> crate::error::Result<()> {
        use std::io::Write;

        let json = serde_json::to_string(record).map_err(|e| crate::error::Error::Parse {
            message: format!("Failed to serialize record: {}", e),
        })?;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.storage_path)
            .map_err(|e| crate::error::Error::IoMessage {
                message: format!("Failed to open metrics file: {}", e),
            })?;

        writeln!(file, "{}", json).map_err(|e| crate::error::Error::IoMessage {
            message: format!("Failed to write record: {}", e),
        })?;

        Ok(())
    }

    /// Load all records from storage
    pub fn load_all(&mut self) -> crate::error::Result<Vec<ExecutionRecord>> {
        if !self.storage_path.exists() {
            return Ok(Vec::new());
        }

        let content =
            fs::read_to_string(&self.storage_path).map_err(|e| crate::error::Error::IoMessage {
                message: format!("Failed to read metrics file: {}", e),
            })?;

        let records: Vec<ExecutionRecord> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();

        // Update cache
        self.recent_records = records.clone();
        if self.recent_records.len() > self.max_cache_size {
            let drain_count = self.recent_records.len() - self.max_cache_size;
            self.recent_records.drain(0..drain_count);
        }

        Ok(records)
    }

    /// Get records for a specific protocol or profile
    pub fn get_records(&self, protocol_or_profile: &str) -> Vec<&ExecutionRecord> {
        self.recent_records
            .iter()
            .filter(|r| r.protocol_or_profile == protocol_or_profile)
            .collect()
    }

    /// Calculate aggregate statistics for a protocol or profile
    pub fn calculate_stats(&self, protocol_or_profile: &str) -> AggregateStats {
        let records = self.get_records(protocol_or_profile);

        if records.is_empty() {
            return AggregateStats::default();
        }

        let count = records.len();
        let successful = records.iter().filter(|r| r.success).count();

        let durations: Vec<u64> = records.iter().map(|r| r.duration_ms).collect();
        let tokens: Vec<u32> = records
            .iter()
            .map(|r| r.tokens_input + r.tokens_output)
            .collect();
        let confidences: Vec<f64> = records.iter().map(|r| r.confidence).collect();

        let avg_duration = durations.iter().sum::<u64>() as f64 / count as f64;
        let avg_tokens = tokens.iter().sum::<u32>() as f64 / count as f64;
        let avg_confidence = confidences.iter().sum::<f64>() / count as f64;
        let success_rate = successful as f64 / count as f64;

        // Calculate standard deviation
        let variance = confidences
            .iter()
            .map(|c| (c - avg_confidence).powi(2))
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();

        // Calculate grade and score
        let (grade, score) = self.calculate_grade(avg_confidence, success_rate);

        AggregateStats {
            execution_count: count,
            avg_duration_ms: avg_duration,
            min_duration_ms: *durations.iter().min().unwrap_or(&0),
            max_duration_ms: *durations.iter().max().unwrap_or(&0),
            avg_tokens,
            avg_confidence,
            min_confidence: confidences.iter().cloned().fold(f64::INFINITY, f64::min),
            max_confidence: confidences
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            success_rate,
            confidence_std_dev: std_dev,
            grade,
            score,
        }
    }

    /// Calculate grade based on metrics
    fn calculate_grade(&self, avg_confidence: f64, success_rate: f64) -> (String, u8) {
        let t = &self.thresholds;

        if avg_confidence >= t.a_confidence && success_rate >= t.a_success_rate {
            ("A".to_string(), 95)
        } else if avg_confidence >= t.b_confidence && success_rate >= t.b_success_rate {
            let score = 80
                + ((avg_confidence - t.b_confidence) / (t.a_confidence - t.b_confidence) * 14.0)
                    as u8;
            ("B".to_string(), score.min(94))
        } else if avg_confidence >= t.c_confidence && success_rate >= t.c_success_rate {
            let score = 70
                + ((avg_confidence - t.c_confidence) / (t.b_confidence - t.c_confidence) * 9.0)
                    as u8;
            ("C".to_string(), score.min(79))
        } else if avg_confidence >= t.d_confidence && success_rate >= t.d_success_rate {
            let score = 60
                + ((avg_confidence - t.d_confidence) / (t.c_confidence - t.d_confidence) * 9.0)
                    as u8;
            ("D".to_string(), score.min(69))
        } else {
            let score = (avg_confidence * 60.0) as u8;
            ("F".to_string(), score.min(59))
        }
    }

    /// Generate a comprehensive report
    pub fn generate_report(&self) -> MetricsReport {
        let mut protocol_stats: HashMap<String, AggregateStats> = HashMap::new();
        let mut profile_stats: HashMap<String, AggregateStats> = HashMap::new();

        // Collect unique protocols and profiles
        let mut protocols: Vec<String> = Vec::new();
        let mut profiles: Vec<String> = Vec::new();

        for record in &self.recent_records {
            if record.is_profile {
                if !profiles.contains(&record.protocol_or_profile) {
                    profiles.push(record.protocol_or_profile.clone());
                }
            } else if !protocols.contains(&record.protocol_or_profile) {
                protocols.push(record.protocol_or_profile.clone());
            }
        }

        // Calculate stats for each
        for protocol in &protocols {
            protocol_stats.insert(protocol.clone(), self.calculate_stats(protocol));
        }

        for profile in &profiles {
            profile_stats.insert(profile.clone(), self.calculate_stats(profile));
        }

        // Calculate overall stats
        let overall = self.calculate_overall_stats();

        MetricsReport {
            generated_at: Utc::now(),
            total_executions: self.recent_records.len(),
            protocol_stats,
            profile_stats,
            overall,
            recommendations: self.generate_recommendations(),
        }
    }

    /// Calculate overall aggregate stats
    fn calculate_overall_stats(&self) -> AggregateStats {
        if self.recent_records.is_empty() {
            return AggregateStats::default();
        }

        let count = self.recent_records.len();
        let successful = self.recent_records.iter().filter(|r| r.success).count();

        let durations: Vec<u64> = self.recent_records.iter().map(|r| r.duration_ms).collect();
        let tokens: Vec<u32> = self
            .recent_records
            .iter()
            .map(|r| r.tokens_input + r.tokens_output)
            .collect();
        let confidences: Vec<f64> = self.recent_records.iter().map(|r| r.confidence).collect();

        let avg_duration = durations.iter().sum::<u64>() as f64 / count as f64;
        let avg_tokens = tokens.iter().sum::<u32>() as f64 / count as f64;
        let avg_confidence = confidences.iter().sum::<f64>() / count as f64;
        let success_rate = successful as f64 / count as f64;

        let variance = confidences
            .iter()
            .map(|c| (c - avg_confidence).powi(2))
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();

        let (grade, score) = self.calculate_grade(avg_confidence, success_rate);

        AggregateStats {
            execution_count: count,
            avg_duration_ms: avg_duration,
            min_duration_ms: *durations.iter().min().unwrap_or(&0),
            max_duration_ms: *durations.iter().max().unwrap_or(&0),
            avg_tokens,
            avg_confidence,
            min_confidence: confidences.iter().cloned().fold(f64::INFINITY, f64::min),
            max_confidence: confidences
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            success_rate,
            confidence_std_dev: std_dev,
            grade,
            score,
        }
    }

    /// Generate recommendations based on metrics
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        let overall = self.calculate_overall_stats();

        if overall.avg_confidence < 0.7 {
            recommendations.push(
                "Low average confidence (< 70%). Consider using deeper profiles like 'paranoid' or 'powercombo'."
                    .to_string(),
            );
        }

        if overall.success_rate < 0.8 {
            recommendations.push(
                "Low success rate (< 80%). Check for API configuration issues or rate limiting."
                    .to_string(),
            );
        }

        if overall.confidence_std_dev > 0.2 {
            recommendations.push(
                "High confidence variance. Results may be inconsistent - verify critical claims."
                    .to_string(),
            );
        }

        if overall.avg_duration_ms > 30000.0 {
            recommendations.push(
                "High average latency (> 30s). Consider using 'quick' profile for faster results."
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Metrics look healthy. Continue monitoring.".to_string());
        }

        recommendations
    }

    /// Get the storage path
    pub fn storage_path(&self) -> &Path {
        &self.storage_path
    }
}

/// Comprehensive metrics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    /// When the report was generated
    pub generated_at: DateTime<Utc>,

    /// Total number of executions
    pub total_executions: usize,

    /// Statistics per protocol
    pub protocol_stats: HashMap<String, AggregateStats>,

    /// Statistics per profile
    pub profile_stats: HashMap<String, AggregateStats>,

    /// Overall aggregate statistics
    pub overall: AggregateStats,

    /// Recommendations based on metrics
    pub recommendations: Vec<String>,
}

impl MetricsReport {
    /// Format report as human-readable text
    pub fn to_text(&self) -> String {
        let mut output = String::new();

        output
            .push_str("═══════════════════════════════════════════════════════════════════════\n");
        output.push_str("                     ReasonKit Metrics Report\n");
        output.push_str(
            "═══════════════════════════════════════════════════════════════════════\n\n",
        );

        output.push_str(&format!(
            "Generated: {}\n",
            self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        output.push_str(&format!("Total Executions: {}\n\n", self.total_executions));

        // Overall stats
        output.push_str("OVERALL METRICS:\n");
        output
            .push_str("───────────────────────────────────────────────────────────────────────\n");
        output.push_str(&format!(
            "  Grade: {} ({}/100)\n",
            self.overall.grade, self.overall.score
        ));
        output.push_str(&format!(
            "  Avg Confidence: {:.1}% (±{:.1}%)\n",
            self.overall.avg_confidence * 100.0,
            self.overall.confidence_std_dev * 100.0
        ));
        output.push_str(&format!(
            "  Success Rate: {:.1}%\n",
            self.overall.success_rate * 100.0
        ));
        output.push_str(&format!(
            "  Avg Duration: {:.0}ms\n",
            self.overall.avg_duration_ms
        ));
        output.push_str(&format!("  Avg Tokens: {:.0}\n\n", self.overall.avg_tokens));

        // Protocol stats
        if !self.protocol_stats.is_empty() {
            output.push_str("PROTOCOL METRICS:\n");
            output.push_str(
                "───────────────────────────────────────────────────────────────────────\n",
            );
            output.push_str(&format!(
                "{:<15} {:>6} {:>10} {:>10} {:>8} {:>8}\n",
                "Protocol", "Grade", "Confidence", "Success", "Duration", "Runs"
            ));
            output.push_str(&format!(
                "{:<15} {:>6} {:>10} {:>10} {:>8} {:>8}\n",
                "───────────", "─────", "──────────", "───────", "────────", "────"
            ));

            for (name, stats) in &self.protocol_stats {
                output.push_str(&format!(
                    "{:<15} {:>6} {:>9.1}% {:>9.1}% {:>7.0}ms {:>8}\n",
                    name,
                    &stats.grade,
                    stats.avg_confidence * 100.0,
                    stats.success_rate * 100.0,
                    stats.avg_duration_ms,
                    stats.execution_count
                ));
            }
            output.push('\n');
        }

        // Profile stats
        if !self.profile_stats.is_empty() {
            output.push_str("PROFILE METRICS:\n");
            output.push_str(
                "───────────────────────────────────────────────────────────────────────\n",
            );
            output.push_str(&format!(
                "{:<15} {:>6} {:>10} {:>10} {:>8} {:>8}\n",
                "Profile", "Grade", "Confidence", "Success", "Duration", "Runs"
            ));
            output.push_str(&format!(
                "{:<15} {:>6} {:>10} {:>10} {:>8} {:>8}\n",
                "───────────", "─────", "──────────", "───────", "────────", "────"
            ));

            for (name, stats) in &self.profile_stats {
                output.push_str(&format!(
                    "{:<15} {:>6} {:>9.1}% {:>9.1}% {:>7.0}ms {:>8}\n",
                    name,
                    &stats.grade,
                    stats.avg_confidence * 100.0,
                    stats.success_rate * 100.0,
                    stats.avg_duration_ms,
                    stats.execution_count
                ));
            }
            output.push('\n');
        }

        // Recommendations
        output.push_str("RECOMMENDATIONS:\n");
        output
            .push_str("───────────────────────────────────────────────────────────────────────\n");
        for rec in &self.recommendations {
            output.push_str(&format!("  • {}\n", rec));
        }

        output
    }

    /// Format report as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Builder for creating execution records
#[derive(Debug, Default)]
pub struct ExecutionRecordBuilder {
    protocol_or_profile: String,
    is_profile: bool,
    start_time: Option<std::time::Instant>,
    tokens_input: u32,
    tokens_output: u32,
    confidence: f64,
    steps_completed: usize,
    steps_total: usize,
    success: bool,
    error: Option<String>,
    step_metrics: Vec<StepMetric>,
    provider: String,
    model: String,
}

impl ExecutionRecordBuilder {
    pub fn new(protocol_or_profile: &str, is_profile: bool) -> Self {
        Self {
            protocol_or_profile: protocol_or_profile.to_string(),
            is_profile,
            start_time: Some(std::time::Instant::now()),
            provider: "unknown".to_string(),
            model: "unknown".to_string(),
            ..Default::default()
        }
    }

    pub fn tokens(mut self, input: u32, output: u32) -> Self {
        self.tokens_input = input;
        self.tokens_output = output;
        self
    }

    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn steps(mut self, completed: usize, total: usize) -> Self {
        self.steps_completed = completed;
        self.steps_total = total;
        self
    }

    pub fn success(mut self, success: bool) -> Self {
        self.success = success;
        self
    }

    pub fn error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self.success = false;
        self
    }

    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn add_step_metric(mut self, metric: StepMetric) -> Self {
        self.step_metrics.push(metric);
        self
    }

    pub fn build(self) -> ExecutionRecord {
        let duration_ms = self
            .start_time
            .map(|s| s.elapsed().as_millis() as u64)
            .unwrap_or(0);

        ExecutionRecord {
            id: uuid::Uuid::new_v4().to_string(),
            protocol_or_profile: self.protocol_or_profile,
            is_profile: self.is_profile,
            timestamp: Utc::now(),
            duration_ms,
            tokens_input: self.tokens_input,
            tokens_output: self.tokens_output,
            confidence: self.confidence,
            steps_completed: self.steps_completed,
            steps_total: self.steps_total,
            success: self.success,
            error: self.error,
            step_metrics: self.step_metrics,
            provider: self.provider,
            model: self.model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_metrics_tracker_creation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        let tracker = MetricsTracker::new(&path);
        assert!(tracker.recent_records.is_empty());
    }

    #[test]
    fn test_record_and_retrieve() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        let mut tracker = MetricsTracker::new(&path);

        let record = ExecutionRecordBuilder::new("gigathink", false)
            .tokens(100, 200)
            .confidence(0.85)
            .steps(3, 3)
            .success(true)
            .provider("anthropic")
            .model("claude-sonnet-4-5")
            .build();

        tracker.record(record).unwrap();

        let records = tracker.get_records("gigathink");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].confidence, 0.85);
    }

    #[test]
    fn test_aggregate_stats() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        let mut tracker = MetricsTracker::new(&path);

        // Add multiple records
        for confidence in [0.7, 0.8, 0.9] {
            let record = ExecutionRecordBuilder::new("laserlogic", false)
                .confidence(confidence)
                .success(true)
                .build();
            tracker.record(record).unwrap();
        }

        let stats = tracker.calculate_stats("laserlogic");
        assert_eq!(stats.execution_count, 3);
        assert!((stats.avg_confidence - 0.8).abs() < 0.01);
        assert_eq!(stats.success_rate, 1.0);
    }

    #[test]
    fn test_grade_calculation() {
        let dir = tempdir().unwrap();
        let tracker = MetricsTracker::new(dir.path().join("metrics.jsonl"));

        // A grade
        let (grade, score) = tracker.calculate_grade(0.95, 0.98);
        assert_eq!(grade, "A");
        assert!(score >= 95);

        // B grade
        let (grade, _score) = tracker.calculate_grade(0.82, 0.88);
        assert_eq!(grade, "B");

        // F grade
        let (grade, score) = tracker.calculate_grade(0.3, 0.4);
        assert_eq!(grade, "F");
        assert!(score < 60);
    }

    #[test]
    fn test_report_generation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        let mut tracker = MetricsTracker::new(&path);

        // Add some records
        let record = ExecutionRecordBuilder::new("paranoid", true)
            .confidence(0.92)
            .success(true)
            .build();
        tracker.record(record).unwrap();

        let report = tracker.generate_report();
        assert_eq!(report.total_executions, 1);
        assert!(!report.recommendations.is_empty());

        let text = report.to_text();
        assert!(text.contains("OVERALL METRICS"));
        assert!(text.contains("RECOMMENDATIONS"));
    }
}
