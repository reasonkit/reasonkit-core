//! # Benchmark Harness for ThinkTools Evaluation
//!
//! Provides infrastructure to measure reasoning quality improvements
//! against established benchmarks (GSM8K, MATH, TruthfulQA, etc.)
//!
//! ## Supported Benchmarks
//!
//! | Benchmark | Type | Metric | Target |
//! |-----------|------|--------|--------|
//! | GSM8K | Math reasoning | Accuracy | 85.9% |
//! | MATH | Advanced math | Accuracy | 36.5% |
//! | TruthfulQA | Factuality | MC1/MC2 | 72% |
//! | Game of 24 | Creative | Success rate | 60%+ |
//! | ARC-C | Science | Accuracy | 90% |

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Benchmark problem from evaluation set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkProblem {
    /// Unique identifier
    pub id: String,
    /// Problem statement
    pub question: String,
    /// Expected answer(s)
    pub answer: Answer,
    /// Optional solution steps
    pub solution: Option<String>,
    /// Problem category/topic
    pub category: Option<String>,
    /// Difficulty level (1-5)
    pub difficulty: Option<u8>,
}

/// Answer type - handles different benchmark formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Answer {
    /// Numeric answer (GSM8K, MATH)
    Numeric(f64),
    /// Text answer
    Text(String),
    /// Multiple choice (ARC, TruthfulQA)
    MultipleChoice { correct: char, options: Vec<String> },
    /// List of acceptable answers
    MultiAnswer(Vec<String>),
}

/// Result of evaluating a single problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub problem_id: String,
    pub correct: bool,
    pub predicted: String,
    pub expected: String,
    pub confidence: f32,
    pub reasoning_steps: usize,
    pub latency_ms: u64,
    pub tokens_used: usize,
    /// Problem category for category-level accuracy
    #[serde(default)]
    pub category: Option<String>,
    /// Problem difficulty for difficulty-level accuracy
    #[serde(default)]
    pub difficulty: Option<u8>,
}

/// Aggregate benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub benchmark_name: String,
    pub total_problems: usize,
    pub correct: usize,
    pub accuracy: f32,
    pub avg_confidence: f32,
    pub avg_latency_ms: f64,
    pub total_tokens: usize,
    /// Accuracy by category
    pub category_accuracy: HashMap<String, f32>,
    /// Accuracy by difficulty
    pub difficulty_accuracy: HashMap<u8, f32>,
    /// Individual results
    pub results: Vec<EvaluationResult>,
    /// Calibration metrics
    pub calibration: CalibrationMetrics,
}

/// Calibration metrics for confidence assessment
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Brier score (lower is better, 0 = perfect)
    pub brier_score: f32,
    /// Expected calibration error
    pub ece: f32,
    /// Overconfidence ratio (predictions with high conf but wrong)
    pub overconfidence_ratio: f32,
    /// Confidence histogram bins
    pub confidence_bins: Vec<ConfidenceBin>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceBin {
    pub range_start: f32,
    pub range_end: f32,
    pub count: usize,
    pub accuracy: f32,
}

impl CalibrationMetrics {
    pub fn compute(results: &[EvaluationResult]) -> Self {
        if results.is_empty() {
            return Self::default();
        }

        // Brier score
        let brier_score: f32 = results
            .iter()
            .map(|r| {
                let outcome = if r.correct { 1.0 } else { 0.0 };
                (r.confidence - outcome).powi(2)
            })
            .sum::<f32>()
            / results.len() as f32;

        // ECE with 10 bins
        let num_bins = 10;
        let mut bins: Vec<Vec<&EvaluationResult>> = vec![Vec::new(); num_bins];

        for result in results {
            let bin_idx = ((result.confidence * num_bins as f32) as usize).min(num_bins - 1);
            bins[bin_idx].push(result);
        }

        let mut ece = 0.0f32;
        let mut confidence_bins = Vec::with_capacity(num_bins);

        for (i, bin) in bins.iter().enumerate() {
            let range_start = i as f32 / num_bins as f32;
            let range_end = (i + 1) as f32 / num_bins as f32;

            if bin.is_empty() {
                confidence_bins.push(ConfidenceBin {
                    range_start,
                    range_end,
                    count: 0,
                    accuracy: 0.0,
                });
                continue;
            }

            let bin_accuracy = bin.iter().filter(|r| r.correct).count() as f32 / bin.len() as f32;
            let bin_confidence: f32 =
                bin.iter().map(|r| r.confidence).sum::<f32>() / bin.len() as f32;

            ece +=
                (bin.len() as f32 / results.len() as f32) * (bin_accuracy - bin_confidence).abs();

            confidence_bins.push(ConfidenceBin {
                range_start,
                range_end,
                count: bin.len(),
                accuracy: bin_accuracy,
            });
        }

        // Overconfidence ratio: high confidence (>0.8) but wrong
        let overconfidence_ratio = results
            .iter()
            .filter(|r| r.confidence > 0.8 && !r.correct)
            .count() as f32
            / results.iter().filter(|r| r.confidence > 0.8).count().max(1) as f32;

        Self {
            brier_score,
            ece,
            overconfidence_ratio,
            confidence_bins,
        }
    }
}

impl BenchmarkResults {
    pub fn compute(benchmark_name: &str, results: Vec<EvaluationResult>) -> Self {
        let total_problems = results.len();
        let correct = results.iter().filter(|r| r.correct).count();
        let accuracy = if total_problems > 0 {
            correct as f32 / total_problems as f32
        } else {
            0.0
        };

        let avg_confidence = if total_problems > 0 {
            results.iter().map(|r| r.confidence).sum::<f32>() / total_problems as f32
        } else {
            0.0
        };

        let avg_latency_ms = if total_problems > 0 {
            results.iter().map(|r| r.latency_ms).sum::<u64>() as f64 / total_problems as f64
        } else {
            0.0
        };

        let total_tokens = results.iter().map(|r| r.tokens_used).sum();

        let calibration = CalibrationMetrics::compute(&results);

        // Compute category-level accuracy
        let mut category_counts: HashMap<String, (usize, usize)> = HashMap::new();
        for result in &results {
            if let Some(ref cat) = result.category {
                let entry = category_counts.entry(cat.clone()).or_insert((0, 0));
                entry.0 += 1; // total
                if result.correct {
                    entry.1 += 1; // correct
                }
            }
        }
        let category_accuracy: HashMap<String, f32> = category_counts
            .into_iter()
            .map(|(cat, (total, correct))| {
                (
                    cat,
                    if total > 0 {
                        correct as f32 / total as f32
                    } else {
                        0.0
                    },
                )
            })
            .collect();

        // Compute difficulty-level accuracy
        let mut difficulty_counts: HashMap<u8, (usize, usize)> = HashMap::new();
        for result in &results {
            if let Some(diff) = result.difficulty {
                let entry = difficulty_counts.entry(diff).or_insert((0, 0));
                entry.0 += 1; // total
                if result.correct {
                    entry.1 += 1; // correct
                }
            }
        }
        let difficulty_accuracy: HashMap<u8, f32> = difficulty_counts
            .into_iter()
            .map(|(diff, (total, correct))| {
                (
                    diff,
                    if total > 0 {
                        correct as f32 / total as f32
                    } else {
                        0.0
                    },
                )
            })
            .collect();

        Self {
            benchmark_name: benchmark_name.to_string(),
            total_problems,
            correct,
            accuracy,
            avg_confidence,
            avg_latency_ms,
            total_tokens,
            category_accuracy,
            difficulty_accuracy,
            results,
            calibration,
        }
    }

    /// Generate a comparison report against baseline
    pub fn compare(&self, baseline: &BenchmarkResults) -> ComparisonReport {
        ComparisonReport {
            benchmark: self.benchmark_name.clone(),
            baseline_accuracy: baseline.accuracy,
            current_accuracy: self.accuracy,
            delta_accuracy: self.accuracy - baseline.accuracy,
            baseline_brier: baseline.calibration.brier_score,
            current_brier: self.calibration.brier_score,
            delta_brier: self.calibration.brier_score - baseline.calibration.brier_score,
            latency_ratio: self.avg_latency_ms / baseline.avg_latency_ms.max(1.0),
            significant_improvement: (self.accuracy - baseline.accuracy) > 0.02,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub benchmark: String,
    pub baseline_accuracy: f32,
    pub current_accuracy: f32,
    pub delta_accuracy: f32,
    pub baseline_brier: f32,
    pub current_brier: f32,
    pub delta_brier: f32,
    pub latency_ratio: f64,
    pub significant_improvement: bool,
}

impl std::fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let delta_sign = if self.delta_accuracy >= 0.0 { "+" } else { "" };
        let brier_sign = if self.delta_brier <= 0.0 { "+" } else { "-" };

        write!(
            f,
            r#"
┌─────────────────────────────────────────────────────────────────────┐
│ BENCHMARK COMPARISON: {}
├─────────────────────────────────────────────────────────────────────┤
│ Accuracy:    {:.1}% → {:.1}% ({}{:.1}%)  {}
│ Brier Score: {:.3} → {:.3} ({}{:.3})
│ Latency:     {:.1}x baseline
│ Significant: {}
└─────────────────────────────────────────────────────────────────────┘"#,
            self.benchmark,
            self.baseline_accuracy * 100.0,
            self.current_accuracy * 100.0,
            delta_sign,
            self.delta_accuracy * 100.0,
            if self.significant_improvement {
                "✓"
            } else {
                "○"
            },
            self.baseline_brier,
            self.current_brier,
            brier_sign,
            self.delta_brier.abs(),
            self.latency_ratio,
            if self.significant_improvement {
                "YES - Improvement detected"
            } else {
                "NO - Within noise margin"
            }
        )
    }
}

/// GSM8K-specific loader
pub mod gsm8k {
    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    /// Load GSM8K problems from JSONL file
    pub fn load_problems(path: impl AsRef<Path>) -> anyhow::Result<Vec<BenchmarkProblem>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut problems = Vec::new();

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let raw: serde_json::Value = serde_json::from_str(&line)?;

            let question = raw["question"].as_str().unwrap_or_default().to_string();

            let answer_str = raw["answer"].as_str().unwrap_or_default();
            // GSM8K answers end with #### <number>
            let answer = extract_gsm8k_answer(answer_str);

            problems.push(BenchmarkProblem {
                id: format!("gsm8k_{}", idx),
                question,
                answer: Answer::Numeric(answer),
                solution: Some(answer_str.to_string()),
                category: None,
                difficulty: None,
            });
        }

        Ok(problems)
    }

    fn extract_gsm8k_answer(answer_str: &str) -> f64 {
        // GSM8K format: "... #### 42"
        if let Some(pos) = answer_str.rfind("####") {
            let num_str = answer_str[pos + 4..].trim();
            // Remove commas from numbers like "1,234"
            let cleaned = num_str.replace(',', "");
            cleaned.parse().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Check if model answer matches expected
    pub fn check_answer(predicted: &str, expected: f64) -> bool {
        // Extract number from predicted answer
        let predicted_num = extract_number_from_response(predicted);

        // Allow small floating point tolerance
        (predicted_num - expected).abs() < 0.01
    }

    fn extract_number_from_response(response: &str) -> f64 {
        // Try to find #### marker first
        if let Some(pos) = response.rfind("####") {
            let after = &response[pos + 4..];
            if let Some(num) = extract_first_number(after) {
                return num;
            }
        }

        // Try "answer is" pattern
        let patterns = ["answer is", "= ", "equals", "result:"];
        for pattern in patterns {
            if let Some(pos) = response.to_lowercase().rfind(pattern) {
                let after = &response[pos + pattern.len()..];
                if let Some(num) = extract_first_number(after) {
                    return num;
                }
            }
        }

        // Last resort: find last number in response
        extract_last_number(response).unwrap_or(0.0)
    }

    fn extract_first_number(s: &str) -> Option<f64> {
        let mut num_str = String::new();
        let mut in_number = false;

        for c in s.chars() {
            if c.is_ascii_digit() || c == '.' || c == '-' {
                in_number = true;
                num_str.push(c);
            } else if c == ',' && in_number {
                // Skip commas in numbers
                continue;
            } else if in_number {
                break;
            }
        }

        num_str.parse().ok()
    }

    fn extract_last_number(s: &str) -> Option<f64> {
        let mut last_num = None;
        let mut current = String::new();

        for c in s.chars() {
            if c.is_ascii_digit() || c == '.' || c == '-' {
                current.push(c);
            } else if c == ',' && !current.is_empty() {
                continue;
            } else if !current.is_empty() {
                if let Ok(n) = current.parse() {
                    last_num = Some(n);
                }
                current.clear();
            }
        }

        if !current.is_empty() {
            if let Ok(n) = current.parse() {
                last_num = Some(n);
            }
        }

        last_num
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_gsm8k_answer_extraction() {
            assert_eq!(extract_gsm8k_answer("The answer is #### 42"), 42.0);
            assert_eq!(
                extract_gsm8k_answer("Step 1... Step 2... #### 1234"),
                1234.0
            );
            assert_eq!(extract_gsm8k_answer("#### 1,234"), 1234.0);
        }

        #[test]
        fn test_check_answer() {
            assert!(check_answer("The answer is 42", 42.0));
            assert!(check_answer("#### 42", 42.0));
            assert!(!check_answer("The answer is 43", 42.0));
        }
    }
}

/// Benchmark runner
pub struct BenchmarkRunner {
    pub problems: Vec<BenchmarkProblem>,
    pub benchmark_name: String,
}

impl BenchmarkRunner {
    pub fn new(benchmark_name: impl Into<String>, problems: Vec<BenchmarkProblem>) -> Self {
        Self {
            problems,
            benchmark_name: benchmark_name.into(),
        }
    }

    /// Load GSM8K benchmark
    pub fn gsm8k(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let problems = gsm8k::load_problems(path)?;
        Ok(Self::new("GSM8K", problems))
    }

    /// Run evaluation with a given evaluator function
    pub async fn run<F, Fut>(&self, evaluator: F, limit: Option<usize>) -> BenchmarkResults
    where
        F: Fn(BenchmarkProblem) -> Fut,
        Fut: std::future::Future<Output = EvaluationResult>,
    {
        let problems = match limit {
            Some(n) => self.problems.iter().take(n).cloned().collect::<Vec<_>>(),
            None => self.problems.clone(),
        };

        let mut results = Vec::with_capacity(problems.len());

        for problem in problems {
            let result = evaluator(problem).await;
            results.push(result);
        }

        BenchmarkResults::compute(&self.benchmark_name, results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_metrics() {
        let results = vec![
            EvaluationResult {
                problem_id: "1".into(),
                correct: true,
                predicted: "42".into(),
                expected: "42".into(),
                confidence: 0.9,
                reasoning_steps: 3,
                latency_ms: 100,
                tokens_used: 500,
                category: Some("arithmetic".into()),
                difficulty: Some(1),
            },
            EvaluationResult {
                problem_id: "2".into(),
                correct: false,
                predicted: "41".into(),
                expected: "42".into(),
                confidence: 0.8,
                reasoning_steps: 3,
                latency_ms: 120,
                tokens_used: 520,
                category: Some("arithmetic".into()),
                difficulty: Some(2),
            },
        ];

        let metrics = CalibrationMetrics::compute(&results);
        assert!(metrics.brier_score > 0.0);
        assert!(metrics.brier_score < 1.0);
    }

    #[test]
    fn test_comparison_report() {
        let baseline = BenchmarkResults {
            benchmark_name: "GSM8K".into(),
            total_problems: 100,
            correct: 78,
            accuracy: 0.78,
            avg_confidence: 0.75,
            avg_latency_ms: 500.0,
            total_tokens: 50000,
            category_accuracy: HashMap::new(),
            difficulty_accuracy: HashMap::new(),
            results: vec![],
            calibration: CalibrationMetrics::default(),
        };

        let improved = BenchmarkResults {
            benchmark_name: "GSM8K".into(),
            total_problems: 100,
            correct: 86,
            accuracy: 0.86,
            avg_confidence: 0.82,
            avg_latency_ms: 800.0,
            total_tokens: 75000,
            category_accuracy: HashMap::new(),
            difficulty_accuracy: HashMap::new(),
            results: vec![],
            calibration: CalibrationMetrics::default(),
        };

        let report = improved.compare(&baseline);
        assert!(report.significant_improvement);
        assert!((report.delta_accuracy - 0.08).abs() < 0.001);
    }
}
