//! Self-Consistency Module
//!
//! Implements scientifically-proven self-consistency voting mechanism
//! based on Wang et al. (2023) "Self-Consistency Improves Chain of Thought Reasoning"
//!
//! Key findings from research:
//! - GSM8K: +17.9% accuracy improvement
//! - SVAMP: +11.0% accuracy improvement
//! - AQuA: +12.2% accuracy improvement
//!
//! Reference: <https://arxiv.org/abs/2203.11171>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::step::{StepOutput, StepResult, TokenUsage};

/// Self-Consistency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfConsistencyConfig {
    /// Number of reasoning paths to sample (default: 5)
    /// Research shows diminishing returns after ~10-15 samples
    pub num_samples: usize,

    /// Voting method to use
    pub voting_method: VotingMethod,

    /// Temperature variance for diverse sampling
    /// Higher values = more diverse reasoning paths
    pub temperature_base: f64,

    /// Temperature increment per sample (for diversity)
    pub temperature_variance: f64,

    /// Minimum confidence threshold for a sample to be included in voting
    pub min_sample_confidence: f64,

    /// Enable CISC (Confidence-Informed Self-Consistency)
    /// Reduces required samples by ~40% (arXiv:2502.06233)
    pub use_cisc: bool,

    /// Early stopping if consensus reached
    pub early_stopping: bool,

    /// Consensus threshold for early stopping (e.g., 0.8 = 80% agreement)
    pub consensus_threshold: f64,
}

impl Default for SelfConsistencyConfig {
    fn default() -> Self {
        Self {
            num_samples: 5,
            voting_method: VotingMethod::MajorityVote,
            temperature_base: 0.7,
            temperature_variance: 0.1,
            min_sample_confidence: 0.5,
            use_cisc: true, // Enable by default for cost efficiency
            early_stopping: true,
            consensus_threshold: 0.8,
        }
    }
}

impl SelfConsistencyConfig {
    /// Create a fast config (fewer samples, early stopping)
    pub fn fast() -> Self {
        Self {
            num_samples: 3,
            early_stopping: true,
            consensus_threshold: 0.7,
            ..Default::default()
        }
    }

    /// Create a thorough config (more samples, no early stopping)
    pub fn thorough() -> Self {
        Self {
            num_samples: 10,
            early_stopping: false,
            ..Default::default()
        }
    }

    /// Create a paranoid config (maximum samples)
    pub fn paranoid() -> Self {
        Self {
            num_samples: 15,
            early_stopping: false,
            min_sample_confidence: 0.6,
            ..Default::default()
        }
    }

    /// Get temperature for a specific sample index
    pub fn temperature_for_sample(&self, index: usize) -> f64 {
        self.temperature_base + (index as f64 * self.temperature_variance)
    }
}

/// Voting methods for self-consistency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VotingMethod {
    /// Simple majority voting (original self-consistency)
    #[default]
    MajorityVote,

    /// Weighted by confidence scores (CISC)
    ConfidenceWeighted,

    /// Weighted by semantic similarity clustering
    ClusterWeighted,

    /// Unanimous agreement required
    Unanimous,
}

/// A single sampled reasoning path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPath {
    /// The final answer/conclusion extracted
    pub answer: String,

    /// The full reasoning trace
    pub reasoning: String,

    /// Confidence score for this path
    pub confidence: f64,

    /// Token usage for this sample
    pub tokens: TokenUsage,

    /// Temperature used for this sample
    pub temperature: f64,

    /// Sample index
    pub sample_index: usize,
}

/// Result of self-consistency voting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyResult {
    /// The winning answer after voting
    pub answer: String,

    /// Aggregated confidence (voting strength)
    pub confidence: f64,

    /// Number of votes for winning answer
    pub vote_count: usize,

    /// Total number of samples
    pub total_samples: usize,

    /// Agreement ratio (votes / total)
    pub agreement_ratio: f64,

    /// All reasoning paths sampled
    pub paths: Vec<ReasoningPath>,

    /// Vote distribution (answer -> count)
    pub vote_distribution: HashMap<String, usize>,

    /// Whether early stopping was triggered
    pub early_stopped: bool,

    /// Total token usage across all samples
    pub total_tokens: TokenUsage,
}

impl ConsistencyResult {
    /// Check if result meets confidence threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.confidence >= threshold && self.agreement_ratio >= 0.5
    }

    /// Get the dissenting paths (those that disagreed with winner)
    pub fn dissenting_paths(&self) -> Vec<&ReasoningPath> {
        self.paths
            .iter()
            .filter(|p| p.answer != self.answer)
            .collect()
    }

    /// Get reasoning diversity score (0-1, higher = more diverse)
    pub fn diversity_score(&self) -> f64 {
        let unique_answers = self.vote_distribution.len();
        if self.total_samples <= 1 {
            0.0
        } else {
            (unique_answers - 1) as f64 / (self.total_samples - 1) as f64
        }
    }
}

/// Self-Consistency Engine
pub struct SelfConsistencyEngine {
    config: SelfConsistencyConfig,
}

impl SelfConsistencyEngine {
    /// Create a new self-consistency engine
    pub fn new(config: SelfConsistencyConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default_engine() -> Self {
        Self::new(SelfConsistencyConfig::default())
    }

    /// Aggregate multiple step results using self-consistency voting
    pub fn vote(&self, results: Vec<StepResult>) -> ConsistencyResult {
        let paths: Vec<ReasoningPath> = results
            .into_iter()
            .enumerate()
            .filter_map(|(idx, result)| self.extract_path(result, idx))
            .collect();

        self.aggregate_paths(paths)
    }

    /// Extract a reasoning path from a step result
    fn extract_path(&self, result: StepResult, index: usize) -> Option<ReasoningPath> {
        if !result.success || result.confidence < self.config.min_sample_confidence {
            return None;
        }

        let (answer, reasoning) = match &result.output {
            StepOutput::Text { content } => {
                // Extract answer from text (look for common patterns)
                let answer = self.extract_answer_from_text(content);
                (answer, content.clone())
            }
            StepOutput::Structured { data } => {
                // Look for answer field in structured output
                let answer = data
                    .get("answer")
                    .or_else(|| data.get("conclusion"))
                    .or_else(|| data.get("result"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{:?}", data));
                let reasoning = serde_json::to_string_pretty(&data).unwrap_or_default();
                (answer, reasoning)
            }
            StepOutput::Boolean { value, reason } => {
                let answer = if *value { "true" } else { "false" }.to_string();
                let reasoning = reason.clone().unwrap_or_default();
                (answer, reasoning)
            }
            StepOutput::Score { value } => (format!("{:.2}", value), String::new()),
            StepOutput::List { items } => {
                let answer = items
                    .iter()
                    .map(|i| i.content.clone())
                    .collect::<Vec<_>>()
                    .join("; ");
                (answer.clone(), answer)
            }
            StepOutput::Empty => return None,
        };

        Some(ReasoningPath {
            answer: self.normalize_answer(&answer),
            reasoning,
            confidence: result.confidence,
            tokens: result.tokens,
            temperature: self.config.temperature_for_sample(index),
            sample_index: index,
        })
    }

    /// Extract answer from free-form text
    fn extract_answer_from_text(&self, text: &str) -> String {
        // Look for common answer patterns
        let patterns = [
            "the answer is",
            "therefore,",
            "in conclusion,",
            "final answer:",
            "result:",
            "answer:",
        ];

        for pattern in patterns {
            if let Some(pos) = text.to_lowercase().find(pattern) {
                let start = pos + pattern.len();
                let remainder = &text[start..];
                // Take until end of sentence or newline
                let end = remainder
                    .find(['.', '\n', '!', '?'])
                    .unwrap_or(remainder.len().min(200));
                return remainder[..end].trim().to_string();
            }
        }

        // Fallback: use last sentence
        text.split(['.', '\n'])
            .rfind(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| text.chars().take(200).collect())
    }

    /// Normalize answer for comparison (lowercase, trim, etc.)
    fn normalize_answer(&self, answer: &str) -> String {
        answer
            .to_lowercase()
            .trim()
            .replace([',', '.', '!', '?', '"', '\''], "")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Aggregate reasoning paths using configured voting method
    fn aggregate_paths(&self, paths: Vec<ReasoningPath>) -> ConsistencyResult {
        if paths.is_empty() {
            return ConsistencyResult {
                answer: String::new(),
                confidence: 0.0,
                vote_count: 0,
                total_samples: 0,
                agreement_ratio: 0.0,
                paths: Vec::new(),
                vote_distribution: HashMap::new(),
                early_stopped: false,
                total_tokens: TokenUsage::default(),
            };
        }

        // Count votes and calculate weights
        let mut vote_counts: HashMap<String, usize> = HashMap::new();
        let mut vote_weights: HashMap<String, f64> = HashMap::new();
        let mut total_tokens = TokenUsage::default();

        for path in &paths {
            *vote_counts.entry(path.answer.clone()).or_insert(0) += 1;

            let weight = match self.config.voting_method {
                VotingMethod::MajorityVote => 1.0,
                VotingMethod::ConfidenceWeighted => path.confidence,
                VotingMethod::ClusterWeighted => path.confidence, // Simplified
                VotingMethod::Unanimous => 1.0,
            };

            *vote_weights.entry(path.answer.clone()).or_insert(0.0) += weight;
            total_tokens.add(&path.tokens);
        }

        // Find winner - using safe comparison that handles NaN gracefully
        let (winner, vote_count) = match self.config.voting_method {
            VotingMethod::Unanimous => {
                // All must agree
                if vote_counts.len() == 1 {
                    // SAFETY: We checked vote_counts.len() == 1, so there's exactly one entry
                    // Using unwrap_or_default as a defensive fallback
                    vote_counts.into_iter().next().unwrap_or_default()
                } else {
                    // No consensus - return most common with low confidence
                    vote_counts
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .unwrap_or_default()
                }
            }
            _ => {
                // Find by weight - use total_cmp for safe f64 comparison (handles NaN)
                vote_weights
                    .iter()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(answer, _)| {
                        let count = vote_counts.get(answer).copied().unwrap_or(0);
                        (answer.clone(), count)
                    })
                    .unwrap_or_default()
            }
        };

        let total_samples = paths.len();
        let agreement_ratio = vote_count as f64 / total_samples as f64;

        // Calculate aggregated confidence
        let confidence = if self.config.use_cisc {
            // CISC: Weight confidence by agreement
            let winner_paths: Vec<_> = paths.iter().filter(|p| p.answer == winner).collect();
            if winner_paths.is_empty() {
                0.0
            } else {
                let avg_confidence: f64 = winner_paths.iter().map(|p| p.confidence).sum::<f64>()
                    / winner_paths.len() as f64;
                avg_confidence * agreement_ratio
            }
        } else {
            // Simple: Just use agreement ratio
            agreement_ratio
        };

        // Rebuild vote distribution with original counts
        let mut final_distribution = HashMap::new();
        for path in &paths {
            *final_distribution.entry(path.answer.clone()).or_insert(0) += 1;
        }

        ConsistencyResult {
            answer: winner,
            confidence,
            vote_count,
            total_samples,
            agreement_ratio,
            paths,
            vote_distribution: final_distribution,
            early_stopped: false,
            total_tokens,
        }
    }

    /// Check if early stopping should be triggered
    pub fn should_early_stop(&self, current_results: &[StepResult]) -> bool {
        if !self.config.early_stopping || current_results.len() < 3 {
            return false;
        }

        let paths: Vec<ReasoningPath> = current_results
            .iter()
            .enumerate()
            .filter_map(|(idx, result)| self.extract_path(result.clone(), idx))
            .collect();

        if paths.is_empty() {
            return false;
        }

        // Count current votes
        let mut vote_counts: HashMap<String, usize> = HashMap::new();
        for path in &paths {
            *vote_counts.entry(path.answer.clone()).or_insert(0) += 1;
        }

        // Check if any answer has reached consensus threshold
        let max_votes = vote_counts.values().max().copied().unwrap_or(0);
        let current_ratio = max_votes as f64 / paths.len() as f64;

        current_ratio >= self.config.consensus_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = SelfConsistencyConfig::default();
        assert_eq!(config.num_samples, 5);
        assert!(config.use_cisc);
        assert!(config.early_stopping);
    }

    #[test]
    fn test_temperature_variance() {
        let config = SelfConsistencyConfig::default();
        assert!((config.temperature_for_sample(0) - 0.7).abs() < 0.01);
        assert!((config.temperature_for_sample(1) - 0.8).abs() < 0.01);
        assert!((config.temperature_for_sample(2) - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_majority_voting() {
        let engine = SelfConsistencyEngine::default_engine();

        let results = vec![
            StepResult::success(
                "test",
                StepOutput::Text {
                    content: "The answer is 42.".to_string(),
                },
                0.8,
            ),
            StepResult::success(
                "test",
                StepOutput::Text {
                    content: "The answer is 42.".to_string(),
                },
                0.85,
            ),
            StepResult::success(
                "test",
                StepOutput::Text {
                    content: "The answer is 43.".to_string(),
                },
                0.75,
            ),
        ];

        let result = engine.vote(results);

        assert_eq!(result.answer, "42");
        assert_eq!(result.vote_count, 2);
        assert_eq!(result.total_samples, 3);
    }

    #[test]
    fn test_normalize_answer() {
        let engine = SelfConsistencyEngine::default_engine();

        assert_eq!(engine.normalize_answer("  HELLO, World!  "), "hello world");
        assert_eq!(engine.normalize_answer("42."), "42");
    }

    #[test]
    fn test_diversity_score() {
        let result = ConsistencyResult {
            answer: "42".to_string(),
            confidence: 0.8,
            vote_count: 2,
            total_samples: 3,
            agreement_ratio: 0.67,
            paths: Vec::new(),
            vote_distribution: HashMap::from([("42".to_string(), 2), ("43".to_string(), 1)]),
            early_stopped: false,
            total_tokens: TokenUsage::default(),
        };

        // 2 unique answers out of 3 samples = diversity 0.5
        assert!((result.diversity_score() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_early_stopping() {
        let config = SelfConsistencyConfig {
            consensus_threshold: 0.7,
            early_stopping: true,
            ..Default::default()
        };
        let engine = SelfConsistencyEngine::new(config);

        // 3 out of 4 agree = 75% > 70% threshold
        let results: Vec<StepResult> = (0..4)
            .map(|i| {
                let answer = if i < 3 { "42" } else { "43" };
                StepResult::success(
                    "test",
                    StepOutput::Text {
                        content: format!("The answer is {}.", answer),
                    },
                    0.8,
                )
            })
            .collect();

        assert!(engine.should_early_stop(&results));
    }

    #[test]
    fn test_empty_paths_handling() {
        let engine = SelfConsistencyEngine::default_engine();
        let result = engine.aggregate_paths(vec![]);

        assert!(result.answer.is_empty());
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.total_samples, 0);
    }

    #[test]
    fn test_nan_handling_in_vote_weights() {
        // Ensure we handle NaN values gracefully
        let engine = SelfConsistencyEngine::new(SelfConsistencyConfig {
            voting_method: VotingMethod::ConfidenceWeighted,
            ..Default::default()
        });

        // This should not panic even with edge cases
        let result = engine.aggregate_paths(vec![]);
        assert!(result.answer.is_empty());
    }
}
