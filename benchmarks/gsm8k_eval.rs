//! GSM8K Benchmark Evaluation for ReasonKit ThinkTools
//!
//! This benchmark measures ACTUAL improvement from ThinkTools on the GSM8K
//! math reasoning dataset. No claims without evidence.
//!
//! Usage:
//!   cargo run --release --bin gsm8k_eval -- --samples 100
//!
//! What we measure:
//!   - Raw prompt accuracy (baseline)
//!   - ThinkTool-enhanced accuracy (with GigaThink, LaserLogic, etc.)
//!   - Delta between them (the REAL improvement, if any)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A GSM8K problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GsmProblem {
    pub question: String,
    pub answer: String, // The final numeric answer
    pub solution: String, // Step-by-step solution
}

/// Evaluation result for a single problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub problem_id: usize,
    pub question: String,
    pub expected_answer: String,

    // Raw prompt results
    pub raw_response: String,
    pub raw_extracted_answer: Option<String>,
    pub raw_correct: bool,
    pub raw_latency_ms: u64,
    pub raw_tokens: u32,

    // ThinkTool-enhanced results
    pub enhanced_response: String,
    pub enhanced_extracted_answer: Option<String>,
    pub enhanced_correct: bool,
    pub enhanced_latency_ms: u64,
    pub enhanced_tokens: u32,
    pub enhanced_profile: String,
}

/// Aggregate benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub dataset: String,
    pub total_problems: usize,
    pub timestamp: String,

    // Raw accuracy
    pub raw_correct: usize,
    pub raw_accuracy: f64,
    pub raw_avg_latency_ms: f64,
    pub raw_total_tokens: u32,

    // Enhanced accuracy
    pub enhanced_correct: usize,
    pub enhanced_accuracy: f64,
    pub enhanced_avg_latency_ms: f64,
    pub enhanced_total_tokens: u32,
    pub enhanced_profile: String,

    // THE DELTA - This is what matters
    pub accuracy_delta: f64,        // enhanced - raw (positive = improvement)
    pub accuracy_delta_percent: f64, // percentage improvement

    // Cost analysis
    pub raw_cost_usd: f64,
    pub enhanced_cost_usd: f64,
    pub cost_multiplier: f64,

    // Individual results
    pub results: Vec<EvalResult>,
}

impl BenchmarkResults {
    pub fn calculate_stats(results: Vec<EvalResult>, profile: &str) -> Self {
        let total = results.len();
        let raw_correct = results.iter().filter(|r| r.raw_correct).count();
        let enhanced_correct = results.iter().filter(|r| r.enhanced_correct).count();

        let raw_accuracy = raw_correct as f64 / total as f64;
        let enhanced_accuracy = enhanced_correct as f64 / total as f64;

        let raw_total_latency: u64 = results.iter().map(|r| r.raw_latency_ms).sum();
        let enhanced_total_latency: u64 = results.iter().map(|r| r.enhanced_latency_ms).sum();

        let raw_total_tokens: u32 = results.iter().map(|r| r.raw_tokens).sum();
        let enhanced_total_tokens: u32 = results.iter().map(|r| r.enhanced_tokens).sum();

        // Approximate cost (Claude pricing: $3/1M input, $15/1M output)
        let raw_cost = raw_total_tokens as f64 * 0.000009; // ~$9 per 1M tokens average
        let enhanced_cost = enhanced_total_tokens as f64 * 0.000009;

        Self {
            dataset: "GSM8K".to_string(),
            total_problems: total,
            timestamp: chrono::Utc::now().to_rfc3339(),
            raw_correct,
            raw_accuracy,
            raw_avg_latency_ms: raw_total_latency as f64 / total as f64,
            raw_total_tokens,
            enhanced_correct,
            enhanced_accuracy,
            enhanced_avg_latency_ms: enhanced_total_latency as f64 / total as f64,
            enhanced_total_tokens,
            enhanced_profile: profile.to_string(),
            accuracy_delta: enhanced_accuracy - raw_accuracy,
            accuracy_delta_percent: ((enhanced_accuracy - raw_accuracy) / raw_accuracy) * 100.0,
            raw_cost_usd: raw_cost,
            enhanced_cost_usd: enhanced_cost,
            cost_multiplier: enhanced_cost / raw_cost,
            results,
        }
    }

    pub fn to_report(&self) -> String {
        let mut report = String::new();

        report.push_str("═══════════════════════════════════════════════════════════════════════\n");
        report.push_str("                    ReasonKit Benchmark Results\n");
        report.push_str("                         GSM8K Math Reasoning\n");
        report.push_str("═══════════════════════════════════════════════════════════════════════\n\n");

        report.push_str(&format!("Dataset:     {}\n", self.dataset));
        report.push_str(&format!("Problems:    {}\n", self.total_problems));
        report.push_str(&format!("Profile:     {}\n", self.enhanced_profile));
        report.push_str(&format!("Timestamp:   {}\n\n", self.timestamp));

        report.push_str("───────────────────────────────────────────────────────────────────────\n");
        report.push_str("                           ACCURACY COMPARISON\n");
        report.push_str("───────────────────────────────────────────────────────────────────────\n\n");

        report.push_str(&format!("  Raw Prompt:        {}/{} ({:.1}%)\n",
            self.raw_correct, self.total_problems, self.raw_accuracy * 100.0));
        report.push_str(&format!("  ThinkTool Enhanced: {}/{} ({:.1}%)\n",
            self.enhanced_correct, self.total_problems, self.enhanced_accuracy * 100.0));
        report.push_str("\n");

        // THE KEY METRIC
        let delta_sign = if self.accuracy_delta >= 0.0 { "+" } else { "" };
        let delta_color = if self.accuracy_delta > 0.0 { "🟢" }
                         else if self.accuracy_delta < 0.0 { "🔴" }
                         else { "⚪" };

        report.push_str(&format!("  {} DELTA: {}{:.1}% ({}{:.1}% relative)\n",
            delta_color, delta_sign, self.accuracy_delta * 100.0,
            delta_sign, self.accuracy_delta_percent));
        report.push_str("\n");

        report.push_str("───────────────────────────────────────────────────────────────────────\n");
        report.push_str("                           COST ANALYSIS\n");
        report.push_str("───────────────────────────────────────────────────────────────────────\n\n");

        report.push_str(&format!("  Raw Tokens:        {} (${:.4})\n",
            self.raw_total_tokens, self.raw_cost_usd));
        report.push_str(&format!("  Enhanced Tokens:   {} (${:.4})\n",
            self.enhanced_total_tokens, self.enhanced_cost_usd));
        report.push_str(&format!("  Cost Multiplier:   {:.1}x\n", self.cost_multiplier));
        report.push_str("\n");

        report.push_str("───────────────────────────────────────────────────────────────────────\n");
        report.push_str("                           LATENCY\n");
        report.push_str("───────────────────────────────────────────────────────────────────────\n\n");

        report.push_str(&format!("  Raw Avg:           {:.0}ms\n", self.raw_avg_latency_ms));
        report.push_str(&format!("  Enhanced Avg:      {:.0}ms\n", self.enhanced_avg_latency_ms));
        report.push_str("\n");

        report.push_str("═══════════════════════════════════════════════════════════════════════\n");
        report.push_str("                           VERDICT\n");
        report.push_str("═══════════════════════════════════════════════════════════════════════\n\n");

        if self.accuracy_delta > 0.05 {
            report.push_str("  ✅ VALIDATED: ThinkTools show meaningful improvement\n");
        } else if self.accuracy_delta > 0.0 {
            report.push_str("  ⚠️  MARGINAL: Small improvement, may not justify cost\n");
        } else if self.accuracy_delta == 0.0 {
            report.push_str("  ⚪ NEUTRAL: No measurable difference\n");
        } else {
            report.push_str("  ❌ NEGATIVE: ThinkTools performed WORSE than raw\n");
        }

        report.push_str("\n");
        report
    }
}

/// Extract numeric answer from LLM response
pub fn extract_answer(response: &str) -> Option<String> {
    // Look for patterns like "#### 42" or "The answer is 42" or "= 42"
    let patterns = [
        r"####\s*(\d+(?:\.\d+)?)",
        r"[Tt]he answer is\s*(\d+(?:\.\d+)?)",
        r"[Aa]nswer:\s*(\d+(?:\.\d+)?)",
        r"=\s*(\d+(?:\.\d+)?)\s*$",
        r"(\d+(?:\.\d+)?)\s*$",
    ];

    for pattern in patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if let Some(caps) = re.captures(response) {
                if let Some(m) = caps.get(1) {
                    return Some(m.as_str().to_string());
                }
            }
        }
    }

    None
}

/// Check if extracted answer matches expected
pub fn check_answer(extracted: &Option<String>, expected: &str) -> bool {
    match extracted {
        Some(ans) => {
            // Normalize both answers
            let ans_clean = ans.trim().replace(",", "");
            let exp_clean = expected.trim().replace(",", "");

            // Try numeric comparison
            if let (Ok(a), Ok(e)) = (ans_clean.parse::<f64>(), exp_clean.parse::<f64>()) {
                (a - e).abs() < 0.01
            } else {
                ans_clean == exp_clean
            }
        }
        None => false,
    }
}

/// Load GSM8K test samples
pub fn load_gsm8k_samples(path: &PathBuf, limit: usize) -> Result<Vec<GsmProblem>> {
    // For now, return some example problems
    // In production, load from actual GSM8K dataset
    let examples = vec![
        GsmProblem {
            question: "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?".to_string(),
            answer: "18".to_string(),
            solution: "16 - 3 - 4 = 9 eggs remaining. 9 * 2 = 18 dollars.".to_string(),
        },
        GsmProblem {
            question: "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?".to_string(),
            answer: "3".to_string(),
            solution: "2 + 2/2 = 2 + 1 = 3 bolts.".to_string(),
        },
        GsmProblem {
            question: "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?".to_string(),
            answer: "70000".to_string(),
            solution: "Cost = 80000 + 50000 = 130000. Value increase = 80000 * 1.5 = 120000. New value = 80000 + 120000 = 200000. Profit = 200000 - 130000 = 70000.".to_string(),
        },
        GsmProblem {
            question: "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?".to_string(),
            answer: "540".to_string(),
            solution: "3 sprints * 3 times * 60 meters = 540 meters.".to_string(),
        },
        GsmProblem {
            question: "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?".to_string(),
            answer: "20".to_string(),
            solution: "Total needed = 20 chickens * 3 cups = 60 cups. Already given = 15 + 25 = 40 cups. Remaining = 60 - 40 = 20 cups.".to_string(),
        },
    ];

    Ok(examples.into_iter().take(limit).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_answer_hash_format() {
        assert_eq!(extract_answer("The answer is #### 42"), Some("42".to_string()));
    }

    #[test]
    fn test_extract_answer_text_format() {
        assert_eq!(extract_answer("Therefore, the answer is 123"), Some("123".to_string()));
    }

    #[test]
    fn test_check_answer_exact() {
        assert!(check_answer(&Some("42".to_string()), "42"));
    }

    #[test]
    fn test_check_answer_with_commas() {
        assert!(check_answer(&Some("1000".to_string()), "1,000"));
    }

    #[test]
    fn test_check_answer_float() {
        assert!(check_answer(&Some("3.14".to_string()), "3.14"));
    }
}
