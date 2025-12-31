//! DeepSeek Validation Engine Benchmark Framework
//!
//! Comprehensive benchmarking and validation testing for the DeepSeek Protocol
//! Validation Engine, measuring performance, accuracy, and enterprise compliance.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use super::{
    executor::{ProtocolInput, ProtocolOutput},
    validation::{DeepSeekValidationResult, ValidationVerdict},
    validation_executor::{ValidatingProtocolExecutor, ValidationExecutorConfig, ValidationLevel},
};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Test scenarios to run
    pub scenarios: Vec<BenchmarkScenario>,
    /// Number of iterations per scenario
    pub iterations: usize,
    /// Validation levels to test
    pub validation_levels: Vec<ValidationLevel>,
    /// Enable statistical analysis
    pub enable_statistics: bool,
    /// Benchmark timeout in seconds
    pub timeout_secs: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            scenarios: vec![
                BenchmarkScenario::BusinessDecision,
                BenchmarkScenario::TechnicalArchitecture,
                BenchmarkScenario::ComplianceAnalysis,
                BenchmarkScenario::RiskAssessment,
            ],
            iterations: 10,
            validation_levels: vec![
                ValidationLevel::None,
                ValidationLevel::Quick,
                ValidationLevel::Standard,
                ValidationLevel::Rigorous,
            ],
            enable_statistics: true,
            timeout_secs: 300,
        }
    }
}

/// Benchmark scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkScenario {
    /// Business decision making
    BusinessDecision,
    /// Technical architecture evaluation
    TechnicalArchitecture,
    /// Compliance and regulatory analysis
    ComplianceAnalysis,
    /// Risk assessment and mitigation
    RiskAssessment,
    /// Creative and strategic thinking
    StrategicPlanning,
    /// Multi-perspective analysis
    MultiPerspectiveAnalysis,
}

/// Benchmark result for a single scenario/level combination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub scenario: BenchmarkScenario,
    pub validation_level: ValidationLevel,
    pub iterations: usize,
    pub average_duration_ms: f64,
    pub average_confidence: f64,
    pub average_validation_confidence: f64,
    pub success_rate: f64,
    pub validation_success_rate: f64,
    pub token_usage: TokenUsageMetrics,
    pub validation_findings: Vec<ValidationFindingStat>,
}

/// Token usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsageMetrics {
    pub average_input_tokens: f64,
    pub average_output_tokens: f64,
    pub average_total_tokens: f64,
    pub average_cost_usd: f64,
    pub token_per_second: f64,
}

/// Validation finding statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFindingStat {
    pub finding_category: String,
    pub average_severity: f64,
    pub frequency: f64,
    pub average_confidence_impact: f64,
}

/// Complete benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub config: BenchmarkConfig,
    pub scenario_results: HashMap<String, ScenarioResult>,
    pub summary: BenchmarkSummary,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
}

/// Benchmark summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_duration_seconds: f64,
    pub total_iterations: usize,
    pub overall_success_rate: f64,
    pub average_confidence_gain: f64,
    pub cost_per_validation: f64,
    pub performance_improvements: Vec<String>,
    pub recommendations: Vec<String>,
}

/// DeepSeek Validation Benchmark Runner
#[derive(Default)]
pub struct DeepSeekBenchmarkRunner {
    config: BenchmarkConfig,
}

impl DeepSeekBenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run complete benchmark suite
    pub async fn run_benchmark(&self) -> Result<BenchmarkResults> {
        let start_time = Instant::now();
        let mut scenario_results = HashMap::new();
        let mut total_iterations = 0;

        for scenario in &self.config.scenarios {
            for validation_level in &self.config.validation_levels {
                let result = self.run_scenario(*scenario, *validation_level).await?;
                total_iterations += result.iterations;
                let key = format!("{:?}_{:?}", scenario, validation_level);
                scenario_results.insert(key, result);
            }
        }

        let total_duration = start_time.elapsed().as_secs_f64();

        let summary = self.calculate_summary(&scenario_results, total_duration, total_iterations);

        Ok(BenchmarkResults {
            config: self.config.clone(),
            scenario_results,
            summary,
            timestamp: chrono::Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    /// Run a single scenario with specific validation level
    async fn run_scenario(
        &self,
        scenario: BenchmarkScenario,
        validation_level: ValidationLevel,
    ) -> Result<ScenarioResult> {
        let mut durations = Vec::new();
        let mut confidences = Vec::new();
        let mut validation_confidences = Vec::new();
        let mut success_count = 0;
        let mut validation_success_count = 0;
        let mut token_metrics = Vec::new();
        let mut findings_stats = HashMap::new();

        for i in 0..self.config.iterations {
            let input = self.generate_scenario_input(scenario, i);

            let start_time = Instant::now();
            let result = self
                .execute_validation_run(&input, validation_level)
                .await?;
            let duration = start_time.elapsed().as_millis() as f64;

            durations.push(duration);
            confidences.push(result.confidence);

            if result.success {
                success_count += 1;
            }

            // Extract validation results if available
            if let Some(validation_data) = result.data.get("deepseek_validation") {
                if let Ok(validation_result) =
                    serde_json::from_value::<DeepSeekValidationResult>(validation_data.clone())
                {
                    validation_confidences.push(validation_result.validation_confidence);

                    if validation_result.verdict == ValidationVerdict::Validated {
                        validation_success_count += 1;
                    }

                    // Track token usage
                    token_metrics.push(TokenUsageMetrics {
                        average_input_tokens: validation_result.tokens_used.input_tokens as f64,
                        average_output_tokens: validation_result.tokens_used.output_tokens as f64,
                        average_total_tokens: validation_result.tokens_used.total_tokens as f64,
                        average_cost_usd: validation_result.tokens_used.cost_usd,
                        token_per_second: validation_result.performance.tokens_per_second,
                    });

                    // Track validation findings
                    self.analyze_findings(&validation_result, &mut findings_stats);
                }
            }
        }

        // Calculate averages
        let average_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let average_confidence = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let average_validation_confidence = if validation_confidences.is_empty() {
            0.0
        } else {
            validation_confidences.iter().sum::<f64>() / validation_confidences.len() as f64
        };

        let success_rate = success_count as f64 / self.config.iterations as f64;
        let validation_success_rate =
            validation_success_count as f64 / self.config.iterations as f64;

        // Calculate token usage averages
        let token_usage = if token_metrics.is_empty() {
            TokenUsageMetrics::default()
        } else {
            TokenUsageMetrics {
                average_input_tokens: token_metrics
                    .iter()
                    .map(|m| m.average_input_tokens)
                    .sum::<f64>()
                    / token_metrics.len() as f64,
                average_output_tokens: token_metrics
                    .iter()
                    .map(|m| m.average_output_tokens)
                    .sum::<f64>()
                    / token_metrics.len() as f64,
                average_total_tokens: token_metrics
                    .iter()
                    .map(|m| m.average_total_tokens)
                    .sum::<f64>()
                    / token_metrics.len() as f64,
                average_cost_usd: token_metrics
                    .iter()
                    .map(|m| m.average_cost_usd)
                    .sum::<f64>()
                    / token_metrics.len() as f64,
                token_per_second: token_metrics
                    .iter()
                    .map(|m| m.token_per_second)
                    .sum::<f64>()
                    / token_metrics.len() as f64,
            }
        };

        // Convert findings stats to vector
        let validation_findings = findings_stats
            .into_iter()
            .map(|(category, stats)| ValidationFindingStat {
                finding_category: category,
                average_severity: stats.average_severity / stats.count as f64,
                frequency: stats.count as f64 / self.config.iterations as f64,
                average_confidence_impact: stats.total_confidence_impact / stats.count as f64,
            })
            .collect();

        Ok(ScenarioResult {
            scenario,
            validation_level,
            iterations: self.config.iterations,
            average_duration_ms: average_duration,
            average_confidence,
            average_validation_confidence,
            success_rate,
            validation_success_rate,
            token_usage,
            validation_findings,
        })
    }

    /// Execute a single validation run
    async fn execute_validation_run(
        &self,
        input: &ProtocolInput,
        validation_level: ValidationLevel,
    ) -> Result<ProtocolOutput> {
        let config = ValidationExecutorConfig {
            validation_level,
            ..Default::default()
        };

        let executor = ValidatingProtocolExecutor::with_configs(Default::default(), config)?;

        // Use balanced profile for consistent benchmarking
        executor
            .execute_profile_with_validation("balanced", input.clone())
            .await
    }

    /// Generate scenario-specific input
    fn generate_scenario_input(
        &self,
        scenario: BenchmarkScenario,
        iteration: usize,
    ) -> ProtocolInput {
        match scenario {
            BenchmarkScenario::BusinessDecision => {
                ProtocolInput::query(format!(
                    "Should we expand to the European market? Consider market size, competition, regulatory requirements, and potential ROI. Iteration: {}",
                    iteration
                ))
            }
            BenchmarkScenario::TechnicalArchitecture => {
                ProtocolInput::query(format!(
                    "Evaluate microservices vs monolithic architecture for a 10,000 user SaaS application. Consider scalability, maintainability, deployment complexity. Iteration: {}",
                    iteration
                ))
            }
            BenchmarkScenario::ComplianceAnalysis => {
                ProtocolInput::query(format!(
                    "Analyze GDPR compliance requirements for a customer analytics platform processing EU citizen data. Iteration: {}",
                    iteration
                ))
            }
            BenchmarkScenario::RiskAssessment => {
                ProtocolInput::query(format!(
                    "Assess cybersecurity risks for a cloud-based financial application handling sensitive customer data. Iteration: {}",
                    iteration
                ))
            }
            BenchmarkScenario::StrategicPlanning => {
                ProtocolInput::query(format!(
                    "Develop a 5-year strategic plan for a technology startup in the AI infrastructure space. Iteration: {}",
                    iteration
                ))
            }
            BenchmarkScenario::MultiPerspectiveAnalysis => {
                ProtocolInput::query(format!(
                    "Analyze the impact of remote work policies from technical, cultural, productivity, and security perspectives. Iteration: {}",
                    iteration
                ))
            }
        }
    }

    /// Analyze validation findings and update statistics
    fn analyze_findings(
        &self,
        validation_result: &DeepSeekValidationResult,
        findings_stats: &mut HashMap<String, FindingStatsAccumulator>,
    ) {
        for finding in &validation_result.findings {
            let category = format!("{:?}", finding.category);
            let stats = findings_stats.entry(category).or_default();

            stats.count += 1;
            stats.average_severity += match finding.severity {
                super::validation::Severity::Critical => 5.0,
                super::validation::Severity::High => 4.0,
                super::validation::Severity::Medium => 3.0,
                super::validation::Severity::Low => 2.0,
                super::validation::Severity::Info => 1.0,
            };

            // Estimate confidence impact based on validation result
            stats.total_confidence_impact += validation_result.validation_confidence;
        }
    }

    /// Calculate benchmark summary
    fn calculate_summary(
        &self,
        scenario_results: &HashMap<String, ScenarioResult>,
        total_duration: f64,
        total_iterations: usize,
    ) -> BenchmarkSummary {
        let mut total_success = 0;
        let mut total_confidence_gain = 0.0;
        let mut total_cost = 0.0;

        for result in scenario_results.values() {
            total_success += (result.success_rate * result.iterations as f64) as usize;
            total_confidence_gain +=
                result.average_validation_confidence - result.average_confidence;
            total_cost += result.token_usage.average_cost_usd * result.iterations as f64;
        }

        let overall_success_rate = total_success as f64 / total_iterations as f64;
        let average_confidence_gain = total_confidence_gain / scenario_results.len() as f64;
        let cost_per_validation = total_cost / total_iterations as f64;

        // Generate recommendations
        let mut performance_improvements = Vec::new();
        let mut recommendations = Vec::new();

        if average_confidence_gain > 0.0 {
            performance_improvements.push(format!(
                "Average confidence improvement: +{:.1}%",
                average_confidence_gain * 100.0
            ));
        }

        if cost_per_validation < 0.05 {
            performance_improvements.push(format!(
                "Cost-effective validation: ${:.3} per analysis",
                cost_per_validation
            ));
        }

        if overall_success_rate > 0.85 {
            recommendations.push("Ready for production deployment".to_string());
        } else if overall_success_rate > 0.70 {
            recommendations.push("Suitable for development and testing".to_string());
        } else {
            recommendations.push("Further optimization recommended".to_string());
        }

        BenchmarkSummary {
            total_duration_seconds: total_duration,
            total_iterations,
            overall_success_rate,
            average_confidence_gain,
            cost_per_validation,
            performance_improvements,
            recommendations,
        }
    }
}

/// Helper struct for accumulating finding statistics
#[derive(Debug, Clone, Default)]
struct FindingStatsAccumulator {
    count: usize,
    average_severity: f64,
    total_confidence_impact: f64,
}

impl Default for TokenUsageMetrics {
    fn default() -> Self {
        Self {
            average_input_tokens: 0.0,
            average_output_tokens: 0.0,
            average_total_tokens: 0.0,
            average_cost_usd: 0.0,
            token_per_second: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.iterations, 10);
        assert!(config.enable_statistics);
    }

    #[test]
    fn test_scenario_input_generation() {
        let runner = DeepSeekBenchmarkRunner::default();
        let input = runner.generate_scenario_input(BenchmarkScenario::BusinessDecision, 1);

        assert!(input.get_str("query").unwrap().contains("European market"));
    }

    // Note: Full benchmark tests would require async execution and real API calls
    // These are tested in integration tests with mock responses
}
