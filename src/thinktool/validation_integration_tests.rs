//! Integration Tests for DeepSeek Protocol Validation Engine
//!
//! Comprehensive test suite validating the DeepSeek Validation Engine's
//! functionality, accuracy, and enterprise compliance features.

use crate::error::Result;
use serde::{Deserialize, Serialize};

use super::{
    executor::{ProtocolExecutor, ProtocolInput, ProtocolOutput},
    validation::{DeepSeekValidationConfig, DeepSeekValidationEngine, ValidationVerdict},
    validation_executor::{ValidatingProtocolExecutor, ValidationExecutorConfig, ValidationLevel},
};

/// Test configuration for validation engine integration tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationTestConfig {
    /// Enable mock mode (for testing without API calls)
    pub use_mock: bool,
    /// Test scenarios to run
    pub test_scenarios: Vec<TestScenario>,
    /// Expected success thresholds
    pub success_threshold: f64,
    /// Maximum test duration
    pub max_duration_ms: u64,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            use_mock: true, // Default to mock for safety
            test_scenarios: vec![
                TestScenario::BasicQuery,
                TestScenario::ComplexAnalysis,
                TestScenario::ComplianceCheck,
                TestScenario::RiskAssessment,
            ],
            success_threshold: 0.80, // 80% success rate required
            max_duration_ms: 30000,  // 30 second timeout
        }
    }
}

/// Test scenarios for validation engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestScenario {
    /// Basic query validation
    BasicQuery,
    /// Complex multi-step analysis
    ComplexAnalysis,
    /// Compliance and regulation validation
    ComplianceCheck,
    /// Risk assessment validation
    RiskAssessment,
    /// Business decision validation
    BusinessDecision,
    /// Technical architecture validation
    TechnicalArchitecture,
}

/// Test result for a single scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenarioResult {
    pub scenario: TestScenario,
    pub success: bool,
    pub duration_ms: u64,
    pub confidence: f64,
    pub validation_result: Option<String>,
    pub error_message: Option<String>,
    pub validation_findings_count: usize,
    pub compliance_status: Option<String>,
}

/// Complete integration test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationTestResults {
    pub config: IntegrationTestConfig,
    pub scenario_results: Vec<TestScenarioResult>,
    pub overall_success: bool,
    pub success_rate: f64,
    pub average_confidence: f64,
    pub total_duration_ms: u64,
    pub recommendations: Vec<String>,
}

/// DeepSeek Validation Engine Integration Test Runner
pub struct ValidationIntegrationTester {
    config: IntegrationTestConfig,
}

impl ValidationIntegrationTester {
    pub fn new(config: IntegrationTestConfig) -> Self {
        Self { config }
    }

    pub fn default() -> Self {
        Self::new(IntegrationTestConfig::default())
    }

    /// Run complete integration test suite
    pub async fn run_integration_tests(&self) -> Result<IntegrationTestResults> {
        let start_time = std::time::Instant::now();
        let mut scenario_results = Vec::new();
        let mut success_count = 0;
        let mut total_confidence = 0.0;

        for scenario in &self.config.test_scenarios {
            let result = self.run_test_scenario(*scenario).await?;

            if result.success {
                success_count += 1;
            }
            total_confidence += result.confidence;

            scenario_results.push(result);
        }

        let total_duration = start_time.elapsed().as_millis() as u64;
        let success_rate = success_count as f64 / self.config.test_scenarios.len() as f64;
        let average_confidence = total_confidence / self.config.test_scenarios.len() as f64;

        let overall_success = success_rate >= self.config.success_threshold;

        let recommendations = self.generate_recommendations(&scenario_results, success_rate);

        Ok(IntegrationTestResults {
            config: self.config.clone(),
            scenario_results,
            overall_success,
            success_rate,
            average_confidence,
            total_duration_ms: total_duration,
            recommendations,
        })
    }

    /// Run a single test scenario
    async fn run_test_scenario(&self, scenario: TestScenario) -> Result<TestScenarioResult> {
        let start_time = std::time::Instant::now();

        let input = self.generate_test_input(scenario);
        let result = self.execute_validation_test(&input).await;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(output) => {
                self.analyze_successful_result(scenario, output, duration_ms)
            }
            Err(e) => {
                Ok(TestScenarioResult {
                    scenario,
                    success: false,
                    duration_ms,
                    confidence: 0.0,
                    validation_result: None,
                    error_message: Some(e.to_string()),
                    validation_findings_count: 0,
                    compliance_status: None,
                })
            }
        }
    }

    /// Execute validation test
    async fn execute_validation_test(&self, input: &ProtocolInput) -> Result<ProtocolOutput> {
        if self.config.use_mock {
            // Use mock executor for testing
            let executor = ProtocolExecutor::mock()?;
            executor.execute_profile("balanced", input.clone()).await
        } else {
            // Use real validation engine
            let config = ValidationExecutorConfig {
                validation_level: ValidationLevel::Standard,
                ..Default::default()
            };

            let executor = ValidatingProtocolExecutor::with_configs(Default::default(), config)?;
            executor.execute_profile_with_validation("balanced", input.clone()).await
        }
    }

    /// Generate test input for scenario
    fn generate_test_input(&self, scenario: TestScenario) -> ProtocolInput {
        match scenario {
            TestScenario::BasicQuery => {
                ProtocolInput::query("What are the key factors for successful product launches?")
            }
            TestScenario::ComplexAnalysis => {
                ProtocolInput::query("Analyze the long-term implications of quantum computing on cybersecurity, considering both threats and opportunities across different industries.")
            }
            TestScenario::ComplianceCheck => {
                ProtocolInput::query("Evaluate GDPR compliance requirements for a mobile app collecting user location data for personalized recommendations in the European market.")
            }
            TestScenario::RiskAssessment => {
                ProtocolInput::query("Assess the cybersecurity risks for a cloud-based financial platform handling sensitive customer transaction data, including data breaches, regulatory risks, and reputational impact.")
            }
            TestScenario::BusinessDecision => {
                ProtocolInput::query("Should a B2B SaaS company prioritize international expansion or domestic market consolidation? Consider market saturation, regulatory complexity, growth potential, and resource allocation.")
            }
            TestScenario::TechnicalArchitecture => {
                ProtocolInput::query("Compare serverless architecture vs container-based microservices for a real-time analytics platform processing 1M events per second. Evaluate scalability, cost, complexity, and operational overhead.")
            }
        }
    }

    /// Analyze successful test result
    fn analyze_successful_result(
        &self,
        scenario: TestScenario,
        output: ProtocolOutput,
        duration_ms: u64,
    ) -> Result<TestScenarioResult> {
        let success = output.success && output.confidence >= 0.70;

        // Extract validation results
        let validation_result = output.data.get("deepseek_validation")
            .map(|v| v.to_string());

        let validation_findings_count = if let Some(validation_data) = output.data.get("deepseek_validation") {
            if let Ok(validation_result) = serde_json::from_value::<super::validation::DeepSeekValidationResult>(
                validation_data.clone(),
            ) {
                validation_result.findings.len()
            } else {
                0
            }
        } else {
            0
        };

        // Extract compliance status
        let compliance_status = if let Some(validation_data) = output.data.get("deepseek_validation") {
            if let Ok(validation_result) = serde_json::from_value::<super::validation::DeepSeekValidationResult>(
                validation_data.clone(),
            ) {
                match validation_result.verdict {
                    ValidationVerdict::Validated => Some("Compliant".to_string()),
                    ValidationVerdict::PartiallyValidated => Some("Partially Compliant".to_string()),
                    ValidationVerdict::NeedsImprovement => Some("Needs Improvement".to_string()),
                    ValidationVerdict::Invalid => Some("Non-Compliant".to_string()),
                    ValidationVerdict::CriticalIssues => Some("Critical Compliance Issues".to_string()),
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(TestScenarioResult {
            scenario,
            success,
            duration_ms,
            confidence: output.confidence,
            validation_result,
            error_message: None,
            validation_findings_count,
            compliance_status,
        })
    }

    /// Generate recommendations based on test results
    fn generate_recommendations(
        &self,
        results: &[TestScenarioResult],
        success_rate: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if success_rate >= 0.90 {
            recommendations.push("Validation engine ready for production deployment".to_string());
        } else if success_rate >= 0.80 {
            recommendations.push("Validation engine ready for enterprise testing".to_string());
        } else {
            recommendations.push("Validation engine needs additional optimization".to_string());
        }

        // Check for specific performance characteristics
        let avg_duration: f64 = results.iter()
            .map(|r| r.duration_ms as f64)
            .sum::<f64>() / results.len() as f64;

        if avg_duration < 5000.0 {
            recommendations.push("Excellent performance: sub-5 second response times".to_string());
        } else if avg_duration < 10000.0 {
            recommendations.push("Good performance: 5-10 second response times".to_string());
        } else {
            recommendations.push("Performance optimization recommended".to_string());
        }

        // Check validation findings
        let avg_findings: f64 = results.iter()
            .map(|r| r.validation_findings_count as f64)
            .sum::<f64>() / results.len() as f64;

        if avg_findings > 2.0 {
            recommendations.push("Strong validation coverage: multiple findings per analysis".to_string());
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_test_config() {
        let config = IntegrationTestConfig::default();
        assert!(config.use_mock);
        assert_eq!(config.success_threshold, 0.80);
    }

    #[test]
    fn test_test_scenario_inputs() {
        let tester = ValidationIntegrationTester::default();

        let query_input = tester.generate_test_input(TestScenario::BasicQuery);
        assert!(query_input.get_str("query").unwrap().contains("product launches"));

        let compliance_input = tester.generate_test_input(TestScenario::ComplianceCheck);
        assert!(compliance_input.get_str("query").unwrap().contains("GDPR"));
    }

    #[tokio::test]
    async fn test_mock_integration_test() {
        let config = IntegrationTestConfig {
            use_mock: true,
            test_scenarios: vec![TestScenario::BasicQuery],
            ..Default::default()
        };

        let tester = ValidationIntegrationTester::new(config);
        let results = tester.run_integration_tests().await.unwrap();

        assert!(!results.scenario_results.is_empty());
        assert!(results.success_rate > 0.0);
    }
}

/// Additional integration test for DeepSeek Validation Engine specific features
#[cfg(test)]
mod deepseek_validation_tests {
    use super::*;

    #[tokio::test]
    async fn test_validation_engine_creation() {
        let engine = DeepSeekValidationEngine::new().unwrap();
        // Engine should be created successfully
        assert!(true); // Placeholder - actual validation would test engine properties
    }

    #[test]
    fn test_validation_config_options() {
        let rigorous_config = DeepSeekValidationConfig::rigorous();
        assert!(rigorous_config.enable_statistical_testing);
        assert!(rigorous_config.enable_compliance_validation);

        let perf_config = DeepSeekValidationConfig::performance();
        assert!(!perf_config.enable_statistical_testing);
        assert!(perf_config.enable_compliance_validation);
    }

    #[test]
    fn test_validation_executor_configs() {
        let enterprise_config = ValidationExecutorConfig::enterprise();
        assert_eq!(enterprise_config.validation_level, ValidationLevel::Rigorous);
        assert!(enterprise_config.validate_protocols.contains(&"proofguard".to_string()));

        let research_config = ValidationExecutorConfig::research();
        assert_eq!(research_config.validation_level, ValidationLevel::Paranoid);
        assert!(research_config.validate_protocols.contains(&"gigathink".to_string()));
    }
}
