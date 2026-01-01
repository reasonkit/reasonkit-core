//! # MiniMax M2 Integration Types
//!
//! This file contains a minimal set of public types needed for the (currently
//! stubbed) `reasonkit::m2` API surface.

use serde::{Deserialize, Serialize};

/// Generic input payload for an M2 run.
pub type ProtocolInput = serde_json::Value;

/// M2 API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2Config {
    pub endpoint: String,
    pub api_key: String,
    pub max_context_length: usize,
    pub max_output_length: usize,
    pub rate_limit: RateLimitConfig,
    pub performance: PerformanceConfig,
}

impl Default for M2Config {
    fn default() -> Self {
        Self {
            endpoint: "https://api.minimax.chat/v1/m2".to_string(),
            api_key: "".to_string(),
            max_context_length: 200_000,
            max_output_length: 128_000,
            rate_limit: RateLimitConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

/// Rate limiting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub rpm: u32,
    pub rps: u32,
    pub burst: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            rpm: 60,
            rps: 1,
            burst: 5,
        }
    }
}

/// Performance tuning options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub cost_reduction_target: f64,
    pub latency_target_ms: u64,
    pub quality_threshold: f64,
    pub enable_caching: bool,
    pub compression_level: u8,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            cost_reduction_target: 92.0,
            latency_target_ms: 2000,
            quality_threshold: 0.90,
            enable_caching: true,
            compression_level: 5,
        }
    }
}

/// High-level integration service configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2IntegrationConfig {
    pub max_concurrent_executions: u32,
    pub default_timeout_ms: u64,
    pub enable_caching: bool,
    pub enable_monitoring: bool,
    pub default_optimization_goals: OptimizationGoals,
}

impl Default for M2IntegrationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_executions: 10,
            default_timeout_ms: 300_000,
            enable_caching: true,
            enable_monitoring: true,
            default_optimization_goals: OptimizationGoals::default(),
        }
    }
}

/// Agent framework types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentFramework {
    ClaudeCode,
    Cline,
    KiloCode,
    Droid,
    RooCode,
    BlackBoxAi,
}

/// Supported use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UseCase {
    CodeAnalysis,
    BugFinding,
    Documentation,
    Architecture,
    General,
}

/// Coarse task type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    CodeAnalysis,
    BugFinding,
    Documentation,
    Architecture,
    General,
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Task domain (small starter set).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskDomain {
    SystemProgramming,
    Web,
    Data,
    General,
}

/// Complexity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
}

/// Quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityLevel {
    Draft,
    Standard,
    High,
}

/// Output size hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputSize {
    Small,
    #[default]
    Medium,
    Large,
}

/// A light time constraint model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeConstraints {
    pub max_duration_ms: Option<u64>,
}

impl Default for TimeConstraints {
    fn default() -> Self {
        Self {
            max_duration_ms: Some(300_000),
        }
    }
}

/// Quality requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityRequirements {
    pub quality_level: QualityLevel,
}

impl Default for QualityRequirements {
    fn default() -> Self {
        Self {
            quality_level: QualityLevel::Standard,
        }
    }
}

/// Optimization goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationGoal {
    Cost,
    Latency,
    Quality,
    BalanceAll,
}

/// Optimization constraints.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationConstraints {
    pub max_cost: Option<f64>,
    pub max_latency_ms: Option<u64>,
    pub min_quality: Option<f64>,
}

/// Performance targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub cost_reduction_target: f64,
    pub latency_reduction_target: f64,
    pub quality_threshold: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            cost_reduction_target: 92.0,
            latency_reduction_target: 0.20,
            quality_threshold: 0.90,
        }
    }
}

/// Optimization goals bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationGoals {
    pub primary_goal: OptimizationGoal,
    pub secondary_goals: Vec<OptimizationGoal>,
    pub constraints: OptimizationConstraints,
    pub performance_targets: PerformanceTargets,
}

impl Default for OptimizationGoals {
    fn default() -> Self {
        Self {
            primary_goal: OptimizationGoal::BalanceAll,
            secondary_goals: vec![],
            constraints: OptimizationConstraints::default(),
            performance_targets: PerformanceTargets::default(),
        }
    }
}

/// Task classification for protocol selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskClassification {
    pub task_type: TaskType,
    pub complexity_level: ComplexityLevel,
    pub domain: TaskDomain,
    pub expected_output_size: OutputSize,
    pub time_constraints: TimeConstraints,
    pub quality_requirements: QualityRequirements,
}

impl From<UseCase> for TaskClassification {
    fn from(use_case: UseCase) -> Self {
        match use_case {
            UseCase::CodeAnalysis => Self {
                task_type: TaskType::CodeAnalysis,
                complexity_level: ComplexityLevel::Complex,
                domain: TaskDomain::SystemProgramming,
                expected_output_size: OutputSize::Large,
                time_constraints: TimeConstraints::default(),
                quality_requirements: QualityRequirements::default(),
            },
            UseCase::BugFinding => Self {
                task_type: TaskType::BugFinding,
                complexity_level: ComplexityLevel::Moderate,
                domain: TaskDomain::SystemProgramming,
                expected_output_size: OutputSize::Medium,
                time_constraints: TimeConstraints::default(),
                quality_requirements: QualityRequirements::default(),
            },
            UseCase::Documentation => Self {
                task_type: TaskType::Documentation,
                complexity_level: ComplexityLevel::Moderate,
                domain: TaskDomain::General,
                expected_output_size: OutputSize::Medium,
                time_constraints: TimeConstraints::default(),
                quality_requirements: QualityRequirements::default(),
            },
            UseCase::Architecture => Self {
                task_type: TaskType::Architecture,
                complexity_level: ComplexityLevel::Complex,
                domain: TaskDomain::General,
                expected_output_size: OutputSize::Large,
                time_constraints: TimeConstraints::default(),
                quality_requirements: QualityRequirements::default(),
            },
            UseCase::General => Self {
                task_type: TaskType::General,
                complexity_level: ComplexityLevel::Moderate,
                domain: TaskDomain::General,
                expected_output_size: OutputSize::Medium,
                time_constraints: TimeConstraints::default(),
                quality_requirements: QualityRequirements::default(),
            },
        }
    }
}

/// Result placeholder produced by the stubbed service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavedResult {
    pub summary: String,
}

// --- Added for Engine Compatibility ---

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProtocolOutput {
    pub result: String,
    pub confidence: f64,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfidenceScores {
    pub overall: f64,
    pub reasoning: f64,
    pub evidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Evidence {
    pub content: String,
    pub source: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub passed: bool,
    pub details: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SynthesisMethod {
    Ensemble,
    WeightedAverage,
    BestOfN,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SynthesisStrategy {
    Standard,
    Aggressive,
    Conservative,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionMetrics {
    pub duration_ms: u64,
    pub token_usage: TokenUsage,
    pub cost_metrics: CostMetrics,
    pub quality_metrics: QualityMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub audit_trail: AuditTrail,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub total: u64,
    pub context: u64,
    pub output: u64,
    pub validation: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostMetrics {
    pub total_cost: f64,
    pub savings: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityMetrics {
    pub reliability: f64,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    pub latency_ms: u64,
    pub throughput: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditTrail {
    pub steps: Vec<String>,
    pub timestamp: u64,
    pub compliance_flags: Vec<ComplianceFlag>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComplianceFlag {
    GDPRCompliant,
    HIPAACompliant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavedProtocol {
    pub phases: Vec<InterleavedPhase>,
    pub constraints: CompositeConstraints,
    pub m2_optimizations: M2Optimizations,
    pub name: String,
    pub id: String,
    pub version: String,
    pub description: String,
    pub framework_compatibility: Vec<String>,
    pub language_support: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavedPhase {
    pub name: String,
    pub parallel_branches: u32,
    pub required_confidence: f64,
    pub validation_methods: Vec<ValidationMethod>,
    pub synthesis_methods: Vec<SynthesisMethod>,
    pub constraints: CompositeConstraints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeConstraints {
    pub time_budget_ms: u64,
    pub token_budget: u64,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub token_budget: TokenBudget,
    pub time_allocation_ms: u64,
    pub priority: u32,
    pub quality_targets: QualityTargets,
    pub parallel_capacity: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    pub total: u64,
    pub context: u64,
    pub output: u64,
    pub validation: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTargets {
    pub min_confidence: f64,
    pub required_depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2Optimizations {
    pub context_optimization: ContextOptimization,
    pub output_optimization: OutputOptimization,
    pub cost_optimization: CostOptimization,
    pub target_parameters: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextOptimization {
    pub method: String, // e.g., "none"
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputOptimization {
    pub format: String, // e.g., "text"
    pub template: String,
    pub max_output_length: u32,
    pub streaming_enabled: bool,
    pub compression_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimization {
    pub strategy: String, // e.g., "balanced"
    pub max_budget: f64,
    pub target_cost_reduction: f64,
    pub target_latency_reduction: f64,
    pub parallel_processing_enabled: bool,
    pub caching_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationMethod {
    SelfCheck,
    PeerReview,
    FormalVerification,
}

// Placeholder for PhaseResult used in engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    pub output: String,
    pub confidence: f64,
}

// Placeholder for ReasoningStepResult
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStepResult {
    pub content: String,
}

pub struct FallbackAction {/* ... */}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Default Value Tests
    // =========================================================================

    mod default_values {
        use super::*;

        #[test]
        fn test_m2_config_default() {
            let config = M2Config::default();

            assert_eq!(config.endpoint, "https://api.minimax.chat/v1/m2");
            assert_eq!(config.api_key, "");
            assert_eq!(config.max_context_length, 200_000);
            assert_eq!(config.max_output_length, 128_000);
        }

        #[test]
        fn test_rate_limit_config_default() {
            let config = RateLimitConfig::default();

            assert_eq!(config.rpm, 60);
            assert_eq!(config.rps, 1);
            assert_eq!(config.burst, 5);
        }

        #[test]
        fn test_performance_config_default() {
            let config = PerformanceConfig::default();

            assert!((config.cost_reduction_target - 92.0).abs() < f64::EPSILON);
            assert_eq!(config.latency_target_ms, 2000);
            assert!((config.quality_threshold - 0.90).abs() < f64::EPSILON);
            assert!(config.enable_caching);
            assert_eq!(config.compression_level, 5);
        }

        #[test]
        fn test_m2_integration_config_default() {
            let config = M2IntegrationConfig::default();

            assert_eq!(config.max_concurrent_executions, 10);
            assert_eq!(config.default_timeout_ms, 300_000);
            assert!(config.enable_caching);
            assert!(config.enable_monitoring);
        }

        #[test]
        fn test_time_constraints_default() {
            let constraints = TimeConstraints::default();

            assert_eq!(constraints.max_duration_ms, Some(300_000));
        }

        #[test]
        fn test_quality_requirements_default() {
            let requirements = QualityRequirements::default();

            assert_eq!(requirements.quality_level, QualityLevel::Standard);
        }

        #[test]
        fn test_optimization_constraints_default() {
            let constraints = OptimizationConstraints::default();

            assert!(constraints.max_cost.is_none());
            assert!(constraints.max_latency_ms.is_none());
            assert!(constraints.min_quality.is_none());
        }

        #[test]
        fn test_performance_targets_default() {
            let targets = PerformanceTargets::default();

            assert!((targets.cost_reduction_target - 92.0).abs() < f64::EPSILON);
            assert!((targets.latency_reduction_target - 0.20).abs() < f64::EPSILON);
            assert!((targets.quality_threshold - 0.90).abs() < f64::EPSILON);
        }

        #[test]
        fn test_optimization_goals_default() {
            let goals = OptimizationGoals::default();

            assert_eq!(goals.primary_goal, OptimizationGoal::BalanceAll);
            assert!(goals.secondary_goals.is_empty());
        }

        #[test]
        fn test_output_size_default() {
            let size = OutputSize::default();

            assert_eq!(size, OutputSize::Medium);
        }

        #[test]
        fn test_protocol_output_default() {
            let output = ProtocolOutput::default();

            assert_eq!(output.result, "");
            assert!((output.confidence - 0.0).abs() < f64::EPSILON);
            assert!(output.evidence.is_empty());
        }

        #[test]
        fn test_confidence_scores_default() {
            let scores = ConfidenceScores::default();

            assert!((scores.overall - 0.0).abs() < f64::EPSILON);
            assert!((scores.reasoning - 0.0).abs() < f64::EPSILON);
            assert!((scores.evidence - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_evidence_default() {
            let evidence = Evidence::default();

            assert_eq!(evidence.content, "");
            assert_eq!(evidence.source, "");
            assert!((evidence.confidence - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_execution_metrics_default() {
            let metrics = ExecutionMetrics::default();

            assert_eq!(metrics.duration_ms, 0);
            assert_eq!(metrics.token_usage.total, 0);
            assert!((metrics.cost_metrics.total_cost - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_token_usage_default() {
            let usage = TokenUsage::default();

            assert_eq!(usage.total, 0);
            assert_eq!(usage.context, 0);
            assert_eq!(usage.output, 0);
            assert_eq!(usage.validation, 0);
        }

        #[test]
        fn test_cost_metrics_default() {
            let metrics = CostMetrics::default();

            assert!((metrics.total_cost - 0.0).abs() < f64::EPSILON);
            assert!((metrics.savings - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_quality_metrics_default() {
            let metrics = QualityMetrics::default();

            assert!((metrics.reliability - 0.0).abs() < f64::EPSILON);
            assert!((metrics.accuracy - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_performance_metrics_default() {
            let metrics = PerformanceMetrics::default();

            assert_eq!(metrics.latency_ms, 0);
            assert!((metrics.throughput - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_audit_trail_default() {
            let trail = AuditTrail::default();

            assert!(trail.steps.is_empty());
            assert_eq!(trail.timestamp, 0);
            assert!(trail.compliance_flags.is_empty());
        }
    }

    // =========================================================================
    // Serialization Round-Trip Tests
    // =========================================================================

    mod serialization {
        use super::*;

        #[test]
        fn test_m2_config_roundtrip() {
            let config = M2Config::default();
            let json = serde_json::to_string(&config).unwrap();
            let deserialized: M2Config = serde_json::from_str(&json).unwrap();

            assert_eq!(config.endpoint, deserialized.endpoint);
            assert_eq!(config.api_key, deserialized.api_key);
            assert_eq!(config.max_context_length, deserialized.max_context_length);
            assert_eq!(config.max_output_length, deserialized.max_output_length);
        }

        #[test]
        fn test_rate_limit_config_roundtrip() {
            let config = RateLimitConfig {
                rpm: 120,
                rps: 2,
                burst: 10,
            };
            let json = serde_json::to_string(&config).unwrap();
            let deserialized: RateLimitConfig = serde_json::from_str(&json).unwrap();

            assert_eq!(config.rpm, deserialized.rpm);
            assert_eq!(config.rps, deserialized.rps);
            assert_eq!(config.burst, deserialized.burst);
        }

        #[test]
        fn test_performance_config_roundtrip() {
            let config = PerformanceConfig {
                cost_reduction_target: 85.5,
                latency_target_ms: 1500,
                quality_threshold: 0.95,
                enable_caching: false,
                compression_level: 9,
            };
            let json = serde_json::to_string(&config).unwrap();
            let deserialized: PerformanceConfig = serde_json::from_str(&json).unwrap();

            assert!(
                (config.cost_reduction_target - deserialized.cost_reduction_target).abs()
                    < f64::EPSILON
            );
            assert_eq!(config.latency_target_ms, deserialized.latency_target_ms);
            assert!(
                (config.quality_threshold - deserialized.quality_threshold).abs() < f64::EPSILON
            );
            assert_eq!(config.enable_caching, deserialized.enable_caching);
            assert_eq!(config.compression_level, deserialized.compression_level);
        }

        #[test]
        fn test_m2_integration_config_roundtrip() {
            let config = M2IntegrationConfig::default();
            let json = serde_json::to_string(&config).unwrap();
            let deserialized: M2IntegrationConfig = serde_json::from_str(&json).unwrap();

            assert_eq!(
                config.max_concurrent_executions,
                deserialized.max_concurrent_executions
            );
            assert_eq!(config.default_timeout_ms, deserialized.default_timeout_ms);
            assert_eq!(config.enable_caching, deserialized.enable_caching);
            assert_eq!(config.enable_monitoring, deserialized.enable_monitoring);
        }

        #[test]
        fn test_optimization_goals_roundtrip() {
            let goals = OptimizationGoals {
                primary_goal: OptimizationGoal::Cost,
                secondary_goals: vec![OptimizationGoal::Latency, OptimizationGoal::Quality],
                constraints: OptimizationConstraints {
                    max_cost: Some(100.0),
                    max_latency_ms: Some(5000),
                    min_quality: Some(0.8),
                },
                performance_targets: PerformanceTargets::default(),
            };
            let json = serde_json::to_string(&goals).unwrap();
            let deserialized: OptimizationGoals = serde_json::from_str(&json).unwrap();

            assert_eq!(goals.primary_goal, deserialized.primary_goal);
            assert_eq!(
                goals.secondary_goals.len(),
                deserialized.secondary_goals.len()
            );
            assert_eq!(
                goals.constraints.max_cost,
                deserialized.constraints.max_cost
            );
        }

        #[test]
        fn test_task_classification_roundtrip() {
            let classification = TaskClassification {
                task_type: TaskType::CodeAnalysis,
                complexity_level: ComplexityLevel::Complex,
                domain: TaskDomain::SystemProgramming,
                expected_output_size: OutputSize::Large,
                time_constraints: TimeConstraints {
                    max_duration_ms: Some(60000),
                },
                quality_requirements: QualityRequirements {
                    quality_level: QualityLevel::High,
                },
            };
            let json = serde_json::to_string(&classification).unwrap();
            let deserialized: TaskClassification = serde_json::from_str(&json).unwrap();

            assert_eq!(classification.task_type, deserialized.task_type);
            assert_eq!(
                classification.complexity_level,
                deserialized.complexity_level
            );
            assert_eq!(classification.domain, deserialized.domain);
            assert_eq!(
                classification.expected_output_size,
                deserialized.expected_output_size
            );
        }

        #[test]
        fn test_interleaved_result_roundtrip() {
            let result = InterleavedResult {
                summary: "Test summary content".to_string(),
            };
            let json = serde_json::to_string(&result).unwrap();
            let deserialized: InterleavedResult = serde_json::from_str(&json).unwrap();

            assert_eq!(result.summary, deserialized.summary);
        }

        #[test]
        fn test_protocol_output_roundtrip() {
            let output = ProtocolOutput {
                result: "Analysis complete".to_string(),
                confidence: 0.95,
                evidence: vec![Evidence {
                    content: "Source code reviewed".to_string(),
                    source: "file.rs".to_string(),
                    confidence: 0.9,
                }],
            };
            let json = serde_json::to_string(&output).unwrap();
            let deserialized: ProtocolOutput = serde_json::from_str(&json).unwrap();

            assert_eq!(output.result, deserialized.result);
            assert!((output.confidence - deserialized.confidence).abs() < f64::EPSILON);
            assert_eq!(output.evidence.len(), deserialized.evidence.len());
        }

        #[test]
        fn test_validation_result_roundtrip() {
            let result = ValidationResult {
                passed: true,
                details: "All checks passed".to_string(),
                score: 0.98,
            };
            let json = serde_json::to_string(&result).unwrap();
            let deserialized: ValidationResult = serde_json::from_str(&json).unwrap();

            assert_eq!(result.passed, deserialized.passed);
            assert_eq!(result.details, deserialized.details);
            assert!((result.score - deserialized.score).abs() < f64::EPSILON);
        }

        #[test]
        fn test_execution_metrics_roundtrip() {
            let metrics = ExecutionMetrics {
                duration_ms: 1500,
                token_usage: TokenUsage {
                    total: 5000,
                    context: 3000,
                    output: 1500,
                    validation: 500,
                },
                cost_metrics: CostMetrics {
                    total_cost: 0.05,
                    savings: 0.02,
                },
                quality_metrics: QualityMetrics {
                    reliability: 0.99,
                    accuracy: 0.95,
                },
                performance_metrics: PerformanceMetrics {
                    latency_ms: 1500,
                    throughput: 100.5,
                },
                audit_trail: AuditTrail {
                    steps: vec!["step1".to_string(), "step2".to_string()],
                    timestamp: 1234567890,
                    compliance_flags: vec![ComplianceFlag::GDPRCompliant],
                },
            };
            let json = serde_json::to_string(&metrics).unwrap();
            let deserialized: ExecutionMetrics = serde_json::from_str(&json).unwrap();

            assert_eq!(metrics.duration_ms, deserialized.duration_ms);
            assert_eq!(metrics.token_usage.total, deserialized.token_usage.total);
            assert_eq!(
                metrics.audit_trail.steps.len(),
                deserialized.audit_trail.steps.len()
            );
        }

        #[test]
        fn test_interleaved_phase_roundtrip() {
            let phase = InterleavedPhase {
                name: "reasoning".to_string(),
                parallel_branches: 3,
                required_confidence: 0.85,
                validation_methods: vec![ValidationMethod::SelfCheck, ValidationMethod::PeerReview],
                synthesis_methods: vec![SynthesisMethod::Ensemble],
                constraints: CompositeConstraints {
                    time_budget_ms: 60000,
                    token_budget: 10000,
                    dependencies: vec!["phase1".to_string()],
                },
            };
            let json = serde_json::to_string(&phase).unwrap();
            let deserialized: InterleavedPhase = serde_json::from_str(&json).unwrap();

            assert_eq!(phase.name, deserialized.name);
            assert_eq!(phase.parallel_branches, deserialized.parallel_branches);
            assert_eq!(
                phase.validation_methods.len(),
                deserialized.validation_methods.len()
            );
        }

        #[test]
        fn test_resource_allocation_roundtrip() {
            let allocation = ResourceAllocation {
                token_budget: TokenBudget {
                    total: 50000,
                    context: 30000,
                    output: 15000,
                    validation: 5000,
                },
                time_allocation_ms: 120000,
                priority: 1,
                quality_targets: QualityTargets {
                    min_confidence: 0.9,
                    required_depth: 3,
                },
                parallel_capacity: 4,
            };
            let json = serde_json::to_string(&allocation).unwrap();
            let deserialized: ResourceAllocation = serde_json::from_str(&json).unwrap();

            assert_eq!(
                allocation.token_budget.total,
                deserialized.token_budget.total
            );
            assert_eq!(allocation.priority, deserialized.priority);
            assert_eq!(allocation.parallel_capacity, deserialized.parallel_capacity);
        }

        #[test]
        fn test_m2_optimizations_roundtrip() {
            let optimizations = M2Optimizations {
                context_optimization: ContextOptimization {
                    method: "semantic".to_string(),
                    compression_ratio: 0.75,
                },
                output_optimization: OutputOptimization {
                    format: "json".to_string(),
                    template: "default".to_string(),
                    max_output_length: 4096,
                    streaming_enabled: true,
                    compression_enabled: false,
                },
                cost_optimization: CostOptimization {
                    strategy: "aggressive".to_string(),
                    max_budget: 1.0,
                    target_cost_reduction: 0.5,
                    target_latency_reduction: 0.3,
                    parallel_processing_enabled: true,
                    caching_enabled: true,
                },
                target_parameters: 1000000,
            };
            let json = serde_json::to_string(&optimizations).unwrap();
            let deserialized: M2Optimizations = serde_json::from_str(&json).unwrap();

            assert_eq!(
                optimizations.context_optimization.method,
                deserialized.context_optimization.method
            );
            assert_eq!(
                optimizations.output_optimization.format,
                deserialized.output_optimization.format
            );
            assert_eq!(
                optimizations.cost_optimization.strategy,
                deserialized.cost_optimization.strategy
            );
        }

        #[test]
        fn test_phase_result_roundtrip() {
            let result = PhaseResult {
                output: "Phase completed successfully".to_string(),
                confidence: 0.92,
            };
            let json = serde_json::to_string(&result).unwrap();
            let deserialized: PhaseResult = serde_json::from_str(&json).unwrap();

            assert_eq!(result.output, deserialized.output);
            assert!((result.confidence - deserialized.confidence).abs() < f64::EPSILON);
        }

        #[test]
        fn test_reasoning_step_result_roundtrip() {
            let result = ReasoningStepResult {
                content: "Reasoning step completed".to_string(),
            };
            let json = serde_json::to_string(&result).unwrap();
            let deserialized: ReasoningStepResult = serde_json::from_str(&json).unwrap();

            assert_eq!(result.content, deserialized.content);
        }
    }

    // =========================================================================
    // Enum Serialization Tests (snake_case)
    // =========================================================================

    mod enum_serialization {
        use super::*;

        #[test]
        fn test_agent_framework_snake_case() {
            assert_eq!(
                serde_json::to_string(&AgentFramework::ClaudeCode).unwrap(),
                "\"claude_code\""
            );
            assert_eq!(
                serde_json::to_string(&AgentFramework::Cline).unwrap(),
                "\"cline\""
            );
            assert_eq!(
                serde_json::to_string(&AgentFramework::KiloCode).unwrap(),
                "\"kilo_code\""
            );
            assert_eq!(
                serde_json::to_string(&AgentFramework::Droid).unwrap(),
                "\"droid\""
            );
            assert_eq!(
                serde_json::to_string(&AgentFramework::RooCode).unwrap(),
                "\"roo_code\""
            );
            assert_eq!(
                serde_json::to_string(&AgentFramework::BlackBoxAi).unwrap(),
                "\"black_box_ai\""
            );
        }

        #[test]
        fn test_agent_framework_deserialize() {
            let claude: AgentFramework = serde_json::from_str("\"claude_code\"").unwrap();
            assert_eq!(claude, AgentFramework::ClaudeCode);

            let kilo: AgentFramework = serde_json::from_str("\"kilo_code\"").unwrap();
            assert_eq!(kilo, AgentFramework::KiloCode);
        }

        #[test]
        fn test_use_case_snake_case() {
            assert_eq!(
                serde_json::to_string(&UseCase::CodeAnalysis).unwrap(),
                "\"code_analysis\""
            );
            assert_eq!(
                serde_json::to_string(&UseCase::BugFinding).unwrap(),
                "\"bug_finding\""
            );
            assert_eq!(
                serde_json::to_string(&UseCase::Documentation).unwrap(),
                "\"documentation\""
            );
            assert_eq!(
                serde_json::to_string(&UseCase::Architecture).unwrap(),
                "\"architecture\""
            );
            assert_eq!(
                serde_json::to_string(&UseCase::General).unwrap(),
                "\"general\""
            );
        }

        #[test]
        fn test_task_type_snake_case() {
            assert_eq!(
                serde_json::to_string(&TaskType::CodeAnalysis).unwrap(),
                "\"code_analysis\""
            );
            assert_eq!(
                serde_json::to_string(&TaskType::BugFinding).unwrap(),
                "\"bug_finding\""
            );
            assert_eq!(
                serde_json::to_string(&TaskType::Documentation).unwrap(),
                "\"documentation\""
            );
            assert_eq!(
                serde_json::to_string(&TaskType::Architecture).unwrap(),
                "\"architecture\""
            );
            assert_eq!(
                serde_json::to_string(&TaskType::General).unwrap(),
                "\"general\""
            );
        }

        #[test]
        fn test_task_domain_snake_case() {
            assert_eq!(
                serde_json::to_string(&TaskDomain::SystemProgramming).unwrap(),
                "\"system_programming\""
            );
            assert_eq!(serde_json::to_string(&TaskDomain::Web).unwrap(), "\"web\"");
            assert_eq!(
                serde_json::to_string(&TaskDomain::Data).unwrap(),
                "\"data\""
            );
            assert_eq!(
                serde_json::to_string(&TaskDomain::General).unwrap(),
                "\"general\""
            );
        }

        #[test]
        fn test_complexity_level_snake_case() {
            assert_eq!(
                serde_json::to_string(&ComplexityLevel::Simple).unwrap(),
                "\"simple\""
            );
            assert_eq!(
                serde_json::to_string(&ComplexityLevel::Moderate).unwrap(),
                "\"moderate\""
            );
            assert_eq!(
                serde_json::to_string(&ComplexityLevel::Complex).unwrap(),
                "\"complex\""
            );
        }

        #[test]
        fn test_quality_level_snake_case() {
            assert_eq!(
                serde_json::to_string(&QualityLevel::Draft).unwrap(),
                "\"draft\""
            );
            assert_eq!(
                serde_json::to_string(&QualityLevel::Standard).unwrap(),
                "\"standard\""
            );
            assert_eq!(
                serde_json::to_string(&QualityLevel::High).unwrap(),
                "\"high\""
            );
        }

        #[test]
        fn test_output_size_snake_case() {
            assert_eq!(
                serde_json::to_string(&OutputSize::Small).unwrap(),
                "\"small\""
            );
            assert_eq!(
                serde_json::to_string(&OutputSize::Medium).unwrap(),
                "\"medium\""
            );
            assert_eq!(
                serde_json::to_string(&OutputSize::Large).unwrap(),
                "\"large\""
            );
        }

        #[test]
        fn test_optimization_goal_snake_case() {
            assert_eq!(
                serde_json::to_string(&OptimizationGoal::Cost).unwrap(),
                "\"cost\""
            );
            assert_eq!(
                serde_json::to_string(&OptimizationGoal::Latency).unwrap(),
                "\"latency\""
            );
            assert_eq!(
                serde_json::to_string(&OptimizationGoal::Quality).unwrap(),
                "\"quality\""
            );
            assert_eq!(
                serde_json::to_string(&OptimizationGoal::BalanceAll).unwrap(),
                "\"balance_all\""
            );
        }

        #[test]
        fn test_synthesis_method_serialization() {
            assert_eq!(
                serde_json::to_string(&SynthesisMethod::Ensemble).unwrap(),
                "\"Ensemble\""
            );
            assert_eq!(
                serde_json::to_string(&SynthesisMethod::WeightedAverage).unwrap(),
                "\"WeightedAverage\""
            );
            assert_eq!(
                serde_json::to_string(&SynthesisMethod::BestOfN).unwrap(),
                "\"BestOfN\""
            );
        }

        #[test]
        fn test_synthesis_strategy_serialization() {
            assert_eq!(
                serde_json::to_string(&SynthesisStrategy::Standard).unwrap(),
                "\"Standard\""
            );
            assert_eq!(
                serde_json::to_string(&SynthesisStrategy::Aggressive).unwrap(),
                "\"Aggressive\""
            );
            assert_eq!(
                serde_json::to_string(&SynthesisStrategy::Conservative).unwrap(),
                "\"Conservative\""
            );
        }

        #[test]
        fn test_compliance_flag_serialization() {
            assert_eq!(
                serde_json::to_string(&ComplianceFlag::GDPRCompliant).unwrap(),
                "\"GDPRCompliant\""
            );
            assert_eq!(
                serde_json::to_string(&ComplianceFlag::HIPAACompliant).unwrap(),
                "\"HIPAACompliant\""
            );
        }

        #[test]
        fn test_validation_method_serialization() {
            assert_eq!(
                serde_json::to_string(&ValidationMethod::SelfCheck).unwrap(),
                "\"SelfCheck\""
            );
            assert_eq!(
                serde_json::to_string(&ValidationMethod::PeerReview).unwrap(),
                "\"PeerReview\""
            );
            assert_eq!(
                serde_json::to_string(&ValidationMethod::FormalVerification).unwrap(),
                "\"FormalVerification\""
            );
        }
    }

    // =========================================================================
    // From Trait Implementation Tests
    // =========================================================================

    mod from_impl {
        use super::*;

        #[test]
        fn test_use_case_code_analysis_to_task_classification() {
            let classification: TaskClassification = UseCase::CodeAnalysis.into();

            assert_eq!(classification.task_type, TaskType::CodeAnalysis);
            assert_eq!(classification.complexity_level, ComplexityLevel::Complex);
            assert_eq!(classification.domain, TaskDomain::SystemProgramming);
            assert_eq!(classification.expected_output_size, OutputSize::Large);
        }

        #[test]
        fn test_use_case_bug_finding_to_task_classification() {
            let classification: TaskClassification = UseCase::BugFinding.into();

            assert_eq!(classification.task_type, TaskType::BugFinding);
            assert_eq!(classification.complexity_level, ComplexityLevel::Moderate);
            assert_eq!(classification.domain, TaskDomain::SystemProgramming);
            assert_eq!(classification.expected_output_size, OutputSize::Medium);
        }

        #[test]
        fn test_use_case_documentation_to_task_classification() {
            let classification: TaskClassification = UseCase::Documentation.into();

            assert_eq!(classification.task_type, TaskType::Documentation);
            assert_eq!(classification.complexity_level, ComplexityLevel::Moderate);
            assert_eq!(classification.domain, TaskDomain::General);
            assert_eq!(classification.expected_output_size, OutputSize::Medium);
        }

        #[test]
        fn test_use_case_architecture_to_task_classification() {
            let classification: TaskClassification = UseCase::Architecture.into();

            assert_eq!(classification.task_type, TaskType::Architecture);
            assert_eq!(classification.complexity_level, ComplexityLevel::Complex);
            assert_eq!(classification.domain, TaskDomain::General);
            assert_eq!(classification.expected_output_size, OutputSize::Large);
        }

        #[test]
        fn test_use_case_general_to_task_classification() {
            let classification: TaskClassification = UseCase::General.into();

            assert_eq!(classification.task_type, TaskType::General);
            assert_eq!(classification.complexity_level, ComplexityLevel::Moderate);
            assert_eq!(classification.domain, TaskDomain::General);
            assert_eq!(classification.expected_output_size, OutputSize::Medium);
        }

        #[test]
        fn test_all_use_cases_have_default_time_constraints() {
            for use_case in [
                UseCase::CodeAnalysis,
                UseCase::BugFinding,
                UseCase::Documentation,
                UseCase::Architecture,
                UseCase::General,
            ] {
                let classification: TaskClassification = use_case.into();
                assert_eq!(
                    classification.time_constraints.max_duration_ms,
                    Some(300_000)
                );
            }
        }

        #[test]
        fn test_all_use_cases_have_default_quality_requirements() {
            for use_case in [
                UseCase::CodeAnalysis,
                UseCase::BugFinding,
                UseCase::Documentation,
                UseCase::Architecture,
                UseCase::General,
            ] {
                let classification: TaskClassification = use_case.into();
                assert_eq!(
                    classification.quality_requirements.quality_level,
                    QualityLevel::Standard
                );
            }
        }
    }

    // =========================================================================
    // Display Trait Implementation Tests
    // =========================================================================

    mod display_impl {
        use super::*;

        #[test]
        fn test_task_type_display_code_analysis() {
            assert_eq!(format!("{}", TaskType::CodeAnalysis), "CodeAnalysis");
        }

        #[test]
        fn test_task_type_display_bug_finding() {
            assert_eq!(format!("{}", TaskType::BugFinding), "BugFinding");
        }

        #[test]
        fn test_task_type_display_documentation() {
            assert_eq!(format!("{}", TaskType::Documentation), "Documentation");
        }

        #[test]
        fn test_task_type_display_architecture() {
            assert_eq!(format!("{}", TaskType::Architecture), "Architecture");
        }

        #[test]
        fn test_task_type_display_general() {
            assert_eq!(format!("{}", TaskType::General), "General");
        }
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    mod clone_tests {
        use super::*;

        #[test]
        fn test_m2_config_clone() {
            let original = M2Config {
                endpoint: "https://custom.api.com".to_string(),
                api_key: "secret-key".to_string(),
                max_context_length: 100_000,
                max_output_length: 50_000,
                rate_limit: RateLimitConfig::default(),
                performance: PerformanceConfig::default(),
            };
            let cloned = original.clone();

            assert_eq!(original.endpoint, cloned.endpoint);
            assert_eq!(original.api_key, cloned.api_key);
            assert_eq!(original.max_context_length, cloned.max_context_length);
        }

        #[test]
        fn test_optimization_goals_clone() {
            let original = OptimizationGoals {
                primary_goal: OptimizationGoal::Cost,
                secondary_goals: vec![OptimizationGoal::Latency],
                constraints: OptimizationConstraints::default(),
                performance_targets: PerformanceTargets::default(),
            };
            let cloned = original.clone();

            assert_eq!(original.primary_goal, cloned.primary_goal);
            assert_eq!(original.secondary_goals.len(), cloned.secondary_goals.len());
        }

        #[test]
        fn test_execution_metrics_clone() {
            let original = ExecutionMetrics {
                duration_ms: 1000,
                token_usage: TokenUsage {
                    total: 5000,
                    context: 3000,
                    output: 1500,
                    validation: 500,
                },
                cost_metrics: CostMetrics::default(),
                quality_metrics: QualityMetrics::default(),
                performance_metrics: PerformanceMetrics::default(),
                audit_trail: AuditTrail {
                    steps: vec!["step1".to_string()],
                    timestamp: 123456,
                    compliance_flags: vec![ComplianceFlag::GDPRCompliant],
                },
            };
            let cloned = original.clone();

            assert_eq!(original.duration_ms, cloned.duration_ms);
            assert_eq!(original.token_usage.total, cloned.token_usage.total);
            assert_eq!(
                original.audit_trail.steps.len(),
                cloned.audit_trail.steps.len()
            );
        }

        #[test]
        fn test_enum_copy() {
            // Test Copy trait for enums
            let framework = AgentFramework::ClaudeCode;
            let copied = framework;
            assert_eq!(framework, copied);

            let use_case = UseCase::CodeAnalysis;
            let copied = use_case;
            assert_eq!(use_case, copied);

            let task_type = TaskType::BugFinding;
            let copied = task_type;
            assert_eq!(task_type, copied);
        }
    }

    // =========================================================================
    // Equality Tests
    // =========================================================================

    mod equality_tests {
        use super::*;

        #[test]
        fn test_agent_framework_equality() {
            assert_eq!(AgentFramework::ClaudeCode, AgentFramework::ClaudeCode);
            assert_ne!(AgentFramework::ClaudeCode, AgentFramework::Cline);
        }

        #[test]
        fn test_use_case_equality() {
            assert_eq!(UseCase::CodeAnalysis, UseCase::CodeAnalysis);
            assert_ne!(UseCase::CodeAnalysis, UseCase::BugFinding);
        }

        #[test]
        fn test_task_type_equality() {
            assert_eq!(TaskType::Documentation, TaskType::Documentation);
            assert_ne!(TaskType::Documentation, TaskType::Architecture);
        }

        #[test]
        fn test_task_domain_equality() {
            assert_eq!(TaskDomain::Web, TaskDomain::Web);
            assert_ne!(TaskDomain::Web, TaskDomain::Data);
        }

        #[test]
        fn test_complexity_level_equality() {
            assert_eq!(ComplexityLevel::Simple, ComplexityLevel::Simple);
            assert_ne!(ComplexityLevel::Simple, ComplexityLevel::Complex);
        }

        #[test]
        fn test_quality_level_equality() {
            assert_eq!(QualityLevel::High, QualityLevel::High);
            assert_ne!(QualityLevel::High, QualityLevel::Draft);
        }

        #[test]
        fn test_output_size_equality() {
            assert_eq!(OutputSize::Large, OutputSize::Large);
            assert_ne!(OutputSize::Large, OutputSize::Small);
        }

        #[test]
        fn test_optimization_goal_equality() {
            assert_eq!(OptimizationGoal::Cost, OptimizationGoal::Cost);
            assert_ne!(OptimizationGoal::Cost, OptimizationGoal::Quality);
        }

        #[test]
        fn test_time_constraints_equality() {
            let a = TimeConstraints {
                max_duration_ms: Some(1000),
            };
            let b = TimeConstraints {
                max_duration_ms: Some(1000),
            };
            let c = TimeConstraints {
                max_duration_ms: Some(2000),
            };

            assert_eq!(a, b);
            assert_ne!(a, c);
        }

        #[test]
        fn test_quality_requirements_equality() {
            let a = QualityRequirements {
                quality_level: QualityLevel::High,
            };
            let b = QualityRequirements {
                quality_level: QualityLevel::High,
            };
            let c = QualityRequirements {
                quality_level: QualityLevel::Draft,
            };

            assert_eq!(a, b);
            assert_ne!(a, c);
        }

        #[test]
        fn test_synthesis_method_equality() {
            assert_eq!(SynthesisMethod::Ensemble, SynthesisMethod::Ensemble);
            assert_ne!(SynthesisMethod::Ensemble, SynthesisMethod::BestOfN);
        }

        #[test]
        fn test_synthesis_strategy_equality() {
            assert_eq!(SynthesisStrategy::Standard, SynthesisStrategy::Standard);
            assert_ne!(SynthesisStrategy::Standard, SynthesisStrategy::Aggressive);
        }

        #[test]
        fn test_compliance_flag_equality() {
            assert_eq!(ComplianceFlag::GDPRCompliant, ComplianceFlag::GDPRCompliant);
            assert_ne!(
                ComplianceFlag::GDPRCompliant,
                ComplianceFlag::HIPAACompliant
            );
        }

        #[test]
        fn test_validation_method_equality() {
            assert_eq!(ValidationMethod::SelfCheck, ValidationMethod::SelfCheck);
            assert_ne!(ValidationMethod::SelfCheck, ValidationMethod::PeerReview);
        }
    }

    // =========================================================================
    // Hash Tests (for AgentFramework)
    // =========================================================================

    mod hash_tests {
        use super::*;
        use std::collections::HashSet;

        #[test]
        fn test_agent_framework_hash() {
            let mut set = HashSet::new();
            set.insert(AgentFramework::ClaudeCode);
            set.insert(AgentFramework::Cline);
            set.insert(AgentFramework::ClaudeCode); // Duplicate

            assert_eq!(set.len(), 2);
            assert!(set.contains(&AgentFramework::ClaudeCode));
            assert!(set.contains(&AgentFramework::Cline));
        }

        #[test]
        fn test_agent_framework_all_variants_hashable() {
            let mut set = HashSet::new();
            set.insert(AgentFramework::ClaudeCode);
            set.insert(AgentFramework::Cline);
            set.insert(AgentFramework::KiloCode);
            set.insert(AgentFramework::Droid);
            set.insert(AgentFramework::RooCode);
            set.insert(AgentFramework::BlackBoxAi);

            assert_eq!(set.len(), 6);
        }
    }

    // =========================================================================
    // Edge Cases and Boundary Conditions
    // =========================================================================

    mod edge_cases {
        use super::*;

        #[test]
        fn test_empty_string_serialization() {
            let config = M2Config {
                endpoint: "".to_string(),
                api_key: "".to_string(),
                max_context_length: 0,
                max_output_length: 0,
                rate_limit: RateLimitConfig::default(),
                performance: PerformanceConfig::default(),
            };
            let json = serde_json::to_string(&config).unwrap();
            let deserialized: M2Config = serde_json::from_str(&json).unwrap();

            assert_eq!(deserialized.endpoint, "");
            assert_eq!(deserialized.api_key, "");
        }

        #[test]
        fn test_max_values() {
            let usage = TokenUsage {
                total: u64::MAX,
                context: u64::MAX,
                output: u64::MAX,
                validation: u64::MAX,
            };
            let json = serde_json::to_string(&usage).unwrap();
            let deserialized: TokenUsage = serde_json::from_str(&json).unwrap();

            assert_eq!(deserialized.total, u64::MAX);
        }

        #[test]
        fn test_zero_values() {
            let metrics = PerformanceMetrics {
                latency_ms: 0,
                throughput: 0.0,
            };
            let json = serde_json::to_string(&metrics).unwrap();
            let deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();

            assert_eq!(deserialized.latency_ms, 0);
            assert!((deserialized.throughput - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_none_optional_fields() {
            let constraints = TimeConstraints {
                max_duration_ms: None,
            };
            let json = serde_json::to_string(&constraints).unwrap();
            let deserialized: TimeConstraints = serde_json::from_str(&json).unwrap();

            assert!(deserialized.max_duration_ms.is_none());
        }

        #[test]
        fn test_empty_vec_serialization() {
            let goals = OptimizationGoals {
                primary_goal: OptimizationGoal::Cost,
                secondary_goals: vec![],
                constraints: OptimizationConstraints::default(),
                performance_targets: PerformanceTargets::default(),
            };
            let json = serde_json::to_string(&goals).unwrap();
            let deserialized: OptimizationGoals = serde_json::from_str(&json).unwrap();

            assert!(deserialized.secondary_goals.is_empty());
        }

        #[test]
        fn test_large_vec_serialization() {
            let trail = AuditTrail {
                steps: (0..1000).map(|i| format!("step_{}", i)).collect(),
                timestamp: 0,
                compliance_flags: vec![],
            };
            let json = serde_json::to_string(&trail).unwrap();
            let deserialized: AuditTrail = serde_json::from_str(&json).unwrap();

            assert_eq!(deserialized.steps.len(), 1000);
        }

        #[test]
        fn test_special_characters_in_strings() {
            let result = InterleavedResult {
                summary: "Test with special chars: \"quotes\", 'apostrophes', \\backslashes\\, and unicode: \u{1F600}".to_string(),
            };
            let json = serde_json::to_string(&result).unwrap();
            let deserialized: InterleavedResult = serde_json::from_str(&json).unwrap();

            assert!(deserialized.summary.contains("quotes"));
            assert!(deserialized.summary.contains("\u{1F600}"));
        }

        #[test]
        fn test_float_precision() {
            let targets = PerformanceTargets {
                cost_reduction_target: 0.123456789012345,
                latency_reduction_target: 0.987654321098765,
                quality_threshold: 0.555555555555555,
            };
            let json = serde_json::to_string(&targets).unwrap();
            let deserialized: PerformanceTargets = serde_json::from_str(&json).unwrap();

            // JSON float precision may differ slightly
            assert!(
                (targets.cost_reduction_target - deserialized.cost_reduction_target).abs() < 1e-10
            );
        }

        #[test]
        fn test_nested_default_values() {
            // Ensure nested structs also get their defaults
            let config = M2Config::default();

            assert_eq!(config.rate_limit.rpm, 60);
            assert_eq!(config.performance.compression_level, 5);
        }

        #[test]
        fn test_protocol_input_json_value() {
            let input: ProtocolInput = serde_json::json!({
                "query": "test query",
                "context": ["ctx1", "ctx2"],
                "nested": {
                    "key": "value"
                }
            });

            assert!(input.is_object());
            assert_eq!(input["query"], "test query");
        }
    }

    // =========================================================================
    // JSON Deserialization Error Handling Tests
    // =========================================================================

    mod deserialization_errors {
        use super::*;

        #[test]
        fn test_invalid_agent_framework() {
            let result: Result<AgentFramework, _> = serde_json::from_str("\"invalid_framework\"");
            assert!(result.is_err());
        }

        #[test]
        fn test_invalid_use_case() {
            let result: Result<UseCase, _> = serde_json::from_str("\"invalid_use_case\"");
            assert!(result.is_err());
        }

        #[test]
        fn test_invalid_task_type() {
            let result: Result<TaskType, _> = serde_json::from_str("\"invalid_task\"");
            assert!(result.is_err());
        }

        #[test]
        fn test_missing_required_field() {
            // M2Config requires all fields
            let json = r#"{"endpoint": "test"}"#;
            let result: Result<M2Config, _> = serde_json::from_str(json);
            assert!(result.is_err());
        }

        #[test]
        fn test_wrong_type_for_field() {
            let json = r#"{"rpm": "not_a_number", "rps": 1, "burst": 5}"#;
            let result: Result<RateLimitConfig, _> = serde_json::from_str(json);
            assert!(result.is_err());
        }
    }

    // =========================================================================
    // Complex Struct Serialization Tests
    // =========================================================================

    mod complex_structs {
        use super::*;

        #[test]
        fn test_interleaved_protocol_full_roundtrip() {
            let protocol = InterleavedProtocol {
                phases: vec![
                    InterleavedPhase {
                        name: "reasoning".to_string(),
                        parallel_branches: 3,
                        required_confidence: 0.85,
                        validation_methods: vec![ValidationMethod::SelfCheck],
                        synthesis_methods: vec![SynthesisMethod::Ensemble],
                        constraints: CompositeConstraints {
                            time_budget_ms: 30000,
                            token_budget: 10000,
                            dependencies: vec![],
                        },
                    },
                    InterleavedPhase {
                        name: "verification".to_string(),
                        parallel_branches: 1,
                        required_confidence: 0.95,
                        validation_methods: vec![ValidationMethod::FormalVerification],
                        synthesis_methods: vec![SynthesisMethod::BestOfN],
                        constraints: CompositeConstraints {
                            time_budget_ms: 20000,
                            token_budget: 5000,
                            dependencies: vec!["reasoning".to_string()],
                        },
                    },
                ],
                constraints: CompositeConstraints {
                    time_budget_ms: 60000,
                    token_budget: 20000,
                    dependencies: vec![],
                },
                m2_optimizations: M2Optimizations {
                    context_optimization: ContextOptimization {
                        method: "semantic".to_string(),
                        compression_ratio: 0.8,
                    },
                    output_optimization: OutputOptimization {
                        format: "json".to_string(),
                        template: "default".to_string(),
                        max_output_length: 4096,
                        streaming_enabled: true,
                        compression_enabled: false,
                    },
                    cost_optimization: CostOptimization {
                        strategy: "balanced".to_string(),
                        max_budget: 1.0,
                        target_cost_reduction: 0.5,
                        target_latency_reduction: 0.3,
                        parallel_processing_enabled: true,
                        caching_enabled: true,
                    },
                    target_parameters: 456_000_000,
                },
                name: "code_analysis_protocol".to_string(),
                id: "proto-001".to_string(),
                version: "1.0.0".to_string(),
                description: "Protocol for code analysis tasks".to_string(),
                framework_compatibility: vec!["claude_code".to_string(), "cline".to_string()],
                language_support: vec!["rust".to_string(), "python".to_string()],
            };

            let json = serde_json::to_string_pretty(&protocol).unwrap();
            let deserialized: InterleavedProtocol = serde_json::from_str(&json).unwrap();

            assert_eq!(protocol.name, deserialized.name);
            assert_eq!(protocol.phases.len(), deserialized.phases.len());
            assert_eq!(
                protocol.framework_compatibility.len(),
                deserialized.framework_compatibility.len()
            );
            assert_eq!(protocol.phases[0].name, deserialized.phases[0].name);
            assert_eq!(protocol.phases[1].constraints.dependencies[0], "reasoning");
        }

        #[test]
        fn test_deeply_nested_config_roundtrip() {
            let config = M2IntegrationConfig {
                max_concurrent_executions: 20,
                default_timeout_ms: 600_000,
                enable_caching: false,
                enable_monitoring: true,
                default_optimization_goals: OptimizationGoals {
                    primary_goal: OptimizationGoal::Quality,
                    secondary_goals: vec![OptimizationGoal::Latency, OptimizationGoal::Cost],
                    constraints: OptimizationConstraints {
                        max_cost: Some(50.0),
                        max_latency_ms: Some(10000),
                        min_quality: Some(0.95),
                    },
                    performance_targets: PerformanceTargets {
                        cost_reduction_target: 80.0,
                        latency_reduction_target: 0.30,
                        quality_threshold: 0.95,
                    },
                },
            };

            let json = serde_json::to_string(&config).unwrap();
            let deserialized: M2IntegrationConfig = serde_json::from_str(&json).unwrap();

            assert_eq!(
                config.max_concurrent_executions,
                deserialized.max_concurrent_executions
            );
            assert_eq!(
                config.default_optimization_goals.primary_goal,
                deserialized.default_optimization_goals.primary_goal
            );
            assert_eq!(
                config.default_optimization_goals.constraints.max_cost,
                deserialized.default_optimization_goals.constraints.max_cost
            );
        }
    }
}
