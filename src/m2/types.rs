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
