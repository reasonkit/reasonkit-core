//! # MiniMax M2 Integration Types
//!
//! Core type definitions for the Interleaved Thinking Protocol Engine integration
//! with MiniMax M2's Agent-Native Architecture.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// MiniMax M2 API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2Config {
    /// API endpoint URL
    pub endpoint: String,
    
    /// API key for authentication
    pub api_key: String,
    
    /// Maximum context length (default: 200,000 tokens)
    pub max_context_length: usize,
    
    /// Maximum output length (default: 128,000 tokens)
    pub max_output_length: usize,
    
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    
    /// Performance optimization settings
    pub performance: PerformanceConfig,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute
    pub rpm: u32,
    
    /// Requests per second
    pub rps: u32,
    
    /// Burst capacity
    pub burst: u32,
}

/// Performance optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Target cost reduction percentage (default: 92)
    pub cost_reduction_target: f64,
    
    /// Target latency in milliseconds
    pub latency_target_ms: u64,
    
    /// Quality threshold (0.0 - 1.0)
    pub quality_threshold: f64,
    
    /// Enable caching
    pub enable_caching: bool,
    
    /// Compression level for context
    pub compression_level: u8,
}

/// Composite instruction constraints for protocol adherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeConstraints {
    /// System-level prompt constraints
    pub system_prompt: SystemPrompt,
    
    /// User query constraints
    pub user_query: UserQuery,
    
    /// Memory context constraints
    pub memory_context: MemoryContext,
    
    /// Tool schema constraints
    pub tool_schemas: Vec<ToolSchema>,
    
    /// Framework-specific constraints
    pub framework_constraints: FrameworkConstraints,
}

/// System prompt constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPrompt {
    /// Base system instruction
    pub instruction: String,
    
    /// Reasoning style guidelines
    pub reasoning_style: ReasoningStyle,
    
    /// Output format requirements
    pub output_format: OutputFormat,
    
    /// Quality standards
    pub quality_standards: QualityStandards,
}

/// Output format requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    /// Structured JSON output
    Structured,
    /// Plain text output
    PlainText,
    /// Markdown formatted output
    Markdown,
    /// Code-formatted output
    Code,
    /// Custom format
    Custom(String),
}

/// Quality standards for protocol execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStandards {
    /// Minimum confidence threshold (0.0 - 1.0)
    pub min_confidence: f64,
    
    /// Require validation before output
    pub require_validation: bool,
    
    /// Require evidence for claims
    pub require_evidence: bool,
}

/// Reasoning style guidelines
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningStyle {
    /// Systematic interleaved thinking
    Interleaved,
    /// Linear sequential reasoning
    Linear,
    /// Tree of thoughts exploration
    TreeOfThoughts,
    /// Chain of thought reasoning
    ChainOfThought,
    /// Multi-perspective analysis
    MultiPerspective,
}

/// User query constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuery {
    /// Original user query
    pub original: String,
    
    /// Clarified/expanded query
    pub clarified: String,
    
    /// Context requirements
    pub context_requirements: ContextRequirements,
    
    /// Expected output type
    pub expected_output: ExpectedOutput,
}

/// Memory context constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    /// Relevant historical context
    pub historical_context: Vec<ContextItem>,
    
    /// Retrieved similar cases
    pub similar_cases: Vec<SimilarCase>,
    
    /// Domain knowledge base
    pub domain_knowledge: Vec<KnowledgeItem>,
    
    /// User preferences and constraints
    pub user_preferences: UserPreferences,
}

/// Tool schema constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name
    pub name: String,
    
    /// Tool description
    pub description: String,
    
    /// Input schema
    pub input_schema: serde_json::Value,
    
    /// Output schema
    pub output_schema: serde_json::Value,
    
    /// Usage constraints
    pub constraints: Vec<UsageConstraint>,
}

/// Framework-specific constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkConstraints {
    /// Target framework
    pub framework: AgentFramework,
    
    /// Framework-specific optimizations
    pub optimizations: Vec<FrameworkOptimization>,
    
    /// Compatibility requirements
    pub compatibility: CompatibilityRequirements,
}

/// Agent framework types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentFramework {
    /// Claude Code
    ClaudeCode,
    /// Cline
    Cline,
    /// Kilo Code
    KiloCode,
    /// Droid (Factory AI)
    Droid,
    /// Roo Code
    RooCode,
    /// BlackBox AI
    BlackBoxAI,
    /// Generic framework
    Generic,
}

/// Interleaved thinking protocol definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavedProtocol {
    /// Protocol identifier
    pub id: String,
    
    /// Protocol name
    pub name: String,
    
    /// Version
    pub version: String,
    
    /// Description
    pub description: String,
    
    /// Interleaved thinking phases
    pub phases: Vec<InterleavedPhase>,
    
    /// M2-specific optimizations
    pub m2_optimizations: M2Optimizations,
    
    /// Framework compatibility
    pub framework_compatibility: Vec<AgentFramework>,
    
    /// Language support
    pub language_support: Vec<ProgrammingLanguage>,
}

/// Interleaved thinking phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavedPhase {
    /// Phase name
    pub name: String,
    
    /// Phase description
    pub description: String,
    
    /// Reasoning depth
    pub depth: u32,
    
    /// Parallel execution branches
    pub parallel_branches: u32,
    
    /// Validation methods
    pub validation_methods: Vec<ValidationMethod>,
    
    /// Synthesis methods
    pub synthesis_methods: Vec<SynthesisMethod>,
    
    /// Phase-specific constraints
    pub constraints: PhaseConstraints,
}

/// M2-specific optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2Optimizations {
    /// Target parameter activation count
    pub target_parameters: u64,
    
    /// Context length optimization
    pub context_optimization: ContextOptimization,
    
    /// Output length optimization
    pub output_optimization: OutputOptimization,
    
    /// Cost optimization strategy
    pub cost_optimization: CostOptimization,
}

/// Validation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationMethod {
    /// Cross-validation between phases
    CrossValidation,
    /// Peer review simulation
    PeerReview,
    /// Empirical testing
    EmpiricalTest,
    /// Consistency checking
    ConsistencyCheck,
    /// Adversarial challenge
    AdversarialChallenge,
}

/// Synthesis methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthesisMethod {
    /// Weighted merging
    WeightedMerge,
    /// Consensus building
    Consensus,
    /// Best-of selection
    BestOf,
    /// Ensemble combination
    Ensemble,
    /// Hierarchical synthesis
    Hierarchical,
}

/// Programming language support
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgrammingLanguage {
    Rust,
    Java,
    Golang,
    Cpp,
    Kotlin,
    ObjectiveC,
    TypeScript,
    JavaScript,
    Python,
    Swift,
    Csharp,
    Scala,
}

/// M2 API request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2Request {
    /// Model identifier
    pub model: String,
    
    /// Input prompt with constraints
    pub prompt: ConstrainedPrompt,
    
    /// Interleaved thinking plan
    pub thinking_plan: InterleavedThinkingPlan,
    
    /// Maximum tokens to generate
    pub max_tokens: usize,
    
    /// Temperature for randomness
    pub temperature: f64,
    
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

/// Constrained prompt after applying composite constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainedPrompt {
    /// Final prompt string
    pub prompt_text: String,
    
    /// Applied constraints summary
    pub applied_constraints: Vec<AppliedConstraint>,
    
    /// Token count
    pub token_count: usize,
    
    /// Optimization notes
    pub optimization_notes: Vec<String>,
}

/// M2 API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2Response {
    /// Generated text
    pub text: String,
    
    /// Token usage
    pub usage: TokenUsage,
    
    /// Reasoning trace
    pub reasoning_trace: ReasoningTrace,
    
    /// Confidence scores
    pub confidence_scores: ConfidenceScores,
    
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input tokens
    pub input_tokens: u32,
    
    /// Output tokens
    pub output_tokens: u32,
    
    /// Total tokens
    pub total_tokens: u32,
    
    /// Cost estimate
    pub cost_estimate: f64,
}

/// Reasoning trace for auditability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// Interleaved phases executed
    pub phases_executed: Vec<PhaseExecution>,
    
    /// Decision points
    pub decision_points: Vec<DecisionPoint>,
    
    /// Validation results
    pub validation_results: Vec<ValidationResult>,
    
    /// Synthesis steps
    pub synthesis_steps: Vec<SynthesisStep>,
}

/// Phase execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseExecution {
    /// Phase name
    pub phase_name: String,
    
    /// Execution start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    
    /// Execution duration
    pub duration: Duration,
    
    /// Input data
    pub input_data: serde_json::Value,
    
    /// Output data
    pub output_data: serde_json::Value,
    
    /// Confidence score
    pub confidence: f64,
}

/// Decision point in reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPoint {
    /// Decision ID
    pub id: String,
    
    /// Decision description
    pub description: String,
    
    /// Options considered
    pub options: Vec<String>,
    
    /// Selected option
    pub selected: String,
    
    /// Reasoning for selection
    pub reasoning: String,
    
    /// Confidence in decision
    pub confidence: f64,
}

/// Confidence scores across dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScores {
    /// Overall confidence
    pub overall: f64,
    
    /// Accuracy confidence
    pub accuracy: f64,
    
    /// Completeness confidence
    pub completeness: f64,
    
    /// Consistency confidence
    pub consistency: f64,
    
    /// Coherence confidence
    pub coherence: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Latency in milliseconds
    pub latency_ms: u64,
    
    /// Cost per token
    pub cost_per_token: f64,
    
    /// Total cost
    pub total_cost: f64,
    
    /// Cost reduction percentage
    pub cost_reduction_percent: f64,
    
    /// Quality score
    pub quality_score: f64,
}

/// Interleaved thinking plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleavedThinkingPlan {
    /// Planning strategy
    pub strategy: PlanningStrategy,
    
    /// Execution phases
    pub phases: Vec<PlanningPhase>,
    
    /// Resource allocation
    pub resource_allocation: ResourceAllocation,
    
    /// Validation checkpoints
    pub validation_checkpoints: Vec<ValidationCheckpoint>,
}

/// Planning strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningStrategy {
    /// Breadth-first exploration
    BreadthFirst,
    /// Depth-first exploration
    DepthFirst,
    /// Bidirectional search
    Bidirectional,
    /// Monte Carlo tree search
    MonteCarloTreeSearch,
    /// Beam search
    BeamSearch,
}

/// Planning phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningPhase {
    /// Phase ID
    pub id: String,
    
    /// Phase name
    pub name: String,
    
    /// Objectives
    pub objectives: Vec<String>,
    
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    
    /// Expected outputs
    pub expected_outputs: Vec<String>,
}

/// Resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Token budget allocation
    pub token_budget: TokenBudget,
    
    /// Time allocation per phase
    pub time_allocation: HashMap<String, Duration>,
    
    /// Parallel execution capacity
    pub parallel_capacity: u32,
    
    /// Quality targets
    pub quality_targets: QualityTargets,
}

/// Token budget allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Total token budget
    pub total: usize,
    
    /// Context allocation
    pub context: usize,
    
    /// Output allocation
    pub output: usize,
    
    /// Reserved for validation
    pub validation: usize,
}

/// Validation checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCheckpoint {
    /// Checkpoint ID
    pub id: String,
    
    /// Phase to validate
    pub phase_id: String,
    
    /// Validation criteria
    pub criteria: Vec<ValidationCriterion>,
    
    /// Minimum confidence threshold
    pub min_confidence: f64,
    
    /// Fallback action if validation fails
    pub fallback_action: FallbackAction,
}

/// Protocol execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolResult {
    /// Result identifier
    pub id: String,
    
    /// Protocol used
    pub protocol_id: String,
    
    /// Execution status
    pub status: ExecutionStatus,
    
    /// Final output
    pub output: ProtocolOutput,
    
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    
    /// Audit trail
    pub audit_trail: AuditTrail,
}

/// Execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    /// Successfully completed
    Completed,
    /// Failed with error
    Failed,
    /// Partially completed
    Partial,
    /// Timeout occurred
    Timeout,
    /// Cancelled by user
    Cancelled,
}

/// Protocol output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolOutput {
    /// Main result
    pub result: serde_json::Value,
    
    /// Supporting evidence
    pub evidence: Vec<Evidence>,
    
    /// Confidence assessment
    pub confidence: f64,
    
    /// Recommendations
    pub recommendations: Vec<String>,
    
    /// Next steps
    pub next_steps: Vec<String>,
}

/// Evidence supporting the result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    
    /// Evidence content
    pub content: String,
    
    /// Confidence in evidence
    pub confidence: f64,
    
    /// Source information
    pub source: String,
}

/// Evidence types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceType {
    /// Empirical data
    Empirical,
    /// Logical derivation
    Logical,
    /// Expert consensus
    ExpertConsensus,
    /// Historical precedent
    Historical,
    /// Theoretical foundation
    Theoretical,
}

/// Execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Total execution time
    pub execution_time: Duration,
    
    /// Token usage
    pub token_usage: TokenUsage,
    
    /// Cost metrics
    pub cost_metrics: CostMetrics,
    
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Cost metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Total cost
    pub total_cost: f64,
    
    /// Cost per reasoning step
    pub cost_per_step: f64,
    
    /// Cost reduction vs baseline
    pub cost_reduction_percent: f64,
    
    /// Cost efficiency score
    pub cost_efficiency: f64,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score
    pub overall_quality: f64,
    
    /// Accuracy score
    pub accuracy: f64,
    
    /// Completeness score
    pub completeness: f64,
    
    /// Consistency score
    pub consistency: f64,
    
    /// Coherence score
    pub coherence: f64,
}

/// Audit trail for compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrail {
    /// Execution ID
    pub execution_id: Uuid,
    
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// User ID (if applicable)
    pub user_id: Option<String>,
    
    /// Protocol version
    pub protocol_version: String,
    
    /// Input hash for integrity
    pub input_hash: String,
    
    /// Output hash for integrity
    pub output_hash: String,
    
    /// Compliance flags
    pub compliance_flags: Vec<ComplianceFlag>,
}

/// Compliance flag
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplianceFlag {
    /// GDPR compliant
    GDPRCompliant,
    /// SOC2 compliant
    SOC2Compliant,
    /// ISO27001 compliant
    ISO27001Compliant,
    /// HIPAA compliant
    HIPAACompliant,
    /// Custom compliance
    Custom(String),
}

// Additional supporting types (abbreviated for space)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextRequirements {
    /// Required context types
    pub required_types: Vec<String>,
    /// Optional context types
    pub optional_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExpectedOutput {
    /// Output format
    pub format: OutputFormat,
    /// Required fields
    pub required_fields: Vec<String>,
    /// Optional fields
    pub optional_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarCase { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeItem { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageConstraint { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkOptimization { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityRequirements { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConstraints { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextOptimization { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputOptimization { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimization { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedConstraint {
    /// Type of constraint applied
    pub constraint_type: String,
    
    /// Description of the constraint
    pub description: String,
    
    /// Impact of the constraint
    pub impact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisStep { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTargets { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCriterion { /* ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackAction { /* ... */ }