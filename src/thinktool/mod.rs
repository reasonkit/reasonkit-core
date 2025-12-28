//! # ThinkTool Protocol Engine
//!
//! Structured reasoning protocols that transform ad-hoc LLM prompting
//! into auditable, reproducible reasoning chains.
//!
//! ## Core Concept
//!
//! A ThinkTool is a structured reasoning protocol that:
//! 1. Defines a reasoning strategy (expansive, deductive, adversarial, etc.)
//! 2. Structures the thought process into auditable steps
//! 3. Produces verifiable output with confidence scores
//! 4. Maintains provenance of the reasoning chain
//!
//! ## Available ThinkTools (OSS)
//!
//! | Tool | Code | Purpose |
//! |------|------|---------|
//! | GigaThink | `gt` | Expansive creative thinking (10+ perspectives) |
//! | LaserLogic | `ll` | Precision deductive reasoning, fallacy detection |
//! | BedRock | `br` | First principles decomposition |
//! | ProofGuard | `pg` | Multi-source verification |
//! | BrutalHonesty | `bh` | Adversarial self-critique |
//!
//! ## Supported LLM Providers (18+)
//!
//! | Tier | Providers |
//! |------|-----------|
//! | Major Cloud | Anthropic, OpenAI, Google Gemini, Vertex AI, Azure OpenAI, AWS Bedrock |
//! | Specialized | xAI (Grok), Groq, Mistral, DeepSeek, Cohere, Perplexity, Cerebras |
//! | Inference | Together AI, Fireworks AI, Alibaba Qwen |
//! | Aggregation | OpenRouter (300+ models), Cloudflare AI Gateway |
//!
//! ## Quick Start - LLM Clients
//!
//! ```rust,ignore
//! use reasonkit::thinktool::{UnifiedLlmClient, LlmRequest};
//!
//! // Use default provider (Anthropic Claude)
//! let client = UnifiedLlmClient::default_anthropic()?;
//!
//! // Use Groq for ultra-fast inference
//! let groq = UnifiedLlmClient::groq("llama-3.3-70b-versatile")?;
//!
//! // Use xAI Grok
//! let grok = UnifiedLlmClient::grok("grok-2")?;
//!
//! // Use OpenRouter for 300+ models
//! let openrouter = UnifiedLlmClient::openrouter("anthropic/claude-sonnet-4")?;
//!
//! // Make a request
//! let response = client.complete(
//!     LlmRequest::new("Explain quantum computing")
//!         .with_system("You are a physics teacher")
//! ).await?;
//! ```
//!
//! ## Provider Auto-Discovery
//!
//! ```rust,ignore
//! use reasonkit::thinktool::{discover_available_providers, create_available_client};
//!
//! // Find all providers with API keys configured
//! let available = discover_available_providers();
//!
//! // Create client using first available provider
//! let client = create_available_client()?;
//! ```
//!
//! ## Protocol Execution Example
//!
//! ```rust,ignore
//! use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let executor = ProtocolExecutor::new()?;
//!
//!     let result = executor.execute(
//!         "gigathink",
//!         ProtocolInput::query("What are the key factors for startup success?")
//!     ).await?;
//!
//!     println!("Confidence: {:.2}", result.confidence);
//!     for perspective in result.perspectives() {
//!         println!("- {}", perspective);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod bedrock_tot;
pub mod benchmark;
pub mod budget;
pub mod calibration;
pub mod consistency;
pub mod debate;
pub mod executor;
pub mod fol;
pub mod llm;
pub mod metrics;
pub mod modules;
pub mod oscillation;
pub mod prm;
pub mod profiles;
pub mod protocol;
pub mod quality;
pub mod registry;
pub mod self_refine;
pub mod socratic;
pub mod step;
pub mod toml_loader;
pub mod tot;
pub mod toulmin;
pub mod trace;
pub mod triangulation;
pub mod tripartite;
pub mod yaml_loader;

// Re-exports
pub use bedrock_tot::{
    BedRockToTConfig, BedRockToTPrompts, BedRockToTResult, Principle, PrincipleExploration,
    PrincipleType,
};
pub use benchmark::{
    Answer, BenchmarkProblem, BenchmarkResults, BenchmarkRunner, CalibrationMetrics,
    ComparisonReport, ConfidenceBin, EvaluationResult,
};
pub use budget::{BudgetConfig, BudgetParseError, BudgetStrategy, BudgetSummary, BudgetTracker};
pub use calibration::{
    platt_scale, temperature_scale, CalibrationConfig, CalibrationDiagnosis, CalibrationReport,
    CalibrationTracker, CategoryCalibration, ConfidenceAdjuster, ConfidenceBin as CalibrationBin,
    Prediction,
};
pub use consistency::{
    ConsistencyResult, ReasoningPath, SelfConsistencyConfig, SelfConsistencyEngine, VotingMethod,
};
pub use debate::{
    AgentRole, Argument, Claim as DebateClaim, DebateArena, DebateConfig, DebatePrompts,
    DebateResult, DebateStats, DebateVerdict, Evidence, Rebuttal as DebateRebuttal, VerdictType,
};
pub use executor::{
    CliToolConfig, ExecutorConfig, ProtocolExecutor, ProtocolInput, ProtocolOutput,
};
pub use fol::{
    Connective, DetectedFallacy, FolArgument, FolConfig, FolFallacy, FolPrompts, FolResult,
    Formula, PremiseAssessment, Quantifier, SoundnessStatus, Term, ValidityStatus,
};
pub use llm::{
    create_available_client, discover_available_providers, get_provider_info, FinishReason,
    LlmClient, LlmConfig, LlmProvider, LlmRequest, LlmResponse, LlmUsage, ProviderExtra,
    ProviderInfo, UnifiedLlmClient,
};
pub use metrics::{
    AggregateStats, ExecutionRecord, ExecutionRecordBuilder, MetricsReport, MetricsTracker,
    StepMetric,
};
pub use modules::*;
pub use oscillation::{
    ConvergentCriterion, ConvergentPhase, CriterionScore, DivergentDimension, DivergentPhase, Idea,
    OscillationConfig, OscillationMetrics, OscillationPrompts, OscillationResult,
};
pub use prm::{
    IssueType, PrmConfig, PrmMetrics, PrmReranker, PrmResult, ScoreAggregation, Severity,
    StepIssue, StepParser, StepScore, VerificationPrompts, VerificationStrategy,
};
pub use profiles::{ProfileRegistry, ReasoningProfile};
pub use protocol::{Protocol, ProtocolStep, ReasoningStrategy, StepAction};
pub use quality::{
    MetricRecord, QualityDashboard, QualityGrade, QualityMetric, QualityReport, QualityScore,
    QualityTargets, TargetViolation, Trend,
};
pub use registry::ProtocolRegistry;
pub use self_refine::{
    DimensionFeedback, FeedbackDimension, IterationFeedback, RefineConfig, RefineIteration,
    RefinePrompts, RefineResult, StopReason,
};
pub use socratic::{
    AnswerType, Aporia, QuestionCategory, SocraticConfig, SocraticPrompts, SocraticQuestion,
    SocraticResult,
};
pub use step::{OutputFormat, StepResult};
pub use tot::{
    parse_thoughts, SearchStrategy, ThoughtNode, ThoughtPrompts, ThoughtState, ToTConfig,
    ToTResult, ToTStats, TreeOfThoughts,
};
pub use toulmin::{
    ArgumentBuilder, ArgumentError, ArgumentEvaluation, ArgumentIssue, Backing, BackingType, Claim,
    ClaimType, EvidenceType, Ground, IssueSeverity as ToulminIssueSeverity, Qualifier, Rebuttal,
    RebuttalSeverity, Scope, ToulminArgument, ToulminComponent, ToulminPrompts, Warrant,
    WarrantType,
};
pub use trace::{ExecutionStatus, ExecutionTrace, StepStatus, StepTrace};
pub use triangulation::{
    Source, SourceTier, SourceType, Stance, TriangulationConfig, TriangulationIssue,
    TriangulationIssueType, TriangulationPrompts, TriangulationResult, Triangulator,
    VerificationConfidence, VerificationRecommendation,
};
pub use tripartite::{
    AlgorithmicDepth, AlgorithmicResponse, AutonomousResponse, CheckResult, CheckStatus,
    ReasoningStep, ReflectiveCheck, ReflectiveResponse, StepType, TripartiteConfig,
    TripartiteMetrics, TripartitePrompts, TripartiteResult,
};
