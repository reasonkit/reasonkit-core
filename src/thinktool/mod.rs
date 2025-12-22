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
//! use reasonkit_core::thinktool::{UnifiedLlmClient, LlmRequest};
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
//! use reasonkit_core::thinktool::{discover_available_providers, create_available_client};
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
//! use reasonkit_core::thinktool::{ProtocolExecutor, ProtocolInput};
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

pub mod protocol;
pub mod step;
pub mod trace;
pub mod executor;
pub mod registry;
pub mod llm;
pub mod profiles;

// Re-exports
pub use protocol::{Protocol, ProtocolStep, ReasoningStrategy, StepAction};
pub use step::{StepResult, OutputFormat};
pub use trace::{ExecutionTrace, StepTrace, ExecutionStatus, StepStatus};
pub use executor::{ProtocolExecutor, ExecutorConfig, ProtocolInput, ProtocolOutput};
pub use registry::ProtocolRegistry;
pub use llm::{
    LlmConfig, LlmProvider, LlmClient, UnifiedLlmClient, LlmRequest, LlmResponse,
    LlmUsage, FinishReason, ProviderExtra, ProviderInfo,
    discover_available_providers, create_available_client, get_provider_info,
};
pub use profiles::{ReasoningProfile, ProfileRegistry};
