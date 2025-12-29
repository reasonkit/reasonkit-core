//! ThinkTool Module Type Definitions
//!
//! ## Architecture Note
//!
//! **Actual execution happens through Protocols, NOT module structs.**
//!
//! The executor in `executor.rs` uses Protocol definitions from `registry.rs`
//! which contain:
//! - Step-by-step prompt templates with handlebars placeholders
//! - LLM call orchestration via `UnifiedLlmClient`
//! - Response parsing with confidence extraction
//!
//! ## Usage
//!
//! ```ignore
//! let executor = ProtocolExecutor::new()?;
//! let result = executor.execute("gigathink", ProtocolInput::query("question")).await?;
//! ```
//!
//! ## Available Protocols
//!
//! - `gigathink` - Expansive creative thinking (10+ perspectives)
//! - `laserlogic` - Precision deductive reasoning with fallacy detection
//! - `bedrock` - First principles decomposition
//! - `proofguard` - Multi-source verification (3+ sources)
//! - `brutalhonesty` - Adversarial self-critique
//!
//! See `registry.rs` for full protocol definitions.

use serde::{Deserialize, Serialize};

pub mod bedrock;
pub mod brutalhonesty;
pub mod gigathink;
pub mod laserlogic;
pub mod proofguard;

pub use bedrock::BedRock;
pub use brutalhonesty::BrutalHonesty;
pub use gigathink::GigaThink;
pub use laserlogic::LaserLogic;
pub use proofguard::ProofGuard;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkToolModuleConfig {
    pub name: String,
    pub version: String,
    pub description: String,
    pub confidence_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkToolContext {
    pub query: String,
    pub previous_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkToolOutput {
    pub module: String,
    pub confidence: f64,
    pub output: serde_json::Value,
}

pub trait ThinkToolModule: Send + Sync {
    fn config(&self) -> &ThinkToolModuleConfig;
    fn execute(&self, context: &ThinkToolContext) -> Result<ThinkToolOutput, crate::error::Error>;
}
