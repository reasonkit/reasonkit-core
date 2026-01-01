//! # ReasonKit Engine Module
//!
//! High-performance async reasoning engine that orchestrates ThinkTool execution
//! with memory integration, streaming support, and concurrent processing.
//!
//! ## Core Components
//!
//! - **ReasoningLoop**: Main async engine for structured reasoning
//! - **ReasoningSession**: Stateful session with context accumulation
//! - **StreamingOutput**: Real-time reasoning step streaming
//!
//! ## Architecture
//!
//! ```text
//! +------------------+     +------------------+     +------------------+
//! |   User Prompt    | --> |  ReasoningLoop   | --> |  Decision Output |
//! +------------------+     +------------------+     +------------------+
//!                                  |
//!                    +-------------+-------------+
//!                    |             |             |
//!               +----v----+  +-----v-----+  +----v----+
//!               | Memory  |  | ThinkTool |  | Profile |
//!               | Query   |  | Executor  |  | System  |
//!               +---------+  +-----------+  +---------+
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::engine::{ReasoningLoop, ReasoningConfig, Profile};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let loop_engine = ReasoningLoop::builder()
//!         .with_profile(Profile::Balanced)
//!         .with_memory(memory_client)
//!         .build()?;
//!
//!     // Streaming execution
//!     let mut stream = loop_engine.reason_stream("Should we adopt microservices?").await?;
//!     while let Some(step) = stream.next().await {
//!         println!("Step: {:?}", step);
//!     }
//!
//!     // Blocking execution
//!     let decision = loop_engine.reason("Should we adopt microservices?").await?;
//!     println!("Decision: {}", decision.conclusion);
//!
//!     Ok(())
//! }
//! ```

pub mod reasoning_loop;

// Re-exports for convenience
pub use reasoning_loop::{
    Decision, MemoryContext, Profile, ReasoningConfig, ReasoningError, ReasoningEvent,
    ReasoningLoop, ReasoningLoopBuilder, ReasoningSession, ReasoningStep, StepKind, StreamHandle,
    ThinkToolResult,
};
