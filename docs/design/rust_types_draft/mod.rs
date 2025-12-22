//! ReasonKit Core Type Definitions
//!
//! This module provides the foundational types for the entire ReasonKit system:
//! - CLI types and configuration
//! - RAG pipeline types (ingestion, embedding, retrieval)
//! - Multi-agent orchestration types
//!
//! # Architecture Reference
//! - CLI: docs/design/CLI_ARCHITECTURE.md
//! - RAG: docs/design/RAG_PIPELINE_ARCHITECTURE.md
//! - Orchestration: docs/design/MULTI_AGENT_ORCHESTRATION.md

pub mod cli;
pub mod rag;
pub mod orchestration;
pub mod common;

// Re-export commonly used types at module root
pub use cli::{GlobalArgs, OutputFormat, VerbosityLevel};
pub use rag::{Document, Chunk, SearchResult, RetrievalConfig};
pub use orchestration::{Task, TaskId, TaskState, AgentTier};
pub use common::{ReasonKitError, Result};
