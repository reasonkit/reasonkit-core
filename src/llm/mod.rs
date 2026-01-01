//! Provider-neutral LLM infrastructure.
//!
//! This module hosts low-level clients/adapters for LLM backends that may not fit
//! the OpenAI-compatible `/chat/completions` shape.
//!
//! Current submodules:
//! - `ollama` - Minimal client for Ollama `/api/chat` (supports Ollama Cloud via local server)

pub mod ollama;
