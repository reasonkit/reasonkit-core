//! Ollama client (provider-neutral).
//!
//! This is a minimal, non-streaming (`stream:false`) client for Ollama's `/api/chat`.
//! It is intended to support **Ollama Cloud** via a locally running Ollama server
//! without downloading local model weights.

pub mod client;
pub mod types;

pub use client::{OllamaClient, OllamaClientError};
