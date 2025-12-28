//! # ARF Core Library
//!
//! The core async runtime and foundational components for the Absolute Reasoning Framework.
//! This crate provides the essential building blocks for high-performance reasoning execution.

pub mod config;
pub mod engine;
pub mod error;
pub mod ipc;
pub mod logging;
pub mod plugins;
pub mod runtime;
pub mod state;
pub mod types;

/// Re-export commonly used types
pub use crate::{
    config::Config,
    engine::ReasoningEngine,
    error::{ArfError, Result},
    runtime::ArfRuntime,
    types::*,
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the ARF core system
pub async fn init() -> Result<ArfRuntime> {
    // Initialize logging
    logging::init()?;

    // Load configuration
    let config = Config::load()?;

    // Initialize state management
    let state = state::StateManager::new(&config).await?;

    // Initialize plugin system
    let plugin_manager = plugins::PluginManager::new(&config)?;

    // Create runtime
    let runtime = ArfRuntime::new(config, state, plugin_manager).await?;

    tracing::info!("ARF Core v{} initialized successfully", VERSION);

    Ok(runtime)
}