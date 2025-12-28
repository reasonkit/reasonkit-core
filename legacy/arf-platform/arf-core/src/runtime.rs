//! Runtime for ARF Core

use crate::{
    config::Config,
    error::Result,
    plugins::PluginManager,
    state::StateManager,
};

/// ARF Runtime - the main execution context
#[derive(Debug)]
pub struct ArfRuntime {
    config: Config,
    state: StateManager,
    plugin_manager: PluginManager,
}

impl ArfRuntime {
    /// Create a new ARF runtime
    pub async fn new(
        config: Config,
        state: StateManager,
        plugin_manager: PluginManager,
    ) -> Result<Self> {
        Ok(Self {
            config,
            state,
            plugin_manager,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the state manager
    pub fn state(&self) -> &StateManager {
        &self.state
    }

    /// Run the runtime main loop
    pub async fn run(&mut self) -> Result<()> {
        tracing::info!("ARF Runtime starting...");
        Ok(())
    }

    /// Shutdown the runtime
    pub async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("ARF Runtime shutting down...");
        Ok(())
    }
}
