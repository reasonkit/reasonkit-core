//! Plugin system for ARF Core

use crate::{config::Config, error::Result};
use std::collections::HashMap;

/// Plugin trait for extending ARF functionality
pub trait Plugin: Send + Sync {
    /// Get plugin name
    fn name(&self) -> &str;

    /// Initialize the plugin
    fn init(&mut self) -> Result<()>;
}

/// Plugin manager for loading and managing plugins
#[derive(Debug)]
pub struct PluginManager {
    plugins: HashMap<String, Box<dyn std::fmt::Debug + Send + Sync>>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new(_config: &Config) -> Result<Self> {
        Ok(Self {
            plugins: HashMap::new(),
        })
    }

    /// Load a plugin by name
    pub fn load(&mut self, _name: &str) -> Result<()> {
        Ok(())
    }
}
