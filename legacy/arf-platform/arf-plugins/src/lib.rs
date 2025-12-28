//! ARF Plugins - Plugin system for Absolute Reasoning Framework
//!
//! This crate provides a WebAssembly-based plugin system that allows
//! extending the reasoning capabilities of ARF with custom modules.

use anyhow::Result;
use wasmtime::{Engine, Module, Store};

/// Plugin manager for loading and executing WebAssembly plugins
pub struct PluginManager {
    engine: Engine,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Result<Self> {
        let engine = Engine::default();
        Ok(Self { engine })
    }

    /// Load a plugin from WebAssembly bytes
    pub fn load_plugin(&self, wasm_bytes: &[u8]) -> Result<Plugin> {
        let module = Module::from_binary(&self.engine, wasm_bytes)?;
        let store = Store::new(&self.engine, ());

        Ok(Plugin { module, store })
    }
}

/// A loaded WebAssembly plugin
pub struct Plugin {
    module: Module,
    store: Store<()>,
}

impl Plugin {
    /// Get the plugin's module
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Get the plugin's store
    pub fn store(&self) -> &Store<()> {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        assert!(manager.is_ok());
    }
}