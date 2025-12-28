//! State management for ARF Core

use crate::{config::Config, error::Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// State value types
#[derive(Debug, Clone)]
pub enum StateValue {
    String(String),
    Number(f64),
    Bool(bool),
    List(Vec<StateValue>),
    Map(HashMap<String, StateValue>),
}

/// State manager for ARF runtime
#[derive(Debug)]
pub struct StateManager {
    store: Arc<RwLock<HashMap<String, StateValue>>>,
}

impl StateManager {
    /// Create a new state manager
    pub async fn new(_config: &Config) -> Result<Self> {
        Ok(Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Get a value from state
    pub async fn get(&self, key: &str) -> Option<StateValue> {
        let store = self.store.read().await;
        store.get(key).cloned()
    }

    /// Set a value in state
    pub async fn set(&self, key: String, value: StateValue) -> Result<()> {
        let mut store = self.store.write().await;
        store.insert(key, value);
        Ok(())
    }

    /// Remove a value from state
    pub async fn remove(&self, key: &str) -> Option<StateValue> {
        let mut store = self.store.write().await;
        store.remove(key)
    }
}
