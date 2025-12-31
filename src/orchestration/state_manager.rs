//! # State Management for Long-Horizon Execution
//!
//! This module provides advanced state management capabilities for maintaining context,
//! memory, and execution state across extended tool calling sequences (100+ calls).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use crate::error::Error;

/// Global execution state manager
pub struct StateManager {
    /// Current execution context
    current_context: Arc<RwLock<ExecutionContext>>,
    /// Historical state snapshots for recovery
    snapshots: Arc<Mutex<Vec<ContextSnapshot>>>,
    /// Persistent storage for long-term state
    persistent_storage: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    /// Memory tracking for context optimization
    memory_tracker: Arc<Mutex<MemoryTracker>>,
    /// Configuration
    config: StateManagerConfig,
}

impl Default for StateManager {
    fn default() -> Self {
        Self::new()
    }
}

impl StateManager {
    pub fn new() -> Self {
        Self {
            current_context: Arc::new(RwLock::new(ExecutionContext::new())),
            snapshots: Arc::new(Mutex::new(Vec::new())),
            persistent_storage: Arc::new(Mutex::new(HashMap::new())),
            memory_tracker: Arc::new(Mutex::new(MemoryTracker::new())),
            config: StateManagerConfig::default(),
        }
    }

    /// Initialize the execution context with initial state
    pub async fn initialize_context(&self, initial_state: &serde_json::Value) -> Result<(), Error> {
        {
            let mut context = self.current_context.write().await;
            context.initialize(initial_state).await?;
        }

        self.create_snapshot().await?;

        tracing::info!(
            "Execution context initialized with {} bytes of initial state",
            serde_json::to_string(initial_state)?.len()
        );

        Ok(())
    }

    /// Update the current execution context
    pub async fn update_context(
        &self,
        updates: &HashMap<String, serde_json::Value>,
    ) -> Result<(), Error> {
        let mut context = self.current_context.write().await;

        for (key, value) in updates {
            context.update_variable(key, value).await?;
        }

        tracing::debug!("Context updated with {} variables", updates.len());
        Ok(())
    }

    /// Get current context snapshot
    pub async fn get_current_context(&self) -> Result<serde_json::Value, Error> {
        let context = self.current_context.read().await;
        context.serialize().await
    }

    /// Store persistent data that survives across tool calls
    pub async fn store_persistent(&self, key: &str, value: serde_json::Value) -> Result<(), Error> {
        let mut storage = self.persistent_storage.lock().await;
        storage.insert(key.to_string(), value.clone());

        tracing::debug!(
            "Stored persistent data: {} ({} bytes)",
            key,
            serde_json::to_string(&value)?.len()
        );

        Ok(())
    }

    /// Retrieve persistent data
    pub async fn get_persistent(&self, key: &str) -> Result<Option<serde_json::Value>, Error> {
        let storage = self.persistent_storage.lock().await;
        Ok(storage.get(key).cloned())
    }

    /// Create a checkpoint snapshot of current state
    pub async fn create_snapshot(&self) -> Result<ContextSnapshot, Error> {
        let (execution_context, tool_call_count) = {
            let context = self.current_context.read().await;
            (context.serialize().await?, context.get_tool_call_count())
        };

        let persistent_data = {
            let storage = self.persistent_storage.lock().await;
            storage.clone()
        };

        let context_size_bytes = serde_json::to_string(&execution_context)?.len();
        let persistent_storage_size_bytes = serde_json::to_string(&persistent_data)?.len();
        let total_size_bytes = context_size_bytes + persistent_storage_size_bytes;

        let (memory_efficiency, peak_usage_mb) = {
            let tracker = self.memory_tracker.lock().await;
            (tracker.calculate_efficiency(), tracker.peak_usage_mb)
        };

        let current_usage_mb = total_size_bytes as f64 / 1_048_576.0;

        let snapshot = ContextSnapshot {
            id: format!("snapshot_{}", chrono::Utc::now().timestamp()),
            timestamp: chrono::Utc::now().timestamp(),
            execution_context,
            persistent_data,
            memory_usage: MemoryUsage {
                context_size_bytes,
                persistent_storage_size_bytes,
                total_size_bytes,
                memory_efficiency,
                peak_usage_mb,
                current_usage_mb,
            },
            tool_call_count,
            checkpoint_metadata: HashMap::new(),
            compressed: false,
            compression_ratio: 1.0,
        };

        // Add to snapshots history
        let mut snapshots = self.snapshots.lock().await;
        snapshots.push(snapshot.clone());

        // Maintain snapshot limit
        if snapshots.len() > self.config.max_snapshots {
            let removed = snapshots.remove(0);
            tracing::debug!("Removed old snapshot: {}", removed.id);
        }

        // Update memory tracking
        {
            let mut tracker = self.memory_tracker.lock().await;
            tracker.record_snapshot(&snapshot);
        }

        tracing::info!(
            "Created snapshot {} with {} bytes of data",
            snapshot.id,
            serde_json::to_string(&snapshot.execution_context)?.len()
        );

        Ok(snapshot)
    }

    /// Restore from a specific snapshot
    pub async fn restore_snapshot(&self, snapshot_id: &str) -> Result<(), Error> {
        let snapshots = self.snapshots.lock().await;

        let snapshot = snapshots
            .iter()
            .find(|s| s.id == snapshot_id)
            .ok_or_else(|| Error::Validation(format!("Snapshot '{}' not found", snapshot_id)))?;

        // Restore execution context
        {
            let mut context = self.current_context.write().await;
            context.deserialize(&snapshot.execution_context).await?;
        }

        // Restore persistent storage
        {
            let mut storage = self.persistent_storage.lock().await;
            *storage = snapshot.persistent_data.clone();
        }

        tracing::info!("Restored from snapshot: {}", snapshot_id);
        Ok(())
    }

    /// Get the most recent snapshot
    pub async fn get_latest_snapshot(&self) -> Result<Option<ContextSnapshot>, Error> {
        let snapshots = self.snapshots.lock().await;
        Ok(snapshots.last().cloned())
    }

    /// Optimize memory usage by compressing old snapshots
    pub async fn optimize_memory(&self) -> Result<MemoryOptimizationResult, Error> {
        let mut storage = self.persistent_storage.lock().await;
        let mut snapshots = self.snapshots.lock().await;

        let mut compression_count = 0;
        let mut original_size = 0;
        let mut compressed_size = 0;

        // Compress old snapshots if needed
        for snapshot in snapshots.iter_mut() {
            if snapshot.timestamp < chrono::Utc::now().timestamp() - 3600 {
                let serialized = serde_json::to_string(&snapshot.execution_context)?;
                original_size += serialized.len();

                // Simple compression (in real implementation, use proper compression)
                let compressed = base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    serialized.as_bytes(),
                );
                compressed_size += compressed.len();

                snapshot.compressed = true;
                snapshot.compression_ratio = if original_size > 0 {
                    compressed_size as f64 / original_size as f64
                } else {
                    1.0
                };

                compression_count += 1;
            }
        }

        // Clean up expired persistent data
        let before_count = storage.len();
        storage.retain(|_key, value| {
            let expire_time = chrono::Utc::now().timestamp() - self.config.data_ttl_seconds;
            value
                .get("timestamp")
                .and_then(|ts| ts.as_i64())
                .map(|ts| ts > expire_time)
                .unwrap_or(true)
        });
        let after_count = storage.len();
        let cleaned_count = before_count - after_count;

        let result = MemoryOptimizationResult {
            compressed_snapshots: compression_count,
            cleaned_data_items: cleaned_count as u32,
            memory_saved_mb: ((original_size - compressed_size) as f64 / 1_048_576.0).max(0.0),
            optimization_timestamp: chrono::Utc::now().timestamp(),
        };

        tracing::info!(
            "Memory optimization completed: {} snapshots compressed, {} data items cleaned",
            compression_count,
            cleaned_count
        );

        Ok(result)
    }

    /// Get current memory usage statistics
    pub async fn get_current_memory_usage(&self) -> Result<MemoryUsage, Error> {
        let context = self.current_context.read().await;
        let storage = self.persistent_storage.lock().await;
        let tracker = self.memory_tracker.lock().await;

        let context_size = serde_json::to_string(&context.serialize().await?)?.len();
        let storage_size = serde_json::to_string(&*storage)?.len();
        let total_size = context_size + storage_size;

        Ok(MemoryUsage {
            context_size_bytes: context_size,
            persistent_storage_size_bytes: storage_size,
            total_size_bytes: total_size,
            memory_efficiency: tracker.calculate_efficiency(),
            peak_usage_mb: tracker.peak_usage_mb,
            current_usage_mb: total_size as f64 / 1_048_576.0,
        })
    }

    /// Clean up expired data
    pub async fn cleanup_expired_data(&self) -> Result<u32, Error> {
        let mut storage = self.persistent_storage.lock().await;
        let expire_time = chrono::Utc::now().timestamp() - self.config.data_ttl_seconds;

        let before_count = storage.len();
        storage.retain(|_key, value| {
            value
                .get("timestamp")
                .and_then(|ts| ts.as_i64())
                .map(|ts| ts > expire_time)
                .unwrap_or(true)
        });

        let cleaned_count = before_count - storage.len();

        tracing::debug!("Cleaned up {} expired data items", cleaned_count);
        Ok(cleaned_count as u32)
    }
}

/// Execution context that maintains state across tool calls
#[derive(Debug)]
struct ExecutionContext {
    /// Current tool call sequence number
    tool_call_count: u32,
    /// Shared variables accessible across tool calls
    shared_variables: HashMap<String, serde_json::Value>,
    /// Execution metadata
    metadata: HashMap<String, serde_json::Value>,
    /// Component-specific state
    component_states: HashMap<String, ComponentState>,
    /// Memory-efficient context cache
    #[allow(dead_code)]
    context_cache: ContextCache,
    /// Created timestamp
    created_at: u64,
}

impl ExecutionContext {
    fn new() -> Self {
        Self {
            tool_call_count: 0,
            shared_variables: HashMap::new(),
            metadata: HashMap::new(),
            component_states: HashMap::new(),
            context_cache: ContextCache::new(),
            created_at: chrono::Utc::now().timestamp() as u64,
        }
    }

    /// Initialize with provided state
    async fn initialize(&mut self, initial_state: &serde_json::Value) -> Result<(), Error> {
        if let Some(variables) = initial_state.get("variables") {
            if let Ok(vars) =
                serde_json::from_value::<HashMap<String, serde_json::Value>>(variables.clone())
            {
                self.shared_variables = vars;
            }
        }

        if let Some(metadata) = initial_state.get("metadata") {
            if let Ok(meta) =
                serde_json::from_value::<HashMap<String, serde_json::Value>>(metadata.clone())
            {
                self.metadata = meta;
            }
        }

        Ok(())
    }

    /// Update a shared variable
    async fn update_variable(&mut self, key: &str, value: &serde_json::Value) -> Result<(), Error> {
        self.shared_variables.insert(key.to_string(), value.clone());
        Ok(())
    }

    /// Get tool call count
    fn get_tool_call_count(&self) -> u32 {
        self.tool_call_count
    }

    /// Increment tool call count
    #[allow(dead_code)]
    fn increment_tool_call_count(&mut self) {
        self.tool_call_count += 1;
    }

    /// Serialize context for persistence
    async fn serialize(&self) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({
            "tool_call_count": self.tool_call_count,
            "shared_variables": self.shared_variables,
            "metadata": self.metadata,
            "component_states": self.component_states,
            "created_at": self.created_at,
        }))
    }

    /// Deserialize context from snapshot
    async fn deserialize(&mut self, data: &serde_json::Value) -> Result<(), Error> {
        if let Some(tool_call_count) = data.get("tool_call_count").and_then(|v| v.as_u64()) {
            self.tool_call_count = tool_call_count as u32;
        }

        if let Some(variables) = data.get("shared_variables") {
            if let Ok(vars) =
                serde_json::from_value::<HashMap<String, serde_json::Value>>(variables.clone())
            {
                self.shared_variables = vars;
            }
        }

        if let Some(metadata) = data.get("metadata") {
            if let Ok(meta) =
                serde_json::from_value::<HashMap<String, serde_json::Value>>(metadata.clone())
            {
                self.metadata = meta;
            }
        }

        if let Some(component_states) = data.get("component_states") {
            if let Ok(states) =
                serde_json::from_value::<HashMap<String, ComponentState>>(component_states.clone())
            {
                self.component_states = states;
            }
        }

        Ok(())
    }
}

/// Component-specific state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComponentState {
    pub component_name: String,
    pub state_data: serde_json::Value,
    pub last_updated: u64,
    pub access_count: u32,
}

/// Context cache for memory optimization
#[derive(Debug)]
#[allow(dead_code)]
struct ContextCache {
    /// Frequently accessed data with LRU eviction
    #[allow(dead_code)]
    lru_cache: HashMap<String, serde_json::Value>,
    /// Cache capacity
    #[allow(dead_code)]
    capacity: usize,
    /// Current cache size
    #[allow(dead_code)]
    current_size: usize,
}

#[allow(dead_code)]
impl ContextCache {
    fn new() -> Self {
        Self {
            lru_cache: HashMap::new(),
            capacity: 100, // Cache up to 100 items
            current_size: 0,
        }
    }

    /// Add item to cache
    fn add(&mut self, key: &str, value: serde_json::Value) {
        if self.lru_cache.len() >= self.capacity && !self.lru_cache.contains_key(key) {
            // Remove least recently used item
            if let Some(key_to_remove) = self.lru_cache.keys().next().cloned() {
                if let Some(removed_value) = self.lru_cache.remove(&key_to_remove) {
                    self.current_size -= serde_json::to_string(&removed_value)
                        .unwrap_or_default()
                        .len();
                }
            }
        }

        self.lru_cache.insert(key.to_string(), value);
        self.current_size += key.len(); // Simplified size calculation
    }

    /// Get item from cache
    fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.lru_cache.get(key)
    }
}

/// Memory tracker for optimization
#[derive(Debug)]
struct MemoryTracker {
    peak_usage_mb: f64,
    usage_history: Vec<MemorySample>,
    #[allow(dead_code)]
    optimization_threshold_mb: f64,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            peak_usage_mb: 0.0,
            usage_history: Vec::new(),
            optimization_threshold_mb: 100.0, // Trigger optimization at 100MB
        }
    }

    /// Record a memory usage sample
    #[allow(dead_code)]
    fn record_sample(&mut self, usage: &MemoryUsage) {
        let sample = MemorySample {
            timestamp: chrono::Utc::now().timestamp(),
            usage_mb: usage.current_usage_mb,
        };

        self.usage_history.push(sample);
        self.peak_usage_mb = self.peak_usage_mb.max(usage.current_usage_mb);

        // Keep only recent samples
        if self.usage_history.len() > 1000 {
            self.usage_history.remove(0);
        }
    }

    /// Record snapshot creation
    fn record_snapshot(&mut self, snapshot: &ContextSnapshot) {
        // Update peak usage
        let snapshot_size_mb = snapshot.memory_usage.current_usage_mb;
        self.peak_usage_mb = self.peak_usage_mb.max(snapshot_size_mb);
    }

    /// Calculate memory efficiency
    fn calculate_efficiency(&self) -> f64 {
        if self.peak_usage_mb == 0.0 {
            return 1.0;
        }

        // Efficiency based on how close current usage is to peak
        let current_usage = self.usage_history.last().map(|s| s.usage_mb).unwrap_or(0.0);

        (self.peak_usage_mb / current_usage.max(self.peak_usage_mb)).min(1.0)
    }
}

/// Configuration for state manager
#[derive(Debug, Clone)]
pub struct StateManagerConfig {
    pub max_snapshots: usize,
    pub data_ttl_seconds: i64,
    pub memory_limit_mb: u64,
    pub auto_optimize: bool,
    pub checkpoint_interval: u32,
}

impl Default for StateManagerConfig {
    fn default() -> Self {
        Self {
            max_snapshots: 50,
            data_ttl_seconds: 3600, // 1 hour
            memory_limit_mb: 1024,  // 1GB
            auto_optimize: true,
            checkpoint_interval: 10, // Every 10 tool calls
        }
    }
}

/// Context snapshot for state persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub id: String,
    pub timestamp: i64,
    pub execution_context: serde_json::Value,
    pub persistent_data: HashMap<String, serde_json::Value>,
    pub memory_usage: MemoryUsage,
    pub tool_call_count: u32,
    pub checkpoint_metadata: HashMap<String, serde_json::Value>,
    /// Compression status
    pub compressed: bool,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub context_size_bytes: usize,
    pub persistent_storage_size_bytes: usize,
    pub total_size_bytes: usize,
    pub memory_efficiency: f64,
    pub peak_usage_mb: f64,
    pub current_usage_mb: f64,
}

#[derive(Debug)]
struct MemorySample {
    #[allow(dead_code)]
    timestamp: i64,
    usage_mb: f64,
}

#[derive(Debug)]
pub struct MemoryOptimizationResult {
    pub compressed_snapshots: u32,
    pub cleaned_data_items: u32,
    pub memory_saved_mb: f64,
    pub optimization_timestamp: i64,
}

/// State persistence interface
#[async_trait::async_trait]
pub trait StatePersistence {
    async fn save_state(&self, state: &serde_json::Value) -> Result<String, Error>;
    async fn load_state(&self, state_id: &str) -> Result<serde_json::Value, Error>;
    async fn delete_state(&self, state_id: &str) -> Result<(), Error>;
    async fn list_states(&self) -> Result<Vec<String>, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_state_manager_creation() {
        let manager = StateManager::new();
        assert!(manager.get_current_context().await.is_ok());
    }

    #[tokio::test]
    async fn test_context_initialization() {
        let manager = StateManager::new();
        let initial_state = serde_json::json!({
            "variables": {
                "user_id": "12345",
                "session_type": "analysis"
            },
            "metadata": {
                "created_by": "test",
                "version": "1.0"
            }
        });

        assert!(manager.initialize_context(&initial_state).await.is_ok());
    }

    #[tokio::test]
    async fn test_persistent_storage() {
        let manager = StateManager::new();
        let test_data = serde_json::json!({"key": "value", "timestamp": 1234567890});

        assert!(manager
            .store_persistent("test_key", test_data.clone())
            .await
            .is_ok());

        let retrieved = manager.get_persistent("test_key").await.unwrap();
        assert_eq!(retrieved, Some(test_data));
    }

    #[tokio::test]
    async fn test_snapshot_creation() {
        let manager = StateManager::new();
        let initial_state = serde_json::json!({"test": "data"});

        manager.initialize_context(&initial_state).await.unwrap();
        let snapshot = manager.create_snapshot().await.unwrap();

        assert!(!snapshot.id.is_empty());
        assert_eq!(snapshot.tool_call_count, 0);
    }

    #[tokio::test]
    async fn test_memory_usage_tracking() {
        let manager = StateManager::new();
        let usage = manager.get_current_memory_usage().await.unwrap();

        assert!(usage.current_usage_mb >= 0.0);
        assert!(usage.memory_efficiency >= 0.0 && usage.memory_efficiency <= 1.0);
    }
}
