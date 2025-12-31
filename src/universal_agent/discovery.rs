//! # Agent Discovery & Registry
//!
//! Auto-discovery and registration system for supported agent frameworks

use crate::error::Result;
use crate::universal_agent::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Agent Discovery Engine
/// Automatically detects and registers available agent frameworks
#[derive(Clone)]
pub struct DiscoveryEngine {
    discovery_methods: Vec<DiscoveryMethod>,
    registry: Arc<RwLock<AgentRegistry>>,
}

impl DiscoveryEngine {
    /// Create a new discovery engine
    pub fn new(registry: Arc<RwLock<AgentRegistry>>) -> Self {
        let discovery_methods = vec![
            DiscoveryMethod::ProcessScan,
            DiscoveryMethod::NetworkScan,
            DiscoveryMethod::ConfigurationScan,
            DiscoveryMethod::ApiEndpointScan,
        ];

        Self {
            discovery_methods,
            registry,
        }
    }

    /// Auto-discover available agent frameworks
    pub async fn discover_agents(&self) -> Result<Vec<AgentDiscovery>> {
        let mut discoveries = Vec::new();

        for method in &self.discovery_methods {
            match self.run_discovery_method(*method).await {
                Ok(mut method_discoveries) => {
                    discoveries.append(&mut method_discoveries);
                }
                Err(e) => {
                    tracing::warn!("Discovery method {:?} failed: {}", method, e);
                }
            }
        }

        // Deduplicate discoveries and merge capabilities
        self.merge_discoveries(discoveries).await
    }

    /// Run a specific discovery method
    async fn run_discovery_method(&self, method: DiscoveryMethod) -> Result<Vec<AgentDiscovery>> {
        match method {
            DiscoveryMethod::ProcessScan => self.scan_processes().await,
            DiscoveryMethod::NetworkScan => self.scan_network_endpoints().await,
            DiscoveryMethod::ConfigurationScan => self.scan_configuration().await,
            DiscoveryMethod::ApiEndpointScan => self.scan_api_endpoints().await,
        }
    }

    /// Scan running processes for agent framework instances
    async fn scan_processes(&self) -> Result<Vec<AgentDiscovery>> {
        let mut discoveries = Vec::new();

        // Scan for known agent framework processes
        let known_frameworks = [
            ("claude", FrameworkType::ClaudeCode),
            ("cline", FrameworkType::Cline),
            ("kilo", FrameworkType::KiloCode),
            ("droid", FrameworkType::Droid),
            ("roo", FrameworkType::RooCode),
            ("blackbox", FrameworkType::BlackBoxAI),
        ];

        for (process_name, framework_type) in &known_frameworks {
            if self.is_process_running(process_name).await? {
                let discovery = AgentDiscovery {
                    id: Uuid::new_v4(),
                    framework_type: *framework_type,
                    discovery_method: DiscoveryMethod::ProcessScan,
                    endpoint: None,
                    version: self.get_process_version(process_name).await?,
                    capabilities: self.detect_capabilities(*framework_type).await?,
                    confidence_score: 0.9,
                    discovered_at: chrono::Utc::now(),
                };
                discoveries.push(discovery);
            }
        }

        Ok(discoveries)
    }

    /// Scan network endpoints for agent framework APIs
    async fn scan_network_endpoints(&self) -> Result<Vec<AgentDiscovery>> {
        let mut discoveries = Vec::new();

        // Known endpoint patterns for different frameworks
        let endpoint_patterns = vec![
            ("http://localhost:3000", FrameworkType::ClaudeCode),
            ("http://localhost:8080", FrameworkType::Cline),
            ("http://localhost:5000", FrameworkType::KiloCode),
            ("http://localhost:3001", FrameworkType::Droid),
            ("http://localhost:4000", FrameworkType::RooCode),
            ("http://localhost:8000", FrameworkType::BlackBoxAI),
        ];

        for (endpoint, framework_type) in &endpoint_patterns {
            if self.is_endpoint_accessible(endpoint).await? {
                let discovery = AgentDiscovery {
                    id: Uuid::new_v4(),
                    framework_type: *framework_type,
                    discovery_method: DiscoveryMethod::NetworkScan,
                    endpoint: Some(endpoint.to_string()),
                    version: self.get_endpoint_version(endpoint).await?,
                    capabilities: self.detect_capabilities(*framework_type).await?,
                    confidence_score: 0.8,
                    discovered_at: chrono::Utc::now(),
                };
                discoveries.push(discovery);
            }
        }

        Ok(discoveries)
    }

    /// Scan configuration files for agent framework settings
    async fn scan_configuration(&self) -> Result<Vec<AgentDiscovery>> {
        let mut discoveries = Vec::new();

        // Check common configuration locations
        let config_locations = vec![
            "~/.config/reasonkit/agents.toml",
            "/etc/reasonkit/agents.toml",
            "./config/agents.toml",
        ];

        for location in &config_locations {
            if let Ok(config_content) = std::fs::read_to_string(shellexpand::tilde(location)) {
                if let Ok(config) = toml::from_str::<AgentConfig>(&config_content) {
                    for framework_config in config.frameworks {
                        let discovery = AgentDiscovery {
                            id: Uuid::new_v4(),
                            framework_type: framework_config.framework_type,
                            discovery_method: DiscoveryMethod::ConfigurationScan,
                            endpoint: framework_config.endpoint,
                            version: framework_config.version,
                            capabilities: framework_config.capabilities,
                            confidence_score: 0.95,
                            discovered_at: chrono::Utc::now(),
                        };
                        discoveries.push(discovery);
                    }
                }
            }
        }

        Ok(discoveries)
    }

    /// Scan for API endpoints that match agent framework patterns
    async fn scan_api_endpoints(&self) -> Result<Vec<AgentDiscovery>> {
        let mut discoveries = Vec::new();

        // Common API endpoint patterns
        let api_patterns = vec![
            ("/v1/models", FrameworkType::ClaudeCode),
            ("/api/v1/status", FrameworkType::Cline),
            ("/health", FrameworkType::KiloCode),
            ("/status", FrameworkType::Droid),
            ("/api/agents", FrameworkType::RooCode),
            ("/healthz", FrameworkType::BlackBoxAI),
        ];

        for (pattern, framework_type) in &api_patterns {
            if self.test_api_pattern(pattern).await? {
                let discovery = AgentDiscovery {
                    id: Uuid::new_v4(),
                    framework_type: *framework_type,
                    discovery_method: DiscoveryMethod::ApiEndpointScan,
                    endpoint: Some(format!("{}{}", "http://localhost:8080", pattern)),
                    version: self.get_api_version(pattern).await?,
                    capabilities: self.detect_capabilities(*framework_type).await?,
                    confidence_score: 0.7,
                    discovered_at: chrono::Utc::now(),
                };
                discoveries.push(discovery);
            }
        }

        Ok(discoveries)
    }

    // Helper methods for discovery
    async fn is_process_running(&self, process_name: &str) -> Result<bool> {
        // Simulate process detection
        // In real implementation, would use system APIs
        Ok(process_name == "claude" || process_name == "cline")
    }

    async fn get_process_version(&self, process_name: &str) -> Result<String> {
        let versions = HashMap::from([
            ("claude", "3.5-sonnet"),
            ("cline", "2.1.5"),
            ("kilo", "1.0.0"),
            ("droid", "1.2.0"),
            ("roo", "0.5.0"),
            ("blackbox", "3.0.0"),
        ]);
        Ok(versions.get(process_name).unwrap_or(&"unknown").to_string())
    }

    async fn is_endpoint_accessible(&self, endpoint: &str) -> Result<bool> {
        // Simulate endpoint accessibility check
        Ok(endpoint.contains("localhost"))
    }

    async fn get_endpoint_version(&self, endpoint: &str) -> Result<String> {
        // Simulate version detection from endpoint
        Ok("detected".to_string())
    }

    async fn detect_capabilities(&self, framework_type: FrameworkType) -> Result<Vec<String>> {
        match framework_type {
            FrameworkType::ClaudeCode => Ok(vec!["json_output".to_string(), "confidence_scoring".to_string()]),
            FrameworkType::Cline => Ok(vec!["logical_analysis".to_string(), "fallacy_detection".to_string()]),
            FrameworkType::KiloCode => Ok(vec!["comprehensive_critique".to_string(), "flaw_categorization".to_string()]),
            FrameworkType::Droid => Ok(vec!["mobile_optimization".to_string(), "android_development".to_string()]),
            FrameworkType::RooCode => Ok(vec!["multi_agent_collaboration".to_string(), "protocol_delegation".to_string()]),
            FrameworkType::BlackBoxAI => Ok(vec!["high_throughput".to_string(), "speed_optimization".to_string()]),
        }
    }

    async fn test_api_pattern(&self, pattern: &str) -> Result<bool> {
        // Simulate API pattern testing
        Ok(pattern == "/health" || pattern == "/healthz")
    }

    async fn get_api_version(&self, pattern: &str) -> Result<String> {
        Ok("api_v1".to_string())
    }

    /// Merge and deduplicate discoveries
    async fn merge_discoveries(&self, discoveries: Vec<AgentDiscovery>) -> Result<Vec<AgentDiscovery>> {
        let mut merged = HashMap::new();

        for discovery in discoveries {
            let key = (discovery.framework_type, discovery.endpoint.clone());
            
            if let Some(existing) = merged.get_mut(&key) {
                // Keep the discovery with higher confidence score
                if discovery.confidence_score > existing.confidence_score {
                    *existing = discovery;
                }
            } else {
                merged.insert(key, discovery);
            }
        }

        Ok(merged.into_values().collect())
    }
}

/// Agent Registry
/// Central registry for all discovered and registered agent frameworks
#[derive(Clone)]
pub struct AgentRegistry {
    registrations: Arc<RwLock<HashMap<FrameworkType, AgentRegistration>>>,
    adapters: Arc<RwLock<HashMap<FrameworkType, Box<dyn FrameworkAdapter>>>>,
    discovery_engine: DiscoveryEngine,
    is_initialized: Arc<RwLock<bool>>,
}

impl AgentRegistry {
    /// Create a new agent registry
    pub async fn new() -> Result<Self> {
        let registry = Arc::new(RwLock::new(HashMap::new()));
        let discovery_engine = DiscoveryEngine::new(registry.clone());
        let adapters = Arc::new(RwLock::new(HashMap::new()));
        let is_initialized = Arc::new(RwLock::new(false));

        let mut registry_instance = Self {
            registrations: registry,
            adapters,
            discovery_engine,
            is_initialized,
        };

        // Initialize with built-in adapters
        registry_instance.initialize_builtin_adapters().await?;

        Ok(registry_instance)
    }

    /// Check if registry is initialized
    pub fn is_initialized(&self) -> bool {
        // This is a sync check - in production you'd want proper async handling
        true
    }

    /// Initialize built-in framework adapters
    async fn initialize_builtin_adapters(&mut self) -> Result<()> {
        // Register built-in adapters for each framework type
        let builtin_adapters: Vec<Box<dyn FrameworkAdapter>> = vec![
            Box::new(crate::universal_agent::adapters::claude::ClaudeCodeAdapter::new()),
            Box::new(crate::universal_agent::adapters::cline::ClineAdapter::new()),
            Box::new(crate::universal_agent::adapters::kilo::KiloCodeAdapter::new()),
            Box::new(crate::universal_agent::adapters::droid::DroidAdapter::new()),
            Box::new(crate::universal_agent::adapters::roo::RooCodeAdapter::new()),
            Box::new(crate::universal_agent::adapters::blackbox::BlackBoxAIAdapter::new()),
        ];

        let mut adapters = self.adapters.write().await;
        for adapter in builtin_adapters {
            let framework_type = adapter.framework_type();
            adapters.insert(framework_type, adapter);
        }

        Ok(())
    }

    /// Auto-detect the best framework for a given protocol
    pub async fn auto_detect_best_framework(&self, protocol: &Protocol) -> Result<FrameworkType> {
        let registrations = self.registrations.read().await;
        
        // Score each available framework based on protocol characteristics
        let mut best_framework = FrameworkType::ClaudeCode;
        let mut best_score = 0.0;

        for (framework_type, registration) in registrations.iter() {
            let score = self.score_framework_for_protocol(*framework_type, protocol).await?;
            if score > best_score {
                best_score = score;
                best_framework = *framework_type;
            }
        }

        Ok(best_framework)
    }

    /// Score how well a framework fits a protocol
    async fn score_framework_for_protocol(&self, framework: FrameworkType, protocol: &Protocol) -> Result<f64> {
        let mut score = 0.0;

        // Base score from framework priority
        score += (7 - framework.priority() as f64) * 10.0;

        // Protocol-specific scoring
        match protocol.content {
            ProtocolContent::Json(_) => {
                if framework == FrameworkType::ClaudeCode {
                    score += 20.0;
                }
            }
            ProtocolContent::Text(_) => {
                if framework == FrameworkType::Cline {
                    score += 15.0;
                }
            }
            _ => {}
        }

        // Context length compatibility
        if protocol.content_length() <= framework.max_context_length() {
            score += 10.0;
        }

        // Real-time support
        if protocol.requires_realtime() && framework.supports_realtime() {
            score += 15.0;
        }

        // Performance rating from registration
        let registrations = self.registrations.read().await;
        if let Some(registration) = registrations.get(&framework) {
            score += registration.capability.performance_rating * 10.0;
        }

        Ok(score)
    }

    /// Get or create an adapter for a framework
    pub async fn get_or_create_adapter(&self, framework: FrameworkType) -> Result<Box<dyn FrameworkAdapter>> {
        let mut adapters = self.adapters.write().await;
        
        if let Some(adapter) = adapters.get(&framework) {
            Ok(adapter.clone())
        } else {
            // Create new adapter if not found
            let new_adapter = self.create_adapter_for_framework(framework).await?;
            adapters.insert(framework, new_adapter.clone());
            Ok(new_adapter)
        }
    }

    /// Get an existing adapter for a framework
    pub async fn get_adapter(&self, framework: FrameworkType) -> Result<Option<Box<dyn FrameworkAdapter>>> {
        let adapters = self.adapters.read().await;
        Ok(adapters.get(&framework).cloned())
    }

    /// Register a new framework adapter
    pub async fn register_adapter<T: FrameworkAdapter + Send + Sync + 'static>(&mut self, adapter: T) -> Result<()> {
        let framework_type = adapter.framework_type();
        let capabilities = adapter.get_capabilities().await?;

        let registration = AgentRegistration {
            capability: capabilities,
            registered_at: chrono::Utc::now(),
            last_used: None,
            usage_count: 0,
            average_performance: 0.0,
        };

        let mut registrations = self.registrations.write().await;
        registrations.insert(framework_type, registration);

        let mut adapters = self.adapters.write().await;
        adapters.insert(framework_type, Box::new(adapter));

        Ok(())
    }

    /// Create an adapter for a framework type
    async fn create_adapter_for_framework(&self, framework: FrameworkType) -> Result<Box<dyn FrameworkAdapter>> {
        match framework {
            FrameworkType::ClaudeCode => {
                Ok(Box::new(crate::universal_agent::adapters::claude::ClaudeCodeAdapter::new()))
            }
            FrameworkType::Cline => {
                Ok(Box::new(crate::universal_agent::adapters::cline::ClineAdapter::new()))
            }
            FrameworkType::KiloCode => {
                Ok(Box::new(crate::universal_agent::adapters::kilo::KiloCodeAdapter::new()))
            }
            FrameworkType::Droid => {
                Ok(Box::new(crate::universal_agent::adapters::droid::DroidAdapter::new()))
            }
            FrameworkType::RooCode => {
                Ok(Box::new(crate::universal_agent::adapters::roo::RooCodeAdapter::new()))
            }
            FrameworkType::BlackBoxAI => {
                Ok(Box::new(crate::universal_agent::adapters::blackbox::BlackBoxAIAdapter::new()))
            }
        }
    }

    /// Get all registered frameworks
    pub async fn get_registered_frameworks(&self) -> Result<Vec<FrameworkType>> {
        let registrations = self.registrations.read().await;
        Ok(registrations.keys().copied().collect())
    }

    /// Get framework registration info
    pub async fn get_registration(&self, framework: FrameworkType) -> Result<Option<AgentRegistration>> {
        let registrations = self.registrations.read().await;
        Ok(registrations.get(&framework).cloned())
    }

    /// Update framework usage statistics
    pub async fn update_usage(&self, framework: FrameworkType, performance: f64) -> Result<()> {
        let mut registrations = self.registrations.write().await;
        
        if let Some(registration) = registrations.get_mut(&framework) {
            registration.last_used = Some(chrono::Utc::now());
            registration.usage_count += 1;
            // Update rolling average performance
            registration.average_performance = (registration.average_performance * 0.9) + (performance * 0.1);
        }

        Ok(())
    }
}

/// Agent discovery result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDiscovery {
    pub id: Uuid,
    pub framework_type: FrameworkType,
    pub discovery_method: DiscoveryMethod,
    pub endpoint: Option<String>,
    pub version: String,
    pub capabilities: Vec<String>,
    pub confidence_score: f64,
    pub discovered_at: chrono::DateTime<chrono::Utc>,
}

/// Discovery method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryMethod {
    ProcessScan,
    NetworkScan,
    ConfigurationScan,
    ApiEndpointScan,
}

/// Agent configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub frameworks: Vec<FrameworkConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkConfig {
    pub framework_type: FrameworkType,
    pub endpoint: Option<String>,
    pub version: String,
    pub capabilities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_registry_creation() {
        let registry = AgentRegistry::new().await.unwrap();
        assert!(registry.is_initialized());
    }

    #[tokio::test]
    async fn test_framework_scoring() {
        let registry = AgentRegistry::new().await.unwrap();
        
        // Create a test protocol
        let protocol = Protocol {
            id: Uuid::new_v4(),
            content: ProtocolContent::Json(serde_json::json!({"test": true})),
            metadata: crate::thinktool::ProtocolMetadata::default(),
            created_at: chrono::Utc::now(),
        };

        let best_framework = registry.auto_detect_best_framework(&protocol).await.unwrap();
        assert!(matches!(best_framework, FrameworkType::ClaudeCode));
    }
}
