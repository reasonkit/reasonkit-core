# ReasonKit Plugin Architecture

> Comprehensive Extension System for Custom ThinkTools, Integrations, and Protocols
> Version: 1.0.0 | Status: Design Specification | License: Apache 2.0

---

## Table of Contents

1. [Overview](#1-overview)
2. [Plugin Types](#2-plugin-types)
3. [Core Architecture](#3-core-architecture)
4. [Rust Plugin System](#4-rust-plugin-system)
5. [Python Plugin System](#5-python-plugin-system)
6. [Plugin Manifest Specification](#6-plugin-manifest-specification)
7. [Plugin Discovery](#7-plugin-discovery)
8. [Plugin API](#8-plugin-api)
9. [Event Hooks](#9-event-hooks)
10. [Security Model](#10-security-model)
11. [Plugin Development SDK](#11-plugin-development-sdk)
12. [Plugin Distribution](#12-plugin-distribution)
13. [Plugin Marketplace](#13-plugin-marketplace)
14. [Testing and Quality](#14-testing-and-quality)
15. [Example Plugins](#15-example-plugins)
16. [Migration Guide](#16-migration-guide)

---

## 1. Overview

### 1.1 Purpose

The ReasonKit Plugin Architecture enables developers to extend the core reasoning framework with:

- **Custom ThinkTools**: Domain-specific reasoning modules
- **Integration Plugins**: LLM providers, storage backends, output formatters
- **Protocol Plugins**: Workflow templates, industry-specific reasoning chains

### 1.2 Design Principles

```
+------------------------------------------------------------------+
|                    PLUGIN ARCHITECTURE PRINCIPLES                 |
+------------------------------------------------------------------+
|                                                                  |
|  1. RUST-FIRST: Native plugins are Rust for performance         |
|  2. SAFETY: Sandboxed execution with capability-based security  |
|  3. COMPOSABLE: Plugins can depend on and extend other plugins  |
|  4. DISCOVERABLE: Auto-discovery from standard locations        |
|  5. VERSIONED: Semantic versioning with compatibility checks    |
|  6. OBSERVABLE: Full tracing and metrics for plugin execution   |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.3 Architecture Overview

```
+------------------------------------------------------------------+
|                     REASONKIT PLUGIN SYSTEM                       |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------------------------------------------------+  |
|  |                    PLUGIN MANAGER                          |  |
|  |  +----------+  +-----------+  +-----------+  +---------+  |  |
|  |  | Discovery|  | Registry  |  | Lifecycle |  | Sandbox |  |  |
|  |  +----------+  +-----------+  +-----------+  +---------+  |  |
|  +-----------------------------------------------------------+  |
|                              |                                   |
|        +--------------------+--------------------+               |
|        |                    |                    |               |
|  +-----v------+      +------v-----+      +------v------+        |
|  | THINKTOOL  |      | INTEGRATION|      | PROTOCOL    |        |
|  | PLUGINS    |      | PLUGINS    |      | PLUGINS     |        |
|  +------------+      +------------+      +-------------+        |
|  | - GigaThink|      | - LLM      |      | - Workflows |        |
|  | - Custom   |      | - Storage  |      | - Templates |        |
|  | - Domain   |      | - Format   |      | - Chains    |        |
|  +------------+      +------------+      +-------------+        |
|                                                                  |
|  +-----------------------------------------------------------+  |
|  |                    CORE APIs                               |  |
|  |  +--------+  +--------+  +--------+  +--------+  +------+ |  |
|  |  | Config |  |  LLM   |  | Storage|  | Trace  |  | Hook | |  |
|  +-----------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. Plugin Types

### 2.1 ThinkTool Plugins

Custom reasoning modules that extend the core ThinkTools (GigaThink, LaserLogic, BedRock, ProofGuard, BrutalHonesty).

| Category               | Examples                                    | Use Case             |
| ---------------------- | ------------------------------------------- | -------------------- |
| **Domain-Specific**    | LegalLogic, MedicalReasoner, FinanceAnalyst | Industry expertise   |
| **Cognitive Patterns** | SocraticDialogue, DevilsAdvocate, RedTeam   | Reasoning strategies |
| **Analysis Tools**     | CodeReview, SecurityAudit, RiskAssessment   | Technical analysis   |
| **Creative Tools**     | IdeaGenerator, StoryWeaver, DesignThinking  | Creative processes   |

### 2.2 Integration Plugins

Connect ReasonKit to external systems and services.

| Category              | Examples                                    | Purpose           |
| --------------------- | ------------------------------------------- | ----------------- |
| **LLM Providers**     | LocalLlama, OllamaAdapter, CustomEndpoint   | Model integration |
| **Storage Backends**  | PostgresStore, MongoStore, S3Archive        | Persistence       |
| **Output Formatters** | PDFExporter, JiraIntegration, SlackNotifier | Delivery          |
| **Data Sources**      | ArxivFetcher, GitHubReader, ConfluenceSync  | Ingestion         |

### 2.3 Protocol Plugins

Pre-defined reasoning workflows and chains.

| Category               | Examples                                      | Purpose       |
| ---------------------- | --------------------------------------------- | ------------- |
| **Industry Protocols** | SOC2Compliance, FDAReview, LegalDiscovery     | Compliance    |
| **Workflow Templates** | DesignReview, IncidentResponse, RootCause     | Processes     |
| **Chain Compositions** | DeepResearch, QuickDecision, ConsensusBuilder | Orchestration |

---

## 3. Core Architecture

### 3.1 Plugin Manager

The central orchestrator for all plugin operations.

```rust
//! Plugin Manager - Core orchestration component
//!
//! Responsibilities:
//! - Plugin discovery and loading
//! - Lifecycle management (init, start, stop, unload)
//! - Dependency resolution
//! - Security enforcement
//! - Resource management

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::Result;

/// Plugin manager configuration
#[derive(Debug, Clone)]
pub struct PluginManagerConfig {
    /// Paths to search for plugins
    pub search_paths: Vec<PathBuf>,

    /// Enable auto-discovery on startup
    pub auto_discover: bool,

    /// Enable hot-reload for development
    pub hot_reload: bool,

    /// Maximum plugins to load
    pub max_plugins: usize,

    /// Default trust level for new plugins
    pub default_trust: TrustLevel,

    /// Resource limits per plugin
    pub resource_limits: ResourceLimits,
}

impl Default for PluginManagerConfig {
    fn default() -> Self {
        Self {
            search_paths: vec![
                PathBuf::from("~/.reasonkit/plugins"),
                PathBuf::from("./plugins"),
            ],
            auto_discover: true,
            hot_reload: cfg!(debug_assertions),
            max_plugins: 100,
            default_trust: TrustLevel::Community,
            resource_limits: ResourceLimits::default(),
        }
    }
}

/// Trust levels for plugins
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    /// Official ReasonKit plugins - full access
    Official = 3,
    /// Verified by ReasonKit team - reviewed access
    Verified = 2,
    /// Community plugins - sandboxed
    Community = 1,
    /// Untrusted - maximum restrictions
    Untrusted = 0,
}

/// Resource limits for plugin execution
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum memory per plugin (bytes)
    pub max_memory: usize,
    /// Maximum CPU time per operation (ms)
    pub max_cpu_time_ms: u64,
    /// Maximum network requests per minute
    pub max_requests_per_minute: u32,
    /// Maximum file system operations per minute
    pub max_fs_ops_per_minute: u32,
    /// Maximum LLM tokens per operation
    pub max_llm_tokens: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 256 * 1024 * 1024, // 256 MB
            max_cpu_time_ms: 30_000,        // 30 seconds
            max_requests_per_minute: 60,
            max_fs_ops_per_minute: 100,
            max_llm_tokens: 10_000,
        }
    }
}

/// Plugin manager state
pub struct PluginManager {
    config: PluginManagerConfig,
    registry: PluginRegistry,
    loader: PluginLoader,
    sandbox: PluginSandbox,
    hooks: HookRegistry,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new(config: PluginManagerConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            registry: PluginRegistry::new(),
            loader: PluginLoader::new(&config),
            sandbox: PluginSandbox::new(&config.resource_limits),
            hooks: HookRegistry::new(),
        })
    }

    /// Discover and load all plugins from search paths
    pub async fn discover_all(&mut self) -> Result<DiscoveryReport> {
        let mut report = DiscoveryReport::new();

        for path in &self.config.search_paths {
            let discovered = self.loader.scan_directory(path).await?;

            for manifest in discovered {
                match self.load_plugin(&manifest).await {
                    Ok(id) => report.loaded.push(id),
                    Err(e) => report.failed.push((manifest.id.clone(), e)),
                }
            }
        }

        Ok(report)
    }

    /// Load a specific plugin
    pub async fn load_plugin(&mut self, manifest: &PluginManifest) -> Result<PluginId> {
        // Validate manifest
        manifest.validate()?;

        // Check dependencies
        self.resolve_dependencies(manifest)?;

        // Determine trust level
        let trust = self.evaluate_trust(manifest)?;

        // Load plugin with appropriate sandbox
        let plugin = self.loader.load(manifest, trust).await?;

        // Register plugin
        let id = self.registry.register(plugin)?;

        // Initialize plugin
        self.initialize_plugin(&id).await?;

        Ok(id)
    }

    /// Unload a plugin
    pub async fn unload_plugin(&mut self, id: &PluginId) -> Result<()> {
        // Call plugin cleanup
        self.cleanup_plugin(id).await?;

        // Remove from registry
        self.registry.unregister(id)?;

        Ok(())
    }

    /// Get a plugin by ID
    pub fn get(&self, id: &PluginId) -> Option<&dyn Plugin> {
        self.registry.get(id)
    }

    /// List all loaded plugins
    pub fn list(&self) -> Vec<PluginInfo> {
        self.registry.list()
    }

    /// Execute a hook across all plugins
    pub async fn execute_hook<T: HookPayload>(&self, hook: HookType, payload: T) -> Result<Vec<HookResult>> {
        self.hooks.execute(hook, payload, &self.registry).await
    }
}
```

### 3.2 Plugin Registry

Maintains the catalog of loaded plugins.

```rust
/// Plugin registry - tracks all loaded plugins
pub struct PluginRegistry {
    plugins: HashMap<PluginId, Box<dyn Plugin>>,
    metadata: HashMap<PluginId, PluginMetadata>,
    dependencies: DependencyGraph,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            metadata: HashMap::new(),
            dependencies: DependencyGraph::new(),
        }
    }

    /// Register a plugin
    pub fn register(&mut self, plugin: Box<dyn Plugin>) -> Result<PluginId> {
        let id = plugin.id().clone();
        let metadata = plugin.metadata().clone();

        // Check for conflicts
        if self.plugins.contains_key(&id) {
            return Err(Error::PluginConflict { id });
        }

        // Update dependency graph
        self.dependencies.add_node(&id, &metadata.dependencies)?;

        self.plugins.insert(id.clone(), plugin);
        self.metadata.insert(id.clone(), metadata);

        Ok(id)
    }

    /// Get a plugin by ID
    pub fn get(&self, id: &PluginId) -> Option<&dyn Plugin> {
        self.plugins.get(id).map(|p| p.as_ref())
    }

    /// Get mutable plugin reference
    pub fn get_mut(&mut self, id: &PluginId) -> Option<&mut Box<dyn Plugin>> {
        self.plugins.get_mut(id)
    }

    /// List all plugins with metadata
    pub fn list(&self) -> Vec<PluginInfo> {
        self.metadata
            .iter()
            .map(|(id, meta)| PluginInfo {
                id: id.clone(),
                name: meta.name.clone(),
                version: meta.version.clone(),
                plugin_type: meta.plugin_type,
                status: self.get_status(id),
            })
            .collect()
    }

    /// Get plugins by type
    pub fn get_by_type(&self, plugin_type: PluginType) -> Vec<&dyn Plugin> {
        self.plugins
            .values()
            .filter(|p| p.metadata().plugin_type == plugin_type)
            .map(|p| p.as_ref())
            .collect()
    }
}
```

---

## 4. Rust Plugin System

### 4.1 Plugin Trait Definition

The core trait that all Rust plugins must implement.

```rust
//! Core plugin trait definitions
//!
//! All ReasonKit plugins implement these traits for lifecycle management,
//! capability declaration, and execution.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;

/// Unique plugin identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PluginId(pub String);

impl PluginId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Plugin type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginType {
    /// Custom ThinkTool (reasoning module)
    ThinkTool,
    /// LLM provider adapter
    LlmProvider,
    /// Storage backend
    Storage,
    /// Output formatter
    Formatter,
    /// Data source connector
    DataSource,
    /// Protocol/workflow definition
    Protocol,
    /// Utility/helper plugin
    Utility,
}

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,
    /// Semantic version
    pub version: String,
    /// Brief description
    pub description: String,
    /// Plugin type
    pub plugin_type: PluginType,
    /// Author information
    pub author: String,
    /// License identifier
    pub license: String,
    /// Repository URL
    pub repository: Option<String>,
    /// Required ReasonKit version
    pub reasonkit_version: String,
    /// Plugin dependencies
    pub dependencies: Vec<PluginDependency>,
    /// Required capabilities
    pub capabilities: Vec<Capability>,
    /// Plugin tags for discovery
    pub tags: Vec<String>,
}

/// Plugin dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    /// Dependency plugin ID
    pub id: String,
    /// Version requirement (semver)
    pub version: String,
    /// Whether optional
    pub optional: bool,
}

/// Capability requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Network access (HTTP requests)
    Network,
    /// File system read access
    FileSystemRead,
    /// File system write access
    FileSystemWrite,
    /// LLM API access
    LlmAccess,
    /// Configuration access
    ConfigAccess,
    /// Secrets/credentials access
    SecretsAccess,
    /// Execute external processes
    ProcessExec,
    /// Access environment variables
    EnvAccess,
}

/// Core plugin trait
#[async_trait]
pub trait Plugin: Send + Sync + 'static {
    /// Get plugin ID
    fn id(&self) -> &PluginId;

    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Initialize plugin (called once after loading)
    async fn initialize(&mut self, ctx: &PluginContext) -> Result<()>;

    /// Start plugin (called when plugin becomes active)
    async fn start(&mut self, ctx: &PluginContext) -> Result<()>;

    /// Stop plugin (called before unloading)
    async fn stop(&mut self, ctx: &PluginContext) -> Result<()>;

    /// Cleanup resources (called during unload)
    async fn cleanup(&mut self) -> Result<()>;

    /// Health check
    fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }

    /// Cast to Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Cast to mutable Any
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Plugin execution context
pub struct PluginContext {
    /// Configuration access
    pub config: Arc<dyn ConfigAccess>,
    /// LLM client (if capability granted)
    pub llm: Option<Arc<dyn LlmAccess>>,
    /// Storage access (if capability granted)
    pub storage: Option<Arc<dyn StorageAccess>>,
    /// Logging/tracing
    pub tracer: Arc<dyn PluginTracer>,
    /// Event emitter for hooks
    pub events: Arc<dyn EventEmitter>,
    /// Resource tracker
    pub resources: Arc<ResourceTracker>,
}

/// Health status for plugins
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}
```

### 4.2 ThinkTool Plugin Trait

Specialized trait for reasoning module plugins.

```rust
//! ThinkTool plugin trait
//!
//! Extends the base Plugin trait with reasoning-specific methods.

use super::{Plugin, PluginContext};
use crate::thinktool::{Protocol, ProtocolInput, ProtocolOutput, StepResult};

/// ThinkTool plugin trait
#[async_trait]
pub trait ThinkToolPlugin: Plugin {
    /// Get the protocol definition for this ThinkTool
    fn protocol(&self) -> &Protocol;

    /// Execute the ThinkTool
    async fn execute(
        &self,
        input: ProtocolInput,
        ctx: &PluginContext,
    ) -> Result<ProtocolOutput>;

    /// Execute a single step (for custom step implementations)
    async fn execute_step(
        &self,
        step_id: &str,
        input: &serde_json::Value,
        ctx: &PluginContext,
    ) -> Result<StepResult> {
        // Default implementation uses LLM
        Err(Error::NotImplemented)
    }

    /// Validate input before execution
    fn validate_input(&self, input: &ProtocolInput) -> Result<()> {
        // Default validation using protocol spec
        let protocol = self.protocol();
        for required in &protocol.input.required {
            if !input.fields.contains_key(required) {
                return Err(Error::Validation(format!(
                    "Missing required field: {}", required
                )));
            }
        }
        Ok(())
    }

    /// Get composable protocols (protocols this can chain with)
    fn composable_with(&self) -> &[String] {
        &self.protocol().metadata.composable_with
    }
}

/// Macro to simplify ThinkTool plugin creation
#[macro_export]
macro_rules! thinktool_plugin {
    (
        $name:ident,
        id = $id:expr,
        name = $display_name:expr,
        version = $version:expr,
        strategy = $strategy:expr,
        description = $desc:expr,
        steps = [$($step:expr),* $(,)?]
    ) => {
        pub struct $name {
            id: PluginId,
            metadata: PluginMetadata,
            protocol: Protocol,
        }

        impl $name {
            pub fn new() -> Self {
                let protocol = Protocol {
                    id: $id.to_string(),
                    name: $display_name.to_string(),
                    version: $version.to_string(),
                    description: $desc.to_string(),
                    strategy: $strategy,
                    steps: vec![$($step),*],
                    ..Default::default()
                };

                Self {
                    id: PluginId::new($id),
                    metadata: PluginMetadata {
                        name: $display_name.to_string(),
                        version: $version.to_string(),
                        description: $desc.to_string(),
                        plugin_type: PluginType::ThinkTool,
                        ..Default::default()
                    },
                    protocol,
                }
            }
        }

        #[async_trait]
        impl Plugin for $name {
            fn id(&self) -> &PluginId { &self.id }
            fn metadata(&self) -> &PluginMetadata { &self.metadata }

            async fn initialize(&mut self, _ctx: &PluginContext) -> Result<()> { Ok(()) }
            async fn start(&mut self, _ctx: &PluginContext) -> Result<()> { Ok(()) }
            async fn stop(&mut self, _ctx: &PluginContext) -> Result<()> { Ok(()) }
            async fn cleanup(&mut self) -> Result<()> { Ok(()) }

            fn as_any(&self) -> &dyn Any { self }
            fn as_any_mut(&mut self) -> &mut dyn Any { self }
        }

        #[async_trait]
        impl ThinkToolPlugin for $name {
            fn protocol(&self) -> &Protocol { &self.protocol }

            async fn execute(
                &self,
                input: ProtocolInput,
                ctx: &PluginContext,
            ) -> Result<ProtocolOutput> {
                // Default execution through context LLM
                let llm = ctx.llm.as_ref()
                    .ok_or(Error::CapabilityDenied(Capability::LlmAccess))?;

                // Execute protocol steps...
                todo!("Implement execution logic")
            }
        }
    };
}
```

### 4.3 Integration Plugin Traits

Traits for LLM providers, storage, and formatters.

```rust
//! Integration plugin traits
//!
//! Specialized traits for different integration types.

/// LLM Provider plugin trait
#[async_trait]
pub trait LlmProviderPlugin: Plugin {
    /// Get provider identifier
    fn provider_id(&self) -> &str;

    /// Get supported models
    fn supported_models(&self) -> &[String];

    /// Create LLM client
    fn create_client(&self, config: &LlmConfig) -> Result<Box<dyn LlmClient>>;

    /// Check if provider is available (API key set, etc.)
    fn is_available(&self) -> bool;

    /// Get pricing information
    fn pricing(&self, model: &str) -> Option<ModelPricing>;
}

/// Storage backend plugin trait
#[async_trait]
pub trait StoragePlugin: Plugin {
    /// Get storage type identifier
    fn storage_type(&self) -> &str;

    /// Create storage client
    async fn connect(&self, config: &StorageConfig) -> Result<Box<dyn StorageClient>>;

    /// Check connection health
    async fn health_check(&self) -> Result<StorageHealth>;

    /// Get storage capabilities
    fn capabilities(&self) -> StorageCapabilities;
}

/// Output formatter plugin trait
#[async_trait]
pub trait FormatterPlugin: Plugin {
    /// Get output format identifier
    fn format_id(&self) -> &str;

    /// Get MIME type
    fn mime_type(&self) -> &str;

    /// Format protocol output
    async fn format(&self, output: &ProtocolOutput, options: &FormatOptions) -> Result<Vec<u8>>;

    /// Get supported templates
    fn templates(&self) -> &[FormatTemplate];
}

/// Data source plugin trait
#[async_trait]
pub trait DataSourcePlugin: Plugin {
    /// Get source type identifier
    fn source_type(&self) -> &str;

    /// Connect to data source
    async fn connect(&self, config: &DataSourceConfig) -> Result<Box<dyn DataSourceClient>>;

    /// List available resources
    async fn list_resources(&self, filter: Option<&str>) -> Result<Vec<ResourceInfo>>;

    /// Fetch a resource
    async fn fetch(&self, resource_id: &str) -> Result<Resource>;
}
```

---

## 5. Python Plugin System

### 5.1 Python Plugin Base Class

Python bindings for plugin development via PyO3.

```python
"""
ReasonKit Python Plugin System

Provides Python bindings for developing ReasonKit plugins.
Plugins are loaded via PyO3 and executed in a sandboxed environment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class PluginType(Enum):
    """Plugin type classification"""
    THINKTOOL = "thinktool"
    LLM_PROVIDER = "llm_provider"
    STORAGE = "storage"
    FORMATTER = "formatter"
    DATA_SOURCE = "data_source"
    PROTOCOL = "protocol"
    UTILITY = "utility"


class Capability(Enum):
    """Required capabilities"""
    NETWORK = "network"
    FILE_SYSTEM_READ = "file_system_read"
    FILE_SYSTEM_WRITE = "file_system_write"
    LLM_ACCESS = "llm_access"
    CONFIG_ACCESS = "config_access"
    SECRETS_ACCESS = "secrets_access"
    PROCESS_EXEC = "process_exec"
    ENV_ACCESS = "env_access"


@dataclass
class PluginMetadata:
    """Plugin metadata specification"""
    name: str
    version: str
    description: str
    plugin_type: PluginType
    author: str = ""
    license: str = "Apache-2.0"
    repository: Optional[str] = None
    reasonkit_version: str = ">=1.0.0"
    dependencies: List[Dict[str, str]] = field(default_factory=list)
    capabilities: List[Capability] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class PluginContext:
    """Execution context provided to plugins"""
    config: Dict[str, Any]
    llm: Optional[Any] = None  # LlmAccess interface
    storage: Optional[Any] = None  # StorageAccess interface
    tracer: Optional[Any] = None  # PluginTracer interface
    events: Optional[Any] = None  # EventEmitter interface


class Plugin(ABC):
    """
    Base class for all ReasonKit Python plugins.

    Implement this class to create custom plugins.

    Example:
        class MyThinkTool(Plugin):
            @property
            def id(self) -> str:
                return "my-thinktool"

            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="My ThinkTool",
                    version="1.0.0",
                    description="A custom reasoning tool",
                    plugin_type=PluginType.THINKTOOL,
                    capabilities=[Capability.LLM_ACCESS],
                )

            async def execute(self, input: Dict[str, Any], ctx: PluginContext) -> Dict[str, Any]:
                # Your logic here
                pass
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique plugin identifier"""
        pass

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        pass

    async def initialize(self, ctx: PluginContext) -> None:
        """Initialize plugin (called once after loading)"""
        pass

    async def start(self, ctx: PluginContext) -> None:
        """Start plugin (called when plugin becomes active)"""
        pass

    async def stop(self, ctx: PluginContext) -> None:
        """Stop plugin (called before unloading)"""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources (called during unload)"""
        pass

    def health_check(self) -> str:
        """Health check - returns 'healthy', 'degraded', or 'unhealthy'"""
        return "healthy"


class ThinkToolPlugin(Plugin):
    """
    Base class for ThinkTool plugins.

    Provides additional methods for protocol-based reasoning.
    """

    @property
    @abstractmethod
    def protocol(self) -> Dict[str, Any]:
        """
        Protocol definition for this ThinkTool.

        Returns a dict matching the Protocol schema:
        {
            "id": "my-tool",
            "name": "My Tool",
            "version": "1.0.0",
            "description": "...",
            "strategy": "expansive",
            "steps": [...],
            ...
        }
        """
        pass

    @abstractmethod
    async def execute(
        self,
        input: Dict[str, Any],
        ctx: PluginContext
    ) -> Dict[str, Any]:
        """
        Execute the ThinkTool.

        Args:
            input: Protocol input fields
            ctx: Plugin execution context

        Returns:
            Protocol output with steps, confidence, etc.
        """
        pass

    async def execute_step(
        self,
        step_id: str,
        input: Dict[str, Any],
        ctx: PluginContext
    ) -> Dict[str, Any]:
        """
        Execute a single protocol step.

        Override for custom step implementations.
        """
        raise NotImplementedError("Custom step execution not implemented")

    def validate_input(self, input: Dict[str, Any]) -> None:
        """Validate input against protocol requirements"""
        protocol = self.protocol
        required = protocol.get("input", {}).get("required", [])

        for field in required:
            if field not in input:
                raise ValueError(f"Missing required field: {field}")

    @property
    def composable_with(self) -> List[str]:
        """Protocols this ThinkTool can chain with"""
        return self.protocol.get("metadata", {}).get("composable_with", [])


class LlmProviderPlugin(Plugin):
    """Base class for LLM provider plugins"""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Provider identifier"""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of supported model identifiers"""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete a chat conversation.

        Args:
            messages: List of {"role": "...", "content": "..."} messages
            model: Model identifier
            **kwargs: Additional provider-specific options

        Returns:
            Response dict with "content", "usage", etc.
        """
        pass

    def is_available(self) -> bool:
        """Check if provider is available (API key configured, etc.)"""
        return True

    def pricing(self, model: str) -> Optional[Dict[str, float]]:
        """
        Get pricing for a model.

        Returns dict with "input_per_1m" and "output_per_1m" (USD per 1M tokens).
        """
        return None
```

### 5.2 Python Plugin Discovery

```python
"""
Plugin discovery mechanism for Python plugins.

Searches standard locations and loads plugins dynamically.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List, Type

from .base import Plugin


def discover_plugins(
    search_paths: List[Path] = None,
    include_installed: bool = True
) -> List[Type[Plugin]]:
    """
    Discover all available plugins.

    Args:
        search_paths: Additional paths to search
        include_installed: Include pip-installed plugins

    Returns:
        List of Plugin classes
    """
    plugins = []

    # Default search paths
    paths = [
        Path.home() / ".reasonkit" / "plugins" / "python",
        Path.cwd() / "plugins",
    ]

    if search_paths:
        paths.extend(search_paths)

    # Search file-based plugins
    for path in paths:
        if path.exists():
            plugins.extend(_discover_in_directory(path))

    # Search installed packages
    if include_installed:
        plugins.extend(_discover_installed_plugins())

    return plugins


def _discover_in_directory(directory: Path) -> List[Type[Plugin]]:
    """Discover plugins in a directory"""
    plugins = []

    for py_file in directory.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                py_file.stem,
                py_file
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[py_file.stem] = module
            spec.loader.exec_module(module)

            # Find Plugin subclasses
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Plugin)
                    and obj is not Plugin
                ):
                    plugins.append(obj)

        except Exception as e:
            print(f"Failed to load plugin from {py_file}: {e}")

    return plugins


def _discover_installed_plugins() -> List[Type[Plugin]]:
    """Discover plugins installed via pip"""
    plugins = []

    # Look for entry points in reasonkit.plugins group
    try:
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points
            eps = entry_points(group="reasonkit.plugins")
        else:
            from importlib.metadata import entry_points
            eps = entry_points().get("reasonkit.plugins", [])

        for ep in eps:
            try:
                plugin_class = ep.load()
                if issubclass(plugin_class, Plugin):
                    plugins.append(plugin_class)
            except Exception as e:
                print(f"Failed to load plugin {ep.name}: {e}")

    except ImportError:
        pass

    return plugins


def load_plugin(path: Path) -> Type[Plugin]:
    """
    Load a single plugin from a file or directory.

    Args:
        path: Path to plugin file or directory with __init__.py

    Returns:
        Plugin class
    """
    if path.is_dir():
        path = path / "__init__.py"

    if not path.exists():
        raise FileNotFoundError(f"Plugin not found: {path}")

    spec = importlib.util.spec_from_file_location("plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the main Plugin class
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Plugin)
            and obj is not Plugin
        ):
            return obj

    raise ValueError(f"No Plugin class found in {path}")
```

---

## 6. Plugin Manifest Specification

### 6.1 Manifest Format (TOML)

```toml
# plugin.toml - Plugin Manifest Specification
# This file must be present in the root of every plugin package

[plugin]
# Required: Unique plugin identifier (lowercase, hyphens allowed)
id = "my-custom-thinktool"

# Required: Display name
name = "My Custom ThinkTool"

# Required: Semantic version
version = "1.0.0"

# Required: Plugin type
# Options: thinktool, llm_provider, storage, formatter, data_source, protocol, utility
type = "thinktool"

# Required: Brief description
description = "A custom reasoning tool for domain-specific analysis"

# Required: Author information
author = "Your Name <email@example.com>"

# Required: License identifier (SPDX)
license = "Apache-2.0"

# Optional: Repository URL
repository = "https://github.com/user/my-plugin"

# Optional: Homepage URL
homepage = "https://my-plugin.example.com"

# Optional: Documentation URL
documentation = "https://my-plugin.example.com/docs"

[plugin.reasonkit]
# Required: Minimum ReasonKit version
min_version = "1.0.0"

# Optional: Maximum compatible version
max_version = "1.0.0"

# Optional: Tested ReasonKit versions
tested_versions = ["1.0.0", "1.1.0"]

[plugin.capabilities]
# List required capabilities
# Each capability expands permissions in the sandbox

# Network access for HTTP requests
network = true

# File system read access
file_system_read = false

# File system write access
file_system_write = false

# LLM API access (required for most ThinkTools)
llm_access = true

# Configuration access
config_access = true

# Secrets/credentials access
secrets_access = false

# Execute external processes
process_exec = false

# Environment variable access
env_access = false

[plugin.dependencies]
# Dependencies on other plugins
# Format: "plugin-id" = "version requirement"

# Example: Depends on core proofguard plugin
# proofguard = ">=1.0.0"

# Example: Optional dependency
# my-extension = { version = ">=0.5.0", optional = true }

[plugin.rust]
# Rust-specific configuration (for Rust plugins)

# Entry point module
entry_point = "src/lib.rs"

# Crate name (if different from plugin id)
crate_name = "my_custom_thinktool"

# Feature flags to enable
features = []

[plugin.python]
# Python-specific configuration (for Python plugins)

# Entry point module
entry_point = "my_plugin/__init__.py"

# Main class name
class_name = "MyCustomThinkTool"

# Python version requirement
python_version = ">=3.9"

# pip dependencies
dependencies = [
    "httpx>=0.24.0",
    "pydantic>=2.0.0",
]

[plugin.metadata]
# Additional metadata for discovery and categorization

# Category for UI grouping
category = "domain-specific"

# Tags for search
tags = ["legal", "compliance", "analysis"]

# Keywords (aliases for search)
keywords = ["law", "regulatory"]

# Icon (path or URL)
icon = "assets/icon.svg"

# Screenshots for marketplace
screenshots = [
    "assets/screenshot1.png",
    "assets/screenshot2.png",
]

[plugin.config]
# Plugin configuration schema
# Used for validation and UI generation

[plugin.config.schema]
# JSON Schema for configuration
type = "object"
properties.api_key = { type = "string", description = "API key for external service" }
properties.model = { type = "string", default = "gpt-4" }
properties.temperature = { type = "number", minimum = 0, maximum = 2, default = 0.7 }
required = ["api_key"]

[plugin.config.env_vars]
# Environment variable mappings
# Format: config_key = "ENV_VAR_NAME"
api_key = "MY_PLUGIN_API_KEY"

[build]
# Build configuration

# Build command
command = "cargo build --release"

# Output artifact
artifact = "target/release/libmy_plugin.so"

# Pre-build hooks
pre_build = ["./scripts/generate-bindings.sh"]

# Post-build hooks
post_build = ["./scripts/validate-artifact.sh"]

[test]
# Test configuration

# Test command
command = "cargo test"

# Integration test command
integration = "cargo test --features integration"

# Benchmark command
benchmark = "cargo bench"
```

### 6.2 Manifest Validation

```rust
/// Plugin manifest validation
pub struct ManifestValidator;

impl ManifestValidator {
    /// Validate a manifest
    pub fn validate(manifest: &PluginManifest) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Required fields
        if manifest.id.is_empty() {
            errors.push(ValidationError::MissingField("id".into()));
        }

        if manifest.version.is_empty() {
            errors.push(ValidationError::MissingField("version".into()));
        }

        // Version format (semver)
        if !semver::Version::parse(&manifest.version).is_ok() {
            errors.push(ValidationError::InvalidFormat {
                field: "version".into(),
                expected: "semver (e.g., 1.0.0)".into(),
            });
        }

        // ID format (lowercase, alphanumeric, hyphens)
        let id_regex = regex::Regex::new(r"^[a-z][a-z0-9-]*$").unwrap();
        if !id_regex.is_match(&manifest.id) {
            errors.push(ValidationError::InvalidFormat {
                field: "id".into(),
                expected: "lowercase alphanumeric with hyphens".into(),
            });
        }

        // ReasonKit version compatibility
        if let Some(min) = &manifest.reasonkit.min_version {
            if !semver::Version::parse(min).is_ok() {
                errors.push(ValidationError::InvalidFormat {
                    field: "reasonkit.min_version".into(),
                    expected: "semver".into(),
                });
            }
        }

        // Capability validation
        for cap in &manifest.capabilities {
            if !Self::is_valid_capability(cap) {
                errors.push(ValidationError::UnknownCapability(cap.clone()));
            }
        }

        // Dependency validation
        for dep in &manifest.dependencies {
            if !semver::VersionReq::parse(&dep.version).is_ok() {
                errors.push(ValidationError::InvalidFormat {
                    field: format!("dependencies.{}", dep.id),
                    expected: "semver requirement (e.g., >=1.0.0)".into(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn is_valid_capability(cap: &str) -> bool {
        matches!(
            cap,
            "network" |
            "file_system_read" |
            "file_system_write" |
            "llm_access" |
            "config_access" |
            "secrets_access" |
            "process_exec" |
            "env_access"
        )
    }
}
```

---

## 7. Plugin Discovery

### 7.1 Discovery Locations

```
Plugin Search Order (First Found Wins):
+---------------------------------------------------------+
|                                                         |
|  1. Explicit Registration (--plugin flag)               |
|     rk-core --plugin /path/to/plugin.so                 |
|                                                         |
|  2. Project-Local Plugins                               |
|     ./plugins/                                          |
|     ./.reasonkit/plugins/                               |
|                                                         |
|  3. User Plugins                                        |
|     ~/.reasonkit/plugins/                               |
|     ~/.config/reasonkit/plugins/                        |
|                                                         |
|  4. System Plugins                                      |
|     /usr/local/share/reasonkit/plugins/                 |
|     /usr/share/reasonkit/plugins/                       |
|                                                         |
|  5. Cargo-Installed Plugins                             |
|     ~/.cargo/bin/ (reasonkit-plugin-* binaries)         |
|                                                         |
|  6. pip-Installed Plugins                               |
|     reasonkit.plugins entry point group                 |
|                                                         |
+---------------------------------------------------------+
```

### 7.2 Discovery Algorithm

```rust
/// Plugin discovery implementation
pub struct PluginDiscovery {
    search_paths: Vec<PathBuf>,
    cache: DiscoveryCache,
}

impl PluginDiscovery {
    /// Discover all available plugins
    pub async fn discover(&mut self) -> Result<Vec<DiscoveredPlugin>> {
        let mut discovered = Vec::new();

        for path in &self.search_paths {
            if !path.exists() {
                continue;
            }

            // Check cache first
            if let Some(cached) = self.cache.get(path) {
                if !cached.is_stale() {
                    discovered.extend(cached.plugins.clone());
                    continue;
                }
            }

            // Scan directory
            let plugins = self.scan_directory(path).await?;

            // Update cache
            self.cache.set(path, &plugins);

            discovered.extend(plugins);
        }

        // Discover installed plugins
        discovered.extend(self.discover_cargo_plugins()?);
        discovered.extend(self.discover_pip_plugins()?);

        // Deduplicate by ID (first wins)
        let mut seen = HashSet::new();
        discovered.retain(|p| seen.insert(p.manifest.id.clone()));

        Ok(discovered)
    }

    /// Scan a directory for plugins
    async fn scan_directory(&self, dir: &Path) -> Result<Vec<DiscoveredPlugin>> {
        let mut plugins = Vec::new();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            // Check for plugin.toml
            let manifest_path = if path.is_dir() {
                path.join("plugin.toml")
            } else if path.extension().map_or(false, |e| e == "toml") {
                path.clone()
            } else {
                continue;
            };

            if manifest_path.exists() {
                match self.load_manifest(&manifest_path) {
                    Ok(manifest) => {
                        plugins.push(DiscoveredPlugin {
                            manifest,
                            path: path.clone(),
                            source: DiscoverySource::FileSystem,
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load manifest {}: {}", manifest_path.display(), e);
                    }
                }
            }
        }

        Ok(plugins)
    }

    /// Discover plugins installed via Cargo
    fn discover_cargo_plugins(&self) -> Result<Vec<DiscoveredPlugin>> {
        let mut plugins = Vec::new();

        if let Some(cargo_home) = std::env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .or_else(|| dirs::home_dir().map(|h| h.join(".cargo")))
        {
            let bin_dir = cargo_home.join("bin");

            for entry in std::fs::read_dir(&bin_dir).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().to_string();

                if name.starts_with("reasonkit-plugin-") {
                    // Query plugin for manifest
                    if let Ok(manifest) = self.query_binary_manifest(&entry.path()) {
                        plugins.push(DiscoveredPlugin {
                            manifest,
                            path: entry.path(),
                            source: DiscoverySource::Cargo,
                        });
                    }
                }
            }
        }

        Ok(plugins)
    }

    /// Discover plugins installed via pip
    fn discover_pip_plugins(&self) -> Result<Vec<DiscoveredPlugin>> {
        // Uses Python interop to query entry points
        // Implementation depends on PyO3 bindings
        Ok(Vec::new())
    }
}
```

### 7.3 Registry Service

Optional online registry for plugin discovery.

```rust
/// Plugin registry client
pub struct RegistryClient {
    base_url: String,
    http: reqwest::Client,
    cache: RegistryCache,
}

impl RegistryClient {
    /// Default registry URL
    pub const DEFAULT_REGISTRY: &'static str = "https://plugins.reasonkit.sh/api/v1";

    /// Search plugins
    pub async fn search(&self, query: &SearchQuery) -> Result<SearchResults> {
        let url = format!("{}/plugins/search", self.base_url);

        let response = self.http
            .get(&url)
            .query(&query)
            .send()
            .await?
            .json()
            .await?;

        Ok(response)
    }

    /// Get plugin details
    pub async fn get_plugin(&self, id: &str) -> Result<PluginDetails> {
        let url = format!("{}/plugins/{}", self.base_url, id);

        let response = self.http
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        Ok(response)
    }

    /// Download plugin package
    pub async fn download(&self, id: &str, version: &str, dest: &Path) -> Result<PathBuf> {
        let url = format!("{}/plugins/{}/versions/{}/download", self.base_url, id, version);

        let response = self.http
            .get(&url)
            .send()
            .await?;

        // Verify signature
        self.verify_package_signature(&response)?;

        // Extract to destination
        let package_path = dest.join(format!("{}-{}.tar.gz", id, version));
        let mut file = std::fs::File::create(&package_path)?;

        let bytes = response.bytes().await?;
        std::io::copy(&mut bytes.as_ref(), &mut file)?;

        Ok(package_path)
    }
}
```

---

## 8. Plugin API

### 8.1 Configuration Access

```rust
/// Configuration access for plugins
#[async_trait]
pub trait ConfigAccess: Send + Sync {
    /// Get a configuration value
    fn get<T: DeserializeOwned>(&self, key: &str) -> Option<T>;

    /// Get a configuration value with default
    fn get_or<T: DeserializeOwned>(&self, key: &str, default: T) -> T;

    /// Get plugin-specific configuration
    fn plugin_config(&self) -> &serde_json::Value;

    /// Get global ReasonKit configuration
    fn global_config(&self) -> &GlobalConfig;

    /// Watch for configuration changes
    async fn watch(&self, key: &str) -> tokio::sync::watch::Receiver<serde_json::Value>;
}

/// Configuration access implementation
pub struct PluginConfigAccess {
    plugin_id: PluginId,
    config: Arc<RwLock<ConfigStore>>,
}

impl ConfigAccess for PluginConfigAccess {
    fn get<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        let config = self.config.read();
        let plugin_config = config.get_plugin(&self.plugin_id)?;

        let value = plugin_config.pointer(&format!("/{}", key.replace('.', "/")))?;
        serde_json::from_value(value.clone()).ok()
    }

    fn get_or<T: DeserializeOwned>(&self, key: &str, default: T) -> T {
        self.get(key).unwrap_or(default)
    }

    fn plugin_config(&self) -> &serde_json::Value {
        let config = self.config.read();
        config.get_plugin(&self.plugin_id)
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()))
    }

    fn global_config(&self) -> &GlobalConfig {
        let config = self.config.read();
        config.global()
    }

    async fn watch(&self, key: &str) -> tokio::sync::watch::Receiver<serde_json::Value> {
        let (tx, rx) = tokio::sync::watch::channel(
            self.get::<serde_json::Value>(key).unwrap_or_default()
        );

        // Set up watcher
        // Implementation depends on config reload mechanism

        rx
    }
}
```

### 8.2 LLM Access

```rust
/// LLM access interface for plugins
#[async_trait]
pub trait LlmAccess: Send + Sync {
    /// Complete a prompt
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse>;

    /// Complete with streaming
    async fn complete_stream(
        &self,
        request: LlmRequest,
    ) -> Result<impl Stream<Item = Result<StreamChunk>>>;

    /// Get available models
    fn available_models(&self) -> Vec<ModelInfo>;

    /// Get current model
    fn current_model(&self) -> &str;

    /// Estimate token count
    fn estimate_tokens(&self, text: &str) -> usize;

    /// Get pricing for current model
    fn pricing(&self) -> ModelPricing;
}

/// Request builder for plugin LLM calls
pub struct PluginLlmRequest {
    inner: LlmRequest,
    budget_tracker: Arc<BudgetTracker>,
}

impl PluginLlmRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            inner: LlmRequest::new(prompt),
            budget_tracker: Arc::new(BudgetTracker::default()),
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.inner = self.inner.with_system(system);
        self
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.inner = self.inner.with_temperature(temp);
        self
    }

    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.inner = self.inner.with_max_tokens(tokens);
        self
    }
}
```

### 8.3 Storage Access

```rust
/// Storage access interface for plugins
#[async_trait]
pub trait StorageAccess: Send + Sync {
    /// Store a value
    async fn put(&self, key: &str, value: &[u8]) -> Result<()>;

    /// Retrieve a value
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Delete a value
    async fn delete(&self, key: &str) -> Result<()>;

    /// List keys with prefix
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Check if key exists
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Store JSON value
    async fn put_json<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let bytes = serde_json::to_vec(value)?;
        self.put(key, &bytes).await
    }

    /// Retrieve JSON value
    async fn get_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match self.get(key).await? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }
}

/// Scoped storage access (namespaced per plugin)
pub struct PluginStorageAccess {
    plugin_id: PluginId,
    backend: Arc<dyn StorageBackend>,
}

impl PluginStorageAccess {
    fn scoped_key(&self, key: &str) -> String {
        format!("plugins/{}/{}", self.plugin_id.0, key)
    }
}

#[async_trait]
impl StorageAccess for PluginStorageAccess {
    async fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        self.backend.put(&self.scoped_key(key), value).await
    }

    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.backend.get(&self.scoped_key(key)).await
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.backend.delete(&self.scoped_key(key)).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.scoped_key(prefix);
        let keys = self.backend.list(&full_prefix).await?;

        // Strip plugin prefix from returned keys
        let prefix_len = format!("plugins/{}/", self.plugin_id.0).len();
        Ok(keys.into_iter().map(|k| k[prefix_len..].to_string()).collect())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        self.backend.exists(&self.scoped_key(key)).await
    }
}
```

### 8.4 Tracing and Logging

```rust
/// Tracing interface for plugins
pub trait PluginTracer: Send + Sync {
    /// Create a new span
    fn span(&self, name: &str) -> tracing::Span;

    /// Log at trace level
    fn trace(&self, msg: &str);

    /// Log at debug level
    fn debug(&self, msg: &str);

    /// Log at info level
    fn info(&self, msg: &str);

    /// Log at warn level
    fn warn(&self, msg: &str);

    /// Log at error level
    fn error(&self, msg: &str);

    /// Record a metric
    fn metric(&self, name: &str, value: f64, tags: &[(&str, &str)]);

    /// Start timing an operation
    fn start_timer(&self, name: &str) -> Timer;
}

/// Scoped tracer for plugins
pub struct ScopedPluginTracer {
    plugin_id: PluginId,
    inner: Arc<dyn tracing::Subscriber>,
}

impl PluginTracer for ScopedPluginTracer {
    fn span(&self, name: &str) -> tracing::Span {
        tracing::info_span!(
            "plugin",
            plugin_id = %self.plugin_id.0,
            operation = %name,
        )
    }

    fn info(&self, msg: &str) {
        tracing::info!(plugin_id = %self.plugin_id.0, "{}", msg);
    }

    fn error(&self, msg: &str) {
        tracing::error!(plugin_id = %self.plugin_id.0, "{}", msg);
    }

    fn metric(&self, name: &str, value: f64, tags: &[(&str, &str)]) {
        // Send to metrics backend
        // Implementation depends on observability stack
    }

    fn start_timer(&self, name: &str) -> Timer {
        Timer::new(name.to_string(), self.plugin_id.clone())
    }
}
```

---

## 9. Event Hooks

### 9.1 Hook Types

```rust
/// Available hook points in the ReasonKit lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookType {
    // ───────────────────────────────────────────────────────────────────────────
    // Protocol Execution Hooks
    // ───────────────────────────────────────────────────────────────────────────

    /// Before protocol execution starts
    PreProtocolExecute,

    /// After protocol execution completes
    PostProtocolExecute,

    /// Before a step executes
    PreStepExecute,

    /// After a step completes
    PostStepExecute,

    /// On protocol execution error
    OnProtocolError,

    // ───────────────────────────────────────────────────────────────────────────
    // LLM Hooks
    // ───────────────────────────────────────────────────────────────────────────

    /// Before LLM request is sent
    PreLlmRequest,

    /// After LLM response received
    PostLlmResponse,

    /// On LLM error
    OnLlmError,

    // ───────────────────────────────────────────────────────────────────────────
    // Storage Hooks
    // ───────────────────────────────────────────────────────────────────────────

    /// Before storage write
    PreStorageWrite,

    /// After storage write
    PostStorageWrite,

    /// Before storage read
    PreStorageRead,

    /// After storage read
    PostStorageRead,

    // ───────────────────────────────────────────────────────────────────────────
    // Lifecycle Hooks
    // ───────────────────────────────────────────────────────────────────────────

    /// On system startup
    OnStartup,

    /// On system shutdown
    OnShutdown,

    /// On plugin load
    OnPluginLoad,

    /// On plugin unload
    OnPluginUnload,

    /// On configuration change
    OnConfigChange,
}

/// Hook handler trait
#[async_trait]
pub trait HookHandler: Send + Sync {
    /// Handle a hook event
    async fn handle(&self, event: &HookEvent) -> Result<HookAction>;

    /// Get hook priority (lower = earlier)
    fn priority(&self) -> i32 {
        0
    }
}

/// Hook event payload
pub struct HookEvent {
    /// Hook type
    pub hook_type: HookType,

    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Event data
    pub data: serde_json::Value,

    /// Context information
    pub context: HookContext,
}

/// Hook context
pub struct HookContext {
    /// Current protocol (if any)
    pub protocol_id: Option<String>,

    /// Current step (if any)
    pub step_id: Option<String>,

    /// Request ID for tracing
    pub request_id: String,

    /// Plugin that triggered the event (if any)
    pub source_plugin: Option<PluginId>,
}

/// Hook action - what to do after hook
pub enum HookAction {
    /// Continue normal execution
    Continue,

    /// Modify the event data
    Modify(serde_json::Value),

    /// Skip the operation
    Skip,

    /// Abort with error
    Abort(String),
}
```

### 9.2 Hook Registration

```rust
/// Hook registry
pub struct HookRegistry {
    handlers: HashMap<HookType, Vec<RegisteredHandler>>,
}

struct RegisteredHandler {
    plugin_id: PluginId,
    handler: Arc<dyn HookHandler>,
    priority: i32,
}

impl HookRegistry {
    /// Register a hook handler
    pub fn register(
        &mut self,
        plugin_id: PluginId,
        hook_type: HookType,
        handler: Arc<dyn HookHandler>,
    ) {
        let entry = self.handlers.entry(hook_type).or_default();
        let priority = handler.priority();

        entry.push(RegisteredHandler {
            plugin_id,
            handler,
            priority,
        });

        // Sort by priority
        entry.sort_by_key(|h| h.priority);
    }

    /// Unregister all handlers for a plugin
    pub fn unregister_plugin(&mut self, plugin_id: &PluginId) {
        for handlers in self.handlers.values_mut() {
            handlers.retain(|h| &h.plugin_id != plugin_id);
        }
    }

    /// Execute hooks for an event
    pub async fn execute(
        &self,
        event: HookEvent,
        registry: &PluginRegistry,
    ) -> Result<Vec<HookResult>> {
        let handlers = match self.handlers.get(&event.hook_type) {
            Some(h) => h,
            None => return Ok(Vec::new()),
        };

        let mut results = Vec::new();
        let mut current_event = event;

        for handler in handlers {
            let span = tracing::info_span!(
                "hook",
                hook_type = ?current_event.hook_type,
                plugin_id = %handler.plugin_id.0,
            );

            let _guard = span.enter();

            match handler.handler.handle(&current_event).await {
                Ok(action) => {
                    match action {
                        HookAction::Continue => {
                            results.push(HookResult::Continue);
                        }
                        HookAction::Modify(data) => {
                            current_event.data = data;
                            results.push(HookResult::Modified);
                        }
                        HookAction::Skip => {
                            results.push(HookResult::Skipped);
                            break;
                        }
                        HookAction::Abort(reason) => {
                            return Err(Error::HookAbort {
                                plugin_id: handler.plugin_id.clone(),
                                reason,
                            });
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Hook error from {}: {}", handler.plugin_id.0, e);
                    results.push(HookResult::Error(e.to_string()));
                }
            }
        }

        Ok(results)
    }
}
```

---

## 10. Security Model

### 10.1 Trust Levels

```
+------------------------------------------------------------------+
|                      PLUGIN TRUST HIERARCHY                       |
+------------------------------------------------------------------+
|                                                                  |
|  OFFICIAL (Trust Level 3)                                        |
|  ├── Source: ReasonKit organization                              |
|  ├── Verification: Signed with ReasonKit key                     |
|  ├── Permissions: Full access to all capabilities                |
|  └── Example: Core ThinkTools, official integrations             |
|                                                                  |
|  VERIFIED (Trust Level 2)                                        |
|  ├── Source: Third-party, reviewed by ReasonKit team             |
|  ├── Verification: Code review + signed by verified publisher    |
|  ├── Permissions: Requested capabilities (reviewed)              |
|  └── Example: Popular community plugins, partner integrations    |
|                                                                  |
|  COMMUNITY (Trust Level 1)                                       |
|  ├── Source: Any third-party                                     |
|  ├── Verification: Automated security scan                       |
|  ├── Permissions: Sandboxed, limited capabilities                |
|  └── Example: Most community plugins                             |
|                                                                  |
|  UNTRUSTED (Trust Level 0)                                       |
|  ├── Source: Unknown/local development                           |
|  ├── Verification: None                                          |
|  ├── Permissions: Maximum sandboxing, no network/secrets         |
|  └── Example: Development plugins, unverified sources            |
|                                                                  |
+------------------------------------------------------------------+
```

### 10.2 Capability Model

```rust
/// Capability enforcement
pub struct CapabilityEnforcer {
    trust_level: TrustLevel,
    granted: HashSet<Capability>,
    denied: HashSet<Capability>,
}

impl CapabilityEnforcer {
    /// Check if capability is allowed
    pub fn check(&self, capability: Capability) -> Result<()> {
        // Explicitly denied
        if self.denied.contains(&capability) {
            return Err(Error::CapabilityDenied(capability));
        }

        // Explicitly granted
        if self.granted.contains(&capability) {
            return Ok(());
        }

        // Check trust level defaults
        match self.trust_level {
            TrustLevel::Official => Ok(()), // All capabilities
            TrustLevel::Verified => self.check_verified_default(capability),
            TrustLevel::Community => self.check_community_default(capability),
            TrustLevel::Untrusted => self.check_untrusted_default(capability),
        }
    }

    fn check_verified_default(&self, cap: Capability) -> Result<()> {
        // Verified plugins can access most capabilities except secrets
        match cap {
            Capability::SecretsAccess => Err(Error::CapabilityDenied(cap)),
            Capability::ProcessExec => Err(Error::CapabilityDenied(cap)),
            _ => Ok(()),
        }
    }

    fn check_community_default(&self, cap: Capability) -> Result<()> {
        // Community plugins have limited access
        match cap {
            Capability::ConfigAccess => Ok(()),
            Capability::LlmAccess => Ok(()),
            Capability::FileSystemRead => Ok(()), // Read-only
            _ => Err(Error::CapabilityDenied(cap)),
        }
    }

    fn check_untrusted_default(&self, cap: Capability) -> Result<()> {
        // Untrusted plugins can only access config and LLM
        match cap {
            Capability::ConfigAccess => Ok(()),
            Capability::LlmAccess => Ok(()),
            _ => Err(Error::CapabilityDenied(cap)),
        }
    }
}
```

### 10.3 Sandboxing

```rust
/// Plugin sandbox for resource isolation
pub struct PluginSandbox {
    limits: ResourceLimits,
    trackers: HashMap<PluginId, ResourceTracker>,
}

impl PluginSandbox {
    /// Create sandboxed execution context
    pub fn create_context(
        &self,
        plugin_id: &PluginId,
        enforcer: CapabilityEnforcer,
    ) -> SandboxedContext {
        let tracker = self.trackers
            .get(plugin_id)
            .cloned()
            .unwrap_or_else(|| ResourceTracker::new(self.limits.clone()));

        SandboxedContext {
            plugin_id: plugin_id.clone(),
            enforcer,
            tracker,
            start_time: std::time::Instant::now(),
        }
    }

    /// Execute with resource limits
    pub async fn execute<F, T>(
        &self,
        ctx: &SandboxedContext,
        f: F,
    ) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        // Check resource budget
        ctx.tracker.check_budget()?;

        // Execute with timeout
        let timeout = Duration::from_millis(self.limits.max_cpu_time_ms);

        match tokio::time::timeout(timeout, f).await {
            Ok(result) => {
                ctx.tracker.record_execution();
                result
            }
            Err(_) => {
                Err(Error::ResourceExhausted {
                    resource: "cpu_time".into(),
                    limit: self.limits.max_cpu_time_ms as usize,
                })
            }
        }
    }
}

/// Resource tracker per plugin
#[derive(Clone)]
pub struct ResourceTracker {
    limits: ResourceLimits,
    memory_used: Arc<AtomicUsize>,
    requests_count: Arc<AtomicU32>,
    fs_ops_count: Arc<AtomicU32>,
    tokens_used: Arc<AtomicU32>,
    window_start: Arc<AtomicU64>,
}

impl ResourceTracker {
    /// Check if within budget
    pub fn check_budget(&self) -> Result<()> {
        // Reset counters if window expired
        self.maybe_reset_window();

        if self.memory_used.load(Ordering::Relaxed) > self.limits.max_memory {
            return Err(Error::ResourceExhausted {
                resource: "memory".into(),
                limit: self.limits.max_memory,
            });
        }

        if self.requests_count.load(Ordering::Relaxed) > self.limits.max_requests_per_minute {
            return Err(Error::ResourceExhausted {
                resource: "requests_per_minute".into(),
                limit: self.limits.max_requests_per_minute as usize,
            });
        }

        if self.tokens_used.load(Ordering::Relaxed) > self.limits.max_llm_tokens {
            return Err(Error::ResourceExhausted {
                resource: "llm_tokens".into(),
                limit: self.limits.max_llm_tokens as usize,
            });
        }

        Ok(())
    }

    /// Record resource usage
    pub fn record_usage(&self, memory: usize, requests: u32, tokens: u32) {
        self.memory_used.fetch_add(memory, Ordering::Relaxed);
        self.requests_count.fetch_add(requests, Ordering::Relaxed);
        self.tokens_used.fetch_add(tokens, Ordering::Relaxed);
    }
}
```

---

## 11. Plugin Development SDK

### 11.1 Plugin Template Generator

```bash
# Create new plugin from template
rk-core plugin new my-thinktool --type thinktool

# Create with specific options
rk-core plugin new my-provider --type llm_provider --language rust

# Create Python plugin
rk-core plugin new my-formatter --type formatter --language python
```

Generated structure:

```
my-thinktool/
├── plugin.toml              # Plugin manifest
├── Cargo.toml               # Rust crate configuration
├── src/
│   ├── lib.rs               # Main library entry
│   └── plugin.rs            # Plugin implementation
├── tests/
│   ├── integration.rs       # Integration tests
│   └── unit.rs              # Unit tests
├── examples/
│   └── basic_usage.rs       # Usage examples
├── README.md                # Plugin documentation
├── LICENSE                  # License file
└── .github/
    └── workflows/
        └── ci.yml           # CI configuration
```

### 11.2 Development Server

```bash
# Start development server with hot-reload
rk-core plugin dev ./my-thinktool

# Watch mode with auto-rebuild
rk-core plugin dev ./my-thinktool --watch

# Test execution
rk-core plugin test ./my-thinktool

# Validate manifest
rk-core plugin validate ./my-thinktool/plugin.toml
```

### 11.3 Testing Utilities

```rust
//! Plugin testing utilities

/// Test harness for plugin development
pub struct PluginTestHarness {
    manager: PluginManager,
    mock_llm: MockLlmClient,
    mock_storage: MockStorageClient,
}

impl PluginTestHarness {
    /// Create test harness with mock dependencies
    pub fn new() -> Self {
        Self {
            manager: PluginManager::new(PluginManagerConfig {
                default_trust: TrustLevel::Untrusted,
                ..Default::default()
            }).unwrap(),
            mock_llm: MockLlmClient::new(),
            mock_storage: MockStorageClient::new(),
        }
    }

    /// Load plugin for testing
    pub async fn load_plugin(&mut self, path: impl AsRef<Path>) -> Result<PluginId> {
        let manifest = load_manifest(path)?;
        self.manager.load_plugin(&manifest).await
    }

    /// Execute ThinkTool with test input
    pub async fn execute_thinktool(
        &self,
        plugin_id: &PluginId,
        input: ProtocolInput,
    ) -> Result<ProtocolOutput> {
        let plugin = self.manager.get(plugin_id)
            .ok_or(Error::NotFound { resource: plugin_id.0.clone() })?;

        let thinktool = plugin.as_any()
            .downcast_ref::<Box<dyn ThinkToolPlugin>>()
            .ok_or(Error::PluginTypeMismatch)?;

        let ctx = self.create_test_context();
        thinktool.execute(input, &ctx).await
    }

    /// Create test context with mocks
    fn create_test_context(&self) -> PluginContext {
        PluginContext {
            config: Arc::new(MockConfigAccess::new()),
            llm: Some(Arc::new(self.mock_llm.clone())),
            storage: Some(Arc::new(self.mock_storage.clone())),
            tracer: Arc::new(TestTracer::new()),
            events: Arc::new(TestEventEmitter::new()),
            resources: Arc::new(ResourceTracker::unlimited()),
        }
    }

    /// Assert output contains expected fields
    pub fn assert_output_contains(&self, output: &ProtocolOutput, field: &str) {
        assert!(
            output.data.contains_key(field),
            "Output missing expected field: {}", field
        );
    }

    /// Assert confidence within range
    pub fn assert_confidence(&self, output: &ProtocolOutput, min: f64, max: f64) {
        assert!(
            output.confidence >= min && output.confidence <= max,
            "Confidence {} not in range [{}, {}]", output.confidence, min, max
        );
    }
}

/// Mock LLM for testing
#[derive(Clone)]
pub struct MockLlmClient {
    responses: Arc<RwLock<HashMap<String, String>>>,
    default_response: String,
}

impl MockLlmClient {
    pub fn new() -> Self {
        Self {
            responses: Arc::new(RwLock::new(HashMap::new())),
            default_response: "Mock response\n\nConfidence: 0.85".to_string(),
        }
    }

    /// Set response for a prompt pattern
    pub fn mock_response(&self, pattern: &str, response: &str) {
        self.responses.write().insert(pattern.to_string(), response.to_string());
    }
}

#[async_trait]
impl LlmAccess for MockLlmClient {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse> {
        let responses = self.responses.read();

        let content = responses
            .iter()
            .find(|(pattern, _)| request.prompt.contains(pattern.as_str()))
            .map(|(_, response)| response.clone())
            .unwrap_or_else(|| self.default_response.clone());

        Ok(LlmResponse {
            content,
            usage: LlmUsage { input_tokens: 100, output_tokens: 150 },
            finish_reason: FinishReason::Stop,
            model: "mock".to_string(),
        })
    }

    fn available_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo { id: "mock".into(), name: "Mock Model".into() }]
    }

    fn current_model(&self) -> &str {
        "mock"
    }

    fn estimate_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }

    fn pricing(&self) -> ModelPricing {
        ModelPricing { input_per_1m: 0.0, output_per_1m: 0.0 }
    }
}
```

---

## 12. Plugin Distribution

### 12.1 Package Format

```
my-plugin-1.0.0.tar.gz
├── plugin.toml              # Manifest (required)
├── signature.sig            # Package signature
├── checksums.sha256         # File checksums
├── lib/
│   └── libmy_plugin.so      # Compiled plugin (Linux)
│   └── libmy_plugin.dylib   # Compiled plugin (macOS)
│   └── my_plugin.dll        # Compiled plugin (Windows)
├── python/                  # Python files (if applicable)
│   └── my_plugin/
├── assets/                  # Plugin assets
│   └── icon.svg
└── README.md                # Documentation
```

### 12.2 Publishing

```bash
# Build release package
rk-core plugin build --release

# Sign package (requires publisher key)
rk-core plugin sign ./target/release/my-plugin-1.0.0.tar.gz

# Publish to registry
rk-core plugin publish

# Publish with specific version
rk-core plugin publish --version 1.0.0

# Publish to private registry
rk-core plugin publish --registry https://internal.example.com/plugins
```

### 12.3 Installation

```bash
# Install from registry
rk-core plugin install my-plugin

# Install specific version
rk-core plugin install my-plugin@1.0.0

# Install from local file
rk-core plugin install ./my-plugin-1.0.0.tar.gz

# Install from GitHub
rk-core plugin install github:user/my-plugin

# Install from URL
rk-core plugin install https://example.com/plugins/my-plugin-1.0.0.tar.gz

# Install with verification disabled (not recommended)
rk-core plugin install my-plugin --no-verify
```

### 12.4 Version Management

```bash
# List installed plugins
rk-core plugin list

# Show plugin details
rk-core plugin info my-plugin

# Update plugin
rk-core plugin update my-plugin

# Update all plugins
rk-core plugin update --all

# Pin version
rk-core plugin pin my-plugin@1.0.0

# Unpin version
rk-core plugin unpin my-plugin

# Rollback to previous version
rk-core plugin rollback my-plugin

# Uninstall plugin
rk-core plugin uninstall my-plugin
```

---

## 13. Plugin Marketplace

### 13.1 Registry Features

```
+------------------------------------------------------------------+
|                    PLUGIN MARKETPLACE FEATURES                    |
+------------------------------------------------------------------+
|                                                                  |
|  DISCOVERY                                                       |
|  ├── Full-text search across plugins                             |
|  ├── Filter by type, category, tags                              |
|  ├── Sort by downloads, rating, recency                          |
|  └── Personalized recommendations                                |
|                                                                  |
|  QUALITY SIGNALS                                                 |
|  ├── Star ratings (1-5)                                          |
|  ├── User reviews                                                |
|  ├── Download counts                                             |
|  ├── Verified publisher badge                                    |
|  ├── Security audit status                                       |
|  └── Compatibility matrix                                        |
|                                                                  |
|  PUBLISHER FEATURES                                              |
|  ├── Publisher verification                                      |
|  ├── Analytics dashboard                                         |
|  ├── Version management                                          |
|  └── Deprecation notices                                         |
|                                                                  |
|  MONETIZATION (Future)                                           |
|  ├── Paid plugins                                                |
|  ├── Subscription models                                         |
|  ├── Revenue sharing                                             |
|  └── Enterprise licensing                                        |
|                                                                  |
+------------------------------------------------------------------+
```

### 13.2 Registry API

```rust
/// Registry API endpoints
impl RegistryClient {
    // ─────────────────────────────────────────────────────────────────────────
    // Search & Discovery
    // ─────────────────────────────────────────────────────────────────────────

    /// GET /plugins/search
    pub async fn search(&self, query: &SearchQuery) -> Result<SearchResults>;

    /// GET /plugins/featured
    pub async fn featured(&self) -> Result<Vec<PluginSummary>>;

    /// GET /plugins/trending
    pub async fn trending(&self, period: TrendingPeriod) -> Result<Vec<PluginSummary>>;

    /// GET /plugins/categories
    pub async fn categories(&self) -> Result<Vec<Category>>;

    /// GET /plugins/tags
    pub async fn tags(&self) -> Result<Vec<Tag>>;

    // ─────────────────────────────────────────────────────────────────────────
    // Plugin Details
    // ─────────────────────────────────────────────────────────────────────────

    /// GET /plugins/{id}
    pub async fn get_plugin(&self, id: &str) -> Result<PluginDetails>;

    /// GET /plugins/{id}/versions
    pub async fn versions(&self, id: &str) -> Result<Vec<VersionInfo>>;

    /// GET /plugins/{id}/versions/{version}
    pub async fn get_version(&self, id: &str, version: &str) -> Result<VersionDetails>;

    /// GET /plugins/{id}/reviews
    pub async fn reviews(&self, id: &str, page: u32) -> Result<ReviewPage>;

    /// GET /plugins/{id}/stats
    pub async fn stats(&self, id: &str) -> Result<PluginStats>;

    // ─────────────────────────────────────────────────────────────────────────
    // Download & Installation
    // ─────────────────────────────────────────────────────────────────────────

    /// GET /plugins/{id}/versions/{version}/download
    pub async fn download(&self, id: &str, version: &str) -> Result<DownloadResponse>;

    /// GET /plugins/{id}/versions/{version}/checksum
    pub async fn checksum(&self, id: &str, version: &str) -> Result<Checksums>;

    // ─────────────────────────────────────────────────────────────────────────
    // Publisher Operations (authenticated)
    // ─────────────────────────────────────────────────────────────────────────

    /// POST /plugins
    pub async fn publish(&self, package: &PluginPackage) -> Result<PublishResult>;

    /// PATCH /plugins/{id}
    pub async fn update_metadata(&self, id: &str, metadata: &UpdateMetadata) -> Result<()>;

    /// DELETE /plugins/{id}/versions/{version}
    pub async fn yank_version(&self, id: &str, version: &str) -> Result<()>;

    /// POST /plugins/{id}/deprecate
    pub async fn deprecate(&self, id: &str, message: &str, replacement: Option<&str>) -> Result<()>;
}
```

---

## 14. Testing and Quality

### 14.1 Testing Requirements

| Requirement           | Description          | Enforcement          |
| --------------------- | -------------------- | -------------------- |
| **Unit Tests**        | Core logic coverage  | >= 80% coverage      |
| **Integration Tests** | End-to-end execution | Required for publish |
| **Security Scan**     | Vulnerability check  | Automated on publish |
| **Documentation**     | README + API docs    | Required fields      |
| **Performance**       | Benchmark baseline   | No > 2x regression   |

### 14.2 Quality Checklist

```toml
# .reasonkit/quality.toml - Quality gate configuration

[quality]
# Minimum test coverage
min_coverage = 80

# Maximum response time (ms)
max_response_time = 5000

# Maximum memory usage (MB)
max_memory = 256

# Required documentation sections
required_docs = ["README.md", "CHANGELOG.md"]

[security]
# Disallowed dependencies
blocked_deps = ["openssl < 3.0.0"]

# Required license compatibility
allowed_licenses = ["MIT", "Apache-2.0", "BSD-3-Clause"]

# Security scan requirements
vuln_severity_block = "high"

[compatibility]
# Minimum ReasonKit version
min_reasonkit = "1.0.0"

# Maximum ReasonKit version (optional)
max_reasonkit = "2.0.0"

# Required Rust version (for Rust plugins)
rust_version = "1.74"

# Required Python version (for Python plugins)
python_version = "3.9"
```

### 14.3 CI/CD Integration

```yaml
# .github/workflows/plugin-ci.yml

name: Plugin CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install ReasonKit
        run: cargo install rk-core

      - name: Validate Manifest
        run: rk-core plugin validate plugin.toml

      - name: Build
        run: cargo build --release

      - name: Test
        run: cargo test

      - name: Integration Test
        run: rk-core plugin test --integration

      - name: Security Scan
        run: rk-core plugin security-scan

      - name: Coverage
        run: |
          cargo install cargo-tarpaulin
          cargo tarpaulin --out Xml

      - name: Check Coverage
        run: |
          coverage=$(grep -oP 'line-rate="\K[0-9.]+' cobertura.xml)
          if (( $(echo "$coverage < 0.80" | bc -l) )); then
            echo "Coverage $coverage is below 80%"
            exit 1
          fi

  publish:
    needs: test
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Release
        run: rk-core plugin build --release

      - name: Sign Package
        run: rk-core plugin sign ./target/release/*.tar.gz
        env:
          REASONKIT_SIGNING_KEY: ${{ secrets.SIGNING_KEY }}

      - name: Publish
        run: rk-core plugin publish
        env:
          REASONKIT_REGISTRY_TOKEN: ${{ secrets.REGISTRY_TOKEN }}
```

---

## 15. Example Plugins

### 15.1 Simple ThinkTool Plugin (Rust)

```rust
//! Legal Analysis ThinkTool
//!
//! Domain-specific reasoning for legal document analysis.

use async_trait::async_trait;
use reasonkit::plugin::*;
use reasonkit::thinktool::*;

/// Legal Analysis ThinkTool
pub struct LegalAnalyzer {
    id: PluginId,
    metadata: PluginMetadata,
    protocol: Protocol,
}

impl LegalAnalyzer {
    pub fn new() -> Self {
        let protocol = Protocol {
            id: "legal-analyzer".to_string(),
            name: "Legal Analyzer".to_string(),
            version: "1.0.0".to_string(),
            description: "Analyzes legal documents for key issues and risks".to_string(),
            strategy: ReasoningStrategy::Analytical,
            input: InputSpec {
                required: vec!["document".to_string()],
                optional: vec!["jurisdiction".to_string(), "focus_areas".to_string()],
            },
            steps: vec![
                ProtocolStep {
                    id: "extract_provisions".to_string(),
                    action: StepAction::Analyze {
                        criteria: vec![
                            "key_terms".to_string(),
                            "obligations".to_string(),
                            "rights".to_string(),
                        ],
                    },
                    prompt_template: r#"
                        Analyze this legal document and extract key provisions:

                        Document:
                        {{document}}

                        {{#if jurisdiction}}Jurisdiction: {{jurisdiction}}{{/if}}

                        Identify:
                        1. Key defined terms
                        2. Obligations of each party
                        3. Rights granted
                        4. Conditions and limitations
                    "#.to_string(),
                    output_format: StepOutputFormat::Structured,
                    min_confidence: 0.7,
                    depends_on: vec![],
                    branch: None,
                },
                ProtocolStep {
                    id: "identify_risks".to_string(),
                    action: StepAction::Critique {
                        severity: CritiqueSeverity::Standard,
                    },
                    prompt_template: r#"
                        Based on the extracted provisions, identify legal risks:

                        Provisions:
                        {{extract_provisions}}

                        For each risk:
                        1. Description of the risk
                        2. Severity (high/medium/low)
                        3. Affected party
                        4. Mitigation suggestion
                    "#.to_string(),
                    output_format: StepOutputFormat::List,
                    min_confidence: 0.65,
                    depends_on: vec!["extract_provisions".to_string()],
                    branch: None,
                },
                ProtocolStep {
                    id: "summary".to_string(),
                    action: StepAction::Synthesize {
                        aggregation: AggregationType::ThematicClustering,
                    },
                    prompt_template: r#"
                        Provide executive summary of legal analysis:

                        Key Provisions:
                        {{extract_provisions}}

                        Identified Risks:
                        {{identify_risks}}

                        Include:
                        1. Document type and purpose
                        2. Key takeaways (3-5 bullets)
                        3. Top risks requiring attention
                        4. Recommended actions
                        5. Overall assessment
                    "#.to_string(),
                    output_format: StepOutputFormat::Structured,
                    min_confidence: 0.75,
                    depends_on: vec!["extract_provisions".to_string(), "identify_risks".to_string()],
                    branch: None,
                },
            ],
            output: OutputSpec {
                format: "LegalAnalysisResult".to_string(),
                fields: vec![
                    "provisions".to_string(),
                    "risks".to_string(),
                    "summary".to_string(),
                    "confidence".to_string(),
                ],
            },
            validation: vec![],
            metadata: ProtocolMetadata {
                category: "legal".to_string(),
                composable_with: vec!["proofguard".to_string(), "brutalhonesty".to_string()],
                typical_tokens: 3000,
                estimated_latency_ms: 8000,
                ..Default::default()
            },
        };

        Self {
            id: PluginId::new("legal-analyzer"),
            metadata: PluginMetadata {
                name: "Legal Analyzer".to_string(),
                version: "1.0.0".to_string(),
                description: "Domain-specific legal document analysis".to_string(),
                plugin_type: PluginType::ThinkTool,
                author: "ReasonKit Community".to_string(),
                license: "Apache-2.0".to_string(),
                repository: Some("https://github.com/reasonkit/legal-analyzer".to_string()),
                reasonkit_version: ">=1.0.0".to_string(),
                dependencies: vec![],
                capabilities: vec![Capability::LlmAccess, Capability::ConfigAccess],
                tags: vec!["legal".to_string(), "compliance".to_string(), "contracts".to_string()],
            },
            protocol,
        }
    }
}

#[async_trait]
impl Plugin for LegalAnalyzer {
    fn id(&self) -> &PluginId {
        &self.id
    }

    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    async fn initialize(&mut self, _ctx: &PluginContext) -> Result<()> {
        Ok(())
    }

    async fn start(&mut self, _ctx: &PluginContext) -> Result<()> {
        Ok(())
    }

    async fn stop(&mut self, _ctx: &PluginContext) -> Result<()> {
        Ok(())
    }

    async fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[async_trait]
impl ThinkToolPlugin for LegalAnalyzer {
    fn protocol(&self) -> &Protocol {
        &self.protocol
    }

    async fn execute(
        &self,
        input: ProtocolInput,
        ctx: &PluginContext,
    ) -> Result<ProtocolOutput> {
        // Validate input
        self.validate_input(&input)?;

        // Get LLM client
        let llm = ctx.llm.as_ref()
            .ok_or(Error::CapabilityDenied(Capability::LlmAccess))?;

        // Execute protocol using standard executor
        let executor = ProtocolExecutor::new()?;
        executor.execute(&self.protocol.id, input).await
    }
}

/// Plugin entry point
#[no_mangle]
pub extern "C" fn reasonkit_plugin_create() -> Box<dyn Plugin> {
    Box::new(LegalAnalyzer::new())
}
```

### 15.2 LLM Provider Plugin (Python)

```python
"""
Ollama LLM Provider Plugin

Connects ReasonKit to locally running Ollama models.
"""

import httpx
from typing import Any, Dict, List, Optional

from reasonkit.plugin import (
    LlmProviderPlugin,
    PluginMetadata,
    PluginType,
    Capability,
    PluginContext,
)


class OllamaProvider(LlmProviderPlugin):
    """Ollama local LLM provider"""

    def __init__(self):
        self._base_url = "http://localhost:11434"
        self._client: Optional[httpx.AsyncClient] = None
        self._models: List[str] = []

    @property
    def id(self) -> str:
        return "ollama-provider"

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Ollama Provider",
            version="1.0.0",
            description="Connect to locally running Ollama models",
            plugin_type=PluginType.LLM_PROVIDER,
            author="ReasonKit Community",
            license="Apache-2.0",
            repository="https://github.com/reasonkit/ollama-provider",
            capabilities=[Capability.NETWORK, Capability.CONFIG_ACCESS],
            tags=["ollama", "local", "llm"],
        )

    @property
    def provider_id(self) -> str:
        return "ollama"

    @property
    def supported_models(self) -> List[str]:
        return self._models

    async def initialize(self, ctx: PluginContext) -> None:
        """Initialize connection to Ollama"""
        # Get config
        self._base_url = ctx.config.get("base_url", "http://localhost:11434")

        # Create HTTP client
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=60.0,
        )

        # Fetch available models
        try:
            response = await self._client.get("/api/tags")
            data = response.json()
            self._models = [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"Warning: Could not fetch Ollama models: {e}")
            self._models = []

    async def start(self, ctx: PluginContext) -> None:
        pass

    async def stop(self, ctx: PluginContext) -> None:
        pass

    async def cleanup(self) -> None:
        if self._client:
            await self._client.aclose()

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            import httpx
            response = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Complete a chat conversation"""
        if not self._client:
            raise RuntimeError("Plugin not initialized")

        # Convert to Ollama format
        prompt = self._format_messages(messages)

        # Make request
        response = await self._client.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 2000),
                },
            },
        )

        data = response.json()

        return {
            "content": data["response"],
            "usage": {
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            "model": model,
            "finish_reason": "stop",
        }

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Ollama"""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"Human: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        parts.append("Assistant:")
        return "\n\n".join(parts)

    def pricing(self, model: str) -> Optional[Dict[str, float]]:
        """Ollama is free (local)"""
        return {"input_per_1m": 0.0, "output_per_1m": 0.0}


# Plugin entry point
def create_plugin():
    return OllamaProvider()
```

### 15.3 Output Formatter Plugin

```rust
//! Markdown Report Formatter Plugin
//!
//! Formats protocol outputs as beautiful Markdown reports.

use async_trait::async_trait;
use reasonkit::plugin::*;
use reasonkit::thinktool::ProtocolOutput;

pub struct MarkdownFormatter {
    id: PluginId,
    metadata: PluginMetadata,
    templates: HashMap<String, String>,
}

impl MarkdownFormatter {
    pub fn new() -> Self {
        let mut templates = HashMap::new();

        // Default template
        templates.insert("default".to_string(), r#"
# {{title}}

> Generated by ReasonKit | {{timestamp}}

## Executive Summary

**Confidence:** {{confidence}}%

{{summary}}

## Detailed Analysis

{{#each steps}}
### {{this.name}}

{{this.content}}

*Confidence: {{this.confidence}}%*

{{/each}}

## Conclusion

{{conclusion}}

---
*Generated with ReasonKit v{{version}}*
"#.to_string());

        Self {
            id: PluginId::new("markdown-formatter"),
            metadata: PluginMetadata {
                name: "Markdown Formatter".to_string(),
                version: "1.0.0".to_string(),
                description: "Format protocol outputs as Markdown reports".to_string(),
                plugin_type: PluginType::Formatter,
                author: "ReasonKit".to_string(),
                license: "Apache-2.0".to_string(),
                capabilities: vec![Capability::ConfigAccess],
                tags: vec!["markdown".to_string(), "report".to_string(), "export".to_string()],
                ..Default::default()
            },
            templates,
        }
    }
}

#[async_trait]
impl Plugin for MarkdownFormatter {
    fn id(&self) -> &PluginId { &self.id }
    fn metadata(&self) -> &PluginMetadata { &self.metadata }

    async fn initialize(&mut self, ctx: &PluginContext) -> Result<()> {
        // Load custom templates from config
        if let Some(custom) = ctx.config.get::<HashMap<String, String>>("templates") {
            self.templates.extend(custom);
        }
        Ok(())
    }

    async fn start(&mut self, _ctx: &PluginContext) -> Result<()> { Ok(()) }
    async fn stop(&mut self, _ctx: &PluginContext) -> Result<()> { Ok(()) }
    async fn cleanup(&mut self) -> Result<()> { Ok(()) }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[async_trait]
impl FormatterPlugin for MarkdownFormatter {
    fn format_id(&self) -> &str {
        "markdown"
    }

    fn mime_type(&self) -> &str {
        "text/markdown"
    }

    async fn format(
        &self,
        output: &ProtocolOutput,
        options: &FormatOptions,
    ) -> Result<Vec<u8>> {
        let template_name = options.template.as_deref().unwrap_or("default");
        let template = self.templates.get(template_name)
            .ok_or_else(|| Error::NotFound {
                resource: format!("template:{}", template_name),
            })?;

        // Build context for template
        let mut context = tera::Context::new();
        context.insert("title", &output.protocol_id);
        context.insert("timestamp", &chrono::Utc::now().to_rfc3339());
        context.insert("confidence", &(output.confidence * 100.0));
        context.insert("version", env!("CARGO_PKG_VERSION"));

        // Extract summary from output
        let summary = output.data.get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("No summary available.");
        context.insert("summary", summary);

        // Format steps
        let steps: Vec<serde_json::Value> = output.steps.iter()
            .map(|step| serde_json::json!({
                "name": step.step_id,
                "content": step.as_text().unwrap_or(""),
                "confidence": step.confidence * 100.0,
            }))
            .collect();
        context.insert("steps", &steps);

        // Extract conclusion
        let conclusion = output.data.get("verdict")
            .or(output.data.get("conclusion"))
            .and_then(|v| v.as_str())
            .unwrap_or("See detailed analysis above.");
        context.insert("conclusion", conclusion);

        // Render template
        let mut tera = tera::Tera::default();
        tera.add_raw_template("report", template)?;
        let rendered = tera.render("report", &context)?;

        Ok(rendered.into_bytes())
    }

    fn templates(&self) -> &[FormatTemplate] {
        &[
            FormatTemplate {
                id: "default".to_string(),
                name: "Default Report".to_string(),
                description: "Standard Markdown report format".to_string(),
            },
        ]
    }
}
```

---

## 16. Migration Guide

### 16.1 Migrating from Built-in Protocols

If you have custom protocols defined in YAML, migrate to plugins:

```yaml
# Before: protocols/my-protocol.yaml
id: my-protocol
name: My Protocol
steps:
  - id: step1
    action:
      type: analyze
    prompt_template: "..."
```

```rust
// After: Create a plugin

use reasonkit::thinktool_plugin;

thinktool_plugin!(
    MyProtocol,
    id = "my-protocol",
    name = "My Protocol",
    version = "1.0.0",
    strategy = ReasoningStrategy::Analytical,
    description = "My custom protocol",
    steps = [
        ProtocolStep {
            id: "step1".to_string(),
            action: StepAction::Analyze { criteria: vec![] },
            prompt_template: "...".to_string(),
            // ...
        },
    ]
);
```

### 16.2 Compatibility Matrix

| ReasonKit Version | Plugin API Version | Notes                      |
| ----------------- | ------------------ | -------------------------- |
| 0.1.x             | 1.0                | Initial plugin support     |
| 0.2.x             | 1.0                | Backward compatible        |
| 1.0.x             | 2.0                | Breaking changes (planned) |

---

## Appendix A: Error Codes

| Code       | Name                | Description                           |
| ---------- | ------------------- | ------------------------------------- |
| `PLUG-001` | `PluginNotFound`    | Plugin ID not found in registry       |
| `PLUG-002` | `PluginLoadFailed`  | Failed to load plugin binary/module   |
| `PLUG-003` | `PluginInitFailed`  | Plugin initialization error           |
| `PLUG-004` | `PluginConflict`    | Duplicate plugin ID                   |
| `PLUG-005` | `ManifestInvalid`   | Invalid plugin.toml                   |
| `PLUG-006` | `DependencyMissing` | Required dependency not found         |
| `PLUG-007` | `DependencyCycle`   | Circular dependency detected          |
| `PLUG-008` | `CapabilityDenied`  | Requested capability not granted      |
| `PLUG-009` | `ResourceExhausted` | Resource limit exceeded               |
| `PLUG-010` | `SignatureInvalid`  | Package signature verification failed |

---

## Appendix B: Best Practices

1. **Keep plugins focused**: One plugin, one responsibility
2. **Version carefully**: Follow semver strictly
3. **Document thoroughly**: Include examples and error cases
4. **Test extensively**: Unit, integration, and performance tests
5. **Handle errors gracefully**: Never crash, always return Result
6. **Respect resource limits**: Monitor and optimize usage
7. **Use hooks sparingly**: Prefer explicit API calls over hooks
8. **Sign your releases**: Enable verification for users
9. **Respond to feedback**: Monitor reviews and issues
10. **Deprecate gracefully**: Provide migration paths

---

_ReasonKit Plugin Architecture v1.0.0 | Apache 2.0_
*https://reasonkit.sh/docs/plugins*
