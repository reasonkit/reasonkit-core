//! # Universal Agent Integration Framework
//! 
//! Makes ReasonKit universally compatible with all major AI agent frameworks
//! by leveraging MiniMax M2's cross-platform capabilities.
//!
//! ## Supported Frameworks
//! - **Claude Code**: JSON-formatted outputs with confidence scoring
//! - **Cline**: Structured logical analysis with fallacy detection  
//! - **Kilo Code**: Comprehensive critique with flaw categorization
//! - **Droid**: Android development with mobile-specific optimizations
//! - **Roo Code**: Multi-agent collaboration with protocol delegation
//! - **BlackBox AI**: High-throughput operations with speed optimization
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    UNIVERSAL AGENT INTEGRATION FRAMEWORK        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
//! │  │  AGENT DISCOVERY│  │ PROTOCOL ENGINE │  │   TRANSLATION   │ │
//! │  │    & REGISTRY   │  │     (M2)        │  │    LAYER        │ │
//! │  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
//! │  │ FRAMEWORK       │  │  PERFORMANCE    │  │   ERROR         │ │
//! │  │   ADAPTERS      │  │  MONITORING     │  │  HANDLING       │ │
//! │  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use crate::error::Result;
use crate::thinktool::{Protocol, ProtocolContent, ThinkToolExecutor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub mod adapters;
pub mod discovery;
pub mod optimization;
pub mod translation;

pub use adapters::*;
pub use discovery::*;
pub use optimization::*;
pub use translation::*;

/// Universal Agent Integration Framework
/// Main orchestrator for all agent framework interactions
#[derive(Clone)]
pub struct UniversalAgentFramework {
    registry: Arc<RwLock<AgentRegistry>>,
    protocol_engine: Arc<RwLock<M2ProtocolEngine>>,
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
    error_handler: Arc<RwLock<ErrorHandler>>,
}

impl UniversalAgentFramework {
    /// Initialize the universal agent framework
    pub async fn new() -> Result<Self> {
        let registry = Arc::new(RwLock::new(AgentRegistry::new().await?));
        let protocol_engine = Arc::new(RwLock::new(M2ProtocolEngine::new().await?));
        let performance_monitor = Arc::new(RwLock::new(PerformanceMonitor::new().await?));
        let error_handler = Arc::new(RwLock::new(ErrorHandler::new().await?));

        Ok(Self {
            registry,
            protocol_engine,
            performance_monitor,
            error_handler,
        })
    }

    /// Process a protocol through the universal framework
    /// Automatically detects the best framework and optimizes for it
    pub async fn process_protocol(
        &self,
        protocol: &Protocol,
        target_framework: Option<FrameworkType>,
    ) -> Result<ProcessedProtocol> {
        let registry = self.registry.read().await;
        let mut protocol_engine = self.protocol_engine.write().await;

        // Auto-detect best framework if not specified
        let framework = if let Some(fw) = target_framework {
            fw
        } else {
            registry.auto_detect_best_framework(protocol).await?
        };

        // Get or create adapter for the framework
        let adapter = registry.get_or_create_adapter(framework).await?;

        // Process through M2-optimized protocol engine
        let optimized_protocol = protocol_engine.optimize_for_framework(framework, protocol).await?;

        // Process through framework-specific adapter
        let result = adapter.process_protocol(&optimized_protocol).await?;

        // Monitor performance
        let mut monitor = self.performance_monitor.write().await;
        monitor.record_performance(framework, &result).await?;

        Ok(result)
    }

    /// Register a new framework adapter
    pub async fn register_framework<T: FrameworkAdapter + Send + Sync + 'static>(
        &self,
        adapter: T,
    ) -> Result<()> {
        let mut registry = self.registry.write().await;
        registry.register_adapter(adapter).await?;
        Ok(())
    }

    /// Get performance metrics for all frameworks
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics> {
        let monitor = self.performance_monitor.read().await;
        monitor.get_comprehensive_metrics().await
    }

    /// Benchmark all frameworks
    pub async fn benchmark_frameworks(&self) -> Result<BenchmarkReport> {
        let registry = self.registry.read().await;
        let mut protocol_engine = self.protocol_engine.write().await;

        let mut results = Vec::new();
        
        for framework in FrameworkType::all() {
            if let Some(adapter) = registry.get_adapter(framework).await? {
                let benchmark_result = adapter.benchmark_performance().await?;
                results.push((framework, benchmark_result));
            }
        }

        protocol_engine.generate_benchmark_report(results).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thinktool::{Profile, ThinkToolExecutor};

    #[tokio::test]
    async fn test_universal_framework_initialization() {
        let framework = UniversalAgentFramework::new().await.unwrap();
        
        // Test that all core components are initialized
        assert!(framework.registry.read().await.is_initialized());
        assert!(framework.protocol_engine.read().await.is_initialized());
        assert!(framework.performance_monitor.read().await.is_initialized());
    }

    #[tokio::test]
    async fn test_protocol_processing() {
        let framework = UniversalAgentFramework::new().await.unwrap();
        
        // Create a test protocol
        let executor = ThinkToolExecutor::new();
        let protocol = executor.create_test_protocol("Test query").await.unwrap();
        
        // Process through universal framework
        let result = framework.process_protocol(&protocol, None).await.unwrap();
        
        assert!(result.confidence_score > 0.8);
        assert!(result.processing_time_ms < 100);
    }
}
