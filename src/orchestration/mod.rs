//! # Long-Horizon Task Orchestration System
//!
//! This module implements ReasonKit's long-horizon task orchestration system that leverages
//! MiniMax M2's exceptional 100+ tool calling capability for complex multi-step operations.
//!
//! ## Core Features
//!
//! - **100+ Tool Calling Support**: Leverage M2's proven long-horizon execution capability
//! - **Complex Task Orchestration**: Coordinate multi-step operations across ReasonKit components
//! - **State Persistence**: Maintain context across extended execution sequences
//! - **Multi-Component Coordination**: Orchestrate Core, Web, Mem, and Pro components
//! - **Performance Monitoring**: Real-time tracking and optimization
//! - **Error Recovery**: Robust handling and recovery mechanisms

pub mod component_coordinator;
pub mod error_recovery;
pub mod long_horizon_orchestrator;
pub mod performance_tracker;
pub mod state_manager;
pub mod task_graph;

use serde::{Deserialize, Serialize};

pub use component_coordinator::{ComponentCoordinator, ComponentTask, CoordinationResult};
pub use error_recovery::{ErrorRecovery, RecoveryResult, RecoveryStrategy};
pub use long_horizon_orchestrator::{
    LongHorizonOrchestrator, OrchestrationConfig, OrchestrationResult, TaskExecutionPlan,
};
pub use performance_tracker::{PerformanceTracker, RealTimeMetrics, ResourceUtilization};

pub type PerformanceMetrics = performance_tracker::PerformanceSummary;
pub use state_manager::{ContextSnapshot, StateManager, StatePersistence};
pub use task_graph::{DependencyGraph, TaskGraph, TaskNode, TaskPriority, TaskStatus};

/// Long-horizon orchestration result with comprehensive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongHorizonResult {
    pub execution_id: String,
    pub success: bool,
    pub total_duration_ms: u64,
    pub tool_call_count: u32,
    pub components_coordinated: Vec<String>,
    pub state_transitions: Vec<StateTransition>,
    pub performance_metrics: LongHorizonPerformanceMetrics,
    pub error_recovery_log: Vec<RecoveryLog>,
    pub final_output: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from_state: String,
    pub to_state: String,
    pub timestamp: i64,
    pub trigger: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongHorizonPerformanceMetrics {
    pub avg_tool_call_time_ms: f64,
    pub throughput: f64,
    pub memory_efficiency: f64,
    pub cost_per_tool_call: f64,
    pub reliability_score: f64,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryLog {
    pub error_id: String,
    pub recovery_strategy: String,
    pub success: bool,
    pub recovery_time_ms: u64,
    pub details: String,
}

/// Configuration for long-horizon orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongHorizonConfig {
    pub max_tool_calls: u32,
    pub timeout_ms: u64,
    pub memory_limit_mb: u64,
    pub enable_error_recovery: bool,
    pub enable_state_persistence: bool,
    pub enable_performance_monitoring: bool,
    pub component_timeout_ms: u64,
    pub batch_size: usize,
    pub checkpoint_interval: u32,
}

impl Default for LongHorizonConfig {
    fn default() -> Self {
        Self {
            max_tool_calls: 100,
            timeout_ms: 3_600_000, // 1 hour
            memory_limit_mb: 2048,
            enable_error_recovery: true,
            enable_state_persistence: true,
            enable_performance_monitoring: true,
            component_timeout_ms: 300_000, // 5 minutes per component
            batch_size: 10,
            checkpoint_interval: 10,
        }
    }
}
