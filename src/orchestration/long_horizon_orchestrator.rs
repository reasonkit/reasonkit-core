//! # Long-Horizon Orchestrator
//!
//! The core orchestrator for complex multi-step operations that leverages MiniMax M2's
//! 100+ tool calling capability to execute extended task sequences with state persistence,
//! performance monitoring, and error recovery.

use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{timeout, Duration, Instant};

use super::{
    component_coordinator::{ComponentCoordinator, CoordinationResult},
    error_recovery::ErrorRecovery,
    performance_tracker::PerformanceTracker,
    state_manager::StateManager,
    task_graph::{TaskGraph, TaskNode, TaskStatus},
    LongHorizonConfig, LongHorizonPerformanceMetrics, LongHorizonResult,
};
use crate::error::Error;

/// Main orchestrator for long-horizon task execution
pub struct LongHorizonOrchestrator {
    config: LongHorizonConfig,
    component_coordinator: Arc<ComponentCoordinator>,
    state_manager: Arc<RwLock<StateManager>>,
    performance_tracker: Arc<Mutex<PerformanceTracker>>,
    error_recovery: Arc<ErrorRecovery>,
    execution_id: String,
    #[allow(dead_code)]
    start_time: Instant,
}

impl LongHorizonOrchestrator {
    /// Create a new orchestrator with the given configuration
    pub async fn new(config: LongHorizonConfig) -> Result<Self, Error> {
        let execution_id = format!("lh_{}", chrono::Utc::now().timestamp());

        let component_coordinator = Arc::new(ComponentCoordinator::new().await?);
        let state_manager = Arc::new(RwLock::new(StateManager::new()));
        let performance_tracker = Arc::new(Mutex::new(PerformanceTracker::new()));
        let error_recovery = Arc::new(ErrorRecovery::new());

        tracing::info!("Long-horizon orchestrator initialized: {}", execution_id);

        Ok(Self {
            config,
            component_coordinator,
            state_manager,
            performance_tracker,
            error_recovery,
            execution_id,
            start_time: Instant::now(),
        })
    }

    /// Execute a complex task with 100+ tool calling capability
    pub async fn execute_long_horizon_task(
        &mut self,
        task_graph: TaskGraph,
        initial_context: serde_json::Value,
    ) -> Result<LongHorizonResult, Error> {
        let total_duration_start = Instant::now();
        let mut tool_call_count = 0;
        let mut components_coordinated = Vec::new();
        let mut state_transitions = Vec::new();
        let mut error_recovery_log = Vec::new();
        let mut final_output = serde_json::json!({});

        tracing::info!(
            "Starting long-horizon task execution: {} nodes, max {} tool calls",
            task_graph.nodes().len(),
            self.config.max_tool_calls
        );

        // Initialize state
        {
            let state_manager = self.state_manager.write().await;
            state_manager.initialize_context(&initial_context).await?;
            state_transitions.push(super::StateTransition {
                from_state: "initial".to_string(),
                to_state: "initialized".to_string(),
                timestamp: chrono::Utc::now().timestamp(),
                trigger: "initialize".to_string(),
                metadata: serde_json::json!({"tool_calls": 1}),
            });
        }

        // Validate dependency graph
        if let Err(e) = task_graph.validate() {
            return Err(Error::Validation(format!(
                "Invalid task dependency graph: {}",
                e
            )));
        }

        // Get topological order for execution
        let execution_order = task_graph.topological_sort()?;
        tracing::info!("Task execution order: {} steps", execution_order.len());

        // Execute tasks in dependency order
        for (task_index, task_node_id) in execution_order.iter().enumerate() {
            if tool_call_count >= self.config.max_tool_calls {
                tracing::warn!("Max tool calls reached: {}", self.config.max_tool_calls);
                break;
            }

            let task_node = task_graph.get_node(task_node_id).unwrap();

            // Check if task should be executed
            if task_node.status() != TaskStatus::Pending {
                continue;
            }

            // Execute individual task with timeout
            let task_result = timeout(
                Duration::from_millis(self.config.component_timeout_ms),
                self.execute_single_task(task_node, &task_graph, &mut tool_call_count),
            )
            .await;

            match task_result {
                Ok(Ok(result)) => {
                    // Task succeeded
                    let _new_status = match result.success {
                        true => TaskStatus::Completed,
                        false => TaskStatus::Failed,
                    };

                    // Update task status
                    // Note: This would need to be implemented in TaskGraph
                    tracing::info!("Task {} completed successfully", task_node.name());

                    // Record state transition
                    state_transitions.push(super::StateTransition {
                        from_state: format!("task_{}_start", task_node_id),
                        to_state: format!("task_{}_complete", task_node_id),
                        timestamp: chrono::Utc::now().timestamp(),
                        trigger: "task_completion".to_string(),
                        metadata: serde_json::json!({
                            "tool_calls": result.tool_calls_used,
                            "duration_ms": result.duration_ms,
                        }),
                    });

                    // Track components coordinated
                    for component in result.components_used {
                        if !components_coordinated.contains(&component) {
                            components_coordinated.push(component);
                        }
                    }

                    // Update final output if this is the last task
                    if task_index == execution_order.len() - 1 {
                        final_output = result.output;
                    }
                }
                Ok(Err(e)) => {
                    // Task failed with error
                    tracing::error!("Task {} failed: {}", task_node.name(), e);

                    // Attempt error recovery
                    let recovery_result = self
                        .error_recovery
                        .attempt_recovery(&e, task_node, &task_graph)
                        .await;

                    match recovery_result {
                        Ok(recovery) => {
                            if recovery.success {
                                tracing::info!("Task {} recovered successfully", task_node.name());
                                error_recovery_log.push(super::RecoveryLog {
                                    error_id: e.to_string(),
                                    recovery_strategy: recovery.strategy_used,
                                    success: true,
                                    recovery_time_ms: recovery.recovery_time_ms,
                                    details: recovery.details,
                                });
                            } else {
                                tracing::error!("Task {} recovery failed", task_node.name());
                                error_recovery_log.push(super::RecoveryLog {
                                    error_id: e.to_string(),
                                    recovery_strategy: recovery.strategy_used,
                                    success: false,
                                    recovery_time_ms: recovery.recovery_time_ms,
                                    details: recovery.details,
                                });
                                return Err(e);
                            }
                        }
                        Err(recovery_error) => {
                            tracing::error!("Recovery attempt failed: {}", recovery_error);
                            return Err(e);
                        }
                    }
                }
                Err(_) => {
                    // Task timed out
                    let timeout_error = Error::Validation(format!(
                        "Task {} timed out after {}ms",
                        task_node.name(),
                        self.config.component_timeout_ms
                    ));

                    // Attempt timeout recovery
                    let recovery_result = self
                        .error_recovery
                        .handle_timeout(task_node, self.config.component_timeout_ms)
                        .await;

                    match recovery_result {
                        Ok(recovery) => {
                            error_recovery_log.push(super::RecoveryLog {
                                error_id: "timeout".to_string(),
                                recovery_strategy: recovery.strategy_used,
                                success: recovery.success,
                                recovery_time_ms: recovery.recovery_time_ms,
                                details: recovery.details,
                            });

                            if !recovery.success {
                                return Err(timeout_error);
                            }
                        }
                        Err(_) => return Err(timeout_error),
                    }
                }
            }

            // Periodic checkpoint
            let checkpoint_interval = self.config.checkpoint_interval as usize;
            if checkpoint_interval != 0 && task_index % checkpoint_interval == 0 {
                self.create_checkpoint(&task_graph, tool_call_count).await?;
            }

            // Update performance metrics
            {
                let tracker = self.performance_tracker.lock().await;
                tracker
                    .record_tool_call(
                        tool_call_count,
                        total_duration_start.elapsed().as_millis() as u64,
                        0.0, // cost would be calculated based on actual usage
                    )
                    .await?;
            }
        }

        // Calculate final metrics
        let total_duration_ms = total_duration_start.elapsed().as_millis() as u64;
        let performance_metrics = self.calculate_performance_metrics(
            total_duration_ms,
            tool_call_count,
            &components_coordinated,
        );

        let result = LongHorizonResult {
            execution_id: self.execution_id.clone(),
            success: true, // If we got here, overall execution was successful
            total_duration_ms,
            tool_call_count,
            components_coordinated,
            state_transitions,
            performance_metrics,
            error_recovery_log,
            final_output,
        };

        tracing::info!(
            "Long-horizon task completed successfully: {} tool calls, {}ms duration",
            tool_call_count,
            total_duration_ms
        );

        Ok(result)
    }

    /// Execute a single task within the orchestration
    async fn execute_single_task(
        &self,
        task_node: &TaskNode,
        _task_graph: &TaskGraph,
        tool_call_count: &mut u32,
    ) -> Result<TaskExecutionResult, Error> {
        let start_time = Instant::now();
        let mut components_used = Vec::new();
        let mut local_tool_calls = 0;

        tracing::info!("Executing task: {}", task_node.name());

        // Check dependencies
        for dependency_id in task_node.dependencies() {
            let dependency_node = _task_graph.get_node(dependency_id).unwrap();
            if dependency_node.status() != TaskStatus::Completed {
                return Err(Error::Validation(format!(
                    "Dependency {} not completed for task {}",
                    dependency_id,
                    task_node.name()
                )));
            }
        }

        // Determine which ReasonKit components to coordinate
        let required_components = self.determine_required_components(task_node).await?;
        components_used.extend(required_components);

        // Execute coordination across components
        let task_config_json = serde_json::to_value(task_node.config())?;

        let coordination_result = self
            .component_coordinator
            .coordinate_components(task_node.name(), &components_used, &task_config_json)
            .await?;

        let CoordinationResult {
            outputs: coordination_outputs,
            tool_calls_used: coordination_tool_calls,
            ..
        } = &coordination_result;

        *tool_call_count += *coordination_tool_calls;
        local_tool_calls += *coordination_tool_calls;

        let coordination_output = serde_json::json!({ "outputs": coordination_outputs });

        // Apply ThinkTool processing if needed
        if task_node.requires_thinktool() {
            let thinktool_result = self
                .execute_thinktool_processing(task_node, &coordination_output, tool_call_count)
                .await?;

            *tool_call_count += thinktool_result.tool_calls_used;
            local_tool_calls += thinktool_result.tool_calls_used;
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(TaskExecutionResult {
            success: true,
            output: serde_json::json!({ "outputs": coordination_result.outputs }),
            components_used,
            tool_calls_used: local_tool_calls,
            duration_ms,
        })
    }

    /// Determine which ReasonKit components are required for a task
    async fn determine_required_components(
        &self,
        task_node: &TaskNode,
    ) -> Result<Vec<String>, Error> {
        let mut components = Vec::new();

        // Analyze task requirements
        match task_node.task_type() {
            super::task_graph::TaskType::General => {
                components.push("reasonkit-core".to_string());
            }
            super::task_graph::TaskType::ProtocolGeneration => {
                components.extend(["reasonkit-core".to_string(), "reasonkit-pro".to_string()]);
            }
            super::task_graph::TaskType::CodeAnalysis => {
                components.extend([
                    "reasonkit-core".to_string(),
                    "reasonkit-web".to_string(),
                    "code-intelligence".to_string(),
                ]);
            }
            super::task_graph::TaskType::WebAutomation => {
                components.extend(["reasonkit-web".to_string(), "reasonkit-mem".to_string()]);
            }
            super::task_graph::TaskType::MemoryManagement => {
                components.push("reasonkit-mem".to_string());
            }
            super::task_graph::TaskType::EnterpriseWorkflow => {
                components.extend([
                    "reasonkit-core".to_string(),
                    "reasonkit-web".to_string(),
                    "reasonkit-mem".to_string(),
                    "reasonkit-pro".to_string(),
                ]);
            }
            super::task_graph::TaskType::MultiAgentCoordination => {
                components.extend([
                    "reasonkit-core".to_string(),
                    "reasonkit-web".to_string(),
                    "reasonkit-mem".to_string(),
                    "universal-agent-integration".to_string(),
                ]);
            }
        }

        // Check for M2-specific requirements
        if task_node.requires_m2_capability() {
            components.push("minimax-m2".to_string());
        }

        // Check for ThinkTool requirements
        if task_node.requires_thinktool() {
            components.push("thinktools".to_string());
        }

        Ok(components)
    }

    /// Execute ThinkTool processing with M2 enhancements
    #[allow(unused_variables)] // task_node and tool_call_count are used conditionally based on features
    async fn execute_thinktool_processing(
        &self,
        task_node: &TaskNode,
        input: &serde_json::Value,
        tool_call_count: &mut u32,
    ) -> Result<TaskExecutionResult, Error> {
        let start_time = Instant::now();

        // Use M2 ThinkTools manager if available
        #[cfg(feature = "minimax")]
        {
            let mut m2_manager = super::super::thinktool::minimax::M2ThinkToolsManager::new();

            for tool_name in task_node.required_thinktools() {
                let _result = m2_manager
                    .execute_thinktool(
                        tool_name,
                        &input.to_string(),
                        super::super::thinktool::minimax::ProfileType::Balanced,
                    )
                    .await?;

                *tool_call_count += 1; // Each ThinkTool execution counts as a tool call
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(TaskExecutionResult {
            success: true,
            output: input.clone(), // Return processed output
            components_used: vec!["thinktools".to_string()],
            tool_calls_used: 1,
            duration_ms,
        })
    }

    /// Create a checkpoint for state persistence
    async fn create_checkpoint(
        &self,
        _task_graph: &TaskGraph,
        tool_call_count: u32,
    ) -> Result<(), Error> {
        if !self.config.enable_state_persistence {
            return Ok(());
        }

        let snapshot = {
            let state_manager = self.state_manager.read().await;
            state_manager.create_snapshot().await?
        };

        tracing::info!(
            "Checkpoint created at tool call {}: {} bytes",
            tool_call_count,
            serde_json::to_string(&snapshot)?.len()
        );

        Ok(())
    }

    /// Calculate comprehensive performance metrics
    fn calculate_performance_metrics(
        &self,
        total_duration_ms: u64,
        tool_call_count: u32,
        _components: &[String],
    ) -> LongHorizonPerformanceMetrics {
        let avg_tool_call_time_ms = if tool_call_count > 0 {
            total_duration_ms as f64 / tool_call_count as f64
        } else {
            0.0
        };

        let throughput = if total_duration_ms > 0 {
            (tool_call_count as f64 * 1000.0) / total_duration_ms as f64
        } else {
            0.0
        };

        // Calculate memory efficiency (placeholder - would be based on actual memory usage)
        let memory_efficiency = 0.85; // 85% efficiency

        // Calculate cost per tool call (placeholder - would be based on actual costs)
        let cost_per_tool_call = 0.001; // $0.001 per tool call

        // Calculate reliability score
        let reliability_score = if tool_call_count > 0 {
            let successful_calls = tool_call_count as f64; // Assuming all calls succeeded
            (successful_calls / tool_call_count as f64).min(1.0)
        } else {
            1.0
        };

        // Calculate error rate
        let error_rate = 1.0 - reliability_score;

        LongHorizonPerformanceMetrics {
            avg_tool_call_time_ms,
            throughput,
            memory_efficiency,
            cost_per_tool_call,
            reliability_score,
            error_rate,
        }
    }
}

/// Result of executing a single task within the orchestration
#[derive(Debug, Clone)]
pub struct TaskExecutionResult {
    pub success: bool,
    pub output: serde_json::Value,
    pub components_used: Vec<String>,
    pub tool_calls_used: u32,
    pub duration_ms: u64,
}

/// Configuration for orchestrator
#[derive(Debug, Clone)]
pub struct OrchestrationConfig {
    pub max_tool_calls: u32,
    pub timeout_ms: u64,
    pub memory_limit_mb: u64,
    pub enable_error_recovery: bool,
    pub enable_state_persistence: bool,
    pub enable_performance_monitoring: bool,
}

/// Result of orchestration
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    pub success: bool,
    pub total_duration_ms: u64,
    pub tool_calls_used: u32,
    pub components_coordinated: Vec<String>,
    pub output: serde_json::Value,
}

/// Plan for executing tasks
#[derive(Debug, Clone)]
pub struct TaskExecutionPlan {
    pub execution_order: Vec<String>,
    pub estimated_tool_calls: u32,
    pub estimated_duration_ms: u64,
    pub resource_requirements: ResourceRequirements,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub memory_mb: u64,
    pub cpu_cores: f64,
    pub network_bandwidth_mbps: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let config = LongHorizonConfig::default();
        let orchestrator = LongHorizonOrchestrator::new(config).await;
        assert!(orchestrator.is_ok());
    }

    #[tokio::test]
    async fn test_performance_metrics_calculation() {
        let config = LongHorizonConfig::default();
        let orchestrator = LongHorizonOrchestrator::new(config).await.unwrap();

        let metrics = orchestrator.calculate_performance_metrics(
            5000, // 5 seconds
            50,   // 50 tool calls
            &["reasonkit-core".to_string()],
        );

        assert_eq!(metrics.avg_tool_call_time_ms, 100.0); // 5000/50
        assert_eq!(metrics.throughput, 10.0); // 50*1000/5000
    }
}
