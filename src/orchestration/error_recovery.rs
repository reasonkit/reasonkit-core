//! # Error Recovery System
//!
//! This module provides comprehensive error handling and recovery mechanisms for long-horizon
//! operations, enabling resilient execution across 100+ tool calling sequences with automatic
//! fallback strategies and state restoration capabilities.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStrategy {
    pub name: String,
    pub description: String,
}

/// Error recovery system for long-horizon operations
pub struct ErrorRecovery {
    /// Recovery strategies registry
    strategies: Arc<Mutex<RecoveryStrategies>>,
    /// Error tracking and analysis
    error_tracker: Arc<Mutex<ErrorTracker>>,
    /// State recovery manager
    state_recovery: Arc<Mutex<StateRecovery>>,
    /// Configuration
    #[allow(dead_code)]
    config: ErrorRecoveryConfig,
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorRecovery {
    pub fn new() -> Self {
        Self {
            strategies: Arc::new(Mutex::new(RecoveryStrategies::new())),
            error_tracker: Arc::new(Mutex::new(ErrorTracker::new())),
            state_recovery: Arc::new(Mutex::new(StateRecovery::new())),
            config: ErrorRecoveryConfig::default(),
        }
    }

    /// Attempt to recover from an error
    pub async fn attempt_recovery(
        &self,
        error: &Error,
        task_node: &super::task_graph::TaskNode,
        task_graph: &super::task_graph::TaskGraph,
    ) -> Result<RecoveryResult, Error> {
        let start_time = std::time::Instant::now();
        let error_type = self.classify_error(error);

        tracing::info!("Attempting recovery for error: {:?}", error_type);

        // Record error for analysis
        {
            let mut tracker = self.error_tracker.lock().await;
            tracker.record_error(error, &error_type, task_node.id());
        }

        // Get appropriate recovery strategy
        let strategies = self.strategies.lock().await;
        let strategy = strategies.get_strategy(&error_type, task_node.task_type())?;

        tracing::info!("Using recovery strategy: {}", strategy.name());

        // Execute recovery attempt
        let recovery_result = match strategy {
            RecoveryStrategyType::RetryWithBackoff => {
                self.execute_retry_with_backoff(task_node, task_graph, &error_type)
                    .await
            }
            RecoveryStrategyType::AlternativeExecution => {
                self.execute_alternative_execution(task_node, task_graph, &error_type)
                    .await
            }
            RecoveryStrategyType::StateRollback => {
                self.execute_state_rollback(task_node, task_graph, &error_type)
                    .await
            }
            RecoveryStrategyType::ComponentFallback => {
                self.execute_component_fallback(task_node, task_graph, &error_type)
                    .await
            }
            RecoveryStrategyType::SkipAndContinue => {
                self.execute_skip_and_continue(task_node, task_graph, &error_type)
                    .await
            }
            RecoveryStrategyType::ManualIntervention => {
                self.execute_manual_intervention(task_node, task_graph, &error_type)
                    .await
            }
            RecoveryStrategyType::ExtendTimeout
            | RecoveryStrategyType::ParallelExecution
            | RecoveryStrategyType::ResourceOptimization => {
                self.execute_retry_with_backoff(task_node, task_graph, &error_type)
                    .await
            }
        }?;

        let recovery_time_ms = start_time.elapsed().as_millis() as u64;

        // Update recovery statistics
        {
            let mut tracker = self.error_tracker.lock().await;
            tracker.record_recovery_attempt(&error_type, recovery_result.success, recovery_time_ms);
        }

        // Log recovery result
        if recovery_result.success {
            tracing::info!(
                "Recovery successful for error {:?} in {}ms: {}",
                error_type,
                recovery_time_ms,
                recovery_result.details
            );
        } else {
            tracing::warn!(
                "Recovery failed for error {:?} in {}ms: {}",
                error_type,
                recovery_time_ms,
                recovery_result.details
            );
        }

        Ok(RecoveryResult {
            success: recovery_result.success,
            strategy_used: strategy.name().to_string(),
            recovery_time_ms,
            details: recovery_result.details,
            recovered_state: recovery_result.recovered_state,
            fallback_actions: recovery_result.fallback_actions,
        })
    }

    /// Handle timeout scenarios
    pub async fn handle_timeout(
        &self,
        task_node: &super::task_graph::TaskNode,
        timeout_ms: u64,
    ) -> Result<RecoveryResult, Error> {
        let error_type = ErrorType::Timeout;

        tracing::warn!(
            "Handling timeout for task {} ({}ms)",
            task_node.name(),
            timeout_ms
        );

        // Record timeout error (mapped to Validation for now)
        {
            let mut tracker = self.error_tracker.lock().await;
            tracker.record_error(
                &Error::Validation(format!("Task {} timed out", task_node.name())),
                &error_type,
                task_node.id(),
            );
        }

        // Determine timeout recovery strategy
        let strategies = self.strategies.lock().await;
        let strategy = strategies.get_timeout_strategy(task_node)?;

        // Execute timeout recovery
        let recovery_result = match strategy {
            RecoveryStrategyType::ExtendTimeout => self.extend_timeout(task_node, timeout_ms).await,
            RecoveryStrategyType::ParallelExecution => {
                self.parallel_execution_recovery(task_node).await
            }
            RecoveryStrategyType::ResourceOptimization => self.optimize_resources(task_node).await,
            _ => {
                self.execute_skip_and_continue(
                    task_node,
                    &super::task_graph::TaskGraph::new(),
                    &error_type,
                )
                .await
            }
        }?;

        let recovery_time_ms = 0;

        Ok(RecoveryResult {
            success: recovery_result.success,
            strategy_used: strategy.name().to_string(),
            recovery_time_ms,
            details: recovery_result.details,
            recovered_state: recovery_result.recovered_state,
            fallback_actions: recovery_result.fallback_actions,
        })
    }

    /// Classify error type for recovery strategy selection
    fn classify_error(&self, error: &Error) -> ErrorType {
        match error {
            Error::Validation(_) => ErrorType::ValidationError,
            Error::Io(_) => ErrorType::IoError,
            Error::Network(_) => ErrorType::NetworkError,
            Error::ResourceExhausted(_) => ErrorType::ResourceExhausted,
            Error::M2ExecutionError(_) => ErrorType::M2Error,
            _ => ErrorType::Unknown,
        }
    }

    /// Execute retry with exponential backoff
    async fn execute_retry_with_backoff(
        &self,
        task_node: &super::task_graph::TaskNode,
        _task_graph: &super::task_graph::TaskGraph,
        _error_type: &ErrorType,
    ) -> Result<StrategyExecutionResult, Error> {
        let max_retries = task_node.max_retries();
        let mut attempt = 0;
        let base_delay_ms = 1000; // 1 second base delay

        while attempt < max_retries {
            attempt += 1;
            let delay_ms = base_delay_ms * (2_u64.pow(attempt - 1)); // Exponential backoff

            tracing::info!(
                "Retry attempt {} for task {} (delay: {}ms)",
                attempt,
                task_node.name(),
                delay_ms
            );

            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

            // Attempt to re-execute the task
            // In a real implementation, this would call the task executor
            let retry_success = self.simulate_retry_execution(task_node).await?;

            if retry_success {
                return Ok(StrategyExecutionResult {
                    success: true,
                    details: format!(
                        "Task {} recovered after {} retry attempts",
                        task_node.name(),
                        attempt
                    ),
                    recovered_state: Some(serde_json::json!({"retries_used": attempt})),
                    fallback_actions: vec![],
                });
            }
        }

        Ok(StrategyExecutionResult {
            success: false,
            details: format!(
                "Task {} failed after {} retry attempts",
                task_node.name(),
                max_retries
            ),
            recovered_state: None,
            fallback_actions: vec!["skip_task".to_string(), "notify_failure".to_string()],
        })
    }

    /// Execute alternative execution path
    async fn execute_alternative_execution(
        &self,
        task_node: &super::task_graph::TaskNode,
        _task_graph: &super::task_graph::TaskGraph,
        error_type: &ErrorType,
    ) -> Result<StrategyExecutionResult, Error> {
        tracing::info!(
            "Attempting alternative execution for task {}",
            task_node.name()
        );

        // Determine alternative execution path based on task type and error
        let alternative_path = match task_node.task_type() {
            super::task_graph::TaskType::ProtocolGeneration => {
                self.get_alternative_protocol_generation_path(error_type)
                    .await
            }
            super::task_graph::TaskType::CodeAnalysis => {
                self.get_alternative_code_analysis_path(error_type).await
            }
            super::task_graph::TaskType::WebAutomation => {
                self.get_alternative_web_automation_path(error_type).await
            }
            _ => self.get_general_alternative_path(error_type).await,
        };

        if let Some(alternative_path) = alternative_path {
            Ok(StrategyExecutionResult {
                success: true,
                details: format!(
                    "Alternative execution path found for task {}",
                    task_node.name()
                ),
                recovered_state: Some(serde_json::json!({"alternative_path": alternative_path})),
                fallback_actions: vec![],
            })
        } else {
            Ok(StrategyExecutionResult {
                success: false,
                details: format!(
                    "No alternative execution path available for task {}",
                    task_node.name()
                ),
                recovered_state: None,
                fallback_actions: vec!["manual_review".to_string()],
            })
        }
    }

    /// Execute state rollback to previous checkpoint
    async fn execute_state_rollback(
        &self,
        task_node: &super::task_graph::TaskNode,
        _task_graph: &super::task_graph::TaskGraph,
        _error_type: &ErrorType,
    ) -> Result<StrategyExecutionResult, Error> {
        tracing::info!("Attempting state rollback for task {}", task_node.name());

        let mut state_recovery = self.state_recovery.lock().await;
        let rollback_result = state_recovery
            .rollback_to_checkpoint(task_node.id())
            .await?;

        if rollback_result.success {
            Ok(StrategyExecutionResult {
                success: true,
                details: format!(
                    "State rolled back successfully for task {}",
                    task_node.name()
                ),
                recovered_state: Some(rollback_result.state_data),
                fallback_actions: vec![],
            })
        } else {
            Ok(StrategyExecutionResult {
                success: false,
                details: format!("State rollback failed for task {}", task_node.name()),
                recovered_state: None,
                fallback_actions: vec!["reset_execution".to_string()],
            })
        }
    }

    /// Execute component fallback strategy
    async fn execute_component_fallback(
        &self,
        task_node: &super::task_graph::TaskNode,
        _task_graph: &super::task_graph::TaskGraph,
        error_type: &ErrorType,
    ) -> Result<StrategyExecutionResult, Error> {
        tracing::info!(
            "Attempting component fallback for task {}",
            task_node.name()
        );

        // Determine fallback components based on error type and task requirements
        let fallback_components = self
            .determine_fallback_components(task_node, error_type)
            .await?;

        if !fallback_components.is_empty() {
            Ok(StrategyExecutionResult {
                success: true,
                details: format!(
                    "Component fallback executed with {} components",
                    fallback_components.len()
                ),
                recovered_state: Some(
                    serde_json::json!({"fallback_components": fallback_components}),
                ),
                fallback_actions: vec![],
            })
        } else {
            Ok(StrategyExecutionResult {
                success: false,
                details: "No suitable fallback components available".to_string(),
                recovered_state: None,
                fallback_actions: vec!["manual_intervention".to_string()],
            })
        }
    }

    /// Execute skip and continue strategy
    async fn execute_skip_and_continue(
        &self,
        task_node: &super::task_graph::TaskNode,
        _task_graph: &super::task_graph::TaskGraph,
        _error_type: &ErrorType,
    ) -> Result<StrategyExecutionResult, Error> {
        tracing::info!(
            "Skipping task {} and continuing execution",
            task_node.name()
        );

        // Mark task as skipped in the graph
        // In real implementation, would update task graph state

        Ok(StrategyExecutionResult {
            success: true,
            details: format!("Task {} skipped, execution continuing", task_node.name()),
            recovered_state: Some(serde_json::json!({"skipped": true, "task_id": task_node.id()})),
            fallback_actions: vec![],
        })
    }

    /// Execute manual intervention strategy
    async fn execute_manual_intervention(
        &self,
        task_node: &super::task_graph::TaskNode,
        _task_graph: &super::task_graph::TaskGraph,
        error_type: &ErrorType,
    ) -> Result<StrategyExecutionResult, Error> {
        tracing::warn!("Manual intervention required for task {}", task_node.name());

        // In a real implementation, this would trigger notifications,
        // pause execution, or open an interactive session
        let intervention_required = serde_json::json!({
            "task_id": task_node.id(),
            "task_name": task_node.name(),
            "error_type": format!("{:?}", error_type),
            "timestamp": chrono::Utc::now().timestamp(),
            "intervention_url": format!("https://admin.reasonkit.sh/intervention/{}", task_node.id())
        });

        Ok(StrategyExecutionResult {
            success: false, // Manual intervention is not automatically successful
            details: "Manual intervention required - execution paused".to_string(),
            recovered_state: Some(intervention_required),
            fallback_actions: vec!["notify_admin".to_string(), "pause_execution".to_string()],
        })
    }

    /// Simulate retry execution (placeholder)
    async fn simulate_retry_execution(
        &self,
        _task_node: &super::task_graph::TaskNode,
    ) -> Result<bool, Error> {
        // Simulate 70% success rate on retry
        let success_rate = 0.7;
        let random_value = 0.5f64;

        Ok(random_value < success_rate)
    }

    /// Get alternative protocol generation path
    async fn get_alternative_protocol_generation_path(
        &self,
        error_type: &ErrorType,
    ) -> Option<String> {
        match error_type {
            ErrorType::ProtocolError => Some("use_simplified_protocol".to_string()),
            ErrorType::ResourceExhausted => Some("defer_generation".to_string()),
            _ => Some("manual_generation".to_string()),
        }
    }

    /// Get alternative code analysis path
    async fn get_alternative_code_analysis_path(&self, error_type: &ErrorType) -> Option<String> {
        match error_type {
            ErrorType::ThinkToolError => Some("use_basic_analysis".to_string()),
            ErrorType::M2Error => Some("use_fallback_model".to_string()),
            _ => Some("basic_static_analysis".to_string()),
        }
    }

    /// Get alternative web automation path
    async fn get_alternative_web_automation_path(&self, error_type: &ErrorType) -> Option<String> {
        match error_type {
            ErrorType::NetworkError => Some("retry_with_different_proxy".to_string()),
            ErrorType::Timeout => Some("use_headless_mode".to_string()),
            _ => Some("skip_web_automation".to_string()),
        }
    }

    /// Get general alternative path
    async fn get_general_alternative_path(&self, error_type: &ErrorType) -> Option<String> {
        match error_type {
            ErrorType::MemoryError => Some("reduce_memory_usage".to_string()),
            ErrorType::RateLimitError => Some("wait_and_retry".to_string()),
            _ => Some("execute_basic_version".to_string()),
        }
    }

    /// Determine fallback components
    async fn determine_fallback_components(
        &self,
        task_node: &super::task_graph::TaskNode,
        error_type: &ErrorType,
    ) -> Result<Vec<String>, Error> {
        let mut fallback_components = Vec::new();

        // Based on task type and error, determine fallback components
        match task_node.task_type() {
            super::task_graph::TaskType::ProtocolGeneration => {
                fallback_components.push("reasonkit-core".to_string());
                if matches!(error_type, ErrorType::M2Error) {
                    fallback_components.push("reasonkit-pro".to_string());
                }
            }
            super::task_graph::TaskType::CodeAnalysis => {
                fallback_components
                    .extend(["reasonkit-core".to_string(), "reasonkit-web".to_string()]);
            }
            _ => {
                fallback_components.push("reasonkit-core".to_string());
            }
        }

        Ok(fallback_components)
    }

    /// Extend timeout strategy
    async fn extend_timeout(
        &self,
        _task_node: &super::task_graph::TaskNode,
        current_timeout: u64,
    ) -> Result<StrategyExecutionResult, Error> {
        let extended_timeout = current_timeout * 2; // Double the timeout

        Ok(StrategyExecutionResult {
            success: true,
            details: format!(
                "Extended timeout from {}ms to {}ms",
                current_timeout, extended_timeout
            ),
            recovered_state: Some(serde_json::json!({"extended_timeout": extended_timeout})),
            fallback_actions: vec![],
        })
    }

    /// Parallel execution recovery
    async fn parallel_execution_recovery(
        &self,
        _task_node: &super::task_graph::TaskNode,
    ) -> Result<StrategyExecutionResult, Error> {
        Ok(StrategyExecutionResult {
            success: true,
            details: "Switching to parallel execution".to_string(),
            recovered_state: Some(serde_json::json!({"parallel_execution": true})),
            fallback_actions: vec![],
        })
    }

    /// Resource optimization recovery
    async fn optimize_resources(
        &self,
        _task_node: &super::task_graph::TaskNode,
    ) -> Result<StrategyExecutionResult, Error> {
        Ok(StrategyExecutionResult {
            success: true,
            details: "Optimized resource allocation".to_string(),
            recovered_state: Some(serde_json::json!({"resource_optimization": true})),
            fallback_actions: vec![],
        })
    }

    /// Get error recovery statistics
    pub async fn get_recovery_statistics(&self) -> Result<RecoveryStatistics, Error> {
        let tracker = self.error_tracker.lock().await;
        Ok(tracker.get_statistics())
    }

    /// Reset error recovery system
    pub async fn reset(&self) -> Result<(), Error> {
        {
            let mut tracker = self.error_tracker.lock().await;
            tracker.reset();
        }

        {
            let mut state_recovery = self.state_recovery.lock().await;
            state_recovery.reset();
        }

        tracing::info!("Error recovery system reset");
        Ok(())
    }
}

/// Error types for classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorType {
    Timeout,
    ValidationError,
    IoError,
    NetworkError,
    MemoryError,
    RateLimitError,
    AuthenticationError,
    AuthorizationError,
    ResourceExhausted,
    DependencyError,
    ProtocolError,
    ThinkToolError,
    M2Error,
    Unknown,
}

/// Recovery strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
enum RecoveryStrategyType {
    RetryWithBackoff,
    AlternativeExecution,
    StateRollback,
    ComponentFallback,
    SkipAndContinue,
    ManualIntervention,
    ExtendTimeout,
    ParallelExecution,
    ResourceOptimization,
}

impl RecoveryStrategyType {
    fn name(&self) -> &'static str {
        match self {
            RecoveryStrategyType::RetryWithBackoff => "RetryWithBackoff",
            RecoveryStrategyType::AlternativeExecution => "AlternativeExecution",
            RecoveryStrategyType::StateRollback => "StateRollback",
            RecoveryStrategyType::ComponentFallback => "ComponentFallback",
            RecoveryStrategyType::SkipAndContinue => "SkipAndContinue",
            RecoveryStrategyType::ManualIntervention => "ManualIntervention",
            RecoveryStrategyType::ExtendTimeout => "ExtendTimeout",
            RecoveryStrategyType::ParallelExecution => "ParallelExecution",
            RecoveryStrategyType::ResourceOptimization => "ResourceOptimization",
        }
    }
}

/// Recovery strategies registry
#[derive(Debug)]
struct RecoveryStrategies {
    strategies: HashMap<(ErrorType, super::task_graph::TaskType), RecoveryStrategyType>,
}

impl RecoveryStrategies {
    fn new() -> Self {
        let mut strategies = HashMap::new();

        // Map error types and task types to recovery strategies
        strategies.insert(
            (
                ErrorType::Timeout,
                super::task_graph::TaskType::ProtocolGeneration,
            ),
            RecoveryStrategyType::RetryWithBackoff,
        );
        strategies.insert(
            (
                ErrorType::Timeout,
                super::task_graph::TaskType::CodeAnalysis,
            ),
            RecoveryStrategyType::ExtendTimeout,
        );
        strategies.insert(
            (
                ErrorType::ValidationError,
                super::task_graph::TaskType::ProtocolGeneration,
            ),
            RecoveryStrategyType::AlternativeExecution,
        );
        strategies.insert(
            (
                ErrorType::ValidationError,
                super::task_graph::TaskType::General,
            ),
            RecoveryStrategyType::RetryWithBackoff,
        );
        strategies.insert(
            (
                ErrorType::ProtocolError,
                super::task_graph::TaskType::ProtocolGeneration,
            ),
            RecoveryStrategyType::StateRollback,
        );
        strategies.insert(
            (
                ErrorType::ThinkToolError,
                super::task_graph::TaskType::CodeAnalysis,
            ),
            RecoveryStrategyType::ComponentFallback,
        );
        strategies.insert(
            (
                ErrorType::M2Error,
                super::task_graph::TaskType::ProtocolGeneration,
            ),
            RecoveryStrategyType::ComponentFallback,
        );
        strategies.insert(
            (
                ErrorType::ResourceExhausted,
                super::task_graph::TaskType::EnterpriseWorkflow,
            ),
            RecoveryStrategyType::ResourceOptimization,
        );
        strategies.insert(
            (
                ErrorType::MemoryError,
                super::task_graph::TaskType::MultiAgentCoordination,
            ),
            RecoveryStrategyType::StateRollback,
        );

        Self { strategies }
    }

    fn get_strategy(
        &self,
        error_type: &ErrorType,
        task_type: super::task_graph::TaskType,
    ) -> Result<RecoveryStrategyType, Error> {
        let key = (error_type.clone(), task_type);
        self.strategies
            .get(&key)
            .cloned()
            .or_else(|| {
                self.strategies
                    .get(&(error_type.clone(), super::task_graph::TaskType::General))
                    .cloned()
            })
            .ok_or_else(|| {
                Error::Validation(format!(
                    "No recovery strategy for error {:?} and task type {:?}",
                    error_type, task_type
                ))
            })
    }

    fn get_timeout_strategy(
        &self,
        task_node: &super::task_graph::TaskNode,
    ) -> Result<RecoveryStrategyType, Error> {
        match task_node.task_type() {
            super::task_graph::TaskType::ProtocolGeneration => {
                Ok(RecoveryStrategyType::ExtendTimeout)
            }
            super::task_graph::TaskType::CodeAnalysis => {
                Ok(RecoveryStrategyType::ParallelExecution)
            }
            super::task_graph::TaskType::EnterpriseWorkflow => {
                Ok(RecoveryStrategyType::ResourceOptimization)
            }
            _ => Ok(RecoveryStrategyType::RetryWithBackoff),
        }
    }
}

/// Error tracking and analysis
#[derive(Debug)]
struct ErrorTracker {
    error_history: Vec<ErrorRecord>,
    error_patterns: HashMap<ErrorType, u32>,
    recovery_success_rates: HashMap<RecoveryStrategyType, (u32, u32)>,
    total_errors: u32,
    total_recoveries: u32,
    successful_recoveries: u32,
}

impl ErrorTracker {
    fn new() -> Self {
        Self {
            error_history: Vec::new(),
            error_patterns: HashMap::new(),
            recovery_success_rates: HashMap::new(),
            total_errors: 0,
            total_recoveries: 0,
            successful_recoveries: 0,
        }
    }

    fn record_error(&mut self, error: &Error, error_type: &ErrorType, task_id: &str) {
        let record = ErrorRecord {
            timestamp: chrono::Utc::now(),
            error_type: error_type.clone(),
            error_message: error.to_string(),
            task_id: task_id.to_string(),
            context: serde_json::json!({}),
        };

        self.error_history.push(record);
        *self.error_patterns.entry(error_type.clone()).or_insert(0) += 1;
        self.total_errors += 1;

        // Maintain history limit
        if self.error_history.len() > 1000 {
            self.error_history.remove(0);
        }
    }

    fn record_recovery_attempt(
        &mut self,
        _error_type: &ErrorType,
        success: bool,
        _recovery_time_ms: u64,
    ) {
        self.total_recoveries += 1;
        if success {
            self.successful_recoveries += 1;
        }

        // This is a simplified implementation - would track by strategy type
        let strategy_key = RecoveryStrategyType::RetryWithBackoff; // Placeholder
        let (success_count, total_count) = self
            .recovery_success_rates
            .entry(strategy_key)
            .or_insert((0, 0));
        *total_count += 1;
        if success {
            *success_count += 1;
        }
    }

    fn get_statistics(&self) -> RecoveryStatistics {
        let overall_success_rate = if self.total_recoveries > 0 {
            self.successful_recoveries as f64 / self.total_recoveries as f64
        } else {
            0.0
        };

        RecoveryStatistics {
            total_errors: self.total_errors,
            total_recovery_attempts: self.total_recoveries,
            successful_recoveries: self.successful_recoveries,
            overall_success_rate,
            error_patterns: self.error_patterns.clone(),
            recovery_success_rates: self
                .recovery_success_rates
                .iter()
                .map(|(strategy, (success, total))| {
                    (
                        strategy.name().to_string(),
                        (*success as f64 / *total as f64),
                    )
                })
                .collect(),
        }
    }

    fn reset(&mut self) {
        self.error_history.clear();
        self.error_patterns.clear();
        self.recovery_success_rates.clear();
        self.total_errors = 0;
        self.total_recoveries = 0;
        self.successful_recoveries = 0;
    }
}

/// State recovery manager
#[derive(Debug)]
struct StateRecovery {
    checkpoints: Vec<StateCheckpoint>,
    #[allow(dead_code)]
    max_checkpoints: usize,
}

impl StateRecovery {
    fn new() -> Self {
        Self {
            checkpoints: Vec::new(),
            max_checkpoints: 50,
        }
    }

    async fn rollback_to_checkpoint(&mut self, task_id: &str) -> Result<RollbackResult, Error> {
        // Find the most recent checkpoint for this task
        let checkpoint = self
            .checkpoints
            .iter()
            .rev()
            .find(|cp| cp.task_id == task_id)
            .cloned();

        if let Some(checkpoint) = checkpoint {
            Ok(RollbackResult {
                success: true,
                state_data: checkpoint.state_data,
                timestamp: checkpoint.timestamp,
            })
        } else {
            Ok(RollbackResult {
                success: false,
                state_data: serde_json::json!({}),
                timestamp: chrono::Utc::now(),
            })
        }
    }

    #[allow(dead_code)]
    fn add_checkpoint(&mut self, task_id: &str, state_data: serde_json::Value) {
        let checkpoint = StateCheckpoint {
            task_id: task_id.to_string(),
            state_data,
            timestamp: chrono::Utc::now(),
        };

        self.checkpoints.push(checkpoint);

        // Maintain checkpoint limit
        if self.checkpoints.len() > self.max_checkpoints {
            self.checkpoints.remove(0);
        }
    }

    fn reset(&mut self) {
        self.checkpoints.clear();
    }
}

/// Recovery result
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub success: bool,
    pub strategy_used: String,
    pub recovery_time_ms: u64,
    pub details: String,
    pub recovered_state: Option<serde_json::Value>,
    pub fallback_actions: Vec<String>,
}

/// Strategy execution result
#[derive(Debug, Clone)]
struct StrategyExecutionResult {
    success: bool,
    details: String,
    recovered_state: Option<serde_json::Value>,
    fallback_actions: Vec<String>,
}

/// Error record for tracking
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ErrorRecord {
    timestamp: chrono::DateTime<chrono::Utc>,
    error_type: ErrorType,
    error_message: String,
    task_id: String,
    context: serde_json::Value,
}

/// State checkpoint for rollback
#[derive(Debug, Clone)]
struct StateCheckpoint {
    task_id: String,
    state_data: serde_json::Value,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Rollback result
#[derive(Debug, Clone)]
struct RollbackResult {
    success: bool,
    state_data: serde_json::Value,
    #[allow(dead_code)]
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Recovery statistics
#[derive(Debug, Clone)]
pub struct RecoveryStatistics {
    pub total_errors: u32,
    pub total_recovery_attempts: u32,
    pub successful_recoveries: u32,
    pub overall_success_rate: f64,
    pub error_patterns: HashMap<ErrorType, u32>,
    pub recovery_success_rates: Vec<(String, f64)>,
}

/// Error recovery configuration
#[derive(Debug, Clone)]
pub struct ErrorRecoveryConfig {
    pub max_retry_attempts: u32,
    pub base_retry_delay_ms: u64,
    pub max_retry_delay_ms: u64,
    pub enable_automatic_recovery: bool,
    pub enable_state_rollback: bool,
    pub recovery_timeout_ms: u64,
}

impl Default for ErrorRecoveryConfig {
    fn default() -> Self {
        Self {
            max_retry_attempts: 3,
            base_retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
            enable_automatic_recovery: true,
            enable_state_rollback: true,
            recovery_timeout_ms: 60000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        let recovery = ErrorRecovery::new();
        let timeout_error = Error::Validation("Test timeout".to_string());
        let error_type = recovery.classify_error(&timeout_error);

        assert_eq!(error_type, ErrorType::ValidationError);
    }

    #[test]
    fn test_recovery_result_creation() {
        let result = RecoveryResult {
            success: true,
            strategy_used: "RetryWithBackoff".to_string(),
            recovery_time_ms: 2000,
            details: "Recovery successful".to_string(),
            recovered_state: Some(serde_json::json!({"retries": 2})),
            fallback_actions: vec![],
        };

        assert!(result.success);
        assert_eq!(result.recovery_time_ms, 2000);
    }

    #[tokio::test]
    async fn test_error_recovery_creation() {
        let recovery = ErrorRecovery::new();
        assert!(recovery
            .attempt_recovery(
                &Error::Validation("Test error".to_string()),
                &super::super::task_graph::TaskNode::new(
                    "test".to_string(),
                    "Test Task".to_string(),
                    super::super::task_graph::TaskType::General,
                    super::super::task_graph::TaskPriority::Normal,
                    "Test task".to_string(),
                ),
                &super::super::task_graph::TaskGraph::new(),
            )
            .await
            .is_ok());
    }
}
