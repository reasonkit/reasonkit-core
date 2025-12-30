//! # MiniMax M2 Integration Module
//!
//! Complete integration of MiniMax M2's Agent-Native Architecture with ReasonKit Core.
//! Provides the Interleaved Thinking Protocol Engine for autonomous reasoning protocols.

pub mod types;
pub mod connector;
pub mod engine;
pub mod protocol_generator;
pub mod benchmarks;

// Re-export main types and components
pub use types::*;
pub use connector::{M2Connector, TokenTracker, TokenUsageStats, RateLimiter};
pub use engine::ThinkingOrchestrator;
pub use protocol_generator::{ProtocolGenerator, TaskClassification, OptimizationGoals};
pub use benchmarks::{M2BenchmarkSuite, BenchmarkReport, TestScenario};

use crate::error::Error;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use tracing::{info, warn, error, instrument};
use anyhow::Result;

/// Main M2 Integration Service
#[derive(Debug)]
pub struct M2IntegrationService {
    /// M2 API connector
    connector: Arc<M2Connector>,
    
    /// Interleaved thinking orchestrator
    orchestrator: Arc<ThinkingOrchestrator>,
    
    /// Protocol generator
    protocol_generator: Arc<ProtocolGenerator>,
    
    /// Active executions
    active_executions: Arc<RwLock<HashMap<String, ActiveExecution>>>,
    
    /// Configuration
    config: M2IntegrationConfig,
}

/// Configuration for M2 integration service
#[derive(Debug, Clone)]
pub struct M2IntegrationConfig {
    /// Maximum concurrent executions
    pub max_concurrent_executions: u32,
    
    /// Default timeout for executions
    pub default_timeout_ms: u64,
    
    /// Enable caching
    pub enable_caching: bool,
    
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    
    /// Default optimization goals
    pub default_optimization_goals: OptimizationGoals,
}

/// Active execution tracking
#[derive(Debug)]
struct ActiveExecution {
    execution_id: String,
    protocol_id: String,
    start_time: std::time::Instant,
    status: ExecutionStatus,
    progress: f64,
}

/// M2 Integration Service Implementation
impl M2IntegrationService {
    /// Create new M2 integration service
    pub async fn new(config: M2Config, integration_config: M2IntegrationConfig) -> Result<Self, Error> {
        info!("Initializing M2 Integration Service");
        
        // Initialize M2 connector
        let connector = Arc::new(M2Connector::new(config).await?);
        
        // Initialize thinking orchestrator
        let orchestrator = Arc::new(ThinkingOrchestrator::new(connector.clone()));
        
        // Initialize protocol generator
        let protocol_generator = Arc::new(ProtocolGenerator::new()?);
        
        let active_executions = Arc::new(RwLock::new(HashMap::new()));
        
        Ok(Self {
            connector,
            orchestrator,
            protocol_generator,
            active_executions,
            config: integration_config,
        })
    }
    
    /// Execute interleaved thinking protocol
    #[instrument(skip(self, input))]
    pub async fn execute_interleaved_thinking(
        &self,
        framework: AgentFramework,
        task: TaskClassification,
        input: ProtocolInput,
        custom_constraints: Option<CompositeConstraints>,
        custom_goals: Option<OptimizationGoals>,
    ) -> Result<InterleavedResult, Error> {
        let execution_id = Uuid::new_v4().to_string();
        
        info!(
            "Starting interleaved thinking execution: {} (framework: {:?})",
            execution_id, framework
        );
        
        // Check concurrent execution limit
        self.check_concurrent_limit().await?;
        
        // Add to active executions
        {
            let mut active_executions = self.active_executions.write().await;
            active_executions.insert(execution_id.clone(), ActiveExecution {
                execution_id: execution_id.clone(),
                protocol_id: task.task_type.to_string(),
                start_time: std::time::Instant::now(),
                status: ExecutionStatus::Completed, // Will be updated
                progress: 0.0,
            });
        }
        
        // Step 1: Generate optimized protocol
        let protocol = self.protocol_generator
            .generate_protocol(&framework, &task, &self.get_default_constraints()?, &custom_goals.unwrap_or_else(|| self.config.default_optimization_goals.clone()))?;
        
        // Step 2: Apply custom constraints if provided
        let constraints = if let Some(custom) = custom_constraints {
            self.merge_constraints(&self.get_default_constraints()?, &custom)?
        } else {
            self.get_default_constraints()?
        };
        
        // Step 3: Execute with orchestrator
        let result = self.orchestrator
            .execute_interleaved_thinking(&protocol, &constraints, &input)
            .await?;
        
        // Update execution status
        {
            let mut active_executions = self.active_executions.write().await;
            if let Some(execution) = active_executions.get_mut(&execution_id) {
                execution.status = ExecutionStatus::Completed;
                execution.progress = 1.0;
            }
        }
        
        info!(
            "Interleaved thinking completed: {} (cost reduction: {:.1}%, quality: {:.2})",
            execution_id, result.metrics.cost_metrics.cost_reduction_percent, result.metrics.quality_metrics.overall_quality
        );
        
        Ok(result)
    }
    
    /// Execute for specific use case (convenience method)
    #[instrument(skip(self))]
    pub async fn execute_for_use_case(
        &self,
        use_case: UseCase,
        input: ProtocolInput,
        framework: Option<AgentFramework>,
    ) -> Result<InterleavedResult, Error> {
        let framework = framework.unwrap_or(AgentFramework::ClaudeCode); // Default to Claude Code
        
        let task = self.classify_use_case(use_case, &input)?;
        
        self.execute_interleaved_thinking(framework, task, input, None, None).await
    }
    
    /// Get execution status
    pub async fn get_execution_status(&self, execution_id: &str) -> Result<Option<ExecutionStatusInfo>, Error> {
        let active_executions = self.active_executions.read().await;
        
        if let Some(execution) = active_executions.get(execution_id) {
            Ok(Some(ExecutionStatusInfo {
                execution_id: execution.execution_id.clone(),
                protocol_id: execution.protocol_id.clone(),
                status: execution.status.clone(),
                progress: execution.progress,
                elapsed_time: execution.start_time.elapsed(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// List active executions
    pub async fn list_active_executions(&self) -> Result<Vec<ExecutionStatusInfo>, Error> {
        let active_executions = self.active_executions.read().await;
        
        Ok(active_executions.values().map(|exec| ExecutionStatusInfo {
            execution_id: exec.execution_id.clone(),
            protocol_id: exec.protocol_id.clone(),
            status: exec.status.clone(),
            progress: exec.progress,
            elapsed_time: exec.start_time.elapsed(),
        }).collect())
    }
    
    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetricsSummary, Error> {
        let active_executions = self.active_executions.read().await;
        
        let total_executions = active_executions.len();
        let completed_executions = active_executions.values()
            .filter(|exec| matches!(exec.status, ExecutionStatus::Completed))
            .count();
        
        let avg_elapsed_time = if total_executions > 0 {
            active_executions.values()
                .map(|exec| exec.start_time.elapsed())
                .sum::<std::time::Duration>() / total_executions as u32
        } else {
            std::time::Duration::from_millis(0)
        };
        
        Ok(PerformanceMetricsSummary {
            total_executions,
            active_executions: total_executions - completed_executions,
            completed_executions,
            average_execution_time: avg_elapsed_time,
            success_rate: if total_executions > 0 {
                completed_executions as f64 / total_executions as f64
            } else {
                0.0
            },
        })
    }
    
    // Helper methods
    async fn check_concurrent_limit(&self) -> Result<(), Error> {
        let active_executions = self.active_executions.read().await;
        
        if active_executions.len() >= self.config.max_concurrent_executions as usize {
            return Err(Error::ResourceExhausted(format!(
                "Maximum concurrent executions ({}) reached",
                self.config.max_concurrent_executions
            )));
        }
        
        Ok(())
    }
    
    fn get_default_constraints(&self) -> Result<CompositeConstraints, Error> {
        Ok(CompositeConstraints {
            system_prompt: SystemPrompt {
                instruction: "You are an expert AI assistant executing systematic interleaved thinking protocols.".to_string(),
                reasoning_style: ReasoningStyle::Interleaved,
                output_format: OutputFormat::Structured,
                quality_standards: QualityStandards {
                    min_confidence: 0.85,
                    require_validation: true,
                    require_evidence: true,
                },
            },
            user_query: UserQuery {
                original: "User query".to_string(),
                clarified: "Clarified user query".to_string(),
                context_requirements: ContextRequirements::default(),
                expected_output: ExpectedOutput::default(),
            },
            memory_context: MemoryContext {
                historical_context: vec![],
                similar_cases: vec![],
                domain_knowledge: vec![],
                user_preferences: UserPreferences::default(),
            },
            tool_schemas: vec![],
            framework_constraints: FrameworkConstraints {
                framework: AgentFramework::ClaudeCode,
                optimizations: vec![],
                compatibility: CompatibilityRequirements::default(),
            },
        })
    }
    
    fn merge_constraints(
        &self,
        base: &CompositeConstraints,
        custom: &CompositeConstraints,
    ) -> Result<CompositeConstraints, Error> {
        let mut merged = base.clone();
        
        // Merge system prompt (custom overrides base)
        if !custom.system_prompt.instruction.is_empty() {
            merged.system_prompt.instruction = custom.system_prompt.instruction.clone();
        }
        
        // Merge user query (custom overrides base)
        if !custom.user_query.original.is_empty() {
            merged.user_query = custom.user_query.clone();
        }
        
        // Merge framework constraints
        if !custom.framework_constraints.optimizations.is_empty() {
            merged.framework_constraints.optimizations = custom.framework_constraints.optimizations.clone();
        }
        
        // Merge tool schemas
        if !custom.tool_schemas.is_empty() {
            merged.tool_schemas = custom.tool_schemas.clone();
        }
        
        Ok(merged)
    }
    
    fn classify_use_case(
        &self,
        use_case: UseCase,
        input: &ProtocolInput,
    ) -> Result<TaskClassification, Error> {
        match use_case {
            UseCase::CodeAnalysis => Ok(TaskClassification {
                task_type: TaskType::CodeAnalysis,
                complexity_level: ComplexityLevel::Complex,
                domain: TaskDomain::SystemProgramming,
                expected_output_size: OutputSize {
                    estimated_tokens: 5000,
                    complexity: ComplexityLevel::Complex,
                },
                time_constraints: TimeConstraints {
                    is_strict: false,
                    target_latency_ms: Some(10000),
                },
                quality_requirements: QualityRequirements {
                    level: QualityLevel::High,
                    critical_factors: vec!["accuracy".to_string(), "completeness".to_string()],
                },
            }),
            
            UseCase::BugFinding => Ok(TaskClassification {
                task_type: TaskType::BugFinding,
                complexity_level: ComplexityLevel::VeryComplex,
                domain: TaskDomain::SystemProgramming,
                expected_output_size: OutputSize {
                    estimated_tokens: 3000,
                    complexity: ComplexityLevel::Complex,
                },
                time_constraints: TimeConstraints {
                    is_strict: true,
                    target_latency_ms: Some(5000),
                },
                quality_requirements: QualityRequirements {
                    level: QualityLevel::Critical,
                    critical_factors: vec!["accuracy".to_string(), "false_positives".to_string()],
                },
            }),
            
            UseCase::Documentation => Ok(TaskClassification {
                task_type: TaskType::Documentation,
                complexity_level: ComplexityLevel::Moderate,
                domain: TaskDomain::General,
                expected_output_size: OutputSize {
                    estimated_tokens: 8000,
                    complexity: ComplexityLevel::Moderate,
                },
                time_constraints: TimeConstraints {
                    is_strict: false,
                    target_latency_ms: Some(15000),
                },
                quality_requirements: QualityRequirements {
                    level: QualityLevel::High,
                    critical_factors: vec!["completeness".to_string(), "clarity".to_string()],
                },
            }),
            
            UseCase::Testing => Ok(TaskClassification {
                task_type: TaskType::Testing,
                complexity_level: ComplexityLevel::Complex,
                domain: TaskDomain::SystemProgramming,
                expected_output_size: OutputSize {
                    estimated_tokens: 4000,
                    complexity: ComplexityLevel::Complex,
                },
                time_constraints: TimeConstraints {
                    is_strict: false,
                    target_latency_ms: Some(8000),
                },
                quality_requirements: QualityRequirements {
                    level: QualityLevel::High,
                    critical_factors: vec!["coverage".to_string(), "accuracy".to_string()],
                },
            }),
        }
    }
}

/// Predefined use cases for easy execution
#[derive(Debug, Clone)]
pub enum UseCase {
    CodeAnalysis,
    BugFinding,
    Documentation,
    Testing,
}

/// Execution status information
#[derive(Debug, Clone)]
pub struct ExecutionStatusInfo {
    pub execution_id: String,
    pub protocol_id: String,
    pub status: ExecutionStatus,
    pub progress: f64,
    pub elapsed_time: std::time::Duration,
}

/// Performance metrics summary
#[derive(Debug, Clone)]
pub struct PerformanceMetricsSummary {
    pub total_executions: usize,
    pub active_executions: usize,
    pub completed_executions: usize,
    pub average_execution_time: std::time::Duration,
    pub success_rate: f64,
}

/// Convenient builder for creating M2 integration service
pub struct M2ServiceBuilder {
    config: Option<M2Config>,
    integration_config: Option<M2IntegrationConfig>,
}

impl M2ServiceBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: None,
            integration_config: None,
        }
    }
    
    /// Set M2 API configuration
    pub fn with_config(mut self, config: M2Config) -> Self {
        self.config = Some(config);
        self
    }
    
    /// Set integration configuration
    pub fn with_integration_config(mut self, config: M2IntegrationConfig) -> Self {
        self.integration_config = Some(config);
        self
    }
    
    /// Build the service
    pub async fn build(self) -> Result<M2IntegrationService, Error> {
        let config = self.config.ok_or_else(|| Error::ConfigError("M2Config required".to_string()))?;
        let integration_config = self.integration_config.unwrap_or_else(|| M2IntegrationConfig {
            max_concurrent_executions: 10,
            default_timeout_ms: 300000, // 5 minutes
            enable_caching: true,
            enable_monitoring: true,
            default_optimization_goals: OptimizationGoals {
                primary_goal: OptimizationGoal::BalanceAll,
                secondary_goals: vec![],
                constraints: OptimizationConstraints {
                    max_cost: Some(10.0),
                    max_latency_ms: Some(30000),
                    min_quality: Some(0.90),
                },
                performance_targets: PerformanceTargets {
                    cost_reduction_target: 92.0,
                    latency_reduction_target: 0.20,
                    quality_threshold: 0.90,
                },
            },
        });
        
        M2IntegrationService::new(config, integration_config).await
    }
}

impl Default for M2ServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Default implementations for required types
impl Default for QualityStandards {
    fn default() -> Self {
        Self {
            min_confidence: 0.85,
            require_validation: true,
            require_evidence: true,
        }
    }
}

impl Default for ContextRequirements {
    fn default() -> Self {
        Self
    }
}

impl Default for ExpectedOutput {
    fn default() -> Self {
        Self
    }
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self
    }
}

impl Default for CompatibilityRequirements {
    fn default() -> Self {
        Self
    }
}

// ProtocolInput type alias for convenience
pub type ProtocolInput = serde_json::Value;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_m2_service_builder() {
        let config = M2Config {
            endpoint: "https://api.minimax.chat/v1/m2".to_string(),
            api_key: "test_key".to_string(),
            max_context_length: 200000,
            max_output_length: 128000,
            rate_limit: RateLimitConfig {
                rpm: 60,
                rps: 1,
                burst: 5,
            },
            performance: PerformanceConfig {
                cost_reduction_target: 92.0,
                latency_target_ms: 2000,
                quality_threshold: 0.90,
                enable_caching: true,
                compression_level: 5,
            },
        };
        
        let service = M2ServiceBuilder::new()
            .with_config(config)
            .build()
            .await;
            
        assert!(service.is_ok());
    }
    
    #[tokio::test]
    async fn test_use_case_classification() {
        let service = M2ServiceBuilder::new().build().await.unwrap();
        
        let input = serde_json::json!("test code");
        let task = service.classify_use_case(UseCase::CodeAnalysis, &input).unwrap();
        
        assert_eq!(task.task_type, TaskType::CodeAnalysis);
        assert_eq!(task.complexity_level, ComplexityLevel::Complex);
    }
}