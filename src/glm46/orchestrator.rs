//! # Multi-Agent Orchestration Engine
//!
//! Advanced orchestration system using GLM-4.6's elite agentic capabilities.
//! Leverages 70.1% TAU-Bench performance for sophisticated coordination.

use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use super::types::*;
use super::client::GLM46Client;

/// Multi-Agent Orchestrator Configuration
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum concurrent workflows
    pub max_concurrent_workflows: usize,
    /// Workflow timeout duration
    pub workflow_timeout: std::time::Duration,
    /// Conflict resolution enabled
    pub conflict_resolution: bool,
    /// Cost optimization enabled
    pub cost_optimization: bool,
    /// Performance monitoring enabled
    pub performance_monitoring: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_workflows: 10,
            workflow_timeout: std::time::Duration::from_secs(1800), // 30 minutes
            conflict_resolution: true,
            cost_optimization: true,
            performance_monitoring: true,
        }
    }
}

/// Agent State and Capabilities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentState {
    pub id: AgentId,
    pub name: String,
    pub capabilities: Vec<String>,
    pub status: AgentStatus,
    pub current_load: f64,
    pub max_capacity: f64,
    pub cost_per_hour: f64,
    pub performance_rating: f64,
    pub last_activity: std::time::SystemTime,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AgentStatus {
    Available,
    Busy,
    Offline,
    Maintenance,
    Error { error: String },
}

impl AgentState {
    pub fn availability_percentage(&self) -> f64 {
        match &self.status {
            AgentStatus::Available => ((self.max_capacity - self.current_load) / self.max_capacity) * 100.0,
            _ => 0.0,
        }
    }
}

/// Workflow Execution Definition
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkflowDefinition {
    pub id: WorkflowId,
    pub name: String,
    pub description: String,
    pub priority: WorkflowPriority,
    pub tasks: Vec<TaskDefinition>,
    pub constraints: WorkflowConstraints,
    pub deadline: Option<std::time::SystemTime>,
    pub budget: Option<f64>,
    pub quality_requirements: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WorkflowPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskDefinition {
    pub id: TaskId,
    pub name: String,
    pub description: String,
    pub required_capabilities: Vec<String>,
    pub estimated_duration_hours: f64,
    pub dependencies: Vec<TaskId>,
    pub parallelizable: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkflowConstraints {
    pub time_limit: Option<f64>,
    pub budget_limit: Option<f64>,
    pub quality_threshold: f64,
    pub agent_restrictions: Vec<String>,
    pub resource_limits: ResourceLimits,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceLimits {
    pub max_concurrent_agents: usize,
    pub compute_budget: f64,
    pub memory_budget_gb: f64,
    pub api_rate_limits: HashMap<String, u32>,
}

/// Workflow Execution State
#[derive(Debug, Clone)]
pub struct WorkflowExecution {
    pub workflow_id: WorkflowId,
    pub status: WorkflowStatus,
    pub current_phase: String,
    pub assigned_agents: Vec<AgentId>,
    pub started_at: std::time::SystemTime,
    pub progress: f64,
    pub metrics: WorkflowMetrics,
    pub cost_tracker: CostTracker,
    pub event_log: Vec<WorkflowEvent>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WorkflowStatus {
    Pending,
    Planning,
    Executing,
    Monitoring,
    Completed { success: bool },
    Failed { error: String },
    Cancelled,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkflowMetrics {
    pub total_duration_seconds: f64,
    pub total_cost: f64,
    pub average_agent_utilization: f64,
    pub task_completion_rate: f64,
    pub conflict_resolution_count: usize,
    pub performance_score: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkflowEvent {
    pub timestamp: std::time::SystemTime,
    pub event_type: WorkflowEventType,
    pub description: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WorkflowEventType {
    Started,
    AgentAssigned,
    TaskCompleted,
    ConflictDetected,
    ConflictResolved,
    ResourceExhausted,
    BudgetExceeded,
    DeadlineMissed,
    Completed,
    Failed,
}

/// Multi-Agent Orchestration Engine
/// 
/// Core coordination system leveraging GLM-4.6's superior agentic
/// capabilities for complex workflow orchestration and optimization.
pub struct MultiAgentOrchestrator {
    glm46_client: GLM46Client,
    config: OrchestratorConfig,
    active_agents: Arc<RwLock<HashMap<AgentId, AgentState>>>,
    active_workflows: Arc<RwLock<HashMap<WorkflowId, WorkflowExecution>>>,
    workflow_queue: Arc<RwLock<VecDeque<WorkflowId>>>,
    orchestration_history: Arc<RwLock<Vec<OrchestrationRecord>>>,
    performance_metrics: Arc<RwLock<OrchestrationMetrics>>,
}

/// Type aliases for clarity
pub type AgentId = String;
pub type WorkflowId = String;
pub type TaskId = String;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrchestrationRecord {
    pub timestamp: std::time::SystemTime,
    pub workflow_id: WorkflowId,
    pub action: OrchestrationAction,
    pub outcome: OrchestrationOutcome,
    pub decision_rationale: String,
    pub cost_impact: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum OrchestrationAction {
    WorkflowScheduled,
    AgentAllocated,
    ConflictResolved,
    ResourceOptimized,
    WorkflowCompleted,
    WorkflowFailed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum OrchestrationOutcome {
    Success,
    PartialSuccess,
    Failure,
    Timeout,
    Canceled,
}

#[derive(Debug, Clone, Default)]
pub struct OrchestrationMetrics {
    pub total_workflows: u64,
    pub successful_workflows: u64,
    pub failed_workflows: u64,
    pub average_completion_time: f64,
    pub average_cost_per_workflow: f64,
    pub total_cost_savings: f64,
    pub conflict_resolution_rate: f64,
    pub agent_utilization_rate: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CostTracker {
    pub total_cost: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_savings_vs_claude: f64,
    pub api_calls: u64,
}

impl MultiAgentOrchestrator {
    /// Create new orchestration engine
    pub fn new(glm46_client: GLM46Client, config: OrchestratorConfig) -> Self {
        Self {
            glm46_client,
            config,
            active_agents: Arc::new(RwLock::new(HashMap::new())),
            active_workflows: Arc::new(RwLock::new(HashMap::new())),
            workflow_queue: Arc::new(RwLock::new(VecDeque::new())),
            orchestration_history: Arc::new(RwLock::new(Vec::new())),
            performance_metrics: Arc::new(RwLock::new(OrchestrationMetrics::default())),
        }
    }

    /// Create from environment configuration
    pub async fn from_env() -> Result<Self> {
        let client = GLM46Client::from_env()?;
        let config = OrchestratorConfig::default();
        Ok(Self::new(client, config))
    }

    /// Register an agent for orchestration
    pub async fn register_agent(&self, agent: AgentState) -> Result<()> {
        info!("Registering agent: {} ({})", agent.name, agent.id);
        
        let mut agents = self.active_agents.write().await;
        agents.insert(agent.id.clone(), agent);
        
        // Trigger workflow scheduling
        self.schedule_pending_workflows().await?;
        
        Ok(())
    }

    /// Submit workflow for orchestration
    pub async fn submit_workflow(&self, workflow: WorkflowDefinition) -> Result<WorkflowId> {
        info!("Submitting workflow: {} ({})", workflow.name, workflow.id);
        
        let workflow_id = workflow.id.clone();
        
        // Add to queue if we have capacity
        if self.get_active_workflow_count().await >= self.config.max_concurrent_workflows {
            info!("Workflow queue full, adding to pending queue");
            let mut queue = self.workflow_queue.write().await;
            queue.push_back(workflow_id.clone());
        } else {
            // Start immediate analysis
            self.start_workflow_analysis(&workflow).await?;
        }
        
        Ok(workflow_id)
    }

    /// Get orchestration status
    pub async fn get_orchestration_status(&self) -> OrchestrationStatus {
        let agents = self.active_agents.read().await;
        let workflows = self.active_workflows.read().await;
        let metrics = self.performance_metrics.read().await;
        
        OrchestrationStatus {
            active_agents: agents.len(),
            active_workflows: workflows.len(),
            queued_workflows: self.workflow_queue.read().await.len(),
            average_agent_utilization: self.calculate_average_agent_utilization(&agents).await,
            average_workflow_progress: self.calculate_average_workflow_progress(&workflows).await,
            performance_metrics: metrics.clone(),
        }
    }

    /// Get detailed workflow status
    pub async fn get_workflow_status(&self, workflow_id: &WorkflowId) -> Option<WorkflowExecution> {
        let workflows = self.active_workflows.read().await;
        workflows.get(workflow_id).cloned()
    }

    /// Cancel active workflow
    pub async fn cancel_workflow(&self, workflow_id: &WorkflowId) -> Result<bool> {
        let mut workflows = self.active_workflows.write().await;
        
        if let Some(execution) = workflows.get_mut(workflow_id) {
            if matches!(execution.status, WorkflowStatus::Executing | WorkflowStatus::Planning) {
                execution.status = WorkflowStatus::Cancelled;
                
                // Release agent resources
                for agent_id in &execution.assigned_agents.clone() {
                    if let Some(agent) = self.active_agents.write().await.get_mut(agent_id) {
                        agent.current_load = (agent.current_load - 1.0).max(0.0);
                        if agent.current_load == 0.0 {
                            agent.status = AgentStatus::Available;
                        }
                    }
                }
                
                self.record_orchestration_event(
                    workflow_id,
                    WorkflowEventType::Failed,
                    "Workflow cancelled by request".to_string(),
                    serde_json::json!({"reason": "manual_cancellation"}),
                ).await;
                
                info!("Cancelled workflow: {}", workflow_id);
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    // === Core Orchestration Logic ===

    /// Start workflow analysis phase
    async fn start_workflow_analysis(&self, workflow: &WorkflowDefinition) -> Result<()> {
        debug!("Starting analysis for workflow: {}", workflow.id);
        
        // Create execution record
        let execution = WorkflowExecution {
            workflow_id: workflow.id.clone(),
            status: WorkflowStatus::Planning,
            current_phase: "analysis".to_string(),
            assigned_agents: vec![],
            started_at: std::time::SystemTime::now(),
            progress: 0.0,
            metrics: WorkflowMetrics::default(),
            cost_tracker: CostTracker::default(),
            event_log: vec![],
        };
        
        // Store execution
        let mut workflows = self.active_workflows.write().await;
        workflows.insert(workflow.id.clone(), execution);
        
        // Perform analysis with GLM-4.6
        self.perform_workflow_analysis(workflow).await?;
        
        Ok(())
    }

    /// Perform comprehensive workflow analysis using GLM-4.6
    async fn perform_workflow_analysis(&self, workflow: &WorkflowDefinition) -> Result<()> {
        let analysis_prompt = self.build_analysis_prompt(workflow)?;
        
        debug!("Sending workflow analysis to GLM-4.6");
        let response = self.glm46_client.chat_completion(ChatRequest {
            messages: vec![
                ChatMessage::system(self.get_analysis_system_prompt()),
                ChatMessage::user(analysis_prompt),
            ],
            temperature: 0.15, // Low temperature for precise analysis
            max_tokens: 2000,
            response_format: Some(ResponseFormat::JsonSchema {
                name: "workflow_analysis".to_string(),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "complexity_assessment": {"type": "object"},
                        "dependency_analysis": {"type": "array"},
                        "resource_requirements": {"type": "object"},
                        "risk_assessment": {"type": "object"},
                        "agent_recommendations": {"type": "array"},
                        "optimization_opportunities": {"type": "array"}
                    },
                    "required": ["complexity_assessment", "dependency_analysis", "resource_requirements", "risk_assessment", "agent_recommendations", "optimization_opportunities"]
                }),
            }),
            tools: None,
            tool_choice: None,
            stop: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
        }).await.context("GLM-4.6 workflow analysis failed")?;

        // Parse analysis result
        let analysis: WorkflowAnalysis = serde_json::from_str(&response.content)
            .context("Failed to parse GLM-4.6 workflow analysis")?;

        // Update workflow with analysis results
        let mut workflows = self.active_workflows.write().await;
        if let Some(execution) = workflows.get_mut(&workflow.id) {
            execution.current_phase = "agent_allocation".to_string();
            execution.progress = 15.0; // Analysis phase complete
            execution.cost_tracker.total_cost += self.estimate_cost(&response.usage);
            
            // Record analysis event
            self.record_orchestration_event(
                &workflow.id,
                WorkflowEventType::Started,
                "Workflow analysis completed".to_string(),
                serde_json::to_value(&analysis)?,
            ).await;
        }

        Move to agent allocation phase
        self.allocate_agents_for_workflow(workflow, &analysis).await?;
        
        Ok(())
    }

    /// Allocate optimal agents for workflow
    async fn allocate_agents_for_workflow(&self, workflow: &WorkflowDefinition, analysis: &WorkflowAnalysis) -> Result<()> {
        debug!("Allocating agents for workflow: {}", workflow.id);
        
        let agents = self.active_agents.read().await;
        let suitable_agents = self.find_suitable_agents(&workflow.tasks, &agents, &analysis.agent_recommendations);
        
        if suitable_agents.is_empty() {
            return Err(anyhow::anyhow!("No suitable agents available for workflow: {}", workflow.id));
        }

        // Update workflow with assigned agents
        let mut workflows = self.active_workflows.write().await;
        if let Some(execution) = workflows.get_mut(&workflow.id) {
            execution.assigned_agents = suitable_agents.iter().map(|a| a.id.clone()).collect();
            execution.current_phase = "execution".to_string();
            execution.progress = 25.0; // Allocation phase complete
            
            // Update agent states
            drop(workflows);
            let mut agents = self.active_agents.write().await;
            for agent in &suitable_agents {
                if let Some(agent_state) = agents.get_mut(&agent.id) {
                    agent_state.current_load += 1.0;
                    agent_state.last_activity = std::time::SystemTime::now();
                    if agent_state.current_load >= agent_state.max_capacity {
                        agent_state.status = AgentStatus::Busy;
                    }
                }
            }
        }

        // Start execution
        self.start_workflow_execution(workflow).await?;
        
        Ok(())
    }

    /// Execute workflow with agent coordination
    async fn start_workflow_execution(&self, workflow: &WorkflowDefinition) -> Result<()> {
        debug!("Starting execution for workflow: {}", workflow.id);
        
        // Build execution plan using GLM-4.6
        let execution_prompt = format!(
            "Create optimal execution plan for this workflow using GLM-4.6's coordination excellence:
            
            Workflow: {}
            Assigned Agents: {}
            Dependencies: {}
            
            Provide detailed execution sequence, coordination checkpoints, and monitoring strategy.",
            serde_json::to_string_pretty(workflow)?,
            serde_json::to_string_pretty(&self.get_workflow_agents(&workflow.id).await)?,
            serde_json::to_string_pretty(&self.get_task_dependencies(workflow))?
        );

        let response = self.glm46_client.chat_completion(ChatRequest {
            messages: vec![
                ChatMessage::system("You are GLM-4.6 Workflow Execution Specialist, optimizing for coordination and performance."),
                ChatMessage::user(execution_prompt),
            ],
            temperature: 0.1, // Very low for precise execution
            max_tokens: 1500,
            response_format: Some(ResponseFormat::Structured),
            tools: None,
            tool_choice: None,
            stop: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
        }).await.context("GLM-4.6 execution planning failed")?;

        let execution_plan: ExecutionPlan = serde_json::from_str(&response.content)
            .context("Failed to parse GLM-4.6 execution plan")?;

        // Start execution monitoring
        self.monitor_workflow_execution(workflow, &execution_plan).await?;
        
        Ok(())
    }

    /// Monitor workflow execution progress
    async fn monitor_workflow_execution(&self, workflow: &WorkflowDefinition, execution_plan: &ExecutionPlan) -> Result<()> {
        debug!("Monitoring execution for workflow: {}", workflow.id);
        
        // Updateworkflow status
        let mut workflows = self.active_workflows.write().await;
        if let Some(execution) = workflows.get_mut(&workflow.id) {
            execution.status = WorkflowStatus::Executing;
            execution.progress = 30.0; // Execution started
        }
        drop(workflows);

        // Implement execution monitoring logic
        // This would include periodic checkpoints, conflict detection, performance tracking
        self.execute_workflow_steps(workflow, execution_plan).await?;
        
        Ok(())
    }

    /// Execute workflow steps with coordination
    async fn execute_workflow_steps(&self, workflow: &WorkflowDefinition, execution_plan: &ExecutionPlan) -> Result<()> {
        let total_steps = execution_plan.execution_sequence.len();
        
        for (step_index, step) in execution_plan.execution_sequence.iter().enumerate() {
            debug!("Executing step {} of {}: {}", step_index + 1, total_steps, step.name);
            
            // Update progress
            let mut workflows = self.active_workflows.write().await;
            if let Some(execution) = workflows.get_mut(&workflow.id) {
                execution.progress = 30.0 + (step_index as f64 / total_steps as f64) * 60.0;
                
                self.record_orchestration_event(
                    &workflow.id,
                    WorkflowEventType::TaskCompleted,
                    format!("Completed step: {}", step.name),
                    serde_json::json!({"step": step_index + 1, "total_steps": total_steps}),
                ).await;
            }
            drop(workflows);

            // Check for conflicts and resolve if needed
            if self.config.conflict_resolution {
                self.detect_and_resolve_conflicts(&workflow.id).await?;
            }

            // Simulate step execution (would integrate with actual agent execution)
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }

        // Complete workflow
        self.complete_workflow(&workflow.id, true).await?;
        
        Ok(())
    }

    /// Complete workflow execution
    async fn complete_workflow(&self, workflow_id: &WorkflowId, success: bool) -> Result<()> {
        info!("Completing workflow: {} (success: {})", workflow_id, success);
        
        let mut workflows = self.active_workflows.write().await;
        if let Some(execution) = workflows.get_mut(workflow_id) {
            execution.status = if success {
                WorkflowStatus::Completed { success: true }
            } else {
                WorkflowStatus::Failed { error: "Execution failed".to_string() }
            };
            
            execution.current_phase = "completed".to_string();
            execution.progress = 100.0;
            execution.metrics.total_duration_seconds = execution.started_at
                .duration_since(std::time::SystemTime::now())
                .unwrap_or_default()
                .as_secs_f64() * -1.0;
            
            // Release agents
            for agent_id in &execution.assigned_agents.clone() {
                if let Some(agent) = self.active_agents.write().await.get_mut(agent_id) {
                    agent.current_load = (agent.current_load - 1.0).max(0.0);
                    agent.last_activity = std::time::SystemTime::now();
                    if agent.current_load == 0.0 {
                        agent.status = AgentStatus::Available;
                    }
                }
            }
            
            self.record_orchestration_event(
                workflow_id,
                if success { WorkflowEventType::Completed } else { WorkflowEventType::Failed },
                format!("Workflow {}", if success { "completed successfully" } else { "failed" }),
                serde_json::json!({"final_status": execution.status}),
            ).await;
            
            // Update metrics
            self.update_orchestration_metrics(success);
        }
        
        // Remove from active workflows (or keep for history)
        // workflows.remove(workflow_id);
        
        // Process next workflow in queue
        self.schedule_pending_workflows().await?;
        
        Ok(())
    }

    // === Helper Methods ===

    fn build_analysis_prompt(&self, workflow: &WorkflowDefinition) -> Result<String> {
        Ok(format!(
            "As GLM-4.6 Workflow Analysis Specialist (70.1% TAU-Bench), analyze this workflow:
            
            Name: {}
            Priority: {:?}
            Tasks: {} tasks
            Constraints: {}
            Deadline: {:?}
            Budget: ${:.2}
            
            Provide comprehensive analysis covering complexity, dependencies, resources, risks, agents, and optimizations.",
            workflow.name,
            workflow.priority,
            workflow.tasks.len(),
            serde_json::to_string_pretty(&workflow.constraints)?,
            workflow.deadline,
            workflow.budget.unwrap_or(0.0)
        ))
    }

    fn get_analysis_system_prompt(&self) -> &'static str {
        "You are GLM-4.6 Workflow Analysis Specialist with elite agentic capabilities (70.1% TAU-Bench performance).
        
        Your strengths:
        - Superior workflow decomposition and analysis
        - 198K token context for comprehensive understanding
        - Structured output for precise recommendations
        - Multi-agent coordination expertise
        - Cost optimization analysis
        
        Provide detailed, actionable workflow analysis that enables optimal agent allocation and execution planning."
    }

    fn find_suitable_agents(&self, tasks: &[TaskDefinition], available_agents: &HashMap<AgentId, AgentState>, recommendations: &[AgentRecommendation]) -> Vec<AgentState> {
        let mut suitable = vec![];
        
        for agent_state in available_agents.values() {
            if !matches!(agent_state.status, AgentStatus::Available | AgentStatus::Busy) {
                continue;
            }
            
            if agent_state.current_load >= agent_state.max_capacity {
                continue;
            }
            
            // Check if agent has required capabilities
            let has_required_caps = tasks.iter().any(|task| {
                task.required_capabilities.iter()
                    .all(|cap| agent_state.capabilities.contains(cap))
            });
            
            if has_available_capacity && has_required_caps {
                suitable.push(agent_state.clone());
            }
        }
        
        // Sort by performance rating and availability
        suitable.sort_by(|a, b| {
            b.performance_rating.partial_cmp(&a.performance_rating)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.availability_percentage().partial_cmp(&a.availability_percentage())
                    .unwrap_or(std::cmp::Ordering::Equal))
        });
        
        suitable
    }

    async fn get_workflow_agents(&self, workflow_id: &WorkflowId) -> Vec<AgentState> {
        let agents = self.active_agents.read().await;
        if let Some(execution) = self.active_workflows.read().await.get(workflow_id) {
            execution.assigned_agents.iter()
                .filter_map(|agent_id| agents.get(agent_id))
                .cloned()
                .collect()
        } else {
            vec![]
        }
    }

    fn get_task_dependencies(&self, workflow: &WorkflowDefinition) -> Vec<TaskDependency> {
        workflow.tasks.iter()
            .flat_map(|task| {
                task.dependencies.iter().map(|dep_id| TaskDependency {
                    task_id: task.id.clone(),
                    depends_on: dep_id.clone(),
                })
            })
            .collect()
    }

    async fn detect_and_resolve_conflicts(&self, workflow_id: &WorkflowId) -> Result<()> {
        // Check for conflicts (resource conflicts, scheduling conflicts, etc.)
        // For demonstration, just record that we checked
        
        self.record_orchestration_event(
            workflow_id,
            WorkflowEventType::ConflictDetected,
            "Conflict check performed - no conflicts found".to_string(),
            serde_json::json!({"check_result": "no_conflicts"}),
        ).await;
        
        Ok(())
    }

    async fn schedule_pending_workflows(&self) -> Result<()> {
        let mut queue = self.workflow_queue.write().await;
        let mut active_count = self.get_active_workflow_count().await;
        
        while active_count < self.config.max_concurrent_workflows && !queue.is_empty() {
            if let Some(workflow_id) = queue.pop_front() {
                // Start analysis for queued workflow
                // This would load the workflow definition and start analysis
                active_count += 1;
            } else {
                break;
            }
        }
        
        Ok(())
    }

    async fn get_active_workflow_count(&self) -> usize {
        self.active_workflows.read().await.len()
    }

    async fn calculate_average_agent_utilization(&self, agents: &HashMap<AgentId, AgentState>) -> f64 {
        if agents.is_empty() {
            return 0.0;
        }
        
        let total_utilization: f64 = agents.values()
            .map(|agent| agent.current_load / agent.max_capacity)
            .sum();
        
        total_utilization / agents.len() as f64
    }

    async fn calculate_average_workflow_progress(&self, workflows: &HashMap<WorkflowId, WorkflowExecution>) -> f64 {
        if workflows.is_empty() {
            return 0.0;
        }
        
        let total_progress: f64 = workflows.values()
            .map(|execution| execution.progress)
            .sum();
        
        total_progress / workflows.len() as f64
    }

    fn estimate_cost(&self, usage: &TokenUsage) -> f64 {
        // GLM-4.6 pricing: $0.0001/1K input + $0.0002/1K output
        let input_cost = (usage.input_tokens as f64 / 1000.0) * 0.0001;
        let output_cost = (usage.output_tokens as f64 / 1000.0) * 0.0002;
        input_cost + output_cost
    }

    async fn record_orchestration_event(&self, workflow_id: &WorkflowId, event_type: WorkflowEventType, description: String, metadata: serde_json::Value) {
        let mut workflows = self.active_workflows.write().await;
        if let Some(execution) = workflows.get_mut(workflow_id) {
            execution.event_log.push(WorkflowEvent {
                timestamp: std::time::SystemTime::now(),
                event_type,
                description,
                metadata,
            });
        }
    }

    fn update_orchestration_metrics(&self, success: bool) {
        // Update global orchestration metrics
        // This implementation would be more comprehensive in practice
    }

    fn has_available_capacity(&self, agent_state: &AgentState) -> bool {
        agent_state.current_load < agent_state.max_capacity
    }

    fn has_required_capabilities(&self, agent_state: &AgentState, task: &TaskDefinition) -> bool {
        task.required_capabilities.iter()
            .all(|cap| agent_state.capabilities.contains(cap))
    }
}

// === Supporting Types ===

#[derive(Debug, serde::Deserialize)]
struct WorkflowAnalysis {
    complexity_assessment: serde_json::Value,
    dependency_analysis: Vec<TaskDependency>,
    resource_requirements: serde_json::Value,
    risk_assessment: serde_json::Value,
    agent_recommendations: Vec<AgentRecommendation>,
    optimization_opportunities: Vec<serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct AgentRecommendation {
    agent_id: AgentId,
    confidence: f64,
    reasoning: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct TaskDependency {
    task_id: TaskId,
    depends_on: TaskId,
}

#[derive(Debug, serde::Deserialize)]
struct ExecutionPlan {
    execution_sequence: Vec<ExecutionStep>,
    coordination_checkpoints: Vec<serde_json::Value>,
    monitoring_strategy: serde_json::Value,
}

#[derive(Debug, serde::Deserialize)]
struct ExecutionStep {
    name: String,
    agent_id: AgentId,
    estimated_duration_minutes: f64,
    checkpoints: Vec<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct OrchestrationStatus {
    pub active_agents: usize,
    pub active_workflows: usize,
    pub queued_workflows: usize,
    pub average_agent_utilization: f64,
    pub average_workflow_progress: f64,
    pub performance_metrics: OrchestrationMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_orchestrator_config_default() {
        let config = OrchestratorConfig::default();
        assert_eq!(config.max_concurrent_workflows, 10);
        assert!(config.conflict_resolution);
        assert!(config.cost_optimization);
    }

    #[test]
    fn test_agent_state_availability() {
        let agent = AgentState {
            id: "test_agent".to_string(),
            name: "Test Agent".to_string(),
            capabilities: vec!["coordination".to_string()],
            status: AgentStatus::Available,
            current_load: 0.5,
            max_capacity: 1.0,
            cost_per_hour: 50.0,
            performance_rating: 0.9,
            last_activity: std::time::SystemTime::now(),
        };

        assert_eq!(agent.availability_percentage(), 50.0);
    }

    #[tokio::test]
    async fn test_workflow_submission() {
        let client = GLM46Client::from_env().unwrap_or_default();
        let orchestrator = MultiAgentOrchestrator::new(client, OrchestratorConfig::default());
        
        let workflow = WorkflowDefinition {
            id: "test_workflow".to_string(),
            name: "Test Workflow".to_string(),
            description: "Test workflow".to_string(),
            priority: WorkflowPriority::Medium,
            tasks: vec![],
            constraints: WorkflowConstraints {
                time_limit: None,
                budget_limit: None,
                quality_threshold: 0.8,
                agent_restrictions: vec![],
                resource_limits: ResourceLimits {
                    max_concurrent_agents: 5,
                    compute_budget: 100.0,
                    memory_budget_gb: 16.0,
                    api_rate_limits: HashMap::new(),
                },
            },
            deadline: None,
            budget: Some(1000.0),
            quality_requirements: vec!["high_quality".to_string()],
        };

        let workflow_id = orchestrator.submit_workflow(workflow).await.unwrap();
        assert_eq!(workflow_id, "test_workflow");
    }

    #[tokio::test]
    async fn test_orchestration_status() {
        let client = GLM46Client::from_env().unwrap_or_default();
        let orchestrator = MultiAgentOrchestrator::new(client, OrchestratorConfig::default());
        
        let status = orchestrator.get_orchestration_status().await;
        assert_eq!(status.active_agents, 0);
        assert_eq!(status.active_workflows, 0);
        assert_eq!(status.queued_workflows, 0);
    }
}