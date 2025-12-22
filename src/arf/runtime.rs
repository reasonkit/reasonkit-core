//! Async runtime management and core execution engine

use crate::{
    error::Result,
    arf::config::Config,
    arf::types::*,
};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main ARF runtime managing all core components
pub struct ArfRuntime {
    config: Config,
    state_manager: Arc<StateManager>,
    plugin_manager: Arc<PluginManager>,
    active_sessions: Arc<RwLock<HashMap<SessionId, ReasoningSession>>>,
}

impl ArfRuntime {
    /// Create a new ARF runtime
    pub async fn new(
        config: Config,
        state_manager: StateManager,
        plugin_manager: PluginManager,
    ) -> Result<Self> {
        let runtime = Self {
            config,
            state_manager: Arc::new(state_manager),
            plugin_manager: Arc::new(plugin_manager),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize runtime components
        runtime.initialize().await?;

        Ok(runtime)
    }

    /// Initialize runtime components
    async fn initialize(&self) -> Result<()> {
        tracing::info!("Initializing ARF Runtime v{}", crate::VERSION);

        // Initialize state manager
        self.state_manager.initialize().await?;

        // Load and initialize plugins
        self.plugin_manager.load_plugins().await?;

        // Start background tasks
        self.start_background_tasks();

        tracing::info!("ARF Runtime initialized successfully");
        Ok(())
    }

    /// Start a new reasoning session
    pub async fn start_session(&self, problem_statement: String) -> Result<ReasoningSession> {
        let session_id = format!("session_{}", uuid::Uuid::new_v4().simple());

        let session = ReasoningSession {
            id: session_id.clone(),
            problem_statement,
            status: SessionStatus::Initialized,
            current_step: 0,
            total_steps: 10, // Absolute Reasoning Framework has 10 steps
            steps: Vec::new(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        // Store session in state
        self.state_manager.save_session(&session).await?;

        // Add to active sessions
        self.active_sessions.write().await.insert(session_id.clone(), session.clone());

        tracing::info!("Started reasoning session: {}", session_id);

        Ok(session)
    }

    /// Execute a reasoning step
    pub async fn execute_step(
        &self,
        session_id: &str,
        step_output: serde_json::Value,
    ) -> Result<ReasoningStep> {
        let mut sessions = self.active_sessions.write().await;
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| ArfError::engine("Session not found"))?;

        if session.status != SessionStatus::Running && session.status != SessionStatus::Initialized {
            return Err(ArfError::engine("Session not in executable state"));
        }

        // Get current step info
        let current_step_index = session.current_step;
        let step_config = self.get_step_config(current_step_index)?;

        // Validate step output
        self.validate_step_output(&step_config, &step_output)?;

        // Create step record
        let step = ReasoningStep {
            id: format!("step_{}_{}", session_id, current_step_index + 1),
            step_number: current_step_index + 1,
            name: step_config.name.clone(),
            instruction: step_config.instruction.clone(),
            cognitive_stance: step_config.cognitive_stance.clone(),
            time_allocation: step_config.time_allocation.clone(),
            status: StepStatus::Completed,
            input: Some(step_output),
            output: None, // Will be set by reasoning logic
            validation_result: Some(ValidationResult {
                is_valid: true,
                score: 1.0,
                errors: vec![],
                warnings: vec![],
                suggestions: vec![],
            }),
            started_at: Some(chrono::Utc::now()),
            completed_at: Some(chrono::Utc::now()),
        };

        // Add step to session
        session.steps.push(step.clone());
        session.current_step += 1;
        session.updated_at = chrono::Utc::now();

        // Check if session is complete
        if session.current_step >= session.total_steps {
            session.status = SessionStatus::Completed;
        } else {
            session.status = SessionStatus::Running;
        }

        // Save updated session
        self.state_manager.save_session(session).await?;

        Ok(step)
    }

    /// Get current step information for a session
    pub async fn get_current_step(&self, session_id: &str) -> Result<Option<ReasoningStep>> {
        let sessions = self.active_sessions.read().await;
        let session = sessions.get(session_id)
            .ok_or_else(|| ArfError::engine("Session not found"))?;

        if session.current_step >= session.total_steps {
            return Ok(None); // Session complete
        }

        let step_config = self.get_step_config(session.current_step)?;
        let step = ReasoningStep {
            id: format!("step_{}_{}", session_id, session.current_step + 1),
            step_number: session.current_step + 1,
            name: step_config.name.clone(),
            instruction: step_config.instruction.clone(),
            cognitive_stance: step_config.cognitive_stance.clone(),
            time_allocation: step_config.time_allocation.clone(),
            status: StepStatus::Pending,
            input: None,
            output: None,
            validation_result: None,
            started_at: None,
            completed_at: None,
        };

        Ok(Some(step))
    }

    /// Get session status
    pub async fn get_session_status(&self, session_id: &str) -> Result<ReasoningSession> {
        let sessions = self.active_sessions.read().await;
        sessions.get(session_id)
            .cloned()
            .ok_or_else(|| ArfError::engine("Session not found"))
    }

    /// List all active sessions
    pub async fn list_sessions(&self) -> Vec<ReasoningSession> {
        let sessions = self.active_sessions.read().await;
        sessions.values().cloned().collect()
    }

    /// Get step configuration for a given step number
    fn get_step_config(&self, step_number: usize) -> Result<StepConfig> {
        // This would load from the Absolute Reasoning Engine schema
        // For now, return a mock configuration
        match step_number {
            0 => Ok(StepConfig {
                name: "Define Scope".to_string(),
                instruction: "Delineate the exact boundaries of the problem. What is IN and what is OUT?".to_string(),
                cognitive_stance: "boundary_setting".to_string(),
                time_allocation: "10%_of_total_process".to_string(),
                output_schema: serde_json::json!({
                    "primary_objective": "string",
                    "boundary_inclusions": ["string"],
                    "boundary_exclusions": ["string"],
                    "success_definition": "string"
                }),
            }),
            1 => Ok(StepConfig {
                name: "Identify Constraints".to_string(),
                instruction: "List every limiting factor: resources, time, physics, laws, ethics.".to_string(),
                cognitive_stance: "reality_check".to_string(),
                time_allocation: "15%_of_total_process".to_string(),
                output_schema: serde_json::json!({
                    "hard_constraints": ["string"],
                    "soft_constraints": ["string"],
                    "resource_limits": {}
                }),
            }),
            // Add more steps as needed...
            _ => Err(ArfError::engine("Step configuration not found")),
        }
    }

    /// Validate step output against schema
    fn validate_step_output(&self, step_config: &StepConfig, output: &serde_json::Value) -> Result<()> {
        // Basic validation - check if output matches expected structure
        // In a full implementation, this would use JSON schema validation
        if !output.is_object() {
            return Err(ArfError::validation("output", "Must be a JSON object"));
        }

        Ok(())
    }

    /// Start background maintenance tasks
    fn start_background_tasks(&self) {
        let sessions = Arc::clone(&self.active_sessions);
        let state_manager = Arc::clone(&self.state_manager);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));

            loop {
                interval.tick().await;

                // Clean up expired sessions
                let mut sessions_write = sessions.write().await;
                let expired_sessions: Vec<SessionId> = sessions_write
                    .iter()
                    .filter(|(_, session)| {
                        chrono::Utc::now().signed_duration_since(session.updated_at)
                            .num_hours() > 24
                    })
                    .map(|(id, _)| id.clone())
                    .collect();

                for session_id in expired_sessions {
                    if let Some(session) = sessions_write.remove(&session_id) {
                        if let Err(e) = state_manager.save_session(&session).await {
                            tracing::error!("Failed to save expired session {}: {}", session_id, e);
                        }
                    }
                }
            }
        });
    }
}

/// Step configuration structure
#[derive(Debug, Clone)]
struct StepConfig {
    name: String,
    instruction: String,
    cognitive_stance: String,
    time_allocation: String,
    output_schema: serde_json::Value,
}