//! # Agency Module
//!
//! Autonomous agency and background task management for the ARF platform.
//! This module enables the system to spawn and manage autonomous agents for long-running tasks.

use crate::arf::types::*;
use crate::error::Result;
use reqwest::Client;
use scraper::{Html, Selector};
use sled::Db;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;

/// Autonomous agent for background task execution
pub struct AutonomousAgent {
    id: String,
    task_type: AgentTaskType,
    status: AgentStatus,
    progress: f64,
    findings: Vec<AgentFinding>,
    start_time: chrono::DateTime<chrono::Utc>,
    handle: Option<JoinHandle<Result<()>>>,
}

/// Types of autonomous tasks
#[derive(Debug, Clone)]
pub enum AgentTaskType {
    Research(String),       // Research a specific topic
    DataCollection(String), // Collect data from sources
    Analysis(String),       // Analyze existing data
    Monitoring(String),     // Monitor for changes
    Synthesis(String),      // Synthesize information
}

/// Agent execution status
#[derive(Debug, Clone)]
pub enum AgentStatus {
    Idle,
    Running,
    Completed,
    Failed(String),
    Terminated,
}

/// Findings from autonomous agent work
#[derive(Debug, Clone)]
pub struct AgentFinding {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub category: String,
    pub content: String,
    pub confidence: f64,
    pub source: String,
}

/// Agency manager for coordinating autonomous agents
pub struct AgencyManager {
    agents: Arc<RwLock<HashMap<String, AutonomousAgent>>>,
    task_queue: mpsc::UnboundedSender<AgentTask>,
    task_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<AgentTask>>>>,
    database: Arc<Db>,
    http_client: Client,
    max_concurrent_agents: usize,
}

#[derive(Debug)]
struct AgentTask {
    agent_id: String,
    task_type: AgentTaskType,
    priority: TaskPriority,
}

#[derive(Debug, Clone)]
enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl AgencyManager {
    /// Create a new agency manager
    pub async fn new(database_path: &str, max_concurrent: usize) -> Result<Self> {
        let database = sled::open(database_path)?;
        let (tx, rx) = mpsc::unbounded_channel();

        let manager = Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            task_queue: tx,
            task_receiver: Arc::new(RwLock::new(Some(rx))),
            database: Arc::new(database),
            http_client: Client::new(),
            max_concurrent_agents: max_concurrent,
        };

        // Start the task dispatcher
        manager.start_task_dispatcher();

        Ok(manager)
    }

    /// Spawn a new autonomous agent
    pub async fn spawn_agent(
        &self,
        task_type: AgentTaskType,
        priority: TaskPriority,
    ) -> Result<String> {
        let agent_id = format!("agent_{}", uuid::Uuid::new_v4().simple());

        let agent = AutonomousAgent {
            id: agent_id.clone(),
            task_type: task_type.clone(),
            status: AgentStatus::Idle,
            progress: 0.0,
            findings: Vec::new(),
            start_time: chrono::Utc::now(),
            handle: None,
        };

        // Check concurrent agent limit
        let agents = self.agents.read().await;
        let running_count = agents
            .values()
            .filter(|a| matches!(a.status, AgentStatus::Running))
            .count();

        if running_count >= self.max_concurrent_agents {
            return Err(ArfError::engine("Maximum concurrent agents reached"));
        }
        drop(agents);

        // Add agent to registry
        let mut agents = self.agents.write().await;
        agents.insert(agent_id.clone(), agent);

        // Queue the task
        let task = AgentTask {
            agent_id: agent_id.clone(),
            task_type,
            priority,
        };

        self.task_queue.send(task)?;

        tracing::info!("Spawned autonomous agent: {}", agent_id);

        Ok(agent_id)
    }

    /// Get agent status
    pub async fn get_agent_status(&self, agent_id: &str) -> Result<AgentStatus> {
        let agents = self.agents.read().await;
        agents
            .get(agent_id)
            .map(|a| a.status.clone())
            .ok_or_else(|| ArfError::engine("Agent not found"))
    }

    /// Get agent findings
    pub async fn get_agent_findings(&self, agent_id: &str) -> Result<Vec<AgentFinding>> {
        let agents = self.agents.read().await;
        agents
            .get(agent_id)
            .map(|a| a.findings.clone())
            .ok_or_else(|| ArfError::engine("Agent not found"))
    }

    /// Terminate an agent
    pub async fn terminate_agent(&self, agent_id: &str) -> Result<()> {
        let mut agents = self.agents.write().await;

        if let Some(agent) = agents.get_mut(agent_id) {
            agent.status = AgentStatus::Terminated;

            if let Some(handle) = agent.handle.take() {
                handle.abort();
            }

            // Save final state
            self.save_agent_state(agent).await?;
        }

        Ok(())
    }

    /// List all agents
    pub async fn list_agents(&self) -> Vec<(String, AgentStatus)> {
        let agents = self.agents.read().await;
        agents
            .iter()
            .map(|(id, agent)| (id.clone(), agent.status.clone()))
            .collect()
    }

    /// Start the task dispatcher
    fn start_task_dispatcher(&self) {
        let agents = Arc::clone(&self.agents);
        let database = Arc::clone(&self.database);
        let http_client = self.http_client.clone();
        let mut receiver = self.task_receiver.write().blocking_lock().take().unwrap();

        tokio::spawn(async move {
            while let Some(task) = receiver.recv().await {
                let agents_clone = Arc::clone(&agents);
                let db_clone = Arc::clone(&database);
                let client_clone = http_client.clone();

                tokio::spawn(async move {
                    if let Err(e) =
                        Self::execute_agent_task(task, agents_clone, db_clone, client_clone).await
                    {
                        tracing::error!("Agent task execution failed: {}", e);
                    }
                });
            }
        });
    }

    /// Execute an agent task
    async fn execute_agent_task(
        task: AgentTask,
        agents: Arc<RwLock<HashMap<String, AutonomousAgent>>>,
        database: Arc<Db>,
        http_client: Client,
    ) -> Result<()> {
        // Get agent
        let mut agents_lock = agents.write().await;
        let agent = agents_lock
            .get_mut(&task.agent_id)
            .ok_or_else(|| ArfError::engine("Agent not found during execution"))?;

        // Update status
        agent.status = AgentStatus::Running;

        // Execute based on task type
        let result = match &task.task_type {
            AgentTaskType::Research(topic) => {
                Self::execute_research_task(agent, topic, &http_client).await
            }
            AgentTaskType::DataCollection(source) => {
                Self::execute_data_collection_task(agent, source, &http_client).await
            }
            AgentTaskType::Analysis(data) => Self::execute_analysis_task(agent, data).await,
            AgentTaskType::Monitoring(target) => {
                Self::execute_monitoring_task(agent, target, &http_client).await
            }
            AgentTaskType::Synthesis(topic) => {
                Self::execute_synthesis_task(agent, topic, &database).await
            }
        };

        // Update final status
        match result {
            Ok(_) => {
                agent.status = AgentStatus::Completed;
                agent.progress = 1.0;
            }
            Err(e) => {
                agent.status = AgentStatus::Failed(e.to_string());
            }
        }

        // Save final state
        Self::save_agent_state_static(agent, &database).await?;

        Ok(())
    }

    /// Execute research task
    async fn execute_research_task(
        agent: &mut AutonomousAgent,
        topic: &str,
        http_client: &Client,
    ) -> Result<()> {
        // Simulate research by searching web
        let search_query = format!("{} research latest developments", topic);
        let findings = Self::web_search(http_client, &search_query).await?;

        for finding in findings {
            agent.findings.push(finding);
            agent.progress += 0.1; // Simulate progress
        }

        Ok(())
    }

    /// Execute data collection task
    async fn execute_data_collection_task(
        agent: &mut AutonomousAgent,
        source: &str,
        http_client: &Client,
    ) -> Result<()> {
        // Collect data from specified source
        let response = http_client.get(source).send().await?;
        let content = response.text().await?;

        let finding = AgentFinding {
            timestamp: chrono::Utc::now(),
            category: "data_collection".to_string(),
            content: content.chars().take(1000).collect(), // Limit size
            confidence: 0.8,
            source: source.to_string(),
        };

        agent.findings.push(finding);
        agent.progress = 1.0;

        Ok(())
    }

    /// Execute analysis task
    async fn execute_analysis_task(agent: &mut AutonomousAgent, data: &str) -> Result<()> {
        // Simple analysis - count words, find patterns
        let word_count = data.split_whitespace().count();
        let has_numbers = data.chars().any(|c| c.is_numeric());

        let analysis = format!(
            "Word count: {}, Contains numbers: {}",
            word_count, has_numbers
        );

        let finding = AgentFinding {
            timestamp: chrono::Utc::now(),
            category: "analysis".to_string(),
            content: analysis,
            confidence: 0.9,
            source: "data_analysis".to_string(),
        };

        agent.findings.push(finding);
        agent.progress = 1.0;

        Ok(())
    }

    /// Execute monitoring task
    async fn execute_monitoring_task(
        agent: &mut AutonomousAgent,
        target: &str,
        http_client: &Client,
    ) -> Result<()> {
        // Monitor target for changes (simplified)
        let response = http_client.get(target).send().await?;
        let status = response.status();

        let finding = AgentFinding {
            timestamp: chrono::Utc::now(),
            category: "monitoring".to_string(),
            content: format!("Status: {}", status),
            confidence: if status.is_success() { 0.9 } else { 0.5 },
            source: target.to_string(),
        };

        agent.findings.push(finding);
        agent.progress = 1.0;

        Ok(())
    }

    /// Execute synthesis task
    async fn execute_synthesis_task(
        agent: &mut AutonomousAgent,
        topic: &str,
        database: &Db,
    ) -> Result<()> {
        // Synthesize information from database
        let key = format!("synthesis_{}", topic);
        let existing_data = database.get(key.as_bytes())?;

        let synthesis = if let Some(data) = existing_data {
            format!("Synthesized data for {}: {} bytes", topic, data.len())
        } else {
            format!("No existing data found for synthesis of {}", topic)
        };

        let finding = AgentFinding {
            timestamp: chrono::Utc::now(),
            category: "synthesis".to_string(),
            content: synthesis,
            confidence: 0.7,
            source: "knowledge_base".to_string(),
        };

        agent.findings.push(finding);
        agent.progress = 1.0;

        Ok(())
    }

    /// Perform web search (simplified)
    async fn web_search(http_client: &Client, query: &str) -> Result<Vec<AgentFinding>> {
        // In a real implementation, this would search actual web sources
        // For simulation, we'll create mock findings
        let findings = vec![AgentFinding {
            timestamp: chrono::Utc::now(),
            category: "research".to_string(),
            content: format!("Latest research on {} shows promising developments", query),
            confidence: 0.8,
            source: "web_search".to_string(),
        }];

        Ok(findings)
    }

    /// Save agent state
    async fn save_agent_state(&self, agent: &AutonomousAgent) -> Result<()> {
        Self::save_agent_state_static(agent, &self.database).await
    }

    /// Static method to save agent state
    async fn save_agent_state_static(agent: &AutonomousAgent, database: &Db) -> Result<()> {
        let key = format!("agent_{}", agent.id);
        let value = serde_json::to_string(agent)?;
        database.insert(key.as_bytes(), value.as_bytes())?;
        Ok(())
    }

    /// Get agency statistics
    pub async fn get_statistics(&self) -> Result<AgencyStats> {
        let agents = self.agents.read().await;

        let total_agents = agents.len();
        let running_agents = agents
            .values()
            .filter(|a| matches!(a.status, AgentStatus::Running))
            .count();
        let completed_agents = agents
            .values()
            .filter(|a| matches!(a.status, AgentStatus::Completed))
            .count();
        let failed_agents = agents
            .values()
            .filter(|a| matches!(a.status, AgentStatus::Failed(_)))
            .count();

        Ok(AgencyStats {
            total_agents,
            running_agents,
            completed_agents,
            failed_agents,
            total_findings: agents.values().map(|a| a.findings.len()).sum(),
        })
    }
}

/// Agency statistics
#[derive(Debug, Clone)]
pub struct AgencyStats {
    pub total_agents: usize,
    pub running_agents: usize,
    pub completed_agents: usize,
    pub failed_agents: usize,
    pub total_findings: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_agency_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = AgencyManager::new(db_path.to_str().unwrap(), 5)
            .await
            .unwrap();
        let stats = manager.get_statistics().await.unwrap();
        assert_eq!(stats.total_agents, 0);
    }

    #[tokio::test]
    async fn test_agent_spawning() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let manager = AgencyManager::new(db_path.to_str().unwrap(), 5)
            .await
            .unwrap();

        let agent_id = manager
            .spawn_agent(
                AgentTaskType::Research("test topic".to_string()),
                TaskPriority::Normal,
            )
            .await
            .unwrap();

        assert!(!agent_id.is_empty());

        let status = manager.get_agent_status(&agent_id).await.unwrap();
        // Status might be Idle or Running depending on timing
        assert!(matches!(status, AgentStatus::Idle | AgentStatus::Running));
    }
}
