//! MCP Server Registry
//!
//! Dynamic server discovery, registration, and health monitoring.

use super::server::{McpServerTrait, ServerStatus};
use super::tools::{GetPromptResult, Prompt, Tool};
use super::types::*;
use crate::error::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Health check status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// Server is healthy
    Healthy,
    /// Server is degraded
    Degraded,
    /// Server is unhealthy
    Unhealthy,
    /// Health check in progress
    Checking,
    /// Unknown status
    Unknown,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Server ID
    pub server_id: Uuid,
    /// Server name
    pub server_name: String,
    /// Health status
    pub status: HealthStatus,
    /// Last check timestamp
    pub checked_at: DateTime<Utc>,
    /// Response time in milliseconds
    pub response_time_ms: Option<f64>,
    /// Error message if unhealthy
    pub error: Option<String>,
}

/// Server registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerRegistration {
    /// Server ID
    pub id: Uuid,
    /// Server name
    pub name: String,
    /// Server info
    pub info: ServerInfo,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last health check
    pub last_health_check: Option<HealthCheck>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// MCP server registry
pub struct McpRegistry {
    /// Registered servers
    servers: Arc<RwLock<HashMap<Uuid, Arc<dyn McpServerTrait>>>>,
    /// Server registrations (metadata)
    registrations: Arc<RwLock<HashMap<Uuid, ServerRegistration>>>,
    /// Health check interval in seconds
    health_check_interval_secs: u64,
    /// Background health check handle
    health_check_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl McpRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            servers: Arc::new(RwLock::new(HashMap::new())),
            registrations: Arc::new(RwLock::new(HashMap::new())),
            health_check_interval_secs: crate::mcp::DEFAULT_HEALTH_CHECK_INTERVAL_SECS,
            health_check_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a new registry with custom health check interval
    pub fn with_health_check_interval(interval_secs: u64) -> Self {
        Self {
            servers: Arc::new(RwLock::new(HashMap::new())),
            registrations: Arc::new(RwLock::new(HashMap::new())),
            health_check_interval_secs: interval_secs,
            health_check_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Register a server
    pub async fn register_server(
        &self,
        server: Arc<dyn McpServerTrait>,
        tags: Vec<String>,
    ) -> Result<Uuid> {
        let info = server.server_info().await;
        let capabilities = server.capabilities().await;

        let registration = ServerRegistration {
            id: Uuid::new_v4(),
            name: info.name.clone(),
            info,
            capabilities,
            registered_at: Utc::now(),
            last_health_check: None,
            tags,
        };

        let id = registration.id;

        let mut servers = self.servers.write().await;
        let mut regs = self.registrations.write().await;

        servers.insert(id, server);
        regs.insert(id, registration);

        Ok(id)
    }

    /// Unregister a server
    pub async fn unregister_server(&self, id: Uuid) -> Result<()> {
        let mut servers = self.servers.write().await;
        let mut regs = self.registrations.write().await;

        if let Some(server) = servers.remove(&id) {
            // Attempt graceful shutdown via Arc
            // Note: We can't unwrap Arc<dyn Trait> as trait objects aren't Sized
            // Just drop the Arc - if it's the last reference, the server will be dropped
            drop(server);
        }

        regs.remove(&id);

        Ok(())
    }

    /// Get a server by ID
    pub async fn get_server(&self, id: Uuid) -> Option<Arc<dyn McpServerTrait>> {
        let servers = self.servers.read().await;
        servers.get(&id).cloned()
    }

    /// List all registered servers
    pub async fn list_servers(&self) -> Vec<ServerRegistration> {
        let regs = self.registrations.read().await;
        regs.values().cloned().collect()
    }

    /// Find servers by tag
    pub async fn find_servers_by_tag(&self, tag: &str) -> Vec<ServerRegistration> {
        let regs = self.registrations.read().await;
        regs.values()
            .filter(|r| r.tags.iter().any(|t| t == tag))
            .cloned()
            .collect()
    }

    /// List all tools from all servers
    pub async fn list_all_tools(&self) -> Result<Vec<Tool>> {
        let servers = self.servers.read().await;
        let mut all_tools = Vec::new();

        for (id, server) in servers.iter() {
            let regs = self.registrations.read().await;
            let server_name = regs.get(id).map(|r| r.name.clone()).unwrap_or_default();

            // Query tools/list from server
            let request = McpRequest::new(
                RequestId::String(Uuid::new_v4().to_string()),
                "tools/list",
                None,
            );

            match server.send_request(request).await {
                Ok(response) => {
                    if let Some(result) = response.result {
                        if let Ok(tools_response) =
                            serde_json::from_value::<ToolsListResponse>(result)
                        {
                            for mut tool in tools_response.tools {
                                tool.server_id = Some(*id);
                                tool.server_name = Some(server_name.clone());
                                all_tools.push(tool);
                            }
                        }
                    }
                }
                Err(_) => {
                    // Server didn't respond - skip
                    continue;
                }
            }
        }

        Ok(all_tools)
    }

    /// List all prompts from all servers
    pub async fn list_all_prompts(&self) -> Result<Vec<Prompt>> {
        let servers = self.servers.read().await;
        let mut all_prompts = Vec::new();

        for (_, server) in servers.iter() {
            // Query prompts/list from server
            let request = McpRequest::new(
                RequestId::String(Uuid::new_v4().to_string()),
                "prompts/list",
                None,
            );

            match server.send_request(request).await {
                Ok(response) => {
                    if let Some(result) = response.result {
                        if let Ok(prompts_response) =
                            serde_json::from_value::<PromptsListResponse>(result)
                        {
                            all_prompts.extend(prompts_response.prompts);
                        }
                    }
                }
                Err(_) => {
                    continue;
                }
            }
        }

        Ok(all_prompts)
    }

    /// Get a prompt from a specific server (or find by name)
    pub async fn get_prompt(
        &self,
        prompt_name: &str,
        arguments: HashMap<String, String>,
        server_id: Option<Uuid>,
    ) -> Result<GetPromptResult> {
        let servers = self.servers.read().await;

        // If server_id is provided, query that server directly
        if let Some(id) = server_id {
            if let Some(server) = servers.get(&id) {
                return self
                    .get_prompt_from_server(server.clone(), prompt_name, arguments)
                    .await;
            } else {
                return Err(Error::NotFound {
                    resource: format!("Server {}", id),
                });
            }
        }

        // Otherwise, broadcast to find the prompt
        // Note: This is inefficient; in a real registry, we'd cache prompt->server mapping
        for (_, server) in servers.iter() {
            if let Ok(result) = self
                .get_prompt_from_server(server.clone(), prompt_name, arguments.clone())
                .await
            {
                return Ok(result);
            }
        }

        Err(Error::NotFound {
            resource: format!("Prompt {}", prompt_name),
        })
    }

    async fn get_prompt_from_server(
        &self,
        server: Arc<dyn McpServerTrait>,
        prompt_name: &str,
        arguments: HashMap<String, String>,
    ) -> Result<GetPromptResult> {
        let params = serde_json::json!({
            "name": prompt_name,
            "arguments": arguments
        });

        let request = McpRequest::new(
            RequestId::String(Uuid::new_v4().to_string()),
            "prompts/get",
            Some(params),
        );

        let response = server.send_request(request).await?;

        if let Some(error) = response.error {
            return Err(Error::Mcp(error.message));
        }

        if let Some(result) = response.result {
            let prompt_result: GetPromptResult = serde_json::from_value(result)
                .map_err(Error::Json)?;
            Ok(prompt_result)
        } else {
            Err(Error::Mcp("Empty response from server".to_string()))
        }
    }

    /// Perform health check on a specific server
    pub async fn check_server_health(&self, id: Uuid) -> Result<HealthCheck> {
        let server = self.get_server(id).await.ok_or_else(|| Error::NotFound {
            resource: format!("Server {}", id),
        })?;

        let regs = self.registrations.read().await;
        let server_name = regs.get(&id).map(|r| r.name.clone()).unwrap_or_default();
        drop(regs);

        let start = std::time::Instant::now();
        let is_healthy = server.health_check().await?;
        let response_time_ms = start.elapsed().as_millis() as f64;

        let status = match server.status().await {
            ServerStatus::Running => HealthStatus::Healthy,
            ServerStatus::Degraded => HealthStatus::Degraded,
            ServerStatus::Unhealthy | ServerStatus::Failed => HealthStatus::Unhealthy,
            _ => HealthStatus::Unknown,
        };

        let health_check = HealthCheck {
            server_id: id,
            server_name,
            status,
            checked_at: Utc::now(),
            response_time_ms: Some(response_time_ms),
            error: if !is_healthy {
                Some("Health check failed".to_string())
            } else {
                None
            },
        };

        // Update registration
        let mut regs = self.registrations.write().await;
        if let Some(reg) = regs.get_mut(&id) {
            reg.last_health_check = Some(health_check.clone());
        }

        Ok(health_check)
    }

    /// Perform health checks on all servers
    pub async fn check_all_health(&self) -> Vec<HealthCheck> {
        let servers = self.servers.read().await;
        let server_ids: Vec<Uuid> = servers.keys().copied().collect();
        drop(servers);

        let mut checks = Vec::new();
        for id in server_ids {
            if let Ok(check) = self.check_server_health(id).await {
                checks.push(check);
            }
        }

        checks
    }

    /// Start background health checking
    pub async fn start_health_monitoring(&self) {
        let servers = self.servers.clone();
        let registrations = self.registrations.clone();
        let interval_secs = self.health_check_interval_secs;

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));

            loop {
                interval.tick().await;

                let servers_guard = servers.read().await;
                let server_ids: Vec<Uuid> = servers_guard.keys().copied().collect();
                drop(servers_guard);

                for id in server_ids {
                    let servers_guard = servers.read().await;
                    if let Some(server) = servers_guard.get(&id).cloned() {
                        drop(servers_guard);

                        let start = std::time::Instant::now();
                        let is_healthy = server.health_check().await.unwrap_or(false);
                        let response_time_ms = start.elapsed().as_millis() as f64;

                        let status = match server.status().await {
                            ServerStatus::Running => HealthStatus::Healthy,
                            ServerStatus::Degraded => HealthStatus::Degraded,
                            ServerStatus::Unhealthy | ServerStatus::Failed => {
                                HealthStatus::Unhealthy
                            }
                            _ => HealthStatus::Unknown,
                        };

                        let mut regs = registrations.write().await;
                        if let Some(reg) = regs.get_mut(&id) {
                            let health_check = HealthCheck {
                                server_id: id,
                                server_name: reg.name.clone(),
                                status,
                                checked_at: Utc::now(),
                                response_time_ms: Some(response_time_ms),
                                error: if !is_healthy {
                                    Some("Health check failed".to_string())
                                } else {
                                    None
                                },
                            };
                            reg.last_health_check = Some(health_check);
                        }
                    }
                }
            }
        });

        let mut handle_lock = self.health_check_handle.write().await;
        *handle_lock = Some(handle);
    }

    /// Stop background health monitoring
    pub async fn stop_health_monitoring(&self) {
        let mut handle_lock = self.health_check_handle.write().await;
        if let Some(handle) = handle_lock.take() {
            handle.abort();
        }
    }

    /// Get registry statistics
    pub async fn statistics(&self) -> RegistryStatistics {
        let regs = self.registrations.read().await;

        let mut healthy = 0;
        let mut degraded = 0;
        let mut unhealthy = 0;
        let mut unknown = 0;

        for reg in regs.values() {
            if let Some(check) = &reg.last_health_check {
                match check.status {
                    HealthStatus::Healthy => healthy += 1,
                    HealthStatus::Degraded => degraded += 1,
                    HealthStatus::Unhealthy => unhealthy += 1,
                    _ => unknown += 1,
                }
            } else {
                unknown += 1;
            }
        }

        RegistryStatistics {
            total_servers: regs.len(),
            healthy_servers: healthy,
            degraded_servers: degraded,
            unhealthy_servers: unhealthy,
            unknown_servers: unknown,
        }
    }
}

impl Default for McpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStatistics {
    /// Total registered servers
    pub total_servers: usize,
    /// Healthy servers
    pub healthy_servers: usize,
    /// Degraded servers
    pub degraded_servers: usize,
    /// Unhealthy servers
    pub unhealthy_servers: usize,
    /// Unknown status servers
    pub unknown_servers: usize,
}

/// Tools list response (from MCP spec)
#[derive(Debug, Deserialize)]
struct ToolsListResponse {
    tools: Vec<Tool>,
    #[allow(dead_code)]
    next_cursor: Option<String>,
}

/// Prompts list response (from MCP spec)
#[derive(Debug, Deserialize)]
struct PromptsListResponse {
    prompts: Vec<Prompt>,
    #[allow(dead_code)]
    next_cursor: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status() {
        let status = HealthStatus::Healthy;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"healthy\"");
    }

    #[tokio::test]
    async fn test_registry_creation() {
        let registry = McpRegistry::new();
        let stats = registry.statistics().await;
        assert_eq!(stats.total_servers, 0);
    }
}
