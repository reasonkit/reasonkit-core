//! MCP Client Implementation
//!
//! Client for connecting to and communicating with MCP servers.
//!
//! This module provides:
//! - Connection management to external MCP servers
//! - Tool execution via RPC
//! - Resource access
//! - Automatic reconnection and error handling

use super::transport::{StdioTransport, Transport};
use super::types::*;
use crate::error::{Error, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// MCP client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpClientConfig {
    /// Server name
    pub name: String,
    /// Server command (e.g., "npx", "node", "python")
    pub command: String,
    /// Command arguments
    pub args: Vec<String>,
    /// Environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Connection timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Auto-reconnect on failure
    #[serde(default = "default_reconnect")]
    pub auto_reconnect: bool,
    /// Maximum reconnection attempts
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_timeout() -> u64 {
    30
}

fn default_reconnect() -> bool {
    true
}

fn default_max_retries() -> u32 {
    3
}

/// MCP client connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionState {
    /// Not connected
    Disconnected,
    /// Connecting
    Connecting,
    /// Connected and initialized
    Connected,
    /// Connection failed
    Failed,
    /// Reconnecting
    Reconnecting,
}

/// MCP client statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientStats {
    /// Total requests sent
    pub requests_sent: u64,
    /// Total responses received
    pub responses_received: u64,
    /// Total errors encountered
    pub errors_total: u64,
    /// Average response time (ms)
    pub avg_response_time_ms: f64,
    /// Connection uptime (seconds)
    pub uptime_secs: u64,
    /// Reconnection attempts
    pub reconnect_attempts: u32,
    /// Last successful request
    pub last_request_at: Option<DateTime<Utc>>,
}

impl Default for ClientStats {
    fn default() -> Self {
        Self {
            requests_sent: 0,
            responses_received: 0,
            errors_total: 0,
            avg_response_time_ms: 0.0,
            uptime_secs: 0,
            reconnect_attempts: 0,
            last_request_at: None,
        }
    }
}

/// MCP client trait
#[async_trait]
pub trait McpClientTrait: Send + Sync {
    /// Connect to the server
    async fn connect(&mut self) -> Result<()>;

    /// Disconnect from the server
    async fn disconnect(&mut self) -> Result<()>;

    /// Get connection state
    async fn state(&self) -> ConnectionState;

    /// List available tools
    async fn list_tools(&self) -> Result<Vec<super::tools::Tool>>;

    /// Call a tool
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<super::tools::ToolResult>;

    /// List available resources
    async fn list_resources(&self) -> Result<Vec<super::tools::ResourceTemplate>>;

    /// Read a resource
    async fn read_resource(&self, uri: &str) -> Result<serde_json::Value>;

    /// Get client statistics
    async fn stats(&self) -> ClientStats;

    /// Perform a health check
    async fn ping(&self) -> Result<bool>;
}

/// Concrete MCP client implementation
pub struct McpClient {
    /// Client ID
    pub id: Uuid,
    /// Client configuration
    pub config: McpClientConfig,
    /// Transport layer
    transport: Arc<RwLock<Option<Arc<dyn Transport>>>>,
    /// Connection state
    state: Arc<RwLock<ConnectionState>>,
    /// Server information (after initialization)
    server_info: Arc<RwLock<Option<ServerInfo>>>,
    /// Server capabilities (after initialization)
    server_capabilities: Arc<RwLock<Option<ServerCapabilities>>>,
    /// Client statistics
    stats: Arc<RwLock<ClientStats>>,
    /// Connected at timestamp
    connected_at: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl McpClient {
    /// Create a new MCP client
    pub fn new(config: McpClientConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            transport: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            server_info: Arc::new(RwLock::new(None)),
            server_capabilities: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(ClientStats::default())),
            connected_at: Arc::new(RwLock::new(None)),
        }
    }

    /// Get server information (must be connected)
    pub async fn server_info(&self) -> Option<ServerInfo> {
        self.server_info.read().await.clone()
    }

    /// Get server capabilities (must be connected)
    pub async fn capabilities(&self) -> Option<ServerCapabilities> {
        self.server_capabilities.read().await.clone()
    }

    /// Update statistics for a successful request
    async fn record_success(&self, response_time_ms: f64) {
        let mut s = self.stats.write().await;
        s.responses_received += 1;
        s.last_request_at = Some(Utc::now());

        // Update average response time (exponential moving average)
        if s.responses_received == 1 {
            s.avg_response_time_ms = response_time_ms;
        } else {
            s.avg_response_time_ms = (s.avg_response_time_ms * 0.9) + (response_time_ms * 0.1);
        }
    }

    /// Update statistics for an error
    async fn record_error(&self) {
        let mut s = self.stats.write().await;
        s.errors_total += 1;
    }

    /// Send a request with automatic retries
    async fn send_request_with_retry(&self, request: McpRequest) -> Result<McpResponse> {
        let mut attempts = 0;
        let max_retries = self.config.max_retries;

        loop {
            let transport_guard = self.transport.read().await;
            let transport = transport_guard
                .as_ref()
                .ok_or_else(|| Error::network("Not connected to server"))?;

            let start = std::time::Instant::now();
            let result = transport.send_request(request.clone()).await;
            let elapsed_ms = start.elapsed().as_millis() as f64;

            match result {
                Ok(response) => {
                    if response.error.is_some() {
                        self.record_error().await;
                    } else {
                        self.record_success(elapsed_ms).await;
                    }
                    return Ok(response);
                }
                Err(e) => {
                    self.record_error().await;
                    attempts += 1;

                    if attempts >= max_retries {
                        return Err(Error::network(format!(
                            "Request failed after {} attempts: {}",
                            attempts, e
                        )));
                    }

                    // Exponential backoff
                    let backoff_ms = 100 * (2_u64.pow(attempts - 1));
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                }
            }
        }
    }
}

#[async_trait]
impl McpClientTrait for McpClient {
    async fn connect(&mut self) -> Result<()> {
        // Set state to connecting
        *self.state.write().await = ConnectionState::Connecting;

        // Convert HashMap to Vec<(String, String)> for env
        let env_vec: Vec<(String, String)> = self.config.env.clone().into_iter().collect();

        // Create stdio transport using spawn method
        let transport =
            StdioTransport::spawn(&self.config.command, self.config.args.clone(), env_vec)
                .await
                .map_err(|e| Error::network(format!("Failed to create transport: {}", e)))?;

        *self.transport.write().await = Some(Arc::new(transport));

        // Send initialize request
        let init_params = serde_json::json!({
            "protocolVersion": crate::mcp::MCP_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "reasonkit-core",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let request = McpRequest::new(
            RequestId::String(Uuid::new_v4().to_string()),
            "initialize",
            Some(init_params),
        );

        let response = self.send_request_with_retry(request).await?;

        if let Some(error) = response.error {
            *self.state.write().await = ConnectionState::Failed;
            return Err(Error::network(format!(
                "Initialize failed: {}",
                error.message
            )));
        }

        // Parse initialization result
        if let Some(result) = response.result {
            if let Ok(init_result) =
                serde_json::from_value::<super::lifecycle::InitializeResult>(result)
            {
                *self.server_info.write().await = Some(init_result.server_info);
                *self.server_capabilities.write().await = Some(init_result.capabilities);
            }
        }

        // Send initialized notification
        let notification = McpNotification {
            jsonrpc: JsonRpcVersion::default(),
            method: "notifications/initialized".to_string(),
            params: None,
        };

        let transport_guard = self.transport.read().await;
        if let Some(transport) = transport_guard.as_ref() {
            transport.send_notification(notification).await.ok();
        }

        *self.state.write().await = ConnectionState::Connected;
        *self.connected_at.write().await = Some(Utc::now());

        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        // Send shutdown request
        let request = McpRequest::new(
            RequestId::String(Uuid::new_v4().to_string()),
            "shutdown",
            None,
        );

        // Best effort - ignore errors
        let _ = self.send_request_with_retry(request).await;

        // Clear transport
        *self.transport.write().await = None;
        *self.state.write().await = ConnectionState::Disconnected;
        *self.connected_at.write().await = None;

        Ok(())
    }

    async fn state(&self) -> ConnectionState {
        *self.state.read().await
    }

    async fn list_tools(&self) -> Result<Vec<super::tools::Tool>> {
        let request = McpRequest::new(
            RequestId::String(Uuid::new_v4().to_string()),
            "tools/list",
            None,
        );

        let response = self.send_request_with_retry(request).await?;

        if let Some(error) = response.error {
            return Err(Error::network(format!(
                "tools/list failed: {}",
                error.message
            )));
        }

        let result = response
            .result
            .ok_or_else(|| Error::network("tools/list response missing result"))?;

        #[derive(Deserialize)]
        struct ToolsListResponse {
            tools: Vec<super::tools::Tool>,
        }

        let tools_response = serde_json::from_value::<ToolsListResponse>(result)
            .map_err(|e| Error::network(format!("Failed to parse tools list: {}", e)))?;

        Ok(tools_response.tools)
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<super::tools::ToolResult> {
        let mut stats = self.stats.write().await;
        stats.requests_sent += 1;
        drop(stats);

        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let request = McpRequest::new(
            RequestId::String(Uuid::new_v4().to_string()),
            "tools/call",
            Some(params),
        );

        let response = self.send_request_with_retry(request).await?;

        if let Some(error) = response.error {
            return Err(Error::network(format!(
                "tools/call failed: {}",
                error.message
            )));
        }

        let result = response
            .result
            .ok_or_else(|| Error::network("tools/call response missing result"))?;

        serde_json::from_value::<super::tools::ToolResult>(result)
            .map_err(|e| Error::network(format!("Failed to parse tool result: {}", e)))
    }

    async fn list_resources(&self) -> Result<Vec<super::tools::ResourceTemplate>> {
        let request = McpRequest::new(
            RequestId::String(Uuid::new_v4().to_string()),
            "resources/list",
            None,
        );

        let response = self.send_request_with_retry(request).await?;

        if let Some(error) = response.error {
            return Err(Error::network(format!(
                "resources/list failed: {}",
                error.message
            )));
        }

        let result = response
            .result
            .ok_or_else(|| Error::network("resources/list response missing result"))?;

        #[derive(Deserialize)]
        struct ResourcesListResponse {
            resources: Vec<super::tools::ResourceTemplate>,
        }

        let resources_response = serde_json::from_value::<ResourcesListResponse>(result)
            .map_err(|e| Error::network(format!("Failed to parse resources list: {}", e)))?;

        Ok(resources_response.resources)
    }

    async fn read_resource(&self, uri: &str) -> Result<serde_json::Value> {
        let params = serde_json::json!({
            "uri": uri
        });

        let request = McpRequest::new(
            RequestId::String(Uuid::new_v4().to_string()),
            "resources/read",
            Some(params),
        );

        let response = self.send_request_with_retry(request).await?;

        if let Some(error) = response.error {
            return Err(Error::network(format!(
                "resources/read failed: {}",
                error.message
            )));
        }

        response
            .result
            .ok_or_else(|| Error::network("resources/read response missing result"))
    }

    async fn stats(&self) -> ClientStats {
        let mut s = self.stats.read().await.clone();

        // Calculate uptime
        if let Some(connected_at) = *self.connected_at.read().await {
            s.uptime_secs = (Utc::now() - connected_at).num_seconds() as u64;
        }

        s
    }

    async fn ping(&self) -> Result<bool> {
        let request = McpRequest::new(RequestId::String(Uuid::new_v4().to_string()), "ping", None);

        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            self.send_request_with_retry(request),
        )
        .await
        {
            Ok(Ok(response)) => Ok(response.error.is_none()),
            Ok(Err(_)) | Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default_values() {
        let config = McpClientConfig {
            name: "test-server".to_string(),
            command: "test".to_string(),
            args: vec![],
            env: HashMap::new(),
            timeout_secs: default_timeout(),
            auto_reconnect: default_reconnect(),
            max_retries: default_max_retries(),
        };

        assert_eq!(config.timeout_secs, 30);
        assert!(config.auto_reconnect);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_connection_state_serialization() {
        let state = ConnectionState::Connected;
        let json = serde_json::to_string(&state).unwrap();
        assert_eq!(json, "\"connected\"");
    }

    #[test]
    fn test_client_stats_default() {
        let stats = ClientStats::default();
        assert_eq!(stats.requests_sent, 0);
        assert_eq!(stats.responses_received, 0);
        assert_eq!(stats.errors_total, 0);
    }

    #[test]
    fn test_client_creation() {
        let config = McpClientConfig {
            name: "test-server".to_string(),
            command: "echo".to_string(),
            args: vec!["hello".to_string()],
            env: HashMap::new(),
            timeout_secs: 30,
            auto_reconnect: true,
            max_retries: 3,
        };

        let client = McpClient::new(config.clone());
        assert_eq!(client.config.name, "test-server");
        assert_eq!(client.config.command, "echo");
    }
}
