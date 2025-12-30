//! MCP Server Implementation
//!
//! Base server trait and concrete implementation for MCP servers.

use super::lifecycle::InitializeResult;
use super::tools::{Tool, ToolHandler, ToolResult};
use super::transport::Transport;
use super::types::*;
use crate::error::{Error, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Server status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerStatus {
    /// Server is starting up
    Starting,
    /// Server is running and healthy
    Running,
    /// Server is degraded (responding slowly)
    Degraded,
    /// Server is not responding
    Unhealthy,
    /// Server is shutting down
    Stopping,
    /// Server has stopped
    Stopped,
    /// Server encountered a fatal error
    Failed,
}

/// Server metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerMetrics {
    /// Total requests handled
    pub requests_total: u64,
    /// Total errors encountered
    pub errors_total: u64,
    /// Average response time (ms)
    pub avg_response_time_ms: f64,
    /// Last successful request timestamp
    pub last_success_at: Option<DateTime<Utc>>,
    /// Last error timestamp
    pub last_error_at: Option<DateTime<Utc>>,
    /// Uptime in seconds
    pub uptime_secs: u64,
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self {
            requests_total: 0,
            errors_total: 0,
            avg_response_time_ms: 0.0,
            last_success_at: None,
            last_error_at: None,
            uptime_secs: 0,
        }
    }
}

/// MCP server trait
#[async_trait]
pub trait McpServerTrait: Send + Sync {
    /// Get server information
    async fn server_info(&self) -> ServerInfo;

    /// Get server capabilities
    async fn capabilities(&self) -> ServerCapabilities;

    /// Initialize the server
    async fn initialize(&mut self, params: serde_json::Value) -> Result<serde_json::Value>;

    /// Shutdown the server
    async fn shutdown(&mut self) -> Result<()>;

    /// Send a request to the server
    async fn send_request(&self, request: McpRequest) -> Result<McpResponse>;

    /// Send a notification to the server (no response expected)
    async fn send_notification(&self, notification: McpNotification) -> Result<()>;

    /// Get current server status
    async fn status(&self) -> ServerStatus;

    /// Get server metrics
    async fn metrics(&self) -> ServerMetrics;

    /// Perform a health check
    async fn health_check(&self) -> Result<bool>;

    /// List available tools
    async fn list_tools(&self) -> Vec<Tool>;

    /// Call a tool
    async fn call_tool(
        &self,
        name: &str,
        arguments: HashMap<String, serde_json::Value>,
    ) -> Result<ToolResult>;

    /// Register a tool
    async fn register_tool(&self, tool: Tool, handler: Arc<dyn ToolHandler>);
}

type ToolRegistry = HashMap<String, (Tool, Arc<dyn ToolHandler>)>;

/// Concrete MCP server implementation
pub struct McpServer {
    /// Server ID
    pub id: Uuid,
    /// Server name
    pub name: String,
    /// Server information
    pub info: ServerInfo,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Transport layer
    transport: Arc<dyn Transport>,
    /// Current status
    status: Arc<RwLock<ServerStatus>>,
    /// Server metrics
    metrics: Arc<RwLock<ServerMetrics>>,
    /// Server started at
    started_at: DateTime<Utc>,
    /// Registered tools
    tools: Arc<RwLock<ToolRegistry>>,
}

impl McpServer {
    /// Create a new MCP server
    pub fn new(
        name: impl Into<String>,
        info: ServerInfo,
        capabilities: ServerCapabilities,
        transport: Arc<dyn Transport>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            info,
            capabilities,
            transport,
            status: Arc::new(RwLock::new(ServerStatus::Starting)),
            metrics: Arc::new(RwLock::new(ServerMetrics::default())),
            started_at: Utc::now(),
            tools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update server status
    pub async fn set_status(&self, status: ServerStatus) {
        let mut s = self.status.write().await;
        *s = status;
    }

    /// Record a successful request
    pub async fn record_success(&self, response_time_ms: f64) {
        let mut m = self.metrics.write().await;
        m.requests_total += 1;
        m.last_success_at = Some(Utc::now());

        // Update average response time (exponential moving average)
        if m.requests_total == 1 {
            m.avg_response_time_ms = response_time_ms;
        } else {
            m.avg_response_time_ms = (m.avg_response_time_ms * 0.9) + (response_time_ms * 0.1);
        }
    }

    /// Record an error
    pub async fn record_error(&self) {
        let mut m = self.metrics.write().await;
        m.errors_total += 1;
        m.last_error_at = Some(Utc::now());
    }

    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        (Utc::now() - self.started_at).num_seconds() as u64
    }
}

#[async_trait]
impl McpServerTrait for McpServer {
    async fn server_info(&self) -> ServerInfo {
        self.info.clone()
    }

    async fn capabilities(&self) -> ServerCapabilities {
        self.capabilities.clone()
    }

    async fn initialize(&mut self, params: serde_json::Value) -> Result<serde_json::Value> {
        let request = McpRequest::new(
            RequestId::String(Uuid::new_v4().to_string()),
            "initialize",
            Some(params),
        );

        let start = std::time::Instant::now();
        let response = self
            .transport
            .send_request(request)
            .await
            .map_err(|e| Error::network(format!("Initialize failed: {}", e)))?;

        let elapsed_ms = start.elapsed().as_millis() as f64;

        if let Some(error) = response.error {
            self.record_error().await;
            return Err(Error::network(format!(
                "Initialize error: {}",
                error.message
            )));
        }

        self.record_success(elapsed_ms).await;
        self.set_status(ServerStatus::Running).await;

        response
            .result
            .ok_or_else(|| Error::network("Initialize response missing result"))
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.set_status(ServerStatus::Stopping).await;

        let request = McpRequest::new(
            RequestId::String(Uuid::new_v4().to_string()),
            "shutdown",
            None,
        );

        let response = self
            .transport
            .send_request(request)
            .await
            .map_err(|e| Error::network(format!("Shutdown failed: {}", e)))?;

        if response.error.is_some() {
            self.set_status(ServerStatus::Failed).await;
        } else {
            self.set_status(ServerStatus::Stopped).await;
        }

        Ok(())
    }

    async fn send_request(&self, request: McpRequest) -> Result<McpResponse> {
        let start = std::time::Instant::now();

        let response = self
            .transport
            .send_request(request)
            .await
            .map_err(|e| Error::network(format!("Request failed: {}", e)))?;

        let elapsed_ms = start.elapsed().as_millis() as f64;

        if response.error.is_some() {
            self.record_error().await;
        } else {
            self.record_success(elapsed_ms).await;
        }

        Ok(response)
    }

    async fn send_notification(&self, notification: McpNotification) -> Result<()> {
        self.transport
            .send_notification(notification)
            .await
            .map_err(|e| Error::network(format!("Notification failed: {}", e)))
    }

    async fn status(&self) -> ServerStatus {
        *self.status.read().await
    }

    async fn metrics(&self) -> ServerMetrics {
        let mut m = self.metrics.read().await.clone();
        m.uptime_secs = self.uptime_secs();
        m
    }

    async fn health_check(&self) -> Result<bool> {
        let request = McpRequest::new(RequestId::String(Uuid::new_v4().to_string()), "ping", None);

        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            self.transport.send_request(request),
        )
        .await
        {
            Ok(Ok(response)) => {
                if response.error.is_none() {
                    self.set_status(ServerStatus::Running).await;
                    Ok(true)
                } else {
                    self.set_status(ServerStatus::Degraded).await;
                    Ok(false)
                }
            }
            Ok(Err(_)) | Err(_) => {
                self.set_status(ServerStatus::Unhealthy).await;
                Ok(false)
            }
        }
    }

    async fn list_tools(&self) -> Vec<Tool> {
        let tools = self.tools.read().await;
        tools.values().map(|(t, _)| t.clone()).collect()
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: HashMap<String, serde_json::Value>,
    ) -> Result<ToolResult> {
        let handler = {
            let tools = self.tools.read().await;
            tools.get(name).map(|(_, handler)| Arc::clone(handler))
        };

        match handler {
            Some(handler) => handler.call(arguments).await,
            None => Err(Error::Mcp(format!("Tool not found: {}", name))),
        }
    }

    async fn register_tool(&self, tool: Tool, handler: Arc<dyn ToolHandler>) {
        let mut tools = self.tools.write().await;
        tools.insert(tool.name.clone(), (tool, handler));
    }
}

/// Server-side Stdio Transport (uses current process stdin/stdout)
pub struct ServerStdioTransport {
    stdout: tokio::sync::Mutex<tokio::io::Stdout>,
}

impl ServerStdioTransport {
    pub fn new() -> Self {
        Self {
            stdout: tokio::sync::Mutex::new(tokio::io::stdout()),
        }
    }
}

impl Default for ServerStdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Transport for ServerStdioTransport {
    async fn send_request(&self, _request: McpRequest) -> std::io::Result<McpResponse> {
        // Server sending request to client (e.g. sampling) - not implemented yet
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Server-to-client requests not supported yet",
        ))
    }

    async fn send_notification(&self, notification: McpNotification) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;
        let json = serde_json::to_string(&notification)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut stdout = self.stdout.lock().await;
        stdout.write_all(json.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
        Ok(())
    }

    async fn close(&self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run the MCP server
pub async fn run_server() -> Result<()> {
    use tokio::io::AsyncBufReadExt;
    use tokio::io::AsyncWriteExt;

    let transport = Arc::new(ServerStdioTransport::new());

    let info = ServerInfo {
        name: "reasonkit-core".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: Some("ReasonKit Core MCP Server".to_string()),
        vendor: Some("ReasonKit Team".to_string()),
    };

    let capabilities = ServerCapabilities {
        logging: Some(LoggingCapability {}),
        prompts: Some(PromptsCapability { list_changed: true }),
        resources: Some(ResourcesCapability {
            subscribe: true,
            list_changed: true,
        }),
        tools: Some(ToolsCapability { list_changed: true }),
    };

    let server = McpServer::new("reasonkit-core", info, capabilities, transport.clone());

    // Main loop
    let stdin = tokio::io::stdin();
    let mut reader = tokio::io::BufReader::new(stdin);
    let mut line = String::new();

    eprintln!("ReasonKit Core MCP Server running on stdio...");

    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .await
            .map_err(|e| Error::network(format!("Failed to read line: {}", e)))?;

        if bytes_read == 0 {
            break; // EOF
        }

        let msg: serde_json::Value = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to parse JSON: {}", e);

                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": serde_json::Value::Null,
                    "error": {
                        "code": ErrorCode::PARSE_ERROR.0,
                        "message": e.to_string()
                    }
                });

                let response_str = serde_json::to_string(&response).map_err(Error::from)?;
                let mut stdout = transport.stdout.lock().await;
                stdout
                    .write_all(response_str.as_bytes())
                    .await
                    .map_err(Error::from)?;
                stdout.write_all(b"\n").await.map_err(Error::from)?;
                stdout.flush().await.map_err(Error::from)?;
                continue;
            }
        };

        // Handle JSON-RPC message
        if let Some(method) = msg.get("method").and_then(|m| m.as_str()) {
            // It's a request or notification
            if let Some(id) = msg.get("id") {
                // Request
                let result: Result<serde_json::Value> = match method {
                    "initialize" => {
                        server.set_status(ServerStatus::Running).await;
                        let init =
                            InitializeResult::new(server.info.clone(), server.capabilities.clone());
                        Ok(serde_json::to_value(init).map_err(Error::from)?)
                    }
                    "shutdown" => {
                        server.set_status(ServerStatus::Stopping).await;
                        Ok(serde_json::json!(null))
                    }
                    "ping" => Ok(serde_json::json!({})),
                    "tools/list" => {
                        let tools = server.list_tools().await;
                        Ok(serde_json::json!({ "tools": tools }))
                    }
                    "tools/call" => {
                        let params = match msg.get("params").and_then(|p| p.as_object()) {
                            Some(params) => params,
                            None => Err(Error::Mcp("Invalid params".to_string()))?,
                        };

                        let name = match params.get("name").and_then(|v| v.as_str()) {
                            Some(name) => name,
                            None => Err(Error::Mcp("Missing tool name".to_string()))?,
                        };

                        let args: std::result::Result<HashMap<String, serde_json::Value>, Error> =
                            match params.get("arguments") {
                                Some(v) if v.is_object() => {
                                    serde_json::from_value(v.clone()).map_err(Error::from)
                                }
                                Some(_) => Err(Error::Mcp("Invalid tool arguments".to_string())),
                                None => Ok(HashMap::new()),
                            };

                        match args {
                            Ok(args) => match server.call_tool(name, args).await {
                                Ok(res) => serde_json::to_value(res).map_err(Error::from),
                                Err(e) => Err(e),
                            },
                            Err(e) => Err(e),
                        }
                    }
                    _ => Err(Error::Mcp(format!("Method not found: {}", method))),
                };

                let (code, message) = match &result {
                    Err(Error::Mcp(message)) if message.starts_with("Method not found:") => {
                        (ErrorCode::METHOD_NOT_FOUND.0, message.clone())
                    }
                    Err(Error::Mcp(message)) if message.starts_with("Tool not found:") => {
                        (ErrorCode::TOOL_NOT_FOUND.0, message.clone())
                    }
                    Err(Error::Mcp(message)) => (ErrorCode::INVALID_PARAMS.0, message.clone()),
                    Err(e) => (ErrorCode::INTERNAL_ERROR.0, e.to_string()),
                    Ok(_) => (0, String::new()),
                };

                let response = match result {
                    Ok(res) => serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": res
                    }),
                    Err(_) => serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": code,
                            "message": message
                        }
                    }),
                };

                let json_line = serde_json::to_string(&response).map_err(Error::from)?;
                let mut stdout = transport.stdout.lock().await;
                stdout
                    .write_all(json_line.as_bytes())
                    .await
                    .map_err(Error::from)?;
                stdout.write_all(b"\n").await.map_err(Error::from)?;
                stdout.flush().await.map_err(Error::from)?;

                if method == "shutdown" {
                    server.set_status(ServerStatus::Stopped).await;
                    break;
                }
            } else {
                // Notification
                if method == "notifications/initialized" {
                    // Handle initialized notification
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::tools::ToolResultContent;

    #[test]
    fn test_server_status() {
        let status = ServerStatus::Running;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"running\"");
    }

    #[test]
    fn test_metrics_default() {
        let metrics = ServerMetrics::default();
        assert_eq!(metrics.requests_total, 0);
        assert_eq!(metrics.errors_total, 0);
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let transport = Arc::new(ServerStdioTransport::new());
        let info = ServerInfo {
            name: "test".to_string(),
            version: "1.0".to_string(),
            description: None,
            vendor: None,
        };
        let capabilities = ServerCapabilities::default();
        let server = McpServer::new("test", info, capabilities, transport);

        struct EchoTool;
        #[async_trait]
        impl ToolHandler for EchoTool {
            async fn call(&self, args: HashMap<String, serde_json::Value>) -> Result<ToolResult> {
                let msg = args
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");
                Ok(ToolResult::text(format!("Echo: {}", msg)))
            }
        }

        server
            .register_tool(Tool::simple("echo", "Echoes back"), Arc::new(EchoTool))
            .await;

        let tools = server.list_tools().await;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        let mut args = HashMap::new();
        args.insert("message".to_string(), serde_json::json!("hello"));

        let result = server.call_tool("echo", args).await.unwrap();
        match &result.content[0] {
            ToolResultContent::Text { text } => assert_eq!(text, "Echo: hello"),
            _ => panic!("Wrong content type"),
        }
    }
}
