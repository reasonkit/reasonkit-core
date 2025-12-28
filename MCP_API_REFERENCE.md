# MCP API Reference

Complete API documentation for the ReasonKit MCP module.

---

## Module: `reasonkit_core::mcp`

### Overview

The `mcp` module provides a Rust-based implementation of the Model Context Protocol (MCP) server registry system.

**Key Components**:
- MCP protocol types (JSON-RPC 2.0)
- Server management and lifecycle
- Transport abstraction (stdio, HTTP/SSE planned)
- Tool discovery and execution
- Health monitoring

---

## Core Types

### `McpMessage`

Enum representing MCP JSON-RPC messages.

```rust
pub enum McpMessage {
    Request(McpRequest),
    Response(McpResponse),
    Notification(McpNotification),
}
```

**Usage**:
```rust
let request = McpMessage::Request(McpRequest {
    jsonrpc: JsonRpcVersion::default(),
    id: RequestId::String("req-1".into()),
    method: "tools/list".into(),
    params: None,
});
```

---

### `McpRequest`

JSON-RPC 2.0 request message.

**Fields**:
- `jsonrpc: JsonRpcVersion` - Always "2.0"
- `id: RequestId` - Request identifier (string or number)
- `method: String` - Method name
- `params: Option<Value>` - Method parameters

**Methods**:

#### `new(id: RequestId, method: impl Into<String>, params: Option<Value>) -> Self`

Create a new request.

```rust
let req = McpRequest::new(
    RequestId::String("1".into()),
    "ping",
    None
);
```

---

### `McpResponse`

JSON-RPC 2.0 response message.

**Fields**:
- `jsonrpc: JsonRpcVersion` - Always "2.0"
- `id: RequestId` - Matching request ID
- `result: Option<Value>` - Success result
- `error: Option<McpError>` - Error result

**Methods**:

#### `success(id: RequestId, result: Value) -> Self`

Create a success response.

```rust
let resp = McpResponse::success(
    RequestId::Number(1),
    serde_json::json!({"status": "ok"})
);
```

#### `error(id: RequestId, error: McpError) -> Self`

Create an error response.

```rust
let resp = McpResponse::error(
    RequestId::Number(1),
    McpError::method_not_found("unknown_method")
);
```

---

### `McpError`

JSON-RPC error object.

**Fields**:
- `code: ErrorCode` - Error code
- `message: String` - Error message
- `data: Option<Value>` - Additional error data

**Error Codes**:
- `PARSE_ERROR` (-32700)
- `INVALID_REQUEST` (-32600)
- `METHOD_NOT_FOUND` (-32601)
- `INVALID_PARAMS` (-32602)
- `INTERNAL_ERROR` (-32603)
- `REQUEST_CANCELLED` (-32800)
- `RESOURCE_NOT_FOUND` (-32801)
- `TOOL_NOT_FOUND` (-32802)
- `INVALID_TOOL_INPUT` (-32803)

**Methods**:

#### `new(code: ErrorCode, message: impl Into<String>) -> Self`

Create a new error.

#### `with_data(code: ErrorCode, message: impl Into<String>, data: Value) -> Self`

Create an error with additional data.

#### `parse_error(message: impl Into<String>) -> Self`

Convenience constructor for parse errors.

#### `method_not_found(method: impl Into<String>) -> Self`

Convenience constructor for method not found errors.

---

## Server Management

### `McpServer`

Concrete MCP server implementation.

**Fields**:
- `id: Uuid` - Unique server ID
- `name: String` - Server name
- `info: ServerInfo` - Server information
- `capabilities: ServerCapabilities` - Server capabilities
- (Private transport and state fields)

**Methods**:

#### `new(name: impl Into<String>, info: ServerInfo, capabilities: ServerCapabilities, transport: Arc<dyn Transport>) -> Self`

Create a new MCP server.

```rust
let server = McpServer::new(
    "my-server",
    server_info,
    capabilities,
    Arc::new(transport)
);
```

#### `async fn set_status(&self, status: ServerStatus)`

Update server status.

#### `async fn record_success(&self, response_time_ms: f64)`

Record a successful request (updates metrics).

#### `async fn record_error(&self)`

Record an error (updates metrics).

#### `fn uptime_secs(&self) -> u64`

Get server uptime in seconds.

---

### `ServerStatus`

Enum representing server status.

```rust
pub enum ServerStatus {
    Starting,
    Running,
    Degraded,
    Unhealthy,
    Stopping,
    Stopped,
    Failed,
}
```

---

### `ServerMetrics`

Server performance metrics.

**Fields**:
- `requests_total: u64` - Total requests handled
- `errors_total: u64` - Total errors encountered
- `avg_response_time_ms: f64` - Average response time
- `last_success_at: Option<DateTime<Utc>>` - Last successful request
- `last_error_at: Option<DateTime<Utc>>` - Last error
- `uptime_secs: u64` - Server uptime

---

### `McpServerTrait`

Async trait for MCP server operations.

**Methods**:

#### `async fn server_info(&self) -> ServerInfo`

Get server information.

#### `async fn capabilities(&self) -> ServerCapabilities`

Get server capabilities.

#### `async fn initialize(&mut self, params: Value) -> Result<Value>`

Initialize the server.

#### `async fn shutdown(&mut self) -> Result<()>`

Shutdown the server gracefully.

#### `async fn send_request(&self, request: McpRequest) -> Result<McpResponse>`

Send a request to the server.

#### `async fn send_notification(&self, notification: McpNotification) -> Result<()>`

Send a notification (no response expected).

#### `async fn status(&self) -> ServerStatus`

Get current server status.

#### `async fn metrics(&self) -> ServerMetrics`

Get server metrics.

#### `async fn health_check(&self) -> Result<bool>`

Perform a health check.

---

## Registry

### `McpRegistry`

Central coordinator for all MCP servers.

**Methods**:

#### `new() -> Self`

Create a new registry with default health check interval (30 seconds).

```rust
let registry = McpRegistry::new();
```

#### `with_health_check_interval(interval_secs: u64) -> Self`

Create a registry with custom health check interval.

```rust
let registry = McpRegistry::with_health_check_interval(60);
```

#### `async fn register_server(&self, server: Arc<dyn McpServerTrait>, tags: Vec<String>) -> Result<Uuid>`

Register a server with tags.

```rust
let server_id = registry.register_server(
    Arc::new(server),
    vec!["thinktool".into(), "reasoning".into()]
).await?;
```

#### `async fn unregister_server(&self, id: Uuid) -> Result<()>`

Unregister a server (with graceful shutdown).

#### `async fn get_server(&self, id: Uuid) -> Option<Arc<dyn McpServerTrait>>`

Get a server by ID.

#### `async fn list_servers(&self) -> Vec<ServerRegistration>`

List all registered servers.

#### `async fn find_servers_by_tag(&self, tag: &str) -> Vec<ServerRegistration>`

Find servers by tag.

```rust
let thinktool_servers = registry.find_servers_by_tag("thinktool").await;
```

#### `async fn list_all_tools(&self) -> Result<Vec<Tool>>`

List all tools from all registered servers.

```rust
let tools = registry.list_all_tools().await?;
for tool in tools {
    println!("{}: {}", tool.name, tool.description.unwrap_or_default());
}
```

#### `async fn check_server_health(&self, id: Uuid) -> Result<HealthCheck>`

Perform health check on a specific server.

#### `async fn check_all_health(&self) -> Vec<HealthCheck>`

Perform health checks on all servers.

#### `async fn start_health_monitoring(&self)`

Start background health monitoring.

```rust
registry.start_health_monitoring().await;
```

#### `async fn stop_health_monitoring(&self)`

Stop background health monitoring.

#### `async fn statistics(&self) -> RegistryStatistics`

Get registry statistics.

```rust
let stats = registry.statistics().await;
println!("Healthy: {}/{}", stats.healthy_servers, stats.total_servers);
```

---

### `ServerRegistration`

Server registration metadata.

**Fields**:
- `id: Uuid` - Server ID
- `name: String` - Server name
- `info: ServerInfo` - Server information
- `capabilities: ServerCapabilities` - Server capabilities
- `registered_at: DateTime<Utc>` - Registration timestamp
- `last_health_check: Option<HealthCheck>` - Last health check result
- `tags: Vec<String>` - Server tags

---

### `HealthCheck`

Health check result.

**Fields**:
- `server_id: Uuid` - Server ID
- `server_name: String` - Server name
- `status: HealthStatus` - Health status
- `checked_at: DateTime<Utc>` - Check timestamp
- `response_time_ms: Option<f64>` - Response time
- `error: Option<String>` - Error message if unhealthy

---

### `HealthStatus`

Health check status.

```rust
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Checking,
    Unknown,
}
```

---

### `RegistryStatistics`

Registry-wide statistics.

**Fields**:
- `total_servers: usize`
- `healthy_servers: usize`
- `degraded_servers: usize`
- `unhealthy_servers: usize`
- `unknown_servers: usize`

---

## Tools

### `Tool`

Tool definition.

**Fields**:
- `name: String` - Tool name (unique per server)
- `description: Option<String>` - Human-readable description
- `input_schema: Value` - JSON Schema for input validation
- `server_id: Option<Uuid>` - Server ID (populated by registry)
- `server_name: Option<String>` - Server name (populated by registry)

**Methods**:

#### `simple(name: impl Into<String>, description: impl Into<String>) -> Self`

Create a simple tool with no required inputs.

```rust
let tool = Tool::simple("ping", "Check server status");
```

#### `with_schema(name: impl Into<String>, description: impl Into<String>, schema: Value) -> Self`

Create a tool with custom input schema.

```rust
let tool = Tool::with_schema(
    "analyze",
    "Analyze text",
    serde_json::json!({
        "type": "object",
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    })
);
```

---

### `ToolInput`

Tool execution input.

**Fields**:
- `name: String` - Tool name
- `arguments: HashMap<String, Value>` - Tool arguments

---

### `ToolResult`

Tool execution result.

**Fields**:
- `content: Vec<ToolResultContent>` - Result content
- `is_error: Option<bool>` - Whether this is an error result

**Methods**:

#### `text(text: impl Into<String>) -> Self`

Create a text result.

```rust
let result = ToolResult::text("Analysis complete");
```

#### `error(message: impl Into<String>) -> Self`

Create an error result.

```rust
let result = ToolResult::error("Analysis failed");
```

#### `with_content(content: Vec<ToolResultContent>) -> Self`

Create a multi-content result.

---

### `ToolResultContent`

Tool result content (enum).

**Variants**:

#### `Text { text: String }`

Text content.

#### `Image { data: String, mime_type: String }`

Image content (base64 encoded).

#### `Resource { uri: String, mime_type: Option<String> }`

Resource reference.

**Methods**:

#### `text(text: impl Into<String>) -> Self`

Create text content.

#### `image(data: impl Into<String>, mime_type: impl Into<String>) -> Self`

Create image content.

#### `resource(uri: impl Into<String>) -> Self`

Create resource reference.

---

## Transport

### `Transport`

Async trait for transport abstraction.

**Methods**:

#### `async fn send_request(&self, request: McpRequest) -> std::io::Result<McpResponse>`

Send a request and wait for response.

#### `async fn send_notification(&self, notification: McpNotification) -> std::io::Result<()>`

Send a notification (no response).

#### `async fn close(&self) -> std::io::Result<()>`

Close the transport.

---

### `StdioTransport`

Standard input/output transport (process spawning).

**Methods**:

#### `async fn spawn(command: impl AsRef<str>, args: Vec<String>, env: Vec<(String, String)>) -> std::io::Result<Self>`

Spawn a new server process.

```rust
let transport = StdioTransport::spawn(
    "python",
    vec!["-m".into(), "mcp_server".into()],
    vec![("API_KEY".into(), "secret".into())]
).await?;
```

---

## Lifecycle

### `InitializeParams`

Initialize request parameters.

**Fields**:
- `protocol_version: String` - Protocol version
- `capabilities: ClientCapabilities` - Client capabilities
- `client_info: ClientInfo` - Client information

**Methods**:

#### `reasonkit() -> Self`

Create default initialize parameters for ReasonKit.

```rust
let params = InitializeParams::reasonkit();
```

#### `with_client_info(name: impl Into<String>, version: impl Into<String>) -> Self`

Create initialize parameters with custom client info.

---

### `InitializeResult`

Initialize response.

**Fields**:
- `protocol_version: String` - Protocol version
- `capabilities: ServerCapabilities` - Server capabilities
- `server_info: ServerInfo` - Server information
- `implementation: Option<Implementation>` - Implementation details

**Methods**:

#### `new(server_info: ServerInfo, capabilities: ServerCapabilities) -> Self`

Create an initialize result.

---

### `ClientInfo`

Client information.

**Fields**:
- `name: String` - Client name
- `version: String` - Client version

---

### `ServerInfo`

Server information.

**Fields**:
- `name: String` - Server name
- `version: String` - Server version
- `description: Option<String>` - Server description
- `vendor: Option<String>` - Vendor/author

---

## Constants

### `MCP_VERSION: &str`

Current MCP protocol version: `"2025-11-25"`

### `DEFAULT_HEALTH_CHECK_INTERVAL_SECS: u64`

Default health check interval: `30` seconds

### `MAX_SERVER_STARTUP_TIMEOUT_SECS: u64`

Maximum server startup timeout: `10` seconds

---

## Examples

### Complete Server Setup

```rust
use reasonkit_core::mcp::{
    McpRegistry, McpServer, StdioTransport,
    ServerInfo, ServerCapabilities, ToolsCapability,
    InitializeParams
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create registry
    let registry = McpRegistry::new();
    registry.start_health_monitoring().await;

    // Spawn server
    let transport = StdioTransport::spawn(
        "rk-thinktool",
        vec!["--module".into(), "gigathink".into()],
        vec![]
    ).await?;

    // Create server
    let mut server = McpServer::new(
        "gigathink",
        ServerInfo {
            name: "GigaThink".into(),
            version: "1.0.0".into(),
            description: Some("Expansive creative thinking".into()),
            vendor: Some("ReasonKit".into()),
        },
        ServerCapabilities {
            tools: Some(ToolsCapability { list_changed: false }),
            ..Default::default()
        },
        Arc::new(transport)
    );

    // Initialize
    let init_result = server.initialize(
        serde_json::to_value(InitializeParams::reasonkit())?
    ).await?;

    println!("Server initialized: {:?}", init_result);

    // Register
    let server_id = registry.register_server(
        Arc::new(server),
        vec!["thinktool".into()]
    ).await?;

    // Discover tools
    let tools = registry.list_all_tools().await?;
    println!("Found {} tools", tools.len());

    Ok(())
}
```

---

## Error Handling

All async methods return `Result<T>` or `std::io::Result<T>`. Use the `?` operator for error propagation:

```rust
let tools = registry.list_all_tools().await?;
```

Common errors:
- `Error::NotFound`: Server not found in registry
- `Error::Network`: Network/transport error
- `std::io::Error`: Transport-level errors

---

## Thread Safety

All types are designed for concurrent use:
- `McpRegistry`: Uses `Arc<RwLock<...>>` internally
- `McpServer`: Uses `Arc<RwLock<...>>` for state
- All methods are `async` and safe to call concurrently

---

## Performance Tips

1. **Registry Size**: Keep total servers < 100 for optimal health check performance
2. **Health Interval**: Increase interval for production (60-120s)
3. **Tool Caching**: Cache `list_all_tools()` results if servers are stable
4. **Background Tasks**: Use `start_health_monitoring()` for automated monitoring

---

**API Version**: 1.0.0
**Last Updated**: 2025-12-22
**Compatibility**: MCP Specification 2025-11-25
