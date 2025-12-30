# MCP (Model Context Protocol) Module

Comprehensive MCP server and client implementation for ReasonKit Core.

## Overview

The MCP module provides:

- **MCP Client**: Connect to external MCP servers and call tools
- **MCP Server**: Serve tools and resources via JSON-RPC 2.0
- **MCP Registry**: Dynamically discover and manage multiple MCP servers
- **Health Monitoring**: Automatic health checks with status tracking
- **Protocol Compliance**: Full MCP specification (2025-11-25) support
- **Rust-Only**: Zero Node.js dependencies (CONS-001 compliant)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ MCP Registry (Coordinator)                                  │
│   - Dynamic server discovery                                │
│   - Health monitoring (30s interval)                        │
│   - Capability aggregation                                  │
│   - Statistics and metrics                                  │
├─────────────────────────────────────────────────────────────┤
│ MCP Client (Consumer) **NEW**                               │
│   - Connect to external MCP servers                         │
│   - Tool execution via JSON-RPC 2.0                         │
│   - Resource access                                         │
│   - Automatic retry with exponential backoff                │
│   - Connection statistics tracking                          │
├─────────────────────────────────────────────────────────────┤
│ MCP Server (Provider)                                       │
│   - Serve tools and resources                               │
│   - Lifecycle management                                    │
│   - Health checks and metrics                               │
│   - Status monitoring                                       │
├─────────────────────────────────────────────────────────────┤
│ Transport Layer                                             │
│   - JSON-RPC 2.0 over stdio (implemented)                   │
│   - HTTP/SSE (placeholder)                                  │
│   - WebSocket (future)                                      │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Files

| File            | Purpose                   | Key Types                                        |
| --------------- | ------------------------- | ------------------------------------------------ |
| `mod.rs`        | Module entry point        | Re-exports all public types                      |
| `types.rs`      | MCP protocol types        | `McpRequest`, `McpResponse`, `McpError`          |
| **`client.rs`** | **NEW** MCP client        | `McpClient`, `McpClientConfig`, `McpClientTrait` |
| `server.rs`     | MCP server implementation | `McpServer`, `McpServerTrait`, `ServerStatus`    |
| `registry.rs`   | Server registry           | `McpRegistry`, `HealthCheck`, `HealthStatus`     |
| `transport.rs`  | Transport implementations | `StdioTransport`, `Transport` trait              |
| `tools.rs`      | Tool definitions          | `Tool`, `ToolResult`, `ResourceTemplate`         |
| `lifecycle.rs`  | Lifecycle management      | `InitializeParams`, `InitializeResult`           |

## Quick Start

### MCP Client Usage (NEW)

```rust
use reasonkit_core::mcp::{McpClient, McpClientConfig, McpClientTrait};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure client to connect to sequential-thinking MCP server
    let config = McpClientConfig {
        name: "sequential-thinking".to_string(),
        command: "npx".to_string(),
        args: vec![
            "-y".to_string(),
            "@modelcontextprotocol/server-sequential-thinking".to_string()
        ],
        env: HashMap::new(),
        timeout_secs: 30,
        auto_reconnect: true,
        max_retries: 3,
    };

    // Create and connect client
    let mut client = McpClient::new(config);
    client.connect().await?;

    // List available tools
    let tools = client.list_tools().await?;
    for tool in &tools {
        println!("Tool: {} - {}", tool.name,
                 tool.description.as_deref().unwrap_or(""));
    }

    // Call a tool
    let result = client.call_tool(
        "think",
        serde_json::json!({
            "query": "Explain chain-of-thought reasoning",
            "depth": 3
        })
    ).await?;

    println!("Result: {:?}", result);

    // Get client statistics
    let stats = client.stats().await;
    println!("Requests sent: {}", stats.requests_sent);
    println!("Avg response time: {:.2}ms", stats.avg_response_time_ms);

    // Check server health
    let healthy = client.ping().await?;
    println!("Server healthy: {}", healthy);

    // Disconnect
    client.disconnect().await?;

    Ok(())
}
```

### MCP Registry Usage

```rust
use reasonkit_core::mcp::{McpRegistry, McpServer, ServerInfo, ServerCapabilities};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create registry
    let registry = McpRegistry::new();

    // Start background health monitoring
    registry.start_health_monitoring().await;

    // Register servers (example - you'd create actual servers)
    // let server = Arc::new(create_my_server());
    // let id = registry.register_server(server, vec!["thinktool".to_string()]).await?;

    // List all available tools from all servers
    let all_tools = registry.list_all_tools().await?;
    println!("Total tools: {}", all_tools.len());

    // Find servers by tag
    let thinktool_servers = registry.find_servers_by_tag("thinktool").await;
    println!("ThinkTool servers: {}", thinktool_servers.len());

    // Get registry statistics
    let stats = registry.statistics().await;
    println!("Total servers: {}", stats.total_servers);
    println!("Healthy: {}", stats.healthy_servers);
    println!("Degraded: {}", stats.degraded_servers);

    // Stop health monitoring when done
    registry.stop_health_monitoring().await;

    Ok(())
}
```

## MCP Protocol Compliance

Based on MCP specification version: **2025-11-25**

### Implemented Features

- ✅ JSON-RPC 2.0 messaging
- ✅ Lifecycle management (`initialize`, `shutdown`)
- ✅ Tool primitives (`tools/list`, `tools/call`)
- ✅ Resource primitives (`resources/list`, `resources/read`)
- ✅ Prompt primitives (types defined)
- ✅ Notifications
- ✅ Progress tracking (types defined)
- ✅ Cancellation (types defined)
- ✅ Health checks (`ping`)

### Client Features (NEW)

| Feature               | Status | Description                         |
| --------------------- | ------ | ----------------------------------- |
| Connection Management | ✅     | Connect, disconnect, reconnect      |
| Tool Discovery        | ✅     | List available tools                |
| Tool Execution        | ✅     | Call tools with arguments           |
| Resource Access       | ✅     | List and read resources             |
| Automatic Retry       | ✅     | Exponential backoff (max 3 retries) |
| Statistics            | ✅     | Request/response metrics            |
| Health Checks         | ✅     | Ping with 5s timeout                |
| State Management      | ✅     | Connection state tracking           |

## Configuration

### Client Configuration (NEW)

```rust
pub struct McpClientConfig {
    /// Server name for identification
    pub name: String,

    /// Command to execute (e.g., "npx", "node", "python")
    pub command: String,

    /// Command arguments
    pub args: Vec<String>,

    /// Environment variables
    pub env: HashMap<String, String>,

    /// Connection timeout (default: 30s)
    pub timeout_secs: u64,

    /// Auto-reconnect on failure (default: true)
    pub auto_reconnect: bool,

    /// Maximum retry attempts (default: 3)
    pub max_retries: u32,
}
```

### Example MCP Server Commands

```bash
# Sequential Thinking Server (Node.js)
npx -y @modelcontextprotocol/server-sequential-thinking

# Filesystem Server (Python)
uvx mcp-server-filesystem /path/to/directory

# GitHub Server (Node.js)
npx -y @modelcontextprotocol/server-github

# Custom ReasonKit ThinkTool Server (Future)
rk-thinktool --module gigathink
```

## Error Handling

### Error Types

All MCP operations return `Result<T, Error>` where errors include:

- `Error::network()` - Network/transport errors
- `Error::validation()` - Schema validation errors
- `Error::NotFound` - Server/resource not found

### Retry Strategy (Client)

The client implements exponential backoff:

1. Initial request (0ms delay)
2. First retry (100ms delay)
3. Second retry (200ms delay)
4. Third retry (400ms delay)
5. Give up, return error

## Connection States

```rust
pub enum ConnectionState {
    Disconnected,  // Not connected
    Connecting,    // Connection in progress
    Connected,     // Connected and initialized
    Failed,        // Connection failed
    Reconnecting,  // Attempting to reconnect
}
```

## Statistics

### Client Statistics (NEW)

```rust
pub struct ClientStats {
    pub requests_sent: u64,
    pub responses_received: u64,
    pub errors_total: u64,
    pub avg_response_time_ms: f64,
    pub uptime_secs: u64,
    pub reconnect_attempts: u32,
    pub last_request_at: Option<DateTime<Utc>>,
}
```

### Registry Statistics

```rust
pub struct RegistryStatistics {
    pub total_servers: usize,
    pub healthy_servers: usize,
    pub degraded_servers: usize,
    pub unhealthy_servers: usize,
    pub unknown_servers: usize,
}
```

## Integration with ThinkTools

The MCP registry is designed to work seamlessly with ReasonKit's ThinkTools:

```rust
// Example: Registering ThinkTool modules as MCP servers

use reasonkit_core::mcp::{
    McpRegistry, ServerInfo, ServerCapabilities, ToolsCapability, Tool
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let registry = McpRegistry::new();

    // Each ThinkTool can be exposed as an MCP server:
    // - GigaThink: Expansive creative thinking
    // - LaserLogic: Precision deductive reasoning
    // - BedRock: First principles decomposition
    // - ProofGuard: Multi-source verification
    // - BrutalHonesty: Adversarial self-critique

    // Tools are automatically discovered via tools/list
    let tools = registry.list_all_tools().await?;

    for tool in tools {
        if let Some(server_name) = tool.server_name {
            println!("ThinkTool: {} from {}", tool.name, server_name);
        }
    }

    Ok(())
}
```

## Testing

```bash
# Run MCP module tests
cargo test --lib mcp

# Run with verbose output
cargo test --lib mcp -- --nocapture

# Run specific client tests
cargo test --lib mcp::client::tests
```

## Performance Considerations

### Client

- **Connection pooling**: Reuse connections when possible
- **Retry backoff**: Exponential backoff prevents server overload
- **Timeout**: 5s health check timeout prevents hanging
- **Statistics**: Minimal overhead, exponential moving average for response time

### Registry

- **Background monitoring**: Runs in separate tokio task
- **Read-write locks**: Concurrent reads, exclusive writes
- **Lazy health checks**: Only on demand or background interval

### Transport

- **Stdio buffering**: Uses `BufReader` for efficient line reading
- **Process management**: Proper cleanup on disconnect
- **JSON streaming**: Line-delimited JSON for easy parsing

## Security

Following CONS-001 (No Node.js):

- ✅ All MCP clients and servers implemented in Rust
- ✅ No Node.js dependencies
- ✅ Memory-safe communication
- ✅ Process isolation via stdio transport
- ✅ Proper error handling without exposing internals

## Future Enhancements

- [ ] HTTP/SSE transport implementation
- [ ] WebSocket transport
- [ ] Connection pooling for clients
- [ ] Load balancing across multiple servers
- [ ] Circuit breaker pattern
- [ ] Rate limiting
- [ ] Authentication/authorization
- [ ] Encryption (TLS for HTTP transport)
- [ ] Server discovery via mDNS/DNS-SD
- [ ] Dynamic capability negotiation

## References

- **MCP Specification**: https://spec.modelcontextprotocol.io
- **MCP GitHub**: https://github.com/modelcontextprotocol
- **JSON-RPC 2.0**: https://www.jsonrpc.org/specification
- **ReasonKit Documentation**: ../../../docs/

## License

Apache 2.0 (consistent with reasonkit-core)

---

**Version**: 1.0.0
**Last Updated**: 2025-12-23
**Maintainer**: ReasonKit Core Team
