# MCP Server Documentation

ReasonKit Core MCP (Model Context Protocol) Server Implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Server Setup](#server-setup)
4. [Available Tools](#available-tools)
5. [Configuration Options](#configuration-options)
6. [Client Integration](#client-integration)
7. [Claude Desktop Integration](#claude-desktop-integration)
8. [Tool Development](#tool-development)
9. [Example Usage](#example-usage)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

---

## Overview

ReasonKit Core provides a Rust-based MCP server implementation that exposes reasoning tools and resources to AI clients. The server follows the MCP specification (version 2025-11-25) and supports:

- **Tools**: Reasoning modules (GigaThink, LaserLogic, BedRock, ProofGuard, BrutalHonesty)
- **Resources**: Session logs, configuration, knowledge base access
- **Prompts**: Pre-defined reasoning templates
- **Transport**: JSON-RPC 2.0 over stdio (primary), HTTP/SSE (planned)

### Architecture

```
+---------------------------------------------------------------+
| MCP Registry (Coordinator)                                    |
|   - Dynamic server discovery                                  |
|   - Health monitoring (30s interval)                          |
|   - Capability aggregation                                    |
+---------------------------------------------------------------+
| MCP Client (Consumer)                                         |
|   - Connect to external MCP servers                           |
|   - Tool execution via JSON-RPC 2.0                           |
|   - Automatic retry with exponential backoff                  |
+---------------------------------------------------------------+
| MCP Server (Provider)                                         |
|   - Serve tools and resources                                 |
|   - Lifecycle management (initialize, shutdown)               |
|   - Health checks and metrics                                 |
+---------------------------------------------------------------+
| Transport Layer                                               |
|   - JSON-RPC 2.0 over stdio (implemented)                     |
|   - HTTP/SSE (placeholder)                                    |
+---------------------------------------------------------------+
```

### Key Features

- **Zero Node.js Dependencies**: 100% Rust implementation (CONS-001 compliant)
- **MCP 2025-11-25 Compliant**: Full protocol specification support
- **Health Monitoring**: Automatic server health checks
- **Tool Registry**: Dynamic tool discovery and registration
- **Protocol Delta**: Immutable citation ledger for source verification

---

## Quick Start

### Running the MCP Server

```bash
# Run the MCP server on stdio (for integration with Claude, etc.)
rk-core mcp-server

# Or using cargo
cargo run --bin rk-core -- mcp-server
```

### Running from Source

```bash
cd /home/zyxsys/RK-PROJECT/reasonkit-core
cargo build --release
./target/release/rk-core mcp-server
```

The server will output status to stderr and communicate via JSON-RPC 2.0 on stdin/stdout:

```
ReasonKit Core MCP Server running on stdio...
```

---

## Server Setup

### Prerequisites

- Rust 1.75+ (for building from source)
- Or pre-built `rk-core` binary

### Installation Options

#### Option 1: Install via Cargo

```bash
cargo install reasonkit
```

#### Option 2: Build from Source

```bash
git clone https://github.com/reasonkit/reasonkit-core.git
cd reasonkit-core
cargo build --release
```

#### Option 3: Universal Installer

```bash
curl -fsSL https://reasonkit.sh/install | bash
```

### Verifying Installation

```bash
# Check version
rk-core --version

# Test MCP server startup
echo '{"jsonrpc":"2.0","id":1,"method":"ping"}' | rk-core mcp-server
```

---

## Available Tools

### Core Reasoning Tools (via `rk_reason`)

| Tool          | Description                                | Profile      |
| ------------- | ------------------------------------------ | ------------ |
| `rk_reason`   | Execute structured reasoning chain         | All profiles |
| `rk_retrieve` | Semantic search across knowledge base      | N/A          |
| `rk_verify`   | Verify claims using Triangulation Protocol | N/A          |

### ThinkTool Modules

The following cognitive modules can be activated via the `rk_reason` tool:

| Module            | Purpose                                          | Best For                                    |
| ----------------- | ------------------------------------------------ | ------------------------------------------- |
| **GigaThink**     | Expansive creative thinking, 10+ perspectives    | Brainstorming, creative problem-solving     |
| **LaserLogic**    | Precision deductive reasoning, fallacy detection | Logical analysis, debugging                 |
| **BedRock**       | First principles decomposition                   | Architecture decisions, root cause analysis |
| **ProofGuard**    | Multi-source verification                        | Fact-checking, research validation          |
| **BrutalHonesty** | Adversarial self-critique                        | Code review, decision validation            |

### Protocol Delta Tools

For source verification and citation management:

| Tool                | Description                                 |
| ------------------- | ------------------------------------------- |
| `delta_anchor`      | Anchor content to immutable citation ledger |
| `delta_verify`      | Verify content against anchored hash        |
| `delta_lookup`      | Look up anchor by hash                      |
| `delta_list_by_url` | List all anchors for a URL                  |
| `delta_stats`       | Get ledger statistics                       |

### Reasoning Profiles

| Profile      | Modules Used                | Confidence Target | Use Case                     |
| ------------ | --------------------------- | ----------------- | ---------------------------- |
| `--quick`    | GigaThink, LaserLogic       | 70%               | Fast answers, simple queries |
| `--balanced` | GT, LL, BedRock, ProofGuard | 80%               | General reasoning            |
| `--deep`     | All 5 modules               | 85%               | Complex analysis             |
| `--paranoid` | All 5 + validation          | 95%               | High-stakes decisions        |

---

## Configuration Options

### Server Capabilities

The server advertises the following capabilities during initialization:

```json
{
  "tools": { "listChanged": true },
  "resources": { "subscribe": true, "listChanged": true },
  "prompts": { "listChanged": true },
  "logging": {}
}
```

### Client Configuration

When connecting to the ReasonKit MCP server from a client:

```rust
use reasonkit_core::mcp::{McpClientConfig, McpClient};
use std::collections::HashMap;

let config = McpClientConfig {
    name: "reasonkit-core".to_string(),
    command: "rk-core".to_string(),
    args: vec!["mcp-server".to_string()],
    env: HashMap::new(),
    timeout_secs: 30,        // Connection timeout
    auto_reconnect: true,    // Reconnect on failure
    max_retries: 3,          // Maximum retry attempts
};
```

### Environment Variables

| Variable                | Description                                 | Default                        |
| ----------------------- | ------------------------------------------- | ------------------------------ |
| `REASONKIT_LOG`         | Log level (trace, debug, info, warn, error) | `info`                         |
| `REASONKIT_DATA_DIR`    | Data directory path                         | `~/.reasonkit`                 |
| `REASONKIT_LEDGER_PATH` | Protocol Delta ledger path                  | `~/.reasonkit/proof_ledger.db` |

---

## Client Integration

### Connecting as an MCP Client

```rust
use reasonkit_core::mcp::{McpClient, McpClientConfig, McpClientTrait};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure client
    let config = McpClientConfig {
        name: "reasonkit-core".to_string(),
        command: "rk-core".to_string(),
        args: vec!["mcp-server".to_string()],
        env: HashMap::new(),
        timeout_secs: 30,
        auto_reconnect: true,
        max_retries: 3,
    };

    // Create and connect
    let mut client = McpClient::new(config);
    client.connect().await?;

    // List available tools
    let tools = client.list_tools().await?;
    for tool in &tools {
        println!("Tool: {} - {}",
            tool.name,
            tool.description.as_deref().unwrap_or(""));
    }

    // Call a reasoning tool
    let result = client.call_tool(
        "rk_reason",
        serde_json::json!({
            "query": "What are the trade-offs of microservices vs monoliths?",
            "profile": "balanced"
        })
    ).await?;

    println!("Result: {:?}", result);

    // Check health
    let healthy = client.ping().await?;
    println!("Server healthy: {}", healthy);

    // Get statistics
    let stats = client.stats().await;
    println!("Requests: {}, Avg response time: {:.2}ms",
        stats.requests_sent,
        stats.avg_response_time_ms);

    // Disconnect
    client.disconnect().await?;

    Ok(())
}
```

### Using the MCP Registry

For managing multiple MCP servers:

```rust
use reasonkit_core::mcp::{McpRegistry, McpServer, ServerInfo, ServerCapabilities};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create registry with 60-second health check interval
    let registry = McpRegistry::with_health_check_interval(60);

    // Start background health monitoring
    registry.start_health_monitoring().await;

    // List all tools from all registered servers
    let tools = registry.list_all_tools().await?;
    println!("Total tools available: {}", tools.len());

    // Find servers by tag
    let reasoning_servers = registry.find_servers_by_tag("thinktool").await;

    // Get registry statistics
    let stats = registry.statistics().await;
    println!("Servers: {} total, {} healthy",
        stats.total_servers,
        stats.healthy_servers);

    // Stop monitoring when done
    registry.stop_health_monitoring().await;

    Ok(())
}
```

---

## Claude Desktop Integration

### Configuration File

Add ReasonKit to your Claude Desktop MCP configuration:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "reasonkit": {
      "command": "rk-core",
      "args": ["mcp-server"],
      "env": {
        "REASONKIT_LOG": "info"
      }
    }
  }
}
```

### Using with Absolute Path

If `rk-core` is not in your PATH:

```json
{
  "mcpServers": {
    "reasonkit": {
      "command": "/home/user/.cargo/bin/rk-core",
      "args": ["mcp-server"]
    }
  }
}
```

### Verification

After restarting Claude Desktop, you should see ReasonKit tools available:

1. Open Claude Desktop
2. Click the MCP icon in the sidebar
3. Verify "reasonkit" server is listed as connected
4. Tools like `rk_reason`, `rk_retrieve`, `rk_verify` should appear

### Using in Conversations

Once connected, you can use ReasonKit tools in Claude:

```
Use the rk_reason tool to analyze the trade-offs between
using Redis vs PostgreSQL for session storage.
```

Claude will call the tool and return structured reasoning results.

---

## Tool Development

### Creating Custom Tools

Implement the `ToolHandler` trait:

```rust
use reasonkit_core::mcp::{Tool, ToolResult, ToolHandler};
use reasonkit_core::error::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use serde_json::Value;

struct MyCustomTool;

#[async_trait]
impl ToolHandler for MyCustomTool {
    async fn call(&self, arguments: HashMap<String, Value>) -> Result<ToolResult> {
        // Extract arguments
        let query = arguments
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        // Process and return result
        Ok(ToolResult::text(format!("Processed: {}", query)))
    }
}

// Define the tool schema
fn my_tool_definition() -> Tool {
    Tool::with_schema(
        "my_custom_tool",
        "Description of what this tool does",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The input query"
                }
            },
            "required": ["query"]
        })
    )
}
```

### Registering Custom Tools

```rust
use std::sync::Arc;

// Register with server
server.register_tool(
    my_tool_definition(),
    Arc::new(MyCustomTool)
).await;
```

---

## Example Usage

### Calling rk_reason via JSON-RPC

```bash
# Send a reasoning request
echo '{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "rk_reason",
    "arguments": {
      "query": "Should I use async Rust or threads for this I/O-bound task?",
      "profile": "balanced"
    }
  }
}' | rk-core mcp-server
```

### Calling rk_retrieve for Knowledge Base Search

```bash
echo '{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "rk_retrieve",
    "arguments": {
      "query": "error handling patterns",
      "limit": 5,
      "threshold": 0.7
    }
  }
}' | rk-core mcp-server
```

### Using Protocol Delta for Citation

```bash
# Anchor content
echo '{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "delta_anchor",
    "arguments": {
      "content": "The exact text to anchor as a citation",
      "url": "https://example.com/source"
    }
  }
}' | rk-core mcp-server
```

### Listing Available Tools

```bash
echo '{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/list"
}' | rk-core mcp-server
```

---

## Troubleshooting

### Common Issues

#### Server Not Starting

**Symptom**: No output when running `rk-core mcp-server`

**Solutions**:

1. Check if the binary exists: `which rk-core`
2. Run with verbose logging: `REASONKIT_LOG=debug rk-core mcp-server`
3. Verify permissions: `chmod +x $(which rk-core)`

#### Connection Timeout

**Symptom**: Client times out during connection

**Solutions**:

1. Increase timeout: `timeout_secs: 60`
2. Check if server is running: `ps aux | grep rk-core`
3. Verify stdio is not blocked by other processes

#### Invalid JSON-RPC Response

**Symptom**: Parse errors when communicating with server

**Solutions**:

1. Ensure single-line JSON (no newlines within the JSON object)
2. Check JSON syntax: `echo '{"jsonrpc":"2.0","id":1,"method":"ping"}' | jq .`
3. Verify server is sending newline-delimited JSON

#### Tool Not Found

**Symptom**: Error `Tool not found: <name>`

**Solutions**:

1. List available tools: `{"method":"tools/list"}`
2. Check tool name spelling (case-sensitive)
3. Verify tool is registered with server

#### Health Check Failures

**Symptom**: Server marked as unhealthy

**Solutions**:

1. Check server logs for errors
2. Verify network connectivity (for HTTP transport)
3. Increase health check timeout if needed
4. Check system resources (memory, CPU)

### Debug Mode

Run with full debugging:

```bash
RUST_BACKTRACE=1 REASONKIT_LOG=trace rk-core mcp-server 2>debug.log
```

### Testing Connection

Use the MCP CLI to test:

```bash
# List servers
rk-core mcp list-servers

# List tools
rk-core mcp list-tools

# Check server health
rk-core mcp ping
```

### Error Codes

| Code   | Name               | Description                  |
| ------ | ------------------ | ---------------------------- |
| -32700 | Parse Error        | Invalid JSON                 |
| -32600 | Invalid Request    | Not a valid JSON-RPC request |
| -32601 | Method Not Found   | Method does not exist        |
| -32602 | Invalid Params     | Invalid method parameters    |
| -32603 | Internal Error     | Server internal error        |
| -32800 | Request Cancelled  | Request was cancelled        |
| -32801 | Resource Not Found | Resource does not exist      |
| -32802 | Tool Not Found     | Tool does not exist          |
| -32803 | Invalid Tool Input | Tool input validation failed |

### Getting Help

1. Check logs: `REASONKIT_LOG=debug rk-core mcp-server 2>&1 | tee mcp.log`
2. Run tests: `cargo test --lib mcp`
3. File an issue: https://github.com/reasonkit/reasonkit-core/issues

---

## API Reference

### JSON-RPC Methods

#### initialize

Initialize the server connection.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": {},
    "clientInfo": {
      "name": "my-client",
      "version": "1.0.0"
    }
  }
}
```

#### shutdown

Gracefully shutdown the server.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "shutdown"
}
```

#### ping

Health check.

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "ping"
}
```

#### tools/list

List available tools.

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/list"
}
```

#### tools/call

Execute a tool.

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": { "key": "value" }
  }
}
```

### Server Info

```json
{
  "name": "reasonkit-core",
  "version": "0.1.0",
  "description": "ReasonKit Core MCP Server",
  "vendor": "ReasonKit Team"
}
```

### Protocol Version

Current MCP protocol version: **2025-11-25**

---

## References

- [MCP Specification](https://spec.modelcontextprotocol.io)
- [MCP GitHub](https://github.com/modelcontextprotocol)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
- [ReasonKit Documentation](https://reasonkit.sh/docs)
- [Internal MCP README](/home/zyxsys/RK-PROJECT/reasonkit-core/src/mcp/README.md)
- [MCP API Reference](/home/zyxsys/RK-PROJECT/reasonkit-core/MCP_API_REFERENCE.md)

---

**Document Version**: 1.0.0
**Last Updated**: 2026-01-01
**License**: Apache 2.0
