//! # MCP (Model Context Protocol) Server Registry
//!
//! Rust-based MCP server implementation for ReasonKit.
//!
//! ## Overview
//!
//! This module provides:
//! - **Server Registry**: Dynamic server discovery and registration
//! - **MCP Client**: Connect to and interact with external MCP servers
//! - **Health Monitoring**: Automatic health checks for registered servers
//! - **Tool Management**: Tool capability reporting and execution
//! - **Protocol Compliance**: Full MCP specification support
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ MCP Registry (Coordinator)                                  │
//! │   - Server discovery                                        │
//! │   - Health monitoring                                       │
//! │   - Capability aggregation                                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │ MCP Client (Consumer)                                       │
//! │   - Connect to external MCP servers                         │
//! │   - Execute tools via RPC                                   │
//! │   - Access resources                                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │ MCP Servers (Multiple instances)                            │
//! │   - ThinkTool servers (GigaThink, LaserLogic, etc.)         │
//! │   - Custom tool servers                                     │
//! │   - Resource providers                                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Transport Layer                                             │
//! │   - JSON-RPC 2.0 over stdio (primary)                       │
//! │   - HTTP/SSE (optional)                                     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## MCP Protocol
//!
//! Based on MCP specification (2025-11-25):
//! - JSON-RPC 2.0 messaging
//! - Lifecycle management (initialize, shutdown)
//! - Tools, Resources, and Prompts primitives
//! - Progress notifications
//! - Cancellation support
//!
//! ## Example: MCP Client
//!
//! ```rust,ignore
//! use reasonkit::mcp::{McpClient, McpClientConfig, McpClientTrait};
//! use std::collections::HashMap;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create client configuration
//!     let config = McpClientConfig {
//!         name: "sequential-thinking".to_string(),
//!         command: "npx".to_string(),
//!         args: vec![
//!             "-y".to_string(),
//!             "@modelcontextprotocol/server-sequential-thinking".to_string()
//!         ],
//!         env: HashMap::new(),
//!         timeout_secs: 30,
//!         auto_reconnect: true,
//!         max_retries: 3,
//!     };
//!
//!     // Connect to the server
//!     let mut client = McpClient::new(config);
//!     client.connect().await?;
//!
//!     // List available tools
//!     let tools = client.list_tools().await?;
//!     println!("Available tools: {:?}", tools);
//!
//!     // Call a tool
//!     let result = client.call_tool(
//!         "think",
//!         serde_json::json!({
//!             "query": "What is chain-of-thought reasoning?"
//!         })
//!     ).await?;
//!
//!     println!("Result: {:?}", result);
//!
//!     // Disconnect
//!     client.disconnect().await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Example: Server Registry
//!
//! ```rust,ignore
//! use reasonkit::mcp::{McpRegistry, McpServerConfig, TransportType};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create registry
//!     let mut registry = McpRegistry::new();
//!
//!     // Register a ThinkTool server
//!     let config = McpServerConfig {
//!         name: "gigathink".to_string(),
//!         command: "rk-thinktool".to_string(),
//!         args: vec!["--module".to_string(), "gigathink".to_string()],
//!         transport: TransportType::Stdio,
//!         env: Default::default(),
//!     };
//!
//!     registry.register_server(config).await?;
//!
//!     // Discover all available tools
//!     let tools = registry.list_all_tools().await?;
//!     for tool in tools {
//!         println!("Tool: {} from server {}", tool.name, tool.server);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod delta_tools;
pub mod lifecycle;
pub mod registry;
#[cfg(feature = "memory")]
pub mod rerank_tools;
pub mod server;
pub mod tools;
pub mod transport;
pub mod types;

// Re-exports
pub use types::{
    ErrorCode, Implementation, JsonRpcVersion, McpError, McpMessage, McpNotification, McpRequest,
    McpResponse, RequestId, ServerCapabilities, ServerInfo,
};

pub use server::{McpServer, McpServerTrait, ServerMetrics, ServerStatus};

pub use registry::{HealthCheck, HealthStatus, McpRegistry, ServerRegistration};

pub use client::{ClientStats, ConnectionState, McpClient, McpClientConfig, McpClientTrait};

pub use tools::{
    Prompt, PromptArgument, ResourceTemplate, Tool, ToolCapability, ToolInput, ToolResult,
};

pub use transport::{StdioTransport, Transport, TransportType};

pub use lifecycle::{ClientInfo, InitializeParams, InitializeResult, PingRequest, ShutdownRequest};

pub use delta_tools::DeltaToolHandler;

#[cfg(feature = "memory")]
pub use rerank_tools::RerankToolHandler;

/// MCP protocol version (2025-11-25)
pub const MCP_VERSION: &str = "2025-11-25";

/// Default health check interval (30 seconds)
pub const DEFAULT_HEALTH_CHECK_INTERVAL_SECS: u64 = 30;

/// Maximum server startup timeout (10 seconds)
pub const MAX_SERVER_STARTUP_TIMEOUT_SECS: u64 = 10;
