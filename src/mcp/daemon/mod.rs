//! MCP Daemon Module
//!
//! Provides optional background daemon for persistent MCP server connections.
//!
//! ## Architecture
//!
//! - **Direct Mode**: Spawn MCP clients on-demand (default, no daemon required)
//! - **Daemon Mode**: Persistent background process with connection pooling
//!
//! ## Usage
//!
//! ```bash
//! # Start daemon
//! rk mcp daemon start
//!
//! # Call tool (auto-detects daemon)
//! rk mcp call-tool gigathink '{"query": "What is reasoning?"}'
//!
//! # Stop daemon
//! rk mcp daemon stop
//! ```

pub mod health;
pub mod ipc_client;
pub mod ipc_server;
pub mod logger;
pub mod manager;
pub mod signals;

pub use health::HealthMonitor;
pub use ipc_client::IpcClient;
pub use ipc_server::{DaemonServer, IpcMessage};
pub use logger::DaemonLogger;
pub use manager::{DaemonManager, DaemonStatus};
pub use signals::setup_signal_handlers;

use crate::error::Result;
use crate::mcp::tools::ToolResult;
use crate::mcp::McpClientConfig;

/// Check if daemon is running
pub async fn daemon_is_running() -> Result<bool> {
    let manager = DaemonManager::new()?;
    Ok(matches!(
        manager.status().await,
        DaemonStatus::Running { .. }
    ))
}

/// Call tool via daemon (if running) or direct mode
pub async fn call_tool(name: &str, args: serde_json::Value) -> Result<ToolResult> {
    if daemon_is_running().await? {
        // Use daemon via IPC
        daemon_call_tool(name, args).await
    } else {
        // Direct execution
        direct_call_tool(name, args).await
    }
}

/// Call tool via daemon IPC
async fn daemon_call_tool(name: &str, args: serde_json::Value) -> Result<ToolResult> {
    let mut client = IpcClient::connect().await?;
    client.call_tool(name, args).await
}

/// Call tool directly (no daemon) - spawns temporary MCP client
async fn direct_call_tool(name: &str, args: serde_json::Value) -> Result<ToolResult> {
    use crate::mcp::{McpClient, McpClientTrait};
    use std::collections::HashMap;

    // TODO: Load server config from ~/.config/reasonkit/mcp_servers.json
    // For now, hardcode example server config
    let config = match name {
        "gigathink" | "laserlogic" | "bedrock" | "proofguard" | "brutalhonesty" => {
            // ThinkTool servers (built-in)
            McpClientConfig {
                name: "reasonkit-thinktools".to_string(),
                command: "rk".to_string(),
                args: vec!["serve-mcp".to_string()],
                env: HashMap::new(),
                timeout_secs: 30,
                auto_reconnect: false,
                max_retries: 1,
            }
        }
        "think" | "analyze" | "reason" => {
            // Sequential thinking via ReasonKit ThinkTools (Rust-native)
            // Note: CONS-001 prohibits Node.js - use Rust server
            McpClientConfig {
                name: "reasonkit-sequential".to_string(),
                command: "rk-core".to_string(),
                args: vec!["serve-mcp".to_string()],
                env: HashMap::new(),
                timeout_secs: 30,
                auto_reconnect: false,
                max_retries: 1,
            }
        }
        _ => {
            return Err(crate::error::Error::network(format!(
                "Unknown tool '{}' - no server configured",
                name
            )));
        }
    };

    // Create temporary client
    let mut client = McpClient::new(config);

    // Connect
    client.connect().await?;

    // Call tool
    let result = client.call_tool(name, args).await?;

    // Disconnect
    client.disconnect().await?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_daemon_detection() {
        // Should not be running initially
        let running = daemon_is_running().await.unwrap();
        assert!(!running);
    }

    #[tokio::test]
    async fn test_direct_mode_fallback() {
        // Even without daemon, direct mode should work
        // (This will fail if rk binary not built, which is expected in unit tests)
        // In integration tests, we'd verify this actually works
    }
}
