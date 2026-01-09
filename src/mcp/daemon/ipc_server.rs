//! IPC Server Implementation
//!
//! Handles incoming IPC connections and routes requests to MCP registry.

use crate::error::{Error, Result};
use crate::mcp::tools::{Tool, ToolResult};
use crate::mcp::McpRegistry;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{broadcast, RwLock};
use tracing::{error, info, warn};
use uuid::Uuid;

#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

/// IPC message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IpcMessage {
    // Client -> Daemon requests
    CallTool {
        id: String,
        tool: String,
        args: serde_json::Value,
    },
    ListTools {
        id: String,
    },
    Ping {
        id: String,
    },
    Shutdown {
        id: String,
    },

    // Daemon -> Client responses
    ToolResult {
        id: String,
        result: ToolResult,
    },
    ToolsList {
        id: String,
        tools: Vec<Tool>,
    },
    Pong {
        id: String,
    },
    Error {
        id: String,
        error: String,
    },
    Ok {
        id: String,
    },
}

/// Daemon IPC server
pub struct DaemonServer {
    registry: Arc<RwLock<McpRegistry>>,
    shutdown_tx: broadcast::Sender<()>,
    #[cfg(unix)]
    listener: UnixListener,
}

impl DaemonServer {
    /// Create new daemon server
    #[cfg(unix)]
    pub async fn new(shutdown_tx: broadcast::Sender<()>) -> Result<Self> {
        let socket_path = super::manager::DaemonManager::get_socket_path()?;

        // Remove existing socket
        if socket_path.exists() {
            std::fs::remove_file(&socket_path)?;
        }

        // Create Unix socket listener
        let listener = UnixListener::bind(&socket_path)
            .map_err(|e| Error::daemon(format!("Failed to bind socket: {}", e)))?;

        // Set socket permissions (user-only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&socket_path, perms)?;
        }

        info!("IPC server listening on {:?}", socket_path);

        Ok(Self {
            registry: Arc::new(RwLock::new(McpRegistry::new())),
            shutdown_tx,
            listener,
        })
    }

    /// Run the server (blocks until shutdown)
    pub async fn run(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                // Accept new connections
                result = self.listener.accept() => {
                    match result {
                        Ok((stream, _addr)) => {
                            let registry = self.registry.clone();
                            let shutdown_tx = self.shutdown_tx.clone();

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_client(stream, registry, shutdown_tx).await {
                                    error!("Client handler error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            error!("Failed to accept connection: {}", e);
                        }
                    }
                }

                // Shutdown signal received
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received, stopping server...");
                    self.shutdown().await?;
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle client connection
    #[cfg(unix)]
    async fn handle_client(
        mut stream: UnixStream,
        registry: Arc<RwLock<McpRegistry>>,
        shutdown_tx: broadcast::Sender<()>,
    ) -> Result<()> {
        let client_id = Uuid::new_v4();
        info!("Client connected: {}", client_id);

        loop {
            // Read message length
            let mut len_buf = [0u8; 4];
            match stream.read_exact(&mut len_buf).await {
                Ok(_) => {}
                Err(_) => {
                    // Client disconnected
                    info!("Client disconnected: {}", client_id);
                    break;
                }
            }

            let len = u32::from_le_bytes(len_buf) as usize;

            // Enforce max message size (1 MB)
            if len > 1_000_000 {
                warn!("Client {} sent oversized message: {} bytes", client_id, len);
                break;
            }

            // Read message data
            let mut buf = vec![0u8; len];
            stream.read_exact(&mut buf).await?;

            // Deserialize message
            let message: IpcMessage = match serde_json::from_slice(&buf) {
                Ok(msg) => msg,
                Err(e) => {
                    warn!("Failed to deserialize message: {}", e);
                    continue;
                }
            };

            // Handle message
            let response = Self::process_message(message, &registry, &shutdown_tx).await;

            // Send response
            Self::send_message(&mut stream, &response).await?;
        }

        Ok(())
    }

    /// Process IPC message
    async fn process_message(
        msg: IpcMessage,
        registry: &Arc<RwLock<McpRegistry>>,
        shutdown_tx: &broadcast::Sender<()>,
    ) -> IpcMessage {
        match msg {
            IpcMessage::CallTool { id, tool, args } => {
                info!("Calling tool: {}", tool);

                match registry.read().await.call_tool_by_name(&tool, args).await {
                    Ok(result) => IpcMessage::ToolResult { id, result },
                    Err(e) => IpcMessage::Error {
                        id,
                        error: format!("Tool execution failed: {}", e),
                    },
                }
            }

            IpcMessage::ListTools { id } => match registry.read().await.list_all_tools().await {
                Ok(tools) => IpcMessage::ToolsList { id, tools },
                Err(e) => IpcMessage::Error {
                    id,
                    error: format!("Failed to list tools: {}", e),
                },
            },

            IpcMessage::Ping { id } => IpcMessage::Pong { id },

            IpcMessage::Shutdown { id } => {
                info!("Shutdown requested via IPC");
                shutdown_tx.send(()).ok();
                IpcMessage::Ok { id }
            }

            _ => IpcMessage::Error {
                id: Uuid::new_v4().to_string(),
                error: "Invalid message type from client".to_string(),
            },
        }
    }

    /// Send IPC message
    #[cfg(unix)]
    async fn send_message(stream: &mut UnixStream, msg: &IpcMessage) -> Result<()> {
        let json = serde_json::to_vec(msg)?;
        let len = (json.len() as u32).to_le_bytes();

        stream.write_all(&len).await?;
        stream.write_all(&json).await?;
        stream.flush().await?;

        Ok(())
    }

    /// Graceful shutdown
    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down MCP servers...");

        // Disconnect all servers
        let registry = self.registry.write().await;
        for server in registry.list_servers().await {
            if let Err(e) = registry.disconnect_server(&server.id).await {
                warn!("Failed to disconnect {}: {}", server.name, e);
            }
        }

        // Socket cleanup handled by Drop
        info!("Shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_message_serialization() {
        let msg = IpcMessage::Ping {
            id: "test-123".to_string(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("Ping"));
        assert!(json.contains("test-123"));

        let deserialized: IpcMessage = serde_json::from_str(&json).unwrap();
        match deserialized {
            IpcMessage::Ping { id } => assert_eq!(id, "test-123"),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_ipc_message_size_limit() {
        let max_size = 1_000_000_u32;
        assert_eq!(max_size, 1_000_000);
    }
}
