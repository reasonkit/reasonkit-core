//! IPC Client Implementation
//!
//! Client for connecting to the MCP daemon via IPC.

use crate::error::{Error, Result};
use crate::mcp::tools::{Tool, ToolResult};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use uuid::Uuid;

#[cfg(unix)]
use tokio::net::UnixStream;

use super::ipc_server::IpcMessage;

/// IPC client for daemon communication
pub struct IpcClient {
    #[cfg(unix)]
    stream: UnixStream,
}

impl IpcClient {
    /// Connect to daemon via IPC
    #[cfg(unix)]
    pub async fn connect() -> Result<Self> {
        let socket_path = super::manager::DaemonManager::get_socket_path()?;

        let stream = UnixStream::connect(&socket_path)
            .await
            .map_err(|e| Error::network(format!("Failed to connect to daemon: {}", e)))?;

        Ok(Self { stream })
    }

    /// Call a tool via daemon
    pub async fn call_tool(&mut self, name: &str, args: serde_json::Value) -> Result<ToolResult> {
        let msg = IpcMessage::CallTool {
            id: Uuid::new_v4().to_string(),
            tool: name.to_string(),
            args,
        };

        self.send_message(&msg).await?;
        let response = self.receive_message().await?;

        match response {
            IpcMessage::ToolResult { result, .. } => Ok(result),
            IpcMessage::Error { error, .. } => Err(Error::network(error)),
            _ => Err(Error::network("Unexpected response type")),
        }
    }

    /// List available tools
    pub async fn list_tools(&mut self) -> Result<Vec<Tool>> {
        let msg = IpcMessage::ListTools {
            id: Uuid::new_v4().to_string(),
        };

        self.send_message(&msg).await?;
        let response = self.receive_message().await?;

        match response {
            IpcMessage::ToolsList { tools, .. } => Ok(tools),
            IpcMessage::Error { error, .. } => Err(Error::network(error)),
            _ => Err(Error::network("Unexpected response type")),
        }
    }

    /// Ping daemon
    pub async fn ping(&mut self) -> Result<bool> {
        let msg = IpcMessage::Ping {
            id: Uuid::new_v4().to_string(),
        };

        self.send_message(&msg).await?;

        match tokio::time::timeout(std::time::Duration::from_secs(5), self.receive_message()).await
        {
            Ok(Ok(IpcMessage::Pong { .. })) => Ok(true),
            _ => Ok(false),
        }
    }

    /// Send shutdown signal to daemon
    pub async fn shutdown(&mut self) -> Result<()> {
        let msg = IpcMessage::Shutdown {
            id: Uuid::new_v4().to_string(),
        };

        self.send_message(&msg).await?;
        let response = self.receive_message().await?;

        match response {
            IpcMessage::Ok { .. } => Ok(()),
            IpcMessage::Error { error, .. } => Err(Error::network(error)),
            _ => Err(Error::network("Unexpected response type")),
        }
    }

    /// Send IPC message
    #[cfg(unix)]
    async fn send_message(&mut self, msg: &IpcMessage) -> Result<()> {
        let json = serde_json::to_vec(msg)?;
        let len = (json.len() as u32).to_le_bytes();

        self.stream.write_all(&len).await?;
        self.stream.write_all(&json).await?;
        self.stream.flush().await?;

        Ok(())
    }

    /// Receive IPC message
    #[cfg(unix)]
    async fn receive_message(&mut self) -> Result<IpcMessage> {
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf).await?;

        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf).await?;

        Ok(serde_json::from_slice(&buf)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ipc_client_connect_fails_when_no_daemon() {
        // Should fail when daemon not running
        let result = IpcClient::connect().await;
        assert!(result.is_err());
    }
}
