//! MCP Transport Layer
//!
//! Transport implementations for MCP communication (stdio, HTTP/SSE).

use super::types::{McpNotification, McpRequest, McpResponse};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

/// Transport type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportType {
    /// Standard input/output (process spawning)
    Stdio,
    /// HTTP with Server-Sent Events
    HttpSse,
    /// WebSocket
    WebSocket,
}

/// Transport trait for MCP communication
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a request and wait for response
    async fn send_request(&self, request: McpRequest) -> std::io::Result<McpResponse>;

    /// Send a notification (no response expected)
    async fn send_notification(&self, notification: McpNotification) -> std::io::Result<()>;

    /// Close the transport
    async fn close(&self) -> std::io::Result<()>;
}

/// Stdio transport (spawns process and communicates via stdin/stdout)
pub struct StdioTransport {
    /// Child process
    child: Mutex<Option<Child>>,
    /// Standard input writer
    stdin: Mutex<Option<ChildStdin>>,
    /// Standard output reader
    stdout: Mutex<Option<BufReader<ChildStdout>>>,
}

impl StdioTransport {
    /// Create a new stdio transport by spawning a command
    pub async fn spawn(
        command: impl AsRef<str>,
        args: Vec<String>,
        env: Vec<(String, String)>,
    ) -> std::io::Result<Self> {
        let mut cmd = Command::new(command.as_ref());
        cmd.args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd.spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("Failed to get stdin"))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("Failed to get stdout"))?;

        let stdout = BufReader::new(stdout);

        Ok(Self {
            child: Mutex::new(Some(child)),
            stdin: Mutex::new(Some(stdin)),
            stdout: Mutex::new(Some(stdout)),
        })
    }

    /// Read a JSON-RPC message from stdout
    async fn read_message(&self) -> std::io::Result<String> {
        let mut stdout_guard = self.stdout.lock().await;

        if let Some(stdout) = stdout_guard.as_mut() {
            let mut line = String::new();
            stdout.read_line(&mut line).await?;
            Ok(line)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "Stdout not available",
            ))
        }
    }

    /// Write a JSON-RPC message to stdin
    async fn write_message(&self, message: &str) -> std::io::Result<()> {
        let mut stdin_guard = self.stdin.lock().await;

        if let Some(stdin) = stdin_guard.as_mut() {
            stdin.write_all(message.as_bytes()).await?;
            stdin.write_all(b"\n").await?;
            stdin.flush().await?;
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "Stdin not available",
            ))
        }
    }
}

#[async_trait]
impl Transport for StdioTransport {
    async fn send_request(&self, request: McpRequest) -> std::io::Result<McpResponse> {
        // Serialize request
        let request_json = serde_json::to_string(&request)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Write request
        self.write_message(&request_json).await?;

        // Read response
        let response_line = self.read_message().await?;

        // Parse response
        let response: McpResponse = serde_json::from_str(&response_line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(response)
    }

    async fn send_notification(&self, notification: McpNotification) -> std::io::Result<()> {
        // Serialize notification
        let notification_json = serde_json::to_string(&notification)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Write notification
        self.write_message(&notification_json).await
    }

    async fn close(&self) -> std::io::Result<()> {
        // Close stdin
        let mut stdin_guard = self.stdin.lock().await;
        stdin_guard.take();
        drop(stdin_guard);

        // Kill child process
        let mut child_guard = self.child.lock().await;
        if let Some(mut child) = child_guard.take() {
            child.kill().await?;
            child.wait().await?;
        }

        Ok(())
    }
}

/// HTTP/SSE transport (not yet implemented)
pub struct HttpSseTransport {
    _base_url: String,
}

impl HttpSseTransport {
    /// Create a new HTTP/SSE transport
    #[allow(dead_code)]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            _base_url: base_url.into(),
        }
    }
}

#[async_trait]
impl Transport for HttpSseTransport {
    async fn send_request(&self, _request: McpRequest) -> std::io::Result<McpResponse> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "HTTP/SSE transport not yet implemented",
        ))
    }

    async fn send_notification(&self, _notification: McpNotification) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "HTTP/SSE transport not yet implemented",
        ))
    }

    async fn close(&self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_type() {
        let transport = TransportType::Stdio;
        let json = serde_json::to_string(&transport).unwrap();
        assert_eq!(json, "\"stdio\"");
    }
}
