//! Inter-Process Communication module for ARF Core

use crate::error::Result;

/// IPC message types
#[derive(Debug, Clone)]
pub enum IpcMessage {
    Request(String),
    Response(String),
    Event(String),
}

/// IPC handler
pub struct IpcHandler;

impl IpcHandler {
    /// Create a new IPC handler
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// Send a message
    pub async fn send(&self, _msg: IpcMessage) -> Result<()> {
        Ok(())
    }
}

impl Default for IpcHandler {
    fn default() -> Self {
        Self
    }
}
