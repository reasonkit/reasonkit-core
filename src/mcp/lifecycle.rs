//! MCP Lifecycle Management
//!
//! Initialize, ping, and shutdown protocol messages.

use super::types::{Implementation, ServerCapabilities, ServerInfo};
use serde::{Deserialize, Serialize};

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client name
    pub name: String,
    /// Client version
    pub version: String,
}

/// Initialize request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    /// Protocol version
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,

    /// Client capabilities
    pub capabilities: ClientCapabilities,

    /// Client information
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo,
}

/// Client capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Supports sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingCapability>,

    /// Supports roots
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,

    /// Experimental capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<serde_json::Value>,
}

/// Sampling capability (for LLM sampling)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingCapability {}

/// Roots capability (for workspace roots)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootsCapability {
    /// Whether list_changed notifications are supported
    #[serde(default, rename = "listChanged")]
    pub list_changed: bool,
}

/// Initialize result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    /// Protocol version
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,

    /// Server capabilities
    pub capabilities: ServerCapabilities,

    /// Server information
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,

    /// Server implementation details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub implementation: Option<Implementation>,
}

/// Shutdown request (no parameters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownRequest {}

/// Ping request (for health checks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingRequest {}

/// Ping response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingResponse {}

impl InitializeParams {
    /// Create default initialize parameters for ReasonKit
    pub fn reasonkit() -> Self {
        Self {
            protocol_version: crate::mcp::MCP_VERSION.to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "reasonkit-core".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }
    }

    /// Create initialize parameters with custom client info
    pub fn with_client_info(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            protocol_version: crate::mcp::MCP_VERSION.to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: name.into(),
                version: version.into(),
            },
        }
    }
}

impl InitializeResult {
    /// Create an initialize result
    pub fn new(server_info: ServerInfo, capabilities: ServerCapabilities) -> Self {
        Self {
            protocol_version: crate::mcp::MCP_VERSION.to_string(),
            capabilities,
            server_info,
            implementation: Some(Implementation {
                name: "reasonkit-core".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_params() {
        let params = InitializeParams::reasonkit();
        assert_eq!(params.client_info.name, "reasonkit-core");
        assert_eq!(params.protocol_version, crate::mcp::MCP_VERSION);
    }

    #[test]
    fn test_initialize_serialization() {
        let params = InitializeParams::reasonkit();
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("protocolVersion"));
        assert!(json.contains("clientInfo"));
    }
}
