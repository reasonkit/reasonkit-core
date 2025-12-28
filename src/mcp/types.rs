//! MCP Protocol Types
//!
//! JSON-RPC 2.0 and MCP-specific type definitions based on the
//! Model Context Protocol specification (2025-11-25).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 version
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonRpcVersion(String);

impl Default for JsonRpcVersion {
    fn default() -> Self {
        Self("2.0".to_string())
    }
}

/// Request identifier (string or number)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// String identifier
    String(String),
    /// Number identifier
    Number(i64),
}

/// MCP JSON-RPC message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpMessage {
    /// Request message
    Request(McpRequest),
    /// Response message
    Response(McpResponse),
    /// Notification message (no response expected)
    Notification(McpNotification),
}

/// JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: JsonRpcVersion,
    /// Request identifier
    pub id: RequestId,
    /// Method name
    pub method: String,
    /// Method parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    /// JSON-RPC version
    pub jsonrpc: JsonRpcVersion,
    /// Request identifier
    pub id: RequestId,
    /// Success result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

/// JSON-RPC notification (no response)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpNotification {
    /// JSON-RPC version
    pub jsonrpc: JsonRpcVersion,
    /// Method name
    pub method: String,
    /// Method parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// JSON-RPC error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    /// Error code
    pub code: ErrorCode,
    /// Error message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Standard JSON-RPC error codes plus MCP extensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ErrorCode(pub i32);

impl ErrorCode {
    /// Parse error
    pub const PARSE_ERROR: Self = Self(-32700);
    /// Invalid request
    pub const INVALID_REQUEST: Self = Self(-32600);
    /// Method not found
    pub const METHOD_NOT_FOUND: Self = Self(-32601);
    /// Invalid params
    pub const INVALID_PARAMS: Self = Self(-32602);
    /// Internal error
    pub const INTERNAL_ERROR: Self = Self(-32603);

    // MCP-specific errors
    /// Request cancelled
    pub const REQUEST_CANCELLED: Self = Self(-32800);
    /// Resource not found
    pub const RESOURCE_NOT_FOUND: Self = Self(-32801);
    /// Tool not found
    pub const TOOL_NOT_FOUND: Self = Self(-32802);
    /// Invalid tool input
    pub const INVALID_TOOL_INPUT: Self = Self(-32803);
}

/// Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
    /// Server description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Vendor/author information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
}

/// Server capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Supports tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Supports resources
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    /// Supports prompts
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    /// Supports logging
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
}

/// Tools capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    /// Whether list_changed notifications are supported
    #[serde(default)]
    pub list_changed: bool,
}

/// Resources capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcesCapability {
    /// Whether list_changed notifications are supported
    #[serde(default)]
    pub list_changed: bool,
    /// Whether subscribe is supported
    #[serde(default)]
    pub subscribe: bool,
}

/// Prompts capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsCapability {
    /// Whether list_changed notifications are supported
    #[serde(default)]
    pub list_changed: bool,
}

/// Logging capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingCapability {}

/// Implementation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Implementation {
    /// Implementation name
    pub name: String,
    /// Implementation version
    pub version: String,
}

/// Progress token for tracking long-running operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProgressToken {
    /// String token
    String(String),
    /// Number token
    Number(i64),
}

/// Progress notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressNotification {
    /// Progress token
    #[serde(rename = "progressToken")]
    pub progress_token: ProgressToken,
    /// Current progress (0.0 to 1.0)
    pub progress: f64,
    /// Total items (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

/// Cancellation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    /// Request ID to cancel
    #[serde(rename = "requestId")]
    pub request_id: RequestId,
    /// Optional reason for cancellation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl McpRequest {
    /// Create a new request
    pub fn new(id: RequestId, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: JsonRpcVersion::default(),
            id,
            method: method.into(),
            params,
        }
    }
}

impl McpResponse {
    /// Create a success response
    pub fn success(id: RequestId, result: Value) -> Self {
        Self {
            jsonrpc: JsonRpcVersion::default(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: RequestId, error: McpError) -> Self {
        Self {
            jsonrpc: JsonRpcVersion::default(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

impl McpError {
    /// Create a new error
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Create an error with additional data
    pub fn with_data(code: ErrorCode, message: impl Into<String>, data: Value) -> Self {
        Self {
            code,
            message: message.into(),
            data: Some(data),
        }
    }

    /// Parse error
    pub fn parse_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::PARSE_ERROR, message)
    }

    /// Invalid request
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::INVALID_REQUEST, message)
    }

    /// Method not found
    pub fn method_not_found(method: impl Into<String>) -> Self {
        Self::new(
            ErrorCode::METHOD_NOT_FOUND,
            format!("Method not found: {}", method.into()),
        )
    }

    /// Internal error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::INTERNAL_ERROR, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let req = McpRequest::new(RequestId::String("test-1".to_string()), "tools/list", None);

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"tools/list\""));
    }

    #[test]
    fn test_error_creation() {
        let err = McpError::method_not_found("test/method");
        assert_eq!(err.code, ErrorCode::METHOD_NOT_FOUND);
        assert!(err.message.contains("test/method"));
    }

    #[test]
    fn test_response_success() {
        let resp = McpResponse::success(RequestId::Number(1), serde_json::json!({"result": "ok"}));

        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_response_error() {
        let resp = McpResponse::error(RequestId::Number(1), McpError::internal_error("test error"));

        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
    }
}
