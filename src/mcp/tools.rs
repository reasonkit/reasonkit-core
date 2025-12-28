//! MCP Tools
//!
//! Tool definitions, inputs, and results for MCP servers.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool name (must be unique per server)
    pub name: String,

    /// Human-readable description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for input validation
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,

    /// Server ID (populated by registry)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_id: Option<Uuid>,

    /// Server name (populated by registry)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_name: Option<String>,
}

/// Tool input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInput {
    /// Tool name
    pub name: String,

    /// Tool arguments (must match inputSchema)
    #[serde(default)]
    pub arguments: HashMap<String, Value>,
}

/// Tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Result content
    pub content: Vec<ToolResultContent>,

    /// Whether the tool execution is finished
    #[serde(skip_serializing_if = "Option::is_none", rename = "isError")]
    pub is_error: Option<bool>,
}

/// Tool result content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    /// Text content
    Text {
        /// Text data
        text: String,
    },
    /// Image content
    Image {
        /// Image data (base64 encoded)
        data: String,
        /// MIME type
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Resource reference
    Resource {
        /// Resource URI
        uri: String,
        /// MIME type
        #[serde(skip_serializing_if = "Option::is_none", rename = "mimeType")]
        mime_type: Option<String>,
    },
}

/// Tool capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapability {
    /// Whether tool supports streaming results
    #[serde(default)]
    pub streaming: bool,

    /// Whether tool supports cancellation
    #[serde(default)]
    pub cancellable: bool,

    /// Estimated execution time in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_duration_ms: Option<u64>,
}

/// Resource template for dynamic resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTemplate {
    /// Template URI pattern (e.g., "file:///{path}")
    #[serde(rename = "uriTemplate")]
    pub uri_template: String,

    /// Resource name
    pub name: String,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// MIME type
    #[serde(skip_serializing_if = "Option::is_none", rename = "mimeType")]
    pub mime_type: Option<String>,
}

/// Prompt definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    /// Prompt name
    pub name: String,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Prompt arguments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<PromptArgument>>,
}

/// Prompt argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    /// Argument name
    pub name: String,

    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Whether the argument is required
    #[serde(default)]
    pub required: bool,
}

impl Tool {
    /// Create a simple tool with basic string input
    pub fn simple(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            server_id: None,
            server_name: None,
        }
    }

    /// Create a tool with custom input schema
    pub fn with_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            input_schema: schema,
            server_id: None,
            server_name: None,
        }
    }
}

impl ToolResult {
    /// Create a text result
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text { text: text.into() }],
            is_error: None,
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent::Text {
                text: message.into(),
            }],
            is_error: Some(true),
        }
    }

    /// Create a multi-content result
    pub fn with_content(content: Vec<ToolResultContent>) -> Self {
        Self {
            content,
            is_error: None,
        }
    }
}

impl ToolResultContent {
    /// Create text content
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create image content
    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Image {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }

    /// Create resource reference
    pub fn resource(uri: impl Into<String>) -> Self {
        Self::Resource {
            uri: uri.into(),
            mime_type: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tool() {
        let tool = Tool::simple("test_tool", "A test tool");
        assert_eq!(tool.name, "test_tool");
        assert!(tool.description.is_some());
    }

    #[test]
    fn test_tool_result() {
        let result = ToolResult::text("Success");
        assert_eq!(result.content.len(), 1);
        assert!(result.is_error.is_none());
    }

    #[test]
    fn test_error_result() {
        let result = ToolResult::error("Failed");
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn test_tool_serialization() {
        let tool = Tool::simple("test", "description");
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"name\":\"test\""));
        assert!(json.contains("inputSchema"));
    }
}
