//! ThinkTool MCP Handlers
//!
//! MCP tool implementations for ReasonKit ThinkTools, exposing
//! structured reasoning capabilities to AI agents via MCP.
//!
//! ## Available Tools
//!
//! | Tool | Description | Profile |
//! |------|-------------|---------|
//! | `gigathink` | Expansive creative thinking (10+ perspectives) | `--creative` |
//! | `laserlogic` | Precision deductive reasoning with fallacy detection | `--balanced` |
//! | `bedrock` | First principles decomposition | `--deep` |
//! | `proofguard` | Multi-source verification (3+ sources) | `--paranoid` |
//! | `brutalhonesty` | Adversarial self-critique | `--paranoid` |
//!
//! ## Usage
//!
//! ### Quick Registration (Recommended)
//!
//! Use `register_thinktools()` to register all 5 tools at once:
//!
//! ```rust,ignore
//! use reasonkit::mcp::{McpServer, register_thinktools};
//!
//! let server = McpServer::new(...);
//! register_thinktools(&server).await;  // Registers all 5 ThinkTools
//! ```
//!
//! ### Individual Handler Registration
//!
//! Register specific tools using individual handlers:
//!
//! ```rust,ignore
//! use reasonkit::mcp::{ThinkToolHandler, GigaThinkHandler, McpServer};
//! use std::sync::Arc;
//!
//! let server = McpServer::new(...);
//!
//! // Register only GigaThink
//! server.register_tool(
//!     ThinkToolHandler::gigathink_tool(),
//!     Arc::new(GigaThinkHandler::new())
//! ).await;
//! ```
//!
//! ### Combined Handler (Legacy)
//!
//! For backward compatibility, use `ThinkToolHandler` with `_tool` routing:
//!
//! ```rust,ignore
//! use reasonkit::mcp::{ThinkToolHandler, ToolHandler};
//!
//! let handler = ThinkToolHandler::new();
//! let tools = handler.tool_definitions();
//!
//! // Register with MCP server
//! for tool in tools {
//!     server.register_tool(tool, Arc::new(handler.clone()));
//! }
//! ```

use crate::error::{Error, Result};
use crate::mcp::tools::{Tool, ToolHandler, ToolResult};
use crate::thinktool::modules::{
    BedRock, BrutalHonesty, GigaThink, LaserLogic, ProofGuard, ThinkToolContext, ThinkToolModule,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info, instrument};

/// ThinkTool MCP Handler
///
/// Provides MCP tool interface for all 5 core ThinkTools:
/// - GigaThink: Expansive creative thinking
/// - LaserLogic: Precision deductive reasoning
/// - BedRock: First principles decomposition
/// - ProofGuard: Multi-source verification
/// - BrutalHonesty: Adversarial self-critique
///
/// # Example
///
/// ```rust,ignore
/// let handler = ThinkToolHandler::new();
/// let args = HashMap::from([("query".to_string(), json!("Should we migrate to microservices?"))]);
/// let result = handler.call_tool("gigathink", args).await?;
/// ```
pub struct ThinkToolHandler {
    /// GigaThink module instance
    gigathink: Arc<GigaThink>,
    /// LaserLogic module instance
    laserlogic: Arc<LaserLogic>,
    /// BedRock module instance
    bedrock: Arc<BedRock>,
    /// ProofGuard module instance
    proofguard: Arc<ProofGuard>,
    /// BrutalHonesty module instance
    brutalhonesty: Arc<BrutalHonesty>,
}

impl Default for ThinkToolHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkToolHandler {
    /// Create a new ThinkToolHandler with default configurations
    pub fn new() -> Self {
        Self {
            gigathink: Arc::new(GigaThink::new()),
            laserlogic: Arc::new(LaserLogic::new()),
            bedrock: Arc::new(BedRock::new()),
            proofguard: Arc::new(ProofGuard::new()),
            brutalhonesty: Arc::new(BrutalHonesty::new()),
        }
    }

    /// Get all ThinkTool definitions for MCP registration
    ///
    /// Returns a vector of Tool definitions with JSON schemas for input validation.
    pub fn tool_definitions() -> Vec<Tool> {
        vec![
            Self::gigathink_tool(),
            Self::laserlogic_tool(),
            Self::bedrock_tool(),
            Self::proofguard_tool(),
            Self::brutalhonesty_tool(),
        ]
    }

    /// GigaThink tool definition
    fn gigathink_tool() -> Tool {
        Tool::with_schema(
            "gigathink",
            "Expansive creative thinking - generates 10+ diverse perspectives across \
            multiple analytical dimensions (economic, technological, social, environmental, \
            political, psychological, ethical, historical, competitive, user experience, \
            risk/opportunity, strategic). Use for brainstorming, exploring problem spaces, \
            or generating comprehensive viewpoints on complex issues.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question, problem, or topic to analyze from multiple perspectives",
                        "minLength": 10
                    },
                    "context": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional previous reasoning steps for chained execution",
                        "default": []
                    },
                    "min_perspectives": {
                        "type": "integer",
                        "description": "Minimum number of perspectives to generate",
                        "minimum": 5,
                        "maximum": 20,
                        "default": 10
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    /// LaserLogic tool definition
    fn laserlogic_tool() -> Tool {
        Tool::with_schema(
            "laserlogic",
            "Precision deductive reasoning with fallacy detection - validates logical \
            arguments, identifies formal fallacies (affirming consequent, denying antecedent, \
            undistributed middle, illicit major/minor), detects contradictions, and assesses \
            argument soundness. Use for analyzing claims, validating reasoning chains, or \
            checking logical consistency.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The argument or statement to analyze logically"
                    },
                    "context": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional previous reasoning steps",
                        "default": []
                    },
                    "detect_fallacies": {
                        "type": "boolean",
                        "description": "Enable formal fallacy detection",
                        "default": true
                    },
                    "check_contradictions": {
                        "type": "boolean",
                        "description": "Enable contradiction detection",
                        "default": true
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    /// BedRock tool definition
    fn bedrock_tool() -> Tool {
        Tool::with_schema(
            "bedrock",
            "First principles decomposition - breaks down complex problems into \
            fundamental axioms, surfaces hidden assumptions, and identifies core \
            building blocks. Use for understanding root causes, challenging assumptions, \
            or building understanding from foundational concepts.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The problem or concept to decompose to first principles"
                    },
                    "context": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional previous reasoning steps",
                        "default": []
                    },
                    "decomposition_depth": {
                        "type": "integer",
                        "description": "Maximum depth of decomposition (levels of 'why')",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    /// ProofGuard tool definition
    fn proofguard_tool() -> Tool {
        Tool::with_schema(
            "proofguard",
            "Multi-source verification - validates claims through triangulation \
            of evidence from multiple independent sources. Implements 3-source minimum \
            requirement, source credibility tiers (primary, secondary, tertiary), and \
            confidence scoring based on source agreement. Use for fact-checking, \
            verifying claims, or assessing evidence quality.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The claim or statement to verify"
                    },
                    "context": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional previous reasoning steps",
                        "default": []
                    },
                    "min_sources": {
                        "type": "integer",
                        "description": "Minimum number of independent sources required",
                        "minimum": 2,
                        "maximum": 10,
                        "default": 3
                    },
                    "verification_strategy": {
                        "type": "string",
                        "description": "Strategy for source verification",
                        "enum": ["triangulation", "consensus", "hierarchical"],
                        "default": "triangulation"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    /// BrutalHonesty tool definition
    fn brutalhonesty_tool() -> Tool {
        Tool::with_schema(
            "brutalhonesty",
            "Adversarial self-critique - identifies weaknesses, biases, blind spots, \
            and implicit assumptions in arguments or plans. Generates devil's advocate \
            counterarguments, detects cognitive biases, and provides ruthless assessment \
            of claims. Use for stress-testing ideas, challenging assumptions, or getting \
            honest feedback on proposals.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The argument, plan, or idea to critique"
                    },
                    "context": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional previous reasoning steps",
                        "default": []
                    },
                    "severity": {
                        "type": "string",
                        "description": "Critique intensity level",
                        "enum": ["gentle", "moderate", "harsh", "ruthless"],
                        "default": "moderate"
                    },
                    "enable_devil_advocate": {
                        "type": "boolean",
                        "description": "Enable devil's advocate mode",
                        "default": true
                    },
                    "detect_cognitive_biases": {
                        "type": "boolean",
                        "description": "Enable cognitive bias detection",
                        "default": true
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    /// Execute a ThinkTool by name
    ///
    /// Dispatches to the appropriate tool handler based on the tool name.
    #[instrument(skip(self, arguments), fields(tool = %name))]
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: HashMap<String, Value>,
    ) -> Result<ToolResult> {
        info!(tool = %name, "Executing ThinkTool");

        match name {
            "gigathink" => self.handle_gigathink(arguments).await,
            "laserlogic" => self.handle_laserlogic(arguments).await,
            "bedrock" => self.handle_bedrock(arguments).await,
            "proofguard" => self.handle_proofguard(arguments).await,
            "brutalhonesty" => self.handle_brutalhonesty(arguments).await,
            _ => {
                error!(tool = %name, "Unknown tool requested");
                Ok(ToolResult::error(format!("Unknown ThinkTool: {}", name)))
            }
        }
    }

    /// Handle GigaThink tool execution
    async fn handle_gigathink(&self, args: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&args, "query")?;
        let context = extract_context(&args);

        debug!(query = %query, context_len = context.len(), "Executing GigaThink");

        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.gigathink.execute(&think_context)?;

        format_output(output)
    }

    /// Handle LaserLogic tool execution
    async fn handle_laserlogic(&self, args: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&args, "query")?;
        let context = extract_context(&args);

        debug!(query = %query, "Executing LaserLogic");

        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.laserlogic.execute(&think_context)?;

        format_output(output)
    }

    /// Handle BedRock tool execution
    async fn handle_bedrock(&self, args: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&args, "query")?;
        let context = extract_context(&args);

        debug!(query = %query, "Executing BedRock");

        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.bedrock.execute(&think_context)?;

        format_output(output)
    }

    /// Handle ProofGuard tool execution
    async fn handle_proofguard(&self, args: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&args, "query")?;
        let context = extract_context(&args);

        debug!(query = %query, "Executing ProofGuard");

        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.proofguard.execute(&think_context)?;

        format_output(output)
    }

    /// Handle BrutalHonesty tool execution
    async fn handle_brutalhonesty(&self, args: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&args, "query")?;
        let context = extract_context(&args);

        debug!(query = %query, "Executing BrutalHonesty");

        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.brutalhonesty.execute(&think_context)?;

        format_output(output)
    }
}

/// Implement ToolHandler trait for integration with MCP server
#[async_trait]
impl ToolHandler for ThinkToolHandler {
    async fn call(&self, arguments: HashMap<String, Value>) -> Result<ToolResult> {
        // Extract tool name from arguments
        let tool_name = arguments
            .get("_tool")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| Error::Mcp("Missing _tool identifier in arguments".into()))?;

        self.call_tool(&tool_name, arguments).await
    }
}

// ============================================================================
// INDIVIDUAL TOOL HANDLERS (for per-tool MCP registration)
// ============================================================================

/// GigaThink MCP Handler
///
/// Provides expansive creative thinking with 10+ diverse perspectives.
/// Ideal for brainstorming, strategic planning, and exploring solution spaces.
///
/// # MCP Tool Parameters
/// - `query` (required): The topic or question to analyze
/// - `context` (optional): Array of previous reasoning steps for context
///
/// # Example
/// ```json
/// { "query": "Should we migrate to microservices?", "context": [] }
/// ```
pub struct GigaThinkHandler {
    module: Arc<GigaThink>,
}

impl Default for GigaThinkHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl GigaThinkHandler {
    pub fn new() -> Self {
        Self {
            module: Arc::new(GigaThink::new()),
        }
    }
}

#[async_trait]
impl ToolHandler for GigaThinkHandler {
    async fn call(&self, arguments: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&arguments, "query")?;
        let context = extract_context(&arguments);
        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.module.execute(&think_context)?;
        format_output(output)
    }
}

/// LaserLogic MCP Handler
///
/// Provides precision deductive reasoning with fallacy detection.
/// Ideal for argument validation, logical analysis, and formal reasoning.
///
/// # MCP Tool Parameters
/// - `query` (required): The logical argument to analyze (format: "Premise 1. Premise 2. Therefore, Conclusion.")
/// - `context` (optional): Array of previous reasoning steps for context
///
/// # Example
/// ```json
/// { "query": "All birds can fly. Penguins are birds. Therefore, penguins can fly.", "context": [] }
/// ```
pub struct LaserLogicHandler {
    module: Arc<LaserLogic>,
}

impl Default for LaserLogicHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl LaserLogicHandler {
    pub fn new() -> Self {
        Self {
            module: Arc::new(LaserLogic::new()),
        }
    }
}

#[async_trait]
impl ToolHandler for LaserLogicHandler {
    async fn call(&self, arguments: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&arguments, "query")?;
        let context = extract_context(&arguments);
        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.module.execute(&think_context)?;
        format_output(output)
    }
}

/// BedRock MCP Handler
///
/// Provides first principles decomposition for fundamental analysis.
/// Ideal for breaking down complex problems to their foundational truths.
///
/// # MCP Tool Parameters
/// - `query` (required): The problem or concept to decompose
/// - `context` (optional): Array of previous reasoning steps for context
///
/// # Example
/// ```json
/// { "query": "Why are electric vehicles becoming popular?", "context": [] }
/// ```
pub struct BedRockHandler {
    module: Arc<BedRock>,
}

impl Default for BedRockHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl BedRockHandler {
    pub fn new() -> Self {
        Self {
            module: Arc::new(BedRock::new()),
        }
    }
}

#[async_trait]
impl ToolHandler for BedRockHandler {
    async fn call(&self, arguments: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&arguments, "query")?;
        let context = extract_context(&arguments);
        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.module.execute(&think_context)?;
        format_output(output)
    }
}

/// ProofGuard MCP Handler
///
/// Provides multi-source verification (3+ sources minimum).
/// Ideal for fact-checking, claim validation, and research verification.
///
/// # MCP Tool Parameters
/// - `query` (required): The claim or assertion to verify
/// - `context` (optional): Array of previous reasoning steps for context
///
/// # Example
/// ```json
/// { "query": "Python is the most popular programming language in 2024", "context": [] }
/// ```
pub struct ProofGuardHandler {
    module: Arc<ProofGuard>,
}

impl Default for ProofGuardHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofGuardHandler {
    pub fn new() -> Self {
        Self {
            module: Arc::new(ProofGuard::new()),
        }
    }
}

#[async_trait]
impl ToolHandler for ProofGuardHandler {
    async fn call(&self, arguments: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&arguments, "query")?;
        let context = extract_context(&arguments);
        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.module.execute(&think_context)?;
        format_output(output)
    }
}

/// BrutalHonesty MCP Handler
///
/// Provides adversarial self-critique for identifying weaknesses.
/// Ideal for stress-testing ideas, uncovering blind spots, and ensuring rigor.
///
/// # MCP Tool Parameters
/// - `query` (required): The idea, plan, or argument to critique
/// - `context` (optional): Array of previous reasoning steps for context
///
/// # Example
/// ```json
/// { "query": "Our startup will succeed because we have a great product", "context": [] }
/// ```
pub struct BrutalHonestyHandler {
    module: Arc<BrutalHonesty>,
}

impl Default for BrutalHonestyHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl BrutalHonestyHandler {
    pub fn new() -> Self {
        Self {
            module: Arc::new(BrutalHonesty::new()),
        }
    }
}

#[async_trait]
impl ToolHandler for BrutalHonestyHandler {
    async fn call(&self, arguments: HashMap<String, Value>) -> Result<ToolResult> {
        let query = extract_required_string(&arguments, "query")?;
        let context = extract_context(&arguments);
        let think_context = ThinkToolContext::with_previous_steps(query, context);
        let output = self.module.execute(&think_context)?;
        format_output(output)
    }
}

/// Register all ThinkTools with an MCP server
///
/// This helper function registers all 5 ThinkTools with their handlers.
///
/// # Example
///
/// ```rust,ignore
/// use reasonkit::mcp::{McpServer, register_thinktools};
///
/// let server = McpServer::new(...);
/// register_thinktools(&server).await;
/// ```
pub async fn register_thinktools<T: crate::mcp::McpServerTrait + ?Sized>(server: &T) {
    // Register GigaThink
    server
        .register_tool(
            ThinkToolHandler::gigathink_tool(),
            Arc::new(GigaThinkHandler::new()),
        )
        .await;

    // Register LaserLogic
    server
        .register_tool(
            ThinkToolHandler::laserlogic_tool(),
            Arc::new(LaserLogicHandler::new()),
        )
        .await;

    // Register BedRock
    server
        .register_tool(
            ThinkToolHandler::bedrock_tool(),
            Arc::new(BedRockHandler::new()),
        )
        .await;

    // Register ProofGuard
    server
        .register_tool(
            ThinkToolHandler::proofguard_tool(),
            Arc::new(ProofGuardHandler::new()),
        )
        .await;

    // Register BrutalHonesty
    server
        .register_tool(
            ThinkToolHandler::brutalhonesty_tool(),
            Arc::new(BrutalHonestyHandler::new()),
        )
        .await;

    tracing::info!(
        "Registered 5 ThinkTools: gigathink, laserlogic, bedrock, proofguard, brutalhonesty"
    );
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Extract required string argument
fn extract_required_string(args: &HashMap<String, Value>, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Mcp(format!("Missing required argument: {}", key)))
}

/// Extract optional context array
fn extract_context(args: &HashMap<String, Value>) -> Vec<String> {
    args.get("context")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Format ThinkToolOutput as ToolResult
fn format_output(output: crate::thinktool::modules::ThinkToolOutput) -> Result<ToolResult> {
    let formatted = json!({
        "module": output.module,
        "confidence": output.confidence,
        "result": output.output
    });

    Ok(ToolResult::text(serde_json::to_string_pretty(&formatted)?))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_creation() {
        let handler = ThinkToolHandler::new();
        assert!(Arc::strong_count(&handler.gigathink) > 0);
    }

    #[test]
    fn test_tool_definitions_count() {
        let tools = ThinkToolHandler::tool_definitions();
        assert_eq!(tools.len(), 5, "Should have 5 ThinkTool definitions");
    }

    #[test]
    fn test_tool_definitions_names() {
        let tools = ThinkToolHandler::tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        assert!(names.contains(&"gigathink"));
        assert!(names.contains(&"laserlogic"));
        assert!(names.contains(&"bedrock"));
        assert!(names.contains(&"proofguard"));
        assert!(names.contains(&"brutalhonesty"));
    }

    #[test]
    fn test_tool_definitions_have_descriptions() {
        let tools = ThinkToolHandler::tool_definitions();
        for tool in tools {
            assert!(
                tool.description.is_some(),
                "Tool {} should have description",
                tool.name
            );
        }
    }

    #[test]
    fn test_tool_definitions_have_schemas() {
        let tools = ThinkToolHandler::tool_definitions();
        for tool in tools {
            assert!(
                tool.input_schema.is_object(),
                "Tool {} should have JSON schema",
                tool.name
            );

            // All tools should require "query" argument
            let required = tool.input_schema.get("required").unwrap();
            assert!(
                required.as_array().unwrap().contains(&json!("query")),
                "Tool {} should require 'query' argument",
                tool.name
            );
        }
    }

    #[test]
    fn test_extract_required_string_success() {
        let mut args = HashMap::new();
        args.insert("query".to_string(), json!("test query"));

        let result = extract_required_string(&args, "query");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test query");
    }

    #[test]
    fn test_extract_required_string_missing() {
        let args = HashMap::new();
        let result = extract_required_string(&args, "query");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_context_present() {
        let mut args = HashMap::new();
        args.insert("context".to_string(), json!(["step 1", "step 2"]));

        let context = extract_context(&args);
        assert_eq!(context.len(), 2);
        assert_eq!(context[0], "step 1");
    }

    #[test]
    fn test_extract_context_missing() {
        let args = HashMap::new();
        let context = extract_context(&args);
        assert!(context.is_empty());
    }

    #[tokio::test]
    async fn test_gigathink_execution() {
        let handler = ThinkToolHandler::new();
        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            json!("What are the implications of AI in healthcare?"),
        );

        let result = handler.handle_gigathink(args).await;
        assert!(result.is_ok());

        let tool_result = result.unwrap();
        assert!(tool_result.is_error.is_none() || !tool_result.is_error.unwrap());
    }

    #[tokio::test]
    async fn test_laserlogic_execution() {
        let handler = ThinkToolHandler::new();
        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            json!("All birds can fly. Penguins are birds. Therefore, penguins can fly."),
        );

        let result = handler.handle_laserlogic(args).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_bedrock_execution() {
        let handler = ThinkToolHandler::new();
        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            json!("Why do companies need to innovate?"),
        );

        let result = handler.handle_bedrock(args).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_proofguard_execution() {
        let handler = ThinkToolHandler::new();
        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            json!("Climate change is primarily caused by human activity"),
        );

        let result = handler.handle_proofguard(args).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_brutalhonesty_execution() {
        let handler = ThinkToolHandler::new();
        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            json!("Our startup will succeed because we have the best team"),
        );

        let result = handler.handle_brutalhonesty(args).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_unknown_tool() {
        let handler = ThinkToolHandler::new();
        let args = HashMap::new();

        let result = handler.call_tool("nonexistent", args).await;
        assert!(result.is_ok());

        let tool_result = result.unwrap();
        assert_eq!(tool_result.is_error, Some(true));
    }

    #[tokio::test]
    async fn test_missing_query_argument() {
        let handler = ThinkToolHandler::new();
        let args = HashMap::new(); // No query

        let result = handler.handle_gigathink(args).await;
        assert!(result.is_err());
    }
}
