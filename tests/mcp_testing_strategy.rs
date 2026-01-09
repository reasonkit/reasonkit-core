//! # Comprehensive Testing Strategy for MCP Tool Registration
//!
//! This module provides a complete test suite for MCP (Model Context Protocol)
//! tool registration, covering the 5 ThinkTools:
//!
//! - **GigaThink**: Expansive creative thinking
//! - **LaserLogic**: Precision deductive reasoning
//! - **BedRock**: First principles decomposition
//! - **ProofGuard**: Multi-source verification
//! - **BrutalHonesty**: Adversarial self-critique
//!
//! ## Test Categories
//!
//! 1. **Unit Tests**: Tool handler isolation with mocks
//! 2. **Integration Tests**: Full tool execution flow
//! 3. **Mock LLM Tests**: CI-compatible LLM simulation
//! 4. **Property-Based Tests**: JSON schema validation
//! 5. **Contract Tests**: MCP protocol compliance
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all MCP tests
//! cargo test --test mcp_testing_strategy --release
//!
//! # Run with feature flags
//! cargo test --test mcp_testing_strategy --features "cli" --release
//!
//! # Run property tests with more cases
//! PROPTEST_CASES=10000 cargo test prop_ --release
//! ```

use async_trait::async_trait;
use proptest::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

// =============================================================================
// SECTION 1: UNIT TEST PATTERNS FOR TOOL HANDLERS
// =============================================================================

/// Mock LLM provider for deterministic testing
///
/// Provides canned responses based on input patterns, enabling
/// fully deterministic unit tests without network calls.
#[derive(Clone)]
pub struct MockLlmProvider {
    /// Response mapping: input pattern -> response
    responses: Arc<RwLock<HashMap<String, MockLlmResponse>>>,
    /// Call counter for verification
    call_count: Arc<AtomicU64>,
    /// Simulated latency in milliseconds
    latency_ms: u64,
    /// Whether to simulate failures
    should_fail: Arc<RwLock<bool>>,
    /// Error message for failures
    failure_message: Arc<RwLock<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MockLlmResponse {
    pub content: String,
    pub tokens_used: u32,
    pub model: String,
    pub finish_reason: String,
}

impl MockLlmProvider {
    pub fn new() -> Self {
        Self {
            responses: Arc::new(RwLock::new(HashMap::new())),
            call_count: Arc::new(AtomicU64::new(0)),
            latency_ms: 0,
            should_fail: Arc::new(RwLock::new(false)),
            failure_message: Arc::new(RwLock::new("Mock LLM failure".to_string())),
        }
    }

    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    pub async fn register_response(&self, pattern: &str, response: MockLlmResponse) {
        let mut responses = self.responses.write().await;
        responses.insert(pattern.to_string(), response);
    }

    pub async fn set_should_fail(&self, fail: bool, message: &str) {
        let mut should_fail = self.should_fail.write().await;
        *should_fail = fail;
        let mut msg = self.failure_message.write().await;
        *msg = message.to_string();
    }

    pub fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::SeqCst)
    }

    pub async fn complete(&self, prompt: &str) -> Result<MockLlmResponse, String> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        // Check if we should fail
        if *self.should_fail.read().await {
            return Err(self.failure_message.read().await.clone());
        }

        // Simulate latency
        if self.latency_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
        }

        // Find matching response
        let responses = self.responses.read().await;
        for (pattern, response) in responses.iter() {
            if prompt.contains(pattern) {
                return Ok(response.clone());
            }
        }

        // Default response
        Ok(MockLlmResponse {
            content: format!("Mock response to: {}", &prompt[..prompt.len().min(50)]),
            tokens_used: 100,
            model: "mock-gpt-4".to_string(),
            finish_reason: "stop".to_string(),
        })
    }
}

impl Default for MockLlmProvider {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------------
// ThinkTool Handler Trait (mirrors production interface)
// -----------------------------------------------------------------------------

/// Result type for tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkToolResult {
    pub content: Vec<ThinkToolContent>,
    pub confidence: f64,
    pub tokens_used: u32,
    pub duration_ms: u64,
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkToolContent {
    Text {
        text: String,
    },
    Reasoning {
        steps: Vec<String>,
        conclusion: String,
    },
    Verification {
        claims: Vec<VerifiedClaim>,
    },
    Critique {
        issues: Vec<CritiqueIssue>,
        severity: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedClaim {
    pub claim: String,
    pub verified: bool,
    pub sources: Vec<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiqueIssue {
    pub issue: String,
    pub severity: String,
    pub suggestion: String,
}

/// Trait for ThinkTool handlers
#[async_trait]
pub trait ThinkToolHandler: Send + Sync {
    /// Tool name
    fn name(&self) -> &str;

    /// Tool description
    fn description(&self) -> &str;

    /// JSON Schema for input validation
    fn input_schema(&self) -> Value;

    /// Execute the tool
    async fn execute(&self, input: HashMap<String, Value>) -> Result<ThinkToolResult, String>;
}

// -----------------------------------------------------------------------------
// Mock ThinkTool Implementations
// -----------------------------------------------------------------------------

/// GigaThink: Expansive creative thinking
pub struct MockGigaThinkHandler {
    llm: MockLlmProvider,
}

impl MockGigaThinkHandler {
    pub fn new(llm: MockLlmProvider) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl ThinkToolHandler for MockGigaThinkHandler {
    fn name(&self) -> &str {
        "gigathink"
    }

    fn description(&self) -> &str {
        "Expansive creative thinking - generates 10+ diverse perspectives on a problem"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The problem or question to analyze"
                },
                "min_perspectives": {
                    "type": "integer",
                    "description": "Minimum number of perspectives to generate",
                    "default": 10,
                    "minimum": 5,
                    "maximum": 20
                },
                "domains": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional: Specific domains to consider"
                }
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: HashMap<String, Value>) -> Result<ThinkToolResult, String> {
        let start = Instant::now();

        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: query")?;

        let min_perspectives = input
            .get("min_perspectives")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        // Call mock LLM
        let prompt = format!(
            "Generate {} diverse perspectives on: {}",
            min_perspectives, query
        );
        let response = self.llm.complete(&prompt).await?;

        // Parse mock response into perspectives
        let perspectives: Vec<String> = (0..min_perspectives)
            .map(|i| format!("Perspective {}: {}", i + 1, response.content))
            .collect();

        Ok(ThinkToolResult {
            content: vec![ThinkToolContent::Reasoning {
                steps: perspectives,
                conclusion: format!("Analyzed from {} different angles", min_perspectives),
            }],
            confidence: 0.85,
            tokens_used: response.tokens_used,
            duration_ms: start.elapsed().as_millis() as u64,
            is_error: None,
        })
    }
}

/// LaserLogic: Precision deductive reasoning
pub struct MockLaserLogicHandler {
    llm: MockLlmProvider,
}

impl MockLaserLogicHandler {
    pub fn new(llm: MockLlmProvider) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl ThinkToolHandler for MockLaserLogicHandler {
    fn name(&self) -> &str {
        "laserlogic"
    }

    fn description(&self) -> &str {
        "Precision deductive reasoning with fallacy detection"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "argument": {
                    "type": "string",
                    "description": "The argument or claim to analyze"
                },
                "check_fallacies": {
                    "type": "boolean",
                    "description": "Whether to check for logical fallacies",
                    "default": true
                },
                "formal_logic": {
                    "type": "boolean",
                    "description": "Use formal logic notation",
                    "default": false
                }
            },
            "required": ["argument"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: HashMap<String, Value>) -> Result<ThinkToolResult, String> {
        let start = Instant::now();

        let argument = input
            .get("argument")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: argument")?;

        let check_fallacies = input
            .get("check_fallacies")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let prompt = format!(
            "Analyze logical structure{}: {}",
            if check_fallacies {
                " and detect fallacies"
            } else {
                ""
            },
            argument
        );
        let response = self.llm.complete(&prompt).await?;

        Ok(ThinkToolResult {
            content: vec![ThinkToolContent::Reasoning {
                steps: vec![
                    "Premise identification".to_string(),
                    "Logical chain analysis".to_string(),
                    format!(
                        "Fallacy check: {}",
                        if check_fallacies {
                            "enabled"
                        } else {
                            "disabled"
                        }
                    ),
                ],
                conclusion: response.content,
            }],
            confidence: 0.92,
            tokens_used: response.tokens_used,
            duration_ms: start.elapsed().as_millis() as u64,
            is_error: None,
        })
    }
}

/// BedRock: First principles decomposition
pub struct MockBedRockHandler {
    llm: MockLlmProvider,
}

impl MockBedRockHandler {
    pub fn new(llm: MockLlmProvider) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl ThinkToolHandler for MockBedRockHandler {
    fn name(&self) -> &str {
        "bedrock"
    }

    fn description(&self) -> &str {
        "First principles decomposition - breaks down problems to fundamental truths"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The problem to decompose"
                },
                "depth": {
                    "type": "integer",
                    "description": "Decomposition depth (1-5)",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 5
                }
            },
            "required": ["problem"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: HashMap<String, Value>) -> Result<ThinkToolResult, String> {
        let start = Instant::now();

        let problem = input
            .get("problem")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: problem")?;

        let depth = input.get("depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize;

        let prompt = format!(
            "Decompose to first principles (depth {}): {}",
            depth, problem
        );
        let response = self.llm.complete(&prompt).await?;

        let principles: Vec<String> = (0..depth)
            .map(|i| format!("Principle L{}: Fundamental truth {}", i + 1, i + 1))
            .collect();

        Ok(ThinkToolResult {
            content: vec![ThinkToolContent::Reasoning {
                steps: principles,
                conclusion: response.content,
            }],
            confidence: 0.88,
            tokens_used: response.tokens_used,
            duration_ms: start.elapsed().as_millis() as u64,
            is_error: None,
        })
    }
}

/// ProofGuard: Multi-source verification
pub struct MockProofGuardHandler {
    llm: MockLlmProvider,
}

impl MockProofGuardHandler {
    pub fn new(llm: MockLlmProvider) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl ThinkToolHandler for MockProofGuardHandler {
    fn name(&self) -> &str {
        "proofguard"
    }

    fn description(&self) -> &str {
        "Multi-source verification - validates claims against multiple sources"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Claims to verify",
                    "minItems": 1,
                    "maxItems": 10
                },
                "min_sources": {
                    "type": "integer",
                    "description": "Minimum sources required per claim",
                    "default": 3,
                    "minimum": 2,
                    "maximum": 5
                },
                "strict_mode": {
                    "type": "boolean",
                    "description": "Require all sources to agree",
                    "default": false
                }
            },
            "required": ["claims"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: HashMap<String, Value>) -> Result<ThinkToolResult, String> {
        let start = Instant::now();

        let claims = input
            .get("claims")
            .and_then(|v| v.as_array())
            .ok_or("Missing required field: claims")?;

        let min_sources = input
            .get("min_sources")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;

        let mut verified_claims = Vec::new();
        let mut total_tokens = 0u32;

        for claim_value in claims {
            let claim = claim_value.as_str().unwrap_or("Unknown claim");
            let prompt = format!("Verify claim with {} sources: {}", min_sources, claim);
            let response = self.llm.complete(&prompt).await?;
            total_tokens += response.tokens_used;

            verified_claims.push(VerifiedClaim {
                claim: claim.to_string(),
                verified: true, // Mock always verifies
                sources: (0..min_sources)
                    .map(|i| format!("Source {}", i + 1))
                    .collect(),
                confidence: 0.9,
            });
        }

        Ok(ThinkToolResult {
            content: vec![ThinkToolContent::Verification {
                claims: verified_claims,
            }],
            confidence: 0.95,
            tokens_used: total_tokens,
            duration_ms: start.elapsed().as_millis() as u64,
            is_error: None,
        })
    }
}

/// BrutalHonesty: Adversarial self-critique
pub struct MockBrutalHonestyHandler {
    llm: MockLlmProvider,
}

impl MockBrutalHonestyHandler {
    pub fn new(llm: MockLlmProvider) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl ThinkToolHandler for MockBrutalHonestyHandler {
    fn name(&self) -> &str {
        "brutalhonesty"
    }

    fn description(&self) -> &str {
        "Adversarial self-critique - identifies weaknesses and blind spots"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to critique"
                },
                "severity": {
                    "type": "string",
                    "enum": ["light", "standard", "adversarial", "brutal"],
                    "description": "Critique severity level",
                    "default": "standard"
                },
                "focus_areas": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Specific areas to focus critique on"
                }
            },
            "required": ["content"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: HashMap<String, Value>) -> Result<ThinkToolResult, String> {
        let start = Instant::now();

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing required field: content")?;

        let severity = input
            .get("severity")
            .and_then(|v| v.as_str())
            .unwrap_or("standard");

        let prompt = format!("Critique with {} severity: {}", severity, content);
        let response = self.llm.complete(&prompt).await?;

        let issue_count = match severity {
            "light" => 2,
            "standard" => 4,
            "adversarial" => 6,
            "brutal" => 8,
            _ => 4,
        };

        let issues: Vec<CritiqueIssue> = (0..issue_count)
            .map(|i| CritiqueIssue {
                issue: format!("Issue {}: Potential weakness identified", i + 1),
                severity: severity.to_string(),
                suggestion: format!("Suggestion {}: Consider alternative approach", i + 1),
            })
            .collect();

        Ok(ThinkToolResult {
            content: vec![ThinkToolContent::Critique {
                issues,
                severity: severity.to_string(),
            }],
            confidence: 0.87,
            tokens_used: response.tokens_used,
            duration_ms: start.elapsed().as_millis() as u64,
            is_error: None,
        })
    }
}

// =============================================================================
// SECTION 2: UNIT TESTS
// =============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // GigaThink Unit Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_gigathink_basic_execution() {
        let llm = MockLlmProvider::new();
        llm.register_response(
            "Generate",
            MockLlmResponse {
                content: "Creative perspective on the problem".to_string(),
                tokens_used: 150,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            },
        )
        .await;

        let handler = MockGigaThinkHandler::new(llm.clone());

        let mut input = HashMap::new();
        input.insert("query".to_string(), json!("How to improve code quality?"));

        let result = handler.execute(input).await.unwrap();

        assert!(result.confidence > 0.8);
        assert!(result.tokens_used > 0);
        assert!(result.is_error.is_none());
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn test_gigathink_custom_perspectives() {
        let llm = MockLlmProvider::new();
        let handler = MockGigaThinkHandler::new(llm);

        let mut input = HashMap::new();
        input.insert("query".to_string(), json!("Test query"));
        input.insert("min_perspectives".to_string(), json!(15));

        let result = handler.execute(input).await.unwrap();

        if let ThinkToolContent::Reasoning { steps, .. } = &result.content[0] {
            assert_eq!(steps.len(), 15);
        } else {
            panic!("Expected Reasoning content");
        }
    }

    #[tokio::test]
    async fn test_gigathink_missing_query_error() {
        let llm = MockLlmProvider::new();
        let handler = MockGigaThinkHandler::new(llm);

        let input = HashMap::new(); // No query

        let result = handler.execute(input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("query"));
    }

    // -------------------------------------------------------------------------
    // LaserLogic Unit Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_laserlogic_with_fallacy_check() {
        let llm = MockLlmProvider::new();
        let handler = MockLaserLogicHandler::new(llm);

        let mut input = HashMap::new();
        input.insert(
            "argument".to_string(),
            json!("All swans are white. This bird is white. Therefore, it is a swan."),
        );
        input.insert("check_fallacies".to_string(), json!(true));

        let result = handler.execute(input).await.unwrap();

        assert!(result.confidence > 0.9);
        if let ThinkToolContent::Reasoning { steps, .. } = &result.content[0] {
            assert!(steps.iter().any(|s| s.contains("Fallacy check: enabled")));
        }
    }

    #[tokio::test]
    async fn test_laserlogic_without_fallacy_check() {
        let llm = MockLlmProvider::new();
        let handler = MockLaserLogicHandler::new(llm);

        let mut input = HashMap::new();
        input.insert(
            "argument".to_string(),
            json!("P implies Q. P. Therefore Q."),
        );
        input.insert("check_fallacies".to_string(), json!(false));

        let result = handler.execute(input).await.unwrap();

        if let ThinkToolContent::Reasoning { steps, .. } = &result.content[0] {
            assert!(steps.iter().any(|s| s.contains("disabled")));
        }
    }

    // -------------------------------------------------------------------------
    // BedRock Unit Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_bedrock_decomposition_depth() {
        let llm = MockLlmProvider::new();
        let handler = MockBedRockHandler::new(llm);

        for depth in [1, 3, 5] {
            let mut input = HashMap::new();
            input.insert("problem".to_string(), json!("Why do startups fail?"));
            input.insert("depth".to_string(), json!(depth));

            let result = handler.execute(input).await.unwrap();

            if let ThinkToolContent::Reasoning { steps, .. } = &result.content[0] {
                assert_eq!(steps.len(), depth as usize);
            }
        }
    }

    // -------------------------------------------------------------------------
    // ProofGuard Unit Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_proofguard_multiple_claims() {
        let llm = MockLlmProvider::new();
        let handler = MockProofGuardHandler::new(llm.clone());

        let mut input = HashMap::new();
        input.insert(
            "claims".to_string(),
            json!(["Claim 1", "Claim 2", "Claim 3"]),
        );
        input.insert("min_sources".to_string(), json!(3));

        let result = handler.execute(input).await.unwrap();

        if let ThinkToolContent::Verification { claims } = &result.content[0] {
            assert_eq!(claims.len(), 3);
            for claim in claims {
                assert_eq!(claim.sources.len(), 3);
            }
        }

        // Should have called LLM once per claim
        assert_eq!(llm.call_count(), 3);
    }

    // -------------------------------------------------------------------------
    // BrutalHonesty Unit Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_brutalhonesty_severity_levels() {
        let llm = MockLlmProvider::new();
        let handler = MockBrutalHonestyHandler::new(llm);

        let severity_to_issues = [
            ("light", 2),
            ("standard", 4),
            ("adversarial", 6),
            ("brutal", 8),
        ];

        for (severity, expected_issues) in severity_to_issues {
            let mut input = HashMap::new();
            input.insert("content".to_string(), json!("My brilliant plan..."));
            input.insert("severity".to_string(), json!(severity));

            let result = handler.execute(input).await.unwrap();

            if let ThinkToolContent::Critique { issues, .. } = &result.content[0] {
                assert_eq!(
                    issues.len(),
                    expected_issues,
                    "Severity {} should produce {} issues",
                    severity,
                    expected_issues
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Error Handling Unit Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_handler_llm_failure_propagation() {
        let llm = MockLlmProvider::new();
        llm.set_should_fail(true, "API rate limit exceeded").await;

        let handler = MockGigaThinkHandler::new(llm);

        let mut input = HashMap::new();
        input.insert("query".to_string(), json!("Test query"));

        let result = handler.execute(input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("rate limit"));
    }

    #[tokio::test]
    async fn test_handler_input_type_validation() {
        let llm = MockLlmProvider::new();
        let handler = MockGigaThinkHandler::new(llm);

        // Wrong type for query (number instead of string)
        let mut input = HashMap::new();
        input.insert("query".to_string(), json!(12345));

        let result = handler.execute(input).await;
        assert!(result.is_err());
    }
}

// =============================================================================
// SECTION 3: INTEGRATION TEST APPROACH FOR TOOL EXECUTION
// =============================================================================

/// MCP Tool Registry for testing
pub struct TestToolRegistry {
    handlers: HashMap<String, Arc<dyn ThinkToolHandler>>,
}

impl TestToolRegistry {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    pub fn register(&mut self, handler: Arc<dyn ThinkToolHandler>) {
        self.handlers.insert(handler.name().to_string(), handler);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ThinkToolHandler>> {
        self.handlers.get(name)
    }

    pub fn list_tools(&self) -> Vec<ToolDefinition> {
        self.handlers
            .values()
            .map(|h| ToolDefinition {
                name: h.name().to_string(),
                description: h.description().to_string(),
                input_schema: h.input_schema(),
            })
            .collect()
    }

    pub async fn call_tool(
        &self,
        name: &str,
        input: HashMap<String, Value>,
    ) -> Result<ThinkToolResult, String> {
        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| format!("Tool not found: {}", name))?;

        handler.execute(input).await
    }
}

impl Default for TestToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Creates a fully configured test registry with all 5 ThinkTools
    async fn create_test_registry() -> TestToolRegistry {
        let llm = MockLlmProvider::new();

        // Register standard responses
        llm.register_response(
            "Generate",
            MockLlmResponse {
                content: "Generated perspectives".to_string(),
                tokens_used: 200,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            },
        )
        .await;

        let mut registry = TestToolRegistry::new();

        registry.register(Arc::new(MockGigaThinkHandler::new(llm.clone())));
        registry.register(Arc::new(MockLaserLogicHandler::new(llm.clone())));
        registry.register(Arc::new(MockBedRockHandler::new(llm.clone())));
        registry.register(Arc::new(MockProofGuardHandler::new(llm.clone())));
        registry.register(Arc::new(MockBrutalHonestyHandler::new(llm.clone())));

        registry
    }

    #[tokio::test]
    async fn test_registry_lists_all_tools() {
        let registry = create_test_registry().await;
        let tools = registry.list_tools();

        assert_eq!(tools.len(), 5);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"gigathink"));
        assert!(names.contains(&"laserlogic"));
        assert!(names.contains(&"bedrock"));
        assert!(names.contains(&"proofguard"));
        assert!(names.contains(&"brutalhonesty"));
    }

    #[tokio::test]
    async fn test_registry_tool_execution_flow() {
        let registry = create_test_registry().await;

        // 1. List tools
        let tools = registry.list_tools();
        assert!(!tools.is_empty());

        // 2. Get tool schema
        let gigathink = tools.iter().find(|t| t.name == "gigathink").unwrap();
        assert!(gigathink.input_schema.get("properties").is_some());

        // 3. Call tool
        let mut input = HashMap::new();
        input.insert("query".to_string(), json!("Integration test query"));

        let result = registry.call_tool("gigathink", input).await.unwrap();
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_chained_tool_execution() {
        let registry = create_test_registry().await;

        // Step 1: Generate ideas with GigaThink
        let mut input1 = HashMap::new();
        input1.insert("query".to_string(), json!("How to improve testing?"));

        let gigathink_result = registry.call_tool("gigathink", input1).await.unwrap();

        // Step 2: Analyze logical structure with LaserLogic
        let gigathink_output = match &gigathink_result.content[0] {
            ThinkToolContent::Reasoning { conclusion, .. } => conclusion.clone(),
            _ => "Generated output".to_string(),
        };

        let mut input2 = HashMap::new();
        input2.insert("argument".to_string(), json!(gigathink_output));

        let laserlogic_result = registry.call_tool("laserlogic", input2).await.unwrap();

        // Step 3: Critique the analysis with BrutalHonesty
        let mut input3 = HashMap::new();
        input3.insert("content".to_string(), json!("Analysis output"));
        input3.insert("severity".to_string(), json!("adversarial"));

        let critique_result = registry.call_tool("brutalhonesty", input3).await.unwrap();

        // All steps should complete successfully
        assert!(gigathink_result.is_error.is_none());
        assert!(laserlogic_result.is_error.is_none());
        assert!(critique_result.is_error.is_none());
    }

    #[tokio::test]
    async fn test_parallel_tool_execution() {
        let registry = Arc::new(create_test_registry().await);

        let mut handles = Vec::new();

        for i in 0..10 {
            let registry = Arc::clone(&registry);
            handles.push(tokio::spawn(async move {
                let mut input = HashMap::new();
                input.insert("query".to_string(), json!(format!("Query {}", i)));
                registry.call_tool("gigathink", input).await
            }));
        }

        let results: Vec<_> = futures::future::join_all(handles).await;

        for result in results {
            let inner = result.unwrap();
            assert!(inner.is_ok());
        }
    }

    #[tokio::test]
    async fn test_tool_not_found_error() {
        let registry = create_test_registry().await;

        let result = registry.call_tool("nonexistent_tool", HashMap::new()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }
}

// =============================================================================
// SECTION 4: MOCK LLM STRATEGIES FOR CI
// =============================================================================

/// Deterministic LLM mock for CI environments
///
/// Provides consistent, reproducible responses based on:
/// - Input hashing for deterministic selection
/// - Configurable response pools
/// - Latency simulation
/// - Error injection
pub struct DeterministicLlmMock {
    /// Response pool keyed by content hash prefix
    response_pool: HashMap<String, Vec<MockLlmResponse>>,
    /// Sequence counter for deterministic selection
    sequence: AtomicU64,
    /// Configuration
    config: DeterministicMockConfig,
}

#[derive(Clone)]
pub struct DeterministicMockConfig {
    /// Whether to use input hash for response selection
    pub hash_based_selection: bool,
    /// Fixed latency in milliseconds (0 = no delay)
    pub latency_ms: u64,
    /// Error rate (0.0 - 1.0)
    pub error_rate: f64,
    /// Maximum tokens per response
    pub max_tokens: u32,
}

impl Default for DeterministicMockConfig {
    fn default() -> Self {
        Self {
            hash_based_selection: true,
            latency_ms: 0,
            error_rate: 0.0,
            max_tokens: 500,
        }
    }
}

impl DeterministicLlmMock {
    pub fn new(config: DeterministicMockConfig) -> Self {
        Self {
            response_pool: Self::default_response_pool(),
            sequence: AtomicU64::new(0),
            config,
        }
    }

    fn default_response_pool() -> HashMap<String, Vec<MockLlmResponse>> {
        let mut pool = HashMap::new();

        // Responses for different query types
        pool.insert(
            "analysis".to_string(),
            vec![
                MockLlmResponse {
                    content: "Based on systematic analysis, the key factors are...".to_string(),
                    tokens_used: 150,
                    model: "mock-gpt-4".to_string(),
                    finish_reason: "stop".to_string(),
                },
                MockLlmResponse {
                    content: "After careful examination, I conclude that...".to_string(),
                    tokens_used: 180,
                    model: "mock-gpt-4".to_string(),
                    finish_reason: "stop".to_string(),
                },
            ],
        );

        pool.insert(
            "creative".to_string(),
            vec![MockLlmResponse {
                content: "Innovative perspective 1: Consider unconventional...".to_string(),
                tokens_used: 200,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            }],
        );

        pool.insert(
            "critique".to_string(),
            vec![MockLlmResponse {
                content: "Critical observation: The argument overlooks...".to_string(),
                tokens_used: 175,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            }],
        );

        pool.insert(
            "default".to_string(),
            vec![MockLlmResponse {
                content: "Response to the query provided.".to_string(),
                tokens_used: 100,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            }],
        );

        pool
    }

    fn hash_input(&self, input: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        hasher.finish()
    }

    fn classify_input(&self, input: &str) -> &str {
        let lower = input.to_lowercase();
        if lower.contains("analyz") || lower.contains("decompos") || lower.contains("logic") {
            "analysis"
        } else if lower.contains("creative")
            || lower.contains("perspectiv")
            || lower.contains("generat")
        {
            "creative"
        } else if lower.contains("critiqu") || lower.contains("weakness") || lower.contains("flaw")
        {
            "critique"
        } else {
            "default"
        }
    }

    pub async fn complete(&self, prompt: &str) -> Result<MockLlmResponse, String> {
        // Simulate latency
        if self.config.latency_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.config.latency_ms)).await;
        }

        // Check for error injection
        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        if self.config.error_rate > 0.0 {
            let error_threshold = (1.0 / self.config.error_rate) as u64;
            if seq % error_threshold == 0 {
                return Err("Simulated LLM API error for testing".to_string());
            }
        }

        // Select response
        let category = self.classify_input(prompt);
        let responses = self.response_pool.get(category).unwrap();

        let index = if self.config.hash_based_selection {
            (self.hash_input(prompt) % responses.len() as u64) as usize
        } else {
            (seq % responses.len() as u64) as usize
        };

        Ok(responses[index].clone())
    }
}

/// Record/Replay LLM mock for golden file testing
pub struct RecordReplayLlmMock {
    /// Recorded interactions
    recordings: Arc<RwLock<Vec<LlmInteraction>>>,
    /// Replay mode
    mode: ReplayMode,
    /// Current replay index
    replay_index: AtomicU64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LlmInteraction {
    pub prompt: String,
    pub response: MockLlmResponse,
    pub timestamp_ms: u64,
}

#[derive(Clone, Copy)]
pub enum ReplayMode {
    Record,
    Replay,
    Passthrough,
}

impl RecordReplayLlmMock {
    pub fn new(mode: ReplayMode) -> Self {
        Self {
            recordings: Arc::new(RwLock::new(Vec::new())),
            mode,
            replay_index: AtomicU64::new(0),
        }
    }

    pub async fn load_recordings(&self, json: &str) -> Result<(), String> {
        let interactions: Vec<LlmInteraction> =
            serde_json::from_str(json).map_err(|e| e.to_string())?;
        let mut recordings = self.recordings.write().await;
        *recordings = interactions;
        Ok(())
    }

    pub async fn save_recordings(&self) -> Result<String, String> {
        let recordings = self.recordings.read().await;
        serde_json::to_string_pretty(&*recordings).map_err(|e| e.to_string())
    }

    pub async fn complete(
        &self,
        prompt: &str,
        fallback: &MockLlmProvider,
    ) -> Result<MockLlmResponse, String> {
        match self.mode {
            ReplayMode::Record => {
                let response = fallback.complete(prompt).await?;
                let interaction = LlmInteraction {
                    prompt: prompt.to_string(),
                    response: response.clone(),
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };
                self.recordings.write().await.push(interaction);
                Ok(response)
            }
            ReplayMode::Replay => {
                let index = self.replay_index.fetch_add(1, Ordering::SeqCst) as usize;
                let recordings = self.recordings.read().await;
                recordings
                    .get(index)
                    .map(|i| i.response.clone())
                    .ok_or_else(|| format!("No recording at index {}", index))
            }
            ReplayMode::Passthrough => fallback.complete(prompt).await,
        }
    }
}

#[cfg(test)]
mod mock_llm_tests {
    use super::*;

    #[tokio::test]
    async fn test_deterministic_mock_consistent_responses() {
        let mock = DeterministicLlmMock::new(DeterministicMockConfig::default());

        let prompt = "Analyze this code for bugs";

        // Same prompt should give same response
        let response1 = mock.complete(prompt).await.unwrap();
        let response2 = mock.complete(prompt).await.unwrap();

        assert_eq!(response1.content, response2.content);
    }

    #[tokio::test]
    async fn test_deterministic_mock_error_injection() {
        let config = DeterministicMockConfig {
            error_rate: 0.5, // 50% error rate
            ..Default::default()
        };
        let mock = DeterministicLlmMock::new(config);

        let mut errors = 0;
        let mut successes = 0;

        for i in 0..100 {
            match mock.complete(&format!("Query {}", i)).await {
                Ok(_) => successes += 1,
                Err(_) => errors += 1,
            }
        }

        // Should have roughly 50% errors (allowing for some variance)
        assert!(errors > 30 && errors < 70);
        assert!(successes > 30 && successes < 70);
    }

    #[tokio::test]
    async fn test_record_replay_mode() {
        let mock = RecordReplayLlmMock::new(ReplayMode::Record);
        let fallback = MockLlmProvider::new();

        // Record some interactions
        mock.complete("Query 1", &fallback).await.unwrap();
        mock.complete("Query 2", &fallback).await.unwrap();

        // Export recordings
        let json = mock.save_recordings().await.unwrap();

        // Create replay mock
        let replay_mock = RecordReplayLlmMock::new(ReplayMode::Replay);
        replay_mock.load_recordings(&json).await.unwrap();

        // Replay should return same responses
        let response1 = replay_mock.complete("ignored", &fallback).await.unwrap();
        let response2 = replay_mock.complete("ignored", &fallback).await.unwrap();

        assert!(response1.tokens_used > 0);
        assert!(response2.tokens_used > 0);
    }
}

// =============================================================================
// SECTION 5: PROPERTY-BASED TESTING FOR JSON SCHEMAS
// =============================================================================

/// Arbitrary JSON value generator with schema constraints
fn arb_json_string() -> impl Strategy<Value = Value> {
    "[a-zA-Z0-9 ]{1,100}".prop_map(|s| json!(s))
}

fn arb_json_integer() -> impl Strategy<Value = Value> {
    (-1000i64..1000).prop_map(|i| json!(i))
}

fn arb_json_number() -> impl Strategy<Value = Value> {
    (-1000.0f64..1000.0).prop_map(|f| json!(f))
}

fn arb_json_boolean() -> impl Strategy<Value = Value> {
    any::<bool>().prop_map(|b| json!(b))
}

fn arb_json_array() -> impl Strategy<Value = Value> {
    prop::collection::vec("[a-zA-Z0-9]{1,20}", 0..10).prop_map(|v| json!(v))
}

/// Strategy for generating valid GigaThink inputs
fn arb_gigathink_input() -> impl Strategy<Value = HashMap<String, Value>> {
    (
        "[a-zA-Z0-9 ?]{10,200}",                                      // query
        prop::option::of(5u64..=20),                                  // min_perspectives
        prop::option::of(prop::collection::vec("[a-z]{3,15}", 0..5)), // domains
    )
        .prop_map(|(query, perspectives, domains)| {
            let mut map = HashMap::new();
            map.insert("query".to_string(), json!(query));
            if let Some(p) = perspectives {
                map.insert("min_perspectives".to_string(), json!(p));
            }
            if let Some(d) = domains {
                map.insert("domains".to_string(), json!(d));
            }
            map
        })
}

/// Strategy for generating valid LaserLogic inputs
fn arb_laserlogic_input() -> impl Strategy<Value = HashMap<String, Value>> {
    (
        "[a-zA-Z0-9 .,]{20,500}",        // argument
        prop::option::of(any::<bool>()), // check_fallacies
        prop::option::of(any::<bool>()), // formal_logic
    )
        .prop_map(|(argument, check_fallacies, formal_logic)| {
            let mut map = HashMap::new();
            map.insert("argument".to_string(), json!(argument));
            if let Some(cf) = check_fallacies {
                map.insert("check_fallacies".to_string(), json!(cf));
            }
            if let Some(fl) = formal_logic {
                map.insert("formal_logic".to_string(), json!(fl));
            }
            map
        })
}

/// Strategy for generating valid ProofGuard inputs
fn arb_proofguard_input() -> impl Strategy<Value = HashMap<String, Value>> {
    (
        prop::collection::vec("[a-zA-Z0-9 ]{10,100}", 1..=10), // claims
        prop::option::of(2u64..=5),                            // min_sources
        prop::option::of(any::<bool>()),                       // strict_mode
    )
        .prop_map(|(claims, min_sources, strict_mode)| {
            let mut map = HashMap::new();
            map.insert("claims".to_string(), json!(claims));
            if let Some(ms) = min_sources {
                map.insert("min_sources".to_string(), json!(ms));
            }
            if let Some(sm) = strict_mode {
                map.insert("strict_mode".to_string(), json!(sm));
            }
            map
        })
}

/// Strategy for generating valid BrutalHonesty inputs
fn arb_brutalhonesty_input() -> impl Strategy<Value = HashMap<String, Value>> {
    (
        "[a-zA-Z0-9 .,!?]{20,500}", // content
        prop::option::of(prop_oneof![
            Just("light"),
            Just("standard"),
            Just("adversarial"),
            Just("brutal"),
        ]),
        prop::option::of(prop::collection::vec("[a-z]{3,15}", 0..5)), // focus_areas
    )
        .prop_map(|(content, severity, focus_areas)| {
            let mut map = HashMap::new();
            map.insert("content".to_string(), json!(content));
            if let Some(s) = severity {
                map.insert("severity".to_string(), json!(s));
            }
            if let Some(fa) = focus_areas {
                map.insert("focus_areas".to_string(), json!(fa));
            }
            map
        })
}

/// JSON Schema validator for testing
pub struct SchemaValidator {
    schema: Value,
}

impl SchemaValidator {
    pub fn new(schema: Value) -> Self {
        Self { schema }
    }

    /// Validates input against schema (simplified)
    pub fn validate(&self, input: &Value) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if let Some(schema_type) = self.schema.get("type").and_then(|t| t.as_str()) {
            match schema_type {
                "object" => {
                    if !input.is_object() {
                        errors.push("Expected object".to_string());
                        return Err(errors);
                    }

                    // Check required fields
                    if let Some(required) = self.schema.get("required").and_then(|r| r.as_array()) {
                        for field in required {
                            if let Some(field_name) = field.as_str() {
                                if input.get(field_name).is_none() {
                                    errors.push(format!("Missing required field: {}", field_name));
                                }
                            }
                        }
                    }

                    // Check property types
                    if let Some(properties) =
                        self.schema.get("properties").and_then(|p| p.as_object())
                    {
                        if let Some(input_obj) = input.as_object() {
                            for (key, value) in input_obj {
                                if let Some(prop_schema) = properties.get(key) {
                                    if let Some(prop_type) =
                                        prop_schema.get("type").and_then(|t| t.as_str())
                                    {
                                        let valid = match prop_type {
                                            "string" => value.is_string(),
                                            "integer" => value.is_i64() || value.is_u64(),
                                            "number" => value.is_number(),
                                            "boolean" => value.is_boolean(),
                                            "array" => value.is_array(),
                                            "object" => value.is_object(),
                                            _ => true,
                                        };
                                        if !valid {
                                            errors.push(format!(
                                                "Field '{}' has wrong type, expected {}",
                                                key, prop_type
                                            ));
                                        }
                                    }

                                    // Check enum constraints
                                    if let Some(enum_values) =
                                        prop_schema.get("enum").and_then(|e| e.as_array())
                                    {
                                        if !enum_values.contains(value) {
                                            errors.push(format!(
                                                "Field '{}' must be one of {:?}",
                                                key, enum_values
                                            ));
                                        }
                                    }

                                    // Check minimum/maximum for integers
                                    if let Some(min) =
                                        prop_schema.get("minimum").and_then(|m| m.as_i64())
                                    {
                                        if let Some(val) = value.as_i64() {
                                            if val < min {
                                                errors.push(format!(
                                                    "Field '{}' must be >= {}",
                                                    key, min
                                                ));
                                            }
                                        }
                                    }
                                    if let Some(max) =
                                        prop_schema.get("maximum").and_then(|m| m.as_i64())
                                    {
                                        if let Some(val) = value.as_i64() {
                                            if val > max {
                                                errors.push(format!(
                                                    "Field '{}' must be <= {}",
                                                    key, max
                                                ));
                                            }
                                        }
                                    }
                                }
                            }

                            // Check additionalProperties
                            if self.schema.get("additionalProperties") == Some(&json!(false)) {
                                if let Some(props) = properties
                                    .keys()
                                    .collect::<std::collections::HashSet<_>>()
                                    .into_iter()
                                    .next()
                                {
                                    for key in input_obj.keys() {
                                        if !properties.contains_key(key) {
                                            errors.push(format!("Unknown property: {}", key));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    // -------------------------------------------------------------------------
    // GigaThink Schema Property Tests
    // -------------------------------------------------------------------------

    #[test]
    fn prop_gigathink_valid_inputs_pass_schema(input in arb_gigathink_input()) {
        let handler = MockGigaThinkHandler::new(MockLlmProvider::new());
        let schema = handler.input_schema();
        let validator = SchemaValidator::new(schema);

        let input_value = serde_json::to_value(&input).unwrap();
        let result = validator.validate(&input_value);

        prop_assert!(result.is_ok(), "Valid input should pass schema: {:?}", result);
    }

    #[test]
    fn prop_gigathink_perspectives_in_range(min_perspectives in 5u64..=20) {
        let handler = MockGigaThinkHandler::new(MockLlmProvider::new());
        let schema = handler.input_schema();

        let props = schema.get("properties").unwrap();
        let min_prop = props.get("min_perspectives").unwrap();

        let schema_min = min_prop.get("minimum").unwrap().as_u64().unwrap();
        let schema_max = min_prop.get("maximum").unwrap().as_u64().unwrap();

        prop_assert!(min_perspectives >= schema_min);
        prop_assert!(min_perspectives <= schema_max);
    }

    // -------------------------------------------------------------------------
    // LaserLogic Schema Property Tests
    // -------------------------------------------------------------------------

    #[test]
    fn prop_laserlogic_valid_inputs_pass_schema(input in arb_laserlogic_input()) {
        let handler = MockLaserLogicHandler::new(MockLlmProvider::new());
        let schema = handler.input_schema();
        let validator = SchemaValidator::new(schema);

        let input_value = serde_json::to_value(&input).unwrap();
        let result = validator.validate(&input_value);

        prop_assert!(result.is_ok(), "Valid input should pass schema: {:?}", result);
    }

    // -------------------------------------------------------------------------
    // ProofGuard Schema Property Tests
    // -------------------------------------------------------------------------

    #[test]
    fn prop_proofguard_valid_inputs_pass_schema(input in arb_proofguard_input()) {
        let handler = MockProofGuardHandler::new(MockLlmProvider::new());
        let schema = handler.input_schema();
        let validator = SchemaValidator::new(schema);

        let input_value = serde_json::to_value(&input).unwrap();
        let result = validator.validate(&input_value);

        prop_assert!(result.is_ok(), "Valid input should pass schema: {:?}", result);
    }

    #[test]
    fn prop_proofguard_claims_count_valid(claims in prop::collection::vec("[a-z]{5,50}", 1..=10)) {
        let handler = MockProofGuardHandler::new(MockLlmProvider::new());
        let schema = handler.input_schema();

        let props = schema.get("properties").unwrap();
        let claims_prop = props.get("claims").unwrap();

        let min_items = claims_prop.get("minItems").unwrap().as_u64().unwrap();
        let max_items = claims_prop.get("maxItems").unwrap().as_u64().unwrap();

        prop_assert!(claims.len() >= min_items as usize);
        prop_assert!(claims.len() <= max_items as usize);
    }

    // -------------------------------------------------------------------------
    // BrutalHonesty Schema Property Tests
    // -------------------------------------------------------------------------

    #[test]
    fn prop_brutalhonesty_valid_inputs_pass_schema(input in arb_brutalhonesty_input()) {
        let handler = MockBrutalHonestyHandler::new(MockLlmProvider::new());
        let schema = handler.input_schema();
        let validator = SchemaValidator::new(schema);

        let input_value = serde_json::to_value(&input).unwrap();
        let result = validator.validate(&input_value);

        prop_assert!(result.is_ok(), "Valid input should pass schema: {:?}", result);
    }

    #[test]
    fn prop_brutalhonesty_severity_valid(severity in prop_oneof![
        Just("light"),
        Just("standard"),
        Just("adversarial"),
        Just("brutal"),
    ]) {
        let handler = MockBrutalHonestyHandler::new(MockLlmProvider::new());
        let schema = handler.input_schema();

        let props = schema.get("properties").unwrap();
        let severity_prop = props.get("severity").unwrap();
        let enum_values = severity_prop.get("enum").unwrap().as_array().unwrap();

        let valid = enum_values.iter().any(|v| v.as_str() == Some(severity));
        prop_assert!(valid, "Severity {} should be valid", severity);
    }

    // -------------------------------------------------------------------------
    // Cross-Tool Invariants
    // -------------------------------------------------------------------------

    #[test]
    fn prop_all_tools_have_required_fields(
        tool_name in prop_oneof![
            Just("gigathink"),
            Just("laserlogic"),
            Just("bedrock"),
            Just("proofguard"),
            Just("brutalhonesty"),
        ]
    ) {
        let llm = MockLlmProvider::new();

        let handler: Box<dyn ThinkToolHandler> = match tool_name {
            "gigathink" => Box::new(MockGigaThinkHandler::new(llm)),
            "laserlogic" => Box::new(MockLaserLogicHandler::new(llm)),
            "bedrock" => Box::new(MockBedRockHandler::new(llm)),
            "proofguard" => Box::new(MockProofGuardHandler::new(llm)),
            "brutalhonesty" => Box::new(MockBrutalHonestyHandler::new(llm)),
            _ => unreachable!(),
        };

        let schema = handler.input_schema();

        // Every tool should have type: object
        prop_assert_eq!(schema.get("type").unwrap(), "object");

        // Every tool should have properties
        prop_assert!(schema.get("properties").is_some());

        // Every tool should have required array
        prop_assert!(schema.get("required").unwrap().is_array());
    }
}

// =============================================================================
// SECTION 6: CONTRACT TESTING FOR MCP COMPLIANCE
// =============================================================================

/// MCP Protocol Contract definitions
mod mcp_contracts {
    use super::*;

    /// JSON-RPC 2.0 request structure
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JsonRpcRequest {
        pub jsonrpc: String,
        pub id: JsonRpcId,
        pub method: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub params: Option<Value>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum JsonRpcId {
        String(String),
        Number(i64),
    }

    /// JSON-RPC 2.0 response structure
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JsonRpcResponse {
        pub jsonrpc: String,
        pub id: JsonRpcId,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<JsonRpcError>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct JsonRpcError {
        pub code: i32,
        pub message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<Value>,
    }

    /// MCP standard error codes
    pub mod error_codes {
        pub const PARSE_ERROR: i32 = -32700;
        pub const INVALID_REQUEST: i32 = -32600;
        pub const METHOD_NOT_FOUND: i32 = -32601;
        pub const INVALID_PARAMS: i32 = -32602;
        pub const INTERNAL_ERROR: i32 = -32603;

        // MCP-specific
        pub const REQUEST_CANCELLED: i32 = -32800;
        pub const RESOURCE_NOT_FOUND: i32 = -32801;
        pub const TOOL_NOT_FOUND: i32 = -32802;
        pub const INVALID_TOOL_INPUT: i32 = -32803;
    }

    /// MCP Tool definition contract
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct McpToolDefinition {
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(rename = "inputSchema")]
        pub input_schema: Value,
    }

    /// MCP Tool call result contract
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct McpToolResult {
        pub content: Vec<McpContent>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "isError")]
        pub is_error: Option<bool>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum McpContent {
        Text {
            text: String,
        },
        Image {
            data: String,
            #[serde(rename = "mimeType")]
            mime_type: String,
        },
        Resource {
            uri: String,
        },
    }

    /// Validates that a response follows JSON-RPC 2.0 spec
    pub fn validate_jsonrpc_response(response: &JsonRpcResponse) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // jsonrpc must be "2.0"
        if response.jsonrpc != "2.0" {
            errors.push(format!("jsonrpc must be '2.0', got '{}'", response.jsonrpc));
        }

        // Must have either result or error, not both
        match (&response.result, &response.error) {
            (Some(_), Some(_)) => {
                errors.push("Response cannot have both result and error".to_string());
            }
            (None, None) => {
                errors.push("Response must have either result or error".to_string());
            }
            _ => {}
        }

        // If error present, validate structure
        if let Some(error) = &response.error {
            if error.message.is_empty() {
                errors.push("Error message cannot be empty".to_string());
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validates that a tool definition follows MCP spec
    pub fn validate_tool_definition(tool: &McpToolDefinition) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Name is required and non-empty
        if tool.name.is_empty() {
            errors.push("Tool name cannot be empty".to_string());
        }

        // Name should be lowercase with underscores
        if !tool
            .name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c == '_')
        {
            errors.push("Tool name should be lowercase with underscores".to_string());
        }

        // inputSchema must be an object with type: "object"
        if tool.input_schema.get("type") != Some(&json!("object")) {
            errors.push("inputSchema must have type: 'object'".to_string());
        }

        // inputSchema should have properties
        if tool.input_schema.get("properties").is_none() {
            errors.push("inputSchema should have 'properties'".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validates tool result follows MCP spec
    pub fn validate_tool_result(result: &McpToolResult) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Content array should not be empty
        if result.content.is_empty() {
            errors.push("Tool result content cannot be empty".to_string());
        }

        // Validate each content item
        for (i, content) in result.content.iter().enumerate() {
            match content {
                McpContent::Text { text } => {
                    if text.is_empty() {
                        errors.push(format!("Content[{}]: Text cannot be empty", i));
                    }
                }
                McpContent::Image { data, mime_type } => {
                    if data.is_empty() {
                        errors.push(format!("Content[{}]: Image data cannot be empty", i));
                    }
                    if !mime_type.starts_with("image/") {
                        errors.push(format!(
                            "Content[{}]: Image mimeType should start with 'image/'",
                            i
                        ));
                    }
                }
                McpContent::Resource { uri } => {
                    if uri.is_empty() {
                        errors.push(format!("Content[{}]: Resource URI cannot be empty", i));
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod contract_tests {
    use super::mcp_contracts::*;
    use super::*;

    // -------------------------------------------------------------------------
    // JSON-RPC 2.0 Contract Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jsonrpc_response_success_contract() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::String("test-1".to_string()),
            result: Some(json!({"tools": []})),
            error: None,
        };

        let validation = validate_jsonrpc_response(&response);
        assert!(validation.is_ok());
    }

    #[test]
    fn test_jsonrpc_response_error_contract() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(1),
            result: None,
            error: Some(JsonRpcError {
                code: error_codes::METHOD_NOT_FOUND,
                message: "Method not found".to_string(),
                data: None,
            }),
        };

        let validation = validate_jsonrpc_response(&response);
        assert!(validation.is_ok());
    }

    #[test]
    fn test_jsonrpc_invalid_version() {
        let response = JsonRpcResponse {
            jsonrpc: "1.0".to_string(), // Wrong version
            id: JsonRpcId::String("test".to_string()),
            result: Some(json!({})),
            error: None,
        };

        let validation = validate_jsonrpc_response(&response);
        assert!(validation.is_err());
        assert!(validation.unwrap_err()[0].contains("2.0"));
    }

    #[test]
    fn test_jsonrpc_both_result_and_error() {
        let response = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::String("test".to_string()),
            result: Some(json!({})),
            error: Some(JsonRpcError {
                code: -32600,
                message: "Error".to_string(),
                data: None,
            }),
        };

        let validation = validate_jsonrpc_response(&response);
        assert!(validation.is_err());
        assert!(validation.unwrap_err()[0].contains("both"));
    }

    // -------------------------------------------------------------------------
    // Tool Definition Contract Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tool_definition_valid() {
        let tool = McpToolDefinition {
            name: "gigathink".to_string(),
            description: Some("Expansive creative thinking".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
        };

        let validation = validate_tool_definition(&tool);
        assert!(validation.is_ok());
    }

    #[test]
    fn test_tool_definition_invalid_name() {
        let tool = McpToolDefinition {
            name: "GigaThink".to_string(), // Mixed case - invalid
            description: None,
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        };

        let validation = validate_tool_definition(&tool);
        assert!(validation.is_err());
        assert!(validation
            .unwrap_err()
            .iter()
            .any(|e| e.contains("lowercase")));
    }

    #[test]
    fn test_tool_definition_missing_schema_type() {
        let tool = McpToolDefinition {
            name: "test_tool".to_string(),
            description: None,
            input_schema: json!({
                "properties": {}
            }), // Missing type: object
        };

        let validation = validate_tool_definition(&tool);
        assert!(validation.is_err());
    }

    // -------------------------------------------------------------------------
    // Tool Result Contract Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tool_result_text_valid() {
        let result = McpToolResult {
            content: vec![McpContent::Text {
                text: "Analysis complete".to_string(),
            }],
            is_error: None,
        };

        let validation = validate_tool_result(&result);
        assert!(validation.is_ok());
    }

    #[test]
    fn test_tool_result_empty_content() {
        let result = McpToolResult {
            content: vec![],
            is_error: None,
        };

        let validation = validate_tool_result(&result);
        assert!(validation.is_err());
        assert!(validation.unwrap_err()[0].contains("empty"));
    }

    #[test]
    fn test_tool_result_empty_text() {
        let result = McpToolResult {
            content: vec![McpContent::Text {
                text: "".to_string(),
            }],
            is_error: None,
        };

        let validation = validate_tool_result(&result);
        assert!(validation.is_err());
    }

    // -------------------------------------------------------------------------
    // ThinkTool MCP Compliance Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_all_thinktools_mcp_compliant() {
        let llm = MockLlmProvider::new();

        let handlers: Vec<Box<dyn ThinkToolHandler>> = vec![
            Box::new(MockGigaThinkHandler::new(llm.clone())),
            Box::new(MockLaserLogicHandler::new(llm.clone())),
            Box::new(MockBedRockHandler::new(llm.clone())),
            Box::new(MockProofGuardHandler::new(llm.clone())),
            Box::new(MockBrutalHonestyHandler::new(llm.clone())),
        ];

        for handler in handlers {
            // Create MCP tool definition
            let tool_def = McpToolDefinition {
                name: handler.name().to_string(),
                description: Some(handler.description().to_string()),
                input_schema: handler.input_schema(),
            };

            // Validate tool definition
            let def_validation = validate_tool_definition(&tool_def);
            assert!(
                def_validation.is_ok(),
                "Tool {} definition invalid: {:?}",
                handler.name(),
                def_validation.unwrap_err()
            );
        }
    }

    #[tokio::test]
    async fn test_tool_result_mcp_conversion() {
        let llm = MockLlmProvider::new();
        let handler = MockGigaThinkHandler::new(llm);

        let mut input = HashMap::new();
        input.insert("query".to_string(), json!("Test query"));

        let result = handler.execute(input).await.unwrap();

        // Convert ThinkToolResult to MCP format
        let mcp_content: Vec<McpContent> = result
            .content
            .iter()
            .map(|c| match c {
                ThinkToolContent::Text { text } => McpContent::Text { text: text.clone() },
                ThinkToolContent::Reasoning { steps, conclusion } => McpContent::Text {
                    text: format!("Steps:\n{}\n\nConclusion: {}", steps.join("\n"), conclusion),
                },
                ThinkToolContent::Verification { claims } => McpContent::Text {
                    text: claims
                        .iter()
                        .map(|c| {
                            format!(
                                "- {}: {}",
                                c.claim,
                                if c.verified { "verified" } else { "unverified" }
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                },
                ThinkToolContent::Critique { issues, severity } => McpContent::Text {
                    text: format!(
                        "Severity: {}\n{}",
                        severity,
                        issues
                            .iter()
                            .map(|i| format!("- {}", i.issue))
                            .collect::<Vec<_>>()
                            .join("\n")
                    ),
                },
            })
            .collect();

        let mcp_result = McpToolResult {
            content: mcp_content,
            is_error: result.is_error,
        };

        let validation = validate_tool_result(&mcp_result);
        assert!(
            validation.is_ok(),
            "MCP result validation failed: {:?}",
            validation
        );
    }

    // -------------------------------------------------------------------------
    // MCP Request/Response Cycle Contract Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tools_list_request_contract() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::String(Uuid::new_v4().to_string()),
            method: "tools/list".to_string(),
            params: None,
        };

        // Serialize and deserialize to verify structure
        let json = serde_json::to_string(&request).unwrap();
        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.jsonrpc, "2.0");
        assert_eq!(parsed.method, "tools/list");
    }

    #[test]
    fn test_tools_call_request_contract() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(1),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "gigathink",
                "arguments": {
                    "query": "How to improve code quality?"
                }
            })),
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.method, "tools/call");
        assert!(parsed.params.is_some());

        let params = parsed.params.unwrap();
        assert_eq!(params.get("name").unwrap(), "gigathink");
    }

    // -------------------------------------------------------------------------
    // Error Code Contract Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_standard_error_codes() {
        // Verify error codes match JSON-RPC 2.0 spec
        assert_eq!(error_codes::PARSE_ERROR, -32700);
        assert_eq!(error_codes::INVALID_REQUEST, -32600);
        assert_eq!(error_codes::METHOD_NOT_FOUND, -32601);
        assert_eq!(error_codes::INVALID_PARAMS, -32602);
        assert_eq!(error_codes::INTERNAL_ERROR, -32603);
    }

    #[test]
    fn test_mcp_error_codes() {
        // MCP-specific error codes are in reserved range
        assert!(error_codes::REQUEST_CANCELLED < -32000);
        assert!(error_codes::RESOURCE_NOT_FOUND < -32000);
        assert!(error_codes::TOOL_NOT_FOUND < -32000);
        assert!(error_codes::INVALID_TOOL_INPUT < -32000);
    }
}

// =============================================================================
// SECTION 7: TEST FIXTURES AND HELPERS
// =============================================================================

/// Test fixture for creating consistent test environments
pub mod fixtures {
    use super::*;

    /// Creates a standard mock LLM with common responses
    pub async fn standard_mock_llm() -> MockLlmProvider {
        let llm = MockLlmProvider::new();

        llm.register_response(
            "perspective",
            MockLlmResponse {
                content: "Multiple perspectives identified".to_string(),
                tokens_used: 200,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            },
        )
        .await;

        llm.register_response(
            "logic",
            MockLlmResponse {
                content: "Logical analysis complete".to_string(),
                tokens_used: 150,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            },
        )
        .await;

        llm.register_response(
            "decompose",
            MockLlmResponse {
                content: "First principles identified".to_string(),
                tokens_used: 175,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            },
        )
        .await;

        llm.register_response(
            "verify",
            MockLlmResponse {
                content: "Claims verified".to_string(),
                tokens_used: 225,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            },
        )
        .await;

        llm.register_response(
            "critique",
            MockLlmResponse {
                content: "Critical issues identified".to_string(),
                tokens_used: 180,
                model: "mock-gpt-4".to_string(),
                finish_reason: "stop".to_string(),
            },
        )
        .await;

        llm
    }

    /// Creates sample inputs for each ThinkTool
    pub fn sample_inputs() -> HashMap<&'static str, HashMap<String, Value>> {
        let mut samples = HashMap::new();

        let mut gigathink_input = HashMap::new();
        gigathink_input.insert("query".to_string(), json!("How to scale a startup?"));
        gigathink_input.insert("min_perspectives".to_string(), json!(10));
        samples.insert("gigathink", gigathink_input);

        let mut laserlogic_input = HashMap::new();
        laserlogic_input.insert(
            "argument".to_string(),
            json!("If P then Q. P is true. Therefore Q."),
        );
        samples.insert("laserlogic", laserlogic_input);

        let mut bedrock_input = HashMap::new();
        bedrock_input.insert("problem".to_string(), json!("Why do projects fail?"));
        bedrock_input.insert("depth".to_string(), json!(3));
        samples.insert("bedrock", bedrock_input);

        let mut proofguard_input = HashMap::new();
        proofguard_input.insert(
            "claims".to_string(),
            json!(["Rust is memory safe", "Rust has zero-cost abstractions"]),
        );
        samples.insert("proofguard", proofguard_input);

        let mut brutalhonesty_input = HashMap::new();
        brutalhonesty_input.insert(
            "content".to_string(),
            json!("My plan is perfect with no flaws"),
        );
        brutalhonesty_input.insert("severity".to_string(), json!("brutal"));
        samples.insert("brutalhonesty", brutalhonesty_input);

        samples
    }
}

// =============================================================================
// SUMMARY: TEST EXECUTION COMMANDS
// =============================================================================
//
// # Run all MCP tests
// cargo test --test mcp_testing_strategy --release
//
// # Run unit tests only
// cargo test --test mcp_testing_strategy unit_tests --release
//
// # Run integration tests only
// cargo test --test mcp_testing_strategy integration_tests --release
//
// # Run property tests with more cases
// PROPTEST_CASES=5000 cargo test --test mcp_testing_strategy prop_ --release
//
// # Run contract tests only
// cargo test --test mcp_testing_strategy contract_tests --release
//
// # Run with verbose output
// cargo test --test mcp_testing_strategy --release -- --nocapture
//
// # Run mock LLM tests
// cargo test --test mcp_testing_strategy mock_llm_tests --release
//
// =============================================================================
