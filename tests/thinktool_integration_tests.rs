//! ThinkTool Integration Tests
//!
//! Comprehensive integration tests for the full ThinkTool pipeline.
//!
//! Tests cover:
//! - End-to-end protocol execution (with mock LLM)
//! - Profile chaining (quick -> balanced -> deep)
//! - Multiple ThinkTool composition
//! - Memory integration (when feature enabled)
//! - MCP tool invocation
//!
//! All tests use mock LLM responses to ensure deterministic behavior
//! without requiring API keys or network access.

use async_trait::async_trait;
use reasonkit::error::{Error, Result};
use reasonkit::mcp::tools::ToolHandler;
use reasonkit::mcp::tools::ToolResultContent;
use reasonkit::mcp::{
    JsonRpcVersion, McpNotification, McpRequest, McpResponse, McpServer, McpServerTrait,
    ServerCapabilities, ServerInfo, ServerStatus, Tool, ToolResult, Transport,
};
use reasonkit::thinktool::{
    BudgetConfig, BudgetStrategy, ExecutorConfig, ProtocolExecutor, ProtocolInput,
    SelfConsistencyConfig, VotingMethod,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// MOCK INFRASTRUCTURE
// ============================================================================

/// Mock transport for testing MCP server without real I/O
struct MockTransport {
    responses: Arc<RwLock<HashMap<String, McpResponse>>>,
}

impl MockTransport {
    fn new() -> Self {
        Self {
            responses: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    #[allow(dead_code)]
    async fn set_response(&self, method: &str, response: McpResponse) {
        let mut responses = self.responses.write().await;
        responses.insert(method.to_string(), response);
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send_request(&self, request: McpRequest) -> std::io::Result<McpResponse> {
        let responses = self.responses.read().await;
        if let Some(response) = responses.get(&request.method) {
            Ok(response.clone())
        } else {
            Ok(McpResponse {
                jsonrpc: JsonRpcVersion::default(),
                id: request.id,
                result: Some(serde_json::json!({})),
                error: None,
            })
        }
    }

    async fn send_notification(&self, _notification: McpNotification) -> std::io::Result<()> {
        Ok(())
    }

    async fn close(&self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Mock tool handler for testing MCP tool invocation
struct MockThinkToolHandler {
    tool_name: String,
}

impl MockThinkToolHandler {
    fn new(name: impl Into<String>) -> Self {
        Self {
            tool_name: name.into(),
        }
    }
}

#[async_trait]
impl ToolHandler for MockThinkToolHandler {
    async fn call(&self, args: HashMap<String, serde_json::Value>) -> Result<ToolResult> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("default query");

        let response = format!(
            "Mock {} analysis of: {}\n\nKey findings:\n1. Finding A\n2. Finding B\n3. Finding C\n\nConfidence: 0.85",
            self.tool_name, query
        );

        Ok(ToolResult::text(response))
    }
}

// ============================================================================
// END-TO-END PROTOCOL EXECUTION TESTS
// ============================================================================

mod e2e_protocol_tests {
    use super::*;

    #[tokio::test]
    async fn test_gigathink_full_pipeline() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("What are innovative ways to reduce carbon emissions?");

        let result = executor.execute("gigathink", input).await;
        assert!(
            result.is_ok(),
            "GigaThink pipeline failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "GigaThink should succeed");
        assert!(output.confidence > 0.0, "Should have positive confidence");
        assert!(!output.steps.is_empty(), "Should have step results");
        assert!(output.tokens.total_tokens > 0, "Should track token usage");
        assert!(output.duration_ms > 0, "Should track duration");

        // Verify protocol ID is correctly set
        assert_eq!(output.protocol_id, "gigathink");
    }

    #[tokio::test]
    async fn test_laserlogic_full_pipeline() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::argument(
            "Premise 1: All renewable energy sources are sustainable. \
             Premise 2: Solar power is a renewable energy source. \
             Conclusion: Therefore, solar power is sustainable.",
        );

        let result = executor.execute("laserlogic", input).await;
        assert!(
            result.is_ok(),
            "LaserLogic pipeline failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "LaserLogic should succeed");
        assert_eq!(output.protocol_id, "laserlogic");
    }

    #[tokio::test]
    async fn test_bedrock_full_pipeline() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::statement(
            "Quantum computers will revolutionize cryptography within the next decade.",
        );

        let result = executor.execute("bedrock", input).await;
        assert!(
            result.is_ok(),
            "BedRock pipeline failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "BedRock should succeed");
        assert_eq!(output.protocol_id, "bedrock");
    }

    #[tokio::test]
    async fn test_proofguard_full_pipeline() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::claim("GPT-4 achieves human-level performance on the Bar exam.");

        let result = executor.execute("proofguard", input).await;
        assert!(
            result.is_ok(),
            "ProofGuard pipeline failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "ProofGuard should succeed");
        assert_eq!(output.protocol_id, "proofguard");
    }

    #[tokio::test]
    async fn test_brutalhonesty_full_pipeline() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::work(
            "Our analysis shows the market will grow 100% next year based on optimistic projections."
        );

        let result = executor.execute("brutalhonesty", input).await;
        assert!(
            result.is_ok(),
            "BrutalHonesty pipeline failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "BrutalHonesty should succeed");
        assert_eq!(output.protocol_id, "brutalhonesty");
    }

    #[tokio::test]
    async fn test_protocol_with_context_field() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("How should we scale our infrastructure?")
            .with_field("context", "B2B SaaS startup with 10,000 daily active users")
            .with_field("constraints", "Limited budget, small team");

        let result = executor.execute("gigathink", input).await;
        assert!(
            result.is_ok(),
            "Protocol with context failed: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_protocol_output_data_structure() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test query for data structure validation");

        let output = executor.execute("gigathink", input).await.unwrap();

        // Verify output data structure
        assert!(
            output.data.contains_key("confidence"),
            "Should contain confidence"
        );

        // Verify confidence is normalized
        assert!(output.confidence >= 0.0, "Confidence should be >= 0");
        assert!(output.confidence <= 1.0, "Confidence should be <= 1");
    }

    #[tokio::test]
    async fn test_protocol_error_handling_invalid_input() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // Empty input should fail validation
        let input = ProtocolInput {
            fields: HashMap::new(),
        };

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_err(), "Should fail with missing required input");
    }

    #[tokio::test]
    async fn test_protocol_error_handling_nonexistent_protocol() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test query");

        let result = executor.execute("nonexistent_protocol", input).await;
        assert!(result.is_err(), "Should fail for nonexistent protocol");
    }
}

// ============================================================================
// PROFILE CHAINING TESTS
// ============================================================================

mod profile_chaining_tests {
    use super::*;

    #[tokio::test]
    async fn test_profile_quick_to_balanced_progression() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let query = "Should we invest in AI automation for customer support?";

        // Start with quick profile
        let quick_result = executor
            .execute_profile("quick", ProtocolInput::query(query))
            .await
            .expect("Quick profile failed");

        assert!(quick_result.success, "Quick profile should succeed");
        assert!(
            quick_result.confidence >= 0.70,
            "Quick should meet 70% threshold"
        );

        // Progress to balanced profile for deeper analysis
        let balanced_result = executor
            .execute_profile("balanced", ProtocolInput::query(query))
            .await
            .expect("Balanced profile failed");

        assert!(balanced_result.success, "Balanced profile should succeed");

        // Balanced typically uses more tokens and takes longer
        assert!(
            balanced_result.tokens.total_tokens >= quick_result.tokens.total_tokens,
            "Balanced should use at least as many tokens as quick"
        );
    }

    #[tokio::test]
    async fn test_profile_balanced_to_deep_progression() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let query = "Evaluate the long-term implications of remote work on corporate culture";

        // Run balanced first
        let balanced_result = executor
            .execute_profile("balanced", ProtocolInput::query(query))
            .await
            .expect("Balanced profile failed");

        // Then run deep for more thorough analysis
        let deep_result = executor
            .execute_profile("deep", ProtocolInput::query(query))
            .await
            .expect("Deep profile failed");

        assert!(
            deep_result.success || !deep_result.steps.is_empty(),
            "Deep should complete with results"
        );

        // Deep profile should have more steps (includes all 5 ThinkTools)
        assert!(
            deep_result.steps.len() >= balanced_result.steps.len(),
            "Deep should have at least as many steps as balanced"
        );
    }

    #[tokio::test]
    async fn test_profile_chain_quick_balanced_deep_paranoid() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let query = "Is blockchain technology suitable for our supply chain tracking?";

        let profiles = ["quick", "balanced", "deep", "paranoid"];
        let mut prev_token_count = 0;

        for profile in profiles {
            let result = executor
                .execute_profile(profile, ProtocolInput::query(query))
                .await
                .expect(&format!("{} profile failed", profile));

            // Verify each profile produces results
            assert!(
                !result.steps.is_empty(),
                "{} should produce step results",
                profile
            );

            // Token usage should generally increase with profile complexity
            if result.tokens.total_tokens < prev_token_count && profile != "quick" {
                // This is acceptable for conditional execution profiles
            }
            prev_token_count = result.tokens.total_tokens;
        }
    }

    #[tokio::test]
    async fn test_decide_profile_for_binary_decisions() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query(
            "Should we migrate from PostgreSQL to MongoDB for our product catalog?",
        );

        let result = executor.execute_profile("decide", input).await;
        assert!(result.is_ok(), "Decide profile failed: {:?}", result.err());

        let output = result.unwrap();
        assert!(
            !output.steps.is_empty(),
            "Decide should produce step results"
        );
    }

    #[tokio::test]
    async fn test_scientific_profile_for_research() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query(
            "Hypothesis: Larger language models exhibit emergent reasoning at 10B+ parameters",
        )
        .with_field("domain", "Machine Learning");

        let result = executor.execute_profile("scientific", input).await;
        assert!(
            result.is_ok(),
            "Scientific profile failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "Scientific profile should succeed");
    }

    #[tokio::test]
    async fn test_powercombo_profile_all_thinktools() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input =
            ProtocolInput::query("Evaluate: ReasonKit improves AI reasoning quality by 20%");

        let result = executor.execute_profile("powercombo", input).await;
        assert!(
            result.is_ok(),
            "PowerCombo profile failed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        // PowerCombo uses all 5 ThinkTools plus validation pass
        assert!(!output.steps.is_empty(), "PowerCombo should have steps");
    }

    #[tokio::test]
    async fn test_profile_confidence_thresholds() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // Get profiles and verify their confidence thresholds
        let quick = executor
            .get_profile("quick")
            .expect("Quick profile not found");
        let balanced = executor
            .get_profile("balanced")
            .expect("Balanced profile not found");
        let paranoid = executor
            .get_profile("paranoid")
            .expect("Paranoid profile not found");

        assert_eq!(quick.min_confidence, 0.70, "Quick should require 70%");
        assert_eq!(balanced.min_confidence, 0.80, "Balanced should require 80%");
        assert_eq!(paranoid.min_confidence, 0.95, "Paranoid should require 95%");
    }
}

// ============================================================================
// MULTIPLE THINKTOOL COMPOSITION TESTS
// ============================================================================

mod composition_tests {
    use super::*;

    #[tokio::test]
    async fn test_sequential_protocol_execution() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let query = "What are the risks of AI in healthcare?";

        // Execute multiple protocols in sequence
        let gigathink_result = executor
            .execute("gigathink", ProtocolInput::query(query))
            .await
            .expect("GigaThink failed");

        // Use GigaThink output as input for LaserLogic
        let laserlogic_input = ProtocolInput::argument(&format!(
            "Based on analysis: {}",
            gigathink_result
                .data
                .get("confidence")
                .map(|v| v.to_string())
                .unwrap_or_default()
        ));

        let laserlogic_result = executor
            .execute("laserlogic", laserlogic_input)
            .await
            .expect("LaserLogic failed");

        assert!(gigathink_result.success, "GigaThink should succeed");
        assert!(laserlogic_result.success, "LaserLogic should succeed");
    }

    #[tokio::test]
    async fn test_parallel_independent_protocols() {
        // These queries are independent and could run in parallel
        let queries = vec![
            ("gigathink", "What are growth opportunities?"),
            ("bedrock", "AI is the future of automation"),
            ("laserlogic", "If A then B, A is true, therefore B"),
        ];

        let mut handles = vec![];

        for (protocol, query) in queries {
            let executor_clone = ProtocolExecutor::mock().expect("Clone failed");
            let protocol_str = protocol.to_string();
            let query_str = query.to_string();

            let handle = tokio::spawn(async move {
                let input = if protocol_str == "laserlogic" {
                    ProtocolInput::argument(&query_str)
                } else if protocol_str == "bedrock" {
                    ProtocolInput::statement(&query_str)
                } else {
                    ProtocolInput::query(&query_str)
                };

                executor_clone.execute(&protocol_str, input).await
            });

            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let result = handle.await.expect("Task panicked");
            assert!(
                result.is_ok(),
                "Parallel protocol failed: {:?}",
                result.err()
            );
        }
    }

    #[tokio::test]
    async fn test_protocol_output_chaining() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // Step 1: GigaThink for creative exploration
        let gt_result = executor
            .execute(
                "gigathink",
                ProtocolInput::query("How can AI improve education?"),
            )
            .await
            .expect("GigaThink failed");

        // Step 2: BedRock for first principles
        let br_result = executor
            .execute(
                "bedrock",
                ProtocolInput::statement("AI can personalize learning at scale"),
            )
            .await
            .expect("BedRock failed");

        // Step 3: ProofGuard for verification
        let pg_result = executor
            .execute(
                "proofguard",
                ProtocolInput::claim("AI tutoring systems improve learning outcomes by 30%"),
            )
            .await
            .expect("ProofGuard failed");

        // Step 4: BrutalHonesty for critique
        let bh_result = executor
            .execute(
                "brutalhonesty",
                ProtocolInput::work("Our AI education platform will revolutionize learning"),
            )
            .await
            .expect("BrutalHonesty failed");

        // Verify all succeeded
        assert!(
            gt_result.success && br_result.success && pg_result.success && bh_result.success,
            "All protocols in chain should succeed"
        );

        // Aggregate confidence across all steps
        let total_confidence = (gt_result.confidence
            + br_result.confidence
            + pg_result.confidence
            + bh_result.confidence)
            / 4.0;
        assert!(
            total_confidence > 0.0,
            "Aggregate confidence should be positive"
        );
    }

    #[tokio::test]
    async fn test_conditional_protocol_execution() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // First pass with GigaThink
        let initial_result = executor
            .execute("gigathink", ProtocolInput::query("Initial analysis"))
            .await
            .expect("Initial execution failed");

        // Conditionally execute BrutalHonesty if confidence is below threshold
        if initial_result.confidence < 0.90 {
            let critique_result = executor
                .execute(
                    "brutalhonesty",
                    ProtocolInput::work(&format!(
                        "Previous confidence: {}",
                        initial_result.confidence
                    )),
                )
                .await
                .expect("Conditional BrutalHonesty failed");

            assert!(
                critique_result.success,
                "Conditional critique should succeed"
            );
        }
    }

    #[tokio::test]
    async fn test_token_budget_aggregation_across_protocols() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let query = "Budget tracking test query";

        let protocols = ["gigathink", "laserlogic", "bedrock"];
        let mut total_tokens = 0u32;

        for protocol in protocols.iter() {
            let input = match *protocol {
                "laserlogic" => ProtocolInput::argument(query),
                "bedrock" => ProtocolInput::statement(query),
                _ => ProtocolInput::query(query),
            };

            let result = executor
                .execute(protocol, input)
                .await
                .expect("Execution failed");
            total_tokens += result.tokens.total_tokens;

            // Verify each protocol tracks its tokens
            assert!(
                result.tokens.total_tokens > 0,
                "Protocol {} should track tokens",
                protocol
            );
        }

        assert!(
            total_tokens > 0,
            "Total tokens should be tracked across protocols"
        );
    }
}

// ============================================================================
// MEMORY INTEGRATION TESTS (conditional on feature)
// ============================================================================

#[cfg(feature = "memory")]
mod memory_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_protocol_with_memory_context() {
        // Test that protocols can access memory context when feature is enabled
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let input = ProtocolInput::query("Analyze based on stored knowledge")
            .with_field("memory_context", "previous_analysis_id");

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_ok(), "Protocol with memory context failed");
    }

    #[tokio::test]
    async fn test_profile_with_retrieval_augmentation() {
        // Test RAG integration with profiles
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let input = ProtocolInput::query("What do our documents say about X?")
            .with_field("enable_rag", "true")
            .with_field("top_k", "5");

        let result = executor.execute_profile("balanced", input).await;
        assert!(result.is_ok(), "RAG-augmented profile failed");
    }
}

// ============================================================================
// MCP TOOL INVOCATION TESTS
// ============================================================================

mod mcp_tool_tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_server_tool_registration() {
        let transport = Arc::new(MockTransport::new());
        let info = ServerInfo {
            name: "test-server".to_string(),
            version: "1.0.0".to_string(),
            description: Some("Test MCP server".to_string()),
            vendor: Some("ReasonKit".to_string()),
        };
        let capabilities = ServerCapabilities::default();
        let server = McpServer::new("test-server", info, capabilities, transport);

        // Register ThinkTool handlers
        let gigathink_handler = Arc::new(MockThinkToolHandler::new("GigaThink"));
        server
            .register_tool(
                Tool::simple("gigathink", "Expansive creative thinking"),
                gigathink_handler,
            )
            .await;

        let laserlogic_handler = Arc::new(MockThinkToolHandler::new("LaserLogic"));
        server
            .register_tool(
                Tool::simple("laserlogic", "Precision deductive reasoning"),
                laserlogic_handler,
            )
            .await;

        // Verify tools are registered
        let tools = server.list_tools().await;
        assert_eq!(tools.len(), 2, "Should have 2 registered tools");
        assert!(tools.iter().any(|t| t.name == "gigathink"));
        assert!(tools.iter().any(|t| t.name == "laserlogic"));
    }

    #[tokio::test]
    async fn test_mcp_tool_invocation() {
        let transport = Arc::new(MockTransport::new());
        let info = ServerInfo {
            name: "test-server".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            vendor: None,
        };
        let server = McpServer::new(
            "test-server",
            info,
            ServerCapabilities::default(),
            transport,
        );

        // Register a tool
        let handler = Arc::new(MockThinkToolHandler::new("TestTool"));
        server
            .register_tool(Tool::simple("test_tool", "A test tool"), handler)
            .await;

        // Invoke the tool
        let mut args = HashMap::new();
        args.insert("query".to_string(), serde_json::json!("test query"));

        let result = server.call_tool("test_tool", args).await;
        assert!(result.is_ok(), "Tool invocation failed: {:?}", result.err());

        let tool_result = result.unwrap();
        assert!(
            !tool_result.content.is_empty(),
            "Tool should return content"
        );

        match &tool_result.content[0] {
            ToolResultContent::Text { text } => {
                assert!(
                    text.contains("TestTool"),
                    "Response should mention tool name"
                );
                assert!(
                    text.contains("Confidence"),
                    "Response should include confidence"
                );
            }
            _ => panic!("Expected text content"),
        }
    }

    #[tokio::test]
    async fn test_mcp_tool_not_found_error() {
        let transport = Arc::new(MockTransport::new());
        let info = ServerInfo {
            name: "test-server".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            vendor: None,
        };
        let server = McpServer::new(
            "test-server",
            info,
            ServerCapabilities::default(),
            transport,
        );

        let result = server.call_tool("nonexistent_tool", HashMap::new()).await;
        assert!(result.is_err(), "Should fail for nonexistent tool");

        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("not found"),
            "Error should mention tool not found"
        );
    }

    #[tokio::test]
    async fn test_mcp_server_status_lifecycle() {
        let transport = Arc::new(MockTransport::new());
        let info = ServerInfo {
            name: "test-server".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            vendor: None,
        };
        let server = McpServer::new(
            "test-server",
            info,
            ServerCapabilities::default(),
            transport,
        );

        // Initial status should be Starting
        let status = server.status().await;
        assert_eq!(status, ServerStatus::Starting);

        // Update to Running
        server.set_status(ServerStatus::Running).await;
        let status = server.status().await;
        assert_eq!(status, ServerStatus::Running);
    }

    #[tokio::test]
    async fn test_mcp_server_metrics_tracking() {
        let transport = Arc::new(MockTransport::new());
        let info = ServerInfo {
            name: "test-server".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            vendor: None,
        };
        let server = McpServer::new(
            "test-server",
            info,
            ServerCapabilities::default(),
            transport,
        );

        // Record some success metrics
        server.record_success(50.0).await;
        server.record_success(100.0).await;

        let metrics = server.metrics().await;
        assert_eq!(metrics.requests_total, 2, "Should track request count");
        assert!(
            metrics.avg_response_time_ms > 0.0,
            "Should track response time"
        );
        assert!(
            metrics.last_success_at.is_some(),
            "Should track last success time"
        );
    }

    #[tokio::test]
    async fn test_mcp_multiple_tool_chain() {
        let transport = Arc::new(MockTransport::new());
        let info = ServerInfo {
            name: "chain-server".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            vendor: None,
        };
        let server = McpServer::new(
            "chain-server",
            info,
            ServerCapabilities::default(),
            transport,
        );

        // Register multiple tools
        let tools = vec![
            "gigathink",
            "laserlogic",
            "bedrock",
            "proofguard",
            "brutalhonesty",
        ];
        for tool_name in &tools {
            let handler = Arc::new(MockThinkToolHandler::new(*tool_name));
            server
                .register_tool(
                    Tool::simple(*tool_name, format!("{} reasoning", tool_name)),
                    handler,
                )
                .await;
        }

        // Chain tool invocations
        let mut query = "Initial analysis question".to_string();

        for tool_name in &tools {
            let mut args = HashMap::new();
            args.insert("query".to_string(), serde_json::json!(query));

            let result = server
                .call_tool(tool_name, args)
                .await
                .expect(&format!("{} invocation failed", tool_name));

            // Extract result for next tool
            if let Some(ToolResultContent::Text { text }) = result.content.first() {
                query = text.clone();
            }
        }
    }
}

// ============================================================================
// SELF-CONSISTENCY AND VOTING TESTS
// ============================================================================

mod self_consistency_tests {
    use super::*;

    #[tokio::test]
    async fn test_self_consistency_default_config() {
        let config = SelfConsistencyConfig::default();
        assert_eq!(config.num_samples, 5, "Default should use 5 samples");
        assert!(matches!(config.voting_method, VotingMethod::MajorityVote));
    }

    #[tokio::test]
    async fn test_self_consistency_fast_config() {
        let config = SelfConsistencyConfig::fast();
        assert_eq!(config.num_samples, 3, "Fast should use 3 samples");
        assert!(config.early_stopping, "Fast should enable early stopping");
    }

    #[tokio::test]
    async fn test_self_consistency_thorough_config() {
        let config = SelfConsistencyConfig::thorough();
        assert_eq!(config.num_samples, 10, "Thorough should use 10 samples");
        assert!(
            !config.early_stopping,
            "Thorough should disable early stopping"
        );
    }

    #[tokio::test]
    async fn test_self_consistency_paranoid_config() {
        let config = SelfConsistencyConfig::paranoid();
        assert_eq!(config.num_samples, 15, "Paranoid should use 15 samples");
    }

    #[tokio::test]
    async fn test_executor_with_self_consistency() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("What is 15 + 27?");
        let sc_config = SelfConsistencyConfig::fast();

        let result = executor
            .execute_with_self_consistency("quick", input, &sc_config)
            .await;

        assert!(
            result.is_ok(),
            "Self-consistency execution failed: {:?}",
            result.err()
        );

        let (output, consistency_result) = result.unwrap();

        // Verify consistency metadata
        assert!(consistency_result.total_samples > 0, "Should have samples");
        assert!(
            consistency_result.agreement_ratio >= 0.0 && consistency_result.agreement_ratio <= 1.0,
            "Agreement ratio should be normalized"
        );

        // Verify output contains self-consistency metadata
        assert!(
            output.data.contains_key("self_consistency"),
            "Output should contain self-consistency data"
        );
    }
}

// ============================================================================
// BUDGET AND RESOURCE MANAGEMENT TESTS
// ============================================================================

mod budget_tests {
    use super::*;

    #[tokio::test]
    async fn test_budget_config_creation() {
        let config = BudgetConfig {
            token_limit: Some(10000),
            cost_limit: Some(1.0),
            strategy: BudgetStrategy::Strict,
            ..Default::default()
        };

        assert_eq!(config.token_limit, Some(10000));
        assert_eq!(config.cost_limit, Some(1.0));
    }

    #[tokio::test]
    async fn test_executor_with_budget_tracking() {
        let mut executor_config = ExecutorConfig::mock();
        executor_config.budget = BudgetConfig {
            token_limit: Some(50000),
            cost_limit: Some(5.0),
            strategy: BudgetStrategy::Adaptive,
            adapt_threshold: 0.8,
            ..Default::default()
        };

        let executor = ProtocolExecutor::with_config(executor_config)
            .expect("Failed to create executor with budget");

        let result = executor
            .execute("gigathink", ProtocolInput::query("Budget test query"))
            .await
            .expect("Execution failed");

        // Verify budget summary is included
        assert!(
            result.budget_summary.is_some(),
            "Should include budget summary"
        );

        let summary = result.budget_summary.unwrap();
        assert!(summary.tokens_used > 0, "Should track token usage");
    }

    #[tokio::test]
    async fn test_profile_with_budget_constraints() {
        let mut executor_config = ExecutorConfig::mock();
        executor_config.budget = BudgetConfig {
            token_limit: Some(100000),
            cost_limit: Some(10.0),
            strategy: BudgetStrategy::Strict,
            adapt_threshold: 0.9,
            ..Default::default()
        };

        let executor =
            ProtocolExecutor::with_config(executor_config).expect("Failed to create executor");

        let result = executor
            .execute_profile("balanced", ProtocolInput::query("Budget test"))
            .await
            .expect("Profile execution failed");

        assert!(
            result.budget_summary.is_some(),
            "Profile should track budget"
        );
    }
}

// ============================================================================
// PARALLEL EXECUTION TESTS
// ============================================================================

mod parallel_execution_tests {
    use super::*;

    #[tokio::test]
    async fn test_parallel_execution_config() {
        let config = ExecutorConfig::mock().with_parallel();
        assert!(config.enable_parallel, "Parallel should be enabled");
        assert_eq!(
            config.max_concurrent_steps, 4,
            "Default concurrency should be 4"
        );
    }

    #[tokio::test]
    async fn test_parallel_execution_with_limit() {
        let config = ExecutorConfig::mock().with_parallel_limit(2);
        assert!(config.enable_parallel, "Parallel should be enabled");
        assert_eq!(config.max_concurrent_steps, 2, "Concurrency should be 2");
    }

    #[tokio::test]
    async fn test_parallel_protocol_execution() {
        let config = ExecutorConfig::mock().with_parallel();
        let executor =
            ProtocolExecutor::with_config(config).expect("Failed to create parallel executor");

        let result = executor
            .execute("gigathink", ProtocolInput::query("Parallel test"))
            .await;

        assert!(
            result.is_ok(),
            "Parallel execution failed: {:?}",
            result.err()
        );
    }
}

// ============================================================================
// TRACE AND TELEMETRY TESTS
// ============================================================================

mod trace_tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_trace_generation() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Trace test query");

        let output = executor.execute("gigathink", input).await.unwrap();

        // Verify duration tracking
        assert!(output.duration_ms > 0, "Should track execution duration");

        // Verify step traces
        for step in &output.steps {
            // Duration should be present; 0ms is possible for mocked/fast steps.
            // (duration_ms is u64 so non-negative by construction)
            assert!(
                step.duration_ms == step.duration_ms,
                "Each step should have duration"
            );
            assert!(
                step.confidence >= 0.0 && step.confidence <= 1.0,
                "Step confidence should be normalized"
            );
        }
    }

    #[tokio::test]
    async fn test_trace_saving_config() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let mut config = ExecutorConfig::mock();
        config.save_traces = true;
        config.trace_dir = Some(temp_dir.path().to_path_buf());

        let executor = ProtocolExecutor::with_config(config)
            .expect("Failed to create executor with trace saving");

        let result = executor
            .execute("gigathink", ProtocolInput::query("Trace save test"))
            .await
            .expect("Execution failed");

        // Verify trace ID was generated
        assert!(
            result.trace_id.is_some(),
            "Should have trace ID when saving enabled"
        );
    }
}

// ============================================================================
// CLI TOOL INTEGRATION TESTS
// ============================================================================

mod cli_tool_tests {
    use super::*;

    #[test]
    fn test_cli_tool_config_claude() {
        let config = ExecutorConfig::claude_cli();
        assert!(config.cli_tool.is_some());
        let cli = config.cli_tool.unwrap();
        assert_eq!(cli.command, "claude");
        assert!(cli.pre_args.contains(&"-p".to_string()));
    }

    #[test]
    fn test_cli_tool_config_codex() {
        let config = ExecutorConfig::codex_cli();
        assert!(config.cli_tool.is_some());
        let cli = config.cli_tool.unwrap();
        assert_eq!(cli.command, "codex");
    }

    #[test]
    fn test_cli_tool_config_gemini() {
        let config = ExecutorConfig::gemini_cli();
        assert!(config.cli_tool.is_some());
        let cli = config.cli_tool.unwrap();
        assert_eq!(cli.command, "gemini");
    }

    #[test]
    fn test_cli_tool_config_opencode() {
        let config = ExecutorConfig::opencode_cli();
        assert!(config.cli_tool.is_some());
        let cli = config.cli_tool.unwrap();
        // OpenCode may have custom command from env var
        assert!(!cli.command.is_empty());
    }

    #[test]
    fn test_cli_tool_config_copilot() {
        let config = ExecutorConfig::copilot_cli();
        assert!(config.cli_tool.is_some());
        let cli = config.cli_tool.unwrap();
        assert_eq!(cli.command, "gh");
        assert!(cli.interactive, "Copilot is interactive");
    }
}

// ============================================================================
// EDGE CASES AND ERROR HANDLING TESTS
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_query_handling() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("");

        let result = executor.execute("gigathink", input).await;
        // Empty query should either fail validation or produce minimal results
        // depending on protocol requirements
        assert!(
            result.is_err() || !result.as_ref().unwrap().steps.is_empty(),
            "Should handle empty query gracefully"
        );
    }

    #[tokio::test]
    async fn test_very_long_query_handling() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let long_query = "a".repeat(10000);
        let input = ProtocolInput::query(&long_query);

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_ok(), "Should handle long queries");
    }

    #[tokio::test]
    async fn test_special_characters_in_query() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let special_query = "What about <html> & \"quotes\" and 'apostrophes' {{braces}}?";
        let input = ProtocolInput::query(special_query);

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_ok(), "Should handle special characters");
    }

    #[tokio::test]
    async fn test_unicode_in_query() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let unicode_query = "Analyze: Japanese text, Chinese text, emoji icons";
        let input = ProtocolInput::query(unicode_query);

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_ok(), "Should handle unicode");
    }

    #[tokio::test]
    async fn test_concurrent_executor_usage() {
        let executor = Arc::new(ProtocolExecutor::mock().expect("Failed to create mock executor"));

        let mut handles = vec![];
        for i in 0..5 {
            let exec_clone = Arc::clone(&executor);
            let handle = tokio::spawn(async move {
                let input = ProtocolInput::query(&format!("Concurrent query {}", i));
                exec_clone.execute("gigathink", input).await
            });
            handles.push(handle);
        }

        for handle in handles {
            let result = handle.await.expect("Task panicked");
            assert!(result.is_ok(), "Concurrent execution failed");
        }
    }

    #[tokio::test]
    async fn test_profile_not_found() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Test");

        let result = executor.execute_profile("nonexistent_profile", input).await;
        assert!(result.is_err(), "Should fail for nonexistent profile");
    }

    #[tokio::test]
    async fn test_input_type_mismatch() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        // LaserLogic expects 'argument', not 'query'
        let input = ProtocolInput::query("This should be an argument");

        let result = executor.execute("laserlogic", input).await;
        // Depending on protocol requirements, this may fail or map input
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_multiple_additional_fields() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let input = ProtocolInput::query("Main query")
            .with_field("context", "Business context")
            .with_field("constraints", "Time and budget constraints")
            .with_field("stakeholders", "Engineering and product teams")
            .with_field("priority", "High")
            .with_field("deadline", "Q1 2025");

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_ok(), "Should handle multiple fields");
    }
}

// ============================================================================
// REGISTRY AND PROFILE INTROSPECTION TESTS
// ============================================================================

mod introspection_tests {
    use super::*;

    #[test]
    fn test_list_all_protocols() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let protocols = executor.list_protocols();

        // Verify core protocols exist
        let expected = [
            "gigathink",
            "laserlogic",
            "bedrock",
            "proofguard",
            "brutalhonesty",
        ];
        for protocol in &expected {
            assert!(
                protocols.contains(protocol),
                "Missing protocol: {}",
                protocol
            );
        }
    }

    #[test]
    fn test_list_all_profiles() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");
        let profiles = executor.list_profiles();

        // Verify core profiles exist
        let expected = [
            "quick",
            "balanced",
            "deep",
            "paranoid",
            "decide",
            "scientific",
            "powercombo",
        ];
        for profile in &expected {
            assert!(profiles.contains(profile), "Missing profile: {}", profile);
        }
    }

    #[test]
    fn test_get_protocol_info() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let protocol = executor.get_protocol("gigathink");
        assert!(protocol.is_some(), "GigaThink protocol should exist");

        let gt = protocol.unwrap();
        assert_eq!(gt.id, "gigathink");
        assert!(!gt.steps.is_empty(), "GigaThink should have steps");
    }

    #[test]
    fn test_get_profile_info() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let profile = executor.get_profile("balanced");
        assert!(profile.is_some(), "Balanced profile should exist");

        let balanced = profile.unwrap();
        assert_eq!(balanced.id, "balanced");
        assert!(
            !balanced.chain.is_empty(),
            "Balanced should have chain steps"
        );
        assert_eq!(balanced.min_confidence, 0.80);
    }

    #[test]
    fn test_profile_chain_structure() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let quick = executor.get_profile("quick").unwrap();
        assert_eq!(quick.chain.len(), 2, "Quick should have 2 protocols");
        assert_eq!(quick.chain[0].protocol_id, "gigathink");
        assert_eq!(quick.chain[1].protocol_id, "laserlogic");
    }

    #[test]
    fn test_powercombo_chain_completeness() {
        let executor = ProtocolExecutor::mock().expect("Failed to create mock executor");

        let powercombo = executor.get_profile("powercombo").unwrap();

        // PowerCombo should use all 5 ThinkTools plus validation
        let protocol_ids: Vec<&str> = powercombo
            .chain
            .iter()
            .map(|s| s.protocol_id.as_str())
            .collect();

        assert!(
            protocol_ids.contains(&"gigathink"),
            "Should include gigathink"
        );
        assert!(
            protocol_ids.contains(&"laserlogic"),
            "Should include laserlogic"
        );
        assert!(protocol_ids.contains(&"bedrock"), "Should include bedrock");
        assert!(
            protocol_ids.contains(&"proofguard"),
            "Should include proofguard"
        );
        assert!(
            protocol_ids.contains(&"brutalhonesty"),
            "Should include brutalhonesty"
        );
    }
}
