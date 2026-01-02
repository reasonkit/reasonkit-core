//! Cross-Crate Integration Tests
//!
//! Integration tests for cross-crate interactions between:
//! - reasonkit-core: ThinkTool execution engine
//! - reasonkit-mem: Memory/storage layer
//! - reasonkit-web: MCP endpoint server
//!
//! # Test Philosophy
//!
//! These tests validate the integration boundaries between crates,
//! ensuring that:
//! 1. Core can properly delegate to mem for storage/retrieval
//! 2. Core can communicate with web via MCP protocol
//! 3. Error conditions propagate correctly across boundaries
//! 4. Feature flags work correctly across crates
//!
//! # Running Tests
//!
//! ```bash
//! # Run all cross-crate tests
//! cargo test --package reasonkit-core --test cross_crate_integration_tests
//!
//! # Run with memory feature enabled
//! cargo test --package reasonkit-core --test cross_crate_integration_tests --features memory
//! ```

// ============================================================================
// MODULE: Core ThinkTool Execution Tests
// ============================================================================

mod thinktool_execution_tests {
    //! Tests for ThinkTool protocol execution

    /// Test: ProtocolExecutor initializes correctly with mock LLM
    #[tokio::test]
    async fn test_protocol_executor_mock_initialization() {
        use reasonkit::thinktool::ProtocolExecutor;

        let executor = ProtocolExecutor::mock();
        assert!(
            executor.is_ok(),
            "Mock executor should initialize: {:?}",
            executor.err()
        );

        let executor = executor.unwrap();
        // Verify executor has loaded default protocols
        let registry = executor.registry();
        assert!(
            registry.get("gigathink").is_some(),
            "Should have gigathink protocol"
        );
        assert!(
            registry.get("laserlogic").is_some(),
            "Should have laserlogic protocol"
        );
        assert!(
            registry.get("bedrock").is_some(),
            "Should have bedrock protocol"
        );
    }

    /// Test: Execute GigaThink protocol end-to-end with mock LLM
    #[tokio::test]
    async fn test_gigathink_execution_e2e() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");

        let input = ProtocolInput::query("What are the key factors for successful AI adoption?");
        let result = executor.execute("gigathink", input).await;

        assert!(
            result.is_ok(),
            "GigaThink should execute successfully: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "Execution should succeed");
        assert!(
            output.confidence >= 0.0 && output.confidence <= 1.0,
            "Confidence should be in [0, 1]"
        );
        assert!(!output.steps.is_empty(), "Should have step results");
        assert_eq!(output.protocol_id, "gigathink", "Protocol ID should match");
    }

    /// Test: Execute LaserLogic for argument validation
    #[tokio::test]
    async fn test_laserlogic_argument_validation() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");

        let input = ProtocolInput::argument(
            "Premise 1: All humans are mortal. \
             Premise 2: Socrates is a human. \
             Conclusion: Therefore, Socrates is mortal.",
        );
        let result = executor.execute("laserlogic", input).await;

        assert!(
            result.is_ok(),
            "LaserLogic should execute: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "Execution should succeed");
        assert_eq!(output.protocol_id, "laserlogic");
    }

    /// Test: Execute BedRock for first principles decomposition
    #[tokio::test]
    async fn test_bedrock_first_principles() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");

        let input = ProtocolInput::statement(
            "What are the fundamental principles of building scalable systems?",
        );
        let result = executor.execute("bedrock", input).await;

        assert!(result.is_ok(), "BedRock should execute: {:?}", result.err());

        let output = result.unwrap();
        assert!(output.success, "Execution should succeed");
    }

    /// Test: Execute ProofGuard for multi-source verification
    #[tokio::test]
    async fn test_proofguard_verification() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");

        let input = ProtocolInput::claim("Neural networks were inspired by biological neurons");
        let result = executor.execute("proofguard", input).await;

        assert!(
            result.is_ok(),
            "ProofGuard should execute: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "Execution should succeed");
    }

    /// Test: Execute BrutalHonesty for adversarial critique
    #[tokio::test]
    async fn test_brutalhonesty_critique() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");

        let input = ProtocolInput::work(
            "Our startup will definitely succeed because we have a great product",
        );
        let result = executor.execute("brutalhonesty", input).await;

        assert!(
            result.is_ok(),
            "BrutalHonesty should execute: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(output.success, "Execution should succeed");
    }

    /// Test: Profile chaining (quick -> balanced -> deep)
    #[tokio::test]
    async fn test_profile_chaining() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");
        let query = "Should we migrate to microservices architecture?";

        // Test quick profile
        let quick_result = executor
            .execute_profile("quick", ProtocolInput::query(query))
            .await;
        assert!(
            quick_result.is_ok(),
            "Quick profile should execute: {:?}",
            quick_result.err()
        );

        // Test balanced profile
        let balanced_result = executor
            .execute_profile("balanced", ProtocolInput::query(query))
            .await;
        assert!(
            balanced_result.is_ok(),
            "Balanced profile should execute: {:?}",
            balanced_result.err()
        );

        // Test deep profile
        let deep_result = executor
            .execute_profile("deep", ProtocolInput::query(query))
            .await;
        assert!(
            deep_result.is_ok(),
            "Deep profile should execute: {:?}",
            deep_result.err()
        );
    }

    /// Test: Parallel step execution
    #[tokio::test]
    async fn test_parallel_step_execution() {
        use reasonkit::thinktool::{
            ExecutorConfig, LlmConfig, LlmProvider, ProtocolExecutor, ProtocolInput,
        };

        let config = ExecutorConfig {
            llm: LlmConfig::for_provider(LlmProvider::OpenAI, "mock-model"),
            enable_parallel: true,
            max_concurrent_steps: 4,
            use_mock: true,
            show_progress: false,
            ..Default::default()
        };

        let executor = ProtocolExecutor::with_config(config).expect("Executor creation failed");
        let input = ProtocolInput::query("Test parallel execution");

        let result = executor.execute("gigathink", input).await;
        assert!(
            result.is_ok(),
            "Parallel execution should work: {:?}",
            result.err()
        );
    }

    /// Test: Budget constraints during execution
    #[tokio::test]
    async fn test_budget_constrained_execution() {
        use reasonkit::thinktool::{
            BudgetConfig, BudgetStrategy, ExecutorConfig, LlmConfig, LlmProvider, ProtocolExecutor,
            ProtocolInput,
        };

        let config = ExecutorConfig {
            llm: LlmConfig::for_provider(LlmProvider::OpenAI, "mock-model"),
            budget: BudgetConfig {
                token_limit: Some(10000),
                cost_limit: Some(1.00),
                strategy: BudgetStrategy::Adaptive,
                adapt_threshold: 0.75,
                ..Default::default()
            },
            use_mock: true,
            show_progress: false,
            ..Default::default()
        };

        let executor = ProtocolExecutor::with_config(config).expect("Executor creation failed");
        let input = ProtocolInput::query("Test budget constraints");

        let result = executor.execute("gigathink", input).await;
        assert!(
            result.is_ok(),
            "Budget-constrained execution should work: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert!(
            output.tokens.total_tokens <= 10000,
            "Should respect token budget"
        );
    }
}

// ============================================================================
// MODULE: Error Case Tests
// ============================================================================

mod error_case_tests {
    //! Tests for error conditions and error propagation

    /// Test: Unknown protocol returns error
    #[tokio::test]
    async fn test_unknown_protocol_error() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");
        let input = ProtocolInput::query("Test query");

        let result = executor.execute("nonexistent_protocol", input).await;
        assert!(result.is_err(), "Unknown protocol should return error");

        let err = result.err().unwrap();
        assert!(
            err.to_string().contains("not found") || err.to_string().contains("unknown"),
            "Error should indicate unknown protocol: {}",
            err
        );
    }

    /// Test: Empty query handling
    #[tokio::test]
    async fn test_empty_query_handling() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");
        let input = ProtocolInput::query("");

        let result = executor.execute("gigathink", input).await;
        // Empty query might succeed (mock) or fail - just ensure no panic
        match result {
            Ok(output) => {
                // If it succeeds, verify it handled gracefully
                assert!(!output.steps.is_empty() || !output.success);
            }
            Err(e) => {
                // If it fails, should be a validation error
                let err_str = e.to_string().to_lowercase();
                assert!(
                    err_str.contains("empty")
                        || err_str.contains("query")
                        || err_str.contains("required"),
                    "Error should indicate empty query issue: {}",
                    e
                );
            }
        }
    }

    /// Test: Timeout handling
    #[tokio::test]
    async fn test_timeout_handling() {
        use reasonkit::thinktool::{
            ExecutorConfig, LlmConfig, LlmProvider, ProtocolExecutor, ProtocolInput,
        };

        let config = ExecutorConfig {
            llm: LlmConfig::for_provider(LlmProvider::OpenAI, "mock-model"),
            timeout_secs: 1, // Very short timeout
            use_mock: true,
            show_progress: false,
            ..Default::default()
        };

        let executor = ProtocolExecutor::with_config(config).expect("Executor creation failed");
        let input = ProtocolInput::query("Test timeout");

        // Mock executor should complete within timeout
        let result = executor.execute("gigathink", input).await;
        assert!(result.is_ok(), "Mock should complete within timeout");
    }

    /// Test: Invalid input type for protocol
    #[tokio::test]
    async fn test_invalid_input_type() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");

        // LaserLogic expects an argument, not just a query
        // This tests input validation
        let input = ProtocolInput::query(""); // Empty input
        let result = executor.execute("laserlogic", input).await;

        // Should either fail gracefully or handle the empty input
        match result {
            Ok(output) => {
                // Mock might succeed but should indicate issue
                assert!(!output.success || output.confidence < 0.5);
            }
            Err(_) => {
                // Expected behavior for invalid input
            }
        }
    }
}

// ============================================================================
// MODULE: Memory Integration Tests (Feature-Gated)
// ============================================================================

#[cfg(feature = "memory")]
mod memory_integration_tests {
    //! Tests for integration with reasonkit-mem when memory feature is enabled

    use reasonkit::embedding;
    use reasonkit::retrieval;
    use reasonkit::storage;

    /// Test: Store document via memory interface
    #[tokio::test]
    async fn test_store_document_via_memory() {
        use reasonkit_mem::service::{Document, MemServiceImpl, MemoryService};
        use std::collections::HashMap;

        let service = MemServiceImpl::in_memory().expect("Failed to create memory service");

        let doc = Document {
            id: None,
            content: "Machine learning is transforming industries.".to_string(),
            metadata: HashMap::new(),
            source: Some("/test/doc.md".to_string()),
            created_at: None,
        };

        let id = service.store_document(&doc).await;
        assert!(
            id.is_ok(),
            "Document storage should succeed: {:?}",
            id.err()
        );

        let stored_id = id.unwrap();
        assert_ne!(stored_id, uuid::Uuid::nil(), "Should have non-nil ID");
    }

    /// Test: Search documents via memory interface
    #[tokio::test]
    async fn test_search_documents_via_memory() {
        use reasonkit_mem::service::{Document, MemServiceImpl, MemoryService};
        use std::collections::HashMap;

        let service = MemServiceImpl::in_memory().expect("Failed to create memory service");

        // Store test documents
        let docs = vec![
            "Machine learning enables predictive analytics.",
            "Deep learning uses neural networks for pattern recognition.",
            "Natural language processing handles text understanding.",
        ];

        for content in docs {
            let doc = Document {
                id: None,
                content: content.to_string(),
                metadata: HashMap::new(),
                source: None,
                created_at: None,
            };
            service.store_document(&doc).await.expect("Store failed");
        }

        // Search using BM25 (sparse search works without embeddings)
        let results = service
            .retriever()
            .search_sparse("machine learning", 5)
            .await;
        assert!(
            results.is_ok(),
            "Search should succeed: {:?}",
            results.err()
        );

        let results = results.unwrap();
        assert!(!results.is_empty(), "Should find matching documents");
    }

    /// Test: Memory service health check
    #[tokio::test]
    async fn test_memory_service_health() {
        use reasonkit_mem::service::{MemServiceImpl, MemoryService};

        let service = MemServiceImpl::in_memory().expect("Failed to create memory service");

        let health = service.health_check().await;
        assert!(health.is_ok(), "Health check should succeed");
        assert!(health.unwrap(), "Service should be healthy");
    }

    /// Test: Memory service shutdown
    #[tokio::test]
    async fn test_memory_service_shutdown() {
        use reasonkit_mem::service::{MemServiceImpl, MemoryService};

        let service = MemServiceImpl::in_memory().expect("Failed to create memory service");

        // Verify healthy before shutdown
        assert!(service.health_check().await.unwrap());

        // Shutdown
        let shutdown_result = service.shutdown().await;
        assert!(shutdown_result.is_ok(), "Shutdown should succeed");

        // Verify unhealthy after shutdown
        assert!(!service.health_check().await.unwrap());
    }

    /// Test: Context window assembly
    #[tokio::test]
    async fn test_context_window_assembly() {
        use reasonkit_mem::service::{Document, MemServiceImpl, MemoryService};
        use std::collections::HashMap;

        let service = MemServiceImpl::in_memory().expect("Failed to create memory service");

        // Store documents
        for i in 0..5 {
            let doc = Document {
                id: None,
                content: format!(
                    "Document {} contains information about AI and machine learning.",
                    i
                ),
                metadata: HashMap::new(),
                source: None,
                created_at: None,
            };
            service.store_document(&doc).await.expect("Store failed");
        }

        // Get context window
        let context = service.get_context("machine learning", 1000).await;
        assert!(
            context.is_ok(),
            "Context assembly should succeed: {:?}",
            context.err()
        );

        let window = context.unwrap();
        assert!(window.total_tokens <= 1000, "Should respect token budget");
    }
}

// ============================================================================
// MODULE: MCP Protocol Tests
// ============================================================================

mod mcp_protocol_tests {
    //! Tests for MCP (Model Context Protocol) interactions

    /// Test: MCP server info structure
    #[test]
    fn test_mcp_server_info() {
        use reasonkit::mcp::{ServerCapabilities, ServerInfo};

        // Create server info
        let info = ServerInfo {
            name: "reasonkit-test".to_string(),
            version: "1.0.0".to_string(),
            description: None,
            vendor: None,
        };

        let capabilities = ServerCapabilities {
            tools: Some(reasonkit::mcp::ToolsCapability { list_changed: true }),
            ..Default::default()
        };

        // Server should initialize correctly
        assert_eq!(info.name, "reasonkit-test");
        assert_eq!(info.version, "1.0.0");
        assert!(capabilities.tools.is_some());
    }

    /// Test: Tool definition and result structure
    #[test]
    fn test_tool_structure() {
        use reasonkit::mcp::{Tool, ToolResult};

        // Create a tool definition
        let tool = Tool {
            name: "test_tool".to_string(),
            description: Some("A test tool".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                }
            }),
            server_id: None,
            server_name: None,
        };

        assert_eq!(tool.name, "test_tool");
        assert!(tool.description.is_some());

        // Create tool result
        let result = ToolResult::text("Test result content");
        assert!(!result.content.is_empty());
    }

    /// Test: MCP request structure
    #[test]
    fn test_mcp_request_structure() {
        use reasonkit::mcp::{JsonRpcVersion, McpRequest, RequestId};

        // Create request
        let request = McpRequest {
            jsonrpc: JsonRpcVersion::default(),
            id: RequestId::Number(1),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": "gigathink",
                "arguments": {
                    "query": "Test query"
                }
            })),
        };

        assert_eq!(request.method, "tools/call");
        assert!(request.params.is_some());

        // Verify serialization
        let serialized = serde_json::to_string(&request);
        assert!(serialized.is_ok(), "Request should serialize");

        let json = serialized.unwrap();
        assert!(json.contains("tools/call"));
        assert!(json.contains("gigathink"));
    }

    /// Test: MCP response structure
    #[test]
    fn test_mcp_response_structure() {
        use reasonkit::mcp::{ErrorCode, JsonRpcVersion, McpError, McpResponse, RequestId};

        // Create success response
        let success_response = McpResponse {
            jsonrpc: JsonRpcVersion::default(),
            id: RequestId::Number(1),
            result: Some(serde_json::json!({
                "content": [{"type": "text", "text": "Success"}]
            })),
            error: None,
        };

        assert!(success_response.result.is_some());
        assert!(success_response.error.is_none());

        // Create error response
        let error_response = McpResponse {
            jsonrpc: JsonRpcVersion::default(),
            id: RequestId::Number(2),
            result: None,
            error: Some(McpError {
                code: ErrorCode::INVALID_REQUEST,
                message: "Invalid Request".to_string(),
                data: None,
            }),
        };

        assert!(error_response.result.is_none());
        assert!(error_response.error.is_some());
    }

    /// Test: JSON-RPC version
    #[test]
    fn test_jsonrpc_version() {
        use reasonkit::mcp::JsonRpcVersion;

        let version = JsonRpcVersion::default();

        // Verify serialization produces "2.0"
        let serialized = serde_json::to_string(&version);
        assert!(serialized.is_ok());
        assert!(serialized.unwrap().contains("2.0"));
    }

    /// Test: MCP client configuration
    #[test]
    fn test_mcp_client_config() {
        use reasonkit::mcp::McpClientConfig;

        let config = McpClientConfig {
            name: "test-server".to_string(),
            command: "npx".to_string(),
            args: vec!["-y".to_string(), "@mcp/test-server".to_string()],
            env: std::collections::HashMap::new(),
            timeout_secs: 30,
            auto_reconnect: true,
            max_retries: 3,
        };

        assert_eq!(config.name, "test-server");
        assert_eq!(config.command, "npx");
        assert!(config.auto_reconnect);
        assert_eq!(config.max_retries, 3);
    }

    /// Test: MCP protocol version constant
    #[test]
    fn test_mcp_version_constant() {
        use reasonkit::mcp::MCP_VERSION;

        assert!(!MCP_VERSION.is_empty());
        assert!(MCP_VERSION.contains("2025"), "Should be 2025 version");
    }
}

// ============================================================================
// MODULE: Concurrency Tests
// ============================================================================

mod concurrency_tests {
    //! Tests for concurrent access patterns

    /// Test: Concurrent protocol execution
    #[tokio::test]
    async fn test_concurrent_protocol_execution() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};
        use std::sync::Arc;

        let executor = Arc::new(ProtocolExecutor::mock().expect("Mock executor creation failed"));

        let mut handles = vec![];

        // Spawn multiple concurrent executions
        for i in 0..5 {
            let executor = Arc::clone(&executor);
            let handle = tokio::spawn(async move {
                let input = ProtocolInput::query(format!("Concurrent query {}", i));
                executor.execute("gigathink", input).await
            });
            handles.push(handle);
        }

        // Collect results
        let mut success_count = 0;
        for handle in handles {
            let result = handle.await.expect("Task panicked");
            if result.is_ok() {
                success_count += 1;
            }
        }

        assert_eq!(success_count, 5, "All concurrent executions should succeed");
    }

    /// Test: Executor is Send + Sync
    #[tokio::test]
    async fn test_executor_thread_safety() {
        use reasonkit::thinktool::ProtocolExecutor;

        fn assert_send_sync<T: Send + Sync>() {}

        // This will fail to compile if ProtocolExecutor is not Send + Sync
        assert_send_sync::<ProtocolExecutor>();
    }
}

// ============================================================================
// MODULE: Performance Baseline Tests
// ============================================================================

mod performance_tests {
    //! Basic performance validation tests

    /// Test: Executor creation performance
    #[tokio::test]
    async fn test_executor_creation_time() {
        use reasonkit::thinktool::ProtocolExecutor;
        use std::time::Instant;

        let start = Instant::now();
        let executor = ProtocolExecutor::mock();
        let creation_time = start.elapsed();

        assert!(executor.is_ok(), "Executor should be created");
        assert!(
            creation_time.as_millis() < 500,
            "Executor creation should be under 500ms, was {:?}",
            creation_time
        );
    }

    /// Test: Protocol execution overhead (mock)
    #[tokio::test]
    async fn test_protocol_execution_time() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};
        use std::time::Instant;

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");
        let input = ProtocolInput::query("Simple test query");

        let start = Instant::now();
        let result = executor.execute("gigathink", input).await;
        let execution_time = start.elapsed();

        assert!(result.is_ok(), "Execution should succeed");
        assert!(
            execution_time.as_millis() < 5000,
            "Mock execution should be under 5s, was {:?}",
            execution_time
        );
    }

    /// Test: Memory efficiency - no unbounded allocation
    #[tokio::test]
    async fn test_memory_efficiency() {
        use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

        let executor = ProtocolExecutor::mock().expect("Mock executor creation failed");

        // Run multiple executions - should not leak memory
        for i in 0..10 {
            let input = ProtocolInput::query(format!("Query iteration {}", i));
            let result = executor.execute("gigathink", input).await;
            assert!(result.is_ok(), "Iteration {} should succeed", i);
        }

        // If we get here without OOM, basic memory efficiency is validated
    }
}

// ============================================================================
// MODULE: Feature Flag Tests
// ============================================================================

mod feature_flag_tests {
    //! Tests for feature flag interactions

    /// Test: Core modules available without features
    #[test]
    fn test_core_modules_available() {
        // Smoke-check module paths exist by touching well-known items.
        let _ = reasonkit::engine::reasoning_loop::ReasoningLoop::builder;
        let _ = std::mem::size_of::<reasonkit::error::Error>;
        let _ = std::mem::size_of::<reasonkit::mcp::McpServer>;
        let _ = reasonkit::thinktool::ProtocolExecutor::mock;
    }

    /// Test: Memory feature detection
    #[test]
    fn test_memory_feature_detection() {
        #[cfg(feature = "memory")]
        {
            use reasonkit::embedding;
            use reasonkit::retrieval;
            use reasonkit::storage;
            // Memory modules available
        }

        #[cfg(not(feature = "memory"))]
        {
            // Memory modules should not be available
            // Just verify we can still use core functionality
            let _ = reasonkit::thinktool::ProtocolExecutor::mock;
        }
    }

    /// Test: Version constant is available
    #[test]
    fn test_version_constant() {
        use reasonkit::VERSION;

        assert!(!VERSION.is_empty(), "VERSION should be set");
        // Verify version format (semver-like)
        assert!(
            VERSION.contains('.'),
            "VERSION should be semver format: {}",
            VERSION
        );
    }
}
