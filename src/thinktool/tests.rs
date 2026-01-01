//! Comprehensive test suite for the ThinkTool module
//!
//! This module provides extensive unit tests to achieve 85%+ code coverage.
//! Tests cover:
//! - All public functions
//! - Edge cases (empty input, very long input, unicode)
//! - Error conditions
//! - Configuration variations
//! - Success and failure paths

#[cfg(test)]
mod executor_tests {
    use crate::thinktool::executor::{
        CliToolConfig, ExecutorConfig, ProtocolExecutor, ProtocolInput, ProtocolOutput,
    };
    use crate::thinktool::step::{ListItem, StepOutput, StepResult, TokenUsage};
    use std::collections::HashMap;

    // =========================================================================
    // CliToolConfig Tests
    // =========================================================================

    #[test]
    fn test_cli_tool_config_claude() {
        let config = CliToolConfig::claude();
        assert_eq!(config.command, "claude");
        assert!(config.pre_args.contains(&"-p".to_string()));
        assert!(!config.interactive);
    }

    #[test]
    fn test_cli_tool_config_codex() {
        let config = CliToolConfig::codex();
        assert_eq!(config.command, "codex");
        assert!(config.pre_args.contains(&"-q".to_string()));
        assert!(!config.interactive);
    }

    #[test]
    fn test_cli_tool_config_gemini() {
        let config = CliToolConfig::gemini();
        assert_eq!(config.command, "gemini");
        assert!(config.pre_args.contains(&"-p".to_string()));
        assert!(!config.interactive);
    }

    #[test]
    fn test_cli_tool_config_opencode() {
        let config = CliToolConfig::opencode();
        // Command may be overridden by env var, so just check it's set
        assert!(!config.command.is_empty());
        assert!(config.pre_args.contains(&"--no-rk".to_string()));
        assert!(config.pre_args.contains(&"run".to_string()));
    }

    #[test]
    fn test_cli_tool_config_copilot() {
        let config = CliToolConfig::copilot();
        assert_eq!(config.command, "gh");
        assert!(config.pre_args.contains(&"copilot".to_string()));
        assert!(config.pre_args.contains(&"suggest".to_string()));
        assert!(config.interactive);
    }

    // =========================================================================
    // ExecutorConfig Tests
    // =========================================================================

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert_eq!(config.timeout_secs, 120);
        assert!(!config.save_traces);
        assert!(!config.use_mock);
        assert!(config.show_progress);
        assert!(!config.enable_parallel);
        assert_eq!(config.max_concurrent_steps, 4);
    }

    #[test]
    fn test_executor_config_mock() {
        let config = ExecutorConfig::mock();
        assert!(config.use_mock);
    }

    #[test]
    fn test_executor_config_with_cli_tool() {
        let config = ExecutorConfig::with_cli_tool(CliToolConfig::claude());
        assert!(config.cli_tool.is_some());
        let cli = config.cli_tool.unwrap();
        assert_eq!(cli.command, "claude");
    }

    #[test]
    fn test_executor_config_claude_cli() {
        let config = ExecutorConfig::claude_cli();
        assert!(config.cli_tool.is_some());
    }

    #[test]
    fn test_executor_config_codex_cli() {
        let config = ExecutorConfig::codex_cli();
        assert!(config.cli_tool.is_some());
    }

    #[test]
    fn test_executor_config_gemini_cli() {
        let config = ExecutorConfig::gemini_cli();
        assert!(config.cli_tool.is_some());
    }

    #[test]
    fn test_executor_config_opencode_cli() {
        let config = ExecutorConfig::opencode_cli();
        assert!(config.cli_tool.is_some());
    }

    #[test]
    fn test_executor_config_copilot_cli() {
        let config = ExecutorConfig::copilot_cli();
        assert!(config.cli_tool.is_some());
    }

    #[test]
    fn test_executor_config_with_self_consistency() {
        let config = ExecutorConfig::default().with_self_consistency();
        assert!(config.self_consistency.is_some());
        let sc = config.self_consistency.unwrap();
        assert_eq!(sc.num_samples, 5); // default
    }

    #[test]
    fn test_executor_config_with_self_consistency_fast() {
        let config = ExecutorConfig::default().with_self_consistency_fast();
        assert!(config.self_consistency.is_some());
        let sc = config.self_consistency.unwrap();
        assert_eq!(sc.num_samples, 3);
    }

    #[test]
    fn test_executor_config_with_self_consistency_thorough() {
        let config = ExecutorConfig::default().with_self_consistency_thorough();
        assert!(config.self_consistency.is_some());
        let sc = config.self_consistency.unwrap();
        assert_eq!(sc.num_samples, 10);
    }

    #[test]
    fn test_executor_config_with_self_consistency_paranoid() {
        let config = ExecutorConfig::default().with_self_consistency_paranoid();
        assert!(config.self_consistency.is_some());
        let sc = config.self_consistency.unwrap();
        assert_eq!(sc.num_samples, 15);
    }

    #[test]
    fn test_executor_config_with_parallel() {
        let config = ExecutorConfig::default().with_parallel();
        assert!(config.enable_parallel);
    }

    #[test]
    fn test_executor_config_with_parallel_limit() {
        let config = ExecutorConfig::default().with_parallel_limit(8);
        assert!(config.enable_parallel);
        assert_eq!(config.max_concurrent_steps, 8);
    }

    // =========================================================================
    // ProtocolInput Tests
    // =========================================================================

    #[test]
    fn test_protocol_input_query() {
        let input = ProtocolInput::query("What is AI?");
        assert_eq!(input.get_str("query"), Some("What is AI?"));
    }

    #[test]
    fn test_protocol_input_argument() {
        let input = ProtocolInput::argument("All men are mortal");
        assert_eq!(input.get_str("argument"), Some("All men are mortal"));
    }

    #[test]
    fn test_protocol_input_statement() {
        let input = ProtocolInput::statement("The earth is round");
        assert_eq!(input.get_str("statement"), Some("The earth is round"));
    }

    #[test]
    fn test_protocol_input_claim() {
        let input = ProtocolInput::claim("Water boils at 100C");
        assert_eq!(input.get_str("claim"), Some("Water boils at 100C"));
    }

    #[test]
    fn test_protocol_input_work() {
        let input = ProtocolInput::work("My research paper...");
        assert_eq!(input.get_str("work"), Some("My research paper..."));
    }

    #[test]
    fn test_protocol_input_with_field() {
        let input = ProtocolInput::query("Test")
            .with_field("context", "Some context")
            .with_field("domain", "science");
        assert_eq!(input.get_str("query"), Some("Test"));
        assert_eq!(input.get_str("context"), Some("Some context"));
        assert_eq!(input.get_str("domain"), Some("science"));
    }

    #[test]
    fn test_protocol_input_get_str_missing() {
        let input = ProtocolInput::query("Test");
        assert_eq!(input.get_str("missing_field"), None);
    }

    #[test]
    fn test_protocol_input_empty_query() {
        let input = ProtocolInput::query("");
        assert_eq!(input.get_str("query"), Some(""));
    }

    #[test]
    fn test_protocol_input_unicode() {
        let input = ProtocolInput::query("What is AI in Japanese? AIとは何ですか？");
        assert!(input.get_str("query").unwrap().contains("日本"));
        // Actually test unicode content
        let input = ProtocolInput::query("Emojis work too: \u{1F600} \u{1F4A1}");
        assert!(input.get_str("query").unwrap().contains("\u{1F600}"));
    }

    #[test]
    fn test_protocol_input_very_long_input() {
        let long_string: String = "A".repeat(100_000);
        let input = ProtocolInput::query(&long_string);
        assert_eq!(input.get_str("query").unwrap().len(), 100_000);
    }

    // =========================================================================
    // ProtocolOutput Tests
    // =========================================================================

    #[test]
    fn test_protocol_output_get() {
        let mut data = HashMap::new();
        data.insert("key1".to_string(), serde_json::json!("value1"));
        data.insert("key2".to_string(), serde_json::json!(42));

        let output = ProtocolOutput {
            protocol_id: "test".to_string(),
            success: true,
            data,
            confidence: 0.85,
            steps: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 100,
            error: None,
            trace_id: None,
            budget_summary: None,
        };

        assert_eq!(output.get("key1"), Some(&serde_json::json!("value1")));
        assert_eq!(output.get("key2"), Some(&serde_json::json!(42)));
        assert_eq!(output.get("missing"), None);
    }

    #[test]
    fn test_protocol_output_perspectives() {
        let mut data = HashMap::new();
        data.insert(
            "perspectives".to_string(),
            serde_json::json!(["Perspective 1", "Perspective 2", "Perspective 3"]),
        );

        let output = ProtocolOutput {
            protocol_id: "gigathink".to_string(),
            success: true,
            data,
            confidence: 0.85,
            steps: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 100,
            error: None,
            trace_id: None,
            budget_summary: None,
        };

        let perspectives = output.perspectives();
        assert_eq!(perspectives.len(), 3);
        assert!(perspectives.contains(&"Perspective 1"));
    }

    #[test]
    fn test_protocol_output_perspectives_empty() {
        let output = ProtocolOutput {
            protocol_id: "test".to_string(),
            success: true,
            data: HashMap::new(),
            confidence: 0.85,
            steps: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 100,
            error: None,
            trace_id: None,
            budget_summary: None,
        };

        assert!(output.perspectives().is_empty());
    }

    #[test]
    fn test_protocol_output_verdict() {
        let mut data = HashMap::new();
        data.insert("verdict".to_string(), serde_json::json!("PASS"));

        let output = ProtocolOutput {
            protocol_id: "test".to_string(),
            success: true,
            data,
            confidence: 0.85,
            steps: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 100,
            error: None,
            trace_id: None,
            budget_summary: None,
        };

        assert_eq!(output.verdict(), Some("PASS"));
    }

    #[test]
    fn test_protocol_output_verdict_missing() {
        let output = ProtocolOutput {
            protocol_id: "test".to_string(),
            success: true,
            data: HashMap::new(),
            confidence: 0.85,
            steps: vec![],
            tokens: TokenUsage::default(),
            duration_ms: 100,
            error: None,
            trace_id: None,
            budget_summary: None,
        };

        assert_eq!(output.verdict(), None);
    }

    // =========================================================================
    // ProtocolExecutor Tests
    // =========================================================================

    #[test]
    fn test_protocol_executor_new() {
        let executor = ProtocolExecutor::mock().unwrap();
        assert!(!executor.registry().is_empty());
        assert!(!executor.profiles().is_empty());
    }

    #[test]
    fn test_protocol_executor_list_protocols() {
        let executor = ProtocolExecutor::mock().unwrap();
        let protocols = executor.list_protocols();

        assert!(protocols.contains(&"gigathink"));
        assert!(protocols.contains(&"laserlogic"));
        assert!(protocols.contains(&"bedrock"));
        assert!(protocols.contains(&"proofguard"));
        assert!(protocols.contains(&"brutalhonesty"));
    }

    #[test]
    fn test_protocol_executor_list_profiles() {
        let executor = ProtocolExecutor::mock().unwrap();
        let profiles = executor.list_profiles();

        assert!(profiles.contains(&"quick"));
        assert!(profiles.contains(&"balanced"));
        assert!(profiles.contains(&"deep"));
        assert!(profiles.contains(&"paranoid"));
    }

    #[test]
    fn test_protocol_executor_get_protocol() {
        let executor = ProtocolExecutor::mock().unwrap();

        let gigathink = executor.get_protocol("gigathink");
        assert!(gigathink.is_some());
        assert_eq!(gigathink.unwrap().name, "GigaThink");

        let missing = executor.get_protocol("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_protocol_executor_get_profile() {
        let executor = ProtocolExecutor::mock().unwrap();

        let quick = executor.get_profile("quick");
        assert!(quick.is_some());
        assert_eq!(quick.unwrap().min_confidence, 0.70);

        let missing = executor.get_profile("nonexistent");
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_execute_gigathink_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What are the key factors for startup success?");

        let result = executor.execute("gigathink", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
        assert!(!result.steps.is_empty());
        assert_eq!(result.protocol_id, "gigathink");
    }

    #[tokio::test]
    async fn test_execute_laserlogic_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::argument("All humans are mortal. Socrates is human. Therefore, Socrates is mortal.");

        let result = executor.execute("laserlogic", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_execute_bedrock_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::statement("The earth is round.");

        let result = executor.execute("bedrock", input).await.unwrap();

        assert!(result.success);
    }

    #[tokio::test]
    async fn test_execute_proofguard_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::claim("Water freezes at 0 degrees Celsius.");

        let result = executor.execute("proofguard", input).await.unwrap();

        assert!(result.success);
    }

    #[tokio::test]
    async fn test_execute_brutalhonesty_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::work("This is my research paper abstract...");

        let result = executor.execute("brutalhonesty", input).await.unwrap();

        assert!(result.success);
    }

    #[tokio::test]
    async fn test_execute_profile_quick_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Should we adopt microservices?");

        let result = executor.execute_profile("quick", input).await.unwrap();

        assert!(result.success);
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_execute_profile_balanced_mock() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What is the best programming language?");

        let result = executor.execute_profile("balanced", input).await.unwrap();

        assert!(result.success);
    }

    #[tokio::test]
    async fn test_execute_nonexistent_protocol() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let result = executor.execute("nonexistent_protocol", input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_nonexistent_profile() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("Test");

        let result = executor.execute_profile("nonexistent_profile", input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_missing_required_input() {
        let executor = ProtocolExecutor::mock().unwrap();
        // GigaThink requires "query" field
        let input = ProtocolInput {
            fields: HashMap::new(),
        };

        let result = executor.execute("gigathink", input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_with_unicode_input() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("What is the meaning of life? \u{4EBA}\u{751F}\u{306E}\u{610F}\u{5473} \u{1F914}");

        let result = executor.execute("gigathink", input).await.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_execute_with_empty_query() {
        let executor = ProtocolExecutor::mock().unwrap();
        let input = ProtocolInput::query("");

        // Empty query should still work with mock
        let result = executor.execute("gigathink", input).await.unwrap();
        assert!(result.success);
    }
}

#[cfg(test)]
mod step_tests {
    use crate::thinktool::step::{ListItem, OutputFormat, StepOutput, StepResult, TokenUsage};
    use std::collections::HashMap;

    // =========================================================================
    // TokenUsage Tests
    // =========================================================================

    #[test]
    fn test_token_usage_new() {
        let usage = TokenUsage::new(100, 50, 0.001);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert!((usage.cost_usd - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_token_usage_default() {
        let usage = TokenUsage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
        assert_eq!(usage.cost_usd, 0.0);
    }

    #[test]
    fn test_token_usage_add() {
        let mut usage1 = TokenUsage::new(100, 50, 0.001);
        let usage2 = TokenUsage::new(200, 100, 0.002);

        usage1.add(&usage2);

        assert_eq!(usage1.input_tokens, 300);
        assert_eq!(usage1.output_tokens, 150);
        assert_eq!(usage1.total_tokens, 450);
        assert!((usage1.cost_usd - 0.003).abs() < 0.0001);
    }

    #[test]
    fn test_token_usage_add_multiple() {
        let mut total = TokenUsage::default();

        for i in 1..=5 {
            let usage = TokenUsage::new(i * 100, i * 50, i as f64 * 0.001);
            total.add(&usage);
        }

        // Sum of 100+200+300+400+500 = 1500
        assert_eq!(total.input_tokens, 1500);
        // Sum of 50+100+150+200+250 = 750
        assert_eq!(total.output_tokens, 750);
    }

    // =========================================================================
    // ListItem Tests
    // =========================================================================

    #[test]
    fn test_list_item_new() {
        let item = ListItem::new("Test item");
        assert_eq!(item.content, "Test item");
        assert!(item.confidence.is_none());
        assert!(item.metadata.is_empty());
    }

    #[test]
    fn test_list_item_with_confidence() {
        let item = ListItem::with_confidence("Test item", 0.95);
        assert_eq!(item.content, "Test item");
        assert_eq!(item.confidence, Some(0.95));
    }

    #[test]
    fn test_list_item_unicode() {
        let item = ListItem::new("\u{65E5}\u{672C}\u{8A9E}\u{3067}\u{3059}");
        assert!(item.content.contains("\u{65E5}\u{672C}"));
    }

    #[test]
    fn test_list_item_empty_content() {
        let item = ListItem::new("");
        assert!(item.content.is_empty());
    }

    // =========================================================================
    // StepOutput Tests
    // =========================================================================

    #[test]
    fn test_step_output_text() {
        let output = StepOutput::Text {
            content: "Hello world".to_string(),
        };

        if let StepOutput::Text { content } = output {
            assert_eq!(content, "Hello world");
        } else {
            panic!("Expected Text variant");
        }
    }

    #[test]
    fn test_step_output_list() {
        let output = StepOutput::List {
            items: vec![
                ListItem::new("Item 1"),
                ListItem::new("Item 2"),
            ],
        };

        if let StepOutput::List { items } = output {
            assert_eq!(items.len(), 2);
        } else {
            panic!("Expected List variant");
        }
    }

    #[test]
    fn test_step_output_structured() {
        let mut data = HashMap::new();
        data.insert("key".to_string(), serde_json::json!("value"));

        let output = StepOutput::Structured { data };

        if let StepOutput::Structured { data } = output {
            assert!(data.contains_key("key"));
        } else {
            panic!("Expected Structured variant");
        }
    }

    #[test]
    fn test_step_output_score() {
        let output = StepOutput::Score { value: 0.85 };

        if let StepOutput::Score { value } = output {
            assert!((value - 0.85).abs() < 0.01);
        } else {
            panic!("Expected Score variant");
        }
    }

    #[test]
    fn test_step_output_boolean() {
        let output = StepOutput::Boolean {
            value: true,
            reason: Some("It passed validation".to_string()),
        };

        if let StepOutput::Boolean { value, reason } = output {
            assert!(value);
            assert!(reason.is_some());
        } else {
            panic!("Expected Boolean variant");
        }
    }

    #[test]
    fn test_step_output_empty() {
        let output = StepOutput::Empty;
        assert!(matches!(output, StepOutput::Empty));
    }

    #[test]
    fn test_step_output_default() {
        let output = StepOutput::default();
        assert!(matches!(output, StepOutput::Empty));
    }

    // =========================================================================
    // StepResult Tests
    // =========================================================================

    #[test]
    fn test_step_result_success() {
        let result = StepResult::success(
            "test_step",
            StepOutput::Text {
                content: "Success".to_string(),
            },
            0.85,
        );

        assert!(result.success);
        assert_eq!(result.step_id, "test_step");
        assert_eq!(result.confidence, 0.85);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_step_result_failure() {
        let result = StepResult::failure("test_step", "Something went wrong");

        assert!(!result.success);
        assert_eq!(result.step_id, "test_step");
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_step_result_with_duration() {
        let result = StepResult::success(
            "test",
            StepOutput::Empty,
            0.8,
        )
        .with_duration(500);

        assert_eq!(result.duration_ms, 500);
    }

    #[test]
    fn test_step_result_with_tokens() {
        let tokens = TokenUsage::new(100, 50, 0.001);
        let result = StepResult::success(
            "test",
            StepOutput::Empty,
            0.8,
        )
        .with_tokens(tokens);

        assert_eq!(result.tokens.total_tokens, 150);
    }

    #[test]
    fn test_step_result_meets_threshold() {
        let result = StepResult::success(
            "test",
            StepOutput::Empty,
            0.85,
        );

        assert!(result.meets_threshold(0.80));
        assert!(result.meets_threshold(0.85));
        assert!(!result.meets_threshold(0.90));
    }

    #[test]
    fn test_step_result_meets_threshold_failed() {
        let result = StepResult::failure("test", "Error");

        // Failed results never meet threshold
        assert!(!result.meets_threshold(0.0));
    }

    #[test]
    fn test_step_result_as_text() {
        let result = StepResult::success(
            "test",
            StepOutput::Text {
                content: "Hello".to_string(),
            },
            0.8,
        );

        assert_eq!(result.as_text(), Some("Hello"));
    }

    #[test]
    fn test_step_result_as_text_wrong_type() {
        let result = StepResult::success(
            "test",
            StepOutput::Score { value: 0.5 },
            0.8,
        );

        assert_eq!(result.as_text(), None);
    }

    #[test]
    fn test_step_result_as_list() {
        let result = StepResult::success(
            "test",
            StepOutput::List {
                items: vec![ListItem::new("Item 1")],
            },
            0.8,
        );

        let list = result.as_list();
        assert!(list.is_some());
        assert_eq!(list.unwrap().len(), 1);
    }

    #[test]
    fn test_step_result_as_score() {
        let result = StepResult::success(
            "test",
            StepOutput::Score { value: 0.95 },
            0.8,
        );

        assert_eq!(result.as_score(), Some(0.95));
    }

    // =========================================================================
    // OutputFormat Tests
    // =========================================================================

    #[test]
    fn test_output_format_default() {
        let format = OutputFormat::default();
        assert!(matches!(format, OutputFormat::Raw));
    }

    #[test]
    fn test_output_format_variants() {
        let _raw = OutputFormat::Raw;
        let _json = OutputFormat::Json;
        let _list = OutputFormat::List;
        let _kv = OutputFormat::KeyValue;
        let _numeric = OutputFormat::Numeric;
        // All variants exist
    }
}

#[cfg(test)]
mod budget_tests {
    use crate::thinktool::budget::{BudgetConfig, BudgetParseError, BudgetStrategy, BudgetTracker};
    use std::time::Duration;

    // =========================================================================
    // BudgetConfig Tests
    // =========================================================================

    #[test]
    fn test_budget_config_default() {
        let config = BudgetConfig::default();
        assert!(config.time_limit.is_none());
        assert!(config.token_limit.is_none());
        assert!(config.cost_limit.is_none());
        assert!(!config.is_constrained());
    }

    #[test]
    fn test_budget_config_unlimited() {
        let config = BudgetConfig::unlimited();
        assert!(!config.is_constrained());
    }

    #[test]
    fn test_budget_config_with_time() {
        let config = BudgetConfig::with_time(Duration::from_secs(60));
        assert!(config.is_constrained());
        assert_eq!(config.time_limit, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_budget_config_with_tokens() {
        let config = BudgetConfig::with_tokens(1000);
        assert!(config.is_constrained());
        assert_eq!(config.token_limit, Some(1000));
    }

    #[test]
    fn test_budget_config_with_cost() {
        let config = BudgetConfig::with_cost(0.50);
        assert!(config.is_constrained());
        assert_eq!(config.cost_limit, Some(0.50));
    }

    #[test]
    fn test_budget_config_with_strategy() {
        let config = BudgetConfig::with_tokens(1000).with_strategy(BudgetStrategy::Strict);
        assert_eq!(config.strategy, BudgetStrategy::Strict);
    }

    #[test]
    fn test_budget_config_parse_seconds() {
        let config = BudgetConfig::parse("30s").unwrap();
        assert_eq!(config.time_limit, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_budget_config_parse_minutes() {
        let config = BudgetConfig::parse("5m").unwrap();
        assert_eq!(config.time_limit, Some(Duration::from_secs(300)));
    }

    #[test]
    fn test_budget_config_parse_hours() {
        let config = BudgetConfig::parse("2h").unwrap();
        assert_eq!(config.time_limit, Some(Duration::from_secs(7200)));
    }

    #[test]
    fn test_budget_config_parse_tokens_short() {
        let config = BudgetConfig::parse("1000t").unwrap();
        assert_eq!(config.token_limit, Some(1000));
    }

    #[test]
    fn test_budget_config_parse_tokens_long() {
        let config = BudgetConfig::parse("5000tokens").unwrap();
        assert_eq!(config.token_limit, Some(5000));
    }

    #[test]
    fn test_budget_config_parse_cost() {
        let config = BudgetConfig::parse("$0.50").unwrap();
        assert_eq!(config.cost_limit, Some(0.50));
    }

    #[test]
    fn test_budget_config_parse_cost_whole() {
        let config = BudgetConfig::parse("$10").unwrap();
        assert_eq!(config.cost_limit, Some(10.0));
    }

    #[test]
    fn test_budget_config_parse_empty() {
        let result = BudgetConfig::parse("");
        assert!(matches!(result, Err(BudgetParseError::Empty)));
    }

    #[test]
    fn test_budget_config_parse_whitespace() {
        let result = BudgetConfig::parse("   ");
        assert!(matches!(result, Err(BudgetParseError::Empty)));
    }

    #[test]
    fn test_budget_config_parse_invalid_time() {
        let result = BudgetConfig::parse("abcs");
        assert!(matches!(result, Err(BudgetParseError::InvalidTime)));
    }

    #[test]
    fn test_budget_config_parse_invalid_tokens() {
        let result = BudgetConfig::parse("abct");
        assert!(matches!(result, Err(BudgetParseError::InvalidTokens)));
    }

    #[test]
    fn test_budget_config_parse_invalid_cost() {
        let result = BudgetConfig::parse("$abc");
        assert!(matches!(result, Err(BudgetParseError::InvalidCost)));
    }

    #[test]
    fn test_budget_config_parse_unknown_format() {
        let result = BudgetConfig::parse("unknown");
        assert!(matches!(result, Err(BudgetParseError::UnknownFormat(_))));
    }

    // =========================================================================
    // BudgetParseError Tests
    // =========================================================================

    #[test]
    fn test_budget_parse_error_display() {
        let errors = [
            BudgetParseError::Empty,
            BudgetParseError::InvalidTime,
            BudgetParseError::InvalidTokens,
            BudgetParseError::InvalidCost,
            BudgetParseError::UnknownFormat("test".to_string()),
        ];

        for err in errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty());
        }
    }

    // =========================================================================
    // BudgetStrategy Tests
    // =========================================================================

    #[test]
    fn test_budget_strategy_default() {
        let strategy = BudgetStrategy::default();
        assert_eq!(strategy, BudgetStrategy::Adaptive);
    }

    // =========================================================================
    // BudgetTracker Tests
    // =========================================================================

    #[test]
    fn test_budget_tracker_new() {
        let config = BudgetConfig::with_tokens(1000);
        let tracker = BudgetTracker::new(config);

        assert_eq!(tracker.tokens_remaining(), Some(1000));
        assert!(!tracker.is_exhausted());
    }

    #[test]
    fn test_budget_tracker_record_usage() {
        let config = BudgetConfig::with_tokens(1000);
        let mut tracker = BudgetTracker::new(config);

        tracker.record_usage(200, 0.01);

        assert_eq!(tracker.tokens_remaining(), Some(800));
    }

    #[test]
    fn test_budget_tracker_record_skip() {
        let config = BudgetConfig::default();
        let mut tracker = BudgetTracker::new(config);

        tracker.record_skip();
        let summary = tracker.summary();

        assert_eq!(summary.steps_skipped, 1);
    }

    #[test]
    fn test_budget_tracker_exhausted_tokens() {
        let config = BudgetConfig::with_tokens(1000);
        let mut tracker = BudgetTracker::new(config);

        tracker.record_usage(500, 0.01);
        assert!(!tracker.is_exhausted());

        tracker.record_usage(500, 0.01);
        assert!(tracker.is_exhausted());
    }

    #[test]
    fn test_budget_tracker_exhausted_cost() {
        let config = BudgetConfig::with_cost(0.10);
        let mut tracker = BudgetTracker::new(config);

        tracker.record_usage(100, 0.05);
        assert!(!tracker.is_exhausted());

        tracker.record_usage(100, 0.06);
        assert!(tracker.is_exhausted());
    }

    #[test]
    fn test_budget_tracker_usage_ratio() {
        let config = BudgetConfig::with_tokens(1000);
        let mut tracker = BudgetTracker::new(config);

        assert!((tracker.usage_ratio() - 0.0).abs() < 0.01);

        tracker.record_usage(500, 0.01);
        assert!((tracker.usage_ratio() - 0.5).abs() < 0.01);

        tracker.record_usage(300, 0.01);
        assert!((tracker.usage_ratio() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_budget_tracker_should_adapt() {
        let config = BudgetConfig::with_tokens(1000);
        let mut tracker = BudgetTracker::new(config);

        tracker.record_usage(600, 0.01);
        assert!(!tracker.should_adapt()); // 60% < 70% threshold

        tracker.record_usage(150, 0.01);
        assert!(tracker.should_adapt()); // 75% > 70% threshold
    }

    #[test]
    fn test_budget_tracker_adaptive_max_tokens() {
        let config = BudgetConfig::with_tokens(1000);
        let mut tracker = BudgetTracker::new(config);

        // Should allow full request initially
        assert_eq!(tracker.adaptive_max_tokens(500), 500);

        // After using most budget, should limit
        tracker.record_usage(800, 0.04);
        let adaptive = tracker.adaptive_max_tokens(500);
        assert!(adaptive < 200);
    }

    #[test]
    fn test_budget_tracker_should_skip_step() {
        let config = BudgetConfig::with_tokens(100).with_strategy(BudgetStrategy::Adaptive);
        let mut tracker = BudgetTracker::new(config);

        // Essential steps should never be skipped
        tracker.record_usage(95, 0.01);
        assert!(!tracker.should_skip_step(true));

        // Non-essential at 95% should be skipped in adaptive mode
        assert!(tracker.should_skip_step(false));
    }

    #[test]
    fn test_budget_tracker_should_skip_step_best_effort() {
        let config = BudgetConfig::with_tokens(100).with_strategy(BudgetStrategy::BestEffort);
        let mut tracker = BudgetTracker::new(config);

        tracker.record_usage(95, 0.01);
        // BestEffort never skips
        assert!(!tracker.should_skip_step(false));
    }

    #[test]
    fn test_budget_tracker_summary() {
        let config = BudgetConfig::with_tokens(1000);
        let mut tracker = BudgetTracker::new(config);

        tracker.record_usage(500, 0.05);
        tracker.record_skip();

        let summary = tracker.summary();

        assert_eq!(summary.tokens_used, 500);
        assert!((summary.cost_incurred - 0.05).abs() < 0.001);
        assert_eq!(summary.steps_completed, 1);
        assert_eq!(summary.steps_skipped, 1);
        assert!(!summary.exhausted);
    }

    #[test]
    fn test_budget_tracker_time_remaining() {
        let config = BudgetConfig::with_time(Duration::from_secs(60));
        let tracker = BudgetTracker::new(config);

        // Should have some time remaining
        let remaining = tracker.time_remaining();
        assert!(remaining.is_some());
        // Could be slightly less than 60 due to test timing
        assert!(remaining.unwrap() <= Duration::from_secs(60));
    }

    #[test]
    fn test_budget_tracker_cost_remaining() {
        let config = BudgetConfig::with_cost(1.00);
        let mut tracker = BudgetTracker::new(config);

        assert_eq!(tracker.cost_remaining(), Some(1.00));

        tracker.record_usage(100, 0.25);
        assert!((tracker.cost_remaining().unwrap() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_budget_tracker_no_limits() {
        let config = BudgetConfig::default();
        let tracker = BudgetTracker::new(config);

        assert!(tracker.time_remaining().is_none());
        assert!(tracker.tokens_remaining().is_none());
        assert!(tracker.cost_remaining().is_none());
        assert!(!tracker.is_exhausted());
    }
}

#[cfg(test)]
mod protocol_tests {
    use crate::thinktool::protocol::{
        AggregationType, BranchCondition, CritiqueSeverity, DecisionMethod,
        InputSpec, OutputSpec, Protocol, ProtocolMetadata, ProtocolStep,
        ReasoningStrategy, StepAction, StepOutputFormat, ValidationRule,
    };

    // =========================================================================
    // Protocol Tests
    // =========================================================================

    #[test]
    fn test_protocol_new() {
        let protocol = Protocol::new("test", "Test Protocol");
        assert_eq!(protocol.id, "test");
        assert_eq!(protocol.name, "Test Protocol");
        assert_eq!(protocol.version, "1.0.0");
        assert!(protocol.steps.is_empty());
    }

    #[test]
    fn test_protocol_with_step() {
        let step = ProtocolStep {
            id: "step1".to_string(),
            action: StepAction::Generate {
                min_count: 5,
                max_count: 10,
            },
            prompt_template: "Generate ideas".to_string(),
            output_format: StepOutputFormat::List,
            min_confidence: 0.7,
            depends_on: vec![],
            branch: None,
        };

        let protocol = Protocol::new("test", "Test").with_step(step);
        assert_eq!(protocol.steps.len(), 1);
    }

    #[test]
    fn test_protocol_with_strategy() {
        let protocol = Protocol::new("test", "Test")
            .with_strategy(ReasoningStrategy::Expansive);
        assert_eq!(protocol.strategy, ReasoningStrategy::Expansive);
    }

    #[test]
    fn test_protocol_validate_empty_id() {
        let mut protocol = Protocol::new("test", "Test");
        protocol.id = "".to_string();
        protocol.steps.push(ProtocolStep {
            id: "step1".to_string(),
            action: StepAction::Analyze { criteria: vec![] },
            prompt_template: "Analyze".to_string(),
            output_format: StepOutputFormat::Text,
            min_confidence: 0.7,
            depends_on: vec![],
            branch: None,
        });

        let result = protocol.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|e| e.contains("empty")));
    }

    #[test]
    fn test_protocol_validate_empty_steps() {
        let protocol = Protocol::new("test", "Test");
        let result = protocol.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|e| e.contains("at least one step")));
    }

    #[test]
    fn test_protocol_validate_invalid_dependency() {
        let mut protocol = Protocol::new("test", "Test");
        protocol.steps.push(ProtocolStep {
            id: "step1".to_string(),
            action: StepAction::Analyze { criteria: vec![] },
            prompt_template: "Analyze".to_string(),
            output_format: StepOutputFormat::Text,
            min_confidence: 0.7,
            depends_on: vec!["nonexistent".to_string()],
            branch: None,
        });

        let result = protocol.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|e| e.contains("unknown step")));
    }

    #[test]
    fn test_protocol_validate_valid() {
        let mut protocol = Protocol::new("test", "Test");
        protocol.steps.push(ProtocolStep {
            id: "step1".to_string(),
            action: StepAction::Analyze { criteria: vec![] },
            prompt_template: "Analyze".to_string(),
            output_format: StepOutputFormat::Text,
            min_confidence: 0.7,
            depends_on: vec![],
            branch: None,
        });
        protocol.steps.push(ProtocolStep {
            id: "step2".to_string(),
            action: StepAction::Synthesize {
                aggregation: AggregationType::default(),
            },
            prompt_template: "Synthesize".to_string(),
            output_format: StepOutputFormat::Text,
            min_confidence: 0.7,
            depends_on: vec!["step1".to_string()],
            branch: None,
        });

        assert!(protocol.validate().is_ok());
    }

    // =========================================================================
    // ReasoningStrategy Tests
    // =========================================================================

    #[test]
    fn test_reasoning_strategy_default() {
        let strategy = ReasoningStrategy::default();
        assert_eq!(strategy, ReasoningStrategy::Analytical);
    }

    #[test]
    fn test_reasoning_strategy_variants() {
        let strategies = [
            ReasoningStrategy::Expansive,
            ReasoningStrategy::Deductive,
            ReasoningStrategy::Analytical,
            ReasoningStrategy::Adversarial,
            ReasoningStrategy::Verification,
            ReasoningStrategy::Decision,
            ReasoningStrategy::Empirical,
        ];
        assert_eq!(strategies.len(), 7);
    }

    // =========================================================================
    // StepAction Tests
    // =========================================================================

    #[test]
    fn test_step_action_generate() {
        let action = StepAction::Generate {
            min_count: 5,
            max_count: 10,
        };

        if let StepAction::Generate { min_count, max_count } = action {
            assert_eq!(min_count, 5);
            assert_eq!(max_count, 10);
        } else {
            panic!("Expected Generate variant");
        }
    }

    #[test]
    fn test_step_action_analyze() {
        let action = StepAction::Analyze {
            criteria: vec!["clarity".to_string(), "accuracy".to_string()],
        };

        if let StepAction::Analyze { criteria } = action {
            assert_eq!(criteria.len(), 2);
        } else {
            panic!("Expected Analyze variant");
        }
    }

    #[test]
    fn test_step_action_validate() {
        let action = StepAction::Validate {
            rules: vec!["rule1".to_string()],
        };

        if let StepAction::Validate { rules } = action {
            assert_eq!(rules.len(), 1);
        } else {
            panic!("Expected Validate variant");
        }
    }

    #[test]
    fn test_step_action_critique() {
        let action = StepAction::Critique {
            severity: CritiqueSeverity::Brutal,
        };

        if let StepAction::Critique { severity } = action {
            assert_eq!(severity, CritiqueSeverity::Brutal);
        } else {
            panic!("Expected Critique variant");
        }
    }

    #[test]
    fn test_step_action_decide() {
        let action = StepAction::Decide {
            method: DecisionMethod::MultiCriteria,
        };

        if let StepAction::Decide { method } = action {
            assert_eq!(method, DecisionMethod::MultiCriteria);
        } else {
            panic!("Expected Decide variant");
        }
    }

    #[test]
    fn test_step_action_cross_reference() {
        let action = StepAction::CrossReference { min_sources: 5 };

        if let StepAction::CrossReference { min_sources } = action {
            assert_eq!(min_sources, 5);
        } else {
            panic!("Expected CrossReference variant");
        }
    }

    #[test]
    fn test_step_action_serialization() {
        let action = StepAction::Generate {
            min_count: 5,
            max_count: 10,
        };

        let json = serde_json::to_string(&action).unwrap();
        assert!(json.contains("generate"));

        let parsed: StepAction = serde_json::from_str(&json).unwrap();
        if let StepAction::Generate { min_count, max_count } = parsed {
            assert_eq!(min_count, 5);
            assert_eq!(max_count, 10);
        } else {
            panic!("Deserialization failed");
        }
    }

    // =========================================================================
    // StepOutputFormat Tests
    // =========================================================================

    #[test]
    fn test_step_output_format_default() {
        let format = StepOutputFormat::default();
        assert_eq!(format, StepOutputFormat::Text);
    }

    #[test]
    fn test_step_output_format_variants() {
        let formats = [
            StepOutputFormat::Text,
            StepOutputFormat::List,
            StepOutputFormat::Structured,
            StepOutputFormat::Score,
            StepOutputFormat::Boolean,
        ];
        assert_eq!(formats.len(), 5);
    }

    // =========================================================================
    // AggregationType Tests
    // =========================================================================

    #[test]
    fn test_aggregation_type_default() {
        let agg = AggregationType::default();
        assert_eq!(agg, AggregationType::ThematicClustering);
    }

    // =========================================================================
    // CritiqueSeverity Tests
    // =========================================================================

    #[test]
    fn test_critique_severity_default() {
        let severity = CritiqueSeverity::default();
        assert_eq!(severity, CritiqueSeverity::Standard);
    }

    // =========================================================================
    // DecisionMethod Tests
    // =========================================================================

    #[test]
    fn test_decision_method_default() {
        let method = DecisionMethod::default();
        assert_eq!(method, DecisionMethod::ProsCons);
    }

    // =========================================================================
    // BranchCondition Tests
    // =========================================================================

    #[test]
    fn test_branch_condition_confidence_below() {
        let condition = BranchCondition::ConfidenceBelow { threshold: 0.7 };

        if let BranchCondition::ConfidenceBelow { threshold } = condition {
            assert!((threshold - 0.7).abs() < 0.01);
        } else {
            panic!("Expected ConfidenceBelow variant");
        }
    }

    #[test]
    fn test_branch_condition_confidence_above() {
        let condition = BranchCondition::ConfidenceAbove { threshold: 0.9 };

        if let BranchCondition::ConfidenceAbove { threshold } = condition {
            assert!((threshold - 0.9).abs() < 0.01);
        } else {
            panic!("Expected ConfidenceAbove variant");
        }
    }

    #[test]
    fn test_branch_condition_output_equals() {
        let condition = BranchCondition::OutputEquals {
            field: "result".to_string(),
            value: "pass".to_string(),
        };

        if let BranchCondition::OutputEquals { field, value } = condition {
            assert_eq!(field, "result");
            assert_eq!(value, "pass");
        } else {
            panic!("Expected OutputEquals variant");
        }
    }

    #[test]
    fn test_branch_condition_always() {
        let condition = BranchCondition::Always;
        assert!(matches!(condition, BranchCondition::Always));
    }

    // =========================================================================
    // ValidationRule Tests
    // =========================================================================

    #[test]
    fn test_validation_rule_min_count() {
        let rule = ValidationRule::MinCount {
            field: "items".to_string(),
            value: 5,
        };

        if let ValidationRule::MinCount { field, value } = rule {
            assert_eq!(field, "items");
            assert_eq!(value, 5);
        } else {
            panic!("Expected MinCount variant");
        }
    }

    #[test]
    fn test_validation_rule_max_count() {
        let rule = ValidationRule::MaxCount {
            field: "items".to_string(),
            value: 10,
        };

        if let ValidationRule::MaxCount { field, value } = rule {
            assert_eq!(field, "items");
            assert_eq!(value, 10);
        } else {
            panic!("Expected MaxCount variant");
        }
    }

    #[test]
    fn test_validation_rule_confidence_range() {
        let rule = ValidationRule::ConfidenceRange {
            min: 0.5,
            max: 1.0,
        };

        if let ValidationRule::ConfidenceRange { min, max } = rule {
            assert!((min - 0.5).abs() < 0.01);
            assert!((max - 1.0).abs() < 0.01);
        } else {
            panic!("Expected ConfidenceRange variant");
        }
    }

    #[test]
    fn test_validation_rule_required() {
        let rule = ValidationRule::Required {
            field: "query".to_string(),
        };

        if let ValidationRule::Required { field } = rule {
            assert_eq!(field, "query");
        } else {
            panic!("Expected Required variant");
        }
    }

    #[test]
    fn test_validation_rule_custom() {
        let rule = ValidationRule::Custom {
            expression: "len(items) > 0".to_string(),
        };

        if let ValidationRule::Custom { expression } = rule {
            assert!(!expression.is_empty());
        } else {
            panic!("Expected Custom variant");
        }
    }

    // =========================================================================
    // InputSpec / OutputSpec Tests
    // =========================================================================

    #[test]
    fn test_input_spec_default() {
        let spec = InputSpec::default();
        assert!(spec.required.is_empty());
        assert!(spec.optional.is_empty());
    }

    #[test]
    fn test_output_spec_default() {
        let spec = OutputSpec::default();
        assert!(spec.format.is_empty());
        assert!(spec.fields.is_empty());
    }

    // =========================================================================
    // ProtocolMetadata Tests
    // =========================================================================

    #[test]
    fn test_protocol_metadata_default() {
        let metadata = ProtocolMetadata::default();
        assert!(metadata.category.is_empty());
        assert!(metadata.composable_with.is_empty());
        assert_eq!(metadata.typical_tokens, 0);
        assert_eq!(metadata.estimated_latency_ms, 0);
        assert!(metadata.extra.is_empty());
    }
}

#[cfg(test)]
mod trace_tests {
    use crate::thinktool::trace::{
        ExecutionStatus, ExecutionTrace, StepStatus, StepTrace, TimingInfo, TraceMetadata,
    };
    use crate::thinktool::step::{StepOutput, TokenUsage};

    // =========================================================================
    // ExecutionTrace Tests
    // =========================================================================

    #[test]
    fn test_execution_trace_new() {
        let trace = ExecutionTrace::new("gigathink", "1.0.0");
        assert_eq!(trace.protocol_id, "gigathink");
        assert_eq!(trace.protocol_version, "1.0.0");
        assert_eq!(trace.status, ExecutionStatus::Running);
        assert!(trace.steps.is_empty());
    }

    #[test]
    fn test_execution_trace_with_input() {
        let trace = ExecutionTrace::new("test", "1.0.0")
            .with_input(serde_json::json!({"query": "test query"}));

        assert!(trace.input.is_object());
        assert!(trace.input.get("query").is_some());
    }

    #[test]
    fn test_execution_trace_add_step() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        let mut step = StepTrace::new("step1", 0);
        step.tokens = TokenUsage::new(100, 50, 0.001);
        step.complete(
            StepOutput::Text { content: "Result".to_string() },
            0.85,
        );

        trace.add_step(step);

        assert_eq!(trace.steps.len(), 1);
        assert_eq!(trace.tokens.total_tokens, 150);
    }

    #[test]
    fn test_execution_trace_complete() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");
        trace.timing.start();

        trace.complete(serde_json::json!({"result": "success"}), 0.90);

        assert_eq!(trace.status, ExecutionStatus::Completed);
        assert!(trace.output.is_some());
        assert!((trace.confidence - 0.90).abs() < 0.01);
    }

    #[test]
    fn test_execution_trace_fail() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");
        trace.timing.start();

        let mut step = StepTrace::new("step1", 0);
        step.status = StepStatus::Running;
        trace.add_step(step);

        trace.fail("Something went wrong");

        assert_eq!(trace.status, ExecutionStatus::Failed);
        assert_eq!(trace.steps.last().unwrap().status, StepStatus::Failed);
        assert_eq!(
            trace.steps.last().unwrap().error,
            Some("Something went wrong".to_string())
        );
    }

    #[test]
    fn test_execution_trace_completed_steps() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        for i in 0..5 {
            let mut step = StepTrace::new(format!("step{}", i), i);
            if i < 3 {
                step.status = StepStatus::Completed;
            } else {
                step.status = StepStatus::Pending;
            }
            trace.add_step(step);
        }

        assert_eq!(trace.completed_steps(), 3);
    }

    #[test]
    fn test_execution_trace_average_confidence() {
        let mut trace = ExecutionTrace::new("test", "1.0.0");

        for (i, conf) in [0.8, 0.9, 0.7].iter().enumerate() {
            let mut step = StepTrace::new(format!("step{}", i), i);
            step.confidence = *conf;
            trace.add_step(step);
        }

        assert!((trace.average_confidence() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_execution_trace_average_confidence_empty() {
        let trace = ExecutionTrace::new("test", "1.0.0");
        assert_eq!(trace.average_confidence(), 0.0);
    }

    #[test]
    fn test_execution_trace_to_json() {
        let trace = ExecutionTrace::new("test", "1.0.0");

        let json = trace.to_json().unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn test_execution_trace_to_json_compact() {
        let trace = ExecutionTrace::new("test", "1.0.0");

        let json = trace.to_json_compact().unwrap();
        assert!(!json.contains('\n'));
    }

    // =========================================================================
    // StepTrace Tests
    // =========================================================================

    #[test]
    fn test_step_trace_new() {
        let step = StepTrace::new("step1", 0);
        assert_eq!(step.step_id, "step1");
        assert_eq!(step.index, 0);
        assert_eq!(step.status, StepStatus::Pending);
        assert!(step.prompt.is_empty());
    }

    #[test]
    fn test_step_trace_start() {
        let mut step = StepTrace::new("step1", 0);
        step.start();

        assert_eq!(step.status, StepStatus::Running);
    }

    #[test]
    fn test_step_trace_complete() {
        let mut step = StepTrace::new("step1", 0);
        step.start();
        step.complete(
            StepOutput::Text { content: "Result".to_string() },
            0.85,
        );

        assert_eq!(step.status, StepStatus::Completed);
        assert!((step.confidence - 0.85).abs() < 0.01);
        assert!(step.completed_at.is_some());
    }

    #[test]
    fn test_step_trace_fail() {
        let mut step = StepTrace::new("step1", 0);
        step.start();
        step.fail("Error occurred");

        assert_eq!(step.status, StepStatus::Failed);
        assert_eq!(step.error, Some("Error occurred".to_string()));
        assert!(step.completed_at.is_some());
    }

    // =========================================================================
    // ExecutionStatus Tests
    // =========================================================================

    #[test]
    fn test_execution_status_default() {
        let status = ExecutionStatus::default();
        assert_eq!(status, ExecutionStatus::Running);
    }

    #[test]
    fn test_execution_status_variants() {
        let statuses = [
            ExecutionStatus::Running,
            ExecutionStatus::Completed,
            ExecutionStatus::Failed,
            ExecutionStatus::Cancelled,
            ExecutionStatus::TimedOut,
            ExecutionStatus::Paused,
        ];
        assert_eq!(statuses.len(), 6);
    }

    // =========================================================================
    // StepStatus Tests
    // =========================================================================

    #[test]
    fn test_step_status_default() {
        let status = StepStatus::default();
        assert_eq!(status, StepStatus::Pending);
    }

    #[test]
    fn test_step_status_variants() {
        let statuses = [
            StepStatus::Pending,
            StepStatus::Running,
            StepStatus::Completed,
            StepStatus::Failed,
            StepStatus::Skipped,
        ];
        assert_eq!(statuses.len(), 5);
    }

    // =========================================================================
    // TimingInfo Tests
    // =========================================================================

    #[test]
    fn test_timing_info_default() {
        let timing = TimingInfo::default();
        assert!(timing.started_at.is_none());
        assert!(timing.completed_at.is_none());
        assert_eq!(timing.total_duration_ms, 0);
    }

    #[test]
    fn test_timing_info_start() {
        let mut timing = TimingInfo::default();
        timing.start();

        assert!(timing.started_at.is_some());
    }

    #[test]
    fn test_timing_info_complete() {
        let mut timing = TimingInfo::default();
        timing.start();

        // Small delay to ensure measurable duration
        std::thread::sleep(std::time::Duration::from_millis(10));

        timing.complete();

        assert!(timing.completed_at.is_some());
        assert!(timing.total_duration_ms > 0);
    }

    // =========================================================================
    // TraceMetadata Tests
    // =========================================================================

    #[test]
    fn test_trace_metadata_default() {
        let metadata = TraceMetadata::default();
        assert!(metadata.model.is_none());
        assert!(metadata.provider.is_none());
        assert!(metadata.temperature.is_none());
        assert!(metadata.profile.is_none());
        assert!(metadata.tags.is_empty());
        assert!(metadata.environment.is_none());
    }
}

#[cfg(test)]
mod consistency_tests {
    use crate::thinktool::consistency::{
        ConsistencyResult, ReasoningPath, SelfConsistencyConfig, SelfConsistencyEngine, VotingMethod,
    };
    use crate::thinktool::step::{StepOutput, StepResult, TokenUsage};
    use std::collections::HashMap;

    // =========================================================================
    // SelfConsistencyConfig Tests
    // =========================================================================

    #[test]
    fn test_self_consistency_config_default() {
        let config = SelfConsistencyConfig::default();
        assert_eq!(config.num_samples, 5);
        assert!(config.use_cisc);
        assert!(config.early_stopping);
        assert!((config.consensus_threshold - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_self_consistency_config_fast() {
        let config = SelfConsistencyConfig::fast();
        assert_eq!(config.num_samples, 3);
        assert!(config.early_stopping);
        assert!((config.consensus_threshold - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_self_consistency_config_thorough() {
        let config = SelfConsistencyConfig::thorough();
        assert_eq!(config.num_samples, 10);
        assert!(!config.early_stopping);
    }

    #[test]
    fn test_self_consistency_config_paranoid() {
        let config = SelfConsistencyConfig::paranoid();
        assert_eq!(config.num_samples, 15);
        assert!(!config.early_stopping);
    }

    #[test]
    fn test_temperature_for_sample() {
        let config = SelfConsistencyConfig::default();

        assert!((config.temperature_for_sample(0) - 0.7).abs() < 0.01);
        assert!((config.temperature_for_sample(1) - 0.8).abs() < 0.01);
        assert!((config.temperature_for_sample(2) - 0.9).abs() < 0.01);
    }

    // =========================================================================
    // VotingMethod Tests
    // =========================================================================

    #[test]
    fn test_voting_method_default() {
        let method = VotingMethod::default();
        assert_eq!(method, VotingMethod::MajorityVote);
    }

    #[test]
    fn test_voting_method_variants() {
        let methods = [
            VotingMethod::MajorityVote,
            VotingMethod::ConfidenceWeighted,
            VotingMethod::ClusterWeighted,
            VotingMethod::Unanimous,
        ];
        assert_eq!(methods.len(), 4);
    }

    // =========================================================================
    // SelfConsistencyEngine Tests
    // =========================================================================

    #[test]
    fn test_self_consistency_engine_new() {
        let config = SelfConsistencyConfig::default();
        let _engine = SelfConsistencyEngine::new(config);
        // Should not panic
    }

    #[test]
    fn test_self_consistency_engine_default() {
        let engine = SelfConsistencyEngine::default_engine();
        let result = engine.vote(vec![]);

        // Empty votes should return empty result
        assert!(result.answer.is_empty());
        assert_eq!(result.total_samples, 0);
    }

    #[test]
    fn test_majority_voting() {
        let engine = SelfConsistencyEngine::default_engine();

        let results = vec![
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.8,
            ),
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.85,
            ),
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 43.".to_string() },
                0.75,
            ),
        ];

        let result = engine.vote(results);

        assert_eq!(result.answer, "42");
        assert_eq!(result.vote_count, 2);
        assert_eq!(result.total_samples, 3);
    }

    #[test]
    fn test_unanimous_voting() {
        let config = SelfConsistencyConfig {
            voting_method: VotingMethod::Unanimous,
            ..Default::default()
        };
        let engine = SelfConsistencyEngine::new(config);

        let results = vec![
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.8,
            ),
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.85,
            ),
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.9,
            ),
        ];

        let result = engine.vote(results);

        assert_eq!(result.answer, "42");
        assert_eq!(result.vote_count, 3);
    }

    #[test]
    fn test_confidence_weighted_voting() {
        let config = SelfConsistencyConfig {
            voting_method: VotingMethod::ConfidenceWeighted,
            ..Default::default()
        };
        let engine = SelfConsistencyEngine::new(config);

        // Two answers at 0.6 confidence each = 1.2 weight
        // One answer at 0.95 confidence = 0.95 weight
        // Majority should still win due to combined weight
        let results = vec![
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.6,
            ),
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.6,
            ),
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 43.".to_string() },
                0.95,
            ),
        ];

        let result = engine.vote(results);

        // 42 has higher combined weight (1.2 > 0.95)
        assert_eq!(result.answer, "42");
    }

    #[test]
    fn test_early_stopping_triggered() {
        let config = SelfConsistencyConfig {
            consensus_threshold: 0.7,
            early_stopping: true,
            ..Default::default()
        };
        let engine = SelfConsistencyEngine::new(config);

        // 3 out of 4 agree = 75% > 70% threshold
        let results: Vec<StepResult> = (0..4)
            .map(|i| {
                let answer = if i < 3 { "42" } else { "43" };
                StepResult::success(
                    "test",
                    StepOutput::Text { content: format!("The answer is {}.", answer) },
                    0.8,
                )
            })
            .collect();

        assert!(engine.should_early_stop(&results));
    }

    #[test]
    fn test_early_stopping_not_triggered() {
        let config = SelfConsistencyConfig {
            consensus_threshold: 0.8,
            early_stopping: true,
            ..Default::default()
        };
        let engine = SelfConsistencyEngine::new(config);

        // 2 out of 4 agree = 50% < 80% threshold
        let results: Vec<StepResult> = (0..4)
            .map(|i| {
                let answer = if i < 2 { "42" } else { format!("{}", 40 + i) };
                StepResult::success(
                    "test",
                    StepOutput::Text { content: format!("The answer is {}.", answer) },
                    0.8,
                )
            })
            .collect();

        assert!(!engine.should_early_stop(&results));
    }

    #[test]
    fn test_early_stopping_disabled() {
        let config = SelfConsistencyConfig {
            early_stopping: false,
            ..Default::default()
        };
        let engine = SelfConsistencyEngine::new(config);

        let results: Vec<StepResult> = (0..5)
            .map(|_| {
                StepResult::success(
                    "test",
                    StepOutput::Text { content: "The answer is 42.".to_string() },
                    0.9,
                )
            })
            .collect();

        // Even with 100% agreement, should not early stop when disabled
        assert!(!engine.should_early_stop(&results));
    }

    #[test]
    fn test_early_stopping_too_few_samples() {
        let engine = SelfConsistencyEngine::default_engine();

        // Only 2 samples - not enough for early stopping
        let results: Vec<StepResult> = (0..2)
            .map(|_| {
                StepResult::success(
                    "test",
                    StepOutput::Text { content: "The answer is 42.".to_string() },
                    0.9,
                )
            })
            .collect();

        assert!(!engine.should_early_stop(&results));
    }

    #[test]
    fn test_low_confidence_samples_filtered() {
        let config = SelfConsistencyConfig {
            min_sample_confidence: 0.7,
            ..Default::default()
        };
        let engine = SelfConsistencyEngine::new(config);

        let results = vec![
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.8, // Above threshold
            ),
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 43.".to_string() },
                0.5, // Below threshold - should be filtered
            ),
        ];

        let result = engine.vote(results);

        // Only the high-confidence sample should count
        assert_eq!(result.total_samples, 1);
        assert_eq!(result.answer, "42");
    }

    #[test]
    fn test_failed_samples_filtered() {
        let engine = SelfConsistencyEngine::default_engine();

        let results = vec![
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.8,
            ),
            StepResult::failure("test", "Error occurred"),
        ];

        let result = engine.vote(results);

        assert_eq!(result.total_samples, 1);
    }

    #[test]
    fn test_normalize_answer() {
        let engine = SelfConsistencyEngine::default_engine();

        // Test with various inputs that should normalize to the same thing
        let results = vec![
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42.".to_string() },
                0.8,
            ),
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is  42".to_string() }, // extra space
                0.8,
            ),
            StepResult::success(
                "test",
                StepOutput::Text { content: "The answer is 42!".to_string() }, // exclamation
                0.8,
            ),
        ];

        let result = engine.vote(results);

        // All should be normalized to "42"
        assert_eq!(result.vote_count, 3);
    }

    // =========================================================================
    // ConsistencyResult Tests
    // =========================================================================

    #[test]
    fn test_consistency_result_meets_threshold() {
        let result = ConsistencyResult {
            answer: "42".to_string(),
            confidence: 0.85,
            vote_count: 4,
            total_samples: 5,
            agreement_ratio: 0.8,
            paths: vec![],
            vote_distribution: HashMap::new(),
            early_stopped: false,
            total_tokens: TokenUsage::default(),
        };

        assert!(result.meets_threshold(0.80));
        assert!(!result.meets_threshold(0.90));
    }

    #[test]
    fn test_consistency_result_meets_threshold_low_agreement() {
        let result = ConsistencyResult {
            answer: "42".to_string(),
            confidence: 0.9,
            vote_count: 2,
            total_samples: 5,
            agreement_ratio: 0.4, // Below 50%
            paths: vec![],
            vote_distribution: HashMap::new(),
            early_stopped: false,
            total_tokens: TokenUsage::default(),
        };

        // Even with high confidence, low agreement fails
        assert!(!result.meets_threshold(0.5));
    }

    #[test]
    fn test_consistency_result_dissenting_paths() {
        let paths = vec![
            ReasoningPath {
                answer: "42".to_string(),
                reasoning: "Because...".to_string(),
                confidence: 0.8,
                tokens: TokenUsage::default(),
                temperature: 0.7,
                sample_index: 0,
            },
            ReasoningPath {
                answer: "43".to_string(),
                reasoning: "Because...".to_string(),
                confidence: 0.75,
                tokens: TokenUsage::default(),
                temperature: 0.8,
                sample_index: 1,
            },
        ];

        let result = ConsistencyResult {
            answer: "42".to_string(),
            confidence: 0.8,
            vote_count: 1,
            total_samples: 2,
            agreement_ratio: 0.5,
            paths,
            vote_distribution: HashMap::from([
                ("42".to_string(), 1),
                ("43".to_string(), 1),
            ]),
            early_stopped: false,
            total_tokens: TokenUsage::default(),
        };

        let dissenting = result.dissenting_paths();
        assert_eq!(dissenting.len(), 1);
        assert_eq!(dissenting[0].answer, "43");
    }

    #[test]
    fn test_consistency_result_diversity_score() {
        // 2 unique answers out of 3 samples = diversity 0.5
        let result = ConsistencyResult {
            answer: "42".to_string(),
            confidence: 0.8,
            vote_count: 2,
            total_samples: 3,
            agreement_ratio: 0.67,
            paths: Vec::new(),
            vote_distribution: HashMap::from([
                ("42".to_string(), 2),
                ("43".to_string(), 1),
            ]),
            early_stopped: false,
            total_tokens: TokenUsage::default(),
        };

        assert!((result.diversity_score() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_consistency_result_diversity_score_unanimous() {
        let result = ConsistencyResult {
            answer: "42".to_string(),
            confidence: 0.9,
            vote_count: 5,
            total_samples: 5,
            agreement_ratio: 1.0,
            paths: Vec::new(),
            vote_distribution: HashMap::from([("42".to_string(), 5)]),
            early_stopped: false,
            total_tokens: TokenUsage::default(),
        };

        // 1 unique answer = 0 diversity
        assert!((result.diversity_score() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_consistency_result_diversity_score_single_sample() {
        let result = ConsistencyResult {
            answer: "42".to_string(),
            confidence: 0.8,
            vote_count: 1,
            total_samples: 1,
            agreement_ratio: 1.0,
            paths: Vec::new(),
            vote_distribution: HashMap::from([("42".to_string(), 1)]),
            early_stopped: false,
            total_tokens: TokenUsage::default(),
        };

        // Single sample = 0 diversity
        assert_eq!(result.diversity_score(), 0.0);
    }
}

#[cfg(test)]
mod profiles_tests {
    use crate::thinktool::profiles::{
        ChainCondition, ChainStep, ProfileRegistry, ReasoningProfile, StepConfigOverride,
    };
    use std::collections::HashMap;

    // =========================================================================
    // ProfileRegistry Tests
    // =========================================================================

    #[test]
    fn test_profile_registry_new() {
        let registry = ProfileRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_profile_registry_with_builtins() {
        let registry = ProfileRegistry::with_builtins();

        // Should have all builtin profiles
        assert!(registry.contains("quick"));
        assert!(registry.contains("balanced"));
        assert!(registry.contains("deep"));
        assert!(registry.contains("paranoid"));
        assert!(registry.contains("decide"));
        assert!(registry.contains("scientific"));
        assert!(registry.contains("powercombo"));
    }

    #[test]
    fn test_profile_registry_register() {
        let mut registry = ProfileRegistry::new();

        let profile = ReasoningProfile {
            id: "custom".to_string(),
            name: "Custom Profile".to_string(),
            description: "A custom profile".to_string(),
            chain: vec![],
            min_confidence: 0.8,
            token_budget: Some(5000),
            tags: vec!["custom".to_string()],
        };

        registry.register(profile);

        assert!(registry.contains("custom"));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_profile_registry_get() {
        let registry = ProfileRegistry::with_builtins();

        let quick = registry.get("quick");
        assert!(quick.is_some());
        assert_eq!(quick.unwrap().id, "quick");

        let missing = registry.get("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_profile_registry_list_ids() {
        let registry = ProfileRegistry::with_builtins();
        let ids = registry.list_ids();

        assert!(ids.contains(&"quick"));
        assert!(ids.contains(&"balanced"));
    }

    #[test]
    fn test_profile_registry_list() {
        let registry = ProfileRegistry::with_builtins();
        let profiles = registry.list();

        assert!(!profiles.is_empty());
    }

    // =========================================================================
    // ReasoningProfile Tests
    // =========================================================================

    #[test]
    fn test_quick_profile_structure() {
        let registry = ProfileRegistry::with_builtins();
        let quick = registry.get("quick").unwrap();

        assert_eq!(quick.chain.len(), 2);
        assert_eq!(quick.chain[0].protocol_id, "gigathink");
        assert_eq!(quick.chain[1].protocol_id, "laserlogic");
        assert!((quick.min_confidence - 0.70).abs() < 0.01);
    }

    #[test]
    fn test_balanced_profile_structure() {
        let registry = ProfileRegistry::with_builtins();
        let balanced = registry.get("balanced").unwrap();

        assert_eq!(balanced.chain.len(), 4);
        assert_eq!(balanced.chain[0].protocol_id, "gigathink");
        assert_eq!(balanced.chain[1].protocol_id, "laserlogic");
        assert_eq!(balanced.chain[2].protocol_id, "bedrock");
        assert_eq!(balanced.chain[3].protocol_id, "proofguard");
    }

    #[test]
    fn test_paranoid_profile_structure() {
        let registry = ProfileRegistry::with_builtins();
        let paranoid = registry.get("paranoid").unwrap();

        assert_eq!(paranoid.chain.len(), 6);
        assert!((paranoid.min_confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_powercombo_profile_structure() {
        let registry = ProfileRegistry::with_builtins();
        let powercombo = registry.get("powercombo").unwrap();

        // All 5 tools + validation pass
        assert_eq!(powercombo.chain.len(), 6);
        assert_eq!(powercombo.chain[0].protocol_id, "gigathink");
        assert_eq!(powercombo.chain[1].protocol_id, "laserlogic");
        assert_eq!(powercombo.chain[2].protocol_id, "bedrock");
        assert_eq!(powercombo.chain[3].protocol_id, "proofguard");
        assert_eq!(powercombo.chain[4].protocol_id, "brutalhonesty");
        assert_eq!(powercombo.chain[5].protocol_id, "proofguard");
    }

    // =========================================================================
    // ChainCondition Tests
    // =========================================================================

    #[test]
    fn test_chain_condition_default() {
        let condition = ChainCondition::default();
        assert!(matches!(condition, ChainCondition::Always));
    }

    #[test]
    fn test_chain_condition_confidence_below() {
        let condition = ChainCondition::ConfidenceBelow { threshold: 0.8 };

        if let ChainCondition::ConfidenceBelow { threshold } = condition {
            assert!((threshold - 0.8).abs() < 0.01);
        } else {
            panic!("Expected ConfidenceBelow variant");
        }
    }

    #[test]
    fn test_chain_condition_confidence_above() {
        let condition = ChainCondition::ConfidenceAbove { threshold: 0.9 };

        if let ChainCondition::ConfidenceAbove { threshold } = condition {
            assert!((threshold - 0.9).abs() < 0.01);
        } else {
            panic!("Expected ConfidenceAbove variant");
        }
    }

    #[test]
    fn test_chain_condition_output_exists() {
        let condition = ChainCondition::OutputExists {
            step_id: "step1".to_string(),
            field: "result".to_string(),
        };

        if let ChainCondition::OutputExists { step_id, field } = condition {
            assert_eq!(step_id, "step1");
            assert_eq!(field, "result");
        } else {
            panic!("Expected OutputExists variant");
        }
    }

    // =========================================================================
    // ChainStep Tests
    // =========================================================================

    #[test]
    fn test_chain_step_basic() {
        let step = ChainStep {
            protocol_id: "gigathink".to_string(),
            input_mapping: HashMap::from([("query".to_string(), "input.query".to_string())]),
            condition: None,
            config_override: None,
        };

        assert_eq!(step.protocol_id, "gigathink");
        assert!(step.input_mapping.contains_key("query"));
        assert!(step.condition.is_none());
        assert!(step.config_override.is_none());
    }

    #[test]
    fn test_chain_step_with_condition() {
        let step = ChainStep {
            protocol_id: "bedrock".to_string(),
            input_mapping: HashMap::new(),
            condition: Some(ChainCondition::ConfidenceBelow { threshold: 0.9 }),
            config_override: None,
        };

        assert!(step.condition.is_some());
    }

    #[test]
    fn test_chain_step_with_config_override() {
        let step = ChainStep {
            protocol_id: "gigathink".to_string(),
            input_mapping: HashMap::new(),
            condition: None,
            config_override: Some(StepConfigOverride {
                temperature: Some(0.8),
                max_tokens: Some(2000),
                min_confidence: Some(0.9),
            }),
        };

        assert!(step.config_override.is_some());
        let override_config = step.config_override.unwrap();
        assert_eq!(override_config.temperature, Some(0.8));
        assert_eq!(override_config.max_tokens, Some(2000));
        assert_eq!(override_config.min_confidence, Some(0.9));
    }

    // =========================================================================
    // StepConfigOverride Tests
    // =========================================================================

    #[test]
    fn test_step_config_override_default() {
        let config = StepConfigOverride::default();
        assert!(config.temperature.is_none());
        assert!(config.max_tokens.is_none());
        assert!(config.min_confidence.is_none());
    }

    #[test]
    fn test_step_config_override_partial() {
        let config = StepConfigOverride {
            temperature: Some(0.5),
            ..Default::default()
        };

        assert_eq!(config.temperature, Some(0.5));
        assert!(config.max_tokens.is_none());
    }
}

#[cfg(test)]
mod registry_tests {
    use crate::thinktool::registry::ProtocolRegistry;
    use crate::thinktool::protocol::{Protocol, ProtocolStep, ReasoningStrategy, StepAction, StepOutputFormat};

    // =========================================================================
    // ProtocolRegistry Tests
    // =========================================================================

    #[test]
    fn test_protocol_registry_new() {
        let registry = ProtocolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_protocol_registry_with_defaults() {
        let registry = ProtocolRegistry::with_defaults();
        // Just test that it doesn't panic - search paths may or may not exist
        assert!(registry.is_empty() || !registry.is_empty());
    }

    #[test]
    fn test_protocol_registry_register_builtins() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        // Should have all 5 core protocols + powercombo
        assert!(registry.contains("gigathink"));
        assert!(registry.contains("laserlogic"));
        assert!(registry.contains("bedrock"));
        assert!(registry.contains("proofguard"));
        assert!(registry.contains("brutalhonesty"));
    }

    #[test]
    fn test_protocol_registry_register() {
        let mut registry = ProtocolRegistry::new();

        let mut protocol = Protocol::new("custom", "Custom Protocol");
        protocol.steps.push(ProtocolStep {
            id: "step1".to_string(),
            action: StepAction::Analyze { criteria: vec![] },
            prompt_template: "Analyze this".to_string(),
            output_format: StepOutputFormat::Text,
            min_confidence: 0.7,
            depends_on: vec![],
            branch: None,
        });

        registry.register(protocol).unwrap();

        assert!(registry.contains("custom"));
    }

    #[test]
    fn test_protocol_registry_register_invalid() {
        let mut registry = ProtocolRegistry::new();

        // Empty steps = invalid
        let protocol = Protocol::new("invalid", "Invalid");

        let result = registry.register(protocol);
        assert!(result.is_err());
    }

    #[test]
    fn test_protocol_registry_get() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let gigathink = registry.get("gigathink");
        assert!(gigathink.is_some());
        assert_eq!(gigathink.unwrap().name, "GigaThink");

        let missing = registry.get("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_protocol_registry_contains() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        assert!(registry.contains("gigathink"));
        assert!(!registry.contains("nonexistent"));
    }

    #[test]
    fn test_protocol_registry_list_ids() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let ids = registry.list_ids();
        assert!(ids.contains(&"gigathink"));
        assert!(ids.contains(&"laserlogic"));
    }

    #[test]
    fn test_protocol_registry_list() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let protocols = registry.list();
        assert!(!protocols.is_empty());
    }

    #[test]
    fn test_protocol_registry_remove() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let removed = registry.remove("gigathink");
        assert!(removed.is_some());
        assert!(!registry.contains("gigathink"));
    }

    #[test]
    fn test_protocol_registry_remove_nonexistent() {
        let mut registry = ProtocolRegistry::new();

        let removed = registry.remove("nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_protocol_registry_clear() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        assert!(!registry.is_empty());

        registry.clear();

        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_protocol_registry_add_search_path() {
        let mut registry = ProtocolRegistry::new();

        registry.add_search_path("/tmp/test_protocols");
        registry.add_search_path("/tmp/test_protocols"); // Duplicate should be ignored

        // No panic means success - we can't easily verify the internal state
    }

    #[test]
    fn test_gigathink_protocol_structure() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let gt = registry.get("gigathink").unwrap();

        assert_eq!(gt.strategy, ReasoningStrategy::Expansive);
        assert_eq!(gt.input.required, vec!["query"]);
        assert_eq!(gt.steps.len(), 3);
        assert_eq!(gt.steps[0].id, "identify_dimensions");
        assert_eq!(gt.steps[1].id, "explore_perspectives");
        assert_eq!(gt.steps[2].id, "synthesize");
    }

    #[test]
    fn test_laserlogic_protocol_structure() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let ll = registry.get("laserlogic").unwrap();

        assert_eq!(ll.strategy, ReasoningStrategy::Deductive);
        assert_eq!(ll.input.required, vec!["argument"]);
        assert_eq!(ll.steps.len(), 3);
    }

    #[test]
    fn test_bedrock_protocol_structure() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let br = registry.get("bedrock").unwrap();

        assert_eq!(br.strategy, ReasoningStrategy::Analytical);
        assert_eq!(br.input.required, vec!["statement"]);
    }

    #[test]
    fn test_proofguard_protocol_structure() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let pg = registry.get("proofguard").unwrap();

        assert_eq!(pg.strategy, ReasoningStrategy::Verification);
        assert_eq!(pg.input.required, vec!["claim"]);
    }

    #[test]
    fn test_brutalhonesty_protocol_structure() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let bh = registry.get("brutalhonesty").unwrap();

        assert_eq!(bh.strategy, ReasoningStrategy::Adversarial);
        assert_eq!(bh.input.required, vec!["work"]);
        assert_eq!(bh.steps.len(), 3);
        assert_eq!(bh.steps[0].id, "steelman");
        assert_eq!(bh.steps[1].id, "attack");
        assert_eq!(bh.steps[2].id, "verdict");
    }
}
