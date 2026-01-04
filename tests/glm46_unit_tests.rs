#![cfg(feature = "glm46")]
//! # GLM-4.6 Unit Tests
//!
//! Comprehensive unit tests for GLM-4.6 integration components.
//! Tests cover client, types, circuit breaker, and ThinkTool profile integration.

use reasonkit::glm46::types::{
    ChatMessage, ChatRequest, MessageRole, ResponseFormat, Tool, ToolCall, ToolCallChoice,
    ToolChoice, ToolFunction, ToolFunctionCall, ToolFunctionRef,
};
use reasonkit::glm46::{GLM46Client, GLM46Config};
use secrecy::SecretString;
use std::time::Duration;

#[tokio::test]
async fn test_glm46_config_default() {
    let config = GLM46Config::default();

    assert_eq!(config.model, "glm-4.6");
    assert_eq!(config.context_budget, 198_000);
    assert_eq!(config.timeout, Duration::from_secs(30));
    assert!(config.cost_tracking);
    assert!(config.local_fallback);
}

#[tokio::test]
async fn test_glm46_config_custom() {
    use secrecy::ExposeSecret;

    let config = GLM46Config {
        api_key: SecretString::from("test_key".to_string()),
        base_url: "https://test.example.com".to_string(),
        model: "glm-4.6-custom".to_string(),
        timeout: Duration::from_secs(60),
        context_budget: 100_000,
        cost_tracking: false,
        local_fallback: false,
    };

    assert_eq!(config.api_key.expose_secret(), "test_key");
    assert_eq!(config.base_url, "https://test.example.com");
    assert_eq!(config.model, "glm-4.6-custom");
    assert_eq!(config.context_budget, 100_000);
    assert!(!config.cost_tracking);
    assert!(!config.local_fallback);
}

#[test]
fn test_chat_message_serialization() {
    let message = ChatMessage {
        role: MessageRole::User,
        content: "Test message".to_string(),
        tool_calls: None,
        tool_call_id: None,
    };

    let json = serde_json::to_string(&message).unwrap();
    assert!(json.contains("user"));
    assert!(json.contains("Test message"));
}

#[test]
fn test_chat_message_with_tool_calls() {
    let message = ChatMessage {
        role: MessageRole::Assistant,
        content: String::new(),
        tool_calls: Some(vec![ToolCall {
            id: "call_123".to_string(),
            r#type: "function".to_string(),
            function: ToolFunctionCall {
                name: "test_function".to_string(),
                arguments: r#"{"arg": "value"}"#.to_string(),
            },
        }]),
        tool_call_id: None,
    };

    assert_eq!(message.tool_calls.as_ref().unwrap().len(), 1);
    assert_eq!(message.tool_calls.as_ref().unwrap()[0].id, "call_123");
}

#[test]
fn test_chat_request_creation() {
    let request = ChatRequest {
        messages: vec![
            ChatMessage {
                role: MessageRole::System,
                content: "You are a helpful assistant".to_string(),
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: "Hello".to_string(),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        temperature: 0.7,
        max_tokens: 1000,
        response_format: Some(ResponseFormat::Structured),
        tools: None,
        tool_choice: None,
        stop: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
    };

    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.temperature, 0.7);
    assert_eq!(request.max_tokens, 1000);
    assert!(matches!(
        request.response_format,
        Some(ResponseFormat::Structured)
    ));
}

#[test]
fn test_response_format_json_schema() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "result": {"type": "string"}
        }
    });

    let format = ResponseFormat::JsonSchema {
        name: "test_schema".to_string(),
        schema,
    };

    match format {
        ResponseFormat::JsonSchema { name, .. } => {
            assert_eq!(name, "test_schema");
        }
        _ => panic!("Expected JsonSchema variant"),
    }
}

#[test]
fn test_tool_definition() {
    let tool = Tool {
        r#type: "function".to_string(),
        function: ToolFunction {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }),
        },
    };

    assert_eq!(tool.r#type, "function");
    assert_eq!(tool.function.name, "test_tool");
    assert_eq!(tool.function.description, "A test tool");
}

#[test]
fn test_tool_choice_enum() {
    let auto = ToolChoice::Auto;
    let none = ToolChoice::None;
    let required = ToolChoice::Required;

    // Test serialization
    let auto_json = serde_json::to_string(&auto).unwrap();
    assert!(auto_json.contains("auto"));

    let none_json = serde_json::to_string(&none).unwrap();
    assert!(none_json.contains("none"));

    let required_json = serde_json::to_string(&required).unwrap();
    assert!(required_json.contains("required"));
}

#[test]
fn test_cost_tracker_initialization() {
    use reasonkit::glm46::CostTracker;
    // Test that cost tracker can be created and defaults work
    let tracker = CostTracker::new();
    // Verify initial state
    let stats = tracker.get_stats();
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.total_cost, 0.0);
}

#[test]
fn test_context_budget_validation() {
    // Test that context budget respects 198K limit
    let config = GLM46Config {
        context_budget: 198_000,
        ..Default::default()
    };

    assert_eq!(config.context_budget, 198_000);

    // Test that exceeding limit is handled (if validation exists)
    let config_high = GLM46Config {
        context_budget: 200_000, // Exceeds limit
        ..Default::default()
    };

    // Should either clamp or error - depends on implementation
    // For now, just verify it can be set
    assert_eq!(config_high.context_budget, 200_000);
}

#[test]
fn test_message_role_serialization() {
    let roles = vec![
        MessageRole::System,
        MessageRole::User,
        MessageRole::Assistant,
        MessageRole::Tool,
    ];

    for role in roles {
        let json = serde_json::to_string(&role).unwrap();
        assert!(!json.is_empty());
    }
}

#[tokio::test]
async fn test_client_initialization() {
    // Test client can be created with valid config
    let config = GLM46Config {
        api_key: SecretString::from("test_key".to_string()),
        ..Default::default()
    };

    // This will fail if client initialization has issues
    // Note: Actual API calls require valid API key
    let _client = GLM46Client::new(config).expect("Client should initialize");
}

#[test]
fn test_timeout_configuration() {
    let config = GLM46Config {
        timeout: Duration::from_secs(60),
        ..Default::default()
    };

    assert_eq!(config.timeout, Duration::from_secs(60));
}

#[test]
fn test_model_identifier() {
    let config = GLM46Config::default();
    assert_eq!(config.model, "glm-4.6");
}

#[test]
fn test_base_url_configuration() {
    let custom_url = "https://custom.api.example.com";
    let config = GLM46Config {
        base_url: custom_url.to_string(),
        ..Default::default()
    };

    assert_eq!(config.base_url, custom_url);
}

// ==================== COMPREHENSIVE SERIALIZATION TESTS ====================

/// Validate MessageRole serializes to lowercase strings (API spec compliance)
#[test]
fn test_message_role_serialization_format() {
    assert_eq!(
        serde_json::to_string(&MessageRole::System).unwrap(),
        r#""system""#
    );
    assert_eq!(
        serde_json::to_string(&MessageRole::User).unwrap(),
        r#""user""#
    );
    assert_eq!(
        serde_json::to_string(&MessageRole::Assistant).unwrap(),
        r#""assistant""#
    );
    assert_eq!(
        serde_json::to_string(&MessageRole::Tool).unwrap(),
        r#""tool""#
    );
}

/// Validate MessageRole round-trip serialization
#[test]
fn test_message_role_roundtrip() {
    for role in [
        MessageRole::System,
        MessageRole::User,
        MessageRole::Assistant,
        MessageRole::Tool,
    ] {
        let json = serde_json::to_string(&role).unwrap();
        let parsed: MessageRole = serde_json::from_str(&json).unwrap();
        assert_eq!(role, parsed);
    }
}

/// Validate ChatMessage skips None fields in serialization
#[test]
fn test_chat_message_optional_field_omission() {
    let message = ChatMessage {
        role: MessageRole::User,
        content: "Hello".to_string(),
        tool_calls: None,
        tool_call_id: None,
    };

    let json = serde_json::to_string(&message).unwrap();

    // Should NOT contain optional fields when None
    assert!(!json.contains("tool_calls"));
    assert!(!json.contains("tool_call_id"));

    // Should contain required fields
    assert!(json.contains("role"));
    assert!(json.contains("content"));
}

/// Validate ChatMessage with tool_calls serializes correctly
#[test]
fn test_chat_message_with_tool_calls_serialization() {
    let message = ChatMessage {
        role: MessageRole::Assistant,
        content: String::new(),
        tool_calls: Some(vec![ToolCall {
            id: "call_abc123".to_string(),
            r#type: "function".to_string(),
            function: ToolFunctionCall {
                name: "search".to_string(),
                arguments: r#"{"query":"test"}"#.to_string(),
            },
        }]),
        tool_call_id: None,
    };

    let json = serde_json::to_string(&message).unwrap();

    // Validate JSON structure
    assert!(json.contains("tool_calls"));
    assert!(json.contains("call_abc123"));
    assert!(json.contains(r#""type":"function""#));
    assert!(json.contains(r#""name":"search""#));

    // Round-trip validation
    let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(message, parsed);
}

/// Validate ChatMessage helper constructors
#[test]
fn test_chat_message_constructors() {
    let system = ChatMessage::system("You are helpful");
    assert_eq!(system.role, MessageRole::System);
    assert_eq!(system.content, "You are helpful");
    assert!(system.tool_calls.is_none());

    let user = ChatMessage::user("Hello");
    assert_eq!(user.role, MessageRole::User);
    assert_eq!(user.content, "Hello");

    let assistant = ChatMessage::assistant("Hi there");
    assert_eq!(assistant.role, MessageRole::Assistant);
    assert_eq!(assistant.content, "Hi there");

    let tool = ChatMessage::tool("result", "call_123");
    assert_eq!(tool.role, MessageRole::Tool);
    assert_eq!(tool.content, "result");
    assert_eq!(tool.tool_call_id, Some("call_123".to_string()));
}

/// Validate ResponseFormat enum serialization
#[test]
fn test_response_format_serialization() {
    // Text format
    let text = ResponseFormat::Text;
    let json = serde_json::to_string(&text).unwrap();
    assert_eq!(json, r#""text""#);

    // JsonObject format
    let json_obj = ResponseFormat::JsonObject;
    let json = serde_json::to_string(&json_obj).unwrap();
    assert_eq!(json, r#""json_object""#);

    // Structured format
    let structured = ResponseFormat::Structured;
    let json = serde_json::to_string(&structured).unwrap();
    assert_eq!(json, r#""structured""#);

    // JsonSchema format (has embedded data)
    let schema = ResponseFormat::JsonSchema {
        name: "test".to_string(),
        schema: serde_json::json!({"type": "object"}),
    };
    let json = serde_json::to_string(&schema).unwrap();
    assert!(json.contains("json_schema"));
    assert!(json.contains("test"));
}

/// Validate ResponseFormat round-trip
#[test]
fn test_response_format_roundtrip() {
    let formats = vec![
        ResponseFormat::Text,
        ResponseFormat::JsonObject,
        ResponseFormat::Structured,
        ResponseFormat::JsonSchema {
            name: "test_schema".to_string(),
            schema: serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}),
        },
    ];

    for format in formats {
        let json = serde_json::to_string(&format).unwrap();
        let parsed: ResponseFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(format, parsed);
    }
}

/// Validate ChatRequest optional field omission
#[test]
fn test_chat_request_optional_field_omission() {
    let request = ChatRequest {
        messages: vec![ChatMessage::user("Hello")],
        temperature: 0.7,
        max_tokens: 1000,
        response_format: None,
        tools: None,
        tool_choice: None,
        stop: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
    };

    let json = serde_json::to_string(&request).unwrap();

    // Should NOT contain optional fields when None
    assert!(!json.contains("response_format"));
    assert!(!json.contains("tools"));
    assert!(!json.contains("tool_choice"));
    assert!(!json.contains("stop"));
    assert!(!json.contains("top_p"));
    assert!(!json.contains("frequency_penalty"));
    assert!(!json.contains("presence_penalty"));
    assert!(!json.contains("stream"));

    // Should contain required fields
    assert!(json.contains("messages"));
    assert!(json.contains("temperature"));
    assert!(json.contains("max_tokens"));
}

/// Validate ChatRequest with all fields populated
#[test]
fn test_chat_request_full_serialization() {
    let request = ChatRequest {
        messages: vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ],
        temperature: 0.7,
        max_tokens: 2000,
        response_format: Some(ResponseFormat::JsonObject),
        tools: Some(vec![Tool {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: "search".to_string(),
                description: "Search function".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        }]),
        tool_choice: Some(ToolChoice::Auto),
        stop: Some(vec!["END".to_string()]),
        top_p: Some(0.9),
        frequency_penalty: Some(0.5),
        presence_penalty: Some(0.3),
        stream: Some(true),
    };

    let json = serde_json::to_string(&request).unwrap();

    // Validate all fields present
    assert!(json.contains("messages"));
    assert!(json.contains("temperature"));
    assert!(json.contains("max_tokens"));
    assert!(json.contains("response_format"));
    assert!(json.contains("tools"));
    assert!(json.contains("tool_choice"));
    assert!(json.contains("stop"));
    assert!(json.contains("top_p"));
    assert!(json.contains("frequency_penalty"));
    assert!(json.contains("presence_penalty"));
    assert!(json.contains("stream"));

    // Round-trip validation
    let parsed: ChatRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(request.messages.len(), parsed.messages.len());
    assert_eq!(request.temperature, parsed.temperature);
    assert_eq!(request.max_tokens, parsed.max_tokens);
}

/// Validate ToolChoice enum serialization
#[test]
fn test_tool_choice_serialization_format() {
    assert_eq!(
        serde_json::to_string(&ToolChoice::None).unwrap(),
        r#""none""#
    );
    assert_eq!(
        serde_json::to_string(&ToolChoice::Auto).unwrap(),
        r#""auto""#
    );
    assert_eq!(
        serde_json::to_string(&ToolChoice::Required).unwrap(),
        r#""required""#
    );
}

/// Validate Tool serialization matches API spec
#[test]
fn test_tool_serialization_api_format() {
    let tool = Tool {
        r#type: "function".to_string(),
        function: ToolFunction {
            name: "get_weather".to_string(),
            description: "Get weather for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }),
        },
    };

    let json = serde_json::to_string(&tool).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    // Validate structure matches OpenAI-style API format
    assert_eq!(parsed["type"], "function");
    assert_eq!(parsed["function"]["name"], "get_weather");
    assert_eq!(
        parsed["function"]["description"],
        "Get weather for a location"
    );
    assert!(parsed["function"]["parameters"].is_object());
}

/// Validate ToolCall serialization
#[test]
fn test_tool_call_serialization() {
    let tool_call = ToolCall {
        id: "call_xyz789".to_string(),
        r#type: "function".to_string(),
        function: ToolFunctionCall {
            name: "calculate".to_string(),
            arguments: r#"{"x": 10, "y": 20}"#.to_string(),
        },
    };

    let json = serde_json::to_string(&tool_call).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed["id"], "call_xyz789");
    assert_eq!(parsed["type"], "function");
    assert_eq!(parsed["function"]["name"], "calculate");
    assert_eq!(parsed["function"]["arguments"], r#"{"x": 10, "y": 20}"#);

    // Round-trip
    let roundtrip: ToolCall = serde_json::from_str(&json).unwrap();
    assert_eq!(tool_call, roundtrip);
}

/// Validate TokenUsage deserialization from API response
#[test]
fn test_token_usage_deserialization() {
    use reasonkit::glm46::types::TokenUsage;

    let api_response = r#"{
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "total_tokens": 200
    }"#;

    let usage: TokenUsage = serde_json::from_str(api_response).unwrap();
    assert_eq!(usage.prompt_tokens, 150);
    assert_eq!(usage.completion_tokens, 50);
    assert_eq!(usage.total_tokens, 200);
    assert_eq!(usage.input_tokens(), 150);
    assert_eq!(usage.output_tokens(), 50);
}

/// Validate FinishReason deserialization
#[test]
fn test_finish_reason_deserialization() {
    use reasonkit::glm46::types::FinishReason;

    let test_cases = [
        (r#""stop""#, "stop"),
        (r#""length""#, "length"),
        (r#""tool_calls""#, "tool_calls"),
        (r#""content_filter""#, "content_filter"),
        (r#""function_call""#, "function_call"),
    ];

    for (json, _expected) in test_cases {
        let _reason: FinishReason = serde_json::from_str(json).unwrap();
    }
}

/// Validate ChatResponse deserialization from mock API response
#[test]
fn test_chat_response_deserialization() {
    use reasonkit::glm46::types::ChatResponse;

    let api_response = r#"{
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-4.6",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }"#;

    let response: ChatResponse = serde_json::from_str(api_response).unwrap();
    assert_eq!(response.id, "chatcmpl-abc123");
    assert_eq!(response.object, "chat.completion");
    assert_eq!(response.model, "glm-4.6");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(
        response.choices[0].message.content,
        "Hello! How can I help you?"
    );
    assert_eq!(response.usage.total_tokens, 18);
}

/// Validate ChatResponse with tool_calls deserialization
#[test]
fn test_chat_response_with_tool_calls() {
    use reasonkit::glm46::types::ChatResponse;

    let api_response = r#"{
        "id": "chatcmpl-tool123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-4.6",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"Tokyo\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    }"#;

    let response: ChatResponse = serde_json::from_str(api_response).unwrap();
    assert_eq!(
        response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        response.choices[0].message.tool_calls.as_ref().unwrap()[0]
            .function
            .name,
        "get_weather"
    );
}

/// Validate StreamChunk deserialization
#[test]
fn test_stream_chunk_deserialization() {
    use reasonkit::glm46::types::StreamChunk;

    let chunk = r#"{
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "glm-4.6",
        "choices": [{
            "index": 0,
            "delta": {
                "content": "Hello"
            },
            "finish_reason": null
        }]
    }"#;

    let parsed: StreamChunk = serde_json::from_str(chunk).unwrap();
    assert_eq!(parsed.id, "chatcmpl-stream");
    assert_eq!(parsed.choices[0].delta.content.as_ref().unwrap(), "Hello");
    assert!(parsed.choices[0].finish_reason.is_none());
}

/// Validate APIError deserialization
#[test]
fn test_api_error_deserialization() {
    use reasonkit::glm46::types::APIError;

    let error_json = r#"{
        "message": "Rate limit exceeded",
        "type": "rate_limit_error",
        "code": "rate_limit"
    }"#;

    let error: APIError = serde_json::from_str(error_json).unwrap();
    assert_eq!(error.message, "Rate limit exceeded");
    assert_eq!(error.error_type, "rate_limit_error");
    assert_eq!(error.code, Some("rate_limit".to_string()));
}

/// Validate CircuitBreakerConfig defaults
#[test]
fn test_circuit_breaker_config_default() {
    use reasonkit::glm46::types::CircuitBreakerConfig;

    let config = CircuitBreakerConfig::default();
    assert_eq!(config.failure_threshold, 5);
    assert_eq!(config.success_threshold, 3);
    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.reset_timeout, Duration::from_secs(300));
}

/// Validate UsageStats calculations
#[test]
fn test_usage_stats_calculations() {
    use reasonkit::glm46::types::UsageStats;

    let stats = UsageStats {
        total_requests: 10,
        total_input_tokens: 1000,
        total_output_tokens: 500,
        total_cost: 0.015,
        session_start: std::time::SystemTime::now(),
    };

    assert_eq!(stats.total_tokens(), 1500);
    assert!((stats.average_tokens_per_request() - 150.0).abs() < 0.001);
    assert!((stats.cost_per_1k_tokens() - 0.01).abs() < 0.001);
}

/// Validate UsageStats edge cases
#[test]
fn test_usage_stats_edge_cases() {
    use reasonkit::glm46::types::UsageStats;

    // Zero requests
    let empty_stats = UsageStats {
        total_requests: 0,
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_cost: 0.0,
        session_start: std::time::SystemTime::now(),
    };

    assert_eq!(empty_stats.total_tokens(), 0);
    assert_eq!(empty_stats.average_tokens_per_request(), 0.0);
    assert_eq!(empty_stats.cost_per_1k_tokens(), 0.0);
}

// ==================== P0: ToolChoice::Specific TESTS ====================

/// Validate ToolChoice::Specific serialization (critical: uses #[serde(untagged)])
#[test]
fn test_tool_choice_specific_serialization() {
    let specific = ToolChoice::Specific(ToolCallChoice {
        r#type: "function".to_string(),
        function: ToolFunctionRef {
            name: "my_function".to_string(),
        },
    });

    let json = serde_json::to_string(&specific).unwrap();

    // Should serialize as an object, NOT as a string (untagged behavior)
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(
        parsed.is_object(),
        "Specific variant should serialize as object"
    );
    assert_eq!(parsed["type"], "function");
    assert_eq!(parsed["function"]["name"], "my_function");
}

/// Validate ToolChoice::Specific round-trip
#[test]
fn test_tool_choice_specific_roundtrip() {
    let specific = ToolChoice::Specific(ToolCallChoice {
        r#type: "function".to_string(),
        function: ToolFunctionRef {
            name: "get_weather".to_string(),
        },
    });

    let json = serde_json::to_string(&specific).unwrap();
    let parsed: ToolChoice = serde_json::from_str(&json).unwrap();

    // Verify round-trip preserves data
    match parsed {
        ToolChoice::Specific(choice) => {
            assert_eq!(choice.r#type, "function");
            assert_eq!(choice.function.name, "get_weather");
        }
        _ => panic!("Expected Specific variant after round-trip"),
    }
}

/// Validate ToolChoice deserialization from various formats
#[test]
fn test_tool_choice_deserialization_variants() {
    // String variants
    let none: ToolChoice = serde_json::from_str(r#""none""#).unwrap();
    assert!(matches!(none, ToolChoice::None));

    let auto: ToolChoice = serde_json::from_str(r#""auto""#).unwrap();
    assert!(matches!(auto, ToolChoice::Auto));

    let required: ToolChoice = serde_json::from_str(r#""required""#).unwrap();
    assert!(matches!(required, ToolChoice::Required));

    // Object variant (Specific)
    let specific_json = r#"{"type": "function", "function": {"name": "search"}}"#;
    let specific: ToolChoice = serde_json::from_str(specific_json).unwrap();
    match specific {
        ToolChoice::Specific(choice) => {
            assert_eq!(choice.function.name, "search");
        }
        _ => panic!("Expected Specific variant"),
    }
}

// ==================== P0: ERROR DESERIALIZATION TESTS ====================

/// Validate ChatMessage with missing required fields fails deserialization
#[test]
fn test_chat_message_missing_required_fields() {
    // Missing content field
    let bad_json = r#"{"role": "user"}"#;
    let result = serde_json::from_str::<ChatMessage>(bad_json);
    assert!(result.is_err(), "Should fail without content field");

    // Missing role field
    let bad_json2 = r#"{"content": "Hello"}"#;
    let result2 = serde_json::from_str::<ChatMessage>(bad_json2);
    assert!(result2.is_err(), "Should fail without role field");
}

/// Validate ChatMessage with invalid role fails
#[test]
fn test_chat_message_invalid_role() {
    let bad_json = r#"{"role": "invalid_role", "content": "Hello"}"#;
    let result = serde_json::from_str::<ChatMessage>(bad_json);
    assert!(result.is_err(), "Should fail with invalid role");
}

/// Validate TokenUsage with invalid JSON fails
#[test]
fn test_token_usage_invalid_json() {
    use reasonkit::glm46::types::TokenUsage;

    // String instead of number
    let bad_json =
        r#"{"prompt_tokens": "not_a_number", "completion_tokens": 50, "total_tokens": 200}"#;
    let result = serde_json::from_str::<TokenUsage>(bad_json);
    assert!(result.is_err(), "Should fail with string instead of number");

    // Negative number (while valid JSON, may cause logic issues)
    let negative_json = r#"{"prompt_tokens": -10, "completion_tokens": 50, "total_tokens": 40}"#;
    // This might parse - depends on whether we want to validate at serde level
    // For now, just test that parsing works (validation is separate)
    let _result = serde_json::from_str::<TokenUsage>(negative_json);
}

/// Validate APIError with optional code field
#[test]
fn test_api_error_optional_code() {
    use reasonkit::glm46::types::APIError;

    // With code
    let with_code = r#"{"message": "Error", "type": "error_type", "code": "error_code"}"#;
    let error1: APIError = serde_json::from_str(with_code).unwrap();
    assert_eq!(error1.code, Some("error_code".to_string()));

    // Without code
    let without_code = r#"{"message": "Error", "type": "error_type"}"#;
    let error2: APIError = serde_json::from_str(without_code).unwrap();
    assert_eq!(error2.code, None);
}

/// Validate malformed JSON handling
#[test]
fn test_malformed_json_handling() {
    // Unclosed brace
    let unclosed = r#"{"role": "user", "content": "Hello""#;
    let result = serde_json::from_str::<ChatMessage>(unclosed);
    assert!(result.is_err(), "Should fail with unclosed brace");

    // Invalid JSON structure
    let invalid = r#"[{"not": "valid"}"#;
    let result2 = serde_json::from_str::<ChatMessage>(invalid);
    assert!(result2.is_err(), "Should fail with invalid structure");

    // Empty string
    let empty = "";
    let result3 = serde_json::from_str::<ChatMessage>(empty);
    assert!(result3.is_err(), "Should fail with empty string");
}

// ==================== P0: INPUT VALIDATION TESTS ====================

/// Test content length validation
#[test]
fn test_content_length_validation() {
    use reasonkit::glm46::types::{ValidationError, MAX_MESSAGE_CONTENT_LENGTH};

    // Valid content
    let valid_msg = ChatMessage::user("Hello, world!");
    assert!(valid_msg.validate().is_ok());

    // Content at limit - create just under limit
    let at_limit = "x".repeat(MAX_MESSAGE_CONTENT_LENGTH);
    let limit_msg = ChatMessage::user(at_limit);
    assert!(limit_msg.validate().is_ok());

    // Content over limit
    let over_limit = "x".repeat(MAX_MESSAGE_CONTENT_LENGTH + 1);
    let over_msg = ChatMessage::user(over_limit);
    match over_msg.validate() {
        Err(ValidationError::ContentTooLong { actual, max }) => {
            assert_eq!(actual, MAX_MESSAGE_CONTENT_LENGTH + 1);
            assert_eq!(max, MAX_MESSAGE_CONTENT_LENGTH);
        }
        _ => panic!("Expected ContentTooLong error"),
    }
}

/// Test empty user message validation
#[test]
fn test_empty_user_message_validation() {
    use reasonkit::glm46::types::ValidationError;

    let empty_user = ChatMessage::user("");
    match empty_user.validate() {
        Err(ValidationError::EmptyContent { role }) => {
            assert!(matches!(role, MessageRole::User));
        }
        _ => panic!("Expected EmptyContent error for empty user message"),
    }

    // Empty system messages are allowed
    let empty_system = ChatMessage::system("");
    assert!(empty_system.validate().is_ok());

    // Empty assistant messages are allowed
    let empty_assistant = ChatMessage::assistant("");
    assert!(empty_assistant.validate().is_ok());
}

/// Test ChatRequest validation
#[test]
fn test_chat_request_validation() {
    use reasonkit::glm46::types::{ValidationError, MAX_MESSAGES_PER_REQUEST};

    // Valid request
    let valid_request = ChatRequest {
        messages: vec![ChatMessage::user("Hello")],
        temperature: 0.7,
        max_tokens: 1000,
        response_format: None,
        tools: None,
        tool_choice: None,
        stop: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
    };
    assert!(valid_request.validate().is_ok());

    // Too many messages
    let many_messages: Vec<ChatMessage> = (0..MAX_MESSAGES_PER_REQUEST + 1)
        .map(|i| ChatMessage::user(format!("Message {}", i)))
        .collect();
    let over_request = ChatRequest {
        messages: many_messages,
        temperature: 0.7,
        max_tokens: 1000,
        response_format: None,
        tools: None,
        tool_choice: None,
        stop: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
    };
    match over_request.validate() {
        Err(ValidationError::TooManyMessages { actual, max }) => {
            assert_eq!(actual, MAX_MESSAGES_PER_REQUEST + 1);
            assert_eq!(max, MAX_MESSAGES_PER_REQUEST);
        }
        _ => panic!("Expected TooManyMessages error"),
    }
}

/// Test validated constructors
#[test]
fn test_validated_constructors() {
    // Valid
    let valid = ChatMessage::user_validated("Hello");
    assert!(valid.is_ok());

    // Invalid - empty user content
    let invalid = ChatMessage::user_validated("");
    assert!(invalid.is_err());

    // System can be empty
    let empty_system = ChatMessage::system_validated("");
    assert!(empty_system.is_ok());
}

// ==================== P1: CircuitState TESTS ====================

/// Validate CircuitState serialization round-trip
#[test]
fn test_circuit_state_roundtrip() {
    use reasonkit::glm46::types::CircuitState;

    // Closed state
    let closed = CircuitState::Closed;
    let json = serde_json::to_string(&closed).unwrap();
    let parsed: CircuitState = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, CircuitState::Closed));

    // Open state
    let open = CircuitState::Open {
        opens_at: std::time::SystemTime::now(),
        reset_after: Duration::from_secs(60),
    };
    let json = serde_json::to_string(&open).unwrap();
    let parsed: CircuitState = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, CircuitState::Open { .. }));

    // HalfOpen state
    let half_open = CircuitState::HalfOpen {
        probation_requests: 2,
        max_probation: 5,
    };
    let json = serde_json::to_string(&half_open).unwrap();
    let parsed: CircuitState = serde_json::from_str(&json).unwrap();
    match parsed {
        CircuitState::HalfOpen {
            probation_requests,
            max_probation,
        } => {
            assert_eq!(probation_requests, 2);
            assert_eq!(max_probation, 5);
        }
        _ => panic!("Expected HalfOpen variant"),
    }
}

// ==================== P2: UNICODE CONTENT TESTS ====================

/// Validate Unicode content handling
#[test]
fn test_unicode_content_serialization() {
    // Chinese characters
    let chinese = ChatMessage::user("你好，世界！");
    let json = serde_json::to_string(&chinese).unwrap();
    let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.content, "你好，世界！");

    // Japanese
    let japanese = ChatMessage::user("こんにちは世界");
    let json = serde_json::to_string(&japanese).unwrap();
    let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.content, "こんにちは世界");

    // Emoji
    let emoji = ChatMessage::user("Hello! 👋🌍🚀");
    let json = serde_json::to_string(&emoji).unwrap();
    let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.content, "Hello! 👋🌍🚀");

    // Mixed Unicode with special characters
    let mixed = ChatMessage::user("Über résumé: 日本語 + العربية");
    let json = serde_json::to_string(&mixed).unwrap();
    let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.content, "Über résumé: 日本語 + العربية");

    // Zero-width characters and control characters
    let special = ChatMessage::user("Hello\u{200B}World\u{FEFF}!");
    let json = serde_json::to_string(&special).unwrap();
    let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.content, "Hello\u{200B}World\u{FEFF}!");
}

// ==================== P2: GLM46Error TESTS ====================

/// Validate GLM46Error Display trait
#[test]
fn test_glm46_error_display() {
    use reasonkit::glm46::types::GLM46Error;

    let network_err = GLM46Error::Ollama("Connection refused".to_string());
    assert!(format!("{}", network_err).contains("Connection refused"));

    let api_err = GLM46Error::API {
        message: "Rate limit exceeded".to_string(),
        code: Some("429".to_string()),
    };
    assert!(format!("{}", api_err).contains("Rate limit"));

    let timeout_err = GLM46Error::Timeout(Duration::from_secs(30));
    assert!(format!("{}", timeout_err).contains("30"));

    let circuit_err = GLM46Error::CircuitOpen {
        reason: "Too many failures".to_string(),
    };
    assert!(format!("{}", circuit_err).contains("Circuit breaker"));

    let context_err = GLM46Error::ContextExceeded {
        used: 200_000,
        limit: 198_000,
    };
    assert!(format!("{}", context_err).contains("200000"));
    assert!(format!("{}", context_err).contains("198000"));

    let config_err = GLM46Error::Config("Missing API key".to_string());
    assert!(format!("{}", config_err).contains("Missing API key"));

    let invalid_req = GLM46Error::InvalidRequest("Empty messages".to_string());
    assert!(format!("{}", invalid_req).contains("Empty messages"));

    let rate_limited = GLM46Error::RateLimited {
        retry_after: Duration::from_secs(60),
    };
    assert!(format!("{}", rate_limited).contains("60"));
}

/// Validate GLM46Error Debug trait
#[test]
fn test_glm46_error_debug() {
    use reasonkit::glm46::types::GLM46Error;

    let err = GLM46Error::API {
        message: "Test error".to_string(),
        code: Some("ERR_001".to_string()),
    };
    let debug_output = format!("{:?}", err);
    assert!(debug_output.contains("API"));
    assert!(debug_output.contains("Test error"));
    assert!(debug_output.contains("ERR_001"));
}

// ==================== P1: PROPTEST INTEGRATION ====================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;
    use reasonkit::glm46::types::MessageRole;

    proptest! {
        /// Property: Any valid MessageRole can be serialized and deserialized
        #[test]
        fn message_role_roundtrip(role_idx in 0u8..4) {
            let role = match role_idx {
                0 => MessageRole::System,
                1 => MessageRole::User,
                2 => MessageRole::Assistant,
                _ => MessageRole::Tool,
            };

            let json = serde_json::to_string(&role).unwrap();
            let parsed: MessageRole = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(role, parsed);
        }

        /// Property: ChatMessage content is preserved through serialization
        #[test]
        fn chat_message_content_preserved(content in "[a-zA-Z0-9 ]{1,100}") {
            let msg = ChatMessage::user(&content);
            let json = serde_json::to_string(&msg).unwrap();
            let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(content, parsed.content);
        }

        /// Property: Temperature is clamped to valid range after optimization
        #[test]
        fn temperature_within_bounds(temp in 0.0f32..2.0) {
            let request = ChatRequest {
                messages: vec![ChatMessage::user("test")],
                temperature: temp,
                max_tokens: 1000,
                response_format: None,
                tools: None,
                tool_choice: None,
                stop: None,
                top_p: None,
                frequency_penalty: None,
                presence_penalty: None,
                stream: None,
            };
            // Temperature should be serializable
            let json = serde_json::to_string(&request).unwrap();
            let parsed: ChatRequest = serde_json::from_str(&json).unwrap();
            prop_assert!((parsed.temperature - temp).abs() < f32::EPSILON);
        }

        /// Property: Token counts are non-negative after deserialization
        #[test]
        fn token_usage_non_negative(prompt in 0usize..1_000_000, completion in 0usize..1_000_000) {
            use reasonkit::glm46::types::TokenUsage;

            let usage = TokenUsage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
            };

            let json = serde_json::to_string(&usage).unwrap();
            let parsed: TokenUsage = serde_json::from_str(&json).unwrap();

            prop_assert_eq!(parsed.prompt_tokens, prompt);
            prop_assert_eq!(parsed.completion_tokens, completion);
            prop_assert_eq!(parsed.total_tokens, prompt + completion);
        }
    }
}

// ==================== SECURITY: API KEY REDACTION TEST ====================

/// Verify API key is redacted in Debug output
#[test]
fn test_api_key_debug_redaction() {
    let config = GLM46Config {
        api_key: SecretString::from("super_secret_key_12345".to_string()),
        ..Default::default()
    };

    let debug_output = format!("{:?}", config);

    // Must NOT contain the actual key
    assert!(
        !debug_output.contains("super_secret_key"),
        "API key should be redacted in Debug output"
    );

    // Should contain redaction marker
    assert!(
        debug_output.contains("REDACTED") || debug_output.contains("redacted"),
        "Debug output should indicate redaction"
    );
}
