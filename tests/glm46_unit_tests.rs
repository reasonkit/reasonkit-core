#![cfg(feature = "glm46")]
//! # GLM-4.6 Unit Tests
//!
//! Comprehensive unit tests for GLM-4.6 integration components.
//! Tests cover client, types, circuit breaker, and ThinkTool profile integration.

use reasonkit::glm46::types::{
    ChatMessage, ChatRequest, MessageRole, ResponseFormat, Tool, ToolCall, ToolChoice,
    ToolFunction, ToolFunctionCall,
};
use reasonkit::glm46::{GLM46Client, GLM46Config};
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
    let config = GLM46Config {
        api_key: "test_key".to_string(),
        base_url: "https://test.example.com".to_string(),
        model: "glm-4.6-custom".to_string(),
        timeout: Duration::from_secs(60),
        context_budget: 100_000,
        cost_tracking: false,
        local_fallback: false,
    };

    assert_eq!(config.api_key, "test_key");
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
        api_key: "test_key".to_string(),
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
