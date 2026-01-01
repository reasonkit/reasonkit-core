//! Property-Based Tests for ReasonKit ThinkTool Reasoning Engine
//!
//! This module provides comprehensive property-based testing using proptest
//! to verify invariants, fuzz edge cases, and ensure robustness of the
//! ThinkTool reasoning system.
//!
//! ## Test Categories
//!
//! 1. **Prompt Parsing** - Any valid UTF-8 input should not panic
//! 2. **Confidence Scoring** - Always bounded between 0.0 and 1.0
//! 3. **JSON Output** - ThinkTool outputs always produce valid JSON
//! 4. **Edge Case Fuzzing** - Boundary conditions and malformed inputs
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --test reasoning_prop_tests
//! # Run with more iterations for thorough fuzzing:
//! PROPTEST_CASES=10000 cargo test --test reasoning_prop_tests
//! ```

use proptest::prelude::*;
use std::collections::HashMap;

// Import from reasonkit crate
use reasonkit::thinktool::{
    calibration::{CalibrationDiagnosis, ConfidenceAdjuster, Prediction},
    step::{ListItem, StepOutput, StepResult, TokenUsage},
    ProtocolInput,
};

// ============================================================================
// STRATEGY GENERATORS
// ============================================================================

/// Generate arbitrary UTF-8 strings including edge cases
fn arbitrary_utf8_string() -> impl Strategy<Value = String> {
    prop_oneof![
        // Empty string
        Just(String::new()),
        // Normal ASCII strings
        "[a-zA-Z0-9 ]{0,1000}".prop_map(|s| s),
        // Unicode strings with various scripts
        any::<String>(),
        // Strings with special characters
        prop::collection::vec(
            prop_oneof![
                Just('\n'),
                Just('\r'),
                Just('\t'),
                Just('\0'),
                Just('\x1b'),     // ESC
                Just('\u{FEFF}'), // BOM
                Just('\u{200B}'), // Zero-width space
                Just('\u{2028}'), // Line separator
                Just('\u{2029}'), // Paragraph separator
                any::<char>(),
            ],
            0..500
        )
        .prop_map(|chars| chars.into_iter().collect::<String>()),
        // Very long strings
        "[a-z]{10000,20000}".prop_map(|s| s),
        // Strings with emoji and combining characters
        prop::collection::vec(
            prop_oneof![
                Just('\u{1F600}'), // Grinning face
                Just('\u{1F4A9}'), // Pile of poo
                Just('\u{0301}'),  // Combining acute accent
                Just('\u{200D}'),  // Zero-width joiner
                any::<char>(),
            ],
            0..100
        )
        .prop_map(|chars| chars.into_iter().collect::<String>()),
    ]
}

/// Generate confidence values in the valid range [0.0, 1.0]
fn valid_confidence() -> impl Strategy<Value = f64> {
    (0.0..=1.0f64).prop_map(|v| v)
}

/// Generate confidence values that may be out of range (for testing clamping)
fn any_confidence() -> impl Strategy<Value = f64> {
    prop_oneof![
        // Valid range
        (0.0..=1.0f64),
        // Below range
        (-1000.0..0.0f64),
        // Above range
        (1.0..1000.0f64),
        // Special floating point values
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        Just(f64::MIN),
        Just(f64::MAX),
        Just(f64::EPSILON),
        Just(-0.0f64),
        Just(0.0f64),
    ]
}

/// Generate f32 confidence for Prediction type
fn any_confidence_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        (0.0..=1.0f32),
        (-1000.0..0.0f32),
        (1.0..1000.0f32),
        Just(f32::NAN),
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
        Just(f32::MIN),
        Just(f32::MAX),
        Just(f32::EPSILON),
        Just(-0.0f32),
        Just(0.0f32),
    ]
}

/// Generate arbitrary step IDs
fn arbitrary_step_id() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::new()),
        "[a-z_][a-z0-9_]{0,50}".prop_map(|s| s),
        arbitrary_utf8_string(),
    ]
}

/// Generate arbitrary token counts
fn arbitrary_token_count() -> impl Strategy<Value = u32> {
    prop_oneof![Just(0u32), (1..100u32), (100..10000u32), Just(u32::MAX),]
}

/// Generate arbitrary cost values
fn arbitrary_cost() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0f64),
        (0.0..0.001f64),
        (0.001..1.0f64),
        (1.0..1000.0f64),
        Just(f64::MAX),
        Just(f64::MIN_POSITIVE),
    ]
}

// ============================================================================
// PROMPT PARSING PROPERTY TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: ProtocolInput::query should never panic for any valid UTF-8 string
    #[test]
    fn prop_protocol_input_query_never_panics(query in arbitrary_utf8_string()) {
        // Should not panic
        let input = ProtocolInput::query(&query);

        // Should have the query field set
        prop_assert!(input.fields.contains_key("query"));

        // The value should be a string
        if let Some(serde_json::Value::String(s)) = input.fields.get("query") {
            prop_assert_eq!(s, &query);
        } else {
            prop_assert!(false, "query field should be a string");
        }
    }

    /// Property: ProtocolInput::argument should never panic for any valid UTF-8 string
    #[test]
    fn prop_protocol_input_argument_never_panics(arg in arbitrary_utf8_string()) {
        let input = ProtocolInput::argument(&arg);
        prop_assert!(input.fields.contains_key("argument"));
    }

    /// Property: ProtocolInput::statement should never panic for any valid UTF-8 string
    #[test]
    fn prop_protocol_input_statement_never_panics(stmt in arbitrary_utf8_string()) {
        let input = ProtocolInput::statement(&stmt);
        prop_assert!(input.fields.contains_key("statement"));
    }

    /// Property: ProtocolInput::claim should never panic for any valid UTF-8 string
    #[test]
    fn prop_protocol_input_claim_never_panics(claim in arbitrary_utf8_string()) {
        let input = ProtocolInput::claim(&claim);
        prop_assert!(input.fields.contains_key("claim"));
    }

    /// Property: ProtocolInput::work should never panic for any valid UTF-8 string
    #[test]
    fn prop_protocol_input_work_never_panics(work in arbitrary_utf8_string()) {
        let input = ProtocolInput::work(&work);
        prop_assert!(input.fields.contains_key("work"));
    }

    /// Property: with_field should chain correctly
    #[test]
    fn prop_protocol_input_chaining(
        query in arbitrary_utf8_string(),
        field_key in "[a-z][a-z0-9_]{0,20}",
        field_value in arbitrary_utf8_string()
    ) {
        let input = ProtocolInput::query(&query)
            .with_field(&field_key, &field_value);

        prop_assert!(input.fields.contains_key("query"));
        prop_assert!(input.fields.contains_key(&field_key));
    }

    /// Property: get_str should return correct value or None
    #[test]
    fn prop_protocol_input_get_str(
        key in "[a-z][a-z0-9_]{0,20}",
        value in arbitrary_utf8_string()
    ) {
        let input = ProtocolInput::query("test").with_field(&key, &value);

        // Existing key should return the value
        prop_assert_eq!(input.get_str(&key), Some(value.as_str()));

        // Non-existing key should return None
        prop_assert_eq!(input.get_str("nonexistent_key_xyz_123"), None);
    }
}

// ============================================================================
// CONFIDENCE SCORING PROPERTY TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: Prediction::new should clamp confidence to [0.0, 1.0]
    #[test]
    fn prop_prediction_confidence_clamped(confidence in any_confidence_f32(), correct: bool) {
        let prediction = Prediction::new(confidence, correct);

        // Handle special floating point values
        if confidence.is_nan() {
            // NaN.clamp returns NaN - verify the behavior
            let clamped = confidence.clamp(0.0, 1.0);
            prop_assert!(clamped.is_nan() || (0.0..=1.0).contains(&clamped));
        } else if confidence.is_infinite() {
            // Infinity clamps to 1.0, -Infinity clamps to 0.0
            if confidence.is_sign_positive() {
                prop_assert_eq!(prediction.confidence, 1.0);
            } else {
                prop_assert_eq!(prediction.confidence, 0.0);
            }
        } else {
            // Normal values should be clamped to [0.0, 1.0]
            prop_assert!(prediction.confidence >= 0.0);
            prop_assert!(prediction.confidence <= 1.0);
        }
    }

    /// Property: StepResult::success should store confidence unchanged
    #[test]
    fn prop_step_result_confidence_preserved(
        step_id in arbitrary_step_id(),
        confidence in valid_confidence()
    ) {
        let output = StepOutput::Text { content: "test".to_string() };
        let result = StepResult::success(&step_id, output, confidence);

        prop_assert_eq!(result.confidence, confidence);
        prop_assert!(result.success);
        prop_assert!(result.error.is_none());
    }

    /// Property: StepResult::failure should have zero confidence
    #[test]
    fn prop_step_result_failure_zero_confidence(
        step_id in arbitrary_step_id(),
        error in arbitrary_utf8_string()
    ) {
        let result = StepResult::failure(&step_id, &error);

        prop_assert_eq!(result.confidence, 0.0);
        prop_assert!(!result.success);
        prop_assert_eq!(result.error, Some(error));
    }

    /// Property: meets_threshold works correctly
    #[test]
    fn prop_step_result_meets_threshold(
        confidence in valid_confidence(),
        threshold in valid_confidence()
    ) {
        let output = StepOutput::Text { content: "test".to_string() };
        let result = StepResult::success("test", output, confidence);

        let meets = result.meets_threshold(threshold);
        prop_assert_eq!(meets, confidence >= threshold);
    }

    /// Property: ConfidenceAdjuster::adjust should always return valid confidence
    #[test]
    fn prop_confidence_adjuster_returns_valid(raw_confidence in (0.0..=1.0f32)) {
        for diagnosis in [
            CalibrationDiagnosis::WellCalibrated,
            CalibrationDiagnosis::SlightlyOverconfident,
            CalibrationDiagnosis::Overconfident,
            CalibrationDiagnosis::SeverelyOverconfident,
            CalibrationDiagnosis::Underconfident,
            CalibrationDiagnosis::Mixed,
            CalibrationDiagnosis::InsufficientData,
        ] {
            let adjusted = ConfidenceAdjuster::adjust(raw_confidence, diagnosis);
            prop_assert!(adjusted >= 0.0, "Adjusted confidence {} should be >= 0.0", adjusted);
            prop_assert!(adjusted <= 1.0, "Adjusted confidence {} should be <= 1.0", adjusted);
        }
    }

    /// Property: confidence_to_qualifier should always return a non-empty string
    #[test]
    fn prop_confidence_qualifier_non_empty(confidence in (0.0..=1.0f32)) {
        let qualifier = ConfidenceAdjuster::confidence_to_qualifier(confidence);
        prop_assert!(!qualifier.is_empty());
    }

    /// Property: Confidence adjustments should be monotonic for overconfidence
    #[test]
    fn prop_overconfidence_adjustment_reduces(raw in (0.01..=1.0f32)) {
        let adjusted_severe = ConfidenceAdjuster::adjust(raw, CalibrationDiagnosis::SeverelyOverconfident);
        let adjusted_over = ConfidenceAdjuster::adjust(raw, CalibrationDiagnosis::Overconfident);
        let adjusted_slight = ConfidenceAdjuster::adjust(raw, CalibrationDiagnosis::SlightlyOverconfident);

        // More severe adjustment should reduce more
        prop_assert!(adjusted_severe <= adjusted_over,
            "Severe {} should reduce more than Over {}", adjusted_severe, adjusted_over);
        prop_assert!(adjusted_over <= adjusted_slight,
            "Over {} should reduce more than Slight {}", adjusted_over, adjusted_slight);
    }
}

// ============================================================================
// JSON OUTPUT PROPERTY TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: StepOutput::Text should always serialize to valid JSON
    #[test]
    fn prop_step_output_text_valid_json(content in arbitrary_utf8_string()) {
        let output = StepOutput::Text { content: content.clone() };
        let json_result = serde_json::to_string(&output);

        prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

        // Should deserialize back correctly
        if let Ok(json) = json_result {
            let parsed: Result<StepOutput, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
        }
    }

    /// Property: StepOutput::List should always serialize to valid JSON
    #[test]
    fn prop_step_output_list_valid_json(
        items in prop::collection::vec(arbitrary_utf8_string(), 0..50)
    ) {
        let list_items: Vec<ListItem> = items.into_iter()
            .map(|content| ListItem::new(content))
            .collect();

        let output = StepOutput::List { items: list_items };
        let json_result = serde_json::to_string(&output);

        prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

        if let Ok(json) = json_result {
            let parsed: Result<StepOutput, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
        }
    }

    /// Property: StepOutput::Structured should always serialize to valid JSON
    #[test]
    fn prop_step_output_structured_valid_json(
        keys in prop::collection::vec("[a-z][a-z0-9_]{0,20}", 0..20),
        values in prop::collection::vec(arbitrary_utf8_string(), 0..20)
    ) {
        let mut data: HashMap<String, serde_json::Value> = HashMap::new();
        for (key, value) in keys.into_iter().zip(values.into_iter()) {
            data.insert(key, serde_json::Value::String(value));
        }

        let output = StepOutput::Structured { data };
        let json_result = serde_json::to_string(&output);

        prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

        if let Ok(json) = json_result {
            let parsed: Result<StepOutput, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
        }
    }

    /// Property: StepOutput::Score should always serialize to valid JSON
    #[test]
    fn prop_step_output_score_valid_json(value in valid_confidence()) {
        let output = StepOutput::Score { value };
        let json_result = serde_json::to_string(&output);

        prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

        if let Ok(json) = json_result {
            let parsed: Result<StepOutput, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
        }
    }

    /// Property: StepOutput::Boolean should always serialize to valid JSON
    #[test]
    fn prop_step_output_boolean_valid_json(
        value: bool,
        reason in proptest::option::of(arbitrary_utf8_string())
    ) {
        let output = StepOutput::Boolean { value, reason };
        let json_result = serde_json::to_string(&output);

        prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

        if let Ok(json) = json_result {
            let parsed: Result<StepOutput, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
        }
    }

    /// Property: StepResult should always serialize to valid JSON
    #[test]
    fn prop_step_result_valid_json(
        step_id in arbitrary_step_id(),
        confidence in valid_confidence(),
        content in arbitrary_utf8_string()
    ) {
        let output = StepOutput::Text { content };
        let result = StepResult::success(&step_id, output, confidence);
        let json_result = serde_json::to_string(&result);

        prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

        if let Ok(json) = json_result {
            let parsed: Result<StepResult, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
        }
    }

    /// Property: Prediction should always serialize to valid JSON
    #[test]
    fn prop_prediction_valid_json(
        confidence in (0.0..=1.0f32),
        correct: bool,
        category in proptest::option::of("[a-z]{1,20}".prop_map(|s| s))
    ) {
        let mut prediction = Prediction::new(confidence, correct);
        if let Some(cat) = category {
            prediction = prediction.with_category(cat);
        }

        let json_result = serde_json::to_string(&prediction);
        prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

        if let Ok(json) = json_result {
            let parsed: Result<Prediction, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
        }
    }

    /// Property: ProtocolInput fields should serialize to valid JSON
    #[test]
    fn prop_protocol_input_valid_json(
        query in arbitrary_utf8_string(),
        extra_key in "[a-z][a-z0-9_]{0,20}",
        extra_value in arbitrary_utf8_string()
    ) {
        let input = ProtocolInput::query(&query).with_field(&extra_key, &extra_value);
        let json_result = serde_json::to_string(&input);

        prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

        if let Ok(json) = json_result {
            let parsed: Result<ProtocolInput, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
        }
    }
}

// ============================================================================
// TOKEN USAGE PROPERTY TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: TokenUsage::new should calculate total correctly
    #[test]
    fn prop_token_usage_total_correct(
        input in arbitrary_token_count(),
        output in arbitrary_token_count()
    ) {
        // Check for overflow
        if input.checked_add(output).is_some() {
            let usage = TokenUsage::new(input, output, 0.0);
            prop_assert_eq!(usage.total_tokens, input + output);
        }
    }

    /// Property: TokenUsage::add should accumulate correctly
    #[test]
    fn prop_token_usage_add_accumulates(
        input1 in 0..10000u32,
        output1 in 0..10000u32,
        cost1 in 0.0..100.0f64,
        input2 in 0..10000u32,
        output2 in 0..10000u32,
        cost2 in 0.0..100.0f64
    ) {
        let mut usage1 = TokenUsage::new(input1, output1, cost1);
        let usage2 = TokenUsage::new(input2, output2, cost2);

        usage1.add(&usage2);

        prop_assert_eq!(usage1.input_tokens, input1 + input2);
        prop_assert_eq!(usage1.output_tokens, output1 + output2);
        prop_assert_eq!(usage1.total_tokens, input1 + output1 + input2 + output2);

        // Cost should be approximately equal (floating point)
        let expected_cost = cost1 + cost2;
        prop_assert!((usage1.cost_usd - expected_cost).abs() < 1e-10);
    }

    /// Property: TokenUsage should serialize to valid JSON
    #[test]
    fn prop_token_usage_valid_json(
        input in arbitrary_token_count(),
        output in arbitrary_token_count(),
        cost in arbitrary_cost()
    ) {
        if input.checked_add(output).is_some() {
            let usage = TokenUsage::new(input, output, cost);
            let json_result = serde_json::to_string(&usage);

            prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

            if let Ok(json) = json_result {
                let parsed: Result<TokenUsage, _> = serde_json::from_str(&json);
                prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
            }
        }
    }
}

// ============================================================================
// LIST ITEM PROPERTY TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: ListItem::new should never panic
    #[test]
    fn prop_list_item_new_never_panics(content in arbitrary_utf8_string()) {
        let item = ListItem::new(&content);
        prop_assert_eq!(item.content, content);
        prop_assert!(item.confidence.is_none());
        prop_assert!(item.metadata.is_empty());
    }

    /// Property: ListItem::with_confidence should store confidence
    #[test]
    fn prop_list_item_with_confidence(
        content in arbitrary_utf8_string(),
        confidence in valid_confidence()
    ) {
        let item = ListItem::with_confidence(&content, confidence);
        prop_assert_eq!(item.content, content);
        prop_assert_eq!(item.confidence, Some(confidence));
    }

    /// Property: ListItem should serialize to valid JSON
    #[test]
    fn prop_list_item_valid_json(
        content in arbitrary_utf8_string(),
        confidence in proptest::option::of(valid_confidence())
    ) {
        let item = if let Some(conf) = confidence {
            ListItem::with_confidence(&content, conf)
        } else {
            ListItem::new(&content)
        };

        let json_result = serde_json::to_string(&item);
        prop_assert!(json_result.is_ok(), "Failed to serialize: {:?}", json_result.err());

        if let Ok(json) = json_result {
            let parsed: Result<ListItem, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok(), "Failed to deserialize: {:?}", parsed.err());
        }
    }
}

// ============================================================================
// EDGE CASE FUZZING TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Fuzz test: Empty strings should be handled gracefully
    #[test]
    fn fuzz_empty_strings(_dummy in 0..1i32) {
        let empty = String::new();

        // ProtocolInput with empty query
        let input = ProtocolInput::query(&empty);
        prop_assert!(input.fields.contains_key("query"));

        // ListItem with empty content
        let item = ListItem::new(&empty);
        prop_assert!(item.content.is_empty());

        // StepOutput::Text with empty content
        let output = StepOutput::Text { content: empty.clone() };
        let json = serde_json::to_string(&output);
        prop_assert!(json.is_ok());

        // StepResult with empty step_id
        let result = StepResult::success(&empty, output, 0.5);
        prop_assert!(result.step_id.is_empty());
    }

    /// Fuzz test: Null bytes in strings
    #[test]
    fn fuzz_null_bytes(prefix in "[a-z]{0,10}", suffix in "[a-z]{0,10}") {
        let with_null = format!("{}\0{}", prefix, suffix);

        // Should handle null bytes without panic
        let input = ProtocolInput::query(&with_null);
        prop_assert!(input.fields.contains_key("query"));

        // JSON serialization should escape null bytes
        let json = serde_json::to_string(&input);
        prop_assert!(json.is_ok());
    }

    /// Fuzz test: Very long strings (stress test)
    #[test]
    fn fuzz_very_long_strings(length in 10000usize..50000usize) {
        let long_string: String = std::iter::repeat('x').take(length).collect();

        // Should handle long strings without stack overflow
        let input = ProtocolInput::query(&long_string);
        prop_assert!(input.fields.contains_key("query"));

        // Should serialize without issues
        let json = serde_json::to_string(&input);
        prop_assert!(json.is_ok());
    }

    /// Fuzz test: Deeply nested JSON values
    #[test]
    fn fuzz_nested_structured_output(depth in 1usize..10usize) {
        let mut data: HashMap<String, serde_json::Value> = HashMap::new();

        // Create nested structure
        let mut current = serde_json::Value::String("leaf".to_string());
        for i in 0..depth {
            let mut wrapper = serde_json::Map::new();
            wrapper.insert(format!("level_{}", i), current);
            current = serde_json::Value::Object(wrapper);
        }

        data.insert("nested".to_string(), current);
        let output = StepOutput::Structured { data };

        let json = serde_json::to_string(&output);
        prop_assert!(json.is_ok());
    }

    /// Fuzz test: Unicode normalization edge cases
    #[test]
    fn fuzz_unicode_normalization(_dummy in 0..1i32) {
        // Various Unicode edge cases
        let edge_cases = vec![
            "\u{FEFF}BOM at start",
            "zero\u{200B}width\u{200B}space",
            "combining\u{0301}accents",
            "\u{202E}RTL override\u{202C}",
            "\u{2028}line\u{2029}separators",
            "\u{FFFD}replacement char",
        ];

        for case in edge_cases {
            // These should not panic
            let _ = ProtocolInput::query(case);
        }
    }

    /// Fuzz test: Boundary confidence values
    #[test]
    fn fuzz_boundary_confidence(_dummy in 0..1i32) {
        let boundaries = vec![
            0.0,
            f64::EPSILON,
            0.5 - f64::EPSILON,
            0.5,
            0.5 + f64::EPSILON,
            1.0 - f64::EPSILON,
            1.0,
        ];

        for conf in boundaries {
            let result = StepResult::success(
                "test",
                StepOutput::Text { content: "test".to_string() },
                conf
            );
            prop_assert!(result.confidence >= 0.0);
            prop_assert!(result.confidence <= 1.0);
        }
    }

    /// Fuzz test: Maximum token counts
    #[test]
    fn fuzz_max_token_counts(_dummy in 0..1i32) {
        // Test near-overflow conditions
        let max_safe = u32::MAX / 2;

        let usage1 = TokenUsage::new(max_safe, max_safe, 0.0);
        prop_assert_eq!(usage1.input_tokens, max_safe);
        prop_assert_eq!(usage1.output_tokens, max_safe);

        // Adding two halves should be safe
        let mut usage = TokenUsage::new(max_safe, 0, 0.0);
        let usage2 = TokenUsage::new(0, max_safe, 0.0);
        usage.add(&usage2);

        prop_assert_eq!(usage.input_tokens, max_safe);
        prop_assert_eq!(usage.output_tokens, max_safe);
    }

    /// Fuzz test: Special JSON characters
    #[test]
    fn fuzz_json_special_chars(_dummy in 0..1i32) {
        let special_cases = vec![
            r#"{"nested": "json"}"#,
            r#"["array", "in", "string"]"#,
            r#"backslash \\ test"#,
            r#"quote \" test"#,
            r#"newline \n tab \t test"#,
            "<script>alert('xss')</script>",
            "null",
            "true",
            "false",
            "123",
            "3.14",
        ];

        for case in special_cases {
            let input = ProtocolInput::query(case);
            let json = serde_json::to_string(&input);
            prop_assert!(json.is_ok(), "Failed to serialize special case: {}", case);

            if let Ok(json_str) = json {
                let parsed: Result<ProtocolInput, _> = serde_json::from_str(&json_str);
                prop_assert!(parsed.is_ok(), "Failed to parse back: {}", case);
            }
        }
    }
}

// ============================================================================
// ROUNDTRIP PROPERTY TESTS (Serialize -> Deserialize = Identity)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: StepOutput roundtrip should preserve data
    #[test]
    fn prop_step_output_roundtrip(content in "[a-zA-Z0-9 ]{0,1000}") {
        let original = StepOutput::Text { content: content.clone() };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: StepOutput = serde_json::from_str(&json).unwrap();

        if let StepOutput::Text { content: parsed_content } = parsed {
            prop_assert_eq!(content, parsed_content);
        } else {
            prop_assert!(false, "Type mismatch after roundtrip");
        }
    }

    /// Property: ListItem roundtrip should preserve data
    #[test]
    fn prop_list_item_roundtrip(
        content in "[a-zA-Z0-9 ]{0,500}",
        confidence in proptest::option::of(valid_confidence())
    ) {
        let original = if let Some(conf) = confidence {
            ListItem::with_confidence(&content, conf)
        } else {
            ListItem::new(&content)
        };

        let json = serde_json::to_string(&original).unwrap();
        let parsed: ListItem = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(original.content, parsed.content);
        prop_assert_eq!(original.confidence, parsed.confidence);
    }

    /// Property: TokenUsage roundtrip should preserve data
    #[test]
    fn prop_token_usage_roundtrip(
        input in 0..1000000u32,
        output in 0..1000000u32,
        cost in 0.0..10000.0f64
    ) {
        let original = TokenUsage::new(input, output, cost);
        let json = serde_json::to_string(&original).unwrap();
        let parsed: TokenUsage = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(original.input_tokens, parsed.input_tokens);
        prop_assert_eq!(original.output_tokens, parsed.output_tokens);
        prop_assert_eq!(original.total_tokens, parsed.total_tokens);
        // Allow small floating point variance
        prop_assert!((original.cost_usd - parsed.cost_usd).abs() < 1e-10);
    }

    /// Property: Prediction roundtrip should preserve data
    #[test]
    fn prop_prediction_roundtrip(
        confidence in (0.0..=1.0f32),
        correct: bool
    ) {
        let original = Prediction::new(confidence, correct);
        let json = serde_json::to_string(&original).unwrap();
        let parsed: Prediction = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(original.correct, parsed.correct);
        // Allow small floating point variance
        prop_assert!((original.confidence - parsed.confidence).abs() < 1e-6);
    }
}

// ============================================================================
// INVARIANT TESTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Invariant: StepResult::success always has success = true
    #[test]
    fn invariant_step_result_success_flag(
        step_id in arbitrary_step_id(),
        confidence in valid_confidence()
    ) {
        let output = StepOutput::Empty;
        let result = StepResult::success(&step_id, output, confidence);
        prop_assert!(result.success);
        prop_assert!(result.error.is_none());
    }

    /// Invariant: StepResult::failure always has success = false
    #[test]
    fn invariant_step_result_failure_flag(
        step_id in arbitrary_step_id(),
        error in "[a-zA-Z0-9 ]{1,100}"
    ) {
        let result = StepResult::failure(&step_id, &error);
        prop_assert!(!result.success);
        prop_assert!(result.error.is_some());
        prop_assert_eq!(result.confidence, 0.0);
    }

    /// Invariant: TokenUsage total_tokens = input_tokens + output_tokens
    #[test]
    fn invariant_token_usage_total(
        input in 0..1000000u32,
        output in 0..1000000u32
    ) {
        let usage = TokenUsage::new(input, output, 0.0);
        prop_assert_eq!(usage.total_tokens, usage.input_tokens + usage.output_tokens);
    }

    /// Invariant: CalibrationDiagnosis categories are mutually exclusive
    #[test]
    fn invariant_calibration_diagnosis_exclusive(
        ece in (0.0..=0.5f32),
        avg_conf in (0.0..=1.0f32),
        accuracy in (0.0..=1.0f32)
    ) {
        let diagnosis = CalibrationDiagnosis::from_metrics(ece, avg_conf, accuracy);

        // Each diagnosis should be one specific variant
        let variants_matched = [
            matches!(diagnosis, CalibrationDiagnosis::WellCalibrated),
            matches!(diagnosis, CalibrationDiagnosis::SlightlyOverconfident),
            matches!(diagnosis, CalibrationDiagnosis::Overconfident),
            matches!(diagnosis, CalibrationDiagnosis::SeverelyOverconfident),
            matches!(diagnosis, CalibrationDiagnosis::Underconfident),
            matches!(diagnosis, CalibrationDiagnosis::Mixed),
        ].iter().filter(|&&x| x).count();

        prop_assert_eq!(variants_matched, 1, "Exactly one diagnosis should match");
    }

    /// Invariant: ConfidenceAdjuster never produces NaN or Infinity
    #[test]
    fn invariant_confidence_adjuster_finite(raw_confidence in (0.0..=1.0f32)) {
        for diagnosis in [
            CalibrationDiagnosis::WellCalibrated,
            CalibrationDiagnosis::SlightlyOverconfident,
            CalibrationDiagnosis::Overconfident,
            CalibrationDiagnosis::SeverelyOverconfident,
            CalibrationDiagnosis::Underconfident,
            CalibrationDiagnosis::Mixed,
            CalibrationDiagnosis::InsufficientData,
        ] {
            let adjusted = ConfidenceAdjuster::adjust(raw_confidence, diagnosis);
            prop_assert!(adjusted.is_finite(), "Adjusted confidence should be finite");
            prop_assert!(!adjusted.is_nan(), "Adjusted confidence should not be NaN");
        }
    }
}

// ============================================================================
// PERFORMANCE PROPERTY TESTS (Ensure operations complete in reasonable time)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Performance: Large list serialization should complete quickly
    #[test]
    fn perf_large_list_serialization(size in 100usize..1000usize) {
        let items: Vec<ListItem> = (0..size)
            .map(|i| ListItem::with_confidence(format!("Item {}", i), 0.5))
            .collect();

        let output = StepOutput::List { items };

        let start = std::time::Instant::now();
        let json = serde_json::to_string(&output);
        let elapsed = start.elapsed();

        prop_assert!(json.is_ok());
        // Should complete in under 100ms for 1000 items
        prop_assert!(elapsed.as_millis() < 100, "Serialization took too long: {:?}", elapsed);
    }

    /// Performance: Large structured data serialization should complete quickly
    #[test]
    fn perf_large_structured_serialization(size in 100usize..500usize) {
        let mut data: HashMap<String, serde_json::Value> = HashMap::new();
        for i in 0..size {
            data.insert(
                format!("key_{}", i),
                serde_json::json!({
                    "value": i,
                    "description": format!("Description for item {}", i),
                    "nested": {"a": 1, "b": 2}
                })
            );
        }

        let output = StepOutput::Structured { data };

        let start = std::time::Instant::now();
        let json = serde_json::to_string(&output);
        let elapsed = start.elapsed();

        prop_assert!(json.is_ok());
        prop_assert!(elapsed.as_millis() < 200, "Serialization took too long: {:?}", elapsed);
    }
}
