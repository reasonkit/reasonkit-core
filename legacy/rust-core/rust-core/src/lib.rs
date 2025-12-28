//! ReasonKit Core - High-performance reasoning engine
//!
//! This Rust library provides:
//! - Fast protocol validation
//! - Efficient step execution tracking
//! - JSON schema validation
//! - Python bindings via PyO3
//!
//! # Performance
//!
//! Rust core is 10-100x faster than pure Python for:
//! - Schema validation
//! - Output parsing
//! - Metrics calculation
//!
//! # Usage from Python
//!
//! ```python
//! from reasonkit_core import validate_output, Protocol, Step
//!
//! # Fast validation
//! is_valid, errors = validate_output(output_json, required_fields)
//! ```

use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod protocol;
mod validation;
mod metrics;

pub use protocol::{Protocol, Step, OutputSchema, CognitiveStance};
pub use validation::{validate_output, ValidationResult};
pub use metrics::{SessionMetrics, StepMetrics};

/// Validate step output JSON against required fields (fast Rust implementation)
#[pyfunction]
fn validate_step_output(
    output_json: &str,
    required_fields: Vec<String>,
) -> PyResult<(bool, Vec<String>)> {
    let parsed: Result<HashMap<String, serde_json::Value>, _> = serde_json::from_str(output_json);

    match parsed {
        Ok(output) => {
            let result = validation::validate_required_fields(&output, &required_fields);
            Ok((result.is_valid, result.errors))
        }
        Err(e) => Ok((false, vec![format!("JSON parse error: {}", e)])),
    }
}

/// Calculate weighted confidence from step outputs
#[pyfunction]
fn calculate_confidence(confidences: Vec<f64>) -> f64 {
    metrics::calculate_weighted_confidence(&confidences)
}

/// Parse and validate JSON output (faster than Python json + validation)
#[pyfunction]
fn parse_and_validate(
    json_str: &str,
    required_fields: Vec<String>,
) -> (bool, String, Vec<String>) {
    match serde_json::from_str::<HashMap<String, serde_json::Value>>(json_str) {
        Ok(parsed) => {
            let mut errors = Vec::new();
            for field in &required_fields {
                if !parsed.contains_key(field) {
                    errors.push(format!("Missing required field: {}", field));
                }
            }
            let output_json = serde_json::to_string(&parsed).unwrap_or_default();
            (errors.is_empty(), output_json, errors)
        }
        Err(e) => (false, String::new(), vec![format!("JSON parse error: {}", e)]),
    }
}

/// Extract text from JSON response (handles markdown code blocks)
#[pyfunction]
fn extract_json_from_response(response: &str) -> String {
    // Remove markdown code blocks
    let mut text = response.trim();

    if text.starts_with("```json") {
        text = &text[7..];
    } else if text.starts_with("```") {
        text = &text[3..];
    }

    if text.ends_with("```") {
        text = &text[..text.len() - 3];
    }

    text.trim().to_string()
}

/// Batch validate multiple JSON outputs
#[pyfunction]
fn batch_validate(
    outputs_json: Vec<String>,
    required_fields: Vec<String>,
) -> Vec<(bool, Vec<String>)> {
    outputs_json
        .iter()
        .map(|json_str| {
            match serde_json::from_str::<HashMap<String, serde_json::Value>>(json_str) {
                Ok(output) => {
                    let result = validation::validate_required_fields(&output, &required_fields);
                    (result.is_valid, result.errors)
                }
                Err(e) => (false, vec![format!("JSON parse error: {}", e)]),
            }
        })
        .collect()
}

/// Python module definition
#[pymodule]
fn reasonkit_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_step_output, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_confidence, m)?)?;
    m.add_function(wrap_pyfunction!(parse_and_validate, m)?)?;
    m.add_function(wrap_pyfunction!(extract_json_from_response, m)?)?;
    m.add_function(wrap_pyfunction!(batch_validate, m)?)?;

    m.add_class::<protocol::PyProtocol>()?;
    m.add_class::<protocol::PyStep>()?;
    m.add_class::<metrics::PySessionMetrics>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_required_fields() {
        let json_str = r#"{"name": "test"}"#;
        let required = vec!["name".to_string(), "value".to_string()];
        let (is_valid, errors) = validate_step_output(json_str, required).unwrap();

        assert!(!is_valid);
        assert!(errors.iter().any(|e| e.contains("value")));
    }

    #[test]
    fn test_calculate_confidence() {
        let confidences = vec![0.8, 0.9, 0.85];
        let result = metrics::calculate_weighted_confidence(&confidences);
        assert!(result > 0.8 && result < 0.95);
    }

    #[test]
    fn test_extract_json() {
        let response = "```json\n{\"key\": \"value\"}\n```";
        let result = extract_json_from_response(response);
        assert_eq!(result, "{\"key\": \"value\"}");
    }

    #[test]
    fn test_parse_and_validate() {
        let json_str = r#"{"name": "test", "value": 42}"#;
        let required = vec!["name".to_string()];
        let (is_valid, _, errors) = parse_and_validate(json_str, required);

        assert!(is_valid);
        assert!(errors.is_empty());
    }
}
