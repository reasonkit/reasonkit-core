//! Fast validation for step outputs

use serde_json::Value;
use std::collections::HashMap;

/// Result of validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn invalid(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.is_valid = false;
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// Validate output against required fields and types
pub fn validate_output_internal(
    output: &HashMap<String, Value>,
    required_fields: &[String],
    field_types: &HashMap<String, String>,
) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Check required fields
    for field in required_fields {
        match output.get(field) {
            None => {
                result.add_error(format!("Missing required field: {}", field));
            }
            Some(value) => {
                if value.is_null() {
                    result.add_error(format!("Required field '{}' is null", field));
                } else if let Some(s) = value.as_str() {
                    if s.trim().is_empty() {
                        result.add_error(format!("Required field '{}' is empty", field));
                    }
                } else if let Some(arr) = value.as_array() {
                    if arr.is_empty() {
                        result.add_warning(format!("Field '{}' is an empty array", field));
                    }
                }
            }
        }
    }

    // Check field types
    for (field, type_spec) in field_types {
        if let Some(value) = output.get(field) {
            if !validate_type(value, type_spec) {
                result.add_error(format!(
                    "Field '{}' has wrong type. Expected: {}",
                    field, type_spec
                ));
            }
        }
    }

    result
}

/// Validate just required fields (fast path)
pub fn validate_required_fields(
    output: &HashMap<String, Value>,
    required_fields: &[String],
) -> ValidationResult {
    let mut result = ValidationResult::valid();

    for field in required_fields {
        if !output.contains_key(field) {
            result.add_error(format!("Missing required field: {}", field));
        } else if let Some(value) = output.get(field) {
            if value.is_null() {
                result.add_error(format!("Required field '{}' is null", field));
            }
        }
    }

    result
}

/// Check if value matches expected type
fn validate_type(value: &Value, type_spec: &str) -> bool {
    let spec_lower = type_spec.to_lowercase();

    match value {
        Value::String(_) => spec_lower.contains("string"),
        Value::Number(_) => {
            spec_lower.contains("number")
                || spec_lower.contains("integer")
                || spec_lower.contains("float")
        }
        Value::Bool(_) => spec_lower.contains("bool"),
        Value::Array(_) => {
            spec_lower.contains("array")
                || spec_lower.starts_with('[')
                || spec_lower.contains("list")
        }
        Value::Object(_) => spec_lower.contains("object") || spec_lower.starts_with('{'),
        Value::Null => true, // Null is handled by required check
    }
}

/// Validate confidence score
pub fn validate_confidence(confidence: f64) -> ValidationResult {
    let mut result = ValidationResult::valid();

    if confidence < 0.0 || confidence > 1.0 {
        result.add_error(format!(
            "Confidence must be between 0.0 and 1.0, got {}",
            confidence
        ));
    }

    if confidence < 0.3 {
        result.add_warning("Very low confidence score".to_string());
    }

    result
}

/// Validate step sequence
pub fn validate_step_sequence(steps: &[(u32, bool)]) -> ValidationResult {
    let mut result = ValidationResult::valid();
    let mut expected = 1u32;

    for (step_num, passed) in steps {
        if *step_num != expected {
            result.add_error(format!(
                "Step {} completed out of order (expected {})",
                step_num, expected
            ));
        }
        if !passed {
            result.add_warning(format!("Step {} did not pass validation", step_num));
        }
        expected = step_num + 1;
    }

    result
}

/// Public function for Python binding
pub fn validate_output(
    output: &HashMap<String, Value>,
    required_fields: &[String],
    field_types: &HashMap<String, String>,
) -> ValidationResult {
    validate_output_internal(output, required_fields, field_types)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_required_present() {
        let mut output = HashMap::new();
        output.insert("name".to_string(), Value::String("test".to_string()));
        output.insert("value".to_string(), Value::Number(42.into()));

        let required = vec!["name".to_string(), "value".to_string()];
        let result = validate_required_fields(&output, &required);

        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_required_missing() {
        let mut output = HashMap::new();
        output.insert("name".to_string(), Value::String("test".to_string()));

        let required = vec!["name".to_string(), "value".to_string()];
        let result = validate_required_fields(&output, &required);

        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].contains("value"));
    }

    #[test]
    fn test_validate_types() {
        let mut output = HashMap::new();
        output.insert("name".to_string(), Value::String("test".to_string()));
        output.insert("count".to_string(), Value::Number(42.into()));
        output.insert("items".to_string(), Value::Array(vec![]));

        let mut types = HashMap::new();
        types.insert("name".to_string(), "string".to_string());
        types.insert("count".to_string(), "number".to_string());
        types.insert("items".to_string(), "[string]".to_string());

        let result = validate_output_internal(&output, &[], &types);
        assert!(result.is_valid);
    }
}
