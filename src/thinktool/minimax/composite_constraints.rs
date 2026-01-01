//! Composite Instruction Constraints for MiniMax M2
//!
//! Implements M2's composite instruction constraints including:
//! - System Prompts
//! - User Queries
//! - Memory Context
//! - Tool Schemas
//!
//! Single-violation-failure scoring for robust constraint management.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Composite instruction constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositeInstruction {
    SystemPrompt(SystemPrompt),
    UserQuery(UserQuery),
    MemoryContext(MemoryContext),
    ToolSchema(ToolSchema),
}

/// System prompt constraint with validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPrompt {
    pub template: String,
    pub constraints: Vec<PromptConstraint>,
    pub variables: HashMap<String, String>,
    pub token_limit: Option<u32>,
}

/// User query constraint with preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuery {
    pub raw_text: String,
    pub sanitized_text: String,
    pub intent: QueryIntent,
    pub complexity_score: f64,
    pub required_tools: Vec<String>,
}

/// Memory context constraint with retention rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    pub context_id: String,
    pub content: String,
    pub relevance_score: f64,
    pub retention_policy: RetentionPolicy,
    pub dependencies: Vec<String>,
}

/// Tool schema constraint with validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub tool_name: String,
    pub input_schema: SchemaDefinition,
    pub output_schema: SchemaDefinition,
    pub constraints: Vec<FieldConstraint>,
}

/// Prompt constraint validation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PromptConstraint {
    MaxTokens(u32),
    MinConfidence(f64),
    RequiredKeywords(Vec<String>),
    ForbiddenKeywords(Vec<String>),
    StyleGuide(String),
}

/// Query intent classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryIntent {
    Creative,
    Analytical,
    Verification,
    Decision,
    Explanation,
    Critique,
}

/// Memory retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionPolicy {
    Session,
    ShortTerm(u32), // minutes
    LongTerm(u32),  // days
    Permanent,
}

/// Schema definition for tool inputs/outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    pub format: SchemaFormat,
    pub fields: Vec<SchemaField>,
    pub validation_rules: Vec<ValidationRule>,
}

/// Schema field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaField {
    pub name: String,
    pub field_type: FieldType,
    pub required: bool,
    pub constraints: Vec<FieldConstraint>,
}

/// Field constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldConstraint {
    MinLength(u32),
    MaxLength(u32),
    Pattern(String),
    Enum(Vec<String>),
    Range(f64, f64),
}

/// Schema format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaFormat {
    JSON,
    XML,
    YAML,
    PlainText,
}

/// Field type definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    Array(Box<FieldType>),
    Object,
}

/// Validation rule for schemas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub severity: ViolationSeverity,
}

/// Constraint violation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub instruction_type: String,
    pub violation_type: String,
    pub description: String,
    pub severity: ViolationSeverity,
    pub location: Option<String>,
}

/// Violation severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViolationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Single-violation-failure result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintResult {
    Passed(f64), // Score from 0.0 to 1.0
    Failed(Vec<ConstraintViolation>),
    Pending,
}

/// Constraint engine for M2 composite instructions
pub struct ConstraintEngine {
    constraints: HashMap<String, CompositeInstruction>,
    #[allow(dead_code)]
    violation_history: Vec<ConstraintViolation>,
    #[allow(dead_code)]
    performance_tracker: PerformanceTracker,
}

impl Default for ConstraintEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintEngine {
    /// Create new constraint engine
    pub fn new() -> Self {
        Self {
            constraints: HashMap::new(),
            violation_history: Vec::new(),
            performance_tracker: PerformanceTracker::new(),
        }
    }

    /// Add composite instruction constraint
    pub fn add_constraint(&mut self, id: String, instruction: CompositeInstruction) {
        self.constraints.insert(id, instruction);
    }

    /// Validate all constraints with single-violation-failure
    pub fn validate_all(&self, inputs: &ValidationInputs) -> ConstraintResult {
        let mut violations = Vec::new();
        let mut total_score = 1.0;

        // Validate system prompt constraints
        if let Some(system_prompt) = inputs.system_prompt.as_ref() {
            match self.validate_system_prompt(system_prompt) {
                Ok(score) => {
                    total_score *= score;
                }
                Err(violation) => {
                    if violation.severity == ViolationSeverity::Critical {
                        violations.push(violation);
                        return ConstraintResult::Failed(violations);
                    }
                    violations.push(violation);
                }
            }
        }

        // Validate user query constraints
        if let Some(user_query) = inputs.user_query.as_ref() {
            match self.validate_user_query(user_query) {
                Ok(score) => {
                    total_score *= score;
                }
                Err(violation) => {
                    if violation.severity == ViolationSeverity::Critical {
                        violations.push(violation);
                        return ConstraintResult::Failed(violations);
                    }
                    violations.push(violation);
                }
            }
        }

        // Validate memory context constraints
        if let Some(memory_context) = inputs.memory_context.as_ref() {
            match self.validate_memory_context(memory_context) {
                Ok(score) => {
                    total_score *= score;
                }
                Err(violation) => {
                    if violation.severity == ViolationSeverity::Critical {
                        violations.push(violation);
                        return ConstraintResult::Failed(violations);
                    }
                    violations.push(violation);
                }
            }
        }

        // Validate tool schema constraints
        for (tool_name, schema) in &inputs.tool_schemas {
            match self.validate_tool_schema(tool_name, schema) {
                Ok(score) => {
                    total_score *= score;
                }
                Err(violation) => {
                    if violation.severity == ViolationSeverity::Critical {
                        violations.push(violation);
                        return ConstraintResult::Failed(violations);
                    }
                    violations.push(violation);
                }
            }
        }

        if !violations.is_empty() {
            ConstraintResult::Failed(violations)
        } else {
            ConstraintResult::Passed(total_score)
        }
    }

    /// Validate system prompt constraints
    fn validate_system_prompt(&self, prompt: &SystemPrompt) -> Result<f64, ConstraintViolation> {
        let mut score = 1.0;

        // Check token limit
        if let Some(limit) = prompt.token_limit {
            let token_count = self.count_tokens(&prompt.template);
            if token_count > limit {
                return Err(ConstraintViolation {
                    instruction_type: "SystemPrompt".to_string(),
                    violation_type: "TokenLimitExceeded".to_string(),
                    description: format!("Prompt exceeds token limit: {} > {}", token_count, limit),
                    severity: ViolationSeverity::Error,
                    location: Some("template".to_string()),
                });
            }
            score *= 0.9; // Penalize for being close to limit
        }

        // Check required keywords
        for constraint in &prompt.constraints {
            match constraint {
                PromptConstraint::RequiredKeywords(keywords) => {
                    for keyword in keywords {
                        if !prompt
                            .template
                            .to_lowercase()
                            .contains(&keyword.to_lowercase())
                        {
                            return Err(ConstraintViolation {
                                instruction_type: "SystemPrompt".to_string(),
                                violation_type: "MissingRequiredKeyword".to_string(),
                                description: format!("Required keyword '{}' not found", keyword),
                                severity: ViolationSeverity::Warning,
                                location: Some("template".to_string()),
                            });
                        }
                    }
                }
                PromptConstraint::ForbiddenKeywords(keywords) => {
                    for keyword in keywords {
                        if prompt
                            .template
                            .to_lowercase()
                            .contains(&keyword.to_lowercase())
                        {
                            return Err(ConstraintViolation {
                                instruction_type: "SystemPrompt".to_string(),
                                violation_type: "ForbiddenKeyword".to_string(),
                                description: format!("Forbidden keyword '{}' found", keyword),
                                severity: ViolationSeverity::Error,
                                location: Some("template".to_string()),
                            });
                        }
                    }
                }
                _ => {} // Other constraints handled elsewhere
            }
        }

        Ok(score)
    }

    /// Validate user query constraints
    fn validate_user_query(&self, query: &UserQuery) -> Result<f64, ConstraintViolation> {
        let mut score = 1.0;

        // Check complexity score
        if query.complexity_score > 1.0 {
            return Err(ConstraintViolation {
                instruction_type: "UserQuery".to_string(),
                violation_type: "ComplexityTooHigh".to_string(),
                description: format!(
                    "Query complexity score too high: {}",
                    query.complexity_score
                ),
                severity: ViolationSeverity::Warning,
                location: None,
            });
        }

        // Check for required tools
        for tool in &query.required_tools {
            if !self.is_tool_available(tool) {
                return Err(ConstraintViolation {
                    instruction_type: "UserQuery".to_string(),
                    violation_type: "UnavailableTool".to_string(),
                    description: format!("Required tool '{}' not available", tool),
                    severity: ViolationSeverity::Critical,
                    location: None,
                });
            }
        }

        // Check sanitization
        if query.raw_text != query.sanitized_text {
            score *= 0.8; // Penalize for needing sanitization
        }

        Ok(score)
    }

    /// Validate memory context constraints
    fn validate_memory_context(&self, context: &MemoryContext) -> Result<f64, ConstraintViolation> {
        let score = 1.0;

        // Check relevance score
        if context.relevance_score < 0.3 {
            return Err(ConstraintViolation {
                instruction_type: "MemoryContext".to_string(),
                violation_type: "LowRelevance".to_string(),
                description: format!(
                    "Memory context relevance too low: {}",
                    context.relevance_score
                ),
                severity: ViolationSeverity::Warning,
                location: None,
            });
        }

        // Check retention policy compatibility
        if let RetentionPolicy::Permanent = &context.retention_policy {
            // Permanent memory should have high relevance
            if context.relevance_score < 0.8 {
                return Err(ConstraintViolation {
                    instruction_type: "MemoryContext".to_string(),
                    violation_type: "PermanentMemoryLowRelevance".to_string(),
                    description: "Permanent memory must have high relevance".to_string(),
                    severity: ViolationSeverity::Error,
                    location: None,
                });
            }
        }

        Ok(score)
    }

    /// Validate tool schema constraints
    fn validate_tool_schema(
        &self,
        tool_name: &str,
        schema: &ToolSchema,
    ) -> Result<f64, ConstraintViolation> {
        let score = 1.0;

        // Validate input schema against field constraints
        for constraint in &schema.constraints {
            // Each FieldConstraint applies to the schema as a whole
            if !self.validate_field_constraint(&schema.input_schema, constraint) {
                return Err(ConstraintViolation {
                    instruction_type: "ToolSchema".to_string(),
                    violation_type: "SchemaValidationFailed".to_string(),
                    description: format!(
                        "Schema validation failed for tool {} with constraint {:?}",
                        tool_name, constraint
                    ),
                    severity: ViolationSeverity::Error,
                    location: Some("input_schema".to_string()),
                });
            }
        }

        Ok(score)
    }

    /// Helper methods
    fn count_tokens(&self, text: &str) -> u32 {
        // Simple token counting - in real implementation, use proper tokenizer
        text.split_whitespace().count() as u32
    }

    fn is_tool_available(&self, tool_name: &str) -> bool {
        // Check if tool is available in the current context
        // This would integrate with the actual tool registry
        matches!(
            tool_name,
            "gigathink" | "laserlogic" | "bedrock" | "proofguard" | "brutalhonesty"
        )
    }

    #[allow(dead_code)]
    fn validate_schema_rule(&self, _schema: &SchemaDefinition, _rule: &ValidationRule) -> bool {
        // Implement schema rule validation
        // This would check actual data against the schema rules
        true // Placeholder
    }

    fn validate_field_constraint(
        &self,
        schema: &SchemaDefinition,
        constraint: &FieldConstraint,
    ) -> bool {
        // Validate schema against field constraints
        match constraint {
            FieldConstraint::MinLength(min) => {
                schema.fields.iter().all(|f| f.name.len() >= *min as usize)
            }
            FieldConstraint::MaxLength(max) => {
                schema.fields.iter().all(|f| f.name.len() <= *max as usize)
            }
            FieldConstraint::Pattern(_pattern) => true, // Would need regex validation
            FieldConstraint::Enum(_values) => true,     // Would check if values are in enum
            FieldConstraint::Range(_min, _max) => true, // Would check numeric ranges
        }
    }
}

/// Input data for constraint validation
#[derive(Debug, Clone)]
pub struct ValidationInputs<'a> {
    pub system_prompt: Option<&'a SystemPrompt>,
    pub user_query: Option<&'a UserQuery>,
    pub memory_context: Option<&'a MemoryContext>,
    pub tool_schemas: HashMap<&'a str, &'a ToolSchema>,
}

impl<'a> Default for ValidationInputs<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ValidationInputs<'a> {
    pub fn new() -> Self {
        Self {
            system_prompt: None,
            user_query: None,
            memory_context: None,
            tool_schemas: HashMap::new(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: &'a SystemPrompt) -> Self {
        self.system_prompt = Some(prompt);
        self
    }

    pub fn with_user_query(mut self, query: &'a UserQuery) -> Self {
        self.user_query = Some(query);
        self
    }

    pub fn with_memory_context(mut self, context: &'a MemoryContext) -> Self {
        self.memory_context = Some(context);
        self
    }

    pub fn add_tool_schema(mut self, name: &'a str, schema: &'a ToolSchema) -> Self {
        self.tool_schemas.insert(name, schema);
        self
    }
}

/// Performance tracking for constraint engine
pub struct PerformanceTracker {
    total_validations: u64,
    successful_validations: u64,
    average_score: f64,
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            total_validations: 0,
            successful_validations: 0,
            average_score: 1.0,
        }
    }

    pub fn record_validation(&mut self, result: &ConstraintResult) {
        self.total_validations += 1;

        if let ConstraintResult::Passed(score) = result {
            self.successful_validations += 1;
            self.average_score = (self.average_score * (self.total_validations - 1) as f64 + score)
                / self.total_validations as f64;
        }
    }

    pub fn get_success_rate(&self) -> f64 {
        if self.total_validations > 0 {
            self.successful_validations as f64 / self.total_validations as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_engine_creation() {
        let engine = ConstraintEngine::new();
        assert_eq!(engine.constraints.len(), 0);
    }

    #[test]
    fn test_system_prompt_validation() {
        let engine = ConstraintEngine::new();

        let prompt = SystemPrompt {
            template: "This is a test prompt with required keyword".to_string(),
            constraints: vec![PromptConstraint::RequiredKeywords(vec![
                "required".to_string()
            ])],
            variables: HashMap::new(),
            token_limit: Some(100),
        };

        let inputs = ValidationInputs::new().with_system_prompt(&prompt);
        let result = engine.validate_all(&inputs);

        match result {
            ConstraintResult::Passed(_) => {
                // Should pass with required keyword
            }
            ConstraintResult::Failed(violations) => {
                panic!("Unexpected validation failure: {:?}", violations);
            }
            ConstraintResult::Pending => panic!("Expected immediate result"),
        }
    }

    #[test]
    fn test_single_violation_failure() {
        let engine = ConstraintEngine::new();

        let prompt = SystemPrompt {
            template: "Forbidden word here".to_string(),
            constraints: vec![PromptConstraint::ForbiddenKeywords(vec![
                "forbidden".to_string()
            ])],
            variables: HashMap::new(),
            token_limit: None,
        };

        let inputs = ValidationInputs::new().with_system_prompt(&prompt);
        let result = engine.validate_all(&inputs);

        match result {
            ConstraintResult::Failed(violations) => {
                assert_eq!(violations.len(), 1);
                assert_eq!(violations[0].violation_type, "ForbiddenKeyword");
            }
            _ => panic!("Expected failure for forbidden keyword"),
        }
    }
}
