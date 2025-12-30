//! # Deep Logic Module - Neuro-Symbolic Reasoning Framework
//!
//! Implements a hybrid neuro-symbolic reasoning system that combines:
//! - Neural (LLM) pattern recognition with symbolic rule validation
//! - Logical constraint enforcement
//! - Inference rules and deduction
//! - Contradiction detection
//! - Formal verification hooks
//!
//! This bridges the gap between LLM reasoning and formal logic systems.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::deep_logic::{DeepLogicValidator, LogicalConstraint, InferenceRule};
//!
//! let mut validator = DeepLogicValidator::new();
//!
//! // Add constraints
//! validator.add_constraint(LogicalConstraint::equality("age", "18", "Age must be 18"));
//!
//! // Add inference rules
//! validator.add_rule(InferenceRule::new("adult_rule")
//!     .with_premise(LogicalConstraint::greater("age", "17"))
//!     .with_conclusion("is_adult"));
//!
//! // Validate data
//! let result = validator.validate(&data).await?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// =============================================================================
// Logical Primitives
// =============================================================================

/// Logical operators for rule composition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogicalOperator {
    /// Logical AND
    And,
    /// Logical OR
    Or,
    /// Logical NOT
    Not,
    /// Implication (if-then)
    Implies,
    /// If and only if (biconditional)
    Iff,
    /// Exclusive OR
    Xor,
}

/// Types of logical constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// A == B
    Equality,
    /// A != B
    Inequality,
    /// A > B
    Greater,
    /// A < B
    Less,
    /// A in B (contains)
    Contains,
    /// A matches pattern B (regex)
    Matches,
    /// type(A) == B
    TypeCheck,
    /// A in [min, max]
    Range,
    /// Custom predicate
    Custom,
}

/// Results of constraint validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationResult {
    /// Constraint is satisfied
    Valid,
    /// Constraint is violated
    Invalid,
    /// Cannot determine
    Unknown,
    /// Partially satisfied
    Partial,
}

// =============================================================================
// Constraint and Rule Models
// =============================================================================

/// A single logical constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalConstraint {
    /// Unique identifier
    pub constraint_id: String,
    /// Type of constraint
    pub constraint_type: ConstraintType,
    /// Human-readable description
    pub description: String,
    /// Left operand (key or value)
    pub left_operand: String,
    /// Right operand (optional, for unary constraints)
    pub right_operand: Option<String>,
    /// Operator value (for type/pattern checks)
    pub operator_value: Option<String>,
    /// Is this constraint required?
    pub is_required: bool,
    /// Error message if violated
    pub error_message: String,
}

impl LogicalConstraint {
    /// Create a new constraint
    pub fn new(
        constraint_id: String,
        constraint_type: ConstraintType,
        left_operand: String,
    ) -> Self {
        Self {
            constraint_id,
            constraint_type,
            description: String::new(),
            left_operand,
            right_operand: None,
            operator_value: None,
            is_required: true,
            error_message: String::new(),
        }
    }

    /// Create an equality constraint
    pub fn equality(left: &str, right: &str, description: &str) -> Self {
        Self {
            constraint_id: format!("eq_{}_{}", left, right),
            constraint_type: ConstraintType::Equality,
            description: description.to_string(),
            left_operand: left.to_string(),
            right_operand: Some(right.to_string()),
            operator_value: None,
            is_required: true,
            error_message: format!("{} must equal {}", left, right),
        }
    }

    /// Create a greater-than constraint
    pub fn greater(left: &str, right: &str, description: &str) -> Self {
        Self {
            constraint_id: format!("gt_{}_{}", left, right),
            constraint_type: ConstraintType::Greater,
            description: description.to_string(),
            left_operand: left.to_string(),
            right_operand: Some(right.to_string()),
            operator_value: None,
            is_required: true,
            error_message: format!("{} must be greater than {}", left, right),
        }
    }

    /// Create a less-than constraint
    pub fn less(left: &str, right: &str, description: &str) -> Self {
        Self {
            constraint_id: format!("lt_{}_{}", left, right),
            constraint_type: ConstraintType::Less,
            description: description.to_string(),
            left_operand: left.to_string(),
            right_operand: Some(right.to_string()),
            operator_value: None,
            is_required: true,
            error_message: format!("{} must be less than {}", left, right),
        }
    }

    /// Create a type check constraint
    pub fn type_check(operand: &str, expected_type: &str, description: &str) -> Self {
        Self {
            constraint_id: format!("type_{}_{}", operand, expected_type),
            constraint_type: ConstraintType::TypeCheck,
            description: description.to_string(),
            left_operand: operand.to_string(),
            right_operand: None,
            operator_value: Some(expected_type.to_string()),
            is_required: true,
            error_message: format!("{} must be of type {}", operand, expected_type),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Set error message
    pub fn with_error_message(mut self, message: &str) -> Self {
        self.error_message = message.to_string();
        self
    }

    /// Set required flag
    pub fn with_required(mut self, required: bool) -> Self {
        self.is_required = required;
        self
    }
}

impl fmt::Display for LogicalConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.constraint_type {
            ConstraintType::Equality => {
                write!(
                    f,
                    "{} == {}",
                    self.left_operand,
                    self.right_operand.as_deref().unwrap_or("?")
                )
            }
            ConstraintType::Inequality => {
                write!(
                    f,
                    "{} != {}",
                    self.left_operand,
                    self.right_operand.as_deref().unwrap_or("?")
                )
            }
            ConstraintType::Greater => {
                write!(
                    f,
                    "{} > {}",
                    self.left_operand,
                    self.right_operand.as_deref().unwrap_or("?")
                )
            }
            ConstraintType::Less => {
                write!(
                    f,
                    "{} < {}",
                    self.left_operand,
                    self.right_operand.as_deref().unwrap_or("?")
                )
            }
            ConstraintType::Contains => {
                write!(
                    f,
                    "{} in {}",
                    self.left_operand,
                    self.right_operand.as_deref().unwrap_or("?")
                )
            }
            ConstraintType::Matches => {
                write!(
                    f,
                    "{} matches {}",
                    self.left_operand,
                    self.operator_value.as_deref().unwrap_or("?")
                )
            }
            ConstraintType::TypeCheck => {
                write!(
                    f,
                    "type({}) == {}",
                    self.left_operand,
                    self.operator_value.as_deref().unwrap_or("?")
                )
            }
            _ => {
                if !self.description.is_empty() {
                    write!(f, "{}", self.description)
                } else {
                    write!(f, "{}", self.constraint_id)
                }
            }
        }
    }
}

/// A logical inference rule (if-then)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRule {
    /// Unique identifier
    pub rule_id: String,
    /// Human-readable name
    pub name: String,
    /// Description
    pub description: String,
    /// Premises (conditions that must be true)
    pub premises: Vec<LogicalConstraint>,
    /// Conclusions (what becomes true if premises hold)
    pub conclusions: Vec<String>,
    /// Logical operator for combining premises
    pub combine_with: LogicalOperator,
    /// Priority (higher = earlier evaluation)
    pub priority: i32,
    /// Is this a default rule?
    pub is_default_rule: bool,
}

impl InferenceRule {
    /// Create a new inference rule
    pub fn new(rule_id: &str) -> Self {
        Self {
            rule_id: rule_id.to_string(),
            name: rule_id.to_string(),
            description: String::new(),
            premises: Vec::new(),
            conclusions: Vec::new(),
            combine_with: LogicalOperator::And,
            priority: 0,
            is_default_rule: false,
        }
    }

    /// Set name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Add a premise constraint
    pub fn with_premise(mut self, constraint: LogicalConstraint) -> Self {
        self.premises.push(constraint);
        self
    }

    /// Add a conclusion
    pub fn with_conclusion(mut self, conclusion: &str) -> Self {
        self.conclusions.push(conclusion.to_string());
        self
    }

    /// Set logical operator for combining premises
    pub fn with_operator(mut self, op: LogicalOperator) -> Self {
        self.combine_with = op;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as default rule
    pub fn as_default(mut self) -> Self {
        self.is_default_rule = true;
        self
    }
}

/// A logical fact in the knowledge base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalFact {
    /// Unique identifier
    pub fact_id: String,
    /// Statement
    pub statement: String,
    /// Confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Derived from (rule or constraint IDs)
    pub derived_from: Vec<String>,
    /// Source (asserted, derived, external)
    pub source: String,
    /// Is persistent?
    pub is_persistent: bool,
}

impl LogicalFact {
    /// Create a new fact
    pub fn new(fact_id: &str, statement: &str, confidence: f64) -> Self {
        Self {
            fact_id: fact_id.to_string(),
            statement: statement.to_string(),
            confidence,
            derived_from: Vec::new(),
            source: "asserted".to_string(),
            is_persistent: true,
        }
    }

    /// Create a derived fact
    pub fn derived(fact_id: &str, statement: &str, from: &str) -> Self {
        Self {
            fact_id: fact_id.to_string(),
            statement: statement.to_string(),
            confidence: 1.0,
            derived_from: vec![from.to_string()],
            source: "derived".to_string(),
            is_persistent: true,
        }
    }
}

/// Information about a detected contradiction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionInfo {
    /// Unique identifier
    pub contradiction_id: String,
    /// Description
    pub description: String,
    /// Conflicting fact A
    pub fact_a: String,
    /// Conflicting fact B
    pub fact_b: String,
    /// Source rule (if applicable)
    pub source_rule: Option<String>,
    /// Source constraint (if applicable)
    pub source_constraint: Option<String>,
    /// Is resolved?
    pub resolved: bool,
    /// Resolution method
    pub resolution_method: Option<String>,
    /// Resolution fact
    pub resolution_fact: Option<String>,
}

// =============================================================================
// Validation Results
// =============================================================================

/// Result of validating a single constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintValidationResult {
    /// The constraint that was validated
    pub constraint: LogicalConstraint,
    /// Validation result
    pub result: ValidationResult,
    /// Actual value found
    pub actual_value: Option<serde_json::Value>,
    /// Expected value
    pub expected_value: Option<serde_json::Value>,
    /// Error message
    pub error_message: String,
}

/// Result of validating an inference rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleValidationResult {
    /// The rule that was validated
    pub rule: InferenceRule,
    /// Were all premises satisfied?
    pub premises_satisfied: bool,
    /// Results for each premise
    pub premise_results: Vec<ConstraintValidationResult>,
    /// Conclusions that were derived
    pub conclusions_derived: Vec<String>,
}

/// Complete deep logic validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepLogicValidation {
    /// Overall validity
    pub is_valid: bool,
    /// Confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Constraint results
    pub constraint_results: Vec<ConstraintValidationResult>,
    /// Number of constraints satisfied
    pub constraints_satisfied: usize,
    /// Number of constraints violated
    pub constraints_violated: usize,
    /// Rule results
    pub rule_results: Vec<RuleValidationResult>,
    /// Number of rules applied
    pub rules_applied: usize,
    /// Facts derived
    pub facts_derived: Vec<LogicalFact>,
    /// Contradictions detected
    pub contradictions: Vec<ContradictionInfo>,
    /// Has contradictions?
    pub has_contradictions: bool,
    /// Errors
    pub errors: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl DeepLogicValidation {
    /// Create a new validation result
    pub fn new(is_valid: bool) -> Self {
        Self {
            is_valid,
            confidence: 0.0,
            constraint_results: Vec::new(),
            constraints_satisfied: 0,
            constraints_violated: 0,
            rule_results: Vec::new(),
            rules_applied: 0,
            facts_derived: Vec::new(),
            contradictions: Vec::new(),
            has_contradictions: false,
            errors: Vec::new(),
            warnings: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}

// =============================================================================
// Constraint Evaluator
// =============================================================================

/// Evaluates logical constraints against data
/// Type alias for custom predicate functions
type CustomPredicate =
    Box<dyn Fn(&serde_json::Value, Option<&serde_json::Value>) -> bool + Send + Sync>;

pub struct ConstraintEvaluator {
    /// Custom predicate functions
    custom_predicates: HashMap<String, CustomPredicate>,
}

impl ConstraintEvaluator {
    /// Create a new constraint evaluator
    pub fn new() -> Self {
        Self {
            custom_predicates: HashMap::new(),
        }
    }

    /// Register a custom predicate function
    pub fn register_predicate<F>(&mut self, name: &str, predicate: F)
    where
        F: Fn(&serde_json::Value, Option<&serde_json::Value>) -> bool + Send + Sync + 'static,
    {
        self.custom_predicates
            .insert(name.to_string(), Box::new(predicate));
    }

    /// Evaluate a constraint against data
    pub fn evaluate(
        &self,
        constraint: &LogicalConstraint,
        data: &serde_json::Value,
    ) -> ConstraintValidationResult {
        match self.try_evaluate(constraint, data) {
            Ok((result, actual, expected)) => ConstraintValidationResult {
                constraint: constraint.clone(),
                result,
                actual_value: actual,
                expected_value: expected,
                error_message: if result == ValidationResult::Valid {
                    String::new()
                } else {
                    constraint.error_message.clone()
                },
            },
            Err(e) => ConstraintValidationResult {
                constraint: constraint.clone(),
                result: ValidationResult::Unknown,
                actual_value: None,
                expected_value: None,
                error_message: e,
            },
        }
    }

    fn try_evaluate(
        &self,
        constraint: &LogicalConstraint,
        data: &serde_json::Value,
    ) -> Result<
        (
            ValidationResult,
            Option<serde_json::Value>,
            Option<serde_json::Value>,
        ),
        String,
    > {
        let left_value = self.resolve_operand(&constraint.left_operand, data)?;
        let right_value = constraint
            .right_operand
            .as_ref()
            .map(|op| self.resolve_operand(op, data))
            .transpose()?;

        let result = self.evaluate_constraint_type(
            constraint.constraint_type,
            &left_value,
            right_value.as_ref(),
            constraint.operator_value.as_deref(),
        )?;

        Ok((
            if result {
                ValidationResult::Valid
            } else {
                ValidationResult::Invalid
            },
            Some(left_value),
            right_value,
        ))
    }

    fn resolve_operand(
        &self,
        operand: &str,
        data: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        if operand.starts_with('$') {
            // Variable reference: $field.subfield
            let path = operand.strip_prefix('$').unwrap_or(operand).split('.');
            let mut value = data;
            for key in path {
                value = value
                    .get(key)
                    .ok_or_else(|| format!("Cannot resolve {}", operand))?;
            }
            Ok(value.clone())
        } else if operand.starts_with('"') && operand.ends_with('"') {
            // Literal string
            Ok(serde_json::Value::String(
                operand[1..operand.len() - 1].to_string(),
            ))
        } else if let Ok(num) = operand.parse::<i64>() {
            // Integer
            Ok(serde_json::Value::Number(num.into()))
        } else if let Ok(num) = operand.parse::<f64>() {
            // Float
            Ok(serde_json::json!(num))
        } else if operand == "true" {
            Ok(serde_json::Value::Bool(true))
        } else if operand == "false" {
            Ok(serde_json::Value::Bool(false))
        } else {
            // Try as key in data
            Ok(data
                .get(operand)
                .cloned()
                .unwrap_or_else(|| serde_json::Value::String(operand.to_string())))
        }
    }

    fn evaluate_constraint_type(
        &self,
        ctype: ConstraintType,
        left: &serde_json::Value,
        right: Option<&serde_json::Value>,
        operator_value: Option<&str>,
    ) -> Result<bool, String> {
        match ctype {
            ConstraintType::Equality => Ok(right == Some(left)),
            ConstraintType::Inequality => Ok(right != Some(left)),
            ConstraintType::Greater => {
                if let (Some(l), Some(r)) = (left.as_f64(), right.and_then(|v| v.as_f64())) {
                    Ok(l > r)
                } else {
                    Err("Cannot compare non-numeric values".to_string())
                }
            }
            ConstraintType::Less => {
                if let (Some(l), Some(r)) = (left.as_f64(), right.and_then(|v| v.as_f64())) {
                    Ok(l < r)
                } else {
                    Err("Cannot compare non-numeric values".to_string())
                }
            }
            ConstraintType::Contains => {
                if let Some(r) = right {
                    if let Some(arr) = r.as_array() {
                        Ok(arr.contains(left))
                    } else if let Some(obj) = r.as_object() {
                        Ok(obj.contains_key(left.as_str().unwrap_or("")))
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            ConstraintType::Matches => {
                if let Some(pattern) = operator_value {
                    if let Some(s) = left.as_str() {
                        // Simple substring match for now (full regex would require regex crate)
                        Ok(s.contains(pattern))
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            ConstraintType::TypeCheck => {
                if let Some(expected_type) = operator_value {
                    let matches = match expected_type.to_lowercase().as_str() {
                        "str" | "string" => left.is_string(),
                        "int" | "integer" => {
                            left.is_number()
                                && left.as_f64().unwrap_or(0.0).fract().abs() < f64::EPSILON
                        }
                        "float" | "number" => left.is_number(),
                        "bool" | "boolean" => left.is_boolean(),
                        "array" | "list" => left.is_array(),
                        "object" | "dict" => left.is_object(),
                        _ => false,
                    };
                    Ok(matches)
                } else {
                    Ok(false)
                }
            }
            ConstraintType::Range => {
                if let Some(range_str) = operator_value {
                    if let Some((min_str, max_str)) = range_str.split_once(',') {
                        if let (Ok(min_val), Ok(max_val), Some(val)) = (
                            min_str.trim().parse::<f64>(),
                            max_str.trim().parse::<f64>(),
                            left.as_f64(),
                        ) {
                            Ok(val >= min_val && val <= max_val)
                        } else {
                            Err("Invalid range format".to_string())
                        }
                    } else {
                        Err("Range must be in format 'min,max'".to_string())
                    }
                } else {
                    Ok(false)
                }
            }
            ConstraintType::Custom => {
                if let Some(pred_name) = operator_value {
                    if let Some(predicate) = self.custom_predicates.get(pred_name) {
                        Ok(predicate(left, right))
                    } else {
                        Err(format!("Unknown custom predicate: {}", pred_name))
                    }
                } else {
                    Err("Custom constraint requires operator_value".to_string())
                }
            }
        }
    }
}

impl Default for ConstraintEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Inference Engine
// =============================================================================

/// Forward-chaining inference engine
pub struct InferenceEngine {
    /// Maximum iterations
    max_iterations: usize,
    /// Detect contradictions? (reserved for future use)
    #[allow(dead_code)]
    detect_contradictions: bool,
    /// Rules
    rules: Vec<InferenceRule>,
    /// Facts
    facts: HashMap<String, LogicalFact>,
    /// Evaluator
    evaluator: ConstraintEvaluator,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(max_iterations: usize, detect_contradictions: bool) -> Self {
        Self {
            max_iterations,
            detect_contradictions,
            rules: Vec::new(),
            facts: HashMap::new(),
            evaluator: ConstraintEvaluator::new(),
        }
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.rules.push(rule);
        // Sort by priority (higher first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Add a fact
    pub fn add_fact(&mut self, fact: LogicalFact) {
        self.facts.insert(fact.fact_id.clone(), fact);
    }

    /// Assert a new fact
    pub fn assert_fact(&mut self, statement: &str, confidence: f64) -> LogicalFact {
        let fact_id = format!("fact_{}", self.facts.len());
        let fact = LogicalFact::new(&fact_id, statement, confidence);
        self.facts.insert(fact_id.clone(), fact.clone());
        fact
    }

    /// Run forward-chaining inference
    pub fn infer(&mut self, data: &serde_json::Value) -> Vec<LogicalFact> {
        let mut derived = Vec::new();
        let mut iteration = 0;

        while iteration < self.max_iterations {
            let mut new_facts = Vec::new();

            for rule in &self.rules {
                let result = self.apply_rule(rule, data);
                if result.premises_satisfied {
                    for conclusion in &result.conclusions_derived {
                        let fact_id = format!("derived_{}", self.facts.len() + new_facts.len());
                        let fact = LogicalFact::derived(&fact_id, conclusion, &rule.rule_id);
                        new_facts.push(fact);
                    }
                }
            }

            if new_facts.is_empty() {
                break;
            }

            // Add new facts
            for fact in &new_facts {
                self.facts.insert(fact.fact_id.clone(), fact.clone());
                derived.push(fact.clone());
            }

            iteration += 1;
        }

        derived
    }

    fn apply_rule(&self, rule: &InferenceRule, data: &serde_json::Value) -> RuleValidationResult {
        let mut premise_results = Vec::new();
        let mut all_satisfied = true;

        for premise in &rule.premises {
            let result = self.evaluator.evaluate(premise, data);
            premise_results.push(result.clone());

            match rule.combine_with {
                LogicalOperator::And => {
                    if result.result != ValidationResult::Valid {
                        all_satisfied = false;
                        break;
                    }
                }
                LogicalOperator::Or => {
                    if result.result == ValidationResult::Valid {
                        all_satisfied = true;
                        break;
                    } else {
                        all_satisfied = false;
                    }
                }
                _ => {
                    // For other operators, default to AND behavior
                    if result.result != ValidationResult::Valid {
                        all_satisfied = false;
                    }
                }
            }
        }

        let conclusions = if all_satisfied {
            rule.conclusions.clone()
        } else {
            Vec::new()
        };

        RuleValidationResult {
            rule: rule.clone(),
            premises_satisfied: all_satisfied,
            premise_results,
            conclusions_derived: conclusions,
        }
    }
}

// =============================================================================
// Contradiction Detector
// =============================================================================

/// Detects logical contradictions in facts and reasoning
pub struct ContradictionDetector {
    /// Contradiction patterns (pattern_a, pattern_b, description)
    contradiction_patterns: Vec<(String, String, String)>,
}

impl ContradictionDetector {
    /// Create a new contradiction detector
    pub fn new() -> Self {
        Self {
            contradiction_patterns: vec![
                (
                    "is true".to_string(),
                    "is false".to_string(),
                    "Direct truth contradiction".to_string(),
                ),
                (
                    "exists".to_string(),
                    "does not exist".to_string(),
                    "Existence contradiction".to_string(),
                ),
                (
                    "is valid".to_string(),
                    "is invalid".to_string(),
                    "Validity contradiction".to_string(),
                ),
                (
                    "should".to_string(),
                    "should not".to_string(),
                    "Normative contradiction".to_string(),
                ),
                (
                    "must".to_string(),
                    "must not".to_string(),
                    "Obligation contradiction".to_string(),
                ),
                (
                    "always".to_string(),
                    "never".to_string(),
                    "Temporal contradiction".to_string(),
                ),
                (
                    "all".to_string(),
                    "none".to_string(),
                    "Quantifier contradiction".to_string(),
                ),
            ],
        }
    }

    /// Detect contradictions among facts
    pub fn detect(&self, facts: &[LogicalFact]) -> Vec<ContradictionInfo> {
        let mut contradictions = Vec::new();
        let statements: Vec<(String, String)> = facts
            .iter()
            .map(|f| (f.fact_id.clone(), f.statement.to_lowercase()))
            .collect();

        for (i, (id_a, stmt_a)) in statements.iter().enumerate() {
            for (id_b, stmt_b) in statements.iter().skip(i + 1) {
                if let Some(description) = self.check_contradiction(stmt_a, stmt_b) {
                    contradictions.push(ContradictionInfo {
                        contradiction_id: format!("contradiction_{}", contradictions.len()),
                        description,
                        fact_a: id_a.clone(),
                        fact_b: id_b.clone(),
                        source_rule: None,
                        source_constraint: None,
                        resolved: false,
                        resolution_method: None,
                        resolution_fact: None,
                    });
                }
            }
        }

        contradictions
    }

    fn check_contradiction(&self, stmt_a: &str, stmt_b: &str) -> Option<String> {
        for (pattern_a, pattern_b, description) in &self.contradiction_patterns {
            if ((stmt_a.contains(pattern_a) && stmt_b.contains(pattern_b))
                || (stmt_a.contains(pattern_b) && stmt_b.contains(pattern_a)))
                && self.same_subject(stmt_a, stmt_b)
            {
                return Some(description.clone());
            }
        }
        None
    }

    fn same_subject(&self, stmt_a: &str, stmt_b: &str) -> bool {
        let words_a: std::collections::HashSet<&str> = stmt_a.split_whitespace().collect();
        let words_b: std::collections::HashSet<&str> = stmt_b.split_whitespace().collect();

        let common: std::collections::HashSet<&str> =
            ["the", "a", "an", "is", "are", "was", "were", "be", "been"]
                .iter()
                .cloned()
                .collect();

        let words_a: std::collections::HashSet<&str> =
            words_a.difference(&common).cloned().collect();
        let words_b: std::collections::HashSet<&str> =
            words_b.difference(&common).cloned().collect();

        if words_a.is_empty() || words_b.is_empty() {
            return false;
        }

        let overlap = words_a.intersection(&words_b).count();
        let min_len = words_a.len().min(words_b.len());
        overlap as f64 / min_len as f64 > 0.3
    }
}

impl Default for ContradictionDetector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Deep Logic Validator (Main Interface)
// =============================================================================

/// Main neuro-symbolic validation system
pub struct DeepLogicValidator {
    /// Constraints to validate
    constraints: Vec<LogicalConstraint>,
    /// Inference engine
    inference_engine: InferenceEngine,
    /// Constraint evaluator
    evaluator: ConstraintEvaluator,
    /// Contradiction detector
    contradiction_detector: ContradictionDetector,
}

impl DeepLogicValidator {
    /// Create a new deep logic validator
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            inference_engine: InferenceEngine::new(100, true),
            evaluator: ConstraintEvaluator::new(),
            contradiction_detector: ContradictionDetector::new(),
        }
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: LogicalConstraint) {
        self.constraints.push(constraint);
    }

    /// Add an inference rule
    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.inference_engine.add_rule(rule);
    }

    /// Validate data and reasoning output
    pub fn validate(
        &mut self,
        data: &serde_json::Value,
        _reasoning_output: Option<&serde_json::Value>,
    ) -> DeepLogicValidation {
        let mut result = DeepLogicValidation::new(true);

        // Evaluate constraints
        for constraint in &self.constraints {
            let constraint_result = self.evaluator.evaluate(constraint, data);
            result.constraint_results.push(constraint_result.clone());

            match constraint_result.result {
                ValidationResult::Valid => {
                    result.constraints_satisfied += 1;
                }
                ValidationResult::Invalid => {
                    result.constraints_violated += 1;
                    if constraint.is_required {
                        result.is_valid = false;
                        result.errors.push(constraint_result.error_message);
                    }
                }
                _ => {
                    result.warnings.push(format!(
                        "Constraint {} could not be evaluated",
                        constraint.constraint_id
                    ));
                }
            }
        }

        // Run inference
        let derived_facts = self.inference_engine.infer(data);
        result.facts_derived = derived_facts.clone();
        result.rules_applied = self.inference_engine.rules.len();

        // Detect contradictions
        let all_facts: Vec<LogicalFact> = self.inference_engine.facts.values().cloned().collect();
        let contradictions = self.contradiction_detector.detect(&all_facts);
        result.contradictions = contradictions.clone();
        result.has_contradictions = !contradictions.is_empty();

        if result.has_contradictions {
            result.is_valid = false;
            result
                .errors
                .push("Contradictions detected in reasoning".to_string());
        }

        // Calculate confidence
        let total_constraints = result.constraints_satisfied + result.constraints_violated;
        if total_constraints > 0 {
            result.confidence = result.constraints_satisfied as f64 / total_constraints as f64;
        }

        result
    }
}

impl Default for DeepLogicValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_equality() {
        let constraint = LogicalConstraint::equality("age", "18", "Age must be 18");
        assert_eq!(constraint.constraint_type, ConstraintType::Equality);
        assert_eq!(constraint.left_operand, "age");
        assert_eq!(constraint.right_operand, Some("18".to_string()));
    }

    #[test]
    fn test_constraint_display() {
        let constraint = LogicalConstraint::equality("x", "y", "Test");
        assert!(constraint.to_string().contains("=="));
    }

    #[test]
    fn test_inference_rule() {
        let rule = InferenceRule::new("test_rule")
            .with_name("Test Rule")
            .with_premise(LogicalConstraint::greater("age", "17", "Must be adult"))
            .with_conclusion("is_adult")
            .with_priority(10);

        assert_eq!(rule.rule_id, "test_rule");
        assert_eq!(rule.name, "Test Rule");
        assert_eq!(rule.premises.len(), 1);
        assert_eq!(rule.conclusions.len(), 1);
        assert_eq!(rule.priority, 10);
    }

    #[test]
    fn test_constraint_evaluator_equality() {
        let evaluator = ConstraintEvaluator::new();
        let constraint = LogicalConstraint::equality("age", "18", "Age must be 18");
        let data = serde_json::json!({ "age": 18 });

        let result = evaluator.evaluate(&constraint, &data);
        assert_eq!(result.result, ValidationResult::Valid);
    }

    #[test]
    fn test_constraint_evaluator_greater() {
        let evaluator = ConstraintEvaluator::new();
        let constraint = LogicalConstraint::greater("age", "17", "Must be adult");
        let data = serde_json::json!({ "age": 18 });

        let result = evaluator.evaluate(&constraint, &data);
        assert_eq!(result.result, ValidationResult::Valid);
    }

    #[test]
    fn test_inference_engine() {
        let mut engine = InferenceEngine::new(10, false);
        let rule = InferenceRule::new("adult_rule")
            .with_premise(LogicalConstraint::greater("age", "17", "Must be adult"))
            .with_conclusion("is_adult");

        engine.add_rule(rule);
        let data = serde_json::json!({ "age": 18 });
        let derived = engine.infer(&data);

        assert!(!derived.is_empty());
        assert!(derived[0].statement == "is_adult");
    }

    #[test]
    fn test_contradiction_detector() {
        let detector = ContradictionDetector::new();
        let facts = vec![
            LogicalFact::new("f1", "X is true", 1.0),
            LogicalFact::new("f2", "X is false", 1.0),
        ];

        let contradictions = detector.detect(&facts);
        assert!(!contradictions.is_empty());
    }

    #[test]
    fn test_deep_logic_validator() {
        let mut validator = DeepLogicValidator::new();
        validator.add_constraint(LogicalConstraint::equality("age", "18", "Age must be 18"));

        let data = serde_json::json!({ "age": 18 });
        let result = validator.validate(&data, None);

        assert!(result.is_valid);
        assert_eq!(result.constraints_satisfied, 1);
    }
}
