//! # Language Parser Module
//!
//! Parses source code across 9+ programming languages using MiniMax M2's multi-language mastery.
//! Provides unified AST representation for code analysis.

use crate::code_intelligence::*;
use crate::error::Error;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, instrument};

/// Unified AST representation for all supported languages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedAST {
    pub language: ProgrammingLanguage,
    pub functions: Vec<FunctionNode>,
    pub classes: Vec<ClassNode>,
    pub variables: Vec<VariableNode>,
    pub imports: Vec<ImportNode>,
    pub comments: Vec<CommentNode>,
    pub complexity_metrics: ComplexityMetrics,
}

/// Function node in AST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionNode {
    pub name: String,
    pub parameters: Vec<ParameterNode>,
    pub return_type: Option<String>,
    pub body: Vec<StatementNode>,
    pub visibility: Visibility,
    pub is_async: bool,
    pub line_number: u32,
    pub complexity: ComplexityMetrics,
}

/// Class node in AST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassNode {
    pub name: String,
    pub methods: Vec<FunctionNode>,
    pub properties: Vec<VariableNode>,
    pub inheritance: Vec<String>,
    pub interfaces: Vec<String>,
    pub visibility: Visibility,
    pub line_number: u32,
}

/// Variable node in AST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableNode {
    pub name: String,
    pub data_type: Option<String>,
    pub initializer: Option<String>,
    pub visibility: Visibility,
    pub is_mutable: bool,
    pub line_number: u32,
}

/// Parameter node in function signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterNode {
    pub name: String,
    pub data_type: Option<String>,
    pub default_value: Option<String>,
    pub is_variadic: bool,
}

/// Import/use statement node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportNode {
    pub module: String,
    pub items: Vec<String>,
    pub alias: Option<String>,
    pub line_number: u32,
}

/// Comment node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentNode {
    pub content: String,
    pub line_number: u32,
    pub comment_type: CommentType,
}

/// Statement node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatementNode {
    pub statement_type: StatementType,
    pub content: String,
    pub line_number: u32,
    pub complexity: ComplexityMetrics,
}

/// Complexity metrics for code elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: f64,
    pub cognitive_complexity: f64,
    pub nesting_depth: u32,
    pub lines_of_code: u32,
    pub halstead_complexity: HalsteadMetrics,
}

/// Halstead complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HalsteadMetrics {
    pub operators: HashMap<String, u32>,
    pub operands: HashMap<String, u32>,
    pub vocabulary: usize,
    pub length: usize,
    pub volume: f64,
    pub difficulty: f64,
}

/// Enums for AST representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    Private,
    Protected,
    Package,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommentType {
    SingleLine,
    MultiLine,
    Documentation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatementType {
    Assignment,
    Declaration,
    Expression,
    ControlFlow,
    Loop,
    Exception,
    Return,
    Break,
    Continue,
}

/// Language parser trait
#[async_trait]
pub trait LanguageParser: Send + Sync {
    /// Parse source code into unified AST
    async fn parse(&self, code: &str) -> Result<UnifiedAST, Error>;

    /// Get language-specific analysis features
    fn get_language_features(&self) -> LanguageFeatures;

    /// Detect language from code content
    fn detect_language(&self, code: &str) -> f64;
}

/// Language-specific features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageFeatures {
    pub language: ProgrammingLanguage,
    pub supports_oop: bool,
    pub supports_generics: bool,
    pub supports_async: bool,
    pub supports_patterns: bool,
    pub type_system: TypeSystem,
    pub memory_management: MemoryManagement,
    pub concurrency_model: ConcurrencyModel,
}

/// Type system characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeSystem {
    Static,
    Dynamic,
    Gradual,
    Structural,
}

/// Memory management model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryManagement {
    Manual,
    Automatic,
    GarbageCollected,
    ReferenceCounted,
}

/// Concurrency model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcurrencyModel {
    Threading,
    AsyncAwait,
    MessagePassing,
    ActorModel,
}

// ============================================================================
// LANGUAGE-SPECIFIC PARSERS
// ============================================================================

/// Rust language parser
pub struct RustParser {
    language_features: LanguageFeatures,
}

impl RustParser {
    pub fn new() -> Self {
        Self {
            language_features: LanguageFeatures {
                language: ProgrammingLanguage::Rust,
                supports_oop: false,
                supports_generics: true,
                supports_async: true,
                supports_patterns: true,
                type_system: TypeSystem::Static,
                memory_management: MemoryManagement::Manual,
                concurrency_model: ConcurrencyModel::AsyncAwait,
            },
        }
    }
}

impl Default for RustParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LanguageParser for RustParser {
    #[instrument(skip(self, code))]
    async fn parse(&self, code: &str) -> Result<UnifiedAST, Error> {
        info!("Parsing Rust code ({} characters)", code.len());

        // Simplified Rust parsing - in production would use rust-analyzer
        let functions = self.extract_rust_functions(code)?;
        let variables = self.extract_rust_variables(code)?;
        let imports = self.extract_rust_imports(code)?;
        let comments = self.extract_rust_comments(code)?;

        let complexity_metrics = self.calculate_rust_complexity(code, &functions)?;

        Ok(UnifiedAST {
            language: ProgrammingLanguage::Rust,
            functions,
            classes: vec![], // Rust doesn't have traditional classes
            variables,
            imports,
            comments,
            complexity_metrics,
        })
    }

    fn get_language_features(&self) -> LanguageFeatures {
        self.language_features.clone()
    }

    fn detect_language(&self, code: &str) -> f64 {
        // Check for Rust-specific patterns
        let rust_keywords = ["fn", "let", "mut", "impl", "struct", "enum", "trait"];
        let mut score = 0.0f64;

        for keyword in &rust_keywords {
            if code.contains(keyword) {
                score += 0.15;
            }
        }

        // Check for Rust-specific syntax
        if code.contains("::") || code.contains("=>") || code.contains("'") {
            score += 0.1;
        }

        score.min(1.0)
    }
}

impl RustParser {
    fn extract_rust_functions(&self, code: &str) -> Result<Vec<FunctionNode>, Error> {
        let mut functions = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        let fn_regex = regex::Regex::new(r"fn\s+(\w+)\s*\(").unwrap();

        for (i, line) in lines.iter().enumerate() {
            if let Some(captures) = fn_regex.captures(line) {
                let function_name = captures[1].to_string();
                let line_number = (i + 1) as u32;

                functions.push(FunctionNode {
                    name: function_name,
                    parameters: vec![], // Would parse parameters in production
                    return_type: None,
                    body: vec![],
                    visibility: Visibility::Public,
                    is_async: false,
                    line_number,
                    complexity: ComplexityMetrics::default(),
                });
            }
        }

        Ok(functions)
    }

    fn extract_rust_variables(&self, code: &str) -> Result<Vec<VariableNode>, Error> {
        let mut variables = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        let let_regex = regex::Regex::new(r"let(?:\s+mut)?\s+(\w+)").unwrap();

        for (i, line) in lines.iter().enumerate() {
            if line.trim_start().starts_with("let") || line.trim_start().starts_with("let mut") {
                let is_mutable = line.trim_start().starts_with("let mut");
                if let Some(captures) = let_regex.captures(line) {
                    let var_name = captures[1].to_string();

                    variables.push(VariableNode {
                        name: var_name,
                        data_type: None,
                        initializer: None,
                        visibility: Visibility::Private,
                        is_mutable,
                        line_number: (i + 1) as u32,
                    });
                }
            }
        }

        Ok(variables)
    }

    fn extract_rust_imports(&self, code: &str) -> Result<Vec<ImportNode>, Error> {
        let mut imports = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            if line.trim_start().starts_with("use ") {
                let import_line = line.trim_start();
                let module = import_line[4..].trim().to_string();

                imports.push(ImportNode {
                    module,
                    items: vec![],
                    alias: None,
                    line_number: (i + 1) as u32,
                });
            }
        }

        Ok(imports)
    }

    fn extract_rust_comments(&self, code: &str) -> Result<Vec<CommentNode>, Error> {
        let mut comments = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if let Some(stripped) = trimmed.strip_prefix("//") {
                let content = stripped.trim().to_string();
                comments.push(CommentNode {
                    content,
                    line_number: (i + 1) as u32,
                    comment_type: CommentType::SingleLine,
                });
            } else if trimmed.starts_with("/*") && trimmed.ends_with("*/") {
                let content = trimmed[2..trimmed.len() - 2].trim().to_string();
                comments.push(CommentNode {
                    content,
                    line_number: (i + 1) as u32,
                    comment_type: CommentType::MultiLine,
                });
            }
        }

        Ok(comments)
    }

    fn calculate_rust_complexity(
        &self,
        code: &str,
        _functions: &[FunctionNode],
    ) -> Result<ComplexityMetrics, Error> {
        let lines_of_code = code.lines().count() as u32;

        // Simplified complexity calculation
        let cyclomatic_complexity = self.calculate_cyclomatic_complexity(code);
        let cognitive_complexity = cyclomatic_complexity * 1.2; // Rust typically has lower cognitive complexity
        let nesting_depth = self.calculate_nesting_depth(code);

        Ok(ComplexityMetrics {
            cyclomatic_complexity,
            cognitive_complexity,
            nesting_depth,
            lines_of_code,
            halstead_complexity: self.calculate_halstead_complexity(code),
        })
    }

    fn calculate_cyclomatic_complexity(&self, code: &str) -> f64 {
        let complexity_keywords = [
            "if", "else", "else if", "while", "for", "match", "||", "&&", "?",
        ];
        let mut complexity = 1.0; // Base complexity

        for keyword in &complexity_keywords {
            complexity += code.matches(keyword).count() as f64;
        }

        complexity
    }

    fn calculate_nesting_depth(&self, code: &str) -> u32 {
        let mut max_depth: u32 = 0;
        let mut current_depth: u32 = 0;

        for ch in code.chars() {
            match ch {
                '{' => {
                    current_depth += 1;
                    max_depth = max_depth.max(current_depth);
                }
                '}' => {
                    current_depth = current_depth.saturating_sub(1);
                }
                _ => {}
            }
        }

        max_depth
    }

    fn calculate_halstead_complexity(&self, code: &str) -> HalsteadMetrics {
        let mut operators = HashMap::new();
        let mut operands = HashMap::new();

        // Simplified operator/operand identification
        let operator_patterns = [
            "+", "-", "*", "/", "=", "==", "!=", "&&", "||", "<", ">", "<=", ">=",
        ];
        let word_patterns = regex::Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b").unwrap();

        for operator in &operator_patterns {
            let count = code.matches(operator).count() as u32;
            if count > 0 {
                operators.insert(operator.to_string(), count);
            }
        }

        for word in word_patterns.find_iter(code) {
            let word_str = word.as_str().to_string();
            if ![
                "let", "fn", "struct", "enum", "impl", "if", "else", "while", "for", "match",
            ]
            .contains(&word_str.as_str())
            {
                *operands.entry(word_str).or_insert(0) += 1;
            }
        }

        let vocabulary = operators.len() + operands.len();
        let length = operators.values().sum::<u32>() + operands.values().sum::<u32>();
        let volume = if length > 0 {
            length as f64 * (length as f64).log2()
        } else {
            0.0
        };
        let difficulty = if vocabulary > 0 {
            (operators.len() as f64 / 2.0) * (operands.len() as f64 / vocabulary as f64)
        } else {
            0.0
        };

        HalsteadMetrics {
            operators,
            operands,
            vocabulary,
            length: length as usize,
            volume,
            difficulty,
        }
    }
}

/// Java language parser
pub struct JavaParser {
    language_features: LanguageFeatures,
}

impl JavaParser {
    pub fn new() -> Self {
        Self {
            language_features: LanguageFeatures {
                language: ProgrammingLanguage::Java,
                supports_oop: true,
                supports_generics: true,
                supports_async: false, // Java doesn't have built-in async/await
                supports_patterns: true,
                type_system: TypeSystem::Static,
                memory_management: MemoryManagement::GarbageCollected,
                concurrency_model: ConcurrencyModel::Threading,
            },
        }
    }
}

impl Default for JavaParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LanguageParser for JavaParser {
    async fn parse(&self, code: &str) -> Result<UnifiedAST, Error> {
        info!("Parsing Java code ({} characters)", code.len());

        let functions = self.extract_java_methods(code)?;
        let classes = self.extract_java_classes(code)?;
        let variables = self.extract_java_variables(code)?;
        let imports = self.extract_java_imports(code)?;
        let comments = self.extract_java_comments(code)?;

        let complexity_metrics = self.calculate_java_complexity(code, &functions)?;

        Ok(UnifiedAST {
            language: ProgrammingLanguage::Java,
            functions,
            classes,
            variables,
            imports,
            comments,
            complexity_metrics,
        })
    }

    fn get_language_features(&self) -> LanguageFeatures {
        self.language_features.clone()
    }

    fn detect_language(&self, code: &str) -> f64 {
        let mut score = 0.0f64;

        // Check for Java-specific patterns
        if code.contains("public class") || code.contains("class ") {
            score += 0.3;
        }
        if code.contains("public static void main") {
            score += 0.4;
        }
        if code.contains("import ") {
            score += 0.2;
        }
        if code.contains(";") {
            score += 0.1;
        }

        score.min(1.0)
    }
}

impl JavaParser {
    fn extract_java_methods(&self, code: &str) -> Result<Vec<FunctionNode>, Error> {
        let mut methods = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        let method_regex =
            regex::Regex::new(r"(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\(")
                .unwrap();

        for (i, line) in lines.iter().enumerate() {
            if let Some(captures) = method_regex.captures(line) {
                let return_type = captures[3].to_string();
                let method_name = captures[4].to_string();
                let line_number = (i + 1) as u32;

                methods.push(FunctionNode {
                    name: method_name,
                    parameters: vec![],
                    return_type: Some(return_type),
                    body: vec![],
                    visibility: self
                        .parse_java_visibility(captures.get(1).map(|m| m.as_str()).unwrap_or("")),
                    is_async: false,
                    line_number,
                    complexity: ComplexityMetrics::default(),
                });
            }
        }

        Ok(methods)
    }

    fn extract_java_classes(&self, code: &str) -> Result<Vec<ClassNode>, Error> {
        let mut classes = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        let class_regex =
            regex::Regex::new(r"(public|private|protected)?\s*class\s+(\w+)").unwrap();

        for (i, line) in lines.iter().enumerate() {
            if let Some(captures) = class_regex.captures(line) {
                let class_name = captures[2].to_string();
                let line_number = (i + 1) as u32;

                classes.push(ClassNode {
                    name: class_name,
                    methods: vec![],
                    properties: vec![],
                    inheritance: vec![],
                    interfaces: vec![],
                    visibility: self
                        .parse_java_visibility(captures.get(1).map(|m| m.as_str()).unwrap_or("")),
                    line_number,
                });
            }
        }

        Ok(classes)
    }

    fn extract_java_variables(&self, code: &str) -> Result<Vec<VariableNode>, Error> {
        let mut variables = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        let var_regex =
            regex::Regex::new(r"(public|private|protected)?\s*(\w+)\s+(\w+)\s*(?:=|;)").unwrap();

        for (i, line) in lines.iter().enumerate() {
            if let Some(captures) = var_regex.captures(line) {
                let data_type = captures[2].to_string();
                let var_name = captures[3].to_string();

                variables.push(VariableNode {
                    name: var_name,
                    data_type: Some(data_type),
                    initializer: None,
                    visibility: self
                        .parse_java_visibility(captures.get(1).map(|m| m.as_str()).unwrap_or("")),
                    is_mutable: true,
                    line_number: (i + 1) as u32,
                });
            }
        }

        Ok(variables)
    }

    fn extract_java_imports(&self, code: &str) -> Result<Vec<ImportNode>, Error> {
        let mut imports = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            if line.trim_start().starts_with("import ") {
                let import_line = line.trim_start();
                let module = import_line[7..].trim().trim_end_matches(';').to_string();

                imports.push(ImportNode {
                    module,
                    items: vec![],
                    alias: None,
                    line_number: (i + 1) as u32,
                });
            }
        }

        Ok(imports)
    }

    fn extract_java_comments(&self, code: &str) -> Result<Vec<CommentNode>, Error> {
        let mut comments = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if let Some(stripped) = trimmed.strip_prefix("//") {
                let content = stripped.trim().to_string();
                comments.push(CommentNode {
                    content,
                    line_number: (i + 1) as u32,
                    comment_type: CommentType::SingleLine,
                });
            } else if trimmed.starts_with("/*") && trimmed.ends_with("*/") {
                let content = trimmed[2..trimmed.len() - 2].trim().to_string();
                comments.push(CommentNode {
                    content,
                    line_number: (i + 1) as u32,
                    comment_type: CommentType::MultiLine,
                });
            }
        }

        Ok(comments)
    }

    fn parse_java_visibility(&self, visibility: &str) -> Visibility {
        match visibility.trim() {
            "public" => Visibility::Public,
            "private" => Visibility::Private,
            "protected" => Visibility::Protected,
            _ => Visibility::Package,
        }
    }

    fn calculate_java_complexity(
        &self,
        code: &str,
        _methods: &[FunctionNode],
    ) -> Result<ComplexityMetrics, Error> {
        let lines_of_code = code.lines().count() as u32;
        let cyclomatic_complexity = self.calculate_cyclomatic_complexity(code);
        let cognitive_complexity = cyclomatic_complexity * 1.3; // Java typically has higher cognitive complexity
        let nesting_depth = self.calculate_nesting_depth(code);

        Ok(ComplexityMetrics {
            cyclomatic_complexity,
            cognitive_complexity,
            nesting_depth,
            lines_of_code,
            halstead_complexity: self.calculate_halstead_complexity(code),
        })
    }

    fn calculate_cyclomatic_complexity(&self, code: &str) -> f64 {
        let complexity_keywords = [
            "if", "else", "else if", "while", "for", "switch", "case", "catch", "&&", "||", "?",
        ];
        let mut complexity = 1.0;

        for keyword in &complexity_keywords {
            complexity += code.matches(keyword).count() as f64;
        }

        complexity
    }

    fn calculate_nesting_depth(&self, code: &str) -> u32 {
        let mut max_depth: u32 = 0;
        let mut current_depth: u32 = 0;

        for ch in code.chars() {
            match ch {
                '{' => {
                    current_depth += 1;
                    max_depth = max_depth.max(current_depth);
                }
                '}' => {
                    current_depth = current_depth.saturating_sub(1);
                }
                _ => {}
            }
        }

        max_depth
    }

    fn calculate_halstead_complexity(&self, code: &str) -> HalsteadMetrics {
        let mut operators = HashMap::new();
        let mut operands = HashMap::new();

        let operator_patterns = [
            "+", "-", "*", "/", "=", "==", "!=", "&&", "||", "<", ">", "<=", ">=", "+=", "-=",
            "*=", "/=",
        ];
        let word_patterns = regex::Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b").unwrap();

        for operator in &operator_patterns {
            let count = code.matches(operator).count() as u32;
            if count > 0 {
                operators.insert(operator.to_string(), count);
            }
        }

        for word in word_patterns.find_iter(code) {
            let word_str = word.as_str().to_string();
            if ![
                "public",
                "private",
                "protected",
                "static",
                "class",
                "if",
                "else",
                "while",
                "for",
                "switch",
                "case",
            ]
            .contains(&word_str.as_str())
            {
                *operands.entry(word_str).or_insert(0) += 1;
            }
        }

        let vocabulary = operators.len() + operands.len();
        let length = operators.values().sum::<u32>() + operands.values().sum::<u32>();
        let volume = if length > 0 {
            length as f64 * (length as f64).log2()
        } else {
            0.0
        };
        let difficulty = if vocabulary > 0 {
            (operators.len() as f64 / 2.0) * (operands.len() as f64 / vocabulary as f64)
        } else {
            0.0
        };

        HalsteadMetrics {
            operators,
            operands,
            vocabulary,
            length: length as usize,
            volume,
            difficulty,
        }
    }
}

// ============================================================================
// PLACEHOLDER PARSERS FOR OTHER LANGUAGES
// ============================================================================

pub struct GolangParser;
pub struct CppParser;
pub struct KotlinParser;
pub struct ObjectiveCParser;
pub struct TypeScriptParser;
pub struct JavaScriptParser;
pub struct PythonParser;

// Implement basic parser traits for other languages
macro_rules! implement_basic_parser {
    ($parser_name:ident, $language:expr, $features:expr) => {
        impl $parser_name {
            pub fn new() -> Self {
                Self
            }
        }

        impl Default for $parser_name {
            fn default() -> Self {
                Self::new()
            }
        }

        #[async_trait]
        impl LanguageParser for $parser_name {
            async fn parse(&self, code: &str) -> Result<UnifiedAST, Error> {
                info!(
                    "Parsing {} code ({} characters)",
                    stringify!($language),
                    code.len()
                );

                // Simplified parsing - would implement full parser in production
                Ok(UnifiedAST {
                    language: $language,
                    functions: vec![],
                    classes: vec![],
                    variables: vec![],
                    imports: vec![],
                    comments: vec![],
                    complexity_metrics: ComplexityMetrics::default(),
                })
            }

            fn get_language_features(&self) -> LanguageFeatures {
                $features
            }

            fn detect_language(&self, _code: &str) -> f64 {
                0.0
            }
        }
    };
}

// Basic language features for each language
fn golang_features() -> LanguageFeatures {
    LanguageFeatures {
        language: ProgrammingLanguage::Golang,
        supports_oop: false,
        supports_generics: true,
        supports_async: true,
        supports_patterns: true,
        type_system: TypeSystem::Static,
        memory_management: MemoryManagement::Manual,
        concurrency_model: ConcurrencyModel::AsyncAwait,
    }
}

fn cpp_features() -> LanguageFeatures {
    LanguageFeatures {
        language: ProgrammingLanguage::Cpp,
        supports_oop: true,
        supports_generics: true,
        supports_async: true,
        supports_patterns: true,
        type_system: TypeSystem::Static,
        memory_management: MemoryManagement::Manual,
        concurrency_model: ConcurrencyModel::Threading,
    }
}

implement_basic_parser!(GolangParser, ProgrammingLanguage::Golang, golang_features());
implement_basic_parser!(CppParser, ProgrammingLanguage::Cpp, cpp_features());
implement_basic_parser!(
    KotlinParser,
    ProgrammingLanguage::Kotlin,
    LanguageFeatures {
        language: ProgrammingLanguage::Kotlin,
        supports_oop: true,
        supports_generics: true,
        supports_async: true,
        supports_patterns: true,
        type_system: TypeSystem::Static,
        memory_management: MemoryManagement::GarbageCollected,
        concurrency_model: ConcurrencyModel::AsyncAwait,
    }
);
implement_basic_parser!(
    ObjectiveCParser,
    ProgrammingLanguage::ObjectiveC,
    LanguageFeatures {
        language: ProgrammingLanguage::ObjectiveC,
        supports_oop: true,
        supports_generics: true,
        supports_async: false,
        supports_patterns: true,
        type_system: TypeSystem::Dynamic,
        memory_management: MemoryManagement::ReferenceCounted,
        concurrency_model: ConcurrencyModel::Threading,
    }
);
implement_basic_parser!(
    TypeScriptParser,
    ProgrammingLanguage::TypeScript,
    LanguageFeatures {
        language: ProgrammingLanguage::TypeScript,
        supports_oop: true,
        supports_generics: true,
        supports_async: true,
        supports_patterns: true,
        type_system: TypeSystem::Static,
        memory_management: MemoryManagement::GarbageCollected,
        concurrency_model: ConcurrencyModel::AsyncAwait,
    }
);
implement_basic_parser!(
    JavaScriptParser,
    ProgrammingLanguage::JavaScript,
    LanguageFeatures {
        language: ProgrammingLanguage::JavaScript,
        supports_oop: true,
        supports_generics: false,
        supports_async: true,
        supports_patterns: true,
        type_system: TypeSystem::Dynamic,
        memory_management: MemoryManagement::GarbageCollected,
        concurrency_model: ConcurrencyModel::AsyncAwait,
    }
);
implement_basic_parser!(
    PythonParser,
    ProgrammingLanguage::Python,
    LanguageFeatures {
        language: ProgrammingLanguage::Python,
        supports_oop: true,
        supports_generics: true,
        supports_async: true,
        supports_patterns: true,
        type_system: TypeSystem::Dynamic,
        memory_management: MemoryManagement::GarbageCollected,
        concurrency_model: ConcurrencyModel::AsyncAwait,
    }
);

impl Default for ComplexityMetrics {
    fn default() -> Self {
        Self {
            cyclomatic_complexity: 1.0,
            cognitive_complexity: 1.0,
            nesting_depth: 1,
            lines_of_code: 1,
            halstead_complexity: HalsteadMetrics {
                operators: HashMap::new(),
                operands: HashMap::new(),
                vocabulary: 0,
                length: 0,
                volume: 0.0,
                difficulty: 0.0,
            },
        }
    }
}

// ============================================================================
// COMPREHENSIVE TEST SUITE
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // TEST FIXTURES
    // ========================================================================

    mod fixtures {
        /// Rust code fixture: simple function with variable
        pub const RUST_SIMPLE: &str = r#"
fn main() {
    let mut x = 42;
    if x > 0 {
        println!("Positive");
    }
}
"#;

        /// Rust code fixture: multiple functions with imports
        pub const RUST_COMPLEX: &str = r#"
use std::collections::HashMap;
use crate::error::Error;

// This is a comment
fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

fn process_data(data: &[u8]) -> Result<(), Error> {
    let mut buffer = Vec::new();
    for byte in data {
        if *byte > 0 {
            buffer.push(*byte);
        }
    }
    Ok(())
}

/* Multi-line comment */
fn main() {
    let result = calculate_sum(1, 2);
    println!("{}", result);
}
"#;

        /// Rust code fixture: async function
        pub const RUST_ASYNC: &str = r#"
use tokio;

async fn fetch_data() -> Result<String, Error> {
    let response = reqwest::get("https://example.com").await?;
    Ok(response.text().await?)
}

async fn main() {
    let data = fetch_data().await.unwrap();
}
"#;

        /// Rust code fixture: struct with impl
        pub const RUST_STRUCT: &str = r#"
struct Calculator {
    value: i32,
}

impl Calculator {
    fn new() -> Self {
        Self { value: 0 }
    }

    fn add(&mut self, n: i32) {
        self.value += n;
    }
}
"#;

        /// Java code fixture: simple class
        pub const JAVA_SIMPLE: &str = r#"
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"#;

        /// Java code fixture: complex class with inheritance
        pub const JAVA_COMPLEX: &str = r#"
import java.util.List;
import java.util.ArrayList;

// Calculator class
public class Calculator extends BaseCalculator {
    private int value;
    protected String name;

    public Calculator() {
        this.value = 0;
    }

    public int add(int a, int b) {
        return a + b;
    }

    private void reset() {
        this.value = 0;
    }

    /* Multi-line comment */
    protected int getValue() {
        return this.value;
    }
}
"#;

        /// Java code fixture: main method
        pub const JAVA_MAIN: &str = r#"
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        for (int i = 0; i < 10; i++) {
            if (i % 2 == 0) {
                System.out.println(i);
            }
        }
    }
}
"#;

        /// Python code fixture
        pub const PYTHON_SIMPLE: &str = r#"
def calculate_sum(a, b):
    return a + b

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, n):
        self.value += n

if __name__ == "__main__":
    calc = Calculator()
    calc.add(5)
"#;

        /// JavaScript code fixture
        pub const JAVASCRIPT_SIMPLE: &str = r#"
function calculateSum(a, b) {
    return a + b;
}

const calculator = {
    value: 0,
    add(n) {
        this.value += n;
    }
};

async function fetchData() {
    const response = await fetch('https://api.example.com');
    return response.json();
}
"#;

        /// Empty code fixture
        pub const EMPTY_CODE: &str = "";

        /// Whitespace-only code fixture
        pub const WHITESPACE_ONLY: &str = "   \n\t\n   ";

        /// Malformed Rust code fixture
        pub const RUST_MALFORMED: &str = r#"
fn incomplete {
    let x =
    if {
}
"#;

        /// Deeply nested code fixture
        pub const DEEPLY_NESTED: &str = r#"
fn deeply_nested() {
    if true {
        if true {
            if true {
                if true {
                    if true {
                        println!("Deep!");
                    }
                }
            }
        }
    }
}
"#;

        /// Code with many control flow statements
        pub const HIGH_COMPLEXITY: &str = r#"
fn complex_function(x: i32, y: i32) -> i32 {
    if x > 0 && y > 0 {
        if x > y {
            return x;
        } else if y > x {
            return y;
        } else {
            return x + y;
        }
    } else if x < 0 || y < 0 {
        for i in 0..10 {
            if i == 5 {
                break;
            }
            match i {
                0 => println!("zero"),
                1 => println!("one"),
                _ => println!("other"),
            }
        }
        return -1;
    }
    0
}
"#;
    }

    // ========================================================================
    // RUST PARSER TESTS
    // ========================================================================

    mod rust_parser_tests {
        use super::*;

        #[tokio::test]
        async fn test_parse_simple_rust_code() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_SIMPLE).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::Rust);
            assert!(!ast.functions.is_empty());
        }

        #[tokio::test]
        async fn test_parse_complex_rust_code() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_COMPLEX).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::Rust);
            assert_eq!(ast.functions.len(), 3); // calculate_sum, process_data, main
        }

        #[tokio::test]
        async fn test_extract_rust_functions() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_COMPLEX).await.unwrap();

            // Assert
            let function_names: Vec<&str> =
                result.functions.iter().map(|f| f.name.as_str()).collect();
            assert!(function_names.contains(&"calculate_sum"));
            assert!(function_names.contains(&"process_data"));
            assert!(function_names.contains(&"main"));
        }

        #[tokio::test]
        async fn test_extract_rust_function_line_numbers() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_COMPLEX).await.unwrap();

            // Assert
            let main_fn = result.functions.iter().find(|f| f.name == "main").unwrap();
            assert!(main_fn.line_number > 0);
        }

        #[tokio::test]
        async fn test_extract_rust_variables() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_SIMPLE).await.unwrap();

            // Assert
            assert!(!result.variables.is_empty());
            let var_x = result.variables.iter().find(|v| v.name == "x");
            assert!(var_x.is_some());
            assert!(var_x.unwrap().is_mutable);
        }

        #[tokio::test]
        async fn test_extract_rust_mutable_vs_immutable() {
            // Arrange
            let parser = RustParser::new();
            let code = r#"
fn test() {
    let immutable = 1;
    let mut mutable = 2;
}
"#;

            // Act
            let result = parser.parse(code).await.unwrap();

            // Assert
            let immutable_var = result.variables.iter().find(|v| v.name == "immutable");
            let mutable_var = result.variables.iter().find(|v| v.name == "mutable");

            assert!(immutable_var.is_some());
            assert!(!immutable_var.unwrap().is_mutable);
            assert!(mutable_var.is_some());
            assert!(mutable_var.unwrap().is_mutable);
        }

        #[tokio::test]
        async fn test_extract_rust_imports() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_COMPLEX).await.unwrap();

            // Assert
            assert!(!result.imports.is_empty());
            let import_modules: Vec<&str> =
                result.imports.iter().map(|i| i.module.as_str()).collect();
            assert!(import_modules.iter().any(|m| m.contains("HashMap")));
            assert!(import_modules.iter().any(|m| m.contains("Error")));
        }

        #[tokio::test]
        async fn test_extract_rust_single_line_comments() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_COMPLEX).await.unwrap();

            // Assert
            let single_line_comments: Vec<&CommentNode> = result
                .comments
                .iter()
                .filter(|c| c.comment_type == CommentType::SingleLine)
                .collect();
            assert!(!single_line_comments.is_empty());
            assert!(single_line_comments
                .iter()
                .any(|c| c.content.contains("comment")));
        }

        #[tokio::test]
        async fn test_extract_rust_multiline_comments() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_COMPLEX).await.unwrap();

            // Assert
            let multiline_comments: Vec<&CommentNode> = result
                .comments
                .iter()
                .filter(|c| c.comment_type == CommentType::MultiLine)
                .collect();
            assert!(!multiline_comments.is_empty());
        }

        #[tokio::test]
        async fn test_parse_empty_code() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::EMPTY_CODE).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert!(ast.functions.is_empty());
            assert!(ast.variables.is_empty());
            assert!(ast.imports.is_empty());
        }

        #[tokio::test]
        async fn test_parse_whitespace_only() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::WHITESPACE_ONLY).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert!(ast.functions.is_empty());
        }

        #[tokio::test]
        async fn test_parse_malformed_rust_code() {
            // Arrange
            let parser = RustParser::new();

            // Act - parser should handle malformed code gracefully
            let result = parser.parse(fixtures::RUST_MALFORMED).await;

            // Assert - should not panic, may return partial results
            assert!(result.is_ok());
        }

        #[test]
        fn test_rust_language_detection_high_confidence() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let score = parser.detect_language(fixtures::RUST_COMPLEX);

            // Assert - Rust code should have high detection score
            assert!(score >= 0.5);
        }

        #[test]
        fn test_rust_language_detection_low_for_java() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let rust_score = parser.detect_language(fixtures::RUST_SIMPLE);
            let java_score = parser.detect_language(fixtures::JAVA_SIMPLE);

            // Assert
            assert!(rust_score > java_score);
        }

        #[test]
        fn test_rust_language_features() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let features = parser.get_language_features();

            // Assert
            assert_eq!(features.language, ProgrammingLanguage::Rust);
            assert!(!features.supports_oop); // Rust uses traits, not OOP
            assert!(features.supports_generics);
            assert!(features.supports_async);
            assert!(features.supports_patterns);
            assert_eq!(features.type_system, TypeSystem::Static);
            assert_eq!(features.memory_management, MemoryManagement::Manual);
            assert_eq!(features.concurrency_model, ConcurrencyModel::AsyncAwait);
        }

        #[test]
        fn test_rust_parser_default() {
            // Arrange & Act
            let parser = RustParser::default();

            // Assert
            let features = parser.get_language_features();
            assert_eq!(features.language, ProgrammingLanguage::Rust);
        }
    }

    // ========================================================================
    // JAVA PARSER TESTS
    // ========================================================================

    mod java_parser_tests {
        use super::*;

        #[tokio::test]
        async fn test_parse_simple_java_code() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_SIMPLE).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::Java);
        }

        #[tokio::test]
        async fn test_parse_complex_java_code() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_COMPLEX).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert!(!ast.classes.is_empty());
            assert!(!ast.functions.is_empty());
        }

        #[tokio::test]
        async fn test_extract_java_classes() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_COMPLEX).await.unwrap();

            // Assert
            assert_eq!(result.classes.len(), 1);
            let calc_class = &result.classes[0];
            assert_eq!(calc_class.name, "Calculator");
            assert_eq!(calc_class.visibility, Visibility::Public);
        }

        #[tokio::test]
        async fn test_extract_java_methods() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_COMPLEX).await.unwrap();

            // Assert
            let method_names: Vec<&str> =
                result.functions.iter().map(|f| f.name.as_str()).collect();
            assert!(method_names.contains(&"add"));
            assert!(method_names.contains(&"reset"));
            assert!(method_names.contains(&"getValue"));
        }

        #[tokio::test]
        async fn test_extract_java_method_visibility() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_COMPLEX).await.unwrap();

            // Assert
            let add_method = result.functions.iter().find(|f| f.name == "add");
            let reset_method = result.functions.iter().find(|f| f.name == "reset");
            let get_value_method = result.functions.iter().find(|f| f.name == "getValue");

            assert!(add_method.is_some());
            assert_eq!(add_method.unwrap().visibility, Visibility::Public);

            assert!(reset_method.is_some());
            assert_eq!(reset_method.unwrap().visibility, Visibility::Private);

            assert!(get_value_method.is_some());
            assert_eq!(get_value_method.unwrap().visibility, Visibility::Protected);
        }

        #[tokio::test]
        async fn test_extract_java_method_return_types() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_COMPLEX).await.unwrap();

            // Assert
            let add_method = result.functions.iter().find(|f| f.name == "add").unwrap();
            assert!(add_method.return_type.is_some());
            assert_eq!(add_method.return_type.as_ref().unwrap(), "int");
        }

        #[tokio::test]
        async fn test_extract_java_imports() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_COMPLEX).await.unwrap();

            // Assert
            assert!(!result.imports.is_empty());
            let import_modules: Vec<&str> =
                result.imports.iter().map(|i| i.module.as_str()).collect();
            assert!(import_modules.iter().any(|m| m.contains("java.util.List")));
            assert!(import_modules
                .iter()
                .any(|m| m.contains("java.util.ArrayList")));
        }

        #[tokio::test]
        async fn test_extract_java_comments() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_COMPLEX).await.unwrap();

            // Assert
            assert!(!result.comments.is_empty());
            let single_line = result
                .comments
                .iter()
                .any(|c| c.comment_type == CommentType::SingleLine);
            let multi_line = result
                .comments
                .iter()
                .any(|c| c.comment_type == CommentType::MultiLine);
            assert!(single_line);
            assert!(multi_line);
        }

        #[tokio::test]
        async fn test_parse_java_main_method() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_MAIN).await.unwrap();

            // Assert
            let main_method = result.functions.iter().find(|f| f.name == "main");
            assert!(main_method.is_some());
        }

        #[test]
        fn test_java_language_detection_high_confidence() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let score = parser.detect_language(fixtures::JAVA_MAIN);

            // Assert - Java code with main method should have high score
            assert!(score >= 0.7);
        }

        #[test]
        fn test_java_language_detection_class_pattern() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let score = parser.detect_language("public class Test { }");

            // Assert
            assert!(score >= 0.3);
        }

        #[test]
        fn test_java_language_features() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let features = parser.get_language_features();

            // Assert
            assert_eq!(features.language, ProgrammingLanguage::Java);
            assert!(features.supports_oop);
            assert!(features.supports_generics);
            assert!(!features.supports_async); // Java uses threads, not async/await
            assert_eq!(features.type_system, TypeSystem::Static);
            assert_eq!(
                features.memory_management,
                MemoryManagement::GarbageCollected
            );
            assert_eq!(features.concurrency_model, ConcurrencyModel::Threading);
        }

        #[test]
        fn test_java_parser_default() {
            // Arrange & Act
            let parser = JavaParser::default();

            // Assert
            let features = parser.get_language_features();
            assert_eq!(features.language, ProgrammingLanguage::Java);
        }

        #[test]
        fn test_parse_java_visibility() {
            // Arrange
            let parser = JavaParser::new();

            // Act & Assert
            assert_eq!(parser.parse_java_visibility("public"), Visibility::Public);
            assert_eq!(parser.parse_java_visibility("private"), Visibility::Private);
            assert_eq!(
                parser.parse_java_visibility("protected"),
                Visibility::Protected
            );
            assert_eq!(parser.parse_java_visibility(""), Visibility::Package);
            assert_eq!(parser.parse_java_visibility("unknown"), Visibility::Package);
        }
    }

    // ========================================================================
    // COMPLEXITY METRICS TESTS
    // ========================================================================

    mod complexity_tests {
        use super::*;

        #[tokio::test]
        async fn test_complexity_metrics_simple_code() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_SIMPLE).await.unwrap();

            // Assert
            assert!(result.complexity_metrics.cyclomatic_complexity >= 1.0);
            assert!(result.complexity_metrics.lines_of_code > 0);
        }

        #[tokio::test]
        async fn test_complexity_metrics_high_complexity() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::HIGH_COMPLEXITY).await.unwrap();

            // Assert - complex code should have higher cyclomatic complexity
            assert!(result.complexity_metrics.cyclomatic_complexity > 5.0);
        }

        #[tokio::test]
        async fn test_nesting_depth_calculation() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::DEEPLY_NESTED).await.unwrap();

            // Assert - deeply nested code should have high nesting depth
            assert!(result.complexity_metrics.nesting_depth >= 5);
        }

        #[tokio::test]
        async fn test_nesting_depth_simple_code() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_SIMPLE).await.unwrap();

            // Assert
            assert!(result.complexity_metrics.nesting_depth <= 3);
        }

        #[tokio::test]
        async fn test_halstead_complexity_metrics() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_COMPLEX).await.unwrap();

            // Assert
            let halstead = &result.complexity_metrics.halstead_complexity;
            assert!(halstead.vocabulary > 0);
            assert!(halstead.length > 0);
            assert!(halstead.volume >= 0.0);
        }

        #[tokio::test]
        async fn test_lines_of_code_calculation() {
            // Arrange
            let parser = RustParser::new();
            let code = "fn a() {}\nfn b() {}\nfn c() {}";

            // Act
            let result = parser.parse(code).await.unwrap();

            // Assert
            assert_eq!(result.complexity_metrics.lines_of_code, 3);
        }

        #[test]
        fn test_complexity_metrics_default() {
            // Arrange & Act
            let metrics = ComplexityMetrics::default();

            // Assert
            assert_eq!(metrics.cyclomatic_complexity, 1.0);
            assert_eq!(metrics.cognitive_complexity, 1.0);
            assert_eq!(metrics.nesting_depth, 1);
            assert_eq!(metrics.lines_of_code, 1);
            assert!(metrics.halstead_complexity.operators.is_empty());
            assert!(metrics.halstead_complexity.operands.is_empty());
        }

        #[tokio::test]
        async fn test_cognitive_complexity_factor() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_COMPLEX).await.unwrap();

            // Assert - cognitive complexity should be derived from cyclomatic
            let cyclomatic = result.complexity_metrics.cyclomatic_complexity;
            let cognitive = result.complexity_metrics.cognitive_complexity;
            assert!((cognitive - cyclomatic * 1.2).abs() < 0.001);
        }
    }

    // ========================================================================
    // PLACEHOLDER PARSER TESTS
    // ========================================================================

    mod placeholder_parser_tests {
        use super::*;

        #[tokio::test]
        async fn test_golang_parser() {
            // Arrange
            let parser = GolangParser::new();

            // Act
            let result = parser.parse("package main\nfunc main() {}").await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::Golang);
        }

        #[tokio::test]
        async fn test_cpp_parser() {
            // Arrange
            let parser = CppParser::new();

            // Act
            let result = parser.parse("#include <iostream>").await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::Cpp);
        }

        #[tokio::test]
        async fn test_kotlin_parser() {
            // Arrange
            let parser = KotlinParser::new();

            // Act
            let result = parser.parse("fun main() {}").await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::Kotlin);
        }

        #[tokio::test]
        async fn test_typescript_parser() {
            // Arrange
            let parser = TypeScriptParser::new();

            // Act
            let result = parser.parse("const x: number = 42;").await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::TypeScript);
        }

        #[tokio::test]
        async fn test_javascript_parser() {
            // Arrange
            let parser = JavaScriptParser::new();

            // Act
            let result = parser.parse(fixtures::JAVASCRIPT_SIMPLE).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::JavaScript);
        }

        #[tokio::test]
        async fn test_python_parser() {
            // Arrange
            let parser = PythonParser::new();

            // Act
            let result = parser.parse(fixtures::PYTHON_SIMPLE).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::Python);
        }

        #[tokio::test]
        async fn test_objectivec_parser() {
            // Arrange
            let parser = ObjectiveCParser::new();

            // Act
            let result = parser.parse("@implementation Test @end").await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.language, ProgrammingLanguage::ObjectiveC);
        }

        #[test]
        fn test_placeholder_parsers_default_detection() {
            // Arrange
            let parsers: Vec<Box<dyn LanguageParser>> = vec![
                Box::new(GolangParser::new()),
                Box::new(CppParser::new()),
                Box::new(KotlinParser::new()),
                Box::new(TypeScriptParser::new()),
                Box::new(JavaScriptParser::new()),
                Box::new(PythonParser::new()),
                Box::new(ObjectiveCParser::new()),
            ];

            // Act & Assert - placeholder parsers return 0.0 for detection
            for parser in parsers {
                let score = parser.detect_language("any code");
                assert_eq!(score, 0.0);
            }
        }

        #[test]
        fn test_golang_features() {
            // Arrange
            let parser = GolangParser::new();

            // Act
            let features = parser.get_language_features();

            // Assert
            assert_eq!(features.language, ProgrammingLanguage::Golang);
            assert!(!features.supports_oop);
            assert!(features.supports_generics);
            assert_eq!(features.type_system, TypeSystem::Static);
        }

        #[test]
        fn test_cpp_features() {
            // Arrange
            let parser = CppParser::new();

            // Act
            let features = parser.get_language_features();

            // Assert
            assert_eq!(features.language, ProgrammingLanguage::Cpp);
            assert!(features.supports_oop);
            assert_eq!(features.memory_management, MemoryManagement::Manual);
            assert_eq!(features.concurrency_model, ConcurrencyModel::Threading);
        }

        #[test]
        fn test_python_features() {
            // Arrange
            let parser = PythonParser::new();

            // Act
            let features = parser.get_language_features();

            // Assert
            assert_eq!(features.language, ProgrammingLanguage::Python);
            assert_eq!(features.type_system, TypeSystem::Dynamic);
            assert_eq!(
                features.memory_management,
                MemoryManagement::GarbageCollected
            );
        }

        #[test]
        fn test_javascript_features() {
            // Arrange
            let parser = JavaScriptParser::new();

            // Act
            let features = parser.get_language_features();

            // Assert
            assert_eq!(features.language, ProgrammingLanguage::JavaScript);
            assert_eq!(features.type_system, TypeSystem::Dynamic);
            assert!(!features.supports_generics);
            assert!(features.supports_async);
        }
    }

    // ========================================================================
    // AST STRUCTURE TESTS
    // ========================================================================

    mod ast_structure_tests {
        use super::*;

        #[test]
        fn test_unified_ast_serialization() {
            // Arrange
            let ast = UnifiedAST {
                language: ProgrammingLanguage::Rust,
                functions: vec![],
                classes: vec![],
                variables: vec![],
                imports: vec![],
                comments: vec![],
                complexity_metrics: ComplexityMetrics::default(),
            };

            // Act
            let json = serde_json::to_string(&ast);

            // Assert
            assert!(json.is_ok());
        }

        #[test]
        fn test_unified_ast_deserialization() {
            // Arrange
            let json = r#"{"language":"rust","functions":[],"classes":[],"variables":[],"imports":[],"comments":[],"complexity_metrics":{"cyclomatic_complexity":1.0,"cognitive_complexity":1.0,"nesting_depth":1,"lines_of_code":1,"halstead_complexity":{"operators":{},"operands":{},"vocabulary":0,"length":0,"volume":0.0,"difficulty":0.0}}}"#;

            // Act
            let ast: Result<UnifiedAST, _> = serde_json::from_str(json);

            // Assert
            assert!(ast.is_ok());
            assert_eq!(ast.unwrap().language, ProgrammingLanguage::Rust);
        }

        #[test]
        fn test_function_node_clone() {
            // Arrange
            let node = FunctionNode {
                name: "test".to_string(),
                parameters: vec![],
                return_type: Some("i32".to_string()),
                body: vec![],
                visibility: Visibility::Public,
                is_async: false,
                line_number: 1,
                complexity: ComplexityMetrics::default(),
            };

            // Act
            let cloned = node.clone();

            // Assert
            assert_eq!(cloned.name, node.name);
            assert_eq!(cloned.return_type, node.return_type);
        }

        #[test]
        fn test_visibility_enum_variants() {
            // Assert all variants are accessible
            let _public = Visibility::Public;
            let _private = Visibility::Private;
            let _protected = Visibility::Protected;
            let _package = Visibility::Package;
        }

        #[test]
        fn test_comment_type_enum_variants() {
            // Assert all variants are accessible
            let _single = CommentType::SingleLine;
            let _multi = CommentType::MultiLine;
            let _doc = CommentType::Documentation;
        }

        #[test]
        fn test_statement_type_enum_variants() {
            // Assert all variants are accessible
            let variants = vec![
                StatementType::Assignment,
                StatementType::Declaration,
                StatementType::Expression,
                StatementType::ControlFlow,
                StatementType::Loop,
                StatementType::Exception,
                StatementType::Return,
                StatementType::Break,
                StatementType::Continue,
            ];
            assert_eq!(variants.len(), 9);
        }

        #[test]
        fn test_type_system_enum() {
            // Assert
            assert_eq!(TypeSystem::Static, TypeSystem::Static);
            assert_ne!(TypeSystem::Static, TypeSystem::Dynamic);
        }

        #[test]
        fn test_memory_management_enum() {
            // Assert
            assert_eq!(MemoryManagement::Manual, MemoryManagement::Manual);
            assert_ne!(MemoryManagement::Manual, MemoryManagement::GarbageCollected);
        }

        #[test]
        fn test_concurrency_model_enum() {
            // Assert
            let variants = vec![
                ConcurrencyModel::Threading,
                ConcurrencyModel::AsyncAwait,
                ConcurrencyModel::MessagePassing,
                ConcurrencyModel::ActorModel,
            ];
            assert_eq!(variants.len(), 4);
        }
    }

    // ========================================================================
    // EDGE CASES AND BOUNDARY CONDITIONS
    // ========================================================================

    mod edge_case_tests {
        use super::*;

        #[tokio::test]
        async fn test_parse_unicode_code() {
            // Arrange
            let parser = RustParser::new();
            let code = r#"
fn greet() {
    let message = "Hello, World!";
    let emoji = "Hello, World!";
    println!("{} {}", message, emoji);
}
"#;

            // Act
            let result = parser.parse(code).await;

            // Assert
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_parse_very_long_function_name() {
            // Arrange
            let parser = RustParser::new();
            let long_name = "a".repeat(1000);
            let code = format!("fn {}() {{}}", long_name);

            // Act
            let result = parser.parse(&code).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.functions.len(), 1);
            assert_eq!(ast.functions[0].name, long_name);
        }

        #[tokio::test]
        async fn test_parse_many_functions() {
            // Arrange
            let parser = RustParser::new();
            let mut code = String::new();
            for i in 0..100 {
                code.push_str(&format!("fn func_{}() {{}}\n", i));
            }

            // Act
            let result = parser.parse(&code).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.functions.len(), 100);
        }

        #[tokio::test]
        async fn test_parse_code_with_special_characters() {
            // Arrange
            let parser = RustParser::new();
            let code = r##"
fn special() {
    let s = "test\n\t\r\\";
    let raw = r#"raw string"#;
}
"##;

            // Act
            let result = parser.parse(code).await;

            // Assert
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_parse_nested_functions() {
            // Arrange
            let parser = RustParser::new();
            let code = r#"
fn outer() {
    fn inner() {
        fn innermost() {}
    }
}
"#;

            // Act
            let result = parser.parse(code).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            // All functions should be detected (flat extraction)
            assert_eq!(ast.functions.len(), 3);
        }

        #[tokio::test]
        async fn test_parse_function_like_in_string() {
            // Arrange
            let parser = RustParser::new();
            let code = r#"
fn real_function() {
    let s = "fn fake_function() {}";
}
"#;

            // Act
            let result = parser.parse(code).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            // Note: current simple regex parser will find both
            // A production parser would handle this correctly
            assert!(!ast.functions.is_empty());
        }

        #[tokio::test]
        async fn test_parse_single_line() {
            // Arrange
            let parser = RustParser::new();
            let code = "fn main() {}";

            // Act
            let result = parser.parse(code).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert_eq!(ast.functions.len(), 1);
            assert_eq!(ast.complexity_metrics.lines_of_code, 1);
        }

        #[test]
        fn test_halstead_empty_code() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let metrics = parser.calculate_halstead_complexity("");

            // Assert
            assert_eq!(metrics.vocabulary, 0);
            assert_eq!(metrics.length, 0);
            assert_eq!(metrics.volume, 0.0);
            assert_eq!(metrics.difficulty, 0.0);
        }

        #[test]
        fn test_nesting_depth_unbalanced_braces() {
            // Arrange
            let parser = RustParser::new();
            let code = "{ { { }"; // Unbalanced - 3 open, 1 close

            // Act
            let depth = parser.calculate_nesting_depth(code);

            // Assert - should not panic, uses saturating_sub
            assert!(depth >= 1);
        }

        #[test]
        fn test_cyclomatic_complexity_no_branches() {
            // Arrange
            let parser = RustParser::new();
            let code = "fn simple() { let x = 1; }";

            // Act
            let complexity = parser.calculate_cyclomatic_complexity(code);

            // Assert - base complexity is 1
            assert_eq!(complexity, 1.0);
        }

        #[tokio::test]
        async fn test_parse_only_comments() {
            // Arrange
            let parser = RustParser::new();
            let code = r#"
// Comment 1
// Comment 2
/* Multi-line */
"#;

            // Act
            let result = parser.parse(code).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert!(ast.functions.is_empty());
            assert!(!ast.comments.is_empty());
        }

        #[tokio::test]
        async fn test_parse_only_imports() {
            // Arrange
            let parser = RustParser::new();
            let code = r#"
use std::collections::HashMap;
use std::io::Read;
use crate::module::Type;
"#;

            // Act
            let result = parser.parse(code).await;

            // Assert
            assert!(result.is_ok());
            let ast = result.unwrap();
            assert!(ast.functions.is_empty());
            assert_eq!(ast.imports.len(), 3);
        }
    }

    // ========================================================================
    // LANGUAGE DETECTION TESTS
    // ========================================================================

    mod language_detection_tests {
        use super::*;

        #[test]
        fn test_rust_detection_with_fn_keyword() {
            let parser = RustParser::new();
            assert!(parser.detect_language("fn main() {}") > 0.0);
        }

        #[test]
        fn test_rust_detection_with_let_keyword() {
            let parser = RustParser::new();
            assert!(parser.detect_language("let x = 42;") > 0.0);
        }

        #[test]
        fn test_rust_detection_with_mut_keyword() {
            let parser = RustParser::new();
            assert!(parser.detect_language("let mut x = 42;") > 0.0);
        }

        #[test]
        fn test_rust_detection_with_impl_keyword() {
            let parser = RustParser::new();
            assert!(parser.detect_language("impl Trait for Type {}") > 0.0);
        }

        #[test]
        fn test_rust_detection_with_struct_keyword() {
            let parser = RustParser::new();
            assert!(parser.detect_language("struct Point { x: i32, y: i32 }") > 0.0);
        }

        #[test]
        fn test_rust_detection_with_enum_keyword() {
            let parser = RustParser::new();
            assert!(parser.detect_language("enum Color { Red, Green, Blue }") > 0.0);
        }

        #[test]
        fn test_rust_detection_with_trait_keyword() {
            let parser = RustParser::new();
            assert!(parser.detect_language("trait Iterator { fn next(&mut self); }") > 0.0);
        }

        #[test]
        fn test_rust_detection_with_path_separator() {
            let parser = RustParser::new();
            assert!(parser.detect_language("std::collections::HashMap") > 0.0);
        }

        #[test]
        fn test_rust_detection_score_capped_at_1() {
            let parser = RustParser::new();
            // Code with all Rust keywords
            let code = "fn main() { let mut x = impl struct enum trait }";
            assert!(parser.detect_language(code) <= 1.0);
        }

        #[test]
        fn test_java_detection_public_class() {
            let parser = JavaParser::new();
            assert!(parser.detect_language("public class Test {}") >= 0.3);
        }

        #[test]
        fn test_java_detection_main_method() {
            let parser = JavaParser::new();
            let code = "public static void main(String[] args) {}";
            assert!(parser.detect_language(code) >= 0.4);
        }

        #[test]
        fn test_java_detection_import() {
            let parser = JavaParser::new();
            assert!(parser.detect_language("import java.util.List;") >= 0.2);
        }

        #[test]
        fn test_java_detection_semicolons() {
            let parser = JavaParser::new();
            assert!(parser.detect_language("int x = 0;") >= 0.1);
        }

        #[test]
        fn test_empty_code_detection() {
            let rust_parser = RustParser::new();
            let java_parser = JavaParser::new();

            assert_eq!(rust_parser.detect_language(""), 0.0);
            assert_eq!(java_parser.detect_language(""), 0.0);
        }
    }

    // ========================================================================
    // SERIALIZATION TESTS
    // ========================================================================

    mod serialization_tests {
        use super::*;

        #[test]
        fn test_visibility_serialization() {
            // Assert serialized format uses snake_case
            let json = serde_json::to_string(&Visibility::Public).unwrap();
            assert_eq!(json, r#""public""#);
        }

        #[test]
        fn test_comment_type_serialization() {
            let json = serde_json::to_string(&CommentType::SingleLine).unwrap();
            assert_eq!(json, r#""single_line""#);
        }

        #[test]
        fn test_type_system_serialization() {
            let json = serde_json::to_string(&TypeSystem::Static).unwrap();
            assert_eq!(json, r#""static""#);
        }

        #[test]
        fn test_memory_management_serialization() {
            let json = serde_json::to_string(&MemoryManagement::GarbageCollected).unwrap();
            assert_eq!(json, r#""garbage_collected""#);
        }

        #[test]
        fn test_concurrency_model_serialization() {
            let json = serde_json::to_string(&ConcurrencyModel::AsyncAwait).unwrap();
            assert_eq!(json, r#""async_await""#);
        }

        #[test]
        fn test_halstead_metrics_serialization() {
            let metrics = HalsteadMetrics {
                operators: HashMap::from([("+".to_string(), 5)]),
                operands: HashMap::from([("x".to_string(), 3)]),
                vocabulary: 2,
                length: 8,
                volume: 24.0,
                difficulty: 1.5,
            };

            let json = serde_json::to_string(&metrics).unwrap();
            assert!(json.contains("operators"));
            assert!(json.contains("operands"));
        }

        #[test]
        fn test_language_features_roundtrip() {
            let features = LanguageFeatures {
                language: ProgrammingLanguage::Rust,
                supports_oop: false,
                supports_generics: true,
                supports_async: true,
                supports_patterns: true,
                type_system: TypeSystem::Static,
                memory_management: MemoryManagement::Manual,
                concurrency_model: ConcurrencyModel::AsyncAwait,
            };

            let json = serde_json::to_string(&features).unwrap();
            let parsed: LanguageFeatures = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.language, features.language);
            assert_eq!(parsed.supports_oop, features.supports_oop);
            assert_eq!(parsed.type_system, features.type_system);
        }
    }

    // ========================================================================
    // INTEGRATION TESTS
    // ========================================================================

    mod integration_tests {
        use super::*;

        #[tokio::test]
        async fn test_full_rust_parsing_workflow() {
            // Arrange
            let parser = RustParser::new();

            // Act - parse complex code
            let result = parser.parse(fixtures::RUST_COMPLEX).await.unwrap();

            // Assert - verify all components are extracted
            assert!(!result.functions.is_empty());
            assert!(!result.variables.is_empty());
            assert!(!result.imports.is_empty());
            assert!(!result.comments.is_empty());
            assert!(result.complexity_metrics.lines_of_code > 0);
            assert!(result.complexity_metrics.cyclomatic_complexity > 1.0);
        }

        #[tokio::test]
        async fn test_full_java_parsing_workflow() {
            // Arrange
            let parser = JavaParser::new();

            // Act
            let result = parser.parse(fixtures::JAVA_COMPLEX).await.unwrap();

            // Assert
            assert!(!result.classes.is_empty());
            assert!(!result.functions.is_empty());
            assert!(!result.imports.is_empty());
            assert!(!result.comments.is_empty());
        }

        #[tokio::test]
        async fn test_parser_produces_valid_ast_for_analysis() {
            // Arrange
            let parser = RustParser::new();

            // Act
            let result = parser.parse(fixtures::RUST_COMPLEX).await.unwrap();

            // Assert - AST should be suitable for further analysis
            for function in &result.functions {
                assert!(!function.name.is_empty());
                assert!(function.line_number > 0);
            }

            for import in &result.imports {
                assert!(!import.module.is_empty());
            }

            for comment in &result.comments {
                assert!(comment.line_number > 0);
            }
        }
    }
}
