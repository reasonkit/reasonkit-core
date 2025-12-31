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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rust_parser() {
        let parser = RustParser::new();
        let code = r#"
fn main() {
    let mut x = 42;
    if x > 0 {
        println!("Positive");
    }
}
"#;

        let ast = parser.parse(code).await;
        assert!(ast.is_ok());
        let ast = ast.unwrap();
        assert_eq!(ast.language, ProgrammingLanguage::Rust);
        assert!(!ast.functions.is_empty());
    }

    #[test]
    fn test_rust_language_detection() {
        let parser = RustParser::new();
        let rust_code = "fn main() { let x = 42; }";
        let java_code = "public class Test { }";

        let rust_score = parser.detect_language(rust_code);
        let java_score = parser.detect_language(java_code);

        assert!(rust_score > java_score);
    }

    #[tokio::test]
    async fn test_java_parser() {
        let parser = JavaParser::new();
        let code = r#"
public class Test {
    public void method() {
        int x = 42;
        if (x > 0) {
            System.out.println("Positive");
        }
    }
}
"#;

        let ast = parser.parse(code).await;
        assert!(ast.is_ok());
        let ast = ast.unwrap();
        assert_eq!(ast.language, ProgrammingLanguage::Java);
        assert!(!ast.classes.is_empty());
    }

    #[test]
    fn test_language_features() {
        let rust_parser = RustParser::new();
        let rust_features = rust_parser.get_language_features();

        assert_eq!(rust_features.language, ProgrammingLanguage::Rust);
        assert!(rust_features.supports_generics);
        assert!(rust_features.supports_async);
        assert_eq!(rust_features.type_system, TypeSystem::Static);
    }
}
