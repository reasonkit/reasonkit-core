//! # Multi-Language Code Intelligence Enhancement
//!
//! Leverages MiniMax M2's exceptional 9+ language mastery and superior 72.5% SWE-bench performance
//! to provide advanced code understanding, optimization, and analysis across multiple programming languages.
//!
//! ## Core Features
//!
//! - **Multi-Language Mastery**: Rust (primary), Java, Golang, C++, Kotlin, Objective-C, TypeScript, JavaScript, Python
//! - **SWE-bench Excellence**: 72.5% SWE-bench Multilingual score performance
//! - **Real-world Coding Tasks**: Test case generation, code optimization, code review
//! - **Cross-Framework Compatibility**: Claude Code, Cline, Kilo Code, Droid, Roo Code, BlackBox AI
//! - **Rust-First Enhancement**: Optimized for ReasonKit's Rust-based architecture
//! - **Interleaved Thinking Protocol**: Integration with M2's advanced reasoning capabilities

pub mod analyzer;
pub mod bug_detector;
pub mod m2_integration;
pub mod optimizer;
pub mod parser;
pub mod test_generator;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

pub use analyzer::CodeAnalyzer;
pub use bug_detector::{BugAnalysisContext, BugDetectorEngine, EnhancedBugFinding};
pub use m2_integration::{M2CodeInsight, M2CodeIntelligenceConnector, M2InsightType};
pub use optimizer::CodeOptimizer;
pub use parser::*;
pub use test_generator::{TestGeneratorEngine, TestSuggestion};

/// Programming languages supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgrammingLanguage {
    Rust,
    Java,
    Golang,
    Cpp,
    Kotlin,
    ObjectiveC,
    TypeScript,
    JavaScript,
    Python,
}

/// Main code intelligence engine that orchestrates all components
pub struct CodeIntelligenceEngine {
    parser: CodeParser,
    analyzer: CodeAnalyzer,
    optimizer: CodeOptimizer,
    bug_detector: BugDetectorEngine,
    test_generator: TestGeneratorEngine,
    m2_integration: M2CodeIntelligenceConnector,
}

impl CodeIntelligenceEngine {
    /// Create new code intelligence engine
    pub async fn new() -> Result<Self, crate::error::Error> {
        let parser = CodeParser::new();
        let analyzer = CodeAnalyzer::new();
        let optimizer = CodeOptimizer::new();
        let bug_detector = BugDetectorEngine::new();
        let test_generator = TestGeneratorEngine::new();
        let m2_integration = M2CodeIntelligenceConnector::new().await?;

        Ok(Self {
            parser,
            analyzer,
            optimizer,
            bug_detector,
            test_generator,
            m2_integration,
        })
    }

    /// Perform comprehensive code analysis
    pub async fn analyze_code(
        &mut self,
        code: &str,
        language: ProgrammingLanguage,
        context: Option<CodeAnalysisContext>,
    ) -> Result<ComprehensiveCodeAnalysis, crate::error::Error> {
        info!(
            "Starting comprehensive code analysis for {:?} ({} chars)",
            language,
            code.len()
        );

        // Parse code into AST
        let ast = self.parser.parse(code, language).await?;

        // Perform static analysis
        let analysis_result = self
            .analyzer
            .analyze(&ast, language, code, context.clone())?;

        // Detect bugs and issues
        let bug_findings = self
            .bug_detector
            .detect_bugs_comprehensive(&ast, language, code, &BugAnalysisContext::default())
            .await?;
        let bug_findings: Vec<BugFinding> = bug_findings.into_iter().map(|f| f.bug).collect();

        // Generate optimization suggestions
        let optimizations = self
            .optimizer
            .generate_optimizations(&ast, language, code, &analysis_result)
            .await?;

        // Generate test cases
        let test_suggestions = self
            .test_generator
            .generate_tests(&ast, language, &analysis_result)
            .await?;

        // Integrate with M2 for advanced reasoning
        let m2_insights = self
            .m2_integration
            .enhance_analysis(&analysis_result, &ast, language)
            .await?;

        // Calculate overall score
        let overall_score = self.calculate_overall_score(&analysis_result, &bug_findings);

        Ok(ComprehensiveCodeAnalysis {
            language,
            ast,
            analysis_result,
            bug_findings: bug_findings.clone(),
            optimization_suggestions: optimizations,
            test_suggestions,
            m2_insights,
            overall_score,
        })
    }

    /// Calculate overall code quality score
    fn calculate_overall_score(&self, analysis: &CodeAnalysisResult, bugs: &[BugFinding]) -> f64 {
        let quality_score = analysis.complexity_score;
        let bug_penalty = bugs
            .iter()
            .map(|b| b.severity.as_penalty_score())
            .sum::<f64>();
        (quality_score - bug_penalty).clamp(0.0, 1.0)
    }
}

/// Comprehensive code analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveCodeAnalysis {
    pub language: ProgrammingLanguage,
    pub ast: UnifiedAST,
    pub analysis_result: CodeAnalysisResult,
    pub bug_findings: Vec<BugFinding>,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    pub test_suggestions: Vec<TestSuggestion>,
    pub m2_insights: Vec<M2CodeInsight>,
    pub overall_score: f64,
}

/// Code analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAnalysisResult {
    pub language: ProgrammingLanguage,
    pub complexity_score: f64,
    pub quality_metrics: CodeQualityMetrics,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    pub bug_findings: Vec<BugFinding>,
    pub test_suggestions: Vec<TestSuggestion>,
    pub performance_metrics: PerformanceMetrics,
    pub cross_language_insights: Vec<CrossLanguageInsight>,
}

/// Bug finding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BugFinding {
    pub severity: BugSeverity,
    pub category: BugCategory,
    pub description: String,
    pub location: CodeLocation,
    pub confidence: f64,
    pub suggested_fix: Option<String>,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub category: OptimizationCategory,
    pub priority: SuggestionPriority,
    pub description: String,
    pub impact: f64,
    pub effort: f64,
    pub code_example: Option<String>,
}

/// Test suite produced by generators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    pub language: ProgrammingLanguage,
    pub framework: Option<TestFramework>,
    pub test_cases: Vec<TestCase>,
}

/// Code analysis context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAnalysisContext {
    pub file_path: Option<String>,
    pub project_type: Option<String>,
    pub performance_critical: bool,
    pub security_critical: bool,
    pub target_language: Option<ProgrammingLanguage>,
}

/// Extension trait for bug severity
trait BugSeverityExt {
    fn as_penalty_score(&self) -> f64;
}

impl BugSeverityExt for BugSeverity {
    fn as_penalty_score(&self) -> f64 {
        match self {
            BugSeverity::Critical => 0.3,
            BugSeverity::High => 0.2,
            BugSeverity::Medium => 0.1,
            BugSeverity::Low => 0.05,
            BugSeverity::Info => 0.0,
        }
    }
}

/// Main code parser
pub struct CodeParser {
    language_parsers: HashMap<ProgrammingLanguage, Arc<dyn LanguageParser + Send + Sync>>,
}

impl CodeParser {
    pub fn new() -> Self {
        let mut language_parsers: HashMap<
            ProgrammingLanguage,
            Arc<dyn LanguageParser + Send + Sync>,
        > = HashMap::new();

        // Initialize language-specific parsers
        language_parsers.insert(ProgrammingLanguage::Rust, Arc::new(RustParser::new()));
        language_parsers.insert(ProgrammingLanguage::Java, Arc::new(JavaParser::new()));
        language_parsers.insert(ProgrammingLanguage::Golang, Arc::new(GolangParser::new()));
        language_parsers.insert(ProgrammingLanguage::Cpp, Arc::new(CppParser::new()));
        language_parsers.insert(ProgrammingLanguage::Kotlin, Arc::new(KotlinParser::new()));
        language_parsers.insert(
            ProgrammingLanguage::ObjectiveC,
            Arc::new(ObjectiveCParser::new()),
        );
        language_parsers.insert(
            ProgrammingLanguage::TypeScript,
            Arc::new(TypeScriptParser::new()),
        );
        language_parsers.insert(
            ProgrammingLanguage::JavaScript,
            Arc::new(JavaScriptParser::new()),
        );
        language_parsers.insert(ProgrammingLanguage::Python, Arc::new(PythonParser::new()));

        Self { language_parsers }
    }

    pub async fn parse(
        &self,
        code: &str,
        language: ProgrammingLanguage,
    ) -> Result<UnifiedAST, crate::error::Error> {
        if let Some(parser) = self.language_parsers.get(&language) {
            parser.parse(code).await
        } else {
            Err(crate::error::Error::CodeIntelligence(format!(
                "Language {:?} is not supported",
                language
            )))
        }
    }

    /// Auto-detect programming language from code content
    pub async fn detect_language(
        &self,
        code: &str,
    ) -> Result<ProgrammingLanguage, crate::error::Error> {
        let mut best_language = ProgrammingLanguage::Rust;
        let mut best_score = 0.0;

        for (language, parser) in &self.language_parsers {
            let score = parser.detect_language(code);
            if score > best_score {
                best_score = score;
                best_language = *language;
            }
        }

        if best_score >= 0.3 {
            Ok(best_language)
        } else {
            Err(crate::error::Error::CodeIntelligence(
                "Failed to detect programming language".to_string(),
            ))
        }
    }
}

impl Default for CodeParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Error types for code intelligence
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Language {0:?} is not supported")]
    LanguageNotSupported(ProgrammingLanguage),
    #[error("Failed to detect programming language")]
    LanguageDetectionFailed,
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    #[error("M2 integration error: {0}")]
    M2IntegrationError(String),
}

/// Code quality metrics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeQualityMetrics {
    pub maintainability_index: f64,
    pub cyclomatic_complexity: f64,
    pub code_coverage_potential: f64,
    pub documentation_quality: f64,
    pub testability_score: f64,
    pub security_score: f64,
    pub performance_score: f64,
}

/// Bug severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BugSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Bug categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BugCategory {
    Security,
    Logic,
    Type,
    Memory,
    Performance,
    Accessibility,
    Concurrency,
    NullPointer,
    ResourceLeak,
    Deadlock,
    RaceCondition,
}

/// Code location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocation {
    pub file_path: String,
    pub line_number: Option<u32>,
    pub column_number: Option<u32>,
    pub function_name: Option<String>,
}

/// Optimization categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationCategory {
    Performance,
    Memory,
    Security,
    Maintainability,
    Testing,
}

/// Suggestion priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestionPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Test types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestType {
    Unit,
    Integration,
    EdgeCase,
    Performance,
    Security,
    Property,
}

/// Test frameworks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestFramework {
    JUnit,
    TestNG,
    PyTest,
    Jest,
    Mocha,
    NUnit,
    GoTest,
    RustTest,
    SwiftXCTest,
    KotlinTest,
}

/// Additional cross-language types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageInsight {
    pub source_language: ProgrammingLanguage,
    pub target_language: ProgrammingLanguage,
    pub insight_type: CrossLanguageInsightType,
    pub description: String,
    pub potential_improvements: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossLanguageInsightType {
    BestPractice,
    Pattern,
    Performance,
    Security,
}

/// Performance metrics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub execution_time_estimate: f64,
    pub memory_usage_estimate: f64,
    pub cpu_intensity: f64,
    pub scalability_score: f64,
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub location: CodeLocation,
    pub impact: f64,
    pub description: String,
    pub potential_optimization: String,
}

/// Cross-language analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageAnalysisResult {
    pub language_analyses: HashMap<ProgrammingLanguage, CodeAnalysisResult>,
    pub common_patterns: Vec<CrossLanguagePattern>,
    pub language_comparisons: Vec<LanguageComparison>,
    pub best_practices: Vec<CrossLanguageBestPractice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguagePattern {
    pub pattern_name: String,
    pub languages: Vec<ProgrammingLanguage>,
    pub description: String,
    pub implementations: HashMap<ProgrammingLanguage, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageComparison {
    pub language1: ProgrammingLanguage,
    pub language2: ProgrammingLanguage,
    pub similarities: Vec<String>,
    pub differences: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLanguageBestPractice {
    pub practice_name: String,
    pub applicable_languages: Vec<ProgrammingLanguage>,
    pub description: String,
    pub examples: HashMap<ProgrammingLanguage, String>,
    pub benefits: Vec<String>,
}

/// Test case structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub test_type: TestType,
    pub description: String,
    pub test_code: String,
    pub target_function: Option<String>,
    pub dependencies: Vec<String>,
    pub expected_outcomes: Vec<String>,
}

/// Code optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeOptimizationResult {
    pub original_code: String,
    pub optimized_code: String,
    pub improvements: Vec<OptimizationImprovement>,
    pub performance_gain_estimate: f64,
    pub maintainability_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationImprovement {
    pub category: OptimizationCategory,
    pub description: String,
    pub impact_score: f64,
    pub code_changes: Vec<String>,
}

/// Code change structure for optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChange {
    pub location: CodeLocation,
    pub original_code: String,
    pub new_code: String,
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_code_intelligence_engine_creation() {
        let engine = CodeIntelligenceEngine::new().await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_rust_code_analysis() {
        let mut engine = CodeIntelligenceEngine::new().await.unwrap();
        let rust_code = r#"
        fn main() {
            let mut x = 42;
            if x > 0 {
                println!("Positive: {}", x);
            }
        }
        "#;

        let result = engine
            .analyze_code(rust_code, ProgrammingLanguage::Rust, None)
            .await;
        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.language, ProgrammingLanguage::Rust);
        assert!(analysis.overall_score >= 0.0 && analysis.overall_score <= 1.0);
    }

    #[tokio::test]
    async fn test_language_detection() {
        let parser = CodeParser::new();

        let rust_code = "fn main() { let x = 42; }";
        let detected = parser.detect_language(rust_code).await;
        assert!(detected.is_ok());
        assert_eq!(detected.unwrap(), ProgrammingLanguage::Rust);
    }
}
