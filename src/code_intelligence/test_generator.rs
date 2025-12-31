//! # Test Generator Engine Module
//!
//! Wrapper engine for comprehensive test case generation across multiple programming languages.
//! Integrates with the main code intelligence system to provide test generation capabilities.

use crate::code_intelligence::*;
use crate::error::Error;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, instrument};

/// Test Generator Engine that wraps the comprehensive test generator
pub struct TestGeneratorEngine {
    /// Inner test generator
    #[allow(dead_code)]
    test_generator: Arc<TestGenerator>,

    /// Default test framework
    default_framework: Option<TestFramework>,

    /// Generation cache
    generation_cache: std::collections::HashMap<String, TestSuite>,
}

/// Test generator wrapper that implements the interface expected by CodeIntelligenceEngine
pub struct TestGenerator;

impl TestGeneratorEngine {
    /// Create new test generator engine
    pub fn new() -> Self {
        let test_generator = Arc::new(TestGenerator);

        Self {
            test_generator,
            default_framework: None,
            generation_cache: std::collections::HashMap::new(),
        }
    }

    /// Generate test suggestions (simple interface for CodeIntelligenceEngine)
    #[instrument(skip(self, ast))]
    pub async fn generate_tests(
        &self,
        ast: &UnifiedAST,
        language: ProgrammingLanguage,
        analysis: &CodeAnalysisResult,
    ) -> Result<Vec<TestSuggestion>, Error> {
        let mut generator = TestGenerator;
        generator.generate_tests(ast, language, analysis).await
    }

    /// Create new test generator engine with default framework
    pub fn new_with_framework(framework: TestFramework) -> Self {
        let mut engine = Self::new();
        engine.default_framework = Some(framework);
        engine
    }

    /// Generate comprehensive test suite
    #[instrument(skip(self, ast, _code))]
    pub async fn generate_comprehensive_tests(
        &mut self,
        ast: &UnifiedAST,
        language: ProgrammingLanguage,
        _code: &str,
        test_types: &[TestType],
    ) -> Result<TestSuite, Error> {
        info!("Generating comprehensive tests for {:?} code", language);

        // Check cache first
        let cache_key = format!(
            "{}_{}_{}",
            language as u8,
            ast.functions.len(),
            test_types.len()
        );
        if let Some(cached_suite) = self.generation_cache.get(&cache_key) {
            info!("Using cached test suite");
            return Ok(cached_suite.clone());
        }

        // Compile-first placeholder suite.
        let test_suite = TestSuite {
            language,
            framework: self.default_framework,
            test_cases: Vec::new(),
        };

        self.generation_cache.insert(cache_key, test_suite.clone());

        Ok(test_suite)
    }

    /// Generate tests with M2 enhancement
    pub async fn generate_m2_enhanced_tests(
        &mut self,
        _ast: &UnifiedAST,
        language: ProgrammingLanguage,
        _code: &str,
        analysis: &CodeAnalysisResult,
    ) -> Result<Vec<TestSuggestion>, Error> {
        info!("Generating M2-enhanced tests for {:?}", language);

        let mut suggestions = Vec::new();

        // Generate based on complexity analysis
        if analysis.complexity_score < 0.6 {
            suggestions.push(TestSuggestion {
                test_type: TestType::Unit,
                description: "M2 suggests comprehensive unit tests due to code complexity"
                    .to_string(),
                target_function: None,
                coverage_area: "High-complexity functions".to_string(),
                priority: SuggestionPriority::High,
            });
        }

        // Generate based on bug findings
        let critical_bugs = analysis
            .bug_findings
            .iter()
            .filter(|b| b.severity == BugSeverity::Critical)
            .count();

        if critical_bugs > 0 {
            suggestions.push(TestSuggestion {
                test_type: TestType::Security,
                description: format!(
                    "M2 suggests security tests due to {} critical bugs found",
                    critical_bugs
                ),
                target_function: None,
                coverage_area: "Critical bug validation".to_string(),
                priority: SuggestionPriority::Critical,
            });
        }

        // Generate based on performance characteristics
        if analysis.performance_metrics.cpu_intensity > 0.7 {
            suggestions.push(TestSuggestion {
                test_type: TestType::Performance,
                description: "M2 suggests performance tests due to high CPU intensity".to_string(),
                target_function: None,
                coverage_area: "Performance validation".to_string(),
                priority: SuggestionPriority::Medium,
            });
        }

        Ok(suggestions)
    }

    /// Clear generation cache
    pub fn clear_cache(&mut self) {
        self.generation_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        CacheStats {
            cached_suites: self.generation_cache.len(),
            cache_size_bytes: self
                .generation_cache
                .values()
                .map(|suite| suite.test_cases.len() * 1000)
                .sum::<usize>(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub cached_suites: usize,
    pub cache_size_bytes: usize,
}

/// Test suggestion for the main code intelligence interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuggestion {
    pub test_type: TestType,
    pub description: String,
    pub target_function: Option<String>,
    pub coverage_area: String,
    pub priority: SuggestionPriority,
}

impl Default for TestGeneratorEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TestGenerator {
    /// Generate test suggestions (simple interface for CodeIntelligenceEngine)
    pub async fn generate_tests(
        &mut self,
        ast: &UnifiedAST,
        _language: ProgrammingLanguage,
        analysis: &CodeAnalysisResult,
    ) -> Result<Vec<TestSuggestion>, Error> {
        let mut suggestions = Vec::new();

        // Generate unit test suggestions for each function
        for function in &ast.functions {
            suggestions.push(TestSuggestion {
                test_type: TestType::Unit,
                description: format!("Unit tests for function '{}'", function.name),
                target_function: Some(function.name.clone()),
                coverage_area: "Function behavior and edge cases".to_string(),
                priority: SuggestionPriority::High,
            });
        }

        // Generate integration test suggestions for classes
        if !ast.classes.is_empty() {
            suggestions.push(TestSuggestion {
                test_type: TestType::Integration,
                description: "Integration tests for class interactions".to_string(),
                target_function: None,
                coverage_area: "Class integration and dependency testing".to_string(),
                priority: SuggestionPriority::Medium,
            });
        }

        // Generate performance test suggestions for complex functions
        let complex_functions = ast
            .functions
            .iter()
            .filter(|f| f.complexity.cyclomatic_complexity > 5.0)
            .count();

        if complex_functions > 0 {
            suggestions.push(TestSuggestion {
                test_type: TestType::Performance,
                description: format!(
                    "Performance tests for {} complex functions",
                    complex_functions
                ),
                target_function: None,
                coverage_area: "Performance characteristics".to_string(),
                priority: SuggestionPriority::Medium,
            });
        }

        // Generate security test suggestions based on analysis
        if analysis
            .bug_findings
            .iter()
            .any(|b| b.category == BugCategory::Security)
        {
            suggestions.push(TestSuggestion {
                test_type: TestType::Security,
                description: "Security tests for vulnerability validation".to_string(),
                target_function: None,
                coverage_area: "Security vulnerabilities and input validation".to_string(),
                priority: SuggestionPriority::High,
            });
        }

        Ok(suggestions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_test_generator_engine_creation() {
        let engine = TestGeneratorEngine::new();
        assert!(engine.generation_cache.is_empty());
    }

    #[tokio::test]
    async fn test_test_suggestion_generation() {
        let mut generator = TestGenerator;
        let ast = UnifiedAST {
            language: ProgrammingLanguage::Rust,
            functions: vec![FunctionNode {
                name: "test_func".to_string(),
                parameters: vec![],
                return_type: None,
                body: vec![],
                visibility: Visibility::Public,
                is_async: false,
                line_number: 1,
                complexity: ComplexityMetrics {
                    cyclomatic_complexity: 6.0, // Complex function
                    cognitive_complexity: 7.0,
                    nesting_depth: 3,
                    lines_of_code: 25,
                    halstead_complexity: HalsteadMetrics {
                        operators: std::collections::HashMap::new(),
                        operands: std::collections::HashMap::new(),
                        vocabulary: 10,
                        length: 30,
                        volume: 99.0,
                        difficulty: 5.0,
                    },
                },
            }],
            classes: vec![],
            variables: vec![],
            imports: vec![],
            comments: vec![],
            complexity_metrics: ComplexityMetrics::default(),
        };

        let analysis = CodeAnalysisResult {
            language: ProgrammingLanguage::Rust,
            complexity_score: 0.7,
            quality_metrics: CodeQualityMetrics::default(),
            optimization_suggestions: vec![],
            bug_findings: vec![BugFinding {
                severity: BugSeverity::Critical,
                category: BugCategory::Security,
                description: "Security issue found".to_string(),
                location: CodeLocation {
                    file_path: "test.rs".to_string(),
                    line_number: Some(1),
                    column_number: None,
                    function_name: Some("test_func".to_string()),
                },
                confidence: 0.9,
                suggested_fix: Some("Fix security issue".to_string()),
            }],
            test_suggestions: vec![],
            performance_metrics: PerformanceMetrics {
                execution_time_estimate: 1.0,
                memory_usage_estimate: 1.0,
                cpu_intensity: 0.8, // High CPU intensity
                scalability_score: 0.6,
                bottlenecks: vec![],
            },
            cross_language_insights: vec![],
        };

        let suggestions = generator
            .generate_tests(&ast, ProgrammingLanguage::Rust, &analysis)
            .await;
        assert!(suggestions.is_ok());
        let suggestions = suggestions.unwrap();

        // Should have unit test, performance test, and security test suggestions
        assert!(suggestions.len() >= 3);
        assert!(suggestions.iter().any(|s| s.test_type == TestType::Unit));
        assert!(suggestions
            .iter()
            .any(|s| s.test_type == TestType::Performance));
        assert!(suggestions
            .iter()
            .any(|s| s.test_type == TestType::Security));
    }

    #[test]
    fn test_cache_statistics() {
        let engine = TestGeneratorEngine::new();
        let stats = engine.get_cache_stats();
        assert_eq!(stats.cached_suites, 0);
        assert_eq!(stats.cache_size_bytes, 0);
    }

    #[test]
    fn test_framework_configuration() {
        let engine = TestGeneratorEngine::new_with_framework(TestFramework::PyTest);
        assert_eq!(engine.default_framework, Some(TestFramework::PyTest));
    }
}
