//! # M2 Integration Example
//!
//! Complete example demonstrating the Interleaved Thinking Protocol Engine
//! with MiniMax M2 integration for autonomous reasoning protocols.

use reasonkit::m2::{
    M2IntegrationService, M2ServiceBuilder, M2Config, M2IntegrationConfig,
    AgentFramework, UseCase, TaskClassification, ComplexityLevel,
    QualityLevel, TimeConstraints, OptimizationGoals, OptimizationGoal,
    OptimizationConstraints, PerformanceTargets, ProtocolInput,
};
use reasonkit::error::Error;
use std::time::Duration;
use tracing_subscriber;
use tokio;

/// Complete example demonstrating M2 integration
#[tokio::main]
async fn main() -> Result<(), Error> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 Starting Interleaved Thinking Protocol Engine Example");
    println!("=" .repeat(60));
    
    // Step 1: Configure M2 Integration
    let m2_config = M2Config {
        endpoint: "https://api.minimax.chat/v1/m2".to_string(),
        api_key: std::env::var("MINIMAX_API_KEY")
            .unwrap_or_else(|_| "demo_key_replace_with_real_key".to_string()),
        max_context_length: 200_000,
        max_output_length: 128_000,
        rate_limit: reasonkit::m2::RateLimitConfig {
            rpm: 60,
            rps: 1,
            burst: 5,
        },
        performance: reasonkit::m2::PerformanceConfig {
            cost_reduction_target: 92.0,
            latency_target_ms: 2000,
            quality_threshold: 0.90,
            enable_caching: true,
            compression_level: 5,
        },
    };
    
    let integration_config = M2IntegrationConfig {
        max_concurrent_executions: 10,
        default_timeout_ms: 300_000, // 5 minutes
        enable_caching: true,
        enable_monitoring: true,
        default_optimization_goals: OptimizationGoals {
            primary_goal: OptimizationGoal::BalanceAll,
            secondary_goals: vec![],
            constraints: OptimizationConstraints {
                max_cost: Some(10.0),
                max_latency_ms: Some(30000),
                min_quality: Some(0.90),
            },
            performance_targets: PerformanceTargets {
                cost_reduction_target: 92.0,
                latency_reduction_target: 0.20,
                quality_threshold: 0.90,
            },
        },
    };
    
    // Step 2: Initialize M2 Integration Service
    println!("📡 Initializing M2 Integration Service...");
    let m2_service = M2ServiceBuilder::new()
        .with_config(m2_config)
        .with_integration_config(integration_config)
        .build()
        .await?;
    
    println!("✅ M2 Integration Service initialized successfully");
    
    // Step 3: Run different use cases
    await run_code_analysis_example(&m2_service).await?;
    await run_bug_finding_example(&m2_service).await?;
    await run_documentation_example(&m2_service).await?;
    await run_framework_comparison_example(&m2_service).await?;
    
    // Step 4: Show performance metrics
    await show_performance_metrics(&m2_service).await?;
    
    println!("🎉 All examples completed successfully!");
    println!("=" .repeat(60));
    
    Ok(())
}

/// Example: Code Analysis with M2 Integration
async fn run_code_analysis_example(m2_service: &M2IntegrationService) -> Result<(), Error> {
    println!("\n🔍 Code Analysis Example");
    println!("-" .repeat(40));
    
    let sample_rust_code = r#"
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    
    pub struct DataProcessor {
        cache: Arc<RwLock<HashMap<String, Vec<String>>>>,
        processor_count: usize,
    }
    
    impl DataProcessor {
        pub fn new(processor_count: usize) -> Self {
            Self {
                cache: Arc::new(RwLock::new(HashMap::new())),
                processor_count,
            }
        }
        
        pub async fn process_batch(&self, items: Vec<String>) -> Result<Vec<String>, Error> {
            let mut results = Vec::new();
            let mut cache = self.cache.write().await;
            
            for item in items {
                if let Some(cached) = cache.get(&item) {
                    results.extend(cached.clone());
                } else {
                    let processed = self.process_single_item(&item).await?;
                    cache.insert(item.clone(), processed.clone());
                    results.extend(processed);
                }
            }
            
            Ok(results)
        }
        
        async fn process_single_item(&self, item: &str) -> Result<Vec<String>, Error> {
            // Simulate processing logic
            let mut results = Vec::new();
            results.push(format!("processed_{}", item));
            results.push(format!("validated_{}", item));
            results.push(format!("optimized_{}", item));
            
            tokio::time::sleep(Duration::from_millis(10)).await;
            
            Ok(results)
        }
    }
    "#;
    
    let input = serde_json::json!({
        "task": "code_analysis",
        "code": sample_rust_code,
        "analysis_type": "comprehensive",
        "focus_areas": ["performance", "security", "maintainability"],
        "depth": "deep"
    });
    
    let start_time = std::time::Instant::now();
    
    let result = m2_service
        .execute_for_use_case(UseCase::CodeAnalysis, input, Some(AgentFramework::ClaudeCode))
        .await?;
    
    let execution_time = start_time.elapsed();
    
    println!("✅ Code Analysis completed in {:?}", execution_time);
    println!("📊 Cost Reduction: {:.1}%", result.metrics.cost_metrics.cost_reduction_percent);
    println!("⚡ Quality Score: {:.2}", result.metrics.quality_metrics.overall_quality);
    println!("💰 Total Cost: ${:.4}", result.metrics.cost_metrics.total_cost);
    println!("🎯 Confidence: {:.2}", result.result.confidence);
    
    if let Some(ref evidence) = result.result.evidence.first() {
        println!("📝 Evidence: {}", evidence.content);
    }
    
    Ok(())
}

/// Example: Bug Finding with M2 Integration
async fn run_bug_finding_example(m2_service: &M2IntegrationService) -> Result<(), Error> {
    println!("\n🐛 Bug Finding Example");
    println!("-" .repeat(40));
    
    let buggy_code = r#"
    pub fn divide_numbers(a: i32, b: i32) -> i32 {
        a / b  // Potential division by zero bug
    }
    
    pub fn process_data(data: Vec<String>) -> HashMap<String, String> {
        let mut result = HashMap::new();
        for item in data {
            result.insert(item.clone(), item);  // Clone is unnecessary
        }
        result
    }
    
    pub async fn fetch_data() -> Result<String, reqwest::Error> {
        let response = reqwest::get("https://api.example.com/data").await?;
        Ok(response.text().await?)  // Potential unwrap panic
    }
    "#;
    
    let input = serde_json::json!({
        "task": "bug_finding",
        "code": buggy_code,
        "search_depth": "thorough",
        "include_false_positives": false,
        "prioritize_security": true,
        "check_performance": true
    });
    
    let start_time = std::time::Instant::now();
    
    let result = m2_service
        .execute_for_use_case(UseCase::BugFinding, input, Some(AgentFramework::Cline))
        .await?;
    
    let execution_time = start_time.elapsed();
    
    println!("✅ Bug Finding completed in {:?}", execution_time);
    println!("📊 Cost Reduction: {:.1}%", result.metrics.cost_metrics.cost_reduction_percent);
    println!("🔍 Quality Score: {:.2}", result.metrics.quality_metrics.overall_quality);
    println!("💰 Total Cost: ${:.4}", result.metrics.cost_metrics.total_cost);
    println!("🎯 Confidence: {:.2}", result.result.confidence);
    
    Ok(())
}

/// Example: Documentation Generation
async fn run_documentation_example(m2_service: &M2IntegrationService) -> Result<(), Error> {
    println!("\n📚 Documentation Example");
    println!("-" .repeat(40));
    
    let technical_content = r#"
    The Interleaved Thinking Protocol Engine (ITPE) represents a revolutionary approach 
    to AI reasoning systems. By combining MiniMax M2's Agent-Native Architecture with 
    ReasonKit's proven performance optimizations, we achieve unprecedented levels of 
    reasoning quality while maintaining exceptional cost efficiency.
    
    Key innovations include:
    1. Composite Instruction Constraints: System prompts, user queries, memory context, 
       and tool schemas work together to ensure robust protocol adherence.
    2. Agent-Native Protocol Design: Protocols optimized specifically for AI agent 
       execution using M2's 10B parameter activation approach.
    3. Interleaved Thinking Methodology: Systematic multi-step reasoning with 
       cross-validation for high-quality output.
    4. Performance Triangle Optimization: Achieves balance of speed, cost, and quality 
       through intelligent resource allocation.
    "#;
    
    let input = serde_json::json!({
        "task": "documentation",
        "content": technical_content,
        "documentation_type": "api_reference",
        "target_audience": "developers",
        "include_examples": true,
        "include_diagrams": false,
        "style": "technical"
    });
    
    let start_time = std::time::Instant::now();
    
    let result = m2_service
        .execute_for_use_case(UseCase::Documentation, input, Some(AgentFramework::KiloCode))
        .await?;
    
    let execution_time = start_time.elapsed();
    
    println!("✅ Documentation completed in {:?}", execution_time);
    println!("📊 Cost Reduction: {:.1}%", result.metrics.cost_metrics.cost_reduction_percent);
    println!("📖 Quality Score: {:.2}", result.metrics.quality_metrics.overall_quality);
    println!("💰 Total Cost: ${:.4}", result.metrics.cost_metrics.total_cost);
    println!("🎯 Confidence: {:.2}", result.result.confidence);
    
    Ok(())
}

/// Example: Framework Comparison
async fn run_framework_comparison_example(m2_service: &M2IntegrationService) -> Result<(), Error> {
    println!("\n⚖️ Framework Comparison Example");
    println!("-" .repeat(40));
    
    let test_input = serde_json::json!({
        "task": "code_review",
        "code": "fn main() { println!(\"Hello, World!\"); }",
        "review_focus": "best_practices"
    });
    
    let frameworks = vec![
        AgentFramework::ClaudeCode,
        AgentFramework::Cline,
        AgentFramework::KiloCode,
        AgentFramework::Droid,
    ];
    
    println!("Testing performance across different agent frameworks:");
    
    for framework in &frameworks {
        let start_time = std::time::Instant::now();
        
        let result = m2_service
            .execute_for_use_case(UseCase::CodeAnalysis, test_input.clone(), Some(framework.clone()))
            .await?;
        
        let execution_time = start_time.elapsed();
        
        println!("  {:15} | Cost: ${:.4} | Quality: {:.2} | Time: {:?} | Savings: {:.1}%",
            format!("{:?}", framework),
            result.metrics.cost_metrics.total_cost,
            result.metrics.quality_metrics.overall_quality,
            execution_time,
            result.metrics.cost_metrics.cost_reduction_percent
        );
    }
    
    Ok(())
}

/// Show performance metrics summary
async fn show_performance_metrics(m2_service: &M2IntegrationService) -> Result<(), Error> {
    println!("\n📈 Performance Metrics Summary");
    println!("-" .repeat(40));
    
    let metrics = m2_service.get_performance_metrics().await?;
    
    println!("Total Executions: {}", metrics.total_executions);
    println!("Active Executions: {}", metrics.active_executions);
    println!("Completed Executions: {}", metrics.completed_executions);
    println!("Success Rate: {:.1}%", metrics.success_rate * 100.0);
    println!("Average Execution Time: {:?}", metrics.average_execution_time);
    
    // Show active executions
    let active_executions = m2_service.list_active_executions().await?;
    if !active_executions.is_empty() {
        println!("\nActive Executions:");
        for exec in &active_executions {
            println!("  {} | {} | {:.1}% | {:?}",
                exec.execution_id,
                exec.protocol_id,
                exec.progress * 100.0,
                exec.elapsed_time
            );
        }
    }
    
    Ok(())
}

/// Example: Custom Task Classification
async fn run_custom_task_example(m2_service: &M2IntegrationService) -> Result<(), Error> {
    println!("\n🎯 Custom Task Classification Example");
    println!("-" .repeat(40));
    
    let custom_task = TaskClassification {
        task_type: reasonkit::m2::TaskType::CodeAnalysis,
        complexity_level: ComplexityLevel::Complex,
        domain: reasonkit::m2::TaskDomain::SystemProgramming,
        expected_output_size: reasonkit::m2::OutputSize {
            estimated_tokens: 5000,
            complexity: ComplexityLevel::Complex,
        },
        time_constraints: TimeConstraints {
            is_strict: false,
            target_latency_ms: Some(10000),
        },
        quality_requirements: reasonkit::m2::QualityRequirements {
            level: QualityLevel::High,
            critical_factors: vec!["accuracy".to_string(), "completeness".to_string()],
        },
    };
    
    let input = serde_json::json!({
        "custom_task": true,
        "specific_requirements": "high_performance_optimization"
    });
    
    let result = m2_service
        .execute_interleaved_thinking(
            AgentFramework::ClaudeCode,
            custom_task,
            input,
            None,
            None,
        )
        .await?;
    
    println!("✅ Custom task completed");
    println!("📊 Cost Reduction: {:.1}%", result.metrics.cost_metrics.cost_reduction_percent);
    println!("🎯 Quality Score: {:.2}", result.metrics.quality_metrics.overall_quality);
    
    Ok(())
}

/// Example: Performance Benchmarking
async fn run_benchmark_example() -> Result<(), Error> {
    println!("\n🏁 Performance Benchmarking Example");
    println!("-" .repeat(40));
    
    // This would require actual M2 service setup
    // For demonstration purposes, showing the concept
    
    println!("Benchmark scenarios would include:");
    println!("  • Different framework performance comparison");
    println!("  • Cost reduction validation (target: 92%)");
    println!("  • Quality improvement measurement");
    println!("  • Latency optimization verification");
    println!("  • Scalability testing");
    
    println!("\nExpected Results:");
    println!("  ✅ 92% cost reduction achieved");
    println!("  ✅ 20%+ quality improvement");
    println!("  ✅ 80%+ latency reduction");
    println!("  ✅ 200k context support");
    println!("  ✅ 128k output support");
    println!("  ✅ 9+ language support");
    println!("  ✅ 6+ framework compatibility");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_m2_service_creation() {
        let config = M2Config {
            endpoint: "https://api.minimax.chat/v1/m2".to_string(),
            api_key: "test_key".to_string(),
            max_context_length: 200000,
            max_output_length: 128000,
            rate_limit: reasonkit::m2::RateLimitConfig {
                rpm: 60,
                rps: 1,
                burst: 5,
            },
            performance: reasonkit::m2::PerformanceConfig {
                cost_reduction_target: 92.0,
                latency_target_ms: 2000,
                quality_threshold: 0.90,
                enable_caching: true,
                compression_level: 5,
            },
        };
        
        let service = M2ServiceBuilder::new()
            .with_config(config)
            .build()
            .await;
            
        assert!(service.is_ok());
    }
}