//! # VIBE Protocol Validation System Demo
//!
//! Complete demonstration of the VIBE protocol validation system showcasing
//! the revolutionary "Agent-as-a-Verifier" paradigm in action.

use crate::thinktool::{Profile, ThinkToolExecutor};
use crate::vibe::benchmarking::{
    BenchmarkCategory, BenchmarkConfig, BenchmarkEngine, BenchmarkProtocol, BenchmarkResult,
    BenchmarkScenario, ExpectedOutcomes, PerformanceThresholds, ProtocolCharacteristics,
    ProtocolComplexity,
};
use crate::vibe::validation::VIBEError;
use crate::vibe::validation_config::{PerformanceSettings, ScoringWeights};
use crate::vibe::*;
use std::collections::HashMap;

/// Comprehensive demo showcasing VIBE validation capabilities
pub struct VIBEDemo {
    vibe_engine: VIBEEngine,
    benchmark_engine: BenchmarkEngine,
    thinktool_executor: ThinkToolExecutor,
}

impl VIBEDemo {
    /// Create new VIBE demo instance
    pub fn new() -> Result<Self, VIBEError> {
        Ok(Self {
            vibe_engine: VIBEEngine::new(),
            benchmark_engine: BenchmarkEngine::new(VIBEEngine::new()),
            thinktool_executor: ThinkToolExecutor::new(),
        })
    }

    /// Run comprehensive VIBE demonstration
    pub async fn run_complete_demo(&self) -> Result<VIBEDemoResult, VIBEError> {
        println!("🚀 Starting VIBE Protocol Validation System Demo");
        println!("{}", "=".repeat(60));

        // Demo 1: Basic Protocol Validation
        let basic_result = self.demo_basic_validation().await?;

        // Demo 2: Multi-Platform Validation
        let multi_platform_result = self.demo_multi_platform_validation().await?;

        // Demo 3: Custom Configuration
        let custom_result = self.demo_custom_configuration().await?;

        // Demo 4: Benchmarking Suite
        let benchmark_result = self.demo_benchmarking().await?;

        // Demo 5: ThinkTool Integration
        let integration_result = self.demo_thinktool_integration().await?;

        // Demo 6: Cross-Platform Validation
        let cross_platform_result = self.demo_cross_platform_validation().await?;

        println!("\n✅ VIBE Demo Complete!");
        println!("{}", "=".repeat(60));

        Ok(VIBEDemoResult {
            basic_validation: basic_result,
            multi_platform_validation: multi_platform_result,
            custom_configuration: custom_result,
            benchmarking: benchmark_result,
            thinktool_integration: integration_result,
            cross_platform_validation: cross_platform_result,
            total_execution_time_ms: 0, // Will be calculated
        })
    }

    /// Demo 1: Basic VIBE validation
    async fn demo_basic_validation(&self) -> Result<BasicValidationDemo, VIBEError> {
        println!("\n📋 Demo 1: Basic Protocol Validation");
        println!("{}", "-".repeat(40));

        let protocol = r#"
Protocol: E-commerce User Authentication
Purpose: Secure user authentication for e-commerce platform
Steps:
1. Validate user credentials (email/password)
2. Check account status and permissions
3. Generate secure session token
4. Log authentication event
5. Redirect to appropriate dashboard
Security Requirements:
- Password must be hashed using bcrypt
- Session tokens expire after 24 hours
- Failed attempts are logged and monitored
- Rate limiting prevents brute force attacks
Error Handling:
- Invalid credentials return generic error
- Account locked after 5 failed attempts
- System errors are logged with correlation IDs
"#;

        let config =
            ValidationConfig::default().with_platforms(vec![Platform::Web, Platform::Backend]);

        let start_time = std::time::Instant::now();
        let result = self.vibe_engine.validate_protocol(protocol, config).await?;
        let execution_time = start_time.elapsed().as_millis();

        println!("✅ Validation completed in {}ms", execution_time);
        println!("📊 Overall VIBE Score: {:.1}/100", result.overall_score);
        println!("🎯 Status: {:?}", result.status);
        println!("🔍 Issues found: {}", result.issues.len());
        println!("💡 Recommendations: {}", result.recommendations.len());

        // Show platform breakdown
        println!("\n📈 Platform Breakdown:");
        for (platform, score) in &result.platform_scores {
            println!("  {:?}: {:.1}/100", platform, score);
        }

        // Show key issues
        if !result.issues.is_empty() {
            println!("\n⚠️  Key Issues:");
            for issue in result.issues.iter().take(3) {
                println!("  - {:?}: {}", issue.severity, issue.description);
            }
        }

        Ok(BasicValidationDemo {
            protocol_content: protocol.to_string(),
            validation_result: result,
            execution_time_ms: execution_time as u64,
        })
    }

    /// Demo 2: Multi-platform validation with all platforms
    async fn demo_multi_platform_validation(&self) -> Result<MultiPlatformDemo, VIBEError> {
        println!("\n🌐 Demo 2: Multi-Platform Validation");
        println!("{}", "-".repeat(40));

        let protocol = r#"
Protocol: Cross-Platform Mobile Banking App
Purpose: Secure mobile banking application with multi-platform support
Platform Requirements:
Web:
- Responsive design for desktop and mobile browsers
- Progressive Web App (PWA) capabilities
- WebSocket real-time notifications
- Accessibility compliance (WCAG 2.1 AA)
Android:
- Material Design 3.0 components
- Biometric authentication support
- Offline transaction queuing
- Android Auto integration
iOS:
- iOS Human Interface Guidelines compliance
- Face ID / Touch ID integration
- Apple Pay integration
- Siri shortcuts support
Backend:
- RESTful API with OpenAPI 3.0 specification
- OAuth 2.0 + OpenID Connect authentication
- PostgreSQL with encryption at rest
- Redis caching for session management
Security:
- End-to-end encryption for all transactions
- PCI DSS compliance requirements
- Multi-factor authentication mandatory
- Real-time fraud detection
"#;

        let config = ValidationConfig::comprehensive().with_minimum_score(75.0);

        let start_time = std::time::Instant::now();
        let result = self.vibe_engine.validate_protocol(protocol, config).await?;
        let execution_time = start_time.elapsed().as_millis();

        println!(
            "✅ Multi-platform validation completed in {}ms",
            execution_time
        );
        println!("📊 Overall VIBE Score: {:.1}/100", result.overall_score);
        println!("🎯 Status: {:?}", result.status);

        // Show detailed platform analysis
        println!("\n🔍 Detailed Platform Analysis:");
        for (platform, platform_result) in &result.detailed_results {
            println!("\n  📱 {:?} Platform:", platform);
            println!("    Score: {:.1}/100", platform_result.score);
            println!("    Issues: {}", platform_result.issues.len());
            println!(
                "    Performance: {}ms avg response",
                platform_result.performance_metrics.average_response_time_ms
            );

            if !platform_result.recommendations.is_empty() {
                println!("    Key Recommendations:");
                for rec in platform_result.recommendations.iter().take(2) {
                    println!("    - {}", rec);
                }
            }
        }

        Ok(MultiPlatformDemo {
            protocol_content: protocol.to_string(),
            validation_result: result,
            execution_time_ms: execution_time as u64,
        })
    }

    /// Demo 3: Custom validation configuration
    async fn demo_custom_configuration(&self) -> Result<CustomConfigDemo, VIBEError> {
        println!("\n⚙️  Demo 3: Custom Validation Configuration");
        println!("{}", "-".repeat(40));

        let protocol = r#"
Protocol: Enterprise Content Management System
Purpose: Scalable CMS for enterprise content management
Performance Requirements:
- Page load time < 2 seconds
- Support for 10,000+ concurrent users
- 99.9% uptime SLA
- Database query optimization required
Security Requirements:
- Role-based access control (RBAC)
- Content encryption at rest and in transit
- Audit logging for all content changes
- GDPR compliance for EU users
Architecture:
- Microservices architecture with API gateway
- Containerized deployment with Kubernetes
- Event-driven communication between services
- Horizontal scaling capabilities
"#;

        // Create custom configuration with specific requirements
        let custom_criteria = ValidationCriteria {
            logical_consistency: true,
            practical_applicability: true,
            platform_compatibility: true,
            performance_requirements: true,
            security_considerations: true,
            user_experience: false, // Not critical for this demo
            custom_metrics: HashMap::from([
                ("scalability_score".to_string(), 85.0),
                ("enterprise_readiness".to_string(), 90.0),
            ]),
        };

        let custom_weights = ScoringWeights {
            logical_consistency: 0.20,
            practical_applicability: 0.20,
            platform_compatibility: 0.15,
            performance_requirements: 0.25, // Higher weight for performance
            security_considerations: 0.20,  // Higher weight for security
            user_experience: 0.00,
            code_quality: 0.00,
            custom_weights: HashMap::from([
                ("scalability_score".to_string(), 0.15),
                ("enterprise_readiness".to_string(), 0.10),
            ]),
        };

        let performance_settings = PerformanceSettings {
            global_timeout_ms: 45000,
            per_platform_timeout_ms: 12000,
            max_concurrent_validations: 3,
            enable_caching: true,
            cache_ttl_seconds: 7200,
            parallel_platform_validation: true,
            resource_limits: ResourceLimits {
                max_memory_mb: 2048,
                max_cpu_percent: 75.0,
                max_disk_usage_mb: 4096,
                max_network_requests: 200,
            },
            performance_monitoring: true,
        };

        let config = ValidationConfig::default()
            .with_platforms(vec![Platform::Web, Platform::Backend])
            .with_criteria(custom_criteria)
            .with_scoring_weights(custom_weights)
            .with_performance_settings(performance_settings)
            .with_minimum_score(80.0); // Higher threshold for enterprise

        let start_time = std::time::Instant::now();
        let result = self
            .vibe_engine
            .validate_protocol(protocol, config.clone())
            .await?;
        let execution_time = start_time.elapsed().as_millis();

        println!(
            "✅ Custom configuration validation completed in {}ms",
            execution_time
        );
        println!("📊 Overall VIBE Score: {:.1}/100", result.overall_score);
        println!("🎯 Status: {:?}", result.status);
        println!("🎯 Minimum Score Required: 80.0");

        // Show confidence interval
        if let Some(confidence) = &result.confidence_interval {
            println!(
                "📈 Confidence Interval: [{:.1}, {:.1}] (95% confidence)",
                confidence.lower, confidence.upper
            );
        }

        // Show custom scoring breakdown
        println!("\n🔍 Custom Scoring Analysis:");
        println!("  Performance Requirements: Weighted at 25%");
        println!("  Security Considerations: Weighted at 20%");
        println!("  Custom Metrics: Scalability + Enterprise Readiness");

        Ok(CustomConfigDemo {
            protocol_content: protocol.to_string(),
            validation_result: result,
            execution_time_ms: execution_time as u64,
            custom_config: config,
        })
    }

    /// Demo 4: Comprehensive benchmarking
    async fn demo_benchmarking(&self) -> Result<BenchmarkDemo, VIBEError> {
        println!("\n📊 Demo 4: Comprehensive Benchmarking");
        println!("{}", "-".repeat(40));

        // Create benchmark scenarios
        let scenarios = vec![
            self.create_simple_protocol_scenario(),
            self.create_complex_protocol_scenario(),
            self.create_performance_focused_scenario(),
        ];

        let suite = BenchmarkSuite {
            suite_id: Uuid::new_v4(),
            name: "VIBE Performance Benchmark Suite".to_string(),
            description: "Comprehensive performance testing for VIBE validation system".to_string(),
            scenarios,
            config: BenchmarkConfig {
                iterations: 3,
                parallel_execution: true,
                max_concurrent_validations: 2,
                warmup_iterations: 1,
                confidence_level: 0.95,
                enable_profiling: true,
                scoring_criteria: None,
            },
            results: Vec::new(),
        };

        let start_time = std::time::Instant::now();
        let result = self
            .benchmark_engine
            .execute_suite(&suite, &ValidationConfig::default())
            .await?;
        let execution_time = start_time.elapsed().as_millis();

        println!("✅ Benchmark suite completed in {}ms", execution_time);
        println!(
            "📊 Average Score: {:.1}/100",
            result.overall_metrics.average_score
        );
        println!(
            "📈 Validation Time: {}ms total",
            result.overall_metrics.total_validation_time_ms
        );
        println!(
            "✅ Success Rate: {}/{} scenarios passed",
            result.overall_metrics.platforms_passed,
            result.overall_metrics.platforms_passed + result.overall_metrics.platforms_failed
        );

        // Show performance statistics
        println!("\n📈 Performance Statistics:");
        println!(
            "  Mean Validation Time: {:.1}ms",
            result.statistics.mean_validation_time_ms
        );
        println!(
            "  Std Deviation: {:.1}ms",
            result.statistics.std_dev_validation_time_ms
        );
        println!(
            "  95th Percentile: {}ms",
            result.statistics.percentile_95_ms
        );
        println!(
            "  Throughput: {:.1} validations/second",
            result.statistics.throughput_validations_per_second
        );

        // Show bottleneck analysis
        // if let Some(bottleneck) = &result.execution_metrics.bottleneck_analysis.slowest_component {
        //     println!("\n🐌 Bottleneck Analysis:");
        //     println!("  Slowest Component: {} ({}ms)", bottleneck.0, bottleneck.1);
        // }

        Ok(BenchmarkDemo {
            benchmark_suite: suite,
            benchmark_result: result,
            execution_time_ms: execution_time as u64,
        })
    }

    /// Demo 5: ThinkTool integration
    async fn demo_thinktool_integration(&self) -> Result<ThinkToolIntegrationDemo, VIBEError> {
        println!("\n🧠 Demo 5: ThinkTool Integration");
        println!("{}", "-".repeat(40));

        let prompt = "Design a comprehensive user registration and onboarding protocol for a SaaS platform that includes email verification, social login options, progressive profiling, and compliance with GDPR and CCPA regulations.";

        println!("📝 Generating protocol from prompt...");
        println!("Prompt: {}", prompt);

        let start_time = std::time::Instant::now();

        // Generate protocol using ThinkTools
        let protocol_content = self
            .thinktool_executor
            .run(prompt, Profile::Balanced)
            .await?;

        println!("✅ Protocol generated successfully");
        println!("📄 Protocol length: {} characters", protocol_content.len());

        // Validate generated protocol with VIBE
        let vibe_config = ValidationConfig::comprehensive().with_platforms(vec![
            Platform::Web,
            Platform::Backend,
            Platform::Simulation,
        ]);

        let validation_result = self
            .vibe_engine
            .validate_protocol(&protocol_content, vibe_config)
            .await?;

        let generation_time = start_time.elapsed().as_millis();

        println!("✅ Protocol validation completed in {}ms", generation_time);
        println!("📊 VIBE Score: {:.1}/100", validation_result.overall_score);
        println!("🎯 Status: {:?}", validation_result.status);

        // Show integration workflow benefits
        println!("\n🔄 Integration Workflow Benefits:");
        println!("  ✅ Automated protocol generation");
        println!("  ✅ Immediate validation feedback");
        println!("  ✅ Quality assurance integration");
        println!("  ✅ Continuous improvement loop");

        Ok(ThinkToolIntegrationDemo {
            original_prompt: prompt.to_string(),
            generated_protocol: protocol_content,
            validation_result,
            total_time_ms: generation_time as u64,
        })
    }

    /// Demo 6: Cross-platform validation with adapters
    async fn demo_cross_platform_validation(&self) -> Result<CrossPlatformDemo, VIBEError> {
        println!("\n🌍 Demo 6: Cross-Platform Validation with Adapters");
        println!("{}", "-".repeat(40));

        let protocol = r#"
Protocol: Global Payment Processing System
Purpose: Multi-region payment processing with real-time fraud detection
Cross-Platform Requirements:
Web Platform:
- PCI DSS Level 1 compliance
- 3D Secure 2.0 integration
- Real-time payment status updates
- Multi-currency support (150+ currencies)
- Accessibility (WCAG 2.1 AAA)
Android Platform:
- Google Pay and Samsung Pay integration
- Biometric authentication
- Offline payment queuing
- Material Design compliance
iOS Platform:
- Apple Pay native integration
- Face ID/Touch ID support
- Apple Business Chat integration
- iOS Human Interface Guidelines
Backend Platform:
- Multi-region deployment (AWS, Azure, GCP)
- Event-driven architecture with Kafka
- Real-time fraud detection with ML
- Horizontal auto-scaling
- 99.99% uptime SLA
Security:
- End-to-end encryption (AES-256)
- HSM-based key management
- SOC 2 Type II compliance
- Regular penetration testing
"#;

        // Create cross-platform validator with multiple adapters
        let cross_platform_config = super::adapters::CrossPlatformConfig {
            default_timeout_ms: 30000,
            max_concurrent_adapters: 3,
            enable_failover: true,
            aggregation_strategy: super::adapters::AggregationStrategy::Consensus,
            health_check_interval_ms: 10000,
        };

        let _cross_validator = super::adapters::CrossPlatformValidator::with_config(
            VIBEEngine::new(),
            cross_platform_config,
        );

        let start_time = std::time::Instant::now();

        // Execute cross-platform validation
        let validation_config = ValidationConfig::comprehensive().with_platforms(vec![
            Platform::Web,
            Platform::Android,
            Platform::IOS,
            Platform::Backend,
            Platform::Simulation,
        ]);

        let result = self
            .vibe_engine
            .validate_protocol(protocol, validation_config)
            .await?;

        let execution_time = start_time.elapsed().as_millis();

        println!(
            "✅ Cross-platform validation completed in {}ms",
            execution_time
        );
        println!("📊 Overall VIBE Score: {:.1}/100", result.overall_score);
        println!("🎯 Status: {:?}", result.status);

        // Show platform consensus analysis
        let platform_scores: Vec<f32> = result.platform_scores.values().cloned().collect();
        let consensus_score = if platform_scores.len() > 1 {
            let mean = platform_scores.iter().sum::<f32>() / platform_scores.len() as f32;
            let variance = platform_scores
                .iter()
                .map(|&score| (score - mean).powi(2))
                .sum::<f32>()
                / platform_scores.len() as f32;
            let consensus = 1.0 - (variance / 100.0).min(1.0);
            consensus * 100.0
        } else {
            100.0
        };

        println!("\n🤝 Platform Consensus Analysis:");
        println!("  Consensus Score: {:.1}%", consensus_score);
        println!("  Platforms Validated: {}", result.platform_scores.len());

        for (platform, score) in &result.platform_scores {
            println!("  {:?}: {:.1}/100", platform, score);
        }

        Ok(CrossPlatformDemo {
            protocol_content: protocol.to_string(),
            validation_result: result,
            execution_time_ms: execution_time as u64,
            consensus_score,
        })
    }

    // Helper methods to create benchmark scenarios
    fn create_simple_protocol_scenario(&self) -> BenchmarkScenario {
        BenchmarkScenario {
            scenario_id: Uuid::new_v4(),
            name: "Simple Protocol Validation".to_string(),
            description: "Basic protocol with minimal complexity".to_string(),
            category: BenchmarkCategory::Performance,
            protocol: BenchmarkProtocol {
                content: "Protocol: Simple Test\nPurpose: Basic testing\nSteps: 1. Test"
                    .to_string(),
                protocol_type: ProtocolType::ThinkToolChain,
                complexity: ProtocolComplexity::Simple,
                characteristics: ProtocolCharacteristics {
                    has_multiple_platforms: false,
                    has_security_requirements: false,
                    has_performance_requirements: false,
                    has_accessibility_requirements: false,
                    has_integration_requirements: false,
                    estimated_validation_time_ms: 500,
                },
            },
            target_platforms: vec![Platform::Web],
            performance_thresholds: PerformanceThresholds {
                max_validation_time_ms: 2000,
                max_memory_usage_mb: 512,
                min_score_threshold: 60.0,
                max_error_rate_percent: 5.0,
            },
            expected_outcomes: ExpectedOutcomes {
                expected_score_range: (70.0, 90.0),
                expected_issues_count: (0, 3),
                expected_platform_scores: HashMap::from([(Platform::Web, 80.0)]),
                required_validations: vec![Platform::Web],
            },
        }
    }

    fn create_complex_protocol_scenario(&self) -> BenchmarkScenario {
        BenchmarkScenario {
            scenario_id: Uuid::new_v4(),
            name: "Complex Multi-Platform Protocol".to_string(),
            description: "Complex protocol requiring multiple platform validation".to_string(),
            category: BenchmarkCategory::Accuracy,
            protocol: BenchmarkProtocol {
                content: "Protocol: Complex Test\nPurpose: Comprehensive testing\nSteps: 1. Initialize 2. Process 3. Validate 4. Output".to_string(),
                protocol_type: ProtocolType::DecisionFramework,
                complexity: ProtocolComplexity::Complex,
                characteristics: ProtocolCharacteristics {
                    has_multiple_platforms: true,
                    has_security_requirements: true,
                    has_performance_requirements: true,
                    has_accessibility_requirements: true,
                    has_integration_requirements: true,
                    estimated_validation_time_ms: 3000,
                },
            },
            target_platforms: vec![Platform::Web, Platform::Backend, Platform::Simulation],
            performance_thresholds: PerformanceThresholds {
                max_validation_time_ms: 8000,
                max_memory_usage_mb: 1024,
                min_score_threshold: 70.0,
                max_error_rate_percent: 3.0,
            },
            expected_outcomes: ExpectedOutcomes {
                expected_score_range: (75.0, 95.0),
                expected_issues_count: (0, 5),
                expected_platform_scores: HashMap::from([
                    (Platform::Web, 85.0),
                    (Platform::Backend, 90.0),
                    (Platform::Simulation, 80.0),
                ]),
                required_validations: vec![Platform::Web, Platform::Backend],
            },
        }
    }

    fn create_performance_focused_scenario(&self) -> BenchmarkScenario {
        BenchmarkScenario {
            scenario_id: Uuid::new_v4(),
            name: "Performance-Critical Protocol".to_string(),
            description: "Protocol with strict performance requirements".to_string(),
            category: BenchmarkCategory::Performance,
            protocol: BenchmarkProtocol {
                content: "Protocol: High Performance\nPurpose: Fast processing\nSteps: 1. Cache 2. Process 3. Respond".to_string(),
                protocol_type: ProtocolType::ReasoningProcess,
                complexity: ProtocolComplexity::Moderate,
                characteristics: ProtocolCharacteristics {
                    has_multiple_platforms: true,
                    has_security_requirements: false,
                    has_performance_requirements: true,
                    has_accessibility_requirements: false,
                    has_integration_requirements: false,
                    estimated_validation_time_ms: 1500,
                },
            },
            target_platforms: vec![Platform::Web, Platform::Backend],
            performance_thresholds: PerformanceThresholds {
                max_validation_time_ms: 3000,
                max_memory_usage_mb: 768,
                min_score_threshold: 80.0,
                max_error_rate_percent: 2.0,
            },
            expected_outcomes: ExpectedOutcomes {
                expected_score_range: (80.0, 95.0),
                expected_issues_count: (0, 2),
                expected_platform_scores: HashMap::from([
                    (Platform::Web, 85.0),
                    (Platform::Backend, 90.0),
                ]),
                required_validations: vec![Platform::Web, Platform::Backend],
            },
        }
    }
}

/// Complete demo result
#[derive(Debug)]
pub struct VIBEDemoResult {
    pub basic_validation: BasicValidationDemo,
    pub multi_platform_validation: MultiPlatformDemo,
    pub custom_configuration: CustomConfigDemo,
    pub benchmarking: BenchmarkDemo,
    pub thinktool_integration: ThinkToolIntegrationDemo,
    pub cross_platform_validation: CrossPlatformDemo,
    pub total_execution_time_ms: u64,
}

/// Demo result types
#[derive(Debug)]
pub struct BasicValidationDemo {
    pub protocol_content: String,
    pub validation_result: ValidationResult,
    pub execution_time_ms: u64,
}

#[derive(Debug)]
pub struct MultiPlatformDemo {
    pub protocol_content: String,
    pub validation_result: ValidationResult,
    pub execution_time_ms: u64,
}

#[derive(Debug)]
pub struct CustomConfigDemo {
    pub protocol_content: String,
    pub validation_result: ValidationResult,
    pub execution_time_ms: u64,
    pub custom_config: ValidationConfig,
}

#[derive(Debug)]
pub struct BenchmarkDemo {
    pub benchmark_suite: BenchmarkSuite,
    pub benchmark_result: BenchmarkResult,
    pub execution_time_ms: u64,
}

#[derive(Debug)]
pub struct ThinkToolIntegrationDemo {
    pub original_prompt: String,
    pub generated_protocol: String,
    pub validation_result: ValidationResult,
    pub total_time_ms: u64,
}

#[derive(Debug)]
pub struct CrossPlatformDemo {
    pub protocol_content: String,
    pub validation_result: ValidationResult,
    pub execution_time_ms: u64,
    pub consensus_score: f32,
}

/// Run the complete VIBE demonstration
pub async fn run_vibe_demo() -> Result<VIBEDemoResult, VIBEError> {
    let demo = VIBEDemo::new()?;
    demo.run_complete_demo().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vibe_demo_creation() {
        let demo = VIBEDemo::new();
        assert!(demo.is_ok());
    }

    #[tokio::test]
    async fn test_basic_validation_demo() {
        let demo = VIBEDemo::new().unwrap();
        let result = demo.demo_basic_validation().await;
        assert!(result.is_ok());

        let demo_result = result.unwrap();
        assert!(demo_result.validation_result.overall_score >= 0.0);
        assert!(demo_result.validation_result.overall_score <= 100.0);
        assert!(demo_result.execution_time_ms > 0);
    }

    #[test]
    fn test_benchmark_scenario_creation() {
        let demo = VIBEDemo::new().unwrap();
        let scenario = demo.create_simple_protocol_scenario();
        assert_eq!(scenario.category, BenchmarkCategory::Performance);
        assert_eq!(scenario.protocol.complexity, ProtocolComplexity::Simple);
    }
}
