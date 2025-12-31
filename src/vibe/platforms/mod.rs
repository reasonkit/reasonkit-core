//! # VIBE Platform Validators
//!
//! Platform-specific validators implementing the "Agent-as-a-Verifier" methodology
//! for comprehensive protocol validation across web, mobile, simulation, and backend environments.

use super::*;
use crate::vibe::validation::{
    IssueCategory, PlatformPerformanceMetrics, PlatformValidationResult, Severity, VIBEError,
    ValidationIssue, ValidationStatus,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
pub use std::collections::HashSet;

pub mod android_validator;
pub mod backend_validator;
pub mod ios_validator;
pub mod simulation_validator;
pub mod web_validator;

// Re-export all validators
pub use android_validator::AndroidValidator;
pub use backend_validator::BackendValidator;
pub use ios_validator::IOSValidator;
pub use simulation_validator::SimulationValidator;
pub use web_validator::WebValidator;

/// Platform enumeration for VIBE validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Platform {
    Web,
    Simulation,
    Android,
    IOS,
    Backend,
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Platform::Web => write!(f, "Web"),
            Platform::Simulation => write!(f, "Simulation"),
            Platform::Android => write!(f, "Android"),
            Platform::IOS => write!(f, "iOS"),
            Platform::Backend => write!(f, "Backend"),
        }
    }
}

/// Platform validator trait - implementing "Agent-as-a-Verifier" methodology
#[async_trait]
pub trait PlatformValidator: Send + Sync {
    /// Validate protocol for this specific platform
    async fn validate_protocol(
        &self,
        protocol_content: &str,
        config: &ValidationConfig,
        platform: Platform,
    ) -> Result<PlatformValidationResult, VIBEError>;

    /// Get platform-specific capabilities and features
    fn get_capabilities(&self) -> PlatformCapabilities;

    /// Get platform-specific requirements
    fn get_requirements(&self) -> PlatformRequirements;

    /// Estimate validation complexity and time
    fn estimate_complexity(&self, protocol_content: &str) -> ValidationComplexity;

    /// Get platform-specific scoring criteria
    fn get_scoring_criteria(&self) -> PlatformScoringCriteria;
}

/// Platform capabilities definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCapabilities {
    pub supported_protocol_types: Vec<ProtocolType>,
    pub supported_features: Vec<String>,
    pub integration_points: Vec<String>,
    pub performance_characteristics: PerformanceCharacteristics,
    pub limitations: Vec<String>,
}

/// Platform requirements definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformRequirements {
    pub minimum_requirements: Vec<String>,
    pub recommended_requirements: Vec<String>,
    pub compatibility_versions: HashMap<String, String>,
    pub security_requirements: Vec<String>,
    pub performance_requirements: Vec<String>,
}

/// Performance characteristics for platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    pub average_validation_time_ms: u64,
    pub memory_usage_profile: MemoryProfile,
    pub network_dependency: NetworkDependency,
    pub cpu_intensity: CpuIntensity,
}

/// Memory usage profile types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryProfile {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Network dependency levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkDependency {
    None,
    Optional,
    Required,
    Critical,
}

/// CPU intensity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuIntensity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Validation complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationComplexity {
    pub complexity_level: ComplexityLevel,
    pub estimated_time_ms: u64,
    pub resource_requirements: ResourceRequirements,
    pub risk_factors: Vec<String>,
}

/// Complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Resource requirements for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub disk_space_mb: u64,
    pub network_bandwidth_mbps: f32,
}

/// Platform-specific scoring criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformScoringCriteria {
    pub primary_criteria: Vec<String>,
    pub secondary_criteria: Vec<String>,
    pub penalty_factors: HashMap<String, f32>,
    pub bonus_factors: HashMap<String, f32>,
}

/// Validation environment for platform testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationEnvironment {
    pub environment_type: EnvironmentType,
    pub configuration: EnvironmentConfiguration,
    pub monitoring: EnvironmentMonitoring,
    pub cleanup: CleanupStrategy,
}

/// Environment types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentType {
    Emulated,
    Real,
    Hybrid,
    Cloud,
}

/// Environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfiguration {
    pub runtime_config: RuntimeConfiguration,
    pub resource_allocation: ResourceAllocation,
    pub network_config: NetworkConfiguration,
    pub security_config: SecurityConfiguration,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfiguration {
    pub platform_version: String,
    pub runtime_version: String,
    pub language_version: String,
    pub framework_version: String,
}

/// Resource allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_limit: Option<f32>,
    pub memory_limit: Option<u64>,
    pub disk_limit: Option<u64>,
    pub network_limit: Option<u64>,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfiguration {
    pub proxy_config: Option<ProxyConfiguration>,
    pub ssl_config: Option<SslConfiguration>,
    pub rate_limits: HashMap<String, u32>,
}

/// Proxy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfiguration {
    pub proxy_url: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

/// SSL/TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SslConfiguration {
    pub certificate_path: Option<String>,
    pub key_path: Option<String>,
    pub ca_bundle_path: Option<String>,
    pub verify_mode: SslVerifyMode,
}

/// SSL verification modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SslVerifyMode {
    None,
    Peer,
    FailIfNoPeerCert,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfiguration {
    pub sandbox_enabled: bool,
    pub permissions: Vec<String>,
    pub encryption_requirements: Vec<String>,
}

/// Environment monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentMonitoring {
    pub metrics_enabled: bool,
    pub log_level: LogLevel,
    pub alert_thresholds: HashMap<String, f32>,
}

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Cleanup strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupStrategy {
    pub cleanup_after_validation: bool,
    pub preserve_artifacts: bool,
    pub cleanup_timeout_ms: u64,
}

/// Base implementation for common platform validator functionality
pub struct BasePlatformValidator {
    pub platform: Platform,
    pub capabilities: PlatformCapabilities,
    pub requirements: PlatformRequirements,
    pub complexity_estimator: ComplexityEstimator,
}

impl BasePlatformValidator {
    pub fn new(platform: Platform) -> Self {
        Self {
            platform,
            capabilities: Self::get_default_capabilities(platform),
            requirements: Self::get_default_requirements(platform),
            complexity_estimator: ComplexityEstimator::new(),
        }
    }

    fn get_default_capabilities(platform: Platform) -> PlatformCapabilities {
        match platform {
            Platform::Web => PlatformCapabilities {
                supported_protocol_types: vec![
                    ProtocolType::ThinkToolChain,
                    ProtocolType::DecisionFramework,
                    ProtocolType::CustomProtocol,
                ],
                supported_features: vec![
                    "UI/UX Validation".to_string(),
                    "Responsive Design".to_string(),
                    "Accessibility Testing".to_string(),
                    "Performance Profiling".to_string(),
                    "Cross-browser Compatibility".to_string(),
                ],
                integration_points: vec![
                    "Browser APIs".to_string(),
                    "Web Standards".to_string(),
                    "DOM Manipulation".to_string(),
                    "CSS Validation".to_string(),
                ],
                performance_characteristics: PerformanceCharacteristics {
                    average_validation_time_ms: 2000,
                    memory_usage_profile: MemoryProfile::Medium,
                    network_dependency: NetworkDependency::Optional,
                    cpu_intensity: CpuIntensity::Medium,
                },
                limitations: vec![
                    "Limited to browser environment".to_string(),
                    "No access to native APIs".to_string(),
                ],
            },
            Platform::Simulation => PlatformCapabilities {
                supported_protocol_types: vec![
                    ProtocolType::ThinkToolChain,
                    ProtocolType::LogicFlow,
                    ProtocolType::ReasoningProcess,
                ],
                supported_features: vec![
                    "Logic Flow Validation".to_string(),
                    "State Management Testing".to_string(),
                    "Decision Tree Analysis".to_string(),
                    "Edge Case Simulation".to_string(),
                    "Performance Simulation".to_string(),
                ],
                integration_points: vec![
                    "Simulation Engine".to_string(),
                    "State Machine".to_string(),
                    "Event System".to_string(),
                ],
                performance_characteristics: PerformanceCharacteristics {
                    average_validation_time_ms: 1000,
                    memory_usage_profile: MemoryProfile::Low,
                    network_dependency: NetworkDependency::None,
                    cpu_intensity: CpuIntensity::Low,
                },
                limitations: vec![
                    "Simulation environment only".to_string(),
                    "No real-world interactions".to_string(),
                ],
            },
            Platform::Android => PlatformCapabilities {
                supported_protocol_types: vec![
                    ProtocolType::ThinkToolChain,
                    ProtocolType::CustomProtocol,
                ],
                supported_features: vec![
                    "Material Design Validation".to_string(),
                    "Touch Interaction Testing".to_string(),
                    "Screen Density Testing".to_string(),
                    "Android Version Compatibility".to_string(),
                    "Performance Testing".to_string(),
                ],
                integration_points: vec![
                    "Android SDK".to_string(),
                    "Material Design".to_string(),
                    "Android APIs".to_string(),
                ],
                performance_characteristics: PerformanceCharacteristics {
                    average_validation_time_ms: 3000,
                    memory_usage_profile: MemoryProfile::High,
                    network_dependency: NetworkDependency::Required,
                    cpu_intensity: CpuIntensity::High,
                },
                limitations: vec![
                    "Android platform only".to_string(),
                    "Requires Android SDK".to_string(),
                ],
            },
            Platform::IOS => PlatformCapabilities {
                supported_protocol_types: vec![
                    ProtocolType::ThinkToolChain,
                    ProtocolType::CustomProtocol,
                ],
                supported_features: vec![
                    "iOS Human Interface Guidelines".to_string(),
                    "Touch Gesture Testing".to_string(),
                    "iOS Version Compatibility".to_string(),
                    "Native iOS Patterns".to_string(),
                    "Performance Testing".to_string(),
                ],
                integration_points: vec![
                    "iOS SDK".to_string(),
                    "Apple Design Guidelines".to_string(),
                    "iOS APIs".to_string(),
                ],
                performance_characteristics: PerformanceCharacteristics {
                    average_validation_time_ms: 3000,
                    memory_usage_profile: MemoryProfile::High,
                    network_dependency: NetworkDependency::Required,
                    cpu_intensity: CpuIntensity::High,
                },
                limitations: vec![
                    "iOS platform only".to_string(),
                    "Requires iOS SDK".to_string(),
                ],
            },
            Platform::Backend => PlatformCapabilities {
                supported_protocol_types: vec![
                    ProtocolType::ThinkToolChain,
                    ProtocolType::DecisionFramework,
                    ProtocolType::ReasoningProcess,
                ],
                supported_features: vec![
                    "API Validation".to_string(),
                    "Data Flow Analysis".to_string(),
                    "Security Testing".to_string(),
                    "Performance Testing".to_string(),
                    "Scalability Testing".to_string(),
                ],
                integration_points: vec![
                    "REST APIs".to_string(),
                    "GraphQL".to_string(),
                    "Database Systems".to_string(),
                    "Message Queues".to_string(),
                ],
                performance_characteristics: PerformanceCharacteristics {
                    average_validation_time_ms: 2500,
                    memory_usage_profile: MemoryProfile::High,
                    network_dependency: NetworkDependency::Critical,
                    cpu_intensity: CpuIntensity::High,
                },
                limitations: vec![
                    "Backend environment only".to_string(),
                    "No UI components".to_string(),
                ],
            },
        }
    }

    fn get_default_requirements(platform: Platform) -> PlatformRequirements {
        match platform {
            Platform::Web => PlatformRequirements {
                minimum_requirements: vec![
                    "Valid HTML structure".to_string(),
                    "CSS styling compliance".to_string(),
                    "JavaScript functionality".to_string(),
                ],
                recommended_requirements: vec![
                    "Responsive design".to_string(),
                    "Accessibility compliance".to_string(),
                    "Cross-browser compatibility".to_string(),
                ],
                compatibility_versions: HashMap::from([
                    ("HTML".to_string(), "HTML5".to_string()),
                    ("CSS".to_string(), "CSS3".to_string()),
                    ("JavaScript".to_string(), "ES6+".to_string()),
                ]),
                security_requirements: vec![
                    "XSS prevention".to_string(),
                    "CSRF protection".to_string(),
                    "Input validation".to_string(),
                ],
                performance_requirements: vec![
                    "Page load < 3 seconds".to_string(),
                    "Interactive < 1 second".to_string(),
                ],
            },
            Platform::Simulation => PlatformRequirements {
                minimum_requirements: vec![
                    "Deterministic logic".to_string(),
                    "State consistency".to_string(),
                    "Error handling".to_string(),
                ],
                recommended_requirements: vec![
                    "Comprehensive test coverage".to_string(),
                    "Edge case handling".to_string(),
                ],
                compatibility_versions: HashMap::new(),
                security_requirements: vec![
                    "Input sanitization".to_string(),
                    "Logic validation".to_string(),
                ],
                performance_requirements: vec!["Response time < 100ms".to_string()],
            },
            Platform::Android => PlatformRequirements {
                minimum_requirements: vec![
                    "Material Design compliance".to_string(),
                    "Touch interaction support".to_string(),
                    "Screen adaptation".to_string(),
                ],
                recommended_requirements: vec![
                    "Performance optimization".to_string(),
                    "Battery efficiency".to_string(),
                ],
                compatibility_versions: HashMap::from([
                    ("Android SDK".to_string(), "API 21+".to_string()),
                    ("Material Design".to_string(), "2.0".to_string()),
                ]),
                security_requirements: vec![
                    "Secure storage".to_string(),
                    "Network security".to_string(),
                ],
                performance_requirements: vec![
                    "App launch < 2 seconds".to_string(),
                    "UI response < 100ms".to_string(),
                ],
            },
            Platform::IOS => PlatformRequirements {
                minimum_requirements: vec![
                    "iOS Human Interface Guidelines".to_string(),
                    "Native iOS patterns".to_string(),
                    "Touch gesture support".to_string(),
                ],
                recommended_requirements: vec![
                    "Performance optimization".to_string(),
                    "Battery efficiency".to_string(),
                ],
                compatibility_versions: HashMap::from([
                    ("iOS".to_string(), "12.0+".to_string()),
                    ("Swift".to_string(), "5.0+".to_string()),
                ]),
                security_requirements: vec![
                    "Keychain usage".to_string(),
                    "Secure networking".to_string(),
                ],
                performance_requirements: vec![
                    "App launch < 2 seconds".to_string(),
                    "UI response < 100ms".to_string(),
                ],
            },
            Platform::Backend => PlatformRequirements {
                minimum_requirements: vec![
                    "API documentation".to_string(),
                    "Error handling".to_string(),
                    "Input validation".to_string(),
                ],
                recommended_requirements: vec![
                    "Rate limiting".to_string(),
                    "Caching strategy".to_string(),
                    "Monitoring setup".to_string(),
                ],
                compatibility_versions: HashMap::from([
                    ("HTTP".to_string(), "1.1+".to_string()),
                    ("JSON".to_string(), "RFC 8259".to_string()),
                ]),
                security_requirements: vec![
                    "Authentication".to_string(),
                    "Authorization".to_string(),
                    "Data encryption".to_string(),
                ],
                performance_requirements: vec![
                    "API response < 200ms".to_string(),
                    "99.9% uptime".to_string(),
                ],
            },
        }
    }

    /// Common validation logic for all platforms
    async fn perform_common_validation(
        &self,
        protocol_content: &str,
        config: &ValidationConfig,
    ) -> Result<CommonValidationResult, VIBEError> {
        let mut issues = Vec::new();
        let mut score = 100.0;

        // Check protocol length and structure
        if protocol_content.trim().is_empty() {
            issues.push(ValidationIssue {
                platform: self.platform,
                severity: Severity::Critical,
                category: IssueCategory::LogicError,
                description: "Protocol content is empty".to_string(),
                location: None,
                suggestion: Some("Provide protocol content".to_string()),
            });
            score -= 30.0;
        }

        // Check for basic protocol structure
        if !self.has_basic_structure(protocol_content) {
            issues.push(ValidationIssue {
                platform: self.platform,
                severity: Severity::Medium,
                category: IssueCategory::LogicError,
                description: "Protocol missing basic structure".to_string(),
                location: None,
                suggestion: Some("Include protocol name, purpose, and steps".to_string()),
            });
            score -= 15.0;
        }

        // Check for required elements based on platform
        if !self.has_platform_specific_elements(protocol_content) {
            issues.push(ValidationIssue {
                platform: self.platform,
                severity: Severity::Medium,
                category: IssueCategory::CompatibilityProblem,
                description: "Platform-specific elements missing".to_string(),
                location: None,
                suggestion: Some("Add platform-specific validation rules".to_string()),
            });
            score -= 10.0;
        }

        // Check for logical consistency if required
        if config.validation_criteria.logical_consistency {
            let consistency_score = self.check_logical_consistency(protocol_content)?;
            score *= consistency_score / 100.0;
        }

        Ok(CommonValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
            performance_metrics: self.generate_performance_metrics(),
        })
    }

    fn has_basic_structure(&self, content: &str) -> bool {
        let lower = content.to_lowercase();
        lower.contains("protocol")
            && (lower.contains("purpose") || lower.contains("objective") || lower.contains("goal"))
            && (lower.contains("step") || lower.contains("process") || lower.contains("procedure"))
    }

    fn has_platform_specific_elements(&self, content: &str) -> bool {
        match self.platform {
            Platform::Web => {
                content.contains("UI")
                    || content.contains("user interface")
                    || content.contains("html")
                    || content.contains("css")
                    || content.contains("javascript")
            }
            Platform::Simulation => {
                content.contains("state")
                    || content.contains("logic")
                    || content.contains("simulation")
                    || content.contains("test")
            }
            Platform::Android => {
                content.contains("android")
                    || content.contains("material")
                    || content.contains("touch")
                    || content.contains("mobile")
            }
            Platform::IOS => {
                content.contains("ios")
                    || content.contains("swift")
                    || content.contains("apple")
                    || content.contains("touch")
            }
            Platform::Backend => {
                content.contains("api")
                    || content.contains("backend")
                    || content.contains("server")
                    || content.contains("database")
            }
        }
    }

    fn check_logical_consistency(&self, content: &str) -> Result<f32, VIBEError> {
        // Simple consistency check - look for contradictory patterns
        let contradictions = self.find_contradictions(content)?;
        let base_score = 100.0 - (contradictions.len() as f32 * 10.0);
        Ok(base_score.clamp(0.0, 100.0))
    }

    fn find_contradictions(&self, content: &str) -> Result<Vec<String>, VIBEError> {
        let mut contradictions = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        // Simple contradiction detection
        for line in &lines {
            let lower = line.to_lowercase();
            if (lower.contains("always") || lower.contains("never")) && lower.contains("but") {
                contradictions.push(format!("Potential contradiction found: {}", line.trim()));
            }
        }

        Ok(contradictions)
    }

    fn generate_performance_metrics(&self) -> PlatformPerformanceMetrics {
        PlatformPerformanceMetrics {
            average_response_time_ms: self
                .capabilities
                .performance_characteristics
                .average_validation_time_ms,
            memory_usage_mb: match self
                .capabilities
                .performance_characteristics
                .memory_usage_profile
            {
                MemoryProfile::Low => 50,
                MemoryProfile::Medium => 150,
                MemoryProfile::High => 500,
                MemoryProfile::VeryHigh => 1000,
            },
            cpu_usage_percent: match self.capabilities.performance_characteristics.cpu_intensity {
                CpuIntensity::Low => 10.0,
                CpuIntensity::Medium => 30.0,
                CpuIntensity::High => 60.0,
                CpuIntensity::VeryHigh => 90.0,
            },
            error_rate_percent: 2.0, // Base error rate
            throughput_requests_per_second: 100.0
                / (self
                    .capabilities
                    .performance_characteristics
                    .average_validation_time_ms as f32
                    / 1000.0),
        }
    }
}

/// Common validation result shared across platforms
#[derive(Debug, Clone)]
struct CommonValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
    #[allow(dead_code)]
    performance_metrics: PlatformPerformanceMetrics,
}

/// Complexity estimator for validation time prediction
pub struct ComplexityEstimator {
    // Base complexity factors for different protocol characteristics
    base_complexity: f32,
    size_factor: f32,
    #[allow(dead_code)]
    logic_factor: f32,
}

impl ComplexityEstimator {
    pub fn new() -> Self {
        Self {
            base_complexity: 1.0,
            size_factor: 0.001,
            logic_factor: 2.0,
        }
    }

    pub fn estimate_complexity(&self, protocol_content: &str) -> ValidationComplexity {
        let content_length = protocol_content.len() as f32;
        let line_count = protocol_content.lines().count() as f32;

        // Calculate complexity based on size and structure
        let size_complexity = content_length * self.size_factor;
        let structure_complexity = line_count * 0.1;

        let total_complexity = self.base_complexity + size_complexity + structure_complexity;

        let complexity_level = match total_complexity {
            x if x < 2.0 => ComplexityLevel::Simple,
            x if x < 5.0 => ComplexityLevel::Moderate,
            x if x < 10.0 => ComplexityLevel::Complex,
            _ => ComplexityLevel::VeryComplex,
        };

        let estimated_time_ms = (total_complexity * 1000.0) as u64;

        ValidationComplexity {
            complexity_level,
            estimated_time_ms,
            resource_requirements: ResourceRequirements {
                cpu_cores: 1,
                memory_mb: (total_complexity * 100.0) as u64,
                disk_space_mb: 10,
                network_bandwidth_mbps: 1.0,
            },
            risk_factors: if total_complexity > 5.0 {
                vec!["High complexity may affect validation accuracy".to_string()]
            } else {
                vec![]
            },
        }
    }
}

impl Default for ComplexityEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_enum_serialization() {
        let platform = Platform::Web;
        let json = serde_json::to_string(&platform).unwrap();
        assert_eq!(json, "\"web\"");

        let deserialized: Platform = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, Platform::Web);
    }

    #[test]
    fn test_complexity_estimator() {
        let estimator = ComplexityEstimator::new();
        let simple_protocol = "Protocol: Test\nPurpose: Testing\nSteps: 1. Test";

        let complexity = estimator.estimate_complexity(simple_protocol);
        assert_eq!(complexity.complexity_level, ComplexityLevel::Simple);

        let complex_protocol =
            "Protocol: Complex Test\nPurpose: Complex testing with many steps\n".repeat(100);
        let complex_complexity = estimator.estimate_complexity(&complex_protocol);
        assert!(matches!(
            complex_complexity.complexity_level,
            ComplexityLevel::Complex | ComplexityLevel::VeryComplex
        ));
    }

    #[test]
    fn test_base_validator_structure_check() {
        let validator = BasePlatformValidator::new(Platform::Web);

        let valid_content = "Protocol: Test\nPurpose: Testing\nSteps: 1. Validate";
        assert!(validator.has_basic_structure(valid_content));

        let invalid_content = "Just some text without proper structure";
        assert!(!validator.has_basic_structure(invalid_content));
    }
}
