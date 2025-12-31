//! # VIBE Validation Configuration
//!
//! Configuration system for VIBE protocol validation across multiple platforms
//! with comprehensive customization options.

use super::*;
use crate::vibe::validation::{Severity, VIBEError};

/// Comprehensive validation configuration for VIBE engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Target platforms for validation
    pub target_platforms: Vec<Platform>,

    /// Minimum acceptable score for validation (0-100)
    pub minimum_score: f32,

    /// Validation criteria to evaluate
    pub validation_criteria: ValidationCriteria,

    /// Environment configuration
    pub environment_config: EnvironmentConfig,

    /// Scoring weights for different criteria
    pub scoring_weights: ScoringWeights,

    /// Platform-specific configurations
    pub platform_configs: HashMap<Platform, PlatformConfig>,

    /// Enable/disable specific validation features
    pub validation_features: ValidationFeatures,

    /// Timeout and performance settings
    pub performance_settings: PerformanceSettings,
}

/// Platform-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    /// Enable this platform validation
    pub enabled: bool,

    /// Custom timeout for this platform (overrides global)
    pub timeout_ms: Option<u64>,

    /// Custom validation depth
    pub validation_depth: ValidationDepth,

    /// Platform-specific criteria weights
    pub criteria_weights: Option<PlatformCriteriaWeights>,

    /// Custom validation rules
    pub custom_rules: Vec<CustomValidationRule>,
}

/// Validation depth levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationDepth {
    /// Basic validation (fast, surface-level checks)
    Basic,
    /// Standard validation (balanced depth and speed)
    Standard,
    /// Comprehensive validation (deep analysis, slower)
    Comprehensive,
    /// Exhaustive validation (maximum thoroughness, slowest)
    Exhaustive,
}

/// Custom validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomValidationRule {
    pub rule_id: String,
    pub rule_type: ValidationRuleType,
    pub condition: ValidationCondition,
    pub action: ValidationAction,
    pub severity: Severity,
    pub description: String,
}

/// Types of validation rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationRuleType {
    /// Check for specific text patterns
    TextPattern,
    /// Check for logical constraints
    LogicConstraint,
    /// Check for performance thresholds
    PerformanceThreshold,
    /// Check for security requirements
    SecurityRequirement,
    /// Check for platform-specific standards
    PlatformStandard,
}

/// Validation conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCondition {
    pub operator: ConditionOperator,
    pub target: ConditionTarget,
    pub value: ConditionValue,
    pub logical_operator: Option<LogicalOperator>,
    pub sub_conditions: Option<Vec<ValidationCondition>>,
}

/// Condition operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterEqual,
    LessEqual,
    Contains,
    NotContains,
    Matches,
    NotMatches,
}

/// Condition targets
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionTarget {
    TextContent,
    TextLength,
    LineCount,
    FunctionCount,
    ClassCount,
    CommentRatio,
    ComplexityScore,
    ErrorCount,
    WarningCount,
    PerformanceScore,
}

/// Condition values
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionValue {
    String(String),
    Integer(i32),
    Float(f32),
    Boolean(bool),
    Regex(String),
    Range { min: f32, max: f32 },
}

/// Logical operators for combining conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalOperator {
    And,
    Or,
    Not,
}

/// Validation actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationAction {
    pub action_type: ActionType,
    pub message: String,
    pub score_adjustment: Option<f32>,
    pub severity_adjustment: Option<Severity>,
}

/// Action types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    /// Add validation issue
    AddIssue,
    /// Adjust score
    AdjustScore,
    /// Skip validation
    SkipValidation,
    /// Mark as warning
    MarkWarning,
    /// Log custom message
    LogMessage,
}

/// Scoring weights for different validation criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Weight for logical consistency (0.0-1.0)
    pub logical_consistency: f32,

    /// Weight for practical applicability (0.0-1.0)
    pub practical_applicability: f32,

    /// Weight for platform compatibility (0.0-1.0)
    pub platform_compatibility: f32,

    /// Weight for performance requirements (0.0-1.0)
    pub performance_requirements: f32,

    /// Weight for security considerations (0.0-1.0)
    pub security_considerations: f32,

    /// Weight for user experience (0.0-1.0)
    pub user_experience: f32,

    /// Weight for code quality (0.0-1.0)
    pub code_quality: f32,

    /// Custom metric weights
    pub custom_weights: HashMap<String, f32>,
}

/// Platform-specific criteria weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCriteriaWeights {
    pub logical_consistency: Option<f32>,
    pub practical_applicability: Option<f32>,
    pub platform_compatibility: Option<f32>,
    pub performance_requirements: Option<f32>,
    pub security_considerations: Option<f32>,
    pub user_experience: Option<f32>,
    pub code_quality: Option<f32>,
}

/// Validation features configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFeatures {
    /// Enable detailed error reporting
    pub detailed_errors: bool,

    /// Enable performance profiling
    pub performance_profiling: bool,

    /// Enable security scanning
    pub security_scanning: bool,

    /// Enable accessibility testing
    pub accessibility_testing: bool,

    /// Enable cross-platform consistency checks
    pub cross_platform_consistency: bool,

    /// Enable automated fix suggestions
    pub automated_fixes: bool,

    /// Enable real-time validation
    pub real_time_validation: bool,

    /// Enable integration with external tools
    pub external_tool_integration: bool,
}

/// Performance and timeout settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Global timeout for validation (milliseconds)
    pub global_timeout_ms: u64,

    /// Timeout per platform (milliseconds)
    pub per_platform_timeout_ms: u64,

    /// Maximum concurrent validations
    pub max_concurrent_validations: usize,

    /// Enable caching of validation results
    pub enable_caching: bool,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,

    /// Enable parallel platform validation
    pub parallel_platform_validation: bool,

    /// Resource usage limits
    pub resource_limits: ResourceLimits,

    /// Enable performance monitoring
    pub performance_monitoring: bool,
}

/// Default validation configuration
impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            target_platforms: vec![Platform::Web, Platform::Simulation, Platform::Backend],
            minimum_score: 70.0,
            validation_criteria: ValidationCriteria {
                logical_consistency: true,
                practical_applicability: true,
                platform_compatibility: true,
                performance_requirements: true,
                security_considerations: false,
                user_experience: true,
                custom_metrics: HashMap::new(),
            },
            environment_config: EnvironmentConfig {
                timeout_ms: 30000,
                resource_limits: ResourceLimits {
                    max_memory_mb: 512,
                    max_cpu_percent: 80.0,
                    max_disk_usage_mb: 1024,
                    max_network_requests: 100,
                },
                network_conditions: NetworkConditions {
                    latency_ms: 50,
                    bandwidth_mbps: 10.0,
                    packet_loss_percent: 0.0,
                    connection_type: ConnectionType::Wifi,
                },
                browser_settings: None,
                mobile_settings: None,
            },
            scoring_weights: ScoringWeights {
                logical_consistency: 0.25,
                practical_applicability: 0.20,
                platform_compatibility: 0.15,
                performance_requirements: 0.15,
                security_considerations: 0.10,
                user_experience: 0.10,
                code_quality: 0.05,
                custom_weights: HashMap::new(),
            },
            platform_configs: HashMap::new(),
            validation_features: ValidationFeatures {
                detailed_errors: true,
                performance_profiling: true,
                security_scanning: false,
                accessibility_testing: false,
                cross_platform_consistency: true,
                automated_fixes: false,
                real_time_validation: false,
                external_tool_integration: false,
            },
            performance_settings: PerformanceSettings {
                global_timeout_ms: 60000,
                per_platform_timeout_ms: 15000,
                max_concurrent_validations: 10,
                enable_caching: true,
                cache_ttl_seconds: 3600,
                parallel_platform_validation: true,
                resource_limits: ResourceLimits {
                    max_memory_mb: 1024,
                    max_cpu_percent: 80.0,
                    max_disk_usage_mb: 2048,
                    max_network_requests: 500,
                },
                performance_monitoring: true,
            },
        }
    }
}

impl ValidationConfig {
    /// Create validation configuration with specific platforms
    pub fn with_platforms(mut self, platforms: Vec<Platform>) -> Self {
        self.target_platforms = platforms;
        self
    }

    /// Enable specific validation criteria
    pub fn with_criteria(mut self, criteria: ValidationCriteria) -> Self {
        self.validation_criteria = criteria;
        self
    }

    /// Set minimum acceptable score
    pub fn with_minimum_score(mut self, score: f32) -> Self {
        self.minimum_score = score.clamp(0.0, 100.0);
        self
    }

    /// Enable specific validation features
    pub fn with_features(mut self, features: ValidationFeatures) -> Self {
        self.validation_features = features;
        self
    }

    /// Configure scoring weights
    pub fn with_scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.scoring_weights = weights;
        self
    }

    /// Add custom platform configuration
    pub fn with_platform_config(mut self, platform: Platform, config: PlatformConfig) -> Self {
        self.platform_configs.insert(platform, config);
        self
    }

    /// Configure performance settings
    pub fn with_performance_settings(mut self, settings: PerformanceSettings) -> Self {
        self.performance_settings = settings;
        self
    }

    /// Set environment configuration
    pub fn with_environment_config(mut self, env_config: EnvironmentConfig) -> Self {
        self.environment_config = env_config;
        self
    }

    /// Enable comprehensive validation (all platforms, all criteria)
    pub fn comprehensive() -> Self {
        Self {
            target_platforms: vec![
                Platform::Web,
                Platform::Simulation,
                Platform::Android,
                Platform::IOS,
                Platform::Backend,
            ],
            validation_criteria: ValidationCriteria {
                logical_consistency: true,
                practical_applicability: true,
                platform_compatibility: true,
                performance_requirements: true,
                security_considerations: true,
                user_experience: true,
                custom_metrics: HashMap::new(),
            },
            minimum_score: 75.0,
            ..Default::default()
        }
    }

    /// Enable quick validation (subset of platforms, faster execution)
    pub fn quick() -> Self {
        Self {
            target_platforms: vec![Platform::Web, Platform::Simulation],
            performance_settings: PerformanceSettings {
                per_platform_timeout_ms: 5000,
                global_timeout_ms: 15000,
                ..Default::default()
            },
            validation_features: ValidationFeatures {
                detailed_errors: false,
                performance_profiling: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Enable security-focused validation
    pub fn security_focused() -> Self {
        let mut config = Self::default();
        config.validation_criteria.security_considerations = true;
        config.validation_features.security_scanning = true;
        config.minimum_score = 85.0;
        config
    }

    /// Enable performance-focused validation
    pub fn performance_focused() -> Self {
        let mut config = Self::default();
        config.validation_criteria.performance_requirements = true;
        config.validation_features.performance_profiling = true;
        config.validation_features.performance_profiling = true;
        config.scoring_weights.performance_requirements = 0.40;
        config.scoring_weights.logical_consistency = 0.15;
        config.scoring_weights.practical_applicability = 0.15;
        config.scoring_weights.platform_compatibility = 0.10;
        config.scoring_weights.user_experience = 0.10;
        config.scoring_weights.security_considerations = 0.10;
        config
    }

    /// Get configuration for specific platform
    pub fn get_platform_config(&self, platform: &Platform) -> PlatformConfig {
        self.platform_configs
            .get(platform)
            .cloned()
            .unwrap_or_else(|| PlatformConfig {
                enabled: self.target_platforms.contains(platform),
                timeout_ms: Some(self.performance_settings.per_platform_timeout_ms),
                validation_depth: ValidationDepth::Standard,
                criteria_weights: None,
                custom_rules: Vec::new(),
            })
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> Result<(), VIBEError> {
        if self.target_platforms.is_empty() {
            return Err(VIBEError::ConfigurationError(
                "At least one platform must be specified".to_string(),
            ));
        }

        if self.minimum_score < 0.0 || self.minimum_score > 100.0 {
            return Err(VIBEError::ConfigurationError(
                "Minimum score must be between 0 and 100".to_string(),
            ));
        }

        // Validate scoring weights sum to reasonable value
        let total_weight = self.scoring_weights.logical_consistency
            + self.scoring_weights.practical_applicability
            + self.scoring_weights.platform_compatibility
            + self.scoring_weights.performance_requirements
            + self.scoring_weights.security_considerations
            + self.scoring_weights.user_experience
            + self.scoring_weights.code_quality;

        if total_weight == 0.0 {
            return Err(VIBEError::ConfigurationError(
                "At least one scoring weight must be positive".to_string(),
            ));
        }

        // Validate platform configurations
        for (platform, config) in &self.platform_configs {
            if !self.target_platforms.contains(platform) && config.enabled {
                return Err(VIBEError::ConfigurationError(format!(
                    "Platform {:?} is configured but not in target platforms",
                    platform
                )));
            }
        }

        // Validate performance settings
        if self.performance_settings.global_timeout_ms == 0 {
            return Err(VIBEError::ConfigurationError(
                "Global timeout must be greater than 0".to_string(),
            ));
        }

        if self.performance_settings.max_concurrent_validations == 0 {
            return Err(VIBEError::ConfigurationError(
                "Maximum concurrent validations must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Create configuration from environment variables
    pub fn from_env() -> Result<Self, VIBEError> {
        let mut config = Self::default();

        // Parse platforms from environment
        if let Ok(platforms_str) = std::env::var("VIBE_PLATFORMS") {
            let platforms: Result<Vec<_>, _> = platforms_str
                .split(',')
                .map(|s| match s.trim() {
                    "web" => Ok(Platform::Web),
                    "simulation" => Ok(Platform::Simulation),
                    "android" => Ok(Platform::Android),
                    "ios" => Ok(Platform::IOS),
                    "backend" => Ok(Platform::Backend),
                    _ => Err(VIBEError::ConfigurationError(format!(
                        "Unknown platform: {}",
                        s
                    ))),
                })
                .collect();

            config.target_platforms =
                platforms.map_err(|e| VIBEError::ConfigurationError(e.to_string()))?;
        }

        // Parse minimum score
        if let Ok(score_str) = std::env::var("VIBE_MINIMUM_SCORE") {
            config.minimum_score = score_str
                .parse::<f32>()
                .map_err(|_| VIBEError::ConfigurationError("Invalid minimum score".to_string()))?
                .clamp(0.0, 100.0);
        }

        // Parse timeout settings
        if let Ok(timeout_str) = std::env::var("VIBE_TIMEOUT_MS") {
            config.environment_config.timeout_ms = timeout_str
                .parse::<u64>()
                .map_err(|_| VIBEError::ConfigurationError("Invalid timeout value".to_string()))?;
        }

        // Parse cache settings
        if let Ok(cache_str) = std::env::var("VIBE_CACHE_TTL") {
            config.performance_settings.cache_ttl_seconds = cache_str
                .parse::<u64>()
                .map_err(|_| VIBEError::ConfigurationError("Invalid cache TTL".to_string()))?;
        }

        // Enable features based on environment
        if let Ok(debug_str) = std::env::var("VIBE_DEBUG") {
            if debug_str.parse::<bool>().unwrap_or(false) {
                config.validation_features.detailed_errors = true;
                config.validation_features.performance_profiling = true;
            }
        }

        Ok(config)
    }

    /// Convert configuration to environment variables
    pub fn to_env_vars(&self) -> HashMap<String, String> {
        let mut env_vars = HashMap::new();

        // Platforms
        let platforms_str = self
            .target_platforms
            .iter()
            .map(|p| match p {
                Platform::Web => "web",
                Platform::Simulation => "simulation",
                Platform::Android => "android",
                Platform::IOS => "ios",
                Platform::Backend => "backend",
            })
            .collect::<Vec<_>>()
            .join(",");
        env_vars.insert("VIBE_PLATFORMS".to_string(), platforms_str);

        // Other settings
        env_vars.insert(
            "VIBE_MINIMUM_SCORE".to_string(),
            self.minimum_score.to_string(),
        );
        env_vars.insert(
            "VIBE_TIMEOUT_MS".to_string(),
            self.environment_config.timeout_ms.to_string(),
        );
        env_vars.insert(
            "VIBE_CACHE_TTL".to_string(),
            self.performance_settings.cache_ttl_seconds.to_string(),
        );

        // Feature flags
        env_vars.insert(
            "VIBE_DETAILED_ERRORS".to_string(),
            self.validation_features.detailed_errors.to_string(),
        );
        env_vars.insert(
            "VIBE_PERFORMANCE_PROFILING".to_string(),
            self.validation_features.performance_profiling.to_string(),
        );
        env_vars.insert(
            "VIBE_SECURITY_SCANNING".to_string(),
            self.validation_features.security_scanning.to_string(),
        );
        env_vars.insert(
            "VIBE_ACCESSIBILITY_TESTING".to_string(),
            self.validation_features.accessibility_testing.to_string(),
        );

        env_vars
    }
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_ms: None,
            validation_depth: ValidationDepth::Standard,
            criteria_weights: None,
            custom_rules: Vec::new(),
        }
    }
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            logical_consistency: 0.25,
            practical_applicability: 0.20,
            platform_compatibility: 0.15,
            performance_requirements: 0.15,
            security_considerations: 0.10,
            user_experience: 0.10,
            code_quality: 0.05,
            custom_weights: HashMap::new(),
        }
    }
}

impl Default for ValidationFeatures {
    fn default() -> Self {
        Self {
            detailed_errors: true,
            performance_profiling: true,
            security_scanning: false,
            accessibility_testing: false,
            cross_platform_consistency: true,
            automated_fixes: false,
            real_time_validation: false,
            external_tool_integration: false,
        }
    }
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            global_timeout_ms: 60000,
            per_platform_timeout_ms: 15000,
            max_concurrent_validations: 10,
            enable_caching: true,
            cache_ttl_seconds: 3600,
            parallel_platform_validation: true,
            resource_limits: ResourceLimits {
                max_memory_mb: 1024,
                max_cpu_percent: 80.0,
                max_disk_usage_mb: 2048,
                max_network_requests: 500,
            },
            performance_monitoring: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ValidationConfig::default();
        assert!(config.target_platforms.contains(&Platform::Web));
        assert_eq!(config.minimum_score, 70.0);
        assert!(config.validation_criteria.logical_consistency);
    }

    #[test]
    fn test_config_with_platforms() {
        let config =
            ValidationConfig::default().with_platforms(vec![Platform::Web, Platform::Backend]);
        assert_eq!(config.target_platforms.len(), 2);
        assert!(config.target_platforms.contains(&Platform::Web));
        assert!(config.target_platforms.contains(&Platform::Backend));
    }

    #[test]
    fn test_comprehensive_config() {
        let config = ValidationConfig::comprehensive();
        assert_eq!(config.target_platforms.len(), 5);
        assert!(config.validation_criteria.security_considerations);
        assert_eq!(config.minimum_score, 75.0);
    }

    #[test]
    fn test_quick_config() {
        let config = ValidationConfig::quick();
        assert_eq!(config.target_platforms.len(), 2);
        assert_eq!(config.performance_settings.per_platform_timeout_ms, 5000);
        assert!(!config.validation_features.detailed_errors);
    }

    #[test]
    fn test_security_focused_config() {
        let config = ValidationConfig::security_focused();
        assert!(config.validation_criteria.security_considerations);
        assert!(config.validation_features.security_scanning);
        assert_eq!(config.minimum_score, 85.0);
    }

    #[test]
    fn test_config_validation() {
        let mut config = ValidationConfig {
            target_platforms: vec![],
            ..Default::default()
        };
        assert!(config.validate().is_err());

        config.target_platforms = vec![Platform::Web];
        config.minimum_score = 150.0;
        assert!(config.validate().is_err());

        config.minimum_score = 70.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_environment_config() {
        std::env::set_var("VIBE_MINIMUM_SCORE", "85");
        std::env::set_var("VIBE_TIMEOUT_MS", "45000");

        let config = ValidationConfig::from_env().unwrap();
        assert_eq!(config.minimum_score, 85.0);
        assert_eq!(config.environment_config.timeout_ms, 45000);

        std::env::remove_var("VIBE_MINIMUM_SCORE");
        std::env::remove_var("VIBE_TIMEOUT_MS");
    }
}
