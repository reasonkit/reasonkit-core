//! Telemetry Configuration
//!
//! Configuration structures for the telemetry system.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable telemetry collection
    pub enabled: bool,

    /// Path to telemetry database
    pub db_path: PathBuf,

    /// Privacy settings
    pub privacy: PrivacyConfig,

    /// Contribute anonymized data to community model
    pub community_contribution: bool,

    /// Batch size for writing events
    pub batch_size: usize,

    /// Flush interval in seconds
    pub flush_interval_secs: u64,

    /// Maximum database size in MB (0 = unlimited)
    pub max_db_size_mb: u64,

    /// Retention period in days (0 = forever)
    pub retention_days: u32,

    /// Enable aggregation for ML training
    pub enable_aggregation: bool,

    /// Aggregation interval in hours
    pub aggregation_interval_hours: u32,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Opt-in by default
            db_path: PathBuf::from(".rk_telemetry.db"),
            privacy: PrivacyConfig::default(),
            community_contribution: false, // Opt-in
            batch_size: 100,
            flush_interval_secs: 60,
            max_db_size_mb: 100, // 100MB default limit
            retention_days: 90,  // 3 months default
            enable_aggregation: true,
            aggregation_interval_hours: 24,
        }
    }
}

impl TelemetryConfig {
    /// Create a minimal config for testing
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            db_path: PathBuf::from(":memory:"),
            privacy: PrivacyConfig::strict(),
            community_contribution: false,
            batch_size: 10,
            flush_interval_secs: 5,
            max_db_size_mb: 0,
            retention_days: 0,
            enable_aggregation: false,
            aggregation_interval_hours: 24,
        }
    }

    /// Create a production config
    pub fn production() -> Self {
        Self {
            enabled: true,
            db_path: Self::default_db_path(),
            privacy: PrivacyConfig::default(),
            community_contribution: false,
            batch_size: 100,
            flush_interval_secs: 60,
            max_db_size_mb: 500,
            retention_days: 365,
            enable_aggregation: true,
            aggregation_interval_hours: 24,
        }
    }

    /// Get the default database path in user's data directory
    pub fn default_db_path() -> PathBuf {
        if let Some(data_dir) = dirs::data_local_dir() {
            data_dir.join("reasonkit").join(".rk_telemetry.db")
        } else {
            PathBuf::from(".rk_telemetry.db")
        }
    }

    /// Load config from environment and/or file
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Check RK_TELEMETRY_ENABLED env var
        if let Ok(val) = std::env::var("RK_TELEMETRY_ENABLED") {
            config.enabled = val.to_lowercase() == "true" || val == "1";
        }

        // Check RK_TELEMETRY_PATH env var
        if let Ok(path) = std::env::var("RK_TELEMETRY_PATH") {
            config.db_path = PathBuf::from(path);
        }

        // Check RK_TELEMETRY_COMMUNITY env var
        if let Ok(val) = std::env::var("RK_TELEMETRY_COMMUNITY") {
            config.community_contribution = val.to_lowercase() == "true" || val == "1";
        }

        config
    }
}

/// Privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Strip PII from all stored data
    pub strip_pii: bool,

    /// Block events containing sensitive keywords
    pub block_sensitive: bool,

    /// Apply differential privacy to aggregates
    pub differential_privacy: bool,

    /// Differential privacy epsilon (lower = more private)
    pub dp_epsilon: f64,

    /// Redact file paths
    pub redact_file_paths: bool,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            strip_pii: true,
            block_sensitive: false,
            differential_privacy: false,
            dp_epsilon: 1.0,
            redact_file_paths: true,
        }
    }
}

impl PrivacyConfig {
    /// Strict privacy settings (maximum protection)
    pub fn strict() -> Self {
        Self {
            strip_pii: true,
            block_sensitive: true,
            differential_privacy: true,
            dp_epsilon: 0.1, // Very private
            redact_file_paths: true,
        }
    }

    /// Relaxed privacy settings (more utility, less privacy)
    pub fn relaxed() -> Self {
        Self {
            strip_pii: true,
            block_sensitive: false,
            differential_privacy: false,
            dp_epsilon: 1.0,
            redact_file_paths: false,
        }
    }
}

/// Consent record for GDPR/CCPA compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Unique consent ID
    pub id: uuid::Uuid,

    /// Timestamp of consent
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Allow local telemetry storage
    pub local_telemetry: bool,

    /// Allow aggregated sharing
    pub aggregated_sharing: bool,

    /// Contribute to community model
    pub community_contribution: bool,

    /// Consent form version (for re-consent when updated)
    pub consent_version: u32,

    /// Hashed IP for legal compliance
    pub ip_hash: Option<String>,
}

impl ConsentRecord {
    /// Current consent form version
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new consent record with all permissions
    pub fn allow_all() -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            local_telemetry: true,
            aggregated_sharing: true,
            community_contribution: true,
            consent_version: Self::CURRENT_VERSION,
            ip_hash: None,
        }
    }

    /// Create a new consent record with minimal permissions
    pub fn minimal() -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            local_telemetry: true,
            aggregated_sharing: false,
            community_contribution: false,
            consent_version: Self::CURRENT_VERSION,
            ip_hash: None,
        }
    }

    /// Create a new consent record denying all telemetry
    pub fn deny_all() -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            local_telemetry: false,
            aggregated_sharing: false,
            community_contribution: false,
            consent_version: Self::CURRENT_VERSION,
            ip_hash: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_disabled() {
        let config = TelemetryConfig::default();
        assert!(!config.enabled); // Opt-in by default
    }

    #[test]
    fn test_strict_privacy() {
        let privacy = PrivacyConfig::strict();
        assert!(privacy.strip_pii);
        assert!(privacy.block_sensitive);
        assert!(privacy.differential_privacy);
        assert!(privacy.dp_epsilon < 1.0); // More private
    }

    #[test]
    fn test_consent_versions() {
        let consent = ConsentRecord::allow_all();
        assert_eq!(consent.consent_version, ConsentRecord::CURRENT_VERSION);
    }

    #[test]
    fn test_from_env() {
        // This test just verifies the method runs without panicking
        let config = TelemetryConfig::from_env();
        // Verify the config was created (db_path is always set)
        assert!(!config.db_path.as_os_str().is_empty());
    }
}
