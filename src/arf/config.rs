//! Configuration loading and management

use crate::arf::types::*;
use crate::error::Result;
use config::{Config as ConfigLoader, File};
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone)]
pub struct Config {
    pub inner: ArfConfig,
}

impl Config {
    /// Load configuration from default locations
    pub fn load() -> Result<Self> {
        Self::load_from_paths(&[
            "arf.toml",
            "arf.yaml",
            "arf.json",
            "/etc/arf/config.toml",
            "/etc/arf/config.yaml",
            "/etc/arf/config.json",
        ])
    }

    /// Load configuration from specific file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();

        if path_ref.extension().is_none() {
            let contents = std::fs::read_to_string(path_ref)?;
            if let Ok(config) = serde_json::from_str::<ArfConfig>(&contents) {
                return Ok(Self { inner: config });
            }
            if let Ok(config) = toml::from_str::<ArfConfig>(&contents) {
                return Ok(Self { inner: config });
            }
            return Err(crate::error::Error::config(
                "Failed to parse extensionless config file".to_string(),
            ));
        }

        let loader = ConfigLoader::builder().add_source(File::from(path_ref));

        let config: ArfConfig = loader.build()?.try_deserialize()?;
        Ok(Self { inner: config })
    }

    /// Load configuration from multiple possible paths
    pub fn load_from_paths<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut loader = ConfigLoader::builder();

        // Try to load from each path that exists
        for path in paths {
            if path.as_ref().exists() {
                loader = loader.add_source(File::from(path.as_ref()));
                break; // Use the first one found
            }
        }

        // Add default configuration
        loader = loader.add_source(config::File::from_str(
            include_str!("../../config/default.toml"),
            config::FileFormat::Toml,
        ));

        let config: ArfConfig = loader.build()?.try_deserialize()?;
        Ok(Self { inner: config })
    }

    /// Create default configuration
    pub fn default_config() -> Self {
        Self {
            inner: ArfConfig {
                version: env!("CARGO_PKG_VERSION").to_string(),
                runtime: RuntimeConfig {
                    max_concurrent_sessions: 10,
                    session_timeout_seconds: 3600,
                    worker_threads: num_cpus::get(),
                },
                engine: EngineConfig {
                    max_steps_per_session: 50,
                    step_timeout_seconds: 300,
                    validation_enabled: true,
                    cognitive_load_monitoring: true,
                },
                plugins: PluginConfig {
                    enabled: true,
                    plugin_directory: "./plugins".to_string(),
                    max_plugins: 100,
                    hot_reload: false,
                },
                logging: LoggingConfig {
                    level: "INFO".to_string(),
                    format: "json".to_string(),
                    file_output: Some("./logs/arf.log".to_string()),
                },
            },
        }
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.inner).map_err(|e| {
            crate::error::Error::config(format!("Failed to serialize config: {}", e))
        })?;

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get runtime configuration
    pub fn runtime(&self) -> &RuntimeConfig {
        &self.inner.runtime
    }

    /// Get engine configuration
    pub fn engine(&self) -> &EngineConfig {
        &self.inner.engine
    }

    /// Get plugin configuration
    pub fn plugins(&self) -> &PluginConfig {
        &self.inner.plugins
    }

    /// Get logging configuration
    pub fn logging(&self) -> &LoggingConfig {
        &self.inner.logging
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default_config();
        assert_eq!(config.runtime().max_concurrent_sessions, 10);
        assert_eq!(config.engine().max_steps_per_session, 50);
        assert!(config.plugins().enabled);
    }

    #[test]
    fn test_config_save_load() {
        let config = Config::default_config();
        let temp_file = NamedTempFile::new().unwrap();

        // Save config
        config.save_to_file(temp_file.path()).unwrap();

        // Load config
        let loaded_config = Config::load_from_file(temp_file.path()).unwrap();

        // Verify they match
        assert_eq!(
            config.runtime().max_concurrent_sessions,
            loaded_config.runtime().max_concurrent_sessions
        );
    }
}
