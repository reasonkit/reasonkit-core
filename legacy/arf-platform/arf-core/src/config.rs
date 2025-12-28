//! Configuration module for ARF Core

use std::path::PathBuf;
use crate::error::Result;

/// ARF Configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// Base directory for ARF data
    pub data_dir: PathBuf,
    /// Enable debug mode
    pub debug: bool,
}

impl Config {
    /// Load configuration from default locations
    pub fn load() -> Result<Self> {
        Ok(Self {
            data_dir: PathBuf::from(".arf"),
            debug: false,
        })
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from(".arf"),
            debug: false,
        }
    }
}
