//! Daemon Logger
//!
//! Manages logging and log rotation for the daemon process.

use crate::error::Result;
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::fmt::format::FmtSpan;

/// Daemon logger configuration
pub struct DaemonLogger {
    log_path: PathBuf,
    level: Level,
}

impl DaemonLogger {
    /// Create new daemon logger
    pub fn new(log_path: PathBuf) -> Result<Self> {
        // Ensure log directory exists
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        Ok(Self {
            log_path,
            level: Level::INFO,
        })
    }

    /// Initialize logging
    pub fn init(&self) -> Result<()> {
        let log_dir = self
            .log_path
            .parent()
            .ok_or_else(|| crate::error::Error::config("Invalid log path"))?;

        let file_appender = tracing_appender::rolling::daily(log_dir, "mcp-daemon.log");

        let subscriber = tracing_subscriber::fmt()
            .with_writer(file_appender)
            .with_ansi(false)
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true)
            .with_level(true)
            .with_max_level(self.level)
            .with_span_events(FmtSpan::CLOSE)
            .json()
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .map_err(|e| crate::error::Error::config(format!("Failed to set logger: {}", e)))?;

        Ok(())
    }

    /// Set log level
    pub fn with_level(mut self, level: Level) -> Self {
        self.level = level;
        self
    }
}
