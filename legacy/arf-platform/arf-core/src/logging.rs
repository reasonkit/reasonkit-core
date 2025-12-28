//! Logging module for ARF Core

use crate::error::Result;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize the logging system
pub fn init() -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .try_init()
        .ok(); // Ignore if already initialized

    Ok(())
}
