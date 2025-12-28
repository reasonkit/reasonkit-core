//! Common types for ARF Core

use std::time::Duration;

/// Reasoning task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Task identifier
pub type TaskId = ulid::Ulid;

/// Timeout configuration
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    pub default: Duration,
    pub max: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default: Duration::from_secs(30),
            max: Duration::from_secs(300),
        }
    }
}
