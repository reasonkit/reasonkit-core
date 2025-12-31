//! # MiniMax M2 Integration Module
//!
//! NOTE: The full M2 integration is currently experimental.
//!
//! This module intentionally provides a small, compiling API surface so that
//! `cargo test --all-features` remains green while the full implementation is
//! iterated on.

pub mod types;

pub use types::*;

// The full implementation files still exist in `src/m2/`, but are intentionally
// not compiled until stabilized.
//
pub mod benchmarks;
pub mod connector;
pub mod engine;
pub mod protocol_generator;

use crate::error::Error;

/// Primary M2 integration service.
///
/// The full interleaved-thinking engine is not yet wired in.
#[derive(Debug, Clone)]
pub struct M2IntegrationService {
    config: M2Config,
    integration_config: M2IntegrationConfig,
}

impl M2IntegrationService {
    /// Create a new service.
    pub async fn new(
        config: M2Config,
        integration_config: M2IntegrationConfig,
    ) -> Result<Self, Error> {
        Ok(Self {
            config,
            integration_config,
        })
    }

    /// Execute an M2 run for a `UseCase`.
    ///
    /// Currently returns `Error::M2ExecutionError` because the integration is a stub.
    pub async fn execute_for_use_case(
        &self,
        _use_case: UseCase,
        _input: ProtocolInput,
        _framework: Option<AgentFramework>,
    ) -> Result<InterleavedResult, Error> {
        let _ = (&self.config, &self.integration_config);
        Err(Error::M2ExecutionError(
            "M2 integration is not implemented in this build".to_string(),
        ))
    }

    /// Map a high-level `UseCase` to a `TaskClassification`.
    pub fn classify_use_case(
        &self,
        use_case: UseCase,
        _input: &ProtocolInput,
    ) -> Result<TaskClassification, Error> {
        Ok(TaskClassification::from(use_case))
    }
}

/// Builder for `M2IntegrationService`.
#[derive(Debug, Clone)]
pub struct M2ServiceBuilder {
    config: Option<M2Config>,
    integration_config: Option<M2IntegrationConfig>,
}

impl M2ServiceBuilder {
    pub fn new() -> Self {
        Self {
            config: None,
            integration_config: None,
        }
    }

    pub fn with_config(mut self, config: M2Config) -> Self {
        self.config = Some(config);
        self
    }

    pub fn with_integration_config(mut self, integration_config: M2IntegrationConfig) -> Self {
        self.integration_config = Some(integration_config);
        self
    }

    pub async fn build(self) -> Result<M2IntegrationService, Error> {
        let config = self.config.unwrap_or_default();
        let integration_config = self.integration_config.unwrap_or_default();
        M2IntegrationService::new(config, integration_config).await
    }
}

impl Default for M2ServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}
