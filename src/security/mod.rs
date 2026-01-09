//! Enterprise Security Module for ReasonKit MCP Server
//!
//! This module implements enterprise-grade security controls for the paid MCP server,
//! designed to satisfy SOC 2 Type II, GDPR, and Fortune 500 procurement requirements.
//!
//! # Security Components
//!
//! - [`api_keys`] - Secure API key generation, storage, and validation
//! - [`rate_limit`] - Multi-tier, tenant-aware rate limiting
//! - [`ip_access`] - Enterprise IP allowlisting and geo-blocking
//! - [`audit`] - SOC 2 compliant audit logging
//! - [`rotation`] - Zero-downtime key rotation
//! - [`brute_force`] - Adaptive brute force protection
//!
//! # Security Disclaimer
//!
//! This module provides security guidance and implementations. Users must:
//! - Engage qualified security professionals for formal security assessments
//! - Conduct independent penetration testing
//! - Obtain professional SOC 2 certification from qualified auditors
//!
//! # Example
//!
//! ```rust,ignore
//! use reasonkit_core::security::{ApiKeyManager, RateLimiter, AuditLogger};
//!
//! // Initialize security components
//! let key_manager = ApiKeyManager::new(kek, db, audit.clone());
//! let rate_limiter = RateLimiter::new(config);
//!
//! // Validate request
//! let validated = key_manager.validate_key(&api_key, &scopes, client_ip).await?;
//! let rate_decision = rate_limiter.check(validated.tenant_id, client_ip, endpoint).await;
//! ```

pub mod api_keys;
pub mod audit;
pub mod brute_force;
pub mod ip_access;
pub mod rate_limit;
pub mod rotation;

// Re-exports for convenience
pub use api_keys::{ApiKeyManager, ApiKeyRecord, GeneratedKey, KeyStatus, ValidatedKey};
pub use audit::{AuditEvent, AuditEventType, AuditLogger, AuditOutcome};
pub use brute_force::{BruteForceAction, BruteForceProtection};
pub use ip_access::{IpAccessController, IpValidationResult, TenantIpConfig};
pub use rate_limit::{RateLimitDecision, RateLimiter, TierLimits};
pub use rotation::{KeyRotationManager, RotationHandle, RotationState};

use std::sync::Arc;

/// Unified security state for middleware integration
#[derive(Clone)]
pub struct SecurityState {
    /// API key manager
    pub api_key_manager: Arc<ApiKeyManager>,
    /// Rate limiter
    pub rate_limiter: Arc<RateLimiter>,
    /// IP access controller
    pub ip_controller: Arc<IpAccessController>,
    /// Brute force protection
    pub brute_force: Arc<BruteForceProtection>,
    /// Audit logger
    pub audit: Arc<AuditLogger>,
}

/// Error types for security operations
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    #[error("Key error: {0}")]
    Key(#[from] api_keys::KeyError),

    #[error("Rate limit error: {0}")]
    RateLimit(String),

    #[error("IP access error: {0}")]
    IpAccess(String),

    #[error("Audit error: {0}")]
    Audit(#[from] audit::AuditError),

    #[error("Rotation error: {0}")]
    Rotation(#[from] rotation::RotationError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for security operations
pub type SecurityResult<T> = std::result::Result<T, SecurityError>;
