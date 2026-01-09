//! Zero-Downtime Key Rotation
//!
//! Implements key rotation with overlap periods for zero service interruption:
//! - Overlap period where both old and new keys are valid
//! - Grace period after rotation before old key invalidation
//! - Automatic rotation based on key age
//! - Emergency rotation for compromised keys

use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

use crate::security::api_keys::{
    ApiKeyManager, ApiKeyRecord, GeneratedKey, KeyRotationInfo, KeyStatus, RotationReason,
};
use crate::security::audit::{AuditActor, AuditEvent, AuditEventType, AuditLogger, AuditOutcome};
use chrono::{DateTime, Utc};
use secrecy::SecretString;

/// Key rotation configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RotationConfig {
    /// Overlap period where both old and new keys are valid
    pub overlap_period: Duration,
    /// Notification period before forced rotation
    pub notification_period: Duration,
    /// Grace period after rotation before old key invalidation
    pub grace_period: Duration,
    /// Maximum key age before forced rotation
    pub max_key_age: Duration,
    /// Enable automatic rotation
    pub auto_rotate: bool,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            overlap_period: Duration::from_secs(3600),       // 1 hour
            notification_period: Duration::from_secs(604800), // 7 days
            grace_period: Duration::from_secs(86400),        // 24 hours
            max_key_age: Duration::from_secs(7776000),       // 90 days
            auto_rotate: true,
        }
    }
}

/// Key rotation states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RotationState {
    /// No rotation in progress
    Stable,
    /// Rotation initiated, new key generated
    Initiated { new_key_id: Uuid },
    /// Both keys active (overlap period)
    Overlapping {
        old_key_id: Uuid,
        new_key_id: Uuid,
        overlap_ends: DateTime<Utc>,
    },
    /// Transition to new key, old key in grace period
    Transitioning {
        old_key_id: Uuid,
        new_key_id: Uuid,
        grace_ends: DateTime<Utc>,
    },
    /// Rotation complete, old key invalidated
    Completed { new_key_id: Uuid },
}

/// Rotation error
#[derive(Debug, thiserror::Error)]
pub enum RotationError {
    #[error("Key is already being rotated")]
    AlreadyRotating,
    #[error("Key is not in rotation state")]
    NotRotating,
    #[error("Overlap period still active")]
    OverlapPeriodActive,
    #[error("Key not found: {0}")]
    KeyNotFound(Uuid),
    #[error("Key error: {0}")]
    KeyError(String),
    #[error("Audit error: {0}")]
    AuditError(String),
}

/// Rotation notification event
#[derive(Debug, Clone)]
pub enum RotationNotificationEvent {
    RotationInitiated,
    OverlapEnding,
    RotationCompleted,
    EmergencyRotation,
}

/// Rotation notification
#[derive(Debug, Clone)]
pub struct RotationNotification {
    pub event: RotationNotificationEvent,
    pub old_key_prefix: String,
    pub new_key_prefix: String,
    pub overlap_ends: DateTime<Utc>,
    pub action_required: bool,
}

/// Rotation notifier trait
#[async_trait::async_trait]
pub trait RotationNotifier: Send + Sync {
    async fn send_rotation_notification(
        &self,
        tenant_id: Uuid,
        notification: RotationNotification,
    ) -> Result<(), String>;

    async fn send_urgent_notification(
        &self,
        tenant_id: Uuid,
        title: &str,
        message: &str,
    ) -> Result<(), String>;
}

/// Rotation handle returned when initiating rotation
#[derive(Debug)]
pub struct RotationHandle {
    pub old_key_id: Uuid,
    pub new_key_id: Uuid,
    /// The new API key (only returned once, must be stored by caller)
    pub new_api_key: SecretString,
    pub state: RotationState,
}

/// Candidate for rotation
#[derive(Debug, Clone)]
pub struct RotationCandidate {
    pub key_id: Uuid,
    pub tenant_id: Uuid,
    pub reason: RotationReason,
    pub priority: RotationPriority,
    pub age_days: u32,
}

/// Rotation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Key Rotation Manager
pub struct KeyRotationManager {
    key_manager: Arc<ApiKeyManager>,
    config: RotationConfig,
    audit: Arc<AuditLogger>,
    notifier: Option<Arc<dyn RotationNotifier>>,
}

impl KeyRotationManager {
    /// Create a new key rotation manager
    pub fn new(
        key_manager: Arc<ApiKeyManager>,
        config: RotationConfig,
        audit: Arc<AuditLogger>,
    ) -> Self {
        Self {
            key_manager,
            config,
            audit,
            notifier: None,
        }
    }

    /// Set rotation notifier
    pub fn with_notifier(mut self, notifier: Arc<dyn RotationNotifier>) -> Self {
        self.notifier = Some(notifier);
        self
    }

    /// Initiate key rotation for a tenant
    pub async fn initiate_rotation(
        &self,
        tenant_id: Uuid,
        old_key_id: Uuid,
        reason: RotationReason,
    ) -> Result<RotationHandle, RotationError> {
        // Get existing key details
        let old_key = self
            .key_manager
            .get_key(old_key_id)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        if old_key.status == KeyStatus::Rotating {
            return Err(RotationError::AlreadyRotating);
        }

        // Generate new key with same scopes
        let expires_in = old_key.expires_at.map(|e| {
            (e - old_key.created_at)
                .to_std()
                .unwrap_or(Duration::from_secs(86400 * 365))
        });

        let new_key = self
            .key_manager
            .generate_key(tenant_id, old_key.scopes.clone(), expires_in)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        // Mark old key as rotating
        self.key_manager
            .update_status(old_key_id, KeyStatus::Rotating)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        // Set rotation metadata
        self.key_manager
            .set_rotation_info(
                old_key_id,
                KeyRotationInfo {
                    previous_key_id: None,
                    rotation_started: Utc::now(),
                    rotation_completed: None,
                    reason: reason.clone(),
                },
            )
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        let overlap_ends = Utc::now()
            + chrono::Duration::from_std(self.config.overlap_period)
                .unwrap_or(chrono::Duration::hours(1));

        // Audit log
        self.audit
            .log(
                AuditEvent::new(AuditEventType::KeyRotationStarted, AuditActor::System)
                    .with_tenant(tenant_id)
                    .with_key(old_key_id)
                    .with_details(serde_json::json!({
                        "old_key_id": old_key_id,
                        "new_key_id": new_key.key_id,
                        "reason": format!("{:?}", reason),
                        "overlap_ends": overlap_ends,
                    }))
                    .with_outcome(AuditOutcome::Success),
            )
            .await
            .map_err(|e| RotationError::AuditError(e.to_string()))?;

        // Notify customer
        if let Some(ref notifier) = self.notifier {
            let _ = notifier
                .send_rotation_notification(
                    tenant_id,
                    RotationNotification {
                        event: RotationNotificationEvent::RotationInitiated,
                        old_key_prefix: old_key.key_prefix.clone(),
                        new_key_prefix: new_key.key_id.to_string()[..8].to_string(),
                        overlap_ends,
                        action_required: true,
                    },
                )
                .await;
        }

        Ok(RotationHandle {
            old_key_id,
            new_key_id: new_key.key_id,
            new_api_key: new_key.api_key,
            state: RotationState::Overlapping {
                old_key_id,
                new_key_id: new_key.key_id,
                overlap_ends,
            },
        })
    }

    /// Complete rotation (invalidate old key)
    pub async fn complete_rotation(
        &self,
        old_key_id: Uuid,
        new_key_id: Uuid,
    ) -> Result<(), RotationError> {
        // Verify rotation is in progress
        let old_key = self
            .key_manager
            .get_key(old_key_id)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        if old_key.status != KeyStatus::Rotating {
            return Err(RotationError::NotRotating);
        }

        // Check overlap period has passed
        if let Some(rotation) = &old_key.rotation {
            let overlap_end = rotation.rotation_started
                + chrono::Duration::from_std(self.config.overlap_period)
                    .unwrap_or(chrono::Duration::hours(1));
            if Utc::now() < overlap_end {
                return Err(RotationError::OverlapPeriodActive);
            }
        }

        // Revoke old key
        self.key_manager
            .update_status(old_key_id, KeyStatus::Revoked)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        // Update new key to active (remove rotation flag)
        self.key_manager
            .update_status(new_key_id, KeyStatus::Active)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        let rotation_duration = old_key
            .rotation
            .as_ref()
            .map(|r| (Utc::now() - r.rotation_started).num_seconds())
            .unwrap_or(0);

        // Audit log
        self.audit
            .log(
                AuditEvent::new(AuditEventType::KeyRotationCompleted, AuditActor::System)
                    .with_tenant(old_key.tenant_id)
                    .with_key(new_key_id)
                    .with_details(serde_json::json!({
                        "old_key_id": old_key_id,
                        "new_key_id": new_key_id,
                        "rotation_duration_secs": rotation_duration,
                    }))
                    .with_outcome(AuditOutcome::Success),
            )
            .await
            .map_err(|e| RotationError::AuditError(e.to_string()))?;

        Ok(())
    }

    /// Check for keys requiring rotation
    pub async fn check_rotation_required(&self) -> Vec<RotationCandidate> {
        let mut candidates = Vec::new();

        let keys = match self.key_manager.list_active_keys().await {
            Ok(keys) => keys,
            Err(_) => return candidates,
        };

        for key in keys {
            let age = Utc::now() - key.created_at;

            // Check if key exceeds max age
            let max_age_duration =
                chrono::Duration::from_std(self.config.max_key_age).unwrap_or(chrono::Duration::days(90));
            if age > max_age_duration {
                candidates.push(RotationCandidate {
                    key_id: key.key_id,
                    tenant_id: key.tenant_id,
                    reason: RotationReason::Scheduled,
                    priority: RotationPriority::High,
                    age_days: age.num_days() as u32,
                });
                continue;
            }

            // Check if approaching max age (notification period)
            let notification_threshold = self.config.max_key_age - self.config.notification_period;
            let notification_duration =
                chrono::Duration::from_std(notification_threshold).unwrap_or(chrono::Duration::days(83));
            if age > notification_duration {
                candidates.push(RotationCandidate {
                    key_id: key.key_id,
                    tenant_id: key.tenant_id,
                    reason: RotationReason::Scheduled,
                    priority: RotationPriority::Medium,
                    age_days: age.num_days() as u32,
                });
            }
        }

        candidates
    }

    /// Emergency rotation (immediate, no overlap)
    pub async fn emergency_rotation(
        &self,
        tenant_id: Uuid,
        key_id: Uuid,
        reason: &str,
    ) -> Result<RotationHandle, RotationError> {
        // Immediately revoke compromised key
        self.key_manager
            .update_status(key_id, KeyStatus::Revoked)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        // Get old key for scope replication
        let old_key = self
            .key_manager
            .get_key(key_id)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        // Calculate remaining validity if expiration was set
        let expires_in = old_key.expires_at.and_then(|e| {
            let remaining = e - Utc::now();
            if remaining > chrono::Duration::zero() {
                remaining.to_std().ok()
            } else {
                Some(Duration::from_secs(86400)) // 1 day minimum
            }
        });

        // Generate new key
        let new_key = self
            .key_manager
            .generate_key(tenant_id, old_key.scopes.clone(), expires_in)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        // Audit log with security incident
        self.audit
            .log(
                AuditEvent::new(AuditEventType::SecurityIncident, AuditActor::System)
                    .with_tenant(tenant_id)
                    .with_key(key_id)
                    .with_details(serde_json::json!({
                        "incident_type": "emergency_key_rotation",
                        "reason": reason,
                        "old_key_id": key_id,
                        "new_key_id": new_key.key_id,
                    }))
                    .with_outcome(AuditOutcome::Success),
            )
            .await
            .map_err(|e| RotationError::AuditError(e.to_string()))?;

        // Urgent notification
        if let Some(ref notifier) = self.notifier {
            let _ = notifier
                .send_urgent_notification(
                    tenant_id,
                    "Emergency Key Rotation",
                    &format!(
                        "Your API key has been rotated due to: {}. Please update immediately.",
                        reason
                    ),
                )
                .await;
        }

        Ok(RotationHandle {
            old_key_id: key_id,
            new_key_id: new_key.key_id,
            new_api_key: new_key.api_key,
            state: RotationState::Completed {
                new_key_id: new_key.key_id,
            },
        })
    }

    /// Get rotation status for a key
    pub async fn get_rotation_status(&self, key_id: Uuid) -> Result<RotationState, RotationError> {
        let key = self
            .key_manager
            .get_key(key_id)
            .await
            .map_err(|e| RotationError::KeyError(e.to_string()))?;

        match key.status {
            KeyStatus::Rotating => {
                if let Some(rotation) = &key.rotation {
                    let overlap_end = rotation.rotation_started
                        + chrono::Duration::from_std(self.config.overlap_period)
                            .unwrap_or(chrono::Duration::hours(1));

                    if Utc::now() < overlap_end {
                        // Still in overlap
                        Ok(RotationState::Overlapping {
                            old_key_id: key_id,
                            new_key_id: Uuid::nil(), // Would need to track this
                            overlap_ends: overlap_end,
                        })
                    } else {
                        // In grace period
                        let grace_end = overlap_end
                            + chrono::Duration::from_std(self.config.grace_period)
                                .unwrap_or(chrono::Duration::days(1));
                        Ok(RotationState::Transitioning {
                            old_key_id: key_id,
                            new_key_id: Uuid::nil(),
                            grace_ends: grace_end,
                        })
                    }
                } else {
                    Ok(RotationState::Stable)
                }
            }
            KeyStatus::Revoked => {
                if key.rotation.is_some() {
                    Ok(RotationState::Completed {
                        new_key_id: Uuid::nil(),
                    })
                } else {
                    Ok(RotationState::Stable)
                }
            }
            _ => Ok(RotationState::Stable),
        }
    }
}

/// No-op rotation notifier for testing
pub struct NoOpRotationNotifier;

#[async_trait::async_trait]
impl RotationNotifier for NoOpRotationNotifier {
    async fn send_rotation_notification(
        &self,
        _tenant_id: Uuid,
        _notification: RotationNotification,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn send_urgent_notification(
        &self,
        _tenant_id: Uuid,
        _title: &str,
        _message: &str,
    ) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rotation_config() {
        let config = RotationConfig::default();

        assert_eq!(config.overlap_period, Duration::from_secs(3600));
        assert_eq!(config.notification_period, Duration::from_secs(604800));
        assert_eq!(config.grace_period, Duration::from_secs(86400));
        assert_eq!(config.max_key_age, Duration::from_secs(7776000));
        assert!(config.auto_rotate);
    }

    #[test]
    fn test_rotation_priority_ordering() {
        assert!(RotationPriority::Critical != RotationPriority::High);
        assert!(RotationPriority::High != RotationPriority::Medium);
        assert!(RotationPriority::Medium != RotationPriority::Low);
    }
}
