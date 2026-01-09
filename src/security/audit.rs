//! SOC 2 Compliant Audit Logging
//!
//! Implements enterprise-grade audit logging with immutable storage,
//! cryptographic chaining, and SIEM integration for SOC 2 Type II compliance.
//!
//! # SOC 2 Control Mapping
//!
//! - CC5.2: System Monitoring - Real-time event logging
//! - CC6.1: Key Management - Key lifecycle events
//! - CC6.3: Authorization - Access grant/deny events
//! - CC6.5: Data Access - Read/write/delete events
//! - CC6.6: Configuration - Config change tracking
//! - CC7.1: System Availability - Health check events
//! - CC7.2: Security Incidents - Threat detection events

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Audit event types (SOC 2 compliant categorization)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuditEventType {
    // Authentication Events (CC6.1)
    ApiKeyCreated,
    ApiKeyValidated,
    ApiKeyValidationFailed,
    ApiKeyRevoked,
    ApiKeyRotated,
    ApiKeyExpired,

    // Authorization Events (CC6.3)
    AccessGranted,
    AccessDenied,
    ScopeEscalationAttempt,

    // Data Access Events (CC6.5)
    DataRead,
    DataWrite,
    DataDelete,
    DataExport,

    // Configuration Events (CC6.6)
    ConfigChanged,
    IpConfigUpdated,
    RateLimitConfigUpdated,
    TenantCreated,
    TenantSuspended,

    // Security Events (CC7.2)
    BruteForceDetected,
    IpBlocked,
    AnomalousActivity,
    SecurityIncident,

    // Rate Limiting Events
    RateLimitExceeded,
    RateLimitThrottled,

    // System Events (CC7.1)
    SystemStartup,
    SystemShutdown,
    HealthCheckFailed,
    BackupCompleted,

    // Key Rotation Events (CC6.1)
    KeyRotationStarted,
    KeyRotationCompleted,
    KeyRotationFailed,
}

/// Actor performing the action
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "actor_type", rename_all = "snake_case")]
pub enum AuditActor {
    /// System-initiated action
    System,
    /// Action via API key
    ApiKey(Uuid),
    /// Action by admin user
    Admin(Uuid),
    /// Action by service account
    ServiceAccount(String),
    /// Unknown/anonymous actor
    Anonymous,
}

/// Audit event outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AuditOutcome {
    Success,
    Failure {
        error_code: String,
        error_message: String,
    },
    Partial {
        completed: u32,
        total: u32,
    },
}

/// Complete audit event (SOC 2 compliant schema)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event identifier
    pub event_id: Uuid,
    /// Event type
    pub event_type: AuditEventType,
    /// Tenant context (if applicable)
    pub tenant_id: Option<Uuid>,
    /// API key context (if applicable)
    pub key_id: Option<Uuid>,
    /// Actor performing action
    pub actor: AuditActor,
    /// Event timestamp (UTC)
    pub timestamp: DateTime<Utc>,
    /// Event details (structured JSON)
    pub details: serde_json::Value,
    /// Client IP address
    pub ip_address: Option<IpAddr>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Request ID for correlation
    pub request_id: Option<Uuid>,
    /// Session ID for correlation
    pub session_id: Option<Uuid>,
    /// Resource being accessed
    pub resource: Option<String>,
    /// Action outcome
    pub outcome: AuditOutcome,
    /// Hash of previous event (chain integrity)
    pub previous_hash: Option<String>,
    /// Hash of this event
    pub event_hash: String,
}

impl AuditEvent {
    /// Create a new audit event
    pub fn new(event_type: AuditEventType, actor: AuditActor) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            event_type,
            tenant_id: None,
            key_id: None,
            actor,
            timestamp: Utc::now(),
            details: serde_json::json!({}),
            ip_address: None,
            user_agent: None,
            request_id: None,
            session_id: None,
            resource: None,
            outcome: AuditOutcome::Success,
            previous_hash: None,
            event_hash: String::new(),
        }
    }

    /// Set tenant ID
    pub fn with_tenant(mut self, tenant_id: Uuid) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }

    /// Set key ID
    pub fn with_key(mut self, key_id: Uuid) -> Self {
        self.key_id = Some(key_id);
        self
    }

    /// Set event details
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = details;
        self
    }

    /// Set IP address
    pub fn with_ip(mut self, ip: IpAddr) -> Self {
        self.ip_address = Some(ip);
        self
    }

    /// Set user agent
    pub fn with_user_agent(mut self, user_agent: String) -> Self {
        self.user_agent = Some(user_agent);
        self
    }

    /// Set request ID for correlation
    pub fn with_request_id(mut self, request_id: Uuid) -> Self {
        self.request_id = Some(request_id);
        self
    }

    /// Set outcome
    pub fn with_outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Set resource
    pub fn with_resource(mut self, resource: String) -> Self {
        self.resource = Some(resource);
        self
    }
}

/// Audit logging configuration
#[derive(Debug, Clone, Deserialize)]
pub struct AuditConfig {
    /// Retention period in days
    pub retention_days: u32,
    /// Batch size for writes
    pub batch_size: usize,
    /// Flush interval (seconds)
    pub flush_interval_secs: u32,
    /// Enable real-time SIEM forwarding
    pub enable_siem: bool,
    /// SIEM endpoint
    pub siem_endpoint: Option<String>,
    /// Enable cryptographic chaining
    pub enable_chaining: bool,
    /// Sensitive fields to redact
    pub sensitive_fields: Vec<String>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            retention_days: 365,
            batch_size: 100,
            flush_interval_secs: 5,
            enable_siem: false,
            siem_endpoint: None,
            enable_chaining: true,
            sensitive_fields: vec![
                "password".to_string(),
                "secret".to_string(),
                "token".to_string(),
                "api_key".to_string(),
                "credential".to_string(),
                "authorization".to_string(),
                "cookie".to_string(),
                "session".to_string(),
                "private_key".to_string(),
            ],
        }
    }
}

/// Audit storage trait
#[async_trait::async_trait]
pub trait AuditStore: Send + Sync {
    /// Store audit events (batch)
    async fn store(&self, events: &[AuditEvent]) -> Result<(), AuditError>;
    /// Query events with filters
    async fn query(
        &self,
        filter: AuditQueryFilter,
        pagination: Pagination,
    ) -> Result<AuditQueryResult, AuditError>;
    /// Get events in time range
    async fn get_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<AuditEvent>, AuditError>;
}

/// SIEM forwarder trait
#[async_trait::async_trait]
pub trait SiemForwarder: Send + Sync {
    /// Forward event to SIEM
    async fn forward(&self, event: &AuditEvent) -> Result<(), AuditError>;
}

/// SOC 2 Compliant Audit Logger
pub struct AuditLogger {
    /// Primary storage (append-only)
    primary_store: Arc<dyn AuditStore>,
    /// Secondary storage (redundancy)
    secondary_store: Option<Arc<dyn AuditStore>>,
    /// SIEM forwarder
    siem_forwarder: Option<Arc<dyn SiemForwarder>>,
    /// Last event hash (for chaining)
    last_hash: RwLock<Option<String>>,
    /// Event buffer for batching
    buffer: RwLock<Vec<AuditEvent>>,
    /// Configuration
    config: AuditConfig,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(primary_store: Arc<dyn AuditStore>, config: AuditConfig) -> Self {
        Self {
            primary_store,
            secondary_store: None,
            siem_forwarder: None,
            last_hash: RwLock::new(None),
            buffer: RwLock::new(Vec::new()),
            config,
        }
    }

    /// Set secondary store for redundancy
    pub fn with_secondary_store(mut self, store: Arc<dyn AuditStore>) -> Self {
        self.secondary_store = Some(store);
        self
    }

    /// Set SIEM forwarder
    pub fn with_siem(mut self, forwarder: Arc<dyn SiemForwarder>) -> Self {
        self.siem_forwarder = Some(forwarder);
        self
    }

    /// Log an audit event
    pub async fn log(&self, mut event: AuditEvent) -> Result<Uuid, AuditError> {
        // Add chaining hash
        if self.config.enable_chaining {
            let last_hash = self.last_hash.read().await;
            event.previous_hash = last_hash.clone();
            event.event_hash = self.compute_hash(&event);
        }

        // Redact sensitive fields
        event.details = self.redact_sensitive_fields(event.details);

        let event_id = event.event_id;

        // Add to buffer
        {
            let mut buffer = self.buffer.write().await;
            buffer.push(event.clone());

            // Flush if buffer is full
            if buffer.len() >= self.config.batch_size {
                self.flush_buffer_internal(&mut buffer).await?;
            }
        }

        // Forward to SIEM in real-time for security events
        if self.is_security_event(&event.event_type) {
            self.forward_to_siem(&event).await?;
        }

        // Update last hash
        if self.config.enable_chaining {
            let mut last_hash = self.last_hash.write().await;
            *last_hash = Some(event.event_hash.clone());
        }

        Ok(event_id)
    }

    /// Flush buffered events
    pub async fn flush(&self) -> Result<(), AuditError> {
        let mut buffer = self.buffer.write().await;
        self.flush_buffer_internal(&mut buffer).await
    }

    async fn flush_buffer_internal(&self, buffer: &mut Vec<AuditEvent>) -> Result<(), AuditError> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Store in primary
        self.primary_store.store(buffer).await?;

        // Store in secondary if configured
        if let Some(secondary) = &self.secondary_store {
            // Best effort - don't fail if secondary is unavailable
            let _ = secondary.store(buffer).await;
        }

        buffer.clear();
        Ok(())
    }

    /// Compute hash for event (chain integrity)
    fn compute_hash(&self, event: &AuditEvent) -> String {
        let mut hasher = Sha256::new();

        // Include all relevant fields
        hasher.update(event.event_id.as_bytes());
        hasher.update(event.timestamp.timestamp().to_le_bytes());
        hasher.update(serde_json::to_vec(&event.event_type).unwrap_or_default());
        hasher.update(serde_json::to_vec(&event.actor).unwrap_or_default());
        if let Some(prev) = &event.previous_hash {
            hasher.update(prev.as_bytes());
        }
        hasher.update(event.details.to_string().as_bytes());

        hex::encode(hasher.finalize())
    }

    /// Check if event type requires real-time SIEM forwarding
    fn is_security_event(&self, event_type: &AuditEventType) -> bool {
        matches!(
            event_type,
            AuditEventType::ApiKeyValidationFailed
                | AuditEventType::AccessDenied
                | AuditEventType::ScopeEscalationAttempt
                | AuditEventType::BruteForceDetected
                | AuditEventType::IpBlocked
                | AuditEventType::AnomalousActivity
                | AuditEventType::SecurityIncident
        )
    }

    /// Forward event to SIEM
    async fn forward_to_siem(&self, event: &AuditEvent) -> Result<(), AuditError> {
        if let Some(forwarder) = &self.siem_forwarder {
            forwarder.forward(event).await?;
        }
        Ok(())
    }

    /// Redact sensitive fields from event details
    fn redact_sensitive_fields(&self, mut details: serde_json::Value) -> serde_json::Value {
        if let Some(obj) = details.as_object_mut() {
            for key in obj.keys().cloned().collect::<Vec<_>>() {
                let key_lower = key.to_lowercase();
                for sensitive in &self.config.sensitive_fields {
                    if key_lower.contains(sensitive) {
                        obj.insert(key.clone(), serde_json::json!("[REDACTED]"));
                        break;
                    }
                }
            }
        }
        details
    }

    /// Query audit logs with filters
    pub async fn query(
        &self,
        filter: AuditQueryFilter,
        pagination: Pagination,
    ) -> Result<AuditQueryResult, AuditError> {
        self.primary_store.query(filter, pagination).await
    }

    /// Verify audit log integrity
    pub async fn verify_integrity(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<IntegrityReport, AuditError> {
        let events = self.primary_store.get_range(start, end).await?;

        let mut valid_count = 0u64;
        let mut invalid_events = Vec::new();
        let mut last_hash: Option<String> = None;

        for event in &events {
            // Verify hash chain
            if event.previous_hash != last_hash {
                invalid_events.push(IntegrityViolation {
                    event_id: event.event_id,
                    violation_type: ViolationType::ChainBroken,
                    details: "Previous hash mismatch".to_string(),
                });
            }

            // Verify event hash
            let computed_hash = self.compute_hash(event);
            if computed_hash != event.event_hash {
                invalid_events.push(IntegrityViolation {
                    event_id: event.event_id,
                    violation_type: ViolationType::HashMismatch,
                    details: "Event hash verification failed".to_string(),
                });
            } else {
                valid_count += 1;
            }

            last_hash = Some(event.event_hash.clone());
        }

        Ok(IntegrityReport {
            start,
            end,
            total_events: valid_count + invalid_events.len() as u64,
            valid_events: valid_count,
            violations: invalid_events,
            verified_at: Utc::now(),
        })
    }
}

/// Audit query filter
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditQueryFilter {
    pub tenant_id: Option<Uuid>,
    pub event_types: Option<Vec<AuditEventType>>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub ip_address: Option<IpAddr>,
    pub key_id: Option<Uuid>,
}

/// Pagination parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub offset: u64,
    pub limit: u64,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            offset: 0,
            limit: 100,
        }
    }
}

/// Query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditQueryResult {
    pub events: Vec<AuditEvent>,
    pub total_count: u64,
    pub has_more: bool,
}

/// Integrity verification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub total_events: u64,
    pub valid_events: u64,
    pub violations: Vec<IntegrityViolation>,
    pub verified_at: DateTime<Utc>,
}

/// Integrity violation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityViolation {
    pub event_id: Uuid,
    pub violation_type: ViolationType,
    pub details: String,
}

/// Types of integrity violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    ChainBroken,
    HashMismatch,
    TimestampAnomaly,
    MissingEvent,
}

/// Audit errors
#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("SIEM forwarding error: {0}")]
    Siem(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Integrity verification failed: {0}")]
    Integrity(String),
}

/// In-memory audit store for testing
pub struct InMemoryAuditStore {
    events: RwLock<Vec<AuditEvent>>,
}

impl InMemoryAuditStore {
    pub fn new() -> Self {
        Self {
            events: RwLock::new(Vec::new()),
        }
    }
}

impl Default for InMemoryAuditStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl AuditStore for InMemoryAuditStore {
    async fn store(&self, events: &[AuditEvent]) -> Result<(), AuditError> {
        let mut store = self.events.write().await;
        store.extend(events.iter().cloned());
        Ok(())
    }

    async fn query(
        &self,
        filter: AuditQueryFilter,
        pagination: Pagination,
    ) -> Result<AuditQueryResult, AuditError> {
        let store = self.events.read().await;
        let filtered: Vec<_> = store
            .iter()
            .filter(|e| {
                if let Some(tenant_id) = filter.tenant_id {
                    if e.tenant_id != Some(tenant_id) {
                        return false;
                    }
                }
                if let Some(ref types) = filter.event_types {
                    if !types.contains(&e.event_type) {
                        return false;
                    }
                }
                if let Some(start) = filter.start_time {
                    if e.timestamp < start {
                        return false;
                    }
                }
                if let Some(end) = filter.end_time {
                    if e.timestamp > end {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        let total_count = filtered.len() as u64;
        let events: Vec<_> = filtered
            .into_iter()
            .skip(pagination.offset as usize)
            .take(pagination.limit as usize)
            .collect();
        let has_more = pagination.offset + events.len() as u64 < total_count;

        Ok(AuditQueryResult {
            events,
            total_count,
            has_more,
        })
    }

    async fn get_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<AuditEvent>, AuditError> {
        let store = self.events.read().await;
        Ok(store
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_event_creation() {
        let event = AuditEvent::new(AuditEventType::ApiKeyCreated, AuditActor::System)
            .with_tenant(Uuid::new_v4())
            .with_details(serde_json::json!({"test": "value"}));

        assert!(matches!(event.event_type, AuditEventType::ApiKeyCreated));
        assert!(event.tenant_id.is_some());
    }

    #[tokio::test]
    async fn test_audit_logger_redaction() {
        let store = Arc::new(InMemoryAuditStore::new());
        let config = AuditConfig::default();
        let logger = AuditLogger::new(store, config);

        let details = serde_json::json!({
            "user": "test",
            "password": "secret123",
            "api_key": "key123"
        });

        let redacted = logger.redact_sensitive_fields(details);

        assert_eq!(redacted["user"], "test");
        assert_eq!(redacted["password"], "[REDACTED]");
        assert_eq!(redacted["api_key"], "[REDACTED]");
    }

    #[tokio::test]
    async fn test_in_memory_store() {
        let store = InMemoryAuditStore::new();

        let event = AuditEvent::new(AuditEventType::ApiKeyCreated, AuditActor::System);

        store.store(&[event.clone()]).await.unwrap();

        let result = store
            .query(AuditQueryFilter::default(), Pagination::default())
            .await
            .unwrap();

        assert_eq!(result.total_count, 1);
        assert_eq!(result.events[0].event_id, event.event_id);
    }
}
