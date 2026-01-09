//! Brute Force Protection
//!
//! Implements adaptive defense against brute force attacks with:
//! - Progressive delays (exponential backoff)
//! - IP blocking after threshold
//! - Key prefix enumeration detection
//! - CAPTCHA integration support
//! - Threat intelligence integration

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::security::audit::{AuditActor, AuditEvent, AuditEventType, AuditLogger, AuditOutcome};

/// Brute force protection configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BruteForceConfig {
    /// Failed attempts before first delay
    pub attempts_before_delay: u32,
    /// Initial delay after threshold (seconds)
    pub initial_delay_secs: u32,
    /// Delay multiplier for each subsequent failure
    pub delay_multiplier: f32,
    /// Maximum delay (seconds)
    pub max_delay_secs: u32,
    /// Failed attempts before IP block
    pub attempts_before_block: u32,
    /// IP block duration (seconds)
    pub block_duration_secs: u32,
    /// Window for counting failures (seconds)
    pub failure_window_secs: u32,
    /// Enable CAPTCHA after this many failures
    pub captcha_threshold: u32,
    /// Reset failures after this period of no attempts (seconds)
    pub reset_after_secs: u32,
}

impl Default for BruteForceConfig {
    fn default() -> Self {
        Self {
            attempts_before_delay: 3,
            initial_delay_secs: 1,
            delay_multiplier: 2.0,
            max_delay_secs: 30,
            attempts_before_block: 10,
            block_duration_secs: 3600, // 1 hour
            failure_window_secs: 300,  // 5 minutes
            captcha_threshold: 5,
            reset_after_secs: 3600, // 1 hour
        }
    }
}

/// Per-IP tracking state
#[derive(Debug, Clone)]
struct IpState {
    /// Failed attempts in current window
    failed_attempts: u32,
    /// Timestamp of first failure in window
    window_start: Instant,
    /// Timestamp of last attempt
    last_attempt: Instant,
    /// Current delay level
    delay_level: u32,
    /// Block expiration (if blocked)
    blocked_until: Option<Instant>,
    /// Suspicious patterns detected
    suspicious_patterns: Vec<SuspiciousPattern>,
}

/// Detected suspicious patterns
#[derive(Debug, Clone)]
pub enum SuspiciousPattern {
    /// Same prefix tried multiple times
    PrefixEnumeration { prefix: String, count: u32 },
    /// Sequential key attempts
    SequentialAttempts,
    /// Known attack signature
    AttackSignature(String),
    /// Geographic anomaly
    GeoAnomaly { expected: String, actual: String },
}

/// Brute force protection action
#[derive(Debug, Clone)]
pub enum BruteForceAction {
    /// Allow request
    Allow,
    /// Require delay before processing
    Delay { seconds: u32 },
    /// Require CAPTCHA verification
    RequireCaptcha { challenge_id: String },
    /// Block request
    Block {
        reason: BlockReason,
        duration_secs: u32,
    },
}

/// Reasons for blocking
#[derive(Debug, Clone)]
pub enum BlockReason {
    ExcessiveFailures,
    AttackPatternDetected,
    ThreatIntelMatch,
    GeoAnomaly,
}

/// Per-prefix tracking state
#[derive(Debug, Clone)]
struct PrefixState {
    ips_attempting: HashMap<IpAddr, u32>,
    last_attempt: Instant,
}

/// Threat intelligence provider trait
#[async_trait::async_trait]
pub trait ThreatIntelProvider: Send + Sync {
    async fn check_ip(&self, ip: IpAddr) -> Option<ThreatInfo>;
}

/// Threat intelligence result
#[derive(Debug, Clone)]
pub struct ThreatInfo {
    pub category: String,
    pub risk_score: u8,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

/// Alert service trait
#[async_trait::async_trait]
pub trait AlertService: Send + Sync {
    async fn send_alert(&self, alert: Alert) -> Result<(), String>;
}

/// Security alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub severity: AlertSeverity,
    pub title: String,
    pub message: String,
    pub metadata: serde_json::Value,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Brute Force Protection Service
pub struct BruteForceProtection {
    config: BruteForceConfig,
    /// Per-IP state
    ip_state: Arc<RwLock<HashMap<IpAddr, IpState>>>,
    /// Per-key-prefix state (for enumeration detection)
    prefix_state: Arc<RwLock<HashMap<String, PrefixState>>>,
    /// Threat intelligence integration (optional)
    threat_intel: Option<Arc<dyn ThreatIntelProvider>>,
    /// Audit logger
    audit: Arc<AuditLogger>,
    /// Alert service (optional)
    alerter: Option<Arc<dyn AlertService>>,
}

impl BruteForceProtection {
    /// Create new brute force protection
    pub fn new(config: BruteForceConfig, audit: Arc<AuditLogger>) -> Self {
        Self {
            config,
            ip_state: Arc::new(RwLock::new(HashMap::new())),
            prefix_state: Arc::new(RwLock::new(HashMap::new())),
            threat_intel: None,
            audit,
            alerter: None,
        }
    }

    /// Set threat intelligence provider
    pub fn with_threat_intel(mut self, provider: Arc<dyn ThreatIntelProvider>) -> Self {
        self.threat_intel = Some(provider);
        self
    }

    /// Set alert service
    pub fn with_alerter(mut self, alerter: Arc<dyn AlertService>) -> Self {
        self.alerter = Some(alerter);
        self
    }

    /// Check if request should be allowed (call BEFORE key validation)
    pub async fn check_request(&self, ip: IpAddr, key_prefix: Option<&str>) -> BruteForceAction {
        // Check IP block first
        if let Some(action) = self.check_ip_block(ip).await {
            return action;
        }

        // Check threat intelligence
        if let Some(ref threat_intel) = self.threat_intel {
            if let Some(threat) = threat_intel.check_ip(ip).await {
                self.block_ip(ip, BlockReason::ThreatIntelMatch, self.config.block_duration_secs * 24)
                    .await;
                return BruteForceAction::Block {
                    duration_secs: self.config.block_duration_secs * 24,
                    reason: BlockReason::ThreatIntelMatch,
                };
            }
        }

        // Check enumeration patterns
        if let Some(prefix) = key_prefix {
            if self.detect_enumeration(ip, prefix).await {
                self.block_ip(ip, BlockReason::AttackPatternDetected, self.config.block_duration_secs)
                    .await;
                return BruteForceAction::Block {
                    duration_secs: self.config.block_duration_secs,
                    reason: BlockReason::AttackPatternDetected,
                };
            }
        }

        // Check failure count and apply progressive penalties
        self.apply_progressive_penalty(ip).await
    }

    /// Record failed authentication attempt
    pub async fn record_failure(&self, ip: IpAddr, key_prefix: Option<&str>, error_type: &str) {
        let now = Instant::now();

        let mut state = self.ip_state.write().await;
        let ip_state = state.entry(ip).or_insert_with(|| IpState {
            failed_attempts: 0,
            window_start: now,
            last_attempt: now,
            delay_level: 0,
            blocked_until: None,
            suspicious_patterns: Vec::new(),
        });

        // Reset window if expired
        if now.duration_since(ip_state.window_start)
            > Duration::from_secs(self.config.failure_window_secs as u64)
        {
            ip_state.failed_attempts = 0;
            ip_state.window_start = now;
        }

        ip_state.failed_attempts += 1;
        ip_state.last_attempt = now;

        // Increase delay level
        if ip_state.failed_attempts >= self.config.attempts_before_delay {
            ip_state.delay_level = ip_state.delay_level.saturating_add(1);
        }

        let failed_attempts = ip_state.failed_attempts;
        let should_block = failed_attempts >= self.config.attempts_before_block;

        // Check for block threshold
        if should_block {
            ip_state.blocked_until =
                Some(now + Duration::from_secs(self.config.block_duration_secs as u64));

            // Drop lock before async operations
            drop(state);

            // Audit log
            let _ = self
                .audit
                .log(
                    AuditEvent::new(AuditEventType::BruteForceDetected, AuditActor::Anonymous)
                        .with_details(serde_json::json!({
                            "ip": ip.to_string(),
                            "failed_attempts": failed_attempts,
                            "block_duration_secs": self.config.block_duration_secs,
                            "key_prefix": key_prefix,
                            "error_type": error_type,
                        }))
                        .with_ip(ip)
                        .with_outcome(AuditOutcome::Failure {
                            error_code: "BRUTE_FORCE".to_string(),
                            error_message: format!("IP blocked after {} failures", failed_attempts),
                        }),
                )
                .await;

            // Alert security team
            if let Some(ref alerter) = self.alerter {
                let _ = alerter
                    .send_alert(Alert {
                        severity: AlertSeverity::High,
                        title: "Brute Force Attack Detected".to_string(),
                        message: format!(
                            "IP {} blocked after {} failed attempts",
                            ip, failed_attempts
                        ),
                        metadata: serde_json::json!({
                            "ip": ip.to_string(),
                            "attempts": failed_attempts,
                        }),
                    })
                    .await;
            }
        } else {
            drop(state);
        }

        // Track prefix attempts for enumeration detection
        if let Some(prefix) = key_prefix {
            self.record_prefix_attempt(ip, prefix).await;
        }
    }

    /// Record successful authentication (reduces penalties)
    pub async fn record_success(&self, ip: IpAddr) {
        let mut state = self.ip_state.write().await;
        if let Some(ip_state) = state.get_mut(&ip) {
            // Don't fully reset - reduce delay level
            ip_state.delay_level = ip_state.delay_level.saturating_sub(2);
            ip_state.failed_attempts = ip_state.failed_attempts.saturating_sub(1);
        }
    }

    async fn check_ip_block(&self, ip: IpAddr) -> Option<BruteForceAction> {
        let state = self.ip_state.read().await;
        if let Some(ip_state) = state.get(&ip) {
            if let Some(blocked_until) = ip_state.blocked_until {
                if Instant::now() < blocked_until {
                    let remaining = blocked_until - Instant::now();
                    return Some(BruteForceAction::Block {
                        duration_secs: remaining.as_secs() as u32,
                        reason: BlockReason::ExcessiveFailures,
                    });
                }
            }
        }
        None
    }

    async fn apply_progressive_penalty(&self, ip: IpAddr) -> BruteForceAction {
        let state = self.ip_state.read().await;

        if let Some(ip_state) = state.get(&ip) {
            // Check if CAPTCHA required
            if ip_state.failed_attempts >= self.config.captcha_threshold
                && ip_state.failed_attempts < self.config.attempts_before_block
            {
                return BruteForceAction::RequireCaptcha {
                    challenge_id: Uuid::new_v4().to_string(),
                };
            }

            // Check if delay required
            if ip_state.delay_level > 0 {
                let delay = self.calculate_delay(ip_state.delay_level);
                return BruteForceAction::Delay { seconds: delay };
            }
        }

        BruteForceAction::Allow
    }

    fn calculate_delay(&self, level: u32) -> u32 {
        let delay = self.config.initial_delay_secs as f32
            * self.config.delay_multiplier.powi(level as i32 - 1);
        (delay as u32).min(self.config.max_delay_secs)
    }

    async fn detect_enumeration(&self, ip: IpAddr, prefix: &str) -> bool {
        let mut state = self.prefix_state.write().await;
        let prefix_state = state.entry(prefix.to_string()).or_insert_with(|| PrefixState {
            ips_attempting: HashMap::new(),
            last_attempt: Instant::now(),
        });

        // Count this IP's attempts on this prefix
        let count = prefix_state.ips_attempting.entry(ip).or_insert(0);
        *count += 1;
        prefix_state.last_attempt = Instant::now();

        // Detect enumeration: many IPs trying same prefix, or one IP trying many
        let unique_ips = prefix_state.ips_attempting.len();
        let this_ip_attempts = *count;

        // Threshold: 5+ IPs trying same prefix, or 10+ attempts from one IP
        unique_ips >= 5 || this_ip_attempts >= 10
    }

    async fn record_prefix_attempt(&self, ip: IpAddr, prefix: &str) {
        let mut state = self.prefix_state.write().await;
        let prefix_state = state.entry(prefix.to_string()).or_insert_with(|| PrefixState {
            ips_attempting: HashMap::new(),
            last_attempt: Instant::now(),
        });

        *prefix_state.ips_attempting.entry(ip).or_insert(0) += 1;
    }

    async fn block_ip(&self, ip: IpAddr, reason: BlockReason, duration_secs: u32) {
        let mut state = self.ip_state.write().await;
        let ip_state = state.entry(ip).or_insert_with(|| IpState {
            failed_attempts: 0,
            window_start: Instant::now(),
            last_attempt: Instant::now(),
            delay_level: 0,
            blocked_until: None,
            suspicious_patterns: Vec::new(),
        });

        ip_state.blocked_until =
            Some(Instant::now() + Duration::from_secs(duration_secs as u64));

        // Drop lock before async audit
        drop(state);

        let _ = self
            .audit
            .log(
                AuditEvent::new(AuditEventType::IpBlocked, AuditActor::System)
                    .with_details(serde_json::json!({
                        "ip": ip.to_string(),
                        "reason": format!("{:?}", reason),
                        "duration_secs": duration_secs,
                    }))
                    .with_ip(ip)
                    .with_outcome(AuditOutcome::Success),
            )
            .await;
    }

    /// Unblock an IP address (admin action)
    pub async fn unblock_ip(&self, ip: IpAddr) {
        let mut state = self.ip_state.write().await;
        if let Some(ip_state) = state.get_mut(&ip) {
            ip_state.blocked_until = None;
            ip_state.failed_attempts = 0;
            ip_state.delay_level = 0;
        }
    }

    /// Get current state for an IP (for monitoring)
    pub async fn get_ip_state(&self, ip: IpAddr) -> Option<IpStateInfo> {
        let state = self.ip_state.read().await;
        state.get(&ip).map(|s| IpStateInfo {
            failed_attempts: s.failed_attempts,
            delay_level: s.delay_level,
            is_blocked: s.blocked_until.map(|b| Instant::now() < b).unwrap_or(false),
            block_remaining_secs: s.blocked_until.and_then(|b| {
                if Instant::now() < b {
                    Some((b - Instant::now()).as_secs() as u32)
                } else {
                    None
                }
            }),
        })
    }
}

/// IP state information for monitoring
#[derive(Debug, Clone)]
pub struct IpStateInfo {
    pub failed_attempts: u32,
    pub delay_level: u32,
    pub is_blocked: bool,
    pub block_remaining_secs: Option<u32>,
}

/// No-op threat intelligence provider for testing
pub struct NoOpThreatIntel;

#[async_trait::async_trait]
impl ThreatIntelProvider for NoOpThreatIntel {
    async fn check_ip(&self, _ip: IpAddr) -> Option<ThreatInfo> {
        None
    }
}

/// No-op alert service for testing
pub struct NoOpAlertService;

#[async_trait::async_trait]
impl AlertService for NoOpAlertService {
    async fn send_alert(&self, _alert: Alert) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::audit::{AuditConfig, InMemoryAuditStore};

    fn create_test_protection() -> BruteForceProtection {
        let audit_store = Arc::new(InMemoryAuditStore::new());
        let audit = Arc::new(AuditLogger::new(audit_store, AuditConfig::default()));
        BruteForceProtection::new(BruteForceConfig::default(), audit)
    }

    #[tokio::test]
    async fn test_allows_initial_request() {
        let protection = create_test_protection();
        let ip = "127.0.0.1".parse().unwrap();

        let action = protection.check_request(ip, None).await;
        assert!(matches!(action, BruteForceAction::Allow));
    }

    #[tokio::test]
    async fn test_delays_after_threshold() {
        let mut config = BruteForceConfig::default();
        config.attempts_before_delay = 2;

        let audit_store = Arc::new(InMemoryAuditStore::new());
        let audit = Arc::new(AuditLogger::new(audit_store, AuditConfig::default()));
        let protection = BruteForceProtection::new(config, audit);

        let ip = "127.0.0.1".parse().unwrap();

        // Record failures
        protection.record_failure(ip, None, "invalid_key").await;
        protection.record_failure(ip, None, "invalid_key").await;
        protection.record_failure(ip, None, "invalid_key").await;

        let action = protection.check_request(ip, None).await;
        assert!(matches!(action, BruteForceAction::Delay { .. }));
    }

    #[tokio::test]
    async fn test_blocks_after_threshold() {
        let mut config = BruteForceConfig::default();
        config.attempts_before_block = 3;

        let audit_store = Arc::new(InMemoryAuditStore::new());
        let audit = Arc::new(AuditLogger::new(audit_store, AuditConfig::default()));
        let protection = BruteForceProtection::new(config, audit);

        let ip = "127.0.0.1".parse().unwrap();

        // Record failures
        for _ in 0..4 {
            protection.record_failure(ip, None, "invalid_key").await;
        }

        let action = protection.check_request(ip, None).await;
        assert!(matches!(action, BruteForceAction::Block { .. }));
    }

    #[tokio::test]
    async fn test_unblock_ip() {
        let protection = create_test_protection();
        let ip = "127.0.0.1".parse().unwrap();

        // Block the IP
        protection
            .block_ip(ip, BlockReason::ExcessiveFailures, 3600)
            .await;

        // Verify blocked
        let action = protection.check_request(ip, None).await;
        assert!(matches!(action, BruteForceAction::Block { .. }));

        // Unblock
        protection.unblock_ip(ip).await;

        // Verify unblocked
        let action = protection.check_request(ip, None).await;
        assert!(matches!(action, BruteForceAction::Allow));
    }

    #[test]
    fn test_calculate_delay() {
        let config = BruteForceConfig {
            initial_delay_secs: 1,
            delay_multiplier: 2.0,
            max_delay_secs: 30,
            ..Default::default()
        };

        let audit_store = Arc::new(crate::security::audit::InMemoryAuditStore::new());
        let audit = Arc::new(AuditLogger::new(audit_store, AuditConfig::default()));
        let protection = BruteForceProtection::new(config, audit);

        assert_eq!(protection.calculate_delay(1), 1);
        assert_eq!(protection.calculate_delay(2), 2);
        assert_eq!(protection.calculate_delay(3), 4);
        assert_eq!(protection.calculate_delay(4), 8);
        assert_eq!(protection.calculate_delay(10), 30); // Capped at max
    }
}
