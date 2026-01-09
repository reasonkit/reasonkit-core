//! Multi-Tier Rate Limiting
//!
//! Implements enterprise-grade rate limiting with:
//! - Token bucket algorithm for burst handling
//! - Sliding window counters for accurate rate tracking
//! - Per-tenant tier-based limits
//! - Global capacity protection
//! - Distributed state support (Redis/DragonflyDB)

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Rate limit tier configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RateLimitConfig {
    pub tiers: HashMap<String, TierLimits>,
    pub global_limits: GlobalLimits,
    pub ip_limits: IpLimits,
}

/// Per-tier rate limits
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TierLimits {
    /// Requests per second
    pub rps: u32,
    /// Requests per minute
    pub rpm: u32,
    /// Requests per hour
    pub rph: u32,
    /// Requests per day
    pub rpd: u32,
    /// Concurrent request limit
    pub concurrent: u32,
    /// Burst allowance (tokens above steady state)
    pub burst_size: u32,
    /// Request timeout (seconds)
    pub timeout_secs: u32,
}

/// Global system limits
#[derive(Debug, Clone, serde::Deserialize)]
pub struct GlobalLimits {
    /// Total system RPS cap
    pub max_global_rps: u32,
    /// Per-endpoint limits
    pub endpoint_limits: HashMap<String, u32>,
}

/// IP-based limits
#[derive(Debug, Clone, serde::Deserialize)]
pub struct IpLimits {
    /// Unauthenticated IP RPS
    pub anonymous_rps: u32,
    /// Authenticated IP RPS multiplier
    pub authenticated_multiplier: f32,
    /// IP blocklist TTL (seconds)
    pub block_ttl_secs: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        let mut tiers = HashMap::new();

        // Free tier
        tiers.insert(
            "free".to_string(),
            TierLimits {
                rps: 1,
                rpm: 20,
                rph: 100,
                rpd: 1000,
                concurrent: 2,
                burst_size: 5,
                timeout_secs: 30,
            },
        );

        // Starter tier
        tiers.insert(
            "starter".to_string(),
            TierLimits {
                rps: 10,
                rpm: 300,
                rph: 5000,
                rpd: 50000,
                concurrent: 10,
                burst_size: 20,
                timeout_secs: 60,
            },
        );

        // Professional tier
        tiers.insert(
            "professional".to_string(),
            TierLimits {
                rps: 50,
                rpm: 2000,
                rph: 50000,
                rpd: 500000,
                concurrent: 50,
                burst_size: 100,
                timeout_secs: 120,
            },
        );

        // Enterprise tier
        tiers.insert(
            "enterprise".to_string(),
            TierLimits {
                rps: 500,
                rpm: 20000,
                rph: 500000,
                rpd: 5000000,
                concurrent: 500,
                burst_size: 1000,
                timeout_secs: 300,
            },
        );

        Self {
            tiers,
            global_limits: GlobalLimits {
                max_global_rps: 10000,
                endpoint_limits: HashMap::new(),
            },
            ip_limits: IpLimits {
                anonymous_rps: 1,
                authenticated_multiplier: 10.0,
                block_ttl_secs: 3600,
            },
        }
    }
}

/// Rate limit decision
#[derive(Debug, Clone)]
pub enum RateLimitDecision {
    /// Request allowed
    Allow {
        remaining: RateLimitRemaining,
        reset_at: chrono::DateTime<chrono::Utc>,
    },
    /// Request throttled (soft limit)
    Throttle {
        retry_after_secs: u32,
        reason: ThrottleReason,
    },
    /// Request blocked (hard limit)
    Block {
        reason: BlockReason,
        block_duration_secs: u32,
    },
}

/// Remaining request counts
#[derive(Debug, Clone)]
pub struct RateLimitRemaining {
    pub second: u32,
    pub minute: u32,
    pub hour: u32,
    pub day: u32,
}

/// Reasons for throttling
#[derive(Debug, Clone)]
pub enum ThrottleReason {
    TenantRpmExceeded,
    GlobalCapacity,
    EndpointLimit,
    ConcurrentLimit,
}

/// Reasons for blocking
#[derive(Debug, Clone)]
pub enum BlockReason {
    DailyQuotaExhausted,
    BruteForceDetected,
    IpBlocked,
    TenantSuspended,
}

/// Per-tenant state
#[derive(Debug)]
struct TenantRateState {
    tier: String,
    /// Sliding window counters [second, minute, hour, day]
    windows: [SlidingWindow; 4],
    /// Current concurrent requests
    concurrent: u32,
    /// Token bucket state
    tokens: f64,
    last_refill: Instant,
}

impl TenantRateState {
    fn new(tier: String) -> Self {
        Self {
            tier,
            windows: [
                SlidingWindow::new(Duration::from_secs(1)),
                SlidingWindow::new(Duration::from_secs(60)),
                SlidingWindow::new(Duration::from_secs(3600)),
                SlidingWindow::new(Duration::from_secs(86400)),
            ],
            concurrent: 0,
            tokens: 0.0,
            last_refill: Instant::now(),
        }
    }
}

/// Sliding window counter
#[derive(Debug)]
struct SlidingWindow {
    counts: Vec<(Instant, u32)>,
    window_size: Duration,
}

impl SlidingWindow {
    fn new(window_size: Duration) -> Self {
        Self {
            counts: Vec::new(),
            window_size,
        }
    }

    fn count_in_window(&self, now: Instant) -> u32 {
        self.counts
            .iter()
            .filter(|(t, _)| now.duration_since(*t) < self.window_size)
            .map(|(_, c)| c)
            .sum()
    }

    fn record(&mut self, now: Instant) {
        // Clean old entries
        self.counts
            .retain(|(t, _)| now.duration_since(*t) < self.window_size);

        // Add new entry or increment recent
        if let Some((last_t, count)) = self.counts.last_mut() {
            if now.duration_since(*last_t) < Duration::from_millis(100) {
                *count += 1;
                return;
            }
        }
        self.counts.push((now, 1));
    }

    fn time_until_slot(&self, now: Instant) -> Duration {
        if let Some((oldest, _)) = self.counts.first() {
            let age = now.duration_since(*oldest);
            if age < self.window_size {
                return self.window_size - age;
            }
        }
        Duration::ZERO
    }
}

/// Per-IP state
#[derive(Debug)]
struct IpRateState {
    requests: SlidingWindow,
    blocked_until: Option<Instant>,
}

/// Global rate state
#[derive(Debug)]
struct GlobalRateState {
    current_rps: u32,
    last_reset: Instant,
}

/// Distributed rate limit backend trait
#[async_trait::async_trait]
pub trait DistributedRateLimitBackend: Send + Sync {
    async fn increment(&self, key: &str, window_secs: u32) -> Result<u64, String>;
    async fn get(&self, key: &str) -> Result<Option<u64>, String>;
    async fn set_block(&self, key: &str, duration_secs: u32) -> Result<(), String>;
    async fn is_blocked(&self, key: &str) -> Result<bool, String>;
}

/// Rate limiter with sliding windows and token bucket
pub struct RateLimiter {
    config: RateLimitConfig,
    /// Per-tenant state
    tenant_state: Arc<RwLock<HashMap<Uuid, TenantRateState>>>,
    /// Per-IP state
    ip_state: Arc<RwLock<HashMap<IpAddr, IpRateState>>>,
    /// Global state
    global_state: Arc<RwLock<GlobalRateState>>,
    /// Distributed backend (optional)
    backend: Option<Arc<dyn DistributedRateLimitBackend>>,
    /// Tenant tier lookup
    tenant_tiers: Arc<RwLock<HashMap<Uuid, String>>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            tenant_state: Arc::new(RwLock::new(HashMap::new())),
            ip_state: Arc::new(RwLock::new(HashMap::new())),
            global_state: Arc::new(RwLock::new(GlobalRateState {
                current_rps: 0,
                last_reset: Instant::now(),
            })),
            backend: None,
            tenant_tiers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set distributed backend for multi-node deployments
    pub fn with_backend(mut self, backend: Arc<dyn DistributedRateLimitBackend>) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set tenant tier
    pub async fn set_tenant_tier(&self, tenant_id: Uuid, tier: String) {
        let mut tiers = self.tenant_tiers.write().await;
        tiers.insert(tenant_id, tier);
    }

    /// Check rate limits for a request
    pub async fn check(
        &self,
        tenant_id: Uuid,
        ip: IpAddr,
        endpoint: &str,
    ) -> RateLimitDecision {
        // 1. Check IP-level blocks first (fastest rejection)
        if let Some(block) = self.check_ip_block(ip).await {
            return block;
        }

        // 2. Check global capacity
        if let Some(throttle) = self.check_global_capacity().await {
            return throttle;
        }

        // 3. Check endpoint-specific limits
        if let Some(throttle) = self.check_endpoint_limit(endpoint).await {
            return throttle;
        }

        // 4. Check tenant limits
        self.check_tenant_limits(tenant_id).await
    }

    async fn check_ip_block(&self, ip: IpAddr) -> Option<RateLimitDecision> {
        let state = self.ip_state.read().await;
        if let Some(ip_state) = state.get(&ip) {
            if let Some(blocked_until) = ip_state.blocked_until {
                if Instant::now() < blocked_until {
                    let remaining = blocked_until - Instant::now();
                    return Some(RateLimitDecision::Block {
                        reason: BlockReason::IpBlocked,
                        block_duration_secs: remaining.as_secs() as u32,
                    });
                }
            }
        }
        None
    }

    async fn check_global_capacity(&self) -> Option<RateLimitDecision> {
        let mut state = self.global_state.write().await;
        let now = Instant::now();

        // Reset counter every second
        if now.duration_since(state.last_reset) >= Duration::from_secs(1) {
            state.current_rps = 0;
            state.last_reset = now;
        }

        if state.current_rps >= self.config.global_limits.max_global_rps {
            return Some(RateLimitDecision::Throttle {
                retry_after_secs: 1,
                reason: ThrottleReason::GlobalCapacity,
            });
        }

        state.current_rps += 1;
        None
    }

    async fn check_endpoint_limit(&self, endpoint: &str) -> Option<RateLimitDecision> {
        if let Some(&limit) = self.config.global_limits.endpoint_limits.get(endpoint) {
            // For simplicity, use same global counter
            // In production, use per-endpoint counters
            let state = self.global_state.read().await;
            if state.current_rps >= limit {
                return Some(RateLimitDecision::Throttle {
                    retry_after_secs: 1,
                    reason: ThrottleReason::EndpointLimit,
                });
            }
        }
        None
    }

    async fn check_tenant_limits(&self, tenant_id: Uuid) -> RateLimitDecision {
        let tier = {
            let tiers = self.tenant_tiers.read().await;
            tiers.get(&tenant_id).cloned().unwrap_or_else(|| "starter".to_string())
        };

        let mut state = self.tenant_state.write().await;
        let tenant_state = state
            .entry(tenant_id)
            .or_insert_with(|| TenantRateState::new(tier.clone()));

        let limits = self
            .config
            .tiers
            .get(&tenant_state.tier)
            .or_else(|| self.config.tiers.get("free"))
            .expect("Free tier must exist");

        let now = Instant::now();

        // Refill tokens (token bucket algorithm)
        let elapsed = now.duration_since(tenant_state.last_refill);
        let refill_rate = limits.rps as f64;
        tenant_state.tokens = (tenant_state.tokens + elapsed.as_secs_f64() * refill_rate)
            .min(limits.burst_size as f64);
        tenant_state.last_refill = now;

        // Check if we have tokens
        if tenant_state.tokens < 1.0 {
            return RateLimitDecision::Throttle {
                retry_after_secs: 1,
                reason: ThrottleReason::TenantRpmExceeded,
            };
        }

        // Check sliding windows
        let window_checks = [
            (0, limits.rps),
            (1, limits.rpm),
            (2, limits.rph),
            (3, limits.rpd),
        ];

        for (idx, limit) in window_checks {
            let count = tenant_state.windows[idx].count_in_window(now);
            if count >= limit {
                // Daily quota exhausted is a block, others are throttle
                if idx == 3 {
                    return RateLimitDecision::Block {
                        reason: BlockReason::DailyQuotaExhausted,
                        block_duration_secs: tenant_state.windows[3]
                            .time_until_slot(now)
                            .as_secs() as u32,
                    };
                }
                return RateLimitDecision::Throttle {
                    retry_after_secs: tenant_state.windows[idx].time_until_slot(now).as_secs()
                        as u32,
                    reason: ThrottleReason::TenantRpmExceeded,
                };
            }
        }

        // Check concurrent limit
        if tenant_state.concurrent >= limits.concurrent {
            return RateLimitDecision::Throttle {
                retry_after_secs: 1,
                reason: ThrottleReason::ConcurrentLimit,
            };
        }

        // Allow and consume
        tenant_state.tokens -= 1.0;
        for window in &mut tenant_state.windows {
            window.record(now);
        }
        tenant_state.concurrent += 1;

        RateLimitDecision::Allow {
            remaining: RateLimitRemaining {
                second: limits.rps.saturating_sub(tenant_state.windows[0].count_in_window(now)),
                minute: limits.rpm.saturating_sub(tenant_state.windows[1].count_in_window(now)),
                hour: limits.rph.saturating_sub(tenant_state.windows[2].count_in_window(now)),
                day: limits.rpd.saturating_sub(tenant_state.windows[3].count_in_window(now)),
            },
            reset_at: chrono::Utc::now() + chrono::Duration::seconds(1),
        }
    }

    /// Record request completion (decrements concurrent counter)
    pub async fn complete_request(&self, tenant_id: Uuid) {
        let mut state = self.tenant_state.write().await;
        if let Some(tenant_state) = state.get_mut(&tenant_id) {
            tenant_state.concurrent = tenant_state.concurrent.saturating_sub(1);
        }
    }

    /// Block an IP address
    pub async fn block_ip(&self, ip: IpAddr, duration_secs: u32) {
        let mut state = self.ip_state.write().await;
        let ip_state = state.entry(ip).or_insert_with(|| IpRateState {
            requests: SlidingWindow::new(Duration::from_secs(60)),
            blocked_until: None,
        });
        ip_state.blocked_until = Some(Instant::now() + Duration::from_secs(duration_secs as u64));
    }
}

/// HTTP headers for rate limit transparency
#[derive(Debug, Clone)]
pub struct RateLimitHeaders {
    /// Total requests allowed in current window
    pub x_ratelimit_limit: u32,
    /// Remaining requests in current window
    pub x_ratelimit_remaining: u32,
    /// Unix timestamp when limit resets
    pub x_ratelimit_reset: u64,
    /// Retry-After header for throttled requests
    pub retry_after: Option<u32>,
}

impl RateLimitHeaders {
    /// Create headers from rate limit decision
    pub fn from_decision(decision: &RateLimitDecision, tier_limits: &TierLimits) -> Self {
        match decision {
            RateLimitDecision::Allow { remaining, reset_at } => Self {
                x_ratelimit_limit: tier_limits.rpm,
                x_ratelimit_remaining: remaining.minute,
                x_ratelimit_reset: reset_at.timestamp() as u64,
                retry_after: None,
            },
            RateLimitDecision::Throttle {
                retry_after_secs, ..
            } => Self {
                x_ratelimit_limit: tier_limits.rpm,
                x_ratelimit_remaining: 0,
                x_ratelimit_reset: (chrono::Utc::now() + chrono::Duration::seconds(*retry_after_secs as i64)).timestamp() as u64,
                retry_after: Some(*retry_after_secs),
            },
            RateLimitDecision::Block {
                block_duration_secs,
                ..
            } => Self {
                x_ratelimit_limit: tier_limits.rpm,
                x_ratelimit_remaining: 0,
                x_ratelimit_reset: (chrono::Utc::now() + chrono::Duration::seconds(*block_duration_secs as i64)).timestamp() as u64,
                retry_after: Some(*block_duration_secs),
            },
        }
    }

    /// Convert to HTTP header pairs
    pub fn to_http_headers(&self) -> Vec<(String, String)> {
        let mut headers = vec![
            (
                "X-RateLimit-Limit".to_string(),
                self.x_ratelimit_limit.to_string(),
            ),
            (
                "X-RateLimit-Remaining".to_string(),
                self.x_ratelimit_remaining.to_string(),
            ),
            (
                "X-RateLimit-Reset".to_string(),
                self.x_ratelimit_reset.to_string(),
            ),
        ];
        if let Some(retry) = self.retry_after {
            headers.push(("Retry-After".to_string(), retry.to_string()));
        }
        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_allows_within_limits() {
        let config = RateLimitConfig::default();
        let limiter = RateLimiter::new(config);
        let tenant_id = Uuid::new_v4();
        let ip = "127.0.0.1".parse().unwrap();

        let decision = limiter.check(tenant_id, ip, "/api/test").await;
        assert!(matches!(decision, RateLimitDecision::Allow { .. }));
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_ip() {
        let config = RateLimitConfig::default();
        let limiter = RateLimiter::new(config);
        let tenant_id = Uuid::new_v4();
        let ip = "127.0.0.1".parse().unwrap();

        limiter.block_ip(ip, 3600).await;

        let decision = limiter.check(tenant_id, ip, "/api/test").await;
        assert!(matches!(decision, RateLimitDecision::Block { .. }));
    }

    #[tokio::test]
    async fn test_sliding_window() {
        let mut window = SlidingWindow::new(Duration::from_secs(1));
        let now = Instant::now();

        window.record(now);
        window.record(now);
        window.record(now);

        assert_eq!(window.count_in_window(now), 3);
    }

    #[test]
    fn test_default_tiers() {
        let config = RateLimitConfig::default();

        assert!(config.tiers.contains_key("free"));
        assert!(config.tiers.contains_key("starter"));
        assert!(config.tiers.contains_key("professional"));
        assert!(config.tiers.contains_key("enterprise"));

        let enterprise = config.tiers.get("enterprise").unwrap();
        assert_eq!(enterprise.rps, 500);
    }
}
