//! # Circuit Breaker Implementation
//!
//! Fault-tolerant circuit breaker for GLM-4.6 API requests.
//! Prevents cascade failures and enables graceful degradation.

use crate::glm46::types::{CircuitBreakerConfig, CircuitState, GLM46Error, GLM46Result};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Circuit breaker for fault tolerance
#[derive(Debug)]
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    config: CircuitBreakerConfig,
    failure_count: Arc<RwLock<u32>>,
    success_count: Arc<RwLock<u32>>,
    last_failure_time: Arc<RwLock<Option<SystemTime>>>,
}

impl CircuitBreaker {
    /// Create new circuit breaker with configuration
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            config,
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Execute operation with circuit breaker protection
    pub async fn execute<F, T>(&self, operation: F) -> GLM46Result<T>
    where
        F: std::future::Future<Output = GLM46Result<T>>,
    {
        // Check if we can attempt operation
        self.check_state().await?;

        // Execute operation
        match operation.await {
            Ok(result) => {
                self.record_success().await;
                Ok(result)
            }
            Err(error) => {
                self.record_failure().await;
                Err(error)
            }
        }
    }

    /// Check current state and allow or deny operation
    async fn check_state(&self) -> GLM46Result<()> {
        let state = self.state.read().await;
        match &*state {
            CircuitState::Closed => Ok(()),
            CircuitState::Open {
                opens_at,
                reset_after,
            } => {
                let now = SystemTime::now();
                if now >= *opens_at + *reset_after {
                    drop(state);
                    self.transition_to_half_open().await;
                    Ok(())
                } else {
                    let time_until_reset = (*opens_at + *reset_after)
                        .duration_since(now)
                        .unwrap_or_default();
                    Err(GLM46Error::CircuitOpen {
                        reason: format!("Circuit open for {:?} more", time_until_reset),
                    })
                }
            }
            CircuitState::HalfOpen { .. } => Ok(()),
        }
    }

    /// Record successful operation
    async fn record_success(&self) {
        let mut state = self.state.write().await;

        match &mut *state {
            CircuitState::Closed => {
                let mut success_count = self.success_count.write().await;
                *success_count = 0; // Reset on success in closed state
            }
            CircuitState::HalfOpen {
                probation_requests,
                max_probation: _,
            } => {
                *probation_requests += 1;
                let mut success_count = self.success_count.write().await;
                *success_count += 1;

                debug!(
                    "Circuit half-open success: {}/{}",
                    success_count, self.config.success_threshold
                );

                if *success_count >= self.config.success_threshold {
                    info!("Circuit breaker closing after {} successes", success_count);
                    *state = CircuitState::Closed;
                    *self.failure_count.write().await = 0;
                    *self.success_count.write().await = 0;
                }
            }
            _ => {}
        }
    }

    /// Record failed operation
    async fn record_failure(&self) {
        let mut failure_count = self.failure_count.write().await;
        *failure_count += 1;

        let mut success_count = self.success_count.write().await;
        *success_count = 0; // Reset success count on failure

        *self.last_failure_time.write().await = Some(SystemTime::now());

        debug!("Circuit breaker failure count: {}", failure_count);

        let mut state = self.state.write().await;

        match &mut *state {
            CircuitState::Closed => {
                if *failure_count >= self.config.failure_threshold {
                    warn!("Circuit breaker opening after {} failures", failure_count);
                    *state = CircuitState::Open {
                        opens_at: SystemTime::now(),
                        reset_after: self.config.reset_timeout,
                    };
                }
            }
            CircuitState::HalfOpen { .. } => {
                warn!("Circuit breaker re-opening due to failure in half-open state");
                *state = CircuitState::Open {
                    opens_at: SystemTime::now(),
                    reset_after: self.config.reset_timeout,
                };
            }
            CircuitState::Open { .. } => {
                // Already open, nothing to do
            }
        }
    }

    /// Transition to half-open state
    async fn transition_to_half_open(&self) {
        info!("Circuit breaker transitioning to half-open state");
        let mut state = self.state.write().await;
        *state = CircuitState::HalfOpen {
            probation_requests: 0,
            max_probation: self.config.success_threshold,
        };
    }

    /// Get current circuit state
    pub async fn get_state(&self) -> CircuitState {
        self.state.read().await.clone()
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> CircuitStats {
        CircuitStats {
            state: self.state.read().await.clone(),
            failure_count: *self.failure_count.read().await,
            success_count: *self.success_count.read().await,
            last_failure_time: *self.last_failure_time.read().await,
        }
    }

    /// Force reset circuit breaker to closed state
    pub async fn reset(&self) {
        info!("Circuit breaker manually reset to closed state");
        let mut state = self.state.write().await;
        *state = CircuitState::Closed;
        *self.failure_count.write().await = 0;
        *self.success_count.write().await = 0;
        *self.last_failure_time.write().await = None;
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitStats {
    pub state: CircuitState,
    pub failure_count: u32,
    pub success_count: u32,
    pub last_failure_time: Option<SystemTime>,
}
