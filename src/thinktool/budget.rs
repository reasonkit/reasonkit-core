//! Budget Configuration for Adaptive Compute Time
//!
//! Enables --budget flag for controlling reasoning execution within constraints.
//!
//! ## Budget Types
//! - Time budget: Maximum wall-clock time (e.g., "10s", "2m")
//! - Token budget: Maximum tokens to consume
//! - Cost budget: Maximum USD to spend (e.g., "$0.10")
//!
//! ## Adaptive Behavior
//! When budget is constrained, the executor will:
//! 1. Skip optional steps if time is running low
//! 2. Reduce max_tokens for remaining steps
//! 3. Use faster/cheaper model tiers when approaching limit
//! 4. Terminate early if budget is exhausted

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Budget configuration for protocol execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum execution time
    pub time_limit: Option<Duration>,

    /// Maximum tokens to consume
    pub token_limit: Option<u32>,

    /// Maximum cost in USD
    pub cost_limit: Option<f64>,

    /// Strategy when approaching budget limits
    #[serde(default)]
    pub strategy: BudgetStrategy,

    /// Percentage of budget at which to start adapting (0.0-1.0)
    #[serde(default = "default_adapt_threshold")]
    pub adapt_threshold: f64,
}

fn default_adapt_threshold() -> f64 {
    0.7 // Start adapting at 70% budget usage
}

/// Strategy for handling budget constraints
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetStrategy {
    /// Strict: Fail if budget would be exceeded
    Strict,
    /// Adaptive: Reduce quality/scope to stay within budget (default)
    #[default]
    Adaptive,
    /// BestEffort: Try to complete as much as possible, may exceed budget
    BestEffort,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            time_limit: None,
            token_limit: None,
            cost_limit: None,
            strategy: BudgetStrategy::default(),
            adapt_threshold: default_adapt_threshold(),
        }
    }
}

impl BudgetConfig {
    /// Create an unlimited budget (no constraints)
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Create a time-limited budget
    pub fn with_time(duration: Duration) -> Self {
        Self {
            time_limit: Some(duration),
            ..Default::default()
        }
    }

    /// Create a token-limited budget
    pub fn with_tokens(limit: u32) -> Self {
        Self {
            token_limit: Some(limit),
            ..Default::default()
        }
    }

    /// Create a cost-limited budget
    pub fn with_cost(usd: f64) -> Self {
        Self {
            cost_limit: Some(usd),
            ..Default::default()
        }
    }

    /// Set budget strategy
    pub fn with_strategy(mut self, strategy: BudgetStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Check if any limits are set
    pub fn is_constrained(&self) -> bool {
        self.time_limit.is_some() || self.token_limit.is_some() || self.cost_limit.is_some()
    }

    /// Parse a budget string like "10s", "1000t", "$0.50"
    pub fn parse(budget_str: &str) -> Result<Self, BudgetParseError> {
        let budget_str = budget_str.trim();

        if budget_str.is_empty() {
            return Err(BudgetParseError::Empty);
        }

        // Cost budget: $X.XX
        if let Some(cost) = budget_str.strip_prefix('$') {
            let usd: f64 = cost.parse().map_err(|_| BudgetParseError::InvalidCost)?;
            return Ok(Self::with_cost(usd));
        }

        // Token budget: XXXt or XXXtokens
        if budget_str.ends_with('t') || budget_str.ends_with("tokens") {
            let num_str = budget_str.trim_end_matches("tokens").trim_end_matches('t');
            let tokens: u32 = num_str
                .parse()
                .map_err(|_| BudgetParseError::InvalidTokens)?;
            return Ok(Self::with_tokens(tokens));
        }

        // Time budget: Xs, Xm, Xh
        if let Some(secs) = budget_str.strip_suffix('s') {
            let seconds: u64 = secs.parse().map_err(|_| BudgetParseError::InvalidTime)?;
            return Ok(Self::with_time(Duration::from_secs(seconds)));
        }

        if let Some(mins) = budget_str.strip_suffix('m') {
            let minutes: u64 = mins.parse().map_err(|_| BudgetParseError::InvalidTime)?;
            return Ok(Self::with_time(Duration::from_secs(minutes * 60)));
        }

        if let Some(hours) = budget_str.strip_suffix('h') {
            let hours_val: u64 = hours.parse().map_err(|_| BudgetParseError::InvalidTime)?;
            return Ok(Self::with_time(Duration::from_secs(hours_val * 3600)));
        }

        Err(BudgetParseError::UnknownFormat(budget_str.to_string()))
    }
}

/// Error parsing budget string
#[derive(Debug, Clone)]
pub enum BudgetParseError {
    /// Budget string is empty
    Empty,
    /// Invalid time format
    InvalidTime,
    /// Invalid token count
    InvalidTokens,
    /// Invalid cost value
    InvalidCost,
    /// Unknown format string
    UnknownFormat(String),
}

impl std::fmt::Display for BudgetParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetParseError::Empty => write!(f, "Budget string is empty"),
            BudgetParseError::InvalidTime => write!(f, "Invalid time format (use Xs, Xm, or Xh)"),
            BudgetParseError::InvalidTokens => write!(f, "Invalid token count (use Xt or Xtokens)"),
            BudgetParseError::InvalidCost => write!(f, "Invalid cost format (use $X.XX)"),
            BudgetParseError::UnknownFormat(s) => write!(f, "Unknown budget format: {}", s),
        }
    }
}

impl std::error::Error for BudgetParseError {}

/// Runtime budget tracker
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    /// Configuration
    config: BudgetConfig,

    /// When execution started
    start_time: Instant,

    /// Tokens consumed so far
    tokens_used: u32,

    /// Cost incurred so far (USD)
    cost_incurred: f64,

    /// Steps completed
    steps_completed: usize,

    /// Steps skipped due to budget
    steps_skipped: usize,
}

impl BudgetTracker {
    /// Create a new budget tracker
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            config,
            start_time: Instant::now(),
            tokens_used: 0,
            cost_incurred: 0.0,
            steps_completed: 0,
            steps_skipped: 0,
        }
    }

    /// Record token and cost usage
    pub fn record_usage(&mut self, tokens: u32, cost: f64) {
        self.tokens_used += tokens;
        self.cost_incurred += cost;
        self.steps_completed += 1;
    }

    /// Record a skipped step
    pub fn record_skip(&mut self) {
        self.steps_skipped += 1;
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get time remaining (if time limit set)
    pub fn time_remaining(&self) -> Option<Duration> {
        self.config
            .time_limit
            .map(|limit| limit.saturating_sub(self.elapsed()))
    }

    /// Get tokens remaining (if token limit set)
    pub fn tokens_remaining(&self) -> Option<u32> {
        self.config
            .token_limit
            .map(|limit| limit.saturating_sub(self.tokens_used))
    }

    /// Get cost remaining (if cost limit set)
    pub fn cost_remaining(&self) -> Option<f64> {
        self.config
            .cost_limit
            .map(|limit| (limit - self.cost_incurred).max(0.0))
    }

    /// Check if budget is exhausted
    pub fn is_exhausted(&self) -> bool {
        if let Some(remaining) = self.time_remaining() {
            if remaining.is_zero() {
                return true;
            }
        }
        if let Some(remaining) = self.tokens_remaining() {
            if remaining == 0 {
                return true;
            }
        }
        if let Some(remaining) = self.cost_remaining() {
            if remaining <= 0.0 {
                return true;
            }
        }
        false
    }

    /// Calculate budget usage ratio (0.0-1.0)
    pub fn usage_ratio(&self) -> f64 {
        let mut max_ratio = 0.0f64;

        if let Some(limit) = self.config.time_limit {
            let ratio = self.elapsed().as_secs_f64() / limit.as_secs_f64();
            max_ratio = max_ratio.max(ratio);
        }

        if let Some(limit) = self.config.token_limit {
            let ratio = self.tokens_used as f64 / limit as f64;
            max_ratio = max_ratio.max(ratio);
        }

        if let Some(limit) = self.config.cost_limit {
            let ratio = self.cost_incurred / limit;
            max_ratio = max_ratio.max(ratio);
        }

        max_ratio.min(1.0)
    }

    /// Check if we should start adapting (approaching limit)
    pub fn should_adapt(&self) -> bool {
        self.usage_ratio() >= self.config.adapt_threshold
    }

    /// Get adaptive max_tokens based on remaining budget
    pub fn adaptive_max_tokens(&self, requested: u32) -> u32 {
        if let Some(remaining) = self.tokens_remaining() {
            // Reserve some tokens for remaining steps
            let reserve = remaining / 4;
            return requested.min(remaining - reserve);
        }
        requested
    }

    /// Check if step should be skipped (non-essential and low budget)
    pub fn should_skip_step(&self, is_essential: bool) -> bool {
        if is_essential {
            return false;
        }

        match self.config.strategy {
            BudgetStrategy::Strict => self.is_exhausted(),
            BudgetStrategy::Adaptive => self.usage_ratio() > 0.9,
            BudgetStrategy::BestEffort => false,
        }
    }

    /// Get budget summary
    pub fn summary(&self) -> BudgetSummary {
        BudgetSummary {
            elapsed: self.elapsed(),
            tokens_used: self.tokens_used,
            cost_incurred: self.cost_incurred,
            steps_completed: self.steps_completed,
            steps_skipped: self.steps_skipped,
            usage_ratio: self.usage_ratio(),
            exhausted: self.is_exhausted(),
        }
    }
}

/// Summary of budget usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSummary {
    /// Time elapsed
    #[serde(with = "duration_serde")]
    pub elapsed: Duration,

    /// Tokens consumed
    pub tokens_used: u32,

    /// Cost in USD
    pub cost_incurred: f64,

    /// Steps completed
    pub steps_completed: usize,

    /// Steps skipped due to budget
    pub steps_skipped: usize,

    /// Usage ratio (0.0-1.0)
    pub usage_ratio: f64,

    /// Whether budget was exhausted
    pub exhausted: bool,
}

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_millis().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_time_seconds() {
        let budget = BudgetConfig::parse("30s").unwrap();
        assert_eq!(budget.time_limit, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_parse_time_minutes() {
        let budget = BudgetConfig::parse("5m").unwrap();
        assert_eq!(budget.time_limit, Some(Duration::from_secs(300)));
    }

    #[test]
    fn test_parse_tokens() {
        let budget = BudgetConfig::parse("1000t").unwrap();
        assert_eq!(budget.token_limit, Some(1000));
    }

    #[test]
    fn test_parse_tokens_full() {
        let budget = BudgetConfig::parse("5000tokens").unwrap();
        assert_eq!(budget.token_limit, Some(5000));
    }

    #[test]
    fn test_parse_cost() {
        let budget = BudgetConfig::parse("$0.50").unwrap();
        assert_eq!(budget.cost_limit, Some(0.50));
    }

    #[test]
    fn test_budget_tracker_usage() {
        let config = BudgetConfig::with_tokens(1000);
        let mut tracker = BudgetTracker::new(config);

        tracker.record_usage(200, 0.01);
        assert_eq!(tracker.tokens_remaining(), Some(800));
        assert!(!tracker.is_exhausted());

        tracker.record_usage(800, 0.04);
        assert_eq!(tracker.tokens_remaining(), Some(0));
        assert!(tracker.is_exhausted());
    }

    #[test]
    fn test_budget_tracker_adapt() {
        let config = BudgetConfig::with_tokens(1000);
        let mut tracker = BudgetTracker::new(config);

        tracker.record_usage(600, 0.03);
        assert!(!tracker.should_adapt()); // 60% < 70% threshold

        tracker.record_usage(150, 0.01);
        assert!(tracker.should_adapt()); // 75% > 70% threshold
    }

    #[test]
    fn test_adaptive_max_tokens() {
        let config = BudgetConfig::with_tokens(1000);
        let mut tracker = BudgetTracker::new(config);

        // Initially can request full amount
        assert_eq!(tracker.adaptive_max_tokens(500), 500);

        // After using some, it limits
        tracker.record_usage(800, 0.04);
        assert!(tracker.adaptive_max_tokens(500) < 200);
    }
}
