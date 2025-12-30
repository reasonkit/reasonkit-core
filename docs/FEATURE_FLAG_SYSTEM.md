# ReasonKit Feature Flag System Design

> Comprehensive feature flag architecture for progressive rollout, experimentation, and access control
> Version: 1.0.0 | Last Updated: 2025-12-28

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Feature Flag Types](#2-feature-flag-types)
3. [Flag Architecture](#3-flag-architecture)
4. [Rust Implementation](#4-rust-implementation)
5. [Python Bindings](#5-python-bindings)
6. [CLI Integration](#6-cli-integration)
7. [Targeting Rules](#7-targeting-rules)
8. [Rollout Strategies](#8-rollout-strategies)
9. [A/B Testing Framework](#9-ab-testing-framework)
10. [Administration](#10-administration)
11. [Monitoring and Analytics](#11-monitoring-and-analytics)
12. [Best Practices](#12-best-practices)
13. [Integration Options](#13-integration-options)
14. [Implementation Roadmap](#14-implementation-roadmap)

---

## 1. Executive Summary

### Purpose

The ReasonKit Feature Flag System provides controlled, progressive rollout of features across different user segments, plans, and environments. It enables:

- **Safe deployments**: Gradual feature release with instant rollback capability
- **Experimentation**: A/B testing to validate feature impact
- **Access control**: Plan-based and user-specific feature gating
- **Operational control**: Kill switches, maintenance mode, and rate limiting

### Design Principles

| Principle                  | Description                                    |
| -------------------------- | ---------------------------------------------- |
| **Performance First**      | Sub-millisecond flag evaluation (Rust-native)  |
| **Type Safety**            | Compile-time verification of flag usage        |
| **Offline Capable**        | Local cache with graceful degradation          |
| **Audit Trail**            | Full history of flag changes and evaluations   |
| **OpenFeature Compatible** | Standard-compliant for future interoperability |

### Architecture Overview

```
                    +-------------------+
                    |   Flag Storage    |
                    |  (YAML/DB/Remote) |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  Evaluation Engine |
                    |   (Rust Core)      |
                    +--------+----------+
                             |
          +------------------+------------------+
          |                  |                  |
    +-----v-----+      +-----v-----+      +-----v-----+
    |  CLI API  |      | Rust API  |      | Python API |
    | rk-core   |      | Library   |      | Bindings   |
    +-----------+      +-----------+      +------------+
```

---

## 2. Feature Flag Types

### 2.1 Release Flags

Control feature availability during rollout phases.

```yaml
release_flags:
  new_powercombo_v2:
    description: "Enhanced PowerCombo with cross-validation"
    type: release
    default: false
    lifecycle:
      created: "2025-01-15"
      expected_removal: "2025-04-15" # After GA
    rules:
      - condition: "environment == 'development'"
        value: true
      - condition: "percentage < 5"
        value: true
```

**Use Cases:**

- Progressive feature rollout (0% -> 5% -> 25% -> 100%)
- Kill switch for production issues
- Feature gating during development

### 2.2 Experiment Flags

Enable controlled experimentation and A/B testing.

```yaml
experiment_flags:
  profile_comparison_exp:
    description: "Compare balanced vs new_balanced profile performance"
    type: experiment
    default_variant: "control"
    variants:
      control:
        profile: "balanced"
        weight: 50
      treatment:
        profile: "new_balanced"
        weight: 50
    metrics:
      primary: "reasoning_quality_score"
      secondary:
        - "response_latency_p95"
        - "token_efficiency"
        - "user_satisfaction"
    guardrails:
      - metric: "error_rate"
        threshold: 0.05
        action: "halt"
    sample_size: 1000
    duration_days: 14
```

**Use Cases:**

- A/B testing new ThinkTool configurations
- Multivariate testing of prompt variations
- Holdout groups for long-term impact analysis

### 2.3 Ops Flags

Control operational behavior and system modes.

```yaml
ops_flags:
  maintenance_mode:
    description: "Enable maintenance mode for all non-critical endpoints"
    type: ops
    default: false

  rate_limit_aggressive:
    description: "Activate aggressive rate limiting during load spikes"
    type: ops
    default: false
    rules:
      - condition: "system.cpu_usage > 80"
        value: true
      - condition: "system.request_queue_depth > 1000"
        value: true

  debug_mode:
    description: "Enable verbose debug logging"
    type: ops
    default: false
    rules:
      - condition: "environment == 'development'"
        value: true
      - condition: "user.email ends_with '@reasonkit.sh'"
        value: true

  trace_sampling_rate:
    description: "Percentage of requests to trace"
    type: ops
    value_type: number
    default: 10
    rules:
      - condition: "environment == 'production'"
        value: 1
      - condition: "environment == 'staging'"
        value: 50
```

**Use Cases:**

- Maintenance mode during deployments
- Dynamic rate limiting based on system load
- Debug mode for internal testing
- Trace sampling control

### 2.4 Permission Flags

Control access based on plans, users, or entitlements.

```yaml
permission_flags:
  advanced_thinktools:
    description: "Access to AtomicBreak, HighReflect, and other pro tools"
    type: permission
    default: false
    rules:
      - condition: "user.plan == 'enterprise'"
        value: true
      - condition: "user.plan == 'pro'"
        value: true

  beta_features:
    description: "Access to beta features for early adopters"
    type: permission
    default: false
    rules:
      - condition: "user.id in beta_users"
        value: true
      - condition: "user.signup_date < '2025-01-01'"
        value: true

  unlimited_tokens:
    description: "No token limits per request"
    type: permission
    default: false
    rules:
      - condition: "user.plan == 'enterprise'"
        value: true
      - condition: "user.has_addon('unlimited_compute')"
        value: true

  custom_protocols:
    description: "Ability to define custom ThinkTool protocols"
    type: permission
    default: false
    rules:
      - condition: "user.plan in ['pro', 'enterprise']"
        value: true
```

**Use Cases:**

- Plan-based feature access (Free/Pro/Enterprise)
- Beta program management
- User-specific overrides
- Addon and entitlement gating

---

## 3. Flag Architecture

### 3.1 Flag Definition Schema

```yaml
# Complete flag definition schema
flag:
  # Required fields
  key: string # Unique identifier (snake_case)
  description: string # Human-readable description
  type: enum # release | experiment | ops | permission
  default: any # Default value when no rules match

  # Optional fields
  value_type: enum # boolean (default) | string | number | json
  enabled: boolean # Master switch (default: true)

  # Targeting rules (evaluated in order)
  rules:
    - condition: string # Condition expression
      value: any # Value if condition matches
      priority: number # Optional priority (lower = higher priority)

  # Metadata
  tags: [string] # For filtering and organization
  owner: string # Team or person responsible
  created_at: datetime # Creation timestamp
  updated_at: datetime # Last modification

  # Lifecycle
  lifecycle:
    status: enum # active | deprecated | archived
    created: date # When flag was created
    expected_removal: date # When flag should be removed

  # Experiment-specific (type: experiment only)
  variants: map # Variant definitions
  metrics: object # Metrics to track
  guardrails: [object] # Safety guardrails
  sample_size: number # Required sample size
  duration_days: number # Experiment duration
```

### 3.2 Evaluation Engine

The evaluation engine processes flags with deterministic, consistent hashing.

```
┌─────────────────────────────────────────────────────────────────┐
│                      EVALUATION FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FLAG LOOKUP                                                 │
│     └── Check if flag exists                                    │
│         ├── Yes: Continue                                       │
│         └── No: Return default or error                         │
│                                                                 │
│  2. MASTER SWITCH CHECK                                         │
│     └── Is flag.enabled == true?                                │
│         ├── Yes: Continue                                       │
│         └── No: Return flag.default                             │
│                                                                 │
│  3. RULE EVALUATION (in order)                                  │
│     └── For each rule:                                          │
│         ├── Evaluate condition with context                     │
│         ├── Match: Return rule.value                            │
│         └── No match: Continue to next rule                     │
│                                                                 │
│  4. DEFAULT FALLBACK                                            │
│     └── No rules matched: Return flag.default                   │
│                                                                 │
│  5. EXPERIMENT VARIANT ASSIGNMENT                               │
│     └── If type == experiment:                                  │
│         ├── Compute bucket: hash(user_id + flag_key) % 100      │
│         ├── Assign variant based on weights                     │
│         └── Cache assignment for consistency                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Condition Expression Language

A simple, safe expression language for targeting rules.

```
EXPRESSION := COMPARISON | LOGICAL_EXPR | MEMBERSHIP | PERCENTAGE

COMPARISON := FIELD OPERATOR VALUE
OPERATOR   := "==" | "!=" | ">" | ">=" | "<" | "<="
FIELD      := "user." ATTR | "context." ATTR | "system." ATTR
VALUE      := STRING | NUMBER | BOOLEAN

LOGICAL_EXPR := EXPRESSION "and" EXPRESSION
              | EXPRESSION "or" EXPRESSION
              | "not" EXPRESSION

MEMBERSHIP := FIELD "in" LIST
            | FIELD "not_in" LIST
            | FIELD "contains" STRING
            | FIELD "starts_with" STRING
            | FIELD "ends_with" STRING

PERCENTAGE := "percentage" "<" NUMBER
```

**Examples:**

```yaml
# Simple comparison
condition: "user.plan == 'enterprise'"

# Membership check
condition: "user.id in beta_users"
condition: "user.email ends_with '@reasonkit.sh'"

# Percentage rollout (consistent hashing)
condition: "percentage < 10"

# Logical combinations
condition: "user.plan == 'pro' and user.signup_date < '2025-01-01'"
condition: "environment == 'production' or user.is_staff == true"

# System conditions
condition: "system.region == 'us-east-1'"
condition: "context.sdk_version >= '2.0.0'"
```

### 3.4 Flag Storage Options

#### Option A: YAML Files (Default for CLI/Local)

```yaml
# config/feature_flags.yaml
version: "1.0"
flags:
  new_thinktool_alpha:
    description: "New experimental ThinkTool"
    type: release
    default: false
    rules:
      - condition: "user.plan == 'enterprise'"
        value: true
      - condition: "percentage < 10"
        value: true
```

#### Option B: SQLite (Local Persistence)

```sql
-- Schema for local SQLite storage
CREATE TABLE feature_flags (
    key TEXT PRIMARY KEY,
    description TEXT,
    type TEXT CHECK(type IN ('release', 'experiment', 'ops', 'permission')),
    value_type TEXT DEFAULT 'boolean',
    default_value TEXT,
    enabled INTEGER DEFAULT 1,
    rules TEXT,  -- JSON array of rules
    metadata TEXT,  -- JSON object for tags, owner, etc.
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE flag_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flag_key TEXT,
    user_id TEXT,
    context TEXT,  -- JSON
    result TEXT,
    evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_evaluations_flag ON flag_evaluations(flag_key);
CREATE INDEX idx_evaluations_time ON flag_evaluations(evaluated_at);
```

#### Option C: Remote Service (Enterprise)

```rust
// Remote flag provider configuration
pub struct RemoteFlagConfig {
    /// API endpoint for flag service
    pub endpoint: String,

    /// API key for authentication
    pub api_key: String,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,

    /// Polling interval for updates
    pub poll_interval_seconds: u64,

    /// Fallback to local on failure
    pub fallback_to_local: bool,
}
```

---

## 4. Rust Implementation

### 4.1 Core Types

```rust
//! Feature Flag System for ReasonKit
//!
//! Provides type-safe, high-performance feature flag evaluation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Flag type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlagType {
    Release,
    Experiment,
    Ops,
    Permission,
}

/// Value type for flags
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FlagValue {
    Boolean(bool),
    String(String),
    Number(f64),
    Json(serde_json::Value),
}

impl Default for FlagValue {
    fn default() -> Self {
        FlagValue::Boolean(false)
    }
}

/// A targeting rule for flag evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlagRule {
    /// Condition expression (e.g., "user.plan == 'enterprise'")
    pub condition: String,

    /// Value to return if condition matches
    pub value: FlagValue,

    /// Optional priority (lower = higher priority)
    #[serde(default)]
    pub priority: i32,
}

/// Lifecycle information for a flag
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlagLifecycle {
    pub status: LifecycleStatus,
    pub created: Option<String>,
    pub expected_removal: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleStatus {
    #[default]
    Active,
    Deprecated,
    Archived,
}

/// Complete feature flag definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlag {
    /// Unique identifier
    pub key: String,

    /// Human-readable description
    pub description: String,

    /// Flag type
    #[serde(rename = "type")]
    pub flag_type: FlagType,

    /// Default value when no rules match
    pub default: FlagValue,

    /// Master enable switch
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Targeting rules (evaluated in order)
    #[serde(default)]
    pub rules: Vec<FlagRule>,

    /// Tags for organization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Responsible owner
    pub owner: Option<String>,

    /// Lifecycle information
    #[serde(default)]
    pub lifecycle: FlagLifecycle,
}

fn default_enabled() -> bool {
    true
}

/// Context for flag evaluation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationContext {
    /// User attributes
    #[serde(default)]
    pub user: UserContext,

    /// Request/environment context
    #[serde(default)]
    pub context: HashMap<String, serde_json::Value>,

    /// System attributes (auto-populated)
    #[serde(default)]
    pub system: SystemContext,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserContext {
    pub id: Option<String>,
    pub email: Option<String>,
    pub plan: Option<String>,
    pub signup_date: Option<String>,
    pub is_staff: bool,

    /// Additional user attributes
    #[serde(flatten)]
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemContext {
    pub environment: Option<String>,
    pub region: Option<String>,
    pub version: Option<String>,
    pub cpu_usage: Option<f64>,
    pub memory_usage: Option<f64>,
}

/// Result of a flag evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// The flag key
    pub flag_key: String,

    /// Evaluated value
    pub value: FlagValue,

    /// Whether this was a default or rule-based value
    pub is_default: bool,

    /// Which rule matched (index), if any
    pub matched_rule_index: Option<usize>,

    /// For experiments: which variant was assigned
    pub variant: Option<String>,

    /// Evaluation timestamp
    pub evaluated_at: chrono::DateTime<chrono::Utc>,
}

/// Errors that can occur during flag evaluation
#[derive(Debug, thiserror::Error)]
pub enum FlagError {
    #[error("Flag not found: {0}")]
    FlagNotFound(String),

    #[error("Invalid condition expression: {0}")]
    InvalidCondition(String),

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}

pub type FlagResult<T> = Result<T, FlagError>;
```

### 4.2 Evaluation Engine

```rust
//! Feature flag evaluation engine

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use sha2::{Sha256, Digest};

/// The core flag evaluation engine
pub struct FlagEngine {
    /// Flag storage backend
    storage: Arc<dyn FlagStorage>,

    /// In-memory cache
    cache: RwLock<HashMap<String, FeatureFlag>>,

    /// User list storage (for "in beta_users" conditions)
    user_lists: RwLock<HashMap<String, Vec<String>>>,

    /// Evaluation history (for analytics)
    history: Option<Arc<dyn EvaluationHistory>>,

    /// Configuration
    config: FlagEngineConfig,
}

#[derive(Debug, Clone)]
pub struct FlagEngineConfig {
    /// Whether to cache flags in memory
    pub enable_cache: bool,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,

    /// Whether to record evaluation history
    pub enable_history: bool,

    /// Default environment
    pub default_environment: String,
}

impl Default for FlagEngineConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            cache_ttl_seconds: 300,
            enable_history: true,
            default_environment: "development".to_string(),
        }
    }
}

impl FlagEngine {
    /// Create a new flag engine with the given storage backend
    pub fn new(storage: Arc<dyn FlagStorage>, config: FlagEngineConfig) -> Self {
        Self {
            storage,
            cache: RwLock::new(HashMap::new()),
            user_lists: RwLock::new(HashMap::new()),
            history: None,
            config,
        }
    }

    /// Create engine with YAML file storage
    pub fn from_yaml(path: impl AsRef<std::path::Path>) -> FlagResult<Self> {
        let storage = YamlFlagStorage::load(path)?;
        Ok(Self::new(Arc::new(storage), FlagEngineConfig::default()))
    }

    /// Evaluate a boolean flag
    pub fn is_enabled(&self, key: &str, context: &EvaluationContext) -> bool {
        match self.evaluate(key, context) {
            Ok(result) => match result.value {
                FlagValue::Boolean(b) => b,
                _ => false,
            },
            Err(_) => false,
        }
    }

    /// Evaluate a flag and return the full result
    pub fn evaluate(&self, key: &str, context: &EvaluationContext) -> FlagResult<EvaluationResult> {
        // Get flag from cache or storage
        let flag = self.get_flag(key)?;

        // Check master switch
        if !flag.enabled {
            return Ok(EvaluationResult {
                flag_key: key.to_string(),
                value: flag.default.clone(),
                is_default: true,
                matched_rule_index: None,
                variant: None,
                evaluated_at: chrono::Utc::now(),
            });
        }

        // Evaluate rules in order
        for (index, rule) in flag.rules.iter().enumerate() {
            if self.evaluate_condition(&rule.condition, context, &flag)? {
                let result = EvaluationResult {
                    flag_key: key.to_string(),
                    value: rule.value.clone(),
                    is_default: false,
                    matched_rule_index: Some(index),
                    variant: None,
                    evaluated_at: chrono::Utc::now(),
                };

                // Record evaluation if history enabled
                if let Some(ref history) = self.history {
                    let _ = history.record(&result, context);
                }

                return Ok(result);
            }
        }

        // No rules matched, return default
        Ok(EvaluationResult {
            flag_key: key.to_string(),
            value: flag.default.clone(),
            is_default: true,
            matched_rule_index: None,
            variant: None,
            evaluated_at: chrono::Utc::now(),
        })
    }

    /// Evaluate an experiment flag and return variant assignment
    pub fn get_variant(&self, key: &str, context: &EvaluationContext) -> FlagResult<String> {
        let flag = self.get_flag(key)?;

        if flag.flag_type != FlagType::Experiment {
            return Err(FlagError::TypeMismatch {
                expected: "experiment".to_string(),
                actual: format!("{:?}", flag.flag_type),
            });
        }

        // Compute consistent bucket based on user ID and flag key
        let user_id = context.user.id.as_deref().unwrap_or("anonymous");
        let bucket = self.compute_bucket(user_id, key);

        // This would use variant weights from the flag definition
        // For now, simple 50/50 split
        let variant = if bucket < 50 { "control" } else { "treatment" };

        Ok(variant.to_string())
    }

    /// Get a flag by key
    fn get_flag(&self, key: &str) -> FlagResult<FeatureFlag> {
        // Check cache first
        if self.config.enable_cache {
            if let Some(flag) = self.cache.read().get(key) {
                return Ok(flag.clone());
            }
        }

        // Load from storage
        let flag = self.storage.get(key)?;

        // Update cache
        if self.config.enable_cache {
            self.cache.write().insert(key.to_string(), flag.clone());
        }

        Ok(flag)
    }

    /// Evaluate a condition expression
    fn evaluate_condition(
        &self,
        condition: &str,
        context: &EvaluationContext,
        flag: &FeatureFlag,
    ) -> FlagResult<bool> {
        // Parse and evaluate the condition
        // This is a simplified implementation

        let condition = condition.trim();

        // Handle percentage conditions
        if condition.starts_with("percentage") {
            return self.evaluate_percentage_condition(condition, context, flag);
        }

        // Handle membership conditions (e.g., "user.id in beta_users")
        if condition.contains(" in ") {
            return self.evaluate_membership_condition(condition, context);
        }

        // Handle comparison conditions
        self.evaluate_comparison_condition(condition, context)
    }

    /// Evaluate percentage-based conditions
    fn evaluate_percentage_condition(
        &self,
        condition: &str,
        context: &EvaluationContext,
        flag: &FeatureFlag,
    ) -> FlagResult<bool> {
        // Parse "percentage < X"
        let parts: Vec<&str> = condition.split('<').collect();
        if parts.len() != 2 {
            return Err(FlagError::InvalidCondition(condition.to_string()));
        }

        let threshold: u32 = parts[1].trim().parse()
            .map_err(|_| FlagError::InvalidCondition(condition.to_string()))?;

        let user_id = context.user.id.as_deref().unwrap_or("anonymous");
        let bucket = self.compute_bucket(user_id, &flag.key);

        Ok(bucket < threshold)
    }

    /// Compute consistent hash bucket (0-99)
    fn compute_bucket(&self, user_id: &str, flag_key: &str) -> u32 {
        let mut hasher = Sha256::new();
        hasher.update(format!("{}:{}", user_id, flag_key).as_bytes());
        let result = hasher.finalize();

        // Use first 4 bytes to compute bucket
        let value = u32::from_be_bytes([result[0], result[1], result[2], result[3]]);
        value % 100
    }

    /// Evaluate membership conditions
    fn evaluate_membership_condition(
        &self,
        condition: &str,
        context: &EvaluationContext,
    ) -> FlagResult<bool> {
        // Parse "field in list_name"
        let parts: Vec<&str> = condition.split(" in ").collect();
        if parts.len() != 2 {
            return Err(FlagError::InvalidCondition(condition.to_string()));
        }

        let field = parts[0].trim();
        let list_name = parts[1].trim();

        // Get field value
        let value = self.get_field_value(field, context)?;

        // Check against user list
        let lists = self.user_lists.read();
        if let Some(list) = lists.get(list_name) {
            return Ok(list.contains(&value));
        }

        Ok(false)
    }

    /// Evaluate comparison conditions
    fn evaluate_comparison_condition(
        &self,
        condition: &str,
        context: &EvaluationContext,
    ) -> FlagResult<bool> {
        // Simple parser for "field == value"
        for op in &["==", "!=", ">=", "<=", ">", "<"] {
            if condition.contains(op) {
                let parts: Vec<&str> = condition.split(op).collect();
                if parts.len() == 2 {
                    let field = parts[0].trim();
                    let expected = parts[1].trim().trim_matches('\'').trim_matches('"');

                    let actual = self.get_field_value(field, context)?;

                    return Ok(match *op {
                        "==" => actual == expected,
                        "!=" => actual != expected,
                        ">" => actual.parse::<f64>().ok() > expected.parse::<f64>().ok(),
                        ">=" => actual.parse::<f64>().ok() >= expected.parse::<f64>().ok(),
                        "<" => actual.parse::<f64>().ok() < expected.parse::<f64>().ok(),
                        "<=" => actual.parse::<f64>().ok() <= expected.parse::<f64>().ok(),
                        _ => false,
                    });
                }
            }
        }

        Err(FlagError::InvalidCondition(condition.to_string()))
    }

    /// Get a field value from context
    fn get_field_value(&self, field: &str, context: &EvaluationContext) -> FlagResult<String> {
        let parts: Vec<&str> = field.split('.').collect();
        if parts.len() < 2 {
            return Err(FlagError::InvalidCondition(format!("Invalid field: {}", field)));
        }

        match parts[0] {
            "user" => {
                let attr = parts[1];
                let value = match attr {
                    "id" => context.user.id.clone(),
                    "email" => context.user.email.clone(),
                    "plan" => context.user.plan.clone(),
                    "signup_date" => context.user.signup_date.clone(),
                    "is_staff" => Some(context.user.is_staff.to_string()),
                    _ => context.user.attributes.get(attr)
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                };
                value.ok_or_else(|| FlagError::InvalidCondition(format!("Unknown user field: {}", attr)))
            }
            "context" => {
                let attr = parts[1];
                context.context.get(attr)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| FlagError::InvalidCondition(format!("Unknown context field: {}", attr)))
            }
            "environment" => {
                context.system.environment.clone()
                    .ok_or_else(|| FlagError::InvalidCondition("Environment not set".to_string()))
            }
            "system" => {
                let attr = parts[1];
                match attr {
                    "environment" => context.system.environment.clone(),
                    "region" => context.system.region.clone(),
                    "version" => context.system.version.clone(),
                    "cpu_usage" => context.system.cpu_usage.map(|v| v.to_string()),
                    "memory_usage" => context.system.memory_usage.map(|v| v.to_string()),
                    _ => None,
                }.ok_or_else(|| FlagError::InvalidCondition(format!("Unknown system field: {}", attr)))
            }
            _ => Err(FlagError::InvalidCondition(format!("Unknown namespace: {}", parts[0]))),
        }
    }

    /// Register a user list for membership conditions
    pub fn register_user_list(&self, name: &str, users: Vec<String>) {
        self.user_lists.write().insert(name.to_string(), users);
    }

    /// List all flags
    pub fn list_flags(&self) -> FlagResult<Vec<FeatureFlag>> {
        self.storage.list()
    }

    /// Reload flags from storage
    pub fn reload(&self) -> FlagResult<()> {
        self.cache.write().clear();
        Ok(())
    }
}

/// Storage backend trait
pub trait FlagStorage: Send + Sync {
    /// Get a flag by key
    fn get(&self, key: &str) -> FlagResult<FeatureFlag>;

    /// List all flags
    fn list(&self) -> FlagResult<Vec<FeatureFlag>>;

    /// Save a flag
    fn save(&self, flag: &FeatureFlag) -> FlagResult<()>;

    /// Delete a flag
    fn delete(&self, key: &str) -> FlagResult<()>;
}

/// Evaluation history trait for analytics
pub trait EvaluationHistory: Send + Sync {
    fn record(&self, result: &EvaluationResult, context: &EvaluationContext) -> FlagResult<()>;
}

/// YAML file-based flag storage
pub struct YamlFlagStorage {
    flags: HashMap<String, FeatureFlag>,
    path: std::path::PathBuf,
}

impl YamlFlagStorage {
    pub fn load(path: impl AsRef<std::path::Path>) -> FlagResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| FlagError::StorageError(e.to_string()))?;

        let config: YamlFlagConfig = serde_yaml::from_str(&content)
            .map_err(|e| FlagError::ParseError(e.to_string()))?;

        let flags = config.flags.into_iter()
            .map(|(k, v)| (k, v))
            .collect();

        Ok(Self {
            flags,
            path: path.as_ref().to_path_buf(),
        })
    }
}

#[derive(Debug, Deserialize)]
struct YamlFlagConfig {
    version: String,
    flags: HashMap<String, FeatureFlag>,
}

impl FlagStorage for YamlFlagStorage {
    fn get(&self, key: &str) -> FlagResult<FeatureFlag> {
        self.flags.get(key)
            .cloned()
            .ok_or_else(|| FlagError::FlagNotFound(key.to_string()))
    }

    fn list(&self) -> FlagResult<Vec<FeatureFlag>> {
        Ok(self.flags.values().cloned().collect())
    }

    fn save(&self, _flag: &FeatureFlag) -> FlagResult<()> {
        // Would write back to YAML file
        Err(FlagError::StorageError("YAML storage is read-only".to_string()))
    }

    fn delete(&self, _key: &str) -> FlagResult<()> {
        Err(FlagError::StorageError("YAML storage is read-only".to_string()))
    }
}
```

### 4.3 Macros for Compile-Time Safety

```rust
//! Compile-time safe feature flag access

/// Macro for type-safe boolean flag access
#[macro_export]
macro_rules! feature_enabled {
    ($engine:expr, $flag:literal, $context:expr) => {{
        // This creates compile-time documentation of flag usage
        const _FLAG_KEY: &str = $flag;
        $engine.is_enabled($flag, $context)
    }};
}

/// Macro for defining flags at compile time
#[macro_export]
macro_rules! define_flags {
    (
        $(
            $name:ident: $type:ty = $default:expr;
        )*
    ) => {
        pub mod flags {
            $(
                pub const $name: &str = stringify!($name);
            )*
        }
    };
}

// Example usage
define_flags! {
    NEW_POWERCOMBO_V2: bool = false;
    ADVANCED_THINKTOOLS: bool = false;
    TRACE_SAMPLING_RATE: f64 = 10.0;
}

// Usage in code:
// if feature_enabled!(engine, flags::NEW_POWERCOMBO_V2, &ctx) { ... }
```

---

## 5. Python Bindings

### 5.1 PyO3 Bindings

```rust
//! Python bindings for feature flags

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
pub struct PyFlagEngine {
    inner: std::sync::Arc<FlagEngine>,
}

#[pymethods]
impl PyFlagEngine {
    #[new]
    #[pyo3(signature = (config_path=None))]
    fn new(config_path: Option<&str>) -> PyResult<Self> {
        let path = config_path.unwrap_or("config/feature_flags.yaml");
        let engine = FlagEngine::from_yaml(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(Self {
            inner: std::sync::Arc::new(engine),
        })
    }

    /// Check if a boolean flag is enabled
    fn is_enabled(&self, key: &str, context: Option<&PyDict>) -> bool {
        let ctx = context
            .map(|d| dict_to_context(d))
            .unwrap_or_default();

        self.inner.is_enabled(key, &ctx)
    }

    /// Evaluate a flag and return the value
    fn evaluate(&self, key: &str, context: Option<&PyDict>) -> PyResult<PyObject> {
        let ctx = context
            .map(|d| dict_to_context(d))
            .unwrap_or_default();

        let result = self.inner.evaluate(key, &ctx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            match result.value {
                FlagValue::Boolean(b) => Ok(b.into_py(py)),
                FlagValue::String(s) => Ok(s.into_py(py)),
                FlagValue::Number(n) => Ok(n.into_py(py)),
                FlagValue::Json(j) => {
                    let s = serde_json::to_string(&j)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                    Ok(s.into_py(py))
                }
            }
        })
    }

    /// Get experiment variant for a user
    fn get_variant(&self, key: &str, context: Option<&PyDict>) -> PyResult<String> {
        let ctx = context
            .map(|d| dict_to_context(d))
            .unwrap_or_default();

        self.inner.get_variant(key, &ctx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// List all flags
    fn list_flags(&self) -> PyResult<Vec<String>> {
        let flags = self.inner.list_flags()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(flags.iter().map(|f| f.key.clone()).collect())
    }

    /// Register a user list
    fn register_user_list(&self, name: &str, users: Vec<String>) {
        self.inner.register_user_list(name, users);
    }
}

fn dict_to_context(dict: &PyDict) -> EvaluationContext {
    // Convert Python dict to EvaluationContext
    // Implementation details omitted for brevity
    EvaluationContext::default()
}

#[pymodule]
fn reasonkit_flags(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFlagEngine>()?;
    Ok(())
}
```

### 5.2 Pure Python API

```python
"""
ReasonKit Feature Flags - Python Client

High-level Python API for feature flag evaluation.
"""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class UserContext:
    """User attributes for flag evaluation."""
    id: Optional[str] = None
    email: Optional[str] = None
    plan: Optional[str] = None
    signup_date: Optional[str] = None
    is_staff: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemContext:
    """System attributes for flag evaluation."""
    environment: Optional[str] = None
    region: Optional[str] = None
    version: Optional[str] = None


@dataclass
class EvaluationContext:
    """Complete evaluation context."""
    user: UserContext = field(default_factory=UserContext)
    context: Dict[str, Any] = field(default_factory=dict)
    system: SystemContext = field(default_factory=SystemContext)


class FeatureFlagClient:
    """
    Client for evaluating feature flags.

    Example usage:

        client = FeatureFlagClient()

        ctx = EvaluationContext(
            user=UserContext(id="user_123", plan="enterprise"),
            system=SystemContext(environment="production")
        )

        if client.is_enabled("new_powercombo_v2", ctx):
            # Use new feature
            pass
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        remote_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the feature flag client.

        Args:
            config_path: Path to local YAML config file
            remote_url: URL for remote flag service (enterprise)
            api_key: API key for remote service
        """
        # Use Rust engine via PyO3 bindings
        from reasonkit import PyFlagEngine

        self._engine = PyFlagEngine(config_path)
        self._remote_url = remote_url
        self._api_key = api_key

    def is_enabled(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
    ) -> bool:
        """
        Check if a boolean flag is enabled.

        Args:
            flag_key: The flag identifier
            context: Evaluation context with user/system attributes

        Returns:
            True if the flag is enabled, False otherwise
        """
        ctx_dict = self._context_to_dict(context)
        return self._engine.is_enabled(flag_key, ctx_dict)

    def evaluate(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
    ) -> Any:
        """
        Evaluate a flag and return its value.

        Args:
            flag_key: The flag identifier
            context: Evaluation context

        Returns:
            The flag value (bool, str, number, or dict)
        """
        ctx_dict = self._context_to_dict(context)
        return self._engine.evaluate(flag_key, ctx_dict)

    def get_variant(
        self,
        experiment_key: str,
        context: Optional[EvaluationContext] = None,
    ) -> str:
        """
        Get the variant assignment for an experiment.

        Args:
            experiment_key: The experiment flag identifier
            context: Evaluation context

        Returns:
            Variant name (e.g., "control", "treatment")
        """
        ctx_dict = self._context_to_dict(context)
        return self._engine.get_variant(experiment_key, ctx_dict)

    def list_flags(self) -> list[str]:
        """List all available flag keys."""
        return self._engine.list_flags()

    def register_user_list(self, name: str, users: list[str]) -> None:
        """
        Register a user list for membership conditions.

        Args:
            name: List name (e.g., "beta_users")
            users: List of user IDs
        """
        self._engine.register_user_list(name, users)

    def _context_to_dict(self, context: Optional[EvaluationContext]) -> Optional[dict]:
        if context is None:
            return None

        return {
            "user": {
                "id": context.user.id,
                "email": context.user.email,
                "plan": context.user.plan,
                "signup_date": context.user.signup_date,
                "is_staff": context.user.is_staff,
                **context.user.attributes,
            },
            "context": context.context,
            "system": {
                "environment": context.system.environment,
                "region": context.system.region,
                "version": context.system.version,
            },
        }


# Convenience functions
_default_client: Optional[FeatureFlagClient] = None


def init(config_path: Optional[str] = None) -> None:
    """Initialize the default feature flag client."""
    global _default_client
    _default_client = FeatureFlagClient(config_path)


def is_enabled(flag_key: str, context: Optional[EvaluationContext] = None) -> bool:
    """Check if a flag is enabled using the default client."""
    if _default_client is None:
        init()
    return _default_client.is_enabled(flag_key, context)


def get_variant(experiment_key: str, context: Optional[EvaluationContext] = None) -> str:
    """Get experiment variant using the default client."""
    if _default_client is None:
        init()
    return _default_client.get_variant(experiment_key, context)
```

---

## 6. CLI Integration

### 6.1 CLI Commands

```bash
# ═══════════════════════════════════════════════════════════════════════════
# FEATURE FLAG CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

# List all flags
rk-core features list
rk-core features list --type release
rk-core features list --type experiment
rk-core features list --tag beta

# Check a specific flag
rk-core features check new_powercombo_v2
rk-core features check new_powercombo_v2 --user-id user_123 --plan enterprise

# Evaluate with full context
rk-core features evaluate advanced_thinktools \
  --user-id user_123 \
  --user-email "test@example.com" \
  --plan pro \
  --environment production

# Get experiment variant
rk-core features variant profile_comparison_exp --user-id user_123

# Show flag details
rk-core features info new_powercombo_v2

# ═══════════════════════════════════════════════════════════════════════════
# USE FLAGS IN THINK COMMAND
# ═══════════════════════════════════════════════════════════════════════════

# Enable specific feature for this request
rk-core think --feature new_powercombo_v2 "Your question"

# Check which features are active
rk-core think --list-features

# Disable a feature for testing
rk-core think --no-feature new_powercombo_v2 "Your question"

# ═══════════════════════════════════════════════════════════════════════════
# ADMIN COMMANDS (requires admin privileges)
# ═══════════════════════════════════════════════════════════════════════════

# Create a new flag
rk-admin flags create new_flag \
  --type release \
  --description "Description of the flag" \
  --default false

# Update flag rollout percentage
rk-admin flags update new_flag --add-rule "percentage < 50"

# Enable/disable flag
rk-admin flags enable new_flag
rk-admin flags disable new_flag

# Delete a flag
rk-admin flags delete old_flag

# Register user list
rk-admin flags register-list beta_users --file beta_users.txt
rk-admin flags register-list beta_users --users user1,user2,user3

# View evaluation history
rk-admin flags history new_flag --last 100
rk-admin flags history --user-id user_123 --since 2025-01-01

# Export flags
rk-admin flags export --format yaml > flags_backup.yaml
rk-admin flags export --format json > flags_backup.json

# Import flags
rk-admin flags import flags_backup.yaml
```

### 6.2 CLI Implementation (Clap)

```rust
#[derive(Subcommand)]
enum FeatureCommands {
    /// List all feature flags
    List {
        /// Filter by flag type
        #[arg(long)]
        r#type: Option<String>,

        /// Filter by tag
        #[arg(long)]
        tag: Option<String>,

        /// Output format (table, json, yaml)
        #[arg(short, long, default_value = "table")]
        format: String,
    },

    /// Check if a flag is enabled
    Check {
        /// Flag key
        key: String,

        /// User ID for evaluation
        #[arg(long)]
        user_id: Option<String>,

        /// User plan
        #[arg(long)]
        plan: Option<String>,

        /// Environment
        #[arg(long, default_value = "development")]
        environment: String,
    },

    /// Evaluate a flag with full context
    Evaluate {
        /// Flag key
        key: String,

        /// User ID
        #[arg(long)]
        user_id: Option<String>,

        /// User email
        #[arg(long)]
        user_email: Option<String>,

        /// User plan
        #[arg(long)]
        plan: Option<String>,

        /// Environment
        #[arg(long)]
        environment: Option<String>,

        /// Additional context (JSON)
        #[arg(long)]
        context: Option<String>,
    },

    /// Get experiment variant assignment
    Variant {
        /// Experiment flag key
        key: String,

        /// User ID
        #[arg(long)]
        user_id: String,
    },

    /// Show detailed flag information
    Info {
        /// Flag key
        key: String,
    },
}

async fn handle_features_command(cmd: FeatureCommands) -> anyhow::Result<()> {
    let engine = FlagEngine::from_yaml("config/feature_flags.yaml")?;

    match cmd {
        FeatureCommands::List { r#type, tag, format } => {
            let flags = engine.list_flags()?;

            let filtered: Vec<_> = flags.into_iter()
                .filter(|f| {
                    let type_match = r#type.as_ref()
                        .map(|t| format!("{:?}", f.flag_type).to_lowercase() == t.to_lowercase())
                        .unwrap_or(true);
                    let tag_match = tag.as_ref()
                        .map(|t| f.tags.contains(t))
                        .unwrap_or(true);
                    type_match && tag_match
                })
                .collect();

            match format.as_str() {
                "json" => println!("{}", serde_json::to_string_pretty(&filtered)?),
                "yaml" => println!("{}", serde_yaml::to_string(&filtered)?),
                _ => {
                    println!("{:<30} {:<12} {:<10} {}", "KEY", "TYPE", "DEFAULT", "DESCRIPTION");
                    println!("{}", "-".repeat(80));
                    for flag in filtered {
                        println!("{:<30} {:<12} {:<10} {}",
                            flag.key,
                            format!("{:?}", flag.flag_type).to_lowercase(),
                            format!("{:?}", flag.default),
                            flag.description.chars().take(40).collect::<String>()
                        );
                    }
                }
            }
        }

        FeatureCommands::Check { key, user_id, plan, environment } => {
            let context = EvaluationContext {
                user: UserContext {
                    id: user_id,
                    plan,
                    ..Default::default()
                },
                system: SystemContext {
                    environment: Some(environment),
                    ..Default::default()
                },
                ..Default::default()
            };

            let enabled = engine.is_enabled(&key, &context);
            println!("Flag '{}' is: {}", key, if enabled { "ENABLED" } else { "DISABLED" });
        }

        FeatureCommands::Evaluate { key, user_id, user_email, plan, environment, context } => {
            let ctx = EvaluationContext {
                user: UserContext {
                    id: user_id,
                    email: user_email,
                    plan,
                    ..Default::default()
                },
                system: SystemContext {
                    environment,
                    ..Default::default()
                },
                context: context
                    .map(|c| serde_json::from_str(&c).unwrap_or_default())
                    .unwrap_or_default(),
            };

            let result = engine.evaluate(&key, &ctx)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        FeatureCommands::Variant { key, user_id } => {
            let context = EvaluationContext {
                user: UserContext {
                    id: Some(user_id.clone()),
                    ..Default::default()
                },
                ..Default::default()
            };

            let variant = engine.get_variant(&key, &context)?;
            println!("User '{}' is assigned to variant: {}", user_id, variant);
        }

        FeatureCommands::Info { key } => {
            let flag = engine.get_flag(&key)?;
            println!("{}", serde_yaml::to_string(&flag)?);
        }
    }

    Ok(())
}
```

---

## 7. Targeting Rules

### 7.1 User Attributes

| Attribute          | Type    | Description            | Example                           |
| ------------------ | ------- | ---------------------- | --------------------------------- |
| `user.id`          | String  | Unique user identifier | `"user_abc123"`                   |
| `user.email`       | String  | User email address     | `"user@example.com"`              |
| `user.plan`        | String  | Subscription plan      | `"free"`, `"pro"`, `"enterprise"` |
| `user.signup_date` | Date    | Account creation date  | `"2025-01-15"`                    |
| `user.is_staff`    | Boolean | Internal staff flag    | `true`, `false`                   |
| `user.*`           | Any     | Custom attributes      | Via `attributes` map              |

### 7.2 Context Attributes

| Attribute             | Type   | Description        | Example                   |
| --------------------- | ------ | ------------------ | ------------------------- |
| `context.sdk_version` | String | Client SDK version | `"2.1.0"`                 |
| `context.platform`    | String | Client platform    | `"cli"`, `"api"`, `"web"` |
| `context.request_id`  | String | Unique request ID  | `"req_xyz789"`            |
| `context.*`           | Any    | Custom context     | Arbitrary key-value       |

### 7.3 System Attributes

| Attribute             | Type   | Description            | Example                                      |
| --------------------- | ------ | ---------------------- | -------------------------------------------- |
| `system.environment`  | String | Deployment environment | `"development"`, `"staging"`, `"production"` |
| `system.region`       | String | Geographic region      | `"us-east-1"`, `"eu-west-1"`                 |
| `system.version`      | String | Application version    | `"1.2.3"`                                    |
| `system.cpu_usage`    | Number | Current CPU usage %    | `75.5`                                       |
| `system.memory_usage` | Number | Current memory usage % | `60.2`                                       |

### 7.4 Operators Reference

| Operator      | Description           | Example                                              |
| ------------- | --------------------- | ---------------------------------------------------- |
| `==`          | Equals                | `user.plan == 'enterprise'`                          |
| `!=`          | Not equals            | `user.plan != 'free'`                                |
| `>`           | Greater than          | `user.usage > 1000`                                  |
| `>=`          | Greater than or equal | `system.cpu_usage >= 80`                             |
| `<`           | Less than             | `percentage < 10`                                    |
| `<=`          | Less than or equal    | `user.age <= 30`                                     |
| `in`          | In list               | `user.id in beta_users`                              |
| `not_in`      | Not in list           | `user.id not_in blacklist`                           |
| `contains`    | String contains       | `user.email contains '@company.com'`                 |
| `starts_with` | String starts with    | `user.email starts_with 'admin'`                     |
| `ends_with`   | String ends with      | `user.email ends_with '@reasonkit.sh'`               |
| `and`         | Logical AND           | `user.plan == 'pro' and user.is_staff == false`      |
| `or`          | Logical OR            | `user.plan == 'enterprise' or user.is_staff == true` |
| `not`         | Logical NOT           | `not user.is_blocked`                                |

---

## 8. Rollout Strategies

### 8.1 Percentage Rollout

Gradual rollout based on consistent hashing of user ID.

```yaml
# Start at 0%, gradually increase
new_feature:
  type: release
  default: false
  rules:
    # Week 1: 1% canary
    - condition: "percentage < 1"
      value: true

    # Week 2: 5% early adopters
    # - condition: "percentage < 5"
    #   value: true

    # Week 3: 25% wider rollout
    # - condition: "percentage < 25"
    #   value: true

    # Week 4: 100% GA
    # - condition: "percentage < 100"
    #   value: true
```

**Rollout Schedule Example:**

| Day | Percentage | Condition                         |
| --- | ---------- | --------------------------------- |
| 1   | 1%         | `percentage < 1`                  |
| 3   | 5%         | `percentage < 5`                  |
| 7   | 10%        | `percentage < 10`                 |
| 14  | 25%        | `percentage < 25`                 |
| 21  | 50%        | `percentage < 50`                 |
| 28  | 100%       | `percentage < 100` or remove flag |

### 8.2 Canary Release

Small percentage with monitoring before wider rollout.

```yaml
new_protocol_engine:
  type: release
  default: false
  rules:
    # Internal testing first
    - condition: "user.is_staff == true"
      value: true

    # Then 1% canary in production
    - condition: "environment == 'production' and percentage < 1"
      value: true

    # Always on in staging
    - condition: "environment == 'staging'"
      value: true
```

### 8.3 Ring Deployment

Progressive rollout through user tiers.

```yaml
major_architecture_change:
  type: release
  default: false
  rules:
    # Ring 0: ReasonKit internal team
    - condition: "user.email ends_with '@reasonkit.sh'"
      value: true

    # Ring 1: Beta users
    - condition: "user.id in beta_users"
      value: true

    # Ring 2: Enterprise customers (high-touch)
    - condition: "user.plan == 'enterprise'"
      value: true

    # Ring 3: Pro customers
    - condition: "user.plan == 'pro'"
      value: true

    # Ring 4: All users
    # - condition: "percentage < 100"
    #   value: true
```

### 8.4 Geographic Rollout

Region-by-region rollout.

```yaml
new_data_center_routing:
  type: release
  default: false
  rules:
    # Start with US regions
    - condition: "system.region starts_with 'us-'"
      value: true

    # Then EU regions
    # - condition: "system.region starts_with 'eu-'"
    #   value: true

    # Finally Asia-Pacific
    # - condition: "system.region starts_with 'ap-'"
    #   value: true
```

---

## 9. A/B Testing Framework

### 9.1 Experiment Definition

```yaml
experiments:
  profile_optimization_exp:
    description: "Test optimized balanced profile vs current"
    type: experiment

    # Variant definitions
    variants:
      control:
        description: "Current balanced profile"
        config:
          profile: "balanced"
        weight: 50 # 50% of traffic

      treatment_a:
        description: "New balanced with extended token budget"
        config:
          profile: "balanced"
          token_budget: 12000
        weight: 25 # 25% of traffic

      treatment_b:
        description: "New balanced with higher confidence threshold"
        config:
          profile: "balanced"
          min_confidence: 0.85
        weight: 25 # 25% of traffic

    # Metrics to track
    metrics:
      primary:
        name: "reasoning_quality_score"
        type: "continuous"
        goal: "maximize"

      secondary:
        - name: "response_latency_p95"
          type: "continuous"
          goal: "minimize"
        - name: "token_usage"
          type: "continuous"
          goal: "minimize"
        - name: "user_satisfaction"
          type: "continuous"
          goal: "maximize"

    # Safety guardrails
    guardrails:
      - metric: "error_rate"
        threshold: 0.05
        operator: "greater_than"
        action: "halt"
        message: "Error rate exceeded 5%, halting experiment"

      - metric: "p99_latency_ms"
        threshold: 5000
        operator: "greater_than"
        action: "alert"
        message: "P99 latency exceeded 5s"

    # Experiment parameters
    sample_size: 2000
    duration_days: 14
    min_sample_per_variant: 500

    # Targeting (who is eligible)
    targeting:
      - condition: "user.plan in ['pro', 'enterprise']"

    # Exclusions
    exclusions:
      - condition: "user.is_staff == true"
      - condition: "user.id in experiment_blacklist"
```

### 9.2 Statistical Analysis

```rust
//! A/B testing statistical analysis

use statrs::distribution::{ContinuousCDF, Normal};

/// Result of an A/B test analysis
#[derive(Debug, Serialize)]
pub struct ExperimentAnalysis {
    /// Experiment key
    pub experiment_key: String,

    /// Analysis timestamp
    pub analyzed_at: DateTime<Utc>,

    /// Sample sizes per variant
    pub sample_sizes: HashMap<String, usize>,

    /// Metric results
    pub metrics: Vec<MetricAnalysis>,

    /// Winner determination
    pub winner: Option<String>,

    /// Confidence level
    pub confidence: f64,

    /// Recommendation
    pub recommendation: ExperimentRecommendation,
}

#[derive(Debug, Serialize)]
pub struct MetricAnalysis {
    pub metric_name: String,
    pub variant_stats: HashMap<String, VariantStats>,
    pub p_value: f64,
    pub is_significant: bool,
    pub effect_size: f64,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Serialize)]
pub struct VariantStats {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: Percentiles,
}

#[derive(Debug, Serialize)]
pub struct Percentiles {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum ExperimentRecommendation {
    /// Not enough data yet
    WaitForMoreData,

    /// Control is winning
    KeepControl,

    /// Treatment is winning
    DeployTreatment { variant: String },

    /// No significant difference
    NoSignificantDifference,

    /// Experiment should be stopped (guardrail violated)
    StopExperiment { reason: String },
}

/// A/B test analyzer
pub struct ExperimentAnalyzer {
    /// Significance level (default 0.05)
    pub significance_level: f64,

    /// Minimum effect size to care about
    pub min_effect_size: f64,
}

impl ExperimentAnalyzer {
    pub fn new() -> Self {
        Self {
            significance_level: 0.05,
            min_effect_size: 0.01,
        }
    }

    /// Calculate sample size needed for given effect size
    pub fn required_sample_size(
        &self,
        baseline_rate: f64,
        minimum_detectable_effect: f64,
        power: f64,
    ) -> usize {
        // Two-sample t-test power analysis
        let alpha = self.significance_level;
        let beta = 1.0 - power;

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_alpha = normal.inverse_cdf(1.0 - alpha / 2.0);
        let z_beta = normal.inverse_cdf(1.0 - beta);

        let p1 = baseline_rate;
        let p2 = baseline_rate + minimum_detectable_effect;
        let p_pooled = (p1 + p2) / 2.0;

        let numerator = (z_alpha * (2.0 * p_pooled * (1.0 - p_pooled)).sqrt()
            + z_beta * (p1 * (1.0 - p1) + p2 * (1.0 - p2)).sqrt()).powi(2);
        let denominator = (p2 - p1).powi(2);

        (numerator / denominator).ceil() as usize
    }

    /// Perform two-sample t-test
    pub fn t_test(
        &self,
        control: &[f64],
        treatment: &[f64],
    ) -> (f64, bool) {
        let n1 = control.len() as f64;
        let n2 = treatment.len() as f64;

        let mean1: f64 = control.iter().sum::<f64>() / n1;
        let mean2: f64 = treatment.iter().sum::<f64>() / n2;

        let var1: f64 = control.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let var2: f64 = treatment.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

        let pooled_se = ((var1 / n1) + (var2 / n2)).sqrt();
        let t_stat = (mean2 - mean1) / pooled_se;

        // Approximate p-value using normal distribution
        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));

        let is_significant = p_value < self.significance_level;

        (p_value, is_significant)
    }

    /// Calculate Cohen's d effect size
    pub fn effect_size(&self, control: &[f64], treatment: &[f64]) -> f64 {
        let mean1: f64 = control.iter().sum::<f64>() / control.len() as f64;
        let mean2: f64 = treatment.iter().sum::<f64>() / treatment.len() as f64;

        let var1: f64 = control.iter().map(|x| (x - mean1).powi(2)).sum::<f64>()
            / (control.len() - 1) as f64;
        let var2: f64 = treatment.iter().map(|x| (x - mean2).powi(2)).sum::<f64>()
            / (treatment.len() - 1) as f64;

        let pooled_std = ((var1 + var2) / 2.0).sqrt();

        (mean2 - mean1) / pooled_std
    }
}
```

### 9.3 Experiment Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT LIFECYCLE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DESIGN                                                      │
│     ├── Define hypothesis                                       │
│     ├── Choose metrics (primary + secondary)                    │
│     ├── Set guardrails                                          │
│     ├── Calculate required sample size                          │
│     └── Define targeting/exclusions                             │
│                                                                 │
│  2. SETUP                                                       │
│     ├── Create experiment flag                                  │
│     ├── Implement variant logic                                 │
│     ├── Set up metric collection                                │
│     └── Configure monitoring/alerts                             │
│                                                                 │
│  3. RUNNING                                                     │
│     ├── Monitor guardrails continuously                         │
│     ├── Check for sample ratio mismatch (SRM)                   │
│     ├── Review early results (with caution!)                    │
│     └── Adjust if guardrail violations                          │
│                                                                 │
│  4. ANALYSIS                                                    │
│     ├── Wait for minimum sample size                            │
│     ├── Check statistical significance                          │
│     ├── Calculate effect size and CI                            │
│     ├── Review secondary metrics                                │
│     └── Consider practical significance                         │
│                                                                 │
│  5. DECISION                                                    │
│     ├── Winner: Deploy treatment, remove flag                   │
│     ├── Loser: Keep control, remove flag                        │
│     ├── Inconclusive: Extend or redesign                        │
│     └── Document learnings                                      │
│                                                                 │
│  6. CLEANUP                                                     │
│     ├── Remove experiment flag                                  │
│     ├── Clean up variant code paths                             │
│     └── Archive experiment results                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Administration

### 10.1 Admin Dashboard UI Specification

```yaml
dashboard:
  pages:
    flags_list:
      title: "Feature Flags"
      components:
        - type: table
          columns:
            - key: Flag Key
            - type: Type
            - enabled: Status
            - rollout: "Rollout %"
            - updated_at: "Last Updated"
          actions:
            - edit
            - toggle_enable
            - view_history
          filters:
            - type: ["release", "experiment", "ops", "permission"]
            - status: ["enabled", "disabled"]
            - tag: [dynamic]

    flag_detail:
      title: "Flag: {key}"
      sections:
        - overview:
            fields: [description, type, owner, lifecycle]
        - targeting:
            rules_editor: true
            rule_preview: true
        - evaluations:
            chart: "evaluations_over_time"
            recent_evaluations: 50
        - history:
            changes: 100

    experiments:
      title: "Experiments"
      components:
        - active_experiments_list
        - experiment_results
        - scheduled_experiments

    experiment_detail:
      sections:
        - setup:
            variants: true
            metrics: true
            guardrails: true
        - results:
            significance_chart: true
            variant_comparison: true
            guardrail_status: true
        - analysis:
            winner_recommendation: true
            statistical_summary: true
```

### 10.2 REST API

```yaml
# OpenAPI specification for Feature Flag API
openapi: "3.0.3"
info:
  title: ReasonKit Feature Flag API
  version: "1.0.0"

paths:
  /flags:
    get:
      summary: List all flags
      parameters:
        - name: type
          in: query
          schema:
            type: string
            enum: [release, experiment, ops, permission]
        - name: tag
          in: query
          schema:
            type: string
      responses:
        200:
          description: List of flags
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/FeatureFlag"

    post:
      summary: Create a new flag
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateFlagRequest"
      responses:
        201:
          description: Flag created

  /flags/{key}:
    get:
      summary: Get flag details
      responses:
        200:
          description: Flag details

    put:
      summary: Update a flag
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/UpdateFlagRequest"
      responses:
        200:
          description: Flag updated

    delete:
      summary: Delete a flag
      responses:
        204:
          description: Flag deleted

  /flags/{key}/evaluate:
    post:
      summary: Evaluate a flag
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/EvaluationContext"
      responses:
        200:
          description: Evaluation result
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/EvaluationResult"

  /flags/{key}/history:
    get:
      summary: Get flag change history
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 100
      responses:
        200:
          description: Change history

  /user-lists:
    get:
      summary: List user lists
    post:
      summary: Create user list

  /user-lists/{name}:
    get:
      summary: Get user list
    put:
      summary: Update user list
    delete:
      summary: Delete user list

components:
  schemas:
    FeatureFlag:
      type: object
      properties:
        key:
          type: string
        description:
          type: string
        type:
          type: string
          enum: [release, experiment, ops, permission]
        default:
          oneOf:
            - type: boolean
            - type: string
            - type: number
        enabled:
          type: boolean
        rules:
          type: array
          items:
            $ref: "#/components/schemas/FlagRule"

    FlagRule:
      type: object
      properties:
        condition:
          type: string
        value:
          oneOf:
            - type: boolean
            - type: string
            - type: number
        priority:
          type: integer

    EvaluationContext:
      type: object
      properties:
        user:
          type: object
          properties:
            id:
              type: string
            email:
              type: string
            plan:
              type: string
        context:
          type: object
        system:
          type: object

    EvaluationResult:
      type: object
      properties:
        flag_key:
          type: string
        value:
          oneOf:
            - type: boolean
            - type: string
            - type: number
        is_default:
          type: boolean
        matched_rule_index:
          type: integer
        variant:
          type: string
```

---

## 11. Monitoring and Analytics

### 11.1 Flag Usage Metrics

```yaml
metrics:
  # Evaluation metrics
  flag_evaluations_total:
    type: counter
    labels: [flag_key, result, rule_matched]
    description: "Total number of flag evaluations"

  flag_evaluation_latency_seconds:
    type: histogram
    labels: [flag_key]
    buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    description: "Flag evaluation latency"

  # Rollout metrics
  flag_rollout_percentage:
    type: gauge
    labels: [flag_key]
    description: "Current rollout percentage"

  flag_enabled_users:
    type: gauge
    labels: [flag_key]
    description: "Estimated number of users with flag enabled"

  # Experiment metrics
  experiment_variant_assignments:
    type: counter
    labels: [experiment_key, variant]
    description: "Number of users assigned to each variant"

  experiment_metric_value:
    type: histogram
    labels: [experiment_key, variant, metric_name]
    description: "Metric values per experiment variant"

  # Error metrics
  flag_errors_total:
    type: counter
    labels: [flag_key, error_type]
    description: "Total number of flag evaluation errors"
```

### 11.2 Dashboards

```yaml
# Grafana dashboard definition
dashboards:
  feature_flags_overview:
    title: "Feature Flags Overview"
    rows:
      - title: "Evaluation Volume"
        panels:
          - type: graph
            title: "Evaluations per Minute"
            query: "sum(rate(flag_evaluations_total[1m])) by (flag_key)"

          - type: stat
            title: "Total Evaluations (24h)"
            query: "sum(increase(flag_evaluations_total[24h]))"

      - title: "Performance"
        panels:
          - type: heatmap
            title: "Evaluation Latency"
            query: "histogram_quantile(0.95, sum(rate(flag_evaluation_latency_seconds_bucket[5m])) by (le))"

          - type: graph
            title: "P99 Latency by Flag"
            query: "histogram_quantile(0.99, sum(rate(flag_evaluation_latency_seconds_bucket[5m])) by (flag_key, le))"

      - title: "Rollout Status"
        panels:
          - type: table
            title: "Active Rollouts"
            columns:
              - flag_key
              - percentage
              - enabled_count
              - last_evaluation

          - type: pie
            title: "Flag Distribution by Type"
            query: "count by (type) (flag_enabled)"

      - title: "Experiments"
        panels:
          - type: table
            title: "Active Experiments"
            columns:
              - experiment_key
              - variants
              - sample_sizes
              - status

          - type: graph
            title: "Variant Assignment Rate"
            query: "sum(rate(experiment_variant_assignments[5m])) by (experiment_key, variant)"

      - title: "Errors"
        panels:
          - type: stat
            title: "Error Rate"
            query: "sum(rate(flag_errors_total[5m])) / sum(rate(flag_evaluations_total[5m]))"
            thresholds:
              - value: 0.001
                color: yellow
              - value: 0.01
                color: red

          - type: log
            title: "Recent Errors"
            query: '{app="reasonkit"} |= "flag_error"'
```

### 11.3 Alerts

```yaml
alerts:
  feature_flag_high_error_rate:
    condition: |
      sum(rate(flag_errors_total[5m])) / sum(rate(flag_evaluations_total[5m])) > 0.01
    severity: critical
    message: "Feature flag error rate exceeds 1%"
    runbook: "docs/runbooks/flag-errors.md"

  feature_flag_slow_evaluations:
    condition: |
      histogram_quantile(0.99, sum(rate(flag_evaluation_latency_seconds_bucket[5m])) by (le)) > 0.1
    severity: warning
    message: "Feature flag P99 latency exceeds 100ms"

  experiment_sample_ratio_mismatch:
    condition: |
      # Detect if variant distribution deviates significantly from expected
      abs(
        sum(experiment_variant_assignments{variant="control"}) /
        sum(experiment_variant_assignments)
        - 0.5
      ) > 0.05
    severity: warning
    message: "Experiment sample ratio mismatch detected"

  experiment_guardrail_violation:
    condition: |
      experiment_guardrail_status == 1
    severity: critical
    message: "Experiment guardrail violated - automatic halt"

  stale_flag_warning:
    condition: |
      time() - flag_last_updated_timestamp > 86400 * 90
    severity: info
    message: "Feature flag hasn't been updated in 90 days, consider cleanup"
```

---

## 12. Best Practices

### 12.1 Naming Conventions

```yaml
naming:
  format: "{scope}_{component}_{feature}_{version}"

  examples:
    # Release flags
    - "release_thinktool_powercombo_v2"
    - "release_api_graphql_support"
    - "release_ui_dark_mode"

    # Experiment flags
    - "exp_profile_token_budget_optimization"
    - "exp_onboarding_flow_v3"

    # Ops flags
    - "ops_maintenance_mode"
    - "ops_rate_limit_aggressive"
    - "ops_debug_verbose_logging"

    # Permission flags
    - "perm_advanced_thinktools"
    - "perm_beta_features"
    - "perm_custom_protocols"

  anti_patterns:
    - "test" # Too vague
    - "new_feature" # Non-descriptive
    - "temp_fix" # Should be documented properly
    - "johns_flag" # Personal names
```

### 12.2 Lifecycle Management

```yaml
lifecycle:
  stages:
    development:
      duration: "Varies"
      description: "Flag under development, may change frequently"

    testing:
      duration: "1-2 weeks"
      description: "Flag being tested in staging/preview"

    rollout:
      duration: "2-4 weeks"
      description: "Progressive rollout to production"

    stable:
      duration: "Until removal"
      description: "Flag is stable, feature fully rolled out"

    deprecated:
      duration: "2 weeks max"
      description: "Flag scheduled for removal"

    archived:
      description: "Flag removed, kept for historical reference"

  cleanup_rules:
    - "Release flags at 100% for > 2 weeks should be removed"
    - "Experiments completed for > 1 week should be cleaned up"
    - "Ops flags unused for > 30 days should be reviewed"
    - "Permission flags are permanent until plan changes"
```

### 12.3 Documentation Requirements

```yaml
documentation:
  required_fields:
    - key: "Unique identifier"
    - description: "What this flag controls"
    - owner: "Team or person responsible"
    - created: "When the flag was created"

  recommended_fields:
    - expected_removal: "When to remove (for release flags)"
    - jira_ticket: "Related ticket/issue"
    - runbook: "Link to operational runbook"
    - related_flags: "Dependencies or related flags"

  example:
    key: "release_thinktool_powercombo_v2"
    description: |
      Enables the new PowerCombo v2 implementation with improved
      cross-validation and 15% better token efficiency.
    owner: "thinktool-team"
    created: "2025-01-15"
    expected_removal: "2025-03-15"
    jira_ticket: "RK-1234"
    runbook: "docs/runbooks/powercombo-rollout.md"
    related_flags:
      - "exp_powercombo_token_optimization"
```

### 12.4 Code Integration Patterns

```rust
// GOOD: Check flag at entry points, not deep in logic
pub async fn process_query(&self, query: &str, context: &EvaluationContext) -> Result<Response> {
    // Check flag once at entry
    let use_new_engine = self.flags.is_enabled("release_new_engine", context);

    if use_new_engine {
        self.new_engine.process(query).await
    } else {
        self.legacy_engine.process(query).await
    }
}

// BAD: Checking flag multiple times in a flow
pub async fn process_query_bad(&self, query: &str, context: &EvaluationContext) -> Result<Response> {
    let result = if self.flags.is_enabled("release_new_engine", context) {
        self.step1_new().await?
    } else {
        self.step1_legacy().await?
    };

    // Same flag checked again - wasteful and potentially inconsistent
    let final_result = if self.flags.is_enabled("release_new_engine", context) {
        self.step2_new(result).await?
    } else {
        self.step2_legacy(result).await?
    };

    Ok(final_result)
}

// GOOD: Use feature flag in configuration, not inline
pub struct EngineConfig {
    pub use_new_validation: bool,
    pub use_experimental_scoring: bool,
}

impl EngineConfig {
    pub fn from_flags(flags: &FlagEngine, context: &EvaluationContext) -> Self {
        Self {
            use_new_validation: flags.is_enabled("release_new_validation", context),
            use_experimental_scoring: flags.is_enabled("exp_scoring_v2", context),
        }
    }
}

// GOOD: Logging flag evaluation for debugging
pub async fn handle_request(&self, req: Request, context: &EvaluationContext) -> Response {
    let flags_snapshot = FlagsSnapshot {
        new_feature: self.flags.is_enabled("release_new_feature", context),
        beta_mode: self.flags.is_enabled("perm_beta_features", context),
    };

    tracing::info!(
        user_id = %context.user.id.as_deref().unwrap_or("anonymous"),
        flags = ?flags_snapshot,
        "Processing request with feature flags"
    );

    // Use flags_snapshot throughout the request
    self.process_with_flags(req, flags_snapshot).await
}
```

---

## 13. Integration Options

### 13.1 Build vs Buy Decision Matrix

| Factor          | Build (ReasonKit Native)         | Buy (LaunchDarkly/etc)    |
| --------------- | -------------------------------- | ------------------------- |
| **Control**     | Full control over implementation | Vendor-dependent          |
| **Cost**        | Development time                 | $25-150/1000 MAU/month    |
| **Performance** | Rust-native, sub-ms              | Network latency (10-50ms) |
| **Features**    | Build what we need               | Comprehensive out-of-box  |
| **Maintenance** | Ongoing investment               | Vendor maintained         |
| **Compliance**  | Full data control                | Depends on vendor         |
| **Integration** | Native to ReasonKit              | SDK integration           |

**Recommendation**: Build native for CLI and core library, integrate with vendor for hosted/enterprise if needed.

### 13.2 OpenFeature Compliance

OpenFeature is an open standard for feature flag evaluation.

```rust
//! OpenFeature provider implementation

use openfeature::{
    EvaluationContext as OFContext,
    EvaluationResult as OFResult,
    Provider,
    ProviderMetadata,
};

/// ReasonKit OpenFeature Provider
pub struct ReasonKitProvider {
    engine: Arc<FlagEngine>,
}

impl Provider for ReasonKitProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            name: "reasonkit".to_string(),
        }
    }

    fn resolve_boolean_value(
        &self,
        flag_key: &str,
        default_value: bool,
        evaluation_context: &OFContext,
    ) -> OFResult<bool> {
        let context = self.convert_context(evaluation_context);

        match self.engine.evaluate(flag_key, &context) {
            Ok(result) => match result.value {
                FlagValue::Boolean(b) => OFResult::ok(b),
                _ => OFResult::err(default_value, "Type mismatch"),
            },
            Err(e) => OFResult::err(default_value, &e.to_string()),
        }
    }

    fn resolve_string_value(
        &self,
        flag_key: &str,
        default_value: String,
        evaluation_context: &OFContext,
    ) -> OFResult<String> {
        let context = self.convert_context(evaluation_context);

        match self.engine.evaluate(flag_key, &context) {
            Ok(result) => match result.value {
                FlagValue::String(s) => OFResult::ok(s),
                _ => OFResult::err(default_value, "Type mismatch"),
            },
            Err(e) => OFResult::err(default_value, &e.to_string()),
        }
    }

    // ... other resolve methods

    fn convert_context(&self, of_ctx: &OFContext) -> EvaluationContext {
        // Convert OpenFeature context to ReasonKit context
        EvaluationContext {
            user: UserContext {
                id: of_ctx.targeting_key.clone(),
                ..Default::default()
            },
            context: of_ctx.attributes.clone(),
            ..Default::default()
        }
    }
}
```

### 13.3 Migration Path from LaunchDarkly

```yaml
migration:
  phase_1_parallel:
    duration: "2 weeks"
    steps:
      - "Set up ReasonKit flag storage"
      - "Import existing flags from LaunchDarkly"
      - "Run both systems in parallel"
      - "Compare evaluation results"

  phase_2_shadow:
    duration: "2 weeks"
    steps:
      - "ReasonKit becomes primary evaluator"
      - "LaunchDarkly remains for comparison"
      - "Log any discrepancies"

  phase_3_cutover:
    duration: "1 week"
    steps:
      - "Disable LaunchDarkly SDK"
      - "ReasonKit is sole evaluator"
      - "Monitor for issues"

  phase_4_cleanup:
    duration: "1 week"
    steps:
      - "Remove LaunchDarkly SDK"
      - "Archive migration code"
      - "Cancel LaunchDarkly subscription"

  rollback_plan:
    - "Keep LaunchDarkly SDK available for 30 days post-migration"
    - "Maintain flag sync script for emergency use"
```

---

## 14. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task                                  | Priority | Effort | Owner     |
| ------------------------------------- | -------- | ------ | --------- |
| Define core types and traits          | P0       | 2d     | Core team |
| Implement YAML storage backend        | P0       | 1d     | Core team |
| Implement condition expression parser | P0       | 3d     | Core team |
| Basic evaluation engine               | P0       | 2d     | Core team |
| Unit tests for evaluation             | P0       | 1d     | Core team |

**Deliverable**: Basic flag evaluation working from YAML files.

### Phase 2: CLI Integration (Week 3)

| Task                               | Priority | Effort | Owner    |
| ---------------------------------- | -------- | ------ | -------- |
| Add `features` subcommand to CLI   | P0       | 2d     | CLI team |
| `--feature` flag for think command | P0       | 1d     | CLI team |
| Integration tests                  | P0       | 2d     | QA       |

**Deliverable**: Feature flags usable from CLI.

### Phase 3: Advanced Features (Week 4-5)

| Task                                       | Priority | Effort | Owner         |
| ------------------------------------------ | -------- | ------ | ------------- |
| Percentage rollout with consistent hashing | P0       | 2d     | Core team     |
| User list support                          | P1       | 1d     | Core team     |
| SQLite storage backend                     | P1       | 2d     | Core team     |
| Evaluation history/analytics               | P1       | 3d     | Core team     |
| Python bindings                            | P1       | 2d     | Bindings team |

**Deliverable**: Full-featured local flag system.

### Phase 4: Experimentation (Week 6-7)

| Task                        | Priority | Effort | Owner     |
| --------------------------- | -------- | ------ | --------- |
| Experiment flag type        | P1       | 3d     | Core team |
| Variant assignment          | P1       | 2d     | Core team |
| Statistical analysis module | P1       | 3d     | Data team |
| Guardrail monitoring        | P1       | 2d     | Core team |

**Deliverable**: A/B testing capability.

### Phase 5: Production Hardening (Week 8)

| Task                     | Priority | Effort | Owner     |
| ------------------------ | -------- | ------ | --------- |
| Performance optimization | P0       | 2d     | Core team |
| Caching layer            | P0       | 1d     | Core team |
| Prometheus metrics       | P1       | 1d     | Ops team  |
| Grafana dashboards       | P2       | 1d     | Ops team  |
| Documentation            | P0       | 2d     | Docs team |

**Deliverable**: Production-ready feature flag system.

---

## Appendix A: Example Configuration File

```yaml
# config/feature_flags.yaml
# ReasonKit Feature Flags Configuration

version: "1.0"

# User lists for membership conditions
user_lists:
  beta_users:
    - "user_001"
    - "user_002"
    - "user_003"
  internal_testers:
    - "alice@reasonkit.sh"
    - "bob@reasonkit.sh"

# Feature flags
flags:
  # ═══════════════════════════════════════════════════════════════════════
  # RELEASE FLAGS
  # ═══════════════════════════════════════════════════════════════════════

  release_powercombo_v2:
    description: "New PowerCombo with improved cross-validation"
    type: release
    default: false
    owner: "thinktool-team"
    lifecycle:
      status: active
      created: "2025-01-15"
      expected_removal: "2025-03-15"
    tags:
      - "thinktool"
      - "q1-2025"
    rules:
      # Staff always get new features
      - condition: "user.email ends_with '@reasonkit.sh'"
        value: true
      # Beta users
      - condition: "user.id in beta_users"
        value: true
      # 10% rollout
      - condition: "percentage < 10"
        value: true

  release_new_rag_engine:
    description: "Optimized RAG engine with better chunking"
    type: release
    default: false
    owner: "retrieval-team"
    rules:
      - condition: "environment == 'development'"
        value: true
      - condition: "environment == 'staging'"
        value: true
      # Production: 5% canary
      - condition: "environment == 'production' and percentage < 5"
        value: true

  # ═══════════════════════════════════════════════════════════════════════
  # EXPERIMENT FLAGS
  # ═══════════════════════════════════════════════════════════════════════

  exp_profile_optimization:
    description: "Test balanced profile with higher token budget"
    type: experiment
    default_variant: "control"
    variants:
      control:
        weight: 50
        config:
          profile: "balanced"
          token_budget: 8000
      treatment:
        weight: 50
        config:
          profile: "balanced"
          token_budget: 12000
    metrics:
      primary: "reasoning_quality_score"
      secondary:
        - "response_latency_p95"
        - "token_usage"
    guardrails:
      - metric: "error_rate"
        threshold: 0.05
        action: "halt"
    sample_size: 1000
    duration_days: 14
    targeting:
      - condition: "user.plan in ['pro', 'enterprise']"

  # ═══════════════════════════════════════════════════════════════════════
  # OPS FLAGS
  # ═══════════════════════════════════════════════════════════════════════

  ops_maintenance_mode:
    description: "Enable maintenance mode"
    type: ops
    default: false
    owner: "platform-team"

  ops_debug_mode:
    description: "Enable verbose debug logging"
    type: ops
    default: false
    rules:
      - condition: "environment == 'development'"
        value: true
      - condition: "user.email ends_with '@reasonkit.sh'"
        value: true

  ops_rate_limit_factor:
    description: "Rate limiting multiplier (1.0 = normal, 0.5 = half rate)"
    type: ops
    value_type: number
    default: 1.0
    rules:
      - condition: "system.cpu_usage > 80"
        value: 0.5

  # ═══════════════════════════════════════════════════════════════════════
  # PERMISSION FLAGS
  # ═══════════════════════════════════════════════════════════════════════

  perm_advanced_thinktools:
    description: "Access to AtomicBreak, HighReflect, and pro tools"
    type: permission
    default: false
    rules:
      - condition: "user.plan == 'enterprise'"
        value: true
      - condition: "user.plan == 'pro'"
        value: true

  perm_beta_features:
    description: "Access to beta features"
    type: permission
    default: false
    rules:
      - condition: "user.id in beta_users"
        value: true
      - condition: "user.signup_date < '2025-01-01'"
        value: true

  perm_custom_protocols:
    description: "Define custom ThinkTool protocols"
    type: permission
    default: false
    rules:
      - condition: "user.plan in ['pro', 'enterprise']"
        value: true

  perm_unlimited_tokens:
    description: "No token limits per request"
    type: permission
    default: false
    rules:
      - condition: "user.plan == 'enterprise'"
        value: true
```

---

## Appendix B: Quick Reference

```bash
# ═══════════════════════════════════════════════════════════════════════════
# FEATURE FLAG QUICK REFERENCE
# ═══════════════════════════════════════════════════════════════════════════

# List flags
rk-core features list
rk-core features list --type release

# Check flag
rk-core features check new_powercombo_v2 --user-id user_123

# Use flag in think
rk-core think --feature new_powercombo_v2 "Your question"

# Get experiment variant
rk-core features variant profile_exp --user-id user_123

# ═══════════════════════════════════════════════════════════════════════════
# CONDITION SYNTAX
# ═══════════════════════════════════════════════════════════════════════════

# Comparisons
user.plan == 'enterprise'
user.age >= 18
system.cpu_usage > 80

# Membership
user.id in beta_users
user.email ends_with '@company.com'

# Percentage (consistent hashing)
percentage < 10    # 10% rollout

# Logical
user.plan == 'pro' and user.is_verified == true
environment == 'staging' or user.is_staff == true

# ═══════════════════════════════════════════════════════════════════════════
# FLAG TYPES
# ═══════════════════════════════════════════════════════════════════════════

# release  - Feature rollout, kill switches
# experiment - A/B tests, multivariate tests
# ops      - Maintenance mode, debug flags, rate limits
# permission - Plan-based access, beta access
```

---

_Document Version: 1.0.0_
_Last Updated: 2025-12-28_
_Author: ReasonKit Engineering Team_
_License: Apache 2.0_
