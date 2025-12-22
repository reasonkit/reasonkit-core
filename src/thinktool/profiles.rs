//! Reasoning Profiles - Compositions of ThinkTool protocols
//!
//! Profiles chain multiple protocols together for different use cases:
//! - `quick`: Fast 2-step analysis (GigaThink → LaserLogic)
//! - `balanced`: Standard 4-module chain
//! - `deep`: Thorough analysis with meta-cognition
//! - `paranoid`: Maximum verification (95% confidence target)
//! - `decide`: Decision support focused
//! - `scientific`: Research and experiments

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;

/// A reasoning profile (composition of protocols)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningProfile {
    /// Profile identifier (e.g., "quick", "balanced", "paranoid")
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Description of what this profile does
    pub description: String,

    /// Protocols in execution order
    pub chain: Vec<ChainStep>,

    /// Expected confidence threshold
    pub min_confidence: f64,

    /// Typical token budget (hint)
    pub token_budget: Option<u32>,

    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

/// A step in a protocol chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStep {
    /// Protocol to execute
    pub protocol_id: String,

    /// Input mapping from previous step outputs
    /// Key: input field name, Value: source expression (e.g., "input.query", "steps.gigathink.perspectives")
    #[serde(default)]
    pub input_mapping: HashMap<String, String>,

    /// Condition to execute (optional)
    #[serde(default)]
    pub condition: Option<ChainCondition>,

    /// Override configuration for this step
    #[serde(default)]
    pub config_override: Option<StepConfigOverride>,
}

/// Conditions for conditional execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChainCondition {
    /// Always execute
    Always,

    /// Execute if previous step confidence below threshold
    ConfidenceBelow {
        /// Confidence threshold
        threshold: f64,
    },

    /// Execute if previous step confidence above threshold
    ConfidenceAbove {
        /// Confidence threshold
        threshold: f64,
    },

    /// Execute if a specific output field exists
    OutputExists {
        /// Step ID to check
        step_id: String,
        /// Field name to check
        field: String,
    },
}

impl Default for ChainCondition {
    fn default() -> Self {
        Self::Always
    }
}

/// Configuration overrides for a chain step
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepConfigOverride {
    /// Override temperature
    pub temperature: Option<f64>,

    /// Override max tokens
    pub max_tokens: Option<u32>,

    /// Override min confidence
    pub min_confidence: Option<f64>,
}

/// Registry of reasoning profiles
#[derive(Debug, Default)]
pub struct ProfileRegistry {
    profiles: HashMap<String, ReasoningProfile>,
}

impl ProfileRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with built-in profiles
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register_builtins();
        registry
    }

    /// Register built-in profiles
    pub fn register_builtins(&mut self) {
        self.register(builtin_quick());
        self.register(builtin_balanced());
        self.register(builtin_deep());
        self.register(builtin_paranoid());
        self.register(builtin_decide());
        self.register(builtin_scientific());
    }

    /// Register a profile
    pub fn register(&mut self, profile: ReasoningProfile) {
        self.profiles.insert(profile.id.clone(), profile);
    }

    /// Get a profile by ID
    pub fn get(&self, id: &str) -> Option<&ReasoningProfile> {
        self.profiles.get(id)
    }

    /// Check if profile exists
    pub fn contains(&self, id: &str) -> bool {
        self.profiles.contains_key(id)
    }

    /// List all profile IDs
    pub fn list_ids(&self) -> Vec<&str> {
        self.profiles.keys().map(|s| s.as_str()).collect()
    }

    /// List all profiles
    pub fn list(&self) -> Vec<&ReasoningProfile> {
        self.profiles.values().collect()
    }

    /// Get profile count
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BUILT-IN PROFILES
// ═══════════════════════════════════════════════════════════════════════════

/// Quick: Fast 2-step analysis
/// GigaThink → LaserLogic
fn builtin_quick() -> ReasoningProfile {
    ReasoningProfile {
        id: "quick".to_string(),
        name: "Quick Analysis".to_string(),
        description: "Fast 2-step analysis for rapid insights".to_string(),
        chain: vec![
            ChainStep {
                protocol_id: "gigathink".to_string(),
                input_mapping: HashMap::from([
                    ("query".to_string(), "input.query".to_string()),
                ]),
                condition: None,
                config_override: Some(StepConfigOverride {
                    max_tokens: Some(1000),
                    ..Default::default()
                }),
            },
            ChainStep {
                protocol_id: "laserlogic".to_string(),
                input_mapping: HashMap::from([
                    // Use synthesize step output (matches gigathink step ID)
                    ("argument".to_string(), "steps.gigathink.synthesize".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
        ],
        min_confidence: 0.70,
        token_budget: Some(3000),
        tags: vec!["fast".to_string(), "creative".to_string()],
    }
}

/// Balanced: Standard 4-module chain
/// GigaThink → LaserLogic → BedRock → ProofGuard
fn builtin_balanced() -> ReasoningProfile {
    ReasoningProfile {
        id: "balanced".to_string(),
        name: "Balanced Analysis".to_string(),
        description: "Standard 4-module chain for thorough but efficient analysis".to_string(),
        chain: vec![
            ChainStep {
                protocol_id: "gigathink".to_string(),
                input_mapping: HashMap::from([
                    ("query".to_string(), "input.query".to_string()),
                    ("context".to_string(), "input.context".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "laserlogic".to_string(),
                input_mapping: HashMap::from([
                    ("argument".to_string(), "steps.gigathink.synthesize".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "bedrock".to_string(),
                input_mapping: HashMap::from([
                    ("statement".to_string(), "steps.laserlogic.conclusion".to_string()),
                ]),
                condition: Some(ChainCondition::ConfidenceBelow { threshold: 0.9 }),
                config_override: None,
            },
            ChainStep {
                protocol_id: "proofguard".to_string(),
                input_mapping: HashMap::from([
                    ("claim".to_string(), "steps.bedrock.axioms".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
        ],
        min_confidence: 0.80,
        token_budget: Some(8000),
        tags: vec!["standard".to_string(), "thorough".to_string()],
    }
}

/// Deep: Thorough analysis with meta-cognition
/// GigaThink → LaserLogic → BedRock → ProofGuard → HighReflect (if available)
fn builtin_deep() -> ReasoningProfile {
    ReasoningProfile {
        id: "deep".to_string(),
        name: "Deep Analysis".to_string(),
        description: "Thorough analysis with first principles and verification".to_string(),
        chain: vec![
            ChainStep {
                protocol_id: "gigathink".to_string(),
                input_mapping: HashMap::from([
                    ("query".to_string(), "input.query".to_string()),
                    ("context".to_string(), "input.context".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "laserlogic".to_string(),
                input_mapping: HashMap::from([
                    ("argument".to_string(), "steps.gigathink.synthesize".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "bedrock".to_string(),
                input_mapping: HashMap::from([
                    ("statement".to_string(), "steps.laserlogic.conclusion".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "proofguard".to_string(),
                input_mapping: HashMap::from([
                    ("claim".to_string(), "steps.bedrock.axioms".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "brutalhonesty".to_string(),
                input_mapping: HashMap::from([
                    ("work".to_string(), "steps.proofguard.verdict".to_string()),
                ]),
                condition: Some(ChainCondition::ConfidenceBelow { threshold: 0.85 }),
                config_override: None,
            },
        ],
        min_confidence: 0.85,
        token_budget: Some(12000),
        tags: vec!["thorough".to_string(), "analytical".to_string()],
    }
}

/// Paranoid: Maximum verification (95% confidence target)
/// GigaThink → LaserLogic → BedRock → ProofGuard → BrutalHonesty → ProofGuard (2nd pass)
fn builtin_paranoid() -> ReasoningProfile {
    ReasoningProfile {
        id: "paranoid".to_string(),
        name: "Paranoid Verification".to_string(),
        description: "Maximum rigor with adversarial critique and multi-pass verification".to_string(),
        chain: vec![
            ChainStep {
                protocol_id: "gigathink".to_string(),
                input_mapping: HashMap::from([
                    ("query".to_string(), "input.query".to_string()),
                    ("context".to_string(), "input.context".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "laserlogic".to_string(),
                input_mapping: HashMap::from([
                    ("argument".to_string(), "steps.gigathink.synthesize".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "bedrock".to_string(),
                input_mapping: HashMap::from([
                    ("statement".to_string(), "steps.laserlogic.conclusion".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "proofguard".to_string(),
                input_mapping: HashMap::from([
                    ("claim".to_string(), "steps.bedrock.axioms".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "brutalhonesty".to_string(),
                input_mapping: HashMap::from([
                    ("work".to_string(), "steps.proofguard.verdict".to_string()),
                ]),
                condition: None,
                config_override: Some(StepConfigOverride {
                    temperature: Some(0.3), // Lower temp for more focused critique
                    ..Default::default()
                }),
            },
            // Second verification pass after critique
            ChainStep {
                protocol_id: "proofguard".to_string(),
                input_mapping: HashMap::from([
                    ("claim".to_string(), "steps.brutalhonesty.verdict".to_string()),
                ]),
                condition: Some(ChainCondition::ConfidenceBelow { threshold: 0.95 }),
                config_override: None,
            },
        ],
        min_confidence: 0.95,
        token_budget: Some(18000),
        tags: vec!["rigorous".to_string(), "verification".to_string(), "adversarial".to_string()],
    }
}

/// Decide: Decision support focused
/// LaserLogic → BedRock → BrutalHonesty
fn builtin_decide() -> ReasoningProfile {
    ReasoningProfile {
        id: "decide".to_string(),
        name: "Decision Support".to_string(),
        description: "Focused on evaluating options and making decisions".to_string(),
        chain: vec![
            ChainStep {
                protocol_id: "laserlogic".to_string(),
                input_mapping: HashMap::from([
                    ("argument".to_string(), "input.query".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "bedrock".to_string(),
                input_mapping: HashMap::from([
                    ("statement".to_string(), "steps.laserlogic.conclusion".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "brutalhonesty".to_string(),
                input_mapping: HashMap::from([
                    ("work".to_string(), "steps.bedrock.reconstruction".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
        ],
        min_confidence: 0.85,
        token_budget: Some(6000),
        tags: vec!["decision".to_string(), "analytical".to_string()],
    }
}

/// Scientific: Research and experiments
/// GigaThink → BedRock → ProofGuard
fn builtin_scientific() -> ReasoningProfile {
    ReasoningProfile {
        id: "scientific".to_string(),
        name: "Scientific Method".to_string(),
        description: "For research, hypothesis testing, and empirical analysis".to_string(),
        chain: vec![
            ChainStep {
                protocol_id: "gigathink".to_string(),
                input_mapping: HashMap::from([
                    ("query".to_string(), "input.query".to_string()),
                    ("constraints".to_string(), "input.constraints".to_string()),
                ]),
                condition: None,
                config_override: Some(StepConfigOverride {
                    temperature: Some(0.8), // Higher for hypothesis generation
                    ..Default::default()
                }),
            },
            ChainStep {
                protocol_id: "bedrock".to_string(),
                input_mapping: HashMap::from([
                    ("statement".to_string(), "steps.gigathink.synthesize".to_string()),
                    ("domain".to_string(), "input.domain".to_string()),
                ]),
                condition: None,
                config_override: None,
            },
            ChainStep {
                protocol_id: "proofguard".to_string(),
                input_mapping: HashMap::from([
                    ("claim".to_string(), "steps.bedrock.axioms".to_string()),
                    ("sources".to_string(), "input.sources".to_string()),
                ]),
                condition: None,
                config_override: Some(StepConfigOverride {
                    min_confidence: Some(0.85),
                    ..Default::default()
                }),
            },
        ],
        min_confidence: 0.85,
        token_budget: Some(8000),
        tags: vec!["research".to_string(), "empirical".to_string(), "verification".to_string()],
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_registry_creation() {
        let registry = ProfileRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_builtin_profiles() {
        let registry = ProfileRegistry::with_builtins();

        assert_eq!(registry.len(), 6);
        assert!(registry.contains("quick"));
        assert!(registry.contains("balanced"));
        assert!(registry.contains("deep"));
        assert!(registry.contains("paranoid"));
        assert!(registry.contains("decide"));
        assert!(registry.contains("scientific"));
    }

    #[test]
    fn test_get_profile() {
        let registry = ProfileRegistry::with_builtins();

        let quick = registry.get("quick").unwrap();
        assert_eq!(quick.chain.len(), 2);
        assert_eq!(quick.min_confidence, 0.70);

        let paranoid = registry.get("paranoid").unwrap();
        assert_eq!(paranoid.chain.len(), 6);
        assert_eq!(paranoid.min_confidence, 0.95);
    }

    #[test]
    fn test_profile_chain_structure() {
        let registry = ProfileRegistry::with_builtins();
        let balanced = registry.get("balanced").unwrap();

        // Verify chain order
        assert_eq!(balanced.chain[0].protocol_id, "gigathink");
        assert_eq!(balanced.chain[1].protocol_id, "laserlogic");
        assert_eq!(balanced.chain[2].protocol_id, "bedrock");
        assert_eq!(balanced.chain[3].protocol_id, "proofguard");

        // Verify conditional execution on bedrock
        assert!(matches!(
            balanced.chain[2].condition,
            Some(ChainCondition::ConfidenceBelow { threshold: 0.9 })
        ));
    }

    #[test]
    fn test_list_profiles() {
        let registry = ProfileRegistry::with_builtins();
        let ids = registry.list_ids();

        assert_eq!(ids.len(), 6);
        assert!(ids.contains(&"quick"));
    }
}
