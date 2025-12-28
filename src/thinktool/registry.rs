//! Protocol Registry
//!
//! Manages loading, storing, and retrieving ThinkTool protocols.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::protocol::Protocol;
use super::toml_loader;
use super::yaml_loader;
use crate::error::{Error, Result};

/// Registry of available ThinkTool protocols
#[derive(Debug, Default)]
pub struct ProtocolRegistry {
    /// Loaded protocols by ID
    protocols: HashMap<String, Protocol>,

    /// Protocol search paths
    search_paths: Vec<PathBuf>,
}

impl ProtocolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with default search paths
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // Add default search paths
        if let Ok(cwd) = std::env::current_dir() {
            registry.add_search_path(cwd.join("protocols"));
        }

        // User config directory
        if let Some(config_dir) = dirs_config_path() {
            registry.add_search_path(config_dir.join("reasonkit").join("protocols"));
        }

        registry
    }

    /// Add a search path for protocol files
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();
        if !self.search_paths.contains(&path) {
            self.search_paths.push(path);
        }
    }

    /// Load all protocols from search paths
    pub fn load_all(&mut self) -> Result<usize> {
        let mut count = 0;

        for path in &self.search_paths.clone() {
            if path.exists() && path.is_dir() {
                count += self.load_from_directory(path)?;
            }
        }

        Ok(count)
    }

    /// Load protocols from a specific directory
    pub fn load_from_directory(&mut self, dir: &Path) -> Result<usize> {
        let mut count = 0;

        let entries = std::fs::read_dir(dir).map_err(|e| Error::IoMessage {
            message: format!("Failed to read directory {}: {}", dir.display(), e),
        })?;

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str());

                match ext {
                    Some("json") => {
                        if let Ok(protocol) = self.load_json_file(&path) {
                            self.register(protocol)?;
                            count += 1;
                        }
                    }
                    Some("yaml") | Some("yml") => {
                        // Load YAML protocols using yaml_loader
                        match yaml_loader::load_from_yaml_file(&path) {
                            Ok(protocols) => {
                                for protocol in protocols {
                                    self.register(protocol)?;
                                    count += 1;
                                }
                                tracing::info!(
                                    "Loaded {} protocols from YAML: {}",
                                    count,
                                    path.display()
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to load YAML protocol {}: {}",
                                    path.display(),
                                    e
                                );
                            }
                        }
                    }
                    Some("toml") => {
                        // Load TOML protocols using toml_loader
                        match toml_loader::load_from_toml_file(&path) {
                            Ok(protocols) => {
                                for protocol in protocols {
                                    self.register(protocol)?;
                                    count += 1;
                                }
                                tracing::info!(
                                    "Loaded {} protocols from TOML: {}",
                                    count,
                                    path.display()
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to load TOML protocol {}: {}",
                                    path.display(),
                                    e
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(count)
    }

    /// Load a protocol from a JSON file
    fn load_json_file(&self, path: &Path) -> Result<Protocol> {
        let content = std::fs::read_to_string(path).map_err(|e| Error::IoMessage {
            message: format!("Failed to read {}: {}", path.display(), e),
        })?;

        let protocol: Protocol = serde_json::from_str(&content).map_err(|e| Error::Parse {
            message: format!("Failed to parse protocol {}: {}", path.display(), e),
        })?;

        // Validate the protocol
        protocol.validate().map_err(|errors| {
            Error::Validation(format!(
                "Invalid protocol {}: {}",
                protocol.id,
                errors.join(", ")
            ))
        })?;

        Ok(protocol)
    }

    /// Load protocols from the standard thinktools_v2.yaml file
    pub fn load_from_yaml(&mut self, path: &Path) -> Result<usize> {
        let protocols = yaml_loader::load_from_yaml_file(path)?;
        let count = protocols.len();

        for protocol in protocols {
            self.register(protocol)?;
        }

        Ok(count)
    }

    /// Register a protocol (by value)
    pub fn register(&mut self, protocol: Protocol) -> Result<()> {
        // Validate before registering
        protocol.validate().map_err(|errors| {
            Error::Validation(format!(
                "Invalid protocol {}: {}",
                protocol.id,
                errors.join(", ")
            ))
        })?;

        let id = protocol.id.clone();
        self.protocols.insert(id, protocol);
        Ok(())
    }

    /// Get a protocol by ID
    pub fn get(&self, id: &str) -> Option<&Protocol> {
        self.protocols.get(id)
    }

    /// Check if a protocol exists
    pub fn contains(&self, id: &str) -> bool {
        self.protocols.contains_key(id)
    }

    /// List all protocol IDs
    pub fn list_ids(&self) -> Vec<&str> {
        self.protocols.keys().map(|s| s.as_str()).collect()
    }

    /// List all protocols
    pub fn list(&self) -> Vec<&Protocol> {
        self.protocols.values().collect()
    }

    /// Get protocol count
    pub fn len(&self) -> usize {
        self.protocols.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.protocols.is_empty()
    }

    /// Remove a protocol by ID
    pub fn remove(&mut self, id: &str) -> Option<Protocol> {
        self.protocols.remove(id)
    }

    /// Clear all protocols
    pub fn clear(&mut self) {
        self.protocols.clear();
    }

    /// Register built-in protocols (hardcoded fallback)
    pub fn register_builtins(&mut self) -> Result<()> {
        // Try to load from YAML first
        let mut loaded_from_yaml = false;
        if let Ok(cwd) = std::env::current_dir() {
            // Check protocols/thinktools_v2.yaml
            let yaml_path = cwd.join("protocols").join("thinktools_v2.yaml");

            if yaml_path.exists() {
                match self.load_from_yaml(&yaml_path) {
                    Ok(count) => {
                        tracing::info!("Loaded {} protocols from thinktools_v2.yaml", count);
                        loaded_from_yaml = true;
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load thinktools_v2.yaml: {}, falling back to hardcoded protocols", e);
                    }
                }
            }
        }

        // Only use hardcoded fallbacks if we failed to load from YAML
        if !loaded_from_yaml {
            tracing::info!("Using hardcoded fallback protocols");
            self.register(builtin_gigathink())?;
            self.register(builtin_laserlogic())?;
            self.register(builtin_bedrock())?;
            self.register(builtin_proofguard())?;
            self.register(builtin_brutalhonesty())?;
        }

        Ok(())
    }
}

/// Get config directory path
fn dirs_config_path() -> Option<PathBuf> {
    dirs::config_dir()
}

// ═══════════════════════════════════════════════════════════════════════════
// BUILT-IN PROTOCOLS (FALLBACK)
// ═══════════════════════════════════════════════════════════════════════════

use super::protocol::{
    AggregationType, CritiqueSeverity, InputSpec, OutputSpec, ProtocolMetadata, ProtocolStep,
    ReasoningStrategy, StepAction, StepOutputFormat,
};

fn builtin_gigathink() -> Protocol {
    Protocol {
        id: "gigathink".to_string(),
        name: "GigaThink".to_string(),
        version: "1.0.0".to_string(),
        description: "Expansive creative thinking - generate 10+ diverse perspectives".to_string(),
        strategy: ReasoningStrategy::Expansive,
        input: InputSpec {
            required: vec!["query".to_string()],
            optional: vec!["context".to_string(), "constraints".to_string()],
        },
        steps: vec![
            ProtocolStep {
                id: "identify_dimensions".to_string(),
                action: StepAction::Generate {
                    min_count: 5,
                    max_count: 10,
                },
                prompt_template:
                    r#"Identify 5-10 distinct dimensions or angles to analyze this question:

Question: {{query}}
{{#if context}}Context: {{context}}{{/if}}
{{#if constraints}}Constraints: {{constraints}}{{/if}}

For each dimension, provide a brief label. Format as a numbered list."#
                        .to_string(),
                output_format: StepOutputFormat::List,
                min_confidence: 0.7,
                depends_on: vec![],
                branch: None,
            },
            ProtocolStep {
                id: "explore_perspectives".to_string(),
                action: StepAction::Analyze {
                    criteria: vec![
                        "novelty".to_string(),
                        "relevance".to_string(),
                        "depth".to_string(),
                    ],
                },
                prompt_template: r#"For each dimension identified, provide:
1. Key insight from this perspective
2. Supporting evidence or example
3. Implications or consequences
4. Confidence score (0.0-1.0)

Dimensions to explore:
{{identify_dimensions}}

Question: {{query}}"#
                    .to_string(),
                output_format: StepOutputFormat::Structured,
                min_confidence: 0.6,
                depends_on: vec!["identify_dimensions".to_string()],
                branch: None,
            },
            ProtocolStep {
                id: "synthesize".to_string(),
                action: StepAction::Synthesize {
                    aggregation: AggregationType::ThematicClustering,
                },
                prompt_template:
                    r#"Synthesize the perspectives into key themes and actionable insights:

Perspectives:
{{explore_perspectives}}

Provide:
1. Major themes (2-4)
2. Key insights (3-5)
3. Recommended actions (if applicable)
4. Areas of uncertainty"#
                        .to_string(),
                output_format: StepOutputFormat::Structured,
                min_confidence: 0.8,
                depends_on: vec!["explore_perspectives".to_string()],
                branch: None,
            },
        ],
        output: OutputSpec {
            format: "GigaThinkResult".to_string(),
            fields: vec![
                "dimensions".to_string(),
                "perspectives".to_string(),
                "themes".to_string(),
                "insights".to_string(),
                "confidence".to_string(),
            ],
        },
        validation: vec![],
        metadata: ProtocolMetadata {
            category: "creative".to_string(),
            composable_with: vec!["laserlogic".to_string(), "brutalhonesty".to_string()],
            typical_tokens: 2500,
            estimated_latency_ms: 5000,
            ..Default::default()
        },
    }
}

fn builtin_laserlogic() -> Protocol {
    Protocol {
        id: "laserlogic".to_string(),
        name: "LaserLogic".to_string(),
        version: "1.0.0".to_string(),
        description: "Precision deductive reasoning with fallacy detection".to_string(),
        strategy: ReasoningStrategy::Deductive,
        input: InputSpec {
            required: vec!["argument".to_string()],
            optional: vec!["context".to_string()],
        },
        steps: vec![
            ProtocolStep {
                id: "extract_claims".to_string(),
                action: StepAction::Analyze {
                    criteria: vec!["clarity".to_string(), "completeness".to_string()],
                },
                prompt_template: r#"Extract the logical structure from this argument:

Argument: {{argument}}

Identify:
1. Main conclusion
2. Supporting premises
3. Implicit assumptions
4. Causal claims (if any)

Format each as a clear statement."#
                    .to_string(),
                output_format: StepOutputFormat::Structured,
                min_confidence: 0.7,
                depends_on: vec![],
                branch: None,
            },
            ProtocolStep {
                id: "check_validity".to_string(),
                action: StepAction::Validate {
                    rules: vec![
                        "logical_consistency".to_string(),
                        "premise_support".to_string(),
                    ],
                },
                prompt_template: r#"Evaluate the logical validity of this argument analysis:

{{extract_claims}}

Based on the claims identified above, check:
1. Do the premises logically lead to the conclusion?
2. Are there gaps in reasoning?
3. Is the argument valid (structure) vs sound (true premises)?
4. Rate logical strength (0.0-1.0)"#
                    .to_string(),
                output_format: StepOutputFormat::Structured,
                min_confidence: 0.8,
                depends_on: vec!["extract_claims".to_string()],
                branch: None,
            },
            ProtocolStep {
                id: "detect_fallacies".to_string(),
                action: StepAction::Critique {
                    severity: CritiqueSeverity::Standard,
                },
                prompt_template: r#"Check for logical fallacies in the argument:

Argument structure:
{{extract_claims}}

Common fallacies to check:
- Ad hominem, Straw man, False dichotomy
- Appeal to authority, Circular reasoning
- Hasty generalization, Post hoc
- Slippery slope, Red herring

For each fallacy found, explain where and why."#
                    .to_string(),
                output_format: StepOutputFormat::List,
                min_confidence: 0.7,
                depends_on: vec!["extract_claims".to_string()],
                branch: None,
            },
        ],
        output: OutputSpec {
            format: "LaserLogicResult".to_string(),
            fields: vec![
                "conclusion".to_string(),
                "premises".to_string(),
                "validity".to_string(),
                "fallacies".to_string(),
                "confidence".to_string(),
            ],
        },
        validation: vec![],
        metadata: ProtocolMetadata {
            category: "analytical".to_string(),
            composable_with: vec!["gigathink".to_string(), "bedrock".to_string()],
            typical_tokens: 1800,
            estimated_latency_ms: 4000,
            ..Default::default()
        },
    }
}

fn builtin_bedrock() -> Protocol {
    Protocol {
        id: "bedrock".to_string(),
        name: "BedRock".to_string(),
        version: "1.0.0".to_string(),
        description: "First principles decomposition - reduce to fundamental axioms".to_string(),
        strategy: ReasoningStrategy::Analytical,
        input: InputSpec {
            required: vec!["statement".to_string()],
            optional: vec!["domain".to_string()],
        },
        steps: vec![
            ProtocolStep {
                id: "decompose".to_string(),
                action: StepAction::Analyze {
                    criteria: vec!["fundamentality".to_string(), "independence".to_string()],
                },
                prompt_template: r#"Decompose this statement to first principles:

Statement: {{statement}}
{{#if domain}}Domain: {{domain}}{{/if}}

Ask repeatedly: "What is this based on? Why is this true?"
Continue until reaching fundamental axioms or assumptions.

Format as a tree structure showing dependencies."#
                    .to_string(),
                output_format: StepOutputFormat::Structured,
                min_confidence: 0.7,
                depends_on: vec![],
                branch: None,
            },
            ProtocolStep {
                id: "identify_axioms".to_string(),
                action: StepAction::Generate {
                    min_count: 3,
                    max_count: 7,
                },
                prompt_template: r#"From the decomposition, identify the foundational axioms:

Decomposition:
{{decompose}}

For each axiom:
1. State clearly
2. Explain why it's fundamental (cannot be further reduced)
3. Note if it's empirical, logical, or definitional
4. Rate certainty (0.0-1.0)"#
                    .to_string(),
                output_format: StepOutputFormat::List,
                min_confidence: 0.8,
                depends_on: vec!["decompose".to_string()],
                branch: None,
            },
            ProtocolStep {
                id: "reconstruct".to_string(),
                action: StepAction::Synthesize {
                    aggregation: AggregationType::WeightedMerge,
                },
                prompt_template: r#"Reconstruct the original statement from axioms:

Axioms:
{{identify_axioms}}

Original statement: {{statement}}

Show the logical path from axioms to statement.
Identify any gaps or leaps in reasoning.
Calculate overall confidence based on axiom certainties."#
                    .to_string(),
                output_format: StepOutputFormat::Structured,
                min_confidence: 0.75,
                depends_on: vec!["identify_axioms".to_string()],
                branch: None,
            },
        ],
        output: OutputSpec {
            format: "BedRockResult".to_string(),
            fields: vec![
                "axioms".to_string(),
                "decomposition".to_string(),
                "reconstruction".to_string(),
                "gaps".to_string(),
                "confidence".to_string(),
            ],
        },
        validation: vec![],
        metadata: ProtocolMetadata {
            category: "analytical".to_string(),
            composable_with: vec!["laserlogic".to_string(), "proofguard".to_string()],
            typical_tokens: 2000,
            estimated_latency_ms: 4500,
            ..Default::default()
        },
    }
}

fn builtin_proofguard() -> Protocol {
    Protocol {
        id: "proofguard".to_string(),
        name: "ProofGuard".to_string(),
        version: "1.0.0".to_string(),
        description: "Multi-source verification using triangulation protocol".to_string(),
        strategy: ReasoningStrategy::Verification,
        input: InputSpec {
            required: vec!["claim".to_string()],
            optional: vec!["sources".to_string()],
        },
        steps: vec![
            ProtocolStep {
                id: "identify_sources".to_string(),
                action: StepAction::CrossReference { min_sources: 3 },
                prompt_template: r#"Identify potential sources to verify this claim:

Claim: {{claim}}
{{#if sources}}Known sources: {{sources}}{{/if}}

List 3+ independent sources that could verify or refute this claim.
Prioritize: official docs, peer-reviewed, primary sources."#
                    .to_string(),
                output_format: StepOutputFormat::List,
                min_confidence: 0.6,
                depends_on: vec![],
                branch: None,
            },
            ProtocolStep {
                id: "verify_each".to_string(),
                action: StepAction::Validate {
                    rules: vec![
                        "source_reliability".to_string(),
                        "claim_support".to_string(),
                    ],
                },
                prompt_template: r#"For each source, evaluate support for the claim:

Claim: {{claim}}
Sources to check:
{{identify_sources}}

For each source:
1. What does it say about the claim?
2. Support level: Confirms / Partially confirms / Neutral / Contradicts
3. Source reliability (0.0-1.0)
4. Key quote or evidence"#
                    .to_string(),
                output_format: StepOutputFormat::Structured,
                min_confidence: 0.7,
                depends_on: vec!["identify_sources".to_string()],
                branch: None,
            },
            ProtocolStep {
                id: "triangulate".to_string(),
                action: StepAction::Synthesize {
                    aggregation: AggregationType::Consensus,
                },
                prompt_template: r#"Apply triangulation to determine claim validity:

Claim: {{claim}}
Source evaluations:
{{verify_each}}

Triangulation rules:
- 3+ independent confirming sources = HIGH confidence
- 2 confirming, 1 neutral = MEDIUM confidence
- Mixed results = LOW confidence, note discrepancies
- Any contradiction = FLAG for review

Provide final verdict and confidence score."#
                    .to_string(),
                output_format: StepOutputFormat::Structured,
                min_confidence: 0.8,
                depends_on: vec!["verify_each".to_string()],
                branch: None,
            },
        ],
        output: OutputSpec {
            format: "ProofGuardResult".to_string(),
            fields: vec![
                "verdict".to_string(),
                "sources".to_string(),
                "evidence".to_string(),
                "discrepancies".to_string(),
                "confidence".to_string(),
            ],
        },
        validation: vec![],
        metadata: ProtocolMetadata {
            category: "verification".to_string(),
            composable_with: vec!["bedrock".to_string(), "brutalhonesty".to_string()],
            typical_tokens: 2200,
            estimated_latency_ms: 5000,
            ..Default::default()
        },
    }
}

fn builtin_brutalhonesty() -> Protocol {
    Protocol {
        id: "brutalhonesty".to_string(),
        name: "BrutalHonesty".to_string(),
        version: "1.0.0".to_string(),
        description: "Adversarial self-critique - find every flaw".to_string(),
        strategy: ReasoningStrategy::Adversarial,
        input: InputSpec {
            required: vec!["work".to_string()],
            optional: vec!["criteria".to_string()],
        },
        steps: vec![
            ProtocolStep {
                id: "steelman".to_string(),
                action: StepAction::Analyze {
                    criteria: vec!["strengths".to_string()],
                },
                prompt_template: r#"First, steelman the work - what are its genuine strengths?

Work to critique:
{{work}}

Identify:
1. What does this do well?
2. What problems does it solve?
3. What is genuinely valuable here?

Be generous but honest."#
                    .to_string(),
                output_format: StepOutputFormat::List,
                min_confidence: 0.7,
                depends_on: vec![],
                branch: None,
            },
            ProtocolStep {
                id: "attack".to_string(),
                action: StepAction::Critique {
                    severity: CritiqueSeverity::Brutal,
                },
                prompt_template: r#"Now be brutally honest - what's wrong with this?

Work:
{{work}}

Strengths identified:
{{steelman}}

Attack from all angles:
1. Logical flaws
2. Missing considerations
3. Weak assumptions
4. Implementation problems
5. Unintended consequences
6. What would a harsh critic say?

Don't hold back. Be specific."#
                    .to_string(),
                output_format: StepOutputFormat::List,
                min_confidence: 0.6,
                depends_on: vec!["steelman".to_string()],
                branch: None,
            },
            ProtocolStep {
                id: "verdict".to_string(),
                action: StepAction::Decide {
                    method: super::protocol::DecisionMethod::ProsCons,
                },
                prompt_template: r#"Final verdict - is this work acceptable?

Strengths:
{{steelman}}

Flaws:
{{attack}}

Provide:
1. Overall assessment (Pass / Conditional Pass / Fail)
2. Most critical issue to fix
3. Confidence in verdict (0.0-1.0)
4. What would make this excellent?"#
                    .to_string(),
                output_format: StepOutputFormat::Structured,
                min_confidence: 0.75,
                depends_on: vec!["steelman".to_string(), "attack".to_string()],
                branch: None,
            },
        ],
        output: OutputSpec {
            format: "BrutalHonestyResult".to_string(),
            fields: vec![
                "strengths".to_string(),
                "flaws".to_string(),
                "verdict".to_string(),
                "critical_fix".to_string(),
                "confidence".to_string(),
            ],
        },
        validation: vec![],
        metadata: ProtocolMetadata {
            category: "critique".to_string(),
            composable_with: vec!["gigathink".to_string(), "proofguard".to_string()],
            typical_tokens: 2000,
            estimated_latency_ms: 4500,
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ProtocolRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_builtins() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        assert_eq!(registry.len(), 6);
        assert!(registry.contains("gigathink"));
        assert!(registry.contains("laserlogic"));
        assert!(registry.contains("bedrock"));
        assert!(registry.contains("proofguard"));
        assert!(registry.contains("brutalhonesty"));
        assert!(registry.contains("powercombo"));
    }

    #[test]
    fn test_get_protocol() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let gt = registry.get("gigathink").unwrap();
        assert_eq!(gt.name, "GigaThink");
        assert_eq!(gt.strategy, ReasoningStrategy::Expansive);
    }

    #[test]
    fn test_list_ids() {
        let mut registry = ProtocolRegistry::new();
        registry.register_builtins().unwrap();

        let ids = registry.list_ids();
        assert_eq!(ids.len(), 6);
        assert!(ids.contains(&"gigathink"));
        assert!(ids.contains(&"powercombo"));
    }
}
