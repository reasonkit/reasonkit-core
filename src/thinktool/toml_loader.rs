//! TOML Protocol Loader
//!
//! Loads ThinkTool protocols from TOML files.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::protocol::{
    CritiqueSeverity, DecisionMethod, InputSpec, OutputSpec, Protocol, ProtocolMetadata,
    ProtocolStep, ReasoningStrategy, StepAction,
};
use crate::error::{Error, Result};

/// TOML ThinkTool module definition
#[derive(Debug, Clone, Deserialize, Serialize)]
struct TomlThinkToolModule {
    id: String,
    name: String,
    #[serde(default)]
    shortcode: String,
    category: String,
    tier: String,
    description: String,
    capabilities: Vec<String>,
    #[serde(default)]
    output_schema: String,
    #[serde(default)]
    parameters: HashMap<String, serde_json::Value>,
    #[serde(default)]
    confidence_factors: Vec<TomlConfidenceFactor>,
    thinking_pattern: TomlThinkingPattern,
    #[serde(default)]
    typical_duration: String,
    #[serde(default)]
    token_cost_estimate: String,
    #[serde(default)]
    steps: Option<Vec<ProtocolStep>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TomlConfidenceFactor {
    factor: String,
    weight: f64,
    formula: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TomlThinkingPattern {
    #[serde(rename = "type")]
    pattern_type: String,
    steps: Vec<String>,
}

/// Root TOML structure
#[derive(Debug, Clone, Deserialize, Serialize)]
struct TomlThinkToolsV2 {
    version: String,
    schema: String,
    #[serde(default)]
    thinktool_modules: HashMap<String, TomlThinkToolModule>,
}

/// Load protocols from a TOML file
pub fn load_from_toml_file(path: &Path) -> Result<Vec<Protocol>> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::IoMessage {
        message: format!("Failed to read TOML file {}: {}", path.display(), e),
    })?;

    load_from_toml_string(&content)
}

/// Load protocols from a TOML string
pub fn load_from_toml_string(toml_content: &str) -> Result<Vec<Protocol>> {
    let toml_data: TomlThinkToolsV2 = toml::from_str(toml_content).map_err(|e| Error::Parse {
        message: format!("Failed to parse TOML: {}", e),
    })?;

    let mut protocols = Vec::new();

    for (module_key, module) in toml_data.thinktool_modules {
        let protocol = convert_toml_module_to_protocol(&module_key, &module)?;
        protocols.push(protocol);
    }

    Ok(protocols)
}

/// Convert a TOML module definition to a Protocol struct
fn convert_toml_module_to_protocol(
    module_key: &str,
    toml_module: &TomlThinkToolModule,
) -> Result<Protocol> {
    // Determine reasoning strategy from category
    let strategy = match toml_module.category.as_str() {
        "divergent" => ReasoningStrategy::Expansive,
        "convergent" => ReasoningStrategy::Deductive,
        "foundational" => ReasoningStrategy::Analytical,
        "verification" => ReasoningStrategy::Verification,
        "adversarial" => ReasoningStrategy::Adversarial,
        _ => ReasoningStrategy::Analytical,
    };

    // Build input spec based on module type
    let input = build_input_spec(module_key);

    // Build steps: use custom defined steps if provided, otherwise fallback to pattern
    let steps = if let Some(custom_steps) = &toml_module.steps {
        custom_steps.clone()
    } else {
        build_steps_from_pattern(&toml_module.thinking_pattern, module_key)?
    };

    // Build output spec
    let output = build_output_spec(&toml_module.name);

    // Build metadata
    let metadata = ProtocolMetadata {
        category: toml_module.category.clone(),
        composable_with: get_composable_modules(module_key),
        typical_tokens: estimate_tokens(&toml_module.token_cost_estimate),
        estimated_latency_ms: estimate_latency(&toml_module.typical_duration),
        ..Default::default()
    };

    let protocol = Protocol {
        id: module_key.to_string(),
        name: toml_module.name.clone(),
        version: "2.0.0".to_string(),
        description: toml_module.description.trim().to_string(),
        strategy,
        input,
        steps,
        output,
        validation: Vec::new(),
        metadata,
    };

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

// Helper functions (duplicated from yaml_loader for now to avoid refactoring)

fn build_input_spec(module_key: &str) -> InputSpec {
    match module_key {
        "gigathink" => InputSpec {
            required: vec!["query".to_string()],
            optional: vec!["context".to_string(), "constraints".to_string()],
        },
        "laserlogic" => InputSpec {
            required: vec!["argument".to_string()],
            optional: vec!["context".to_string()],
        },
        "bedrock" => InputSpec {
            required: vec!["statement".to_string()],
            optional: vec!["domain".to_string()],
        },
        "proofguard" => InputSpec {
            required: vec!["claim".to_string()],
            optional: vec!["sources".to_string()],
        },
        "brutalhonesty" => InputSpec {
            required: vec!["work".to_string()],
            optional: vec!["criteria".to_string()],
        },
        _ => InputSpec::default(),
    }
}

fn build_steps_from_pattern(
    _pattern: &TomlThinkingPattern,
    module_key: &str,
) -> Result<Vec<ProtocolStep>> {
    // Note: In a real implementation, we would parse the pattern steps.
    // For now, we reuse the hardcoded builders from the original implementation
    // to ensure consistency, keyed by the module ID.
    match module_key {
        "gigathink" => Ok(build_gigathink_steps()),
        "laserlogic" => Ok(build_laserlogic_steps()),
        "bedrock" => Ok(build_bedrock_steps()),
        "proofguard" => Ok(build_proofguard_steps()),
        "brutalhonesty" => Ok(build_brutalhonesty_steps()),
        "powercombo" => Ok(build_gigathink_steps()),
        _ => Err(Error::Validation(format!(
            "Unknown module type: {}",
            module_key
        ))),
    }
}

// Re-implement step builders (copy from yaml_loader.rs)
// This duplication is acceptable for this task to avoid breaking existing code
// by refactoring yaml_loader.rs into a shared module.

use super::protocol::{AggregationType as AggType, StepOutputFormat as StepFmt};

fn build_gigathink_steps() -> Vec<ProtocolStep> {
    vec![
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
            output_format: StepFmt::List,
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
            output_format: StepFmt::Structured,
            min_confidence: 0.6,
            depends_on: vec!["identify_dimensions".to_string()],
            branch: None,
        },
        ProtocolStep {
            id: "synthesize".to_string(),
            action: StepAction::Synthesize {
                aggregation: AggType::ThematicClustering,
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
            output_format: StepFmt::Structured,
            min_confidence: 0.8,
            depends_on: vec!["explore_perspectives".to_string()],
            branch: None,
        },
    ]
}

fn build_laserlogic_steps() -> Vec<ProtocolStep> {
    vec![
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
            output_format: StepFmt::Structured,
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
2. Are there gaps in the reasoning chain?
3. Is the argument valid (logical structure) vs sound (true premises)?
4. Rate the logical strength (0.0-1.0) with justification"#
                .to_string(),
            output_format: StepFmt::Structured,
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
            output_format: StepFmt::List,
            min_confidence: 0.7,
            depends_on: vec!["extract_claims".to_string()],
            branch: None,
        },
    ]
}

fn build_bedrock_steps() -> Vec<ProtocolStep> {
    vec![
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
            output_format: StepFmt::Structured,
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
            output_format: StepFmt::List,
            min_confidence: 0.8,
            depends_on: vec!["decompose".to_string()],
            branch: None,
        },
        ProtocolStep {
            id: "reconstruct".to_string(),
            action: StepAction::Synthesize {
                aggregation: AggType::WeightedMerge,
            },
            prompt_template: r#"Reconstruct the original statement from axioms:

Axioms:
{{identify_axioms}}

Original statement: {{statement}}

Show the logical path from axioms to statement.
Identify any gaps or leaps in reasoning.
Calculate overall confidence based on axiom certainties."#
                .to_string(),
            output_format: StepFmt::Structured,
            min_confidence: 0.75,
            depends_on: vec!["identify_axioms".to_string()],
            branch: None,
        },
    ]
}

fn build_proofguard_steps() -> Vec<ProtocolStep> {
    vec![
        ProtocolStep {
            id: "identify_sources".to_string(),
            action: StepAction::CrossReference { min_sources: 3 },
            prompt_template: r#"Identify potential sources to verify this claim:

Claim: {{claim}}
{{#if sources}}Known sources: {{sources}}{{/if}}

List 3+ independent sources that could verify or refute this claim.
Prioritize: official docs, peer-reviewed, primary sources."#
                .to_string(),
            output_format: StepFmt::List,
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
            output_format: StepFmt::Structured,
            min_confidence: 0.7,
            depends_on: vec!["identify_sources".to_string()],
            branch: None,
        },
        ProtocolStep {
            id: "triangulate".to_string(),
            action: StepAction::Synthesize {
                aggregation: AggType::Consensus,
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
            output_format: StepFmt::Structured,
            min_confidence: 0.8,
            depends_on: vec!["verify_each".to_string()],
            branch: None,
        },
    ]
}

fn build_brutalhonesty_steps() -> Vec<ProtocolStep> {
    vec![
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
            output_format: StepFmt::List,
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
            output_format: StepFmt::List,
            min_confidence: 0.6,
            depends_on: vec!["steelman".to_string()],
            branch: None,
        },
        ProtocolStep {
            id: "verdict".to_string(),
            action: StepAction::Decide {
                method: DecisionMethod::ProsCons,
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
            output_format: StepFmt::Structured,
            min_confidence: 0.75,
            depends_on: vec!["steelman".to_string(), "attack".to_string()],
            branch: None,
        },
    ]
}

fn build_output_spec(module_name: &str) -> OutputSpec {
    let format = format!("{}Result", module_name.replace(" ", ""));
    let fields = match module_name {
        "GigaThink" => vec![
            "dimensions".to_string(),
            "perspectives".to_string(),
            "themes".to_string(),
            "insights".to_string(),
            "confidence".to_string(),
        ],
        "LaserLogic" => vec![
            "conclusion".to_string(),
            "premises".to_string(),
            "validity".to_string(),
            "fallacies".to_string(),
            "confidence".to_string(),
        ],
        "BedRock" => vec![
            "axioms".to_string(),
            "decomposition".to_string(),
            "reconstruction".to_string(),
            "gaps".to_string(),
            "confidence".to_string(),
        ],
        "ProofGuard" => vec![
            "verdict".to_string(),
            "sources".to_string(),
            "evidence".to_string(),
            "discrepancies".to_string(),
            "confidence".to_string(),
        ],
        "BrutalHonesty" => vec![
            "strengths".to_string(),
            "flaws".to_string(),
            "verdict".to_string(),
            "critical_fix".to_string(),
            "confidence".to_string(),
        ],
        _ => vec!["confidence".to_string()],
    };

    OutputSpec { format, fields }
}

fn get_composable_modules(module_key: &str) -> Vec<String> {
    match module_key {
        "gigathink" => vec!["laserlogic".to_string(), "brutalhonesty".to_string()],
        "laserlogic" => vec!["gigathink".to_string(), "bedrock".to_string()],
        "bedrock" => vec!["laserlogic".to_string(), "proofguard".to_string()],
        "proofguard" => vec!["bedrock".to_string(), "brutalhonesty".to_string()],
        "brutalhonesty" => vec!["gigathink".to_string(), "proofguard".to_string()],
        _ => vec![],
    }
}

fn estimate_tokens(cost_estimate: &str) -> u32 {
    match cost_estimate {
        "low" => 1000,
        "medium" => 2000,
        "medium-high" => 2500,
        "high" => 3000,
        _ => 2000,
    }
}

fn estimate_latency(duration: &str) -> u32 {
    if let Some(range) = duration.strip_suffix('s') {
        if let Some((low, high)) = range.split_once('-') {
            if let (Ok(low_val), Ok(high_val)) = (low.parse::<u32>(), high.parse::<u32>()) {
                return ((low_val + high_val) / 2) * 1000;
            }
        }
    }
    5000
}
