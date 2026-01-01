# ThinkTools: Structured Reasoning Protocols

> Turn Prompts into Protocols - "Designed, Not Dreamed"

ThinkTools are structured reasoning modules that transform ad-hoc LLM prompting into auditable, reproducible reasoning chains. Each tool implements a specific analytical strategy, producing verifiable output with confidence scores and maintaining full provenance of the reasoning process.

---

## Table of Contents

1. [Overview](#overview)
2. [ThinkTool Family](#thinktool-family)
3. [GigaThink: Expansive Creative Thinking](#gigathink-expansive-creative-thinking)
4. [LaserLogic: Precision Deductive Reasoning](#laserlogic-precision-deductive-reasoning)
5. [BedRock: First Principles Decomposition](#bedrock-first-principles-decomposition)
6. [ProofGuard: Multi-Source Verification](#proofguard-multi-source-verification)
7. [BrutalHonesty: Adversarial Self-Critique](#brutalhonesty-adversarial-self-critique)
8. [PowerCombo: Chaining ThinkTools](#powercombo-chaining-thinktools)
9. [Profile Selection](#profile-selection)
10. [Configuration Reference](#configuration-reference)
11. [Real Output Examples](#real-output-examples)

---

## Overview

### What is a ThinkTool?

A ThinkTool is a structured reasoning protocol that:

1. **Defines a reasoning strategy** - expansive, deductive, adversarial, etc.
2. **Structures thought into auditable steps** - each step is traceable
3. **Produces verifiable output with confidence scores** - calibrated certainty
4. **Maintains provenance** - full chain of reasoning preserved

### Why ThinkTools?

| Problem with Raw Prompting     | ThinkTool Solution                  |
| ------------------------------ | ----------------------------------- |
| Inconsistent reasoning quality | Structured, reproducible protocols  |
| No confidence calibration      | Evidence-based confidence scoring   |
| Difficult to audit decisions   | Full execution trace and provenance |
| Single perspective blindness   | Multi-dimensional analysis          |
| Hidden assumptions             | Explicit assumption surfacing       |

### Quick Start

```rust
use reasonkit_core::thinktool::{ProtocolExecutor, ProtocolInput};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let executor = ProtocolExecutor::new()?;

    // Execute GigaThink for creative analysis
    let result = executor.execute(
        "gigathink",
        ProtocolInput::query("What are the key factors for startup success?")
    ).await?;

    println!("Confidence: {:.0}%", result.confidence * 100.0);
    for perspective in result.perspectives() {
        println!("- {}", perspective);
    }

    Ok(())
}
```

---

## ThinkTool Family

| Module            | Code | Purpose                                          | Profile      |
| ----------------- | ---- | ------------------------------------------------ | ------------ |
| **GigaThink**     | `gt` | Expansive creative thinking, 10+ perspectives    | `--creative` |
| **LaserLogic**    | `ll` | Precision deductive reasoning, fallacy detection | `--balanced` |
| **BedRock**       | `br` | First principles decomposition                   | `--deep`     |
| **ProofGuard**    | `pg` | Multi-source verification                        | `--paranoid` |
| **BrutalHonesty** | `bh` | Adversarial self-critique                        | `--paranoid` |

### Confidence Weight Allocation

Each ThinkTool contributes to composite confidence with calibrated weights:

| Module        | Weight | Rationale                                 |
| ------------- | ------ | ----------------------------------------- |
| ProofGuard    | 0.30   | Verification-focused, highest reliability |
| LaserLogic    | 0.25   | Formal logic, high precision              |
| BedRock       | 0.25   | Foundational analysis                     |
| GigaThink     | 0.15   | Creative expansion, lower certainty       |
| BrutalHonesty | 0.15   | Critique-focused, conservative            |

---

## GigaThink: Expansive Creative Thinking

### Purpose

GigaThink generates 10+ diverse perspectives through systematic exploration of analytical dimensions. It implements divergent thinking to explore problems from multiple angles before convergence.

### Key Features

- **10+ Perspectives Guaranteed** - Minimum of 10 distinct viewpoints
- **12 Analytical Dimensions** - Systematic multi-dimensional coverage
- **Cross-Validation** - Built-in coherence validation
- **Theme Identification** - Discovers patterns across perspectives
- **Insight Synthesis** - Actionable insights from analysis

### Analytical Dimensions

| Dimension                    | Focus Area                                |
| ---------------------------- | ----------------------------------------- |
| Economic/Financial           | Costs, revenues, market forces            |
| Technological/Innovation     | Enabling tech, constraints, opportunities |
| Social/Cultural              | Stakeholder interests, adoption barriers  |
| Environmental/Sustainability | Ecological impact, resource use           |
| Political/Regulatory         | Regulations, policy changes               |
| Psychological/Behavioral     | Cognitive biases, decision patterns       |
| Ethical/Moral                | Fairness, harms, benefits                 |
| Historical/Evolutionary      | Precedents, patterns                      |
| Competitive/Market           | Competition, positioning                  |
| User Experience/Adoption     | Usability, friction                       |
| Risk/Opportunity             | Downsides, upsides                        |
| Long-term/Strategic          | Future impact, optionality                |

### Configuration

```rust
use reasonkit_core::thinktool::modules::{GigaThink, GigaThinkBuilder, AnalysisDimension};

// Default configuration
let gigathink = GigaThink::new();

// Custom configuration via builder
let gigathink = GigaThinkBuilder::new()
    .min_perspectives(12)          // Require at least 12 perspectives
    .max_perspectives(20)          // Cap at 20
    .min_confidence(0.80)          // Higher confidence threshold
    .cross_validation(true)        // Enable validation
    .dimensions(vec![              // Focus on specific dimensions
        AnalysisDimension::Economic,
        AnalysisDimension::Technological,
        AnalysisDimension::Competitive,
    ])
    .novelty_weight(0.35)          // Weight for novelty scoring
    .depth_weight(0.40)            // Weight for depth scoring
    .coherence_weight(0.25)        // Weight for coherence
    .build();
```

### Default Configuration Values

```rust
GigaThinkConfig {
    min_perspectives: 10,
    max_perspectives: 15,
    min_confidence: 0.70,
    enable_cross_validation: true,
    min_query_length: 10,
    max_query_length: 5000,
    dimensions: vec![],  // Empty = all dimensions
    novelty_weight: 0.30,
    depth_weight: 0.40,
    coherence_weight: 0.30,
    max_execution_time_ms: Some(10000),
}
```

### Usage Example

```rust
use reasonkit_core::thinktool::modules::{GigaThink, ThinkToolModule, ThinkToolContext};

let module = GigaThink::new();
let context = ThinkToolContext::new("What are the implications of AI on employment?");

let result = module.execute(&context)?;

println!("Module: {}", result.module);
println!("Confidence: {:.0}%", result.confidence * 100.0);

// Access perspectives
if let Some(perspectives) = result.get_array("perspectives") {
    for p in perspectives {
        if let Some(title) = p.get("title").and_then(|v| v.as_str()) {
            println!("- {}", title);
        }
    }
}
```

---

## LaserLogic: Precision Deductive Reasoning

### Purpose

LaserLogic performs rigorous logical analysis with formal fallacy detection, argument structure validation, and syllogism analysis. Based on classical deductive logic and NL2FOL research.

### Key Features

- **10+ Fallacy Types** - Detects formal and informal fallacies
- **Argument Form Detection** - Identifies modus ponens, syllogisms, etc.
- **Validity Checking** - Verifies conclusion follows from premises
- **Soundness Checking** - Valid + true premises verification
- **Contradiction Detection** - Finds conflicting statements

### Detectable Fallacies

**Formal Fallacies (Invalid Argument Structure):**

| Fallacy              | Pattern                          | Description                                        |
| -------------------- | -------------------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| Affirming Consequent | P->Q, Q                          | - P                                                | "If rain then wet ground. Ground is wet. Therefore it rained." |
| Denying Antecedent   | P->Q, ~P                         | - ~Q                                               | "If rain then wet. No rain. Therefore not wet."                |
| Undistributed Middle | All A are B, All C are B         | - All A are C                                      | Middle term not distributed                                    |
| Illicit Major/Minor  | Distribution errors in syllogism | Term distribution violations                       |
| Four Terms           | 4 terms in 3-term syllogism      | Equivocation in terms                              |
| Existential Fallacy  | Universal -> Existential         | "All unicorns are white, so some unicorn is white" |

**Semi-Formal and Causal Fallacies:**

| Fallacy            | Description                                |
| ------------------ | ------------------------------------------ |
| Circular Reasoning | Conclusion restated in premises            |
| Non Sequitur       | Conclusion does not follow                 |
| Post Hoc           | Temporal sequence implies causation        |
| Slippery Slope     | Unwarranted chain of consequences          |
| False Dichotomy    | Only two options presented when more exist |
| Equivocation       | Same term used with different meanings     |

### Argument Forms Recognized

| Form                   | Pattern                  | Example       |
| ---------------------- | ------------------------ | ------------- | ------------------------------------------------ |
| Modus Ponens           | P->Q, P                  | - Q           | "If rain, ground wet. It rains. Therefore, wet." |
| Modus Tollens          | P->Q, ~Q                 | - ~P          | "If rain, wet. Not wet. Therefore, no rain."     |
| Hypothetical Syllogism | P->Q, Q->R               | - P->R        | Chain reasoning                                  |
| Disjunctive Syllogism  | P v Q, ~P                | - Q           | Process of elimination                           |
| Categorical Syllogism  | All M are P, All S are M | - All S are P | Term-based reasoning                             |

### Configuration

```rust
use reasonkit_core::thinktool::modules::{LaserLogic, LaserLogicConfig};

// Default configuration
let laser = LaserLogic::new();

// Quick mode - validity and basic fallacy detection
let laser_quick = LaserLogic::with_config(LaserLogicConfig::quick());

// Deep mode - all features enabled
let laser_deep = LaserLogic::with_config(LaserLogicConfig::deep());

// Paranoid mode - strictest validation
let laser_paranoid = LaserLogic::with_config(LaserLogicConfig::paranoid());

// Custom configuration
let laser_custom = LaserLogic::with_config(LaserLogicConfig {
    detect_fallacies: true,
    check_validity: true,
    check_soundness: true,
    max_premise_depth: 10,
    analyze_syllogisms: true,
    detect_contradictions: true,
    confidence_threshold: 0.7,
    verbose_output: false,
});
```

### Default Configuration Values

```rust
LaserLogicConfig {
    detect_fallacies: true,
    check_validity: true,
    check_soundness: true,
    max_premise_depth: 10,
    analyze_syllogisms: true,
    detect_contradictions: true,
    confidence_threshold: 0.7,
    verbose_output: false,
}
```

### Usage Example

```rust
use reasonkit_core::thinktool::modules::{LaserLogic, ThinkToolModule, ThinkToolContext};

let laser = LaserLogic::new();

// Direct argument analysis
let result = laser.analyze_argument(
    &["All humans are mortal", "Socrates is human"],
    "Socrates is mortal"
)?;

println!("Argument Form: {:?}", result.argument_form);
println!("Validity: {:?}", result.validity);
println!("Soundness: {:?}", result.soundness);
println!("Verdict: {}", result.verdict());

// Check for fallacies
if result.has_fallacies() {
    println!("Fallacies detected:");
    for fallacy in &result.fallacies {
        println!("  - {:?}: {}", fallacy.fallacy, fallacy.evidence);
    }
}

// Via ThinkToolContext
let context = ThinkToolContext::new(
    "All humans are mortal. Socrates is human. Therefore, Socrates is mortal."
);
let output = laser.execute(&context)?;
```

---

## BedRock: First Principles Decomposition

### Purpose

BedRock reduces problems to fundamental axioms through recursive analysis, then rebuilds understanding using Tree-of-Thoughts exploration. Implements Elon Musk-style first principles thinking.

### Key Features

- **Recursive Decomposition** - Breaks problems into components
- **Axiom Identification** - Finds self-evident truths
- **Assumption Surfacing** - Exposes hidden premises
- **Gap Detection** - Identifies reasoning holes
- **Reconstruction Paths** - Rebuilds from foundations

### Principle Types

| Type       | Weight | Description                   | Example                |
| ---------- | ------ | ----------------------------- | ---------------------- |
| Axiom      | 1.0    | Self-evident truth            | "A = A", physical laws |
| Definition | 0.95   | Clarifying terminology        | "What does X mean?"    |
| Empirical  | 0.80   | Based on data/observation     | Research findings      |
| Derived    | 0.75   | Logically derived from axioms | Conclusions            |
| Assumption | 0.50   | Assumed for argument          | Unstated premises      |
| Contested  | 0.30   | Debatable claim               | Opinions               |

### Configuration

```rust
use reasonkit_core::thinktool::modules::{BedRock, BedRockConfig};

// Default configuration
let bedrock = BedRock::new();

// Custom configuration
let bedrock = BedRock::with_config(BedRockConfig {
    max_depth: 3,               // Maximum decomposition depth
    axiom_threshold: 0.85,      // Fundamentality threshold for axioms
    branching_factor: 3,        // Parallel thought branches
    min_confidence: 0.5,        // Minimum confidence threshold
    strict_assumptions: true,   // Require explicit assumptions
    max_principles: 20,         // Maximum principles to identify
});
```

### Default Configuration Values

```rust
BedRockConfig {
    max_depth: 3,
    axiom_threshold: 0.85,
    branching_factor: 3,
    min_confidence: 0.5,
    strict_assumptions: true,
    max_principles: 20,
}
```

### Usage Example

```rust
use reasonkit_core::thinktool::modules::{BedRock, ThinkToolModule, ThinkToolContext};

let bedrock = BedRock::new();
let context = ThinkToolContext::new("Why are electric vehicles better than gas cars?");

let output = bedrock.execute(&context)?;

// Access structured output
let json = &output.output;

// Get axioms
if let Some(axioms) = json.get("axioms").and_then(|v| v.as_array()) {
    println!("Foundational Axioms:");
    for axiom in axioms {
        if let Some(statement) = axiom.get("statement").and_then(|v| v.as_str()) {
            println!("  - {}", statement);
        }
    }
}

// Get assumptions
if let Some(assumptions) = json.get("assumptions").and_then(|v| v.as_array()) {
    println!("Hidden Assumptions:");
    for assumption in assumptions {
        if let Some(statement) = assumption.get("statement").and_then(|v| v.as_str()) {
            println!("  - {}", statement);
        }
    }
}

// Get insights
if let Some(insights) = json.get("insights").and_then(|v| v.as_array()) {
    println!("Key Insights:");
    for insight in insights {
        if let Some(text) = insight.as_str() {
            println!("  - {}", text);
        }
    }
}
```

---

## ProofGuard: Multi-Source Verification

### Purpose

ProofGuard triangulates claims across 3+ independent sources to verify factual accuracy. Implements the three-source rule (CONS-006) with source tier ranking and contradiction detection.

### Key Features

- **3+ Source Requirement** - Enforces triangulation protocol
- **Source Tier Ranking** - Weights by source quality
- **Contradiction Detection** - Identifies conflicting evidence
- **Confidence Scoring** - Calibrated verification scores
- **Stance Analysis** - Support, Contradict, Neutral, Partial

### Source Tiers

| Tier                 | Weight | Examples                                             |
| -------------------- | ------ | ---------------------------------------------------- |
| Primary (Tier 1)     | 1.0    | Official docs, peer-reviewed papers, primary sources |
| Secondary (Tier 2)   | 0.7    | Reputable news, expert blogs, industry reports       |
| Independent (Tier 3) | 0.4    | Community content, forums                            |
| Unverified (Tier 4)  | 0.2    | Social media, unknown sources                        |

### Verification Verdicts

| Verdict              | Description                            |
| -------------------- | -------------------------------------- |
| Verified             | High confidence, 3+ supporting sources |
| Partially Verified   | Needs qualifier, mixed evidence        |
| Contested            | Conflicting evidence from sources      |
| Insufficient Sources | Need more sources                      |
| Refuted              | Evidence contradicts claim             |
| Inconclusive         | Unable to determine                    |

### Configuration

```rust
use reasonkit_core::thinktool::modules::ProofGuard;
use reasonkit_core::thinktool::triangulation::TriangulationConfig;

// Default configuration
let proofguard = ProofGuard::new();

// Strict mode - requires 2 Tier 1 sources
let proofguard_strict = ProofGuard::strict();

// Relaxed mode - 1 Tier 1 source sufficient
let proofguard_relaxed = ProofGuard::relaxed();

// Custom configuration
let proofguard = ProofGuard::with_config(TriangulationConfig {
    min_sources: 3,
    min_tier1_sources: 1,
    verification_threshold: 0.6,
    require_verified_urls: false,
    require_domain_diversity: true,
    ..Default::default()
});
```

### Usage Example

```rust
use reasonkit_core::thinktool::modules::{ProofGuard, ThinkToolModule, ThinkToolContext};

let proofguard = ProofGuard::new();

// JSON input with sources
let input_json = r#"{
    "claim": "Rust is memory-safe without garbage collection",
    "sources": [
        {"name": "Rust Book", "tier": "Primary", "stance": "Support", "verified": true},
        {"name": "ACM Paper", "tier": "Primary", "stance": "Support", "domain": "PL"},
        {"name": "Tech Blog", "tier": "Secondary", "stance": "Support"}
    ]
}"#;

let context = ThinkToolContext::new(input_json);
let output = proofguard.execute(&context)?;

println!("Confidence: {:.0}%", output.confidence * 100.0);

// Parse output
if let Some(verdict) = output.get_str("verdict") {
    println!("Verdict: {}", verdict);
}
if let Some(is_verified) = output.get("is_verified").and_then(|v| v.as_bool()) {
    println!("Verified: {}", is_verified);
}
```

---

## BrutalHonesty: Adversarial Self-Critique

### Purpose

BrutalHonesty performs red-team analysis to find flaws before others do, challenges assumptions aggressively, and scores confidence with appropriate skepticism.

### Key Features

- **Assumption Hunting** - Extracts and questions implicit assumptions
- **Flaw Detection** - Categorizes weaknesses by severity
- **Skeptical Scoring** - Adjusts confidence downward based on issues
- **Devil's Advocate** - Argues against position to stress-test it
- **Cognitive Bias Detection** - Identifies bias patterns

### Critique Severity Levels

| Level    | Skepticism Multiplier | Description                       |
| -------- | --------------------- | --------------------------------- |
| Gentle   | 0.90 (10% reduction)  | Constructive feedback focus       |
| Standard | 0.80 (20% reduction)  | Balanced flaw detection           |
| Harsh    | 0.65 (35% reduction)  | Aggressive assumption challenging |
| Ruthless | 0.50 (50% reduction)  | No mercy, find every flaw         |

### Flaw Categories

| Category     | Description                                |
| ------------ | ------------------------------------------ |
| Logical      | Fallacy, contradiction, non-sequitur       |
| Evidential   | Missing data, weak sources, cherry-picking |
| Assumption   | Unexamined premises, hidden biases         |
| Scope        | Overgeneralization, false dichotomy        |
| Temporal     | Recency bias, ignoring history             |
| Adversarial  | Vulnerability to counter-arguments         |
| Completeness | Missing considerations, blind spots        |

### Flaw Severities

| Severity | Confidence Penalty | Description                       |
| -------- | ------------------ | --------------------------------- |
| Minor    | -0.02              | Worth noting but not critical     |
| Moderate | -0.08              | Should be addressed               |
| Major    | -0.15              | Significantly weakens argument    |
| Critical | -0.30              | Fundamentally undermines position |

### Configuration

```rust
use reasonkit_core::thinktool::modules::{BrutalHonesty, BrutalHonestyBuilder, CritiqueSeverity};

// Default configuration
let brutal = BrutalHonesty::new();

// Builder pattern
let brutal = BrutalHonesty::builder()
    .severity(CritiqueSeverity::Ruthless)
    .enable_devil_advocate(true)
    .check_confirmation_bias(true)
    .min_confidence_threshold(0.50)
    .max_flaws_reported(10)
    .build();
```

### Default Configuration Values

```rust
BrutalHonestyConfig {
    severity: CritiqueSeverity::Standard,
    enable_devil_advocate: true,
    check_confirmation_bias: true,
    min_confidence_threshold: 0.50,
    max_flaws_reported: 10,
    focus_areas: vec![],  // Empty = all areas
}
```

### Usage Example

```rust
use reasonkit_core::thinktool::modules::{
    BrutalHonesty, BrutalHonestyBuilder, CritiqueSeverity,
    ThinkToolModule, ThinkToolContext
};

let brutal = BrutalHonesty::builder()
    .severity(CritiqueSeverity::Harsh)
    .enable_devil_advocate(true)
    .build();

let context = ThinkToolContext::new(
    "Our startup will succeed because we have the best team"
);

let output = brutal.execute(&context)?;

println!("Confidence: {:.0}%", output.confidence * 100.0);

// Access structured analysis
if let Some(verdict) = output.get_str("verdict") {
    println!("Verdict: {}", verdict);
}

if let Some(analysis) = output.get("analysis") {
    if let Some(flaws) = analysis.get("flaws").and_then(|v| v.as_array()) {
        println!("Flaws detected: {}", flaws.len());
        for flaw in flaws {
            if let Some(desc) = flaw.get("description").and_then(|v| v.as_str()) {
                println!("  - {}", desc);
            }
        }
    }
}

if let Some(devils_advocate) = output.get_str("devils_advocate") {
    println!("Devil's Advocate: {}", devils_advocate);
}
```

---

## PowerCombo: Chaining ThinkTools

### Overview

PowerCombo chains all 5 ThinkTools in sequence for maximum reasoning power with cross-validation. This is the ultimate reasoning mode for high-stakes decisions.

### Chain Sequence

```
GigaThink (creative exploration, temp: 0.8)
    |
    v
LaserLogic (logical validation)
    |
    v
BedRock (first principles decomposition)
    |
    v
ProofGuard (verification, min_confidence: 0.9)
    |
    v
BrutalHonesty (adversarial critique, temp: 0.3)
    |
    v (if confidence < 95%)
ProofGuard (second verification pass)
```

### Configuration

- **Minimum Confidence**: 95%
- **Token Budget**: ~25,000 tokens
- **Tags**: ultimate, all-tools, maximum-rigor

### Usage

```rust
use reasonkit_core::thinktool::{ProtocolExecutor, ProtocolInput};

let executor = ProtocolExecutor::new()?;

// Execute the powercombo profile
let result = executor.execute_profile(
    "powercombo",
    ProtocolInput::query("Should we acquire this company for $50M?")
).await?;

println!("Confidence: {:.0}%", result.confidence * 100.0);
println!("Success: {}", result.success);
println!("Steps executed: {}", result.steps.len());
println!("Tokens used: {}", result.tokens.total_tokens);
```

### Custom Profile Creation

You can create custom chains by defining a `ReasoningProfile`:

```rust
use reasonkit_core::thinktool::profiles::{
    ReasoningProfile, ChainStep, ChainCondition, StepConfigOverride
};
use std::collections::HashMap;

let custom_profile = ReasoningProfile {
    id: "my_profile".to_string(),
    name: "My Custom Profile".to_string(),
    description: "Custom 3-tool chain".to_string(),
    chain: vec![
        ChainStep {
            protocol_id: "gigathink".to_string(),
            input_mapping: HashMap::from([
                ("query".to_string(), "input.query".to_string()),
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
            protocol_id: "brutalhonesty".to_string(),
            input_mapping: HashMap::from([
                ("work".to_string(), "steps.laserlogic.check_validity".to_string()),
            ]),
            condition: Some(ChainCondition::ConfidenceBelow { threshold: 0.9 }),
            config_override: Some(StepConfigOverride {
                temperature: Some(0.3),
                ..Default::default()
            }),
        },
    ],
    min_confidence: 0.85,
    token_budget: Some(10000),
    tags: vec!["custom".to_string()],
};
```

---

## Profile Selection

### Available Profiles

| Profile      | Modules                           | Min Confidence | Token Budget | Use Case                       |
| ------------ | --------------------------------- | -------------- | ------------ | ------------------------------ |
| `quick`      | gt, ll                            | 70%            | 3,000        | Rapid insights, time-sensitive |
| `balanced`   | gt, ll, br, pg                    | 80%            | 8,000        | Standard analysis              |
| `deep`       | gt, ll, br, pg, bh (conditional)  | 85%            | 12,000       | Thorough analysis              |
| `paranoid`   | gt, ll, br, pg, bh, pg (2nd pass) | 95%            | 18,000       | Maximum verification           |
| `decide`     | ll, br, bh                        | 85%            | 6,000        | Decision support               |
| `scientific` | gt, br, pg                        | 85%            | 8,000        | Research, experiments          |
| `powercombo` | All 5 + validation                | 95%            | 25,000       | Ultimate reasoning             |

### Profile Details

#### Quick Profile

```
GigaThink (max_tokens: 1000)
    |
    v
LaserLogic
```

Best for: Brainstorming, initial exploration, time-constrained situations.

#### Balanced Profile

```
GigaThink
    |
    v
LaserLogic
    |
    v (if confidence < 90%)
BedRock
    |
    v
ProofGuard
```

Best for: General-purpose analysis, most business decisions.

#### Deep Profile

```
GigaThink
    |
    v
LaserLogic
    |
    v
BedRock
    |
    v
ProofGuard
    |
    v (if confidence < 85%)
BrutalHonesty
```

Best for: Important decisions requiring thorough analysis.

#### Paranoid Profile

```
GigaThink
    |
    v
LaserLogic
    |
    v
BedRock
    |
    v
ProofGuard
    |
    v
BrutalHonesty (temp: 0.3)
    |
    v (if confidence < 95%)
ProofGuard (2nd verification)
```

Best for: High-stakes decisions, regulatory compliance, security analysis.

### CLI Usage

```bash
# Quick analysis
rk think --profile quick "What is the market opportunity?"

# Balanced analysis (default)
rk think "Should we enter this market?"

# Deep analysis
rk think --profile deep "Evaluate this acquisition target"

# Paranoid verification
rk think --profile paranoid "Is this claim factually accurate?"

# PowerCombo (ultimate)
rk think --profile powercombo "Should we pivot our business model?"
```

---

## Configuration Reference

### ExecutorConfig

```rust
ExecutorConfig {
    llm: LlmConfig::default(),           // LLM provider configuration
    timeout_secs: 120,                   // Global timeout
    save_traces: false,                  // Save execution traces
    trace_dir: None,                     // Trace output directory
    verbose: false,                      // Verbose logging
    use_mock: false,                     // Use mock LLM for testing
    budget: BudgetConfig::default(),     // Token/cost budget
    cli_tool: None,                      // CLI tool for shell-out
    self_consistency: None,              // Self-consistency voting config
    show_progress: true,                 // Progress indicators
    enable_parallel: false,              // Parallel step execution
    max_concurrent_steps: 4,             // Max concurrent when parallel
}
```

### LLM Provider Configuration

```rust
use reasonkit_core::thinktool::{UnifiedLlmClient, LlmConfig, LlmProvider};

// Anthropic Claude (default)
let client = UnifiedLlmClient::default_anthropic()?;

// OpenAI GPT-4
let client = UnifiedLlmClient::new(LlmConfig {
    provider: LlmProvider::OpenAI,
    model: "gpt-4".to_string(),
    ..Default::default()
})?;

// Groq (ultra-fast inference)
let client = UnifiedLlmClient::groq("llama-3.3-70b-versatile")?;

// xAI Grok
let client = UnifiedLlmClient::grok("grok-2")?;

// DeepSeek
let client = UnifiedLlmClient::new(LlmConfig {
    provider: LlmProvider::DeepSeek,
    model: "deepseek-chat".to_string(),
    ..Default::default()
})?;

// OpenRouter (300+ models)
let client = UnifiedLlmClient::openrouter("anthropic/claude-sonnet-4")?;
```

### Self-Consistency Configuration

```rust
use reasonkit_core::thinktool::consistency::SelfConsistencyConfig;

// Default (5 samples, majority vote)
let config = SelfConsistencyConfig::default();

// Fast (3 samples, 70% threshold)
let config = SelfConsistencyConfig::fast();

// Thorough (10 samples, no early stopping)
let config = SelfConsistencyConfig::thorough();

// Paranoid (15 samples, max accuracy)
let config = SelfConsistencyConfig::paranoid();

// Execute with self-consistency
let (output, consistency_result) = executor.execute_with_self_consistency(
    "balanced",
    input,
    &config
).await?;
```

---

## Real Output Examples

### GigaThink Output

```json
{
  "module": "GigaThink",
  "confidence": 0.78,
  "dimensions": [
    "Economic/Financial",
    "Technological/Innovation",
    "Social/Cultural",
    "Environmental/Sustainability",
    "Political/Regulatory",
    "Psychological/Behavioral",
    "Ethical/Moral",
    "Historical/Evolutionary",
    "Competitive/Market",
    "User Experience/Adoption",
    "Risk/Opportunity",
    "Long-term/Strategic"
  ],
  "perspectives": [
    {
      "id": "perspective_1",
      "dimension": "Economic/Financial",
      "title": "Economic/Financial Analysis",
      "key_insight": "The Economic/Financial lens reveals unique factors that warrant deeper exploration.",
      "confidence": 0.76,
      "quality_score": 0.75
    }
    // ... 10+ more perspectives
  ],
  "themes": [
    {
      "id": "theme_1",
      "title": "Cross-Dimensional Patterns",
      "description": "Patterns that emerge across multiple analytical dimensions.",
      "contributing_count": 4,
      "confidence": 0.78
    }
  ],
  "insights": [
    {
      "id": "insight_1",
      "content": "High-confidence analysis from 10 perspectives suggests actionable opportunities.",
      "actionability": 0.8,
      "confidence": 0.85
    }
  ],
  "cross_validated": true,
  "metadata": {
    "version": "2.1.0",
    "duration_ms": 245,
    "dimensions_count": 12,
    "perspectives_count": 10
  }
}
```

### LaserLogic Output

```json
{
  "module": "LaserLogic",
  "confidence": 0.85,
  "validity": "Valid",
  "validity_explanation": "Argument follows valid Categorical Syllogism pattern",
  "soundness": "Sound",
  "soundness_explanation": "Argument is valid and premises have high confidence (100%)",
  "argument_form": "Categorical Syllogism (term-based reasoning)",
  "fallacies": [],
  "contradictions": [],
  "verdict": "SOUND: Argument is valid with true premises",
  "suggestions": [],
  "reasoning_steps": [
    "Step 1: Identifying argument form...",
    "  Detected: Categorical Syllogism (term-based reasoning)",
    "Step 2: Checking logical validity...",
    "  Argument follows valid Categorical Syllogism pattern",
    "Step 3: Scanning for fallacies...",
    "Step 4: Checking for contradictions...",
    "  No contradictions found.",
    "Step 5: Evaluating soundness...",
    "  Argument is valid and premises have high confidence (100%)"
  ]
}
```

### BrutalHonesty Output

```json
{
  "module": "BrutalHonesty",
  "confidence": 0.52,
  "verdict": "weak",
  "confidence_warning": "Confidence 52% is below threshold 50%",
  "severity_applied": "Standard",
  "analysis": {
    "assumptions": [
      {
        "assumption": "Use of 'will' implies unstated certainty or universality",
        "confidence": 0.75,
        "risk": "Assumes future outcome is certain",
        "likely_valid": false
      },
      {
        "assumption": "Use of 'best' implies unstated certainty or universality",
        "confidence": 0.6,
        "risk": "Assumes optimal status without comparison",
        "likely_valid": true
      }
    ],
    "flaws": [
      {
        "category": "completeness",
        "severity": "major",
        "description": "No counter-arguments or alternative viewpoints presented",
        "trigger": null,
        "remediation": "Steel-man opposing positions before dismissing them"
      }
    ],
    "strengths": [],
    "flaw_count": 1,
    "strength_count": 0
  },
  "devils_advocate": "What if the underlying assumptions about market conditions, timing, or execution are wrong? What specific failure modes have been considered?",
  "critical_fix": "Address completeness issue: No counter-arguments or alternative viewpoints presented",
  "metadata": {
    "input_length": 56,
    "previous_steps_count": 0,
    "skepticism_multiplier": 0.8
  }
}
```

### ProofGuard Output

```json
{
  "module": "ProofGuard",
  "confidence": 0.87,
  "verdict": "verified",
  "claim": "Rust is memory-safe without garbage collection",
  "verification_score": 0.92,
  "is_verified": true,
  "confidence_level": "High",
  "recommendation": "Accept as fact",
  "sources": [
    {
      "name": "Rust Book",
      "tier": "Primary",
      "weight": 1.0,
      "stance": "Support",
      "verified": true,
      "effective_weight": 1.0
    },
    {
      "name": "ACM Paper",
      "tier": "Primary",
      "weight": 1.0,
      "stance": "Support",
      "verified": false,
      "effective_weight": 0.9
    },
    {
      "name": "Tech Blog",
      "tier": "Secondary",
      "weight": 0.7,
      "stance": "Support",
      "verified": false,
      "effective_weight": 0.63
    }
  ],
  "contradictions": [],
  "issues": [],
  "stats": {
    "total_sources": 3,
    "supporting_count": 3,
    "contradicting_count": 0,
    "neutral_count": 0,
    "tier1_count": 2,
    "tier2_count": 1,
    "tier3_count": 0,
    "tier4_count": 0,
    "source_diversity": 0.67,
    "triangulation_weight": 2.53
  }
}
```

### BedRock Output

```json
{
  "module": "BedRock",
  "confidence": 0.65,
  "query": "Electric vehicles are better than gas cars",
  "axioms": [
    {
      "id": 2,
      "statement": "Correlation does not imply causation",
      "fundamentality": 1.0,
      "confidence": 0.95,
      "evidence": []
    },
    {
      "id": 3,
      "statement": "A single counterexample disproves a universal claim",
      "fundamentality": 1.0,
      "confidence": 0.95,
      "evidence": []
    }
  ],
  "assumptions": [
    {
      "id": 4,
      "statement": "Both alternatives must be well-understood",
      "confidence": 0.55,
      "challenges": [
        "Assumption may not hold in all contexts",
        "Implicit bias may be present"
      ]
    }
  ],
  "decomposition": [
    {
      "id": 0,
      "statement": "Electric vehicles are better than gas cars",
      "type": "Contested",
      "fundamentality": 0.0,
      "confidence": 1.0,
      "depth": 0,
      "parent_id": null
    }
    // ... additional principles
  ],
  "reconstruction": [
    {
      "path": [2, 1, 0],
      "confidence": 0.855,
      "complete": true,
      "gaps": []
    }
  ],
  "gaps": [
    {
      "description": "1 contested claim(s) require resolution",
      "severity": 0.8,
      "suggestion": "Provide evidence to resolve contested claims"
    }
  ],
  "insights": [
    "Analysis rests on 2 axiomatic foundation(s)",
    "1 hidden assumption(s) identified that could be challenged",
    "1 critical gap(s) in reasoning require attention",
    "Decomposition reached 1 level(s) of depth",
    "1 contested claim(s) identified - these are debatable"
  ],
  "metadata": {
    "max_depth": 1,
    "total_principles": 5,
    "axioms": 2,
    "assumptions": 1,
    "contested": 1,
    "completeness": 0.7
  }
}
```

---

## Additional Resources

- **API Reference**: See rustdoc at `cargo doc --open`
- **Protocol Definitions**: `/protocols/*.yaml`
- **Examples**: `/examples/*.rs`
- **Benchmarks**: `/benches/`

---

_ReasonKit ThinkTools - Designed, Not Dreamed_
*https://reasonkit.sh*
