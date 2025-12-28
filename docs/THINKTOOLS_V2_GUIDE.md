# ThinkTools V2 Protocol Guide
> Production-Ready Reasoning Module Orchestration for AI Agents

**Version:** 2.0.0
**Status:** Production
**License:** Apache 2.0 (Open Source)
**Created:** 2025-12-22

---

## Table of Contents

1. [Overview](#overview)
2. [Core Modules](#core-modules)
3. [Reasoning Profiles](#reasoning-profiles)
4. [Chaining Protocol](#chaining-protocol)
5. [Confidence Scoring](#confidence-scoring)
6. [JSON Schemas](#json-schemas)
7. [Usage Examples](#usage-examples)
8. [Integration Guide](#integration-guide)
9. [Best Practices](#best-practices)

---

## Overview

ThinkTools V2 is a production-grade reasoning protocol that orchestrates 5 core cognitive modules with formal JSON schemas, automatic confidence scoring, and contradiction detection. It transforms ad-hoc prompting into structured, auditable reasoning chains.

### Key Capabilities

- **Structured Output**: All modules produce JSON conforming to formal schemas
- **Confidence Scoring**: Automatic 0.0-1.0 confidence calculation with factor breakdown
- **Module Chaining**: 4 execution modes (parallel, sequential, feedback, multi-pass)
- **Contradiction Detection**: Cross-module validation and conflict resolution
- **Verbose Thinking**: Step-by-step reasoning traces for transparency
- **Profile-Based**: 4 profiles from quick (70% confidence) to paranoid (95%)

### Architecture

```
User Query
    ↓
Profile Selection (quick/balanced/deep/paranoid)
    ↓
Module Activation (gt, ll, br, pg, bh)
    ↓
Execution (parallel/sequential/feedback/multi-pass)
    ↓
Cross-Module Analysis (contradiction detection)
    ↓
Confidence Scoring (weighted average + penalties)
    ↓
Final Synthesis (structured output + recommendation)
```

---

## Core Modules

### 1. GigaThink (gt) - Divergent Exploration

**Purpose:** Expansive creative thinking with 10+ perspectives

**Capabilities:**
- Multi-perspective generation (10-25 viewpoints)
- Lateral thinking patterns
- Emergent insight detection
- Cross-domain analogies
- Semantic clustering

**Output Schema:** `schemas/thinktools/gigathink_output.json`

**Confidence Factors:**
| Factor | Weight | Formula |
|--------|--------|---------|
| Perspective Diversity | 0.25 | unique_clusters / total_perspectives |
| Insight Novelty | 0.20 | novel_connections / total_connections |
| Domain Coverage | 0.15 | domains_explored / relevant_domains |
| Contradiction Detection | 0.20 | 1.0 - (contradictions / claims) |
| Actionability | 0.20 | actionable_insights / total_insights |

**Typical Duration:** 30-90s
**Use Cases:** Brainstorming, exploration, creative problem-solving

---

### 2. LaserLogic (ll) - Convergent Reasoning

**Purpose:** Precision deductive reasoning with fallacy detection

**Capabilities:**
- Deductive chain construction
- Formal fallacy detection (18 types)
- Argument structure validation
- Premise soundness testing
- Logical dependency graphing

**Output Schema:** `schemas/thinktools/laserlogic_output.json`

**Fallacy Taxonomy (18 types):**
- Ad Hominem
- Straw Man
- False Dichotomy
- Slippery Slope
- Appeal to Authority
- Circular Reasoning
- Hasty Generalization
- Post Hoc Ergo Propter Hoc
- Red Herring
- Appeal to Emotion
- Burden of Proof
- Composition/Division
- Equivocation
- False Cause
- Middle Ground
- No True Scotsman
- Special Pleading
- Tu Quoque

**Confidence Factors:**
| Factor | Weight | Formula |
|--------|--------|---------|
| Premise Validity | 0.30 | valid_premises / total_premises |
| Chain Coherence | 0.25 | 1.0 - (gaps / chain_length) |
| Fallacy Absence | 0.25 | 1.0 - (fallacies / assertions) |
| Conclusion Strength | 0.20 | supporting_evidence / required_evidence |

**Typical Duration:** 20-60s
**Use Cases:** Argument validation, logical analysis, proof verification

---

### 3. BedRock (br) - Foundational Decomposition

**Purpose:** First principles decomposition and axiom rebuilding

**Capabilities:**
- Recursive decomposition (5 levels)
- Axiom extraction and verification
- Assumption surfacing
- Ground truth identification
- Coherence-checked reconstruction

**Output Schema:** `schemas/thinktools/bedrock_output.json`

**Decomposition Levels:**
1. **Surface Observations** - What we see
2. **Underlying Mechanisms** - How it works
3. **Governing Principles** - Why it works
4. **Fundamental Axioms** - What must be true
5. **Universal Truths** - Laws of nature/logic

**Confidence Factors:**
| Factor | Weight | Formula |
|--------|--------|---------|
| Axiom Soundness | 0.35 | verified_axioms / total_axioms |
| Decomposition Completeness | 0.25 | levels_reached / target_depth |
| Assumption Identification | 0.20 | assumptions_surfaced / estimated |
| Rebuild Coherence | 0.20 | 1.0 - (gaps / total_steps) |

**Typical Duration:** 40-120s
**Use Cases:** System design, architecture, fundamental understanding

---

### 4. ProofGuard (pg) - Multi-Source Verification

**Purpose:** Triangulation-based fact verification

**Capabilities:**
- 3-source triangulation (mandatory)
- Source tier classification (1/2/3)
- Contradiction detection
- URL verification tracking
- Confidence interval computation

**Output Schema:** `schemas/thinktools/proofguard_output.json`

**Source Tiers:**
| Tier | Name | Examples | Weight | Confidence Boost |
|------|------|----------|--------|------------------|
| 1 | Authoritative | Official docs, papers, GitHub | 1.0 | +15% |
| 2 | Secondary | Tech blogs, framework docs | 0.8 | +10% |
| 3 | Independent | Community implementations | 0.6 | +5% |

**Verification Levels:**
| Level | Symbol | Criteria | Confidence Multiplier |
|-------|--------|----------|----------------------|
| VERIFIED | ✓ | 3+ sources agree, URLs accessed | 1.15 |
| LIKELY | ~ | 2 sources agree, 1 indirect | 1.05 |
| UNVERIFIED | ? | Single source or missing verification | 0.80 |
| CONTRADICTED | ✗ | Sources disagree | 0.50 |

**Confidence Factors:**
| Factor | Weight | Formula |
|--------|--------|---------|
| Source Count | 0.20 | min(sources / 3, 1.0) |
| Source Diversity | 0.25 | unique_domains / total_sources |
| Tier Quality | 0.20 | weighted_tier_average |
| Contradiction Absence | 0.25 | 1.0 - (contradictions / claims) |
| Verification Completeness | 0.10 | verified_claims / total_claims |

**Typical Duration:** 60-180s
**Use Cases:** Fact-checking, research validation, claim verification

---

### 5. BrutalHonesty (bh) - Adversarial Critique

**Purpose:** Red-team adversarial critique and flaw detection

**Capabilities:**
- Adversarial assumption testing
- Edge case enumeration
- Failure mode prediction
- Cognitive bias detection
- Devil's advocate reasoning

**Output Schema:** `schemas/thinktools/brutalhonesty_output.json`

**Attack Vectors:**
- Assumption Challenges
- Edge Case Stress Tests
- Logical Gap Probing
- Evidence Quality Questioning
- Bias Detection
- Overconfidence Correction
- Hidden Dependency Surfacing
- Failure Mode Analysis

**Cognitive Biases Detected:**
- Confirmation Bias
- Anchoring Bias
- Availability Heuristic
- Hindsight Bias
- Overconfidence
- Sunk Cost Fallacy
- Bandwagon Effect
- Dunning-Kruger
- Status Quo Bias
- Optimism Bias

**Confidence Factors:**
| Factor | Weight | Formula |
|--------|--------|---------|
| Critique Depth | 0.25 | critiques_generated / min_critiques |
| Fatal Flaw Detection | 0.30 | 1.0 if fatal, 0.5 if none |
| Edge Case Coverage | 0.20 | edge_cases_tested / estimated |
| Bias Identification | 0.15 | biases_detected / scan_categories |
| Remediation Quality | 0.10 | issues_addressed / issues_found |

**Typical Duration:** 30-90s
**Use Cases:** Pre-deployment validation, security review, critical decisions

---

## Reasoning Profiles

### Quick Analysis (`--quick`)

**Modules:** GigaThink + LaserLogic
**Execution:** Parallel
**Target Confidence:** 70%
**Duration:** ~60s

**Use Cases:**
- Fast exploratory analysis
- Initial brainstorming
- Low-stakes decisions

**Chain Pattern:**
```
GigaThink (parallel) →
LaserLogic (parallel) →
Synthesize
```

---

### Balanced Reasoning (`--balanced`)

**Modules:** GigaThink + LaserLogic + BedRock + ProofGuard
**Execution:** Sequential
**Target Confidence:** 80%
**Duration:** ~180s

**Use Cases:**
- Standard technical analysis
- Design decisions
- Code architecture

**Chain Pattern:**
```
GigaThink →
BedRock →
LaserLogic →
ProofGuard →
Synthesize
```

---

### Deep Analysis (`--deep`)

**Modules:** All 5 modules
**Execution:** Sequential with feedback
**Target Confidence:** 85%
**Duration:** ~300s

**Use Cases:**
- Complex system design
- Research synthesis
- Critical decisions

**Chain Pattern:**
```
GigaThink →
BedRock →
LaserLogic →
ProofGuard →
BrutalHonesty →
(iterate if bh finds flaws) →
Synthesize
```

---

### Paranoid Verification (`--paranoid`)

**Modules:** All 5 modules
**Execution:** Sequential with multi-pass
**Target Confidence:** 95%
**Duration:** ~600s

**Use Cases:**
- Security-critical decisions
- High-stakes architecture
- Production safety analysis

**Chain Pattern:**
```
Pass 1: gt → br → ll → pg
Pass 2: bh → identify flaws
Pass 3: Re-run failed modules with fixes
Pass 4: Final pg verification → bh re-critique
Synthesize
```

---

## Chaining Protocol

### Execution Modes

#### 1. Parallel
- **Use:** Modules are independent (e.g., gt + ll)
- **Latency:** Low
- **Pattern:** Run simultaneously, merge outputs

#### 2. Sequential
- **Use:** Each module builds on previous
- **Latency:** Medium
- **Pattern:** Pipe outputs forward in chain

#### 3. Sequential with Feedback
- **Use:** Need adversarial validation
- **Latency:** High
- **Pattern:** Run chain, critique, loop back if needed

#### 4. Sequential with Multi-Pass
- **Use:** Maximum verification (paranoid)
- **Latency:** Very High
- **Pattern:** Multiple complete passes until confidence target met

### Dependency Graph

```
Entry Points (no dependencies):
  - GigaThink
  - BedRock

Optional Dependencies:
  - LaserLogic: Works better with gt/br context
  - ProofGuard: Verifies claims from any prior module

Required Dependencies:
  - BrutalHonesty: MUST have prior output to critique
```

### Data Flow

```
GigaThink Output → LaserLogic, ProofGuard, BrutalHonesty
BedRock Output → LaserLogic, ProofGuard
LaserLogic Output → ProofGuard, BrutalHonesty
ProofGuard Output → BrutalHonesty, Synthesis
BrutalHonesty Output → Synthesis, Re-execution Trigger
```

---

## Confidence Scoring

### Overall Formula

```
CONFIDENCE = (
  Σ(module_confidence × module_weight) / Σ(module_weights)
) × contradiction_penalty × profile_multiplier
```

### Module Weights

| Module | Weight | Rationale |
|--------|--------|-----------|
| GigaThink | 0.15 | Exploratory, less definitive |
| LaserLogic | 0.25 | Logical rigor is high value |
| BedRock | 0.20 | Foundational strength matters |
| ProofGuard | 0.30 | Verification is highest weight |
| BrutalHonesty | 0.10 | Adjusts, doesn't determine |

### Contradiction Penalties

| Severity | Penalty Factor |
|----------|---------------|
| None | 1.00 |
| Minor | 0.95 |
| Moderate | 0.85 |
| Major | 0.70 |
| Blocking | 0.50 |

### Profile Multipliers

| Profile | Multiplier | Rationale |
|---------|-----------|-----------|
| Quick | 1.00 | No adjustment |
| Balanced | 1.05 | Multi-module validation |
| Deep | 1.10 | Thorough analysis |
| Paranoid | 1.15 | Maximum verification |

### Calibration Bands

| Confidence | Label | Recommended Action |
|-----------|-------|-------------------|
| 95-100% | Very High | Proceed with implementation |
| 85-94% | High | Proceed with monitoring |
| 70-84% | Moderate | Proceed with caution |
| 50-69% | Low | Gather more data |
| <50% | Insufficient | Do not proceed |

---

## JSON Schemas

All module outputs conform to formal JSON schemas located in:
```
reasonkit-core/schemas/thinktools/
├── gigathink_output.json
├── laserlogic_output.json
├── bedrock_output.json
├── proofguard_output.json
├── brutalhonesty_output.json
└── synthesis_output.json
```

### Schema Validation

```rust
use reasonkit_core::schemas::validate_thinktool_output;

let output = gigathink.execute(query).await?;
validate_thinktool_output("gigathink", &output)?;
```

### Common Schema Fields

All modules include:
- `module`: Module identifier
- `version`: Semver version
- `timestamp`: ISO 8601 execution time
- `query`: Original input
- `confidence`: Confidence object with factors
- `thinking_trace`: Step-by-step reasoning
- `metadata`: Execution stats

---

## Usage Examples

### CLI Usage

```bash
# Quick analysis
rk-core think --profile quick "Should we use RAPTOR?"

# Balanced reasoning
rk-core think --profile balanced "Evaluate BGE-M3 vs E5"

# Deep analysis with verbose output
rk-core think --profile deep --verbose "Is this architecture sound?"

# Paranoid verification
rk-core think --profile paranoid "Security review: OAuth flow"

# Custom module selection
rk-core think --modules gt,ll,pg "Quick fact-check"

# JSON output
rk-core think --profile balanced --output-format json "Query" > result.json
```

### Rust API

```rust
use reasonkit_core::thinktools::{ThinkToolOrchestrator, ReasoningProfile};

#[tokio::main]
async fn main() -> Result<()> {
    let orchestrator = ThinkToolOrchestrator::new();

    let result = orchestrator
        .think("Evaluate RAPTOR hierarchical chunking")
        .profile(ReasoningProfile::Balanced)
        .verbose(true)
        .execute()
        .await?;

    println!("Confidence: {:.1}%", result.confidence.overall * 100.0);
    println!("Recommendation: {}", result.synthesis.recommendation);

    // Access module outputs
    if let Some(proofguard) = result.module_outputs.get("proofguard") {
        println!("Verification: {:?}", proofguard.verification_results);
    }

    Ok(())
}
```

### Python Bindings

```python
from reasonkit import ThinkToolOrchestrator, ReasoningProfile

orchestrator = ThinkToolOrchestrator()

result = orchestrator.think(
    query="Should we adopt RAPTOR?",
    profile=ReasoningProfile.BALANCED,
    verbose=True
)

print(f"Confidence: {result.confidence.overall:.1%}")
print(f"Recommendation: {result.synthesis.recommendation}")

# Access individual modules
for module_name, output in result.module_outputs.items():
    print(f"{module_name} confidence: {output.confidence.overall:.2f}")
```

---

## Integration Guide

### Step 1: Installation

```bash
# Install reasonkit-core
cargo install reasonkit

# Or build from source
cd reasonkit-core
cargo build --release
```

### Step 2: Configuration

Create `config/thinktools.toml`:

```toml
[thinktools]
default_profile = "balanced"
verbose_logging = true
output_format = "json"

[modules.gigathink]
min_perspectives = 10
max_perspectives = 25
creativity_boost = 1.2

[modules.laserlogic]
max_chain_depth = 10
fallacy_detection_threshold = 0.8

[modules.bedrock]
decomposition_depth = 5
axiom_verification_required = true

[modules.proofguard]
minimum_sources = 3
source_diversity_required = true

[modules.brutalhonesty]
min_critiques = 5
attack_intensity = 0.8
```

### Step 3: Basic Usage

```rust
use reasonkit_core::thinktools::ThinkToolOrchestrator;

let orchestrator = ThinkToolOrchestrator::from_config("config/thinktools.toml")?;
let result = orchestrator.think("Your query").execute().await?;
```

### Step 4: Custom Chains

```rust
let result = orchestrator
    .think("Custom analysis")
    .modules(vec!["gigathink", "proofguard", "brutalhonesty"])
    .execution_mode(ExecutionMode::Sequential)
    .confidence_target(0.90)
    .execute()
    .await?;
```

---

## Best Practices

### 1. Choose the Right Profile

| Scenario | Recommended Profile |
|----------|-------------------|
| Quick brainstorm | `--quick` |
| Design decision | `--balanced` |
| Research synthesis | `--deep` |
| Security review | `--paranoid` |

### 2. Interpret Confidence Scores

- **95%+**: Strong evidence, proceed confidently
- **85-94%**: Good analysis, monitor implementation
- **70-84%**: Decent understanding, plan validation
- **50-69%**: Insufficient data, gather more
- **<50%**: Do not proceed, re-analyze

### 3. Handle Contradictions

When modules disagree:
1. **Investigate** the nature of contradiction
2. **Re-run** affected modules with tighter constraints
3. **Consult** additional sources via ProofGuard
4. **Accept lower confidence** if irreconcilable

### 4. Leverage Verbose Thinking

Enable verbose mode to:
- Debug reasoning chains
- Audit AI decisions
- Understand confidence calculations
- Train new team members

### 5. Validate Against Schemas

Always validate JSON output:
```rust
validate_thinktool_output("gigathink", &output)?;
```

### 6. Monitor Performance

Track key metrics:
- Latency per profile
- Confidence calibration (actual vs predicted)
- Contradiction rates
- Re-execution frequency

---

## Advanced Topics

### Custom Module Development

To create a new ThinkTool module:

1. Define JSON schema in `schemas/thinktools/yourmodule_output.json`
2. Implement `ThinkToolModule` trait
3. Add confidence factor calculations
4. Register in orchestrator
5. Update chaining dependencies

### Multi-Pass Optimization

For paranoid profile, optimize multi-pass:
- Cache verified claims across passes
- Skip redundant ProofGuard searches
- Focus BrutalHonesty on weak points from previous pass

### Confidence Tuning

Adjust weights based on your domain:
```rust
let orchestrator = ThinkToolOrchestrator::new()
    .module_weight("proofguard", 0.40)  // Higher for fact-heavy work
    .module_weight("laserlogic", 0.30)  // Higher for formal proofs
    .module_weight("gigathink", 0.10);  // Lower for focused analysis
```

---

## Troubleshooting

### Low Confidence Scores

**Problem:** Confidence consistently below target

**Solutions:**
1. Check for cross-module contradictions
2. Increase source count in ProofGuard
3. Run BrutalHonesty critique
4. Use higher profile (e.g., deep instead of balanced)

### Contradictions Between Modules

**Problem:** GigaThink and LaserLogic disagree

**Solutions:**
1. Review verbose thinking traces
2. Identify specific conflict points
3. Re-run with tighter constraints
4. Accept lower confidence with caveats

### Slow Execution

**Problem:** Profile takes too long

**Solutions:**
1. Use lower profile for time-sensitive work
2. Cache ProofGuard source lookups
3. Parallelize independent modules
4. Optimize token usage in prompts

---

## API Reference

See full API documentation at:
- Rust Docs: `cargo doc --open`
- Online: https://docs.rs/reasonkit-core

---

## Changelog

### v2.0.0 (2025-12-22)
- Initial ThinkTools V2 protocol
- 5 core modules with formal schemas
- 4 reasoning profiles
- Automatic confidence scoring
- Contradiction detection
- Verbose thinking patterns

---

## Contributing

ThinkTools V2 is open source (Apache 2.0). Contributions welcome:

1. Fork `reasonkit-core`
2. Create feature branch
3. Add tests for new modules
4. Submit PR with schema validation

---

## License

Apache 2.0 - See LICENSE file

---

*ThinkTools V2 Protocol | ReasonKit Core | https://reasonkit.sh*
