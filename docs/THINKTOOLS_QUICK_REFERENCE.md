# ThinkTools V2 Quick Reference Card

> One-page cheat sheet for AI agents and developers

**Version:** 2.0.0 | **License:** Apache 2.0

---

## 🧩 Core Modules

| Module | ID | Purpose | Duration | Key Metric |
|--------|----|---------|---------:|------------|
| **GigaThink** | `gt` | Expansive creative thinking | 30-90s | 10+ perspectives |
| **LaserLogic** | `ll` | Deductive reasoning | 20-60s | 0 fallacies |
| **BedRock** | `br` | First principles | 40-120s | 5 layers deep |
| **ProofGuard** | `pg` | Multi-source verification | 60-180s | 3+ sources |
| **BrutalHonesty** | `bh` | Adversarial critique | 30-90s | 5+ critiques |

---

## 🎯 Reasoning Profiles

| Profile | Modules | Execution | Confidence | Duration | Use When |
|---------|---------|-----------|:----------:|---------:|----------|
| `--quick` | gt, ll | Parallel | 70% | ~60s | Fast exploration |
| `--balanced` | gt, ll, br, pg | Sequential | 80% | ~180s | Standard analysis |
| `--deep` | All 5 | Sequential + Feedback | 85% | ~300s | Complex problems |
| `--paranoid` | All 5 | Multi-Pass | 95% | ~600s | Security critical |

---

## ⚖️ Confidence Scoring

### Module Weights
```
ProofGuard:     0.30  (highest - verification is critical)
LaserLogic:     0.25  (logical rigor)
BedRock:        0.20  (foundational strength)
GigaThink:      0.15  (exploratory)
BrutalHonesty:  0.10  (adjusts, doesn't determine)
```

### Contradiction Penalties
```
None:     1.00  (no penalty)
Minor:    0.95  (5% reduction)
Moderate: 0.85  (15% reduction)
Major:    0.70  (30% reduction)
Blocking: 0.50  (50% reduction)
```

### Profile Multipliers
```
Quick:    1.00  (baseline)
Balanced: 1.05  (+5% for thoroughness)
Deep:     1.10  (+10%)
Paranoid: 1.15  (+15%)
```

### Calibration Bands
```
95-100%: Very High → Proceed with implementation
85-94%:  High      → Proceed with monitoring
70-84%:  Moderate  → Proceed with caution
50-69%:  Low       → Gather more data
<50%:    Too Low   → DO NOT PROCEED
```

---

## 🔗 Chain Patterns

### Quick (Parallel)
```
gt ─────┐
        ├─→ Synthesize
ll ─────┘
```

### Balanced (Sequential)
```
gt → br → ll → pg → Synthesize
```

### Deep (With Feedback)
```
gt → br → ll → pg → bh ──→ Synthesize
                     └─→ Re-run if flaws found
```

### Paranoid (Multi-Pass)
```
Pass 1: gt → br → ll → pg
Pass 2: bh → identify flaws
Pass 3: Re-run failed modules
Pass 4: Final pg → bh → Synthesize
```

---

## 🎯 Fuzzy Aliases (Intent-Based Routing)

ReasonKit supports intuitive aliases for all commands:

| Intent | Aliases | Routes To |
|--------|---------|-----------|
| **Creative** | `gt`, `giga`, `creative`, `rainbow` | GigaThink |
| **Logical** | `ll`, `laser`, `logic`, `deduce` | LaserLogic |
| **Foundation** | `br`, `roots`, `foundation`, `base` | BedRock |
| **Verify** | `pg`, `proof`, `verify`, `check`, `guard` | ProofGuard |
| **Critique** | `bh`, `brutal`, `critique`, `honest` | BrutalHonesty |

**Examples:**
```bash
rk-core think --protocol rainbow "Brainstorm ideas"       # → GigaThink
rk-core think --protocol verify "Check this claim"        # → ProofGuard
rk-core think --protocol brutal "Find flaws"              # → BrutalHonesty
rk-core think --protocol roots "Break down assumptions"   # → BedRock
rk-core think --protocol logic "Validate this argument"   # → LaserLogic
```

---

## 📝 CLI Commands

```bash
# Quick analysis
rk-core think --profile quick "Query"

# Balanced (default)
rk-core think "Query"
rk-core think --profile balanced "Query"

# Deep analysis with verbose logs
rk-core think --profile deep --verbose "Query"

# Paranoid verification
rk-core think --profile paranoid "Query"

# Custom modules
rk-core think --modules gt,pg,bh "Query"

# JSON output
rk-core think --output-format json "Query" > output.json

# Set confidence target
rk-core think --confidence-target 0.90 "Query"
```

---

## 🦀 Rust API

```rust
use reasonkit_core::thinktools::{ThinkToolOrchestrator, ReasoningProfile};

let orchestrator = ThinkToolOrchestrator::new();

let result = orchestrator
    .think("Your query here")
    .profile(ReasoningProfile::Balanced)
    .verbose(true)
    .execute()
    .await?;

println!("Confidence: {:.1}%", result.confidence.overall * 100.0);
```

---

## 🐍 Python API

```python
from reasonkit import ThinkToolOrchestrator, ReasoningProfile

orchestrator = ThinkToolOrchestrator()

result = orchestrator.think(
    query="Your query",
    profile=ReasoningProfile.BALANCED,
    verbose=True
)

print(f"Confidence: {result.confidence.overall:.1%}")
```

---

## 📊 Module Outputs

### GigaThink
```json
{
  "perspectives": [/* 10-25 viewpoints */],
  "emergent_insights": [/* cross-perspective insights */],
  "cross_domain_analogies": [/* analogies */],
  "confidence": {"overall": 0.85, "factors": {...}}
}
```

### LaserLogic
```json
{
  "premises": [/* extracted premises */],
  "deductive_chains": [/* logical chains */],
  "fallacies_detected": [/* 18 fallacy types */],
  "conclusion": {"statement": "...", "strength": 0.90}
}
```

### BedRock
```json
{
  "decomposition_layers": [/* 5 layers */],
  "axioms": [/* fundamental axioms */],
  "assumptions_surfaced": [/* hidden assumptions */],
  "reconstruction": {"steps": [...], "coherence": 0.95}
}
```

### ProofGuard
```json
{
  "claims_extracted": [/* factual claims */],
  "verification_results": [/* per-claim verification */],
  "triangulation_table": [/* 3-source table */],
  "contradictions": [/* conflicts found */]
}
```

### BrutalHonesty
```json
{
  "critiques": [/* adversarial critiques */],
  "edge_cases": [/* failure scenarios */],
  "biases_detected": [/* cognitive biases */],
  "overall_assessment": {"verdict": "...", "fatal_flaws_found": 0}
}
```

---

## 🔍 Contradiction Detection

### Comparison Pairs
```
GigaThink ↔ LaserLogic:  Are gt insights logically sound?
BedRock ↔ ProofGuard:    Do br axioms match pg facts?
LaserLogic ↔ BrutalHonesty: Did bh find fallacies ll missed?
ProofGuard ↔ BrutalHonesty: Did bh find unverified claims?
```

### Resolution Strategy
1. Investigate nature of contradiction
2. Re-run affected modules with tighter constraints
3. Lower confidence to minimum of conflicting modules
4. Require explicit resolution before proceeding

---

## 🛠️ Configuration

### `config/thinktools.toml`
```toml
[thinktools]
default_profile = "balanced"
verbose_logging = true

[modules.gigathink]
min_perspectives = 10
max_perspectives = 25

[modules.laserlogic]
fallacy_detection_threshold = 0.8

[modules.bedrock]
decomposition_depth = 5

[modules.proofguard]
minimum_sources = 3

[modules.brutalhonesty]
attack_intensity = 0.8
```

---

## ✅ Best Practices

### Choose Right Profile
- **Brainstorming?** → `--quick`
- **Design decision?** → `--balanced`
- **Research synthesis?** → `--deep`
- **Security review?** → `--paranoid`

### Interpret Confidence
- **95%+** = Ship it
- **85-94%** = Monitor closely
- **70-84%** = Plan validation
- **50-69%** = Need more data
- **<50%** = Don't proceed

### Handle Low Confidence
1. Check for contradictions
2. Add more sources (ProofGuard)
3. Run BrutalHonesty critique
4. Use higher profile

---

## 📚 Resources

| Resource | Location |
|----------|----------|
| **Full Protocol** | `protocols/thinktools_v2.yaml` |
| **User Guide** | `docs/THINKTOOLS_V2_GUIDE.md` |
| **JSON Schemas** | `schemas/thinktools/*.json` |
| **API Docs** | https://docs.rs/reasonkit-core |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Low confidence | Run higher profile (deep/paranoid) |
| Module conflicts | Check verbose logs, re-run with constraints |
| Slow execution | Use lower profile or cache ProofGuard sources |
| Schema validation fails | Ensure all required fields present |

---

## 📐 Formulas

### Overall Confidence
```
CONFIDENCE =
  (Σ(module_conf × module_weight) / Σ(weights))
  × contradiction_penalty
  × profile_multiplier
```

### Module Confidence (Generic)
```
MODULE_CONF = Σ(factor_value × factor_weight)
```

### Contradiction Penalty
```
PENALTY =
  1.00 if no contradictions
  0.95 if minor
  0.85 if moderate
  0.70 if major
  0.50 if blocking
```

---

## 🎓 Decision Tree

```
Need reasoning? → Yes
    ↓
Time critical? → Yes → --quick (70% conf, 60s)
    ↓ No
    ↓
Standard analysis? → Yes → --balanced (80% conf, 180s)
    ↓ No
    ↓
Complex problem? → Yes → --deep (85% conf, 300s)
    ↓ No
    ↓
Security/Safety critical? → Yes → --paranoid (95% conf, 600s)
```

---

*ThinkTools V2 Quick Reference | reasonkit-core | Apache 2.0*
*https://reasonkit.sh*
