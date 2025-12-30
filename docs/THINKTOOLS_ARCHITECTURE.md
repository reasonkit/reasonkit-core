# ThinkTools V2 Architecture Diagram

> Visual reference for system design and data flow

**Version:** 2.0.0
**Date:** 2025-12-22

---

## System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                      THINKTOOLS V2 ORCHESTRATOR                     │
│                   Production Reasoning Framework                    │
└────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            ┌───────▼────────┐          ┌────────▼────────┐
            │  Profile       │          │  Configuration   │
            │  Selection     │          │  Manager         │
            └───────┬────────┘          └────────┬─────────┘
                    │                            │
                    └──────────┬─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Module Orchestrator │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐      ┌───────────────┐     ┌───────────────┐
│  Execution    │      │  Dependency   │     │  Confidence   │
│  Engine       │      │  Resolver     │     │  Scorer       │
└───────┬───────┘      └───────┬───────┘     └───────┬───────┘
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   5 Core Modules    │
                    └─────────────────────┘
```

---

## Core Modules Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CORE REASONING MODULES                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  GigaThink   │  │ LaserLogic   │  │  BedRock     │                  │
│  │    (gt)      │  │    (ll)      │  │    (br)      │                  │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤                  │
│  │ Divergent    │  │ Convergent   │  │ Foundational │                  │
│  │ Exploration  │  │ Reasoning    │  │ Decomposition│                  │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤                  │
│  │ 10+ Perspec  │  │ Fallacy Scan │  │ 5 Layers     │                  │
│  │ Analogies    │  │ Logic Chains │  │ Axioms       │                  │
│  │ Insights     │  │ Validation   │  │ Assumptions  │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│         │                 │                 │                            │
│         │                 │                 │                            │
│  ┌──────────────┐  ┌──────────────┐        │                            │
│  │ ProofGuard   │  │BrutalHonesty │        │                            │
│  │    (pg)      │  │    (bh)      │        │                            │
│  ├──────────────┤  ├──────────────┤        │                            │
│  │ Verification │  │ Adversarial  │        │                            │
│  │ Triangulation│  │ Critique     │        │                            │
│  ├──────────────┤  ├──────────────┤        │                            │
│  │ 3+ Sources   │  │ 5+ Critiques │        │                            │
│  │ Tiers 1/2/3  │  │ Edge Cases   │        │                            │
│  │ Contradictions│  │ Biases      │        │                            │
│  └──────────────┘  └──────────────┘        │                            │
│         │                 │                 │                            │
│         └─────────────────┴─────────────────┘                            │
│                           │                                              │
│                    ┌──────▼──────┐                                       │
│                    │  Synthesis   │                                      │
│                    └─────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
Entry Points (No Dependencies)
        │
        ├─── GigaThink (gt)
        │         │
        │         ├──→ Perspectives
        │         ├──→ Insights
        │         └──→ Analogies
        │
        └─── BedRock (br)
                  │
                  ├──→ Decomposition
                  ├──→ Axioms
                  └──→ Assumptions

Optional Dependencies
        │
        └─── LaserLogic (ll)
                  │
                  ├──← (optional) GigaThink output
                  ├──← (optional) BedRock output
                  │
                  ├──→ Premises
                  ├──→ Chains
                  └──→ Fallacies

Verification Layer
        │
        └─── ProofGuard (pg)
                  │
                  ├──← (optional) GigaThink claims
                  ├──← (optional) LaserLogic claims
                  ├──← (optional) BedRock claims
                  │
                  ├──→ Triangulation
                  ├──→ Verification
                  └──→ Contradictions

Critique Layer (Requires Prior Output)
        │
        └─── BrutalHonesty (bh)
                  │
                  ├──← (required) Any module output
                  │
                  ├──→ Critiques
                  ├──→ Edge Cases
                  ├──→ Biases
                  └──→ Re-execution triggers

Final Layer
        │
        └─── Synthesis
                  │
                  ├──← All module outputs
                  │
                  ├──→ Cross-module analysis
                  ├──→ Confidence score
                  └──→ Recommendation
```

---

## Execution Flow: 4 Profiles

### Quick Profile (Parallel)

```
  START
    │
    ├─────────────┬─────────────┐
    │             │             │
    ▼             ▼             ▼
GigaThink    LaserLogic    (parallel)
    │             │
    └──────┬──────┘
           │
           ▼
      Merge Outputs
           │
           ▼
       Synthesize
           │
           ▼
      Confidence: 70%
      Duration: ~60s
```

### Balanced Profile (Sequential)

```
  START
    │
    ▼
GigaThink
    │
    ▼
 BedRock
    │
    ▼
LaserLogic
    │
    ▼
ProofGuard
    │
    ▼
 Synthesize
    │
    ▼
Confidence: 80%
Duration: ~180s
```

### Deep Profile (Sequential + Feedback)

```
  START
    │
    ▼
GigaThink
    │
    ▼
 BedRock
    │
    ▼
LaserLogic
    │
    ▼
ProofGuard
    │
    ▼
BrutalHonesty ──┐
    │           │
    ▼           │
Flaws Found?    │
    │           │
    ├─Yes───────┘ (re-run failed modules)
    │
    ▼─No
 Synthesize
    │
    ▼
Confidence: 85%
Duration: ~300s
```

### Paranoid Profile (Multi-Pass)

```
  START
    │
    ▼
┌─────────────────────┐
│ PASS 1: Initial     │
│ gt→br→ll→pg         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ PASS 2: Critique    │
│ bh → identify flaws │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ PASS 3: Re-execute  │
│ Failed modules      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ PASS 4: Final Check │
│ pg → bh             │
└──────────┬──────────┘
           │
           ▼
       Synthesize
           │
           ▼
    Confidence: 95%
    Duration: ~600s
```

---

## Data Flow Architecture

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│              MODULE EXECUTION PIPELINE                    │
│                                                            │
│  Module 1 Output                                          │
│  ┌──────────────────────────────────────┐                │
│  │ {                                     │                │
│  │   "module": "gigathink",              │                │
│  │   "perspectives": [...],              │                │
│  │   "confidence": {                     │                │
│  │     "overall": 0.82,                  │                │
│  │     "factors": {...}                  │                │
│  │   }                                   │                │
│  │ }                                     │                │
│  └───────────────┬──────────────────────┘                │
│                  │                                         │
│                  ▼                                         │
│  Module 2 Input (includes Module 1 output)                │
│  ┌──────────────────────────────────────┐                │
│  │ Input: User query + gt output         │                │
│  └───────────────┬──────────────────────┘                │
│                  │                                         │
│                  ▼                                         │
│  [Repeat for each module in chain...]                     │
│                                                            │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│           CROSS-MODULE ANALYSIS                           │
│                                                            │
│  Contradiction Detection                                  │
│  ┌────────────────────────────────────────┐              │
│  │ Compare: gt ↔ ll, br ↔ pg, ll ↔ bh... │              │
│  │ Result: ["No conflicts", ...]          │              │
│  └────────────────────────────────────────┘              │
│                                                            │
│  Agreement Tracking                                       │
│  ┌────────────────────────────────────────┐              │
│  │ gt + ll agree on insight X → boost    │              │
│  │ br + pg agree on axiom Y → boost       │              │
│  └────────────────────────────────────────┘              │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│           CONFIDENCE CALCULATION                          │
│                                                            │
│  Step 1: Module Scores                                    │
│  ┌────────────────────────────────────────┐              │
│  │ gt: 0.82, ll: 0.88, br: 0.85,         │              │
│  │ pg: 0.91, bh: 0.75                     │              │
│  └────────────────────────────────────────┘              │
│                                                            │
│  Step 2: Apply Weights                                    │
│  ┌────────────────────────────────────────┐              │
│  │ weighted_avg =                          │              │
│  │   (0.82×0.15) + (0.88×0.25) +          │              │
│  │   (0.85×0.20) + (0.91×0.30) +          │              │
│  │   (0.75×0.10) = 0.861                  │              │
│  └────────────────────────────────────────┘              │
│                                                            │
│  Step 3: Apply Penalties                                  │
│  ┌────────────────────────────────────────┐              │
│  │ contradiction_penalty = 1.00 (none)    │              │
│  │ profile_multiplier = 1.05 (balanced)   │              │
│  │                                         │              │
│  │ final = 0.861 × 1.00 × 1.05 = 0.904    │              │
│  └────────────────────────────────────────┘              │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  FINAL SYNTHESIS                          │
│                                                            │
│  {                                                         │
│    "confidence": {                                         │
│      "overall": 0.904,                                     │
│      "band": "Very High Confidence",                       │
│      "recommended_action": "Proceed with implementation"   │
│    },                                                      │
│    "synthesis": {                                          │
│      "summary": "...",                                     │
│      "key_findings": [...],                                │
│      "decision_matrix": [...]                              │
│    },                                                      │
│    "recommendation": {                                     │
│      "action": "...",                                      │
│      "rationale": "...",                                   │
│      "risks": [...]                                        │
│    }                                                       │
│  }                                                         │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ JSON Output  │
                      │ (validated)  │
                      └──────────────┘
```

---

## Confidence Scoring Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CONFIDENCE SCORING ENGINE                          │
└──────────────────────────────────────────────────────────────────────┘

Input: Module Outputs
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 1: GigaThink                                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Factors:                                                   │  │
│  │   • perspective_diversity:   0.80 × 0.25 = 0.200         │  │
│  │   • insight_novelty:         0.85 × 0.20 = 0.170         │  │
│  │   • domain_coverage:         0.75 × 0.15 = 0.112         │  │
│  │   • contradiction_detection: 0.90 × 0.20 = 0.180         │  │
│  │   • actionability:           0.88 × 0.20 = 0.176         │  │
│  │                                                            │  │
│  │ Module Confidence: 0.838                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  [Repeat for each module...]                                     │
│                                                                   │
│  Final Module Scores:                                            │
│    • GigaThink:      0.838 (weight: 0.15)                       │
│    • LaserLogic:     0.890 (weight: 0.25)                       │
│    • BedRock:        0.870 (weight: 0.20)                       │
│    • ProofGuard:     0.920 (weight: 0.30)                       │
│    • BrutalHonesty:  0.750 (weight: 0.10)                       │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Weighted Average Calculation                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ (0.838 × 0.15) +                                          │  │
│  │ (0.890 × 0.25) +                                          │  │
│  │ (0.870 × 0.20) +                                          │  │
│  │ (0.920 × 0.30) +                                          │  │
│  │ (0.750 × 0.10) = 0.875                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Apply Contradiction Penalty                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Contradictions detected: None                             │  │
│  │ Penalty factor: 1.00                                      │  │
│  │                                                            │  │
│  │ 0.875 × 1.00 = 0.875                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Apply Profile Multiplier                                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Profile: Balanced                                         │  │
│  │ Multiplier: 1.05                                          │  │
│  │                                                            │  │
│  │ 0.875 × 1.05 = 0.919                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Final Confidence: 91.9%                                         │
│  Band: Very High Confidence (95-100%)                           │
│  Action: Proceed with implementation                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Schema Validation Flow

```
┌──────────────────┐
│ Module Execution │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│               Generate JSON Output                        │
│  {                                                         │
│    "module": "gigathink",                                 │
│    "version": "2.0.0",                                    │
│    "timestamp": "2025-12-22T10:00:00Z",                   │
│    "query": "...",                                        │
│    "perspectives": [...],                                 │
│    "confidence": {...}                                    │
│  }                                                         │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│           Load JSON Schema                                │
│  schemas/thinktools/gigathink_output.json                │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│               Validate Against Schema                     │
│                                                            │
│  Required Fields:                                         │
│    ✓ module: "gigathink"                                 │
│    ✓ version: "2.0.0"                                    │
│    ✓ timestamp: valid ISO 8601                           │
│    ✓ query: string                                       │
│    ✓ perspectives: array[10-25]                          │
│    ✓ confidence: object with overall/factors             │
│                                                            │
│  Type Validation:                                         │
│    ✓ All types match schema                              │
│                                                            │
│  Enum Validation:                                         │
│    ✓ All enums within allowed values                     │
└────────┬─────────────────────────────────────────────────┘
         │
         ├─Valid──→ ┌──────────────────┐
         │          │ Store Output     │
         │          │ Continue Pipeline│
         │          └──────────────────┘
         │
         └─Invalid→ ┌──────────────────┐
                    │ Validation Error │
                    │ Halt Execution   │
                    │ Log Details      │
                    └──────────────────┘
```

---

## File Structure Map

```
reasonkit-core/
│
├── protocols/
│   └── thinktools_v2.yaml ────────────┐
│       (31.6 KB)                       │
│       • Module definitions            │ PROTOCOL
│       • Profile specifications        │ LAYER
│       • Chaining rules                │
│       • Confidence formulas           │
│                                       │
├── schemas/                            │
│   └── thinktools/                     │
│       ├── README.md ──────────────────┼─────┐
│       ├── gigathink_output.json       │     │
│       ├── laserlogic_output.json      │     │ SCHEMA
│       ├── bedrock_output.json         │     │ LAYER
│       ├── proofguard_output.json      │     │
│       ├── brutalhonesty_output.json   │     │
│       └── synthesis_output.json       │     │
│           (6 schemas, ~65 KB)         │     │
│                                       │     │
├── docs/                               │     │
│   ├── THINKTOOLS_V2_GUIDE.md ────────┼─────┼───┐
│   │   (32.7 KB - comprehensive)      │     │   │
│   ├── THINKTOOLS_QUICK_REFERENCE.md  │     │   │ DOCS
│   │   (7.8 KB - cheat sheet)         │     │   │ LAYER
│   └── THINKTOOLS_ARCHITECTURE.md     │     │   │
│       (this file)                     │     │   │
│                                       │     │   │
└── THINKTOOLS_V2_IMPLEMENTATION_      │     │   │
    SUMMARY.md ────────────────────────┼─────┼───┘
    (13 KB - project overview)         │     │
                                       │     │
                            ┌──────────┘     │
                            │                │
                            ▼                ▼
                    Implementation    Production Use
                    (Rust/Python)     (CLI/API)
```

---

## Integration Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     REASONKIT ECOSYSTEM                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    reasonkit-core                            │    │
│  │                                                               │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │    │
│  │  │    RAG     │  │   Vector   │  │   ThinkTools V2    │    │    │
│  │  │  Pipeline  │  │  Database  │  │   Orchestrator     │    │    │
│  │  └─────┬──────┘  └─────┬──────┘  └─────────┬──────────┘    │    │
│  │        │               │                    │                │    │
│  │        └───────────────┴────────────────────┘                │    │
│  │                        │                                      │    │
│  │                        ▼                                      │    │
│  │               ┌────────────────┐                             │    │
│  │               │  rk-core CLI   │                             │    │
│  │               └────────┬───────┘                             │    │
│  └────────────────────────┼───────────────────────────────────┘    │
│                            │                                         │
│  ┌─────────────────────────┼───────────────────────────────────┐   │
│  │                    reasonkit-pro                             │   │
│  │                         │                                     │   │
│  │  ┌──────────────────────┼──────────────────────────┐        │   │
│  │  │ Advanced ThinkTools: │                          │        │   │
│  │  │ • AtomicBreak        │                          │        │   │
│  │  │ • HighReflect        │  (Extends V2 Protocol)  │        │   │
│  │  │ • RiskRadar          │                          │        │   │
│  │  │ • DeciDomatic        │                          │        │   │
│  │  │ • SciEngine          │                          │        │   │
│  │  └──────────────────────┴──────────────────────────┘        │   │
│  └───────────────────────────────────────────────────────────────   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

_ThinkTools V2 Architecture | reasonkit-core | 2025-12-22_
*https://reasonkit.sh*
