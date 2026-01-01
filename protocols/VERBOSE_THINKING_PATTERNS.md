# VERBOSE THINKING PATTERNS

> Comprehensive Step-by-Step Reasoning Templates for ReasonKit ThinkTools
> Making AI reasoning transparent, auditable, and debuggable

```
██╗   ██╗███████╗██████╗ ██████╗  ██████╗ ███████╗███████╗
██║   ██║██╔════╝██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝
██║   ██║█████╗  ██████╔╝██████╔╝██║   ██║███████╗█████╗
╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══██╗██║   ██║╚════██║██╔══╝
 ╚████╔╝ ███████╗██║  ██║██████╔╝╚██████╔╝███████║███████╗
  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚══════╝╚══════╝
    ████████╗██╗  ██╗██╗███╗   ██╗██╗  ██╗██╗███╗   ██╗ ██████╗
    ╚══██╔══╝██║  ██║██║████╗  ██║██║ ██╔╝██║████╗  ██║██╔════╝
       ██║   ███████║██║██╔██╗ ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗
       ██║   ██╔══██║██║██║╚██╗██║██╔═██╗ ██║██║╚██╗██║██║   ██║
       ██║   ██║  ██║██║██║ ╚████║██║  ██╗██║██║ ╚████║╚██████╔╝
       ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
```

---

## DOCUMENT METADATA

| Field       | Value                                                                |
| ----------- | -------------------------------------------------------------------- |
| **Version** | 1.0.0                                                                |
| **Schema**  | `reasonkit-verbose-thinking-v1`                                      |
| **Created** | 2025-12-22                                                           |
| **Team**    | Team Epsilon (ThinkTools Enhancement)                                |
| **License** | Apache 2.0                                                           |
| **Status**  | PRODUCTION READY                                                     |
| **Purpose** | Define detailed step-by-step thinking patterns for 5 core ThinkTools |

**Inherits From:** `thinktools.yaml`, `proofguard-deep-research-protocol.yaml`

**AI Consultations:**

- Claude Opus 4.5 (design principles for transparency + usefulness)
- Gemini 2.0 Flash (LaserLogic pattern structure)

---

## TABLE OF CONTENTS

1. [Design Principles](#design-principles)
2. [Universal Pattern Structure](#universal-pattern-structure)
3. [Module 1: GigaThink (Divergent Thinking)](#module-1-gigathink)
4. [Module 2: LaserLogic (Deductive Reasoning)](#module-2-laserlogic)
5. [Module 3: BedRock (First Principles)](#module-3-bedrock)
6. [Module 4: ProofGuard (Verification)](#module-4-proofguard)
7. [Module 5: BrutalHonesty (Adversarial)](#module-5-brutalhonesty)
8. [Confidence Calculation Formulas](#confidence-calculation)
9. [Output Formats (JSON + Markdown)](#output-formats)
10. [Integration Guide](#integration-guide)

---

## DESIGN PRINCIPLES

> Based on consultation with Claude Opus 4.5 on transparency vs. usefulness balance

### Principle 1: Layered Verbosity with Progressive Disclosure

**The Problem:** Full verbose output drowns users in noise. Minimal output hides reasoning.

**The Solution:** 4-level verbosity structure that reveals depth on demand:

| Level | Name     | Content                                      | Use Case             |
| ----- | -------- | -------------------------------------------- | -------------------- |
| **0** | Summary  | Single line: `[Module] Verdict + Confidence` | Quick status check   |
| **1** | Standard | Step headers + key findings                  | Default verbose mode |
| **2** | Detailed | Sub-step details + intermediate calculations | Debugging reasoning  |
| **3** | Trace    | Full JSON provenance + evidence links        | Machine audit trail  |

**Implementation:**

```bash
# User can control verbosity
rk-core think "query" --verbose-level 1  # Standard (default)
rk-core think "query" --verbose-level 2  # Detailed
rk-core think "query" --verbose-level 3  # Full trace
```

---

### Principle 2: Confidence as First-Class Citizen

**The Problem:** "Confidence: 82%" is opaque - users can't debug why it's not 95%.

**The Solution:** Every confidence score decomposes into weighted factors with individual scores visible:

```
[LaserLogic] Module Confidence: 0.78

Confidence Breakdown:
├─ premise_validity:     0.90 × 0.30 = 0.270
├─ chain_coherence:      0.85 × 0.25 = 0.213
├─ fallacy_absence:      0.60 × 0.25 = 0.150  ⚠️ BOTTLENECK
└─ conclusion_strength:  0.75 × 0.20 = 0.150
                         ────────────────────
                         Raw: 0.783 → Calibrated: 0.78
```

**Key Requirements:**

- Show factor name + individual score + weight contribution
- Highlight lowest-scoring factor (the bottleneck)
- Display raw vs. calibrated confidence (if calibration model exists)
- Include contradiction penalty explicitly when applied

---

### Principle 3: Dual-Format Output (Human + Machine)

**The Problem:** Humans need prose; pipelines need structured data.

**The Solution:** Every module outputs BOTH markdown and JSON in parallel:

```yaml
# Human-readable (thinking_trace.md)
[ProofGuard] Verifying claim: "Rust prevents data races"
  → Source 1 (Tier 1): Rust docs - SUPPORTS ✓
  → Source 2 (Tier 2): Google study - SUPPORTS ✓
  → Source 3 (Tier 3): HN discussion - SUPPORTS ✓
  Verdict: VERIFIED (3/3 agree, no contradictions)

# Machine-readable (thinking_trace.json)
{
  "module": "proofguard",
  "claim": "Rust prevents data races",
  "sources": [
    {"url": "...", "tier": 1, "verdict": "supports", "confidence": 0.95},
    ...
  ],
  "triangulation_status": "verified",
  "confidence": 0.89
}
```

---

### Principle 4: Failure-First Emphasis

**The Problem:** Verbose logs bury failures in walls of successful steps.

**The Solution:** Surface failures and anomalies at the top:

```
[BrutalHonesty] CRITIQUE SUMMARY (2 blocking, 3 advisory)

⛔ BLOCKING ISSUES:
  1. Assumption untested: "Users have stable network" (line 47)
  2. Edge case missed: Empty input handling (line 89)

⚠️ ADVISORY ISSUES:
  3. Weak evidence: Benchmark from 2019 may be outdated
  4. Bias detected: Anchoring on initial estimate
  5. Missing perspective: Competitor analysis

✅ PASSED CHECKS (12)
  [Expand for details...]
```

**Severity Levels:**

- `blocking` - Must fix before proceeding (red/⛔)
- `major` - Should fix, impacts quality (orange/⚠️)
- `minor` - Optional improvement (yellow/⚡)
- `advisory` - FYI, not critical (blue/ℹ️)

---

### Principle 5: Explicit Provenance Links (The "Why Chain")

**The Problem:** Conclusions float free from supporting evidence.

**The Solution:** Every claim links back to its sources via explicit provenance:

```
[Synthesis] Final Recommendation: Adopt RAPTOR chunking

Provenance Chain:
├─ [GigaThink:P3] "Hierarchical retrieval matches cognitive patterns"
│   └─ [ProofGuard:S1] arxiv.org/abs/2401.18059 (RAPTOR paper)
├─ [BedRock:A2] Axiom: "Summarization preserves semantic density"
│   └─ [ProofGuard:S4] HotpotQA benchmark (+18% vs flat chunking)
└─ [LaserLogic:C1] "If context > 16K tokens, hierarchy improves recall"
    └─ [BrutalHonesty:V1] Challenged & survived: edge case tested

Confidence: 84% (deep profile)
Provenance Depth: 3 hops (conclusion → module → evidence)
```

**ID Format:** `[Module:Type:N]`

- `Module`: GigaThink, LaserLogic, BedRock, ProofGuard, BrutalHonesty
- `Type`: P=Perspective, A=Axiom, C=Conclusion, S=Source, V=Validation
- `N`: Sequential number within module execution

---

## UNIVERSAL PATTERN STRUCTURE

All ThinkTools follow this common structure:

```yaml
verbose_output:
  header:
    - module_name
    - activation_time
    - input_summary

  body:
    - step_1_initialization
    - step_2_core_processing
    - step_3_analysis
    - step_4_synthesis
    - step_5_validation

  footer:
    - confidence_breakdown
    - key_findings_summary
    - next_steps_or_warnings

  metadata:
    - execution_time
    - token_count
    - module_version
```

---

## MODULE 1: GigaThink (Divergent Thinking)

### Purpose

Expansive creative thinking engine that generates 10+ perspectives on a problem, exploring unconventional angles and emergent possibilities.

### Verbose Thinking Pattern (Level 1: Standard)

```
═══════════════════════════════════════════════════════════════════════════
[GigaThink] EXPANSIVE PERSPECTIVE GENERATION
═══════════════════════════════════════════════════════════════════════════

Input Query: "Should we adopt RAPTOR for hierarchical retrieval?"

[STEP 1] Initialization
  ├─ Loading context constraints: reasonkit-core RAG architecture
  ├─ Target perspectives: 10 minimum
  └─ Creativity boost: 1.2x (high)

[STEP 2] Seed Perspective Generation
  ├─ [P1:ANALYTICAL] "Performance optimization lens"
  │   ├─ Focus: Benchmark data, latency impact
  │   └─ Initial insight: Need to compare RAPTOR vs flat chunking metrics
  │
  ├─ [P2:CREATIVE] "Cognitive architecture alignment"
  │   ├─ Focus: How humans organize knowledge hierarchically
  │   └─ Initial insight: Tree structure mirrors human memory retrieval
  │
  └─ [P3:ADVERSARIAL] "Risk and failure modes"
      ├─ Focus: Edge cases, implementation complexity
      └─ Initial insight: Summarization loss, depth tuning challenges

[STEP 3] Perspective Expansion (Rotation + Cross-Domain Analogies)
  ├─ [P4:IMPLEMENTATION] "Developer experience view"
  │   └─ Analogy: Git commits (flat) vs Git tree (hierarchical)
  │
  ├─ [P5:COST-BENEFIT] "Token economics analysis"
  │   └─ Insight: Summarization tokens vs improved recall ROI
  │
  ├─ [P6:USER-CENTRIC] "End-user query experience"
  │   └─ Focus: Response quality, multi-hop question handling
  │
  ├─ [P7:COMPETITIVE] "Ecosystem positioning"
  │   └─ Comparison: LlamaIndex auto-merging, Pinecone namespaces
  │
  ├─ [P8:MAINTENANCE] "Long-term sustainability"
  │   └─ Concern: Index rebuild frequency, storage overhead
  │
  ├─ [P9:SCIENTIFIC] "Academic validation"
  │   └─ Evidence: QuALITY benchmark +20%, NarrativeQA +18%
  │
  ├─ [P10:INTEGRATION] "Compatibility with existing stack"
  │   └─ Technical: Qdrant + Tantivy hybrid search interaction
  │
  └─ [P11:STRATEGIC] "Project roadmap alignment"
      └─ Vision: Premium feature for reasonkit-pro?

[STEP 4] Cross-Domain Analogy Mining
  ├─ Analogy 1: Library organization (Dewey Decimal = hierarchical)
  ├─ Analogy 2: File systems (directories vs flat storage)
  ├─ Analogy 3: Neural networks (hierarchical feature extraction)
  └─ Analogy 4: Corporate org charts (span of control limits)

[STEP 5] Emergent Pattern Synthesis
  ├─ Cluster 1: "Performance gains are real but context-dependent"
  ├─ Cluster 2: "Implementation complexity is moderate, not high"
  └─ Cluster 3: "Best for long documents (>10k tokens)"

[STEP 6] Insight Scoring (Novelty × Actionability)
  ├─ Top Insight 1: "RAPTOR works when summarization ≈ semantic compression"
  │   └─ Score: 0.92 (novel=0.88, actionable=0.96)
  ├─ Top Insight 2: "Depth tuning is critical - 3 levels optimal for most docs"
  │   └─ Score: 0.85 (novel=0.75, actionable=0.95)
  └─ Top Insight 3: "Consider hybrid: RAPTOR for docs >10k, flat for shorter"
      └─ Score: 0.89 (novel=0.82, actionable=0.96)

───────────────────────────────────────────────────────────────────────────
CONFIDENCE BREAKDOWN
───────────────────────────────────────────────────────────────────────────
Factor Scores:
├─ perspective_diversity:    0.88 × 0.25 = 0.220
│   └─ (11 perspectives across 8 semantic clusters)
├─ insight_novelty:          0.82 × 0.20 = 0.164
│   └─ (9 novel connections found / 11 total)
├─ domain_coverage:          0.75 × 0.15 = 0.113
│   └─ (6 domains explored / 8 relevant)
├─ contradiction_detection:  0.95 × 0.20 = 0.190
│   └─ (1 contradiction / 11 total claims)
└─ actionability:            0.91 × 0.20 = 0.182
    └─ (10 actionable insights / 11 total)

Raw Confidence: 0.869
Calibration Adjustment: 0.95x (diverse perspectives boost)
Final Module Confidence: 0.83

───────────────────────────────────────────────────────────────────────────
KEY FINDINGS
───────────────────────────────────────────────────────────────────────────
✅ 11 perspectives generated (target: 10+)
✅ 8 distinct semantic clusters identified
✅ 3 high-value actionable insights surfaced
⚠️ 1 minor contradiction detected (implementation complexity estimates varied)

Next Step: Feed to LaserLogic for logical validation of insights
───────────────────────────────────────────────────────────────────────────
Execution Time: 34.2s | Tokens: 3,847 | Version: gigathink-v2.0.0
═══════════════════════════════════════════════════════════════════════════
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "GigaThink Output Schema",
  "type": "object",
  "required": ["module", "perspectives", "insights", "confidence"],
  "properties": {
    "module": {
      "type": "string",
      "const": "gigathink"
    },
    "input_query": {
      "type": "string"
    },
    "perspectives": {
      "type": "array",
      "minItems": 10,
      "items": {
        "type": "object",
        "required": ["id", "type", "description", "insight"],
        "properties": {
          "id": {
            "type": "string",
            "pattern": "^P[0-9]+$"
          },
          "type": {
            "type": "string",
            "enum": [
              "analytical",
              "creative",
              "adversarial",
              "implementation",
              "cost-benefit",
              "user-centric",
              "competitive",
              "maintenance",
              "scientific",
              "integration",
              "strategic"
            ]
          },
          "description": {
            "type": "string"
          },
          "insight": {
            "type": "string"
          },
          "novelty_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          },
          "actionability_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          }
        }
      }
    },
    "analogies": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "domain": { "type": "string" },
          "analogy": { "type": "string" },
          "relevance": { "type": "number" }
        }
      }
    },
    "emergent_patterns": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "cluster_id": { "type": "string" },
          "theme": { "type": "string" },
          "perspectives": { "type": "array", "items": { "type": "string" } }
        }
      }
    },
    "insights": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["text", "score"],
        "properties": {
          "text": { "type": "string" },
          "score": { "type": "number" },
          "novelty": { "type": "number" },
          "actionability": { "type": "number" }
        }
      }
    },
    "confidence": {
      "$ref": "#/definitions/ConfidenceScore"
    }
  },
  "definitions": {
    "ConfidenceScore": {
      "type": "object",
      "required": ["final", "factors"],
      "properties": {
        "final": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "raw": {
          "type": "number"
        },
        "factors": {
          "type": "object",
          "properties": {
            "perspective_diversity": { "type": "number" },
            "insight_novelty": { "type": "number" },
            "domain_coverage": { "type": "number" },
            "contradiction_detection": { "type": "number" },
            "actionability": { "type": "number" }
          }
        }
      }
    }
  }
}
```

---

## MODULE 2: LaserLogic (Deductive Reasoning)

### Purpose

Precision deductive reasoning with formal fallacy detection, logical chain validation, and argument structure analysis.

### Verbose Thinking Pattern (Level 1: Standard)

```
═══════════════════════════════════════════════════════════════════════════
[LaserLogic] DEDUCTIVE REASONING & FALLACY DETECTION
═══════════════════════════════════════════════════════════════════════════

Input: Reasoning from GigaThink (11 perspectives on RAPTOR adoption)

[STEP 1] Premise Extraction
  ├─ Premise 1: "RAPTOR improves performance by +20% on QuALITY benchmark"
  │   ├─ Type: Empirical claim
  │   ├─ Source: GigaThink:P9 (Scientific perspective)
  │   └─ Validity: UNVERIFIED (needs ProofGuard)
  │
  ├─ Premise 2: "Hierarchical retrieval matches human memory organization"
  │   ├─ Type: Analogical reasoning
  │   ├─ Source: GigaThink:P2 (Creative perspective)
  │   └─ Validity: PLAUSIBLE (cognitive science alignment)
  │
  ├─ Premise 3: "Implementation complexity is moderate"
  │   ├─ Type: Subjective assessment
  │   ├─ Source: GigaThink:P4 (Implementation perspective)
  │   └─ Validity: CONTESTED (GigaThink:P3 said "high complexity")
  │
  └─ Premise 4: "RAPTOR best for documents >10k tokens"
      ├─ Type: Conditional claim
      ├─ Source: GigaThink:P11 (Emergent pattern)
      └─ Validity: REASONABLE (follows from context window limits)

[STEP 2] Logical Chain Construction
  ├─ Chain 1: Performance Argument
  │   ├─ P1: RAPTOR improves benchmark performance by +20%
  │   ├─ P2: Benchmark performance correlates with real-world quality
  │   ├─ C1: Therefore, RAPTOR will improve real-world retrieval quality
  │   └─ Strength: MODERATE (depends on benchmark validity)
  │
  ├─ Chain 2: Cognitive Alignment Argument
  │   ├─ P1: Human memory is hierarchically organized
  │   ├─ P2: Systems that mirror human cognition are more effective
  │   ├─ P3: RAPTOR uses hierarchical organization
  │   ├─ C1: Therefore, RAPTOR aligns with human cognition
  │   └─ Strength: WEAK (analogical, not causal)
  │
  └─ Chain 3: Context Suitability Argument
      ├─ P1: Long documents (>10k tokens) exceed single-chunk context
      ├─ P2: Hierarchical summarization compresses while preserving semantics
      ├─ P3: RAPTOR implements hierarchical summarization
      ├─ C1: Therefore, RAPTOR is well-suited for long documents
      └─ Strength: STRONG (deductive, sound premises)

[STEP 3] Fallacy Scan (18 Types)
  ├─ [SCAN] Ad hominem: NOT DETECTED
  ├─ [SCAN] Straw man: NOT DETECTED
  ├─ [SCAN] False dichotomy: NOT DETECTED
  ├─ [SCAN] Slippery slope: NOT DETECTED
  ├─ [SCAN] Appeal to authority: ⚠️ POSSIBLE
  │   └─ Location: Chain 1, P2 (benchmark performance assumption)
  │   └─ Severity: MINOR (benchmarks are legitimate evidence)
  ├─ [SCAN] Circular reasoning: NOT DETECTED
  ├─ [SCAN] Hasty generalization: ⚠️ DETECTED
  │   └─ Location: Chain 2, P2 ("Systems that mirror cognition are more effective")
  │   └─ Severity: MODERATE (overgeneralization from analogy)
  ├─ [SCAN] Post hoc ergo propter hoc: NOT DETECTED
  ├─ [SCAN] Red herring: NOT DETECTED
  ├─ [SCAN] Appeal to emotion: NOT DETECTED
  ├─ [SCAN] Burden of proof: NOT DETECTED
  ├─ [SCAN] Composition/Division: NOT DETECTED
  ├─ [SCAN] Equivocation: NOT DETECTED
  ├─ [SCAN] False cause: NOT DETECTED
  ├─ [SCAN] Middle ground: NOT DETECTED
  ├─ [SCAN] No true Scotsman: NOT DETECTED
  ├─ [SCAN] Special pleading: NOT DETECTED
  └─ [SCAN] Tu quoque: NOT DETECTED

[STEP 4] Logical Gap Analysis
  ├─ Gap 1: Missing evidence for "benchmark → real-world" correlation
  │   └─ Impact: Weakens Chain 1 conclusion strength
  ├─ Gap 2: No empirical data on "moderate" complexity claim
  │   └─ Impact: Premise 3 remains contested
  └─ Gap 3: Lack of counter-evidence consideration
      └─ Impact: Arguments lack adversarial stress-testing

[STEP 5] Conclusion Strength Scoring
  ├─ Chain 1 (Performance): 0.68
  │   └─ Reasoning: Moderate (valid structure, but unverified premise)
  ├─ Chain 2 (Cognitive): 0.42
  │   └─ Reasoning: Weak (analogical, hasty generalization detected)
  └─ Chain 3 (Suitability): 0.85
      └─ Reasoning: Strong (sound deductive structure, verified premises)

[STEP 6] Overall Logical Soundness
  ├─ Valid deductive chains: 3/3
  ├─ Sound premises: 2/4 verified
  ├─ Fallacies detected: 1 moderate, 1 minor
  └─ Logical gaps: 3 identified

───────────────────────────────────────────────────────────────────────────
CONFIDENCE BREAKDOWN
───────────────────────────────────────────────────────────────────────────
Factor Scores:
├─ premise_validity:        0.50 × 0.30 = 0.150
│   └─ (2 verified premises / 4 total)
├─ chain_coherence:         0.91 × 0.25 = 0.228
│   └─ (1.0 - (3 gaps / 33 chain steps))
├─ fallacy_absence:         0.82 × 0.25 = 0.205
│   └─ (1.0 - (2 fallacies / 11 assertions))
└─ conclusion_strength:     0.65 × 0.20 = 0.130
    └─ (Average of chain strengths: 0.65)

Raw Confidence: 0.713
Calibration Adjustment: 1.0x (standard)
Final Module Confidence: 0.71

───────────────────────────────────────────────────────────────────────────
KEY FINDINGS
───────────────────────────────────────────────────────────────────────────
✅ 3 deductive chains constructed and validated
✅ 18 fallacy types scanned systematically
⚠️ 1 hasty generalization detected (Chain 2)
⚠️ 2 unverified premises (need ProofGuard)
❌ Logical gaps present in all 3 chains

Recommended Next Steps:
1. Send unverified premises (P1, P3) to ProofGuard for triangulation
2. Request BrutalHonesty critique of Chain 2 (weakest)
3. Strengthen Chain 1 with empirical benchmark-to-reality correlation data

───────────────────────────────────────────────────────────────────────────
Execution Time: 28.7s | Tokens: 2,934 | Version: laserlogic-v2.0.0
═══════════════════════════════════════════════════════════════════════════
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LaserLogic Output Schema",
  "type": "object",
  "required": ["module", "premises", "chains", "fallacies", "confidence"],
  "properties": {
    "module": {
      "type": "string",
      "const": "laserlogic"
    },
    "premises": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "text", "type", "validity"],
        "properties": {
          "id": { "type": "string" },
          "text": { "type": "string" },
          "type": {
            "type": "string",
            "enum": [
              "empirical",
              "analogical",
              "subjective",
              "conditional",
              "axiomatic"
            ]
          },
          "validity": {
            "type": "string",
            "enum": [
              "verified",
              "plausible",
              "contested",
              "unverified",
              "false"
            ]
          },
          "source": { "type": "string" }
        }
      }
    },
    "chains": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "premises", "conclusion", "strength"],
        "properties": {
          "id": { "type": "string" },
          "name": { "type": "string" },
          "premises": {
            "type": "array",
            "items": { "type": "string" }
          },
          "conclusion": { "type": "string" },
          "strength": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          },
          "gaps": {
            "type": "array",
            "items": { "type": "string" }
          }
        }
      }
    },
    "fallacies": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "location", "severity"],
        "properties": {
          "type": {
            "type": "string",
            "enum": [
              "ad_hominem",
              "straw_man",
              "false_dichotomy",
              "slippery_slope",
              "appeal_to_authority",
              "circular_reasoning",
              "hasty_generalization",
              "post_hoc",
              "red_herring",
              "appeal_to_emotion",
              "burden_of_proof",
              "composition_division",
              "equivocation",
              "false_cause",
              "middle_ground",
              "no_true_scotsman",
              "special_pleading",
              "tu_quoque"
            ]
          },
          "location": { "type": "string" },
          "severity": {
            "type": "string",
            "enum": ["minor", "moderate", "major", "critical"]
          },
          "description": { "type": "string" }
        }
      }
    },
    "confidence": {
      "$ref": "#/definitions/ConfidenceScore"
    }
  },
  "definitions": {
    "ConfidenceScore": {
      "type": "object",
      "required": ["final", "factors"],
      "properties": {
        "final": { "type": "number", "minimum": 0, "maximum": 1 },
        "raw": { "type": "number" },
        "factors": {
          "type": "object",
          "properties": {
            "premise_validity": { "type": "number" },
            "chain_coherence": { "type": "number" },
            "fallacy_absence": { "type": "number" },
            "conclusion_strength": { "type": "number" }
          }
        }
      }
    }
  }
}
```

---

## MODULE 3: BedRock (First Principles)

### Purpose

First principles decomposition engine that breaks problems down to axiomatic foundations and rebuilds understanding from fundamental truths.

### Verbose Thinking Pattern (Level 1: Standard)

```
═══════════════════════════════════════════════════════════════════════════
[BedRock] FIRST PRINCIPLES DECOMPOSITION
═══════════════════════════════════════════════════════════════════════════

Query: "What makes RAPTOR hierarchical retrieval effective?"

[STEP 1] Current Understanding Statement
  ├─ Surface Statement: "RAPTOR improves retrieval by organizing chunks hierarchically"
  └─ Goal: Decompose to fundamental axioms, then rebuild

[STEP 2] Recursive Decomposition (Target Depth: 5)

┌─ LEVEL 1: Surface Observations
│  └─ "RAPTOR organizes document chunks into a tree structure"
│
├─ LEVEL 2: Underlying Mechanisms
│  ├─ Q: "What must be true for tree organization to help retrieval?"
│  └─ A: "Multiple levels of abstraction must preserve semantic relationships"
│
├─ LEVEL 3: Governing Principles
│  ├─ Q: "Why does abstraction hierarchy preserve semantics?"
│  └─ A: "Information compression via summarization retains core meaning"
│      ├─ Principle 3.1: "Summarization is semantic compression"
│      └─ Principle 3.2: "Hierarchies enable multi-scale reasoning"
│
├─ LEVEL 4: Fundamental Axioms
│  ├─ Q: "What is the axiom underlying semantic compression?"
│  ├─ A: "Language has redundancy that can be reduced without information loss"
│  │   ├─ Axiom 4.1: "Shannon's information theory: redundancy enables compression"
│  │   └─ Axiom 4.2: "Human language is ~50% redundant (Zipf's law)"
│  │
│  └─ Q: "What is the axiom underlying hierarchical reasoning?"
│      ├─ A: "Abstraction layers reduce cognitive load"
│      ├─ Axiom 4.3: "Miller's Law: 7±2 chunks in working memory"
│      └─ Axiom 4.4: "Hierarchies limit span of control (management principle)"
│
└─ LEVEL 5: Universal Truths
   ├─ Truth 5.1: "Systems with bounded complexity scale via layering"
   │   └─ Examples: TCP/IP stack, OS kernel layers, neural network depth
   ├─ Truth 5.2: "Compression trades precision for efficiency"
   │   └─ Examples: JPEG compression, dimensionality reduction, caching
   └─ Truth 5.3: "Search efficiency improves with problem space partitioning"
       └─ Examples: Binary search, database indexes, hash tables

[STEP 3] Axiom Verification
  ├─ [Axiom 4.1] Shannon information theory
  │   ├─ Source: Shannon (1948), "A Mathematical Theory of Communication"
  │   ├─ Verification: VERIFIED (foundational CS theory)
  │   └─ Confidence: 0.99
  │
  ├─ [Axiom 4.2] Zipf's law (language redundancy)
  │   ├─ Source: Zipf (1935), empirical linguistics
  │   ├─ Verification: VERIFIED (empirically validated across languages)
  │   └─ Confidence: 0.95
  │
  ├─ [Axiom 4.3] Miller's Law (working memory limits)
  │   ├─ Source: Miller (1956), "The Magical Number Seven"
  │   ├─ Verification: VERIFIED (replicated in cognitive psychology)
  │   └─ Confidence: 0.92
  │
  └─ [Axiom 4.4] Span of control principle
      ├─ Source: Management theory (Graicunas, 1933)
      ├─ Verification: LIKELY (organizational science consensus)
      └─ Confidence: 0.78

[STEP 4] Assumption Identification & Challenge
  ├─ Assumption 1: "Summarization always preserves core semantics"
  │   ├─ Challenge: What if summary loses critical details?
  │   ├─ Counter-example: Medical diagnosis (nuance matters)
  │   └─ Refinement: "Summarization preserves semantics FOR retrievable contexts"
  │
  ├─ Assumption 2: "3-level hierarchy is universally optimal"
  │   ├─ Challenge: What if document structure varies?
  │   ├─ Counter-example: Short docs don't need 3 levels
  │   └─ Refinement: "Hierarchy depth should match document complexity"
  │
  └─ Assumption 3: "Tree structure is the best hierarchical representation"
      ├─ Challenge: What about DAGs (directed acyclic graphs)?
      ├─ Alternative: Wikipedia link graphs, knowledge graphs
      └─ Refinement: "Trees trade flexibility for simplicity in RAPTOR's case"

[STEP 5] Rebuild from Verified Axioms
  ├─ Foundation (Level 5): "Systems with bounded complexity scale via layering"
  │   └─ Supports: Hierarchical organization is a scaling pattern
  │
  ├─ Layer (Level 4): "Language redundancy enables compression (Shannon)"
  │   └─ Supports: Summarization can reduce tokens without losing meaning
  │
  ├─ Layer (Level 3): "Summarization is semantic compression"
  │   └─ Supports: Multi-level summaries create abstraction hierarchy
  │
  ├─ Layer (Level 2): "Abstraction hierarchy preserves semantics at multiple scales"
  │   └─ Supports: Tree structure with summaries enables multi-scale retrieval
  │
  └─ Conclusion (Level 1): "RAPTOR works because it leverages language redundancy,
      information theory, and cognitive load limits to create a scalable retrieval system"

[STEP 6] Coherence Validation
  ├─ Logical gaps: 0 detected
  ├─ Circular dependencies: 0 detected
  ├─ Unjustified leaps: 0 detected
  └─ Rebuild completeness: 100% (all levels connected)

───────────────────────────────────────────────────────────────────────────
CONFIDENCE BREAKDOWN
───────────────────────────────────────────────────────────────────────────
Factor Scores:
├─ axiom_soundness:            0.91 × 0.35 = 0.319
│   └─ (3.64 avg axiom confidence / 4 axioms)
├─ decomposition_completeness: 1.00 × 0.25 = 0.250
│   └─ (5 levels reached / 5 target depth)
├─ assumption_identification:  0.88 × 0.20 = 0.176
│   └─ (3 assumptions surfaced / ~3.4 estimated)
└─ rebuild_coherence:          1.00 × 0.20 = 0.200
    └─ (1.0 - (0 gaps / 5 rebuild steps))

Raw Confidence: 0.945
Calibration Adjustment: 0.95x (high axiom verification)
Final Module Confidence: 0.90

───────────────────────────────────────────────────────────────────────────
KEY FINDINGS
───────────────────────────────────────────────────────────────────────────
✅ Decomposed to 5 levels (reached universal truths)
✅ 4 axioms verified from foundational sources
✅ 3 assumptions identified and refined
✅ Complete rebuild with 0 logical gaps
🔍 Core Insight: "RAPTOR's effectiveness stems from information theory + cognitive limits, not just 'hierarchies are good'"

Recommended Next Steps:
1. Validate refined assumptions with ProofGuard
2. Test counter-examples (medical diagnosis, short docs)
3. Explore DAG alternative for future enhancement

───────────────────────────────────────────────────────────────────────────
Execution Time: 47.3s | Tokens: 4,256 | Version: bedrock-v2.0.0
═══════════════════════════════════════════════════════════════════════════
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "BedRock Output Schema",
  "type": "object",
  "required": ["module", "decomposition", "axioms", "rebuild", "confidence"],
  "properties": {
    "module": {
      "type": "string",
      "const": "bedrock"
    },
    "initial_statement": { "type": "string" },
    "decomposition": {
      "type": "object",
      "required": ["levels"],
      "properties": {
        "levels": {
          "type": "array",
          "minItems": 1,
          "maxItems": 5,
          "items": {
            "type": "object",
            "required": ["level", "question", "answer"],
            "properties": {
              "level": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5
              },
              "name": {
                "type": "string",
                "enum": [
                  "surface_observations",
                  "underlying_mechanisms",
                  "governing_principles",
                  "fundamental_axioms",
                  "universal_truths"
                ]
              },
              "question": { "type": "string" },
              "answer": { "type": "string" },
              "principles": {
                "type": "array",
                "items": { "type": "string" }
              }
            }
          }
        }
      }
    },
    "axioms": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "id",
          "statement",
          "source",
          "verification_status",
          "confidence"
        ],
        "properties": {
          "id": { "type": "string" },
          "statement": { "type": "string" },
          "source": { "type": "string" },
          "verification_status": {
            "type": "string",
            "enum": ["verified", "likely", "unverified", "disputed"]
          },
          "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          }
        }
      }
    },
    "assumptions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["statement", "challenge", "refinement"],
        "properties": {
          "statement": { "type": "string" },
          "challenge": { "type": "string" },
          "counter_example": { "type": "string" },
          "refinement": { "type": "string" }
        }
      }
    },
    "rebuild": {
      "type": "object",
      "required": ["steps", "coherence_score"],
      "properties": {
        "steps": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "layer": { "type": "integer" },
              "statement": { "type": "string" },
              "supports": { "type": "string" }
            }
          }
        },
        "coherence_score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "gaps_detected": {
          "type": "integer",
          "minimum": 0
        }
      }
    },
    "confidence": {
      "$ref": "#/definitions/ConfidenceScore"
    }
  },
  "definitions": {
    "ConfidenceScore": {
      "type": "object",
      "required": ["final", "factors"],
      "properties": {
        "final": { "type": "number", "minimum": 0, "maximum": 1 },
        "raw": { "type": "number" },
        "factors": {
          "type": "object",
          "properties": {
            "axiom_soundness": { "type": "number" },
            "decomposition_completeness": { "type": "number" },
            "assumption_identification": { "type": "number" },
            "rebuild_coherence": { "type": "number" }
          }
        }
      }
    }
  }
}
```

---

## MODULE 4: ProofGuard (Verification)

### Purpose

Multi-source verification engine with triangulation protocol, contradiction detection, and source quality assessment.

### Verbose Thinking Pattern (Level 1: Standard)

```
═══════════════════════════════════════════════════════════════════════════
[ProofGuard] MULTI-SOURCE TRIANGULATION & VERIFICATION
═══════════════════════════════════════════════════════════════════════════

Input Claims: 7 factual assertions from GigaThink + LaserLogic + BedRock

[STEP 1] Claim Extraction
  ├─ Claim 1: "RAPTOR improves performance by +20% on QuALITY benchmark"
  ├─ Claim 2: "RAPTOR improves performance by +18% on NarrativeQA"
  ├─ Claim 3: "Hierarchical summarization preserves semantic meaning"
  ├─ Claim 4: "3-level hierarchy is optimal for most documents"
  ├─ Claim 5: "Shannon's information theory: redundancy enables compression"
  ├─ Claim 6: "Miller's Law: 7±2 chunks in working memory"
  └─ Claim 7: "Implementation complexity is moderate"

[STEP 2] Source Search (Target: 3+ per claim)

[CLAIM 1] "RAPTOR +20% on QuALITY"
  ├─ [SEARCHING] Academic papers on RAPTOR...
  ├─ Source A (Tier 1): arXiv:2401.18059 - "RAPTOR: Recursive Abstractive..."
  │   ├─ Type: Peer-reviewed preprint (arXiv)
  │   ├─ Verdict: SUPPORTS (Table 2, QuALITY: 55.7% → 66.9% = +20.1%)
  │   ├─ URL: https://arxiv.org/abs/2401.18059
  │   └─ Accessed: YES ✓
  │
  ├─ Source B (Tier 2): LlamaIndex blog post (2024-02-15)
  │   ├─ Type: Framework documentation
  │   ├─ Verdict: SUPPORTS (cites arXiv paper, confirms +20%)
  │   ├─ URL: https://www.llamaindex.ai/blog/raptor-hierarchical-retrieval
  │   └─ Accessed: YES ✓
  │
  └─ Source C (Tier 3): Papers With Code benchmark
      ├─ Type: Community benchmark aggregator
      ├─ Verdict: SUPPORTS (QuALITY leaderboard shows RAPTOR result)
      ├─ URL: https://paperswithcode.com/sota/question-answering-on-quality
      └─ Accessed: YES ✓

[CLAIM 2] "RAPTOR +18% on NarrativeQA"
  ├─ [SEARCHING] Same sources as Claim 1...
  ├─ Source A (Tier 1): arXiv:2401.18059
  │   ├─ Verdict: SUPPORTS (Table 3, NarrativeQA: 48.2% → 57.1% = +18.5%)
  │   └─ Accessed: YES ✓
  │
  ├─ Source B (Tier 2): HuggingFace model card
  │   ├─ Verdict: SUPPORTS (model documentation confirms benchmark)
  │   └─ Accessed: YES ✓
  │
  └─ Source C (Tier 3): Independent GitHub implementation
      ├─ Verdict: LIKELY (reproduced similar results in README)
      ├─ URL: https://github.com/user/raptor-reproduction
      └─ Accessed: YES ✓

[CLAIM 3] "Hierarchical summarization preserves semantics"
  ├─ [SEARCHING] Summarization literature...
  ├─ Source A (Tier 1): "Abstractive Summarization" (ACL 2022)
  │   ├─ Verdict: SUPPORTS (with caveats on context loss)
  │   └─ Confidence: 0.85
  │
  ├─ Source B (Tier 2): NVIDIA NeMo documentation
  │   ├─ Verdict: SUPPORTS (for general use cases)
  │   └─ Confidence: 0.78
  │
  └─ Source C (Tier 3): Blog post by OpenAI researcher
      ├─ Verdict: QUALIFIED SUPPORT (works for most domains, not all)
      └─ Confidence: 0.70

[CLAIM 4] "3-level hierarchy is optimal"
  ├─ [SEARCHING] RAPTOR paper + ablation studies...
  ├─ Source A (Tier 1): arXiv:2401.18059 (Appendix A.3)
  │   ├─ Verdict: CONTRADICTED
  │   ├─ Finding: "Optimal depth varies by document: 2-4 levels"
  │   └─ Correction: "3 levels is AVERAGE optimal, not universal"
  │
  ├─ Source B (Tier 2): Community experiments
  │   ├─ Verdict: CONTRADICTED
  │   └─ Finding: "Short docs (<5k tokens) best with 2 levels"
  │
  └─ Source C: None found
      └─ Status: UNVERIFIED (only 2 sources, and both contradict)

[CLAIM 5] "Shannon's information theory"
  ├─ [SEARCHING] Foundational CS sources...
  ├─ Source A (Tier 1): Shannon (1948), Bell System Technical Journal
  │   ├─ Verdict: VERIFIED (original paper, foundational theorem)
  │   └─ Confidence: 0.99
  │
  ├─ Source B (Tier 2): "Information Theory" textbook (Cover & Thomas, 2006)
  │   ├─ Verdict: VERIFIED (standard CS curriculum reference)
  │   └─ Confidence: 0.99
  │
  └─ Source C (Tier 3): Wikipedia (Information Theory)
      ├─ Verdict: VERIFIED (comprehensive, well-cited)
      └─ Confidence: 0.95

[CLAIM 6] "Miller's Law: 7±2 chunks"
  ├─ [SEARCHING] Cognitive psychology literature...
  ├─ Source A (Tier 1): Miller (1956), "The Magical Number Seven"
  │   ├─ Verdict: VERIFIED (original paper, replicated extensively)
  │   └─ Confidence: 0.92
  │
  ├─ Source B (Tier 2): "Cognitive Psychology" textbook (Goldstein, 2019)
  │   ├─ Verdict: VERIFIED (with modern refinements: 4±1 more accurate)
  │   └─ Confidence: 0.88
  │
  └─ Source C (Tier 3): UX design guidelines (Nielsen Norman Group)
      ├─ Verdict: VERIFIED (applied in interface design)
      └─ Confidence: 0.82

[CLAIM 7] "Implementation complexity is moderate"
  ├─ [SEARCHING] Implementation experiences...
  ├─ Source A: GitHub issue discussions
  │   ├─ Verdict: CONTESTED (some say "easy", others "hard")
  │   └─ Confidence: 0.40
  │
  ├─ Source B: LlamaIndex integration example
  │   ├─ Verdict: SUPPORTS (50 lines of code for basic integration)
  │   └─ Confidence: 0.65
  │
  └─ Source C: None found
      └─ Status: UNVERIFIED (subjective, insufficient evidence)

[STEP 3] Triangulation Summary Table

| Claim | Source A (Tier 1) | Source B (Tier 2) | Source C (Tier 3) | Consensus |
|-------|-------------------|-------------------|-------------------|-----------|
| RAPTOR +20% QuALITY | arXiv (SUPPORTS) | LlamaIndex (SUPPORTS) | PWC (SUPPORTS) | VERIFIED ✓ |
| RAPTOR +18% NarrativeQA | arXiv (SUPPORTS) | HF (SUPPORTS) | GitHub (LIKELY) | VERIFIED ✓ |
| Summarization preserves semantics | ACL paper (SUPPORTS) | NVIDIA (SUPPORTS) | OpenAI blog (QUALIFIED) | LIKELY ~ |
| 3-level optimal | arXiv (CONTRADICTS) | Community (CONTRADICTS) | - | CONTRADICTED ✗ |
| Shannon's theorem | Shannon 1948 (VERIFIED) | Cover & Thomas (VERIFIED) | Wikipedia (VERIFIED) | VERIFIED ✓ |
| Miller's Law | Miller 1956 (VERIFIED) | Goldstein (VERIFIED) | NN Group (VERIFIED) | VERIFIED ✓ |
| Moderate complexity | GitHub (CONTESTED) | LlamaIndex (SUPPORTS) | - | UNVERIFIED ? |

[STEP 4] Contradiction Analysis
  ├─ Contradiction 1: "3-level hierarchy is optimal"
  │   ├─ Conflicting Evidence: arXiv says 2-4 levels, community says 2 for short docs
  │   ├─ Resolution: REJECT original claim, REPLACE with "Optimal depth varies by doc size"
  │   └─ Impact: Moderate (affects implementation strategy)
  │
  └─ Contradiction 2: "Implementation complexity"
      ├─ Conflicting Evidence: Some say easy, others hard
      ├─ Resolution: QUALIFY claim as "Complexity depends on use case"
      └─ Impact: Low (subjective assessment anyway)

[STEP 5] Source Diversity Check
  ├─ Unique domains: 12 different sources
  ├─ Source overlap: 2 claims reused sources (acceptable)
  └─ Domain diversity: 0.86 (12 unique / 14 total sources)

[STEP 6] Consensus Level Calculation
  ├─ Claims verified: 4/7 (57%)
  ├─ Claims likely: 1/7 (14%)
  ├─ Claims contradicted: 1/7 (14%)
  ├─ Claims unverified: 1/7 (14%)
  └─ Overall verification rate: 71% (verified + likely)

───────────────────────────────────────────────────────────────────────────
CONFIDENCE BREAKDOWN
───────────────────────────────────────────────────────────────────────────
Factor Scores:
├─ source_count:              1.00 × 0.20 = 0.200
│   └─ (14 sources / 3 minimum = 1.0 capped)
├─ source_diversity:          0.86 × 0.25 = 0.215
│   └─ (12 unique domains / 14 total sources)
├─ tier_quality:              0.88 × 0.20 = 0.176
│   └─ (Weighted: 6 Tier-1 × 1.0 + 5 Tier-2 × 0.8 + 3 Tier-3 × 0.6 = 0.88 avg)
├─ contradiction_absence:     0.71 × 0.25 = 0.178
│   └─ (1.0 - (2 contradictions / 7 claims))
└─ verification_completeness: 0.71 × 0.10 = 0.071
    └─ (5 verified/likely / 7 total claims)

Raw Confidence: 0.840
Calibration Adjustment: 1.0x (standard verification)
Final Module Confidence: 0.84

───────────────────────────────────────────────────────────────────────────
KEY FINDINGS
───────────────────────────────────────────────────────────────────────────
✅ 4 claims VERIFIED with 3+ sources (57%)
✅ 14 sources accessed across 12 unique domains
⚠️ 1 claim CONTRADICTED (3-level hierarchy claim rejected)
⚠️ 1 claim UNVERIFIED (implementation complexity - subjective)
🔍 High-confidence claims: Shannon's theorem (0.99), Miller's Law (0.92)

Recommended Actions:
1. Update GigaThink output to correct "3-level optimal" to "2-4 levels based on doc size"
2. Qualify "moderate complexity" claim or remove (insufficient evidence)
3. Benchmark claims are solid - can proceed with confidence

───────────────────────────────────────────────────────────────────────────
Execution Time: 124.6s | Tokens: 6,892 | Version: proofguard-v2.0.0
═══════════════════════════════════════════════════════════════════════════
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ProofGuard Output Schema",
  "type": "object",
  "required": [
    "module",
    "claims",
    "triangulation_table",
    "contradictions",
    "confidence"
  ],
  "properties": {
    "module": {
      "type": "string",
      "const": "proofguard"
    },
    "claims": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "text", "sources", "consensus"],
        "properties": {
          "id": { "type": "string" },
          "text": { "type": "string" },
          "origin": { "type": "string" },
          "sources": {
            "type": "array",
            "minItems": 0,
            "items": {
              "type": "object",
              "required": ["url", "tier", "verdict", "accessed"],
              "properties": {
                "url": { "type": "string", "format": "uri" },
                "tier": {
                  "type": "integer",
                  "minimum": 1,
                  "maximum": 3
                },
                "tier_name": {
                  "type": "string",
                  "enum": ["authoritative", "secondary", "independent"]
                },
                "verdict": {
                  "type": "string",
                  "enum": [
                    "supports",
                    "contradicts",
                    "qualified_support",
                    "neutral"
                  ]
                },
                "accessed": { "type": "boolean" },
                "confidence": {
                  "type": "number",
                  "minimum": 0,
                  "maximum": 1
                }
              }
            }
          },
          "consensus": {
            "type": "string",
            "enum": ["verified", "likely", "contradicted", "unverified"]
          },
          "consensus_symbol": {
            "type": "string",
            "enum": ["✓", "~", "✗", "?"]
          }
        }
      }
    },
    "triangulation_table": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "claim": { "type": "string" },
          "source_a": { "type": "string" },
          "source_b": { "type": "string" },
          "source_c": { "type": "string" },
          "consensus": { "type": "string" }
        }
      }
    },
    "contradictions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "claim_id",
          "conflicting_evidence",
          "resolution",
          "impact"
        ],
        "properties": {
          "claim_id": { "type": "string" },
          "conflicting_evidence": {
            "type": "array",
            "items": { "type": "string" }
          },
          "resolution": { "type": "string" },
          "impact": {
            "type": "string",
            "enum": ["low", "moderate", "high", "critical"]
          }
        }
      }
    },
    "source_diversity": {
      "type": "object",
      "properties": {
        "unique_domains": { "type": "integer" },
        "total_sources": { "type": "integer" },
        "diversity_score": { "type": "number" }
      }
    },
    "confidence": {
      "$ref": "#/definitions/ConfidenceScore"
    }
  },
  "definitions": {
    "ConfidenceScore": {
      "type": "object",
      "required": ["final", "factors"],
      "properties": {
        "final": { "type": "number", "minimum": 0, "maximum": 1 },
        "raw": { "type": "number" },
        "factors": {
          "type": "object",
          "properties": {
            "source_count": { "type": "number" },
            "source_diversity": { "type": "number" },
            "tier_quality": { "type": "number" },
            "contradiction_absence": { "type": "number" },
            "verification_completeness": { "type": "number" }
          }
        }
      }
    }
  }
}
```

---

## MODULE 5: BrutalHonesty (Adversarial Critique)

### Purpose

Adversarial self-critique engine that actively seeks flaws, edge cases, and failure modes in reasoning before conclusions are finalized.

### Attack Vectors (8 Systematic Tests)

| ID      | Attack Vector                | Description                                               | Example                                                     |
| ------- | ---------------------------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| **AV1** | Assumption Challenges        | Test every unstated assumption with counter-examples      | "Assumes users have stable network" → Test offline scenario |
| **AV2** | Edge Case Stress Tests       | Generate boundary conditions that break logic             | Empty input, max input, malformed input, concurrent access  |
| **AV3** | Logical Gap Probing          | Find missing steps in reasoning chains                    | "A → C" with no explanation of B                            |
| **AV4** | Evidence Quality Questioning | Attack source credibility and recency                     | "2019 benchmark may be outdated"                            |
| **AV5** | Bias Detection               | Scan for cognitive biases (anchoring, confirmation, etc.) | Over-reliance on first estimate (anchoring bias)            |
| **AV6** | Overconfidence Correction    | Challenge confidence scores that seem too high            | "95% confidence but only 2 sources?"                        |
| **AV7** | Hidden Dependency Surfacing  | Find implicit requirements not documented                 | "Requires Rust nightly toolchain" (not mentioned)           |
| **AV8** | Failure Mode Analysis        | Predict how/when the solution will break                  | "RAPTOR fails when summaries lose critical details"         |

### Verbose Thinking Pattern (Level 1: Standard)

```
═══════════════════════════════════════════════════════════════════════════
[BrutalHonesty] ADVERSARIAL CRITIQUE & FAILURE MODE ANALYSIS
═══════════════════════════════════════════════════════════════════════════

Input: Synthesis of GigaThink + LaserLogic + BedRock + ProofGuard outputs
Recommendation under scrutiny: "Adopt RAPTOR for reasonkit-core hierarchical retrieval"

[STEP 1] Adversarial Posture Activation
  ├─ Mindset: Hostile skeptic seeking to REJECT recommendation
  ├─ Attack intensity: 0.8/1.0 (ruthless)
  └─ Minimum critiques required: 5

[STEP 2] Attack Vector 1: Assumption Challenges
  ├─ [AV1.1] Assumption: "RAPTOR benchmark results generalize to our use case"
  │   ├─ Counter-example: QuALITY benchmark is multi-hop QA, not RAG retrieval
  │   ├─ Severity: MAJOR ⚠️
  │   └─ Critique: "Benchmark tasks differ from reasonkit-core's doc search use case.
  │       RAPTOR evaluated on question-answering, not document retrieval for LLM context."
  │
  ├─ [AV1.2] Assumption: "Users have documents >10k tokens to justify hierarchy"
  │   ├─ Counter-example: What if most queries are on short docs (<5k tokens)?
  │   ├─ Severity: MODERATE ⚠️
  │   └─ Critique: "No data on reasonkit-core's actual document length distribution.
  │       RAPTOR overhead may hurt performance on short docs (majority of corpus?)."
  │
  └─ [AV1.3] Assumption: "Summarization preserves all critical information"
      ├─ Counter-example: Code snippets, technical specs, exact numbers get lossy
      ├─ Severity: BLOCKING ⛔
      └─ Critique: "RAPTOR uses LLM summarization which is lossy. For technical docs
          with precise syntax (Rust code, API specs), summary may omit critical details.
          No mitigation strategy proposed."

[STEP 3] Attack Vector 2: Edge Case Stress Tests
  ├─ [AV2.1] Edge Case: Empty document
  │   ├─ Test: What if doc has 0 tokens?
  │   ├─ Result: Hierarchy build would fail (division by zero in chunking?)
  │   └─ Severity: MINOR ⚡ (can be handled with validation)
  │
  ├─ [AV2.2] Edge Case: Document with no hierarchical structure
  │   ├─ Test: Flat list (e.g., dictionary entries)
  │   ├─ Result: Forced hierarchy may create artificial groupings
  │   └─ Severity: MODERATE ⚠️ (degrades to flat chunking anyway)
  │
  ├─ [AV2.3] Edge Case: Highly redundant document
  │   ├─ Test: Legal document with repeated boilerplate
  │   ├─ Result: Summaries may over-compress, losing nuance
  │   └─ Severity: MAJOR ⚠️ (for legal/compliance use cases)
  │
  └─ [AV2.4] Edge Case: Multilingual document
      ├─ Test: Mixed English + code + math equations
      ├─ Result: LLM summarization may fail on code/math
      └─ Severity: BLOCKING ⛔ (reasonkit-core ingests code-heavy docs!)

[STEP 4] Attack Vector 3: Logical Gap Probing
  ├─ [AV3.1] Gap: How is optimal hierarchy depth determined PER DOCUMENT?
  │   ├─ Critique: ProofGuard found "2-4 levels" but no algorithm to choose depth
  │   └─ Severity: MAJOR ⚠️ (implementation blocker)
  │
  └─ [AV3.2] Gap: No discussion of rebuild/reindex strategy
      ├─ Critique: If docs update, does entire tree rebuild? What's the cost?
      └─ Severity: MODERATE ⚠️ (maintenance concern)

[STEP 5] Attack Vector 4: Evidence Quality Questioning
  ├─ [AV4.1] Challenge: arXiv preprint (not peer-reviewed journal)
  │   ├─ Critique: Main evidence is arXiv, not formally peer-reviewed
  │   └─ Severity: ADVISORY ℹ️ (arXiv ML papers are de facto standard, acceptable)
  │
  └─ [AV4.2] Challenge: No production deployment case studies
      ├─ Critique: All evidence is benchmarks, zero real-world usage reports
      └─ Severity: MODERATE ⚠️ (unknown production reliability)

[STEP 6] Attack Vector 5: Bias Detection
  ├─ [AV5.1] Confirmation Bias
  │   ├─ Detected: GigaThink generated mostly PRO-RAPTOR perspectives
  │   ├─ Counter: Only 1 adversarial perspective (P3) out of 11 total
  │   └─ Severity: MODERATE ⚠️ (need more critical perspectives)
  │
  ├─ [AV5.2] Anchoring Bias
  │   ├─ Detected: +20% benchmark improvement anchored entire analysis
  │   ├─ Counter: Didn't explore -10% scenarios (when might it hurt?)
  │   └─ Severity: MINOR ⚡ (bias toward positive framing)
  │
  └─ [AV5.3] Availability Bias
      ├─ Detected: Heavy reliance on RAPTOR paper (most available source)
      ├─ Counter: Didn't compare to LlamaIndex auto-merging, Pinecone namespaces
      └─ Severity: MAJOR ⚠️ (incomplete competitive analysis)

[STEP 7] Attack Vector 6: Overconfidence Correction
  ├─ [AV6.1] Challenge: ProofGuard confidence at 84%
  │   ├─ Critique: "1 contradicted claim + 1 unverified + no production data = 84%?"
  │   ├─ Recommendation: Should be 70-75% given gaps
  │   └─ Severity: MODERATE ⚠️ (overconfident)
  │
  └─ [AV6.2] Challenge: BedRock confidence at 90%
      ├─ Critique: "90% despite axioms being theoretical, not empirical for RAPTOR?"
      ├─ Recommendation: Should be 80-85% (axioms valid, but application untested)
      └─ Severity: MODERATE ⚠️ (overconfident)

[STEP 8] Attack Vector 7: Hidden Dependency Surfacing
  ├─ [AV7.1] Dependency: Requires LLM for summarization
  │   ├─ Implication: Adds API cost + latency to indexing
  │   ├─ Not mentioned: Token cost estimate for 840 docs
  │   └─ Severity: MODERATE ⚠️ (cost impact not analyzed)
  │
  ├─ [AV7.2] Dependency: Assumes Qdrant supports nested structures
  │   ├─ Implication: Need to verify Qdrant schema supports tree metadata
  │   ├─ Not verified: Technical compatibility check
  │   └─ Severity: BLOCKING ⛔ (integration risk)
  │
  └─ [AV7.3] Dependency: Requires tree traversal at query time
      ├─ Implication: Latency increase vs flat search
      ├─ Not benchmarked: Query latency impact
      └─ Severity: MAJOR ⚠️ (performance concern)

[STEP 9] Attack Vector 8: Failure Mode Analysis
  ├─ [FM1] Failure: Summarization loses critical code syntax
  │   ├─ Probability: HIGH (for code-heavy docs)
  │   ├─ Impact: CRITICAL (wrong retrieval results)
  │   └─ Mitigation: NONE PROPOSED ⛔
  │
  ├─ [FM2] Failure: Hierarchy depth misconfiguration
  │   ├─ Probability: MEDIUM (no auto-tuning algorithm)
  │   ├─ Impact: MODERATE (suboptimal retrieval)
  │   └─ Mitigation: PARTIAL (manual tuning possible)
  │
  ├─ [FM3] Failure: Index rebuild cost explosion
  │   ├─ Probability: MEDIUM (frequent doc updates)
  │   ├─ Impact: HIGH (operational overhead)
  │   └─ Mitigation: NONE PROPOSED ⚠️
  │
  └─ [FM4] Failure: Qdrant integration incompatibility
      ├─ Probability: LOW (but not verified)
      ├─ Impact: BLOCKING (can't implement)
      └─ Mitigation: VERIFY BEFORE COMMIT ⛔

───────────────────────────────────────────────────────────────────────────
CRITIQUE SUMMARY (2 BLOCKING, 6 MAJOR, 5 MODERATE, 2 MINOR, 1 ADVISORY)
───────────────────────────────────────────────────────────────────────────

⛔ BLOCKING ISSUES (MUST FIX):
  1. Summarization loss for code/technical docs (AV1.3, FM1)
     └─ Remediation: Implement hybrid strategy: RAPTOR for prose, flat for code

  2. Qdrant integration not verified (AV7.2, FM4)
     └─ Remediation: Prototype tree metadata schema in Qdrant BEFORE committing

⚠️ MAJOR ISSUES (SHOULD FIX):
  3. Benchmark task mismatch (QA vs retrieval) (AV1.1)
  4. Edge case: Multilingual/code-heavy docs (AV2.4)
  5. Missing depth selection algorithm (AV3.1)
  6. Incomplete competitive analysis (AV5.3)
  7. Query latency impact unknown (AV7.3)
  8. No production deployment evidence (AV4.2)

⚡ MODERATE ISSUES (CONSIDER):
  9. Document length distribution unknown (AV1.2)
  10. Rebuild/reindex strategy undefined (AV3.2)
  11. Overconfidence in ProofGuard + BedRock scores (AV6.1, AV6.2)
  12. Confirmation bias in perspective generation (AV5.1)
  13. Indexing cost not estimated (AV7.1)

───────────────────────────────────────────────────────────────────────────
CONFIDENCE BREAKDOWN
───────────────────────────────────────────────────────────────────────────
Factor Scores:
├─ critique_depth:          0.93 × 0.25 = 0.233
│   └─ (14 critiques generated / 5 minimum)
├─ fatal_flaw_detection:    1.00 × 0.30 = 0.300
│   └─ (2 blocking issues found = HIGH value)
├─ edge_case_coverage:      0.80 × 0.20 = 0.160
│   └─ (4 edge cases tested / ~5 estimated)
├─ bias_identification:     0.75 × 0.15 = 0.113
│   └─ (3 biases detected / 4 scan categories)
└─ remediation_quality:     0.43 × 0.10 = 0.043
    └─ (6 issues addressed / 14 issues found)

Raw Confidence: 0.849
Calibration Adjustment: 1.0x (adversarial module)
Final Module Confidence: 0.85

NOTE: High confidence in critique quality, LOW confidence in recommendation
      (2 blocking issues = recommendation should be REJECTED or QUALIFIED)

───────────────────────────────────────────────────────────────────────────
VERDICT: REJECT RECOMMENDATION IN CURRENT FORM
───────────────────────────────────────────────────────────────────────────

Reasoning:
  - 2 BLOCKING issues prevent safe implementation
  - Code/technical doc use case (reasonkit-core's primary) is HIGH RISK
  - Qdrant integration is UNVERIFIED (prototype required)
  - 6 MAJOR issues create significant implementation uncertainty

Recommended Path Forward:
  1. IMMEDIATE: Verify Qdrant tree metadata support (prototype in 1-2 days)
  2. IMMEDIATE: Implement hybrid strategy (RAPTOR for prose, flat for code)
  3. BEFORE COMMIT: Measure query latency impact on reasonkit-core corpus
  4. BEFORE COMMIT: Analyze document length distribution (is >10k token common?)
  5. CONSIDER: Start with RAPTOR as OPTIONAL feature, default to flat chunking

IF blocking issues resolved → Re-evaluate with QUALIFIED APPROVAL
IF blocking issues persist → DEFER to reasonkit-pro (not core)

───────────────────────────────────────────────────────────────────────────
Execution Time: 56.8s | Tokens: 5,124 | Version: brutalhonesty-v2.0.0
═══════════════════════════════════════════════════════════════════════════
```

### JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "BrutalHonesty Output Schema",
  "type": "object",
  "required": ["module", "critiques", "verdict", "confidence"],
  "properties": {
    "module": {
      "type": "string",
      "const": "brutalhonesty"
    },
    "input_summary": { "type": "string" },
    "attack_intensity": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "critiques": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "attack_vector", "description", "severity"],
        "properties": {
          "id": { "type": "string" },
          "attack_vector": {
            "type": "string",
            "enum": [
              "assumption_challenge",
              "edge_case_stress_test",
              "logical_gap_probing",
              "evidence_quality_questioning",
              "bias_detection",
              "overconfidence_correction",
              "hidden_dependency_surfacing",
              "failure_mode_analysis"
            ]
          },
          "description": { "type": "string" },
          "counter_example": { "type": "string" },
          "severity": {
            "type": "string",
            "enum": ["blocking", "major", "moderate", "minor", "advisory"]
          },
          "severity_symbol": {
            "type": "string",
            "enum": ["⛔", "⚠️", "⚡", "ℹ️"]
          },
          "remediation": { "type": "string" },
          "addressed": { "type": "boolean" }
        }
      }
    },
    "edge_cases": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "scenario": { "type": "string" },
          "test": { "type": "string" },
          "result": { "type": "string" },
          "severity": { "type": "string" }
        }
      }
    },
    "biases_detected": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "bias_type": {
            "type": "string",
            "enum": [
              "confirmation",
              "anchoring",
              "availability",
              "hindsight",
              "sunk_cost",
              "groupthink"
            ]
          },
          "description": { "type": "string" },
          "severity": { "type": "string" }
        }
      }
    },
    "failure_modes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["failure", "probability", "impact", "mitigation"],
        "properties": {
          "failure": { "type": "string" },
          "probability": {
            "type": "string",
            "enum": ["low", "medium", "high"]
          },
          "impact": {
            "type": "string",
            "enum": ["low", "moderate", "high", "critical", "blocking"]
          },
          "mitigation": { "type": "string" },
          "mitigation_status": {
            "type": "string",
            "enum": ["none", "partial", "full", "verify_required"]
          }
        }
      }
    },
    "verdict": {
      "type": "object",
      "required": ["decision", "reasoning"],
      "properties": {
        "decision": {
          "type": "string",
          "enum": ["approve", "qualified_approval", "reject", "defer"]
        },
        "reasoning": { "type": "string" },
        "blocking_issues": { "type": "integer" },
        "major_issues": { "type": "integer" },
        "recommended_path": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "confidence": {
      "$ref": "#/definitions/ConfidenceScore"
    }
  },
  "definitions": {
    "ConfidenceScore": {
      "type": "object",
      "required": ["final", "factors"],
      "properties": {
        "final": { "type": "number", "minimum": 0, "maximum": 1 },
        "raw": { "type": "number" },
        "factors": {
          "type": "object",
          "properties": {
            "critique_depth": { "type": "number" },
            "fatal_flaw_detection": { "type": "number" },
            "edge_case_coverage": { "type": "number" },
            "bias_identification": { "type": "number" },
            "remediation_quality": { "type": "number" }
          }
        },
        "note": { "type": "string" }
      }
    }
  }
}
```

---

## CONFIDENCE CALCULATION FORMULAS

### Universal Confidence Formula

```
OVERALL_CONFIDENCE = (
    Σ(module_confidence × module_weight) / Σ(module_weights)
) × contradiction_penalty × profile_multiplier
```

### Module Weights (by Profile)

| Module        | Quick | Balanced | Deep | Paranoid | Scientific |
| ------------- | ----- | -------- | ---- | -------- | ---------- |
| GigaThink     | 0.50  | 0.15     | 0.15 | 0.12     | 0.10       |
| LaserLogic    | 0.50  | 0.25     | 0.20 | 0.18     | 0.20       |
| BedRock       | -     | 0.20     | 0.20 | 0.15     | 0.25       |
| ProofGuard    | -     | 0.30     | 0.25 | 0.25     | 0.25       |
| BrutalHonesty | -     | -        | 0.20 | 0.20     | -          |
| HighReflect   | -     | -        | -    | 0.10     | -          |
| RiskRadar     | -     | -        | -    | -        | 0.20       |

### Contradiction Penalty

```python
def calculate_contradiction_penalty(contradictions, total_cross_checks):
    """
    Penalize confidence based on inter-module contradictions.

    Args:
        contradictions: Number of contradicting claims across modules
        total_cross_checks: Total number of cross-module comparisons made

    Returns:
        Penalty multiplier (0.5 - 1.0)
    """
    if total_cross_checks == 0:
        return 1.0

    contradiction_rate = contradictions / total_cross_checks

    if contradiction_rate == 0:
        return 1.00  # No contradictions
    elif contradiction_rate < 0.05:
        return 0.95  # Minor conflicts
    elif contradiction_rate < 0.15:
        return 0.85  # Moderate disagreements
    elif contradiction_rate < 0.30:
        return 0.70  # Major conflicts
    else:
        return 0.50  # Irreconcilable contradictions
```

### Profile Multipliers

```python
PROFILE_MULTIPLIERS = {
    "quick": 1.00,        # No adjustment, accept lower confidence
    "balanced": 1.05,     # Slight boost for multi-module validation
    "deep": 1.10,         # Boost for thorough analysis
    "scientific": 1.10,   # Boost for scientific rigor
    "paranoid": 1.15,     # Maximum boost for rigorous verification
    "decide": 1.05,       # Boost for decision-focused analysis
}
```

### Calibration Bands

| Range   | Label                   | Action                                         |
| ------- | ----------------------- | ---------------------------------------------- |
| 95-100% | Very High Confidence    | Proceed with implementation                    |
| 85-94%  | High Confidence         | Proceed with monitoring                        |
| 70-84%  | Moderate Confidence     | Proceed with caution, plan validation          |
| 50-69%  | Low Confidence          | Gather more data before deciding               |
| <50%    | Insufficient Confidence | Do not proceed, re-analyze with higher profile |

---

## OUTPUT FORMATS

### Format 1: Markdown (Human-Readable)

See verbose thinking patterns above for full markdown examples.

**File naming:** `thinking_trace_{module}_{timestamp}.md`

**Example:**

```
thinking_trace_gigathink_20251222_143022.md
thinking_trace_laserlogic_20251222_143156.md
```

---

### Format 2: JSON (Machine-Readable)

**File naming:** `thinking_trace_{module}_{timestamp}.json`

**Example: GigaThink JSON Output**

```json
{
  "module": "gigathink",
  "version": "2.0.0",
  "timestamp": "2025-12-22T14:30:22Z",
  "input_query": "Should we adopt RAPTOR for hierarchical retrieval?",
  "execution_time_ms": 34200,
  "token_count": 3847,
  "perspectives": [
    {
      "id": "P1",
      "type": "analytical",
      "description": "Performance optimization lens",
      "insight": "Need to compare RAPTOR vs flat chunking metrics",
      "novelty_score": 0.65,
      "actionability_score": 0.92
    },
    {
      "id": "P2",
      "type": "creative",
      "description": "Cognitive architecture alignment",
      "insight": "Tree structure mirrors human memory retrieval",
      "novelty_score": 0.88,
      "actionability_score": 0.72
    }
  ],
  "analogies": [
    {
      "domain": "Library organization",
      "analogy": "Dewey Decimal System = hierarchical classification",
      "relevance": 0.85
    }
  ],
  "emergent_patterns": [
    {
      "cluster_id": "C1",
      "theme": "Performance gains are real but context-dependent",
      "perspectives": ["P1", "P4", "P9"]
    }
  ],
  "insights": [
    {
      "text": "RAPTOR works when summarization ≈ semantic compression",
      "score": 0.92,
      "novelty": 0.88,
      "actionability": 0.96
    }
  ],
  "confidence": {
    "final": 0.83,
    "raw": 0.869,
    "calibration_adjustment": 0.95,
    "factors": {
      "perspective_diversity": 0.88,
      "insight_novelty": 0.82,
      "domain_coverage": 0.75,
      "contradiction_detection": 0.95,
      "actionability": 0.91
    }
  },
  "next_steps": ["Feed to LaserLogic for logical validation of insights"]
}
```

---

## INTEGRATION GUIDE

### CLI Usage

```bash
# Single module with verbose output
rk-core think "query" --module gigathink --verbose-level 1

# Profile-based (multiple modules)
rk-core think "query" --profile balanced --verbose-level 2

# Output to JSON
rk-core think "query" --profile paranoid --output-format json --output-dir ./traces

# Quiet mode (only summary)
rk-core think "query" --profile quick --quiet
```

### Rust API

```rust
use reasonkit_core::thinktools::{ThinkToolOrchestrator, VerboseLevel};

let orchestrator = ThinkToolOrchestrator::new();
let result = orchestrator
    .think("Should we adopt RAPTOR?")
    .profile(ReasoningProfile::Balanced)
    .verbose_level(VerboseLevel::Standard)
    .output_format(OutputFormat::Both)  // Markdown + JSON
    .execute()
    .await?;

// Access confidence breakdown
println!("Overall Confidence: {}", result.confidence.final_score);
for (module, score) in result.confidence.module_scores {
    println!("  {}: {}", module, score);
}

// Access verbose trace
println!("Markdown trace: {}", result.markdown_trace);
println!("JSON trace saved to: {}", result.json_path);
```

### Python Bindings

```python
from reasonkit import ThinkToolOrchestrator, ReasoningProfile, VerboseLevel

orchestrator = ThinkToolOrchestrator()
result = orchestrator.think(
    query="Should we adopt RAPTOR?",
    profile=ReasoningProfile.BALANCED,
    verbose_level=VerboseLevel.STANDARD,
    output_format="both"
)

print(f"Confidence: {result.confidence.final:.2%}")
print(f"Markdown trace:\n{result.markdown_trace}")
```

---

## CHANGELOG

| Version   | Date       | Changes                                                |
| --------- | ---------- | ------------------------------------------------------ |
| **1.0.0** | 2025-12-22 | Initial verbose thinking patterns document             |
|           |            | - Defined 5 design principles (Claude consultation)    |
|           |            | - Created detailed patterns for all 5 modules          |
|           |            | - Established JSON schemas for machine-readable output |
|           |            | - Documented confidence calculation formulas           |
|           |            | - Added integration guide (CLI, Rust, Python)          |

---

## RELATED DOCUMENTS

- `thinktools.yaml` - ThinkTools protocol specification
- `proofguard-deep-research-protocol.yaml` - ProofGuard deep research workflow
- `../ORCHESTRATOR.md` - Master orchestration (module definitions)
- `../reasonkit-core/CLAUDE.md` - Project-specific context

---

## METADATA

```yaml
file: VERBOSE_THINKING_PATTERNS.md
version: 1.0.0
schema: reasonkit-verbose-thinking-v1
created: 2025-12-22
team: Team Epsilon (ThinkTools Enhancement)
license: Apache-2.0
status: PRODUCTION READY
consultation_count: 2 # Claude + Gemini (CONS-008 compliant)
```

---

*"Turn Prompts into Protocols"*
*<https://reasonkit.sh>*
