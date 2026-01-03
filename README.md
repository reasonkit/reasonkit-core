<div align="center">

# ReasonKit

### **The AI Reasoning Engine**

> _"From Prompt to Cognitive Engineering."_ — Turn Prompts into Protocols.

**Auditable Reasoning for Production AI | Rust-Native | SSR/SSG Compatible**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./brand/readme/reasonkit-core_hero.png">
  <source media="(prefers-color-scheme: light)" srcset="./brand/readme/reasonkit-core_hero.png">
  <img src="./brand/readme/reasonkit-core_hero.png" alt="ReasonKit - Auditable Reasoning for Production AI" width="100%">
</picture>

[![CI](https://img.shields.io/github/actions/workflow/status/reasonkit/reasonkit-core/ci.yml?branch=main&style=flat-square&logo=github&label=CI&color=06b6d4&logoColor=06b6d4)](https://github.com/reasonkit/reasonkit-core/actions)
[![Crates.io](https://img.shields.io/crates/v/reasonkit-core?style=flat-square&logo=rust&color=10b981&logoColor=f9fafb)](https://crates.io/crates/reasonkit-core)
[![Docs](https://img.shields.io/badge/docs-reasonkit.sh-06b6d4?style=flat-square&logo=readme&logoColor=f9fafb)](https://docs.reasonkit.sh)
[![License](https://img.shields.io/badge/license-Apache%202.0-a855f7?style=flat-square&labelColor=030508)](https://github.com/reasonkit/reasonkit-core/blob/main/LICENSE)
[![Architecture](https://img.shields.io/badge/stack-Rust%E2%80%A2MCP%E2%80%A2LLMs-f97316?style=flat-square&labelColor=030508)](https://reasonkit.sh)

[Website](https://reasonkit.sh) | [Documentation](https://docs.reasonkit.sh) | [GitHub](https://github.com/reasonkit/reasonkit-core)

</div>

---

## The Problem We Solve

**Most AI is a slot machine.** Insert prompt → pull lever → hope for coherence.

**ReasonKit is a factory.** Input data → execute protocol → get deterministic, auditable output.

LLMs are fundamentally **probabilistic**. Same prompt → different outputs. This creates critical failures:

| Failure           | Impact                    | Our Solution                                      |
| ----------------- | ------------------------- | ------------------------------------------------- |
| **Inconsistency** | Unreliable for production | Deterministic protocol execution                  |
| **Hallucination** | Dangerous falsehoods      | Multi-source triangulation + adversarial critique |
| **Opacity**       | No audit trail            | Complete execution tracing with confidence scores |

**We don't eliminate probability** (impossible). **We constrain it** through structured protocols that force probabilistic outputs into deterministic execution paths.

---

## Quick Start

**1. Install (Universal)**
```bash
curl -fsSL https://reasonkit.sh/install | bash
# Installs 'rk' alias automatically
```

**2. Choose Your Workflow**

### 🤖 Claude Code (Opus 4.5)
*Agentic CLI. No API key required.*
```bash
claude mcp add reasonkit -- rk serve-mcp
claude "Use ReasonKit to analyze: Should we migrate to microservices?"
```

### 🌐 ChatGPT (Browser)
*Manual MCP Bridge. Injects the reasoning protocol directly into the chat.*
```bash
# Generate strict protocol
rk protocol "Should we migrate to microservices?" | pbcopy

# → Paste into ChatGPT: "Execute this protocol..."
```

### ⚡ Gemini 3.0 Pro (API)
*Native CLI integration with Google's latest preview.*
```bash
export GEMINI_API_KEY=AIza...
rk think --model gemini-3.0-pro-preview "Should we migrate to microservices?"
```

> **Note:** The `rk` command is the shorthand alias for `rk`.

**30 seconds to structured reasoning.**

---

## ThinkTools: The 5-Step Reasoning Chain

Each ThinkTool acts as a **variance reduction filter**, transforming probabilistic outputs into increasingly deterministic reasoning paths.

![ReasonKit Protocol Chain - Turn Prompts into Protocols](./brand/readme/powercombo_process.png)

![ReasonKit Core ThinkTool Chain - Variance Reduction](./brand/readme/thinktool_cards_deck.svg)

![ReasonKit Variance Reduction Chart](./brand/readme/chart_variance_reduction.png)

| ThinkTool         | Operation    | What It Does                                    |
| ----------------- | ------------ | ----------------------------------------------- |
| **GigaThink**     | `Diverge()`  | Generate 10+ perspectives, explore widely       |
| **LaserLogic**    | `Converge()` | Detect fallacies, validate logic, find gaps     |
| **BedRock**       | `Ground()`   | First principles decomposition, identify axioms |
| **ProofGuard**    | `Verify()`   | Multi-source triangulation, require 3+ sources  |
| **BrutalHonesty** | `Critique()` | Adversarial red team, attack your own reasoning |

### Variance Reduction: The Chain Effect

**Result:** Raw LLM variance ~85% → Protocol-constrained variance ~28%

---

## Reasoning Profiles

Pre-configured chains for different rigor levels:

![ReasonKit Core Reasoning Profiles Scale](./brand/readme/reasoning_profiles_scale.svg)

```bash
# Fast analysis (70% confidence target)
rk think --profile quick "Is this email phishing?"

# Standard analysis (80% confidence target)
rk think --profile balanced "Should we use microservices?"

# Thorough analysis (85% confidence target)
rk think --profile deep "Design A/B test for feature X"

# Maximum rigor (95% confidence target)
rk think --profile paranoid "Validate cryptographic implementation"
```

| Profile      | Chain                   | Confidence | Use Case           |
| ------------ | ----------------------- | ---------- | ------------------ |
| `--quick`    | GigaThink → LaserLogic  | 70%        | Fast sanity checks |
| `--balanced` | All 5 ThinkTools        | 80%        | Standard decisions |
| `--deep`     | All 5 + meta-cognition  | 85%        | Complex problems   |
| `--paranoid` | All 5 + validation pass | 95%        | Critical decisions |

---

## See It In Action

![ReasonKit Terminal Experience](./brand/readme/terminal_mockup.png)

```text
$ rk think --profile balanced "Should we migrate to microservices?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ThinkTool Chain: GigaThink → LaserLogic → BedRock → ProofGuard
Variance:        85% → 72% → 58% → 42% → 28%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[GigaThink] 10 PERSPECTIVES GENERATED                         Variance: 85%
  1. OPERATIONAL: Maintenance overhead +40% initially
  2. TEAM TOPOLOGY: Conway's Law - do we have the teams?
  3. COST ANALYSIS: Infrastructure scales non-linearly
  ...
  → Variance after exploration: 72% (-13%)

[LaserLogic] HIDDEN ASSUMPTIONS DETECTED                      Variance: 72%
  ⚠ Assuming network latency is negligible
  ⚠ Assuming team has distributed tracing expertise
  ⚠ Logical gap: No evidence microservices solve stated problem
  → Variance after validation: 58% (-14%)

[BedRock] FIRST PRINCIPLES DECOMPOSITION                      Variance: 58%
  • Axiom: Monoliths are simpler to reason about (empirical)
  • Axiom: Distributed systems introduce partitions (CAP theorem)
  • Gap: Cannot prove maintainability improvement without data
  → Variance after grounding: 42% (-16%)

[ProofGuard] TRIANGULATION RESULT                             Variance: 42%
  • 3/5 sources: Microservices increase complexity initially
  • 2/5 sources: Some teams report success
  • Confidence: 0.72 (MEDIUM) - Mixed evidence
  → Variance after verification: 28% (-14%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERDICT: conditional_yes | Confidence: 87% | Duration: 2.3s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**What This Shows:**

- **Transparency:** See exactly where confidence comes from
- **Auditability:** Every step logged and verifiable
- **Deterministic Path:** Same protocol → same execution flow
- **Variance Reduction:** Quantified uncertainty reduction at each stage

---

## Architecture

The ReasonKit architecture uses a **Protocol Engine** wrapper to enforce deterministic execution over probabilistic LLM outputs.

![ReasonKit Core Architecture Exploded View](./brand/readme/core_architecture_exploded.png)

![ReasonKit ThinkTool Chain Architecture](./brand/readme/architecture_diagram.png)

**Three-Layer Architecture:**

1. **Probabilistic LLM** (Unavoidable)
   - LLMs generate tokens probabilistically
   - Same prompt → different outputs
   - We **cannot eliminate** this

2. **Deterministic Protocol Engine** (Our Innovation)
   - Wraps the probabilistic LLM layer
   - Enforces strict execution paths
   - Validates outputs against schemas
   - State machine ensures consistent flow

3. **ThinkTool Chain** (Variance Reduction)
   - Each ThinkTool reduces variance
   - Multi-stage validation catches errors
   - Confidence scoring quantifies uncertainty

**Key Components:**

- **Protocol Engine:** Orchestrates execution with strict state management
- **ThinkTools:** Modular cognitive operations with defined contracts
- **LLM Integration:** Unified client (Claude, GPT, Gemini, 18+ providers)
- **Telemetry:** Local SQLite for execution traces + variance metrics

<details>
<summary><strong>Architecture (Mermaid Diagram)</strong></summary>

```mermaid
flowchart LR
    subgraph CLI["ReasonKit CLI (rk)"]
      A[User Command<br/>rk think --profile balanced]
    end

    subgraph PROTOCOL["Deterministic Protocol Engine"]
      B1[State Machine<br/>Execution Plan]
      B2[ThinkTool Orchestrator]
      B3[(SQLite Trace DB)]
    end

    subgraph LLM["LLM Layer (Probabilistic)"]
      C1[Provider Router]
      C2[Claude / GPT / Gemini / ...]
    end

    subgraph TOOLS["ThinkTools · Variance Reduction"]
      G["GigaThink<br/>Diverge()"]
      LZ["LaserLogic<br/>Converge()"]
      BR["BedRock<br/>Ground()"]
      PG["ProofGuard<br/>Verify()"]
      BH["BrutalHonesty<br/>Critique()"]
    end

    A --> B1 --> B2 --> G --> LZ --> BR --> PG --> BH --> B3
    B2 --> C1 --> C2 --> B2

    classDef core fill:#030508,stroke:#06b6d4,stroke-width:1px,color:#f9fafb;
    classDef tool fill:#0a0d14,stroke:#10b981,stroke-width:1px,color:#f9fafb;
    classDef llm fill:#111827,stroke:#a855f7,stroke-width:1px,color:#f9fafb;

    class CLI,PROTOCOL core;
    class G,LZ,BR,PG,BH tool;
    class LLM,llm C1,C2;
```

</details>

---

## Built for Production

ReasonKit is written in Rust because reasoning infrastructure demands reliability.

| Capability               | What It Means for You                               |
| ------------------------ | --------------------------------------------------- |
| **Predictable Latency**  | <5ms orchestration overhead, no GC pauses           |
| **Memory Safety**        | Zero crashes from null pointers or buffer overflows |
| **Single Binary**        | Deploy anywhere, no Python environment required     |
| **Fearless Concurrency** | Run 100+ reasoning chains in parallel safely        |
| **Type Safety**          | Errors caught at compile time, not runtime          |

**Benchmarked Performance** ([view full report](./docs/reference/PERFORMANCE.md)):

| Operation                          | Time  | Target |
| ---------------------------------- | ----- | ------ |
| Protocol orchestration             | 4.4ms | <10ms  |
| RRF Fusion (100 elements)          | 33μs  | <5ms   |
| Document chunking (10 KB)          | 27μs  | <5ms   |
| RAPTOR tree traversal (1000 nodes) | 33μs  | <5ms   |

**Why This Matters:**

Your AI reasoning shouldn't crash in production. It shouldn't pause for garbage collection during critical decisions. It shouldn't require complex environment management to deploy.

ReasonKit's Rust foundation ensures deterministic, auditable execution every time—the same engineering choice trusted by Linux, Cloudflare, Discord, and AWS for their most critical infrastructure.

---

## Memory Infrastructure (Optional)

**Memory modules (storage, embedding, retrieval, RAPTOR, indexing) are available in the standalone [`reasonkit-mem`](https://crates.io/crates/reasonkit-mem) crate.**

Enable the `memory` feature to use these modules:

```toml
[dependencies]
reasonkit-core = { version = "0.1", features = ["memory"] }
```

**Features:**

- Qdrant vector database (embedded mode)
- Hybrid search (dense + sparse fusion)
- RAPTOR hierarchical retrieval
- Local embeddings (BGE-M3 ONNX)
- BM25 full-text search (Tantivy)

---

## Installation

**Primary Method (Universal):**

```bash
curl -fsSL https://reasonkit.sh/install | bash
```

<details>
<summary><strong>Alternative Methods</strong></summary>

```bash
# Cargo (Rust) - Recommended for Developers
cargo install reasonkit-core

# From Source (Latest Features)
git clone https://github.com/reasonkit/reasonkit-core
cd reasonkit-core && cargo build --release
```

Python bindings available via PyO3 (build from source with `--features python`).

</details>

---

## How to Use

**Command Structure:** `rk <command> [options] [arguments]`

**Standard Operations:**

```bash
# Balanced analysis (5-step protocol)
rk think --profile balanced "Should we migrate our monolith to microservices?"

# Quick sanity check (2-step protocol)
rk think --profile quick "Is this email a phishing attempt?"

# Maximum rigor (paranoid mode)
rk think --profile paranoid "Validate this cryptographic implementation"

# Scientific method (research & experiments)
rk think --profile scientific "Design A/B test for feature X"
```

**With Memory (RAG):**

```bash
# Ingest documents
rk ingest document.pdf

# Query with RAG
rk query "What are the key findings in the research papers?"

# View execution traces
rk trace list
rk trace export <id>
```

---

## Contributing: The 5 Gates of Quality

We demand excellence. All contributions must pass **The 5 Gates of Quality**:

![ReasonKit Quality Gates Shield](./brand/readme/quality_gates_shield.png)

```bash
# Clone & Setup
git clone https://github.com/reasonkit/reasonkit-core
cd reasonkit-core

# The 5 Gates (MANDATORY)
cargo build --release        # Gate 1: Compilation (Exit 0)
cargo clippy -- -D warnings  # Gate 2: Linting (0 errors)
cargo fmt --check            # Gate 3: Formatting (Pass)
cargo test --all-features    # Gate 4: Testing (100% pass)
cargo bench                  # Gate 5: Performance (<5% regression)
```

**Quality Score Target:** 8.0/10 minimum for release.

See [CONTRIBUTING.md](CONTRIBUTING.md) for complete guidelines.

---

## 🏷️ Community Badge

If you use ReasonKit in your project, add our badge:

```markdown
[![Reasoned By ReasonKit](https://raw.githubusercontent.com/reasonkit/reasonkit-core/main/brand/badges/reasoned-by.svg)](https://reasonkit.sh)
```

See [Community Badges](brand/COMMUNITY_BADGES.md) for all variants and usage guidelines.

---

## 🎨 Branding & Design

- [Brand Playbook](brand/BRAND_PLAYBOOK.md) - Complete brand guidelines
- [Component Spec](brand/REASONUI_COMPONENT_SPEC.md) - UI component system
- [Motion Guidelines](brand/MOTION_DESIGN_GUIDELINES.md) - Animation system
- [3D Assets](brand/3D_ASSET_STRATEGY.md) - WebGL integration guide
- [Integration Guide](brand/BRANDING_INTEGRATION_GUIDE.md) - Complete integration instructions

---

## Design Philosophy: Honest Engineering

**We don't claim to eliminate probability.** That's impossible. LLMs are probabilistic by design.

**We do claim to constrain it.** Through structured protocols, multi-stage validation, and deterministic execution paths, we transform probabilistic token generation into auditable reasoning chains.

| What We Battle    | How We Battle It                                 | What We're Honest About                           |
| ----------------- | ------------------------------------------------ | ------------------------------------------------- |
| **Inconsistency** | Deterministic protocol execution                 | LLM outputs still vary, but execution paths don't |
| **Hallucination** | Multi-source triangulation, adversarial critique | Can't eliminate, but can detect and flag          |
| **Opacity**       | Full execution tracing, confidence scoring       | Transparency doesn't guarantee correctness        |
| **Uncertainty**   | Explicit confidence metrics, variance reduction  | We quantify uncertainty, not eliminate it         |

---

## Version & Maturity

| Component            | Status        | Notes                                                  |
| -------------------- | ------------- | ------------------------------------------------------ |
| **ThinkTools Chain** | ✅ Stable     | Core reasoning protocols production-ready              |
| **MCP Server**       | ✅ Stable     | Model Context Protocol integration                     |
| **CLI**              | 🔶 Scaffolded | `mcp`, `serve-mcp`, `completions` work; others planned |
| **Memory Features**  | ✅ Stable     | Via `reasonkit-mem` crate                              |
| **Python Bindings**  | 🔶 Beta       | Build from source with `--features python`             |

**Current Version:** v0.1.2 | [CHANGELOG](CHANGELOG.md) | [Releases](https://github.com/reasonkit/reasonkit-core/releases)

### Verify Installation

```bash
# Check version
rk --version

# Verify MCP server starts
rk serve-mcp --help

# Run a quick test (requires LLM API key)
OPENAI_API_KEY=your-key rk mcp
```

---

## License

**Apache 2.0** - See [LICENSE](https://github.com/reasonkit/reasonkit-core/blob/main/LICENSE)

**Open Source Core:** All core reasoning protocols and ThinkTools are open source under Apache 2.0.

---

<div align="center">

![ReasonKit Ecosystem Connection](./brand/readme/ecosystem_connection.png)

**ReasonKit** — Turn Prompts into Protocols

_Designed, Not Dreamed_

[Website](https://reasonkit.sh) | [Documentation](https://docs.reasonkit.sh) | [GitHub](https://github.com/reasonkit/reasonkit-core)

</div>
