<div align="center">

# ReasonKit

```
██████╗ ███████╗ █████╗ ███████╗ ██████╗ ███╗   ██╗██╗  ██╗██╗████████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║██║ ██╔╝██║╚══██╔══╝
██████╔╝█████╗  ███████║███████╗██║   ██║██╔██╗ ██║█████╔╝ ██║   ██║
██╔══██╗██╔══╝  ██╔══██║╚════██║██║   ██║██║╚██╗██║██╔═██╗ ██║   ██║
██║  ██║███████╗██║  ██║███████║╚██████╔╝██║ ╚████║██║  ██╗██║   ██║
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝   ╚═╝
```

### **Turn Prompts into Protocols**

> *"Designed, Not Dreamed."* — Structure beats intelligence. Engineering over hope.

**Industrial-Grade Reasoning Infrastructure | Rust-Native | Zero-Trust Logic**

[![CI](https://img.shields.io/github/actions/workflow/status/reasonkit/reasonkit-core/ci.yml?branch=main&style=flat-square&logo=github&label=CI&color=00d2ff&logoColor=00d2ff)](https://github.com/reasonkit/reasonkit-core/actions)
[![Crates.io](https://img.shields.io/crates/v/reasonkit-core?style=flat-square&logo=rust&color=00d2ff&logoColor=00d2ff)](https://crates.io/crates/reasonkit-core)
[![License](https://img.shields.io/badge/license-Apache%202.0-00d2ff?style=flat-square)](LICENSE)

[Website](https://reasonkit.sh) | [Documentation](https://docs.reasonkit.sh) | [GitHub](https://github.com/reasonkit/reasonkit-core)

</div>

---

## ⚡ The Problem We Solve

**Most AI is a slot machine.** Insert prompt → pull lever → hope for coherence.

**ReasonKit is a factory.** Input data → define protocol → get deterministic, auditable output.

LLMs are fundamentally **probabilistic**. Same prompt → different outputs. This creates three critical failures:

| Failure        | Impact                      | Our Solution                                          |
| -------------- | --------------------------- | ----------------------------------------------------- |
| **Inconsistency** | Unreliable for production   | Deterministic protocol execution                      |
| **Hallucination** | Dangerous falsehoods        | Multi-source triangulation + adversarial critique     |
| **Opacity**     | No audit trail              | Complete execution tracing with confidence scores      |

**We don't eliminate probability** (impossible). **We constrain it** through structured protocols that force probabilistic outputs into deterministic execution paths.

<div align="center">
  <img src="https://raw.githubusercontent.com/reasonkit/reasonkit-core/main/assets/img/hero_reasoning_engine.png" alt="Reasoning Engine Visualization" width="800" />
</div>

---

## 🚀 Quick Start

```bash
# Install (Universal)
curl -fsSL https://reasonkit.sh/install | bash

# Or via Cargo
cargo install reasonkit-core

# Run your first analysis
rk-core think --profile balanced "Should we migrate to microservices?"
```

**30 seconds to structured reasoning.**

---

## 🧠 ThinkTools: Variance Reduction Filters

Each ThinkTool acts as a **variance reduction filter**, transforming probabilistic outputs into increasingly deterministic reasoning paths.

| ThinkTool       | Operation                              | Variance Reduction                        |
| --------------- | -------------------------------------- | ---------------------------------------- |
| **GigaThink** 💡 | `Diverge()` - 10+ perspectives        | Exploit probability to explore widely     |
| **LaserLogic** ⚡ | `Converge()` - Logical validation      | Detect fallacies, hidden assumptions     |
| **BedRock** 🪨 | `Ground()` - First principles         | Rebuild from axioms, identify gaps       |
| **ProofGuard** 🛡️ | `Verify()` - Multi-source triangulation | Require 3+ sources, flag contradictions |
| **BrutalHonesty** 🔥 | `Critique()` - Adversarial red team   | Attack your reasoning, find flaws         |

**The Chain Effect:**

- **Single LLM call:** 85% variance, unpredictable
- **GigaThink → LaserLogic:** 40-60% reduction
- **+ BedRock → ProofGuard:** 70-85% reduction
- **+ BrutalHonesty:** 85-95% reduction

**Result:** Raw LLM variance ~85% → Protocol-constrained variance ~28%

<div align="center">
  <img src="https://raw.githubusercontent.com/reasonkit/reasonkit-core/main/assets/img/chart_variance_reduction.png" alt="Variance Reduction Graph" width="600" />
</div>

---

## 📊 Reasoning Profiles

Pre-configured chains for different rigor levels:

```bash
# Fast analysis (70% confidence target)
rk-core think --profile quick "Is this email phishing?"

# Standard analysis (80% confidence target)
rk-core think --profile balanced "Should we use microservices?"

# Thorough analysis (85% confidence target)
rk-core think --profile deep "Design A/B test for feature X"

# Maximum rigor (95% confidence target)
rk-core think --profile paranoid "Validate cryptographic implementation"
```

| Profile     | Chain                        | Confidence | Use Case           |
| ----------- | ---------------------------- | ---------- | ------------------ |
| `--quick`   | GigaThink → LaserLogic       | 70%        | Fast sanity checks |
| `--balanced` | All 5 ThinkTools             | 80%        | Standard decisions |
| `--deep`    | All 5 + meta-cognition       | 85%        | Complex problems   |
| `--paranoid` | All 5 + validation pass     | 95%        | Critical decisions |

---

## 📈 Execution Trace Example

<div align="center">
  <img src="https://raw.githubusercontent.com/reasonkit/reasonkit-core/main/assets/img/terminal_mockup.png" alt="ReasonKit CLI Interface" width="700" />
</div>

```text
ThinkTool Chain: GigaThink → LaserLogic → BedRock → ProofGuard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Variance Reduction: 85% → 72% → 58% → 42% → 28%

[GigaThink] 10 PERSPECTIVES GENERATED (Variance: 85%)
  1. OPERATIONAL: Maintenance overhead +40% initially
  2. TEAM TOPOLOGY: Conway's Law - do we have the teams?
  3. COST ANALYSIS: Infrastructure scales non-linearly
  ...
  Variance after exploration: 72% (-13%)

[LaserLogic] HIDDEN ASSUMPTIONS DETECTED (Variance: 72%)
  - Assuming network latency is negligible (Fallacy: Distributed Computing)
  - Assuming team has distributed tracing expertise
  - Logical gap: No evidence microservices solve stated problem
  Variance after validation: 58% (-14%)

[BedRock] FIRST PRINCIPLES DECOMPOSITION (Variance: 58%)
  - Axiom: Monoliths are simpler to reason about (empirical)
  - Axiom: Distributed systems introduce partitions (CAP theorem)
  - Gap: Cannot prove maintainability improvement without data
  Variance after grounding: 42% (-16%)

[ProofGuard] TRIANGULATION RESULT (Variance: 42%)
  - 3/5 sources: Microservices increase complexity initially
  - 2/5 sources: Some teams report success
  - Confidence: 0.72 (MEDIUM) - Mixed evidence
  Variance after verification: 28% (-14%)

[Verdict]
Confidence: 87% | Execution: 48ms | Grade: B- | Final Variance: 28%
Recommendation: HALT. Prerequisites not met. Assess team capability first.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Probabilistic Input → Deterministic Protocol → Auditable Output
```

**What This Shows:**

- **Transparency:** See exactly where confidence comes from
- **Auditability:** Every step logged and verifiable
- **Deterministic Path:** Same protocol → same execution flow
- **Variance Reduction:** Quantified uncertainty reduction at each stage

---

## 🏗️ Architecture: Deterministic Wrapper for Probabilistic Core

```
                    ┌─────────────────────────────────┐
                    │      ReasonKit CLI              │
                    │  (Rust-based Orchestrator)      │
                    │  Protocol: Deterministic        │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │     Protocol Engine             │
                    │  (Deterministic State Machine)  │
                    │  Execution Trace: SQLite        │
                    │  ─────────────────────────────  │
                    │  ⚠️  LLM Layer (Probabilistic)  │
                    │     ↓ Constrained by Protocol   │
                    │  ✅  Protocol Layer (Deterministic)│
                    └───────────────┬─────────────────┘
                                    │
        ┌───────────┬───────────────┼───────────────┬───────────┐
        ▼           ▼               ▼               ▼           ▼
   GigaThink   LaserLogic      BedRock       ProofGuard   BrutalHonesty
   (Diverge)   (Converge)      (Ground)       (Verify)       (Critique)
        │           │               │               │           │
        └───────────┴───────────────┴───────────────┴───────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │     Telemetry & Audit Trail     │
                    │  (Local SQLite, Privacy-First)   │
                    │  Variance Reduction Metrics      │
                    └─────────────────────────────────┘
```

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

---

## 🛡️ The Rust Supremacy Doctrine

**Zero overhead. Type safety. Memory safety. Fearless concurrency.**

| Feature        | Specification        | Impact                              |
| -------------- | -------------------- | ----------------------------------- |
| **Zero Overhead** | Rust-native binaries | 10-100x faster than Python          |
| **Type Safety** | Strict compilation   | Errors caught at build time         |
| **Memory Safety** | No GC pauses         | Predictable latency (<5ms)          |
| **Concurrency** | Fearless parallelism | 100+ reasoning chains in parallel   |
| **Determinism** | No undefined behavior | Auditable execution traces          |

**We don't use Python for core logic.** We don't tolerate interpreter overhead. We don't compromise on performance.

<div align="center">
  <img src="https://raw.githubusercontent.com/reasonkit/reasonkit-core/main/assets/img/chart_speed_comparison.png" alt="Python vs Rust Performance Comparison" width="600" />
</div>

---

## 💾 Memory Infrastructure (Optional)

**Memory modules (storage, embedding, retrieval, RAPTOR, indexing) are now in the standalone [`reasonkit-mem`](https://crates.io/crates/reasonkit-mem) crate.**

Enable the `memory` feature to use these modules:

```toml
[dependencies]
reasonkit-core = { version = "1.0", features = ["memory"] }
```

This automatically includes `reasonkit-mem` and re-exports its modules. The RAG engine (with full LLM integration) remains in `reasonkit-core` and uses `reasonkit-mem` for storage/retrieval operations.

**Features:**
- Qdrant vector database (embedded mode)
- Hybrid search (dense + sparse fusion)
- RAPTOR hierarchical retrieval
- Local embeddings (BGE-M3 ONNX)
- BM25 full-text search (Tantivy)

---

## 🔧 Installation

**Primary Method (Universal):**
```bash
curl -fsSL https://reasonkit.sh/install | bash
```

<details>
<summary><strong>Alternative Methods</strong></summary>

```bash
# Cargo (Rust) - Recommended for Developers
cargo install reasonkit-core

# npm (Node.js) - CLI Wrapper
npm install -g @reasonkit/cli

# uv (Python) - Bindings Only (PIP IS BANNED)
uv pip install reasonkit
```

</details>

---

## 📖 Usage Examples

**Standard Operations:**

```bash
# Balanced analysis (5-step protocol)
rk-core think --profile balanced "Should we migrate our monolith to microservices?"

# Quick sanity check (2-step protocol)
rk-core think --profile quick "Is this email a phishing attempt?"

# Maximum rigor (paranoid mode)
rk-core think --profile paranoid "Validate this cryptographic implementation"

# Scientific method (research & experiments)
rk-core think --profile scientific "Design A/B test for feature X"
```

**With Memory (RAG):**

```bash
# Ingest documents
rk-core ingest document.pdf

# Query with RAG
rk-core query "What are the key findings in the research papers?"

# View execution traces
rk-core trace list
rk-core trace export <id>
```

---

## 🤝 Contributing: The 5 Gates of Quality

We demand excellence. All contributions must pass **The 5 Gates of Quality**:

```bash
# Clone & Setup
git clone https://github.com/reasonkit/reasonkit-core
cd reasonkit-core

# The 5 Gates (MANDATORY - CONS-009)
cargo build --release        # Gate 1: Compilation (Exit 0)
cargo clippy -- -D warnings  # Gate 2: Linting (0 errors)
cargo fmt --check            # Gate 3: Formatting (Pass)
cargo test --all-features    # Gate 4: Testing (100% pass)
cargo bench                  # Gate 5: Performance (<5% regression)
```

**Quality Score Target:** 8.0/10 minimum for release.

See [CONTRIBUTING.md](CONTRIBUTING.md) for complete guidelines.

---

## 🎯 Design Philosophy: Honest Engineering

**We don't claim to eliminate probability.** That's impossible. LLMs are probabilistic by design.

**We do claim to constrain it.** Through structured protocols, multi-stage validation, and deterministic execution paths, we transform probabilistic token generation into auditable reasoning chains.

| What We Battle | How We Battle It                                    | What We're Honest About                              |
| -------------- | --------------------------------------------------- | ---------------------------------------------------- |
| **Inconsistency** | Deterministic protocol execution                    | LLM outputs still vary, but execution paths don't    |
| **Hallucination** | Multi-source triangulation, adversarial critique   | Can't eliminate, but can detect and flag            |
| **Opacity**     | Full execution tracing, confidence scoring          | Transparency doesn't guarantee correctness           |
| **Uncertainty** | Explicit confidence metrics, variance reduction     | We quantify uncertainty, not eliminate it           |

**Industrial Cyberpunk Aesthetic:**

- **Colors:** Cyan (`#00d2ff`) primary, Void Black (`#03060b`) background
- **Typography:** IBM Plex Serif (headlines), Space Grotesk (body), IBM Plex Mono (code)
- **Voice:** Authoritative, technical, clear, confident, transparent
- **Principle:** "Designed, Not Dreamed" — Structure beats intelligence. Engineering over hope.

**We don't build software. We build machinery for the mind. We don't eliminate chaos. We constrain it.**

---

## 📜 License

**Apache 2.0** - See [LICENSE](LICENSE)

**Open Source Core:** All core reasoning protocols and ThinkTools are open source under Apache 2.0.

---

<div align="center">

**ReasonKit** — Turn Prompts into Protocols

*Designed, Not Dreamed*

[Website](https://reasonkit.sh) | [Documentation](https://docs.reasonkit.sh) | [GitHub](https://github.com/reasonkit/reasonkit-core) | [Discord](https://discord.gg/reasonkit)

</div>
