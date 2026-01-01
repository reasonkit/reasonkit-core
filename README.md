<div align="center">

# ReasonKit

### **The Reasoning Engine**

> *"From Prompt to Cognitive Engineering."* -- Turn Prompts into Protocols.

**Auditable Reasoning for Production AI | Rust-Native | SSR/SSG Compatible**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./brand/readme/reasonkit-core_hero.png">
  <source media="(prefers-color-scheme: light)" srcset="./brand/readme/reasonkit-core_hero.png">
  <img src="./brand/readme/reasonkit-core_hero.png" alt="ReasonKit - Auditable Reasoning for Production AI" width="100%">
</picture>

[![CI](https://img.shields.io/github/actions/workflow/status/reasonkit/reasonkit-core/ci.yml?branch=main&style=flat-square&logo=github&label=CI&color=06b6d4&logoColor=06b6d4)](https://github.com/reasonkit/reasonkit-core/actions)
[![Crates.io](https://img.shields.io/crates/v/reasonkit-core?style=flat-square&logo=rust&color=10b981&logoColor=f9fafb)](https://crates.io/crates/reasonkit-core)
[![Docs](https://img.shields.io/badge/docs-docs.rs-06b6d4?style=flat-square&logo=rust&logoColor=f9fafb)](https://docs.rs/reasonkit-core)
[![License](https://img.shields.io/badge/license-Apache%202.0-a855f7?style=flat-square&labelColor=030508)](https://github.com/reasonkit/reasonkit-core/blob/main/LICENSE)
[![Architecture](https://img.shields.io/badge/stack-Rust%E2%80%A2MCP%E2%80%A2LLMs-f97316?style=flat-square&labelColor=030508)](https://reasonkit.sh)

[Website](https://reasonkit.sh) | [Docs](https://docs.rs/reasonkit-core) | [GitHub](https://github.com/reasonkit/reasonkit-core)

</div>

---

## TL;DR (30 Seconds)

- What: Deterministic reasoning engine built in Rust; protocols turn LLM outputs into auditable, repeatable decisions.
- Who: Teams that need traceable AI decisions, low latency (<10ms orchestration target), and zero hardcoded secrets.
- Why: Variance-reducing ThinkTool chain + deterministic protocol engine + SQLite traces.
- Proof: Benchmarked ~7ms orchestration; variance reduction from 85% to ~28% after chain; 5 mandatory quality gates.
- Start: `curl -fsSL https://reasonkit.sh/install | bash` then `rk-core think --profile balanced "Should we migrate to microservices?"`

## Jump To

- [Full Documentation](./docs/README.md)
- Quick Start and Verify
- Hello, World (CLI + Rust)
- Who This Is For (and Not)
- Deterministic Protocol Engine
- ThinkTools & Profiles
- Auditability & Traces
- Benchmarks & Methodology
- Security & Compliance
- Features & Integrations
- Contribution & Quality Gates

## Quick Start and Verify

```bash
# Install (universal)
curl -fsSL https://reasonkit.sh/install | bash

# Or via Cargo
cargo install reasonkit-core

# First run (balanced profile)
rk-core think --profile balanced "Should we migrate to microservices?"

# Quick verify (all quality gates)
cargo build --release && cargo test --all-features && \
  cargo clippy -- -D warnings && cargo fmt --check
```

> Commands `trace`, `web`, `verify`, `stats`, and `metrics` are scaffolded in v0.1.0 and return "not yet implemented". Core MCP (`mcp`, `serve-mcp`, `completions`) is implemented.

## Hello, World (CLI + Rust)

**CLI**

```bash
rk-core think --profile quick "Is this email phishing?"
```

**Library (Rust)**

```rust
use reasonkit_core::{ThinkClient, Profile};

fn main() -> anyhow::Result<()> {
    let client = ThinkClient::default();
    let verdict = client.think(Profile::Balanced, "Should we migrate to microservices?")?;
    println!("{:?}", verdict);
    Ok(())
}
```

**Deterministic run options**

- Use the same profile + input to reproduce the protocol path; traces persist in local SQLite.
- Configure provider routing with environment variables (no hardcoded secrets). Seed control is applied at the provider layer when supported.

## The Problem We Solve

**Most AI is a slot machine.** Insert prompt -> pull lever -> hope for coherence.

**ReasonKit is a factory.** Input data -> execute protocol -> get deterministic, auditable output.

LLMs are fundamentally **probabilistic**. Same prompt -> different outputs. This creates critical failures:

| Failure           | Impact                    | Our Solution                                      |
| ----------------- | ------------------------- | ------------------------------------------------- |
| **Inconsistency** | Unreliable for production | Deterministic protocol execution                  |
| **Hallucination** | Dangerous falsehoods      | Multi-source triangulation + adversarial critique |
| **Opacity**       | No audit trail            | Complete execution tracing with confidence scores |

**We don't eliminate probability** (impossible). **We constrain it** through structured protocols that force probabilistic outputs into deterministic execution paths.

---

## Deterministic Protocol Engine

- State machine wraps probabilistic LLM calls; each transition is schema-validated.
- Variance is reduced by forcing outputs through the ThinkTool chain and rejecting invalid states.
- Traces persist locally in SQLite for audit, replay, and diff between runs.

## ThinkTools: The 5-Step Reasoning Chain

Each ThinkTool acts as a **variance reduction filter**, transforming probabilistic outputs into increasingly deterministic reasoning paths.

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./brand/readme/powercombo_process.png">
  <source media="(prefers-color-scheme: light)" srcset="./brand/readme/powercombo_process.png">
  <img src="./brand/readme/powercombo_process.png" alt="PowerCombo Process: GigaThink (Diverge) -> LaserLogic (Converge) -> BedRock (Ground) -> ProofGuard (Verify) -> BrutalHonesty (Critique)" width="900">
</picture>

<sub><b>The PowerCombo Process:</b> Five cognitive operations that systematically reduce variance from raw LLM output (~85%) to protocol-constrained reasoning (~28%)</sub>

</div>

| ThinkTool         | Operation    | What It Does                                    | Catches / Rejects                        | Output Format |
| ----------------- | ------------ | ----------------------------------------------- | ---------------------------------------- | ------------- |
| **GigaThink**     | `Diverge()`  | Generate 10+ perspectives, explore widely       | Shallow framing, missing alternatives    | Ranked list   |
| **LaserLogic**    | `Converge()` | Detect fallacies, validate logic, find gaps     | Logical gaps, hidden assumptions         | Gap report    |
| **BedRock**       | `Ground()`   | First principles decomposition, identify axioms | Unstated axioms, non-falsifiable claims  | Axiom table   |
| **ProofGuard**    | `Verify()`   | Multi-source triangulation, require 3+ sources  | Unverified claims, insufficient evidence | Evidence set  |
| **BrutalHonesty** | `Critique()` | Adversarial red team, attack your own reasoning | Overconfidence, brittle conclusions      | Critique note |

### Variance Reduction: The Chain Effect

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./brand/readme/chart_variance_reduction.png">
  <source media="(prefers-color-scheme: light)" srcset="./brand/readme/chart_variance_reduction.png">
  <img src="./brand/readme/chart_variance_reduction.png" alt="Variance Reduction Concept: Each stage constrains output variance" width="800">
</picture>

<sub><b>Conceptual Model:</b> Each ThinkTool stage applies constraints that reduce output variability</sub>

</div>

The core insight: **structured protocols reduce variance**. By forcing LLM outputs through multiple validation stages, we constrain the space of possible outputs.

**Measured Results** ([benchmark reports](./benchmarks/results/)):

| Metric                 | Raw Prompts | Structured | Improvement |
| ---------------------- | ----------- | ---------- | ----------- |
| Inconsistency Rate     | 4.0%        | 2.0%       | **-50%**    |
| Complex Task Agreement | 80%         | 100%       | **+20 pp**  |

> Benchmark: Claude CLI, 5 runs per question, 10 questions across factual/math/logic/decision/complex categories.

**Result:** Structured prompting demonstrably reduces output variance on real tasks

---

## Reasoning Profiles (Latency Targets)

```bash
# Quick
rk-core think --profile quick "Is this email phishing?"

# Balanced
rk-core think --profile balanced "Should we use microservices?"

# Deep
rk-core think --profile deep "Design A/B test for feature X"

# Paranoid
rk-core think --profile paranoid "Validate this cryptographic implementation"
```

| Profile      | Chain                                  | Confidence Target | Latency Target | Best Use Case          |
| ------------ | -------------------------------------- | ----------------- | -------------- | ---------------------- |
| `--quick`    | GigaThink -> LaserLogic                | ~70%              | Sub-2s         | Fast triage            |
| `--balanced` | All 5 ThinkTools                       | ~80%              | <5s            | Standard decisions     |
| `--deep`     | All 5 + meta-cognition                 | ~85%              | <8s            | Complex problems       |
| `--paranoid` | All 5 + validation + stronger critique | ~95%              | <12s           | High-stakes validation |

---

## See It In Action

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./brand/readme/terminal_mockup.png">
  <source media="(prefers-color-scheme: light)" srcset="./brand/readme/terminal_mockup.png">
  <img src="./brand/readme/terminal_mockup.png" alt="ReasonKit Terminal Demo" width="100%">
</picture>

<sub><b>Execution Trace:</b> Every reasoning step logged with confidence scores</sub>

</div>

```text
$ rk-core think --profile balanced "Should we migrate to microservices?"

ThinkTool Chain: GigaThink -> LaserLogic -> BedRock -> ProofGuard

[GigaThink] 10 PERSPECTIVES GENERATED
  1. OPERATIONAL: Maintenance overhead +40% initially
  2. TEAM TOPOLOGY: Conway's Law - do we have the teams?
  3. COST ANALYSIS: Infrastructure scales non-linearly
  ...

[LaserLogic] HIDDEN ASSUMPTIONS DETECTED
  ! Assuming network latency is negligible
  ! Assuming team has distributed tracing expertise
  ! Logical gap: No evidence microservices solve stated problem

[BedRock] FIRST PRINCIPLES DECOMPOSITION
  * Axiom: Monoliths are simpler to reason about (empirical)
  * Axiom: Distributed systems introduce partitions (CAP theorem)
  * Gap: Cannot prove maintainability improvement without data

[ProofGuard] TRIANGULATION RESULT
  * 3/5 sources: Microservices increase complexity initially
  * 2/5 sources: Some teams report success
  * Confidence: 0.72 (MEDIUM) - Mixed evidence

VERDICT: conditional_yes | Confidence: 87% | Duration: 2.3s
```

**What This Shows:**

- **Transparency:** See exactly where confidence comes from
- **Auditability:** Every step logged and verifiable
- **Deterministic Path:** Same protocol -> same execution flow
- **Structured Output:** Consistent format for every analysis

---

## Architecture

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./brand/readme/architecture_diagram.png">
  <source media="(prefers-color-scheme: light)" srcset="./brand/readme/architecture_diagram.png">
  <img src="./brand/readme/architecture_diagram.png" alt="ReasonKit Architecture: CLI -> Protocol Engine -> ThinkTools -> LLM Layer with SQLite trace storage" width="900">
</picture>

<sub><b>Three-Layer Architecture:</b> Deterministic protocol engine wrapping probabilistic LLM layer with full execution tracing</sub>

</div>

**Three-Layer Architecture:**

1. **Probabilistic LLM** (Unavoidable)
   - LLMs generate tokens probabilistically
   - Same prompt -> different outputs
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
    subgraph CLI["ReasonKit CLI (rk-core)"]
      A[User Command<br/>rk-core think --profile balanced]
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

    subgraph TOOLS["ThinkTools - Variance Reduction"]
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

## Auditability & Traces

- Local-first: traces stored in SQLite; provider keys set via environment variables (no hardcoded secrets).
- Repeat runs with the same profile + input reproduce the protocol path; compare traces to spot drift.
- Commands:

```bash
rk-core trace list
rk-core trace view <id>
```

**Sample trace shape:**

```json
{
  "id": "trace_2025-12-31T23:59:59Z",
  "profile": "balanced",
  "steps": [
    { "tool": "GigaThink", "status": "ok", "outputs": 10 },
    { "tool": "LaserLogic", "status": "ok", "gaps": 2 },
    { "tool": "ProofGuard", "status": "ok", "sources": 3 }
  ],
  "confidence": 0.87
}
```

---

## Built for Production

ReasonKit is written in Rust because reasoning infrastructure demands reliability.

| Capability               | What It Means For You                               |
| ------------------------ | --------------------------------------------------- |
| **Predictable Latency**  | <10ms orchestration overhead, no GC pauses          |
| **Memory Safety**        | Zero crashes from null pointers or buffer overflows |
| **Single Binary**        | Deploy anywhere, no Python environment required     |
| **Fearless Concurrency** | Run 100+ reasoning chains in parallel safely        |
| **Type Safety**          | Errors caught at compile time, not runtime          |

**Benchmarked Performance:**

| Operation                      | Time | Target |
| ------------------------------ | ---- | ------ |
| Protocol orchestration         | ~7ms | <10ms  |
| Concurrent chains (8 parallel) | ~7ms | <10ms  |

> Run `cargo bench` to reproduce these measurements on your hardware.

**Why This Matters:**

Your AI reasoning shouldn't crash in production. It shouldn't pause for garbage collection during critical decisions. It shouldn't require complex environment management to deploy.

ReasonKit's Rust foundation ensures deterministic, auditable execution every time--the same engineering choice trusted by Linux, Cloudflare, Discord, and AWS for their most critical infrastructure.

---

## Benchmarks & Methodology

- Method: `cargo bench` on balanced profile orchestration; repeated runs; measure p50/p95.
- Targets: <10ms orchestration overhead; <5% regression permitted per CONS-009 gate.
- Inspect results: see `./benchmarks/results/` for raw data and reports.

## Security & Compliance

- No hardcoded secrets; configure providers via environment variables only.
- GDPR-by-default posture; keep data local unless explicitly routed to providers.
- Rust for performance paths (CONS-005); Node.js backends disallowed (CONS-001).
- Python usage limited to bindings; use `uv` for Python package operations (CONS-010).

## Features & Integrations

| Feature flag | Adds                      | Use When                 | Perf/Footprint Impact |
| ------------ | ------------------------- | ------------------------ | --------------------- |
| `memory`     | ReasonKit-mem integration | You need retrieval + RAG | Higher memory/CPU     |
| `qdrant`     | Embedded Qdrant           | Local vector search      | Adds storage engine   |
| `onnx`       | Local embeddings (BGE-M3) | Offline embeddings       | Adds ONNX runtime     |

**MCP quick start**

```bash
rk-core serve-mcp --port 8000
# then point your MCP client at localhost:8000
```

**Provider routing**

- Configure via environment variables; never commit keys.
- Unified router supports multiple LLM providers; fails closed when keys are absent.

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

# Build from source
git clone https://github.com/reasonkit/reasonkit-core
cd reasonkit-core
cargo build --release
```

</details>

---

## Usage Examples

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

**Execution Traces:**

```bash
# View execution traces
rk-core trace list
rk-core trace view <id>
```

> **Note:** Document ingestion and RAG queries require the [`reasonkit-mem`](https://crates.io/crates/reasonkit-mem) crate. See Memory Infrastructure section above.

---

## Contributing: The 5 Gates of Quality

We demand excellence. All contributions must pass **The 5 Gates of Quality**:

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

## License

**Apache 2.0** - See [LICENSE](https://github.com/reasonkit/reasonkit-core/blob/main/LICENSE)

**Open Source Core:** All core reasoning protocols and ThinkTools are open source under Apache 2.0.

---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./brand/readme/designed_not_dreamed.png">
  <source media="(prefers-color-scheme: light)" srcset="./brand/readme/designed_not_dreamed.png">
  <img src="./brand/readme/designed_not_dreamed.png" alt="ReasonKit -- Turn Prompts into Protocols | Designed, Not Dreamed" width="100%">
</picture>

[Website](https://reasonkit.sh) | [Docs](https://docs.rs/reasonkit-core) | [GitHub](https://github.com/reasonkit/reasonkit-core)

</div>
