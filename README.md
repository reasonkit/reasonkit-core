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

### Turn Prompts into Protocols

**Engineered in Rust. Designed for Determinism.**

[![CI](https://img.shields.io/github/actions/workflow/status/reasonkit/reasonkit-core/ci.yml?branch=main&style=flat-square&logo=github&label=CI&color=00d2ff)](https://github.com/reasonkit/reasonkit-core/actions)
[![Crates.io](https://img.shields.io/crates/v/reasonkit-core?style=flat-square&logo=rust&color=orange)](https://crates.io/crates/reasonkit-core)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Discord](https://img.shields.io/discord/1234567890?style=flat-square&logo=discord&label=Discord&color=5865F2)](https://discord.gg/reasonkit)

[Website](https://reasonkit.sh) | [Documentation](https://docs.reasonkit.sh) | [Discord](https://discord.gg/reasonkit)

</div>

---

## ⚡ Stop Gambling with Probabilistic AI

**Most AI is a slot machine.** You insert a prompt, pull the lever, and hope for a coherent result.
**ReasonKit is a factory.** You input raw data, define the protocol, and get a deterministic, auditable output.

We force Large Language Models to abandon their chaotic nature and adhere to strict, verifiable logic paths.

> **"Designed, Not Dreamed."** — Structure beats intelligence.

---

## 🛡️ The Rust Supremacy

We don't use Python for core logic. We don't tolerate interpreter overhead.
ReasonKit is built on **The Rust Supremacy** doctrine:

| Feature | Specification | Why It Matters |
|---------|---------------|----------------|
| **Zero Overhead** | Rust-native binaries | 10-100x faster than Python chains |
| **Type Safety** | Strict compilation | Logic errors caught at build time, not runtime |
| **Memory Safety** | No GC pauses | Predictable latency (<5ms core loops) |
| **Concurrency** | Fearless parallelism | Run 100+ reasoning chains in parallel |

---

## 🚀 Install in 30 Seconds

```bash
curl -fsSL https://reasonkit.sh/install | bash
```

<details>
<summary>Other installation methods</summary>

```bash
# Cargo (Rust) - Recommended for Developers
cargo install reasonkit-core

# npm (Node.js)
npm install -g @reasonkit/cli

# pip (Python) - Bindings Only
pip install reasonkit
```

</details>

---

## 🔧 Quick Usage

```bash
# Balanced Analysis (Standard 5-step Protocol)
rk-core think --profile balanced "Should we migrate our monolith to microservices?"

# Quick Sanity Check (2-step Protocol)
rk-core think --profile quick "Is this email a phishing attempt?"

# Maximum Rigor (Paranoid Mode - 95% Confidence Target)
rk-core think --profile paranoid "Validate this cryptographic implementation proposal."
```

**Sample Output:**

```text
ThinkTool Chain: GigaThink -> LaserLogic -> BedRock -> ProofGuard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[GigaThink] 10 PERSPECTIVES GENERATED
  1. OPERATIONAL: Maintenance overhead increases by ~40% initially.
  2. TEAM TOPOLOGY: Conway's Law - do we have the teams to support this?
  ...

[LaserLogic] HIDDEN ASSUMPTIONS DETECTED
  - Assuming network latency is negligible (Fallacy: Fallacies of Distributed Computing)
  - Assuming current team has distributed tracing expertise.

[Verdict]
Confidence: 87% | Execution: 48ms | Grade: B-
Recommendation: HALT. Prerequisites not met.
```

---

## 🧠 ThinkTools (Cognitive Modules)

Standard LLMs hallucinate. ReasonKit **ThinkTools** enforce specific cognitive operations.

| Module | Function | Description |
|--------|----------|-------------|
| **GigaThink** | `Diverge()` | Forces 10+ distinct perspectives to prevent tunnel vision. |
| **LaserLogic** | `Converge()` | Deductive reasoning engine. Hunts for logical fallacies. |
| **BedRock** | `Ground()` | First principles decomposition. Strips away complexity. |
| **ProofGuard** | `Verify()` | Evidence triangulation. Checks claims against known facts. |
| **BrutalHonesty** | `Critique()` | Adversarial self-correction. The "Red Team" in a box. |

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────┐
                    │         ReasonKit CLI           │
                    │   (Rust-based Orchestrator)     │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │       Protocol Engine           │
                    │   (Deterministic State Machine) │
                    └───────────────┬─────────────────┘
                                    │
        ┌───────────┬───────────────┼───────────────┬───────────┐
        ▼           ▼               ▼               ▼           ▼
   GigaThink   LaserLogic      BedRock       ProofGuard   BrutalHonesty
   (Diverge)   (Converge)      (Ground)       (Verify)       (Cut)
        │           │               │               │           │
        └───────────┴───────────────┴───────────────┴───────────┘
```

---

## 🤝 Contributing

We demand excellence. Code must be performant, safe, and documented.

```bash
# Clone & Setup
git clone https://github.com/reasonkit/reasonkit-core
cd reasonkit-core

# The 5 Gates of Quality
cargo build --release        # Gate 1: Compilation
cargo clippy -- -D warnings  # Gate 2: Linting
cargo fmt --check            # Gate 3: Formatting
cargo test --all-features    # Gate 4: Testing
cargo bench                  # Gate 5: Performance
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed protocols.

---

## 📜 License

Apache 2.0 - See [LICENSE](LICENSE)

---

<div align="center">

**ReasonKit** - Turn Prompts into Protocols
*Designed, Not Dreamed*

[Website](https://reasonkit.sh) | [Docs](https://docs.reasonkit.sh) | [GitHub](https://github.com/reasonkit/reasonkit-core)

</div>
