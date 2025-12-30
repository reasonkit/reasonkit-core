# ReasonKit v1.0.0 Release Notes

> **"From Prompt to Protocol"**

We are proud to announce the first stable release of ReasonKit Core, marking the shift from experimental AI to engineered cognitive pipelines.

## 🚀 Key Features

### 1. Config-Driven Architecture (New!)

ReasonKit is no longer just a Rust binary; it's a configurable engine.

- **YAML Protocols:** Define how the AI thinks in `protocols/thinktools_v2.yaml`.
- **Custom Profiles:** create your own reasoning chains in `protocols/profiles_v2.yaml`.
- **Hot-Reload:** Change a protocol description, see it update instantly.

### 2. The 5 Core ThinkTools

The engine now ships with 5 stabilized cognitive modules:

- **GigaThink (`gt`):** Divergent thinking (10+ perspectives).
- **LaserLogic (`ll`):** Deductive reasoning & fallacy detection.
- **BedRock (`br`):** First principles decomposition.
- **ProofGuard (`pg`):** Triangulated verification.
- **BrutalHonesty (`bh`):** Adversarial critique.

### 3. Glass Box Integration (LangChain)

Stop building black-box agents.

- New `examples/rk_langchain_agent.py` demonstrates how to use ReasonKit as a verifying backend for Python agents.
- Agents can "offload" hard reasoning to ReasonKit and get structured, auditable JSON back.

### 4. The "Reasoning Arena" Benchmark

We are proposing a new community standard for AI integrity in `BENCHMARKS.md`.

- Moved beyond "vibe checks" to structural integrity metrics.
- 5 standard "Hard Reasoning" prompts to test depth, breadth, and grounding.

---

## 🛠️ Fixes & Improvements

- **Performance:** All core reasoning loops validated at <5ms latency (excluding LLM inference).
- **Stability:** Resolved JSON parsing issues in the Python integration.
- **PowerCombo:** Marked as `experimental` while we refine the composite logic.

## 📦 Installation

```bash
cargo install reasonkit-core
# or build from source
git clone https://github.com/reasonkit/reasonkit-core
cd reasonkit-core
cargo build --release
```

## 🔗 Links

- [Documentation](https://docs.reasonkit.sh)
- [Reasoning Arena](BENCHMARKS.md)
- [Glass Box Concepts](CONCEPTS.md)
