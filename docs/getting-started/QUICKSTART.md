# ReasonKit in 5 Minutes

> **Goal:** Working AI reasoning in your terminal. Right now.

---

## What is ReasonKit?

**One sentence:** ReasonKit turns messy LLM prompts into structured, auditable reasoning chains.

**Before ReasonKit:**

```
You: "Should I use microservices?"
LLM: [2000 words of maybe-correct rambling]
```

**After ReasonKit:**

```
You: rk think "Should I use microservices?" --profile balanced
Output: 10 perspectives analyzed, 3 hidden assumptions found, verdict with 82% confidence
```

**Time to understand: 30 seconds.**

---

## Install (60 seconds)

### Option A: One-liner (Recommended)

```bash
curl -fsSL https://reasonkit.sh/install | bash
```

### Option B: Cargo (Rust developers)

```bash
cargo install reasonkit-core
```

### Option C: Build from source

```bash
git clone https://github.com/reasonkit/reasonkit-core && cd reasonkit-core && cargo build --release
```

**Verify it worked:**

```bash
rk --version
# Expected: reasonkit-core 0.1.0
```

**Problem?** Jump to [Troubleshooting](#troubleshooting-30-seconds) below.

---

## First Command (90 seconds)

### Step 1: Set your API key

```bash
# Anthropic Claude (recommended)
export ANTHROPIC_API_KEY="sk-ant-..."

# Or OpenAI
export OPENAI_API_KEY="sk-..."

# Or any of 18+ providers (see docs/integrations/)
```

### Step 2: Run your first analysis

Copy-paste this exact command:

```bash
rk think "Should a startup use microservices or a monolith?" --profile quick
```

### Step 3: See the output

```
Protocol: quick (GigaThink -> LaserLogic)
Model: claude-sonnet-4

[GigaThink] 10 PERSPECTIVES GENERATED
  1. TEAM SIZE: Microservices need 20+ engineers to maintain
  2. DEPLOYMENT: Monolith = 1 deploy, Microservices = N deploys
  3. DEBUGGING: Distributed tracing is hard
  4. ITERATION SPEED: Monolith 3x faster initially
  ...

[LaserLogic] HIDDEN ASSUMPTIONS DETECTED
  ! You're assuming scale is your problem (it isn't)
  ! You're assuming team has DevOps maturity
  ! Logical gap: No evidence microservices solves stated problem

VERDICT: Start with monolith | Confidence: 78% | Time: 1.8s
```

**You now have structured AI reasoning.**

---

## Make It Useful (120 seconds remaining)

### Profile Selection

| Use Case           | Command              | Time   |
| ------------------ | -------------------- | ------ |
| Quick sanity check | `--profile quick`    | ~30s   |
| Important decision | `--profile balanced` | ~2min  |
| Major architecture | `--profile deep`     | ~5min  |
| Production release | `--profile paranoid` | ~10min |

### Real Examples (Copy-Paste Ready)

**Code review:**

```bash
rk think "Review this PR: https://github.com/org/repo/pull/123" --profile balanced
```

**Architecture decision:**

```bash
rk think "GraphQL vs REST for a mobile banking app" --profile deep
```

**Risk assessment:**

```bash
rk think "What could go wrong with this deployment plan?" --profile paranoid
```

**Debug assistance:**

```bash
rk think "Why might a React component re-render infinitely?" --profile quick
```

---

## Bonus: Individual ThinkTools

Each ThinkTool catches a specific blind spot:

```bash
# Generate 10+ perspectives you missed
rk think "Evaluate AI safety" --protocol gigathink

# Find logical fallacies in an argument
rk think "Is this reasoning valid: X therefore Y" --protocol laserlogic

# First principles decomposition
rk think "What really matters for user growth?" --protocol bedrock

# Verify claims with sources
rk think "Is Rust really faster than Go?" --protocol proofguard

# Adversarial self-critique
rk think "Critique my startup idea: X" --protocol brutalhonesty
```

---

## Troubleshooting (30 seconds)

### "command not found: rk"

```bash
# Add cargo bin to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Make it permanent
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc  # or ~/.bashrc
source ~/.zshrc
```

### "API key not found"

```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set it
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "Rate limit exceeded"

```bash
# Use a faster/cheaper provider
rk think "query" --provider groq

# Or set budget limits
rk think "query" --budget "$0.10"
```

### Build errors?

See [Installation & Troubleshooting](INSTALLATION_TROUBLESHOOTING.md) for platform-specific fixes.

---

## What's Next?

| Goal                 | Resource                                               |
| -------------------- | ------------------------------------------------------ |
| Full CLI options     | [CLI Reference](../reference/CLI_REFERENCE.md)         |
| ThinkTool deep dive  | [ThinkTools Guide](../thinktools/THINKTOOLS_GUIDE.md)  |
| Provider setup       | [Integrations](../integrations/README.md)              |
| Real-world use cases | [Use Cases](../process/USE_CASES.md)                   |
| Rust API             | [API Reference](../reference/API_REFERENCE.md)         |

---

## Quick Reference Card

```
INSTALL:     curl -fsSL https://reasonkit.sh/install | bash
VERIFY:      rk --version
API KEY:     export ANTHROPIC_API_KEY="sk-ant-..."

PROFILES:
  --profile quick      Fast (30s, 70% confidence)
  --profile balanced   Standard (2min, 80% confidence)
  --profile deep       Thorough (5min, 85% confidence)
  --profile paranoid   Maximum (10min, 95% confidence)

THINKTOOLS:
  --protocol gigathink      10+ perspectives
  --protocol laserlogic     Logic validation
  --protocol bedrock        First principles
  --protocol proofguard     Source verification
  --protocol brutalhonesty  Adversarial critique

PROVIDERS:
  --provider anthropic      Claude (default)
  --provider openai         GPT-4
  --provider groq           Ultra-fast
  --provider openrouter     300+ models

OUTPUT:
  --format text             Human-readable (default)
  --format json             Machine-parseable
```

---

**Total time: Under 5 minutes.**

**Website:** [reasonkit.sh](https://reasonkit.sh) | **Docs:** [docs.rs/reasonkit-core](https://docs.rs/reasonkit-core)
