# ReasonKit Developer Quickstart

> Get structured AI reasoning in 2 minutes

---

## 30-Second Install

Choose your method:

```bash
# Option A: Universal installer (recommended)
curl -fsSL https://reasonkit.sh/install | bash

# Option B: Cargo (Rust developers)
cargo install reasonkit-core

# Option C: Build from source
git clone https://github.com/reasonkit/reasonkit-core && cd reasonkit-core && cargo build --release
```

Verify installation:

```bash
rk-core --version
```

---

## First Query (Copy-Paste Ready)

```bash
# Set your API key (Anthropic, OpenAI, or any of 18+ providers)
export ANTHROPIC_API_KEY="sk-ant-..."

# Run your first structured reasoning query
rk-core think "Should I use microservices or a monolith?" --profile balanced
```

**Expected output:**

```
Protocol: balanced (GigaThink -> LaserLogic -> BedRock -> ProofGuard)
Model: claude-sonnet-4

Step 1/4: Expansive Analysis...     [2.1s]
Step 2/4: Logical Validation...     [1.8s]
Step 3/4: First Principles...       [2.4s]
Step 4/4: Verification...           [1.9s]

RESULT (Confidence: 82%)

10 PERSPECTIVES ANALYZED:
1. Scale: Microservices for >50 engineers, monolith for <20
2. Complexity: Monolith 3x faster to iterate initially
3. Deployment: Microservices need DevOps maturity
...

FIRST PRINCIPLES:
- Core need: Ship features fast
- Hidden assumption: "Scale" isn't your bottleneck yet
- Axiom: Premature distribution is premature optimization

VERDICT: Start monolith, extract services when pain is real.

Tokens: 2,847 | Cost: $0.032 | Time: 8.2s
```

---

## Understanding the Output

| Section              | What It Means                                             |
| -------------------- | --------------------------------------------------------- |
| **Protocol**         | Which ThinkTools ran (e.g., `balanced` = 4 modules)       |
| **Confidence**       | How certain the analysis is (70-95% depending on profile) |
| **Perspectives**     | Different angles you might have missed (from GigaThink)   |
| **First Principles** | Core axioms and hidden assumptions (from BedRock)         |
| **Verdict**          | Synthesized recommendation after all analysis             |
| **Tokens/Cost**      | LLM usage for transparency                                |

---

## Profile Selection Guide

| Profile      | When to Use                             | Time   | Confidence |
| ------------ | --------------------------------------- | ------ | ---------- |
| `--quick`    | Daily decisions, sanity checks          | ~30s   | 70%        |
| `--balanced` | Important choices, need multiple angles | ~2min  | 80%        |
| `--deep`     | Major decisions, thorough analysis      | ~5min  | 85%        |
| `--paranoid` | High-stakes, cannot afford mistakes     | ~10min | 95%        |

### Quick Reference

```bash
# Fast sanity check
rk-core think "Is this approach reasonable?" --profile quick

# Standard analysis (default)
rk-core think "Evaluate this technical decision" --profile balanced

# Thorough deep-dive
rk-core think "Should we pivot our product strategy?" --profile deep

# Maximum rigor (adversarial critique included)
rk-core think "Is this production-ready?" --profile paranoid
```

---

## ThinkTools Explained

Each ThinkTool catches a specific blind spot:

| Tool              | Icon | What It Catches                      | Shortcut             |
| ----------------- | ---- | ------------------------------------ | -------------------- |
| **GigaThink**     | `gt` | Angles you missed (10+ perspectives) | Divergent thinking   |
| **LaserLogic**    | `ll` | Flawed reasoning, fallacies          | Convergent analysis  |
| **BedRock**       | `br` | Overcomplicated answers              | First principles     |
| **ProofGuard**    | `pg` | Unverified claims                    | Source triangulation |
| **BrutalHonesty** | `bh` | Your blind spots                     | Adversarial critique |

### Run Individual Tools

```bash
# Just want perspectives?
rk-core think "Evaluate X" --protocol gigathink

# Check logic only?
rk-core think "Is this argument valid?" --protocol laserlogic

# First principles?
rk-core think "What really matters here?" --protocol bedrock
```

---

## Integration Examples

### Python

```python
import subprocess
import json

def reason(query: str, profile: str = "balanced") -> dict:
    """Execute structured reasoning via CLI."""
    result = subprocess.run(
        ["rk-core", "think", query, "--profile", profile, "--format", "json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

# Usage
analysis = reason("Should we adopt GraphQL?")
print(f"Confidence: {analysis['confidence']}")
print(f"Verdict: {analysis['verdict']}")
```

### Node.js

```javascript
const { execSync } = require("child_process");

function reason(query, profile = "balanced") {
  const result = execSync(
    `rk-core think "${query}" --profile ${profile} --format json`,
    { encoding: "utf-8" },
  );
  return JSON.parse(result);
}

// Usage
const analysis = reason("Is this API design correct?");
console.log(`Confidence: ${analysis.confidence}`);
console.log(`Verdict: ${analysis.verdict}`);
```

### Shell Script

```bash
#!/bin/bash
# analyze.sh - Quick analysis wrapper

QUERY="$1"
PROFILE="${2:-balanced}"

rk-core think "$QUERY" --profile "$PROFILE" --save-trace

# Check exit code
if [ $? -eq 0 ]; then
    echo "Analysis complete. Trace saved to ./traces/"
else
    echo "Analysis failed."
    exit 1
fi
```

### HTTP API

Start the server:

```bash
rk-core serve --host 0.0.0.0 --port 8080
```

Query via curl:

```bash
curl -X POST http://localhost:8080/think \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Evaluate this architecture",
    "profile": "balanced"
  }'
```

---

## LLM Provider Configuration

ReasonKit supports 18+ providers. Set your preferred provider:

### Major Providers

```bash
# Anthropic Claude (default)
export ANTHROPIC_API_KEY="sk-ant-..."
rk-core think "query" --provider anthropic

# OpenAI GPT-4
export OPENAI_API_KEY="sk-..."
rk-core think "query" --provider openai --model gpt-4-turbo

# Google Gemini
export GEMINI_API_KEY="..."
rk-core think "query" --provider gemini

# Groq (ultra-fast, 10x speed)
export GROQ_API_KEY="gsk_..."
rk-core think "query" --provider groq
```

### OpenRouter (300+ Models)

Access any model through one API:

```bash
export OPENROUTER_API_KEY="sk-or-..."

# Use Claude via OpenRouter
rk-core think "query" --provider openrouter --model anthropic/claude-opus-4

# Use Llama 3
rk-core think "query" --provider openrouter --model meta-llama/llama-3-70b
```

---

## Troubleshooting

### "Command not found: rk-core"

```bash
# Check if installed
which rk-core

# If using cargo install, ensure ~/.cargo/bin is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Add to shell profile
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
```

### "API key not found"

```bash
# Verify key is set
echo $ANTHROPIC_API_KEY

# Set temporarily
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use config file
mkdir -p ~/.reasonkit
cat > ~/.reasonkit/config.toml << 'EOF'
[thinktool]
default_provider = "anthropic"
EOF
```

### "Rate limit exceeded"

```bash
# Use budget limits to control usage
rk-core think "query" --profile balanced --budget "30s"
rk-core think "query" --profile balanced --budget "$0.25"

# Switch to cheaper/faster provider
rk-core think "query" --provider groq
```

### "Timeout during analysis"

```bash
# Increase timeout
rk-core think "complex query" --profile deep --budget "5m"

# Use faster profile
rk-core think "complex query" --profile quick
```

### Output Not JSON-Parseable

```bash
# Use explicit JSON format flag
rk-core think "query" --profile balanced --format json
```

---

## Next Steps

| Goal                  | Resource                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Full CLI options      | [`docs/CLI_REFERENCE.md`](/home/zyxsys/RK-PROJECT/reasonkit-core/docs/CLI_REFERENCE.md)                           |
| API documentation     | [`docs/API_REFERENCE.md`](/home/zyxsys/RK-PROJECT/reasonkit-core/docs/API_REFERENCE.md)                           |
| ThinkTool deep dive   | [`docs/THINKTOOLS_QUICK_REFERENCE.md`](/home/zyxsys/RK-PROJECT/reasonkit-core/docs/THINKTOOLS_QUICK_REFERENCE.md) |
| Architecture overview | [`ARCHITECTURE.md`](/home/zyxsys/RK-PROJECT/reasonkit-core/ARCHITECTURE.md)                                       |
| Real-world examples   | [`docs/USE_CASES.md`](/home/zyxsys/RK-PROJECT/reasonkit-core/docs/USE_CASES.md)                                   |

---

## Quick Reference Card

```
INSTALL:     curl -fsSL https://reasonkit.sh/install | bash
VERIFY:      rk-core --version
API KEY:     export ANTHROPIC_API_KEY="sk-ant-..."

PROFILES:
  --quick     Fast (30s, 70% confidence)
  --balanced  Standard (2min, 80% confidence)
  --deep      Thorough (5min, 85% confidence)
  --paranoid  Maximum (10min, 95% confidence)

TOOLS:
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
  --save-trace              Save execution trace

BUDGET:
  --budget "30s"            Time limit
  --budget "$0.50"          Cost limit
  --budget "1000t"          Token limit
```

---

**Version:** 0.1.0 | **License:** Apache 2.0 | **Website:** [reasonkit.sh](https://reasonkit.sh)
