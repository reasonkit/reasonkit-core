# Anthropic (Claude) Integration Guide

> ReasonKit + Claude: The Best-in-Class Reasoning Stack
> "Turn Prompts into Protocols with Claude's Extended Thinking"

**Provider:** Anthropic
**Models:** Claude Opus 4.5, Claude Sonnet 4, Claude Haiku
**Best For:** Complex reasoning, extended thinking, safety-critical analysis

---

## Quick Start (5 Lines)

```bash
# 1. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Install ReasonKit
cargo install reasonkit-core

# 3. Run analysis
rk think --provider anthropic "Should we adopt microservices?"
```

---

## Environment Setup

### API Key Configuration

```bash
# Option 1: Environment variable (recommended)
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Option 2: In ~/.reasonkit/config.toml
[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

# Option 3: Direct (not recommended for security)
[providers.anthropic]
api_key = "sk-ant-..."
```

### Verify Setup

```bash
# Test API connection
rk think --provider anthropic --model claude-haiku "Hello" --mock false

# Check configuration
rk config show | grep anthropic
```

---

## Full Configuration Options

### ~/.ReasonKit/config.toml

```toml
[providers.anthropic]
# API Configuration
api_key_env = "ANTHROPIC_API_KEY"
api_base_url = "https://api.anthropic.com"      # Optional: for proxies
api_version = "2024-01-01"                       # API version

# Default Model Settings
default_model = "claude-sonnet-4"
temperature = 0.7
max_tokens = 4096
top_p = 0.95

# Extended Thinking (Claude Opus 4.5 / Sonnet 4)
enable_thinking = true
thinking_budget = 16000                          # Max thinking tokens

# Safety Settings
stop_sequences = ["</output>", "HALT"]

# Rate Limiting
requests_per_minute = 50
retry_attempts = 3
retry_delay_ms = 1000
```

### Available Models

| Model               | ID                         | Context | Best For                         | Cost/1M tokens       |
| ------------------- | -------------------------- | ------- | -------------------------------- | -------------------- |
| **Claude Opus 4.5** | `claude-opus-4-20250514`   | 200K    | Deep research, complex reasoning | $15 in / $75 out     |
| **Claude Sonnet 4** | `claude-sonnet-4-20250514` | 200K    | Balanced: speed + quality        | $3 in / $15 out      |
| **Claude Haiku**    | `claude-3-5-haiku-latest`  | 200K    | Fast, cost-effective             | $0.25 in / $1.25 out |

---

## ThinkTool Usage Examples

### Single Protocol Execution

```bash
# GigaThink: Creative expansion with Claude Sonnet
rk think "What are novel approaches to database scaling?" \
  --provider anthropic \
  --model claude-sonnet-4 \
  --protocol gigathink

# LaserLogic: Logical validation with Haiku (fast)
rk think "Validate this argument: If A then B, A is true, therefore B" \
  --provider anthropic \
  --model claude-3-5-haiku-latest \
  --protocol laserlogic

# ProofGuard: Evidence verification with Opus (thorough)
rk think "Verify: Rust is memory-safe by default" \
  --provider anthropic \
  --model claude-opus-4 \
  --protocol proofguard

# BrutalHonesty: Adversarial critique
rk think "Critique this startup idea: AI-powered pet food delivery" \
  --provider anthropic \
  --protocol brutalhonesty
```

### Profile-Based Execution

```bash
# Quick analysis (GigaThink + LaserLogic)
rk think "Is GraphQL better than REST?" \
  --provider anthropic \
  --profile quick

# Balanced analysis (all 5 modules, sequential)
rk think "Should we rewrite in Rust?" \
  --provider anthropic \
  --profile balanced

# Deep analysis with Opus (maximum rigor)
rk think "Evaluate this security architecture" \
  --provider anthropic \
  --model claude-opus-4 \
  --profile deep

# Paranoid mode (multi-pass verification)
rk think "Is this smart contract safe to deploy?" \
  --provider anthropic \
  --model claude-opus-4 \
  --profile paranoid \
  --save-trace
```

### Extended Thinking (Claude Opus 4.5 / Sonnet 4)

```bash
# Enable extended thinking for complex problems
rk think "Design a distributed consensus algorithm" \
  --provider anthropic \
  --model claude-opus-4 \
  --thinking \
  --thinking-budget 16000

# View thinking process in trace
rk think "Prove P = NP is unlikely" \
  --provider anthropic \
  --profile paranoid \
  --thinking \
  --save-trace \
  --trace-dir ./traces
```

---

## Rust API Integration

```rust
use reasonkit_core::providers::AnthropicProvider;
use reasonkit_core::thinktool::{ThinkToolOrchestrator, ReasoningProfile};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize provider
    let provider = AnthropicProvider::new()
        .model("claude-sonnet-4")
        .temperature(0.7)
        .max_tokens(4096)
        .enable_thinking(true)
        .thinking_budget(16000)
        .build()?;

    // Create orchestrator with Anthropic
    let orchestrator = ThinkToolOrchestrator::with_provider(provider);

    // Execute with profile
    let result = orchestrator
        .think("Analyze this architecture decision")
        .profile(ReasoningProfile::Balanced)
        .execute()
        .await?;

    println!("Confidence: {:.1}%", result.confidence.overall * 100.0);
    println!("Result: {}", result.output);

    // Access thinking trace (if enabled)
    if let Some(thinking) = &result.thinking_trace {
        println!("Thinking: {}", thinking);
    }

    Ok(())
}
```

---

## Python API Integration

```python
from reasonkit import ThinkToolOrchestrator, ReasoningProfile
from reasonkit.providers import AnthropicProvider

# Initialize provider
provider = AnthropicProvider(
    model="claude-sonnet-4",
    temperature=0.7,
    max_tokens=4096,
    enable_thinking=True,
    thinking_budget=16000
)

# Create orchestrator
orchestrator = ThinkToolOrchestrator(provider=provider)

# Execute analysis
result = orchestrator.think(
    query="Should we adopt event sourcing?",
    profile=ReasoningProfile.BALANCED
)

print(f"Confidence: {result.confidence.overall:.1%}")
print(f"Result: {result.output}")

# Access extended thinking
if result.thinking_trace:
    print(f"Thinking: {result.thinking_trace[:500]}...")
```

---

## Claude CLI Integration

ReasonKit integrates with the official Claude CLI (`claude`) for AI-to-AI consultation:

```bash
# One-shot consultation via Claude CLI
claude -p "Review this ReasonKit analysis for blind spots: $(cat analysis.json)"

# Pipe ReasonKit output to Claude for critique
rk think "Design a caching strategy" --format json | \
  claude -p "Find flaws in this reasoning chain"

# Use Claude CLI for quick checks
claude -p "Is this Rust code safe?" --allowedTools "Read" < src/lib.rs
```

---

## Cost Estimation

### Per-Query Cost Calculator

| Profile           | Model  | Avg Tokens | Estimated Cost |
| ----------------- | ------ | ---------- | -------------- |
| `--quick`         | Haiku  | ~2,000     | $0.003         |
| `--quick`         | Sonnet | ~2,000     | $0.036         |
| `--balanced`      | Haiku  | ~5,000     | $0.008         |
| `--balanced`      | Sonnet | ~5,000     | $0.090         |
| `--deep`          | Sonnet | ~10,000    | $0.180         |
| `--deep`          | Opus   | ~10,000    | $0.900         |
| `--paranoid`      | Opus   | ~20,000    | $1.800         |
| Extended Thinking | Opus   | ~50,000    | $4.500         |

### Monthly Budget Examples

```bash
# Developer: 100 balanced queries/day with Sonnet
# 100 * 30 * $0.09 = $270/month

# Team: 500 mixed queries/day
# Quick (Haiku):    200 * 30 * $0.003 = $18
# Balanced (Sonnet): 250 * 30 * $0.09  = $675
# Deep (Sonnet):     50 * 30 * $0.18  = $270
# Total: ~$963/month

# Enterprise: Heavy usage with Opus
# Use budget controls to cap costs
rk think "Complex analysis" \
  --provider anthropic \
  --model claude-opus-4 \
  --budget "$5.00"
```

---

## Troubleshooting

### Common Issues

#### 1. "Authentication error: Invalid API key"

```bash
# Verify key is set
echo $ANTHROPIC_API_KEY | head -c 20

# Check key format (should start with sk-ant-)
# Get new key from: https://console.anthropic.com/

# Test directly
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -H "anthropic-version: 2024-01-01" \
  -d '{"model":"claude-3-5-haiku-latest","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

#### 2. "Rate limit exceeded"

```bash
# Add rate limiting to config
[providers.anthropic]
requests_per_minute = 30
retry_attempts = 5
retry_delay_ms = 2000

# Or use budget-based throttling
rk think "Query" --budget "10t/min"
```

#### 3. "Context length exceeded"

```bash
# Use a model with larger context
rk think "Long document analysis" \
  --provider anthropic \
  --model claude-sonnet-4  # 200K context

# Or truncate input
rk think "Summary of large doc" \
  --max-input-tokens 50000
```

#### 4. "Extended thinking not supported"

```bash
# Only Opus 4.5 and Sonnet 4 support extended thinking
# Use correct model
rk think "Complex problem" \
  --provider anthropic \
  --model claude-opus-4 \
  --thinking

# Check model supports thinking
rk providers show anthropic
```

#### 5. "Output truncated"

```bash
# Increase max_tokens
rk think "Long analysis" \
  --provider anthropic \
  --max-tokens 8192

# Or in config
[providers.anthropic]
max_tokens = 8192
```

---

## Best Practices

### Model Selection Strategy

```
Use Case                    Recommended Model
---------------------------------------------
Quick exploration           claude-3-5-haiku-latest
Daily development           claude-sonnet-4
Complex reasoning           claude-opus-4
Extended thinking           claude-opus-4 + --thinking
Cost-sensitive batch        claude-3-5-haiku-latest
Safety-critical             claude-opus-4 + --paranoid
```

### Optimal Configuration

```toml
# Production configuration for Anthropic
[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

# Model tiers based on task
[providers.anthropic.models]
fast = "claude-3-5-haiku-latest"
balanced = "claude-sonnet-4"
deep = "claude-opus-4"

# Auto-select based on profile
[providers.anthropic.profile_mapping]
quick = "fast"
balanced = "balanced"
deep = "deep"
paranoid = "deep"
```

### Cost Control

```bash
# Set hard budget limit
rk think "Expensive query" \
  --provider anthropic \
  --budget "$1.00"

# Use Haiku for exploration, upgrade for final
rk think "Draft analysis" --model claude-3-5-haiku-latest
rk think "Final analysis" --model claude-sonnet-4 --profile deep
```

---

## Security Considerations

1. **Never commit API keys** - Use environment variables
2. **Enable audit logging** - Track all API calls
3. **Use Claude's safety features** - Built-in content filtering
4. **Implement rate limiting** - Prevent runaway costs
5. **Review thinking traces** - Verify reasoning quality

```bash
# Enable full audit trail
[logging]
level = "info"
log_api_calls = true
log_file = "~/.reasonkit/logs/anthropic.log"
```

---

## Resources

- **Anthropic Console:** <https://console.anthropic.com/>
- **API Documentation:** <https://docs.anthropic.com/>
- **Claude CLI:** <https://github.com/anthropics/claude-code>
- **Pricing:** <https://www.anthropic.com/pricing>
- **Model Card:** <https://docs.anthropic.com/claude/docs/models-overview>

---

_ReasonKit + Anthropic Integration Guide | v1.0.0 | Apache 2.0_
_"See How Your AI Thinks"_
