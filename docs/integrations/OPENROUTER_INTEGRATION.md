# OpenRouter Integration Guide

> ReasonKit + OpenRouter: Access 300+ Models Through One API
> "Turn Prompts into Protocols with Any Model You Need"

**Provider:** OpenRouter
**Models:** 300+ models from Anthropic, OpenAI, Google, Meta, Mistral, and more
**Best For:** Model flexibility, cost optimization, fallback strategies, multi-model workflows

---

## Quick Start (5 Lines)

```bash
# 1. Set API key
export OPENROUTER_API_KEY="sk-or-..."

# 2. Install ReasonKit
cargo install reasonkit-core

# 3. Run analysis with any model
rk think --provider openrouter --model anthropic/claude-sonnet-4 "Analyze this"
```

---

## Environment Setup

### API Key Configuration

```bash
# Option 1: Environment variable (recommended)
export OPENROUTER_API_KEY="sk-or-..."
# Get key from: https://openrouter.ai/keys

# Option 2: In ~/.reasonkit/config.toml
[providers.openrouter]
api_key_env = "OPENROUTER_API_KEY"

# Option 3: With site identification (for analytics)
export OPENROUTER_SITE_URL="https://reasonkit.sh"
export OPENROUTER_SITE_NAME="ReasonKit"
```

### Verify Setup

```bash
# Test API connection
rk think --provider openrouter --model meta-llama/llama-3.3-70b-instruct "Hello"

# List available models
rk providers show openrouter

# Search for specific models
rk providers search openrouter "claude"
```

---

## Full Configuration Options

### ~/.ReasonKit/config.toml

```toml
[providers.openrouter]
# API Configuration
api_key_env = "OPENROUTER_API_KEY"
api_base_url = "https://openrouter.ai/api/v1"

# Site Identification (for analytics dashboard)
site_url = "https://reasonkit.sh"
site_name = "ReasonKit"

# Default Model Settings
default_model = "anthropic/claude-sonnet-4"
temperature = 0.7
max_tokens = 4096
top_p = 0.95

# Routing Options
route = "fallback"                    # fallback, lowest-latency, lowest-cost
fallback_models = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.0-flash"
]

# Cost Control
max_cost_per_request = 1.00          # USD
require_credits = true                # Fail if no credits

# Rate Limiting
requests_per_minute = 100
retry_attempts = 3
retry_delay_ms = 1000
```

### Model Naming Convention

OpenRouter uses `provider/model-name` format:

```bash
# Anthropic Models
anthropic/claude-opus-4
anthropic/claude-sonnet-4
anthropic/claude-3-5-haiku

# OpenAI Models
openai/gpt-4o
openai/gpt-4o-mini
openai/o1
openai/o1-mini

# Google Models
google/gemini-2.0-flash
google/gemini-2.0-pro
google/gemini-1.5-pro

# Meta Models (FREE!)
meta-llama/llama-3.3-70b-instruct
meta-llama/llama-3.1-405b-instruct

# Mistral Models
mistralai/mistral-large-2
mistralai/mistral-small-3
mistralai/codestral-latest

# DeepSeek Models
deepseek/deepseek-chat
deepseek/deepseek-r1

# And 300+ more...
```

---

## ThinkTool Usage Examples

### Single Protocol Execution

```bash
# GigaThink with Claude
rk think "Generate creative solutions" \
  --provider openrouter \
  --model anthropic/claude-sonnet-4 \
  --protocol gigathink

# LaserLogic with GPT-4
rk think "Validate this argument" \
  --provider openrouter \
  --model openai/gpt-4o \
  --protocol laserlogic

# ProofGuard with Gemini
rk think "Verify these claims" \
  --provider openrouter \
  --model google/gemini-2.0-flash \
  --protocol proofguard

# BedRock with DeepSeek R1
rk think "First principles analysis" \
  --provider openrouter \
  --model deepseek/deepseek-r1 \
  --protocol bedrock

# BrutalHonesty with Llama
rk think "Critique this plan" \
  --provider openrouter \
  --model meta-llama/llama-3.3-70b-instruct \
  --protocol brutalhonesty
```

### Profile-Based Execution

```bash
# Quick analysis with fast model
rk think "Quick review" \
  --provider openrouter \
  --model google/gemini-2.0-flash \
  --profile quick

# Balanced with Claude Sonnet
rk think "Evaluate this design" \
  --provider openrouter \
  --model anthropic/claude-sonnet-4 \
  --profile balanced

# Deep analysis with Claude Opus
rk think "Comprehensive analysis" \
  --provider openrouter \
  --model anthropic/claude-opus-4 \
  --profile deep

# Paranoid with o1
rk think "Security audit" \
  --provider openrouter \
  --model openai/o1 \
  --profile paranoid
```

### Free Models (Great for Development!)

```bash
# Meta Llama - FREE
rk think "Analyze this code" \
  --provider openrouter \
  --model meta-llama/llama-3.3-70b-instruct:free

# Google Gemma - FREE
rk think "Quick check" \
  --provider openrouter \
  --model google/gemma-2-9b-it:free

# Mistral - FREE
rk think "Code review" \
  --provider openrouter \
  --model mistralai/devstral-small:free

# Use free models for development, paid for production
```

### Auto-Routing (Let OpenRouter Choose)

```bash
# Lowest cost route
rk think "Analyze this" \
  --provider openrouter \
  --route lowest-cost

# Lowest latency route
rk think "Quick response needed" \
  --provider openrouter \
  --route lowest-latency

# Fallback chain (try each until success)
rk think "Critical analysis" \
  --provider openrouter \
  --route fallback \
  --fallback "anthropic/claude-sonnet-4,openai/gpt-4o,google/gemini-2.0-flash"
```

### Multi-Model Comparison

```bash
# Compare reasoning across models
rk compare "Should we use microservices?" \
  --provider openrouter \
  --models "anthropic/claude-sonnet-4,openai/gpt-4o,google/gemini-2.0-flash" \
  --profile balanced

# Consensus from multiple models
rk consensus "Is this architecture sound?" \
  --provider openrouter \
  --models "anthropic/claude-opus-4,openai/o1,deepseek/deepseek-r1" \
  --threshold 0.8
```

---

## Rust API Integration

```rust
use reasonkit_core::providers::OpenRouterProvider;
use reasonkit_core::thinktool::{ThinkToolOrchestrator, ReasoningProfile};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize OpenRouter with specific model
    let provider = OpenRouterProvider::new()
        .model("anthropic/claude-sonnet-4")
        .temperature(0.7)
        .max_tokens(4096)
        .site_url("https://reasonkit.sh")
        .site_name("ReasonKit")
        .build()?;

    // Create orchestrator
    let orchestrator = ThinkToolOrchestrator::with_provider(provider);

    // Execute with profile
    let result = orchestrator
        .think("Analyze this architecture")
        .profile(ReasoningProfile::Balanced)
        .execute()
        .await?;

    println!("Model: {}", result.model_used);
    println!("Confidence: {:.1}%", result.confidence.overall * 100.0);
    println!("Result: {}", result.output);

    Ok(())
}
```

### Multi-Model Workflow

```rust
use reasonkit_core::providers::OpenRouterProvider;
use reasonkit_core::thinktool::ThinkToolOrchestrator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create providers for different models
    let claude = OpenRouterProvider::new()
        .model("anthropic/claude-sonnet-4")
        .build()?;

    let gpt = OpenRouterProvider::new()
        .model("openai/gpt-4o")
        .build()?;

    let gemini = OpenRouterProvider::new()
        .model("google/gemini-2.0-flash")
        .build()?;

    // Compare across models
    let query = "Should we adopt event sourcing?";

    let results = futures::join!(
        ThinkToolOrchestrator::with_provider(claude).think(query).execute(),
        ThinkToolOrchestrator::with_provider(gpt).think(query).execute(),
        ThinkToolOrchestrator::with_provider(gemini).think(query).execute(),
    );

    // Synthesize consensus
    // ...

    Ok(())
}
```

---

## Python API Integration

```python
from reasonkit import ThinkToolOrchestrator, ReasoningProfile
from reasonkit.providers import OpenRouterProvider

# Initialize OpenRouter
provider = OpenRouterProvider(
    model="anthropic/claude-sonnet-4",
    temperature=0.7,
    max_tokens=4096,
    site_url="https://reasonkit.sh",
    site_name="ReasonKit"
)

# Create orchestrator
orchestrator = ThinkToolOrchestrator(provider=provider)

# Execute analysis
result = orchestrator.think(
    query="Evaluate this business model",
    profile=ReasoningProfile.BALANCED
)

print(f"Model: {result.model_used}")
print(f"Confidence: {result.confidence.overall:.1%}")
print(f"Result: {result.output}")
```

### Multi-Model Comparison

```python
import asyncio
from reasonkit import ThinkToolOrchestrator
from reasonkit.providers import OpenRouterProvider

async def compare_models(query, models):
    results = {}

    for model in models:
        provider = OpenRouterProvider(model=model)
        orchestrator = ThinkToolOrchestrator(provider=provider)
        result = await orchestrator.think_async(query)
        results[model] = result

    return results

# Compare top models
models = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.0-flash",
    "meta-llama/llama-3.3-70b-instruct"
]

results = asyncio.run(compare_models(
    "Should we use Kubernetes or Docker Swarm?",
    models
))

for model, result in results.items():
    print(f"{model}: {result.confidence.overall:.1%} confidence")
```

---

## Cost Estimation

### Per-Query Cost by Model

| Model              | Context | Cost/1M in | Cost/1M out | Balanced Query |
| ------------------ | ------- | ---------- | ----------- | -------------- |
| claude-opus-4      | 200K    | $15.00     | $75.00      | $0.90          |
| claude-sonnet-4    | 200K    | $3.00      | $15.00      | $0.09          |
| gpt-4o             | 128K    | $2.50      | $10.00      | $0.063         |
| gpt-4o-mini        | 128K    | $0.15      | $0.60       | $0.004         |
| gemini-2.0-flash   | 1M      | $0.075     | $0.30       | $0.002         |
| gemini-1.5-pro     | 2M      | $1.25      | $5.00       | $0.031         |
| llama-3.3-70b:free | 128K    | FREE       | FREE        | FREE           |
| deepseek-r1        | 128K    | $0.55      | $2.19       | $0.014         |

### Cost Optimization Strategies

```bash
# Strategy 1: Use free models for development
rk think "Test query" \
  --provider openrouter \
  --model meta-llama/llama-3.3-70b-instruct:free

# Strategy 2: Use lowest-cost routing
rk think "Analysis" \
  --provider openrouter \
  --route lowest-cost

# Strategy 3: Budget caps
rk think "Expensive analysis" \
  --provider openrouter \
  --budget "$0.50"

# Strategy 4: Tiered model selection
rk think "Draft" --model meta-llama/llama-3.3-70b-instruct:free
rk think "Final" --model anthropic/claude-sonnet-4
```

### Monthly Budget Examples

```bash
# Developer (cost-optimized):
# 50% free models, 30% Gemini Flash, 20% Claude Sonnet
# 500 queries/day
# Free: 250 * 30 * $0 = $0
# Flash: 150 * 30 * $0.002 = $9
# Sonnet: 100 * 30 * $0.09 = $270
# Total: ~$279/month

# Team (balanced):
# Mix of models based on task
# 1000 queries/day
# Total: ~$800-1500/month depending on model mix

# Enterprise (quality-first):
# Mostly premium models
# 2000 queries/day
# Total: ~$3000-5000/month
```

---

## Troubleshooting

### Common Issues

#### 1. "Invalid API key"

```bash
# Verify key is set
echo $OPENROUTER_API_KEY | head -c 15

# Check key format (should start with sk-or-)
# Get new key from: https://openrouter.ai/keys

# Test directly
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

#### 2. "Model not found"

```bash
# Check exact model name
rk providers search openrouter "claude"

# Model names are case-sensitive and require provider prefix
# Wrong: claude-sonnet-4
# Right: anthropic/claude-sonnet-4

# Some models require credits
# Add credits at: https://openrouter.ai/credits
```

#### 3. "Insufficient credits"

```bash
# Check credit balance
curl https://openrouter.ai/api/v1/auth/key \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"

# Add credits at: https://openrouter.ai/credits

# Or use free models
rk think "Query" \
  --provider openrouter \
  --model meta-llama/llama-3.3-70b-instruct:free
```

#### 4. "Rate limit exceeded"

```bash
# OpenRouter has per-model rate limits
# Add delays
[providers.openrouter]
requests_per_minute = 30
retry_delay_ms = 2000

# Or use fallback routing
rk think "Query" \
  --route fallback \
  --fallback "anthropic/claude-sonnet-4,openai/gpt-4o"
```

#### 5. "Model temporarily unavailable"

```bash
# Some models have limited availability
# Use fallback routing for reliability
[providers.openrouter]
route = "fallback"
fallback_models = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.0-flash"
]

# Or specify alternatives
rk think "Query" \
  --provider openrouter \
  --model anthropic/claude-sonnet-4 \
  --fallback openai/gpt-4o
```

#### 6. "Response slower than expected"

```bash
# Use lowest-latency routing
rk think "Time-sensitive query" \
  --provider openrouter \
  --route lowest-latency

# Or choose known-fast models
# Fast: Gemini Flash, GPT-4o-mini, Llama instruct
# Slower: o1, Claude Opus, 405B models
```

---

## Best Practices

### Model Selection Strategy

```
Use Case                    Recommended Model
---------------------------------------------
Development/testing         meta-llama/llama-3.3-70b-instruct:free
Quick analysis             google/gemini-2.0-flash
Balanced quality           anthropic/claude-sonnet-4
Complex reasoning          openai/o1 or deepseek/deepseek-r1
Long context              google/gemini-1.5-pro
Maximum quality           anthropic/claude-opus-4
Code generation           mistralai/codestral-latest
Cost-sensitive            openai/gpt-4o-mini
```

### Optimal Configuration

```toml
# Production configuration for OpenRouter
[providers.openrouter]
api_key_env = "OPENROUTER_API_KEY"
site_url = "https://your-app.com"
site_name = "YourApp"

# Default to reliable model
default_model = "anthropic/claude-sonnet-4"

# Fallback for reliability
route = "fallback"
fallback_models = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.0-flash",
    "meta-llama/llama-3.3-70b-instruct"
]

# Cost controls
max_cost_per_request = 1.00

# Profile-based model selection
[providers.openrouter.profile_mapping]
quick = "google/gemini-2.0-flash"
balanced = "anthropic/claude-sonnet-4"
deep = "openai/o1"
paranoid = "anthropic/claude-opus-4"
```

### Multi-Model Workflows

```bash
# Workflow 1: Draft + Refine
rk think "Initial analysis" \
  --provider openrouter \
  --model meta-llama/llama-3.3-70b-instruct:free \
  --profile quick > draft.json

rk think "Refine: $(cat draft.json)" \
  --provider openrouter \
  --model anthropic/claude-sonnet-4 \
  --profile deep

# Workflow 2: Multi-Perspective
for model in "anthropic/claude-sonnet-4" "openai/gpt-4o" "google/gemini-2.0-flash"; do
  rk think "Evaluate this decision" \
    --provider openrouter \
    --model "$model" \
    --profile balanced
done

# Workflow 3: Consensus
rk consensus "Is this architecture sound?" \
  --provider openrouter \
  --models "anthropic/claude-opus-4,openai/o1,deepseek/deepseek-r1" \
  --threshold 0.8
```

---

## OpenRouter vs Direct Provider Access

| Feature          | OpenRouter        | Direct Access         |
| ---------------- | ----------------- | --------------------- |
| Model variety    | 300+ models       | Single provider       |
| Single API key   | Yes               | Multiple keys needed  |
| Fallback routing | Built-in          | Manual implementation |
| Cost tracking    | Unified dashboard | Per-provider          |
| Latency          | +10-50ms overhead | Direct                |
| Free models      | Available         | Provider-specific     |
| Pricing          | Pass-through      | Direct pricing        |

**When to use OpenRouter:**

- Need access to multiple providers
- Want fallback/routing capabilities
- Unified billing preferred
- Testing different models
- Cost optimization needed

**When to use direct access:**

- Maximum performance needed
- Single provider sufficient
- Enterprise contracts in place
- Minimal latency critical

---

## Resources

- **OpenRouter Dashboard:** <https://openrouter.ai/>
- **API Documentation:** <https://openrouter.ai/docs>
- **Model List:** <https://openrouter.ai/models>
- **Pricing:** <https://openrouter.ai/pricing>
- **Credits:** <https://openrouter.ai/credits>
- **Status:** <https://status.openrouter.ai/>

---

*ReasonKit + OpenRouter Integration Guide | v1.0.0 | Apache 2.0*
*"See How Your AI Thinks - With Any Model"*
