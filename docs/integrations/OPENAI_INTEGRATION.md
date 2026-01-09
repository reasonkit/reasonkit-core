# OpenAI (GPT) Integration Guide

> ReasonKit + OpenAI: Structured Reasoning with GPT-4 and o1
> "Turn Prompts into Protocols with Industry-Standard Models"

**Provider:** OpenAI
**Models:** GPT-4o, GPT-4-turbo, o1, o1-mini
**Best For:** General reasoning, JSON mode, function calling, embeddings

---

## Quick Start (5 Lines)

```bash
# 1. Set API key
export OPENAI_API_KEY="sk-..."

# 2. Install ReasonKit
cargo install reasonkit-core

# 3. Run analysis
rk think --provider openai "Evaluate this business strategy"
```

---

## Environment Setup

### API Key Configuration

```bash
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY="sk-proj-..."

# Option 2: In ~/.reasonkit/config.toml
[providers.openai]
api_key_env = "OPENAI_API_KEY"

# Option 3: Organization ID (for enterprise)
export OPENAI_ORG_ID="org-..."
```

### Verify Setup

```bash
# Test API connection
rk think --provider openai --model gpt-4o-mini "Hello" --mock false

# Check available models
rk providers show openai
```

---

## Full Configuration Options

### ~/.ReasonKit/config.toml

```toml
[providers.openai]
# API Configuration
api_key_env = "OPENAI_API_KEY"
organization_id_env = "OPENAI_ORG_ID"    # Optional: for enterprise
api_base_url = "https://api.openai.com/v1"  # Optional: for Azure/proxies

# Default Model Settings
default_model = "gpt-4o"
temperature = 0.7
max_tokens = 4096
top_p = 0.95
frequency_penalty = 0.0
presence_penalty = 0.0

# Response Format
response_format = "text"                  # text, json_object, json_schema
seed = 42                                 # For reproducibility

# Rate Limiting
requests_per_minute = 60
tokens_per_minute = 150000
retry_attempts = 3
retry_delay_ms = 1000
```

### Available Models

| Model           | ID            | Context | Best For             | Cost/1M tokens       |
| --------------- | ------------- | ------- | -------------------- | -------------------- |
| **GPT-4o**      | `gpt-4o`      | 128K    | Balanced performance | $2.50 in / $10 out   |
| **GPT-4o-mini** | `gpt-4o-mini` | 128K    | Fast, cost-effective | $0.15 in / $0.60 out |
| **GPT-4-turbo** | `gpt-4-turbo` | 128K    | Long context, vision | $10 in / $30 out     |
| **o1**          | `o1`          | 200K    | Complex reasoning    | $15 in / $60 out     |
| **o1-mini**     | `o1-mini`     | 128K    | Fast reasoning       | $3 in / $12 out      |

---

## ThinkTool Usage Examples

### Single Protocol Execution

```bash
# GigaThink: Creative expansion with GPT-4o
rk think "Generate product feature ideas" \
  --provider openai \
  --model gpt-4o \
  --protocol gigathink

# LaserLogic: Logical validation with GPT-4o-mini (fast)
rk think "Check this argument for logical fallacies" \
  --provider openai \
  --model gpt-4o-mini \
  --protocol laserlogic

# ProofGuard: Evidence verification
rk think "Verify these performance claims" \
  --provider openai \
  --model gpt-4o \
  --protocol proofguard

# BedRock: First principles with o1 (deep reasoning)
rk think "Break down the fundamentals of distributed systems" \
  --provider openai \
  --model o1 \
  --protocol bedrock

# BrutalHonesty: Adversarial critique
rk think "Critique my marketing strategy" \
  --provider openai \
  --protocol brutalhonesty
```

### Profile-Based Execution

```bash
# Quick analysis (GPT-4o-mini for speed)
rk think "Is this API design good?" \
  --provider openai \
  --model gpt-4o-mini \
  --profile quick

# Balanced analysis (GPT-4o default)
rk think "Evaluate this architecture" \
  --provider openai \
  --profile balanced

# Deep analysis with o1 (maximum reasoning)
rk think "Design a fault-tolerant system" \
  --provider openai \
  --model o1 \
  --profile deep

# Paranoid mode (multi-pass verification)
rk think "Review this financial model" \
  --provider openai \
  --model gpt-4o \
  --profile paranoid \
  --save-trace
```

### JSON Mode (Structured Output)

```bash
# Force JSON output
rk think "Analyze these metrics" \
  --provider openai \
  --model gpt-4o \
  --response-format json_object \
  --format json

# With JSON schema validation
rk think "Extract entities from this text" \
  --provider openai \
  --response-format json_schema \
  --schema '{"type":"object","properties":{"entities":{"type":"array"}}}'
```

### Reasoning Models (o1 Series)

```bash
# o1 for complex multi-step reasoning
rk think "Solve this optimization problem" \
  --provider openai \
  --model o1 \
  --profile deep

# o1-mini for faster reasoning tasks
rk think "Debug this algorithm" \
  --provider openai \
  --model o1-mini \
  --protocol laserlogic

# Note: o1 models have built-in chain-of-thought
# ReasonKit enhances with structured protocols
```

---

## Rust API Integration

```rust
use reasonkit_core::providers::OpenAIProvider;
use reasonkit_core::thinktool::{ThinkToolOrchestrator, ReasoningProfile};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize provider
    let provider = OpenAIProvider::new()
        .model("gpt-4o")
        .temperature(0.7)
        .max_tokens(4096)
        .response_format("json_object")
        .seed(42)  // For reproducibility
        .build()?;

    // Create orchestrator with OpenAI
    let orchestrator = ThinkToolOrchestrator::with_provider(provider);

    // Execute with profile
    let result = orchestrator
        .think("Analyze market trends")
        .profile(ReasoningProfile::Balanced)
        .execute()
        .await?;

    println!("Confidence: {:.1}%", result.confidence.overall * 100.0);
    println!("Result: {}", result.output);

    Ok(())
}
```

---

## Python API Integration

```python
from reasonkit import ThinkToolOrchestrator, ReasoningProfile
from reasonkit.providers import OpenAIProvider

# Initialize provider
provider = OpenAIProvider(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=4096,
    response_format="json_object",
    seed=42
)

# Create orchestrator
orchestrator = ThinkToolOrchestrator(provider=provider)

# Execute analysis
result = orchestrator.think(
    query="Evaluate this investment opportunity",
    profile=ReasoningProfile.BALANCED
)

print(f"Confidence: {result.confidence.overall:.1%}")
print(f"Result: {result.output}")
```

---

## OpenAI Embeddings Integration

ReasonKit uses OpenAI embeddings for the knowledge base:

```bash
# Configure embeddings
[embedding]
provider = "openai"
model = "text-embedding-3-small"    # or text-embedding-3-large
dimension = 1536                     # 1536 for small, 3072 for large
api_key_env = "OPENAI_API_KEY"
```

```rust
// Rust: Generate embeddings
use reasonkit_core::embedding::OpenAIEmbedder;

let embedder = OpenAIEmbedder::new()
    .model("text-embedding-3-small")
    .build()?;

let embedding = embedder.embed("Your text here").await?;
```

---

## Cost Estimation

### Per-Query Cost Calculator

| Profile      | Model       | Avg Tokens | Estimated Cost |
| ------------ | ----------- | ---------- | -------------- |
| `--quick`    | GPT-4o-mini | ~2,000     | $0.002         |
| `--quick`    | GPT-4o      | ~2,000     | $0.027         |
| `--balanced` | GPT-4o-mini | ~5,000     | $0.004         |
| `--balanced` | GPT-4o      | ~5,000     | $0.063         |
| `--deep`     | GPT-4o      | ~10,000    | $0.125         |
| `--deep`     | o1          | ~10,000    | $0.750         |
| `--paranoid` | GPT-4o      | ~20,000    | $0.250         |
| `--paranoid` | o1          | ~20,000    | $1.500         |

### Monthly Budget Examples

```bash
# Developer: 100 queries/day with GPT-4o-mini
# 100 * 30 * $0.004 = $12/month

# Team: 500 mixed queries/day
# Quick (mini):      200 * 30 * $0.002 = $12
# Balanced (4o):     250 * 30 * $0.063 = $473
# Deep (4o):          50 * 30 * $0.125 = $188
# Total: ~$673/month

# Cost-optimized strategy
rk think "Initial analysis" --model gpt-4o-mini
rk think "Final verification" --model gpt-4o --profile deep
```

---

## Azure OpenAI Integration

For enterprise Azure OpenAI deployments:

```toml
[providers.openai]
api_base_url = "https://YOUR-RESOURCE.openai.azure.com"
api_key_env = "AZURE_OPENAI_KEY"
api_version = "2024-02-15-preview"
deployment_name = "gpt-4o-deployment"

# Azure-specific settings
azure_ad_token_env = "AZURE_AD_TOKEN"  # Optional: for AAD auth
```

```bash
# Environment setup for Azure
export AZURE_OPENAI_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com"

# Use Azure deployment
rk think "Enterprise analysis" \
  --provider openai \
  --api-base "$AZURE_OPENAI_ENDPOINT" \
  --model gpt-4o-deployment
```

---

## Troubleshooting

### Common Issues

#### 1. "Authentication error: Invalid API key"

```bash
# Verify key is set
echo $OPENAI_API_KEY | head -c 10

# Check key format (should start with sk-)
# Get new key from: https://platform.openai.com/api-keys

# Test directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### 2. "Rate limit exceeded (429)"

```bash
# Add rate limiting to config
[providers.openai]
requests_per_minute = 30
tokens_per_minute = 50000
retry_attempts = 5
retry_delay_ms = 2000

# Check your tier limits
# https://platform.openai.com/account/limits
```

#### 3. "Context length exceeded"

```bash
# Use appropriate model for context size
# GPT-4o: 128K context
# o1: 200K context

# Truncate input if needed
rk think "Long document" --max-input-tokens 50000

# Or use ReasonKit's chunking
rk ingest document.pdf  # Auto-chunks
rk query "Summary" --raptor  # Uses RAPTOR tree
```

#### 4. "Model not found"

```bash
# Check available models
rk providers show openai

# Use correct model ID
rk think "Query" --provider openai --model gpt-4o

# Note: Some models require API access approval
# Apply at: https://platform.openai.com/
```

#### 5. "JSON mode output invalid"

```bash
# Ensure prompt requests JSON
rk think "Return analysis as JSON with keys: score, reasons, recommendation" \
  --provider openai \
  --response-format json_object

# Or use JSON schema
rk think "Extract data" \
  --response-format json_schema \
  --schema '{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]}'
```

#### 6. "o1 model not responding as expected"

```bash
# o1 models have built-in reasoning - don't over-prompt
# Bad: "Think step by step and..."
# Good: "Solve this problem: ..."

# Use ReasonKit profiles which optimize for o1
rk think "Complex problem" \
  --provider openai \
  --model o1 \
  --profile deep  # Profile auto-adjusts prompts
```

---

## Best Practices

### Model Selection Strategy

```
Use Case                    Recommended Model
---------------------------------------------
Quick exploration           gpt-4o-mini
Daily development           gpt-4o
Complex reasoning           o1
Fast reasoning              o1-mini
Long context (>50K)         gpt-4o / o1
Cost-sensitive batch        gpt-4o-mini
JSON/structured output      gpt-4o
```

### Optimal Configuration

```toml
# Production configuration for OpenAI
[providers.openai]
api_key_env = "OPENAI_API_KEY"

# Model tiers
[providers.openai.models]
fast = "gpt-4o-mini"
balanced = "gpt-4o"
reasoning = "o1"
reasoning_fast = "o1-mini"

# Auto-select based on profile
[providers.openai.profile_mapping]
quick = "fast"
balanced = "balanced"
deep = "reasoning"
paranoid = "reasoning"
```

### Cost Control

```bash
# Set hard budget limit
rk think "Expensive analysis" \
  --provider openai \
  --budget "$1.00"

# Use mini for drafts, upgrade for final
rk think "Draft" --model gpt-4o-mini --profile quick
rk think "Final" --model gpt-4o --profile deep

# Monitor usage
rk metrics cost --provider openai --period month
```

### Reproducibility

```bash
# Use seed for reproducible outputs
rk think "Analysis" \
  --provider openai \
  --model gpt-4o \
  --seed 42 \
  --temperature 0

# Same seed + temperature=0 = consistent output
```

---

## Comparison: GPT-4o vs o1

| Feature           | GPT-4o    | o1                 |
| ----------------- | --------- | ------------------ |
| Speed             | Fast      | Slower (reasoning) |
| Cost              | Lower     | Higher             |
| JSON mode         | Yes       | Limited            |
| Function calling  | Yes       | No                 |
| Vision            | Yes       | No                 |
| Complex reasoning | Good      | Excellent          |
| Multi-step math   | Good      | Excellent          |
| Code generation   | Excellent | Good               |

**Recommendation:**

- Use GPT-4o for most ReasonKit tasks
- Use o1 for `--deep` and `--paranoid` profiles
- Use GPT-4o-mini for `--quick` and cost-sensitive tasks

---

## Resources

- **OpenAI Platform:** <https://platform.openai.com/>
- **API Documentation:** <https://platform.openai.com/docs/>
- **Pricing:** <https://openai.com/pricing>
- **Rate Limits:** <https://platform.openai.com/account/limits>
- **Azure OpenAI:** <https://azure.microsoft.com/en-us/products/ai-services/openai-service>

---

_ReasonKit + OpenAI Integration Guide | v1.0.0 | Apache 2.0_
_"See How Your AI Thinks"_
