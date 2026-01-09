# Groq Integration Guide

> ReasonKit + Groq: Ultra-Fast Inference for Rapid Reasoning
> "Turn Prompts into Protocols at Lightning Speed"

**Provider:** Groq
**Hardware:** LPU (Language Processing Unit)
**Models:** Llama 3.3 70B, Llama 3.1 405B, Mixtral 8x7B, Gemma 2
**Best For:** Fast iteration, real-time analysis, cost-effective inference

---

## Quick Start (5 Lines)

```bash
# 1. Set API key
export GROQ_API_KEY="gsk_..."

# 2. Install ReasonKit
cargo install reasonkit-core

# 3. Run analysis (10x faster than typical cloud!)
rk think --provider groq "Analyze this code quickly"
```

---

## Environment Setup

### API Key Configuration

```bash
# Option 1: Environment variable (recommended)
export GROQ_API_KEY="gsk_..."
# Get key from: https://console.groq.com/keys

# Option 2: In ~/.reasonkit/config.toml
[providers.groq]
api_key_env = "GROQ_API_KEY"
```

### Verify Setup

```bash
# Test API connection
rk think --provider groq --model llama-3.3-70b-versatile "Hello"

# Check available models
rk providers show groq
```

---

## Full Configuration Options

### ~/.ReasonKit/config.toml

```toml
[providers.groq]
# API Configuration
api_key_env = "GROQ_API_KEY"
api_base_url = "https://api.groq.com/openai/v1"  # OpenAI-compatible

# Default Model Settings
default_model = "llama-3.3-70b-versatile"
temperature = 0.7
max_tokens = 4096
top_p = 0.95

# Stop Sequences
stop_sequences = ["</output>", "HALT"]

# Rate Limiting (Groq has generous limits)
requests_per_minute = 30
tokens_per_minute = 100000
retry_attempts = 3
retry_delay_ms = 500
```

### Available Models

| Model              | ID                         | Context | Speed     | Best For          | Cost/1M tokens        |
| ------------------ | -------------------------- | ------- | --------- | ----------------- | --------------------- |
| **Llama 3.3 70B**  | `llama-3.3-70b-versatile`  | 128K    | 330 tok/s | General purpose   | $0.59 in / $0.79 out  |
| **Llama 3.1 405B** | `llama-3.1-405b-reasoning` | 128K    | 100 tok/s | Complex reasoning | $5.00 in / $10.00 out |
| **Llama 3.1 70B**  | `llama-3.1-70b-versatile`  | 128K    | 330 tok/s | Balanced          | $0.59 in / $0.79 out  |
| **Llama 3.1 8B**   | `llama-3.1-8b-instant`     | 128K    | 750 tok/s | Ultra-fast        | $0.05 in / $0.08 out  |
| **Mixtral 8x7B**   | `mixtral-8x7b-32768`       | 32K     | 575 tok/s | MoE efficiency    | $0.24 in / $0.24 out  |
| **Gemma 2 9B**     | `gemma2-9b-it`             | 8K      | 500 tok/s | Instruction tuned | $0.20 in / $0.20 out  |

**Speed Note:** Groq's LPU delivers 5-10x faster inference than GPU-based providers!

---

## ThinkTool Usage Examples

### Single Protocol Execution

```bash
# GigaThink: Rapid creative expansion
rk think "Generate startup ideas in AI space" \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --protocol gigathink

# LaserLogic: Fast logical validation
rk think "Check this argument for fallacies" \
  --provider groq \
  --model llama-3.1-8b-instant \
  --protocol laserlogic

# ProofGuard: Quick verification
rk think "Verify these claims about Rust" \
  --provider groq \
  --protocol proofguard

# BedRock: First principles with 405B (complex)
rk think "Decompose the fundamentals of distributed consensus" \
  --provider groq \
  --model llama-3.1-405b-reasoning \
  --protocol bedrock

# BrutalHonesty: Rapid adversarial critique
rk think "Critique my API design" \
  --provider groq \
  --protocol brutalhonesty
```

### Profile-Based Execution

```bash
# Quick analysis (Llama 8B - lightning fast)
rk think "Quick code review" \
  --provider groq \
  --model llama-3.1-8b-instant \
  --profile quick

# Balanced analysis (Llama 70B)
rk think "Evaluate this architecture" \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --profile balanced

# Deep analysis with 405B
rk think "Complex system design analysis" \
  --provider groq \
  --model llama-3.1-405b-reasoning \
  --profile deep

# Paranoid mode (multi-pass)
rk think "Security audit this code" \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --profile paranoid
```

### Speed-Optimized Workflows

```bash
# Batch analysis with ultra-fast model
for file in src/*.rs; do
  rk think "Review: $(cat $file)" \
    --provider groq \
    --model llama-3.1-8b-instant \
    --profile quick \
    --timeout 10s
done

# Real-time code review loop
watch -n 5 'rk think "Check latest changes" \
  --provider groq \
  --model llama-3.1-8b-instant \
  --protocol laserlogic'

# Interactive rapid iteration
rk think --provider groq --model llama-3.1-8b-instant \
  --interactive  # Start interactive session
```

### Using Groq for Draft + Refine Workflow

```bash
# 1. Fast draft with Groq (cheap + fast)
rk think "Analyze this problem" \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --profile quick \
  --format json > draft.json

# 2. Refine with Claude/GPT (higher quality)
rk think "Improve and verify this analysis: $(cat draft.json)" \
  --provider anthropic \
  --model claude-sonnet-4 \
  --profile deep
```

---

## Rust API Integration

```rust
use reasonkit_core::providers::GroqProvider;
use reasonkit_core::thinktool::{ThinkToolOrchestrator, ReasoningProfile};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize Groq provider
    let provider = GroqProvider::new()
        .model("llama-3.3-70b-versatile")
        .temperature(0.7)
        .max_tokens(4096)
        .build()?;

    // Create orchestrator
    let orchestrator = ThinkToolOrchestrator::with_provider(provider);

    // Execute with profile (FAST!)
    let result = orchestrator
        .think("Analyze this quickly")
        .profile(ReasoningProfile::Quick)
        .execute()
        .await?;

    println!("Confidence: {:.1}%", result.confidence.overall * 100.0);
    println!("Result: {}", result.output);
    println!("Time: {}ms", result.execution_time_ms);  // Very fast!

    Ok(())
}
```

### Batch Processing with Groq

```rust
use reasonkit_core::providers::GroqProvider;
use reasonkit_core::thinktool::ThinkToolOrchestrator;
use futures::stream::{self, StreamExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = GroqProvider::new()
        .model("llama-3.1-8b-instant")  // Ultra-fast for batch
        .build()?;

    let orchestrator = ThinkToolOrchestrator::with_provider(provider);

    // Process many items in parallel
    let items = vec!["item1", "item2", "item3", /* ... */];

    let results: Vec<_> = stream::iter(items)
        .map(|item| {
            let orch = orchestrator.clone();
            async move {
                orch.think(&format!("Analyze: {}", item))
                    .profile(ReasoningProfile::Quick)
                    .execute()
                    .await
            }
        })
        .buffer_unordered(10)  // 10 concurrent requests
        .collect()
        .await;

    Ok(())
}
```

---

## Python API Integration

```python
from reasonkit import ThinkToolOrchestrator, ReasoningProfile
from reasonkit.providers import GroqProvider
import time

# Initialize Groq provider
provider = GroqProvider(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=4096
)

# Create orchestrator
orchestrator = ThinkToolOrchestrator(provider=provider)

# Execute analysis (will be FAST)
start = time.time()
result = orchestrator.think(
    query="Analyze this architecture decision",
    profile=ReasoningProfile.BALANCED
)
elapsed = time.time() - start

print(f"Confidence: {result.confidence.overall:.1%}")
print(f"Result: {result.output}")
print(f"Time: {elapsed:.2f}s")  # Often < 5 seconds!
```

### Batch Processing with Python

```python
import asyncio
from reasonkit import ThinkToolOrchestrator
from reasonkit.providers import GroqProvider

provider = GroqProvider(model="llama-3.1-8b-instant")
orchestrator = ThinkToolOrchestrator(provider=provider)

async def process_batch(items):
    tasks = [
        orchestrator.think_async(f"Analyze: {item}")
        for item in items
    ]
    return await asyncio.gather(*tasks)

# Process 100 items quickly
items = [f"item_{i}" for i in range(100)]
results = asyncio.run(process_batch(items))
```

---

## Cost Estimation

### Per-Query Cost Calculator

| Profile      | Model      | Avg Tokens | Estimated Cost |
| ------------ | ---------- | ---------- | -------------- |
| `--quick`    | Llama 8B   | ~2,000     | $0.0002        |
| `--quick`    | Llama 70B  | ~2,000     | $0.0014        |
| `--balanced` | Llama 8B   | ~5,000     | $0.0004        |
| `--balanced` | Llama 70B  | ~5,000     | $0.0035        |
| `--deep`     | Llama 70B  | ~10,000    | $0.0069        |
| `--deep`     | Llama 405B | ~10,000    | $0.075         |
| `--paranoid` | Llama 70B  | ~20,000    | $0.014         |
| `--paranoid` | Llama 405B | ~20,000    | $0.150         |

**Groq is EXTREMELY cost-effective!**

### Monthly Budget Examples

```bash
# Developer: 500 queries/day (heavy usage)
# 500 * 30 * $0.0035 = $52.50/month (Llama 70B balanced)

# Team: 2000 queries/day
# Quick (8B):        1000 * 30 * $0.0002 = $6
# Balanced (70B):     800 * 30 * $0.0035 = $84
# Deep (70B):         200 * 30 * $0.0069 = $41.40
# Total: ~$131/month (vs $1000+ with premium providers)

# Comparison: Same workload with Claude Sonnet
# 2000 * 30 * $0.09 = $5,400/month
# Groq savings: 97%!
```

### Speed vs Cost Trade-off

| Provider      | Speed     | Cost (balanced) | Best For         |
| ------------- | --------- | --------------- | ---------------- |
| Groq 8B       | 750 tok/s | $0.0004         | Rapid iteration  |
| Groq 70B      | 330 tok/s | $0.0035         | Daily use        |
| Claude Sonnet | 50 tok/s  | $0.09           | Quality-critical |
| GPT-4o        | 60 tok/s  | $0.063          | Complex tasks    |

---

## Troubleshooting

### Common Issues

#### 1. "Invalid API key"

```bash
# Verify key is set
echo $GROQ_API_KEY | head -c 10

# Check key format (should start with gsk_)
# Get new key from: https://console.groq.com/keys

# Test directly
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"
```

#### 2. "Rate limit exceeded"

```bash
# Groq has per-minute token limits
# Free tier: 6K tokens/min on 70B

# Add delays for batch processing
[providers.groq]
requests_per_minute = 25
retry_delay_ms = 1000

# Or upgrade to paid tier for higher limits
```

#### 3. "Model not available"

```bash
# Check current available models
rk providers show groq

# Some models rotate availability
# Use the versatile models for stability:
# - llama-3.3-70b-versatile (recommended)
# - llama-3.1-8b-instant (fastest)
```

#### 4. "Context length exceeded"

```bash
# Check model context limits
# Llama 3.x: 128K context
# Mixtral: 32K context
# Gemma 2: 8K context

# Use appropriate model
rk think "Long analysis" \
  --provider groq \
  --model llama-3.3-70b-versatile  # 128K context
```

#### 5. "Response quality lower than expected"

```bash
# Groq runs open-source models (not GPT-4/Claude)
# For critical tasks, use draft+refine workflow:

# 1. Fast draft with Groq
rk think "Initial analysis" \
  --provider groq --profile quick > draft.json

# 2. Verify/improve with premium provider
rk think "Verify: $(cat draft.json)" \
  --provider anthropic --profile deep
```

#### 6. "Timeout on 405B model"

```bash
# 405B is slower (but still fast for its size)
# Increase timeout
rk think "Complex query" \
  --provider groq \
  --model llama-3.1-405b-reasoning \
  --timeout 60s

# Or use 70B for most tasks
```

---

## Best Practices

### Model Selection Strategy

```
Use Case                    Recommended Model
---------------------------------------------
Rapid iteration             llama-3.1-8b-instant
Code review                 llama-3.3-70b-versatile
General reasoning           llama-3.3-70b-versatile
Complex analysis            llama-3.1-405b-reasoning
Batch processing            llama-3.1-8b-instant
Cost-sensitive              llama-3.1-8b-instant
```

### Optimal Configuration

```toml
# Production configuration for Groq
[providers.groq]
api_key_env = "GROQ_API_KEY"

# Model tiers based on speed needs
[providers.groq.models]
instant = "llama-3.1-8b-instant"      # 750 tok/s
fast = "llama-3.3-70b-versatile"      # 330 tok/s
reasoning = "llama-3.1-405b-reasoning"  # 100 tok/s

# Auto-select based on profile
[providers.groq.profile_mapping]
quick = "instant"
balanced = "fast"
deep = "reasoning"
paranoid = "fast"  # Multi-pass is faster than 405B
```

### Leverage Speed Advantage

```bash
# Use Groq for:
# 1. Rapid prototyping
# 2. CI/CD checks
# 3. Real-time feedback
# 4. Batch processing
# 5. Cost optimization

# Example: CI/CD integration
rk think "Review PR diff: $(git diff main)" \
  --provider groq \
  --model llama-3.1-8b-instant \
  --profile quick \
  --timeout 30s
```

### Hybrid Strategy (Recommended)

```bash
# Use Groq for speed, premium for quality

# Development loop: Groq
while developing:
  rk think "Quick check" --provider groq --profile quick
  # Iterate fast

# Final review: Claude/GPT
rk think "Final analysis" --provider anthropic --profile deep
```

---

## Groq vs Other Providers

| Feature | Groq        | Claude      | GPT-4       | Gemini      |
| ------- | ----------- | ----------- | ----------- | ----------- |
| Speed   | 750 tok/s   | 50 tok/s    | 60 tok/s    | 100 tok/s   |
| Cost    | Lowest      | Medium      | Medium      | Low         |
| Quality | Good        | Excellent   | Excellent   | Good        |
| Context | 128K        | 200K        | 128K        | 2M          |
| Models  | Open-source | Proprietary | Proprietary | Proprietary |

**When to choose Groq:**

- Speed is critical
- Cost optimization needed
- Rapid iteration/prototyping
- Batch processing
- CI/CD integration
- Open-source model preference

**When to use others:**

- Maximum reasoning quality needed
- Extended thinking required
- Safety-critical decisions
- Complex multi-step tasks

---

## Resources

- **Groq Console:** <https://console.groq.com/>
- **API Documentation:** <https://console.groq.com/docs>
- **Pricing:** <https://groq.com/pricing/>
- **LPU Technology:** <https://groq.com/technology/>
- **Status Page:** <https://status.groq.com/>

---

_ReasonKit + Groq Integration Guide | v1.0.0 | Apache 2.0_
_"See How Your AI Thinks - At Lightning Speed"_
