# Google Gemini Integration Guide

> ReasonKit + Gemini: Massive Context, Multimodal Reasoning
> "Turn Prompts into Protocols with 2M Token Context"

**Provider:** Google AI / Vertex AI
**Models:** Gemini 2.0 Flash, Gemini 2.0 Pro, Gemini 1.5 Pro
**Best For:** Long-context analysis, multimodal inputs, document processing

---

## Quick Start (5 Lines)

```bash
# 1. Set API key
export GEMINI_API_KEY="..."

# 2. Install ReasonKit
cargo install reasonkit-core

# 3. Run analysis
rk think --provider gemini "Analyze this architecture"
```

---

## Environment Setup

### API Key Configuration

```bash
# Option 1: Google AI Studio (simplest)
export GEMINI_API_KEY="..."
# Get key from: https://aistudio.google.com/app/apikey

# Option 2: Vertex AI (enterprise)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export VERTEX_PROJECT="your-project-id"
export VERTEX_LOCATION="us-central1"

# Option 3: In ~/.reasonkit/config.toml
[providers.gemini]
api_key_env = "GEMINI_API_KEY"
```

### Verify Setup

```bash
# Test API connection
rk think --provider gemini --model gemini-2.0-flash "Hello"

# Check available models
rk providers show gemini
```

---

## Full Configuration Options

### ~/.ReasonKit/config.toml

```toml
[providers.gemini]
# API Configuration (Google AI Studio)
api_key_env = "GEMINI_API_KEY"
api_base_url = "https://generativelanguage.googleapis.com"

# OR Vertex AI Configuration
use_vertex = false
vertex_project_env = "VERTEX_PROJECT"
vertex_location = "us-central1"
credentials_path_env = "GOOGLE_APPLICATION_CREDENTIALS"

# Default Model Settings
default_model = "gemini-2.0-flash"
temperature = 0.7
max_tokens = 8192
top_p = 0.95
top_k = 40

# Safety Settings
safety_threshold = "BLOCK_MEDIUM_AND_ABOVE"  # BLOCK_NONE, BLOCK_LOW, BLOCK_MEDIUM, BLOCK_HIGH

# Rate Limiting
requests_per_minute = 60
retry_attempts = 3
retry_delay_ms = 1000
```

### Available Models

| Model                         | ID                              | Context | Best For             | Cost/1M tokens        |
| ----------------------------- | ------------------------------- | ------- | -------------------- | --------------------- |
| **Gemini 2.0 Flash**          | `gemini-2.0-flash`              | 1M      | Fast, cost-effective | $0.075 in / $0.30 out |
| **Gemini 2.0 Flash Thinking** | `gemini-2.0-flash-thinking-exp` | 1M      | Reasoning tasks      | $0.075 in / $0.30 out |
| **Gemini 1.5 Pro**            | `gemini-1.5-pro`                | 2M      | Maximum context      | $1.25 in / $5.00 out  |
| **Gemini 1.5 Flash**          | `gemini-1.5-flash`              | 1M      | Budget option        | $0.075 in / $0.30 out |

---

## ThinkTool Usage Examples

### Single Protocol Execution

```bash
# GigaThink: Creative expansion
rk think "Brainstorm use cases for AI in healthcare" \
  --provider gemini \
  --model gemini-2.0-flash \
  --protocol gigathink

# LaserLogic: Logical validation
rk think "Validate this mathematical proof" \
  --provider gemini \
  --protocol laserlogic

# ProofGuard: Evidence verification with long context
rk think "Verify claims in this research paper" \
  --provider gemini \
  --model gemini-1.5-pro \
  --protocol proofguard

# BedRock: First principles decomposition
rk think "Break down the fundamentals of quantum computing" \
  --provider gemini \
  --protocol bedrock

# BrutalHonesty: Adversarial critique
rk think "Critique this product roadmap" \
  --provider gemini \
  --protocol brutalhonesty
```

### Profile-Based Execution

```bash
# Quick analysis (Gemini Flash for speed)
rk think "Quick review of this code" \
  --provider gemini \
  --model gemini-2.0-flash \
  --profile quick

# Balanced analysis
rk think "Evaluate this system design" \
  --provider gemini \
  --profile balanced

# Deep analysis with long context
rk think "Comprehensive analysis of this codebase" \
  --provider gemini \
  --model gemini-1.5-pro \
  --profile deep

# Paranoid mode with thinking model
rk think "Security audit this contract" \
  --provider gemini \
  --model gemini-2.0-flash-thinking-exp \
  --profile paranoid
```

### Long Context Analysis (Killer Feature)

```bash
# Analyze entire codebase (up to 2M tokens!)
rk think "Review this entire repository for issues" \
  --provider gemini \
  --model gemini-1.5-pro \
  --max-input-tokens 500000 \
  --profile deep

# Process long documents
rk think "Summarize this 500-page document" \
  --provider gemini \
  --model gemini-1.5-pro \
  --attach document.pdf

# Multi-document analysis
rk think "Compare these three research papers" \
  --provider gemini \
  --model gemini-1.5-pro \
  --attach paper1.pdf \
  --attach paper2.pdf \
  --attach paper3.pdf
```

### Multimodal Inputs

```bash
# Image analysis
rk think "Analyze this architecture diagram" \
  --provider gemini \
  --model gemini-2.0-flash \
  --attach diagram.png

# Video analysis (YouTube)
rk think "Summarize this video with timestamps" \
  --provider gemini \
  --model gemini-2.0-flash \
  --attach "https://youtube.com/watch?v=..."

# Audio transcription and analysis
rk think "Transcribe and analyze this meeting" \
  --provider gemini \
  --model gemini-2.0-flash \
  --attach meeting.mp3
```

---

## Rust API Integration

```rust
use reasonkit_core::providers::GeminiProvider;
use reasonkit_core::thinktool::{ThinkToolOrchestrator, ReasoningProfile};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize provider (Google AI Studio)
    let provider = GeminiProvider::new()
        .model("gemini-2.0-flash")
        .temperature(0.7)
        .max_tokens(8192)
        .build()?;

    // OR Initialize for Vertex AI
    let vertex_provider = GeminiProvider::vertex()
        .project("your-project-id")
        .location("us-central1")
        .model("gemini-2.0-flash")
        .build()?;

    // Create orchestrator
    let orchestrator = ThinkToolOrchestrator::with_provider(provider);

    // Execute with profile
    let result = orchestrator
        .think("Analyze this architecture")
        .profile(ReasoningProfile::Balanced)
        .execute()
        .await?;

    println!("Confidence: {:.1}%", result.confidence.overall * 100.0);
    println!("Result: {}", result.output);

    Ok(())
}
```

### Multimodal with Rust

```rust
use reasonkit_core::providers::{GeminiProvider, Attachment};

let provider = GeminiProvider::new()
    .model("gemini-2.0-flash")
    .build()?;

let orchestrator = ThinkToolOrchestrator::with_provider(provider);

// With image attachment
let result = orchestrator
    .think("Describe this diagram")
    .attach(Attachment::image("diagram.png")?)
    .profile(ReasoningProfile::Quick)
    .execute()
    .await?;

// With multiple documents
let result = orchestrator
    .think("Compare these papers")
    .attach(Attachment::pdf("paper1.pdf")?)
    .attach(Attachment::pdf("paper2.pdf")?)
    .profile(ReasoningProfile::Deep)
    .execute()
    .await?;
```

---

## Python API Integration

```python
from reasonkit import ThinkToolOrchestrator, ReasoningProfile
from reasonkit.providers import GeminiProvider

# Initialize provider (Google AI Studio)
provider = GeminiProvider(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=8192
)

# OR Initialize for Vertex AI
provider = GeminiProvider.from_vertex(
    project="your-project-id",
    location="us-central1",
    model="gemini-2.0-flash"
)

# Create orchestrator
orchestrator = ThinkToolOrchestrator(provider=provider)

# Execute analysis
result = orchestrator.think(
    query="Evaluate this business strategy",
    profile=ReasoningProfile.BALANCED
)

print(f"Confidence: {result.confidence.overall:.1%}")
print(f"Result: {result.output}")
```

### Multimodal with Python

```python
from reasonkit import ThinkToolOrchestrator, Attachment

orchestrator = ThinkToolOrchestrator(provider=provider)

# With image
result = orchestrator.think(
    query="Analyze this diagram",
    attachments=[Attachment.image("diagram.png")],
    profile=ReasoningProfile.BALANCED
)

# With long document
result = orchestrator.think(
    query="Summarize this document",
    attachments=[Attachment.pdf("document.pdf")],
    profile=ReasoningProfile.DEEP
)

# With video (YouTube URL)
result = orchestrator.think(
    query="Extract key points from this video",
    attachments=[Attachment.url("https://youtube.com/watch?v=...")],
    profile=ReasoningProfile.BALANCED
)
```

---

## Gemini CLI Integration

ReasonKit integrates with the Gemini CLI for AI-to-AI consultation:

```bash
# Install Gemini CLI
npm install -g @anthropic-ai/gemini-cli  # Check official source

# One-shot consultation
gemini -p "Review this analysis for blind spots: $(cat analysis.json)"

# Pipe ReasonKit output to Gemini
rk think "Design a caching strategy" --format json | \
  gemini -p "Find flaws in this reasoning"

# Use sandbox mode for untrusted inputs
gemini --sandbox -p "Analyze this code"
```

---

## Cost Estimation

### Per-Query Cost Calculator

| Profile             | Model     | Avg Tokens | Estimated Cost |
| ------------------- | --------- | ---------- | -------------- |
| `--quick`           | 2.0 Flash | ~2,000     | $0.0008        |
| `--quick`           | 1.5 Pro   | ~2,000     | $0.012         |
| `--balanced`        | 2.0 Flash | ~5,000     | $0.002         |
| `--balanced`        | 1.5 Pro   | ~5,000     | $0.031         |
| `--deep`            | 2.0 Flash | ~10,000    | $0.004         |
| `--deep`            | 1.5 Pro   | ~10,000    | $0.063         |
| `--paranoid`        | 1.5 Pro   | ~20,000    | $0.125         |
| Long context (500K) | 1.5 Pro   | ~500,000   | $3.125         |

### Monthly Budget Examples

```bash
# Developer: 100 queries/day with Gemini Flash
# 100 * 30 * $0.002 = $6/month (VERY cost-effective!)

# Team: Heavy document processing
# Quick (Flash):     200 * 30 * $0.0008 = $4.80
# Balanced (Flash):  300 * 30 * $0.002  = $18.00
# Deep (1.5 Pro):     50 * 30 * $0.063  = $94.50
# Long context:       10 * 30 * $3.125  = $937.50
# Total: ~$1,055/month (mostly from long context)

# Cost-optimized for long context
rk think "Process large document" \
  --provider gemini \
  --model gemini-2.0-flash \  # Use Flash even for long context
  --budget "$5.00"
```

---

## Vertex AI Enterprise Setup

For production enterprise deployments:

```toml
[providers.gemini]
use_vertex = true
vertex_project_env = "VERTEX_PROJECT"
vertex_location = "us-central1"
credentials_path_env = "GOOGLE_APPLICATION_CREDENTIALS"

# Enterprise features
enable_grounding = true              # Ground responses with Google Search
enable_safety_attributes = true      # Get safety ratings
enable_citation_metadata = true      # Get source citations
```

```bash
# Set up Vertex AI credentials
gcloud auth application-default login
export VERTEX_PROJECT="my-project"
export VERTEX_LOCATION="us-central1"

# Use Vertex AI
rk think "Enterprise analysis" \
  --provider gemini \
  --vertex \
  --model gemini-2.0-flash
```

---

## Troubleshooting

### Common Issues

#### 1. "Invalid API key"

```bash
# Verify key is set
echo $GEMINI_API_KEY | head -c 10

# Get key from Google AI Studio
# https://aistudio.google.com/app/apikey

# Test directly
curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY"
```

#### 2. "Quota exceeded"

```bash
# Check quotas in Google Cloud Console
# https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas

# Add rate limiting
[providers.gemini]
requests_per_minute = 30
retry_attempts = 5
retry_delay_ms = 2000

# Or upgrade to paid tier
```

#### 3. "Safety filter blocked response"

```bash
# Adjust safety settings (use carefully)
[providers.gemini]
safety_threshold = "BLOCK_ONLY_HIGH"

# Or check your prompt content
# Gemini has stricter safety filters than some providers
```

#### 4. "Context length exceeded"

```bash
# Use model with larger context
# Gemini 1.5 Pro: 2M tokens (!)
rk think "Long analysis" \
  --provider gemini \
  --model gemini-1.5-pro

# Check token count before sending
rk tokens count "your long text here"
```

#### 5. "Multimodal input failed"

```bash
# Ensure file is supported format
# Images: PNG, JPEG, WEBP, GIF
# Video: MP4, MOV
# Audio: MP3, WAV, M4A
# Documents: PDF

# Check file size limits (varies by model)
# Generally: Images < 20MB, Video < 2GB

# Use direct URL for large files
rk think "Analyze video" \
  --provider gemini \
  --attach "https://storage.googleapis.com/bucket/video.mp4"
```

#### 6. "Vertex AI authentication failed"

```bash
# Re-authenticate
gcloud auth application-default login

# Or use service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

# Verify project access
gcloud config set project YOUR_PROJECT_ID
gcloud ai models list --region=us-central1
```

---

## Best Practices

### Model Selection Strategy

```
Use Case                      Recommended Model
-----------------------------------------------
Quick exploration             gemini-2.0-flash
Daily development             gemini-2.0-flash
Long document (>100K tokens)  gemini-1.5-pro
Complex reasoning             gemini-2.0-flash-thinking-exp
Multimodal analysis           gemini-2.0-flash
Cost-sensitive batch          gemini-1.5-flash
Enterprise production         gemini-1.5-pro (Vertex AI)
```

### Optimal Configuration

```toml
# Production configuration for Gemini
[providers.gemini]
api_key_env = "GEMINI_API_KEY"

# Model tiers
[providers.gemini.models]
fast = "gemini-2.0-flash"
balanced = "gemini-2.0-flash"
long_context = "gemini-1.5-pro"
reasoning = "gemini-2.0-flash-thinking-exp"

# Auto-select based on profile
[providers.gemini.profile_mapping]
quick = "fast"
balanced = "balanced"
deep = "reasoning"
paranoid = "reasoning"
```

### Leverage Long Context

```bash
# Gemini excels at long-context tasks
# Use it for:
# - Full codebase analysis
# - Multi-document comparison
# - Long transcript analysis
# - Comprehensive literature review

# Example: Analyze entire project
find src -name "*.rs" -exec cat {} \; | \
  rk think "Review this codebase" \
    --provider gemini \
    --model gemini-1.5-pro \
    --profile deep
```

### Cost Control

```bash
# Set budget limits
rk think "Analysis" \
  --provider gemini \
  --budget "$1.00"

# Use Flash for most tasks (10x cheaper than Pro)
# Reserve Pro for long-context only

# Monitor usage
rk metrics cost --provider gemini --period month
```

---

## Gemini vs Other Providers

| Feature     | Gemini               | Claude       | GPT-4     |
| ----------- | -------------------- | ------------ | --------- |
| Max context | 2M tokens            | 200K         | 128K      |
| Multimodal  | Images, Video, Audio | Images, PDFs | Images    |
| Speed       | Very fast            | Fast         | Fast      |
| Cost        | Lowest               | Medium       | Medium    |
| Reasoning   | Good                 | Excellent    | Excellent |
| Safety      | Strictest            | Moderate     | Moderate  |

**When to choose Gemini:**

- Long document processing
- Video/audio analysis
- Cost-sensitive workloads
- Multimodal inputs
- Google Cloud integration

---

## Resources

- **Google AI Studio:** <https://aistudio.google.com/>
- **Vertex AI:** <https://cloud.google.com/vertex-ai>
- **API Documentation:** <https://ai.google.dev/docs>
- **Pricing:** <https://ai.google.dev/pricing>
- **Gemini CLI:** <https://github.com/google-gemini/gemini-cli> (check official)

---

*ReasonKit + Google Gemini Integration Guide | v1.0.0 | Apache 2.0*
*"See How Your AI Thinks"*
