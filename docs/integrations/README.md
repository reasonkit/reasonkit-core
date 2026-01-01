# ReasonKit Integrations

> Connect ReasonKit to any LLM provider or CLI tool
> "Turn Prompts into Protocols - Anywhere"

---

## LLM Provider Integration Guides

Comprehensive guides for integrating ReasonKit with major LLM providers:

| Provider               | Guide                                                    | Best For                                              | Quick Start                                   |
| ---------------------- | -------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------- |
| **Anthropic (Claude)** | [ANTHROPIC_INTEGRATION.md](./ANTHROPIC_INTEGRATION.md)   | Complex reasoning, extended thinking, safety-critical | `rk-core think --provider anthropic "query"`  |
| **OpenAI (GPT)**       | [OPENAI_INTEGRATION.md](./OPENAI_INTEGRATION.md)         | General reasoning, JSON mode, embeddings              | `rk-core think --provider openai "query"`     |
| **Google (Gemini)**    | [GOOGLE_INTEGRATION.md](./GOOGLE_INTEGRATION.md)         | Long context (2M), multimodal, documents              | `rk-core think --provider gemini "query"`     |
| **Groq**               | [GROQ_INTEGRATION.md](./GROQ_INTEGRATION.md)             | Ultra-fast inference, rapid iteration                 | `rk-core think --provider groq "query"`       |
| **OpenRouter**         | [OPENROUTER_INTEGRATION.md](./OPENROUTER_INTEGRATION.md) | 300+ models, fallback routing, cost optimization      | `rk-core think --provider openrouter "query"` |

---

## Quick Comparison

### Speed vs Quality vs Cost

| Provider          | Speed     | Quality   | Cost     | Best Profile            |
| ----------------- | --------- | --------- | -------- | ----------------------- |
| **Groq**          | Fastest   | Good      | Lowest   | `--quick`               |
| **Gemini Flash**  | Very Fast | Good      | Very Low | `--quick`, `--balanced` |
| **OpenAI GPT-4o** | Fast      | Excellent | Medium   | `--balanced`            |
| **Claude Sonnet** | Fast      | Excellent | Medium   | `--balanced`, `--deep`  |
| **Claude Opus**   | Medium    | Best      | High     | `--deep`, `--paranoid`  |
| **OpenAI o1**     | Slow      | Excellent | High     | `--deep`, `--paranoid`  |

### Feature Matrix

| Feature           | Anthropic | OpenAI   | Gemini | Groq    | OpenRouter |
| ----------------- | --------- | -------- | ------ | ------- | ---------- |
| Max Context       | 200K      | 200K     | 2M     | 128K    | Varies     |
| Extended Thinking | Yes       | Yes (o1) | Yes    | No      | Via models |
| JSON Mode         | No        | Yes      | Yes    | Yes     | Via models |
| Vision            | Yes       | Yes      | Yes    | No      | Via models |
| Audio             | No        | Yes      | Yes    | Yes     | Via models |
| Video             | No        | No       | Yes    | No      | Via models |
| Embeddings        | No        | Yes      | Yes    | No      | Via models |
| Free Tier         | No        | No       | Yes    | Limited | Yes        |

---

## Environment Setup (All Providers)

Set all API keys for maximum flexibility:

```bash
# Core providers
export ANTHROPIC_API_KEY="sk-ant-..."    # Anthropic Claude
export OPENAI_API_KEY="sk-..."           # OpenAI GPT
export GEMINI_API_KEY="..."              # Google Gemini
export GROQ_API_KEY="gsk_..."            # Groq
export OPENROUTER_API_KEY="sk-or-..."    # OpenRouter (300+ models)

# Additional providers
export XAI_API_KEY="xai-..."             # xAI Grok
export MISTRAL_API_KEY="..."             # Mistral AI
export DEEPSEEK_API_KEY="..."            # DeepSeek
```

---

## Default Configuration

### ~/.ReasonKit/config.toml

```toml
# Default provider for rk-core think
[thinktool]
default_provider = "anthropic"
default_model = "claude-sonnet-4"

# Provider-specific defaults
[providers.anthropic]
default_model = "claude-sonnet-4"

[providers.openai]
default_model = "gpt-4o"

[providers.gemini]
default_model = "gemini-2.0-flash"

[providers.groq]
default_model = "llama-3.3-70b-versatile"

[providers.openrouter]
default_model = "anthropic/claude-sonnet-4"
```

---

## Usage Patterns

### Single Provider

```bash
# Use specific provider
rk-core think --provider anthropic "Analyze this code"
rk-core think --provider openai "Evaluate this design"
rk-core think --provider gemini "Summarize this document"
rk-core think --provider groq "Quick review"
```

### Provider + Model

```bash
# Specify exact model
rk-core think --provider anthropic --model claude-opus-4 "Deep analysis"
rk-core think --provider openai --model o1 "Complex reasoning"
rk-core think --provider gemini --model gemini-1.5-pro "Long document"
rk-core think --provider groq --model llama-3.1-405b-reasoning "Hard problem"
```

### Profile-Based Selection

```bash
# Profiles auto-select appropriate models
rk-core think --profile quick "Fast check"     # Uses fastest available
rk-core think --profile balanced "Standard"    # Uses balanced model
rk-core think --profile deep "Thorough"        # Uses reasoning model
rk-core think --profile paranoid "Critical"    # Uses best model
```

---

## CLI Tool Integration

ReasonKit can act as the default reasoning layer inside popular CLI agents:

| Tool           | Injection Method         | Bypass    |
| -------------- | ------------------------ | --------- |
| `claude`       | `--append-system-prompt` | `--no-rk` |
| `gemini`       | Prompt prefix            | `--no-rk` |
| `codex`        | Prompt prefix            | `--no-rk` |
| `opencode`     | Prompt prefix            | `--no-rk` |
| `cursor-agent` | Prompt prefix            | `--no-rk` |
| `copilot`      | via `gh copilot`         | `--no-rk` |

### Install CLI Defaults

```bash
bash reasonkit-core/scripts/install_cli_defaults.sh
source ~/.zshrc
```

### Central Config

All defaults are controlled by `reasonkit-core/config/cli_defaults.toml`:

```bash
export RK_CONFIG="$HOME/RK-PROJECT/reasonkit-core/config/cli_defaults.toml"
```

Example:

```toml
[defaults]
profile = "balanced"
protocol_dir = "protocols/cli"

[tools.codex]
profile = "paranoid"
```

### Override Per-Session

```bash
export RK_PROFILE=quick
# or
export RK_PROFILE=paranoid
```

### Bypass ReasonKit

```bash
claude --no-rk  # Run without ReasonKit injection
```

---

## Protocol Files

Protocols are stored in `reasonkit-core/protocols/cli/`:

- `balanced.md`
- `quick.md`
- `paranoid.md`

---

## Cost Management

### Budget Controls

```bash
# Set per-query budget
rk-core think "Expensive query" --budget "$1.00"

# Set time budget
rk-core think "Time-limited" --budget "60s"

# Set token budget
rk-core think "Token-limited" --budget "5000t"
```

### Cost Tracking

```bash
# View usage
rk-core metrics cost --period day
rk-core metrics cost --period month
rk-core metrics cost --provider anthropic
```

---

## Additional Resources

- [TOOLING_SPEC.md](./TOOLING_SPEC.md) - CLI tool integration details
- [CONSULTATIONS.md](./CONSULTATIONS.md) - AI-to-AI consultation patterns
- [CLI_REFERENCE.md](../CLI_REFERENCE.md) - Full CLI documentation
- [THINKTOOLS_QUICK_REFERENCE.md](../THINKTOOLS_QUICK_REFERENCE.md) - ThinkTool cheat sheet

---

*ReasonKit Integrations | v1.0.0 | Apache 2.0*
*"See How Your AI Thinks"*
