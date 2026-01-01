# ADR-004: Provider-Agnostic LLM Interface

## Status

**Accepted** - 2024-12-28

## Context

ReasonKit relies on Large Language Models (LLMs) for reasoning steps. The LLM landscape is rapidly evolving with multiple providers:

- **OpenAI**: GPT-4, GPT-4o, o1-series
- **Anthropic**: Claude 3.5, Claude 4 (Sonnet, Opus)
- **Google**: Gemini 2.0, Gemini 3
- **Mistral**: Mistral Large, Devstral
- **Open Source**: Llama 3.3, DeepSeek, Qwen
- **Aggregators**: OpenRouter, Together AI, Groq

Users have different requirements:

1. **Cost**: Some providers are 10-100x cheaper for similar capability
2. **Privacy**: Enterprise users may require on-premise deployment
3. **Latency**: Edge cases require low-latency providers
4. **Capability**: Specific tasks may favor specific models
5. **Compliance**: Regulated industries have provider restrictions

We evaluated several approaches:

| Approach                    | Pros                          | Cons                              |
| --------------------------- | ----------------------------- | --------------------------------- |
| **Hard-coded Provider**     | Simple, optimized             | Vendor lock-in, no flexibility    |
| **Configuration Switch**    | User choice, simple           | Duplicated code per provider      |
| **Abstract Interface**      | Maximum flexibility, testable | More code, abstraction overhead   |
| **LiteLLM/Similar Wrapper** | Pre-built, many providers     | External dependency, less control |

### Key Requirements

1. **Provider Choice**: Users must be able to choose their LLM provider
2. **Consistent Experience**: Same ReasonKit behavior regardless of provider
3. **Easy Switching**: Changing providers should require only configuration
4. **Testability**: Must be able to mock LLM responses for testing
5. **Streaming Support**: Real-time output for interactive use cases
6. **Error Normalization**: Consistent error handling across providers

## Decision

**We will implement an abstract LLM interface with provider-specific adapters.**

### Core Interface

```rust
/// Abstract interface for LLM providers
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Provider identifier
    fn id(&self) -> &'static str;

    /// Check if provider is available/configured
    async fn health_check(&self) -> Result<HealthStatus, ProviderError>;

    /// Send a completion request
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError>;

    /// Send a streaming completion request
    fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, ProviderError>> + Send>>;

    /// Estimate tokens for input
    fn estimate_tokens(&self, text: &str) -> usize;

    /// Get model capabilities
    fn capabilities(&self) -> ModelCapabilities;
}

/// Normalized completion request
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stop_sequences: Vec<String>,
    pub system_prompt: Option<String>,
}

/// Normalized completion response
pub struct CompletionResponse {
    pub content: String,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
    pub model: String,
    pub latency_ms: u64,
}
```

### Provider Adapters

```rust
// OpenAI-compatible adapter (works with OpenAI, Azure, local servers)
pub struct OpenAiAdapter {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
}

// Anthropic adapter
pub struct AnthropicAdapter {
    client: reqwest::Client,
    api_key: String,
}

// OpenRouter adapter (300+ models)
pub struct OpenRouterAdapter {
    client: reqwest::Client,
    api_key: String,
    preferred_providers: Vec<String>,
}

// Mock adapter for testing
pub struct MockAdapter {
    responses: Vec<CompletionResponse>,
    call_count: AtomicUsize,
}
```

### Configuration

```toml
# ~/.reasonkit/config.toml

[llm]
# Primary provider
provider = "anthropic"
model = "claude-sonnet-4-20250514"

# Fallback chain
fallback = ["openrouter", "openai"]

[llm.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[llm.openai]
api_key_env = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"  # Override for Azure/local

[llm.openrouter]
api_key_env = "OPENROUTER_API_KEY"
site_name = "ReasonKit"
```

### Runtime Selection

```rust
pub struct LlmManager {
    providers: HashMap<String, Box<dyn LlmProvider>>,
    primary: String,
    fallback_chain: Vec<String>,
}

impl LlmManager {
    /// Get the configured provider with fallback support
    pub async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        // Try primary
        if let Ok(response) = self.providers[&self.primary].complete(request.clone()).await {
            return Ok(response);
        }

        // Try fallbacks
        for provider_id in &self.fallback_chain {
            if let Ok(response) = self.providers[provider_id].complete(request.clone()).await {
                return Ok(response);
            }
        }

        Err(ProviderError::AllProvidersFailed)
    }
}
```

## Consequences

### Positive

1. **User Choice**: Users can use any supported provider without code changes
2. **Vendor Independence**: No lock-in to specific LLM provider
3. **Cost Optimization**: Easy switching to cheaper providers for specific tasks
4. **Testability**: Mock adapter enables deterministic testing
5. **Fallback Support**: Automatic failover when providers are unavailable
6. **Consistent API**: Same ReasonKit experience regardless of underlying LLM
7. **Future-Proof**: New providers added without changing core code

### Negative

1. **Abstraction Overhead**: Interface layer adds code complexity
2. **Lowest Common Denominator**: Some provider-specific features may be unavailable
3. **Maintenance Burden**: Each provider adapter requires updates when APIs change
4. **Configuration Complexity**: Users must understand provider setup

### Mitigations

| Negative                  | Mitigation                                                           |
| ------------------------- | -------------------------------------------------------------------- |
| Abstraction overhead      | Keep interface minimal; optimize hot paths                           |
| Lowest common denominator | `capabilities()` method exposes provider-specific features           |
| Maintenance burden        | OpenAI-compatible API covers many providers; community contributions |
| Configuration complexity  | Sensible defaults; auto-detection of API keys from environment       |

### Provider Support Matrix

| Provider     | Status  | Notes                                   |
| ------------ | ------- | --------------------------------------- |
| OpenAI       | Stable  | GPT-4, GPT-4o, o1-series                |
| Anthropic    | Stable  | Claude 3.5, Claude 4                    |
| OpenRouter   | Stable  | 300+ models, automatic routing          |
| Azure OpenAI | Stable  | Via OpenAI adapter with custom base_url |
| Ollama       | Stable  | Local models via OpenAI-compatible API  |
| Groq         | Stable  | Fast inference for open models          |
| Together AI  | Planned | Open model hosting                      |
| BedRock      | Planned | AWS-hosted models                       |

## Related Documents

- `/home/zyxsys/RK-PROJECT/reasonkit-core/src/thinktool/llm.rs` - LLM interface
- `/home/zyxsys/RK-PROJECT/ORCHESTRATOR.md` - Agent swarm hierarchy
- `/home/zyxsys/RK-PROJECT/config/AGENT_DREAM_TEAM.md` - Model selection guide

## References

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/claude/reference)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [LiteLLM](https://github.com/BerriAI/litellm) - Similar concept in Python
