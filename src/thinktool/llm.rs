//! LLM Provider abstraction for ThinkTool Protocol Engine
//!
//! Supports 18+ providers through unified OpenAI-compatible interface:
//! - Anthropic, OpenAI, OpenRouter (original)
//! - Google Gemini, Google Vertex AI
//! - xAI (Grok), Groq, Mistral, DeepSeek
//! - Together AI, Fireworks AI, Alibaba Qwen
//! - Cloudflare AI, AWS Bedrock, Azure OpenAI
//! - Perplexity, Cohere, Cerebras
//!
//! ## Architecture
//!
//! Most modern LLM providers expose OpenAI-compatible `/chat/completions` endpoints,
//! enabling a unified client with provider-specific configuration. This module
//! leverages that standardization while supporting provider-specific features.
//!
//! ## Aggregation Layers
//!
//! For maximum flexibility, consider using aggregation layers:
//! - **OpenRouter**: 300+ models, automatic fallbacks, BYOK support
//! - **Cloudflare AI Gateway**: Unified endpoint, 350+ models, analytics
//! - **LiteLLM**: Python proxy for 100+ providers (external dependency)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::error::{Error, Result};

// ═══════════════════════════════════════════════════════════════════════════
// PROVIDER CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for LLM providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Provider identifier
    pub provider: LlmProvider,

    /// Model identifier (provider-specific format)
    pub model: String,

    /// API key (read from env if not provided)
    pub api_key: Option<String>,

    /// API base URL (optional override)
    pub base_url: Option<String>,

    /// Temperature for generation (0.0 - 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Provider-specific extra configuration
    #[serde(default)]
    pub extra: ProviderExtra,
}

/// Provider-specific extra configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderExtra {
    /// Azure resource name (for Azure OpenAI)
    pub azure_resource: Option<String>,

    /// Azure deployment name (for Azure OpenAI)
    pub azure_deployment: Option<String>,

    /// AWS region (for Bedrock)
    pub aws_region: Option<String>,

    /// Google Cloud project ID (for Vertex AI)
    pub gcp_project: Option<String>,

    /// Google Cloud location (for Vertex AI)
    pub gcp_location: Option<String>,

    /// Cloudflare account ID (for AI Gateway)
    pub cf_account_id: Option<String>,

    /// Cloudflare gateway ID (for AI Gateway)
    pub cf_gateway_id: Option<String>,

    /// Target provider for gateway routing (OpenRouter, Cloudflare)
    pub gateway_provider: Option<String>,
}

fn default_temperature() -> f64 {
    0.7
}

fn default_max_tokens() -> u32 {
    2000
}

fn default_timeout() -> u64 {
    60
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::Anthropic,
            model: "claude-opus-4-5".to_string(), // Latest flagship (Dec 2025)
            api_key: None,
            base_url: None,
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            timeout_secs: default_timeout(),
            extra: ProviderExtra::default(),
        }
    }
}

impl LlmConfig {
    /// Create config for a specific provider with model
    pub fn for_provider(provider: LlmProvider, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set API key
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set base URL override
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set Azure-specific configuration
    pub fn with_azure(
        mut self,
        resource: impl Into<String>,
        deployment: impl Into<String>,
    ) -> Self {
        self.extra.azure_resource = Some(resource.into());
        self.extra.azure_deployment = Some(deployment.into());
        self
    }

    /// Set AWS Bedrock region
    pub fn with_aws_region(mut self, region: impl Into<String>) -> Self {
        self.extra.aws_region = Some(region.into());
        self
    }

    /// Set Google Cloud configuration
    pub fn with_gcp(mut self, project: impl Into<String>, location: impl Into<String>) -> Self {
        self.extra.gcp_project = Some(project.into());
        self.extra.gcp_location = Some(location.into());
        self
    }

    /// Set Cloudflare AI Gateway configuration
    pub fn with_cloudflare_gateway(
        mut self,
        account_id: impl Into<String>,
        gateway_id: impl Into<String>,
    ) -> Self {
        self.extra.cf_account_id = Some(account_id.into());
        self.extra.cf_gateway_id = Some(gateway_id.into());
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PROVIDER ENUM (18+ PROVIDERS)
// ═══════════════════════════════════════════════════════════════════════════

/// Supported LLM providers (OpenAI-compatible where applicable)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum LlmProvider {
    // ─────────────────────────────────────────────────────────────────────────
    // TIER 1: Major Cloud Providers
    // ─────────────────────────────────────────────────────────────────────────
    /// Anthropic Claude models (native API)
    #[default]
    Anthropic,

    /// OpenAI GPT models
    OpenAI,

    /// Google Gemini via AI Studio (OpenAI-compatible)
    /// Base: https://generativelanguage.googleapis.com/v1beta/openai/
    GoogleGemini,

    /// Google Vertex AI (requires GCP auth)
    /// Base: https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi
    GoogleVertex,

    /// Azure OpenAI Service
    /// Base: https://{resource}.openai.azure.com/openai/deployments/{deployment}
    AzureOpenAI,

    /// AWS Bedrock (OpenAI-compatible via Converse API)
    /// Base: https://bedrock-runtime.{region}.amazonaws.com
    AWSBedrock,

    // ─────────────────────────────────────────────────────────────────────────
    // TIER 2: Specialized AI Providers
    // ─────────────────────────────────────────────────────────────────────────
    /// xAI Grok models
    /// Base: https://api.x.ai/v1
    XAI,

    /// Groq (ultra-fast inference)
    /// Base: https://api.groq.com/openai/v1
    Groq,

    /// Mistral AI
    /// Base: https://api.mistral.ai/v1/
    Mistral,

    /// DeepSeek (research-focused)
    /// Base: https://api.deepseek.com/v1
    DeepSeek,

    /// Cohere (enterprise NLP)
    /// Base: https://api.cohere.ai/v1
    Cohere,

    /// Perplexity (search-augmented)
    /// Base: https://api.perplexity.ai
    Perplexity,

    /// Cerebras (wafer-scale inference)
    /// Base: https://api.cerebras.ai/v1
    Cerebras,

    // ─────────────────────────────────────────────────────────────────────────
    // TIER 3: Inference Platforms
    // ─────────────────────────────────────────────────────────────────────────
    /// Together AI (open model hosting)
    /// Base: https://api.together.xyz/v1
    TogetherAI,

    /// Fireworks AI (fast open model inference)
    /// Base: https://api.fireworks.ai/inference/v1
    FireworksAI,

    /// Alibaba Qwen / DashScope
    /// Base: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    AlibabaQwen,

    // ─────────────────────────────────────────────────────────────────────────
    // TIER 4: Aggregation/Gateway Layers
    // ─────────────────────────────────────────────────────────────────────────
    /// OpenRouter (300+ models, automatic fallbacks)
    /// Base: https://openrouter.ai/api/v1
    OpenRouter,

    /// Cloudflare AI Gateway (unified endpoint)
    /// Base: https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/
    CloudflareAI,
}

impl LlmProvider {
    /// Get all available providers
    pub fn all() -> &'static [LlmProvider] {
        &[
            LlmProvider::Anthropic,
            LlmProvider::OpenAI,
            LlmProvider::GoogleGemini,
            LlmProvider::GoogleVertex,
            LlmProvider::AzureOpenAI,
            LlmProvider::AWSBedrock,
            LlmProvider::XAI,
            LlmProvider::Groq,
            LlmProvider::Mistral,
            LlmProvider::DeepSeek,
            LlmProvider::Cohere,
            LlmProvider::Perplexity,
            LlmProvider::Cerebras,
            LlmProvider::TogetherAI,
            LlmProvider::FireworksAI,
            LlmProvider::AlibabaQwen,
            LlmProvider::OpenRouter,
            LlmProvider::CloudflareAI,
        ]
    }

    /// Get environment variable name for API key
    pub fn env_var(&self) -> &'static str {
        match self {
            LlmProvider::Anthropic => "ANTHROPIC_API_KEY",
            LlmProvider::OpenAI => "OPENAI_API_KEY",
            LlmProvider::GoogleGemini => "GEMINI_API_KEY",
            LlmProvider::GoogleVertex => "GOOGLE_APPLICATION_CREDENTIALS",
            LlmProvider::AzureOpenAI => "AZURE_OPENAI_API_KEY",
            LlmProvider::AWSBedrock => "AWS_ACCESS_KEY_ID", // Also needs AWS_SECRET_ACCESS_KEY
            LlmProvider::XAI => "XAI_API_KEY",
            LlmProvider::Groq => "GROQ_API_KEY",
            LlmProvider::Mistral => "MISTRAL_API_KEY",
            LlmProvider::DeepSeek => "DEEPSEEK_API_KEY",
            LlmProvider::Cohere => "COHERE_API_KEY",
            LlmProvider::Perplexity => "PERPLEXITY_API_KEY",
            LlmProvider::Cerebras => "CEREBRAS_API_KEY",
            LlmProvider::TogetherAI => "TOGETHER_API_KEY",
            LlmProvider::FireworksAI => "FIREWORKS_API_KEY",
            LlmProvider::AlibabaQwen => "DASHSCOPE_API_KEY",
            LlmProvider::OpenRouter => "OPENROUTER_API_KEY",
            LlmProvider::CloudflareAI => "CLOUDFLARE_API_KEY",
        }
    }

    /// Get default base URL for provider
    pub fn default_base_url(&self) -> &'static str {
        match self {
            LlmProvider::Anthropic => "https://api.anthropic.com/v1",
            LlmProvider::OpenAI => "https://api.openai.com/v1",
            LlmProvider::GoogleGemini => "https://generativelanguage.googleapis.com/v1beta/openai",
            LlmProvider::GoogleVertex => "https://aiplatform.googleapis.com/v1", // Needs project/location
            LlmProvider::AzureOpenAI => "https://RESOURCE.openai.azure.com/openai", // Needs resource
            LlmProvider::AWSBedrock => "https://bedrock-runtime.us-east-1.amazonaws.com", // Needs region
            LlmProvider::XAI => "https://api.x.ai/v1",
            LlmProvider::Groq => "https://api.groq.com/openai/v1",
            LlmProvider::Mistral => "https://api.mistral.ai/v1",
            LlmProvider::DeepSeek => "https://api.deepseek.com/v1",
            LlmProvider::Cohere => "https://api.cohere.ai/v1",
            LlmProvider::Perplexity => "https://api.perplexity.ai",
            LlmProvider::Cerebras => "https://api.cerebras.ai/v1",
            LlmProvider::TogetherAI => "https://api.together.xyz/v1",
            LlmProvider::FireworksAI => "https://api.fireworks.ai/inference/v1",
            LlmProvider::AlibabaQwen => "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            LlmProvider::OpenRouter => "https://openrouter.ai/api/v1",
            LlmProvider::CloudflareAI => "https://gateway.ai.cloudflare.com/v1", // Needs account/gateway
        }
    }

    /// Get default model for provider (updated Dec 2025 - using standardized aliases)
    pub fn default_model(&self) -> &'static str {
        match self {
            // Tier 1: Major Cloud (Latest flagships)
            LlmProvider::Anthropic => "claude-opus-4-5", // Nov 2025 flagship
            LlmProvider::OpenAI => "gpt-5.1",            // Latest GPT (user confirmed)
            LlmProvider::GoogleGemini => "gemini-3.0-pro", // Nov 2025 #1 LMArena
            LlmProvider::GoogleVertex => "gemini-3.0-pro", // Same via Vertex
            LlmProvider::AzureOpenAI => "gpt-5.1",       // Via Azure
            LlmProvider::AWSBedrock => "anthropic.claude-opus-4-5-v1:0", // Via Bedrock

            // Tier 2: Specialized (Latest releases)
            LlmProvider::XAI => "grok-4.1", // Nov 2025, 2M context
            LlmProvider::Groq => "llama-3.3-70b-versatile", // Speed-optimized (unchanged)
            LlmProvider::Mistral => "mistral-large-3", // Dec 2025, 675B MoE
            LlmProvider::DeepSeek => "deepseek-v3.2", // Dec 2025, GPT-5 parity
            LlmProvider::Cohere => "command-a", // 111B flagship
            LlmProvider::Perplexity => "sonar-pro", // Latest search-augmented
            LlmProvider::Cerebras => "llama-4-scout", // 2600 tok/s (fastest)

            // Tier 3: Inference Platforms (Latest open models)
            LlmProvider::TogetherAI => "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            LlmProvider::FireworksAI => "accounts/fireworks/models/llama-v4-scout-instruct",
            LlmProvider::AlibabaQwen => "qwen3-max", // 1T+ params

            // Tier 4: Aggregation (Latest via gateway)
            LlmProvider::OpenRouter => "anthropic/claude-opus-4-5", // Route to best
            LlmProvider::CloudflareAI => "@cf/meta/llama-4-scout-instruct-fp8-fast",
        }
    }

    /// Check if provider uses native Anthropic API format
    pub fn is_anthropic_format(&self) -> bool {
        matches!(self, LlmProvider::Anthropic)
    }

    /// Check if provider uses OpenAI-compatible format
    pub fn is_openai_compatible(&self) -> bool {
        !self.is_anthropic_format()
    }

    /// Check if provider requires special authentication
    pub fn requires_special_auth(&self) -> bool {
        matches!(
            self,
            LlmProvider::AzureOpenAI | LlmProvider::AWSBedrock | LlmProvider::GoogleVertex
        )
    }

    /// Get provider display name
    pub fn display_name(&self) -> &'static str {
        match self {
            LlmProvider::Anthropic => "Anthropic",
            LlmProvider::OpenAI => "OpenAI",
            LlmProvider::GoogleGemini => "Google Gemini (AI Studio)",
            LlmProvider::GoogleVertex => "Google Vertex AI",
            LlmProvider::AzureOpenAI => "Azure OpenAI",
            LlmProvider::AWSBedrock => "AWS Bedrock",
            LlmProvider::XAI => "xAI (Grok)",
            LlmProvider::Groq => "Groq",
            LlmProvider::Mistral => "Mistral AI",
            LlmProvider::DeepSeek => "DeepSeek",
            LlmProvider::Cohere => "Cohere",
            LlmProvider::Perplexity => "Perplexity",
            LlmProvider::Cerebras => "Cerebras",
            LlmProvider::TogetherAI => "Together AI",
            LlmProvider::FireworksAI => "Fireworks AI",
            LlmProvider::AlibabaQwen => "Alibaba Qwen",
            LlmProvider::OpenRouter => "OpenRouter",
            LlmProvider::CloudflareAI => "Cloudflare AI",
        }
    }
}

impl std::fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// REQUEST/RESPONSE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Request to LLM
#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// System prompt
    pub system: Option<String>,

    /// User message (the prompt)
    pub prompt: String,

    /// Temperature override
    pub temperature: Option<f64>,

    /// Max tokens override
    pub max_tokens: Option<u32>,
}

impl LlmRequest {
    /// Create a simple request with just a prompt
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            system: None,
            prompt: prompt.into(),
            temperature: None,
            max_tokens: None,
        }
    }

    /// Add system prompt
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }
}

/// Response from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Generated text content
    pub content: String,

    /// Model used
    pub model: String,

    /// Finish reason
    pub finish_reason: FinishReason,

    /// Token usage
    pub usage: LlmUsage,

    /// Provider that handled the request
    #[serde(default)]
    pub provider: Option<LlmProvider>,
}

/// Why generation stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum FinishReason {
    /// Natural stop (end of response)
    #[default]
    Stop,
    /// Hit max tokens limit
    MaxTokens,
    /// Content filtered
    ContentFilter,
    /// Error occurred
    Error,
}

/// Token usage from LLM call
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmUsage {
    /// Input/prompt tokens
    pub input_tokens: u32,

    /// Output/completion tokens
    pub output_tokens: u32,

    /// Total tokens
    pub total_tokens: u32,
}

impl LlmUsage {
    /// Calculate cost in USD based on model pricing (approximate - Dec 2025 prices)
    pub fn cost_usd(&self, model: &str) -> f64 {
        // Approximate pricing per 1M tokens (input/output)
        let (input_price, output_price) = match model {
            // Anthropic (Dec 2025)
            m if m.contains("claude-opus-4-5") || m.contains("claude-opus-4.5") => (15.0, 75.0),
            m if m.contains("claude-opus-4") => (15.0, 75.0),
            m if m.contains("claude-sonnet-4") => (3.0, 15.0),
            m if m.contains("claude-3-5-sonnet") => (3.0, 15.0),
            m if m.contains("claude-3-haiku") => (0.25, 1.25),

            // OpenAI (Dec 2025)
            m if m.contains("gpt-5.1") || m.contains("gpt-5") => (5.0, 15.0),
            m if m.contains("gpt-4o") => (2.5, 10.0),
            m if m.contains("gpt-4-turbo") => (10.0, 30.0),
            m if m.contains("gpt-3.5") => (0.5, 1.5),
            m if m.contains("o1") || m.contains("o3") || m.contains("o4") => (15.0, 60.0),

            // Google Gemini (Dec 2025)
            m if m.contains("gemini-3.0-pro") => (1.5, 6.0),
            m if m.contains("gemini-2.5-pro") => (1.25, 5.0),
            m if m.contains("gemini-2.0-flash") || m.contains("gemini-2.5-flash") => (0.1, 0.4),
            m if m.contains("gemini-1.5-pro") => (1.25, 5.0),
            m if m.contains("gemini-1.5-flash") => (0.075, 0.3),

            // xAI Grok (Dec 2025)
            m if m.contains("grok-4.1") || m.contains("grok-4") => (3.0, 12.0),
            m if m.contains("grok-2") || m.contains("grok-3") => (2.0, 10.0),

            // Groq (ultra-fast, cheap)
            m if m.contains("llama") && m.contains("groq") => (0.05, 0.08),
            m if m.contains("mixtral") && m.contains("groq") => (0.24, 0.24),
            m if m.contains("llama-3.3-70b-versatile") => (0.59, 0.79),

            // Mistral (Dec 2025)
            m if m.contains("mistral-large-3") => (2.5, 7.5),
            m if m.contains("mistral-large") => (2.0, 6.0),
            m if m.contains("ministral") => (0.1, 0.3),
            m if m.contains("mistral-small") => (0.2, 0.6),
            m if m.contains("codestral") => (0.2, 0.6),

            // DeepSeek (Dec 2025 - extremely cheap)
            m if m.contains("deepseek-v3.2") || m.contains("deepseek-v3") => (0.27, 1.10),
            m if m.contains("deepseek") => (0.14, 0.28),

            // Llama 4 (Dec 2025)
            m if m.contains("llama-4") || m.contains("Llama-4") => (0.18, 0.59),
            m if m.contains("llama-3.3-70b") => (0.88, 0.88),
            m if m.contains("llama-3.1-405b") => (3.5, 3.5),

            // Qwen (Dec 2025)
            m if m.contains("qwen3") || m.contains("qwen-max") => (0.4, 1.2),
            m if m.contains("qwen") => (0.3, 0.3),

            // Cohere (Dec 2025)
            m if m.contains("command-a") => (2.5, 10.0),
            m if m.contains("command-r-plus") => (2.5, 10.0),
            m if m.contains("command-r") => (0.15, 0.6),

            // Perplexity (Dec 2025)
            m if m.contains("sonar-pro") => (3.0, 15.0),
            m if m.contains("sonar") => (1.0, 1.0),

            // Cerebras (Dec 2025 - speed premium)
            m if m.contains("cerebras") => (0.6, 0.6),

            // Default conservative estimate
            _ => (1.0, 3.0),
        };

        let input_cost = (self.input_tokens as f64 / 1_000_000.0) * input_price;
        let output_cost = (self.output_tokens as f64 / 1_000_000.0) * output_price;

        input_cost + output_cost
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LLM CLIENT TRAIT
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for LLM client implementations
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Generate a completion
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse>;

    /// Get the provider
    fn provider(&self) -> LlmProvider;

    /// Get the model name
    fn model(&self) -> &str;
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED LLM CLIENT
// ═══════════════════════════════════════════════════════════════════════════

/// Unified LLM client supporting 18+ providers
pub struct UnifiedLlmClient {
    config: LlmConfig,
    http_client: reqwest::Client,
}

impl UnifiedLlmClient {
    /// Create a new client with configuration
    pub fn new(config: LlmConfig) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| Error::Network(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            http_client,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CONVENIENCE CONSTRUCTORS
    // ─────────────────────────────────────────────────────────────────────────

    /// Create with default configuration (Anthropic Claude)
    pub fn default_anthropic() -> Result<Self> {
        Self::new(LlmConfig::default())
    }

    /// Create for OpenAI
    pub fn openai(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::OpenAI, model))
    }

    /// Create for OpenRouter (300+ models)
    pub fn openrouter(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::OpenRouter, model))
    }

    /// Create for Google Gemini (AI Studio)
    pub fn gemini(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::GoogleGemini, model))
    }

    /// Create for xAI Grok
    pub fn grok(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::XAI, model))
    }

    /// Create for Groq (ultra-fast)
    pub fn groq(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::Groq, model))
    }

    /// Create for Mistral
    pub fn mistral(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::Mistral, model))
    }

    /// Create for DeepSeek
    pub fn deepseek(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::DeepSeek, model))
    }

    /// Create for Together AI
    pub fn together(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::TogetherAI, model))
    }

    /// Create for Fireworks AI
    pub fn fireworks(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::FireworksAI, model))
    }

    /// Create for Alibaba Qwen
    pub fn qwen(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::AlibabaQwen, model))
    }

    /// Create for Cohere
    pub fn cohere(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::Cohere, model))
    }

    /// Create for Perplexity
    pub fn perplexity(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::Perplexity, model))
    }

    /// Create for Cerebras
    pub fn cerebras(model: impl Into<String>) -> Result<Self> {
        Self::new(LlmConfig::for_provider(LlmProvider::Cerebras, model))
    }

    /// Create for Azure OpenAI
    pub fn azure(
        resource: impl Into<String>,
        deployment: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self> {
        Self::new(
            LlmConfig::for_provider(LlmProvider::AzureOpenAI, model)
                .with_azure(resource, deployment),
        )
    }

    /// Create for Cloudflare AI Gateway
    pub fn cloudflare_gateway(
        account_id: impl Into<String>,
        gateway_id: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self> {
        Self::new(
            LlmConfig::for_provider(LlmProvider::CloudflareAI, model)
                .with_cloudflare_gateway(account_id, gateway_id),
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // API KEY & URL RESOLUTION
    // ─────────────────────────────────────────────────────────────────────────

    /// Get API key from config or environment
    fn get_api_key(&self) -> Result<String> {
        if let Some(key) = &self.config.api_key {
            return Ok(key.clone());
        }

        let env_var = self.config.provider.env_var();
        std::env::var(env_var).map_err(|_| {
            Error::Config(format!(
                "API key not found. Set {} or provide in config",
                env_var
            ))
        })
    }

    /// Get base URL for provider (with dynamic construction where needed)
    fn get_base_url(&self) -> Result<String> {
        if let Some(url) = &self.config.base_url {
            return Ok(url.clone());
        }

        match self.config.provider {
            LlmProvider::AzureOpenAI => {
                let resource = self
                    .config
                    .extra
                    .azure_resource
                    .as_ref()
                    .ok_or_else(|| Error::Config("Azure resource name required".to_string()))?;
                let deployment =
                    self.config.extra.azure_deployment.as_ref().ok_or_else(|| {
                        Error::Config("Azure deployment name required".to_string())
                    })?;
                Ok(format!(
                    "https://{}.openai.azure.com/openai/deployments/{}",
                    resource, deployment
                ))
            }
            LlmProvider::GoogleVertex => {
                let project = self
                    .config
                    .extra
                    .gcp_project
                    .as_ref()
                    .ok_or_else(|| Error::Config("GCP project ID required".to_string()))?;
                let default_location = "us-central1".to_string();
                let location = self
                    .config
                    .extra
                    .gcp_location
                    .as_ref()
                    .unwrap_or(&default_location);
                Ok(format!(
                    "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models",
                    location, project, location
                ))
            }
            LlmProvider::AWSBedrock => {
                let default_region = "us-east-1".to_string();
                let region = self
                    .config
                    .extra
                    .aws_region
                    .as_ref()
                    .unwrap_or(&default_region);
                Ok(format!("https://bedrock-runtime.{}.amazonaws.com", region))
            }
            LlmProvider::CloudflareAI => {
                let account_id =
                    self.config.extra.cf_account_id.as_ref().ok_or_else(|| {
                        Error::Config("Cloudflare account ID required".to_string())
                    })?;
                let gateway_id =
                    self.config.extra.cf_gateway_id.as_ref().ok_or_else(|| {
                        Error::Config("Cloudflare gateway ID required".to_string())
                    })?;
                Ok(format!(
                    "https://gateway.ai.cloudflare.com/v1/{}/{}/openai",
                    account_id, gateway_id
                ))
            }
            _ => Ok(self.config.provider.default_base_url().to_string()),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // API CALLS
    // ─────────────────────────────────────────────────────────────────────────

    /// Call Anthropic API (native format)
    async fn call_anthropic(&self, request: LlmRequest) -> Result<LlmResponse> {
        let api_key = self.get_api_key()?;
        let base_url = self.get_base_url()?;
        let url = format!("{}/messages", base_url);

        let messages = vec![serde_json::json!({
            "role": "user",
            "content": request.prompt
        })];

        let body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": request.max_tokens.unwrap_or(self.config.max_tokens),
            "temperature": request.temperature.unwrap_or(self.config.temperature),
            "system": request.system.unwrap_or_else(|| "You are a helpful assistant.".to_string()),
            "messages": messages
        });

        let response = self
            .http_client
            .post(&url)
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Network(format!("Anthropic API request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(Error::Network(format!(
                "Anthropic API error {}: {}",
                status, text
            )));
        }

        let json: AnthropicResponse = response.json().await.map_err(|e| Error::Parse {
            message: format!("Failed to parse Anthropic response: {}", e),
        })?;

        Ok(LlmResponse {
            content: json
                .content
                .first()
                .map(|c| c.text.clone())
                .unwrap_or_default(),
            model: json.model,
            finish_reason: match json.stop_reason.as_deref() {
                Some("end_turn") => FinishReason::Stop,
                Some("max_tokens") => FinishReason::MaxTokens,
                _ => FinishReason::Stop,
            },
            usage: LlmUsage {
                input_tokens: json.usage.input_tokens,
                output_tokens: json.usage.output_tokens,
                total_tokens: json.usage.input_tokens + json.usage.output_tokens,
            },
            provider: Some(LlmProvider::Anthropic),
        })
    }

    /// Call OpenAI-compatible API (works for most providers)
    async fn call_openai_compatible(&self, request: LlmRequest) -> Result<LlmResponse> {
        let api_key = self.get_api_key()?;
        let base_url = self.get_base_url()?;
        let url = format!("{}/chat/completions", base_url);

        let mut messages = Vec::new();

        if let Some(system) = &request.system {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }

        messages.push(serde_json::json!({
            "role": "user",
            "content": request.prompt
        }));

        let body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": request.max_tokens.unwrap_or(self.config.max_tokens),
            "temperature": request.temperature.unwrap_or(self.config.temperature),
            "messages": messages
        });

        let mut req = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json");

        // Provider-specific headers
        match self.config.provider {
            LlmProvider::OpenRouter => {
                req = req
                    .header("HTTP-Referer", "https://reasonkit.sh")
                    .header("X-Title", "ReasonKit ThinkTool");
            }
            LlmProvider::AzureOpenAI => {
                req = req
                    .header("api-key", &api_key)
                    .header("api-version", "2024-02-15-preview");
            }
            LlmProvider::GoogleGemini => {
                // Google AI Studio uses x-goog-api-key
                req = req.header("x-goog-api-key", &api_key);
            }
            _ => {}
        }

        let response = req.json(&body).send().await.map_err(|e| {
            Error::Network(format!(
                "{} API request failed: {}",
                self.config.provider, e
            ))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(Error::Network(format!(
                "{} API error {}: {}",
                self.config.provider, status, text
            )));
        }

        let json: OpenAIResponse = response.json().await.map_err(|e| Error::Parse {
            message: format!("Failed to parse {} response: {}", self.config.provider, e),
        })?;

        let choice = json.choices.first().ok_or_else(|| Error::Parse {
            message: "No choices in response".to_string(),
        })?;

        Ok(LlmResponse {
            content: choice.message.content.clone().unwrap_or_default(),
            model: json.model,
            finish_reason: match choice.finish_reason.as_deref() {
                Some("stop") => FinishReason::Stop,
                Some("length") => FinishReason::MaxTokens,
                Some("content_filter") => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
            },
            usage: LlmUsage {
                input_tokens: json.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
                output_tokens: json
                    .usage
                    .as_ref()
                    .map(|u| u.completion_tokens)
                    .unwrap_or(0),
                total_tokens: json.usage.as_ref().map(|u| u.total_tokens).unwrap_or(0),
            },
            provider: Some(self.config.provider),
        })
    }
}

#[async_trait]
impl LlmClient for UnifiedLlmClient {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse> {
        if self.config.provider.is_anthropic_format() {
            self.call_anthropic(request).await
        } else {
            self.call_openai_compatible(request).await
        }
    }

    fn provider(&self) -> LlmProvider {
        self.config.provider
    }

    fn model(&self) -> &str {
        &self.config.model
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PROVIDER DISCOVERY & MULTI-PROVIDER SUPPORT
// ═══════════════════════════════════════════════════════════════════════════

/// Discover available providers based on environment variables
pub fn discover_available_providers() -> Vec<LlmProvider> {
    LlmProvider::all()
        .iter()
        .filter(|p| {
            // Skip providers needing special auth for simple discovery
            if p.requires_special_auth() {
                return false;
            }
            std::env::var(p.env_var()).is_ok()
        })
        .copied()
        .collect()
}

/// Create a client for the first available provider
pub fn create_available_client() -> Result<UnifiedLlmClient> {
    let available = discover_available_providers();

    if available.is_empty() {
        return Err(Error::Config(
            "No LLM provider API keys found. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, etc.".to_string()
        ));
    }

    let provider = available[0];
    UnifiedLlmClient::new(LlmConfig {
        provider,
        model: provider.default_model().to_string(),
        ..Default::default()
    })
}

/// Provider info for documentation/UI
#[derive(Debug, Clone, Serialize)]
pub struct ProviderInfo {
    /// Provider identifier
    pub id: LlmProvider,
    /// Human-readable provider name
    pub name: &'static str,
    /// Environment variable for API key
    pub env_var: &'static str,
    /// Default model for this provider
    pub default_model: &'static str,
    /// Base URL for API calls
    pub base_url: &'static str,
    /// Whether API key is currently available
    pub is_available: bool,
}

/// Get info for all providers
pub fn get_provider_info() -> Vec<ProviderInfo> {
    LlmProvider::all()
        .iter()
        .map(|p| ProviderInfo {
            id: *p,
            name: p.display_name(),
            env_var: p.env_var(),
            default_model: p.default_model(),
            base_url: p.default_base_url(),
            is_available: std::env::var(p.env_var()).is_ok(),
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// API RESPONSE TYPES
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    model: String,
    content: Vec<AnthropicContent>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_count() {
        assert_eq!(LlmProvider::all().len(), 18);
    }

    #[test]
    fn test_provider_env_vars() {
        assert_eq!(LlmProvider::Anthropic.env_var(), "ANTHROPIC_API_KEY");
        assert_eq!(LlmProvider::OpenAI.env_var(), "OPENAI_API_KEY");
        assert_eq!(LlmProvider::GoogleGemini.env_var(), "GEMINI_API_KEY");
        assert_eq!(LlmProvider::XAI.env_var(), "XAI_API_KEY");
        assert_eq!(LlmProvider::Groq.env_var(), "GROQ_API_KEY");
        assert_eq!(LlmProvider::Mistral.env_var(), "MISTRAL_API_KEY");
        assert_eq!(LlmProvider::DeepSeek.env_var(), "DEEPSEEK_API_KEY");
        assert_eq!(LlmProvider::TogetherAI.env_var(), "TOGETHER_API_KEY");
        assert_eq!(LlmProvider::FireworksAI.env_var(), "FIREWORKS_API_KEY");
        assert_eq!(LlmProvider::AlibabaQwen.env_var(), "DASHSCOPE_API_KEY");
        assert_eq!(LlmProvider::OpenRouter.env_var(), "OPENROUTER_API_KEY");
        assert_eq!(LlmProvider::CloudflareAI.env_var(), "CLOUDFLARE_API_KEY");
    }

    #[test]
    fn test_provider_base_urls() {
        assert_eq!(
            LlmProvider::Anthropic.default_base_url(),
            "https://api.anthropic.com/v1"
        );
        assert_eq!(
            LlmProvider::OpenAI.default_base_url(),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            LlmProvider::Groq.default_base_url(),
            "https://api.groq.com/openai/v1"
        );
        assert_eq!(LlmProvider::XAI.default_base_url(), "https://api.x.ai/v1");
        assert_eq!(
            LlmProvider::Mistral.default_base_url(),
            "https://api.mistral.ai/v1"
        );
        assert_eq!(
            LlmProvider::DeepSeek.default_base_url(),
            "https://api.deepseek.com/v1"
        );
        assert_eq!(
            LlmProvider::TogetherAI.default_base_url(),
            "https://api.together.xyz/v1"
        );
    }

    #[test]
    fn test_provider_compatibility() {
        assert!(LlmProvider::Anthropic.is_anthropic_format());
        assert!(!LlmProvider::OpenAI.is_anthropic_format());

        assert!(!LlmProvider::Anthropic.is_openai_compatible());
        assert!(LlmProvider::OpenAI.is_openai_compatible());
        assert!(LlmProvider::Groq.is_openai_compatible());
        assert!(LlmProvider::XAI.is_openai_compatible());
        assert!(LlmProvider::GoogleGemini.is_openai_compatible());
    }

    #[test]
    fn test_special_auth_providers() {
        assert!(LlmProvider::AzureOpenAI.requires_special_auth());
        assert!(LlmProvider::AWSBedrock.requires_special_auth());
        assert!(LlmProvider::GoogleVertex.requires_special_auth());
        assert!(!LlmProvider::OpenAI.requires_special_auth());
        assert!(!LlmProvider::Groq.requires_special_auth());
    }

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::default();
        assert_eq!(config.provider, LlmProvider::Anthropic);
        assert_eq!(config.model, "claude-opus-4-5");
        assert_eq!(config.temperature, 0.7);
    }

    #[test]
    fn test_llm_config_builder() {
        let config = LlmConfig::for_provider(LlmProvider::Groq, "llama-3.3-70b-versatile")
            .with_temperature(0.5)
            .with_max_tokens(4000);

        assert_eq!(config.provider, LlmProvider::Groq);
        assert_eq!(config.model, "llama-3.3-70b-versatile");
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, 4000);
    }

    #[test]
    fn test_azure_config() {
        let config = LlmConfig::for_provider(LlmProvider::AzureOpenAI, "gpt-4o")
            .with_azure("my-resource", "my-deployment");

        assert_eq!(config.extra.azure_resource, Some("my-resource".to_string()));
        assert_eq!(
            config.extra.azure_deployment,
            Some("my-deployment".to_string())
        );
    }

    #[test]
    fn test_cloudflare_config() {
        let config = LlmConfig::for_provider(LlmProvider::CloudflareAI, "@cf/meta/llama-3.3-70b")
            .with_cloudflare_gateway("account123", "gateway456");

        assert_eq!(config.extra.cf_account_id, Some("account123".to_string()));
        assert_eq!(config.extra.cf_gateway_id, Some("gateway456".to_string()));
    }

    #[test]
    fn test_llm_request_builder() {
        let request = LlmRequest::new("Hello")
            .with_system("You are helpful")
            .with_temperature(0.5)
            .with_max_tokens(100);

        assert_eq!(request.prompt, "Hello");
        assert_eq!(request.system, Some("You are helpful".to_string()));
        assert_eq!(request.temperature, Some(0.5));
        assert_eq!(request.max_tokens, Some(100));
    }

    #[test]
    fn test_cost_calculation() {
        let usage = LlmUsage {
            input_tokens: 1000,
            output_tokens: 500,
            total_tokens: 1500,
        };

        // Claude Sonnet pricing: $3/1M input, $15/1M output
        let cost = usage.cost_usd("claude-sonnet-4");
        assert!(cost > 0.0);
        assert!(cost < 0.02);

        // GPT-3.5 should be cheaper
        let cost_gpt35 = usage.cost_usd("gpt-3.5-turbo");
        assert!(cost_gpt35 < cost);

        // Groq should be very cheap
        let cost_groq = usage.cost_usd("llama-groq");
        assert!(cost_groq < cost_gpt35);

        // DeepSeek should be cheap
        let cost_deepseek = usage.cost_usd("deepseek-chat");
        assert!(cost_deepseek < cost);
    }

    #[test]
    fn test_client_creation() {
        let config = LlmConfig::default();
        let client = UnifiedLlmClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_provider_info() {
        let info = get_provider_info();
        assert_eq!(info.len(), 18);

        let anthropic = info
            .iter()
            .find(|i| i.id == LlmProvider::Anthropic)
            .unwrap();
        assert_eq!(anthropic.name, "Anthropic");
        assert_eq!(anthropic.env_var, "ANTHROPIC_API_KEY");
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(LlmProvider::Anthropic.to_string(), "Anthropic");
        assert_eq!(LlmProvider::XAI.to_string(), "xAI (Grok)");
        assert_eq!(
            LlmProvider::GoogleGemini.to_string(),
            "Google Gemini (AI Studio)"
        );
    }
}
