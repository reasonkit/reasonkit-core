# ReasonKit Integration Guide

> Complete guide for integrating ReasonKit into your applications, workflows, and AI agent systems.

**Version:** 0.1.0 | **Last Updated:** 2026-01-01

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Core Integration Patterns](#core-integration-patterns)
  - [CLI Integration](#cli-integration)
  - [Rust Library Integration](#rust-library-integration)
  - [Python Integration](#python-integration)
- [LLM Provider Setup](#llm-provider-setup)
- [MCP Server Integration](#mcp-server-integration)
- [Memory System Configuration](#memory-system-configuration)
- [Web Sensing Setup](#web-sensing-setup)
- [Framework Integrations](#framework-integrations)
  - [LangChain Integration](#langchain-integration)
  - [CrewAI Integration](#crewai-integration)
  - [AutoGen Integration](#autogen-integration)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 30-Second Setup

```bash
# Install ReasonKit
curl -fsSL https://reasonkit.sh/install | bash

# Verify installation
rk-core --version

# Run your first reasoning query
rk-core think --profile balanced "Should we migrate to microservices?"
```

### Minimal Rust Integration

```rust
use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let executor = ProtocolExecutor::new()?;

    let result = executor.execute(
        "gigathink",
        ProtocolInput::query("What factors drive startup success?")
    ).await?;

    println!("Confidence: {:.2}", result.confidence);
    for step in &result.steps {
        println!("- {}", step.as_text().unwrap_or_default());
    }

    Ok(())
}
```

### Minimal Python Integration

```python
from reasonkit import Reasoner, Profile

# Create reasoner instance
r = Reasoner()

# Run a balanced analysis
result = r.think_with_profile(Profile.Balanced, "Should we pivot our product?")
print(f"Confidence: {result.confidence:.1%}")
```

---

## Installation

### Primary Method (Universal Installer)

```bash
curl -fsSL https://reasonkit.sh/install | bash
```

This installs the `rk-core` binary to `~/.local/bin/`.

### Cargo (Rust Developers)

```bash
# Install CLI only
cargo install reasonkit-core

# With memory features (RAG/KB)
cargo install reasonkit-core --features memory

# With all features
cargo install reasonkit-core --all-features
```

### Build from Source

```bash
git clone https://github.com/reasonkit/reasonkit-core
cd reasonkit-core

# Release build
cargo build --release

# With memory infrastructure
cargo build --release --features memory

# Run tests to verify
cargo test --all-features
```

### Python Bindings

```bash
# Build Python bindings (requires maturin)
pip install maturin
cd reasonkit-core
maturin develop --release

# Or install from PyPI (when published)
pip install reasonkit
```

### Verify Installation

```bash
# Check CLI is working
rk-core --version

# List available protocols
rk-core think --list

# Test with mock LLM (no API key needed)
rk-core think --mock --profile quick "Test query"
```

---

## Core Integration Patterns

### CLI Integration

The CLI is the simplest integration path. Use it from shell scripts, CI/CD pipelines, or any language that can spawn processes.

#### Basic Usage

```bash
# Quick analysis (2-step chain)
rk-core think --profile quick "Is this email phishing?"

# Balanced analysis (5-step chain)
rk-core think --profile balanced "Should we use microservices?"

# Deep analysis with full chain
rk-core think --profile deep "Design A/B test for feature X"

# Maximum rigor (paranoid mode)
rk-core think --profile paranoid "Validate this cryptographic implementation"
```

#### Run Specific Protocols

```bash
# GigaThink: Generate 10+ perspectives
rk-core think --protocol gigathink "Analyze market trends"

# LaserLogic: Detect logical fallacies
rk-core think --protocol laserlogic "Evaluate this argument"

# BedRock: First principles decomposition
rk-core think --protocol bedrock "Break down the problem"

# ProofGuard: Multi-source verification
rk-core think --protocol proofguard "Verify this claim"

# BrutalHonesty: Adversarial self-critique
rk-core think --protocol brutalhonesty "Find flaws in this plan"
```

#### JSON Output for Scripting

```bash
# Get structured JSON output
rk-core think --format json --profile balanced "Analyze this decision" | jq .

# Save to file
rk-core think --format json --profile balanced "Query" > analysis.json
```

#### Using Different LLM Providers

```bash
# Anthropic Claude (default)
rk-core think --provider anthropic "Your query"

# OpenAI GPT
rk-core think --provider openai --model gpt-4o "Your query"

# Google Gemini
rk-core think --provider gemini --model gemini-2.0-flash "Your query"

# DeepSeek
rk-core think --provider deepseek "Your query"

# Groq (ultra-fast inference)
rk-core think --provider groq --model llama-3.3-70b-versatile "Your query"

# OpenRouter (300+ models)
rk-core think --provider openrouter --model anthropic/claude-3.5-sonnet "Your query"
```

#### Shell Script Integration

```bash
#!/bin/bash
# analyze_decision.sh - Structured decision analysis

DECISION="$1"

if [ -z "$DECISION" ]; then
    echo "Usage: $0 \"decision to analyze\""
    exit 1
fi

# Run analysis and capture JSON output
RESULT=$(rk-core think --format json --profile balanced "$DECISION")

# Extract confidence score
CONFIDENCE=$(echo "$RESULT" | jq -r '.confidence')

# Extract verdict if available
VERDICT=$(echo "$RESULT" | jq -r '.steps[-1].verdict // "no verdict"')

echo "Decision Analysis"
echo "================="
echo "Query: $DECISION"
echo "Confidence: $CONFIDENCE"
echo "Verdict: $VERDICT"
echo ""
echo "Full analysis saved to analysis_$(date +%Y%m%d_%H%M%S).json"
echo "$RESULT" > "analysis_$(date +%Y%m%d_%H%M%S).json"
```

### Rust Library Integration

Add ReasonKit to your `Cargo.toml`:

```toml
[dependencies]
reasonkit-core = "0.1"
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"

# Optional: Memory infrastructure for RAG/KB
reasonkit-core = { version = "0.1", features = ["memory"] }
```

#### Basic Protocol Execution

```rust
use reasonkit::thinktool::{
    ProtocolExecutor, ProtocolInput, ExecutorConfig
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create executor with default config
    let executor = ProtocolExecutor::new()?;

    // Execute GigaThink protocol
    let result = executor.execute(
        "gigathink",
        ProtocolInput::query("What makes a successful startup?")
    ).await?;

    println!("=== GigaThink Analysis ===");
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("\nPerspectives:");
    for (i, step) in result.steps.iter().enumerate() {
        if let Some(text) = step.as_text() {
            println!("{}. {}", i + 1, text);
        }
    }

    Ok(())
}
```

#### Custom Configuration

```rust
use reasonkit::thinktool::{
    ProtocolExecutor, ProtocolInput, ExecutorConfig,
    llm::{LlmConfig, LlmProvider}
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Custom configuration
    let mut config = ExecutorConfig::default();

    // LLM settings
    config.llm.provider = LlmProvider::Anthropic;
    config.llm.model = "claude-sonnet-4-20250514".to_string();
    config.llm.temperature = 0.7;
    config.llm.max_tokens = 2000;

    // Execution settings
    config.timeout_secs = 120;
    config.verbose = true;
    config.save_traces = true;
    config.trace_dir = Some(PathBuf::from("./traces"));

    // Enable parallel step execution
    config.enable_parallel = true;
    config.max_concurrent_steps = 4;

    // Create executor with custom config
    let executor = ProtocolExecutor::with_config(config)?;

    // Execute with profile
    let result = executor.execute_profile(
        "balanced",
        ProtocolInput::query("Should we adopt Kubernetes?")
    ).await?;

    println!("Result: {:?}", result);

    Ok(())
}
```

#### Using the Unified LLM Client

```rust
use reasonkit::thinktool::{UnifiedLlmClient, LlmRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create client for specific provider
    let client = UnifiedLlmClient::default_anthropic()?;

    // Or use Groq for ultra-fast inference
    // let client = UnifiedLlmClient::groq("llama-3.3-70b-versatile")?;

    // Or use OpenRouter for 300+ models
    // let client = UnifiedLlmClient::openrouter("anthropic/claude-sonnet-4")?;

    // Make a request
    let response = client.complete(
        LlmRequest::new("Explain the CAP theorem in distributed systems")
            .with_system("You are a distributed systems expert. Be concise.")
            .with_temperature(0.5)
            .with_max_tokens(500)
    ).await?;

    println!("Response: {}", response.content);
    println!("Tokens used: {:?}", response.usage);

    Ok(())
}
```

#### Provider Auto-Discovery

```rust
use reasonkit::thinktool::{
    discover_available_providers,
    create_available_client
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Find all providers with API keys configured
    let available = discover_available_providers();

    println!("Available providers:");
    for provider in &available {
        println!("  - {:?}", provider);
    }

    // Create client using first available provider
    let client = create_available_client()?;

    // Use the client...

    Ok(())
}
```

#### Streaming Responses

```rust
use reasonkit::thinktool::{UnifiedLlmClient, LlmRequest};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = UnifiedLlmClient::default_anthropic()?;

    let mut stream = client.complete_stream(
        LlmRequest::new("Write a short story about AI")
            .with_stream(true)
    ).await?;

    print!("Story: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => print!("{}", text),
            Err(e) => eprintln!("\nError: {}", e),
        }
    }
    println!();

    Ok(())
}
```

### Python Integration

Build the Python bindings first:

```bash
cd reasonkit-core
maturin develop --release
```

#### Basic Usage

```python
from reasonkit import Reasoner, Profile

# Create reasoner (defaults to real LLM)
r = Reasoner()

# Or use mock mode for testing
r_mock = Reasoner(use_mock=True)

# Run individual ThinkTools
result = r.run_gigathink("What factors drive startup success?")
print(f"Perspectives: {len(result.perspectives())}")
for p in result.perspectives():
    print(f"  - {p}")

# Run with profile
result = r.think_with_profile(Profile.Balanced, "Should we migrate to microservices?")
print(f"Confidence: {result.confidence:.1%}")
```

#### Convenience Functions

```python
from reasonkit import (
    run_gigathink,
    run_laserlogic,
    run_bedrock,
    run_proofguard,
    run_brutalhonesty,
    quick_think,
    balanced_think,
    deep_think,
    paranoid_think,
    version
)

# Check version
print(f"ReasonKit v{version()}")

# Quick ThinkTool execution (no Reasoner needed)
result = run_gigathink("Analyze market trends", use_mock=True)

# Profile-based execution
result = balanced_think("Should we pivot our product?")
print(f"Confidence: {result.confidence:.1%}")

# High-stakes validation
result = paranoid_think("Validate this security implementation")
```

#### Error Handling

```python
from reasonkit import Reasoner, Profile, ReasonerError

try:
    r = Reasoner()
    result = r.think_with_profile(Profile.Balanced, "Query")
except ReasonerError as e:
    print(f"ReasonKit error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## LLM Provider Setup

ReasonKit supports 18+ LLM providers. Configure via environment variables:

### Anthropic (Default)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

### Google Gemini

```bash
export GOOGLE_API_KEY="..."
# Or for Vertex AI:
export GOOGLE_PROJECT_ID="your-project"
export GOOGLE_LOCATION="us-central1"
```

### DeepSeek

```bash
export DEEPSEEK_API_KEY="..."
```

### Groq (Ultra-Fast Inference)

```bash
export GROQ_API_KEY="..."
```

### OpenRouter (300+ Models)

```bash
export OPENROUTER_API_KEY="..."
```

### xAI (Grok)

```bash
export XAI_API_KEY="..."
```

### AWS Bedrock

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

### Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="your-deployment"
```

### Complete Provider Reference

| Provider | Env Variable | Default Model |
|----------|-------------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| OpenAI | `OPENAI_API_KEY` | gpt-4o |
| Google Gemini | `GOOGLE_API_KEY` | gemini-2.0-flash |
| Vertex AI | `GOOGLE_PROJECT_ID` | gemini-2.0-flash |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | gpt-4o |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` | anthropic.claude-sonnet-4-v1:0 |
| xAI | `XAI_API_KEY` | grok-2 |
| Groq | `GROQ_API_KEY` | llama-3.3-70b-versatile |
| Mistral | `MISTRAL_API_KEY` | mistral-large-latest |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
| Cohere | `COHERE_API_KEY` | command-r-plus |
| Perplexity | `PERPLEXITY_API_KEY` | sonar-pro |
| Cerebras | `CEREBRAS_API_KEY` | llama-3.3-70b |
| Together AI | `TOGETHER_API_KEY` | meta-llama/Llama-3.3-70B |
| Fireworks AI | `FIREWORKS_API_KEY` | llama-v3p3-70b-instruct |
| Alibaba Qwen | `QWEN_API_KEY` | qwen-max |
| Cloudflare AI | `CLOUDFLARE_API_KEY` | @cf/meta/llama-3.3-70b |
| OpenRouter | `OPENROUTER_API_KEY` | anthropic/claude-3.5-sonnet |

---

## MCP Server Integration

ReasonKit implements the Model Context Protocol (MCP) for seamless AI agent integration.

### Starting the MCP Server

```bash
# Start MCP server
rk-core serve-mcp

# The server listens on stdio by default
```

### Claude Desktop Integration

Add to your Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "reasonkit": {
      "command": "rk-core",
      "args": ["serve-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

### MCP Client Integration (Rust)

```rust
use reasonkit::mcp::{McpClient, McpClientConfig};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Connect to MCP server
    let config = McpClientConfig {
        server_command: "rk-core".to_string(),
        server_args: vec!["serve-mcp".to_string()],
        ..Default::default()
    };

    let client = McpClient::connect(config).await?;

    // List available tools
    let tools = client.list_tools().await?;
    println!("Available tools: {:?}", tools);

    // Call a tool
    let mut args = HashMap::new();
    args.insert("query".to_string(), serde_json::json!("Analyze this decision"));
    args.insert("profile".to_string(), serde_json::json!("balanced"));

    let result = client.call_tool("think", args).await?;
    println!("Result: {:?}", result);

    Ok(())
}
```

### MCP Tools Available

| Tool | Description | Arguments |
|------|-------------|-----------|
| `think` | Execute reasoning protocol | `query`, `profile`, `protocol` |
| `verify` | Triangulate claims with 3+ sources | `claim`, `sources` |
| `web` | Deep research with web + KB | `query`, `depth` |

---

## Memory System Configuration

Enable the `memory` feature for RAG and knowledge base capabilities:

```toml
[dependencies]
reasonkit-core = { version = "0.1", features = ["memory"] }
```

### Basic Memory Setup

```rust
use reasonkit::storage::{DualLayerMemory, DualLayerConfig, MemoryEntry};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create dual-layer memory with default config
    let config = DualLayerConfig::default();
    let memory = DualLayerMemory::new(config).await?;

    // Store a memory entry
    let entry = MemoryEntry::new("Important context about the project")
        .with_importance(0.8)
        .with_tags(vec!["project".to_string(), "context".to_string()]);

    let id = memory.store(entry).await?;
    println!("Stored with ID: {}", id);

    // Retrieve context for a query
    let results = memory.retrieve_context("project context", 10).await?;
    for result in results {
        println!("Score: {:.2}, Content: {}", result.score, result.content);
    }

    Ok(())
}
```

### Embedding Configuration

```rust
use reasonkit::embedding::{EmbeddingService, EmbeddingConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Remote embeddings (OpenAI)
    let config = EmbeddingConfig {
        provider: "openai".to_string(),
        model: "text-embedding-3-small".to_string(),
        ..Default::default()
    };
    let service = EmbeddingService::new(config).await?;

    // Generate embeddings
    let embeddings = service.embed(&["Hello world", "Goodbye world"]).await?;

    Ok(())
}
```

### Hybrid Search (Dense + Sparse)

```rust
use reasonkit::retrieval::{HybridRetriever, RetrievalConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = RetrievalConfig {
        top_k: 10,
        min_score: 0.1,
        alpha: 0.7,  // Dense weight (0.7 dense, 0.3 sparse)
        use_raptor: false,
        rerank: true,
    };

    let retriever = HybridRetriever::new(config).await?;

    let results = retriever.search("query about machine learning", 10).await?;

    for result in results {
        println!("Score: {:.3}, Text: {:.100}...", result.score, result.text);
    }

    Ok(())
}
```

### RAPTOR Hierarchical Retrieval

```rust
use reasonkit::raptor::{RaptorTree, RaptorConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = RaptorConfig {
        max_depth: 3,
        cluster_size: 10,
        summarization_model: "claude-sonnet-4".to_string(),
    };

    let mut tree = RaptorTree::new(config);

    // Build tree from documents
    tree.build_from_documents(&documents).await?;

    // Search with hierarchical context
    let results = tree.search("complex query", 5).await?;

    Ok(())
}
```

---

## Web Sensing Setup

ReasonKit Web provides browser automation and web content extraction.

### Installation

```bash
# Build reasonkit-web
cd reasonkit-web
cargo build --release

# Run the MCP server
./target/release/reasonkit-web
```

### Configuration

```bash
# Chrome/Chromium path (optional, auto-detected)
export CHROME_PATH="/usr/bin/chromium-browser"

# Headless mode (default: true)
export HEADLESS=true

# Request timeout
export REQUEST_TIMEOUT=30000
```

### Integration with ReasonKit Core

```rust
use reasonkit::web_interface::WebInterface;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let web = WebInterface::connect().await?;

    // Navigate and capture
    let page = web.navigate("https://example.com").await?;

    // Extract content
    let content = page.extract_content().await?;
    println!("Title: {}", content.title);
    println!("Text: {:.500}...", content.text);

    // Take screenshot
    let screenshot = page.screenshot().await?;
    std::fs::write("screenshot.png", screenshot)?;

    Ok(())
}
```

---

## Framework Integrations

### LangChain Integration

```python
from langchain.tools import Tool
from reasonkit import Reasoner, Profile

def reason_with_profile(query: str, profile: str = "balanced") -> str:
    """Run ReasonKit reasoning on a query."""
    r = Reasoner()
    profile_enum = getattr(Profile, profile.capitalize())
    result = r.think_with_profile(profile_enum, query)
    return f"Confidence: {result.confidence:.1%}\n\n{result.summary()}"

# Create LangChain tool
reasonkit_tool = Tool(
    name="ReasonKit",
    func=reason_with_profile,
    description="Use structured reasoning to analyze complex decisions. "
                "Input should be a question or decision to analyze."
)

# Use in a LangChain agent
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools=[reasonkit_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("Should we migrate from PostgreSQL to MongoDB?")
```

### CrewAI Integration

```python
from crewai import Agent, Task, Crew
from reasonkit import Reasoner, Profile
import subprocess
import json

class ReasonKitTool:
    """ReasonKit reasoning tool for CrewAI agents."""

    def __init__(self):
        self.name = "ReasonKit Analyzer"
        self.description = "Analyze decisions with structured reasoning protocols"

    def run(self, query: str) -> str:
        # Use CLI for reliability
        result = subprocess.run(
            ["rk-core", "think", "--format", "json", "--profile", "balanced", query],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        data = json.loads(result.stdout)
        return f"Confidence: {data['confidence']:.1%}\n{json.dumps(data['steps'], indent=2)}"

# Create CrewAI agents
analyst = Agent(
    role="Strategic Analyst",
    goal="Provide rigorous analysis of business decisions",
    backstory="Expert in structured decision-making frameworks",
    tools=[ReasonKitTool()],
    verbose=True
)

# Create task
analysis_task = Task(
    description="Analyze whether we should expand to European markets",
    agent=analyst
)

# Run crew
crew = Crew(agents=[analyst], tasks=[analysis_task])
result = crew.kickoff()
```

### AutoGen Integration

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import subprocess
import json

def reasonkit_analyze(query: str, profile: str = "balanced") -> dict:
    """Execute ReasonKit analysis and return structured results."""
    result = subprocess.run(
        ["rk-core", "think", "--format", "json", "--profile", profile, query],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {"error": result.stderr, "success": False}

    return {"result": json.loads(result.stdout), "success": True}

# Register function for AutoGen
llm_config = {
    "functions": [
        {
            "name": "reasonkit_analyze",
            "description": "Analyze a decision or query using ReasonKit's structured reasoning protocols",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The decision or question to analyze"
                    },
                    "profile": {
                        "type": "string",
                        "enum": ["quick", "balanced", "deep", "paranoid"],
                        "description": "Analysis depth profile"
                    }
                },
                "required": ["query"]
            }
        }
    ],
    "config_list": config_list_from_json("OAI_CONFIG_LIST")
}

# Create agents
assistant = AssistantAgent(
    name="analyst",
    llm_config=llm_config,
    system_message="You are a strategic analyst. Use reasonkit_analyze for rigorous decision analysis."
)

user_proxy = UserProxyAgent(
    name="user",
    function_map={"reasonkit_analyze": reasonkit_analyze}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Should we acquire our competitor?"
)
```

---

## Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM rust:1.74-slim as builder

WORKDIR /app
COPY . .
RUN cargo build --release --features memory

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/rk-core /usr/local/bin/

# Create data directory
RUN mkdir -p /data
VOLUME /data

ENV REASONKIT_DATA_DIR=/data
ENV RUST_LOG=info

EXPOSE 8080

CMD ["rk-core", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  reasonkit:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - reasonkit-data:/data
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - RUST_LOG=info
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant-data:/qdrant/storage

volumes:
  reasonkit-data:
  qdrant-data:
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reasonkit
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reasonkit
  template:
    metadata:
      labels:
        app: reasonkit
    spec:
      containers:
      - name: reasonkit
        image: reasonkit/reasonkit-core:latest
        ports:
        - containerPort: 8080
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: reasonkit-secrets
              key: anthropic-api-key
        - name: RUST_LOG
          value: "info"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: reasonkit
spec:
  selector:
    app: reasonkit
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `REASONKIT_DATA_DIR` | Data directory path | `./data` |
| `REASONKIT_CONFIG` | Config file path | None |
| `RUST_LOG` | Log level | `warn` |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `TELEMETRY_ENABLED` | Enable telemetry | `true` |
| `TELEMETRY_DB_PATH` | Telemetry DB path | `.rk_telemetry.db` |

---

## Troubleshooting

### Common Issues

#### 1. "No API key found"

```bash
# Check if key is set
echo $ANTHROPIC_API_KEY

# Set the key
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use mock mode for testing
rk-core think --mock "Test query"
```

#### 2. "Connection refused" (MCP Server)

```bash
# Check if server is running
ps aux | grep rk-core

# Start server manually
rk-core serve-mcp &

# Check logs
rk-core serve-mcp 2>&1 | tee server.log
```

#### 3. "Timeout during LLM call"

```bash
# Increase timeout
rk-core think --timeout 300 "Complex query"

# Or use faster provider
rk-core think --provider groq "Query"
```

#### 4. "Memory feature not enabled"

```bash
# Rebuild with memory feature
cargo build --release --features memory

# Or install with feature
cargo install reasonkit-core --features memory
```

#### 5. Build Errors

```bash
# Update Rust toolchain
rustup update

# Clear cache and rebuild
cargo clean
cargo build --release

# Check for missing dependencies
cargo check 2>&1 | head -50
```

### Debug Mode

```bash
# Enable verbose logging
RUST_LOG=debug rk-core think "Query"

# Maximum verbosity
RUST_LOG=trace rk-core think -vvv "Query"
```

### Getting Help

1. **Documentation**: https://docs.rs/reasonkit-core
2. **GitHub Issues**: https://github.com/reasonkit/reasonkit-core/issues
3. **Discussions**: https://github.com/reasonkit/reasonkit-core/discussions
4. **Website**: https://reasonkit.sh

---

## Version Compatibility

| ReasonKit | Rust | Python | MCP Protocol |
|-----------|------|--------|--------------|
| 0.1.x | 1.74+ | 3.10+ | 1.0 |

---

## Next Steps

- [ThinkTools Guide](./THINKTOOLS_GUIDE.md) - Deep dive into reasoning protocols
- [Custom ThinkTools](./CUSTOM_THINKTOOLS.md) - Create your own protocols
- [CLI Cookbook](./CLI_COOKBOOK.md) - Advanced CLI usage patterns
- [API Reference](./API_REFERENCE.md) - Complete API documentation
- [Use Cases](./USE_CASES.md) - Real-world examples

---

_"Turn Prompts into Protocols"_

**ReasonKit** | https://reasonkit.sh
