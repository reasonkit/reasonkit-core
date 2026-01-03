# ReasonKit Core API Reference

**Version:** 1.0.0
**License:** Apache 2.0
**Website:** https://reasonkit.sh

ReasonKit Core is a Rust-based structured reasoning infrastructure with CLI, library API, and MCP server interfaces.

---

## Table of Contents

1. [CLI Reference](#cli-reference)
2. [Library API (Rust)](#library-api-rust)
3. [Python Bindings (Beta)](#python-bindings)
4. [MCP Server API](#mcp-server-api)
5. [SQLite Schema](#sqlite-schema)
6. [Configuration](#configuration)
7. [Error Types](#error-types)

---

## CLI Reference

The `rk` binary provides the command-line interface for ReasonKit.

### Global Options

| Flag         | Short | Environment Variable | Description                               |
| ------------ | ----- | -------------------- | ----------------------------------------- |
| `--verbose`  | `-v`  | -                    | Increase verbosity (use -v, -vv, or -vvv) |
| `--config`   | `-c`  | `REASONKIT_CONFIG`   | Path to configuration file                |
| `--data-dir` | `-d`  | `REASONKIT_DATA_DIR` | Data directory (default: `./data`)        |

### Commands

#### `rk think` - Execute ThinkTools

Execute structured reasoning protocols (ThinkTools).

```bash
rk think [OPTIONS] <QUERY>
```

**Arguments:**

| Argument  | Required              | Description                   |
| --------- | --------------------- | ----------------------------- |
| `<QUERY>` | Yes (unless `--list`) | The query or input to process |

**Options:**

| Option          | Short | Default          | Description                                                                          |
| --------------- | ----- | ---------------- | ------------------------------------------------------------------------------------ |
| `--protocol`    | `-p`  | -                | Protocol to execute (gigathink, laserlogic, bedrock, proofguard, brutalhonesty)      |
| `--profile`     | -     | -                | Profile to execute (quick, balanced, deep, paranoid, decide, scientific, powercombo) |
| `--provider`    | -     | `anthropic`      | LLM provider (see [Providers](#llm-providers))                                       |
| `--model`       | `-m`  | Provider default | LLM model to use                                                                     |
| `--temperature` | `-t`  | `0.7`            | Temperature for generation (0.0-2.0)                                                 |
| `--max-tokens`  | -     | `2000`           | Maximum tokens to generate                                                           |
| `--budget`      | `-b`  | -                | Budget constraint (e.g., "30s", "1000t", "$0.50")                                    |
| `--mock`        | -     | `false`          | Use mock LLM for testing                                                             |
| `--save-trace`  | -     | `false`          | Save execution trace                                                                 |
| `--trace-dir`   | -     | -                | Directory for trace output                                                           |
| `--format`      | `-f`  | `text`           | Output format (text, json)                                                           |
| `--list`        | -     | -                | List available protocols and profiles                                                |

**Examples:**

```bash
# Quick analysis with default provider
rk think "What are the pros and cons of microservices?"

# Use a specific protocol
rk think -p gigathink "Explore startup success factors"

# Use a profile chain
rk think --profile paranoid "Is this investment safe?"

# Use OpenRouter with specific model
rk think --provider openrouter --model anthropic/claude-sonnet-4 "Analyze this code"

# Budget-constrained execution
rk think --budget "30s" "Quick analysis needed"

# JSON output with trace
rk think -f json --save-trace "Complex question" > result.json

# List available protocols
rk think --list
```

#### `rk web` - Deep Research

Combines web search, knowledge base retrieval, and ThinkTool protocols.

```bash
rk web [OPTIONS] <QUERY>
```

**Aliases:** `dive`, `research`, `deep`, `d`

**Options:**

| Option       | Short | Default     | Description                                        |
| ------------ | ----- | ----------- | -------------------------------------------------- |
| `--depth`    | `-d`  | `standard`  | Research depth (quick, standard, deep, exhaustive) |
| `--web`      | -     | `true`      | Include web search results                         |
| `--kb`       | -     | `true`      | Include knowledge base results                     |
| `--provider` | -     | `anthropic` | LLM provider                                       |
| `--format`   | `-f`  | `text`      | Output format (text, json, markdown)               |
| `--output`   | `-o`  | -           | Save report to file                                |

**Depth Levels:**

| Depth        | Profile                | Duration | Description          |
| ------------ | ---------------------- | -------- | -------------------- |
| `quick`      | GigaThink only         | ~30s     | Fast exploration     |
| `standard`   | GigaThink + LaserLogic | ~2min    | Standard research    |
| `deep`       | Full balanced          | ~5min    | Thorough analysis    |
| `exhaustive` | Paranoid profile       | ~10min   | Maximum verification |

**Examples:**

```bash
# Standard research
rk web "Latest developments in quantum computing"

# Deep research with markdown output
rk web --depth deep -f markdown -o report.md "AI safety research"

# Quick exploration without web search
rk web --depth quick --no-web "Explain transformer architecture"
```

#### `rk verify` - Triangulation

Verify claims using the three-source rule.

```bash
rk verify [OPTIONS] <CLAIM>
```

**Aliases:** `v`, `triangulate`

**Options:**

| Option      | Short | Default | Description                            |
| ----------- | ----- | ------- | -------------------------------------- |
| `--sources` | `-s`  | `3`     | Minimum number of sources required     |
| `--web`     | -     | `true`  | Include web search for verification    |
| `--kb`      | -     | `true`  | Include knowledge base sources         |
| `--anchor`  | -     | `false` | Anchor verified content to ProofLedger |
| `--format`  | `-f`  | `text`  | Output format (text, json, markdown)   |
| `--output`  | `-o`  | -       | Save verification report to file       |

**Examples:**

```bash
# Verify a claim
rk verify "GPT-4 has 1.8 trillion parameters"

# Anchor verified claims
rk verify --anchor "Einstein published relativity in 1905"

# Require more sources
rk verify -s 5 "This medication is safe for long-term use"
```

#### `rk trace` - Audit Trail Management

View and manage execution traces.

```bash
rk trace <ACTION>
```

**Subcommands:**

| Action      | Description           |
| ----------- | --------------------- |
| `list`      | List saved traces     |
| `view <ID>` | View a specific trace |
| `clean`     | Delete traces         |

**List Options:**

```bash
rk trace list [OPTIONS]
  --dir, -d <PATH>       Trace directory
  --protocol, -p <NAME>  Filter by protocol
  --limit, -l <N>        Limit results (default: 20)
```

**View Options:**

```bash
rk trace view <ID> [OPTIONS]
  --dir, -d <PATH>       Trace directory
  --format, -f <FMT>     Output format (text, json)
```

**Clean Options:**

```bash
rk trace clean [OPTIONS]
  --dir, -d <PATH>       Trace directory
  --all                  Delete all traces
  --keep-days <N>        Keep traces from last N days
```

#### `rk metrics` - Quality Metrics

View ThinkTools execution metrics and quality reports.

```bash
rk metrics <ACTION>
```

**Subcommands:**

| Action         | Description                              |
| -------------- | ---------------------------------------- |
| `report`       | Show metrics report with grades          |
| `stats <NAME>` | Statistics for specific protocol/profile |
| `path`         | Show metrics storage location            |
| `clear`        | Clear all metrics data                   |

**Report Options:**

```bash
rk metrics report [OPTIONS]
  --format, -f <FMT>     Output format (text, json)
  --filter, -F <NAME>    Filter by protocol or profile
  --output, -o <PATH>    Save report to file
```

#### `rk serve` - API Server

Start the API server.

```bash
rk serve [OPTIONS]
```

**Options:**

| Option   | Default     | Description     |
| -------- | ----------- | --------------- |
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8080`      | Port to bind to |

#### `rk completions` - Shell Completions

Generate shell completions.

```bash
rk completions <SHELL>
```

**Supported Shells:** `zsh`, `bash`, `fish`, `powershell`, `elvish`

**Installation Examples:**

```bash
# Zsh (Oh-My-Zsh)
rk completions zsh > $ZSH_CUSTOM/plugins/reasonkit/_rk

# Bash
rk completions bash > ~/.bash_completion.d/rk

# Fish
rk completions fish > ~/.config/fish/completions/rk.fish
```

### Memory Feature Commands

The following commands require the `memory` feature:

```bash
cargo build --release --features memory
```

#### `rk ingest` - Document Ingestion

```bash
rk ingest [OPTIONS] <PATH>
  --doc-type, -t <TYPE>  Document type (paper, documentation, code, note)
  --recursive, -r        Process directories recursively
```

#### `rk query` - Knowledge Base Query

```bash
rk query [OPTIONS] <QUERY>
  --top-k, -k <N>        Number of results (default: 5)
  --hybrid               Use hybrid search (BM25 + vector)
  --raptor               Use RAPTOR tree retrieval
  --format, -f <FMT>     Output format (text, json, markdown)
```

#### `rk rag` - RAG Operations

```bash
rk rag query [OPTIONS] <QUERY>
  --top-k, -k <N>        Chunks to retrieve (default: 5)
  --min-score <SCORE>    Minimum relevance (0.0-1.0)
  --mode <MODE>          RAG mode (quick, balanced, thorough)
  --no-llm               Retrieval only, no generation

rk rag retrieve <QUERY>
  --top-k, -k <N>        Number of results

rk rag stats        Show knowledge base statistics
```

### LLM Providers

ReasonKit supports 18+ LLM providers:

| Tier            | Provider       | Flag Value     | Env Variable                     | Default Model           |
| --------------- | -------------- | -------------- | -------------------------------- | ----------------------- |
| **Major Cloud** | Anthropic      | `anthropic`    | `ANTHROPIC_API_KEY`              | claude-sonnet-4         |
|                 | OpenAI         | `openai`       | `OPENAI_API_KEY`                 | gpt-4o                  |
|                 | Google Gemini  | `gemini`       | `GEMINI_API_KEY`                 | gemini-2.0-flash        |
|                 | Google Vertex  | `vertex`       | `GOOGLE_APPLICATION_CREDENTIALS` | gemini-2.0-flash        |
|                 | Azure OpenAI   | `azure`        | `AZURE_OPENAI_API_KEY`           | gpt-4o                  |
|                 | AWS Bedrock    | `bedrock`      | `AWS_ACCESS_KEY_ID`              | claude-sonnet-4         |
| **Specialized** | xAI (Grok)     | `xai`          | `XAI_API_KEY`                    | grok-2                  |
|                 | Groq           | `groq`         | `GROQ_API_KEY`                   | llama-3.3-70b-versatile |
|                 | Mistral        | `mistral`      | `MISTRAL_API_KEY`                | mistral-large-latest    |
|                 | DeepSeek       | `deepseek`     | `DEEPSEEK_API_KEY`               | deepseek-chat           |
|                 | Cohere         | `cohere`       | `COHERE_API_KEY`                 | command-r-plus          |
|                 | Perplexity     | `perplexity`   | `PERPLEXITY_API_KEY`             | sonar-pro               |
|                 | Cerebras       | `cerebras`     | `CEREBRAS_API_KEY`               | llama-3.3-70b           |
| **Inference**   | Together AI    | `together`     | `TOGETHER_API_KEY`               | Llama-3.3-70B-Instruct  |
|                 | Fireworks AI   | `fireworks`    | `FIREWORKS_API_KEY`              | llama-v3p3-70b-instruct |
|                 | Alibaba Qwen   | `qwen`         | `DASHSCOPE_API_KEY`              | qwen-max                |
|                 | Cloudflare AI  | `cloudflare`   | `CLOUDFLARE_API_KEY`             | llama-3.3-70b-instruct  |
| **Aggregation** | OpenRouter     | `openrouter`   | `OPENROUTER_API_KEY`             | claude-3.5-sonnet       |
| **CLI Tools**   | Claude CLI     | `claude-cli`   | (browser auth)                   | -                       |
|                 | Codex CLI      | `codex-cli`    | (browser auth)                   | -                       |
|                 | Gemini CLI     | `gemini-cli`   | (browser auth)                   | -                       |
|                 | OpenCode CLI   | `opencode-cli` | (browser auth)                   | -                       |
|                 | GitHub Copilot | `copilot-cli`  | (browser auth)                   | -                       |

---

## Library API (Rust)

### Core Structs

#### `ProtocolExecutor`

The main entry point for executing protocols and profiles.

```rust
use reasonkit::thinktool::{ProtocolExecutor, ProtocolInput, ExecutorConfig};

// Create with default configuration
let executor = ProtocolExecutor::new()?;

// Create with custom configuration
let config = ExecutorConfig {
    llm: LlmConfig::default(),
    timeout_secs: 120,
    save_traces: true,
    trace_dir: Some(PathBuf::from("./traces")),
    verbose: false,
    use_mock: false,
    budget: BudgetConfig::default(),
    cli_tool: None,
    self_consistency: None,
    show_progress: true,
};
let executor = ProtocolExecutor::with_config(config)?;

// Create for testing (mock LLM)
let mock_executor = ProtocolExecutor::mock()?;
```

**Methods:**

```rust
impl ProtocolExecutor {
    /// Execute a single protocol
    pub async fn execute(
        &self,
        protocol_id: &str,
        input: ProtocolInput
    ) -> Result<ProtocolOutput>;

    /// Execute a reasoning profile (chain of protocols)
    pub async fn execute_profile(
        &self,
        profile_id: &str,
        input: ProtocolInput
    ) -> Result<ProtocolOutput>;

    /// Execute with Self-Consistency voting
    pub async fn execute_with_self_consistency(
        &self,
        profile_id: &str,
        input: ProtocolInput,
        sc_config: &SelfConsistencyConfig,
    ) -> Result<(ProtocolOutput, ConsistencyResult)>;

    /// List available protocols
    pub fn list_protocols(&self) -> Vec<&str>;

    /// List available profiles
    pub fn list_profiles(&self) -> Vec<&str>;

    /// Get protocol by ID
    pub fn get_protocol(&self, id: &str) -> Option<&Protocol>;

    /// Get profile by ID
    pub fn get_profile(&self, id: &str) -> Option<&ReasoningProfile>;
}
```

#### `ProtocolInput`

Input data for protocol execution.

```rust
use reasonkit::thinktool::ProtocolInput;

// Create with query (for GigaThink)
let input = ProtocolInput::query("What are the key factors for startup success?");

// Create with argument (for LaserLogic)
let input = ProtocolInput::argument("All swans are white");

// Create with statement (for BedRock)
let input = ProtocolInput::statement("Microservices are better than monoliths");

// Create with claim (for ProofGuard)
let input = ProtocolInput::claim("GPT-4 has 1.8 trillion parameters");

// Create with work to critique (for BrutalHonesty)
let input = ProtocolInput::work("My analysis of the market...");

// Add additional fields
let input = ProtocolInput::query("Main question")
    .with_field("context", "Additional background")
    .with_field("constraints", "Must consider cost");
```

#### `ProtocolOutput`

Output from protocol execution.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolOutput {
    /// Protocol that was executed
    pub protocol_id: String,

    /// Whether execution succeeded
    pub success: bool,

    /// Output data (step results)
    pub data: HashMap<String, serde_json::Value>,

    /// Overall confidence score (0.0-1.0)
    pub confidence: f64,

    /// Step results
    pub steps: Vec<StepResult>,

    /// Total token usage
    pub tokens: TokenUsage,

    /// Execution time in milliseconds
    pub duration_ms: u64,

    /// Error message if failed
    pub error: Option<String>,

    /// Trace ID (if saved)
    pub trace_id: Option<String>,

    /// Budget usage summary
    pub budget_summary: Option<BudgetSummary>,
}

impl ProtocolOutput {
    /// Get a field from output data
    pub fn get(&self, key: &str) -> Option<&serde_json::Value>;

    /// Get perspectives (for GigaThink)
    pub fn perspectives(&self) -> Vec<&str>;

    /// Get verdict
    pub fn verdict(&self) -> Option<&str>;
}
```

#### `ExecutionTrace`

Complete audit trail of execution.

```rust
use reasonkit::thinktool::{ExecutionTrace, ExecutionStatus, StepTrace};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Unique trace identifier
    pub id: Uuid,

    /// Protocol that was executed
    pub protocol_id: String,

    /// Protocol version
    pub protocol_version: String,

    /// Input provided
    pub input: serde_json::Value,

    /// Step-by-step execution record
    pub steps: Vec<StepTrace>,

    /// Final output
    pub output: Option<serde_json::Value>,

    /// Overall status
    pub status: ExecutionStatus,

    /// Timing information
    pub timing: TimingInfo,

    /// Total token usage
    pub tokens: TokenUsage,

    /// Overall confidence
    pub confidence: f64,

    /// Execution metadata
    pub metadata: TraceMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    Paused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}
```

#### `LlmConfig` and `LlmProvider`

LLM provider configuration.

```rust
use reasonkit::thinktool::{LlmConfig, LlmProvider};

// Default configuration (Anthropic)
let config = LlmConfig::default();

// Specific provider with model
let config = LlmConfig::for_provider(LlmProvider::OpenRouter, "anthropic/claude-sonnet-4")
    .with_temperature(0.7)
    .with_max_tokens(2000);

// OpenAI with API key
let config = LlmConfig::for_provider(LlmProvider::OpenAI, "gpt-4o")
    .with_api_key("sk-...");

// Azure OpenAI
let config = LlmConfig::for_provider(LlmProvider::AzureOpenAI, "gpt-4")
    .with_azure("my-resource", "my-deployment");

// AWS Bedrock
let config = LlmConfig::for_provider(LlmProvider::AWSBedrock, "anthropic.claude-v2")
    .with_aws_region("us-east-1");
```

### Profiles

Available reasoning profiles:

| Profile      | Chain                    | Min Confidence | Use Case          |
| ------------ | ------------------------ | -------------- | ----------------- |
| `quick`      | GigaThink -> LaserLogic  | 70%            | Fast analysis     |
| `balanced`   | gt -> ll -> br -> pg     | 80%            | Standard analysis |
| `deep`       | All 5 + conditional BH   | 85%            | Thorough analysis |
| `paranoid`   | All 5 + 2nd verification | 95%            | Maximum rigor     |
| `decide`     | ll -> br -> bh           | 85%            | Decision support  |
| `scientific` | gt -> br -> pg           | 85%            | Research          |
| `powercombo` | All 5 + cross-validation | 95%            | Ultimate mode     |

### Custom Protocol Definition

Define custom protocols using YAML or TOML:

```yaml
# protocols/my_protocol.yaml
id: my_custom_protocol
name: "My Custom Protocol"
version: "1.0.0"
description: "Custom reasoning protocol"

input:
  required:
    - query
  optional:
    - context

steps:
  - id: analyze
    name: "Initial Analysis"
    action:
      type: generate
      min_count: 5
      max_count: 10
    prompt_template: |
      Analyze the following query from multiple perspectives:

      Query: {{query}}
      {{#if context}}Context: {{context}}{{/if}}

      Generate 5-10 distinct perspectives.

  - id: synthesize
    name: "Synthesis"
    depends_on:
      - analyze
    action:
      type: synthesize
      sources:
        - analyze
    prompt_template: |
      Synthesize the following perspectives into key insights:

      {{analyze}}

      Provide a coherent summary.

output:
  - analyze
  - synthesize
```

Load custom protocols:

```rust
use reasonkit::thinktool::{ProtocolExecutor, yaml_loader};

let mut executor = ProtocolExecutor::new()?;

// Load from YAML file
let protocol = yaml_loader::load_protocol_from_yaml_file("my_protocol.yaml")?;
executor.registry_mut().register(protocol);
```

### Complete Example

```rust
use reasonkit::thinktool::{
    ProtocolExecutor, ProtocolInput, ExecutorConfig,
    LlmConfig, LlmProvider
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure with OpenRouter
    let config = ExecutorConfig {
        llm: LlmConfig::for_provider(
            LlmProvider::OpenRouter,
            "anthropic/claude-sonnet-4"
        ),
        save_traces: true,
        trace_dir: Some("./traces".into()),
        ..Default::default()
    };

    let executor = ProtocolExecutor::with_config(config)?;

    // Execute the balanced profile
    let input = ProtocolInput::query("Should we adopt Kubernetes?")
        .with_field("context", "Small startup with 5 engineers");

    let result = executor.execute_profile("balanced", input).await?;

    println!("Success: {}", result.success);
    println!("Confidence: {:.2}", result.confidence);
    println!("Duration: {}ms", result.duration_ms);
    println!("Tokens: {}", result.tokens.total_tokens);

    // Access step results
    for step in &result.steps {
        println!("\nStep: {}", step.step_id);
        println!("  Confidence: {:.2}", step.confidence);
        if let Some(text) = step.as_text() {
            println!("  Output: {}", &text[..200.min(text.len())]);
        }
    }

    Ok(())
}
```

---

## Python Bindings

ReasonKit provides Python bindings via PyO3 (beta - build from source only).

> **Note:** Python bindings are in beta and require building from source. No PyPI package is available yet.

### Installation

```bash
# Install maturin (build tool for Python bindings)
uv pip install maturin

# Build Python bindings from source
cd reasonkit-core
maturin develop --release --features python
```

### Basic Usage

```python
from reasonkit import Reasoner

# Create a reasoner (uses real LLM by default)
reasoner = Reasoner()

# Or create with mock LLM for testing
reasoner = Reasoner(use_mock=True)

# Execute a protocol
result_json = reasoner.think("gigathink", "What factors drive startup success?")

# Parse the result
import json
result = json.loads(result_json)
print(f"Success: {result['success']}")
print(f"Confidence: {result['confidence']}")

# List available protocols
protocols = reasoner.list_protocols()
print(f"Available protocols: {protocols}")
```

### Return Type

The `think()` method returns a JSON string representing `ProtocolOutput`:

```python
{
    "protocol_id": "gigathink",
    "success": true,
    "confidence": 0.85,
    "data": {
        "expand_perspectives": {...},
        "prioritize": {...},
        "synthesize": {...}
    },
    "steps": [...],
    "tokens": {
        "input_tokens": 500,
        "output_tokens": 1200,
        "total_tokens": 1700,
        "cost_usd": 0.0025
    },
    "duration_ms": 3450,
    "error": null,
    "trace_id": null
}
```

### Async Support

For async Python applications, wrap in an executor:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from reasonkit import Reasoner

executor = ThreadPoolExecutor(max_workers=4)
reasoner = Reasoner()

async def think_async(protocol: str, query: str) -> dict:
    loop = asyncio.get_event_loop()
    result_json = await loop.run_in_executor(
        executor,
        lambda: reasoner.think(protocol, query)
    )
    return json.loads(result_json)

# Usage
async def main():
    result = await think_async("gigathink", "My question")
    print(result)

asyncio.run(main())
```

### Type Hints

```python
from typing import TypedDict, List, Optional

class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float

class StepResult(TypedDict):
    step_id: str
    success: bool
    confidence: float
    duration_ms: int
    tokens: TokenUsage

class ProtocolOutput(TypedDict):
    protocol_id: str
    success: bool
    confidence: float
    data: dict
    steps: List[StepResult]
    tokens: TokenUsage
    duration_ms: int
    error: Optional[str]
    trace_id: Optional[str]
```

---

## MCP Server API

ReasonKit implements the Model Context Protocol (MCP) specification (2025-11-25).

### Architecture

```
+----------------------------------------------+
| MCP Registry (Coordinator)                   |
|   - Server discovery                         |
|   - Health monitoring                        |
|   - Capability aggregation                   |
+----------------------------------------------+
| MCP Client (Consumer)                        |
|   - Connect to external MCP servers          |
|   - Execute tools via RPC                    |
|   - Access resources                         |
+----------------------------------------------+
| MCP Servers (ThinkTools)                     |
|   - GigaThink, LaserLogic, etc.              |
|   - Custom tool servers                      |
+----------------------------------------------+
| Transport Layer                              |
|   - JSON-RPC 2.0 over stdio (primary)        |
|   - HTTP/SSE (optional)                      |
+----------------------------------------------+
```

### Available Tools

| Tool Name         | Description                    | Input Schema                               |
| ----------------- | ------------------------------ | ------------------------------------------ |
| `gigathink`       | Multi-perspective expansion    | `{ query: string, perspectives?: number }` |
| `laserlogic`      | Logical validation             | `{ argument: string }`                     |
| `bedrock`         | First principles decomposition | `{ statement: string }`                    |
| `proofguard`      | Source triangulation           | `{ claim: string, sources?: number }`      |
| `brutalhonesty`   | Adversarial critique           | `{ work: string }`                         |
| `profile_execute` | Execute a reasoning profile    | `{ profile: string, query: string }`       |

### Tool Definition Schema

```json
{
  "name": "gigathink",
  "description": "Multi-perspective expansion - generates 10+ viewpoints",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The question or topic to analyze"
      },
      "perspectives": {
        "type": "integer",
        "minimum": 5,
        "maximum": 20,
        "default": 10,
        "description": "Number of perspectives to generate"
      }
    },
    "required": ["query"]
  }
}
```

### Tool Result Schema

```json
{
  "content": [
    {
      "type": "text",
      "text": "Analysis results..."
    }
  ],
  "isError": false
}
```

### MCP Client Usage (Rust)

```rust
use reasonkit::mcp::{McpClient, McpClientConfig, McpClientTrait};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = McpClientConfig {
        name: "sequential-thinking".to_string(),
        command: "npx".to_string(),
        args: vec![
            "-y".to_string(),
            "@modelcontextprotocol/server-sequential-thinking".to_string()
        ],
        env: HashMap::new(),
        timeout_secs: 30,
        auto_reconnect: true,
        max_retries: 3,
    };

    let mut client = McpClient::new(config);
    client.connect().await?;

    // List tools
    let tools = client.list_tools().await?;
    for tool in tools {
        println!("Tool: {} - {}", tool.name, tool.description.unwrap_or_default());
    }

    // Call a tool
    let result = client.call_tool(
        "think",
        serde_json::json!({
            "query": "What is chain-of-thought reasoning?"
        })
    ).await?;

    println!("Result: {:?}", result);

    client.disconnect().await?;
    Ok(())
}
```

### Claude Desktop Integration

Add to Claude Desktop configuration (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "reasonkit": {
      "command": "rk",
      "args": ["serve", "--mcp"]
    }
  }
}
```

### JSON-RPC 2.0 Messages

**Initialize Request:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": {},
    "clientInfo": {
      "name": "my-client",
      "version": "1.0.0"
    }
  }
}
```

**Initialize Response:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-11-25",
    "capabilities": {
      "tools": { "listChanged": true },
      "resources": { "subscribe": false }
    },
    "serverInfo": {
      "name": "reasonkit-core",
      "version": "1.0.0"
    }
  }
}
```

**Tool Call Request:**

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "gigathink",
    "arguments": {
      "query": "Analyze the impact of AI on employment"
    }
  }
}
```

**Tool Call Response:**

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "## Perspectives\n\n1. **Economic View**...\n2. **Labor View**..."
      }
    ]
  }
}
```

---

## SQLite Schema

ReasonKit uses SQLite for local telemetry and trace storage.

### Database Location

Default path: `~/.local/share/reasonkit/.rk_telemetry.db`

### Core Tables

#### `sessions`

Tracks CLI sessions.

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,                    -- UUID
    started_at TEXT NOT NULL,               -- ISO 8601 timestamp
    ended_at TEXT,                          -- ISO 8601 timestamp
    duration_ms INTEGER,
    tool_count INTEGER DEFAULT 0,
    query_count INTEGER DEFAULT 0,
    feedback_count INTEGER DEFAULT 0,
    profile TEXT,                           -- Reasoning profile used
    success_rate REAL,
    client_version TEXT,
    os_family TEXT
);
```

#### `queries`

Individual query events (privacy-preserving).

```sql
CREATE TABLE queries (
    id TEXT PRIMARY KEY,                    -- UUID
    session_id TEXT NOT NULL,               -- FK to sessions
    timestamp TEXT NOT NULL,                -- ISO 8601

    -- Query metadata (no raw text - privacy)
    query_hash TEXT NOT NULL,               -- SHA-256 of normalized query
    query_length INTEGER NOT NULL,
    query_token_count INTEGER,
    query_type TEXT,                        -- search/reason/code/general

    -- Execution metrics
    latency_ms INTEGER NOT NULL,
    tool_calls INTEGER DEFAULT 0,
    retrieval_count INTEGER DEFAULT 0,

    -- Results
    result_count INTEGER,
    result_quality_score REAL,
    error_occurred INTEGER DEFAULT 0,
    error_category TEXT,
    profile_used TEXT,
    tools_used TEXT,                        -- JSON array

    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX idx_queries_session ON queries(session_id);
CREATE INDEX idx_queries_timestamp ON queries(timestamp);
CREATE INDEX idx_queries_type ON queries(query_type);
```

#### `reasoning_traces`

ThinkTool execution traces.

```sql
CREATE TABLE reasoning_traces (
    id TEXT PRIMARY KEY,                    -- UUID
    session_id TEXT NOT NULL,
    query_id TEXT,
    timestamp TEXT NOT NULL,

    thinktool_name TEXT NOT NULL,           -- GigaThink, LaserLogic, etc.
    step_count INTEGER NOT NULL,
    total_ms INTEGER NOT NULL,
    avg_step_ms REAL,
    coherence_score REAL,
    depth_score REAL,
    step_types TEXT,                        -- JSON array

    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (query_id) REFERENCES queries(id)
);

CREATE INDEX idx_traces_thinktool ON reasoning_traces(thinktool_name);
```

#### `tool_usage`

Tool invocation tracking.

```sql
CREATE TABLE tool_usage (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query_id TEXT,
    timestamp TEXT NOT NULL,

    tool_name TEXT NOT NULL,
    tool_category TEXT,                     -- search/file/shell/mcp/reasoning
    execution_ms INTEGER NOT NULL,
    success INTEGER NOT NULL,
    error_type TEXT,
    input_size_bytes INTEGER,
    output_size_bytes INTEGER,

    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX idx_tool_usage_tool ON tool_usage(tool_name);
```

#### `feedback`

User feedback events.

```sql
CREATE TABLE feedback (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query_id TEXT,
    timestamp TEXT NOT NULL,

    feedback_type TEXT NOT NULL,            -- thumbs_up/thumbs_down/explicit/implicit
    rating INTEGER,                         -- 1-5 stars
    category TEXT,                          -- accuracy/relevance/speed/format
    context_hash TEXT,

    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (query_id) REFERENCES queries(id)
);
```

### Query Examples

**Recent Session Summary:**

```sql
SELECT
    s.id,
    s.started_at,
    s.duration_ms,
    s.query_count,
    s.success_rate,
    COUNT(DISTINCT f.id) as feedback_items
FROM sessions s
LEFT JOIN feedback f ON s.id = f.session_id
WHERE s.started_at > datetime('now', '-7 days')
GROUP BY s.id
ORDER BY s.started_at DESC
LIMIT 20;
```

**Tool Performance Stats:**

```sql
SELECT
    tool_name,
    COUNT(*) as invocations,
    AVG(execution_ms) as avg_ms,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
FROM tool_usage
WHERE timestamp > datetime('now', '-30 days')
GROUP BY tool_name
ORDER BY invocations DESC;
```

**ThinkTool Effectiveness:**

```sql
SELECT
    thinktool_name,
    COUNT(*) as usage_count,
    AVG(step_count) as avg_steps,
    AVG(total_ms) as avg_execution_ms,
    AVG(coherence_score) as avg_coherence
FROM reasoning_traces
WHERE timestamp > datetime('now', '-30 days')
GROUP BY thinktool_name
ORDER BY usage_count DESC;
```

**Query Latency Percentiles:**

```sql
SELECT
    query_type,
    COUNT(*) as count,
    AVG(latency_ms) as avg_ms,
    MIN(latency_ms) as min_ms,
    MAX(latency_ms) as max_ms
FROM queries
WHERE timestamp > datetime('now', '-7 days')
GROUP BY query_type;
```

---

## Configuration

### Configuration File

ReasonKit supports TOML configuration files.

**Default locations:**

- `./reasonkit.toml` (project-local)
- `~/.config/reasonkit/config.toml` (user)
- `/etc/reasonkit/config.toml` (system)

**Example `reasonkit.toml`:**

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
temperature = 0.7
max_tokens = 2000
timeout_secs = 60

[llm.anthropic]
api_key = "${ANTHROPIC_API_KEY}"  # Environment variable substitution

[llm.openrouter]
api_key = "${OPENROUTER_API_KEY}"
default_model = "anthropic/claude-sonnet-4"

[executor]
timeout_secs = 120
save_traces = true
trace_dir = "./traces"
show_progress = true

[budget]
max_time_secs = 300
max_tokens = 50000
max_cost_usd = 1.00
strategy = "adaptive"  # none, fixed, adaptive

[telemetry]
enabled = true
db_path = "~/.local/share/reasonkit/.rk_telemetry.db"
retention_days = 90

[memory]
enabled = false
qdrant_url = "http://localhost:6334"
```

### Environment Variables

| Variable              | Description                                 | Default       |
| --------------------- | ------------------------------------------- | ------------- |
| `REASONKIT_CONFIG`    | Path to configuration file                  | Auto-detected |
| `REASONKIT_DATA_DIR`  | Data directory                              | `./data`      |
| `REASONKIT_LOG_LEVEL` | Log level (trace, debug, info, warn, error) | `warn`        |
| `ANTHROPIC_API_KEY`   | Anthropic API key                           | -             |
| `OPENAI_API_KEY`      | OpenAI API key                              | -             |
| `OPENROUTER_API_KEY`  | OpenRouter API key                          | -             |
| `GEMINI_API_KEY`      | Google Gemini API key                       | -             |
| `GROQ_API_KEY`        | Groq API key                                | -             |
| `XAI_API_KEY`         | xAI (Grok) API key                          | -             |
| `MISTRAL_API_KEY`     | Mistral API key                             | -             |
| `DEEPSEEK_API_KEY`    | DeepSeek API key                            | -             |
| `COHERE_API_KEY`      | Cohere API key                              | -             |
| `PERPLEXITY_API_KEY`  | Perplexity API key                          | -             |
| `CEREBRAS_API_KEY`    | Cerebras API key                            | -             |
| `TOGETHER_API_KEY`    | Together AI API key                         | -             |
| `FIREWORKS_API_KEY`   | Fireworks AI API key                        | -             |
| `DASHSCOPE_API_KEY`   | Alibaba Qwen API key                        | -             |
| `CLOUDFLARE_API_KEY`  | Cloudflare AI API key                       | -             |

### Provider Configuration

**Anthropic:**

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"

[llm.anthropic]
api_key = "${ANTHROPIC_API_KEY}"
```

**OpenRouter (300+ models):**

```toml
[llm]
provider = "openrouter"
model = "anthropic/claude-sonnet-4"

[llm.openrouter]
api_key = "${OPENROUTER_API_KEY}"
```

**Azure OpenAI:**

```toml
[llm]
provider = "azure"
model = "gpt-4"

[llm.azure]
api_key = "${AZURE_OPENAI_API_KEY}"
resource = "my-resource"
deployment = "my-deployment"
```

**AWS Bedrock:**

```toml
[llm]
provider = "bedrock"
model = "anthropic.claude-v2"

[llm.bedrock]
region = "us-east-1"
# Uses AWS credentials from environment or ~/.aws/credentials
```

---

## Error Types

ReasonKit uses the `Error` enum for all error conditions.

```rust
use reasonkit::Error;

pub enum Error {
    /// I/O error
    Io(std::io::Error),

    /// JSON serialization error
    Json(serde_json::Error),

    /// PDF processing error
    Pdf(String),

    /// Document not found
    DocumentNotFound(String),

    /// Chunk not found
    ChunkNotFound(String),

    /// Embedding error
    Embedding(String),

    /// Indexing error
    Indexing(String),

    /// Retrieval error
    Retrieval(String),

    /// Storage error
    Storage(String),

    /// Configuration error
    Config(String),

    /// Network/HTTP error
    Network(String),

    /// Schema validation error
    Validation(String),

    /// Resource not found
    NotFound { resource: String },

    /// Parse error
    Parse { message: String },

    /// Qdrant error
    Qdrant(String),

    /// Tantivy search error
    Tantivy(String),
}
```

### Error Constructors

```rust
// Create specific errors
let err = Error::config("Invalid provider configuration");
let err = Error::network("Connection timeout");
let err = Error::validation("Missing required field: query");

// Wrap errors with context
let result = some_operation()
    .context("Failed to initialize LLM client")?;
```

### HTTP Error Codes (API Server)

| Code | Error Type   | Description                                |
| ---- | ------------ | ------------------------------------------ |
| 400  | `Validation` | Invalid input or schema validation failure |
| 401  | `Config`     | Missing or invalid API key                 |
| 404  | `NotFound`   | Protocol or resource not found             |
| 429  | `Network`    | Rate limit exceeded                        |
| 500  | `*`          | Unexpected internal error                  |

**Error Response Body:**

```json
{
  "error": "validation",
  "message": "Missing required field: query",
  "recoverable": true
}
```

---

## Quick Reference

### ThinkTools

| Tool          | Shortcut | Purpose                               | Input Field |
| ------------- | -------- | ------------------------------------- | ----------- |
| GigaThink     | `gt`     | Multi-perspective expansion           | `query`     |
| LaserLogic    | `ll`     | Logical validation, fallacy detection | `argument`  |
| BedRock       | `br`     | First principles decomposition        | `statement` |
| ProofGuard    | `pg`     | Source triangulation                  | `claim`     |
| BrutalHonesty | `bh`     | Adversarial self-critique             | `work`      |

### Profiles

| Profile      | Modules              | Confidence | Use Case         |
| ------------ | -------------------- | ---------- | ---------------- |
| `quick`      | gt -> ll             | 70%        | Fast analysis    |
| `balanced`   | gt -> ll -> br -> pg | 80%        | Standard         |
| `deep`       | All 5                | 85%        | Thorough         |
| `paranoid`   | All 5 + 2nd pass     | 95%        | Maximum rigor    |
| `decide`     | ll -> br -> bh       | 85%        | Decision support |
| `scientific` | gt -> br -> pg       | 85%        | Research         |
| `powercombo` | All 5 + validation   | 95%        | Ultimate mode    |

### Common Commands

```bash
# Quick analysis
rk think "Your question"

# Use specific profile
rk think --profile balanced "Complex question"

# Deep research
rk web --depth deep "Research topic"

# Verify a claim
rk verify "The claim to verify"

# View metrics
rk metrics report

# List available tools
rk think --list
```

---

**Version:** 1.0.0
**Last Updated:** December 2025
**License:** Apache 2.0
**Website:** https://reasonkit.sh
