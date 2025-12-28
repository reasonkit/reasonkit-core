# CLI ARCHITECTURE DESIGN SPECIFICATION
> ReasonKit Core Command Line Interface
> Version: 1.0.0 | Status: Design Phase
> Author: ReasonKit Team

---

## 1. EXECUTIVE SUMMARY

This document specifies the complete CLI architecture for `rk-core`, the ReasonKit
command-line interface. The design follows Unix philosophy, Rust CLI best practices,
and provides both interactive and scriptable interfaces.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RK-CORE CLI OVERVIEW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  rk-core                                                                    │
│  ├── ingest      # Document ingestion pipeline                              │
│  ├── query       # Knowledge base queries                                   │
│  ├── index       # Index management                                         │
│  ├── search      # Direct search operations                                 │
│  ├── embed       # Embedding operations                                     │
│  ├── export      # Data export                                              │
│  ├── import      # Data import                                              │
│  ├── stats       # Statistics and metrics                                   │
│  ├── serve       # HTTP API server                                          │
│  ├── config      # Configuration management                                 │
│  ├── doctor      # Health checks and diagnostics                            │
│  └── completions # Shell completions                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. DESIGN PRINCIPLES

### 2.1 Core Principles

| Principle | Implementation |
|-----------|----------------|
| **Unix Philosophy** | One thing well, composable, text streams |
| **Progressive Disclosure** | Simple defaults, expert options available |
| **Scriptability** | JSON/JSONL output, exit codes, stdin support |
| **Discoverability** | Rich help, examples, shell completions |
| **Performance** | Async by default, streaming output, progress bars |

### 2.2 Output Philosophy

```
Default: Human-readable with colors
--json: Machine-readable JSON
--jsonl: Streaming JSON Lines (for large outputs)
--quiet: Minimal output (exit codes only)
--verbose/-v: Detailed logging (stackable: -vvv)
```

---

## 3. COMMAND HIERARCHY

### 3.1 Global Options

```
rk-core [GLOBAL_OPTIONS] <COMMAND> [COMMAND_OPTIONS]

GLOBAL_OPTIONS:
  -c, --config <PATH>       Configuration file path
                            [env: REASONKIT_CONFIG]
                            [default: ~/.config/reasonkit/config.toml]

  -d, --data-dir <PATH>     Data directory
                            [env: REASONKIT_DATA_DIR]
                            [default: ~/.local/share/reasonkit]

  -v, --verbose...          Increase verbosity (-v, -vv, -vvv)

  -q, --quiet               Suppress non-essential output

  --color <WHEN>            Color output mode
                            [possible values: auto, always, never]
                            [default: auto]

  --json                    Output in JSON format

  --jsonl                   Output in JSON Lines format (streaming)

  -h, --help                Print help information

  -V, --version             Print version information
```

### 3.2 Command Structure

```
rk-core
│
├── ingest                  # Document ingestion
│   ├── file <PATH>         # Ingest single file
│   ├── dir <PATH>          # Ingest directory
│   ├── url <URL>           # Ingest from URL
│   ├── github <REPO>       # Ingest GitHub repository
│   └── stdin               # Ingest from stdin
│
├── query <QUERY>           # Natural language query
│   ├── --top-k <N>         # Number of results
│   ├── --hybrid            # Use hybrid search
│   ├── --raptor            # Use RAPTOR tree
│   ├── --rerank            # Apply reranking
│   ├── --filter <EXPR>     # Metadata filter
│   └── --explain           # Show retrieval explanation
│
├── search                  # Direct search operations
│   ├── vector <QUERY>      # Dense vector search
│   ├── keyword <QUERY>     # BM25 keyword search
│   ├── hybrid <QUERY>      # Hybrid search
│   └── similar <DOC_ID>    # Find similar documents
│
├── embed                   # Embedding operations
│   ├── text <TEXT>         # Embed text
│   ├── file <PATH>         # Embed file content
│   └── batch <PATH>        # Batch embed from file
│
├── index                   # Index management
│   ├── build               # Build/rebuild index
│   ├── status              # Show index status
│   ├── optimize            # Optimize index
│   ├── repair              # Repair corrupted index
│   └── clear               # Clear all indexes
│
├── export                  # Data export
│   ├── documents           # Export documents
│   ├── embeddings          # Export embeddings
│   ├── index               # Export index
│   └── all                 # Export everything
│
├── import                  # Data import
│   ├── documents <PATH>    # Import documents
│   ├── embeddings <PATH>   # Import embeddings
│   └── index <PATH>        # Import index
│
├── stats                   # Statistics
│   ├── overview            # General overview
│   ├── documents           # Document statistics
│   ├── index               # Index statistics
│   └── performance         # Performance metrics
│
├── serve                   # HTTP API server
│   ├── --host <HOST>       # Bind host
│   ├── --port <PORT>       # Bind port
│   ├── --workers <N>       # Worker threads
│   └── --tls               # Enable TLS
│
├── config                  # Configuration
│   ├── show                # Show current config
│   ├── edit                # Edit config in $EDITOR
│   ├── set <KEY> <VALUE>   # Set config value
│   ├── get <KEY>           # Get config value
│   └── reset               # Reset to defaults
│
├── doctor                  # Diagnostics
│   ├── check               # Run all health checks
│   ├── storage             # Check storage health
│   ├── index               # Check index health
│   └── network             # Check network connectivity
│
└── completions             # Shell completions
    ├── bash                # Generate bash completions
    ├── zsh                 # Generate zsh completions
    ├── fish                # Generate fish completions
    └── powershell          # Generate PowerShell completions
```

---

## 4. DETAILED COMMAND SPECIFICATIONS

### 4.1 INGEST Command

```
rk-core ingest <SUBCOMMAND>

PURPOSE:
  Ingest documents into the knowledge base. Supports multiple formats
  and sources with automatic format detection.

SUBCOMMANDS:

  file <PATH>
    Ingest a single file.

    OPTIONS:
      -t, --type <TYPE>       Document type override
                              [possible values: paper, doc, code, note, auto]
                              [default: auto]

      --tags <TAGS>           Comma-separated tags

      --source <SOURCE>       Source identifier

      --no-chunk              Skip chunking (store as single document)

      --chunk-size <N>        Chunk size in tokens
                              [default: 512]

      --chunk-overlap <N>     Chunk overlap in tokens
                              [default: 50]

    EXAMPLES:
      rk-core ingest file paper.pdf
      rk-core ingest file report.md --type doc --tags "quarterly,finance"
      rk-core ingest file code.rs --chunk-size 256

  dir <PATH>
    Ingest all files in a directory.

    OPTIONS:
      -r, --recursive         Process subdirectories
                              [default: true]

      --include <GLOB>        Include files matching glob
                              [default: "*"]

      --exclude <GLOB>        Exclude files matching glob

      --parallel <N>          Parallel ingestion threads
                              [default: 4]

      --dry-run               Show what would be ingested

    EXAMPLES:
      rk-core ingest dir ./papers --include "*.pdf"
      rk-core ingest dir ./src --include "*.rs" --recursive
      rk-core ingest dir ./docs --exclude "node_modules/*" --dry-run

  url <URL>
    Ingest content from a URL.

    OPTIONS:
      --depth <N>             Crawl depth for websites
                              [default: 1]

      --follow-links          Follow internal links

      --user-agent <UA>       Custom user agent

    EXAMPLES:
      rk-core ingest url https://docs.example.com
      rk-core ingest url https://github.com/user/repo/blob/main/README.md

  github <REPO>
    Ingest a GitHub repository.

    OPTIONS:
      --branch <BRANCH>       Branch to ingest
                              [default: main]

      --include <GLOB>        Include files matching glob

      --exclude <GLOB>        Exclude files matching glob

      --token <TOKEN>         GitHub token for private repos
                              [env: GITHUB_TOKEN]

    EXAMPLES:
      rk-core ingest github anthropics/claude-code
      rk-core ingest github myorg/private-repo --token $GITHUB_TOKEN

  stdin
    Ingest content from stdin.

    OPTIONS:
      --format <FMT>          Input format
                              [possible values: text, json, jsonl]
                              [default: text]

      --meta <KEY=VALUE>      Metadata key-value pairs (repeatable)

    EXAMPLES:
      cat document.txt | rk-core ingest stdin
      curl https://api.example.com/docs | rk-core ingest stdin --format json

OUTPUT:
  Default: Progress bar with ingestion stats
  --json:  {"ingested": 10, "chunks": 245, "errors": 0, "duration_ms": 1234}
```

### 4.2 QUERY Command

```
rk-core query <QUERY> [OPTIONS]

PURPOSE:
  Query the knowledge base using natural language. Returns relevant
  documents/chunks ranked by relevance.

ARGUMENTS:
  <QUERY>                   Natural language query string

OPTIONS:
  -k, --top-k <N>           Number of results to return
                            [default: 5]

  --hybrid                  Use hybrid search (vector + BM25)
                            [default: true]

  --alpha <FLOAT>           Hybrid fusion weight (0=BM25, 1=vector)
                            [default: 0.5]

  --raptor                  Use RAPTOR tree retrieval
                            [default: false]

  --rerank                  Apply cross-encoder reranking
                            [default: false]

  --rerank-model <MODEL>    Reranking model
                            [default: cross-encoder/ms-marco-MiniLM-L-6-v2]

  -f, --filter <EXPR>       Metadata filter expression
                            Syntax: field:value, field>value, field<value
                            Multiple filters: AND logic

  --min-score <FLOAT>       Minimum similarity score
                            [default: 0.0]

  --include-content         Include full document content in output
                            [default: false]

  --explain                 Show retrieval explanation (scores, reasoning)
                            [default: false]

  --context-window <N>      Context tokens to include around matches
                            [default: 100]

EXAMPLES:
  # Basic query
  rk-core query "What is chain of thought prompting?"

  # With filters
  rk-core query "RAPTOR retrieval" --filter "type:paper" --filter "year>2023"

  # Full retrieval pipeline
  rk-core query "hybrid search techniques" --hybrid --raptor --rerank --top-k 10

  # JSON output for scripting
  rk-core query "vector databases" --json | jq '.results[0].content'

  # With explanation
  rk-core query "embedding models" --explain --top-k 3

OUTPUT (default):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Query: "What is chain of thought prompting?"                        │
  │ Results: 5 (hybrid search, 0.234s)                                  │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │ [1] Chain of Thought Prompting (Score: 0.923)                       │
  │     Source: papers/cot_prompting_2022.pdf                           │
  │     Type: paper | Year: 2022                                        │
  │     ─────────────────────────────────────────────────────────────   │
  │     "Chain-of-thought prompting enables complex reasoning           │
  │     capabilities in large language models by providing..."          │
  │                                                                     │
  │ [2] Zero-Shot Reasoners (Score: 0.891)                              │
  │     ...                                                             │
  └─────────────────────────────────────────────────────────────────────┘

OUTPUT (--json):
  {
    "query": "What is chain of thought prompting?",
    "results": [
      {
        "id": "chunk_abc123",
        "score": 0.923,
        "content": "Chain-of-thought prompting enables...",
        "metadata": {
          "source": "papers/cot_prompting_2022.pdf",
          "type": "paper",
          "year": 2022,
          "chunk_index": 3
        }
      }
    ],
    "search_type": "hybrid",
    "duration_ms": 234
  }
```

### 4.3 SEARCH Command

```
rk-core search <SUBCOMMAND>

PURPOSE:
  Direct search operations bypassing the full query pipeline.
  Useful for testing, debugging, and advanced use cases.

SUBCOMMANDS:

  vector <QUERY>
    Pure dense vector search.

    OPTIONS:
      -k, --top-k <N>         Number of results [default: 10]
      --model <MODEL>         Embedding model to use
      --threshold <FLOAT>     Minimum similarity threshold

  keyword <QUERY>
    BM25 keyword search.

    OPTIONS:
      -k, --top-k <N>         Number of results [default: 10]
      --field <FIELD>         Field to search [default: content]
      --boost <FIELD:WEIGHT>  Boost specific fields

  hybrid <QUERY>
    Combined vector + keyword search.

    OPTIONS:
      -k, --top-k <N>         Number of results [default: 10]
      --alpha <FLOAT>         Fusion weight [default: 0.5]
      --fusion <METHOD>       Fusion method [rrf, linear] [default: rrf]

  similar <DOC_ID>
    Find documents similar to a given document.

    OPTIONS:
      -k, --top-k <N>         Number of results [default: 10]
      --exclude-self          Exclude the source document
```

### 4.4 INDEX Command

```
rk-core index <SUBCOMMAND>

PURPOSE:
  Manage search indexes (both vector and BM25).

SUBCOMMANDS:

  build [OPTIONS]
    Build or rebuild indexes.

    OPTIONS:
      --force                 Force rebuild even if up-to-date
      --parallel <N>          Parallel build threads [default: 4]
      --batch-size <N>        Batch size for embedding [default: 32]

    OUTPUT:
      Building indexes...
      ├── BM25 index: 840 documents indexed
      ├── Vector index: 12,450 embeddings indexed
      └── RAPTOR tree: 3 levels built
      Done in 45.2s

  status
    Show index status.

    OUTPUT:
      ┌────────────────────────────────────────────────────┐
      │ INDEX STATUS                                       │
      ├────────────────────────────────────────────────────┤
      │ BM25 Index                                         │
      │   Documents: 840                                   │
      │   Terms: 125,432                                   │
      │   Size: 12.3 MB                                    │
      │   Last updated: 2025-12-11 18:30:00                │
      │                                                    │
      │ Vector Index (HNSW)                                │
      │   Vectors: 12,450                                  │
      │   Dimensions: 1536                                 │
      │   Size: 89.2 MB                                    │
      │   Last updated: 2025-12-11 18:30:00                │
      │                                                    │
      │ RAPTOR Tree                                        │
      │   Levels: 3                                        │
      │   Root summaries: 42                               │
      │   Last updated: 2025-12-11 18:30:00                │
      └────────────────────────────────────────────────────┘

  optimize
    Optimize indexes for better performance.

  repair
    Repair corrupted or inconsistent indexes.

  clear
    Clear all indexes (requires confirmation).
```

### 4.5 SERVE Command

```
rk-core serve [OPTIONS]

PURPOSE:
  Start the HTTP API server for programmatic access.

OPTIONS:
  -H, --host <HOST>         Bind host
                            [default: 127.0.0.1]

  -p, --port <PORT>         Bind port
                            [default: 8080]

  -w, --workers <N>         Worker threads
                            [default: number of CPUs]

  --tls                     Enable TLS

  --cert <PATH>             TLS certificate path
                            [required if --tls]

  --key <PATH>              TLS private key path
                            [required if --tls]

  --cors                    Enable CORS
                            [default: false]

  --cors-origins <ORIGINS>  Allowed CORS origins
                            [default: *]

  --api-key <KEY>           Require API key for authentication
                            [env: REASONKIT_API_KEY]

  --metrics                 Enable Prometheus metrics endpoint
                            [default: false]

  --openapi                 Serve OpenAPI spec at /openapi.json
                            [default: true]

OUTPUT:
  ┌─────────────────────────────────────────────────────────────┐
  │ ReasonKit API Server v0.1.0                                 │
  ├─────────────────────────────────────────────────────────────┤
  │ Listening: http://127.0.0.1:8080                            │
  │ Workers: 8                                                  │
  │ TLS: disabled                                               │
  │ Auth: API key required                                      │
  │                                                             │
  │ Endpoints:                                                  │
  │   POST /query         Query the knowledge base              │
  │   POST /search        Direct search                         │
  │   POST /embed         Generate embeddings                   │
  │   POST /ingest        Ingest documents                      │
  │   GET  /stats         Get statistics                        │
  │   GET  /health        Health check                          │
  │   GET  /openapi.json  OpenAPI specification                 │
  │                                                             │
  │ Press Ctrl+C to stop                                        │
  └─────────────────────────────────────────────────────────────┘
```

### 4.6 DOCTOR Command

```
rk-core doctor [SUBCOMMAND]

PURPOSE:
  Run health checks and diagnostics.

SUBCOMMANDS:

  check (default)
    Run all health checks.

    OUTPUT:
      ┌─────────────────────────────────────────────────────────────┐
      │ REASONKIT HEALTH CHECK                                      │
      ├─────────────────────────────────────────────────────────────┤
      │ ✓ Configuration valid                                       │
      │ ✓ Data directory accessible                                 │
      │ ✓ BM25 index healthy (840 documents)                        │
      │ ✓ Vector index healthy (12,450 vectors)                     │
      │ ✓ Embedding API reachable                                   │
      │ ⚠ RAPTOR tree outdated (rebuild recommended)                │
      │ ✓ Disk space sufficient (45.2 GB free)                      │
      ├─────────────────────────────────────────────────────────────┤
      │ Status: HEALTHY (1 warning)                                 │
      │ Run 'rk-core index build' to address warnings               │
      └─────────────────────────────────────────────────────────────┘

  storage
    Check storage health.

  index
    Check index integrity.

  network
    Check network connectivity (embedding APIs, etc).
```

---

## 5. EXIT CODES

```
0   - Success
1   - General error
2   - Invalid arguments
3   - Configuration error
4   - Storage error
5   - Index error
6   - Network error
7   - Permission denied
10  - Partial success (some items processed with errors)
```

---

## 6. CONFIGURATION

### 6.1 Configuration File (config.toml)

```toml
# ~/.config/reasonkit/config.toml

[general]
data_dir = "~/.local/share/reasonkit"
log_level = "info"
color = "auto"

[embedding]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 1536
batch_size = 32

[embedding.openai]
api_key = "${OPENAI_API_KEY}"
base_url = "https://api.openai.com/v1"

[embedding.local]
model_path = "~/.local/share/reasonkit/models/bge-m3"

[indexing]
chunk_size = 512
chunk_overlap = 50
enable_raptor = true
raptor_levels = 3

[search]
default_top_k = 5
hybrid_alpha = 0.5
enable_rerank = false
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

[storage]
backend = "local"  # local, qdrant

[storage.qdrant]
url = "http://localhost:6333"
collection = "reasonkit"

[server]
host = "127.0.0.1"
port = 8080
workers = 0  # 0 = auto-detect
```

### 6.2 Environment Variables

```bash
REASONKIT_CONFIG       # Config file path
REASONKIT_DATA_DIR     # Data directory
REASONKIT_LOG_LEVEL    # Log level (trace, debug, info, warn, error)
OPENAI_API_KEY         # OpenAI API key
GITHUB_TOKEN           # GitHub token for private repos
REASONKIT_API_KEY      # API key for server authentication
```

---

## 7. RUST TYPE DEFINITIONS

### 7.1 CLI Structures (clap derive)

```rust
//! CLI argument structures for rk-core

use clap::{Parser, Subcommand, Args, ValueEnum};
use std::path::PathBuf;

/// ReasonKit Core - Knowledge Base for AI Reasoning
#[derive(Parser)]
#[command(name = "rk-core")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Configuration file path
    #[arg(short, long, env = "REASONKIT_CONFIG")]
    pub config: Option<PathBuf>,

    /// Data directory
    #[arg(short, long, env = "REASONKIT_DATA_DIR")]
    pub data_dir: Option<PathBuf>,

    /// Increase verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Suppress non-essential output
    #[arg(short, long)]
    pub quiet: bool,

    /// Color output mode
    #[arg(long, value_enum, default_value = "auto")]
    pub color: ColorMode,

    /// Output in JSON format
    #[arg(long)]
    pub json: bool,

    /// Output in JSON Lines format
    #[arg(long)]
    pub jsonl: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum ColorMode {
    Auto,
    Always,
    Never,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Ingest documents into the knowledge base
    Ingest {
        #[command(subcommand)]
        source: IngestSource,
    },

    /// Query the knowledge base
    Query(QueryArgs),

    /// Direct search operations
    Search {
        #[command(subcommand)]
        search_type: SearchType,
    },

    /// Embedding operations
    Embed {
        #[command(subcommand)]
        operation: EmbedOperation,
    },

    /// Index management
    Index {
        #[command(subcommand)]
        action: IndexAction,
    },

    /// Export data
    Export {
        #[command(subcommand)]
        target: ExportTarget,
    },

    /// Import data
    Import {
        #[command(subcommand)]
        target: ImportTarget,
    },

    /// Show statistics
    Stats {
        #[command(subcommand)]
        category: Option<StatsCategory>,
    },

    /// Start API server
    Serve(ServeArgs),

    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Health checks and diagnostics
    Doctor {
        #[command(subcommand)]
        check: Option<DoctorCheck>,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        shell: clap_complete::Shell,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// INGEST
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
pub enum IngestSource {
    /// Ingest a single file
    File {
        /// Path to file
        path: PathBuf,

        #[command(flatten)]
        options: IngestOptions,
    },

    /// Ingest directory
    Dir {
        /// Path to directory
        path: PathBuf,

        /// Process subdirectories
        #[arg(short, long, default_value = "true")]
        recursive: bool,

        /// Include files matching glob
        #[arg(long)]
        include: Option<String>,

        /// Exclude files matching glob
        #[arg(long)]
        exclude: Option<String>,

        /// Parallel ingestion threads
        #[arg(long, default_value = "4")]
        parallel: usize,

        /// Show what would be ingested
        #[arg(long)]
        dry_run: bool,

        #[command(flatten)]
        options: IngestOptions,
    },

    /// Ingest from URL
    Url {
        /// URL to ingest
        url: String,

        /// Crawl depth for websites
        #[arg(long, default_value = "1")]
        depth: usize,

        /// Follow internal links
        #[arg(long)]
        follow_links: bool,

        #[command(flatten)]
        options: IngestOptions,
    },

    /// Ingest GitHub repository
    Github {
        /// Repository (owner/repo)
        repo: String,

        /// Branch to ingest
        #[arg(long, default_value = "main")]
        branch: String,

        /// Include files matching glob
        #[arg(long)]
        include: Option<String>,

        /// Exclude files matching glob
        #[arg(long)]
        exclude: Option<String>,

        /// GitHub token
        #[arg(long, env = "GITHUB_TOKEN")]
        token: Option<String>,

        #[command(flatten)]
        options: IngestOptions,
    },

    /// Ingest from stdin
    Stdin {
        /// Input format
        #[arg(long, default_value = "text")]
        format: InputFormat,

        /// Metadata key-value pairs
        #[arg(long = "meta", value_parser = parse_key_val)]
        metadata: Vec<(String, String)>,

        #[command(flatten)]
        options: IngestOptions,
    },
}

#[derive(Args)]
pub struct IngestOptions {
    /// Document type override
    #[arg(short = 't', long, default_value = "auto")]
    pub doc_type: DocumentType,

    /// Comma-separated tags
    #[arg(long)]
    pub tags: Option<String>,

    /// Source identifier
    #[arg(long)]
    pub source: Option<String>,

    /// Skip chunking
    #[arg(long)]
    pub no_chunk: bool,

    /// Chunk size in tokens
    #[arg(long, default_value = "512")]
    pub chunk_size: usize,

    /// Chunk overlap in tokens
    #[arg(long, default_value = "50")]
    pub chunk_overlap: usize,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum DocumentType {
    Auto,
    Paper,
    Doc,
    Code,
    Note,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum InputFormat {
    Text,
    Json,
    Jsonl,
}

// ═══════════════════════════════════════════════════════════════════════════
// QUERY
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Args)]
pub struct QueryArgs {
    /// Query string
    pub query: String,

    /// Number of results
    #[arg(short = 'k', long, default_value = "5")]
    pub top_k: usize,

    /// Use hybrid search
    #[arg(long, default_value = "true")]
    pub hybrid: bool,

    /// Hybrid fusion weight (0=BM25, 1=vector)
    #[arg(long, default_value = "0.5")]
    pub alpha: f32,

    /// Use RAPTOR tree retrieval
    #[arg(long)]
    pub raptor: bool,

    /// Apply reranking
    #[arg(long)]
    pub rerank: bool,

    /// Reranking model
    #[arg(long)]
    pub rerank_model: Option<String>,

    /// Metadata filter (repeatable)
    #[arg(short = 'f', long = "filter")]
    pub filters: Vec<String>,

    /// Minimum similarity score
    #[arg(long, default_value = "0.0")]
    pub min_score: f32,

    /// Include full content
    #[arg(long)]
    pub include_content: bool,

    /// Show retrieval explanation
    #[arg(long)]
    pub explain: bool,

    /// Context window around matches
    #[arg(long, default_value = "100")]
    pub context_window: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// SEARCH
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
pub enum SearchType {
    /// Dense vector search
    Vector {
        query: String,
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        threshold: Option<f32>,
    },

    /// BM25 keyword search
    Keyword {
        query: String,
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
        #[arg(long, default_value = "content")]
        field: String,
    },

    /// Hybrid search
    Hybrid {
        query: String,
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
        #[arg(long, default_value = "0.5")]
        alpha: f32,
        #[arg(long, default_value = "rrf")]
        fusion: FusionMethod,
    },

    /// Find similar documents
    Similar {
        doc_id: String,
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
        #[arg(long)]
        exclude_self: bool,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum FusionMethod {
    Rrf,
    Linear,
}

// ═══════════════════════════════════════════════════════════════════════════
// SERVE
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Args)]
pub struct ServeArgs {
    /// Bind host
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    pub host: String,

    /// Bind port
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// Worker threads (0 = auto)
    #[arg(short, long, default_value = "0")]
    pub workers: usize,

    /// Enable TLS
    #[arg(long)]
    pub tls: bool,

    /// TLS certificate path
    #[arg(long, required_if_eq("tls", "true"))]
    pub cert: Option<PathBuf>,

    /// TLS private key path
    #[arg(long, required_if_eq("tls", "true"))]
    pub key: Option<PathBuf>,

    /// Enable CORS
    #[arg(long)]
    pub cors: bool,

    /// CORS allowed origins
    #[arg(long, default_value = "*")]
    pub cors_origins: String,

    /// API key for authentication
    #[arg(long, env = "REASONKIT_API_KEY")]
    pub api_key: Option<String>,

    /// Enable Prometheus metrics
    #[arg(long)]
    pub metrics: bool,

    /// Serve OpenAPI spec
    #[arg(long, default_value = "true")]
    pub openapi: bool,
}

// Additional subcommands omitted for brevity...
// Full implementation would include all remaining enums

/// Parse key=value pairs
fn parse_key_val(s: &str) -> Result<(String, String), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}
```

---

## 8. SHELL COMPLETIONS

### 8.1 Installation

```bash
# Bash
rk-core completions bash > ~/.local/share/bash-completion/completions/rk-core

# Zsh
rk-core completions zsh > ~/.zfunc/_rk-core

# Fish
rk-core completions fish > ~/.config/fish/completions/rk-core.fish

# PowerShell
rk-core completions powershell >> $PROFILE
```

---

## 9. INTEGRATION EXAMPLES

### 9.1 Scripting Examples

```bash
#!/bin/bash
# Example: Ingest all PDFs and query

# Ingest
rk-core ingest dir ./papers --include "*.pdf" --parallel 8

# Query and process results
rk-core query "neural network architectures" --json --top-k 20 | \
  jq -r '.results[] | "\(.score)\t\(.metadata.source)"' | \
  sort -rn | head -10

# Export for backup
rk-core export all --output backup_$(date +%Y%m%d).tar.gz
```

### 9.2 Pipeline Integration

```bash
# LLM CLI integration
rk-core query "RAG techniques" --json | \
  jq -r '.results[0].content' | \
  llm -s "Summarize this research finding"

# GitHub Actions
rk-core doctor check --json | jq -e '.status == "healthy"'
```

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-11 | Initial design specification |

---

*"Great CLIs are discovered, not designed."*
*- ReasonKit Engineering*
