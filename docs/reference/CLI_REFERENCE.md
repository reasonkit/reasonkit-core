# ReasonKit Core CLI Reference

> Quick reference guide for the `rk` command-line interface

**Version:** 1.0.0
**License:** Apache 2.0

---

## Status

Implemented:

- `mcp`
- `serve-mcp`
- `completions`

Planned (scaffolded; returns "not implemented" in v1.0.0):

- `ingest`, `query`, `think`, `index`, `stats`, `export`, `serve`, `web`, `verify`, `trace`, `rag`, `metrics`

---

## Installation

```bash
# Install from source
cargo install --path reasonkit-core

# Or via Cargo (when published)
cargo install reasonkit-core

# Verify installation
rk --version
```

---

## Global Options

Available for all commands:

```bash
rk [OPTIONS] <COMMAND>

OPTIONS:
  -v, --verbose              Increase verbosity (-v=INFO, -vv=DEBUG, -vvv=TRACE)
  -c, --config <PATH>        Configuration file path [env: REASONKIT_CONFIG]
  -d, --data-dir <PATH>      Data directory [default: ./data] [env: REASONKIT_DATA_DIR]
  -h, --help                 Print help
  -V, --version              Print version
```

### Examples

```bash
# Debug logging
rk -vv mcp list-servers

# Custom data directory
rk -d /mnt/data/reasonkit mcp list-tools

# Use config file
rk -c ~/.reasonkit/config.toml mcp list-prompts
```

---

## Commands

### 1. ingest - Ingest Documents (planned)

Ingest documents (PDF, Markdown, HTML, JSON) into the knowledge base.

#### Syntax

```bash
rk ingest <PATH> [OPTIONS]
```

#### Arguments

- `<PATH>` - Path to document file or directory

#### Options

- `--doc-type <TYPE>` - Document type [default: paper]
  - `paper` - Academic papers
  - `documentation` - Technical docs
  - `code` - Source code
  - `note` - Personal notes
  - `transcript` - Meeting/interview transcripts
  - `benchmark` - Benchmark data

- `-r, --recursive` - Process directories recursively

#### Examples

```bash
# Ingest a single PDF paper
rk ingest paper.pdf

# Ingest all PDFs in a directory
rk ingest ./papers --doc-type paper --recursive

# Ingest documentation
rk ingest ./docs --doc-type documentation --recursive

# With verbose logging
rk -vv ingest ./data/papers --recursive
```

#### Output

```
Ingesting: paper.pdf
✓ Parsed 15 pages, 8,432 words
✓ Created 42 chunks
✓ Embedded 42 chunks
✓ Indexed in BM25 and vector stores
✓ Completed in 3.2s

Total: 1 document, 42 chunks
```

---

### 2. query - Search Knowledge Base (planned)

Search the knowledge base using semantic (vector) or hybrid (vector + BM25) search.

#### Syntax

```bash
rk query <QUERY> [OPTIONS]
```

#### Arguments

- `<QUERY>` - Search query string

#### Options

- `-k, --top-k <N>` - Number of results to return [default: 5]
- `--hybrid` - Use hybrid search (dense + sparse)
- `--raptor` - Use RAPTOR tree retrieval
- `-f, --format <FORMAT>` - Output format [default: text]
  - `text` - Human-readable text
  - `json` - JSON format
  - `markdown` - Markdown format

#### Examples

```bash
# Basic semantic search
rk query "What is chain-of-thought reasoning?"

# Get top 10 results
rk query "quantum computing applications" --top-k 10

# Hybrid search (vector + BM25)
rk query "transformer architecture" --hybrid

# RAPTOR tree retrieval (for long documents)
rk query "explain the full paper methodology" --raptor --top-k 5

# JSON output for scripting
rk query "neural networks" --format json > results.json

# Markdown output
rk query "attention mechanism" --format markdown
```

#### Output (text format)

```
Query: What is chain-of-thought reasoning?
Retrieved 5 results in 45ms

[1] Score: 0.872 | Source: dense
Chain-of-thought (CoT) prompting is a method that improves large language
model reasoning by encouraging the model to generate intermediate reasoning
steps. Introduced by Wei et al. (2022), it shows that LLMs can perform
complex multi-step reasoning when prompted to "think step by step."

[2] Score: 0.845 | Source: hybrid
...
```

#### Output (JSON format)

```json
{
  "query": "What is chain-of-thought reasoning?",
  "results": [
    {
      "score": 0.872,
      "document_id": "uuid-here",
      "chunk": {
        "id": "chunk-uuid",
        "text": "Chain-of-thought (CoT) prompting...",
        "index": 15,
        "section": "Introduction",
        "page": 3
      },
      "match_source": "dense"
    }
  ],
  "retrieval_time_ms": 45
}
```

---

### 3. think - Execute ThinkTool Protocols (planned)

Execute structured reasoning protocols using LLMs.

#### Syntax

```bash
rk think [QUERY] [OPTIONS]
```

#### Arguments

- `<QUERY>` - The input query or question (optional if using `--list`)

#### Options

**Protocol Selection:**

- `-p, --protocol <NAME>` - Execute specific protocol
  - `gigathink` (gt) - Expansive creative thinking
  - `laserlogic` (ll) - Precision deductive reasoning
  - `bedrock` (br) - First principles decomposition
  - `proofguard` (pg) - Multi-source verification
  - `brutalhonesty` (bh) - Adversarial self-critique

- `--profile <NAME>` - Execute reasoning profile
  - `quick` - Fast 3-step analysis (70% confidence)
  - `balanced` - Standard 5-module chain (80% confidence)
  - `deep` - Thorough analysis (85% confidence)
  - `paranoid` - Maximum verification (95% confidence)

**LLM Configuration:**

- `--provider <PROVIDER>` - LLM provider [default: anthropic]
  - `anthropic` - Anthropic Claude
  - `openai` - OpenAI GPT
  - `openrouter` - OpenRouter (300+ models)
  - `groq` - Groq (ultra-fast)
  - `gemini` - Google Gemini
  - `xai` - xAI Grok
  - `deepseek` - DeepSeek
  - `mistral` - Mistral AI

- `-m, --model <MODEL>` - Specific model name
- `-t, --temperature <TEMP>` - Generation temperature [default: 0.7]
- `--max-tokens <N>` - Maximum tokens [default: 2000]

**Budget Control:**

- `-b, --budget <BUDGET>` - Compute budget
  - Time: `30s`, `5m`, `1h`
  - Tokens: `1000t`, `5000tokens`
  - Cost: `$0.50`, `$1.00`

**Tracing:**

- `--save-trace` - Save execution trace
- `--trace-dir <DIR>` - Trace output directory [default: ./traces]

**Other:**

- `--mock` - Use mock LLM (for testing)
- `--list` - List available protocols and profiles

#### Examples

```bash
# Execute specific protocol
rk think "Should we adopt microservices?" --protocol gigathink

# Execute profile (multiple protocols)
rk think "Evaluate this startup idea" --profile balanced

# Custom LLM configuration
rk think "Analyze this code" \
  --protocol laserlogic \
  --provider openai \
  --model gpt-4-turbo \
  --temperature 0.3

# With time budget
rk think "Research quantum computing" \
  --profile deep \
  --budget "60s"

# With cost budget
rk think "Complex analysis task" \
  --protocol proofguard \
  --budget "$0.50"

# Save execution trace
rk think "Is this architecture sound?" \
  --profile paranoid \
  --save-trace \
  --trace-dir ./analysis-traces

# List available options
rk think --list

# Use ultra-fast Groq
rk think "Quick code review" \
  --protocol laserlogic \
  --provider groq \
  --model llama-3.3-70b-versatile

# Use OpenRouter for access to 300+ models
rk think "Deep analysis" \
  --provider openrouter \
  --model anthropic/claude-opus-4
```

#### Output

```
Protocol: GigaThink
Model: claude-sonnet-4 (Anthropic)
Budget: 60s (remaining: 45s)

Step 1/5: Expansive Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% [2.3s]
✓ Generated 10 perspectives

Step 2/5: Deductive Reasoning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% [1.8s]
✓ Validated 8/10 perspectives

Step 3/5: First Principles
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% [3.1s]
✓ Decomposed to 4 core axioms

Step 4/5: Verification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% [2.4s]
✓ Cross-verified 3 sources

Step 5/5: Synthesis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% [1.9s]
✓ Generated final output

═══════════════════════════════════════════════════════════
RESULT

Confidence: 82%

Key Findings:
1. [First finding with rationale]
2. [Second finding with supporting evidence]
3. [Third finding with alternatives considered]

Reasoning Chain:
→ Perspective A: [summary]
→ Perspective B: [summary]
→ Core Axiom 1: [axiom]
→ Verification: 3/3 sources agree

═══════════════════════════════════════════════════════════

Execution Summary:
  Total Time: 11.5s
  Total Tokens: 3,842 (2,140 in, 1,702 out)
  Total Cost: $0.042
  Trace: ./traces/analysis-2025-12-23-012345.json
```

---

### 4. index - Manage Index (planned)

Manage BM25 and vector indexes.

#### Syntax

```bash
rk index <ACTION>
```

#### Actions

- `rebuild` - Rebuild all indexes from scratch
- `stats` - Show index statistics
- `optimize` - Optimize indexes for performance

#### Examples

```bash
# Rebuild indexes
rk index rebuild

# Show statistics
rk index stats

# Optimize for performance
rk index optimize
```

#### Output (stats)

```
Index Statistics
═══════════════

BM25 Index:
  Documents: 43
  Total terms: 156,832
  Unique terms: 12,456
  Index size: 45 MB
  Avg doc length: 3,647 tokens

Vector Index (HNSW):
  Vectors: 1,247
  Dimension: 1536
  M (connections): 16
  ef_construction: 200
  Index size: 98 MB
  Avg query time: 12ms

Last rebuild: 2025-12-23 01:15:22 UTC
```

---

### 5. stats - Knowledge Base Statistics (planned)

Display comprehensive statistics about the knowledge base.

#### Syntax

```bash
rk stats
```

#### Examples

```bash
# Show all stats
rk stats

# With verbose logging
rk -v stats
```

#### Output

```
═══════════════════════════════════════════════════════════
ReasonKit Core Statistics
═══════════════════════════════════════════════════════════

Documents
─────────────────────────────────────────────────────────
  Total: 43 documents
  By type:
    Papers:        38 (88%)
    Documentation:  5 (12%)
    Code:          0 (0%)

Content
─────────────────────────────────────────────────────────
  Total chunks: 1,247
  Total tokens: ~312,000
  Total words:  234,567
  Avg words/doc: 5,455

Embeddings
─────────────────────────────────────────────────────────
  Dense:  1,247 (1536-dim)
  Sparse: 1,247 (BM25)
  Model:  text-embedding-3-small (OpenAI)

Storage
─────────────────────────────────────────────────────────
  Backend: Qdrant
  Collections: 2
  Total size: 243 MB
    - Documents: 15 MB
    - Vectors:  228 MB

Index
─────────────────────────────────────────────────────────
  BM25 index:   45 MB
  HNSW index:   98 MB
  Total index: 143 MB

RAPTOR
─────────────────────────────────────────────────────────
  Trees: 8
  Total nodes: 342
  Max depth: 3
  Avg cluster size: 12

Last updated: 2025-12-23 01:30:45 UTC
═══════════════════════════════════════════════════════════
```

---

### 6. export - Export Data (planned)

Export documents and metadata to JSON or JSONL format.

#### Syntax

```bash
rk export <OUTPUT> [OPTIONS]
```

#### Arguments

- `<OUTPUT>` - Output file path

#### Options

- `-f, --format <FORMAT>` - Export format [default: jsonl]
  - `jsonl` - JSON Lines (one document per line)
  - `json` - Pretty-printed JSON array

#### Examples

```bash
# Export to JSONL (recommended for large datasets)
rk export documents.jsonl

# Export to JSON
rk export backup.json --format json

# Export with compression (pipe to gzip)
rk export documents.jsonl | gzip > documents.jsonl.gz
```

#### JSONL Output Format

```jsonl
{"id":"uuid-1","doc_type":"paper","source":{...},"content":{...},"metadata":{...},"chunks":[...]}
{"id":"uuid-2","doc_type":"paper","source":{...},"content":{...},"metadata":{...},"chunks":[...]}
```

#### JSON Output Format

```json
[
  {
    "id": "uuid-1",
    "doc_type": "paper",
    "source": { ... },
    "content": { ... },
    "metadata": { ... },
    "chunks": [ ... ]
  },
  { ... }
]
```

---

### 7. serve - Start API Server (planned)

Start HTTP API server for programmatic access.

#### Syntax

```bash
rk serve [OPTIONS]
```

#### Options

- `--host <HOST>` - Host to bind to [default: 127.0.0.1]
- `-p, --port <PORT>` - Port to bind to [default: 8080]

#### Examples

```bash
# Start on default port
rk serve

# Custom host and port
rk serve --host 0.0.0.0 --port 3000

# Bind to all interfaces
rk serve --host 0.0.0.0

# With verbose logging
rk -vv serve
```

#### Output

```
ReasonKit Core API Server
═══════════════════════════════════════
Version: 1.0.0
Listening on: http://127.0.0.1:8080

Endpoints:
  GET  /health              Health check
  POST /query               Search query
  POST /ingest              Ingest document
  GET  /stats               Statistics
  POST /think               Execute protocol

Press Ctrl+C to stop
```

#### API Endpoints

**Health Check**

```bash
curl http://localhost:8080/health
# {"status": "ok", "version": "1.0.0"}
```

**Query**

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "chain of thought", "top_k": 5, "hybrid": true}'
```

**Think (Protocol Execution)**

```bash
curl -X POST http://localhost:8080/think \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Should we adopt microservices?",
    "protocol": "gigathink",
    "provider": "anthropic"
  }'
```

---

## Environment Variables

Configure ReasonKit Core via environment variables:

### Paths

```bash
export REASONKIT_DATA_DIR="./data"           # Data directory
export REASONKIT_CONFIG="~/.reasonkit/config.toml"  # Config file
```

### API Keys

```bash
# Embedding providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Vector database
export QDRANT_API_KEY="..."
export QDRANT_URL="http://localhost:6333"

# LLM providers (for ThinkTools)
export OPENROUTER_API_KEY="sk-or-..."
export GROQ_API_KEY="gsk_..."
export XAI_API_KEY="xai-..."
export MISTRAL_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GEMINI_API_KEY="..."
```

### Configuration

```bash
# Logging
export RUST_LOG="info"                # info, debug, trace
export RUST_BACKTRACE="1"            # Enable backtraces

# Performance
export REASONKIT_EMBEDDING_BATCH_SIZE="100"
export REASONKIT_MAX_WORKERS="8"
```

---

## Configuration File

Create `~/.reasonkit/config.toml` or use `-c` flag:

```toml
[storage]
backend = "qdrant"
qdrant_url = "http://localhost:6333"
qdrant_api_key_env = "QDRANT_API_KEY"

[embedding]
model = "text-embedding-3-small"
dimension = 1536
api_key_env = "OPENAI_API_KEY"
batch_size = 100
enable_cache = true
cache_ttl_secs = 86400

[retrieval]
top_k = 10
min_score = 0.5
alpha = 0.7              # 0=sparse only, 1=dense only
fusion_strategy = "rrf"  # rrf, weighted_sum, comb_sum, comb_mnz
use_raptor = false

[thinktool]
default_provider = "anthropic"
default_model = "claude-sonnet-4"
temperature = 0.7
max_tokens = 2000
save_traces = true
trace_dir = "~/.reasonkit/traces"

[thinktool.budget]
default_time_secs = 60
default_token_limit = 10000
default_cost_usd = 1.0
```

---

## Common Workflows

### 1. Build a Knowledge Base from Papers

```bash
# Download papers
mkdir -p ./data/papers
# ... download PDFs to ./data/papers

# Ingest all papers
rk ingest ./data/papers --doc-type paper --recursive

# Verify ingestion
rk stats

# Query the knowledge base
rk query "What is tree-of-thought prompting?" --hybrid --top-k 5
```

### 2. Execute Structured Reasoning

```bash
# Quick analysis
rk think "Is this code safe to deploy?" --profile quick

# Deep analysis with verification
rk think "Evaluate this architecture decision" \
  --profile paranoid \
  --save-trace

# Budget-limited analysis
rk think "Research this topic" \
  --profile deep \
  --budget "2m" \
  --budget "$0.50"
```

### 3. Export and Backup

```bash
# Export to JSONL
rk export backup-$(date +%Y%m%d).jsonl

# Compress backup
rk export backup.jsonl | gzip > backup.jsonl.gz

# Restore from backup
gunzip -c backup.jsonl.gz | rk import -
```

### 4. API Server Deployment

```bash
# Development server
rk serve

# Production server (bind to all interfaces)
rk serve --host 0.0.0.0 --port 8080

# Behind nginx reverse proxy
rk serve --host 127.0.0.1 --port 8080
```

---

## Exit Codes

| Code | Meaning                        |
| ---- | ------------------------------ |
| 0    | Success                        |
| 1    | General error                  |
| 2    | Configuration error            |
| 3    | Storage/database error         |
| 4    | Embedding error (API failure)  |
| 5    | LLM error (protocol execution) |
| 6    | Not found (document/index)     |

---

## Troubleshooting

### Issue: "Qdrant connection failed"

```bash
# Check Qdrant is running
docker ps | grep qdrant

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Or use in-memory storage
export REASONKIT_STORAGE_BACKEND="memory"
```

### Issue: "OpenAI API error: rate limit"

```bash
# Reduce batch size
export REASONKIT_EMBEDDING_BATCH_SIZE="20"

# Add delays between requests (edit config.toml)
[embedding]
batch_delay_ms = 100
```

### Issue: "Out of memory during ingestion"

```bash
# Process in smaller batches
rk ingest ./papers/batch1 --recursive
rk ingest ./papers/batch2 --recursive

# Reduce chunk size (edit config.toml)
[processing]
chunk_size = 300
chunk_overlap = 50
```

### Issue: "Protocol execution timeout"

```bash
# Increase budget
rk think "query" --protocol gigathink --budget "5m"

# Use faster provider
rk think "query" --protocol laserlogic --provider groq
```

---

## Performance Tips

1. **Batch ingestion**: Ingest documents in batches of 100-500
2. **Enable caching**: Set `enable_cache = true` in config
3. **Tune alpha**: Lower (0.3-0.5) for keyword queries, higher (0.7-0.9) for semantic
4. **Use Groq for speed**: `--provider groq` for 10x faster inference
5. **RAPTOR for long docs**: Use `--raptor` for documents >10k tokens
6. **Parallel workers**: Set `REASONKIT_MAX_WORKERS` to CPU count

---

## See Also

- **API Reference**: `docs/reference/API_REFERENCE.md`
- **Architecture**: `ARCHITECTURE.md`
- **ThinkTools**: `docs/thinktools/THINKTOOLS_QUICK_REFERENCE.md`
- **Project Docs**: `CLAUDE.md`

---

**Last Updated:** 2025-12-23
**Version:** 1.0.0
