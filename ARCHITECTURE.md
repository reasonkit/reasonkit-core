# ReasonKit Core - Knowledge Base Architecture

## Rust-First RAG/Vector System for AI Reasoning Enhancement

> **STATUS**: DESIGN DRAFT - LOCAL ONLY (Not yet on GitHub)
> **VERSION**: 1.0.0
> **DATE**: 2025-12-11

---

## 1. ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REASONKIT-CORE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 5: RETRIEVAL & QUERY                                          │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │   │
│  │  │ HYBRID SEARCH│ │ RAPTOR       │ │ RERANKING    │                 │   │
│  │  │ (BM25+Vector)│ │ (Tree Query) │ │ (ColBERT)    │                 │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│  ┌─────────────────────────────────┴───────────────────────────────────┐   │
│  │ LAYER 4: INDEXING                                                    │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │   │
│  │  │ HNSW INDEX   │ │ BM25 INDEX   │ │ RAPTOR TREE  │                 │   │
│  │  │ (hnswlib-rs) │ │ (tantivy)    │ │ (hierarchical)│                │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│  ┌─────────────────────────────────┴───────────────────────────────────┐   │
│  │ LAYER 3: EMBEDDING                                                   │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │   │
│  │  │ DENSE EMBED  │ │ SPARSE EMBED │ │ COLBERT      │                 │   │
│  │  │ (E5/BGE)     │ │ (SPLADE)     │ │ (Late Inter.)│                 │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│  ┌─────────────────────────────────┴───────────────────────────────────┐   │
│  │ LAYER 2: PROCESSING                                                  │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │   │
│  │  │ CHUNKING     │ │ CLEANING     │ │ METADATA     │                 │   │
│  │  │ (semantic)   │ │ (normalize)  │ │ (extraction) │                 │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│  ┌─────────────────────────────────┴───────────────────────────────────┐   │
│  │ LAYER 1: INGESTION                                                   │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐  │   │
│  │  │ PDF          │ │ HTML/MD      │ │ JSON/JSONL   │ │ GITHUB     │  │   │
│  │  │ (pdf_oxide)  │ │ (pulldown)   │ │ (serde)      │ │ (octocrab) │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STORAGE: Qdrant (Primary) | DuckDB (Metadata) | JSONL (Raw)         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. TECHNOLOGY DECISIONS

### 2.1 Vector Database: Qdrant (PRIMARY)

**Rationale:**

- Written in Rust (aligns with Rust-First philosophy)
- Highest QPS (1,200+) and lowest latency (1.6ms) in benchmarks
- Hybrid search support (dense + sparse vectors)
- Excellent filtering capabilities
- 24x compression with asymmetric quantization

**Alternatives Considered:**
| Option | Verdict | Reason |
|--------|---------|--------|
| Milvus | DEFER | Better for billion-scale, overkill for our needs |
| LanceDB | KEEP AS OPTION | Good for edge/embedded use cases |
| ChromaDB | REJECT | Python-first, slower |

### 2.2 Full-Text Search: Tantivy

**Rationale:**

- Rust-native full-text search engine
- BM25 implementation for hybrid search
- 10x+ faster than Lucene in some benchmarks
- Apache 2.0 license

### 2.3 Embedding Strategy: Hybrid

```yaml
dense_embedding:
  model: "BAAI/bge-m3" # or "intfloat/e5-large-v2"
  dimensions: 1024
  multilingual: true
  use_case: "Semantic similarity"

sparse_embedding:
  model: "naver/splade-v3"
  use_case: "Keyword/exact match"

late_interaction:
  model: "jina-colbert-v2"
  use_case: "High-precision reranking"
  dimensions: "128 per token"
```

### 2.4 Document Processing: Rust Libraries

| Format     | Library                 | Notes                          |
| ---------- | ----------------------- | ------------------------------ |
| PDF        | `pdf_oxide`             | 47.9x faster than alternatives |
| HTML       | `scraper` + `html5ever` | Rust-native                    |
| Markdown   | `pulldown-cmark`        | CommonMark compliant           |
| JSON/JSONL | `serde_json`            | Standard                       |
| EPUB       | `epub-rs`               | For documentation              |

### 2.5 RAPTOR Implementation

Based on [RAPTOR paper](https://arxiv.org/abs/2401.18059) (ICLR 2024):

```
RAPTOR Tree Structure:
                    [ROOT SUMMARY]
                    /            \
         [CLUSTER A]              [CLUSTER B]
         /    |    \              /    |    \
      [C1]  [C2]  [C3]         [C4]  [C5]  [C6]
      /|\   /|\   /|\          /|\   /|\   /|\
    chunks...                chunks...
```

**Benefits:**

- +20% absolute accuracy on QuALITY benchmark
- Captures both fine-grained and high-level understanding
- State-of-the-art on multi-hop reasoning tasks

---

## 3. FILE FORMAT STRATEGY

### 3.1 JSON as Primary Format (Confirmed)

**Rationale:**

- Rust has excellent JSON support (`serde_json`)
- Human-readable for debugging
- Schema validation with JSON Schema
- Easy to process incrementally (JSONL)
- Widely supported by all tools

### 3.2 Data Schemas

#### 3.2.1 Document Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["id", "type", "content", "metadata"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique document identifier (UUID)"
    },
    "type": {
      "type": "string",
      "enum": ["paper", "documentation", "code", "note"]
    },
    "source": {
      "type": "object",
      "properties": {
        "url": { "type": "string" },
        "path": { "type": "string" },
        "retrieved_at": { "type": "string", "format": "date-time" }
      }
    },
    "content": {
      "type": "object",
      "properties": {
        "raw": { "type": "string" },
        "chunks": {
          "type": "array",
          "items": { "$ref": "#/definitions/chunk" }
        }
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "title": { "type": "string" },
        "authors": { "type": "array", "items": { "type": "string" } },
        "date": { "type": "string" },
        "tags": { "type": "array", "items": { "type": "string" } },
        "citations": { "type": "integer" },
        "venue": { "type": "string" }
      }
    }
  },
  "definitions": {
    "chunk": {
      "type": "object",
      "properties": {
        "id": { "type": "string" },
        "text": { "type": "string" },
        "start_char": { "type": "integer" },
        "end_char": { "type": "integer" },
        "embedding_id": { "type": "string" }
      }
    }
  }
}
```

#### 3.2.2 Embedding Schema

```json
{
  "type": "object",
  "required": ["id", "chunk_id", "vector", "model"],
  "properties": {
    "id": { "type": "string" },
    "chunk_id": { "type": "string" },
    "document_id": { "type": "string" },
    "vector": {
      "type": "array",
      "items": { "type": "number" }
    },
    "model": { "type": "string" },
    "dimensions": { "type": "integer" },
    "created_at": { "type": "string", "format": "date-time" }
  }
}
```

### 3.3 Storage Layout

```
data/
├── papers/
│   ├── raw/              # Original PDFs
│   │   └── arxiv_2401.18059.pdf
│   └── processed/        # Extracted JSON
│       └── arxiv_2401.18059.json
├── docs/
│   ├── raw/              # Original HTML/MD
│   │   └── claude-code/
│   └── processed/        # Extracted JSON
│       └── claude-code.jsonl
├── embeddings/
│   ├── dense/            # Dense vector embeddings
│   │   └── bge-m3/
│   └── sparse/           # Sparse embeddings
│       └── splade/
├── indexes/
│   ├── hnsw/             # HNSW index files
│   ├── bm25/             # Tantivy index
│   └── raptor/           # RAPTOR tree structure
└── metadata/
    └── catalog.json      # Master document catalog
```

---

## 4. ACADEMIC PAPERS TO DOWNLOAD

### 4.1 Core Reasoning Papers (Priority 1)

| Paper                                   | arXiv      | Status  |
| --------------------------------------- | ---------- | ------- |
| Chain-of-Thought Prompting (Wei et al.) | 2201.11903 | PENDING |
| Self-Consistency (Wang et al.)          | 2203.11171 | PENDING |
| Tree of Thoughts (Yao et al.)           | 2305.10601 | PENDING |
| RAPTOR (Sarthi et al.)                  | 2401.18059 | PENDING |
| Let's Verify Step by Step (OpenAI)      | -          | PENDING |
| Reflexion (Shinn et al.)                | 2303.11366 | PENDING |
| Constitutional AI (Anthropic)           | 2212.08073 | PENDING |

### 4.2 Retrieval & Embedding Papers (Priority 2)

| Paper                       | arXiv      | Status  |
| --------------------------- | ---------- | ------- |
| ColBERT (Khattab & Zaharia) | 2004.12832 | PENDING |
| E5 Embeddings               | 2212.03533 | PENDING |
| BGE-M3                      | 2402.03216 | PENDING |
| Semantic Entropy (Nature)   | -          | PENDING |

### 4.3 Benchmark Papers (Priority 3)

| Paper          | arXiv      | Status  |
| -------------- | ---------- | ------- |
| GSM8K          | 2110.14168 | PENDING |
| MATH Benchmark | 2103.03874 | PENDING |
| MMLU           | 2009.03300 | PENDING |

---

## 5. DOCUMENTATION TO INDEX

### 5.1 CLI Tools (Priority 1)

| Tool                | Source                                              | Status  |
| ------------------- | --------------------------------------------------- | ------- |
| Claude Code         | https://github.com/anthropics/claude-code           | PENDING |
| Gemini CLI          | https://github.com/google-gemini/gemini-cli         | PENDING |
| OpenAI Codex        | https://github.com/openai/codex                     | PENDING |
| MCP Servers         | https://github.com/modelcontextprotocol/servers     | PENDING |
| Sequential Thinking | modelcontextprotocol/servers/src/sequentialthinking | PENDING |

### 5.2 APIs & SDKs (Priority 2)

| API           | Source                           | Status  |
| ------------- | -------------------------------- | ------- |
| Anthropic API | https://docs.anthropic.com       | PENDING |
| OpenAI API    | https://platform.openai.com/docs | PENDING |
| Google AI     | https://ai.google.dev/docs       | PENDING |
| OpenRouter    | https://openrouter.ai/docs       | PENDING |

### 5.3 Frameworks (Priority 3)

| Framework  | Source                            | Status  |
| ---------- | --------------------------------- | ------- |
| LangChain  | https://python.langchain.com/docs | PENDING |
| LlamaIndex | https://docs.llamaindex.ai        | PENDING |
| DSPy       | https://dspy-docs.vercel.app      | PENDING |

---

## 6. IMPLEMENTATION PLAN

### Phase 1: Foundation (Week 1)

```
□ Set up Cargo workspace
□ Implement PDF ingestion (pdf_oxide)
□ Implement JSON serialization (serde)
□ Create document schema validation
□ Download first batch of papers
```

### Phase 2: Processing (Week 2)

```
□ Implement semantic chunking
□ Create metadata extraction
□ Set up Tantivy for BM25
□ Implement basic retrieval
```

### Phase 3: Embedding (Week 3)

```
□ Integrate embedding model (ONNX or API)
□ Set up Qdrant (local mode)
□ Implement HNSW indexing
□ Create hybrid search
```

### Phase 4: RAPTOR (Week 4)

```
□ Implement clustering (GMM)
□ Create summarization pipeline
□ Build hierarchical tree
□ Implement tree-based retrieval
```

---

## 7. RUST DEPENDENCIES (Cargo.toml)

```toml
[package]
name = "reasonkit-core"
version = "1.0.0"
edition = "2024"

[dependencies]
# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# PDF Processing
pdf_oxide = "0.1"  # or lopdf for control

# Text Processing
pulldown-cmark = "0.9"  # Markdown
scraper = "0.17"        # HTML
regex = "1.10"

# Vector DB
qdrant-client = "1.8"

# Full-text Search
tantivy = "0.21"

# HNSW Index
hnswlib-rs = "0.3"

# Async Runtime
tokio = { version = "1", features = ["full"] }

# HTTP Client
reqwest = { version = "0.11", features = ["json"] }

# CLI
clap = { version = "4", features = ["derive"] }

# Error Handling
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# UUID
uuid = { version = "1", features = ["v4", "serde"] }

# Date/Time
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
criterion = "0.5"
```

---

## 8. BENCHMARKING TARGETS

| Operation       | Target                | Notes                     |
| --------------- | --------------------- | ------------------------- |
| PDF extraction  | <100ms per page       | pdf_oxide claims 53ms/PDF |
| Chunking        | <10ms per 1000 tokens |                           |
| Embedding (API) | <500ms per chunk      | Network bound             |
| HNSW search     | <10ms top-100         | hnswlib benchmark         |
| BM25 search     | <5ms                  | Tantivy benchmark         |
| Hybrid search   | <20ms total           | Combined                  |

---

## 9. OPEN QUESTIONS

1. **Embedding model hosting**: Local ONNX or API?
   - API: Simpler, more accurate (GPT/Claude embeddings)
   - Local: Faster, no cost, works offline
   - **DECISION NEEDED**

2. **Qdrant deployment**: Embedded or server mode?
   - Embedded: Simpler, single binary
   - Server: More scalable, dashboard
   - **RECOMMENDATION**: Start embedded, upgrade if needed

3. **RAPTOR summarization**: Which LLM?
   - GPT-4o: High quality, cost
   - Claude Haiku: Good quality, lower cost
   - Local (Llama): Free, lower quality
   - **RECOMMENDATION**: Claude Haiku for balance

---

**END OF DOCUMENT**
