# ReasonKit Core API Documentation

**Version:** 1.0.0
**License:** Apache 2.0
**Repository:** https://github.com/reasonkit/reasonkit-core

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Architecture](#core-architecture)
4. [Core Types](#core-types)
5. [Main Entry Points](#main-entry-points)
6. [Module Reference](#module-reference)
7. [Configuration](#configuration)
8. [Usage Examples](#usage-examples)
9. [Error Handling](#error-handling)

---

## Overview

ReasonKit Core is a Rust-first knowledge base and RAG (Retrieval-Augmented Generation) system designed for AI reasoning enhancement. It provides a complete pipeline for document ingestion, embedding, indexing, retrieval, and LLM-powered query answering.

### Key Features

- **Document Ingestion**: PDF, Markdown, HTML, JSON/JSONL processing
- **Embedding**: OpenAI API integration with caching, local ONNX models (optional)
- **Hybrid Search**: BM25 (Tantivy) + Dense Vector Search (Qdrant)
- **RAG Engine**: Combines retrieval with LLM generation
- **ThinkTool Protocol System**: Structured reasoning protocols for LLMs
- **Multiple LLM Providers**: 18+ providers including Anthropic, OpenAI, Gemini, Groq, xAI, and more
- **RAPTOR Tree**: Hierarchical retrieval (optional)
- **Storage Backends**: In-memory, file-based, and Qdrant vector database

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 5: RETRIEVAL & QUERY                                  │
│   Hybrid Search | RAPTOR Tree Query | Reranking             │
├─────────────────────────────────────────────────────────────┤
│ LAYER 4: INDEXING                                           │
│   HNSW Index | BM25 Index | RAPTOR Tree                     │
├─────────────────────────────────────────────────────────────┤
│ LAYER 3: EMBEDDING                                          │
│   Dense Embed | Sparse Embed | ColBERT                      │
├─────────────────────────────────────────────────────────────┤
│ LAYER 2: PROCESSING                                         │
│   Chunking | Cleaning | Metadata Extraction                 │
├─────────────────────────────────────────────────────────────┤
│ LAYER 1: INGESTION                                          │
│   PDF | HTML/MD | JSON/JSONL | GitHub                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Cargo

```bash
cargo add reasonkit-core
```

### Features

```toml
[dependencies]
reasonkit-core = { version = "1.0.0", features = ["local-embeddings"] }
```

Available features:

- `cli` (default): Command-line interface
- `local-embeddings`: ONNX-based local embedding models (BGE-M3, E5)
- `arf`: Advanced Reasoning Framework (optional)

---

## Core Architecture

### Module Structure

```
reasonkit_core/
├── lib.rs              # Core types: Document, Chunk, SearchResult
├── rag/                # RAG engine combining retrieval + generation
├── retrieval/          # Hybrid search (BM25 + vector)
├── thinktool/          # ThinkTool protocol execution + LLM clients
├── embedding/          # Embedding providers (OpenAI, local ONNX)
├── storage/            # Storage backends (in-memory, file, Qdrant)
├── indexing/           # BM25 text indexing (Tantivy)
├── processing/         # Chunking and text processing
├── ingestion/          # Document parsers (PDF, MD, HTML, JSON)
└── raptor/             # RAPTOR hierarchical retrieval
```

---

## Core Types

### Document

The central data structure representing a document in the knowledge base.

```rust
pub struct Document {
    pub id: Uuid,
    pub doc_type: DocumentType,
    pub source: Source,
    pub content: DocumentContent,
    pub metadata: Metadata,
    pub processing: ProcessingStatus,
    pub chunks: Vec<Chunk>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}
```

**Document Types:**

- `Paper`: Academic paper (PDF)
- `Documentation`: Technical documentation
- `Code`: Source code
- `Note`: User notes
- `Transcript`: Meeting/interview transcript
- `Benchmark`: Benchmark data

**Example:**

```rust
use reasonkit_core::{Document, DocumentType, Source, SourceType};
use chrono::Utc;

let source = Source {
    source_type: SourceType::Arxiv,
    url: Some("https://arxiv.org/abs/2401.18059".to_string()),
    path: None,
    arxiv_id: Some("2401.18059".to_string()),
    github_repo: None,
    retrieved_at: Utc::now(),
    version: None,
};

let doc = Document::new(DocumentType::Paper, source)
    .with_content("This is the paper content...".to_string());
```

### Chunk

A text chunk extracted from a document.

```rust
pub struct Chunk {
    pub id: Uuid,
    pub text: String,
    pub index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub token_count: Option<usize>,
    pub section: Option<String>,
    pub page: Option<usize>,
    pub embedding_ids: EmbeddingIds,
}
```

### SearchResult / HybridResult

Result from a hybrid search query.

```rust
pub struct HybridResult {
    pub doc_id: Uuid,
    pub chunk_id: Uuid,
    pub text: String,
    pub score: f32,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
    pub match_source: MatchSource,
}
```

**Match Sources:**

- `Dense`: Vector similarity match
- `Sparse`: BM25 keyword match
- `Hybrid`: Combined score
- `Raptor`: RAPTOR tree match

### RetrievalConfig

Configuration for retrieval operations.

```rust
pub struct RetrievalConfig {
    pub top_k: usize,
    pub min_score: f32,
    pub alpha: f32,  // 0.0 = sparse only, 1.0 = dense only
    pub use_raptor: bool,
    pub rerank: bool,
    pub fusion_strategy: FusionStrategy,
}
```

---

## Main Entry Points

### 1. KnowledgeBase

High-level API for managing documents and querying.

```rust
use reasonkit_core::retrieval::KnowledgeBase;

// Create an in-memory knowledge base
let kb = KnowledgeBase::in_memory()?;

// Add a document
kb.add(&doc).await?;

// Query
let results = kb.query("What is chain-of-thought?", 5).await?;

// Get statistics
let stats = kb.stats().await?;
```

### 2. HybridRetriever

Lower-level API for hybrid search (BM25 + vector).

```rust
use reasonkit_core::retrieval::HybridRetriever;

let retriever = HybridRetriever::in_memory()?;

// Add documents
retriever.add_document(&doc).await?;

// Sparse-only search (BM25)
let results = retriever.search_sparse("machine learning", 10).await?;

// Dense-only search (vector)
let results = retriever.search_dense("neural networks", 10).await?;

// Hybrid search with configuration
let config = RetrievalConfig {
    top_k: 10,
    alpha: 0.7,  // 70% dense, 30% sparse
    ..Default::default()
};
let results = retriever.search_hybrid("query", None, &config).await?;
```

### 3. RagEngine

RAG (Retrieval-Augmented Generation) engine combining search with LLM generation.

```rust
use reasonkit_core::rag::{RagEngine, RagConfig};
use reasonkit_core::thinktool::UnifiedLlmClient;

// Create RAG engine
let engine = RagEngine::in_memory()?;

// Optional: Set LLM client for generation
let llm = UnifiedLlmClient::default_anthropic()?;
let engine = engine.with_llm(llm);

// Add documents
engine.add_document(&doc).await?;

// Query with RAG
let response = engine.query("How does self-consistency work?").await?;

println!("Answer: {}", response.answer);
for source in response.sources {
    println!("- [score: {:.3}] {}", source.score, source.text);
}
```

**RAG Configuration Presets:**

```rust
// Quick mode (3 chunks, fast)
let config = RagConfig::quick();

// Thorough mode (10 chunks, hybrid search)
let config = RagConfig::thorough();

let engine = engine.with_config(config);
```

### 4. UnifiedLlmClient (ThinkTool)

Unified client for 18+ LLM providers.

```rust
use reasonkit_core::thinktool::{UnifiedLlmClient, LlmRequest};

// Anthropic Claude (default)
let client = UnifiedLlmClient::default_anthropic()?;

// OpenAI
let client = UnifiedLlmClient::openai("gpt-4")?;

// Groq (ultra-fast inference)
let client = UnifiedLlmClient::groq("llama-3.3-70b-versatile")?;

// xAI Grok
let client = UnifiedLlmClient::grok("grok-2")?;

// Google Gemini
let client = UnifiedLlmClient::gemini("gemini-2.0-flash-exp")?;

// OpenRouter (300+ models)
let client = UnifiedLlmClient::openrouter("anthropic/claude-sonnet-4")?;

// Make a request
let request = LlmRequest::new("Explain quantum entanglement")
    .with_system("You are a physics teacher")
    .with_max_tokens(1000);

let response = client.complete(request).await?;
println!("{}", response.content);
```

**Provider Auto-Discovery:**

```rust
use reasonkit_core::thinktool::{discover_available_providers, create_available_client};

// Find all providers with API keys configured
let available = discover_available_providers();
for provider in available {
    println!("Available: {:?}", provider);
}

// Create client using first available provider
let client = create_available_client()?;
```

---

## Module Reference

### `embedding`

Text embedding functionality.

```rust
use reasonkit_core::embedding::{
    EmbeddingConfig, EmbeddingPipeline, OpenAIEmbedding, EmbeddingProvider
};
use std::sync::Arc;

// OpenAI embeddings
let config = EmbeddingConfig::default();  // text-embedding-3-small
let provider = Arc::new(OpenAIEmbedding::new(config)?);
let pipeline = EmbeddingPipeline::new(provider);

// Embed text
let embedding = pipeline.embed_text("Hello, world!").await?;

// Embed chunks
let results = pipeline.embed_chunks(&doc.chunks).await?;
```

**Local Embeddings (requires `local-embeddings` feature):**

```rust
#[cfg(feature = "local-embeddings")]
{
    let config = EmbeddingConfig::bge_m3();  // BGE-M3 ONNX
    // Or: EmbeddingConfig::e5_small()
}
```

### `storage`

Document and embedding storage.

```rust
use reasonkit_core::storage::{Storage, AccessContext, AccessLevel};
use std::path::PathBuf;

// In-memory storage
let storage = Storage::in_memory();

// File-based storage
let storage = Storage::file(PathBuf::from("./data/storage")).await?;

// Qdrant storage
let storage = Storage::qdrant(
    "localhost",
    6333,     // HTTP port
    6334,     // gRPC port
    "reasonkit".to_string(),  // collection name
    1536,     // vector dimension
    false     // embedded mode
).await?;

// Access control context
let context = AccessContext::new(
    "user_id".to_string(),
    AccessLevel::Admin,
    "store_document".to_string()
);

// Store document
storage.store_document(&doc, &context).await?;

// Retrieve document
let doc = storage.get_document(&doc_id, &context).await?;
```

### `indexing`

BM25 text indexing using Tantivy.

```rust
use reasonkit_core::indexing::{IndexManager, BM25Index};
use std::path::PathBuf;

// In-memory index
let index = IndexManager::in_memory()?;

// Persistent index
let index = IndexManager::open(PathBuf::from("./data/index"))?;

// Index documents
index.index_document(&doc)?;

// Search
let results = index.search_bm25("quantum mechanics", 10)?;

// Statistics
let stats = index.stats()?;
println!("Indexed {} chunks", stats.chunk_count);

// Optimize index (merge segments)
index.optimize()?;
```

### `thinktool`

ThinkTool protocol execution and LLM integration.

```rust
use reasonkit_core::thinktool::{
    ProtocolExecutor, ProtocolInput, ReasoningProfile
};

// Create executor
let executor = ProtocolExecutor::new()?;

// Execute a protocol
let result = executor.execute(
    "gigathink",  // Protocol name
    ProtocolInput::query("What factors drive startup success?")
).await?;

println!("Confidence: {:.2}", result.confidence);
```

**Available ThinkTools (OSS):**

- `gigathink` (gt): Expansive creative thinking (10+ perspectives)
- `laserlogic` (ll): Precision deductive reasoning, fallacy detection
- `bedrock` (br): First principles decomposition
- `proofguard` (pg): Multi-source verification
- `brutalhonesty` (bh): Adversarial self-critique

**Reasoning Profiles:**

- `--quick`: Fast 3-step analysis (70% confidence)
- `--balanced`: Standard 5-module chain (80% confidence)
- `--deep`: Thorough analysis (85% confidence)
- `--paranoid`: Maximum verification (95% confidence)

### `retrieval`

Hybrid retrieval combining BM25 and vector search.

```rust
use reasonkit_core::retrieval::{
    HybridRetriever, fusion::FusionStrategy, RetrievalConfig
};

let retriever = HybridRetriever::in_memory()?;

// Configure fusion strategy
let config = RetrievalConfig {
    top_k: 10,
    alpha: 0.7,  // 70% dense, 30% sparse
    fusion_strategy: FusionStrategy::ReciprocalRankFusion { k: 60 },
    ..Default::default()
};

let results = retriever.search_hybrid("query", None, &config).await?;
```

**Fusion Strategies:**

- `ReciprocalRankFusion { k }`: RRF with configurable k parameter
- `LinearCombination { weights }`: Weighted score combination
- `VoteCount`: Count-based fusion

### `raptor`

RAPTOR hierarchical retrieval tree (optional).

```rust
use reasonkit_core::raptor::RaptorTree;

let mut tree = RaptorTree::new(
    3,   // max_depth
    10   // cluster_size
);

// Build tree from chunks
tree.build_from_chunks(
    &chunks,
    &embedder_fn,
    &summarizer_fn
).await?;

// Query the tree
let results = tree.query("query", 5, &embedder_fn).await?;
```

---

## Configuration

### EmbeddingConfig

```rust
pub struct EmbeddingConfig {
    pub model: String,
    pub dimension: usize,
    pub api_endpoint: Option<String>,
    pub api_key_env: Option<String>,
    pub batch_size: usize,
    pub normalize: bool,
    pub timeout_secs: u64,
    pub enable_cache: bool,
    pub cache_ttl_secs: u64,
}
```

**Defaults:**

- Model: `text-embedding-3-small` (OpenAI)
- Dimension: 1536
- Batch size: 100
- Cache TTL: 24 hours

### RetrievalConfig

```rust
pub struct RetrievalConfig {
    pub top_k: usize,            // Number of results
    pub min_score: f32,          // Minimum relevance score
    pub alpha: f32,              // Dense/sparse balance (0-1)
    pub use_raptor: bool,        // Enable RAPTOR tree
    pub rerank: bool,            // Enable reranking
    pub fusion_strategy: FusionStrategy,
}
```

**Defaults:**

- top_k: 10
- min_score: 0.0
- alpha: 0.7 (favor semantic search)
- fusion_strategy: ReciprocalRankFusion

### RagConfig

```rust
pub struct RagConfig {
    pub top_k: usize,
    pub min_score: f32,
    pub max_context_tokens: usize,
    pub include_sources: bool,
    pub system_prompt: String,
    pub sparse_only: bool,
    pub hybrid_alpha: f32,
}
```

**Defaults:**

- top_k: 5
- max_context_tokens: 2000
- sparse_only: true (BM25-only by default)

---

## Usage Examples

### Example 1: Simple Knowledge Base

```rust
use reasonkit_core::{
    retrieval::KnowledgeBase,
    Document, DocumentType, Source, SourceType,
};
use chrono::Utc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create knowledge base
    let kb = KnowledgeBase::in_memory()?;

    // Create a document
    let source = Source {
        source_type: SourceType::Local,
        url: None,
        path: Some("./notes.md".to_string()),
        arxiv_id: None,
        github_repo: None,
        retrieved_at: Utc::now(),
        version: None,
    };

    let doc = Document::new(DocumentType::Note, source)
        .with_content("Chain-of-thought prompting enables complex reasoning by breaking problems into intermediate steps.".to_string());

    // Add to knowledge base
    kb.add(&doc).await?;

    // Query
    let results = kb.query("How does chain-of-thought work?", 5).await?;

    for result in results {
        println!("Score: {:.3} - {}", result.score, result.text);
    }

    Ok(())
}
```

### Example 2: RAG with LLM Generation

```rust
use reasonkit_core::{
    rag::{RagEngine, RagConfig},
    thinktool::UnifiedLlmClient,
    Document, DocumentType, Source, SourceType,
};
use chrono::Utc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Setup RAG engine with LLM
    let llm = UnifiedLlmClient::default_anthropic()?;
    let config = RagConfig::thorough();

    let engine = RagEngine::in_memory()?
        .with_llm(llm)
        .with_config(config);

    // Add documents
    let doc = Document::new(
        DocumentType::Documentation,
        Source {
            source_type: SourceType::Github,
            url: Some("https://github.com/anthropics/claude-code".to_string()),
            path: None,
            arxiv_id: None,
            github_repo: Some("anthropics/claude-code".to_string()),
            retrieved_at: Utc::now(),
            version: None,
        }
    ).with_content("Claude Code is an AI coding agent...".to_string());

    engine.add_document(&doc).await?;

    // Query with RAG
    let response = engine.query("What is Claude Code?").await?;

    println!("Answer:\n{}\n", response.answer);
    println!("Sources:");
    for (i, source) in response.sources.iter().enumerate() {
        println!("{}. [score: {:.3}] {}", i + 1, source.score, source.text);
    }
    println!("\nStats:");
    println!("- Chunks retrieved: {}", response.retrieval_stats.chunks_retrieved);
    println!("- Chunks used: {}", response.retrieval_stats.chunks_used);
    println!("- Context tokens: {}", response.retrieval_stats.context_tokens);

    Ok(())
}
```

### Example 3: Hybrid Search with Custom Fusion

```rust
use reasonkit_core::{
    retrieval::{HybridRetriever, fusion::FusionStrategy},
    embedding::{EmbeddingPipeline, OpenAIEmbedding, EmbeddingConfig},
    indexing::IndexManager,
    storage::Storage,
    RetrievalConfig,
};
use std::sync::Arc;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Setup components
    let storage = Storage::file(PathBuf::from("./data/storage")).await?;
    let index = IndexManager::open(PathBuf::from("./data/index"))?;

    let embedding_provider = Arc::new(OpenAIEmbedding::openai()?);
    let pipeline = Arc::new(EmbeddingPipeline::new(embedding_provider));

    // Create retriever
    let retriever = HybridRetriever::new(storage, index)
        .with_embedding_pipeline(pipeline);

    // Configure hybrid search
    let config = RetrievalConfig {
        top_k: 20,
        min_score: 0.1,
        alpha: 0.6,  // 60% dense, 40% sparse
        fusion_strategy: FusionStrategy::ReciprocalRankFusion { k: 60 },
        ..Default::default()
    };

    // Search
    let results = retriever.search_hybrid(
        "neural network architectures",
        None,
        &config
    ).await?;

    for result in results {
        println!(
            "Score: {:.3} (dense: {:?}, sparse: {:?}) - {}",
            result.score,
            result.dense_score,
            result.sparse_score,
            result.text
        );
    }

    Ok(())
}
```

### Example 4: Multi-Provider LLM Routing

```rust
use reasonkit_core::thinktool::{
    UnifiedLlmClient, LlmRequest, discover_available_providers
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Discover available providers
    let available = discover_available_providers();
    println!("Available providers: {:?}", available);

    // Create clients for different use cases
    let fast_client = UnifiedLlmClient::groq("llama-3.3-70b-versatile")?;
    let smart_client = UnifiedLlmClient::openai("gpt-4")?;
    let balanced_client = UnifiedLlmClient::default_anthropic()?;

    // Fast query
    let request = LlmRequest::new("Summarize this in one sentence")
        .with_max_tokens(100);
    let response = fast_client.complete(request).await?;
    println!("Fast: {}", response.content);

    // Smart query
    let request = LlmRequest::new("Solve this complex math problem")
        .with_system("You are a math expert")
        .with_max_tokens(1000);
    let response = smart_client.complete(request).await?;
    println!("Smart: {}", response.content);

    Ok(())
}
```

### Example 5: ThinkTool Protocol Execution

```rust
use reasonkit_core::thinktool::{ProtocolExecutor, ProtocolInput};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let executor = ProtocolExecutor::new()?;

    // Execute GigaThink protocol (expansive thinking)
    let result = executor.execute(
        "gigathink",
        ProtocolInput::query("What are the key challenges in AGI development?")
    ).await?;

    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Perspectives:");
    for perspective in result.perspectives() {
        println!("- {}", perspective);
    }

    // Execute ProofGuard protocol (verification)
    let result = executor.execute(
        "proofguard",
        ProtocolInput::query("Verify this claim: All LLMs use transformer architecture")
    ).await?;

    println!("\nVerification confidence: {:.2}%", result.confidence * 100.0);

    Ok(())
}
```

### Example 6: Persistent Storage with Qdrant

```rust
use reasonkit_core::{
    retrieval::HybridRetriever,
    storage::{Storage, AccessContext, AccessLevel},
    indexing::IndexManager,
    Document,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Setup Qdrant storage
    let storage = Storage::qdrant(
        "localhost",
        6333,
        6334,
        "my_collection".to_string(),
        1536,  // OpenAI text-embedding-3-small dimension
        false  // Use remote Qdrant server
    ).await?;

    // Setup BM25 index
    let index = IndexManager::open(PathBuf::from("./data/index"))?;

    // Create retriever
    let retriever = HybridRetriever::new(storage, index);

    // Add documents with access control
    let context = AccessContext::new(
        "admin".to_string(),
        AccessLevel::Admin,
        "index_documents".to_string()
    );

    retriever.add_document(&doc).await?;

    // Query
    let results = retriever.search("query", 10).await?;

    Ok(())
}
```

---

## Error Handling

ReasonKit Core uses a custom error type that wraps all error variants:

```rust
pub enum Error {
    Io(String),
    Parse(String),
    Network(String),
    Validation(String),
    Embedding(String),
    Retrieval(String),
    Indexing(String),
    Query(String),
    Storage(String),
    Configuration(String),
    Processing(String),
    ThinkTool(String),
    External(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Example:**

```rust
use reasonkit_core::{Result, Error};

fn process_document() -> Result<()> {
    // ... operation that might fail
    Err(Error::validation("Invalid document format".to_string()))
}

match process_document() {
    Ok(_) => println!("Success!"),
    Err(e) => eprintln!("Error: {}", e),
}
```

---

## Environment Variables

### Required for API-based Embeddings

```bash
# OpenAI embeddings
export OPENAI_API_KEY="sk-..."

# Anthropic (for Claude LLM)
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="..."

# Groq
export GROQ_API_KEY="gsk_..."

# xAI Grok
export XAI_API_KEY="xai-..."

# OpenRouter (access 300+ models)
export OPENROUTER_API_KEY="sk-or-..."
```

### Optional Configuration

```bash
# Qdrant connection
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY="your-api-key"

# Custom embedding endpoint
export EMBEDDING_API_ENDPOINT="https://your-endpoint.com/v1/embeddings"
```

---

## Performance Considerations

### Batch Processing

```rust
// Efficient: Process documents in batches
let docs = vec![doc1, doc2, doc3];
for doc in docs {
    retriever.add_document(&doc).await?;
}
```

### Caching

```rust
// Embedding cache is enabled by default
let config = EmbeddingConfig {
    enable_cache: true,
    cache_ttl_secs: 86400,  // 24 hours
    ..Default::default()
};
```

### Index Optimization

```rust
// Periodically optimize the BM25 index
index.optimize()?;
```

---

## Testing

Run the test suite:

```bash
# All tests
cargo test

# Specific module
cargo test --package reasonkit-core --lib retrieval

# With output
cargo test -- --nocapture

# Integration tests
cargo test --test '*'
```

---

## CLI Usage

ReasonKit Core includes a command-line interface:

```bash
# Build the CLI
cargo build --release

# Run commands
./target/release/rk-core --help

# Ingest documents
./target/release/rk-core ingest --path ./data/papers --recursive

# Query the knowledge base
./target/release/rk-core query "What is chain-of-thought prompting?" --top-k 10

# Get statistics
./target/release/rk-core stats

# Serve as HTTP API
./target/release/rk-core serve --port 8080
```

---

## Additional Resources

- **GitHub Repository**: https://github.com/reasonkit/reasonkit-core
- **Documentation**: https://docs.rs/reasonkit-core
- **Examples**: `/examples` directory in the repository
- **Architecture Guide**: `ARCHITECTURE.md`
- **Source Overview**: `docs/SOURCE_OVERVIEW.md`

---

## License

Apache 2.0 - See [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Generated:** 2025-12-23
**ReasonKit Core Version:** 1.0.0
