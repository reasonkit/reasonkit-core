# BGE-M3 & RAPTOR Implementation Plan

> Internalized from ProofGuard-verified deep research

**Status**: Implementation Ready
**Confidence**: 92% (BGE-M3), 85% (RAPTOR)
**Date**: 2025-12-11

---

## 1. Research Findings Summary (VERIFIED)

### BGE-M3 Technical Specs

| Property            | Value                  | Verification                         |
| ------------------- | ---------------------- | ------------------------------------ |
| Embedding Dimension | 1024                   | HuggingFace + NVIDIA + bge-model.com |
| Max Context Length  | 8192 tokens            | 3 sources confirmed                  |
| Languages           | 100+                   | Official docs                        |
| Parameters          | 568M                   | Model architecture                   |
| License             | MIT                    | Compatible with Apache 2.0           |
| Base Model          | XLM-RoBERTa (extended) | arXiv 2402.03216                     |

### BGE-M3 "3 M's" Multi-Functionality

```
SINGLE MODEL PASS → 3 OUTPUTS:
┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐
│ DENSE (1024d)│ │ SPARSE       │ │ ColBERT (Multi-Vec)  │
│ Semantic     │ │ Lexical      │ │ Token-level          │
│ Similarity   │ │ Keywords     │ │ Late Interaction     │
└──────────────┘ └──────────────┘ └──────────────────────┘
```

### RAPTOR Performance (VERIFIED)

| Dataset      | RAPTOR Score | Baseline | Improvement   |
| ------------ | ------------ | -------- | ------------- |
| QuALITY      | 82.6%        | 62.3%    | **+20.3 pts** |
| QuALITY-HARD | 76.2%        | 54.7%    | **+21.5 pts** |
| QASPER       | F1 55.7%     | 53.9%    | +1.8 pts      |
| NarrativeQA  | METEOR 19.1  | 10.6     | +8.5 pts      |

---

## 2. Decision Matrix (INTERNALIZED)

| Component                   | reasonkit-core | reasonkit-pro | Confidence |
| --------------------------- | -------------- | ------------- | ---------- |
| BGE-M3 Dense+Sparse         | ✅ YES         | ✅ YES        | 92%        |
| BGE-M3 ColBERT              | ❌ No          | ✅ YES        | 88%        |
| RAPTOR Full                 | ❌ No          | ✅ YES        | 85%        |
| Hierarchical Index (simple) | ✅ YES         | ✅ YES        | 90%        |
| ONNX Local Inference        | ✅ YES         | ✅ YES        | 95%        |

---

## 3. Implementation Plan for reasonkit-core

### Phase 1: ONNX Runtime Integration (PRIORITY 1)

```toml
# Add to Cargo.toml
ort = "2.0"           # ONNX Runtime for Rust
tokenizers = "0.19"   # HuggingFace tokenizers
```

**Files to modify:**

- `Cargo.toml` - Add dependencies
- `src/embedding/mod.rs` - Replace LocalEmbedding placeholder
- `src/embedding/bge_m3.rs` - New BGE-M3 provider

**Implementation steps:**

1. Add ort dependency with CPU feature
2. Create BGE-M3 ONNX model loader
3. Implement encode() returning dense + sparse
4. Integrate with existing EmbeddingResult type
5. Benchmark against OpenAI embeddings

### Phase 2: Simple Hierarchical Indexing (PRIORITY 2)

**Not full RAPTOR** - Simple multi-granularity indexing:

```
Document-level (summary/abstract)     ← Top
    ↓
Section-level (headings + first para) ← Middle
    ↓
Chunk-level (100-500 tokens)          ← Bottom (leaves)
```

**No LLM summarization required** - Use existing metadata:

- Academic papers: Use abstract as document-level
- Docs: Use first H1 + intro paragraph
- Code: Use file docstring + function signatures

### Phase 3: RRF Fusion Enhancement (PRIORITY 3)

Current: `hybrid_search()` exists
Enhancement: Configurable weights for BGE-M3 outputs

```rust
// Hybrid ranking formula
s_rank = w_dense * s_dense + w_sparse * s_sparse
// Default: w_dense = 0.7, w_sparse = 0.3
```

---

## 4. ONNX Model Setup

### Download Commands

```bash
# Create models directory
mkdir -p models/bge-m3-onnx

# Option 1: HuggingFace CLI
huggingface-cli download aapot/bge-m3-onnx --local-dir models/bge-m3-onnx

# Option 2: Direct download
wget https://huggingface.co/aapot/bge-m3-onnx/resolve/main/model.onnx
wget https://huggingface.co/aapot/bge-m3-onnx/resolve/main/tokenizer.json
```

### Model Files Expected

```
models/bge-m3-onnx/
├── model.onnx           # ~2.2GB (fp32) or ~1.1GB (fp16)
├── tokenizer.json       # HuggingFace tokenizer config
├── config.json          # Model config
└── special_tokens_map.json
```

---

## 5. Rust Implementation Sketch

### BGE-M3 Provider

```rust
// src/embedding/bge_m3.rs

use ort::{Session, SessionBuilder, Environment};
use tokenizers::Tokenizer;

pub struct BgeM3Provider {
    session: Session,
    tokenizer: Tokenizer,
    dimension: usize,  // 1024
}

impl BgeM3Provider {
    pub fn new(model_path: &Path) -> Result<Self> {
        let env = Environment::builder()
            .with_name("bge_m3")
            .build()?;

        let session = SessionBuilder::new(&env)?
            .with_model_from_file(model_path.join("model.onnx"))?;

        let tokenizer = Tokenizer::from_file(
            model_path.join("tokenizer.json")
        )?;

        Ok(Self {
            session,
            tokenizer,
            dimension: 1024,
        })
    }

    pub fn encode(&self, texts: &[&str]) -> Result<BgeM3Output> {
        // Tokenize
        let encodings = self.tokenizer.encode_batch(texts, true)?;

        // Run ONNX inference
        let outputs = self.session.run(inputs)?;

        // Extract dense and sparse vectors
        Ok(BgeM3Output {
            dense: outputs["dense_vecs"].try_extract()?,
            sparse: outputs["sparse_vecs"].try_extract()?,
        })
    }
}

pub struct BgeM3Output {
    pub dense: Vec<Vec<f32>>,   // [batch_size, 1024]
    pub sparse: Vec<SparseVec>, // Lexical weights
}
```

---

## 6. Fit with Existing Code

### Current Embedding Architecture (from `src/embedding/mod.rs`)

```rust
// Already exists:
pub struct EmbeddingResult {
    pub dense: Vec<f32>,
    pub sparse: Option<SparseVector>,  // ← Perfect for BGE-M3!
}

// LocalEmbedding is placeholder:
pub struct LocalEmbedding {
    dimension: usize,
    // TODO: Implement ONNX-based local embedding
}
```

**Integration point**: Replace `LocalEmbedding` placeholder with `BgeM3Provider`

### Current Retrieval Architecture (from `src/retrieval/mod.rs`)

```rust
// Already exists:
pub struct HybridRetriever {
    // Combines dense + sparse search
}
```

**Integration point**: BGE-M3 outputs directly feed HybridRetriever

---

## 7. Timeline Estimate

| Phase                 | Effort   | Dependencies           |
| --------------------- | -------- | ---------------------- |
| ONNX Integration      | 1-2 days | ort crate              |
| BGE-M3 Provider       | 1-2 days | ONNX + tokenizers      |
| Hierarchical Indexing | 2-3 days | BGE-M3 working         |
| RRF Enhancement       | 1 day    | Existing hybrid search |

**Total**: 5-8 days of focused work

---

## 8. Deferred to reasonkit-pro

| Feature             | Reason                      | Complexity |
| ------------------- | --------------------------- | ---------- |
| Full RAPTOR         | LLM costs, build complexity | High       |
| ColBERT reranking   | Multi-vector storage needs  | Medium     |
| GMM clustering      | Requires linfa port         | Medium     |
| UMAP dimensionality | Requires umap-rs            | Medium     |

---

## 9. Sources (Triangulated)

### Primary (Tier 1)

- https://huggingface.co/BAAI/bge-m3
- https://arxiv.org/abs/2401.18059
- https://github.com/parthsarthi03/raptor

### Secondary (Tier 2)

- https://build.nvidia.com/baai/bge-m3/modelcard
- https://github.com/yuniko-software/bge-m3-qdrant-sample
- https://ragflow.io/blog/long-context-rag-raptor

### Independent (Tier 3)

- https://bge-model.com/bge/bge_m3.html
- https://web.stanford.edu/class/cs224n/final-reports/256925521.pdf
- https://crates.io/crates/embed_anything

---

_Implementation plan based on ProofGuard-verified research with 3+ sources per claim._
