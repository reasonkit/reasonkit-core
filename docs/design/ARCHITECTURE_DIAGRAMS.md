# ReasonKit Architecture Diagrams
> Consolidated Visual Reference for All System Components
> Version: 1.0.0 | Last Updated: 2025-12-11

---

## Table of Contents

1. [System Overview](#system-overview)
2. [CLI Command Flow](#cli-command-flow)
3. [RAG Pipeline Architecture](#rag-pipeline-architecture)
4. [Multi-Agent Orchestration](#multi-agent-orchestration)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Component Interactions](#component-interactions)

---

## System Overview

### High-Level Architecture (ASCII)

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                                  REASONKIT CORE                                        ║
║                        "Turn Prompts into Protocols"                                   ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                        ║
║   ┌──────────────────────────────────────────────────────────────────────────────┐    ║
║   │                           USER INTERFACE LAYER                                │    ║
║   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    ║
║   │  │   CLI    │  │HTTP API  │  │   MCP    │  │  Python  │  │  Hooks   │       │    ║
║   │  │ rk-core  │  │  :8080   │  │ Server   │  │ Bindings │  │ System   │       │    ║
║   │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │    ║
║   └───────┼─────────────┼────────────┼─────────────┼─────────────┼──────────────┘    ║
║           │             │            │             │             │                    ║
║           └─────────────┴────────────┴─────────────┴─────────────┘                    ║
║                                      │                                                ║
║   ┌──────────────────────────────────┼─────────────────────────────────────────────┐  ║
║   │                         ORCHESTRATION LAYER                                     │  ║
║   │                                  │                                              │  ║
║   │  ┌──────────────────────────────▼───────────────────────────────────────────┐  │  ║
║   │  │                        TASK ROUTER                                        │  │  ║
║   │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │  ║
║   │  │  │ Capability  │  │    Cost     │  │    Load     │  │  Priority   │      │  │  ║
║   │  │  │  Matching   │  │ Optimization│  │  Balancing  │  │   Queuing   │      │  │  ║
║   │  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │  │  ║
║   │  └──────────────────────────────────────────────────────────────────────────┘  │  ║
║   │                                  │                                              │  ║
║   │  ┌──────────────────────────────▼───────────────────────────────────────────┐  │  ║
║   │  │                        AGENT SWARM                                        │  │  ║
║   │  │                                                                           │  │  ║
║   │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐              │  │  ║
║   │  │  │  TIER 1   │  │  TIER 2   │  │  TIER 3   │  │  TIER 4   │              │  │  ║
║   │  │  │Governance │  │ Executive │  │Engineering│  │Specialist │              │  │  ║
║   │  │  │           │  │           │  │           │  │           │              │  │  ║
║   │  │  │ • Opus    │  │ • Gemini  │  │ • Codex   │  │ • Math    │              │  │  ║
║   │  │  │           │  │ • Sonnet  │  │ • Grok    │  │ • Security│              │  │  ║
║   │  │  │           │  │           │  │ • Haiku   │  │           │              │  │  ║
║   │  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘              │  │  ║
║   │  └──────────────────────────────────────────────────────────────────────────┘  │  ║
║   └────────────────────────────────────────────────────────────────────────────────┘  ║
║                                      │                                                ║
║   ┌──────────────────────────────────┼─────────────────────────────────────────────┐  ║
║   │                         RAG PIPELINE LAYER                                      │  ║
║   │                                  │                                              │  ║
║   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  ║
║   │  │ LAYER 1  │  │ LAYER 2  │  │ LAYER 3  │  │ LAYER 4  │  │ LAYER 5  │         │  ║
║   │  │          │  │          │  │          │  │          │  │          │         │  ║
║   │  │ Ingest   │─▶│ Process  │─▶│ Embed    │─▶│ Index    │─▶│ Retrieve │         │  ║
║   │  │          │  │          │  │          │  │          │  │          │         │  ║
║   │  │ PDF,MD   │  │ Chunk    │  │ OpenAI   │  │ Qdrant   │  │ Hybrid   │         │  ║
║   │  │ HTML,etc │  │ Clean    │  │ Local    │  │ Tantivy  │  │ RAPTOR   │         │  ║
║   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │  ║
║   └────────────────────────────────────────────────────────────────────────────────┘  ║
║                                      │                                                ║
║   ┌──────────────────────────────────┼─────────────────────────────────────────────┐  ║
║   │                         STORAGE LAYER                                           │  ║
║   │                                                                                 │  ║
║   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │  ║
║   │  │    Qdrant     │  │   Tantivy     │  │    JSONL      │  │   SQLite      │    │  ║
║   │  │ (Vectors)     │  │   (BM25)      │  │   (Docs)      │  │   (Meta)      │    │  ║
║   │  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘    │  ║
║   └────────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                        ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

### High-Level Architecture (Mermaid)

```mermaid
graph TB
    subgraph UI["User Interface Layer"]
        CLI[CLI: rk-core]
        API[HTTP API :8080]
        MCP[MCP Server]
        PY[Python Bindings]
    end

    subgraph ORCH["Orchestration Layer"]
        Router[Task Router]
        subgraph Agents["Agent Swarm"]
            T1[Tier 1: Governance]
            T2[Tier 2: Executive]
            T3[Tier 3: Engineering]
            T4[Tier 4: Specialist]
        end
    end

    subgraph RAG["RAG Pipeline"]
        L1[Layer 1: Ingest]
        L2[Layer 2: Process]
        L3[Layer 3: Embed]
        L4[Layer 4: Index]
        L5[Layer 5: Retrieve]
    end

    subgraph Storage["Storage Layer"]
        Qdrant[(Qdrant)]
        Tantivy[(Tantivy)]
        JSONL[(JSONL)]
        SQLite[(SQLite)]
    end

    CLI --> Router
    API --> Router
    MCP --> Router
    PY --> Router

    Router --> T1
    Router --> T2
    Router --> T3
    Router --> T4

    T3 --> L1
    L1 --> L2 --> L3 --> L4 --> L5

    L4 --> Qdrant
    L4 --> Tantivy
    L1 --> JSONL
    L5 --> SQLite
```

---

## CLI Command Flow

### Command Dispatch (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLI COMMAND DISPATCH                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   USER INPUT                                                                 │
│       │                                                                      │
│       ▼                                                                      │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                         ARGUMENT PARSER                             │    │
│   │                         (clap derive)                               │    │
│   └───────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│       ┌───────────────────────────┼───────────────────────────────┐         │
│       │                           │                               │         │
│       ▼                           ▼                               ▼         │
│   ┌─────────┐               ┌─────────┐                     ┌─────────┐    │
│   │ Global  │               │ Command │                     │ Config  │    │
│   │  Args   │               │  Match  │                     │  Load   │    │
│   │         │               │         │                     │         │    │
│   │--config │               │ ingest  │                     │ TOML    │    │
│   │--verbose│               │ query   │                     │ ENV     │    │
│   │--format │               │ search  │                     │         │    │
│   └────┬────┘               │ embed   │                     └────┬────┘    │
│        │                    │ index   │                          │         │
│        │                    │ export  │                          │         │
│        │                    │ serve   │                          │         │
│        │                    │ doctor  │                          │         │
│        │                    └────┬────┘                          │         │
│        │                         │                               │         │
│        └─────────────────────────┼───────────────────────────────┘         │
│                                  │                                          │
│                                  ▼                                          │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      COMMAND EXECUTOR                               │    │
│   │                                                                     │    │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │    │
│   │  │ Validate │─▶│ Execute  │─▶│ Format   │─▶│  Output  │           │    │
│   │  │   Args   │  │ Command  │  │ Result   │  │ (stdout) │           │    │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                        EXIT CODE                                    │    │
│   │                                                                     │    │
│   │    0=Success  1=Error  2=Config  3=IO  4=Network  5=Auth           │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Command Hierarchy (Mermaid)

```mermaid
graph TD
    RK[rk-core] --> G[Global Options]
    RK --> C[Commands]

    G --> G1[--config]
    G --> G2[--verbose]
    G --> G3[--format]
    G --> G4[--data-dir]

    C --> INGEST[ingest]
    C --> QUERY[query]
    C --> SEARCH[search]
    C --> EMBED[embed]
    C --> INDEX[index]
    C --> EXPORT[export]
    C --> SERVE[serve]
    C --> DOCTOR[doctor]

    INGEST --> I1[--path]
    INGEST --> I2[--recursive]
    INGEST --> I3[--chunking]
    INGEST --> I4[--embed]

    QUERY --> Q1[query text]
    QUERY --> Q2[--profile]
    QUERY --> Q3[--top-k]
    QUERY --> Q4[--raptor]

    SEARCH --> S1[--method]
    SEARCH --> S2[--alpha]
    SEARCH --> S3[--fusion]
    SEARCH --> S4[--rerank]

    INDEX --> IX[subcommands]
    IX --> IX1[create]
    IX --> IX2[list]
    IX --> IX3[delete]
    IX --> IX4[rebuild]
```

---

## RAG Pipeline Architecture

### 5-Layer Pipeline (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              RAG PIPELINE ARCHITECTURE                               │
│                              5-Layer RAPTOR Design                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║ LAYER 1: INGESTION                                                             ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                ║  │
│  ║   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                        ║  │
│  ║   │ PDF │  │ MD  │  │HTML │  │JSON │  │ TXT │  │DOCX │                        ║  │
│  ║   └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                        ║  │
│  ║      │        │        │        │        │        │                           ║  │
│  ║      └────────┴────────┴────────┼────────┴────────┴────────┐                  ║  │
│  ║                                 ▼                          │                  ║  │
│  ║                    ┌────────────────────────┐              │                  ║  │
│  ║                    │    FORMAT DETECTOR     │              │                  ║  │
│  ║                    │    & PARSER ROUTER     │              │                  ║  │
│  ║                    └───────────┬────────────┘              │                  ║  │
│  ║                                │                           │                  ║  │
│  ║                                ▼                           ▼                  ║  │
│  ║                    ┌────────────────────────────────────────┐                 ║  │
│  ║                    │         Document { ... }               │                 ║  │
│  ║                    └────────────────────────────────────────┘                 ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                                       │                                              │
│                                       ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║ LAYER 2: PROCESSING                                                            ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                ║  │
│  ║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     ║  │
│  ║  │   CLEAN     │───▶│   CHUNK     │───▶│  ANNOTATE   │───▶│   DEDUPE    │     ║  │
│  ║  │             │    │             │    │             │    │             │     ║  │
│  ║  │ • Strip     │    │ • Semantic  │    │ • Metadata  │    │ • Hash      │     ║  │
│  ║  │ • Normalize │    │ • Sentence  │    │ • Sections  │    │ • Compare   │     ║  │
│  ║  │ • Unicode   │    │ • Fixed     │    │ • Links     │    │             │     ║  │
│  ║  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘     ║  │
│  ║                                                                                ║  │
│  ║                           ┌───────────────────┐                                ║  │
│  ║                           │  Chunk[] Output   │                                ║  │
│  ║                           └───────────────────┘                                ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                                       │                                              │
│                                       ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║ LAYER 3: EMBEDDING                                                             ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                ║  │
│  ║   ┌─────────────────────────────────────────────────────────────────────┐     ║  │
│  ║   │                     EMBEDDING ROUTER                                 │     ║  │
│  ║   └────────────────────────────┬────────────────────────────────────────┘     ║  │
│  ║                                │                                              ║  │
│  ║        ┌───────────────────────┼───────────────────────┐                      ║  │
│  ║        │                       │                       │                      ║  │
│  ║        ▼                       ▼                       ▼                      ║  │
│  ║   ┌─────────┐            ┌─────────┐            ┌─────────┐                   ║  │
│  ║   │ OpenAI  │            │ Voyage  │            │  Local  │                   ║  │
│  ║   │ ada-002 │            │ voyage-2│            │  ONNX   │                   ║  │
│  ║   │ 1536d   │            │ 1024d   │            │  384d   │                   ║  │
│  ║   └─────────┘            └─────────┘            └─────────┘                   ║  │
│  ║                                                                                ║  │
│  ║                      ┌─────────────────────┐                                   ║  │
│  ║                      │  Vec<f32> Vectors   │                                   ║  │
│  ║                      └─────────────────────┘                                   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                                       │                                              │
│                                       ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║ LAYER 4: INDEXING                                                              ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                ║  │
│  ║   ┌───────────────────────┐         ┌───────────────────────┐                 ║  │
│  ║   │     VECTOR INDEX      │         │      BM25 INDEX       │                 ║  │
│  ║   │       (Qdrant)        │         │      (Tantivy)        │                 ║  │
│  ║   │                       │         │                       │                 ║  │
│  ║   │  ┌─────────────────┐  │         │  ┌─────────────────┐  │                 ║  │
│  ║   │  │      HNSW       │  │         │  │   Inverted      │  │                 ║  │
│  ║   │  │   m=16, ef=100  │  │         │  │   Index         │  │                 ║  │
│  ║   │  └─────────────────┘  │         │  │   k1=1.2 b=0.75 │  │                 ║  │
│  ║   │                       │         │  └─────────────────┘  │                 ║  │
│  ║   └───────────────────────┘         └───────────────────────┘                 ║  │
│  ║                                                                                ║  │
│  ║              ┌────────────────────────────────────────────┐                    ║  │
│  ║              │            RAPTOR TREE                     │                    ║  │
│  ║              │                                            │                    ║  │
│  ║              │    Level 4: [Document Summary]             │                    ║  │
│  ║              │              /         \                   │                    ║  │
│  ║              │    Level 3: [Section]  [Section]           │                    ║  │
│  ║              │             /    \        /    \           │                    ║  │
│  ║              │    Level 2: [P1] [P2]   [P3]  [P4]         │                    ║  │
│  ║              │                                            │                    ║  │
│  ║              │    Level 1: [C1][C2][C3][C4][C5][C6]       │                    ║  │
│  ║              │              (leaf chunks)                 │                    ║  │
│  ║              └────────────────────────────────────────────┘                    ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                                       │                                              │
│                                       ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════════════╗  │
│  ║ LAYER 5: RETRIEVAL                                                             ║  │
│  ╠═══════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                ║  │
│  ║   QUERY ──────┐                                                                ║  │
│  ║               │                                                                ║  │
│  ║               ▼                                                                ║  │
│  ║   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐         ║  │
│  ║   │  QUERY EXPAND   │────▶│  PARALLEL SEARCH │────▶│     FUSION      │         ║  │
│  ║   │  (optional)     │     │                 │     │                 │         ║  │
│  ║   │  • Synonyms     │     │  ┌────────────┐ │     │  • RRF (k=60)   │         ║  │
│  ║   │  • HyDE         │     │  │ BM25       │ │     │  • Linear       │         ║  │
│  ║   └─────────────────┘     │  └────────────┘ │     │  • Weighted     │         ║  │
│  ║                           │  ┌────────────┐ │     └────────┬────────┘         ║  │
│  ║                           │  │ Vector     │ │              │                  ║  │
│  ║                           │  └────────────┘ │              ▼                  ║  │
│  ║                           │  ┌────────────┐ │     ┌─────────────────┐         ║  │
│  ║                           │  │ RAPTOR     │ │     │    RERANKER     │         ║  │
│  ║                           │  └────────────┘ │     │  (optional)     │         ║  │
│  ║                           └─────────────────┘     │  Cross-encoder  │         ║  │
│  ║                                                   └────────┬────────┘         ║  │
│  ║                                                            │                  ║  │
│  ║                                                            ▼                  ║  │
│  ║                                               ┌─────────────────────┐         ║  │
│  ║                                               │  SearchResult[]     │         ║  │
│  ║                                               └─────────────────────┘         ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline (Mermaid)

```mermaid
flowchart TB
    subgraph L1["Layer 1: Ingestion"]
        PDF[PDF Parser]
        MD[Markdown Parser]
        HTML[HTML Parser]
        JSON[JSON Parser]
        FD[Format Detector]

        PDF --> FD
        MD --> FD
        HTML --> FD
        JSON --> FD
        FD --> DOC[Document]
    end

    subgraph L2["Layer 2: Processing"]
        CLEAN[Clean]
        CHUNK[Chunk]
        ANNOTATE[Annotate]
        DEDUPE[Dedupe]

        DOC --> CLEAN --> CHUNK --> ANNOTATE --> DEDUPE --> CHUNKS[Chunk[]]
    end

    subgraph L3["Layer 3: Embedding"]
        ROUTER[Embedding Router]
        OPENAI[OpenAI]
        VOYAGE[Voyage]
        LOCAL[Local ONNX]

        CHUNKS --> ROUTER
        ROUTER --> OPENAI
        ROUTER --> VOYAGE
        ROUTER --> LOCAL
        OPENAI --> VECS[Vectors]
        VOYAGE --> VECS
        LOCAL --> VECS
    end

    subgraph L4["Layer 4: Indexing"]
        QDRANT[(Qdrant HNSW)]
        TANTIVY[(Tantivy BM25)]
        RAPTOR[RAPTOR Tree]

        VECS --> QDRANT
        CHUNKS --> TANTIVY
        VECS --> RAPTOR
    end

    subgraph L5["Layer 5: Retrieval"]
        QUERY[Query]
        EXPAND[Query Expand]
        SEARCH[Parallel Search]
        FUSION[Fusion RRF/Linear]
        RERANK[Reranker]
        RESULTS[Results]

        QUERY --> EXPAND --> SEARCH --> FUSION --> RERANK --> RESULTS
        QDRANT --> SEARCH
        TANTIVY --> SEARCH
        RAPTOR --> SEARCH
    end
```

### Hybrid Search Detail (Mermaid)

```mermaid
sequenceDiagram
    participant Q as Query
    participant BM as BM25 Index
    participant VS as Vector Store
    participant RR as RAPTOR Tree
    participant F as Fusion (RRF)
    participant RE as Reranker
    participant R as Results

    Q->>BM: Lexical Search
    Q->>VS: Semantic Search
    Q->>RR: Hierarchical Search

    BM-->>F: BM25 Results (score, rank)
    VS-->>F: Vector Results (score, rank)
    RR-->>F: RAPTOR Results (score, rank)

    Note over F: RRF: score = Σ 1/(k + rank)
    Note over F: k = 60 (default)

    F->>RE: Merged Candidates
    RE->>R: Reranked Top-K
```

---

## Multi-Agent Orchestration

### Agent Hierarchy (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MULTI-AGENT HIERARCHY                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                           ┌─────────────────────┐                                │
│                           │      TIER 1         │                                │
│                           │    GOVERNANCE       │                                │
│                           │                     │                                │
│                           │  ┌───────────────┐  │                                │
│                           │  │ Claude Opus   │  │                                │
│                           │  │   4.5         │  │                                │
│                           │  │ (200K ctx)    │  │                                │
│                           │  └───────────────┘  │                                │
│                           │                     │                                │
│                           │  • Final arbiter    │                                │
│                           │  • Architecture     │                                │
│                           │  • Quality gates    │                                │
│                           └──────────┬──────────┘                                │
│                                      │                                           │
│                    ┌─────────────────┼─────────────────┐                         │
│                    │                 │                 │                         │
│                    ▼                 ▼                 ▼                         │
│     ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐            │
│     │      TIER 2      │  │      TIER 2      │  │      TIER 2      │            │
│     │    EXECUTIVE     │  │    EXECUTIVE     │  │    EXECUTIVE     │            │
│     │                  │  │                  │  │                  │            │
│     │ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │            │
│     │ │ Gemini 2.5   │ │  │ │Claude Sonnet │ │  │ │ DeepSeek V3  │ │            │
│     │ │   Pro        │ │  │ │    4.5       │ │  │ │              │ │            │
│     │ │ (2M ctx)     │ │  │ │ (200K ctx)   │ │  │ │ (164K ctx)   │ │            │
│     │ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │            │
│     │                  │  │                  │  │                  │            │
│     │ • Lead engineer  │  │ • Tech review    │  │ • Architecture   │            │
│     │ • Large refactor │  │ • Quality assure │  │ • System design  │            │
│     └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘            │
│              │                     │                     │                       │
│              └─────────────────────┼─────────────────────┘                       │
│                                    │                                             │
│         ┌──────────────────────────┼──────────────────────────┐                  │
│         │                          │                          │                  │
│         ▼                          ▼                          ▼                  │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐           │
│  │    TIER 3    │          │    TIER 3    │          │    TIER 3    │           │
│  │ ENGINEERING  │          │ ENGINEERING  │          │ ENGINEERING  │           │
│  │              │          │              │          │              │           │
│  │┌────────────┐│          │┌────────────┐│          │┌────────────┐│           │
│  ││ GPT Codex  ││          ││ Grok Code  ││          ││Claude Haiku││           │
│  ││    5.1     ││          ││   Fast     ││          ││    3.6     ││           │
│  ││ (100K ctx) ││          ││ (128K ctx) ││          ││ (200K ctx) ││           │
│  │└────────────┘│          │└────────────┘│          │└────────────┘│           │
│  │              │          │              │          │              │           │
│  │• Complex Rust│          │• Rapid proto │          │• Documentation│          │
│  │• Unsafe code │          │• Tests       │          │• Comments     │          │
│  └──────────────┘          └──────────────┘          └──────────────┘           │
│                                    │                                             │
│                    ┌───────────────┼───────────────┐                             │
│                    │               │               │                             │
│                    ▼               ▼               ▼                             │
│            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│            │   TIER 4    │  │   TIER 4    │  │   TIER 4    │                     │
│            │ SPECIALIST  │  │ SPECIALIST  │  │ SPECIALIST  │                     │
│            │             │  │             │  │             │                     │
│            │┌───────────┐│  │┌───────────┐│  │┌───────────┐│                     │
│            ││DeepSeek   ││  ││Grok-4     ││  ││Mistral    ││                     │
│            ││ Math V3   ││  ││  High     ││  ││ Large 3   ││                     │
│            │└───────────┘│  │└───────────┘│  │└───────────┘│                     │
│            │             │  │             │  │             │                     │
│            │• Math verify│  │• Security   │  │• Performance│                     │
│            │• Proofs     │  │• Red team   │  │• Analysis   │                     │
│            └─────────────┘  └─────────────┘  └─────────────┘                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Task Routing Flow (Mermaid)

```mermaid
flowchart TD
    TASK[New Task] --> EXTRACT[Extract Requirements]
    EXTRACT --> CAP[Capability Match]
    CAP --> COST[Cost Filter]
    COST --> LOAD[Load Balance]
    LOAD --> SCORE[Score Candidates]

    SCORE --> BEST{Best Agent?}
    BEST -->|Found| ASSIGN[Assign Task]
    BEST -->|None| QUEUE[Queue or Escalate]

    ASSIGN --> EXEC[Execute]
    EXEC --> RESULT{Success?}

    RESULT -->|Yes| COMPLETE[Complete]
    RESULT -->|No| RETRY{Can Retry?}

    RETRY -->|Yes| EXEC
    RETRY -->|No| ESCALATE[Escalate to Higher Tier]

    ESCALATE --> CAP

    subgraph Scoring["Score Calculation"]
        S1[Capability Score × 0.35]
        S2[Cost Efficiency × 0.25]
        S3[Load Factor × 0.15]
        S4[Success Rate × 0.15]
        S5[Context Familiarity × 0.10]
    end
```

### Escalation Chain (Mermaid)

```mermaid
stateDiagram-v2
    [*] --> Tier4: Task Created

    Tier4 --> Tier3: Capability Exceeded
    Tier4 --> Tier3: Timeout

    Tier3 --> Tier2: Complex Decision
    Tier3 --> Tier2: Repeated Failure

    Tier2 --> Tier1: Architectural Impact
    Tier2 --> Tier1: Security Critical

    Tier1 --> Human: Budget Exceeded
    Tier1 --> Human: Ethical Concern
    Tier1 --> Human: Unresolvable

    Tier4 --> [*]: Completed
    Tier3 --> [*]: Completed
    Tier2 --> [*]: Completed
    Tier1 --> [*]: Completed
    Human --> [*]: Resolved
```

---

## Data Flow Diagrams

### Document Ingestion Flow

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as CLI
    participant ING as Ingestion
    participant PROC as Processing
    participant EMB as Embedding
    participant IDX as Indexing

    U->>CLI: rk-core ingest --path ./docs
    CLI->>ING: Discover files
    ING->>ING: Detect formats

    loop Each Document
        ING->>ING: Parse content
        ING->>PROC: Document
        PROC->>PROC: Clean text
        PROC->>PROC: Chunk
        PROC->>PROC: Annotate metadata
        PROC->>EMB: Chunk[]

        alt Embeddings enabled
            EMB->>EMB: Generate vectors
            EMB->>IDX: Chunk[] + Vec<f32>[]
        else
            PROC->>IDX: Chunk[]
        end

        IDX->>IDX: Update Qdrant
        IDX->>IDX: Update Tantivy
        IDX->>IDX: Update RAPTOR tree
    end

    IDX-->>CLI: Stats
    CLI-->>U: "Ingested N docs, M chunks"
```

### Query Execution Flow

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as CLI
    participant QE as Query Engine
    participant BM as BM25 Index
    participant VS as Vector Store
    participant RT as RAPTOR Tree
    participant RR as Reranker

    U->>CLI: rk-core query "question" --profile balanced
    CLI->>QE: Parse query

    QE->>QE: Expand query (optional)

    par Parallel Search
        QE->>BM: BM25 search
        QE->>VS: Vector search
        QE->>RT: RAPTOR traverse
    end

    BM-->>QE: BM25 results
    VS-->>QE: Vector results
    RT-->>QE: RAPTOR results

    QE->>QE: RRF Fusion

    alt Reranking enabled
        QE->>RR: Candidates
        RR-->>QE: Reranked
    end

    QE-->>CLI: SearchResult[]
    CLI-->>U: Formatted output
```

---

## Component Interactions

### Module Dependency Graph

```mermaid
graph TB
    subgraph Core["Core Module"]
        TYPES[types/]
        ERROR[error.rs]
        CONFIG[config/]
    end

    subgraph Pipeline["Pipeline Modules"]
        ING[ingestion/]
        PROC[processing/]
        EMB[embedding/]
        IDX[indexing/]
        RET[retrieval/]
    end

    subgraph Storage["Storage Modules"]
        QDRANT[storage/qdrant]
        TANTIVY[storage/tantivy]
        JSONL[storage/jsonl]
    end

    subgraph Orchestration["Orchestration Module"]
        ROUTER[router.rs]
        AGENT[agents/]
        MSG[message_bus.rs]
        STATE[state.rs]
    end

    subgraph External["External Services"]
        OPENAI[OpenAI API]
        ANTHROPIC[Anthropic API]
        GOOGLE[Google AI API]
    end

    TYPES --> ING
    TYPES --> PROC
    TYPES --> EMB
    TYPES --> IDX
    TYPES --> RET
    TYPES --> ROUTER

    ERROR --> ING
    ERROR --> EMB
    ERROR --> ROUTER

    CONFIG --> ING
    CONFIG --> EMB
    CONFIG --> ROUTER

    ING --> PROC --> EMB --> IDX --> RET

    EMB --> OPENAI
    AGENT --> ANTHROPIC
    AGENT --> GOOGLE
    AGENT --> OPENAI

    IDX --> QDRANT
    IDX --> TANTIVY
    ING --> JSONL

    ROUTER --> AGENT
    ROUTER --> MSG
    ROUTER --> STATE
```

### API Endpoint Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HTTP API ENDPOINTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GET    /health                 → Health check                              │
│   GET    /version                → Version info                              │
│                                                                              │
│   POST   /v1/ingest              → Ingest documents                          │
│   GET    /v1/documents           → List documents                            │
│   GET    /v1/documents/:id       → Get document                              │
│   DELETE /v1/documents/:id       → Delete document                           │
│                                                                              │
│   POST   /v1/query               → RAG query                                 │
│   POST   /v1/search              → Direct search                             │
│                                                                              │
│   POST   /v1/embed               → Generate embeddings                       │
│                                                                              │
│   GET    /v1/indices             → List indices                              │
│   POST   /v1/indices             → Create index                              │
│   GET    /v1/indices/:name/stats → Index statistics                          │
│   DELETE /v1/indices/:name       → Delete index                              │
│                                                                              │
│   GET    /v1/stats               → System statistics                         │
│   GET    /v1/config              → Current configuration                     │
│                                                                              │
│   WebSocket /v1/ws               → Streaming responses                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Related Documents

- [CLI Architecture](./CLI_ARCHITECTURE.md)
- [RAG Pipeline Architecture](./RAG_PIPELINE_ARCHITECTURE.md)
- [Multi-Agent Orchestration](./MULTI_AGENT_ORCHESTRATION.md)
- [AGENT_DREAM_TEAM.md](../../../AGENT_DREAM_TEAM.md)
- [ORCHESTRATOR.md](../../../ORCHESTRATOR.md)

---

*ReasonKit Architecture Diagrams v1.0.0*
*Designed, Not Dreamed | Turn Prompts into Protocols*
