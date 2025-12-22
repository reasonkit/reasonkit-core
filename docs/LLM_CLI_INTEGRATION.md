# LLM CLI & Datasette Ecosystem Integration
> ReasonKit Data Infrastructure: CLI-First AI Workflows
> Simon Willison's LLM ecosystem for reproducible, auditable AI operations

---

## EXECUTIVE SUMMARY

| Capability | Tool | ReasonKit Use Case | Status |
|------------|------|-------------------|--------|
| **CLI LLM Access** | `llm` | Universal model routing | INTEGRATE |
| **Conversation Logging** | logs.db | Audit trails, research replay | CRITICAL |
| **Embeddings** | `llm embed` | Semantic search, clustering | HIGH |
| **RAG Tools** | llm-tools-rag | Document retrieval | HIGH |
| **Data Exploration** | Datasette | SQL-based AI querying | MEDIUM |
| **Local Models** | llm-ollama, llm-gguf | Cost-free inference | HIGH |
| **Tool Plugins** | llm-tools-* | Calculator, SQL, web search | HIGH |

---

## 1. CORE INSTALLATION

### 1.1 Base Setup

```bash
# Install with uv (recommended)
uv tool install llm

# Or with pip
pip install llm

# Verify installation
llm --version
llm models list
```

### 1.2 Essential Plugins

```bash
# Model Providers (install what you need)
llm install llm-anthropic       # Claude models (Claude 4, Sonnet, Haiku)
llm install llm-gemini          # Google Gemini
llm install llm-mistral         # Mistral AI
llm install llm-openrouter      # Multi-provider routing (100+ models)
llm install llm-ollama          # Local Ollama models
llm install llm-gguf            # Direct GGUF files

# Embedding Models
llm install llm-sentence-transformers  # Local embeddings
llm install llm-embed-onnx            # ONNX optimized

# Tool Plugins
llm install llm-tools-simpleeval      # Math calculations
llm install llm-tools-sqlite          # Database queries
llm install llm-tools-datasette       # Remote Datasette
llm install llm-tools-exa             # Web search with sources
llm install llm-tools-rag             # RAG retrieval

# Fragment Loaders
llm install llm-fragments-github      # GitHub repo/issues/PRs
llm install llm-fragments-pdf         # PDF extraction
llm install llm-fragments-pypi        # PyPI package docs

# Utility Plugins
llm install llm-cluster               # Semantic clustering
llm install llm-cmd                   # Shell command generation
llm install llm-jq                    # JSON processing
```

### 1.3 API Key Configuration

```bash
# Set keys for providers
llm keys set openai          # OpenAI/GPT models
llm keys set anthropic       # Claude models
llm keys set gemini          # Google Gemini
llm keys set openrouter      # OpenRouter aggregator
llm keys set exa             # Exa search API

# Verify keys
llm keys
```

---

## 2. REASONKIT INTEGRATION PATTERNS

### 2.1 ReasonKit Protocol Execution via LLM CLI

```bash
# Quick protocol (5 steps, 30s, 0.7 confidence)
llm -m gpt-4o-mini -s "Execute ReasonKit QUICK protocol: 5 reasoning steps, validate each, output confidence score 0.7+ required" \
    "Analyze: Should we use Redis or Qdrant for session caching?"

# Scientific protocol (7 steps, 2min, 0.85 confidence)
llm -m claude-sonnet-4 -s "Execute ReasonKit SCIENTIFIC protocol: 7 reasoning steps with evidence, citations required, 0.85 confidence threshold" \
    "Compare: LangChain vs DSPy for structured reasoning pipelines"

# Absolute protocol (10 steps, 5min, 0.95 confidence)
llm -m claude-opus-4 -s "Execute ReasonKit ABSOLUTE protocol: 10 exhaustive steps, adversarial self-critique, 0.95 confidence minimum" \
    "Design: Multi-agent consensus system for legal document review"
```

### 2.2 Integration with Rust Core

```bash
# Use LLM CLI to generate Rust code for review
llm -m claude-sonnet-4 -f reasonkit-core/src/lib.rs \
    "Generate a new module for hybrid search combining BM25 and vector similarity"

# Validate Rust code with ReasonKit ThinkTools
cat generated_code.rs | llm -m gpt-4o -s "You are ProofGuard. Verify this Rust code for correctness, safety, and performance. Flag any unsafe blocks, unwraps, or panic paths."

# Generate tests
llm -m claude-sonnet-4 -s "Generate comprehensive Rust unit tests" -f reasonkit-core/src/hybrid_search.rs > tests.rs
```

### 2.3 Document Processing Pipeline

```bash
# Extract from PDF papers
llm -m claude-sonnet-4 -a paper.pdf --schema 'title, authors: [string], methodology, key_findings: [string], limitations: [string]' \
    "Extract structured metadata" > paper_metadata.json

# Batch process research papers
for pdf in papers/*.pdf; do
    llm -m gpt-4o-mini -a "$pdf" \
        "Summarize in 3 sentences: research question, methodology, key contribution" \
        >> papers_summary.txt
done

# Build embeddings for semantic search
llm embed-multi papers -d research.db \
    --sql 'SELECT id, title || ": " || abstract as content FROM papers' \
    -m sentence-transformers/all-MiniLM-L12-v2 --store

# Cluster papers by theme
llm cluster papers 8 -d research.db --summary -m gpt-4o
```

---

## 3. SQLITE/DATASETTE DATA WORKFLOWS

### 3.1 Conversation Logging Analysis

```bash
# View conversation history
llm logs -n 20

# Search past conversations
llm logs -q "rust async"

# Export to CSV for analysis
sqlite3 $(llm logs path) ".mode csv" ".headers on" "SELECT * FROM responses" > llm_history.csv

# Open in Datasette for exploration
datasette $(llm logs path)

# Cost analysis by model
sqlite3 $(llm logs path) "
SELECT
    model,
    COUNT(*) as calls,
    SUM(input_tokens) as total_input,
    SUM(output_tokens) as total_output,
    ROUND(SUM(input_tokens * 0.000003 + output_tokens * 0.000015), 4) as estimated_cost_usd
FROM responses
WHERE datetime(datetime_utc) > datetime('now', '-7 days')
GROUP BY model
ORDER BY estimated_cost_usd DESC;
"
```

### 3.2 ReasonKit Data Integration

```bash
# Import ReasonKit session data to LLM ecosystem
sqlite-utils insert research.db sessions session_data.json

# Create embeddings from reasoning traces
llm embed-multi traces -d research.db \
    --sql 'SELECT id, reasoning_chain as content FROM reasoning_traces' --store

# Query with natural language
llm -T 'SQLite("research.db")' "Find sessions where confidence dropped below 0.7"
```

### 3.3 Knowledge Base Querying

```bash
# Load SOURCE_OVERVIEW.md context
llm -f reasonkit-core/docs/SOURCE_OVERVIEW.md \
    "Which sources should be checked daily for breaking changes?"

# RAG over indexed documentation
llm -T RAG "What's the best approach for hybrid search in Qdrant?"

# GitHub context loading
llm -f github:anthropics/claude-code "Explain the hook system architecture"
llm -f issue:anthropics/claude-code/123 "Summarize the issue and proposed solutions"
```

---

## 4. REUSABLE TEMPLATES

### 4.1 Create Templates

```bash
# Save a template for code review
llm -s "You are an expert Rust code reviewer. Focus on: safety, performance, idiomatic patterns, error handling. Rate each category 1-5 and provide specific improvements." \
    --save rust-review

# Save a template for research synthesis
llm -s "You are a research analyst. Synthesize findings from multiple sources, identify consensus and conflicts, rate evidence quality 1-5, cite all sources." \
    --save research-synthesis

# Save a template for ReasonKit protocol
llm -s "Execute ReasonKit SCIENTIFIC protocol (7 steps):
1. UNDERSTAND: Clarify the problem
2. DECOMPOSE: Break into sub-problems
3. RESEARCH: Gather evidence (cite sources)
4. ANALYZE: Apply logical reasoning
5. SYNTHESIZE: Combine insights
6. VALIDATE: Self-critique, find flaws
7. CONCLUDE: Final answer with confidence [0.00-1.00]" \
    --save rk-scientific
```

### 4.2 Use Templates

```bash
# Use rust-review template
cat src/main.rs | llm -t rust-review

# Use research-synthesis with multiple sources
llm -t research-synthesis -f paper1.pdf -f paper2.pdf -f notes.md \
    "What are the key methodological differences?"

# Use ReasonKit protocol template
llm -t rk-scientific "Should ReasonKit integrate with LangGraph or build custom orchestration?"
```

---

## 5. LOCAL MODEL DEPLOYMENT

### 5.1 Ollama Integration

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.2          # Fast general purpose
ollama pull qwen2.5-coder:7b  # Code specialized
ollama pull deepseek-r1:14b   # Advanced reasoning

# Use via LLM CLI
llm -m llama3.2 "Quick explanation of async/await in Rust"
llm -m qwen2.5-coder:7b -f buggy_code.rs "Find and fix the bugs"
llm -m deepseek-r1:14b "Prove that P implies Q using formal logic"
```

### 5.2 GGUF Direct Loading

```bash
# Load local GGUF model
llm -m gguf:/path/to/model.gguf "Query"

# With custom parameters
llm -m gguf:/models/codellama-34b-Q4_K_M.gguf \
    -o n_gpu_layers 32 \
    -o temperature 0.1 \
    "Generate Rust code for async HTTP client"
```

### 5.3 Cost Optimization Strategy

```
Model Selection Tiers:
├── Exploration (cheap)
│   ├── gpt-4o-mini     ($0.15/1M in)
│   ├── claude-haiku    ($0.25/1M in)
│   └── llama3.2        (FREE - local)
├── Analysis (balanced)
│   ├── claude-sonnet   ($3.00/1M in)
│   ├── gpt-4o          ($2.50/1M in)
│   └── qwen2.5-coder   (FREE - local)
└── Critical (premium)
    ├── claude-opus     ($15.00/1M in)
    └── deepseek-r1     (FREE - local)
```

---

## 6. TOOL PLUGINS FOR DATA WORK

### 6.1 SQL Database Querying

```bash
# Query research database
llm -T 'SQLite("research.db")' "Show papers with sample size > 1000"

# Query remote Datasette
llm -T 'Datasette("https://datasette.example.com/db")' \
    "Find correlations between methodology and result significance"
```

### 6.2 Web Search with Citations

```bash
# Search with Exa (returns full content + citations)
llm -T get_answer "What's the latest on Claude 4 model capabilities?"

# Web search for research
llm -T web_search "MCP protocol implementation best practices 2025"
```

### 6.3 Math Calculations

```bash
# Precise arithmetic (no LLM hallucination)
llm -T simple_eval "Calculate: (47.5 * 892) / 3.14159"

# Statistical calculations
llm -T simple_eval "mean([23, 45, 67, 89, 12, 34])"
```

### 6.4 RAG Retrieval

```bash
# Query embedded documents
llm -T RAG "How does ReasonKit handle multi-agent consensus?"

# With specific collection
llm -T 'RAG("papers")' "What embedding models work best for code search?"
```

---

## 7. BATCH PROCESSING PATTERNS

### 7.1 Parallel Processing

```bash
# Process files in parallel
for file in documents/*.md; do
    (llm -m gpt-4o-mini "Summarize key points" < "$file" > "summaries/$(basename $file)") &
done
wait

# Rate-limited parallel
parallel --jobs 4 --delay 0.5 \
    'llm -m claude-haiku "Extract: title, date, author" < {}' ::: papers/*.txt
```

### 7.2 Pipeline Chaining

```bash
# Multi-stage analysis pipeline
cat raw_data.json | \
    llm -m gpt-4o-mini "Extract key entities as JSON list" | \
    llm -m claude-sonnet "Identify relationships between entities" | \
    llm -m gpt-4o "Generate knowledge graph in Mermaid format"
```

### 7.3 Structured Output Extraction

```bash
# Extract with schema
llm -m claude-sonnet \
    --schema 'company: str, revenue_m: float, employees: int, founded: int, sector: str' \
    "Extract company data" < company_description.txt

# Multi-record extraction
llm -m gpt-4o \
    --schema-multi 'paper_title, authors: [str], year: int, citations: int' \
    "Extract all referenced papers" < literature_review.md > citations.json
```

---

## 8. PYTHON API FOR AUTOMATION

### 8.1 Basic Usage

```python
import llm

# Get model
model = llm.get_model("claude-sonnet-4")

# Simple prompt
response = model.prompt("Explain quantum computing in one paragraph")
print(response.text())

# Streaming
for chunk in model.prompt("Write a haiku about Rust"):
    print(chunk, end="", flush=True)

# Conversations
conversation = model.conversation()
conversation.prompt("What is ReasonKit?")
conversation.prompt("How does it compare to LangChain?")
```

### 8.2 Structured Output

```python
import llm
from pydantic import BaseModel
from typing import List

class PaperMetadata(BaseModel):
    title: str
    authors: List[str]
    year: int
    methodology: str
    findings: List[str]

model = llm.get_model("claude-sonnet-4")
response = model.prompt(
    "Extract metadata from this paper abstract...",
    schema=PaperMetadata
)
paper = PaperMetadata.model_validate_json(response.text())
```

### 8.3 Batch Processing

```python
import llm
from pathlib import Path
import json

model = llm.get_model("gpt-4o-mini")

def process_papers(papers_dir: Path):
    results = []
    for pdf in papers_dir.glob("*.pdf"):
        response = model.prompt(
            "Extract: title, methodology, key findings",
            attachments=[llm.Attachment(path=str(pdf))],
            schema=llm.schema_dsl("title, methodology, findings: list")
        )
        results.append({
            "file": pdf.name,
            "data": json.loads(response.text())
        })
    return results

# Usage
papers = process_papers(Path("research_papers"))
```

---

## 9. REASONKIT-SPECIFIC WORKFLOWS

### 9.1 Research Triangulation Protocol

```bash
#!/bin/bash
# triangulate.sh - Verify claims with 3+ sources

QUERY="$1"

echo "=== SOURCE 1: Web Search ==="
llm -T web_search "$QUERY" | tee source1.txt

echo "=== SOURCE 2: RAG (Internal KB) ==="
llm -T RAG "$QUERY" | tee source2.txt

echo "=== SOURCE 3: GitHub ==="
llm -f github:anthropics/claude-code "$QUERY context" | tee source3.txt

echo "=== SYNTHESIS ==="
cat source1.txt source2.txt source3.txt | \
    llm -m claude-sonnet -t research-synthesis \
    "Synthesize these 3 sources, note conflicts, provide final answer with confidence"
```

### 9.2 Reasoning Trace Capture

```bash
# Capture full reasoning for audit
llm -m claude-opus-4 -o thinking 1 -o thinking_budget 16000 \
    "Design a consensus algorithm for multi-agent fact verification" \
    2>&1 | tee reasoning_trace.txt

# Parse thinking blocks
grep -A 1000 '<thinking>' reasoning_trace.txt | grep -B 1000 '</thinking>'
```

### 9.3 Automated Literature Review

```bash
#!/bin/bash
# lit_review.sh - Automated literature synthesis

# 1. Embed all papers
llm embed-multi papers -d lit.db \
    --sql 'SELECT id, abstract as content FROM papers' --store

# 2. Cluster into themes
llm cluster papers 8 -d lit.db --summary > themes.json

# 3. For each theme, synthesize
jq -r '.[] | .summary' themes.json | while read theme; do
    echo "## $theme"
    llm similar papers -c "query: $theme" -d lit.db -n 5 | \
        jq -r '.content' | \
        llm -m gpt-4o "Synthesize these papers on: $theme"
done > literature_review.md
```

---

## 10. INTEGRATION CHECKLIST

### 10.1 Setup Tasks

- [ ] Install `llm` CLI tool
- [ ] Install essential plugins (llm-anthropic, llm-ollama, llm-tools-*)
- [ ] Configure API keys
- [ ] Create ReasonKit-specific templates
- [ ] Set up logs.db backup schedule

### 10.2 Daily Workflows

- [ ] Morning: Check logs.db for overnight analysis results
- [ ] Research: Use RAG + web search with triangulation
- [ ] Development: Code generation + ProofGuard validation
- [ ] Evening: Review cost analysis, optimize model selection

### 10.3 Weekly Maintenance

- [ ] Archive logs.db to backup
- [ ] Update embeddings with new documents
- [ ] Re-cluster knowledge base
- [ ] Review and optimize templates

---

## 11. DATASETTE DEPLOYMENT

### 11.1 Local Exploration

```bash
# Open LLM logs in Datasette
datasette $(llm logs path)

# Open ReasonKit research database
datasette research.db --open

# Multiple databases
datasette logs.db research.db knowledge.db --open
```

### 11.2 Publish to GitHub Pages

```bash
# Install publishing plugin
pip install datasette-publish-github

# Publish research database
datasette publish github research.db \
    --repo username/reasonkit-data \
    --title "ReasonKit Research Database"
```

---

## APPENDIX A: Plugin Reference

| Plugin | Command | Use Case |
|--------|---------|----------|
| llm-anthropic | `llm -m claude-*` | Claude models |
| llm-ollama | `llm -m llama3.2` | Local Ollama |
| llm-openrouter | `llm -m openrouter/*` | Multi-provider |
| llm-tools-sqlite | `-T SQLite("db")` | Database queries |
| llm-tools-rag | `-T RAG` | Document retrieval |
| llm-tools-exa | `-T web_search` | Web search |
| llm-cluster | `llm cluster` | Semantic clustering |
| llm-fragments-github | `-f github:repo` | GitHub context |
| llm-fragments-pdf | `-a paper.pdf` | PDF extraction |

---

## APPENDIX B: Cost Estimation

```python
# Calculate monthly costs (estimate)
# Based on typical research usage

USAGE = {
    "exploration": {"model": "gpt-4o-mini", "tokens": 5_000_000, "rate": 0.15/1_000_000},
    "analysis": {"model": "claude-sonnet", "tokens": 1_000_000, "rate": 3.00/1_000_000},
    "critical": {"model": "claude-opus", "tokens": 100_000, "rate": 15.00/1_000_000},
}

total = sum(u["tokens"] * u["rate"] for u in USAGE.values())
print(f"Estimated monthly cost: ${total:.2f}")  # ~$5.25/month typical
```

---

*Document Version: 1.0*
*Integration with: Simon Willison's LLM CLI ecosystem*
*See: https://llm.datasette.io/en/latest/*
