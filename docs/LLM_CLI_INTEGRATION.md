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

## 5. VOICE AND AUDIO PROCESSING

### 5.1 Audio Transcription Plugins

```bash
# OpenAI Whisper API
llm install llm-whisper-api
llm whisper-api recording.mp3 > transcript.txt
curl -s 'https://example.com/audio.mp3' | llm whisper-api -

# Groq's ultra-fast Whisper (near real-time)
llm install llm-groq-whisper
llm keys set groq
llm groq-whisper interview.mp3 --response-format verbose_json
llm groq-whisper foreign-audio.mp3 --translate  # Auto-translate to English

# Gemini native audio (under 1 cent for 7+ minutes)
llm -m gemini-1.5-flash-8b-latest 'Transcribe and summarize key points' \
  -a https://example.com/podcast.mp3
```

### 5.2 Podcast Analysis Pipeline

```bash
#!/bin/bash
# podcast-analyzer.sh - Complete podcast processing
EPISODE_URL="$1"

# Download audio
yt-dlp -x --audio-format mp3 -o episode.mp3 "$EPISODE_URL"

# Transcribe with Groq (fastest)
llm groq-whisper episode.mp3 > transcript.txt

# Generate show notes
cat transcript.txt | llm -s "Generate show notes with timestamps, key topics, and notable quotes"

# Extract action items
cat transcript.txt | llm -s "Extract all action items, recommendations, and resources mentioned"

rm episode.mp3
```

### 5.3 Meeting Notes Automation

```bash
#!/bin/bash
# meeting-notes.sh - Process meeting recordings
RECORDING="$1"

# Transcribe
llm groq-whisper "$RECORDING" > raw_transcript.txt

# Process with ReasonKit protocol
cat raw_transcript.txt | llm -t rk-scientific \
  "Analyze this meeting transcript:
   1. Key decisions made
   2. Action items with owners
   3. Open questions
   4. Follow-up required"
```

---

## 6. IMAGE AND VIDEO ANALYSIS

### 6.1 Vision with Attachments

```bash
# GPT-4o vision analysis
llm -m gpt-4o "Describe this architectural style" -a building.jpg

# OCR from scanned documents
llm -m gpt-4o "Extract all text, preserving formatting" -a document.jpg

# Multiple images comparison
llm -m claude-sonnet-4.5 "Compare these designs" -a design1.png -a design2.png

# URL-based images
llm -m gemini-2.0-flash "Analyze this chart" -a https://example.com/chart.png

# Diagram analysis for documentation
llm -m gpt-4o "Convert this architecture diagram to Mermaid syntax" -a diagram.png
```

### 6.2 YouTube Video Analysis (Gemini)

```bash
# Gemini can process YouTube URLs directly
llm -m gemini-3-pro-preview \
  -a 'https://www.youtube.com/watch?v=VIDEO_ID' \
  'Produce a summary with timestamps and key quotes'

# Technical tutorial analysis
llm -m gemini-2.0-flash \
  -a 'https://www.youtube.com/watch?v=TUTORIAL_ID' \
  'Extract step-by-step instructions with code examples'
```

### 6.3 Video Frame Extraction

```bash
# For models that don't support direct video
llm install llm-video-frames

# Extract frames and analyze
llm -f 'video-frames:presentation.mp4?fps=1&timestamps=1' \
  'Summarize each slide transition' -m gpt-4o

# YouTube transcript fallback (no Gemini)
yt-dlp --write-auto-sub --skip-download --sub-format vtt \
  --output transcript "$VIDEO_URL"
cat transcript.en.vtt | grep -v "^[0-9]" | awk '!seen[$0]++' | \
  llm -m claude-3-haiku "Summarize this video"
```

---

## 7. DEVELOPER PRODUCTIVITY TOOLS

### 7.1 Git Commit Automation

```bash
# Create global git hooks directory
mkdir -p ~/.git_hooks

# Auto-generate commit messages from staged changes
cat > ~/.git_hooks/prepare-commit-msg << 'EOF'
#!/bin/sh
if [ -n "$2" ]; then exit 0; fi
commit_msg=$(git diff --cached | llm -s "Write a conventional commit message (feat/fix/docs/refactor) with scope. First line under 72 chars.")
echo "$commit_msg" > "$1"
EOF

chmod +x ~/.git_hooks/prepare-commit-msg
git config --global core.hooksPath ~/.git_hooks
```

### 7.2 Shell Command Generation (llm-cmd)

```bash
llm install llm-cmd

# Generate shell commands from natural language
llm cmd find all python files modified this week
llm cmd list kubernetes pods in namespace production
llm cmd undo last git commit
llm cmd compress all jpg files in directory to 80% quality
llm cmd show disk usage sorted by size
# Returns executable command, no markdown or explanation
```

### 7.3 Codebase Analysis (files-to-prompt)

```bash
pip install files-to-prompt

# Generate documentation from source
files-to-prompt src -e py -c | \
  llm -m o3-mini -s "Write API documentation with examples"

# Security audit
files-to-prompt . -e js --line-numbers | \
  llm -s "Identify security vulnerabilities with line references"

# Test generation
files-to-prompt myproject -e py -c | \
  llm -s "Generate pytest test cases for untested functions"

# Architecture analysis
files-to-prompt src -e rs -c | \
  llm -m claude-sonnet-4 "Analyze this Rust codebase architecture and suggest improvements"
```

### 7.4 Vim/Neovim Integration

```vim
" Select text in visual mode, then pipe through LLM
:'<,'>!llm -s "Refactor for readability"
:'<,'>!llm -s "Add comprehensive docstrings"
:'<,'>!llm -s "Convert to async/await pattern"

" Quick code explanation
:'<,'>!llm -s "Explain this code in comments"
```

### 7.5 GitHub Actions Integration

```yaml
name: LLM Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install LLM
        run: pip install llm llm-github-models
      - name: Review Changes
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          git diff origin/main -- '*.py' | \
          llm -s "Review these changes for bugs, security issues, and style" \
          > review.md
      - name: Post Review
        uses: actions/github-script@v7
        with:
          script: |
            const review = require('fs').readFileSync('review.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: review
            });
```

---

## 8. SECURITY AND ENTERPRISE DEPLOYMENT

### 8.1 Secure Key Management

```bash
# Store keys securely (interactive prompt, not in shell history)
llm keys set openai
llm keys set anthropic

# List configured keys (names only, not values)
llm keys list

# Key storage location
llm keys path  # ~/.config/io.datasette.llm/keys.json

# Environment variable alternative (for CI/CD)
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='...'
```

### 8.2 Audit Logging

```bash
# All prompts logged automatically to SQLite
llm logs path                # Show database location
llm logs -n 10               # Recent entries
llm logs -q "security"       # Full-text search
llm logs -m gpt-4o           # Filter by model

# Disable for sensitive operations
llm logs off
llm "sensitive prompt here"
llm logs on

# Export logs for compliance
sqlite3 $(llm logs path) ".mode csv" ".headers on" \
  "SELECT datetime_utc, model, prompt, response FROM responses" \
  > audit_log.csv
```

### 8.3 Privacy with Local Models

```bash
# Zero data leaves your machine
llm install llm-ollama
ollama pull llama3.2

# Process confidential documents locally
llm -m llama3.2:latest "Analyze this confidential report" -a report.pdf

# Local embeddings (no API calls)
llm install llm-sentence-transformers
llm embed-multi docs -d local.db --files docs/ '**/*.md' \
  -m sentence-transformers/all-MiniLM-L6-v2 --store
```

### 8.4 Enterprise Compliance Wrapper

```bash
#!/bin/bash
# llm-audited.sh - Enterprise compliance wrapper
AUDIT_LOG="/var/log/llm-audit.log"
USER=$(whoami)
TIMESTAMP=$(date -Iseconds)

# Log request
echo "[$TIMESTAMP] [$USER] Prompt: ${1:0:100}..." >> "$AUDIT_LOG"

# Execute
RESPONSE=$(llm "$@" 2>&1)

# Log response hash (not content for privacy)
HASH=$(echo "$RESPONSE" | sha256sum | cut -d' ' -f1)
echo "[$TIMESTAMP] [$USER] Response hash: $HASH" >> "$AUDIT_LOG"

echo "$RESPONSE"
```

---

## 9. KNOWLEDGE MANAGEMENT WITH OBSIDIAN

### 9.1 Obsidian Local GPT Setup

```bash
# 1. Install Ollama
brew install ollama  # or curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1

# 2. In Obsidian: Install "Local GPT" from Community Plugins
# 3. Configure:
#    - URL: http://127.0.0.1:11434
#    - Model: llama3.1:latest
```

### 9.2 Embedding Your Obsidian Vault

```bash
# Create searchable embeddings from vault
llm embed-multi notes \
  -d obsidian-search.db \
  --files ~/Obsidian/vault/ '**/*.md' \
  -m sentence-transformers/all-MiniLM-L6-v2 --store

# Semantic search across all notes
llm similar notes -d obsidian-search.db -c "project management insights"

# RAG over personal knowledge base
QUERY="How did I solve the authentication problem last month?"
CONTEXT=$(llm similar notes -d obsidian-search.db -c "$QUERY" -n 5 | jq -r '.[].content')
echo "$CONTEXT" | llm -s "Answer based on my notes: $QUERY"
```

### 9.3 Daily Notes Processing

```bash
#!/bin/bash
# obsidian-daily.sh - Process daily notes

VAULT="$HOME/Obsidian/vault"
TODAY=$(date +%Y-%m-%d)

# Summarize today's notes
cat "$VAULT/Daily Notes/$TODAY.md" | \
  llm -s "Extract: tasks completed, insights, open questions"

# Find related notes
llm similar notes -d obsidian-search.db \
  -c "$(cat $VAULT/Daily\ Notes/$TODAY.md)" -n 3

# Generate weekly review
cat "$VAULT/Daily Notes"/$(date -d "7 days ago" +%Y-%m-%d).md \
    "$VAULT/Daily Notes"/$(date +%Y-%m-%d).md | \
  llm -s "Write a weekly review: accomplishments, learnings, next week focus"
```

---

## 10. LOCAL MODEL DEPLOYMENT

### 10.1 Ollama Integration

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

### 10.2 GGUF Direct Loading

```bash
# Load local GGUF model
llm -m gguf:/path/to/model.gguf "Query"

# With custom parameters
llm -m gguf:/models/codellama-34b-Q4_K_M.gguf \
    -o n_gpu_layers 32 \
    -o temperature 0.1 \
    "Generate Rust code for async HTTP client"
```

### 10.3 Cost Optimization Strategy

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

## 11. TOOL PLUGINS FOR DATA WORK

### 11.1 SQL Database Querying

```bash
# Query research database
llm -T 'SQLite("research.db")' "Show papers with sample size > 1000"

# Query remote Datasette
llm -T 'Datasette("https://datasette.example.com/db")' \
    "Find correlations between methodology and result significance"
```

### 11.2 Web Search with Citations

```bash
# Search with Exa (returns full content + citations)
llm -T get_answer "What's the latest on Claude 4 model capabilities?"

# Web search for research
llm -T web_search "MCP protocol implementation best practices 2025"
```

### 11.3 Math Calculations

```bash
# Precise arithmetic (no LLM hallucination)
llm -T simple_eval "Calculate: (47.5 * 892) / 3.14159"

# Statistical calculations
llm -T simple_eval "mean([23, 45, 67, 89, 12, 34])"
```

### 11.4 RAG Retrieval

```bash
# Query embedded documents
llm -T RAG "How does ReasonKit handle multi-agent consensus?"

# With specific collection
llm -T 'RAG("papers")' "What embedding models work best for code search?"
```

---

## 12. BATCH PROCESSING PATTERNS

### 12.1 Parallel Processing

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

### 12.2 Pipeline Chaining

```bash
# Multi-stage analysis pipeline
cat raw_data.json | \
    llm -m gpt-4o-mini "Extract key entities as JSON list" | \
    llm -m claude-sonnet "Identify relationships between entities" | \
    llm -m gpt-4o "Generate knowledge graph in Mermaid format"
```

### 12.3 Structured Output Extraction

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

## 13. PYTHON API FOR AUTOMATION

### 13.1 Basic Usage

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

### 13.2 Structured Output

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

### 13.3 Batch Processing

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

## 14. REASONKIT-SPECIFIC WORKFLOWS

### 14.1 Research Triangulation Protocol

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

### 14.2 Reasoning Trace Capture

```bash
# Capture full reasoning for audit
llm -m claude-opus-4 -o thinking 1 -o thinking_budget 16000 \
    "Design a consensus algorithm for multi-agent fact verification" \
    2>&1 | tee reasoning_trace.txt

# Parse thinking blocks
grep -A 1000 '<thinking>' reasoning_trace.txt | grep -B 1000 '</thinking>'
```

### 14.3 Automated Literature Review

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

## 15. INTEGRATION CHECKLIST

### 15.1 Setup Tasks

- [ ] Install `llm` CLI tool
- [ ] Install essential plugins (llm-anthropic, llm-ollama, llm-tools-*)
- [ ] Configure API keys
- [ ] Create ReasonKit-specific templates
- [ ] Set up logs.db backup schedule

### 15.2 Daily Workflows

- [ ] Morning: Check logs.db for overnight analysis results
- [ ] Research: Use RAG + web search with triangulation
- [ ] Development: Code generation + ProofGuard validation
- [ ] Evening: Review cost analysis, optimize model selection

### 15.3 Weekly Maintenance

- [ ] Archive logs.db to backup
- [ ] Update embeddings with new documents
- [ ] Re-cluster knowledge base
- [ ] Review and optimize templates

---

## 16. DATASETTE DEPLOYMENT

### 16.1 Local Exploration

```bash
# Open LLM logs in Datasette
datasette $(llm logs path)

# Open ReasonKit research database
datasette research.db --open

# Multiple databases
datasette logs.db research.db knowledge.db --open
```

### 16.2 Publish to GitHub Pages

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
