#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# FULL KNOWLEDGE BASE RE-INDEX SCRIPT
# ReasonKit Project - Comprehensive Documentation Refresh
#═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/$(whoami)/RK-PROJECT}"
RK_CORE="$PROJECT_ROOT/reasonkit-core"
DATA_DIR="$RK_CORE/data"
DOCS_DIR="$DATA_DIR/docs"
LOGS_DIR="$DATA_DIR/reindex_logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

mkdir -p "$LOGS_DIR" "$DOCS_DIR/raw" "$DOCS_DIR/processed"

LOG_FILE="$LOGS_DIR/reindex_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo "  FULL KNOWLEDGE BASE RE-INDEX"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"

#───────────────────────────────────────────────────────────────────────────────
# GITHUB REPOSITORY UPDATES
#───────────────────────────────────────────────────────────────────────────────

update_or_clone_repo() {
    local repo_url="$1"
    local target_dir="$2"
    local repo_name="$3"

    echo -e "${BLUE}[$repo_name]${NC} Updating..."

    if [[ -d "$target_dir/.git" ]]; then
        cd "$target_dir"
        git fetch origin --prune
        git reset --hard origin/$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
        echo -e "${GREEN}✓ Updated${NC}"
    else
        rm -rf "$target_dir"
        git clone --depth=1 "$repo_url" "$target_dir"
        echo -e "${GREEN}✓ Cloned${NC}"
    fi
    cd "$DOCS_DIR"
}

echo -e "\n${CYAN}=== UPDATING GITHUB REPOSITORIES ===${NC}\n"

# HIGH PRIORITY - AI Provider SDKs & CLIs
HIGH_PRIORITY_REPOS=(
    "https://github.com/anthropics/claude-code|$DOCS_DIR/raw/claude-code|Claude Code CLI"
    "https://github.com/anthropics/anthropic-sdk-python|$DOCS_DIR/raw/anthropic-sdk-python|Anthropic Python SDK"
    "https://github.com/google-gemini/gemini-cli|$DOCS_DIR/raw/gemini-cli|Gemini CLI"
    "https://github.com/openai/openai-python|$DOCS_DIR/raw/openai-python|OpenAI Python SDK"
    "https://github.com/modelcontextprotocol/specification|$DOCS_DIR/raw/mcp-spec|MCP Specification"
    "https://github.com/modelcontextprotocol/servers|$DOCS_DIR/raw/mcp-servers|MCP Servers"
    "https://github.com/modelcontextprotocol/python-sdk|$DOCS_DIR/raw/mcp-python-sdk|MCP Python SDK"
    "https://github.com/continuedev/continue|$DOCS_DIR/raw/continue|Continue.dev"
    "https://github.com/Aider-AI/aider|$DOCS_DIR/raw/aider|Aider"
)

for entry in "${HIGH_PRIORITY_REPOS[@]}"; do
    IFS='|' read -r url dir name <<< "$entry"
    update_or_clone_repo "$url" "$dir" "$name" || echo -e "${YELLOW}⚠ Failed: $name${NC}"
done

# MEDIUM PRIORITY - Frameworks & Libraries
MEDIUM_PRIORITY_REPOS=(
    "https://github.com/langchain-ai/langchain|$DOCS_DIR/raw/langchain|LangChain"
    "https://github.com/run-llama/llama_index|$DOCS_DIR/raw/llama-index|LlamaIndex"
    "https://github.com/stanfordnlp/dspy|$DOCS_DIR/raw/dspy|DSPy"
    "https://github.com/qdrant/qdrant|$DOCS_DIR/raw/qdrant|Qdrant"
    "https://github.com/quickwit-oss/tantivy|$DOCS_DIR/raw/tantivy|Tantivy"
)

for entry in "${MEDIUM_PRIORITY_REPOS[@]}"; do
    IFS='|' read -r url dir name <<< "$entry"
    update_or_clone_repo "$url" "$dir" "$name" || echo -e "${YELLOW}⚠ Failed: $name${NC}"
done

#───────────────────────────────────────────────────────────────────────────────
# EXTRACT DOCUMENTATION TO JSONL
#───────────────────────────────────────────────────────────────────────────────

extract_to_jsonl() {
    local source_dir="$1"
    local output_file="$2"
    local source_name="$3"

    echo -e "${BLUE}[$source_name]${NC} Extracting documentation..."

    if [[ ! -d "$source_dir" ]]; then
        echo -e "${RED}✗ Source directory not found${NC}"
        return 1
    fi

    local count=0
    > "$output_file"

    while IFS= read -r -d '' file; do
        local content
        content=$(cat "$file" | jq -Rs '.')
        local relative_path="${file#$source_dir/}"
        local filename=$(basename "$file")
        local title="${filename%.*}"

        echo "{\"id\": \"${source_name}_$(echo "$relative_path" | md5sum | cut -d' ' -f1)\", \"source\": \"$source_name\", \"path\": \"$relative_path\", \"title\": \"$title\", \"content\": $content, \"type\": \"documentation\", \"indexed_at\": \"$(date -Iseconds)\"}" >> "$output_file"
        ((count++))
    done < <(find "$source_dir" -type f \( -name "*.md" -o -name "*.mdx" -o -name "*.rst" \) -print0 2>/dev/null)

    echo -e "${GREEN}✓ Extracted $count documents${NC}"
}

echo -e "\n${CYAN}=== EXTRACTING DOCUMENTATION ===${NC}\n"

# Extract all repos to JSONL
for dir in "$DOCS_DIR/raw"/*/; do
    [[ -d "$dir" ]] || continue
    name=$(basename "$dir")
    extract_to_jsonl "$dir" "$DOCS_DIR/processed/${name}.jsonl" "$name"
done

#───────────────────────────────────────────────────────────────────────────────
# COMBINE ALL DOCUMENTATION
#───────────────────────────────────────────────────────────────────────────────

echo -e "\n${CYAN}=== COMBINING DOCUMENTATION INDEX ===${NC}\n"

cat "$DOCS_DIR/processed"/*.jsonl > "$DOCS_DIR/all_docs.jsonl" 2>/dev/null || true
total_docs=$(wc -l < "$DOCS_DIR/all_docs.jsonl" 2>/dev/null || echo "0")

echo -e "${GREEN}Total documents indexed: $total_docs${NC}"

#───────────────────────────────────────────────────────────────────────────────
# GENERATE STATUS REPORT
#───────────────────────────────────────────────────────────────────────────────

cat > "$DATA_DIR/reindex_status.json" << EOF
{
  "last_reindex": "$(date -Iseconds)",
  "total_documents": $total_docs,
  "sources": $(ls -1 "$DOCS_DIR/processed"/*.jsonl 2>/dev/null | wc -l),
  "log_file": "$LOG_FILE",
  "status": "completed"
}
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  RE-INDEX COMPLETE"
echo "  Finished: $(date)"
echo "  Total Documents: $total_docs"
echo "  Log: $LOG_FILE"
echo "════════════════════════════════════════════════════════════════"
