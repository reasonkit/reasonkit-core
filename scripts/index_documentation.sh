#!/bin/bash
# ReasonKit Core - Documentation Indexer
# Downloads and indexes documentation from GitHub repositories and websites
#
# Usage: ./index_documentation.sh [--all | --cli | --mcp | --api]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/docs"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"

# Create directories
mkdir -p "$RAW_DIR" "$PROCESSED_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "${BLUE}[====]${NC} $1"; }

# Clone or update a GitHub repository
clone_or_update_repo() {
    local repo="$1"
    local name="$2"
    local target="${RAW_DIR}/${name}"

    log_info "Processing: $repo"

    if [[ -d "$target/.git" ]]; then
        log_info "Updating existing clone: $name"
        git -C "$target" pull --quiet || log_warn "Failed to update $name"
    else
        log_info "Cloning: $repo → $name"
        git clone --depth 1 --quiet "https://github.com/${repo}.git" "$target" || {
            log_error "Failed to clone $repo"
            return 1
        }
    fi

    log_info "Success: $name"
}

# Extract markdown and important files from a repo
extract_docs() {
    local name="$1"
    local source="${RAW_DIR}/${name}"
    local output="${PROCESSED_DIR}/${name}.jsonl"

    if [[ ! -d "$source" ]]; then
        log_warn "Source not found: $source"
        return 1
    fi

    log_info "Extracting docs from: $name"

    # Create JSONL with document metadata
    echo "" > "$output"

    # Find and process markdown files
    find "$source" -type f \( -name "*.md" -o -name "*.mdx" \) | while read -r file; do
        local rel_path="${file#$source/}"
        local content=$(cat "$file" 2>/dev/null | jq -Rs '.')

        if [[ -n "$content" && "$content" != '""' ]]; then
            echo "{\"source\":\"github\",\"repo\":\"$name\",\"path\":\"$rel_path\",\"content\":$content}" >> "$output"
        fi
    done

    local count=$(wc -l < "$output" | tr -d ' ')
    log_info "Extracted $count documents from $name"
}

# Index CLI tools
index_cli_tools() {
    log_section "=== Indexing CLI Tools ==="

    # Claude Code
    clone_or_update_repo "anthropics/claude-code" "claude-code"
    extract_docs "claude-code"

    # Gemini CLI
    clone_or_update_repo "google-gemini/gemini-cli" "gemini-cli"
    extract_docs "gemini-cli"

    # Aider
    clone_or_update_repo "paul-gauthier/aider" "aider"
    extract_docs "aider"

    # Continue.dev
    clone_or_update_repo "continuedev/continue" "continue"
    extract_docs "continue"

    log_section "=== CLI Tools Complete ==="
}

# Index MCP Protocol
index_mcp() {
    log_section "=== Indexing MCP Protocol ==="

    # MCP Specification
    clone_or_update_repo "modelcontextprotocol/specification" "mcp-spec"
    extract_docs "mcp-spec"

    # MCP Servers (includes Sequential Thinking)
    clone_or_update_repo "modelcontextprotocol/servers" "mcp-servers"
    extract_docs "mcp-servers"

    # MCP Python SDK
    clone_or_update_repo "modelcontextprotocol/python-sdk" "mcp-python-sdk"
    extract_docs "mcp-python-sdk"

    # MCP TypeScript SDK
    clone_or_update_repo "modelcontextprotocol/typescript-sdk" "mcp-typescript-sdk"
    extract_docs "mcp-typescript-sdk"

    log_section "=== MCP Protocol Complete ==="
}

# Index Rust libraries
index_rust_libs() {
    log_section "=== Indexing Rust Libraries ==="

    # Qdrant Rust Client
    clone_or_update_repo "qdrant/rust-client" "qdrant-rust"
    extract_docs "qdrant-rust"

    # Tantivy
    clone_or_update_repo "quickwit-oss/tantivy" "tantivy"
    extract_docs "tantivy"

    # hnswlib-rs
    clone_or_update_repo "jean-pierreBoth/hnswlib-rs" "hnswlib-rs"
    extract_docs "hnswlib-rs"

    log_section "=== Rust Libraries Complete ==="
}

# Index AI frameworks
index_frameworks() {
    log_section "=== Indexing AI Frameworks ==="

    # LangChain
    clone_or_update_repo "langchain-ai/langchain" "langchain"
    extract_docs "langchain"

    # LlamaIndex
    clone_or_update_repo "run-llama/llama_index" "llamaindex"
    extract_docs "llamaindex"

    # DSPy
    clone_or_update_repo "stanfordnlp/dspy" "dspy"
    extract_docs "dspy"

    log_section "=== AI Frameworks Complete ==="
}

# Generate combined index
generate_index() {
    log_section "=== Generating Combined Index ==="

    local combined="${DATA_DIR}/all_docs.jsonl"
    local metadata="${DATA_DIR}/index_status.json"

    # Combine all JSONL files
    cat "${PROCESSED_DIR}"/*.jsonl > "$combined" 2>/dev/null || true

    local total_docs=$(wc -l < "$combined" | tr -d ' ')

    # Generate metadata
    cat > "$metadata" << EOF
{
  "generated_at": "$(date -Iseconds)",
  "total_documents": $total_docs,
  "sources": [
EOF

    local first=true
    for jsonl in "${PROCESSED_DIR}"/*.jsonl; do
        if [[ -f "$jsonl" ]]; then
            local name=$(basename "$jsonl" .jsonl)
            local count=$(wc -l < "$jsonl" | tr -d ' ')

            if [[ "$first" == "true" ]]; then
                first=false
            else
                echo "," >> "$metadata"
            fi

            echo -n "    {\"name\": \"$name\", \"documents\": $count}" >> "$metadata"
        fi
    done

    cat >> "$metadata" << EOF

  ]
}
EOF

    log_info "Combined index: $total_docs documents"
    log_info "Metadata written to: $metadata"
}

# Show usage
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all        Index everything (default)"
    echo "  --cli        Index CLI tools only"
    echo "  --mcp        Index MCP protocol only"
    echo "  --rust       Index Rust libraries only"
    echo "  --frameworks Index AI frameworks only"
    echo "  --help       Show this help message"
    echo ""
    echo "Documentation will be saved to: $DATA_DIR"
}

# Main
main() {
    log_info "ReasonKit Core - Documentation Indexer"
    log_info "Output directory: $DATA_DIR"
    echo ""

    # Check for git
    if ! command -v git &> /dev/null; then
        log_error "git is required but not installed"
        exit 1
    fi

    case "${1:---all}" in
        --all)
            index_cli_tools
            index_mcp
            index_rust_libs
            index_frameworks
            ;;
        --cli)
            index_cli_tools
            ;;
        --mcp)
            index_mcp
            ;;
        --rust)
            index_rust_libs
            ;;
        --frameworks)
            index_frameworks
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac

    generate_index

    echo ""
    log_info "Documentation indexing complete!"
    log_info "Run 'rk-core ingest ${DATA_DIR}/all_docs.jsonl' to add to knowledge base"
}

main "$@"
