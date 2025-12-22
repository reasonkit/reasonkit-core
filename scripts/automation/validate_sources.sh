#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# SOURCE VALIDATION & AUTO-UPDATE SYSTEM
# ReasonKit Project - Deep Research Triangulation Protocol Enforcement
#═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")}"
CONFIG_DIR="$PROJECT_ROOT/reasonkit-core/config"
DATA_DIR="$PROJECT_ROOT/reasonkit-core/data"
LOGS_DIR="$DATA_DIR/validation_logs"
SOURCES_FILE="$CONFIG_DIR/monitored_sources.json"
STATUS_FILE="$DATA_DIR/source_status.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
TIER="${TIER:-all}"
VERBOSE="${VERBOSE:-false}"
UPDATE_CACHE="${UPDATE_CACHE:-true}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"

#───────────────────────────────────────────────────────────────────────────────
# USAGE
#───────────────────────────────────────────────────────────────────────────────
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --tier=TIER         Validation tier: high|medium|low|all (default: all)
    --verbose           Enable verbose output
    --no-cache          Skip cache updates
    --parallel=N        Max parallel validations (default: 10)
    --sources=FILE      Custom sources JSON file
    --help              Show this help message

Tiers:
    high    - Daily refresh: Anthropic, OpenAI, Google, xAI, OpenCode.ai, MCP
    medium  - Weekly refresh: CLI tools, frameworks, Rust libraries
    low     - Monthly refresh: Academic papers, stable documentation
    all     - Validate all sources

Examples:
    $(basename "$0") --tier=high           # Validate high-priority sources only
    $(basename "$0") --tier=all --verbose  # Full validation with detailed output
    $(basename "$0") --parallel=20         # Increase parallelism for speed
EOF
    exit 0
}

#───────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
#───────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --tier=*) TIER="${1#*=}" ;;
        --verbose) VERBOSE="true" ;;
        --no-cache) UPDATE_CACHE="false" ;;
        --parallel=*) MAX_PARALLEL="${1#*=}" ;;
        --sources=*) SOURCES_FILE="${1#*=}" ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

#───────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
#───────────────────────────────────────────────────────────────────────────────
mkdir -p "$LOGS_DIR" "$CONFIG_DIR" "$(dirname "$STATUS_FILE")"

# Initialize sources file if not exists
if [[ ! -f "$SOURCES_FILE" ]]; then
    echo -e "${YELLOW}Creating default monitored sources configuration...${NC}"
    cat > "$SOURCES_FILE" << 'SOURCES'
{
  "version": "1.0",
  "last_updated": "",
  "sources": {
    "high_priority": {
      "anthropic": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://docs.anthropic.com/", "type": "documentation", "name": "Anthropic Docs"},
          {"url": "https://github.com/anthropics/anthropic-sdk-python", "type": "github", "name": "Anthropic Python SDK"},
          {"url": "https://github.com/anthropics/claude-code", "type": "github", "name": "Claude Code CLI"},
          {"url": "https://www.anthropic.com/news", "type": "blog", "name": "Anthropic News"},
          {"url": "https://github.com/anthropics/courses", "type": "github", "name": "Anthropic Courses"}
        ]
      },
      "openai": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://platform.openai.com/docs", "type": "documentation", "name": "OpenAI Docs"},
          {"url": "https://github.com/openai/openai-python", "type": "github", "name": "OpenAI Python SDK"},
          {"url": "https://github.com/openai/codex", "type": "github", "name": "Codex CLI"},
          {"url": "https://openai.com/blog", "type": "blog", "name": "OpenAI Blog"},
          {"url": "https://github.com/openai/swarm", "type": "github", "name": "OpenAI Swarm"}
        ]
      },
      "google": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://ai.google.dev/docs", "type": "documentation", "name": "Google AI Docs"},
          {"url": "https://github.com/google-gemini/generative-ai-python", "type": "github", "name": "Gemini Python SDK"},
          {"url": "https://github.com/google-gemini/gemini-cli", "type": "github", "name": "Gemini CLI"},
          {"url": "https://aistudio.google.com/", "type": "webapp", "name": "AI Studio"}
        ]
      },
      "xai": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://docs.x.ai/", "type": "documentation", "name": "xAI Docs"},
          {"url": "https://github.com/xai-org/grok", "type": "github", "name": "Grok"},
          {"url": "https://x.ai/blog", "type": "blog", "name": "xAI Blog"}
        ]
      },
      "opencode_ai": {
        "refresh_interval": "daily",
        "priority": "SPECIAL_FOCUS",
        "urls": [
          {"url": "https://opencode.ai/", "type": "webapp", "name": "OpenCode.ai"},
          {"url": "https://github.com/opencode-ai/opencode", "type": "github", "name": "OpenCode Repo"},
          {"url": "https://docs.opencode.ai/", "type": "documentation", "name": "OpenCode Docs"}
        ]
      },
      "mcp_protocol": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://modelcontextprotocol.io/", "type": "documentation", "name": "MCP Official Site"},
          {"url": "https://github.com/modelcontextprotocol/specification", "type": "github", "name": "MCP Spec"},
          {"url": "https://github.com/modelcontextprotocol/servers", "type": "github", "name": "MCP Servers"},
          {"url": "https://github.com/modelcontextprotocol/python-sdk", "type": "github", "name": "MCP Python SDK"},
          {"url": "https://github.com/modelcontextprotocol/typescript-sdk", "type": "github", "name": "MCP TypeScript SDK"}
        ]
      },
      "openrouter": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://openrouter.ai/docs", "type": "documentation", "name": "OpenRouter Docs"},
          {"url": "https://openrouter.ai/models", "type": "api", "name": "OpenRouter Models"}
        ]
      },
      "deepseek": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://www.deepseek.com/", "type": "webapp", "name": "DeepSeek"},
          {"url": "https://github.com/deepseek-ai/DeepSeek-V3", "type": "github", "name": "DeepSeek V3"},
          {"url": "https://api-docs.deepseek.com/", "type": "documentation", "name": "DeepSeek API Docs"}
        ]
      },
      "groq": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://console.groq.com/docs", "type": "documentation", "name": "Groq Docs"},
          {"url": "https://github.com/groq/groq-python", "type": "github", "name": "Groq Python SDK"}
        ]
      },
      "cohere": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://docs.cohere.com/", "type": "documentation", "name": "Cohere Docs"},
          {"url": "https://github.com/cohere-ai/cohere-python", "type": "github", "name": "Cohere Python SDK"}
        ]
      },
      "qwen": {
        "refresh_interval": "daily",
        "urls": [
          {"url": "https://github.com/QwenLM/Qwen", "type": "github", "name": "Qwen"},
          {"url": "https://qwenlm.github.io/", "type": "documentation", "name": "Qwen Docs"}
        ]
      }
    },
    "medium_priority": {
      "cli_tools": {
        "refresh_interval": "weekly",
        "urls": [
          {"url": "https://github.com/continuedev/continue", "type": "github", "name": "Continue.dev"},
          {"url": "https://github.com/Aider-AI/aider", "type": "github", "name": "Aider"},
          {"url": "https://cursor.com/docs", "type": "documentation", "name": "Cursor Docs"},
          {"url": "https://github.com/getcursor/cursor", "type": "github", "name": "Cursor Repo"},
          {"url": "https://codeium.com/windsurf", "type": "webapp", "name": "Windsurf"},
          {"url": "https://www.kiro.dev/", "type": "webapp", "name": "Kiro Code"}
        ]
      },
      "ide_integration": {
        "refresh_interval": "weekly",
        "urls": [
          {"url": "https://code.visualstudio.com/docs", "type": "documentation", "name": "VS Code Docs"},
          {"url": "https://github.com/microsoft/vscode-copilot-release", "type": "github", "name": "Copilot Extension"},
          {"url": "https://docs.github.com/en/copilot", "type": "documentation", "name": "GitHub Copilot Docs"}
        ]
      },
      "frameworks": {
        "refresh_interval": "weekly",
        "urls": [
          {"url": "https://python.langchain.com/docs", "type": "documentation", "name": "LangChain Docs"},
          {"url": "https://docs.llamaindex.ai/", "type": "documentation", "name": "LlamaIndex Docs"},
          {"url": "https://dspy-docs.vercel.app/", "type": "documentation", "name": "DSPy Docs"},
          {"url": "https://github.com/stanfordnlp/dspy", "type": "github", "name": "DSPy Repo"}
        ]
      },
      "rust_libraries": {
        "refresh_interval": "weekly",
        "urls": [
          {"url": "https://github.com/qdrant/qdrant", "type": "github", "name": "Qdrant"},
          {"url": "https://github.com/quickwit-oss/tantivy", "type": "github", "name": "Tantivy"},
          {"url": "https://docs.rs/qdrant-client", "type": "documentation", "name": "qdrant-client docs"},
          {"url": "https://docs.rs/tantivy", "type": "documentation", "name": "tantivy docs"}
        ]
      }
    },
    "low_priority": {
      "academic": {
        "refresh_interval": "monthly",
        "urls": [
          {"url": "https://arxiv.org/list/cs.AI/recent", "type": "academic", "name": "arXiv cs.AI"},
          {"url": "https://arxiv.org/list/cs.CL/recent", "type": "academic", "name": "arXiv cs.CL"},
          {"url": "https://arxiv.org/list/cs.LG/recent", "type": "academic", "name": "arXiv cs.LG"},
          {"url": "https://neurips.cc/", "type": "academic", "name": "NeurIPS"},
          {"url": "https://iclr.cc/", "type": "academic", "name": "ICLR"}
        ]
      }
    }
  }
}
SOURCES
    echo -e "${GREEN}Created: $SOURCES_FILE${NC}"
fi

#───────────────────────────────────────────────────────────────────────────────
# VALIDATION FUNCTIONS
#───────────────────────────────────────────────────────────────────────────────

validate_url() {
    local url="$1"
    local name="$2"
    local type="$3"
    local timeout=10

    # Use curl to check URL
    local http_code
    local start_time end_time duration
    start_time=$(date +%s%3N)

    if http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" -L "$url" 2>/dev/null); then
        end_time=$(date +%s%3N)
        duration=$((end_time - start_time))

        case $http_code in
            200|301|302)
                echo "{\"url\": \"$url\", \"name\": \"$name\", \"status\": \"OK\", \"http_code\": $http_code, \"latency_ms\": $duration, \"timestamp\": \"$(date -Iseconds)\"}"
                [[ "$VERBOSE" == "true" ]] && echo -e "${GREEN}✓ $name ($http_code, ${duration}ms)${NC}" >&2
                return 0
                ;;
            403|401)
                echo "{\"url\": \"$url\", \"name\": \"$name\", \"status\": \"AUTH_REQUIRED\", \"http_code\": $http_code, \"latency_ms\": $duration, \"timestamp\": \"$(date -Iseconds)\"}"
                [[ "$VERBOSE" == "true" ]] && echo -e "${YELLOW}⚠ $name (Auth required: $http_code)${NC}" >&2
                return 1
                ;;
            404)
                echo "{\"url\": \"$url\", \"name\": \"$name\", \"status\": \"NOT_FOUND\", \"http_code\": $http_code, \"latency_ms\": $duration, \"timestamp\": \"$(date -Iseconds)\"}"
                [[ "$VERBOSE" == "true" ]] && echo -e "${RED}✗ $name (404 Not Found)${NC}" >&2
                return 1
                ;;
            *)
                echo "{\"url\": \"$url\", \"name\": \"$name\", \"status\": \"ERROR\", \"http_code\": $http_code, \"latency_ms\": $duration, \"timestamp\": \"$(date -Iseconds)\"}"
                [[ "$VERBOSE" == "true" ]] && echo -e "${RED}✗ $name (HTTP $http_code)${NC}" >&2
                return 1
                ;;
        esac
    else
        echo "{\"url\": \"$url\", \"name\": \"$name\", \"status\": \"TIMEOUT\", \"http_code\": 0, \"latency_ms\": $((timeout * 1000)), \"timestamp\": \"$(date -Iseconds)\"}"
        [[ "$VERBOSE" == "true" ]] && echo -e "${RED}✗ $name (Timeout)${NC}" >&2
        return 1
    fi
}

#───────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATION LOGIC
#───────────────────────────────────────────────────────────────────────────────
run_validation() {
    local log_file="$LOGS_DIR/validation_$(date +%Y%m%d_%H%M%S).json"
    local results=()
    local total=0
    local success=0
    local failed=0

    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  SOURCE VALIDATION - $(date)${NC}"
    echo -e "${CYAN}  Tier: $TIER | Parallel: $MAX_PARALLEL${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"

    # Read sources and validate based on tier
    local sources_to_validate=""
    case "$TIER" in
        high)
            sources_to_validate=$(jq -r '.sources.high_priority | to_entries[] | .value.urls[] | @base64' "$SOURCES_FILE")
            ;;
        medium)
            sources_to_validate=$(jq -r '.sources.medium_priority | to_entries[] | .value.urls[] | @base64' "$SOURCES_FILE")
            ;;
        low)
            sources_to_validate=$(jq -r '.sources.low_priority | to_entries[] | .value.urls[] | @base64' "$SOURCES_FILE")
            ;;
        all)
            sources_to_validate=$(jq -r '
                (.sources.high_priority | to_entries[] | .value.urls[]),
                (.sources.medium_priority | to_entries[] | .value.urls[]),
                (.sources.low_priority | to_entries[] | .value.urls[])
                | @base64' "$SOURCES_FILE")
            ;;
    esac

    echo ""
    echo -e "${BLUE}Validating sources...${NC}"

    # Process in parallel using xargs
    echo "$sources_to_validate" | while read -r encoded; do
        [[ -z "$encoded" ]] && continue
        local decoded
        decoded=$(echo "$encoded" | base64 -d 2>/dev/null) || continue

        local url name type
        url=$(echo "$decoded" | jq -r '.url')
        name=$(echo "$decoded" | jq -r '.name')
        type=$(echo "$decoded" | jq -r '.type')

        result=$(validate_url "$url" "$name" "$type")
        results+=("$result")
        ((total++))

        if echo "$result" | jq -e '.status == "OK"' >/dev/null 2>&1; then
            ((success++))
        else
            ((failed++))
        fi
    done

    # Generate summary
    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo -e "${GREEN}✓ Successful: $success${NC}"
    echo -e "${RED}✗ Failed: $failed${NC}"
    echo -e "${BLUE}Total: $total${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

    # Write log
    {
        echo "{"
        echo "  \"validation_run\": {"
        echo "    \"timestamp\": \"$(date -Iseconds)\","
        echo "    \"tier\": \"$TIER\","
        echo "    \"total\": $total,"
        echo "    \"success\": $success,"
        echo "    \"failed\": $failed,"
        echo "    \"results\": ["
        for i in "${!results[@]}"; do
            echo "      ${results[$i]}$([ $i -lt $((${#results[@]} - 1)) ] && echo ",")"
        done
        echo "    ]"
        echo "  }"
        echo "}"
    } > "$log_file"

    echo -e "${GREEN}Log written to: $log_file${NC}"

    # Update status file
    if [[ "$UPDATE_CACHE" == "true" ]]; then
        cp "$log_file" "$STATUS_FILE"
        echo -e "${GREEN}Status file updated: $STATUS_FILE${NC}"
    fi

    return $failed
}

#───────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
#───────────────────────────────────────────────────────────────────────────────
run_validation
exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo -e "\n${GREEN}All sources validated successfully!${NC}"
else
    echo -e "\n${YELLOW}Warning: $exit_code sources failed validation${NC}"
fi

exit $exit_code
