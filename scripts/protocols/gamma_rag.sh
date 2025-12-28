#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#                    DATA-AWARE RAG: GAMMA PROTOCOL V2
#                      Shadow Logic Detection Layer
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE: Grep-based Literal Scanner that detects "Shadow Logic" -
#          data access patterns that bypass clean Code Graphs.
#
# ADDRESSES: "Ghost Dependency" flaw where vector search misses exact strings
#            like SQL table names, JSON keys, and raw API endpoints.
#
# LICENSE: Apache-2.0
#
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-./shadow_logic_report}"
VERBOSE="${VERBOSE:-false}"

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────────
# PATTERN DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# SQL patterns (case insensitive)
SQL_PATTERNS=(
    'SELECT\s+.*\s+FROM\s+[a-zA-Z_]+'
    'INSERT\s+INTO\s+[a-zA-Z_]+'
    'UPDATE\s+[a-zA-Z_]+\s+SET'
    'DELETE\s+FROM\s+[a-zA-Z_]+'
    'JOIN\s+[a-zA-Z_]+\s+ON'
    'CREATE\s+(TABLE|INDEX|VIEW)'
    'ALTER\s+TABLE'
    'DROP\s+(TABLE|INDEX|VIEW)'
)

# ORM patterns
ORM_PATTERNS=(
    '\.query\('
    '\.filter\('
    '\.find_by\('
    '\.where\('
    'Model\.create'
    'Model\.update'
    'Table\.'
    '\.objects\.'
    '\.findOne\('
    '\.findMany\('
)

# JSON key access patterns
JSON_PATTERNS=(
    "'\w+_id'"
    '"\w+_id"'
    '\["[a-z_]+"\]'
    "\.get\(['\"][a-z_]+['\"]\)"
    'data\[['\''"][a-z_]+['\''"]\]'
)

# Config/environment patterns
CONFIG_PATTERNS=(
    'env\['
    'getenv\('
    'os\.environ'
    'process\.env\.'
    'config\.'
    'settings\.'
    'Config\.'
    'Settings\.'
)

# API endpoint patterns
API_PATTERNS=(
    'https?://[^\s"'\'']+'
    '/api/v[0-9]+/'
    '@(Get|Post|Put|Delete|Patch|Route)\('
    'fetch\(['\''"]'
    'axios\.(get|post|put|delete)\('
)

# File path patterns
FILE_PATTERNS=(
    '/data/[^\s"'\'']+'
    '/var/[^\s"'\'']+'
    '/tmp/[^\s"'\'']+'
    '\.csv'
    '\.json'
    '\.xml'
    '\.yaml'
    '\.parquet'
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

log_info() {
    echo -e "${CYAN}[Data-Aware RAG]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_ghost() {
    echo -e "${RED}[GHOST DEPENDENCY]${NC} $1"
}

# Search for patterns in target directory
search_patterns() {
    local pattern_type="$1"
    local target_dir="$2"
    shift 2
    local patterns=("$@")

    local output_file="${OUTPUT_DIR}/${pattern_type}_matches.txt"
    > "$output_file"

    local match_count=0

    for pattern in "${patterns[@]}"; do
        if [[ "$VERBOSE" == "true" ]]; then
            log_info "Searching: $pattern"
        fi

        # Use grep with extended regex, ignore binary files
        local results
        results=$(grep -rn -E "$pattern" "$target_dir" \
            --include="*.py" \
            --include="*.js" \
            --include="*.ts" \
            --include="*.rs" \
            --include="*.go" \
            --include="*.java" \
            --include="*.rb" \
            --include="*.php" \
            --include="*.sh" \
            --include="*.sql" \
            --include="*.yaml" \
            --include="*.yml" \
            --include="*.json" \
            --include="*.toml" \
            2>/dev/null || true)

        if [[ -n "$results" ]]; then
            echo "# Pattern: $pattern" >> "$output_file"
            echo "$results" >> "$output_file"
            echo "" >> "$output_file"
            match_count=$((match_count + $(echo "$results" | wc -l)))
        fi
    done

    echo "$match_count"
}

# Extract table names from SQL matches
extract_tables() {
    local sql_file="$1"
    local output_file="${OUTPUT_DIR}/tables_found.txt"

    grep -oE '(FROM|INTO|UPDATE|JOIN|TABLE|VIEW)\s+[a-zA-Z_]+' "$sql_file" 2>/dev/null | \
        sed 's/FROM\s*//; s/INTO\s*//; s/UPDATE\s*//; s/JOIN\s*//; s/TABLE\s*//; s/VIEW\s*//' | \
        sort -u > "$output_file"

    wc -l < "$output_file"
}

# Generate risk assessment
assess_risk() {
    local match_file="$1"
    local pattern_type="$2"

    local high_risk=0
    local medium_risk=0
    local low_risk=0

    case "$pattern_type" in
        "sql")
            # Raw SQL in app code = HIGH risk
            high_risk=$(grep -c "SELECT\|INSERT\|UPDATE\|DELETE" "$match_file" 2>/dev/null || echo 0)
            ;;
        "json")
            # Hardcoded JSON keys = MEDIUM risk
            medium_risk=$(wc -l < "$match_file" 2>/dev/null || echo 0)
            ;;
        "config")
            # Config access = LOW risk (if using proper config management)
            low_risk=$(wc -l < "$match_file" 2>/dev/null || echo 0)
            ;;
        "api")
            # Inline endpoints = MEDIUM risk
            medium_risk=$(wc -l < "$match_file" 2>/dev/null || echo 0)
            ;;
        "file")
            # Hardcoded paths = MEDIUM risk
            medium_risk=$(wc -l < "$match_file" 2>/dev/null || echo 0)
            ;;
    esac

    echo "$high_risk:$medium_risk:$low_risk"
}

# Generate final report
generate_report() {
    local target_dir="$1"
    local report_file="${OUTPUT_DIR}/shadow_logic_report.md"

    cat > "$report_file" << EOF
# Shadow Logic Detection Report

**Generated:** $(date -Iseconds)
**Target Directory:** $target_dir

## Summary

EOF

    # Read counts from files
    local sql_count=$(wc -l < "${OUTPUT_DIR}/sql_matches.txt" 2>/dev/null || echo 0)
    local orm_count=$(wc -l < "${OUTPUT_DIR}/orm_matches.txt" 2>/dev/null || echo 0)
    local json_count=$(wc -l < "${OUTPUT_DIR}/json_matches.txt" 2>/dev/null || echo 0)
    local config_count=$(wc -l < "${OUTPUT_DIR}/config_matches.txt" 2>/dev/null || echo 0)
    local api_count=$(wc -l < "${OUTPUT_DIR}/api_matches.txt" 2>/dev/null || echo 0)
    local file_count=$(wc -l < "${OUTPUT_DIR}/file_matches.txt" 2>/dev/null || echo 0)

    local total=$((sql_count + orm_count + json_count + config_count + api_count + file_count))

    cat >> "$report_file" << EOF
| Category | Matches | Risk Level |
|----------|---------|------------|
| SQL Patterns | $sql_count | HIGH |
| ORM Patterns | $orm_count | LOW |
| JSON Keys | $json_count | MEDIUM |
| Config Access | $config_count | LOW |
| API Endpoints | $api_count | MEDIUM |
| File Paths | $file_count | MEDIUM |
| **TOTAL** | **$total** | - |

## Ghost Dependencies

These are shadow dependencies NOT tracked in typical Code Graph analysis:

EOF

    # Add ghost dependencies (SQL matches are the primary culprits)
    if [[ -f "${OUTPUT_DIR}/sql_matches.txt" ]] && [[ -s "${OUTPUT_DIR}/sql_matches.txt" ]]; then
        echo "### Raw SQL (HIGH RISK)" >> "$report_file"
        echo '```' >> "$report_file"
        head -50 "${OUTPUT_DIR}/sql_matches.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    # Tables found
    if [[ -f "${OUTPUT_DIR}/tables_found.txt" ]] && [[ -s "${OUTPUT_DIR}/tables_found.txt" ]]; then
        echo "### Database Tables Referenced" >> "$report_file"
        echo '```' >> "$report_file"
        cat "${OUTPUT_DIR}/tables_found.txt" >> "$report_file"
        echo '```' >> "$report_file"
    fi

    cat >> "$report_file" << EOF

## Recommendations

1. **HIGH RISK (SQL):** Convert raw SQL to parameterized queries or ORM
2. **MEDIUM RISK (JSON):** Use typed data contracts instead of string keys
3. **MEDIUM RISK (API):** Move endpoints to configuration management
4. **LOW RISK (Config):** Already using config - ensure secrets are externalized

## Files Modified

See individual match files in \`${OUTPUT_DIR}/\` for full details.
EOF

    echo "$report_file"
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

main() {
    local target_dir="${1:-.}"

    log_info "═══════════════════════════════════════════════════════════════"
    log_info "         DATA-AWARE RAG: Shadow Logic Detection"
    log_info "═══════════════════════════════════════════════════════════════"
    log_info "Target: $target_dir"
    log_info "Output: $OUTPUT_DIR"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Layer 3: Literal Scanner
    log_info "[Layer 3: Literal Scanner] Starting pattern search..."
    echo ""

    # SQL patterns
    log_info "Scanning for SQL patterns..."
    sql_count=$(search_patterns "sql" "$target_dir" "${SQL_PATTERNS[@]}")
    log_info "  Found: $sql_count SQL matches"

    # ORM patterns
    log_info "Scanning for ORM patterns..."
    orm_count=$(search_patterns "orm" "$target_dir" "${ORM_PATTERNS[@]}")
    log_info "  Found: $orm_count ORM matches"

    # JSON patterns
    log_info "Scanning for JSON key patterns..."
    json_count=$(search_patterns "json" "$target_dir" "${JSON_PATTERNS[@]}")
    log_info "  Found: $json_count JSON key matches"

    # Config patterns
    log_info "Scanning for config/env patterns..."
    config_count=$(search_patterns "config" "$target_dir" "${CONFIG_PATTERNS[@]}")
    log_info "  Found: $config_count config matches"

    # API patterns
    log_info "Scanning for API endpoint patterns..."
    api_count=$(search_patterns "api" "$target_dir" "${API_PATTERNS[@]}")
    log_info "  Found: $api_count API endpoint matches"

    # File path patterns
    log_info "Scanning for file path patterns..."
    file_count=$(search_patterns "file" "$target_dir" "${FILE_PATTERNS[@]}")
    log_info "  Found: $file_count file path matches"

    echo ""

    # Extract table names
    if [[ -f "${OUTPUT_DIR}/sql_matches.txt" ]]; then
        log_info "Extracting database table names..."
        table_count=$(extract_tables "${OUTPUT_DIR}/sql_matches.txt")
        log_info "  Found: $table_count unique tables"
    fi

    echo ""

    # Generate report
    log_info "Generating Shadow Logic Report..."
    report_file=$(generate_report "$target_dir")
    log_success "Report generated: $report_file"

    echo ""
    log_info "═══════════════════════════════════════════════════════════════"

    # Summary output
    local total=$((sql_count + orm_count + json_count + config_count + api_count + file_count))

    if [[ $sql_count -gt 0 ]]; then
        log_ghost "Found $sql_count raw SQL patterns - these are GHOST DEPENDENCIES!"
        log_warn "Code Graph will NOT track these. Vector search may miss exact matches."
    fi

    log_info "Total Shadow Logic patterns found: $total"
    log_info "Full report: $report_file"

    # Exit with warning code if ghost dependencies found
    if [[ $sql_count -gt 0 ]]; then
        exit 1
    fi
}

# Run with directory argument or current directory
main "$@"
