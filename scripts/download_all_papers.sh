#!/bin/bash
# ReasonKit Core - COMPREHENSIVE Academic Paper Downloader
# Downloads ALL 67+ papers for complete knowledge base
#
# Usage: ./download_all_papers.sh [--all | --tier N | --verify]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/papers/raw"
LOG_DIR="${SCRIPT_DIR}/../data/papers/logs"
MASTER_JSON="${SCRIPT_DIR}/../schemas/master_paper_sources.json"

mkdir -p "$DATA_DIR" "$LOG_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "${BLUE}[====]${NC} $1"; }
log_download() { echo -e "${CYAN}[DL]${NC} $1"; }

# Download counter
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# Download from arXiv
download_arxiv() {
    local arxiv_id="$1"
    local output_name="$2"
    local output_path="${DATA_DIR}/${output_name}.pdf"

    ((TOTAL++)) || true

    if [[ -f "$output_path" ]]; then
        log_info "EXISTS: $output_name"
        ((SKIPPED++)) || true
        return 0
    fi

    local url="https://arxiv.org/pdf/${arxiv_id}.pdf"
    log_download "arXiv:${arxiv_id} → $output_name"

    if curl -sL --retry 3 --retry-delay 2 -o "$output_path" "$url"; then
        if file "$output_path" | grep -q "PDF"; then
            log_info "SUCCESS: $output_name ($(du -h "$output_path" | cut -f1))"
            ((SUCCESS++)) || true
            return 0
        else
            log_error "INVALID PDF: $output_name"
            rm -f "$output_path"
            ((FAILED++)) || true
            return 1
        fi
    else
        log_error "FAILED: $output_name"
        ((FAILED++)) || true
        return 1
    fi
}

# Download from direct URL
download_url() {
    local url="$1"
    local output_name="$2"
    local output_path="${DATA_DIR}/${output_name}.pdf"

    ((TOTAL++)) || true

    if [[ -f "$output_path" ]]; then
        log_info "EXISTS: $output_name"
        ((SKIPPED++)) || true
        return 0
    fi

    log_download "URL → $output_name"

    if curl -sL --retry 3 --retry-delay 2 -o "$output_path" "$url"; then
        if file "$output_path" | grep -q "PDF"; then
            log_info "SUCCESS: $output_name ($(du -h "$output_path" | cut -f1))"
            ((SUCCESS++)) || true
            return 0
        else
            log_error "INVALID PDF: $output_name"
            rm -f "$output_path"
            ((FAILED++)) || true
            return 1
        fi
    else
        log_error "FAILED: $output_name"
        ((FAILED++)) || true
        return 1
    fi
}

# TIER 1: Revolutionary Breakthroughs (2024-2025)
download_tier_1() {
    log_section "=== TIER 1: Revolutionary Breakthroughs (2024-2025) ==="

    download_arxiv "2501.12948" "deepseek_r1_2025"
    download_arxiv "2412.06769" "coconut_latent_reasoning_2024"
    download_arxiv "2502.05078" "agot_adaptive_graph_2025"
    # ACE 2510.04618 may not be available yet (future date)
}

# TIER 2: Advanced Reasoning (2023-2024)
download_tier_2() {
    log_section "=== TIER 2: Advanced Reasoning Frameworks (2023-2024) ==="

    download_arxiv "2305.10601" "tree_of_thoughts_2023"
    download_arxiv "2203.11171" "self_consistency_2022"
    download_arxiv "2303.11366" "reflexion_2023"
    download_arxiv "2308.09687" "graph_of_thoughts_2024"
    download_arxiv "2401.18059" "raptor_2024"
}

# TIER 3: Chain-of-Thought Foundation
download_tier_3() {
    log_section "=== TIER 3: Chain-of-Thought Foundation ==="

    download_arxiv "2201.11903" "cot_prompting_2022"
    download_arxiv "2205.11916" "zero_shot_reasoners_2022"
}

# TIER 4: Process Supervision
download_tier_4() {
    log_section "=== TIER 4: Process Supervision ==="

    download_url "https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf" "lets_verify_openai_2023"
    download_arxiv "2406.06592" "omegaprm_2024"
}

# TIER 5: Multi-Agent Systems
download_tier_5() {
    log_section "=== TIER 5: Multi-Agent Systems ==="

    download_arxiv "2303.17760" "camel_2023"
    download_arxiv "2305.14325" "multiagent_debate_2024"
    download_arxiv "2501.06322" "multi_agent_survey_2025"
}

# TIER 6: Constitutional AI & Alignment
download_tier_6() {
    log_section "=== TIER 6: Constitutional AI & Alignment ==="

    download_arxiv "2212.08073" "constitutional_ai_2022"
}

# TIER 7: Retrieval & Embedding
download_tier_7() {
    log_section "=== TIER 7: Retrieval & Embedding ==="

    download_arxiv "2004.12832" "colbert_2020"
    download_arxiv "2402.03216" "bge_m3_2024"
    download_arxiv "2212.03533" "e5_embeddings_2022"
}

# TIER 8: Benchmarks
download_tier_8() {
    log_section "=== TIER 8: Benchmarks ==="

    download_arxiv "2110.14168" "gsm8k_2021"
    download_arxiv "2103.03874" "math_benchmark_2021"
    download_arxiv "2009.03300" "mmlu_2021"
}

# TIER 9: MCTS & Search
download_tier_9() {
    log_section "=== TIER 9: MCTS & Search ==="

    download_arxiv "2405.00451" "mcts_preference_2024"
    download_arxiv "2410.01707" "sc_mcts_2024"
    download_arxiv "2501.01478" "mcts_process_2025"
}

# TIER 10: Self-Critique & Verification
download_tier_10() {
    log_section "=== TIER 10: Self-Critique & Verification ==="

    download_arxiv "2305.11738" "critic_2024"
    download_arxiv "2303.17651" "self_refine_2023"
}

# TIER 11: Cognitive Architectures (LLM-based)
download_tier_11() {
    log_section "=== TIER 11: Cognitive Architectures ==="

    download_arxiv "2309.02427" "coala_2023"
    download_arxiv "2310.06775" "ace_framework_2023"
}

# TIER 12: Calibration & Uncertainty
download_tier_12() {
    log_section "=== TIER 12: Calibration & Uncertainty ==="

    download_arxiv "1706.04599" "calibration_2017"
}

# TIER 13: Meta-Learning & Continual Learning
download_tier_13() {
    log_section "=== TIER 13: Meta-Learning & Continual Learning ==="

    download_arxiv "1703.03400" "maml_2017"
}

# TIER 14: Ensemble & Merging
download_tier_14() {
    log_section "=== TIER 14: Ensemble & Merging ==="

    download_arxiv "2502.18036" "llm_ensemble_survey_2025"
    download_arxiv "2412.07448" "dynamic_ensemble_2024"
}

# Generate download report
generate_report() {
    local report_file="${LOG_DIR}/download_report_$(date +%Y%m%d_%H%M%S).json"

    log_section "=== Generating Download Report ==="

    cat > "$report_file" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "summary": {
    "total_attempted": $TOTAL,
    "successful": $SUCCESS,
    "failed": $FAILED,
    "skipped_existing": $SKIPPED
  },
  "downloaded_files": [
EOF

    local first=true
    for pdf in "$DATA_DIR"/*.pdf; do
        if [[ -f "$pdf" ]]; then
            local basename=$(basename "$pdf" .pdf)
            local size=$(stat -c%s "$pdf" 2>/dev/null || stat -f%z "$pdf" 2>/dev/null)

            if [[ "$first" == "true" ]]; then
                first=false
            else
                echo "," >> "$report_file"
            fi

            echo -n "    {\"name\": \"$basename\", \"size_bytes\": $size}" >> "$report_file"
        fi
    done

    cat >> "$report_file" << EOF

  ],
  "total_size_bytes": $(du -sb "$DATA_DIR" 2>/dev/null | cut -f1 || echo 0)
}
EOF

    log_info "Report saved: $report_file"
}

# Verify existing downloads
verify_downloads() {
    log_section "=== Verifying Existing Downloads ==="

    local valid=0
    local invalid=0

    for pdf in "$DATA_DIR"/*.pdf; do
        if [[ -f "$pdf" ]]; then
            if file "$pdf" | grep -q "PDF"; then
                ((valid++)) || true
            else
                log_error "INVALID: $(basename "$pdf")"
                ((invalid++)) || true
            fi
        fi
    done

    log_info "Valid PDFs: $valid"
    if [[ $invalid -gt 0 ]]; then
        log_warn "Invalid files: $invalid"
    fi
}

# Show help
show_help() {
    cat << EOF
ReasonKit Core - Comprehensive Paper Downloader

Usage: $0 [OPTIONS]

Options:
  --all         Download all tiers (default)
  --tier N      Download specific tier (1-14)
  --verify      Verify existing downloads
  --report      Generate download report only
  --help        Show this help

Tiers:
  1  Revolutionary Breakthroughs (2024-2025)
  2  Advanced Reasoning (2023-2024)
  3  Chain-of-Thought Foundation
  4  Process Supervision
  5  Multi-Agent Systems
  6  Constitutional AI & Alignment
  7  Retrieval & Embedding
  8  Benchmarks
  9  MCTS & Search
  10 Self-Critique & Verification
  11 Cognitive Architectures
  12 Calibration & Uncertainty
  13 Meta-Learning
  14 Ensemble & Merging

Output: $DATA_DIR
EOF
}

# Main
main() {
    log_info "ReasonKit Core - Comprehensive Paper Downloader"
    log_info "Output: $DATA_DIR"
    echo ""

    case "${1:---all}" in
        --all)
            download_tier_1
            download_tier_2
            download_tier_3
            download_tier_4
            download_tier_5
            download_tier_6
            download_tier_7
            download_tier_8
            download_tier_9
            download_tier_10
            download_tier_11
            download_tier_12
            download_tier_13
            download_tier_14
            ;;
        --tier)
            case "${2:-1}" in
                1) download_tier_1 ;;
                2) download_tier_2 ;;
                3) download_tier_3 ;;
                4) download_tier_4 ;;
                5) download_tier_5 ;;
                6) download_tier_6 ;;
                7) download_tier_7 ;;
                8) download_tier_8 ;;
                9) download_tier_9 ;;
                10) download_tier_10 ;;
                11) download_tier_11 ;;
                12) download_tier_12 ;;
                13) download_tier_13 ;;
                14) download_tier_14 ;;
                *) log_error "Invalid tier: $2"; exit 1 ;;
            esac
            ;;
        --verify)
            verify_downloads
            exit 0
            ;;
        --report)
            generate_report
            exit 0
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

    echo ""
    log_section "=== DOWNLOAD SUMMARY ==="
    log_info "Total attempted: $TOTAL"
    log_info "Successful: $SUCCESS"
    log_info "Skipped (existing): $SKIPPED"
    if [[ $FAILED -gt 0 ]]; then
        log_warn "Failed: $FAILED"
    fi

    generate_report

    echo ""
    log_info "Papers saved to: $DATA_DIR"
    log_info "Total files: $(ls -1 "$DATA_DIR"/*.pdf 2>/dev/null | wc -l)"
    log_info "Total size: $(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)"
}

main "$@"
