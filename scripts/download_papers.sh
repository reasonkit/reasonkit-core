#!/bin/bash
# ReasonKit Core - Academic Paper Downloader
# Downloads papers for theoretical grounding of ReasonKit
#
# Usage: ./download_papers.sh [--all | --priority N | --paper ID]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/papers/raw"
PROCESSED_DIR="${SCRIPT_DIR}/../data/papers/processed"

# Create directories
mkdir -p "$DATA_DIR" "$PROCESSED_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Download a paper from arXiv
download_arxiv() {
    local arxiv_id="$1"
    local output_name="$2"
    local output_path="${DATA_DIR}/${output_name}.pdf"

    if [[ -f "$output_path" ]]; then
        log_info "Already exists: $output_name"
        return 0
    fi

    local url="https://arxiv.org/pdf/${arxiv_id}.pdf"
    log_info "Downloading: $output_name from arXiv:${arxiv_id}"

    if curl -sL -o "$output_path" "$url"; then
        # Verify it's a valid PDF
        if file "$output_path" | grep -q "PDF"; then
            log_info "Success: $output_name"
            return 0
        else
            log_error "Invalid PDF: $output_name"
            rm -f "$output_path"
            return 1
        fi
    else
        log_error "Failed to download: $output_name"
        return 1
    fi
}

# Download from direct URL
download_url() {
    local url="$1"
    local output_name="$2"
    local output_path="${DATA_DIR}/${output_name}.pdf"

    if [[ -f "$output_path" ]]; then
        log_info "Already exists: $output_name"
        return 0
    fi

    log_info "Downloading: $output_name from URL"

    if curl -sL -o "$output_path" "$url"; then
        if file "$output_path" | grep -q "PDF"; then
            log_info "Success: $output_name"
            return 0
        else
            log_error "Invalid PDF: $output_name"
            rm -f "$output_path"
            return 1
        fi
    else
        log_error "Failed to download: $output_name"
        return 1
    fi
}

# Core Reasoning Papers (Priority 1)
download_priority_1() {
    log_info "=== Downloading Priority 1 Papers (Core Reasoning) ==="

    # Chain-of-Thought Prompting
    download_arxiv "2201.11903" "cot_wei_2022"

    # Self-Consistency
    download_arxiv "2203.11171" "self_consistency_wang_2022"

    # Tree of Thoughts
    download_arxiv "2305.10601" "tree_of_thoughts_yao_2023"

    # RAPTOR
    download_arxiv "2401.18059" "raptor_sarthi_2024"

    # Reflexion
    download_arxiv "2303.11366" "reflexion_shinn_2023"

    # Constitutional AI
    download_arxiv "2212.08073" "constitutional_ai_anthropic_2022"

    # Let's Verify Step by Step (OpenAI - direct URL)
    download_url "https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf" "lets_verify_openai_2023"

    # OmegaPRM
    download_arxiv "2406.06592" "omegaprm_2024"

    log_info "=== Priority 1 Complete ==="
}

# Retrieval & Embedding Papers (Priority 2)
download_priority_2() {
    log_info "=== Downloading Priority 2 Papers (Retrieval & Embedding) ==="

    # ColBERT
    download_arxiv "2004.12832" "colbert_khattab_2020"

    # BGE-M3
    download_arxiv "2402.03216" "bge_m3_2024"

    # E5 Embeddings
    download_arxiv "2212.03533" "e5_embeddings_2022"

    # DeepSeek-R1
    download_arxiv "2501.12948" "deepseek_r1_2025"

    log_info "=== Priority 2 Complete ==="
}

# Benchmark Papers (Priority 3)
download_priority_3() {
    log_info "=== Downloading Priority 3 Papers (Benchmarks) ==="

    # GSM8K
    download_arxiv "2110.14168" "gsm8k_cobbe_2021"

    # MATH Benchmark
    download_arxiv "2103.03874" "math_hendrycks_2021"

    # MMLU
    download_arxiv "2009.03300" "mmlu_hendrycks_2020"

    log_info "=== Priority 3 Complete ==="
}

# Generate metadata JSON for downloaded papers
generate_metadata() {
    log_info "=== Generating Metadata ==="

    local metadata_file="${DATA_DIR}/../download_status.json"

    echo "{" > "$metadata_file"
    echo '  "generated_at": "'$(date -Iseconds)'",' >> "$metadata_file"
    echo '  "papers": [' >> "$metadata_file"

    local first=true
    for pdf in "$DATA_DIR"/*.pdf; do
        if [[ -f "$pdf" ]]; then
            local basename=$(basename "$pdf" .pdf)
            local size=$(stat -c%s "$pdf" 2>/dev/null || stat -f%z "$pdf" 2>/dev/null)

            if [[ "$first" == "true" ]]; then
                first=false
            else
                echo "," >> "$metadata_file"
            fi

            echo -n '    {"name": "'$basename'", "size": '$size', "status": "downloaded"}' >> "$metadata_file"
        fi
    done

    echo '' >> "$metadata_file"
    echo '  ]' >> "$metadata_file"
    echo "}" >> "$metadata_file"

    log_info "Metadata written to: $metadata_file"
}

# Main
main() {
    log_info "ReasonKit Core - Paper Downloader"
    log_info "Output directory: $DATA_DIR"
    echo ""

    case "${1:-all}" in
        --all)
            download_priority_1
            download_priority_2
            download_priority_3
            ;;
        --priority)
            case "${2:-1}" in
                1) download_priority_1 ;;
                2) download_priority_2 ;;
                3) download_priority_3 ;;
                *) log_error "Unknown priority: $2"; exit 1 ;;
            esac
            ;;
        --help)
            echo "Usage: $0 [--all | --priority N | --help]"
            echo ""
            echo "Options:"
            echo "  --all         Download all papers (default)"
            echo "  --priority N  Download only priority N papers (1, 2, or 3)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            download_priority_1
            download_priority_2
            download_priority_3
            ;;
    esac

    generate_metadata

    echo ""
    log_info "Download complete!"
    log_info "Papers saved to: $DATA_DIR"
    log_info "Total files: $(ls -1 "$DATA_DIR"/*.pdf 2>/dev/null | wc -l)"
}

main "$@"
