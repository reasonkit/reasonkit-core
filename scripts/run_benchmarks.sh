#!/usr/bin/env bash
# ReasonKit Core - Comprehensive Benchmark Runner
# Performance Target: All core loops < 5ms
#
# Usage:
#   ./scripts/run_benchmarks.sh              # Run all benchmarks
#   ./scripts/run_benchmarks.sh fusion       # Run specific benchmark
#   ./scripts/run_benchmarks.sh --baseline   # Save baseline for comparison
#   ./scripts/run_benchmarks.sh --compare    # Compare against baseline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Benchmark configuration
BENCH_DIR="target/criterion"
BASELINE_NAME="master"
SAVE_BASELINE=false
COMPARE_BASELINE=false
SPECIFIC_BENCH=""

# Performance thresholds (in nanoseconds)
THRESHOLD_5MS=5000000  # 5ms in nanoseconds

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            SAVE_BASELINE=true
            shift
            ;;
        --compare)
            COMPARE_BASELINE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [BENCHMARK]"
            echo ""
            echo "Options:"
            echo "  --baseline    Save current results as baseline"
            echo "  --compare     Compare against saved baseline"
            echo "  --help        Show this help message"
            echo ""
            echo "Benchmarks:"
            echo "  retrieval     Retrieval and search operations"
            echo "  fusion        RRF and fusion algorithms"
            echo "  embedding     Embedding generation and caching"
            echo "  raptor        RAPTOR tree operations"
            echo "  ingestion     Document ingestion pipeline"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all benchmarks"
            echo "  $0 fusion             # Run only fusion benchmarks"
            echo "  $0 --baseline         # Save baseline"
            echo "  $0 --compare fusion   # Compare fusion against baseline"
            exit 0
            ;;
        *)
            SPECIFIC_BENCH="$1"
            shift
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ReasonKit Core Benchmark Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Build in release mode first
echo -e "${YELLOW}Building in release mode...${NC}"
cargo build --release --quiet

# Available benchmarks
BENCHMARKS=(
    "retrieval_bench"
    "fusion_bench"
    "embedding_bench"
    "raptor_bench"
    "ingestion_bench"
)

# Filter benchmarks if specific one requested
if [[ -n "${SPECIFIC_BENCH}" ]]; then
    case "${SPECIFIC_BENCH}" in
        retrieval)
            BENCHMARKS=("retrieval_bench")
            ;;
        fusion)
            BENCHMARKS=("fusion_bench")
            ;;
        embedding)
            BENCHMARKS=("embedding_bench")
            ;;
        raptor)
            BENCHMARKS=("raptor_bench")
            ;;
        ingestion)
            BENCHMARKS=("ingestion_bench")
            ;;
        *)
            echo -e "${RED}Unknown benchmark: ${SPECIFIC_BENCH}${NC}"
            echo "Available: retrieval, fusion, embedding, raptor, ingestion"
            exit 1
            ;;
    esac
fi

# Run benchmarks
for bench in "${BENCHMARKS[@]}"; do
    echo ""
    echo -e "${GREEN}Running: ${bench}${NC}"
    echo "----------------------------------------"

    if [[ "${SAVE_BASELINE}" == "true" ]]; then
        cargo bench --bench "${bench}" -- --save-baseline "${BASELINE_NAME}"
    elif [[ "${COMPARE_BASELINE}" == "true" ]]; then
        cargo bench --bench "${bench}" -- --baseline "${BASELINE_NAME}"
    else
        cargo bench --bench "${bench}"
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Benchmark Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if HTML reports were generated
if [[ -d "${BENCH_DIR}" ]]; then
    echo -e "${BLUE}HTML reports available at:${NC}"
    echo "  file://${PROJECT_ROOT}/${BENCH_DIR}/report/index.html"
    echo ""
fi

# Performance analysis
echo -e "${YELLOW}Performance Analysis:${NC}"
echo "  Target: All core loops < 5ms"
echo ""

# Check for violations (simplified - would need actual parsing in production)
if [[ -d "${BENCH_DIR}" ]]; then
    echo "  Check detailed results in Criterion reports above"
    echo ""

    # Look for any benchmarks that might exceed threshold
    echo -e "${YELLOW}Note: Review the following for < 5ms compliance:${NC}"
    echo "  - RRF fusion operations (should be < 5ms for 100 results)"
    echo "  - Embedding cache hits (should be < 1ms)"
    echo "  - Tree traversal (should be < 5ms for 1000 nodes)"
    echo "  - Chunk operations (should be < 5ms for typical docs)"
fi

echo ""
echo -e "${GREEN}Benchmarks complete!${NC}"
echo ""

# Baseline management tips
if [[ "${SAVE_BASELINE}" == "true" ]]; then
    echo -e "${BLUE}Baseline saved as '${BASELINE_NAME}'${NC}"
    echo "Run with --compare to check for regressions"
elif [[ "${COMPARE_BASELINE}" == "true" ]]; then
    echo -e "${BLUE}Comparison complete${NC}"
    echo "Check for significant changes (> 5% regression)"
else
    echo -e "${YELLOW}Tip:${NC} Use --baseline to save current results"
    echo -e "${YELLOW}Tip:${NC} Use --compare to check for regressions"
fi

echo ""
