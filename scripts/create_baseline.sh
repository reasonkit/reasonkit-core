#!/usr/bin/env bash
# Create Baseline Measurements
# Task: rk-project.core.benchmarks #50
#
# This script creates baseline measurements for reasonkit-core before improvements.
# It runs both performance benchmarks (Criterion.rs) and reasoning quality benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Creating Baseline Measurements${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create results directory
RESULTS_DIR="benchmarks/results/baseline"
mkdir -p "${RESULTS_DIR}"

# Save environment info
echo -e "${YELLOW}Capturing environment...${NC}"
{
    echo "Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    echo "Rust: $(rustc --version)"
    echo "Cargo: $(cargo --version)"
    echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
    echo "Git branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
} > "${RESULTS_DIR}/environment.txt"

cat "${RESULTS_DIR}/environment.txt"
echo ""

# Performance Benchmarks (Criterion.rs)
echo -e "${YELLOW}Running performance benchmarks...${NC}"
echo "This may take several minutes..."
echo ""

if ./scripts/run_benchmarks.sh --baseline; then
    echo -e "${GREEN}✓ Performance baseline saved${NC}"
    echo "  Baseline name: master"
    echo "  Location: target/criterion/"
else
    echo -e "${YELLOW}⚠ Performance benchmarks failed or incomplete${NC}"
    echo "  Check output above for errors"
fi

echo ""

# Reasoning Quality Benchmarks
echo -e "${YELLOW}Running reasoning quality benchmarks...${NC}"
echo ""

# Check if API key is set
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo -e "${YELLOW}⚠ No API key found (ANTHROPIC_API_KEY or OPENAI_API_KEY)${NC}"
    echo "  Skipping reasoning quality benchmarks"
    echo "  Set API key and run manually:"
    echo "    cargo run --release --bin gsm8k_eval -- --samples 100"
else
    echo "Running GSM8K baseline (100 samples)..."
    if cargo run --release --bin gsm8k_eval -- --samples 100 --output "${RESULTS_DIR}/gsm8k_baseline.json" 2>&1 | tee "${RESULTS_DIR}/gsm8k_baseline.log"; then
        echo -e "${GREEN}✓ GSM8K baseline saved${NC}"
        echo "  Location: ${RESULTS_DIR}/gsm8k_baseline.json"
    else
        echo -e "${YELLOW}⚠ GSM8K benchmark failed${NC}"
        echo "  Check ${RESULTS_DIR}/gsm8k_baseline.log for errors"
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Baseline Creation Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Review baseline results"
echo "  2. Update benchmarks/BASELINE_MEASUREMENTS.md with actual numbers"
echo "  3. Commit baseline to version control"
echo "  4. Use baseline for comparison after improvements"

