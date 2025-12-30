#!/usr/bin/env bash
# ReasonKit Benchmark Validation Script
# Purpose: Validate benchmark performance with real data (Task 119)
# Usage: ./scripts/validate_benchmarks.sh [--mock-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
MOCK_ONLY=false
SAMPLES=20
SC_SAMPLES=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mock-only)
            MOCK_ONLY=true
            shift
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --sc-samples)
            SC_SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mock-only      Run only mock mode (no API calls)"
            echo "  --samples N      Number of problems to test (default: 20)"
            echo "  --sc-samples N   Self-Consistency samples (default: 5)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        ReasonKit Benchmark Validation Suite                        ║${NC}"
echo -e "${CYAN}║              Task 119: Validate with Real Data                      ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if release build exists
if [[ ! -f "target/release/rk-bench" ]] && [[ ! -f "target/release/bench" ]]; then
    echo -e "${YELLOW}Building release binary...${NC}"
    cargo build --release --bin bench 2>&1 | grep -E "(Compiling|Finished)" || true
    echo ""
fi

# Check API keys
USE_MOCK=true
if [[ "$MOCK_ONLY" == "false" ]]; then
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]] || [[ -n "${OPENAI_API_KEY:-}" ]]; then
        echo -e "${GREEN}✅ API key found. Running with real LLM calls.${NC}"
        USE_MOCK=false
    else
        echo -e "${YELLOW}⚠️  No API key found. Running in mock mode only.${NC}"
        echo -e "${YELLOW}   Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real validation.${NC}"
        USE_MOCK=true
    fi
else
    echo -e "${BLUE}ℹ️  Mock-only mode requested.${NC}"
    USE_MOCK=true
fi

echo ""

# Create results directory
RESULTS_DIR="benchmark_validation_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}📊 Results will be saved to: ${RESULTS_DIR}${NC}"
echo ""

# Phase 1: Criterion Benchmarks (Performance)
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Phase 1: Criterion Performance Benchmarks${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if cargo bench --bench retrieval_bench --bench fusion_bench -- --quick 2>&1 | tee "$RESULTS_DIR/criterion.log" | grep -q "test result"; then
    echo -e "${GREEN}✅ Criterion benchmarks completed${NC}"
else
    echo -e "${YELLOW}⚠️  Criterion benchmarks may have issues (check log)${NC}"
fi

echo ""

# Phase 2: Mock Mode Validation
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Phase 2: Mock Mode Validation (Logic Check)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Running GSM8K benchmark (mock mode, ${SAMPLES} samples)...${NC}"

if cargo run --release --bin bench -- \
    --benchmark gsm8k \
    --samples "$SAMPLES" \
    --self-consistency "$SC_SAMPLES" \
    --mock \
    --profile balanced \
    --output json \
    > "$RESULTS_DIR/gsm8k_mock.json" 2>&1; then
    
    echo -e "${GREEN}✅ Mock mode validation complete${NC}"
    
    # Validate JSON structure
    if command -v jq >/dev/null; then
        if jq empty "$RESULTS_DIR/gsm8k_mock.json" 2>/dev/null; then
            echo -e "${GREEN}✅ Results JSON is valid${NC}"
            
            # Show summary
            echo ""
            echo -e "${BLUE}Mock Mode Results Summary:${NC}"
            jq -r '
                "Baseline Accuracy: \(.baseline_accuracy * 100 | floor)%",
                "SC Accuracy: \(.sc_accuracy * 100 | floor)%",
                "Delta: \(.accuracy_delta * 100 | floor)%",
                "Significant: \(.is_significant)",
                "Token Multiplier: \(.token_multiplier | floor)x"
            ' "$RESULTS_DIR/gsm8k_mock.json"
        else
            echo -e "${RED}❌ Results JSON is invalid${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠️  Install 'jq' for JSON validation: apt install jq / brew install jq${NC}"
    fi
else
    echo -e "${RED}❌ Mock mode validation failed${NC}"
    exit 1
fi

echo ""

# Phase 3: Real Data Validation (if API key available)
if [[ "$USE_MOCK" == "false" ]]; then
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Phase 3: Real Data Validation${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    echo -e "${YELLOW}⚠️  This will make real API calls and incur costs (~\$0.50-\$2.00)${NC}"
    echo -e "${YELLOW}   Press Ctrl+C to cancel, or wait 5 seconds to continue...${NC}"
    sleep 5
    
    echo ""
    echo -e "${BLUE}Running GSM8K benchmark (real LLM, ${SAMPLES} samples)...${NC}"
    
    if cargo run --release --bin bench -- \
        --benchmark gsm8k \
        --samples "$SAMPLES" \
        --self-consistency "$SC_SAMPLES" \
        --mock false \
        --profile balanced \
        --output json \
        > "$RESULTS_DIR/gsm8k_real.json" 2>&1; then
        
        echo -e "${GREEN}✅ Real data validation complete${NC}"
        
        # Validate and show results
        if command -v jq >/dev/null; then
            if jq empty "$RESULTS_DIR/gsm8k_real.json" 2>/dev/null; then
                echo ""
                echo -e "${BLUE}Real Data Results Summary:${NC}"
                jq -r '
                    "Baseline Accuracy: \(.baseline_accuracy * 100 | floor)%",
                    "SC Accuracy: \(.sc_accuracy * 100 | floor)%",
                    "Delta: \(.accuracy_delta * 100 | floor)%",
                    "Significant: \(.is_significant)",
                    "Token Multiplier: \(.token_multiplier | floor)x",
                    "Avg Latency: \(.baseline_avg_latency_ms | floor)ms"
                ' "$RESULTS_DIR/gsm8k_real.json"
                
                # Check if improvement is significant
                DELTA=$(jq -r '.accuracy_delta' "$RESULTS_DIR/gsm8k_real.json")
                SIGNIFICANT=$(jq -r '.is_significant' "$RESULTS_DIR/gsm8k_real.json")
                
                echo ""
                if (( $(echo "$DELTA > 0.05" | bc -l) )) && [[ "$SIGNIFICANT" == "true" ]]; then
                    echo -e "${GREEN}✅ VALIDATED: Meaningful improvement observed (+$(echo "$DELTA * 100" | bc -l | xargs printf "%.1f")%)${NC}"
                elif (( $(echo "$DELTA > 0" | bc -l) )); then
                    echo -e "${YELLOW}⚠️  MARGINAL: Improvement observed but may not be significant${NC}"
                else
                    echo -e "${RED}❌ NO IMPROVEMENT: ThinkTools did not improve accuracy${NC}"
                fi
            else
                echo -e "${RED}❌ Results JSON is invalid${NC}"
                exit 1
            fi
        fi
    else
        echo -e "${RED}❌ Real data validation failed${NC}"
        echo -e "${YELLOW}   Check API key and network connectivity${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⏭️  Skipping real data validation (mock-only or no API key)${NC}"
fi

echo ""

# Final Summary
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Validation Summary${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}✅ Validation complete!${NC}"
echo ""
echo -e "${BLUE}Results saved to: ${RESULTS_DIR}${NC}"
echo ""
echo "Files generated:"
ls -lh "$RESULTS_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
echo ""

if command -v jq >/dev/null && [[ -f "$RESULTS_DIR/gsm8k_mock.json" ]]; then
    echo -e "${BLUE}Quick Stats:${NC}"
    echo "  Mock Mode: $(jq -r '"\(.baseline_accuracy * 100 | floor)% baseline, \(.sc_accuracy * 100 | floor)% SC"' "$RESULTS_DIR/gsm8k_mock.json")"
    if [[ -f "$RESULTS_DIR/gsm8k_real.json" ]]; then
        echo "  Real Data: $(jq -r '"\(.baseline_accuracy * 100 | floor)% baseline, \(.sc_accuracy * 100 | floor)% SC"' "$RESULTS_DIR/gsm8k_real.json")"
    fi
fi

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review results in: $RESULTS_DIR"
echo "  2. Check HTML reports: target/criterion/report/index.html"
echo "  3. Document findings in: docs/BENCHMARK_VALIDATION_REPORT.md"
echo ""

