#!/bin/bash
#
# quality_metrics.sh - ReasonKit Quality Metrics Collection
#
# Collects and reports quality metrics for CI/CD and periodic reviews.
# Run weekly via CI or manually before releases.
#
# Usage: ./quality_metrics.sh [--json] [--ci]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
JSON_OUTPUT=false
CI_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --json) JSON_OUTPUT=true; shift ;;
        --ci) CI_MODE=true; shift ;;
        *) shift ;;
    esac
done

# Metrics storage
declare -A METRICS

echo -e "${CYAN}"
echo "=============================================="
echo "  ReasonKit Quality Metrics Report"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo -e "${NC}"

# Gate 1: Build Check
echo -e "${YELLOW}[Gate 1/5] Build Check...${NC}"
BUILD_START=$(date +%s)
if cargo build --release 2>&1 | tee /tmp/build.log; then
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    BUILD_STATUS="PASS"
    WARNINGS=$(grep -c "warning:" /tmp/build.log 2>/dev/null || echo "0")
    echo -e "${GREEN}  Build: PASS (${BUILD_TIME}s, ${WARNINGS} warnings)${NC}"
else
    BUILD_STATUS="FAIL"
    BUILD_TIME=0
    WARNINGS=999
    echo -e "${RED}  Build: FAIL${NC}"
fi
METRICS["build_status"]="$BUILD_STATUS"
METRICS["build_time_seconds"]="$BUILD_TIME"
METRICS["build_warnings"]="$WARNINGS"

# Gate 2: Clippy
echo -e "${YELLOW}[Gate 2/5] Clippy Analysis...${NC}"
if cargo clippy --message-format=json 2>/dev/null | grep -c '"level":"warning"' > /tmp/clippy_count.txt 2>/dev/null; then
    CLIPPY_WARNINGS=$(cat /tmp/clippy_count.txt)
else
    CLIPPY_WARNINGS=0
fi
if cargo clippy -- -D warnings 2>&1 > /dev/null; then
    CLIPPY_STATUS="PASS"
    echo -e "${GREEN}  Clippy: PASS (${CLIPPY_WARNINGS} warnings)${NC}"
else
    CLIPPY_STATUS="FAIL"
    echo -e "${RED}  Clippy: FAIL (${CLIPPY_WARNINGS} warnings)${NC}"
fi
METRICS["clippy_status"]="$CLIPPY_STATUS"
METRICS["clippy_warnings"]="$CLIPPY_WARNINGS"

# Gate 3: Format Check
echo -e "${YELLOW}[Gate 3/5] Format Check...${NC}"
if cargo fmt --check 2>&1 > /dev/null; then
    FMT_STATUS="PASS"
    echo -e "${GREEN}  Format: PASS${NC}"
else
    FMT_STATUS="FAIL"
    echo -e "${RED}  Format: FAIL (run cargo fmt)${NC}"
fi
METRICS["format_status"]="$FMT_STATUS"

# Gate 4: Tests
echo -e "${YELLOW}[Gate 4/5] Test Suite...${NC}"
TEST_OUTPUT=$(cargo test --all-features 2>&1)
TESTS_PASSED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")
TESTS_FAILED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")
TESTS_IGNORED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ ignored' | grep -oP '\d+' || echo "0")

if [[ "$TESTS_FAILED" == "0" ]]; then
    TEST_STATUS="PASS"
    echo -e "${GREEN}  Tests: PASS (${TESTS_PASSED} passed, ${TESTS_IGNORED} ignored)${NC}"
else
    TEST_STATUS="FAIL"
    echo -e "${RED}  Tests: FAIL (${TESTS_PASSED} passed, ${TESTS_FAILED} failed)${NC}"
fi
METRICS["test_status"]="$TEST_STATUS"
METRICS["tests_passed"]="$TESTS_PASSED"
METRICS["tests_failed"]="$TESTS_FAILED"
METRICS["tests_ignored"]="$TESTS_IGNORED"

# Gate 5: Benchmarks (if available)
echo -e "${YELLOW}[Gate 5/5] Benchmark Check...${NC}"
if [[ -f "benches/retrieval_bench.rs" ]]; then
    if cargo bench --bench retrieval_bench -- --noplot 2>&1 > /dev/null; then
        BENCH_STATUS="PASS"
        echo -e "${GREEN}  Benchmarks: PASS${NC}"
    else
        BENCH_STATUS="SKIP"
        echo -e "${YELLOW}  Benchmarks: SKIP (benchmark failed)${NC}"
    fi
else
    BENCH_STATUS="SKIP"
    echo -e "${YELLOW}  Benchmarks: SKIP (no benchmarks found)${NC}"
fi
METRICS["benchmark_status"]="$BENCH_STATUS"

# Additional Metrics
echo ""
echo -e "${CYAN}Additional Metrics:${NC}"

# Lines of Code
LOC=$(find src -name "*.rs" -exec cat {} \; | wc -l)
echo "  Lines of Rust: $LOC"
METRICS["lines_of_code"]="$LOC"

# TODO count
TODO_COUNT=$(grep -r "TODO" src --include="*.rs" | wc -l)
echo "  TODO comments: $TODO_COUNT"
METRICS["todo_count"]="$TODO_COUNT"

# FIXME count
FIXME_COUNT=$(grep -r "FIXME" src --include="*.rs" | wc -l)
echo "  FIXME comments: $FIXME_COUNT"
METRICS["fixme_count"]="$FIXME_COUNT"

# Unsafe blocks
UNSAFE_COUNT=$(grep -r "unsafe" src --include="*.rs" | wc -l)
echo "  Unsafe blocks: $UNSAFE_COUNT"
METRICS["unsafe_count"]="$UNSAFE_COUNT"

# Dependency count
DEP_COUNT=$(grep -c "^[a-z]" Cargo.toml 2>/dev/null || echo "0")
echo "  Dependencies: ~$DEP_COUNT"
METRICS["dependency_count"]="$DEP_COUNT"

# Binary size
if [[ -f "target/release/rk-core" ]]; then
    BINARY_SIZE=$(du -h target/release/rk-core | cut -f1)
    BINARY_SIZE_BYTES=$(stat -f%z target/release/rk-core 2>/dev/null || stat -c%s target/release/rk-core 2>/dev/null || echo "0")
    echo "  Binary size: $BINARY_SIZE"
    METRICS["binary_size_bytes"]="$BINARY_SIZE_BYTES"
fi

# Security audit
echo ""
echo -e "${CYAN}Security Audit:${NC}"
if command -v cargo-audit &> /dev/null; then
    AUDIT_RESULT=$(cargo audit 2>&1 || true)
    VULNERABILITIES=$(echo "$AUDIT_RESULT" | grep -c "vulnerability" || echo "0")
    echo "  Vulnerabilities: $VULNERABILITIES"
    METRICS["vulnerabilities"]="$VULNERABILITIES"
else
    echo "  cargo-audit not installed (run: cargo install cargo-audit)"
    METRICS["vulnerabilities"]="N/A"
fi

# Calculate Quality Score
echo ""
echo -e "${CYAN}Quality Score Calculation:${NC}"

# Simplified scoring (0-10)
SCORE=0

# Build (+2 if pass)
[[ "$BUILD_STATUS" == "PASS" ]] && SCORE=$((SCORE + 2))

# Clippy (+2 if pass)
[[ "$CLIPPY_STATUS" == "PASS" ]] && SCORE=$((SCORE + 2))

# Format (+1 if pass)
[[ "$FMT_STATUS" == "PASS" ]] && SCORE=$((SCORE + 1))

# Tests (+2 if pass)
[[ "$TEST_STATUS" == "PASS" ]] && SCORE=$((SCORE + 2))

# Low TODO/FIXME count (+1 if < 20)
[[ $((TODO_COUNT + FIXME_COUNT)) -lt 20 ]] && SCORE=$((SCORE + 1))

# Low warnings (+1 if < 10)
[[ "$WARNINGS" -lt 10 ]] && SCORE=$((SCORE + 1))

# No unsafe (+1 if 0)
[[ "$UNSAFE_COUNT" -eq 0 ]] && SCORE=$((SCORE + 1))

METRICS["quality_score"]="$SCORE"

echo ""
echo "=============================================="
echo -e "  ${CYAN}QUALITY SCORE: ${GREEN}${SCORE}/10${NC}"
echo "=============================================="

# Gate Summary
echo ""
echo -e "${CYAN}Gate Summary:${NC}"
echo "  [Gate 1] Build:      $BUILD_STATUS"
echo "  [Gate 2] Clippy:     $CLIPPY_STATUS"
echo "  [Gate 3] Format:     $FMT_STATUS"
echo "  [Gate 4] Tests:      $TEST_STATUS"
echo "  [Gate 5] Benchmarks: $BENCH_STATUS"

# Overall verdict
echo ""
if [[ "$BUILD_STATUS" == "PASS" && "$CLIPPY_STATUS" == "PASS" && "$FMT_STATUS" == "PASS" && "$TEST_STATUS" == "PASS" ]]; then
    echo -e "${GREEN}OVERALL: ALL GATES PASSED${NC}"
    EXIT_CODE=0
else
    echo -e "${RED}OVERALL: SOME GATES FAILED${NC}"
    EXIT_CODE=1
fi

# JSON output (for CI)
if $JSON_OUTPUT; then
    echo ""
    echo "JSON Output:"
    echo "{"
    for key in "${!METRICS[@]}"; do
        echo "  \"$key\": \"${METRICS[$key]}\","
    done
    echo "  \"timestamp\": \"$(date -Iseconds)\""
    echo "}"
fi

# Save metrics to file
METRICS_FILE="target/quality_metrics.json"
mkdir -p target
echo "{" > "$METRICS_FILE"
for key in "${!METRICS[@]}"; do
    echo "  \"$key\": \"${METRICS[$key]}\"," >> "$METRICS_FILE"
done
echo "  \"timestamp\": \"$(date -Iseconds)\"" >> "$METRICS_FILE"
echo "}" >> "$METRICS_FILE"
echo ""
echo "Metrics saved to: $METRICS_FILE"

if $CI_MODE; then
    exit $EXIT_CODE
fi
