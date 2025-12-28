#!/usr/bin/env bash
# ReasonKit Core - Local Quality Check Script
# Run all 5 quality gates locally before pushing
# Reference: ORCHESTRATOR.md CONS-009

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         ReasonKit Core - Quality Gates Check                 ║${NC}"
echo -e "${BLUE}║         5 Mandatory Gates (CONS-009)                          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

GATES_PASSED=0
GATES_FAILED=0

# ═══════════════════════════════════════════════════════════════════════════
# GATE 1: Build Check
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[GATE 1]${NC} Building project (release)..."
if cargo build --release; then
    echo -e "${GREEN}✅ Gate 1: Build PASSED${NC}"
    GATES_PASSED=$((GATES_PASSED + 1))
else
    echo -e "${RED}❌ Gate 1: Build FAILED${NC}"
    GATES_FAILED=$((GATES_FAILED + 1))
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# GATE 2: Lint with Clippy
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[GATE 2]${NC} Running Clippy (lint)..."
if cargo clippy -- -D warnings; then
    echo -e "${GREEN}✅ Gate 2: Clippy PASSED${NC}"
    GATES_PASSED=$((GATES_PASSED + 1))
else
    echo -e "${RED}❌ Gate 2: Clippy FAILED${NC}"
    GATES_FAILED=$((GATES_FAILED + 1))
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# GATE 3: Format Check
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[GATE 3]${NC} Checking code formatting..."
if cargo fmt --check; then
    echo -e "${GREEN}✅ Gate 3: Format PASSED${NC}"
    GATES_PASSED=$((GATES_PASSED + 1))
else
    echo -e "${RED}❌ Gate 3: Format FAILED${NC}"
    echo -e "${YELLOW}💡 Tip: Run 'cargo fmt' to auto-fix formatting${NC}"
    GATES_FAILED=$((GATES_FAILED + 1))
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# GATE 4: Tests
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[GATE 4]${NC} Running tests..."
if cargo test --all-features; then
    echo -e "${GREEN}✅ Gate 4: Tests PASSED${NC}"
    GATES_PASSED=$((GATES_PASSED + 1))
else
    echo -e "${RED}❌ Gate 4: Tests FAILED${NC}"
    GATES_FAILED=$((GATES_FAILED + 1))
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# GATE 5: Benchmarks (optional, non-blocking)
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[GATE 5]${NC} Running benchmarks (optional)..."
if cargo bench --bench retrieval_bench -- --noplot 2>/dev/null; then
    echo -e "${GREEN}✅ Gate 5: Benchmarks PASSED${NC}"
else
    echo -e "${YELLOW}⚠️  Gate 5: Benchmarks incomplete (non-blocking)${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Additional Quality Checks
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[BONUS]${NC} Additional quality checks..."

# Security audit
echo -n "Security audit: "
if command -v cargo-audit &> /dev/null; then
    if cargo audit 2>/dev/null; then
        echo -e "${GREEN}✅ No vulnerabilities${NC}"
    else
        echo -e "${YELLOW}⚠️  Vulnerabilities found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  cargo-audit not installed${NC}"
fi

# Code metrics
echo ""
echo -e "${BLUE}Code Metrics:${NC}"
echo "  Lines of Rust: $(find src -name '*.rs' -exec cat {} \; | wc -l)"
echo "  TODO count: $(grep -r 'TODO' src --include='*.rs' 2>/dev/null | wc -l || echo 0)"
echo "  FIXME count: $(grep -r 'FIXME' src --include='*.rs' 2>/dev/null | wc -l || echo 0)"
echo "  Unsafe blocks: $(grep -r 'unsafe' src --include='*.rs' 2>/dev/null | wc -l || echo 0)"

if [ -f target/release/rk-core ]; then
    BINARY_SIZE=$(ls -lh target/release/rk-core | awk '{print $5}')
    echo "  Binary size (release): $BINARY_SIZE"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     Quality Gates Summary                     ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Gates Passed: $GATES_PASSED / 4 (required)"
echo "  Gates Failed: $GATES_FAILED / 4"
echo ""

if [ $GATES_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL QUALITY GATES PASSED! 🎉${NC}"
    echo ""
    echo "Your code is ready to commit and push."
    echo ""
    echo "Next steps:"
    echo "  1. git add ."
    echo "  2. git commit -m 'your message'"
    echo "  3. git push"
    exit 0
else
    echo -e "${RED}❌ $GATES_FAILED QUALITY GATE(S) FAILED${NC}"
    echo ""
    echo "Please fix the issues above before committing."
    echo ""
    echo "Common fixes:"
    echo "  - cargo fmt                  # Auto-fix formatting"
    echo "  - cargo clippy --fix         # Auto-fix some Clippy issues"
    echo "  - cargo test                 # Run tests to see failures"
    exit 1
fi
