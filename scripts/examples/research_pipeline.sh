#!/bin/bash
# research_pipeline.sh - Complete research pipeline using ReasonKit
#
# Usage: ./research_pipeline.sh "Your research topic"
#
# This script demonstrates a full research workflow:
# 1. Brainstorm angles (GigaThink)
# 2. Deep research with web sources
# 3. Verify key claims (ProofGuard)
# 4. Generate summary

set -e

TOPIC="$1"
if [[ -z "$TOPIC" ]]; then
    echo "Usage: $0 \"Your research topic\""
    exit 1
fi

OUTPUT_DIR="./research_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    ReasonKit Research Pipeline${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Topic: $TOPIC"
echo "Output: $OUTPUT_DIR"
echo ""

# Find rk
RK_CORE="${RK_CORE:-rk}"
if ! command -v "$RK_CORE" &> /dev/null; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/../../target/release/rk" ]]; then
        RK_CORE="$SCRIPT_DIR/../../target/release/rk"
    else
        echo "Error: rk not found. Build with: cargo build --release"
        exit 1
    fi
fi

# Step 1: Brainstorm angles
echo -e "${GREEN}[1/4]${NC} Brainstorming research angles with GigaThink..."
"$RK_CORE" think --protocol gigathink "$TOPIC" > "$OUTPUT_DIR/01_perspectives.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/01_perspectives.txt"

# Step 2: Deep research
echo -e "${GREEN}[2/4]${NC} Deep research with rk-web..."
"$RK_CORE" web --depth deep "$TOPIC" --web > "$OUTPUT_DIR/02_research.md" 2>&1 || true
echo "      → Saved to $OUTPUT_DIR/02_research.md"

# Step 3: Verify claims
echo -e "${GREEN}[3/4]${NC} Verifying findings with ProofGuard..."
"$RK_CORE" think --profile paranoid "Verify the key findings and claims about: $TOPIC" \
    > "$OUTPUT_DIR/03_verification.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/03_verification.txt"

# Step 4: Summary
echo -e "${GREEN}[4/4]${NC} Generating summary..."
"$RK_CORE" think --profile balanced "Summarize the key findings and recommendations for: $TOPIC" \
    > "$OUTPUT_DIR/04_summary.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/04_summary.txt"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    Research Pipeline Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"
