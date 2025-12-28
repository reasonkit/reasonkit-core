#!/bin/bash
# decision_matrix.sh - Structured decision analysis using ReasonKit
#
# Usage: ./decision_matrix.sh "Decision question" "Option1" "Option2" ["Option3" ...]
#
# This script demonstrates decision support:
# 1. Analyze each option with GigaThink
# 2. Logical comparison with LaserLogic
# 3. First principles with BedRock
# 4. Final recommendation

set -e

DECISION="$1"
shift
OPTIONS=("$@")

if [[ -z "$DECISION" ]] || [[ ${#OPTIONS[@]} -lt 2 ]]; then
    echo "Usage: $0 \"Decision question\" \"Option1\" \"Option2\" [\"Option3\" ...]"
    echo ""
    echo "Example:"
    echo "  $0 \"Which database should we use?\" \"PostgreSQL\" \"MongoDB\" \"Redis\""
    exit 1
fi

OUTPUT_DIR="./decision_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    ReasonKit Decision Matrix${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Decision: $DECISION"
echo "Options: ${OPTIONS[*]}"
echo "Output: $OUTPUT_DIR"
echo ""

# Find rk-core
RK_CORE="${RK_CORE:-rk-core}"
if ! command -v "$RK_CORE" &> /dev/null; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/../../target/release/rk-core" ]]; then
        RK_CORE="$SCRIPT_DIR/../../target/release/rk-core"
    else
        echo "Error: rk-core not found. Build with: cargo build --release"
        exit 1
    fi
fi

STEP=1
TOTAL=$((${#OPTIONS[@]} + 3))

# Analyze each option
for opt in "${OPTIONS[@]}"; do
    echo -e "${GREEN}[$STEP/$TOTAL]${NC} Analyzing option: ${YELLOW}$opt${NC}"
    "$RK_CORE" think --protocol gigathink \
        "Analyze the pros and cons of '$opt' for the decision: $DECISION" \
        > "$OUTPUT_DIR/option_${opt// /_}.txt" 2>&1
    echo "      → Saved analysis"
    ((STEP++))
done

# Logical comparison
echo -e "${GREEN}[$STEP/$TOTAL]${NC} Logical comparison with LaserLogic..."
"$RK_CORE" think --protocol laserlogic \
    "Compare these options for '$DECISION': ${OPTIONS[*]}. Which is logically superior and why?" \
    > "$OUTPUT_DIR/logical_comparison.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/logical_comparison.txt"
((STEP++))

# First principles
echo -e "${GREEN}[$STEP/$TOTAL]${NC} First principles analysis with BedRock..."
"$RK_CORE" think --protocol bedrock \
    "What are the fundamental factors that should drive the decision: $DECISION?" \
    > "$OUTPUT_DIR/first_principles.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/first_principles.txt"
((STEP++))

# Final recommendation
echo -e "${GREEN}[$STEP/$TOTAL]${NC} Generating recommendation..."
"$RK_CORE" think --profile deep \
    "Based on thorough analysis, recommend the best option for '$DECISION' from: ${OPTIONS[*]}. Provide clear reasoning." \
    > "$OUTPUT_DIR/recommendation.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/recommendation.txt"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    Decision Analysis Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"
