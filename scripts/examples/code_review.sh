#!/bin/bash
# code_review.sh - Structured code review using ReasonKit
#
# Usage: ./code_review.sh <file_or_directory>
#
# This script demonstrates code review:
# 1. Security analysis (paranoid mode)
# 2. Logic validation (LaserLogic)
# 3. Adversarial critique (BrutalHonesty)
# 4. Summary with recommendations

set -e

TARGET="$1"
if [[ -z "$TARGET" ]]; then
    echo "Usage: $0 <file_or_directory>"
    exit 1
fi

if [[ ! -e "$TARGET" ]]; then
    echo "Error: $TARGET does not exist"
    exit 1
fi

OUTPUT_DIR="./review_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    ReasonKit Code Review${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Target: $TARGET"
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

# Get code content
if [[ -d "$TARGET" ]]; then
    CODE=$(find "$TARGET" -type f \( -name "*.rs" -o -name "*.py" -o -name "*.js" -o -name "*.ts" \) -exec cat {} \; | head -500)
else
    CODE=$(cat "$TARGET" | head -500)
fi

if [[ -z "$CODE" ]]; then
    echo "Error: No code found in $TARGET"
    exit 1
fi

# Step 1: Security analysis
echo -e "${GREEN}[1/4]${NC} ${RED}Security Analysis${NC} (paranoid mode)..."
"$RK_CORE" think --profile paranoid \
    "Perform a security review of this code. Look for vulnerabilities, injection risks, authentication issues, and OWASP top 10 issues:

$CODE" \
    > "$OUTPUT_DIR/01_security.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/01_security.txt"

# Step 2: Logic validation
echo -e "${GREEN}[2/4]${NC} Logic validation with LaserLogic..."
"$RK_CORE" think --protocol laserlogic \
    "Validate the logical correctness of this code. Check for edge cases, off-by-one errors, null handling, and logical flaws:

$CODE" \
    > "$OUTPUT_DIR/02_logic.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/02_logic.txt"

# Step 3: Adversarial critique
echo -e "${GREEN}[3/4]${NC} Adversarial critique with BrutalHonesty..."
"$RK_CORE" think --protocol brutalhonesty \
    "Be brutally honest about this code. Find all bugs, antipatterns, technical debt, and issues:

$CODE" \
    > "$OUTPUT_DIR/03_critique.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/03_critique.txt"

# Step 4: Summary
echo -e "${GREEN}[4/4]${NC} Generating summary..."
"$RK_CORE" think --profile balanced \
    "Summarize the code review findings and provide prioritized recommendations for improvement:

$CODE" \
    > "$OUTPUT_DIR/04_summary.txt" 2>&1
echo "      → Saved to $OUTPUT_DIR/04_summary.txt"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    Code Review Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"
