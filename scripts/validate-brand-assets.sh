#!/bin/bash
# =============================================================================
# ReasonKit Brand Asset Validation Script
# =============================================================================
# Validates brand assets for completeness and consistency.
# Run this in CI to ensure brand integrity.
#
# Usage:
#   ./scripts/validate-brand-assets.sh [OPTIONS]
#
# Options:
#   --verbose     Show detailed output
#   --strict      Fail on warnings (for CI)
#   --help        Show this help message
#
# Exit codes:
#   0 - All validations passed
#   1 - Critical validation failures
#   2 - Warnings only (with --strict, treated as failure)
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_ROOT="$(dirname "$SCRIPT_DIR")"
BRAND_DIR="$CORE_ROOT/brand"
README_FILE="$CORE_ROOT/README.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
VERBOSE=false
STRICT=false

# Counters
ERRORS=0
WARNINGS=0
PASSES=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        --help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║             ReasonKit Brand Asset Validation                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    ((PASSES++)) || true
}

warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    ((WARNINGS++)) || true
}

fail() {
    echo -e "  ${RED}✗${NC} $1"
    ((ERRORS++)) || true
}

verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "    ${CYAN}→${NC} $1"
    fi
}

# -----------------------------------------------------------------------------
# Validation: Required directories exist
# -----------------------------------------------------------------------------

echo -e "${YELLOW}Checking required directories...${NC}"

REQUIRED_DIRS=(
    "logos"
    "banners"
    "badges"
    "diagrams"
    "thinktools"
    "favicons"
    "readme"
    "avatars"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$BRAND_DIR/$dir" ]; then
        file_count=$(find "$BRAND_DIR/$dir" -type f | wc -l | tr -d ' ')
        pass "$dir/ exists ($file_count files)"
    else
        fail "$dir/ missing"
    fi
done

# -----------------------------------------------------------------------------
# Validation: Required core files exist
# -----------------------------------------------------------------------------

echo ""
echo -e "${YELLOW}Checking required core files...${NC}"

REQUIRED_FILES=(
    "BRAND_PLAYBOOK.md"
    "logos/logo-icon.svg"
    "logos/logo-full.svg"
    "logos/logo-wordmark.svg"
    "favicons/favicon.svg"
    "favicons/favicon-32.png"
    "banners/banner-og-image.svg"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$BRAND_DIR/$file" ]; then
        pass "$file exists"
    else
        fail "$file missing"
    fi
done

# -----------------------------------------------------------------------------
# Validation: README image paths are valid
# -----------------------------------------------------------------------------

echo ""
echo -e "${YELLOW}Validating README image paths...${NC}"

if [ -f "$README_FILE" ]; then
    # Extract image paths from README (both markdown and HTML)
    while IFS= read -r line; do
        # Match ./brand/ paths
        if [[ "$line" =~ src=\"\.\/brand\/([^\"]+)\" ]]; then
            img_path="${BASH_REMATCH[1]}"
            if [ -f "$BRAND_DIR/$img_path" ]; then
                pass "README: brand/$img_path exists"
            else
                fail "README: brand/$img_path NOT FOUND"
            fi
        fi
    done < "$README_FILE"
else
    fail "README.md not found"
fi

# -----------------------------------------------------------------------------
# Validation: SVG brand colors
# -----------------------------------------------------------------------------

echo ""
echo -e "${YELLOW}Checking brand color usage in SVGs...${NC}"

BRAND_COLORS=(
    "#06b6d4"  # Cyan (Primary)
    "#a855f7"  # Purple
    "#ec4899"  # Pink
    "#10b981"  # Green
    "#f97316"  # Orange
    "#fbbf24"  # Yellow
)

# Check that at least some SVGs use brand colors
svg_count=$(find "$BRAND_DIR" -name "*.svg" | wc -l | tr -d ' ')
if [ "$svg_count" -gt 0 ]; then
    pass "Found $svg_count SVG files"

    # Sample check: logos should use brand cyan
    if [ -f "$BRAND_DIR/logos/logo-icon.svg" ]; then
        if grep -qi "#06b6d4\|#00d2ff" "$BRAND_DIR/logos/logo-icon.svg" 2>/dev/null; then
            pass "Logo uses brand cyan color"
        else
            warn "Logo may not use brand cyan color"
        fi
    fi
else
    warn "No SVG files found in brand/"
fi

# -----------------------------------------------------------------------------
# Validation: Asset sizes
# -----------------------------------------------------------------------------

echo ""
echo -e "${YELLOW}Checking asset sizes...${NC}"

# Check for oversized PNGs (> 3MB is suspicious)
large_files=0
while IFS= read -r file; do
    size=$(stat --format=%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    if [ "$size" -gt 3145728 ]; then
        warn "Large file (>3MB): $file ($(numfmt --to=iec $size 2>/dev/null || echo "${size}B"))"
        ((large_files++)) || true
    fi
done < <(find "$BRAND_DIR" -name "*.png" 2>/dev/null)

if [ "$large_files" -eq 0 ]; then
    pass "No oversized PNG files (>3MB)"
fi

# Check total brand directory size
total_size=$(du -sb "$BRAND_DIR" | cut -f1)
total_human=$(numfmt --to=iec $total_size 2>/dev/null || echo "${total_size}B")
if [ "$total_size" -lt 52428800 ]; then  # 50MB threshold
    pass "Total brand size: $total_human (under 50MB)"
else
    warn "Total brand size: $total_human (consider optimizing)"
fi

# -----------------------------------------------------------------------------
# Validation: Sync script exists
# -----------------------------------------------------------------------------

echo ""
echo -e "${YELLOW}Checking infrastructure...${NC}"

if [ -f "$SCRIPT_DIR/sync-brand-to-site.sh" ]; then
    pass "sync-brand-to-site.sh exists"
    if [ -x "$SCRIPT_DIR/sync-brand-to-site.sh" ]; then
        pass "sync-brand-to-site.sh is executable"
    else
        warn "sync-brand-to-site.sh not executable"
    fi
else
    fail "sync-brand-to-site.sh missing"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                         SUMMARY                                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASSES"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "  ${RED}Errors:${NC}   $ERRORS"
echo ""

# Determine exit code
if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}VALIDATION FAILED: $ERRORS errors found${NC}"
    exit 1
elif [ "$WARNINGS" -gt 0 ] && [ "$STRICT" = true ]; then
    echo -e "${YELLOW}VALIDATION FAILED (strict mode): $WARNINGS warnings found${NC}"
    exit 2
elif [ "$WARNINGS" -gt 0 ]; then
    echo -e "${YELLOW}VALIDATION PASSED WITH WARNINGS${NC}"
    exit 0
else
    echo -e "${GREEN}VALIDATION PASSED${NC}"
    exit 0
fi
