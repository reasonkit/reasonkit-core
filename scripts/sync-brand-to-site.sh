#!/bin/bash
# =============================================================================
# ReasonKit Brand Asset Sync Script
# =============================================================================
# Syncs brand assets from reasonkit-core/brand/ to reasonkit-site/assets/brand/
#
# Usage:
#   ./scripts/sync-brand-to-site.sh [OPTIONS]
#
# Options:
#   --dry-run     Show what would be synced without making changes
#   --verbose     Show detailed output
#   --help        Show this help message
#
# This script ensures reasonkit-site always has the latest brand assets
# from the canonical source in reasonkit-core/brand/
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$CORE_ROOT")"

SOURCE_DIR="$CORE_ROOT/brand"
TARGET_DIR="$PROJECT_ROOT/reasonkit-site/assets/brand"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
DRY_RUN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
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
echo "║              ReasonKit Brand Asset Sync                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Verify source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}ERROR: Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

# Verify target parent exists
TARGET_PARENT="$(dirname "$TARGET_DIR")"
if [ ! -d "$TARGET_PARENT" ]; then
    echo -e "${RED}ERROR: Target parent directory not found: $TARGET_PARENT${NC}"
    echo "Is reasonkit-site checked out at $PROJECT_ROOT/reasonkit-site?"
    exit 1
fi

# Show configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Source: $SOURCE_DIR"
echo "  Target: $TARGET_DIR"
echo "  Mode:   $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'LIVE')"
echo ""

# Count source assets
SOURCE_COUNT=$(find "$SOURCE_DIR" -type f \( -name "*.svg" -o -name "*.png" -o -name "*.md" \) | wc -l)
echo -e "${CYAN}Source assets: $SOURCE_COUNT files${NC}"
echo ""

# Build rsync options
RSYNC_OPTS="-av --delete"
[ "$DRY_RUN" = true ] && RSYNC_OPTS="$RSYNC_OPTS --dry-run"
[ "$VERBOSE" = true ] && RSYNC_OPTS="$RSYNC_OPTS --itemize-changes"

# Directories to sync (selective sync for website needs)
SYNC_DIRS=(
    "logos"
    "favicons"
    "banners"
    "thinktools"
    "badges"
    "patterns"
    "diagrams"
)

# Create target directory if needed
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$TARGET_DIR"
fi

# Sync each directory
echo -e "${YELLOW}Syncing directories:${NC}"
for dir in "${SYNC_DIRS[@]}"; do
    if [ -d "$SOURCE_DIR/$dir" ]; then
        echo -e "  ${GREEN}→${NC} $dir/"
        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$TARGET_DIR/$dir"
        fi
        rsync $RSYNC_OPTS "$SOURCE_DIR/$dir/" "$TARGET_DIR/$dir/" 2>/dev/null || true
    else
        echo -e "  ${YELLOW}⚠${NC} $dir/ (not found, skipping)"
    fi
done

# Also sync key documentation
echo ""
echo -e "${YELLOW}Syncing documentation:${NC}"
if [ -f "$SOURCE_DIR/BRAND_PLAYBOOK.md" ]; then
    echo -e "  ${GREEN}→${NC} BRAND_PLAYBOOK.md"
    if [ "$DRY_RUN" = false ]; then
        cp "$SOURCE_DIR/BRAND_PLAYBOOK.md" "$TARGET_DIR/"
    fi
fi

echo ""

# Count synced assets
if [ "$DRY_RUN" = false ] && [ -d "$TARGET_DIR" ]; then
    TARGET_COUNT=$(find "$TARGET_DIR" -type f \( -name "*.svg" -o -name "*.png" -o -name "*.md" \) | wc -l)
    echo -e "${GREEN}✓ Sync complete: $TARGET_COUNT files in target${NC}"
else
    echo -e "${YELLOW}Dry run complete - no changes made${NC}"
fi

echo ""
echo -e "${CYAN}Brand assets synchronized from:${NC}"
echo "  reasonkit-core/brand/ → reasonkit-site/assets/brand/"
echo ""
echo -e "${GREEN}Done!${NC}"
