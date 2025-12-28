#!/bin/bash
# install-rk-commands.sh - Install ReasonKit CLI commands
#
# This script installs rk-web, rk-think, and related shortcuts
# to make them available system-wide.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                ReasonKit CLI Commands Installation${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Find rk-core binary (check both workspace and project level)
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
if [[ -f "$WORKSPACE_ROOT/target/release/rk-core" ]]; then
    RK_CORE_BIN="$WORKSPACE_ROOT/target/release/rk-core"
elif [[ -f "$PROJECT_ROOT/target/release/rk-core" ]]; then
    RK_CORE_BIN="$PROJECT_ROOT/target/release/rk-core"
else
    echo "Building rk-core in release mode..."
    cd "$PROJECT_ROOT"
    cargo build --release
    # Check both locations again
    if [[ -f "$WORKSPACE_ROOT/target/release/rk-core" ]]; then
        RK_CORE_BIN="$WORKSPACE_ROOT/target/release/rk-core"
    else
        RK_CORE_BIN="$PROJECT_ROOT/target/release/rk-core"
    fi
fi
echo "Found rk-core at: $RK_CORE_BIN"

# Determine installation directory
if [[ -d "$HOME/.local/bin" ]]; then
    INSTALL_DIR="$HOME/.local/bin"
elif [[ -d "$HOME/bin" ]]; then
    INSTALL_DIR="$HOME/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
fi

echo "Installation directory: $INSTALL_DIR"
echo ""

# Install rk-core binary
echo -n "Installing rk-core... "
cp "$RK_CORE_BIN" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/rk-core"
echo -e "${GREEN}done${NC}"

# Install wrapper scripts
SCRIPTS=(rk-web rk-web rk-think rk-gt rk-ll rk-br rk-pg rk-bh)
for script in "${SCRIPTS[@]}"; do
    echo -n "Installing $script... "
    if [[ -L "$SCRIPT_DIR/$script" ]]; then
        # It's a symlink, resolve and copy
        target=$(readlink "$SCRIPT_DIR/$script")
        cp "$SCRIPT_DIR/$target" "$INSTALL_DIR/$script"
    else
        cp "$SCRIPT_DIR/$script" "$INSTALL_DIR/"
    fi
    chmod +x "$INSTALL_DIR/$script"
    echo -e "${GREEN}done${NC}"
done

echo ""

# Check if INSTALL_DIR is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo -e "${RED}Warning: $INSTALL_DIR is not in your PATH${NC}"
    echo ""
    echo "Add this to your shell profile (~/.bashrc or ~/.zshrc):"
    echo ""
    echo -e "  ${CYAN}export PATH=\"\$PATH:$INSTALL_DIR\"${NC}"
    echo ""
fi

echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    Installation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Available commands:"
echo ""
echo "  rk-web   - Web research with web + KB + reasoning"
echo "  rk-web  - Deprecated alias for rk-web"
echo "  rk-think - ThinkTools structured reasoning"
echo "  rk-gt    - GigaThink (multi-perspective expansion)"
echo "  rk-ll    - LaserLogic (precision deduction)"
echo "  rk-br    - BedRock (first principles)"
echo "  rk-pg    - ProofGuard (verification)"
echo "  rk-bh    - BrutalHonesty (self-critique)"
echo ""
echo "Quick start:"
echo ""
echo "  rk-web \"What is chain-of-thought prompting?\""
echo "  rk-think --quick \"Analyze this approach\""
echo "  rk-gt \"Generate 10 perspectives on AI safety\""
echo ""
echo "For help: rk-web --help | rk-think --help"
