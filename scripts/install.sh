#!/bin/bash
# ReasonKit One-Liner Installer
# Install: curl -fsSL https://reasonkit.sh/install | bash
#
# This script installs ReasonKit CLI and ThinkTools for AI reasoning enhancement.
#
# What gets installed:
#   - rk-core    : Main CLI binary
#   - rk-think   : ThinkTools structured reasoning
#   - rk-web     : Web research with triangulation
#   - rk-verify  : Claim verification (3-source rule)
#   - rk-gt      : GigaThink (multi-perspective expansion)
#   - rk-ll      : LaserLogic (precision deduction)
#   - rk-br      : BedRock (first principles)
#   - rk-pg      : ProofGuard (verification)
#   - rk-bh      : BrutalHonesty (self-critique)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}       ██████╗ ███████╗ █████╗ ███████╗ ██████╗ ███╗   ██╗██╗  ██╗██╗████████╗${NC}"
echo -e "${CYAN}       ██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║██║ ██╔╝██║╚══██╔══╝${NC}"
echo -e "${CYAN}       ██████╔╝█████╗  ███████║███████╗██║   ██║██╔██╗ ██║█████╔╝ ██║   ██║   ${NC}"
echo -e "${CYAN}       ██╔══██╗██╔══╝  ██╔══██║╚════██║██║   ██║██║╚██╗██║██╔═██╗ ██║   ██║   ${NC}"
echo -e "${CYAN}       ██║  ██║███████╗██║  ██║███████║╚██████╔╝██║ ╚████║██║  ██╗██║   ██║   ${NC}"
echo -e "${CYAN}       ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝   ╚═╝   ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}       AI Thinking Enhancement System - Turn Prompts into Protocols${NC}"
echo ""

# Detect OS and Architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)     OS_NAME="linux";;
    Darwin*)    OS_NAME="macos";;
    CYGWIN*|MINGW*|MSYS*) OS_NAME="windows";;
    *)          echo -e "${RED}Unsupported OS: $OS${NC}"; exit 1;;
esac

case "$ARCH" in
    x86_64|amd64)  ARCH_NAME="x86_64";;
    aarch64|arm64) ARCH_NAME="aarch64";;
    *)             echo -e "${RED}Unsupported architecture: $ARCH${NC}"; exit 1;;
esac

echo -e "Detected: ${GREEN}$OS_NAME-$ARCH_NAME${NC}"
echo ""

# Determine installation directory
if [[ -w "/usr/local/bin" ]]; then
    INSTALL_DIR="/usr/local/bin"
elif [[ -d "$HOME/.local/bin" ]]; then
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
elif [[ -d "$HOME/bin" ]]; then
    INSTALL_DIR="$HOME/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
fi

echo -e "Install directory: ${GREEN}$INSTALL_DIR${NC}"
echo ""

# Check for Rust/Cargo (required for source build)
BINARY_AVAILABLE=false
GITHUB_RELEASE_URL="https://github.com/reasonkit/reasonkit-core/releases/latest/download"

# Try to download pre-built binary first
BINARY_NAME="rk-core-${OS_NAME}-${ARCH_NAME}"
if [[ "$OS_NAME" == "windows" ]]; then
    BINARY_NAME="${BINARY_NAME}.exe"
fi

echo -e "${CYAN}Checking for pre-built binary...${NC}"

# For now, build from source (pre-built binaries coming soon)
if command -v cargo &> /dev/null; then
    echo -e "${GREEN}Cargo found. Building from source...${NC}"
    echo ""

    # Clone or update repo
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    echo -e "${CYAN}Cloning ReasonKit repository...${NC}"
    git clone --depth 1 https://github.com/reasonkit/reasonkit-core.git 2>/dev/null || {
        # If clone fails, try direct cargo install
        echo -e "${YELLOW}GitHub clone failed. Trying cargo install...${NC}"
        cargo install reasonkit --root "$HOME/.cargo" 2>/dev/null || {
            echo -e "${RED}Cargo install failed. Please install from source:${NC}"
            echo ""
            echo "    git clone https://github.com/reasonkit/reasonkit-core.git"
            echo "    cd reasonkit-core"
            echo "    cargo build --release"
            echo "    ./scripts/install-rk-commands.sh"
            exit 1
        }
    }

    if [[ -d "reasonkit-core" ]]; then
        cd reasonkit-core

        echo -e "${CYAN}Building ReasonKit (this may take a few minutes)...${NC}"
        cargo build --release

        # Install binaries
        echo ""
        echo -e "${CYAN}Installing binaries...${NC}"

        # Main binary
        cp target/release/rk-core "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/rk-core"
        echo -e "  ${GREEN}rk-core${NC} installed"

        # Wrapper scripts
        if [[ -f scripts/rk-think ]]; then
            cp scripts/rk-think "$INSTALL_DIR/"
            chmod +x "$INSTALL_DIR/rk-think"
            echo -e "  ${GREEN}rk-think${NC} installed"
        fi

        if [[ -f scripts/rk-web ]]; then
            cp scripts/rk-web "$INSTALL_DIR/"
            chmod +x "$INSTALL_DIR/rk-web"
            echo -e "  ${GREEN}rk-web${NC} installed"
        fi

        if [[ -f scripts/rk-dive ]]; then
            cp scripts/rk-dive "$INSTALL_DIR/"
            chmod +x "$INSTALL_DIR/rk-dive"
            echo -e "  ${GREEN}rk-dive${NC} installed (alias)"
        fi

        # Protocol shortcuts (symlinks to rk-think)
        for cmd in rk-gt rk-ll rk-br rk-pg rk-bh; do
            ln -sf "$INSTALL_DIR/rk-think" "$INSTALL_DIR/$cmd" 2>/dev/null || \
            cp "$INSTALL_DIR/rk-think" "$INSTALL_DIR/$cmd"
            echo -e "  ${GREEN}$cmd${NC} installed"
        done
    fi

    # Cleanup
    rm -rf "$TEMP_DIR"

else
    echo -e "${RED}Cargo not found.${NC}"
    echo ""
    echo "Please install Rust first:"
    echo ""
    echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo ""
    echo "Then run this installer again."
    exit 1
fi

# Check if INSTALL_DIR is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo ""
    echo -e "${YELLOW}Note: $INSTALL_DIR is not in your PATH${NC}"
    echo ""
    echo "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    echo ""
    echo -e "    ${CYAN}export PATH=\"\$PATH:$INSTALL_DIR\"${NC}"
    echo ""
    echo "Then restart your shell or run:"
    echo ""
    echo -e "    ${CYAN}source ~/.bashrc${NC}  # or source ~/.zshrc"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    Installation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}THINKTOOLS - Structured Reasoning Modules:${NC}"
echo ""
echo "  rk-think   Execute reasoning protocols"
echo "    --quick     Fast 3-step analysis (GigaThink + LaserLogic)"
echo "    --balanced  Standard 5-module chain"
echo "    --deep      Thorough analysis"
echo "    --paranoid  Maximum verification"
echo ""
echo "  rk-gt      GigaThink   - Multi-perspective expansion (10+ viewpoints)"
echo "  rk-ll      LaserLogic  - Precision deductive reasoning"
echo "  rk-br      BedRock     - First principles decomposition"
echo "  rk-pg      ProofGuard  - Multi-source verification"
echo "  rk-bh      BrutalHonesty - Adversarial self-critique"
echo ""
echo -e "${CYAN}WEB RESEARCH:${NC}"
echo ""
echo "  rk-web \"query\"           Standard research"
echo "  rk-web --deep \"query\"    Thorough with triangulation"
echo ""
echo -e "${CYAN}CLAIM VERIFICATION (Triangulated Truth):${NC}"
echo ""
echo "  rk-core verify \"claim\"    Verify with 3+ sources"
echo ""
echo -e "${CYAN}Quick Start:${NC}"
echo ""
echo "  rk-think \"What are the key factors for X?\""
echo "  rk-gt \"Generate 10 perspectives on AI safety\""
echo "  rk-web --deep \"Compare RAPTOR vs ColBERT for RAG\""
echo ""
echo -e "Documentation: ${CYAN}https://reasonkit.sh/docs${NC}"
echo ""
