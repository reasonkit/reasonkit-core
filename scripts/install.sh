#!/bin/bash
# ReasonKit Universal Installer
# Install: curl -fsSL https://reasonkit.sh/install | bash
#
# This script:
#   1. Detects your platform (macOS/Linux, x86_64/arm64)
#   2. Installs ReasonKit via the best available method
#   3. Validates the installation
#   4. Shows you how to get started immediately
#
# Supports: macOS (Intel + Apple Silicon), Linux (x86_64, aarch64), WSL2

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION="0.1.0"
GITHUB_REPO="reasonkit/reasonkit-core"
GITHUB_RELEASE_URL="https://github.com/${GITHUB_REPO}/releases/latest/download"

# Colors (with fallback for non-color terminals)
if [ -t 1 ] && [ -n "$TERM" ] && [ "$TERM" != "dumb" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    CYAN='\033[0;36m'
    YELLOW='\033[0;33m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    CYAN=''
    YELLOW=''
    BOLD=''
    NC=''
fi

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_banner() {
    echo ""
    echo -e "${CYAN}${BOLD}"
    echo "  ____                            _  ___ _   "
    echo " |  _ \ ___  __ _ ___  ___  _ __ | |/ (_) |_ "
    echo " | |_) / _ \/ _\` / __|/ _ \| '_ \| ' /| | __|"
    echo " |  _ <  __/ (_| \__ \ (_) | | | | . \| | |_ "
    echo " |_| \_\___|\__,_|___/\___/|_| |_|_|\_\_|\__|"
    echo ""
    echo -e "${NC}${CYAN}  Turn Prompts into Protocols${NC}"
    echo -e "${CYAN}  https://reasonkit.sh${NC}"
    echo ""
}

info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

fatal() {
    error "$1"
    echo ""
    echo "For help, see: https://reasonkit.sh/docs/installation-troubleshooting"
    exit 1
}

# ============================================================================
# PLATFORM DETECTION
# ============================================================================

detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    
    case "$OS" in
        Linux*)     OS_NAME="linux";;
        Darwin*)    OS_NAME="macos";;
        CYGWIN*|MINGW*|MSYS*) 
            OS_NAME="windows"
            warn "Windows detected. For best experience, use WSL2."
            ;;
        *)          fatal "Unsupported OS: $OS";;
    esac

    case "$ARCH" in
        x86_64|amd64)   ARCH_NAME="x86_64";;
        aarch64|arm64)  ARCH_NAME="aarch64";;
        *)              fatal "Unsupported architecture: $ARCH";;
    esac

    success "Detected: ${OS_NAME}-${ARCH_NAME}"
}

# ============================================================================
# INSTALLATION DIRECTORY
# ============================================================================

determine_install_dir() {
    # Priority: /usr/local/bin (if writable) > ~/.local/bin > ~/.cargo/bin
    if [ -w "/usr/local/bin" ]; then
        INSTALL_DIR="/usr/local/bin"
    elif [ -d "$HOME/.local/bin" ]; then
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
    elif [ -d "$HOME/.cargo/bin" ]; then
        INSTALL_DIR="$HOME/.cargo/bin"
    else
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
    fi

    success "Install directory: $INSTALL_DIR"
}

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

check_dependencies() {
    info "Checking dependencies..."
    
    MISSING_DEPS=""
    
    # Check for git (required for source build)
    if ! command -v git &> /dev/null; then
        MISSING_DEPS="$MISSING_DEPS git"
    fi
    
    # Check for curl or wget
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        MISSING_DEPS="$MISSING_DEPS curl"
    fi
    
    if [ -n "$MISSING_DEPS" ]; then
        warn "Missing dependencies:$MISSING_DEPS"
        echo ""
        echo "Install them with:"
        case "$OS_NAME" in
            linux)
                echo "  sudo apt install$MISSING_DEPS   # Debian/Ubuntu"
                echo "  sudo dnf install$MISSING_DEPS   # Fedora"
                echo "  sudo pacman -S$MISSING_DEPS     # Arch"
                ;;
            macos)
                echo "  brew install$MISSING_DEPS"
                ;;
        esac
        echo ""
    fi
}

check_rust() {
    if command -v cargo &> /dev/null; then
        RUST_VERSION=$(rustc --version 2>/dev/null | cut -d' ' -f2)
        success "Rust found: $RUST_VERSION"
        HAS_RUST=true
    else
        HAS_RUST=false
        warn "Rust not found. Will attempt to install from source if pre-built binary unavailable."
    fi
}

# ============================================================================
# INSTALLATION METHODS
# ============================================================================

try_prebuilt_binary() {
    info "Checking for pre-built binary..."
    
    BINARY_NAME="rk-core-${OS_NAME}-${ARCH_NAME}"
    if [ "$OS_NAME" = "windows" ]; then
        BINARY_NAME="${BINARY_NAME}.exe"
    fi
    
    DOWNLOAD_URL="${GITHUB_RELEASE_URL}/${BINARY_NAME}"
    
    # Check if binary exists
    if command -v curl &> /dev/null; then
        HTTP_CODE=$(curl -sI -o /dev/null -w "%{http_code}" "$DOWNLOAD_URL" 2>/dev/null || echo "000")
    elif command -v wget &> /dev/null; then
        HTTP_CODE=$(wget --spider -S "$DOWNLOAD_URL" 2>&1 | grep "HTTP/" | tail -1 | awk '{print $2}' || echo "000")
    else
        HTTP_CODE="000"
    fi
    
    if [ "$HTTP_CODE" = "200" ]; then
        info "Downloading pre-built binary..."
        
        if command -v curl &> /dev/null; then
            curl -fsSL "$DOWNLOAD_URL" -o "$INSTALL_DIR/rk-core"
        else
            wget -q "$DOWNLOAD_URL" -O "$INSTALL_DIR/rk-core"
        fi
        
        chmod +x "$INSTALL_DIR/rk-core"
        success "Pre-built binary installed"
        return 0
    else
        info "Pre-built binary not available for ${OS_NAME}-${ARCH_NAME}"
        return 1
    fi
}

install_via_cargo() {
    if [ "$HAS_RUST" != "true" ]; then
        info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
        success "Rust installed"
    fi
    
    info "Installing ReasonKit via cargo (this may take 2-5 minutes)..."
    
    if cargo install reasonkit-core 2>/dev/null; then
        success "ReasonKit installed via cargo"
        
        # Copy to install dir if cargo bin is different
        if [ "$INSTALL_DIR" != "$HOME/.cargo/bin" ]; then
            cp "$HOME/.cargo/bin/rk-core" "$INSTALL_DIR/rk-core" 2>/dev/null || true
        fi
        return 0
    else
        warn "cargo install failed, trying source build..."
        return 1
    fi
}

install_from_source() {
    info "Building from source..."
    
    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    cd "$TEMP_DIR"
    
    info "Cloning repository..."
    if ! git clone --depth 1 "https://github.com/${GITHUB_REPO}.git" 2>/dev/null; then
        fatal "Failed to clone repository. Check your internet connection."
    fi
    
    cd reasonkit-core
    
    info "Building (this may take 2-5 minutes)..."
    if ! cargo build --release 2>/dev/null; then
        echo ""
        error "Build failed. Common fixes:"
        echo ""
        case "$OS_NAME" in
            linux)
                echo "  sudo apt install build-essential pkg-config libssl-dev"
                ;;
            macos)
                echo "  xcode-select --install"
                echo "  brew install openssl@3"
                ;;
        esac
        echo ""
        echo "See: https://reasonkit.sh/docs/installation-troubleshooting"
        exit 1
    fi
    
    # Install binary
    cp target/release/rk-core "$INSTALL_DIR/"
    chmod +x "$INSTALL_DIR/rk-core"
    
    success "Built and installed from source"
}

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

configure_path() {
    # Check if install dir is in PATH
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        echo ""
        warn "$INSTALL_DIR is not in your PATH"
        echo ""
        
        # Detect shell and provide appropriate instructions
        SHELL_NAME=$(basename "$SHELL")
        case "$SHELL_NAME" in
            zsh)
                RC_FILE="$HOME/.zshrc"
                ;;
            bash)
                if [ -f "$HOME/.bash_profile" ]; then
                    RC_FILE="$HOME/.bash_profile"
                else
                    RC_FILE="$HOME/.bashrc"
                fi
                ;;
            fish)
                RC_FILE="$HOME/.config/fish/config.fish"
                ;;
            *)
                RC_FILE="$HOME/.profile"
                ;;
        esac
        
        echo "Add this to $RC_FILE:"
        echo ""
        if [ "$SHELL_NAME" = "fish" ]; then
            echo -e "  ${CYAN}fish_add_path $INSTALL_DIR${NC}"
        else
            echo -e "  ${CYAN}export PATH=\"\$PATH:$INSTALL_DIR\"${NC}"
        fi
        echo ""
        echo "Then run:"
        echo -e "  ${CYAN}source $RC_FILE${NC}"
        echo ""
        
        # Offer to add automatically
        if [ -t 0 ]; then
            read -p "Add to PATH automatically? [Y/n] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                if [ "$SHELL_NAME" = "fish" ]; then
                    echo "fish_add_path $INSTALL_DIR" >> "$RC_FILE"
                else
                    echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$RC_FILE"
                fi
                success "Added to $RC_FILE"
                export PATH="$PATH:$INSTALL_DIR"
            fi
        fi
    fi
}

# ============================================================================
# VALIDATION
# ============================================================================

validate_installation() {
    echo ""
    info "Validating installation..."
    
    # Find the binary
    RK_BINARY=""
    if [ -x "$INSTALL_DIR/rk-core" ]; then
        RK_BINARY="$INSTALL_DIR/rk-core"
    elif command -v rk-core &> /dev/null; then
        RK_BINARY=$(command -v rk-core)
    fi
    
    if [ -z "$RK_BINARY" ]; then
        error "rk-core binary not found after installation"
        echo ""
        echo "Try running manually:"
        echo "  $INSTALL_DIR/rk-core --version"
        return 1
    fi
    
    # Test version command
    VERSION_OUTPUT=$("$RK_BINARY" --version 2>&1)
    if [ $? -eq 0 ]; then
        success "Binary works: $VERSION_OUTPUT"
    else
        warn "Binary found but --version failed"
        echo "Output: $VERSION_OUTPUT"
    fi
    
    # Test help command
    if "$RK_BINARY" --help &> /dev/null; then
        success "Help command works"
    fi
    
    return 0
}

# ============================================================================
# QUICK START GUIDE
# ============================================================================

print_quickstart() {
    echo ""
    echo -e "${GREEN}${BOLD}============================================${NC}"
    echo -e "${GREEN}${BOLD}   Installation Complete!${NC}"
    echo -e "${GREEN}${BOLD}============================================${NC}"
    echo ""
    echo -e "${BOLD}Quick Start:${NC}"
    echo ""
    echo "1. Set your API key:"
    echo -e "   ${CYAN}export ANTHROPIC_API_KEY=\"sk-ant-...\"${NC}"
    echo ""
    echo "2. Run your first analysis:"
    echo -e "   ${CYAN}rk-core think \"Should I use microservices?\" --profile quick${NC}"
    echo ""
    echo -e "${BOLD}ThinkTools:${NC}"
    echo ""
    echo "  rk-core think \"query\" --profile quick      # Fast analysis"
    echo "  rk-core think \"query\" --profile balanced   # Standard"
    echo "  rk-core think \"query\" --profile deep       # Thorough"
    echo "  rk-core think \"query\" --profile paranoid   # Maximum rigor"
    echo ""
    echo -e "${BOLD}Individual Tools:${NC}"
    echo ""
    echo "  --protocol gigathink      # 10+ perspectives"
    echo "  --protocol laserlogic     # Logic validation"
    echo "  --protocol bedrock        # First principles"
    echo "  --protocol proofguard     # Source verification"
    echo "  --protocol brutalhonesty  # Adversarial critique"
    echo ""
    echo -e "${BOLD}Documentation:${NC}"
    echo ""
    echo "  Website:  https://reasonkit.sh"
    echo "  Docs:     https://reasonkit.sh/docs"
    echo "  GitHub:   https://github.com/reasonkit/reasonkit-core"
    echo ""
    echo -e "${CYAN}Turn prompts into protocols.${NC}"
    echo ""
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    print_banner
    
    detect_platform
    determine_install_dir
    check_dependencies
    check_rust
    
    echo ""
    info "Installing ReasonKit..."
    
    # Try installation methods in order of preference
    if try_prebuilt_binary; then
        : # Success
    elif install_via_cargo; then
        : # Success
    else
        install_from_source
    fi
    
    configure_path
    validate_installation
    print_quickstart
}

# Run main function
main "$@"
