# ReasonKit Oh-My-Zsh Plugin
# AI Thinking Enhancement System - Turn Prompts into Protocols
#
# Installation:
#   1. Clone/copy to $ZSH_CUSTOM/plugins/reasonkit/
#   2. Add 'reasonkit' to plugins array in ~/.zshrc
#   3. Restart shell or run: source ~/.zshrc
#
# Configuration (add to ~/.zshrc before sourcing):
#   export RK_DEFAULT_PROVIDER="anthropic"
#   export RK_DEFAULT_PROFILE="balanced"
#   export RK_KEYBINDS_ENABLED="true"
#   export RK_CORE="/path/to/rk-core"  # Optional: custom binary path

# ============================================================================
# PLUGIN DIRECTORY DETECTION
# ============================================================================

# Get the directory where this plugin is installed
RK_PLUGIN_DIR="${0:A:h}"

# ============================================================================
# COMPLETION SETUP
# ============================================================================

# Add completions to fpath
if [[ -d "$RK_PLUGIN_DIR" ]]; then
    fpath=("$RK_PLUGIN_DIR" $fpath)
fi

# Generate completions if rk-core is available
if command -v rk-core &> /dev/null; then
    # Cache completions for faster loading
    if [[ ! -f "$RK_PLUGIN_DIR/_rk-core" ]] || \
       [[ "$(rk-core --version 2>/dev/null)" != "$(cat $RK_PLUGIN_DIR/.rk-version 2>/dev/null)" ]]; then
        rk-core completions zsh > "$RK_PLUGIN_DIR/_rk-core" 2>/dev/null
        rk-core --version > "$RK_PLUGIN_DIR/.rk-version" 2>/dev/null
    fi
fi

# ============================================================================
# WIDGET LOADING
# ============================================================================

# Source ZLE widgets if available
if [[ -f "$RK_PLUGIN_DIR/functions/rk-widgets.zsh" ]]; then
    source "$RK_PLUGIN_DIR/functions/rk-widgets.zsh"
fi

# ============================================================================
# ALIASES
# ============================================================================

# Core command aliases
alias rk='rk-core'
alias rkq='rk-core think --profile quick'
alias rkb='rk-core think --profile balanced'
alias rkd='rk-core think --profile deep'
alias rkp='rk-core think --profile paranoid'
alias rkpc='rk-core think --profile powercombo'
alias rks='rk-core think --profile scientific'

# Subcommand aliases
alias rkw='rk-core web'
alias rkv='rk-core verify'
alias rkm='rk-core metrics report'
alias rkc='rk-compare'

# Quick actions
alias rk-deep='rk-core web --depth deep'
alias rk-exhaust='rk-core web --depth exhaustive'

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Set defaults if not already configured
export RK_DEFAULT_PROVIDER="${RK_DEFAULT_PROVIDER:-anthropic}"
export RK_DEFAULT_PROFILE="${RK_DEFAULT_PROFILE:-balanced}"
export RK_KEYBINDS_ENABLED="${RK_KEYBINDS_ENABLED:-true}"

# Auto-detect rk-core binary
if [[ -z "$RK_CORE" ]]; then
    if command -v rk-core &> /dev/null; then
        export RK_CORE="rk-core"
    elif [[ -x "$HOME/.cargo/bin/rk-core" ]]; then
        export RK_CORE="$HOME/.cargo/bin/rk-core"
    fi
fi

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Quick think shorthand
think() {
    rk-core think --profile "${RK_DEFAULT_PROFILE:-balanced}" "$@"
}

# Deep research shorthand
research() {
    rk-core web --depth deep "$@"
}

# Verify a claim
verify() {
    rk-core verify "$@"
}

# Explain something
explain() {
    rk-core think --profile quick "Explain in simple terms: $*"
}

# Debug current directory's code
debug-here() {
    local pattern="${1:-*.rs}"
    find . -name "$pattern" -type f | head -5 | xargs cat | \
        rk-core think --profile deep "Debug this code and find issues:"
}

# Review staged git changes
review-staged() {
    git diff --cached | rk-core think --profile paranoid "Code review these changes:"
}

# Generate commit message for staged changes
commit-msg() {
    git diff --cached | rk-core think --profile quick \
        "Generate a conventional commit message (type: description format) for these changes:"
}

# ============================================================================
# PROMPT INTEGRATION (Optional)
# ============================================================================

# Last reasoning confidence (for prompt display)
typeset -g RK_LAST_CONFIDENCE=""
typeset -g RK_LAST_PROFILE=""

# Parse reasoning output for confidence
_rk_parse_confidence() {
    local output="$1"
    RK_LAST_CONFIDENCE=$(echo "$output" | grep -oP 'Confidence:\s*\K[\d.]+%' | tail -1)
}

# Prompt segment for Powerlevel10k (add to POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS)
prompt_reasonkit() {
    if [[ -n "$RK_LAST_CONFIDENCE" ]]; then
        local color="green"
        local conf_num=${RK_LAST_CONFIDENCE%\%}
        (( conf_num < 70 )) && color="yellow"
        (( conf_num < 50 )) && color="red"
        p10k segment -f $color -t "RK:$RK_LAST_CONFIDENCE"
    fi
}

# ============================================================================
# STARTUP MESSAGE
# ============================================================================

_rk_startup_message() {
    if [[ -n "$RK_CORE" ]] && command -v "$RK_CORE" &> /dev/null; then
        local version=$($RK_CORE --version 2>/dev/null | head -1)
        echo "\033[38;5;201mReasonKit\033[0m loaded ($version)"
        echo "  \033[2mType 'rk --help' for commands or use Ctrl+R keybindings\033[0m"
    else
        echo "\033[33mReasonKit plugin loaded but rk-core not found\033[0m"
        echo "  \033[2mInstall: cargo install reasonkit\033[0m"
    fi
}

# Show startup message (can be disabled with RK_QUIET=1)
if [[ -z "$RK_QUIET" ]]; then
    _rk_startup_message
fi

# ============================================================================
# KEYBINDING REFERENCE
# ============================================================================

# Display keybinding reference
rk-keys() {
    echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
    echo "\033[1;36m ReasonKit Keybindings\033[0m"
    echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
    echo ""
    echo "\033[1mCore Reasoning:\033[0m"
    echo "  \033[36mCtrl+R T\033[0m  Quick Think (fast 2-step analysis)"
    echo "  \033[36mCtrl+R D\033[0m  Deep Think (thorough 5-module analysis)"
    echo "  \033[36mCtrl+R P\033[0m  PowerCombo (maximum reasoning power)"
    echo "  \033[36mAlt+R\033[0m     Inline Reason (replace buffer with result)"
    echo "  \033[36mCtrl+R E\033[0m  Explain Error (analyze last command failure)"
    echo ""
    echo "\033[1mInteractive Selectors (requires fzf):\033[0m"
    echo "  \033[36mCtrl+R F\033[0m  Profile Selector (quick/balanced/deep/...)"
    echo "  \033[36mCtrl+R V\033[0m  Provider Selector (anthropic/openai/...)"
    echo "  \033[36mCtrl+R H\033[0m  History Search (previous RK commands)"
    echo "  \033[36mCtrl+R B\033[0m  Command Builder (interactive command)"
    echo ""
    echo "\033[1mAliases:\033[0m"
    echo "  \033[36mrk\033[0m        rk-core"
    echo "  \033[36mrkq\033[0m       rk-core think --profile quick"
    echo "  \033[36mrkd\033[0m       rk-core think --profile deep"
    echo "  \033[36mrkp\033[0m       rk-core think --profile paranoid"
    echo "  \033[36mrkpc\033[0m      rk-core think --profile powercombo"
    echo ""
    echo "\033[1mFunctions:\033[0m"
    echo "  \033[36mthink\033[0m     Quick reasoning on any topic"
    echo "  \033[36mresearch\033[0m  Deep web research"
    echo "  \033[36mverify\033[0m    Triangulate a claim"
    echo "  \033[36mexplain\033[0m   Simple explanation of a concept"
    echo ""
}
