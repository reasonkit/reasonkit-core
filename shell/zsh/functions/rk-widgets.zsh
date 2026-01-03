# ReasonKit ZLE Widgets
# Zsh Line Editor widgets for interactive reasoning
#
# Installation:
#   source /path/to/rk-widgets.zsh
#
# Usage:
#   Ctrl+R T  - Quick think on current buffer
#   Ctrl+R D  - Deep think on current buffer
#   Ctrl+R P  - PowerCombo on current buffer
#   Ctrl+R E  - Explain last error
#   Alt+R     - Inline reason (replace buffer with result)
#   Ctrl+R H  - History search for RK commands (requires fzf)
#   Ctrl+R V  - Provider selector (requires fzf)
#   Ctrl+R F  - Profile selector (requires fzf)

# ============================================================================
# CONFIGURATION
# ============================================================================

# ReasonKit binary path (auto-detected or set manually)
RK_CORE="${RK_CORE:-rk-core}"

# Default provider (can be overridden)
RK_DEFAULT_PROVIDER="${RK_DEFAULT_PROVIDER:-anthropic}"

# Default profile for quick operations
RK_DEFAULT_PROFILE="${RK_DEFAULT_PROFILE:-quick}"

# Enable/disable keybindings
RK_KEYBINDS_ENABLED="${RK_KEYBINDS_ENABLED:-true}"

# Spinner characters for async operations
RK_SPINNER=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')

# ============================================================================
# CORE WIDGETS
# ============================================================================

# Quick Think Widget (Ctrl+R T)
# Runs quick profile on current buffer content
rk-quick-think() {
    local query="$BUFFER"
    if [[ -n "$query" ]]; then
        echo ""
        echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
        echo "\033[1;36m ReasonKit Quick Think\033[0m"
        echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
        echo ""
        $RK_CORE think --profile quick --provider "$RK_DEFAULT_PROVIDER" "$query"
        echo ""
        zle reset-prompt
    else
        zle -M "ReasonKit: Buffer is empty"
    fi
}
zle -N rk-quick-think

# Deep Think Widget (Ctrl+R D)
# Runs deep profile for thorough analysis
rk-deep-think() {
    local query="$BUFFER"
    if [[ -n "$query" ]]; then
        echo ""
        echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
        echo "\033[1;36m ReasonKit Deep Think\033[0m"
        echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
        echo ""
        $RK_CORE think --profile deep --provider "$RK_DEFAULT_PROVIDER" "$query"
        echo ""
        zle reset-prompt
    else
        zle -M "ReasonKit: Buffer is empty"
    fi
}
zle -N rk-deep-think

# PowerCombo Widget (Ctrl+R P)
# Runs full PowerCombo - maximum reasoning power
rk-powercombo() {
    local query="$BUFFER"
    if [[ -n "$query" ]]; then
        echo ""
        echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
        echo "\033[1;36m ReasonKit PowerCombo - Maximum Reasoning\033[0m"
        echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
        echo ""
        $RK_CORE think --profile powercombo --provider "$RK_DEFAULT_PROVIDER" "$query"
        echo ""
        zle reset-prompt
    else
        zle -M "ReasonKit: Buffer is empty"
    fi
}
zle -N rk-powercombo

# Inline Reason Widget (Alt+R)
# Replaces buffer content with reasoning result
rk-inline-reason() {
    local query="$BUFFER"
    if [[ -n "$query" ]]; then
        # Show thinking indicator
        BUFFER="[ReasonKit thinking...]"
        zle redisplay

        # Get result (extract final output, not trace)
        local result=$($RK_CORE think --profile quick --format json "$query" 2>/dev/null | \
            grep -oP '"content"\s*:\s*"\K[^"]+' | tail -1)

        if [[ -n "$result" ]]; then
            BUFFER="$result"
        else
            BUFFER="$query"  # Restore on failure
            zle -M "ReasonKit: No result returned"
        fi
        CURSOR=${#BUFFER}
        zle redisplay
    fi
}
zle -N rk-inline-reason

# Explain Error Widget (Ctrl+R E)
# Explains the last command's error
rk-explain-error() {
    # Try to get last error from various sources
    local last_cmd=$(fc -ln -1 2>/dev/null)
    local last_err=""

    # Check common error capture locations
    if [[ -f /tmp/rk_last_stderr ]]; then
        last_err=$(cat /tmp/rk_last_stderr 2>/dev/null)
    fi

    if [[ -z "$last_err" && -n "$?" && "$?" -ne 0 ]]; then
        last_err="Command failed with exit code: $?"
    fi

    if [[ -n "$last_cmd" ]]; then
        echo ""
        echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
        echo "\033[1;36m ReasonKit Error Explanation\033[0m"
        echo "\033[38;5;201m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
        echo ""
        echo "\033[33mCommand:\033[0m $last_cmd"
        if [[ -n "$last_err" ]]; then
            echo "\033[31mError:\033[0m $last_err"
        fi
        echo ""
        $RK_CORE think --profile quick "Explain this shell command error and suggest fixes. Command: '$last_cmd'. Error output: '$last_err'"
        echo ""
        zle reset-prompt
    else
        zle -M "ReasonKit: No previous command found"
    fi
}
zle -N rk-explain-error

# ============================================================================
# FZF-STYLE INTERACTIVE WIDGETS
# ============================================================================

# Profile Selector (Ctrl+R F) - requires fzf
rk-profile-select() {
    if ! command -v fzf &> /dev/null; then
        zle -M "ReasonKit: fzf not installed"
        return 1
    fi

    local profiles=(
        "quick:Fast 2-step analysis (GigaThink + LaserLogic)"
        "balanced:Standard 4-module chain"
        "deep:Thorough 5-module analysis"
        "paranoid:Maximum verification with BrutalHonesty"
        "powercombo:All 5 tools + cross-validation"
        "scientific:Scientific method automation"
        "decide:Decision support with risk analysis"
    )

    local selected=$(printf '%s\n' "${profiles[@]}" | \
        fzf --prompt="RK Profile: " \
            --height=12 \
            --delimiter=":" \
            --with-nth=1 \
            --preview='echo "Description: {2}"' \
            --preview-window=down:1:wrap)

    if [[ -n "$selected" ]]; then
        local profile_name=$(echo "$selected" | cut -d: -f1)
        LBUFFER="${LBUFFER}--profile $profile_name "
        zle redisplay
    fi
}
zle -N rk-profile-select

# Provider Selector (Ctrl+R V) - requires fzf
rk-provider-select() {
    if ! command -v fzf &> /dev/null; then
        zle -M "ReasonKit: fzf not installed"
        return 1
    fi

    local providers=(
        "anthropic:Anthropic Claude (default)"
        "openai:OpenAI GPT"
        "gemini:Google Gemini (AI Studio)"
        "groq:Groq (ultra-fast inference)"
        "mistral:Mistral AI"
        "deepseek:DeepSeek"
        "openrouter:OpenRouter (300+ models)"
        "claude-cli:Claude CLI (local)"
        "codex-cli:OpenAI Codex CLI (local)"
        "gemini-cli:Gemini CLI (local)"
    )

    local selected=$(printf '%s\n' "${providers[@]}" | \
        fzf --prompt="RK Provider: " \
            --height=15 \
            --delimiter=":" \
            --with-nth=1 \
            --preview='echo "Description: {2}"' \
            --preview-window=down:1:wrap)

    if [[ -n "$selected" ]]; then
        local provider_name=$(echo "$selected" | cut -d: -f1)
        LBUFFER="${LBUFFER}--provider $provider_name "
        zle redisplay
    fi
}
zle -N rk-provider-select

# History Search (Ctrl+R H) - requires fzf
rk-history-search() {
    if ! command -v fzf &> /dev/null; then
        zle -M "ReasonKit: fzf not installed"
        return 1
    fi

    local selected=$(fc -l 1 | grep -E "(rk-core|rk |rkq|rkd|rkp)" | \
        fzf --tac \
            --prompt="RK History: " \
            --height=20 \
            --preview='echo {}' \
            --preview-window=down:3:wrap)

    if [[ -n "$selected" ]]; then
        # Remove the history number prefix
        local cmd=$(echo "$selected" | sed 's/^[[:space:]]*[0-9]*[[:space:]]*//')
        BUFFER="$cmd"
        CURSOR=${#BUFFER}
        zle redisplay
    fi
}
zle -N rk-history-search

# Command Builder (Ctrl+R B) - Interactive command construction
rk-command-builder() {
    if ! command -v fzf &> /dev/null; then
        zle -M "ReasonKit: fzf not installed"
        return 1
    fi

    local commands=(
        "think:Execute structured reasoning protocols"
        "web:Deep research with web search"
        "verify:Triangulate and verify claims"
        "rag:Retrieval-Augmented Generation"
        "metrics:View execution metrics"
        "compare:Compare raw vs enhanced"
    )

    local selected=$(printf '%s\n' "${commands[@]}" | \
        fzf --prompt="RK Command: " \
            --height=10 \
            --delimiter=":" \
            --with-nth=1 \
            --preview='echo "Description: {2}"' \
            --preview-window=down:1:wrap)

    if [[ -n "$selected" ]]; then
        local cmd_name=$(echo "$selected" | cut -d: -f1)
        BUFFER="rk $cmd_name "
        CURSOR=${#BUFFER}
        zle redisplay
    fi
}
zle -N rk-command-builder

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Quick think alias function
rk-help() {
    $RK_CORE think --profile quick "How do I: $*"
}

# Debug a file
rk-debug() {
    local file="$1"
    if [[ -f "$file" ]]; then
        cat "$file" | $RK_CORE think --profile deep "Debug this code and find issues:"
    else
        echo "File not found: $file"
        return 1
    fi
}

# Code review
rk-review() {
    local file="$1"
    if [[ -f "$file" ]]; then
        cat "$file" | $RK_CORE think --profile paranoid "Code review this file:"
    else
        echo "File not found: $file"
        return 1
    fi
}

# Git commit message generation
rk-commit-msg() {
    local diff=$(git diff --cached 2>/dev/null)
    if [[ -n "$diff" ]]; then
        echo "$diff" | $RK_CORE think --profile quick "Generate a conventional commit message for these changes:"
    else
        echo "No staged changes"
        return 1
    fi
}

# ============================================================================
# KEYBINDING SETUP
# ============================================================================

_rk_setup_keybindings() {
    if [[ "$RK_KEYBINDS_ENABLED" != "true" ]]; then
        return
    fi

    # Core reasoning widgets (Ctrl+R prefix)
    bindkey '^Rt' rk-quick-think      # Ctrl+R T - Quick think
    bindkey '^Rd' rk-deep-think       # Ctrl+R D - Deep think
    bindkey '^Rp' rk-powercombo       # Ctrl+R P - PowerCombo
    bindkey '^Re' rk-explain-error    # Ctrl+R E - Explain error

    # Inline replacement (Alt+R)
    bindkey '\er' rk-inline-reason    # Alt+R - Inline reason

    # fzf-style selectors (Ctrl+R prefix)
    bindkey '^Rf' rk-profile-select   # Ctrl+R F - Profile selector
    bindkey '^Rv' rk-provider-select  # Ctrl+R V - Provider selector
    bindkey '^Rh' rk-history-search   # Ctrl+R H - History search
    bindkey '^Rb' rk-command-builder  # Ctrl+R B - Command builder
}

# Auto-setup keybindings
_rk_setup_keybindings

# ============================================================================
# ERROR CAPTURE HOOK (optional)
# ============================================================================

# Capture stderr for error explanation widget
# Add to your .zshrc: preexec() { _rk_capture_preexec "$1"; }
_rk_capture_preexec() {
    # This is a hook that can be added to preexec
    # It captures stderr for the error explanation widget
    exec 2> >(tee /tmp/rk_last_stderr >&2)
}

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

echo "\033[38;5;201mReasonKit ZLE Widgets loaded\033[0m"
echo "  \033[2mCtrl+R T\033[0m Quick Think  \033[2mCtrl+R D\033[0m Deep Think  \033[2mCtrl+R P\033[0m PowerCombo"
echo "  \033[2mAlt+R\033[0m Inline Reason  \033[2mCtrl+R E\033[0m Explain Error"
if command -v fzf &> /dev/null; then
    echo "  \033[2mCtrl+R F\033[0m Profile  \033[2mCtrl+R V\033[0m Provider  \033[2mCtrl+R H\033[0m History"
fi
