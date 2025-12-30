# ReasonKit Zsh Integration Research

> Deep Research: Optional Zsh Shell Integration for ReasonKit
> Status: Research Complete | Ready for Implementation Planning

---

## Executive Summary

Zsh integration offers significant UX enhancements for ReasonKit CLI users. This research identifies **5 integration tiers** ranging from basic shell completions (low effort, high value) to advanced async prompt integration (higher effort, specialized value).

### Value Matrix

| Integration Tier         | Effort | Impact | Priority |
| ------------------------ | ------ | ------ | -------- |
| Shell Completions        | Low    | High   | P0       |
| ZLE Widgets              | Medium | High   | P1       |
| Oh-My-Zsh Plugin         | Medium | Medium | P2       |
| fzf-style Keybindings    | Medium | High   | P1       |
| Async Prompt Integration | High   | Low    | P3       |

---

## Tier 1: Shell Completions (P0 - Immediate Value)

### Technology: `clap_complete`

The `clap_complete` crate provides **automatic shell completion generation** for Rust CLIs built with clap.

```rust
// src/completions.rs
use clap::{Command, CommandFactory};
use clap_complete::{generate, Shell};
use std::io;

pub fn generate_completions(shell: Shell) {
    let mut cmd = crate::Cli::command();
    generate(shell, &mut cmd, "rk-core", &mut io::stdout());
}
```

### Integration Pattern

```rust
// Add to main.rs CLI args
#[derive(Subcommand)]
enum Commands {
    // ... existing commands

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

// Handler
Commands::Completions { shell } => {
    generate_completions(shell);
}
```

### User Installation

```bash
# Generate and install Zsh completions
rk-core completions zsh > ~/.zsh/completions/_rk-core

# Or for Oh-My-Zsh users
rk-core completions zsh > $ZSH_CUSTOM/plugins/reasonkit/_rk-core
```

### What Gets Completed

With clap_complete, users get automatic completion for:

- **Subcommands**: `rk-core <TAB>` → `think`, `compare`, `metrics`, `rag`, `mcp`, etc.
- **Flags**: `rk-core think --<TAB>` → `--profile`, `--provider`, `--mock`, `--verbose`, etc.
- **Enum values**: `rk-core think --profile <TAB>` → `quick`, `balanced`, `deep`, `paranoid`, `powercombo`
- **Provider selection**: `--provider <TAB>` → all 23 providers including CLI tools

### Dependencies

```toml
# Cargo.toml
[dependencies]
clap_complete = "4.5"
```

### Effort Estimate

- **Implementation**: ~2 hours
- **Testing**: ~1 hour
- **Documentation**: ~30 minutes

---

## Tier 2: ZLE Widgets (P1 - Interactive Enhancement)

### Technology: Zsh Line Editor (ZLE)

ZLE widgets are custom functions bound to keystrokes that can intercept, modify, or execute commands.

### Core Widget Patterns

#### Pattern A: Quick Think Widget

```zsh
# ~/.zsh/functions/rk-widgets.zsh

# Ctrl+R+T: Quick think on current buffer
rk-quick-think() {
    local query="$BUFFER"
    if [[ -n "$query" ]]; then
        BUFFER="rk-core think --profile quick \"$query\""
        zle accept-line
    fi
}
zle -N rk-quick-think
bindkey '^Rt' rk-quick-think
```

#### Pattern B: Inline Reasoning Insert

```zsh
# Alt+R: Insert reasoning result at cursor
rk-inline-reason() {
    local query="$BUFFER"
    if [[ -n "$query" ]]; then
        local result=$(rk-core think --profile quick "$query" 2>/dev/null | tail -1)
        BUFFER="$result"
        CURSOR=${#BUFFER}
        zle redisplay
    fi
}
zle -N rk-inline-reason
bindkey '^[r' rk-inline-reason
```

#### Pattern C: Error Explanation Widget

```zsh
# Ctrl+E: Explain last command's error
rk-explain-error() {
    local last_cmd=$(fc -ln -1)
    local last_err=$(cat /tmp/last_stderr 2>/dev/null)
    if [[ -n "$last_err" ]]; then
        echo ""
        rk-core think --profile quick "Explain this error from '$last_cmd': $last_err"
    fi
}
zle -N rk-explain-error
bindkey '^E' rk-explain-error

# Capture stderr for error explanation
precmd() {
    # Capture is done via preexec hook
}
preexec() {
    exec 2> >(tee /tmp/last_stderr >&2)
}
```

### Widget Categories

| Category        | Keybinding | Function                    |
| --------------- | ---------- | --------------------------- |
| Quick Think     | `Ctrl+R T` | Run quick profile on buffer |
| Deep Think      | `Ctrl+R D` | Run deep profile on buffer  |
| Explain Error   | `Ctrl+E`   | Explain last error          |
| Inline Reason   | `Alt+R`    | Replace buffer with result  |
| Command Suggest | `Alt+S`    | Suggest command for task    |

### Effort Estimate

- **Implementation**: ~4 hours
- **Testing**: ~2 hours
- **Documentation**: ~1 hour

---

## Tier 3: Oh-My-Zsh Plugin (P2 - Ecosystem Integration)

### Plugin Structure

```
$ZSH_CUSTOM/plugins/reasonkit/
├── reasonkit.plugin.zsh    # Main plugin file
├── _rk-core                # Completion script
├── functions/
│   ├── rk-widgets.zsh      # Widget definitions
│   └── rk-helpers.zsh      # Helper functions
└── README.md               # Plugin documentation
```

### Main Plugin File

```zsh
# reasonkit.plugin.zsh

# Load completion
fpath=($ZSH_CUSTOM/plugins/reasonkit $fpath)

# Load widgets
source "$ZSH_CUSTOM/plugins/reasonkit/functions/rk-widgets.zsh"
source "$ZSH_CUSTOM/plugins/reasonkit/functions/rk-helpers.zsh"

# Aliases
alias rk='rk-core'
alias rkq='rk-core think --profile quick'
alias rkd='rk-core think --profile deep'
alias rkp='rk-core think --profile paranoid'
alias rkc='rk-core compare'
alias rkm='rk-core metrics report'

# Environment
export RK_DEFAULT_PROFILE="balanced"
export RK_DEFAULT_PROVIDER="anthropic"

# Keybindings (optional, user can disable)
if [[ -z "$RK_NO_KEYBINDS" ]]; then
    bindkey '^Rt' rk-quick-think
    bindkey '^Rd' rk-deep-think
    bindkey '^E' rk-explain-error
    bindkey '^[r' rk-inline-reason
fi
```

### Helper Functions

```zsh
# rk-helpers.zsh

# Ask ReasonKit about a command
rk-help() {
    rk-core think --profile quick "How do I: $*"
}

# Debug a file with ReasonKit
rk-debug() {
    local file="$1"
    if [[ -f "$file" ]]; then
        cat "$file" | rk-core think --profile deep "Debug this code and find issues:"
    else
        echo "File not found: $file"
    fi
}

# Code review with ReasonKit
rk-review() {
    local file="$1"
    if [[ -f "$file" ]]; then
        cat "$file" | rk-core think --profile paranoid "Code review this file:"
    fi
}

# Git commit message generation
rk-commit-msg() {
    local diff=$(git diff --cached)
    if [[ -n "$diff" ]]; then
        echo "$diff" | rk-core think --profile quick "Generate a conventional commit message for these changes:"
    else
        echo "No staged changes"
    fi
}
```

### Installation

```bash
# Clone to Oh-My-Zsh custom plugins
git clone https://github.com/reasonkit/zsh-plugin $ZSH_CUSTOM/plugins/reasonkit

# Add to .zshrc plugins array
plugins=(... reasonkit)
```

### Effort Estimate

- **Implementation**: ~6 hours
- **Testing**: ~2 hours
- **Documentation**: ~2 hours
- **Repo setup**: ~1 hour

---

## Tier 4: fzf-Style Keybindings (P1 - Power User UX)

### Inspiration: fzf Integration Pattern

fzf's Zsh integration provides an excellent model:

```zsh
# fzf pattern: Ctrl+T for file picker
# ReasonKit pattern: Ctrl+R for reasoning picker
```

### Interactive Profile Selector

```zsh
# Select profile interactively with fzf
rk-profile-select() {
    local profiles=("quick" "balanced" "deep" "paranoid" "powercombo" "scientific")
    local selected=$(printf '%s\n' "${profiles[@]}" | fzf --prompt="Profile: " --height=10)
    if [[ -n "$selected" ]]; then
        LBUFFER="${LBUFFER}--profile $selected "
        zle redisplay
    fi
}
zle -N rk-profile-select
bindkey '^Rp' rk-profile-select
```

### Interactive Provider Selector

```zsh
# Select provider interactively
rk-provider-select() {
    local providers=$(rk-core --help | grep -A 50 "provider" | grep -E "^\s+\w" | awk '{print $1}')
    local selected=$(echo "$providers" | fzf --prompt="Provider: " --height=20)
    if [[ -n "$selected" ]]; then
        LBUFFER="${LBUFFER}--provider $selected "
        zle redisplay
    fi
}
zle -N rk-provider-select
bindkey '^Rv' rk-provider-select
```

### History Search with Reasoning

```zsh
# Search previous ReasonKit queries
rk-history-search() {
    local selected=$(fc -l 1 | grep "rk-core" | fzf --tac --prompt="RK History: ")
    if [[ -n "$selected" ]]; then
        local cmd=$(echo "$selected" | sed 's/^[0-9]*\s*//')
        BUFFER="$cmd"
        CURSOR=${#BUFFER}
        zle redisplay
    fi
}
zle -N rk-history-search
bindkey '^Rh' rk-history-search
```

### Keybinding Reference Card

| Keybinding | Function          | Description                       |
| ---------- | ----------------- | --------------------------------- |
| `Ctrl+R P` | Profile selector  | fzf picker for reasoning profiles |
| `Ctrl+R V` | Provider selector | fzf picker for LLM providers      |
| `Ctrl+R H` | History search    | Search previous RK commands       |
| `Ctrl+R T` | Quick think       | Run quick profile on buffer       |
| `Ctrl+R D` | Deep think        | Run deep profile on buffer        |
| `Ctrl+R C` | Compare           | Compare raw vs enhanced           |

### Effort Estimate

- **Implementation**: ~4 hours
- **fzf dependency handling**: ~1 hour
- **Testing**: ~2 hours

---

## Tier 5: Async Prompt Integration (P3 - Advanced)

### Inspiration: Powerlevel10k Instant Prompt

Powerlevel10k uses background processes (`gitstatus`) to provide fast prompt updates without blocking.

### Pattern: Background Reasoning Status

```zsh
# Track last reasoning operation in prompt
typeset -g RK_LAST_CONFIDENCE=""
typeset -g RK_LAST_PROFILE=""

# After RK command, update prompt vars
rk-post-hook() {
    if [[ "$1" == rk-core* ]]; then
        # Parse output for confidence
        RK_LAST_CONFIDENCE=$(echo "$2" | grep -oP 'confidence: \K[\d.]+')
        RK_LAST_PROFILE=$(echo "$1" | grep -oP '\-\-profile \K\w+')
    fi
}

# Prompt segment
rk_prompt_segment() {
    if [[ -n "$RK_LAST_CONFIDENCE" ]]; then
        local color="green"
        (( RK_LAST_CONFIDENCE < 0.7 )) && color="yellow"
        (( RK_LAST_CONFIDENCE < 0.5 )) && color="red"
        echo "%F{$color}RK:${RK_LAST_CONFIDENCE}%f"
    fi
}
```

### Async Thinking Indicator

```zsh
# Show spinner while RK is thinking
typeset -g RK_THINKING=0

preexec() {
    if [[ "$1" == rk-core* ]]; then
        RK_THINKING=1
        # Start async spinner
        (while [[ -f /tmp/rk_thinking ]]; do
            for s in '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏'; do
                printf "\r%s ReasonKit thinking..." "$s"
                sleep 0.1
            done
        done) &
        touch /tmp/rk_thinking
    fi
}

precmd() {
    rm -f /tmp/rk_thinking
    RK_THINKING=0
}
```

### Effort Estimate

- **Implementation**: ~8 hours
- **Prompt compatibility testing**: ~4 hours
- **Powerlevel10k integration**: ~4 hours

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

| Task                  | Priority | Effort | Deliverable                         |
| --------------------- | -------- | ------ | ----------------------------------- |
| Add clap_complete     | P0       | 2h     | Shell completions for Zsh/Bash/Fish |
| Basic documentation   | P0       | 1h     | Installation instructions           |
| Test on common shells | P0       | 2h     | Verified completions                |

### Phase 2: Interactive (Week 2-3)

| Task             | Priority | Effort | Deliverable           |
| ---------------- | -------- | ------ | --------------------- |
| Core ZLE widgets | P1       | 4h     | 5 essential widgets   |
| fzf integration  | P1       | 4h     | Interactive selectors |
| Keybinding docs  | P1       | 2h     | Reference card        |

### Phase 3: Ecosystem (Week 4+)

| Task              | Priority | Effort | Deliverable               |
| ----------------- | -------- | ------ | ------------------------- |
| Oh-My-Zsh plugin  | P2       | 6h     | Installable plugin        |
| Plugin repo setup | P2       | 2h     | GitHub repo with releases |
| Community testing | P2       | N/A    | Beta feedback             |

### Phase 4: Advanced (Future)

| Task                  | Priority | Effort | Deliverable       |
| --------------------- | -------- | ------ | ----------------- |
| Async prompt          | P3       | 8h     | Background status |
| Powerlevel10k segment | P3       | 4h     | Native segment    |

---

## Dependencies

### Required

```toml
# Cargo.toml additions
[dependencies]
clap_complete = "4.5"

[build-dependencies]
clap_complete = "4.5"  # For build-time generation
```

### Optional (User-Side)

| Dependency    | Purpose               | Required For    |
| ------------- | --------------------- | --------------- |
| fzf           | Interactive selection | Tier 4 features |
| Oh-My-Zsh     | Plugin framework      | Tier 3 features |
| Powerlevel10k | Prompt integration    | Tier 5 features |

---

## Competitive Analysis

### Existing AI CLI Shell Integrations

| Tool                   | Shell Integration  | Approach                          |
| ---------------------- | ------------------ | --------------------------------- |
| **ShellGPT**           | Basic aliases      | Python wrapper, `sgpt` alias      |
| **Butterfish**         | Shell mode         | Embedded shell with AI context    |
| **ShellSage**          | Widget-based       | Zsh widgets for error explanation |
| **Shai**               | Inline replacement | Replace buffer with AI response   |
| **GitHub Copilot CLI** | Suggest command    | `gh copilot suggest`              |

### ReasonKit Differentiation

1. **Structured reasoning** - Not just chat, but auditable protocol execution
2. **Profile-based** - Quick/Balanced/Deep/Paranoid modes
3. **Multi-provider** - 23 providers including CLI tools
4. **Metrics integration** - Confidence scores in prompt
5. **Native Rust** - Fast completions, no Python startup lag

---

## Risks and Mitigations

| Risk                       | Probability | Impact | Mitigation                               |
| -------------------------- | ----------- | ------ | ---------------------------------------- |
| Shell compatibility issues | Medium      | Medium | Test on Zsh 5.8+, document requirements  |
| fzf dependency complaints  | Low         | Low    | Make fzf features optional               |
| Keybinding conflicts       | Medium      | Low    | Use `Ctrl+R` prefix, allow customization |
| Slow widget execution      | Low         | High   | Background execution, caching            |

---

## Recommendation

**Start with Tier 1 (Shell Completions)** - immediate high value, low effort.

```bash
# This should be the next commit:
cargo add clap_complete
# Then implement the completions subcommand
```

The shell completions alone will significantly improve UX for power users, and the foundation enables all subsequent tiers.

---

## References

1. clap_complete documentation: https://docs.rs/clap_complete/latest/clap_complete/
2. Zsh ZLE documentation: https://zsh.sourceforge.io/Doc/Release/Zsh-Line-Editor.html
3. fzf shell integration: https://github.com/junegunn/fzf#key-bindings-for-command-line
4. Oh-My-Zsh plugin guide: https://github.com/ohmyzsh/ohmyzsh/wiki/Customization
5. Powerlevel10k async: https://github.com/romkatv/powerlevel10k

---

_Research completed: 2025-12-24_
_Status: Ready for implementation planning_
