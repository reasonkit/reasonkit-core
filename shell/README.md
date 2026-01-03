# ReasonKit Shell Integration

Turn Prompts into Protocols directly from your terminal.

## Quick Start

### 1. Shell Completions (All Shells)

```bash
# Zsh
rk completions zsh > ~/.zsh/completions/_rk

# Bash
rk completions bash > ~/.bash_completion.d/rk

# Fish
rk completions fish > ~/.config/fish/completions/rk.fish

# PowerShell
rk completions powershell >> $PROFILE
```

### 2. Oh-My-Zsh Plugin (Recommended)

```bash
# Clone plugin
git clone https://github.com/reasonkit/reasonkit-zsh $ZSH_CUSTOM/plugins/reasonkit

# Or copy manually
cp -r shell/zsh/* $ZSH_CUSTOM/plugins/reasonkit/

# Add to ~/.zshrc
plugins=(... reasonkit)

# Reload
source ~/.zshrc
```

### 3. Manual Widget Loading

```bash
# Add to ~/.zshrc
source /path/to/reasonkit-core/shell/zsh/functions/rk-widgets.zsh
```

## Keybindings

| Keybinding | Function          | Description                  |
| ---------- | ----------------- | ---------------------------- |
| `Ctrl+R T` | Quick Think       | Fast 2-step analysis         |
| `Ctrl+R D` | Deep Think        | Thorough 5-module analysis   |
| `Ctrl+R P` | PowerCombo        | Maximum reasoning power      |
| `Alt+R`    | Inline Reason     | Replace buffer with result   |
| `Ctrl+R E` | Explain Error     | Analyze last command failure |
| `Ctrl+R F` | Profile Selector  | fzf picker for profiles      |
| `Ctrl+R V` | Provider Selector | fzf picker for providers     |
| `Ctrl+R H` | History Search    | Search previous RK commands  |
| `Ctrl+R B` | Command Builder   | Interactive command builder  |

## Aliases

| Alias  | Command                              |
| ------ | ------------------------------------ |
| `rk`   | `rk`                            |
| `rkq`  | `rk think --profile quick`      |
| `rkb`  | `rk think --profile balanced`   |
| `rkd`  | `rk think --profile deep`       |
| `rkp`  | `rk think --profile paranoid`   |
| `rkpc` | `rk think --profile powercombo` |
| `rkw`  | `rk web`                        |
| `rkv`  | `rk verify`                     |
| `rkm`  | `rk metrics report`             |

## Helper Functions

```bash
# Quick reasoning
think "What is the best approach for X?"

# Deep research
research "Latest developments in Y"

# Verify a claim
verify "Claim to triangulate"

# Explain a concept
explain "Quantum computing"

# Debug code in current directory
debug-here "*.py"

# Review staged git changes
review-staged

# Generate commit message
commit-msg
```

## Configuration

Add to `~/.zshrc` before sourcing the plugin:

```bash
# Default provider (anthropic, openai, groq, etc.)
export RK_DEFAULT_PROVIDER="anthropic"

# Default profile (quick, balanced, deep, paranoid)
export RK_DEFAULT_PROFILE="balanced"

# Enable/disable keybindings
export RK_KEYBINDS_ENABLED="true"

# Custom binary path (optional)
export RK_CORE="/path/to/rk"

# Suppress startup message
export RK_QUIET=1
```

## Requirements

- **Required**: `rk` binary
- **Optional**: `fzf` for interactive selectors

## Prompt Integration (Powerlevel10k)

Add to your Powerlevel10k config:

```bash
# Add to POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS
typeset -g POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS=(
    ...
    reasonkit  # Shows last confidence score
)
```

## Troubleshooting

### Keybindings not working

1. Check if keybindings are enabled: `echo $RK_KEYBINDS_ENABLED`
2. Verify widget loading: `zle -la | grep rk-`
3. Check for conflicts: `bindkey | grep '^R'`

### Completions not working

1. Regenerate: `rk completions zsh > ~/.zsh/completions/_rk`
2. Rebuild completion cache: `rm ~/.zcompdump* && compinit`

### fzf selectors not appearing

1. Check fzf installation: `which fzf`
2. Install fzf: `brew install fzf` or `apt install fzf`
