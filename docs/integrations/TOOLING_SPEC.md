# ReasonKit CLI Integration Spec

## Goals

- Make ReasonKit the default reasoning layer across common CLIs.
- Enforce structured outputs without exposing internal chain-of-thought.
- Provide easy override (`--no-rk`) and profile selection (`RK_PROFILE`).

## Wrapper Behavior

The install script creates wrappers in `~/.local/bin` that call `reasonkit-core/scripts/rk-cli-wrapper.sh` and inject the ReasonKit protocol by default.

## Central Config

Defaults live in `reasonkit-core/config/cli_defaults.toml`. You can override the path with `RK_CONFIG`.

Example:

```toml
[defaults]
profile = "balanced"
protocol_dir = "protocols/cli"

[tools.claude]
profile = "balanced"
```

### Supported Tools

#### Claude Code (`claude`)

- Injects with `--append-system-prompt`.
- Preserves all user flags and prompts.

#### Gemini CLI (`gemini`)

- Prefixes the prompt with the ReasonKit protocol.
- If no prompt is provided, starts interactive mode with the protocol prompt.

#### Codex CLI (`codex`)

- Wraps `codex` (interactive) and `codex exec` / `codex review` (non-interactive).
- For other subcommands, wrapper passes through without modification.

#### OpenCode (`opencode`)

- Wraps interactive and `opencode run` with protocol prefix.
- Other subcommands pass through unchanged.

#### Cursor Agent (`cursor-agent`)

- Prefixes initial prompt for interactive sessions.
- Non-prompt subcommands pass through unchanged.

#### GitHub Copilot CLI (`copilot` -> `gh copilot`)

- Prefixes prompts for `suggest`, `explain`, and `generate`.
- Other subcommands pass through unchanged.

## Profiles

- `quick` for speed
- `balanced` (default)
- `paranoid` for maximum rigor

`RK_PROFILE` overrides the config for a single session.

## Bypass

Add `--no-rk` to any command to run without ReasonKit injection.

## Known Limitations

- Complex flag parsing with multi-arg flags may require `--no-rk`.
- Non-prompt subcommands are passed through unchanged.
