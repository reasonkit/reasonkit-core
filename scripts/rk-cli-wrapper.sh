#!/usr/bin/env bash
set -euo pipefail

tool="${RK_TOOL:-}"
real_bin="${RK_REAL_BIN:-}"

if [[ -z "$tool" || -z "$real_bin" ]]; then
  echo "ReasonKit wrapper misconfigured: RK_TOOL/RK_REAL_BIN required" >&2
  exit 1
fi

config_file="${RK_CONFIG:-}"
if [[ -z "$config_file" ]]; then
  if [[ -f "$HOME/.config/reasonkit/cli_defaults.toml" ]]; then
    config_file="$HOME/.config/reasonkit/cli_defaults.toml"
  else
    fallback_config="$(cd "$(dirname "${BASH_SOURCE[0]}")/../config" && pwd)/cli_defaults.toml"
    if [[ -f "$fallback_config" ]]; then
      config_file="$fallback_config"
    fi
  fi
fi

toml_get() {
  local file="$1"
  local section="$2"
  local key="$3"
  awk -v section="[$section]" -v key="$key" '
    BEGIN { in_section = 0 }
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*\[/ {
      gsub(/[[:space:]]/, "", $0)
      in_section = ($0 == section)
      next
    }
    in_section && $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
      sub("^[[:space:]]*" key "[[:space:]]*=[[:space:]]*", "", $0)
      gsub(/^[\"\047]/, "", $0)
      gsub(/[\"\047][[:space:]]*$/, "", $0)
      print $0
      exit
    }
  ' "$file"
}

no_rk=0
args=()
for arg in "$@"; do
  if [[ "$arg" == "--no-rk" ]]; then
    no_rk=1
    continue
  fi
  args+=("$arg")
done

if [[ "$no_rk" -eq 1 ]]; then
  exec "$real_bin" "${args[@]}"
fi

for arg in "${args[@]}"; do
  case "$arg" in
    -h|--help)
      exec "$real_bin" "${args[@]}"
      ;;
  esac
done

default_profile=""
default_protocol_dir=""
tool_profile=""
if [[ -n "$config_file" && -f "$config_file" ]]; then
  default_profile="$(toml_get "$config_file" "defaults" "profile")"
  default_protocol_dir="$(toml_get "$config_file" "defaults" "protocol_dir")"
  tool_key="$(printf "%s" "$tool" | tr '-' '_')"
  tool_profile="$(toml_get "$config_file" "tools.$tool_key" "profile")"
fi

profile="${RK_PROFILE:-${tool_profile:-${default_profile:-balanced}}}"
protocol_dir="${RK_PROTOCOL_DIR:-${default_protocol_dir:-}}"
if [[ -n "$protocol_dir" ]]; then
  if [[ "$protocol_dir" != /* ]]; then
    if [[ -n "$config_file" ]]; then
      config_dir="$(cd "$(dirname "$config_file")" && pwd)"
      protocol_dir="$config_dir/$protocol_dir"
    else
      protocol_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../$protocol_dir" && pwd)"
    fi
  fi
else
  protocol_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../protocols/cli" && pwd)"
fi
protocol_file="$protocol_dir/$profile.md"
if [[ ! -f "$protocol_file" ]]; then
  echo "ReasonKit protocol not found: $protocol_file" >&2
  exit 1
fi
protocol="$(cat "$protocol_file")"

wrap_prompt() {
  printf "%s\n\nUSER_REQUEST:\n%s" "$protocol" "$1"
}

join_words() {
  local out=""
  for w in "$@"; do
    if [[ -z "$out" ]]; then
      out="$w"
    else
      out="$out $w"
    fi
  done
  printf "%s" "$out"
}

rk_gemini() {
  local prompt=""
  local interactive=0
  local new_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--prompt)
        shift
        prompt="${1:-}"
        ;;
      -i|--prompt-interactive)
        interactive=1
        shift
        prompt="${1:-}"
        ;;
      --)
        shift
        prompt="$(join_words "$@")"
        break
        ;;
      -* )
        new_args+=("$1")
        ;;
      * )
        if [[ -z "$prompt" ]]; then
          prompt="$1"
        else
          prompt="$prompt $1"
        fi
        ;;
    esac
    shift
  done

  # Only inject protocol when there's an actual prompt
  if [[ -z "$prompt" ]]; then
    exec "$real_bin" "${new_args[@]}"
  fi

  local wrapped
  wrapped="$(wrap_prompt "$prompt")"
  if [[ "$interactive" -eq 1 ]]; then
    exec "$real_bin" "${new_args[@]}" --prompt-interactive "$wrapped"
  fi
  exec "$real_bin" "${new_args[@]}" "$wrapped"
}

rk_codex() {
  local first="${1:-}"
  case "$first" in
    login|logout|mcp|app-server|completion|resume|cloud|features|help|sandbox|apply|mcp-server)
      exec "$real_bin" "${args[@]}"
      ;;
  esac

  local subcmd=""
  local rest=("${args[@]}")
  if [[ "$first" == "exec" || "$first" == "review" ]]; then
    subcmd="$first"
    rest=("${args[@]:1}")
  fi

  local prompt=""
  local new_args=()
  local i=0
  while [[ $i -lt ${#rest[@]} ]]; do
    local a="${rest[$i]}"
    case "$a" in
      --)
        prompt="$(join_words "${rest[@]:$((i+1))}")"
        i=${#rest[@]}
        ;;
      -* )
        new_args+=("$a")
        ;;
      * )
        if [[ -z "$prompt" ]]; then
          prompt="$a"
        else
          prompt="$prompt $a"
        fi
        ;;
    esac
    i=$((i+1))
  done

  # Only inject protocol when there's an actual prompt
  if [[ -z "$prompt" ]]; then
    if [[ -n "$subcmd" ]]; then
      exec "$real_bin" "$subcmd" "${new_args[@]}"
    fi
    exec "$real_bin" "${new_args[@]}"
  fi

  local wrapped
  wrapped="$(wrap_prompt "$prompt")"

  if [[ -n "$subcmd" ]]; then
    exec "$real_bin" "$subcmd" "${new_args[@]}" "$wrapped"
  fi
  exec "$real_bin" "${new_args[@]}" "$wrapped"
}

rk_opencode() {
  local first="${1:-}"
  case "$first" in
    acp|attach|auth|agent|upgrade|uninstall|serve|web|models|stats|export|import|github|pr|session|help)
      exec "$real_bin" "${args[@]}"
      ;;
  esac

  local subcmd=""
  local rest=("${args[@]}")
  if [[ "$first" == "run" ]]; then
    subcmd="$first"
    rest=("${args[@]:1}")
  fi

  local prompt=""
  local new_args=()
  local i=0
  while [[ $i -lt ${#rest[@]} ]]; do
    local a="${rest[$i]}"
    case "$a" in
      --)
        prompt="$(join_words "${rest[@]:$((i+1))}")"
        i=${#rest[@]}
        ;;
      -* )
        new_args+=("$a")
        ;;
      * )
        if [[ -z "$prompt" ]]; then
          prompt="$a"
        else
          prompt="$prompt $a"
        fi
        ;;
    esac
    i=$((i+1))
  done

  # Only inject protocol when there's an actual prompt
  if [[ -z "$prompt" ]]; then
    if [[ -n "$subcmd" ]]; then
      exec "$real_bin" "$subcmd" "${new_args[@]}"
    fi
    exec "$real_bin" "${new_args[@]}"
  fi

  local wrapped
  wrapped="$(wrap_prompt "$prompt")"

  if [[ -n "$subcmd" ]]; then
    exec "$real_bin" "$subcmd" "${new_args[@]}" "$wrapped"
  fi
  exec "$real_bin" "${new_args[@]}" -p "$wrapped"
}

rk_cursor_agent() {
  local first="${1:-}"
  case "$first" in
    install-shell-integration|uninstall-shell-integration|login|logout|mcp|status|whoami|update|upgrade|create-chat|ls|resume|help)
      exec "$real_bin" "${args[@]}"
      ;;
  esac

  local subcmd=""
  local rest=("${args[@]}")
  if [[ "$first" == "agent" ]]; then
    subcmd="$first"
    rest=("${args[@]:1}")
  fi

  local prompt=""
  local new_args=()
  local i=0
  while [[ $i -lt ${#rest[@]} ]]; do
    local a="${rest[$i]}"
    case "$a" in
      --)
        prompt="$(join_words "${rest[@]:$((i+1))}")"
        i=${#rest[@]}
        ;;
      -* )
        new_args+=("$a")
        ;;
      * )
        if [[ -z "$prompt" ]]; then
          prompt="$a"
        else
          prompt="$prompt $a"
        fi
        ;;
    esac
    i=$((i+1))
  done

  # Only inject protocol when there's an actual prompt
  # For interactive sessions (no prompt), just run the real binary
  if [[ -z "$prompt" ]]; then
    if [[ -n "$subcmd" ]]; then
      exec "$real_bin" "$subcmd" "${new_args[@]}"
    fi
    exec "$real_bin" "${new_args[@]}"
  fi

  local wrapped
  wrapped="$(wrap_prompt "$prompt")"

  if [[ -n "$subcmd" ]]; then
    exec "$real_bin" "$subcmd" "${new_args[@]}" "$wrapped"
  fi
  exec "$real_bin" "${new_args[@]}" "$wrapped"
}

rk_copilot() {
  # Standalone Copilot CLI (not gh copilot)
  local first="${1:-}"
  case "$first" in
    login|logout|mcp|status|whoami|update|upgrade|help|version)
      exec "$real_bin" "${args[@]}"
      ;;
  esac

  local prompt=""
  local new_args=()
  local i=0
  while [[ $i -lt ${#args[@]} ]]; do
    local a="${args[$i]}"
    case "$a" in
      --)
        prompt="$(join_words "${args[@]:$((i+1))}")"
        i=${#args[@]}
        ;;
      -* )
        new_args+=("$a")
        ;;
      * )
        if [[ -z "$prompt" ]]; then
          prompt="$a"
        else
          prompt="$prompt $a"
        fi
        ;;
    esac
    i=$((i+1))
  done

  # Only inject protocol when there's an actual prompt
  if [[ -z "$prompt" ]]; then
    exec "$real_bin" "${new_args[@]}"
  fi

  local wrapped
  wrapped="$(wrap_prompt "$prompt")"
  exec "$real_bin" "${new_args[@]}" "$wrapped"
}

case "$tool" in
  claude)
    exec "$real_bin" --append-system-prompt "$protocol" "${args[@]}"
    ;;
  gemini)
    rk_gemini "${args[@]}"
    ;;
  codex)
    rk_codex "${args[@]}"
    ;;
  opencode)
    rk_opencode "${args[@]}"
    ;;
  cursor-agent)
    rk_cursor_agent "${args[@]}"
    ;;
  copilot)
    rk_copilot "${args[@]}"
    ;;
  *)
    exec "$real_bin" "${args[@]}"
    ;;
esac
