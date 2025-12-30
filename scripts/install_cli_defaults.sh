#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WRAPPER="$ROOT/scripts/rk-cli-wrapper.sh"
TARGET_DIR="$HOME/.local/bin"

mkdir -p "$TARGET_DIR"

find_real_bin() {
  local tool="$1"
  IFS=':' read -r -a parts <<< "$PATH"
  for p in "${parts[@]}"; do
    if [[ "$p" == "$TARGET_DIR" ]]; then
      continue
    fi
    if [[ -x "$p/$tool" ]]; then
      echo "$p/$tool"
      return 0
    fi
  done
  if [[ -x "$TARGET_DIR/$tool" ]]; then
    echo "$TARGET_DIR/$tool"
    return 0
  fi
  return 1
}

install_wrapper() {
  local tool="$1"
  local real_bin="$2"
  local wrapper_path="$TARGET_DIR/$tool"

  if [[ -f "$real_bin" ]] && grep -q "ReasonKit CLI wrapper" "$real_bin" 2>/dev/null; then
    if [[ -x "$real_bin.rk-real" ]]; then
      real_bin="$real_bin.rk-real"
    fi
  fi

  if [[ "$real_bin" == "$wrapper_path" ]]; then
    local moved="$wrapper_path.rk-real"
    mv "$wrapper_path" "$moved"
    real_bin="$moved"
  fi

  if [[ -e "$wrapper_path" ]]; then
    if ! grep -q "ReasonKit CLI wrapper" "$wrapper_path" 2>/dev/null; then
      mv "$wrapper_path" "$wrapper_path.rk-backup.$(date +%Y%m%d%H%M%S)"
    fi
  fi

  cat > "$wrapper_path" <<EOF
#!/usr/bin/env bash
# ReasonKit CLI wrapper (generated)
export RK_TOOL="$tool"
export RK_REAL_BIN="$real_bin"
export RK_CONFIG="\${RK_CONFIG:-$ROOT/config/cli_defaults.toml}"
exec "$WRAPPER" "\$@"
EOF
  chmod +x "$wrapper_path"
}

install_tool_if_present() {
  local tool="$1"
  local real_bin
  if ! real_bin=$(find_real_bin "$tool"); then
    echo "[skip] $tool not found" >&2
    return 0
  fi
  install_wrapper "$tool" "$real_bin"
  echo "[ok] $tool -> $real_bin"
}

# Standard CLI tools
install_tool_if_present "claude"
install_tool_if_present "gemini"
install_tool_if_present "codex"
install_tool_if_present "opencode"
install_tool_if_present "cursor-agent"

# GitHub Copilot CLI (via gh)
if command -v gh >/dev/null 2>&1; then
  install_wrapper "copilot" "$(command -v gh)"
  echo "[ok] copilot -> gh"
else
  echo "[skip] gh not found (copilot wrapper not installed)" >&2
fi

ZSHRC="$HOME/.zshrc"
BLOCK_BEGIN="# ReasonKit CLI defaults (managed)"
BLOCK_END="# End ReasonKit CLI defaults"

if ! grep -q "$BLOCK_BEGIN" "$ZSHRC" 2>/dev/null; then
  cat >> "$ZSHRC" <<EOF

$BLOCK_BEGIN
if [[ ":\$PATH:" != *":\$HOME/.local/bin:"* ]]; then
  export PATH="\$HOME/.local/bin:\$PATH"
fi
export RK_CONFIG="\${RK_CONFIG:-$ROOT/config/cli_defaults.toml}"
$BLOCK_END
EOF
  echo "[ok] updated $ZSHRC"
else
  echo "[ok] $ZSHRC already configured"
fi
