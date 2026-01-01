# ReasonKit Protocol Mode - VS Code Theme

> **"Write code in the Void"** - High-contrast, distraction-free theme for ReasonKit

---

## Installation

### Manual Installation

1. Copy this entire directory to your VS Code extensions folder:
   ```bash
   # Linux
   cp -r reasonkit-core/brand/expansion-packs/vscode-extension ~/.vscode/extensions/reasonkit-protocol-mode
   
   # macOS
   cp -r reasonkit-core/brand/expansion-packs/vscode-extension ~/.vscode/extensions/reasonkit-protocol-mode
   
   # Windows
   xcopy reasonkit-core\brand\expansion-packs\vscode-extension %USERPROFILE%\.vscode\extensions\reasonkit-protocol-mode /E /I
   ```

2. Reload VS Code (Command Palette → "Developer: Reload Window")

3. Select the theme:
   - Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
   - Type "Color Theme"
   - Select "ReasonKit Protocol Mode"

### From Source (Development)

```bash
cd reasonkit-core/brand/expansion-packs/vscode-extension
code .
```

---

## Features

- **Void Black Background** (#030508) - Deep, distraction-free
- **High Contrast** - Maximum readability
- **Brand Colors:**
  - Cyan (#06b6d4) - Functions, active elements
  - Purple (#a855f7) - Logic control, keywords
  - Green (#10b981) - Strings, success
  - Orange (#f97316) - Errors, warnings
  - Pink (#ec4899) - Constants, special elements

- **Syntax Highlighting:**
  - Rust keywords in Purple
  - Functions in Cyan
  - Strings in Green
  - Errors in Orange
  - Comments in muted gray

---

## Brand Philosophy

**"Designed, Not Dreamed"**

This theme embodies the ReasonKit brand:
- Industrial aesthetic
- Technical precision
- No distractions
- Professional quality

---

## Color Reference

| Element | Color | Hex |
|---------|-------|-----|
| Background | Void Black | #030508 |
| Foreground | Pure White | #f9fafb |
| Primary (Cyan) | Active | #06b6d4 |
| Secondary (Purple) | Logic | #a855f7 |
| Success (Green) | Strings | #10b981 |
| Alert (Orange) | Errors | #f97316 |
| Tertiary (Pink) | Constants | #ec4899 |

---

## Customization

Edit `themes/reasonkit-protocol-mode.json` to customize colors.

---

## License

Apache 2.0 - See [LICENSE](../../../../LICENSE)

---

**ReasonKit** - Turn Prompts into Protocols  
<https://reasonkit.sh>

