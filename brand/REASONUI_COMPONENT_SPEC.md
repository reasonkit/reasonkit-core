# ReasonUI Component Specification

## "Industrial Console" - Visual Reasoning Interface System

> **Classification:** Component Design System for ReasonKit Ecosystem
> **Purpose:** Turn ReasonKit from a backend library into a visible, branded ecosystem
> **Philosophy:** "Designed, Not Dreamed" - Industrial precision, transparent reasoning

---

## Overview

ReasonUI is a strictly typed UI Component Kit that enables developers to build reasoning interfaces that _look_ like ReasonKit. These components visualize the structured, auditable reasoning process that ReasonKit enables.

### Core Principles

1. **Transparency First** - Every component shows _how_ reasoning works
2. **Industrial Aesthetic** - Heavy, precise, machinery-like interactions
3. **Brand Consistency** - Uses ReasonKit color system (Cyan/Purple/Pink)
4. **Accessibility** - WCAG AAA compliant, high-contrast mode support
5. **Type Safety** - Strictly typed in all implementations (TypeScript/Rust)

---

## Component Catalog

### 1. TraceNode

**Purpose:** Visualize a single step in the reasoning chain.

**Visual Design:**

- Hexagonal node (matching Luminous Polyhedron logo)
- Glass morphism effect (subtle backdrop blur)
- Cyan border with glow on active state
- Connection points (6) for parent/child relationships
- Status indicator (dot) in top-right corner

**States:**

- `pending` - Gray outline, no fill
- `processing` - Cyan pulse animation
- `completed` - Green fill, solid border
- `error` - Orange/red fill, alert border
- `skipped` - Muted gray, dashed border

**Props (TypeScript):**

```typescript
interface TraceNodeProps {
  id: string;
  label: string;
  status: "pending" | "processing" | "completed" | "error" | "skipped";
  confidence?: number; // 0-1
  metadata?: Record<string, unknown>;
  children?: TraceNodeProps[];
  onSelect?: (id: string) => void;
}
```

**Props (Rust - via web-sys/leptos):**

```rust
#[derive(Clone, Debug)]
pub struct TraceNode {
    pub id: String,
    pub label: String,
    pub status: TraceStatus,
    pub confidence: Option<f64>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub children: Vec<TraceNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TraceStatus {
    Pending,
    Processing,
    Completed,
    Error,
    Skipped,
}
```

**HTML/CSS Implementation:**

```html
<div class="trace-node trace-node--completed" data-id="node-1">
  <div class="trace-node__status-indicator"></div>
  <div class="trace-node__label">Reasoning Step</div>
  <div class="trace-node__confidence">85%</div>
  <div class="trace-node__connections">
    <div class="connection-point connection-point--top"></div>
    <div class="connection-point connection-point--right"></div>
    <!-- ... 4 more connection points -->
  </div>
</div>
```

---

### 2. ConfidenceMeter

**Purpose:** Radial or linear gauge displaying reasoning confidence.

**Visual Design:**

- Radial: Circular gauge (0-360°) with gradient fill (Cyan → Green)
- Linear: Horizontal bar with segmented zones
- Animated fill with "snap-to-value" physics
- Numeric display (percentage) in center/below
- Color zones:
  - 0-50%: Orange/Red (low confidence)
  - 50-75%: Yellow (moderate)
  - 75-90%: Cyan (good)
  - 90-100%: Green (high)

**Props (TypeScript):**

```typescript
interface ConfidenceMeterProps {
  value: number; // 0-1
  variant?: "radial" | "linear";
  showLabel?: boolean;
  size?: "sm" | "md" | "lg";
  animated?: boolean;
}
```

**Props (Rust):**

```rust
#[derive(Clone, Debug)]
pub struct ConfidenceMeter {
    pub value: f64, // 0.0-1.0
    pub variant: MeterVariant,
    pub show_label: bool,
    pub size: MeterSize,
    pub animated: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MeterVariant {
    Radial,
    Linear,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MeterSize {
    Small,
    Medium,
    Large,
}
```

**HTML/CSS Implementation:**

```html
<div class="confidence-meter confidence-meter--radial" data-value="0.85">
  <svg class="confidence-meter__gauge" viewBox="0 0 100 100">
    <circle class="confidence-meter__track" cx="50" cy="50" r="45" />
    <circle
      class="confidence-meter__fill"
      cx="50"
      cy="50"
      r="45"
      stroke-dasharray="283"
      stroke-dashoffset="42"
    />
  </svg>
  <div class="confidence-meter__value">85%</div>
</div>
```

---

### 3. LogStream

**Purpose:** Terminal-like scrolling text area for raw reasoning logs.

**Visual Design:**

- Dark background (Void Black `#030508`)
- Monospace font (JetBrains Mono)
- Syntax highlighting for log levels:
  - `INFO`: Cyan text
  - `DEBUG`: Purple text
  - `WARN`: Yellow text
  - `ERROR`: Orange/Red text
  - `SUCCESS`: Green text
- Auto-scroll with "snap" to bottom
- Timestamp prefix (ISO 8601 format)
- Line numbers (optional)

**Props (TypeScript):**

```typescript
interface LogStreamProps {
  logs: LogEntry[];
  maxLines?: number;
  autoScroll?: boolean;
  showTimestamps?: boolean;
  showLineNumbers?: boolean;
  filter?: LogLevel[];
}

interface LogEntry {
  timestamp: Date;
  level: "INFO" | "DEBUG" | "WARN" | "ERROR" | "SUCCESS";
  message: string;
  metadata?: Record<string, unknown>;
}
```

**Props (Rust):**

```rust
#[derive(Clone, Debug)]
pub struct LogStream {
    pub logs: Vec<LogEntry>,
    pub max_lines: Option<usize>,
    pub auto_scroll: bool,
    pub show_timestamps: bool,
    pub show_line_numbers: bool,
    pub filter: Vec<LogLevel>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: LogLevel,
    pub message: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LogLevel {
    Info,
    Debug,
    Warn,
    Error,
    Success,
}
```

**HTML/CSS Implementation:**

```html
<div class="log-stream">
  <div class="log-stream__header">
    <span class="log-stream__title">Reasoning Log</span>
    <div class="log-stream__controls">
      <button class="log-stream__clear">Clear</button>
    </div>
  </div>
  <div class="log-stream__content">
    <div class="log-entry log-entry--info">
      <span class="log-entry__timestamp">2026-01-01T12:00:00Z</span>
      <span class="log-entry__level">INFO</span>
      <span class="log-entry__message">Initializing reasoning protocol</span>
    </div>
    <!-- More log entries -->
  </div>
</div>
```

---

### 4. StatusToggle

**Purpose:** "Industrial switch" style toggles for active/inactive states.

**Visual Design:**

- Large, heavy toggle switch (machinery aesthetic)
- Cyan glow when active
- "Snap" animation (no easing, instant state change)
- Label above/below toggle
- Optional icon (checkmark/X, or custom SVG)

**States:**

- `off` - Gray, left position
- `on` - Cyan with glow, right position
- `disabled` - Muted, no interaction

**Props (TypeScript):**

```typescript
interface StatusToggleProps {
  checked: boolean;
  label?: string;
  disabled?: boolean;
  size?: "sm" | "md" | "lg";
  onChange?: (checked: boolean) => void;
  icon?: "check" | "x" | "custom";
  customIcon?: React.ReactNode;
}
```

**Props (Rust):**

```rust
#[derive(Clone, Debug)]
pub struct StatusToggle {
    pub checked: bool,
    pub label: Option<String>,
    pub disabled: bool,
    pub size: ToggleSize,
    pub icon: Option<ToggleIcon>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToggleSize {
    Small,
    Medium,
    Large,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToggleIcon {
    Check,
    X,
    Custom(String), // SVG path
}
```

**HTML/CSS Implementation:**

```html
<label class="status-toggle">
  <input type="checkbox" class="status-toggle__input" checked />
  <span class="status-toggle__slider"></span>
  <span class="status-toggle__label">Enable Reasoning</span>
</label>
```

---

## Implementation Guidelines

### React/TypeScript

**Installation:**

```bash
npm install @reasonkit/ui
# or
yarn add @reasonkit/ui
```

**Usage:**

```tsx
import {
  TraceNode,
  ConfidenceMeter,
  LogStream,
  StatusToggle,
} from "@reasonkit/ui";

function ReasoningDashboard() {
  return (
    <div className="reasoning-dashboard">
      <StatusToggle
        checked={isActive}
        onChange={setIsActive}
        label="Reasoning Active"
      />
      <ConfidenceMeter value={0.85} variant="radial" />
      <TraceNode
        id="step-1"
        label="Initial Analysis"
        status="completed"
        confidence={0.85}
      />
      <LogStream logs={logEntries} autoScroll />
    </div>
  );
}
```

### Rust (Leptos/WebAssembly)

**Cargo.toml:**

```toml
[dependencies]
reasonkit-ui = { version = "0.1.0", features = ["leptos"] }
leptos = "0.6"
```

**Usage:**

```rust
use reasonkit_ui::{TraceNode, ConfidenceMeter, LogStream, StatusToggle};

#[component]
pub fn ReasoningDashboard() -> impl IntoView {
    let (is_active, set_is_active) = create_signal(false);
    let confidence = create_signal(0.85);

    view! {
        <div class="reasoning-dashboard">
            <StatusToggle
                checked=is_active
                on_change=move |v| set_is_active.set(v)
                label="Reasoning Active"
            />
            <ConfidenceMeter value=confidence variant=MeterVariant::Radial />
            <TraceNode
                id="step-1"
                label="Initial Analysis"
                status=TraceStatus::Completed
                confidence=Some(0.85)
            />
            <LogStream logs=log_entries auto_scroll=true />
        </div>
    }
}
```

### HTML/CSS (Vanilla)

**Include CSS:**

```html
<link rel="stylesheet" href="https://cdn.reasonkit.sh/ui/v1/reasonui.css" />
```

**Usage:**

```html
<div class="reasoning-dashboard">
  <label class="status-toggle">
    <input type="checkbox" class="status-toggle__input" checked />
    <span class="status-toggle__slider"></span>
    <span class="status-toggle__label">Reasoning Active</span>
  </label>

  <div class="confidence-meter confidence-meter--radial" data-value="0.85">
    <!-- SVG gauge -->
  </div>

  <div class="trace-node trace-node--completed" data-id="step-1">
    <!-- Node content -->
  </div>

  <div class="log-stream">
    <!-- Log entries -->
  </div>
</div>
```

---

## Styling System

All components use ReasonKit brand colors via CSS variables:

```css
:root {
  --rk-cyan: #06b6d4;
  --rk-purple: #a855f7;
  --rk-pink: #ec4899;
  --rk-green: #10b981;
  --rk-orange: #f97316;
  --rk-bg-void: #030508;
  --rk-bg-deep: #0a0d14;
  --rk-text-primary: #f9fafb;
}
```

**High-Contrast Mode:**
All components automatically adapt to `@media (prefers-contrast: high)` with:

- Pure black backgrounds (`#000000`)
- Pure white text (`#ffffff`)
- 100% saturation borders

---

## Animation Guidelines

See [Motion Design Physics Guidelines](./MOTION_DESIGN_GUIDELINES.md) for detailed animation specifications.

**Quick Reference:**

- **Snap-to-Grid**: Instant state changes (no easing)
- **Data Flow**: Pulse animations (Cyan → Green direction)
- **Error Glitch**: Chromatic aberration "twitch" before settling

---

## Accessibility

- **WCAG AAA** compliant by default
- **Keyboard navigation** fully supported
- **Screen reader** labels on all interactive elements
- **Focus indicators** with 3px cyan outline
- **High-contrast mode** automatic adaptation

---

## Version History

| Version | Date       | Changes               |
| ------- | ---------- | --------------------- |
| 1.0.0   | 2026-01-01 | Initial specification |

---

**"Designed, Not Dreamed" - Turn Prompts into Protocols**
*https://reasonkit.sh*
