# 404 Void Page - System Error Interface

> **Classification:** Brand Extension - Error Page Design
> **Purpose:** Transform error pages into functional system outputs
> **Philosophy:** "Even errors are part of the system" - Industrial, precise, informative

---

## Concept

Instead of a generic "Whoops" page, the 404 error is a **functional system output** that maintains brand identity and provides useful information.

**Visual Design:**

- Single blinking cursor in center of black screen
- Terminal-like text output
- System diagnostic information
- Recovery action button

---

## HTML Implementation

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ERROR::404 - ReasonKit</title>
    <style>
      /* Import brand CSS or inline critical styles */
      @import url("/main.css");

      .void-container {
        min-height: 100vh;
        background: #030508; /* Void Black */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        font-family: "JetBrains Mono", monospace;
        color: #f9fafb;
        padding: 2rem;
      }

      .error-output {
        max-width: 800px;
        width: 100%;
      }

      .error-line {
        margin: 0.5rem 0;
        opacity: 0;
        animation: typeIn 0.5s forwards;
      }

      .error-line:nth-child(1) {
        animation-delay: 0.2s;
      }
      .error-line:nth-child(2) {
        animation-delay: 0.8s;
      }
      .error-line:nth-child(3) {
        animation-delay: 1.4s;
      }
      .error-line:nth-child(4) {
        animation-delay: 2s;
      }
      .error-line:nth-child(5) {
        animation-delay: 2.6s;
      }

      @keyframes typeIn {
        to {
          opacity: 1;
        }
      }

      .cursor {
        display: inline-block;
        width: 8px;
        height: 20px;
        background: #06b6d4; /* Cyan */
        animation: blink 1s infinite;
        margin-left: 4px;
      }

      @keyframes blink {
        0%,
        50% {
          opacity: 1;
        }
        51%,
        100% {
          opacity: 0;
        }
      }

      .error-code {
        color: #f97316; /* Orange */
        font-weight: bold;
      }

      .error-label {
        color: #06b6d4; /* Cyan */
      }

      .error-value {
        color: #10b981; /* Green */
      }

      .recovery-button {
        margin-top: 3rem;
        padding: 1rem 2rem;
        background: transparent;
        border: 2px solid #06b6d4;
        color: #06b6d4;
        font-family: "Inter", sans-serif;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.2s;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .recovery-button:hover {
        background: #06b6d4;
        color: #030508;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
      }

      .recovery-button:active {
        transform: scale(0.98);
      }

      /* High-contrast mode */
      @media (prefers-contrast: high) {
        .void-container {
          background: #000000;
        }
        .error-code,
        .error-label,
        .error-value {
          color: #ffffff;
        }
        .cursor {
          background: #ffffff;
        }
        .recovery-button {
          border-color: #ffffff;
          color: #ffffff;
        }
        .recovery-button:hover {
          background: #ffffff;
          color: #000000;
        }
      }

      /* Reduced motion */
      @media (prefers-reduced-motion: reduce) {
        .error-line {
          animation: none;
          opacity: 1;
        }
        .cursor {
          animation: none;
          opacity: 1;
        }
      }
    </style>
  </head>
  <body>
    <div class="void-container">
      <div class="error-output">
        <div class="error-line">
          <span class="error-label">></span>
          <span class="error-code">ERROR::404</span>
        </div>
        <div class="error-line">
          <span class="error-label">></span>
          <span class="error-value">PATH_TRACING_FAILED</span>
        </div>
        <div class="error-line">
          <span class="error-label">></span>
          <span class="error-value">REASONING_CHAIN_BROKEN</span>
        </div>
        <div class="error-line">
          <span class="error-label">></span>
          <span class="error-value">INITIATING_RECOVERY</span
          ><span class="cursor"></span>
        </div>
        <div class="error-line">
          <span class="error-label">></span>
          <span class="error-value">STATUS: READY</span>
        </div>
      </div>

      <button class="recovery-button" onclick="window.location.href='/'">
        REBOOT SYSTEM
      </button>
    </div>

    <script>
      // Optional: Add typing effect
      const lines = document.querySelectorAll(".error-line");
      lines.forEach((line, index) => {
        const text = line.textContent;
        line.textContent = "";
        line.style.opacity = "1";

        setTimeout(() => {
          let i = 0;
          const typeInterval = setInterval(() => {
            if (i < text.length) {
              line.textContent += text[i];
              i++;
            } else {
              clearInterval(typeInterval);
              if (index === lines.length - 2) {
                // Add cursor to last line
                const cursor = document.createElement("span");
                cursor.className = "cursor";
                line.appendChild(cursor);
              }
            }
          }, 30);
        }, index * 600);
      });
    </script>
  </body>
</html>
```

---

## React/Next.js Implementation

```tsx
// pages/404.tsx or app/not-found.tsx
import { useEffect, useState } from "react";
import Link from "next/link";

export default function NotFound() {
  const [lines, setLines] = useState<string[]>([]);
  const [showButton, setShowButton] = useState(false);

  useEffect(() => {
    const errorLines = [
      "> ERROR::404",
      "> PATH_TRACING_FAILED",
      "> REASONING_CHAIN_BROKEN",
      "> INITIATING_RECOVERY...",
      "> STATUS: READY",
    ];

    let currentLine = 0;
    const interval = setInterval(() => {
      if (currentLine < errorLines.length) {
        setLines((prev) => [...prev, errorLines[currentLine]]);
        currentLine++;
      } else {
        clearInterval(interval);
        setShowButton(true);
      }
    }, 600);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="void-container">
      <div className="error-output">
        {lines.map((line, index) => (
          <div key={index} className="error-line">
            {line}
            {index === lines.length - 1 && <span className="cursor" />}
          </div>
        ))}
      </div>

      {showButton && (
        <Link href="/">
          <button className="recovery-button">REBOOT SYSTEM</button>
        </Link>
      )}
    </div>
  );
}
```

---

## CSS Variables Integration

```css
/* Use brand CSS variables */
.void-container {
  background: var(--background); /* #030508 */
  color: var(--text); /* #f9fafb */
}

.error-code {
  color: var(--alert); /* #f97316 */
}

.error-label {
  color: var(--primary); /* #06b6d4 */
}

.error-value {
  color: var(--success); /* #10b981 */
}

.cursor {
  background: var(--primary); /* #06b6d4 */
}

.recovery-button {
  border-color: var(--primary);
  color: var(--primary);
}

.recovery-button:hover {
  background: var(--primary);
  color: var(--background);
  box-shadow: 0 0 20px var(--primary-glow);
}
```

---

## Additional Error Pages

### 500 Internal Server Error

```html
<div class="error-output">
  <div class="error-line">
    <span class="error-label">></span>
    <span class="error-code">ERROR::500</span>
  </div>
  <div class="error-line">
    <span class="error-label">></span>
    <span class="error-value">SYSTEM_FAULT_DETECTED</span>
  </div>
  <div class="error-line">
    <span class="error-label">></span>
    <span class="error-value">PROTOCOL_EXECUTION_FAILED</span>
  </div>
  <div class="error-line">
    <span class="error-label">></span>
    <span class="error-value">FALLBACK_MODE_ACTIVATED</span>
  </div>
</div>
```

### 403 Forbidden

```html
<div class="error-output">
  <div class="error-line">
    <span class="error-label">></span>
    <span class="error-code">ERROR::403</span>
  </div>
  <div class="error-line">
    <span class="error-label">></span>
    <span class="error-value">ACCESS_DENIED</span>
  </div>
  <div class="error-line">
    <span class="error-label">></span>
    <span class="error-value">AUTHORIZATION_FAILED</span>
  </div>
  <div class="error-line">
    <span class="error-label">></span>
    <span class="error-value">INSUFFICIENT_PERMISSIONS</span>
  </div>
</div>
```

---

## Brand Alignment

### Industrial Aesthetic

- ✅ Terminal-like interface
- ✅ System diagnostic output
- ✅ Precise, technical language
- ✅ No playful elements

### "Designed, Not Dreamed"

- ✅ Functional error information
- ✅ Clear recovery path
- ✅ Consistent with brand identity
- ✅ Professional, not apologetic

---

## Accessibility

- ✅ High-contrast mode support
- ✅ Reduced motion support
- ✅ Keyboard navigation
- ✅ Screen reader friendly
- ✅ Clear error messages

---

## File Location

```
reasonkit-site/
├── 404.html (or pages/404.tsx)
├── 500.html (or pages/500.tsx)
└── 403.html (or pages/403.tsx)
```

---

**Last Updated:** 2025-01-01  
**Status:** ✅ Specification Complete
