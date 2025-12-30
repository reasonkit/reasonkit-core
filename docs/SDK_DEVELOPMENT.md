# ReasonKit SDK Development and Publishing Guide

> Comprehensive guide for developing, testing, and publishing ReasonKit SDKs across multiple languages.

**Version:** 1.0.0
**Last Updated:** 2025-12-28
**Core Version Compatibility:** reasonkit-core v0.1.0+
**License:** Apache 2.0

---

## Table of Contents

1. [Overview](#1-overview)
2. [SDK Strategy](#2-sdk-strategy)
3. [Python SDK](#3-python-sdk)
4. [Node.js SDK](#4-nodejs-sdk)
5. [Go SDK](#5-go-sdk)
6. [Common SDK Features](#6-common-sdk-features)
7. [SDK Testing](#7-sdk-testing)
8. [Documentation](#8-documentation)
9. [Publishing](#9-publishing)
10. [Versioning and Compatibility](#10-versioning-and-compatibility)
11. [Developer Experience](#11-developer-experience)
12. [Quality Gates](#12-quality-gates)

---

## 1. Overview

### Purpose

This guide defines the strategy, architecture, and implementation details for ReasonKit SDKs across multiple programming languages. The goal is to provide first-class developer experiences that match the power of the Rust core while feeling native to each language ecosystem.

### Design Philosophy

```
+-------------------+     +-------------------+     +-------------------+
|   Python SDK      |     |   Node.js SDK     |     |      Go SDK       |
|   (PyO3 FFI)      |     |   (napi-rs FFI)   |     |    (cgo FFI)      |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                         |
         v                         v                         v
+-----------------------------------------------------------------------+
|                        reasonkit-core (Rust)                          |
|                                                                        |
|  +-------------+  +-------------+  +-------------+  +---------------+  |
|  | ThinkTools  |  | Embedding   |  | Retrieval   |  | Storage       |  |
|  | Protocol    |  | Pipeline    |  | Engine      |  | Backends      |  |
|  +-------------+  +-------------+  +-------------+  +---------------+  |
+-----------------------------------------------------------------------+
```

### Key Principles

| Principle          | Description                                    |
| ------------------ | ---------------------------------------------- |
| **Native Feel**    | Each SDK should feel idiomatic to its language |
| **Type Safety**    | Full type coverage with IDE support            |
| **Async-First**    | Async/await where the language supports it     |
| **Error Handling** | Language-appropriate error patterns            |
| **Documentation**  | Rich docs, examples, and tutorials             |
| **Performance**    | Minimal overhead over direct Rust FFI          |

---

## 2. SDK Strategy

### Language Priority Matrix

| Priority | Language           | Binding Technology | Target Audience               | Timeline |
| -------- | ------------------ | ------------------ | ----------------------------- | -------- |
| **P0**   | Python             | PyO3               | Data Scientists, ML Engineers | v0.1.0   |
| **P1**   | Node.js/TypeScript | napi-rs            | Web Developers, Full-Stack    | v0.2.0   |
| **P2**   | Go                 | cgo                | Backend Engineers, DevOps     | v0.3.0   |
| **P3**   | Ruby               | rb-sys (future)    | Web Developers                | v0.5.0+  |
| **P3**   | Java/Kotlin        | JNI (future)       | Enterprise                    | v0.5.0+  |

### SDK vs CLI Comparison

| Feature               | SDK                          | CLI                     |
| --------------------- | ---------------------------- | ----------------------- |
| **Use Case**          | Programmatic integration     | Interactive/scripting   |
| **Installation**      | `pip install`, `npm install` | `cargo install`, `brew` |
| **Configuration**     | Code-based                   | File/env-based          |
| **Output**            | Structured objects           | Text/JSON               |
| **Async Support**     | Full async/await             | N/A                     |
| **Error Handling**    | Exceptions/errors            | Exit codes              |
| **Memory Management** | Language-native              | N/A                     |

### Parity Goals

All SDKs MUST support:

1. **ThinkTool Execution** - All 5 open-source ThinkTools
2. **Profile Support** - quick, balanced, deep, paranoid
3. **LLM Provider Integration** - 18+ providers via unified interface
4. **Execution Traces** - Full observability and logging
5. **Budget Management** - Time, token, and cost limits
6. **Configuration** - API keys, endpoints, timeouts

---

## 3. Python SDK

### Package Information

```
Package Name: reasonkit
PyPI: https://pypi.org/project/reasonkit/
Repository: https://github.com/reasonkit/reasonkit-python
Documentation: https://docs.reasonkit.sh/python
```

### Architecture

```
reasonkit/
├── __init__.py              # Package entry, version, exports
├── _core.pyi                # Type stubs for Rust bindings
├── client.py                # Main ReasonKit client class
├── config.py                # Configuration management
├── thinktools/
│   ├── __init__.py          # ThinkTool exports
│   ├── executor.py          # Protocol executor
│   ├── gigathink.py         # GigaThink wrapper
│   ├── laserlogic.py        # LaserLogic wrapper
│   ├── bedrock.py           # BedRock wrapper
│   ├── proofguard.py        # ProofGuard wrapper
│   └── brutalhonesty.py     # BrutalHonesty wrapper
├── profiles/
│   ├── __init__.py          # Profile exports
│   ├── base.py              # Base profile class
│   ├── quick.py             # Quick profile
│   ├── balanced.py          # Balanced profile
│   ├── deep.py              # Deep profile
│   └── paranoid.py          # Paranoid profile
├── providers/
│   ├── __init__.py          # Provider exports
│   ├── base.py              # Base provider interface
│   ├── anthropic.py         # Anthropic Claude
│   ├── openai.py            # OpenAI GPT
│   ├── openrouter.py        # OpenRouter multi-model
│   └── ...                  # Other providers
├── types/
│   ├── __init__.py          # Type exports
│   ├── results.py           # Result types
│   ├── traces.py            # Execution trace types
│   └── errors.py            # Error types
├── exceptions.py            # Custom exceptions
├── utils.py                 # Utility functions
└── py.typed                 # PEP 561 marker
```

### API Design

#### Basic Usage

```python
from reasonkit import ReasonKit

# Initialize client
rk = ReasonKit(api_key="sk-ant-...")

# Simple thinking request
result = rk.think(
    query="Should we adopt microservices architecture?",
    profile="balanced"
)

print(result.conclusion)
print(f"Confidence: {result.confidence:.2%}")

# Access reasoning trace
for step in result.trace.steps:
    print(f"[{step.name}] {step.summary}")
```

#### Advanced Usage

```python
from reasonkit import ReasonKit
from reasonkit.thinktools import GigaThink, LaserLogic, ProofGuard
from reasonkit.config import Config

# Custom configuration
config = Config(
    provider="anthropic",
    model="claude-sonnet-4",
    temperature=0.7,
    max_tokens=4000,
    timeout=60,
    retry_attempts=3,
)

rk = ReasonKit(config=config)

# Execute specific ThinkTools
result = rk.think(
    query="Evaluate this startup idea: AI-powered code review",
    thinktools=[GigaThink(), LaserLogic(), ProofGuard()],
    budget="60s",  # Time budget
    save_trace=True
)

# Access detailed results
print(f"Perspectives generated: {len(result.perspectives)}")
print(f"Logical fallacies detected: {result.fallacies}")
print(f"Verification status: {result.verification_status}")

# Save execution trace
result.trace.save("./traces/startup-analysis.json")
```

#### Async Support

```python
import asyncio
from reasonkit import AsyncReasonKit

async def analyze_multiple():
    rk = AsyncReasonKit(api_key="sk-ant-...")

    # Concurrent analysis
    queries = [
        "What are the risks of cloud migration?",
        "How should we approach security?",
        "What's the optimal team structure?"
    ]

    results = await asyncio.gather(*[
        rk.think(query=q, profile="quick")
        for q in queries
    ])

    for query, result in zip(queries, results):
        print(f"Q: {query}")
        print(f"A: {result.conclusion}\n")

asyncio.run(analyze_multiple())
```

#### Streaming Support

```python
from reasonkit import ReasonKit

rk = ReasonKit(api_key="sk-ant-...")

# Stream execution steps
for event in rk.think_stream(
    query="Analyze the quantum computing market",
    profile="deep"
):
    if event.type == "step_start":
        print(f"Starting: {event.step_name}")
    elif event.type == "step_complete":
        print(f"Completed: {event.step_name}")
        print(f"  Output: {event.output[:100]}...")
    elif event.type == "final":
        print(f"\nFinal: {event.conclusion}")
```

### PyO3 Bindings Implementation

#### Cargo.toml Configuration

```toml
[lib]
name = "reasonkit"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
```

#### Rust Binding Code

```rust
// src/bindings.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// ReasonKit Python bindings
#[pyclass]
pub struct Reasoner {
    inner: crate::thinktool::ProtocolExecutor,
}

#[pymethods]
impl Reasoner {
    /// Create a new Reasoner instance
    #[new]
    #[pyo3(signature = (api_key=None, provider=None, model=None))]
    pub fn new(
        api_key: Option<String>,
        provider: Option<String>,
        model: Option<String>,
    ) -> PyResult<Self> {
        let config = crate::thinktool::ExecutorConfig::builder()
            .api_key(api_key)
            .provider(provider.unwrap_or_else(|| "anthropic".to_string()))
            .model(model)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let inner = crate::thinktool::ProtocolExecutor::with_config(config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    /// Execute a ThinkTool protocol
    #[pyo3(signature = (query, protocol=None, profile=None, budget=None))]
    pub fn think(
        &self,
        py: Python<'_>,
        query: String,
        protocol: Option<String>,
        profile: Option<String>,
        budget: Option<String>,
    ) -> PyResult<PyObject> {
        // Execute protocol
        let result = py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                self.inner.execute(
                    protocol.as_deref().unwrap_or("balanced"),
                    crate::thinktool::ProtocolInput::query(&query),
                ).await
            })
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Convert to Python dict
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("conclusion", result.conclusion)?;
        dict.set_item("confidence", result.confidence)?;
        dict.set_item("perspectives", result.perspectives())?;

        Ok(dict.into())
    }

    /// List available protocols
    pub fn list_protocols(&self) -> Vec<String> {
        vec![
            "gigathink".to_string(),
            "laserlogic".to_string(),
            "bedrock".to_string(),
            "proofguard".to_string(),
            "brutalhonesty".to_string(),
        ]
    }

    /// List available profiles
    pub fn list_profiles(&self) -> Vec<String> {
        vec![
            "quick".to_string(),
            "balanced".to_string(),
            "deep".to_string(),
            "paranoid".to_string(),
        ]
    }
}

/// Module initialization
#[pymodule]
fn reasonkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Reasoner>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
```

### Type Stubs

```python
# reasonkit/_core.pyi
from typing import Optional, List, Dict, Any

class Reasoner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None: ...

    def think(
        self,
        query: str,
        protocol: Optional[str] = None,
        profile: Optional[str] = None,
        budget: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    def list_protocols(self) -> List[str]: ...
    def list_profiles(self) -> List[str]: ...

__version__: str
```

### Error Handling

```python
# reasonkit/exceptions.py
class ReasonKitError(Exception):
    """Base exception for ReasonKit errors."""
    pass

class ConfigurationError(ReasonKitError):
    """Raised when configuration is invalid."""
    pass

class ProviderError(ReasonKitError):
    """Raised when LLM provider returns an error."""
    pass

class RateLimitError(ProviderError):
    """Raised when rate limited by provider."""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after

class TimeoutError(ReasonKitError):
    """Raised when operation times out."""
    pass

class BudgetExceededError(ReasonKitError):
    """Raised when budget (time/tokens/cost) is exceeded."""
    def __init__(self, message: str, budget_type: str, limit: float, used: float):
        super().__init__(message)
        self.budget_type = budget_type
        self.limit = limit
        self.used = used

class ValidationError(ReasonKitError):
    """Raised when input validation fails."""
    pass
```

### Build and Distribution

#### pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "reasonkit"
version = "0.1.0"
description = "Structured AI reasoning framework - Turn Prompts into Protocols"
readme = "README.md"
license = { text = "Apache-2.0" }
authors = [{ name = "ReasonKit Team", email = "hello@reasonkit.sh" }]
keywords = ["ai", "reasoning", "llm", "thinking", "protocols"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
requires-python = ">=3.9"
dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
    "typing-extensions>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
    "maturin>=1.4.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "mkdocstrings[python]>=0.24.0",
]

[project.urls]
Homepage = "https://reasonkit.sh"
Documentation = "https://docs.reasonkit.sh/python"
Repository = "https://github.com/reasonkit/reasonkit-python"
Changelog = "https://github.com/reasonkit/reasonkit-python/blob/main/CHANGELOG.md"

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "reasonkit._core"

[tool.ruff]
line-length = 100
target-version = "py39"

[tool.mypy]
python_version = "3.9"
strict = true
```

---

## 4. Node.js SDK

### Package Information

```
Package Name: @reasonkit/core
npm: https://www.npmjs.com/package/@reasonkit/core
Repository: https://github.com/reasonkit/reasonkit-node
Documentation: https://docs.reasonkit.sh/node
```

### Architecture

```
@reasonkit/core/
├── src/
│   ├── index.ts             # Package entry
│   ├── client.ts            # Main ReasonKit client
│   ├── config.ts            # Configuration types
│   ├── native.ts            # napi-rs bindings interface
│   ├── thinktools/
│   │   ├── index.ts         # ThinkTool exports
│   │   ├── executor.ts      # Protocol executor
│   │   ├── gigathink.ts     # GigaThink wrapper
│   │   ├── laserlogic.ts    # LaserLogic wrapper
│   │   ├── bedrock.ts       # BedRock wrapper
│   │   ├── proofguard.ts    # ProofGuard wrapper
│   │   └── brutalhonesty.ts # BrutalHonesty wrapper
│   ├── profiles/
│   │   ├── index.ts         # Profile exports
│   │   └── types.ts         # Profile type definitions
│   ├── providers/
│   │   ├── index.ts         # Provider exports
│   │   ├── base.ts          # Base provider interface
│   │   └── ...              # Provider implementations
│   └── types/
│       ├── index.ts         # Type exports
│       ├── results.ts       # Result types
│       ├── traces.ts        # Trace types
│       └── errors.ts        # Error types
├── native/                  # Rust napi-rs bindings
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
├── package.json
├── tsconfig.json
└── README.md
```

### API Design

#### Basic Usage

```typescript
import { ReasonKit } from "@reasonkit/core";

// Initialize client
const rk = new ReasonKit({ apiKey: "sk-ant-..." });

// Simple thinking request
const result = await rk.think({
  query: "Should we adopt microservices architecture?",
  profile: "balanced",
});

console.log(result.conclusion);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);

// Access reasoning trace
for (const step of result.trace.steps) {
  console.log(`[${step.name}] ${step.summary}`);
}
```

#### Advanced Usage

```typescript
import { ReasonKit, Config, ThinkTools } from "@reasonkit/core";

// Custom configuration
const config: Config = {
  provider: "anthropic",
  model: "claude-sonnet-4",
  temperature: 0.7,
  maxTokens: 4000,
  timeout: 60000,
  retryAttempts: 3,
};

const rk = new ReasonKit(config);

// Execute specific ThinkTools
const result = await rk.think({
  query: "Evaluate this startup idea: AI-powered code review",
  thinktools: [
    ThinkTools.GigaThink,
    ThinkTools.LaserLogic,
    ThinkTools.ProofGuard,
  ],
  budget: "60s",
  saveTrace: true,
});

// Access detailed results
console.log(`Perspectives: ${result.perspectives.length}`);
console.log(`Fallacies: ${result.fallacies.join(", ")}`);
console.log(`Verification: ${result.verificationStatus}`);

// Save execution trace
await result.trace.save("./traces/startup-analysis.json");
```

#### Streaming Support

```typescript
import { ReasonKit } from "@reasonkit/core";

const rk = new ReasonKit({ apiKey: "sk-ant-..." });

// Stream execution steps
for await (const event of rk.thinkStream({
  query: "Analyze the quantum computing market",
  profile: "deep",
})) {
  switch (event.type) {
    case "step_start":
      console.log(`Starting: ${event.stepName}`);
      break;
    case "step_complete":
      console.log(`Completed: ${event.stepName}`);
      console.log(`  Output: ${event.output.slice(0, 100)}...`);
      break;
    case "final":
      console.log(`\nFinal: ${event.conclusion}`);
      break;
  }
}
```

### napi-rs Bindings

#### native/Cargo.toml

```toml
[package]
name = "reasonkit-node-native"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
reasonkit-core = { path = "../../reasonkit-core" }
napi = { version = "2", features = ["async", "serde-json"] }
napi-derive = "2"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[build-dependencies]
napi-build = "2"
```

#### native/src/lib.rs

```rust
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct ThinkResult {
    pub conclusion: String,
    pub confidence: f64,
    pub perspectives: Vec<String>,
    pub trace: Option<String>,
}

#[napi(object)]
pub struct ThinkOptions {
    pub query: String,
    pub protocol: Option<String>,
    pub profile: Option<String>,
    pub budget: Option<String>,
}

#[napi]
pub struct Reasoner {
    inner: reasonkit_core::thinktool::ProtocolExecutor,
}

#[napi]
impl Reasoner {
    #[napi(constructor)]
    pub fn new(api_key: Option<String>) -> Result<Self> {
        let config = reasonkit_core::thinktool::ExecutorConfig::builder()
            .api_key(api_key)
            .build()
            .map_err(|e| Error::from_reason(e.to_string()))?;

        let inner = reasonkit_core::thinktool::ProtocolExecutor::with_config(config)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self { inner })
    }

    #[napi]
    pub async fn think(&self, options: ThinkOptions) -> Result<ThinkResult> {
        let protocol = options.protocol.as_deref().unwrap_or("balanced");
        let input = reasonkit_core::thinktool::ProtocolInput::query(&options.query);

        let result = self.inner
            .execute(protocol, input)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(ThinkResult {
            conclusion: result.conclusion,
            confidence: result.confidence,
            perspectives: result.perspectives(),
            trace: result.trace.map(|t| serde_json::to_string(&t).unwrap()),
        })
    }

    #[napi]
    pub fn list_protocols(&self) -> Vec<String> {
        vec![
            "gigathink".to_string(),
            "laserlogic".to_string(),
            "bedrock".to_string(),
            "proofguard".to_string(),
            "brutalhonesty".to_string(),
        ]
    }
}
```

### TypeScript Types

```typescript
// src/types/index.ts

export interface Config {
  apiKey?: string;
  provider?: "anthropic" | "openai" | "openrouter" | "groq" | "gemini";
  model?: string;
  temperature?: number;
  maxTokens?: number;
  timeout?: number;
  retryAttempts?: number;
  baseUrl?: string;
}

export interface ThinkOptions {
  query: string;
  protocol?: string;
  profile?: "quick" | "balanced" | "deep" | "paranoid";
  thinktools?: ThinkTool[];
  budget?: string;
  saveTrace?: boolean;
  context?: string;
}

export interface ThinkResult {
  conclusion: string;
  confidence: number;
  perspectives: string[];
  fallacies: string[];
  verificationStatus: "verified" | "partial" | "unverified";
  trace: ExecutionTrace;
}

export interface ExecutionTrace {
  protocolId: string;
  steps: StepTrace[];
  startedAt: Date;
  completedAt: Date;
  totalTokens: number;
  totalCostUsd: number;
  save(path: string): Promise<void>;
}

export interface StepTrace {
  stepId: string;
  name: string;
  startedAt: Date;
  completedAt: Date;
  status: "success" | "failed" | "skipped";
  input: string;
  output: string;
  tokensUsed: number;
}

export type ThinkTool =
  | "gigathink"
  | "laserlogic"
  | "bedrock"
  | "proofguard"
  | "brutalhonesty";

export interface StreamEvent {
  type: "step_start" | "step_complete" | "step_error" | "final";
  stepName?: string;
  output?: string;
  conclusion?: string;
  error?: Error;
}
```

### package.json

```json
{
  "name": "@reasonkit/core",
  "version": "0.1.0",
  "description": "Structured AI reasoning framework - Turn Prompts into Protocols",
  "main": "dist/index.js",
  "module": "dist/index.mjs",
  "types": "dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.mjs",
      "require": "./dist/index.js"
    }
  },
  "files": ["dist", "native", "README.md"],
  "scripts": {
    "build": "tsup && napi build --release",
    "build:debug": "tsup && napi build",
    "test": "vitest",
    "test:coverage": "vitest --coverage",
    "lint": "eslint src --ext .ts",
    "typecheck": "tsc --noEmit",
    "prepublishOnly": "npm run build"
  },
  "keywords": [
    "ai",
    "reasoning",
    "llm",
    "thinking",
    "protocols",
    "anthropic",
    "openai"
  ],
  "author": "ReasonKit Team <hello@reasonkit.sh>",
  "license": "Apache-2.0",
  "repository": {
    "type": "git",
    "url": "https://github.com/reasonkit/reasonkit-node.git"
  },
  "homepage": "https://reasonkit.sh",
  "bugs": {
    "url": "https://github.com/reasonkit/reasonkit-node/issues"
  },
  "engines": {
    "node": ">=18.0.0"
  },
  "devDependencies": {
    "@napi-rs/cli": "^2.18.0",
    "@types/node": "^20.0.0",
    "eslint": "^8.57.0",
    "tsup": "^8.0.0",
    "typescript": "^5.3.0",
    "vitest": "^1.2.0"
  },
  "napi": {
    "name": "reasonkit",
    "triples": {
      "defaults": true,
      "additional": [
        "x86_64-unknown-linux-musl",
        "aarch64-unknown-linux-gnu",
        "aarch64-apple-darwin",
        "aarch64-unknown-linux-musl"
      ]
    }
  }
}
```

---

## 5. Go SDK

### Package Information

```
Import Path: github.com/reasonkit/reasonkit-go
Documentation: https://pkg.go.dev/github.com/reasonkit/reasonkit-go
Repository: https://github.com/reasonkit/reasonkit-go
```

### Architecture

```
github.com/reasonkit/reasonkit-go/
├── client.go                # Main ReasonKit client
├── config.go                # Configuration types
├── thinktools.go            # ThinkTool definitions
├── profiles.go              # Profile definitions
├── results.go               # Result types
├── traces.go                # Execution trace types
├── errors.go                # Error types
├── providers/
│   ├── provider.go          # Provider interface
│   ├── anthropic.go         # Anthropic provider
│   ├── openai.go            # OpenAI provider
│   └── openrouter.go        # OpenRouter provider
├── internal/
│   └── ffi/
│       ├── bindings.go      # cgo bindings
│       └── bindings.h       # C header
├── examples/
│   ├── basic/
│   │   └── main.go
│   └── advanced/
│       └── main.go
├── go.mod
├── go.sum
└── README.md
```

### API Design

#### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/reasonkit/reasonkit-go"
)

func main() {
    // Initialize client
    client, err := reasonkit.NewClient("sk-ant-...")
    if err != nil {
        log.Fatal(err)
    }

    // Simple thinking request
    result, err := client.Think(context.Background(), reasonkit.ThinkRequest{
        Query:   "Should we adopt microservices architecture?",
        Profile: "balanced",
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result.Conclusion)
    fmt.Printf("Confidence: %.1f%%\n", result.Confidence*100)

    // Access reasoning trace
    for _, step := range result.Trace.Steps {
        fmt.Printf("[%s] %s\n", step.Name, step.Summary)
    }
}
```

#### Advanced Usage

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "github.com/reasonkit/reasonkit-go"
)

func main() {
    // Custom configuration
    config := reasonkit.Config{
        Provider:      "anthropic",
        Model:         "claude-sonnet-4",
        Temperature:   0.7,
        MaxTokens:     4000,
        Timeout:       60 * time.Second,
        RetryAttempts: 3,
    }

    client, err := reasonkit.NewClientWithConfig(config)
    if err != nil {
        log.Fatal(err)
    }

    // Execute specific ThinkTools
    result, err := client.Think(context.Background(), reasonkit.ThinkRequest{
        Query: "Evaluate this startup idea: AI-powered code review",
        ThinkTools: []string{
            reasonkit.GigaThink,
            reasonkit.LaserLogic,
            reasonkit.ProofGuard,
        },
        Budget:    "60s",
        SaveTrace: true,
    })
    if err != nil {
        log.Fatal(err)
    }

    // Access detailed results
    fmt.Printf("Perspectives: %d\n", len(result.Perspectives))
    fmt.Printf("Fallacies: %v\n", result.Fallacies)
    fmt.Printf("Verification: %s\n", result.VerificationStatus)

    // Save execution trace
    if err := result.Trace.Save("./traces/startup-analysis.json"); err != nil {
        log.Fatal(err)
    }
}
```

#### Concurrent Usage

```go
package main

import (
    "context"
    "fmt"
    "log"
    "sync"

    "github.com/reasonkit/reasonkit-go"
)

func main() {
    client, err := reasonkit.NewClient("sk-ant-...")
    if err != nil {
        log.Fatal(err)
    }

    queries := []string{
        "What are the risks of cloud migration?",
        "How should we approach security?",
        "What's the optimal team structure?",
    }

    var wg sync.WaitGroup
    results := make(chan *reasonkit.ThinkResult, len(queries))

    for _, query := range queries {
        wg.Add(1)
        go func(q string) {
            defer wg.Done()
            result, err := client.Think(context.Background(), reasonkit.ThinkRequest{
                Query:   q,
                Profile: "quick",
            })
            if err != nil {
                log.Printf("Error: %v", err)
                return
            }
            results <- result
        }(query)
    }

    wg.Wait()
    close(results)

    for result := range results {
        fmt.Printf("Conclusion: %s\n\n", result.Conclusion)
    }
}
```

### Type Definitions

```go
// client.go
package reasonkit

import (
    "context"
    "time"
)

// Client is the main ReasonKit client.
type Client struct {
    config   Config
    provider Provider
}

// NewClient creates a new ReasonKit client with the given API key.
func NewClient(apiKey string) (*Client, error) {
    return NewClientWithConfig(Config{
        APIKey:   apiKey,
        Provider: "anthropic",
    })
}

// NewClientWithConfig creates a new ReasonKit client with custom configuration.
func NewClientWithConfig(config Config) (*Client, error) {
    // Implementation
}

// Think executes a reasoning request.
func (c *Client) Think(ctx context.Context, req ThinkRequest) (*ThinkResult, error) {
    // Implementation
}

// ThinkStream returns a channel of streaming events.
func (c *Client) ThinkStream(ctx context.Context, req ThinkRequest) (<-chan StreamEvent, error) {
    // Implementation
}

// config.go
type Config struct {
    APIKey        string
    Provider      string
    Model         string
    Temperature   float64
    MaxTokens     int
    Timeout       time.Duration
    RetryAttempts int
    BaseURL       string
}

// thinktools.go
const (
    GigaThink     = "gigathink"
    LaserLogic    = "laserlogic"
    BedRock       = "bedrock"
    ProofGuard    = "proofguard"
    BrutalHonesty = "brutalhonesty"
)

// profiles.go
const (
    ProfileQuick    = "quick"
    ProfileBalanced = "balanced"
    ProfileDeep     = "deep"
    ProfileParanoid = "paranoid"
)

// results.go
type ThinkRequest struct {
    Query      string
    Protocol   string
    Profile    string
    ThinkTools []string
    Budget     string
    SaveTrace  bool
    Context    string
}

type ThinkResult struct {
    Conclusion         string
    Confidence         float64
    Perspectives       []string
    Fallacies          []string
    VerificationStatus string
    Trace              *ExecutionTrace
}

// traces.go
type ExecutionTrace struct {
    ProtocolID  string
    Steps       []StepTrace
    StartedAt   time.Time
    CompletedAt time.Time
    TotalTokens int
    TotalCostUSD float64
}

func (t *ExecutionTrace) Save(path string) error {
    // Implementation
}

type StepTrace struct {
    StepID      string
    Name        string
    StartedAt   time.Time
    CompletedAt time.Time
    Status      string
    Input       string
    Output      string
    TokensUsed  int
}

// errors.go
type Error struct {
    Code    string
    Message string
    Cause   error
}

func (e *Error) Error() string {
    return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

var (
    ErrConfiguration = &Error{Code: "CONFIGURATION_ERROR"}
    ErrProvider      = &Error{Code: "PROVIDER_ERROR"}
    ErrRateLimit     = &Error{Code: "RATE_LIMIT_ERROR"}
    ErrTimeout       = &Error{Code: "TIMEOUT_ERROR"}
    ErrBudgetExceeded = &Error{Code: "BUDGET_EXCEEDED_ERROR"}
    ErrValidation    = &Error{Code: "VALIDATION_ERROR"}
)
```

### go.mod

```go
module github.com/reasonkit/reasonkit-go

go 1.21

require (
    golang.org/x/sync v0.6.0
)
```

---

## 6. Common SDK Features

### Configuration

All SDKs MUST support these configuration options:

| Option           | Type     | Default          | Description              |
| ---------------- | -------- | ---------------- | ------------------------ |
| `api_key`        | string   | env var          | API key for LLM provider |
| `provider`       | string   | `"anthropic"`    | LLM provider name        |
| `model`          | string   | provider default | Model identifier         |
| `temperature`    | float    | `0.7`            | Generation temperature   |
| `max_tokens`     | int      | `4000`           | Max output tokens        |
| `timeout`        | duration | `60s`            | Request timeout          |
| `retry_attempts` | int      | `3`              | Retry count on failure   |
| `base_url`       | string   | provider default | Custom API endpoint      |

### Environment Variables

```bash
# Primary API key (auto-detected by provider)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...

# ReasonKit-specific
REASONKIT_PROVIDER=anthropic
REASONKIT_MODEL=claude-sonnet-4
REASONKIT_TIMEOUT=60
REASONKIT_LOG_LEVEL=info
```

### Error Handling Patterns

#### Python

```python
from reasonkit.exceptions import (
    ReasonKitError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    BudgetExceededError,
)

try:
    result = rk.think(query="...", profile="deep")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e.budget_type} limit {e.limit}, used {e.used}")
except TimeoutError:
    print("Request timed out")
except ProviderError as e:
    print(f"Provider error: {e}")
except ReasonKitError as e:
    print(f"General error: {e}")
```

#### TypeScript

```typescript
import {
  ReasonKitError,
  ProviderError,
  RateLimitError,
  TimeoutError,
  BudgetExceededError,
} from "@reasonkit/core";

try {
  const result = await rk.think({ query: "...", profile: "deep" });
} catch (error) {
  if (error instanceof RateLimitError) {
    console.log(`Rate limited. Retry after ${error.retryAfter}s`);
  } else if (error instanceof BudgetExceededError) {
    console.log(`Budget exceeded: ${error.budgetType}`);
  } else if (error instanceof TimeoutError) {
    console.log("Request timed out");
  } else if (error instanceof ProviderError) {
    console.log(`Provider error: ${error.message}`);
  } else if (error instanceof ReasonKitError) {
    console.log(`General error: ${error.message}`);
  }
}
```

#### Go

```go
result, err := client.Think(ctx, req)
if err != nil {
    var rateLimitErr *reasonkit.RateLimitError
    var budgetErr *reasonkit.BudgetExceededError

    switch {
    case errors.As(err, &rateLimitErr):
        log.Printf("Rate limited. Retry after %v", rateLimitErr.RetryAfter)
    case errors.As(err, &budgetErr):
        log.Printf("Budget exceeded: %s", budgetErr.BudgetType)
    case errors.Is(err, context.DeadlineExceeded):
        log.Println("Request timed out")
    default:
        log.Printf("Error: %v", err)
    }
}
```

### Logging

All SDKs MUST support configurable logging levels:

| Level   | Description                   |
| ------- | ----------------------------- |
| `error` | Only errors                   |
| `warn`  | Errors and warnings           |
| `info`  | General operational info      |
| `debug` | Detailed debugging info       |
| `trace` | Full request/response logging |

#### Python

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("reasonkit")

# Or configure via environment
# REASONKIT_LOG_LEVEL=debug
```

#### TypeScript

```typescript
import { ReasonKit, LogLevel } from "@reasonkit/core";

const rk = new ReasonKit({
  apiKey: "...",
  logLevel: LogLevel.Debug,
});
```

#### Go

```go
import "github.com/reasonkit/reasonkit-go/log"

log.SetLevel(log.DebugLevel)
```

### Async Support

| Language | Async Pattern         | Notes                        |
| -------- | --------------------- | ---------------------------- |
| Python   | `async/await`         | AsyncReasonKit class         |
| Node.js  | `async/await`         | All methods async by default |
| Go       | goroutines + channels | ThinkStream returns channel  |

---

## 7. SDK Testing

### Test Structure

```
tests/
├── unit/
│   ├── test_client.py       # Client unit tests
│   ├── test_config.py       # Config unit tests
│   ├── test_thinktools.py   # ThinkTool unit tests
│   └── test_errors.py       # Error handling tests
├── integration/
│   ├── test_providers.py    # Provider integration tests
│   ├── test_streaming.py    # Streaming tests
│   └── test_budgets.py      # Budget enforcement tests
├── e2e/
│   ├── test_full_workflow.py  # Full workflow tests
│   └── test_cross_sdk.py      # Cross-SDK consistency
└── fixtures/
    ├── mock_responses.json  # Mock LLM responses
    └── test_traces.json     # Test execution traces
```

### Unit Tests

#### Python Example

```python
# tests/unit/test_client.py
import pytest
from unittest.mock import Mock, patch
from reasonkit import ReasonKit
from reasonkit.exceptions import ConfigurationError

class TestReasonKitClient:
    def test_init_with_api_key(self):
        rk = ReasonKit(api_key="test-key")
        assert rk.config.api_key == "test-key"

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        rk = ReasonKit()
        assert rk.config.api_key == "env-key"

    def test_init_missing_key_raises(self):
        with pytest.raises(ConfigurationError):
            ReasonKit()

    @patch("reasonkit.client.ProtocolExecutor")
    def test_think_calls_executor(self, mock_executor):
        mock_executor.return_value.execute.return_value = Mock(
            conclusion="Test conclusion",
            confidence=0.85,
        )
        rk = ReasonKit(api_key="test-key")
        result = rk.think(query="Test query", profile="balanced")
        assert result.conclusion == "Test conclusion"

    def test_list_protocols(self):
        rk = ReasonKit(api_key="test-key")
        protocols = rk.list_protocols()
        assert "gigathink" in protocols
        assert "laserlogic" in protocols

    def test_list_profiles(self):
        rk = ReasonKit(api_key="test-key")
        profiles = rk.list_profiles()
        assert "balanced" in profiles
        assert "paranoid" in profiles
```

#### TypeScript Example

```typescript
// tests/unit/client.test.ts
import { describe, it, expect, vi } from "vitest";
import { ReasonKit } from "../src/client";

describe("ReasonKit Client", () => {
  it("should initialize with API key", () => {
    const rk = new ReasonKit({ apiKey: "test-key" });
    expect(rk.config.apiKey).toBe("test-key");
  });

  it("should throw on missing API key", () => {
    expect(() => new ReasonKit({})).toThrow("API key required");
  });

  it("should call executor on think", async () => {
    const rk = new ReasonKit({ apiKey: "test-key" });
    vi.spyOn(rk["executor"], "execute").mockResolvedValue({
      conclusion: "Test conclusion",
      confidence: 0.85,
      perspectives: [],
      trace: null,
    });

    const result = await rk.think({
      query: "Test query",
      profile: "balanced",
    });

    expect(result.conclusion).toBe("Test conclusion");
  });
});
```

#### Go Example

```go
// client_test.go
package reasonkit

import (
    "context"
    "testing"
)

func TestNewClient(t *testing.T) {
    client, err := NewClient("test-key")
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if client.config.APIKey != "test-key" {
        t.Error("API key not set")
    }
}

func TestNewClient_MissingKey(t *testing.T) {
    _, err := NewClient("")
    if err == nil {
        t.Error("expected error for missing API key")
    }
}

func TestClient_ListProtocols(t *testing.T) {
    client, _ := NewClient("test-key")
    protocols := client.ListProtocols()

    expected := []string{"gigathink", "laserlogic", "bedrock", "proofguard", "brutalhonesty"}
    for _, p := range expected {
        found := false
        for _, actual := range protocols {
            if actual == p {
                found = true
                break
            }
        }
        if !found {
            t.Errorf("missing protocol: %s", p)
        }
    }
}
```

### Integration Tests

```python
# tests/integration/test_providers.py
import pytest
import os
from reasonkit import ReasonKit

@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
class TestAnthropicProvider:
    def test_think_with_anthropic(self):
        rk = ReasonKit(provider="anthropic")
        result = rk.think(
            query="What is 2+2?",
            profile="quick"
        )
        assert "4" in result.conclusion.lower()
        assert result.confidence > 0.5

    def test_streaming_with_anthropic(self):
        rk = ReasonKit(provider="anthropic")
        events = list(rk.think_stream(
            query="What is 2+2?",
            profile="quick"
        ))
        assert len(events) > 0
        assert events[-1].type == "final"
```

### E2E Tests (Cross-SDK Consistency)

```python
# tests/e2e/test_cross_sdk.py
"""
Cross-SDK consistency tests.
Verifies that Python, Node.js, and Go SDKs produce equivalent results.
"""
import subprocess
import json
import pytest

QUERY = "What are the benefits of microservices?"
PROFILE = "quick"

def run_python_sdk():
    result = subprocess.run(
        ["python", "-c", f"""
from reasonkit import ReasonKit
rk = ReasonKit(api_key='test')
result = rk.think(query='{QUERY}', profile='{PROFILE}', mock=True)
print(json.dumps({{'conclusion': result.conclusion, 'confidence': result.confidence}}))
"""],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

def run_node_sdk():
    result = subprocess.run(
        ["node", "-e", f"""
const {{ ReasonKit }} = require('@reasonkit/core');
const rk = new ReasonKit({{ apiKey: 'test', mock: true }});
rk.think({{ query: '{QUERY}', profile: '{PROFILE}' }}).then(r =>
    console.log(JSON.stringify({{ conclusion: r.conclusion, confidence: r.confidence }}))
);
"""],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

def run_go_sdk():
    result = subprocess.run(
        ["go", "run", "./cmd/test", "-query", QUERY, "-profile", PROFILE, "-mock"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

class TestCrossSDKConsistency:
    @pytest.mark.e2e
    def test_mock_results_match(self):
        python_result = run_python_sdk()
        node_result = run_node_sdk()
        go_result = run_go_sdk()

        # Mock results should be identical
        assert python_result == node_result == go_result

    @pytest.mark.e2e
    def test_error_codes_match(self):
        # Test that all SDKs return the same error codes
        pass
```

---

## 8. Documentation

### README Template

````markdown
# ReasonKit Python SDK

> Structured AI reasoning framework - Turn Prompts into Protocols

[![PyPI](https://img.shields.io/pypi/v/reasonkit)](https://pypi.org/project/reasonkit/)
[![Python](https://img.shields.io/pypi/pyversions/reasonkit)](https://pypi.org/project/reasonkit/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

## Installation

```bash
pip install reasonkit
```
````

## Quick Start

```python
from reasonkit import ReasonKit

rk = ReasonKit(api_key="sk-ant-...")

result = rk.think(
    query="Should we adopt microservices?",
    profile="balanced"
)

print(result.conclusion)
```

## Documentation

- [Getting Started](https://docs.reasonkit.sh/python/getting-started)
- [API Reference](https://docs.reasonkit.sh/python/api)
- [Examples](https://github.com/reasonkit/reasonkit-python/tree/main/examples)

## License

Apache 2.0 - see [LICENSE](LICENSE)

```

### API Reference Structure

```

docs/
├── index.md # Overview
├── getting-started.md # Quick start guide
├── installation.md # Installation instructions
├── configuration.md # Configuration options
├── thinktools/
│ ├── index.md # ThinkTools overview
│ ├── gigathink.md # GigaThink documentation
│ ├── laserlogic.md # LaserLogic documentation
│ ├── bedrock.md # BedRock documentation
│ ├── proofguard.md # ProofGuard documentation
│ └── brutalhonesty.md # BrutalHonesty documentation
├── profiles/
│ ├── index.md # Profiles overview
│ └── custom.md # Custom profiles
├── providers/
│ ├── index.md # Providers overview
│ ├── anthropic.md # Anthropic configuration
│ ├── openai.md # OpenAI configuration
│ └── openrouter.md # OpenRouter configuration
├── api-reference/
│ ├── client.md # Client class reference
│ ├── results.md # Result types
│ ├── traces.md # Execution traces
│ └── errors.md # Error types
├── guides/
│ ├── async.md # Async usage guide
│ ├── streaming.md # Streaming guide
│ ├── budgets.md # Budget management
│ └── testing.md # Testing guide
└── migration/
└── v0-to-v1.md # Migration guides

````

### Auto-Generated API Docs

#### Python (pdoc/mkdocstrings)

```yaml
# mkdocs.yml
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_signature_annotations: true
            members_order: source
````

#### TypeScript (TypeDoc)

```json
// typedoc.json
{
  "entryPoints": ["src/index.ts"],
  "out": "docs/api",
  "plugin": ["typedoc-plugin-markdown"],
  "readme": "none"
}
```

#### Go (godoc)

```bash
# Generate documentation
godoc -http=:6060

# Or use pkgsite
go install golang.org/x/pkgsite/cmd/pkgsite@latest
pkgsite -http=:6060
```

---

## 9. Publishing

### Python (PyPI)

#### Build Process

```bash
# Install build tools
pip install maturin twine

# Build wheel
maturin build --release

# Build source distribution
maturin sdist

# Check package
twine check dist/*
```

#### Publishing Workflow

```yaml
# .github/workflows/publish-python.yml
name: Publish Python SDK

on:
  push:
    tags:
      - "python-v*"

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          args: --release --out dist

      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.python-version }}
          path: dist

  publish:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Download wheels
        uses: actions/download-artifact@v4
        with:
          pattern: wheels-*
          merge-multiple: true
          path: dist

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### Node.js (npm)

#### Build Process

```bash
# Install dependencies
npm install

# Build TypeScript and native modules
npm run build

# Pack for inspection
npm pack

# Verify package contents
tar -tzf reasonkit-core-*.tgz
```

#### Publishing Workflow

```yaml
# .github/workflows/publish-node.yml
name: Publish Node.js SDK

on:
  push:
    tags:
      - "node-v*"

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node-version: [18, 20, 22]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Test
        run: npm test

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: bindings-${{ matrix.os }}-node${{ matrix.node-version }}
          path: |
            *.node
            dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: bindings-*
          merge-multiple: true

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          registry-url: https://registry.npmjs.org

      - name: Publish
        run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

### Go (pkg.go.dev)

#### Module Setup

```bash
# Initialize module
go mod init github.com/reasonkit/reasonkit-go

# Tag release
git tag v0.1.0
git push origin v0.1.0

# pkg.go.dev automatically indexes tagged releases
```

#### Publishing Process

1. Ensure `go.mod` has correct module path
2. Create annotated tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
3. Push tag: `git push origin v0.1.0`
4. pkg.go.dev auto-indexes within ~1 hour
5. Verify at: `https://pkg.go.dev/github.com/reasonkit/reasonkit-go@v0.1.0`

---

## 10. Versioning and Compatibility

### Semantic Versioning

All SDKs follow [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH

MAJOR: Breaking API changes
MINOR: New features (backward compatible)
PATCH: Bug fixes (backward compatible)
```

### Version Naming Convention

| Component             | Version Format        | Example                                    |
| --------------------- | --------------------- | ------------------------------------------ |
| reasonkit-core (Rust) | `v0.1.0`              | `v0.1.0`                                   |
| Python SDK            | `0.1.0` (PyPI)        | `reasonkit==0.1.0`                         |
| Node.js SDK           | `0.1.0` (npm)         | `@reasonkit/core@0.1.0`                    |
| Go SDK                | `v0.1.0` (Go modules) | `github.com/reasonkit/reasonkit-go@v0.1.0` |

### Compatibility Matrix

| Core Version | Python SDK | Node.js SDK | Go SDK | Status  |
| ------------ | ---------- | ----------- | ------ | ------- |
| 0.1.x        | 0.1.x      | 0.1.x       | 0.1.x  | Current |
| 0.2.x        | 0.2.x      | 0.2.x       | 0.2.x  | Planned |
| 1.0.x        | 1.0.x      | 1.0.x       | 1.0.x  | Future  |

### Deprecation Policy

1. **Deprecation Notice**: Feature marked deprecated in MINOR release
2. **Warning Period**: 2 MINOR releases (minimum 3 months)
3. **Removal**: Feature removed in next MAJOR release
4. **Migration Guide**: Provided with deprecation notice

#### Deprecation Example

```python
# Python
import warnings

def old_method(self):
    warnings.warn(
        "old_method is deprecated and will be removed in v2.0. "
        "Use new_method instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method()
```

```typescript
// TypeScript
/**
 * @deprecated Use `newMethod` instead. Will be removed in v2.0.
 */
oldMethod(): void {
  console.warn('oldMethod is deprecated. Use newMethod instead.');
  return this.newMethod();
}
```

---

## 11. Developer Experience

### IDE Support

#### Python Type Stubs

```python
# reasonkit/py.typed  (empty marker file)

# reasonkit/_core.pyi (type stubs)
from typing import Optional, List, Dict, Any

class Reasoner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None: ...
    # ... full type definitions
```

#### TypeScript Declarations

```typescript
// Auto-generated from tsup/tsc
// dist/index.d.ts
export interface Config { ... }
export interface ThinkResult { ... }
export class ReasonKit { ... }
```

#### Go Documentation

```go
// Package reasonkit provides a structured AI reasoning framework.
//
// Example usage:
//
//     client, err := reasonkit.NewClient("api-key")
//     if err != nil {
//         log.Fatal(err)
//     }
//
//     result, err := client.Think(ctx, reasonkit.ThinkRequest{
//         Query: "Your question",
//         Profile: "balanced",
//     })
package reasonkit
```

### Examples Repository

```
examples/
├── python/
│   ├── basic_usage.py
│   ├── async_usage.py
│   ├── streaming.py
│   ├── custom_profiles.py
│   ├── budget_management.py
│   └── integration_langchain.py
├── node/
│   ├── basic-usage.ts
│   ├── streaming.ts
│   ├── express-api.ts
│   └── nextjs-integration/
├── go/
│   ├── basic/
│   ├── concurrent/
│   └── gin-api/
└── real-world/
    ├── code-review-bot/
    ├── research-assistant/
    └── decision-support-system/
```

### Copy-Paste Snippets

Provide ready-to-use snippets for common use cases:

```python
# Quick analysis
result = rk.think("Your question", profile="quick")

# Thorough analysis with trace
result = rk.think("Your question", profile="deep", save_trace=True)

# Cost-controlled analysis
result = rk.think("Your question", profile="balanced", budget="$0.10")

# Specific ThinkTools
result = rk.think("Your question", thinktools=["gigathink", "proofguard"])
```

---

## 12. Quality Gates

### SDK Release Checklist

Before releasing any SDK version:

- [ ] All unit tests pass (100% pass rate)
- [ ] All integration tests pass
- [ ] Type coverage complete (no `any` leaks)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number follows SemVer
- [ ] Cross-SDK consistency verified
- [ ] Security audit passed
- [ ] Performance benchmarks acceptable
- [ ] README examples tested

### CI/CD Quality Gates

```yaml
# .github/workflows/sdk-quality.yml
name: SDK Quality Gates

on: [push, pull_request]

jobs:
  python-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Lint (ruff)
        run: ruff check .
      - name: Type check (mypy)
        run: mypy src/
      - name: Test
        run: pytest --cov=reasonkit --cov-report=xml
      - name: Coverage gate (>80%)
        run: |
          coverage report --fail-under=80

  node-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
      - name: Install dependencies
        run: npm ci
      - name: Lint (eslint)
        run: npm run lint
      - name: Type check (tsc)
        run: npm run typecheck
      - name: Test
        run: npm run test:coverage
      - name: Coverage gate
        run: |
          # Verify coverage is >80%

  go-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: "1.21"
      - name: Lint (golangci-lint)
        uses: golangci/golangci-lint-action@v4
      - name: Test
        run: go test -v -race -coverprofile=coverage.out ./...
      - name: Coverage gate
        run: |
          go tool cover -func=coverage.out | grep total | awk '{print $3}'
```

### Performance Benchmarks

Track SDK overhead compared to direct Rust FFI:

| Operation        | Rust Core | Python SDK | Node.js SDK | Go SDK |
| ---------------- | --------- | ---------- | ----------- | ------ |
| Think (quick)    | 100ms     | 105ms      | 103ms       | 102ms  |
| Think (balanced) | 500ms     | 510ms      | 505ms       | 503ms  |
| Streaming start  | 50ms      | 55ms       | 52ms        | 51ms   |
| Memory overhead  | -         | +5MB       | +10MB       | +2MB   |

**Target**: <10% overhead for all operations.

---

## Appendix A: SDK Repository Structure

### Python SDK Repository

```
reasonkit-python/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── publish.yml
│       └── docs.yml
├── src/
│   └── reasonkit/
│       └── ...
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── docs/
├── examples/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE
└── CONTRIBUTING.md
```

### Node.js SDK Repository

```
reasonkit-node/
├── .github/
│   └── workflows/
├── src/
│   └── ...
├── native/
│   ├── Cargo.toml
│   └── src/
├── tests/
├── docs/
├── examples/
├── package.json
├── tsconfig.json
├── README.md
├── CHANGELOG.md
├── LICENSE
└── CONTRIBUTING.md
```

### Go SDK Repository

```
reasonkit-go/
├── .github/
│   └── workflows/
├── cmd/
│   └── example/
├── internal/
│   └── ffi/
├── examples/
├── testdata/
├── client.go
├── client_test.go
├── config.go
├── ...
├── go.mod
├── go.sum
├── README.md
├── CHANGELOG.md
├── LICENSE
└── CONTRIBUTING.md
```

---

## Appendix B: Quick Reference

### SDK Installation Commands

```bash
# Python
pip install reasonkit
# or
uv pip install reasonkit

# Node.js
npm install @reasonkit/core
# or
yarn add @reasonkit/core
# or
pnpm add @reasonkit/core

# Go
go get github.com/reasonkit/reasonkit-go
```

### Quick Start Code

```python
# Python
from reasonkit import ReasonKit
rk = ReasonKit(api_key="...")
result = rk.think("Your question", profile="balanced")
```

```typescript
// Node.js
import { ReasonKit } from "@reasonkit/core";
const rk = new ReasonKit({ apiKey: "..." });
const result = await rk.think({ query: "Your question", profile: "balanced" });
```

```go
// Go
client, _ := reasonkit.NewClient("...")
result, _ := client.Think(ctx, reasonkit.ThinkRequest{
    Query: "Your question",
    Profile: "balanced",
})
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-12-28
**Maintainer:** ReasonKit Team
**Review Schedule:** Quarterly or on major SDK release
