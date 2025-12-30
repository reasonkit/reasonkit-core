# ReasonKit Public API Surface Design

> **Target:** v1.0.0
> **Scope:** Core traits and interfaces for ThinkTools

## 1. Core Philosophy

The ReasonKit public API is designed for **extensibility without fragility**. We expose interfaces that allow developers to build custom ThinkTools while keeping the core execution engine internal and safe.

**Principles:**

- **Trait-First:** Behavior is defined by traits (`Protocol`, `Step`, `Executor`).
- **Type-Safe:** Leverage Rust's type system to prevent invalid protocol states.
- **Stable:** The public surface area is minimal to allow internal evolution.

## 2. Core Traits

### 2.1 The `ThinkTool` Trait

The fundamental unit of reasoning. Any struct implementing this can be executed by the engine.

```rust
pub trait ThinkTool: Send + Sync {
    /// Unique identifier for this tool
    fn id(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// The core execution logic
    ///
    /// Takes a context and returns a step-by-step trace of the reasoning process.
    async fn execute(&self, input: &ProtocolInput, ctx: &ExecutionContext) -> Result<ProtocolOutput>;

    /// Define required capabilities (e.g., "internet_access", "code_execution")
    fn requirements(&self) -> Vec<Capability>;
}
```

### 2.2 The `Protocol` Definition

For declarative protocols (TOML/YAML based), we expose a builder interface.

```rust
pub struct ProtocolBuilder {
    id: String,
    steps: Vec<ProtocolStep>,
}

impl ProtocolBuilder {
    pub fn new(id: impl Into<String>) -> Self;
    pub fn add_step(mut self, step: ProtocolStep) -> Self;
    pub fn build(self) -> Result<Protocol>;
}
```

## 3. Public Types

### 3.1 Input & Output

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolInput {
    pub query: String,
    pub context: Option<String>,
    pub constraints: Option<Vec<String>>,
    // extensible metadata
    pub meta: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolOutput {
    pub success: bool,
    pub confidence: f64,
    pub data: Value, // Structured result
    pub trace_id: String,
    pub duration_ms: u64,
}
```

### 3.2 Execution Context

Provides access to resources during execution.

```rust
pub struct ExecutionContext {
    pub llm: Box<dyn LlmClient>,
    pub storage: Option<Box<dyn StorageProvider>>,
    pub logger: Logger,
}
```

## 4. Extension Points

### 4.1 Custom Steps

Developers can implement custom reasoning steps in Rust.

```rust
pub trait CustomStep: Send + Sync {
    async fn run(&self, input: &StepInput, ctx: &ExecutionContext) -> Result<StepOutput>;
}
```

### 4.2 Custom Providers

While ReasonKit ships with standard providers (OpenAI, Anthropic), users can implement the `LlmProvider` trait for custom models.

```rust
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse>;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}
```

## 5. Security Boundary

- **Sandboxing:** Custom steps execute within the same process but should be logically isolated.
- **Secrets:** API keys are never passed directly to custom steps; the `ExecutionContext` handles authentication internally via the `LlmClient`.

## 6. Stability Guarantee

We follow semantic versioning.

- `reasonkit::thinktool` - Stable
- `reasonkit::experimental` - Unstable, subject to change.
