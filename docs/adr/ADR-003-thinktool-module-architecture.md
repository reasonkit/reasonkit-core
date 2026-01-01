# ADR-003: ThinkTool Module Architecture

## Status

**Accepted** - 2024-12-28

## Context

ReasonKit provides structured reasoning through composable cognitive modules called "ThinkTools." The architecture must support:

1. **Composability**: Modules should chain together in various configurations
2. **Extensibility**: Users and contributors should be able to create custom modules
3. **Testability**: Each module must be independently testable
4. **Clear Contracts**: Input/output formats must be well-defined
5. **Profile Support**: Common module combinations (profiles) should be first-class

We evaluated several architectural patterns:

| Pattern                             | Pros                                            | Cons                                              |
| ----------------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| **Monolithic Engine**               | Simple implementation, optimized data flow      | Hard to extend, all-or-nothing, difficult testing |
| **Plugin System (dynamic loading)** | Maximum flexibility, runtime extension          | Complex, security concerns, version compatibility |
| **Trait-Based Modules**             | Compile-time safety, clear interfaces, testable | Requires recompilation for new modules            |
| **Actor Model**                     | Excellent concurrency, isolation                | Overhead for simple cases, complex debugging      |
| **Pipeline Pattern**                | Clear data flow, easy visualization             | Less flexible branching, linear only              |

### Core Requirements

1. **Type Safety**: Reasoning errors should be caught at compile time when possible
2. **Performance**: Module dispatch should add minimal overhead (<100us)
3. **Introspection**: Must be able to inspect module behavior for auditing
4. **Configuration**: Modules need per-invocation configuration options
5. **Error Handling**: Graceful degradation when individual modules fail

## Decision

**We will use a trait-based module architecture with profiles for common combinations.**

### Core Trait Definition

```rust
/// Core trait that all ThinkTool modules must implement
pub trait ThinkTool: Send + Sync {
    /// Unique identifier for this module
    fn id(&self) -> &'static str;

    /// Human-readable name
    fn name(&self) -> &'static str;

    /// Short description of purpose
    fn description(&self) -> &'static str;

    /// Execute reasoning step
    fn execute(&self, context: &ReasoningContext) -> Result<StepOutput, ThinkToolError>;

    /// Validate that module can run with given input
    fn validate(&self, context: &ReasoningContext) -> Result<(), ValidationError>;

    /// Estimated token usage for this step
    fn estimate_tokens(&self, context: &ReasoningContext) -> TokenEstimate;
}

/// Optional trait for modules that support streaming output
pub trait StreamingThinkTool: ThinkTool {
    fn execute_streaming(
        &self,
        context: &ReasoningContext,
    ) -> Pin<Box<dyn Stream<Item = StreamChunk> + Send>>;
}
```

### Profile System

Profiles define common module combinations with pre-configured parameters:

```rust
pub enum Profile {
    Quick,       // gt, ll - Fast 3-step analysis
    Balanced,    // gt, ll, br, pg - Standard 5-module chain
    Deep,        // gt, ll, br, pg, hr - Thorough analysis
    Scientific,  // se, ab, br, pg - Research & experiments
    Paranoid,    // gt, ll, br, pg, bh, rr - Maximum verification
    Decide,      // dm, rr, ll, hr - Decision support
    Custom(Vec<ModuleConfig>),
}

impl Profile {
    pub fn modules(&self) -> Vec<Box<dyn ThinkTool>> {
        match self {
            Profile::Quick => vec![
                Box::new(GigaThink::default()),
                Box::new(LaserLogic::default()),
            ],
            // ... other profiles
        }
    }
}
```

### Module Registry

Central registry enables discovery and runtime configuration:

```rust
pub struct ModuleRegistry {
    modules: HashMap<&'static str, Box<dyn ThinkTool>>,
    profiles: HashMap<String, Profile>,
}

impl ModuleRegistry {
    pub fn register<T: ThinkTool + 'static>(&mut self, module: T);
    pub fn get(&self, id: &str) -> Option<&dyn ThinkTool>;
    pub fn list(&self) -> Vec<ModuleInfo>;
    pub fn profile(&self, name: &str) -> Option<&Profile>;
}
```

### Built-in Modules (OSS - ReasonKit-core)

| Module        | ID   | Purpose                                            |
| ------------- | ---- | -------------------------------------------------- |
| GigaThink     | `gt` | Expansive creative thinking, 10+ perspectives      |
| LaserLogic    | `ll` | Precision deductive reasoning, fallacy detection   |
| BedRock       | `br` | First principles decomposition, axiom rebuilding   |
| ProofGuard    | `pg` | Multi-source verification, contradiction detection |
| BrutalHonesty | `bh` | Adversarial self-critique, flaw detection          |

### Advanced Modules (ReasonKit-pro)

| Module      | ID   | Purpose                                       |
| ----------- | ---- | --------------------------------------------- |
| AtomicBreak | `ab` | Point-by-point decomposition                  |
| HighReflect | `hr` | Meta-cognition, bias detection                |
| RiskRadar   | `rr` | Threat identification, probability estimation |
| DeciDomatic | `dm` | Multi-criteria analysis                       |
| SciEngine   | `se` | Scientific method automation                  |

## Consequences

### Positive

1. **Type Safety**: Trait bounds catch interface mismatches at compile time
2. **Testability**: Each module implements a common interface; easy to mock
3. **Clear Boundaries**: Input/output contracts are explicit in types
4. **Performance**: Static dispatch for hot paths; dynamic dispatch only where needed
5. **Extensibility**: Implement `ThinkTool` trait to add new modules
6. **Composability**: Profiles combine modules declaratively
7. **Introspection**: `id()`, `name()`, `description()` enable runtime discovery

### Negative

1. **Recompilation Required**: New modules require code changes and recompilation
2. **Rust Knowledge**: Contributors must understand trait-based design
3. **Binary Size**: Each module increases binary size
4. **Versioning Complexity**: Module interface changes require careful migration

### Mitigations

| Negative       | Mitigation                                                      |
| -------------- | --------------------------------------------------------------- |
| Recompilation  | Feature flags for optional modules; WASM plugin support planned |
| Rust knowledge | Thorough documentation; example module template                 |
| Binary size    | LTO optimization; optional features for advanced modules        |
| Versioning     | Semantic versioning; deprecation warnings; migration guides     |

### Extension Points

1. **Custom Modules**: Implement `ThinkTool` trait
2. **Custom Profiles**: Use `Profile::Custom` with module list
3. **Module Configuration**: Per-module config in `ModuleConfig`
4. **Plugin System**: Planned WASM-based plugins for user modules

## Related Documents

- `/home/zyxsys/RK-PROJECT/reasonkit-core/docs/THINKTOOLS_ARCHITECTURE.md` - Detailed architecture
- `/home/zyxsys/RK-PROJECT/reasonkit-core/src/thinktool/mod.rs` - Implementation
- `/home/zyxsys/RK-PROJECT/reasonkit-core/docs/CUSTOM_THINKTOOLS.md` - Extension guide

## References

- [Rust Traits](https://doc.rust-lang.org/book/ch10-02-traits.html)
- [Design Patterns in Rust](https://rust-unofficial.github.io/patterns/)
