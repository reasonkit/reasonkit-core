# ReasonKit Core Examples

This directory contains runnable examples demonstrating various features of ReasonKit Core.

## Quick Start Examples

### GLM-4.6 Integration

**`glm46_quick_start.rs`** - Complete GLM-4.6 integration example

```bash
# Set API key
export GLM46_API_KEY="your-api-key-here"

# Run example
cargo run --example glm46_quick_start
```

**Features demonstrated:**

- Basic client setup and configuration
- Chat completion with GLM-4.6
- Structured output for agent coordination
- Cost tracking and optimization
- Error handling patterns

**Prerequisites:**

- `GLM46_API_KEY` environment variable set
- GLM-4.6 API access (via OpenRouter or direct)

### Core ThinkTool Protocol

**`quick-start-core.rs`** - Basic ThinkTool protocol usage

```bash
cargo run --example quick-start-core
```

**Features demonstrated:**

- Protocol creation and execution
- Reasoning strategies
- Step-by-step reasoning process

## Integration Examples

### Memory Service

**`memory_service_example.rs`** - Memory service integration

```bash
cargo run --example memory_service_example --features memory
```

### Web Adapter

**`web_adapter_example.rs`** - Web adapter usage

```bash
cargo run --example web_adapter_example
```

### M2 Integration

**`m2_integration_example.rs`** - Minimax M2 model integration

```bash
cargo run --example m2_integration_example --features minimax
```

## Advanced Examples

### Aesthetic Demo

**`aesthetic_demo.rs`** - Aesthetic engine demonstration

```bash
cargo run --example aesthetic_demo --features aesthetic
```

### Vibe Quick Start

**`vibe_quick_start.rs`** - Vibe engine quick start

```bash
cargo run --example vibe_quick_start --features vibe
```

### Power Combo

**`powercombo.rs`** - Combined features demonstration

```bash
cargo run --example powercombo
```

## Running Examples

### Basic Run

```bash
cargo run --example <example-name>
```

### With Features

```bash
cargo run --example <example-name> --features <feature1>,<feature2>
```

### With Environment Variables

```bash
GLM46_API_KEY="your-key" cargo run --example glm46_quick_start
```

## Example Categories

| Category        | Examples                                                                  | Purpose              |
| --------------- | ------------------------------------------------------------------------- | -------------------- |
| **Quick Start** | `glm46_quick_start`, `quick-start-core`                                   | Get started quickly  |
| **Integration** | `memory_service_example`, `web_adapter_example`, `m2_integration_example` | Integration patterns |
| **Advanced**    | `aesthetic_demo`, `vibe_quick_start`, `powercombo`                        | Advanced features    |

## Contributing Examples

When adding new examples:

1. Create example file in `examples/` directory
2. Add `[[example]]` entry to `Cargo.toml`
3. Include comprehensive documentation comments
4. Add to this README
5. Test that example compiles and runs

## Documentation

For more detailed documentation:

- **GLM-4.6 Integration**: See `src/glm46/INTEGRATION_GUIDE.md`
- **API Reference**: Run `cargo doc --open`
- **Getting Started**: See `GETTING_STARTED.md`

---

**Last Updated:** 2026-01-02
