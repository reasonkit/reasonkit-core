# Error Messages UX Guide

> ReasonKit CLI Error Messages - User Experience Guidelines
> Version: 1.0.0 | Status: Design Specification
> Author: UX Design Agent

---

## 1. EXECUTIVE SUMMARY

Good error messages are critical for developer experience. This guide establishes comprehensive standards for error messages in the ReasonKit CLI, ensuring users can quickly understand what went wrong, why it happened, and how to fix it.

### Design Principles

| Principle      | Description                                              | Priority |
| -------------- | -------------------------------------------------------- | -------- |
| **Specific**   | Tell users exactly what went wrong, not generic messages | Critical |
| **Actionable** | Always suggest how to fix the problem                    | Critical |
| **Contextual** | Include relevant details about the failed operation      | High     |
| **Linkable**   | Point to documentation when helpful                      | High     |
| **Consistent** | Use the same structure across all errors                 | Medium   |

---

## 2. ERROR MESSAGE STRUCTURE

### 2.1 Standard Error Format

Every error message should follow this structure:

```
Error: <Short summary of what went wrong>

<Explanation with context>

How to fix:
  <Numbered steps or commands to resolve the issue>

Docs: <URL to relevant documentation>
```

### 2.2 Visual Format (Terminal with Colors)

```
Error: No LLM provider configured

The ReasonKit CLI requires an LLM provider to execute reasoning protocols.
No API key was found for any supported provider.

How to fix:
  Set an API key for your preferred provider:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key

  Or use Ollama for local models:
    export RK_PROVIDER=ollama
    ollama serve

Docs: https://reasonkit.sh/docs/configuration
```

### 2.3 Anatomy of an Error Message

```
[1] Error Type Indicator (colored red)
    "Error:" prefix for errors
    "Warning:" prefix for non-fatal issues
    "Info:" prefix for informational messages

[2] Summary Line (clear, non-technical)
    - One sentence maximum
    - No jargon or internal codes
    - Describes WHAT happened, not WHY

[3] Context Block (optional)
    - Technical details when useful
    - What was being attempted
    - Current state information

[4] Resolution Block (required for errors)
    - Concrete, actionable steps
    - Command examples when applicable
    - Alternative approaches

[5] Documentation Link (optional)
    - URL to detailed documentation
    - Error code reference
```

---

## 3. ERROR CATEGORIES AND TEMPLATES

### 3.1 Exit Codes

| Exit Code | Category            | Description                      |
| --------- | ------------------- | -------------------------------- |
| 0         | Success             | Operation completed successfully |
| 1         | General Error       | Unspecified error occurred       |
| 2         | Configuration Error | Missing or invalid configuration |
| 3         | API Error           | LLM provider API error           |
| 4         | Runtime Error       | Execution or processing error    |
| 5         | Input Error         | Invalid user input               |
| 6         | Network Error       | Connection or timeout issues     |
| 7         | Storage Error       | File or database issues          |
| 10        | Partial Success     | Some operations failed           |

### 3.2 Configuration Errors (Exit Code: 2)

#### E2001: No Provider Configured

```
Error: No LLM provider configured

How to fix:
  Set an API key for your preferred provider:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key

  Or use Ollama for local models:
    export RK_PROVIDER=ollama
    ollama serve

Docs: https://reasonkit.sh/docs/configuration
```

#### E2002: Invalid API Key Format

```
Error: Invalid API key format

The ANTHROPIC_API_KEY value does not match the expected format.
Anthropic keys should start with "sk-ant-".

Current value starts with: "sk-op..."

How to fix:
  1. Verify you're using the correct key for the provider
  2. Check for extra whitespace or quotes:
     echo $ANTHROPIC_API_KEY | cat -A

  3. Re-export the key:
     export ANTHROPIC_API_KEY="sk-ant-api03-..."

Docs: https://reasonkit.sh/docs/providers/anthropic
```

#### E2003: Config File Not Found

```
Error: Configuration file not found

Expected location: ~/.config/reasonkit/config.toml

How to fix:
  Create a configuration file:
    mkdir -p ~/.config/reasonkit
    rk-core config init

  Or specify a custom location:
    rk-core --config /path/to/config.toml think "query"

Docs: https://reasonkit.sh/docs/configuration
```

#### E2004: Invalid Config Syntax

```
Error: Invalid configuration file syntax

Parse error at line 23, column 15:
  expected "=" after key "embedding.model"

How to fix:
  1. Check the syntax at line 23 in ~/.config/reasonkit/config.toml
  2. Validate your config:
     rk-core config validate
  3. Reset to defaults if needed:
     rk-core config reset

Docs: https://reasonkit.sh/docs/configuration#syntax
```

#### E2005: Missing Required Config Value

```
Error: Missing required configuration value

The 'embedding.provider' setting is required but not set.

How to fix:
  Add to your config file (~/.config/reasonkit/config.toml):
    [embedding]
    provider = "openai"

  Or set via environment:
    export RK_EMBEDDING_PROVIDER=openai

Docs: https://reasonkit.sh/docs/configuration#embedding
```

### 3.3 API Errors (Exit Code: 3)

#### E3001: Rate Limit Exceeded

```
Error: API rate limit exceeded

The Anthropic API rate limit has been reached.

Details:
  Provider: anthropic
  Requests this minute: 60/60
  Reset in: 45 seconds

How to fix:
  1. Wait 45 seconds and retry:
     sleep 45 && rk-core think "your query"

  2. Use automatic retry with backoff:
     rk-core think "query" --retry 3

  3. Switch to a different provider:
     rk-core think "query" --provider openai

  4. Upgrade your API plan for higher limits

Docs: https://reasonkit.sh/docs/errors/rate-limits
```

#### E3002: Invalid API Key

```
Error: Invalid API key

The Anthropic API rejected the provided API key.

Details:
  Provider: anthropic
  Response: 401 Unauthorized
  Message: "Invalid API key provided"

How to fix:
  1. Verify your API key at https://console.anthropic.com/settings/keys
  2. Ensure the key has not been revoked
  3. Re-export the key:
     export ANTHROPIC_API_KEY="sk-ant-..."

  4. Test the key:
     rk-core doctor check --network

Docs: https://reasonkit.sh/docs/providers/anthropic#api-keys
```

#### E3003: Model Not Found

```
Error: Model not found

The requested model 'claude-5-opus' is not available.

Details:
  Provider: anthropic
  Requested: claude-5-opus
  Available models: claude-sonnet-4-20250514, claude-opus-4-20250514, claude-3-5-haiku-20241022

How to fix:
  Use a valid model name:
    rk-core think "query" --model claude-sonnet-4-20250514

  List available models:
    rk-core models list --provider anthropic

Docs: https://reasonkit.sh/docs/models
```

#### E3004: Request Timeout

```
Error: API request timed out

The request to the Anthropic API did not complete within 60 seconds.

Details:
  Provider: anthropic
  Endpoint: /v1/messages
  Timeout: 60s
  Query length: 2,450 tokens

How to fix:
  1. Retry with a longer timeout:
     rk-core think "query" --timeout 120

  2. Reduce the query complexity or length

  3. Check provider status:
     https://status.anthropic.com

  4. Try a faster model:
     rk-core think "query" --model claude-3-5-haiku-20241022

Docs: https://reasonkit.sh/docs/errors/timeouts
```

#### E3005: Context Length Exceeded

```
Error: Context length exceeded

The input exceeds the model's maximum context window.

Details:
  Model: claude-sonnet-4-20250514
  Input tokens: 250,000
  Maximum allowed: 200,000

How to fix:
  1. Reduce input size or split into smaller queries

  2. Use a model with larger context:
     rk-core think "query" --model gemini-2.5-pro
     (supports up to 1M tokens)

  3. Enable automatic chunking:
     rk-core think "query" --chunk-input

Docs: https://reasonkit.sh/docs/models#context-limits
```

#### E3006: Insufficient Credits

```
Error: Insufficient API credits

Your Anthropic account does not have sufficient credits.

Details:
  Provider: anthropic
  Balance: $0.00
  Estimated cost: $0.15

How to fix:
  1. Add credits at https://console.anthropic.com/settings/billing

  2. Use a different provider with available credits:
     rk-core think "query" --provider openai

  3. Use a local model (free):
     rk-core think "query" --provider ollama

Docs: https://reasonkit.sh/docs/providers#billing
```

### 3.4 Runtime Errors (Exit Code: 4)

#### E4001: Profile Not Found

```
Error: Profile not found

The reasoning profile 'extreme' does not exist.

Available profiles:
  --quick       Fast 2-step analysis (GigaThink, LaserLogic)
  --balanced    Standard 4-step chain
  --deep        Thorough 5-step analysis
  --paranoid    Maximum verification (6 steps)
  --scientific  Research-focused chain
  --decide      Decision support chain

How to fix:
  Use a valid profile:
    rk-core think "query" --profile balanced

  Or specify protocols directly:
    rk-core think "query" --protocol gigathink,laserlogic

Docs: https://reasonkit.sh/docs/profiles
```

#### E4002: Protocol Execution Failed

```
Error: Protocol execution failed

The LaserLogic protocol failed at step 'validate_logic'.

Details:
  Protocol: laserlogic
  Step: validate_logic (2 of 3)
  Error: LLM response did not match expected schema
  Trace ID: a1b2c3d4-e5f6-7890

How to fix:
  1. Retry the operation (transient LLM errors are common):
     rk-core think "query" --profile balanced

  2. View the full execution trace:
     rk-core trace view a1b2c3d4-e5f6-7890

  3. Simplify your query or use a lower temperature:
     rk-core think "query" --temperature 0.5

  4. Report persistent issues:
     https://github.com/ReasonKit/reasonkit-core/issues

Docs: https://reasonkit.sh/docs/errors/execution
```

#### E4003: Trace Write Failed

```
Error: Failed to write execution trace

Could not save the execution trace to disk.

Details:
  Trace ID: a1b2c3d4-e5f6-7890
  Path: ~/.local/share/reasonkit/traces/a1b2c3d4.json
  Error: Permission denied

How to fix:
  1. Check directory permissions:
     ls -la ~/.local/share/reasonkit/traces/

  2. Fix permissions:
     chmod 755 ~/.local/share/reasonkit/traces/

  3. Or specify a different trace directory:
     export REASONKIT_TRACE_DIR=/tmp/rk-traces
     rk-core think "query"

Docs: https://reasonkit.sh/docs/traces
```

#### E4004: Invalid Input

```
Error: Invalid input

The query cannot be empty.

How to fix:
  Provide a query string:
    rk-core think "What is chain of thought prompting?"

  Or pipe input from a file:
    cat question.txt | rk-core think --stdin

Docs: https://reasonkit.sh/docs/cli#think-command
```

#### E4005: Parse Error

```
Error: Failed to parse LLM response

The LLM response could not be parsed as the expected format.

Details:
  Protocol: gigathink
  Expected: JSON with 'perspectives' array
  Received: Markdown-formatted text
  Trace ID: a1b2c3d4-e5f6-7890

How to fix:
  1. Retry the operation (LLM responses can be inconsistent)

  2. Use structured output mode:
     rk-core think "query" --structured

  3. Lower the temperature for more consistent output:
     rk-core think "query" --temperature 0.3

  4. View the raw response:
     rk-core trace view a1b2c3d4-e5f6-7890 --raw

Docs: https://reasonkit.sh/docs/errors/parsing
```

### 3.5 Network Errors (Exit Code: 6)

#### E6001: Connection Failed

```
Error: Connection failed

Could not connect to the Anthropic API.

Details:
  Endpoint: https://api.anthropic.com/v1/messages
  Error: Connection refused

How to fix:
  1. Check your internet connection:
     curl -I https://api.anthropic.com

  2. Check if the API is available:
     https://status.anthropic.com

  3. If behind a proxy, configure it:
     export HTTPS_PROXY=http://proxy:8080

  4. Try again with retry logic:
     rk-core think "query" --retry 3

Docs: https://reasonkit.sh/docs/errors/network
```

#### E6002: DNS Resolution Failed

```
Error: DNS resolution failed

Could not resolve hostname 'api.anthropic.com'.

How to fix:
  1. Check your DNS configuration:
     nslookup api.anthropic.com

  2. Try using a public DNS:
     echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

  3. Verify your network connection

Docs: https://reasonkit.sh/docs/errors/network
```

#### E6003: SSL/TLS Error

```
Error: SSL/TLS handshake failed

Could not establish a secure connection to the API.

Details:
  Endpoint: https://api.anthropic.com
  Error: certificate verify failed

How to fix:
  1. Update your CA certificates:
     sudo apt update && sudo apt install ca-certificates
     # or on macOS:
     brew install ca-certificates

  2. Check system time (TLS requires accurate time):
     date

  3. If using a corporate proxy, install its CA certificate

Docs: https://reasonkit.sh/docs/errors/tls
```

### 3.6 Storage Errors (Exit Code: 7)

#### E7001: Index Corrupted

```
Error: Search index is corrupted

The BM25 index file appears to be corrupted or incomplete.

Details:
  Index path: ~/.local/share/reasonkit/index/bm25
  Error: Invalid header magic bytes

How to fix:
  1. Rebuild the index:
     rk-core index build --force

  2. If the problem persists, clear and rebuild:
     rk-core index clear
     rk-core index build

  3. Check disk health for hardware issues

Docs: https://reasonkit.sh/docs/storage#troubleshooting
```

#### E7002: Disk Space Full

```
Error: Insufficient disk space

Cannot write to the data directory.

Details:
  Path: ~/.local/share/reasonkit/
  Available: 52 MB
  Required: ~200 MB (estimated)

How to fix:
  1. Free up disk space

  2. Move data directory to a larger volume:
     export REASONKIT_DATA_DIR=/mnt/data/reasonkit

  3. Clean up old traces:
     rk-core traces prune --older-than 30d

Docs: https://reasonkit.sh/docs/storage
```

---

## 4. COLOR CODING

### 4.1 Color Semantics

| Element        | Color          | ANSI Code    | Usage               |
| -------------- | -------------- | ------------ | ------------------- |
| Error prefix   | Red            | `\x1b[31m`   | "Error:" text       |
| Warning prefix | Yellow         | `\x1b[33m`   | "Warning:" text     |
| Info prefix    | Cyan           | `\x1b[36m`   | "Info:" text        |
| Hint prefix    | Green          | `\x1b[32m`   | "Hint:" text        |
| Code/Commands  | Cyan           | `\x1b[36m`   | Inline commands     |
| URLs           | Blue/Underline | `\x1b[34;4m` | Documentation links |
| Emphasis       | Bold           | `\x1b[1m`    | Key information     |
| Dim            | Gray           | `\x1b[90m`   | Secondary details   |

### 4.2 Example with Colors

```
\x1b[31mError:\x1b[0m No LLM provider configured

How to fix:
  Set an API key for your preferred provider:
    \x1b[36mexport ANTHROPIC_API_KEY=your-key\x1b[0m
    \x1b[36mexport OPENAI_API_KEY=your-key\x1b[0m

  Or use Ollama for local models:
    \x1b[36mexport RK_PROVIDER=ollama\x1b[0m
    \x1b[36mollama serve\x1b[0m

Docs: \x1b[34;4mhttps://reasonkit.sh/docs/configuration\x1b[0m
```

### 4.3 No-Color Mode

When `--color never` is set or `NO_COLOR` environment variable is present:

```
Error: No LLM provider configured

How to fix:
  Set an API key for your preferred provider:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key

  Or use Ollama for local models:
    export RK_PROVIDER=ollama
    ollama serve

Docs: https://reasonkit.sh/docs/configuration
```

---

## 5. OUTPUT MODES

### 5.1 Default Mode (Human-Readable)

Standard formatted error output as shown in all examples above.

### 5.2 Quiet Mode (`--quiet`)

Minimal output for scripting:

```
Error: No LLM provider configured
```

Exit code only when used with `--exit-code-only`:

```bash
$ rk-core think "query" --quiet --exit-code-only
$ echo $?
2
```

### 5.3 Machine-Readable Mode (`--json`)

```json
{
  "error": {
    "code": "E2001",
    "category": "configuration",
    "message": "No LLM provider configured",
    "details": {
      "checked_providers": ["anthropic", "openai", "ollama"],
      "checked_env_vars": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RK_PROVIDER"]
    },
    "suggestions": [
      "Set ANTHROPIC_API_KEY environment variable",
      "Set OPENAI_API_KEY environment variable",
      "Configure Ollama as local provider"
    ],
    "docs_url": "https://reasonkit.sh/docs/configuration"
  },
  "exit_code": 2,
  "timestamp": "2025-12-28T14:30:00Z"
}
```

### 5.4 Verbose Mode (`-v`, `-vv`, `-vvv`)

Adds debugging information:

```
Error: No LLM provider configured

Debug info:
  Searched locations:
    - $ANTHROPIC_API_KEY (not set)
    - $OPENAI_API_KEY (not set)
    - $RK_PROVIDER (not set)
    - ~/.config/reasonkit/config.toml (file not found)
    - /etc/reasonkit/config.toml (file not found)

  System info:
    - OS: linux x86_64
    - ReasonKit version: 1.0.0
    - Rust version: 1.75.0

How to fix:
  Set an API key for your preferred provider:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key

Docs: https://reasonkit.sh/docs/configuration
```

---

## 6. WARNING MESSAGES

### 6.1 Warning Format

Warnings are non-fatal issues that don't stop execution:

```
Warning: RAPTOR tree is outdated

The RAPTOR index was last built 7 days ago.
Current: 12,450 chunks | Index: 11,200 chunks

Recommendation:
  Rebuild the index for better results:
    rk-core index build

Continuing with existing index...
```

### 6.2 Warning Categories

| Category      | Color  | Example                             |
| ------------- | ------ | ----------------------------------- |
| Performance   | Yellow | "Query may be slow without index"   |
| Deprecation   | Yellow | "The --format flag is deprecated"   |
| Compatibility | Yellow | "Model X may not support feature Y" |
| Quality       | Yellow | "Low confidence result (45%)"       |

### 6.3 Suppressing Warnings

```bash
# Suppress all warnings
rk-core think "query" --no-warnings

# Suppress specific warning types
rk-core think "query" --ignore-warning deprecation
```

---

## 7. INFO MESSAGES

### 7.1 Info Format

Informational messages about graceful degradation or status:

```
Info: Web search unavailable

No search API key configured.
Proceeding with knowledge base only.

To enable web search:
  export TAVILY_API_KEY="..."
  # or
  export SERPER_API_KEY="..."
```

### 7.2 Hint Format

Helpful suggestions that improve the experience:

```
Hint: Speed up future queries

Consider building the RAPTOR index for faster retrieval:
  rk-core index build --raptor

This typically improves query latency by 40-60%.
```

---

## 8. IMPLEMENTATION GUIDANCE

### 8.1 Rust Error Types

```rust
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RkError {
    // Configuration errors (exit code 2)
    #[error("No LLM provider configured")]
    NoProviderConfigured,

    #[error("Invalid API key format for {provider}")]
    InvalidApiKeyFormat { provider: String },

    #[error("Configuration file not found: {path}")]
    ConfigNotFound { path: String },

    // API errors (exit code 3)
    #[error("API rate limit exceeded")]
    RateLimitExceeded {
        provider: String,
        retry_after: Option<u64>,
    },

    #[error("Invalid API key")]
    InvalidApiKey { provider: String },

    #[error("Model not found: {model}")]
    ModelNotFound {
        model: String,
        available: Vec<String>,
    },

    // Runtime errors (exit code 4)
    #[error("Profile not found: {name}")]
    ProfileNotFound { name: String },

    #[error("Protocol execution failed")]
    ProtocolFailed {
        protocol: String,
        step: String,
        trace_id: String,
    },

    // Network errors (exit code 6)
    #[error("Connection failed to {endpoint}")]
    ConnectionFailed { endpoint: String },
}

impl RkError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::NoProviderConfigured
            | Self::InvalidApiKeyFormat { .. }
            | Self::ConfigNotFound { .. } => 2,

            Self::RateLimitExceeded { .. }
            | Self::InvalidApiKey { .. }
            | Self::ModelNotFound { .. } => 3,

            Self::ProfileNotFound { .. }
            | Self::ProtocolFailed { .. } => 4,

            Self::ConnectionFailed { .. } => 6,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            Self::NoProviderConfigured => "E2001",
            Self::InvalidApiKeyFormat { .. } => "E2002",
            Self::ConfigNotFound { .. } => "E2003",
            Self::RateLimitExceeded { .. } => "E3001",
            Self::InvalidApiKey { .. } => "E3002",
            Self::ModelNotFound { .. } => "E3003",
            Self::ProfileNotFound { .. } => "E4001",
            Self::ProtocolFailed { .. } => "E4002",
            Self::ConnectionFailed { .. } => "E6001",
        }
    }

    pub fn docs_url(&self) -> String {
        format!(
            "https://reasonkit.sh/docs/errors/{}",
            self.error_code().to_lowercase()
        )
    }
}
```

### 8.2 Error Formatter

```rust
use colored::Colorize;

pub struct ErrorFormatter {
    pub color_enabled: bool,
    pub verbose: u8,
    pub json_output: bool,
}

impl ErrorFormatter {
    pub fn format(&self, error: &RkError) -> String {
        if self.json_output {
            return self.format_json(error);
        }

        let mut output = String::new();

        // Error prefix
        if self.color_enabled {
            output.push_str(&"Error:".red().bold().to_string());
        } else {
            output.push_str("Error:");
        }
        output.push(' ');
        output.push_str(&error.to_string());
        output.push_str("\n\n");

        // Context (if verbose)
        if self.verbose > 0 {
            output.push_str(&self.format_context(error));
        }

        // How to fix
        output.push_str(&self.format_suggestions(error));

        // Docs link
        output.push_str("\nDocs: ");
        if self.color_enabled {
            output.push_str(&error.docs_url().blue().underline().to_string());
        } else {
            output.push_str(&error.docs_url());
        }
        output.push('\n');

        output
    }

    fn format_suggestions(&self, error: &RkError) -> String {
        let suggestions = match error {
            RkError::NoProviderConfigured => vec![
                "Set an API key for your preferred provider:",
                "  export ANTHROPIC_API_KEY=your-key",
                "  export OPENAI_API_KEY=your-key",
                "",
                "Or use Ollama for local models:",
                "  export RK_PROVIDER=ollama",
                "  ollama serve",
            ],
            RkError::RateLimitExceeded { retry_after, .. } => {
                let wait_msg = retry_after
                    .map(|s| format!("Wait {} seconds and retry", s))
                    .unwrap_or_else(|| "Wait and retry".to_string());
                vec![
                    &wait_msg,
                    "Use automatic retry: rk-core think \"query\" --retry 3",
                    "Switch provider: rk-core think \"query\" --provider openai",
                ]
            }
            // ... other error types
            _ => vec!["See documentation for details"],
        };

        let mut output = String::from("How to fix:\n");
        for line in suggestions {
            output.push_str("  ");
            if self.color_enabled && line.trim().starts_with("export ")
                || line.trim().starts_with("rk-core ")
            {
                output.push_str(&line.cyan().to_string());
            } else {
                output.push_str(line);
            }
            output.push('\n');
        }
        output
    }

    fn format_json(&self, error: &RkError) -> String {
        serde_json::json!({
            "error": {
                "code": error.error_code(),
                "message": error.to_string(),
                "docs_url": error.docs_url()
            },
            "exit_code": error.exit_code()
        })
        .to_string()
    }

    fn format_context(&self, error: &RkError) -> String {
        // Return context details based on error type and verbosity level
        String::new()
    }
}
```

### 8.3 Usage in CLI

```rust
fn main() {
    let result = run_cli();

    if let Err(error) = result {
        let formatter = ErrorFormatter {
            color_enabled: atty::is(atty::Stream::Stderr)
                && std::env::var("NO_COLOR").is_err(),
            verbose: cli_args.verbose,
            json_output: cli_args.json,
        };

        eprintln!("{}", formatter.format(&error));
        std::process::exit(error.exit_code());
    }
}
```

---

## 9. TESTING ERROR MESSAGES

### 9.1 Checklist for Each Error

- [ ] Message clearly states what went wrong
- [ ] At least one actionable fix is provided
- [ ] Command examples are copy-pasteable
- [ ] Exit code is appropriate
- [ ] JSON output includes all relevant fields
- [ ] No-color mode is readable
- [ ] Documentation link is valid

### 9.2 User Testing Questions

1. Can the user understand what happened without technical knowledge?
2. Can the user fix the problem without searching the internet?
3. Is the most common fix listed first?
4. Are command examples correct and complete?

---

## 10. ANTI-PATTERNS TO AVOID

### 10.1 Bad Error Messages

| Bad                    | Why              | Good                                               |
| ---------------------- | ---------------- | -------------------------------------------------- |
| "Error: E2001"         | Meaningless code | "Error: No LLM provider configured"                |
| "Something went wrong" | Too vague        | "Failed to connect to the Anthropic API"           |
| "Invalid input"        | No context       | "Invalid budget format: expected '30s' or '$0.50'" |
| "Permission denied"    | No solution      | "Permission denied. Run: chmod 755 ~/.reasonkit"   |
| "Unexpected error"     | Unhelpful        | Specific error with trace ID for debugging         |

### 10.2 Rules to Follow

| Rule                 | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| **No jargon alone**  | Codes like E2001 are fine if accompanied by plain text            |
| **No blaming**       | Say "The key format is invalid" not "You provided an invalid key" |
| **No dead ends**     | Every error must suggest at least one action                      |
| **No guessing**      | If you don't know the cause, say so and suggest diagnostics       |
| **No walls of text** | Use structure (headers, bullets) for scannability                 |

---

## 11. LOCALIZATION CONSIDERATIONS

### 11.1 Message Structure for i18n

```rust
// Error messages should be structured for easy translation
pub struct ErrorMessage {
    pub summary: &'static str,       // "No LLM provider configured"
    pub context: Option<String>,     // Dynamic context
    pub suggestions: Vec<&'static str>,
    pub docs_path: &'static str,     // "/docs/configuration"
}

// Base URL is configured separately
const DOCS_BASE: &str = "https://reasonkit.sh";
```

### 11.2 Avoiding Concatenation

```rust
// Bad: Hard to translate
format!("The {} API rejected the request", provider)

// Good: Parameterized message
messages::api_rejected(provider: &str) -> String
```

---

## VERSION HISTORY

| Version | Date       | Changes                |
| ------- | ---------- | ---------------------- |
| 1.0.0   | 2025-12-28 | Initial error UX guide |

---

_"Error messages are a conversation with the user. Make it a helpful one."_
_- ReasonKit UX Design_
