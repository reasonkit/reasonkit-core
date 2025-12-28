# Security Best Practices

ReasonKit Core is designed with security and privacy as first-class citizens. This guide outlines the best practices for maintaining a secure environment when using ReasonKit.

## 1. API Key Management

ReasonKit interacts with multiple LLM providers. Protecting your API keys is critical.

### Do's:
*   **Use Environment Variables:** Prefer loading keys from environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) rather than hardcoding them in `config.toml`.
*   **Use `.env` files:** If using `.env` files for local development, ensure they are listed in your `.gitignore`.
*   **Rotate Keys Regularly:** Periodically rotate your provider API keys to minimize the impact of a potential leak.
*   **Principle of Least Privilege:** Use provider features (like OpenAI Project Keys) to restrict a key's access to only the necessary models and organizations.

### Dont's:
*   **Never commit keys to Version Control:** Avoid committing any file containing a raw API key.
*   **Avoid `--api-key` CLI flags in shared environments:** Command-line arguments can be visible in process lists (e.g., `ps aux`). Use environment variables instead.

## 2. Data Privacy & PII

ReasonKit includes built-in features to protect sensitive data.

### Local-First RAG
If your data is highly sensitive, use the **Local-First** mode:
*   Use `backend = "local"` in `storage.local` and `embedding.local`.
*   This keeps your documents and vector indexes on your physical machine, sending only processed prompts to LLMs.

### PII Stripping (Telemetry)
ReasonKit telemetry (if enabled) automatically scrubs Personally Identifiable Information (PII) using regex patterns.
*   It masks emails, credit card numbers, and suspected API keys before transmission.
*   You can configure additional custom patterns in `telemetry/privacy.rs` if needed.

## 3. Secure Protocol Definitions

When creating custom ThinkTools via TOML:
*   **Input Validation:** Use `input.required` and `input.optional` fields to define expected inputs.
*   **Prompt Injection Awareness:** Be aware that user input provided via `{{query}}` is interpolated into prompts. Always instruct the LLM to treat inputs as data, not instructions, within your `prompt_template`.
*   **Sandboxing:** ReasonKit executes protocols in a controlled engine, but the LLM output is still subject to the provider's safety filters.

## 4. Network Security

*   **HTTPS:** All provider interactions use HTTPS by default.
*   **CORS:** If running the ReasonKit API server, configure the `cors` setting in `config.toml` to only allow trusted origins.
*   **Firewalls:** If running in a VPC, ensure outgoing traffic is allowed to your chosen LLM provider endpoints.

## 5. Reporting Vulnerabilities

If you discover a security vulnerability in ReasonKit, please **do not file a public issue**.

Email your findings to: **security@reasonkit.sh**

We aim to respond to all security reports within 48 hours and provide a fix or mitigation plan as quickly as possible.
