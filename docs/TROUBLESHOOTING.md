# Troubleshooting Guide

This guide helps you resolve common issues encountered while using ReasonKit Core.

> **Quick Links:**
>
> - [Installation Issues](INSTALLATION_TROUBLESHOOTING.md) - Comprehensive installation troubleshooting
> - [Error UX Guide](design/ERROR_UX_GUIDE.md) - Understanding error messages
> - [Contributing Guide](../CONTRIBUTING.md) - Development setup and quality gates
> - [GitHub Issues](https://github.com/reasonkit/reasonkit-core/issues) - Report bugs
> - [Discord](https://discord.gg/reasonkit) - Community support

## Configuration Issues

### "Config file not found"

**Symptoms:** ReasonKit fails to start or uses unexpected default values.
**Solution:**

- Ensure `config.toml` exists in your current directory or specify the path via the `REASONKIT_CONFIG` environment variable.
- Run `rk-core --version` to verify the installation.

### "Validation error: Invalid protocol schema"

**Symptoms:** Custom ThinkTool (TOML/YAML) fails to load.
**Solution:**

- Check for syntax errors in your TOML/YAML file.
- Verify all required fields (`id`, `name`, `steps`) are present.
- Ensure `version` is set to "2.0.0".

## Provider & LLM Errors

### "Embedding error: Missing API Key"

**Symptoms:** RAG operations fail during ingestion or retrieval.
**Solution:**

- Set the appropriate environment variable for your provider (e.g., `export OPENAI_API_KEY="..."`).
- Check the `[embedding.api]` section in your `config.toml` to ensure the `provider` matches your API key.

### "Network error: Rate limit exceeded (429)"

**Symptoms:** Protocol execution stops abruptly with a 429 error.
**Solution:**

- Implement exponential backoff in your client application.
- Check your provider's usage dashboard to see if you've hit your monthly or tier-based quota.
- Increase the `step_delay_ms` in `config.toml` to slow down multi-step protocols.

## Storage & RAG Issues

### "Qdrant error: Connection refused"

**Symptoms:** Unable to index or search documents.
**Solution:**

- If using **embedded mode**, ensure no other process is locking the data directory.
- If using an **external server**, verify Qdrant is running and the `host`/`port` in `config.toml` are correct.
- Check the logs for permissions issues on the `./data` directory.

### "Tantivy error: Index already exists"

**Symptoms:** Failure during initial indexing.
**Solution:**

- Tantivy usually handles this, but if the index is corrupted, you may need to delete the `./data/indexes` directory and re-ingest your documents.

## Document Processing

### "PDF processing error: Failed to extract text"

**Symptoms:** Ingested PDF results in empty chunks.
**Solution:**

- Ensure the PDF contains text and is not just a collection of images (OCR is not supported in the core library).
- Check if the PDF is password-protected or encrypted.

## CLI Specifics

### "Command not found: rk"

**Symptoms:** Shell doesn't recognize the `rk` or `rk-core` command.
**Solution:**

- Ensure `~/.cargo/bin` is in your `PATH`.
- Run `source $HOME/.cargo/env`.

### Log Verbosity

If you're stuck, increase the log level to `debug` or `trace` in your `config.toml`:

```toml
[general]
log_level = "debug"
```

Or via environment variable:

```bash
export RUST_LOG=debug
rk-core think gigathink "..."
```

## Development & Build Issues

### "Compilation error: indicatif with --all-features"

**Symptoms:** `cargo test --all-features` fails with `indicatif` or `console` crate errors.

**Solution:**

- **Known Issue:** This is an upstream dependency issue documented in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
- **Workaround:** Use default features for local testing:
  ```bash
  cargo test  # Without --all-features
  ```
- **CI:** CI handles full feature testing separately.
- **Status:** Tracked in release checklist, awaiting upstream fix.

### "Clippy warnings: unused variables"

**Symptoms:** `cargo clippy -- -D warnings` fails with unused variable warnings.

**Solution:**

- Prefix unused variables with `_` (e.g., `_document_id` instead of `document_id`)
- Or use the variable (remove if truly unused)
- Run `cargo clippy --fix` for auto-fixable issues

### "Build fails: missing system dependencies"

**Symptoms:** Compilation fails with missing library errors (e.g., `openssl`, `pkg-config`).

**Solution:**

- **Linux:** `sudo apt-get install build-essential pkg-config libssl-dev` (Debian/Ubuntu)
- **macOS:** `brew install openssl pkg-config`
- **Windows:** Install [vcpkg](https://vcpkg.io/) or use WSL

See [INSTALLATION_TROUBLESHOOTING.md](INSTALLATION_TROUBLESHOOTING.md) for platform-specific details.

## Performance Issues

### "ThinkTool execution is slow"

**Symptoms:** Protocol execution takes > 10 seconds for simple queries.

**Diagnosis:**

1. Check LLM API latency: `rk-core think --profile quick "test" --verbose`
2. Check network connectivity to LLM provider
3. Review telemetry logs for bottlenecks

**Solutions:**

- Use `--profile quick` for faster execution (2-step vs 5-step)
- Check API rate limits (may be throttling)
- Verify network latency to provider
- Consider caching for repeated queries

### "High memory usage during RAG operations"

**Symptoms:** Process memory grows significantly during document ingestion.

**Solutions:**

- Use streaming/chunked processing for large documents
- Enable embedded Qdrant mode (more memory-efficient)
- Process documents in batches
- Monitor with `htop` or `top` during operations

## Probabilistic Output Issues

### "Same prompt gives different results"

**Symptoms:** Identical prompts produce varying outputs.

**Explanation:** This is expected behavior - LLMs are probabilistic by design. ReasonKit constrains this through protocols, but cannot eliminate it.

**Solutions:**

- Use `--temperature 0.0` for more deterministic outputs (if supported by provider)
- Review execution traces to see where variance occurs
- Use `--profile paranoid` for maximum consistency (95% confidence target)
- Check confidence scores in execution traces

See [README.md](../README.md#-the-probabilistic-problem-and-how-we-battle-it) for details on how ReasonKit battles probabilistic side effects.

### "Low confidence scores in results"

**Symptoms:** ThinkTool execution reports confidence < 70%.

**Solutions:**

- Use `--profile deep` or `--profile paranoid` for higher confidence
- Review ProofGuard triangulation results (may need more sources)
- Check if query is too ambiguous or requires domain expertise
- Consider breaking complex queries into smaller parts

## Telemetry & Logging

### "Telemetry database errors"

**Symptoms:** Warnings about SQLite database creation or access.

**Solutions:**

- Check permissions on `~/.local/share/reasonkit/` directory
- Verify disk space is available
- Disable telemetry if not needed: `export RK_TELEMETRY_ENABLED=false`
- Database location: `~/.local/share/reasonkit/.rk_telemetry.db` (Linux/Mac)

### "Logs not appearing"

**Symptoms:** No log output even with `RUST_LOG=debug`.

**Solutions:**

- Verify `RUST_LOG` is set: `echo $RUST_LOG`
- Check `config.toml` log level setting
- Ensure logging is enabled: `[general] log_level = "debug"`
- Try `RUST_LOG=trace` for maximum verbosity

## Still Having Trouble?

### Diagnostic Checklist

Before opening an issue, try:

1. ✅ **Updated to latest version:** `cargo install reasonkit-core --force`
2. ✅ **Checked logs:** `RUST_LOG=debug rk-core <command>`
3. ✅ **Verified installation:** `rk-core --version`
4. ✅ **Read relevant docs:** [Quickstart](QUICKSTART.md), [ThinkTool Guide](THINKTOOLS_V2_GUIDE.md)
5. ✅ **Searched existing issues:** [GitHub Issues](https://github.com/reasonkit/reasonkit-core/issues)

### Getting Help

| Resource               | Best For                         | Link                                                                   |
| ---------------------- | -------------------------------- | ---------------------------------------------------------------------- |
| **GitHub Issues**      | Bug reports, feature requests    | [Issues](https://github.com/reasonkit/reasonkit-core/issues)           |
| **GitHub Discussions** | Q&A, ideas, longer conversations | [Discussions](https://github.com/reasonkit/reasonkit-core/discussions) |
| **Discord**            | Real-time community help         | [Discord](https://discord.gg/reasonkit)                                |
| **Documentation**      | Comprehensive guides             | [docs.reasonkit.sh](https://docs.reasonkit.sh)                         |

### Reporting Issues

When reporting an issue, include:

- **Environment:** OS, Rust version, ReasonKit version
- **Error message:** Full error output
- **Steps to reproduce:** Minimal reproduction case
- **Expected vs actual:** What you expected vs what happened
- **Logs:** Relevant log output (with `RUST_LOG=debug`)

Use the [GitHub issue templates](https://github.com/reasonkit/reasonkit-core/issues/new/choose) for structured reporting.

---

**Related Documentation:**

- [Installation Troubleshooting](INSTALLATION_TROUBLESHOOTING.md) - Platform-specific installation issues
- [Error UX Guide](design/ERROR_UX_GUIDE.md) - Understanding error messages
- [Contributing Guide](../CONTRIBUTING.md) - Development setup and quality gates
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md) - Performance tuning
