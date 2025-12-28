# Troubleshooting Guide

This guide helps you resolve common issues encountered while using ReasonKit Core.

## Configuration Issues

### "Config file not found"
**Symptoms:** ReasonKit fails to start or uses unexpected default values.
**Solution:** 
*   Ensure `config.toml` exists in your current directory or specify the path via the `REASONKIT_CONFIG` environment variable.
*   Run `rk-core --version` to verify the installation.

### "Validation error: Invalid protocol schema"
**Symptoms:** Custom ThinkTool (TOML/YAML) fails to load.
**Solution:**
*   Check for syntax errors in your TOML/YAML file.
*   Verify all required fields (`id`, `name`, `steps`) are present.
*   Ensure `version` is set to "2.0.0".

## Provider & LLM Errors

### "Embedding error: Missing API Key"
**Symptoms:** RAG operations fail during ingestion or retrieval.
**Solution:**
*   Set the appropriate environment variable for your provider (e.g., `export OPENAI_API_KEY="..."`).
*   Check the `[embedding.api]` section in your `config.toml` to ensure the `provider` matches your API key.

### "Network error: Rate limit exceeded (429)"
**Symptoms:** Protocol execution stops abruptly with a 429 error.
**Solution:**
*   Implement exponential backoff in your client application.
*   Check your provider's usage dashboard to see if you've hit your monthly or tier-based quota.
*   Increase the `step_delay_ms` in `config.toml` to slow down multi-step protocols.

## Storage & RAG Issues

### "Qdrant error: Connection refused"
**Symptoms:** Unable to index or search documents.
**Solution:**
*   If using **embedded mode**, ensure no other process is locking the data directory.
*   If using an **external server**, verify Qdrant is running and the `host`/`port` in `config.toml` are correct.
*   Check the logs for permissions issues on the `./data` directory.

### "Tantivy error: Index already exists"
**Symptoms:** Failure during initial indexing.
**Solution:**
*   Tantivy usually handles this, but if the index is corrupted, you may need to delete the `./data/indexes` directory and re-ingest your documents.

## Document Processing

### "PDF processing error: Failed to extract text"
**Symptoms:** Ingested PDF results in empty chunks.
**Solution:**
*   Ensure the PDF contains text and is not just a collection of images (OCR is not supported in the core library).
*   Check if the PDF is password-protected or encrypted.

## CLI Specifics

### "Command not found: rk"
**Symptoms:** Shell doesn't recognize the `rk` or `rk-core` command.
**Solution:**
*   Ensure `~/.cargo/bin` is in your `PATH`.
*   Run `source $HOME/.cargo/env`.

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

## Still Having Trouble?

1.  **Check the logs:** ReasonKit logs detailed error context.
2.  **Search the Docs:** Re-read the [Quickstart](QUICKSTART.md) and [ThinkTool Guide](THINKTOOLS_V2_GUIDE.md).
3.  **Community:** Check the [GitHub Discussions](https://github.com/reasonkit/reasonkit-core/discussions) for similar issues.
