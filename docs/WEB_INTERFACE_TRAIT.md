# Web Browser Adapter Trait - Design Documentation

**Version**: 1.0.0
**Date**: 2026-01-01
**Status**: Implemented & Tested
**Location**: `src/web_interface.rs`

## Overview

The `WebBrowserAdapter` trait provides a unified abstraction layer for reasonkit-core to interface with reasonkit-web's browser automation and content extraction capabilities.

### Core Purpose

- **Decouple** reasonkit-core from specific web service implementations
- **Support multiple bindings** (MCP, HTTP, FFI, etc.)
- **Enable local and remote** browser automation
- **Provide type-safe** async interfaces with comprehensive error handling

### Architecture Diagram

```
┌─────────────────┐
│  ReasonKit Core │
│    (Rust)       │
└────────┬────────┘
         │
         │ uses WebBrowserAdapter trait
         │
         ├──────────────────────────────┐
         │                              │
    ┌────▼──────┐         ┌─────────────▼────┐
    │ McpWeb     │         │ HttpWebAdapter   │
    │ Adapter    │         │ (JSON-RPC HTTP)  │
    └────┬──────┘         └─────────┬────────┘
         │                         │
         │ MCP stdio              │ HTTP
         │                         │
    ┌────▼──────────────────────────▼────┐
    │    ReasonKit Web                    │
    │    (Browser Controller + CDP)       │
    │                                     │
    │  • Navigate                         │
    │  • Extract Content                  │
    │  • Capture Screenshots              │
    └──────────────────────────────────────┘
```

## Trait Definition

### Lifecycle Methods

#### `connect() -> WebAdapterResult<()>`

Initialize and connect to the web browser service.

**Responsibilities:**
- Validate service availability
- Initialize connection pools
- Verify API compatibility
- May start browser process (headless Chrome)
- May connect to MCP server

**Errors:**
- `WebAdapterError::NotConnected` - Service unavailable

**Example Implementation:**
```rust
async fn connect(&mut self) -> WebAdapterResult<()> {
    // Start browser process or connect to MCP server
    self.client.connect().await?;
    self.connected = true;
    Ok(())
}
```

#### `disconnect() -> WebAdapterResult<()>`

Disconnect and cleanup resources.

**Responsibilities:**
- Close browser processes gracefully
- Save session state if applicable
- Clear connection pools
- Idempotent (safe to call multiple times)

**Example Implementation:**
```rust
async fn disconnect(&mut self) -> WebAdapterResult<()> {
    self.client.disconnect().await?;
    self.connected = false;
    Ok(())
}
```

#### `is_connected() -> bool`

Check if adapter is currently connected (synchronous).

### Navigation Methods

#### `navigate(url: &str, options: NavigateOptions) -> WebAdapterResult<PageHandle>`

Navigate to a URL and return a page handle.

**Parameters:**
- `url`: Target URL (must be valid HTTP/HTTPS)
- `options`:
  - `timeout_ms`: Max wait time (default: 30s)
  - `wait_until`: Event to wait for (`Load`, `DomContentLoaded`, `NetworkIdle`)
  - `inject_js`: JavaScript to execute after load
  - `headers`: Custom request headers
  - `user_agent`: User agent override
  - `viewport`: Viewport dimensions
  - `follow_redirects`: Follow HTTP redirects

**Returns:**
- `PageHandle` with unique ID, URL, title, and activity status

**Errors:**
- `WebAdapterError::InvalidUrl` - Malformed URL
- `WebAdapterError::NavigationTimeout` - Exceeded timeout
- `WebAdapterError::NavigationFailed` - Network/SSL/HTTP errors

**Implementation Notes:**
- MUST validate URL format
- MUST respect timeout_ms
- SHOULD handle redirects
- SHOULD inject JavaScript after page load
- MUST return handle with unique ID

**Example Usage:**
```rust
let page = adapter.navigate(
    "https://example.com",
    NavigateOptions {
        timeout_ms: 30000,
        wait_until: NavigateWaitEvent::Load,
        viewport: Some((1920, 1080)),
        ..Default::default()
    }
).await?;

println!("Loaded: {}", page.url);
```

#### `go_back() -> WebAdapterResult<PageHandle>`

Go back in browser history.

#### `go_forward() -> WebAdapterResult<PageHandle>`

Go forward in browser history.

#### `reload() -> WebAdapterResult<PageHandle>`

Reload current page.

### Content Extraction Methods

#### `extract_content(page: &PageHandle, options: ExtractOptions) -> WebAdapterResult<ExtractedContent>`

Extract content from a page.

**Parameters:**
- `page`: Page handle to extract from
- `options`:
  - `content_selector`: CSS selector for main content (optional, auto-detect)
  - `extract_links`: Include links in output (default: true)
  - `extract_images`: Include images (default: false)
  - `extract_structured_data`: Extract JSON-LD/microdata (default: false)
  - `remove_scripts`: Strip script/style tags (default: true)
  - `min_text_length`: Minimum text chunk size (default: 20)
  - `detect_language`: Language detection (default: false)
  - `custom_js`: Custom extraction JavaScript

**Returns:**
- `ExtractedContent`:
  - `text`: Main body text (normalized)
  - `html`: Extracted HTML (optional)
  - `links`: Vector of extracted links
  - `images`: Vector of extracted images
  - `metadata`: Page metadata (title, description, og tags)
  - `structured_data`: JSON-LD or microdata (optional)
  - `language`: Detected language code
  - `confidence`: Extraction confidence (0.0-1.0)

**Errors:**
- `WebAdapterError::ExtractionFailed` - Extraction failed
- `WebAdapterError::InvalidSelector` - Bad CSS selector
- `WebAdapterError::JavaScriptError` - Custom JS failed

**Implementation Notes:**
- MUST extract main text content
- SHOULD auto-detect main content area
- SHOULD normalize whitespace
- SHOULD include confidence score
- MUST handle special HTML elements (tables, code blocks)

**Example Usage:**
```rust
let content = adapter.extract_content(
    &page,
    ExtractOptions {
        extract_links: true,
        extract_images: true,
        detect_language: true,
        ..Default::default()
    }
).await?;

println!("Text: {}", content.text);
println!("Links: {:?}", content.links);
println!("Confidence: {:.2}", content.confidence);
```

#### `execute_js(page: &PageHandle, script: &str) -> WebAdapterResult<serde_json::Value>`

Execute custom JavaScript on page.

**Parameters:**
- `page`: Page to execute on
- `script`: JavaScript code

**Returns:**
- Serialized result as JSON value

**Errors:**
- `WebAdapterError::JavaScriptError` - Execution failed

**Implementation Notes:**
- MUST timeout execution (>30s)
- MUST return JSON-serializable result
- SHOULD return last expression value
- MUST handle errors gracefully

**Example Usage:**
```rust
let result = adapter.execute_js(
    &page,
    r#"document.querySelectorAll('a').length"#
).await?;

println!("Link count: {}", result);
```

#### `get_text(page: &PageHandle, selector: &str) -> WebAdapterResult<String>`

Get text content from CSS selector.

**Parameters:**
- `page`: Page to query
- `selector`: CSS selector

**Returns:**
- Text content of matched element

**Errors:**
- `WebAdapterError::InvalidSelector` - Bad selector
- `WebAdapterError::ExtractionFailed` - Element not found

### Screenshot & Capture Methods

#### `capture_screenshot(page: &PageHandle, options: CaptureOptions) -> WebAdapterResult<CapturedPage>`

Capture page as screenshot or document.

**Parameters:**
- `page`: Page to capture
- `options`:
  - `format`: Capture format (PNG, JPEG, PDF, MHTML, HTML, WebP)
  - `full_page`: Full page or viewport (default: true)
  - `timeout_ms`: Capture timeout (default: 10s)
  - `quality`: JPEG/WebP quality 0-100 (default: 80)
  - `omit_background`: Transparent background (default: false)
  - `device_scale_factor`: Pixel density (default: 1.0)
  - `delay_ms`: Wait before capture (default: 0)
  - `execute_js`: JavaScript to run before capture

**Returns:**
- `CapturedPage`:
  - `format`: Captured format
  - `data`: Raw binary data (PNG bytes, PDF bytes, etc.)
  - `mime_type`: MIME type
  - `size_bytes`: Data size
  - `metadata`: Viewport, scale factor, etc.

**Errors:**
- `WebAdapterError::CaptureFailed` - Capture failed
- `WebAdapterError::UnsupportedFormat` - Format not supported

**Implementation Notes:**
- MUST support PNG, JPEG, PDF at minimum
- MAY support MHTML, WebP, HTML if available
- MUST respect quality setting for lossy formats
- SHOULD wait for delay_ms before capture
- SHOULD execute custom JavaScript before capture
- MUST include capture metadata

**Example Usage:**
```rust
// PNG screenshot
let png = adapter.capture_screenshot(
    &page,
    CaptureOptions::default()
        .format(CaptureFormat::Png)
        .full_page(true)
).await?;

// PDF document
let pdf = adapter.capture_screenshot(
    &page,
    CaptureOptions::default()
        .format(CaptureFormat::Pdf)
).await?;

// JPEG with quality
let jpeg = adapter.capture_screenshot(
    &page,
    CaptureOptions::default()
        .format(CaptureFormat::Jpeg)
        .quality(90)
).await?;
```

### Diagnostic Methods

#### `diagnostics() -> serde_json::Value`

Get connection status and diagnostics.

**Returns:**
- JSON object with:
  - `connected`: Connection status
  - `uptime`: Session duration
  - `requests_processed`: Total requests
  - `errors`: Error count
  - Custom adapter-specific fields

#### `name() -> &str`

Get adapter name (for logging).

#### `version() -> &str`

Get adapter version.

## Error Handling

### Error Types

```rust
pub enum WebAdapterError {
    NavigationFailed { message: String },
    ExtractionFailed { message: String },
    CaptureFailed { format: CaptureFormat, message: String },
    NavigationTimeout(u64),
    InvalidUrl(String),
    NotConnected,
    ConnectionLost,
    InvalidSelector(String),
    JavaScriptError { message: String },
    UnsupportedFormat(CaptureFormat),
    NotFound(String),
    Network(String),
    Serialization(String),
    Generic(String),
}
```

### Error Handling Best Practices

1. **Transient Failures**: Implement automatic retry with exponential backoff
2. **Connection Loss**: Reconnect automatically on recoverable errors
3. **Validation**: Validate inputs before operations
4. **Logging**: Log all errors with context for debugging
5. **User Info**: Provide actionable error messages

## Implementation Examples

### 1. MCP Web Adapter (Local Server)

```rust
struct McpWebAdapter {
    client: McpClient,
    connected: bool,
}

#[async_trait]
impl WebBrowserAdapter for McpWebAdapter {
    async fn connect(&mut self) -> WebAdapterResult<()> {
        self.client = McpClient::connect("stdio").await?;
        self.connected = true;
        Ok(())
    }

    async fn navigate(
        &mut self,
        url: &str,
        options: NavigateOptions,
    ) -> WebAdapterResult<PageHandle> {
        self.client.call_tool("navigate", serde_json::json!({
            "url": url,
            "timeout_ms": options.timeout_ms,
            "wait_until": options.wait_until.to_string(),
        })).await
    }
    // ... implement other methods
}
```

### 2. HTTP Web Adapter (Remote Server)

```rust
struct HttpWebAdapter {
    client: reqwest::Client,
    endpoint: String,
    connected: bool,
}

#[async_trait]
impl WebBrowserAdapter for HttpWebAdapter {
    async fn connect(&mut self) -> WebAdapterResult<()> {
        let resp = self.client
            .get(&format!("{}/health", self.endpoint))
            .send()
            .await?;

        if resp.status().is_success() {
            self.connected = true;
            Ok(())
        } else {
            Err(WebAdapterError::NotConnected)
        }
    }

    async fn navigate(
        &mut self,
        url: &str,
        options: NavigateOptions,
    ) -> WebAdapterResult<PageHandle> {
        self.client
            .post(&format!("{}/navigate", self.endpoint))
            .json(&serde_json::json!({ "url": url, "options": options }))
            .send()
            .await?
            .json()
            .await
            .map_err(|e| WebAdapterError::Navigation(e.to_string()))
    }
    // ... implement other methods
}
```

### 3. Local FFI Adapter (Same-Process)

```rust
struct LocalWebAdapter {
    browser: Arc<BrowserController>,
    connected: bool,
}

#[async_trait]
impl WebBrowserAdapter for LocalWebAdapter {
    async fn connect(&mut self) -> WebAdapterResult<()> {
        *self.browser = BrowserController::new().await?;
        self.connected = true;
        Ok(())
    }

    async fn navigate(
        &mut self,
        url: &str,
        options: NavigateOptions,
    ) -> WebAdapterResult<PageHandle> {
        self.browser.navigate(url, options).await
    }
    // ... implement other methods
}
```

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigate_options_default() {
        let opts = NavigateOptions::default();
        assert_eq!(opts.timeout_ms, 30000);
    }

    #[tokio::test]
    async fn test_capture_options_builder() {
        let opts = CaptureOptions::default()
            .format(CaptureFormat::Jpeg)
            .quality(90);
        assert_eq!(opts.format, CaptureFormat::Jpeg);
        assert_eq!(opts.quality, Some(90));
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_full_workflow() {
    let mut adapter = create_test_adapter().await;

    // Navigate
    let page = adapter.navigate("https://example.com", Default::default()).await.unwrap();

    // Extract
    let content = adapter.extract_content(&page, Default::default()).await.unwrap();
    assert!(!content.text.is_empty());

    // Capture
    let screenshot = adapter.capture_screenshot(&page, Default::default()).await.unwrap();
    assert!(!screenshot.data.is_empty());
}
```

## Performance Considerations

### Optimization Strategies

1. **Connection Pooling**: Reuse browser instances and connections
2. **Caching**: Cache navigation results and extracted content
3. **Parallel Operations**: Execute multiple captures in parallel
4. **Resource Limits**: Set timeouts on all operations
5. **Memory Management**: Clean up resources after each operation

### Benchmarks

- Navigation: ~1-3 seconds (network dependent)
- Content Extraction: ~200-500ms
- Screenshot (PNG): ~300-800ms
- PDF Generation: ~1-2 seconds

## Migration Guide

### From reasonkit-web directly to WebBrowserAdapter

**Before:**
```rust
use reasonkit_web::browser::BrowserController;

let browser = BrowserController::new().await?;
let page = browser.navigate("https://example.com").await?;
```

**After:**
```rust
use reasonkit::web_interface::{WebBrowserAdapter, NavigateOptions};

let mut adapter = create_adapter().await?; // McpWebAdapter, HttpWebAdapter, etc.
adapter.connect().await?;
let page = adapter.navigate("https://example.com", NavigateOptions::default()).await?;
```

## Contributing

### Adding New Methods

1. Define new methods in the trait
2. Add comprehensive documentation
3. Implement in all adapters
4. Add tests in all test suites
5. Update examples

### Adding New Error Types

1. Add variant to `WebAdapterError` enum
2. Update error handling in implementations
3. Document error conditions
4. Add test cases

## Changelog

### v1.0.0 (2026-01-01)

- Initial trait design
- Core methods: navigate, extract_content, capture_screenshot
- Error handling framework
- Navigation history support
- JavaScript execution
- Comprehensive documentation
- Example implementations (Mock, MCP stub, HTTP stub)

## References

- [async-trait documentation](https://docs.rs/async-trait/)
- [thiserror documentation](https://docs.rs/thiserror/)
- [ReasonKit Core Architecture](./ARCHITECTURE.md)
- [ReasonKit Web Documentation](../reasonkit-web/docs/)

## License

Apache License 2.0 - See LICENSE file
