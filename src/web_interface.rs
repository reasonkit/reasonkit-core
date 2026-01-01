//! Web Browser Interface Trait for ReasonKit Core
//!
//! This module defines the abstraction layer for reasonkit-core to interact with
//! reasonkit-web's browser automation and content extraction capabilities.
//!
//! # Architecture
//!
//! ```text
//! ReasonKit Core ──▶ WebBrowserAdapter ──▶ ReasonKit Web
//!                   (trait interface)
//!        ├─ navigate()
//!        ├─ extract_content()
//!        └─ capture_screenshot()
//! ```
//!
//! # Design Principles
//!
//! - **Async-First**: All operations use `async-trait` for future compatibility
//! - **Type-Safe**: Strong types for URLs, content, and capture formats
//! - **Error Handling**: Comprehensive error types via `thiserror`
//! - **Flexible Implementations**: Support local MCP server, FFI, or HTTP bindings
//! - **Performance**: Connection pooling and caching by default
//!
//! # Example
//!
//! ```rust,no_run
//! use reasonkit::web_interface::{WebBrowserAdapter, NavigateOptions, CaptureFormat};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Implementation will use concrete adapter (e.g., McpWebAdapter, HttpWebAdapter)
//!     // let adapter = McpWebAdapter::new(config).await?;
//!
//!     // Navigate to URL
//!     // let page = adapter.navigate(
//!     //     "https://example.com",
//!     //     NavigateOptions::default(),
//!     // ).await?;
//!
//!     // Extract main content
//!     // let content = adapter.extract_content(
//!     //     &page,
//!     //     ExtractOptions::default(),
//!     // ).await?;
//!
//!     // Capture screenshot
//!     // let screenshot = adapter.capture_screenshot(
//!     //     &page,
//!     //     CaptureOptions::default().format(CaptureFormat::Png),
//!     // ).await?;
//!
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// ═══════════════════════════════════════════════════════════════════════════
// ERROR TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Web browser adapter error types
#[derive(Error, Debug)]
pub enum WebAdapterError {
    /// Navigation to URL failed
    #[error("Navigation failed: {message}")]
    NavigationFailed { message: String },

    /// Content extraction failed
    #[error("Content extraction failed: {message}")]
    ExtractionFailed { message: String },

    /// Screenshot/capture failed
    #[error("Capture failed: {format:?} - {message}")]
    CaptureFailed {
        format: CaptureFormat,
        message: String,
    },

    /// Page navigation timed out
    #[error("Navigation timeout after {0}ms")]
    NavigationTimeout(u64),

    /// Invalid URL provided
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    /// Adapter not connected
    #[error("Adapter not connected to web service")]
    NotConnected,

    /// Connection lost
    #[error("Connection to web service lost")]
    ConnectionLost,

    /// Selector parsing error
    #[error("Invalid CSS selector: {0}")]
    InvalidSelector(String),

    /// JavaScript execution failed
    #[error("JavaScript execution failed: {message}")]
    JavaScriptError { message: String },

    /// Unsupported capture format for current implementation
    #[error("Capture format not supported: {0:?}")]
    UnsupportedFormat(CaptureFormat),

    /// Resource not found (HTTP 404, etc.)
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Generic adapter error
    #[error("{0}")]
    Generic(String),
}

/// Result type alias for web adapter operations
pub type WebAdapterResult<T> = std::result::Result<T, WebAdapterError>;

// ═══════════════════════════════════════════════════════════════════════════
// DATA TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Represents a page/tab in the browser
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageHandle {
    /// Unique page identifier
    pub id: String,
    /// Current page URL
    pub url: String,
    /// Page title
    pub title: String,
    /// Whether page is still valid/accessible
    pub is_active: bool,
}

impl fmt::Display for PageHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Page({}: {})", self.id, self.url)
    }
}

/// Options for page navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigateOptions {
    /// Maximum time to wait for page load (milliseconds)
    /// Default: 30000 (30 seconds)
    pub timeout_ms: u64,

    /// Wait until event
    /// Possible values: "load", "domcontentloaded", "networkidle"
    /// Default: "load"
    pub wait_until: NavigateWaitEvent,

    /// JavaScript to execute after page load
    pub inject_js: Option<String>,

    /// Headers to send with navigation request
    pub headers: HashMap<String, String>,

    /// User agent override
    pub user_agent: Option<String>,

    /// Viewport dimensions (width, height)
    pub viewport: Option<(u32, u32)>,

    /// Follow redirects (default: true)
    pub follow_redirects: bool,
}

impl Default for NavigateOptions {
    fn default() -> Self {
        Self {
            timeout_ms: 30000,
            wait_until: NavigateWaitEvent::Load,
            inject_js: None,
            headers: HashMap::new(),
            user_agent: None,
            viewport: None,
            follow_redirects: true,
        }
    }
}

/// Page load wait event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NavigateWaitEvent {
    /// Wait for page load event
    Load,
    /// Wait for DOM content loaded event
    DomContentLoaded,
    /// Wait for network to idle (no pending requests)
    NetworkIdle,
}

impl fmt::Display for NavigateWaitEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load => write!(f, "load"),
            Self::DomContentLoaded => write!(f, "domcontentloaded"),
            Self::NetworkIdle => write!(f, "networkidle"),
        }
    }
}

/// Extracted content from a page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    /// Main body text
    pub text: String,

    /// Extracted HTML (optional, if structured extraction requested)
    pub html: Option<String>,

    /// Extracted links
    pub links: Vec<ExtractedLink>,

    /// Extracted images
    pub images: Vec<ExtractedImage>,

    /// Extracted metadata
    pub metadata: ContentMetadata,

    /// Structured data (JSON-LD, microdata, etc.)
    pub structured_data: Option<serde_json::Value>,

    /// Language detection
    pub language: Option<String>,

    /// Extraction confidence (0.0-1.0)
    pub confidence: f32,
}

/// Extracted link from content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedLink {
    /// Link text
    pub text: String,
    /// Link URL
    pub href: String,
    /// Link title attribute
    pub title: Option<String>,
}

/// Extracted image from content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedImage {
    /// Image URL
    pub src: String,
    /// Alt text
    pub alt: Option<String>,
    /// Image title
    pub title: Option<String>,
}

/// Content metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentMetadata {
    /// Page title
    pub title: Option<String>,
    /// Page description
    pub description: Option<String>,
    /// Open Graph image
    pub og_image: Option<String>,
    /// Open Graph title
    pub og_title: Option<String>,
    /// Content type (text/html, application/json, etc.)
    pub content_type: Option<String>,
    /// Character encoding
    pub charset: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Publication date
    pub publish_date: Option<String>,
    /// Custom meta tags
    pub custom_meta: HashMap<String, String>,
}

impl Default for ContentMetadata {
    fn default() -> Self {
        Self {
            title: None,
            description: None,
            og_image: None,
            og_title: None,
            content_type: None,
            charset: None,
            author: None,
            publish_date: None,
            custom_meta: HashMap::new(),
        }
    }
}

/// Options for content extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractOptions {
    /// CSS selector for main content area (optional, auto-detect if not specified)
    pub content_selector: Option<String>,

    /// Extract links (default: true)
    pub extract_links: bool,

    /// Extract images (default: false)
    pub extract_images: bool,

    /// Extract structured data (default: false)
    pub extract_structured_data: bool,

    /// Remove script and style tags (default: true)
    pub remove_scripts: bool,

    /// Minimum text length to include (default: 20 chars)
    pub min_text_length: usize,

    /// Detect language (default: false)
    pub detect_language: bool,

    /// Custom JavaScript to execute for extraction
    pub custom_js: Option<String>,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            content_selector: None,
            extract_links: true,
            extract_images: false,
            extract_structured_data: false,
            remove_scripts: true,
            min_text_length: 20,
            detect_language: false,
            custom_js: None,
        }
    }
}

/// Screenshot/capture format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum CaptureFormat {
    /// PNG image (lossless, recommended)
    Png,
    /// JPEG image (compressed, smaller file size)
    Jpeg,
    /// PDF document
    Pdf,
    /// MHTML archive (page + resources)
    Mhtml,
    /// Full HTML source
    Html,
    /// WebP image (modern, good compression)
    Webp,
}

impl fmt::Display for CaptureFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Png => write!(f, "png"),
            Self::Jpeg => write!(f, "jpeg"),
            Self::Pdf => write!(f, "pdf"),
            Self::Mhtml => write!(f, "mhtml"),
            Self::Html => write!(f, "html"),
            Self::Webp => write!(f, "webp"),
        }
    }
}

/// Captured page content/screenshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedPage {
    /// Format of captured content
    pub format: CaptureFormat,

    /// Raw captured data (PNG bytes, PDF bytes, HTML string, etc.)
    pub data: Vec<u8>,

    /// MIME type
    pub mime_type: String,

    /// File size in bytes
    pub size_bytes: usize,

    /// Capture metadata
    pub metadata: CaptureMetadata,
}

impl CapturedPage {
    /// Get captured data as string (for text formats)
    pub fn as_string(&self) -> WebAdapterResult<String> {
        String::from_utf8(self.data.clone())
            .map_err(|e| WebAdapterError::Serialization(e.to_string()))
    }
}

/// Capture metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureMetadata {
    /// Page URL that was captured
    pub url: String,

    /// Page title at time of capture
    pub title: Option<String>,

    /// Viewport width used for capture
    pub viewport_width: u32,

    /// Viewport height used for capture
    pub viewport_height: u32,

    /// Whether full page was captured (vs viewport)
    pub full_page: bool,

    /// Device scale factor (1.0 for normal, 2.0 for retina, etc.)
    pub device_scale_factor: f32,
}

/// Options for page capture/screenshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureOptions {
    /// Capture format (default: PNG)
    pub format: CaptureFormat,

    /// Capture full page or just viewport (default: true for PNG/JPEG, false for PDF)
    pub full_page: bool,

    /// Timeout for capture (milliseconds, default: 10000)
    pub timeout_ms: u64,

    /// Quality for JPEG/WebP (0-100, default: 80)
    pub quality: Option<u8>,

    /// Omit background (PNG only, default: false)
    pub omit_background: bool,

    /// Device scale factor (default: 1.0)
    pub device_scale_factor: Option<f32>,

    /// Wait delay before capture (milliseconds, default: 0)
    pub delay_ms: u64,

    /// JavaScript to execute before capture
    pub execute_js: Option<String>,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            format: CaptureFormat::Png,
            full_page: true,
            timeout_ms: 10000,
            quality: Some(80),
            omit_background: false,
            device_scale_factor: None,
            delay_ms: 0,
            execute_js: None,
        }
    }
}

impl CaptureOptions {
    /// Set capture format
    pub fn format(mut self, format: CaptureFormat) -> Self {
        self.format = format;
        self
    }

    /// Set full page capture
    pub fn full_page(mut self, full: bool) -> Self {
        self.full_page = full;
        self
    }

    /// Set quality (for JPEG/WebP)
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = Some(quality.min(100));
        self
    }

    /// Set timeout
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAIT DEFINITION
// ═══════════════════════════════════════════════════════════════════════════

/// Web browser adapter trait for reasonkit-core
///
/// Provides abstraction for browser automation and content extraction.
/// Implementations can use MCP servers, HTTP binding, FFI, or other mechanisms
/// to communicate with reasonkit-web.
///
/// # Implementing the Trait
///
/// - **McpWebAdapter**: Uses MCP stdio server (local or remote)
/// - **HttpWebAdapter**: Uses HTTP JSON-RPC binding
/// - **LocalWebAdapter**: Direct FFI binding for same-process usage
///
/// # Connection Lifecycle
///
/// 1. Create adapter with configuration
/// 2. Call `connect()` to establish connection
/// 3. Use navigation/extraction/capture methods
/// 4. Call `disconnect()` when done
///
/// Implementations MUST handle reconnection automatically on transient failures.
#[async_trait]
pub trait WebBrowserAdapter: Send + Sync {
    // ─────────────────────────────────────────────────────────────────────
    // LIFECYCLE
    // ─────────────────────────────────────────────────────────────────────

    /// Initialize and connect to the web browser service
    ///
    /// # Errors
    ///
    /// Returns `WebAdapterError::NotConnected` if service is unavailable.
    ///
    /// # Implementation Notes
    ///
    /// - May start a browser process (headless Chrome, etc.)
    /// - May connect to an existing MCP server
    /// - May verify API compatibility
    /// - Should implement connection pooling if needed
    async fn connect(&mut self) -> WebAdapterResult<()>;

    /// Disconnect from web service and cleanup resources
    ///
    /// # Implementation Notes
    ///
    /// - Should close browser processes gracefully
    /// - Should save cache/session state if applicable
    /// - Idempotent (safe to call multiple times)
    async fn disconnect(&mut self) -> WebAdapterResult<()>;

    /// Check if adapter is currently connected
    fn is_connected(&self) -> bool;

    // ─────────────────────────────────────────────────────────────────────
    // NAVIGATION
    // ─────────────────────────────────────────────────────────────────────

    /// Navigate to a URL and return a page handle
    ///
    /// # Arguments
    ///
    /// * `url` - Target URL to navigate to
    /// * `options` - Navigation options (timeout, wait event, etc.)
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `WebAdapterError::InvalidUrl` if URL is malformed
    /// - `WebAdapterError::NavigationTimeout` if timeout exceeded
    /// - `WebAdapterError::NavigationFailed` for other failures (network, SSL, 404, etc.)
    ///
    /// # Implementation Notes
    ///
    /// - MUST validate URL before navigation
    /// - MUST respect timeout_ms in options
    /// - MUST wait for specified event (load, domcontentloaded, etc.)
    /// - SHOULD respect custom headers and user agent if provided
    /// - SHOULD inject JavaScript after load if provided
    /// - SHOULD handle redirects according to options
    /// - MUST return page handle with unique ID and current URL
    async fn navigate(
        &mut self,
        url: &str,
        options: NavigateOptions,
    ) -> WebAdapterResult<PageHandle>;

    /// Go back in browser history
    ///
    /// # Errors
    ///
    /// Returns `WebAdapterError::NavigationFailed` if unable to go back.
    async fn go_back(&mut self) -> WebAdapterResult<PageHandle>;

    /// Go forward in browser history
    ///
    /// # Errors
    ///
    /// Returns `WebAdapterError::NavigationFailed` if unable to go forward.
    async fn go_forward(&mut self) -> WebAdapterResult<PageHandle>;

    /// Reload current page
    ///
    /// # Errors
    ///
    /// Returns `WebAdapterError::NavigationFailed` if unable to reload.
    async fn reload(&mut self) -> WebAdapterResult<PageHandle>;

    // ─────────────────────────────────────────────────────────────────────
    // CONTENT EXTRACTION
    // ─────────────────────────────────────────────────────────────────────

    /// Extract content from a page
    ///
    /// # Arguments
    ///
    /// * `page` - Page handle to extract from
    /// * `options` - Extraction options
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `WebAdapterError::ExtractionFailed` if extraction fails
    /// - `WebAdapterError::InvalidSelector` if custom selector is invalid
    /// - `WebAdapterError::JavaScriptError` if custom JS fails
    ///
    /// # Implementation Notes
    ///
    /// - MUST extract main content text
    /// - SHOULD auto-detect main content area if no selector provided
    /// - SHOULD extract links and images according to options
    /// - SHOULD execute custom JavaScript if provided
    /// - SHOULD detect language if requested
    /// - MUST include confidence score (0.0-1.0)
    /// - SHOULD normalize whitespace
    /// - SHOULD extract metadata (title, description, og tags)
    async fn extract_content(
        &mut self,
        page: &PageHandle,
        options: ExtractOptions,
    ) -> WebAdapterResult<ExtractedContent>;

    /// Execute custom JavaScript on page
    ///
    /// # Arguments
    ///
    /// * `page` - Page to execute on
    /// * `script` - JavaScript code to execute
    ///
    /// # Returns
    ///
    /// Serialized result of JavaScript execution as JSON value
    ///
    /// # Errors
    ///
    /// Returns `WebAdapterError::JavaScriptError` if execution fails.
    ///
    /// # Implementation Notes
    ///
    /// - MUST timeout execution if it takes too long (>30s)
    /// - MUST return JSON-serializable result
    /// - SHOULD return last expression value
    async fn execute_js(
        &mut self,
        page: &PageHandle,
        script: &str,
    ) -> WebAdapterResult<serde_json::Value>;

    /// Get text content using CSS selector
    ///
    /// # Arguments
    ///
    /// * `page` - Page to query
    /// * `selector` - CSS selector
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `WebAdapterError::InvalidSelector` if selector is invalid
    /// - `WebAdapterError::ExtractionFailed` if element not found
    async fn get_text(&mut self, page: &PageHandle, selector: &str) -> WebAdapterResult<String>;

    // ─────────────────────────────────────────────────────────────────────
    // SCREENSHOTS & CAPTURE
    // ─────────────────────────────────────────────────────────────────────

    /// Capture page as screenshot or document
    ///
    /// # Arguments
    ///
    /// * `page` - Page to capture
    /// * `options` - Capture options (format, quality, etc.)
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `WebAdapterError::CaptureFailed` if capture fails
    /// - `WebAdapterError::UnsupportedFormat` if format not supported
    ///
    /// # Implementation Notes
    ///
    /// - MUST support PNG, JPEG, PDF formats at minimum
    /// - MAY support MHTML, WebP if available
    /// - MUST respect quality setting for JPEG/WebP
    /// - MUST capture full page or viewport according to options
    /// - SHOULD wait for delay_ms before capturing
    /// - SHOULD execute custom JavaScript before capture if provided
    /// - MUST include capture metadata (viewport size, scale factor, etc.)
    async fn capture_screenshot(
        &mut self,
        page: &PageHandle,
        options: CaptureOptions,
    ) -> WebAdapterResult<CapturedPage>;

    // ─────────────────────────────────────────────────────────────────────
    // DIAGNOSTICS
    // ─────────────────────────────────────────────────────────────────────

    /// Get connection status and diagnostics
    ///
    /// # Returns
    ///
    /// JSON object with connection info, statistics, etc.
    fn diagnostics(&self) -> serde_json::Value;

    /// Get adapter name (for logging/debugging)
    fn name(&self) -> &str;

    /// Get adapter version
    fn version(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigate_options_default() {
        let opts = NavigateOptions::default();
        assert_eq!(opts.timeout_ms, 30000);
        assert_eq!(opts.wait_until, NavigateWaitEvent::Load);
        assert!(!opts.follow_redirects == false);
    }

    #[test]
    fn test_capture_options_builder() {
        let opts = CaptureOptions::default()
            .format(CaptureFormat::Jpeg)
            .quality(90)
            .full_page(false);

        assert_eq!(opts.format, CaptureFormat::Jpeg);
        assert_eq!(opts.quality, Some(90));
        assert!(!opts.full_page);
    }

    #[test]
    fn test_capture_format_display() {
        assert_eq!(CaptureFormat::Png.to_string(), "png");
        assert_eq!(CaptureFormat::Jpeg.to_string(), "jpeg");
        assert_eq!(CaptureFormat::Pdf.to_string(), "pdf");
    }

    #[test]
    fn test_page_handle_display() {
        let page = PageHandle {
            id: "page-1".to_string(),
            url: "https://example.com".to_string(),
            title: "Example".to_string(),
            is_active: true,
        };

        assert_eq!(page.to_string(), "Page(page-1: https://example.com)");
    }

    #[test]
    fn test_extract_options_default() {
        let opts = ExtractOptions::default();
        assert!(opts.extract_links);
        assert!(!opts.extract_images);
        assert!(!opts.extract_structured_data);
        assert!(opts.remove_scripts);
    }

    #[test]
    fn test_content_metadata_default() {
        let meta = ContentMetadata::default();
        assert!(meta.title.is_none());
        assert!(meta.custom_meta.is_empty());
    }

    #[test]
    fn test_navigate_wait_event_display() {
        assert_eq!(NavigateWaitEvent::Load.to_string(), "load");
        assert_eq!(
            NavigateWaitEvent::DomContentLoaded.to_string(),
            "domcontentloaded"
        );
        assert_eq!(NavigateWaitEvent::NetworkIdle.to_string(), "networkidle");
    }

    #[test]
    fn test_quality_clamping() {
        let opts = CaptureOptions::default().quality(150);
        assert_eq!(opts.quality, Some(100));
    }

    #[test]
    fn test_capture_page_as_string() {
        let page = CapturedPage {
            format: CaptureFormat::Html,
            data: "<html>test</html>".as_bytes().to_vec(),
            mime_type: "text/html".to_string(),
            size_bytes: 17,
            metadata: CaptureMetadata {
                url: "https://example.com".to_string(),
                title: None,
                viewport_width: 1024,
                viewport_height: 768,
                full_page: false,
                device_scale_factor: 1.0,
            },
        };

        assert!(page.as_string().is_ok());
        assert_eq!(page.as_string().unwrap(), "<html>test</html>");
    }
}
