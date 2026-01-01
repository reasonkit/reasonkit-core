//! Web browser adapter trait for core <-> web integration.
//!
//! This trait defines the contract between `reasonkit-core` and `reasonkit-web`.
//! Implementations live in `reasonkit-web`, consumers live in `reasonkit-core`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use thiserror::Error;

/// Result type for web adapter operations.
pub type WebAdapterResult<T> = Result<T, WebAdapterError>;

/// Errors that can occur during web operations.
#[derive(Error, Debug)]
pub enum WebAdapterError {
    #[error("Connection failed: {0}")]
    Connection(String),

    #[error("Navigation failed: {0}")]
    Navigation(String),

    #[error("Extraction failed: {0}")]
    Extraction(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Element not found: {0}")]
    ElementNotFound(String),

    #[error("JavaScript error: {0}")]
    JavaScript(String),

    #[error("Screenshot failed: {0}")]
    Screenshot(String),

    #[error("Not connected")]
    NotConnected,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Handle to a loaded page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageHandle {
    pub id: String,
    pub url: String,
    pub title: Option<String>,
    pub status_code: u16,
    pub load_time_ms: u64,
}

/// Options for navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigateOptions {
    pub wait_until: WaitUntil,
    pub timeout: Duration,
    pub user_agent: Option<String>,
    pub headers: Vec<(String, String)>,
    pub viewport: Option<Viewport>,
}

impl Default for NavigateOptions {
    fn default() -> Self {
        Self {
            wait_until: WaitUntil::NetworkIdle,
            timeout: Duration::from_secs(30),
            user_agent: None,
            headers: Vec::new(),
            viewport: None,
        }
    }
}

/// When to consider navigation complete.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WaitUntil {
    Load,
    DOMContentLoaded,
    NetworkIdle,
    NetworkAlmostIdle,
}

/// Viewport dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
    pub device_scale_factor: f64,
    pub is_mobile: bool,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            device_scale_factor: 1.0,
            is_mobile: false,
        }
    }
}

/// Options for content extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractOptions {
    pub format: ExtractFormat,
    pub include_metadata: bool,
    pub clean_html: bool,
    pub include_links: bool,
    pub include_images: bool,
    pub max_length: Option<usize>,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            format: ExtractFormat::Markdown,
            include_metadata: true,
            clean_html: true,
            include_links: true,
            include_images: false,
            max_length: None,
        }
    }
}

/// Format for extracted content.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExtractFormat {
    PlainText,
    Markdown,
    Html,
    Json,
}

/// Extracted content from a page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    pub text: String,
    pub format: ExtractFormat,
    pub title: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub published_date: Option<String>,
    pub word_count: usize,
    pub links: Vec<Link>,
    pub images: Vec<Image>,
    pub metadata: Value,
}

/// A hyperlink from extracted content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub text: String,
    pub href: String,
    pub rel: Option<String>,
}

/// An image from extracted content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub src: String,
    pub alt: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// Options for screenshot capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureOptions {
    pub format: CaptureFormat,
    pub quality: u8,
    pub full_page: bool,
    pub clip: Option<ClipRect>,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            format: CaptureFormat::Png,
            quality: 90,
            full_page: true,
            clip: None,
        }
    }
}

/// Format for captured images.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CaptureFormat {
    Png,
    Jpeg,
    Webp,
}

/// Rectangle for clipping screenshots.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClipRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// A captured page (screenshot or PDF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedPage {
    pub handle: PageHandle,
    pub format: CaptureFormat,
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Core abstraction for web browser operations.
///
/// This trait is implemented by `reasonkit-web` and consumed by `reasonkit-core`.
/// It provides a unified interface for web browsing, content extraction, and capture.
///
/// # Example
///
/// ```ignore
/// use reasonkit_core::traits::{WebBrowserAdapter, NavigateOptions, ExtractOptions};
///
/// async fn example(browser: &mut impl WebBrowserAdapter) -> WebAdapterResult<()> {
///     browser.connect().await?;
///
///     let page = browser.navigate("https://example.com", NavigateOptions::default()).await?;
///     let content = browser.extract_content(&page, ExtractOptions::default()).await?;
///
///     println!("Title: {:?}", content.title);
///     println!("Word count: {}", content.word_count);
///
///     browser.disconnect().await?;
///     Ok(())
/// }
/// ```
#[async_trait]
pub trait WebBrowserAdapter: Send + Sync {
    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    /// Connect to the browser instance.
    async fn connect(&mut self) -> WebAdapterResult<()>;

    /// Disconnect from the browser instance.
    async fn disconnect(&mut self) -> WebAdapterResult<()>;

    /// Check if currently connected.
    fn is_connected(&self) -> bool;

    // ─────────────────────────────────────────────────────────────────────────
    // Navigation
    // ─────────────────────────────────────────────────────────────────────────

    /// Navigate to a URL and return a handle to the loaded page.
    async fn navigate(&self, url: &str, options: NavigateOptions) -> WebAdapterResult<PageHandle>;

    /// Wait for the page to finish loading.
    async fn wait_for_load(&self, handle: &PageHandle, timeout: Duration) -> WebAdapterResult<()>;

    /// Go back in browser history.
    async fn go_back(&self, handle: &PageHandle) -> WebAdapterResult<()>;

    /// Go forward in browser history.
    async fn go_forward(&self, handle: &PageHandle) -> WebAdapterResult<()>;

    /// Reload the current page.
    async fn reload(&self, handle: &PageHandle) -> WebAdapterResult<()>;

    // ─────────────────────────────────────────────────────────────────────────
    // Content Extraction
    // ─────────────────────────────────────────────────────────────────────────

    /// Extract content from the page in the specified format.
    async fn extract_content(
        &self,
        handle: &PageHandle,
        options: ExtractOptions,
    ) -> WebAdapterResult<ExtractedContent>;

    /// Extract all links from the page.
    async fn extract_links(&self, handle: &PageHandle) -> WebAdapterResult<Vec<Link>>;

    /// Extract structured data using a CSS selector.
    async fn extract_structured(
        &self,
        handle: &PageHandle,
        selector: &str,
    ) -> WebAdapterResult<Value>;

    /// Get the raw HTML of the page.
    async fn get_html(&self, handle: &PageHandle) -> WebAdapterResult<String>;

    // ─────────────────────────────────────────────────────────────────────────
    // Capture
    // ─────────────────────────────────────────────────────────────────────────

    /// Capture a screenshot of the page.
    async fn capture_screenshot(
        &self,
        handle: &PageHandle,
        options: CaptureOptions,
    ) -> WebAdapterResult<CapturedPage>;

    /// Capture the page as a PDF.
    async fn capture_pdf(&self, handle: &PageHandle) -> WebAdapterResult<Vec<u8>>;

    // ─────────────────────────────────────────────────────────────────────────
    // Interaction
    // ─────────────────────────────────────────────────────────────────────────

    /// Click an element matching the selector.
    async fn click(&self, handle: &PageHandle, selector: &str) -> WebAdapterResult<()>;

    /// Type text into an element matching the selector.
    async fn type_text(
        &self,
        handle: &PageHandle,
        selector: &str,
        text: &str,
    ) -> WebAdapterResult<()>;

    /// Select an option from a dropdown.
    async fn select_option(
        &self,
        handle: &PageHandle,
        selector: &str,
        value: &str,
    ) -> WebAdapterResult<()>;

    /// Scroll the page.
    async fn scroll(&self, handle: &PageHandle, x: f64, y: f64) -> WebAdapterResult<()>;

    /// Wait for an element to appear.
    async fn wait_for_selector(
        &self,
        handle: &PageHandle,
        selector: &str,
        timeout: Duration,
    ) -> WebAdapterResult<()>;

    // ─────────────────────────────────────────────────────────────────────────
    // JavaScript
    // ─────────────────────────────────────────────────────────────────────────

    /// Evaluate JavaScript and return the result.
    async fn evaluate_js(&self, handle: &PageHandle, script: &str) -> WebAdapterResult<Value>;

    /// Inject a script into the page.
    async fn inject_script(&self, handle: &PageHandle, script: &str) -> WebAdapterResult<()>;

    // ─────────────────────────────────────────────────────────────────────────
    // Cookies & Storage
    // ─────────────────────────────────────────────────────────────────────────

    /// Get all cookies for the current page.
    async fn get_cookies(&self, handle: &PageHandle) -> WebAdapterResult<Vec<Cookie>>;

    /// Set a cookie.
    async fn set_cookie(&self, handle: &PageHandle, cookie: Cookie) -> WebAdapterResult<()>;

    /// Clear all cookies.
    async fn clear_cookies(&self, handle: &PageHandle) -> WebAdapterResult<()>;

    /// Get local storage value.
    async fn get_local_storage(
        &self,
        handle: &PageHandle,
        key: &str,
    ) -> WebAdapterResult<Option<String>>;

    /// Set local storage value.
    async fn set_local_storage(
        &self,
        handle: &PageHandle,
        key: &str,
        value: &str,
    ) -> WebAdapterResult<()>;
}

/// A browser cookie.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cookie {
    pub name: String,
    pub value: String,
    pub domain: Option<String>,
    pub path: Option<String>,
    pub expires: Option<i64>,
    pub http_only: bool,
    pub secure: bool,
    pub same_site: Option<String>,
}
