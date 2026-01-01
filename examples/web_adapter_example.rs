//! Example: WebBrowserAdapter Trait Usage
//!
//! This example demonstrates comprehensive usage of the WebBrowserAdapter trait
//! from the traits module. It shows:
//!
//! - Creating mock implementations
//! - Navigation and page lifecycle
//! - Content extraction patterns
//! - Screenshot and capture operations
//! - Browser interaction (click, type, scroll)
//! - JavaScript evaluation
//! - Cookie and storage management
//! - Error handling patterns
//! - Both sync and async usage patterns
//!
//! # Running this example
//!
//! ```bash
//! cargo run --example web_adapter_example
//! ```
//!
//! # Architecture
//!
//! The WebBrowserAdapter trait defines the contract between reasonkit-core and
//! reasonkit-web. This allows reasonkit-core to work with any browser backend
//! that implements the trait (Chromium, Firefox, etc.).
//!
//! ```text
//! reasonkit-core (consumer)
//!        |
//!        v
//! WebBrowserAdapter trait <-- defined in reasonkit-core/src/traits/web.rs
//!        ^
//!        |
//! reasonkit-web (implementation)
//!        |
//!        v
//! Browser Engine (Chromium via CDP, Playwright, etc.)
//! ```

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

// Import from the traits module
use reasonkit::traits::{
    CaptureFormat, CaptureOptions, CapturedPage, ClipRect, Cookie, ExtractFormat, ExtractOptions,
    ExtractedContent, Image, Link, NavigateOptions, PageHandle, Viewport, WaitUntil,
    WebAdapterError, WebAdapterResult, WebBrowserAdapter,
};

// ============================================================================
// MOCK IMPLEMENTATION
// ============================================================================

/// Mock implementation of WebBrowserAdapter for demonstration.
///
/// In production, you would use the implementation from reasonkit-web.
/// This mock is useful for:
/// - Testing without a real browser
/// - Understanding the trait interface
/// - Prototyping before integration
struct MockBrowserAdapter {
    connected: AtomicBool,
    page_counter: AtomicU32,
    pages: Arc<std::sync::RwLock<HashMap<String, MockPage>>>,
    cookies: Arc<std::sync::RwLock<Vec<Cookie>>>,
    local_storage: Arc<std::sync::RwLock<HashMap<String, String>>>,
}

/// Internal representation of a mock page.
#[derive(Clone)]
struct MockPage {
    url: String,
    title: String,
    html: String,
    status_code: u16,
    history: Vec<String>,
    history_index: usize,
}

impl MockBrowserAdapter {
    /// Create a new mock browser adapter.
    fn new() -> Self {
        Self {
            connected: AtomicBool::new(false),
            page_counter: AtomicU32::new(0),
            pages: Arc::new(std::sync::RwLock::new(HashMap::new())),
            cookies: Arc::new(std::sync::RwLock::new(Vec::new())),
            local_storage: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Generate a unique page ID.
    fn next_page_id(&self) -> String {
        let id = self.page_counter.fetch_add(1, Ordering::SeqCst);
        format!("page-{}", id)
    }

    /// Create a mock page for a URL.
    fn create_mock_page(&self, url: &str) -> MockPage {
        let title = if url.contains("example.com") {
            "Example Domain"
        } else if url.contains("github.com") {
            "GitHub"
        } else {
            "Mock Page"
        };

        MockPage {
            url: url.to_string(),
            title: title.to_string(),
            html: format!(
                r#"<!DOCTYPE html>
<html>
<head><title>{}</title></head>
<body>
    <h1>{}</h1>
    <p>This is mock content for {}</p>
    <a href="https://example.com/link1">Link 1</a>
    <a href="https://example.com/link2">Link 2</a>
    <img src="https://example.com/image.png" alt="Mock image" />
    <form>
        <input type="text" id="search" name="q" />
        <button type="submit">Search</button>
    </form>
</body>
</html>"#,
                title, title, url
            ),
            status_code: 200,
            history: vec![url.to_string()],
            history_index: 0,
        }
    }

    /// Extract mock text content from HTML.
    fn extract_text(&self, html: &str) -> String {
        // Simple mock extraction - remove tags
        let mut text = html.to_string();
        // Remove script and style tags with content
        let re_script = regex::Regex::new(r"<script[^>]*>[\s\S]*?</script>").unwrap();
        let re_style = regex::Regex::new(r"<style[^>]*>[\s\S]*?</style>").unwrap();
        text = re_script.replace_all(&text, "").to_string();
        text = re_style.replace_all(&text, "").to_string();
        // Remove remaining tags
        let re_tags = regex::Regex::new(r"<[^>]+>").unwrap();
        text = re_tags.replace_all(&text, " ").to_string();
        // Normalize whitespace
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

#[async_trait]
impl WebBrowserAdapter for MockBrowserAdapter {
    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    async fn connect(&mut self) -> WebAdapterResult<()> {
        println!("[MockBrowser] Connecting to browser...");
        tokio::time::sleep(Duration::from_millis(100)).await;
        self.connected.store(true, Ordering::SeqCst);
        println!("[MockBrowser] Connected successfully");
        Ok(())
    }

    async fn disconnect(&mut self) -> WebAdapterResult<()> {
        println!("[MockBrowser] Disconnecting from browser...");
        self.connected.store(false, Ordering::SeqCst);
        self.pages.write().unwrap().clear();
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    // -------------------------------------------------------------------------
    // Navigation
    // -------------------------------------------------------------------------

    async fn navigate(&self, url: &str, options: NavigateOptions) -> WebAdapterResult<PageHandle> {
        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        // Validate URL
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(WebAdapterError::Navigation(format!("Invalid URL: {}", url)));
        }

        println!(
            "[MockBrowser] Navigating to: {} (wait: {:?}, timeout: {:?})",
            url, options.wait_until, options.timeout
        );

        // Simulate load time
        tokio::time::sleep(Duration::from_millis(50)).await;

        let page_id = self.next_page_id();
        let mock_page = self.create_mock_page(url);

        // Store page
        self.pages
            .write()
            .unwrap()
            .insert(page_id.clone(), mock_page.clone());

        Ok(PageHandle {
            id: page_id,
            url: url.to_string(),
            title: Some(mock_page.title),
            status_code: mock_page.status_code,
            load_time_ms: 50,
        })
    }

    async fn wait_for_load(&self, handle: &PageHandle, timeout: Duration) -> WebAdapterResult<()> {
        println!(
            "[MockBrowser] Waiting for page {} to load (timeout: {:?})",
            handle.id, timeout
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    async fn go_back(&self, handle: &PageHandle) -> WebAdapterResult<()> {
        println!("[MockBrowser] Going back from page {}", handle.id);
        let mut pages = self.pages.write().unwrap();
        if let Some(page) = pages.get_mut(&handle.id) {
            if page.history_index > 0 {
                page.history_index -= 1;
                page.url = page.history[page.history_index].clone();
            }
        }
        Ok(())
    }

    async fn go_forward(&self, handle: &PageHandle) -> WebAdapterResult<()> {
        println!("[MockBrowser] Going forward from page {}", handle.id);
        let mut pages = self.pages.write().unwrap();
        if let Some(page) = pages.get_mut(&handle.id) {
            if page.history_index < page.history.len() - 1 {
                page.history_index += 1;
                page.url = page.history[page.history_index].clone();
            }
        }
        Ok(())
    }

    async fn reload(&self, handle: &PageHandle) -> WebAdapterResult<()> {
        println!("[MockBrowser] Reloading page {}", handle.id);
        tokio::time::sleep(Duration::from_millis(25)).await;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Content Extraction
    // -------------------------------------------------------------------------

    async fn extract_content(
        &self,
        handle: &PageHandle,
        options: ExtractOptions,
    ) -> WebAdapterResult<ExtractedContent> {
        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        println!(
            "[MockBrowser] Extracting content from {} (format: {:?})",
            handle.id, options.format
        );

        let pages = self.pages.read().unwrap();
        let page = pages
            .get(&handle.id)
            .ok_or_else(|| WebAdapterError::Navigation("Page not found".to_string()))?;

        let text = self.extract_text(&page.html);
        let formatted_text = match options.format {
            ExtractFormat::PlainText => text.clone(),
            ExtractFormat::Markdown => format!("# {}\n\n{}", page.title, text),
            ExtractFormat::Html => page.html.clone(),
            ExtractFormat::Json => json!({
                "title": page.title,
                "text": text
            })
            .to_string(),
        };

        let links = if options.include_links {
            vec![
                Link {
                    text: "Link 1".to_string(),
                    href: "https://example.com/link1".to_string(),
                    rel: None,
                },
                Link {
                    text: "Link 2".to_string(),
                    href: "https://example.com/link2".to_string(),
                    rel: Some("nofollow".to_string()),
                },
            ]
        } else {
            vec![]
        };

        let images = if options.include_images {
            vec![Image {
                src: "https://example.com/image.png".to_string(),
                alt: Some("Mock image".to_string()),
                width: Some(800),
                height: Some(600),
            }]
        } else {
            vec![]
        };

        Ok(ExtractedContent {
            text: formatted_text,
            format: options.format,
            title: Some(page.title.clone()),
            description: Some("A mock page description".to_string()),
            author: None,
            published_date: None,
            word_count: text.split_whitespace().count(),
            links,
            images,
            metadata: json!({
                "url": page.url,
                "status_code": page.status_code
            }),
        })
    }

    async fn extract_links(&self, handle: &PageHandle) -> WebAdapterResult<Vec<Link>> {
        println!("[MockBrowser] Extracting links from {}", handle.id);
        Ok(vec![Link {
            text: "Example Link".to_string(),
            href: "https://example.com".to_string(),
            rel: None,
        }])
    }

    async fn extract_structured(
        &self,
        handle: &PageHandle,
        selector: &str,
    ) -> WebAdapterResult<Value> {
        println!(
            "[MockBrowser] Extracting structured data from {} using selector: {}",
            handle.id, selector
        );
        Ok(json!({
            "selector": selector,
            "elements": [
                {"text": "Element 1", "class": "item"},
                {"text": "Element 2", "class": "item"}
            ]
        }))
    }

    async fn get_html(&self, handle: &PageHandle) -> WebAdapterResult<String> {
        let pages = self.pages.read().unwrap();
        let page = pages
            .get(&handle.id)
            .ok_or_else(|| WebAdapterError::Navigation("Page not found".to_string()))?;
        Ok(page.html.clone())
    }

    // -------------------------------------------------------------------------
    // Capture
    // -------------------------------------------------------------------------

    async fn capture_screenshot(
        &self,
        handle: &PageHandle,
        options: CaptureOptions,
    ) -> WebAdapterResult<CapturedPage> {
        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        println!(
            "[MockBrowser] Capturing screenshot of {} (format: {:?}, full_page: {})",
            handle.id, options.format, options.full_page
        );

        // Mock image data (minimal PNG header)
        let data = match options.format {
            CaptureFormat::Png => vec![
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // IHDR chunk
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            ],
            CaptureFormat::Jpeg => vec![0xFF, 0xD8, 0xFF, 0xE0],
            CaptureFormat::Webp => vec![0x52, 0x49, 0x46, 0x46],
        };

        Ok(CapturedPage {
            handle: handle.clone(),
            format: options.format,
            data,
            width: 1920,
            height: if options.full_page { 3000 } else { 1080 },
        })
    }

    async fn capture_pdf(&self, handle: &PageHandle) -> WebAdapterResult<Vec<u8>> {
        println!("[MockBrowser] Capturing PDF of {}", handle.id);
        // Mock PDF header
        Ok(vec![0x25, 0x50, 0x44, 0x46, 0x2D, 0x31, 0x2E, 0x34])
    }

    // -------------------------------------------------------------------------
    // Interaction
    // -------------------------------------------------------------------------

    async fn click(&self, handle: &PageHandle, selector: &str) -> WebAdapterResult<()> {
        println!("[MockBrowser] Clicking {} on {}", selector, handle.id);
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    async fn type_text(
        &self,
        handle: &PageHandle,
        selector: &str,
        text: &str,
    ) -> WebAdapterResult<()> {
        println!(
            "[MockBrowser] Typing '{}' into {} on {}",
            text, selector, handle.id
        );
        tokio::time::sleep(Duration::from_millis(text.len() as u64 * 5)).await;
        Ok(())
    }

    async fn select_option(
        &self,
        handle: &PageHandle,
        selector: &str,
        value: &str,
    ) -> WebAdapterResult<()> {
        println!(
            "[MockBrowser] Selecting '{}' in {} on {}",
            value, selector, handle.id
        );
        Ok(())
    }

    async fn scroll(&self, handle: &PageHandle, x: f64, y: f64) -> WebAdapterResult<()> {
        println!("[MockBrowser] Scrolling to ({}, {}) on {}", x, y, handle.id);
        Ok(())
    }

    async fn wait_for_selector(
        &self,
        handle: &PageHandle,
        selector: &str,
        timeout: Duration,
    ) -> WebAdapterResult<()> {
        println!(
            "[MockBrowser] Waiting for selector {} on {} (timeout: {:?})",
            selector, handle.id, timeout
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // JavaScript
    // -------------------------------------------------------------------------

    async fn evaluate_js(&self, handle: &PageHandle, script: &str) -> WebAdapterResult<Value> {
        println!(
            "[MockBrowser] Evaluating JS on {}: {}",
            handle.id,
            &script[..script.len().min(50)]
        );

        // Return mock result based on script content
        if script.contains("document.title") {
            Ok(json!("Mock Page Title"))
        } else if script.contains("querySelectorAll") {
            Ok(json!(5))
        } else {
            Ok(json!({"success": true, "result": "mock_value"}))
        }
    }

    async fn inject_script(&self, handle: &PageHandle, script: &str) -> WebAdapterResult<()> {
        println!(
            "[MockBrowser] Injecting script into {}: {} chars",
            handle.id,
            script.len()
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Cookies & Storage
    // -------------------------------------------------------------------------

    async fn get_cookies(&self, _handle: &PageHandle) -> WebAdapterResult<Vec<Cookie>> {
        let cookies = self.cookies.read().unwrap().clone();
        Ok(cookies)
    }

    async fn set_cookie(&self, _handle: &PageHandle, cookie: Cookie) -> WebAdapterResult<()> {
        println!("[MockBrowser] Setting cookie: {}", cookie.name);
        self.cookies.write().unwrap().push(cookie);
        Ok(())
    }

    async fn clear_cookies(&self, _handle: &PageHandle) -> WebAdapterResult<()> {
        println!("[MockBrowser] Clearing all cookies");
        self.cookies.write().unwrap().clear();
        Ok(())
    }

    async fn get_local_storage(
        &self,
        _handle: &PageHandle,
        key: &str,
    ) -> WebAdapterResult<Option<String>> {
        let storage = self.local_storage.read().unwrap();
        Ok(storage.get(key).cloned())
    }

    async fn set_local_storage(
        &self,
        _handle: &PageHandle,
        key: &str,
        value: &str,
    ) -> WebAdapterResult<()> {
        println!("[MockBrowser] Setting localStorage: {} = {}", key, value);
        self.local_storage
            .write()
            .unwrap()
            .insert(key.to_string(), value.to_string());
        Ok(())
    }
}

// ============================================================================
// DEMONSTRATION FUNCTIONS
// ============================================================================

/// Demonstrates navigation and page lifecycle.
async fn demo_navigation(browser: &impl WebBrowserAdapter) -> WebAdapterResult<()> {
    println!("\n--- Navigation Demo ---\n");

    // 1. Basic navigation
    println!("1. Basic navigation...");
    let page = browser
        .navigate("https://example.com", NavigateOptions::default())
        .await?;
    println!("   Page ID: {}", page.id);
    println!("   URL: {}", page.url);
    println!("   Title: {:?}", page.title);
    println!("   Status: {}", page.status_code);
    println!("   Load time: {}ms", page.load_time_ms);

    // 2. Navigation with options
    println!("\n2. Navigation with custom options...");
    let options = NavigateOptions {
        wait_until: WaitUntil::NetworkIdle,
        timeout: Duration::from_secs(60),
        user_agent: Some("ReasonKit/1.0".to_string()),
        headers: vec![("Accept-Language".to_string(), "en-US".to_string())],
        viewport: Some(Viewport {
            width: 1920,
            height: 1080,
            device_scale_factor: 2.0,
            is_mobile: false,
        }),
    };
    let page2 = browser.navigate("https://github.com", options).await?;
    println!("   Navigated to: {}", page2.url);

    // 3. Wait for load
    println!("\n3. Waiting for page load...");
    browser
        .wait_for_load(&page, Duration::from_secs(30))
        .await?;
    println!("   Page loaded");

    // 4. History navigation
    println!("\n4. History navigation...");
    browser.go_back(&page).await?;
    println!("   Went back");
    browser.go_forward(&page).await?;
    println!("   Went forward");
    browser.reload(&page).await?;
    println!("   Reloaded");

    Ok(())
}

/// Demonstrates content extraction patterns.
async fn demo_content_extraction(browser: &impl WebBrowserAdapter) -> WebAdapterResult<()> {
    println!("\n--- Content Extraction Demo ---\n");

    let page = browser
        .navigate("https://example.com/article", NavigateOptions::default())
        .await?;

    // 1. Plain text extraction
    println!("1. Plain text extraction...");
    let content = browser
        .extract_content(
            &page,
            ExtractOptions {
                format: ExtractFormat::PlainText,
                include_metadata: true,
                clean_html: true,
                include_links: false,
                include_images: false,
                max_length: None,
            },
        )
        .await?;
    println!("   Title: {:?}", content.title);
    println!("   Word count: {}", content.word_count);
    println!(
        "   Text preview: {}...",
        &content.text[..content.text.len().min(80)]
    );

    // 2. Markdown extraction
    println!("\n2. Markdown extraction...");
    let content = browser
        .extract_content(
            &page,
            ExtractOptions {
                format: ExtractFormat::Markdown,
                include_links: true,
                include_images: true,
                ..Default::default()
            },
        )
        .await?;
    println!("   Links found: {}", content.links.len());
    println!("   Images found: {}", content.images.len());

    // 3. Extract links only
    println!("\n3. Extracting links...");
    let links = browser.extract_links(&page).await?;
    for link in &links {
        println!("   - {} -> {}", link.text, link.href);
    }

    // 4. Structured data extraction
    println!("\n4. Structured data extraction...");
    let structured = browser.extract_structured(&page, ".product-card").await?;
    println!(
        "   Result: {}",
        serde_json::to_string_pretty(&structured)
            .unwrap_or_else(|_| "<failed to serialize structured result>".to_string())
    );

    // 5. Raw HTML
    println!("\n5. Getting raw HTML...");
    let html = browser.get_html(&page).await?;
    println!("   HTML length: {} chars", html.len());

    Ok(())
}

/// Demonstrates screenshot and capture operations.
async fn demo_capture(browser: &impl WebBrowserAdapter) -> WebAdapterResult<()> {
    println!("\n--- Capture Demo ---\n");

    let page = browser
        .navigate("https://example.com", NavigateOptions::default())
        .await?;

    // 1. PNG screenshot (full page)
    println!("1. PNG screenshot (full page)...");
    let screenshot = browser
        .capture_screenshot(
            &page,
            CaptureOptions {
                format: CaptureFormat::Png,
                quality: 100,
                full_page: true,
                clip: None,
            },
        )
        .await?;
    println!("   Format: {:?}", screenshot.format);
    println!("   Size: {}x{}", screenshot.width, screenshot.height);
    println!("   Data size: {} bytes", screenshot.data.len());

    // 2. JPEG screenshot (viewport only)
    println!("\n2. JPEG screenshot (viewport only)...");
    let screenshot = browser
        .capture_screenshot(
            &page,
            CaptureOptions {
                format: CaptureFormat::Jpeg,
                quality: 85,
                full_page: false,
                clip: None,
            },
        )
        .await?;
    println!("   Size: {}x{}", screenshot.width, screenshot.height);

    // 3. Screenshot with clipping
    println!("\n3. Screenshot with clipping...");
    let screenshot = browser
        .capture_screenshot(
            &page,
            CaptureOptions {
                format: CaptureFormat::Png,
                quality: 100,
                full_page: false,
                clip: Some(ClipRect {
                    x: 0.0,
                    y: 0.0,
                    width: 800.0,
                    height: 600.0,
                }),
            },
        )
        .await?;
    println!("   Clipped screenshot captured");

    // 4. PDF capture
    println!("\n4. PDF capture...");
    let pdf_data = browser.capture_pdf(&page).await?;
    println!("   PDF size: {} bytes", pdf_data.len());

    Ok(())
}

/// Demonstrates browser interaction operations.
async fn demo_interaction(browser: &impl WebBrowserAdapter) -> WebAdapterResult<()> {
    println!("\n--- Interaction Demo ---\n");

    let page = browser
        .navigate("https://example.com/form", NavigateOptions::default())
        .await?;

    // 1. Click element
    println!("1. Clicking element...");
    browser.click(&page, "#submit-button").await?;
    println!("   Clicked #submit-button");

    // 2. Type text
    println!("\n2. Typing text...");
    browser
        .type_text(&page, "#search-input", "ReasonKit search query")
        .await?;
    println!("   Typed into #search-input");

    // 3. Select option
    println!("\n3. Selecting dropdown option...");
    browser
        .select_option(&page, "#country-select", "US")
        .await?;
    println!("   Selected 'US' in dropdown");

    // 4. Scroll
    println!("\n4. Scrolling page...");
    browser.scroll(&page, 0.0, 500.0).await?;
    println!("   Scrolled to (0, 500)");

    // 5. Wait for element
    println!("\n5. Waiting for element...");
    browser
        .wait_for_selector(&page, ".dynamic-content", Duration::from_secs(10))
        .await?;
    println!("   Element appeared");

    Ok(())
}

/// Demonstrates JavaScript evaluation.
async fn demo_javascript(browser: &impl WebBrowserAdapter) -> WebAdapterResult<()> {
    println!("\n--- JavaScript Demo ---\n");

    let page = browser
        .navigate("https://example.com", NavigateOptions::default())
        .await?;

    // 1. Simple evaluation
    println!("1. Simple JS evaluation...");
    let result = browser.evaluate_js(&page, "document.title").await?;
    println!("   document.title = {}", result);

    // 2. Complex evaluation
    println!("\n2. Complex JS evaluation...");
    let result = browser
        .evaluate_js(&page, "document.querySelectorAll('a').length")
        .await?;
    println!("   Link count = {}", result);

    // 3. Return object
    println!("\n3. Returning object from JS...");
    let result = browser
        .evaluate_js(
            &page,
            r#"({
                url: window.location.href,
                width: window.innerWidth,
                height: window.innerHeight
            })"#,
        )
        .await?;
    println!(
        "   Result: {}",
        serde_json::to_string_pretty(&result)
            .unwrap_or_else(|_| "<failed to serialize js result>".to_string())
    );

    // 4. Inject script
    println!("\n4. Injecting script...");
    browser
        .inject_script(
            &page,
            r#"
            window.reasonkit = {
                version: '1.0.0',
                ready: true
            };
            "#,
        )
        .await?;
    println!("   Script injected");

    Ok(())
}

/// Demonstrates cookie and storage management.
async fn demo_cookies_storage(browser: &impl WebBrowserAdapter) -> WebAdapterResult<()> {
    println!("\n--- Cookies & Storage Demo ---\n");

    let page = browser
        .navigate("https://example.com", NavigateOptions::default())
        .await?;

    // 1. Set cookie
    println!("1. Setting cookie...");
    let cookie = Cookie {
        name: "session".to_string(),
        value: "abc123".to_string(),
        domain: Some(".example.com".to_string()),
        path: Some("/".to_string()),
        expires: Some(chrono::Utc::now().timestamp() + 3600),
        http_only: true,
        secure: true,
        same_site: Some("Strict".to_string()),
    };
    browser.set_cookie(&page, cookie).await?;
    println!("   Cookie 'session' set");

    // 2. Get cookies
    println!("\n2. Getting cookies...");
    let cookies = browser.get_cookies(&page).await?;
    for cookie in &cookies {
        println!(
            "   - {} = {} (domain: {:?})",
            cookie.name, cookie.value, cookie.domain
        );
    }

    // 3. Set local storage
    println!("\n3. Setting local storage...");
    browser
        .set_local_storage(&page, "user_prefs", r#"{"theme":"dark"}"#)
        .await?;
    println!("   localStorage 'user_prefs' set");

    // 4. Get local storage
    println!("\n4. Getting local storage...");
    if let Some(value) = browser.get_local_storage(&page, "user_prefs").await? {
        println!("   user_prefs = {}", value);
    }

    // 5. Clear cookies
    println!("\n5. Clearing cookies...");
    browser.clear_cookies(&page).await?;
    println!("   Cookies cleared");

    Ok(())
}

/// Demonstrates error handling patterns.
async fn demo_error_handling(browser: &mut MockBrowserAdapter) -> WebAdapterResult<()> {
    println!("\n--- Error Handling Demo ---\n");

    // 1. Not connected error
    println!("1. Handling NotConnected error...");
    browser.disconnect().await?;
    match browser
        .navigate("https://example.com", NavigateOptions::default())
        .await
    {
        Ok(_) => println!("   Navigation succeeded (unexpected)"),
        Err(WebAdapterError::NotConnected) => println!("   NotConnected error (expected)"),
        Err(e) => println!("   Other error: {}", e),
    }

    // Reconnect for further tests
    browser.connect().await?;

    // 2. Invalid URL error
    println!("\n2. Handling invalid URL error...");
    match browser
        .navigate("not-a-valid-url", NavigateOptions::default())
        .await
    {
        Ok(_) => println!("   Navigation succeeded (unexpected)"),
        Err(WebAdapterError::Navigation(msg)) => println!("   Navigation error: {}", msg),
        Err(e) => println!("   Other error: {}", e),
    }

    // 3. Timeout handling
    println!("\n3. Timeout handling pattern...");
    let result = tokio::time::timeout(
        Duration::from_millis(100),
        browser.navigate("https://example.com", NavigateOptions::default()),
    )
    .await;

    match result {
        Ok(Ok(page)) => println!("   Navigated before timeout: {}", page.url),
        Ok(Err(e)) => println!("   Navigation error: {}", e),
        Err(_) => println!("   Operation timed out"),
    }

    Ok(())
}

// ============================================================================
// SYNCHRONOUS WRAPPER PATTERN
// ============================================================================

/// Demonstrates how to use async WebBrowserAdapter from synchronous code.
mod sync_wrapper {
    use super::*;
    use tokio::runtime::Runtime;

    /// Synchronous wrapper around WebBrowserAdapter.
    pub struct SyncBrowserAdapter<B: WebBrowserAdapter> {
        inner: std::sync::Mutex<B>,
        runtime: Runtime,
    }

    impl<B: WebBrowserAdapter> SyncBrowserAdapter<B> {
        /// Create a new synchronous wrapper.
        pub fn new(adapter: B) -> Self {
            Self {
                inner: std::sync::Mutex::new(adapter),
                runtime: Runtime::new().expect("Failed to create Tokio runtime"),
            }
        }

        /// Connect synchronously.
        pub fn connect(&self) -> WebAdapterResult<()> {
            let mut adapter = self.inner.lock().unwrap();
            self.runtime.block_on(adapter.connect())
        }

        /// Disconnect synchronously.
        pub fn disconnect(&self) -> WebAdapterResult<()> {
            let mut adapter = self.inner.lock().unwrap();
            self.runtime.block_on(adapter.disconnect())
        }

        /// Navigate synchronously.
        pub fn navigate(
            &self,
            url: &str,
            options: NavigateOptions,
        ) -> WebAdapterResult<PageHandle> {
            let adapter = self.inner.lock().unwrap();
            self.runtime.block_on(adapter.navigate(url, options))
        }

        /// Extract content synchronously.
        pub fn extract_content(
            &self,
            handle: &PageHandle,
            options: ExtractOptions,
        ) -> WebAdapterResult<ExtractedContent> {
            let adapter = self.inner.lock().unwrap();
            self.runtime
                .block_on(adapter.extract_content(handle, options))
        }

        /// Capture screenshot synchronously.
        pub fn capture_screenshot(
            &self,
            handle: &PageHandle,
            options: CaptureOptions,
        ) -> WebAdapterResult<CapturedPage> {
            let adapter = self.inner.lock().unwrap();
            self.runtime
                .block_on(adapter.capture_screenshot(handle, options))
        }
    }

    /// Demonstrate synchronous usage.
    pub fn demo_sync_usage() {
        println!("\n--- Synchronous Wrapper Demo ---\n");

        let async_adapter = MockBrowserAdapter::new();
        let sync_adapter = SyncBrowserAdapter::new(async_adapter);

        // Connect (sync)
        println!("1. Connecting (sync)...");
        match sync_adapter.connect() {
            Ok(_) => println!("   Connected"),
            Err(e) => println!("   Error: {}", e),
        }

        // Navigate (sync)
        println!("\n2. Navigating (sync)...");
        match sync_adapter.navigate("https://example.com", NavigateOptions::default()) {
            Ok(page) => {
                println!("   Navigated to: {}", page.url);

                // Extract content (sync)
                println!("\n3. Extracting content (sync)...");
                match sync_adapter.extract_content(&page, ExtractOptions::default()) {
                    Ok(content) => println!("   Word count: {}", content.word_count),
                    Err(e) => println!("   Error: {}", e),
                }
            }
            Err(e) => println!("   Error: {}", e),
        }

        // Disconnect (sync)
        println!("\n4. Disconnecting (sync)...");
        match sync_adapter.disconnect() {
            Ok(_) => println!("   Disconnected"),
            Err(e) => println!("   Error: {}", e),
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

#[tokio::main]
async fn main() -> WebAdapterResult<()> {
    println!("=======================================================");
    println!("    ReasonKit WebBrowserAdapter Trait Usage Example");
    println!("=======================================================");

    // Create and connect browser
    let mut browser = MockBrowserAdapter::new();
    browser.connect().await?;

    // Run async demos
    demo_navigation(&browser).await?;
    demo_content_extraction(&browser).await?;
    demo_capture(&browser).await?;
    demo_interaction(&browser).await?;
    demo_javascript(&browser).await?;
    demo_cookies_storage(&browser).await?;
    demo_error_handling(&mut browser).await?;

    // Disconnect
    println!("\n--- Cleanup ---\n");
    browser.disconnect().await?;
    println!("Browser disconnected");

    // Run sync demo
    sync_wrapper::demo_sync_usage();

    println!("\n=======================================================");
    println!("                    Example Complete");
    println!("=======================================================");
    println!("\nKey Takeaways:");
    println!("  1. WebBrowserAdapter trait provides unified interface for browser operations");
    println!("  2. All operations are async-first (use tokio runtime)");
    println!("  3. Supports navigation, content extraction, screenshots, and interaction");
    println!("  4. Multiple extraction formats: PlainText, Markdown, HTML, JSON");
    println!("  5. Use SyncBrowserAdapter wrapper for synchronous codebases");
    println!("  6. Error handling uses structured WebAdapterError enum");
    println!("  7. Cookie and localStorage management for session handling");
    println!("\nFor production use, see reasonkit-web crate for real implementations.");

    Ok(())
}
