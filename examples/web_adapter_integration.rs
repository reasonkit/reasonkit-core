//! Example: Web Browser Adapter Integration
//!
//! This example demonstrates how to implement the WebBrowserAdapter trait
//! and use it to navigate, extract content, and capture screenshots.
//!
//! # Run with:
//! ```bash
//! cargo run --example web_adapter_integration
//! ```

use async_trait::async_trait;
use reasonkit::web_interface::{
    CaptureFormat, CaptureOptions, ExtractOptions, NavigateOptions, NavigateWaitEvent, PageHandle,
    WebAdapterResult, WebBrowserAdapter,
};
use serde_json::json;

// ═══════════════════════════════════════════════════════════════════════════
// MOCK IMPLEMENTATION FOR DEMONSTRATION
// ═══════════════════════════════════════════════════════════════════════════

/// Mock web browser adapter for demonstration
///
/// In production, this would be replaced with:
/// - McpWebAdapter (MCP stdio server binding)
/// - HttpWebAdapter (HTTP JSON-RPC binding)
/// - LocalWebAdapter (Direct FFI binding)
struct MockWebAdapter {
    connected: bool,
    page_counter: u32,
}

impl MockWebAdapter {
    fn new() -> Self {
        Self {
            connected: false,
            page_counter: 0,
        }
    }
}

#[async_trait]
impl WebBrowserAdapter for MockWebAdapter {
    async fn connect(&mut self) -> WebAdapterResult<()> {
        println!("[MockAdapter] Connecting to web service...");
        self.connected = true;
        Ok(())
    }

    async fn disconnect(&mut self) -> WebAdapterResult<()> {
        println!("[MockAdapter] Disconnecting from web service...");
        self.connected = false;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    async fn navigate(
        &mut self,
        url: &str,
        options: NavigateOptions,
    ) -> WebAdapterResult<PageHandle> {
        if !self.is_connected() {
            return Err(reasonkit::web_interface::WebAdapterError::NotConnected);
        }

        // Validate URL
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(reasonkit::web_interface::WebAdapterError::InvalidUrl(
                url.to_string(),
            ));
        }

        println!(
            "[MockAdapter] Navigating to: {} (wait_until: {})",
            url, options.wait_until
        );

        self.page_counter += 1;
        Ok(PageHandle {
            id: format!("page-{}", self.page_counter),
            url: url.to_string(),
            title: "Mock Page Title".to_string(),
            is_active: true,
        })
    }

    async fn go_back(&mut self) -> WebAdapterResult<PageHandle> {
        println!("[MockAdapter] Going back in history...");
        Ok(PageHandle {
            id: "page-back".to_string(),
            url: "https://example.com/previous".to_string(),
            title: "Previous Page".to_string(),
            is_active: true,
        })
    }

    async fn go_forward(&mut self) -> WebAdapterResult<PageHandle> {
        println!("[MockAdapter] Going forward in history...");
        Ok(PageHandle {
            id: "page-forward".to_string(),
            url: "https://example.com/next".to_string(),
            title: "Next Page".to_string(),
            is_active: true,
        })
    }

    async fn reload(&mut self) -> WebAdapterResult<PageHandle> {
        println!("[MockAdapter] Reloading current page...");
        Ok(PageHandle {
            id: "page-reload".to_string(),
            url: "https://example.com".to_string(),
            title: "Reloaded Page".to_string(),
            is_active: true,
        })
    }

    async fn extract_content(
        &mut self,
        page: &PageHandle,
        options: ExtractOptions,
    ) -> WebAdapterResult<reasonkit::web_interface::ExtractedContent> {
        if !self.is_connected() {
            return Err(reasonkit::web_interface::WebAdapterError::NotConnected);
        }

        println!(
            "[MockAdapter] Extracting content from: {} (extract_links: {}, extract_images: {})",
            page.url, options.extract_links, options.extract_images
        );

        Ok(reasonkit::web_interface::ExtractedContent {
            text: "This is the main content of the page. Lorem ipsum dolor sit amet.".to_string(),
            html: Some("<div><p>This is the main content of the page.</p></div>".to_string()),
            links: vec![
                reasonkit::web_interface::ExtractedLink {
                    text: "Home".to_string(),
                    href: "https://example.com".to_string(),
                    title: Some("Home Page".to_string()),
                },
                reasonkit::web_interface::ExtractedLink {
                    text: "About".to_string(),
                    href: "https://example.com/about".to_string(),
                    title: None,
                },
            ],
            images: vec![],
            metadata: reasonkit::web_interface::ContentMetadata {
                title: Some("Example Page".to_string()),
                description: Some("An example web page".to_string()),
                ..Default::default()
            },
            structured_data: None,
            language: Some("en".to_string()),
            confidence: 0.95,
        })
    }

    async fn execute_js(
        &mut self,
        page: &PageHandle,
        script: &str,
    ) -> WebAdapterResult<serde_json::Value> {
        println!(
            "[MockAdapter] Executing JavaScript on {}: {}",
            page.url, script
        );
        Ok(json!({ "result": "success", "value": 42 }))
    }

    async fn get_text(&mut self, page: &PageHandle, selector: &str) -> WebAdapterResult<String> {
        println!(
            "[MockAdapter] Getting text from selector: {} on {}",
            selector, page.url
        );
        Ok("Selected element text".to_string())
    }

    async fn capture_screenshot(
        &mut self,
        page: &PageHandle,
        options: CaptureOptions,
    ) -> WebAdapterResult<reasonkit::web_interface::CapturedPage> {
        if !self.is_connected() {
            return Err(reasonkit::web_interface::WebAdapterError::NotConnected);
        }

        println!(
            "[MockAdapter] Capturing {} screenshot from: {}",
            options.format, page.url
        );

        // Mock PNG data (1x1 transparent pixel)
        let png_data = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
        ];

        Ok(reasonkit::web_interface::CapturedPage {
            format: options.format,
            data: png_data,
            mime_type: "image/png".to_string(),
            size_bytes: 8,
            metadata: reasonkit::web_interface::CaptureMetadata {
                url: page.url.clone(),
                title: Some(page.title.clone()),
                viewport_width: 1024,
                viewport_height: 768,
                full_page: options.full_page,
                device_scale_factor: options.device_scale_factor.unwrap_or(1.0),
            },
        })
    }

    fn diagnostics(&self) -> serde_json::Value {
        json!({
            "adapter": "MockWebAdapter",
            "connected": self.is_connected(),
            "pages_created": self.page_counter,
        })
    }

    fn name(&self) -> &str {
        "mock-web-adapter"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE USAGE
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Web Browser Adapter Integration Example ===\n");

    // Create and connect adapter
    let mut adapter = MockWebAdapter::new();
    println!("1. Connecting to adapter...");
    adapter.connect().await?;
    println!("   Connected: {}\n", adapter.is_connected());

    // Navigate to a page
    println!("2. Navigating to a website...");
    let page = adapter
        .navigate(
            "https://example.com",
            NavigateOptions {
                timeout_ms: 30000,
                wait_until: NavigateWaitEvent::Load,
                ..Default::default()
            },
        )
        .await?;
    println!("   Page: {}\n", page);

    // Extract content
    println!("3. Extracting page content...");
    let content = adapter
        .extract_content(&page, ExtractOptions::default())
        .await?;
    println!("   Text: {}", &content.text[..60.min(content.text.len())]);
    println!("   Links found: {}", content.links.len());
    println!("   Confidence: {:.2}\n", content.confidence);

    // Execute custom JavaScript
    println!("4. Executing custom JavaScript...");
    let js_result = adapter
        .execute_js(&page, "document.querySelectorAll('a').length")
        .await?;
    println!("   Result: {}\n", js_result);

    // Get text from selector
    println!("5. Getting text from CSS selector...");
    let text = adapter.get_text(&page, "h1").await?;
    println!("   Text: {}\n", text);

    // Capture screenshot (PNG)
    println!("6. Capturing PNG screenshot...");
    let screenshot = adapter
        .capture_screenshot(&page, CaptureOptions::default().format(CaptureFormat::Png))
        .await?;
    println!("   Format: {:?}", screenshot.format);
    println!("   Size: {} bytes", screenshot.size_bytes);
    println!("   Mime type: {}\n", screenshot.mime_type);

    // Capture as PDF
    println!("7. Capturing as PDF...");
    let pdf = adapter
        .capture_screenshot(&page, CaptureOptions::default().format(CaptureFormat::Pdf))
        .await?;
    println!("   Format: {:?}", pdf.format);
    println!("   Size: {} bytes\n", pdf.size_bytes);

    // Navigation history
    println!("8. Testing navigation history...");
    let prev = adapter.go_back().await?;
    println!("   Previous page: {}", prev.url);
    let next = adapter.go_forward().await?;
    println!("   Next page: {}\n", next.url);

    // Reload page
    println!("9. Reloading page...");
    let reloaded = adapter.reload().await?;
    println!("   Reloaded: {}\n", reloaded.title);

    // Diagnostics
    println!("10. Getting diagnostics...");
    let diag = adapter.diagnostics();
    println!("   {}\n", serde_json::to_string_pretty(&diag)?);

    // Disconnect
    println!("11. Disconnecting...");
    adapter.disconnect().await?;
    println!("   Disconnected: {}", !adapter.is_connected());

    println!("\n=== Example Complete ===");
    Ok(())
}
