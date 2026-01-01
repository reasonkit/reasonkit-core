//! Trait Integration Tests
//!
//! Comprehensive integration tests for cross-crate trait implementations.
//!
//! Tests cover:
//! - MemoryService trait operations (storage, retrieval, embedding)
//! - WebBrowserAdapter trait operations (navigation, extraction, capture)
//! - Mock implementations for testing without external dependencies
//! - End-to-end workflows using trait abstractions
//!
//! These tests verify that trait contracts are correctly defined and
//! can be implemented by both mock and real implementations.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;

// Import trait definitions from reasonkit-core
// Note: These are in src/traits/mod.rs
use reasonkit::traits::{
    CaptureFormat, CaptureOptions, CapturedPage, ClipRect, Cookie, ExtractFormat, ExtractOptions,
    ExtractedContent, Image, Link, NavigateOptions, PageHandle, Viewport, WaitUntil,
    WebAdapterError, WebAdapterResult, WebBrowserAdapter,
};
use reasonkit::traits::{
    Chunk, ContextWindow, DistanceMetric, Document, HybridConfig, IndexConfig, IndexStats,
    MemoryConfig, MemoryError, MemoryResult, MemoryService, RetrievalSource, SearchResult,
};

// ============================================================================
// MOCK MEMORY SERVICE IMPLEMENTATION
// ============================================================================

/// Mock implementation of MemoryService for testing.
///
/// This implementation stores documents and chunks in memory with
/// configurable behavior for testing different scenarios.
struct MockMemoryService {
    documents: Arc<RwLock<HashMap<Uuid, Document>>>,
    chunks: Arc<RwLock<HashMap<Uuid, Chunk>>>,
    embeddings: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    config: RwLock<MemoryConfig>,
    healthy: AtomicBool,
    operation_count: AtomicUsize,
}

impl MockMemoryService {
    fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            chunks: Arc::new(RwLock::new(HashMap::new())),
            embeddings: Arc::new(RwLock::new(HashMap::new())),
            config: RwLock::new(MemoryConfig::default()),
            healthy: AtomicBool::new(true),
            operation_count: AtomicUsize::new(0),
        }
    }

    fn with_config(config: MemoryConfig) -> Self {
        let mut service = Self::new();
        service.config = RwLock::new(config);
        service
    }

    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::SeqCst);
    }

    fn operation_count(&self) -> usize {
        self.operation_count.load(Ordering::SeqCst)
    }

    fn increment_ops(&self) {
        self.operation_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Generate a mock embedding for a text.
    fn generate_mock_embedding(&self, text: &str) -> Vec<f32> {
        // Simple deterministic embedding based on text hash
        let hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        let dim = 384; // Default embedding dimension
        (0..dim)
            .map(|i| ((hash.wrapping_add(i as u32)) % 1000) as f32 / 1000.0)
            .collect()
    }
}

#[async_trait]
impl MemoryService for MockMemoryService {
    // ─────────────────────────────────────────────────────────────────────────
    // Storage Operations
    // ─────────────────────────────────────────────────────────────────────────

    async fn store_document(&self, doc: &Document) -> MemoryResult<Uuid> {
        self.increment_ops();

        if !self.healthy.load(Ordering::SeqCst) {
            return Err(MemoryError::Storage("Service unhealthy".to_string()));
        }

        let id = doc.id.unwrap_or_else(Uuid::new_v4);
        let mut doc_clone = doc.clone();
        doc_clone.id = Some(id);
        doc_clone.created_at = Some(chrono::Utc::now().timestamp());

        let mut docs = self.documents.write().await;
        docs.insert(id, doc_clone);

        Ok(id)
    }

    async fn store_chunks(&self, chunks: &[Chunk]) -> MemoryResult<Vec<Uuid>> {
        self.increment_ops();

        if !self.healthy.load(Ordering::SeqCst) {
            return Err(MemoryError::Storage("Service unhealthy".to_string()));
        }

        let mut stored_chunks = self.chunks.write().await;
        let mut ids = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let id = chunk.id.unwrap_or_else(Uuid::new_v4);
            let mut chunk_clone = chunk.clone();
            chunk_clone.id = Some(id);
            stored_chunks.insert(id, chunk_clone);
            ids.push(id);
        }

        Ok(ids)
    }

    async fn delete_document(&self, id: Uuid) -> MemoryResult<()> {
        self.increment_ops();

        let mut docs = self.documents.write().await;
        if docs.remove(&id).is_none() {
            return Err(MemoryError::NotFound(id));
        }

        // Also remove associated chunks
        let mut chunks = self.chunks.write().await;
        chunks.retain(|_, chunk| chunk.document_id != id);

        Ok(())
    }

    async fn update_document(&self, id: Uuid, doc: &Document) -> MemoryResult<()> {
        self.increment_ops();

        let mut docs = self.documents.write().await;
        if !docs.contains_key(&id) {
            return Err(MemoryError::NotFound(id));
        }

        let mut doc_clone = doc.clone();
        doc_clone.id = Some(id);
        docs.insert(id, doc_clone);

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Retrieval Operations
    // ─────────────────────────────────────────────────────────────────────────

    async fn search(&self, query: &str, top_k: usize) -> MemoryResult<Vec<SearchResult>> {
        self.increment_ops();

        let query_embedding = self.generate_mock_embedding(query);
        let chunks = self.chunks.read().await;

        let mut results: Vec<SearchResult> = chunks
            .values()
            .map(|chunk| {
                let chunk_embedding = chunk
                    .embedding
                    .clone()
                    .unwrap_or_else(|| self.generate_mock_embedding(&chunk.content));

                // Cosine similarity (simplified)
                let score = query_embedding
                    .iter()
                    .zip(chunk_embedding.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();

                SearchResult {
                    chunk: chunk.clone(),
                    score,
                    source: RetrievalSource::Vector,
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }

    async fn hybrid_search(
        &self,
        query: &str,
        top_k: usize,
        config: HybridConfig,
    ) -> MemoryResult<Vec<SearchResult>> {
        self.increment_ops();

        // For mock, just use vector search with hybrid source indicator
        let mut results = self.search(query, top_k).await?;

        // Adjust scores based on config weights
        for result in &mut results {
            result.score *= config.vector_weight;
            result.source = RetrievalSource::Hybrid;
        }

        if config.use_reranker {
            results.truncate(config.reranker_top_k.min(results.len()));
        }

        Ok(results)
    }

    async fn get_by_id(&self, id: Uuid) -> MemoryResult<Option<Document>> {
        self.increment_ops();

        let docs = self.documents.read().await;
        Ok(docs.get(&id).cloned())
    }

    async fn get_context(&self, query: &str, max_tokens: usize) -> MemoryResult<ContextWindow> {
        self.increment_ops();

        let results = self.search(query, 10).await?;

        let mut total_tokens = 0;
        let mut selected_chunks = Vec::new();
        let mut truncated = false;

        for result in results {
            // Estimate tokens (simple word count * 1.3)
            let chunk_tokens =
                (result.chunk.content.split_whitespace().count() as f32 * 1.3) as usize;

            if total_tokens + chunk_tokens > max_tokens {
                truncated = true;
                break;
            }

            total_tokens += chunk_tokens;
            selected_chunks.push(result);
        }

        Ok(ContextWindow {
            chunks: selected_chunks,
            total_tokens,
            truncated,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Embedding Operations
    // ─────────────────────────────────────────────────────────────────────────

    async fn embed(&self, text: &str) -> MemoryResult<Vec<f32>> {
        self.increment_ops();

        let embedding = self.generate_mock_embedding(text);

        // Cache the embedding
        let mut embeddings = self.embeddings.write().await;
        embeddings.insert(text.to_string(), embedding.clone());

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[&str]) -> MemoryResult<Vec<Vec<f32>>> {
        self.increment_ops();

        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Index Management
    // ─────────────────────────────────────────────────────────────────────────

    async fn create_index(&self, _config: IndexConfig) -> MemoryResult<()> {
        self.increment_ops();
        // Mock: No-op for in-memory storage
        Ok(())
    }

    async fn rebuild_index(&self) -> MemoryResult<()> {
        self.increment_ops();
        // Mock: Re-embed all chunks
        let chunks = self.chunks.read().await;
        for chunk in chunks.values() {
            let _ = self.generate_mock_embedding(&chunk.content);
        }
        Ok(())
    }

    async fn get_stats(&self) -> MemoryResult<IndexStats> {
        self.increment_ops();

        let docs = self.documents.read().await;
        let chunks = self.chunks.read().await;
        let embeddings = self.embeddings.read().await;

        Ok(IndexStats {
            total_documents: docs.len(),
            total_chunks: chunks.len(),
            total_vectors: embeddings.len(),
            index_size_bytes: (docs.len() * 1000 + chunks.len() * 500) as u64,
            last_updated: chrono::Utc::now().timestamp(),
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Configuration
    // ─────────────────────────────────────────────────────────────────────────

    fn config(&self) -> &MemoryConfig {
        // This is tricky with async - return a reference to config
        // In real impl, this would need interior mutability pattern
        // For mock, we'll use a workaround
        unsafe {
            // SAFETY: We only read from config, and the RwLock ensures exclusive access
            // This is a mock implementation for testing purposes only
            &*(&self.config as *const RwLock<MemoryConfig> as *const MemoryConfig)
        }
    }

    fn set_config(&mut self, config: MemoryConfig) {
        // Synchronous config update
        let _ = futures::executor::block_on(async {
            let mut cfg = self.config.write().await;
            *cfg = config;
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Health & Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    async fn health_check(&self) -> MemoryResult<bool> {
        Ok(self.healthy.load(Ordering::SeqCst))
    }

    async fn flush(&self) -> MemoryResult<()> {
        self.increment_ops();
        // Mock: No-op for in-memory storage
        Ok(())
    }

    async fn shutdown(&self) -> MemoryResult<()> {
        self.increment_ops();
        self.healthy.store(false, Ordering::SeqCst);
        Ok(())
    }
}

// ============================================================================
// MOCK WEB BROWSER ADAPTER IMPLEMENTATION
// ============================================================================

/// Mock implementation of WebBrowserAdapter for testing.
///
/// This implementation simulates browser operations without
/// requiring an actual browser instance.
struct MockWebBrowserAdapter {
    connected: AtomicBool,
    pages: Arc<RwLock<HashMap<String, MockPage>>>,
    cookies: Arc<RwLock<Vec<Cookie>>>,
    local_storage: Arc<RwLock<HashMap<String, String>>>,
    operation_count: AtomicUsize,
}

struct MockPage {
    handle: PageHandle,
    html: String,
    links: Vec<Link>,
    images: Vec<Image>,
}

impl MockWebBrowserAdapter {
    fn new() -> Self {
        Self {
            connected: AtomicBool::new(false),
            pages: Arc::new(RwLock::new(HashMap::new())),
            cookies: Arc::new(RwLock::new(Vec::new())),
            local_storage: Arc::new(RwLock::new(HashMap::new())),
            operation_count: AtomicUsize::new(0),
        }
    }

    fn operation_count(&self) -> usize {
        self.operation_count.load(Ordering::SeqCst)
    }

    fn increment_ops(&self) {
        self.operation_count.fetch_add(1, Ordering::SeqCst);
    }

    fn generate_mock_page(url: &str) -> MockPage {
        let id = Uuid::new_v4().to_string();
        let title = format!("Mock Page - {}", url);

        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head><title>{}</title></head>
<body>
<h1>Welcome to {}</h1>
<p>This is mock content for testing.</p>
<a href="/page1">Link 1</a>
<a href="/page2">Link 2</a>
<img src="/image1.png" alt="Image 1" />
</body>
</html>"#,
            title, url
        );

        let links = vec![
            Link {
                text: "Link 1".to_string(),
                href: "/page1".to_string(),
                rel: None,
            },
            Link {
                text: "Link 2".to_string(),
                href: "/page2".to_string(),
                rel: None,
            },
        ];

        let images = vec![Image {
            src: "/image1.png".to_string(),
            alt: Some("Image 1".to_string()),
            width: Some(100),
            height: Some(100),
        }];

        MockPage {
            handle: PageHandle {
                id,
                url: url.to_string(),
                title: Some(title),
                status_code: 200,
                load_time_ms: 150,
            },
            html,
            links,
            images,
        }
    }
}

#[async_trait]
impl WebBrowserAdapter for MockWebBrowserAdapter {
    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    async fn connect(&mut self) -> WebAdapterResult<()> {
        self.increment_ops();
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&mut self) -> WebAdapterResult<()> {
        self.increment_ops();
        self.connected.store(false, Ordering::SeqCst);
        self.pages.write().await.clear();
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Navigation
    // ─────────────────────────────────────────────────────────────────────────

    async fn navigate(&self, url: &str, _options: NavigateOptions) -> WebAdapterResult<PageHandle> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let page = Self::generate_mock_page(url);
        let handle = page.handle.clone();

        let mut pages = self.pages.write().await;
        pages.insert(handle.id.clone(), page);

        Ok(handle)
    }

    async fn wait_for_load(&self, _handle: &PageHandle, timeout: Duration) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        // Simulate a short wait
        tokio::time::sleep(Duration::from_millis(10).min(timeout)).await;
        Ok(())
    }

    async fn go_back(&self, handle: &PageHandle) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::Navigation("Page not found".to_string()));
        }

        Ok(())
    }

    async fn go_forward(&self, handle: &PageHandle) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::Navigation("Page not found".to_string()));
        }

        Ok(())
    }

    async fn reload(&self, handle: &PageHandle) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::Navigation("Page not found".to_string()));
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Content Extraction
    // ─────────────────────────────────────────────────────────────────────────

    async fn extract_content(
        &self,
        handle: &PageHandle,
        options: ExtractOptions,
    ) -> WebAdapterResult<ExtractedContent> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        let page = pages
            .get(&handle.id)
            .ok_or_else(|| WebAdapterError::Extraction("Page not found".to_string()))?;

        let text = match options.format {
            ExtractFormat::PlainText => {
                "Welcome to the mock page. This is mock content for testing.".to_string()
            }
            ExtractFormat::Markdown => "# Welcome\n\nThis is mock content for testing.".to_string(),
            ExtractFormat::Html => page.html.clone(),
            ExtractFormat::Json => serde_json::json!({
                "title": page.handle.title,
                "content": "Mock content"
            })
            .to_string(),
        };

        let links = if options.include_links {
            page.links.clone()
        } else {
            Vec::new()
        };

        let images = if options.include_images {
            page.images.clone()
        } else {
            Vec::new()
        };

        Ok(ExtractedContent {
            text,
            format: options.format,
            title: page.handle.title.clone(),
            description: Some("Mock page description".to_string()),
            author: None,
            published_date: None,
            word_count: 10,
            links,
            images,
            metadata: serde_json::json!({}),
        })
    }

    async fn extract_links(&self, handle: &PageHandle) -> WebAdapterResult<Vec<Link>> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        let page = pages
            .get(&handle.id)
            .ok_or_else(|| WebAdapterError::Extraction("Page not found".to_string()))?;

        Ok(page.links.clone())
    }

    async fn extract_structured(
        &self,
        handle: &PageHandle,
        selector: &str,
    ) -> WebAdapterResult<Value> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::Extraction("Page not found".to_string()));
        }

        // Return mock structured data
        Ok(serde_json::json!({
            "selector": selector,
            "matches": [
                {"text": "Mock match 1"},
                {"text": "Mock match 2"}
            ]
        }))
    }

    async fn get_html(&self, handle: &PageHandle) -> WebAdapterResult<String> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        let page = pages
            .get(&handle.id)
            .ok_or_else(|| WebAdapterError::Extraction("Page not found".to_string()))?;

        Ok(page.html.clone())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Capture
    // ─────────────────────────────────────────────────────────────────────────

    async fn capture_screenshot(
        &self,
        handle: &PageHandle,
        options: CaptureOptions,
    ) -> WebAdapterResult<CapturedPage> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::Screenshot("Page not found".to_string()));
        }

        // Return mock screenshot data
        let data = vec![0u8; 1000]; // Mock image data

        Ok(CapturedPage {
            handle: handle.clone(),
            format: options.format,
            data,
            width: 1920,
            height: if options.full_page { 5000 } else { 1080 },
        })
    }

    async fn capture_pdf(&self, handle: &PageHandle) -> WebAdapterResult<Vec<u8>> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::Screenshot("Page not found".to_string()));
        }

        // Return mock PDF data
        Ok(vec![0x25, 0x50, 0x44, 0x46]) // PDF magic bytes
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Interaction
    // ─────────────────────────────────────────────────────────────────────────

    async fn click(&self, handle: &PageHandle, selector: &str) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::ElementNotFound(selector.to_string()));
        }

        Ok(())
    }

    async fn type_text(
        &self,
        handle: &PageHandle,
        selector: &str,
        _text: &str,
    ) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::ElementNotFound(selector.to_string()));
        }

        Ok(())
    }

    async fn select_option(
        &self,
        handle: &PageHandle,
        selector: &str,
        _value: &str,
    ) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::ElementNotFound(selector.to_string()));
        }

        Ok(())
    }

    async fn scroll(&self, handle: &PageHandle, _x: f64, _y: f64) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::Navigation("Page not found".to_string()));
        }

        Ok(())
    }

    async fn wait_for_selector(
        &self,
        handle: &PageHandle,
        _selector: &str,
        _timeout: Duration,
    ) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::Navigation("Page not found".to_string()));
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JavaScript
    // ─────────────────────────────────────────────────────────────────────────

    async fn evaluate_js(&self, handle: &PageHandle, script: &str) -> WebAdapterResult<Value> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::JavaScript("Page not found".to_string()));
        }

        // Return mock JS result
        Ok(serde_json::json!({
            "script": script,
            "result": "mock result"
        }))
    }

    async fn inject_script(&self, handle: &PageHandle, _script: &str) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let pages = self.pages.read().await;
        if !pages.contains_key(&handle.id) {
            return Err(WebAdapterError::JavaScript("Page not found".to_string()));
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cookies & Storage
    // ─────────────────────────────────────────────────────────────────────────

    async fn get_cookies(&self, _handle: &PageHandle) -> WebAdapterResult<Vec<Cookie>> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let cookies = self.cookies.read().await;
        Ok(cookies.clone())
    }

    async fn set_cookie(&self, _handle: &PageHandle, cookie: Cookie) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let mut cookies = self.cookies.write().await;
        cookies.push(cookie);
        Ok(())
    }

    async fn clear_cookies(&self, _handle: &PageHandle) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let mut cookies = self.cookies.write().await;
        cookies.clear();
        Ok(())
    }

    async fn get_local_storage(
        &self,
        _handle: &PageHandle,
        key: &str,
    ) -> WebAdapterResult<Option<String>> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let storage = self.local_storage.read().await;
        Ok(storage.get(key).cloned())
    }

    async fn set_local_storage(
        &self,
        _handle: &PageHandle,
        key: &str,
        value: &str,
    ) -> WebAdapterResult<()> {
        self.increment_ops();

        if !self.is_connected() {
            return Err(WebAdapterError::NotConnected);
        }

        let mut storage = self.local_storage.write().await;
        storage.insert(key.to_string(), value.to_string());
        Ok(())
    }
}

// ============================================================================
// MEMORY SERVICE TRAIT TESTS
// ============================================================================

mod memory_service_tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Storage Operations Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_store_and_retrieve_document() {
        let service = MockMemoryService::new();

        let doc = Document {
            id: None,
            content: "Test document content for storage verification".to_string(),
            metadata: HashMap::from([("key".to_string(), "value".to_string())]),
            source: Some("test".to_string()),
            created_at: None,
        };

        // Store document
        let id = service.store_document(&doc).await.unwrap();
        assert!(!id.is_nil(), "Should return valid UUID");

        // Retrieve document
        let retrieved = service.get_by_id(id).await.unwrap();
        assert!(retrieved.is_some(), "Should find stored document");

        let retrieved_doc = retrieved.unwrap();
        assert_eq!(retrieved_doc.content, doc.content);
        assert_eq!(retrieved_doc.id, Some(id));
    }

    #[tokio::test]
    async fn test_store_multiple_documents() {
        let service = MockMemoryService::new();

        let docs: Vec<Document> = (0..5)
            .map(|i| Document {
                id: None,
                content: format!("Document {} content", i),
                metadata: HashMap::new(),
                source: Some(format!("source_{}", i)),
                created_at: None,
            })
            .collect();

        let mut ids = Vec::new();
        for doc in &docs {
            let id = service.store_document(doc).await.unwrap();
            ids.push(id);
        }

        // Verify all documents can be retrieved
        for (i, id) in ids.iter().enumerate() {
            let doc = service.get_by_id(*id).await.unwrap().unwrap();
            assert!(doc.content.contains(&format!("Document {}", i)));
        }
    }

    #[tokio::test]
    async fn test_store_and_retrieve_chunks() {
        let service = MockMemoryService::new();
        let doc_id = Uuid::new_v4();

        let chunks: Vec<Chunk> = (0..3)
            .map(|i| Chunk {
                id: None,
                document_id: doc_id,
                content: format!("Chunk {} content", i),
                index: i,
                embedding: None,
                metadata: HashMap::new(),
            })
            .collect();

        let ids = service.store_chunks(&chunks).await.unwrap();
        assert_eq!(ids.len(), 3, "Should return 3 chunk IDs");

        // Verify chunks are stored (via search)
        let results = service.search("Chunk", 10).await.unwrap();
        assert_eq!(results.len(), 3, "Should find all 3 chunks");
    }

    #[tokio::test]
    async fn test_delete_document() {
        let service = MockMemoryService::new();

        let doc = Document {
            id: None,
            content: "Document to delete".to_string(),
            metadata: HashMap::new(),
            source: None,
            created_at: None,
        };

        let id = service.store_document(&doc).await.unwrap();

        // Verify it exists
        assert!(service.get_by_id(id).await.unwrap().is_some());

        // Delete it
        service.delete_document(id).await.unwrap();

        // Verify it's gone
        assert!(service.get_by_id(id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete_nonexistent_document() {
        let service = MockMemoryService::new();
        let fake_id = Uuid::new_v4();

        let result = service.delete_document(fake_id).await;
        assert!(result.is_err(), "Should error on nonexistent document");

        match result.unwrap_err() {
            MemoryError::NotFound(id) => assert_eq!(id, fake_id),
            _ => panic!("Expected NotFound error"),
        }
    }

    #[tokio::test]
    async fn test_update_document() {
        let service = MockMemoryService::new();

        let doc = Document {
            id: None,
            content: "Original content".to_string(),
            metadata: HashMap::new(),
            source: None,
            created_at: None,
        };

        let id = service.store_document(&doc).await.unwrap();

        // Update the document
        let updated_doc = Document {
            id: Some(id),
            content: "Updated content".to_string(),
            metadata: HashMap::from([("updated".to_string(), "true".to_string())]),
            source: None,
            created_at: None,
        };

        service.update_document(id, &updated_doc).await.unwrap();

        // Verify update
        let retrieved = service.get_by_id(id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Updated content");
        assert_eq!(retrieved.metadata.get("updated"), Some(&"true".to_string()));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Retrieval Operations Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_search_basic() {
        let service = MockMemoryService::new();
        let doc_id = Uuid::new_v4();

        let chunks = vec![
            Chunk {
                id: None,
                document_id: doc_id,
                content: "Machine learning is a subset of artificial intelligence".to_string(),
                index: 0,
                embedding: None,
                metadata: HashMap::new(),
            },
            Chunk {
                id: None,
                document_id: doc_id,
                content: "Deep learning uses neural networks".to_string(),
                index: 1,
                embedding: None,
                metadata: HashMap::new(),
            },
        ];

        service.store_chunks(&chunks).await.unwrap();

        let results = service.search("machine learning", 5).await.unwrap();
        assert!(!results.is_empty(), "Should find relevant chunks");
        assert!(results[0].score > 0.0, "Should have positive score");
        assert_eq!(results[0].source, RetrievalSource::Vector);
    }

    #[tokio::test]
    async fn test_search_top_k_limit() {
        let service = MockMemoryService::new();
        let doc_id = Uuid::new_v4();

        // Store many chunks
        let chunks: Vec<Chunk> = (0..20)
            .map(|i| Chunk {
                id: None,
                document_id: doc_id,
                content: format!("Content chunk number {}", i),
                index: i,
                embedding: None,
                metadata: HashMap::new(),
            })
            .collect();

        service.store_chunks(&chunks).await.unwrap();

        let results = service.search("content", 5).await.unwrap();
        assert_eq!(results.len(), 5, "Should respect top_k limit");
    }

    #[tokio::test]
    async fn test_hybrid_search() {
        let service = MockMemoryService::new();
        let doc_id = Uuid::new_v4();

        let chunks = vec![Chunk {
            id: None,
            document_id: doc_id,
            content: "Hybrid search combines vector and keyword search".to_string(),
            index: 0,
            embedding: None,
            metadata: HashMap::new(),
        }];

        service.store_chunks(&chunks).await.unwrap();

        let config = HybridConfig {
            vector_weight: 0.7,
            bm25_weight: 0.3,
            use_reranker: true,
            reranker_top_k: 5,
        };

        let results = service
            .hybrid_search("hybrid search", 10, config)
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].source, RetrievalSource::Hybrid);
    }

    #[tokio::test]
    async fn test_get_context_window() {
        let service = MockMemoryService::new();
        let doc_id = Uuid::new_v4();

        let chunks: Vec<Chunk> = (0..5)
            .map(|i| Chunk {
                id: None,
                document_id: doc_id,
                content: format!("This is test chunk number {} with some content words", i),
                index: i,
                embedding: None,
                metadata: HashMap::new(),
            })
            .collect();

        service.store_chunks(&chunks).await.unwrap();

        let context = service.get_context("test chunk", 100).await.unwrap();
        assert!(!context.chunks.is_empty(), "Should have chunks");
        assert!(context.total_tokens <= 100, "Should respect token limit");
    }

    #[tokio::test]
    async fn test_get_context_truncation() {
        let service = MockMemoryService::new();
        let doc_id = Uuid::new_v4();

        // Create chunks with known word counts
        let chunks: Vec<Chunk> = (0..10)
            .map(|i| Chunk {
                id: None,
                document_id: doc_id,
                content: format!("Word1 word2 word3 word4 word5 chunk{}", i),
                index: i,
                embedding: None,
                metadata: HashMap::new(),
            })
            .collect();

        service.store_chunks(&chunks).await.unwrap();

        // Request very small context
        let context = service.get_context("word", 10).await.unwrap();
        assert!(
            context.truncated,
            "Should be truncated with small token limit"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Embedding Operations Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_embed_single_text() {
        let service = MockMemoryService::new();

        let embedding = service.embed("Test text for embedding").await.unwrap();
        assert_eq!(embedding.len(), 384, "Should have 384 dimensions");
        assert!(embedding.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[tokio::test]
    async fn test_embed_batch() {
        let service = MockMemoryService::new();

        let texts = vec!["Text one", "Text two", "Text three"];
        let embeddings = service.embed_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3, "Should return 3 embeddings");
        for emb in &embeddings {
            assert_eq!(emb.len(), 384, "Each embedding should have 384 dimensions");
        }
    }

    #[tokio::test]
    async fn test_embed_deterministic() {
        let service = MockMemoryService::new();
        let text = "Same text produces same embedding";

        let emb1 = service.embed(text).await.unwrap();
        let emb2 = service.embed(text).await.unwrap();

        assert_eq!(emb1, emb2, "Same text should produce same embedding");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Index Management Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_index() {
        let service = MockMemoryService::new();

        let config = IndexConfig {
            name: "test_index".to_string(),
            dimensions: 768,
            metric: DistanceMetric::Cosine,
            ef_construction: 200,
            m: 16,
        };

        let result = service.create_index(config).await;
        assert!(result.is_ok(), "Index creation should succeed");
    }

    #[tokio::test]
    async fn test_rebuild_index() {
        let service = MockMemoryService::new();
        let doc_id = Uuid::new_v4();

        // Store some chunks first
        let chunks = vec![Chunk {
            id: None,
            document_id: doc_id,
            content: "Content to reindex".to_string(),
            index: 0,
            embedding: None,
            metadata: HashMap::new(),
        }];
        service.store_chunks(&chunks).await.unwrap();

        let result = service.rebuild_index().await;
        assert!(result.is_ok(), "Index rebuild should succeed");
    }

    #[tokio::test]
    async fn test_get_stats() {
        let service = MockMemoryService::new();

        // Store some data
        let doc = Document {
            id: None,
            content: "Stats test".to_string(),
            metadata: HashMap::new(),
            source: None,
            created_at: None,
        };
        service.store_document(&doc).await.unwrap();

        let stats = service.get_stats().await.unwrap();
        assert_eq!(stats.total_documents, 1);
        assert!(stats.index_size_bytes > 0);
        assert!(stats.last_updated > 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Health & Lifecycle Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_health_check_healthy() {
        let service = MockMemoryService::new();

        let healthy = service.health_check().await.unwrap();
        assert!(healthy, "New service should be healthy");
    }

    #[tokio::test]
    async fn test_health_check_unhealthy() {
        let service = MockMemoryService::new();
        service.set_healthy(false);

        let healthy = service.health_check().await.unwrap();
        assert!(!healthy, "Service should report unhealthy");
    }

    #[tokio::test]
    async fn test_unhealthy_service_rejects_operations() {
        let service = MockMemoryService::new();
        service.set_healthy(false);

        let doc = Document {
            id: None,
            content: "Test".to_string(),
            metadata: HashMap::new(),
            source: None,
            created_at: None,
        };

        let result = service.store_document(&doc).await;
        assert!(
            result.is_err(),
            "Unhealthy service should reject operations"
        );
    }

    #[tokio::test]
    async fn test_flush() {
        let service = MockMemoryService::new();
        let result = service.flush().await;
        assert!(result.is_ok(), "Flush should succeed");
    }

    #[tokio::test]
    async fn test_shutdown() {
        let service = MockMemoryService::new();

        let result = service.shutdown().await;
        assert!(result.is_ok(), "Shutdown should succeed");

        // Service should be unhealthy after shutdown
        let healthy = service.health_check().await.unwrap();
        assert!(!healthy, "Service should be unhealthy after shutdown");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // End-to-End Workflow Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_full_rag_workflow() {
        let service = MockMemoryService::new();

        // Step 1: Ingest documents
        let docs: Vec<Document> = vec![
            Document {
                id: None,
                content: "Machine learning enables computers to learn from data".to_string(),
                metadata: HashMap::from([("topic".to_string(), "ML".to_string())]),
                source: Some("textbook".to_string()),
                created_at: None,
            },
            Document {
                id: None,
                content: "Neural networks are inspired by biological neurons".to_string(),
                metadata: HashMap::from([("topic".to_string(), "DL".to_string())]),
                source: Some("paper".to_string()),
                created_at: None,
            },
        ];

        let mut doc_ids = Vec::new();
        for doc in &docs {
            let id = service.store_document(doc).await.unwrap();
            doc_ids.push(id);
        }

        // Step 2: Create chunks with embeddings
        for (i, doc) in docs.iter().enumerate() {
            let chunks = vec![Chunk {
                id: None,
                document_id: doc_ids[i],
                content: doc.content.clone(),
                index: 0,
                embedding: Some(service.embed(&doc.content).await.unwrap()),
                metadata: doc.metadata.clone(),
            }];
            service.store_chunks(&chunks).await.unwrap();
        }

        // Step 3: Search for relevant content
        let results = service.search("machine learning neural", 5).await.unwrap();
        assert!(!results.is_empty(), "Should find relevant content");

        // Step 4: Get context for LLM
        let context = service
            .get_context("explain machine learning", 1000)
            .await
            .unwrap();
        assert!(!context.chunks.is_empty(), "Should have context chunks");

        // Step 5: Verify stats
        let stats = service.get_stats().await.unwrap();
        assert_eq!(stats.total_documents, 2);
        assert_eq!(stats.total_chunks, 2);
    }

    #[tokio::test]
    async fn test_operation_counting() {
        let service = MockMemoryService::new();

        assert_eq!(service.operation_count(), 0);

        let doc = Document {
            id: None,
            content: "Test".to_string(),
            metadata: HashMap::new(),
            source: None,
            created_at: None,
        };

        let id = service.store_document(&doc).await.unwrap();
        service.get_by_id(id).await.unwrap();
        service.search("test", 5).await.unwrap();

        assert_eq!(service.operation_count(), 3, "Should count all operations");
    }
}

// ============================================================================
// WEB BROWSER ADAPTER TRAIT TESTS
// ============================================================================

mod web_browser_adapter_tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_connect_and_disconnect() {
        let mut adapter = MockWebBrowserAdapter::new();

        assert!(!adapter.is_connected(), "Should start disconnected");

        adapter.connect().await.unwrap();
        assert!(
            adapter.is_connected(),
            "Should be connected after connect()"
        );

        adapter.disconnect().await.unwrap();
        assert!(
            !adapter.is_connected(),
            "Should be disconnected after disconnect()"
        );
    }

    #[tokio::test]
    async fn test_operations_require_connection() {
        let adapter = MockWebBrowserAdapter::new();

        let result = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await;
        assert!(result.is_err(), "Should fail when not connected");

        match result.unwrap_err() {
            WebAdapterError::NotConnected => {}
            _ => panic!("Expected NotConnected error"),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Navigation Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_navigate_basic() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        assert_eq!(handle.url, "https://example.com");
        assert!(handle.title.is_some());
        assert_eq!(handle.status_code, 200);
        assert!(handle.load_time_ms > 0);
    }

    #[tokio::test]
    async fn test_navigate_with_options() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let options = NavigateOptions {
            wait_until: WaitUntil::NetworkIdle,
            timeout: Duration::from_secs(60),
            user_agent: Some("Custom Agent".to_string()),
            headers: vec![("X-Custom".to_string(), "value".to_string())],
            viewport: Some(Viewport {
                width: 1280,
                height: 720,
                device_scale_factor: 2.0,
                is_mobile: false,
            }),
        };

        let handle = adapter
            .navigate("https://example.com/page", options)
            .await
            .unwrap();
        assert!(handle.url.contains("example.com"));
    }

    #[tokio::test]
    async fn test_wait_for_load() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let result = adapter.wait_for_load(&handle, Duration::from_secs(5)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_navigation_history() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        // Test back/forward (mock just verifies page exists)
        adapter.go_back(&handle).await.unwrap();
        adapter.go_forward(&handle).await.unwrap();
        adapter.reload(&handle).await.unwrap();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Content Extraction Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_extract_content_plain_text() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let options = ExtractOptions {
            format: ExtractFormat::PlainText,
            ..Default::default()
        };

        let content = adapter.extract_content(&handle, options).await.unwrap();
        assert!(!content.text.is_empty());
        assert_eq!(content.format, ExtractFormat::PlainText);
        assert!(content.word_count > 0);
    }

    #[tokio::test]
    async fn test_extract_content_markdown() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let options = ExtractOptions {
            format: ExtractFormat::Markdown,
            include_links: true,
            include_images: false,
            ..Default::default()
        };

        let content = adapter.extract_content(&handle, options).await.unwrap();
        assert!(content.text.contains('#'), "Markdown should have headers");
        assert!(!content.links.is_empty(), "Should include links");
        assert!(content.images.is_empty(), "Should exclude images");
    }

    #[tokio::test]
    async fn test_extract_links() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let links = adapter.extract_links(&handle).await.unwrap();
        assert!(!links.is_empty(), "Should find links");

        for link in &links {
            assert!(!link.text.is_empty());
            assert!(!link.href.is_empty());
        }
    }

    #[tokio::test]
    async fn test_extract_structured() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let data = adapter
            .extract_structured(&handle, "div.content")
            .await
            .unwrap();
        assert!(data.is_object(), "Should return JSON object");
        assert!(data.get("matches").is_some());
    }

    #[tokio::test]
    async fn test_get_html() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let html = adapter.get_html(&handle).await.unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<html>"));
        assert!(html.contains("</html>"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Capture Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_capture_screenshot() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let options = CaptureOptions {
            format: CaptureFormat::Png,
            quality: 90,
            full_page: false,
            clip: None,
        };

        let captured = adapter.capture_screenshot(&handle, options).await.unwrap();
        assert!(!captured.data.is_empty());
        assert_eq!(captured.format, CaptureFormat::Png);
        assert_eq!(captured.width, 1920);
        assert_eq!(captured.height, 1080);
    }

    #[tokio::test]
    async fn test_capture_full_page_screenshot() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let options = CaptureOptions {
            format: CaptureFormat::Jpeg,
            quality: 80,
            full_page: true,
            clip: None,
        };

        let captured = adapter.capture_screenshot(&handle, options).await.unwrap();
        assert!(captured.height > 1080, "Full page should be taller");
    }

    #[tokio::test]
    async fn test_capture_pdf() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let pdf = adapter.capture_pdf(&handle).await.unwrap();
        assert!(!pdf.is_empty());
        // Check PDF magic bytes
        assert_eq!(&pdf[0..4], &[0x25, 0x50, 0x44, 0x46]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Interaction Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_click_element() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let result = adapter.click(&handle, "button.submit").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_type_text() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let result = adapter
            .type_text(&handle, "input.search", "test query")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_select_option() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let result = adapter.select_option(&handle, "select#country", "US").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_scroll() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let result = adapter.scroll(&handle, 0.0, 500.0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_for_selector() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let result = adapter
            .wait_for_selector(&handle, "div.loaded", Duration::from_secs(5))
            .await;
        assert!(result.is_ok());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JavaScript Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_evaluate_js() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let result = adapter
            .evaluate_js(&handle, "document.title")
            .await
            .unwrap();

        assert!(result.is_object());
        assert!(result.get("result").is_some());
    }

    #[tokio::test]
    async fn test_inject_script() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        let result = adapter
            .inject_script(&handle, "window.customVar = 'test';")
            .await;
        assert!(result.is_ok());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cookies & Storage Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_cookie_operations() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        // Initially no cookies
        let cookies = adapter.get_cookies(&handle).await.unwrap();
        assert!(cookies.is_empty());

        // Set a cookie
        let cookie = Cookie {
            name: "session".to_string(),
            value: "abc123".to_string(),
            domain: Some("example.com".to_string()),
            path: Some("/".to_string()),
            expires: None,
            http_only: true,
            secure: true,
            same_site: Some("Strict".to_string()),
        };
        adapter.set_cookie(&handle, cookie).await.unwrap();

        // Verify cookie was set
        let cookies = adapter.get_cookies(&handle).await.unwrap();
        assert_eq!(cookies.len(), 1);
        assert_eq!(cookies[0].name, "session");

        // Clear cookies
        adapter.clear_cookies(&handle).await.unwrap();
        let cookies = adapter.get_cookies(&handle).await.unwrap();
        assert!(cookies.is_empty());
    }

    #[tokio::test]
    async fn test_local_storage_operations() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();

        // Initially empty
        let value = adapter.get_local_storage(&handle, "key").await.unwrap();
        assert!(value.is_none());

        // Set value
        adapter
            .set_local_storage(&handle, "key", "value")
            .await
            .unwrap();

        // Get value
        let value = adapter.get_local_storage(&handle, "key").await.unwrap();
        assert_eq!(value, Some("value".to_string()));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // End-to-End Workflow Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_full_web_scraping_workflow() {
        let mut adapter = MockWebBrowserAdapter::new();

        // Connect
        adapter.connect().await.unwrap();

        // Navigate to page
        let options = NavigateOptions {
            wait_until: WaitUntil::NetworkIdle,
            timeout: Duration::from_secs(30),
            ..Default::default()
        };
        let handle = adapter
            .navigate("https://example.com/data", options)
            .await
            .unwrap();

        // Wait for dynamic content
        adapter
            .wait_for_load(&handle, Duration::from_secs(10))
            .await
            .unwrap();

        // Extract content
        let content = adapter
            .extract_content(
                &handle,
                ExtractOptions {
                    format: ExtractFormat::Markdown,
                    include_links: true,
                    include_images: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(content.title.is_some());
        assert!(!content.text.is_empty());

        // Extract links for further crawling
        let links = adapter.extract_links(&handle).await.unwrap();
        assert!(!links.is_empty());

        // Capture screenshot for documentation
        let screenshot = adapter
            .capture_screenshot(
                &handle,
                CaptureOptions {
                    format: CaptureFormat::Png,
                    full_page: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(!screenshot.data.is_empty());

        // Disconnect
        adapter.disconnect().await.unwrap();
    }

    #[tokio::test]
    async fn test_form_interaction_workflow() {
        let mut adapter = MockWebBrowserAdapter::new();
        adapter.connect().await.unwrap();

        let handle = adapter
            .navigate("https://example.com/form", NavigateOptions::default())
            .await
            .unwrap();

        // Fill form fields
        adapter
            .type_text(&handle, "input#username", "testuser")
            .await
            .unwrap();
        adapter
            .type_text(&handle, "input#password", "secret")
            .await
            .unwrap();
        adapter
            .select_option(&handle, "select#country", "US")
            .await
            .unwrap();

        // Submit form
        adapter.click(&handle, "button[type=submit]").await.unwrap();

        // Wait for result
        adapter
            .wait_for_selector(&handle, ".success-message", Duration::from_secs(5))
            .await
            .unwrap();

        adapter.disconnect().await.unwrap();
    }

    #[tokio::test]
    async fn test_operation_counting() {
        let mut adapter = MockWebBrowserAdapter::new();

        assert_eq!(adapter.operation_count(), 0);

        adapter.connect().await.unwrap();
        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();
        adapter.extract_links(&handle).await.unwrap();
        adapter.disconnect().await.unwrap();

        assert_eq!(adapter.operation_count(), 4, "Should count all operations");
    }
}

// ============================================================================
// CROSS-TRAIT INTEGRATION TESTS
// ============================================================================

mod cross_trait_integration_tests {
    use super::*;

    /// Test a workflow that uses both MemoryService and WebBrowserAdapter.
    /// This simulates a web content indexing pipeline.
    #[tokio::test]
    async fn test_web_to_memory_pipeline() {
        // Initialize services
        let mut browser = MockWebBrowserAdapter::new();
        let memory = MockMemoryService::new();

        // Connect browser
        browser.connect().await.unwrap();

        // Crawl a page
        let handle = browser
            .navigate("https://example.com/article", NavigateOptions::default())
            .await
            .unwrap();

        // Extract content
        let content = browser
            .extract_content(
                &handle,
                ExtractOptions {
                    format: ExtractFormat::PlainText,
                    include_metadata: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Store in memory
        let doc = Document {
            id: None,
            content: content.text,
            metadata: HashMap::from([
                ("title".to_string(), content.title.unwrap_or_default()),
                ("url".to_string(), handle.url.clone()),
            ]),
            source: Some(handle.url),
            created_at: None,
        };

        let doc_id = memory.store_document(&doc).await.unwrap();

        // Create and store chunks
        let chunks = vec![Chunk {
            id: None,
            document_id: doc_id,
            content: doc.content.clone(),
            index: 0,
            embedding: Some(memory.embed(&doc.content).await.unwrap()),
            metadata: doc.metadata.clone(),
        }];

        memory.store_chunks(&chunks).await.unwrap();

        // Verify we can search for the content
        let results = memory.search("mock content", 5).await.unwrap();
        assert!(!results.is_empty(), "Should find indexed web content");

        // Cleanup
        browser.disconnect().await.unwrap();
    }

    /// Test concurrent operations across both traits.
    #[tokio::test]
    async fn test_concurrent_trait_operations() {
        let browser = Arc::new(tokio::sync::RwLock::new(MockWebBrowserAdapter::new()));
        let memory = Arc::new(MockMemoryService::new());

        // Connect browser
        browser.write().await.connect().await.unwrap();

        // Run concurrent operations
        let browser_clone = Arc::clone(&browser);
        let memory_clone = Arc::clone(&memory);

        let web_task = tokio::spawn(async move {
            let browser = browser_clone.read().await;
            let handle = browser
                .navigate("https://example.com", NavigateOptions::default())
                .await
                .unwrap();
            browser.extract_links(&handle).await.unwrap()
        });

        let memory_task = tokio::spawn(async move {
            let doc = Document {
                id: None,
                content: "Concurrent test".to_string(),
                metadata: HashMap::new(),
                source: None,
                created_at: None,
            };
            memory_clone.store_document(&doc).await.unwrap()
        });

        // Wait for both to complete
        let (links, doc_id) = tokio::join!(web_task, memory_task);
        assert!(!links.unwrap().is_empty());
        assert!(!doc_id.unwrap().is_nil());

        // Cleanup
        browser.write().await.disconnect().await.unwrap();
    }
}

// ============================================================================
// TRAIT COMPATIBILITY TESTS
// ============================================================================

mod trait_compatibility_tests {
    use super::*;

    /// Verify that trait objects work correctly (dyn MemoryService).
    #[tokio::test]
    async fn test_memory_service_as_trait_object() {
        let service: Box<dyn MemoryService> = Box::new(MockMemoryService::new());

        let doc = Document {
            id: None,
            content: "Trait object test".to_string(),
            metadata: HashMap::new(),
            source: None,
            created_at: None,
        };

        let id = service.store_document(&doc).await.unwrap();
        let retrieved = service.get_by_id(id).await.unwrap();
        assert!(retrieved.is_some());
    }

    /// Verify that trait objects work correctly (dyn WebBrowserAdapter).
    #[tokio::test]
    async fn test_web_adapter_as_trait_object() {
        let mut adapter: Box<dyn WebBrowserAdapter> = Box::new(MockWebBrowserAdapter::new());

        adapter.connect().await.unwrap();
        assert!(adapter.is_connected());

        let handle = adapter
            .navigate("https://example.com", NavigateOptions::default())
            .await
            .unwrap();
        assert!(!handle.id.is_empty());

        adapter.disconnect().await.unwrap();
    }

    /// Test that Arc<dyn MemoryService> works for shared ownership.
    #[tokio::test]
    async fn test_memory_service_shared_ownership() {
        let service: Arc<dyn MemoryService> = Arc::new(MockMemoryService::new());

        let service1 = Arc::clone(&service);
        let service2 = Arc::clone(&service);

        let handle1 = tokio::spawn(async move { service1.health_check().await.unwrap() });

        let handle2 = tokio::spawn(async move { service2.get_stats().await.unwrap() });

        let (health, stats) = tokio::join!(handle1, handle2);
        assert!(health.unwrap());
        assert_eq!(stats.unwrap().total_documents, 0);
    }
}
