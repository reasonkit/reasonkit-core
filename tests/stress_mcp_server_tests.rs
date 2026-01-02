//! Stress Tests for MCP Server
//!
//! This module provides comprehensive stress testing for MCP (Model Context Protocol)
//! server under heavy concurrent load:
//! - 1000+ concurrent MCP requests
//! - Request/response latency under load
//! - Memory leak detection during sustained load
//! - Connection handling stress
//!
//! ## Running Stress Tests
//!
//! ```bash
//! # Run all MCP stress tests
//! cargo test --test stress_mcp_server_tests --release -- --nocapture --test-threads=1
//!
//! # Run specific stress test
//! cargo test stress_concurrent_mcp_requests --release -- --nocapture
//! ```
//!
//! ## Test Categories
//!
//! 1. **Concurrent Requests**: 1000+ simultaneous MCP tool calls
//! 2. **Request Storm**: Rapid-fire sequential requests
//! 3. **Mixed Workload**: Combination of different MCP operations
//! 4. **Connection Limits**: Maximum connection handling

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{Barrier, RwLock, Semaphore};
use tokio::time::timeout;
use uuid::Uuid;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

/// Number of concurrent MCP requests to simulate
const CONCURRENT_REQUESTS: usize = 1000;

/// Maximum concurrent connections
const MAX_CONNECTIONS: usize = 500;

/// Total requests per stress test
const TOTAL_REQUESTS: usize = 10_000;

/// Stress test timeout (seconds)
const STRESS_TEST_TIMEOUT_SECS: u64 = 300;

/// Memory check interval
const MEMORY_CHECK_INTERVAL: usize = 500;

/// Maximum acceptable memory growth ratio
const MAX_MEMORY_GROWTH_RATIO: f64 = 2.0;

// ============================================================================
// MOCK MCP TYPES
// ============================================================================

/// Mock MCP request for stress testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockMcpRequest {
    pub id: Uuid,
    pub method: String,
    pub params: HashMap<String, serde_json::Value>,
    pub timestamp: u64,
}

impl MockMcpRequest {
    pub fn new(method: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            method: method.to_string(),
            params: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    pub fn with_param(mut self, key: &str, value: serde_json::Value) -> Self {
        self.params.insert(key.to_string(), value);
        self
    }

    pub fn tools_list() -> Self {
        Self::new("tools/list")
    }

    pub fn tools_call(tool_name: &str, args: HashMap<String, serde_json::Value>) -> Self {
        Self::new("tools/call")
            .with_param("name", serde_json::Value::String(tool_name.to_string()))
            .with_param("arguments", serde_json::to_value(args).unwrap())
    }

    pub fn resources_list() -> Self {
        Self::new("resources/list")
    }

    pub fn prompts_list() -> Self {
        Self::new("prompts/list")
    }
}

/// Mock MCP response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockMcpResponse {
    pub id: Uuid,
    pub result: Option<serde_json::Value>,
    pub error: Option<MockMcpError>,
    pub latency_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockMcpError {
    pub code: i32,
    pub message: String,
}

// ============================================================================
// MOCK MCP SERVER
// ============================================================================

/// Mock MCP server for stress testing
///
/// Simulates server behavior without actual network I/O
pub struct MockMcpServer {
    /// Active connections
    connections: AtomicU64,
    /// Total requests handled
    requests_handled: AtomicU64,
    /// Total errors
    errors_total: AtomicU64,
    /// Registered tools
    tools: RwLock<HashMap<String, MockTool>>,
    /// Request processing delay (simulated)
    processing_delay_us: u64,
    /// Maximum connections allowed
    max_connections: usize,
}

#[derive(Debug, Clone)]
struct MockTool {
    name: String,
    description: String,
    complexity: u32, // Simulated processing complexity
}

impl MockMcpServer {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            connections: AtomicU64::new(0),
            requests_handled: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            tools: RwLock::new(HashMap::new()),
            processing_delay_us: 100, // 100 microseconds simulated delay
            max_connections: MAX_CONNECTIONS,
        }
    }

    pub fn with_processing_delay(mut self, delay_us: u64) -> Self {
        self.processing_delay_us = delay_us;
        self
    }

    /// Register a mock tool
    pub async fn register_tool(&self, name: &str, description: &str, complexity: u32) {
        let mut tools = self.tools.write().await;
        tools.insert(
            name.to_string(),
            MockTool {
                name: name.to_string(),
                description: description.to_string(),
                complexity,
            },
        );
    }

    /// Handle an incoming connection
    pub async fn connect(&self) -> Result<MockMcpConnection<'_>, &'static str> {
        let current = self.connections.fetch_add(1, Ordering::SeqCst);
        if current as usize >= self.max_connections {
            self.connections.fetch_sub(1, Ordering::SeqCst);
            return Err("Connection limit reached");
        }
        Ok(MockMcpConnection {
            server: self,
            id: Uuid::new_v4(),
        })
    }

    /// Handle an MCP request
    pub async fn handle_request(&self, request: MockMcpRequest) -> MockMcpResponse {
        let start = Instant::now();

        // Simulate processing delay
        if self.processing_delay_us > 0 {
            tokio::time::sleep(Duration::from_micros(self.processing_delay_us)).await;
        }

        let result = match request.method.as_str() {
            "tools/list" => self.handle_tools_list().await,
            "tools/call" => self.handle_tools_call(&request).await,
            "resources/list" => self.handle_resources_list().await,
            "prompts/list" => self.handle_prompts_list().await,
            _ => Err(MockMcpError {
                code: -32601,
                message: format!("Method not found: {}", request.method),
            }),
        };

        self.requests_handled.fetch_add(1, Ordering::SeqCst);

        let latency_ns = start.elapsed().as_nanos() as u64;

        match result {
            Ok(result) => MockMcpResponse {
                id: request.id,
                result: Some(result),
                error: None,
                latency_ns,
            },
            Err(error) => {
                self.errors_total.fetch_add(1, Ordering::SeqCst);
                MockMcpResponse {
                    id: request.id,
                    result: None,
                    error: Some(error),
                    latency_ns,
                }
            }
        }
    }

    async fn handle_tools_list(&self) -> Result<serde_json::Value, MockMcpError> {
        let tools = self.tools.read().await;
        let tool_list: Vec<serde_json::Value> = tools
            .values()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                })
            })
            .collect();
        Ok(serde_json::json!({ "tools": tool_list }))
    }

    async fn handle_tools_call(
        &self,
        request: &MockMcpRequest,
    ) -> Result<serde_json::Value, MockMcpError> {
        let tool_name = request
            .params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MockMcpError {
                code: -32602,
                message: "Missing tool name".to_string(),
            })?;

        let tools = self.tools.read().await;
        let tool = tools.get(tool_name).ok_or_else(|| MockMcpError {
            code: -32602,
            message: format!("Unknown tool: {}", tool_name),
        })?;

        // Simulate variable processing based on tool complexity
        if tool.complexity > 0 {
            tokio::time::sleep(Duration::from_micros(tool.complexity as u64 * 10)).await;
        }

        Ok(serde_json::json!({
            "content": [{
                "type": "text",
                "text": format!("Executed {} successfully", tool_name)
            }]
        }))
    }

    async fn handle_resources_list(&self) -> Result<serde_json::Value, MockMcpError> {
        Ok(serde_json::json!({
            "resources": []
        }))
    }

    async fn handle_prompts_list(&self) -> Result<serde_json::Value, MockMcpError> {
        Ok(serde_json::json!({
            "prompts": []
        }))
    }

    /// Get server statistics
    pub fn stats(&self) -> McpServerStats {
        McpServerStats {
            active_connections: self.connections.load(Ordering::SeqCst),
            total_requests: self.requests_handled.load(Ordering::SeqCst),
            total_errors: self.errors_total.load(Ordering::SeqCst),
        }
    }
}

pub struct MockMcpConnection<'a> {
    server: &'a MockMcpServer,
    #[allow(dead_code)]
    id: Uuid,
}

impl<'a> MockMcpConnection<'a> {
    pub async fn request(&self, request: MockMcpRequest) -> MockMcpResponse {
        self.server.handle_request(request).await
    }
}

impl<'a> Drop for MockMcpConnection<'a> {
    fn drop(&mut self) {
        self.server.connections.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub struct McpServerStats {
    pub active_connections: u64,
    pub total_requests: u64,
    pub total_errors: u64,
}

// ============================================================================
// STRESS TEST METRICS
// ============================================================================

/// Metrics collected during MCP stress tests
#[derive(Debug, Default)]
pub struct McpStressMetrics {
    pub requests_sent: AtomicU64,
    pub requests_succeeded: AtomicU64,
    pub requests_failed: AtomicU64,
    pub connection_errors: AtomicU64,
    pub total_latency_ns: AtomicU64,
    pub min_latency_ns: AtomicU64,
    pub max_latency_ns: AtomicU64,
}

impl McpStressMetrics {
    pub fn new() -> Self {
        Self {
            min_latency_ns: AtomicU64::new(u64::MAX),
            ..Default::default()
        }
    }

    pub fn record_success(&self, latency_ns: u64) {
        self.requests_sent.fetch_add(1, Ordering::SeqCst);
        self.requests_succeeded.fetch_add(1, Ordering::SeqCst);
        self.total_latency_ns
            .fetch_add(latency_ns, Ordering::SeqCst);

        // Update min
        let mut min = self.min_latency_ns.load(Ordering::SeqCst);
        while latency_ns < min {
            match self.min_latency_ns.compare_exchange_weak(
                min,
                latency_ns,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(m) => min = m,
            }
        }

        // Update max
        let mut max = self.max_latency_ns.load(Ordering::SeqCst);
        while latency_ns > max {
            match self.max_latency_ns.compare_exchange_weak(
                max,
                latency_ns,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(m) => max = m,
            }
        }
    }

    pub fn record_failure(&self) {
        self.requests_sent.fetch_add(1, Ordering::SeqCst);
        self.requests_failed.fetch_add(1, Ordering::SeqCst);
    }

    pub fn record_connection_error(&self) {
        self.connection_errors.fetch_add(1, Ordering::SeqCst);
    }

    pub fn summary(&self) -> McpStressSummary {
        let sent = self.requests_sent.load(Ordering::SeqCst);
        let succeeded = self.requests_succeeded.load(Ordering::SeqCst);
        let failed = self.requests_failed.load(Ordering::SeqCst);
        let total_latency = self.total_latency_ns.load(Ordering::SeqCst);

        McpStressSummary {
            requests_sent: sent,
            requests_succeeded: succeeded,
            requests_failed: failed,
            connection_errors: self.connection_errors.load(Ordering::SeqCst),
            avg_latency_us: if succeeded > 0 {
                (total_latency / succeeded) / 1000
            } else {
                0
            },
            min_latency_us: self.min_latency_ns.load(Ordering::SeqCst) / 1000,
            max_latency_us: self.max_latency_ns.load(Ordering::SeqCst) / 1000,
            success_rate: if sent > 0 {
                succeeded as f64 / sent as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug)]
pub struct McpStressSummary {
    pub requests_sent: u64,
    pub requests_succeeded: u64,
    pub requests_failed: u64,
    pub connection_errors: u64,
    pub avg_latency_us: u64,
    pub min_latency_us: u64,
    pub max_latency_us: u64,
    pub success_rate: f64,
}

impl std::fmt::Display for McpStressSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Requests: {} sent, {} succeeded, {} failed ({:.2}% success)\n\
             Connection errors: {}\n\
             Latency: avg={}us, min={}us, max={}us",
            self.requests_sent,
            self.requests_succeeded,
            self.requests_failed,
            self.success_rate * 100.0,
            self.connection_errors,
            self.avg_latency_us,
            self.min_latency_us,
            self.max_latency_us
        )
    }
}

// ============================================================================
// MEMORY TRACKER (Shared with other stress tests)
// ============================================================================

#[derive(Debug, Default)]
pub struct MemoryTracker {
    initial_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    current_bytes: AtomicU64,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_initial(&self) {
        let current = Self::get_process_memory();
        self.initial_bytes.store(current, Ordering::SeqCst);
        self.peak_bytes.store(current, Ordering::SeqCst);
    }

    pub fn sample(&self) {
        let current = Self::get_process_memory();
        self.current_bytes.store(current, Ordering::SeqCst);

        let mut peak = self.peak_bytes.load(Ordering::SeqCst);
        while current > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                current,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn get_process_memory() -> u64 {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            let parts: Vec<&str> = statm.split_whitespace().collect();
            if let Some(rss_pages) = parts.get(1) {
                if let Ok(pages) = rss_pages.parse::<u64>() {
                    return pages * 4096;
                }
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    fn get_process_memory() -> u64 {
        0
    }

    pub fn growth_ratio(&self) -> f64 {
        let initial = self.initial_bytes.load(Ordering::SeqCst);
        let peak = self.peak_bytes.load(Ordering::SeqCst);
        if initial > 0 {
            peak as f64 / initial as f64
        } else {
            1.0
        }
    }

    pub fn has_leak(&self) -> bool {
        self.growth_ratio() > MAX_MEMORY_GROWTH_RATIO
    }
}

// ============================================================================
// STRESS TEST: 1000 CONCURRENT MCP REQUESTS
// ============================================================================

/// Stress test for handling 1000+ concurrent MCP requests
///
/// Simulates a heavy load scenario where many clients make requests
/// to the MCP server simultaneously.
#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn stress_concurrent_mcp_requests() {
    let server = Arc::new(MockMcpServer::new());
    let metrics = Arc::new(McpStressMetrics::new());
    let memory_tracker = Arc::new(MemoryTracker::new());

    // Register some mock tools
    server
        .register_tool("gigathink", "Expansive creative thinking", 10)
        .await;
    server
        .register_tool("laserlogic", "Precision deductive reasoning", 5)
        .await;
    server
        .register_tool("bedrock", "First principles decomposition", 15)
        .await;
    server
        .register_tool("proofguard", "Multi-source verification", 20)
        .await;
    server
        .register_tool("brutalhonesty", "Adversarial self-critique", 8)
        .await;

    memory_tracker.record_initial();

    let barrier = Arc::new(Barrier::new(CONCURRENT_REQUESTS));
    let semaphore = Arc::new(Semaphore::new(MAX_CONNECTIONS));

    let start = Instant::now();
    let mut handles = Vec::new();

    for _client_id in 0..CONCURRENT_REQUESTS {
        let server = Arc::clone(&server);
        let metrics = Arc::clone(&metrics);
        let memory_tracker = Arc::clone(&memory_tracker);
        let barrier = Arc::clone(&barrier);
        let semaphore = Arc::clone(&semaphore);

        handles.push(tokio::spawn(async move {
            barrier.wait().await;

            let _permit = match semaphore.acquire().await {
                Ok(p) => p,
                Err(_) => {
                    metrics.record_connection_error();
                    return;
                }
            };

            // Establish connection
            let connection = match server.connect().await {
                Ok(c) => c,
                Err(_) => {
                    metrics.record_connection_error();
                    return;
                }
            };

            // Make multiple requests per connection
            let ops_per_client = TOTAL_REQUESTS / CONCURRENT_REQUESTS;
            for i in 0..ops_per_client {
                // Vary request types
                let request = match i % 5 {
                    0 => MockMcpRequest::tools_list(),
                    1 => MockMcpRequest::tools_call("gigathink", HashMap::new()),
                    2 => MockMcpRequest::tools_call("laserlogic", HashMap::new()),
                    3 => MockMcpRequest::resources_list(),
                    _ => MockMcpRequest::prompts_list(),
                };

                let response = connection.request(request).await;

                if response.error.is_none() {
                    metrics.record_success(response.latency_ns);
                } else {
                    metrics.record_failure();
                }

                if i % MEMORY_CHECK_INTERVAL == 0 {
                    memory_tracker.sample();
                }
            }
        }));
    }

    // Wait for all with timeout
    let result = timeout(Duration::from_secs(STRESS_TEST_TIMEOUT_SECS), async {
        for handle in handles {
            let _ = handle.await;
        }
    })
    .await;

    let elapsed = start.elapsed();
    let summary = metrics.summary();
    let server_stats = server.stats();

    println!("\n=== Stress Test: 1000 Concurrent MCP Requests ===");
    println!("Duration: {:?}", elapsed);
    println!("{}", summary);
    println!(
        "Server stats: connections={}, requests={}, errors={}",
        server_stats.active_connections, server_stats.total_requests, server_stats.total_errors
    );
    println!(
        "Memory growth: {:.2}x, leak detected: {}",
        memory_tracker.growth_ratio(),
        memory_tracker.has_leak()
    );

    // Calculate throughput
    let throughput = summary.requests_succeeded as f64 / elapsed.as_secs_f64();
    println!("Throughput: {:.0} requests/sec", throughput);

    // Assertions
    assert!(result.is_ok(), "Test timed out");
    assert!(
        summary.success_rate > 0.95,
        "Success rate too low: {:.2}%",
        summary.success_rate * 100.0
    );
    assert!(
        !memory_tracker.has_leak(),
        "Memory leak detected: {:.2}x growth",
        memory_tracker.growth_ratio()
    );
}

// ============================================================================
// STRESS TEST: MCP REQUEST STORM
// ============================================================================

/// Stress test for rapid-fire sequential requests
///
/// Tests server stability under sustained high-rate request load.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn stress_mcp_request_storm() {
    let server = Arc::new(MockMcpServer::new().with_processing_delay(50));
    let metrics = Arc::new(McpStressMetrics::new());

    server.register_tool("test_tool", "Test tool", 1).await;

    // Establish a single connection and hammer it
    let connection = server.connect().await.unwrap();

    let start = Instant::now();
    let target_duration = Duration::from_secs(10);
    let mut request_count = 0u64;

    while start.elapsed() < target_duration {
        let request = MockMcpRequest::tools_call("test_tool", HashMap::new());
        let response = connection.request(request).await;

        if response.error.is_none() {
            metrics.record_success(response.latency_ns);
        } else {
            metrics.record_failure();
        }

        request_count += 1;

        // Yield occasionally
        if request_count % 100 == 0 {
            tokio::task::yield_now().await;
        }
    }

    let summary = metrics.summary();
    let throughput = request_count as f64 / start.elapsed().as_secs_f64();

    println!("\n=== Stress Test: MCP Request Storm ===");
    println!("Duration: {:?}", start.elapsed());
    println!("Total requests: {}", request_count);
    println!("Throughput: {:.0} requests/sec", throughput);
    println!("{}", summary);

    assert!(
        summary.success_rate > 0.99,
        "Storm test success rate too low: {:.2}%",
        summary.success_rate * 100.0
    );
}

// ============================================================================
// STRESS TEST: MIXED MCP WORKLOAD
// ============================================================================

/// Stress test with mixed MCP operation types
///
/// Simulates realistic workload with varied request patterns.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn stress_mixed_mcp_workload() {
    let server = Arc::new(MockMcpServer::new());
    let metrics = Arc::new(McpStressMetrics::new());

    // Register tools with varying complexity
    server.register_tool("fast_tool", "Fast operation", 1).await;
    server
        .register_tool("medium_tool", "Medium operation", 10)
        .await;
    server
        .register_tool("slow_tool", "Slow operation", 50)
        .await;
    server
        .register_tool("complex_tool", "Complex operation", 100)
        .await;

    let concurrent_clients = 50;
    let requests_per_client = 200;
    let barrier = Arc::new(Barrier::new(concurrent_clients));

    let mut handles = Vec::new();

    for _client_id in 0..concurrent_clients {
        let server = Arc::clone(&server);
        let metrics = Arc::clone(&metrics);
        let barrier = Arc::clone(&barrier);

        handles.push(tokio::spawn(async move {
            barrier.wait().await;

            let connection = server.connect().await.unwrap();

            for i in 0..requests_per_client {
                // Weighted distribution of request types
                let request = match i % 20 {
                    0..=9 => MockMcpRequest::tools_call("fast_tool", HashMap::new()),
                    10..=14 => MockMcpRequest::tools_call("medium_tool", HashMap::new()),
                    15..=17 => MockMcpRequest::tools_call("slow_tool", HashMap::new()),
                    18 => MockMcpRequest::tools_call("complex_tool", HashMap::new()),
                    _ => MockMcpRequest::tools_list(),
                };

                let response = connection.request(request).await;

                if response.error.is_none() {
                    metrics.record_success(response.latency_ns);
                } else {
                    metrics.record_failure();
                }
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    let summary = metrics.summary();

    println!("\n=== Stress Test: Mixed MCP Workload ===");
    println!("{}", summary);

    assert!(
        summary.success_rate > 0.99,
        "Mixed workload success rate too low: {:.2}%",
        summary.success_rate * 100.0
    );
}

// ============================================================================
// STRESS TEST: CONNECTION LIMITS
// ============================================================================

/// Stress test for connection limit handling
///
/// Tests behavior when connection limits are exceeded.
#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn stress_connection_limits() {
    let server = Arc::new(MockMcpServer::new()); // Default MAX_CONNECTIONS
    let metrics = Arc::new(McpStressMetrics::new());

    // Try to establish more connections than allowed
    let connection_attempts = MAX_CONNECTIONS + 200;
    let barrier = Arc::new(Barrier::new(connection_attempts));

    let mut handles = Vec::new();
    let successful_connections = Arc::new(AtomicU64::new(0));
    let failed_connections = Arc::new(AtomicU64::new(0));

    for _ in 0..connection_attempts {
        let server = Arc::clone(&server);
        let metrics = Arc::clone(&metrics);
        let barrier = Arc::clone(&barrier);
        let successful = Arc::clone(&successful_connections);
        let failed = Arc::clone(&failed_connections);

        handles.push(tokio::spawn(async move {
            barrier.wait().await;

            match server.connect().await {
                Ok(connection) => {
                    successful.fetch_add(1, Ordering::SeqCst);

                    // Make a request to verify connection works
                    let request = MockMcpRequest::tools_list();
                    let response = connection.request(request).await;

                    if response.error.is_none() {
                        metrics.record_success(response.latency_ns);
                    } else {
                        metrics.record_failure();
                    }

                    // Hold connection briefly
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(_) => {
                    failed.fetch_add(1, Ordering::SeqCst);
                    metrics.record_connection_error();
                }
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    let successful = successful_connections.load(Ordering::SeqCst);
    let failed = failed_connections.load(Ordering::SeqCst);
    let summary = metrics.summary();

    println!("\n=== Stress Test: Connection Limits ===");
    println!(
        "Connection attempts: {}, successful: {}, rejected: {}",
        connection_attempts, successful, failed
    );
    println!("{}", summary);

    // We expect some connections to be rejected
    assert!(
        failed > 0,
        "Expected some connection rejections when exceeding limit"
    );
    assert!(
        successful <= MAX_CONNECTIONS as u64,
        "More connections than limit: {} > {}",
        successful,
        MAX_CONNECTIONS
    );
}

// ============================================================================
// STRESS TEST: LONG-RUNNING MCP CONNECTIONS
// ============================================================================

/// Stress test for long-running connection stability
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Long-running test, run explicitly with: cargo test stress_long_running_mcp --release -- --ignored"]
async fn stress_long_running_mcp() {
    let server = Arc::new(MockMcpServer::new());
    let metrics = Arc::new(McpStressMetrics::new());
    let memory_tracker = Arc::new(MemoryTracker::new());

    server.register_tool("test_tool", "Test tool", 5).await;

    memory_tracker.record_initial();

    let connection = server.connect().await.unwrap();
    let duration = Duration::from_secs(120); // 2 minutes
    let start = Instant::now();
    let mut iteration = 0u64;

    println!("Starting long-running MCP stress test (2 minutes)...");

    while start.elapsed() < duration {
        iteration += 1;

        let request = MockMcpRequest::tools_call("test_tool", HashMap::new());
        let response = connection.request(request).await;

        if response.error.is_none() {
            metrics.record_success(response.latency_ns);
        } else {
            metrics.record_failure();
        }

        if iteration % 5000 == 0 {
            memory_tracker.sample();
            let elapsed = start.elapsed().as_secs();
            println!(
                "  [{:>3}s] Iteration {}, requests: {}",
                elapsed,
                iteration,
                metrics.summary().requests_succeeded
            );
        }

        if iteration % 100 == 0 {
            tokio::task::yield_now().await;
        }
    }

    let summary = metrics.summary();

    println!("\n=== Stress Test: Long-Running MCP ===");
    println!("Total iterations: {}", iteration);
    println!("{}", summary);
    println!(
        "Memory growth: {:.2}x, leak: {}",
        memory_tracker.growth_ratio(),
        memory_tracker.has_leak()
    );

    assert!(
        summary.success_rate > 0.999,
        "Long-running success rate too low: {:.2}%",
        summary.success_rate * 100.0
    );
}
