//! WebSocket MCP Server Example
//!
//! This example demonstrates a production-ready WebSocket-based MCP server with:
//! - API key authentication (header or first-message)
//! - Subscription tier enforcement
//! - Connection tracking and rate limiting
//! - Graceful shutdown handling
//!
//! # Running the Example
//!
//! ```bash
//! cargo run --example ws_mcp_server
//! ```
//!
//! # Testing with websocat
//!
//! Header-based auth:
//! ```bash
//! websocat -H "Authorization: Bearer rk_test_pro_key" ws://localhost:3000/ws
//! ```
//!
//! Message-based auth:
//! ```bash
//! websocat ws://localhost:3000/ws/auth
//! # Then send: {"api_key": "rk_test_pro_key"}
//! ```

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        ConnectInfo, State,
    },
    http::HeaderMap,
    response::IntoResponse,
    routing::{any, get},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::signal;
use tracing::{debug, error, info, warn, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// Import the ws_auth module (would normally be from the crate)
// For this example, we inline the necessary types

// ============================================================================
// Subscription Tiers (simplified from ws_auth module)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SubscriptionTier {
    Free,
    Pro,
    Team,
    Enterprise,
}

impl SubscriptionTier {
    pub fn max_connections(&self) -> usize {
        match self {
            SubscriptionTier::Free => 1,
            SubscriptionTier::Pro => 5,
            SubscriptionTier::Team => 25,
            SubscriptionTier::Enterprise => 100,
        }
    }

    pub fn rate_limit(&self) -> u32 {
        match self {
            SubscriptionTier::Free => 60,
            SubscriptionTier::Pro => 300,
            SubscriptionTier::Team => 1000,
            SubscriptionTier::Enterprise => 10000,
        }
    }

    pub fn max_message_size(&self) -> usize {
        match self {
            SubscriptionTier::Free => 64 * 1024,
            SubscriptionTier::Pro => 1024 * 1024,
            SubscriptionTier::Team => 10 * 1024 * 1024,
            SubscriptionTier::Enterprise => 100 * 1024 * 1024,
        }
    }
}

impl std::fmt::Display for SubscriptionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubscriptionTier::Free => write!(f, "free"),
            SubscriptionTier::Pro => write!(f, "pro"),
            SubscriptionTier::Team => write!(f, "team"),
            SubscriptionTier::Enterprise => write!(f, "enterprise"),
        }
    }
}

// ============================================================================
// API Key Types
// ============================================================================

#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    pub key_id: String,
    pub owner_id: String,
    pub tier: SubscriptionTier,
    pub expires_at: Option<Instant>,
}

// ============================================================================
// Connection Tracking
// ============================================================================

#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub connection_id: uuid::Uuid,
    pub key_id: String,
    pub tier: SubscriptionTier,
    pub connected_at: Instant,
    pub request_count: u32,
    pub rate_window_start: Instant,
}

#[derive(Debug, Default)]
pub struct ConnectionTracker {
    connections: RwLock<HashMap<uuid::Uuid, ConnectionInfo>>,
    connection_counts: RwLock<HashMap<String, usize>>,
}

impl ConnectionTracker {
    pub fn register(&self, key_info: &ApiKeyInfo) -> Result<ConnectionInfo, String> {
        let mut counts = self.connection_counts.write();
        let current = counts.get(&key_info.key_id).copied().unwrap_or(0);

        if current >= key_info.tier.max_connections() {
            return Err(format!(
                "Connection limit exceeded for tier '{}': max {} connections",
                key_info.tier,
                key_info.tier.max_connections()
            ));
        }

        let now = Instant::now();
        let info = ConnectionInfo {
            connection_id: uuid::Uuid::new_v4(),
            key_id: key_info.key_id.clone(),
            tier: key_info.tier,
            connected_at: now,
            request_count: 0,
            rate_window_start: now,
        };

        *counts.entry(key_info.key_id.clone()).or_insert(0) += 1;
        self.connections
            .write()
            .insert(info.connection_id, info.clone());

        Ok(info)
    }

    pub fn unregister(&self, connection_id: uuid::Uuid) {
        let mut conns = self.connections.write();
        if let Some(info) = conns.remove(&connection_id) {
            let mut counts = self.connection_counts.write();
            if let Some(count) = counts.get_mut(&info.key_id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    counts.remove(&info.key_id);
                }
            }
        }
    }

    pub fn check_rate_limit(&self, connection_id: uuid::Uuid) -> Result<(), String> {
        let mut conns = self.connections.write();
        if let Some(info) = conns.get_mut(&connection_id) {
            let now = Instant::now();

            if now.duration_since(info.rate_window_start) > Duration::from_secs(60) {
                info.rate_window_start = now;
                info.request_count = 0;
            }

            info.request_count += 1;

            if info.request_count > info.tier.rate_limit() {
                return Err(format!(
                    "Rate limit exceeded: {} requests per minute allowed",
                    info.tier.rate_limit()
                ));
            }
        }
        Ok(())
    }

    pub fn total_connections(&self) -> usize {
        self.connections.read().len()
    }
}

// ============================================================================
// Server State
// ============================================================================

#[derive(Clone)]
pub struct ServerState {
    api_keys: Arc<RwLock<HashMap<String, ApiKeyInfo>>>,
    tracker: Arc<ConnectionTracker>,
}

impl ServerState {
    pub fn new() -> Self {
        let mut keys = HashMap::new();

        // Add test API keys
        keys.insert(
            "rk_test_free_key".to_string(),
            ApiKeyInfo {
                key_id: "key_free_001".to_string(),
                owner_id: "user_free".to_string(),
                tier: SubscriptionTier::Free,
                expires_at: None,
            },
        );

        keys.insert(
            "rk_test_pro_key".to_string(),
            ApiKeyInfo {
                key_id: "key_pro_001".to_string(),
                owner_id: "user_pro".to_string(),
                tier: SubscriptionTier::Pro,
                expires_at: None,
            },
        );

        keys.insert(
            "rk_test_team_key".to_string(),
            ApiKeyInfo {
                key_id: "key_team_001".to_string(),
                owner_id: "org_team".to_string(),
                tier: SubscriptionTier::Team,
                expires_at: None,
            },
        );

        keys.insert(
            "rk_test_enterprise_key".to_string(),
            ApiKeyInfo {
                key_id: "key_enterprise_001".to_string(),
                owner_id: "org_enterprise".to_string(),
                tier: SubscriptionTier::Enterprise,
                expires_at: None,
            },
        );

        Self {
            api_keys: Arc::new(RwLock::new(keys)),
            tracker: Arc::new(ConnectionTracker::default()),
        }
    }

    pub fn validate_key(&self, api_key: &str) -> Option<ApiKeyInfo> {
        self.api_keys.read().get(api_key).cloned()
    }

    pub fn extract_key_from_headers(&self, headers: &HeaderMap) -> Option<String> {
        headers
            .get("Authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| {
                if s.starts_with("Bearer ") {
                    s[7..].to_string()
                } else {
                    s.to_string()
                }
            })
    }
}

// ============================================================================
// MCP Protocol Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

impl JsonRpcResponse {
    fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

// ============================================================================
// WebSocket Handlers
// ============================================================================

/// Auth message for first-message authentication
#[derive(Debug, Deserialize)]
struct WsAuthMessage {
    api_key: String,
}

/// Auth result sent back to client
#[derive(Debug, Serialize)]
struct WsAuthResult {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    connection_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rate_limit: Option<u32>,
}

/// WebSocket handler with header-based authentication
async fn ws_handler_header_auth(
    ws: WebSocketUpgrade,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<ServerState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // Extract API key from header
    let api_key = match state.extract_key_from_headers(&headers) {
        Some(key) => key,
        None => {
            warn!(remote_addr = %addr, "Missing API key in header");
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                "Missing Authorization header",
            )
                .into_response();
        }
    };

    // Validate API key
    let key_info = match state.validate_key(&api_key) {
        Some(info) => info,
        None => {
            warn!(remote_addr = %addr, "Invalid API key");
            return (axum::http::StatusCode::UNAUTHORIZED, "Invalid API key").into_response();
        }
    };

    // Register connection
    let conn_info = match state.tracker.register(&key_info) {
        Ok(info) => info,
        Err(e) => {
            warn!(remote_addr = %addr, error = %e, "Connection registration failed");
            return (axum::http::StatusCode::TOO_MANY_REQUESTS, e).into_response();
        }
    };

    info!(
        connection_id = %conn_info.connection_id,
        tier = %conn_info.tier,
        remote_addr = %addr,
        "WebSocket connection authenticated via header"
    );

    let tracker = Arc::clone(&state.tracker);
    let connection_id = conn_info.connection_id;

    ws.on_upgrade(move |socket| handle_mcp_socket(socket, conn_info, tracker))
        .into_response()
}

/// WebSocket handler with first-message authentication
async fn ws_handler_message_auth(
    ws: WebSocketUpgrade,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<ServerState>,
) -> impl IntoResponse {
    info!(remote_addr = %addr, "WebSocket upgrade requested (message auth)");

    ws.on_upgrade(move |socket| handle_unauthenticated_socket(socket, addr, state))
}

/// Handle unauthenticated socket - wait for auth message
async fn handle_unauthenticated_socket(
    mut socket: WebSocket,
    addr: SocketAddr,
    state: ServerState,
) {
    // Wait for auth message with timeout
    let auth_timeout = Duration::from_secs(30);

    let auth_result = tokio::time::timeout(auth_timeout, socket.recv()).await;

    let auth_msg = match auth_result {
        Ok(Some(Ok(Message::Text(text)))) => match serde_json::from_str::<WsAuthMessage>(&text) {
            Ok(msg) => msg,
            Err(e) => {
                let _ = send_auth_error(&mut socket, "Invalid auth message format").await;
                warn!(remote_addr = %addr, error = %e, "Invalid auth message");
                return;
            }
        },
        Ok(Some(Ok(_))) => {
            let _ = send_auth_error(&mut socket, "First message must be JSON auth message").await;
            return;
        }
        Ok(Some(Err(e))) => {
            warn!(remote_addr = %addr, error = %e, "WebSocket error during auth");
            return;
        }
        Ok(None) => {
            warn!(remote_addr = %addr, "Connection closed before auth");
            return;
        }
        Err(_) => {
            let _ = send_auth_error(&mut socket, "Authentication timeout").await;
            warn!(remote_addr = %addr, "Auth timeout");
            return;
        }
    };

    // Validate API key
    let key_info = match state.validate_key(&auth_msg.api_key) {
        Some(info) => info,
        None => {
            let _ = send_auth_error(&mut socket, "Invalid API key").await;
            warn!(remote_addr = %addr, "Invalid API key in auth message");
            return;
        }
    };

    // Register connection
    let conn_info = match state.tracker.register(&key_info) {
        Ok(info) => info,
        Err(e) => {
            let _ = send_auth_error(&mut socket, &e).await;
            warn!(remote_addr = %addr, error = %e, "Connection registration failed");
            return;
        }
    };

    // Send auth success
    let auth_result = WsAuthResult {
        success: true,
        error: None,
        connection_id: Some(conn_info.connection_id.to_string()),
        tier: Some(conn_info.tier.to_string()),
        rate_limit: Some(conn_info.tier.rate_limit()),
    };

    if let Ok(json) = serde_json::to_string(&auth_result) {
        let _ = socket.send(Message::Text(json.into())).await;
    }

    info!(
        connection_id = %conn_info.connection_id,
        tier = %conn_info.tier,
        remote_addr = %addr,
        "WebSocket connection authenticated via message"
    );

    // Continue with MCP handling
    handle_mcp_socket(socket, conn_info, Arc::clone(&state.tracker)).await;
}

async fn send_auth_error(socket: &mut WebSocket, error: &str) -> Result<(), axum::Error> {
    let result = WsAuthResult {
        success: false,
        error: Some(error.to_string()),
        connection_id: None,
        tier: None,
        rate_limit: None,
    };

    if let Ok(json) = serde_json::to_string(&result) {
        socket.send(Message::Text(json.into())).await?;
    }

    socket
        .send(Message::Close(Some(axum::extract::ws::CloseFrame {
            code: axum::extract::ws::close_code::POLICY,
            reason: error.to_string().into(),
        })))
        .await?;

    Ok(())
}

/// Handle authenticated MCP WebSocket connection
async fn handle_mcp_socket(
    mut socket: WebSocket,
    conn_info: ConnectionInfo,
    tracker: Arc<ConnectionTracker>,
) {
    let connection_id = conn_info.connection_id;
    let tier = conn_info.tier;

    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Check rate limit
                if let Err(e) = tracker.check_rate_limit(connection_id) {
                    let error_resp = JsonRpcResponse::error(None, -32000, e);
                    if let Ok(json) = serde_json::to_string(&error_resp) {
                        let _ = socket.send(Message::Text(json.into())).await;
                    }
                    continue;
                }

                // Check message size
                if text.len() > tier.max_message_size() {
                    let error_resp = JsonRpcResponse::error(
                        None,
                        -32000,
                        format!(
                            "Message size {} exceeds tier limit {}",
                            text.len(),
                            tier.max_message_size()
                        ),
                    );
                    if let Ok(json) = serde_json::to_string(&error_resp) {
                        let _ = socket.send(Message::Text(json.into())).await;
                    }
                    continue;
                }

                // Parse JSON-RPC request
                let request: JsonRpcRequest = match serde_json::from_str(&text) {
                    Ok(req) => req,
                    Err(e) => {
                        let error_resp =
                            JsonRpcResponse::error(None, -32700, format!("Parse error: {}", e));
                        if let Ok(json) = serde_json::to_string(&error_resp) {
                            let _ = socket.send(Message::Text(json.into())).await;
                        }
                        continue;
                    }
                };

                debug!(
                    connection_id = %connection_id,
                    method = %request.method,
                    "Processing MCP request"
                );

                // Handle MCP methods
                let response = match request.method.as_str() {
                    "initialize" => handle_initialize(request.id, request.params, &tier),
                    "initialized" => continue, // Notification, no response
                    "tools/list" => handle_tools_list(request.id),
                    "tools/call" => handle_tools_call(request.id, request.params),
                    "ping" => JsonRpcResponse::success(request.id, json!({"pong": true})),
                    _ => JsonRpcResponse::error(
                        request.id,
                        -32601,
                        format!("Method not found: {}", request.method),
                    ),
                };

                if let Ok(json) = serde_json::to_string(&response) {
                    if socket.send(Message::Text(json.into())).await.is_err() {
                        break;
                    }
                }
            }
            Ok(Message::Binary(data)) => {
                // Echo binary for now
                if socket.send(Message::Binary(data)).await.is_err() {
                    break;
                }
            }
            Ok(Message::Ping(data)) => {
                if socket.send(Message::Pong(data)).await.is_err() {
                    break;
                }
            }
            Ok(Message::Pong(_)) => {}
            Ok(Message::Close(_)) => {
                info!(connection_id = %connection_id, "Client initiated close");
                break;
            }
            Err(e) => {
                error!(connection_id = %connection_id, error = %e, "WebSocket error");
                break;
            }
        }
    }

    // Clean up
    tracker.unregister(connection_id);
    info!(connection_id = %connection_id, "Connection closed");
}

// ============================================================================
// MCP Method Handlers
// ============================================================================

fn handle_initialize(
    id: Option<Value>,
    _params: Option<Value>,
    tier: &SubscriptionTier,
) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": { "listChanged": true }
            },
            "serverInfo": {
                "name": "reasonkit-mcp-ws",
                "version": env!("CARGO_PKG_VERSION"),
                "tier": tier.to_string(),
                "rateLimit": tier.rate_limit(),
                "maxMessageSize": tier.max_message_size()
            }
        }),
    )
}

fn handle_tools_list(id: Option<Value>) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo back the input",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message to echo"
                            }
                        },
                        "required": ["message"]
                    }
                },
                {
                    "name": "ping",
                    "description": "Test connectivity",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }),
    )
}

fn handle_tools_call(id: Option<Value>, params: Option<Value>) -> JsonRpcResponse {
    let params = match params {
        Some(p) => p,
        None => {
            return JsonRpcResponse::error(id, -32602, "Missing params");
        }
    };

    let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    match name {
        "echo" => {
            let message = arguments
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("(no message)");

            JsonRpcResponse::success(
                id,
                json!({
                    "content": [
                        {
                            "type": "text",
                            "text": format!("Echo: {}", message)
                        }
                    ]
                }),
            )
        }
        "ping" => JsonRpcResponse::success(
            id,
            json!({
                "content": [
                    {
                        "type": "text",
                        "text": "pong"
                    }
                ]
            }),
        ),
        _ => JsonRpcResponse::error(id, -32602, format!("Unknown tool: {}", name)),
    }
}

// ============================================================================
// REST Endpoints
// ============================================================================

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "reasonkit-mcp-ws",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Server stats endpoint
async fn server_stats(State(state): State<ServerState>) -> impl IntoResponse {
    Json(json!({
        "activeConnections": state.tracker.total_connections(),
        "uptime": "TODO"
    }))
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,ws_mcp_server=debug")),
        )
        .init();

    info!("Starting ReasonKit MCP WebSocket Server");

    // Create server state
    let state = ServerState::new();

    // Build router
    let app = Router::new()
        // WebSocket endpoints
        .route("/ws", any(ws_handler_header_auth))
        .route("/ws/auth", any(ws_handler_message_auth))
        // REST endpoints
        .route("/health", get(health_check))
        .route("/stats", get(server_stats))
        .with_state(state);

    // Bind to address
    let addr = "0.0.0.0:3000";
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    info!("Listening on {}", addr);
    info!("WebSocket endpoints:");
    info!("  - wss://localhost:3000/ws      (header auth: Authorization: Bearer <key>)");
    info!("  - wss://localhost:3000/ws/auth (first-message auth)");
    info!("");
    info!("Test API keys:");
    info!("  - rk_test_free_key       (Free tier: 1 conn, 60 req/min)");
    info!("  - rk_test_pro_key        (Pro tier: 5 conns, 300 req/min)");
    info!("  - rk_test_team_key       (Team tier: 25 conns, 1000 req/min)");
    info!("  - rk_test_enterprise_key (Enterprise: 100 conns, 10000 req/min)");

    // Run server with graceful shutdown
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .unwrap();

    info!("Server shutdown complete");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        }
        _ = terminate => {
            info!("Received terminate signal, starting graceful shutdown");
        }
    }
}
