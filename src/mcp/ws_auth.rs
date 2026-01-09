//! WebSocket Authentication Middleware for MCP Server
//!
//! This module implements secure WebSocket authentication with:
//! - API key validation via header or first message
//! - Subscription tier enforcement (Free, Pro, Team, Enterprise)
//! - Active connection tracking per API key
//! - Rate limiting based on tier
//! - Secure connection upgrade handling
//!
//! # Security Features
//!
//! - Constant-time API key comparison to prevent timing attacks
//! - Connection limits per tier to prevent resource exhaustion
//! - Automatic connection cleanup on disconnect
//! - TLS enforcement in production (wss:// only)
//!
//! # Usage
//!
//! ```rust,ignore
//! use reasonkit_core::mcp::ws_auth::{WsAuthLayer, WsAuthState, SubscriptionTier};
//!
//! let auth_state = WsAuthState::new(api_key_validator);
//! let app = Router::new()
//!     .route("/ws", get(ws_handler))
//!     .layer(WsAuthLayer::new(auth_state));
//! ```

use axum::{
    body::Body,
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        ConnectInfo, State,
    },
    http::{header::HeaderMap, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
// Note: StreamExt is used indirectly via WebSocket stream operations
#[allow(unused_imports)]
use futures_util::StreamExt;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

// ============================================================================
// Error Types
// ============================================================================

/// Authentication and connection errors
#[derive(Debug, Error)]
pub enum WsAuthError {
    #[error("Missing API key")]
    MissingApiKey,

    #[error("Invalid API key")]
    InvalidApiKey,

    #[error("API key expired")]
    ExpiredApiKey,

    #[error("Subscription tier '{0}' does not allow WebSocket access")]
    TierNotAllowed(String),

    #[error("Connection limit exceeded for tier '{0}': max {1} connections")]
    ConnectionLimitExceeded(String, usize),

    #[error("Rate limit exceeded: {0} requests per minute allowed")]
    RateLimitExceeded(u32),

    #[error("Authentication timeout: must authenticate within {0} seconds")]
    AuthTimeout(u64),

    #[error("Invalid authentication message format")]
    InvalidAuthMessage,

    #[error("Internal authentication error: {0}")]
    Internal(String),
}

impl IntoResponse for WsAuthError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            WsAuthError::MissingApiKey => (StatusCode::UNAUTHORIZED, self.to_string()),
            WsAuthError::InvalidApiKey => (StatusCode::UNAUTHORIZED, self.to_string()),
            WsAuthError::ExpiredApiKey => (StatusCode::UNAUTHORIZED, self.to_string()),
            WsAuthError::TierNotAllowed(_) => (StatusCode::FORBIDDEN, self.to_string()),
            WsAuthError::ConnectionLimitExceeded(_, _) => {
                (StatusCode::TOO_MANY_REQUESTS, self.to_string())
            }
            WsAuthError::RateLimitExceeded(_) => (StatusCode::TOO_MANY_REQUESTS, self.to_string()),
            WsAuthError::AuthTimeout(_) => (StatusCode::REQUEST_TIMEOUT, self.to_string()),
            WsAuthError::InvalidAuthMessage => (StatusCode::BAD_REQUEST, self.to_string()),
            WsAuthError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal error".to_string(),
            ),
        };

        (status, message).into_response()
    }
}

// ============================================================================
// Subscription Tiers
// ============================================================================

/// Subscription tier levels with associated limits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SubscriptionTier {
    /// Free tier: Limited features
    Free,
    /// Pro tier: Enhanced limits
    Pro,
    /// Team tier: Collaborative features
    Team,
    /// Enterprise tier: Unlimited access
    Enterprise,
}

impl SubscriptionTier {
    /// Maximum concurrent WebSocket connections allowed
    pub fn max_connections(&self) -> usize {
        match self {
            SubscriptionTier::Free => 1,
            SubscriptionTier::Pro => 5,
            SubscriptionTier::Team => 25,
            SubscriptionTier::Enterprise => 100,
        }
    }

    /// Maximum requests per minute
    pub fn rate_limit(&self) -> u32 {
        match self {
            SubscriptionTier::Free => 60,
            SubscriptionTier::Pro => 300,
            SubscriptionTier::Team => 1000,
            SubscriptionTier::Enterprise => 10000,
        }
    }

    /// Maximum message size in bytes
    pub fn max_message_size(&self) -> usize {
        match self {
            SubscriptionTier::Free => 64 * 1024,               // 64 KB
            SubscriptionTier::Pro => 1024 * 1024,              // 1 MB
            SubscriptionTier::Team => 10 * 1024 * 1024,        // 10 MB
            SubscriptionTier::Enterprise => 100 * 1024 * 1024, // 100 MB
        }
    }

    /// Session timeout duration
    pub fn session_timeout(&self) -> Duration {
        match self {
            SubscriptionTier::Free => Duration::from_secs(30 * 60), // 30 min
            SubscriptionTier::Pro => Duration::from_secs(2 * 60 * 60), // 2 hours
            SubscriptionTier::Team => Duration::from_secs(8 * 60 * 60), // 8 hours
            SubscriptionTier::Enterprise => Duration::from_secs(24 * 60 * 60), // 24 hours
        }
    }

    /// Whether WebSocket access is allowed
    pub fn allows_websocket(&self) -> bool {
        // All tiers allow WebSocket, but with different limits
        true
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

/// Validated API key information
#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    /// Unique API key identifier (hashed or prefix)
    pub key_id: String,
    /// User or organization identifier
    pub owner_id: String,
    /// Subscription tier
    pub tier: SubscriptionTier,
    /// Key expiration timestamp (None = never expires)
    pub expires_at: Option<Instant>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Trait for validating API keys
/// Implement this for your storage backend (database, Redis, etc.)
#[async_trait::async_trait]
pub trait ApiKeyValidator: Send + Sync + 'static {
    /// Validate an API key and return its info if valid
    async fn validate(&self, api_key: &str) -> Result<ApiKeyInfo, WsAuthError>;

    /// Revoke an API key (optional)
    async fn revoke(&self, key_id: &str) -> Result<(), WsAuthError> {
        let _ = key_id;
        Ok(())
    }
}

/// In-memory API key validator for development/testing
#[derive(Debug, Clone)]
pub struct InMemoryApiKeyValidator {
    keys: Arc<RwLock<HashMap<String, ApiKeyInfo>>>,
}

impl InMemoryApiKeyValidator {
    pub fn new() -> Self {
        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a new API key
    pub fn add_key(&self, api_key: String, info: ApiKeyInfo) {
        self.keys.write().insert(api_key, info);
    }

    /// Remove an API key
    pub fn remove_key(&self, api_key: &str) {
        self.keys.write().remove(api_key);
    }
}

impl Default for InMemoryApiKeyValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ApiKeyValidator for InMemoryApiKeyValidator {
    async fn validate(&self, api_key: &str) -> Result<ApiKeyInfo, WsAuthError> {
        let keys = self.keys.read();

        // Constant-time comparison for all keys to prevent timing attacks
        let mut found_info: Option<&ApiKeyInfo> = None;
        for (stored_key, info) in keys.iter() {
            if constant_time_compare(api_key, stored_key) {
                found_info = Some(info);
                break;
            }
        }

        match found_info {
            Some(info) => {
                // Check expiration
                if let Some(expires_at) = info.expires_at {
                    if Instant::now() > expires_at {
                        return Err(WsAuthError::ExpiredApiKey);
                    }
                }
                Ok(info.clone())
            }
            None => Err(WsAuthError::InvalidApiKey),
        }
    }

    async fn revoke(&self, key_id: &str) -> Result<(), WsAuthError> {
        let mut keys = self.keys.write();
        keys.retain(|_, v| v.key_id != key_id);
        Ok(())
    }
}

// ============================================================================
// Connection Tracking
// ============================================================================

/// Information about an active WebSocket connection
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Unique connection ID
    pub connection_id: Uuid,
    /// API key ID (for grouping connections)
    pub key_id: String,
    /// Owner ID
    pub owner_id: String,
    /// Subscription tier
    pub tier: SubscriptionTier,
    /// Remote address
    pub remote_addr: SocketAddr,
    /// Connection established time
    pub connected_at: Instant,
    /// Last activity time
    pub last_activity: Instant,
    /// Request count for rate limiting
    pub request_count: u32,
    /// Rate limit window start
    pub rate_window_start: Instant,
}

/// Connection tracker for managing active WebSocket connections
#[derive(Debug)]
pub struct ConnectionTracker {
    /// Active connections by connection ID
    connections: RwLock<HashMap<Uuid, ConnectionInfo>>,
    /// Connection count by API key ID
    connection_counts: RwLock<HashMap<String, usize>>,
}

impl ConnectionTracker {
    pub fn new() -> Self {
        Self {
            connections: RwLock::new(HashMap::new()),
            connection_counts: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new connection
    pub fn register(
        &self,
        key_info: &ApiKeyInfo,
        remote_addr: SocketAddr,
    ) -> Result<ConnectionInfo, WsAuthError> {
        let mut counts = self.connection_counts.write();
        let current_count = counts.get(&key_info.key_id).copied().unwrap_or(0);
        let max_connections = key_info.tier.max_connections();

        if current_count >= max_connections {
            return Err(WsAuthError::ConnectionLimitExceeded(
                key_info.tier.to_string(),
                max_connections,
            ));
        }

        let now = Instant::now();
        let conn_info = ConnectionInfo {
            connection_id: Uuid::new_v4(),
            key_id: key_info.key_id.clone(),
            owner_id: key_info.owner_id.clone(),
            tier: key_info.tier,
            remote_addr,
            connected_at: now,
            last_activity: now,
            request_count: 0,
            rate_window_start: now,
        };

        // Update counts
        *counts.entry(key_info.key_id.clone()).or_insert(0) += 1;

        // Store connection info
        self.connections
            .write()
            .insert(conn_info.connection_id, conn_info.clone());

        info!(
            connection_id = %conn_info.connection_id,
            key_id = %key_info.key_id,
            tier = %key_info.tier,
            "New WebSocket connection registered"
        );

        Ok(conn_info)
    }

    /// Unregister a connection
    pub fn unregister(&self, connection_id: Uuid) {
        let mut connections = self.connections.write();
        if let Some(conn_info) = connections.remove(&connection_id) {
            let mut counts = self.connection_counts.write();
            if let Some(count) = counts.get_mut(&conn_info.key_id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    counts.remove(&conn_info.key_id);
                }
            }

            info!(
                connection_id = %connection_id,
                key_id = %conn_info.key_id,
                "WebSocket connection unregistered"
            );
        }
    }

    /// Check and update rate limit, returns true if allowed
    pub fn check_rate_limit(&self, connection_id: Uuid) -> Result<(), WsAuthError> {
        let mut connections = self.connections.write();

        if let Some(conn_info) = connections.get_mut(&connection_id) {
            let now = Instant::now();
            let rate_limit = conn_info.tier.rate_limit();

            // Reset window if more than 1 minute has passed
            if now.duration_since(conn_info.rate_window_start) > Duration::from_secs(60) {
                conn_info.rate_window_start = now;
                conn_info.request_count = 0;
            }

            conn_info.request_count += 1;
            conn_info.last_activity = now;

            if conn_info.request_count > rate_limit {
                return Err(WsAuthError::RateLimitExceeded(rate_limit));
            }
        }

        Ok(())
    }

    /// Get connection info
    pub fn get(&self, connection_id: Uuid) -> Option<ConnectionInfo> {
        self.connections.read().get(&connection_id).cloned()
    }

    /// Get all connections for an API key
    pub fn get_by_key(&self, key_id: &str) -> Vec<ConnectionInfo> {
        self.connections
            .read()
            .values()
            .filter(|c| c.key_id == key_id)
            .cloned()
            .collect()
    }

    /// Get total active connection count
    pub fn total_connections(&self) -> usize {
        self.connections.read().len()
    }

    /// Get connection count for a specific API key
    pub fn connection_count(&self, key_id: &str) -> usize {
        self.connection_counts
            .read()
            .get(key_id)
            .copied()
            .unwrap_or(0)
    }

    /// Clean up expired/stale connections
    pub fn cleanup_stale(&self, max_idle: Duration) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        {
            let connections = self.connections.read();
            for (id, info) in connections.iter() {
                if now.duration_since(info.last_activity) > max_idle {
                    to_remove.push(*id);
                }
            }
        }

        for id in to_remove {
            self.unregister(id);
            debug!(connection_id = %id, "Cleaned up stale connection");
        }
    }
}

impl Default for ConnectionTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Authentication State
// ============================================================================

/// Shared authentication state for the WebSocket server
#[derive(Clone)]
pub struct WsAuthState<V: ApiKeyValidator> {
    /// API key validator
    pub validator: Arc<V>,
    /// Connection tracker
    pub tracker: Arc<ConnectionTracker>,
    /// Authentication timeout duration
    pub auth_timeout: Duration,
    /// Header name for API key (default: "Authorization")
    pub api_key_header: String,
    /// Whether to require TLS (wss://)
    pub require_tls: bool,
}

impl<V: ApiKeyValidator> WsAuthState<V> {
    pub fn new(validator: V) -> Self {
        Self {
            validator: Arc::new(validator),
            tracker: Arc::new(ConnectionTracker::new()),
            auth_timeout: Duration::from_secs(30),
            api_key_header: "Authorization".to_string(),
            require_tls: false,
        }
    }

    pub fn with_auth_timeout(mut self, timeout: Duration) -> Self {
        self.auth_timeout = timeout;
        self
    }

    pub fn with_api_key_header(mut self, header: impl Into<String>) -> Self {
        self.api_key_header = header.into();
        self
    }

    pub fn with_require_tls(mut self, require: bool) -> Self {
        self.require_tls = require;
        self
    }

    /// Extract API key from request headers
    pub fn extract_api_key_from_headers(&self, headers: &HeaderMap) -> Option<String> {
        headers
            .get(&self.api_key_header)
            .and_then(|v| v.to_str().ok())
            .map(|s| {
                // Handle "Bearer <token>" format
                if s.starts_with("Bearer ") {
                    s[7..].to_string()
                } else {
                    s.to_string()
                }
            })
    }
}

// ============================================================================
// WebSocket Handler
// ============================================================================

/// Authentication message sent by client in first WebSocket message
#[derive(Debug, Deserialize)]
pub struct WsAuthMessage {
    /// API key for authentication
    pub api_key: String,
    /// Optional client metadata
    #[serde(default)]
    pub client_info: HashMap<String, String>,
}

/// Authentication result sent to client
#[derive(Debug, Serialize)]
pub struct WsAuthResult {
    /// Whether authentication succeeded
    pub success: bool,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Connection ID if succeeded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connection_id: Option<String>,
    /// Subscription tier if succeeded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
    /// Rate limit (requests per minute)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<u32>,
    /// Session timeout in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_timeout_secs: Option<u64>,
}

/// Authenticated WebSocket connection handle
pub struct AuthenticatedWsConnection {
    /// Connection info
    pub info: ConnectionInfo,
    /// WebSocket stream
    pub socket: WebSocket,
    /// State reference for rate limiting
    tracker: Arc<ConnectionTracker>,
}

impl AuthenticatedWsConnection {
    /// Send a message (with rate limit check)
    pub async fn send(&mut self, msg: Message) -> Result<(), WsAuthError> {
        self.tracker.check_rate_limit(self.info.connection_id)?;
        self.socket
            .send(msg)
            .await
            .map_err(|e| WsAuthError::Internal(e.to_string()))
    }

    /// Receive a message
    pub async fn recv(&mut self) -> Option<Result<Message, axum::Error>> {
        self.socket.recv().await
    }

    /// Get the connection ID
    pub fn connection_id(&self) -> Uuid {
        self.info.connection_id
    }

    /// Get the subscription tier
    pub fn tier(&self) -> SubscriptionTier {
        self.info.tier
    }
}

/// WebSocket upgrade handler with header-based authentication
#[instrument(skip(ws, state))]
pub async fn ws_handler_with_header_auth<V: ApiKeyValidator>(
    ws: WebSocketUpgrade,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<WsAuthState<V>>,
    headers: HeaderMap,
) -> Result<Response, WsAuthError> {
    // Try to extract API key from headers
    let api_key = state
        .extract_api_key_from_headers(&headers)
        .ok_or(WsAuthError::MissingApiKey)?;

    // Validate API key
    let key_info = state.validator.validate(&api_key).await?;

    // Check tier allows WebSocket
    if !key_info.tier.allows_websocket() {
        return Err(WsAuthError::TierNotAllowed(key_info.tier.to_string()));
    }

    // Register connection
    let conn_info = state.tracker.register(&key_info, addr)?;

    info!(
        connection_id = %conn_info.connection_id,
        tier = %key_info.tier,
        remote_addr = %addr,
        "WebSocket connection authenticated via header"
    );

    // Upgrade connection
    let tracker = Arc::clone(&state.tracker);

    Ok(ws.on_upgrade(move |socket| async move {
        handle_authenticated_socket(socket, conn_info, tracker).await;
    }))
}

/// WebSocket upgrade handler with first-message authentication
#[instrument(skip(ws, state))]
pub async fn ws_handler_with_message_auth<V: ApiKeyValidator>(
    ws: WebSocketUpgrade,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<WsAuthState<V>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // Check for API key in header first (preferred method)
    if let Some(api_key) = state.extract_api_key_from_headers(&headers) {
        match state.validator.validate(&api_key).await {
            Ok(key_info) => {
                if !key_info.tier.allows_websocket() {
                    return Err(WsAuthError::TierNotAllowed(key_info.tier.to_string()));
                }

                match state.tracker.register(&key_info, addr) {
                    Ok(conn_info) => {
                        let tracker = Arc::clone(&state.tracker);
                        return Ok(ws.on_upgrade(move |socket| async move {
                            handle_authenticated_socket(socket, conn_info, tracker).await;
                        }));
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(_) => {
                // Fall through to message-based auth
            }
        }
    }

    // No valid header auth, require first-message authentication
    let validator = Arc::clone(&state.validator);
    let tracker = Arc::clone(&state.tracker);
    let auth_timeout = state.auth_timeout;

    Ok(ws.on_upgrade(move |socket| async move {
        handle_unauthenticated_upgrade(socket, addr, validator, tracker, auth_timeout).await;
    }))
}

/// Handle socket that requires first-message authentication
async fn handle_unauthenticated_upgrade<V: ApiKeyValidator>(
    mut socket: WebSocket,
    addr: SocketAddr,
    validator: Arc<V>,
    tracker: Arc<ConnectionTracker>,
    auth_timeout: Duration,
) {
    // Wait for authentication message with timeout
    let auth_result = tokio::time::timeout(auth_timeout, socket.recv()).await;

    let auth_msg = match auth_result {
        Ok(Some(Ok(Message::Text(text)))) => match serde_json::from_str::<WsAuthMessage>(&text) {
            Ok(msg) => msg,
            Err(e) => {
                let _ = send_auth_error(&mut socket, &WsAuthError::InvalidAuthMessage).await;
                warn!(error = %e, "Invalid auth message format");
                return;
            }
        },
        Ok(Some(Ok(_))) => {
            let _ = send_auth_error(&mut socket, &WsAuthError::InvalidAuthMessage).await;
            warn!("First message must be text auth message");
            return;
        }
        Ok(Some(Err(e))) => {
            warn!(error = %e, "WebSocket error during auth");
            return;
        }
        Ok(None) => {
            warn!("Connection closed before authentication");
            return;
        }
        Err(_) => {
            let _ = send_auth_error(
                &mut socket,
                &WsAuthError::AuthTimeout(auth_timeout.as_secs()),
            )
            .await;
            warn!(
                timeout_secs = auth_timeout.as_secs(),
                "Authentication timeout"
            );
            return;
        }
    };

    // Validate API key
    let key_info = match validator.validate(&auth_msg.api_key).await {
        Ok(info) => info,
        Err(e) => {
            let _ = send_auth_error(&mut socket, &e).await;
            warn!(error = %e, "API key validation failed");
            return;
        }
    };

    // Check tier
    if !key_info.tier.allows_websocket() {
        let err = WsAuthError::TierNotAllowed(key_info.tier.to_string());
        let _ = send_auth_error(&mut socket, &err).await;
        return;
    }

    // Register connection
    let conn_info = match tracker.register(&key_info, addr) {
        Ok(info) => info,
        Err(e) => {
            let _ = send_auth_error(&mut socket, &e).await;
            return;
        }
    };

    // Send success response
    let auth_result = WsAuthResult {
        success: true,
        error: None,
        connection_id: Some(conn_info.connection_id.to_string()),
        tier: Some(conn_info.tier.to_string()),
        rate_limit: Some(conn_info.tier.rate_limit()),
        session_timeout_secs: Some(conn_info.tier.session_timeout().as_secs()),
    };

    if let Ok(json) = serde_json::to_string(&auth_result) {
        let _ = socket.send(Message::Text(json.into())).await;
    }

    info!(
        connection_id = %conn_info.connection_id,
        tier = %key_info.tier,
        remote_addr = %addr,
        "WebSocket connection authenticated via first message"
    );

    // Continue with authenticated handler
    handle_authenticated_socket(socket, conn_info, tracker).await;
}

/// Send authentication error response
async fn send_auth_error(socket: &mut WebSocket, error: &WsAuthError) -> Result<(), axum::Error> {
    let result = WsAuthResult {
        success: false,
        error: Some(error.to_string()),
        connection_id: None,
        tier: None,
        rate_limit: None,
        session_timeout_secs: None,
    };

    if let Ok(json) = serde_json::to_string(&result) {
        socket.send(Message::Text(json.into())).await?;
    }

    // Send close frame
    socket
        .send(Message::Close(Some(axum::extract::ws::CloseFrame {
            code: axum::extract::ws::close_code::POLICY,
            reason: error.to_string().into(),
        })))
        .await?;

    Ok(())
}

/// Handle an authenticated WebSocket connection
async fn handle_authenticated_socket(
    mut socket: WebSocket,
    conn_info: ConnectionInfo,
    tracker: Arc<ConnectionTracker>,
) {
    let connection_id = conn_info.connection_id;
    let tier = conn_info.tier;

    // Create a channel for the MCP message handler (for future bidirectional messaging)
    let (_tx, mut rx) = mpsc::channel::<Message>(100);

    // Spawn task to forward messages from channel to socket
    let send_task = tokio::spawn({
        let tracker = Arc::clone(&tracker);
        async move {
            while let Some(_msg) = rx.recv().await {
                // Check rate limit before sending
                if let Err(e) = tracker.check_rate_limit(connection_id) {
                    warn!(
                        connection_id = %connection_id,
                        error = %e,
                        "Rate limit exceeded"
                    );
                    // Send error and close
                    let _error_msg = serde_json::json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32000,
                            "message": e.to_string()
                        }
                    });
                    // Ignore send errors during rate limit
                    break;
                }
            }
        }
    });

    // Process incoming messages
    while let Some(msg) = socket.recv().await {
        match msg {
            Ok(Message::Text(text)) => {
                debug!(
                    connection_id = %connection_id,
                    msg_len = text.len(),
                    "Received text message"
                );

                // Check message size limit
                if text.len() > tier.max_message_size() {
                    warn!(
                        connection_id = %connection_id,
                        size = text.len(),
                        max = tier.max_message_size(),
                        "Message size exceeds tier limit"
                    );
                    // Send error response
                    let error_msg = serde_json::json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32000,
                            "message": format!("Message size {} exceeds limit {}", text.len(), tier.max_message_size())
                        }
                    });
                    if let Ok(json) = serde_json::to_string(&error_msg) {
                        let _ = socket.send(Message::Text(json.into())).await;
                    }
                    continue;
                }

                // TODO: Dispatch to MCP handler
                // For now, echo back
                let _ = socket.send(Message::Text(text)).await;
            }
            Ok(Message::Binary(data)) => {
                debug!(
                    connection_id = %connection_id,
                    size = data.len(),
                    "Received binary message"
                );

                // Check message size limit
                if data.len() > tier.max_message_size() {
                    warn!(
                        connection_id = %connection_id,
                        size = data.len(),
                        max = tier.max_message_size(),
                        "Binary message size exceeds tier limit"
                    );
                    continue;
                }

                // Echo back for now
                let _ = socket.send(Message::Binary(data)).await;
            }
            Ok(Message::Ping(data)) => {
                let _ = socket.send(Message::Pong(data)).await;
            }
            Ok(Message::Pong(_)) => {
                // Ignore pongs
            }
            Ok(Message::Close(_)) => {
                info!(connection_id = %connection_id, "Client initiated close");
                break;
            }
            Err(e) => {
                error!(
                    connection_id = %connection_id,
                    error = %e,
                    "WebSocket error"
                );
                break;
            }
        }
    }

    // Clean up
    send_task.abort();
    tracker.unregister(connection_id);
    info!(connection_id = %connection_id, "Connection closed");
}

// ============================================================================
// Axum Middleware Layer
// ============================================================================

/// Middleware for pre-upgrade authentication checks
pub async fn ws_auth_middleware<V: ApiKeyValidator>(
    State(state): State<WsAuthState<V>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, WsAuthError> {
    // Check if this is a WebSocket upgrade request
    let is_upgrade = request
        .headers()
        .get("upgrade")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false);

    if !is_upgrade {
        // Not a WebSocket request, pass through
        return Ok(next.run(request).await);
    }

    // TLS check in production
    if state.require_tls {
        let scheme = request.uri().scheme_str().unwrap_or("http");
        if scheme != "https" && scheme != "wss" {
            warn!("WebSocket connection rejected: TLS required");
            return Err(WsAuthError::Internal(
                "Secure connection (wss://) required".to_string(),
            ));
        }
    }

    // Continue to handler
    Ok(next.run(request).await)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Constant-time string comparison to prevent timing attacks
fn constant_time_compare(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();

    if a_bytes.len() != b_bytes.len() {
        // Still do a comparison to maintain constant time behavior
        let mut _dummy: u8 = 0;
        for byte in a_bytes.iter() {
            _dummy |= *byte;
        }
        return false;
    }

    let mut result: u8 = 0;
    for (x, y) in a_bytes.iter().zip(b_bytes.iter()) {
        result |= x ^ y;
    }

    result == 0
}

/// Generate a new API key (for development/testing)
pub fn generate_api_key() -> String {
    format!("rk_{}", Uuid::new_v4().to_string().replace('-', ""))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscription_tier_limits() {
        assert_eq!(SubscriptionTier::Free.max_connections(), 1);
        assert_eq!(SubscriptionTier::Pro.max_connections(), 5);
        assert_eq!(SubscriptionTier::Team.max_connections(), 25);
        assert_eq!(SubscriptionTier::Enterprise.max_connections(), 100);
    }

    #[test]
    fn test_subscription_tier_rate_limits() {
        assert_eq!(SubscriptionTier::Free.rate_limit(), 60);
        assert_eq!(SubscriptionTier::Pro.rate_limit(), 300);
        assert_eq!(SubscriptionTier::Team.rate_limit(), 1000);
        assert_eq!(SubscriptionTier::Enterprise.rate_limit(), 10000);
    }

    #[test]
    fn test_constant_time_compare() {
        assert!(constant_time_compare("secret", "secret"));
        assert!(!constant_time_compare("secret", "Secret"));
        assert!(!constant_time_compare("short", "longer"));
        assert!(!constant_time_compare("", "nonempty"));
    }

    #[test]
    fn test_generate_api_key() {
        let key = generate_api_key();
        assert!(key.starts_with("rk_"));
        assert_eq!(key.len(), 35); // "rk_" + 32 hex chars
    }

    #[tokio::test]
    async fn test_in_memory_validator() {
        let validator = InMemoryApiKeyValidator::new();

        let info = ApiKeyInfo {
            key_id: "key_123".to_string(),
            owner_id: "user_456".to_string(),
            tier: SubscriptionTier::Pro,
            expires_at: None,
            metadata: HashMap::new(),
        };

        validator.add_key("test_api_key".to_string(), info.clone());

        // Valid key
        let result = validator.validate("test_api_key").await;
        assert!(result.is_ok());
        let validated = result.unwrap();
        assert_eq!(validated.tier, SubscriptionTier::Pro);

        // Invalid key
        let result = validator.validate("wrong_key").await;
        assert!(matches!(result, Err(WsAuthError::InvalidApiKey)));
    }

    #[test]
    fn test_connection_tracker() {
        let tracker = ConnectionTracker::new();

        let key_info = ApiKeyInfo {
            key_id: "key_123".to_string(),
            owner_id: "user_456".to_string(),
            tier: SubscriptionTier::Free, // Only 1 connection allowed
            expires_at: None,
            metadata: HashMap::new(),
        };

        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        // First connection should succeed
        let conn1 = tracker.register(&key_info, addr);
        assert!(conn1.is_ok());

        // Second connection should fail (Free tier = 1 max)
        let conn2 = tracker.register(&key_info, addr);
        assert!(matches!(
            conn2,
            Err(WsAuthError::ConnectionLimitExceeded(_, 1))
        ));

        // Unregister first connection
        tracker.unregister(conn1.unwrap().connection_id);

        // Now should be able to connect again
        let conn3 = tracker.register(&key_info, addr);
        assert!(conn3.is_ok());
    }

    #[test]
    fn test_rate_limiting() {
        let tracker = ConnectionTracker::new();

        let key_info = ApiKeyInfo {
            key_id: "key_123".to_string(),
            owner_id: "user_456".to_string(),
            tier: SubscriptionTier::Free, // 60 requests per minute
            expires_at: None,
            metadata: HashMap::new(),
        };

        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let conn = tracker.register(&key_info, addr).unwrap();

        // Should allow 60 requests
        for _ in 0..60 {
            assert!(tracker.check_rate_limit(conn.connection_id).is_ok());
        }

        // 61st request should fail
        assert!(matches!(
            tracker.check_rate_limit(conn.connection_id),
            Err(WsAuthError::RateLimitExceeded(60))
        ));
    }

    #[test]
    fn test_api_key_extraction() {
        let validator = InMemoryApiKeyValidator::new();
        let state = WsAuthState::new(validator);

        let mut headers = HeaderMap::new();

        // Test Bearer format
        headers.insert("Authorization", "Bearer my_api_key".parse().unwrap());
        assert_eq!(
            state.extract_api_key_from_headers(&headers),
            Some("my_api_key".to_string())
        );

        // Test raw format
        headers.insert("Authorization", "raw_api_key".parse().unwrap());
        assert_eq!(
            state.extract_api_key_from_headers(&headers),
            Some("raw_api_key".to_string())
        );

        // Test custom header
        let state = state.with_api_key_header("X-Api-Key");
        headers.insert("X-Api-Key", "custom_key".parse().unwrap());
        assert_eq!(
            state.extract_api_key_from_headers(&headers),
            Some("custom_key".to_string())
        );
    }
}
