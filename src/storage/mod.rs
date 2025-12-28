//! Storage module for ReasonKit Core
//!
//! Provides document and chunk storage using Qdrant vector database.

use crate::{embedding::cosine_similarity, Document, Error, Result};
use async_trait::async_trait;
use qdrant_client::qdrant::{
    CreateCollection, DeletePoints, Distance, GetPoints, PointId, PointStruct, QuantizationConfig,
    ScalarQuantization, ScrollPoints, SearchPoints, UpsertPoints, VectorParams, VectorsConfig,
};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Security configuration for Qdrant connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantSecurityConfig {
    /// API key for authentication
    pub api_key: Option<String>,
    /// Enable TLS
    pub tls_enabled: bool,
    /// CA certificate path
    pub ca_cert_path: Option<String>,
    /// Client certificate path
    pub client_cert_path: Option<String>,
    /// Client key path
    pub client_key_path: Option<String>,
    /// Skip TLS verification (not recommended for production)
    pub skip_tls_verify: bool,
}

impl Default for QdrantSecurityConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            tls_enabled: true,
            ca_cert_path: None,
            client_cert_path: None,
            client_key_path: None,
            skip_tls_verify: false,
        }
    }
}

/// Connection pool configuration for Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConnectionConfig {
    /// Maximum number of connections in the pool
    pub max_connections: usize,
    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Maximum idle time for connections in seconds
    pub max_idle_secs: u64,
    /// Security configuration
    pub security: QdrantSecurityConfig,
}

impl Default for QdrantConnectionConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            connect_timeout_secs: 30,
            request_timeout_secs: 60,
            health_check_interval_secs: 300, // 5 minutes
            max_idle_secs: 600,              // 10 minutes
            security: QdrantSecurityConfig::default(),
        }
    }
}

/// Convert qdrant Value to serde_json Value
fn qdrant_value_to_json(value: &qdrant_client::qdrant::Value) -> serde_json::Value {
    use qdrant_client::qdrant::value::Kind;

    match &value.kind {
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(v)) => serde_json::Value::Bool(*v),
        Some(Kind::IntegerValue(v)) => serde_json::Value::Number((*v).into()),
        Some(Kind::DoubleValue(v)) => {
            serde_json::Value::Number(serde_json::Number::from_f64(*v).unwrap_or(0.into()))
        }
        Some(Kind::StringValue(v)) => serde_json::Value::String(v.clone()),
        Some(Kind::ListValue(v)) => {
            let items = v.values.iter().map(qdrant_value_to_json).collect();
            serde_json::Value::Array(items)
        }
        Some(Kind::StructValue(v)) => {
            let fields = v
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                .collect();
            serde_json::Value::Object(fields)
        }
        None => serde_json::Value::Null,
    }
}

/// Access level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Read-only access
    Read,
    /// Read and write access
    ReadWrite,
    /// Full administrative access
    Admin,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Required level for read operations
    pub read_level: AccessLevel,
    /// Required level for write operations
    pub write_level: AccessLevel,
    /// Required level for delete operations
    pub delete_level: AccessLevel,
    /// Required level for admin operations
    pub admin_level: AccessLevel,
    /// Enable audit logging
    pub enable_audit_log: bool,
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            read_level: AccessLevel::Read,
            write_level: AccessLevel::ReadWrite,
            delete_level: AccessLevel::ReadWrite,
            admin_level: AccessLevel::Admin,
            enable_audit_log: true,
        }
    }
}

/// Access context for operations
#[derive(Debug, Clone)]
pub struct AccessContext {
    /// User identifier
    pub user_id: String,
    /// User's access level
    pub access_level: AccessLevel,
    /// Operation being performed
    pub operation: String,
    /// Timestamp of the operation
    pub timestamp: i64,
}

impl AccessContext {
    /// Create a new access context
    pub fn new(user_id: String, access_level: AccessLevel, operation: String) -> Self {
        Self {
            user_id,
            access_level,
            operation,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Check if this context has permission for the required level
    pub fn has_permission(
        &self,
        required_level: &AccessLevel,
        _config: &AccessControlConfig,
    ) -> bool {
        match required_level {
            AccessLevel::Read => matches!(
                self.access_level,
                AccessLevel::Read | AccessLevel::ReadWrite | AccessLevel::Admin
            ),
            AccessLevel::ReadWrite => matches!(
                self.access_level,
                AccessLevel::ReadWrite | AccessLevel::Admin
            ),
            AccessLevel::Admin => matches!(self.access_level, AccessLevel::Admin),
        }
    }
}

/// Configuration for embedding cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCacheConfig {
    /// Maximum number of embeddings to cache
    pub max_size: usize,
    /// TTL for cached embeddings in seconds
    pub ttl_secs: u64,
}

impl Default for EmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            max_size: 10000,
            ttl_secs: 3600, // 1 hour
        }
    }
}

/// Cached embedding entry
#[derive(Debug, Clone)]
struct CachedEmbedding {
    /// The embedding vector
    embedding: Vec<f32>,
    /// When this entry was created
    created_at: Instant,
}

/// LRU cache for embeddings
#[derive(Debug)]
pub struct EmbeddingCache {
    /// Cache storage
    cache: HashMap<Uuid, CachedEmbedding>,
    /// Access order for LRU eviction
    access_order: Vec<Uuid>,
    /// Configuration
    config: EmbeddingCacheConfig,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(config: EmbeddingCacheConfig) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            config,
        }
    }

    /// Put an embedding in the cache
    pub fn put(&mut self, chunk_id: Uuid, embedding: Vec<f32>) {
        let entry = CachedEmbedding {
            embedding,
            created_at: Instant::now(),
        };

        // Remove existing entry if present
        if self.cache.contains_key(&chunk_id) {
            self.access_order.retain(|&id| id != chunk_id);
        }

        // Add new entry
        self.cache.insert(chunk_id, entry);
        self.access_order.push(chunk_id);

        // Evict if over capacity
        while self.cache.len() > self.config.max_size {
            let oldest_id = self.access_order.remove(0);
            self.cache.remove(&oldest_id);
        }
    }

    /// Get an embedding from the cache
    pub fn get(&mut self, chunk_id: &Uuid) -> Option<Vec<f32>> {
        // Check if entry exists and is not expired
        if let Some(entry) = self.cache.get(chunk_id) {
            if entry.created_at.elapsed().as_secs() <= self.config.ttl_secs {
                // Update access order for LRU
                self.access_order.retain(|&id| id != *chunk_id);
                self.access_order.push(*chunk_id);
                return Some(entry.embedding.clone());
            } else {
                // Remove expired entry
                self.cache.remove(chunk_id);
                self.access_order.retain(|&id| id != *chunk_id);
            }
        }
        None
    }

    /// Clean up expired entries
    pub fn cleanup_expired(&mut self) {
        let mut to_remove = Vec::new();

        for (id, entry) in &self.cache {
            if entry.created_at.elapsed().as_secs() > self.config.ttl_secs {
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            self.cache.remove(&id);
            self.access_order.retain(|&order_id| order_id != id);
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Number of documents stored
    pub document_count: usize,
    /// Number of chunks stored
    pub chunk_count: usize,
    /// Number of embeddings stored
    pub embedding_count: usize,
    /// Total size in bytes
    pub size_bytes: u64,
}

/// Storage backend trait
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store a document
    async fn store_document(&self, doc: &Document, context: &AccessContext) -> Result<()>;

    /// Get a document by ID
    async fn get_document(&self, id: &Uuid, context: &AccessContext) -> Result<Option<Document>>;

    /// Delete a document
    async fn delete_document(&self, id: &Uuid, context: &AccessContext) -> Result<()>;

    /// List all documents
    async fn list_documents(&self, context: &AccessContext) -> Result<Vec<Uuid>>;

    /// Store embeddings for a chunk
    async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        context: &AccessContext,
    ) -> Result<()>;

    /// Get embeddings for a chunk
    async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Vec<f32>>>;

    /// Search by vector similarity
    async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>>;

    /// Get storage statistics
    async fn stats(&self, context: &AccessContext) -> Result<StorageStats>;
}

/// In-memory storage backend for testing
pub struct InMemoryStorage {
    documents: Arc<RwLock<HashMap<Uuid, Document>>>,
    embeddings: Arc<RwLock<HashMap<Uuid, Vec<f32>>>>,
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            embeddings: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl InMemoryStorage {
    /// Create a new in-memory storage
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl StorageBackend for InMemoryStorage {
    async fn store_document(&self, doc: &Document, _context: &AccessContext) -> Result<()> {
        let mut docs = self.documents.write().await;
        docs.insert(doc.id, doc.clone());
        Ok(())
    }

    async fn get_document(&self, id: &Uuid, _context: &AccessContext) -> Result<Option<Document>> {
        let docs = self.documents.read().await;
        Ok(docs.get(id).cloned())
    }

    async fn delete_document(&self, id: &Uuid, _context: &AccessContext) -> Result<()> {
        let mut docs = self.documents.write().await;
        docs.remove(id);
        Ok(())
    }

    async fn list_documents(&self, _context: &AccessContext) -> Result<Vec<Uuid>> {
        let docs = self.documents.read().await;
        Ok(docs.keys().cloned().collect())
    }

    async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        _context: &AccessContext,
    ) -> Result<()> {
        let mut embs = self.embeddings.write().await;
        embs.insert(*chunk_id, embeddings.to_vec());
        Ok(())
    }

    async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        _context: &AccessContext,
    ) -> Result<Option<Vec<f32>>> {
        let embs = self.embeddings.read().await;
        Ok(embs.get(chunk_id).cloned())
    }

    async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        _context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>> {
        let embs = self.embeddings.read().await;
        let mut results: Vec<(Uuid, f32)> = embs
            .iter()
            .map(|(id, emb)| (*id, cosine_similarity(query_embedding, emb)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        Ok(results)
    }

    async fn stats(&self, _context: &AccessContext) -> Result<StorageStats> {
        let docs = self.documents.read().await;
        let embs = self.embeddings.read().await;

        // Count actual chunks from documents, not embeddings
        let chunk_count: usize = docs.values().map(|d| d.chunks.len()).sum();

        Ok(StorageStats {
            document_count: docs.len(),
            chunk_count,
            embedding_count: embs.len(),
            size_bytes: 0, // Not tracked in memory
        })
    }
}

/// Pooled Qdrant connection entry
struct PooledConnection {
    client: Qdrant,
    last_used: Instant,
    #[allow(dead_code)]
    created_at: Instant,
}

/// Simple connection pool for Qdrant clients
struct QdrantConnectionPool {
    connections: Vec<PooledConnection>,
    config: QdrantConnectionConfig,
    client_config: qdrant_client::config::QdrantConfig,
}

impl QdrantConnectionPool {
    fn new(
        client_config: qdrant_client::config::QdrantConfig,
        config: QdrantConnectionConfig,
    ) -> Self {
        Self {
            connections: Vec::new(),
            config,
            client_config,
        }
    }

    async fn get_connection(&mut self) -> Result<&mut Qdrant> {
        // First, try to find an available connection
        let available_index = self.connections.iter().position(|conn| {
            conn.last_used.elapsed() < Duration::from_secs(self.config.max_idle_secs)
        });

        if let Some(index) = available_index {
            self.connections[index].last_used = Instant::now();
            return Ok(&mut self.connections[index].client);
        }

        // Create new connection if under limit
        if self.connections.len() < self.config.max_connections {
            let client = Qdrant::new(self.client_config.clone())
                .map_err(|e| Error::io(format!("Failed to create Qdrant client: {}", e)))?;

            self.connections.push(PooledConnection {
                client,
                last_used: Instant::now(),
                created_at: Instant::now(),
            });

            let len = self.connections.len();
            return Ok(&mut self.connections[len - 1].client);
        }

        Err(Error::io("Connection pool exhausted".to_string()))
    }

    #[allow(dead_code)]
    fn cleanup_expired(&mut self) {
        self.connections.retain(|conn| {
            conn.created_at.elapsed() < Duration::from_secs(self.config.max_idle_secs)
        });
    }

    async fn health_check(&mut self) -> Result<()> {
        if let Ok(client) = self.get_connection().await {
            // Simple health check - try to list collections
            client
                .list_collections()
                .await
                .map_err(|e| Error::io(format!("Health check failed: {}", e)))?;
        }
        Ok(())
    }
}

/// File-based storage backend (JSON files)
pub struct FileStorage {
    base_path: PathBuf,
    documents: Arc<RwLock<HashMap<Uuid, Document>>>,
}

impl FileStorage {
    /// Create a new file-based storage
    pub async fn new(base_path: PathBuf) -> Result<Self> {
        // Create directories if they don't exist
        tokio::fs::create_dir_all(&base_path)
            .await
            .map_err(|e| Error::io(format!("Failed to create storage directory: {}", e)))?;
        tokio::fs::create_dir_all(base_path.join("documents"))
            .await
            .map_err(|e| Error::io(format!("Failed to create documents directory: {}", e)))?;
        tokio::fs::create_dir_all(base_path.join("embeddings"))
            .await
            .map_err(|e| Error::io(format!("Failed to create embeddings directory: {}", e)))?;

        // Load existing documents
        let documents = Arc::new(RwLock::new(HashMap::new()));

        let storage = Self {
            base_path,
            documents,
        };
        storage.load_documents().await?;

        Ok(storage)
    }

    async fn load_documents(&self) -> Result<()> {
        let docs_path = self.base_path.join("documents");

        let mut entries = tokio::fs::read_dir(&docs_path)
            .await
            .map_err(|e| Error::io(format!("Failed to read documents directory: {}", e)))?;

        let mut docs = self.documents.write().await;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| Error::io(format!("Failed to read directory entry: {}", e)))?
        {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let content = tokio::fs::read_to_string(&path)
                    .await
                    .map_err(|e| Error::io(format!("Failed to read document file: {}", e)))?;
                let doc: Document = serde_json::from_str(&content)
                    .map_err(|e| Error::parse(format!("Failed to parse document: {}", e)))?;
                docs.insert(doc.id, doc);
            }
        }

        Ok(())
    }

    fn doc_path(&self, id: &Uuid) -> PathBuf {
        self.base_path
            .join("documents")
            .join(format!("{}.json", id))
    }

    fn embedding_path(&self, id: &Uuid) -> PathBuf {
        self.base_path
            .join("embeddings")
            .join(format!("{}.bin", id))
    }
}

#[async_trait]
impl StorageBackend for FileStorage {
    async fn store_document(&self, doc: &Document, _context: &AccessContext) -> Result<()> {
        let path = self.doc_path(&doc.id);
        let content = serde_json::to_string_pretty(doc)
            .map_err(|e| Error::parse(format!("Failed to serialize document: {}", e)))?;
        tokio::fs::write(&path, content)
            .await
            .map_err(|e| Error::io(format!("Failed to write document: {}", e)))?;

        let mut docs = self.documents.write().await;
        docs.insert(doc.id, doc.clone());

        Ok(())
    }

    async fn get_document(&self, id: &Uuid, _context: &AccessContext) -> Result<Option<Document>> {
        let docs = self.documents.read().await;
        Ok(docs.get(id).cloned())
    }

    async fn delete_document(&self, id: &Uuid, _context: &AccessContext) -> Result<()> {
        let path = self.doc_path(id);
        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| Error::io(format!("Failed to delete document: {}", e)))?;
        }

        let mut docs = self.documents.write().await;
        docs.remove(id);

        Ok(())
    }

    async fn list_documents(&self, _context: &AccessContext) -> Result<Vec<Uuid>> {
        let docs = self.documents.read().await;
        Ok(docs.keys().cloned().collect())
    }

    async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        _context: &AccessContext,
    ) -> Result<()> {
        let path = self.embedding_path(chunk_id);

        // Store as binary for efficiency
        let bytes: Vec<u8> = embeddings.iter().flat_map(|f| f.to_le_bytes()).collect();

        tokio::fs::write(&path, bytes)
            .await
            .map_err(|e| Error::io(format!("Failed to write embeddings: {}", e)))?;

        Ok(())
    }

    async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        _context: &AccessContext,
    ) -> Result<Option<Vec<f32>>> {
        let path = self.embedding_path(chunk_id);

        if !path.exists() {
            return Ok(None);
        }

        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| Error::io(format!("Failed to read embeddings: {}", e)))?;

        let embeddings: Vec<f32> = bytes
            .chunks(4)
            .map(|chunk: &[u8]| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect();

        Ok(Some(embeddings))
    }

    async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        _context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>> {
        let embeddings_dir = self.base_path.join("embeddings");
        let mut results: Vec<(Uuid, f32)> = Vec::new();

        let mut entries = tokio::fs::read_dir(&embeddings_dir)
            .await
            .map_err(|e| Error::io(format!("Failed to read embeddings directory: {}", e)))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| Error::io(format!("Failed to read entry: {}", e)))?
        {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "bin") {
                // Extract UUID from filename
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(id) = Uuid::parse_str(stem) {
                        // Load embeddings
                        let bytes = tokio::fs::read(&path)
                            .await
                            .map_err(|e| Error::io(format!("Failed to read embeddings: {}", e)))?;

                        let embeddings: Vec<f32> = bytes
                            .chunks(4)
                            .map(|chunk: &[u8]| {
                                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                                f32::from_le_bytes(arr)
                            })
                            .collect();

                        let score = cosine_similarity(query_embedding, &embeddings);
                        results.push((id, score));
                    }
                }
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    async fn stats(&self, _context: &AccessContext) -> Result<StorageStats> {
        let docs = self.documents.read().await;
        let chunk_count: usize = docs.values().map(|d| d.chunks.len()).sum();

        // Count embedding files
        let embeddings_dir = self.base_path.join("embeddings");
        let mut embedding_count = 0;

        if let Ok(mut entries) = tokio::fs::read_dir(&embeddings_dir).await {
            while let Ok(Some(_)) = entries.next_entry().await {
                embedding_count += 1;
            }
        }

        // Calculate approximate size
        let mut size_bytes: u64 = 0;
        let docs_dir = self.base_path.join("documents");
        if let Ok(mut entries) = tokio::fs::read_dir(&docs_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Ok(metadata) = entry.metadata().await {
                    size_bytes += metadata.len();
                }
            }
        }

        Ok(StorageStats {
            document_count: docs.len(),
            chunk_count,
            embedding_count,
            size_bytes,
        })
    }
}

/// Qdrant vector database storage backend
pub struct QdrantStorage {
    pool: Arc<RwLock<QdrantConnectionPool>>,
    collection_name: String,
    vector_size: usize,
    embedding_cache: Arc<RwLock<EmbeddingCache>>,
    access_control: AccessControlConfig,
}

impl QdrantStorage {
    /// Create a new Qdrant storage backend
    pub async fn new(
        host: &str,
        port: u16,
        grpc_port: u16,
        collection_name: String,
        vector_size: usize,
        embedded: bool,
    ) -> Result<Self> {
        Self::new_with_config(
            host,
            port,
            grpc_port,
            collection_name,
            vector_size,
            embedded,
            QdrantConnectionConfig::default(),
            EmbeddingCacheConfig::default(),
            AccessControlConfig::default(),
        )
        .await
    }

    /// Create a new Qdrant storage backend with custom configuration
    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_config(
        host: &str,
        port: u16,
        _grpc_port: u16,
        collection_name: String,
        vector_size: usize,
        embedded: bool,
        conn_config: QdrantConnectionConfig,
        cache_config: EmbeddingCacheConfig,
        access_config: AccessControlConfig,
    ) -> Result<Self> {
        let config = if embedded {
            qdrant_client::config::QdrantConfig::from_url("http://localhost:6333")
        } else {
            qdrant_client::config::QdrantConfig::from_url(&format!("http://{}:{}", host, port))
        };

        let pool = Arc::new(RwLock::new(QdrantConnectionPool::new(
            config,
            conn_config.clone(),
        )));
        let embedding_cache = Arc::new(RwLock::new(EmbeddingCache::new(cache_config)));

        let storage = Self {
            pool: pool.clone(),
            collection_name: collection_name.clone(),
            vector_size,
            embedding_cache,
            access_control: access_config,
        };

        // Ensure collection exists using a connection from the pool
        {
            let mut pool_guard = pool.write().await;
            let client = pool_guard.get_connection().await?;
            Self::ensure_collection(client, &collection_name, vector_size).await?;
        }

        // Start background health check task
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(conn_config.health_check_interval_secs));
            loop {
                interval.tick().await;
                let mut pool = pool_clone.write().await;
                if let Err(e) = pool.health_check().await {
                    tracing::warn!("Qdrant health check failed: {}", e);
                }
            }
        });

        Ok(storage)
    }

    async fn ensure_collection(
        client: &Qdrant,
        collection_name: &str,
        vector_size: usize,
    ) -> Result<()> {
        // Check if collection exists
        let collections = client
            .list_collections()
            .await
            .map_err(|e| Error::io(format!("Failed to list collections: {}", e)))?;

        let collection_exists = collections
            .collections
            .iter()
            .any(|c| c.name == collection_name);

        if !collection_exists {
            // Create collection with vector configuration
            let vector_params = VectorParams {
                size: vector_size as u64,
                distance: Distance::Cosine as i32,
                hnsw_config: None,
                quantization_config: Some(QuantizationConfig {
                    quantization: Some(
                        qdrant_client::qdrant::quantization_config::Quantization::Scalar(
                            ScalarQuantization {
                                r#type: qdrant_client::qdrant::QuantizationType::Int8 as i32,
                                quantile: None,
                                always_ram: None,
                            },
                        ),
                    ),
                }),
                on_disk: None,
                datatype: None,
                multivector_config: None,
            };

            let collection_params = CreateCollection {
                collection_name: collection_name.to_string(),
                vectors_config: Some(VectorsConfig {
                    config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                        vector_params,
                    )),
                }),
                ..Default::default()
            };

            client
                .create_collection(collection_params)
                .await
                .map_err(|e| Error::io(format!("Failed to create collection: {}", e)))?;
        }

        Ok(())
    }

    fn point_id_from_uuid(uuid: &Uuid) -> PointId {
        PointId::from(uuid.to_string())
    }

    fn uuid_from_point_id(point_id: &PointId) -> Option<Uuid> {
        // Extract UUID from PointId using the proper API
        match &point_id.point_id_options {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid_str)) => {
                Uuid::parse_str(uuid_str).ok()
            }
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => {
                tracing::warn!(
                    "Cannot convert numeric PointId {} to UUID - using UUID strings is required",
                    num
                );
                None
            }
            None => None,
        }
    }

    fn check_access(&self, context: &AccessContext, required_level: &AccessLevel) -> Result<()> {
        if !context.has_permission(required_level, &self.access_control) {
            return Err(Error::validation(format!(
                "Access denied: user {} requires {:?} level for operation '{}', has {:?}",
                context.user_id, required_level, context.operation, context.access_level
            )));
        }

        if self.access_control.enable_audit_log {
            tracing::info!(
                "Access granted: user={}, operation={}, level={:?}, timestamp={}",
                context.user_id,
                context.operation,
                context.access_level,
                context.timestamp
            );
        }

        Ok(())
    }
}

#[async_trait]
impl StorageBackend for QdrantStorage {
    async fn store_document(&self, doc: &Document, context: &AccessContext) -> Result<()> {
        self.check_access(context, &self.access_control.write_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        // Store document metadata as payload
        let payload: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&serde_json::to_string(doc)?)
                .map_err(|e| Error::parse(format!("Failed to serialize document: {}", e)))?;

        let point = PointStruct::new(
            Self::point_id_from_uuid(&doc.id),
            vec![], // No vectors for document metadata
            payload,
        );

        let points = vec![point];
        let upsert_points = UpsertPoints {
            collection_name: self.collection_name.clone(),
            wait: Some(true),
            points,
            ..Default::default()
        };

        client
            .upsert_points(upsert_points)
            .await
            .map_err(|e| Error::io(format!("Failed to store document: {}", e)))?;

        Ok(())
    }

    async fn get_document(&self, id: &Uuid, context: &AccessContext) -> Result<Option<Document>> {
        self.check_access(context, &self.access_control.read_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let point_id = Self::point_id_from_uuid(id);

        let get_points = GetPoints {
            collection_name: self.collection_name.clone(),
            ids: vec![point_id],
            with_payload: Some(qdrant_client::qdrant::WithPayloadSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(true),
                ),
            }),
            with_vectors: Some(qdrant_client::qdrant::WithVectorsSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_vectors_selector::SelectorOptions::Enable(false),
                ),
            }),
            ..Default::default()
        };

        let response = client
            .get_points(get_points)
            .await
            .map_err(|e| Error::io(format!("Failed to get document: {}", e)))?;

        if let Some(point) = response.result.first() {
            // Convert qdrant payload to serde_json Value
            let json_payload: std::collections::HashMap<String, serde_json::Value> = point
                .payload
                .iter()
                .map(|(k, v)| (k.clone(), qdrant_value_to_json(v)))
                .collect();

            let doc: Document = serde_json::from_value(serde_json::Value::Object(
                json_payload.into_iter().collect(),
            ))
            .map_err(|e| Error::parse(format!("Failed to deserialize document: {}", e)))?;
            Ok(Some(doc))
        } else {
            Ok(None)
        }
    }

    async fn delete_document(&self, id: &Uuid, context: &AccessContext) -> Result<()> {
        self.check_access(context, &self.access_control.delete_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let point_id = Self::point_id_from_uuid(id);

        let delete_points = DeletePoints {
            collection_name: self.collection_name.clone(),
            wait: Some(true),
            points: Some(qdrant_client::qdrant::PointsSelector {
                points_selector_one_of: Some(
                    qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Points(
                        qdrant_client::qdrant::PointsIdsList {
                            ids: vec![point_id],
                        },
                    ),
                ),
            }),
            ..Default::default()
        };

        client
            .delete_points(delete_points)
            .await
            .map_err(|e| Error::io(format!("Failed to delete document: {}", e)))?;

        Ok(())
    }

    async fn list_documents(&self, context: &AccessContext) -> Result<Vec<Uuid>> {
        self.check_access(context, &self.access_control.read_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        // Scroll through all points to get document IDs
        let mut all_ids = Vec::new();
        let mut offset = None;

        loop {
            let scroll_points = ScrollPoints {
                collection_name: self.collection_name.clone(),
                limit: Some(100),
                offset,
                with_payload: Some(qdrant_client::qdrant::WithPayloadSelector {
                    selector_options: Some(
                        qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(
                            false,
                        ),
                    ),
                }),
                with_vectors: Some(qdrant_client::qdrant::WithVectorsSelector {
                    selector_options: Some(
                        qdrant_client::qdrant::with_vectors_selector::SelectorOptions::Enable(
                            false,
                        ),
                    ),
                }),
                ..Default::default()
            };

            let response = client
                .scroll(scroll_points)
                .await
                .map_err(|e| Error::io(format!("Failed to scroll points: {}", e)))?;

            for point in &response.result {
                if let Some(id) = &point.id {
                    if let Some(uuid) = Self::uuid_from_point_id(id) {
                        all_ids.push(uuid);
                    }
                }
            }

            if response.next_page_offset.is_none() {
                break;
            }
            offset = response.next_page_offset;
        }

        Ok(all_ids)
    }

    async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        context: &AccessContext,
    ) -> Result<()> {
        self.check_access(context, &self.access_control.write_level)?;

        if embeddings.len() != self.vector_size {
            return Err(Error::validation(format!(
                "Embedding size {} does not match configured vector size {}",
                embeddings.len(),
                self.vector_size
            )));
        }

        // Cache the embedding
        {
            let mut cache = self.embedding_cache.write().await;
            cache.put(*chunk_id, embeddings.to_vec());
        }

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let point_id = Self::point_id_from_uuid(chunk_id);

        // Create payload with chunk metadata
        let mut payload: std::collections::HashMap<String, serde_json::Value> =
            std::collections::HashMap::new();
        payload.insert(
            "chunk_id".to_string(),
            serde_json::Value::String(chunk_id.to_string()),
        );

        let point = PointStruct::new(point_id, embeddings.to_vec(), payload);

        let points = vec![point];
        let upsert_points = UpsertPoints {
            collection_name: self.collection_name.clone(),
            wait: Some(true),
            points,
            ..Default::default()
        };

        client
            .upsert_points(upsert_points)
            .await
            .map_err(|e| Error::io(format!("Failed to store embeddings: {}", e)))?;

        Ok(())
    }

    async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Vec<f32>>> {
        self.check_access(context, &self.access_control.read_level)?;

        // Check cache first
        {
            let mut cache = self.embedding_cache.write().await;
            cache.cleanup_expired(); // Clean up expired entries
            if let Some(embedding) = cache.get(chunk_id) {
                return Ok(Some(embedding));
            }
        }

        // Not in cache, retrieve from Qdrant
        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let point_id = Self::point_id_from_uuid(chunk_id);

        let get_points = GetPoints {
            collection_name: self.collection_name.clone(),
            ids: vec![point_id],
            with_payload: Some(qdrant_client::qdrant::WithPayloadSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(false),
                ),
            }),
            with_vectors: Some(qdrant_client::qdrant::WithVectorsSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_vectors_selector::SelectorOptions::Enable(true),
                ),
            }),
            ..Default::default()
        };

        let response = client
            .get_points(get_points)
            .await
            .map_err(|e| Error::io(format!("Failed to get embeddings: {}", e)))?;

        if let Some(point) = response.result.first() {
            // Extract vector data from the retrieved point
            if let Some(vectors) = &point.vectors {
                use qdrant_client::qdrant::vector_output::Vector as OutputVector;
                use qdrant_client::qdrant::vectors_output::VectorsOptions;
                match &vectors.vectors_options {
                    Some(VectorsOptions::Vector(vector_output)) => {
                        // Use into_vector() to get the Vector enum, then extract dense data
                        match vector_output.clone().into_vector() {
                            OutputVector::Dense(dense) => {
                                let embedding = dense.data;
                                self.embedding_cache
                                    .write()
                                    .await
                                    .put(*chunk_id, embedding.clone());
                                Ok(Some(embedding))
                            }
                            _ => Ok(None), // Sparse/MultiDense not supported for caching
                        }
                    }
                    Some(VectorsOptions::Vectors(named_vectors)) => {
                        // For named vectors, try to get the default vector
                        if let Some(vector_output) = named_vectors.vectors.get("") {
                            match vector_output.clone().into_vector() {
                                OutputVector::Dense(dense) => {
                                    let embedding = dense.data;
                                    self.embedding_cache
                                        .write()
                                        .await
                                        .put(*chunk_id, embedding.clone());
                                    Ok(Some(embedding))
                                }
                                _ => Ok(None),
                            }
                        } else if let Some((_, vector_output)) = named_vectors.vectors.iter().next()
                        {
                            // Fallback to first available vector
                            match vector_output.clone().into_vector() {
                                OutputVector::Dense(dense) => {
                                    let embedding = dense.data;
                                    self.embedding_cache
                                        .write()
                                        .await
                                        .put(*chunk_id, embedding.clone());
                                    Ok(Some(embedding))
                                }
                                _ => Ok(None),
                            }
                        } else {
                            Ok(None)
                        }
                    }
                    None => Ok(None),
                }
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>> {
        self.check_access(context, &self.access_control.read_level)?;

        if query_embedding.len() != self.vector_size {
            return Err(Error::validation(format!(
                "Query embedding size {} does not match configured vector size {}",
                query_embedding.len(),
                self.vector_size
            )));
        }

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let search_points = SearchPoints {
            collection_name: self.collection_name.clone(),
            vector: query_embedding.to_vec(),
            limit: top_k as u64,
            with_payload: Some(qdrant_client::qdrant::WithPayloadSelector {
                selector_options: Some(
                    qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(true),
                ),
            }),
            ..Default::default()
        };

        let response = client
            .search_points(search_points)
            .await
            .map_err(|e| Error::io(format!("Failed to search vectors: {}", e)))?;

        let results = response
            .result
            .into_iter()
            .filter_map(|scored_point| {
                scored_point
                    .id
                    .as_ref()
                    .and_then(Self::uuid_from_point_id)
                    .map(|uuid| (uuid, scored_point.score))
            })
            .collect();

        Ok(results)
    }

    async fn stats(&self, context: &AccessContext) -> Result<StorageStats> {
        self.check_access(context, &self.access_control.admin_level)?;

        let mut pool = self.pool.write().await;
        let client = pool.get_connection().await?;

        let collection_info = client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| Error::io(format!("Failed to get collection info: {}", e)))?;

        let points_count = collection_info
            .result
            .as_ref()
            .map(|info| info.points_count.unwrap_or(0))
            .unwrap_or(0);

        // Estimate document count (points without vectors are documents)
        let document_count = points_count.saturating_sub(points_count);

        Ok(StorageStats {
            document_count: document_count as usize,
            chunk_count: points_count as usize,
            embedding_count: points_count as usize,
            size_bytes: 0, // Qdrant doesn't expose this directly
        })
    }
}
pub struct Storage {
    backend: Box<dyn StorageBackend>,
}

impl Storage {
    /// Create storage with in-memory backend
    pub fn in_memory() -> Self {
        Self {
            backend: Box::new(InMemoryStorage::new()),
        }
    }

    /// Create storage with file backend
    pub async fn file(base_path: PathBuf) -> Result<Self> {
        Ok(Self {
            backend: Box::new(FileStorage::new(base_path).await?),
        })
    }

    /// Create storage with Qdrant backend
    pub async fn qdrant(
        host: &str,
        port: u16,
        grpc_port: u16,
        collection_name: String,
        vector_size: usize,
        embedded: bool,
    ) -> Result<Self> {
        Ok(Self {
            backend: Box::new(
                QdrantStorage::new(
                    host,
                    port,
                    grpc_port,
                    collection_name,
                    vector_size,
                    embedded,
                )
                .await?,
            ),
        })
    }

    /// Create storage with Qdrant backend and custom configuration
    #[allow(clippy::too_many_arguments)]
    pub async fn qdrant_with_config(
        host: &str,
        port: u16,
        grpc_port: u16,
        collection_name: String,
        vector_size: usize,
        embedded: bool,
        conn_config: QdrantConnectionConfig,
        cache_config: EmbeddingCacheConfig,
        access_config: AccessControlConfig,
    ) -> Result<Self> {
        Ok(Self {
            backend: Box::new(
                QdrantStorage::new_with_config(
                    host,
                    port,
                    grpc_port,
                    collection_name,
                    vector_size,
                    embedded,
                    conn_config,
                    cache_config,
                    access_config,
                )
                .await?,
            ),
        })
    }

    /// Store a document
    pub async fn store_document(&self, doc: &Document, context: &AccessContext) -> Result<()> {
        self.backend.store_document(doc, context).await
    }

    /// Get a document by ID
    pub async fn get_document(
        &self,
        id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Document>> {
        self.backend.get_document(id, context).await
    }

    /// Delete a document
    pub async fn delete_document(&self, id: &Uuid, context: &AccessContext) -> Result<()> {
        self.backend.delete_document(id, context).await
    }

    /// List all documents
    pub async fn list_documents(&self, context: &AccessContext) -> Result<Vec<Uuid>> {
        self.backend.list_documents(context).await
    }

    /// Store embeddings
    pub async fn store_embeddings(
        &self,
        chunk_id: &Uuid,
        embeddings: &[f32],
        context: &AccessContext,
    ) -> Result<()> {
        self.backend
            .store_embeddings(chunk_id, embeddings, context)
            .await
    }

    /// Get embeddings by chunk ID
    pub async fn get_embeddings(
        &self,
        chunk_id: &Uuid,
        context: &AccessContext,
    ) -> Result<Option<Vec<f32>>> {
        self.backend.get_embeddings(chunk_id, context).await
    }

    /// Search by vector
    pub async fn search_by_vector(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        context: &AccessContext,
    ) -> Result<Vec<(Uuid, f32)>> {
        self.backend
            .search_by_vector(query_embedding, top_k, context)
            .await
    }

    /// Get stats
    pub async fn stats(&self, context: &AccessContext) -> Result<StorageStats> {
        self.backend.stats(context).await
    }
}

pub mod benchmarks;

// Temporarily disabled due to compilation errors
// pub mod optimized;

/// Embedded storage configuration for local-first usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedStorageConfig {
    /// Path for storage data (used by file backend)
    pub data_path: PathBuf,
    /// Collection name for Qdrant
    pub collection_name: String,
    /// Vector dimension size
    pub vector_size: usize,
    /// Whether to require running Qdrant server (vs file-only mode)
    pub require_qdrant: bool,
    /// Qdrant URL for embedded mode (default: http://localhost:6333)
    pub qdrant_url: String,
}

impl Default for EmbeddedStorageConfig {
    fn default() -> Self {
        Self {
            data_path: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("reasonkit")
                .join("storage"),
            collection_name: "reasonkit_default".to_string(),
            vector_size: 1536, // OpenAI ada-002 default
            require_qdrant: false,
            qdrant_url: "http://localhost:6333".to_string(),
        }
    }
}

impl EmbeddedStorageConfig {
    /// Create config for file-only mode (no Qdrant required)
    pub fn file_only(data_path: PathBuf) -> Self {
        Self {
            data_path,
            require_qdrant: false,
            ..Default::default()
        }
    }

    /// Create config for local Qdrant mode
    pub fn with_qdrant(qdrant_url: &str, collection_name: &str, vector_size: usize) -> Self {
        Self {
            qdrant_url: qdrant_url.to_string(),
            collection_name: collection_name.to_string(),
            vector_size,
            require_qdrant: true,
            ..Default::default()
        }
    }
}

/// Create embedded storage with automatic fallback
///
/// This function attempts to create the best available storage:
/// 1. If `require_qdrant` is true and Qdrant is available: QdrantStorage
/// 2. Otherwise: FileStorage as fallback
///
/// # Example
/// ```ignore
/// let config = EmbeddedStorageConfig::default();
/// let storage = create_embedded_storage(config).await?;
/// ```
pub async fn create_embedded_storage(config: EmbeddedStorageConfig) -> Result<Storage> {
    // Ensure data directory exists
    if !config.data_path.exists() {
        std::fs::create_dir_all(&config.data_path).map_err(|e| {
            Error::io(format!(
                "Failed to create storage directory {:?}: {}",
                config.data_path, e
            ))
        })?;
        tracing::info!(path = ?config.data_path, "Created storage data directory");
    }

    if config.require_qdrant {
        // Try to connect to Qdrant
        match check_qdrant_health(&config.qdrant_url).await {
            Ok(()) => {
                tracing::info!(url = %config.qdrant_url, "Connected to Qdrant server");
                // Parse URL for host and port
                let url = config.qdrant_url.trim_start_matches("http://");
                let parts: Vec<&str> = url.split(':').collect();
                let host = parts.first().unwrap_or(&"localhost");
                let port: u16 = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(6333);

                return Storage::qdrant(
                    host,
                    port,
                    port + 1, // gRPC port typically port + 1
                    config.collection_name,
                    config.vector_size,
                    true, // embedded mode
                )
                .await;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Qdrant not available, require_qdrant=true will fail"
                );
                return Err(Error::io(format!(
                    "Qdrant required but not available at {}: {}",
                    config.qdrant_url, e
                )));
            }
        }
    }

    // Use file storage as fallback
    tracing::info!(path = ?config.data_path, "Using file-based storage (Qdrant not required)");
    Storage::file(config.data_path).await
}

/// Check if Qdrant server is healthy
async fn check_qdrant_health(url: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| Error::io(format!("Failed to create HTTP client: {}", e)))?;

    let health_url = format!("{}/readyz", url);
    let response = client
        .get(&health_url)
        .send()
        .await
        .map_err(|e| Error::io(format!("Qdrant health check failed: {}", e)))?;

    if response.status().is_success() {
        Ok(())
    } else {
        Err(Error::io(format!(
            "Qdrant health check returned status: {}",
            response.status()
        )))
    }
}

/// Get the default storage path for embedded mode
pub fn default_storage_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("reasonkit")
        .join("storage")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_config_default() {
        let config = EmbeddedStorageConfig::default();
        assert!(!config.require_qdrant);
        assert_eq!(config.vector_size, 1536);
        assert_eq!(config.collection_name, "reasonkit_default");
    }

    #[test]
    fn test_embedded_config_file_only() {
        let config = EmbeddedStorageConfig::file_only(PathBuf::from("/tmp/test"));
        assert!(!config.require_qdrant);
        assert_eq!(config.data_path, PathBuf::from("/tmp/test"));
    }

    #[test]
    fn test_embedded_config_with_qdrant() {
        let config =
            EmbeddedStorageConfig::with_qdrant("http://localhost:6334", "test_collection", 768);
        assert!(config.require_qdrant);
        assert_eq!(config.qdrant_url, "http://localhost:6334");
        assert_eq!(config.collection_name, "test_collection");
        assert_eq!(config.vector_size, 768);
    }

    #[test]
    fn test_default_storage_path() {
        let path = default_storage_path();
        assert!(path.ends_with("reasonkit/storage") || path.ends_with("reasonkit\\storage"));
    }

    #[tokio::test]
    async fn test_in_memory_storage() {
        use crate::{DocumentType, Source, SourceType};
        use chrono::Utc;

        let storage = Storage::in_memory();
        let context = AccessContext::new(
            "test_user".to_string(),
            AccessLevel::Admin,
            "test".to_string(),
        );

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("test.md".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let doc =
            Document::new(DocumentType::Note, source).with_content("Test content".to_string());

        storage.store_document(&doc, &context).await.unwrap();
        let retrieved = storage.get_document(&doc.id, &context).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content.raw, "Test content");
    }

    #[tokio::test]
    async fn test_file_storage_creation() {
        let temp_dir = std::env::temp_dir().join("reasonkit_storage_test");
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        let storage = Storage::file(temp_dir.clone()).await.unwrap();
        let context = AccessContext::new(
            "test_user".to_string(),
            AccessLevel::Admin,
            "test".to_string(),
        );

        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 0);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    async fn test_embedded_storage_file_fallback() {
        let temp_dir = std::env::temp_dir().join("reasonkit_embedded_test");
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }

        // Should fall back to file storage when require_qdrant is false
        let config = EmbeddedStorageConfig::file_only(temp_dir.clone());
        let storage = create_embedded_storage(config).await.unwrap();

        let context = AccessContext::new(
            "test_user".to_string(),
            AccessLevel::Admin,
            "test".to_string(),
        );

        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 0);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
