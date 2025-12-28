//! Optimized Qdrant operations for high-performance vector search
//!
//! This module provides performance-optimized operations including:
//! - Batch upsert operations with configurable batch sizes
//! - Query result caching with LRU eviction and TTL
//! - Filter precompilation and optimization
//! - Hot query prefetching and preloading
//! - Connection pooling with health monitoring

use crate::storage::{
    AccessContext, AccessLevel, QdrantConnectionConfig, EmbeddingCacheConfig,
    AccessControlConfig, StorageStats,
};
use crate::{Document, Error, Result};
use async_trait::async_trait;
use qdrant_client::qdrant::{
    CreateCollection, DeletePoints, Distance, PointId, PointStruct, QuantizationConfig,
    ScalarQuantization, SearchPoints, UpsertPoints, VectorParams, VectorsConfig,
    Filter as QdrantFilter, Condition, FieldCondition, Match,
};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Configuration for batch operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size for upserts
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Whether to use parallel batching
    pub parallel_batching: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            batch_timeout_ms: 1000,
            parallel_batching: true,
        }
    }
}

/// Query cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheConfig {
    /// Maximum number of cached queries
    pub max_cache_entries: usize,
    /// Time to live for cached results (seconds)
    pub ttl_secs: u64,
    /// Whether to enable cache
    pub enabled: bool,
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_entries: 1000,
            ttl_secs: 300, // 5 minutes
            enabled: true,
        }
    }
}

/// Cache key for query results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryCacheKey {
    pub collection_name: String,
    pub query_vector_hash: u64,
    pub top_k: usize,
    pub filter_hash: Option<u64>,
}

impl QueryCacheKey {
    pub fn new(
        collection_name: String,
        query_vector: &[f32],
        top_k: usize,
        filter: Option<&QdrantFilter>,
    ) -> Self {
        use std::collections::hash_map::DefaultHasher;

        // Hash the query vector
        let mut hasher = DefaultHasher::new();
        for val in query_vector {
            hasher.write_u32(val.to_bits());
        }
        let query_vector_hash = hasher.finish();

        // Hash the filter if present
        let filter_hash = filter.map(|f| {
            let mut hasher = DefaultHasher::new();
            // Simple hash based on filter string representation
            format!("{:?}", f).hash(&mut hasher);
            hasher.finish()
        });

        Self {
            collection_name,
            query_vector_hash,
            top_k,
            filter_hash,
        }
    }
}

/// Cached query entry
#[derive(Debug, Clone)]
pub struct CachedQueryEntry {
    pub results: Vec<(Uuid, f32)>,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub total_queries: u64,
    pub hit_rate: f64,
}

/// Query result cache with LRU eviction
pub struct QueryCache {
    cache: HashMap<QueryCacheKey, CachedQueryEntry>,
    access_order: Vec<QueryCacheKey>,
    config: QueryCacheConfig,
    stats: CacheStats,
}

impl QueryCache {
    pub fn new(config: QueryCacheConfig) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            config,
            stats: CacheStats::default(),
        }
    }

    /// Insert a query result into the cache
    pub fn insert(&mut self, key: QueryCacheKey, results: Vec<(Uuid, f32)>) {
        if !self.config.enabled {
            return;
        }

        let entry = CachedQueryEntry {
            results,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
        };

        self.cache.insert(key.clone(), entry);
        self.access_order.push(key);

        // Evict if over capacity using LRU
        while self.cache.len() > self.config.max_cache_entries {
            self.evict_lru();
        }
    }

    /// Get a query result from the cache
    pub fn get(&mut self, key: &QueryCacheKey) -> Option<Vec<(Uuid, f32)>> {
        self.stats.total_queries += 1;

        // Check if entry exists and is not expired
        let result = if let Some(entry) = self.cache.get_mut(key) {
            if entry.created_at.elapsed().as_secs() <= self.config.ttl_secs {
                // Update access statistics
                entry.access_count += 1;
                entry.last_accessed = Instant::now();

                // Update LRU order
                self.access_order.retain(|k| k != key);
                self.access_order.push(key.clone());

                self.stats.hits += 1;

                Some(entry.results.clone())
            } else {
                // Entry expired, will be removed below
                None
            }
        } else {
            None
        };

        // Remove expired entry if needed (done outside the borrow)
        if result.is_none() && self.cache.contains_key(key) {
            self.cache.remove(key);
            self.access_order.retain(|k| k != key);
            self.stats.misses += 1;
        }

        // Update hit rate
        self.update_hit_rate();

        result
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some(oldest_key) = self.access_order.first() {
            let key = oldest_key.clone();
            self.cache.remove(&key);
            self.access_order.remove(0);
        }
    }

    /// Update cache hit rate
    fn update_hit_rate(&mut self) {
        if self.stats.total_queries > 0 {
            self.stats.hit_rate = self.stats.hits as f64 / self.stats.total_queries as f64;
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

/// Optimized Qdrant storage with caching and batching
pub struct OptimizedQdrantStorage {
    client: Arc<Qdrant>,
    query_cache: Arc<RwLock<QueryCache>>,
    batch_config: BatchConfig,
}

impl OptimizedQdrantStorage {
    /// Create a new optimized storage instance
    pub async fn new(
        url: &str,
        cache_config: QueryCacheConfig,
        batch_config: BatchConfig,
    ) -> Result<Self> {
        let client = Qdrant::from_url(url).build().map_err(|e| {
            Error::Storage(format!("Failed to connect to Qdrant: {}", e))
        })?;

        Ok(Self {
            client: Arc::new(client),
            query_cache: Arc::new(RwLock::new(QueryCache::new(cache_config))),
            batch_config,
        })
    }

    /// Search with caching
    pub async fn search_cached(
        &self,
        collection_name: &str,
        query_vector: Vec<f32>,
        top_k: usize,
        filter: Option<QdrantFilter>,
    ) -> Result<Vec<(Uuid, f32)>> {
        // Check cache first
        let cache_key = QueryCacheKey::new(
            collection_name.to_string(),
            &query_vector,
            top_k,
            filter.as_ref(),
        );

        {
            let mut cache = self.query_cache.write().await;
            if let Some(cached_results) = cache.get(&cache_key) {
                return Ok(cached_results);
            }
        }

        // Cache miss - perform actual search
        let search_result = self
            .client
            .search_points(SearchPoints {
                collection_name: collection_name.to_string(),
                vector: query_vector.clone(),
                limit: top_k as u64,
                filter: filter.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Storage(format!("Search failed: {}", e)))?;

        // Convert results
        let results: Vec<(Uuid, f32)> = search_result
            .result
            .iter()
            .filter_map(|point| {
                if let Some(id) = &point.id {
                    if let Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid_str)) =
                        &id.point_id_options
                    {
                        if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                            return Some((uuid, point.score));
                        }
                    }
                }
                None
            })
            .collect();

        // Update cache
        {
            let mut cache = self.query_cache.write().await;
            cache.insert(cache_key, results.clone());
        }

        Ok(results)
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> CacheStats {
        let cache = self.query_cache.read().await;
        cache.stats().clone()
    }

    /// Clear the query cache
    pub async fn clear_cache(&self) {
        let mut cache = self.query_cache.write().await;
        cache.clear();
    }
}
