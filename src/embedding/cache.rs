//! Embedding cache for reducing API calls and improving performance
//!
//! Provides an in-memory LRU cache with TTL support for embedding vectors.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Entry in the embedding cache
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The embedding vector
    embedding: Vec<f32>,
    /// When this entry was created
    created_at: Instant,
    /// Last access time (for LRU)
    last_accessed: Instant,
}

impl CacheEntry {
    fn new(embedding: Vec<f32>) -> Self {
        let now = Instant::now();
        Self {
            embedding,
            created_at: now,
            last_accessed: now,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }
}

/// Thread-safe embedding cache with LRU eviction and TTL support
pub struct EmbeddingCache {
    /// Cache storage (key -> entry)
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Maximum number of entries
    max_entries: usize,
    /// Time-to-live in seconds (0 = no expiration)
    ttl_secs: u64,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of cached embeddings
    /// * `ttl_secs` - Time-to-live in seconds (0 for no expiration)
    pub fn new(max_entries: usize, ttl_secs: u64) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
            ttl_secs,
        }
    }

    /// Get an embedding from the cache
    ///
    /// Returns `None` if not found or expired.
    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        let mut cache = self.cache.write().ok()?;

        if let Some(entry) = cache.get_mut(key) {
            // Check if expired
            if self.ttl_secs > 0 && entry.is_expired(Duration::from_secs(self.ttl_secs)) {
                cache.remove(key);
                return None;
            }

            // Update access time
            entry.touch();
            Some(entry.embedding.clone())
        } else {
            None
        }
    }

    /// Put an embedding into the cache
    ///
    /// If the cache is full, evicts the least recently used entry.
    pub fn put(&self, key: String, embedding: Vec<f32>) {
        let mut cache = self.cache.write().expect("Failed to acquire cache lock");

        // If at capacity, evict LRU entry
        if cache.len() >= self.max_entries && !cache.contains_key(&key) {
            self.evict_lru(&mut cache);
        }

        cache.insert(key, CacheEntry::new(embedding));
    }

    /// Evict the least recently used entry
    fn evict_lru(&self, cache: &mut HashMap<String, CacheEntry>) {
        if cache.is_empty() {
            return;
        }

        // Find LRU entry
        let lru_key = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            cache.remove(&key);
        }
    }

    /// Clear all expired entries
    pub fn clear_expired(&self) {
        if self.ttl_secs == 0 {
            return; // No TTL, nothing to clear
        }

        let mut cache = self.cache.write().expect("Failed to acquire cache lock");
        let ttl = Duration::from_secs(self.ttl_secs);

        cache.retain(|_, entry| !entry.is_expired(ttl));
    }

    /// Clear all entries
    pub fn clear(&self) {
        let mut cache = self.cache.write().expect("Failed to acquire cache lock");
        cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read().expect("Failed to acquire cache lock");

        let expired_count = if self.ttl_secs > 0 {
            let ttl = Duration::from_secs(self.ttl_secs);
            cache.values().filter(|entry| entry.is_expired(ttl)).count()
        } else {
            0
        };

        CacheStats {
            total_entries: cache.len(),
            max_entries: self.max_entries,
            expired_entries: expired_count,
            utilization: cache.len() as f32 / self.max_entries as f32,
        }
    }

    /// Get the number of cached entries
    pub fn len(&self) -> usize {
        self.cache.read().ok().map(|c| c.len()).unwrap_or(0)
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Maximum allowed entries
    pub max_entries: usize,
    /// Number of expired entries
    pub expired_entries: usize,
    /// Cache utilization (0.0 - 1.0)
    pub utilization: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_cache_basic() {
        let cache = EmbeddingCache::new(3, 0);

        // Insert entries
        cache.put("key1".to_string(), vec![1.0, 2.0, 3.0]);
        cache.put("key2".to_string(), vec![4.0, 5.0, 6.0]);

        // Retrieve entries
        assert_eq!(cache.get("key1"), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(cache.get("key2"), Some(vec![4.0, 5.0, 6.0]));
        assert_eq!(cache.get("key3"), None);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let cache = EmbeddingCache::new(2, 0);

        cache.put("key1".to_string(), vec![1.0]);
        cache.put("key2".to_string(), vec![2.0]);

        // Access key1 to make it more recently used
        cache.get("key1");

        // This should evict key2 (LRU)
        cache.put("key3".to_string(), vec![3.0]);

        assert_eq!(cache.get("key1"), Some(vec![1.0]));
        assert_eq!(cache.get("key2"), None); // Evicted
        assert_eq!(cache.get("key3"), Some(vec![3.0]));
    }

    #[test]
    fn test_cache_ttl() {
        let cache = EmbeddingCache::new(10, 1); // 1 second TTL

        cache.put("key1".to_string(), vec![1.0, 2.0]);
        assert_eq!(cache.get("key1"), Some(vec![1.0, 2.0]));

        // Wait for expiration
        thread::sleep(Duration::from_millis(1100));
        assert_eq!(cache.get("key1"), None); // Expired
    }

    #[test]
    fn test_cache_stats() {
        let cache = EmbeddingCache::new(5, 0);

        cache.put("key1".to_string(), vec![1.0]);
        cache.put("key2".to_string(), vec![2.0]);

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.max_entries, 5);
        assert_eq!(stats.utilization, 0.4);
    }

    #[test]
    fn test_cache_clear() {
        let cache = EmbeddingCache::new(10, 0);

        cache.put("key1".to_string(), vec![1.0]);
        cache.put("key2".to_string(), vec![2.0]);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_clear_expired() {
        let cache = EmbeddingCache::new(10, 1); // 1 second TTL

        cache.put("key1".to_string(), vec![1.0]);
        cache.put("key2".to_string(), vec![2.0]);

        thread::sleep(Duration::from_millis(1100));

        cache.clear_expired();
        assert_eq!(cache.len(), 0);
    }
}
