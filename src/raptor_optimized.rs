//! Optimized RAPTOR implementation with caching and beam search
//!
//! Improvements over basic RAPTOR:
//! - Beam search for efficient tree traversal
//! - Node embedding caching
//! - Parallel clustering
//! - Early termination pruning

use crate::{Document, Chunk, Result, Error};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Optimized RAPTOR node with caching support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedRaptorNode {
    pub id: Uuid,
    pub text: String,
    pub children: Vec<Uuid>,
    pub parent: Option<Uuid>,
    pub level: usize,
    pub embedding: Option<Vec<f32>>,

    // Optimization fields
    #[serde(skip)]
    pub embedding_cached: bool,
    #[serde(skip)]
    pub last_accessed: Option<std::time::Instant>,
}

/// Configuration for optimized RAPTOR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorOptConfig {
    /// Maximum tree depth
    pub max_depth: usize,

    /// Number of nodes to cluster together
    pub cluster_size: usize,

    /// Beam width for search (number of paths to explore)
    pub beam_width: usize,

    /// Cache size (number of embeddings to keep in memory)
    pub cache_size: usize,

    /// Enable parallel clustering during build
    pub parallel_clustering: bool,

    /// Minimum similarity threshold for early termination
    pub min_similarity: f32,
}

impl Default for RaptorOptConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            cluster_size: 5,
            beam_width: 3,
            cache_size: 1000,
            parallel_clustering: true,
            min_similarity: 0.1,
        }
    }
}

/// Optimized RAPTOR tree with caching and beam search
pub struct OptimizedRaptorTree {
    nodes: HashMap<Uuid, OptimizedRaptorNode>,
    roots: Vec<Uuid>,
    config: RaptorOptConfig,

    // Caching
    embedding_cache: Arc<RwLock<HashMap<Uuid, Vec<f32>>>>,
    access_order: Arc<RwLock<Vec<Uuid>>>,
}

impl OptimizedRaptorTree {
    /// Create a new optimized RAPTOR tree
    pub fn new(config: RaptorOptConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            config,
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Build tree from chunks (with parallel clustering if enabled)
    pub async fn build_from_chunks(
        &mut self,
        chunks: &[Chunk],
        embedder: &dyn Fn(&str) -> Result<Vec<f32>>,
        summarizer: &dyn Fn(&str) -> Result<String>,
    ) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        // Level 0: Create leaf nodes
        let leaf_nodes = self.create_leaf_nodes(chunks, embedder).await?;

        // Build hierarchical levels
        let mut current_level_nodes = leaf_nodes;
        for level in 1..=self.config.max_depth {
            if current_level_nodes.len() <= self.config.cluster_size {
                for node in &current_level_nodes {
                    self.roots.push(node.id);
                }
                break;
            }

            let next_level_nodes = if self.config.parallel_clustering {
                self.build_level_parallel(&current_level_nodes, level, embedder, summarizer).await?
            } else {
                self.build_level_sequential(&current_level_nodes, level, embedder, summarizer).await?
            };

            current_level_nodes = next_level_nodes;
        }

        // Remaining nodes become roots
        for node in current_level_nodes {
            self.roots.push(node.id);
        }

        Ok(())
    }

    /// Create leaf nodes with embedding caching
    async fn create_leaf_nodes(
        &mut self,
        chunks: &[Chunk],
        embedder: &dyn Fn(&str) -> Result<Vec<f32>>,
    ) -> Result<Vec<OptimizedRaptorNode>> {
        let mut leaf_nodes = Vec::new();

        for chunk in chunks {
            let embedding = embedder(&chunk.text)?;

            // Cache the embedding
            self.cache_embedding(chunk.id, embedding.clone());

            let node = OptimizedRaptorNode {
                id: chunk.id,
                text: chunk.text.clone(),
                children: Vec::new(),
                parent: None,
                level: 0,
                embedding: Some(embedding),
                embedding_cached: true,
                last_accessed: Some(std::time::Instant::now()),
            };

            self.nodes.insert(node.id, node.clone());
            leaf_nodes.push(node);
        }

        Ok(leaf_nodes)
    }

    /// Build level using sequential processing
    async fn build_level_sequential(
        &mut self,
        nodes: &[OptimizedRaptorNode],
        level: usize,
        embedder: &dyn Fn(&str) -> Result<Vec<f32>>,
        summarizer: &dyn Fn(&str) -> Result<String>,
    ) -> Result<Vec<OptimizedRaptorNode>> {
        let mut next_level_nodes = Vec::new();

        for i in (0..nodes.len()).step_by(self.config.cluster_size) {
            let cluster_end = (i + self.config.cluster_size).min(nodes.len());
            let cluster = &nodes[i..cluster_end];

            let cluster_node = self.create_cluster_node(cluster, level, embedder, summarizer)?;
            next_level_nodes.push(cluster_node);
        }

        Ok(next_level_nodes)
    }

    /// Build level using parallel processing (with rayon)
    async fn build_level_parallel(
        &mut self,
        nodes: &[OptimizedRaptorNode],
        level: usize,
        embedder: &dyn Fn(&str) -> Result<Vec<f32>>,
        summarizer: &dyn Fn(&str) -> Result<String>,
    ) -> Result<Vec<OptimizedRaptorNode>> {
        use rayon::prelude::*;

        let clusters: Vec<_> = (0..nodes.len())
            .step_by(self.config.cluster_size)
            .collect();

        let next_level_nodes: Result<Vec<_>> = clusters
            .par_iter()
            .map(|&i| {
                let cluster_end = (i + self.config.cluster_size).min(nodes.len());
                let cluster = &nodes[i..cluster_end];
                self.create_cluster_node(cluster, level, embedder, summarizer)
            })
            .collect();

        next_level_nodes
    }

    /// Create a cluster node from a group of nodes
    fn create_cluster_node(
        &mut self,
        cluster: &[OptimizedRaptorNode],
        level: usize,
        embedder: &dyn Fn(&str) -> Result<Vec<f32>>,
        summarizer: &dyn Fn(&str) -> Result<String>,
    ) -> Result<OptimizedRaptorNode> {
        if cluster.len() == 1 {
            let mut node = cluster[0].clone();
            node.level = level;
            return Ok(node);
        }

        // Create summary
        let cluster_texts: Vec<String> = cluster.iter()
            .map(|n| n.text.clone())
            .collect();
        let combined_text = cluster_texts.join("\n\n");

        let summary = summarizer(&combined_text)?;
        let embedding = embedder(&summary)?;

        let cluster_node_id = Uuid::new_v4();

        // Cache embedding
        self.cache_embedding(cluster_node_id, embedding.clone());

        let cluster_node = OptimizedRaptorNode {
            id: cluster_node_id,
            text: summary,
            children: cluster.iter().map(|n| n.id).collect(),
            parent: None,
            level,
            embedding: Some(embedding),
            embedding_cached: true,
            last_accessed: Some(std::time::Instant::now()),
        };

        // Update parent references
        for child in cluster {
            if let Some(child_node) = self.nodes.get_mut(&child.id) {
                child_node.parent = Some(cluster_node_id);
            }
        }

        self.nodes.insert(cluster_node.id, cluster_node.clone());
        Ok(cluster_node)
    }

    /// Beam search through the tree
    ///
    /// More efficient than exhaustive search - only explores top-k most promising paths
    pub fn search_beam(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<(Uuid, f32)>> {
        let mut beam: Vec<(Uuid, f32)> = Vec::new();

        // Initialize beam with root nodes
        for root_id in &self.roots {
            if let Some(node) = self.nodes.get(root_id) {
                if let Some(embedding) = self.get_embedding(&node.id) {
                    let score = cosine_similarity(query_embedding, &embedding);
                    if score >= self.config.min_similarity {
                        beam.push((*root_id, score));
                    }
                }
            }
        }

        if beam.is_empty() {
            return Ok(Vec::new());
        }

        // Traverse down the tree level by level
        let mut current_level = self.config.max_depth;

        while current_level > 0 {
            beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            beam.truncate(self.config.beam_width);

            let mut next_beam = Vec::new();

            for (node_id, _parent_score) in &beam {
                if let Some(node) = self.nodes.get(node_id) {
                    if node.level == 0 {
                        // Already at leaf level
                        next_beam.push((*node_id, _parent_score));
                        continue;
                    }

                    for child_id in &node.children {
                        if let Some(child) = self.nodes.get(child_id) {
                            if let Some(embedding) = self.get_embedding(&child.id) {
                                let score = cosine_similarity(query_embedding, &embedding);
                                if score >= self.config.min_similarity {
                                    next_beam.push((*child_id, score));
                                }
                            }
                        }
                    }
                }
            }

            if next_beam.is_empty() {
                break;
            }

            beam = next_beam;
            current_level -= 1;
        }

        // Final ranking
        beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        beam.truncate(top_k);

        Ok(beam)
    }

    /// Get embedding from cache or node
    fn get_embedding(&self, node_id: &Uuid) -> Option<Vec<f32>> {
        // Check cache first
        if let Ok(cache) = self.embedding_cache.read() {
            if let Some(embedding) = cache.get(node_id) {
                // Update access order
                if let Ok(mut order) = self.access_order.write() {
                    order.retain(|id| id != node_id);
                    order.push(*node_id);
                }
                return Some(embedding.clone());
            }
        }

        // Fall back to node storage
        if let Some(node) = self.nodes.get(node_id) {
            if let Some(ref embedding) = node.embedding {
                self.cache_embedding(*node_id, embedding.clone());
                return Some(embedding.clone());
            }
        }

        None
    }

    /// Cache an embedding with LRU eviction
    fn cache_embedding(&self, node_id: Uuid, embedding: Vec<f32>) {
        if let Ok(mut cache) = self.embedding_cache.write() {
            // Evict if over capacity
            while cache.len() >= self.config.cache_size {
                if let Ok(mut order) = self.access_order.write() {
                    if let Some(oldest_id) = order.first().cloned() {
                        order.remove(0);
                        cache.remove(&oldest_id);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            cache.insert(node_id, embedding);

            // Update access order
            if let Ok(mut order) = self.access_order.write() {
                order.push(node_id);
            }
        }
    }

    /// Get tree statistics
    pub fn stats(&self) -> RaptorOptStats {
        let mut level_counts = HashMap::new();
        let mut cached_count = 0;

        for node in self.nodes.values() {
            *level_counts.entry(node.level).or_insert(0) += 1;
            if node.embedding_cached {
                cached_count += 1;
            }
        }

        let cache_size = self.embedding_cache.read()
            .map(|c| c.len())
            .unwrap_or(0);

        RaptorOptStats {
            total_nodes: self.nodes.len(),
            leaf_nodes: level_counts.get(&0).copied().unwrap_or(0),
            max_depth: self.config.max_depth,
            level_counts,
            root_count: self.roots.len(),
            cache_size,
            cache_hit_rate: (cached_count as f32) / (self.nodes.len() as f32),
        }
    }
}

/// Statistics for optimized RAPTOR tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorOptStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub max_depth: usize,
    pub level_counts: HashMap<usize, usize>,
    pub root_count: usize,
    pub cache_size: usize,
    pub cache_hit_rate: f32,
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_embedder(_text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.1, 0.2, 0.3, 0.4, 0.5])
    }

    fn mock_summarizer(text: &str) -> Result<String> {
        Ok(format!("Summary: {}", text.chars().take(50).collect::<String>()))
    }

    #[tokio::test]
    async fn test_beam_search() {
        let config = RaptorOptConfig {
            max_depth: 2,
            cluster_size: 3,
            beam_width: 2,
            ..Default::default()
        };

        let mut tree = OptimizedRaptorTree::new(config);

        let chunks = vec![
            Chunk {
                id: Uuid::new_v4(),
                text: "Test chunk 1".to_string(),
                index: 0,
                start_char: 0,
                end_char: 12,
                token_count: Some(3),
                section: None,
                page: None,
                embedding_ids: Default::default(),
            },
            Chunk {
                id: Uuid::new_v4(),
                text: "Test chunk 2".to_string(),
                index: 1,
                start_char: 13,
                end_char: 25,
                token_count: Some(3),
                section: None,
                page: None,
                embedding_ids: Default::default(),
            },
        ];

        tree.build_from_chunks(&chunks, &mock_embedder, &mock_summarizer).await.unwrap();

        let query_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let results = tree.search_beam(&query_embedding, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results[0].1 > 0.0);
    }

    #[tokio::test]
    async fn test_caching() {
        let config = RaptorOptConfig {
            cache_size: 2,
            ..Default::default()
        };

        let tree = OptimizedRaptorTree::new(config);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        tree.cache_embedding(id1, vec![1.0, 2.0]);
        tree.cache_embedding(id2, vec![3.0, 4.0]);

        assert!(tree.get_embedding(&id1).is_some());
        assert!(tree.get_embedding(&id2).is_some());

        // This should evict id1 (LRU)
        tree.cache_embedding(id3, vec![5.0, 6.0]);

        assert!(tree.get_embedding(&id3).is_some());
    }
}
