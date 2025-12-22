//! RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) module
//!
//! Implements hierarchical clustering and summarization for improved retrieval quality.

use crate::{Document, Chunk, Result, Error};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// RAPTOR tree node representing a cluster of chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorNode {
    /// Unique ID for this node
    pub id: Uuid,
    /// Text content (summary for internal nodes, original text for leaves)
    pub text: String,
    /// Child nodes (empty for leaf nodes)
    pub children: Vec<Uuid>,
    /// Parent node (None for root)
    pub parent: Option<Uuid>,
    /// Tree level (0 = leaves, higher = more abstract)
    pub level: usize,
    /// Embedding vector for this node
    pub embedding: Option<Vec<f32>>,
}

/// RAPTOR tree for hierarchical retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorTree {
    /// All nodes in the tree
    pub nodes: HashMap<Uuid, RaptorNode>,
    /// Root node IDs
    pub roots: Vec<Uuid>,
    /// Maximum depth of the tree
    pub max_depth: usize,
    /// Cluster size for summarization
    pub cluster_size: usize,
}

impl RaptorTree {
    /// Create a new empty RAPTOR tree
    pub fn new(max_depth: usize, cluster_size: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            max_depth,
            cluster_size,
        }
    }

    /// Build RAPTOR tree from document chunks
    pub async fn build_from_chunks(
        &mut self,
        chunks: &[Chunk],
        embedder: &dyn Fn(&str) -> Result<Vec<f32>>,
        summarizer: &dyn Fn(&str) -> Result<String>,
    ) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        // Level 0: Create leaf nodes from chunks
        let mut leaf_nodes = Vec::new();
        for chunk in chunks {
            let embedding = embedder(&chunk.text)?;
            let node = RaptorNode {
                id: chunk.id,
                text: chunk.text.clone(),
                children: Vec::new(),
                parent: None,
                level: 0,
                embedding: Some(embedding),
            };
            self.nodes.insert(node.id, node.clone());
            leaf_nodes.push(node);
        }

        // Build hierarchical levels
        let mut current_level_nodes = leaf_nodes;
        for level in 1..=self.max_depth {
            if current_level_nodes.len() <= self.cluster_size {
                // Not enough nodes to cluster, make them roots
                for node in &current_level_nodes {
                    self.roots.push(node.id);
                }
                break;
            }

            let next_level_nodes = self.build_level(
                &current_level_nodes,
                level,
                embedder,
                summarizer,
            ).await?;

            current_level_nodes = next_level_nodes;
        }

        // Any remaining nodes become roots
        for node in current_level_nodes {
            self.roots.push(node.id);
        }

        Ok(())
    }

    /// Build a single level of the tree by clustering and summarizing
    async fn build_level(
        &mut self,
        nodes: &[RaptorNode],
        level: usize,
        embedder: &dyn Fn(&str) -> Result<Vec<f32>>,
        summarizer: &dyn Fn(&str) -> Result<String>,
    ) -> Result<Vec<RaptorNode>> {
        let mut next_level_nodes = Vec::new();

        for i in (0..nodes.len()).step_by(self.cluster_size) {
            let cluster_end = (i + self.cluster_size).min(nodes.len());
            let cluster = &nodes[i..cluster_end];

            if cluster.len() == 1 {
                // Single node, promote to next level
                let mut node = cluster[0].clone();
                node.level = level;
                next_level_nodes.push(node);
                continue;
            }

            // Create cluster summary
            let cluster_texts: Vec<String> = cluster.iter()
                .map(|n| n.text.clone())
                .collect();
            let combined_text = cluster_texts.join("\n\n");

            let summary = summarizer(&combined_text)?;
            let embedding = embedder(&summary)?;

            let cluster_node = RaptorNode {
                id: Uuid::new_v4(),
                text: summary,
                children: cluster.iter().map(|n| n.id).collect(),
                parent: None,
                level,
                embedding: Some(embedding),
            };

            // Update parent references for children
            for child in cluster {
                if let Some(child_node) = self.nodes.get_mut(&child.id) {
                    child_node.parent = Some(cluster_node.id);
                }
            }

            self.nodes.insert(cluster_node.id, cluster_node.clone());
            next_level_nodes.push(cluster_node);
        }

        Ok(next_level_nodes)
    }

    /// Search the RAPTOR tree using tree traversal
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        let mut candidates = Vec::new();

        // Search all nodes
        for (node_id, node) in &self.nodes {
            if let Some(embedding) = &node.embedding {
                if embedding.len() == query_embedding.len() {
                    let similarity = cosine_similarity(query_embedding, embedding);
                    candidates.push((*node_id, similarity));
                }
            }
        }

        // Sort by similarity descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top candidates and expand to leaf nodes
        let mut leaf_results = Vec::new();
        for (node_id, score) in candidates.into_iter().take(top_k * 2) {
            leaf_results.extend(self.expand_to_leaves(node_id, score));
        }

        // Remove duplicates and sort again
        let mut unique_results = HashMap::new();
        for (leaf_id, score) in leaf_results {
            unique_results.entry(leaf_id)
                .and_modify(|e: &mut f32| *e = e.max(score))
                .or_insert(score);
        }

        let mut final_results: Vec<(Uuid, f32)> = unique_results.into_iter().collect();
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(final_results.into_iter().take(top_k).collect())
    }

    /// Expand a node to its leaf descendants with score propagation
    fn expand_to_leaves(&self, node_id: Uuid, score: f32) -> Vec<(Uuid, f32)> {
        let mut results = Vec::new();

        if let Some(node) = self.nodes.get(&node_id) {
            if node.children.is_empty() {
                // This is a leaf node
                results.push((node_id, score));
            } else {
                // Recursively expand children
                for child_id in &node.children {
                    results.extend(self.expand_to_leaves(*child_id, score));
                }
            }
        }

        results
    }

    /// Get all leaf nodes under a given node
    pub fn get_leaf_nodes(&self, node_id: Uuid) -> Vec<Uuid> {
        self.expand_to_leaves(node_id, 1.0).into_iter().map(|(id, _)| id).collect()
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: &Uuid) -> Option<&RaptorNode> {
        self.nodes.get(node_id)
    }

    /// Get tree statistics
    pub fn stats(&self) -> RaptorStats {
        let mut level_counts = HashMap::new();
        let mut total_nodes = 0;
        let mut leaf_nodes = 0;

        for node in self.nodes.values() {
            *level_counts.entry(node.level).or_insert(0) += 1;
            total_nodes += 1;

            if node.children.is_empty() {
                leaf_nodes += 1;
            }
        }

        RaptorStats {
            total_nodes,
            leaf_nodes,
            max_depth: self.max_depth,
            level_counts,
            root_count: self.roots.len(),
        }
    }
}

/// Statistics for RAPTOR tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorStats {
    /// Total number of nodes in the tree
    pub total_nodes: usize,
    /// Number of leaf nodes
    pub leaf_nodes: usize,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Node count per level
    pub level_counts: HashMap<usize, usize>,
    /// Number of root nodes
    pub root_count: usize,
}

/// Compute cosine similarity between two vectors
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
        Ok(format!("Summary of: {}", text.chars().take(50).collect::<String>()))
    }

    #[tokio::test]
    async fn test_raptor_tree_build() {
        let mut tree = RaptorTree::new(2, 3);

        let chunks = vec![
            Chunk {
                id: Uuid::new_v4(),
                text: "First chunk of text".to_string(),
                index: 0,
                start_char: 0,
                end_char: 20,
                token_count: Some(5),
                section: None,
                page: None,
                embedding_ids: Default::default(),
            },
            Chunk {
                id: Uuid::new_v4(),
                text: "Second chunk of text".to_string(),
                index: 1,
                start_char: 21,
                end_char: 42,
                token_count: Some(5),
                section: None,
                page: None,
                embedding_ids: Default::default(),
            },
            Chunk {
                id: Uuid::new_v4(),
                text: "Third chunk of text".to_string(),
                index: 2,
                start_char: 43,
                end_char: 63,
                token_count: Some(5),
                section: None,
                page: None,
                embedding_ids: Default::default(),
            },
            Chunk {
                id: Uuid::new_v4(),
                text: "Fourth chunk of text".to_string(),
                index: 3,
                start_char: 64,
                end_char: 85,
                token_count: Some(5),
                section: None,
                page: None,
                embedding_ids: Default::default(),
            },
        ];

        tree.build_from_chunks(&chunks, &mock_embedder, &mock_summarizer).await.unwrap();

        let stats = tree.stats();
        assert!(stats.total_nodes > chunks.len()); // Should have internal nodes
        assert_eq!(stats.leaf_nodes, chunks.len());
    }

    #[test]
    fn test_raptor_search() {
        let mut tree = RaptorTree::new(1, 2);
        let node_id = Uuid::new_v4();

        let node = RaptorNode {
            id: node_id,
            text: "Test node".to_string(),
            children: vec![],
            parent: None,
            level: 0,
            embedding: Some(vec![1.0, 0.0, 0.0]),
        };

        tree.nodes.insert(node_id, node);
        tree.roots.push(node_id);

        let query = vec![1.0, 0.0, 0.0];
        let results = tree.search(&query, 5).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, node_id);
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }
}