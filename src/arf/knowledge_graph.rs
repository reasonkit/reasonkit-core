//! # Knowledge Graph Module
//!
//! Persistent knowledge graph for storing and retrieving reasoning patterns.
//! This module creates a semantic network of concepts, solutions, and relationships.

use crate::arf::types::*;
use crate::error::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Dfs;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use sled::Db;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Knowledge node in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub id: String,
    pub node_type: NodeType,
    pub content: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Types of knowledge nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Problem,
    Solution,
    Concept,
    Pattern,
    Session,
    Step,
    Finding,
}

/// Relationship between knowledge nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: EdgeType,
    pub weight: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Types of relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    Solves,      // Solution solves problem
    RelatedTo,   // General relationship
    SimilarTo,   // Semantic similarity
    Follows,     // Sequential relationship
    Contains,    // Hierarchical containment
    References,  // Citation or reference
    Contradicts, // Contradictory relationship
}

/// Knowledge graph for storing reasoning patterns
pub struct KnowledgeGraph {
    graph: Arc<RwLock<DiGraph<KnowledgeNode, KnowledgeEdge>>>,
    node_index: Arc<RwLock<HashMap<String, NodeIndex>>>,
    database: Arc<Db>,
    embedding_model: Option<SentenceEmbeddingsModel>,
}

impl KnowledgeGraph {
    /// Create a new knowledge graph
    pub async fn new(database_path: &str) -> Result<Self> {
        let database = sled::open(database_path)?;

        // Try to load existing graph, or create new one
        let graph = Self::load_graph(&database)
            .await
            .unwrap_or_else(|_| DiGraph::new());

        Ok(Self {
            graph: Arc::new(RwLock::new(graph)),
            node_index: Arc::new(RwLock::new(HashMap::new())),
            database: Arc::new(database),
            embedding_model: None, // Would be initialized with actual model
        })
    }

    /// Add a reasoning session to the knowledge graph
    pub async fn add_session(&self, session: &ReasoningSession) -> Result<()> {
        // Create session node
        let session_node = KnowledgeNode {
            id: format!("session_{}", session.id),
            node_type: NodeType::Session,
            content: session.problem_statement.clone(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("status".to_string(), serde_json::json!(session.status));
                meta.insert(
                    "step_count".to_string(),
                    serde_json::json!(session.steps.len()),
                );
                meta
            },
            embedding: None,
            created_at: session.created_at,
            updated_at: session.updated_at,
        };

        self.add_node(session_node).await?;

        // Add problem node
        let problem_node = KnowledgeNode {
            id: format!("problem_{}", session.id),
            node_type: NodeType::Problem,
            content: session.problem_statement.clone(),
            metadata: HashMap::new(),
            embedding: None,
            created_at: session.created_at,
            updated_at: session.updated_at,
        };

        let problem_idx = self.add_node(problem_node).await?;

        // Connect session to problem
        self.add_edge(
            format!("session_{}", session.id),
            format!("problem_{}", session.id),
            EdgeType::Solves,
            1.0,
        )
        .await?;

        // Add reasoning steps
        for step in &session.steps {
            let step_node = KnowledgeNode {
                id: format!("step_{}", step.id),
                node_type: NodeType::Step,
                content: step.instruction.clone(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert(
                        "step_number".to_string(),
                        serde_json::json!(step.step_number),
                    );
                    meta.insert(
                        "cognitive_stance".to_string(),
                        serde_json::json!(step.cognitive_stance),
                    );
                    meta
                },
                embedding: None,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            };

            self.add_node(step_node).await?;

            // Connect step to session
            self.add_edge(
                format!("step_{}", step.id),
                format!("session_{}", session.id),
                EdgeType::Contains,
                0.8,
            )
            .await?;
        }

        // Save graph to database
        self.save_graph().await?;

        Ok(())
    }

    /// Search for similar problems or solutions
    pub async fn semantic_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        // Generate embedding for query (simplified - would use actual model)
        let query_embedding = self.generate_embedding(query).await?;

        // Search through nodes
        let graph = self.graph.read().await;
        let node_index = self.node_index.read().await;

        for (node_id, &node_idx) in node_index.iter() {
            if let Some(node) = graph.node_weight(node_idx) {
                if let Some(node_embedding) = &node.embedding {
                    let similarity = self.cosine_similarity(&query_embedding, node_embedding);

                    if similarity > 0.7 {
                        // Similarity threshold
                        results.push(SearchResult {
                            node_id: node_id.clone(),
                            content: node.content.clone(),
                            node_type: node.node_type.clone(),
                            similarity_score: similarity,
                            metadata: node.metadata.clone(),
                        });
                    }
                }
            }
        }

        // Sort by similarity and limit results
        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        results.truncate(limit);

        Ok(results)
    }

    /// Find solution patterns for similar problems
    pub async fn find_solution_patterns(
        &self,
        problem_description: &str,
    ) -> Result<Vec<SolutionPattern>> {
        // Search for similar problems
        let similar_problems = self.semantic_search(problem_description, 5).await?;

        let mut patterns = Vec::new();

        for problem in similar_problems {
            // Find solutions that solved this problem
            let solutions = self.find_solutions_for_problem(&problem.node_id).await?;

            for solution in solutions {
                patterns.push(SolutionPattern {
                    problem_description: problem.content.clone(),
                    solution_description: solution.content.clone(),
                    success_rate: self.calculate_success_rate(&solution.node_id).await?,
                    context: solution.metadata.clone(),
                });
            }
        }

        Ok(patterns)
    }

    /// Add a new knowledge node
    async fn add_node(&self, node: KnowledgeNode) -> Result<NodeIndex> {
        let mut graph = self.graph.write().await;
        let mut node_index = self.node_index.write().await;

        // Generate embedding if model is available
        let node_with_embedding = if self.embedding_model.is_some() {
            KnowledgeNode {
                embedding: Some(self.generate_embedding(&node.content).await?),
                ..node
            }
        } else {
            node
        };

        let node_idx = graph.add_node(node_with_embedding.clone());
        node_index.insert(node_with_embedding.id.clone(), node_idx);

        Ok(node_idx)
    }

    /// Add an edge between nodes
    async fn add_edge(
        &self,
        source_id: String,
        target_id: String,
        edge_type: EdgeType,
        weight: f64,
    ) -> Result<()> {
        let graph = self.graph.read().await;
        let node_index = self.node_index.read().await;

        let source_idx = *node_index
            .get(&source_id)
            .ok_or_else(|| ArfError::engine("Source node not found"))?;
        let target_idx = *node_index
            .get(&target_id)
            .ok_or_else(|| ArfError::engine("Target node not found"))?;

        let edge = KnowledgeEdge {
            source_id,
            target_id,
            edge_type,
            weight,
            metadata: HashMap::new(),
        };

        drop(node_index);
        let mut graph_mut = self.graph.write().await;
        graph_mut.add_edge(source_idx, target_idx, edge);

        Ok(())
    }

    /// Generate text embedding (simplified)
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // In a real implementation, this would use the embedding model
        // For simulation, we'll create a simple hash-based embedding
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // Create a simple embedding vector
        let embedding: Vec<f32> = (0..384)
            .map(|i| ((hash.wrapping_mul(i as u64)) % 1000) as f32 / 1000.0)
            .collect();

        Ok(embedding)
    }

    /// Calculate cosine similarity between embeddings
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot_product / (norm_a * norm_b)) as f64
        }
    }

    /// Find solutions for a given problem
    async fn find_solutions_for_problem(&self, problem_id: &str) -> Result<Vec<KnowledgeNode>> {
        let graph = self.graph.read().await;
        let node_index = self.node_index.read().await;

        let mut solutions = Vec::new();

        if let Some(&problem_idx) = node_index.get(problem_id) {
            // Find all outgoing edges (solutions that solve this problem)
            let mut dfs = Dfs::new(&*graph, problem_idx);

            while let Some(node_idx) = dfs.next(&*graph) {
                if let Some(node) = graph.node_weight(node_idx) {
                    if matches!(node.node_type, NodeType::Solution) {
                        solutions.push(node.clone());
                    }
                }
            }
        }

        Ok(solutions)
    }

    /// Calculate success rate for a solution
    async fn calculate_success_rate(&self, solution_id: &str) -> Result<f64> {
        // Simplified - in reality would analyze historical success data
        Ok(0.85) // Mock success rate
    }

    /// Save graph to database
    async fn save_graph(&self) -> Result<()> {
        let graph = self.graph.read().await;
        let serialized = serde_json::to_string(&*graph)?;
        self.database
            .insert(b"knowledge_graph", serialized.as_bytes())?;
        Ok(())
    }

    /// Load graph from database
    async fn load_graph(database: &Db) -> Result<DiGraph<KnowledgeNode, KnowledgeEdge>> {
        if let Some(data) = database.get(b"knowledge_graph")? {
            let graph: DiGraph<KnowledgeNode, KnowledgeEdge> = serde_json::from_slice(&data)?;
            Ok(graph)
        } else {
            Ok(DiGraph::new())
        }
    }

    /// Get knowledge graph statistics
    pub async fn get_statistics(&self) -> Result<KnowledgeStats> {
        let graph = self.graph.read().await;
        let node_index = self.node_index.read().await;

        let total_nodes = graph.node_count();
        let total_edges = graph.edge_count();

        let node_types = graph.node_weights().fold(HashMap::new(), |mut acc, node| {
            let count = acc.entry(format!("{:?}", node.node_type)).or_insert(0);
            *count += 1;
            acc
        });

        Ok(KnowledgeStats {
            total_nodes,
            total_edges,
            node_types,
            database_size: self.database.size_on_disk().unwrap_or(0),
        })
    }
}

/// Search result from knowledge graph
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: String,
    pub content: String,
    pub node_type: NodeType,
    pub similarity_score: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Solution pattern found in knowledge graph
#[derive(Debug, Clone)]
pub struct SolutionPattern {
    pub problem_description: String,
    pub solution_description: String,
    pub success_rate: f64,
    pub context: HashMap<String, serde_json::Value>,
}

/// Knowledge graph statistics
#[derive(Debug, Clone)]
pub struct KnowledgeStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub node_types: HashMap<String, usize>,
    pub database_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_knowledge_graph_creation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let kg = KnowledgeGraph::new(db_path.to_str().unwrap())
            .await
            .unwrap();
        let stats = kg.get_statistics().await.unwrap();
        assert_eq!(stats.total_nodes, 0);
    }

    #[tokio::test]
    async fn test_session_addition() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let kg = KnowledgeGraph::new(db_path.to_str().unwrap())
            .await
            .unwrap();

        let session = ReasoningSession {
            id: "test_session".to_string(),
            problem_statement: "Test problem".to_string(),
            status: SessionStatus::Completed,
            current_step: 1,
            total_steps: 1,
            steps: vec![],
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        kg.add_session(&session).await.unwrap();

        let stats = kg.get_statistics().await.unwrap();
        assert!(stats.total_nodes >= 2); // Session + Problem nodes
    }
}
