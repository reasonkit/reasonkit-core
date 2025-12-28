//! Performance benchmarking for storage backends
//!
//! Provides comprehensive benchmarking suite for storage operations,
//! search latency, and resource usage analysis.

use crate::storage::{AccessContext, AccessLevel, Storage};
use crate::{Document, DocumentType, Result, Source, SourceType};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Benchmark results for storage operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageBenchmarkResults {
    /// Document storage results
    pub document_storage: OperationBenchmark,
    /// Document retrieval results
    pub document_retrieval: OperationBenchmark,
    /// Embedding storage results
    pub embedding_storage: OperationBenchmark,
    /// Embedding retrieval results
    pub embedding_retrieval: OperationBenchmark,
    /// Vector search results
    pub vector_search: OperationBenchmark,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Overall benchmark metadata
    pub metadata: BenchmarkMetadata,
}

/// Individual operation benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationBenchmark {
    /// Number of operations performed
    pub operations_count: usize,
    /// Total duration
    pub total_duration: Duration,
    /// Average latency per operation
    pub avg_latency: Duration,
    /// Minimum latency
    pub min_latency: Duration,
    /// Maximum latency
    pub max_latency: Duration,
    /// 95th percentile latency
    pub p95_latency: Duration,
    /// Operations per second
    pub ops_per_second: f64,
    /// Error count
    pub error_count: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Average memory usage in bytes
    pub avg_memory_bytes: u64,
    /// Memory efficiency (operations per byte)
    pub memory_efficiency: f64,
}

/// Benchmark metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// Benchmark start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Benchmark duration
    pub duration: Duration,
    /// Storage backend type
    pub backend_type: String,
    /// Configuration used
    pub config: serde_json::Value,
    /// Test dataset size
    pub dataset_size: usize,
}

/// Storage performance benchmarker
pub struct StorageBenchmarker {
    storage: Storage,
    context: AccessContext,
}

impl StorageBenchmarker {
    /// Create a new benchmarker
    pub fn new(storage: Storage, user_id: String, access_level: AccessLevel) -> Self {
        let context = AccessContext::new(user_id, access_level, "benchmark".to_string());
        Self { storage, context }
    }

    /// Run comprehensive storage benchmarks
    pub async fn run_comprehensive_benchmark(
        &self,
        document_count: usize,
        embedding_count: usize,
        vector_size: usize,
    ) -> Result<StorageBenchmarkResults> {
        let start_time = Utc::now();
        let benchmark_start = Instant::now();

        // Generate test data
        let documents = self.generate_test_documents(document_count)?;
        let embeddings = self.generate_test_embeddings(embedding_count, vector_size)?;

        // Run individual benchmarks
        let document_storage = self.benchmark_document_storage(&documents).await?;
        let document_retrieval = self.benchmark_document_retrieval(&documents).await?;
        let embedding_storage = self.benchmark_embedding_storage(&embeddings).await?;
        let embedding_retrieval = self.benchmark_embedding_retrieval(&embeddings).await?;
        let vector_search = self.benchmark_vector_search(&embeddings, 100).await?;

        // Collect memory stats (simplified - would need actual memory profiling)
        let memory_stats = self.collect_memory_stats().await?;

        let metadata = BenchmarkMetadata {
            start_time,
            duration: benchmark_start.elapsed(),
            backend_type: "qdrant".to_string(), // Would be dynamic
            config: serde_json::json!({}),
            dataset_size: document_count + embedding_count,
        };

        Ok(StorageBenchmarkResults {
            document_storage,
            document_retrieval,
            embedding_storage,
            embedding_retrieval,
            vector_search,
            memory_stats,
            metadata,
        })
    }

    /// Benchmark document storage operations
    async fn benchmark_document_storage(
        &self,
        documents: &[Document],
    ) -> Result<OperationBenchmark> {
        let mut latencies = Vec::new();
        let mut error_count = 0;

        for doc in documents {
            let start = Instant::now();
            match self.storage.store_document(doc, &self.context).await {
                Ok(_) => {
                    latencies.push(start.elapsed());
                }
                Err(_) => {
                    error_count += 1;
                }
            }
        }

        Ok(self.calculate_operation_stats(latencies, error_count))
    }

    /// Benchmark document retrieval operations
    async fn benchmark_document_retrieval(
        &self,
        documents: &[Document],
    ) -> Result<OperationBenchmark> {
        let mut latencies = Vec::new();
        let mut error_count = 0;

        for doc in documents {
            let start = Instant::now();
            match self.storage.get_document(&doc.id, &self.context).await {
                Ok(Some(_)) => {
                    latencies.push(start.elapsed());
                }
                Ok(None) => {
                    error_count += 1; // Document not found
                }
                Err(_) => {
                    error_count += 1;
                }
            }
        }

        Ok(self.calculate_operation_stats(latencies, error_count))
    }

    /// Benchmark embedding storage operations
    async fn benchmark_embedding_storage(
        &self,
        embeddings: &[(Uuid, Vec<f32>)],
    ) -> Result<OperationBenchmark> {
        let mut latencies = Vec::new();
        let mut error_count = 0;

        for (chunk_id, embedding) in embeddings {
            let start = Instant::now();
            match self
                .storage
                .store_embeddings(chunk_id, embedding, &self.context)
                .await
            {
                Ok(_) => {
                    latencies.push(start.elapsed());
                }
                Err(_) => {
                    error_count += 1;
                }
            }
        }

        Ok(self.calculate_operation_stats(latencies, error_count))
    }

    /// Benchmark embedding retrieval operations
    async fn benchmark_embedding_retrieval(
        &self,
        embeddings: &[(Uuid, Vec<f32>)],
    ) -> Result<OperationBenchmark> {
        let mut latencies = Vec::new();
        let mut error_count = 0;

        for (chunk_id, _) in embeddings {
            let start = Instant::now();
            match self.storage.get_embeddings(chunk_id, &self.context).await {
                Ok(Some(_)) => {
                    latencies.push(start.elapsed());
                }
                Ok(None) => {
                    error_count += 1; // Embedding not found
                }
                Err(_) => {
                    error_count += 1;
                }
            }
        }

        Ok(self.calculate_operation_stats(latencies, error_count))
    }

    /// Benchmark vector search operations
    async fn benchmark_vector_search(
        &self,
        embeddings: &[(Uuid, Vec<f32>)],
        searches_per_embedding: usize,
    ) -> Result<OperationBenchmark> {
        let mut latencies = Vec::new();
        let mut error_count = 0;

        // Use first embedding as query vector
        if let Some((_, query_vector)) = embeddings.first() {
            for _ in 0..(embeddings.len() * searches_per_embedding) {
                let start = Instant::now();
                match self
                    .storage
                    .search_by_vector(query_vector, 10, &self.context)
                    .await
                {
                    Ok(_) => {
                        latencies.push(start.elapsed());
                    }
                    Err(_) => {
                        error_count += 1;
                    }
                }
            }
        }

        Ok(self.calculate_operation_stats(latencies, error_count))
    }

    /// Calculate operation statistics from latency measurements
    fn calculate_operation_stats(
        &self,
        mut latencies: Vec<Duration>,
        error_count: usize,
    ) -> OperationBenchmark {
        if latencies.is_empty() {
            return OperationBenchmark {
                operations_count: 0,
                total_duration: Duration::ZERO,
                avg_latency: Duration::ZERO,
                min_latency: Duration::ZERO,
                max_latency: Duration::ZERO,
                p95_latency: Duration::ZERO,
                ops_per_second: 0.0,
                error_count,
            };
        }

        latencies.sort();
        let operations_count = latencies.len();
        let total_duration: Duration = latencies.iter().sum();
        let avg_latency = total_duration / operations_count as u32;
        let min_latency = latencies[0];
        let max_latency = latencies[latencies.len() - 1];

        // Calculate 95th percentile
        let p95_index = (operations_count as f64 * 0.95) as usize;
        let p95_index = p95_index.min(latencies.len() - 1);
        let p95_latency = latencies[p95_index];

        let ops_per_second = operations_count as f64 / total_duration.as_secs_f64();

        OperationBenchmark {
            operations_count,
            total_duration,
            avg_latency,
            min_latency,
            max_latency,
            p95_latency,
            ops_per_second,
            error_count,
        }
    }

    /// Collect memory usage statistics
    async fn collect_memory_stats(&self) -> Result<MemoryStats> {
        // Get current storage stats as proxy for memory usage
        let stats = self.storage.stats(&self.context).await?;

        // Estimate memory usage (simplified)
        let estimated_memory_bytes = (stats.document_count + stats.chunk_count) * 1024; // Rough estimate

        Ok(MemoryStats {
            peak_memory_bytes: estimated_memory_bytes as u64,
            avg_memory_bytes: estimated_memory_bytes as u64,
            memory_efficiency: stats.document_count as f64 / estimated_memory_bytes as f64,
        })
    }

    /// Generate test documents
    fn generate_test_documents(&self, count: usize) -> Result<Vec<Document>> {
        let mut documents = Vec::new();

        for i in 0..count {
            let source = Source {
                source_type: SourceType::Local,
                url: None,
                path: Some(format!("/test/doc_{}.md", i)),
                arxiv_id: None,
                github_repo: None,
                retrieved_at: Utc::now(),
                version: None,
            };

            let doc = Document::new(DocumentType::Note, source)
                .with_content(format!("Test document content {}", i));

            documents.push(doc);
        }

        Ok(documents)
    }

    /// Generate test embeddings
    fn generate_test_embeddings(
        &self,
        count: usize,
        vector_size: usize,
    ) -> Result<Vec<(Uuid, Vec<f32>)>> {
        let mut embeddings = Vec::new();

        for _ in 0..count {
            let chunk_id = Uuid::new_v4();
            let embedding = (0..vector_size).map(|i| (i as f32 * 0.1).sin()).collect();
            embeddings.push((chunk_id, embedding));
        }

        Ok(embeddings)
    }
}

/// Run storage benchmarks with default configuration
pub async fn run_storage_benchmarks(
    storage: Storage,
    document_count: usize,
    embedding_count: usize,
    vector_size: usize,
) -> Result<StorageBenchmarkResults> {
    let benchmarker =
        StorageBenchmarker::new(storage, "benchmark_user".to_string(), AccessLevel::Admin);

    benchmarker
        .run_comprehensive_benchmark(document_count, embedding_count, vector_size)
        .await
}
