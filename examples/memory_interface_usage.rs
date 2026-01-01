//! Example: Using the Memory Interface Trait
//!
//! This example demonstrates how reasonkit-core can interface with reasonkit-mem
//! through the MemoryService trait.
//!
//! To run: `cargo run --example memory_interface_usage --features memory`

#[cfg(feature = "memory")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use reasonkit::memory_interface::*;
    use std::sync::Arc;

    println!("=== ReasonKit Memory Interface Example ===\n");

    // In a real application, you would get an implementation from reasonkit-mem
    // For now, we just show the trait definition and usage patterns
    println!("MemoryService trait provides:");
    println!("  1. Document Storage:");
    println!("     - store_document(&doc) -> Uuid");
    println!("     - store_documents(&docs) -> Vec<Uuid>");
    println!("     - get_document(id) -> Option<Document>");
    println!("     - delete_document(id) -> ()");
    println!("     - list_documents() -> Vec<Uuid>\n");

    println!("  2. Search & Retrieval:");
    println!("     - search(query, top_k) -> Vec<SearchResult>");
    println!("     - search_with_config(query, config) -> Vec<SearchResult>");
    println!("     - search_by_vector(embedding, top_k) -> Vec<SearchResult>");
    println!("     - search_by_keywords(query, top_k) -> Vec<SearchResult>\n");

    println!("  3. Context Assembly:");
    println!("     - get_context(query, top_k) -> ContextWindow");
    println!("     - get_context_with_config(query, config) -> ContextWindow");
    println!("     - get_document_chunks(doc_id) -> Vec<Chunk>\n");

    println!("  4. Embeddings:");
    println!("     - embed(text) -> Vec<f32>");
    println!("     - embed_batch(texts) -> Vec<Vec<f32>>\n");

    println!("  5. Indexing:");
    println!("     - build_indexes() -> ()");
    println!("     - rebuild_indexes() -> ()");
    println!("     - check_index_health() -> IndexStats\n");

    println!("  6. Stats & Health:");
    println!("     - stats() -> MemoryStats");
    println!("     - is_healthy() -> bool\n");

    // Example usage patterns (pseudo-code)
    println!("Example Usage Patterns:\n");

    println!("1. Storing Documents:");
    println!("   let doc_id = memory.store_document(&document).await?;");
    println!("   let ids = memory.store_documents(&documents).await?;\n");

    println!("2. Searching:");
    println!("   let results = memory.search(\"query\", 10).await?;");
    println!("   for result in results {{");
    println!("       println!(\"Score: {{}}, Text: {{}}\", result.score, result.chunk.text);");
    println!("   }}\n");

    println!("3. Getting Context for LLM:");
    println!("   let context = memory.get_context(\"What is RAG?\", 5).await?;");
    println!("   println!(\"Found {{}} chunks, {{}} tokens\", ");
    println!("       context.chunks.len(), context.token_count);\n");

    println!("4. Advanced Search:");
    println!("   let config = ContextConfig {{");
    println!("       top_k: 20,");
    println!("       min_score: 0.5,");
    println!("       alpha: 0.7,  // Hybrid search");
    println!("       use_raptor: true,  // Use hierarchical retrieval");
    println!("       rerank: true,  // Cross-encoder reranking");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   let context = memory.get_context_with_config(query, &config).await?;\n");

    println!("5. Monitoring:");
    println!("   let stats = memory.stats().await?;");
    println!("   println!(\"Documents: {{}}, Chunks: {{}}\", ");
    println!("       stats.document_count, stats.chunk_count);\n");

    println!("\n=== Design Highlights ===\n");
    println!("✓ Async-first: All operations use tokio::task::spawn");
    println!("✓ Trait-based: Multiple implementations possible (Qdrant, file, in-memory)");
    println!("✓ Result-oriented: All operations return Result<T> with proper error handling");
    println!("✓ Batch operations: Efficient bulk document and embedding processing");
    println!("✓ Hybrid search: Dense (semantic) + Sparse (keyword) with RRF fusion");
    println!("✓ Advanced retrieval: RAPTOR trees, cross-encoder reranking optional");
    println!("✓ Context quality: Built-in diversity, coverage, and relevance metrics");
    println!("✓ Type safe: Full type checking with serde serialization support\n");

    println!("=== Feature Flags ===\n");
    println!("This module always exists in reasonkit-core");
    println!("But requires 'memory' feature to use with reasonkit-mem");
    println!("Without the feature, you can still define custom implementations");

    Ok(())
}

#[cfg(not(feature = "memory"))]
fn main() {
    println!("This example requires the 'memory' feature.");
    println!("Run with: cargo run --example memory_interface_usage --features memory");
}
