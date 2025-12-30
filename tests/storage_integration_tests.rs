//! Integration tests for the storage module
//!
//! These tests verify end-to-end functionality across storage backends
//! and test realistic usage patterns.

#![cfg(feature = "memory")]

use chrono::Utc;
use reasonkit::storage::{AccessContext, AccessLevel, Storage};
use reasonkit::{Document, DocumentType, Source, SourceType};
use std::time::Instant;
use uuid::Uuid;

// Helper functions
fn create_test_document(content: &str) -> reasonkit_mem::Document {
    let source = Source {
        source_type: SourceType::Local,
        url: None,
        path: Some(format!("/test/{}.md", content)),
        arxiv_id: None,
        github_repo: None,
        retrieved_at: Utc::now(),
        version: None,
    };

    let doc = Document::new(DocumentType::Note, source).with_content(content.to_string());
    reasonkit_mem::Document::from(doc)
}

fn create_test_context(user_id: &str, level: AccessLevel) -> AccessContext {
    AccessContext::new(user_id.to_string(), level, "integration_test".to_string())
}

#[cfg(test)]
mod end_to_end_tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_full_workflow() {
        let storage = Storage::in_memory();
        let context = create_test_context("test_user", AccessLevel::Admin);

        // Create and store multiple documents
        let docs = vec![
            create_test_document("Document 1 content"),
            create_test_document("Document 2 content"),
            create_test_document("Document 3 content"),
        ];

        // Store all documents
        for doc in &docs {
            storage.store_document(doc, &context).await.unwrap();
        }

        // Verify all documents are stored
        let all_docs = storage.list_documents(&context).await.unwrap();
        assert_eq!(all_docs.len(), 3);

        // Retrieve each document
        for doc in &docs {
            let retrieved = storage.get_document(&doc.id, &context).await.unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().content.raw, doc.content.raw);
        }

        // Store embeddings for each document
        let mut chunk_ids = Vec::new();
        for (i, _doc) in docs.iter().enumerate() {
            let chunk_id = Uuid::new_v4();
            chunk_ids.push(chunk_id);

            // Create simple embeddings based on document index
            let embeddings = vec![i as f32 * 0.1, (i + 1) as f32 * 0.1, (i + 2) as f32 * 0.1];
            storage
                .store_embeddings(&chunk_id, &embeddings, &context)
                .await
                .unwrap();
        }

        // Test vector search
        let query_embedding = vec![0.05, 0.15, 0.25]; // Similar to first document
        let search_results = storage
            .search_by_vector(&query_embedding, 2, &context)
            .await
            .unwrap();

        assert!(!search_results.is_empty());
        assert!(search_results[0].1 > search_results[1].1); // First result more similar

        // Test stats
        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 3);
        assert_eq!(stats.embedding_count, 3);

        // Clean up - delete all documents
        for doc in &docs {
            storage.delete_document(&doc.id, &context).await.unwrap();
        }

        // Verify cleanup
        let final_docs = storage.list_documents(&context).await.unwrap();
        assert_eq!(final_docs.len(), 0);
    }

    #[tokio::test]
    async fn test_file_storage_full_workflow() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let storage = Storage::file(temp_dir.path().to_path_buf()).await.unwrap();
        let context = create_test_context("test_user", AccessLevel::Admin);

        // Create test documents with different content
        let docs = vec![
            create_test_document("File storage test 1"),
            create_test_document("File storage test 2"),
        ];

        // Store documents
        for doc in &docs {
            storage.store_document(doc, &context).await.unwrap();
        }

        // Store embeddings
        let chunk_id = Uuid::new_v4();
        let embeddings = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        storage
            .store_embeddings(&chunk_id, &embeddings, &context)
            .await
            .unwrap();

        // Retrieve and verify
        for doc in &docs {
            let retrieved = storage.get_document(&doc.id, &context).await.unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().content.raw, doc.content.raw);
        }

        let retrieved_embeddings = storage.get_embeddings(&chunk_id, &context).await.unwrap();
        assert!(retrieved_embeddings.is_some());
        assert_eq!(retrieved_embeddings.unwrap(), embeddings);

        // Test search
        let query = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let results = storage.search_by_vector(&query, 1, &context).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, chunk_id);

        // Test stats
        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 2);
        assert_eq!(stats.embedding_count, 1);
        assert!(stats.size_bytes > 0); // Should have some size
    }

    #[tokio::test]
    async fn test_storage_load_simulation() {
        let storage = Storage::in_memory();
        let context = create_test_context("load_test_user", AccessLevel::Admin);

        // Simulate a realistic load pattern
        let mut doc_ids = vec![];

        // Phase 1: Initial bulk load
        for i in 0..50 {
            let doc = create_test_document(&format!("Bulk load document {}", i));
            storage.store_document(&doc, &context).await.unwrap();
            doc_ids.push(doc.id);
        }

        // Phase 2: Add embeddings for some documents
        for (i, &_doc_id) in doc_ids.iter().enumerate() {
            if i % 2 == 0 {
                // Add embeddings to even-indexed documents
                let chunk_id = Uuid::new_v4();
                let embeddings = vec![i as f32 * 0.01; 128]; // 128-dimensional embeddings
                storage
                    .store_embeddings(&chunk_id, &embeddings, &context)
                    .await
                    .unwrap();
            }
        }

        // Phase 3: Simulate search workload
        for _ in 0..20 {
            let query = vec![0.5; 128]; // Neutral query
            let results = storage.search_by_vector(&query, 5, &context).await.unwrap();
            assert!(results.len() <= 5);
        }

        // Phase 4: Random access pattern
        for &doc_id in doc_ids.iter().step_by(3) {
            // Every 3rd document
            let retrieved = storage.get_document(&doc_id, &context).await.unwrap();
            assert!(retrieved.is_some());
        }

        // Phase 5: Bulk deletion
        for &doc_id in doc_ids.iter().step_by(2) {
            // Delete every 2nd document
            storage.delete_document(&doc_id, &context).await.unwrap();
        }

        // Final verification
        let remaining_docs = storage.list_documents(&context).await.unwrap();
        assert_eq!(remaining_docs.len(), 25); // Half should remain

        let final_stats = storage.stats(&context).await.unwrap();
        assert_eq!(final_stats.document_count, 25);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_storage_performance_baseline() {
        let storage = Storage::in_memory();
        let context = create_test_context("perf_test_user", AccessLevel::Admin);

        // Measure document storage performance
        let mut docs = vec![];
        for i in 0..100 {
            docs.push(create_test_document(&format!("Performance test doc {}", i)));
        }

        let start = Instant::now();
        for doc in &docs {
            storage.store_document(doc, &context).await.unwrap();
        }
        let store_duration = start.elapsed();

        // Measure retrieval performance
        let start = Instant::now();
        for doc in &docs {
            let _ = storage.get_document(&doc.id, &context).await.unwrap();
        }
        let retrieve_duration = start.elapsed();

        // Basic performance assertions (these are baseline expectations)
        assert!(
            store_duration.as_millis() < 1000,
            "Storage took too long: {:?}",
            store_duration
        );
        assert!(
            retrieve_duration.as_millis() < 500,
            "Retrieval took too long: {:?}",
            retrieve_duration
        );

        println!(
            "Storage performance - Store: {:?}, Retrieve: {:?}",
            store_duration, retrieve_duration
        );
    }

    #[tokio::test]
    async fn test_embedding_performance() {
        let storage = Storage::in_memory();
        let context = create_test_context("perf_test_user", AccessLevel::Admin);

        // Create embeddings of various sizes
        let sizes = vec![64, 128, 256, 512];
        let mut chunk_ids = vec![];

        for &size in &sizes {
            let chunk_id = Uuid::new_v4();
            chunk_ids.push(chunk_id);

            let embeddings = vec![0.1; size];
            storage
                .store_embeddings(&chunk_id, &embeddings, &context)
                .await
                .unwrap();
        }

        // Test search performance
        let query = vec![0.1; 512]; // Match largest embedding size

        let start = Instant::now();
        for _ in 0..50 {
            let _ = storage.search_by_vector(&query, 5, &context).await.unwrap();
        }
        let search_duration = start.elapsed();

        assert!(
            search_duration.as_millis() < 200,
            "Search took too long: {:?}",
            search_duration
        );

        println!(
            "Embedding search performance: {:?} for 50 searches",
            search_duration
        );
    }
}

#[cfg(test)]
mod reliability_tests {
    use super::*;

    #[tokio::test]
    async fn test_storage_error_recovery() {
        let storage = Storage::in_memory();
        let context = create_test_context("reliability_test_user", AccessLevel::Admin);

        // Test with valid data first
        let doc = create_test_document("Valid document");

        // Store valid document
        storage.store_document(&doc, &context).await.unwrap();

        // Test retrieval of non-existent document
        let non_existent_id = Uuid::new_v4();
        let result = storage.get_document(&non_existent_id, &context).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test deletion of non-existent document
        let delete_result = storage.delete_document(&non_existent_id, &context).await;
        assert!(delete_result.is_ok()); // Should not error

        // Test retrieval of non-existent embeddings
        let embedding_result = storage.get_embeddings(&non_existent_id, &context).await;
        assert!(embedding_result.is_ok());
        assert!(embedding_result.unwrap().is_none());

        // Verify original document still exists
        let original = storage.get_document(&doc.id, &context).await.unwrap();
        assert!(original.is_some());
    }

    #[tokio::test]
    async fn test_storage_data_integrity() {
        let storage = Storage::in_memory();
        let context = create_test_context("integrity_test_user", AccessLevel::Admin);

        // Create document with specific content
        let original_content = "Original content with special characters: éñüñ";
        let mut doc = create_test_document(original_content);

        // Store document
        storage.store_document(&doc, &context).await.unwrap();

        // Modify the original document object (should not affect stored version)
        doc.content.raw = "Modified content".to_string();

        // Retrieve and verify integrity
        let retrieved = storage
            .get_document(&doc.id, &context)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retrieved.content.raw, original_content);

        // Test with binary-like content
        let binary_content = "Binary-like content: \x00\x01\x02\x03";
        let binary_doc = create_test_document(binary_content);

        storage.store_document(&binary_doc, &context).await.unwrap();
        let binary_retrieved = storage
            .get_document(&binary_doc.id, &context)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(binary_retrieved.content.raw, binary_content);
    }

    #[tokio::test]
    async fn test_storage_isolation() {
        // Test that different storage instances are isolated
        let storage1 = Storage::in_memory();
        let storage2 = Storage::in_memory();
        let context = create_test_context("isolation_test_user", AccessLevel::Admin);

        let doc1 = create_test_document("Document for storage 1");
        let doc2 = create_test_document("Document for storage 2");

        // Store in different storages
        storage1.store_document(&doc1, &context).await.unwrap();
        storage2.store_document(&doc2, &context).await.unwrap();

        // Verify isolation
        let retrieved1_from_1 = storage1.get_document(&doc1.id, &context).await.unwrap();
        let retrieved2_from_1 = storage1.get_document(&doc2.id, &context).await.unwrap();
        let retrieved1_from_2 = storage2.get_document(&doc1.id, &context).await.unwrap();
        let retrieved2_from_2 = storage2.get_document(&doc2.id, &context).await.unwrap();

        assert!(retrieved1_from_1.is_some());
        assert!(retrieved2_from_1.is_none()); // doc2 not in storage1
        assert!(retrieved1_from_2.is_none()); // doc1 not in storage2
        assert!(retrieved2_from_2.is_some());
    }
}

// Note: The concurrent access test was removed because Storage doesn't implement Clone.
// To test concurrency, you would need to wrap Storage in Arc<RwLock<Storage>> or
// use a shared storage implementation. The in-module tests in storage/mod.rs
// handle concurrent access testing with proper synchronization.
