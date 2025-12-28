//! Comprehensive unit tests for the storage module
//!
//! Tests cover:
//! - Security controls and access management
//! - Storage backend implementations (InMemoryStorage, FileStorage)
//! - Storage manager operations
//! - Error handling and edge cases
//!
//! Note: Tests for internal types (QdrantConnectionPool, EmbeddingCache internals)
//! have been moved to in-module tests in storage/mod.rs.

#![cfg(feature = "memory")]

use chrono::Utc;
use reasonkit::embedding::cosine_similarity;
use reasonkit::storage::{
    AccessContext, AccessControlConfig, AccessLevel, EmbeddingCacheConfig, FileStorage,
    InMemoryStorage, QdrantConnectionConfig, QdrantSecurityConfig, Storage, StorageBackend,
};
use reasonkit::{Document, DocumentType, Source, SourceType};
use tempfile::TempDir;
use uuid::Uuid;

// Helper function to create a test document
fn create_test_document() -> Document {
    let source = Source {
        source_type: SourceType::Local,
        url: None,
        path: Some("/test/doc.md".to_string()),
        arxiv_id: None,
        github_repo: None,
        retrieved_at: Utc::now(),
        version: None,
    };

    Document::new(DocumentType::Note, source).with_content("Test content".to_string())
}

// Helper function to create a test access context
fn create_test_context(user_id: &str, level: AccessLevel, operation: &str) -> AccessContext {
    AccessContext::new(user_id.to_string(), level, operation.to_string())
}

#[cfg(test)]
mod security_tests {
    use super::*;

    #[test]
    fn test_access_context_creation() {
        let context = create_test_context("user1", AccessLevel::Read, "read_operation");

        assert_eq!(context.user_id, "user1");
        assert_eq!(context.access_level, AccessLevel::Read);
        assert_eq!(context.operation, "read_operation");
    }

    #[test]
    fn test_access_level_hierarchy() {
        let config = AccessControlConfig::default();

        // Test Read level permissions
        let read_context = create_test_context("user", AccessLevel::Read, "test");
        assert!(read_context.has_permission(&AccessLevel::Read, &config));
        assert!(!read_context.has_permission(&AccessLevel::ReadWrite, &config));
        assert!(!read_context.has_permission(&AccessLevel::Admin, &config));

        // Test ReadWrite level permissions
        let write_context = create_test_context("user", AccessLevel::ReadWrite, "test");
        assert!(write_context.has_permission(&AccessLevel::Read, &config));
        assert!(write_context.has_permission(&AccessLevel::ReadWrite, &config));
        assert!(!write_context.has_permission(&AccessLevel::Admin, &config));

        // Test Admin level permissions
        let admin_context = create_test_context("user", AccessLevel::Admin, "test");
        assert!(admin_context.has_permission(&AccessLevel::Read, &config));
        assert!(admin_context.has_permission(&AccessLevel::ReadWrite, &config));
        assert!(admin_context.has_permission(&AccessLevel::Admin, &config));
    }

    #[test]
    fn test_access_control_config_defaults() {
        let config = AccessControlConfig::default();

        assert_eq!(config.read_level, AccessLevel::Read);
        assert_eq!(config.write_level, AccessLevel::ReadWrite);
        assert_eq!(config.delete_level, AccessLevel::ReadWrite);
        assert_eq!(config.admin_level, AccessLevel::Admin);
        assert!(config.enable_audit_log);
    }

    #[test]
    fn test_security_config_defaults() {
        let config = QdrantSecurityConfig::default();

        assert!(config.api_key.is_none());
        assert!(config.tls_enabled);
        assert!(config.ca_cert_path.is_none());
        assert!(config.client_cert_path.is_none());
        assert!(config.client_key_path.is_none());
        assert!(!config.skip_tls_verify);
    }

    #[test]
    fn test_connection_config_defaults() {
        let config = QdrantConnectionConfig::default();

        assert_eq!(config.max_connections, 10);
        assert_eq!(config.connect_timeout_secs, 30);
        assert_eq!(config.request_timeout_secs, 60);
        assert_eq!(config.health_check_interval_secs, 300);
        assert_eq!(config.max_idle_secs, 600);
    }

    #[test]
    fn test_embedding_cache_config_defaults() {
        let config = EmbeddingCacheConfig::default();

        assert_eq!(config.max_size, 10000);
        assert_eq!(config.ttl_secs, 3600);
    }
}

#[cfg(test)]
mod storage_backend_tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_storage_document_operations() {
        let storage = InMemoryStorage::new();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");
        let doc = create_test_document();

        // Test store
        storage.store_document(&doc, &context).await.unwrap();

        // Test get
        let retrieved = storage.get_document(&doc.id, &context).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content.raw, "Test content");

        // Test list
        let docs = storage.list_documents(&context).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0], doc.id);

        // Test delete
        storage.delete_document(&doc.id, &context).await.unwrap();
        let retrieved_after_delete = storage.get_document(&doc.id, &context).await.unwrap();
        assert!(retrieved_after_delete.is_none());
    }

    #[tokio::test]
    async fn test_in_memory_storage_embeddings() {
        let storage = InMemoryStorage::new();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");
        let chunk_id = Uuid::new_v4();
        let embeddings = vec![1.0, 2.0, 3.0];

        // Test store embeddings
        storage
            .store_embeddings(&chunk_id, &embeddings, &context)
            .await
            .unwrap();

        // Test get embeddings
        let retrieved = storage.get_embeddings(&chunk_id, &context).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), embeddings);
    }

    #[tokio::test]
    async fn test_in_memory_storage_vector_search() {
        let storage = InMemoryStorage::new();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");

        // Store some embeddings
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let emb1 = vec![1.0, 0.0, 0.0]; // Unit vector along x-axis
        let emb2 = vec![0.0, 1.0, 0.0]; // Unit vector along y-axis

        storage
            .store_embeddings(&id1, &emb1, &context)
            .await
            .unwrap();
        storage
            .store_embeddings(&id2, &emb2, &context)
            .await
            .unwrap();

        // Search with query similar to emb1
        let query = vec![0.9, 0.1, 0.0];
        let results = storage.search_by_vector(&query, 2, &context).await.unwrap();

        assert_eq!(results.len(), 2);
        // First result should be id1 (more similar to query)
        assert_eq!(results[0].0, id1);
        assert!(results[0].1 > results[1].1); // Higher similarity score
    }

    #[tokio::test]
    async fn test_file_storage_document_operations() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");
        let doc = create_test_document();

        // Test store
        storage.store_document(&doc, &context).await.unwrap();

        // Test get
        let retrieved = storage.get_document(&doc.id, &context).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content.raw, "Test content");

        // Test list
        let docs = storage.list_documents(&context).await.unwrap();
        assert_eq!(docs.len(), 1);

        // Test delete
        storage.delete_document(&doc.id, &context).await.unwrap();
        let retrieved_after_delete = storage.get_document(&doc.id, &context).await.unwrap();
        assert!(retrieved_after_delete.is_none());
    }

    #[tokio::test]
    async fn test_file_storage_embeddings() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");
        let chunk_id = Uuid::new_v4();
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test store embeddings
        storage
            .store_embeddings(&chunk_id, &embeddings, &context)
            .await
            .unwrap();

        // Test get embeddings
        let retrieved = storage.get_embeddings(&chunk_id, &context).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), embeddings);
    }

    #[tokio::test]
    async fn test_file_storage_vector_search() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path().to_path_buf())
            .await
            .unwrap();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");

        // Store embeddings
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0];

        storage
            .store_embeddings(&id1, &emb1, &context)
            .await
            .unwrap();
        storage
            .store_embeddings(&id2, &emb2, &context)
            .await
            .unwrap();

        // Search
        let query = vec![0.9, 0.1, 0.0];
        let results = storage.search_by_vector(&query, 2, &context).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id1); // More similar to query
    }

    #[tokio::test]
    async fn test_in_memory_storage_stats() {
        let storage = InMemoryStorage::new();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");

        // Initial stats
        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 0);
        assert_eq!(stats.embedding_count, 0);

        // Add document
        let doc = create_test_document();
        storage.store_document(&doc, &context).await.unwrap();

        // Add embedding
        let chunk_id = Uuid::new_v4();
        storage
            .store_embeddings(&chunk_id, &[1.0, 2.0, 3.0], &context)
            .await
            .unwrap();

        // Check stats
        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.embedding_count, 1);
    }
}

#[cfg(test)]
mod storage_manager_tests {
    use super::*;

    #[tokio::test]
    async fn test_storage_in_memory_facade() {
        let storage = Storage::in_memory();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");
        let doc = create_test_document();

        // Test basic operations through facade
        storage.store_document(&doc, &context).await.unwrap();
        let retrieved = storage.get_document(&doc.id, &context).await.unwrap();
        assert!(retrieved.is_some());

        // Test list
        let docs = storage.list_documents(&context).await.unwrap();
        assert_eq!(docs.len(), 1);

        // Test delete
        storage.delete_document(&doc.id, &context).await.unwrap();
        let docs_after = storage.list_documents(&context).await.unwrap();
        assert_eq!(docs_after.len(), 0);
    }

    #[tokio::test]
    async fn test_storage_file_facade() {
        let temp_dir = TempDir::new().unwrap();
        let storage = Storage::file(temp_dir.path().to_path_buf()).await.unwrap();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");
        let doc = create_test_document();

        // Test basic operations
        storage.store_document(&doc, &context).await.unwrap();
        let retrieved = storage.get_document(&doc.id, &context).await.unwrap();
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_storage_embeddings_through_facade() {
        let storage = Storage::in_memory();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");
        let chunk_id = Uuid::new_v4();
        let embeddings = vec![1.0, 2.0, 3.0];

        // Store and retrieve through facade
        storage
            .store_embeddings(&chunk_id, &embeddings, &context)
            .await
            .unwrap();

        let retrieved = storage.get_embeddings(&chunk_id, &context).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), embeddings);
    }

    #[tokio::test]
    async fn test_storage_vector_search_through_facade() {
        let storage = Storage::in_memory();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        storage
            .store_embeddings(&id1, &[1.0, 0.0, 0.0], &context)
            .await
            .unwrap();
        storage
            .store_embeddings(&id2, &[0.0, 1.0, 0.0], &context)
            .await
            .unwrap();

        let results = storage
            .search_by_vector(&[0.9, 0.1, 0.0], 2, &context)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id1);
    }

    #[tokio::test]
    async fn test_storage_stats_through_facade() {
        let storage = Storage::in_memory();
        let context = create_test_context("test_user", AccessLevel::Admin, "test");
        let doc = create_test_document();

        storage.store_document(&doc, &context).await.unwrap();

        let chunk_id = Uuid::new_v4();
        storage
            .store_embeddings(&chunk_id, &[1.0, 2.0, 3.0], &context)
            .await
            .unwrap();

        let stats = storage.stats(&context).await.unwrap();
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.embedding_count, 1);
    }
}

#[cfg(test)]
mod utility_tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![-1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let zero_vec = vec![0.0, 0.0, 0.0];
        let normal_vec = vec![1.0, 2.0, 3.0];

        // Similarity with zero vector should be 0
        assert_eq!(cosine_similarity(&zero_vec, &normal_vec), 0.0);
        assert_eq!(cosine_similarity(&zero_vec, &zero_vec), 0.0);
    }

    #[test]
    fn test_cosine_similarity_scaling_invariance() {
        // Cosine similarity should be scale-invariant
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![2.0, 4.0, 6.0]; // Same direction, different magnitude

        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 0.001);
    }
}
