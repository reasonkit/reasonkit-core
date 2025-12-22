//! Comprehensive unit tests for the storage module
//!
//! Tests cover:
//! - Connection pooling functionality
//! - Security controls and access management
//! - Embedding cache operations
//! - Storage backend implementations
//! - Error handling and edge cases

use reasonkit_core::storage::*;
use reasonkit_core::{Document, DocumentType, Source, SourceType};
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[cfg(test)]
mod tests {
    use super::*;

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

        Document::new(DocumentType::Note, source)
            .with_content("Test content".to_string())
    }

    // Helper function to create a test access context
    fn create_test_context(user_id: &str, level: AccessLevel, operation: &str) -> AccessContext {
        AccessContext::new(
            user_id.to_string(),
            level,
            operation.to_string(),
        )
    }

    mod connection_pool_tests {
        use super::*;
        use qdrant_client::prelude::*;

        #[tokio::test]
        async fn test_connection_pool_creation() {
            let config = QdrantClientConfig::from_url("http://localhost:6333");
            let pool_config = QdrantConnectionConfig::default();

            let pool = QdrantConnectionPool::new(config, pool_config);
            assert_eq!(pool.connections.len(), 0);
            assert_eq!(pool.config.max_connections, 10);
        }

        #[tokio::test]
        async fn test_connection_pool_max_connections() {
            let config = QdrantClientConfig::from_url("http://localhost:6333");
            let mut pool_config = QdrantConnectionConfig::default();
            pool_config.max_connections = 2;

            let mut pool = QdrantConnectionPool::new(config, pool_config);

            // This test would require a running Qdrant instance
            // For now, we test the logic without actual connections
            assert_eq!(pool.connections.len(), 0);
        }

        #[tokio::test]
        async fn test_connection_pool_cleanup() {
            let config = QdrantClientConfig::from_url("http://localhost:6333");
            let pool_config = QdrantConnectionConfig::default();

            let mut pool = QdrantConnectionPool::new(config, pool_config);
            pool.cleanup_expired();

            // Should not panic and connections should remain empty
            assert_eq!(pool.connections.len(), 0);
        }
    }

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
    }

    mod embedding_cache_tests {
        use super::*;
        use std::time::Duration;

        #[test]
        fn test_embedding_cache_creation() {
            let config = EmbeddingCacheConfig {
                max_size: 100,
                ttl_secs: 3600,
            };

            let cache = EmbeddingCache::new(config);
            assert_eq!(cache.cache.len(), 0);
            assert_eq!(cache.access_order.len(), 0);
            assert_eq!(cache.config.max_size, 100);
            assert_eq!(cache.config.ttl_secs, 3600);
        }

        #[test]
        fn test_embedding_cache_put_and_get() {
            let config = EmbeddingCacheConfig {
                max_size: 10,
                ttl_secs: 3600,
            };

            let mut cache = EmbeddingCache::new(config);
            let chunk_id = Uuid::new_v4();
            let embedding = vec![1.0, 2.0, 3.0];

            // Put embedding
            cache.put(chunk_id, embedding.clone());

            // Get embedding
            let retrieved = cache.get(&chunk_id);
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap(), embedding.as_slice());
        }

        #[test]
        fn test_embedding_cache_expiration() {
            let config = EmbeddingCacheConfig {
                max_size: 10,
                ttl_secs: 0, // Expire immediately
            };

            let mut cache = EmbeddingCache::new(config);
            let chunk_id = Uuid::new_v4();
            let embedding = vec![1.0, 2.0, 3.0];

            // Put embedding
            cache.put(chunk_id, embedding);

            // Get embedding (should be expired)
            let retrieved = cache.get(&chunk_id);
            assert!(retrieved.is_none());
        }

        #[test]
        fn test_embedding_cache_eviction() {
            let config = EmbeddingCacheConfig {
                max_size: 2,
                ttl_secs: 3600,
            };

            let mut cache = EmbeddingCache::new(config);

            // Fill cache to capacity
            let id1 = Uuid::new_v4();
            let id2 = Uuid::new_v4();
            let id3 = Uuid::new_v4();

            cache.put(id1, vec![1.0]);
            cache.put(id2, vec![2.0]);
            cache.put(id3, vec![3.0]); // Should evict id1

            // Check that id1 is evicted
            assert!(cache.get(&id1).is_none());
            assert!(cache.get(&id2).is_some());
            assert!(cache.get(&id3).is_some());
        }

        #[test]
        fn test_embedding_cache_cleanup() {
            let config = EmbeddingCacheConfig {
                max_size: 10,
                ttl_secs: 0, // Expire immediately
            };

            let mut cache = EmbeddingCache::new(config);
            let chunk_id = Uuid::new_v4();

            cache.put(chunk_id, vec![1.0, 2.0, 3.0]);
            cache.cleanup_expired();

            // Should be cleaned up
            assert!(cache.get(&chunk_id).is_none());
        }
    }

    mod storage_backend_tests {
        use super::*;
        use tempfile::TempDir;

        #[tokio::test]
        async fn test_in_memory_storage_operations() {
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
            storage.store_embeddings(&chunk_id, &embeddings, &context).await.unwrap();

            // Test get embeddings
            let retrieved = storage.get_embeddings(&chunk_id, &context).await.unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap(), embeddings);
        }

        #[tokio::test]
        async fn test_in_memory_storage_search() {
            let storage = InMemoryStorage::new();
            let context = create_test_context("test_user", AccessLevel::Admin, "test");

            // Store some embeddings
            let id1 = Uuid::new_v4();
            let id2 = Uuid::new_v4();
            let emb1 = vec![1.0, 0.0, 0.0]; // Unit vector along x-axis
            let emb2 = vec![0.0, 1.0, 0.0]; // Unit vector along y-axis

            storage.store_embeddings(&id1, &emb1, &context).await.unwrap();
            storage.store_embeddings(&id2, &emb2, &context).await.unwrap();

            // Search with query similar to emb1
            let query = vec![0.9, 0.1, 0.0];
            let results = storage.search_by_vector(&query, 2, &context).await.unwrap();

            assert_eq!(results.len(), 2);
            // First result should be id1 (more similar to query)
            assert_eq!(results[0].0, id1);
            assert!(results[0].1 > results[1].1); // Higher similarity score
        }

        #[tokio::test]
        async fn test_file_storage_operations() {
            let temp_dir = TempDir::new().unwrap();
            let storage = FileStorage::new(temp_dir.path().to_path_buf()).await.unwrap();
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
            let storage = FileStorage::new(temp_dir.path().to_path_buf()).await.unwrap();
            let context = create_test_context("test_user", AccessLevel::Admin, "test");
            let chunk_id = Uuid::new_v4();
            let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0];

            // Test store embeddings
            storage.store_embeddings(&chunk_id, &embeddings, &context).await.unwrap();

            // Test get embeddings
            let retrieved = storage.get_embeddings(&chunk_id, &context).await.unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap(), embeddings);
        }
    }

    mod storage_manager_tests {
        use super::*;
        use tempfile::TempDir;

        #[tokio::test]
        async fn test_storage_in_memory() {
            let storage = Storage::in_memory();
            let context = create_test_context("test_user", AccessLevel::Admin, "test");
            let doc = create_test_document();

            // Test basic operations
            storage.store_document(&doc, &context).await.unwrap();
            let retrieved = storage.get_document(&doc.id, &context).await.unwrap();
            assert!(retrieved.is_some());
        }

        #[tokio::test]
        async fn test_storage_file() {
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
        async fn test_storage_stats() {
            let storage = Storage::in_memory();
            let context = create_test_context("test_user", AccessLevel::Admin, "test");
            let doc = create_test_document();

            // Store document
            storage.store_document(&doc, &context).await.unwrap();

            // Store embeddings
            let chunk_id = Uuid::new_v4();
            let embeddings = vec![1.0, 2.0, 3.0];
            storage.store_embeddings(&chunk_id, &embeddings, &context).await.unwrap();

            // Get stats
            let stats = storage.stats(&context).await.unwrap();
            assert_eq!(stats.document_count, 1);
            assert_eq!(stats.embedding_count, 1);
        }
    }

    mod error_handling_tests {
        use super::*;

        #[tokio::test]
        async fn test_access_denied_error() {
            let storage = InMemoryStorage::new();
            let doc = create_test_document();

            // Try to store with insufficient permissions
            let context = create_test_context("user", AccessLevel::Read, "store_doc");

            let result = storage.store_document(&doc, &context).await;
            assert!(result.is_err());

            if let Err(crate::Error::Validation(msg)) = result {
                assert!(msg.contains("Access denied"));
            } else {
                panic!("Expected validation error");
            }
        }

        #[tokio::test]
        async fn test_embedding_size_validation() {
            let storage = InMemoryStorage::new();
            let context = create_test_context("user", AccessLevel::Admin, "store_embeddings");
            let chunk_id = Uuid::new_v4();

            // This test is for QdrantStorage which validates vector size
            // For InMemoryStorage, this should work fine
            let embeddings = vec![1.0, 2.0, 3.0];
            let result = storage.store_embeddings(&chunk_id, &embeddings, &context).await;
            assert!(result.is_ok());
        }
    }

    mod utility_tests {
        use super::*;

        #[test]
        fn test_cosine_similarity_edge_cases() {
            // Test with zero vectors
            let zero_vec = vec![0.0, 0.0, 0.0];
            let normal_vec = vec![1.0, 2.0, 3.0];

            assert_eq!(cosine_similarity(&zero_vec, &normal_vec), 0.0);
            assert_eq!(cosine_similarity(&zero_vec, &zero_vec), 0.0);

            // Test with identical vectors
            let vec1 = vec![1.0, 2.0, 3.0];
            let vec2 = vec![1.0, 2.0, 3.0];
            assert!((cosine_similarity(&vec1, &vec2) - 1.0).abs() < 0.001);

            // Test with orthogonal vectors
            let vec3 = vec![1.0, 0.0, 0.0];
            let vec4 = vec![0.0, 1.0, 0.0];
            assert!((cosine_similarity(&vec3, &vec4) - 0.0).abs() < 0.001);
        }

        #[test]
        fn test_uuid_point_conversion() {
            // This would test QdrantStorage::point_id_from_uuid and uuid_from_point_id
            // but those methods are private. We can test through public interface.
            let uuid = Uuid::new_v4();

            // Test that we can convert back and forth conceptually
            // (actual implementation uses u128 -> u64 conversion)
            let point_id = QdrantStorage::point_id_from_uuid(&uuid);
            match point_id {
                qdrant_client::qdrant::PointId::Num(_) => {
                    // Valid conversion
                }
                _ => panic!("Expected Num point ID"),
            }
        }
    }
}