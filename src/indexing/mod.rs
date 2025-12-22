//! Indexing module for ReasonKit Core
//!
//! Provides BM25 text indexing using Tantivy for sparse retrieval.

use crate::{Document, Result, Error};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tantivy::{
    collector::TopDocs,
    query::QueryParser,
    schema::{Schema, Field, STORED, TEXT, STRING, Value},
    Index, ReloadPolicy, TantivyDocument,
    directory::MmapDirectory,
};
use uuid::Uuid;

/// Index statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    /// Number of indexed documents
    pub document_count: usize,
    /// Number of indexed chunks
    pub chunk_count: usize,
    /// Index size in bytes
    pub size_bytes: u64,
    /// Last updated timestamp
    pub last_updated: Option<String>,
}

/// BM25 index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Memory budget for indexing (bytes)
    pub memory_budget: usize,
    /// Number of threads for indexing
    pub num_threads: usize,
    /// Whether to store the original text
    pub store_text: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            memory_budget: 50_000_000, // 50MB
            num_threads: 4,
            store_text: true,
        }
    }
}

/// Fields in the Tantivy index
struct IndexFields {
    doc_id: Field,
    chunk_id: Field,
    text: Field,
    section: Field,
}

/// BM25 text index using Tantivy
pub struct BM25Index {
    index: Index,
    fields: IndexFields,
    config: IndexConfig,
}

impl BM25Index {
    /// Create a new in-memory BM25 index
    pub fn in_memory() -> Result<Self> {
        let (schema, fields) = Self::build_schema();
        let index = Index::create_in_ram(schema);

        Ok(Self {
            index,
            fields,
            config: IndexConfig::default(),
        })
    }

    /// Create or open a BM25 index at the given path
    pub fn open(path: PathBuf) -> Result<Self> {
        let (schema, fields) = Self::build_schema();

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&path)
            .map_err(|e| Error::io(format!("Failed to create index directory: {}", e)))?;

        let dir = MmapDirectory::open(&path)
            .map_err(|e| Error::io(format!("Failed to open index directory: {}", e)))?;

        let index = Index::open_or_create(dir, schema)
            .map_err(|e| Error::indexing(format!("Failed to create index: {}", e)))?;

        Ok(Self {
            index,
            fields,
            config: IndexConfig::default(),
        })
    }

    /// Build the Tantivy schema
    fn build_schema() -> (Schema, IndexFields) {
        let mut schema_builder = Schema::builder();

        let doc_id = schema_builder.add_text_field("doc_id", STRING | STORED);
        let chunk_id = schema_builder.add_text_field("chunk_id", STRING | STORED);
        let text = schema_builder.add_text_field("text", TEXT | STORED);
        let section = schema_builder.add_text_field("section", TEXT | STORED);

        let schema = schema_builder.build();
        let fields = IndexFields {
            doc_id,
            chunk_id,
            text,
            section,
        };

        (schema, fields)
    }

    /// Index a document's chunks
    pub fn index_document(&self, doc: &Document) -> Result<usize> {
        let mut writer = self.index.writer(self.config.memory_budget)
            .map_err(|e| Error::indexing(format!("Failed to create index writer: {}", e)))?;

        let mut indexed = 0;
        for chunk in &doc.chunks {
            let mut tantivy_doc = TantivyDocument::new();
            tantivy_doc.add_text(self.fields.doc_id, doc.id.to_string());
            tantivy_doc.add_text(self.fields.chunk_id, chunk.id.to_string());
            tantivy_doc.add_text(self.fields.text, &chunk.text);
            if let Some(ref section) = chunk.section {
                tantivy_doc.add_text(self.fields.section, section);
            }

            writer.add_document(tantivy_doc)
                .map_err(|e| Error::indexing(format!("Failed to add document: {}", e)))?;
            indexed += 1;
        }

        writer.commit()
            .map_err(|e| Error::indexing(format!("Failed to commit index: {}", e)))?;

        Ok(indexed)
    }

    /// Index multiple documents
    pub fn index_documents(&self, docs: &[Document]) -> Result<usize> {
        let mut writer = self.index.writer(self.config.memory_budget)
            .map_err(|e| Error::indexing(format!("Failed to create index writer: {}", e)))?;

        let mut total_indexed = 0;
        for doc in docs {
            for chunk in &doc.chunks {
                let mut tantivy_doc = TantivyDocument::new();
                tantivy_doc.add_text(self.fields.doc_id, doc.id.to_string());
                tantivy_doc.add_text(self.fields.chunk_id, chunk.id.to_string());
                tantivy_doc.add_text(self.fields.text, &chunk.text);
                if let Some(ref section) = chunk.section {
                    tantivy_doc.add_text(self.fields.section, section);
                }

                writer.add_document(tantivy_doc)
                    .map_err(|e| Error::indexing(format!("Failed to add document: {}", e)))?;
                total_indexed += 1;
            }
        }

        writer.commit()
            .map_err(|e| Error::indexing(format!("Failed to commit index: {}", e)))?;

        Ok(total_indexed)
    }

    /// Search the index using BM25
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<BM25Result>> {
        let reader = self.index.reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| Error::indexing(format!("Failed to create reader: {}", e)))?;

        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.fields.text, self.fields.section]);
        let query = query_parser.parse_query(query)
            .map_err(|e| Error::query(format!("Failed to parse query: {}", e)))?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))
            .map_err(|e| Error::query(format!("Search failed: {}", e)))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)
                .map_err(|e| Error::query(format!("Failed to retrieve document: {}", e)))?;

            let doc_id = retrieved_doc
                .get_first(self.fields.doc_id)
                .and_then(|v: &tantivy::schema::OwnedValue| v.as_str())
                .map(|s: &str| s.to_string())
                .unwrap_or_default();

            let chunk_id = retrieved_doc
                .get_first(self.fields.chunk_id)
                .and_then(|v: &tantivy::schema::OwnedValue| v.as_str())
                .map(|s: &str| s.to_string())
                .unwrap_or_default();

            let text = retrieved_doc
                .get_first(self.fields.text)
                .and_then(|v: &tantivy::schema::OwnedValue| v.as_str())
                .map(|s: &str| s.to_string())
                .unwrap_or_default();

            results.push(BM25Result {
                doc_id: Uuid::parse_str(&doc_id).unwrap_or_default(),
                chunk_id: Uuid::parse_str(&chunk_id).unwrap_or_default(),
                score,
                text,
            });
        }

        Ok(results)
    }

    /// Delete all documents from a specific document ID
    pub fn delete_document(&self, doc_id: &Uuid) -> Result<()> {
        let mut writer: tantivy::IndexWriter<TantivyDocument> = self.index.writer(self.config.memory_budget)
            .map_err(|e| Error::indexing(format!("Failed to create writer: {}", e)))?;

        let term = tantivy::Term::from_field_text(self.fields.doc_id, &doc_id.to_string());
        writer.delete_term(term);

        writer.commit()
            .map_err(|e| Error::indexing(format!("Failed to commit delete: {}", e)))?;

        Ok(())
    }

    /// Get index statistics
    pub fn stats(&self) -> Result<IndexStats> {
        let reader = self.index.reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| Error::indexing(format!("Failed to create reader: {}", e)))?;

        let searcher = reader.searcher();

        Ok(IndexStats {
            document_count: 0, // Would need to count unique doc_ids
            chunk_count: searcher.num_docs() as usize,
            size_bytes: 0, // Would need directory size
            last_updated: None,
        })
    }

    /// Optimize the index (merge segments)
    pub fn optimize(&self) -> Result<()> {
        let writer: tantivy::IndexWriter<TantivyDocument> = self.index.writer(self.config.memory_budget)
            .map_err(|e| Error::indexing(format!("Failed to create writer: {}", e)))?;

        // Wait for merging to complete
        writer.wait_merging_threads()
            .map_err(|e| Error::indexing(format!("Failed to wait for merge: {}", e)))?;

        Ok(())
    }
}

/// Result from BM25 search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Result {
    /// Document ID
    pub doc_id: Uuid,
    /// Chunk ID
    pub chunk_id: Uuid,
    /// BM25 score
    pub score: f32,
    /// Matched text
    pub text: String,
}

/// Index manager for managing multiple index types
pub struct IndexManager {
    bm25: BM25Index,
    base_path: PathBuf,
}

impl IndexManager {
    /// Create a new index manager with in-memory indexes
    pub fn in_memory() -> Result<Self> {
        Ok(Self {
            bm25: BM25Index::in_memory()?,
            base_path: PathBuf::from("."),
        })
    }

    /// Create a new index manager with persistent indexes
    pub fn open(base_path: PathBuf) -> Result<Self> {
        let bm25_path = base_path.join("bm25");
        Ok(Self {
            bm25: BM25Index::open(bm25_path)?,
            base_path,
        })
    }

    /// Index a document
    pub fn index_document(&self, doc: &Document) -> Result<usize> {
        self.bm25.index_document(doc)
    }

    /// Search using BM25
    pub fn search_bm25(&self, query: &str, top_k: usize) -> Result<Vec<BM25Result>> {
        self.bm25.search(query, top_k)
    }

    /// Delete a document from all indexes
    pub fn delete_document(&self, doc_id: &Uuid) -> Result<()> {
        self.bm25.delete_document(doc_id)
    }

    /// Get combined index statistics
    pub fn stats(&self) -> Result<IndexStats> {
        self.bm25.stats()
    }

    /// Optimize all indexes
    pub fn optimize(&self) -> Result<()> {
        self.bm25.optimize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DocumentType, Source, SourceType, EmbeddingIds, Chunk};
    use chrono::Utc;

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

        let mut doc = Document::new(DocumentType::Note, source)
            .with_content("This is a test document about machine learning and artificial intelligence.".to_string());

        // Add some chunks
        doc.chunks = vec![
            Chunk {
                id: Uuid::new_v4(),
                text: "Machine learning is a subset of artificial intelligence.".to_string(),
                index: 0,
                start_char: 0,
                end_char: 55,
                token_count: Some(10),
                section: Some("Introduction".to_string()),
                page: None,
                embedding_ids: EmbeddingIds::default(),
            },
            Chunk {
                id: Uuid::new_v4(),
                text: "Deep learning uses neural networks with many layers.".to_string(),
                index: 1,
                start_char: 56,
                end_char: 107,
                token_count: Some(9),
                section: Some("Deep Learning".to_string()),
                page: None,
                embedding_ids: EmbeddingIds::default(),
            },
        ];

        doc
    }

    #[test]
    fn test_bm25_index_and_search() {
        let index = BM25Index::in_memory().unwrap();
        let doc = create_test_document();

        // Index the document
        let indexed = index.index_document(&doc).unwrap();
        assert_eq!(indexed, 2);

        // Search
        let results = index.search("machine learning", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].text.contains("Machine learning"));
    }

    #[test]
    fn test_index_manager() {
        let manager = IndexManager::in_memory().unwrap();
        let doc = create_test_document();

        // Index
        manager.index_document(&doc).unwrap();

        // Search BM25
        let results = manager.search_bm25("neural networks", 5).unwrap();
        assert!(!results.is_empty());
    }
}
