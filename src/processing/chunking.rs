//! Document Chunking Module
//!
//! Provides multiple chunking strategies for splitting documents into smaller pieces
//! suitable for embedding and retrieval.

use crate::{Chunk, Document, DocumentType, EmbeddingIds};
use uuid::Uuid;

/// Chunking strategy configuration
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Target chunk size in tokens (approximate)
    pub chunk_size: usize,
    /// Overlap between chunks in tokens
    pub chunk_overlap: usize,
    /// Minimum chunk size (don't create tiny chunks)
    pub min_chunk_size: usize,
    /// Strategy to use
    pub strategy: ChunkingStrategy,
    /// Preserve sentence boundaries
    pub respect_sentences: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            min_chunk_size: 100,
            strategy: ChunkingStrategy::Recursive,
            respect_sentences: true,
        }
    }
}

/// Chunking strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Fixed token count chunks (simple splitting)
    FixedSize,
    /// Split on semantic boundaries (paragraphs, sections)
    Semantic,
    /// Recursive character splitting (try different delimiters)
    Recursive,
    /// Document-type aware chunking
    DocumentAware,
}

/// Chunking error
#[derive(Debug, thiserror::Error)]
pub enum ChunkingError {
    #[error("Text too short for chunking: {0} characters")]
    TextTooShort(usize),
    #[error("Invalid chunk size: {0}")]
    InvalidChunkSize(usize),
    #[error("Chunking failed: {0}")]
    ChunkingFailed(String),
}

/// Chunk a document into smaller pieces
pub fn chunk_document(
    document: &Document,
    config: &ChunkingConfig,
) -> Result<Vec<Chunk>, ChunkingError> {
    let text = &document.content.raw;

    if text.is_empty() {
        return Ok(Vec::new());
    }

    if text.len() < config.min_chunk_size {
        return Err(ChunkingError::TextTooShort(text.len()));
    }

    if config.chunk_size < config.min_chunk_size {
        return Err(ChunkingError::InvalidChunkSize(config.chunk_size));
    }

    let chunks = match config.strategy {
        ChunkingStrategy::FixedSize => chunk_fixed_size(text, config, document.id),
        ChunkingStrategy::Semantic => chunk_semantic(text, config, document.id, &document.doc_type),
        ChunkingStrategy::Recursive => {
            chunk_recursive(text, config, document.id, &document.doc_type)
        }
        ChunkingStrategy::DocumentAware => {
            chunk_document_aware(text, config, document.id, &document.doc_type)
        }
    }?;

    Ok(chunks)
}

/// Fixed-size chunking (simple token-based splitting)
fn chunk_fixed_size(
    text: &str,
    config: &ChunkingConfig,
    _document_id: Uuid,
) -> Result<Vec<Chunk>, ChunkingError> {
    let mut chunks = Vec::new();
    let chunk_size_chars = estimate_chars_from_tokens(config.chunk_size);
    let overlap_chars = estimate_chars_from_tokens(config.chunk_overlap);

    let mut start = 0;
    let mut index = 0;

    while start < text.len() {
        let end = (start + chunk_size_chars).min(text.len());
        let chunk_text = &text[start..end];

        if chunk_text.trim().len() < config.min_chunk_size {
            break; // Don't create tiny chunks at the end
        }

        let token_count = super::estimate_tokens(chunk_text);

        chunks.push(Chunk {
            id: Uuid::new_v4(),
            text: chunk_text.to_string(),
            index,
            start_char: start,
            end_char: end,
            token_count: Some(token_count),
            section: None,
            page: None,
            embedding_ids: EmbeddingIds::default(),
        });

        // Move start position with overlap
        let next_start = end.saturating_sub(overlap_chars);
        if next_start > start {
            start = next_start;
        } else {
            // Ensure we always make progress
            start += 1;
        }
        index += 1;

        // Prevent infinite loop (should be covered by loop condition, but good safety)
        if start >= text.len() {
            break;
        }
    }

    Ok(chunks)
}

/// Semantic chunking (split on paragraph/section boundaries)
fn chunk_semantic(
    text: &str,
    config: &ChunkingConfig,
    document_id: Uuid,
    doc_type: &DocumentType,
) -> Result<Vec<Chunk>, ChunkingError> {
    // First, split into paragraphs
    let paragraphs = super::split_paragraphs(text);

    if paragraphs.is_empty() {
        return chunk_fixed_size(text, config, document_id);
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_start = 0;
    let mut chunk_index = 0;
    let chunk_size_chars = estimate_chars_from_tokens(config.chunk_size);
    let overlap_chars = estimate_chars_from_tokens(config.chunk_overlap);

    for paragraph in paragraphs.iter() {
        let para_text = paragraph.trim();
        if para_text.is_empty() {
            continue;
        }

        // If adding this paragraph would exceed chunk size, finalize current chunk
        if !current_chunk.is_empty()
            && (current_chunk.len() + para_text.len() + 1) > chunk_size_chars
        {
            // Create chunk from accumulated text
            let end_pos = current_start + current_chunk.len();
            let token_count = super::estimate_tokens(&current_chunk);

            chunks.push(Chunk {
                id: Uuid::new_v4(),
                text: current_chunk.clone(),
                index: chunk_index,
                start_char: current_start,
                end_char: end_pos,
                token_count: Some(token_count),
                section: extract_section_header(paragraph, doc_type),
                page: None,
                embedding_ids: EmbeddingIds::default(),
            });

            // Start new chunk with overlap
            let overlap_text = extract_overlap(&current_chunk, overlap_chars);
            current_chunk = format!("{}{}", overlap_text, para_text);
            current_start = end_pos.saturating_sub(overlap_chars);
            chunk_index += 1;
        } else {
            // Add paragraph to current chunk
            if current_chunk.is_empty() {
                current_start = text.find(para_text).unwrap_or(current_start);
            } else {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(para_text);
        }
    }

    // Add final chunk if there's remaining text
    if !current_chunk.trim().is_empty() {
        let end_pos = current_start + current_chunk.len();
        let token_count = super::estimate_tokens(&current_chunk);

        chunks.push(Chunk {
            id: Uuid::new_v4(),
            text: current_chunk,
            index: chunk_index,
            start_char: current_start,
            end_char: end_pos,
            token_count: Some(token_count),
            section: None,
            page: None,
            embedding_ids: EmbeddingIds::default(),
        });
    }

    Ok(chunks)
}

/// Recursive chunking (try different delimiters in order)
fn chunk_recursive(
    text: &str,
    config: &ChunkingConfig,
    document_id: Uuid,
    doc_type: &DocumentType,
) -> Result<Vec<Chunk>, ChunkingError> {
    let _chunk_size_chars = estimate_chars_from_tokens(config.chunk_size);

    // Try different delimiters in order of preference
    let delimiters = if matches!(doc_type, DocumentType::Code) {
        vec!["\n\n\n", "\n\n", "\n", ". ", " "]
    } else {
        // Documentation and other types use the same delimiters
        vec!["\n\n", "\n", ". ", " "]
    };

    chunk_recursive_internal(text, config, document_id, &delimiters, 0)
}

fn chunk_recursive_internal(
    text: &str,
    config: &ChunkingConfig,
    document_id: Uuid,
    delimiters: &[&str],
    delimiter_idx: usize,
) -> Result<Vec<Chunk>, ChunkingError> {
    if delimiter_idx >= delimiters.len() {
        // Fallback: character-level splitting
        return chunk_fixed_size(text, config, document_id);
    }

    let delimiter = delimiters[delimiter_idx];
    let chunk_size_chars = estimate_chars_from_tokens(config.chunk_size);
    let overlap_chars = estimate_chars_from_tokens(config.chunk_overlap);

    // Split by delimiter
    let parts: Vec<&str> = text.split(delimiter).collect();

    if parts.len() <= 1 {
        // Delimiter not found, try next one
        return chunk_recursive_internal(text, config, document_id, delimiters, delimiter_idx + 1);
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_start = 0;
    let mut chunk_index = 0;

    for part in parts {
        let part_trimmed = part.trim();
        if part_trimmed.is_empty() {
            continue;
        }

        let part_with_delim = if current_chunk.is_empty() {
            part_trimmed.to_string()
        } else {
            format!("{}{}", delimiter, part_trimmed)
        };

        if (current_chunk.len() + part_with_delim.len()) > chunk_size_chars
            && !current_chunk.is_empty()
        {
            // Finalize current chunk
            let end_pos = current_start + current_chunk.len();
            let token_count = super::estimate_tokens(&current_chunk);

            chunks.push(Chunk {
                id: Uuid::new_v4(),
                text: current_chunk.clone(),
                index: chunk_index,
                start_char: current_start,
                end_char: end_pos,
                token_count: Some(token_count),
                section: None,
                page: None,
                embedding_ids: EmbeddingIds::default(),
            });

            // Start new chunk with overlap
            let overlap_text = extract_overlap(&current_chunk, overlap_chars);
            current_chunk = format!("{}{}", overlap_text, part_with_delim);
            current_start = end_pos.saturating_sub(overlap_chars);
            chunk_index += 1;
        } else {
            if current_chunk.is_empty() {
                current_start = text.find(part_trimmed).unwrap_or(current_start);
            }
            current_chunk.push_str(&part_with_delim);
        }
    }

    // Add final chunk
    if !current_chunk.trim().is_empty() {
        let end_pos = current_start + current_chunk.len();
        let token_count = super::estimate_tokens(&current_chunk);

        chunks.push(Chunk {
            id: Uuid::new_v4(),
            text: current_chunk,
            index: chunk_index,
            start_char: current_start,
            end_char: end_pos,
            token_count: Some(token_count),
            section: None,
            page: None,
            embedding_ids: EmbeddingIds::default(),
        });
    }

    // If chunks are still too large, recursively chunk them
    let mut final_chunks = Vec::new();
    for chunk in chunks {
        if chunk.text.len() > chunk_size_chars * 2 {
            // Chunk is too large, recursively split it
            let sub_chunks = chunk_recursive_internal(
                &chunk.text,
                config,
                document_id,
                delimiters,
                delimiter_idx + 1,
            )?;
            final_chunks.extend(sub_chunks);
        } else {
            final_chunks.push(chunk);
        }
    }

    Ok(final_chunks)
}

/// Document-aware chunking (uses document type to choose best strategy)
fn chunk_document_aware(
    text: &str,
    config: &ChunkingConfig,
    document_id: Uuid,
    doc_type: &DocumentType,
) -> Result<Vec<Chunk>, ChunkingError> {
    match doc_type {
        DocumentType::Code => {
            // For code: split on function/class boundaries
            chunk_code_aware(text, config, document_id)
        }
        DocumentType::Documentation | DocumentType::Paper => {
            // For docs/papers: try markdown-aware first, fall back to semantic
            if text.contains('#') {
                chunk_markdown_aware(text, config, document_id)
            } else {
                chunk_semantic(text, config, document_id, doc_type)
            }
        }
        _ => {
            // Default: recursive chunking
            chunk_recursive(text, config, document_id, doc_type)
        }
    }
}

/// Code-aware chunking (split on function/class boundaries)
fn chunk_code_aware(
    text: &str,
    config: &ChunkingConfig,
    document_id: Uuid,
) -> Result<Vec<Chunk>, ChunkingError> {
    // Simple approach: split on double newlines (common in code formatting)
    // In production, could use AST parsing for better boundaries
    let parts: Vec<&str> = text.split("\n\n\n").collect();

    if parts.len() <= 1 {
        // Fall back to recursive chunking
        return chunk_recursive(text, config, document_id, &DocumentType::Code);
    }

    let mut chunks = Vec::new();
    let mut current_pos = 0;

    for (idx, part) in parts.iter().enumerate() {
        let part_trimmed = part.trim();
        if part_trimmed.is_empty() {
            continue;
        }

        let start_pos = text[current_pos..]
            .find(part_trimmed)
            .map(|p| current_pos + p)
            .unwrap_or(current_pos);
        let end_pos = start_pos + part_trimmed.len();
        let token_count = super::estimate_tokens(part_trimmed);

        chunks.push(Chunk {
            id: Uuid::new_v4(),
            text: part_trimmed.to_string(),
            index: idx,
            start_char: start_pos,
            end_char: end_pos,
            token_count: Some(token_count),
            section: extract_function_name(part_trimmed),
            page: None,
            embedding_ids: EmbeddingIds::default(),
        });

        current_pos = end_pos;
    }

    Ok(chunks)
}

/// Markdown-aware chunking (split on headers)
fn chunk_markdown_aware(
    text: &str,
    config: &ChunkingConfig,
    document_id: Uuid,
) -> Result<Vec<Chunk>, ChunkingError> {
    // Split on markdown headers (# ## ###)
    let header_pattern = regex::Regex::new(r"(?m)^#{1,6}\s+.+$").unwrap();
    let mut chunks = Vec::new();
    let mut last_header_end = 0;
    let mut chunk_index = 0;
    let chunk_size_chars = estimate_chars_from_tokens(config.chunk_size);

    for mat in header_pattern.find_iter(text) {
        let header_start = mat.start();

        // If there's content between headers, create a chunk
        if header_start > last_header_end {
            let section_text = &text[last_header_end..header_start].trim();
            if !section_text.is_empty() && section_text.len() >= config.min_chunk_size {
                let token_count = super::estimate_tokens(section_text);
                let header_text = extract_previous_header(&text[..last_header_end]);

                chunks.push(Chunk {
                    id: Uuid::new_v4(),
                    text: section_text.to_string(),
                    index: chunk_index,
                    start_char: last_header_end,
                    end_char: header_start,
                    token_count: Some(token_count),
                    section: header_text,
                    page: None,
                    embedding_ids: EmbeddingIds::default(),
                });
                chunk_index += 1;
            }
        }

        last_header_end = header_start;
    }

    // Add final section
    if last_header_end < text.len() {
        let section_text = &text[last_header_end..].trim();
        if !section_text.is_empty() && section_text.len() >= config.min_chunk_size {
            let token_count = super::estimate_tokens(section_text);
            let header_text = extract_previous_header(&text[..last_header_end]);

            chunks.push(Chunk {
                id: Uuid::new_v4(),
                text: section_text.to_string(),
                index: chunk_index,
                start_char: last_header_end,
                end_char: text.len(),
                token_count: Some(token_count),
                section: header_text,
                page: None,
                embedding_ids: EmbeddingIds::default(),
            });
        }
    }

    // If no headers found or chunks are too large, fall back to semantic chunking
    if chunks.is_empty() || chunks.iter().any(|c| c.text.len() > chunk_size_chars * 2) {
        return chunk_semantic(text, config, document_id, &DocumentType::Documentation);
    }

    Ok(chunks)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Estimate character count from token count (rough: ~4 chars per token)
fn estimate_chars_from_tokens(tokens: usize) -> usize {
    tokens * 4
}

/// Extract overlap text from end of chunk
fn extract_overlap(text: &str, overlap_chars: usize) -> String {
    if text.len() <= overlap_chars {
        return text.to_string();
    }

    // Calculate where to start searching for a sentence boundary
    let start_search = text.len().saturating_sub(overlap_chars);
    let overlap_region = &text[start_search..];

    // Try to find a sentence boundary in the overlap region
    // Prefer finding a boundary that gives us roughly overlap_chars
    if let Some(sentence_start) = overlap_region.find(|c: char| c.is_uppercase()) {
        // Check if preceded by punctuation
        if start_search + sentence_start >= 2 {
            let prev_chars =
                &text[start_search + sentence_start - 2..start_search + sentence_start];
            if prev_chars.ends_with(". ")
                || prev_chars.ends_with("! ")
                || prev_chars.ends_with("? ")
            {
                return text[start_search + sentence_start..].to_string();
            }
        }
    }

    // Fallback: just return the last overlap_chars
    text[start_search..].to_string()
}

/// Extract section header from paragraph (for semantic chunking)
fn extract_section_header(paragraph: &str, _doc_type: &DocumentType) -> Option<String> {
    // Look for markdown headers
    if let Some(header_match) = regex::Regex::new(r"^#{1,6}\s+(.+)$")
        .ok()
        .and_then(|re| re.captures(paragraph.lines().next().unwrap_or("")))
    {
        return Some(header_match.get(1).unwrap().as_str().trim().to_string());
    }

    // Look for all-caps lines (common in papers)
    if let Some(first_line) = paragraph.lines().next() {
        if first_line.len() > 5
            && first_line
                .chars()
                .all(|c| c.is_uppercase() || c.is_whitespace() || c.is_ascii_punctuation())
        {
            return Some(first_line.trim().to_string());
        }
    }

    None
}

/// Extract function name from code (simple heuristic)
fn extract_function_name(code: &str) -> Option<String> {
    // Look for common function patterns
    let patterns = vec![
        r"fn\s+(\w+)",
        r"function\s+(\w+)",
        r"def\s+(\w+)",
        r"pub\s+fn\s+(\w+)",
    ];

    for pattern in patterns {
        if let Some(captures) = regex::Regex::new(pattern)
            .ok()
            .and_then(|re| re.captures(code))
        {
            return Some(captures.get(1).unwrap().as_str().to_string());
        }
    }

    None
}

/// Extract previous header from text
fn extract_previous_header(text: &str) -> Option<String> {
    regex::Regex::new(r"(?m)^#{1,6}\s+(.+)$")
        .ok()
        .and_then(|re| {
            re.captures_iter(text)
                .last()
                .and_then(|cap| cap.get(1))
                .map(|m| m.as_str().trim().to_string())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DocumentType, Source, SourceType};
    use chrono::Utc;

    fn create_test_document(text: &str, doc_type: DocumentType) -> Document {
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some("/test/doc.txt".to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        Document::new(doc_type, source).with_content(text.to_string())
    }

    #[test]
    fn test_fixed_size_chunking() {
        let text = "This is a test document. ".repeat(100); // ~2800 chars
        let doc = create_test_document(&text, DocumentType::Note);
        let config = ChunkingConfig {
            chunk_size: 512,
            chunk_overlap: 50,
            min_chunk_size: 100,
            strategy: ChunkingStrategy::FixedSize,
            respect_sentences: true,
        };

        let chunks = chunk_document(&doc, &config).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks.len() > 1); // Should create multiple chunks

        // Verify chunk properties
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
            assert!(chunk.start_char < chunk.end_char);
            assert!(chunk.token_count.is_some());
        }
    }

    #[test]
    fn test_semantic_chunking() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let doc = create_test_document(text, DocumentType::Documentation);
        let config = ChunkingConfig {
            chunk_size: 100, // Small to force multiple chunks
            chunk_overlap: 10,
            min_chunk_size: 10,
            strategy: ChunkingStrategy::Semantic,
            respect_sentences: true,
        };

        let chunks = chunk_document(&doc, &config).unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_recursive_chunking() {
        let text = "Sentence one. Sentence two. Sentence three. ".repeat(20);
        let doc = create_test_document(&text, DocumentType::Note);
        let config = ChunkingConfig {
            chunk_size: 200,
            chunk_overlap: 20,
            min_chunk_size: 50,
            strategy: ChunkingStrategy::Recursive,
            respect_sentences: true,
        };

        let chunks = chunk_document(&doc, &config).unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_aware_chunking() {
        let text =
            "# Header 1\n\nContent under header 1.\n\n## Header 2\n\nContent under header 2.";
        let doc = create_test_document(text, DocumentType::Documentation);
        let config = ChunkingConfig {
            chunk_size: 200,
            chunk_overlap: 10,
            min_chunk_size: 10,
            strategy: ChunkingStrategy::DocumentAware,
            respect_sentences: true,
        };

        let chunks = chunk_document(&doc, &config).unwrap();
        assert!(!chunks.is_empty());

        // Should have section headers
        assert!(chunks.iter().any(|c| c.section.is_some()));
    }

    #[test]
    fn test_empty_text() {
        let doc = create_test_document("", DocumentType::Note);
        let config = ChunkingConfig::default();
        let chunks = chunk_document(&doc, &config).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_text_too_short() {
        let doc = create_test_document("Short", DocumentType::Note);
        let config = ChunkingConfig {
            min_chunk_size: 100,
            ..Default::default()
        };
        let result = chunk_document(&doc, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_overlap_extraction() {
        let text = "This is a long sentence. This is another sentence. Final sentence.";
        let overlap = extract_overlap(text, 20);
        assert!(!overlap.is_empty());
        assert!(overlap.len() <= 20);
    }
}
