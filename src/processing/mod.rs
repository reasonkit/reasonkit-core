//! Processing module for ReasonKit Core
//!
//! Provides document and text processing utilities for the RAG pipeline.
//!
//! ## Overview
//!
//! This module handles:
//! - Text normalization and cleaning
//! - Token counting and estimation
//! - Text chunking strategies
//! - Processing pipeline orchestration

use crate::{Document, ProcessingState};

/// Document chunking module
pub mod chunking;

/// Text normalization options
#[derive(Debug, Clone, Default)]
pub struct NormalizationOptions {
    /// Remove extra whitespace
    pub collapse_whitespace: bool,
    /// Convert to lowercase
    pub lowercase: bool,
    /// Remove punctuation
    pub remove_punctuation: bool,
    /// Trim leading/trailing whitespace
    pub trim: bool,
}

impl NormalizationOptions {
    /// Default normalization for search indexing
    pub fn for_indexing() -> Self {
        Self {
            collapse_whitespace: true,
            lowercase: false,
            remove_punctuation: false,
            trim: true,
        }
    }

    /// Aggressive normalization for matching
    pub fn for_matching() -> Self {
        Self {
            collapse_whitespace: true,
            lowercase: true,
            remove_punctuation: true,
            trim: true,
        }
    }
}

/// Normalize text according to options
pub fn normalize_text(text: &str, options: &NormalizationOptions) -> String {
    let mut result = text.to_string();

    if options.trim {
        result = result.trim().to_string();
    }

    if options.collapse_whitespace {
        result = collapse_whitespace(&result);
    }

    if options.lowercase {
        result = result.to_lowercase();
    }

    if options.remove_punctuation {
        result = result
            .chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect();
    }

    result
}

/// Collapse multiple whitespace characters into single spaces
fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_whitespace = false;

    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_whitespace {
                result.push(' ');
            }
            prev_whitespace = true;
        } else {
            result.push(c);
            prev_whitespace = false;
        }
    }

    result
}

/// Estimate token count for text (rough approximation: ~4 chars per token)
pub fn estimate_tokens(text: &str) -> usize {
    // Simple heuristic: ~4 characters per token for English text
    // This is a rough estimate that works reasonably well for most cases
    text.len().div_ceil(4)
}

/// Count words in text
pub fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Processing pipeline for documents
pub struct ProcessingPipeline {
    normalization: NormalizationOptions,
}

impl Default for ProcessingPipeline {
    fn default() -> Self {
        Self {
            normalization: NormalizationOptions::for_indexing(),
        }
    }
}

impl ProcessingPipeline {
    /// Create a new pipeline with custom normalization
    pub fn with_normalization(normalization: NormalizationOptions) -> Self {
        Self { normalization }
    }

    /// Process a document's content
    pub fn process_content(&self, content: &str) -> String {
        normalize_text(content, &self.normalization)
    }

    /// Update document processing status
    pub fn mark_processing(doc: &mut Document) {
        doc.processing.status = ProcessingState::Processing;
    }

    /// Mark document as processed
    pub fn mark_complete(doc: &mut Document) {
        doc.processing.status = ProcessingState::Completed;
        doc.processing.indexed = true;
    }

    /// Mark document processing as failed
    pub fn mark_failed(doc: &mut Document, error: &str) {
        doc.processing.status = ProcessingState::Failed;
        doc.processing.errors.push(error.to_string());
    }
}

/// Extract sentences from text
pub fn extract_sentences(text: &str) -> Vec<&str> {
    // Simple sentence splitting on common terminators
    // For production, consider using a proper NLP library
    text.split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Split text into paragraphs
pub fn split_paragraphs(text: &str) -> Vec<&str> {
    text.split("\n\n")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text() {
        let options = NormalizationOptions::for_indexing();
        assert_eq!(normalize_text("  Hello   World  ", &options), "Hello World");
    }

    #[test]
    fn test_normalize_for_matching() {
        let options = NormalizationOptions::for_matching();
        assert_eq!(normalize_text("Hello, World!", &options), "hello world");
    }

    #[test]
    fn test_estimate_tokens() {
        // ~4 chars per token
        assert_eq!(estimate_tokens("hello"), 2); // 5 chars -> ~2 tokens
        assert_eq!(estimate_tokens("hello world"), 3); // 11 chars -> ~3 tokens
    }

    #[test]
    fn test_count_words() {
        assert_eq!(count_words("hello world"), 2);
        assert_eq!(count_words("  hello   world  "), 2);
    }

    #[test]
    fn test_extract_sentences() {
        let text = "Hello world. How are you? I am fine!";
        let sentences = extract_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world");
    }

    #[test]
    fn test_split_paragraphs() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird one.";
        let paragraphs = split_paragraphs(text);
        assert_eq!(paragraphs.len(), 3);
    }
}
