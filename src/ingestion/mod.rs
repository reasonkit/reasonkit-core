//! Document ingestion module for ReasonKit Core
//!
//! Provides functionality to ingest documents from various formats:
//! - PDF (via lopdf)
//! - Markdown (via pulldown-cmark)
//! - HTML (via scraper)
//! - JSON/JSONL (via serde)

pub mod pdf;

use crate::{Document, DocumentType, Error, Metadata, Result, Source, SourceType};
use chrono::Utc;
use std::path::Path;

/// Trait for document ingesters
pub trait Ingester {
    /// Ingest a document from a file path
    fn ingest(&self, path: &Path) -> Result<Document>;

    /// Check if this ingester can handle the given file
    fn can_handle(&self, path: &Path) -> bool;
}

/// Main document ingester that delegates to format-specific ingesters
pub struct DocumentIngester {
    pdf_ingester: pdf::PdfIngester,
}

impl DocumentIngester {
    /// Create a new document ingester
    pub fn new() -> Self {
        Self {
            pdf_ingester: pdf::PdfIngester::new(),
        }
    }

    /// Ingest a document from a file path, auto-detecting format
    pub fn ingest(&self, path: &Path) -> Result<Document> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase());

        match extension.as_deref() {
            Some("pdf") => self.pdf_ingester.ingest(path),
            Some("md" | "markdown") => self.ingest_markdown(path),
            Some("html" | "htm") => self.ingest_html(path),
            Some("json") => self.ingest_json(path),
            Some("jsonl") => self.ingest_jsonl(path),
            Some("txt") => self.ingest_text(path),
            _ => Err(Error::Config(format!(
                "Unsupported file format: {:?}",
                path
            ))),
        }
    }

    /// Ingest a markdown file
    fn ingest_markdown(&self, path: &Path) -> Result<Document> {
        use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
        use std::fs;

        let content = fs::read_to_string(path)?;

        // Parse markdown to extract text and metadata
        let mut options = Options::empty();
        options.insert(Options::ENABLE_TABLES);
        options.insert(Options::ENABLE_FOOTNOTES);

        let parser = Parser::new_ext(&content, options);
        let mut text = String::new();
        let mut title: Option<String> = None;
        let mut in_heading = false;

        for event in parser {
            match event {
                Event::Start(Tag::Heading {
                    level: pulldown_cmark::HeadingLevel::H1,
                    ..
                }) => {
                    in_heading = true;
                }
                Event::End(TagEnd::Heading(pulldown_cmark::HeadingLevel::H1)) => {
                    in_heading = false;
                }
                Event::Text(t) => {
                    if in_heading && title.is_none() {
                        title = Some(t.to_string());
                    }
                    text.push_str(&t);
                    text.push(' ');
                }
                Event::SoftBreak | Event::HardBreak => {
                    text.push('\n');
                }
                _ => {}
            }
        }

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some(path.to_string_lossy().to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let mut doc = Document::new(DocumentType::Documentation, source)
            .with_content(text.trim().to_string());

        doc.metadata = Metadata {
            title,
            ..Default::default()
        };

        Ok(doc)
    }

    /// Ingest an HTML file
    fn ingest_html(&self, path: &Path) -> Result<Document> {
        use scraper::{Html, Selector};
        use std::fs;

        let content = fs::read_to_string(path)?;
        let document = Html::parse_document(&content);

        // Extract title
        let title_selector = Selector::parse("title").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|e| e.text().collect::<String>());

        // Extract body text
        let body_selector = Selector::parse("body").unwrap();
        let text = document
            .select(&body_selector)
            .next()
            .map(|e| e.text().collect::<Vec<_>>().join(" "))
            .unwrap_or_default();

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some(path.to_string_lossy().to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let mut doc = Document::new(DocumentType::Documentation, source)
            .with_content(text.trim().to_string());

        doc.metadata = Metadata {
            title,
            ..Default::default()
        };

        Ok(doc)
    }

    /// Ingest a JSON file
    fn ingest_json(&self, path: &Path) -> Result<Document> {
        use std::fs;

        let content = fs::read_to_string(path)?;

        // Try to parse as a Document first
        if let Ok(doc) = serde_json::from_str::<Document>(&content) {
            return Ok(doc);
        }

        // Otherwise, treat as raw content
        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some(path.to_string_lossy().to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        Ok(Document::new(DocumentType::Note, source).with_content(content))
    }

    /// Ingest a JSONL file (one document per line)
    fn ingest_jsonl(&self, path: &Path) -> Result<Document> {
        use std::fs;
        use std::io::{BufRead, BufReader};

        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);

        let mut all_content = String::new();

        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                // Try to extract content field if it exists
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(content) = json.get("content").and_then(|c| c.as_str()) {
                        all_content.push_str(content);
                        all_content.push('\n');
                    } else {
                        all_content.push_str(&line);
                        all_content.push('\n');
                    }
                }
            }
        }

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some(path.to_string_lossy().to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        Ok(Document::new(DocumentType::Documentation, source)
            .with_content(all_content.trim().to_string()))
    }

    /// Ingest a plain text file
    fn ingest_text(&self, path: &Path) -> Result<Document> {
        use std::fs;

        let content = fs::read_to_string(path)?;

        let source = Source {
            source_type: SourceType::Local,
            url: None,
            path: Some(path.to_string_lossy().to_string()),
            arxiv_id: None,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        Ok(Document::new(DocumentType::Note, source).with_content(content))
    }
}

impl Default for DocumentIngester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_markdown_ingestion() {
        let mut file = NamedTempFile::with_suffix(".md").unwrap();
        writeln!(file, "# Test Title\n\nThis is test content.").unwrap();

        let ingester = DocumentIngester::new();
        let doc = ingester.ingest(file.path()).unwrap();

        assert!(doc.content.raw.contains("Test Title"));
        assert!(doc.content.raw.contains("test content"));
    }

    #[test]
    fn test_text_ingestion() {
        let mut file = NamedTempFile::with_suffix(".txt").unwrap();
        writeln!(file, "Plain text content").unwrap();

        let ingester = DocumentIngester::new();
        let doc = ingester.ingest(file.path()).unwrap();

        assert!(doc.content.raw.contains("Plain text"));
    }
}
