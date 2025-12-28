//! PDF ingestion module using lopdf
//!
//! Extracts text content from PDF files for indexing in the knowledge base.

use crate::{Document, DocumentType, Error, Metadata, Result, Source, SourceType};
use chrono::Utc;
use lopdf::Document as PdfDocument;
use std::path::Path;
use tracing::{debug, info, warn};

/// PDF document ingester using lopdf
pub struct PdfIngester {
    /// Whether to extract metadata from PDF
    extract_metadata: bool,
}

impl PdfIngester {
    /// Create a new PDF ingester
    pub fn new() -> Self {
        Self {
            extract_metadata: true,
        }
    }

    /// Ingest a PDF file and extract text content
    pub fn ingest(&self, path: &Path) -> Result<Document> {
        info!("Ingesting PDF: {:?}", path);

        let pdf_doc = PdfDocument::load(path)
            .map_err(|e| Error::pdf(format!("Failed to load PDF: {}", e)))?;

        let mut full_text = String::new();
        let page_count = pdf_doc.get_pages().len();

        debug!("PDF has {} pages", page_count);

        // Extract text from each page
        for (page_num, _) in pdf_doc.get_pages() {
            match self.extract_page_text(&pdf_doc, page_num) {
                Ok(text) => {
                    if !text.is_empty() {
                        full_text.push_str(&text);
                        full_text.push('\n');
                    }
                }
                Err(e) => {
                    warn!("Failed to extract text from page {}: {}", page_num, e);
                }
            }
        }

        // Clean up the extracted text
        let cleaned_text = self.clean_text(&full_text);

        // Extract metadata if enabled
        let metadata = if self.extract_metadata {
            self.extract_metadata(&pdf_doc, path)
        } else {
            Metadata::default()
        };

        // Determine source type based on filename
        let source_type = self.detect_source_type(path);
        let arxiv_id = self.extract_arxiv_id(path);

        let source = Source {
            source_type,
            url: None,
            path: Some(path.to_string_lossy().to_string()),
            arxiv_id,
            github_repo: None,
            retrieved_at: Utc::now(),
            version: None,
        };

        let mut doc = Document::new(DocumentType::Paper, source).with_content(cleaned_text);

        doc.metadata = metadata;

        info!(
            "Extracted {} chars from {} pages",
            doc.content.char_count, page_count
        );

        Ok(doc)
    }

    /// Extract text from a single page
    fn extract_page_text(&self, doc: &PdfDocument, page_num: u32) -> Result<String> {
        let page_id = doc
            .page_iter()
            .nth((page_num - 1) as usize)
            .ok_or_else(|| Error::pdf(format!("Page {} not found", page_num)))?;

        let content = doc
            .get_page_content(page_id)
            .map_err(|e| Error::pdf(format!("Failed to get page content: {}", e)))?;

        // Parse content stream and extract text
        let text = self.parse_content_stream(&content, doc);

        Ok(text)
    }

    /// Parse PDF content stream to extract text
    fn parse_content_stream(&self, content: &[u8], _doc: &PdfDocument) -> String {
        let mut text = String::new();
        let content_str = String::from_utf8_lossy(content);

        // Simple text extraction - look for text operators
        // This is a simplified approach; full implementation would parse the content stream properly
        let mut in_text = false;
        let mut current_text = String::new();

        for line in content_str.lines() {
            let line = line.trim();

            // BT = Begin Text, ET = End Text
            if line == "BT" {
                in_text = true;
                continue;
            }
            if line == "ET" {
                if !current_text.is_empty() {
                    text.push_str(&current_text);
                    text.push(' ');
                    current_text.clear();
                }
                in_text = false;
                continue;
            }

            if in_text {
                // Look for text showing operators: Tj, TJ, ', "
                if let Some(text_content) = self.extract_text_from_operator(line) {
                    current_text.push_str(&text_content);
                }
            }
        }

        text
    }

    /// Extract text from PDF text operators
    fn extract_text_from_operator(&self, line: &str) -> Option<String> {
        let line = line.trim();

        // Tj operator: (text) Tj
        if line.ends_with("Tj") {
            if let Some(start) = line.find('(') {
                if let Some(end) = line.rfind(')') {
                    let text = &line[start + 1..end];
                    return Some(self.decode_pdf_string(text));
                }
            }
        }

        // TJ operator: [(text) num (text)] TJ
        if line.ends_with("TJ") {
            let mut result = String::new();
            let mut in_string = false;
            let mut current = String::new();

            for c in line.chars() {
                match c {
                    '(' => {
                        in_string = true;
                        current.clear();
                    }
                    ')' => {
                        if in_string {
                            result.push_str(&self.decode_pdf_string(&current));
                            in_string = false;
                        }
                    }
                    _ if in_string => {
                        current.push(c);
                    }
                    _ => {}
                }
            }

            if !result.is_empty() {
                return Some(result);
            }
        }

        None
    }

    /// Decode PDF string escapes
    fn decode_pdf_string(&self, s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('(') => result.push('('),
                    Some(')') => result.push(')'),
                    Some(d) if d.is_ascii_digit() => {
                        // Octal escape
                        let mut octal = String::from(d);
                        while octal.len() < 3 {
                            if let Some(&next) = chars.peek() {
                                if next.is_ascii_digit() {
                                    octal.push(chars.next().unwrap());
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        if let Ok(code) = u8::from_str_radix(&octal, 8) {
                            result.push(code as char);
                        }
                    }
                    Some(other) => result.push(other),
                    None => {}
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Clean extracted text
    fn clean_text(&self, text: &str) -> String {
        // Remove excessive whitespace
        let mut cleaned = String::new();
        let mut prev_was_space = false;

        for c in text.chars() {
            if c.is_whitespace() {
                if !prev_was_space {
                    cleaned.push(' ');
                    prev_was_space = true;
                }
            } else {
                cleaned.push(c);
                prev_was_space = false;
            }
        }

        // Remove common PDF artifacts
        cleaned = cleaned.replace("\u{0000}", "");
        cleaned = cleaned.replace("\u{FEFF}", ""); // BOM

        cleaned.trim().to_string()
    }

    /// Extract metadata from PDF
    fn extract_metadata(&self, doc: &PdfDocument, path: &Path) -> Metadata {
        let mut metadata = Metadata::default();

        // Helper to convert PDF string to Rust string
        let pdf_to_string = |obj: &lopdf::Object| -> Option<String> {
            match obj {
                lopdf::Object::String(bytes, _) => String::from_utf8(bytes.clone()).ok(),
                lopdf::Object::Name(bytes) => String::from_utf8(bytes.clone()).ok(),
                _ => None,
            }
        };

        // Try to get document info dictionary
        if let Ok(info) = doc.trailer.get(b"Info") {
            if let Ok(info_ref) = info.as_reference() {
                if let Ok(info_dict) = doc.get_dictionary(info_ref) {
                    // Title
                    if let Ok(title) = info_dict.get(b"Title") {
                        metadata.title = pdf_to_string(title);
                    }

                    // Author - convert to Author struct
                    if let Ok(author) = info_dict.get(b"Author") {
                        if let Some(author_str) = pdf_to_string(author) {
                            metadata.authors.push(crate::Author {
                                name: author_str,
                                affiliation: None,
                                email: None,
                            });
                        }
                    }

                    // Subject -> store as abstract
                    if let Ok(subject) = info_dict.get(b"Subject") {
                        if let Some(abstract_text) = pdf_to_string(subject) {
                            metadata.abstract_text = Some(abstract_text);
                        }
                    }

                    // Keywords -> store as tags
                    if let Ok(keywords) = info_dict.get(b"Keywords") {
                        if let Some(keywords_str) = pdf_to_string(keywords) {
                            metadata.tags = keywords_str
                                .split(',')
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect();
                        }
                    }
                }
            }
        }

        // Fall back to filename for title if not found
        if metadata.title.is_none() {
            metadata.title = path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.replace('_', " "));
        }

        metadata
    }

    /// Detect source type from filename
    fn detect_source_type(&self, path: &Path) -> SourceType {
        let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

        if filename.contains("arxiv") || filename.starts_with("2") {
            SourceType::Arxiv
        } else {
            SourceType::Local
        }
    }

    /// Extract arXiv ID from filename
    fn extract_arxiv_id(&self, path: &Path) -> Option<String> {
        let filename = path.file_stem().and_then(|s| s.to_str())?;

        // Pattern: anything_XXXX.XXXXX or arxiv_XXXX.XXXXX
        let re = regex::Regex::new(r"(\d{4}\.\d{4,5})").ok()?;

        re.captures(filename)
            .and_then(|caps| caps.get(1))
            .map(|m| m.as_str().to_string())
    }
}

impl Default for PdfIngester {
    fn default() -> Self {
        Self::new()
    }
}

impl super::Ingester for PdfIngester {
    fn ingest(&self, path: &Path) -> Result<Document> {
        PdfIngester::ingest(self, path)
    }

    fn can_handle(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase() == "pdf")
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_pdf_string() {
        let ingester = PdfIngester::new();

        assert_eq!(ingester.decode_pdf_string("hello"), "hello");
        assert_eq!(ingester.decode_pdf_string("hello\\nworld"), "hello\nworld");
        assert_eq!(ingester.decode_pdf_string("test\\(paren\\)"), "test(paren)");
    }

    #[test]
    fn test_extract_arxiv_id() {
        let ingester = PdfIngester::new();

        let path = Path::new("/data/papers/arxiv_2401.18059.pdf");
        assert_eq!(
            ingester.extract_arxiv_id(path),
            Some("2401.18059".to_string())
        );

        let path = Path::new("/data/papers/cot_2201.11903.pdf");
        assert_eq!(
            ingester.extract_arxiv_id(path),
            Some("2201.11903".to_string())
        );

        let path = Path::new("/data/papers/random_paper.pdf");
        assert_eq!(ingester.extract_arxiv_id(path), None);
    }

    #[test]
    fn test_clean_text() {
        let ingester = PdfIngester::new();

        let dirty = "  hello   world  \n\n  test  ";
        assert_eq!(ingester.clean_text(dirty), "hello world test");
    }
}
