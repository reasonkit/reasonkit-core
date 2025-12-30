//! ProofLedger - Immutable Citation Ledger
//!
//! Part of Protocol Delta: The Anchor
//!
//! Provides cryptographic binding for citations to prevent drift and ensure
//! research claims remain verifiable over time.

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, Result as SqliteResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during ProofLedger operations
#[derive(Error, Debug)]
pub enum ProofLedgerError {
    /// Database operation failed
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Hash not found in ledger
    #[error("Hash not found: {0}")]
    HashNotFound(String),

    /// Content drift detected
    #[error("Content drift detected: expected {expected}, got {actual}")]
    DriftDetected {
        /// Expected hash
        expected: String,
        /// Actual hash of current content
        actual: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for ProofLedger operations
pub type Result<T> = std::result::Result<T, ProofLedgerError>;

/// An immutable anchor representing a snapshot of content at a point in time
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Anchor {
    /// SHA-256 hash of the content (primary key)
    pub hash: String,

    /// Source URL
    pub url: String,

    /// Timestamp when the content was anchored
    pub timestamp: DateTime<Utc>,

    /// Snippet of the content (first 200 chars)
    pub content_snippet: String,

    /// Full content (stored separately in content-addressable storage)
    /// Only the snippet is kept in the ledger for efficiency
    #[serde(skip)]
    pub full_content: Option<String>,

    /// Optional metadata (JSON)
    pub metadata: Option<String>,
}

/// Verification result from drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether verification passed
    pub verified: bool,

    /// Original hash from anchor
    pub original_hash: String,

    /// Current content hash
    pub current_hash: String,

    /// Human-readable message
    pub message: String,

    /// Original anchor data
    pub anchor: Anchor,
}

/// The ProofLedger - manages immutable citations
pub struct ProofLedger {
    /// SQLite connection for the ledger
    conn: Connection,

    /// Path to the ledger database
    ledger_path: PathBuf,
}

impl ProofLedger {
    /// Create a new ProofLedger with the specified ledger path
    ///
    /// # Arguments
    ///
    /// * `ledger_path` - Path to the SQLite database file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use reasonkit::verification::ProofLedger;
    ///
    /// let ledger = ProofLedger::new("./proof_ledger.db")?;
    /// # Ok::<(), reasonkit::verification::ProofLedgerError>(())
    /// ```
    pub fn new<P: AsRef<Path>>(ledger_path: P) -> Result<Self> {
        let path = ledger_path.as_ref().to_path_buf();
        let conn = Connection::open(&path)?;

        // Create tables if they don't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS anchors (
                hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                content_snippet TEXT NOT NULL,
                metadata TEXT
            )",
            [],
        )?;

        // Create index on URL for faster lookups
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_anchors_url ON anchors(url)",
            [],
        )?;

        // Create index on timestamp for temporal queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_anchors_timestamp ON anchors(timestamp)",
            [],
        )?;

        Ok(Self {
            conn,
            ledger_path: path,
        })
    }

    /// Create an in-memory ProofLedger (for testing)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use reasonkit::verification::ProofLedger;
    ///
    /// let ledger = ProofLedger::in_memory()?;
    /// # Ok::<(), reasonkit::verification::ProofLedgerError>(())
    /// ```
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;

        conn.execute(
            "CREATE TABLE anchors (
                hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                content_snippet TEXT NOT NULL,
                metadata TEXT
            )",
            [],
        )?;

        conn.execute("CREATE INDEX idx_anchors_url ON anchors(url)", [])?;

        conn.execute(
            "CREATE INDEX idx_anchors_timestamp ON anchors(timestamp)",
            [],
        )?;

        Ok(Self {
            conn,
            ledger_path: PathBuf::from(":memory:"),
        })
    }

    /// Compute SHA-256 hash of content
    fn compute_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Anchor content to the ledger
    ///
    /// Creates an immutable anchor for the given content and URL.
    /// Returns the hash ID which can be used for citations.
    ///
    /// # Arguments
    ///
    /// * `content` - The full content to anchor
    /// * `url` - Source URL
    /// * `metadata` - Optional JSON metadata
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use reasonkit::verification::ProofLedger;
    ///
    /// let ledger = ProofLedger::in_memory()?;
    /// let hash = ledger.anchor(
    ///     "The global AI market size was valued at USD 196.63 billion in 2023.",
    ///     "https://example.com/ai-market",
    ///     None,
    /// )?;
    /// println!("Citation hash: {}", hash);
    /// # Ok::<(), reasonkit::verification::ProofLedgerError>(())
    /// ```
    pub fn anchor(&self, content: &str, url: &str, metadata: Option<String>) -> Result<String> {
        let hash = Self::compute_hash(content);
        let timestamp = Utc::now();

        // Create snippet (first 200 chars)
        let snippet = if content.len() > 200 {
            format!("{}...", &content[..200])
        } else {
            content.to_string()
        };

        // Try to insert; ignore if already exists
        let result = self.conn.execute(
            "INSERT OR IGNORE INTO anchors (hash, url, timestamp, content_snippet, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![hash, url, timestamp.to_rfc3339(), snippet, metadata],
        );

        match result {
            Ok(rows) if rows > 0 => {
                tracing::info!("Anchored new proof: {}... -> {}", &hash[..8], url);
            }
            Ok(_) => {
                tracing::debug!("Existing anchor found: {}...", &hash[..8]);
            }
            Err(e) => return Err(ProofLedgerError::Database(e)),
        }

        Ok(hash)
    }

    /// Retrieve an anchor by hash
    ///
    /// # Arguments
    ///
    /// * `hash` - The SHA-256 hash to look up
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use reasonkit::verification::ProofLedger;
    ///
    /// let ledger = ProofLedger::in_memory()?;
    /// let hash = ledger.anchor("test content", "https://example.com", None)?;
    /// let anchor = ledger.get_anchor(&hash)?;
    /// assert_eq!(anchor.url, "https://example.com");
    /// # Ok::<(), reasonkit::verification::ProofLedgerError>(())
    /// ```
    pub fn get_anchor(&self, hash: &str) -> Result<Anchor> {
        let mut stmt = self.conn.prepare(
            "SELECT hash, url, timestamp, content_snippet, metadata
             FROM anchors WHERE hash = ?1",
        )?;

        let anchor = stmt.query_row(params![hash], |row| {
            Ok(Anchor {
                hash: row.get(0)?,
                url: row.get(1)?,
                timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|_| rusqlite::Error::InvalidQuery)?,
                content_snippet: row.get(3)?,
                full_content: None,
                metadata: row.get(4)?,
            })
        })?;

        Ok(anchor)
    }

    /// Verify current content against anchored hash
    ///
    /// Detects if the content has drifted from the original anchored version.
    ///
    /// # Arguments
    ///
    /// * `hash` - The original hash from the citation
    /// * `current_content` - The current content to verify
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use reasonkit::verification::ProofLedger;
    ///
    /// let ledger = ProofLedger::in_memory()?;
    /// let original = "Original content";
    /// let hash = ledger.anchor(original, "https://example.com", None)?;
    ///
    /// // Verify with same content
    /// let result = ledger.verify(&hash, original)?;
    /// assert!(result.verified);
    ///
    /// // Verify with different content (drift)
    /// let result = ledger.verify(&hash, "Modified content")?;
    /// assert!(!result.verified);
    /// # Ok::<(), reasonkit::verification::ProofLedgerError>(())
    /// ```
    pub fn verify(&self, hash: &str, current_content: &str) -> Result<VerificationResult> {
        // Get the original anchor
        let anchor = self.get_anchor(hash)?;

        // Compute hash of current content
        let current_hash = Self::compute_hash(current_content);

        if current_hash == hash {
            Ok(VerificationResult {
                verified: true,
                original_hash: hash.to_string(),
                current_hash,
                message: "VERIFIED: Content matches original anchor".to_string(),
                anchor,
            })
        } else {
            Ok(VerificationResult {
                verified: false,
                original_hash: hash.to_string(),
                current_hash: current_hash.clone(),
                message: format!(
                    "DRIFT DETECTED: Expected {}..., got {}...",
                    &hash[..8],
                    &current_hash[..8]
                ),
                anchor,
            })
        }
    }

    /// Check for content drift by re-fetching and verifying
    ///
    /// This is a higher-level function that would typically:
    /// 1. Re-fetch content from the URL
    /// 2. Verify against the anchor
    /// 3. Return drift status
    ///
    /// Note: This function requires external fetch capability.
    /// For now, it just verifies the provided content.
    ///
    /// # Arguments
    ///
    /// * `hash` - The original hash
    /// * `refetched_content` - Content re-fetched from the source
    pub fn check_drift(&self, hash: &str, refetched_content: &str) -> Result<VerificationResult> {
        self.verify(hash, refetched_content)
    }

    /// List all anchors for a given URL
    ///
    /// Useful for finding all citations from a particular source.
    ///
    /// # Arguments
    ///
    /// * `url` - The URL to search for
    pub fn list_by_url(&self, url: &str) -> Result<Vec<Anchor>> {
        let mut stmt = self.conn.prepare(
            "SELECT hash, url, timestamp, content_snippet, metadata
             FROM anchors WHERE url = ?1
             ORDER BY timestamp DESC",
        )?;

        let anchors = stmt
            .query_map(params![url], |row| {
                Ok(Anchor {
                    hash: row.get(0)?,
                    url: row.get(1)?,
                    timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|_| rusqlite::Error::InvalidQuery)?,
                    content_snippet: row.get(3)?,
                    full_content: None,
                    metadata: row.get(4)?,
                })
            })?
            .collect::<SqliteResult<Vec<_>>>()?;

        Ok(anchors)
    }

    /// Count total anchors in the ledger
    pub fn count(&self) -> Result<i64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM anchors", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Get the ledger database path
    pub fn ledger_path(&self) -> &Path {
        &self.ledger_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_computation() {
        let content = "The global AI market size was valued at USD 196.63 billion in 2023.";
        let hash = ProofLedger::compute_hash(content);

        // Hash should be 64 hex characters (SHA-256)
        assert_eq!(hash.len(), 64);

        // Same content should produce same hash
        let hash2 = ProofLedger::compute_hash(content);
        assert_eq!(hash, hash2);

        // Different content should produce different hash
        let hash3 = ProofLedger::compute_hash("Different content");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_anchor_creation() {
        let ledger = ProofLedger::in_memory().unwrap();
        let content = "Test content for anchoring";
        let url = "https://example.com/test";

        let hash = ledger.anchor(content, url, None).unwrap();

        // Hash should be 64 hex characters
        assert_eq!(hash.len(), 64);

        // Should be able to retrieve the anchor
        let anchor = ledger.get_anchor(&hash).unwrap();
        assert_eq!(anchor.url, url);
        assert!(anchor.content_snippet.contains("Test content"));
    }

    #[test]
    fn test_duplicate_anchor() {
        let ledger = ProofLedger::in_memory().unwrap();
        let content = "Duplicate test";
        let url = "https://example.com";

        let hash1 = ledger.anchor(content, url, None).unwrap();
        let hash2 = ledger.anchor(content, url, None).unwrap();

        // Same content should produce same hash
        assert_eq!(hash1, hash2);

        // Should only have one entry
        assert_eq!(ledger.count().unwrap(), 1);
    }

    #[test]
    fn test_verification_success() {
        let ledger = ProofLedger::in_memory().unwrap();
        let content = "Original immutable content";
        let hash = ledger.anchor(content, "https://example.com", None).unwrap();

        let result = ledger.verify(&hash, content).unwrap();

        assert!(result.verified);
        assert_eq!(result.original_hash, result.current_hash);
        assert!(result.message.contains("VERIFIED"));
    }

    #[test]
    fn test_drift_detection() {
        let ledger = ProofLedger::in_memory().unwrap();
        let original = "Original content";
        let modified = "Modified content";

        let hash = ledger
            .anchor(original, "https://example.com", None)
            .unwrap();

        let result = ledger.verify(&hash, modified).unwrap();

        assert!(!result.verified);
        assert_ne!(result.original_hash, result.current_hash);
        assert!(result.message.contains("DRIFT DETECTED"));
    }

    #[test]
    fn test_list_by_url() {
        let ledger = ProofLedger::in_memory().unwrap();
        let url = "https://example.com/article";

        ledger.anchor("Content 1", url, None).unwrap();
        ledger.anchor("Content 2", url, None).unwrap();
        ledger
            .anchor("Content 3", "https://different.com", None)
            .unwrap();

        let anchors = ledger.list_by_url(url).unwrap();

        // Should have 2 anchors from the target URL
        assert_eq!(anchors.len(), 2);

        // All should have the correct URL
        assert!(anchors.iter().all(|a| a.url == url));
    }

    #[test]
    fn test_metadata_storage() {
        let ledger = ProofLedger::in_memory().unwrap();
        let metadata = r#"{"type": "market_stat", "confidence": 0.95}"#.to_string();

        let hash = ledger
            .anchor(
                "Content with metadata",
                "https://example.com",
                Some(metadata.clone()),
            )
            .unwrap();

        let anchor = ledger.get_anchor(&hash).unwrap();
        assert_eq!(anchor.metadata, Some(metadata));
    }

    #[test]
    fn test_snippet_truncation() {
        let ledger = ProofLedger::in_memory().unwrap();
        let long_content = "A".repeat(300);

        let hash = ledger
            .anchor(&long_content, "https://example.com", None)
            .unwrap();
        let anchor = ledger.get_anchor(&hash).unwrap();

        // Snippet should be truncated to ~203 chars (200 + "...")
        assert!(anchor.content_snippet.len() <= 204);
        assert!(anchor.content_snippet.ends_with("..."));
    }

    #[test]
    fn test_hash_not_found() {
        let ledger = ProofLedger::in_memory().unwrap();
        let fake_hash = "0".repeat(64);

        let result = ledger.get_anchor(&fake_hash);
        assert!(result.is_err());
    }
}
