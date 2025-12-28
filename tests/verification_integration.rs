//! Integration tests for Protocol Delta verification system (ProofLedger)

use reasonkit::verification::{ProofLedger, ProofLedgerError};
use tempfile::tempdir;

#[test]
fn test_proofledger_lifecycle() {
    // Create temporary directory for test database
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_ledger.db");

    // Create ProofLedger
    let ledger = ProofLedger::new(&db_path).unwrap();

    // Anchor some content
    let content1 = "The global AI market size was valued at USD 196.63 billion in 2023.";
    let url1 = "https://www.grandviewresearch.com/industry-analysis/ai-market";

    let hash1 = ledger.anchor(content1, url1, None).unwrap();

    // Verify hash format
    assert_eq!(hash1.len(), 64);
    assert!(hash1.chars().all(|c| c.is_ascii_hexdigit()));

    // Retrieve anchor
    let anchor = ledger.get_anchor(&hash1).unwrap();
    assert_eq!(anchor.url, url1);
    assert!(anchor.content_snippet.contains("196.63 billion"));

    // Verify original content
    let result = ledger.verify(&hash1, content1).unwrap();
    assert!(result.verified);

    // Detect drift
    let modified_content = "The global AI market size was valued at USD 250 billion in 2024.";
    let drift_result = ledger.verify(&hash1, modified_content).unwrap();
    assert!(!drift_result.verified);
    assert!(drift_result.message.contains("DRIFT DETECTED"));

    // Close and reopen database
    drop(ledger);

    let ledger2 = ProofLedger::new(&db_path).unwrap();

    // Verify persistence
    let anchor2 = ledger2.get_anchor(&hash1).unwrap();
    assert_eq!(anchor2.url, url1);

    // Cleanup is automatic with tempdir
}

#[test]
fn test_multiple_anchors_same_url() {
    let ledger = ProofLedger::in_memory().unwrap();

    let url = "https://example.com/article";

    // Anchor multiple pieces of content from same URL
    let hash1 = ledger
        .anchor(
            "First claim from the article",
            url,
            Some(r#"{"section": "introduction"}"#.to_string()),
        )
        .unwrap();

    let hash2 = ledger
        .anchor(
            "Second claim from the article",
            url,
            Some(r#"{"section": "methodology"}"#.to_string()),
        )
        .unwrap();

    let hash3 = ledger
        .anchor(
            "Third claim from the article",
            url,
            Some(r#"{"section": "results"}"#.to_string()),
        )
        .unwrap();

    // All should be different hashes
    assert_ne!(hash1, hash2);
    assert_ne!(hash2, hash3);
    assert_ne!(hash1, hash3);

    // List all anchors from this URL
    let anchors = ledger.list_by_url(url).unwrap();
    assert_eq!(anchors.len(), 3);

    // All should have the same URL
    assert!(anchors.iter().all(|a| a.url == url));

    // Should have different content snippets
    let snippets: Vec<_> = anchors.iter().map(|a| &a.content_snippet).collect();
    assert!(snippets.iter().any(|s| s.contains("First claim")));
    assert!(snippets.iter().any(|s| s.contains("Second claim")));
    assert!(snippets.iter().any(|s| s.contains("Third claim")));
}

#[test]
fn test_idempotent_anchoring() {
    let ledger = ProofLedger::in_memory().unwrap();

    let content = "Idempotent test content";
    let url = "https://example.com";

    // Anchor same content multiple times
    let hash1 = ledger.anchor(content, url, None).unwrap();
    let hash2 = ledger.anchor(content, url, None).unwrap();
    let hash3 = ledger.anchor(content, url, None).unwrap();

    // All should be the same hash
    assert_eq!(hash1, hash2);
    assert_eq!(hash2, hash3);

    // Should only have one entry in the ledger
    assert_eq!(ledger.count().unwrap(), 1);
}

#[test]
fn test_verification_workflow() {
    let ledger = ProofLedger::in_memory().unwrap();

    // Simulate a research workflow
    struct ResearchClaim {
        text: &'static str,
        url: &'static str,
    }

    let claims = vec![
        ResearchClaim {
            text: "GPT-4 achieved 86.4% on MMLU benchmark",
            url: "https://arxiv.org/abs/2303.08774",
        },
        ResearchClaim {
            text: "Chain-of-thought prompting improves reasoning by 15%",
            url: "https://arxiv.org/abs/2201.11903",
        },
        ResearchClaim {
            text: "RAPTOR improves long-context QA by 20%",
            url: "https://arxiv.org/abs/2401.18059",
        },
    ];

    // Anchor all claims
    let mut hashes = Vec::new();
    for claim in &claims {
        let metadata = serde_json::json!({
            "type": "benchmark_result",
            "source": "arxiv_paper"
        })
        .to_string();

        let hash = ledger
            .anchor(claim.text, claim.url, Some(metadata))
            .unwrap();
        hashes.push(hash);
    }

    // Verify all claims
    for (i, hash) in hashes.iter().enumerate() {
        let result = ledger.verify(hash, claims[i].text).unwrap();
        assert!(result.verified);
        assert_eq!(result.anchor.url, claims[i].url);
    }

    // Simulate content change (drift)
    let modified_claim = "GPT-4 achieved 90.0% on MMLU benchmark"; // Different number
    let drift_result = ledger.verify(&hashes[0], modified_claim).unwrap();
    assert!(!drift_result.verified);

    // Verify we can still get all claims
    assert_eq!(ledger.count().unwrap(), 3);
}

#[test]
fn test_hash_collision_resistance() {
    let ledger = ProofLedger::in_memory().unwrap();

    // These strings are similar but should produce different hashes
    let content1 = "The market grew by 5.0%";
    let content2 = "The market grew by 5.1%";
    let content3 = "The market grew by 5.0% "; // Trailing space

    let hash1 = ledger
        .anchor(content1, "https://example.com", None)
        .unwrap();
    let hash2 = ledger
        .anchor(content2, "https://example.com", None)
        .unwrap();
    let hash3 = ledger
        .anchor(content3, "https://example.com", None)
        .unwrap();

    // All should be different
    assert_ne!(hash1, hash2);
    assert_ne!(hash2, hash3);
    assert_ne!(hash1, hash3);

    // Should have 3 different entries
    assert_eq!(ledger.count().unwrap(), 3);
}

#[test]
fn test_large_content_handling() {
    let ledger = ProofLedger::in_memory().unwrap();

    // Create content larger than snippet size (200 chars)
    let large_content = "A".repeat(1000) + " This is the important part at the end.";

    let hash = ledger
        .anchor(&large_content, "https://example.com", None)
        .unwrap();

    // Retrieve anchor
    let anchor = ledger.get_anchor(&hash).unwrap();

    // Snippet should be truncated
    assert!(anchor.content_snippet.len() <= 204); // 200 + "..."
    assert!(anchor.content_snippet.ends_with("..."));

    // But verification should still work with full content
    let result = ledger.verify(&hash, &large_content).unwrap();
    assert!(result.verified);

    // Modified large content should fail verification
    let modified = "B".repeat(1000) + " This is the important part at the end.";
    let drift = ledger.verify(&hash, &modified).unwrap();
    assert!(!drift.verified);
}

#[test]
fn test_metadata_json_roundtrip() {
    let ledger = ProofLedger::in_memory().unwrap();

    let metadata = serde_json::json!({
        "type": "market_stat",
        "confidence": 0.95,
        "source": "grand_view_research",
        "date": "2023-12-15",
        "tags": ["AI", "market", "statistics"]
    })
    .to_string();

    let hash = ledger
        .anchor(
            "Content with rich metadata",
            "https://example.com",
            Some(metadata.clone()),
        )
        .unwrap();

    let anchor = ledger.get_anchor(&hash).unwrap();

    // Metadata should roundtrip
    assert_eq!(anchor.metadata, Some(metadata.clone()));

    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&metadata).unwrap();
    assert_eq!(parsed["confidence"], 0.95);
    assert_eq!(parsed["tags"][0], "AI");
}

#[test]
fn test_timestamp_ordering() {
    let ledger = ProofLedger::in_memory().unwrap();
    let url = "https://example.com";

    // Anchor content sequentially
    let hash1 = ledger.anchor("First", url, None).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));

    let hash2 = ledger.anchor("Second", url, None).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));

    let hash3 = ledger.anchor("Third", url, None).unwrap();

    // Retrieve anchors
    let anchor1 = ledger.get_anchor(&hash1).unwrap();
    let anchor2 = ledger.get_anchor(&hash2).unwrap();
    let anchor3 = ledger.get_anchor(&hash3).unwrap();

    // Timestamps should be in order
    assert!(anchor1.timestamp < anchor2.timestamp);
    assert!(anchor2.timestamp < anchor3.timestamp);

    // list_by_url should return in descending timestamp order
    let anchors = ledger.list_by_url(url).unwrap();
    assert_eq!(anchors.len(), 3);

    // Most recent first
    assert!(anchors[0].content_snippet.contains("Third"));
    assert!(anchors[2].content_snippet.contains("First"));
}

#[test]
fn test_error_handling() {
    let ledger = ProofLedger::in_memory().unwrap();

    // Test hash not found
    let fake_hash = "0".repeat(64);
    let result = ledger.get_anchor(&fake_hash);
    assert!(result.is_err());

    match result {
        Err(ProofLedgerError::Database(_)) => {} // Expected
        _ => panic!("Expected Database error for non-existent hash"),
    }
}

#[test]
fn test_unicode_content() {
    let ledger = ProofLedger::in_memory().unwrap();

    // Test with various Unicode characters
    let content = "市场规模达到 1966.3亿美元 🚀 Æther φιλοσοφία";
    let hash = ledger.anchor(content, "https://example.com", None).unwrap();

    // Verify Unicode content
    let result = ledger.verify(&hash, content).unwrap();
    assert!(result.verified);

    // Retrieve and check snippet
    let anchor = ledger.get_anchor(&hash).unwrap();
    assert!(anchor.content_snippet.contains("市场"));
    assert!(anchor.content_snippet.contains("🚀"));
}

#[test]
fn test_empty_and_whitespace_content() {
    let ledger = ProofLedger::in_memory().unwrap();

    // Empty string
    let hash1 = ledger
        .anchor("", "https://example.com/empty", None)
        .unwrap();
    let anchor1 = ledger.get_anchor(&hash1).unwrap();
    assert_eq!(anchor1.content_snippet, "");

    // Whitespace only
    let hash2 = ledger
        .anchor("   \n\t  ", "https://example.com/whitespace", None)
        .unwrap();
    let anchor2 = ledger.get_anchor(&hash2).unwrap();
    assert_eq!(anchor2.content_snippet, "   \n\t  ");

    // These should have different hashes
    assert_ne!(hash1, hash2);
}
