//! Protocol Delta Demo
//!
//! Demonstrates the ProofLedger immutable citation ledger.
//!
//! Run with:
//! ```bash
//! cargo run --example protocol_delta_demo
//! ```

use reasonkit::verification::ProofLedger;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  PROTOCOL DELTA: PROOFLEDGER DEMONSTRATION");
    println!("  \"We do not quote the wind. We quote the stone.\"");
    println!("═══════════════════════════════════════════════════════════\n");

    // Create in-memory ledger for demo
    let ledger = ProofLedger::in_memory()?;

    // Simulate research workflow
    println!("📚 Research Phase: Anchoring Claims\n");

    let claims = vec![
        (
            "GPT-4 achieved 86.4% accuracy on the MMLU benchmark.",
            "https://arxiv.org/abs/2303.08774",
            "benchmark_result",
        ),
        (
            "Chain-of-thought prompting improves reasoning performance by approximately 15%.",
            "https://arxiv.org/abs/2201.11903",
            "methodology",
        ),
        (
            "RAPTOR demonstrates a 20% improvement on QuALITY benchmark for long-context QA.",
            "https://arxiv.org/abs/2401.18059",
            "performance_metric",
        ),
    ];

    let mut hashes = Vec::new();

    for (i, (text, url, claim_type)) in claims.iter().enumerate() {
        let metadata = serde_json::json!({
            "type": claim_type,
            "source": "arxiv_paper",
            "confidence": 0.95,
            "extracted_at": "2025-12-23T00:00:00Z"
        })
        .to_string();

        let hash = ledger.anchor(text, url, Some(metadata))?;

        println!("  ✓ Anchored claim {} (hash: {}...)", i + 1, &hash[..8]);
        hashes.push((hash, *text, *url));
    }

    println!("\n📄 Generated Research Report\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Generate report with citations
    for (i, (_, text, _)) in hashes.iter().enumerate() {
        println!("{}. {} [{}]\n", i + 1, text, i + 1);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("References (Protocol Delta Format):\n");

    for (i, (hash, _, url)) in hashes.iter().enumerate() {
        println!(
            "[{}] sha256:{}... (Verified 2025-12-23)",
            i + 1,
            &hash[..12]
        );
        println!("    → {}\n", url);
    }

    // Demonstrate verification
    println!("\n🔍 Verification Phase: Checking Content Integrity\n");

    for (i, (hash, text, _)) in hashes.iter().enumerate() {
        let result = ledger.verify(hash, text)?;
        if result.verified {
            println!("  ✓ Claim {} verified (hash matches)", i + 1);
        } else {
            println!("  ✗ Claim {} DRIFT DETECTED!", i + 1);
        }
    }

    // Demonstrate drift detection
    println!("\n⚠️  Drift Detection: Simulating Content Change\n");

    let modified_claim = "GPT-4 achieved 90.0% accuracy on the MMLU benchmark."; // Changed!
    let drift_result = ledger.verify(&hashes[0].0, modified_claim)?;

    if !drift_result.verified {
        println!("  ✗ DRIFT DETECTED!");
        println!(
            "    Original hash: {}...",
            &drift_result.original_hash[..12]
        );
        println!("    Current hash:  {}...", &drift_result.current_hash[..12]);
        println!("    Message: {}", drift_result.message);
    }

    // Query by URL
    println!("\n📊 Query Phase: List All Citations from Source\n");

    let url = "https://arxiv.org/abs/2303.08774";
    let anchors = ledger.list_by_url(url)?;

    println!("  Found {} anchor(s) from {}:", anchors.len(), url);
    for anchor in anchors {
        println!("    - Hash: {}...", &anchor.hash[..12]);
        println!("      Snippet: {}", anchor.content_snippet);
        println!("      Timestamp: {}\n", anchor.timestamp);
    }

    // Statistics
    println!("📈 Ledger Statistics\n");
    println!("  Total anchors: {}", ledger.count()?);
    println!("  Ledger path: {:?}", ledger.ledger_path());

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Demo complete! Immutable citations established.");
    println!("═══════════════════════════════════════════════════════════\n");

    Ok(())
}
