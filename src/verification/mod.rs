//! Verification Module - Protocol Delta Implementation
//!
//! This module implements Protocol Delta's verification capabilities,
//! ensuring research claims remain anchored to immutable sources.
//!
//! ## Components
//!
//! - **ProofLedger**: Immutable citation ledger with cryptographic binding
//! - **Anchor**: Snapshot of content at a specific point in time
//!
//! ## Philosophy
//!
//! > "We do not quote the wind. We quote the stone."
//!
//! Protocol Delta replaces weak URL citations with cryptographically-bound
//! anchors that can detect content drift over time.
//!
//! ## Usage
//!
//! ```rust
//! use reasonkit::verification::ProofLedger;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a ledger
//! let ledger = ProofLedger::in_memory()?;
//!
//! // Anchor a claim
//! let content = "The global AI market was valued at $196.63B in 2023.";
//! let hash = ledger.anchor(
//!     content,
//!     "https://example.com/ai-market",
//!     None
//! )?;
//!
//! // Later: verify the content hasn't drifted
//! let result = ledger.verify(&hash, content)?;
//! assert!(result.verified);
//! # Ok(())
//! # }
//! ```
//!
//! ## Citation Format
//!
//! Instead of:
//! ```text
//! The market grew by 5% [1].
//! [1] https://finance.yahoo.com/...
//! ```
//!
//! Use:
//! ```text
//! The market grew by 5% [1].
//! [1] sha256:8f4a... (Verified 2025-12-23) → https://finance.yahoo.com/...
//! ```

pub mod proof_ledger;

// Re-exports
pub use proof_ledger::{Anchor, ProofLedger, ProofLedgerError, VerificationResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify main types are accessible
        let _ledger = ProofLedger::in_memory();
    }
}
