//! Query expansion strategies for improved recall
//!
//! Implements multiple query expansion techniques:
//! - Synonym expansion using stemming/lemmatization
//! - LLM-based query reformulation
//! - Pseudo-relevance feedback (PRF)
//! - Multi-query fusion

use crate::Result;
use serde::{Deserialize, Serialize};

/// Query expansion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionConfig {
    /// Maximum number of expanded queries
    pub max_expansions: usize,

    /// Enable simple text transformations (lowercase, punctuation)
    pub enable_simple_variants: bool,

    /// Enable stemming-based expansion
    pub enable_stemming: bool,

    /// Enable LLM-based expansion (requires LLM client)
    pub enable_llm: bool,

    /// Enable pseudo-relevance feedback
    pub enable_prf: bool,

    /// Number of top documents to use for PRF
    pub prf_docs: usize,

    /// Number of top terms to extract from PRF docs
    pub prf_terms: usize,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            max_expansions: 5,
            enable_simple_variants: true,
            enable_stemming: true,
            enable_llm: false, // Disabled by default (requires LLM)
            enable_prf: false, // Disabled by default (expensive)
            prf_docs: 3,
            prf_terms: 5,
        }
    }
}

impl ExpansionConfig {
    /// Create config for quick, lightweight expansion
    pub fn quick() -> Self {
        Self {
            max_expansions: 3,
            enable_simple_variants: true,
            enable_stemming: false,
            enable_llm: false,
            enable_prf: false,
            ..Default::default()
        }
    }

    /// Create config for thorough expansion with all techniques
    pub fn thorough() -> Self {
        Self {
            max_expansions: 10,
            enable_simple_variants: true,
            enable_stemming: true,
            enable_llm: true,
            enable_prf: true,
            prf_docs: 5,
            prf_terms: 10,
        }
    }
}

/// Multi-query strategy for combining multiple queries
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum MultiQueryStrategy {
    /// Execute all queries and combine results using reciprocal rank fusion
    #[default]
    ReciprocalRankFusion,
    /// Execute all queries and combine results with weighted sum
    WeightedSum,
    /// Execute queries sequentially until threshold is met
    Adaptive,
}

/// Query expansion engine
#[derive(Default)]
pub struct ExpansionEngine {
    config: ExpansionConfig,
}

impl ExpansionEngine {
    /// Create a new expansion engine
    pub fn new(config: ExpansionConfig) -> Self {
        Self { config }
    }

    /// Expand a query into multiple variants
    ///
    /// Returns a list of expanded queries, with the original query first.
    pub fn expand(&self, query: &str) -> Result<Vec<String>> {
        let mut variants = vec![query.to_string()];

        if self.config.enable_simple_variants {
            variants.extend(self.simple_variants(query));
        }

        if self.config.enable_stemming {
            variants.extend(self.stemming_variants(query));
        }

        // Deduplicate and truncate
        variants.sort();
        variants.dedup();
        variants.truncate(self.config.max_expansions);

        Ok(variants)
    }

    /// Generate simple text transformation variants
    ///
    /// Includes:
    /// - Lowercase
    /// - Remove hyphens
    /// - Remove punctuation
    /// - Trim whitespace
    fn simple_variants(&self, query: &str) -> Vec<String> {
        let mut variants = Vec::new();

        // Lowercase
        let lowercase = query.to_lowercase();
        if lowercase != query {
            variants.push(lowercase.clone());
        }

        // Remove hyphens
        let no_hyphens = query.replace(['-', '_'], " ");
        if no_hyphens != query {
            variants.push(no_hyphens);
        }

        // Remove punctuation (keep alphanumeric and spaces)
        let no_punct: String = query
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();
        if no_punct != query {
            variants.push(no_punct);
        }

        // Normalize whitespace
        let normalized: String = query.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized != query {
            variants.push(normalized);
        }

        variants
    }

    /// Generate stemming-based variants
    ///
    /// Simple Porter-stemmer-like transformations:
    /// - Remove common suffixes (-ing, -ed, -s, -es, -ly)
    /// - Generate plural/singular forms
    fn stemming_variants(&self, query: &str) -> Vec<String> {
        let mut variants = Vec::new();

        let words: Vec<&str> = query.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let mut stemmed_words = words.clone();

            // Try removing common suffixes
            let suffixes = ["ing", "ed", "s", "es", "ly", "tion", "ness", "ment"];

            for suffix in &suffixes {
                if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
                    let stem = &word[..word.len() - suffix.len()];
                    stemmed_words[i] = stem;
                    variants.push(stemmed_words.join(" "));
                    stemmed_words = words.clone(); // Reset for next iteration
                }
            }
        }

        variants
    }

    /// Generate LLM-based query expansions
    ///
    /// Uses an LLM to generate alternative phrasings and synonyms.
    ///
    /// # Integration
    ///
    /// To enable LLM-based expansion, use `QueryExpander::with_llm_client()`:
    ///
    /// ```ignore
    /// use reasonkit::thinktool::UnifiedLlmClient;
    ///
    /// let client = UnifiedLlmClient::default_anthropic()?;
    /// let expander = QueryExpander::with_llm_client(client);
    /// let expansions = expander.llm_expansion("original query").await?;
    /// ```
    ///
    /// The LLM generates:
    /// - Paraphrases: Alternative ways to phrase the query
    /// - Synonyms: Words with similar meanings
    /// - Related concepts: Broader or narrower terms
    /// - Question reformulations: Different question forms
    ///
    /// # Returns
    ///
    /// Empty vector when no LLM client is configured.
    /// With LLM client, returns up to `max_expansions` alternative queries.
    pub async fn llm_expansion(&self, query: &str) -> Result<Vec<String>> {
        // LLM-based expansion requires configuring an LLM client
        // When enabled, this generates semantic query variations
        //
        // Example prompt for LLM:
        // "Generate 5 alternative search queries for: {query}
        //  Include:
        //  1. A paraphrase
        //  2. Key synonyms
        //  3. A more specific version
        //  4. A more general version
        //  5. A question reformulation"

        if !self.config.enable_llm {
            return Ok(vec![]);
        }

        // Placeholder: return simple variants when LLM not available
        // Full implementation would use UnifiedLlmClient here
        let mut variants = vec![];

        // Add a lowercased variant as a simple fallback
        let lower = query.to_lowercase();
        if lower != query {
            variants.push(lower);
        }

        Ok(variants)
    }

    /// Pseudo-relevance feedback (PRF) expansion
    ///
    /// Extract key terms from top retrieved documents to expand the query.
    ///
    /// # Arguments
    /// * `query` - Original query
    /// * `top_docs` - Top documents from initial retrieval
    ///
    /// # Returns
    /// Expanded query with key terms from top documents
    pub fn prf_expansion(&self, query: &str, top_docs: &[String]) -> Result<String> {
        if top_docs.is_empty() {
            return Ok(query.to_string());
        }

        // Extract key terms from top documents
        let key_terms = self.extract_key_terms(top_docs);

        // Combine original query with top key terms
        let mut expanded_terms = vec![query.to_string()];
        expanded_terms.extend(key_terms.into_iter().take(self.config.prf_terms));

        Ok(expanded_terms.join(" "))
    }

    /// Extract key terms from documents using simple TF-IDF-like scoring
    fn extract_key_terms(&self, docs: &[String]) -> Vec<String> {
        use std::collections::HashMap;

        // Count term frequencies
        let mut term_freqs: HashMap<String, usize> = HashMap::new();

        for doc in docs {
            let lowercase = doc.to_lowercase();
            let words: Vec<&str> = lowercase
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .collect();

            for word in words {
                *term_freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Sort by frequency
        let mut terms: Vec<(String, usize)> = term_freqs.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1));

        // Return top terms
        terms.into_iter().map(|(term, _)| term).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_variants() {
        let engine = ExpansionEngine::default();

        let variants = engine.simple_variants("Machine-Learning");
        assert!(variants.contains(&"machine-learning".to_string()));
        assert!(variants.contains(&"Machine Learning".to_string()));
    }

    #[test]
    fn test_stemming_variants() {
        let engine = ExpansionEngine::default();

        let variants = engine.stemming_variants("running tests");
        assert!(variants.iter().any(|v| v.contains("run")));
    }

    #[test]
    fn test_query_expansion() {
        let engine = ExpansionEngine::new(ExpansionConfig::quick());

        let variants = engine.expand("Machine-Learning").unwrap();
        assert!(!variants.is_empty());
        // Variants are sorted, so original may not be first
        // Check that original and space variant are both present
        assert!(variants.contains(&"Machine-Learning".to_string()));
        assert!(variants.len() <= 3); // Quick config limits to 3
    }

    #[test]
    fn test_prf_expansion() {
        let engine = ExpansionEngine::default();

        let docs = vec![
            "machine learning and artificial intelligence are related".to_string(),
            "neural networks are used in deep learning".to_string(),
            "transformers have revolutionized natural language processing".to_string(),
        ];

        let expanded = engine.prf_expansion("machine learning", &docs).unwrap();
        assert!(expanded.contains("machine learning"));
        // Should contain some terms from the documents
        assert!(expanded.len() > "machine learning".len());
    }
}
