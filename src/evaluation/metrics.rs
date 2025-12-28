//! Core retrieval evaluation metrics.
//!
//! Implements standard IR metrics:
//! - Recall@K
//! - Precision@K
//! - NDCG@K
//! - MRR
//! - MAP

use std::collections::{HashMap, HashSet};

/// Result for a single query
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Query ID
    pub query_id: String,
    /// Retrieved document IDs in ranked order
    pub retrieved_ids: Vec<String>,
    /// Ground truth: set of relevant document IDs
    pub relevant_ids: HashSet<String>,
    /// Optional: graded relevance scores (for NDCG)
    pub relevance_grades: Option<HashMap<String, f64>>,
}

impl QueryResult {
    pub fn new(
        query_id: impl Into<String>,
        retrieved: Vec<String>,
        relevant: HashSet<String>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            retrieved_ids: retrieved,
            relevant_ids: relevant,
            relevance_grades: None,
        }
    }

    pub fn with_grades(mut self, grades: HashMap<String, f64>) -> Self {
        self.relevance_grades = Some(grades);
        self
    }
}

/// Evaluation result for a single query
#[derive(Debug, Clone, Default)]
pub struct EvaluationResult {
    pub query_id: String,
    pub recall: f64,
    pub precision: f64,
    pub ndcg: f64,
    pub mrr: f64,
    pub ap: f64,
    pub k: usize,
}

/// Retrieval metrics for a single query
#[derive(Debug, Clone)]
pub struct RetrievalMetrics {
    pub recall_at_k: HashMap<usize, f64>,
    pub precision_at_k: HashMap<usize, f64>,
    pub ndcg_at_k: HashMap<usize, f64>,
    pub mrr: f64,
    pub map: f64,
}

impl RetrievalMetrics {
    /// Compute all metrics for given K values
    pub fn compute_all(
        retrieved: &[String],
        relevant: &HashSet<String>,
        k_values: &[usize],
    ) -> Self {
        let mut recall_at_k = HashMap::new();
        let mut precision_at_k = HashMap::new();
        let mut ndcg_at_k = HashMap::new();

        for &k in k_values {
            recall_at_k.insert(k, recall_at_k_impl(retrieved, relevant, k));
            precision_at_k.insert(k, precision_at_k_impl(retrieved, relevant, k));
            ndcg_at_k.insert(k, ndcg_at_k_binary(retrieved, relevant, k));
        }

        let mrr = mean_reciprocal_rank_single(retrieved, relevant);
        let map = average_precision_impl(retrieved, relevant);

        Self {
            recall_at_k,
            precision_at_k,
            ndcg_at_k,
            mrr,
            map,
        }
    }

    /// Compute metrics for a single K value
    pub fn compute(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> EvaluationResult {
        EvaluationResult {
            query_id: String::new(),
            recall: recall_at_k_impl(retrieved, relevant, k),
            precision: precision_at_k_impl(retrieved, relevant, k),
            ndcg: ndcg_at_k_binary(retrieved, relevant, k),
            mrr: mean_reciprocal_rank_single(retrieved, relevant),
            ap: average_precision_impl(retrieved, relevant),
            k,
        }
    }
}

/// Recall@K: Proportion of relevant documents retrieved in top-K
///
/// Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|
///
/// # Arguments
/// * `retrieved` - Document IDs in ranked order
/// * `relevant` - Set of relevant document IDs
/// * `k` - Number of top results to consider
///
/// # Returns
/// Recall value between 0.0 and 1.0
pub fn recall_at_k(retrieved: &[impl AsRef<str>], relevant: &HashSet<String>, k: usize) -> f64 {
    let retrieved_str: Vec<String> = retrieved.iter().map(|s| s.as_ref().to_string()).collect();
    recall_at_k_impl(&retrieved_str, relevant, k)
}

fn recall_at_k_impl(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let top_k: HashSet<_> = retrieved.iter().take(k).cloned().collect();
    let hits = relevant.intersection(&top_k).count();

    hits as f64 / relevant.len() as f64
}

/// Precision@K: Proportion of top-K documents that are relevant
///
/// Precision@K = |Relevant ∩ Retrieved@K| / K
///
/// # Arguments
/// * `retrieved` - Document IDs in ranked order
/// * `relevant` - Set of relevant document IDs
/// * `k` - Number of top results to consider
///
/// # Returns
/// Precision value between 0.0 and 1.0
pub fn precision_at_k(retrieved: &[impl AsRef<str>], relevant: &HashSet<String>, k: usize) -> f64 {
    let retrieved_str: Vec<String> = retrieved.iter().map(|s| s.as_ref().to_string()).collect();
    precision_at_k_impl(&retrieved_str, relevant, k)
}

fn precision_at_k_impl(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }

    let actual_k = k.min(retrieved.len());
    if actual_k == 0 {
        return 0.0;
    }

    let hits = retrieved
        .iter()
        .take(actual_k)
        .filter(|doc| relevant.contains(*doc))
        .count();

    hits as f64 / actual_k as f64
}

/// NDCG@K: Normalized Discounted Cumulative Gain
///
/// DCG@K = Σ(i=1 to K) rel_i / log2(i+1)
/// NDCG@K = DCG@K / IDCG@K
///
/// # Arguments
/// * `retrieved` - Document IDs in ranked order
/// * `relevant` - Set of relevant document IDs (binary relevance)
/// * `k` - Number of top results to consider
///
/// # Returns
/// NDCG value between 0.0 and 1.0
pub fn ndcg_at_k(retrieved: &[impl AsRef<str>], relevant: &HashSet<String>, k: usize) -> f64 {
    let retrieved_str: Vec<String> = retrieved.iter().map(|s| s.as_ref().to_string()).collect();
    ndcg_at_k_binary(&retrieved_str, relevant, k)
}

fn ndcg_at_k_binary(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    // DCG: sum of relevance / log2(rank + 1)
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, doc)| relevant.contains(*doc))
        .map(|(i, _)| 1.0 / (i as f64 + 2.0).log2()) // log2(i+2) = log2(rank+1) where rank is 1-indexed
        .sum();

    // IDCG: ideal DCG (all relevant docs at top)
    let num_relevant_in_k = k.min(relevant.len());
    let idcg: f64 = (0..num_relevant_in_k)
        .map(|i| 1.0 / (i as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        return 0.0;
    }

    dcg / idcg
}

/// NDCG@K with graded relevance
///
/// # Arguments
/// * `retrieved` - Document IDs in ranked order
/// * `relevance_grades` - Map of document ID to relevance grade (0.0 to 1.0 or higher)
/// * `k` - Number of top results to consider
pub fn ndcg_at_k_graded(
    retrieved: &[String],
    relevance_grades: &HashMap<String, f64>,
    k: usize,
) -> f64 {
    if relevance_grades.is_empty() {
        return 0.0;
    }

    // DCG with graded relevance
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, doc)| {
            let rel = relevance_grades.get(doc).copied().unwrap_or(0.0);
            (2_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum();

    // IDCG: sort by relevance grade descending
    let mut sorted_grades: Vec<f64> = relevance_grades.values().copied().collect();
    sorted_grades.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let idcg: f64 = sorted_grades
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| (2_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        return 0.0;
    }

    dcg / idcg
}

/// MRR: Mean Reciprocal Rank (for single query)
///
/// RR = 1 / rank of first relevant document
///
/// # Arguments
/// * `retrieved` - Document IDs in ranked order
/// * `relevant` - Set of relevant document IDs
///
/// # Returns
/// Reciprocal rank (1.0 if first result is relevant, 0.5 if second, etc.)
pub fn mean_reciprocal_rank(results: &[QueryResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    let sum: f64 = results
        .iter()
        .map(|r| mean_reciprocal_rank_single(&r.retrieved_ids, &r.relevant_ids))
        .sum();

    sum / results.len() as f64
}

fn mean_reciprocal_rank_single(retrieved: &[String], relevant: &HashSet<String>) -> f64 {
    for (i, doc) in retrieved.iter().enumerate() {
        if relevant.contains(doc) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Average Precision for a single query
///
/// AP = (1/|Relevant|) × Σ(k=1 to n) Precision@k × rel(k)
///
/// # Arguments
/// * `retrieved` - Document IDs in ranked order
/// * `relevant` - Set of relevant document IDs
pub fn average_precision(retrieved: &[impl AsRef<str>], relevant: &HashSet<String>) -> f64 {
    let retrieved_str: Vec<String> = retrieved.iter().map(|s| s.as_ref().to_string()).collect();
    average_precision_impl(&retrieved_str, relevant)
}

fn average_precision_impl(retrieved: &[String], relevant: &HashSet<String>) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let mut num_relevant_seen = 0;
    let mut sum_precision = 0.0;

    for (i, doc) in retrieved.iter().enumerate() {
        if relevant.contains(doc) {
            num_relevant_seen += 1;
            // Precision at this position
            let precision = num_relevant_seen as f64 / (i as f64 + 1.0);
            sum_precision += precision;
        }
    }

    sum_precision / relevant.len() as f64
}

/// Mean Average Precision across multiple queries
///
/// MAP = (1/|Q|) × Σ(q=1 to |Q|) AP(q)
pub fn mean_average_precision(results: &[QueryResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    let sum: f64 = results
        .iter()
        .map(|r| average_precision_impl(&r.retrieved_ids, &r.relevant_ids))
        .sum();

    sum / results.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_relevant(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    fn make_retrieved(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_recall_at_k_perfect() {
        let retrieved = make_retrieved(&["a", "b", "c", "d", "e"]);
        let relevant = make_relevant(&["a", "b", "c"]);

        assert_eq!(recall_at_k_impl(&retrieved, &relevant, 3), 1.0);
        assert_eq!(recall_at_k_impl(&retrieved, &relevant, 5), 1.0);
    }

    #[test]
    fn test_recall_at_k_partial() {
        let retrieved = make_retrieved(&["a", "x", "b", "y", "c"]);
        let relevant = make_relevant(&["a", "b", "c"]);

        // At k=1, only "a" is retrieved (1/3 relevant)
        assert!((recall_at_k_impl(&retrieved, &relevant, 1) - 1.0 / 3.0).abs() < 0.001);

        // At k=3, "a" and "b" are retrieved (2/3 relevant)
        assert!((recall_at_k_impl(&retrieved, &relevant, 3) - 2.0 / 3.0).abs() < 0.001);

        // At k=5, all are retrieved
        assert_eq!(recall_at_k_impl(&retrieved, &relevant, 5), 1.0);
    }

    #[test]
    fn test_recall_at_k_none() {
        let retrieved = make_retrieved(&["x", "y", "z"]);
        let relevant = make_relevant(&["a", "b", "c"]);

        assert_eq!(recall_at_k_impl(&retrieved, &relevant, 3), 0.0);
    }

    #[test]
    fn test_recall_at_k_empty_relevant() {
        let retrieved = make_retrieved(&["a", "b", "c"]);
        let relevant = HashSet::new();

        assert_eq!(recall_at_k_impl(&retrieved, &relevant, 3), 0.0);
    }

    #[test]
    fn test_precision_at_k_perfect() {
        let retrieved = make_retrieved(&["a", "b", "c"]);
        let relevant = make_relevant(&["a", "b", "c", "d", "e"]);

        assert_eq!(precision_at_k_impl(&retrieved, &relevant, 3), 1.0);
    }

    #[test]
    fn test_precision_at_k_partial() {
        let retrieved = make_retrieved(&["a", "x", "b", "y", "c"]);
        let relevant = make_relevant(&["a", "b", "c"]);

        // At k=1: 1 relevant out of 1
        assert_eq!(precision_at_k_impl(&retrieved, &relevant, 1), 1.0);

        // At k=2: 1 relevant out of 2
        assert_eq!(precision_at_k_impl(&retrieved, &relevant, 2), 0.5);

        // At k=5: 3 relevant out of 5
        assert_eq!(precision_at_k_impl(&retrieved, &relevant, 5), 0.6);
    }

    #[test]
    fn test_mrr_first_position() {
        let retrieved = make_retrieved(&["a", "b", "c"]);
        let relevant = make_relevant(&["a"]);

        assert_eq!(mean_reciprocal_rank_single(&retrieved, &relevant), 1.0);
    }

    #[test]
    fn test_mrr_second_position() {
        let retrieved = make_retrieved(&["x", "a", "c"]);
        let relevant = make_relevant(&["a"]);

        assert_eq!(mean_reciprocal_rank_single(&retrieved, &relevant), 0.5);
    }

    #[test]
    fn test_mrr_third_position() {
        let retrieved = make_retrieved(&["x", "y", "a"]);
        let relevant = make_relevant(&["a"]);

        assert!((mean_reciprocal_rank_single(&retrieved, &relevant) - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_mrr_not_found() {
        let retrieved = make_retrieved(&["x", "y", "z"]);
        let relevant = make_relevant(&["a"]);

        assert_eq!(mean_reciprocal_rank_single(&retrieved, &relevant), 0.0);
    }

    #[test]
    fn test_ndcg_perfect() {
        let retrieved = make_retrieved(&["a", "b", "c", "x", "y"]);
        let relevant = make_relevant(&["a", "b", "c"]);

        // Perfect ranking: all relevant docs at top
        assert!((ndcg_at_k_binary(&retrieved, &relevant, 5) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ndcg_partial() {
        let retrieved = make_retrieved(&["x", "a", "y", "b", "c"]);
        let relevant = make_relevant(&["a", "b", "c"]);

        // Not perfect: relevant docs are at positions 2, 4, 5
        let ndcg = ndcg_at_k_binary(&retrieved, &relevant, 5);
        assert!(ndcg > 0.0 && ndcg < 1.0);
    }

    #[test]
    fn test_average_precision() {
        let retrieved = make_retrieved(&["a", "x", "b", "y", "c"]);
        let relevant = make_relevant(&["a", "b", "c"]);

        // AP = (1/3) * (1/1 + 2/3 + 3/5) = (1/3) * (1 + 0.667 + 0.6) ≈ 0.756
        let ap = average_precision_impl(&retrieved, &relevant);
        assert!(ap > 0.7 && ap < 0.8);
    }

    #[test]
    fn test_average_precision_perfect() {
        let retrieved = make_retrieved(&["a", "b", "c", "x", "y"]);
        let relevant = make_relevant(&["a", "b", "c"]);

        // AP = (1/3) * (1/1 + 2/2 + 3/3) = (1/3) * 3 = 1.0
        let ap = average_precision_impl(&retrieved, &relevant);
        assert_eq!(ap, 1.0);
    }

    #[test]
    fn test_retrieval_metrics_compute() {
        let retrieved = make_retrieved(&["a", "b", "x", "c", "y"]);
        let relevant = make_relevant(&["a", "b", "c"]);

        let metrics = RetrievalMetrics::compute_all(&retrieved, &relevant, &[5, 10]);

        assert!(metrics.recall_at_k.contains_key(&5));
        assert!(metrics.precision_at_k.contains_key(&5));
        assert!(metrics.ndcg_at_k.contains_key(&5));
        assert!(metrics.mrr > 0.0);
        assert!(metrics.map > 0.0);
    }
}
