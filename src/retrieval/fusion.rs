//! Result fusion strategies for hybrid retrieval
//!
//! Implements various fusion algorithms:
//! - RRF (Reciprocal Rank Fusion) - rank-based fusion
//! - Weighted Sum - score-based fusion (original implementation)
//! - RBF (Rank-Biased Fusion) - decay-based fusion

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Fusion strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Weighted sum of normalized scores
    WeightedSum {
        /// Weight for dense (vector) results (0.0-1.0)
        dense_weight: f32,
    },
    /// Reciprocal Rank Fusion (RRF)
    ReciprocalRankFusion {
        /// Constant k for RRF formula (typically 60)
        k: usize,
    },
    /// Rank-Biased Fusion (RBF)
    RankBiasedFusion {
        /// Persistence parameter (0.0-1.0, typically 0.8)
        rho: f32,
    },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::ReciprocalRankFusion { k: 60 }
    }
}

/// Result from a single retrieval method
#[derive(Debug, Clone)]
pub struct RankedResult {
    /// Item identifier (chunk ID)
    pub id: Uuid,
    /// Score from this retrieval method
    pub score: f32,
    /// Rank position (0-indexed)
    pub rank: usize,
}

/// Fused result combining multiple retrieval methods
#[derive(Debug, Clone)]
pub struct FusedResult {
    /// Item identifier
    pub id: Uuid,
    /// Combined fusion score
    pub fusion_score: f32,
    /// Individual scores from each method
    pub method_scores: HashMap<String, f32>,
}

/// Fusion engine for combining retrieval results
pub struct FusionEngine {
    strategy: FusionStrategy,
}

impl FusionEngine {
    /// Create a new fusion engine with the given strategy
    pub fn new(strategy: FusionStrategy) -> Self {
        Self { strategy }
    }

    /// Create with RRF strategy (recommended)
    pub fn rrf(k: usize) -> Self {
        Self::new(FusionStrategy::ReciprocalRankFusion { k })
    }

    /// Create with weighted sum strategy
    pub fn weighted(dense_weight: f32) -> Self {
        Self::new(FusionStrategy::WeightedSum { dense_weight })
    }

    /// Fuse results from multiple retrieval methods
    ///
    /// # Arguments
    /// * `results` - Map of method name to ranked results
    ///
    /// # Returns
    /// Fused results sorted by fusion score (descending)
    pub fn fuse(&self, results: HashMap<String, Vec<RankedResult>>) -> Result<Vec<FusedResult>> {
        match &self.strategy {
            FusionStrategy::WeightedSum { dense_weight } => {
                self.fuse_weighted_sum(results, *dense_weight)
            }
            FusionStrategy::ReciprocalRankFusion { k } => self.fuse_rrf(results, *k),
            FusionStrategy::RankBiasedFusion { rho } => self.fuse_rbf(results, *rho),
        }
    }

    /// Reciprocal Rank Fusion (RRF)
    ///
    /// Formula: score(d) = sum over all methods of: 1 / (k + rank(d))
    /// where k is a constant (typically 60)
    ///
    /// Reference: "Reciprocal Rank Fusion outperforms Condorcet and
    /// individual Rank Learning Methods" (Cormack et al., 2009)
    fn fuse_rrf(
        &self,
        results: HashMap<String, Vec<RankedResult>>,
        k: usize,
    ) -> Result<Vec<FusedResult>> {
        let mut scores: HashMap<Uuid, (f32, HashMap<String, f32>)> = HashMap::new();

        for (method_name, method_results) in results {
            for result in method_results {
                let rrf_score = 1.0 / (k as f32 + result.rank as f32 + 1.0);

                let (total_score, method_scores) = scores
                    .entry(result.id)
                    .or_insert_with(|| (0.0, HashMap::new()));

                *total_score += rrf_score;
                method_scores.insert(method_name.clone(), result.score);
            }
        }

        let mut fused_results: Vec<FusedResult> = scores
            .into_iter()
            .map(|(id, (fusion_score, method_scores))| FusedResult {
                id,
                fusion_score,
                method_scores,
            })
            .collect();

        // Sort by fusion score descending
        fused_results.sort_by(|a, b| {
            b.fusion_score
                .partial_cmp(&a.fusion_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Weighted sum fusion with score normalization
    ///
    /// Normalizes scores from each method to [0, 1] range using min-max normalization,
    /// then combines using weighted sum.
    fn fuse_weighted_sum(
        &self,
        results: HashMap<String, Vec<RankedResult>>,
        dense_weight: f32,
    ) -> Result<Vec<FusedResult>> {
        let sparse_weight = 1.0 - dense_weight;

        // Normalize scores for each method
        let mut normalized_results: HashMap<String, Vec<(Uuid, f32)>> = HashMap::new();

        for (method_name, method_results) in results {
            if method_results.is_empty() {
                continue;
            }

            // Find min and max scores for normalization
            let min_score = method_results
                .iter()
                .map(|r| r.score)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let max_score = method_results
                .iter()
                .map(|r| r.score)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(1.0);

            let score_range = max_score - min_score;

            let normalized: Vec<(Uuid, f32)> = if score_range > 1e-6 {
                method_results
                    .into_iter()
                    .map(|r| {
                        let norm_score = (r.score - min_score) / score_range;
                        (r.id, norm_score)
                    })
                    .collect()
            } else {
                // All scores are the same, normalize to 1.0
                method_results.into_iter().map(|r| (r.id, 1.0)).collect()
            };

            normalized_results.insert(method_name, normalized);
        }

        // Combine normalized scores
        let mut scores: HashMap<Uuid, (f32, HashMap<String, f32>)> = HashMap::new();

        for (method_name, method_results) in normalized_results {
            let weight = if method_name == "dense" {
                dense_weight
            } else {
                sparse_weight
            };

            for (id, norm_score) in method_results {
                let (total_score, method_scores) =
                    scores.entry(id).or_insert_with(|| (0.0, HashMap::new()));

                *total_score += norm_score * weight;
                method_scores.insert(method_name.clone(), norm_score);
            }
        }

        let mut fused_results: Vec<FusedResult> = scores
            .into_iter()
            .map(|(id, (fusion_score, method_scores))| FusedResult {
                id,
                fusion_score,
                method_scores,
            })
            .collect();

        fused_results.sort_by(|a, b| {
            b.fusion_score
                .partial_cmp(&a.fusion_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }

    /// Rank-Biased Fusion (RBF)
    ///
    /// Formula: score(d) = sum over all methods of: rho^rank(d)
    /// where rho is the persistence parameter (0 < rho < 1)
    ///
    /// Reference: "Rank-Biased Precision for Measurement of Retrieval
    /// Effectiveness" (Moffat & Zobel, 2008)
    fn fuse_rbf(
        &self,
        results: HashMap<String, Vec<RankedResult>>,
        rho: f32,
    ) -> Result<Vec<FusedResult>> {
        let mut scores: HashMap<Uuid, (f32, HashMap<String, f32>)> = HashMap::new();

        for (method_name, method_results) in results {
            for result in method_results {
                // RBF score: rho^rank
                let rbf_score = rho.powi(result.rank as i32);

                let (total_score, method_scores) = scores
                    .entry(result.id)
                    .or_insert_with(|| (0.0, HashMap::new()));

                *total_score += rbf_score;
                method_scores.insert(method_name.clone(), result.score);
            }
        }

        let mut fused_results: Vec<FusedResult> = scores
            .into_iter()
            .map(|(id, (fusion_score, method_scores))| FusedResult {
                id,
                fusion_score,
                method_scores,
            })
            .collect();

        fused_results.sort_by(|a, b| {
            b.fusion_score
                .partial_cmp(&a.fusion_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(fused_results)
    }
}

/// Convert scored results to ranked results
pub fn to_ranked_results(results: Vec<(Uuid, f32)>) -> Vec<RankedResult> {
    results
        .into_iter()
        .enumerate()
        .map(|(rank, (id, score))| RankedResult { id, score, rank })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion() {
        let engine = FusionEngine::rrf(60);

        let mut results = HashMap::new();

        // Sparse results
        results.insert(
            "sparse".to_string(),
            vec![
                RankedResult {
                    id: Uuid::nil(),
                    score: 10.0,
                    rank: 0,
                },
                RankedResult {
                    id: Uuid::from_u128(1),
                    score: 8.0,
                    rank: 1,
                },
                RankedResult {
                    id: Uuid::from_u128(2),
                    score: 6.0,
                    rank: 2,
                },
            ],
        );

        // Dense results (different order)
        results.insert(
            "dense".to_string(),
            vec![
                RankedResult {
                    id: Uuid::from_u128(1),
                    score: 0.9,
                    rank: 0,
                },
                RankedResult {
                    id: Uuid::nil(),
                    score: 0.8,
                    rank: 1,
                },
                RankedResult {
                    id: Uuid::from_u128(3),
                    score: 0.7,
                    rank: 2,
                },
            ],
        );

        let fused = engine.fuse(results).unwrap();

        // Both Uuid::from_u128(1) and Uuid::nil() have same RRF score
        // (each is rank 0 in one and rank 1 in another)
        // Check that both appear in top 2 results
        assert!(fused.len() >= 2);
        let top_ids: Vec<_> = fused.iter().take(2).map(|r| r.id).collect();
        assert!(top_ids.contains(&Uuid::from_u128(1)));
        assert!(top_ids.contains(&Uuid::nil()));
    }

    #[test]
    fn test_weighted_sum_fusion() {
        let engine = FusionEngine::weighted(0.7);

        let mut results = HashMap::new();

        results.insert(
            "sparse".to_string(),
            vec![
                RankedResult {
                    id: Uuid::nil(),
                    score: 10.0,
                    rank: 0,
                },
                RankedResult {
                    id: Uuid::from_u128(1),
                    score: 5.0,
                    rank: 1,
                },
            ],
        );

        results.insert(
            "dense".to_string(),
            vec![
                RankedResult {
                    id: Uuid::from_u128(1),
                    score: 0.9,
                    rank: 0,
                },
                RankedResult {
                    id: Uuid::nil(),
                    score: 0.5,
                    rank: 1,
                },
            ],
        );

        let fused = engine.fuse(results).unwrap();

        assert_eq!(fused.len(), 2);
        // Check that scores are normalized and combined
        assert!(fused[0].fusion_score <= 1.0);
    }

    #[test]
    fn test_to_ranked_results() {
        let scored = vec![
            (Uuid::nil(), 0.9),
            (Uuid::from_u128(1), 0.8),
            (Uuid::from_u128(2), 0.7),
        ];

        let ranked = to_ranked_results(scored);

        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].rank, 0);
        assert_eq!(ranked[1].rank, 1);
        assert_eq!(ranked[2].rank, 2);
    }
}
