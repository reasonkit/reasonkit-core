//! # BedRock + Tree-of-Thoughts Integration
//!
//! Combines first principles decomposition (BedRock) with parallel
//! exploration (Tree-of-Thoughts) for comprehensive reasoning.
//!
//! ## Approach
//!
//! 1. BedRock decomposes problem into fundamental principles
//! 2. ToT explores multiple paths from each principle
//! 3. Results are synthesized back to answer
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::bedrock_tot::{BedRockToT, BedRockToTConfig};
//!
//! let engine = BedRockToT::new(BedRockToTConfig::default());
//! let result = engine.analyze("Complex problem").await?;
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for BedRock + ToT integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedRockToTConfig {
    /// Maximum decomposition depth for BedRock
    pub max_decomposition_depth: usize,
    /// Branching factor for ToT at each principle
    pub tot_branching_factor: usize,
    /// Maximum ToT depth per principle
    pub tot_max_depth: usize,
    /// Pruning threshold for ToT
    pub tot_prune_threshold: f32,
    /// Whether to explore all principles in parallel
    pub parallel_exploration: bool,
    /// Minimum principle score to explore with ToT
    pub min_principle_score: f32,
}

impl Default for BedRockToTConfig {
    fn default() -> Self {
        Self {
            max_decomposition_depth: 3,
            tot_branching_factor: 3,
            tot_max_depth: 4,
            tot_prune_threshold: 0.4,
            parallel_exploration: true,
            min_principle_score: 0.5,
        }
    }
}

/// A fundamental principle from BedRock decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Principle {
    /// Principle identifier
    pub id: usize,
    /// The principle statement
    pub statement: String,
    /// Type of principle
    pub principle_type: PrincipleType,
    /// How fundamental is this (0-1, 1 = axiom)
    pub fundamentality: f32,
    /// Confidence in this principle
    pub confidence: f32,
    /// Parent principle (if derived)
    pub parent: Option<usize>,
    /// Child principles (if decomposed further)
    pub children: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrincipleType {
    /// Self-evident truth (axiom)
    Axiom,
    /// Derived from axioms
    Derived,
    /// Assumed for argument
    Assumption,
    /// Empirical observation
    Empirical,
    /// Definition
    Definition,
}

/// Result of ToT exploration for a principle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipleExploration {
    /// The principle explored
    pub principle_id: usize,
    /// Best path found
    pub best_path: Vec<String>,
    /// Score of best path
    pub best_score: f32,
    /// Number of paths explored
    pub paths_explored: usize,
    /// Insights discovered
    pub insights: Vec<String>,
    /// Contradictions found
    pub contradictions: Vec<String>,
}

/// Complete BedRock + ToT result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedRockToTResult {
    /// Original problem
    pub problem: String,
    /// Decomposed principles
    pub principles: Vec<Principle>,
    /// ToT explorations per principle
    pub explorations: Vec<PrincipleExploration>,
    /// Synthesized conclusion
    pub conclusion: String,
    /// Overall confidence
    pub confidence: f32,
    /// Key insights from exploration
    pub key_insights: Vec<String>,
    /// Unresolved issues
    pub unresolved: Vec<String>,
}

impl BedRockToTResult {
    pub fn axiom_count(&self) -> usize {
        self.principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Axiom)
            .count()
    }

    pub fn exploration_coverage(&self) -> f32 {
        if self.principles.is_empty() {
            return 0.0;
        }
        self.explorations.len() as f32 / self.principles.len() as f32
    }
}

/// Prompt templates for BedRock + ToT
pub struct BedRockToTPrompts;

impl BedRockToTPrompts {
    /// Decompose into first principles
    pub fn decompose(problem: &str) -> String {
        format!(
            r#"Apply FIRST PRINCIPLES thinking to decompose this problem.

PROBLEM: {problem}

Break down into fundamental truths/principles:

1. AXIOMS: What are the self-evident truths that don't require proof?
   - Physical laws, mathematical truths, logical necessities

2. DEFINITIONS: What terms need precise definition?
   - Clarify ambiguous concepts

3. ASSUMPTIONS: What are we assuming that could be challenged?
   - Identify hidden assumptions

4. EMPIRICAL FACTS: What observable facts are relevant?
   - Data, measurements, observations

For each principle:
- State it clearly
- Rate fundamentality (0-1, 1 = axiom)
- Rate confidence (0-1)

Respond in JSON:
{{
    "principles": [
        {{
            "statement": "...",
            "type": "Axiom|Derived|Assumption|Empirical|Definition",
            "fundamentality": 0.0-1.0,
            "confidence": 0.0-1.0
        }}
    ]
}}"#,
            problem = problem
        )
    }

    /// Generate ToT thoughts for a principle
    pub fn explore_principle(principle: &str, problem: &str, depth: usize) -> String {
        format!(
            r#"Explore different reasoning paths from this principle.

PROBLEM: {problem}

PRINCIPLE: {principle}

CURRENT DEPTH: {depth}

Generate 3 different possible directions to reason from this principle:

DIRECTION 1: [Most direct/obvious path]
DIRECTION 2: [Alternative approach]
DIRECTION 3: [Creative/unexpected angle]

For each direction:
- What does this lead to?
- What insights does it reveal?
- What contradictions might arise?
- Score (0-1) for promise

Format:
THOUGHT 1: ... (Score: X.X)
THOUGHT 2: ... (Score: X.X)
THOUGHT 3: ... (Score: X.X)"#,
            problem = problem,
            principle = principle,
            depth = depth
        )
    }

    /// Synthesize explorations into conclusion
    pub fn synthesize(problem: &str, principles: &str, explorations: &str) -> String {
        format!(
            r#"Synthesize the first principles analysis and exploration into a conclusion.

PROBLEM: {problem}

FUNDAMENTAL PRINCIPLES IDENTIFIED:
{principles}

EXPLORATION RESULTS:
{explorations}

Synthesize:
1. What have we learned from the first principles decomposition?
2. What insights came from exploring each principle?
3. Were there any contradictions that need resolving?
4. What is the overall conclusion?

Provide:
- CONCLUSION: The synthesized answer
- KEY_INSIGHTS: Most important discoveries
- UNRESOLVED: Issues that couldn't be resolved
- CONFIDENCE: Overall confidence (0-1)

Respond in JSON format."#,
            problem = problem,
            principles = principles,
            explorations = explorations
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BedRockToTConfig::default();
        assert_eq!(config.max_decomposition_depth, 3);
        assert_eq!(config.tot_branching_factor, 3);
    }

    #[test]
    fn test_principle_types() {
        let axiom = Principle {
            id: 0,
            statement: "A = A".into(),
            principle_type: PrincipleType::Axiom,
            fundamentality: 1.0,
            confidence: 1.0,
            parent: None,
            children: vec![],
        };

        assert_eq!(axiom.principle_type, PrincipleType::Axiom);
        assert_eq!(axiom.fundamentality, 1.0);
    }

    #[test]
    fn test_result_coverage() {
        let result = BedRockToTResult {
            problem: "Test".into(),
            principles: vec![
                Principle {
                    id: 0,
                    statement: "P1".into(),
                    principle_type: PrincipleType::Axiom,
                    fundamentality: 1.0,
                    confidence: 0.9,
                    parent: None,
                    children: vec![],
                },
                Principle {
                    id: 1,
                    statement: "P2".into(),
                    principle_type: PrincipleType::Derived,
                    fundamentality: 0.7,
                    confidence: 0.8,
                    parent: Some(0),
                    children: vec![],
                },
            ],
            explorations: vec![PrincipleExploration {
                principle_id: 0,
                best_path: vec!["Step 1".into()],
                best_score: 0.9,
                paths_explored: 5,
                insights: vec!["Insight 1".into()],
                contradictions: vec![],
            }],
            conclusion: "Conclusion".into(),
            confidence: 0.85,
            key_insights: vec!["Key insight".into()],
            unresolved: vec![],
        };

        assert_eq!(result.axiom_count(), 1);
        assert!((result.exploration_coverage() - 0.5).abs() < 0.01);
    }
}
