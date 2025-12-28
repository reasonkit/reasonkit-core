//! # Divergent-Convergent Oscillation
//!
//! Implements the cognitive oscillation pattern for creative reasoning.
//!
//! ## Scientific Foundation
//!
//! Based on Guilford's Structure of Intellect model (1967) and research on
//! creative cognition:
//! - Divergent thinking: Fluency, flexibility, originality, elaboration
//! - Convergent thinking: Evaluation, selection, refinement
//! - Optimal creativity requires oscillation between both modes
//!
//! ## The Oscillation Pattern
//!
//! ```text
//! DIVERGE → CONVERGE → DIVERGE → CONVERGE → ... → FINAL
//!    ↓         ↓          ↓          ↓
//!  Expand    Focus     Expand     Focus
//!  Options   Best      Around    Optimal
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::oscillation::{OscillationEngine, OscillationConfig};
//!
//! let engine = OscillationEngine::new(OscillationConfig::default());
//! let result = engine.oscillate(problem).await?;
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for divergent-convergent oscillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationConfig {
    /// Number of oscillation cycles
    pub cycles: usize,
    /// Minimum ideas per divergent phase
    pub min_ideas_per_diverge: usize,
    /// Maximum ideas to carry forward per convergent phase
    pub max_ideas_to_converge: usize,
    /// Divergent thinking dimensions
    pub divergent_dimensions: Vec<DivergentDimension>,
    /// Convergent evaluation criteria
    pub convergent_criteria: Vec<ConvergentCriterion>,
    /// Whether to track idea lineage
    pub track_lineage: bool,
}

impl Default for OscillationConfig {
    fn default() -> Self {
        Self {
            cycles: 3,
            min_ideas_per_diverge: 5,
            max_ideas_to_converge: 3,
            divergent_dimensions: vec![
                DivergentDimension::Fluency,
                DivergentDimension::Flexibility,
                DivergentDimension::Originality,
                DivergentDimension::Elaboration,
            ],
            convergent_criteria: vec![
                ConvergentCriterion::Feasibility,
                ConvergentCriterion::Impact,
                ConvergentCriterion::Novelty,
                ConvergentCriterion::Alignment,
            ],
            track_lineage: true,
        }
    }
}

impl OscillationConfig {
    /// GigaThink-optimized configuration (10+ perspectives)
    pub fn gigathink() -> Self {
        Self {
            cycles: 3,
            min_ideas_per_diverge: 10,
            max_ideas_to_converge: 5,
            divergent_dimensions: vec![
                DivergentDimension::Fluency,
                DivergentDimension::Flexibility,
                DivergentDimension::Originality,
                DivergentDimension::Elaboration,
                DivergentDimension::Analogical,
                DivergentDimension::Contrarian,
            ],
            convergent_criteria: vec![
                ConvergentCriterion::Feasibility,
                ConvergentCriterion::Impact,
                ConvergentCriterion::Novelty,
                ConvergentCriterion::Alignment,
            ],
            track_lineage: true,
        }
    }

    /// Quick brainstorming mode
    pub fn quick() -> Self {
        Self {
            cycles: 2,
            min_ideas_per_diverge: 5,
            max_ideas_to_converge: 2,
            ..Default::default()
        }
    }
}

/// Dimensions of divergent thinking (Guilford, 1967)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergentDimension {
    /// Generate many ideas quickly
    Fluency,
    /// Generate ideas across different categories
    Flexibility,
    /// Generate novel/unusual ideas
    Originality,
    /// Add detail to ideas
    Elaboration,
    /// Generate ideas through analogies
    Analogical,
    /// Generate opposing/contrarian ideas
    Contrarian,
    /// Generate ideas by combining existing ones
    Combinatorial,
    /// Generate ideas by inverting constraints
    Inversion,
}

impl DivergentDimension {
    /// Get prompt guidance for this dimension
    pub fn prompt_guidance(&self) -> &'static str {
        match self {
            Self::Fluency => "Generate as many ideas as possible, without filtering",
            Self::Flexibility => {
                "Generate ideas from different perspectives, domains, and categories"
            }
            Self::Originality => "Generate unusual, surprising, or unconventional ideas",
            Self::Elaboration => "Add specific details, mechanisms, and implementation paths",
            Self::Analogical => "Draw ideas from analogous domains and transfer insights",
            Self::Contrarian => "Challenge assumptions and generate opposing viewpoints",
            Self::Combinatorial => "Combine and recombine existing ideas in new ways",
            Self::Inversion => "Invert the problem - what would make it worse? Then reverse",
        }
    }
}

/// Criteria for convergent evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergentCriterion {
    /// Can this be implemented?
    Feasibility,
    /// What impact would this have?
    Impact,
    /// How novel is this approach?
    Novelty,
    /// How well does this align with goals?
    Alignment,
    /// What are the risks?
    Risk,
    /// What resources are required?
    ResourceCost,
    /// How quickly can this be done?
    TimeToValue,
    /// How does this compare to alternatives?
    ComparativeAdvantage,
}

impl ConvergentCriterion {
    /// Get evaluation question for this criterion
    pub fn evaluation_question(&self) -> &'static str {
        match self {
            Self::Feasibility => "How implementable is this idea given current constraints?",
            Self::Impact => "What magnitude of positive change would this create?",
            Self::Novelty => "How unique is this compared to existing approaches?",
            Self::Alignment => "How well does this serve the stated goals?",
            Self::Risk => "What could go wrong and how severe would it be?",
            Self::ResourceCost => "What resources (time, money, effort) are required?",
            Self::TimeToValue => "How quickly can this start delivering value?",
            Self::ComparativeAdvantage => "Why is this better than the alternatives?",
        }
    }
}

/// A single idea generated during divergent thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Idea {
    /// Unique identifier
    pub id: usize,
    /// The idea content
    pub content: String,
    /// Which dimension generated this
    pub dimension: DivergentDimension,
    /// Parent idea (if evolved from another)
    pub parent_id: Option<usize>,
    /// Generation cycle (0 = first divergent phase)
    pub cycle: usize,
    /// Convergent evaluation scores
    pub scores: Vec<CriterionScore>,
    /// Overall priority score after convergent phase
    pub priority: f32,
    /// Whether this survived to the final result
    pub survived: bool,
}

/// Score for a single convergent criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionScore {
    pub criterion: ConvergentCriterion,
    pub score: f32, // 0.0 - 1.0
    pub rationale: String,
}

/// Result of a single divergent phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergentPhase {
    pub cycle: usize,
    pub ideas_generated: Vec<Idea>,
    pub dimension_coverage: Vec<(DivergentDimension, usize)>,
    pub fluency_score: f32,     // ideas per dimension
    pub flexibility_score: f32, // category diversity
}

/// Result of a single convergent phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergentPhase {
    pub cycle: usize,
    pub ideas_evaluated: usize,
    pub ideas_selected: Vec<usize>, // IDs of selected ideas
    pub elimination_rationale: Vec<String>,
    pub selection_rationale: Vec<String>,
}

/// Complete oscillation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationResult {
    /// Original problem
    pub problem: String,
    /// All generated ideas (full history)
    pub all_ideas: Vec<Idea>,
    /// Divergent phase results
    pub divergent_phases: Vec<DivergentPhase>,
    /// Convergent phase results
    pub convergent_phases: Vec<ConvergentPhase>,
    /// Final selected ideas
    pub final_ideas: Vec<Idea>,
    /// Synthesis of final ideas
    pub synthesis: String,
    /// Overall creativity metrics
    pub metrics: OscillationMetrics,
}

/// Metrics for the oscillation process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationMetrics {
    /// Total ideas generated across all cycles
    pub total_ideas: usize,
    /// Ideas that survived to final round
    pub surviving_ideas: usize,
    /// Survival rate
    pub survival_rate: f32,
    /// Average fluency (ideas per cycle)
    pub avg_fluency: f32,
    /// Category diversity score
    pub flexibility_score: f32,
    /// Originality score (based on idea uniqueness)
    pub originality_score: f32,
    /// Number of complete cycles
    pub cycles_completed: usize,
}

impl OscillationResult {
    /// Get ideas from a specific cycle
    pub fn ideas_from_cycle(&self, cycle: usize) -> Vec<&Idea> {
        self.all_ideas.iter().filter(|i| i.cycle == cycle).collect()
    }

    /// Get ideas by dimension
    pub fn ideas_by_dimension(&self, dim: DivergentDimension) -> Vec<&Idea> {
        self.all_ideas
            .iter()
            .filter(|i| i.dimension == dim)
            .collect()
    }

    /// Get idea lineage (ancestors of an idea)
    pub fn get_lineage(&self, idea_id: usize) -> Vec<&Idea> {
        let mut lineage = vec![];
        let mut current_id = Some(idea_id);

        while let Some(id) = current_id {
            if let Some(idea) = self.all_ideas.iter().find(|i| i.id == id) {
                lineage.push(idea);
                current_id = idea.parent_id;
            } else {
                break;
            }
        }

        lineage.reverse();
        lineage
    }
}

/// Prompt templates for oscillation
pub struct OscillationPrompts;

impl OscillationPrompts {
    /// Generate divergent thinking prompt
    pub fn diverge(
        problem: &str,
        dimensions: &[DivergentDimension],
        prior_ideas: &[String],
    ) -> String {
        let dimension_guidance: String = dimensions
            .iter()
            .enumerate()
            .map(|(i, d)| format!("{}. {:?}: {}", i + 1, d, d.prompt_guidance()))
            .collect::<Vec<_>>()
            .join("\n");

        let prior = if prior_ideas.is_empty() {
            "None yet - this is the first cycle.".to_string()
        } else {
            prior_ideas
                .iter()
                .enumerate()
                .map(|(i, idea)| format!("{}. {}", i + 1, idea))
                .collect::<Vec<_>>()
                .join("\n")
        };

        format!(
            r#"DIVERGENT THINKING PHASE - Generate Ideas

PROBLEM: {problem}

Use these thinking dimensions to generate diverse ideas:
{dimension_guidance}

PRIOR IDEAS (to build on or differentiate from):
{prior}

Generate at least 5 ideas, covering multiple dimensions.
For each idea, specify:
- IDEA: [The core idea]
- DIMENSION: [Which dimension it came from]
- ELABORATION: [Key details, mechanism, or implementation]

Be creative, be bold, defer judgment. Quantity over quality in this phase.

Format each idea clearly:
IDEA 1:
- Content: ...
- Dimension: ...
- Elaboration: ..."#,
            problem = problem,
            dimension_guidance = dimension_guidance,
            prior = prior
        )
    }

    /// Generate convergent evaluation prompt
    pub fn converge(
        problem: &str,
        ideas: &[String],
        criteria: &[ConvergentCriterion],
        max_select: usize,
    ) -> String {
        let criteria_list: String = criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {:?}: {}", i + 1, c, c.evaluation_question()))
            .collect::<Vec<_>>()
            .join("\n");

        let ideas_list: String = ideas
            .iter()
            .enumerate()
            .map(|(i, idea)| format!("IDEA {}: {}", i + 1, idea))
            .collect::<Vec<_>>()
            .join("\n\n");

        format!(
            r#"CONVERGENT THINKING PHASE - Evaluate and Select

PROBLEM: {problem}

IDEAS TO EVALUATE:
{ideas_list}

EVALUATION CRITERIA:
{criteria_list}

For each idea, score it on each criterion (0.0 to 1.0) with a brief rationale.

Then SELECT the top {max_select} ideas to carry forward.
Explain why you're eliminating the others.

Format:
EVALUATION:
Idea 1: [criterion scores and rationale]
Idea 2: [criterion scores and rationale]
...

SELECTED (top {max_select}):
1. Idea X - [why this was chosen]
2. Idea Y - [why this was chosen]
...

ELIMINATED:
- Idea Z - [why eliminated]
..."#,
            problem = problem,
            ideas_list = ideas_list,
            criteria_list = criteria_list,
            max_select = max_select
        )
    }

    /// Synthesize final ideas into coherent result
    pub fn synthesize(problem: &str, final_ideas: &[String]) -> String {
        let ideas_formatted: String = final_ideas
            .iter()
            .enumerate()
            .map(|(i, idea)| format!("{}. {}", i + 1, idea))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"SYNTHESIS PHASE - Integrate Best Ideas

PROBLEM: {problem}

FINAL SELECTED IDEAS:
{ideas_formatted}

Create a coherent synthesis that:
1. Integrates the best elements of each idea
2. Resolves any tensions between them
3. Provides a unified approach
4. Identifies implementation priorities

SYNTHESIS:
[Provide a 2-3 paragraph synthesis]

KEY TAKEAWAYS:
1. [Most important insight]
2. [Second most important insight]
3. [Third most important insight]

RECOMMENDED APPROACH:
[Concise action plan]"#,
            problem = problem,
            ideas_formatted = ideas_formatted
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OscillationConfig::default();
        assert_eq!(config.cycles, 3);
        assert!(config.min_ideas_per_diverge >= 5);
    }

    #[test]
    fn test_gigathink_config() {
        let config = OscillationConfig::gigathink();
        assert_eq!(config.min_ideas_per_diverge, 10);
        assert!(config
            .divergent_dimensions
            .contains(&DivergentDimension::Analogical));
        assert!(config
            .divergent_dimensions
            .contains(&DivergentDimension::Contrarian));
    }

    #[test]
    fn test_divergent_dimensions() {
        let dim = DivergentDimension::Fluency;
        assert!(dim.prompt_guidance().contains("many"));

        let dim = DivergentDimension::Originality;
        assert!(dim.prompt_guidance().contains("unusual"));
    }

    #[test]
    fn test_convergent_criteria() {
        let crit = ConvergentCriterion::Feasibility;
        assert!(crit.evaluation_question().contains("implement"));

        let crit = ConvergentCriterion::Impact;
        assert!(crit.evaluation_question().contains("change"));
    }

    #[test]
    fn test_oscillation_result_lineage() {
        let result = OscillationResult {
            problem: "Test".into(),
            all_ideas: vec![
                Idea {
                    id: 0,
                    content: "Root idea".into(),
                    dimension: DivergentDimension::Fluency,
                    parent_id: None,
                    cycle: 0,
                    scores: vec![],
                    priority: 0.8,
                    survived: true,
                },
                Idea {
                    id: 1,
                    content: "Child idea".into(),
                    dimension: DivergentDimension::Elaboration,
                    parent_id: Some(0),
                    cycle: 1,
                    scores: vec![],
                    priority: 0.9,
                    survived: true,
                },
                Idea {
                    id: 2,
                    content: "Grandchild idea".into(),
                    dimension: DivergentDimension::Originality,
                    parent_id: Some(1),
                    cycle: 2,
                    scores: vec![],
                    priority: 0.95,
                    survived: true,
                },
            ],
            divergent_phases: vec![],
            convergent_phases: vec![],
            final_ideas: vec![],
            synthesis: "".into(),
            metrics: OscillationMetrics {
                total_ideas: 3,
                surviving_ideas: 3,
                survival_rate: 1.0,
                avg_fluency: 1.0,
                flexibility_score: 1.0,
                originality_score: 0.8,
                cycles_completed: 3,
            },
        };

        let lineage = result.get_lineage(2);
        assert_eq!(lineage.len(), 3);
        assert_eq!(lineage[0].id, 0);
        assert_eq!(lineage[1].id, 1);
        assert_eq!(lineage[2].id, 2);
    }

    #[test]
    fn test_metrics() {
        let metrics = OscillationMetrics {
            total_ideas: 30,
            surviving_ideas: 5,
            survival_rate: 5.0 / 30.0,
            avg_fluency: 10.0,
            flexibility_score: 0.85,
            originality_score: 0.75,
            cycles_completed: 3,
        };

        assert!(metrics.survival_rate < 0.2);
        assert_eq!(metrics.cycles_completed, 3);
    }
}
