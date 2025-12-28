//! # Multi-Agent Debate Architecture
//!
//! Implements adversarial debate between multiple agents for improved factuality.
//! Based on ICML 2024 research showing +20% factuality improvement.
//!
//! ## Scientific Foundation
//!
//! - Du et al. (ICML 2024): "Improving Factuality and Reasoning through Self-Debate"
//! - Irving et al. (2018): "AI Safety via Debate"
//!
//! ## Core Concept
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    MULTI-AGENT DEBATE                               │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │   ADVOCATE ◄─────────────────────────► CRITIC                      │
//! │   (Pro position)        Rounds        (Con position)               │
//! │        │                               │                           │
//! │        └───────────┬───────────────────┘                           │
//! │                    ▼                                               │
//! │              SYNTHESIZER                                           │
//! │        (Weighs arguments, final verdict)                           │
//! │                    │                                               │
//! │                    ▼                                               │
//! │              FINAL OUTPUT                                          │
//! │        (Balanced, fact-checked conclusion)                         │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::debate::{DebateArena, AgentRole, DebateConfig};
//!
//! let arena = DebateArena::new(DebateConfig {
//!     rounds: 3,
//!     ..Default::default()
//! });
//!
//! let result = arena.debate("Is nuclear power safe?").await?;
//! println!("Verdict: {:?}", result.verdict);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Role of an agent in the debate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    /// Argues in favor of the proposition
    Advocate,
    /// Argues against / finds flaws
    Critic,
    /// Weighs evidence and synthesizes
    Synthesizer,
    /// Checks facts and sources
    FactChecker,
    /// Considers alternatives
    DevilsAdvocate,
}

impl AgentRole {
    pub fn system_prompt(&self) -> &'static str {
        match self {
            AgentRole::Advocate => {
                "You are the ADVOCATE. Your role is to argue in FAVOR of the proposition.
Present the strongest possible case. Use evidence, logic, and persuasion.
Acknowledge weaknesses only to preempt counterarguments.
Your goal: Make the most compelling case for the position."
            }
            AgentRole::Critic => {
                "You are the CRITIC. Your role is to find FLAWS in the proposition.
Challenge assumptions. Identify weak evidence. Find logical gaps.
Present counterarguments and alternative explanations.
Your goal: Expose weaknesses and potential errors."
            }
            AgentRole::Synthesizer => {
                "You are the SYNTHESIZER. Your role is to weigh all arguments fairly.
Evaluate the strength of each side's evidence and logic.
Identify where the truth likely lies. Note remaining uncertainties.
Your goal: Produce a balanced, well-reasoned verdict."
            }
            AgentRole::FactChecker => {
                "You are the FACT-CHECKER. Your role is to verify claims.
Check sources. Identify unsupported assertions. Flag misinformation.
Rate the factual accuracy of each claim (0-100%).
Your goal: Ensure all claims are grounded in verifiable facts."
            }
            AgentRole::DevilsAdvocate => {
                "You are the DEVIL'S ADVOCATE. Your role is to consider alternatives.
What if the opposite were true? What are we missing?
Challenge consensus. Explore edge cases and unlikely scenarios.
Your goal: Ensure all perspectives have been considered."
            }
        }
    }
}

/// A single argument in the debate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    /// Which agent made this argument
    pub role: AgentRole,
    /// The argument content
    pub content: String,
    /// Round number
    pub round: usize,
    /// Claims made in this argument
    pub claims: Vec<Claim>,
    /// Evidence cited
    pub evidence: Vec<Evidence>,
    /// Strength rating (0.0-1.0)
    pub strength: f32,
    /// Rebuttals to previous arguments
    pub rebuttals: Vec<Rebuttal>,
    /// Points conceded to the opponent
    #[serde(default)]
    pub concessions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub statement: String,
    pub confidence: f32,
    pub verified: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub description: String,
    pub source: Option<String>,
    pub credibility: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rebuttal {
    /// What is being rebutted
    pub target_claim: String,
    /// The counter-argument
    pub counter: String,
    /// Strength of the rebuttal
    pub effectiveness: f32,
}

/// Configuration for the debate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateConfig {
    /// Number of debate rounds
    pub rounds: usize,
    /// Roles to include
    pub roles: Vec<AgentRole>,
    /// Whether to include fact-checking
    pub fact_check: bool,
    /// Whether to include devil's advocate
    pub devils_advocate: bool,
    /// Minimum argument strength to continue
    pub min_strength_threshold: f32,
    /// Whether to allow concession of points
    pub allow_concessions: bool,
}

impl Default for DebateConfig {
    fn default() -> Self {
        Self {
            rounds: 3,
            roles: vec![
                AgentRole::Advocate,
                AgentRole::Critic,
                AgentRole::Synthesizer,
            ],
            fact_check: true,
            devils_advocate: false,
            min_strength_threshold: 0.3,
            allow_concessions: true,
        }
    }
}

/// Final verdict from the debate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateVerdict {
    /// Verdict type
    pub verdict_type: VerdictType,
    /// Summary of the conclusion
    pub summary: String,
    /// Confidence in the verdict
    pub confidence: f32,
    /// Key points that won the debate
    pub winning_points: Vec<String>,
    /// Unresolved issues
    pub unresolved: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerdictType {
    /// Proposition is supported
    Affirmed,
    /// Proposition is refuted
    Refuted,
    /// Evidence is balanced/mixed
    Balanced,
    /// Cannot determine
    Inconclusive,
    /// True with qualifications
    PartiallyAffirmed,
    /// More investigation needed
    RequiresFurtherInvestigation,
}

/// Complete result of a debate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateResult {
    /// The proposition debated
    pub proposition: String,
    /// All arguments made
    pub arguments: Vec<Argument>,
    /// The final verdict
    pub verdict: DebateVerdict,
    /// Debate statistics
    pub stats: DebateStats,
    /// Factual claims verified (if fact-checking enabled)
    pub fact_check_results: HashMap<String, bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebateStats {
    pub total_rounds: usize,
    pub advocate_arguments: usize,
    pub critic_arguments: usize,
    pub claims_made: usize,
    pub claims_rebutted: usize,
    pub evidence_cited: usize,
    pub concessions_made: usize,
    pub avg_argument_strength: f32,
}

/// The debate arena where agents debate
pub struct DebateArena {
    pub config: DebateConfig,
    arguments: Vec<Argument>,
    proposition: Option<String>,
    current_round: usize,
}

impl DebateArena {
    pub fn new(config: DebateConfig) -> Self {
        Self {
            config,
            arguments: Vec::new(),
            proposition: None,
            current_round: 0,
        }
    }

    pub fn set_proposition(&mut self, proposition: impl Into<String>) {
        self.proposition = Some(proposition.into());
        self.arguments.clear();
        self.current_round = 0;
    }

    /// Add an argument from an agent
    pub fn add_argument(&mut self, argument: Argument) {
        self.arguments.push(argument);
    }

    /// Get arguments from a specific role
    pub fn get_arguments_by_role(&self, role: AgentRole) -> Vec<&Argument> {
        self.arguments.iter().filter(|a| a.role == role).collect()
    }

    /// Get arguments from a specific round
    pub fn get_arguments_by_round(&self, round: usize) -> Vec<&Argument> {
        self.arguments.iter().filter(|a| a.round == round).collect()
    }

    /// Get the debate transcript
    pub fn transcript(&self) -> String {
        let mut output = String::new();

        output.push_str("═══════════════════════════════════════════════════════════════\n");
        output.push_str("                        DEBATE TRANSCRIPT                       \n");
        output.push_str("═══════════════════════════════════════════════════════════════\n\n");

        if let Some(ref prop) = self.proposition {
            output.push_str(&format!("PROPOSITION: {}\n\n", prop));
        }

        for round in 0..=self.current_round {
            let round_args = self.get_arguments_by_round(round);
            if !round_args.is_empty() {
                output.push_str(&format!(
                    "━━━ ROUND {} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n",
                    round + 1
                ));

                for arg in round_args {
                    output.push_str(&format!(
                        "[{:?}] (Strength: {:.0}%)\n",
                        arg.role,
                        arg.strength * 100.0
                    ));
                    output.push_str(&format!("{}\n", arg.content));

                    if !arg.rebuttals.is_empty() {
                        output.push_str("\n  Rebuttals:\n");
                        for rebuttal in &arg.rebuttals {
                            output.push_str(&format!(
                                "    → RE: \"{}\" - {} (effectiveness: {:.0}%)\n",
                                rebuttal.target_claim,
                                rebuttal.counter,
                                rebuttal.effectiveness * 100.0
                            ));
                        }
                    }
                    output.push('\n');
                }
            }
        }

        output
    }

    /// Compute debate statistics
    pub fn compute_stats(&self) -> DebateStats {
        let advocate_args = self.get_arguments_by_role(AgentRole::Advocate).len();
        let critic_args = self.get_arguments_by_role(AgentRole::Critic).len();

        let claims_made: usize = self.arguments.iter().map(|a| a.claims.len()).sum();
        let claims_rebutted: usize = self.arguments.iter().map(|a| a.rebuttals.len()).sum();
        let evidence_cited: usize = self.arguments.iter().map(|a| a.evidence.len()).sum();

        let avg_strength = if !self.arguments.is_empty() {
            self.arguments.iter().map(|a| a.strength).sum::<f32>() / self.arguments.len() as f32
        } else {
            0.0
        };

        DebateStats {
            total_rounds: self.current_round + 1,
            advocate_arguments: advocate_args,
            critic_arguments: critic_args,
            claims_made,
            claims_rebutted,
            evidence_cited,
            concessions_made: self.arguments.iter().map(|a| a.concessions.len()).sum(),
            avg_argument_strength: avg_strength,
        }
    }

    /// Synthesize a verdict from the debate
    pub fn synthesize_verdict(&self) -> DebateVerdict {
        let advocate_strength: f32 = self
            .get_arguments_by_role(AgentRole::Advocate)
            .iter()
            .map(|a| a.strength)
            .sum();

        let critic_strength: f32 = self
            .get_arguments_by_role(AgentRole::Critic)
            .iter()
            .map(|a| a.strength)
            .sum();

        let advocate_count = self.get_arguments_by_role(AgentRole::Advocate).len() as f32;
        let critic_count = self.get_arguments_by_role(AgentRole::Critic).len() as f32;

        let avg_advocate = if advocate_count > 0.0 {
            advocate_strength / advocate_count
        } else {
            0.0
        };
        let avg_critic = if critic_count > 0.0 {
            critic_strength / critic_count
        } else {
            0.0
        };

        let (verdict_type, confidence) = if (avg_advocate - avg_critic).abs() < 0.1 {
            (VerdictType::Balanced, 0.5)
        } else if avg_advocate > avg_critic + 0.2 {
            (
                VerdictType::Affirmed,
                0.6 + (avg_advocate - avg_critic) * 0.3,
            )
        } else if avg_critic > avg_advocate + 0.2 {
            (
                VerdictType::Refuted,
                0.6 + (avg_critic - avg_advocate) * 0.3,
            )
        } else if avg_advocate > avg_critic {
            (
                VerdictType::PartiallyAffirmed,
                0.5 + (avg_advocate - avg_critic) * 0.2,
            )
        } else if self.arguments.is_empty() {
            (VerdictType::Inconclusive, 0.0)
        } else {
            (VerdictType::RequiresFurtherInvestigation, 0.4)
        };

        // Extract winning points (strongest claims from winning side)
        let winning_points = match verdict_type {
            VerdictType::Affirmed | VerdictType::PartiallyAffirmed => self
                .get_arguments_by_role(AgentRole::Advocate)
                .iter()
                .flat_map(|a| a.claims.iter())
                .filter(|c| c.confidence > 0.7)
                .map(|c| c.statement.clone())
                .take(3)
                .collect(),
            VerdictType::Refuted => self
                .get_arguments_by_role(AgentRole::Critic)
                .iter()
                .flat_map(|a| a.claims.iter())
                .filter(|c| c.confidence > 0.7)
                .map(|c| c.statement.clone())
                .take(3)
                .collect(),
            _ => Vec::new(),
        };

        // Collect unresolved points
        let unresolved: Vec<String> = self
            .arguments
            .iter()
            .flat_map(|a| a.claims.iter())
            .filter(|c| c.verified == Some(false) || c.confidence < 0.5)
            .map(|c| c.statement.clone())
            .take(3)
            .collect();

        DebateVerdict {
            verdict_type,
            summary: format!(
                "After {} rounds of debate with {} arguments, the proposition is {:?}",
                self.current_round + 1,
                self.arguments.len(),
                verdict_type
            ),
            confidence: confidence.min(1.0),
            winning_points,
            unresolved,
            recommendations: vec![],
        }
    }

    /// Build the full debate result
    pub fn build_result(&self) -> DebateResult {
        DebateResult {
            proposition: self.proposition.clone().unwrap_or_default(),
            arguments: self.arguments.clone(),
            verdict: self.synthesize_verdict(),
            stats: self.compute_stats(),
            fact_check_results: HashMap::new(),
        }
    }

    /// Reset for a new debate
    pub fn reset(&mut self) {
        self.arguments.clear();
        self.proposition = None;
        self.current_round = 0;
    }

    /// Move to next round
    pub fn next_round(&mut self) {
        self.current_round += 1;
    }
}

impl Default for DebateArena {
    fn default() -> Self {
        Self::new(DebateConfig::default())
    }
}

/// Prompt templates for debate agents
pub struct DebatePrompts;

impl DebatePrompts {
    /// Opening argument for advocate
    pub fn advocate_opening(proposition: &str) -> String {
        format!(
            r#"You are the ADVOCATE in a structured debate.

PROPOSITION: {proposition}

Present your opening argument IN FAVOR of this proposition.

Your argument should include:
1. THESIS: Your main position (1-2 sentences)
2. EVIDENCE: 2-3 key pieces of supporting evidence
3. REASONING: How the evidence supports your thesis
4. ANTICIPATION: Address likely counterarguments

Be persuasive but factual. Cite sources where possible.
Rate your confidence in each claim (0-100%).

Respond in this format:
THESIS: ...
EVIDENCE: 1. ... 2. ... 3. ...
REASONING: ...
ANTICIPATION: ...
OVERALL_STRENGTH: X%"#,
            proposition = proposition
        )
    }

    /// Opening argument for critic
    pub fn critic_opening(proposition: &str, advocate_arg: &str) -> String {
        format!(
            r#"You are the CRITIC in a structured debate.

PROPOSITION: {proposition}

ADVOCATE'S ARGUMENT:
{advocate_arg}

Present your critique AGAINST this proposition and the advocate's argument.

Your critique should include:
1. WEAKNESSES: Key flaws in the advocate's argument
2. COUNTER-EVIDENCE: Evidence that contradicts the proposition
3. ALTERNATIVE: Better explanations for the evidence
4. CONCLUSION: Why the proposition should be rejected or qualified

Be rigorous but fair. Attack the argument, not the arguer.
Rate your confidence in each counter-claim (0-100%).

Respond in this format:
WEAKNESSES: 1. ... 2. ... 3. ...
COUNTER_EVIDENCE: ...
ALTERNATIVE: ...
CONCLUSION: ...
OVERALL_STRENGTH: X%"#,
            proposition = proposition,
            advocate_arg = advocate_arg
        )
    }

    /// Rebuttal for advocate
    pub fn advocate_rebuttal(proposition: &str, critic_arg: &str, previous_args: &str) -> String {
        format!(
            r#"You are the ADVOCATE in round 2 of a structured debate.

PROPOSITION: {proposition}

CRITIC'S ARGUMENT:
{critic_arg}

PREVIOUS ARGUMENTS:
{previous_args}

Rebut the critic's arguments and strengthen your case.

Your rebuttal should:
1. ADDRESS each of the critic's main points
2. STRENGTHEN your original argument
3. PROVIDE new evidence if available
4. CONCEDE points if they are valid (shows intellectual honesty)

Respond in this format:
REBUTTALS:
- RE: "[critic's point]" → [your counter]
NEW_EVIDENCE: ...
CONCESSIONS: ... (if any)
UPDATED_POSITION: ...
STRENGTH: X%"#,
            proposition = proposition,
            critic_arg = critic_arg,
            previous_args = previous_args
        )
    }

    /// Final synthesis
    pub fn synthesizer_verdict(proposition: &str, transcript: &str) -> String {
        format!(
            r#"You are the SYNTHESIZER. Your role is to deliver the final verdict.

PROPOSITION: {proposition}

DEBATE TRANSCRIPT:
{transcript}

Analyze the debate objectively and deliver your verdict.

Consider:
1. Which side presented stronger evidence?
2. Which side had better reasoning?
3. Were key claims refuted or supported?
4. What remains uncertain?

Your verdict should include:
1. VERDICT: Affirmed / Refuted / Balanced / Inconclusive / Partially Affirmed
2. CONFIDENCE: 0-100%
3. KEY_FACTORS: What determined the outcome
4. WINNING_POINTS: Strongest arguments from the winning side
5. UNRESOLVED: Issues that couldn't be settled
6. RECOMMENDATIONS: What should be done next

Respond in JSON format."#,
            proposition = proposition,
            transcript = transcript
        )
    }

    /// Fact-checker prompt
    pub fn fact_checker(claims: &str) -> String {
        format!(
            r#"You are the FACT-CHECKER. Verify the factual accuracy of these claims.

CLAIMS TO VERIFY:
{claims}

For each claim, provide:
1. CLAIM: The exact claim
2. VERDICT: True / False / Partially True / Unverifiable
3. CONFIDENCE: 0-100%
4. SOURCE: Evidence for your verdict
5. CORRECTION: If false/partial, what is correct

Respond in JSON format with an array of fact-check results."#,
            claims = claims
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debate_arena_creation() {
        let arena = DebateArena::new(DebateConfig::default());
        assert_eq!(arena.config.rounds, 3);
        assert!(arena.proposition.is_none());
    }

    #[test]
    fn test_add_argument() {
        let mut arena = DebateArena::new(DebateConfig::default());
        arena.set_proposition("AI is beneficial");

        arena.add_argument(Argument {
            role: AgentRole::Advocate,
            content: "AI improves productivity".into(),
            round: 0,
            claims: vec![Claim {
                statement: "AI saves time".into(),
                confidence: 0.9,
                verified: None,
            }],
            evidence: vec![],
            strength: 0.8,
            rebuttals: vec![],
            concessions: vec![],
        });

        assert_eq!(arena.arguments.len(), 1);
    }

    #[test]
    fn test_verdict_synthesis() {
        let mut arena = DebateArena::new(DebateConfig::default());
        arena.set_proposition("Test proposition");

        arena.add_argument(Argument {
            role: AgentRole::Advocate,
            content: "Strong argument for".into(),
            round: 0,
            claims: vec![Claim {
                statement: "Claim 1".into(),
                confidence: 0.9,
                verified: Some(true),
            }],
            evidence: vec![],
            strength: 0.9,
            rebuttals: vec![],
            concessions: vec![],
        });

        arena.add_argument(Argument {
            role: AgentRole::Critic,
            content: "Weak argument against".into(),
            round: 0,
            claims: vec![],
            evidence: vec![],
            strength: 0.4,
            rebuttals: vec![],
            concessions: vec![],
        });

        let verdict = arena.synthesize_verdict();
        assert!(matches!(
            verdict.verdict_type,
            VerdictType::Affirmed | VerdictType::PartiallyAffirmed
        ));
    }

    #[test]
    fn test_agent_role_prompts() {
        assert!(AgentRole::Advocate.system_prompt().contains("FAVOR"));
        assert!(AgentRole::Critic.system_prompt().contains("FLAWS"));
        assert!(AgentRole::Synthesizer.system_prompt().contains("weigh"));
    }

    #[test]
    fn test_debate_stats() {
        let mut arena = DebateArena::new(DebateConfig::default());
        arena.set_proposition("Test");

        arena.add_argument(Argument {
            role: AgentRole::Advocate,
            content: "Arg 1".into(),
            round: 0,
            claims: vec![
                Claim {
                    statement: "C1".into(),
                    confidence: 0.9,
                    verified: None,
                },
                Claim {
                    statement: "C2".into(),
                    confidence: 0.8,
                    verified: None,
                },
            ],
            evidence: vec![Evidence {
                description: "E1".into(),
                source: Some("Source 1".into()),
                credibility: 0.9,
            }],
            strength: 0.85,
            rebuttals: vec![],
            concessions: vec![],
        });

        let stats = arena.compute_stats();
        assert_eq!(stats.advocate_arguments, 1);
        assert_eq!(stats.claims_made, 2);
        assert_eq!(stats.evidence_cited, 1);
    }
}
