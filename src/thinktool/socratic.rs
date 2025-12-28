//! # Socratic Questioning Engine
//!
//! Implements the Socratic method for deep inquiry and BrutalHonesty critique.
//!
//! ## Scientific Foundation
//!
//! Based on Socratic pedagogy and epistemic inquiry:
//! - Elenchus (cross-examination) to expose contradictions
//! - Maieutics (midwifery of ideas) to draw out understanding
//! - Aporia (puzzlement) as a path to deeper understanding
//!
//! ## The Socratic Method
//!
//! ```text
//! CLAIM → CLARIFY → CHALLENGE → CONSEQUENCE → QUESTION ASSUMPTIONS
//!    ↓        ↓          ↓            ↓               ↓
//!  What?    Define    Evidence?   Implications?   Why believe?
//! ```
//!
//! ## Question Categories (Paul-Elder Framework)
//!
//! 1. Clarification questions
//! 2. Probing assumptions
//! 3. Probing reasons/evidence
//! 4. Viewpoint/perspective questions
//! 5. Probing implications/consequences
//! 6. Questions about the question
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::thinktool::socratic::{SocraticEngine, SocraticConfig};
//!
//! let engine = SocraticEngine::new(SocraticConfig::default());
//! let result = engine.examine(claim).await?;
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for Socratic questioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocraticConfig {
    /// Question categories to use
    pub categories: Vec<QuestionCategory>,
    /// Minimum questions per category
    pub min_questions_per_category: usize,
    /// Maximum total questions
    pub max_questions: usize,
    /// Depth of follow-up (nested questioning)
    pub follow_up_depth: usize,
    /// Enable aporia detection (finding genuine puzzlement)
    pub detect_aporia: bool,
    /// BrutalHonesty mode (more aggressive questioning)
    pub brutal_honesty: bool,
}

impl Default for SocraticConfig {
    fn default() -> Self {
        Self {
            categories: vec![
                QuestionCategory::Clarification,
                QuestionCategory::Assumptions,
                QuestionCategory::Evidence,
                QuestionCategory::Viewpoints,
                QuestionCategory::Implications,
                QuestionCategory::MetaQuestions,
            ],
            min_questions_per_category: 1,
            max_questions: 12,
            follow_up_depth: 2,
            detect_aporia: true,
            brutal_honesty: false,
        }
    }
}

impl SocraticConfig {
    /// BrutalHonesty-optimized configuration
    pub fn brutal_honesty() -> Self {
        Self {
            categories: vec![
                QuestionCategory::Clarification,
                QuestionCategory::Assumptions,
                QuestionCategory::Evidence,
                QuestionCategory::Viewpoints,
                QuestionCategory::Implications,
                QuestionCategory::MetaQuestions,
                QuestionCategory::DevilsAdvocate,
                QuestionCategory::SteelMan,
            ],
            min_questions_per_category: 2,
            max_questions: 20,
            follow_up_depth: 3,
            detect_aporia: true,
            brutal_honesty: true,
        }
    }

    /// Quick examination
    pub fn quick() -> Self {
        Self {
            categories: vec![
                QuestionCategory::Clarification,
                QuestionCategory::Assumptions,
                QuestionCategory::Evidence,
            ],
            min_questions_per_category: 1,
            max_questions: 6,
            follow_up_depth: 1,
            detect_aporia: false,
            brutal_honesty: false,
        }
    }
}

/// Categories of Socratic questions (Paul-Elder Framework extended)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuestionCategory {
    /// What do you mean by...? Can you clarify?
    Clarification,
    /// What are you assuming? Why assume that?
    Assumptions,
    /// What evidence supports this? How do you know?
    Evidence,
    /// How would others view this? What alternatives exist?
    Viewpoints,
    /// What follows from this? What are the consequences?
    Implications,
    /// Why is this question important? What's the deeper question?
    MetaQuestions,
    /// What's the strongest counter-argument?
    DevilsAdvocate,
    /// What's the strongest version of this argument?
    SteelMan,
}

impl QuestionCategory {
    /// Get description of this category
    pub fn description(&self) -> &'static str {
        match self {
            Self::Clarification => "Questions that seek to understand exactly what is meant",
            Self::Assumptions => "Questions that probe underlying assumptions and presuppositions",
            Self::Evidence => "Questions that examine reasons, evidence, and justification",
            Self::Viewpoints => "Questions that explore alternative perspectives",
            Self::Implications => "Questions that investigate consequences and implications",
            Self::MetaQuestions => "Questions about the question itself",
            Self::DevilsAdvocate => "Questions that challenge by taking the opposing view",
            Self::SteelMan => "Questions that strengthen the argument to test its limits",
        }
    }

    /// Example questions in this category
    pub fn examples(&self) -> Vec<&'static str> {
        match self {
            Self::Clarification => vec![
                "What exactly do you mean by...?",
                "Can you give me an example?",
                "How does this relate to...?",
                "Can you put that another way?",
            ],
            Self::Assumptions => vec![
                "What are you assuming here?",
                "Why do you assume that's true?",
                "What if that assumption were wrong?",
                "What would have to be true for this to hold?",
            ],
            Self::Evidence => vec![
                "What evidence supports this?",
                "How do you know this is true?",
                "What would convince you otherwise?",
                "What's the source of this belief?",
            ],
            Self::Viewpoints => vec![
                "How would [X] see this?",
                "What's the opposing view?",
                "Are there alternative explanations?",
                "Who might disagree and why?",
            ],
            Self::Implications => vec![
                "What follows from this?",
                "If this is true, what else must be true?",
                "What are the practical consequences?",
                "How does this affect...?",
            ],
            Self::MetaQuestions => vec![
                "Why is this question important?",
                "Is this the right question to ask?",
                "What's the deeper issue here?",
                "How would we know if we've answered this?",
            ],
            Self::DevilsAdvocate => vec![
                "What's the strongest argument against this?",
                "How could this fail?",
                "What would a critic say?",
                "What evidence would disprove this?",
            ],
            Self::SteelMan => vec![
                "What's the strongest form of this argument?",
                "How could this be made more defensible?",
                "What additional evidence would strengthen this?",
                "What objections does this already address?",
            ],
        }
    }
}

/// A single Socratic question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocraticQuestion {
    /// Unique identifier
    pub id: usize,
    /// The question text
    pub question: String,
    /// Question category
    pub category: QuestionCategory,
    /// Follow-up to which question (None if root)
    pub follows_up: Option<usize>,
    /// Depth level (0 = root question)
    pub depth: usize,
    /// Expected type of answer
    pub answer_type: AnswerType,
    /// Whether this question was answered
    pub answered: bool,
    /// The answer if provided
    pub answer: Option<String>,
}

/// Expected type of answer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnswerType {
    /// A definition or clarification
    Definition,
    /// Evidence or justification
    Evidence,
    /// An explanation or reason
    Explanation,
    /// Acknowledgment of uncertainty
    Uncertainty,
    /// A counter-example
    CounterExample,
    /// A reformulation
    Reformulation,
    /// A concession
    Concession,
}

/// State of aporia (productive puzzlement)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aporia {
    /// Description of the puzzlement
    pub description: String,
    /// What beliefs are in tension?
    pub tension: Vec<String>,
    /// Questions that led to aporia
    pub triggering_questions: Vec<usize>,
    /// Potential paths forward
    pub potential_resolutions: Vec<String>,
    /// Is this a genuine philosophical puzzle or just confusion?
    pub genuine_puzzle: bool,
}

/// Result of Socratic examination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocraticResult {
    /// Original claim being examined
    pub claim: String,
    /// All questions asked
    pub questions: Vec<SocraticQuestion>,
    /// Questions by category
    pub category_coverage: Vec<(QuestionCategory, usize)>,
    /// Discovered aporias
    pub aporias: Vec<Aporia>,
    /// Key insights from examination
    pub insights: Vec<String>,
    /// Exposed weaknesses
    pub weaknesses: Vec<String>,
    /// Hidden assumptions discovered
    pub hidden_assumptions: Vec<String>,
    /// Revised understanding
    pub revised_claim: Option<String>,
    /// Confidence in original claim after examination
    pub post_examination_confidence: f32,
}

impl SocraticResult {
    /// Was aporia reached?
    pub fn reached_aporia(&self) -> bool {
        self.aporias.iter().any(|a| a.genuine_puzzle)
    }

    /// Questions answered ratio
    pub fn answer_rate(&self) -> f32 {
        if self.questions.is_empty() {
            return 0.0;
        }
        self.questions.iter().filter(|q| q.answered).count() as f32 / self.questions.len() as f32
    }

    /// Average depth of questioning
    pub fn avg_depth(&self) -> f32 {
        if self.questions.is_empty() {
            return 0.0;
        }
        self.questions.iter().map(|q| q.depth as f32).sum::<f32>() / self.questions.len() as f32
    }

    /// Format examination summary
    pub fn format_summary(&self) -> String {
        let answered = self.questions.iter().filter(|q| q.answered).count();
        format!(
            "Socratic Examination: {} questions ({}/{} answered), {} assumptions exposed, {} weaknesses found, confidence: {:.0}%",
            self.questions.len(),
            answered,
            self.questions.len(),
            self.hidden_assumptions.len(),
            self.weaknesses.len(),
            self.post_examination_confidence * 100.0
        )
    }
}

/// Prompt templates for Socratic questioning
pub struct SocraticPrompts;

impl SocraticPrompts {
    /// Generate initial questions for a claim
    pub fn examine_claim(claim: &str, categories: &[QuestionCategory]) -> String {
        let category_guidance: String = categories
            .iter()
            .map(|c| format!("- {:?}: {}", c, c.description()))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"Apply the SOCRATIC METHOD to examine this claim.

CLAIM: {claim}

Generate probing questions in these categories:
{category_guidance}

For each question:
1. State the question clearly
2. Indicate the category
3. Explain what the question seeks to uncover
4. Suggest what type of answer would be satisfactory

Generate at least one question per category.
Be genuinely curious. Seek to understand, not to defeat.

Format:
QUESTION 1:
- Category: [category]
- Question: [the question]
- Purpose: [what this seeks to uncover]
- Answer type expected: [definition/evidence/explanation/etc.]

QUESTION 2:
..."#,
            claim = claim,
            category_guidance = category_guidance
        )
    }

    /// Generate follow-up questions based on an answer
    pub fn follow_up(original_question: &str, answer: &str, depth: usize, brutal: bool) -> String {
        let mode = if brutal {
            "Be RUTHLESSLY PROBING. Do not accept vague answers. Push for precision."
        } else {
            "Be genuinely curious. Seek deeper understanding."
        };

        format!(
            r#"Generate follow-up Socratic questions.

ORIGINAL QUESTION: {original_question}

ANSWER PROVIDED: {answer}

CURRENT DEPTH: {depth}

{mode}

Based on this answer:
1. What remains unclear or undefined?
2. What assumptions does this answer make?
3. What evidence supports this answer?
4. What are the implications of this answer?
5. What contradictions or tensions exist?

Generate 2-3 follow-up questions that probe deeper.

Format:
FOLLOW-UP 1:
- Question: [the question]
- Category: [what type of question]
- Purpose: [what this seeks to uncover]

FOLLOW-UP 2:
..."#,
            original_question = original_question,
            answer = answer,
            depth = depth,
            mode = mode
        )
    }

    /// Identify aporia (productive puzzlement)
    pub fn detect_aporia(claim: &str, questions: &[String], answers: &[String]) -> String {
        let qa_pairs: String = questions
            .iter()
            .zip(answers.iter())
            .enumerate()
            .map(|(i, (q, a))| format!("Q{}: {}\nA{}: {}", i + 1, q, i + 1, a))
            .collect::<Vec<_>>()
            .join("\n\n");

        format!(
            r#"Analyze this Socratic examination for APORIA (genuine puzzlement).

ORIGINAL CLAIM: {claim}

EXAMINATION:
{qa_pairs}

Aporia occurs when:
1. Genuinely held beliefs come into conflict
2. The examination reveals we don't know what we thought we knew
3. There's no easy resolution without abandoning a cherished belief

Analyze:
1. Are there tensions between beliefs expressed?
2. Has any belief been undermined by the questioning?
3. Is there genuine puzzlement, or just confusion?
4. What are the potential paths forward?

Format:
APORIA_DETECTED: [yes/no]
TENSION: [describe the conflicting beliefs]
GENUINE_PUZZLE: [is this a real philosophical puzzle or just confusion?]
POTENTIAL_RESOLUTIONS:
1. [resolution option 1]
2. [resolution option 2]

INSIGHTS:
- [key insight 1]
- [key insight 2]"#,
            claim = claim,
            qa_pairs = qa_pairs
        )
    }

    /// BrutalHonesty Socratic examination
    pub fn brutal_honesty_examine(claim: &str) -> String {
        format!(
            r#"Apply BRUTAL HONESTY Socratic examination to this claim.

CLAIM: {claim}

You are the relentless questioner. Your job is to:
1. EXPOSE every hidden assumption
2. CHALLENGE every piece of evidence
3. FIND every weakness
4. REVEAL every bias
5. TEST every implication

Ask the HARDEST questions:

CLARIFICATION (be pedantic):
- What EXACTLY do you mean by each term?
- How is this not just [alternative interpretation]?

ASSUMPTIONS (be suspicious):
- What are you ASSUMING that could be completely wrong?
- What would have to be true that you haven't established?

EVIDENCE (be skeptical):
- What HARD EVIDENCE supports this?
- Why should we believe that evidence?
- What contrary evidence exists?

IMPLICATIONS (be thorough):
- If this is true, what ELSE must be true?
- What are the uncomfortable consequences?

DEVIL'S ADVOCATE:
- What's the STRONGEST case against this?
- How would your smartest critic respond?

Generate at least 10 probing questions.
Do not pull punches. Seek truth, not comfort."#,
            claim = claim
        )
    }

    /// Synthesize insights from examination
    pub fn synthesize(claim: &str, questions: &[String], insights: &[String]) -> String {
        let questions_formatted: String = questions
            .iter()
            .enumerate()
            .map(|(i, q)| format!("{}. {}", i + 1, q))
            .collect::<Vec<_>>()
            .join("\n");

        let insights_formatted: String = insights
            .iter()
            .enumerate()
            .map(|(i, insight)| format!("{}. {}", i + 1, insight))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"Synthesize the Socratic examination.

ORIGINAL CLAIM: {claim}

QUESTIONS ASKED:
{questions_formatted}

INSIGHTS DISCOVERED:
{insights_formatted}

Provide:
1. REVISED_CLAIM: A more defensible version (if needed)
2. HIDDEN_ASSUMPTIONS: Assumptions that were exposed
3. WEAKNESSES: Vulnerabilities in the original claim
4. STRENGTHS: What remains solid after examination
5. CONFIDENCE: How confident can we be after this examination? (0-100%)

Format as structured output."#,
            claim = claim,
            questions_formatted = questions_formatted,
            insights_formatted = insights_formatted
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SocraticConfig::default();
        assert!(!config.brutal_honesty);
        assert!(config.categories.contains(&QuestionCategory::Clarification));
        assert!(config.categories.contains(&QuestionCategory::Assumptions));
    }

    #[test]
    fn test_brutal_honesty_config() {
        let config = SocraticConfig::brutal_honesty();
        assert!(config.brutal_honesty);
        assert!(config
            .categories
            .contains(&QuestionCategory::DevilsAdvocate));
        assert!(config.categories.contains(&QuestionCategory::SteelMan));
        assert!(config.max_questions >= 15);
    }

    #[test]
    fn test_question_categories() {
        let cat = QuestionCategory::Assumptions;
        assert!(cat.description().contains("assumptions"));
        assert!(!cat.examples().is_empty());
    }

    #[test]
    fn test_socratic_result() {
        let result = SocraticResult {
            claim: "All swans are white".into(),
            questions: vec![
                SocraticQuestion {
                    id: 0,
                    question: "What do you mean by 'all'?".into(),
                    category: QuestionCategory::Clarification,
                    follows_up: None,
                    depth: 0,
                    answer_type: AnswerType::Definition,
                    answered: true,
                    answer: Some("Every swan that exists".into()),
                },
                SocraticQuestion {
                    id: 1,
                    question: "Have you observed all swans?".into(),
                    category: QuestionCategory::Evidence,
                    follows_up: Some(0),
                    depth: 1,
                    answer_type: AnswerType::Evidence,
                    answered: true,
                    answer: Some("No, but all I've seen are white".into()),
                },
            ],
            category_coverage: vec![
                (QuestionCategory::Clarification, 1),
                (QuestionCategory::Evidence, 1),
            ],
            aporias: vec![],
            insights: vec!["Claim is based on limited observation".into()],
            weaknesses: vec!["Cannot verify claim about unobserved swans".into()],
            hidden_assumptions: vec!["Future swans will be like past swans".into()],
            revised_claim: Some("All observed swans have been white".into()),
            post_examination_confidence: 0.4,
        };

        assert_eq!(result.answer_rate(), 1.0);
        assert!(!result.reached_aporia());
        assert!(result.format_summary().contains("2/2 answered"));
    }

    #[test]
    fn test_aporia() {
        let aporia = Aporia {
            description: "We believe in free will but also causation".into(),
            tension: vec![
                "Free will requires uncaused choices".into(),
                "All events are caused".into(),
            ],
            triggering_questions: vec![3, 5],
            potential_resolutions: vec!["Compatibilism".into(), "Libertarian free will".into()],
            genuine_puzzle: true,
        };

        assert!(aporia.genuine_puzzle);
        assert_eq!(aporia.tension.len(), 2);
    }

    #[test]
    fn test_avg_depth() {
        let result = SocraticResult {
            claim: "Test".into(),
            questions: vec![
                SocraticQuestion {
                    id: 0,
                    question: "Q1".into(),
                    category: QuestionCategory::Clarification,
                    follows_up: None,
                    depth: 0,
                    answer_type: AnswerType::Definition,
                    answered: true,
                    answer: None,
                },
                SocraticQuestion {
                    id: 1,
                    question: "Q2".into(),
                    category: QuestionCategory::Evidence,
                    follows_up: Some(0),
                    depth: 1,
                    answer_type: AnswerType::Evidence,
                    answered: false,
                    answer: None,
                },
                SocraticQuestion {
                    id: 2,
                    question: "Q3".into(),
                    category: QuestionCategory::Implications,
                    follows_up: Some(1),
                    depth: 2,
                    answer_type: AnswerType::Explanation,
                    answered: true,
                    answer: None,
                },
            ],
            category_coverage: vec![],
            aporias: vec![],
            insights: vec![],
            weaknesses: vec![],
            hidden_assumptions: vec![],
            revised_claim: None,
            post_examination_confidence: 0.5,
        };

        assert!((result.avg_depth() - 1.0).abs() < 0.01);
        assert!((result.answer_rate() - 2.0 / 3.0).abs() < 0.01);
    }
}
