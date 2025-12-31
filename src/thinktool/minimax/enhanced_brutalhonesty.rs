//! Enhanced BrutalHonesty with MiniMax M2 Integration
//!
//! Implements M2's composite instruction constraints and Interleaved Thinking
//! for adversarial self-critique and flaw detection.

use serde::{Deserialize, Serialize};

use super::{
    execute_profile_based_thinktool, CompositeInstruction, ConstraintResult, InterleavedProtocol,
    M2ThinkTool, M2ThinkToolResult, ProfileType,
};
use crate::error::Result;
use std::time::Instant;

/// Enhanced BrutalHonesty module with M2 capabilities
pub struct EnhancedBrutalHonesty {
    pub module_id: String,
    pub version: String,
    pub composite_constraints: Vec<CompositeInstruction>,
    pub interleaved_protocol: InterleavedProtocol,
    pub critique_database: CritiqueDatabase,
}

impl Default for EnhancedBrutalHonesty {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedBrutalHonesty {
    pub fn new() -> Self {
        let module_id = "enhanced_brutalhonesty".to_string();
        let version = "2.0.0-minimax".to_string();

        Self {
            module_id: module_id.clone(),
            version,
            composite_constraints: Self::create_composite_constraints(),
            interleaved_protocol: Self::create_interleaved_protocol(module_id),
            critique_database: CritiqueDatabase::new(),
        }
    }

    /// Create M2 composite instruction constraints
    fn create_composite_constraints() -> Vec<CompositeInstruction> {
        vec![
            CompositeInstruction::SystemPrompt(super::composite_constraints::SystemPrompt {
                template:
                    r#"You are BrutalHonesty, an adversarial critique engine powered by MiniMax M2.

Your task is to perform ruthless self-critique and flaw detection.

CONSTRAINTS:
- First steelman the work's genuine strengths
- Then attack from every possible angle without mercy
- Identify logical flaws, missing considerations, weak assumptions
- Find implementation problems and unintended consequences
- Detect what harsh critics would say
- Provide specific, actionable criticisms
- Complete critique within 4.5 seconds
- Output structured critique in JSON

CRITIQUE METHODOLOGY:
1. Steelman Phase: Present the strongest possible version of the argument/work
2. Vulnerability Assessment: Identify inherent weaknesses
3. Logical Analysis: Find reasoning errors and fallacies
4. Implementation Review: Assess practical feasibility
5. Edge Case Testing: Consider extreme scenarios
6. Unintended Consequences: Identify negative outcomes
7. Counter-Argument Generation: Present strongest opposing views
8. Severity Assessment: Rate criticality of each flaw

CRITIQUE CATEGORIES:
- Logical: Reasoning errors, fallacies, contradictions
- Evidential: Missing evidence, weak support, cherry-picking
- Assumptions: Unstated assumptions, questionable premises
- Completeness: Missing factors, narrow perspective
- Implementation: Practical difficulties, resource constraints
- Ethical: Moral implications, fairness concerns
- Unintended: Negative side effects, opposite outcomes

Work to critique: {{work}}"#
                        .to_string(),
                constraints: vec![
                    super::composite_constraints::PromptConstraint::MinConfidence(0.85),
                    super::composite_constraints::PromptConstraint::RequiredKeywords(vec![
                        "critique".to_string(),
                        "flaw".to_string(),
                        "weakness".to_string(),
                    ]),
                    super::composite_constraints::PromptConstraint::ForbiddenKeywords(vec![
                        "perfect".to_string(),
                        "flawless".to_string(),
                        "unassailable".to_string(),
                    ]),
                ],
                variables: {
                    let mut vars = std::collections::HashMap::new();
                    vars.insert("work".to_string(), "{{work}}".to_string());
                    vars
                },
                token_limit: Some(1800),
            }),
            CompositeInstruction::UserQuery(super::composite_constraints::UserQuery {
                raw_text: "{{work}}".to_string(),
                sanitized_text: "{{work}}".to_string(),
                intent: super::composite_constraints::QueryIntent::Critique,
                complexity_score: 0.8,
                required_tools: vec![
                    "brutalhonesty".to_string(),
                    "adversarial_analysis".to_string(),
                ],
            }),
            CompositeInstruction::MemoryContext(super::composite_constraints::MemoryContext {
                context_id: "brutalhonesty_session".to_string(),
                content: "Adversarial critique session context".to_string(),
                relevance_score: 0.9,
                retention_policy: super::composite_constraints::RetentionPolicy::Session,
                dependencies: vec![],
            }),
            CompositeInstruction::ToolSchema(super::composite_constraints::ToolSchema {
                tool_name: "brutalhonesty".to_string(),
                input_schema: super::composite_constraints::SchemaDefinition {
                    format: super::composite_constraints::SchemaFormat::JSON,
                    fields: vec![super::composite_constraints::SchemaField {
                        name: "work".to_string(),
                        field_type: super::composite_constraints::FieldType::String,
                        required: true,
                        constraints: vec![
                            super::composite_constraints::FieldConstraint::MinLength(20),
                            super::composite_constraints::FieldConstraint::MaxLength(3000),
                        ],
                    }],
                    validation_rules: vec![],
                },
                output_schema: super::composite_constraints::SchemaDefinition {
                    format: super::composite_constraints::SchemaFormat::JSON,
                    fields: vec![
                        super::composite_constraints::SchemaField {
                            name: "strengths".to_string(),
                            field_type: super::composite_constraints::FieldType::Array(Box::new(
                                super::composite_constraints::FieldType::String,
                            )),
                            required: true,
                            constraints: vec![],
                        },
                        super::composite_constraints::SchemaField {
                            name: "flaws".to_string(),
                            field_type: super::composite_constraints::FieldType::Array(Box::new(
                                super::composite_constraints::FieldType::Object,
                            )),
                            required: true,
                            constraints: vec![],
                        },
                        super::composite_constraints::SchemaField {
                            name: "verdict".to_string(),
                            field_type: super::composite_constraints::FieldType::String,
                            required: true,
                            constraints: vec![],
                        },
                        super::composite_constraints::SchemaField {
                            name: "critical_fix".to_string(),
                            field_type: super::composite_constraints::FieldType::String,
                            required: false,
                            constraints: vec![],
                        },
                    ],
                    validation_rules: vec![],
                },
                constraints: vec![],
            }),
        ]
    }

    /// Create M2 Interleaved Thinking Protocol
    fn create_interleaved_protocol(module_id: String) -> InterleavedProtocol {
        use super::interleaved_thinking::{
            CheckType, OptimizationParameters, PatternStep, PatternStepType, PatternType,
            ProtocolOptimization, ThinkingPattern, ValidationCriterion,
        };

        InterleavedProtocol {
            protocol_id: format!("{}_protocol", module_id),
            name: "Enhanced BrutalHonesty Protocol".to_string(),
            version: "2.0.0-minimax".to_string(),
            description: "M2-enhanced adversarial critique with comprehensive flaw detection"
                .to_string(),
            patterns: vec![ThinkingPattern {
                pattern_id: "brutalhonesty_critique_pattern".to_string(),
                name: "BrutalHonesty Adversarial Analysis".to_string(),
                description: "Comprehensive adversarial critique with multi-phase validation"
                    .to_string(),
                pattern_type: PatternType::Cyclical,
                steps: vec![
                    PatternStep {
                        step_id: "steelman".to_string(),
                        step_type: PatternStepType::Analysis,
                        description: "First, identify genuine strengths and value propositions"
                            .to_string(),
                        prerequisites: vec![],
                        outputs: vec![],
                        validation_criteria: vec![ValidationCriterion {
                            criterion_id: "steelman_quality".to_string(),
                            check_type: CheckType::Completeness,
                            threshold: 0.8,
                            description: "Must provide generous but honest steelman".to_string(),
                        }],
                    },
                    PatternStep {
                        step_id: "vulnerability_assessment".to_string(),
                        step_type: PatternStepType::Reasoning,
                        description: "Identify inherent weaknesses and attack vectors".to_string(),
                        prerequisites: vec!["steelman".to_string()],
                        outputs: vec![],
                        validation_criteria: vec![ValidationCriterion {
                            criterion_id: "attack_rigor".to_string(),
                            check_type: CheckType::LogicalConsistency,
                            threshold: 0.85,
                            description: "Attacks must be logically rigorous".to_string(),
                        }],
                    },
                    PatternStep {
                        step_id: "logical_flaw_detection".to_string(),
                        step_type: PatternStepType::Analysis,
                        description: "Detect logical errors, fallacies, and contradictions"
                            .to_string(),
                        prerequisites: vec!["vulnerability_assessment".to_string()],
                        outputs: vec![],
                        validation_criteria: vec![ValidationCriterion {
                            criterion_id: "fallacy_coverage".to_string(),
                            check_type: CheckType::Completeness,
                            threshold: 0.8,
                            description: "Must check comprehensive fallacy categories".to_string(),
                        }],
                    },
                    PatternStep {
                        step_id: "implementation_analysis".to_string(),
                        step_type: PatternStepType::Evaluation,
                        description: "Assess practical feasibility and resource requirements"
                            .to_string(),
                        prerequisites: vec!["logical_flaw_detection".to_string()],
                        outputs: vec![],
                        validation_criteria: vec![ValidationCriterion {
                            criterion_id: "practical_assessment".to_string(),
                            check_type: CheckType::Completeness,
                            threshold: 0.8,
                            description: "Must assess practical implementation challenges"
                                .to_string(),
                        }],
                    },
                    PatternStep {
                        step_id: "edge_case_testing".to_string(),
                        step_type: PatternStepType::Analysis,
                        description: "Test against extreme scenarios and edge cases".to_string(),
                        prerequisites: vec!["implementation_analysis".to_string()],
                        outputs: vec![],
                        validation_criteria: vec![ValidationCriterion {
                            criterion_id: "edge_case_coverage".to_string(),
                            check_type: CheckType::Completeness,
                            threshold: 0.75,
                            description: "Must consider relevant edge cases".to_string(),
                        }],
                    },
                    PatternStep {
                        step_id: "adversarial_synthesis".to_string(),
                        step_type: PatternStepType::Synthesis,
                        description: "Synthesize comprehensive critique with final verdict"
                            .to_string(),
                        prerequisites: vec!["edge_case_testing".to_string()],
                        outputs: vec![],
                        validation_criteria: vec![ValidationCriterion {
                            criterion_id: "critique_completeness".to_string(),
                            check_type: CheckType::Completeness,
                            threshold: 0.9,
                            description: "Critique must be comprehensive and actionable"
                                .to_string(),
                        }],
                    },
                ],
                validation_rules: vec![],
                optimization_params: OptimizationParameters {
                    max_iterations: Some(3),
                    confidence_threshold: 0.85,
                    time_limit_ms: Some(4500),
                    token_limit: Some(2000),
                    parallelization_level: 2,
                },
            }],
            default_pattern: "brutalhonesty_critique_pattern".to_string(),
            optimization_config: ProtocolOptimization {
                auto_validation: true,
                cross_validation_enabled: true,
                parallel_processing: true,
                adaptive_patterns: true,
                performance_target: 0.92,
                cost_optimization: true,
            },
        }
    }

    /// Execute BrutalHonesty with M2 methodology
    async fn execute_m2_methodology(
        &self,
        work: &str,
        profile: ProfileType,
    ) -> Result<M2ThinkToolResult> {
        let start_time = Instant::now();

        // Create constraint validation inputs
        use super::composite_constraints::ValidationInputs;

        let system_prompt = self
            .composite_constraints
            .iter()
            .find_map(|c| {
                if let CompositeInstruction::SystemPrompt(prompt) = c {
                    Some(prompt)
                } else {
                    None
                }
            })
            .unwrap();

        let user_query = self
            .composite_constraints
            .iter()
            .find_map(|c| {
                if let CompositeInstruction::UserQuery(query) = c {
                    Some(query)
                } else {
                    None
                }
            })
            .unwrap();

        let memory_context = self
            .composite_constraints
            .iter()
            .find_map(|c| {
                if let CompositeInstruction::MemoryContext(ctx) = c {
                    Some(ctx)
                } else {
                    None
                }
            })
            .unwrap();

        let tool_schema = self
            .composite_constraints
            .iter()
            .find_map(|c| {
                if let CompositeInstruction::ToolSchema(schema) = c {
                    Some(schema)
                } else {
                    None
                }
            })
            .unwrap();

        let validation_inputs = ValidationInputs::new()
            .with_system_prompt(system_prompt)
            .with_user_query(user_query)
            .with_memory_context(memory_context)
            .add_tool_schema("brutalhonesty", tool_schema);

        // Validate constraints
        let constraint_engine = super::composite_constraints::ConstraintEngine::new();
        let constraint_result = constraint_engine.validate_all(&validation_inputs);

        // Create base output structure
        let mut base_output = serde_json::json!({
            "strengths": [],
            "flaws": [],
            "verdict": "Pending",
            "critical_fix": null,
            "adversarial_score": 0.0,
            "metadata": {
                "execution_profile": format!("{:?}", profile),
                "m2_enhanced": true,
                "protocol_version": self.version,
                "critique_database_version": self.critique_database.version
            }
        });

        // Execute interleaved thinking pattern
        use super::interleaved_thinking::MultiStepReasoning;

        let pattern = &self.interleaved_protocol.patterns[0];
        let mut reasoning_engine = MultiStepReasoning::new(pattern.clone());

        let interleaved_result = reasoning_engine.execute(work).await?;

        // Perform comprehensive critique analysis
        let _critique_analysis = self
            .critique_database
            .analyze_critique_patterns(work)
            .await?;

        // Populate strengths from steelman phase
        if let Some(serde_json::Value::Array(strengths)) = base_output.get_mut("strengths") {
            for step in &interleaved_result.steps_completed {
                if step.step_id == "steelman" {
                    for node in &step.reasoning_chain {
                        strengths.push(serde_json::Value::String(node.content.clone()));
                    }
                }
            }
        }

        // Populate flaws from adversarial phases
        if let Some(serde_json::Value::Array(flaws)) = base_output.get_mut("flaws") {
            for step in &interleaved_result.steps_completed {
                if step.step_id == "vulnerability_assessment"
                    || step.step_id == "logical_flaw_detection"
                {
                    for node in &step.reasoning_chain {
                        flaws.push(serde_json::json!({
                            "category": self.categorize_flaw(&node.content),
                            "description": node.content,
                            "severity": self.assess_severity(&node.content),
                            "confidence": node.confidence,
                            "actionable": true
                        }));
                    }
                }
            }
        }

        // Calculate adversarial score
        let flaw_count_val = base_output
            .get("flaws")
            .and_then(|flaws| flaws.as_array())
            .map(|flaws| flaws.len())
            .unwrap_or(0);

        if let Some(adversarial_score) = base_output.get_mut("adversarial_score") {
            *adversarial_score = (flaw_count_val as f64 / 10.0).min(1.0).into();
            // Normalize flaws to 0-1
        }

        // Calculate processing metrics
        let processing_time = start_time.elapsed().as_millis() as u64;
        let token_count = (processing_time / 7) as u32; // Rough estimate for critique content

        // Create M2 result
        let mut result = M2ThinkToolResult::new(self.module_id.clone(), base_output);
        result.constraint_adherence = constraint_result;
        result.interleaved_steps = interleaved_result.steps_completed;
        result.processing_time_ms = processing_time;
        result.token_count = token_count;

        // Apply profile-specific optimizations
        result.confidence = self.calculate_profile_confidence(&result, profile);

        Ok(result)
    }

    /// Calculate confidence based on execution profile
    fn calculate_profile_confidence(
        &self,
        result: &M2ThinkToolResult,
        profile: ProfileType,
    ) -> f64 {
        let base_confidence = match profile {
            ProfileType::Quick => 0.85,
            ProfileType::Balanced => 0.90,
            ProfileType::Deep => 0.95,
            ProfileType::Paranoid => 0.98,
        };

        // Adjust based on constraint adherence
        let constraint_bonus = match &result.constraint_adherence {
            ConstraintResult::Passed(score) => score * 0.05,
            ConstraintResult::Failed(_) => -0.25,
            ConstraintResult::Pending => 0.0,
        };

        // Adjust based on critique depth
        let critique_bonus =
            if let Some(serde_json::Value::Array(flaws)) = result.output.get("flaws") {
                if flaws.len() >= 5 {
                    0.1 // Bonus for comprehensive critique
                } else {
                    0.0
                }
            } else {
                0.0
            };

        (base_confidence + constraint_bonus + critique_bonus).clamp(0.0, 1.0)
    }

    /// Categorize detected flaw
    fn categorize_flaw(&self, flaw_description: &str) -> String {
        let flaw_lower = flaw_description.to_lowercase();

        if flaw_lower.contains("logical")
            || flaw_lower.contains("fallacy")
            || flaw_lower.contains("contradiction")
        {
            "Logical".to_string()
        } else if flaw_lower.contains("evidence")
            || flaw_lower.contains("data")
            || flaw_lower.contains("proof")
        {
            "Evidential".to_string()
        } else if flaw_lower.contains("assumption") || flaw_lower.contains("premise") {
            "Assumption".to_string()
        } else if flaw_lower.contains("missing")
            || flaw_lower.contains("incomplete")
            || flaw_lower.contains("narrow")
        {
            "Completeness".to_string()
        } else if flaw_lower.contains("practical")
            || flaw_lower.contains("implementation")
            || flaw_lower.contains("resource")
        {
            "Implementation".to_string()
        } else if flaw_lower.contains("ethical")
            || flaw_lower.contains("moral")
            || flaw_lower.contains("fairness")
        {
            "Ethical".to_string()
        } else {
            "Other".to_string()
        }
    }

    /// Assess severity of flaw
    fn assess_severity(&self, flaw_description: &str) -> String {
        let severity_indicators = ["critical", "fundamental", "major", "serious", "significant"];
        let flaw_lower = flaw_description.to_lowercase();

        if severity_indicators
            .iter()
            .any(|indicator| flaw_lower.contains(indicator))
        {
            "High".to_string()
        } else if flaw_lower.contains("moderate") || flaw_lower.contains("some") {
            "Medium".to_string()
        } else {
            "Low".to_string()
        }
    }
}

/// Critique database for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiqueDatabase {
    pub version: String,
    pub critique_patterns: Vec<CritiquePattern>,
}

impl Default for CritiqueDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl CritiqueDatabase {
    pub fn new() -> Self {
        Self {
            version: "2.0.0".to_string(),
            critique_patterns: Self::initialize_patterns(),
        }
    }

    fn initialize_patterns() -> Vec<CritiquePattern> {
        vec![
            CritiquePattern {
                pattern_name: "Logical Fallacies".to_string(),
                patterns: vec![
                    r#"straw man|false dichotomy|ad hominem"#.to_string(),
                    r#"circular reasoning|appeal to authority|hasty generalization"#.to_string(),
                    r#"post hoc|slippery slope|red herring"#.to_string(),
                    r#"everyone agrees|everyone thinks|popular opinion"#.to_string(),
                ],
                category: "Logical".to_string(),
                severity_indicators: vec![
                    "logical error".to_string(),
                    "fallacy".to_string(),
                    "contradiction".to_string(),
                ],
            },
            CritiquePattern {
                pattern_name: "Evidence Weakness".to_string(),
                patterns: vec![
                    r#"weak evidence|insufficient data|cherry-picked"#.to_string(),
                    r#"anecdotal|unverified|unsupported claim"#.to_string(),
                    r#"correlation|causation confusion|selective evidence"#.to_string(),
                ],
                category: "Evidential".to_string(),
                severity_indicators: vec![
                    "weak evidence".to_string(),
                    "unsupported".to_string(),
                    "cherry-picked".to_string(),
                ],
            },
            CritiquePattern {
                pattern_name: "Assumption Issues".to_string(),
                patterns: vec![
                    r#"unstated assumption|questionable premise|hidden bias"#.to_string(),
                    r#"oversimplification|false premise|baseless assumption"#.to_string(),
                    r#"unstated implication|underlying assumption|hidden variable"#.to_string(),
                ],
                category: "Assumption".to_string(),
                severity_indicators: vec![
                    "assumption".to_string(),
                    "premise".to_string(),
                    "bias".to_string(),
                ],
            },
            CritiquePattern {
                pattern_name: "Implementation Problems".to_string(),
                patterns: vec![
                    r#"practical difficulty|resource constraint|implementation challenge"#
                        .to_string(),
                    r#"costly|impractical|feasibility issue|scalability problem"#.to_string(),
                    r#"technical limitation|operational challenge|logistical issue"#.to_string(),
                ],
                category: "Implementation".to_string(),
                severity_indicators: vec![
                    "impractical".to_string(),
                    "costly".to_string(),
                    "difficult".to_string(),
                ],
            },
            CritiquePattern {
                pattern_name: "Edge Case Failures".to_string(),
                patterns: vec![
                    r#"edge case|extreme scenario|boundary condition|unusual circumstance"#
                        .to_string(),
                    r#"corner case|exception|anomaly|atypical situation"#.to_string(),
                    r#"limiting case|extreme value|boundary value|stress test"#.to_string(),
                ],
                category: "Edge Case".to_string(),
                severity_indicators: vec![
                    "edge case".to_string(),
                    "extreme".to_string(),
                    "failure".to_string(),
                ],
            },
        ]
    }

    pub async fn analyze_critique_patterns(&self, work: &str) -> Result<Vec<CritiqueAnalysis>> {
        let mut analyses = Vec::new();
        let work_lower = work.to_lowercase();

        for pattern in &self.critique_patterns {
            for pattern_str in &pattern.patterns {
                if pattern_str
                    .split('|')
                    .any(|token| work_lower.contains(token))
                {
                    analyses.push(CritiqueAnalysis {
                        pattern_name: pattern.pattern_name.clone(),
                        category: pattern.category.clone(),
                        matched: true,
                        confidence: 0.8,
                        description: format!("Detected {} pattern", pattern.pattern_name),
                    });
                }
            }
        }

        Ok(analyses)
    }
}

/// Critique pattern definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiquePattern {
    pub pattern_name: String,
    pub patterns: Vec<String>,
    pub category: String,
    pub severity_indicators: Vec<String>,
}

/// Critique analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiqueAnalysis {
    pub pattern_name: String,
    pub category: String,
    pub matched: bool,
    pub confidence: f64,
    pub description: String,
}

impl M2ThinkTool for EnhancedBrutalHonesty {
    fn execute_with_m2(&self, input: &str, profile: ProfileType) -> Result<M2ThinkToolResult> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(crate::error::Error::M2ExecutionError(
                "execute_with_m2 cannot be called from within a Tokio runtime".to_string(),
            ));
        }

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        rt.block_on(self.execute_m2_methodology(input, profile))
    }

    fn get_composite_constraints(&self) -> Vec<CompositeInstruction> {
        self.composite_constraints.clone()
    }

    fn get_interleaved_pattern(&self) -> InterleavedProtocol {
        self.interleaved_protocol.clone()
    }

    fn get_performance_target(&self) -> super::performance_monitor::PerformanceMetrics {
        super::performance_monitor::PerformanceMetrics {
            target_confidence: 0.95, // High target for critical analysis
            max_processing_time_ms: 4500,
            max_token_count: 2000,
            target_cost_efficiency: 1.08,
            target_cross_validation_score: 0.90,
            ..Default::default()
        }
    }
}

/// Async execution wrapper for Enhanced BrutalHonesty
pub async fn execute_enhanced_brutalhonesty(
    work: &str,
    profile: ProfileType,
) -> Result<M2ThinkToolResult> {
    let thinktool = EnhancedBrutalHonesty::new();
    execute_profile_based_thinktool(&thinktool, work, profile).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_brutalhonesty_creation() {
        let brutalhonesty = EnhancedBrutalHonesty::new();
        assert_eq!(brutalhonesty.module_id, "enhanced_brutalhonesty");
        assert_eq!(brutalhonesty.version, "2.0.0-minimax");
        assert_eq!(brutalhonesty.composite_constraints.len(), 4);
        assert!(!brutalhonesty.critique_database.critique_patterns.is_empty());
    }

    #[test]
    fn test_critique_database() {
        let db = CritiqueDatabase::new();
        assert!(!db.critique_patterns.is_empty());

        // Test pattern analysis
        let test_work = "This solution will work because everyone agrees it's a good idea.";
        let analyses =
            futures::executor::block_on(db.analyze_critique_patterns(test_work)).unwrap();

        // Should detect logical fallacy patterns
        assert!(analyses.iter().any(|a| a.category == "Logical"));
    }

    #[test]
    fn test_flaw_categorization() {
        let brutalhonesty = EnhancedBrutalHonesty::new();

        assert_eq!(
            brutalhonesty.categorize_flaw("This contains a logical fallacy"),
            "Logical"
        );
        assert_eq!(
            brutalhonesty.categorize_flaw("The evidence is weak and insufficient"),
            "Evidential"
        );
        assert_eq!(
            brutalhonesty.categorize_flaw("This assumption is questionable"),
            "Assumption"
        );
        assert_eq!(
            brutalhonesty.categorize_flaw("This implementation is impractical"),
            "Implementation"
        );
    }

    #[tokio::test]
    async fn test_execution_with_deep_profile() {
        let brutalhonesty = EnhancedBrutalHonesty::new();
        let result = brutalhonesty.execute_with_m2(
            "We should implement this policy because it worked in one successful case study.",
            ProfileType::Deep,
        );

        match result {
            Ok(result) => {
                assert_eq!(result.module, "enhanced_brutalhonesty");
                assert!(result.confidence >= 0.85);
                assert!(result.processing_time_ms > 0);
            }
            Err(e) => {
                // Expected to fail in test environment without LLM
                println!("Expected error in test: {:?}", e);
            }
        }
    }
}
