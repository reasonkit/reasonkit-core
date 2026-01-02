//! M2 Protocol Generator
//!
//! Responsible for generating `InterleavedProtocol` definitions based on task classification
//! and high-level requirements.

use crate::error::Error;
use crate::m2::types::*;
use uuid::Uuid;

#[derive(Debug, Clone, Default)]
pub struct ProtocolGenerator;

impl ProtocolGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generates a full execution protocol based on the classified task.
    pub fn generate_protocol(
        &self,
        classification: &TaskClassification,
        name: Option<String>,
    ) -> Result<InterleavedProtocol, Error> {
        let protocol_id = Uuid::new_v4().to_string();
        let name = name.unwrap_or_else(|| format!("m2-protocol-{}", protocol_id));

        // Determine phases based on complexity and task type
        let phases = self.determine_phases(classification);

        // Determine global constraints
        let constraints = self.determine_constraints(classification);

        // Determine optimizations
        let optimizations = self.determine_optimizations(classification);

        Ok(InterleavedProtocol {
            id: protocol_id,
            name,
            version: "1.0.0".to_string(),
            description: format!("Generated protocol for {:?} task", classification.task_type),
            phases,
            constraints,
            m2_optimizations: optimizations,
            framework_compatibility: vec![],
            language_support: vec![],
        })
    }

    fn determine_phases(&self, classification: &TaskClassification) -> Vec<InterleavedPhase> {
        match classification.complexity_level {
            ComplexityLevel::Simple => vec![
                self.create_phase("reasoning", 1, 0.7),
                self.create_phase("verification", 1, 0.8),
            ],
            ComplexityLevel::Moderate => vec![
                self.create_phase("analysis", 2, 0.75),
                self.create_phase("synthesis", 1, 0.8),
                self.create_phase("verification", 1, 0.85),
            ],
            ComplexityLevel::Complex => vec![
                self.create_phase("decomposition", 1, 0.8),
                self.create_phase("parallel_analysis", 3, 0.8),
                self.create_phase("integration", 1, 0.85),
                self.create_phase("final_validation", 2, 0.9),
            ],
        }
    }

    fn create_phase(&self, name: &str, branches: u32, confidence: f64) -> InterleavedPhase {
        InterleavedPhase {
            name: name.to_string(),
            parallel_branches: branches,
            required_confidence: confidence,
            validation_methods: vec![ValidationMethod::SelfCheck],
            synthesis_methods: vec![SynthesisMethod::WeightedAverage],
            constraints: CompositeConstraints {
                time_budget_ms: 5000,
                token_budget: 4000,
                dependencies: vec![],
            },
        }
    }

    fn determine_constraints(&self, classification: &TaskClassification) -> CompositeConstraints {
        let (time, tokens) = match classification.expected_output_size {
            OutputSize::Small => (10_000, 2000),
            OutputSize::Medium => (30_000, 8000),
            OutputSize::Large => (60_000, 32000),
        };

        CompositeConstraints {
            time_budget_ms: time,
            token_budget: tokens,
            dependencies: vec![],
        }
    }

    fn determine_optimizations(&self, _classification: &TaskClassification) -> M2Optimizations {
        M2Optimizations {
            target_parameters: 200_000_000_000,
            context_optimization: ContextOptimization {
                method: "auto".to_string(),
                compression_ratio: 0.8,
            },
            output_optimization: OutputOptimization {
                format: "markdown".to_string(),
                template: "standard".to_string(),
                max_output_length: 16000,
                streaming_enabled: true,
                compression_enabled: false,
            },
            cost_optimization: CostOptimization {
                strategy: "balanced".to_string(),
                max_budget: 5.0,
                target_cost_reduction: 0.5,
                target_latency_reduction: 0.2,
                parallel_processing_enabled: true,
                caching_enabled: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // HELPER FUNCTIONS
    // ============================================================================

    /// Creates a TaskClassification with specified parameters for testing
    fn create_classification(
        task_type: TaskType,
        complexity: ComplexityLevel,
        domain: TaskDomain,
        output_size: OutputSize,
    ) -> TaskClassification {
        TaskClassification {
            task_type,
            complexity_level: complexity,
            domain,
            expected_output_size: output_size,
            time_constraints: TimeConstraints::default(),
            quality_requirements: QualityRequirements::default(),
        }
    }

    /// Creates a simple classification for quick tests
    fn simple_classification() -> TaskClassification {
        create_classification(
            TaskType::General,
            ComplexityLevel::Simple,
            TaskDomain::General,
            OutputSize::Small,
        )
    }

    /// Creates a moderate classification for quick tests
    fn moderate_classification() -> TaskClassification {
        create_classification(
            TaskType::Documentation,
            ComplexityLevel::Moderate,
            TaskDomain::General,
            OutputSize::Medium,
        )
    }

    /// Creates a complex classification for quick tests
    fn complex_classification() -> TaskClassification {
        create_classification(
            TaskType::CodeAnalysis,
            ComplexityLevel::Complex,
            TaskDomain::SystemProgramming,
            OutputSize::Large,
        )
    }

    // ============================================================================
    // PROTOCOL STRUCTURE GENERATION TESTS
    // ============================================================================

    #[test]
    fn test_simple_protocol_generation() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator
            .generate_protocol(&classification, Some("test-proto".into()))
            .unwrap();

        assert_eq!(protocol.name, "test-proto");
        assert_eq!(protocol.phases.len(), 2);
        assert_eq!(protocol.phases[0].name, "reasoning");
        assert_eq!(protocol.phases[1].name, "verification");
    }

    #[test]
    fn test_complex_protocol_generation() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.phases.len(), 4);
        assert_eq!(protocol.phases[1].parallel_branches, 3); // parallel_analysis
        assert!(protocol.constraints.token_budget >= 30000);
    }

    #[test]
    fn test_moderate_protocol_generation() {
        let generator = ProtocolGenerator::new();
        let classification = moderate_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.phases.len(), 3);
        assert_eq!(protocol.phases[0].name, "analysis");
        assert_eq!(protocol.phases[1].name, "synthesis");
        assert_eq!(protocol.phases[2].name, "verification");
    }

    #[test]
    fn test_protocol_has_unique_id() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol1 = generator.generate_protocol(&classification, None).unwrap();
        let protocol2 = generator.generate_protocol(&classification, None).unwrap();

        assert_ne!(protocol1.id, protocol2.id);
        assert!(!protocol1.id.is_empty());
        assert!(!protocol2.id.is_empty());
    }

    #[test]
    fn test_protocol_id_is_valid_uuid() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        // Attempt to parse the ID as a UUID - should succeed
        let parsed = Uuid::parse_str(&protocol.id);
        assert!(parsed.is_ok(), "Protocol ID should be a valid UUID");
    }

    #[test]
    fn test_protocol_version_is_set() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.version, "1.0.0");
    }

    #[test]
    fn test_protocol_description_includes_task_type() {
        let generator = ProtocolGenerator::new();

        // Test with General task type
        let general_classification = simple_classification();
        let protocol = generator
            .generate_protocol(&general_classification, None)
            .unwrap();
        assert!(protocol.description.contains("General"));

        // Test with CodeAnalysis task type
        let code_classification = create_classification(
            TaskType::CodeAnalysis,
            ComplexityLevel::Simple,
            TaskDomain::General,
            OutputSize::Small,
        );
        let protocol = generator
            .generate_protocol(&code_classification, None)
            .unwrap();
        assert!(protocol.description.contains("CodeAnalysis"));
    }

    #[test]
    fn test_auto_generated_name_contains_protocol_id() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert!(protocol.name.starts_with("m2-protocol-"));
        assert!(protocol.name.contains(&protocol.id));
    }

    #[test]
    fn test_custom_name_is_preserved() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let custom_name = "my-custom-protocol-name";
        let protocol = generator
            .generate_protocol(&classification, Some(custom_name.to_string()))
            .unwrap();

        assert_eq!(protocol.name, custom_name);
    }

    // ============================================================================
    // PHASE STRUCTURE TESTS
    // ============================================================================

    #[test]
    fn test_simple_phases_have_correct_confidence() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.phases[0].required_confidence, 0.7);
        assert_eq!(protocol.phases[1].required_confidence, 0.8);
    }

    #[test]
    fn test_moderate_phases_have_correct_confidence() {
        let generator = ProtocolGenerator::new();
        let classification = moderate_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.phases[0].required_confidence, 0.75); // analysis
        assert_eq!(protocol.phases[1].required_confidence, 0.8); // synthesis
        assert_eq!(protocol.phases[2].required_confidence, 0.85); // verification
    }

    #[test]
    fn test_complex_phases_have_correct_confidence() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.phases[0].required_confidence, 0.8); // decomposition
        assert_eq!(protocol.phases[1].required_confidence, 0.8); // parallel_analysis
        assert_eq!(protocol.phases[2].required_confidence, 0.85); // integration
        assert_eq!(protocol.phases[3].required_confidence, 0.9); // final_validation
    }

    #[test]
    fn test_simple_phases_have_single_branch() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        for phase in &protocol.phases {
            assert_eq!(phase.parallel_branches, 1);
        }
    }

    #[test]
    fn test_moderate_analysis_has_two_branches() {
        let generator = ProtocolGenerator::new();
        let classification = moderate_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.phases[0].parallel_branches, 2); // analysis
        assert_eq!(protocol.phases[1].parallel_branches, 1); // synthesis
        assert_eq!(protocol.phases[2].parallel_branches, 1); // verification
    }

    #[test]
    fn test_complex_parallel_analysis_has_three_branches() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.phases[0].parallel_branches, 1); // decomposition
        assert_eq!(protocol.phases[1].parallel_branches, 3); // parallel_analysis
        assert_eq!(protocol.phases[2].parallel_branches, 1); // integration
        assert_eq!(protocol.phases[3].parallel_branches, 2); // final_validation
    }

    #[test]
    fn test_all_phases_have_validation_methods() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        for phase in &protocol.phases {
            assert!(!phase.validation_methods.is_empty());
            assert!(phase
                .validation_methods
                .contains(&ValidationMethod::SelfCheck));
        }
    }

    #[test]
    fn test_all_phases_have_synthesis_methods() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        for phase in &protocol.phases {
            assert!(!phase.synthesis_methods.is_empty());
            assert!(phase
                .synthesis_methods
                .contains(&SynthesisMethod::WeightedAverage));
        }
    }

    #[test]
    fn test_phase_constraints_have_default_values() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        for phase in &protocol.phases {
            assert_eq!(phase.constraints.time_budget_ms, 5000);
            assert_eq!(phase.constraints.token_budget, 4000);
            assert!(phase.constraints.dependencies.is_empty());
        }
    }

    // ============================================================================
    // CONSTRAINT TESTS BY OUTPUT SIZE
    // ============================================================================

    #[test]
    fn test_small_output_constraints() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::General,
            ComplexityLevel::Simple,
            TaskDomain::General,
            OutputSize::Small,
        );

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.constraints.time_budget_ms, 10_000);
        assert_eq!(protocol.constraints.token_budget, 2000);
    }

    #[test]
    fn test_medium_output_constraints() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::General,
            ComplexityLevel::Simple,
            TaskDomain::General,
            OutputSize::Medium,
        );

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.constraints.time_budget_ms, 30_000);
        assert_eq!(protocol.constraints.token_budget, 8000);
    }

    #[test]
    fn test_large_output_constraints() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::General,
            ComplexityLevel::Simple,
            TaskDomain::General,
            OutputSize::Large,
        );

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.constraints.time_budget_ms, 60_000);
        assert_eq!(protocol.constraints.token_budget, 32000);
    }

    #[test]
    fn test_constraints_have_empty_dependencies() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert!(protocol.constraints.dependencies.is_empty());
    }

    // ============================================================================
    // OPTIMIZATION TESTS
    // ============================================================================

    #[test]
    fn test_optimizations_target_parameters() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.m2_optimizations.target_parameters, 200_000_000_000);
    }

    #[test]
    fn test_context_optimization_defaults() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(
            protocol.m2_optimizations.context_optimization.method,
            "auto"
        );
        assert_eq!(
            protocol
                .m2_optimizations
                .context_optimization
                .compression_ratio,
            0.8
        );
    }

    #[test]
    fn test_output_optimization_defaults() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        let output_opt = &protocol.m2_optimizations.output_optimization;
        assert_eq!(output_opt.format, "markdown");
        assert_eq!(output_opt.template, "standard");
        assert_eq!(output_opt.max_output_length, 16000);
        assert!(output_opt.streaming_enabled);
        assert!(!output_opt.compression_enabled);
    }

    #[test]
    fn test_cost_optimization_defaults() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        let cost_opt = &protocol.m2_optimizations.cost_optimization;
        assert_eq!(cost_opt.strategy, "balanced");
        assert_eq!(cost_opt.max_budget, 5.0);
        assert_eq!(cost_opt.target_cost_reduction, 0.5);
        assert_eq!(cost_opt.target_latency_reduction, 0.2);
        assert!(cost_opt.parallel_processing_enabled);
        assert!(cost_opt.caching_enabled);
    }

    // ============================================================================
    // SERIALIZATION / DESERIALIZATION TESTS
    // ============================================================================

    #[test]
    fn test_protocol_serialization_roundtrip() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&protocol).expect("Serialization should succeed");

        // Deserialize back
        let deserialized: InterleavedProtocol =
            serde_json::from_str(&json).expect("Deserialization should succeed");

        // Verify key fields match
        assert_eq!(protocol.id, deserialized.id);
        assert_eq!(protocol.name, deserialized.name);
        assert_eq!(protocol.version, deserialized.version);
        assert_eq!(protocol.description, deserialized.description);
        assert_eq!(protocol.phases.len(), deserialized.phases.len());
    }

    #[test]
    fn test_protocol_pretty_json_serialization() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        // Pretty print should succeed
        let pretty_json =
            serde_json::to_string_pretty(&protocol).expect("Pretty serialization should succeed");

        // Should contain expected structure
        assert!(pretty_json.contains("\"phases\""));
        assert!(pretty_json.contains("\"constraints\""));
        assert!(pretty_json.contains("\"m2_optimizations\""));
    }

    #[test]
    fn test_phase_serialization_roundtrip() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();
        let original_phase = &protocol.phases[0];

        let json = serde_json::to_string(original_phase).expect("Phase serialization should work");
        let deserialized: InterleavedPhase =
            serde_json::from_str(&json).expect("Phase deserialization should work");

        assert_eq!(original_phase.name, deserialized.name);
        assert_eq!(
            original_phase.parallel_branches,
            deserialized.parallel_branches
        );
        assert_eq!(
            original_phase.required_confidence,
            deserialized.required_confidence
        );
    }

    #[test]
    fn test_constraints_serialization_roundtrip() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();
        let original = &protocol.constraints;

        let json = serde_json::to_string(original).expect("Constraints serialization should work");
        let deserialized: CompositeConstraints =
            serde_json::from_str(&json).expect("Constraints deserialization should work");

        assert_eq!(original.time_budget_ms, deserialized.time_budget_ms);
        assert_eq!(original.token_budget, deserialized.token_budget);
        assert_eq!(original.dependencies.len(), deserialized.dependencies.len());
    }

    #[test]
    fn test_optimizations_serialization_roundtrip() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();
        let original = &protocol.m2_optimizations;

        let json =
            serde_json::to_string(original).expect("Optimizations serialization should work");
        let deserialized: M2Optimizations =
            serde_json::from_str(&json).expect("Optimizations deserialization should work");

        assert_eq!(original.target_parameters, deserialized.target_parameters);
        assert_eq!(
            original.context_optimization.method,
            deserialized.context_optimization.method
        );
        assert_eq!(
            original.output_optimization.format,
            deserialized.output_optimization.format
        );
        assert_eq!(
            original.cost_optimization.strategy,
            deserialized.cost_optimization.strategy
        );
    }

    // ============================================================================
    // VARIOUS PROTOCOL TYPES / TASK TYPES TESTS
    // ============================================================================

    #[test]
    fn test_code_analysis_protocol() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::CodeAnalysis,
            ComplexityLevel::Complex,
            TaskDomain::SystemProgramming,
            OutputSize::Large,
        );

        let protocol = generator
            .generate_protocol(&classification, Some("code-analysis".into()))
            .unwrap();

        assert!(protocol.description.contains("CodeAnalysis"));
        assert_eq!(protocol.phases.len(), 4); // Complex has 4 phases
    }

    #[test]
    fn test_bug_finding_protocol() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::BugFinding,
            ComplexityLevel::Moderate,
            TaskDomain::SystemProgramming,
            OutputSize::Medium,
        );

        let protocol = generator
            .generate_protocol(&classification, Some("bug-finding".into()))
            .unwrap();

        assert!(protocol.description.contains("BugFinding"));
        assert_eq!(protocol.phases.len(), 3); // Moderate has 3 phases
    }

    #[test]
    fn test_documentation_protocol() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::Documentation,
            ComplexityLevel::Simple,
            TaskDomain::General,
            OutputSize::Medium,
        );

        let protocol = generator
            .generate_protocol(&classification, Some("docs".into()))
            .unwrap();

        assert!(protocol.description.contains("Documentation"));
        assert_eq!(protocol.phases.len(), 2); // Simple has 2 phases
    }

    #[test]
    fn test_architecture_protocol() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::Architecture,
            ComplexityLevel::Complex,
            TaskDomain::General,
            OutputSize::Large,
        );

        let protocol = generator
            .generate_protocol(&classification, Some("architecture".into()))
            .unwrap();

        assert!(protocol.description.contains("Architecture"));
        assert_eq!(protocol.phases.len(), 4); // Complex has 4 phases
    }

    #[test]
    fn test_general_protocol() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::General,
            ComplexityLevel::Moderate,
            TaskDomain::General,
            OutputSize::Small,
        );

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert!(protocol.description.contains("General"));
    }

    // ============================================================================
    // DOMAIN VARIATION TESTS
    // ============================================================================

    #[test]
    fn test_system_programming_domain() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::CodeAnalysis,
            ComplexityLevel::Complex,
            TaskDomain::SystemProgramming,
            OutputSize::Large,
        );

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        // Protocol should be generated regardless of domain
        assert!(!protocol.id.is_empty());
        assert_eq!(protocol.phases.len(), 4);
    }

    #[test]
    fn test_web_domain() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::CodeAnalysis,
            ComplexityLevel::Moderate,
            TaskDomain::Web,
            OutputSize::Medium,
        );

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert!(!protocol.id.is_empty());
        assert_eq!(protocol.phases.len(), 3);
    }

    #[test]
    fn test_data_domain() {
        let generator = ProtocolGenerator::new();
        let classification = create_classification(
            TaskType::General,
            ComplexityLevel::Simple,
            TaskDomain::Data,
            OutputSize::Small,
        );

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert!(!protocol.id.is_empty());
        assert_eq!(protocol.phases.len(), 2);
    }

    // ============================================================================
    // EDGE CASE TESTS
    // ============================================================================

    #[test]
    fn test_empty_name_option() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        // None should generate auto name
        let protocol = generator.generate_protocol(&classification, None).unwrap();
        assert!(protocol.name.starts_with("m2-protocol-"));

        // Empty string should be preserved (if explicitly provided)
        let protocol = generator
            .generate_protocol(&classification, Some(String::new()))
            .unwrap();
        assert!(protocol.name.is_empty());
    }

    #[test]
    fn test_special_characters_in_name() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let special_name = "protocol-with-special_chars.v1.2.3!@#$%";
        let protocol = generator
            .generate_protocol(&classification, Some(special_name.to_string()))
            .unwrap();

        assert_eq!(protocol.name, special_name);
    }

    #[test]
    fn test_unicode_in_name() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let unicode_name = "protocol-unicode";
        let protocol = generator
            .generate_protocol(&classification, Some(unicode_name.to_string()))
            .unwrap();

        assert_eq!(protocol.name, unicode_name);
    }

    #[test]
    fn test_generator_is_stateless() {
        let generator = ProtocolGenerator::new();
        let classification1 = simple_classification();
        let classification2 = complex_classification();

        // Generate multiple protocols
        let p1 = generator.generate_protocol(&classification1, None).unwrap();
        let p2 = generator.generate_protocol(&classification2, None).unwrap();
        let p3 = generator.generate_protocol(&classification1, None).unwrap();

        // All should be independent
        assert_ne!(p1.id, p2.id);
        assert_ne!(p1.id, p3.id);
        assert_ne!(p2.id, p3.id);

        // Same classification should produce same structure
        assert_eq!(p1.phases.len(), p3.phases.len());
        assert_ne!(p1.phases.len(), p2.phases.len());
    }

    #[test]
    fn test_generator_clone() {
        let generator1 = ProtocolGenerator::new();
        let generator2 = generator1.clone();

        let classification = simple_classification();

        let p1 = generator1.generate_protocol(&classification, None).unwrap();
        let p2 = generator2.generate_protocol(&classification, None).unwrap();

        // Both should produce valid protocols
        assert!(!p1.id.is_empty());
        assert!(!p2.id.is_empty());
        assert_eq!(p1.phases.len(), p2.phases.len());
    }

    #[test]
    fn test_generator_default() {
        let generator = ProtocolGenerator;
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert!(!protocol.id.is_empty());
        assert!(!protocol.name.is_empty());
    }

    // ============================================================================
    // VALIDATION METHOD AND SYNTHESIS METHOD TESTS
    // ============================================================================

    #[test]
    fn test_validation_methods_are_consistent() {
        let generator = ProtocolGenerator::new();

        // Test across all complexity levels
        for complexity in [
            ComplexityLevel::Simple,
            ComplexityLevel::Moderate,
            ComplexityLevel::Complex,
        ] {
            let classification = create_classification(
                TaskType::General,
                complexity,
                TaskDomain::General,
                OutputSize::Medium,
            );

            let protocol = generator.generate_protocol(&classification, None).unwrap();

            for phase in &protocol.phases {
                assert_eq!(phase.validation_methods.len(), 1);
                assert_eq!(phase.validation_methods[0], ValidationMethod::SelfCheck);
            }
        }
    }

    #[test]
    fn test_synthesis_methods_are_consistent() {
        let generator = ProtocolGenerator::new();

        // Test across all complexity levels
        for complexity in [
            ComplexityLevel::Simple,
            ComplexityLevel::Moderate,
            ComplexityLevel::Complex,
        ] {
            let classification = create_classification(
                TaskType::General,
                complexity,
                TaskDomain::General,
                OutputSize::Medium,
            );

            let protocol = generator.generate_protocol(&classification, None).unwrap();

            for phase in &protocol.phases {
                assert_eq!(phase.synthesis_methods.len(), 1);
                assert_eq!(phase.synthesis_methods[0], SynthesisMethod::WeightedAverage);
            }
        }
    }

    // ============================================================================
    // PROTOCOL SCHEMA VALIDATION TESTS
    // ============================================================================

    #[test]
    fn test_protocol_has_all_required_fields() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        // Verify all required fields are present and not empty/default
        assert!(!protocol.id.is_empty());
        assert!(!protocol.name.is_empty());
        assert!(!protocol.version.is_empty());
        assert!(!protocol.description.is_empty());
        assert!(!protocol.phases.is_empty());
    }

    #[test]
    fn test_phases_have_all_required_fields() {
        let generator = ProtocolGenerator::new();
        let classification = complex_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        for phase in &protocol.phases {
            assert!(!phase.name.is_empty());
            assert!(phase.parallel_branches >= 1);
            assert!(phase.required_confidence > 0.0 && phase.required_confidence <= 1.0);
            assert!(!phase.validation_methods.is_empty());
            assert!(!phase.synthesis_methods.is_empty());
        }
    }

    #[test]
    fn test_confidence_values_are_valid() {
        let generator = ProtocolGenerator::new();

        // Test all complexity levels
        for complexity in [
            ComplexityLevel::Simple,
            ComplexityLevel::Moderate,
            ComplexityLevel::Complex,
        ] {
            let classification = create_classification(
                TaskType::General,
                complexity,
                TaskDomain::General,
                OutputSize::Medium,
            );

            let protocol = generator.generate_protocol(&classification, None).unwrap();

            for phase in &protocol.phases {
                assert!(
                    phase.required_confidence >= 0.0,
                    "Confidence should be non-negative"
                );
                assert!(
                    phase.required_confidence <= 1.0,
                    "Confidence should not exceed 1.0"
                );
            }
        }
    }

    #[test]
    fn test_token_budgets_are_positive() {
        let generator = ProtocolGenerator::new();

        for output_size in [OutputSize::Small, OutputSize::Medium, OutputSize::Large] {
            let classification = create_classification(
                TaskType::General,
                ComplexityLevel::Simple,
                TaskDomain::General,
                output_size,
            );

            let protocol = generator.generate_protocol(&classification, None).unwrap();

            assert!(
                protocol.constraints.token_budget > 0,
                "Token budget should be positive"
            );
        }
    }

    #[test]
    fn test_time_budgets_are_positive() {
        let generator = ProtocolGenerator::new();

        for output_size in [OutputSize::Small, OutputSize::Medium, OutputSize::Large] {
            let classification = create_classification(
                TaskType::General,
                ComplexityLevel::Simple,
                TaskDomain::General,
                output_size,
            );

            let protocol = generator.generate_protocol(&classification, None).unwrap();

            assert!(
                protocol.constraints.time_budget_ms > 0,
                "Time budget should be positive"
            );
        }
    }

    // ============================================================================
    // FRAMEWORK AND LANGUAGE SUPPORT TESTS
    // ============================================================================

    #[test]
    fn test_framework_compatibility_is_empty_by_default() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert!(protocol.framework_compatibility.is_empty());
    }

    #[test]
    fn test_language_support_is_empty_by_default() {
        let generator = ProtocolGenerator::new();
        let classification = simple_classification();

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert!(protocol.language_support.is_empty());
    }

    // ============================================================================
    // TASK CLASSIFICATION FROM USE CASE TESTS
    // ============================================================================

    #[test]
    fn test_use_case_code_analysis_classification() {
        let classification = TaskClassification::from(UseCase::CodeAnalysis);

        assert_eq!(classification.task_type, TaskType::CodeAnalysis);
        assert_eq!(classification.complexity_level, ComplexityLevel::Complex);
        assert_eq!(classification.domain, TaskDomain::SystemProgramming);
        assert_eq!(classification.expected_output_size, OutputSize::Large);
    }

    #[test]
    fn test_use_case_bug_finding_classification() {
        let classification = TaskClassification::from(UseCase::BugFinding);

        assert_eq!(classification.task_type, TaskType::BugFinding);
        assert_eq!(classification.complexity_level, ComplexityLevel::Moderate);
        assert_eq!(classification.domain, TaskDomain::SystemProgramming);
        assert_eq!(classification.expected_output_size, OutputSize::Medium);
    }

    #[test]
    fn test_use_case_documentation_classification() {
        let classification = TaskClassification::from(UseCase::Documentation);

        assert_eq!(classification.task_type, TaskType::Documentation);
        assert_eq!(classification.complexity_level, ComplexityLevel::Moderate);
        assert_eq!(classification.domain, TaskDomain::General);
        assert_eq!(classification.expected_output_size, OutputSize::Medium);
    }

    #[test]
    fn test_use_case_architecture_classification() {
        let classification = TaskClassification::from(UseCase::Architecture);

        assert_eq!(classification.task_type, TaskType::Architecture);
        assert_eq!(classification.complexity_level, ComplexityLevel::Complex);
        assert_eq!(classification.domain, TaskDomain::General);
        assert_eq!(classification.expected_output_size, OutputSize::Large);
    }

    #[test]
    fn test_use_case_general_classification() {
        let classification = TaskClassification::from(UseCase::General);

        assert_eq!(classification.task_type, TaskType::General);
        assert_eq!(classification.complexity_level, ComplexityLevel::Moderate);
        assert_eq!(classification.domain, TaskDomain::General);
        assert_eq!(classification.expected_output_size, OutputSize::Medium);
    }

    // ============================================================================
    // INTEGRATION TESTS - COMPLETE WORKFLOW
    // ============================================================================

    #[test]
    fn test_complete_protocol_generation_workflow() {
        let generator = ProtocolGenerator::new();

        // Generate from UseCase
        let use_case = UseCase::CodeAnalysis;
        let classification = TaskClassification::from(use_case);

        let protocol = generator
            .generate_protocol(&classification, Some("integration-test".into()))
            .unwrap();

        // Verify complete protocol structure
        assert_eq!(protocol.name, "integration-test");
        assert!(!protocol.id.is_empty());
        assert_eq!(protocol.version, "1.0.0");

        // Complex task should have 4 phases
        assert_eq!(protocol.phases.len(), 4);

        // Large output should have high token budget
        assert_eq!(protocol.constraints.token_budget, 32000);

        // Serialize and verify it's valid JSON
        let json = serde_json::to_string(&protocol).unwrap();
        assert!(!json.is_empty());

        // Deserialize and verify roundtrip
        let deserialized: InterleavedProtocol = serde_json::from_str(&json).unwrap();
        assert_eq!(protocol.id, deserialized.id);
    }

    #[test]
    fn test_multiple_protocols_for_all_use_cases() {
        let generator = ProtocolGenerator::new();

        let use_cases = vec![
            UseCase::CodeAnalysis,
            UseCase::BugFinding,
            UseCase::Documentation,
            UseCase::Architecture,
            UseCase::General,
        ];

        for use_case in use_cases {
            let classification = TaskClassification::from(use_case);
            let protocol = generator.generate_protocol(&classification, None);

            assert!(
                protocol.is_ok(),
                "Protocol generation should succeed for {:?}",
                use_case
            );

            let protocol = protocol.unwrap();
            assert!(!protocol.id.is_empty());
            assert!(!protocol.phases.is_empty());
        }
    }
}
