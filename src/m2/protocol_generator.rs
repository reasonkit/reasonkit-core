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

    #[test]
    fn test_simple_protocol_generation() {
        let generator = ProtocolGenerator::new();
        let classification = TaskClassification {
            task_type: TaskType::General,
            complexity_level: ComplexityLevel::Simple,
            domain: TaskDomain::General,
            expected_output_size: OutputSize::Small,
            time_constraints: TimeConstraints::default(),
            quality_requirements: QualityRequirements::default(),
        };

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
        let classification = TaskClassification {
            task_type: TaskType::CodeAnalysis,
            complexity_level: ComplexityLevel::Complex,
            domain: TaskDomain::SystemProgramming,
            expected_output_size: OutputSize::Large,
            time_constraints: TimeConstraints::default(),
            quality_requirements: QualityRequirements::default(),
        };

        let protocol = generator.generate_protocol(&classification, None).unwrap();

        assert_eq!(protocol.phases.len(), 4);
        assert_eq!(protocol.phases[1].parallel_branches, 3); // parallel_analysis
        assert!(protocol.constraints.token_budget >= 30000);
    }
}
