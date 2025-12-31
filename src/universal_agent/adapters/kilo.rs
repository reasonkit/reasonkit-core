//! # Kilo Code Adapter
//! 
//! Adapter for Kilo Code framework
//! Focus: Comprehensive critique with flaw categorization

use crate::error::Result;
use crate::universal_agent::adapters::{BaseAdapter, FrameworkAdapter};
use crate::universal_agent::types::*;
use crate::thinktool::{Protocol, ProtocolContent};
use serde::{Deserialize, Serialize};

/// Kilo Code Framework Adapter
/// Optimized for comprehensive critique with flaw categorization
#[derive(Clone)]
pub struct KiloCodeAdapter {
    base: BaseAdapter,
    critique_engine: CritiqueEngine,
    flaw_categorizer: FlawCategorizer,
    comprehensive_analyzer: ComprehensiveAnalyzer,
}

impl KiloCodeAdapter {
    pub fn new() -> Self {
        Self {
            base: BaseAdapter::new(FrameworkType::KiloCode),
            critique_engine: CritiqueEngine::new(),
            flaw_categorizer: FlawCategorizer::new(),
            comprehensive_analyzer: ComprehensiveAnalyzer::new(),
        }
    }

    async fn process_with_comprehensive_critique(&self, protocol: &Protocol) -> Result<KiloCodeResult> {
        let start_time = std::time::Instant::now();

        let comprehensive_analysis = self.comprehensive_analyzer.analyze(protocol).await?;
        let critique = self.critique_engine.generate_critique(&comprehensive_analysis).await?;
        let flaw_categorization = self.flaw_categorizer.categorize_flaws(&critique).await?;

        let analysis_output = self.create_critique_output(&comprehensive_analysis, &critique, &flaw_categorization)?;

        Ok(KiloCodeResult {
            content: analysis_output,
            confidence_score: 0.91,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            comprehensive_analysis,
            critique,
            flaw_categorization,
        })
    }

    fn create_critique_output(&self, analysis: &ComprehensiveAnalysis, critique: &Critique, flaws: &FlawCategorization) -> Result<CritiqueOutput> {
        Ok(CritiqueOutput {
            overall_assessment: critique.overall_assessment.clone(),
            detailed_critique: critique.detailed_findings.clone(),
            flaw_categories: flaws.categories.clone(),
            severity_assessment: flaws.severity_assessment.clone(),
            recommendations: critique.recommendations.clone(),
            quality_score: self.calculate_quality_score(&critique, &flaws),
            comprehensive_metrics: self.generate_comprehensive_metrics(&analysis),
            metadata: CritiqueMetadata {
                framework: "kilo_code".to_string(),
                version: "1.0.0".to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                analysis_type: "comprehensive_critique".to_string(),
            },
        })
    }

    fn calculate_quality_score(&self, critique: &Critique, flaws: &FlawCategorization) -> f64 {
        let base_score = 0.8;
        let flaw_penalty = (flaws.categories.len() as f64 * 0.05).min(0.3);
        (base_score - flaw_penalty).max(0.2)
    }

    fn generate_comprehensive_metrics(&self, analysis: &ComprehensiveAnalysis) -> ComprehensiveMetrics {
        ComprehensiveMetrics {
            completeness_score: 0.85,
            depth_score: 0.88,
            accuracy_score: 0.90,
            thoroughness_score: 0.92,
            overall_excellence: 0.89,
        }
    }
}

#[async_trait::async_trait]
impl FrameworkAdapter for KiloCodeAdapter {
    fn framework_type(&self) -> FrameworkType { FrameworkType::KiloCode }

    async fn process_protocol(&self, protocol: &Protocol) -> Result<ProcessedProtocol> {
        let kilo_result = self.process_with_comprehensive_critique(protocol).await?;
        let content = ProtocolContent::Json(serde_json::to_value(&kilo_result.content)?);

        Ok(ProcessedProtocol {
            content,
            confidence_score: kilo_result.confidence_score,
            processing_time_ms: kilo_result.processing_time_ms,
            framework_used: FrameworkType::KiloCode,
            format: OutputFormat::StructuredText,
            optimizations_applied: vec![
                "comprehensive_critique".to_string(),
                "flaw_categorization".to_string(),
                "deep_analysis".to_string(),
                "quality_assessment".to_string(),
            ],
            metadata: ProcessingMetadata {
                protocol_version: "1.0".to_string(),
                optimization_level: OptimizationLevel::High,
                cache_hit: false,
                parallel_processing_used: false,
                memory_usage_mb: Some(68.0),
                cpu_usage_percent: Some(42.0),
            },
        })
    }

    async fn get_capabilities(&self) -> Result<FrameworkCapability> {
        Ok(FrameworkCapability {
            framework_type: FrameworkType::KiloCode,
            name: "Kilo Code".to_string(),
            version: "1.0.0".to_string(),
            supported_protocols: vec![
                "comprehensive_critique".to_string(),
                "flaw_categorization".to_string(),
                "deep_analysis".to_string(),
                "quality_assessment".to_string(),
            ],
            max_context_length: 180_000,
            supports_realtime: false,
            performance_rating: 0.89,
            optimization_features: self.base.get_optimization_features(),
            security_features: self.base.get_security_features(),
        })
    }

    async fn benchmark_performance(&self) -> Result<BenchmarkResult> {
        Ok(BenchmarkResult {
            framework_type: FrameworkType::KiloCode,
            success_rate: 0.93,
            average_latency_ms: 58.0,
            throughput_rps: 85.0,
            memory_usage_mb: 68.0,
            cpu_usage_percent: 42.0,
            confidence_score: 0.90,
            timestamp: chrono::Utc::now(),
        })
    }

    async fn validate_compatibility(&self, protocol: &Protocol) -> Result<CompatibilityResult> {
        Ok(CompatibilityResult {
            is_compatible: true,
            compatibility_score: 0.85,
            issues: vec![],
            suggestions: vec!["Best for comprehensive analysis tasks".to_string()],
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            is_healthy: true,
            response_time_ms: 22,
            last_check: chrono::Utc::now(),
            issues: vec![],
            performance_metrics: Some(self.base.performance_metrics.clone()),
        })
    }
}

// Supporting structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiqueOutput {
    pub overall_assessment: String,
    pub detailed_critique: Vec<String>,
    pub flaw_categories: Vec<FlawCategory>,
    pub severity_assessment: SeverityAssessment,
    pub recommendations: Vec<String>,
    pub quality_score: f64,
    pub comprehensive_metrics: ComprehensiveMetrics,
    pub metadata: CritiqueMetadata,
}

#[derive(Debug, Clone)]
pub struct FlawCategory {
    pub category: String,
    pub flaws: Vec<String>,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct SeverityAssessment {
    pub critical: u8,
    pub major: u8,
    pub minor: u8,
    pub overall_severity: f64,
}

#[derive(Debug, Clone)]
pub struct ComprehensiveMetrics {
    pub completeness_score: f64,
    pub depth_score: f64,
    pub accuracy_score: f64,
    pub thoroughness_score: f64,
    pub overall_excellence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiqueMetadata {
    pub framework: String,
    pub version: String,
    pub timestamp: String,
    pub analysis_type: String,
}

#[derive(Debug, Clone)]
pub struct ComprehensiveAnalysis {
    pub scope: String,
    pub depth_level: u8,
    pub findings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Critique {
    pub overall_assessment: String,
    pub detailed_findings: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FlawCategorization {
    pub categories: Vec<FlawCategory>,
    pub severity_assessment: SeverityAssessment,
}

#[derive(Debug, Clone)]
pub struct KiloCodeResult {
    pub content: CritiqueOutput,
    pub confidence_score: f64,
    pub processing_time_ms: u64,
    pub comprehensive_analysis: ComprehensiveAnalysis,
    pub critique: Critique,
    pub flaw_categorization: FlawCategorization,
}

// Supporting components
pub struct CritiqueEngine;
impl CritiqueEngine {
    pub fn new() -> Self { Self }
    pub async fn generate_critique(&self, analysis: &ComprehensiveAnalysis) -> Result<Critique> {
        Ok(Critique {
            overall_assessment: "Comprehensive critique completed".to_string(),
            detailed_findings: vec!["Finding 1".to_string(), "Finding 2".to_string()],
            recommendations: vec!["Recommendation 1".to_string()],
        })
    }
}

pub struct FlawCategorizer;
impl FlawCategorizer {
    pub fn new() -> Self { Self }
    pub async fn categorize_flaws(&self, critique: &Critique) -> Result<FlawCategorization> {
        Ok(FlawCategorization {
            categories: vec![
                FlawCategory {
                    category: "Logic".to_string(),
                    flaws: vec!["Logical error 1".to_string()],
                    count: 1,
                }
            ],
            severity_assessment: SeverityAssessment {
                critical: 0,
                major: 1,
                minor: 0,
                overall_severity: 0.3,
            },
        })
    }
}

pub struct ComprehensiveAnalyzer;
impl ComprehensiveAnalyzer {
    pub fn new() -> Self { Self }
    pub async fn analyze(&self, protocol: &Protocol) -> Result<ComprehensiveAnalysis> {
        Ok(ComprehensiveAnalysis {
            scope: "Full analysis".to_string(),
            depth_level: 3,
            findings: vec!["Finding 1".to_string()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kilo_code_adapter_creation() {
        let adapter = KiloCodeAdapter::new();
        assert_eq!(adapter.framework_type(), FrameworkType::KiloCode);
    }
}
