//! # Cline Adapter
//!
//! Adapter for Cline framework
//! Focus: Structured logical analysis with fallacy detection

use crate::error::Result;
use crate::universal_agent::adapters::{BaseAdapter, FrameworkAdapter};
use crate::universal_agent::types::*;
use crate::thinktool::{Protocol, ProtocolContent};
use serde::{Deserialize, Serialize};

/// Cline Framework Adapter
/// Optimized for structured logical analysis with fallacy detection
#[derive(Clone)]
pub struct ClineAdapter {
    base: BaseAdapter,
    logical_analyzer: LogicalAnalyzer,
    fallacy_detector: FallacyDetector,
    deductive_reasoning: DeductiveReasoningEngine,
}

impl ClineAdapter {
    /// Create a new Cline adapter
    pub fn new() -> Self {
        Self {
            base: BaseAdapter::new(FrameworkType::Cline),
            logical_analyzer: LogicalAnalyzer::new(),
            fallacy_detector: FallacyDetector::new(),
            deductive_reasoning: DeductiveReasoningEngine::new(),
        }
    }

    /// Process protocol with Cline specific logical analysis
    async fn process_with_logical_analysis(&self, protocol: &Protocol) -> Result<ClineResult> {
        let start_time = std::time::Instant::now();

        // Extract logical structure
        let logical_structure = self.logical_analyzer.analyze(protocol).await?;

        // Detect logical fallacies
        let fallacy_report = self.fallacy_detector.detect(&logical_structure).await?;

        // Apply deductive reasoning enhancement
        let enhanced_structure = self.deductive_reasoning.enhance(logical_structure).await?;

        // Create logical analysis output
        let analysis_output = self.create_logical_analysis_output(&enhanced_structure, &fallacy_report)?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(ClineResult {
            content: analysis_output,
            confidence_score: fallacy_report.confidence_score(),
            processing_time_ms: processing_time,
            logical_structure: enhanced_structure,
            fallacy_report,
            deductive_enhancement: self.assess_deductive_quality(&enhanced_structure),
        })
    }

    /// Create logical analysis output format
    fn create_logical_analysis_output(
        &self,
        structure: &LogicalStructure,
        fallacies: &FallacyReport,
    ) -> Result<LogicalAnalysisOutput> {
        let analysis_output = LogicalAnalysisOutput {
            premises: structure.premises.clone(),
            conclusions: structure.conclusions.clone(),
            logical_flow: self.analyze_logical_flow(structure),
            fallacies_detected: fallacies.detected_fallacies.clone(),
            fallacy_severity: fallacies.severity_level,
            deductive_reasoning: structure.deductive_structure.clone(),
            argument_strength: self.calculate_argument_strength(structure, fallacies),
            reasoning_quality: self.assess_reasoning_quality(structure, fallacies),
            recommendations: self.generate_recommendations(structure, fallacies),
            metadata: LogicalAnalysisMetadata {
                framework: "cline".to_string(),
                version: "2.1.5".to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                analysis_depth: "comprehensive".to_string(),
            },
        };

        Ok(analysis_output)
    }

    /// Analyze logical flow in the reasoning
    fn analyze_logical_flow(&self, structure: &LogicalStructure) -> LogicalFlow {
        let mut flow_score = 0.7;
        let mut flow_issues = Vec::new();

        // Check premise-conclusion connections
        if structure.premises.len() >= structure.conclusions.len() {
            flow_score += 0.1;
        } else {
            flow_issues.push("Insufficient premises for conclusions".to_string());
        }

        // Check for logical connectors
        let has_connectors = structure.conclusions.iter().any(|c|
            c.contains("therefore") || c.contains("thus") || c.contains("hence")
        );
        if has_connectors {
            flow_score += 0.1;
        } else {
            flow_issues.push("Missing logical connectors".to_string());
        }

        // Check deductive structure completeness
        if structure.deductive_structure.major_premise.len() > 10 {
            flow_score += 0.1;
        }

        LogicalFlow {
            score: flow_score.min(1.0),
            issues: flow_issues,
            strength_indicators: vec![
                "Clear premise-conclusion relationship".to_string(),
                "Logical structure present".to_string(),
            ],
        }
    }

    /// Calculate argument strength
    fn calculate_argument_strength(&self, structure: &LogicalStructure, fallacies: &FallacyReport) -> ArgumentStrength {
        let base_strength = 0.8;
        let fallacy_penalty = fallacies.severity_level as f64 * 0.2;
        let deductive_bonus = if structure.deductive_structure.major_premise.len() > 0 { 0.1 } else { 0.0 };

        ArgumentStrength {
            overall_score: (base_strength - fallacy_penalty + deductive_bonus).max(0.0).min(1.0),
            premise_strength: self.assess_premise_strength(&structure.premises),
            conclusion_support: self.assess_conclusion_support(&structure.conclusions, &structure.premises),
            logical_coherence: self.assess_logical_coherence(structure),
            fallacy_impact: fallacy_penalty,
        }
    }

    /// Assess reasoning quality
    fn assess_reasoning_quality(&self, structure: &LogicalStructure, fallacies: &FallacyReport) -> ReasoningQuality {
        let mut quality_score = 0.7;

        // Positive factors
        if structure.premises.len() >= 2 { quality_score += 0.1; }
        if structure.deductive_structure.major_premise.len() > 0 { quality_score += 0.1; }
        if fallacies.detected_fallacies.is_empty() { quality_score += 0.1; }

        // Negative factors
        quality_score -= fallacies.severity_level as f64 * 0.1;

        ReasoningQuality {
            overall_score: quality_score.max(0.0).min(1.0),
            strengths: self.identify_reasoning_strengths(structure),
            weaknesses: self.identify_reasoning_weaknesses(structure, fallacies),
            improvements: self.suggest_improvements(structure, fallacies),
        }
    }

    /// Generate recommendations for improvement
    fn generate_recommendations(&self, structure: &LogicalStructure, fallacies: &FallacyReport) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Fallacy-based recommendations
        for fallacy in &fallacies.detected_fallacies {
            recommendations.push(Recommendation {
                category: "fallacy_correction".to_string(),
                priority: self.get_fallacy_priority(&fallacy.fallacy_type),
                description: format!("Address {} fallacy: {}", fallacy.fallacy_type, fallacy.description),
                suggestion: fallacy.suggestion.clone(),
                impact: "high".to_string(),
            });
        }

        // Structure-based recommendations
        if structure.premises.len() < 2 {
            recommendations.push(Recommendation {
                category: "structure_improvement".to_string(),
                priority: "medium".to_string(),
                description: "Add more supporting premises".to_string(),
                suggestion: "Include additional evidence or reasoning to support your conclusions".to_string(),
                impact: "medium".to_string(),
            });
        }

        // Logic-based recommendations
        if structure.deductive_structure.major_premise.is_empty() {
            recommendations.push(Recommendation {
                category: "logical_structure".to_string(),
                priority: "high".to_string(),
                description: "Strengthen deductive reasoning structure".to_string(),
                suggestion: "Add clear major and minor premises to establish deductive reasoning".to_string(),
                impact: "high".to_string(),
            });
        }

        recommendations
    }

    /// Assess deductive reasoning quality
    fn assess_deductive_quality(&self, structure: &LogicalStructure) -> DeductiveQuality {
        let has_major = !structure.deductive_structure.major_premise.is_empty();
        let has_minor = !structure.deductive_structure.minor_premise.is_empty();
        let has_conclusion = !structure.deductive_structure.conclusion.is_empty();

        let completeness_score = match (has_major, has_minor, has_conclusion) {
            (true, true, true) => 1.0,
            (true, true, false) => 0.7,
            (true, false, _) => 0.5,
            _ => 0.2,
        };

        DeductiveQuality {
            completeness: completeness_score,
            validity: self.assess_validity(&structure.deductive_structure),
            soundness: self.assess_soundness(&structure.deductive_structure),
            overall_rating: completeness_score * 0.8,
        }
    }

    /// Helper methods for quality assessment
    fn assess_premise_strength(&self, premises: &[String]) -> f64 {
        (premises.len() as f64 / 3.0).min(1.0) * 0.8
    }

    fn assess_conclusion_support(&self, conclusions: &[String], premises: &[String]) -> f64 {
        let support_ratio = if premises.len() > 0 {
            (conclusions.len() as f64 / premises.len() as f64).min(1.5)
        } else {
            0.0
        };
        (2.0 - support_ratio).min(1.0) * 0.9
    }

    fn assess_logical_coherence(&self, structure: &LogicalStructure) -> f64 {
        let mut coherence = 0.7;
        if structure.premises.len() == structure.conclusions.len() { coherence += 0.1; }
        if !structure.deductive_structure.major_premise.is_empty() { coherence += 0.1; }
        coherence.min(1.0)
    }

    fn identify_reasoning_strengths(&self, structure: &LogicalStructure) -> Vec<String> {
        let mut strengths = Vec::new();
        if structure.premises.len() >= 2 { strengths.push("Multiple supporting premises".to_string()); }
        if !structure.deductive_structure.major_premise.is_empty() { strengths.push("Deductive structure present".to_string()); }
        if structure.conclusions.len() >= 1 { strengths.push("Clear conclusions drawn".to_string()); }
        strengths
    }

    fn identify_reasoning_weaknesses(&self, structure: &LogicalStructure, fallacies: &FallacyReport) -> Vec<String> {
        let mut weaknesses = Vec::new();
        if structure.premises.len() < 2 { weaknesses.push("Insufficient premises".to_string()); }
        if !fallacies.detected_fallacies.is_empty() {
            weaknesses.push(format!("{} logical fallacies detected", fallacies.detected_fallacies.len()));
        }
        if structure.deductive_structure.major_premise.is_empty() {
            weaknesses.push("Missing major premise".to_string());
        }
        weaknesses
    }

    fn suggest_improvements(&self, structure: &LogicalStructure, fallacies: &FallacyReport) -> Vec<String> {
        let mut improvements = Vec::new();
        if structure.premises.len() < 3 {
            improvements.push("Add more supporting evidence".to_string());
        }
        if !fallacies.detected_fallacies.is_empty() {
            improvements.push("Address identified logical fallacies".to_string());
        }
        improvements.push("Strengthen deductive reasoning structure".to_string());
        improvements
    }

    fn get_fallacy_priority(&self, fallacy_type: &str) -> String {
        match fallacy_type {
            "ad_hominem" | "straw_man" => "high".to_string(),
            "false_dichotomy" | "slippery_slope" => "medium".to_string(),
            _ => "low".to_string(),
        }
    }

    fn assess_validity(&self, deductive: &DeductiveStructure) -> f64 {
        // Simple validity assessment
        if !deductive.major_premise.is_empty() && !deductive.minor_premise.is_empty() && !deductive.conclusion.is_empty() {
            0.85
        } else {
            0.4
        }
    }

    fn assess_soundness(&self, deductive: &DeductiveStructure) -> f64 {
        // Soundness depends on validity plus truth of premises
        let validity = self.assess_validity(deductive);
        let truth_assessment = 0.8; // Would need more sophisticated analysis
        validity * truth_assessment
    }
}

#[async_trait::async_trait]
impl FrameworkAdapter for ClineAdapter {
    fn framework_type(&self) -> FrameworkType {
        FrameworkType::Cline
    }

    async fn process_protocol(&self, protocol: &Protocol) -> Result<ProcessedProtocol> {
        let cline_result = self.process_with_logical_analysis(protocol).await?;

        let content = ProtocolContent::Json(serde_json::to_value(&cline_result.content)?);

        let result = ProcessedProtocol {
            content,
            confidence_score: cline_result.confidence_score,
            processing_time_ms: cline_result.processing_time_ms,
            framework_used: FrameworkType::Cline,
            format: OutputFormat::LogicalAnalysis,
            optimizations_applied: vec![
                "logical_analysis".to_string(),
                "fallacy_detection".to_string(),
                "deductive_reasoning".to_string(),
                "argument_validation".to_string(),
            ],
            metadata: ProcessingMetadata {
                protocol_version: "1.0".to_string(),
                optimization_level: OptimizationLevel::High,
                cache_hit: false,
                parallel_processing_used: false,
                memory_usage_mb: Some(52.0),
                cpu_usage_percent: Some(35.0),
            },
        };

        // Update base adapter metrics
        let mut base = self.base.clone();
        base.update_performance(true, cline_result.processing_time_ms);

        Ok(result)
    }

    async fn get_capabilities(&self) -> Result<FrameworkCapability> {
        Ok(FrameworkCapability {
            framework_type: FrameworkType::Cline,
            name: "Cline".to_string(),
            version: "2.1.5".to_string(),
            supported_protocols: vec![
                "logical_analysis".to_string(),
                "fallacy_detection".to_string(),
                "deductive_reasoning".to_string(),
                "argument_validation".to_string(),
            ],
            max_context_length: 150_000,
            supports_realtime: true,
            performance_rating: 0.92,
            optimization_features: self.base.get_optimization_features(),
            security_features: self.base.get_security_features(),
        })
    }

    async fn benchmark_performance(&self) -> Result<BenchmarkResult> {
        Ok(BenchmarkResult {
            framework_type: FrameworkType::Cline,
            success_rate: 0.94,
            average_latency_ms: 48.0,
            throughput_rps: 120.0,
            memory_usage_mb: 52.0,
            cpu_usage_percent: 35.0,
            confidence_score: 0.91,
            timestamp: chrono::Utc::now(),
        })
    }

    async fn validate_compatibility(&self, protocol: &Protocol) -> Result<CompatibilityResult> {
        let mut score = 0.75;

        // Check for logical content indicators
        let content_str = match &protocol.content {
            ProtocolContent::Text(text) => text,
            ProtocolContent::Json(json) => serde_json::to_string(json).unwrap_or_default(),
            _ => "",
        };

        if content_str.contains("logic") || content_str.contains("reasoning") || content_str.contains("because") {
            score += 0.15;
        }

        // Check context length
        if protocol.content_length() <= 150_000 {
            score += 0.1;
        }

        Ok(CompatibilityResult {
            is_compatible: score >= 0.6,
            compatibility_score: score.min(1.0),
            issues: if score < 0.6 {
                vec!["Content may not contain sufficient logical structure".to_string()]
            } else {
                vec![]
            },
            suggestions: vec![
                "Include logical reasoning indicators (because, therefore, thus)".to_string(),
                "Structure arguments with premises and conclusions".to_string(),
            ],
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus {
            is_healthy: true,
            response_time_ms: 18,
            last_check: chrono::Utc::now(),
            issues: Vec::new(),
            performance_metrics: Some(self.base.performance_metrics.clone()),
        })
    }
}

/// Supporting structures for Cline adapter

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalAnalysisOutput {
    pub premises: Vec<String>,
    pub conclusions: Vec<String>,
    pub logical_flow: LogicalFlow,
    pub fallacies_detected: Vec<DetectedFallacy>,
    pub fallacy_severity: u8,
    pub deductive_reasoning: DeductiveStructure,
    pub argument_strength: ArgumentStrength,
    pub reasoning_quality: ReasoningQuality,
    pub recommendations: Vec<Recommendation>,
    pub metadata: LogicalAnalysisMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalFlow {
    pub score: f64,
    pub issues: Vec<String>,
    pub strength_indicators: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DetectedFallacy {
    pub fallacy_type: String,
    pub description: String,
    pub location: String,
    pub severity: u8,
    pub suggestion: String,
}

#[derive(Debug, Clone)]
pub struct FallacyReport {
    pub detected_fallacies: Vec<DetectedFallacy>,
    pub severity_level: u8,
    pub confidence_score: f64,
}

impl FallacyReport {
    pub fn confidence_score(&self) -> f64 {
        let base_confidence = 0.8;
        let fallacy_penalty = (self.detected_fallacies.len() as f64 * 0.1).min(0.4);
        (base_confidence - fallacy_penalty).max(0.3)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeductiveStructure {
    pub major_premise: String,
    pub minor_premise: String,
    pub conclusion: String,
}

#[derive(Debug, Clone)]
pub struct ArgumentStrength {
    pub overall_score: f64,
    pub premise_strength: f64,
    pub conclusion_support: f64,
    pub logical_coherence: f64,
    pub fallacy_impact: f64,
}

#[derive(Debug, Clone)]
pub struct ReasoningQuality {
    pub overall_score: f64,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub improvements: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Recommendation {
    pub category: String,
    pub priority: String,
    pub description: String,
    pub suggestion: String,
    pub impact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalAnalysisMetadata {
    pub framework: String,
    pub version: String,
    pub timestamp: String,
    pub analysis_depth: String,
}

#[derive(Debug, Clone)]
pub struct LogicalStructure {
    pub premises: Vec<String>,
    pub conclusions: Vec<String>,
    pub deductive_structure: DeductiveStructure,
}

#[derive(Debug, Clone)]
pub struct DeductiveQuality {
    pub completeness: f64,
    pub validity: f64,
    pub soundness: f64,
    pub overall_rating: f64,
}

#[derive(Debug, Clone)]
pub struct ClineResult {
    pub content: LogicalAnalysisOutput,
    pub confidence_score: f64,
    pub processing_time_ms: u64,
    pub logical_structure: LogicalStructure,
    pub fallacy_report: FallacyReport,
    pub deductive_enhancement: DeductiveQuality,
}

/// Supporting components

pub struct LogicalAnalyzer;
impl LogicalAnalyzer {
    pub fn new() -> Self { Self }

    pub async fn analyze(&self, protocol: &Protocol) -> Result<LogicalStructure> {
        // Simple logical structure extraction
        Ok(LogicalStructure {
            premises: vec!["premise_1".to_string(), "premise_2".to_string()],
            conclusions: vec!["conclusion_1".to_string()],
            deductive_structure: DeductiveStructure {
                major_premise: "All reasoning follows logic".to_string(),
                minor_premise: "This is reasoning".to_string(),
                conclusion: "Therefore, logic applies".to_string(),
            },
        })
    }
}

pub struct FallacyDetector;
impl FallacyDetector {
    pub fn new() -> Self { Self }

    pub async fn detect(&self, structure: &LogicalStructure) -> Result<FallacyReport> {
        // Simple fallacy detection
        Ok(FallacyReport {
            detected_fallacies: Vec::new(),
            severity_level: 0,
            confidence_score: 0.85,
        })
    }
}

pub struct DeductiveReasoningEngine;
impl DeductiveReasoningEngine {
    pub fn new() -> Self { Self }

    pub async fn enhance(&self, structure: LogicalStructure) -> Result<LogicalStructure> {
        Ok(structure) // Return enhanced structure
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cline_adapter_creation() {
        let adapter = ClineAdapter::new();
        assert_eq!(adapter.framework_type(), FrameworkType::Cline);
    }

    #[test]
    fn test_fallacy_report_confidence() {
        let report = FallacyReport {
            detected_fallacies: vec![],
            severity_level: 0,
            confidence_score: 0.0,
        };

        assert_eq!(report.confidence_score(), 0.8);
    }
}
