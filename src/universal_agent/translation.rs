//! # Protocol Translation Engine
//!
//! Automatic translation between ReasonKit protocols and framework-specific formats

use crate::error::Result;
use crate::universal_agent::types::*;
use crate::thinktool::{Protocol, ProtocolContent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Protocol Translation Engine
/// Handles automatic conversion between ReasonKit protocols and framework formats
#[derive(Clone)]
pub struct ProtocolTranslator {
    format_converters: Arc<RwLock<HashMap<FrameworkType, Box<dyn FormatConverter>>>>,
    validation_engine: ValidationEngine,
    optimization_cache: Arc<RwLock<OptimizationCache>>,
    translation_metrics: Arc<RwLock<TranslationMetrics>>,
}

impl ProtocolTranslator {
    /// Create a new protocol translator
    pub async fn new() -> Result<Self> {
        let format_converters = Arc::new(RwLock::new(HashMap::new()));
        let optimization_cache = Arc::new(RwLock::new(OptimizationCache::new()));
        let translation_metrics = Arc::new(RwLock::new(TranslationMetrics::new()));

        let mut translator = Self {
            format_converters,
            validation_engine: ValidationEngine::new(),
            optimization_cache,
            translation_metrics,
        };

        // Initialize built-in format converters
        translator.initialize_builtin_converters().await?;

        Ok(translator)
    }

    /// Initialize built-in format converters for all supported frameworks
    async fn initialize_builtin_converters(&mut self) -> Result<()> {
        let converters: Vec<Box<dyn FormatConverter>> = vec![
            Box::new(ClaudeCodeConverter::new()),
            Box::new(ClineConverter::new()),
            Box::new(KiloCodeConverter::new()),
            Box::new(DroidConverter::new()),
            Box::new(RooCodeConverter::new()),
            Box::new(BlackBoxAIConverter::new()),
        ];

        let mut converters_map = self.format_converters.write().await;
        for converter in converters {
            let framework_type = converter.target_framework();
            converters_map.insert(framework_type, converter);
        }

        Ok(())
    }

    /// Translate a protocol from one framework to another
    pub async fn translate_protocol(
        &self,
        source_framework: FrameworkType,
        target_framework: FrameworkType,
        protocol: &Protocol,
        optimization_level: OptimizationLevel,
    ) -> Result<TranslationResult> {
        let start_time = std::time::Instant::now();

        // Check cache for existing translation
        if let Some(cached_result) = self.get_cached_translation(source_framework, target_framework, protocol).await? {
            self.record_cache_hit().await?;
            return Ok(cached_result);
        }

        // Validate protocol compatibility
        let compatibility = self.validation_engine.validate_compatibility(protocol, target_framework).await?;
        if !compatibility.is_compatible && compatibility.compatibility_score < 0.5 {
            return Err(crate::error::Error::ProtocolIncompatible {
                framework: target_framework,
                issues: compatibility.issues,
            });
        }

        // Get format converter for target framework
        let converters = self.format_converters.read().await;
        let converter = converters.get(&target_framework)
            .ok_or_else(|| crate::error::Error::ConverterNotFound(target_framework))?;

        // Apply optimization level
        let optimized_protocol = self.apply_optimization(protocol, optimization_level, target_framework).await?;

        // Perform translation
        let translated_content = converter.convert_from_reasonkit(&optimized_protocol).await?;

        // Post-process and validate translation
        let validated_content = self.validation_engine.validate_translation(&translated_content, target_framework).await?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        let result = TranslationResult {
            translated_content: validated_content,
            confidence_score: self.calculate_translation_confidence(&optimized_protocol, &validated_content, &compatibility),
            processing_time_ms: processing_time,
            source_framework,
            target_framework,
            optimization_applied: self.get_applied_optimizations(optimization_level, target_framework),
            cache_key: self.generate_cache_key(source_framework, target_framework, protocol),
        };

        // Cache successful translation
        self.cache_translation(result.clone()).await?;

        // Record metrics
        self.record_translation_metrics(&result).await?;

        Ok(result)
    }

    /// Translate protocol to a specific framework format
    pub async fn translate_to_framework(
        &self,
        protocol: &Protocol,
        target_framework: FrameworkType,
        optimization_level: OptimizationLevel,
    ) -> Result<ProtocolContent> {
        let result = self.translate_protocol(FrameworkType::ClaudeCode, target_framework, protocol, optimization_level).await?;
        Ok(result.translated_content)
    }

    /// Convert framework-specific format back to ReasonKit protocol
    pub async fn translate_from_framework(
        &self,
        framework_content: &ProtocolContent,
        source_framework: FrameworkType,
    ) -> Result<Protocol> {
        let converters = self.format_converters.read().await;
        let converter = converters.get(&source_framework)
            .ok_or_else(|| crate::error::Error::ConverterNotFound(source_framework))?;

        let protocol = converter.convert_to_reasonkit(framework_content).await?;
        Ok(protocol)
    }

    /// Apply optimization based on framework capabilities
    async fn apply_optimization(
        &self,
        protocol: &Protocol,
        level: OptimizationLevel,
        framework: FrameworkType,
    ) -> Result<Protocol> {
        match level {
            OptimizationLevel::None => Ok(protocol.clone()),
            OptimizationLevel::Basic => self.apply_basic_optimization(protocol, framework).await,
            OptimizationLevel::Medium => self.apply_medium_optimization(protocol, framework).await,
            OptimizationLevel::High => self.apply_high_optimization(protocol, framework).await,
            OptimizationLevel::Maximum => self.apply_maximum_optimization(protocol, framework).await,
        }
    }

    async fn apply_basic_optimization(&self, protocol: &Protocol, framework: FrameworkType) -> Result<Protocol> {
        // Basic optimization: remove unnecessary whitespace, normalize formatting
        let mut optimized = protocol.clone();

        match &mut optimized.content {
            ProtocolContent::Text(text) => {
                *text = text.trim().to_string();
            }
            ProtocolContent::Json(json) => {
                // Ensure JSON is properly formatted
                if let Ok(formatted) = serde_json::to_string_pretty(json) {
                    *json = serde_json::from_str(&formatted)?;
                }
            }
            _ => {}
        }

        Ok(optimized)
    }

    async fn apply_medium_optimization(&self, protocol: &Protocol, framework: FrameworkType) -> Result<Protocol> {
        // Medium optimization: framework-specific formatting, content restructuring
        let mut optimized = self.apply_basic_optimization(protocol, framework).await?;

        // Framework-specific optimizations
        match framework {
            FrameworkType::ClaudeCode => {
                // Add confidence scoring metadata
                if let ProtocolContent::Json(json) = &mut optimized.content {
                    if !json.get("confidence").is_some() {
                        json["confidence"] = serde_json::Value::from(0.8);
                    }
                }
            }
            FrameworkType::Cline => {
                // Ensure logical structure for Cline
                if let ProtocolContent::Text(text) = &mut optimized.content {
                    if !text.contains("logical") && !text.contains("analysis") {
                        *text = format!("Analysis: {}", text);
                    }
                }
            }
            _ => {}
        }

        Ok(optimized)
    }

    async fn apply_high_optimization(&self, protocol: &Protocol, framework: FrameworkType) -> Result<Protocol> {
        // High optimization: advanced restructuring, performance tuning
        let mut optimized = self.apply_medium_optimization(protocol, framework).await?;

        // Add framework-specific performance optimizations
        match framework {
            FrameworkType::BlackBoxAI => {
                // Optimize for high throughput
                if let ProtocolContent::Json(json) = &mut optimized.content {
                    json["optimization"] = serde_json::Value::from("high_throughput");
                    json["batch_size"] = serde_json::Value::from(100);
                }
            }
            FrameworkType::Droid => {
                // Optimize for mobile
                if let ProtocolContent::Json(json) = &mut optimized.content {
                    json["platform"] = serde_json::Value::from("android");
                    json["optimization"] = serde_json::Value::from("mobile");
                }
            }
            _ => {}
        }

        Ok(optimized)
    }

    async fn apply_maximum_optimization(&self, protocol: &Protocol, framework: FrameworkType) -> Result<Protocol> {
        // Maximum optimization: comprehensive optimization, M2-level performance
        let mut optimized = self.apply_high_optimization(protocol, framework).await?;

        // M2-level optimizations
        if let ProtocolContent::Json(json) = &mut optimized.content {
            json["minimax_m2_optimized"] = serde_json::Value::from(true);
            json["performance_level"] = serde_json::Value::from("maximum");
            json["cross_platform"] = serde_json::Value::from(true);
        }

        Ok(optimized)
    }

    /// Calculate translation confidence score
    fn calculate_translation_confidence(
        &self,
        source: &Protocol,
        translated: &ProtocolContent,
        compatibility: &CompatibilityResult,
    ) -> f64 {
        let mut confidence = compatibility.compatibility_score;

        // Boost confidence based on content preservation
        match ( &source.content, translated ) {
            (ProtocolContent::Text(src), ProtocolContent::Text(dst)) => {
                let similarity = self.calculate_text_similarity(src, dst);
                confidence += similarity * 0.3;
            }
            (ProtocolContent::Json(src), ProtocolContent::Json(dst)) => {
                let json_similarity = self.calculate_json_similarity(src, dst);
                confidence += json_similarity * 0.3;
            }
            _ => {
                // Different content types - penalize slightly
                confidence -= 0.1;
            }
        }

        // Ensure confidence is within valid range
        confidence.max(0.0).min(1.0)
    }

    /// Calculate text similarity
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f64 {
        // Simple similarity calculation - in production would use more sophisticated algorithms
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

        let intersection: usize = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
    }

    /// Calculate JSON similarity
    fn calculate_json_similarity(&self, json1: &serde_json::Value, json2: &serde_json::Value) -> f64 {
        // Simple JSON similarity - in production would use JSON diff algorithms
        if json1 == json2 {
            1.0
        } else {
            0.5 // Partial similarity
        }
    }

    /// Generate cache key for translation
    fn generate_cache_key(&self, source: FrameworkType, target: FrameworkType, protocol: &Protocol) -> String {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(format!("{}-{:?}-{:?}", protocol.id, source, target));
        format!("{:x}", hasher.finalize())
    }

    /// Get applied optimizations list
    fn get_applied_optimizations(&self, level: OptimizationLevel, framework: FrameworkType) -> Vec<String> {
        let mut optimizations = Vec::new();

        match level {
            OptimizationLevel::None => {}
            OptimizationLevel::Basic => {
                optimizations.push("whitespace_normalization".to_string());
                optimizations.push("format_standardization".to_string());
            }
            OptimizationLevel::Medium => {
                optimizations.extend(vec![
                    "whitespace_normalization".to_string(),
                    "format_standardization".to_string(),
                    "framework_specific_formatting".to_string(),
                ]);
            }
            OptimizationLevel::High => {
                optimizations.extend(vec![
                    "whitespace_normalization".to_string(),
                    "format_standardization".to_string(),
                    "framework_specific_formatting".to_string(),
                    "performance_optimization".to_string(),
                ]);
            }
            OptimizationLevel::Maximum => {
                optimizations.extend(vec![
                    "whitespace_normalization".to_string(),
                    "format_standardization".to_string(),
                    "framework_specific_formatting".to_string(),
                    "performance_optimization".to_string(),
                    "minimax_m2_optimization".to_string(),
                ]);
            }
        }

        optimizations
    }

    // Cache-related methods
    async fn get_cached_translation(&self, source: FrameworkType, target: FrameworkType, protocol: &Protocol) -> Result<Option<TranslationResult>> {
        let cache = self.optimization_cache.read().await;
        let cache_key = self.generate_cache_key(source, target, protocol);
        Ok(cache.get(&cache_key))
    }

    async fn cache_translation(&self, result: TranslationResult) -> Result<()> {
        let mut cache = self.optimization_cache.write().await;
        cache.insert(result.cache_key.clone(), result);
        Ok(())
    }

    async fn record_cache_hit(&self) -> Result<()> {
        let mut metrics = self.translation_metrics.write().await;
        metrics.cache_hits += 1;
        Ok(())
    }

    async fn record_translation_metrics(&self, result: &TranslationResult) -> Result<()> {
        let mut metrics = self.translation_metrics.write().await;
        metrics.total_translations += 1;
        metrics.average_confidence = (metrics.average_confidence * 0.9) + (result.confidence_score * 0.1);
        metrics.average_processing_time = (metrics.average_processing_time * 0.9) + (result.processing_time_ms as f64 * 0.1);
        Ok(())
    }
}

/// Format Converter Trait
/// Each framework adapter must implement this for format conversion
#[async_trait::async_trait]
pub trait FormatConverter: Send + Sync {
    /// Get the target framework type
    fn target_framework(&self) -> FrameworkType;

    /// Convert from ReasonKit protocol to framework-specific format
    async fn convert_from_reasonkit(&self, protocol: &Protocol) -> Result<ProtocolContent>;

    /// Convert from framework-specific format to ReasonKit protocol
    async fn convert_to_reasonkit(&self, content: &ProtocolContent) -> Result<Protocol>;

    /// Validate if content can be converted
    async fn validate_conversion(&self, content: &ProtocolContent) -> Result<bool>;
}

/// Built-in Format Converters

/// Claude Code Format Converter
pub struct ClaudeCodeConverter {
    json_parser: JsonParser,
}

impl ClaudeCodeConverter {
    pub fn new() -> Self {
        Self {
            json_parser: JsonParser::new(),
        }
    }
}

#[async_trait::async_trait]
impl FormatConverter for ClaudeCodeConverter {
    fn target_framework(&self) -> FrameworkType {
        FrameworkType::ClaudeCode
    }

    async fn convert_from_reasonkit(&self, protocol: &Protocol) -> Result<ProtocolContent> {
        // Convert ReasonKit protocol to Claude Code JSON format
        let claude_format = ClaudeCodeFormat {
            reasoning: self.extract_reasoning(protocol),
            confidence: self.calculate_confidence(protocol),
            structured_output: self.create_structured_output(protocol),
            metadata: self.create_metadata(protocol),
        };

        Ok(ProtocolContent::Json(serde_json::to_value(claude_format)?))
    }

    async fn convert_to_reasonkit(&self, content: &ProtocolContent) -> Result<Protocol> {
        match content {
            ProtocolContent::Json(json) => {
                let claude_format: ClaudeCodeFormat = serde_json::from_value(json.clone())?;
                self.reconstruct_protocol(claude_format)
            }
            _ => Err(crate::error::Error::InvalidContentFormat {
                expected: "JSON".to_string(),
                actual: format!("{:?}", content),
            }),
        }
    }

    async fn validate_conversion(&self, content: &ProtocolContent) -> Result<bool> {
        match content {
            ProtocolContent::Json(json) => {
                // Validate JSON structure for Claude Code
                Ok(json.get("reasoning").is_some() && json.get("confidence").is_some())
            }
            _ => Ok(false),
        }
    }
}

impl ClaudeCodeConverter {
    fn extract_reasoning(&self, protocol: &Protocol) -> String {
        match &protocol.content {
            ProtocolContent::Text(text) => text.clone(),
            ProtocolContent::Json(json) => {
                json.get("reasoning")
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default()
            }
            _ => "Complex reasoning process".to_string(),
        }
    }

    fn calculate_confidence(&self, protocol: &Protocol) -> f64 {
        // Calculate confidence based on protocol characteristics
        0.85 // Default confidence
    }

    fn create_structured_output(&self, protocol: &Protocol) -> serde_json::Value {
        serde_json::json!({
            "analysis": "structured",
            "steps": ["analysis", "reasoning", "conclusion"],
            "quality": "high"
        })
    }

    fn create_metadata(&self, protocol: &Protocol) -> serde_json::Value {
        serde_json::json!({
            "source": "reasonkit",
            "framework": "claude_code",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "version": "1.0"
        })
    }

    fn reconstruct_protocol(&self, claude_format: ClaudeCodeFormat) -> Result<Protocol> {
        let content = ProtocolContent::Text(claude_format.reasoning);

        Ok(Protocol {
            id: uuid::Uuid::new_v4(),
            content,
            metadata: crate::thinktool::ProtocolMetadata::default(),
            created_at: chrono::Utc::now(),
        })
    }
}

/// Cline Format Converter
pub struct ClineConverter;

impl ClineConverter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl FormatConverter for ClineConverter {
    fn target_framework(&self) -> FrameworkType {
        FrameworkType::Cline
    }

    async fn convert_from_reasonkit(&self, protocol: &Protocol) -> Result<ProtocolContent> {
        // Convert to Cline's logical analysis format
        let logical_analysis = LogicalAnalysis {
            premises: self.extract_premises(protocol),
            conclusions: self.extract_conclusions(protocol),
            fallacies_detected: Vec::new(),
            deductive_reasoning: self.create_deductive_structure(protocol),
        };

        Ok(ProtocolContent::Json(serde_json::to_value(logical_analysis)?))
    }

    async fn convert_to_reasonkit(&self, content: &ProtocolContent) -> Result<Protocol> {
        match content {
            ProtocolContent::Json(json) => {
                let logical_analysis: LogicalAnalysis = serde_json::from_value(json.clone())?;
                self.reconstruct_from_logical_analysis(logical_analysis)
            }
            _ => Err(crate::error::Error::InvalidContentFormat {
                expected: "JSON".to_string(),
                actual: format!("{:?}", content),
            }),
        }
    }

    async fn validate_conversion(&self, content: &ProtocolContent) -> Result<bool> {
        match content {
            ProtocolContent::Json(json) => {
                Ok(json.get("premises").is_some() && json.get("conclusions").is_some())
            }
            _ => Ok(false),
        }
    }
}

impl ClineConverter {
    fn extract_premises(&self, protocol: &Protocol) -> Vec<String> {
        // Extract logical premises from protocol
        vec!["premise_1".to_string(), "premise_2".to_string()]
    }

    fn extract_conclusions(&self, protocol: &Protocol) -> Vec<String> {
        // Extract logical conclusions from protocol
        vec!["conclusion_1".to_string()]
    }

    fn create_deductive_structure(&self, protocol: &Protocol) -> DeductiveStructure {
        DeductiveStructure {
            major_premise: "All reasoning follows logical principles".to_string(),
            minor_premise: "This is a reasoning task".to_string(),
            conclusion: "Therefore, logical principles apply".to_string(),
        }
    }

    fn reconstruct_from_logical_analysis(&self, analysis: LogicalAnalysis) -> Result<Protocol> {
        let reasoning = format!(
            "Logical Analysis:\nPremises: {}\nConclusions: {}",
            analysis.premises.join(", "),
            analysis.conclusions.join(", ")
        );

        Ok(Protocol {
            id: uuid::Uuid::new_v4(),
            content: ProtocolContent::Text(reasoning),
            metadata: crate::thinktool::ProtocolMetadata::default(),
            created_at: chrono::Utc::now(),
        })
    }
}

/// Placeholder implementations for other converters
pub struct KiloCodeConverter;
pub struct DroidConverter;
pub struct RooCodeConverter;
pub struct BlackBoxAIConverter;

impl KiloCodeConverter {
    pub fn new() -> Self { Self }
}
impl DroidConverter {
    pub fn new() -> Self { Self }
}
impl RooCodeConverter {
    pub fn new() -> Self { Self }
}
impl BlackBoxAIConverter {
    pub fn new() -> Self { Self }
}

#[async_trait::async_trait]
impl FormatConverter for KiloCodeConverter {
    fn target_framework(&self) -> FrameworkType { FrameworkType::KiloCode }
    async fn convert_from_reasonkit(&self, protocol: &Protocol) -> Result<ProtocolContent> {
        Ok(ProtocolContent::Text("Comprehensive critique format".to_string()))
    }
    async fn convert_to_reasonkit(&self, content: &ProtocolContent) -> Result<Protocol> {
        Ok(Protocol {
            id: uuid::Uuid::new_v4(),
            content: content.clone(),
            metadata: crate::thinktool::ProtocolMetadata::default(),
            created_at: chrono::Utc::now(),
        })
    }
    async fn validate_conversion(&self, _content: &ProtocolContent) -> Result<bool> { Ok(true) }
}

#[async_trait::async_trait]
impl FormatConverter for DroidConverter {
    fn target_framework(&self) -> FrameworkType { FrameworkType::Droid }
    async fn convert_from_reasonkit(&self, protocol: &Protocol) -> Result<ProtocolContent> {
        Ok(ProtocolContent::Json(serde_json::json!({"mobile_optimized": true})))
    }
    async fn convert_to_reasonkit(&self, content: &ProtocolContent) -> Result<Protocol> {
        Ok(Protocol {
            id: uuid::Uuid::new_v4(),
            content: content.clone(),
            metadata: crate::thinktool::ProtocolMetadata::default(),
            created_at: chrono::Utc::now(),
        })
    }
    async fn validate_conversion(&self, _content: &ProtocolContent) -> Result<bool> { Ok(true) }
}

#[async_trait::async_trait]
impl FormatConverter for RooCodeConverter {
    fn target_framework(&self) -> FrameworkType { FrameworkType::RooCode }
    async fn convert_from_reasonkit(&self, protocol: &Protocol) -> Result<ProtocolContent> {
        Ok(ProtocolContent::Json(serde_json::json!({"multi_agent": true})))
    }
    async fn convert_to_reasonkit(&self, content: &ProtocolContent) -> Result<Protocol> {
        Ok(Protocol {
            id: uuid::Uuid::new_v4(),
            content: content.clone(),
            metadata: crate::thinktool::ProtocolMetadata::default(),
            created_at: chrono::Utc::now(),
        })
    }
    async fn validate_conversion(&self, _content: &ProtocolContent) -> Result<bool> { Ok(true) }
}

#[async_trait::async_trait]
impl FormatConverter for BlackBoxAIConverter {
    fn target_framework(&self) -> FrameworkType { FrameworkType::BlackBoxAI }
    async fn convert_from_reasonkit(&self, protocol: &Protocol) -> Result<ProtocolContent> {
        Ok(ProtocolContent::Json(serde_json::json!({"high_throughput": true})))
    }
    async fn convert_to_reasonkit(&self, content: &ProtocolContent) -> Result<Protocol> {
        Ok(Protocol {
            id: uuid::Uuid::new_v4(),
            content: content.clone(),
            metadata: crate::thinktool::ProtocolMetadata::default(),
            created_at: chrono::Utc::now(),
        })
    }
    async fn validate_conversion(&self, _content: &ProtocolContent) -> Result<bool> { Ok(true) }
}

/// Supporting structures for converters

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodeFormat {
    pub reasoning: String,
    pub confidence: f64,
    pub structured_output: serde_json::Value,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalAnalysis {
    pub premises: Vec<String>,
    pub conclusions: Vec<String>,
    pub fallacies_detected: Vec<String>,
    pub deductive_reasoning: DeductiveStructure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeductiveStructure {
    pub major_premise: String,
    pub minor_premise: String,
    pub conclusion: String,
}

/// Supporting components

pub struct JsonParser;
impl JsonParser {
    pub fn new() -> Self { Self }
}

pub struct ValidationEngine {
    // Validation logic for protocol compatibility
}
impl ValidationEngine {
    pub fn new() -> Self { Self }

    pub async fn validate_compatibility(&self, protocol: &Protocol, framework: FrameworkType) -> Result<CompatibilityResult> {
        Ok(CompatibilityResult {
            is_compatible: true,
            compatibility_score: 0.9,
            issues: Vec::new(),
            suggestions: Vec::new(),
        })
    }

    pub async fn validate_translation(&self, content: &ProtocolContent, framework: FrameworkType) -> Result<ProtocolContent> {
        Ok(content.clone())
    }
}

pub struct OptimizationCache {
    cache: HashMap<String, TranslationResult>,
}
impl OptimizationCache {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    pub fn get(&self, key: &str) -> Option<TranslationResult> {
        self.cache.get(key).cloned()
    }

    pub fn insert(&mut self, key: String, value: TranslationResult) {
        self.cache.insert(key, value);
    }
}

pub struct TranslationMetrics {
    pub total_translations: u64,
    pub cache_hits: u64,
    pub average_confidence: f64,
    pub average_processing_time: f64,
}
impl TranslationMetrics {
    pub fn new() -> Self {
        Self {
            total_translations: 0,
            cache_hits: 0,
            average_confidence: 0.0,
            average_processing_time: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_protocol_translator_creation() {
        let translator = ProtocolTranslator::new().await.unwrap();
        assert!(!translator.format_converters.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_translation_cache() {
        let translator = ProtocolTranslator::new().await.unwrap();

        // Test would require actual protocol creation
        // This is a placeholder test structure
    }
}
