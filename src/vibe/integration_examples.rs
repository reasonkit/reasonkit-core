//! # VIBE Integration Examples
//!
//! Practical examples showing how to integrate VIBE protocol validation
//! with ReasonKit's ThinkTools and broader ecosystem.

use super::*;
use crate::thinktool::{Profile, ThinkToolExecutor};
use crate::verification::ProofLedger;
use crate::vibe::validation::{Priority, Severity, VIBEError};

/// Example integration between ThinkTool generation and VIBE validation
pub struct ThinkToolVIBEIntegration {
    vibe_engine: super::validation::VIBEEngine,
    thinktool_executor: ThinkToolExecutor,
    proof_ledger: ProofLedger,
}

impl ThinkToolVIBEIntegration {
    /// Create new integration instance
    pub fn new() -> Result<Self, VIBEError> {
        Ok(Self {
            vibe_engine: super::validation::VIBEEngine::new(),
            thinktool_executor: ThinkToolExecutor::new(),
            proof_ledger: ProofLedger::in_memory()?,
        })
    }

    /// Generate protocol using ThinkTools and validate with VIBE
    pub async fn generate_and_validate_protocol(
        &self,
        prompt: &str,
        profile: Profile,
        validation_config: ValidationConfig,
    ) -> Result<IntegratedValidationResult, VIBEError> {
        let start_time = std::time::Instant::now();

        // Step 1: Generate protocol using ThinkTools
        tracing::info!("Generating protocol using ThinkTools...");
        let protocol_content = self.thinktool_executor.run(prompt, profile).await?;

        // Step 2: Validate protocol with VIBE
        tracing::info!("Validating protocol with VIBE...");
        let vibe_result = self
            .vibe_engine
            .validate_protocol(&protocol_content, validation_config)
            .await?;

        // Step 3: Create immutable citation ledger for protocol claims
        let claims = self.extract_protocol_claims(&protocol_content)?;
        let anchored_claims = self.anchor_protocol_claims(claims).await?;

        // Step 4: Generate comprehensive report
        let report =
            self.generate_integrated_report(&protocol_content, &vibe_result, &anchored_claims)?;

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(IntegratedValidationResult {
            protocol_content,
            vibe_result,
            anchored_claims,
            report,
            execution_time_ms: execution_time,
            generation_timestamp: chrono::Utc::now(),
        })
    }

    /// Batch validation of multiple protocols
    pub async fn batch_validate_protocols(
        &self,
        protocols: Vec<(String, Profile)>,
        validation_config: ValidationConfig,
    ) -> Result<Vec<IntegratedValidationResult>, VIBEError> {
        let mut results = Vec::new();

        for (prompt, profile) in protocols {
            let result = self
                .generate_and_validate_protocol(&prompt, profile, validation_config.clone())
                .await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Continuous validation pipeline for protocol improvement
    pub async fn continuous_validation_pipeline(
        &self,
        initial_prompt: &str,
        max_iterations: usize,
        improvement_threshold: f32,
    ) -> Result<ContinuousValidationResult, VIBEError> {
        let mut current_prompt = initial_prompt.to_string();
        let mut iteration_results = Vec::new();
        let mut best_score = 0.0;
        let mut best_result = None;

        for iteration in 0..max_iterations {
            tracing::info!("Starting iteration {}/{}", iteration + 1, max_iterations);

            let result = self
                .generate_and_validate_protocol(
                    &current_prompt,
                    Profile::Balanced,
                    ValidationConfig::comprehensive(),
                )
                .await?;

            iteration_results.push(IterationResult {
                iteration_number: iteration + 1,
                result: result.clone(),
                improvements_made: Vec::new(),
            });

            // Check if this is the best result so far
            if result.vibe_result.overall_score > best_score {
                best_score = result.vibe_result.overall_score;
                best_result = Some(result.clone());

                // Generate improvement suggestions
                let improvements = self.generate_improvement_suggestions(&result)?;
                iteration_results[iteration].improvements_made = improvements.clone();

                // Update prompt for next iteration if improvements were made
                if !improvements.is_empty() {
                    current_prompt =
                        self.update_prompt_with_improvements(&current_prompt, &improvements)?;
                }
            }

            // Check if we've reached the improvement threshold
            if best_score >= improvement_threshold {
                tracing::info!("Reached improvement threshold: {:.1}", best_score);
                break;
            }
        }

        let total_iterations = iteration_results.len();

        Ok(ContinuousValidationResult {
            initial_prompt: initial_prompt.to_string(),
            final_prompt: current_prompt,
            best_score,
            best_result,
            total_iterations,
            iteration_results,
            improvement_achieved: best_score > 0.0,
        })
    }

    // Helper methods
    fn extract_protocol_claims(
        &self,
        protocol_content: &str,
    ) -> Result<Vec<ProtocolClaim>, VIBEError> {
        let mut claims = Vec::new();

        // Extract structured claims from protocol content
        let claim_pattern =
            regex::Regex::new(r"(must|should|requires|ensures|guarantees)\s+([^.!?]+)").unwrap();
        let content_lower = protocol_content.to_lowercase();
        for cap in claim_pattern.captures_iter(&content_lower) {
            claims.push(ProtocolClaim {
                claim_text: cap[2].trim().to_string(),
                claim_type: ClaimType::Requirement,
                source_line: 0, // Would be extracted from actual parsing
                verification_status: VerificationStatus::Pending,
            });
        }

        // Extract factual claims
        let fact_pattern = regex::Regex::new(r"(is|are|will|must be)\s+([^.!?]+)").unwrap();
        for cap in fact_pattern.captures_iter(&content_lower) {
            claims.push(ProtocolClaim {
                claim_text: cap[2].trim().to_string(),
                claim_type: ClaimType::Fact,
                source_line: 0,
                verification_status: VerificationStatus::Pending,
            });
        }

        Ok(claims)
    }

    async fn anchor_protocol_claims(
        &self,
        claims: Vec<ProtocolClaim>,
    ) -> Result<Vec<AnchoredClaim>, VIBEError> {
        let mut anchored_claims = Vec::new();

        for claim in claims {
            // Anchor each claim in the proof ledger
            let anchor_hash =
                self.proof_ledger
                    .anchor(&claim.claim_text, "reasonkit://protocol", None)?;

            anchored_claims.push(AnchoredClaim {
                claim,
                anchor_hash,
                anchored_at: chrono::Utc::now(),
                verification_history: Vec::new(),
            });
        }

        Ok(anchored_claims)
    }

    fn generate_integrated_report(
        &self,
        protocol_content: &str,
        vibe_result: &super::validation::ValidationResult,
        anchored_claims: &[AnchoredClaim],
    ) -> Result<IntegratedReport, VIBEError> {
        let report = IntegratedReport {
            protocol_summary: ProtocolSummary {
                content_hash: self.calculate_content_hash(protocol_content)?,
                word_count: protocol_content.split_whitespace().count(),
                complexity_score: self.calculate_complexity_score(protocol_content),
                platform_coverage: vibe_result.platform_scores.keys().cloned().collect(),
            },
            vibe_assessment: VIBEAssessment {
                overall_score: vibe_result.overall_score,
                platform_breakdown: vibe_result.platform_scores.clone(),
                confidence_interval: vibe_result.confidence_interval.clone(),
                validation_status: vibe_result.status,
                key_issues: vibe_result.issues.iter().take(5).cloned().collect(),
                recommendations: vibe_result
                    .recommendations
                    .iter()
                    .map(|r| r.description.clone())
                    .collect(),
            },
            claim_verification: ClaimVerification {
                total_claims: anchored_claims.len(),
                verified_claims: anchored_claims
                    .iter()
                    .filter(|c| c.claim.verification_status == VerificationStatus::Verified)
                    .count(),
                pending_verification: anchored_claims
                    .iter()
                    .filter(|c| c.claim.verification_status == VerificationStatus::Pending)
                    .count(),
                failed_verification: anchored_claims
                    .iter()
                    .filter(|c| c.claim.verification_status == VerificationStatus::Failed)
                    .count(),
                verification_rate: if anchored_claims.is_empty() {
                    0.0
                } else {
                    anchored_claims
                        .iter()
                        .filter(|c| c.claim.verification_status == VerificationStatus::Verified)
                        .count() as f32
                        / anchored_claims.len() as f32
                },
            },
            quality_metrics: QualityMetrics {
                logical_consistency_score: self.calculate_logical_consistency(protocol_content),
                practical_applicability_score: vibe_result.overall_score * 0.8, // Simplified
                platform_compatibility_score: self.calculate_platform_compatibility(vibe_result),
                code_quality_score: self.estimate_code_quality(protocol_content),
            },
            generated_at: chrono::Utc::now(),
        };

        Ok(report)
    }

    fn generate_improvement_suggestions(
        &self,
        result: &IntegratedValidationResult,
    ) -> Result<Vec<Improvement>, VIBEError> {
        let mut improvements = Vec::new();

        // Analyze VIBE issues for improvements
        for issue in &result.vibe_result.issues {
            match issue.severity {
                Severity::Critical => {
                    improvements.push(Improvement {
                        category: "Critical Fix".to_string(),
                        description: format!("Address critical issue: {}", issue.description),
                        priority: Priority::Critical,
                        estimated_impact: 15.0,
                        implementation_effort: "High".to_string(),
                    });
                }
                Severity::High => {
                    improvements.push(Improvement {
                        category: "High Priority".to_string(),
                        description: format!("Fix high priority issue: {}", issue.description),
                        priority: Priority::High,
                        estimated_impact: 10.0,
                        implementation_effort: "Medium".to_string(),
                    });
                }
                _ => {
                    improvements.push(Improvement {
                        category: "Enhancement".to_string(),
                        description: format!("Consider improving: {}", issue.description),
                        priority: Priority::Medium,
                        estimated_impact: 5.0,
                        implementation_effort: "Low".to_string(),
                    });
                }
            }
        }

        // Add platform-specific improvements
        for (platform, score) in &result.vibe_result.platform_scores {
            if *score < 70.0 {
                improvements.push(Improvement {
                    category: format!("{} Optimization", platform),
                    description: format!("Improve {} platform score from {:.1}", platform, score),
                    priority: Priority::High,
                    estimated_impact: 8.0,
                    implementation_effort: "Medium".to_string(),
                });
            }
        }

        Ok(improvements)
    }

    fn update_prompt_with_improvements(
        &self,
        original_prompt: &str,
        improvements: &[Improvement],
    ) -> Result<String, VIBEError> {
        let mut enhanced_prompt = original_prompt.to_string();

        // Add improvement context to prompt
        enhanced_prompt.push_str("\n\nAdditional considerations based on validation:");

        for improvement in improvements.iter().take(3) {
            // Limit to top 3 improvements
            enhanced_prompt.push_str(&format!("\n- {}", improvement.description));
        }

        enhanced_prompt
            .push_str("\nPlease incorporate these considerations into the protocol design.");

        Ok(enhanced_prompt)
    }

    fn calculate_content_hash(&self, content: &str) -> Result<String, VIBEError> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        Ok(format!("{:x}", hasher.finish()))
    }

    fn calculate_complexity_score(&self, content: &str) -> f32 {
        let word_count = content.split_whitespace().count() as f32;
        let sentence_count = content.split('.').count() as f32;
        let avg_words_per_sentence = if sentence_count > 0.0 {
            word_count / sentence_count
        } else {
            0.0
        };

        // Complexity score based on length and structure
        let length_factor = (word_count / 100.0).min(2.0); // Cap at 2.0
        let structure_factor = (avg_words_per_sentence / 20.0).min(1.5); // Cap at 1.5

        (length_factor + structure_factor) * 25.0 // Scale to 0-100
    }

    fn calculate_logical_consistency(&self, content: &str) -> f32 {
        // Simplified logical consistency calculation
        let sentences: Vec<&str> = content.split('.').collect();
        let mut contradictions = 0;

        // Simple contradiction detection
        for (i, sentence1) in sentences.iter().enumerate() {
            for sentence2 in sentences.iter().skip(i + 1) {
                if self.sentences_contradict(sentence1, sentence2) {
                    contradictions += 1;
                }
            }
        }

        let max_contradictions = (sentences.len() * (sentences.len() - 1)) / 2;
        let consistency_score = if max_contradictions > 0 {
            100.0 - ((contradictions as f32 / max_contradictions as f32) * 100.0)
        } else {
            100.0
        };

        consistency_score.clamp(0.0, 100.0)
    }

    fn sentences_contradict(&self, sentence1: &str, sentence2: &str) -> bool {
        // Simple contradiction detection
        let s1_lower = sentence1.to_lowercase();
        let s2_lower = sentence2.to_lowercase();
        let s1_words: std::collections::HashSet<&str> = s1_lower.split_whitespace().collect();
        let s2_words: std::collections::HashSet<&str> = s2_lower.split_whitespace().collect();

        // Check for contradictory terms
        let contradictions = vec![
            ("always", "never"),
            ("must", "cannot"),
            ("will", "will not"),
            ("is", "is not"),
        ];

        for (word1, word2) in &contradictions {
            if (s1_words.contains(word1) && s2_words.contains(word2))
                || (s1_words.contains(word2) && s2_words.contains(word1))
            {
                return true;
            }
        }

        false
    }

    fn calculate_platform_compatibility(
        &self,
        result: &super::validation::ValidationResult,
    ) -> f32 {
        if result.platform_scores.is_empty() {
            return 0.0;
        }

        let scores: Vec<f32> = result.platform_scores.values().cloned().collect();
        let min_score = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_score = scores.iter().fold(0.0f32, |a, &b| a.max(b));

        // Compatibility score based on score range (smaller range = better compatibility)
        let range = max_score - min_score;
        (100.0 - range).clamp(0.0, 100.0)
    }

    fn estimate_code_quality(&self, content: &str) -> f32 {
        // Simple code quality estimation
        let has_error_handling = content.contains("error") || content.contains("exception");
        let has_validation = content.contains("validate") || content.contains("check");
        let has_documentation = content.contains("#") || content.contains("//");

        let mut score: f32 = 50.0; // Base score

        if has_error_handling {
            score += 20.0;
        }
        if has_validation {
            score += 15.0;
        }
        if has_documentation {
            score += 15.0;
        }

        score.clamp(0.0, 100.0)
    }
}

/// Protocol claim for verification
#[derive(Debug, Clone)]
pub struct ProtocolClaim {
    pub claim_text: String,
    pub claim_type: ClaimType,
    pub source_line: usize,
    pub verification_status: VerificationStatus,
}

/// Types of protocol claims
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimType {
    Requirement,
    Fact,
    Constraint,
    Guarantee,
}

/// Verification status for claims
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    Pending,
    Verified,
    Failed,
    Unknown,
}

/// Anchored claim with immutable reference
#[derive(Debug, Clone)]
pub struct AnchoredClaim {
    pub claim: ProtocolClaim,
    pub anchor_hash: String,
    pub anchored_at: chrono::DateTime<chrono::Utc>,
    pub verification_history: Vec<VerificationRecord>,
}

/// Verification record
#[derive(Debug, Clone)]
pub struct VerificationRecord {
    pub verified_at: chrono::DateTime<chrono::Utc>,
    pub verification_method: String,
    pub result: VerificationResult,
    pub notes: Option<String>,
}

/// Result of verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub verified: bool,
    pub confidence: f32,
    pub evidence: Vec<String>,
}

/// Integrated validation result combining ThinkTool generation and VIBE validation
#[derive(Debug, Clone)]
pub struct IntegratedValidationResult {
    pub protocol_content: String,
    pub vibe_result: super::validation::ValidationResult,
    pub anchored_claims: Vec<AnchoredClaim>,
    pub report: IntegratedReport,
    pub execution_time_ms: u64,
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Comprehensive integrated report
#[derive(Debug, Clone)]
pub struct IntegratedReport {
    pub protocol_summary: ProtocolSummary,
    pub vibe_assessment: VIBEAssessment,
    pub claim_verification: ClaimVerification,
    pub quality_metrics: QualityMetrics,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Protocol summary information
#[derive(Debug, Clone)]
pub struct ProtocolSummary {
    pub content_hash: String,
    pub word_count: usize,
    pub complexity_score: f32,
    pub platform_coverage: Vec<Platform>,
}

/// VIBE validation assessment
#[derive(Debug, Clone)]
pub struct VIBEAssessment {
    pub overall_score: f32,
    pub platform_breakdown: HashMap<Platform, f32>,
    pub confidence_interval: Option<super::validation::ConfidenceInterval>,
    pub validation_status: super::validation::ValidationStatus,
    pub key_issues: Vec<super::validation::ValidationIssue>,
    pub recommendations: Vec<String>,
}

/// Claim verification summary
#[derive(Debug, Clone)]
pub struct ClaimVerification {
    pub total_claims: usize,
    pub verified_claims: usize,
    pub pending_verification: usize,
    pub failed_verification: usize,
    pub verification_rate: f32,
}

/// Quality metrics for the protocol
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub logical_consistency_score: f32,
    pub practical_applicability_score: f32,
    pub platform_compatibility_score: f32,
    pub code_quality_score: f32,
}

/// Continuous validation result
#[derive(Debug, Clone)]
pub struct ContinuousValidationResult {
    pub initial_prompt: String,
    pub final_prompt: String,
    pub best_score: f32,
    pub best_result: Option<IntegratedValidationResult>,
    pub iteration_results: Vec<IterationResult>,
    pub total_iterations: usize,
    pub improvement_achieved: bool,
}

/// Individual iteration result
#[derive(Debug, Clone)]
pub struct IterationResult {
    pub iteration_number: usize,
    pub result: IntegratedValidationResult,
    pub improvements_made: Vec<Improvement>,
}

/// Improvement suggestion
#[derive(Debug, Clone)]
pub struct Improvement {
    pub category: String,
    pub description: String,
    pub priority: Priority,
    pub estimated_impact: f32,
    pub implementation_effort: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_integrated_validation() {
        let integration = ThinkToolVIBEIntegration::new().unwrap();

        let prompt = "Design a user authentication protocol for a web application";
        let config = ValidationConfig::quick();

        let result = integration
            .generate_and_validate_protocol(prompt, Profile::Balanced, config)
            .await
            .unwrap();

        assert!(!result.protocol_content.is_empty());
        assert!(result.vibe_result.overall_score >= 0.0);
        assert!(result.vibe_result.overall_score <= 100.0);
        assert!(result.execution_time_ms > 0);
    }

    #[test]
    fn test_protocol_claim_extraction() {
        let integration = ThinkToolVIBEIntegration::new().unwrap();
        let protocol = "The system must validate user input. The response will be immediate. It should handle errors gracefully.";

        let claims = integration.extract_protocol_claims(protocol).unwrap();

        assert!(!claims.is_empty());
        assert!(claims
            .iter()
            .any(|c| c.claim_type == ClaimType::Requirement));
        assert!(claims.iter().any(|c| c.claim_type == ClaimType::Fact));
    }

    #[test]
    fn test_complexity_score_calculation() {
        let integration = ThinkToolVIBEIntegration::new().unwrap();

        let simple_protocol = "Protocol: Test. Purpose: Testing.";
        let complex_protocol = "Protocol: Complex Test. Purpose: Comprehensive testing with multiple steps and validation processes.";

        let simple_score = integration.calculate_complexity_score(simple_protocol);
        let complex_score = integration.calculate_complexity_score(complex_protocol);

        assert!(complex_score > simple_score);
        assert!(simple_score >= 0.0);
        assert!(complex_score <= 100.0);
    }

    #[test]
    fn test_logical_consistency_calculation() {
        let integration = ThinkToolVIBEIntegration::new().unwrap();

        let consistent_protocol = "The system must validate input. It will process the data.";
        let inconsistent_protocol =
            "The system must always validate input. It never validates input.";

        let consistent_score = integration.calculate_logical_consistency(consistent_protocol);
        let inconsistent_score = integration.calculate_logical_consistency(inconsistent_protocol);

        assert!(consistent_score > inconsistent_score);
        assert!(consistent_score <= 100.0);
    }
}
