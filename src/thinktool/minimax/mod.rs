//! MiniMax M2 Agent-Native ThinkTools Enhancement
//!
//! This module implements MiniMax M2's composite instruction constraints
//! and Interleaved Thinking System to enhance ReasonKit's ThinkTools.

use serde::{Deserialize, Serialize};

pub mod composite_constraints;
pub mod enhanced_bedrock;
pub mod enhanced_brutalhonesty;
pub mod enhanced_gigathink;
pub mod enhanced_laserlogic;
pub mod enhanced_proofguard;
pub mod interleaved_thinking;
pub mod performance_monitor;
pub mod profile_optimizer;

pub use composite_constraints::{
    CompositeInstruction, ConstraintEngine, ConstraintResult, ConstraintViolation, MemoryContext,
    SystemPrompt, ToolSchema, UserQuery,
};

pub use interleaved_thinking::{
    CrossValidation, InterleavedProtocol, InterleavedResult, InterleavedStep, MultiStepReasoning,
    ThinkingPattern,
};

pub use profile_optimizer::{ConfidenceTarget, OptimizationResult, ProfileOptimizer, ProfileType};

pub use performance_monitor::{MonitoringResult, PerformanceMetrics, PerformanceMonitor};

pub use enhanced_bedrock::{execute_enhanced_bedrock, EnhancedBedRock};
pub use enhanced_brutalhonesty::{execute_enhanced_brutalhonesty, EnhancedBrutalHonesty};
pub use enhanced_gigathink::{execute_enhanced_gigathink, EnhancedGigaThink};
pub use enhanced_laserlogic::{execute_enhanced_laserlogic, EnhancedLaserLogic};
pub use enhanced_proofguard::{execute_enhanced_proofguard, EnhancedProofGuard};

/// M2-level performance target
pub const M2_PERFORMANCE_TARGET: f64 = 0.92;

/// Cost reduction target compared to standard implementations
pub const COST_REDUCTION_TARGET: f64 = 0.08;

/// Enhanced ThinkTool result with M2 capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2ThinkToolResult {
    pub module: String,
    pub confidence: f64,
    pub output: serde_json::Value,
    pub performance_metrics: Option<PerformanceMetrics>,
    pub constraint_adherence: ConstraintResult,
    pub interleaved_steps: Vec<InterleavedStep>,
    pub profile_optimization: Option<OptimizationResult>,
    pub processing_time_ms: u64,
    pub token_count: u32,
    pub cost_efficiency: f64,
}

impl M2ThinkToolResult {
    /// Create a new M2-enhanced result
    pub fn new(module: String, base_output: serde_json::Value) -> Self {
        Self {
            module,
            confidence: 0.0, // Will be calculated based on M2 methodology
            output: base_output,
            performance_metrics: None,
            constraint_adherence: ConstraintResult::Pending,
            interleaved_steps: Vec::new(),
            profile_optimization: None,
            processing_time_ms: 0,
            token_count: 0,
            cost_efficiency: 0.0,
        }
    }

    /// Calculate M2-level confidence based on constraint adherence and cross-validation
    pub fn calculate_m2_confidence(&mut self) {
        let constraint_score = match &self.constraint_adherence {
            ConstraintResult::Passed(score) => *score,
            ConstraintResult::Failed(_) => 0.0,
            ConstraintResult::Pending => 0.5,
        };

        let cross_validation_score = if !self.interleaved_steps.is_empty() {
            self.interleaved_steps
                .iter()
                .filter(|step| step.cross_validation_passed)
                .count() as f64
                / self.interleaved_steps.len() as f64
        } else {
            0.7 // Default for non-interleaved
        };

        let performance_score = self.cost_efficiency.max(0.1);

        // M2 confidence formula: weighted combination of factors
        self.confidence =
            (constraint_score * 0.4 + cross_validation_score * 0.4 + performance_score * 0.2)
                .clamp(0.0, 1.0);
    }

    /// Calculate cost efficiency for M2 target
    pub fn calculate_cost_efficiency(&mut self, baseline_tokens: u32, baseline_cost: f64) {
        if baseline_cost > 0.0 {
            let current_cost = (self.token_count as f64) / baseline_tokens as f64 * baseline_cost;
            self.cost_efficiency = (baseline_cost / current_cost).min(2.0); // Cap at 2x improvement
        } else {
            self.cost_efficiency = 1.0;
        }
    }
}

/// Enhanced ThinkTool interface with M2 capabilities
pub trait M2ThinkTool {
    fn execute_with_m2(
        &self,
        input: &str,
        profile: ProfileType,
    ) -> crate::error::Result<M2ThinkToolResult>;

    fn get_composite_constraints(&self) -> Vec<CompositeInstruction>;

    fn get_interleaved_pattern(&self) -> InterleavedProtocol;

    fn get_performance_target(&self) -> PerformanceMetrics;
}

/// Profile-based ThinkTool execution
pub async fn execute_profile_based_thinktool<T: M2ThinkTool>(
    thinktool: &T,
    input: &str,
    profile: ProfileType,
) -> crate::error::Result<M2ThinkToolResult> {
    let mut result = thinktool.execute_with_m2(input, profile.clone())?;

    // Apply profile optimization
    let mut optimizer = ProfileOptimizer::new();
    if let Some(optimization) = optimizer.optimize_for_profile(&result, profile) {
        let confidence_multiplier = optimization.confidence_multiplier;
        result.profile_optimization = Some(optimization.clone());
        result.confidence *= confidence_multiplier;
    }

    // Apply M2 performance calculations
    result.calculate_m2_confidence();
    result.calculate_cost_efficiency(2000, 0.05); // Baseline: 2000 tokens, $0.05

    Ok(result)
}

/// MiniMax M2 ThinkTools Manager
pub struct M2ThinkToolsManager {
    pub gigathink: EnhancedGigaThink,
    pub laserlogic: EnhancedLaserLogic,
    pub bedrock: EnhancedBedRock,
    pub proofguard: EnhancedProofGuard,
    pub brutalhonesty: EnhancedBrutalHonesty,
    pub performance_monitor: PerformanceMonitor,
    pub profile_optimizer: ProfileOptimizer,
}

impl Default for M2ThinkToolsManager {
    fn default() -> Self {
        Self::new()
    }
}

impl M2ThinkToolsManager {
    pub fn new() -> Self {
        Self {
            gigathink: EnhancedGigaThink::new(),
            laserlogic: EnhancedLaserLogic::new(),
            bedrock: EnhancedBedRock::new(),
            proofguard: EnhancedProofGuard::new(),
            brutalhonesty: EnhancedBrutalHonesty::new(),
            performance_monitor: PerformanceMonitor::new(),
            profile_optimizer: ProfileOptimizer::new(),
        }
    }

    /// Execute any ThinkTool with M2 enhancements
    pub async fn execute_thinktool(
        &mut self,
        tool_name: &str,
        input: &str,
        profile: ProfileType,
    ) -> crate::error::Result<M2ThinkToolResult> {
        let profile_for_monitoring = profile.clone();

        let result = match tool_name {
            "enhanced_gigathink" | "gigathink" => {
                execute_profile_based_thinktool(&self.gigathink, input, profile.clone()).await
            }
            "enhanced_laserlogic" | "laserlogic" => {
                execute_profile_based_thinktool(&self.laserlogic, input, profile.clone()).await
            }
            "enhanced_bedrock" | "bedrock" => {
                execute_profile_based_thinktool(&self.bedrock, input, profile.clone()).await
            }
            "enhanced_proofguard" | "proofguard" => {
                execute_profile_based_thinktool(&self.proofguard, input, profile.clone()).await
            }
            "enhanced_brutalhonesty" | "brutalhonesty" => {
                execute_profile_based_thinktool(&self.brutalhonesty, input, profile.clone()).await
            }
            _ => {
                return Err(crate::error::Error::Validation(format!(
                    "Unknown ThinkTool: {}. Available tools: enhanced_gigathink, enhanced_laserlogic, enhanced_bedrock, enhanced_proofguard, enhanced_brutalhonesty",
                    tool_name
                )));
            }
        };

        // Monitor performance
        if let Ok(ref result) = result {
            let monitoring_result = self.performance_monitor.monitor_execution(
                result,
                &profile_for_monitoring,
                tool_name,
            );
            tracing::info!(
                "M2 ThinkTool performance: confidence={:.2}, time={}ms, efficiency={:.2}",
                monitoring_result.performance_metrics.achieved_confidence,
                monitoring_result
                    .performance_metrics
                    .achieved_processing_time_ms,
                monitoring_result
                    .performance_metrics
                    .achieved_cost_efficiency
            );
        }

        result
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> performance_monitor::PerformanceSummary {
        self.performance_monitor.get_performance_summary()
    }

    /// Get available ThinkTools
    pub fn list_available_tools(&self) -> Vec<String> {
        vec![
            "enhanced_gigathink".to_string(),
            "enhanced_laserlogic".to_string(),
            "enhanced_bedrock".to_string(),
            "enhanced_proofguard".to_string(),
            "enhanced_brutalhonesty".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m2_result_creation() {
        let result =
            M2ThinkToolResult::new("gigathink".to_string(), serde_json::json!({"test": true}));
        assert_eq!(result.module, "gigathink");
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_m2_confidence_calculation() {
        let mut result = M2ThinkToolResult::new("test".to_string(), serde_json::json!({}));
        result.constraint_adherence = ConstraintResult::Passed(0.9);
        result.interleaved_steps = vec![
            InterleavedStep {
                step_id: "1".to_string(),
                description: "step 1".to_string(),
                reasoning_chain: vec![],
                cross_validation_passed: true,
                confidence: 0.8,
                validation_results: vec![],
                dependencies: vec![],
                estimated_duration_ms: 0,
                actual_duration_ms: None,
            },
            InterleavedStep {
                step_id: "2".to_string(),
                description: "step 2".to_string(),
                reasoning_chain: vec![],
                cross_validation_passed: true,
                confidence: 0.9,
                validation_results: vec![],
                dependencies: vec![],
                estimated_duration_ms: 0,
                actual_duration_ms: None,
            },
        ];
        result.cost_efficiency = 1.2;

        result.calculate_m2_confidence();

        // Should calculate based on: 0.9 * 0.4 + 1.0 * 0.4 + 1.2 * 0.2 = 0.36 + 0.4 + 0.24 = 1.0
        assert!((result.confidence - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_m2_manager_creation() {
        let manager = M2ThinkToolsManager::new();
        let tools = manager.list_available_tools();
        assert_eq!(tools.len(), 5);
        assert!(tools.contains(&"enhanced_gigathink".to_string()));
    }
}
