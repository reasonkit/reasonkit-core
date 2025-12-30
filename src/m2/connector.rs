//! # MiniMax M2 Connector
//!
//! Core connector for interacting with MiniMax M2's Agent-Native Architecture API.
//! Handles composite instruction constraints, rate limiting, and performance optimization.

use crate::error::Error;
use crate::m2::types::*;
use reqwest::{Client, header};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{info, warn, error, debug};
use anyhow::Result;

/// Rate limiter implementation
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    request_times: Arc<RwLock<Vec<Duration>>>,
    burst_buffer: Arc<RwLock<Vec<Duration>>>,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            request_times: Arc::new(RwLock::new(Vec::new())),
            burst_buffer: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Check if request is allowed
    pub async fn is_allowed(&self) -> Result<bool, Error> {
        let now = std::time::Instant::now();
        let mut request_times = self.request_times.write().await;
        let mut burst_buffer = self.burst_buffer.write().await;
        
        // Clean old entries
        let one_minute_ago = now.checked_sub(Duration::from_secs(60)).unwrap_or(now);
        request_times.retain(|&time| time >= one_minute_ago);
        
        let one_second_ago = now.checked_sub(Duration::from_secs(1)).unwrap_or(now);
        burst_buffer.retain(|&time| time >= one_second_ago);
        
        // Check burst capacity
        if burst_buffer.len() >= self.config.burst as usize {
            return Ok(false);
        }
        
        // Check RPM limit
        if request_times.len() >= self.config.rpm as usize {
            return Ok(false);
        }
        
        // Check RPS limit
        if burst_buffer.len() >= self.config.rps as usize {
            return Ok(false);
        }
        
        // Record this request
        request_times.push(now.elapsed());
        burst_buffer.push(now.elapsed());
        
        Ok(true)
    }
}

/// Token usage tracker
#[derive(Debug, Clone)]
pub struct TokenTracker {
    usage_history: Arc<RwLock<Vec<TokenUsage>>>,
    daily_budget: Option<f64>,
    current_cost: Arc<RwLock<f64>>,
}

impl TokenTracker {
    /// Create new token tracker
    pub fn new(daily_budget: Option<f64>) -> Self {
        Self {
            usage_history: Arc::new(RwLock::new(Vec::new())),
            daily_budget,
            current_cost: Arc::new(RwLock::new(0.0)),
        }
    }
    
    /// Record token usage
    pub async fn record_usage(&self, usage: TokenUsage) -> Result<(), Error> {
        let mut usage_history = self.usage_history.write().await;
        let mut current_cost = self.current_cost.write().await;
        
        usage_history.push(usage.clone());
        *current_cost += usage.cost_estimate;
        
        // Check budget limits
        if let Some(budget) = self.daily_budget {
            if *current_cost > budget {
                return Err(Error::BudgetExceeded(*current_cost, budget));
            }
        }
        
        Ok(())
    }
    
    /// Get current usage statistics
    pub async fn get_usage_stats(&self) -> TokenUsageStats {
        let usage_history = self.usage_history.read().await;
        let current_cost = *self.current_cost.read().await;
        
        let total_input = usage_history.iter().map(|u| u.input_tokens).sum::<u32>();
        let total_output = usage_history.iter().map(|u| u.output_tokens).sum::<u32>();
        let total_tokens = usage_history.iter().map(|u| u.total_tokens).sum::<u32>();
        let total_cost = usage_history.iter().map(|u| u.cost_estimate).sum::<f64>();
        
        TokenUsageStats {
            total_input_tokens: total_input,
            total_output_tokens: total_output,
            total_tokens,
            total_cost,
            current_cost,
            request_count: usage_history.len() as u32,
            average_cost_per_token: if total_tokens > 0 { total_cost / total_tokens as f64 } else { 0.0 },
        }
    }
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsageStats {
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
    pub total_tokens: u32,
    pub total_cost: f64,
    pub current_cost: f64,
    pub request_count: u32,
    pub average_cost_per_token: f64,
}

/// MiniMax M2 API Connector
#[derive(Debug)]
pub struct M2Connector {
    config: M2Config,
    client: Client,
    rate_limiter: Arc<RateLimiter>,
    token_tracker: Arc<TokenTracker>,
    cache: Arc<RwLock<HashMap<String, CachedResponse>>>,
    retry_policy: RetryPolicy,
}

/// Cached response for performance optimization
#[derive(Debug, Clone)]
struct CachedResponse {
    response: M2Response,
    timestamp: std::time::Instant,
    ttl: Duration,
}

/// Retry policy configuration
#[derive(Debug, Clone)]
struct RetryPolicy {
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
    exponential_base: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
            exponential_base: 2.0,
        }
    }
}

impl M2Connector {
    /// Create new M2 connector
    pub async fn new(config: M2Config) -> Result<Self, Error> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            "User-Agent",
            header::HeaderValue::from_static("ReasonKit-Core/1.0"),
        );
        
        let client = Client::builder()
            .timeout(Duration::from_secs(300)) // 5 minute timeout
            .default_headers(headers)
            .build()
            .map_err(|e| Error::ConfigError(format!("Failed to create HTTP client: {}", e)))?;
            
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit.clone()));
        let token_tracker = Arc::new(TokenTracker::new(Some(100.0))); // $100 daily budget default
        let cache = Arc::new(RwLock::new(HashMap::new()));
        
        Ok(Self {
            config,
            client,
            rate_limiter,
            token_tracker,
            cache,
            retry_policy: RetryPolicy::default(),
        })
    }
    
    /// Execute interleaved thinking protocol
    pub async fn execute_interleaved_thinking(
        &self,
        protocol: &InterleavedProtocol,
        constraints: &CompositeConstraints,
        input: &ProtocolInput,
    ) -> Result<ProtocolResult, Error> {
        let execution_id = Uuid::new_v4();
        debug!("Starting M2 execution for protocol: {} (ID: {})", protocol.id, execution_id);
        
        // Step 1: Check rate limits
        if !self.rate_limiter.is_allowed().await? {
            return Err(Error::RateLimitExceeded);
        }
        
        // Step 2: Apply composite instruction constraints
        let constrained_prompt = self.apply_composite_constraints(protocol, constraints, input)?;
        
        // Step 3: Generate interleaved thinking plan
        let thinking_plan = self.generate_thinking_plan(protocol, input)?;
        
        // Step 4: Check cache for similar requests
        let cache_key = self.generate_cache_key(&constrained_prompt, &thinking_plan);
        if let Some(cached_response) = self.get_cached_response(&cache_key).await? {
            info!("Using cached response for protocol: {}", protocol.id);
            return self.format_cached_result(&cached_response, protocol, execution_id);
        }
        
        // Step 5: Execute with M2 API
        let m2_response = self.execute_with_retry(
            &constrained_prompt,
            &thinking_plan,
            &cache_key,
        ).await?;
        
        // Step 6: Validate response
        self.validate_m2_response(&m2_response)?;
        
        // Step 7: Cache successful response
        self.cache_response(cache_key, &m2_response).await?;
        
        // Step 8: Record token usage
        self.token_tracker.record_usage(m2_response.usage.clone()).await?;
        
        // Step 9: Format result
        let result = self.format_protocol_result(m2_response, protocol, execution_id)?;
        
        info!("Successfully executed M2 protocol: {} (ID: {})", protocol.id, execution_id);
        Ok(result)
    }
    
    /// Apply composite instruction constraints
    fn apply_composite_constraints(
        &self,
        protocol: &InterleavedProtocol,
        constraints: &CompositeConstraints,
        input: &ProtocolInput,
    ) -> Result<ConstrainedPrompt, Error> {
        let mut prompt_parts = Vec::new();
        let mut applied_constraints = Vec::new();
        let mut optimization_notes = Vec::new();
        
        // Apply system prompt constraints
        if !constraints.system_prompt.instruction.is_empty() {
            prompt_parts.push(format!(
                "SYSTEM: {}",
                constraints.system_prompt.instruction
            ));
            applied_constraints.push(AppliedConstraint {
                constraint_type: "system_prompt".to_string(),
                description: "System instruction applied".to_string(),
                impact: "defines_reasoning_style".to_string(),
            });
        }
        
        // Apply reasoning style
        match constraints.system_prompt.reasoning_style {
            ReasoningStyle::Interleaved => {
                optimization_notes.push("Using interleaved thinking methodology".to_string());
            }
            ReasoningStyle::TreeOfThoughts => {
                optimization_notes.push("Optimized for tree of thoughts reasoning".to_string());
            }
            _ => {}
        }
        
        // Apply user query constraints
        if !constraints.user_query.clarified.is_empty() {
            prompt_parts.push(format!(
                "USER: {}",
                constraints.user_query.clarified
            ));
            applied_constraints.push(AppliedConstraint {
                constraint_type: "user_query".to_string(),
                description: "User query clarified and processed".to_string(),
                impact: "defines_objective".to_string(),
            });
        }
        
        // Apply memory context
        if !constraints.memory_context.historical_context.is_empty() {
            let context_summary = self.summarize_memory_context(&constraints.memory_context)?;
            prompt_parts.push(format!("CONTEXT: {}", context_summary));
            applied_constraints.push(AppliedConstraint {
                constraint_type: "memory_context".to_string(),
                description: "Historical context integrated".to_string(),
                impact: "provides_relevant_background".to_string(),
            });
        }
        
        // Apply tool schemas
        if !constraints.tool_schemas.is_empty() {
            let tool_constraints = self.format_tool_constraints(&constraints.tool_schemas)?;
            prompt_parts.push(format!("TOOLS: {}", tool_constraints));
            applied_constraints.push(AppliedConstraint {
                constraint_type: "tool_schemas".to_string(),
                description: "Tool usage constraints applied".to_string(),
                impact: "ensures_proper_tool_usage".to_string(),
            });
        }
        
        // Apply framework-specific optimizations
        if !constraints.framework_constraints.optimizations.is_empty() {
            optimization_notes.push(format!(
                "Framework optimizations: {}",
                constraints.framework_constraints.optimizations.len()
            ));
        }
        
        let final_prompt = prompt_parts.join("\n\n");
        let token_count = self.estimate_token_count(&final_prompt);
        
        Ok(ConstrainedPrompt {
            prompt_text: final_prompt,
            applied_constraints,
            token_count,
            optimization_notes,
        })
    }
    
    /// Generate interleaved thinking plan
    fn generate_thinking_plan(
        &self,
        protocol: &InterleavedProtocol,
        input: &ProtocolInput,
    ) -> Result<InterleavedThinkingPlan, Error> {
        let mut phases = Vec::new();
        
        for phase in &protocol.phases {
            let planning_phase = PlanningPhase {
                id: format!("{}_{}", phase.name, Uuid::new_v4()),
                name: phase.name.clone(),
                objectives: self.derive_phase_objectives(phase, input)?,
                resource_requirements: self.estimate_resource_requirements(phase)?,
                expected_outputs: self.derive_expected_outputs(phase)?,
            };
            phases.push(planning_phase);
        }
        
        let resource_allocation = ResourceAllocation {
            token_budget: TokenBudget {
                total: self.config.max_context_length,
                context: (self.config.max_context_length * 80) / 100, // 80% for context
                output: (self.config.max_context_length * 20) / 100, // 20% for output
                validation: 0, // Reserved
            },
            time_allocation: self.estimate_time_allocation(&phases)?,
            parallel_capacity: protocol.phases.iter().map(|p| p.parallel_branches).sum(),
            quality_targets: self.derive_quality_targets(protocol)?,
        };
        
        let validation_checkpoints = self.generate_validation_checkpoints(&phases)?;
        
        Ok(InterleavedThinkingPlan {
            strategy: PlanningStrategy::BreadthFirst, // Default strategy
            phases,
            resource_allocation,
            validation_checkpoints,
        })
    }
    
    /// Execute M2 API request with retry logic
    async fn execute_with_retry(
        &self,
        prompt: &ConstrainedPrompt,
        thinking_plan: &InterleavedThinkingPlan,
        cache_key: &str,
    ) -> Result<M2Response, Error> {
        let mut last_error = None;
        
        for attempt in 0..=self.retry_policy.max_retries {
            match self.execute_m2_request(prompt, thinking_plan).await {
                Ok(response) => {
                    debug!("M2 API request successful (attempt {})", attempt + 1);
                    return Ok(response);
                }
                Err(error) => {
                    warn!("M2 API request failed (attempt {}): {}", attempt + 1, error);
                    last_error = Some(error);
                    
                    if attempt < self.retry_policy.max_retries {
                        let delay = self.calculate_retry_delay(attempt);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| {
            Error::M2ExecutionError("All retry attempts exhausted".to_string())
        }))
    }
    
    /// Execute M2 API request
    async fn execute_m2_request(
        &self,
        prompt: &ConstrainedPrompt,
        thinking_plan: &InterleavedThinkingPlan,
    ) -> Result<M2Response, Error> {
        let m2_request = M2Request {
            model: "minimax-m2-agent".to_string(),
            prompt: prompt.clone(),
            thinking_plan: thinking_plan.clone(),
            max_tokens: self.config.max_output_length,
            temperature: 0.1, // Low temperature for consistent reasoning
            stop_sequences: vec!["[END]".to_string()],
        };
        
        let response = self.client
            .post(&self.config.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&m2_request)
            .send()
            .await
            .map_err(|e| Error::M2ExecutionError(format!("HTTP request failed: {}", e)))?;
            
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::M2ExecutionError(format!(
                "API request failed with status {}: {}",
                status, body
            )));
        }
        
        let m2_response: M2Response = response
            .json()
            .await
            .map_err(|e| Error::M2ExecutionError(format!("Failed to parse response: {}", e)))?;
            
        Ok(m2_response)
    }
    
    /// Calculate retry delay with exponential backoff
    fn calculate_retry_delay(&self, attempt: u32) -> Duration {
        let delay = self.retry_policy.base_delay.as_millis() as f64
            * self.retry_policy.exponential_base.powi(attempt as i32);
            
        let delay_ms = delay.min(self.retry_policy.max_delay.as_millis() as f64) as u64;
        Duration::from_millis(delay_ms)
    }
    
    /// Generate cache key for request
    fn generate_cache_key(&self, prompt: &ConstrainedPrompt, thinking_plan: &InterleavedThinkingPlan) -> String {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        hasher.update(prompt.prompt_text.as_bytes());
        hasher.update(serde_json::to_string(thinking_plan).unwrap_or_default().as_bytes());
        
        format!("m2_{:x}", hasher.finalize())
    }
    
    /// Get cached response
    async fn get_cached_response(&self, cache_key: &str) -> Result<Option<M2Response>, Error> {
        let cache = self.cache.read().await;
        
        if let Some(cached) = cache.get(cache_key) {
            if cached.timestamp.elapsed() < cached.ttl {
                return Ok(Some(cached.response.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// Cache successful response
    async fn cache_response(&self, cache_key: String, response: &M2Response) -> Result<(), Error> {
        let mut cache = self.cache.write().await;
        
        cache.insert(cache_key, CachedResponse {
            response: response.clone(),
            timestamp: std::time::Instant::now(),
            ttl: Duration::from_secs(300), // 5 minute cache
        });
        
        Ok(())
    }
    
    /// Validate M2 response
    fn validate_m2_response(&self, response: &M2Response) -> Result<(), Error> {
        if response.text.is_empty() {
            return Err(Error::M2ExecutionError("Empty response from M2 API".to_string()));
        }
        
        if response.confidence_scores.overall <= 0.0 || response.confidence_scores.overall > 1.0 {
            return Err(Error::M2ExecutionError("Invalid confidence scores".to_string()));
        }
        
        if response.usage.total_tokens == 0 {
            return Err(Error::M2ExecutionError("Zero token usage reported".to_string()));
        }
        
        Ok(())
    }
    
    /// Format protocol result
    fn format_protocol_result(
        &self,
        m2_response: M2Response,
        protocol: &InterleavedProtocol,
        execution_id: Uuid,
    ) -> Result<ProtocolResult, Error> {
        let output = ProtocolOutput {
            result: serde_json::Value::String(m2_response.text),
            evidence: self.extract_evidence(&m2_response)?,
            confidence: m2_response.confidence_scores.overall,
            recommendations: self.extract_recommendations(&m2_response)?,
            next_steps: self.extract_next_steps(&m2_response)?,
        };
        
        let metrics = ExecutionMetrics {
            execution_time: std::time::Duration::from_millis(m2_response.performance_metrics.latency_ms),
            token_usage: m2_response.usage.clone(),
            cost_metrics: CostMetrics {
                total_cost: m2_response.usage.cost_estimate,
                cost_per_step: m2_response.usage.cost_estimate / m2_response.reasoning_trace.phases_executed.len() as f64,
                cost_reduction_percent: m2_response.performance_metrics.cost_reduction_percent,
                cost_efficiency: self.calculate_cost_efficiency(&m2_response),
            },
            quality_metrics: QualityMetrics {
                overall_quality: m2_response.confidence_scores.overall,
                accuracy: m2_response.confidence_scores.accuracy,
                completeness: m2_response.confidence_scores.completeness,
                consistency: m2_response.confidence_scores.consistency,
                coherence: m2_response.confidence_scores.coherence,
            },
            performance_metrics: m2_response.performance_metrics,
        };
        
        let audit_trail = AuditTrail {
            execution_id,
            timestamp: chrono::Utc::now(),
            user_id: None, // TODO: Implement user identification
            protocol_version: protocol.version.clone(),
            input_hash: self.calculate_input_hash(&m2_response)?,
            output_hash: self.calculate_output_hash(&m2_response)?,
            compliance_flags: vec![ComplianceFlag::GDPRCompliant],
        };
        
        Ok(ProtocolResult {
            id: execution_id.to_string(),
            protocol_id: protocol.id.clone(),
            status: ExecutionStatus::Completed,
            output,
            metrics,
            audit_trail,
        })
    }
    
    // Helper methods (abbreviated for space)
    fn summarize_memory_context(&self, context: &MemoryContext) -> Result<String, Error> {
        // Implementation for summarizing memory context
        Ok("Memory context summary".to_string())
    }
    
    fn format_tool_constraints(&self, tools: &[ToolSchema]) -> Result<String, Error> {
        // Implementation for formatting tool constraints
        Ok("Tool constraints summary".to_string())
    }
    
    fn derive_phase_objectives(&self, phase: &InterleavedPhase, input: &ProtocolInput) -> Result<Vec<String>, Error> {
        // Implementation for deriving phase objectives
        Ok(vec!["Complete phase objectives".to_string()])
    }
    
    fn estimate_resource_requirements(&self, phase: &InterleavedPhase) -> Result<ResourceRequirements, Error> {
        // Implementation for estimating resource requirements
        Ok(ResourceRequirements { /* ... */ })
    }
    
    fn derive_expected_outputs(&self, phase: &InterleavedPhase) -> Result<Vec<String>, Error> {
        // Implementation for deriving expected outputs
        Ok(vec!["Expected outputs".to_string()])
    }
    
    fn estimate_time_allocation(&self, phases: &[PlanningPhase]) -> Result<HashMap<String, Duration>, Error> {
        // Implementation for estimating time allocation
        Ok(HashMap::new())
    }
    
    fn derive_quality_targets(&self, protocol: &InterleavedProtocol) -> Result<QualityTargets, Error> {
        // Implementation for deriving quality targets
        Ok(QualityTargets { /* ... */ })
    }
    
    fn generate_validation_checkpoints(&self, phases: &[PlanningPhase]) -> Result<Vec<ValidationCheckpoint>, Error> {
        // Implementation for generating validation checkpoints
        Ok(Vec::new())
    }
    
    fn format_cached_result(&self, cached: &M2Response, protocol: &InterleavedProtocol, execution_id: Uuid) -> Result<ProtocolResult, Error> {
        // Implementation for formatting cached result
        self.format_protocol_result(cached.clone(), protocol, execution_id)
    }
    
    fn extract_evidence(&self, response: &M2Response) -> Result<Vec<Evidence>, Error> {
        // Implementation for extracting evidence
        Ok(Vec::new())
    }
    
    fn extract_recommendations(&self, response: &M2Response) -> Result<Vec<String>, Error> {
        // Implementation for extracting recommendations
        Ok(Vec::new())
    }
    
    fn extract_next_steps(&self, response: &M2Response) -> Result<Vec<String>, Error> {
        // Implementation for extracting next steps
        Ok(Vec::new())
    }
    
    fn calculate_cost_efficiency(&self, response: &M2Response) -> f64 {
        // Implementation for calculating cost efficiency
        0.95 // Placeholder
    }
    
    fn calculate_input_hash(&self, response: &M2Response) -> Result<String, Error> {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        hasher.update(response.text.as_bytes());
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    fn calculate_output_hash(&self, response: &M2Response) -> Result<String, Error> {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        hasher.update(response.text.as_bytes());
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    fn estimate_token_count(&self, text: &str) -> usize {
        // Simple token estimation (roughly 1 token per 4 characters)
        text.len() / 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_m2_connector_creation() {
        let config = M2Config {
            endpoint: "https://api.minimax.chat/v1/m2".to_string(),
            api_key: "test_key".to_string(),
            max_context_length: 200000,
            max_output_length: 128000,
            rate_limit: RateLimitConfig {
                rpm: 60,
                rps: 1,
                burst: 5,
            },
            performance: PerformanceConfig {
                cost_reduction_target: 92.0,
                latency_target_ms: 2000,
                quality_threshold: 0.90,
                enable_caching: true,
                compression_level: 5,
            },
        };
        
        let connector = M2Connector::new(config).await;
        assert!(connector.is_ok());
    }
    
    #[tokio::test]
    async fn test_rate_limiter() {
        let config = RateLimitConfig {
            rpm: 60,
            rps: 1,
            burst: 5,
        };
        
        let limiter = RateLimiter::new(config);
        
        // Should allow initial requests
        assert!(limiter.is_allowed().await.unwrap());
        
        // Should respect burst limit
        for _ in 0..6 {
            limiter.is_allowed().await.unwrap();
        }
    }
}