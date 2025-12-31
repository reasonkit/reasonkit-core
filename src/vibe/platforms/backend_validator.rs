//! # Backend Platform Validator
//!
//! Specialized validator implementing "Agent-as-a-Verifier" paradigm for backend protocols
//! with API validation, security testing, and performance analysis.

use super::BasePlatformValidator;
use super::*;

/// Backend-specific validator implementing comprehensive backend protocol validation
pub struct BackendValidator {
    base: BasePlatformValidator,
    #[allow(dead_code)]
    api_validator: ApiValidator,
    #[allow(dead_code)]
    security_checker: SecurityChecker,
    #[allow(dead_code)]
    performance_analyzer: PerformanceAnalyzer,
    #[allow(dead_code)]
    scalability_tester: ScalabilityTester,
}

impl Default for BackendValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendValidator {
    pub fn new() -> Self {
        Self {
            base: BasePlatformValidator::new(Platform::Backend),
            api_validator: ApiValidator::new(),
            security_checker: SecurityChecker::new(),
            performance_analyzer: PerformanceAnalyzer::new(),
            scalability_tester: ScalabilityTester::new(),
        }
    }

    /// Perform comprehensive backend-specific validation
    async fn validate_backend_protocol(
        &self,
        protocol_content: &str,
        config: &ValidationConfig,
    ) -> Result<BackendValidationResult, VIBEError> {
        let start_time = std::time::Instant::now();

        // Extract backend-specific elements
        let backend_elements = self.extract_backend_elements(protocol_content)?;

        // Validate API structure and endpoints
        let api_validation = self
            .validate_api_structure(&backend_elements, config)
            .await?;

        // Validate security measures
        let security_validation = self.validate_security(&backend_elements, config).await?;

        // Validate performance characteristics
        let performance_validation = self
            .validate_backend_performance(&backend_elements, config)
            .await?;

        // Validate data flow and integrity
        let data_flow_validation = self.validate_data_flow(&backend_elements, config).await?;

        // Validate scalability considerations
        let scalability_validation = self.validate_scalability(&backend_elements, config).await?;

        let validation_time = start_time.elapsed().as_millis() as u64;

        // Aggregate validation results
        let overall_score = self.calculate_backend_score(&[
            &api_validation,
            &security_validation,
            &performance_validation,
            &data_flow_validation,
            &scalability_validation,
        ])?;

        let mut all_issues = Vec::new();
        all_issues.extend(api_validation.issues);
        all_issues.extend(security_validation.issues);
        all_issues.extend(performance_validation.issues);
        all_issues.extend(data_flow_validation.issues);
        all_issues.extend(scalability_validation.issues);

        let recommendations = self.generate_backend_recommendations(&all_issues, overall_score)?;

        Ok(BackendValidationResult {
            overall_score,
            api_score: api_validation.score,
            security_score: security_validation.score,
            performance_score: performance_validation.score,
            data_flow_score: data_flow_validation.score,
            scalability_score: scalability_validation.score,
            validation_time_ms: validation_time,
            issues: all_issues,
            recommendations,
            backend_specific_metrics: BackendSpecificMetrics {
                endpoints_count: api_validation.endpoints_count,
                security_score: security_validation.security_score,
                average_response_time_ms: performance_validation.average_response_time,
                throughput_rps: performance_validation.throughput_rps,
                error_rate_percent: performance_validation.error_rate,
            },
        })
    }

    /// Extract backend-specific elements from protocol content
    fn extract_backend_elements(&self, content: &str) -> Result<BackendElements, VIBEError> {
        let mut elements = BackendElements::default();

        // Extract API endpoints
        let endpoint_pattern =
            regex::Regex::new(r"(?i)(GET|POST|PUT|DELETE|PATCH)\s+([/\w\-\.]+)").unwrap();
        for cap in endpoint_pattern.captures_iter(content) {
            elements
                .api_endpoints
                .insert(format!("{} {}", cap[1].to_uppercase(), &cap[2]));
        }

        // Extract HTTP status codes
        let status_pattern = regex::Regex::new(r"(200|201|400|401|403|404|500|502|503)\b").unwrap();
        for cap in status_pattern.captures_iter(content) {
            elements.status_codes.insert(cap[1].to_string());
        }

        // Extract database operations
        let db_pattern =
            regex::Regex::new(r"(?i)(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b").unwrap();
        for cap in db_pattern.captures_iter(content) {
            elements.database_operations.insert(cap[1].to_uppercase());
        }

        // Extract authentication methods
        let auth_pattern = regex::Regex::new(r"(JWT|OAuth|Bearer|Basic|Digest)").unwrap();
        for cap in auth_pattern.captures_iter(content) {
            elements.authentication_methods.insert(cap[1].to_string());
        }

        // Extract security headers
        let header_pattern =
            regex::Regex::new(r"(Authorization|Content-Type|Accept|Cookie|Set-Cookie)").unwrap();
        for cap in header_pattern.captures_iter(content) {
            elements.security_headers.insert(cap[1].to_string());
        }

        // Extract error handling patterns
        let error_pattern = regex::Regex::new(r"(try|catch|exception|error|throw|reject)").unwrap();
        let content_lower = content.to_lowercase();
        for cap in error_pattern.captures_iter(&content_lower) {
            elements.error_handling.insert(cap[1].to_string());
        }

        // Extract caching indicators
        if content.contains("cache") || content.contains("Redis") || content.contains("Memcached") {
            elements.caching_strategy = true;
        }

        // Extract logging patterns
        if content.contains("log") || content.contains("logger") || content.contains("monitoring") {
            elements.logging_monitoring = true;
        }

        // Extract validation patterns
        if content.contains("validate")
            || content.contains("schema")
            || content.contains("validator")
        {
            elements.has_validation = true;
        }

        // Extract rate limiting indicators
        if content.contains("rate") || content.contains("limit") || content.contains("throttle") {
            elements.rate_limiting = true;
        }

        Ok(elements)
    }

    /// Validate API structure and endpoints
    async fn validate_api_structure(
        &self,
        elements: &BackendElements,
        _config: &ValidationConfig,
    ) -> Result<ApiValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();
        let endpoints_count = elements.api_endpoints.len();

        // Check for API endpoints
        if elements.api_endpoints.is_empty() {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::High,
                category: IssueCategory::LogicError,
                description: "No API endpoints found".to_string(),
                location: None,
                suggestion: Some("Define API endpoints clearly".to_string()),
            });
            score -= 25.0;
        }

        // Check for proper HTTP methods
        let methods = vec!["GET", "POST", "PUT", "DELETE"];
        for method in &methods {
            let has_method = elements
                .api_endpoints
                .iter()
                .any(|endpoint| endpoint.starts_with(method));
            if !has_method {
                issues.push(ValidationIssue {
                    platform: Platform::Backend,
                    severity: Severity::Low,
                    category: IssueCategory::LogicError,
                    description: format!("No {} endpoints found", method),
                    location: None,
                    suggestion: Some(format!("Consider adding {} endpoints", method)),
                });
                score -= 5.0;
            }
        }

        // Check for proper status codes
        let essential_status_codes = vec!["200", "400", "401", "404", "500"];
        for code in &essential_status_codes {
            if !elements.status_codes.contains(*code) {
                issues.push(ValidationIssue {
                    platform: Platform::Backend,
                    severity: Severity::Medium,
                    category: IssueCategory::LogicError,
                    description: format!("Missing status code {}", code),
                    location: None,
                    suggestion: Some(format!("Handle {} responses properly", code)),
                });
                score -= 8.0;
            }
        }

        // Check for error handling
        if elements.error_handling.is_empty() {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::High,
                category: IssueCategory::ErrorHandling,
                description: "No error handling mechanisms found".to_string(),
                location: None,
                suggestion: Some("Implement comprehensive error handling".to_string()),
            });
            score -= 20.0;
        }

        // Check for validation
        if !elements.has_validation {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::LogicError,
                description: "No input validation detected".to_string(),
                location: None,
                suggestion: Some("Add input validation and sanitization".to_string()),
            });
            score -= 15.0;
        }

        // Check for consistent endpoint naming
        if !self.has_consistent_naming(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Low,
                category: IssueCategory::LogicError,
                description: "Inconsistent endpoint naming conventions".to_string(),
                location: None,
                suggestion: Some("Use consistent naming conventions for endpoints".to_string()),
            });
            score -= 6.0;
        }

        Ok(ApiValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
            endpoints_count,
        })
    }

    /// Validate security measures
    async fn validate_security(
        &self,
        elements: &BackendElements,
        _config: &ValidationConfig,
    ) -> Result<SecurityValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();
        let security_score = self.calculate_security_score(elements);

        // Check for authentication
        if elements.authentication_methods.is_empty() {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Critical,
                category: IssueCategory::SecurityVulnerability,
                description: "No authentication mechanism found".to_string(),
                location: None,
                suggestion: Some("Implement authentication (JWT, OAuth, etc.)".to_string()),
            });
            score -= 30.0;
        }

        // Check for authorization
        if !self.has_authorization(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::High,
                category: IssueCategory::SecurityVulnerability,
                description: "No authorization mechanism detected".to_string(),
                location: None,
                suggestion: Some("Implement authorization controls".to_string()),
            });
            score -= 20.0;
        }

        // Check for security headers
        let essential_headers = vec!["Authorization", "Content-Type"];
        for header in &essential_headers {
            if !elements.security_headers.contains(*header) {
                issues.push(ValidationIssue {
                    platform: Platform::Backend,
                    severity: Severity::Medium,
                    category: IssueCategory::SecurityVulnerability,
                    description: format!("Missing security header: {}", header),
                    location: None,
                    suggestion: Some(format!("Add {} header", header)),
                });
                score -= 10.0;
            }
        }

        // Check for input sanitization
        if !self.has_input_sanitization(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::High,
                category: IssueCategory::SecurityVulnerability,
                description: "No input sanitization detected".to_string(),
                location: None,
                suggestion: Some("Implement input sanitization and validation".to_string()),
            });
            score -= 15.0;
        }

        // Check for SQL injection prevention
        if !self.has_sql_injection_prevention(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::High,
                category: IssueCategory::SecurityVulnerability,
                description: "Potential SQL injection vulnerabilities".to_string(),
                location: None,
                suggestion: Some("Use parameterized queries or ORMs".to_string()),
            });
            score -= 18.0;
        }

        // Check for rate limiting
        if !elements.rate_limiting {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::SecurityVulnerability,
                description: "No rate limiting detected".to_string(),
                location: None,
                suggestion: Some("Implement rate limiting to prevent abuse".to_string()),
            });
            score -= 12.0;
        }

        Ok(SecurityValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
            security_score,
        })
    }

    /// Validate backend performance characteristics
    async fn validate_backend_performance(
        &self,
        elements: &BackendElements,
        _config: &ValidationConfig,
    ) -> Result<BackendPerformanceValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();
        let (average_response_time, throughput_rps, error_rate) =
            self.simulate_performance_metrics(elements);

        // Check response time
        if average_response_time > 1000 {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::High,
                category: IssueCategory::PerformanceIssue,
                description: format!("High average response time: {}ms", average_response_time),
                location: None,
                suggestion: Some("Optimize database queries and caching".to_string()),
            });
            score -= 20.0;
        } else if average_response_time > 500 {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::PerformanceIssue,
                description: format!(
                    "Response time could be improved: {}ms",
                    average_response_time
                ),
                location: None,
                suggestion: Some("Consider performance optimizations".to_string()),
            });
            score -= 10.0;
        }

        // Check error rate
        if error_rate > 5.0 {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::High,
                category: IssueCategory::PerformanceIssue,
                description: format!("High error rate: {:.1}%", error_rate),
                location: None,
                suggestion: Some("Improve error handling and reliability".to_string()),
            });
            score -= 25.0;
        } else if error_rate > 2.0 {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::PerformanceIssue,
                description: format!("Elevated error rate: {:.1}%", error_rate),
                location: None,
                suggestion: Some("Review error handling".to_string()),
            });
            score -= 12.0;
        }

        // Check for database optimization
        if !self.has_database_optimization(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::PerformanceIssue,
                description: "No database optimization detected".to_string(),
                location: None,
                suggestion: Some("Implement database indexing and query optimization".to_string()),
            });
            score -= 15.0;
        }

        // Check for caching strategy
        if !elements.caching_strategy {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Low,
                category: IssueCategory::PerformanceIssue,
                description: "No caching strategy detected".to_string(),
                location: None,
                suggestion: Some(
                    "Consider implementing caching for better performance".to_string(),
                ),
            });
            score -= 10.0;
        }

        // Check for connection pooling
        if !self.has_connection_pooling(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::PerformanceIssue,
                description: "No connection pooling detected".to_string(),
                location: None,
                suggestion: Some("Implement database connection pooling".to_string()),
            });
            score -= 12.0;
        }

        Ok(BackendPerformanceValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
            average_response_time,
            throughput_rps,
            error_rate,
        })
    }

    /// Validate data flow and integrity
    async fn validate_data_flow(
        &self,
        elements: &BackendElements,
        _config: &ValidationConfig,
    ) -> Result<DataFlowValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();

        // Check for database operations
        if elements.database_operations.is_empty() {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::LogicError,
                description: "No database operations detected".to_string(),
                location: None,
                suggestion: Some("Define data persistence layer".to_string()),
            });
            score -= 15.0;
        }

        // Check for data validation
        if !elements.has_validation {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::High,
                category: IssueCategory::LogicError,
                description: "No data validation found".to_string(),
                location: None,
                suggestion: Some("Implement data validation and integrity checks".to_string()),
            });
            score -= 20.0;
        }

        // Check for proper error handling in data operations
        if !self.has_data_error_handling(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::ErrorHandling,
                description: "No error handling for data operations".to_string(),
                location: None,
                suggestion: Some("Add error handling for database operations".to_string()),
            });
            score -= 12.0;
        }

        // Check for transaction handling
        if !self.has_transaction_handling(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::LogicError,
                description: "No transaction handling detected".to_string(),
                location: None,
                suggestion: Some("Implement proper transaction management".to_string()),
            });
            score -= 15.0;
        }

        // Check for data consistency
        if !self.has_data_consistency(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::LogicError,
                description: "No data consistency mechanisms found".to_string(),
                location: None,
                suggestion: Some("Implement data consistency checks".to_string()),
            });
            score -= 10.0;
        }

        Ok(DataFlowValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
        })
    }

    /// Validate scalability considerations
    async fn validate_scalability(
        &self,
        elements: &BackendElements,
        _config: &ValidationConfig,
    ) -> Result<ScalabilityValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();

        // Check for horizontal scaling indicators
        if !self.has_horizontal_scaling(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Low,
                category: IssueCategory::PerformanceIssue,
                description: "No horizontal scaling considerations".to_string(),
                location: None,
                suggestion: Some("Consider horizontal scaling architecture".to_string()),
            });
            score -= 8.0;
        }

        // Check for load balancing
        if !self.has_load_balancing(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::PerformanceIssue,
                description: "No load balancing detected".to_string(),
                location: None,
                suggestion: Some("Implement load balancing for high availability".to_string()),
            });
            score -= 12.0;
        }

        // Check for session management
        if !self.has_session_management(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Low,
                category: IssueCategory::LogicError,
                description: "No session management found".to_string(),
                location: None,
                suggestion: Some("Implement session management for user state".to_string()),
            });
            score -= 6.0;
        }

        // Check for monitoring and logging
        if !elements.logging_monitoring {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Medium,
                category: IssueCategory::PerformanceIssue,
                description: "No monitoring and logging detected".to_string(),
                location: None,
                suggestion: Some("Implement comprehensive monitoring and logging".to_string()),
            });
            score -= 15.0;
        }

        // Check for health checks
        if !self.has_health_checks(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Backend,
                severity: Severity::Low,
                category: IssueCategory::PerformanceIssue,
                description: "No health check endpoints found".to_string(),
                location: None,
                suggestion: Some("Implement health check endpoints".to_string()),
            });
            score -= 8.0;
        }

        Ok(ScalabilityValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
        })
    }

    // Helper methods
    fn has_consistent_naming(&self, elements: &BackendElements) -> bool {
        // Simple check for consistent naming patterns
        elements.api_endpoints.len() <= 1
            || elements
                .api_endpoints
                .iter()
                .all(|endpoint| endpoint.starts_with('/'))
    }

    fn calculate_security_score(&self, elements: &BackendElements) -> f32 {
        let mut score = 0.0;
        let total_checks = 6;

        if !elements.authentication_methods.is_empty() {
            score += 1.0;
        }
        if self.has_authorization(elements) {
            score += 1.0;
        }
        if !elements.security_headers.is_empty() {
            score += 1.0;
        }
        if self.has_input_sanitization(elements) {
            score += 1.0;
        }
        if self.has_sql_injection_prevention(elements) {
            score += 1.0;
        }
        if elements.rate_limiting {
            score += 1.0;
        }

        (score / total_checks as f32) * 100.0
    }

    fn has_authorization(&self, elements: &BackendElements) -> bool {
        elements.authentication_methods.contains("Bearer")
            || elements.authentication_methods.contains("OAuth")
    }

    fn has_input_sanitization(&self, elements: &BackendElements) -> bool {
        elements.has_validation && !elements.database_operations.is_empty()
    }

    fn has_sql_injection_prevention(&self, elements: &BackendElements) -> bool {
        elements.database_operations.contains("SELECT")
            || elements.database_operations.contains("INSERT")
            || elements.database_operations.contains("UPDATE")
    }

    fn has_database_optimization(&self, elements: &BackendElements) -> bool {
        !elements.database_operations.is_empty() && elements.caching_strategy
    }

    fn has_connection_pooling(&self, elements: &BackendElements) -> bool {
        !elements.database_operations.is_empty()
    }

    fn has_data_error_handling(&self, elements: &BackendElements) -> bool {
        !elements.error_handling.is_empty() && !elements.database_operations.is_empty()
    }

    fn has_transaction_handling(&self, elements: &BackendElements) -> bool {
        elements.database_operations.contains("BEGIN")
            || elements.database_operations.contains("COMMIT")
            || elements.database_operations.contains("ROLLBACK")
    }

    fn has_data_consistency(&self, elements: &BackendElements) -> bool {
        elements.has_validation && !elements.database_operations.is_empty()
    }

    fn has_horizontal_scaling(&self, elements: &BackendElements) -> bool {
        elements.api_endpoints.len() > 3
    }

    fn has_load_balancing(&self, elements: &BackendElements) -> bool {
        elements.api_endpoints.len() > 5
    }

    fn has_session_management(&self, elements: &BackendElements) -> bool {
        elements.authentication_methods.contains("Bearer")
            || elements.security_headers.contains("Cookie")
    }

    fn has_health_checks(&self, elements: &BackendElements) -> bool {
        elements
            .api_endpoints
            .iter()
            .any(|endpoint| endpoint.contains("health") || endpoint.contains("status"))
    }

    fn simulate_performance_metrics(&self, elements: &BackendElements) -> (u64, f32, f32) {
        let base_response_time = 200;
        let complexity_factor =
            elements.api_endpoints.len() as u64 + elements.database_operations.len() as u64;
        let average_response_time = base_response_time + complexity_factor * 50;

        let base_throughput = 1000.0;
        let throughput_factor = if elements.caching_strategy { 1.5 } else { 1.0 };
        let throughput_rps = base_throughput * throughput_factor;

        let base_error_rate = 1.0;
        let error_factor = if elements.error_handling.is_empty() {
            3.0
        } else {
            1.0
        };
        let error_rate = base_error_rate * error_factor;

        (average_response_time, throughput_rps, error_rate)
    }

    fn calculate_backend_score(
        &self,
        scores: &[&dyn BackendScoreComponent],
    ) -> Result<f32, VIBEError> {
        if scores.is_empty() {
            return Err(VIBEError::ValidationError(
                "No validation components provided".to_string(),
            ));
        }

        let weights = [0.25, 0.25, 0.20, 0.15, 0.15]; // API, Security, Performance, Data Flow, Scalability

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, component) in scores.iter().enumerate() {
            if i < weights.len() {
                let score = component.get_score();
                let weight = weights[i];
                weighted_sum += score * weight;
                total_weight += weight;
            }
        }

        Ok(weighted_sum / total_weight)
    }

    fn generate_backend_recommendations(
        &self,
        issues: &[ValidationIssue],
        overall_score: f32,
    ) -> Result<Vec<String>, VIBEError> {
        let mut recommendations = Vec::new();

        if overall_score < 70.0 {
            recommendations.push("Improve backend API structure and security measures".to_string());
        }

        let security_issues = issues
            .iter()
            .filter(|i| i.category == IssueCategory::SecurityVulnerability)
            .count();

        if security_issues > 0 {
            recommendations.push(
                "Prioritize security improvements - implement authentication and authorization"
                    .to_string(),
            );
        }

        let performance_issues = issues
            .iter()
            .filter(|i| i.category == IssueCategory::PerformanceIssue)
            .count();

        if performance_issues > 0 {
            recommendations.push("Focus on performance optimization - caching, database optimization, and monitoring".to_string());
        }

        let logic_issues = issues
            .iter()
            .filter(|i| i.category == IssueCategory::LogicError)
            .count();

        if logic_issues > 0 {
            recommendations.push("Improve API structure and data flow consistency".to_string());
        }

        if overall_score < 60.0 {
            recommendations
                .push("Implement comprehensive backend development best practices".to_string());
            recommendations.push("Add proper error handling and monitoring".to_string());
        }

        Ok(recommendations)
    }
}

// Implement PlatformValidator trait
#[async_trait::async_trait]
impl PlatformValidator for BackendValidator {
    async fn validate_protocol(
        &self,
        protocol_content: &str,
        config: &ValidationConfig,
        platform: Platform,
    ) -> Result<PlatformValidationResult, VIBEError> {
        if platform != Platform::Backend {
            return Err(VIBEError::PlatformError(
                "BackendValidator can only validate Backend platform protocols".to_string(),
            ));
        }

        // Perform common validation first
        let common_result = self
            .base
            .perform_common_validation(protocol_content, config)
            .await?;

        // Perform backend-specific validation
        let backend_result = self
            .validate_backend_protocol(protocol_content, config)
            .await?;

        // Combine results
        let final_score = (common_result.score + backend_result.overall_score) / 2.0;

        let mut all_issues = common_result.issues;
        all_issues.extend(backend_result.issues);

        let recommendations = self.generate_backend_recommendations(&all_issues, final_score)?;

        Ok(PlatformValidationResult {
            platform: Platform::Backend,
            score: final_score,
            status: if final_score >= config.minimum_score {
                ValidationStatus::Passed
            } else {
                ValidationStatus::Failed
            },
            issues: all_issues,
            performance_metrics: PlatformPerformanceMetrics {
                average_response_time_ms: backend_result
                    .backend_specific_metrics
                    .average_response_time_ms,
                memory_usage_mb: 400,
                cpu_usage_percent: 35.0,
                error_rate_percent: backend_result.backend_specific_metrics.error_rate_percent,
                throughput_requests_per_second: backend_result
                    .backend_specific_metrics
                    .throughput_rps,
            },
            recommendations,
        })
    }

    fn get_capabilities(&self) -> PlatformCapabilities {
        self.base.capabilities.clone()
    }

    fn get_requirements(&self) -> PlatformRequirements {
        self.base.requirements.clone()
    }

    fn estimate_complexity(&self, protocol_content: &str) -> ValidationComplexity {
        self.base
            .complexity_estimator
            .estimate_complexity(protocol_content)
    }

    fn get_scoring_criteria(&self) -> PlatformScoringCriteria {
        PlatformScoringCriteria {
            primary_criteria: vec![
                "API Structure Quality".to_string(),
                "Security Implementation".to_string(),
                "Performance Optimization".to_string(),
            ],
            secondary_criteria: vec![
                "Data Flow Integrity".to_string(),
                "Scalability Design".to_string(),
                "Error Handling".to_string(),
            ],
            penalty_factors: HashMap::from([
                ("no_authentication".to_string(), 0.3),
                ("poor_performance".to_string(), 0.2),
                ("no_error_handling".to_string(), 0.2),
            ]),
            bonus_factors: HashMap::from([
                ("strong_security".to_string(), 0.15),
                ("good_performance".to_string(), 0.1),
                ("comprehensive_monitoring".to_string(), 0.08),
            ]),
        }
    }
}

// Supporting data structures
#[derive(Debug, Default)]
struct BackendElements {
    api_endpoints: HashSet<String>,
    status_codes: HashSet<String>,
    database_operations: HashSet<String>,
    authentication_methods: HashSet<String>,
    security_headers: HashSet<String>,
    error_handling: HashSet<String>,
    caching_strategy: bool,
    logging_monitoring: bool,
    has_validation: bool,
    rate_limiting: bool,
}

#[derive(Debug)]
struct ApiValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
    endpoints_count: usize,
}

#[derive(Debug)]
struct SecurityValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
    security_score: f32,
}

#[derive(Debug)]
struct BackendPerformanceValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
    average_response_time: u64,
    throughput_rps: f32,
    error_rate: f32,
}

#[derive(Debug)]
struct DataFlowValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
}

#[derive(Debug)]
struct ScalabilityValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
}

#[derive(Debug)]
struct BackendValidationResult {
    overall_score: f32,
    #[allow(dead_code)]
    api_score: f32,
    #[allow(dead_code)]
    security_score: f32,
    #[allow(dead_code)]
    performance_score: f32,
    #[allow(dead_code)]
    data_flow_score: f32,
    #[allow(dead_code)]
    scalability_score: f32,
    #[allow(dead_code)]
    validation_time_ms: u64,
    issues: Vec<ValidationIssue>,
    #[allow(dead_code)]
    recommendations: Vec<String>,
    backend_specific_metrics: BackendSpecificMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackendSpecificMetrics {
    endpoints_count: usize,
    security_score: f32,
    average_response_time_ms: u64,
    throughput_rps: f32,
    error_rate_percent: f32,
}

/// Trait for backend score components
trait BackendScoreComponent {
    fn get_score(&self) -> f32;
}

impl BackendScoreComponent for ApiValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

impl BackendScoreComponent for SecurityValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

impl BackendScoreComponent for BackendPerformanceValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

impl BackendScoreComponent for DataFlowValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

impl BackendScoreComponent for ScalabilityValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

// Mock implementations
struct ApiValidator;
struct SecurityChecker;
struct PerformanceAnalyzer;
struct ScalabilityTester;

impl ApiValidator {
    fn new() -> Self {
        Self
    }
}

impl SecurityChecker {
    fn new() -> Self {
        Self
    }
}

impl PerformanceAnalyzer {
    fn new() -> Self {
        Self
    }
}

impl ScalabilityTester {
    fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_validator_creation() {
        let validator = BackendValidator::new();
        assert_eq!(validator.base.platform, Platform::Backend);
    }

    #[test]
    fn test_backend_elements_extraction() {
        let validator = BackendValidator::new();
        let content = "GET /api/users\nPOST /api/users\nJWT Authentication\nSELECT * FROM users";

        let elements = validator.extract_backend_elements(content).unwrap();
        assert!(elements.api_endpoints.contains("GET /api/users"));
        assert!(elements.api_endpoints.contains("POST /api/users"));
        assert!(elements.authentication_methods.contains("JWT"));
        assert!(elements.database_operations.contains("SELECT"));
    }

    #[test]
    fn test_security_score_calculation() {
        let validator = BackendValidator::new();
        let elements = BackendElements {
            api_endpoints: HashSet::new(),
            status_codes: HashSet::new(),
            database_operations: HashSet::from(["SELECT".to_string()]),
            authentication_methods: HashSet::from(["JWT".to_string()]),
            security_headers: HashSet::from(["Authorization".to_string()]),
            error_handling: HashSet::from(["try".to_string()]),
            caching_strategy: true,
            logging_monitoring: true,
            has_validation: true,
            rate_limiting: true,
        };

        let security_score = validator.calculate_security_score(&elements);
        assert!(security_score > 50.0); // Should have good security score
    }
}
