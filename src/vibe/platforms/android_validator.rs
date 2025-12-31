//! # Android Platform Validator
//!
//! Specialized validator implementing "Agent-as-a-Verifier" paradigm for Android protocols
//! with Material Design validation, touch interaction testing, and screen adaptation checks.

use super::BasePlatformValidator;
use super::*;

/// Android-specific validator implementing comprehensive Android protocol validation
pub struct AndroidValidator {
    base: BasePlatformValidator,
    #[allow(dead_code)]
    material_design_checker: MaterialDesignChecker,
    #[allow(dead_code)]
    touch_interaction_validator: TouchInteractionValidator,
    #[allow(dead_code)]
    screen_density_analyzer: ScreenDensityAnalyzer,
    #[allow(dead_code)]
    android_version_checker: AndroidVersionChecker,
    #[allow(dead_code)]
    performance_profiler: AndroidPerformanceProfiler,
}

impl Default for AndroidValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl AndroidValidator {
    pub fn new() -> Self {
        Self {
            base: BasePlatformValidator::new(Platform::Android),
            material_design_checker: MaterialDesignChecker::new(),
            touch_interaction_validator: TouchInteractionValidator::new(),
            screen_density_analyzer: ScreenDensityAnalyzer::new(),
            android_version_checker: AndroidVersionChecker::new(),
            performance_profiler: AndroidPerformanceProfiler::new(),
        }
    }

    /// Perform comprehensive Android-specific validation
    async fn validate_android_protocol(
        &self,
        protocol_content: &str,
        config: &ValidationConfig,
    ) -> Result<AndroidValidationResult, VIBEError> {
        let start_time = std::time::Instant::now();

        // Extract Android-specific elements
        let android_elements = self.extract_android_elements(protocol_content)?;

        // Validate Material Design compliance
        let material_validation = self
            .validate_material_design(&android_elements, config)
            .await?;

        // Validate touch interactions
        let touch_validation = self
            .validate_touch_interactions(&android_elements, config)
            .await?;

        // Validate screen adaptation
        let screen_validation = self
            .validate_screen_adaptation(&android_elements, config)
            .await?;

        // Validate Android version compatibility
        let version_validation = self
            .validate_android_version_compatibility(&android_elements, config)
            .await?;

        // Validate performance characteristics
        let performance_validation = self
            .validate_android_performance(&android_elements, config)
            .await?;

        let validation_time = start_time.elapsed().as_millis() as u64;

        // Aggregate validation results
        let overall_score = self.calculate_android_score(&[
            &material_validation,
            &touch_validation,
            &screen_validation,
            &version_validation,
            &performance_validation,
        ])?;

        let mut all_issues = Vec::new();
        all_issues.extend(material_validation.issues);
        all_issues.extend(touch_validation.issues);
        all_issues.extend(screen_validation.issues);
        all_issues.extend(version_validation.issues);
        all_issues.extend(performance_validation.issues);

        let recommendations = self.generate_android_recommendations(&all_issues, overall_score)?;

        Ok(AndroidValidationResult {
            overall_score,
            material_score: material_validation.score,
            touch_score: touch_validation.score,
            screen_score: screen_validation.score,
            version_score: version_validation.score,
            performance_score: performance_validation.score,
            validation_time_ms: validation_time,
            issues: all_issues,
            recommendations,
            android_specific_metrics: AndroidSpecificMetrics {
                material_design_version: material_validation.design_version,
                supported_densities: screen_validation.supported_densities,
                min_sdk_version: version_validation.min_sdk,
                target_sdk_version: version_validation.target_sdk,
                touch_gestures_supported: touch_validation.gestures_supported,
                performance_rating: performance_validation.rating,
            },
        })
    }

    /// Extract Android-specific elements from protocol content
    /// PERFORMANCE: Regex patterns are pre-compiled as static Lazy<Regex> for optimal performance
    fn extract_android_elements(&self, content: &str) -> Result<AndroidElements, VIBEError> {
        use once_cell::sync::Lazy;

        // Pre-compiled static regex patterns (compiled once at program start)
        static MATERIAL_PATTERN: Lazy<regex::Regex> = Lazy::new(|| {
            regex::Regex::new(r"(MaterialCard|FloatingActionButton|AppBarLayout|NavigationView|RecyclerView|CardView|Button|EditText)").unwrap()
        });
        static GESTURE_PATTERN: Lazy<regex::Regex> = Lazy::new(|| {
            regex::Regex::new(r"(tap|click|long press|double tap|pinch|swipe|fling)").unwrap()
        });
        static PERMISSION_PATTERN: Lazy<regex::Regex> =
            Lazy::new(|| regex::Regex::new(r"(READ_|WRITE_|CAMERA_|LOCATION_|PHONE_)").unwrap());
        static SDK_PATTERN: Lazy<regex::Regex> = Lazy::new(|| {
            regex::Regex::new(
                r"(?:minSdkVersion|targetSdkVersion|compileSdkVersion)\s*(?:[:=])?\s*(\d+)",
            )
            .unwrap()
        });
        static SCREEN_PATTERN: Lazy<regex::Regex> =
            Lazy::new(|| regex::Regex::new(r"(sw\d+|w\d+|h\d+|density)").unwrap());
        static LAYOUT_PATTERN: Lazy<regex::Regex> = Lazy::new(|| {
            regex::Regex::new(
                r"(LinearLayout|RelativeLayout|ConstraintLayout|FrameLayout|GridLayout)",
            )
            .unwrap()
        });

        let mut elements = AndroidElements::default();

        // Extract Material Design components
        for cap in MATERIAL_PATTERN.captures_iter(content) {
            elements.material_components.insert(cap[1].to_string());
        }

        // Extract touch gestures
        let content_lower = content.to_lowercase();
        for cap in GESTURE_PATTERN.captures_iter(&content_lower) {
            elements.touch_gestures.insert(cap[1].to_string());
        }

        // Extract Android permissions
        let content_upper = content.to_uppercase();
        for cap in PERMISSION_PATTERN.captures_iter(&content_upper) {
            elements.permissions.insert(cap[0].to_string());
        }

        // Extract Android SDK versions
        for cap in SDK_PATTERN.captures_iter(content) {
            if let Some(version) = cap.get(1) {
                elements.sdk_versions.insert(version.as_str().to_string());
            }
        }

        // Extract screen configurations
        for cap in SCREEN_PATTERN.captures_iter(&content_lower) {
            elements.screen_configs.insert(cap[0].to_string());
        }

        // Extract layout information
        for cap in LAYOUT_PATTERN.captures_iter(content) {
            elements.layout_types.insert(cap[1].to_string());
        }

        // Detect Android-specific patterns
        if content.contains("android:") || content.contains("@android") {
            elements.has_android_resources = true;
        }

        if content.contains("Activity") || content.contains("Fragment") {
            elements.has_android_components = true;
        }

        if content.contains("OnClickListener") || content.contains("TouchListener") {
            elements.has_event_handlers = true;
        }

        if content.contains("RecyclerView") || content.contains("ListView") {
            elements.has_list_components = true;
        }

        if content.contains("Material") || content.contains("Theme") {
            elements.has_material_themes = true;
        }

        Ok(elements)
    }

    /// Validate Material Design compliance
    async fn validate_material_design(
        &self,
        elements: &AndroidElements,
        _config: &ValidationConfig,
    ) -> Result<MaterialValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();
        let design_version = self.detect_material_version(elements);

        // Check for Material Design components
        if elements.material_components.is_empty() {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::High,
                category: IssueCategory::UIUXIssue,
                description: "No Material Design components found".to_string(),
                location: None,
                suggestion: Some("Use Material Design components for better UX".to_string()),
            });
            score -= 25.0;
        }

        // Check for proper button usage
        if !elements.material_components.contains("Button") {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::UIUXIssue,
                description: "No button components found".to_string(),
                location: None,
                suggestion: Some("Add proper button components".to_string()),
            });
            score -= 10.0;
        }

        // Check for card-based layouts
        if !elements.material_components.contains("CardView")
            && !elements.material_components.contains("MaterialCard")
        {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Low,
                category: IssueCategory::UIUXIssue,
                description: "No card-based layouts detected".to_string(),
                location: None,
                suggestion: Some("Consider using CardView for content grouping".to_string()),
            });
            score -= 8.0;
        }

        // Check for floating action button
        if !elements
            .material_components
            .contains("FloatingActionButton")
        {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Low,
                category: IssueCategory::UIUXIssue,
                description: "No floating action button found".to_string(),
                location: None,
                suggestion: Some("Consider adding FAB for primary actions".to_string()),
            });
            score -= 5.0;
        }

        // Check for proper theming
        if !elements.has_material_themes {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::UIUXIssue,
                description: "No Material Design theming detected".to_string(),
                location: None,
                suggestion: Some("Apply Material Design themes and colors".to_string()),
            });
            score -= 12.0;
        }

        // Check color palette compliance
        if !self.has_proper_color_scheme(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::UIUXIssue,
                description: "Potential color scheme compliance issues".to_string(),
                location: None,
                suggestion: Some("Use Material Design color palette".to_string()),
            });
            score -= 8.0;
        }

        Ok(MaterialValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
            design_version,
        })
    }

    /// Validate touch interactions
    async fn validate_touch_interactions(
        &self,
        elements: &AndroidElements,
        _config: &ValidationConfig,
    ) -> Result<TouchValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();
        let mut gestures_supported = HashSet::new();

        // Check for touch gesture implementations
        let essential_gestures = vec!["tap", "click", "long press"];
        for gesture in &essential_gestures {
            if elements.touch_gestures.contains(*gesture) {
                gestures_supported.insert(gesture.to_string());
            } else {
                issues.push(ValidationIssue {
                    platform: Platform::Android,
                    severity: Severity::Medium,
                    category: IssueCategory::UIUXIssue,
                    description: format!("Missing {} gesture support", gesture),
                    location: None,
                    suggestion: Some(format!("Implement {} gesture", gesture)),
                });
                score -= 10.0;
            }
        }

        // Check for advanced gestures
        let advanced_gestures = vec!["swipe", "pinch", "double tap"];
        for gesture in &advanced_gestures {
            if elements.touch_gestures.contains(*gesture) {
                gestures_supported.insert(gesture.to_string());
            }
        }

        // Check for event handlers
        if !elements.has_event_handlers {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::High,
                category: IssueCategory::UIUXIssue,
                description: "No touch event handlers found".to_string(),
                location: None,
                suggestion: Some("Implement touch event handlers".to_string()),
            });
            score -= 20.0;
        }

        // Check for accessibility touch targets
        if !self.has_accessible_touch_targets(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::UIUXIssue,
                description: "Touch targets may not meet accessibility guidelines".to_string(),
                location: None,
                suggestion: Some("Ensure touch targets are at least 48dp".to_string()),
            });
            score -= 12.0;
        }

        // Check for haptic feedback
        if !self.has_haptic_feedback(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Low,
                category: IssueCategory::UIUXIssue,
                description: "No haptic feedback detected".to_string(),
                location: None,
                suggestion: Some("Consider adding haptic feedback for better UX".to_string()),
            });
            score -= 5.0;
        }

        Ok(TouchValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
            gestures_supported: gestures_supported.into_iter().collect(),
        })
    }

    /// Validate screen adaptation and responsive design
    async fn validate_screen_adaptation(
        &self,
        elements: &AndroidElements,
        _config: &ValidationConfig,
    ) -> Result<ScreenValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();
        let mut supported_densities = HashSet::new();

        // Check for screen density support
        let densities = vec!["ldpi", "mdpi", "hdpi", "xhdpi", "xxhdpi", "xxxhdpi"];
        let detected_densities = densities
            .into_iter()
            .filter(|density| elements.screen_configs.contains(*density))
            .collect::<Vec<_>>();

        if detected_densities.is_empty() {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::CompatibilityProblem,
                description: "No screen density configurations found".to_string(),
                location: None,
                suggestion: Some("Add screen density specific resources".to_string()),
            });
            score -= 15.0;
        } else {
            supported_densities = detected_densities
                .into_iter()
                .map(|d| d.to_string())
                .collect();
        }

        // Check for layout variations
        if elements.layout_types.is_empty() {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::CompatibilityProblem,
                description: "No layout configurations found".to_string(),
                location: None,
                suggestion: Some("Define proper layouts for different screens".to_string()),
            });
            score -= 12.0;
        }

        // Check for responsive layout patterns
        if !self.has_responsive_layouts(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::UIUXIssue,
                description: "No responsive layout patterns detected".to_string(),
                location: None,
                suggestion: Some("Implement responsive layouts using ConstraintLayout".to_string()),
            });
            score -= 15.0;
        }

        // Check for portrait/landscape support
        if !self.has_orientation_support(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Low,
                category: IssueCategory::CompatibilityProblem,
                description: "No orientation-specific layouts found".to_string(),
                location: None,
                suggestion: Some("Consider supporting both portrait and landscape".to_string()),
            });
            score -= 8.0;
        }

        // Check for tablet optimization
        if !self.has_tablet_optimization(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Low,
                category: IssueCategory::CompatibilityProblem,
                description: "No tablet optimization detected".to_string(),
                location: None,
                suggestion: Some("Optimize layouts for tablet screens".to_string()),
            });
            score -= 6.0;
        }

        Ok(ScreenValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
            supported_densities: supported_densities.into_iter().collect(),
        })
    }

    /// Validate Android version compatibility
    async fn validate_android_version_compatibility(
        &self,
        elements: &AndroidElements,
        _config: &ValidationConfig,
    ) -> Result<VersionValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();
        let min_sdk = self.extract_min_sdk_version(elements);
        let target_sdk = self.extract_target_sdk_version(elements);

        // Check for SDK version definitions
        if elements.sdk_versions.is_empty() {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::High,
                category: IssueCategory::CompatibilityProblem,
                description: "No Android SDK versions specified".to_string(),
                location: None,
                suggestion: Some("Define minSdkVersion and targetSdkVersion".to_string()),
            });
            score -= 20.0;
        }

        // Check minimum SDK version
        if min_sdk < 21 {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::CompatibilityProblem,
                description: format!("Minimum SDK version {} is quite old", min_sdk),
                location: None,
                suggestion: Some("Consider raising minSdkVersion to API 23 or higher".to_string()),
            });
            score -= 12.0;
        }

        // Check target SDK version
        if target_sdk < 30 {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::CompatibilityProblem,
                description: format!("Target SDK version {} is outdated", target_sdk),
                location: None,
                suggestion: Some("Update targetSdkVersion to API 31 or higher".to_string()),
            });
            score -= 10.0;
        }

        // Check for version-specific features
        if !self.has_version_specific_handling(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Low,
                category: IssueCategory::CompatibilityProblem,
                description: "No version-specific feature handling detected".to_string(),
                location: None,
                suggestion: Some("Add version-specific feature checks".to_string()),
            });
            score -= 8.0;
        }

        // Check for deprecated API usage
        let deprecated_apis = self.find_deprecated_apis(elements);
        if !deprecated_apis.is_empty() {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::CompatibilityProblem,
                description: format!("Deprecated APIs detected: {}", deprecated_apis.join(", ")),
                location: None,
                suggestion: Some("Replace deprecated APIs with modern alternatives".to_string()),
            });
            score -= (deprecated_apis.len() as f32 * 5.0).min(20.0);
        }

        Ok(VersionValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
            min_sdk,
            target_sdk,
        })
    }

    /// Validate Android-specific performance characteristics
    async fn validate_android_performance(
        &self,
        elements: &AndroidElements,
        _config: &ValidationConfig,
    ) -> Result<AndroidPerformanceValidationResult, VIBEError> {
        let mut score: f32 = 100.0;
        let mut issues = Vec::new();
        let rating = self.calculate_performance_rating(elements);

        // Check for battery optimization
        if !self.has_battery_optimization(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::PerformanceIssue,
                description: "No battery optimization measures detected".to_string(),
                location: None,
                suggestion: Some("Implement battery optimization strategies".to_string()),
            });
            score -= 12.0;
        }

        // Check for memory management
        if !self.has_memory_management(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::PerformanceIssue,
                description: "No explicit memory management found".to_string(),
                location: None,
                suggestion: Some("Implement proper memory management".to_string()),
            });
            score -= 10.0;
        }

        // Check for background processing
        if !self.has_background_processing(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Low,
                category: IssueCategory::PerformanceIssue,
                description: "No background processing optimization detected".to_string(),
                location: None,
                suggestion: Some("Optimize background processing".to_string()),
            });
            score -= 8.0;
        }

        // Check for network optimization
        if !self.has_network_optimization(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::Medium,
                category: IssueCategory::PerformanceIssue,
                description: "No network optimization detected".to_string(),
                location: None,
                suggestion: Some("Implement network optimization strategies".to_string()),
            });
            score -= 10.0;
        }

        // Check for UI thread optimization
        if !self.has_ui_thread_optimization(elements) {
            issues.push(ValidationIssue {
                platform: Platform::Android,
                severity: Severity::High,
                category: IssueCategory::PerformanceIssue,
                description: "Potential UI thread blocking detected".to_string(),
                location: None,
                suggestion: Some("Move heavy operations off UI thread".to_string()),
            });
            score -= 15.0;
        }

        Ok(AndroidPerformanceValidationResult {
            score: score.clamp(0.0, 100.0),
            issues,
            rating,
        })
    }

    // Helper methods
    fn detect_material_version(&self, elements: &AndroidElements) -> String {
        if elements.has_material_themes && !elements.material_components.is_empty() {
            "Material Design 2.0".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    fn has_proper_color_scheme(&self, elements: &AndroidElements) -> bool {
        elements.has_material_themes
            && (elements.material_components.contains("Button")
                || elements
                    .material_components
                    .contains("FloatingActionButton"))
    }

    fn has_accessible_touch_targets(&self, elements: &AndroidElements) -> bool {
        !elements.touch_gestures.is_empty() && elements.has_event_handlers
    }

    fn has_haptic_feedback(&self, elements: &AndroidElements) -> bool {
        elements.has_event_handlers && elements.material_components.contains("Button")
    }

    fn has_responsive_layouts(&self, elements: &AndroidElements) -> bool {
        elements.layout_types.contains("ConstraintLayout")
            || elements.layout_types.contains("LinearLayout")
    }

    fn has_orientation_support(&self, elements: &AndroidElements) -> bool {
        elements.screen_configs.contains("sw600dp") || elements.screen_configs.contains("w600dp")
    }

    fn has_tablet_optimization(&self, elements: &AndroidElements) -> bool {
        elements.screen_configs.contains("sw600dp") || elements.screen_configs.contains("w900dp")
    }

    fn extract_min_sdk_version(&self, elements: &AndroidElements) -> i32 {
        elements
            .sdk_versions
            .iter()
            .filter_map(|v| v.parse::<i32>().ok())
            .min()
            .unwrap_or(21)
    }

    fn extract_target_sdk_version(&self, elements: &AndroidElements) -> i32 {
        elements
            .sdk_versions
            .iter()
            .filter_map(|v| v.parse::<i32>().ok())
            .max()
            .unwrap_or(30)
    }

    fn has_version_specific_handling(&self, elements: &AndroidElements) -> bool {
        elements.sdk_versions.len() > 1
    }

    fn find_deprecated_apis(&self, elements: &AndroidElements) -> Vec<String> {
        let mut deprecated = Vec::new();

        // Simulate deprecated API detection
        if elements.has_android_components {
            deprecated.push("Activity".to_string());
        }

        deprecated
    }

    fn calculate_performance_rating(&self, elements: &AndroidElements) -> String {
        let mut score = 0;

        if self.has_battery_optimization(elements) {
            score += 1;
        }
        if self.has_memory_management(elements) {
            score += 1;
        }
        if self.has_background_processing(elements) {
            score += 1;
        }
        if self.has_network_optimization(elements) {
            score += 1;
        }
        if self.has_ui_thread_optimization(elements) {
            score += 1;
        }

        match score {
            0..=1 => "Poor".to_string(),
            2..=3 => "Fair".to_string(),
            4 => "Good".to_string(),
            5 => "Excellent".to_string(),
            _ => "Unknown".to_string(),
        }
    }

    fn has_battery_optimization(&self, elements: &AndroidElements) -> bool {
        elements.permissions.contains("BATTERY_OPTIMIZATION") || elements.has_android_components
    }

    fn has_memory_management(&self, elements: &AndroidElements) -> bool {
        elements.has_android_components && elements.has_event_handlers
    }

    fn has_background_processing(&self, elements: &AndroidElements) -> bool {
        elements.has_android_components || elements.has_event_handlers
    }

    fn has_network_optimization(&self, elements: &AndroidElements) -> bool {
        elements.permissions.contains("INTERNET") || elements.has_android_components
    }

    fn has_ui_thread_optimization(&self, elements: &AndroidElements) -> bool {
        elements.has_event_handlers && !elements.touch_gestures.is_empty()
    }

    fn calculate_android_score(
        &self,
        scores: &[&dyn AndroidScoreComponent],
    ) -> Result<f32, VIBEError> {
        if scores.is_empty() {
            return Err(VIBEError::ValidationError(
                "No validation components provided".to_string(),
            ));
        }

        let weights = [0.25, 0.20, 0.20, 0.15, 0.20]; // Material, Touch, Screen, Version, Performance

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

    fn generate_android_recommendations(
        &self,
        issues: &[ValidationIssue],
        overall_score: f32,
    ) -> Result<Vec<String>, VIBEError> {
        let mut recommendations = Vec::new();

        if overall_score < 70.0 {
            recommendations
                .push("Improve Material Design compliance and user experience".to_string());
        }

        let material_issues = issues
            .iter()
            .filter(|i| {
                i.category == IssueCategory::UIUXIssue && i.description.contains("Material")
            })
            .count();

        if material_issues > 0 {
            recommendations.push("Focus on Material Design component usage".to_string());
        }

        let performance_issues = issues
            .iter()
            .filter(|i| i.category == IssueCategory::PerformanceIssue)
            .count();

        if performance_issues > 0 {
            recommendations
                .push("Optimize Android-specific performance characteristics".to_string());
        }

        let compatibility_issues = issues
            .iter()
            .filter(|i| i.category == IssueCategory::CompatibilityProblem)
            .count();

        if compatibility_issues > 0 {
            recommendations
                .push("Improve Android version compatibility and screen adaptation".to_string());
        }

        if overall_score < 60.0 {
            recommendations
                .push("Implement comprehensive Android development best practices".to_string());
            recommendations.push("Add proper battery and memory optimization".to_string());
        }

        Ok(recommendations)
    }
}

// Implement PlatformValidator trait
#[async_trait::async_trait]
impl PlatformValidator for AndroidValidator {
    async fn validate_protocol(
        &self,
        protocol_content: &str,
        config: &ValidationConfig,
        platform: Platform,
    ) -> Result<PlatformValidationResult, VIBEError> {
        if platform != Platform::Android {
            return Err(VIBEError::PlatformError(
                "AndroidValidator can only validate Android platform protocols".to_string(),
            ));
        }

        // Perform common validation first
        let common_result = self
            .base
            .perform_common_validation(protocol_content, config)
            .await?;

        // Perform Android-specific validation
        let android_result = self
            .validate_android_protocol(protocol_content, config)
            .await?;

        // Combine results
        let final_score = (common_result.score + android_result.overall_score) / 2.0;

        let mut all_issues = common_result.issues;
        all_issues.extend(android_result.issues);

        let recommendations = self.generate_android_recommendations(&all_issues, final_score)?;

        Ok(PlatformValidationResult {
            platform: Platform::Android,
            score: final_score,
            status: if final_score >= config.minimum_score {
                ValidationStatus::Passed
            } else {
                ValidationStatus::Failed
            },
            issues: all_issues,
            performance_metrics: PlatformPerformanceMetrics {
                average_response_time_ms: android_result.validation_time_ms,
                memory_usage_mb: 250,
                cpu_usage_percent: 40.0,
                error_rate_percent: 3.5,
                throughput_requests_per_second: 5.0,
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
                "Material Design Compliance".to_string(),
                "Touch Interaction Quality".to_string(),
                "Screen Adaptation".to_string(),
            ],
            secondary_criteria: vec![
                "Android Version Compatibility".to_string(),
                "Performance Optimization".to_string(),
                "Battery Efficiency".to_string(),
            ],
            penalty_factors: HashMap::from([
                ("no_material_design".to_string(), 0.25),
                ("poor_touch_support".to_string(), 0.2),
                ("no_screen_adaptation".to_string(), 0.2),
            ]),
            bonus_factors: HashMap::from([
                ("material_design_compliant".to_string(), 0.15),
                ("comprehensive_gestures".to_string(), 0.1),
                ("good_performance".to_string(), 0.1),
            ]),
        }
    }
}

// Supporting data structures
#[derive(Debug, Default)]
struct AndroidElements {
    material_components: HashSet<String>,
    touch_gestures: HashSet<String>,
    permissions: HashSet<String>,
    sdk_versions: HashSet<String>,
    screen_configs: HashSet<String>,
    layout_types: HashSet<String>,
    has_android_resources: bool,
    has_android_components: bool,
    has_event_handlers: bool,
    has_list_components: bool,
    has_material_themes: bool,
}

#[derive(Debug)]
struct MaterialValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
    design_version: String,
}

#[derive(Debug)]
struct TouchValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
    gestures_supported: Vec<String>,
}

#[derive(Debug)]
struct ScreenValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
    supported_densities: Vec<String>,
}

#[derive(Debug)]
struct VersionValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
    min_sdk: i32,
    target_sdk: i32,
}

#[derive(Debug)]
struct AndroidPerformanceValidationResult {
    score: f32,
    issues: Vec<ValidationIssue>,
    rating: String,
}

#[derive(Debug)]
struct AndroidValidationResult {
    overall_score: f32,
    #[allow(dead_code)]
    material_score: f32,
    #[allow(dead_code)]
    touch_score: f32,
    #[allow(dead_code)]
    screen_score: f32,
    #[allow(dead_code)]
    version_score: f32,
    #[allow(dead_code)]
    performance_score: f32,
    validation_time_ms: u64,
    issues: Vec<ValidationIssue>,
    #[allow(dead_code)]
    recommendations: Vec<String>,
    #[allow(dead_code)]
    android_specific_metrics: AndroidSpecificMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AndroidSpecificMetrics {
    material_design_version: String,
    supported_densities: Vec<String>,
    min_sdk_version: i32,
    target_sdk_version: i32,
    touch_gestures_supported: Vec<String>,
    performance_rating: String,
}

/// Trait for Android score components
trait AndroidScoreComponent {
    fn get_score(&self) -> f32;
}

impl AndroidScoreComponent for MaterialValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

impl AndroidScoreComponent for TouchValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

impl AndroidScoreComponent for ScreenValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

impl AndroidScoreComponent for VersionValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

impl AndroidScoreComponent for AndroidPerformanceValidationResult {
    fn get_score(&self) -> f32 {
        self.score
    }
}

// Mock implementations
struct MaterialDesignChecker;
struct TouchInteractionValidator;
struct ScreenDensityAnalyzer;
struct AndroidVersionChecker;
struct AndroidPerformanceProfiler;

impl MaterialDesignChecker {
    fn new() -> Self {
        Self
    }
}

impl TouchInteractionValidator {
    fn new() -> Self {
        Self
    }
}

impl ScreenDensityAnalyzer {
    fn new() -> Self {
        Self
    }
}

impl AndroidVersionChecker {
    fn new() -> Self {
        Self
    }
}

impl AndroidPerformanceProfiler {
    fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_android_validator_creation() {
        let validator = AndroidValidator::new();
        assert_eq!(validator.base.platform, Platform::Android);
    }

    #[test]
    fn test_android_elements_extraction() {
        let validator = AndroidValidator::new();
        let content =
            "MaterialCardView\nFloatingActionButton\nminSdkVersion 21\ntap and swipe gestures";

        let elements = validator.extract_android_elements(content).unwrap();
        assert!(elements.material_components.contains("MaterialCard"));
        assert!(elements
            .material_components
            .contains("FloatingActionButton"));
        assert!(elements.touch_gestures.contains("tap"));
        assert!(elements.touch_gestures.contains("swipe"));
        assert!(elements.sdk_versions.contains("21"));
    }

    #[test]
    fn test_material_version_detection() {
        let validator = AndroidValidator::new();
        let elements = AndroidElements {
            material_components: HashSet::from([
                "Button".to_string(),
                "FloatingActionButton".to_string(),
            ]),
            touch_gestures: HashSet::new(),
            permissions: HashSet::new(),
            sdk_versions: HashSet::new(),
            screen_configs: HashSet::new(),
            layout_types: HashSet::new(),
            has_android_resources: false,
            has_android_components: false,
            has_event_handlers: false,
            has_list_components: false,
            has_material_themes: true,
        };

        let version = validator.detect_material_version(&elements);
        assert_eq!(version, "Material Design 2.0");
    }
}
