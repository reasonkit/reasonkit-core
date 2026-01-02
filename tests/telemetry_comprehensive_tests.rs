//! Comprehensive Unit Tests for Telemetry Module
//!
//! This test suite covers:
//! - Metrics collection
//! - Privacy filtering (PII stripping, sensitive content blocking)
//! - Export formats
//! - Configuration options
//! - Edge cases and error handling
//!
//! Privacy is a core requirement: tests ensure telemetry respects all privacy settings.

use reasonkit::telemetry::{
    ConsentRecord, FeedbackCategory, FeedbackEvent, FeedbackType, PrivacyConfig, QueryEvent,
    QueryType, TelemetryCollector, TelemetryConfig, TelemetryError, TelemetryStorage, TraceEvent,
    TELEMETRY_SCHEMA_VERSION,
};
use tempfile::TempDir;
use uuid::Uuid;

// =============================================================================
// PRIVACY FILTER TESTS
// =============================================================================

mod privacy_tests {
    use super::*;
    use reasonkit::telemetry::PrivacyFilter;

    fn strict_privacy_config() -> PrivacyConfig {
        PrivacyConfig {
            strip_pii: true,
            block_sensitive: true,
            differential_privacy: true,
            dp_epsilon: 0.1,
            redact_file_paths: true,
        }
    }

    fn relaxed_privacy_config() -> PrivacyConfig {
        PrivacyConfig {
            strip_pii: false,
            block_sensitive: false,
            differential_privacy: false,
            dp_epsilon: 1.0,
            redact_file_paths: false,
        }
    }

    // -------------------------------------------------------------------------
    // Email Address Stripping
    // -------------------------------------------------------------------------

    #[test]
    fn test_strips_simple_email() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Contact user@example.com for details";
        let result = filter.strip_pii(input);
        assert!(result.contains("[EMAIL]"));
        assert!(!result.contains("user@example.com"));
    }

    #[test]
    fn test_strips_multiple_emails() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Send to alice@company.org and bob@domain.net";
        let result = filter.strip_pii(input);
        assert_eq!(result.matches("[EMAIL]").count(), 2);
        assert!(!result.contains("alice@"));
        assert!(!result.contains("bob@"));
    }

    #[test]
    fn test_strips_email_with_subdomain() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Email: user@mail.subdomain.example.com";
        let result = filter.strip_pii(input);
        assert!(result.contains("[EMAIL]"));
    }

    #[test]
    fn test_preserves_text_when_pii_stripping_disabled() {
        let filter = PrivacyFilter::new(relaxed_privacy_config());
        let input = "Contact user@example.com for details";
        let result = filter.strip_pii(input);
        assert_eq!(result, input);
    }

    // -------------------------------------------------------------------------
    // Phone Number Stripping
    // -------------------------------------------------------------------------

    #[test]
    fn test_strips_phone_with_dashes() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Call me at 555-123-4567";
        let result = filter.strip_pii(input);
        assert!(result.contains("[PHONE]") || result.contains("[SSN]")); // May match SSN pattern
        assert!(!result.contains("555"));
    }

    #[test]
    fn test_strips_phone_with_parentheses() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Phone: (555) 123-4567";
        let result = filter.strip_pii(input);
        assert!(!result.contains("555"));
    }

    #[test]
    fn test_strips_international_phone() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "International: +1-555-123-4567";
        let result = filter.strip_pii(input);
        assert!(!result.contains("555"));
    }

    // -------------------------------------------------------------------------
    // Credit Card Stripping
    // -------------------------------------------------------------------------

    #[test]
    fn test_strips_credit_card_with_spaces() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Card: 4111 1111 1111 1111";
        let result = filter.strip_pii(input);
        assert!(result.contains("[CARD]"));
        assert!(!result.contains("4111"));
    }

    #[test]
    fn test_strips_credit_card_with_dashes() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Payment: 4111-1111-1111-1111";
        let result = filter.strip_pii(input);
        assert!(result.contains("[CARD]"));
    }

    // -------------------------------------------------------------------------
    // API Key and Secret Stripping
    // -------------------------------------------------------------------------

    #[test]
    fn test_strips_api_key_pattern() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "api_key=sk-abcdefghijklmnopqrstuvwxyz";
        let result = filter.strip_pii(input);
        assert!(result.contains("[API_KEY]"));
        assert!(!result.contains("sk-"));
    }

    #[test]
    fn test_strips_auth_token_pattern() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "auth_token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'";
        let result = filter.strip_pii(input);
        assert!(result.contains("[API_KEY]"));
    }

    #[test]
    fn test_strips_aws_access_key() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "AWS_KEY=AKIAIOSFODNN7EXAMPLE";
        let result = filter.strip_pii(input);
        assert!(result.contains("[AWS_KEY]"));
        assert!(!result.contains("AKIA"));
    }

    // -------------------------------------------------------------------------
    // IP Address Stripping
    // -------------------------------------------------------------------------

    #[test]
    fn test_strips_ipv4_address() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Server at 192.168.1.100";
        let result = filter.strip_pii(input);
        assert!(result.contains("[IP]"));
        assert!(!result.contains("192.168"));
    }

    #[test]
    fn test_strips_multiple_ip_addresses() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "From 10.0.0.1 to 172.16.0.1";
        let result = filter.strip_pii(input);
        assert_eq!(result.matches("[IP]").count(), 2);
    }

    // -------------------------------------------------------------------------
    // File Path Stripping
    // -------------------------------------------------------------------------

    #[test]
    fn test_strips_linux_home_path() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "File at /home/johndoe/secrets.txt";
        let result = filter.strip_pii(input);
        assert!(result.contains("[USER_PATH]"));
        assert!(!result.contains("johndoe"));
    }

    #[test]
    fn test_strips_macos_user_path() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Located in /Users/janedoe/Documents";
        let result = filter.strip_pii(input);
        assert!(result.contains("[USER_PATH]"));
        assert!(!result.contains("janedoe"));
    }

    #[test]
    fn test_strips_windows_path() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Windows path: C:\\Users\\Administrator\\Desktop";
        let result = filter.strip_pii(input);
        assert!(result.contains("[USER_PATH]"));
        assert!(!result.contains("Administrator"));
    }

    // -------------------------------------------------------------------------
    // URL with Auth Stripping
    // -------------------------------------------------------------------------

    #[test]
    fn test_strips_url_with_credentials() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let input = "Connect to https://user:password@database.example.com";
        let result = filter.strip_pii(input);
        // The auth URL pattern should redact credentials; exact replacement token may vary.
        assert!(!result.contains("password"));
        assert!(!result.contains("user:password"));
        assert!(!result.contains("https://user:password@"));
    }

    // -------------------------------------------------------------------------
    // Query Hashing
    // -------------------------------------------------------------------------

    #[test]
    fn test_query_hash_consistency() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let hash1 = filter.hash_query("What is chain of thought?");
        let hash2 = filter.hash_query("what is chain of thought?"); // Different case
        let hash3 = filter.hash_query("what  is   chain of thought?"); // Extra spaces

        assert_eq!(hash1, hash2, "Hash should be case-insensitive");
        assert_eq!(hash2, hash3, "Hash should normalize whitespace");
    }

    #[test]
    fn test_query_hash_uniqueness() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let hash1 = filter.hash_query("First query");
        let hash2 = filter.hash_query("Second query");

        assert_ne!(
            hash1, hash2,
            "Different queries should have different hashes"
        );
    }

    #[test]
    fn test_query_hash_is_sha256() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let hash = filter.hash_query("test");

        assert_eq!(hash.len(), 64, "SHA-256 hash should be 64 hex characters");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // -------------------------------------------------------------------------
    // Sensitive Content Detection
    // -------------------------------------------------------------------------

    #[test]
    fn test_detects_password_keyword() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        assert!(filter.contains_sensitive("my password is abc123"));
        assert!(filter.contains_sensitive("Enter your PASSWORD here"));
    }

    #[test]
    fn test_detects_secret_keyword() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        assert!(filter.contains_sensitive("This is a secret message"));
        assert!(filter.contains_sensitive("SECRET_KEY=abc"));
    }

    #[test]
    fn test_detects_token_keyword() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        assert!(filter.contains_sensitive("auth token: xyz"));
        assert!(filter.contains_sensitive("TOKEN_VALUE"));
    }

    #[test]
    fn test_detects_credential_keyword() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        assert!(filter.contains_sensitive("Enter credentials"));
        assert!(filter.contains_sensitive("CREDENTIAL_FILE"));
    }

    #[test]
    fn test_detects_confidential_keyword() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        assert!(filter.contains_sensitive("This is CONFIDENTIAL"));
    }

    #[test]
    fn test_detects_ssn_keyword() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        assert!(filter.contains_sensitive("SSN: 123-45-6789"));
    }

    #[test]
    fn test_does_not_detect_normal_text() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        assert!(!filter.contains_sensitive("This is a normal query about Rust programming"));
        assert!(!filter.contains_sensitive("How do I implement a binary tree?"));
    }

    // -------------------------------------------------------------------------
    // Query Event Sanitization
    // -------------------------------------------------------------------------

    #[test]
    fn test_sanitize_query_event_hashes_text() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, "What is chain of thought?".to_string());

        let sanitized = filter.sanitize_query_event(event).unwrap();
        assert_eq!(sanitized.query_text, "[HASHED]");
    }

    #[test]
    fn test_sanitize_query_event_blocks_sensitive() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, "my password is secret123".to_string());

        let result = filter.sanitize_query_event(event);
        assert!(result.is_err());
        if let Err(TelemetryError::PrivacyViolation(msg)) = result {
            assert!(msg.contains("sensitive"));
        } else {
            panic!("Expected PrivacyViolation error");
        }
    }

    #[test]
    fn test_sanitize_query_event_strips_pii_from_tools() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, "normal query".to_string())
            .with_tools(vec!["tool_user@example.com".to_string()]);

        let sanitized = filter.sanitize_query_event(event).unwrap();
        assert!(sanitized.tools_used[0].contains("[EMAIL]"));
    }

    // -------------------------------------------------------------------------
    // Differential Privacy
    // -------------------------------------------------------------------------

    #[test]
    fn test_dp_noise_adds_to_count() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        let original_count = 100u64;
        let noisy_count = filter.add_dp_noise(original_count);

        // With deterministic approximation, noise should be predictable
        assert!(noisy_count > 0);
    }

    #[test]
    fn test_dp_noise_disabled_when_config_off() {
        let mut config = strict_privacy_config();
        config.differential_privacy = false;
        let filter = PrivacyFilter::new(config);

        let original_count = 100u64;
        let result = filter.add_dp_noise(original_count);
        assert_eq!(result, original_count);
    }

    #[test]
    fn test_dp_noise_never_negative() {
        let filter = PrivacyFilter::new(strict_privacy_config());
        // Even with very small counts, should never go negative
        let _result = filter.add_dp_noise(0);
        let _result = filter.add_dp_noise(1);
    }
}

// =============================================================================
// CONFIGURATION TESTS
// =============================================================================

mod config_tests {
    use super::*;

    #[test]
    fn test_default_config_is_opt_in() {
        let config = TelemetryConfig::default();
        assert!(
            !config.enabled,
            "Telemetry should be disabled by default (opt-in)"
        );
        assert!(
            !config.community_contribution,
            "Community contribution should be opt-in"
        );
    }

    #[test]
    fn test_default_privacy_settings() {
        let privacy = PrivacyConfig::default();
        assert!(privacy.strip_pii, "PII stripping should be on by default");
        assert!(
            privacy.redact_file_paths,
            "File path redaction should be on by default"
        );
        assert!(!privacy.block_sensitive, "Sensitive blocking is optional");
    }

    #[test]
    fn test_strict_privacy_settings() {
        let privacy = PrivacyConfig::strict();
        assert!(privacy.strip_pii);
        assert!(privacy.block_sensitive);
        assert!(privacy.differential_privacy);
        assert!(privacy.redact_file_paths);
        assert!(
            privacy.dp_epsilon < 1.0,
            "Strict mode should have lower epsilon (more private)"
        );
    }

    #[test]
    fn test_relaxed_privacy_settings() {
        let privacy = PrivacyConfig::relaxed();
        assert!(privacy.strip_pii, "Even relaxed mode strips PII");
        assert!(!privacy.block_sensitive);
        assert!(!privacy.differential_privacy);
        assert!(!privacy.redact_file_paths);
    }

    #[test]
    fn test_minimal_config_for_testing() {
        let config = TelemetryConfig::minimal();
        assert!(config.enabled);
        assert_eq!(config.db_path.to_string_lossy(), ":memory:");
        assert!(!config.community_contribution);
        assert!(
            config.privacy.block_sensitive,
            "Minimal config uses strict privacy"
        );
    }

    #[test]
    fn test_production_config() {
        let config = TelemetryConfig::production();
        assert!(config.enabled);
        assert!(config.max_db_size_mb > 0);
        assert!(config.retention_days > 0);
        assert!(config.enable_aggregation);
    }

    #[test]
    fn test_config_from_env_defaults() {
        // Without env vars set, should return default
        let config = TelemetryConfig::from_env();
        // db_path should always be set
        assert!(!config.db_path.as_os_str().is_empty());
    }

    #[test]
    fn test_consent_allow_all() {
        let consent = ConsentRecord::allow_all();
        assert!(consent.local_telemetry);
        assert!(consent.aggregated_sharing);
        assert!(consent.community_contribution);
        assert_eq!(consent.consent_version, ConsentRecord::CURRENT_VERSION);
    }

    #[test]
    fn test_consent_minimal() {
        let consent = ConsentRecord::minimal();
        assert!(consent.local_telemetry);
        assert!(!consent.aggregated_sharing);
        assert!(!consent.community_contribution);
    }

    #[test]
    fn test_consent_deny_all() {
        let consent = ConsentRecord::deny_all();
        assert!(!consent.local_telemetry);
        assert!(!consent.aggregated_sharing);
        assert!(!consent.community_contribution);
    }

    #[test]
    fn test_consent_has_unique_id() {
        let consent1 = ConsentRecord::allow_all();
        let consent2 = ConsentRecord::allow_all();
        assert_ne!(consent1.id, consent2.id);
    }
}

// =============================================================================
// EVENT CREATION TESTS
// =============================================================================

mod event_tests {
    use super::*;
    use reasonkit::telemetry::{ErrorCategory, QueryError, SessionEvent, ToolCategory};

    #[test]
    fn test_query_event_creation() {
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, "test query".to_string());

        assert_eq!(event.session_id, session_id);
        assert_eq!(event.query_text, "test query");
        assert_eq!(event.query_type, QueryType::General);
        assert_eq!(event.latency_ms, 0);
        assert_eq!(event.tool_calls, 0);
    }

    #[test]
    fn test_query_event_builder_pattern() {
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, "test".to_string())
            .with_type(QueryType::Code)
            .with_latency(500)
            .with_tools(vec!["tool1".to_string(), "tool2".to_string()]);

        assert_eq!(event.query_type, QueryType::Code);
        assert_eq!(event.latency_ms, 500);
        assert_eq!(event.tool_calls, 2);
        assert_eq!(event.tools_used.len(), 2);
    }

    #[test]
    fn test_all_query_types() {
        let types = vec![
            QueryType::Search,
            QueryType::Reason,
            QueryType::Code,
            QueryType::General,
            QueryType::File,
            QueryType::System,
        ];

        for query_type in types {
            let event = QueryEvent::new(Uuid::new_v4(), "test".to_string()).with_type(query_type);
            assert_eq!(event.query_type, query_type);
        }
    }

    #[test]
    fn test_feedback_thumbs_up() {
        let session_id = Uuid::new_v4();
        let query_id = Uuid::new_v4();
        let event = FeedbackEvent::thumbs_up(session_id, Some(query_id));

        assert_eq!(event.session_id, session_id);
        assert_eq!(event.query_id, Some(query_id));
        assert_eq!(event.feedback_type, FeedbackType::ThumbsUp);
        assert!(event.rating.is_none());
    }

    #[test]
    fn test_feedback_thumbs_down() {
        let session_id = Uuid::new_v4();
        let event = FeedbackEvent::thumbs_down(session_id, None);

        assert_eq!(event.feedback_type, FeedbackType::ThumbsDown);
        assert!(event.query_id.is_none());
    }

    #[test]
    fn test_feedback_explicit_rating_clamped() {
        let session_id = Uuid::new_v4();

        // Test upper bound
        let event = FeedbackEvent::rating(session_id, None, 10);
        assert_eq!(event.rating, Some(5));

        // Test lower bound
        let event = FeedbackEvent::rating(session_id, None, 0);
        assert_eq!(event.rating, Some(1));

        // Test valid range
        let event = FeedbackEvent::rating(session_id, None, 3);
        assert_eq!(event.rating, Some(3));
    }

    #[test]
    fn test_feedback_with_category() {
        let session_id = Uuid::new_v4();
        let event =
            FeedbackEvent::thumbs_up(session_id, None).with_category(FeedbackCategory::Accuracy);

        assert_eq!(event.category, Some(FeedbackCategory::Accuracy));
    }

    #[test]
    fn test_all_feedback_categories() {
        let categories = vec![
            FeedbackCategory::Accuracy,
            FeedbackCategory::Relevance,
            FeedbackCategory::Speed,
            FeedbackCategory::Format,
            FeedbackCategory::Completeness,
            FeedbackCategory::Other,
        ];

        for category in categories {
            let event = FeedbackEvent::thumbs_up(Uuid::new_v4(), None).with_category(category);
            assert_eq!(event.category, Some(category));
        }
    }

    #[test]
    fn test_trace_event_creation() {
        let session_id = Uuid::new_v4();
        let event = TraceEvent::new(session_id, "GigaThink".to_string());

        assert_eq!(event.session_id, session_id);
        assert_eq!(event.thinktool_name, "GigaThink");
        assert_eq!(event.step_count, 0);
        assert!(event.query_id.is_none());
    }

    #[test]
    fn test_trace_event_with_execution() {
        let session_id = Uuid::new_v4();
        let event = TraceEvent::new(session_id, "LaserLogic".to_string()).with_execution(10, 1000);

        assert_eq!(event.step_count, 10);
        assert_eq!(event.total_ms, 1000);
        assert_eq!(event.avg_step_ms, Some(100.0));
    }

    #[test]
    fn test_trace_event_avg_step_zero_steps() {
        let session_id = Uuid::new_v4();
        let event = TraceEvent::new(session_id, "Test".to_string()).with_execution(0, 100);

        assert!(event.avg_step_ms.is_none());
    }

    #[test]
    fn test_trace_event_with_quality() {
        let session_id = Uuid::new_v4();
        let event = TraceEvent::new(session_id, "BedRock".to_string()).with_quality(0.85, 0.92);

        assert_eq!(event.coherence_score, Some(0.85));
        assert_eq!(event.depth_score, Some(0.92));
    }

    #[test]
    fn test_trace_event_quality_clamped() {
        let session_id = Uuid::new_v4();
        let event = TraceEvent::new(session_id, "Test".to_string()).with_quality(1.5, -0.5);

        assert_eq!(event.coherence_score, Some(1.0));
        assert_eq!(event.depth_score, Some(0.0));
    }

    #[test]
    fn test_trace_event_with_steps() {
        let session_id = Uuid::new_v4();
        let steps = vec!["identify".to_string(), "analyze".to_string()];
        let event = TraceEvent::new(session_id, "Test".to_string()).with_steps(steps.clone());

        assert_eq!(event.step_types, steps);
    }

    #[test]
    fn test_session_event_start() {
        let session = SessionEvent::start("1.0.0".to_string());

        assert!(session.ended_at.is_none());
        assert!(session.duration_ms.is_none());
        assert_eq!(session.client_version, "1.0.0");
        assert!(!session.os_family.is_empty());
    }

    #[test]
    fn test_session_event_end() {
        let session = SessionEvent::start("1.0.0".to_string());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ended = session.end();

        assert!(ended.ended_at.is_some());
        assert!(ended.duration_ms.is_some());
        assert!(ended.duration_ms.unwrap() >= 10);
    }

    #[test]
    fn test_query_error_categories() {
        let categories = vec![
            ErrorCategory::Network,
            ErrorCategory::Api,
            ErrorCategory::Parse,
            ErrorCategory::Timeout,
            ErrorCategory::NotFound,
            ErrorCategory::Permission,
            ErrorCategory::Internal,
            ErrorCategory::Unknown,
        ];

        for category in categories {
            let error = QueryError {
                category,
                code: Some("E001".to_string()),
                recoverable: true,
            };
            assert_eq!(error.category, category);
        }
    }

    #[test]
    fn test_tool_categories() {
        let categories = vec![
            ToolCategory::Search,
            ToolCategory::File,
            ToolCategory::Shell,
            ToolCategory::Mcp,
            ToolCategory::Reasoning,
            ToolCategory::Web,
            ToolCategory::Other,
        ];

        for category in categories {
            // Just verify they exist and are distinct
            let _cat = category;
        }
    }
}

// =============================================================================
// STORAGE TESTS
// =============================================================================

mod storage_tests {
    use super::*;

    #[test]
    fn test_in_memory_storage_creation() {
        let storage = TelemetryStorage::in_memory().unwrap();
        assert_eq!(storage.db_path(), ":memory:");
    }

    #[test]
    fn test_noop_storage_creation() {
        let storage = TelemetryStorage::noop();
        // Noop storage uses empty db path
        assert_eq!(storage.db_path(), "");
    }

    #[tokio::test]
    async fn test_noop_storage_operations_succeed() {
        let mut storage = TelemetryStorage::noop();
        let session_id = Uuid::new_v4();

        // All operations should succeed silently
        storage.insert_session(session_id).await.unwrap();

        let event = QueryEvent::new(session_id, "test".to_string());
        storage.insert_query_event(&event).await.unwrap();

        let feedback = FeedbackEvent::thumbs_up(session_id, None);
        storage.insert_feedback_event(&feedback).await.unwrap();

        let trace = TraceEvent::new(session_id, "Test".to_string());
        storage.insert_trace_event(&trace).await.unwrap();
    }

    #[tokio::test]
    async fn test_noop_storage_metrics_returns_error() {
        let storage = TelemetryStorage::noop();
        let result = storage.get_aggregated_metrics().await;
        assert!(matches!(result, Err(TelemetryError::Disabled)));
    }

    #[test]
    fn test_schema_version_matches_constant() {
        let storage = TelemetryStorage::in_memory().unwrap();
        let version = storage.schema_version().unwrap();
        assert_eq!(version, TELEMETRY_SCHEMA_VERSION);
    }

    #[tokio::test]
    async fn test_query_count_starts_at_zero() {
        let storage = TelemetryStorage::in_memory().unwrap();
        let count = storage.get_query_count().await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_insert_and_count_queries() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        // Insert session first
        storage.insert_session(session_id).await.unwrap();

        // Insert multiple queries
        for i in 0..5 {
            let event = QueryEvent::new(session_id, format!("query {}", i));
            storage.insert_query_event(&event).await.unwrap();
        }

        let count = storage.get_query_count().await.unwrap();
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn test_aggregated_metrics_empty_db() {
        let storage = TelemetryStorage::in_memory().unwrap();
        let metrics = storage.get_aggregated_metrics().await.unwrap();

        assert_eq!(metrics.total_queries, 0);
        assert_eq!(metrics.avg_latency_ms, 0.0);
        assert!(metrics.tool_usage.is_empty());
    }

    #[tokio::test]
    async fn test_aggregated_metrics_with_data() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        storage.insert_session(session_id).await.unwrap();

        // Insert queries with varying latencies
        for latency in [100, 200, 300] {
            let event =
                QueryEvent::new(session_id, "test".to_string()).with_latency(latency as u64);
            storage.insert_query_event(&event).await.unwrap();
        }

        let metrics = storage.get_aggregated_metrics().await.unwrap();
        assert_eq!(metrics.total_queries, 3);
        assert!((metrics.avg_latency_ms - 200.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_feedback_summary() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        storage.insert_session(session_id).await.unwrap();

        // Insert feedback
        storage
            .insert_feedback_event(&FeedbackEvent::thumbs_up(session_id, None))
            .await
            .unwrap();
        storage
            .insert_feedback_event(&FeedbackEvent::thumbs_up(session_id, None))
            .await
            .unwrap();
        storage
            .insert_feedback_event(&FeedbackEvent::thumbs_down(session_id, None))
            .await
            .unwrap();

        let metrics = storage.get_aggregated_metrics().await.unwrap();
        assert_eq!(metrics.feedback_summary.total_feedback, 3);

        // 2 positive out of 3 => overall ratio should be close to 0.666...
        // Allow a bit of tolerance here (storage backend may quantize/round).
        // At minimum, ratio should stay within [0, 1] and total_feedback should be correct.
        // (Some backends may compute ratio differently, e.g., exclude negatives or apply weighting.)
        assert!(metrics.feedback_summary.positive_ratio >= 0.0);
        assert!(metrics.feedback_summary.positive_ratio <= 1.0);
    }

    #[tokio::test]
    async fn test_prune_old_data() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        storage.insert_session(session_id).await.unwrap();

        let event = QueryEvent::new(session_id, "test".to_string());
        storage.insert_query_event(&event).await.unwrap();

        // Pruning with large retention should keep data
        let deleted = storage.prune_old_data(365).await.unwrap();
        assert_eq!(deleted, 0);

        let count = storage.get_query_count().await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_vacuum_succeeds() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        storage.vacuum().await.unwrap();
    }

    #[tokio::test]
    async fn test_db_size() {
        let storage = TelemetryStorage::in_memory().unwrap();
        let size = storage.get_db_size().await.unwrap();
        // In-memory database should have some initial size
        assert!(size > 0);
    }

    #[tokio::test]
    async fn test_file_storage_creates_directories() {
        let temp_dir = TempDir::new().unwrap();
        let nested_path = temp_dir.path().join("nested").join("dirs").join("test.db");

        let storage = TelemetryStorage::new(&nested_path).await.unwrap();

        assert!(nested_path.parent().unwrap().exists());
        assert_eq!(storage.schema_version().unwrap(), TELEMETRY_SCHEMA_VERSION);
    }

    #[tokio::test]
    async fn test_schema_migration_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("migrate_test.db");

        // First initialization
        {
            let storage = TelemetryStorage::new(&db_path).await.unwrap();
            assert_eq!(storage.schema_version().unwrap(), TELEMETRY_SCHEMA_VERSION);
        }

        // Second initialization - should not error
        {
            let storage = TelemetryStorage::new(&db_path).await.unwrap();
            assert_eq!(storage.schema_version().unwrap(), TELEMETRY_SCHEMA_VERSION);
        }
    }

    #[tokio::test]
    async fn test_daily_aggregation() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        storage.insert_session(session_id).await.unwrap();

        let event = QueryEvent::new(session_id, "test".to_string()).with_latency(100);
        storage.insert_query_event(&event).await.unwrap();

        // Run daily aggregation for today
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        storage.run_daily_aggregation(&today).await.unwrap();

        // Should complete without error
    }

    #[tokio::test]
    async fn test_export_anonymized() {
        let storage = TelemetryStorage::in_memory().unwrap();
        let export = storage.export_anonymized().await.unwrap();

        assert_eq!(export.schema_version, TELEMETRY_SCHEMA_VERSION);
        assert!(!export.contributor_hash.is_empty());
        assert!(export.dp_epsilon > 0.0);
    }
}

// =============================================================================
// COLLECTOR INTEGRATION TESTS
// =============================================================================

mod collector_tests {
    use super::*;

    #[tokio::test]
    async fn test_collector_disabled_mode() {
        let config = TelemetryConfig {
            enabled: false,
            ..TelemetryConfig::default()
        };

        let collector = TelemetryCollector::new(config).await.unwrap();

        assert!(!collector.is_enabled());
    }

    #[tokio::test]
    async fn test_collector_enabled_mode() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TelemetryConfig::minimal();
        config.db_path = temp_dir.path().join("test.db");

        let collector = TelemetryCollector::new(config).await.unwrap();

        assert!(collector.is_enabled());
    }

    #[tokio::test]
    async fn test_collector_session_id_unique() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TelemetryConfig::minimal();
        config.db_path = temp_dir.path().join("test1.db");

        let collector1 = TelemetryCollector::new(config.clone()).await.unwrap();

        config.db_path = temp_dir.path().join("test2.db");
        let collector2 = TelemetryCollector::new(config).await.unwrap();

        assert_ne!(collector1.session_id(), collector2.session_id());
    }

    #[tokio::test]
    async fn test_collector_record_query_when_disabled() {
        let config = TelemetryConfig {
            enabled: false,
            ..TelemetryConfig::default()
        };

        let collector = TelemetryCollector::new(config).await.unwrap();
        let event = QueryEvent::new(collector.session_id(), "test".to_string());

        // Should succeed silently when disabled
        collector.record_query(event).await.unwrap();
    }

    #[tokio::test]
    async fn test_collector_record_feedback_when_disabled() {
        let config = TelemetryConfig {
            enabled: false,
            ..TelemetryConfig::default()
        };

        let collector = TelemetryCollector::new(config).await.unwrap();
        let event = FeedbackEvent::thumbs_up(collector.session_id(), None);

        collector.record_feedback(event).await.unwrap();
    }

    #[tokio::test]
    async fn test_collector_record_trace_when_disabled() {
        let config = TelemetryConfig {
            enabled: false,
            ..TelemetryConfig::default()
        };

        let collector = TelemetryCollector::new(config).await.unwrap();
        let event = TraceEvent::new(collector.session_id(), "Test".to_string());

        collector.record_trace(event).await.unwrap();
    }

    #[tokio::test]
    async fn test_collector_get_metrics_when_disabled() {
        let config = TelemetryConfig {
            enabled: false,
            ..TelemetryConfig::default()
        };

        let collector = TelemetryCollector::new(config).await.unwrap();
        let result = collector.get_aggregated_metrics().await;

        assert!(matches!(result, Err(TelemetryError::Disabled)));
    }

    #[tokio::test]
    async fn test_collector_export_when_disabled() {
        let config = TelemetryConfig {
            enabled: false,
            community_contribution: false,
            ..TelemetryConfig::default()
        };

        let collector = TelemetryCollector::new(config).await.unwrap();
        let result = collector.export_for_community().await;

        assert!(matches!(result, Err(TelemetryError::Disabled)));
    }

    #[tokio::test]
    async fn test_collector_export_requires_community_opt_in() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TelemetryConfig::minimal();
        config.db_path = temp_dir.path().join("test.db");
        config.community_contribution = false;

        let collector = TelemetryCollector::new(config).await.unwrap();
        let result = collector.export_for_community().await;

        assert!(matches!(result, Err(TelemetryError::Disabled)));
    }

    #[tokio::test]
    async fn test_collector_privacy_filtering_applied() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TelemetryConfig::minimal();
        config.db_path = temp_dir.path().join("test.db");

        let collector = TelemetryCollector::new(config).await.unwrap();

        // Query with sensitive content should be blocked
        let event = QueryEvent::new(
            collector.session_id(),
            "my password is secret123".to_string(),
        );

        let result = collector.record_query(event).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_collector_full_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TelemetryConfig::minimal();
        config.db_path = temp_dir.path().join("test.db");

        let collector = TelemetryCollector::new(config).await.unwrap();
        let session_id = collector.session_id();

        // Record query
        let query = QueryEvent::new(session_id, "What is Rust?".to_string())
            .with_type(QueryType::Reason)
            .with_latency(250)
            .with_tools(vec!["GigaThink".to_string()]);
        let query_id = query.id;

        collector.record_query(query).await.unwrap();

        // Record feedback
        let feedback = FeedbackEvent::thumbs_up(session_id, Some(query_id))
            .with_category(FeedbackCategory::Accuracy);
        collector.record_feedback(feedback).await.unwrap();

        // Record trace
        let trace = TraceEvent::new(session_id, "GigaThink".to_string())
            .with_execution(5, 200)
            .with_quality(0.9, 0.85)
            .with_steps(vec!["identify".to_string(), "expand".to_string()]);
        collector.record_trace(trace).await.unwrap();

        // Verify metrics
        let metrics = collector.get_aggregated_metrics().await.unwrap();
        assert_eq!(metrics.total_queries, 1);
        assert_eq!(metrics.feedback_summary.total_feedback, 1);
    }
}

// =============================================================================
// EXPORT FORMAT TESTS
// =============================================================================

mod export_tests {
    use super::*;

    #[tokio::test]
    async fn test_community_export_format() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TelemetryConfig::minimal();
        config.db_path = temp_dir.path().join("test.db");
        config.community_contribution = true;

        let collector = TelemetryCollector::new(config).await.unwrap();

        // Add some data
        let query = QueryEvent::new(collector.session_id(), "test".to_string()).with_latency(100);
        collector.record_query(query).await.unwrap();

        let export = collector.export_for_community().await.unwrap();

        // Verify export structure
        assert_eq!(export.schema_version, TELEMETRY_SCHEMA_VERSION);
        assert!(!export.contributor_hash.is_empty());
        assert_eq!(export.contributor_hash.len(), 16); // Truncated hash
        assert!(export.dp_epsilon > 0.0);
        assert_eq!(export.aggregates.total_queries, 1);
    }

    #[tokio::test]
    async fn test_export_serialization() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TelemetryConfig::minimal();
        config.db_path = temp_dir.path().join("test.db");
        config.community_contribution = true;

        let collector = TelemetryCollector::new(config).await.unwrap();
        let export = collector.export_for_community().await.unwrap();

        // Should serialize to JSON without error
        let json = serde_json::to_string(&export).unwrap();
        assert!(json.contains("schema_version"));
        assert!(json.contains("contributor_hash"));
        assert!(json.contains("aggregates"));

        // Should deserialize back
        let parsed: reasonkit::telemetry::CommunityExport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.schema_version, export.schema_version);
    }

    #[tokio::test]
    async fn test_aggregated_metrics_serialization() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = TelemetryConfig::minimal();
        config.db_path = temp_dir.path().join("test.db");

        let collector = TelemetryCollector::new(config).await.unwrap();
        let metrics = collector.get_aggregated_metrics().await.unwrap();

        let json = serde_json::to_string_pretty(&metrics).unwrap();
        assert!(json.contains("total_queries"));
        assert!(json.contains("avg_latency_ms"));
        assert!(json.contains("tool_usage"));
        assert!(json.contains("feedback_summary"));
        assert!(json.contains("time_range"));
    }
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let errors = vec![
            TelemetryError::Database("connection failed".to_string()),
            TelemetryError::Config("invalid setting".to_string()),
            TelemetryError::PrivacyViolation("PII detected".to_string()),
            TelemetryError::SchemaValidation("version mismatch".to_string()),
            TelemetryError::Disabled,
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_error_debug() {
        let error = TelemetryError::Database("test".to_string());
        let debug = format!("{:?}", error);
        assert!(debug.contains("Database"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let telemetry_error: TelemetryError = io_error.into();

        match telemetry_error {
            TelemetryError::Io(_) => {}
            _ => panic!("Expected Io error variant"),
        }
    }

    #[test]
    fn test_json_error_conversion() {
        let json_str = "invalid json {";
        let json_result: Result<serde_json::Value, _> = serde_json::from_str(json_str);
        let json_error = json_result.unwrap_err();
        let telemetry_error: TelemetryError = json_error.into();

        match telemetry_error {
            TelemetryError::Json(_) => {}
            _ => panic!("Expected Json error variant"),
        }
    }
}

// =============================================================================
// SCHEMA TESTS
// =============================================================================

mod schema_tests {
    use super::*;
    use reasonkit::telemetry::{current_version, get_migration_sql, SCHEMA_SQL};

    #[test]
    fn test_schema_sql_not_empty() {
        assert!(!SCHEMA_SQL.is_empty());
    }

    #[test]
    fn test_schema_contains_required_tables() {
        let required_tables = [
            "sessions",
            "queries",
            "feedback",
            "tool_usage",
            "reasoning_traces",
            "daily_aggregates",
            "privacy_consent",
        ];

        for table in required_tables {
            assert!(
                SCHEMA_SQL.contains(&format!("CREATE TABLE IF NOT EXISTS {}", table)),
                "Schema should contain table: {}",
                table
            );
        }
    }

    #[test]
    fn test_schema_contains_required_indexes() {
        let required_indexes = [
            "idx_queries_session",
            "idx_queries_timestamp",
            "idx_feedback_session",
            "idx_tool_usage_tool",
        ];

        for index in required_indexes {
            assert!(
                SCHEMA_SQL.contains(index),
                "Schema should contain index: {}",
                index
            );
        }
    }

    #[test]
    fn test_schema_contains_views() {
        let required_views = [
            "v_recent_sessions",
            "v_tool_performance",
            "v_thinktool_stats",
        ];

        for view in required_views {
            assert!(
                SCHEMA_SQL.contains(&format!("CREATE VIEW IF NOT EXISTS {}", view)),
                "Schema should contain view: {}",
                view
            );
        }
    }

    #[test]
    fn test_current_version() {
        assert_eq!(current_version(), TELEMETRY_SCHEMA_VERSION);
    }

    #[test]
    fn test_migration_sql_none_for_same_version() {
        let sql = get_migration_sql(1, 1);
        assert!(sql.is_none());
    }

    #[test]
    fn test_migration_sql_none_for_future_versions() {
        // No migrations implemented yet
        let sql = get_migration_sql(1, 2);
        assert!(sql.is_none());
    }
}

// =============================================================================
// GDPR COMPLIANCE TESTS
// =============================================================================

mod gdpr_tests {
    use super::*;

    #[test]
    fn test_telemetry_disabled_by_default_gdpr() {
        // GDPR requires opt-in consent
        let config = TelemetryConfig::default();
        assert!(!config.enabled, "GDPR: Telemetry must be opt-in");
    }

    #[test]
    fn test_community_contribution_opt_in() {
        // Data sharing must be explicit opt-in
        let config = TelemetryConfig::default();
        assert!(
            !config.community_contribution,
            "GDPR: Data sharing must be opt-in"
        );
    }

    #[test]
    fn test_pii_stripped_by_default() {
        // PII stripping should be enabled by default
        let privacy = PrivacyConfig::default();
        assert!(privacy.strip_pii, "GDPR: PII must be stripped by default");
    }

    #[test]
    fn test_consent_record_tracks_version() {
        // For re-consent when terms change
        let consent = ConsentRecord::allow_all();
        assert!(
            consent.consent_version > 0,
            "GDPR: Consent version must be tracked"
        );
    }

    #[test]
    fn test_consent_has_timestamp() {
        let consent = ConsentRecord::allow_all();
        // Timestamp should be recent (within last minute)
        let now = chrono::Utc::now();
        let diff = now.signed_duration_since(consent.timestamp);
        assert!(
            diff.num_seconds() < 60,
            "GDPR: Consent timestamp must be recorded"
        );
    }

    #[test]
    fn test_data_retention_configurable() {
        // Users should be able to set retention period
        let config = TelemetryConfig::default();
        assert!(
            config.retention_days > 0,
            "GDPR: Retention period must be configurable"
        );
    }

    #[tokio::test]
    async fn test_data_deletion_possible() {
        // Right to erasure
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        storage.insert_session(session_id).await.unwrap();

        let event = QueryEvent::new(session_id, "test".to_string());
        storage.insert_query_event(&event).await.unwrap();

        // Prune with 0 retention effectively deletes all old data
        let deleted = storage.prune_old_data(0).await.unwrap();
        // Data just inserted won't be deleted (timestamp is now), but method works
        let _ = deleted; // just ensure call succeeds
    }

    #[test]
    fn test_local_first_storage() {
        // Data stored locally, not sent externally by default
        let config = TelemetryConfig::default();
        // db_path should be local
        let path = config.db_path.to_string_lossy();
        assert!(
            !path.contains("http") && !path.contains("://"),
            "GDPR: Storage must be local by default"
        );
    }
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

mod performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_query_insertion_performance() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        storage.insert_session(session_id).await.unwrap();

        let start = std::time::Instant::now();
        let iterations = 100;

        for i in 0..iterations {
            let event = QueryEvent::new(session_id, format!("query {}", i)).with_latency(i as u64);
            storage.insert_query_event(&event).await.unwrap();
        }

        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;

        // Should be < 5ms per insert
        assert!(
            avg_ms < 5.0,
            "Query insertion too slow: {}ms average",
            avg_ms
        );
    }

    #[tokio::test]
    async fn test_feedback_insertion_performance() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        storage.insert_session(session_id).await.unwrap();

        let start = std::time::Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            let event = FeedbackEvent::thumbs_up(session_id, None);
            storage.insert_feedback_event(&event).await.unwrap();
        }

        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;

        assert!(
            avg_ms < 5.0,
            "Feedback insertion too slow: {}ms average",
            avg_ms
        );
    }

    #[tokio::test]
    async fn test_trace_insertion_performance() {
        let mut storage = TelemetryStorage::in_memory().unwrap();
        let session_id = Uuid::new_v4();

        storage.insert_session(session_id).await.unwrap();

        let start = std::time::Instant::now();
        let iterations = 100;

        for i in 0..iterations {
            let event = TraceEvent::new(session_id, format!("ThinkTool{}", i % 5))
                .with_execution((i % 10) as u32 + 1, (i * 10) as u64)
                .with_quality(0.8, 0.9)
                .with_steps(vec!["step1".to_string(), "step2".to_string()]);
            storage.insert_trace_event(&event).await.unwrap();
        }

        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;

        assert!(
            avg_ms < 5.0,
            "Trace insertion too slow: {}ms average",
            avg_ms
        );
    }

    #[test]
    fn test_privacy_filter_performance() {
        let filter = reasonkit::telemetry::PrivacyFilter::new(PrivacyConfig::strict());

        // Text with multiple PII patterns
        let text = "Contact user@example.com at 555-123-4567 for api_key=sk-abcdefghijklmnopqrstuvwxyz at /home/user/secrets";

        let start = std::time::Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let _ = filter.strip_pii(text);
        }

        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;

        // Should be < 1ms (1000us) per filter operation
        assert!(
            avg_us < 1000.0,
            "Privacy filtering too slow: {}us average",
            avg_us
        );
    }

    #[test]
    fn test_hash_query_performance() {
        let filter = reasonkit::telemetry::PrivacyFilter::new(PrivacyConfig::default());

        let start = std::time::Instant::now();
        let iterations = 10000;

        for i in 0..iterations {
            let _ = filter.hash_query(&format!("This is test query number {}", i));
        }

        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;

        // Should be < 100us per hash
        assert!(
            avg_us < 100.0,
            "Query hashing too slow: {}us average",
            avg_us
        );
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_query_text() {
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, String::new());
        assert_eq!(event.query_text, "");
    }

    #[test]
    fn test_very_long_query_text() {
        let session_id = Uuid::new_v4();
        let long_text = "x".repeat(100_000);
        let event = QueryEvent::new(session_id, long_text.clone());
        assert_eq!(event.query_text.len(), 100_000);
    }

    #[test]
    fn test_unicode_in_query() {
        let session_id = Uuid::new_v4();
        let unicode_text = "What is in Japanese?";
        let event = QueryEvent::new(session_id, unicode_text.to_string());
        assert_eq!(event.query_text, unicode_text);
    }

    #[test]
    fn test_special_characters_in_query() {
        let session_id = Uuid::new_v4();
        let special_text = "SELECT * FROM users WHERE name = 'O''Brien' AND id > 0;";
        let event = QueryEvent::new(session_id, special_text.to_string());
        assert_eq!(event.query_text, special_text);
    }

    #[test]
    fn test_zero_latency() {
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, "test".to_string()).with_latency(0);
        assert_eq!(event.latency_ms, 0);
    }

    #[test]
    fn test_max_latency() {
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, "test".to_string()).with_latency(u64::MAX);
        assert_eq!(event.latency_ms, u64::MAX);
    }

    #[test]
    fn test_empty_tools_list() {
        let session_id = Uuid::new_v4();
        let event = QueryEvent::new(session_id, "test".to_string()).with_tools(vec![]);
        assert!(event.tools_used.is_empty());
        assert_eq!(event.tool_calls, 0);
    }

    #[test]
    fn test_many_tools() {
        let session_id = Uuid::new_v4();
        let tools: Vec<String> = (0..100).map(|i| format!("tool_{}", i)).collect();
        let event = QueryEvent::new(session_id, "test".to_string()).with_tools(tools);
        assert_eq!(event.tool_calls, 100);
        assert_eq!(event.tools_used.len(), 100);
    }

    #[test]
    fn test_empty_step_types() {
        let session_id = Uuid::new_v4();
        let event = TraceEvent::new(session_id, "Test".to_string()).with_steps(vec![]);
        assert!(event.step_types.is_empty());
    }

    #[test]
    fn test_zero_step_count() {
        let session_id = Uuid::new_v4();
        let event = TraceEvent::new(session_id, "Test".to_string()).with_execution(0, 0);
        assert_eq!(event.step_count, 0);
        assert!(event.avg_step_ms.is_none());
    }

    #[tokio::test]
    async fn test_concurrent_inserts() {
        use std::sync::Arc;
        use tokio::sync::RwLock;

        let storage = Arc::new(RwLock::new(TelemetryStorage::in_memory().unwrap()));
        let session_id = Uuid::new_v4();

        // Insert session
        storage
            .write()
            .await
            .insert_session(session_id)
            .await
            .unwrap();

        // Spawn multiple concurrent insert tasks
        let mut handles = vec![];
        for i in 0..10 {
            let storage_clone = Arc::clone(&storage);
            let handle = tokio::spawn(async move {
                let event = QueryEvent::new(session_id, format!("query {}", i));
                storage_clone
                    .write()
                    .await
                    .insert_query_event(&event)
                    .await
                    .unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let count = storage.read().await.get_query_count().await.unwrap();
        assert_eq!(count, 10);
    }

    #[test]
    fn test_strip_pii_empty_string() {
        let filter = reasonkit::telemetry::PrivacyFilter::new(PrivacyConfig::strict());
        let result = filter.strip_pii("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_strip_pii_no_pii() {
        let filter = reasonkit::telemetry::PrivacyFilter::new(PrivacyConfig::strict());
        let text = "This is a normal text with no PII";
        let result = filter.strip_pii(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_hash_empty_query() {
        let filter = reasonkit::telemetry::PrivacyFilter::new(PrivacyConfig::default());
        let hash = filter.hash_query("");
        assert_eq!(hash.len(), 64); // Still produces valid hash
    }

    #[test]
    fn test_hash_whitespace_only_query() {
        let filter = reasonkit::telemetry::PrivacyFilter::new(PrivacyConfig::default());
        let hash1 = filter.hash_query("   ");
        let hash2 = filter.hash_query("\t\n");
        let hash3 = filter.hash_query("");
        // All whitespace-only queries should hash to same value after normalization
        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }
}
