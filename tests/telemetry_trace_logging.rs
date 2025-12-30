//! Integration test for SQLite trace logging
//!
//! Tests that reasoning traces are correctly logged to SQLite database.

use reasonkit::telemetry::{
    TelemetryCollector, TelemetryConfig, TelemetryError, TelemetryResult, TraceEvent,
};
use tempfile::TempDir;

#[tokio::test]
async fn test_trace_logging_basic() -> TelemetryResult<()> {
    // Create temporary directory for test database
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join(".rk_telemetry.db");

    // Initialize telemetry with test database
    let mut config = TelemetryConfig::minimal();
    config.db_path = db_path.clone();

    let collector = TelemetryCollector::new(config).await?;
    let session_id = collector.session_id();

    // Create a test trace event
    let trace_event = TraceEvent::new(session_id, "GigaThink".to_string())
        .with_execution(5, 1234)
        .with_quality(0.85, 0.92)
        .with_steps(vec![
            "identify_dimensions".to_string(),
            "explore_perspectives".to_string(),
        ]);
    let trace_event_id = trace_event.id;
    let trace_event_session_id = trace_event.session_id;

    // Record the trace
    collector.record_trace(trace_event.clone()).await?;

    // Verify trace was written to database
    use rusqlite::Connection;
    let conn = Connection::open(&db_path).map_err(|e| TelemetryError::Database(e.to_string()))?;

    // Query the reasoning_traces table
    let mut stmt = conn.prepare(
        "SELECT id, session_id, thinktool_name, step_count, total_ms, coherence_score, depth_score
         FROM reasoning_traces
         WHERE id = ?1"
    )
    .map_err(|e| TelemetryError::Database(e.to_string()))?;

    let trace_row = stmt
        .query_row([&trace_event_id.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?, // id
                row.get::<_, String>(1)?, // session_id
                row.get::<_, String>(2)?, // thinktool_name
                row.get::<_, i64>(3)?,    // step_count
                row.get::<_, i64>(4)?,    // total_ms
                row.get::<_, f64>(5)?,    // coherence_score
                row.get::<_, f64>(6)?,    // depth_score
            ))
        })
        .map_err(|e| TelemetryError::Database(e.to_string()))?;

    // Verify all fields match
    assert_eq!(trace_row.0, trace_event_id.to_string());
    assert_eq!(trace_row.1, trace_event_session_id.to_string());
    assert_eq!(trace_row.2, "GigaThink");
    assert_eq!(trace_row.3, 5);
    assert_eq!(trace_row.4, 1234);
    assert!((trace_row.5 - 0.85).abs() < 0.001);
    assert!((trace_row.6 - 0.92).abs() < 0.001);

    Ok(())
}

#[tokio::test]
async fn test_trace_logging_multiple_traces() -> TelemetryResult<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join(".rk_telemetry.db");

    let mut config = TelemetryConfig::minimal();
    config.db_path = db_path.clone();

    let collector = TelemetryCollector::new(config).await?;
    let session_id = collector.session_id();

    // Record multiple traces
    let thinktools = ["GigaThink", "LaserLogic", "BedRock", "ProofGuard"];

    for (i, thinktool) in thinktools.iter().enumerate() {
        let trace_event = TraceEvent::new(session_id, thinktool.to_string())
            .with_execution(((i + 1) * 2) as u32, ((i + 1) * 100) as u64)
            .with_quality(0.8 + (i as f64 * 0.05), 0.75 + (i as f64 * 0.05))
            .with_steps(vec!["step1".to_string(), "step2".to_string()]);

        collector.record_trace(trace_event).await?;
    }

    // Verify all traces were written
    use rusqlite::Connection;
    let conn = Connection::open(&db_path).map_err(|e| TelemetryError::Database(e.to_string()))?;

    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_traces WHERE session_id = ?1",
            [&session_id.to_string()],
            |row| row.get(0),
        )
        .map_err(|e| TelemetryError::Database(e.to_string()))?;

    assert_eq!(count, 4);

    // Verify we can query by thinktool
    let gigathink_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_traces WHERE thinktool_name = 'GigaThink'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| TelemetryError::Database(e.to_string()))?;

    assert_eq!(gigathink_count, 1);

    Ok(())
}

#[tokio::test]
async fn test_trace_logging_with_query() -> TelemetryResult<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join(".rk_telemetry.db");

    let mut config = TelemetryConfig::minimal();
    config.db_path = db_path.clone();

    let collector = TelemetryCollector::new(config).await?;
    let session_id = collector.session_id();

    // Record query event
    use reasonkit::telemetry::{QueryEvent, QueryType};
    let query_event = QueryEvent::new(session_id, "Test query".to_string())
        .with_type(QueryType::Reason)
        .with_latency(500)
        .with_tools(vec!["GigaThink".to_string()]);
    let query_id = query_event.id; // Use the actual query event's ID

    collector.record_query(query_event).await?;

    // Record trace linked to query
    let mut trace_event = TraceEvent::new(session_id, "GigaThink".to_string())
        .with_execution(3, 450)
        .with_quality(0.88, 0.91)
        .with_steps(vec![
            "identify".to_string(),
            "explore".to_string(),
            "synthesize".to_string(),
        ]);
    trace_event.query_id = Some(query_id);
    let trace_event_id = trace_event.id;

    collector.record_trace(trace_event.clone()).await?;

    // Verify trace is linked to query
    use rusqlite::Connection;
    let conn = Connection::open(&db_path).map_err(|e| TelemetryError::Database(e.to_string()))?;

    let linked_query_id: Option<String> = conn
        .query_row(
            "SELECT query_id FROM reasoning_traces WHERE id = ?1",
            [&trace_event_id.to_string()],
            |row| row.get(0),
        )
        .map_err(|e| TelemetryError::Database(e.to_string()))?;

    assert_eq!(linked_query_id, Some(query_id.to_string()));

    Ok(())
}

#[tokio::test]
async fn test_trace_logging_disabled() -> TelemetryResult<()> {
    // Test that trace logging is skipped when telemetry is disabled
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join(".rk_telemetry.db");

    let mut config = TelemetryConfig::minimal();
    config.enabled = false; // Disabled
    config.db_path = db_path.clone();

    let collector = TelemetryCollector::new(config).await?;

    // Should not fail when disabled
    let trace_event = TraceEvent::new(collector.session_id(), "GigaThink".to_string())
        .with_execution(1, 100)
        .with_quality(0.8, 0.8);

    // Should succeed but not write to database
    collector.record_trace(trace_event).await?;

    // Verify database was not created
    assert!(!db_path.exists());

    Ok(())
}

#[tokio::test]
async fn test_trace_logging_step_types_json() -> TelemetryResult<()> {
    // Test that step_types are correctly serialized as JSON
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join(".rk_telemetry.db");

    let mut config = TelemetryConfig::minimal();
    config.db_path = db_path.clone();

    let collector = TelemetryCollector::new(config).await?;
    let session_id = collector.session_id();

    let step_types = vec![
        "identify_dimensions".to_string(),
        "explore_perspectives".to_string(),
        "synthesize".to_string(),
    ];

    let trace_event = TraceEvent::new(session_id, "GigaThink".to_string())
        .with_execution(step_types.len() as u32, 500)
        .with_quality(0.9, 0.95)
        .with_steps(step_types.clone());
    let trace_event_id = trace_event.id;

    collector.record_trace(trace_event.clone()).await?;

    // Verify step_types JSON is stored correctly
    use rusqlite::Connection;
    let conn = Connection::open(&db_path).map_err(|e| TelemetryError::Database(e.to_string()))?;

    let step_types_json: String = conn
        .query_row(
            "SELECT step_types FROM reasoning_traces WHERE id = ?1",
            [&trace_event_id.to_string()],
            |row| row.get(0),
        )
        .map_err(|e| TelemetryError::Database(e.to_string()))?;

    // Parse JSON and verify
    let parsed: Vec<String> = serde_json::from_str(&step_types_json)?;
    assert_eq!(parsed, step_types);

    Ok(())
}

#[tokio::test]
async fn test_trace_logging_performance() -> TelemetryResult<()> {
    // Test that trace logging is fast (< 5ms overhead)
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join(".rk_telemetry.db");

    let mut config = TelemetryConfig::minimal();
    config.db_path = db_path.clone();

    let collector = TelemetryCollector::new(config).await?;
    let session_id = collector.session_id();

    // Record 100 traces and measure time
    let start = std::time::Instant::now();

    for i in 0..100 {
        let trace_event = TraceEvent::new(session_id, format!("ThinkTool{}", i % 5))
            .with_execution((i % 10) as u32 + 1, (i * 10) as u64)
            .with_quality(0.8, 0.8)
            .with_steps(vec!["step1".to_string()]);

        collector.record_trace(trace_event).await?;
    }

    let elapsed = start.elapsed();
    let avg_per_trace = elapsed.as_millis() as f64 / 100.0;

    // Should be < 5ms per trace on average
    assert!(
        avg_per_trace < 5.0,
        "Average trace logging time {}ms exceeds 5ms target",
        avg_per_trace
    );

    Ok(())
}
