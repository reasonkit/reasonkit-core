# Task 49: SQLite DB Logic Implementation

> **Status:** IN PROGRESS  
> **Assigned Model:** Claude Sonnet 4.5  
> **Date:** 2025-12-29  
> **Related:** [TASK_47_ML_SCHEMA_PLAN.md](TASK_47_ML_SCHEMA_PLAN.md), [ADR-002](../adr/ADR-002-sqlite-for-audit-trail.md)

---

## Executive Summary

Implement SQLite database logic for RALL telemetry system. Build on existing `TelemetryStorage` implementation to add:

- Complete CRUD operations for all tables
- Efficient query methods for ML training
- Batch operations for performance
- Migration system for schema updates
- Transaction management for data integrity

**Current State:** Basic storage exists in `src/telemetry/storage.rs` - needs enhancement for ML operations.

---

## Objectives

1. **Complete CRUD Operations:** Implement all database operations for ML tables
2. **Query Optimization:** Add efficient queries for ML training data export
3. **Batch Operations:** Implement batch inserts for performance
4. **Migration System:** Build schema migration framework
5. **Transaction Management:** Ensure ACID compliance for critical operations

---

## Current Implementation Analysis

### Existing Functionality (from `src/telemetry/storage.rs`)

| Feature                 | Status      | Notes                           |
| ----------------------- | ----------- | ------------------------------- |
| Database initialization | ✅ Complete | `new()`, `initialize_default()` |
| Schema migration        | ✅ Partial  | Basic version checking exists   |
| Session management      | ✅ Complete | `insert_session()`              |
| Query event storage     | ✅ Complete | `insert_query_event()`          |
| Feedback storage        | ✅ Complete | `insert_feedback_event()`       |
| Trace storage           | ✅ Complete | `insert_trace_event()`          |
| Aggregated metrics      | ✅ Partial  | Basic aggregation exists        |
| Privacy export          | ✅ Complete | `export_anonymized()`           |

### Implementation Gaps

1. **ML Table Operations:**
   - [ ] `model_training_runs` CRUD
   - [ ] `feature_vectors` CRUD
   - [ ] `model_predictions` CRUD
   - [ ] `ab_test_results` CRUD

2. **Advanced Queries:**
   - [ ] Training data export queries
   - [ ] Model performance queries
   - [ ] Feature importance queries
   - [ ] A/B test comparison queries

3. **Batch Operations:**
   - [ ] Batch insert for feature vectors
   - [ ] Batch insert for predictions
   - [ ] Batch update for aggregates

4. **Migration System:**
   - [ ] Version-based migrations
   - [ ] Rollback support
   - [ ] Migration validation

5. **Transaction Management:**
   - [ ] Explicit transaction support
   - [ ] Savepoint support
   - [ ] Deadlock handling

---

## Implementation Plan

### Phase 1: ML Table Operations (Week 1)

#### 1.1 Model Training Runs

```rust
impl TelemetryStorage {
    /// Insert a model training run
    pub async fn insert_training_run(
        &self,
        run: &ModelTrainingRun,
    ) -> TelemetryResult<()> {
        // Implementation
    }

    /// Get training run by ID
    pub async fn get_training_run(
        &self,
        run_id: &str,
    ) -> TelemetryResult<ModelTrainingRun> {
        // Implementation
    }

    /// List training runs
    pub async fn list_training_runs(
        &self,
        model_type: Option<&str>,
        limit: Option<u32>,
    ) -> TelemetryResult<Vec<ModelTrainingRun>> {
        // Implementation
    }

    /// Update training run with results
    pub async fn update_training_run_results(
        &self,
        run_id: &str,
        metrics: &TrainingMetrics,
    ) -> TelemetryResult<()> {
        // Implementation
    }
}
```

#### 1.2 Feature Vectors

```rust
impl TelemetryStorage {
    /// Insert feature vector
    pub async fn insert_feature_vector(
        &self,
        vector: &FeatureVector,
    ) -> TelemetryResult<()> {
        // Implementation
    }

    /// Get feature vectors for query
    pub async fn get_feature_vectors_for_query(
        &self,
        query_id: &str,
    ) -> TelemetryResult<Vec<FeatureVector>> {
        // Implementation
    }

    /// Batch insert feature vectors
    pub async fn batch_insert_feature_vectors(
        &self,
        vectors: &[FeatureVector],
    ) -> TelemetryResult<()> {
        // Implementation with transaction
    }

    /// Get similar queries (by embedding)
    pub async fn find_similar_queries(
        &self,
        embedding: &[f32],
        limit: u32,
    ) -> TelemetryResult<Vec<String>> {
        // Implementation with cosine similarity
    }
}
```

#### 1.3 Model Predictions

```rust
impl TelemetryStorage {
    /// Insert model prediction
    pub async fn insert_prediction(
        &self,
        prediction: &ModelPrediction,
    ) -> TelemetryResult<()> {
        // Implementation
    }

    /// Update prediction with ground truth
    pub async fn update_prediction_ground_truth(
        &self,
        prediction_id: &str,
        actual_value: &str,
    ) -> TelemetryResult<()> {
        // Implementation
    }

    /// Get predictions for model run
    pub async fn get_predictions_for_run(
        &self,
        run_id: &str,
    ) -> TelemetryResult<Vec<ModelPrediction>> {
        // Implementation
    }

    /// Calculate model accuracy
    pub async fn calculate_model_accuracy(
        &self,
        run_id: &str,
    ) -> TelemetryResult<f64> {
        // Implementation with SQL aggregation
    }
}
```

#### 1.4 A/B Test Results

```rust
impl TelemetryStorage {
    /// Create A/B test
    pub async fn create_ab_test(
        &self,
        test: &ABTest,
    ) -> TelemetryResult<String> {
        // Implementation
    }

    /// Record A/B test result
    pub async fn record_ab_result(
        &self,
        test_id: &str,
        variant: &str,
        success: bool,
    ) -> TelemetryResult<()> {
        // Implementation
    }

    /// Get A/B test results
    pub async fn get_ab_test_results(
        &self,
        test_id: &str,
    ) -> TelemetryResult<ABTestResults> {
        // Implementation with statistical analysis
    }

    /// Finalize A/B test
    pub async fn finalize_ab_test(
        &self,
        test_id: &str,
        winner: &str,
        confidence: f64,
    ) -> TelemetryResult<()> {
        // Implementation
    }
}
```

### Phase 2: Advanced Queries (Week 1-2)

#### 2.1 Training Data Export

```rust
impl TelemetryStorage {
    /// Export training data for ML framework
    pub async fn export_training_data(
        &self,
        time_range: &TimeRange,
        format: ExportFormat,
    ) -> TelemetryResult<TrainingDataExport> {
        // Implementation using v_training_data_export view
    }

    /// Get training samples with labels
    pub async fn get_labeled_samples(
        &self,
        limit: Option<u32>,
    ) -> TelemetryResult<Vec<LabeledSample>> {
        // Implementation
    }
}
```

#### 2.2 Model Performance Queries

```rust
impl TelemetryStorage {
    /// Get model performance metrics
    pub async fn get_model_performance(
        &self,
        model_version: &str,
    ) -> TelemetryResult<ModelPerformance> {
        // Implementation using v_model_performance view
    }

    /// Compare model versions
    pub async fn compare_models(
        &self,
        version_a: &str,
        version_b: &str,
    ) -> TelemetryResult<ModelComparison> {
        // Implementation
    }
}
```

### Phase 3: Batch Operations (Week 2)

#### 3.1 Batch Insert Framework

```rust
impl TelemetryStorage {
    /// Batch insert with transaction
    pub async fn batch_insert<T>(
        &self,
        items: &[T],
        insert_fn: impl Fn(&Connection, &T) -> Result<()>,
    ) -> TelemetryResult<()> {
        // Generic batch insert with transaction
    }

    /// Batch update aggregates
    pub async fn batch_update_aggregates(
        &self,
        updates: &[AggregateUpdate],
    ) -> TelemetryResult<()> {
        // Implementation
    }
}
```

### Phase 4: Migration System (Week 2)

#### 4.1 Migration Framework

```rust
pub struct Migration {
    version: u32,
    description: String,
    up: Box<dyn Fn(&Connection) -> Result<()>>,
    down: Option<Box<dyn Fn(&Connection) -> Result<()>>>,
}

impl TelemetryStorage {
    /// Apply migration
    pub async fn migrate(
        &self,
        migration: &Migration,
    ) -> TelemetryResult<()> {
        // Implementation with transaction
    }

    /// Rollback migration
    pub async fn rollback(
        &self,
        version: u32,
    ) -> TelemetryResult<()> {
        // Implementation
    }

    /// Get current schema version
    pub async fn get_schema_version(&self) -> TelemetryResult<u32> {
        // Implementation
    }
}
```

### Phase 5: Transaction Management (Week 2)

#### 5.1 Transaction Support

```rust
impl TelemetryStorage {
    /// Execute in transaction
    pub async fn transaction<F, T>(
        &self,
        f: F,
    ) -> TelemetryResult<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        // Implementation with proper error handling
    }

    /// Create savepoint
    pub async fn savepoint<F, T>(
        &self,
        name: &str,
        f: F,
    ) -> TelemetryResult<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        // Implementation with rollback on error
    }
}
```

---

## Implementation Checklist

### Core Operations

- [ ] **Model Training Runs:**
  - [ ] `insert_training_run()`
  - [ ] `get_training_run()`
  - [ ] `list_training_runs()`
  - [ ] `update_training_run_results()`

- [ ] **Feature Vectors:**
  - [ ] `insert_feature_vector()`
  - [ ] `get_feature_vectors_for_query()`
  - [ ] `batch_insert_feature_vectors()`
  - [ ] `find_similar_queries()`

- [ ] **Model Predictions:**
  - [ ] `insert_prediction()`
  - [ ] `update_prediction_ground_truth()`
  - [ ] `get_predictions_for_run()`
  - [ ] `calculate_model_accuracy()`

- [ ] **A/B Tests:**
  - [ ] `create_ab_test()`
  - [ ] `record_ab_result()`
  - [ ] `get_ab_test_results()`
  - [ ] `finalize_ab_test()`

### Advanced Features

- [ ] **Training Data Export:**
  - [ ] `export_training_data()`
  - [ ] `get_labeled_samples()`
  - [ ] Support for CSV/JSON/Parquet formats

- [ ] **Model Performance:**
  - [ ] `get_model_performance()`
  - [ ] `compare_models()`
  - [ ] Statistical significance calculation

- [ ] **Batch Operations:**
  - [ ] Generic `batch_insert()` framework
  - [ ] `batch_update_aggregates()`
  - [ ] Performance optimization (bulk inserts)

- [ ] **Migration System:**
  - [ ] Migration framework
  - [ ] Version tracking
  - [ ] Rollback support

- [ ] **Transaction Management:**
  - [ ] `transaction()` wrapper
  - [ ] `savepoint()` support
  - [ ] Deadlock retry logic

---

## Performance Targets

| Operation                  | Target  | Measurement |
| -------------------------- | ------- | ----------- |
| Single insert              | < 1ms   | p99 latency |
| Batch insert (100 items)   | < 50ms  | Total time  |
| Training data export (10K) | < 1s    | Total time  |
| Similarity search          | < 100ms | p99 latency |
| Model accuracy calc        | < 500ms | Total time  |

---

## Testing Strategy

### Unit Tests

- [ ] Test all CRUD operations
- [ ] Test batch operations
- [ ] Test transaction rollback
- [ ] Test migration system

### Integration Tests

- [ ] Test with real SQLite database
- [ ] Test with concurrent access
- [ ] Test with large datasets (10K+ rows)
- [ ] Test migration scenarios

### Performance Tests

- [ ] Benchmark insert operations
- [ ] Benchmark query operations
- [ ] Benchmark batch operations
- [ ] Benchmark training data export

---

## Code Quality Requirements

1. **Error Handling:**
   - Use `thiserror` for error types
   - Provide actionable error messages
   - Handle SQLite-specific errors (locked, busy)

2. **Documentation:**
   - Document all public methods
   - Include usage examples
   - Document performance characteristics

3. **Testing:**
   - 100% coverage for critical paths
   - Integration tests for all operations
   - Performance benchmarks

4. **Safety:**
   - No `unsafe` blocks without justification
   - Proper transaction handling
   - Deadlock prevention

---

## Related Documents

- [TASK_47_ML_SCHEMA_PLAN.md](TASK_47_ML_SCHEMA_PLAN.md) - Schema design
- [ADR-002: SQLite for Audit Trail](../adr/ADR-002-sqlite-for-audit-trail.md) - Architecture decision
- [Telemetry Storage](../src/telemetry/storage.rs) - Current implementation

---

**Status:** ✅ **PLANNING COMPLETE**  
**Next Action:** Begin Phase 1 (ML Table Operations)  
**Owner:** RAG/DB Team  
**Updated:** 2025-12-29
