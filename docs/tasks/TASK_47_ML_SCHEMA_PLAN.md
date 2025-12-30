# Task 47: ML Schema Design & Validation

> **Status:** IN PROGRESS  
> **Assigned Models:** Gemini 3.0 Pro + GPT-5.2  
> **Date:** 2025-12-29  
> **Related:** [PARALLEL_AGENT_ORCHESTRATION_SYSTEM.md](../../../archive/misc/PARALLEL_AGENT_ORCHESTRATION_SYSTEM.md)

---

## Executive Summary

Design and validate the ML schema for RALL (ReasonKit Adaptive Learning Loop) telemetry system. The schema must support:

- Privacy-first data collection (PII stripping, differential privacy)
- ML training data aggregation
- Pattern detection and clustering
- Query classification and routing optimization

**Current State:** Schema exists in `src/telemetry/schema.rs` (v1) - needs validation and enhancement.

---

## Objectives

1. **Validate Existing Schema:** Review current RALL telemetry schema for ML suitability
2. **Enhance for ML:** Add fields/tables needed for ML training and pattern detection
3. **Privacy Compliance:** Ensure schema supports GDPR-compliant data collection
4. **Performance Optimization:** Optimize schema for fast aggregation queries
5. **Documentation:** Create comprehensive schema documentation

---

## Current Schema Analysis

### Existing Tables (from `src/telemetry/schema.rs`)

| Table              | Purpose                     | ML Suitability | Status   |
| ------------------ | --------------------------- | -------------- | -------- |
| `sessions`         | Session tracking            | ✅ Good        | Complete |
| `queries`          | Query events (PII-stripped) | ✅ Good        | Complete |
| `feedback`         | User feedback               | ✅ Good        | Complete |
| `tool_usage`       | Tool invocation tracking    | ✅ Good        | Complete |
| `reasoning_traces` | ThinkTool execution traces  | ✅ Good        | Complete |
| `daily_aggregates` | Pre-computed daily stats    | ✅ Good        | Complete |
| `query_clusters`   | K-means clustering results  | ✅ Good        | Complete |
| `privacy_consent`  | Consent tracking            | ✅ Good        | Complete |
| `redaction_log`    | Redaction audit trail       | ✅ Good        | Complete |

### Schema Strengths

1. **Privacy-First Design:**
   - No raw query text (only hashes)
   - PII stripping built-in
   - Consent tracking
   - Redaction logging

2. **ML-Ready Features:**
   - Query clustering support
   - Daily aggregates for training
   - Tool usage metrics
   - Feedback summary

3. **Performance:**
   - Proper indexes on common queries
   - Pre-computed aggregates
   - Efficient foreign keys

### Schema Gaps for ML

1. **Missing ML-Specific Tables:**
   - [ ] Model training runs (track model versions, hyperparameters)
   - [ ] Feature vectors (store embeddings for similarity search)
   - [ ] Prediction logs (track model predictions vs. actuals)
   - [ ] A/B test results (compare model versions)

2. **Missing Fields:**
   - [ ] Query embeddings (for semantic clustering)
   - [ ] Feature extraction metadata (which features were used)
   - [ ] Model confidence scores (for calibration)
   - [ ] Training data snapshots (for reproducibility)

3. **Missing Views:**
   - [ ] Training data export view (formatted for ML frameworks)
   - [ ] Feature importance view (which features matter most)
   - [ ] Model performance view (accuracy, precision, recall)

---

## Enhanced Schema Design

### New Tables for ML

#### 1. `model_training_runs`

```sql
CREATE TABLE IF NOT EXISTS model_training_runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,

    -- Model metadata
    model_type TEXT NOT NULL,              -- classifier/routing/optimization
    model_version TEXT NOT NULL,            -- Semantic version
    hyperparameters TEXT,                  -- JSON: learning_rate, batch_size, etc.

    -- Training data
    training_samples INTEGER NOT NULL,
    validation_samples INTEGER,
    test_samples INTEGER,

    -- Performance metrics
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    training_loss REAL,
    validation_loss REAL,

    -- Metadata
    training_duration_ms INTEGER,
    feature_set_version TEXT,               -- Which feature extraction version
    notes TEXT
);
```

#### 2. `feature_vectors`

```sql
CREATE TABLE IF NOT EXISTS feature_vectors (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,                 -- FK to queries
    computed_at TEXT NOT NULL,

    -- Feature vector (stored as JSON array for flexibility)
    embedding TEXT NOT NULL,                -- JSON array of floats
    embedding_model TEXT,                   -- Which model generated it
    embedding_dim INTEGER,                  -- Dimension count

    -- Extracted features (for interpretability)
    features TEXT,                          -- JSON: query_length, tool_count, etc.
    feature_version TEXT,                   -- Feature extraction schema version

    FOREIGN KEY (query_id) REFERENCES queries(id)
);

CREATE INDEX IF NOT EXISTS idx_feature_vectors_query ON feature_vectors(query_id);
CREATE INDEX IF NOT EXISTS idx_feature_vectors_model ON feature_vectors(embedding_model);
```

#### 3. `model_predictions`

```sql
CREATE TABLE IF NOT EXISTS model_predictions (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,                 -- FK to queries
    model_run_id TEXT NOT NULL,             -- FK to model_training_runs
    predicted_at TEXT NOT NULL,

    -- Prediction data
    prediction TEXT NOT NULL,               -- Predicted value (JSON)
    confidence REAL,                        -- Model confidence (0.0-1.0)
    prediction_type TEXT NOT NULL,         -- classification/routing/optimization

    -- Ground truth (if available)
    actual_value TEXT,                     -- Actual outcome (for training)
    is_correct INTEGER,                    -- Boolean: prediction == actual

    FOREIGN KEY (query_id) REFERENCES queries(id),
    FOREIGN KEY (model_run_id) REFERENCES model_training_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_query ON model_predictions(query_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_correct ON model_predictions(is_correct);
```

#### 4. `ab_test_results`

```sql
CREATE TABLE IF NOT EXISTS ab_test_results (
    id TEXT PRIMARY KEY,
    test_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,

    -- Test configuration
    variant_a_model TEXT NOT NULL,         -- Model version A
    variant_b_model TEXT NOT NULL,         -- Model version B
    traffic_split REAL NOT NULL,           -- 0.5 = 50/50 split

    -- Results
    variant_a_samples INTEGER DEFAULT 0,
    variant_b_samples INTEGER DEFAULT 0,
    variant_a_success_rate REAL,
    variant_b_success_rate REAL,
    statistical_significance REAL,         -- p-value

    -- Decision
    winner TEXT,                           -- 'A', 'B', or 'tie'
    confidence REAL,                       -- Statistical confidence

    -- Metadata
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_ab_tests_name ON ab_test_results(test_name);
```

### Enhanced Views for ML

#### 1. `v_training_data_export`

```sql
CREATE VIEW IF NOT EXISTS v_training_data_export AS
SELECT
    q.id as query_id,
    q.query_hash,
    q.query_type,
    q.query_length,
    q.query_token_count,
    q.latency_ms,
    q.tool_calls,
    q.result_quality_score,
    q.error_occurred,
    fv.embedding,
    fv.features,
    f.feedback_type,
    f.rating,
    CASE WHEN f.feedback_type = 'thumbs_up' THEN 1 ELSE 0 END as positive_label
FROM queries q
LEFT JOIN feature_vectors fv ON q.id = fv.query_id
LEFT JOIN feedback f ON q.id = f.query_id
WHERE q.timestamp > datetime('now', '-90 days')  -- Last 90 days
  AND fv.embedding IS NOT NULL;                   -- Only queries with embeddings
```

#### 2. `v_model_performance`

```sql
CREATE VIEW IF NOT EXISTS v_model_performance AS
SELECT
    mtr.model_type,
    mtr.model_version,
    mtr.started_at,
    COUNT(mp.id) as prediction_count,
    SUM(CASE WHEN mp.is_correct = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(mp.id) as accuracy,
    AVG(mp.confidence) as avg_confidence,
    mtr.accuracy as training_accuracy,
    mtr.f1_score as training_f1
FROM model_training_runs mtr
LEFT JOIN model_predictions mp ON mtr.id = mp.model_run_id
GROUP BY mtr.id
ORDER BY mtr.started_at DESC;
```

#### 3. `v_feature_importance`

```sql
CREATE VIEW IF NOT EXISTS v_feature_importance AS
SELECT
    json_extract(fv.features, '$.query_length') as query_length,
    json_extract(fv.features, '$.tool_count') as tool_count,
    json_extract(fv.features, '$.profile') as profile,
    AVG(q.result_quality_score) as avg_quality,
    COUNT(*) as sample_count
FROM feature_vectors fv
JOIN queries q ON fv.query_id = q.id
WHERE fv.features IS NOT NULL
GROUP BY query_length, tool_count, profile
ORDER BY sample_count DESC;
```

---

## Validation Checklist

### Schema Validation

- [ ] **Privacy Compliance:**
  - [ ] No PII in any table
  - [ ] All sensitive data hashed
  - [ ] Consent tracking complete
  - [ ] Redaction logging functional

- [ ] **ML Suitability:**
  - [ ] Feature vectors can be extracted
  - [ ] Training data can be exported
  - [ ] Model predictions can be tracked
  - [ ] A/B tests can be conducted

- [ ] **Performance:**
  - [ ] Indexes on all foreign keys
  - [ ] Indexes on timestamp columns
  - [ ] Aggregation queries optimized
  - [ ] Views are efficient

- [ ] **Data Quality:**
  - [ ] Foreign key constraints enforced
  - [ ] Check constraints on numeric ranges
  - [ ] NOT NULL constraints where appropriate
  - [ ] Default values for optional fields

### ML Integration Validation

- [ ] **Feature Extraction:**
  - [ ] Query embeddings can be computed
  - [ ] Feature vectors can be stored
  - [ ] Feature versioning works

- [ ] **Model Training:**
  - [ ] Training data can be exported
  - [ ] Model runs can be tracked
  - [ ] Hyperparameters can be stored

- [ ] **Model Deployment:**
  - [ ] Predictions can be logged
  - [ ] Ground truth can be recorded
  - [ ] Performance can be measured

- [ ] **A/B Testing:**
  - [ ] Variants can be tracked
  - [ ] Results can be compared
  - [ ] Statistical significance can be computed

---

## Implementation Plan

### Phase 1: Schema Enhancement (Week 1)

1. **Add ML Tables:**
   - [ ] Create `model_training_runs` table
   - [ ] Create `feature_vectors` table
   - [ ] Create `model_predictions` table
   - [ ] Create `ab_test_results` table

2. **Add Indexes:**
   - [ ] Index foreign keys
   - [ ] Index timestamp columns
   - [ ] Index model version columns

3. **Add Views:**
   - [ ] Create `v_training_data_export` view
   - [ ] Create `v_model_performance` view
   - [ ] Create `v_feature_importance` view

### Phase 2: Validation (Week 2)

1. **Schema Validation:**
   - [ ] Run SQLite schema validation
   - [ ] Test foreign key constraints
   - [ ] Test index performance
   - [ ] Test view queries

2. **Privacy Validation:**
   - [ ] Verify no PII leakage
   - [ ] Test redaction logging
   - [ ] Test consent tracking

3. **ML Validation:**
   - [ ] Test feature vector storage
   - [ ] Test training data export
   - [ ] Test prediction logging

### Phase 3: Documentation (Week 2)

1. **Schema Documentation:**
   - [ ] Document all tables
   - [ ] Document all views
   - [ ] Document indexes
   - [ ] Create ER diagram

2. **ML Integration Guide:**
   - [ ] How to extract features
   - [ ] How to train models
   - [ ] How to log predictions
   - [ ] How to run A/B tests

---

## Model Assignment

### Gemini 3.0 Pro (Primary)

**Responsibilities:**

- High-level ML architecture design
- Schema validation and enhancement recommendations
- Privacy compliance review
- Performance optimization suggestions

**Deliverables:**

- ML schema enhancement proposal
- Privacy compliance audit
- Performance optimization plan

### GPT-5.2 (Secondary)

**Responsibilities:**

- Structured data reasoning
- SQL query optimization
- Index design recommendations
- View design for ML workflows

**Deliverables:**

- Optimized SQL queries
- Index recommendations
- View definitions

---

## Success Criteria

1. **Schema Completeness:**
   - All ML tables created
   - All views functional
   - All indexes optimized

2. **Privacy Compliance:**
   - Zero PII in schema
   - Consent tracking complete
   - Redaction logging functional

3. **ML Readiness:**
   - Feature vectors can be stored
   - Training data can be exported
   - Model predictions can be tracked

4. **Performance:**
   - Aggregation queries < 100ms
   - Training data export < 1s for 10K samples
   - Indexes reduce query time by 10x+

---

## Related Documents

- [ADR-002: SQLite for Audit Trail](../adr/ADR-002-sqlite-for-audit-trail.md)
- [Telemetry Schema](../src/telemetry/schema.rs)
- [PARALLEL_AGENT_ORCHESTRATION_SYSTEM.md](../../../archive/misc/PARALLEL_AGENT_ORCHESTRATION_SYSTEM.md)

---

**Status:** ✅ **PLANNING COMPLETE**  
**Next Action:** Begin Phase 1 (Schema Enhancement)  
**Owner:** ML Integration Team  
**Updated:** 2025-12-29
