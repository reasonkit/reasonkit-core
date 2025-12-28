//! Fast metrics calculation

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Metrics for a single step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    pub step_id: String,
    pub step_number: u32,
    pub duration_seconds: f64,
    pub confidence: f64,
    pub validation_passed: bool,
    pub retry_count: u32,
}

/// Metrics for a complete session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub session_id: String,
    pub total_steps: u32,
    pub completed_steps: u32,
    pub final_confidence: f64,
    pub total_duration_seconds: f64,
    pub validation_pass_rate: f64,
    pub total_retries: u32,
}

impl SessionMetrics {
    /// Calculate from step metrics
    pub fn from_steps(session_id: String, steps: &[StepMetrics]) -> Self {
        let total_steps = steps.len() as u32;
        let completed_steps = steps.iter().filter(|s| s.validation_passed).count() as u32;
        let total_duration: f64 = steps.iter().map(|s| s.duration_seconds).sum();
        let total_retries: u32 = steps.iter().map(|s| s.retry_count).sum();

        let validation_pass_rate = if total_steps > 0 {
            completed_steps as f64 / total_steps as f64
        } else {
            0.0
        };

        let final_confidence = calculate_weighted_confidence(
            &steps.iter().map(|s| s.confidence).collect::<Vec<_>>(),
        );

        Self {
            session_id,
            total_steps,
            completed_steps,
            final_confidence,
            total_duration_seconds: total_duration,
            validation_pass_rate,
            total_retries,
        }
    }
}

/// Calculate weighted confidence (later steps weighted more)
pub fn calculate_weighted_confidence(confidences: &[f64]) -> f64 {
    if confidences.is_empty() {
        return 0.0;
    }

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for (i, &conf) in confidences.iter().enumerate() {
        // Weight increases with step number
        let weight = 1.0 + (i as f64 * 0.1);
        weighted_sum += conf * weight;
        total_weight += weight;
    }

    weighted_sum / total_weight
}

/// Calculate simple average confidence
pub fn calculate_average_confidence(confidences: &[f64]) -> f64 {
    if confidences.is_empty() {
        return 0.0;
    }
    confidences.iter().sum::<f64>() / confidences.len() as f64
}

/// Calculate confidence percentiles
pub fn calculate_percentiles(mut values: Vec<f64>) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = values.len();

    let p25 = values[len / 4];
    let p50 = values[len / 2];
    let p75 = values[len * 3 / 4];

    (p25, p50, p75)
}

/// Aggregate metrics across multiple sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub total_sessions: u32,
    pub avg_confidence: f64,
    pub avg_validation_rate: f64,
    pub avg_duration: f64,
    pub total_steps_completed: u32,
    pub completion_rate: f64,
}

impl AggregateMetrics {
    pub fn from_sessions(sessions: &[SessionMetrics]) -> Self {
        let total_sessions = sessions.len() as u32;

        if total_sessions == 0 {
            return Self {
                total_sessions: 0,
                avg_confidence: 0.0,
                avg_validation_rate: 0.0,
                avg_duration: 0.0,
                total_steps_completed: 0,
                completion_rate: 0.0,
            };
        }

        let avg_confidence =
            sessions.iter().map(|s| s.final_confidence).sum::<f64>() / total_sessions as f64;
        let avg_validation_rate =
            sessions.iter().map(|s| s.validation_pass_rate).sum::<f64>() / total_sessions as f64;
        let avg_duration =
            sessions.iter().map(|s| s.total_duration_seconds).sum::<f64>() / total_sessions as f64;
        let total_steps_completed: u32 = sessions.iter().map(|s| s.completed_steps).sum();

        let completed_sessions = sessions
            .iter()
            .filter(|s| s.completed_steps == s.total_steps)
            .count();
        let completion_rate = completed_sessions as f64 / total_sessions as f64;

        Self {
            total_sessions,
            avg_confidence,
            avg_validation_rate,
            avg_duration,
            total_steps_completed,
            completion_rate,
        }
    }
}

/// Python-exposed SessionMetrics
#[pyclass(name = "SessionMetrics")]
pub struct PySessionMetrics {
    inner: SessionMetrics,
}

#[pymethods]
impl PySessionMetrics {
    #[new]
    fn new(
        session_id: String,
        total_steps: u32,
        completed_steps: u32,
        final_confidence: f64,
        total_duration_seconds: f64,
        validation_pass_rate: f64,
        total_retries: u32,
    ) -> Self {
        Self {
            inner: SessionMetrics {
                session_id,
                total_steps,
                completed_steps,
                final_confidence,
                total_duration_seconds,
                validation_pass_rate,
                total_retries,
            },
        }
    }

    #[getter]
    fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    #[getter]
    fn total_steps(&self) -> u32 {
        self.inner.total_steps
    }

    #[getter]
    fn final_confidence(&self) -> f64 {
        self.inner.final_confidence
    }

    #[getter]
    fn completion_rate(&self) -> f64 {
        if self.inner.total_steps == 0 {
            0.0
        } else {
            self.inner.completed_steps as f64 / self.inner.total_steps as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_confidence() {
        let confidences = vec![0.7, 0.8, 0.9];
        let result = calculate_weighted_confidence(&confidences);

        // Later steps weighted more, so should be > simple average (0.8)
        assert!(result > 0.8);
        assert!(result < 0.9);
    }

    #[test]
    fn test_aggregate_metrics() {
        let sessions = vec![
            SessionMetrics {
                session_id: "1".to_string(),
                total_steps: 5,
                completed_steps: 5,
                final_confidence: 0.8,
                total_duration_seconds: 10.0,
                validation_pass_rate: 1.0,
                total_retries: 0,
            },
            SessionMetrics {
                session_id: "2".to_string(),
                total_steps: 5,
                completed_steps: 4,
                final_confidence: 0.7,
                total_duration_seconds: 15.0,
                validation_pass_rate: 0.8,
                total_retries: 1,
            },
        ];

        let agg = AggregateMetrics::from_sessions(&sessions);
        assert_eq!(agg.total_sessions, 2);
        assert!((agg.avg_confidence - 0.75).abs() < 0.01);
        assert_eq!(agg.completion_rate, 0.5); // 1 of 2 completed all steps
    }
}
