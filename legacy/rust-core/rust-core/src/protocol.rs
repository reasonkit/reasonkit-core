//! Protocol definitions and structures

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cognitive stance for each step
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveStance {
    BoundarySetting,
    RealityCheck,
    QualityGate,
    Reductionist,
    Creative,
    Empirical,
    Analytical,
    Integrative,
    Critical,
    Communication,
}

impl CognitiveStance {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::BoundarySetting => "boundary_setting",
            Self::RealityCheck => "reality_check",
            Self::QualityGate => "quality_gate",
            Self::Reductionist => "reductionist",
            Self::Creative => "creative",
            Self::Empirical => "empirical",
            Self::Analytical => "analytical",
            Self::Integrative => "integrative",
            Self::Critical => "critical",
            Self::Communication => "communication",
        }
    }
}

/// Output schema for a step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSchema {
    pub fields: HashMap<String, String>,
    pub required_fields: Vec<String>,
}

/// A single reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub step: u32,
    pub id: String,
    pub name: String,
    pub instruction: String,
    pub cognitive_stance: CognitiveStance,
    pub time_allocation: String,
    pub output_schema: OutputSchema,
    pub validation_rules: Vec<String>,
    pub failure_modes: Vec<String>,
}

/// A complete reasoning protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Protocol {
    pub module: String,
    pub version: String,
    pub description: String,
    pub steps: Vec<Step>,
}

impl Protocol {
    /// Load from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Export to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Get step by ID
    pub fn get_step(&self, id: &str) -> Option<&Step> {
        self.steps.iter().find(|s| s.id == id)
    }

    /// Get step by number
    pub fn get_step_by_number(&self, num: u32) -> Option<&Step> {
        self.steps.iter().find(|s| s.step == num)
    }
}

/// Python-exposed Protocol
#[pyclass(name = "Protocol")]
pub struct PyProtocol {
    inner: Protocol,
}

#[pymethods]
impl PyProtocol {
    #[new]
    fn new(json: &str) -> PyResult<Self> {
        let inner = Protocol::from_json(json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn module(&self) -> &str {
        &self.inner.module
    }

    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }

    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    #[getter]
    fn step_count(&self) -> usize {
        self.inner.steps.len()
    }

    fn get_step(&self, id: &str) -> Option<PyStep> {
        self.inner.get_step(id).map(|s| PyStep { inner: s.clone() })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

/// Python-exposed Step
#[pyclass(name = "Step")]
pub struct PyStep {
    inner: Step,
}

#[pymethods]
impl PyStep {
    #[getter]
    fn step(&self) -> u32 {
        self.inner.step
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn instruction(&self) -> &str {
        &self.inner.instruction
    }

    #[getter]
    fn cognitive_stance(&self) -> &str {
        self.inner.cognitive_stance.as_str()
    }

    #[getter]
    fn required_fields(&self) -> Vec<String> {
        self.inner.output_schema.required_fields.clone()
    }

    #[getter]
    fn validation_rules(&self) -> Vec<String> {
        self.inner.validation_rules.clone()
    }
}
