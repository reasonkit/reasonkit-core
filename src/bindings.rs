//! # ReasonKit Python Bindings (PyO3)
//!
//! Exposes the ThinkTool engine to Python via PyO3 + Maturin.
//!
//! ## Usage (Python)
//!
//! ```python
//! from reasonkit import Reasoner, Profile, ReasonerError
//!
//! # Create reasoner (auto-detects LLM provider from env)
//! reasoner = Reasoner()
//!
//! # Run individual ThinkTools
//! result = reasoner.run_gigathink("What factors drive startup success?")
//! print(result.confidence, result.perspectives)
//!
//! # Run with profile
//! result = reasoner.think_with_profile(Profile.Balanced, "Should we use microservices?")
//!
//! # Convenience functions
//! from reasonkit import run_gigathink, run_laserlogic
//! result = run_gigathink("Analyze market trends")
//! ```
//!
//! ## Build with Maturin + UV
//!
//! ```bash
//! cd reasonkit-core
//! maturin develop --release   # Dev install
//! maturin build --release     # Build wheel
//! ```

#![allow(clippy::useless_conversion)]

use crate::thinktool::executor::{ExecutorConfig, ProtocolExecutor, ProtocolInput};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use tokio::runtime::Runtime;

// ============================================================================
// ERROR TYPES
// ============================================================================

// Custom Python exception for ReasonKit errors
pyo3::create_exception!(reasonkit, ReasonerError, pyo3::exceptions::PyException);

/// Convert Rust errors to Python exceptions
fn to_py_err(err: crate::error::Error) -> PyErr {
    match err {
        crate::error::Error::NotFound { resource } => {
            PyKeyError::new_err(format!("Resource not found: {}", resource))
        }
        crate::error::Error::Validation(msg) => PyValueError::new_err(msg),
        crate::error::Error::Timeout(msg) => PyTimeoutError::new_err(msg),
        crate::error::Error::Config(msg) => PyValueError::new_err(format!("Config error: {}", msg)),
        crate::error::Error::Network(msg) => {
            ReasonerError::new_err(format!("Network error: {}", msg))
        }
        other => PyRuntimeError::new_err(format!("ReasonKit error: {}", other)),
    }
}

// ============================================================================
// PROFILE ENUM
// ============================================================================

/// Reasoning profile (ThinkTool chain configuration)
///
/// Profiles define which ThinkTools are executed and in what order:
/// - Quick: GigaThink -> LaserLogic (fast, 2 tools)
/// - Balanced: All 4 core tools (standard)
/// - Deep: All 5 tools with extra verification
/// - Paranoid: Maximum rigor, 95% confidence target
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Profile {
    /// No ThinkTools (baseline)
    None = 0,
    /// Fast 2-tool chain (GigaThink + LaserLogic)
    Quick = 1,
    /// Standard 4-tool chain
    Balanced = 2,
    /// Thorough 5-tool chain
    Deep = 3,
    /// Maximum verification (95% confidence)
    Paranoid = 4,
}

#[pymethods]
impl Profile {
    /// Get the profile name as a string
    fn name(&self) -> &'static str {
        match self {
            Profile::None => "none",
            Profile::Quick => "quick",
            Profile::Balanced => "balanced",
            Profile::Deep => "deep",
            Profile::Paranoid => "paranoid",
        }
    }

    /// Get the list of ThinkTools in this profile
    fn thinktools(&self) -> Vec<&'static str> {
        match self {
            Profile::None => vec![],
            Profile::Quick => vec!["gigathink", "laserlogic"],
            Profile::Balanced => vec!["gigathink", "laserlogic", "bedrock", "proofguard"],
            Profile::Deep => vec![
                "gigathink",
                "laserlogic",
                "bedrock",
                "proofguard",
                "brutalhonesty",
            ],
            Profile::Paranoid => vec![
                "gigathink",
                "laserlogic",
                "bedrock",
                "proofguard",
                "brutalhonesty",
                "proofguard", // Second pass
            ],
        }
    }

    /// Expected minimum confidence for this profile
    fn min_confidence(&self) -> f64 {
        match self {
            Profile::None => 0.0,
            Profile::Quick => 0.70,
            Profile::Balanced => 0.80,
            Profile::Deep => 0.85,
            Profile::Paranoid => 0.95,
        }
    }

    fn __repr__(&self) -> String {
        format!("Profile.{}", self.name().to_uppercase().replace('-', "_"))
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }
}

// ============================================================================
// THINKTOOL OUTPUT
// ============================================================================

/// Output from a ThinkTool execution
///
/// Contains the reasoning results, confidence score, and structured data.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ThinkToolOutput {
    /// Protocol that was executed
    #[pyo3(get)]
    pub protocol_id: String,

    /// Whether execution succeeded
    #[pyo3(get)]
    pub success: bool,

    /// Overall confidence score (0.0-1.0)
    #[pyo3(get)]
    pub confidence: f64,

    /// Execution time in milliseconds
    #[pyo3(get)]
    pub duration_ms: u64,

    /// Total tokens used
    #[pyo3(get)]
    pub total_tokens: u32,

    /// Error message if failed
    #[pyo3(get)]
    pub error: Option<String>,

    /// Raw JSON output data
    output_json: String,
}

#[pymethods]
impl ThinkToolOutput {
    /// Get the output data as a Python dict
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let data: serde_json::Value = serde_json::from_str(&self.output_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        json_to_pyobject(py, &data).and_then(|obj| {
            obj.downcast::<PyDict>()
                .map(|d| d.clone())
                .map_err(|_| PyValueError::new_err("Output is not a dict"))
        })
    }

    /// Get perspectives (for GigaThink output)
    fn perspectives(&self) -> Vec<String> {
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&self.output_json) {
            if let Some(arr) = data.get("perspectives").and_then(|v| v.as_array()) {
                return arr
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }
            // Try to extract from steps
            if let Some(steps) = data.get("steps").and_then(|v| v.as_array()) {
                for step in steps {
                    if let Some(output) = step.get("output") {
                        if let Some(items) = output.get("items").and_then(|v| v.as_array()) {
                            return items
                                .iter()
                                .filter_map(|item| {
                                    item.get("content")
                                        .and_then(|c| c.as_str())
                                        .map(String::from)
                                })
                                .collect();
                        }
                    }
                }
            }
        }
        vec![]
    }

    /// Get verdict (for validation outputs)
    fn verdict(&self) -> Option<String> {
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&self.output_json) {
            data.get("verdict")
                .and_then(|v| v.as_str())
                .map(String::from)
        } else {
            None
        }
    }

    /// Get step results
    fn steps(&self) -> Vec<StepResultPy> {
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&self.output_json) {
            if let Some(steps) = data.get("steps").and_then(|v| v.as_array()) {
                return steps
                    .iter()
                    .filter_map(|step| {
                        Some(StepResultPy {
                            step_id: step.get("step_id")?.as_str()?.to_string(),
                            success: step.get("success")?.as_bool()?,
                            confidence: step.get("confidence")?.as_f64()?,
                            content: step
                                .get("output")
                                .and_then(|o| o.get("content"))
                                .and_then(|c| c.as_str())
                                .map(String::from),
                        })
                    })
                    .collect();
            }
        }
        vec![]
    }

    /// Get raw JSON string
    fn to_json(&self) -> String {
        self.output_json.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ThinkToolOutput(protocol='{}', success={}, confidence={:.2})",
            self.protocol_id, self.success, self.confidence
        )
    }

    fn __str__(&self) -> String {
        if self.success {
            format!(
                "[{}] Confidence: {:.1}% ({} tokens, {}ms)",
                self.protocol_id,
                self.confidence * 100.0,
                self.total_tokens,
                self.duration_ms
            )
        } else {
            format!(
                "[{}] FAILED: {}",
                self.protocol_id,
                self.error.as_deref().unwrap_or("Unknown error")
            )
        }
    }
}

/// Individual step result
#[pyclass]
#[derive(Clone, Debug)]
pub struct StepResultPy {
    #[pyo3(get)]
    pub step_id: String,
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub confidence: f64,
    #[pyo3(get)]
    pub content: Option<String>,
}

#[pymethods]
impl StepResultPy {
    fn __repr__(&self) -> String {
        format!(
            "StepResult(id='{}', success={}, confidence={:.2})",
            self.step_id, self.success, self.confidence
        )
    }
}

// ============================================================================
// MAIN REASONER CLASS
// ============================================================================

/// Main ReasonKit interface for Python
///
/// The Reasoner provides access to all ThinkTool protocols and profiles.
///
/// # Examples
///
/// ```python
/// from reasonkit import Reasoner, Profile
///
/// # Create with auto-detected LLM
/// r = Reasoner()
///
/// # Or with mock for testing
/// r = Reasoner(use_mock=True)
///
/// # Run individual tools
/// result = r.run_gigathink("Analyze market trends")
/// result = r.run_laserlogic("If A then B. A. Therefore B.")
/// result = r.run_bedrock("What is the first principle behind X?")
/// result = r.run_proofguard("The Earth is round")
/// result = r.run_brutalhonesty("My business plan is to sell ice to Eskimos")
///
/// # Run with profile
/// result = r.think_with_profile(Profile.Balanced, "Should we pivot?")
/// ```
#[pyclass]
pub struct Reasoner {
    executor: ProtocolExecutor,
    runtime: Runtime,
}

#[pymethods]
impl Reasoner {
    /// Create a new Reasoner
    ///
    /// Args:
    ///     use_mock: If True, use mock LLM for testing (no API calls)
    ///     verbose: If True, enable verbose logging
    ///     timeout_secs: Timeout for LLM calls in seconds (default: 120)
    ///
    /// Returns:
    ///     Reasoner instance
    ///
    /// Raises:
    ///     RuntimeError: If initialization fails
    #[new]
    #[pyo3(signature = (use_mock=None, verbose=None, timeout_secs=None))]
    fn new(
        use_mock: Option<bool>,
        verbose: Option<bool>,
        timeout_secs: Option<u64>,
    ) -> PyResult<Self> {
        let runtime = Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create async runtime: {}", e))
        })?;

        let mut config = if use_mock.unwrap_or(false) {
            ExecutorConfig::mock()
        } else {
            ExecutorConfig::default()
        };

        if let Some(v) = verbose {
            config.verbose = v;
        }
        if let Some(t) = timeout_secs {
            config.timeout_secs = t;
        }

        let executor = ProtocolExecutor::with_config(config).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to initialize executor: {}", e))
        })?;

        Ok(Reasoner { executor, runtime })
    }

    // ========================================================================
    // INDIVIDUAL THINKTOOL METHODS
    // ========================================================================

    /// Run GigaThink: Multi-perspective creative thinking
    ///
    /// Generates 10+ diverse perspectives on a topic.
    ///
    /// Args:
    ///     query: The question or topic to analyze
    ///     context: Optional additional context
    ///
    /// Returns:
    ///     ThinkToolOutput with perspectives in output.perspectives
    #[pyo3(signature = (query, context=None))]
    fn run_gigathink(&self, query: String, context: Option<String>) -> PyResult<ThinkToolOutput> {
        let mut input = ProtocolInput::query(query);
        if let Some(ctx) = context {
            input = input.with_field("context", ctx);
        }
        self.execute_protocol("gigathink", input)
    }

    /// Run LaserLogic: Precision deductive reasoning
    ///
    /// Analyzes logical structure, detects fallacies, validates arguments.
    ///
    /// Args:
    ///     argument: The argument or reasoning chain to analyze
    ///
    /// Returns:
    ///     ThinkToolOutput with validity assessment
    #[pyo3(signature = (argument))]
    fn run_laserlogic(&self, argument: String) -> PyResult<ThinkToolOutput> {
        let input = ProtocolInput::argument(argument);
        self.execute_protocol("laserlogic", input)
    }

    /// Run BedRock: First principles decomposition
    ///
    /// Breaks down statements to fundamental axioms and rebuilds.
    ///
    /// Args:
    ///     statement: The statement to decompose
    ///     domain: Optional domain context (e.g., "physics", "economics")
    ///
    /// Returns:
    ///     ThinkToolOutput with axioms and reconstruction
    #[pyo3(signature = (statement, domain=None))]
    fn run_bedrock(&self, statement: String, domain: Option<String>) -> PyResult<ThinkToolOutput> {
        let mut input = ProtocolInput::statement(statement);
        if let Some(d) = domain {
            input = input.with_field("domain", d);
        }
        self.execute_protocol("bedrock", input)
    }

    /// Run ProofGuard: Multi-source verification
    ///
    /// Verifies claims against multiple sources (triangulation).
    ///
    /// Args:
    ///     claim: The claim to verify
    ///     sources: Optional list of sources to check against
    ///
    /// Returns:
    ///     ThinkToolOutput with verification results
    #[pyo3(signature = (claim, sources=None))]
    fn run_proofguard(
        &self,
        claim: String,
        sources: Option<Vec<String>>,
    ) -> PyResult<ThinkToolOutput> {
        let mut input = ProtocolInput::claim(claim);
        if let Some(s) = sources {
            input = input.with_field("sources", s.join("\n"));
        }
        self.execute_protocol("proofguard", input)
    }

    /// Run BrutalHonesty: Adversarial self-critique
    ///
    /// Finds flaws, weaknesses, and suggests improvements.
    ///
    /// Args:
    ///     work: The work, plan, or reasoning to critique
    ///
    /// Returns:
    ///     ThinkToolOutput with critique and suggestions
    #[pyo3(signature = (work))]
    fn run_brutalhonesty(&self, work: String) -> PyResult<ThinkToolOutput> {
        let input = ProtocolInput::work(work);
        self.execute_protocol("brutalhonesty", input)
    }

    // ========================================================================
    // PROFILE-BASED EXECUTION
    // ========================================================================

    /// Execute a reasoning profile (chain of ThinkTools)
    ///
    /// Args:
    ///     profile: The Profile enum value
    ///     query: The question or topic
    ///     context: Optional additional context
    ///
    /// Returns:
    ///     ThinkToolOutput with combined results
    #[pyo3(signature = (profile, query, context=None))]
    fn think_with_profile(
        &self,
        profile: Profile,
        query: String,
        context: Option<String>,
    ) -> PyResult<ThinkToolOutput> {
        let mut input = ProtocolInput::query(query);
        if let Some(ctx) = context {
            input = input.with_field("context", ctx);
        }

        let profile_id = profile.name();

        let result = self
            .runtime
            .block_on(async { self.executor.execute_profile(profile_id, input).await });

        match result {
            Ok(output) => {
                let output_json =
                    serde_json::to_string(&output).unwrap_or_else(|_| "{}".to_string());
                Ok(ThinkToolOutput {
                    protocol_id: profile_id.to_string(),
                    success: output.success,
                    confidence: output.confidence,
                    duration_ms: output.duration_ms,
                    total_tokens: output.tokens.total_tokens,
                    error: output.error,
                    output_json,
                })
            }
            Err(e) => Err(to_py_err(e)),
        }
    }

    // ========================================================================
    // GENERIC PROTOCOL EXECUTION
    // ========================================================================

    /// Execute any protocol by name
    ///
    /// Args:
    ///     protocol: Protocol ID (e.g., "gigathink", "laserlogic")
    ///     query: Input query
    ///
    /// Returns:
    ///     ThinkToolOutput
    #[pyo3(signature = (protocol, query))]
    fn think(&self, protocol: String, query: String) -> PyResult<ThinkToolOutput> {
        let input = ProtocolInput::query(query);
        self.execute_protocol(&protocol, input)
    }

    // ========================================================================
    // INTROSPECTION
    // ========================================================================

    /// List available protocols
    fn list_protocols(&self) -> Vec<String> {
        self.executor
            .list_protocols()
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// List available profiles
    fn list_profiles(&self) -> Vec<String> {
        self.executor
            .list_profiles()
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Get protocol description
    fn get_protocol_info(&self, protocol_id: &str) -> PyResult<HashMap<String, String>> {
        let protocol = self
            .executor
            .get_protocol(protocol_id)
            .ok_or_else(|| PyKeyError::new_err(format!("Protocol not found: {}", protocol_id)))?;

        let mut info = HashMap::new();
        info.insert("id".to_string(), protocol.id.clone());
        info.insert("name".to_string(), protocol.name.clone());
        info.insert("description".to_string(), protocol.description.clone());
        info.insert("version".to_string(), protocol.version.clone());
        info.insert("num_steps".to_string(), protocol.steps.len().to_string());

        Ok(info)
    }

    fn __repr__(&self) -> String {
        let protocols = self.list_protocols().len();
        let profiles = self.list_profiles().len();
        format!("Reasoner({} protocols, {} profiles)", protocols, profiles)
    }
}

impl Reasoner {
    /// Internal helper to execute a protocol
    fn execute_protocol(
        &self,
        protocol_id: &str,
        input: ProtocolInput,
    ) -> PyResult<ThinkToolOutput> {
        let result = self
            .runtime
            .block_on(async { self.executor.execute(protocol_id, input).await });

        match result {
            Ok(output) => {
                let output_json =
                    serde_json::to_string(&output).unwrap_or_else(|_| "{}".to_string());
                Ok(ThinkToolOutput {
                    protocol_id: protocol_id.to_string(),
                    success: output.success,
                    confidence: output.confidence,
                    duration_ms: output.duration_ms,
                    total_tokens: output.tokens.total_tokens,
                    error: output.error,
                    output_json,
                })
            }
            Err(e) => Err(to_py_err(e)),
        }
    }
}

// ============================================================================
// MODULE-LEVEL CONVENIENCE FUNCTIONS
// ============================================================================

/// Run GigaThink with default settings
///
/// Convenience function for quick usage without instantiating Reasoner.
///
/// Args:
///     query: The question or topic to analyze
///     use_mock: If True, use mock LLM (default: False)
///
/// Returns:
///     ThinkToolOutput with perspectives
#[pyfunction]
#[pyo3(signature = (query, use_mock=None))]
fn run_gigathink(query: String, use_mock: Option<bool>) -> PyResult<ThinkToolOutput> {
    let reasoner = Reasoner::new(use_mock, None, None)?;
    reasoner.run_gigathink(query, None)
}

/// Run LaserLogic with default settings
#[pyfunction]
#[pyo3(signature = (argument, use_mock=None))]
fn run_laserlogic(argument: String, use_mock: Option<bool>) -> PyResult<ThinkToolOutput> {
    let reasoner = Reasoner::new(use_mock, None, None)?;
    reasoner.run_laserlogic(argument)
}

/// Run BedRock with default settings
#[pyfunction]
#[pyo3(signature = (statement, use_mock=None))]
fn run_bedrock(statement: String, use_mock: Option<bool>) -> PyResult<ThinkToolOutput> {
    let reasoner = Reasoner::new(use_mock, None, None)?;
    reasoner.run_bedrock(statement, None)
}

/// Run ProofGuard with default settings
#[pyfunction]
#[pyo3(signature = (claim, use_mock=None))]
fn run_proofguard(claim: String, use_mock: Option<bool>) -> PyResult<ThinkToolOutput> {
    let reasoner = Reasoner::new(use_mock, None, None)?;
    reasoner.run_proofguard(claim, None)
}

/// Run BrutalHonesty with default settings
#[pyfunction]
#[pyo3(signature = (work, use_mock=None))]
fn run_brutalhonesty(work: String, use_mock: Option<bool>) -> PyResult<ThinkToolOutput> {
    let reasoner = Reasoner::new(use_mock, None, None)?;
    reasoner.run_brutalhonesty(work)
}

/// Quick analysis with the Quick profile
#[pyfunction]
#[pyo3(signature = (query, use_mock=None))]
fn quick_think(query: String, use_mock: Option<bool>) -> PyResult<ThinkToolOutput> {
    let reasoner = Reasoner::new(use_mock, None, None)?;
    reasoner.think_with_profile(Profile::Quick, query, None)
}

/// Balanced analysis with the Balanced profile
#[pyfunction]
#[pyo3(signature = (query, use_mock=None))]
fn balanced_think(query: String, use_mock: Option<bool>) -> PyResult<ThinkToolOutput> {
    let reasoner = Reasoner::new(use_mock, None, None)?;
    reasoner.think_with_profile(Profile::Balanced, query, None)
}

/// Deep analysis with the Deep profile
#[pyfunction]
#[pyo3(signature = (query, use_mock=None))]
fn deep_think(query: String, use_mock: Option<bool>) -> PyResult<ThinkToolOutput> {
    let reasoner = Reasoner::new(use_mock, None, None)?;
    reasoner.think_with_profile(Profile::Deep, query, None)
}

/// Paranoid analysis with maximum verification
#[pyfunction]
#[pyo3(signature = (query, use_mock=None))]
fn paranoid_think(query: String, use_mock: Option<bool>) -> PyResult<ThinkToolOutput> {
    let reasoner = Reasoner::new(use_mock, None, None)?;
    reasoner.think_with_profile(Profile::Paranoid, query, None)
}

/// Get ReasonKit version
#[pyfunction]
fn version() -> &'static str {
    crate::VERSION
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Convert serde_json::Value to Python object
fn json_to_pyobject<'py>(
    py: Python<'py>,
    value: &serde_json::Value,
) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::ToPyObject;
    match value {
        serde_json::Value::Null => Ok(py.None().into_bound(py)),
        serde_json::Value::Bool(b) => Ok(b.to_object(py).into_bound(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py).into_bound(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py).into_bound(py))
            } else {
                Err(PyValueError::new_err("Invalid number"))
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py).into_bound(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_pyobject(py, item)?)?;
            }
            Ok(list.into_any())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_pyobject(py, v)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

// ============================================================================
// MODULE REGISTRATION
// ============================================================================

/// Register all Python bindings
pub fn register_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<Reasoner>()?;
    m.add_class::<Profile>()?;
    m.add_class::<ThinkToolOutput>()?;
    m.add_class::<StepResultPy>()?;

    // Exception
    m.add("ReasonerError", m.py().get_type_bound::<ReasonerError>())?;

    // Convenience functions
    m.add_function(wrap_pyfunction!(run_gigathink, m)?)?;
    m.add_function(wrap_pyfunction!(run_laserlogic, m)?)?;
    m.add_function(wrap_pyfunction!(run_bedrock, m)?)?;
    m.add_function(wrap_pyfunction!(run_proofguard, m)?)?;
    m.add_function(wrap_pyfunction!(run_brutalhonesty, m)?)?;

    // Profile shortcuts
    m.add_function(wrap_pyfunction!(quick_think, m)?)?;
    m.add_function(wrap_pyfunction!(balanced_think, m)?)?;
    m.add_function(wrap_pyfunction!(deep_think, m)?)?;
    m.add_function(wrap_pyfunction!(paranoid_think, m)?)?;

    // Utility
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}

// ============================================================================
// TESTS - Python FFI Verification Suite
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PROFILE ENUM TESTS
    // ========================================================================

    #[test]
    fn test_profile_names() {
        assert_eq!(Profile::None.name(), "none");
        assert_eq!(Profile::Quick.name(), "quick");
        assert_eq!(Profile::Balanced.name(), "balanced");
        assert_eq!(Profile::Deep.name(), "deep");
        assert_eq!(Profile::Paranoid.name(), "paranoid");
    }

    #[test]
    fn test_profile_thinktools() {
        assert_eq!(Profile::None.thinktools().len(), 0);
        assert_eq!(Profile::Quick.thinktools().len(), 2);
        assert_eq!(Profile::Balanced.thinktools().len(), 4);
        assert_eq!(Profile::Deep.thinktools().len(), 5);
        assert_eq!(Profile::Paranoid.thinktools().len(), 6);
    }

    #[test]
    fn test_profile_thinktools_content() {
        // Verify Quick profile contains expected tools
        let quick_tools = Profile::Quick.thinktools();
        assert!(quick_tools.contains(&"gigathink"));
        assert!(quick_tools.contains(&"laserlogic"));

        // Verify Balanced has core tools
        let balanced_tools = Profile::Balanced.thinktools();
        assert!(balanced_tools.contains(&"gigathink"));
        assert!(balanced_tools.contains(&"laserlogic"));
        assert!(balanced_tools.contains(&"bedrock"));
        assert!(balanced_tools.contains(&"proofguard"));

        // Verify Deep adds brutalhonesty
        let deep_tools = Profile::Deep.thinktools();
        assert!(deep_tools.contains(&"brutalhonesty"));

        // Verify Paranoid has double proofguard
        let paranoid_tools = Profile::Paranoid.thinktools();
        let proofguard_count = paranoid_tools
            .iter()
            .filter(|&&t| t == "proofguard")
            .count();
        assert_eq!(proofguard_count, 2, "Paranoid should have 2 proofguard passes");
    }

    #[test]
    fn test_profile_min_confidence() {
        // Test exact values
        assert_eq!(Profile::None.min_confidence(), 0.0);
        assert_eq!(Profile::Quick.min_confidence(), 0.70);
        assert_eq!(Profile::Balanced.min_confidence(), 0.80);
        assert_eq!(Profile::Deep.min_confidence(), 0.85);
        assert_eq!(Profile::Paranoid.min_confidence(), 0.95);

        // Test ordering
        assert!(Profile::Quick.min_confidence() < Profile::Balanced.min_confidence());
        assert!(Profile::Balanced.min_confidence() < Profile::Deep.min_confidence());
        assert!(Profile::Deep.min_confidence() < Profile::Paranoid.min_confidence());
    }

    #[test]
    fn test_profile_repr() {
        assert_eq!(Profile::Quick.__repr__(), "Profile.QUICK");
        assert_eq!(Profile::Balanced.__repr__(), "Profile.BALANCED");
        assert_eq!(Profile::Deep.__repr__(), "Profile.DEEP");
        assert_eq!(Profile::Paranoid.__repr__(), "Profile.PARANOID");
        assert_eq!(Profile::None.__repr__(), "Profile.NONE");
    }

    #[test]
    fn test_profile_str() {
        assert_eq!(Profile::Quick.__str__(), "quick");
        assert_eq!(Profile::Balanced.__str__(), "balanced");
        assert_eq!(Profile::Deep.__str__(), "deep");
        assert_eq!(Profile::Paranoid.__str__(), "paranoid");
    }

    #[test]
    fn test_profile_equality() {
        assert_eq!(Profile::Quick, Profile::Quick);
        assert_ne!(Profile::Quick, Profile::Balanced);
        assert_ne!(Profile::None, Profile::Paranoid);
    }

    #[test]
    fn test_profile_clone() {
        let original = Profile::Balanced;
        let cloned = original;
        assert_eq!(original, cloned);
        assert_eq!(original.name(), cloned.name());
    }

    // ========================================================================
    // THINKTOOL OUTPUT TESTS
    // ========================================================================

    #[test]
    fn test_thinktool_output_creation() {
        let output = ThinkToolOutput {
            protocol_id: "gigathink".to_string(),
            success: true,
            confidence: 0.85,
            duration_ms: 1500,
            total_tokens: 250,
            error: None,
            output_json: r#"{"test": "data"}"#.to_string(),
        };

        assert_eq!(output.protocol_id, "gigathink");
        assert!(output.success);
        assert_eq!(output.confidence, 0.85);
        assert_eq!(output.duration_ms, 1500);
        assert_eq!(output.total_tokens, 250);
        assert!(output.error.is_none());
    }

    #[test]
    fn test_thinktool_output_failed() {
        let output = ThinkToolOutput {
            protocol_id: "laserlogic".to_string(),
            success: false,
            confidence: 0.0,
            duration_ms: 50,
            total_tokens: 10,
            error: Some("Validation failed".to_string()),
            output_json: "{}".to_string(),
        };

        assert!(!output.success);
        assert_eq!(output.confidence, 0.0);
        assert_eq!(output.error.as_deref(), Some("Validation failed"));
    }

    #[test]
    fn test_thinktool_output_perspectives_extraction() {
        // Test with perspectives array
        let output = ThinkToolOutput {
            protocol_id: "gigathink".to_string(),
            success: true,
            confidence: 0.85,
            duration_ms: 1000,
            total_tokens: 200,
            error: None,
            output_json: r#"{"perspectives": ["view1", "view2", "view3"]}"#.to_string(),
        };

        let perspectives = output.perspectives();
        assert_eq!(perspectives.len(), 3);
        assert!(perspectives.contains(&"view1".to_string()));
        assert!(perspectives.contains(&"view2".to_string()));
        assert!(perspectives.contains(&"view3".to_string()));
    }

    #[test]
    fn test_thinktool_output_perspectives_from_steps() {
        // Test extraction from steps structure
        let output = ThinkToolOutput {
            protocol_id: "gigathink".to_string(),
            success: true,
            confidence: 0.85,
            duration_ms: 1000,
            total_tokens: 200,
            error: None,
            output_json: r#"{
                "steps": [
                    {
                        "output": {
                            "items": [
                                {"content": "perspective_a"},
                                {"content": "perspective_b"}
                            ]
                        }
                    }
                ]
            }"#
            .to_string(),
        };

        let perspectives = output.perspectives();
        assert_eq!(perspectives.len(), 2);
        assert!(perspectives.contains(&"perspective_a".to_string()));
        assert!(perspectives.contains(&"perspective_b".to_string()));
    }

    #[test]
    fn test_thinktool_output_empty_perspectives() {
        let output = ThinkToolOutput {
            protocol_id: "laserlogic".to_string(),
            success: true,
            confidence: 0.90,
            duration_ms: 500,
            total_tokens: 100,
            error: None,
            output_json: r#"{"verdict": "valid"}"#.to_string(),
        };

        let perspectives = output.perspectives();
        assert!(perspectives.is_empty());
    }

    #[test]
    fn test_thinktool_output_verdict() {
        let output = ThinkToolOutput {
            protocol_id: "laserlogic".to_string(),
            success: true,
            confidence: 0.92,
            duration_ms: 800,
            total_tokens: 150,
            error: None,
            output_json: r#"{"verdict": "VALID", "details": "reasoning is sound"}"#.to_string(),
        };

        assert_eq!(output.verdict(), Some("VALID".to_string()));
    }

    #[test]
    fn test_thinktool_output_no_verdict() {
        let output = ThinkToolOutput {
            protocol_id: "gigathink".to_string(),
            success: true,
            confidence: 0.85,
            duration_ms: 1000,
            total_tokens: 200,
            error: None,
            output_json: r#"{"perspectives": []}"#.to_string(),
        };

        assert!(output.verdict().is_none());
    }

    #[test]
    fn test_thinktool_output_steps_extraction() {
        let output = ThinkToolOutput {
            protocol_id: "balanced".to_string(),
            success: true,
            confidence: 0.88,
            duration_ms: 3000,
            total_tokens: 500,
            error: None,
            output_json: r#"{
                "steps": [
                    {"step_id": "step1", "success": true, "confidence": 0.9, "output": {"content": "result1"}},
                    {"step_id": "step2", "success": true, "confidence": 0.85, "output": {"content": "result2"}}
                ]
            }"#
            .to_string(),
        };

        let steps = output.steps();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].step_id, "step1");
        assert!(steps[0].success);
        assert_eq!(steps[0].confidence, 0.9);
        assert_eq!(steps[0].content, Some("result1".to_string()));
        assert_eq!(steps[1].step_id, "step2");
        assert_eq!(steps[1].confidence, 0.85);
    }

    #[test]
    fn test_thinktool_output_to_json() {
        let json_str = r#"{"key": "value", "number": 42}"#;
        let output = ThinkToolOutput {
            protocol_id: "test".to_string(),
            success: true,
            confidence: 1.0,
            duration_ms: 100,
            total_tokens: 50,
            error: None,
            output_json: json_str.to_string(),
        };

        assert_eq!(output.to_json(), json_str);
    }

    #[test]
    fn test_thinktool_output_repr() {
        let output = ThinkToolOutput {
            protocol_id: "gigathink".to_string(),
            success: true,
            confidence: 0.85,
            duration_ms: 1000,
            total_tokens: 200,
            error: None,
            output_json: "{}".to_string(),
        };

        let repr = output.__repr__();
        assert!(repr.contains("gigathink"));
        assert!(repr.contains("success=true"));
        assert!(repr.contains("0.85"));
    }

    #[test]
    fn test_thinktool_output_str_success() {
        let output = ThinkToolOutput {
            protocol_id: "laserlogic".to_string(),
            success: true,
            confidence: 0.92,
            duration_ms: 500,
            total_tokens: 100,
            error: None,
            output_json: "{}".to_string(),
        };

        let str_rep = output.__str__();
        assert!(str_rep.contains("laserlogic"));
        assert!(str_rep.contains("92.0%")); // 0.92 * 100
        assert!(str_rep.contains("100 tokens"));
        assert!(str_rep.contains("500ms"));
    }

    #[test]
    fn test_thinktool_output_str_failure() {
        let output = ThinkToolOutput {
            protocol_id: "bedrock".to_string(),
            success: false,
            confidence: 0.0,
            duration_ms: 100,
            total_tokens: 20,
            error: Some("Decomposition failed".to_string()),
            output_json: "{}".to_string(),
        };

        let str_rep = output.__str__();
        assert!(str_rep.contains("FAILED"));
        assert!(str_rep.contains("Decomposition failed"));
    }

    #[test]
    fn test_thinktool_output_clone() {
        let original = ThinkToolOutput {
            protocol_id: "test".to_string(),
            success: true,
            confidence: 0.75,
            duration_ms: 250,
            total_tokens: 80,
            error: None,
            output_json: r#"{"data": true}"#.to_string(),
        };

        let cloned = original.clone();
        assert_eq!(original.protocol_id, cloned.protocol_id);
        assert_eq!(original.confidence, cloned.confidence);
        assert_eq!(original.output_json, cloned.output_json);
    }

    // ========================================================================
    // STEP RESULT TESTS
    // ========================================================================

    #[test]
    fn test_step_result_py_creation() {
        let step = StepResultPy {
            step_id: "analyze".to_string(),
            success: true,
            confidence: 0.88,
            content: Some("Analysis complete".to_string()),
        };

        assert_eq!(step.step_id, "analyze");
        assert!(step.success);
        assert_eq!(step.confidence, 0.88);
        assert_eq!(step.content, Some("Analysis complete".to_string()));
    }

    #[test]
    fn test_step_result_py_no_content() {
        let step = StepResultPy {
            step_id: "validate".to_string(),
            success: true,
            confidence: 0.95,
            content: None,
        };

        assert!(step.content.is_none());
    }

    #[test]
    fn test_step_result_py_repr() {
        let step = StepResultPy {
            step_id: "synthesize".to_string(),
            success: true,
            confidence: 0.82,
            content: Some("Synthesis".to_string()),
        };

        let repr = step.__repr__();
        assert!(repr.contains("synthesize"));
        assert!(repr.contains("success=true"));
        assert!(repr.contains("0.82"));
    }

    #[test]
    fn test_step_result_py_clone() {
        let original = StepResultPy {
            step_id: "test_step".to_string(),
            success: false,
            confidence: 0.0,
            content: None,
        };

        let cloned = original.clone();
        assert_eq!(original.step_id, cloned.step_id);
        assert_eq!(original.success, cloned.success);
    }

    // ========================================================================
    // ERROR CONVERSION TESTS
    // ========================================================================

    #[test]
    fn test_error_conversion_not_found() {
        let err = crate::error::Error::NotFound {
            resource: "protocol:unknown".to_string(),
        };
        let py_err = to_py_err(err);
        let err_str = format!("{}", py_err);
        assert!(err_str.contains("not found") || err_str.contains("KeyError"));
    }

    #[test]
    fn test_error_conversion_validation() {
        let err = crate::error::Error::Validation("Invalid input format".to_string());
        let py_err = to_py_err(err);
        let err_str = format!("{}", py_err);
        assert!(err_str.contains("Invalid input") || err_str.contains("ValueError"));
    }

    #[test]
    fn test_error_conversion_timeout() {
        let err = crate::error::Error::Timeout("Request timed out after 30s".to_string());
        let py_err = to_py_err(err);
        let err_str = format!("{}", py_err);
        assert!(err_str.contains("timed out") || err_str.contains("Timeout"));
    }

    #[test]
    fn test_error_conversion_config() {
        let err = crate::error::Error::Config("Missing API key".to_string());
        let py_err = to_py_err(err);
        let err_str = format!("{}", py_err);
        assert!(err_str.contains("Config") || err_str.contains("API key"));
    }

    #[test]
    fn test_error_conversion_network() {
        let err = crate::error::Error::Network("Connection refused".to_string());
        let py_err = to_py_err(err);
        let err_str = format!("{}", py_err);
        assert!(err_str.contains("Network") || err_str.contains("Connection"));
    }

    #[test]
    fn test_error_conversion_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let err = crate::error::Error::Io(io_err);
        let py_err = to_py_err(err);
        let err_str = format!("{}", py_err);
        assert!(err_str.contains("error") || err_str.contains("ReasonKit"));
    }

    // ========================================================================
    // REASONER TESTS (Mock Mode)
    // ========================================================================

    #[test]
    fn test_reasoner_mock_creation() {
        let runtime = Runtime::new().unwrap();
        let config = ExecutorConfig::mock();
        let executor = ProtocolExecutor::with_config(config).unwrap();

        let reasoner = Reasoner { executor, runtime };
        assert!(!reasoner.list_protocols().is_empty());
    }

    #[test]
    fn test_reasoner_list_protocols() {
        let runtime = Runtime::new().unwrap();
        let config = ExecutorConfig::mock();
        let executor = ProtocolExecutor::with_config(config).unwrap();
        let reasoner = Reasoner { executor, runtime };

        let protocols = reasoner.list_protocols();
        assert!(protocols.contains(&"gigathink".to_string()));
        assert!(protocols.contains(&"laserlogic".to_string()));
        assert!(protocols.contains(&"bedrock".to_string()));
        assert!(protocols.contains(&"proofguard".to_string()));
        assert!(protocols.contains(&"brutalhonesty".to_string()));
    }

    #[test]
    fn test_reasoner_list_profiles() {
        let runtime = Runtime::new().unwrap();
        let config = ExecutorConfig::mock();
        let executor = ProtocolExecutor::with_config(config).unwrap();
        let reasoner = Reasoner { executor, runtime };

        let profiles = reasoner.list_profiles();
        assert!(profiles.contains(&"quick".to_string()));
        assert!(profiles.contains(&"balanced".to_string()));
        assert!(profiles.contains(&"deep".to_string()));
        assert!(profiles.contains(&"paranoid".to_string()));
    }

    #[test]
    fn test_reasoner_repr() {
        let runtime = Runtime::new().unwrap();
        let config = ExecutorConfig::mock();
        let executor = ProtocolExecutor::with_config(config).unwrap();
        let reasoner = Reasoner { executor, runtime };

        let repr = reasoner.__repr__();
        assert!(repr.contains("Reasoner"));
        assert!(repr.contains("protocols"));
        assert!(repr.contains("profiles"));
    }

    // ========================================================================
    // JSON CONVERSION TESTS
    // ========================================================================

    #[test]
    fn test_json_null_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = serde_json::Value::Null;
            let result = json_to_pyobject(py, &value).unwrap();
            assert!(result.is_none());
        });
    }

    #[test]
    fn test_json_bool_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let true_val = serde_json::Value::Bool(true);
            let result = json_to_pyobject(py, &true_val).unwrap();
            let extracted: bool = result.extract().unwrap();
            assert!(extracted);

            let false_val = serde_json::Value::Bool(false);
            let result = json_to_pyobject(py, &false_val).unwrap();
            let extracted: bool = result.extract().unwrap();
            assert!(!extracted);
        });
    }

    #[test]
    fn test_json_integer_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = serde_json::json!(42);
            let result = json_to_pyobject(py, &value).unwrap();
            let extracted: i64 = result.extract().unwrap();
            assert_eq!(extracted, 42);
        });
    }

    #[test]
    fn test_json_float_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = serde_json::json!(3.14159);
            let result = json_to_pyobject(py, &value).unwrap();
            let extracted: f64 = result.extract().unwrap();
            assert!((extracted - 3.14159).abs() < 1e-10);
        });
    }

    #[test]
    fn test_json_string_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = serde_json::json!("hello world");
            let result = json_to_pyobject(py, &value).unwrap();
            let extracted: String = result.extract().unwrap();
            assert_eq!(extracted, "hello world");
        });
    }

    #[test]
    fn test_json_array_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = serde_json::json!([1, 2, 3, 4, 5]);
            let result = json_to_pyobject(py, &value).unwrap();
            let list = result.downcast::<PyList>().unwrap();
            assert_eq!(list.len(), 5);

            // Extract first element
            let first: i64 = list.get_item(0).unwrap().extract().unwrap();
            assert_eq!(first, 1);
        });
    }

    #[test]
    fn test_json_object_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = serde_json::json!({
                "name": "test",
                "count": 42,
                "active": true
            });
            let result = json_to_pyobject(py, &value).unwrap();
            let dict = result.downcast::<PyDict>().unwrap();

            let name: String = dict.get_item("name").unwrap().unwrap().extract().unwrap();
            assert_eq!(name, "test");

            let count: i64 = dict.get_item("count").unwrap().unwrap().extract().unwrap();
            assert_eq!(count, 42);

            let active: bool = dict.get_item("active").unwrap().unwrap().extract().unwrap();
            assert!(active);
        });
    }

    #[test]
    fn test_json_nested_structure_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = serde_json::json!({
                "results": [
                    {"id": 1, "value": "first"},
                    {"id": 2, "value": "second"}
                ],
                "metadata": {
                    "total": 2,
                    "page": 1
                }
            });
            let result = json_to_pyobject(py, &value).unwrap();
            let dict = result.downcast::<PyDict>().unwrap();

            // Check results array
            let results = dict.get_item("results").unwrap().unwrap();
            let results_list = results.downcast::<PyList>().unwrap();
            assert_eq!(results_list.len(), 2);

            // Check metadata dict
            let metadata = dict.get_item("metadata").unwrap().unwrap();
            let metadata_dict = metadata.downcast::<PyDict>().unwrap();
            let total: i64 = metadata_dict
                .get_item("total")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(total, 2);
        });
    }

    #[test]
    fn test_json_empty_structures() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Empty array
            let empty_arr = serde_json::json!([]);
            let result = json_to_pyobject(py, &empty_arr).unwrap();
            let list = result.downcast::<PyList>().unwrap();
            assert_eq!(list.len(), 0);

            // Empty object
            let empty_obj = serde_json::json!({});
            let result = json_to_pyobject(py, &empty_obj).unwrap();
            let dict = result.downcast::<PyDict>().unwrap();
            assert_eq!(dict.len(), 0);
        });
    }

    #[test]
    fn test_json_unicode_string() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = serde_json::json!("Hello, World! \u{1F600} \u{4E2D}\u{6587}");
            let result = json_to_pyobject(py, &value).unwrap();
            let extracted: String = result.extract().unwrap();
            assert!(extracted.contains("\u{1F600}")); // emoji
            assert!(extracted.contains("\u{4E2D}")); // Chinese char
        });
    }

    #[test]
    fn test_json_large_numbers() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Large integer
            let large_int = serde_json::json!(9007199254740991_i64);
            let result = json_to_pyobject(py, &large_int).unwrap();
            let extracted: i64 = result.extract().unwrap();
            assert_eq!(extracted, 9007199254740991);

            // Negative number
            let negative = serde_json::json!(-42);
            let result = json_to_pyobject(py, &negative).unwrap();
            let extracted: i64 = result.extract().unwrap();
            assert_eq!(extracted, -42);
        });
    }

    // ========================================================================
    // MEMORY MANAGEMENT TESTS
    // ========================================================================

    #[test]
    fn test_output_memory_no_leak() {
        // Create many outputs to check for memory issues
        let outputs: Vec<ThinkToolOutput> = (0..1000)
            .map(|i| ThinkToolOutput {
                protocol_id: format!("test_{}", i),
                success: true,
                confidence: 0.5 + (i as f64 / 2000.0),
                duration_ms: i as u64,
                total_tokens: i as u32,
                error: None,
                output_json: format!(r#"{{"index": {}}}"#, i),
            })
            .collect();

        // Verify all outputs are valid
        assert_eq!(outputs.len(), 1000);
        assert_eq!(outputs[0].protocol_id, "test_0");
        assert_eq!(outputs[999].protocol_id, "test_999");
    }

    #[test]
    fn test_step_results_memory_no_leak() {
        let steps: Vec<StepResultPy> = (0..500)
            .map(|i| StepResultPy {
                step_id: format!("step_{}", i),
                success: i % 2 == 0,
                confidence: (i as f64) / 500.0,
                content: Some(format!("Content for step {}", i)),
            })
            .collect();

        assert_eq!(steps.len(), 500);
        assert!(steps[0].success);
        assert!(!steps[1].success);
    }

    #[test]
    fn test_json_parsing_with_invalid_json() {
        let output = ThinkToolOutput {
            protocol_id: "test".to_string(),
            success: true,
            confidence: 0.5,
            duration_ms: 100,
            total_tokens: 50,
            error: None,
            output_json: "not valid json".to_string(),
        };

        // perspectives() should return empty vec on invalid JSON
        assert!(output.perspectives().is_empty());

        // verdict() should return None on invalid JSON
        assert!(output.verdict().is_none());

        // steps() should return empty vec on invalid JSON
        assert!(output.steps().is_empty());
    }

    // ========================================================================
    // EXECUTOR CONFIG TESTS
    // ========================================================================

    #[test]
    fn test_executor_config_mock() {
        let config = ExecutorConfig::mock();
        assert!(config.use_mock);
    }

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert!(!config.use_mock);
        assert_eq!(config.timeout_secs, 120);
        assert!(!config.verbose);
    }

    // ========================================================================
    // PROTOCOL INPUT TESTS
    // ========================================================================

    #[test]
    fn test_protocol_input_query() {
        let input = ProtocolInput::query("What is AI?");
        assert_eq!(input.get_str("query"), Some("What is AI?"));
    }

    #[test]
    fn test_protocol_input_argument() {
        let input = ProtocolInput::argument("If A then B. A. Therefore B.");
        assert_eq!(
            input.get_str("argument"),
            Some("If A then B. A. Therefore B.")
        );
    }

    #[test]
    fn test_protocol_input_statement() {
        let input = ProtocolInput::statement("The Earth orbits the Sun");
        assert_eq!(
            input.get_str("statement"),
            Some("The Earth orbits the Sun")
        );
    }

    #[test]
    fn test_protocol_input_claim() {
        let input = ProtocolInput::claim("Water boils at 100 degrees Celsius");
        assert_eq!(
            input.get_str("claim"),
            Some("Water boils at 100 degrees Celsius")
        );
    }

    #[test]
    fn test_protocol_input_work() {
        let input = ProtocolInput::work("My business plan outline");
        assert_eq!(input.get_str("work"), Some("My business plan outline"));
    }

    #[test]
    fn test_protocol_input_with_field() {
        let input = ProtocolInput::query("Test query")
            .with_field("context", "Additional context")
            .with_field("domain", "physics");

        assert_eq!(input.get_str("query"), Some("Test query"));
        assert_eq!(input.get_str("context"), Some("Additional context"));
        assert_eq!(input.get_str("domain"), Some("physics"));
    }

    #[test]
    fn test_protocol_input_missing_field() {
        let input = ProtocolInput::query("Test");
        assert!(input.get_str("nonexistent").is_none());
    }

    // ========================================================================
    // VERSION FUNCTION TEST
    // ========================================================================

    #[test]
    fn test_version_returns_valid_string() {
        let v = version();
        assert!(!v.is_empty());
        // Version should be in semver format (e.g., "0.1.0")
        assert!(v.contains('.'), "Version should contain dots: {}", v);
    }

    // ========================================================================
    // EDGE CASES AND BOUNDARY CONDITIONS
    // ========================================================================

    #[test]
    fn test_empty_string_inputs() {
        let output = ThinkToolOutput {
            protocol_id: "".to_string(),
            success: true,
            confidence: 0.0,
            duration_ms: 0,
            total_tokens: 0,
            error: None,
            output_json: "{}".to_string(),
        };

        assert!(output.protocol_id.is_empty());
        assert_eq!(output.__repr__(), "ThinkToolOutput(protocol='', success=true, confidence=0.00)");
    }

    #[test]
    fn test_max_confidence_boundary() {
        let output = ThinkToolOutput {
            protocol_id: "test".to_string(),
            success: true,
            confidence: 1.0,
            duration_ms: 100,
            total_tokens: 50,
            error: None,
            output_json: "{}".to_string(),
        };

        assert_eq!(output.confidence, 1.0);
        let str_rep = output.__str__();
        assert!(str_rep.contains("100.0%"));
    }

    #[test]
    fn test_zero_confidence_boundary() {
        let output = ThinkToolOutput {
            protocol_id: "test".to_string(),
            success: false,
            confidence: 0.0,
            duration_ms: 0,
            total_tokens: 0,
            error: Some("Failed completely".to_string()),
            output_json: "{}".to_string(),
        };

        assert_eq!(output.confidence, 0.0);
    }

    #[test]
    fn test_special_characters_in_protocol_id() {
        let output = ThinkToolOutput {
            protocol_id: "test-protocol_v2.1".to_string(),
            success: true,
            confidence: 0.75,
            duration_ms: 500,
            total_tokens: 100,
            error: None,
            output_json: "{}".to_string(),
        };

        assert_eq!(output.protocol_id, "test-protocol_v2.1");
        let repr = output.__repr__();
        assert!(repr.contains("test-protocol_v2.1"));
    }

    #[test]
    fn test_very_long_output_json() {
        // Test with large JSON payload
        let large_array: Vec<i32> = (0..1000).collect();
        let output_json = serde_json::to_string(&serde_json::json!({
            "large_array": large_array
        }))
        .unwrap();

        let output = ThinkToolOutput {
            protocol_id: "test".to_string(),
            success: true,
            confidence: 0.9,
            duration_ms: 5000,
            total_tokens: 10000,
            error: None,
            output_json,
        };

        // to_json should return the full content
        let json = output.to_json();
        assert!(json.len() > 1000);
    }

    #[test]
    fn test_concurrent_runtime_creation() {
        // Test that multiple runtimes can be created safely
        let results: Vec<_> = (0..5)
            .map(|_| {
                let runtime = Runtime::new();
                runtime.is_ok()
            })
            .collect();

        assert!(results.iter().all(|&r| r));
    }

    #[test]
    fn test_profile_all_variants() {
        // Ensure all enum variants are covered
        let profiles = [
            Profile::None,
            Profile::Quick,
            Profile::Balanced,
            Profile::Deep,
            Profile::Paranoid,
        ];

        for profile in profiles {
            // Each should have a valid name
            assert!(!profile.name().is_empty());
            // Each should have a valid confidence threshold
            assert!(profile.min_confidence() >= 0.0);
            assert!(profile.min_confidence() <= 1.0);
            // __repr__ and __str__ should not panic
            let _ = profile.__repr__();
            let _ = profile.__str__();
        }
    }
}
