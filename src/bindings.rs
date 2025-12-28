#![allow(clippy::useless_conversion)]

use crate::thinktool::executor::{ExecutorConfig, ProtocolExecutor, ProtocolInput};
use pyo3::prelude::*;
use tokio::runtime::Runtime;

#[pyclass]
pub struct Reasoner {
    executor: ProtocolExecutor,
    runtime: Runtime,
}

#[pymethods]
impl Reasoner {
    #[new]
    #[pyo3(signature = (use_mock=None))]
    fn new(use_mock: Option<bool>) -> PyResult<Self> {
        let runtime = Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create runtime: {}",
                e
            ))
        })?;

        let config = if use_mock.unwrap_or(false) {
            ExecutorConfig::mock()
        } else {
            ExecutorConfig::default()
        };

        let executor = ProtocolExecutor::with_config(config).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialize executor: {}",
                e
            ))
        })?;

        Ok(Reasoner { executor, runtime })
    }

    /// Execute a protocol with a query string
    /// Returns the JSON string of the full ProtocolOutput object
    fn think(&self, protocol: String, query: String) -> PyResult<String> {
        let input = ProtocolInput::query(query);

        let result = self
            .runtime
            .block_on(async { self.executor.execute(&protocol, input).await });

        match result {
            Ok(output) => {
                // Debug print to see what we are actually getting
                // println!("RUST DEBUG: {:?}", output);
                serde_json::to_string(&output).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Serialization error: {}",
                        e
                    ))
                })
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Execution error: {}",
                e
            ))),
        }
    }

    /// List available protocols
    fn list_protocols(&self) -> Vec<String> {
        self.executor
            .list_protocols()
            .into_iter()
            .map(String::from)
            .collect()
    }
}
