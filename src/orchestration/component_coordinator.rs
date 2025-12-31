//! # Component Coordination System
//!
//! This module coordinates execution across all ReasonKit components (Core, Web, Mem, Pro)
//! and MiniMax M2 integration for complex long-horizon operations.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::error::Error;

#[derive(Debug, Default)]
struct CoreComponentStub;

impl CoreComponentStub {
    async fn new() -> Result<Self, Error> {
        Ok(Self)
    }

    async fn generate_protocol(
        &mut self,
        _input: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "generate_protocol"}))
    }

    async fn analyze_code(
        &mut self,
        _input: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "analyze_code"}))
    }

    async fn execute_thinktool(
        &mut self,
        _tool_name: &str,
        _input: &str,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "execute_thinktool"}))
    }

    async fn execute_general_task(
        &mut self,
        _task_name: &str,
        _config: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "execute_general_task"}))
    }
}

#[derive(Debug, Default)]
struct WebComponentStub;

impl WebComponentStub {
    async fn new() -> Result<Self, Error> {
        Ok(Self)
    }

    async fn browse_url(
        &mut self,
        url: &str,
        _config: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "browse_url", "url": url}))
    }

    async fn take_screenshot(
        &mut self,
        target: &str,
        _config: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "take_screenshot", "target": target}))
    }

    async fn extract_content(
        &mut self,
        selector: &str,
        _config: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "extract_content", "selector": selector}))
    }

    async fn execute_web_task(
        &mut self,
        task_name: &str,
        _config: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "execute_web_task", "task": task_name}))
    }
}

#[derive(Debug, Default)]
struct MemComponentStub {
    data: HashMap<String, serde_json::Value>,
}

impl MemComponentStub {
    async fn new() -> Result<Self, Error> {
        Ok(Self::default())
    }

    async fn store(&mut self, key: &str, value: serde_json::Value) -> Result<(), Error> {
        self.data.insert(key.to_string(), value);
        Ok(())
    }

    async fn retrieve(&mut self, key: &str) -> Result<Option<serde_json::Value>, Error> {
        Ok(self.data.get(key).cloned())
    }

    async fn search(&mut self, query: &str) -> Result<Vec<serde_json::Value>, Error> {
        // naive substring search on JSON strings
        let mut results = Vec::new();
        for (key, value) in &self.data {
            if key.contains(query) || value.to_string().contains(query) {
                results.push(serde_json::json!({"key": key, "value": value}));
            }
        }
        Ok(results)
    }

    async fn execute_memory_task(
        &mut self,
        task_name: &str,
        _config: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "execute_memory_task", "task": task_name}))
    }
}

#[derive(Debug, Default)]
struct ProComponentStub;

impl ProComponentStub {
    async fn new() -> Result<Self, Error> {
        Ok(Self)
    }

    async fn execute_digital_employee_workflow(
        &mut self,
        _workflow: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "execute_digital_employee_workflow"}))
    }

    async fn execute_enterprise_automation(
        &mut self,
        _automation_config: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "execute_enterprise_automation"}))
    }

    async fn execute_pro_task(
        &mut self,
        task_name: &str,
        _config: serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        Ok(serde_json::json!({"stub": true, "op": "execute_pro_task", "task": task_name}))
    }
}

/// Placeholder for future component-level tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentTask {
    pub component: String,
    pub operation: String,
    pub payload: serde_json::Value,
}

/// Coordinates execution across ReasonKit components
pub struct ComponentCoordinator {
    /// Core ReasonKit component
    core_component: Arc<Mutex<CoreComponentStub>>,
    /// Web sensing component
    web_component: Arc<Mutex<WebComponentStub>>,
    /// Memory layer component
    mem_component: Arc<Mutex<MemComponentStub>>,
    /// Professional/Advanced features
    pro_component: Arc<Mutex<ProComponentStub>>,
    /// MiniMax M2 integration
    m2_integration: Arc<Mutex<super::super::m2::M2IntegrationService>>,
    /// Component health monitoring
    health_monitor: Arc<Mutex<ComponentHealthMonitor>>,
    /// Execution registry
    execution_registry: Arc<Mutex<ComponentExecutionRegistry>>,
}

impl ComponentCoordinator {
    pub async fn new() -> Result<Self, Error> {
        tracing::info!("Initializing Component Coordinator");

        let core_component = Arc::new(Mutex::new(CoreComponentStub::new().await?));
        let web_component = Arc::new(Mutex::new(WebComponentStub::new().await?));
        let mem_component = Arc::new(Mutex::new(MemComponentStub::new().await?));
        let pro_component = Arc::new(Mutex::new(ProComponentStub::new().await?));

        // Initialize M2 integration
        let m2_config = super::super::m2::M2Config::default();
        let m2_integration_config = super::super::m2::M2IntegrationConfig::default();
        let m2_integration = Arc::new(Mutex::new(
            super::super::m2::M2IntegrationService::new(m2_config, m2_integration_config).await?,
        ));

        let health_monitor = Arc::new(Mutex::new(ComponentHealthMonitor::new()));
        let execution_registry = Arc::new(Mutex::new(ComponentExecutionRegistry::new()));

        Ok(Self {
            core_component,
            web_component,
            mem_component,
            pro_component,
            m2_integration,
            health_monitor,
            execution_registry,
        })
    }

    /// Coordinate components for a specific task
    pub async fn coordinate_components(
        &self,
        task_name: &str,
        required_components: &[String],
        task_config: &serde_json::Value,
    ) -> Result<CoordinationResult, Error> {
        let start_time = std::time::Instant::now();
        let mut tool_calls_used = 0;
        let mut outputs = HashMap::new();
        let mut errors = Vec::new();

        tracing::info!(
            " Coordinating {} components for task: {}",
            required_components.len(),
            task_name
        );

        // Create execution plan
        let execution_plan = self
            .create_execution_plan(required_components, task_config)
            .await?;

        // Execute components according to plan
        for component_name in &execution_plan.execution_order {
            match component_name.as_str() {
                "reasonkit-core" => {
                    match self.execute_core_component(task_name, task_config).await {
                        Ok(output) => {
                            outputs.insert("core".to_string(), output);
                            tool_calls_used += 1;
                        }
                        Err(e) => {
                            errors.push(format!("Core component error: {}", e));
                            tracing::error!("Core component failed: {}", e);
                        }
                    }
                }
                "reasonkit-web" => match self.execute_web_component(task_name, task_config).await {
                    Ok(output) => {
                        outputs.insert("web".to_string(), output);
                        tool_calls_used += 1;
                    }
                    Err(e) => {
                        errors.push(format!("Web component error: {}", e));
                        tracing::error!("Web component failed: {}", e);
                    }
                },
                "reasonkit-mem" => match self.execute_mem_component(task_name, task_config).await {
                    Ok(output) => {
                        outputs.insert("memory".to_string(), output);
                        tool_calls_used += 1;
                    }
                    Err(e) => {
                        errors.push(format!("Memory component error: {}", e));
                        tracing::error!("Memory component failed: {}", e);
                    }
                },
                "reasonkit-pro" => match self.execute_pro_component(task_name, task_config).await {
                    Ok(output) => {
                        outputs.insert("pro".to_string(), output);
                        tool_calls_used += 1;
                    }
                    Err(e) => {
                        errors.push(format!("Pro component error: {}", e));
                        tracing::error!("Pro component failed: {}", e);
                    }
                },
                "minimax-m2" => {
                    match self.execute_m2_integration(task_name, task_config).await {
                        Ok(output) => {
                            outputs.insert("m2".to_string(), output);
                            tool_calls_used += 2; // M2 typically uses multiple tool calls
                        }
                        Err(e) => {
                            errors.push(format!("M2 integration error: {}", e));
                            tracing::error!("M2 integration failed: {}", e);
                        }
                    }
                }
                "thinktools" => match self.execute_thinktools(task_name, task_config).await {
                    Ok(output) => {
                        outputs.insert("thinktools".to_string(), output);
                        tool_calls_used += 1;
                    }
                    Err(e) => {
                        errors.push(format!("ThinkTools error: {}", e));
                        tracing::error!("ThinkTools failed: {}", e);
                    }
                },
                _ => {
                    tracing::warn!("Unknown component: {}", component_name);
                }
            }
        }

        // Update health monitor
        {
            let mut monitor = self.health_monitor.lock().await;
            monitor.record_execution(task_name, required_components, errors.is_empty());
        }

        // Register execution
        {
            let mut registry = self.execution_registry.lock().await;
            registry.register_execution(ComponentExecutionRecord {
                task_name: task_name.to_string(),
                timestamp: chrono::Utc::now().timestamp(),
                components_used: required_components.to_vec(),
                duration_ms: start_time.elapsed().as_millis() as u64,
                success: errors.is_empty(),
                tool_calls_used,
                outputs: outputs.clone(),
                errors: errors.clone(),
            });
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;
        let overall_success = errors.is_empty();
        let error_count = errors.len();

        let result = CoordinationResult {
            success: overall_success,
            outputs,
            tool_calls_used,
            duration_ms,
            components_used: required_components.to_vec(),
            errors,
        };

        if overall_success {
            tracing::info!(
                "Component coordination completed successfully: {} tool calls, {}ms",
                tool_calls_used,
                duration_ms
            );
        } else {
            tracing::warn!(
                "Component coordination completed with {} errors: {}ms",
                error_count,
                duration_ms
            );
        }

        Ok(result)
    }

    /// Execute ReasonKit Core component
    async fn execute_core_component(
        &self,
        task_name: &str,
        config: &serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        let mut core = self.core_component.lock().await;

        // Determine execution method based on task type
        let task_type = config
            .get("task_type")
            .and_then(|v| v.as_str())
            .unwrap_or("general");

        match task_type {
            "protocol_generation" => {
                let protocol_input = config
                    .get("protocol_input")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"query": task_name}));

                // Execute protocol generation
                let result = core.generate_protocol(protocol_input).await?;
                Ok(serde_json::json!({
                    "type": "protocol_generation",
                    "result": result,
                    "component": "reasonkit-core"
                }))
            }
            "code_analysis" => {
                let code_input = config
                    .get("code_input")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"code": ""}));

                // Execute code analysis
                let result = core.analyze_code(code_input).await?;
                Ok(serde_json::json!({
                    "type": "code_analysis",
                    "result": result,
                    "component": "reasonkit-core"
                }))
            }
            "thinktool_execution" => {
                let tool_name = config
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("gigathink");
                let input = config
                    .get("input")
                    .and_then(|v| v.as_str())
                    .unwrap_or(task_name);

                // Execute ThinkTool
                let result = core.execute_thinktool(tool_name, input).await?;
                Ok(serde_json::json!({
                    "type": "thinktool_execution",
                    "tool": tool_name,
                    "result": result,
                    "component": "reasonkit-core"
                }))
            }
            _ => {
                // General execution
                let result = core.execute_general_task(task_name, config.clone()).await?;
                Ok(serde_json::json!({
                    "type": "general",
                    "result": result,
                    "component": "reasonkit-core"
                }))
            }
        }
    }

    /// Execute ReasonKit Web component
    async fn execute_web_component(
        &self,
        task_name: &str,
        config: &serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        let mut web = self.web_component.lock().await;

        // Determine web operation
        let operation = config
            .get("web_operation")
            .and_then(|v| v.as_str())
            .unwrap_or("browse");

        match operation {
            "browse" => {
                let url = config.get("url").and_then(|v| v.as_str()).ok_or_else(|| {
                    Error::Validation("URL required for browse operation".to_string())
                })?;

                let result = web.browse_url(url, config.clone()).await?;
                Ok(serde_json::json!({
                    "type": "web_browse",
                    "url": url,
                    "result": result,
                    "component": "reasonkit-web"
                }))
            }
            "screenshot" => {
                let target = config
                    .get("target")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        Error::Validation("Target required for screenshot".to_string())
                    })?;

                let result = web.take_screenshot(target, config.clone()).await?;
                Ok(serde_json::json!({
                    "type": "web_screenshot",
                    "target": target,
                    "result": result,
                    "component": "reasonkit-web"
                }))
            }
            "extract_content" => {
                let selector =
                    config
                        .get("selector")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            Error::Validation(
                                "Selector required for content extraction".to_string(),
                            )
                        })?;

                let result = web.extract_content(selector, config.clone()).await?;
                Ok(serde_json::json!({
                    "type": "content_extraction",
                    "selector": selector,
                    "result": result,
                    "component": "reasonkit-web"
                }))
            }
            _ => {
                let result = web.execute_web_task(task_name, config.clone()).await?;
                Ok(serde_json::json!({
                    "type": "web_general",
                    "result": result,
                    "component": "reasonkit-web"
                }))
            }
        }
    }

    /// Execute ReasonKit Memory component
    async fn execute_mem_component(
        &self,
        task_name: &str,
        config: &serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        let mut mem = self.mem_component.lock().await;

        let operation = config
            .get("memory_operation")
            .and_then(|v| v.as_str())
            .unwrap_or("store");

        match operation {
            "store" => {
                let key = config.get("key").and_then(|v| v.as_str()).ok_or_else(|| {
                    Error::Validation("Key required for memory store".to_string())
                })?;
                let value = config.get("value").cloned().ok_or_else(|| {
                    Error::Validation("Value required for memory store".to_string())
                })?;

                mem.store(key, value).await?;
                Ok(serde_json::json!({
                    "type": "memory_store",
                    "key": key,
                    "status": "stored",
                    "component": "reasonkit-mem"
                }))
            }
            "retrieve" => {
                let key = config.get("key").and_then(|v| v.as_str()).ok_or_else(|| {
                    Error::Validation("Key required for memory retrieve".to_string())
                })?;

                let value = mem.retrieve(key).await?;
                Ok(serde_json::json!({
                    "type": "memory_retrieve",
                    "key": key,
                    "value": value,
                    "component": "reasonkit-mem"
                }))
            }
            "search" => {
                let query = config
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        Error::Validation("Query required for memory search".to_string())
                    })?;

                let results = mem.search(query).await?;
                Ok(serde_json::json!({
                    "type": "memory_search",
                    "query": query,
                    "results": results,
                    "component": "reasonkit-mem"
                }))
            }
            _ => {
                let result = mem.execute_memory_task(task_name, config.clone()).await?;
                Ok(serde_json::json!({
                    "type": "memory_general",
                    "result": result,
                    "component": "reasonkit-mem"
                }))
            }
        }
    }

    /// Execute ReasonKit Pro component
    async fn execute_pro_component(
        &self,
        task_name: &str,
        config: &serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        let mut pro = self.pro_component.lock().await;

        let operation = config
            .get("pro_operation")
            .and_then(|v| v.as_str())
            .unwrap_or("digital_employee");

        match operation {
            "digital_employee" => {
                let workflow = config
                    .get("workflow")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"task": task_name}));

                let result = pro
                    .execute_digital_employee_workflow(workflow.clone())
                    .await?;
                Ok(serde_json::json!({
                    "type": "digital_employee",
                    "workflow": workflow,
                    "result": result,
                    "component": "reasonkit-pro"
                }))
            }
            "enterprise_automation" => {
                let automation_config = config
                    .get("automation_config")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"task": task_name}));

                let result = pro
                    .execute_enterprise_automation(automation_config.clone())
                    .await?;
                Ok(serde_json::json!({
                    "type": "enterprise_automation",
                    "config": automation_config,
                    "result": result,
                    "component": "reasonkit-pro"
                }))
            }
            _ => {
                let result = pro.execute_pro_task(task_name, config.clone()).await?;
                Ok(serde_json::json!({
                    "type": "pro_general",
                    "result": result,
                    "component": "reasonkit-pro"
                }))
            }
        }
    }

    /// Execute MiniMax M2 integration
    async fn execute_m2_integration(
        &self,
        task_name: &str,
        config: &serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        let m2 = self.m2_integration.lock().await;

        let use_case = config
            .get("m2_use_case")
            .and_then(|v| v.as_str())
            .unwrap_or("general");

        let use_case_enum = match use_case {
            "code_analysis" => super::super::m2::UseCase::CodeAnalysis,
            "bug_finding" => super::super::m2::UseCase::BugFinding,
            "documentation" => super::super::m2::UseCase::Documentation,
            "architecture" => super::super::m2::UseCase::Architecture,
            _ => super::super::m2::UseCase::General,
        };

        let input = config
            .get("m2_input")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"task": task_name}));

        let framework = config
            .get("framework")
            .and_then(|v| v.as_str())
            .and_then(|s| match s {
                "claude_code" => Some(super::super::m2::AgentFramework::ClaudeCode),
                "cline" => Some(super::super::m2::AgentFramework::Cline),
                _ => None,
            });

        let result = m2
            .execute_for_use_case(use_case_enum, input, framework)
            .await?;

        Ok(serde_json::json!({
            "type": "m2_execution",
            "use_case": use_case,
            "result": result,
            "component": "minimax-m2"
        }))
    }

    /// Execute ThinkTools with M2 enhancements
    #[allow(unused_variables)] // config is used conditionally based on features
    async fn execute_thinktools(
        &self,
        task_name: &str,
        config: &serde_json::Value,
    ) -> Result<serde_json::Value, Error> {
        #[cfg(feature = "minimax")]
        {
            let tool_name = config
                .get("thinktool_name")
                .and_then(|v| v.as_str())
                .unwrap_or("enhanced_gigathink");

            let input = config
                .get("thinktool_input")
                .and_then(|v| v.as_str())
                .unwrap_or(task_name);

            let mut m2_manager = super::super::thinktool::minimax::M2ThinkToolsManager::new();
            let result = m2_manager
                .execute_thinktool(
                    tool_name,
                    input,
                    super::super::thinktool::minimax::ProfileType::Balanced,
                )
                .await?;

            Ok(serde_json::json!({
                "type": "thinktool_execution",
                "tool": tool_name,
                "result": result,
                "component": "thinktools"
            }))
        }

        #[cfg(not(feature = "minimax"))]
        {
            // Fallback to basic ThinkTool execution
            let mut core = self.core_component.lock().await;
            let result = core.execute_thinktool("gigathink", task_name).await?;

            Ok(serde_json::json!({
                "type": "thinktool_basic",
                "tool": "gigathink",
                "result": result,
                "component": "thinktools"
            }))
        }
    }

    /// Create execution plan for components
    async fn create_execution_plan(
        &self,
        required_components: &[String],
        _task_config: &serde_json::Value,
    ) -> Result<ComponentExecutionPlan, Error> {
        let mut execution_order = Vec::new();
        let mut dependencies = HashMap::new();

        // Determine execution order based on dependencies
        for component in required_components {
            match component.as_str() {
                "reasonkit-core" => {
                    execution_order.push("reasonkit-core".to_string());
                    if !dependencies.contains_key("reasonkit-core") {
                        dependencies.insert("reasonkit-core".to_string(), vec![]);
                    }
                }
                "reasonkit-web" => {
                    // Web often depends on core for processing
                    execution_order.push("reasonkit-web".to_string());
                    dependencies.insert(
                        "reasonkit-web".to_string(),
                        vec!["reasonkit-core".to_string()],
                    );
                }
                "reasonkit-mem" => {
                    // Memory can be used by any component
                    execution_order.push("reasonkit-mem".to_string());
                    dependencies.insert("reasonkit-mem".to_string(), vec![]);
                }
                "reasonkit-pro" => {
                    // Pro often depends on core and web
                    execution_order.push("reasonkit-pro".to_string());
                    dependencies.insert(
                        "reasonkit-pro".to_string(),
                        vec!["reasonkit-core".to_string(), "reasonkit-web".to_string()],
                    );
                }
                "minimax-m2" => {
                    // M2 can work independently but often benefits from core data
                    execution_order.push("minimax-m2".to_string());
                    dependencies
                        .insert("minimax-m2".to_string(), vec!["reasonkit-core".to_string()]);
                }
                "thinktools" => {
                    // ThinkTools typically run after core processing
                    execution_order.push("thinktools".to_string());
                    dependencies
                        .insert("thinktools".to_string(), vec!["reasonkit-core".to_string()]);
                }
                _ => {
                    execution_order.push(component.clone());
                    dependencies.insert(component.clone(), vec![]);
                }
            }
        }

        // Adjust order based on dependencies
        execution_order = self.topological_sort_with_dependencies(execution_order, dependencies)?;

        Ok(ComponentExecutionPlan {
            execution_order,
            estimated_duration_ms: self.estimate_execution_duration(required_components),
            resource_requirements: self.calculate_resource_requirements(required_components),
        })
    }

    /// Perform topological sort with dependencies
    fn topological_sort_with_dependencies(
        &self,
        order: Vec<String>,
        dependencies: HashMap<String, Vec<String>>,
    ) -> Result<Vec<String>, Error> {
        let mut result = Vec::new();
        let mut processed = HashSet::new();

        // Simple dependency resolution (in practice, would be more sophisticated)
        while result.len() < order.len() {
            let mut progress = false;

            for component in &order {
                if processed.contains(component) {
                    continue;
                }

                let deps: &[String] = match dependencies.get(component) {
                    Some(deps) => deps,
                    None => &[],
                };
                let deps_satisfied = deps.iter().all(|dep| processed.contains(dep));

                if deps_satisfied {
                    result.push(component.clone());
                    processed.insert(component.clone());
                    progress = true;
                }
            }

            if !progress {
                // Break circular dependency by adding remaining items
                for component in &order {
                    if !processed.contains(component) {
                        result.push(component.clone());
                        processed.insert(component.clone());
                    }
                }
                break;
            }
        }

        Ok(result)
    }

    /// Estimate execution duration for components
    fn estimate_execution_duration(&self, components: &[String]) -> u64 {
        let base_duration = 1000; // 1 second base
        let component_multiplier = match components.len() {
            0..=2 => 1.0,
            3..=5 => 1.5,
            _ => 2.0,
        };

        (base_duration as f64 * component_multiplier) as u64
    }

    /// Calculate resource requirements
    fn calculate_resource_requirements(&self, components: &[String]) -> ResourceRequirements {
        let mut cpu_cores = 1.0;
        let mut memory_mb = 512;
        let mut network_mbps = 10.0;

        for component in components {
            match component.as_str() {
                "reasonkit-core" => {
                    cpu_cores += 1.0;
                    memory_mb += 256;
                }
                "reasonkit-web" => {
                    cpu_cores += 0.5;
                    memory_mb += 128;
                    network_mbps += 50.0;
                }
                "reasonkit-mem" => {
                    memory_mb += 256;
                }
                "reasonkit-pro" => {
                    cpu_cores += 2.0;
                    memory_mb += 512;
                }
                "minimax-m2" => {
                    cpu_cores += 1.0;
                    memory_mb += 128;
                }
                "thinktools" => {
                    cpu_cores += 0.5;
                    memory_mb += 64;
                }
                _ => {}
            }
        }

        ResourceRequirements {
            cpu_cores,
            memory_mb,
            network_bandwidth_mbps: network_mbps,
            disk_io_mb: 100,
        }
    }

    /// Get component health status
    pub async fn get_component_health(&self) -> Result<ComponentHealthStatus, Error> {
        let monitor = self.health_monitor.lock().await;
        Ok(monitor.get_health_status())
    }

    /// Get execution history
    pub async fn get_execution_history(&self) -> Result<Vec<ComponentExecutionRecord>, Error> {
        let registry = self.execution_registry.lock().await;
        Ok(registry.get_history())
    }
}

/// Result of component coordination
#[derive(Debug, Clone)]
pub struct CoordinationResult {
    pub success: bool,
    pub outputs: HashMap<String, serde_json::Value>,
    pub tool_calls_used: u32,
    pub duration_ms: u64,
    pub components_used: Vec<String>,
    pub errors: Vec<String>,
}

/// Component execution plan
#[derive(Debug, Clone)]
pub struct ComponentExecutionPlan {
    pub execution_order: Vec<String>,
    pub estimated_duration_ms: u64,
    pub resource_requirements: ResourceRequirements,
}

/// Resource requirements for component execution
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub cpu_cores: f64,
    pub memory_mb: u64,
    pub network_bandwidth_mbps: f64,
    pub disk_io_mb: u64,
}

/// Component health monitor
#[derive(Debug)]
struct ComponentHealthMonitor {
    component_health: HashMap<String, ComponentHealth>,
    last_check: std::time::Instant,
}

impl ComponentHealthMonitor {
    fn new() -> Self {
        Self {
            component_health: HashMap::new(),
            last_check: std::time::Instant::now(),
        }
    }

    fn record_execution(&mut self, _task_name: &str, components: &[String], success: bool) {
        for component in components {
            let health = self
                .component_health
                .entry(component.clone())
                .or_insert_with(ComponentHealth::new);

            health.record_execution(success);
        }
    }

    fn get_health_status(&self) -> ComponentHealthStatus {
        let mut overall_health: f64 = 1.0;
        let mut component_status = HashMap::new();

        for (component, health) in &self.component_health {
            let health_score = health.get_health_score();
            component_status.insert(component.clone(), health_score);
            overall_health = overall_health.min(health_score);
        }

        ComponentHealthStatus {
            overall_health,
            component_health: component_status,
            last_check: self.last_check.elapsed().as_secs(),
        }
    }
}

/// Individual component health
#[derive(Debug)]
struct ComponentHealth {
    total_executions: u32,
    successful_executions: u32,
    last_execution: Option<std::time::Instant>,
    #[allow(dead_code)]
    average_duration_ms: f64,
}

impl ComponentHealth {
    fn new() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            last_execution: None,
            average_duration_ms: 0.0,
        }
    }

    fn record_execution(&mut self, success: bool) {
        self.total_executions += 1;
        if success {
            self.successful_executions += 1;
        }
        self.last_execution = Some(std::time::Instant::now());
    }

    fn get_health_score(&self) -> f64 {
        if self.total_executions == 0 {
            return 1.0;
        }

        let success_rate = self.successful_executions as f64 / self.total_executions as f64;
        let recency_factor = if let Some(last) = self.last_execution {
            let elapsed = last.elapsed().as_secs();
            if elapsed < 300 {
                // 5 minutes
                1.0
            } else if elapsed < 1800 {
                // 30 minutes
                0.8
            } else {
                0.5
            }
        } else {
            0.7
        };

        success_rate * recency_factor
    }
}

/// Component health status
#[derive(Debug)]
pub struct ComponentHealthStatus {
    pub overall_health: f64,
    pub component_health: HashMap<String, f64>,
    pub last_check: u64,
}

/// Component execution registry
#[derive(Debug)]
struct ComponentExecutionRegistry {
    execution_history: Vec<ComponentExecutionRecord>,
    max_history: usize,
}

impl ComponentExecutionRegistry {
    fn new() -> Self {
        Self {
            execution_history: Vec::new(),
            max_history: 1000,
        }
    }

    fn register_execution(&mut self, record: ComponentExecutionRecord) {
        self.execution_history.push(record);

        // Maintain history limit
        if self.execution_history.len() > self.max_history {
            self.execution_history.remove(0);
        }
    }

    fn get_history(&self) -> Vec<ComponentExecutionRecord> {
        self.execution_history.clone()
    }
}

/// Component execution record
#[derive(Debug, Clone)]
pub struct ComponentExecutionRecord {
    pub task_name: String,
    pub timestamp: i64,
    pub components_used: Vec<String>,
    pub duration_ms: u64,
    pub success: bool,
    pub tool_calls_used: u32,
    pub outputs: HashMap<String, serde_json::Value>,
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordination_result_creation() {
        let result = CoordinationResult {
            success: true,
            outputs: HashMap::new(),
            tool_calls_used: 5,
            duration_ms: 1000,
            components_used: vec!["reasonkit-core".to_string()],
            errors: vec![],
        };

        assert!(result.success);
        assert_eq!(result.tool_calls_used, 5);
        assert_eq!(result.duration_ms, 1000);
    }

    #[test]
    fn test_resource_requirements() {
        let req = ResourceRequirements {
            cpu_cores: 2.5,
            memory_mb: 1024,
            network_bandwidth_mbps: 100.0,
            disk_io_mb: 200,
        };

        assert_eq!(req.cpu_cores, 2.5);
        assert_eq!(req.memory_mb, 1024);
    }
}
