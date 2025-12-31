//! # Task Graph Management
//!
//! This module implements a dependency graph system for managing complex multi-step
//! task execution with support for parallelization, prioritization, and monitoring.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
    Urgent = 5,
}

/// Task execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
    Cancelled,
}

/// Task type categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    General,
    ProtocolGeneration,
    CodeAnalysis,
    WebAutomation,
    MemoryManagement,
    EnterpriseWorkflow,
    MultiAgentCoordination,
}

/// A single task node in the dependency graph
#[derive(Debug, Clone)]
pub struct TaskNode {
    id: String,
    name: String,
    task_type: TaskType,
    priority: TaskPriority,
    status: TaskStatus,
    dependencies: HashSet<String>,
    description: String,
    config: TaskConfig,
    metadata: serde_json::Value,
    estimated_duration_ms: u64,
    required_components: Vec<String>,
    required_thinktools: Vec<String>,
    requires_m2_capability: bool,
    created_at: u64,
    started_at: Option<u64>,
    completed_at: Option<u64>,
    retry_count: u32,
    max_retries: u32,
}

impl TaskNode {
    pub fn new(
        id: String,
        name: String,
        task_type: TaskType,
        priority: TaskPriority,
        description: String,
    ) -> Self {
        let now = chrono::Utc::now().timestamp() as u64;

        Self {
            id,
            name,
            task_type,
            priority,
            status: TaskStatus::Pending,
            dependencies: HashSet::new(),
            description,
            config: TaskConfig::default(),
            metadata: serde_json::json!({}),
            estimated_duration_ms: 0,
            required_components: Vec::new(),
            required_thinktools: Vec::new(),
            requires_m2_capability: false,
            created_at: now,
            started_at: None,
            completed_at: None,
            retry_count: 0,
            max_retries: 3,
        }
    }

    pub fn with_dependency(mut self, dependency_id: &str) -> Self {
        self.dependencies.insert(dependency_id.to_string());
        self
    }

    pub fn with_config(mut self, config: TaskConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.estimated_duration_ms = duration_ms;
        self
    }

    pub fn with_components(mut self, components: Vec<String>) -> Self {
        self.required_components = components;
        self
    }

    pub fn with_thinktools(mut self, thinktools: Vec<String>) -> Self {
        self.required_thinktools = thinktools;
        self
    }

    pub fn requires_m2(mut self, requires: bool) -> Self {
        self.requires_m2_capability = requires;
        self
    }

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    // Getters
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn task_type(&self) -> TaskType {
        self.task_type
    }
    pub fn priority(&self) -> TaskPriority {
        self.priority
    }
    pub fn status(&self) -> TaskStatus {
        self.status
    }
    pub fn dependencies(&self) -> &HashSet<String> {
        &self.dependencies
    }
    pub fn description(&self) -> &str {
        &self.description
    }
    pub fn config(&self) -> &TaskConfig {
        &self.config
    }
    pub fn metadata(&self) -> &serde_json::Value {
        &self.metadata
    }
    pub fn estimated_duration_ms(&self) -> u64 {
        self.estimated_duration_ms
    }
    pub fn required_components(&self) -> &[String] {
        &self.required_components
    }
    pub fn required_thinktools(&self) -> &[String] {
        &self.required_thinktools
    }
    pub fn requires_m2_capability(&self) -> bool {
        self.requires_m2_capability
    }
    pub fn created_at(&self) -> u64 {
        self.created_at
    }
    pub fn started_at(&self) -> Option<u64> {
        self.started_at
    }
    pub fn completed_at(&self) -> Option<u64> {
        self.completed_at
    }
    pub fn retry_count(&self) -> u32 {
        self.retry_count
    }
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }
    pub fn requires_thinktool(&self) -> bool {
        !self.required_thinktools.is_empty()
    }

    // Status management
    pub fn mark_running(&mut self) {
        self.status = TaskStatus::Running;
        self.started_at = Some(chrono::Utc::now().timestamp() as u64);
    }

    pub fn mark_completed(&mut self) {
        self.status = TaskStatus::Completed;
        self.completed_at = Some(chrono::Utc::now().timestamp() as u64);
    }

    pub fn mark_failed(&mut self) {
        self.status = TaskStatus::Failed;
        self.completed_at = Some(chrono::Utc::now().timestamp() as u64);
    }

    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries && self.status == TaskStatus::Failed
    }
}

/// Configuration for a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub timeout_ms: u64,
    pub memory_limit_mb: u64,
    pub parallel_execution: bool,
    pub resource_requirements: ResourceRequirements,
    pub custom_parameters: HashMap<String, serde_json::Value>,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 300_000, // 5 minutes
            memory_limit_mb: 512,
            parallel_execution: false,
            resource_requirements: ResourceRequirements::default(),
            custom_parameters: HashMap::new(),
        }
    }
}

/// Resource requirements for task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: f64,
    pub memory_mb: u64,
    pub network_bandwidth_mbps: f64,
    pub disk_io_mb: u64,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1.0,
            memory_mb: 512,
            network_bandwidth_mbps: 10.0,
            disk_io_mb: 100,
        }
    }
}

/// Dependency graph for managing task relationships
pub type DependencyGraph = TaskGraph;

#[derive(Debug)]
pub struct TaskGraph {
    nodes: HashMap<String, TaskNode>,
    edges: HashMap<String, HashSet<String>>, // adjacency list: node -> dependent nodes
    reverse_edges: HashMap<String, HashSet<String>>, // reverse adjacency: node -> dependency nodes
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
        }
    }

    /// Add a task node to the graph
    pub fn add_node(&mut self, node: TaskNode) -> Result<(), Error> {
        let node_id = node.id().to_string();

        if self.nodes.contains_key(&node_id) {
            return Err(Error::Validation(format!(
                "Task node '{}' already exists",
                node_id
            )));
        }

        self.nodes.insert(node_id.clone(), node);
        self.edges.insert(node_id.clone(), HashSet::new());
        self.reverse_edges.insert(node_id.clone(), HashSet::new());

        Ok(())
    }

    /// Add a dependency relationship between tasks
    pub fn add_dependency(&mut self, from: &str, to: &str) -> Result<(), Error> {
        // Validate that both nodes exist
        if !self.nodes.contains_key(from) {
            return Err(Error::Validation(format!(
                "Source task '{}' does not exist",
                from
            )));
        }

        if !self.nodes.contains_key(to) {
            return Err(Error::Validation(format!(
                "Target task '{}' does not exist",
                to
            )));
        }

        // Add dependency edge
        self.edges
            .entry(from.to_string())
            .or_default()
            .insert(to.to_string());

        self.reverse_edges
            .entry(to.to_string())
            .or_default()
            .insert(from.to_string());

        Ok(())
    }

    /// Remove a task node from the graph
    pub fn remove_node(&mut self, node_id: &str) -> Result<(), Error> {
        if !self.nodes.contains_key(node_id) {
            return Err(Error::Validation(format!(
                "Task node '{}' does not exist",
                node_id
            )));
        }

        // Remove all edges to and from this node
        let dependents = self.edges.remove(node_id).unwrap_or_default();
        for dependent in dependents {
            self.reverse_edges
                .entry(dependent.clone())
                .or_default()
                .remove(node_id);
        }

        let dependencies = self.reverse_edges.remove(node_id).unwrap_or_default();
        for dependency in dependencies {
            self.edges
                .entry(dependency.clone())
                .or_default()
                .remove(node_id);
        }

        // Remove reverse edges
        self.reverse_edges.remove(node_id);

        // Remove the node
        self.nodes.remove(node_id);

        Ok(())
    }

    /// Get a task node by ID
    pub fn get_node(&self, node_id: &str) -> Option<&TaskNode> {
        self.nodes.get(node_id)
    }

    /// Get a mutable task node by ID
    pub fn get_node_mut(&mut self, node_id: &str) -> Option<&mut TaskNode> {
        self.nodes.get_mut(node_id)
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<&TaskNode> {
        self.nodes.values().collect()
    }

    /// Get all node IDs in the graph
    pub fn node_ids(&self) -> Vec<&String> {
        self.nodes.keys().collect()
    }

    /// Get dependent tasks (tasks that depend on this task)
    pub fn get_dependents(&self, node_id: &str) -> Option<&HashSet<String>> {
        self.edges.get(node_id)
    }

    /// Get dependency tasks (tasks that this task depends on)
    pub fn get_dependencies(&self, node_id: &str) -> Option<&HashSet<String>> {
        self.reverse_edges.get(node_id)
    }

    /// Check if the graph has cycles
    pub fn has_cycles(&self) -> bool {
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();

        for node_id in self.nodes.keys() {
            if !visited.contains(node_id)
                && self.has_cycle_dfs(node_id, &mut visited, &mut recursion_stack)
            {
                return true;
            }
        }

        false
    }

    /// Perform DFS cycle detection
    fn has_cycle_dfs(
        &self,
        node_id: &str,
        visited: &mut HashSet<String>,
        recursion_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node_id.to_string());
        recursion_stack.insert(node_id.to_string());

        if let Some(dependents) = self.edges.get(node_id) {
            for dependent in dependents {
                if !visited.contains(dependent) {
                    if self.has_cycle_dfs(dependent, visited, recursion_stack) {
                        return true;
                    }
                } else if recursion_stack.contains(dependent) {
                    return true;
                }
            }
        }

        recursion_stack.remove(node_id);
        false
    }

    /// Validate the graph structure
    pub fn validate(&self) -> Result<(), Error> {
        // Check for cycles
        if self.has_cycles() {
            return Err(Error::Validation(
                "Task dependency graph contains cycles".to_string(),
            ));
        }

        // Check for orphaned dependencies
        for (node_id, dependencies) in &self.reverse_edges {
            for dependency in dependencies {
                if !self.nodes.contains_key(dependency) {
                    return Err(Error::Validation(format!(
                        "Task '{}' depends on non-existent task '{}'",
                        node_id, dependency
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get topological sort of nodes
    pub fn topological_sort(&self) -> Result<Vec<String>, Error> {
        // Validate the graph first
        self.validate()?;

        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut result: Vec<String> = Vec::new();

        // Calculate in-degrees
        for node_id in self.nodes.keys() {
            let in_degree_count = self
                .reverse_edges
                .get(node_id)
                .map(|deps| deps.len())
                .unwrap_or(0);
            in_degree.insert(node_id.clone(), in_degree_count);
        }

        // Find nodes with zero in-degree
        for (node_id, degree) in &in_degree {
            if *degree == 0 {
                queue.push_back(node_id.clone());
            }
        }

        // Process nodes
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id.clone());

            // Reduce in-degree of dependent nodes
            if let Some(dependents) = self.edges.get(&node_id) {
                for dependent in dependents {
                    if let Some(degree) = in_degree.get_mut(dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dependent.clone());
                        }
                    }
                }
            }
        }

        // Check if all nodes were processed
        if result.len() != self.nodes.len() {
            return Err(Error::Validation(
                "Unable to topologically sort - graph may have cycles".to_string(),
            ));
        }

        Ok(result)
    }

    /// Get ready-to-execute tasks (all dependencies satisfied)
    pub fn get_ready_tasks(&self) -> Vec<String> {
        let mut ready_tasks = Vec::new();

        for (node_id, node) in &self.nodes {
            if node.status() == TaskStatus::Pending {
                // Check if all dependencies are completed
                let mut all_deps_completed = true;
                if let Some(dependencies) = self.reverse_edges.get(node_id) {
                    for dep_id in dependencies {
                        if let Some(dep_node) = self.nodes.get(dep_id) {
                            if dep_node.status() != TaskStatus::Completed {
                                all_deps_completed = false;
                                break;
                            }
                        }
                    }
                }

                if all_deps_completed {
                    ready_tasks.push(node_id.clone());
                }
            }
        }

        // Sort by priority (higher priority first)
        ready_tasks.sort_by(|a, b| {
            let node_a = self.nodes.get(a).unwrap();
            let node_b = self.nodes.get(b).unwrap();
            node_b.priority().cmp(&node_a.priority())
        });

        ready_tasks
    }

    /// Get execution statistics
    pub fn get_statistics(&self) -> TaskGraphStatistics {
        let total_nodes = self.nodes.len();
        let mut completed_nodes = 0;
        let mut running_nodes = 0;
        let mut pending_nodes = 0;
        let mut failed_nodes = 0;

        let mut total_estimated_duration = 0u64;
        let mut total_critical_path_duration = 0u64;

        for node in self.nodes.values() {
            match node.status() {
                TaskStatus::Completed => completed_nodes += 1,
                TaskStatus::Running => running_nodes += 1,
                TaskStatus::Pending => pending_nodes += 1,
                TaskStatus::Failed => failed_nodes += 1,
                _ => {}
            }

            total_estimated_duration += node.estimated_duration_ms();
        }

        // Calculate critical path (simplified - longest path through dependencies)
        if let Ok(topological_order) = self.topological_sort() {
            let mut node_completion_times: HashMap<String, u64> = HashMap::new();

            for node_id in topological_order {
                let mut max_dep_time = 0u64;
                if let Some(dependencies) = self.reverse_edges.get(&node_id) {
                    for dep_id in dependencies {
                        if let Some(dep_time) = node_completion_times.get(dep_id) {
                            max_dep_time = max_dep_time.max(*dep_time);
                        }
                    }
                }

                let node = self.nodes.get(&node_id).unwrap();
                let completion_time = max_dep_time + node.estimated_duration_ms();
                node_completion_times.insert(node_id.clone(), completion_time);
                total_critical_path_duration = total_critical_path_duration.max(completion_time);
            }
        }

        TaskGraphStatistics {
            total_nodes,
            completed_nodes,
            running_nodes,
            pending_nodes,
            failed_nodes,
            total_estimated_duration,
            critical_path_duration: total_critical_path_duration,
            completion_percentage: if total_nodes > 0 {
                (completed_nodes as f64 / total_nodes as f64) * 100.0
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug)]
pub struct TaskGraphStatistics {
    pub total_nodes: usize,
    pub completed_nodes: usize,
    pub running_nodes: usize,
    pub pending_nodes: usize,
    pub failed_nodes: usize,
    pub total_estimated_duration: u64,
    pub critical_path_duration: u64,
    pub completion_percentage: f64,
}

impl fmt::Display for TaskGraphStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TaskGraph Statistics:\n\
             - Total Tasks: {}\n\
             - Completed: {} ({:.1}%)\n\
             - Running: {}\n\
             - Pending: {}\n\
             - Failed: {}\n\
             - Estimated Duration: {:.1} seconds\n\
             - Critical Path: {:.1} seconds",
            self.total_nodes,
            self.completed_nodes,
            self.completion_percentage,
            self.running_nodes,
            self.pending_nodes,
            self.failed_nodes,
            self.total_estimated_duration as f64 / 1000.0,
            self.critical_path_duration as f64 / 1000.0
        )
    }
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "Pending"),
            TaskStatus::Running => write!(f, "Running"),
            TaskStatus::Completed => write!(f, "Completed"),
            TaskStatus::Failed => write!(f, "Failed"),
            TaskStatus::Skipped => write!(f, "Skipped"),
            TaskStatus::Cancelled => write!(f, "Cancelled"),
        }
    }
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskPriority::Low => write!(f, "Low"),
            TaskPriority::Normal => write!(f, "Normal"),
            TaskPriority::High => write!(f, "High"),
            TaskPriority::Critical => write!(f, "Critical"),
            TaskPriority::Urgent => write!(f, "Urgent"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_node_creation() {
        let task = TaskNode::new(
            "test-1".to_string(),
            "Test Task".to_string(),
            TaskType::ProtocolGeneration,
            TaskPriority::High,
            "A test task".to_string(),
        );

        assert_eq!(task.id(), "test-1");
        assert_eq!(task.name(), "Test Task");
        assert_eq!(task.task_type(), TaskType::ProtocolGeneration);
        assert_eq!(task.priority(), TaskPriority::High);
        assert_eq!(task.status(), TaskStatus::Pending);
        assert_eq!(task.dependencies().len(), 0);
    }

    #[test]
    fn test_task_node_with_dependencies() {
        let task = TaskNode::new(
            "test-1".to_string(),
            "Test Task".to_string(),
            TaskType::CodeAnalysis,
            TaskPriority::Normal,
            "A test task".to_string(),
        )
        .with_dependency("dep-1")
        .with_dependency("dep-2");

        assert_eq!(task.dependencies().len(), 2);
        assert!(task.dependencies().contains("dep-1"));
        assert!(task.dependencies().contains("dep-2"));
    }

    #[test]
    fn test_task_graph_creation() {
        let mut graph = TaskGraph::new();

        let task1 = TaskNode::new(
            "task-1".to_string(),
            "Task 1".to_string(),
            TaskType::ProtocolGeneration,
            TaskPriority::Normal,
            "First task".to_string(),
        );

        let task2 = TaskNode::new(
            "task-2".to_string(),
            "Task 2".to_string(),
            TaskType::CodeAnalysis,
            TaskPriority::High,
            "Second task".to_string(),
        );

        assert!(graph.add_node(task1).is_ok());
        assert!(graph.add_node(task2).is_ok());
        assert_eq!(graph.nodes().len(), 2);
    }

    #[test]
    fn test_task_graph_dependencies() {
        let mut graph = TaskGraph::new();

        let task1 = TaskNode::new(
            "task-1".to_string(),
            "Task 1".to_string(),
            TaskType::ProtocolGeneration,
            TaskPriority::Normal,
            "First task".to_string(),
        );

        let task2 = TaskNode::new(
            "task-2".to_string(),
            "Task 2".to_string(),
            TaskType::CodeAnalysis,
            TaskPriority::High,
            "Second task".to_string(),
        );

        graph.add_node(task1).unwrap();
        graph.add_node(task2).unwrap();
        assert!(graph.add_dependency("task-1", "task-2").is_ok());
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = TaskGraph::new();

        let task1 = TaskNode::new(
            "task-1".to_string(),
            "Task 1".to_string(),
            TaskType::ProtocolGeneration,
            TaskPriority::Normal,
            "First task".to_string(),
        );

        let task2 = TaskNode::new(
            "task-2".to_string(),
            "Task 2".to_string(),
            TaskType::CodeAnalysis,
            TaskPriority::High,
            "Second task".to_string(),
        );

        graph.add_node(task1).unwrap();
        graph.add_node(task2).unwrap();
        graph.add_dependency("task-1", "task-2").unwrap();
        graph.add_dependency("task-2", "task-1").unwrap();

        assert!(graph.has_cycles());
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = TaskGraph::new();

        let task1 = TaskNode::new(
            "task-1".to_string(),
            "Task 1".to_string(),
            TaskType::ProtocolGeneration,
            TaskPriority::Normal,
            "First task".to_string(),
        );

        let task2 = TaskNode::new(
            "task-2".to_string(),
            "Task 2".to_string(),
            TaskType::CodeAnalysis,
            TaskPriority::High,
            "Second task".to_string(),
        );

        let task3 = TaskNode::new(
            "task-3".to_string(),
            "Task 3".to_string(),
            TaskType::WebAutomation,
            TaskPriority::Normal,
            "Third task".to_string(),
        );

        graph.add_node(task1).unwrap();
        graph.add_node(task2).unwrap();
        graph.add_node(task3).unwrap();
        graph.add_dependency("task-1", "task-2").unwrap();
        graph.add_dependency("task-2", "task-3").unwrap();

        let order = graph.topological_sort().unwrap();
        assert_eq!(order.len(), 3);

        // Verify that task-1 comes before task-2, and task-2 comes before task-3
        let task1_pos = order.iter().position(|id| id == "task-1").unwrap();
        let task2_pos = order.iter().position(|id| id == "task-2").unwrap();
        let task3_pos = order.iter().position(|id| id == "task-3").unwrap();

        assert!(task1_pos < task2_pos);
        assert!(task2_pos < task3_pos);
    }
}
