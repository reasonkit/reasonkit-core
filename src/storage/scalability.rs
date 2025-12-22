//! Scalability assessment and production deployment strategies
//!
//! This module provides comprehensive scalability analysis and deployment
//! recommendations for ReasonKit database systems in production environments.

use crate::storage::{QdrantConnectionConfig, QdrantSecurityConfig, EmbeddingCacheConfig, AccessControlConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Scalability assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityAssessment {
    /// Current workload analysis
    pub workload_analysis: WorkloadAnalysis,
    /// Horizontal scaling recommendations
    pub horizontal_scaling: HorizontalScalingPlan,
    /// Replication strategy
    pub replication_strategy: ReplicationStrategy,
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
    /// Performance projections
    pub performance_projections: PerformanceProjections,
    /// Cost analysis
    pub cost_analysis: CostAnalysis,
}

/// Workload analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadAnalysis {
    /// Estimated daily document ingestion
    pub daily_documents: usize,
    /// Estimated daily queries
    pub daily_queries: usize,
    /// Peak concurrent users
    pub peak_concurrent_users: usize,
    /// Average document size in bytes
    pub avg_document_size_bytes: usize,
    /// Average embedding vector size
    pub avg_embedding_size: usize,
    /// Read/write ratio
    pub read_write_ratio: f32,
    /// Data retention period in days
    pub retention_days: usize,
}

/// Horizontal scaling recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizontalScalingPlan {
    /// Recommended number of Qdrant nodes
    pub qdrant_nodes: usize,
    /// Recommended number of application servers
    pub app_servers: usize,
    /// Sharding strategy
    pub sharding_strategy: ShardingStrategy,
    /// Auto-scaling configuration
    pub auto_scaling: AutoScalingConfig,
    /// Geographic distribution
    pub geographic_distribution: Vec<String>,
}

/// Sharding strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingStrategy {
    /// Shard by document type
    pub shard_by_document_type: bool,
    /// Shard by time periods
    pub shard_by_time: bool,
    /// Shard by user/organization
    pub shard_by_user: bool,
    /// Number of shards per node
    pub shards_per_node: usize,
    /// Replication factor per shard
    pub replication_factor: usize,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Minimum number of nodes
    pub min_nodes: usize,
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Scale up threshold (CPU usage %)
    pub scale_up_threshold: f32,
    /// Scale down threshold (CPU usage %)
    pub scale_down_threshold: f32,
    /// Cooldown period in seconds
    pub cooldown_seconds: u64,
}

/// Replication strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationStrategy {
    /// Replication factor
    pub replication_factor: usize,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    /// Write quorum
    pub write_quorum: usize,
    /// Read quorum
    pub read_quorum: usize,
    /// Cross-region replication
    pub cross_region_replication: bool,
    /// Backup frequency in hours
    pub backup_frequency_hours: u64,
}

/// Consistency levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Strong consistency
    Strong,
    /// Eventual consistency
    Eventual,
    /// Bounded staleness
    BoundedStaleness,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Load balancer type
    pub load_balancer_type: LoadBalancerType,
    /// Health check configuration
    pub health_checks: HealthCheckConfig,
    /// Session affinity
    pub session_affinity: bool,
    /// Geographic routing
    pub geographic_routing: bool,
}

/// Load balancer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancerType {
    /// Layer 4 load balancing
    Layer4,
    /// Layer 7 load balancing
    Layer7,
    /// DNS-based load balancing
    Dns,
    /// Client-side load balancing
    ClientSide,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval in seconds
    pub interval_seconds: u64,
    /// Health check timeout in seconds
    pub timeout_seconds: u64,
    /// Unhealthy threshold
    pub unhealthy_threshold: usize,
    /// Healthy threshold
    pub healthy_threshold: usize,
}

/// Performance projections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProjections {
    /// Projected queries per second
    pub projected_qps: usize,
    /// Projected latency percentiles (p50, p95, p99)
    pub projected_latency_ms: LatencyPercentiles,
    /// Storage requirements in TB
    pub storage_requirements_tb: f32,
    /// Network bandwidth requirements in Gbps
    pub network_bandwidth_gbps: f32,
    /// CPU core requirements
    pub cpu_cores_required: usize,
    /// Memory requirements in GB
    pub memory_required_gb: usize,
}

/// Latency percentiles in milliseconds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f32,
    pub p95: f32,
    pub p99: f32,
}

/// Cost analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    /// Monthly infrastructure cost
    pub monthly_infrastructure_cost: f32,
    /// Monthly storage cost
    pub monthly_storage_cost: f32,
    /// Monthly network cost
    pub monthly_network_cost: f32,
    /// Cost per 1M queries
    pub cost_per_million_queries: f32,
    /// Break-even analysis
    pub break_even_analysis: BreakEvenAnalysis,
}

/// Break-even analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakEvenAnalysis {
    /// Monthly revenue needed to break even
    pub monthly_revenue_needed: f32,
    /// Customer acquisition cost
    pub customer_acquisition_cost: f32,
    /// Customer lifetime value
    pub customer_lifetime_value: f32,
}

/// Scalability assessment engine
pub struct ScalabilityAssessor;

impl ScalabilityAssessor {
    /// Perform comprehensive scalability assessment
    pub fn assess_scalability(workload: WorkloadAnalysis) -> ScalabilityAssessment {
        let horizontal_scaling = Self::design_horizontal_scaling(&workload);
        let replication_strategy = Self::design_replication_strategy(&workload);
        let load_balancing = Self::design_load_balancing(&workload);
        let performance_projections = Self::project_performance(&workload, &horizontal_scaling);
        let cost_analysis = Self::analyze_costs(&workload, &horizontal_scaling);

        ScalabilityAssessment {
            workload_analysis: workload,
            horizontal_scaling,
            replication_strategy,
            load_balancing,
            performance_projections,
            cost_analysis,
        }
    }

    /// Design horizontal scaling plan
    fn design_horizontal_scaling(workload: &WorkloadAnalysis) -> HorizontalScalingPlan {
        // Calculate required Qdrant nodes based on workload
        let documents_per_day = workload.daily_documents;
        let queries_per_day = workload.daily_queries;

        // Rough estimates: 10k documents/day per node, 100k queries/day per node
        let qdrant_nodes = ((documents_per_day / 10000).max(1) + (queries_per_day / 100000).max(1)) as usize;

        // Application servers: 1 per 50 concurrent users
        let app_servers = (workload.peak_concurrent_users / 50).max(1);

        let sharding_strategy = ShardingStrategy {
            shard_by_document_type: true,
            shard_by_time: workload.retention_days > 365, // Shard by time if retention > 1 year
            shard_by_user: workload.peak_concurrent_users > 1000, // Multi-tenant sharding
            shards_per_node: 4,
            replication_factor: 3,
        };

        let auto_scaling = AutoScalingConfig {
            min_nodes: qdrant_nodes.min(3),
            max_nodes: (qdrant_nodes * 3).max(10),
            scale_up_threshold: 70.0,
            scale_down_threshold: 30.0,
            cooldown_seconds: 300,
        };

        HorizontalScalingPlan {
            qdrant_nodes,
            app_servers,
            sharding_strategy,
            auto_scaling,
            geographic_distribution: vec!["us-east-1".to_string(), "us-west-2".to_string()], // Default regions
        }
    }

    /// Design replication strategy
    fn design_replication_strategy(workload: &WorkloadAnalysis) -> ReplicationStrategy {
        let replication_factor = if workload.daily_documents > 100000 { 5 } else { 3 };

        let consistency_level = if workload.read_write_ratio > 0.8 {
            ConsistencyLevel::Strong // High read ratio needs strong consistency
        } else {
            ConsistencyLevel::Eventual
        };

        ReplicationStrategy {
            replication_factor,
            consistency_level,
            write_quorum: replication_factor / 2 + 1,
            read_quorum: 1, // Allow stale reads for performance
            cross_region_replication: workload.peak_concurrent_users > 10000,
            backup_frequency_hours: 24,
        }
    }

    /// Design load balancing configuration
    fn design_load_balancing(workload: &WorkloadAnalysis) -> LoadBalancingConfig {
        let load_balancer_type = if workload.peak_concurrent_users > 10000 {
            LoadBalancerType::Layer7
        } else {
            LoadBalancerType::Layer4
        };

        let health_checks = HealthCheckConfig {
            interval_seconds: 30,
            timeout_seconds: 5,
            unhealthy_threshold: 3,
            healthy_threshold: 2,
        };

        LoadBalancingConfig {
            load_balancer_type,
            health_checks,
            session_affinity: workload.peak_concurrent_users < 1000, // Sticky sessions for small deployments
            geographic_routing: workload.peak_concurrent_users > 5000,
        }
    }

    /// Project performance metrics
    fn project_performance(workload: &WorkloadAnalysis, scaling: &HorizontalScalingPlan) -> PerformanceProjections {
        let total_nodes = scaling.qdrant_nodes;
        let queries_per_second = workload.daily_queries / 86400; // Convert daily to per second

        // Estimate QPS per node
        let qps_per_node = 1000; // Conservative estimate
        let projected_qps = qps_per_node * total_nodes;

        // Latency estimates based on node count
        let base_latency = 50.0; // Base p50 latency in ms
        let latency_multiplier = 1.0 / (total_nodes as f32).sqrt(); // More nodes = lower latency
        let p50 = base_latency * latency_multiplier;
        let p95 = p50 * 2.0;
        let p99 = p50 * 5.0;

        // Storage requirements
        let daily_data_gb = (workload.daily_documents * workload.avg_document_size_bytes) as f32 / 1_000_000_000.0;
        let total_storage_tb = (daily_data_gb * workload.retention_days as f32) / 1000.0;

        // Network bandwidth (rough estimate)
        let network_bandwidth_gbps = (queries_per_second as f32 * 0.01).max(1.0); // 10KB per query

        // Resource requirements
        let cpu_cores_required = total_nodes * 8; // 8 cores per node
        let memory_required_gb = total_nodes * 32; // 32GB per node

        PerformanceProjections {
            projected_qps,
            projected_latency_ms: LatencyPercentiles { p50, p95, p99 },
            storage_requirements_tb: total_storage_tb,
            network_bandwidth_gbps,
            cpu_cores_required,
            memory_required_gb,
        }
    }

    /// Analyze costs
    fn analyze_costs(workload: &WorkloadAnalysis, scaling: &HorizontalScalingPlan) -> CostAnalysis {
        let nodes = scaling.qdrant_nodes;

        // Rough cost estimates (AWS pricing)
        let cost_per_node_per_month = 500.0; // EC2 + EBS + networking
        let monthly_infrastructure_cost = nodes as f32 * cost_per_node_per_month;

        let storage_cost_per_tb_per_month = 50.0; // S3 + backups
        let monthly_storage_cost = 1.0 * storage_cost_per_tb_per_month; // Assume 1TB initially

        let network_cost_per_gbps_per_month = 100.0;
        let monthly_network_cost = 1.0 * network_cost_per_gbps_per_month; // Assume 1Gbps

        let queries_per_month = workload.daily_queries * 30;
        let cost_per_million_queries = 0.5; // $0.50 per million queries
        let query_cost = (queries_per_month as f32 / 1_000_000.0) * cost_per_million_queries;

        let total_monthly_cost = monthly_infrastructure_cost + monthly_storage_cost + monthly_network_cost + query_cost;

        // Break-even analysis (simplified)
        let monthly_revenue_needed = total_monthly_cost * 1.5; // 50% margin
        let customer_acquisition_cost = 100.0; // Per customer
        let customer_lifetime_value = monthly_revenue_needed / 100.0; // Assume 100 customers

        CostAnalysis {
            monthly_infrastructure_cost,
            monthly_storage_cost,
            monthly_network_cost,
            cost_per_million_queries,
            break_even_analysis: BreakEvenAnalysis {
                monthly_revenue_needed,
                customer_acquisition_cost,
                customer_lifetime_value,
            },
        }
    }
}

/// Generate production-ready configuration
pub fn generate_production_config(assessment: &ScalabilityAssessment) -> ProductionConfig {
    ProductionConfig {
        qdrant_config: QdrantConnectionConfig {
            max_connections: assessment.horizontal_scaling.qdrant_nodes * 100,
            connect_timeout_secs: 60,
            request_timeout_secs: 120,
            health_check_interval_secs: 60,
            max_idle_secs: 300,
            security: QdrantSecurityConfig {
                api_key: Some("production-api-key".to_string()),
                tls_enabled: true,
                ca_cert_path: Some("/etc/ssl/certs/ca-certificates.crt".to_string()),
                client_cert_path: None,
                client_key_path: None,
                skip_tls_verify: false,
            },
        },
        cache_config: EmbeddingCacheConfig {
            max_size: 50000, // Larger cache for production
            ttl_secs: 7200, // 2 hours
        },
        access_config: AccessControlConfig {
            read_level: crate::storage::AccessLevel::Read,
            write_level: crate::storage::AccessLevel::ReadWrite,
            delete_level: crate::storage::AccessLevel::ReadWrite,
            admin_level: crate::storage::AccessLevel::Admin,
            enable_audit_log: true,
        },
        deployment_config: DeploymentConfig {
            replicas: assessment.replication_strategy.replication_factor,
            shards: assessment.horizontal_scaling.sharding_strategy.shards_per_node,
            regions: assessment.horizontal_scaling.geographic_distribution.clone(),
        },
    }
}

/// Production configuration bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    pub qdrant_config: QdrantConnectionConfig,
    pub cache_config: EmbeddingCacheConfig,
    pub access_config: AccessControlConfig,
    pub deployment_config: DeploymentConfig,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    pub replicas: usize,
    pub shards: usize,
    pub regions: Vec<String>,
}

/// Quick scalability assessment for common scenarios
pub fn quick_assessment(scenario: ScalabilityScenario) -> ScalabilityAssessment {
    let workload = match scenario {
        ScalabilityScenario::SmallTeam => WorkloadAnalysis {
            daily_documents: 100,
            daily_queries: 1000,
            peak_concurrent_users: 10,
            avg_document_size_bytes: 100000, // 100KB
            avg_embedding_size: 768,
            read_write_ratio: 0.9,
            retention_days: 365,
        },
        ScalabilityScenario::GrowingStartup => WorkloadAnalysis {
            daily_documents: 1000,
            daily_queries: 10000,
            peak_concurrent_users: 100,
            avg_document_size_bytes: 500000, // 500KB
            avg_embedding_size: 1024,
            read_write_ratio: 0.8,
            retention_days: 730,
        },
        ScalabilityScenario::Enterprise => WorkloadAnalysis {
            daily_documents: 10000,
            daily_queries: 100000,
            peak_concurrent_users: 1000,
            avg_document_size_bytes: 2000000, // 2MB
            avg_embedding_size: 1536,
            read_write_ratio: 0.7,
            retention_days: 2555, // 7 years
        },
        ScalabilityScenario::LargeScale => WorkloadAnalysis {
            daily_documents: 100000,
            daily_queries: 1000000,
            peak_concurrent_users: 10000,
            avg_document_size_bytes: 5000000, // 5MB
            avg_embedding_size: 2048,
            read_write_ratio: 0.6,
            retention_days: 2555,
        },
    };

    ScalabilityAssessor::assess_scalability(workload)
}

/// Predefined scalability scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalabilityScenario {
    /// Small team (1-10 users)
    SmallTeam,
    /// Growing startup (10-100 users)
    GrowingStartup,
    /// Enterprise deployment (100-1000 users)
    Enterprise,
    /// Large-scale deployment (1000+ users)
    LargeScale,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_assessment_small_team() {
        let assessment = quick_assessment(ScalabilityScenario::SmallTeam);
        assert_eq!(assessment.workload_analysis.daily_documents, 100);
        assert!(assessment.horizontal_scaling.qdrant_nodes >= 1);
        assert!(assessment.performance_projections.projected_qps > 0);
    }

    #[test]
    fn test_scalability_assessor() {
        let workload = WorkloadAnalysis {
            daily_documents: 1000,
            daily_queries: 10000,
            peak_concurrent_users: 50,
            avg_document_size_bytes: 100000,
            avg_embedding_size: 768,
            read_write_ratio: 0.8,
            retention_days: 365,
        };

        let assessment = ScalabilityAssessor::assess_scalability(workload);
        assert!(assessment.horizontal_scaling.qdrant_nodes >= 1);
        assert!(assessment.replication_strategy.replication_factor >= 3);
    }
}