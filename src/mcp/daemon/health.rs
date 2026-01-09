//! Health Monitoring
//!
//! Monitors MCP server health and triggers reconnection on failure.

use crate::mcp::McpRegistry;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Health monitor for MCP servers
pub struct HealthMonitor {
    interval: Duration,
    registry: Arc<RwLock<McpRegistry>>,
}

impl HealthMonitor {
    /// Create new health monitor
    pub fn new(registry: Arc<RwLock<McpRegistry>>, interval_secs: u64) -> Self {
        Self {
            interval: Duration::from_secs(interval_secs),
            registry,
        }
    }

    /// Run health monitoring loop (blocking)
    pub async fn run(&self) {
        info!("Health monitor started (interval: {:?})", self.interval);

        let mut interval = tokio::time::interval(self.interval);

        loop {
            interval.tick().await;

            let registry = self.registry.read().await;
            let servers = registry.list_servers().await;

            for server in servers {
                // Ping server
                match registry.ping_server(&server.id).await {
                    Ok(true) => {
                        // Server healthy
                        if let Some(health) = &server.last_health_check {
                            if health.status != crate::mcp::registry::HealthStatus::Healthy {
                                info!("Server {} recovered", server.name);
                            }
                        }
                    }
                    Ok(false) | Err(_) => {
                        // Server unhealthy
                        warn!("Server {} unhealthy, attempting reconnect", server.name);

                        if let Err(e) = registry.reconnect_server(&server.id).await {
                            error!("Failed to reconnect {}: {}", server.name, e);
                        }
                    }
                }
            }
        }
    }
}
