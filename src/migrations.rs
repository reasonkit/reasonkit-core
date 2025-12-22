//! Database migration system for ReasonKit Core
//!
//! Provides version-controlled schema changes and data migrations
//! for all database backends (Qdrant, Sled, etc.).

use crate::{Result, Error};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Migration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationMeta {
    /// Unique migration ID
    pub id: Uuid,
    /// Migration version (timestamp-based)
    pub version: u64,
    /// Human-readable name
    pub name: String,
    /// Description of what this migration does
    pub description: String,
    /// When this migration was created
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Which database backends this migration applies to
    pub backends: Vec<String>,
}

/// Migration trait that all migrations must implement
#[async_trait]
pub trait Migration: Send + Sync {
    /// Get migration metadata
    fn meta(&self) -> MigrationMeta;

    /// Apply the migration (up)
    async fn up(&self, ctx: &mut MigrationContext) -> Result<()>;

    /// Rollback the migration (down)
    async fn down(&self, ctx: &mut MigrationContext) -> Result<()>;
}

/// Context passed to migrations during execution
pub struct MigrationContext {
    /// Database connections/backends
    pub backends: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
    /// Migration working directory
    pub work_dir: PathBuf,
    /// Additional context data
    pub data: HashMap<String, serde_json::Value>,
}

impl MigrationContext {
    /// Create a new migration context
    pub fn new(work_dir: PathBuf) -> Self {
        Self {
            backends: HashMap::new(),
            work_dir,
            data: HashMap::new(),
        }
    }

    /// Add a backend to the context
    pub fn with_backend<T: 'static + Send + Sync>(
        mut self,
        name: &str,
        backend: T,
    ) -> Self {
        self.backends.insert(name.to_string(), Box::new(backend));
        self
    }

    /// Get a backend from the context
    pub fn get_backend<T: 'static>(&self, name: &str) -> Result<&T> {
        self.backends
            .get(name)
            .and_then(|b| b.downcast_ref::<T>())
            .ok_or_else(|| Error::not_found(format!("Backend '{}' not found or wrong type", name)))
    }

    /// Get a backend mutably from the context
    pub fn get_backend_mut<T: 'static>(&mut self, name: &str) -> Result<&mut T> {
        self.backends
            .get_mut(name)
            .and_then(|b| b.downcast_mut::<T>())
            .ok_or_else(|| Error::not_found(format!("Backend '{}' not found or wrong type", name)))
    }
}

/// Migration registry for tracking available migrations
pub struct MigrationRegistry {
    migrations: HashMap<Uuid, Box<dyn Migration>>,
    versions: HashMap<u64, Uuid>,
}

impl MigrationRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            migrations: HashMap::new(),
            versions: HashMap::new(),
        }
    }

    /// Register a migration
    pub fn register<M: Migration + 'static>(&mut self, migration: M) -> Result<()> {
        let meta = migration.meta();
        let id = meta.id;
        let version = meta.version;

        if self.versions.contains_key(&version) {
            return Err(Error::invalid_input(format!(
                "Migration version {} already exists",
                version
            )));
        }

        self.migrations.insert(id, Box::new(migration));
        self.versions.insert(version, id);

        Ok(())
    }

    /// Get a migration by ID
    pub fn get(&self, id: &Uuid) -> Option<&dyn Migration> {
        self.migrations.get(id).map(|m| m.as_ref())
    }

    /// Get all migrations sorted by version
    pub fn all_sorted(&self) -> Vec<&dyn Migration> {
        let mut migrations: Vec<&dyn Migration> = self.migrations.values().map(|m| m.as_ref()).collect();
        migrations.sort_by_key(|m| m.meta().version);
        migrations
    }

    /// Get migrations between versions (inclusive)
    pub fn between(&self, from_version: u64, to_version: u64) -> Vec<&dyn Migration> {
        self.all_sorted()
            .into_iter()
            .filter(|m| {
                let v = m.meta().version;
                v >= from_version && v <= to_version
            })
            .collect()
    }
}

impl Default for MigrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Migration runner for executing migrations
pub struct MigrationRunner {
    registry: MigrationRegistry,
    work_dir: PathBuf,
}

impl MigrationRunner {
    /// Create a new migration runner
    pub fn new(registry: MigrationRegistry, work_dir: PathBuf) -> Self {
        Self { registry, work_dir }
    }

    /// Run migrations up to a target version
    pub async fn migrate_up(
        &self,
        target_version: Option<u64>,
        backends: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
    ) -> Result<Vec<MigrationResult>> {
        let current_version = self.get_current_version().await?;
        let target_version = target_version.unwrap_or(u64::MAX);

        let migrations = self.registry.between(current_version + 1, target_version);
        let mut results = Vec::new();

        for migration in migrations {
            let mut ctx = MigrationContext::new(self.work_dir.clone());
            for (name, backend) in &backends {
                // Clone the backend for each migration (simplified - in practice you'd want Arc or similar)
                ctx.backends.insert(name.clone(), backend.clone());
            }

            let start_time = std::time::Instant::now();
            let result = migration.up(&mut ctx).await;
            let duration = start_time.elapsed();

            let migration_result = MigrationResult {
                migration_id: migration.meta().id,
                migration_name: migration.meta().name.clone(),
                direction: MigrationDirection::Up,
                success: result.is_ok(),
                error: result.err(),
                duration,
            };

            results.push(migration_result.clone());

            if !migration_result.success {
                break; // Stop on first failure
            }

            // Update current version
            self.set_current_version(migration.meta().version).await?;
        }

        Ok(results)
    }

    /// Rollback migrations down to a target version
    pub async fn migrate_down(
        &self,
        target_version: u64,
        backends: HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
    ) -> Result<Vec<MigrationResult>> {
        let current_version = self.get_current_version().await?;
        if current_version <= target_version {
            return Ok(Vec::new());
        }

        let migrations = self.registry.between(target_version + 1, current_version);
        let mut results = Vec::new();

        // Run migrations in reverse order for rollback
        for migration in migrations.into_iter().rev() {
            let mut ctx = MigrationContext::new(self.work_dir.clone());
            for (name, backend) in &backends {
                ctx.backends.insert(name.clone(), backend.clone());
            }

            let start_time = std::time::Instant::now();
            let result = migration.down(&mut ctx).await;
            let duration = start_time.elapsed();

            let migration_result = MigrationResult {
                migration_id: migration.meta().id,
                migration_name: migration.meta().name.clone(),
                direction: MigrationDirection::Down,
                success: result.is_ok(),
                error: result.err(),
                duration,
            };

            results.push(migration_result.clone());

            if !migration_result.success {
                break; // Stop on first failure
            }

            // Update current version
            self.set_current_version(migration.meta().version - 1).await?;
        }

        Ok(results)
    }

    /// Get current database version
    async fn get_current_version(&self) -> Result<u64> {
        let version_file = self.work_dir.join("current_version");
        if !version_file.exists() {
            return Ok(0);
        }

        let content = tokio::fs::read_to_string(&version_file).await
            .map_err(|e| Error::io(format!("Failed to read version file: {}", e)))?;

        content.trim().parse::<u64>()
            .map_err(|e| Error::parse(format!("Invalid version format: {}", e)))
    }

    /// Set current database version
    async fn set_current_version(&self, version: u64) -> Result<()> {
        tokio::fs::create_dir_all(&self.work_dir).await
            .map_err(|e| Error::io(format!("Failed to create work directory: {}", e)))?;

        let version_file = self.work_dir.join("current_version");
        tokio::fs::write(&version_file, version.to_string()).await
            .map_err(|e| Error::io(format!("Failed to write version file: {}", e)))?;

        Ok(())
    }
}

/// Result of a migration execution
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Migration ID
    pub migration_id: Uuid,
    /// Migration name
    pub migration_name: String,
    /// Migration direction
    pub direction: MigrationDirection,
    /// Whether the migration succeeded
    pub success: bool,
    /// Error if migration failed
    pub error: Option<Error>,
    /// How long the migration took
    pub duration: std::time::Duration,
}

/// Migration direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationDirection {
    /// Up migration (apply)
    Up,
    /// Down migration (rollback)
    Down,
}

/// Helper macro for creating migrations
#[macro_export]
macro_rules! migration {
    ($name:ident, $version:expr, $desc:expr, $backends:expr) => {
        #[derive(Debug)]
        pub struct $name;

        impl $crate::migrations::Migration for $name {
            fn meta(&self) -> $crate::migrations::MigrationMeta {
                $crate::migrations::MigrationMeta {
                    id: uuid::Uuid::new_v4(),
                    version: $version,
                    name: stringify!($name).to_string(),
                    description: $desc.to_string(),
                    created_at: chrono::Utc::now(),
                    backends: $backends.iter().map(|s| s.to_string()).collect(),
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    migration!(TestMigration001, 1, "Test migration 1", &["qdrant"]);

    impl Migration for TestMigration001 {
        async fn up(&self, _ctx: &mut MigrationContext) -> Result<()> {
            // Test migration - do nothing
            Ok(())
        }

        async fn down(&self, _ctx: &mut MigrationContext) -> Result<()> {
            // Test rollback - do nothing
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_migration_registry() {
        let mut registry = MigrationRegistry::new();
        let migration = TestMigration001;

        registry.register(migration).unwrap();

        let all = registry.all_sorted();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].meta().version, 1);
    }

    #[tokio::test]
    async fn test_migration_runner() {
        let temp_dir = tempfile::tempdir().unwrap();
        let work_dir = temp_dir.path().to_path_buf();

        let mut registry = MigrationRegistry::new();
        registry.register(TestMigration001).unwrap();

        let runner = MigrationRunner::new(registry, work_dir.clone());

        // Test initial version
        let version = runner.get_current_version().await.unwrap();
        assert_eq!(version, 0);

        // Test migration up
        let backends = HashMap::new();
        let results = runner.migrate_up(None, backends).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].success);

        // Test version after migration
        let version = runner.get_current_version().await.unwrap();
        assert_eq!(version, 1);
    }
}