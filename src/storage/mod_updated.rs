//! Storage module for ReasonKit Core
//!
//! Provides document and chunk storage using Qdrant vector database.
//!
//! ## Modules
//! - `optimized`: High-performance Qdrant operations with batching and caching
//! - `benchmarks`: Performance benchmarking suite
//! - `scalability`: Scalability assessment and production deployment

use crate::{embedding::cosine_similarity, Document, Error, Result};
use async_trait::async_trait;
use qdrant_client::qdrant::{
    CreateCollection, DeletePoints, Distance, GetPoints, PointId, PointStruct, QuantizationConfig,
    ScalarQuantization, ScrollPoints, SearchPoints, UpsertPoints, VectorParams,
    VectorsConfig,
};
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

pub mod benchmarks;
pub mod optimized;
pub mod scalability;

// Continue with the rest of the existing mod.rs content...
