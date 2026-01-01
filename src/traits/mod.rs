//! Core trait definitions for ReasonKit cross-crate integration.
//!
//! This module defines the interface contracts between:
//! - `reasonkit-core` and `reasonkit-mem` (memory/storage operations)
//! - `reasonkit-core` and `reasonkit-web` (web browsing operations)
//!
//! These traits enable parallel development and loose coupling between crates.

mod memory;
mod web;

pub use memory::*;
pub use web::*;
