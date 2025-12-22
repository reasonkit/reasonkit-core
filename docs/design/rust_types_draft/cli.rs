//! CLI Type Definitions
//!
//! Types for the ReasonKit command-line interface.
//! Reference: docs/design/CLI_ARCHITECTURE.md

use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL OPTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Global arguments available to all commands
#[derive(Debug, Clone, Parser)]
pub struct GlobalArgs {
    /// Path to configuration file
    #[arg(short, long, global = true, env = "REASONKIT_CONFIG")]
    pub config: Option<PathBuf>,

    /// Data directory for indices and cache
    #[arg(long, global = true, env = "REASONKIT_DATA_DIR")]
    pub data_dir: Option<PathBuf>,

    /// Increase verbosity (-v, -vv, -vvv)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Suppress non-error output
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Output format
    #[arg(long, global = true, default_value = "text")]
    pub format: OutputFormat,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,
}

impl GlobalArgs {
    /// Get effective verbosity level
    pub fn verbosity(&self) -> VerbosityLevel {
        if self.quiet {
            VerbosityLevel::Quiet
        } else {
            match self.verbose {
                0 => VerbosityLevel::Normal,
                1 => VerbosityLevel::Verbose,
                2 => VerbosityLevel::Debug,
                _ => VerbosityLevel::Trace,
            }
        }
    }

    /// Get effective data directory
    pub fn effective_data_dir(&self) -> PathBuf {
        self.data_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("./data"))
    }
}

/// Output format for CLI responses
#[derive(Debug, Clone, Copy, Default, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output for scripting
    Json,
    /// Compact JSON (no pretty printing)
    JsonCompact,
    /// YAML output
    Yaml,
    /// Markdown table format
    Markdown,
}

/// Verbosity levels for output control
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerbosityLevel {
    Quiet,
    Normal,
    Verbose,
    Debug,
    Trace,
}

// ═══════════════════════════════════════════════════════════════════════════
// TOP-LEVEL COMMANDS
// ═══════════════════════════════════════════════════════════════════════════

/// Main CLI application
#[derive(Debug, Parser)]
#[command(
    name = "rk-core",
    version,
    author,
    about = "ReasonKit Core - Structured reasoning and retrieval engine",
    long_about = "ReasonKit Core provides RAG (Retrieval-Augmented Generation) \
                  capabilities with structured reasoning protocols."
)]
pub struct Cli {
    #[command(flatten)]
    pub global: GlobalArgs,

    #[command(subcommand)]
    pub command: Commands,
}

/// Top-level command enumeration
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Ingest documents into the knowledge base
    Ingest(IngestArgs),

    /// Query the knowledge base with natural language
    Query(QueryArgs),

    /// Direct search operations (BM25/vector/hybrid)
    Search(SearchArgs),

    /// Generate embeddings for text
    Embed(EmbedArgs),

    /// Manage search indices
    Index(IndexArgs),

    /// Export data in various formats
    Export(ExportArgs),

    /// Start the HTTP API server
    Serve(ServeArgs),

    /// Run system diagnostics
    Doctor(DoctorArgs),

    /// Generate shell completions
    Completions(CompletionsArgs),

    /// Show current configuration
    Config(ConfigArgs),

    /// Display system statistics
    Stats(StatsArgs),
}

// ═══════════════════════════════════════════════════════════════════════════
// INGEST COMMAND
// ═══════════════════════════════════════════════════════════════════════════

/// Arguments for the ingest command
#[derive(Debug, Clone, Parser)]
pub struct IngestArgs {
    /// Path to file or directory to ingest
    #[arg(required = true)]
    pub path: PathBuf,

    /// Process directories recursively
    #[arg(short, long)]
    pub recursive: bool,

    /// File patterns to include (glob syntax)
    #[arg(long, default_values_t = vec!["**/*.pdf".to_string(), "**/*.md".to_string()])]
    pub include: Vec<String>,

    /// File patterns to exclude (glob syntax)
    #[arg(long)]
    pub exclude: Vec<String>,

    /// Chunking strategy
    #[arg(long, default_value = "semantic")]
    pub chunking: ChunkingStrategy,

    /// Maximum chunk size in tokens
    #[arg(long, default_value = "512")]
    pub chunk_size: usize,

    /// Chunk overlap in tokens
    #[arg(long, default_value = "64")]
    pub chunk_overlap: usize,

    /// Generate embeddings during ingestion
    #[arg(long)]
    pub embed: bool,

    /// Embedding model to use
    #[arg(long, default_value = "text-embedding-3-small")]
    pub embed_model: String,

    /// Number of parallel workers
    #[arg(short = 'j', long, default_value = "4")]
    pub parallel: usize,

    /// Skip existing documents (based on hash)
    #[arg(long)]
    pub skip_existing: bool,

    /// Dry run - show what would be ingested
    #[arg(long)]
    pub dry_run: bool,
}

/// Chunking strategies for document processing
#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkingStrategy {
    /// Fixed token count chunks
    Fixed,
    /// Sentence-boundary aware chunking
    Sentence,
    /// Semantic chunking based on content
    Semantic,
    /// Section-based chunking (headers)
    Section,
    /// Paragraph-based chunking
    Paragraph,
}

// ═══════════════════════════════════════════════════════════════════════════
// QUERY COMMAND
// ═══════════════════════════════════════════════════════════════════════════

/// Arguments for the query command
#[derive(Debug, Clone, Parser)]
pub struct QueryArgs {
    /// Natural language query
    #[arg(required = true)]
    pub query: String,

    /// Reasoning profile to use
    #[arg(short, long, default_value = "balanced")]
    pub profile: ReasoningProfile,

    /// Number of results to return
    #[arg(short = 'k', long, default_value = "10")]
    pub top_k: usize,

    /// Minimum relevance score (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    pub threshold: f64,

    /// Use RAPTOR hierarchical retrieval
    #[arg(long)]
    pub raptor: bool,

    /// RAPTOR tree traversal depth
    #[arg(long, default_value = "3")]
    pub raptor_depth: usize,

    /// Expand query with synonyms/related terms
    #[arg(long)]
    pub expand_query: bool,

    /// Include source citations
    #[arg(long)]
    pub cite: bool,

    /// Show confidence scores
    #[arg(long)]
    pub show_confidence: bool,
}

/// Reasoning profiles for query processing
#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningProfile {
    /// Quick 3-step analysis (70% confidence)
    Quick,
    /// Standard 5-module chain (80% confidence)
    Balanced,
    /// Thorough analysis (85% confidence)
    Deep,
    /// Scientific method (85% confidence)
    Scientific,
    /// Maximum verification (95% confidence)
    Paranoid,
    /// Decision support
    Decide,
}

// ═══════════════════════════════════════════════════════════════════════════
// SEARCH COMMAND
// ═══════════════════════════════════════════════════════════════════════════

/// Arguments for the search command
#[derive(Debug, Clone, Parser)]
pub struct SearchArgs {
    /// Search query
    #[arg(required = true)]
    pub query: String,

    /// Search method
    #[arg(short, long, default_value = "hybrid")]
    pub method: SearchMethod,

    /// Number of results
    #[arg(short = 'k', long, default_value = "10")]
    pub top_k: usize,

    /// Hybrid search alpha (0=BM25, 1=vector)
    #[arg(long, default_value = "0.5")]
    pub alpha: f64,

    /// Fusion method for hybrid search
    #[arg(long, default_value = "rrf")]
    pub fusion: FusionMethod,

    /// RRF k parameter
    #[arg(long, default_value = "60")]
    pub rrf_k: u32,

    /// Filter by source
    #[arg(long)]
    pub source: Option<String>,

    /// Filter by document type
    #[arg(long)]
    pub doc_type: Option<String>,

    /// Apply reranking
    #[arg(long)]
    pub rerank: bool,

    /// Reranking model
    #[arg(long, default_value = "cross-encoder/ms-marco-MiniLM-L-6-v2")]
    pub rerank_model: String,
}

/// Search methods
#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMethod {
    /// BM25 lexical search
    Bm25,
    /// Dense vector similarity search
    Vector,
    /// Combined BM25 + vector search
    Hybrid,
}

/// Fusion methods for hybrid search
#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion
    Rrf,
    /// Linear combination
    Linear,
    /// Weighted average
    Weighted,
}

// ═══════════════════════════════════════════════════════════════════════════
// EMBED COMMAND
// ═══════════════════════════════════════════════════════════════════════════

/// Arguments for the embed command
#[derive(Debug, Clone, Parser)]
pub struct EmbedArgs {
    /// Text to embed (or use --file)
    pub text: Option<String>,

    /// File containing text to embed
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Embedding model to use
    #[arg(short, long, default_value = "text-embedding-3-small")]
    pub model: String,

    /// Embedding dimensions (if model supports)
    #[arg(long)]
    pub dimensions: Option<usize>,

    /// Output embeddings to file
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Batch size for multiple texts
    #[arg(long, default_value = "100")]
    pub batch_size: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// INDEX COMMAND
// ═══════════════════════════════════════════════════════════════════════════

/// Arguments for the index command
#[derive(Debug, Clone, Parser)]
pub struct IndexArgs {
    #[command(subcommand)]
    pub action: IndexAction,
}

/// Index management actions
#[derive(Debug, Clone, Subcommand)]
pub enum IndexAction {
    /// Create a new index
    Create {
        /// Index name
        name: String,
        /// Index type
        #[arg(long, default_value = "hybrid")]
        index_type: IndexType,
        /// Embedding dimensions
        #[arg(long, default_value = "1536")]
        dimensions: usize,
    },
    /// List all indices
    List,
    /// Show index statistics
    Stats {
        /// Index name
        name: String,
    },
    /// Delete an index
    Delete {
        /// Index name
        name: String,
        /// Skip confirmation
        #[arg(long)]
        force: bool,
    },
    /// Rebuild an index
    Rebuild {
        /// Index name
        name: String,
    },
    /// Optimize an index
    Optimize {
        /// Index name
        name: String,
    },
}

/// Index types
#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    /// BM25 full-text index
    Bm25,
    /// Vector similarity index
    Vector,
    /// Combined BM25 + vector index
    Hybrid,
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPORT COMMAND
// ═══════════════════════════════════════════════════════════════════════════

/// Arguments for the export command
#[derive(Debug, Clone, Parser)]
pub struct ExportArgs {
    /// Output file path
    #[arg(required = true)]
    pub output: PathBuf,

    /// Export format
    #[arg(short, long, default_value = "jsonl")]
    pub format: ExportFormat,

    /// Include embeddings in export
    #[arg(long)]
    pub include_embeddings: bool,

    /// Filter by source
    #[arg(long)]
    pub source: Option<String>,

    /// Maximum documents to export
    #[arg(long)]
    pub limit: Option<usize>,
}

/// Export formats
#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    /// JSON Lines format
    Jsonl,
    /// JSON array
    Json,
    /// CSV format
    Csv,
    /// Parquet format
    Parquet,
}

// ═══════════════════════════════════════════════════════════════════════════
// SERVE COMMAND
// ═══════════════════════════════════════════════════════════════════════════

/// Arguments for the serve command
#[derive(Debug, Clone, Parser)]
pub struct ServeArgs {
    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// Enable CORS for all origins
    #[arg(long)]
    pub cors: bool,

    /// API key for authentication
    #[arg(long, env = "REASONKIT_API_KEY")]
    pub api_key: Option<String>,

    /// Maximum request body size (bytes)
    #[arg(long, default_value = "10485760")]
    pub max_body_size: usize,

    /// Request timeout (seconds)
    #[arg(long, default_value = "60")]
    pub timeout: u64,

    /// Number of worker threads
    #[arg(long)]
    pub workers: Option<usize>,
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY COMMANDS
// ═══════════════════════════════════════════════════════════════════════════

/// Arguments for the doctor command
#[derive(Debug, Clone, Parser)]
pub struct DoctorArgs {
    /// Run all checks
    #[arg(long)]
    pub all: bool,

    /// Check configuration
    #[arg(long)]
    pub config: bool,

    /// Check indices
    #[arg(long)]
    pub indices: bool,

    /// Check embedding providers
    #[arg(long)]
    pub embeddings: bool,

    /// Check external dependencies
    #[arg(long)]
    pub deps: bool,

    /// Attempt to fix issues
    #[arg(long)]
    pub fix: bool,
}

/// Arguments for the completions command
#[derive(Debug, Clone, Parser)]
pub struct CompletionsArgs {
    /// Shell to generate completions for
    #[arg(required = true)]
    pub shell: Shell,
}

/// Supported shells for completions
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
    Elvish,
}

/// Arguments for the config command
#[derive(Debug, Clone, Parser)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub action: Option<ConfigAction>,
}

/// Config management actions
#[derive(Debug, Clone, Subcommand)]
pub enum ConfigAction {
    /// Show current configuration
    Show,
    /// Initialize default configuration
    Init {
        /// Overwrite existing config
        #[arg(long)]
        force: bool,
    },
    /// Validate configuration
    Validate,
    /// Get a specific config value
    Get { key: String },
    /// Set a config value
    Set { key: String, value: String },
}

/// Arguments for the stats command
#[derive(Debug, Clone, Parser)]
pub struct StatsArgs {
    /// Show detailed statistics
    #[arg(long)]
    pub detailed: bool,

    /// Include performance metrics
    #[arg(long)]
    pub perf: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// EXIT CODES
// ═══════════════════════════════════════════════════════════════════════════

/// Exit codes for CLI operations
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum ExitCode {
    /// Successful execution
    Success = 0,
    /// General error
    GeneralError = 1,
    /// Configuration error
    ConfigError = 2,
    /// Input/output error
    IoError = 3,
    /// Network error
    NetworkError = 4,
    /// Authentication error
    AuthError = 5,
    /// Resource not found
    NotFound = 6,
    /// Invalid argument
    InvalidArgument = 7,
    /// Operation timeout
    Timeout = 8,
    /// Interrupted by user
    Interrupted = 130,
}

impl From<ExitCode> for i32 {
    fn from(code: ExitCode) -> Self {
        code as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verbosity_levels() {
        let args = GlobalArgs {
            config: None,
            data_dir: None,
            verbose: 0,
            quiet: false,
            format: OutputFormat::Text,
            no_color: false,
        };
        assert_eq!(args.verbosity(), VerbosityLevel::Normal);

        let args = GlobalArgs { verbose: 2, ..args.clone() };
        assert_eq!(args.verbosity(), VerbosityLevel::Debug);

        let args = GlobalArgs { quiet: true, verbose: 3, ..args };
        assert_eq!(args.verbosity(), VerbosityLevel::Quiet);
    }

    #[test]
    fn test_cli_parsing() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
