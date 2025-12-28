//! ReasonKit Core CLI
//!
//! AI Thinking Enhancement System - Turn Prompts into Protocols
//!
//! ReasonKit makes AI reasoning structured, auditable, and reliable through:
//! - **ThinkTools**: Structured reasoning modules (GigaThink, LaserLogic, ProofGuard...)
//! - **Protocol Architecture**: Analyze → Conclude → Implement → Verify
//! - **Triangulated Truth**: 3-source verification for every claim
//! - **Multi-Agent Swarms**: Specialized agents working in concert

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[cfg(feature = "memory")]
use reasonkit::ingestion::DocumentIngester;

#[cfg(feature = "memory")]
use reasonkit::indexing::IndexManager;
#[cfg(feature = "memory")]
use reasonkit::rag::{RagConfig, RagEngine};
#[cfg(feature = "memory")]
use reasonkit::thinktool::UnifiedLlmClient;
use reasonkit::thinktool::{
    BudgetConfig, CliToolConfig, ExecutionStatus, ExecutionTrace, ExecutorConfig, LlmConfig,
    LlmProvider, ProtocolExecutor, ProtocolInput, ProtocolOutput, StepStatus,
};
use reasonkit::verification::ProofLedger;
use reasonkit::web::{SearchConfig, SearchProvider, WebSearcher};
#[cfg(feature = "memory")]
use reasonkit::DocumentType;

/// ReasonKit - AI Thinking Enhancement System
///
/// Turn Prompts into Protocols. Make AI reasoning structured, auditable, and reliable.
///
/// CORE FEATURES:
///   • ThinkTools     - Structured reasoning modules (GigaThink, LaserLogic, ProofGuard)
///   • Protocols      - Analyze → Conclude → Implement → Verify workflow
///   • Triangulation  - 3-source verification for every claim
///   • Agent Swarms   - Specialized AI agents working in concert
#[derive(Parser)]
#[command(name = "rk-core")]
#[command(author = "ReasonKit Team <hello@reasonkit.sh>")]
#[command(version)]
#[command(about = "AI Thinking Enhancement System - Turn Prompts into Protocols")]
#[command(long_about = r#"
ReasonKit makes AI reasoning STRUCTURED, AUDITABLE, and RELIABLE.

CORE DIFFERENTIATORS:

  ThinkTools (Structured Reasoning Modules)
    • GigaThink   - Expansive creative thinking, 10+ perspectives
    • LaserLogic  - Precision deductive reasoning, fallacy detection
    • BedRock     - First principles decomposition
    • ProofGuard  - Multi-source verification, contradiction detection
    • BrutalHonesty - Adversarial self-critique

  Protocol Architecture
    Analyze → Conclude → Implement → Verify → Persist
    Every response follows an engineered process, not ad-hoc generation.

  Triangulated Truth
    NO claim without 3 independent sources. Cryptographic anchoring
    via ProofLedger. Drift detection over time.

  Multi-Agent Swarms
    Governance → Executive → Engineering → Specialist tiers
    Right model for each task. Automatic task routing.

QUICK START:
  rk-core think "Your question"           # Apply ThinkTools
  rk-core web "Research topic"            # Deep research with verification
  rk-core verify "claim" --sources 3      # Triangulate a claim

Website: https://reasonkit.sh
"#)]
struct Cli {
    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Configuration file path
    #[arg(short, long, env = "REASONKIT_CONFIG")]
    config: Option<PathBuf>,

    /// Data directory
    #[arg(short, long, env = "REASONKIT_DATA_DIR", default_value = "./data")]
    data_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // ═══════════════════════════════════════════════════════════════════════
    // CORE: ThinkTools - Structured Reasoning (THE PURPOSE)
    // ═══════════════════════════════════════════════════════════════════════
    /// \[CORE\] Execute structured reasoning protocols (ThinkTools)
    ///
    /// ThinkTools are the heart of ReasonKit - structured reasoning modules
    /// that improve AI thinking through defined protocols:
    ///
    ///   • GigaThink (gt)     - Expansive creative thinking, 10+ perspectives
    ///   • LaserLogic (ll)    - Precision deductive reasoning, fallacy detection
    ///   • BedRock (br)       - First principles decomposition
    ///   • ProofGuard (pg)    - Multi-source verification
    ///   • BrutalHonesty (bh) - Adversarial self-critique
    ///
    /// Profiles combine modules into reasoning chains:
    ///   --quick     GigaThink → LaserLogic (fast)
    ///   --balanced  gt → ll → br → pg (standard)
    ///   --paranoid  All 5 + BH loop (maximum verification)
    #[command(alias = "t")]
    Think {
        /// The query or input to process (not required with --list)
        #[arg(required_unless_present = "list")]
        query: Option<String>,

        /// Protocol to execute (gigathink, laserlogic, bedrock, proofguard, brutalhonesty)
        #[arg(short, long)]
        protocol: Option<String>,

        /// Profile to execute (quick, balanced, deep, paranoid, decide, scientific)
        #[arg(long)]
        profile: Option<String>,

        /// LLM provider (anthropic, openai, openrouter)
        #[arg(long, default_value = "anthropic")]
        provider: ProviderArg,

        /// LLM model to use
        #[arg(short, long)]
        model: Option<String>,

        /// Temperature for generation (0.0-2.0)
        #[arg(short, long, default_value = "0.7")]
        temperature: f64,

        /// Maximum tokens to generate
        #[arg(long, default_value = "2000")]
        max_tokens: u32,

        /// Budget for adaptive compute time (e.g., "30s", "1000t", "$0.50")
        /// Formats: Xs/Xm/Xh (time), Xt/Xtokens (tokens), $X.XX (cost)
        #[arg(short, long)]
        budget: Option<String>,

        /// Use mock LLM (for testing)
        #[arg(long)]
        mock: bool,

        /// Save execution trace
        #[arg(long)]
        save_trace: bool,

        /// Trace output directory
        #[arg(long)]
        trace_dir: Option<PathBuf>,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,

        /// List available protocols and profiles
        #[arg(long)]
        list: bool,
    },

    /// \[CORE\] Deep research with ThinkTools + Web + KB (reasonkit-web)
    ///
    /// Combines structured reasoning with multi-source research:
    ///   1. Web search (DuckDuckGo/Tavily/Serper)
    ///   2. Knowledge base context
    ///   3. ThinkTool protocol chain
    ///   4. Source triangulation
    ///
    /// Depth levels map to ThinkTool profiles:
    ///   quick      → GigaThink only (~30s)
    ///   standard   → GigaThink + LaserLogic (~2min)
    ///   deep       → Full balanced profile (~5min)
    ///   exhaustive → Paranoid profile with triangulation (~10min)
    #[command(alias = "dive", alias = "research", alias = "deep", alias = "d")]
    Web {
        /// Research question or topic
        query: String,

        /// Depth of research (quick, standard, deep, exhaustive)
        #[arg(short, long, default_value = "standard")]
        depth: WebDepth,

        /// Include web search results
        #[arg(long, default_value = "true")]
        web: bool,

        /// Include knowledge base results
        #[arg(long, default_value = "true")]
        kb: bool,

        /// LLM provider (anthropic, openai, openrouter)
        #[arg(long, default_value = "anthropic")]
        provider: ProviderArg,

        /// Output format (text, json, markdown)
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,

        /// Save research report to file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// \[CORE\] Triangulate and verify claims with 3+ sources
    ///
    /// The THREE-SOURCE RULE: NO claim without independent verification.
    ///
    /// Verification workflow:
    ///   1. Search web for supporting/contradicting evidence
    ///   2. Cross-reference with knowledge base
    ///   3. Apply ProofGuard reasoning protocol
    ///   4. Anchor verified content via ProofLedger (cryptographic hash)
    ///   5. Track source drift over time
    ///
    /// Source quality tiers:
    ///   Tier 1: Official docs, GitHub, arXiv papers (highest trust)
    ///   Tier 2: Reputable tech blogs, verified experts
    ///   Tier 3: Community content (requires corroboration)
    #[command(alias = "v", alias = "triangulate")]
    Verify {
        /// Claim or statement to verify
        claim: String,

        /// Minimum number of sources required
        #[arg(short, long, default_value = "3")]
        sources: usize,

        /// Include web search for verification
        #[arg(long, default_value = "true")]
        web: bool,

        /// Include knowledge base sources
        #[arg(long, default_value = "true")]
        kb: bool,

        /// Anchor verified content to ProofLedger
        #[arg(long)]
        anchor: bool,

        /// Output format (text, json, markdown)
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,

        /// Save verification report to file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    // ═══════════════════════════════════════════════════════════════════════
    // SUPPORTING: Knowledge Base & Infrastructure (requires 'memory' feature)
    // ═══════════════════════════════════════════════════════════════════════
    /// Ingest documents into the knowledge base (requires 'memory' feature)
    #[cfg(feature = "memory")]
    Ingest {
        /// Path to document or directory
        path: PathBuf,

        /// Document type (paper, documentation, code, note)
        #[arg(short = 't', long, default_value = "paper")]
        doc_type: String,

        /// Process recursively if directory
        #[arg(short, long)]
        recursive: bool,
    },

    /// Query the knowledge base (requires 'memory' feature)
    #[cfg(feature = "memory")]
    Query {
        /// Query string
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,

        /// Use hybrid search (BM25 + vector)
        #[arg(long)]
        hybrid: bool,

        /// Use RAPTOR tree retrieval
        #[arg(long)]
        raptor: bool,

        /// Output format (text, json, markdown)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Manage the search index (requires 'memory' feature)
    #[cfg(feature = "memory")]
    Index {
        #[command(subcommand)]
        action: IndexAction,
    },

    /// Show statistics about the knowledge base
    Stats,

    /// Export knowledge base data (requires 'memory' feature)
    #[cfg(feature = "memory")]
    Export {
        /// Output path
        output: PathBuf,

        /// Export format (json, jsonl)
        #[arg(short, long, default_value = "jsonl")]
        format: String,
    },

    /// Start the API server
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },

    /// View and manage execution traces
    Trace {
        #[command(subcommand)]
        action: TraceAction,
    },

    /// Retrieval-Augmented Generation queries (requires 'memory' feature)
    #[cfg(feature = "memory")]
    Rag {
        #[command(subcommand)]
        action: RagAction,
    },

    /// View ThinkTools execution metrics and reports
    ///
    /// Continuous measurement system for tracking reasoning quality.
    /// Provides grades, scores, and recommendations based on execution history.
    #[command(alias = "m")]
    Metrics {
        #[command(subcommand)]
        action: MetricsAction,
    },

    // ═══════════════════════════════════════════════════════════════════════
    // SHELL: Completions & Shell Integration
    // ═══════════════════════════════════════════════════════════════════════
    /// Generate shell completions for Zsh, Bash, Fish, PowerShell, or Elvish
    ///
    /// Installation examples:
    ///
    ///   Zsh (Oh-My-Zsh):
    ///     rk-core completions zsh > $ZSH_CUSTOM/plugins/reasonkit/_rk-core
    ///
    ///   Zsh (standard):
    ///     rk-core completions zsh > ~/.zsh/completions/_rk-core
    ///
    ///   Bash:
    ///     rk-core completions bash > ~/.bash_completion.d/rk-core
    ///
    ///   Fish:
    ///     rk-core completions fish > ~/.config/fish/completions/rk-core.fish
    ///
    ///   PowerShell:
    ///     rk-core completions powershell >> $PROFILE
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

/// Web research depth levels
#[derive(Clone, Copy, Debug, ValueEnum)]
enum WebDepth {
    /// Quick analysis (GigaThink only, ~30s)
    Quick,
    /// Standard research (GigaThink + LaserLogic, ~2min)
    Standard,
    /// Deep research (full balanced profile, ~5min)
    Deep,
    /// Exhaustive analysis (paranoid profile, ~10min)
    Exhaustive,
}

#[cfg(feature = "memory")]
#[derive(Subcommand)]
enum RagAction {
    /// Query the knowledge base with RAG
    Query {
        /// The query to answer
        query: String,

        /// Number of chunks to retrieve
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,

        /// Minimum relevance score (0.0-1.0)
        #[arg(long, default_value = "0.1")]
        min_score: f32,

        /// RAG mode: quick, balanced, thorough
        #[arg(long, default_value = "balanced")]
        mode: RagMode,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,

        /// Don't use LLM (retrieval only)
        #[arg(long)]
        no_llm: bool,
    },

    /// Retrieve relevant chunks without generation
    Retrieve {
        /// The query
        query: String,

        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
    },

    /// Show RAG knowledge base stats
    Stats,
}

/// RAG mode presets
#[cfg(feature = "memory")]
#[derive(Clone, Copy, Debug, ValueEnum)]
enum RagMode {
    /// Fast, fewer chunks (3)
    Quick,
    /// Balanced (5 chunks)
    Balanced,
    /// Comprehensive (10 chunks)
    Thorough,
}

#[derive(Subcommand)]
enum TraceAction {
    /// List saved traces
    List {
        /// Trace directory
        #[arg(short, long)]
        dir: Option<PathBuf>,

        /// Filter by protocol
        #[arg(short, long)]
        protocol: Option<String>,

        /// Limit number of results
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// View a specific trace
    View {
        /// Trace ID or file path
        id: String,

        /// Trace directory (if using ID)
        #[arg(short, long)]
        dir: Option<PathBuf>,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
    },

    /// Delete traces
    Clean {
        /// Trace directory
        #[arg(short, long)]
        dir: Option<PathBuf>,

        /// Delete all traces
        #[arg(long)]
        all: bool,

        /// Keep traces from last N days
        #[arg(long)]
        keep_days: Option<u32>,
    },
}

#[derive(Subcommand)]
enum MetricsAction {
    /// Show metrics report with grades and scores
    Report {
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,

        /// Filter by protocol or profile
        #[arg(short, long)]
        filter: Option<String>,

        /// Save report to file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show statistics for a specific protocol or profile
    Stats {
        /// Protocol or profile name (e.g., "gigathink", "paranoid")
        name: String,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
    },

    /// Show metrics storage location
    Path,

    /// Clear all metrics data
    Clear {
        /// Skip confirmation
        #[arg(long)]
        yes: bool,
    },
}

/// LLM Provider argument - supports 18+ API providers + CLI tools
#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProviderArg {
    // ─────────────────────────────────────────────────────────────────────────
    // TIER 1: Major Cloud Providers (API-based)
    // ─────────────────────────────────────────────────────────────────────────
    /// Anthropic Claude (default)
    Anthropic,
    /// OpenAI GPT
    #[value(name = "openai")]
    OpenAI,
    /// Google Gemini (AI Studio)
    Gemini,
    /// Google Vertex AI
    Vertex,
    /// Azure OpenAI
    Azure,
    /// AWS Bedrock
    Bedrock,

    // ─────────────────────────────────────────────────────────────────────────
    // TIER 2: Specialized Providers (API-based)
    // ─────────────────────────────────────────────────────────────────────────
    /// xAI Grok
    Xai,
    /// Groq (ultra-fast)
    Groq,
    /// Mistral AI
    Mistral,
    /// DeepSeek
    Deepseek,
    /// Cohere
    Cohere,
    /// Perplexity
    Perplexity,
    /// Cerebras (fastest inference)
    Cerebras,

    // ─────────────────────────────────────────────────────────────────────────
    // TIER 3: Inference Platforms (API-based)
    // ─────────────────────────────────────────────────────────────────────────
    /// Together AI
    Together,
    /// Fireworks AI
    Fireworks,
    /// Alibaba Qwen
    Qwen,
    /// Cloudflare AI
    Cloudflare,

    // ─────────────────────────────────────────────────────────────────────────
    // TIER 4: Aggregation (API-based)
    // ─────────────────────────────────────────────────────────────────────────
    /// OpenRouter (300+ models)
    #[value(name = "openrouter")]
    OpenRouter,

    // ─────────────────────────────────────────────────────────────────────────
    // TIER 5: CLI Tools (Browser-based auth, shell out)
    // ─────────────────────────────────────────────────────────────────────────
    /// Claude CLI (claude -p "...")
    #[value(name = "claude-cli")]
    ClaudeCli,
    /// OpenAI Codex CLI (codex "...")
    #[value(name = "codex-cli")]
    CodexCli,
    /// Gemini CLI (gemini -p "...")
    #[value(name = "gemini-cli")]
    GeminiCli,
    /// OpenCode CLI (opencode "...")
    #[value(name = "opencode-cli")]
    OpencodeCli,
    /// GitHub Copilot CLI (gh copilot suggest)
    #[value(name = "copilot-cli")]
    CopilotCli,
}

#[allow(dead_code)]
impl ProviderArg {
    /// Check if this provider uses CLI tool execution
    pub fn is_cli_tool(&self) -> bool {
        matches!(
            self,
            ProviderArg::ClaudeCli
                | ProviderArg::CodexCli
                | ProviderArg::GeminiCli
                | ProviderArg::OpencodeCli
                | ProviderArg::CopilotCli
        )
    }

    /// Get CLI command for CLI tool providers
    pub fn cli_command(&self) -> Option<&'static str> {
        match self {
            ProviderArg::ClaudeCli => Some("claude"),
            ProviderArg::CodexCli => Some("codex"),
            ProviderArg::GeminiCli => Some("gemini"),
            ProviderArg::OpencodeCli => Some("opencode"),
            ProviderArg::CopilotCli => Some("gh"),
            _ => None,
        }
    }

    /// Get default model for API-based providers
    pub fn default_model(&self) -> &'static str {
        match self {
            // Tier 1: Major Cloud
            ProviderArg::Anthropic => "claude-sonnet-4-20250514",
            ProviderArg::OpenAI => "gpt-4o",
            ProviderArg::Gemini => "gemini-2.0-flash",
            ProviderArg::Vertex => "gemini-2.0-flash",
            ProviderArg::Azure => "gpt-4o",
            ProviderArg::Bedrock => "anthropic.claude-sonnet-4-v1:0",
            // Tier 2: Specialized
            ProviderArg::Xai => "grok-2",
            ProviderArg::Groq => "llama-3.3-70b-versatile",
            ProviderArg::Mistral => "mistral-large-latest",
            ProviderArg::Deepseek => "deepseek-chat",
            ProviderArg::Cohere => "command-r-plus",
            ProviderArg::Perplexity => "sonar-pro",
            ProviderArg::Cerebras => "llama-3.3-70b",
            // Tier 3: Inference Platforms
            ProviderArg::Together => "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            ProviderArg::Fireworks => "accounts/fireworks/models/llama-v3p3-70b-instruct",
            ProviderArg::Qwen => "qwen-max",
            ProviderArg::Cloudflare => "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            // Tier 4: Aggregation
            ProviderArg::OpenRouter => "anthropic/claude-3.5-sonnet",
            // Tier 5: CLI Tools (not used for model selection)
            ProviderArg::ClaudeCli => "claude-sonnet-4",
            ProviderArg::CodexCli => "gpt-4o",
            ProviderArg::GeminiCli => "gemini-2.0-flash",
            ProviderArg::OpencodeCli => "default",
            ProviderArg::CopilotCli => "copilot",
        }
    }
}

impl From<ProviderArg> for LlmProvider {
    fn from(arg: ProviderArg) -> Self {
        match arg {
            // Tier 1: Major Cloud
            ProviderArg::Anthropic => LlmProvider::Anthropic,
            ProviderArg::OpenAI => LlmProvider::OpenAI,
            ProviderArg::Gemini => LlmProvider::GoogleGemini,
            ProviderArg::Vertex => LlmProvider::GoogleVertex,
            ProviderArg::Azure => LlmProvider::AzureOpenAI,
            ProviderArg::Bedrock => LlmProvider::AWSBedrock,
            // Tier 2: Specialized
            ProviderArg::Xai => LlmProvider::XAI,
            ProviderArg::Groq => LlmProvider::Groq,
            ProviderArg::Mistral => LlmProvider::Mistral,
            ProviderArg::Deepseek => LlmProvider::DeepSeek,
            ProviderArg::Cohere => LlmProvider::Cohere,
            ProviderArg::Perplexity => LlmProvider::Perplexity,
            ProviderArg::Cerebras => LlmProvider::Cerebras,
            // Tier 3: Inference Platforms
            ProviderArg::Together => LlmProvider::TogetherAI,
            ProviderArg::Fireworks => LlmProvider::FireworksAI,
            ProviderArg::Qwen => LlmProvider::AlibabaQwen,
            ProviderArg::Cloudflare => LlmProvider::CloudflareAI,
            // Tier 4: Aggregation
            ProviderArg::OpenRouter => LlmProvider::OpenRouter,
            // Tier 5: CLI Tools - map to corresponding API provider for config
            ProviderArg::ClaudeCli => LlmProvider::Anthropic,
            ProviderArg::CodexCli => LlmProvider::OpenAI,
            ProviderArg::GeminiCli => LlmProvider::GoogleGemini,
            ProviderArg::OpencodeCli => LlmProvider::OpenAI,
            ProviderArg::CopilotCli => LlmProvider::OpenAI,
        }
    }
}

/// Output format
#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[cfg(feature = "memory")]
#[derive(Subcommand)]
enum IndexAction {
    /// Build or rebuild the index
    Build {
        /// Force rebuild even if up to date
        #[arg(short, long)]
        force: bool,
    },

    /// Show index status
    Status,

    /// Optimize the index
    Optimize,
}

fn setup_logging(verbosity: u8) {
    let level = match verbosity {
        0 => Level::WARN,
        1 => Level::INFO,
        2 => Level::DEBUG,
        _ => Level::TRACE,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(verbosity >= 3)
        .with_line_number(verbosity >= 3)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    setup_logging(cli.verbose);

    info!("ReasonKit Core v{}", env!("CARGO_PKG_VERSION"));
    info!("Data directory: {:?}", cli.data_dir);

    match cli.command {
        #[cfg(feature = "memory")]
        Commands::Ingest {
            path,
            doc_type,
            recursive,
        } => {
            info!("Ingesting documents from {:?}", path);
            info!("Document type: {}", doc_type);
            info!("Recursive: {}", recursive);

            let ingester = DocumentIngester::new();
            let index_path = cli.data_dir.join("index");
            let index = IndexManager::open(index_path)
                .map_err(|e| anyhow::anyhow!("Failed to open index: {}", e))?;

            // Collect files to process
            let files: Vec<PathBuf> = if path.is_dir() {
                if recursive {
                    walkdir::WalkDir::new(&path)
                        .into_iter()
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_file())
                        .map(|e| e.path().to_path_buf())
                        .collect()
                } else {
                    std::fs::read_dir(&path)?
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_file())
                        .map(|e| e.path())
                        .collect()
                }
            } else {
                vec![path.clone()]
            };

            let mut success_count = 0;
            let mut error_count = 0;

            for file_path in &files {
                match ingester.ingest(file_path) {
                    Ok(mut doc) => {
                        // Override document type if specified
                        doc.doc_type = match doc_type.as_str() {
                            "paper" => DocumentType::Paper,
                            "documentation" | "docs" => DocumentType::Documentation,
                            "code" => DocumentType::Code,
                            "note" | "notes" => DocumentType::Note,
                            "transcript" => DocumentType::Transcript,
                            _ => doc.doc_type,
                        };

                        // Index the document
                        match index.index_document(&doc) {
                            Ok(chunks) => {
                                println!("✓ {} ({} chunks)", file_path.display(), chunks);
                                success_count += 1;
                            }
                            Err(e) => {
                                eprintln!("✗ {} - Index error: {}", file_path.display(), e);
                                error_count += 1;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("✗ {} - {}", file_path.display(), e);
                        error_count += 1;
                    }
                }
            }

            println!();
            println!(
                "Ingestion complete: {} succeeded, {} failed",
                success_count, error_count
            );
        }

        #[cfg(feature = "memory")]
        Commands::Query {
            query,
            top_k,
            hybrid,
            raptor,
            format,
        } => {
            info!("Querying: {}", query);
            info!("Top-k: {}, Hybrid: {}, RAPTOR: {}", top_k, hybrid, raptor);

            let index_path = cli.data_dir.join("index");
            let index = IndexManager::open(index_path)
                .map_err(|e| anyhow::anyhow!("Failed to open index: {}", e))?;

            // Use BM25 search (always available, no embedding required)
            let results = index
                .search_bm25(&query, top_k)
                .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;

            if results.is_empty() {
                println!("No results found for: \"{}\"", query);
                return Ok(());
            }

            match format.as_str() {
                "json" => {
                    println!("{}", serde_json::to_string_pretty(&results)?);
                }
                "markdown" | "md" => {
                    println!("# Search Results for: \"{}\"\n", query);
                    for (i, result) in results.iter().enumerate() {
                        println!("## {}. Score: {:.3}\n", i + 1, result.score);
                        println!("{}\n", result.text);
                        println!("---\n");
                    }
                }
                _ => {
                    // Text format (default)
                    println!();
                    println!(
                        "═══════════════════════════════════════════════════════════════════════"
                    );
                    println!("                         Search Results");
                    println!(
                        "═══════════════════════════════════════════════════════════════════════"
                    );
                    println!();
                    println!("Query: \"{}\"", query);
                    println!("Results: {}", results.len());
                    println!();

                    for (i, result) in results.iter().enumerate() {
                        println!("{}. [score: {:.3}]", i + 1, result.score);
                        println!("   {}", truncate_text(&result.text, 200));
                        println!();
                    }
                }
            }
        }

        #[cfg(feature = "memory")]
        Commands::Index { action } => {
            let index_path = cli.data_dir.join("index");

            match action {
                IndexAction::Build { force } => {
                    info!("Building index (force={})", force);

                    if force && index_path.exists() {
                        std::fs::remove_dir_all(&index_path)?;
                        println!("Removed existing index");
                    }

                    let index = IndexManager::open(index_path.clone())
                        .map_err(|e| anyhow::anyhow!("Failed to create index: {}", e))?;

                    // Index all documents from data directory
                    let docs_path = cli.data_dir.join("documents");
                    if !docs_path.exists() {
                        println!("No documents directory found at {:?}", docs_path);
                        println!("Use 'rk-core ingest <path>' to add documents first.");
                        return Ok(());
                    }

                    let ingester = DocumentIngester::new();
                    let mut total_chunks = 0;
                    let mut doc_count = 0;

                    for entry in walkdir::WalkDir::new(&docs_path)
                        .into_iter()
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_file())
                    {
                        if let Ok(doc) = ingester.ingest(entry.path()) {
                            match index.index_document(&doc) {
                                Ok(chunks) => {
                                    total_chunks += chunks;
                                    doc_count += 1;
                                }
                                Err(e) => {
                                    eprintln!("Warning: Failed to index {:?}: {}", entry.path(), e);
                                }
                            }
                        }
                    }

                    println!(
                        "Index built: {} documents, {} chunks",
                        doc_count, total_chunks
                    );
                }
                IndexAction::Status => {
                    info!("Checking index status");

                    if !index_path.exists() {
                        println!("No index found at {:?}", index_path);
                        println!("Run 'rk-core index build' to create one.");
                        return Ok(());
                    }

                    let index = IndexManager::open(index_path.clone())
                        .map_err(|e| anyhow::anyhow!("Failed to open index: {}", e))?;

                    let stats = index
                        .stats()
                        .map_err(|e| anyhow::anyhow!("Failed to get stats: {}", e))?;

                    println!();
                    println!(
                        "═══════════════════════════════════════════════════════════════════════"
                    );
                    println!("                         Index Status");
                    println!(
                        "═══════════════════════════════════════════════════════════════════════"
                    );
                    println!();
                    println!("  Path:           {:?}", index_path);
                    println!("  Documents:      {}", stats.document_count);
                    println!("  Chunks:         {}", stats.chunk_count);
                    println!("  Size:           {} bytes", stats.size_bytes);
                    if let Some(updated) = &stats.last_updated {
                        println!("  Last updated:   {}", updated);
                    }
                    println!();
                }
                IndexAction::Optimize => {
                    info!("Optimizing index");

                    if !index_path.exists() {
                        println!("No index found at {:?}", index_path);
                        return Ok(());
                    }

                    let index = IndexManager::open(index_path)
                        .map_err(|e| anyhow::anyhow!("Failed to open index: {}", e))?;

                    index
                        .optimize()
                        .map_err(|e| anyhow::anyhow!("Optimization failed: {}", e))?;

                    println!("Index optimized successfully");
                }
            }
        }

        Commands::Stats => {
            info!("Showing statistics");

            println!();
            println!("═══════════════════════════════════════════════════════════════════════");
            println!("                    ReasonKit Knowledge Base Stats");
            println!("═══════════════════════════════════════════════════════════════════════");
            println!();

            // Index stats (only available with memory feature)
            #[cfg(feature = "memory")]
            {
                let index_path = cli.data_dir.join("index");
                if index_path.exists() {
                    if let Ok(index) = IndexManager::open(index_path) {
                        if let Ok(stats) = index.stats() {
                            println!("INDEX (BM25):");
                            println!("  Chunks indexed:  {}", stats.chunk_count);
                            println!("  Size:            {} bytes", stats.size_bytes);
                            println!();
                        }
                    }
                } else {
                    println!("INDEX: Not built (run 'rk-core index build')");
                    println!();
                }
            }
            #[cfg(not(feature = "memory"))]
            {
                println!("INDEX: Enable 'memory' feature for index stats");
                println!();
            }

            // RAG stats
            let rag_path = cli.data_dir.join("rag");
            if rag_path.exists() {
                println!("RAG:");
                println!("  Path:            {:?}", rag_path);
                // Get directory size
                let size: u64 = walkdir::WalkDir::new(&rag_path)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum();
                println!("  Size:            {} bytes", size);
                println!();
            }

            // Trace stats
            let trace_path = cli.data_dir.join("traces");
            if trace_path.exists() {
                let trace_count = std::fs::read_dir(&trace_path)
                    .map(|entries| entries.filter_map(|e| e.ok()).count())
                    .unwrap_or(0);
                println!("TRACES:");
                println!("  Count:           {}", trace_count);
                println!("  Path:            {:?}", trace_path);
                println!();
            }

            // Data directory info
            println!("DATA DIRECTORY:");
            println!("  Path:            {:?}", cli.data_dir);
            let total_size: u64 = walkdir::WalkDir::new(&cli.data_dir)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum();
            println!(
                "  Total size:      {} bytes ({:.2} MB)",
                total_size,
                total_size as f64 / 1_000_000.0
            );
            println!();
        }

        #[cfg(feature = "memory")]
        Commands::Export { output, format } => {
            info!("Exporting to {:?} in {} format", output, format);

            let index_path = cli.data_dir.join("index");
            if !index_path.exists() {
                println!("No index found. Nothing to export.");
                return Ok(());
            }

            let index = IndexManager::open(index_path)
                .map_err(|e| anyhow::anyhow!("Failed to open index: {}", e))?;

            // Search for all documents (empty query returns all with BM25)
            let results = index.search_bm25("*", 10000)?;

            let file = std::fs::File::create(&output)?;
            let mut writer = std::io::BufWriter::new(file);

            match format.as_str() {
                "json" => {
                    use std::io::Write;
                    serde_json::to_writer_pretty(&mut writer, &results)?;
                    writeln!(writer)?;
                }
                _ => {
                    // Default to JSONL format
                    use std::io::Write;
                    for result in &results {
                        serde_json::to_writer(&mut writer, result)?;
                        writeln!(writer)?;
                    }
                }
            }

            println!("Exported {} records to {:?}", results.len(), output);
        }

        Commands::Serve { host, port } => {
            info!("Starting API server on {}:{}", host, port);

            println!();
            println!("═══════════════════════════════════════════════════════════════════════");
            println!("                    ReasonKit API Server");
            println!("═══════════════════════════════════════════════════════════════════════");
            println!();
            println!("  Endpoint:  http://{}:{}", host, port);
            println!("  Status:    Starting...");
            println!();
            println!("API server requires the 'server' feature.");
            println!("For now, use the CLI commands or MCP server instead:");
            println!();
            println!("  CLI:   rk-core query \"your query\"");
            println!("  MCP:   Configure reasonkit-core as an MCP server");
            println!();
            println!("Full HTTP API coming in a future release.");
        }

        Commands::Think {
            query,
            protocol,
            profile,
            provider,
            model,
            temperature,
            max_tokens,
            budget,
            mock,
            save_trace,
            trace_dir,
            format,
            list,
        } => {
            // Handle --list first
            if list {
                print_thinktool_list()?;
                return Ok(());
            }

            // Query is required at this point
            let query = query.ok_or_else(|| anyhow::anyhow!("Query is required"))?;

            // Build LLM config (now uses ProviderArg::default_model())
            let llm_config = LlmConfig {
                provider: provider.into(),
                model: model.unwrap_or_else(|| provider.default_model().to_string()),
                temperature,
                max_tokens,
                ..Default::default()
            };

            // Parse budget if provided
            let budget_config = if let Some(budget_str) = budget {
                BudgetConfig::parse(&budget_str)
                    .map_err(|e| anyhow::anyhow!("Invalid budget format: {}", e))?
            } else {
                BudgetConfig::default()
            };

            if budget_config.is_constrained() {
                info!("Budget constraint: {:?}", budget_config);
            }

            // Build CLI tool config if using a CLI tool provider
            let cli_tool = match provider {
                ProviderArg::ClaudeCli => Some(CliToolConfig::claude()),
                ProviderArg::CodexCli => Some(CliToolConfig::codex()),
                ProviderArg::GeminiCli => Some(CliToolConfig::gemini()),
                ProviderArg::OpencodeCli => Some(CliToolConfig::opencode()),
                ProviderArg::CopilotCli => Some(CliToolConfig::copilot()),
                _ => None,
            };

            // Build executor config
            let config = ExecutorConfig {
                llm: llm_config,
                use_mock: mock,
                save_traces: save_trace,
                trace_dir: trace_dir.or_else(|| {
                    if save_trace {
                        Some(cli.data_dir.join("traces"))
                    } else {
                        None
                    }
                }),
                verbose: cli.verbose > 0,
                budget: budget_config,
                cli_tool,
                ..Default::default()
            };

            let executor = ProtocolExecutor::with_config(config)
                .map_err(|e| anyhow::anyhow!("Failed to create executor: {}", e))?;

            // Execute protocol or profile
            let result = if let Some(profile_id) = profile {
                info!("Executing profile: {}", profile_id);
                let input = ProtocolInput::query(&query);
                executor
                    .execute_profile(&profile_id, input)
                    .await
                    .map_err(|e| anyhow::anyhow!("Profile execution failed: {}", e))?
            } else {
                let protocol_id = protocol.unwrap_or_else(|| "gigathink".to_string());
                let protocol_lower = protocol_id.to_lowercase();
                let normalized_protocol = match protocol_lower.as_str() {
                    "gigathink" | "gt" | "giga" | "creative" | "rainbow" => "gigathink",
                    "laserlogic" | "ll" | "laser" | "logic" | "deduce" => "laserlogic",
                    "bedrock" | "br" | "roots" | "foundation" | "base" => "bedrock",
                    "proofguard" | "pg" | "proof" | "verify" | "check" | "guard" => "proofguard",
                    "brutalhonesty" | "bh" | "brutal" | "critique" | "honest" => "brutalhonesty",
                    _ => protocol_lower.as_str(),
                };

                info!(
                    "Executing protocol: {} (normalized from {})",
                    normalized_protocol, protocol_id
                );

                // Build appropriate input based on protocol
                let input = match normalized_protocol {
                    "gigathink" => ProtocolInput::query(&query),
                    "laserlogic" => ProtocolInput::argument(&query),
                    "bedrock" => ProtocolInput::statement(&query),
                    "proofguard" => ProtocolInput::claim(&query),
                    "brutalhonesty" => ProtocolInput::work(&query),
                    _ => ProtocolInput::query(&query),
                };

                executor
                    .execute(normalized_protocol, input)
                    .await
                    .map_err(|e| anyhow::anyhow!("Protocol execution failed: {}", e))?
            };

            // Output results
            match format {
                OutputFormat::Json => {
                    let json = serde_json::to_string_pretty(&result)
                        .map_err(|e| anyhow::anyhow!("JSON serialization failed: {}", e))?;
                    println!("{}", json);
                }
                OutputFormat::Text => {
                    print_think_result(&result);
                }
            }

            // Show trace info if saved
            if let Some(trace_id) = &result.trace_id {
                info!("Trace saved: {}", trace_id);
            }
        }

        Commands::Trace { action } => {
            let trace_dir = cli.data_dir.join("traces");

            match action {
                TraceAction::List {
                    dir,
                    protocol,
                    limit,
                } => {
                    let dir = dir.unwrap_or(trace_dir);
                    list_traces(&dir, protocol.as_deref(), limit)?;
                }

                TraceAction::View { id, dir, format } => {
                    let dir = dir.unwrap_or(trace_dir);
                    view_trace(&id, &dir, format)?;
                }

                TraceAction::Clean {
                    dir,
                    all,
                    keep_days,
                } => {
                    let dir = dir.unwrap_or(trace_dir);
                    clean_traces(&dir, all, keep_days)?;
                }
            }
        }

        #[cfg(feature = "memory")]
        Commands::Rag { action } => {
            let rag_dir = cli.data_dir.join("rag");

            match action {
                RagAction::Query {
                    query,
                    top_k,
                    min_score,
                    mode,
                    format,
                    no_llm,
                } => {
                    // Build RAG config based on mode
                    let mut config = match mode {
                        RagMode::Quick => RagConfig::quick(),
                        RagMode::Balanced => RagConfig::default(),
                        RagMode::Thorough => RagConfig::thorough(),
                    };
                    config.top_k = top_k;
                    config.min_score = min_score;

                    info!(
                        "RAG query: \"{}\" (mode: {:?}, top_k: {})",
                        query, mode, top_k
                    );

                    // Create RAG engine
                    let engine = RagEngine::persistent(rag_dir.clone())
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to create RAG engine: {}", e))?
                        .with_config(config);

                    // Optionally add LLM client
                    let engine = if !no_llm {
                        match UnifiedLlmClient::default_anthropic() {
                            Ok(client) => engine.with_llm(client),
                            Err(e) => {
                                info!("LLM not available ({}), using retrieval-only mode", e);
                                engine
                            }
                        }
                    } else {
                        engine
                    };

                    let response = engine
                        .query(&query)
                        .await
                        .map_err(|e| anyhow::anyhow!("RAG query failed: {}", e))?;

                    match format {
                        OutputFormat::Json => {
                            println!("{}", serde_json::to_string_pretty(&response)?);
                        }
                        OutputFormat::Text => {
                            print_rag_response(&response);
                        }
                    }
                }

                RagAction::Retrieve {
                    query,
                    top_k,
                    format,
                } => {
                    info!("Retrieving: \"{}\" (top_k: {})", query, top_k);

                    let engine = RagEngine::persistent(rag_dir)
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to create RAG engine: {}", e))?;

                    let results = engine
                        .retrieve(&query, top_k)
                        .await
                        .map_err(|e| anyhow::anyhow!("Retrieval failed: {}", e))?;

                    match format {
                        OutputFormat::Json => {
                            println!("{}", serde_json::to_string_pretty(&results)?);
                        }
                        OutputFormat::Text => {
                            println!();
                            println!("═══════════════════════════════════════════════════════════════════════");
                            println!("                         Retrieval Results");
                            println!("═══════════════════════════════════════════════════════════════════════");
                            println!();
                            println!("Query: \"{}\"", query);
                            println!("Results: {}", results.len());
                            println!();

                            for (i, result) in results.iter().enumerate() {
                                println!("{}. [score: {:.3}]", i + 1, result.score);
                                println!("   {}", truncate_text(&result.text, 200));
                                println!();
                            }
                        }
                    }
                }

                RagAction::Stats => {
                    let engine = RagEngine::persistent(rag_dir)
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to create RAG engine: {}", e))?;

                    let stats = engine
                        .stats()
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to get stats: {}", e))?;

                    println!();
                    println!(
                        "═══════════════════════════════════════════════════════════════════════"
                    );
                    println!("                      RAG Knowledge Base Stats");
                    println!(
                        "═══════════════════════════════════════════════════════════════════════"
                    );
                    println!();
                    println!("  Documents:       {}", stats.document_count);
                    println!("  Chunks:          {}", stats.chunk_count);
                    println!("  Indexed chunks:  {}", stats.indexed_chunks);
                    println!("  Embeddings:      {}", stats.embedding_count);
                    println!("  Storage size:    {} bytes", stats.storage_bytes);
                    println!("  Index size:      {} bytes", stats.index_bytes);
                    println!();
                }
            }
        }

        Commands::Metrics { action } => {
            use reasonkit::thinktool::MetricsTracker;

            let metrics_dir = cli.data_dir.join("metrics");
            std::fs::create_dir_all(&metrics_dir)?;
            let metrics_path = metrics_dir.join("execution_metrics.jsonl");

            match action {
                MetricsAction::Report {
                    format,
                    filter: _filter,
                    output,
                } => {
                    let mut tracker = MetricsTracker::new(&metrics_path);
                    let _ = tracker.load_all(); // Load existing records

                    let report = tracker.generate_report();

                    let output_text = match format {
                        OutputFormat::Json => report.to_json()?,
                        OutputFormat::Text => report.to_text(),
                    };

                    if let Some(output_path) = output {
                        std::fs::write(&output_path, &output_text)?;
                        println!("Report saved to: {}", output_path.display());
                    } else {
                        println!("{}", output_text);
                    }
                }

                MetricsAction::Stats { name, format } => {
                    let mut tracker = MetricsTracker::new(&metrics_path);
                    let _ = tracker.load_all();

                    let stats = tracker.calculate_stats(&name);

                    match format {
                        OutputFormat::Json => {
                            println!("{}", serde_json::to_string_pretty(&stats)?);
                        }
                        OutputFormat::Text => {
                            println!();
                            println!("═══════════════════════════════════════════════════════════════════════");
                            println!("                    Metrics: {}", name);
                            println!("═══════════════════════════════════════════════════════════════════════");
                            println!();
                            if stats.execution_count == 0 {
                                println!("  No executions recorded for '{}'", name);
                            } else {
                                println!(
                                    "  Grade:           {} ({}/100)",
                                    stats.grade, stats.score
                                );
                                println!("  Executions:      {}", stats.execution_count);
                                println!(
                                    "  Avg Confidence:  {:.1}% (±{:.1}%)",
                                    stats.avg_confidence * 100.0,
                                    stats.confidence_std_dev * 100.0
                                );
                                println!(
                                    "  Confidence Range: {:.1}% - {:.1}%",
                                    stats.min_confidence * 100.0,
                                    stats.max_confidence * 100.0
                                );
                                println!("  Success Rate:    {:.1}%", stats.success_rate * 100.0);
                                println!("  Avg Duration:    {:.0}ms", stats.avg_duration_ms);
                                println!(
                                    "  Duration Range:  {}ms - {}ms",
                                    stats.min_duration_ms, stats.max_duration_ms
                                );
                                println!("  Avg Tokens:      {:.0}", stats.avg_tokens);
                            }
                            println!();
                        }
                    }
                }

                MetricsAction::Path => {
                    println!("Metrics storage: {}", metrics_path.display());
                }

                MetricsAction::Clear { yes } => {
                    if !yes {
                        println!("This will delete all metrics data. Use --yes to confirm.");
                    } else if metrics_path.exists() {
                        std::fs::remove_file(&metrics_path)?;
                        println!("Metrics data cleared.");
                    } else {
                        println!("No metrics data found.");
                    }
                }
            }
        }

        Commands::Web {
            query,
            depth,
            web,
            kb,
            provider,
            format,
            output,
        } => {
            info!("Web research run: {}", query);

            // Map depth to profile
            let profile_id = match depth {
                WebDepth::Quick => "quick",
                WebDepth::Standard => "balanced",
                WebDepth::Deep => "deep",
                WebDepth::Exhaustive => "paranoid",
            };

            println!();
            println!("═══════════════════════════════════════════════════════════════════════");
            println!("                     ReasonKit Web: Deep Research");
            println!("═══════════════════════════════════════════════════════════════════════");
            println!();
            println!("Query:   \"{}\"", query);
            println!("Depth:   {:?} (profile: {})", depth, profile_id);
            println!(
                "Sources: {}{}",
                if web { "🌐 Web " } else { "" },
                if kb { "📚 KB" } else { "" }
            );
            println!();

            // Step 1: Web Search (if enabled)
            let mut web_context = String::new();
            if web {
                println!("🌐 Searching the web...");
                let search_config = SearchConfig {
                    provider: SearchProvider::Auto,
                    num_results: match depth {
                        WebDepth::Quick => 3,
                        WebDepth::Standard => 5,
                        WebDepth::Deep => 8,
                        WebDepth::Exhaustive => 10,
                    },
                    ..Default::default()
                };
                let searcher = WebSearcher::new(search_config);

                match searcher.search(&query).await {
                    Ok(results) => {
                        if !results.is_empty() {
                            println!("   Found {} web results", results.len());
                            web_context = results
                                .iter()
                                .enumerate()
                                .map(|(i, r)| {
                                    format!(
                                        "[{}] {}\nURL: {}\n{}",
                                        i + 1,
                                        r.title,
                                        r.url,
                                        r.snippet
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("\n\n");
                        } else {
                            println!("   No web results found");
                        }
                    }
                    Err(e) => {
                        println!(
                            "   Web search failed: {} (continuing without web results)",
                            e
                        );
                    }
                }
            }

            // Step 2: Knowledge Base Search (if enabled and memory feature available)
            #[cfg(feature = "memory")]
            let mut kb_context = String::new();
            #[cfg(not(feature = "memory"))]
            let kb_context = String::new();
            #[cfg(feature = "memory")]
            if kb {
                let index_path = cli.data_dir.join("index");
                if index_path.exists() {
                    if let Ok(index) = IndexManager::open(index_path) {
                        if let Ok(results) = index.search_bm25(&query, 5) {
                            if !results.is_empty() {
                                println!(
                                    "📚 Knowledge Base: Found {} relevant chunks",
                                    results.len()
                                );
                                kb_context = results
                                    .iter()
                                    .map(|r| r.text.clone())
                                    .collect::<Vec<_>>()
                                    .join("\n\n---\n\n");
                            }
                        }
                    }
                }
            }
            #[cfg(not(feature = "memory"))]
            if kb {
                println!("📚 Knowledge Base: Enable 'memory' feature for KB search");
            }

            // Step 2: Execute ThinkTool protocol
            let llm_config = LlmConfig {
                provider: provider.into(),
                model: provider.default_model().to_string(),
                temperature: 0.7,
                max_tokens: 4000,
                ..Default::default()
            };

            let config = ExecutorConfig {
                llm: llm_config,
                use_mock: false,
                save_traces: true,
                trace_dir: Some(cli.data_dir.join("traces")),
                verbose: cli.verbose > 0,
                ..Default::default()
            };

            let executor = ProtocolExecutor::with_config(config)
                .map_err(|e| anyhow::anyhow!("Failed to create executor: {}", e))?;

            // Build enriched input with web and KB context
            let enriched_query = {
                let mut parts = vec![format!("Research Question: {}", query)];

                if !web_context.is_empty() {
                    parts.push(format!("\n\n## Web Search Results\n\n{}", web_context));
                }

                if !kb_context.is_empty() {
                    parts.push(format!("\n\n## Knowledge Base Context\n\n{}", kb_context));
                }

                if web_context.is_empty() && kb_context.is_empty() {
                    query.clone()
                } else {
                    parts.push(
                        "\n\nProvide comprehensive analysis integrating all sources:".to_string(),
                    );
                    parts.join("")
                }
            };

            println!("🧠 Executing {} reasoning protocol...", profile_id);
            println!();

            let input = ProtocolInput::query(&enriched_query);
            let result = executor
                .execute_profile(profile_id, input)
                .await
                .map_err(|e| anyhow::anyhow!("Research failed: {}", e))?;

            // Build output report
            let mut report = String::new();
            report.push_str("# Deep Research Report\n\n");
            report.push_str(&format!("**Query:** {}\n\n", query));
            report.push_str(&format!("**Depth:** {:?}\n\n", depth));
            report.push_str(&format!(
                "**Confidence:** {:.1}%\n\n",
                result.confidence * 100.0
            ));
            report.push_str(&format!("**Duration:** {}ms\n\n", result.duration_ms));
            report.push_str("---\n\n");
            report.push_str("## Analysis\n\n");

            // Extract key findings from result data
            for (key, value) in &result.data {
                if key != "confidence" {
                    report.push_str(&format!("### {}\n\n", key));
                    if let Some(s) = value.as_str() {
                        report.push_str(&format!("{}\n\n", s));
                    } else if let Some(arr) = value.as_array() {
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                report.push_str(&format!("- {}\n", s));
                            }
                        }
                        report.push('\n');
                    }
                }
            }

            // Output based on format
            match format {
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                }
                OutputFormat::Text => {
                    println!("{}", report);
                }
            }

            // Save to file if requested
            if let Some(output_path) = output {
                std::fs::write(&output_path, &report)?;
                println!("📄 Report saved to: {:?}", output_path);
            }

            println!();
            println!("═══════════════════════════════════════════════════════════════════════");
            println!(
                "  Web research complete | Confidence: {:.1}% | Duration: {}ms",
                result.confidence * 100.0,
                result.duration_ms
            );
            println!("═══════════════════════════════════════════════════════════════════════");
        }

        Commands::Verify {
            claim,
            sources: min_sources,
            web,
            kb,
            anchor,
            format,
            output,
        } => {
            info!("Verifying claim: {}", claim);

            println!();
            println!("═══════════════════════════════════════════════════════════════════════");
            println!("                     ReasonKit Verify: Triangulated Truth");
            println!("═══════════════════════════════════════════════════════════════════════");
            println!();
            println!("Claim:    \"{}\"", claim);
            println!("Required: {} independent sources", min_sources);
            println!(
                "Sources:  {}{}",
                if web { "🌐 Web " } else { "" },
                if kb { "📚 KB" } else { "" }
            );
            println!();

            let mut sources_found: Vec<(String, String, String)> = Vec::new(); // (title, url, snippet)

            // Step 1: Web Search for verification
            if web {
                println!("🌐 Searching for verification sources...");
                let search_config = SearchConfig {
                    provider: SearchProvider::Auto,
                    num_results: min_sources * 2,
                    ..Default::default()
                };
                let searcher = WebSearcher::new(search_config);

                match searcher.search(&claim).await {
                    Ok(results) => {
                        println!("   Found {} potential sources", results.len());
                        for r in results {
                            sources_found.push((r.title, r.url, r.snippet));
                        }
                    }
                    Err(e) => {
                        println!("   Web search failed: {}", e);
                    }
                }
            }

            // Step 2: Knowledge Base search (requires memory feature)
            #[cfg(feature = "memory")]
            if kb {
                let index_path = cli.data_dir.join("index");
                if index_path.exists() {
                    if let Ok(index) = IndexManager::open(index_path) {
                        if let Ok(results) = index.search_bm25(&claim, min_sources) {
                            println!("📚 Knowledge Base: {} relevant chunks", results.len());
                            for r in results {
                                sources_found.push((
                                    "Knowledge Base".to_string(),
                                    "local://kb".to_string(),
                                    r.text.clone(),
                                ));
                            }
                        }
                    }
                }
            }
            #[cfg(not(feature = "memory"))]
            if kb {
                println!("📚 Knowledge Base: Enable 'memory' feature for KB search");
            }

            // Step 3: Apply ProofGuard reasoning
            println!("🛡️  Applying ProofGuard verification protocol...");

            let llm_config = LlmConfig {
                provider: LlmProvider::Anthropic,
                model: "claude-sonnet-4-20250514".to_string(),
                temperature: 0.3, // Lower for verification
                max_tokens: 2000,
                ..Default::default()
            };

            let config = ExecutorConfig {
                llm: llm_config,
                use_mock: false,
                save_traces: true,
                trace_dir: Some(cli.data_dir.join("traces")),
                verbose: cli.verbose > 0,
                ..Default::default()
            };

            let executor = ProtocolExecutor::with_config(config)
                .map_err(|e| anyhow::anyhow!("Failed to create executor: {}", e))?;

            // Build verification prompt with sources
            let sources_text = sources_found
                .iter()
                .enumerate()
                .map(|(i, (title, url, snippet))| {
                    format!("[Source {}] {}\nURL: {}\n{}", i + 1, title, url, snippet)
                })
                .collect::<Vec<_>>()
                .join("\n\n");

            let verification_prompt = format!(
                "CLAIM TO VERIFY:\n\"{}\"\n\n\
                SOURCES FOUND:\n{}\n\n\
                VERIFICATION TASK:\n\
                1. Analyze each source for relevance to the claim\n\
                2. Categorize: SUPPORTS, CONTRADICTS, or NEUTRAL\n\
                3. Assess source quality (Tier 1/2/3)\n\
                4. Determine if {} independent sources support the claim\n\
                5. Final verdict: VERIFIED, UNVERIFIED, or CONTRADICTED",
                claim, sources_text, min_sources
            );

            let input = ProtocolInput::query(&verification_prompt);
            let result: ProtocolOutput = executor
                .execute("proofguard", input)
                .await
                .map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;

            // Build verification report
            let mut report = String::new();
            report.push_str("# Verification Report\n\n");
            report.push_str(&format!("**Claim:** {}\n\n", claim));
            report.push_str(&format!("**Sources Found:** {}\n", sources_found.len()));
            report.push_str(&format!(
                "**Required:** {} independent sources\n\n",
                min_sources
            ));
            report.push_str("---\n\n");
            report.push_str("## Sources\n\n");
            for (i, (title, url, _)) in sources_found.iter().enumerate() {
                report.push_str(&format!("{}. [{}]({})\n", i + 1, title, url));
            }
            report.push_str("\n---\n\n");
            report.push_str("## Analysis\n\n");

            for (key, value) in result.data.iter() {
                if key != "confidence" {
                    if let Some(s) = value.as_str() {
                        report.push_str(&format!("{}\n\n", s));
                    } else {
                        // For non-string values, convert to string
                        report.push_str(&format!("{}\n\n", value));
                    }
                }
            }

            // Verification status
            let verified = sources_found.len() >= min_sources && result.confidence > 0.7;

            println!();
            if verified {
                println!(
                    "✅ CLAIM VERIFIED ({}/{} sources, {:.0}% confidence)",
                    sources_found.len(),
                    min_sources,
                    result.confidence * 100.0
                );
            } else if result.confidence < 0.3 {
                println!(
                    "❌ CLAIM CONTRADICTED ({} sources, {:.0}% confidence)",
                    sources_found.len(),
                    result.confidence * 100.0
                );
            } else {
                println!(
                    "⚠️  CLAIM UNVERIFIED ({}/{} sources, {:.0}% confidence)",
                    sources_found.len(),
                    min_sources,
                    result.confidence * 100.0
                );
            }

            // Output based on format
            match format {
                OutputFormat::Json => {
                    let json_report = serde_json::json!({
                        "claim": claim,
                        "verified": verified,
                        "confidence": result.confidence,
                        "sources_found": sources_found.len(),
                        "sources_required": min_sources,
                        "sources": sources_found.iter().map(|(t, u, _)| {
                            serde_json::json!({"title": t, "url": u})
                        }).collect::<Vec<_>>(),
                    });
                    println!("{}", serde_json::to_string_pretty(&json_report)?);
                }
                OutputFormat::Text => {
                    println!("\n{}", report);
                }
            }

            // Save to file if requested
            if let Some(output_path) = output {
                std::fs::write(&output_path, &report)?;
                println!("📄 Report saved to: {:?}", output_path);
            }

            // Anchor to ProofLedger if requested and verified
            if anchor && verified {
                println!("⚓ Anchoring to ProofLedger...");

                // Create or open the ProofLedger
                let ledger_path = cli.data_dir.join("proof_ledger.db");

                if let Some(parent) = ledger_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }

                match ProofLedger::new(&ledger_path) {
                    Ok(ledger) => {
                        // Generate a source URL from the claim
                        let source_url = format!(
                            "reasonkit://verify/{}",
                            claim
                                .chars()
                                .filter(|c: &char| c.is_alphanumeric() || *c == ' ')
                                .take(50)
                                .collect::<String>()
                                .replace(' ', "-")
                                .to_lowercase()
                        );

                        // Anchor the verified report
                        match ledger.anchor(
                            &report,
                            &source_url,
                            Some(format!(
                                r#"{{"claim": "{}", "sources": {}, "verified": true}}"#,
                                claim.replace('"', "\\\""),
                                min_sources
                            )),
                        ) {
                            Ok(hash) => {
                                println!("   ✓ Anchored with hash: {}", &hash[..16]);
                                println!("   📁 Ledger: {:?}", ledger_path);
                            }
                            Err(e) => {
                                println!("   ⚠ Anchoring failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("   ⚠ Could not open ProofLedger: {}", e);
                    }
                }
            }

            println!();
            println!("═══════════════════════════════════════════════════════════════════════");
        }

        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            generate(shell, &mut cmd, "rk-core", &mut std::io::stdout());
        }
    }

    Ok(())
}

/// Print available protocols and profiles
fn print_thinktool_list() -> anyhow::Result<()> {
    let executor = ProtocolExecutor::mock()
        .map_err(|e| anyhow::anyhow!("Failed to create executor: {}", e))?;

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                     ReasonKit ThinkTool Protocols");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("PROTOCOLS (individual reasoning modules):");
    println!("──────────────────────────────────────────────────────────────────────");

    for id in executor.list_protocols() {
        if let Some(p) = executor.get_protocol(id) {
            println!("  {:15} - {}", id, p.description);
        }
    }

    println!();
    println!("PROFILES (protocol chains):");
    println!("──────────────────────────────────────────────────────────────────────");

    for id in executor.list_profiles() {
        if let Some(p) = executor.get_profile(id) {
            let chain: Vec<&str> = p.chain.iter().map(|s| s.protocol_id.as_str()).collect();
            println!(
                "  {:15} - {} ({:.0}% confidence)",
                id,
                p.description,
                p.min_confidence * 100.0
            );
            println!("                    Chain: {}", chain.join(" → "));
        }
    }

    println!();
    println!("USAGE EXAMPLES:");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  rk-core think \"What drives startup success?\"");
    println!("  rk-core think \"Is microservices the right choice?\" --protocol laserlogic");
    println!("  rk-core think \"Should we use Rust or Go?\" --profile paranoid");
    println!("  rk-core think \"Analyze this claim\" --mock  # Testing without API key");
    println!();
    println!("BUDGET EXAMPLES (adaptive compute time):");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  rk-core think \"Query\" --budget 30s       # Max 30 seconds");
    println!("  rk-core think \"Query\" --budget 2m        # Max 2 minutes");
    println!("  rk-core think \"Query\" --budget 1000t     # Max 1000 tokens");
    println!("  rk-core think \"Query\" --budget $0.50     # Max $0.50 USD");
    println!();

    Ok(())
}

#[allow(dead_code)]
struct Theme {
    primary: &'static str,
    secondary: &'static str,
    accent: &'static str,
    title: &'static str,   // For tool name styling
    tagline: &'static str, // For tagline styling
    reset: &'static str,
    border: &'static str,
}

impl Theme {
    fn for_protocol(protocol: &str) -> Self {
        match protocol {
            // 💡 GIGATHINK: Gold + Purple, Bold + Italic = Brilliant Ideas/Expansive
            "gigathink" => Theme {
                primary: "\x1b[38;5;220m",   // Gold/Bright Yellow
                secondary: "\x1b[38;5;135m", // Purple
                accent: "\x1b[1;38;5;220m",  // Bold Gold
                title: "\x1b[1;3;38;5;220m", // Bold Italic Gold
                tagline: "\x1b[3;38;5;135m", // Italic Purple
                reset: "\x1b[0m",
                border: "\x1b[38;5;220m",
            },
            // 🌈 POWERCOMBO: Rainbow cycling = All Tools Combined/Maximum Power
            "powercombo" => Theme {
                primary: "\x1b[38;5;196m",  // Red (start of rainbow)
                secondary: "\x1b[38;5;21m", // Blue (end of rainbow)
                accent: "\x1b[1;38;5;226m", // Bold Yellow (middle)
                title: "\x1b[1;38;5;196m",  // Bold Red
                tagline: "\x1b[38;5;46m",   // Green
                reset: "\x1b[0m",
                border: "\x1b[38;5;201m", // Magenta
            },
            // ⚡ LASERLOGIC: Green + White, Bold + Underline = Precision/Sharp
            "laserlogic" => Theme {
                primary: "\x1b[32m",   // Green
                secondary: "\x1b[97m", // Bright White
                accent: "\x1b[1;32m",  // Bold Green
                title: "\x1b[1;4;32m", // Bold Underline Green
                tagline: "\x1b[2;37m", // Dim White
                reset: "\x1b[0m",
                border: "\x1b[32m",
            },
            // 🛡️ PROOFGUARD: White + Blue, Bold + Reverse = Authoritative/Verified
            "proofguard" => Theme {
                primary: "\x1b[1;37m",   // Bold White
                secondary: "\x1b[34m",   // Blue
                accent: "\x1b[1;47;34m", // Bold White BG Blue FG
                title: "\x1b[1;7;37m",   // Bold Reverse White
                tagline: "\x1b[3;34m",   // Italic Blue
                reset: "\x1b[0m",
                border: "\x1b[1;37m",
            },
            // 🔥 BRUTALHONESTY: Red + Yellow, Bold + Blink-like = Aggressive/Warning
            "brutalhonesty" => Theme {
                primary: "\x1b[31m",   // Red
                secondary: "\x1b[33m", // Yellow
                accent: "\x1b[1;31m",  // Bold Red
                title: "\x1b[1;4;31m", // Bold Underline Red (aggressive)
                tagline: "\x1b[1;33m", // Bold Yellow (warning)
                reset: "\x1b[0m",
                border: "\x1b[31m",
            },
            // 🪨 BEDROCK: Yellow/Amber + Gray, Dim = Solid/Foundational
            "bedrock" => Theme {
                primary: "\x1b[33m",   // Yellow
                secondary: "\x1b[90m", // Gray
                accent: "\x1b[1;33m",  // Bold Yellow
                title: "\x1b[1;33m",   // Bold Yellow (solid)
                tagline: "\x1b[2;90m", // Dim Gray (grounded)
                reset: "\x1b[0m",
                border: "\x1b[33m",
            },
            // Default: Blue
            _ => Theme {
                primary: "\x1b[34m",   // Blue
                secondary: "\x1b[37m", // White
                accent: "\x1b[1;34m",  // Bold Blue
                title: "\x1b[1;34m",   // Bold Blue
                tagline: "\x1b[2;37m", // Dim White
                reset: "\x1b[0m",
                border: "\x1b[34m",
            },
        }
    }
}

/// ThinkTool branding - COMPACT design for minimal CLI space
#[allow(dead_code)]
struct ToolBranding {
    name: &'static str,
    tagline: &'static str,
    icon: &'static str,
    /// Compact 2-line mini logo
    mini_logo: [&'static str; 2],
    /// Border character style (for future use)
    border_char: char,
}

/// Get complete ThinkTool branding for display - COMPACT version
fn get_tool_branding(protocol: &str) -> ToolBranding {
    match protocol {
        "gigathink" => ToolBranding {
            name: "GigaThink",
            tagline: "10+ Perspectives · Brilliant Ideas",
            icon: "💡",
            mini_logo: [
                "╭─────────────────────────────────╮",
                "│ 💡 GIGATHINK · Idea Explosion  │",
            ],
            border_char: '◆',
        },
        "powercombo" => ToolBranding {
            name: "PowerCombo",
            tagline: "All Tools · Maximum Reasoning Power",
            icon: "🌈",
            mini_logo: [
                "╭─────────────────────────────────╮",
                "│ 🌈 POWERCOMBO · Ultimate Mode  │",
            ],
            border_char: '★',
        },
        "laserlogic" => ToolBranding {
            name: "LaserLogic",
            tagline: "Precision Deduction · Fallacy Detection",
            icon: "⚡",
            mini_logo: [
                "╭─────────────────────────────────╮",
                "│ ⚡ LASERLOGIC · Precision Mode │",
            ],
            border_char: '▸',
        },
        "bedrock" => ToolBranding {
            name: "BedRock",
            tagline: "First Principles · Axiom Analysis",
            icon: "🪨",
            mini_logo: [
                "╭─────────────────────────────────╮",
                "│ 🪨 BEDROCK · Foundation Layer  │",
            ],
            border_char: '▪',
        },
        "proofguard" => ToolBranding {
            name: "ProofGuard",
            tagline: "3-Source Triangulation · Verification",
            icon: "🛡️",
            mini_logo: [
                "╭─────────────────────────────────╮",
                "│ 🛡️  PROOFGUARD · Truth Engine  │",
            ],
            border_char: '◈',
        },
        "brutalhonesty" => ToolBranding {
            name: "BrutalHonesty",
            tagline: "Adversarial Critique · No Mercy",
            icon: "🔥",
            mini_logo: [
                "╭─────────────────────────────────╮",
                "│ 🔥 BRUTALHONESTY · Red Team    │",
            ],
            border_char: '✖',
        },
        _ => ToolBranding {
            name: "ReasonKit",
            tagline: "Turn Prompts into Protocols",
            icon: "🧠",
            mini_logo: [
                "╭─────────────────────────────────╮",
                "│ 🧠 REASONKIT · ThinkTools      │",
            ],
            border_char: '─',
        },
    }
}

/// Legacy function for backward compatibility
#[allow(dead_code)]
fn get_thinktool_description(protocol: &str) -> (&'static str, &'static str) {
    let branding = get_tool_branding(protocol);
    (branding.name, branding.tagline)
}

/// Get confidence level interpretation
fn interpret_confidence(conf: f64) -> (&'static str, &'static str) {
    if conf >= 0.95 {
        ("VERY HIGH", "\x1b[1;32m")
    }
    // Bold Green
    else if conf >= 0.80 {
        ("HIGH", "\x1b[32m")
    }
    // Green
    else if conf >= 0.60 {
        ("MODERATE", "\x1b[33m")
    }
    // Yellow
    else if conf >= 0.40 {
        ("LOW", "\x1b[33m")
    }
    // Yellow
    else {
        ("VERY LOW", "\x1b[31m")
    } // Red
}

/// Print execution result in text format - COMPACT BRANDED OUTPUT
fn print_think_result(result: &reasonkit::thinktool::ProtocolOutput) {
    let theme = Theme::for_protocol(&result.protocol_id);
    let branding = get_tool_branding(&result.protocol_id);
    let (conf_level, conf_color) = interpret_confidence(result.confidence);
    let reset = "\x1b[0m";
    let dim = "\x1b[2m";
    let bold = "\x1b[1m";

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPACT BRANDED HEADER (2 lines - tool name + separator)
    // ═══════════════════════════════════════════════════════════════════════════
    println!();
    println!(
        "{}{} {}{}{} {} · {}{}{}",
        theme.primary,
        branding.icon,
        theme.title,
        branding.name,
        reset,
        theme.primary,
        theme.tagline,
        branding.tagline,
        reset
    );
    println!(
        "{}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{}",
        theme.border, reset
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // EXECUTION METRICS
    // ═══════════════════════════════════════════════════════════════════════════
    println!();
    println!(
        "{}┌─────────────────────────────────────────────────────────────────────────┐{}",
        theme.border, reset
    );
    println!(
        "{}│{}  {}EXECUTION METRICS{}                                                      {}│{}",
        theme.border, reset, bold, reset, theme.border, reset
    );
    println!(
        "{}├─────────────────────────────────────────────────────────────────────────┤{}",
        theme.border, reset
    );

    // Status
    let status_icon = if result.success { "✓" } else { "✗" };
    let status_color = if result.success {
        "\x1b[32m"
    } else {
        "\x1b[31m"
    };
    let status_text = if result.success { "SUCCESS" } else { "FAILED" };
    println!(
        "{}│{}  Status:        {}{} {:<10}{}                                       {}│{}",
        theme.border, reset, status_color, status_icon, status_text, reset, theme.border, reset
    );

    // Confidence with interpretation and visual bar
    let conf_pct = result.confidence * 100.0;
    let conf_bar_filled = (result.confidence * 20.0) as usize;
    let conf_bar_empty = 20 - conf_bar_filled;
    let conf_bar = format!(
        "{}{}{}{}",
        conf_color,
        "█".repeat(conf_bar_filled),
        dim,
        "░".repeat(conf_bar_empty)
    );
    println!(
        "{}│{}  Confidence:    {}{:.1}%{} ({}{}{}) [{}{}]         {}│{}",
        theme.border,
        reset,
        conf_color,
        conf_pct,
        reset,
        conf_color,
        conf_level,
        reset,
        conf_bar,
        reset,
        theme.border,
        reset
    );

    // Duration
    let duration_secs = result.duration_ms as f64 / 1000.0;
    println!(
        "{}│{}  Duration:      {}{:.2}s{} ({}ms)                                        {}│{}",
        theme.border,
        reset,
        theme.primary,
        duration_secs,
        reset,
        result.duration_ms,
        theme.border,
        reset
    );

    // Tokens
    println!(
        "{}│{}  Tokens:        {}in:{} {} {}out:{} {} {}total:{} {}                  {}│{}",
        theme.border,
        reset,
        dim,
        reset,
        result.tokens.input_tokens,
        dim,
        reset,
        result.tokens.output_tokens,
        dim,
        reset,
        result.tokens.total_tokens,
        theme.border,
        reset
    );

    // Cost
    println!(
        "{}│{}  Cost:          {}${:.4} USD{}                                           {}│{}",
        theme.border, reset, theme.primary, result.tokens.cost_usd, reset, theme.border, reset
    );

    println!(
        "{}└─────────────────────────────────────────────────────────────────────────┘{}",
        theme.border, reset
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // BUDGET SUMMARY (if applicable)
    // ═══════════════════════════════════════════════════════════════════════════
    if let Some(budget) = &result.budget_summary {
        println!();
        println!(
            "{}┌─────────────────────────────────────────────────────────────────────────┐{}",
            theme.border, reset
        );
        println!("{}│{}  {}BUDGET ANALYSIS{}                                                        {}│{}",
            theme.border, reset, bold, reset, theme.border, reset);
        println!(
            "{}├─────────────────────────────────────────────────────────────────────────┤{}",
            theme.border, reset
        );

        let usage_pct = budget.usage_ratio * 100.0;
        let usage_color = if usage_pct > 90.0 {
            "\x1b[31m"
        } else if usage_pct > 70.0 {
            "\x1b[33m"
        } else {
            "\x1b[32m"
        };
        println!(
            "{}│{}  Usage:         {}{:.1}%{} of budget consumed                             {}│{}",
            theme.border, reset, usage_color, usage_pct, reset, theme.border, reset
        );
        println!("{}│{}  Steps:         \x1b[32m{}{} completed, \x1b[33m{}{} skipped                            {}│{}",
            theme.border, reset, budget.steps_completed, reset,
            budget.steps_skipped, reset, theme.border, reset);
        println!(
            "{}│{}  Tokens Used:   {}{}{}                                                  {}│{}",
            theme.border, reset, theme.primary, budget.tokens_used, reset, theme.border, reset
        );
        println!(
            "{}│{}  Cost Incurred: {}${:.4}{}                                               {}│{}",
            theme.border, reset, theme.primary, budget.cost_incurred, reset, theme.border, reset
        );
        println!(
            "{}│{}  Elapsed:       {}{}ms{}                                                 {}│{}",
            theme.border,
            reset,
            theme.primary,
            budget.elapsed.as_millis(),
            reset,
            theme.border,
            reset
        );

        if budget.exhausted {
            println!("{}│{}  \x1b[1;33m⚠ BUDGET EXHAUSTED - Some steps may have been skipped{}               {}│{}",
                theme.border, reset, reset, theme.border, reset);
        }
        println!(
            "{}└─────────────────────────────────────────────────────────────────────────┘{}",
            theme.border, reset
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXECUTION TRACE
    // ═══════════════════════════════════════════════════════════════════════════
    if !result.steps.is_empty() {
        println!();
        println!(
            "{}┌─────────────────────────────────────────────────────────────────────────┐{}",
            theme.border, reset
        );
        println!("{}│{}  {}EXECUTION TRACE{}                                                        {}│{}",
            theme.border, reset, bold, reset, theme.border, reset);
        println!(
            "{}├─────────────────────────────────────────────────────────────────────────┤{}",
            theme.border, reset
        );

        for (i, step) in result.steps.iter().enumerate() {
            let step_status = if step.success { "✓" } else { "✗" };
            let step_color = if step.success { "\x1b[32m" } else { "\x1b[31m" };
            let step_conf_pct = step.confidence * 100.0;
            let (step_level, step_conf_color) = interpret_confidence(step.confidence);

            println!(
                "{}│{}  {}[{:02}]{} {} {} {:<30}                            {}│{}",
                theme.border,
                reset,
                dim,
                i + 1,
                reset,
                step_color,
                step_status,
                reset,
                theme.border,
                reset
            );
            println!(
                "{}│{}       Step:       {}{}{}                                              {}│{}",
                theme.border, reset, theme.secondary, step.step_id, reset, theme.border, reset
            );
            println!("{}│{}       Confidence: {}{:.1}%{} ({}{}{})                                  {}│{}",
                theme.border, reset, step_conf_color, step_conf_pct, reset,
                step_conf_color, step_level, reset, theme.border, reset);
            println!(
                "{}│{}       Duration:   {}{}ms{}                                            {}│{}",
                theme.border, reset, dim, step.duration_ms, reset, theme.border, reset
            );

            if i < result.steps.len() - 1 {
                println!("{}│{}       {}↓{}                                                            {}│{}",
                    theme.border, reset, dim, reset, theme.border, reset);
            }
        }
        println!(
            "{}└─────────────────────────────────────────────────────────────────────────┘{}",
            theme.border, reset
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // OUTPUT DATA
    // ═══════════════════════════════════════════════════════════════════════════
    println!();
    println!(
        "{}┌─────────────────────────────────────────────────────────────────────────┐{}",
        theme.border, reset
    );
    println!(
        "{}│{}  {}OUTPUT DATA{}                                                            {}│{}",
        theme.border, reset, bold, reset, theme.border, reset
    );
    println!(
        "{}├─────────────────────────────────────────────────────────────────────────┤{}",
        theme.border, reset
    );

    for (key, value) in &result.data {
        if key == "confidence" {
            continue;
        }
        println!(
            "{}│{}  {}{}:{}                                                                  {}│{}",
            theme.border, reset, theme.secondary, key, reset, theme.border, reset
        );
        let display = format_json_value(value, 4);
        for line in display.lines() {
            println!(
                "{}│{} {}                                                                   {}│{}",
                theme.border, reset, line, theme.border, reset
            );
        }
    }
    println!(
        "{}└─────────────────────────────────────────────────────────────────────────┘{}",
        theme.border, reset
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // ERROR (if any)
    // ═══════════════════════════════════════════════════════════════════════════
    if let Some(err) = &result.error {
        println!();
        println!("\x1b[31m┌─────────────────────────────────────────────────────────────────────────┐\x1b[0m");
        println!("\x1b[31m│\x1b[0m  \x1b[1;31mERROR\x1b[0m                                                                   \x1b[31m│\x1b[0m");
        println!("\x1b[31m├─────────────────────────────────────────────────────────────────────────┤\x1b[0m");
        println!("\x1b[31m│\x1b[0m  {}                                                                    \x1b[31m│\x1b[0m", err);
        println!("\x1b[31m└─────────────────────────────────────────────────────────────────────────┘\x1b[0m");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FOOTER
    // ═══════════════════════════════════════════════════════════════════════════
    println!();
    println!(
        "{}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{}",
        dim, reset
    );
    println!(
        "{}  ReasonKit · Turn Prompts into Protocols · https://reasonkit.sh{}",
        dim, reset
    );
    println!(
        "{}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{}",
        dim, reset
    );
    println!();
}

/// Format JSON value for display
fn format_json_value(value: &serde_json::Value, indent: usize) -> String {
    let prefix = " ".repeat(indent);
    match value {
        serde_json::Value::String(s) => format!("{}\"{}\"", prefix, s),
        serde_json::Value::Array(arr) => {
            // Check if this is an array of ListItem-like objects (has "content" field)
            let is_list_items = arr.iter().all(|v| {
                v.as_object()
                    .map(|obj| obj.contains_key("content"))
                    .unwrap_or(false)
            });

            if is_list_items && !arr.is_empty() {
                // Format as a clean numbered list
                let items: Vec<String> = arr
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let content = v
                            .get("content")
                            .and_then(|c| c.as_str())
                            .unwrap_or("(empty)");
                        format!("{}{}. {}", prefix, i + 1, content)
                    })
                    .collect();
                items.join("\n")
            } else {
                let items: Vec<String> = arr
                    .iter()
                    .map(|v| format_json_value(v, indent + 2))
                    .collect();
                if items.is_empty() {
                    format!("{}[]", prefix)
                } else {
                    format!("{}[\n{}\n{}]", prefix, items.join(",\n"), prefix)
                }
            }
        }
        serde_json::Value::Object(obj) => {
            // Check if this is a StepOutput-like object with "type" and "items"/"content"
            if let Some(type_val) = obj.get("type") {
                if type_val == "list" {
                    if let Some(items) = obj.get("items") {
                        return format_json_value(items, indent);
                    }
                } else if type_val == "text" {
                    if let Some(content) = obj.get("content").and_then(|c| c.as_str()) {
                        return format!("{}{}", prefix, content);
                    }
                }
            }

            // Default object formatting
            let items: Vec<String> = obj
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}{}: {}",
                        " ".repeat(indent + 2),
                        k,
                        format_json_value(v, 0).trim()
                    )
                })
                .collect();
            if items.is_empty() {
                format!("{}{{}}", prefix)
            } else {
                format!("{}{{\n{}\n{}}}", prefix, items.join(",\n"), prefix)
            }
        }
        _ => format!("{}{}", prefix, value),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRACE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

/// List traces in a directory
fn list_traces(dir: &PathBuf, protocol_filter: Option<&str>, limit: usize) -> anyhow::Result<()> {
    if !dir.exists() {
        println!(
            "No traces found. Directory does not exist: {}",
            dir.display()
        );
        println!("Run 'rk-core think --save-trace \"your query\"' to create traces.");
        return Ok(());
    }

    let mut traces: Vec<(String, ExecutionTrace)> = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(trace) = serde_json::from_str::<ExecutionTrace>(&content) {
                    // Apply protocol filter
                    if let Some(filter) = protocol_filter {
                        if !trace.protocol_id.contains(filter) {
                            continue;
                        }
                    }
                    traces.push((path.display().to_string(), trace));
                }
            }
        }
    }

    // Sort by timestamp (newest first)
    traces.sort_by(|a, b| b.1.timing.started_at.cmp(&a.1.timing.started_at));

    // Apply limit
    traces.truncate(limit);

    if traces.is_empty() {
        println!("No traces found.");
        return Ok(());
    }

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                       Execution Traces");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "{:36}  {:15}  {:10}  {:8}  Duration",
        "ID", "Protocol", "Status", "Conf."
    );
    println!("──────────────────────────────────────────────────────────────────────");

    for (_, trace) in &traces {
        let status = match trace.status {
            ExecutionStatus::Completed => "✓ Done",
            ExecutionStatus::Failed => "✗ Failed",
            ExecutionStatus::Running => "⋯ Running",
            ExecutionStatus::Cancelled => "⊘ Cancel",
            ExecutionStatus::TimedOut => "⏱ Timeout",
            ExecutionStatus::Paused => "⏸ Paused",
        };

        println!(
            "{:36}  {:15}  {:10}  {:6.1}%  {}ms",
            trace.id.to_string(),
            &trace.protocol_id[..trace.protocol_id.len().min(15)],
            status,
            trace.confidence * 100.0,
            trace.timing.total_duration_ms
        );
    }

    println!();
    println!("Total: {} traces", traces.len());
    println!("Use 'rk-core trace view <ID>' to view details.");

    Ok(())
}

/// View a specific trace
fn view_trace(id: &str, dir: &PathBuf, format: OutputFormat) -> anyhow::Result<()> {
    // Try to find trace by ID or path
    let trace_path = if std::path::Path::new(id).exists() {
        PathBuf::from(id)
    } else {
        // Search for trace by ID
        find_trace_by_id(id, dir)?
    };

    let content = std::fs::read_to_string(&trace_path)
        .map_err(|e| anyhow::anyhow!("Failed to read trace: {}", e))?;

    let trace: ExecutionTrace = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse trace: {}", e))?;

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&trace)?);
        }
        OutputFormat::Text => {
            print_trace_details(&trace);
        }
    }

    Ok(())
}

/// Find trace file by ID
fn find_trace_by_id(id: &str, dir: &PathBuf) -> anyhow::Result<PathBuf> {
    if !dir.exists() {
        return Err(anyhow::anyhow!(
            "Trace directory does not exist: {}",
            dir.display()
        ));
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if filename.contains(id) {
            return Ok(path);
        }
    }

    Err(anyhow::anyhow!("Trace not found: {}", id))
}

/// Print trace details in text format
fn print_trace_details(trace: &ExecutionTrace) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                         Execution Trace");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("Trace ID:    {}", trace.id);
    println!(
        "Protocol:    {} v{}",
        trace.protocol_id, trace.protocol_version
    );
    println!("Status:      {:?}", trace.status);
    println!("Confidence:  {:.1}%", trace.confidence * 100.0);
    println!();

    println!("TIMING:");
    println!("──────────────────────────────────────────────────────────────────────");
    if let Some(started) = trace.timing.started_at {
        println!("  Started:    {}", started.format("%Y-%m-%d %H:%M:%S UTC"));
    }
    if let Some(completed) = trace.timing.completed_at {
        println!(
            "  Completed:  {}",
            completed.format("%Y-%m-%d %H:%M:%S UTC")
        );
    }
    println!(
        "  Duration:   {}ms total, {}ms in LLM",
        trace.timing.total_duration_ms, trace.timing.llm_duration_ms
    );
    println!();

    println!("TOKENS:");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  Input:  {}", trace.tokens.input_tokens);
    println!("  Output: {}", trace.tokens.output_tokens);
    println!("  Total:  {}", trace.tokens.total_tokens);
    println!("  Cost:   ${:.4}", trace.tokens.cost_usd);
    println!();

    println!("METADATA:");
    println!("──────────────────────────────────────────────────────────────────────");
    if let Some(model) = &trace.metadata.model {
        println!("  Model:    {}", model);
    }
    if let Some(provider) = &trace.metadata.provider {
        println!("  Provider: {}", provider);
    }
    if let Some(temp) = trace.metadata.temperature {
        println!("  Temp:     {}", temp);
    }
    println!();

    println!("INPUT:");
    println!("──────────────────────────────────────────────────────────────────────");
    println!(
        "{}",
        serde_json::to_string_pretty(&trace.input).unwrap_or_default()
    );
    println!();

    println!("STEPS ({}):", trace.steps.len());
    println!("──────────────────────────────────────────────────────────────────────");
    for (i, step) in trace.steps.iter().enumerate() {
        let status_icon = match step.status {
            StepStatus::Completed => "✓",
            StepStatus::Failed => "✗",
            StepStatus::Skipped => "⊘",
            StepStatus::Running => "⋯",
            StepStatus::Pending => "○",
        };

        println!(
            "  {}. {} {} (confidence: {:.1}%, {}ms)",
            i + 1,
            status_icon,
            step.step_id,
            step.confidence * 100.0,
            step.duration_ms
        );

        if !step.prompt.is_empty() {
            let prompt_preview = if step.prompt.len() > 100 {
                format!("{}...", &step.prompt[..100])
            } else {
                step.prompt.clone()
            };
            println!("     Prompt: {}", prompt_preview.replace('\n', " "));
        }

        if let Some(err) = &step.error {
            println!("     Error: {}", err);
        }
    }
    println!();

    if let Some(output) = &trace.output {
        println!("OUTPUT:");
        println!("──────────────────────────────────────────────────────────────────────");
        println!(
            "{}",
            serde_json::to_string_pretty(output).unwrap_or_default()
        );
    }
    println!();
}

/// Clean up traces
fn clean_traces(dir: &PathBuf, all: bool, keep_days: Option<u32>) -> anyhow::Result<()> {
    if !dir.exists() {
        println!("No traces to clean. Directory does not exist.");
        return Ok(());
    }

    if !all && keep_days.is_none() {
        return Err(anyhow::anyhow!(
            "Specify --all or --keep-days to clean traces"
        ));
    }

    let cutoff = keep_days.map(|days| chrono::Utc::now() - chrono::Duration::days(days as i64));

    let mut deleted = 0;
    let mut kept = 0;

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }

        let should_delete = if all {
            true
        } else if let Some(cutoff_time) = cutoff {
            // Check file modification time
            if let Ok(metadata) = path.metadata() {
                if let Ok(modified) = metadata.modified() {
                    let modified_dt: chrono::DateTime<chrono::Utc> = modified.into();
                    modified_dt < cutoff_time
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if should_delete {
            std::fs::remove_file(&path)?;
            deleted += 1;
        } else {
            kept += 1;
        }
    }

    println!("Cleaned {} traces, kept {}", deleted, kept);

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// RAG HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Print RAG response in text format
#[cfg(feature = "memory")]
fn print_rag_response(response: &reasonkit::rag::RagResponse) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                           RAG Response");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("Query: \"{}\"", response.query);
    println!();

    println!("ANSWER:");
    println!("──────────────────────────────────────────────────────────────────────");
    println!("{}", response.answer);
    println!();

    if !response.sources.is_empty() {
        println!("SOURCES ({}):", response.sources.len());
        println!("──────────────────────────────────────────────────────────────────────");
        for (i, source) in response.sources.iter().enumerate() {
            println!(
                "  {}. [score: {:.3}] {}",
                i + 1,
                source.score,
                truncate_text(&source.text, 100)
            );
        }
        println!();
    }

    println!("RETRIEVAL STATS:");
    println!("──────────────────────────────────────────────────────────────────────");
    println!(
        "  Chunks retrieved: {}",
        response.retrieval_stats.chunks_retrieved
    );
    println!(
        "  Chunks used:      {}",
        response.retrieval_stats.chunks_used
    );
    println!(
        "  Context tokens:   {}",
        response.retrieval_stats.context_tokens
    );
    println!(
        "  Retrieval time:   {}ms",
        response.retrieval_stats.retrieval_time_ms
    );
    if let Some(tokens) = response.tokens_used {
        println!("  LLM tokens:       {}", tokens);
    }
    println!();
}

/// Truncate text to specified length
#[cfg(feature = "memory")]
fn truncate_text(text: &str, max_len: usize) -> String {
    let text = text.replace('\n', " ");
    if text.len() <= max_len {
        text
    } else {
        format!("{}...", &text[..max_len.saturating_sub(3)])
    }
}
