//! ReasonKit Core CLI
//!
//! AI Thinking Enhancement System - Turn Prompts into Protocols

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use std::path::{Path, PathBuf};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use reasonkit::thinktool::llm::LlmProvider;
use reasonkit::thinktool::{BudgetConfig, ExecutorConfig, ProtocolExecutor, ProtocolInput};

// Import MCP CLI module
#[path = "bin/mcp_cli.rs"]
mod mcp_cli;
use mcp_cli::{run_mcp_command, McpCli};

#[derive(Parser)]
#[command(name = "rk")]
#[command(author = "ReasonKit Team <for@ReasonKit.sh>")]
#[command(version)]
#[command(about = "AI Thinking Enhancement System - Turn Prompts into Protocols")]
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

    /// Initialize telemetry database (runs automatically on first use)
    #[arg(long, hide = true)]
    init_telemetry: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// \[CORE\] Execute structured reasoning protocols (ThinkTools)
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

        /// Budget for adaptive compute time (e.g., "30s", "5m", "1000t", "$0.50")
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

    /// Manage MCP (Model Context Protocol) servers and tools
    Mcp(McpCli),

    /// Start the ReasonKit Core MCP Server
    ServeMcp,

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
    #[command(alias = "m")]
    Metrics {
        #[command(subcommand)]
        action: MetricsAction,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum WebDepth {
    Quick,
    Standard,
    Deep,
    Exhaustive,
}

#[cfg(feature = "memory")]
#[derive(Subcommand)]
enum RagAction {
    Query {
        query: String,
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,
        #[arg(long, default_value = "0.1")]
        min_score: f32,
        #[arg(long, default_value = "balanced")]
        mode: RagMode,
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
        #[arg(long)]
        no_llm: bool,
    },
    Retrieve {
        query: String,
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
    },
    Stats,
}

#[cfg(feature = "memory")]
#[derive(Clone, Copy, Debug, ValueEnum)]
enum RagMode {
    Quick,
    Balanced,
    Thorough,
}

#[derive(Subcommand)]
enum TraceAction {
    List {
        #[arg(short, long)]
        dir: Option<PathBuf>,
        #[arg(short, long)]
        protocol: Option<String>,
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    View {
        id: String,
        #[arg(short, long)]
        dir: Option<PathBuf>,
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
    },
    Clean {
        #[arg(short, long)]
        dir: Option<PathBuf>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        keep_days: Option<u32>,
    },
}

#[derive(Subcommand)]
enum MetricsAction {
    Report {
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
        #[arg(short, long)]
        filter: Option<String>,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    Stats {
        name: String,
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
    },
    Path,
    Clear {
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProviderArg {
    Anthropic,
    #[value(name = "openai")]
    OpenAI,
    Gemini,
    Vertex,
    Azure,
    Bedrock,
    Xai,
    Groq,
    Mistral,
    Deepseek,
    Cohere,
    Perplexity,
    Cerebras,
    Together,
    Fireworks,
    Qwen,
    Cloudflare,
    #[value(name = "openrouter")]
    OpenRouter,
    #[value(name = "claude-cli")]
    ClaudeCli,
    #[value(name = "codex-cli")]
    CodexCli,
    #[value(name = "gemini-cli")]
    GeminiCli,
    #[value(name = "opencode-cli")]
    OpencodeCli,
    #[value(name = "copilot-cli")]
    CopilotCli,
}

impl ProviderArg {
    #[allow(dead_code)]
    pub fn default_model(&self) -> &'static str {
        match self {
            ProviderArg::Anthropic => "claude-sonnet-4-20250514",
            ProviderArg::OpenAI => "gpt-4o",
            ProviderArg::Gemini => "gemini-2.0-flash",
            ProviderArg::Vertex => "gemini-2.0-flash",
            ProviderArg::Azure => "gpt-4o",
            ProviderArg::Bedrock => "anthropic.claude-sonnet-4-v1:0",
            ProviderArg::Xai => "grok-2",
            ProviderArg::Groq => "llama-3.3-70b-versatile",
            ProviderArg::Mistral => "mistral-large-latest",
            ProviderArg::Deepseek => "deepseek-chat",
            ProviderArg::Cohere => "command-r-plus",
            ProviderArg::Perplexity => "sonar-pro",
            ProviderArg::Cerebras => "llama-3.3-70b",
            ProviderArg::Together => "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            ProviderArg::Fireworks => "accounts/fireworks/models/llama-v3p3-70b-instruct",
            ProviderArg::Qwen => "qwen-max",
            ProviderArg::Cloudflare => "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            ProviderArg::OpenRouter => "anthropic/claude-3.5-sonnet",
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
            ProviderArg::Anthropic => LlmProvider::Anthropic,
            ProviderArg::OpenAI => LlmProvider::OpenAI,
            ProviderArg::Gemini => LlmProvider::GoogleGemini,
            ProviderArg::Vertex => LlmProvider::GoogleVertex,
            ProviderArg::Azure => LlmProvider::AzureOpenAI,
            ProviderArg::Bedrock => LlmProvider::AWSBedrock,
            ProviderArg::Xai => LlmProvider::XAI,
            ProviderArg::Groq => LlmProvider::Groq,
            ProviderArg::Mistral => LlmProvider::Mistral,
            ProviderArg::Deepseek => LlmProvider::DeepSeek,
            ProviderArg::Cohere => LlmProvider::Cohere,
            ProviderArg::Perplexity => LlmProvider::Perplexity,
            ProviderArg::Cerebras => LlmProvider::Cerebras,
            ProviderArg::Together => LlmProvider::TogetherAI,
            ProviderArg::Fireworks => LlmProvider::FireworksAI,
            ProviderArg::Qwen => LlmProvider::AlibabaQwen,
            ProviderArg::Cloudflare => LlmProvider::CloudflareAI,
            ProviderArg::OpenRouter => LlmProvider::OpenRouter,
            ProviderArg::ClaudeCli => LlmProvider::Anthropic,
            ProviderArg::CodexCli => LlmProvider::OpenAI,
            ProviderArg::GeminiCli => LlmProvider::GoogleGemini,
            ProviderArg::OpencodeCli => LlmProvider::Opencode,
            ProviderArg::CopilotCli => LlmProvider::OpenAI,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[cfg(feature = "memory")]
#[derive(Subcommand)]
enum IndexAction {
    Build {
        #[arg(short, long)]
        force: bool,
    },
    Status,
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

async fn initialize_telemetry_if_enabled() -> anyhow::Result<()> {
    use reasonkit::telemetry::{TelemetryConfig, TelemetryStorage};
    let config = TelemetryConfig::from_env();
    if config.enabled {
        let db_path = if config.db_path == Path::new(".rk_telemetry.db") {
            TelemetryConfig::default_db_path()
        } else {
            config.db_path.clone()
        };
        match TelemetryStorage::new(&db_path).await {
            Ok(_) => {
                tracing::debug!(path = %db_path.display(), "Telemetry database initialized");
            }
            Err(e) => {
                tracing::warn!(error = %e, path = %db_path.display(), "Failed to initialize telemetry database");
            }
        }
    }
    Ok(())
}

fn unimplemented_command(name: &str) -> anyhow::Result<()> {
    anyhow::bail!(
        "{name} command is not implemented in this release. See the CLI reference for status."
    )
}

/// Parse budget string into BudgetConfig, with user-friendly error handling
fn parse_budget(budget_str: &str) -> anyhow::Result<BudgetConfig> {
    BudgetConfig::parse(budget_str).map_err(|e| {
        anyhow::anyhow!(
            "Invalid budget format: {}. Use formats like '30s', '5m', '1000t', or '$0.50'",
            e
        )
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    info!("ReasonKit Core v{}", env!("CARGO_PKG_VERSION"));

    if cli.init_telemetry {
        initialize_telemetry_if_enabled().await?;
    } else {
        tokio::spawn(async move {
            let _ = initialize_telemetry_if_enabled().await;
        });
    }

    match cli.command {
        Commands::Mcp(mcp_cli) => {
            run_mcp_command(mcp_cli).await?;
        }

        Commands::ServeMcp => {
            reasonkit::mcp::server::run_server().await?;
        }

        #[cfg(feature = "memory")]
        Commands::Ingest { .. } => {
            return unimplemented_command("ingest");
        }

        #[cfg(feature = "memory")]
        Commands::Query { .. } => {
            return unimplemented_command("query");
        }

        #[cfg(feature = "memory")]
        Commands::Index { .. } => {
            return unimplemented_command("index");
        }

        Commands::Stats => {
            return unimplemented_command("stats");
        }

        #[cfg(feature = "memory")]
        Commands::Export { .. } => {
            return unimplemented_command("export");
        }

        Commands::Serve { .. } => {
            return unimplemented_command("serve");
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
            let executor = if mock {
                ProtocolExecutor::mock()?
            } else {
                let mut config = ExecutorConfig::default();
                config.llm.provider = provider.into();
                if let Some(m) = model {
                    config.llm.model = m;
                }
                config.llm.temperature = temperature;
                config.llm.max_tokens = max_tokens;
                config.save_traces = save_trace;
                config.trace_dir = trace_dir;
                config.verbose = cli.verbose > 0;

                // Parse and apply budget configuration if provided
                if let Some(ref budget_str) = budget {
                    config.budget = parse_budget(budget_str)?;
                    if cli.verbose > 0 {
                        info!("Budget configured: {:?}", config.budget);
                    }
                }

                ProtocolExecutor::with_config(config)?
            };

            if list {
                println!("Available Protocols:");
                for p in executor.list_protocols() {
                    println!("  - {}", p);
                }
                println!("\nAvailable Profiles:");
                for p in executor.list_profiles() {
                    println!("  - {}", p);
                }
                return Ok(());
            }

            let q =
                query.ok_or_else(|| anyhow::anyhow!("Query is required unless --list is used"))?;
            let input = ProtocolInput::query(q);

            let output = if let Some(proto) = protocol {
                executor.execute(&proto, input).await?
            } else {
                let prof = profile.unwrap_or_else(|| "balanced".to_string());
                executor.execute_profile(&prof, input).await?
            };

            match format {
                OutputFormat::Text => {
                    println!("Thinking Process:");
                    for step in &output.steps {
                        println!("\n[{}] {}", step.step_id, step.as_text().unwrap_or(""));
                    }
                    println!("\nConfidence: {:.2}", output.confidence);

                    // Display budget summary if budget was configured
                    if let Some(ref summary) = output.budget_summary {
                        println!("\nBudget Summary:");
                        println!("  Steps completed: {}", summary.steps_completed);
                        println!("  Steps skipped: {}", summary.steps_skipped);
                        println!("  Tokens used: {}", summary.tokens_used);
                        println!("  Cost incurred: ${:.4}", summary.cost_incurred);
                        println!("  Time elapsed: {:?}", summary.elapsed);
                    }
                }
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
            }
        }

        Commands::Web { .. } => {
            return unimplemented_command("web");
        }

        Commands::Verify { .. } => {
            return unimplemented_command("verify");
        }

        Commands::Trace { .. } => {
            return unimplemented_command("trace");
        }

        #[cfg(feature = "memory")]
        Commands::Rag { .. } => {
            return unimplemented_command("rag");
        }

        Commands::Metrics { .. } => {
            return unimplemented_command("metrics");
        }

        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            generate(shell, &mut cmd, "rk", &mut std::io::stdout());
        }
    }

    Ok(())
}
