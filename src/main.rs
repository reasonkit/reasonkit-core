//! ReasonKit Core CLI
//!
//! Command-line interface for the ReasonKit knowledge base.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use reasonkit_core::thinktool::{
    ExecutorConfig, ProtocolExecutor, ProtocolInput,
    LlmConfig, LlmProvider, ExecutionTrace, ExecutionStatus, StepStatus,
};

/// ReasonKit Core - Knowledge Base for AI Reasoning
#[derive(Parser)]
#[command(name = "rk-core")]
#[command(author = "ReasonKit Team")]
#[command(version)]
#[command(about = "Rust-first knowledge base and RAG system for AI reasoning enhancement")]
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
    /// Ingest documents into the knowledge base
    Ingest {
        /// Path to document or directory
        path: PathBuf,

        /// Document type (paper, documentation, code, note)
        #[arg(short, long, default_value = "paper")]
        doc_type: String,

        /// Process recursively if directory
        #[arg(short, long)]
        recursive: bool,
    },

    /// Query the knowledge base
    Query {
        /// Query string
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,

        /// Use hybrid search
        #[arg(long)]
        hybrid: bool,

        /// Use RAPTOR tree retrieval
        #[arg(long)]
        raptor: bool,

        /// Output format (text, json, markdown)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Manage the index
    Index {
        #[command(subcommand)]
        action: IndexAction,
    },

    /// Show statistics about the knowledge base
    Stats,

    /// Export data
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

    /// Execute structured reasoning protocols (ThinkTool)
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

    /// View and manage execution traces
    Trace {
        #[command(subcommand)]
        action: TraceAction,
    },
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

/// LLM Provider argument
#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProviderArg {
    Anthropic,
    OpenAI,
    OpenRouter,
}

impl From<ProviderArg> for LlmProvider {
    fn from(arg: ProviderArg) -> Self {
        match arg {
            ProviderArg::Anthropic => LlmProvider::Anthropic,
            ProviderArg::OpenAI => LlmProvider::OpenAI,
            ProviderArg::OpenRouter => LlmProvider::OpenRouter,
        }
    }
}

/// Output format
#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

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

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    setup_logging(cli.verbose);

    info!("ReasonKit Core v{}", env!("CARGO_PKG_VERSION"));
    info!("Data directory: {:?}", cli.data_dir);

    match cli.command {
        Commands::Ingest { path, doc_type, recursive } => {
            info!("Ingesting documents from {:?}", path);
            info!("Document type: {}", doc_type);
            info!("Recursive: {}", recursive);

            // TODO: Implement ingestion
            println!("Ingestion not yet implemented");
            println!("Would ingest: {:?} as {}", path, doc_type);
        }

        Commands::Query { query, top_k, hybrid, raptor, format } => {
            info!("Querying: {}", query);
            info!("Top-k: {}, Hybrid: {}, RAPTOR: {}", top_k, hybrid, raptor);

            // TODO: Implement query
            println!("Query not yet implemented");
            println!("Would search for: \"{}\" (top {})", query, top_k);
            println!("Format: {}", format);
        }

        Commands::Index { action } => {
            match action {
                IndexAction::Build { force } => {
                    info!("Building index (force={})", force);
                    // TODO: Implement index build
                    println!("Index build not yet implemented");
                }
                IndexAction::Status => {
                    info!("Checking index status");
                    // TODO: Implement index status
                    println!("Index status not yet implemented");
                }
                IndexAction::Optimize => {
                    info!("Optimizing index");
                    // TODO: Implement index optimization
                    println!("Index optimization not yet implemented");
                }
            }
        }

        Commands::Stats => {
            info!("Showing statistics");

            // TODO: Implement stats
            println!("Statistics:");
            println!("  Documents: (not implemented)");
            println!("  Chunks: (not implemented)");
            println!("  Index size: (not implemented)");
        }

        Commands::Export { output, format } => {
            info!("Exporting to {:?} in {} format", output, format);

            // TODO: Implement export
            println!("Export not yet implemented");
        }

        Commands::Serve { host, port } => {
            info!("Starting API server on {}:{}", host, port);

            // TODO: Implement API server
            println!("API server not yet implemented");
            println!("Would serve on http://{}:{}", host, port);
        }

        Commands::Think {
            query,
            protocol,
            profile,
            provider,
            model,
            temperature,
            max_tokens,
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

            // Build LLM config
            let default_model = match provider {
                ProviderArg::Anthropic => "claude-sonnet-4-20250514",
                ProviderArg::OpenAI => "gpt-4o",
                ProviderArg::OpenRouter => "anthropic/claude-3.5-sonnet",
            };

            let llm_config = LlmConfig {
                provider: provider.into(),
                model: model.unwrap_or_else(|| default_model.to_string()),
                temperature,
                max_tokens,
                ..Default::default()
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
                ..Default::default()
            };

            let executor = ProtocolExecutor::with_config(config)
                .map_err(|e| anyhow::anyhow!("Failed to create executor: {}", e))?;

            // Execute protocol or profile
            let result = if let Some(profile_id) = profile {
                info!("Executing profile: {}", profile_id);
                let input = ProtocolInput::query(&query);
                executor.execute_profile(&profile_id, input).await
                    .map_err(|e| anyhow::anyhow!("Profile execution failed: {}", e))?
            } else {
                // Default to gigathink if no protocol specified
                let protocol_id = protocol.unwrap_or_else(|| "gigathink".to_string());
                info!("Executing protocol: {}", protocol_id);

                // Build appropriate input based on protocol
                let input = match protocol_id.as_str() {
                    "gigathink" => ProtocolInput::query(&query),
                    "laserlogic" => ProtocolInput::argument(&query),
                    "bedrock" => ProtocolInput::statement(&query),
                    "proofguard" => ProtocolInput::claim(&query),
                    "brutalhonesty" => ProtocolInput::work(&query),
                    _ => ProtocolInput::query(&query),
                };

                executor.execute(&protocol_id, input).await
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
                TraceAction::List { dir, protocol, limit } => {
                    let dir = dir.unwrap_or(trace_dir);
                    list_traces(&dir, protocol.as_deref(), limit)?;
                }

                TraceAction::View { id, dir, format } => {
                    let dir = dir.unwrap_or(trace_dir);
                    view_trace(&id, &dir, format)?;
                }

                TraceAction::Clean { dir, all, keep_days } => {
                    let dir = dir.unwrap_or(trace_dir);
                    clean_traces(&dir, all, keep_days)?;
                }
            }
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
            println!("  {:15} - {} ({:.0}% confidence)", id, p.description, p.min_confidence * 100.0);
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

    Ok(())
}

/// Print execution result in text format
fn print_think_result(result: &reasonkit_core::thinktool::ProtocolOutput) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                         ThinkTool Result");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("Protocol: {}", result.protocol_id);
    println!("Status:   {}", if result.success { "✓ Success" } else { "✗ Failed" });
    println!("Confidence: {:.1}%", result.confidence * 100.0);
    println!("Duration: {}ms", result.duration_ms);
    println!("Tokens:   {} input + {} output = {} total (${:.4})",
        result.tokens.input_tokens, result.tokens.output_tokens, result.tokens.total_tokens,
        result.tokens.cost_usd);
    println!();

    if !result.steps.is_empty() {
        println!("EXECUTION STEPS:");
        println!("──────────────────────────────────────────────────────────────────────");
        for step in &result.steps {
            let status = if step.success { "✓" } else { "✗" };
            println!("  {} {} (confidence: {:.1}%, {}ms)",
                status, step.step_id, step.confidence * 100.0, step.duration_ms);
        }
        println!();
    }

    println!("OUTPUT DATA:");
    println!("──────────────────────────────────────────────────────────────────────");
    for (key, value) in &result.data {
        if key == "confidence" {
            continue; // Skip redundant confidence
        }
        let display = format_json_value(value, 2);
        println!("  {}:", key);
        println!("{}", display);
    }

    if let Some(err) = &result.error {
        println!();
        println!("ERROR: {}", err);
    }

    println!();
}

/// Format JSON value for display
fn format_json_value(value: &serde_json::Value, indent: usize) -> String {
    let prefix = " ".repeat(indent);
    match value {
        serde_json::Value::String(s) => format!("{}\"{}\"", prefix, s),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter()
                .map(|v| format_json_value(v, indent + 2))
                .collect();
            if items.is_empty() {
                format!("{}[]", prefix)
            } else {
                format!("{}[\n{}\n{}]", prefix, items.join(",\n"), prefix)
            }
        }
        serde_json::Value::Object(obj) => {
            let items: Vec<String> = obj.iter()
                .map(|(k, v)| format!("{}{}: {}", " ".repeat(indent + 2), k, format_json_value(v, 0).trim()))
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
        println!("No traces found. Directory does not exist: {}", dir.display());
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
    traces.sort_by(|a, b| {
        b.1.timing.started_at.cmp(&a.1.timing.started_at)
    });

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
    println!("{:36}  {:15}  {:10}  {:8}  {}", "ID", "Protocol", "Status", "Conf.", "Duration");
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

        println!("{:36}  {:15}  {:10}  {:6.1}%  {}ms",
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
        return Err(anyhow::anyhow!("Trace directory does not exist: {}", dir.display()));
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
    println!("Protocol:    {} v{}", trace.protocol_id, trace.protocol_version);
    println!("Status:      {:?}", trace.status);
    println!("Confidence:  {:.1}%", trace.confidence * 100.0);
    println!();

    println!("TIMING:");
    println!("──────────────────────────────────────────────────────────────────────");
    if let Some(started) = trace.timing.started_at {
        println!("  Started:    {}", started.format("%Y-%m-%d %H:%M:%S UTC"));
    }
    if let Some(completed) = trace.timing.completed_at {
        println!("  Completed:  {}", completed.format("%Y-%m-%d %H:%M:%S UTC"));
    }
    println!("  Duration:   {}ms total, {}ms in LLM",
        trace.timing.total_duration_ms, trace.timing.llm_duration_ms);
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
    println!("{}", serde_json::to_string_pretty(&trace.input).unwrap_or_default());
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

        println!("  {}. {} {} (confidence: {:.1}%, {}ms)",
            i + 1, status_icon, step.step_id, step.confidence * 100.0, step.duration_ms);

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
        println!("{}", serde_json::to_string_pretty(output).unwrap_or_default());
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
        return Err(anyhow::anyhow!("Specify --all or --keep-days to clean traces"));
    }

    let cutoff = keep_days.map(|days| {
        chrono::Utc::now() - chrono::Duration::days(days as i64)
    });

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
