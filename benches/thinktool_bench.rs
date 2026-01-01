//! ThinkTools Protocol Execution Benchmark Suite
//!
//! Comprehensive performance benchmarks for the ThinkTools protocol engine.
//! Measures all critical execution paths with target: < 5ms for non-LLM operations.
//!
//! ## Benchmark Categories
//!
//! 1. **Protocol Execution Latency** - End-to-end protocol execution time
//! 2. **Prompt Parsing Throughput** - Template rendering and input processing
//! 3. **Confidence Scoring Overhead** - Extraction and calculation costs
//! 4. **Profile Switching Cost** - Registry lookup and chain construction
//! 5. **Concurrent Execution Scaling** - Parallel protocol execution efficiency
//!
//! ## Performance Targets
//!
//! | Operation                    | Target      | Rationale                     |
//! |------------------------------|-------------|-------------------------------|
//! | Protocol orchestration       | < 10ms      | Exclude LLM time              |
//! | Template rendering           | < 1ms       | Pure string operations        |
//! | Confidence extraction        | < 0.5ms     | Single regex match            |
//! | Profile lookup               | < 0.1ms     | HashMap access                |
//! | Registry initialization      | < 50ms      | Startup cost                  |
//! | Step overhead                | < 2ms/step  | Orchestration per step        |
//!
//! ## Usage
//!
//! ```bash
//! # Run all ThinkTools benchmarks
//! cargo bench --bench thinktool_bench
//!
//! # Run specific benchmark group
//! cargo bench --bench thinktool_bench -- "protocol_execution"
//!
//! # Generate HTML reports
//! cargo bench --bench thinktool_bench -- --save-baseline main
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

// Import from the library (lib.name = "reasonkit" in Cargo.toml)
use reasonkit::thinktool::{
    executor::{ExecutorConfig, ProtocolExecutor, ProtocolInput},
    profiles::{ProfileRegistry, ReasoningProfile},
    protocol::{Protocol, ProtocolStep, ReasoningStrategy, StepAction, StepOutputFormat},
    registry::ProtocolRegistry,
    step::{ListItem, StepOutput, StepResult, TokenUsage},
};

// Shared tokio runtime reused across all benchmarks to avoid per-benchmark
// startup/teardown overhead and syscalls.
static RT: Lazy<Runtime> = Lazy::new(|| Runtime::new().expect("initialize tokio runtime"));

// =============================================================================
// TEST DATA GENERATORS
// =============================================================================

/// Generate test queries of varying complexity for realistic benchmarking
fn test_queries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("tiny", "2+2?"),
        ("short", "What is the capital of France?"),
        (
            "medium",
            "Explain the trade-offs between microservices and monolith architectures for a startup.",
        ),
        (
            "long",
            "Analyze the economic implications of artificial intelligence on the labor market \
             over the next decade, considering automation, job displacement, new job creation, \
             skill requirements, geographic distribution of impact, policy responses, and \
             potential mitigation strategies.",
        ),
        (
            "complex",
            "Given a distributed system with eventual consistency requirements, compare the \
             trade-offs between CRDTs, operational transforms, and consensus protocols like \
             Raft and Paxos. Consider latency, partition tolerance, implementation complexity, \
             debugging difficulty, and real-world deployment patterns. Provide specific \
             recommendations for different use cases including collaborative editing, \
             financial transactions, and IoT sensor networks.",
        ),
    ]
}

/// Generate test prompt templates with varying placeholder counts
fn test_templates() -> Vec<(&'static str, &'static str, usize)> {
    vec![
        ("simple", "Question: {{query}}\nProvide a clear answer.", 1),
        (
            "context",
            "Question: {{query}}\n{{#if context}}Context: {{context}}{{/if}}\nAnalyze thoroughly.",
            2,
        ),
        (
            "multi_placeholder",
            r#"Query: {{query}}
Context: {{context}}
Constraints: {{constraints}}
Domain: {{domain}}
Previous Analysis: {{previous}}
Format: {{format}}
Provide comprehensive analysis considering all inputs."#,
            6,
        ),
        (
            "nested_step",
            r#"Based on the previous step output:
{{identify_dimensions}}

And the exploration results:
{{explore_perspectives}}

Synthesize into coherent themes for: {{query}}"#,
            3,
        ),
    ]
}

/// Generate mock LLM responses with varying confidence patterns
fn mock_responses() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "simple_confidence",
            "This is the answer.\n\nConfidence: 0.85",
        ),
        (
            "embedded_confidence",
            "Based on analysis (confidence: 0.92), the primary factors are...",
        ),
        (
            "multi_line",
            r#"1. First point
2. Second point
3. Third point

Analysis complete.
Confidence: 0.88"#,
        ),
        (
            "numbered_list",
            r#"1. Machine learning applications
2. Deep learning architectures
3. Natural language processing
4. Computer vision systems
5. Reinforcement learning

Confidence: 0.91"#,
        ),
        (
            "bullet_list",
            r#"- Key insight one
- Key insight two with more detail
- Key insight three
* Bonus point using asterisk

Confidence: 0.78"#,
        ),
        (
            "complex_structured",
            r#"**Theme 1: Technology**
AI and machine learning are transforming industries.

**Theme 2: Economics**
Labor market disruption is inevitable.

**Theme 3: Policy**
Regulatory frameworks are lagging.

Final assessment: The trends suggest significant transformation.

Confidence: 0.84"#,
        ),
        (
            "no_confidence",
            "This response has no explicit confidence score anywhere in the text.",
        ),
    ]
}

// =============================================================================
// BENCHMARK 1: PROTOCOL EXECUTION LATENCY
// =============================================================================

/// Benchmark end-to-end protocol execution with mock LLM
/// Measures: orchestration overhead, step sequencing, output aggregation
fn bench_protocol_execution(c: &mut Criterion) {
    let rt = &*RT;

    let config = ExecutorConfig {
        use_mock: true,
        show_progress: false, // Disable for cleaner benchmark output
        ..Default::default()
    };
    let executor = ProtocolExecutor::with_config(config).unwrap();

    let mut group = c.benchmark_group("protocol_execution");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(100);

    // Test all protocols with medium complexity query
    let protocols = vec![
        "gigathink",
        "laserlogic",
        "bedrock",
        "proofguard",
        "brutalhonesty",
    ];
    let query = "Analyze the trade-offs between microservices and monolith architectures.";

    for protocol_id in protocols {
        let input = ProtocolInput::query(query);

        group.bench_with_input(
            BenchmarkId::new("single_protocol", protocol_id),
            &input,
            |b, input| {
                b.to_async(rt).iter(|| async {
                    let result = executor.execute(protocol_id, input.clone()).await;
                    black_box(result)
                });
            },
        );
    }

    // Test with varying query complexity
    for (name, query) in test_queries() {
        let input = ProtocolInput::query(query);

        group.bench_with_input(
            BenchmarkId::new("gigathink_complexity", name),
            &input,
            |b, input| {
                b.to_async(rt).iter(|| async {
                    let result = executor.execute("gigathink", input.clone()).await;
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK 2: PROMPT PARSING THROUGHPUT
// =============================================================================

/// Benchmark template rendering performance
/// Measures: placeholder substitution, conditional handling, regex operations
fn bench_prompt_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("prompt_parsing");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(200);

    // Create test input with all possible fields
    let mut fields = HashMap::new();
    fields.insert(
        "query".to_string(),
        serde_json::Value::String("Test query".to_string()),
    );
    fields.insert(
        "context".to_string(),
        serde_json::Value::String("Additional context for analysis".to_string()),
    );
    fields.insert(
        "constraints".to_string(),
        serde_json::Value::String("Time constraint: 1 hour".to_string()),
    );
    fields.insert(
        "domain".to_string(),
        serde_json::Value::String("Technology".to_string()),
    );
    fields.insert(
        "previous".to_string(),
        serde_json::Value::String("Previous step output".to_string()),
    );
    fields.insert(
        "format".to_string(),
        serde_json::Value::String("JSON".to_string()),
    );

    let input = ProtocolInput { fields };

    // Create previous step outputs for nested template tests
    let mut previous_outputs: HashMap<String, StepOutput> = HashMap::new();
    previous_outputs.insert(
        "identify_dimensions".to_string(),
        StepOutput::List {
            items: vec![
                ListItem::new("Dimension 1: Technical feasibility"),
                ListItem::new("Dimension 2: Cost analysis"),
                ListItem::new("Dimension 3: Time constraints"),
            ],
        },
    );
    previous_outputs.insert(
        "explore_perspectives".to_string(),
        StepOutput::Text {
            content: "Perspectives have been explored from multiple angles...".to_string(),
        },
    );

    // Benchmark template rendering throughput
    for (name, template, placeholder_count) in test_templates() {
        group.throughput(Throughput::Elements(placeholder_count as u64));
        group.bench_with_input(
            BenchmarkId::new("render_template", name),
            &(template, &input, &previous_outputs),
            |b, (template, input, outputs)| {
                b.iter(|| {
                    // Inline template rendering logic (mirroring executor)
                    let mut result = (*template).to_string();

                    // Replace input placeholders
                    for (key, value) in &input.fields {
                        let placeholder = format!("{{{{{}}}}}", key);
                        let value_str = match value {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        result = result.replace(&placeholder, &value_str);
                    }

                    // Replace step output placeholders
                    for (key, output) in *outputs {
                        let placeholder = format!("{{{{{}}}}}", key);
                        let value_str = match output {
                            StepOutput::Text { content } => content.clone(),
                            StepOutput::List { items } => items
                                .iter()
                                .map(|i| i.content.clone())
                                .collect::<Vec<_>>()
                                .join("\n"),
                            other => serde_json::to_string(other).unwrap_or_default(),
                        };
                        result = result.replace(&placeholder, &value_str);
                    }

                    // Cleanup conditionals (simplified)
                    result = result
                        .lines()
                        .filter(|l| !l.contains("{{#if") && !l.contains("{{/if"))
                        .collect::<Vec<_>>()
                        .join("\n");

                    black_box(result)
                });
            },
        );
    }

    // Benchmark batch template rendering (simulating profile chain)
    let templates: Vec<&str> = test_templates().iter().map(|(_, t, _)| *t).collect();
    group.bench_function("batch_render_4_templates", |b| {
        b.iter(|| {
            let results: Vec<String> = templates
                .iter()
                .map(|template| {
                    let mut result = (*template).to_string();
                    for (key, value) in &input.fields {
                        let placeholder = format!("{{{{{}}}}}", key);
                        result = result.replace(&placeholder, value.as_str().unwrap_or(""));
                    }
                    result
                })
                .collect();
            black_box(results)
        });
    });

    group.finish();
}

// =============================================================================
// BENCHMARK 3: CONFIDENCE SCORING OVERHEAD
// =============================================================================

/// Benchmark confidence extraction and calculation
/// Measures: regex matching, float parsing, aggregation
fn bench_confidence_scoring(c: &mut Criterion) {
    use once_cell::sync::Lazy;
    use regex::Regex;

    static CONFIDENCE_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?i)confidence[:\s]+(\d+\.?\d*)").expect("Invalid regex"));

    let mut group = c.benchmark_group("confidence_scoring");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(500);

    // Benchmark confidence extraction from various response formats
    for (name, response) in mock_responses() {
        group.bench_with_input(
            BenchmarkId::new("extract_confidence", name),
            &response,
            |b, response| {
                b.iter(|| {
                    let confidence = if let Some(caps) = CONFIDENCE_RE.captures(response) {
                        caps.get(1)
                            .and_then(|m| m.as_str().parse::<f64>().ok())
                            .map(|v| v.min(1.0))
                    } else {
                        None
                    };
                    black_box(confidence)
                });
            },
        );
    }

    // Benchmark aggregation of multiple confidence scores
    let confidence_values: Vec<f64> = vec![0.85, 0.92, 0.78, 0.88, 0.91, 0.84, 0.79, 0.95, 0.82];

    for count in [3, 5, 10, 20] {
        let values: Vec<f64> = confidence_values
            .iter()
            .cycle()
            .take(count)
            .copied()
            .collect();

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("aggregate_confidence", count),
            &values,
            |b, values| {
                b.iter(|| {
                    let sum: f64 = values.iter().sum();
                    let avg = sum / values.len() as f64;
                    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                    black_box((avg, min, max))
                });
            },
        );
    }

    // Benchmark weighted confidence aggregation (as used in profiles)
    let weighted_scores: Vec<(f64, f64)> = vec![
        (0.85, 1.0), // weight 1.0
        (0.92, 1.5), // higher weight
        (0.78, 0.5), // lower weight
        (0.88, 1.0),
        (0.91, 2.0), // highest weight
    ];

    group.bench_function("weighted_confidence_5_steps", |b| {
        b.iter(|| {
            let total_weight: f64 = weighted_scores.iter().map(|(_, w)| w).sum();
            let weighted_sum: f64 = weighted_scores.iter().map(|(c, w)| c * w).sum();
            let weighted_avg = weighted_sum / total_weight;
            black_box(weighted_avg)
        });
    });

    group.finish();
}

// =============================================================================
// BENCHMARK 4: PROFILE SWITCHING COST
// =============================================================================

/// Benchmark profile lookup and chain construction
/// Measures: registry access, profile resolution, chain building
fn bench_profile_switching(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_switching");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(500);

    // Benchmark registry initialization
    group.bench_function("registry_init", |b| {
        b.iter(|| {
            let mut registry = ProtocolRegistry::new();
            registry.register_builtins().unwrap();
            black_box(registry)
        });
    });

    // Benchmark profile registry initialization
    group.bench_function("profile_registry_init", |b| {
        b.iter(|| {
            let registry = ProfileRegistry::with_builtins();
            black_box(registry)
        });
    });

    // Pre-initialize registries for lookup benchmarks
    let profile_registry = ProfileRegistry::with_builtins();
    let mut protocol_registry = ProtocolRegistry::new();
    protocol_registry.register_builtins().unwrap();

    // Benchmark profile lookup by ID
    let profile_ids = [
        "quick",
        "balanced",
        "deep",
        "paranoid",
        "decide",
        "scientific",
    ];
    for profile_id in profile_ids {
        group.bench_with_input(
            BenchmarkId::new("lookup_profile", profile_id),
            &profile_id,
            |b, id| {
                b.iter(|| {
                    let profile = profile_registry.get(id);
                    black_box(profile)
                });
            },
        );
    }

    // Benchmark protocol lookup by ID
    let protocol_ids = [
        "gigathink",
        "laserlogic",
        "bedrock",
        "proofguard",
        "brutalhonesty",
    ];
    for protocol_id in protocol_ids {
        group.bench_with_input(
            BenchmarkId::new("lookup_protocol", protocol_id),
            &protocol_id,
            |b, id| {
                b.iter(|| {
                    let protocol = protocol_registry.get(id);
                    black_box(protocol)
                });
            },
        );
    }

    // Benchmark chain resolution (looking up all protocols in a profile)
    for profile_id in ["quick", "balanced", "paranoid"] {
        let profile = profile_registry.get(profile_id).unwrap();
        let chain_length = profile.chain.len();

        group.throughput(Throughput::Elements(chain_length as u64));
        group.bench_with_input(
            BenchmarkId::new("resolve_chain", profile_id),
            &profile,
            |b, profile| {
                b.iter(|| {
                    let protocols: Vec<Option<&Protocol>> = profile
                        .chain
                        .iter()
                        .map(|step| protocol_registry.get(&step.protocol_id))
                        .collect();
                    black_box(protocols)
                });
            },
        );
    }

    // Benchmark profile list iteration
    group.bench_function("list_all_profiles", |b| {
        b.iter(|| {
            let ids = profile_registry.list_ids();
            black_box(ids)
        });
    });

    group.bench_function("list_all_protocols", |b| {
        b.iter(|| {
            let ids = protocol_registry.list_ids();
            black_box(ids)
        });
    });

    group.finish();
}

// =============================================================================
// BENCHMARK 5: CONCURRENT EXECUTION SCALING
// =============================================================================

/// Benchmark parallel protocol execution efficiency
/// Measures: thread scaling, lock contention, async overhead
fn bench_concurrent_execution(c: &mut Criterion) {
    let rt = &*RT;

    let config = ExecutorConfig {
        use_mock: true,
        show_progress: false,
        enable_parallel: false, // Test sequential first
        ..Default::default()
    };
    let executor = Arc::new(ProtocolExecutor::with_config(config).unwrap());

    let mut group = c.benchmark_group("concurrent_execution");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(50);

    let query = "Analyze startup architecture decisions.";

    // Benchmark sequential execution at different concurrency levels
    for concurrency in [1, 2, 4, 8, 16] {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("sequential_executor", concurrency),
            &concurrency,
            |b, &n| {
                b.to_async(rt).iter(|| async {
                    let executor = executor.clone();
                    let futures: Vec<_> = (0..n)
                        .map(|i| {
                            let exec = executor.clone();
                            let input = ProtocolInput::query(format!("{} (Query {})", query, i));
                            async move { exec.execute("gigathink", input).await }
                        })
                        .collect();

                    let results = futures::future::join_all(futures).await;
                    black_box(results)
                });
            },
        );
    }

    // Benchmark with parallel execution enabled
    let parallel_config = ExecutorConfig {
        use_mock: true,
        show_progress: false,
        enable_parallel: true,
        max_concurrent_steps: 4,
        ..Default::default()
    };
    let parallel_executor = Arc::new(ProtocolExecutor::with_config(parallel_config).unwrap());

    for concurrency in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel_executor", concurrency),
            &concurrency,
            |b, &n| {
                b.to_async(rt).iter(|| async {
                    let executor = parallel_executor.clone();
                    let futures: Vec<_> = (0..n)
                        .map(|i| {
                            let exec = executor.clone();
                            let input = ProtocolInput::query(format!("{} (Query {})", query, i));
                            async move { exec.execute("gigathink", input).await }
                        })
                        .collect();

                    let results = futures::future::join_all(futures).await;
                    black_box(results)
                });
            },
        );
    }

    // Benchmark mixed protocol execution (realistic workload)
    let protocols = vec!["gigathink", "laserlogic", "bedrock"];
    group.bench_function("mixed_protocols_3", |b| {
        b.to_async(rt).iter(|| async {
            let executor = executor.clone();
            let futures: Vec<_> = protocols
                .iter()
                .enumerate()
                .map(|(i, protocol)| {
                    let exec = executor.clone();
                    let input = ProtocolInput::query(format!("Query for {}", protocol));
                    let protocol = protocol.to_string();
                    async move { exec.execute(&protocol, input).await }
                })
                .collect();

            let results = futures::future::join_all(futures).await;
            black_box(results)
        });
    });

    group.finish();
}

// =============================================================================
// BENCHMARK 6: PROFILE CHAIN EXECUTION
// =============================================================================

/// Benchmark full profile chain execution
/// Measures: chain overhead, step hand-off, cumulative latency
fn bench_profile_chains(c: &mut Criterion) {
    let rt = &*RT;

    let config = ExecutorConfig {
        use_mock: true,
        show_progress: false,
        ..Default::default()
    };
    let executor = ProtocolExecutor::with_config(config).unwrap();

    let mut group = c.benchmark_group("profile_chains");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(50);

    let query = "Should startups use microservices or monolith architecture?";
    let input = ProtocolInput::query(query);

    // quick = GigaThink -> LaserLogic (2 protocols)
    group.bench_function("quick_profile_2_steps", |b| {
        b.to_async(rt).iter(|| async {
            let result = executor.execute_profile("quick", input.clone()).await;
            black_box(result)
        });
    });

    // balanced = GigaThink -> LaserLogic -> BedRock -> ProofGuard (4 protocols)
    group.bench_function("balanced_profile_4_steps", |b| {
        b.to_async(rt).iter(|| async {
            let result = executor.execute_profile("balanced", input.clone()).await;
            black_box(result)
        });
    });

    // deep = All 5 protocols (conditional)
    group.bench_function("deep_profile_5_steps", |b| {
        b.to_async(rt).iter(|| async {
            let result = executor.execute_profile("deep", input.clone()).await;
            black_box(result)
        });
    });

    // paranoid = All 5 + validation pass (6 protocols)
    group.bench_function("paranoid_profile_6_steps", |b| {
        b.to_async(rt).iter(|| async {
            let result = executor.execute_profile("paranoid", input.clone()).await;
            black_box(result)
        });
    });

    // powercombo = Maximum rigor
    group.bench_function("powercombo_profile_max", |b| {
        b.to_async(rt).iter(|| async {
            let result = executor.execute_profile("powercombo", input.clone()).await;
            black_box(result)
        });
    });

    group.finish();
}

// =============================================================================
// BENCHMARK 7: STEP OVERHEAD ANALYSIS
// =============================================================================

/// Benchmark per-step orchestration overhead
/// Measures: overhead independent of LLM call time
fn bench_step_overhead(c: &mut Criterion) {
    let rt = &*RT;

    let config = ExecutorConfig {
        use_mock: true,
        show_progress: false,
        ..Default::default()
    };
    let executor = ProtocolExecutor::with_config(config).unwrap();

    let mut group = c.benchmark_group("step_overhead");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    // Measure individual protocol execution to determine per-step overhead
    // Each protocol has 3 steps, so we can calculate overhead = total_time / 3
    let protocols_with_steps = vec![
        ("gigathink", 3),
        ("laserlogic", 3),
        ("bedrock", 3),
        ("proofguard", 3),
        ("brutalhonesty", 3),
    ];

    for (protocol, expected_steps) in protocols_with_steps {
        let input = ProtocolInput::query("Benchmark query for step overhead measurement");

        group.throughput(Throughput::Elements(expected_steps as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "protocol_steps",
                format!("{}_{}_steps", protocol, expected_steps),
            ),
            &input,
            |b, input| {
                b.to_async(rt).iter(|| async {
                    let result = executor.execute(protocol, input.clone()).await;
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK 8: LIST PARSING PERFORMANCE
// =============================================================================

/// Benchmark list item extraction from LLM responses
/// Measures: regex parsing, multi-format handling
fn bench_list_parsing(c: &mut Criterion) {
    use once_cell::sync::Lazy;
    use regex::Regex;

    static NUMBERED_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^\d+[\.\)]\s*(.+)$").expect("Invalid regex"));
    static BOLD_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^\*\*([^*]+)\*\*[:\s-]*(.*)$").expect("Invalid regex"));

    let mut group = c.benchmark_group("list_parsing");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(200);

    // Different response formats to parse
    let numbered_response = r#"1. First point about machine learning
2. Second point about neural networks
3. Third point about deep learning
4. Fourth point about computer vision
5. Fifth point about NLP
6. Sixth point about reinforcement learning
7. Seventh point about transformers
8. Eighth point about attention mechanisms
9. Ninth point about embeddings
10. Tenth point about fine-tuning

Confidence: 0.88"#;

    let bullet_response = r#"- Machine learning fundamentals
- Neural network architectures
- Deep learning frameworks
- Computer vision applications
* Natural language processing
* Reinforcement learning concepts
- Transformer models

Confidence: 0.85"#;

    let bold_response = r#"**Technical Feasibility**: The project is technically viable
**Cost Analysis**: Total cost estimated at $500K
**Time Constraints**: Delivery within 6 months
**Risk Assessment**: Medium risk due to dependencies
**Resource Requirements**: Team of 5 engineers needed

Confidence: 0.82"#;

    group.bench_function("parse_numbered_10_items", |b| {
        b.iter(|| {
            let items: Vec<String> = numbered_response
                .lines()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    NUMBERED_RE
                        .captures(trimmed)
                        .map(|caps| caps[1].to_string())
                })
                .collect();
            black_box(items)
        });
    });

    group.bench_function("parse_bullet_7_items", |b| {
        b.iter(|| {
            let items: Vec<String> = bullet_response
                .lines()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    trimmed
                        .strip_prefix('-')
                        .or(trimmed.strip_prefix('*'))
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                })
                .collect();
            black_box(items)
        });
    });

    group.bench_function("parse_bold_5_items", |b| {
        b.iter(|| {
            let items: Vec<(String, String)> = bold_response
                .lines()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    BOLD_RE
                        .captures(trimmed)
                        .map(|caps| (caps[1].trim().to_string(), caps[2].trim().to_string()))
                })
                .collect();
            black_box(items)
        });
    });

    // Combined parsing (realistic scenario)
    let mixed_response = r#"1. First numbered item
**Bold Header**: Description here
- Bullet point one
2. Second numbered item
* Asterisk bullet
**Another Bold**: More description
3. Third numbered item

Confidence: 0.79"#;

    group.bench_function("parse_mixed_format", |b| {
        b.iter(|| {
            let mut items: Vec<String> = Vec::new();

            for line in mixed_response.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.to_lowercase().starts_with("confidence") {
                    continue;
                }

                if let Some(caps) = NUMBERED_RE.captures(trimmed) {
                    items.push(caps[1].to_string());
                } else if let Some(text) = trimmed.strip_prefix('-').or(trimmed.strip_prefix('*')) {
                    items.push(text.trim().to_string());
                } else if let Some(caps) = BOLD_RE.captures(trimmed) {
                    items.push(format!("{}: {}", &caps[1], &caps[2]));
                }
            }

            black_box(items)
        });
    });

    group.finish();
}

// =============================================================================
// BENCHMARK 9: EXECUTOR INITIALIZATION
// =============================================================================

/// Benchmark executor startup costs
/// Measures: registry loading, client initialization
fn bench_executor_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("executor_init");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(50);

    // Mock executor (no network initialization)
    group.bench_function("mock_executor_init", |b| {
        b.iter(|| {
            let config = ExecutorConfig::mock();
            let executor = ProtocolExecutor::with_config(config).unwrap();
            black_box(executor)
        });
    });

    // Default executor (would initialize LLM client)
    // Note: This will fail if no API keys are set, so we use mock
    group.bench_function("default_config_executor_init", |b| {
        b.iter(|| {
            let config = ExecutorConfig {
                use_mock: true, // Still use mock to avoid API calls
                ..Default::default()
            };
            let executor = ProtocolExecutor::with_config(config).unwrap();
            black_box(executor)
        });
    });

    // Executor with parallel enabled
    group.bench_function("parallel_executor_init", |b| {
        b.iter(|| {
            let config = ExecutorConfig {
                use_mock: true,
                enable_parallel: true,
                max_concurrent_steps: 4,
                ..Default::default()
            };
            let executor = ProtocolExecutor::with_config(config).unwrap();
            black_box(executor)
        });
    });

    group.finish();
}

// =============================================================================
// BENCHMARK 10: TOKEN USAGE TRACKING
// =============================================================================

/// Benchmark token usage aggregation
/// Measures: addition overhead, cost calculation
fn bench_token_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_usage");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(500);

    let usage_samples: Vec<TokenUsage> = (0..20)
        .map(|i| {
            TokenUsage::new(
                100 + i * 50,                // input tokens
                50 + i * 25,                 // output tokens
                0.001 + (i as f64) * 0.0005, // cost
            )
        })
        .collect();

    // Single addition
    group.bench_function("add_single_usage", |b| {
        let other = TokenUsage::new(100, 50, 0.001);
        b.iter(|| {
            let mut base = TokenUsage::default();
            base.add(&other);
            black_box(base)
        });
    });

    // Aggregate multiple
    for count in [5, 10, 20] {
        let samples: Vec<&TokenUsage> = usage_samples.iter().take(count).collect();

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("aggregate_usage", count),
            &samples,
            |b, samples| {
                b.iter(|| {
                    let mut total = TokenUsage::default();
                    for usage in samples.iter() {
                        total.add(usage);
                    }
                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    name = protocol_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .with_plots();
    targets =
        bench_protocol_execution,
        bench_profile_chains,
        bench_step_overhead
);

criterion_group!(
    name = parsing_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2));
    targets =
        bench_prompt_parsing,
        bench_confidence_scoring,
        bench_list_parsing
);

criterion_group!(
    name = registry_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2));
    targets =
        bench_profile_switching,
        bench_executor_init,
        bench_token_usage
);

criterion_group!(
    name = concurrency_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .sample_size(30);
    targets = bench_concurrent_execution
);

criterion_main!(
    protocol_benches,
    parsing_benches,
    registry_benches,
    concurrency_benches
);
