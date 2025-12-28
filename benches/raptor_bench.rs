//! Benchmarks for RAPTOR tree operations
//!
//! Performance target: < 5ms for tree traversal and search operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use reasonkit::raptor::{RaptorNode, RaptorTree};
use uuid::Uuid;

/// Create mock RAPTOR tree for benchmarking
fn create_mock_tree(num_leaves: usize, max_depth: usize, cluster_size: usize) -> RaptorTree {
    let mut tree = RaptorTree::new(max_depth, cluster_size);

    // Create leaf nodes
    for i in 0..num_leaves {
        let node = RaptorNode {
            id: Uuid::from_u128(i as u128),
            text: format!("Leaf node {} with some content for testing", i),
            children: Vec::new(),
            parent: None,
            level: 0,
            embedding: Some((0..384).map(|j| ((i + j) as f32 * 0.01).sin()).collect()),
        };
        tree.nodes.insert(node.id, node);
    }

    // Build hierarchical structure (simplified)
    let mut current_level: Vec<Uuid> = tree.nodes.keys().copied().collect();
    let mut level = 1;

    while current_level.len() > cluster_size && level <= max_depth {
        let mut next_level = Vec::new();

        for chunk in current_level.chunks(cluster_size) {
            let parent_id = Uuid::new_v4();
            let parent_text = format!("Summary of {} nodes at level {}", chunk.len(), level);

            let parent_node = RaptorNode {
                id: parent_id,
                text: parent_text,
                children: chunk.to_vec(),
                parent: None,
                level,
                embedding: Some((0..384).map(|j| (j as f32 * 0.01).sin()).collect()),
            };

            // Update children to point to parent
            for child_id in chunk {
                if let Some(child) = tree.nodes.get_mut(child_id) {
                    child.parent = Some(parent_id);
                }
            }

            tree.nodes.insert(parent_id, parent_node);
            next_level.push(parent_id);
        }

        current_level = next_level;
        level += 1;
    }

    tree.roots = current_level;
    tree
}

/// Benchmark tree creation
fn bench_tree_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_creation");

    let sizes = vec![10, 50, 100, 500];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.iter(|| {
                let tree = create_mock_tree(black_box(s), 3, 5);
                black_box(tree);
            });
        });
    }

    group.finish();
}

/// Benchmark tree traversal (depth-first)
fn bench_tree_traversal_dfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_traversal_dfs");

    let sizes = vec![50, 100, 500, 1000];

    for size in sizes {
        let tree = create_mock_tree(size, 3, 5);
        group.throughput(Throughput::Elements(tree.nodes.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut visited = Vec::new();
                let mut stack: Vec<Uuid> = tree.roots.clone();

                while let Some(node_id) = stack.pop() {
                    if let Some(node) = tree.nodes.get(&node_id) {
                        visited.push(node_id);
                        stack.extend(node.children.iter().rev());
                    }
                }

                black_box(visited);
            });
        });
    }

    group.finish();
}

/// Benchmark tree traversal (breadth-first)
fn bench_tree_traversal_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_traversal_bfs");

    let sizes = vec![50, 100, 500, 1000];

    for size in sizes {
        let tree = create_mock_tree(size, 3, 5);
        group.throughput(Throughput::Elements(tree.nodes.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut visited = Vec::new();
                let mut queue: std::collections::VecDeque<Uuid> =
                    tree.roots.iter().copied().collect();

                while let Some(node_id) = queue.pop_front() {
                    if let Some(node) = tree.nodes.get(&node_id) {
                        visited.push(node_id);
                        queue.extend(node.children.iter());
                    }
                }

                black_box(visited);
            });
        });
    }

    group.finish();
}

/// Benchmark node lookup by ID
fn bench_node_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_lookup");

    let tree = create_mock_tree(1000, 3, 5);
    let node_ids: Vec<Uuid> = tree.nodes.keys().take(100).copied().collect();

    group.bench_function("sequential_lookup", |b| {
        b.iter(|| {
            for id in &node_ids {
                let node = tree.nodes.get(black_box(id));
                black_box(node);
            }
        });
    });

    group.finish();
}

/// Benchmark finding nodes by level
fn bench_find_by_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_by_level");

    let tree = create_mock_tree(1000, 3, 5);

    for level in 0..=3 {
        group.bench_with_input(BenchmarkId::from_parameter(level), &level, |b, &l| {
            b.iter(|| {
                let nodes: Vec<&RaptorNode> =
                    tree.nodes.values().filter(|n| n.level == l).collect();
                black_box(nodes);
            });
        });
    }

    group.finish();
}

/// Benchmark parent-child navigation
fn bench_parent_child_navigation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parent_child_navigation");

    let tree = create_mock_tree(500, 3, 5);

    // Find a leaf node with parent
    let leaf_with_parent = tree
        .nodes
        .values()
        .find(|n| n.level == 0 && n.parent.is_some())
        .unwrap();

    group.bench_function("get_parent", |b| {
        b.iter(|| {
            if let Some(parent_id) = leaf_with_parent.parent {
                let parent = tree.nodes.get(black_box(&parent_id));
                black_box(parent);
            }
        });
    });

    // Find an internal node with children
    let node_with_children = tree
        .nodes
        .values()
        .find(|n| n.level > 0 && !n.children.is_empty())
        .unwrap();

    group.bench_function("get_children", |b| {
        b.iter(|| {
            let children: Vec<&RaptorNode> = node_with_children
                .children
                .iter()
                .filter_map(|id| tree.nodes.get(black_box(id)))
                .collect();
            black_box(children);
        });
    });

    group.finish();
}

/// Benchmark tree statistics calculation
fn bench_tree_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_statistics");

    let sizes = vec![100, 500, 1000];

    for size in sizes {
        let tree = create_mock_tree(size, 3, 5);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let total_nodes = tree.nodes.len();
                let leaf_nodes = tree.nodes.values().filter(|n| n.level == 0).count();
                let internal_nodes = total_nodes - leaf_nodes;
                let max_depth = tree.nodes.values().map(|n| n.level).max().unwrap_or(0);
                let avg_children: f32 = tree
                    .nodes
                    .values()
                    .filter(|n| !n.children.is_empty())
                    .map(|n| n.children.len() as f32)
                    .sum::<f32>()
                    / internal_nodes.max(1) as f32;

                black_box((
                    total_nodes,
                    leaf_nodes,
                    internal_nodes,
                    max_depth,
                    avg_children,
                ));
            });
        });
    }

    group.finish();
}

/// Benchmark similarity search in tree (mock)
fn bench_similarity_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_search");

    let tree = create_mock_tree(1000, 3, 5);
    let query_embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();

    group.bench_function("search_all_levels", |b| {
        b.iter(|| {
            // Calculate cosine similarity with all nodes
            let mut scores: Vec<(Uuid, f32)> = tree
                .nodes
                .iter()
                .filter_map(|(id, node)| {
                    node.embedding.as_ref().map(|emb| {
                        let dot: f32 = emb.iter().zip(&query_embedding).map(|(a, b)| a * b).sum();
                        let norm_a: f32 = emb.iter().map(|a| a * a).sum::<f32>().sqrt();
                        let norm_q: f32 = query_embedding.iter().map(|b| b * b).sum::<f32>().sqrt();
                        let similarity = dot / (norm_a * norm_q);
                        (*id, similarity)
                    })
                })
                .collect();

            // Sort by score
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Take top 10
            let top_10: Vec<(Uuid, f32)> = scores.into_iter().take(10).collect();
            black_box(top_10);
        });
    });

    group.bench_function("search_single_level", |b| {
        b.iter(|| {
            // Search only at level 0 (leaves)
            let mut scores: Vec<(Uuid, f32)> = tree
                .nodes
                .iter()
                .filter(|(_, node)| node.level == 0)
                .filter_map(|(id, node)| {
                    node.embedding.as_ref().map(|emb| {
                        let dot: f32 = emb.iter().zip(&query_embedding).map(|(a, b)| a * b).sum();
                        let norm_a: f32 = emb.iter().map(|a| a * a).sum::<f32>().sqrt();
                        let norm_q: f32 = query_embedding.iter().map(|b| b * b).sum::<f32>().sqrt();
                        let similarity = dot / (norm_a * norm_q);
                        (*id, similarity)
                    })
                })
                .collect();

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_10: Vec<(Uuid, f32)> = scores.into_iter().take(10).collect();
            black_box(top_10);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tree_creation,
    bench_tree_traversal_dfs,
    bench_tree_traversal_bfs,
    bench_node_lookup,
    bench_find_by_level,
    bench_parent_child_navigation,
    bench_tree_statistics,
    bench_similarity_search,
);

criterion_main!(benches);
