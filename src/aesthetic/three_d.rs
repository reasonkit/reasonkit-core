//! # 3D Design Evaluation Module
//!
//! Assessment for 3D rendering with React Three Fiber, Three.js, and WebGL.
//! Leverages M2's proven 7,000+ R3F instance capability.

use super::types::*;

/// M2's proven React Three Fiber instance capability
pub const M2_R3F_INSTANCE_CAPABILITY: u64 = 7000;

/// 3D Design Evaluator
pub struct ThreeDEvaluator {
    performance_targets: ThreeDPerformanceTargets,
}

impl ThreeDEvaluator {
    /// Create a new 3D evaluator with performance targets
    pub fn new(targets: ThreeDPerformanceTargets) -> Self {
        Self {
            performance_targets: targets,
        }
    }

    /// Perform comprehensive 3D design assessment
    pub fn evaluate(&self, input: &ThreeDDesignInput) -> ThreeDAssessmentResult {
        let visual_quality = self.evaluate_visual_quality(input);
        let performance_score = self.evaluate_performance(input);
        let interaction_quality = self.evaluate_interaction(input);

        // Estimate metrics
        let polygon_count = self.estimate_polygon_count(input);
        let texture_memory_mb = self.estimate_texture_memory(input);
        let draw_calls = self.estimate_draw_calls(input);
        let estimated_fps = self.estimate_fps(polygon_count, draw_calls);

        // Get R3F instance count if applicable
        let r3f_instance_count = match input.framework {
            ThreeDFramework::ReactThreeFiber => Some(self.count_r3f_instances(input)),
            _ => None,
        };

        let issues = self.detect_issues(
            polygon_count,
            texture_memory_mb,
            draw_calls,
            estimated_fps,
            r3f_instance_count,
        );

        let optimizations = self.generate_optimizations(
            polygon_count,
            texture_memory_mb,
            draw_calls,
            estimated_fps,
        );

        let score = visual_quality * 0.35 + performance_score * 0.45 + interaction_quality * 0.20;

        ThreeDAssessmentResult {
            score,
            visual_quality,
            performance_score,
            interaction_quality,
            r3f_instance_count,
            polygon_count,
            texture_memory_mb,
            draw_calls,
            estimated_fps,
            issues,
            optimizations,
        }
    }

    /// Evaluate visual quality of 3D scene
    fn evaluate_visual_quality(&self, input: &ThreeDDesignInput) -> f64 {
        let mut score: f64 = 0.7;

        // Framework-specific bonuses
        match input.framework {
            ThreeDFramework::ReactThreeFiber => score += 0.15, // Modern, declarative
            ThreeDFramework::ThreeJs => score += 0.10,
            ThreeDFramework::BabylonJs => score += 0.10,
            ThreeDFramework::WebGPU => score += 0.20, // Cutting edge
            ThreeDFramework::WebGL => score += 0.05,
        }

        score.min(1.0)
    }

    /// Evaluate performance characteristics
    fn evaluate_performance(&self, input: &ThreeDDesignInput) -> f64 {
        let polygon_count = self.estimate_polygon_count(input);
        let draw_calls = self.estimate_draw_calls(input);
        let estimated_fps = self.estimate_fps(polygon_count, draw_calls);

        let mut score: f64 = 0.0;

        // FPS scoring
        if estimated_fps >= self.performance_targets.target_fps {
            score += 0.4;
        } else if estimated_fps >= self.performance_targets.target_fps / 2 {
            score += 0.2;
        }

        // Polygon budget
        if polygon_count <= self.performance_targets.max_polygons {
            score += 0.3;
        } else if polygon_count <= self.performance_targets.max_polygons * 2 {
            score += 0.15;
        }

        // Draw call efficiency
        if draw_calls <= self.performance_targets.max_draw_calls {
            score += 0.3;
        } else if draw_calls <= self.performance_targets.max_draw_calls * 2 {
            score += 0.15;
        }

        score.min(1.0)
    }

    /// Evaluate interaction quality
    fn evaluate_interaction(&self, input: &ThreeDDesignInput) -> f64 {
        match input.framework {
            ThreeDFramework::ReactThreeFiber => 0.90, // Great interaction model
            ThreeDFramework::BabylonJs => 0.88,
            ThreeDFramework::ThreeJs => 0.80,
            _ => 0.75,
        }
    }

    /// Estimate polygon count from scene data
    fn estimate_polygon_count(&self, input: &ThreeDDesignInput) -> u64 {
        match &input.scene_data {
            ThreeDSceneData::GltfData(data) => {
                // Rough estimation based on file size
                (data.len() as u64 / 50).max(1000)
            }
            ThreeDSceneData::R3FCode(code) => {
                // Estimate based on mesh/geometry mentions
                let mesh_count = code.matches("mesh").count() + code.matches("Mesh").count();
                (mesh_count as u64 * 1000).max(500)
            }
            ThreeDSceneData::ThreeJsCode(code) => {
                let geometry_count =
                    code.matches("Geometry").count() + code.matches("BufferGeometry").count();
                (geometry_count as u64 * 2000).max(500)
            }
            ThreeDSceneData::SceneGraph(json) => {
                // Count nodes in scene graph
                let node_str = json.to_string();
                let mesh_count = node_str.matches("\"type\":\"Mesh\"").count()
                    + node_str.matches("\"type\":\"mesh\"").count();
                (mesh_count as u64 * 1500).max(1000)
            }
        }
    }

    /// Estimate texture memory usage
    fn estimate_texture_memory(&self, input: &ThreeDDesignInput) -> f64 {
        match &input.scene_data {
            ThreeDSceneData::GltfData(data) => {
                // Rough: 1MB per 10KB of GLTF (textures are largest)
                (data.len() as f64 / 10_000.0).max(10.0)
            }
            ThreeDSceneData::R3FCode(code) => {
                // Count texture loads
                let texture_count = code.matches("useTexture").count()
                    + code.matches("textureLoader").count()
                    + code.matches("map=").count();
                (texture_count as f64 * 4.0).max(8.0) // 4MB per texture average
            }
            _ => 32.0, // Default estimate
        }
    }

    /// Estimate draw calls
    fn estimate_draw_calls(&self, input: &ThreeDDesignInput) -> u32 {
        match &input.scene_data {
            ThreeDSceneData::R3FCode(code) => {
                // Each mesh without instancing = 1 draw call
                let mesh_count = code.matches("<mesh").count() + code.matches("<Mesh").count();

                // Check for instancing (reduces draw calls)
                let has_instancing = code.contains("instancedMesh")
                    || code.contains("InstancedMesh")
                    || code.contains("instances");

                if has_instancing {
                    (mesh_count as u32 / 10).max(5)
                } else {
                    mesh_count as u32
                }
            }
            ThreeDSceneData::ThreeJsCode(code) => {
                let mesh_count = code.matches("new Mesh").count();
                mesh_count as u32
            }
            _ => 50, // Default estimate
        }
    }

    /// Estimate FPS based on complexity
    fn estimate_fps(&self, polygon_count: u64, draw_calls: u32) -> u32 {
        // Simplified estimation model
        let complexity_factor = (polygon_count as f64 / 500_000.0) + (draw_calls as f64 / 50.0);

        let base_fps = 60.0;
        let estimated = base_fps / (1.0 + complexity_factor * 0.5);

        estimated as u32
    }

    /// Count React Three Fiber instances
    fn count_r3f_instances(&self, input: &ThreeDDesignInput) -> u64 {
        match &input.scene_data {
            ThreeDSceneData::R3FCode(code) => {
                // Count various R3F components
                let instances = code.matches("<mesh").count()
                    + code.matches("<Mesh").count()
                    + code.matches("<group").count()
                    + code.matches("<Group").count()
                    + code.matches("<Instance").count()
                    + code.matches("<Points").count()
                    + code.matches("<Line").count();

                instances as u64
            }
            _ => 0,
        }
    }

    /// Detect 3D-specific issues
    fn detect_issues(
        &self,
        polygon_count: u64,
        texture_memory_mb: f64,
        draw_calls: u32,
        estimated_fps: u32,
        r3f_count: Option<u64>,
    ) -> Vec<ThreeDIssue> {
        let mut issues = Vec::new();

        // Polygon budget
        if polygon_count > self.performance_targets.max_polygons {
            issues.push(ThreeDIssue {
                severity: IssueSeverity::Major,
                category: ThreeDIssueCategory::Performance,
                description: format!(
                    "Polygon count ({}) exceeds budget ({})",
                    polygon_count, self.performance_targets.max_polygons
                ),
                suggestion: "Consider LOD, mesh simplification, or culling".to_string(),
            });
        }

        // Texture memory
        if texture_memory_mb > self.performance_targets.max_texture_memory_mb as f64 {
            issues.push(ThreeDIssue {
                severity: IssueSeverity::Major,
                category: ThreeDIssueCategory::Memory,
                description: format!(
                    "Texture memory ({:.0}MB) exceeds budget ({}MB)",
                    texture_memory_mb, self.performance_targets.max_texture_memory_mb
                ),
                suggestion: "Compress textures, reduce resolution, or use texture atlases"
                    .to_string(),
            });
        }

        // Draw calls
        if draw_calls > self.performance_targets.max_draw_calls {
            issues.push(ThreeDIssue {
                severity: IssueSeverity::Major,
                category: ThreeDIssueCategory::Performance,
                description: format!(
                    "Draw calls ({}) exceeds budget ({})",
                    draw_calls, self.performance_targets.max_draw_calls
                ),
                suggestion: "Use instancing, merge geometries, or implement batching".to_string(),
            });
        }

        // FPS
        if estimated_fps < self.performance_targets.target_fps {
            issues.push(ThreeDIssue {
                severity: if estimated_fps < 30 {
                    IssueSeverity::Critical
                } else {
                    IssueSeverity::Major
                },
                category: ThreeDIssueCategory::Performance,
                description: format!(
                    "Estimated FPS ({}) below target ({})",
                    estimated_fps, self.performance_targets.target_fps
                ),
                suggestion: "Optimize geometry, reduce effects, implement culling".to_string(),
            });
        }

        // R3F instance count (M2 capability check)
        if let Some(count) = r3f_count {
            if count > M2_R3F_INSTANCE_CAPABILITY {
                issues.push(ThreeDIssue {
                    severity: IssueSeverity::Minor,
                    category: ThreeDIssueCategory::Compatibility,
                    description: format!(
                        "R3F instance count ({}) exceeds M2 verified capability ({})",
                        count, M2_R3F_INSTANCE_CAPABILITY
                    ),
                    suggestion: "Consider chunking or progressive loading for very large scenes"
                        .to_string(),
                });
            }
        }

        issues
    }

    /// Generate optimization recommendations
    fn generate_optimizations(
        &self,
        polygon_count: u64,
        texture_memory_mb: f64,
        draw_calls: u32,
        estimated_fps: u32,
    ) -> Vec<ThreeDOptimization> {
        let mut opts = Vec::new();

        // LOD recommendation
        if polygon_count > 100_000 {
            opts.push(ThreeDOptimization {
                category: "Level of Detail".to_string(),
                current_value: format!("{} polygons", polygon_count),
                suggested_value: "Use 3 LOD levels".to_string(),
                expected_improvement: "50-70% polygon reduction at distance".to_string(),
                priority: OptimizationPriority::High,
            });
        }

        // Instancing
        if draw_calls > 50 {
            opts.push(ThreeDOptimization {
                category: "Instanced Rendering".to_string(),
                current_value: format!("{} draw calls", draw_calls),
                suggested_value: "InstancedMesh for repeated objects".to_string(),
                expected_improvement: "Up to 90% draw call reduction".to_string(),
                priority: OptimizationPriority::High,
            });
        }

        // Texture compression
        if texture_memory_mb > 128.0 {
            opts.push(ThreeDOptimization {
                category: "Texture Compression".to_string(),
                current_value: format!("{:.0}MB VRAM", texture_memory_mb),
                suggested_value: "Use KTX2/Basis compression".to_string(),
                expected_improvement: "3-5x memory reduction".to_string(),
                priority: OptimizationPriority::Medium,
            });
        }

        // Frustum culling
        if estimated_fps < 60 {
            opts.push(ThreeDOptimization {
                category: "Culling".to_string(),
                current_value: "Default culling".to_string(),
                suggested_value: "Enable frustum + occlusion culling".to_string(),
                expected_improvement: "20-40% performance improvement".to_string(),
                priority: OptimizationPriority::Medium,
            });
        }

        opts
    }
}

/// R3F-specific analyzer
pub struct ReactThreeFiberAnalyzer;

impl ReactThreeFiberAnalyzer {
    /// Analyze R3F code for best practices
    pub fn analyze(code: &str) -> R3FAnalysisResult {
        let uses_suspense = code.contains("Suspense");
        let uses_drei = code.contains("@react-three/drei");
        let uses_postprocessing = code.contains("@react-three/postprocessing");
        let uses_instancing = code.contains("InstancedMesh") || code.contains("Instance");
        let uses_leva = code.contains("leva");

        let best_practices_score =
            Self::calculate_best_practices_score(uses_suspense, uses_drei, uses_instancing);

        R3FAnalysisResult {
            uses_suspense,
            uses_drei,
            uses_postprocessing,
            uses_instancing,
            uses_leva,
            best_practices_score,
            suggestions: Self::generate_suggestions(uses_suspense, uses_drei, uses_instancing),
        }
    }

    fn calculate_best_practices_score(suspense: bool, drei: bool, instancing: bool) -> f64 {
        let mut score: f64 = 0.6;

        if suspense {
            score += 0.15;
        }
        if drei {
            score += 0.15;
        }
        if instancing {
            score += 0.10;
        }

        score.min(1.0)
    }

    fn generate_suggestions(suspense: bool, drei: bool, instancing: bool) -> Vec<String> {
        let mut suggestions = Vec::new();

        if !suspense {
            suggestions.push("Use <Suspense> for async loading of models".to_string());
        }
        if !drei {
            suggestions.push("Consider @react-three/drei for common helpers".to_string());
        }
        if !instancing {
            suggestions.push("Use instancing for repeated objects".to_string());
        }

        suggestions
    }
}

/// R3F analysis result
#[derive(Debug, Clone)]
pub struct R3FAnalysisResult {
    pub uses_suspense: bool,
    pub uses_drei: bool,
    pub uses_postprocessing: bool,
    pub uses_instancing: bool,
    pub uses_leva: bool,
    pub best_practices_score: f64,
    pub suggestions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3d_evaluation() {
        let evaluator = ThreeDEvaluator::new(ThreeDPerformanceTargets::default());

        let input = ThreeDDesignInput {
            framework: ThreeDFramework::ReactThreeFiber,
            scene_data: ThreeDSceneData::R3FCode(
                r#"
                <Canvas>
                    <mesh position={[0, 0, 0]}>
                        <boxGeometry />
                        <meshStandardMaterial />
                    </mesh>
                </Canvas>
            "#
                .to_string(),
            ),
            performance_targets: ThreeDPerformanceTargets::default(),
            platform: Platform::Web,
        };

        let result = evaluator.evaluate(&input);
        assert!(result.score > 0.0);
        assert!(result.r3f_instance_count.is_some());
    }

    #[test]
    fn test_r3f_analyzer() {
        let code = r#"
            import { Suspense } from 'react';
            import { Canvas } from '@react-three/fiber';
            import { OrbitControls } from '@react-three/drei';
        "#;

        let result = ReactThreeFiberAnalyzer::analyze(code);
        assert!(result.uses_suspense);
        assert!(result.uses_drei);
    }

    #[test]
    fn test_m2_capability() {
        assert_eq!(M2_R3F_INSTANCE_CAPABILITY, 7000);
    }
}
