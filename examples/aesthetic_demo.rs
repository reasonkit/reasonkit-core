//! # Aesthetic Expression Mastery Example
//!
//! Demonstrates M2-enhanced UI/UX assessment capabilities.
//!
//! Run with: cargo run --example aesthetic_demo --features aesthetic

use reasonkit::aesthetic::{
    vibe_benchmark::{VIBE_ANDROID_BENCHMARK, VIBE_IOS_BENCHMARK, VIBE_WEB_BENCHMARK},
    AestheticConfig, AestheticExpressionService, ColorTokens, ComponentType, DesignData,
    DesignInput, DesignTokens, Platform, ThreeDDesignInput, ThreeDFramework,
    ThreeDPerformanceTargets, ThreeDSceneData, TypographyTokens,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ReasonKit Aesthetic Expression Mastery System               ║");
    println!("║  M2-Enhanced UI/UX Assessment                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Display M2 VIBE Benchmark Targets
    println!("📊 M2 VIBE Benchmark Targets:");
    println!("   • VIBE-Web:     {:.1}%", VIBE_WEB_BENCHMARK * 100.0);
    println!("   • VIBE-Android: {:.1}%", VIBE_ANDROID_BENCHMARK * 100.0);
    println!("   • VIBE-iOS:     {:.1}%", VIBE_IOS_BENCHMARK * 100.0);
    println!();

    // Create service with default config
    let config = AestheticConfig::default();
    let service = AestheticExpressionService::new(config).await?;

    // Example 1: Button Component Assessment
    println!("═══════════════════════════════════════════════════════════════");
    println!("🎨 Example 1: Button Component Assessment");
    println!("═══════════════════════════════════════════════════════════════");

    let button_input = DesignInput {
        data: DesignData::Html(
            r#"
            <button class="btn-primary">
                Get Started
            </button>
        "#
            .to_string(),
        ),
        platform: Platform::Web,
        context: Some("Primary CTA button for landing page".to_string()),
        component_type: ComponentType::Button,
        design_tokens: Some(DesignTokens {
            colors: ColorTokens::reasonkit_brand(),
            typography: TypographyTokens::reasonkit_brand(),
            ..Default::default()
        }),
    };

    let result = service.assess_design(button_input).await?;

    println!("\n📈 Assessment Results:");
    println!(
        "   Overall Score:        {:.1}%",
        result.overall_score * 100.0
    );
    println!(
        "   Visual Score:         {:.1}%",
        result.visual.score * 100.0
    );
    println!(
        "   Usability Score:      {:.1}%",
        result.usability.score * 100.0
    );
    println!(
        "   Accessibility Score:  {:.1}%",
        result.accessibility.score * 100.0
    );
    println!(
        "   Performance Score:    {:.1}%",
        result.performance.score * 100.0
    );
    println!(
        "   VIBE Compliance:      {:.1}%",
        result.vibe_compliance.overall_score * 100.0
    );
    println!(
        "   Passes VIBE Threshold: {}",
        if result.vibe_compliance.passes_threshold {
            "✅ YES"
        } else {
            "❌ NO"
        }
    );

    if !result.recommendations.is_empty() {
        println!("\n📋 Top Recommendations:");
        for (i, rec) in result.recommendations.iter().take(3).enumerate() {
            println!("   {}. [{:?}] {}", i + 1, rec.priority, rec.title);
        }
    }

    // Example 2: React Three Fiber 3D Assessment
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("🔮 Example 2: React Three Fiber 3D Assessment");
    println!("═══════════════════════════════════════════════════════════════");

    let r3f_input = ThreeDDesignInput {
        framework: ThreeDFramework::ReactThreeFiber,
        scene_data: ThreeDSceneData::R3FCode(
            r##"
            import { Canvas } from '@react-three/fiber';
            import { Suspense } from 'react';
            import { OrbitControls, Environment } from '@react-three/drei';

            function Scene() {
                return (
                    <Canvas>
                        <Suspense fallback={null}>
                            <OrbitControls />
                            <Environment preset="city" />
                            <mesh position={[0, 0, 0]}>
                                <boxGeometry args={[1, 1, 1]} />
                                <meshStandardMaterial color="#06b6d4" />
                            </mesh>
                            <mesh position={[2, 0, 0]}>
                                <sphereGeometry args={[0.5, 32, 32]} />
                                <meshStandardMaterial color="#a855f7" />
                            </mesh>
                        </Suspense>
                    </Canvas>
                );
            }
        "##
            .to_string(),
        ),
        performance_targets: ThreeDPerformanceTargets::default(),
        platform: Platform::Web,
    };

    let r3f_result = service.assess_3d_design(r3f_input).await?;

    println!("\n📈 3D Assessment Results:");
    println!("   Overall Score:       {:.1}%", r3f_result.score * 100.0);
    println!(
        "   Visual Quality:      {:.1}%",
        r3f_result.visual_quality * 100.0
    );
    println!(
        "   Performance Score:   {:.1}%",
        r3f_result.performance_score * 100.0
    );
    println!(
        "   Interaction Quality: {:.1}%",
        r3f_result.interaction_quality * 100.0
    );

    if let Some(count) = r3f_result.r3f_instance_count {
        println!("   R3F Instances:       {} (M2 capability: 7,000+)", count);
    }

    println!("   Estimated FPS:       {} fps", r3f_result.estimated_fps);
    println!("   Polygon Count:       {}", r3f_result.polygon_count);
    println!("   Draw Calls:          {}", r3f_result.draw_calls);

    if !r3f_result.optimizations.is_empty() {
        println!("\n⚡ Optimization Suggestions:");
        for opt in r3f_result.optimizations.iter().take(3) {
            println!("   • {}: {}", opt.category, opt.suggested_value);
        }
    }

    // Example 3: Cross-Platform Validation
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("📱 Example 3: Cross-Platform Validation");
    println!("═══════════════════════════════════════════════════════════════");

    let nav_input = DesignInput {
        data: DesignData::Html(
            r#"
            <nav class="navigation">
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/features">Features</a></li>
                    <li><a href="/pricing">Pricing</a></li>
                </ul>
            </nav>
        "#
            .to_string(),
        ),
        platform: Platform::Web,
        context: Some("Main navigation component".to_string()),
        component_type: ComponentType::Navigation,
        design_tokens: Some(DesignTokens {
            colors: ColorTokens::reasonkit_brand(),
            typography: TypographyTokens::reasonkit_brand(),
            ..Default::default()
        }),
    };

    let cross_result = service.validate_cross_platform(nav_input).await?;

    println!("\n📈 Cross-Platform Results:");
    println!("   Overall Score:      {:.1}%", cross_result.score * 100.0);
    println!(
        "   Consistency Score:  {:.1}%",
        cross_result.consistency_score * 100.0
    );

    println!("\n   Platform Compliance:");
    for (platform, result) in &cross_result.platform_results {
        println!(
            "   • {:?}: {:.1}%",
            platform,
            result.compliance_score * 100.0
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("✅ Aesthetic Expression Mastery Demo Complete!");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
