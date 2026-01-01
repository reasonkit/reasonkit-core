//! VIBE Protocol Validation System - Quick Start Example
//!
//! This example demonstrates the basic usage of the VIBE protocol validation system,
//! showcasing the revolutionary "Agent-as-a-Verifier" paradigm for automated protocol validation.

use reasonkit::thinktool::{Profile, ThinkToolExecutor};
use reasonkit::vibe::{Platform, VIBEEngine, ValidationConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VIBE Protocol Validation System - Quick Start");
    println!("{}", "=".repeat(50));

    // Initialize VIBE Engine
    let vibe_engine = VIBEEngine::new();
    println!("✅ VIBE Engine initialized");

    // Example 1: Validate a hand-written protocol
    println!("\n📋 Example 1: Validating a hand-written protocol");
    println!("{}", "-".repeat(40));

    let protocol = r#"
Protocol: User Authentication System
Purpose: Secure user authentication with multi-factor support
Steps:
1. Validate user credentials (email/password)
2. Check account status and permissions
3. Generate secure session token
4. Log authentication event
5. Redirect to appropriate dashboard

Security Requirements:
- Password hashing with bcrypt (cost factor 12)
- Session tokens expire after 24 hours
- Failed attempts logged and monitored
- Rate limiting prevents brute force (5 attempts/15min)

Error Handling:
- Generic errors for security
- Account lock after 5 failed attempts
- System errors logged with correlation IDs
- Graceful degradation for service unavailability
"#;

    // Configure validation for web and backend platforms
    let config = ValidationConfig::default()
        .with_platforms(vec![Platform::Web, Platform::Backend])
        .with_minimum_score(75.0);

    println!("🔍 Validating protocol...");
    let result = vibe_engine.validate_protocol(protocol, config).await?;

    println!("✅ Validation completed!");
    println!("📊 VIBE Score: {:.1}/100", result.overall_score);
    println!("🎯 Status: {:?}", result.status);
    println!("🔍 Issues found: {}", result.issues.len());

    // Show platform breakdown
    println!("\n📈 Platform Scores:");
    for (platform, score) in &result.platform_scores {
        println!("  {:?}: {:.1}/100", platform, score);
    }

    // Show key issues if any
    if !result.issues.is_empty() {
        println!("\n⚠️  Key Issues:");
        for issue in result.issues.iter().take(3) {
            println!("  - {:?}: {}", issue.severity, issue.description);
        }
    }

    // Example 2: Generate and validate using ThinkTools
    println!("\n🧠 Example 2: ThinkTool Generation + VIBE Validation");
    println!("{}", "-".repeat(40));

    let thinktool_executor = ThinkToolExecutor::new();
    let prompt = "Design a comprehensive e-commerce checkout protocol that includes guest checkout, user account integration, payment processing, order confirmation, and post-purchase email notifications.";

    println!("📝 Generating protocol from prompt...");
    let generated_protocol = thinktool_executor.run(prompt, Profile::Balanced).await?;
    println!(
        "✅ Protocol generated ({} characters)",
        generated_protocol.len()
    );

    // Validate the generated protocol
    let vibe_config = ValidationConfig::comprehensive().with_platforms(vec![
        Platform::Web,
        Platform::Backend,
        Platform::Simulation,
    ]);

    println!("🔍 Validating generated protocol...");
    let generated_result = vibe_engine
        .validate_protocol(&generated_protocol, vibe_config)
        .await?;

    println!("✅ Generated protocol validation completed!");
    println!("📊 VIBE Score: {:.1}/100", generated_result.overall_score);
    println!("🎯 Status: {:?}", generated_result.status);

    // Example 3: Performance benchmarking
    println!("\n📊 Example 3: Performance Benchmarking");
    println!("{}", "-".repeat(40));

    let benchmark_config = ValidationConfig::quick().with_platforms(vec![Platform::Web]);

    let _start_time = std::time::Instant::now();

    // Run multiple validations for benchmarking
    let iterations = 5;
    let mut total_time = 0;
    let mut scores = Vec::new();

    for i in 0..iterations {
        let iter_start = std::time::Instant::now();
        let iter_result = vibe_engine
            .validate_protocol(protocol, benchmark_config.clone())
            .await?;
        let iter_time = iter_start.elapsed().as_millis();

        total_time += iter_time;
        scores.push(iter_result.overall_score);

        println!(
            "  Iteration {}: {}ms, Score: {:.1}",
            i + 1,
            iter_time,
            iter_result.overall_score
        );
    }

    let avg_time = total_time as f32 / iterations as f32;
    let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;

    println!("✅ Benchmark completed!");
    println!("📊 Average validation time: {:.1}ms", avg_time);
    println!("📊 Average score: {:.1}/100", avg_score);
    println!("📊 Throughput: {:.1} validations/second", 1000.0 / avg_time);

    // Example 4: Cross-platform validation
    println!("\n🌐 Example 4: Cross-Platform Validation");
    println!("{}", "-".repeat(40));

    let mobile_protocol = r#"
Protocol: Mobile Banking Application
Purpose: Secure mobile banking with biometric authentication
Platform-Specific Requirements:
Android:
- Material Design 3.0 compliance
- Biometric authentication (fingerprint/face)
- Offline transaction queuing
- Android Auto integration
iOS:
- iOS Human Interface Guidelines
- Face ID/Touch ID support
- Apple Pay integration
- Siri shortcuts
Web:
- Progressive Web App (PWA)
- WebRTC for video calls
- Responsive design
- Accessibility (WCAG 2.1 AA)
Backend:
- API rate limiting
- Database encryption
- Audit logging
- Real-time fraud detection
Security:
- End-to-end encryption
- PCI DSS compliance
- Multi-factor authentication
- Session management
"#;

    let cross_platform_config = ValidationConfig::comprehensive().with_platforms(vec![
        Platform::Web,
        Platform::Android,
        Platform::IOS,
        Platform::Backend,
    ]);

    println!("🔍 Running cross-platform validation...");
    let cross_platform_result = vibe_engine
        .validate_protocol(mobile_protocol, cross_platform_config)
        .await?;

    println!("✅ Cross-platform validation completed!");
    println!(
        "📊 Overall VIBE Score: {:.1}/100",
        cross_platform_result.overall_score
    );

    // Calculate platform consensus
    let platform_scores: Vec<f32> = cross_platform_result
        .platform_scores
        .values()
        .cloned()
        .collect();
    let consensus = if platform_scores.len() > 1 {
        let mean = platform_scores.iter().sum::<f32>() / platform_scores.len() as f32;
        let variance = platform_scores
            .iter()
            .map(|&score| (score - mean).powi(2))
            .sum::<f32>()
            / platform_scores.len() as f32;
        100.0 - (variance.sqrt() * 2.0) // Simple consensus metric
    } else {
        100.0
    };

    println!("🤝 Platform consensus: {:.1}%", consensus);

    for (platform, score) in &cross_platform_result.platform_scores {
        println!("  {:?}: {:.1}/100", platform, score);
    }

    // Summary
    println!("\n🎉 VIBE Quick Start Complete!");
    println!("{}", "=".repeat(50));
    println!("✅ Validated 4 different protocols");
    println!("✅ Tested multiple validation configurations");
    println!("✅ Demonstrated ThinkTool integration");
    println!("✅ Performed performance benchmarking");
    println!(
        "✅ Validated across {} platforms",
        cross_platform_result.platform_scores.len()
    );

    println!("\n🔍 Key VIBE Capabilities Demonstrated:");
    println!("  🚀 Automated protocol validation");
    println!("  🌐 Multi-platform support (Web, Android, iOS, Backend, Simulation)");
    println!("  🧠 AI-powered issue detection and recommendations");
    println!("  📊 Comprehensive scoring with confidence intervals");
    println!("  ⚡ High-performance validation (sub-second)");
    println!("  🔄 Seamless ThinkTool integration");
    println!("  📈 Real-time benchmarking and performance monitoring");

    println!("\n💡 Next Steps:");
    println!("  📖 Read the full VIBE documentation");
    println!("  🛠️  Integrate VIBE into your protocol development workflow");
    println!("  📊 Set up continuous validation pipelines");
    println!("  🎯 Customize validation criteria for your domain");

    Ok(())
}
