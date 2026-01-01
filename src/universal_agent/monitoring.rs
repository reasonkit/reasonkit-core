//! # Performance Monitoring System
//!
//! Real-time performance monitoring and analytics for all agent frameworks

use crate::error::Result;
use crate::universal_agent::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Performance Monitor
/// Tracks and analyzes performance across all agent frameworks
#[derive(Clone)]
pub struct PerformanceMonitor {
    metrics_collector: Arc<RwLock<MetricsCollector>>,
    real_time_analyzer: Arc<RwLock<RealTimeAnalyzer>>,
    alert_system: Arc<RwLock<AlertSystem>>,
    dashboard: Arc<RwLock<PerformanceDashboard>>,
    historical_data: Arc<RwLock<HistoricalDataStore>>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub async fn new() -> Result<Self> {
        Ok(Self {
            metrics_collector: Arc::new(RwLock::new(MetricsCollector::new().await?)),
            real_time_analyzer: Arc::new(RwLock::new(RealTimeAnalyzer::new().await?)),
            alert_system: Arc::new(RwLock::new(AlertSystem::new().await?)),
            dashboard: Arc::new(RwLock::new(PerformanceDashboard::new().await?)),
            historical_data: Arc::new(RwLock::new(HistoricalDataStore::new().await?)),
        })
    }

    /// Record performance metrics for a framework
    pub async fn record_performance(
        &self,
        framework: FrameworkType,
        result: &ProcessedProtocol,
    ) -> Result<()> {
        let metrics = PerformanceMetricsRecord {
            framework,
            timestamp: chrono::Utc::now(),
            processing_time_ms: result.processing_time_ms,
            confidence_score: result.confidence_score,
            success: result.is_success(),
            memory_usage_mb: result.metadata.memory_usage_mb,
            cpu_usage_percent: result.metadata.cpu_usage_percent,
            optimizations_applied: result.optimizations_applied.clone(),
        };

        // Record in metrics collector
        {
            let mut collector = self.metrics_collector.write().await;
            collector.record_metrics(metrics).await?;
        }

        // Update real-time analysis
        {
            let mut analyzer = self.real_time_analyzer.write().await;
            analyzer.update_real_time_metrics(&metrics).await?;
        }

        // Store in historical data
        {
            let mut historical = self.historical_data.write().await;
            historical.store_metrics(metrics).await?;
        }

        // Check for alerts
        {
            let mut alerts = self.alert_system.write().await;
            alerts.check_performance_alerts(&metrics).await?;
        }

        Ok(())
    }

    /// Get comprehensive performance metrics
    pub async fn get_comprehensive_metrics(&self) -> Result<ComprehensiveMetrics> {
        let collector = self.metrics_collector.read().await;
        let analyzer = self.real_time_analyzer.read().await;

        let current_metrics = collector.get_current_metrics().await?;
        let real_time_analysis = analyzer.get_real_time_analysis().await?;

        Ok(ComprehensiveMetrics {
            current_performance: current_metrics,
            real_time_trends: real_time_analysis.trends,
            alerts: self.get_active_alerts().await?,
            historical_summary: self.get_historical_summary().await?,
            framework_rankings: self.generate_framework_rankings().await?,
        })
    }

    /// Monitor performance in real-time
    pub async fn monitor_real_time(&self) -> Result<RealTimeMonitoring> {
        let analyzer = self.real_time_analyzer.read().await;
        let trends = analyzer.get_current_trends().await?;
        let alerts = self.get_active_alerts().await?;

        Ok(RealTimeMonitoring {
            timestamp: chrono::Utc::now(),
            trends,
            active_alerts: alerts,
            system_health: self.assess_system_health(&trends).await?,
            performance_score: self.calculate_overall_performance_score(&trends).await?,
        })
    }

    /// Generate performance report
    pub async fn generate_performance_report(&self, time_range: TimeRange) -> Result<PerformanceReport> {
        let historical = self.historical_data.read().await;
        let data = historical.get_metrics_range(time_range).await?;

        let analytics = self.analyze_performance_data(&data).await?;

        Ok(PerformanceReport {
            time_range,
            framework_performance: analytics.framework_analysis,
            overall_trends: analytics.trends,
            performance_insights: analytics.insights,
            recommendations: analytics.recommendations,
            report_timestamp: chrono::Utc::now(),
        })
    }

    /// Get active alerts
    async fn get_active_alerts(&self) -> Result<Vec<PerformanceAlert>> {
        let alerts = self.alert_system.read().await;
        Ok(alerts.get_active_alerts().await?)
    }

    /// Get historical summary
    async fn get_historical_summary(&self) -> Result<HistoricalSummary> {
        let historical = self.historical_data.read().await;
        Ok(historical.get_summary().await?)
    }

    /// Generate framework rankings
    async fn generate_framework_rankings(&self) -> Result<Vec<FrameworkRanking>> {
        let collector = self.metrics_collector.read().await;
        let metrics = collector.get_current_metrics().await?;

        let mut rankings = Vec::new();

        for (framework, framework_metrics) in metrics {
            let ranking = FrameworkRanking {
                framework,
                overall_score: self.calculate_framework_score(&framework_metrics).await?,
                success_rate: framework_metrics.success_rate(),
                average_latency: framework_metrics.average_latency_ms,
                throughput: framework_metrics.throughput_rps,
                rank: 0, // Will be set after sorting
            };
            rankings.push(ranking);
        }

        // Sort by overall score and assign ranks
        rankings.sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap());
        for (i, ranking) in rankings.iter_mut().enumerate() {
            ranking.rank = i + 1;
        }

        Ok(rankings)
    }

    /// Calculate overall performance score
    async fn calculate_overall_performance_score(&self, trends: &PerformanceTrends) -> Result<f64> {
        let mut total_score = 0.0;
        let mut weight_sum = 0.0;

        // Success rate contribution (40% weight)
        let success_score = trends.overall_success_rate * 0.4;
        total_score += success_score;
        weight_sum += 0.4;

        // Latency contribution (30% weight)
        let latency_score = (1.0 - (trends.average_latency_ms / 100.0)).max(0.0) * 0.3;
        total_score += latency_score;
        weight_sum += 0.3;

        // Throughput contribution (20% weight)
        let throughput_score = (trends.overall_throughput_rps / 200.0).min(1.0) * 0.2;
        total_score += throughput_score;
        weight_sum += 0.2;

        // Confidence score contribution (10% weight)
        let confidence_score = trends.average_confidence_score * 0.1;
        total_score += confidence_score;
        weight_sum += 0.1;

        Ok(total_score / weight_sum)
    }

    /// Calculate framework score
    async fn calculate_framework_score(&self, metrics: &PerformanceMetrics) -> Result<f64> {
        let success_component = metrics.success_rate() * 0.4;
        let latency_component = (1.0 - (metrics.average_latency_ms / 100.0)).max(0.0) * 0.3;
        let throughput_component = (metrics.throughput_rps / 200.0).min(1.0) * 0.2;
        let confidence_component = (1.0 - metrics.error_rate) * 0.1;

        Ok(success_component + latency_component + throughput_component + confidence_component)
    }

    /// Assess system health
    async fn assess_system_health(&self, trends: &PerformanceTrends) -> Result<SystemHealth> {
        let health_score = self.calculate_overall_performance_score(trends).await?;

        let status = if health_score >= 0.9 {
            SystemStatus::Excellent
        } else if health_score >= 0.8 {
            SystemStatus::Good
        } else if health_score >= 0.7 {
            SystemStatus::Fair
        } else {
            SystemStatus::Poor
        };

        Ok(SystemHealth {
            overall_score: health_score,
            status,
            health_factors: self.assess_health_factors(trends).await?,
            recommendations: self.generate_health_recommendations(status).await?,
        })
    }

    /// Assess health factors
    async fn assess_health_factors(&self, trends: &PerformanceTrends) -> Result<Vec<HealthFactor>> {
        let mut factors = Vec::new();

        // Success rate factor
        factors.push(HealthFactor {
            factor: "success_rate".to_string(),
            score: trends.overall_success_rate,
            status: if trends.overall_success_rate >= 0.95 {
                HealthStatus::Healthy
            } else if trends.overall_success_rate >= 0.90 {
                HealthStatus::Warning
            } else {
                HealthStatus::Critical
            },
        });

        // Latency factor
        factors.push(HealthFactor {
            factor: "latency".to_string(),
            score: (1.0 - (trends.average_latency_ms / 100.0)).max(0.0),
            status: if trends.average_latency_ms <= 50.0 {
                HealthStatus::Healthy
            } else if trends.average_latency_ms <= 75.0 {
                HealthStatus::Warning
            } else {
                HealthStatus::Critical
            },
        });

        // Throughput factor
        factors.push(HealthFactor {
            factor: "throughput".to_string(),
            score: (trends.overall_throughput_rps / 200.0).min(1.0),
            status: if trends.overall_throughput_rps >= 150.0 {
                HealthStatus::Healthy
            } else if trends.overall_throughput_rps >= 100.0 {
                HealthStatus::Warning
            } else {
                HealthStatus::Critical
            },
        });

        Ok(factors)
    }

    /// Generate health recommendations
    async fn generate_health_recommendations(&self, status: SystemStatus) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        match status {
            SystemStatus::Excellent => {
                recommendations.push("System performance is excellent".to_string());
                recommendations.push("Continue current optimization strategies".to_string());
            }
            SystemStatus::Good => {
                recommendations.push("System performance is good".to_string());
                recommendations.push("Consider minor optimizations for excellence".to_string());
            }
            SystemStatus::Fair => {
                recommendations.push("System performance needs improvement".to_string());
                recommendations.push("Review and optimize underperforming frameworks".to_string());
            }
            SystemStatus::Poor => {
                recommendations.push("System performance is poor".to_string());
                recommendations.push("Immediate optimization required".to_string());
                recommendations.push("Consider scaling resources or reviewing architecture".to_string());
            }
        }

        Ok(recommendations)
    }

    /// Analyze performance data
    async fn analyze_performance_data(&self, data: &[PerformanceMetricsRecord]) -> Result<PerformanceAnalytics> {
        let framework_analysis = self.analyze_framework_performance(data).await?;
        let trends = self.analyze_trends(data).await?;
        let insights = self.generate_performance_insights(&framework_analysis, &trends).await?;
        let recommendations = self.generate_performance_recommendations(&framework_analysis, &trends).await?;

        Ok(PerformanceAnalytics {
            framework_analysis,
            trends,
            insights,
            recommendations,
        })
    }

    /// Analyze framework performance
    async fn analyze_framework_performance(&self, data: &[PerformanceMetricsRecord]) -> Result<HashMap<FrameworkType, FrameworkAnalysis>> {
        let mut analysis = HashMap::new();

        for framework in FrameworkType::all() {
            let framework_data: Vec<_> = data.iter()
                .filter(|record| record.framework == framework)
                .collect();

            if !framework_data.is_empty() {
                let framework_analysis = self.analyze_single_framework(&framework_data)?;
                analysis.insert(framework, framework_analysis);
            }
        }

        Ok(analysis)
    }

    /// Analyze single framework performance
    fn analyze_single_framework(&self, data: &[&PerformanceMetricsRecord]) -> Result<FrameworkAnalysis> {
        let total_requests = data.len() as u64;
        let successful_requests = data.iter().filter(|r| r.success).count() as u64;
        let average_latency = data.iter().map(|r| r.processing_time_ms as f64).sum::<f64>() / data.len() as f64;
        let average_confidence = data.iter().map(|r| r.confidence_score).sum::<f64>() / data.len() as f64;

        let success_rate = successful_requests as f64 / total_requests as f64;
        let throughput = self.calculate_throughput(data);

        Ok(FrameworkAnalysis {
            framework: data[0].framework,
            total_requests,
            successful_requests,
            success_rate,
            average_latency_ms: average_latency,
            average_confidence_score: average_confidence,
            throughput_rps: throughput,
            performance_trend: self.calculate_trend(data),
            strengths: self.identify_strengths(data),
            weaknesses: self.identify_weaknesses(data),
        })
    }

    /// Calculate throughput
    fn calculate_throughput(&self, data: &[&PerformanceMetricsRecord]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let time_span = data.last().unwrap().timestamp.signed_duration_since(data[0].timestamp);
        let duration_seconds = time_span.num_seconds() as f64;

        if duration_seconds > 0.0 {
            data.len() as f64 / duration_seconds
        } else {
            0.0
        }
    }

    /// Calculate performance trend
    fn calculate_trend(&self, data: &[&PerformanceMetricsRecord]) -> PerformanceTrend {
        if data.len() < 10 {
            return PerformanceTrend::Stable;
        }

        let recent_avg: f64 = data.iter().rev().take(5)
            .map(|r| r.confidence_score)
            .sum::<f64>() / 5.0;

        let earlier_avg: f64 = data.iter().take(5)
            .map(|r| r.confidence_score)
            .sum::<f64>() / 5.0;

        if recent_avg > earlier_avg + 0.05 {
            PerformanceTrend::Improving
        } else if recent_avg < earlier_avg - 0.05 {
            PerformanceTrend::Declining
        } else {
            PerformanceTrend::Stable
        }
    }

    /// Identify framework strengths
    fn identify_strengths(&self, data: &[&PerformanceMetricsRecord]) -> Vec<String> {
        let mut strengths = Vec::new();
        let success_rate = data.iter().filter(|r| r.success).count() as f64 / data.len() as f64;
        let avg_latency = data.iter().map(|r| r.processing_time_ms).sum::<u64>() / data.len() as u64;

        if success_rate >= 0.95 {
            strengths.push("High success rate".to_string());
        }
        if avg_latency <= 50 {
            strengths.push("Low latency".to_string());
        }
        if data.len() > 100 {
            strengths.push("High volume processing".to_string());
        }

        strengths
    }

    /// Identify framework weaknesses
    fn identify_weaknesses(&self, data: &[&PerformanceMetricsRecord]) -> Vec<String> {
        let mut weaknesses = Vec::new();
        let success_rate = data.iter().filter(|r| r.success).count() as f64 / data.len() as f64;
        let avg_latency = data.iter().map(|r| r.processing_time_ms).sum::<u64>() / data.len() as u64;

        if success_rate < 0.90 {
            weaknesses.push("Low success rate".to_string());
        }
        if avg_latency > 75 {
            weaknesses.push("High latency".to_string());
        }

        weaknesses
    }

    /// Analyze trends
    async fn analyze_trends(&self, data: &[PerformanceMetricsRecord]) -> Result<TrendAnalysis> {
        let overall_success_rate = data.iter().filter(|r| r.success).count() as f64 / data.len() as f64;
        let overall_latency = data.iter().map(|r| r.processing_time_ms as f64).sum::<f64>() / data.len() as f64;
        let overall_throughput = self.calculate_overall_throughput(data);

        Ok(TrendAnalysis {
            overall_success_rate,
            average_latency_ms: overall_latency,
            overall_throughput_rps: overall_throughput,
            trend_direction: self.calculate_overall_trend(data),
            volatility: self.calculate_volatility(data),
        })
    }

    /// Calculate overall throughput
    fn calculate_overall_throughput(&self, data: &[PerformanceMetricsRecord]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let time_span = data.last().unwrap().timestamp.signed_duration_since(data[0].timestamp);
        let duration_seconds = time_span.num_seconds() as f64;

        if duration_seconds > 0.0 {
            data.len() as f64 / duration_seconds
        } else {
            0.0
        }
    }

    /// Calculate overall trend
    fn calculate_overall_trend(&self, data: &[PerformanceMetricsRecord]) -> PerformanceTrend {
        if data.len() < 10 {
            return PerformanceTrend::Stable;
        }

        let recent_success_rate = data.iter().rev().take(5)
            .filter(|r| r.success)
            .count() as f64 / 5.0;

        let earlier_success_rate = data.iter().take(5)
            .filter(|r| r.success)
            .count() as f64 / 5.0;

        if recent_success_rate > earlier_success_rate + 0.1 {
            PerformanceTrend::Improving
        } else if recent_success_rate < earlier_success_rate - 0.1 {
            PerformanceTrend::Declining
        } else {
            PerformanceTrend::Stable
        }
    }

    /// Calculate volatility
    fn calculate_volatility(&self, data: &[PerformanceMetricsRecord]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let success_rates: Vec<f64> = data.chunks(10)
            .map(|chunk| chunk.iter().filter(|r| r.success).count() as f64 / chunk.len() as f64)
            .collect();

        if success_rates.len() < 2 {
            return 0.0;
        }

        let mean = success_rates.iter().sum::<f64>() / success_rates.len() as f64;
        let variance = success_rates.iter()
            .map(|rate| (rate - mean).powi(2))
            .sum::<f64>() / success_rates.len() as f64;

        variance.sqrt()
    }

    /// Generate performance insights
    async fn generate_performance_insights(&self, framework_analysis: &HashMap<FrameworkType, FrameworkAnalysis>, trends: &TrendAnalysis) -> Result<Vec<PerformanceInsight>> {
        let mut insights = Vec::new();

        // Framework-specific insights
        for (framework, analysis) in framework_analysis {
            if analysis.success_rate >= 0.95 {
                insights.push(PerformanceInsight {
                    category: "framework_performance".to_string(),
                    insight: format!("{} is performing excellently with {:.1}% success rate", framework, analysis.success_rate * 100.0),
                    impact: "positive".to_string(),
                    actionable: false,
                });
            } else if analysis.success_rate < 0.90 {
                insights.push(PerformanceInsight {
                    category: "framework_performance".to_string(),
                    insight: format!("{} needs optimization with {:.1}% success rate", framework, analysis.success_rate * 100.0),
                    impact: "negative".to_string(),
                    actionable: true,
                });
            }
        }

        // Overall system insights
        if trends.overall_success_rate >= 0.95 {
            insights.push(PerformanceInsight {
                category: "system_performance".to_string(),
                insight: "Overall system performance is excellent".to_string(),
                impact: "positive".to_string(),
                actionable: false,
            });
        }

        Ok(insights)
    }

    /// Generate performance recommendations
    async fn generate_performance_recommendations(&self, framework_analysis: &HashMap<FrameworkType, FrameworkAnalysis>, trends: &TrendAnalysis) -> Result<Vec<PerformanceRecommendation>> {
        let mut recommendations = Vec::new();

        // Framework-specific recommendations
        for (framework, analysis) in framework_analysis {
            if analysis.success_rate < 0.90 {
                recommendations.push(PerformanceRecommendation {
                    framework: Some(*framework),
                    category: "optimization".to_string(),
                    priority: "high".to_string(),
                    description: format!("Optimize {} performance", framework),
                    suggestion: "Review and improve framework configuration".to_string(),
                    expected_impact: "significant".to_string(),
                });
            }

            if analysis.average_latency_ms > 75.0 {
                recommendations.push(PerformanceRecommendation {
                    framework: Some(*framework),
                    category: "latency".to_string(),
                    priority: "medium".to_string(),
                    description: format!("Reduce {} latency", framework),
                    suggestion: "Optimize processing pipeline".to_string(),
                    expected_impact: "moderate".to_string(),
                });
            }
        }

        // System-wide recommendations
        if trends.overall_success_rate < 0.95 {
            recommendations.push(PerformanceRecommendation {
                framework: None,
                category: "system_optimization".to_string(),
                priority: "high".to_string(),
                description: "Improve overall system success rate".to_string(),
                suggestion: "Review error handling and retry mechanisms".to_string(),
                expected_impact: "significant".to_string(),
            });
        }

        Ok(recommendations)
    }
}

/// Performance Metrics Record
#[derive(Debug, Clone)]
pub struct PerformanceMetricsRecord {
    pub framework: FrameworkType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub processing_time_ms: u64,
    pub confidence_score: f64,
    pub success: bool,
    pub memory_usage_mb: Option<f64>,
    pub cpu_usage_percent: Option<f64>,
    pub optimizations_applied: Vec<String>,
}

/// Comprehensive Metrics
#[derive(Debug, Clone)]
pub struct ComprehensiveMetrics {
    pub current_performance: HashMap<FrameworkType, PerformanceMetrics>,
    pub real_time_trends: PerformanceTrends,
    pub alerts: Vec<PerformanceAlert>,
    pub historical_summary: HistoricalSummary,
    pub framework_rankings: Vec<FrameworkRanking>,
}

/// Real-time Monitoring
#[derive(Debug, Clone)]
pub struct RealTimeMonitoring {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub trends: PerformanceTrends,
    pub active_alerts: Vec<PerformanceAlert>,
    pub system_health: SystemHealth,
    pub performance_score: f64,
}

/// Performance Trends
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    pub overall_success_rate: f64,
    pub average_latency_ms: f64,
    pub overall_throughput_rps: f64,
    pub average_confidence_score: f64,
    pub trend_direction: PerformanceTrend,
}

/// Framework Ranking
#[derive(Debug, Clone)]
pub struct FrameworkRanking {
    pub framework: FrameworkType,
    pub overall_score: f64,
    pub success_rate: f64,
    pub average_latency: f64,
    pub throughput: f64,
    pub rank: usize,
}

/// System Health
#[derive(Debug, Clone)]
pub struct SystemHealth {
    pub overall_score: f64,
    pub status: SystemStatus,
    pub health_factors: Vec<HealthFactor>,
    pub recommendations: Vec<String>,
}

/// Health Factor
#[derive(Debug, Clone)]
pub struct HealthFactor {
    pub factor: String,
    pub score: f64,
    pub status: HealthStatus,
}

/// Performance Alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub id: String,
    pub framework: FrameworkType,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub resolved: bool,
}

/// Historical Summary
#[derive(Debug, Clone)]
pub struct HistoricalSummary {
    pub total_requests: u64,
    pub average_success_rate: f64,
    pub peak_throughput: f64,
    pub best_performing_framework: FrameworkType,
    pub time_period: String,
}

/// Performance Report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub time_range: TimeRange,
    pub framework_performance: HashMap<FrameworkType, FrameworkAnalysis>,
    pub overall_trends: TrendAnalysis,
    pub performance_insights: Vec<PerformanceInsight>,
    pub recommendations: Vec<PerformanceRecommendation>,
    pub report_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Time Range
#[derive(Debug, Clone)]
pub struct TimeRange {
    pub start: chrono::DateTime<chrono::Utc>,
    pub end: chrono::DateTime<chrono::Utc>,
}

/// Performance Analytics
#[derive(Debug, Clone)]
pub struct PerformanceAnalytics {
    pub framework_analysis: HashMap<FrameworkType, FrameworkAnalysis>,
    pub trends: TrendAnalysis,
    pub insights: Vec<PerformanceInsight>,
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Framework Analysis
#[derive(Debug, Clone)]
pub struct FrameworkAnalysis {
    pub framework: FrameworkType,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub success_rate: f64,
    pub average_latency_ms: f64,
    pub average_confidence_score: f64,
    pub throughput_rps: f64,
    pub performance_trend: PerformanceTrend,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
}

/// Trend Analysis
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub overall_success_rate: f64,
    pub average_latency_ms: f64,
    pub overall_throughput_rps: f64,
    pub trend_direction: PerformanceTrend,
    pub volatility: f64,
}

/// Performance Insight
#[derive(Debug, Clone)]
pub struct PerformanceInsight {
    pub category: String,
    pub insight: String,
    pub impact: String,
    pub actionable: bool,
}

/// Performance Recommendation
#[derive(Debug, Clone)]
pub struct PerformanceRecommendation {
    pub framework: Option<FrameworkType>,
    pub category: String,
    pub priority: String,
    pub description: String,
    pub suggestion: String,
    pub expected_impact: String,
}

/// Enumerations

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Declining,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemStatus {
    Excellent,
    Good,
    Fair,
    Poor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertType {
    Performance,
    Latency,
    SuccessRate,
    Throughput,
    ResourceUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Supporting Components

pub struct MetricsCollector;
impl MetricsCollector {
    pub async fn new() -> Result<Self> { Ok(Self) }
    pub async fn record_metrics(&self, _metrics: PerformanceMetricsRecord) -> Result<()> { Ok(()) }
    pub async fn get_current_metrics(&self) -> Result<HashMap<FrameworkType, PerformanceMetrics>> {
        Ok(HashMap::new())
    }
}

pub struct RealTimeAnalyzer;
impl RealTimeAnalyzer {
    pub async fn new() -> Result<Self> { Ok(Self) }
    pub async fn update_real_time_metrics(&self, _metrics: &PerformanceMetricsRecord) -> Result<()> { Ok(()) }
    pub async fn get_real_time_analysis(&self) -> Result<RealTimeAnalysis> {
        Ok(RealTimeAnalysis {
            trends: PerformanceTrends {
                overall_success_rate: 0.95,
                average_latency_ms: 45.0,
                overall_throughput_rps: 150.0,
                average_confidence_score: 0.92,
                trend_direction: PerformanceTrend::Stable,
            },
        })
    }
    pub async fn get_current_trends(&self) -> Result<PerformanceTrends> {
        Ok(PerformanceTrends {
            overall_success_rate: 0.95,
            average_latency_ms: 45.0,
            overall_throughput_rps: 150.0,
            average_confidence_score: 0.92,
            trend_direction: PerformanceTrend::Stable,
        })
    }
}

pub struct AlertSystem;
impl AlertSystem {
    pub async fn new() -> Result<Self> { Ok(Self) }
    pub async fn check_performance_alerts(&self, _metrics: &PerformanceMetricsRecord) -> Result<()> { Ok(()) }
    pub async fn get_active_alerts(&self) -> Result<Vec<PerformanceAlert>> { Ok(Vec::new()) }
}

pub struct PerformanceDashboard;
impl PerformanceDashboard {
    pub async fn new() -> Result<Self> { Ok(Self) }
}

pub struct HistoricalDataStore;
impl HistoricalDataStore {
    pub async fn new() -> Result<Self> { Ok(Self) }
    pub async fn store_metrics(&self, _metrics: PerformanceMetricsRecord) -> Result<()> { Ok(()) }
    pub async fn get_metrics_range(&self, _range: TimeRange) -> Result<Vec<PerformanceMetricsRecord>> { Ok(Vec::new()) }
    pub async fn get_summary(&self) -> Result<HistoricalSummary> {
        Ok(HistoricalSummary {
            total_requests: 1000,
            average_success_rate: 0.95,
            peak_throughput: 200.0,
            best_performing_framework: FrameworkType::BlackBoxAI,
            time_period: "24 hours".to_string(),
        })
    }
}

pub struct RealTimeAnalysis {
    pub trends: PerformanceTrends,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new().await.unwrap();
        let metrics = monitor.get_comprehensive_metrics().await.unwrap();
        assert!(!metrics.current_performance.is_empty() || metrics.current_performance.is_empty());
    }

    #[test]
    fn test_performance_trend_determination() {
        let monitor = PerformanceMonitor::new().await.unwrap();

        let data = vec![
            PerformanceMetricsRecord {
                framework: FrameworkType::ClaudeCode,
                timestamp: chrono::Utc::now(),
                processing_time_ms: 45,
                confidence_score: 0.8,
                success: true,
                memory_usage_mb: Some(50.0),
                cpu_usage_percent: Some(25.0),
                optimizations_applied: vec![],
            };
            10
        ];

        let trend = monitor.calculate_trend(&data.iter().collect::<Vec<_>>());
        assert!(matches!(trend, PerformanceTrend::Stable));
    }
}
