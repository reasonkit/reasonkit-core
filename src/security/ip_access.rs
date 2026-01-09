//! Enterprise IP Access Control
//!
//! Implements IP allowlisting for enterprise customers with:
//! - CIDR range support
//! - Geo-IP blocking
//! - VPN/proxy detection
//! - Real-time configuration updates

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::security::audit::{AuditActor, AuditEvent, AuditEventType, AuditLogger, AuditOutcome};

/// Tenant IP configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TenantIpConfig {
    pub tenant_id: Uuid,
    /// Feature flag: is IP allowlisting enabled
    pub allowlist_enabled: bool,
    /// Allowed IP addresses and CIDR ranges (as strings for serialization)
    pub allowed_networks: Vec<String>,
    /// Allowed countries (ISO 3166-1 alpha-2)
    pub allowed_countries: Option<Vec<String>>,
    /// Denied countries (takes precedence over allowed)
    pub denied_countries: Vec<String>,
    /// Allow requests from known VPN/proxy IPs
    pub allow_vpn: bool,
    /// Allow requests from known datacenter IPs
    pub allow_datacenter: bool,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Updated by (admin user ID)
    pub updated_by: Uuid,
}

impl Default for TenantIpConfig {
    fn default() -> Self {
        Self {
            tenant_id: Uuid::nil(),
            allowlist_enabled: false,
            allowed_networks: Vec::new(),
            allowed_countries: None,
            denied_countries: Vec::new(),
            allow_vpn: true,
            allow_datacenter: true,
            updated_at: chrono::Utc::now(),
            updated_by: Uuid::nil(),
        }
    }
}

/// IP validation result
#[derive(Debug, Clone)]
pub enum IpValidationResult {
    /// IP is allowed
    Allowed {
        matched_network: Option<String>,
        country: Option<String>,
    },
    /// IP is denied
    Denied {
        reason: IpDenialReason,
        details: String,
    },
    /// Tenant has no IP restrictions
    NoRestrictions,
}

/// Reasons for IP denial
#[derive(Debug, Clone)]
pub enum IpDenialReason {
    /// IP not in allowlist
    NotInAllowlist,
    /// Country not allowed
    CountryBlocked,
    /// Known VPN/proxy when not allowed
    VpnDetected,
    /// Known threat actor IP
    ThreatIntelMatch,
    /// Datacenter IP when not allowed
    DatacenterIp,
}

/// GeoIP lookup result
#[derive(Debug, Clone, Default)]
pub struct GeoIpInfo {
    pub country_code: Option<String>,
    pub country_name: Option<String>,
    pub city: Option<String>,
    pub is_vpn: bool,
    pub is_proxy: bool,
    pub is_datacenter: bool,
    pub asn: Option<u32>,
    pub org: Option<String>,
}

/// GeoIP provider trait
#[async_trait::async_trait]
pub trait GeoIpProvider: Send + Sync {
    async fn lookup(&self, ip: IpAddr) -> GeoIpInfo;
}

/// Threat intelligence provider trait
#[async_trait::async_trait]
pub trait ThreatIntelIpProvider: Send + Sync {
    async fn check_ip(&self, ip: IpAddr) -> Option<ThreatIpInfo>;
}

/// Threat intelligence result
#[derive(Debug, Clone)]
pub struct ThreatIpInfo {
    pub category: String,
    pub risk_score: u8,
}

/// Configuration error
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Network range too large: {0}")]
    NetworkTooLarge(String),
    #[error("Invalid network format: {0}")]
    InvalidNetwork(String),
    #[error("Invalid country code: {0}")]
    InvalidCountry(String),
}

/// IP validation cache entry
struct CacheEntry {
    result: IpValidationResult,
    expires_at: std::time::Instant,
}

/// Enterprise IP Access Controller
pub struct IpAccessController {
    /// Per-tenant configurations
    configs: Arc<RwLock<HashMap<Uuid, TenantIpConfig>>>,
    /// GeoIP database handle
    geoip: Option<Arc<dyn GeoIpProvider>>,
    /// Threat intelligence feed
    threat_intel: Option<Arc<dyn ThreatIntelIpProvider>>,
    /// Audit logger
    audit: Arc<AuditLogger>,
    /// Cache for recently validated IPs (tenant_id:ip -> result)
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Cache TTL
    cache_ttl: std::time::Duration,
}

impl IpAccessController {
    /// Create a new IP access controller
    pub fn new(audit: Arc<AuditLogger>) -> Self {
        Self {
            configs: Arc::new(RwLock::new(HashMap::new())),
            geoip: None,
            threat_intel: None,
            audit,
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_ttl: std::time::Duration::from_secs(300), // 5 minutes
        }
    }

    /// Set GeoIP provider
    pub fn with_geoip(mut self, provider: Arc<dyn GeoIpProvider>) -> Self {
        self.geoip = Some(provider);
        self
    }

    /// Set threat intelligence provider
    pub fn with_threat_intel(mut self, provider: Arc<dyn ThreatIntelIpProvider>) -> Self {
        self.threat_intel = Some(provider);
        self
    }

    /// Validate an IP address for a tenant
    pub async fn validate(&self, tenant_id: Uuid, ip: IpAddr) -> IpValidationResult {
        // Check cache first
        if let Some(cached) = self.check_cache(tenant_id, ip).await {
            return cached;
        }

        let configs = self.configs.read().await;
        let config = match configs.get(&tenant_id) {
            Some(c) if c.allowlist_enabled => c.clone(),
            _ => return IpValidationResult::NoRestrictions,
        };
        drop(configs);

        // Check threat intelligence first (always)
        if let Some(ref threat_intel) = self.threat_intel {
            if let Some(threat) = threat_intel.check_ip(ip).await {
                let result = IpValidationResult::Denied {
                    reason: IpDenialReason::ThreatIntelMatch,
                    details: format!("IP flagged: {}", threat.category),
                };
                self.audit_denial(tenant_id, ip, &result).await;
                return result;
            }
        }

        // Get GeoIP info
        let geo_info = if let Some(ref geoip) = self.geoip {
            geoip.lookup(ip).await
        } else {
            GeoIpInfo::default()
        };

        // Check denied countries
        if let Some(country) = &geo_info.country_code {
            if config.denied_countries.contains(country) {
                let result = IpValidationResult::Denied {
                    reason: IpDenialReason::CountryBlocked,
                    details: format!("Country {} is blocked", country),
                };
                self.audit_denial(tenant_id, ip, &result).await;
                return result;
            }
        }

        // Check allowed countries (if specified)
        if let Some(allowed) = &config.allowed_countries {
            if let Some(country) = &geo_info.country_code {
                if !allowed.contains(country) {
                    let result = IpValidationResult::Denied {
                        reason: IpDenialReason::CountryBlocked,
                        details: format!("Country {} not in allowlist", country),
                    };
                    self.audit_denial(tenant_id, ip, &result).await;
                    return result;
                }
            }
        }

        // Check VPN/proxy
        if (geo_info.is_vpn || geo_info.is_proxy) && !config.allow_vpn {
            let result = IpValidationResult::Denied {
                reason: IpDenialReason::VpnDetected,
                details: "VPN/proxy connections not allowed".to_string(),
            };
            self.audit_denial(tenant_id, ip, &result).await;
            return result;
        }

        // Check datacenter
        if geo_info.is_datacenter && !config.allow_datacenter {
            let result = IpValidationResult::Denied {
                reason: IpDenialReason::DatacenterIp,
                details: "Datacenter IPs not allowed".to_string(),
            };
            self.audit_denial(tenant_id, ip, &result).await;
            return result;
        }

        // Check network allowlist
        for network_str in &config.allowed_networks {
            if self.ip_in_network(ip, network_str) {
                let result = IpValidationResult::Allowed {
                    matched_network: Some(network_str.clone()),
                    country: geo_info.country_code.clone(),
                };
                self.cache_result(tenant_id, ip, result.clone()).await;
                return result;
            }
        }

        // IP not in any allowed network
        let result = IpValidationResult::Denied {
            reason: IpDenialReason::NotInAllowlist,
            details: "IP not in any allowed network range".to_string(),
        };
        self.audit_denial(tenant_id, ip, &result).await;
        result
    }

    /// Update tenant IP configuration
    pub async fn update_config(
        &self,
        config: TenantIpConfig,
        admin_id: Uuid,
    ) -> Result<(), ConfigError> {
        // Validate network ranges
        for network in &config.allowed_networks {
            self.validate_network(network)?;
        }

        // Clear cache for this tenant
        self.invalidate_cache(config.tenant_id).await;

        // Update configuration
        let mut configs = self.configs.write().await;
        configs.insert(config.tenant_id, config.clone());
        drop(configs);

        // Audit log
        let _ = self
            .audit
            .log(
                AuditEvent::new(AuditEventType::IpConfigUpdated, AuditActor::Admin(admin_id))
                    .with_tenant(config.tenant_id)
                    .with_details(serde_json::json!({
                        "networks_count": config.allowed_networks.len(),
                        "countries_allowed": config.allowed_countries,
                        "countries_denied": config.denied_countries,
                        "allow_vpn": config.allow_vpn,
                        "allow_datacenter": config.allow_datacenter,
                    }))
                    .with_outcome(AuditOutcome::Success),
            )
            .await;

        Ok(())
    }

    /// Get tenant configuration
    pub async fn get_config(&self, tenant_id: Uuid) -> Option<TenantIpConfig> {
        let configs = self.configs.read().await;
        configs.get(&tenant_id).cloned()
    }

    fn validate_network(&self, network: &str) -> Result<(), ConfigError> {
        // Parse CIDR notation
        let parts: Vec<&str> = network.split('/').collect();
        if parts.len() != 2 {
            // Single IP is OK
            if parts.len() == 1 {
                // Validate IP format
                network
                    .parse::<IpAddr>()
                    .map_err(|_| ConfigError::InvalidNetwork(format!("Invalid IP: {}", network)))?;
                return Ok(());
            }
            return Err(ConfigError::InvalidNetwork(format!(
                "Invalid CIDR: {}",
                network
            )));
        }

        let prefix: u8 = parts[1]
            .parse()
            .map_err(|_| ConfigError::InvalidNetwork(format!("Invalid prefix: {}", network)))?;

        // Check if IP part is valid
        let ip: IpAddr = parts[0]
            .parse()
            .map_err(|_| ConfigError::InvalidNetwork(format!("Invalid IP: {}", network)))?;

        // Reject overly broad ranges
        match ip {
            IpAddr::V4(_) if prefix < 16 => {
                return Err(ConfigError::NetworkTooLarge(
                    "IPv4 networks must be /16 or smaller".to_string(),
                ));
            }
            IpAddr::V6(_) if prefix < 48 => {
                return Err(ConfigError::NetworkTooLarge(
                    "IPv6 networks must be /48 or smaller".to_string(),
                ));
            }
            _ => {}
        }

        Ok(())
    }

    fn ip_in_network(&self, ip: IpAddr, network: &str) -> bool {
        // Simple CIDR matching implementation
        let parts: Vec<&str> = network.split('/').collect();

        if parts.len() == 1 {
            // Single IP comparison
            return network.parse::<IpAddr>().map(|n| n == ip).unwrap_or(false);
        }

        if parts.len() != 2 {
            return false;
        }

        let network_ip: IpAddr = match parts[0].parse() {
            Ok(ip) => ip,
            Err(_) => return false,
        };

        let prefix: u8 = match parts[1].parse() {
            Ok(p) => p,
            Err(_) => return false,
        };

        match (ip, network_ip) {
            (IpAddr::V4(ip), IpAddr::V4(net)) => {
                let ip_bits = u32::from(ip);
                let net_bits = u32::from(net);
                let mask = if prefix == 0 {
                    0
                } else {
                    !0u32 << (32 - prefix)
                };
                (ip_bits & mask) == (net_bits & mask)
            }
            (IpAddr::V6(ip), IpAddr::V6(net)) => {
                let ip_bits = u128::from(ip);
                let net_bits = u128::from(net);
                let mask = if prefix == 0 {
                    0
                } else {
                    !0u128 << (128 - prefix)
                };
                (ip_bits & mask) == (net_bits & mask)
            }
            _ => false, // Mismatched IP versions
        }
    }

    async fn check_cache(&self, tenant_id: Uuid, ip: IpAddr) -> Option<IpValidationResult> {
        let key = format!("{}:{}", tenant_id, ip);
        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(&key) {
            if entry.expires_at > std::time::Instant::now() {
                return Some(entry.result.clone());
            }
        }
        None
    }

    async fn cache_result(&self, tenant_id: Uuid, ip: IpAddr, result: IpValidationResult) {
        let key = format!("{}:{}", tenant_id, ip);
        let mut cache = self.cache.write().await;
        cache.insert(
            key,
            CacheEntry {
                result,
                expires_at: std::time::Instant::now() + self.cache_ttl,
            },
        );
    }

    async fn invalidate_cache(&self, tenant_id: Uuid) {
        let mut cache = self.cache.write().await;
        let prefix = format!("{}:", tenant_id);
        cache.retain(|k, _| !k.starts_with(&prefix));
    }

    async fn audit_denial(&self, tenant_id: Uuid, ip: IpAddr, result: &IpValidationResult) {
        if let IpValidationResult::Denied { reason, details } = result {
            let _ = self
                .audit
                .log(
                    AuditEvent::new(AuditEventType::AccessDenied, AuditActor::Anonymous)
                        .with_tenant(tenant_id)
                        .with_details(serde_json::json!({
                            "reason": format!("{:?}", reason),
                            "details": details,
                        }))
                        .with_ip(ip)
                        .with_outcome(AuditOutcome::Failure {
                            error_code: "IP_DENIED".to_string(),
                            error_message: details.clone(),
                        }),
                )
                .await;
        }
    }
}

/// No-op GeoIP provider for testing
pub struct NoOpGeoIpProvider;

#[async_trait::async_trait]
impl GeoIpProvider for NoOpGeoIpProvider {
    async fn lookup(&self, _ip: IpAddr) -> GeoIpInfo {
        GeoIpInfo::default()
    }
}

/// No-op threat intel provider for testing
pub struct NoOpThreatIntelIpProvider;

#[async_trait::async_trait]
impl ThreatIntelIpProvider for NoOpThreatIntelIpProvider {
    async fn check_ip(&self, _ip: IpAddr) -> Option<ThreatIpInfo> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::audit::{AuditConfig, InMemoryAuditStore};

    fn create_test_controller() -> IpAccessController {
        let audit_store = Arc::new(InMemoryAuditStore::new());
        let audit = Arc::new(AuditLogger::new(audit_store, AuditConfig::default()));
        IpAccessController::new(audit)
    }

    #[tokio::test]
    async fn test_no_restrictions_when_disabled() {
        let controller = create_test_controller();
        let tenant_id = Uuid::new_v4();
        let ip: IpAddr = "192.168.1.1".parse().unwrap();

        let result = controller.validate(tenant_id, ip).await;
        assert!(matches!(result, IpValidationResult::NoRestrictions));
    }

    #[tokio::test]
    async fn test_allows_ip_in_allowlist() {
        let controller = create_test_controller();
        let tenant_id = Uuid::new_v4();
        let ip: IpAddr = "192.168.1.100".parse().unwrap();

        let config = TenantIpConfig {
            tenant_id,
            allowlist_enabled: true,
            allowed_networks: vec!["192.168.1.0/24".to_string()],
            ..Default::default()
        };

        controller
            .update_config(config, Uuid::new_v4())
            .await
            .unwrap();

        let result = controller.validate(tenant_id, ip).await;
        assert!(matches!(result, IpValidationResult::Allowed { .. }));
    }

    #[tokio::test]
    async fn test_denies_ip_not_in_allowlist() {
        let controller = create_test_controller();
        let tenant_id = Uuid::new_v4();
        let ip: IpAddr = "10.0.0.1".parse().unwrap();

        let config = TenantIpConfig {
            tenant_id,
            allowlist_enabled: true,
            allowed_networks: vec!["192.168.1.0/24".to_string()],
            ..Default::default()
        };

        controller
            .update_config(config, Uuid::new_v4())
            .await
            .unwrap();

        let result = controller.validate(tenant_id, ip).await;
        assert!(matches!(
            result,
            IpValidationResult::Denied {
                reason: IpDenialReason::NotInAllowlist,
                ..
            }
        ));
    }

    #[test]
    fn test_ip_in_network_v4() {
        let controller = create_test_controller();

        assert!(controller.ip_in_network("192.168.1.100".parse().unwrap(), "192.168.1.0/24"));
        assert!(controller.ip_in_network("192.168.1.1".parse().unwrap(), "192.168.1.0/24"));
        assert!(!controller.ip_in_network("192.168.2.1".parse().unwrap(), "192.168.1.0/24"));
        assert!(controller.ip_in_network("10.0.0.5".parse().unwrap(), "10.0.0.0/8"));
    }

    #[test]
    fn test_validate_network_rejects_large_ranges() {
        let controller = create_test_controller();

        // Should reject /8 for IPv4
        assert!(controller.validate_network("10.0.0.0/8").is_err());

        // Should accept /16
        assert!(controller.validate_network("192.168.0.0/16").is_ok());

        // Should accept single IP
        assert!(controller.validate_network("192.168.1.1").is_ok());
    }
}
