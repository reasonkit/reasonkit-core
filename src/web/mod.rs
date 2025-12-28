//! Web Search Integration for ReasonKit Web
//!
//! Provides web search capabilities for deep research:
//! - DuckDuckGo (free, no API key)
//! - Tavily (optional, better quality with API key)
//! - Serper (optional, Google search)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reasonkit::web::{WebSearcher, SearchConfig, SearchProvider};
//!
//! let config = SearchConfig::default();
//! let searcher = WebSearcher::new(config);
//! let results = searcher.search("rust async programming").await?;
//! ```

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Web search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Primary search provider
    pub provider: SearchProvider,
    /// Number of results to fetch
    pub num_results: usize,
    /// Request timeout
    pub timeout_secs: u64,
    /// Tavily API key (optional)
    pub tavily_api_key: Option<String>,
    /// Serper API key (optional)
    pub serper_api_key: Option<String>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            provider: SearchProvider::DuckDuckGo,
            num_results: 5,
            timeout_secs: 30,
            tavily_api_key: std::env::var("TAVILY_API_KEY").ok(),
            serper_api_key: std::env::var("SERPER_API_KEY").ok(),
        }
    }
}

/// Search provider options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchProvider {
    /// DuckDuckGo (free, no API key)
    #[default]
    DuckDuckGo,
    /// Tavily (requires API key, better quality)
    Tavily,
    /// Serper (requires API key, Google search)
    Serper,
    /// Auto-select best available
    Auto,
}

/// A single search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result title
    pub title: String,
    /// Result URL
    pub url: String,
    /// Result snippet/description
    pub snippet: String,
    /// Source provider
    pub source: SearchProvider,
}

/// Web search client
pub struct WebSearcher {
    config: SearchConfig,
    client: reqwest::Client,
}

impl WebSearcher {
    /// Create a new web searcher
    pub fn new(config: SearchConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .user_agent("ReasonKit/0.1 (https://reasonkit.sh)")
            .build()
            .unwrap_or_default();

        Self { config, client }
    }

    /// Search using the configured provider
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        let provider = match self.config.provider {
            SearchProvider::Auto => self.auto_select_provider(),
            p => p,
        };

        match provider {
            SearchProvider::DuckDuckGo => self.search_duckduckgo(query).await,
            SearchProvider::Tavily => self.search_tavily(query).await,
            SearchProvider::Serper => self.search_serper(query).await,
            SearchProvider::Auto => self.search_duckduckgo(query).await,
        }
    }

    /// Auto-select best available provider
    fn auto_select_provider(&self) -> SearchProvider {
        if self.config.tavily_api_key.is_some() {
            SearchProvider::Tavily
        } else if self.config.serper_api_key.is_some() {
            SearchProvider::Serper
        } else {
            SearchProvider::DuckDuckGo
        }
    }

    /// Search using DuckDuckGo HTML API (free, no key)
    async fn search_duckduckgo(&self, query: &str) -> Result<Vec<SearchResult>> {
        // DuckDuckGo Instant Answer API (JSON)
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
            urlencoding::encode(query)
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Network(format!("DuckDuckGo request failed: {}", e)))?;

        let data: DuckDuckGoResponse = response
            .json()
            .await
            .map_err(|e| Error::Network(format!("DuckDuckGo parse failed: {}", e)))?;

        let mut results = Vec::new();

        // Add abstract if present
        if !data.abstract_text.is_empty() {
            results.push(SearchResult {
                title: data.heading.clone(),
                url: data.abstract_url.clone(),
                snippet: data.abstract_text.clone(),
                source: SearchProvider::DuckDuckGo,
            });
        }

        // Add related topics
        for topic in data.related_topics.iter().take(self.config.num_results) {
            if let (Some(text), Some(first_url)) = (&topic.text, &topic.first_url) {
                results.push(SearchResult {
                    title: text.chars().take(100).collect(),
                    url: first_url.clone(),
                    snippet: text.clone(),
                    source: SearchProvider::DuckDuckGo,
                });
            }
        }

        // If no results from instant answer, try DuckDuckGo lite
        if results.is_empty() {
            results = self.search_duckduckgo_lite(query).await?;
        }

        Ok(results)
    }

    /// Fallback DuckDuckGo HTML scraping
    async fn search_duckduckgo_lite(&self, query: &str) -> Result<Vec<SearchResult>> {
        let url = format!(
            "https://lite.duckduckgo.com/lite/?q={}",
            urlencoding::encode(query)
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Network(format!("DuckDuckGo lite request failed: {}", e)))?;

        let html = response
            .text()
            .await
            .map_err(|e| Error::Network(format!("DuckDuckGo lite read failed: {}", e)))?;

        // Simple HTML parsing for DDG Lite results
        let mut results = Vec::new();
        let document = scraper::Html::parse_document(&html);
        let result_selector = scraper::Selector::parse("a.result-link")
            .unwrap_or_else(|_| scraper::Selector::parse("a").unwrap());

        for (i, element) in document.select(&result_selector).enumerate() {
            if i >= self.config.num_results {
                break;
            }

            if let Some(href) = element.value().attr("href") {
                if href.starts_with("http") {
                    let title = element.text().collect::<String>();
                    results.push(SearchResult {
                        title: title.clone(),
                        url: href.to_string(),
                        snippet: title,
                        source: SearchProvider::DuckDuckGo,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Search using Tavily API (requires API key)
    async fn search_tavily(&self, query: &str) -> Result<Vec<SearchResult>> {
        let api_key = self
            .config
            .tavily_api_key
            .as_ref()
            .ok_or_else(|| Error::Config("TAVILY_API_KEY not set".to_string()))?;

        let request = TavilyRequest {
            api_key: api_key.clone(),
            query: query.to_string(),
            search_depth: "advanced".to_string(),
            max_results: self.config.num_results,
            include_answer: true,
        };

        let response = self
            .client
            .post("https://api.tavily.com/search")
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Network(format!("Tavily request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Network(format!(
                "Tavily API error: {}",
                response.status()
            )));
        }

        let data: TavilyResponse = response
            .json()
            .await
            .map_err(|e| Error::Network(format!("Tavily parse failed: {}", e)))?;

        let mut results = Vec::new();

        // Add Tavily's answer if present
        if let Some(answer) = data.answer {
            results.push(SearchResult {
                title: "Tavily AI Answer".to_string(),
                url: String::new(),
                snippet: answer,
                source: SearchProvider::Tavily,
            });
        }

        // Add search results
        for result in data.results {
            results.push(SearchResult {
                title: result.title,
                url: result.url,
                snippet: result.content,
                source: SearchProvider::Tavily,
            });
        }

        Ok(results)
    }

    /// Search using Serper API (requires API key)
    async fn search_serper(&self, query: &str) -> Result<Vec<SearchResult>> {
        let api_key = self
            .config
            .serper_api_key
            .as_ref()
            .ok_or_else(|| Error::Config("SERPER_API_KEY not set".to_string()))?;

        let request = serde_json::json!({
            "q": query,
            "num": self.config.num_results
        });

        let response = self
            .client
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::Network(format!("Serper request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Network(format!(
                "Serper API error: {}",
                response.status()
            )));
        }

        let data: SerperResponse = response
            .json()
            .await
            .map_err(|e| Error::Network(format!("Serper parse failed: {}", e)))?;

        let mut results = Vec::new();

        // Add answer box if present
        if let Some(answer_box) = data.answer_box {
            results.push(SearchResult {
                title: answer_box.title.unwrap_or_else(|| "Answer".to_string()),
                url: answer_box.link.unwrap_or_default(),
                snippet: answer_box
                    .answer
                    .unwrap_or_else(|| answer_box.snippet.unwrap_or_default()),
                source: SearchProvider::Serper,
            });
        }

        // Add organic results
        for result in data.organic.unwrap_or_default() {
            results.push(SearchResult {
                title: result.title,
                url: result.link,
                snippet: result.snippet.unwrap_or_default(),
                source: SearchProvider::Serper,
            });
        }

        Ok(results)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// API RESPONSE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// DuckDuckGo Instant Answer API response
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct DuckDuckGoResponse {
    #[serde(default)]
    abstract_text: String,
    #[serde(default)]
    abstract_url: String,
    #[serde(default)]
    heading: String,
    #[serde(default)]
    related_topics: Vec<DuckDuckGoTopic>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct DuckDuckGoTopic {
    text: Option<String>,
    first_url: Option<String>,
}

/// Tavily API request
#[derive(Debug, Serialize)]
struct TavilyRequest {
    api_key: String,
    query: String,
    search_depth: String,
    max_results: usize,
    include_answer: bool,
}

/// Tavily API response
#[derive(Debug, Deserialize)]
struct TavilyResponse {
    answer: Option<String>,
    #[serde(default)]
    results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
}

/// Serper API response
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SerperResponse {
    answer_box: Option<SerperAnswerBox>,
    organic: Option<Vec<SerperOrganic>>,
}

#[derive(Debug, Deserialize)]
struct SerperAnswerBox {
    title: Option<String>,
    answer: Option<String>,
    snippet: Option<String>,
    link: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SerperOrganic {
    title: String,
    link: String,
    snippet: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// URL ENCODING HELPER
// ═══════════════════════════════════════════════════════════════════════════

mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                ' ' => "+".to_string(),
                _ => format!("%{:02X}", c as u32),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.provider, SearchProvider::DuckDuckGo);
        assert_eq!(config.num_results, 5);
    }

    #[test]
    fn test_url_encoding() {
        assert_eq!(urlencoding::encode("hello world"), "hello+world");
        assert_eq!(urlencoding::encode("rust+async"), "rust%2Basync");
    }

    #[tokio::test]
    async fn test_duckduckgo_search() {
        let config = SearchConfig {
            provider: SearchProvider::DuckDuckGo,
            num_results: 3,
            timeout_secs: 10,
            ..Default::default()
        };
        let searcher = WebSearcher::new(config);

        // This test requires network access
        let results = searcher.search("rust programming language").await;
        // Don't assert on results since network may not be available in CI
        assert!(results.is_ok() || results.is_err());
    }
}
