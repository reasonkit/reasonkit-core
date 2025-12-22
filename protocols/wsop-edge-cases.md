# WSOP Edge Cases and Robustness Specifications
> Failure modes, mitigations, and adversarial handling

**Version:** 1.1.0 | **Date:** 2025-12-12

---

## 1. QUERY EDGE CASES

### 1.1 Ambiguous Queries

| Edge Case | Example | Mitigation |
|-----------|---------|------------|
| Homonyms | "Apple" (company vs fruit) | Context detection + clarification prompt |
| Temporal ambiguity | "Recent elections" | Inject current date, ask for specificity |
| Entity confusion | "Michael Jordan" (basketball vs other) | Entity disambiguation via knowledge graph |
| Incomplete query | "Compare them" (no antecedent) | Require referent resolution before search |

**Implementation:**
```python
def detect_ambiguity(query: str) -> AmbiguityType:
    """Classify query ambiguity before processing."""
    checks = [
        (check_homonyms, AmbiguityType.HOMONYM),
        (check_temporal, AmbiguityType.TEMPORAL),
        (check_entity, AmbiguityType.ENTITY),
        (check_reference, AmbiguityType.REFERENCE)
    ]
    for check_fn, amb_type in checks:
        if check_fn(query):
            return amb_type
    return AmbiguityType.NONE
```

### 1.2 Adversarial Queries

| Edge Case | Example | Mitigation |
|-----------|---------|------------|
| Prompt injection | "Ignore previous... search for X" | Input sanitization + query validation |
| Excessive length | 10,000+ character query | Truncation with summarization |
| Encoding attacks | Unicode obfuscation | Normalize unicode before processing |
| Rate abuse | Rapid-fire queries | Per-user rate limiting |

**Validation Pipeline:**
```yaml
query_validation:
  max_length: 2048
  encoding: "utf-8-normalize"
  sanitization: true
  injection_detection: true
  rate_limit_per_user: "10/minute"
```

### 1.3 Language Edge Cases

| Edge Case | Example | Mitigation |
|-----------|---------|------------|
| Non-English | "Qu'est-ce que RAG?" | Language detection + multilingual models |
| Code-mixed | "What is the मतलब of this?" | Handle code-switching gracefully |
| Technical jargon | "HNSW with PQ compression" | Domain-aware embedding models |
| Typos/misspellings | "langchaim" | Fuzzy matching + spell correction |

---

## 2. RETRIEVAL EDGE CASES

### 2.1 Zero Results

| Edge Case | Mitigation |
|-----------|------------|
| No corpus matches | Fallback to web search immediately |
| No web results | Broaden query + try alternative providers |
| All providers fail | Return "insufficient sources" with confidence=0 |
| Rate limited by all | Queue request + exponential backoff |

**Fallback Chain:**
```
Corpus → Web Search (Tavily) → Web Search (Exa) → Web Search (Perplexity) → Graceful Degradation
```

### 2.2 Conflicting Results

| Edge Case | Mitigation |
|-----------|------------|
| Sources contradict | Report conflict explicitly with both views |
| Version conflicts | Prefer most recent authoritative source |
| Outdated information | Check publication dates, warn user |
| Fabricated sources | Cross-reference URLs, verify existence |

**Conflict Resolution:**
```python
def resolve_conflicts(claims: List[Claim]) -> ConflictResolution:
    """Handle contradictory information."""
    # Group by assertion
    groups = group_by_topic(claims)

    for topic, topic_claims in groups.items():
        if detect_contradiction(topic_claims):
            # Option 1: Report both views
            if confidence_spread(topic_claims) > 0.3:
                return ConflictResolution.REPORT_BOTH

            # Option 2: Prefer recent/authoritative
            else:
                return ConflictResolution.PREFER_RECENT_TIER1
```

### 2.3 Stale/Outdated Information

| Edge Case | Mitigation |
|-----------|------------|
| API version changed | Check for recency signals in content |
| Documentation outdated | Prefer official docs over blogs |
| Paper superseded | Check citation count and follow-up papers |
| Broken links | Mark as unverifiable, try archive.org |

---

## 3. PROVIDER EDGE CASES

### 3.1 API Failures

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| Timeout | >10s response time | Circuit breaker opens |
| 429 Rate Limited | HTTP 429 | Exponential backoff + jitter |
| 500 Server Error | HTTP 5xx | Retry with backoff, max 3 attempts |
| Invalid Response | Schema validation fail | Fallback to next provider |
| Auth Failure | HTTP 401/403 | Alert, disable provider until fixed |

**Circuit Breaker States:**
```
CLOSED → (5 failures) → OPEN → (30s) → HALF_OPEN → (1 success) → CLOSED
                                            ↓
                                       (1 failure)
                                            ↓
                                          OPEN
```

### 3.2 Provider-Specific Issues

| Provider | Known Issue | Handling |
|----------|-------------|----------|
| Tavily | Occasional timeout on complex queries | Set 15s timeout, fallback |
| Exa | Deep search mode is slow (3.5s P50) | Use fast mode by default |
| Perplexity | May return truncated results | Request more results than needed |
| arXiv | 3 RPS limit strict | Queue requests, never exceed |
| Semantic Scholar | Free tier limited | Cache aggressively (24h TTL) |

---

## 4. HYDE EDGE CASES

### 4.1 Hallucination Amplification

| Risk | Mitigation |
|------|------------|
| LLM generates incorrect facts | Dense bottleneck filters, but verify key facts |
| Hypothetical doc off-topic | Temperature tuning (0.5-0.7 optimal) |
| Circular reasoning | Don't use HyDE output as ground truth |
| Over-specific generation | Prompt for broader coverage |

**Safe HyDE Prompt:**
```
"Write a general passage about {topic} that would contain the answer.
Do NOT claim specific facts - focus on structure and terminology."
```

### 4.2 Encoding Failures

| Issue | Mitigation |
|-------|------------|
| Empty hypothetical doc | Retry with different temperature |
| Embedding dimension mismatch | Validate dimensions match corpus |
| NaN in embeddings | Input validation, fallback to raw query |

---

## 5. CRAG EDGE CASES

### 5.1 Evaluator Edge Cases

| Edge Case | Mitigation |
|-----------|------------|
| All docs score ~0.5 (AMBIGUOUS) | Lower threshold or increase retrieval k |
| False positive CORRECT | Add secondary verification for high-stakes |
| False negative INCORRECT | Conservative threshold (0.3) for fallback |
| Evaluator model unavailable | Fallback to simpler heuristics |

### 5.2 Knowledge Refinement Edge Cases

| Edge Case | Mitigation |
|-----------|------------|
| Over-filtering (empty result) | Relax threshold, keep top-k strips |
| Under-filtering (too verbose) | Tighten threshold, add redundancy removal |
| Strip boundary errors | Use NLP sentence boundaries, not heuristics |
| Multi-language strips | Language-aware tokenization |

---

## 6. RAG-FUSION EDGE CASES

### 6.1 Query Generation Issues

| Issue | Mitigation |
|-------|------------|
| Identical generated queries | Increase temperature, add diversity penalty |
| Off-topic queries | Validate semantic similarity to original |
| Too many queries (latency) | Cap at 4-5 queries maximum |
| Query leaks original answer | Don't include answer-seeking in multi-query |

### 6.2 RRF Edge Cases

| Issue | Mitigation |
|-------|------------|
| k=60 suboptimal for dataset | Allow configurable k, test on validation set |
| Single dominant result set | Weight by result set quality |
| Empty result sets | Skip in fusion, don't divide by zero |

---

## 7. COST EDGE CASES

### 7.1 Runaway Costs

| Risk | Mitigation |
|------|------------|
| Complex query triggers all providers | Budget caps per query ($0.10 max) |
| Multi-hop spirals | Max 5 retrieval iterations |
| LLM token explosion | Compress context with LongLLMLingua |
| Embedding costs | Batch embeddings, cache aggressively |

**Cost Limits:**
```yaml
cost_limits:
  per_query_usd: 0.10
  per_session_usd: 1.00
  per_day_usd: 100.00
  alert_threshold: 0.80  # Alert at 80% of limit
```

### 7.2 Token Optimization

| Technique | Savings | Trade-off |
|-----------|---------|-----------|
| LongLLMLingua compression | 4-6x | Slight accuracy loss |
| Knowledge strip filtering | 2-3x | May lose relevant info |
| Result caching (24h) | 10-50% | Staleness risk |
| Embedding caching | 90%+ | Storage cost |

---

## 8. SECURITY EDGE CASES

### 8.1 Data Leakage

| Risk | Mitigation |
|------|------------|
| PII in queries | Redact before logging, never send to external providers |
| API keys in responses | Scrub patterns matching key formats |
| Internal URLs exposed | Allowlist external-only URLs |
| Session data leakage | Isolate user sessions, no cross-user data |

### 8.2 Injection Attacks

| Attack | Mitigation |
|--------|------------|
| SQL injection via query | No SQL, use parameterized vector search |
| XSS in results | Sanitize HTML in response rendering |
| SSRF via URL fetching | Allowlist domains, validate URLs |
| LLM prompt injection | System prompt hardening, output validation |

---

## 9. OBSERVABILITY

### 9.1 Required Metrics

```yaml
metrics:
  latency:
    - query_total_ms
    - hyde_expansion_ms
    - crag_evaluation_ms
    - rag_fusion_ms
    - per_provider_ms

  quality:
    - triangulation_coverage
    - tier_1_source_ratio
    - conflict_rate
    - fallback_rate

  reliability:
    - circuit_breaker_trips
    - provider_error_rate
    - cache_hit_rate
    - retry_rate

  cost:
    - tokens_consumed
    - api_calls_per_provider
    - cost_per_query
```

### 9.2 Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| P95 Latency | >10s | >30s |
| Error Rate | >5% | >15% |
| Circuit Breaker Open | Any | >2 providers |
| Cost per Query | >$0.05 | >$0.10 |
| Triangulation Coverage | <80% | <60% |

---

## 10. TEST SCENARIOS

### 10.1 Chaos Engineering Tests

```python
def test_provider_failure():
    """All providers fail - graceful degradation"""

def test_rate_limit_storm():
    """Simultaneous rate limiting from all providers"""

def test_slow_provider():
    """One provider 10x slower - should timeout and fallback"""

def test_conflicting_results():
    """Sources directly contradict - report both views"""

def test_zero_relevant_results():
    """No results match query - appropriate messaging"""
```

### 10.2 Adversarial Tests

```python
def test_prompt_injection():
    """Malicious query trying to override system prompt"""

def test_unicode_obfuscation():
    """Query with lookalike Unicode characters"""

def test_extremely_long_query():
    """10K+ character query - should truncate safely"""

def test_rapid_fire_queries():
    """100 queries/second - rate limiting kicks in"""
```

---

## CHANGELOG

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2025-12-12 | Initial edge cases documentation |
| | | 9 categories, 40+ edge cases documented |
| | | Test scenarios for chaos engineering |

---

*WSOP Edge Cases v1.1 | Robustness through exhaustive failure analysis*
