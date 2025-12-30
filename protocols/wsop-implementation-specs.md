# Web Search Optimization Protocol - Implementation Specifications v1.1

> Testable Implementation Specs Derived from Academic Paper Analysis

**Created:** 2025-12-12
**Base Protocol:** `web-search-optimization-protocol.yaml` v1.0.0
**Source Papers:**

- HyDE (arXiv:2212.10496) - ACL 2023
- Query Rewriting (arXiv:2305.14283) - EMNLP 2023
- CRAG (arXiv:2401.15884) - Corrective RAG
- RAG-Fusion (arXiv:2402.03367) - Multi-query RRF

---

## 1. HYDE QUERY EXPANSION

### 1.1 Algorithm Specification

```python
def hyde_expand(query: str, llm: LLM, encoder: Encoder) -> Vector:
    """
    HyDE: Hypothetical Document Embeddings

    Key Insight: Dense bottleneck filters hallucinations.
    The encoder captures only the document-relevant aspects,
    discarding incorrect specifics in the hypothetical answer.

    Reference: Section 2.2, Figure 1 of arXiv:2212.10496
    """
    # Step 1: Generate hypothetical document
    instruction = (
        "Write a passage that answers the question. "
        "Include specific facts and technical details."
    )
    hypothetical_doc = llm.generate(
        prompt=f"{instruction}\n\nQuestion: {query}",
        temperature=0.7,  # Allow creative generation
        max_tokens=256    # ~1 paragraph
    )

    # Step 2: Encode hypothetical document (NOT the query)
    # This is the key difference from standard retrieval
    query_vector = encoder.encode(hypothetical_doc)

    # Step 3: Search corpus using document-to-document similarity
    return query_vector
```

### 1.2 Testable Assertions

| Test ID  | Assertion                                       | Expected Outcome                      |
| -------- | ----------------------------------------------- | ------------------------------------- |
| HYDE-001 | HyDE generates valid hypothetical document      | Non-empty, contextually relevant text |
| HYDE-002 | HyDE embedding differs from raw query embedding | Cosine similarity < 0.95              |
| HYDE-003 | HyDE retrieval improves recall vs raw query     | Recall@10 improvement >= 5%           |
| HYDE-004 | HyDE works zero-shot (no training data)         | No labeled data required              |
| HYDE-005 | Dense bottleneck filters hallucinations         | False facts in hyp-doc not in results |

### 1.3 Integration Points

```yaml
integration:
  trigger: "query_complexity in ['moderate', 'complex']"
  position: "phase_2_expansion"
  fallback: "raw_query_embedding"
  cache_key: "hyde:{query_hash}"
  cache_ttl: 3600 # 1 hour
```

---

## 2. CORRECTIVE RAG (CRAG)

### 2.1 Retrieval Evaluator Specification

```python
class RetrievalEvaluator:
    """
    Lightweight retrieval evaluator for CRAG.
    Determines if retrieved documents are suitable for answering.

    Reference: Section 3.1 of arXiv:2401.15884
    Architecture: T5-large fine-tuned on labeled relevance data
    """

    ACTIONS = {
        "CORRECT": 1.0,     # Confidence >= threshold
        "INCORRECT": 0.0,   # Confidence < lower_threshold
        "AMBIGUOUS": 0.5    # Between thresholds
    }

    def __init__(
        self,
        model: str = "t5-large",
        upper_threshold: float = 0.7,
        lower_threshold: float = 0.3
    ):
        self.model = model
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def evaluate(
        self,
        query: str,
        documents: List[Document]
    ) -> Tuple[str, float]:
        """
        Evaluate retrieval quality and determine action.

        Returns:
            action: "CORRECT", "INCORRECT", or "AMBIGUOUS"
            confidence: 0.0-1.0 score
        """
        # Score each document
        scores = [
            self._score_document(query, doc)
            for doc in documents
        ]

        # Aggregate using max (any relevant doc is sufficient)
        max_score = max(scores) if scores else 0.0

        # Determine action
        if max_score >= self.upper_threshold:
            action = "CORRECT"
        elif max_score <= self.lower_threshold:
            action = "INCORRECT"
        else:
            action = "AMBIGUOUS"

        return action, max_score

    def _score_document(self, query: str, doc: Document) -> float:
        """Score single document relevance."""
        # T5 scoring prompt
        input_text = f"Query: {query}\nDocument: {doc.text}\nRelevant:"
        return self.model.score(input_text, ["Yes", "No"])
```

### 2.2 Knowledge Refinement Algorithm

```python
def knowledge_refinement(
    documents: List[Document],
    query: str
) -> str:
    """
    CRAG Knowledge Refinement: Decompose-Filter-Recompose

    Key Insight: Internal knowledge strips retrieved knowledge into
    fine-grained units, filters irrelevant, reconstructs cleanly.

    Reference: Section 3.2 of arXiv:2401.15884
    """
    refined_units = []

    for doc in documents:
        # DECOMPOSE: Split into fine-grained knowledge strips
        # Using heuristic sentence/segment boundary detection
        knowledge_strips = decompose_to_strips(doc.text)

        for strip in knowledge_strips:
            # FILTER: Score each strip for query relevance
            relevance = score_strip_relevance(strip, query)

            if relevance >= STRIP_THRESHOLD:
                refined_units.append(strip)

    # RECOMPOSE: Concatenate relevant strips
    refined_knowledge = " ".join(refined_units)

    return refined_knowledge


def decompose_to_strips(text: str) -> List[str]:
    """
    Decompose document into knowledge strips.
    Strips are minimal units of factual information.
    """
    # Sentence-level decomposition
    sentences = sent_tokenize(text)

    # Further split long sentences at clause boundaries
    strips = []
    for sent in sentences:
        if len(sent.split()) > 25:  # Long sentence
            # Split at conjunctions, semicolons
            sub_strips = split_at_clause_boundaries(sent)
            strips.extend(sub_strips)
        else:
            strips.append(sent)

    return strips
```

### 2.3 Web Search Fallback

```python
def crag_pipeline(
    query: str,
    corpus_retriever: Retriever,
    web_search: WebSearchAPI,
    evaluator: RetrievalEvaluator,
    generator: LLM
) -> str:
    """
    Complete CRAG pipeline with web search fallback.

    Performance: +7% on PopQA, +14.9% FactScore on Biography
    Reference: Table 1 of arXiv:2401.15884
    """
    # Step 1: Initial retrieval from static corpus
    initial_docs = corpus_retriever.retrieve(query, k=5)

    # Step 2: Evaluate retrieval quality
    action, confidence = evaluator.evaluate(query, initial_docs)

    # Step 3: Take corrective action
    if action == "CORRECT":
        # Use refined internal knowledge
        context = knowledge_refinement(initial_docs, query)

    elif action == "INCORRECT":
        # Web search fallback - static corpus insufficient
        web_results = web_search.search(query, num_results=10)
        context = knowledge_refinement(web_results, query)

    elif action == "AMBIGUOUS":
        # Combine both sources
        web_results = web_search.search(query, num_results=5)
        combined = initial_docs + web_results
        context = knowledge_refinement(combined, query)

    # Step 4: Generate answer with refined context
    answer = generator.generate(
        prompt=f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    )

    return answer
```

### 2.4 Testable Assertions

| Test ID  | Assertion                                | Expected Outcome                                   |
| -------- | ---------------------------------------- | -------------------------------------------------- |
| CRAG-001 | Evaluator classifies into 3 categories   | Output in {CORRECT, INCORRECT, AMBIGUOUS}          |
| CRAG-002 | Web search triggered on INCORRECT        | web_search.search() called when action="INCORRECT" |
| CRAG-003 | Knowledge refinement reduces token count | refined_len < original_len \* 0.7                  |
| CRAG-004 | Decomposition produces valid strips      | Each strip is 1-2 sentences                        |
| CRAG-005 | Performance matches paper benchmarks     | PopQA accuracy >= baseline + 5%                    |
| CRAG-006 | Ambiguous combines both sources          | Both corpus and web results in context             |

---

## 3. RAG-FUSION (Multi-Query + RRF)

### 3.1 Multi-Query Generation

```python
def generate_multi_queries(
    original_query: str,
    llm: LLM,
    num_queries: int = 4
) -> List[str]:
    """
    Generate multiple query perspectives for RAG-Fusion.

    Key Insight: Different query formulations surface different
    relevant documents. Combining via RRF captures diversity.

    Reference: Section 2.1 of arXiv:2402.03367
    """
    prompt = f"""Generate {num_queries} different search queries that would
help answer this question. Each query should explore a different aspect
or use different terminology.

Original question: {original_query}

Generated queries (one per line):"""

    response = llm.generate(
        prompt=prompt,
        temperature=0.8,  # High temp for diversity
        max_tokens=200
    )

    queries = [q.strip() for q in response.split("\n") if q.strip()]

    # Always include original query
    if original_query not in queries:
        queries.insert(0, original_query)

    return queries[:num_queries + 1]
```

### 3.2 Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(
    result_sets: List[List[Document]],
    k: int = 60  # Smoothing constant (standard value)
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion for combining multiple ranked lists.

    Formula: RRF_score(d) = sum(1 / (rank(d) + k)) for each list

    Key Insight: RRF is parameter-free (k=60 is standard),
    robust to outliers, and handles heterogeneous score scales.

    Reference: Section 2.2 of arXiv:2402.03367
    Original: Cormack et al., SIGIR 2009
    """
    doc_scores: Dict[str, float] = {}
    doc_objects: Dict[str, Document] = {}

    for result_set in result_sets:
        for rank, doc in enumerate(result_set, start=1):
            doc_id = doc.id

            # RRF formula: 1 / (rank + k)
            rrf_contribution = 1.0 / (rank + k)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0
                doc_objects[doc_id] = doc

            doc_scores[doc_id] += rrf_contribution

    # Sort by RRF score descending
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [(doc_objects[doc_id], score) for doc_id, score in sorted_docs]
```

### 3.3 Complete RAG-Fusion Pipeline

```python
def rag_fusion_pipeline(
    query: str,
    retriever: Retriever,
    llm: LLM,
    generator: LLM,
    k: int = 60,
    top_k_per_query: int = 10,
    final_top_k: int = 5
) -> str:
    """
    Complete RAG-Fusion pipeline.

    Trade-off: 1.77x slower but more comprehensive answers.
    Reference: Table 2 of arXiv:2402.03367
    """
    # Step 1: Generate multiple query perspectives
    queries = generate_multi_queries(query, llm, num_queries=4)

    # Step 2: Retrieve for each query (parallelizable)
    result_sets = []
    for q in queries:
        results = retriever.retrieve(q, k=top_k_per_query)
        result_sets.append(results)

    # Step 3: Fuse with RRF
    fused_results = reciprocal_rank_fusion(result_sets, k=k)

    # Step 4: Take top-k from fused results
    top_docs = [doc for doc, score in fused_results[:final_top_k]]

    # Step 5: Generate answer
    context = "\n\n".join([doc.text for doc in top_docs])
    answer = generator.generate(
        prompt=f"Based on the following sources:\n{context}\n\nAnswer: {query}"
    )

    return answer
```

### 3.4 Testable Assertions

| Test ID  | Assertion                              | Expected Outcome                         |
| -------- | -------------------------------------- | ---------------------------------------- |
| RAGF-001 | Multi-query generates diverse queries  | Jaccard similarity < 0.5 between queries |
| RAGF-002 | RRF score formula is correct           | score = sum(1/(rank+k))                  |
| RAGF-003 | RRF with k=60 produces stable rankings | Ranking variance < 5% across runs        |
| RAGF-004 | Fusion improves recall vs single query | Recall@10 improvement >= 10%             |
| RAGF-005 | Original query always included         | original_query in generated_queries      |
| RAGF-006 | Latency within acceptable bounds       | total_time < 1.77 \* single_query_time   |

---

## 4. QUERY REWRITING (Rewrite-Retrieve-Read)

### 4.1 Trainable Rewriter Specification

```python
class QueryRewriter:
    """
    Trainable query rewriter using Reinforcement Learning.

    Architecture: T5-large or similar seq2seq model
    Training: PPO with reward = EM + F1 + Hit

    Reference: Section 3 of arXiv:2305.14283
    """

    def __init__(
        self,
        model: str = "t5-large",
        use_rl: bool = True
    ):
        self.model = load_model(model)
        self.use_rl = use_rl

    def rewrite(self, query: str, context: Optional[str] = None) -> str:
        """
        Rewrite query for improved retrieval.

        Args:
            query: Original user query
            context: Optional conversation context

        Returns:
            Rewritten query optimized for web search
        """
        if context:
            input_text = f"Context: {context}\nQuery: {query}\nRewrite:"
        else:
            input_text = f"Query: {query}\nRewrite:"

        rewritten = self.model.generate(
            input_text,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

        return rewritten


class PPORewardFunction:
    """
    Reward function for RL-based rewriter training.

    Reward = alpha * EM + beta * F1 + gamma * Hit

    Reference: Equation 4 of arXiv:2305.14283
    """

    def __init__(
        self,
        alpha: float = 0.4,  # Exact match weight
        beta: float = 0.4,   # F1 weight
        gamma: float = 0.2   # Hit (retrieval success) weight
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_reward(
        self,
        generated_answer: str,
        ground_truth: str,
        retrieval_hit: bool
    ) -> float:
        """Compute reward for PPO training."""
        em = exact_match(generated_answer, ground_truth)
        f1 = token_f1(generated_answer, ground_truth)
        hit = 1.0 if retrieval_hit else 0.0

        reward = (
            self.alpha * em +
            self.beta * f1 +
            self.gamma * hit
        )

        return reward
```

### 4.2 Rewrite-Retrieve-Read Pipeline

```python
def rewrite_retrieve_read(
    query: str,
    rewriter: QueryRewriter,
    retriever: WebSearchAPI,
    reader: LLM
) -> str:
    """
    Complete Rewrite-Retrieve-Read pipeline.

    Key Insight: "There is inevitably a gap between the input text
    and the needed knowledge in retrieval" - proactive rewriting
    addresses this gap.

    Reference: Figure 1 of arXiv:2305.14283
    """
    # REWRITE: Transform query for better retrieval
    rewritten_query = rewriter.rewrite(query)

    # RETRIEVE: Search using rewritten query
    documents = retriever.search(rewritten_query, num_results=5)

    # READ: Generate answer from retrieved context
    context = "\n".join([doc.text for doc in documents])
    answer = reader.generate(
        prompt=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    return answer
```

### 4.3 Testable Assertions

| Test ID | Assertion                       | Expected Outcome             |
| ------- | ------------------------------- | ---------------------------- |
| QRW-001 | Rewriter produces valid query   | Non-empty, <64 tokens        |
| QRW-002 | Rewritten differs from original | edit_distance > 0            |
| QRW-003 | Rewritten improves retrieval    | Recall improvement >= 5%     |
| QRW-004 | PPO reward in valid range       | 0.0 <= reward <= 1.0         |
| QRW-005 | T5-large can be fine-tuned      | Model accepts PPO gradients  |
| QRW-006 | Web search integration works    | Retriever returns valid docs |

---

## 5. INTEGRATED WSOP PIPELINE

### 5.1 Complete Integration

```python
class WebSearchOptimizationPipeline:
    """
    Integrated WSOP combining all techniques.

    Components:
    - HyDE for query expansion
    - CRAG for corrective retrieval
    - RAG-Fusion for multi-query diversity
    - Query Rewriting for web search optimization
    """

    def __init__(self, config: WSOPConfig):
        self.hyde_expander = HyDEExpander(config.llm)
        self.crag_evaluator = RetrievalEvaluator()
        self.rag_fusion = RAGFusionPipeline()
        self.rewriter = QueryRewriter()
        self.web_search = WebSearchAPI(config.search_api)
        self.generator = config.llm

    def search(
        self,
        query: str,
        complexity: str = "auto"
    ) -> SearchResult:
        """
        Execute optimized web search.

        Flow:
        1. Classify complexity
        2. Apply appropriate techniques
        3. Evaluate and correct
        4. Synthesize answer
        """
        # Step 1: Complexity routing
        if complexity == "auto":
            complexity = self._classify_complexity(query)

        # Step 2: Query processing
        if complexity == "simple":
            # Direct retrieval
            results = self._simple_search(query)

        elif complexity == "moderate":
            # HyDE + single retrieval
            expanded_query = self.hyde_expander.expand(query)
            results = self.web_search.search(expanded_query)

            # CRAG evaluation
            action, conf = self.crag_evaluator.evaluate(query, results)
            if action == "INCORRECT":
                results = self._fallback_search(query)

        elif complexity == "complex":
            # Full pipeline: Rewrite + HyDE + RAG-Fusion + CRAG
            rewritten = self.rewriter.rewrite(query)
            expanded = self.hyde_expander.expand(rewritten)

            # Multi-query with fusion
            queries = [query, rewritten, expanded]
            results = self.rag_fusion.search_and_fuse(queries)

            # Corrective evaluation
            action, conf = self.crag_evaluator.evaluate(query, results)
            if action != "CORRECT":
                results = self._knowledge_refinement(results, query)

        # Step 3: Credibility scoring
        scored_results = self._score_credibility(results)

        # Step 4: Generate answer
        answer = self._generate_answer(query, scored_results)

        return SearchResult(
            query=query,
            answer=answer,
            sources=scored_results,
            complexity=complexity
        )
```

### 5.2 Testable End-to-End Assertions

| Test ID  | Assertion                            | Expected Outcome                      |
| -------- | ------------------------------------ | ------------------------------------- |
| WSOP-001 | Simple queries skip expansion        | HyDE not called for simple            |
| WSOP-002 | Complex queries use full pipeline    | All 4 techniques engaged              |
| WSOP-003 | Fallback triggered on low confidence | Web search called when conf < 0.3     |
| WSOP-004 | Credibility scoring applied          | All results have credibility tier     |
| WSOP-005 | Answer includes citations            | Sources cited in output               |
| WSOP-006 | Latency scales with complexity       | simple < moderate < complex           |
| WSOP-007 | Pipeline handles empty results       | Graceful degradation on no results    |
| WSOP-008 | RRF integration works                | Fusion applied to multi-query results |

---

## 6. TEST SUITE SPECIFICATION

### 6.1 Unit Tests

```python
# tests/test_wsop_implementation.py

class TestHyDE:
    def test_generates_hypothetical_doc(self):
        """HYDE-001: HyDE generates valid hypothetical document"""

    def test_embedding_differs_from_raw(self):
        """HYDE-002: HyDE embedding differs from raw query embedding"""

    def test_improves_recall(self):
        """HYDE-003: HyDE retrieval improves recall vs raw query"""


class TestCRAG:
    def test_evaluator_classifies_correctly(self):
        """CRAG-001: Evaluator classifies into 3 categories"""

    def test_web_search_on_incorrect(self):
        """CRAG-002: Web search triggered on INCORRECT"""

    def test_knowledge_refinement_reduces_tokens(self):
        """CRAG-003: Knowledge refinement reduces token count"""


class TestRAGFusion:
    def test_multi_query_diversity(self):
        """RAGF-001: Multi-query generates diverse queries"""

    def test_rrf_formula_correct(self):
        """RAGF-002: RRF score formula is correct"""
        k = 60
        ranks = [1, 3, 5]
        expected = sum(1/(r + k) for r in ranks)
        # Assert RRF implementation matches


class TestQueryRewriting:
    def test_rewriter_produces_valid_query(self):
        """QRW-001: Rewriter produces valid query"""

    def test_rewritten_differs(self):
        """QRW-002: Rewritten differs from original"""


class TestIntegration:
    def test_complexity_routing(self):
        """WSOP-001: Simple queries skip expansion"""

    def test_full_pipeline_complex(self):
        """WSOP-002: Complex queries use full pipeline"""
```

### 6.2 Benchmark Tests

```python
class TestBenchmarks:
    """Performance benchmarks against paper claims."""

    def test_crag_popqa_improvement(self):
        """CRAG claims +7% on PopQA"""
        baseline = run_baseline_popqa()
        crag_result = run_crag_popqa()
        assert crag_result >= baseline + 0.05  # Conservative 5%

    def test_hyde_contriever_improvement(self):
        """HyDE claims 'significantly outperforms Contriever'"""
        contriever = run_contriever_trec()
        hyde = run_hyde_trec()
        assert hyde > contriever

    def test_rag_fusion_latency(self):
        """RAG-Fusion claims 1.77x slower"""
        single_time = measure_single_query()
        fusion_time = measure_rag_fusion()
        assert fusion_time < single_time * 2.0  # Allow up to 2x
```

---

## 7. CONFIGURATION SPECIFICATION

```yaml
# config/wsop.yaml

wsop:
  version: "1.1.0"

  hyde:
    enabled: true
    llm: "claude-sonnet-4"
    temperature: 0.7
    max_tokens: 256

  crag:
    enabled: true
    evaluator_model: "t5-large"
    upper_threshold: 0.7
    lower_threshold: 0.3
    strip_threshold: 0.5

  rag_fusion:
    enabled: true
    num_queries: 4
    rrf_k: 60
    top_k_per_query: 10
    final_top_k: 5

  query_rewriting:
    enabled: true
    model: "t5-large"
    use_rl: false # Set true when RL-trained model available
    max_length: 64

  routing:
    auto_classify: true
    simple_threshold: 0.3
    complex_threshold: 0.7

  fallback:
    web_search_on_incorrect: true
    max_web_results: 10
```

---

## 8. CHANGELOG

| Version | Date       | Changes                                               |
| ------- | ---------- | ----------------------------------------------------- |
| 1.1.0   | 2025-12-12 | Added testable implementation specs from PDF analysis |
|         |            | Integrated HyDE, CRAG, RAG-Fusion, Query Rewriting    |
|         |            | Created 30+ testable assertions                       |
|         |            | Added end-to-end pipeline specification               |

---

_WSOP Implementation Specs v1.1 | Derived from 4 academic papers | 30+ testable assertions_
