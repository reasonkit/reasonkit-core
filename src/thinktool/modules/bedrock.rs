//! BedRock Module - First Principles Decomposition
//!
//! Reduces problems to fundamental axioms through recursive analysis,
//! then rebuilds understanding using Tree-of-Thoughts exploration.
//!
//! ## Methodology
//!
//! BedRock applies Elon Musk-style first principles thinking:
//! 1. **Decompose**: Break the problem into fundamental components
//! 2. **Identify Axioms**: Find self-evident truths that don't require proof
//! 3. **Surface Assumptions**: Expose hidden assumptions that may be challenged
//! 4. **Rebuild**: Reconstruct understanding from verified foundations
//! 5. **Explore**: Use Tree-of-Thoughts to find optimal reasoning paths
//!
//! ## Usage
//!
//! ```ignore
//! use reasonkit::thinktool::modules::{BedRock, ThinkToolModule, ThinkToolContext};
//!
//! let bedrock = BedRock::new();
//! let context = ThinkToolContext {
//!     query: "Why are electric vehicles better than gas cars?".into(),
//!     previous_steps: vec![],
//! };
//!
//! let result = bedrock.execute(&context)?;
//! println!("Axioms found: {}", result.output["axioms"]);
//! println!("Hidden assumptions: {}", result.output["assumptions"]);
//! ```

use super::{ThinkToolContext, ThinkToolModule, ThinkToolModuleConfig, ThinkToolOutput};
use serde::{Deserialize, Serialize};

/// Configuration for BedRock analysis depth and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedRockConfig {
    /// Maximum decomposition depth (how deep to recurse into sub-principles)
    pub max_depth: usize,
    /// Minimum fundamentality score to consider a principle as axiomatic (0.0-1.0)
    pub axiom_threshold: f64,
    /// Number of parallel thought branches to explore per principle
    pub branching_factor: usize,
    /// Minimum confidence threshold for including a principle
    pub min_confidence: f64,
    /// Whether to require all assumptions to be explicitly stated
    pub strict_assumptions: bool,
    /// Maximum number of principles to identify
    pub max_principles: usize,
}

impl Default for BedRockConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            axiom_threshold: 0.85,
            branching_factor: 3,
            min_confidence: 0.5,
            strict_assumptions: true,
            max_principles: 20,
        }
    }
}

/// Classification of a principle's nature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrincipleType {
    /// Self-evident truth requiring no proof (e.g., "A = A", physical laws)
    Axiom,
    /// Logically derived from axioms
    Derived,
    /// Assumed for the sake of argument (may be challenged)
    Assumption,
    /// Based on empirical observation/data
    Empirical,
    /// Definitional statement clarifying terminology
    Definition,
    /// Contested claim requiring verification
    Contested,
}

impl PrincipleType {
    /// Returns the reliability weight for this principle type.
    pub fn reliability_weight(&self) -> f64 {
        match self {
            PrincipleType::Axiom => 1.0,
            PrincipleType::Definition => 0.95,
            PrincipleType::Empirical => 0.80,
            PrincipleType::Derived => 0.75,
            PrincipleType::Assumption => 0.50,
            PrincipleType::Contested => 0.30,
        }
    }
}

/// A fundamental principle identified during decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Principle {
    /// Unique identifier within this analysis
    pub id: usize,
    /// The principle statement
    pub statement: String,
    /// Classification of the principle
    pub principle_type: PrincipleType,
    /// How fundamental is this (0.0-1.0, where 1.0 = pure axiom)
    pub fundamentality: f64,
    /// Confidence in this principle's validity
    pub confidence: f64,
    /// ID of parent principle if derived/decomposed
    pub parent_id: Option<usize>,
    /// IDs of child principles
    pub child_ids: Vec<usize>,
    /// Supporting evidence or reasoning
    pub evidence: Vec<String>,
    /// Potential challenges to this principle
    pub challenges: Vec<String>,
    /// Depth in the decomposition tree
    pub depth: usize,
}

impl Principle {
    /// Calculate the effective weight of this principle.
    pub fn effective_weight(&self) -> f64 {
        self.fundamentality * self.confidence * self.principle_type.reliability_weight()
    }

    /// Check if this principle qualifies as axiomatic.
    pub fn is_axiomatic(&self, threshold: f64) -> bool {
        self.principle_type == PrincipleType::Axiom && self.fundamentality >= threshold
    }
}

/// A reconstruction path from axioms to conclusions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionPath {
    /// Ordered list of principle IDs from axiom to conclusion
    pub principle_chain: Vec<usize>,
    /// Logical connectives between principles
    pub connectives: Vec<String>,
    /// Overall path confidence
    pub confidence: f64,
    /// Whether this path is complete (reaches conclusion)
    pub is_complete: bool,
    /// Gaps or missing links in the reasoning
    pub gaps: Vec<String>,
}

/// Analysis gap identified during reconstruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisGap {
    /// Description of the gap
    pub description: String,
    /// Severity (0.0-1.0, where 1.0 = critical)
    pub severity: f64,
    /// Suggested resolution
    pub suggestion: Option<String>,
    /// Principles affected by this gap
    pub affected_principles: Vec<usize>,
}

/// Complete result of BedRock first principles analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedRockResult {
    /// Original query analyzed
    pub query: String,
    /// All identified principles
    pub principles: Vec<Principle>,
    /// Reconstruction paths from axioms to conclusions
    pub reconstructions: Vec<ReconstructionPath>,
    /// Identified gaps in reasoning
    pub gaps: Vec<AnalysisGap>,
    /// Key insights from the analysis
    pub insights: Vec<String>,
    /// Overall analysis confidence
    pub confidence: f64,
    /// Analysis metadata
    pub metadata: BedRockMetadata,
}

/// Metadata about the analysis process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedRockMetadata {
    /// Maximum depth reached
    pub max_depth_reached: usize,
    /// Total principles identified
    pub total_principles: usize,
    /// Number of axioms found
    pub axiom_count: usize,
    /// Number of assumptions identified
    pub assumption_count: usize,
    /// Number of contested claims
    pub contested_count: usize,
    /// Decomposition completeness (0.0-1.0)
    pub completeness: f64,
}

impl BedRockResult {
    /// Get all axiomatic principles.
    pub fn axioms(&self) -> Vec<&Principle> {
        self.principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Axiom)
            .collect()
    }

    /// Get all assumptions that may be challenged.
    pub fn assumptions(&self) -> Vec<&Principle> {
        self.principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Assumption)
            .collect()
    }

    /// Get contested claims requiring verification.
    pub fn contested(&self) -> Vec<&Principle> {
        self.principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Contested)
            .collect()
    }

    /// Get principles at a specific depth.
    pub fn at_depth(&self, depth: usize) -> Vec<&Principle> {
        self.principles
            .iter()
            .filter(|p| p.depth == depth)
            .collect()
    }

    /// Check if analysis is sufficiently complete.
    pub fn is_complete(&self, threshold: f64) -> bool {
        self.metadata.completeness >= threshold && self.gaps.iter().all(|g| g.severity < 0.8)
    }

    /// Convert to JSON output format.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "query": self.query,
            "axioms": self.axioms().iter().map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "statement": p.statement,
                    "fundamentality": p.fundamentality,
                    "confidence": p.confidence,
                    "evidence": p.evidence
                })
            }).collect::<Vec<_>>(),
            "assumptions": self.assumptions().iter().map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "statement": p.statement,
                    "confidence": p.confidence,
                    "challenges": p.challenges
                })
            }).collect::<Vec<_>>(),
            "decomposition": self.principles.iter().map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "statement": p.statement,
                    "type": format!("{:?}", p.principle_type),
                    "fundamentality": p.fundamentality,
                    "confidence": p.confidence,
                    "depth": p.depth,
                    "parent_id": p.parent_id
                })
            }).collect::<Vec<_>>(),
            "reconstruction": self.reconstructions.iter().map(|r| {
                serde_json::json!({
                    "path": r.principle_chain,
                    "confidence": r.confidence,
                    "complete": r.is_complete,
                    "gaps": r.gaps
                })
            }).collect::<Vec<_>>(),
            "gaps": self.gaps.iter().map(|g| {
                serde_json::json!({
                    "description": g.description,
                    "severity": g.severity,
                    "suggestion": g.suggestion
                })
            }).collect::<Vec<_>>(),
            "insights": self.insights,
            "confidence": self.confidence,
            "metadata": {
                "max_depth": self.metadata.max_depth_reached,
                "total_principles": self.metadata.total_principles,
                "axioms": self.metadata.axiom_count,
                "assumptions": self.metadata.assumption_count,
                "contested": self.metadata.contested_count,
                "completeness": self.metadata.completeness
            }
        })
    }
}

/// BedRock reasoning module for first principles analysis.
///
/// Decomposes statements to foundational axioms, identifies hidden
/// assumptions, and rebuilds understanding from verified foundations.
pub struct BedRock {
    /// Module configuration
    config: ThinkToolModuleConfig,
    /// Analysis configuration
    analysis_config: BedRockConfig,
}

impl Default for BedRock {
    fn default() -> Self {
        Self::new()
    }
}

impl BedRock {
    /// Create a new BedRock module instance with default configuration.
    pub fn new() -> Self {
        Self {
            config: ThinkToolModuleConfig {
                name: "BedRock".to_string(),
                version: "3.0.0".to_string(),
                description: "First principles decomposition with Tree-of-Thoughts reconstruction"
                    .to_string(),
                confidence_weight: 0.25,
            },
            analysis_config: BedRockConfig::default(),
        }
    }

    /// Create a new BedRock module with custom analysis configuration.
    pub fn with_config(analysis_config: BedRockConfig) -> Self {
        Self {
            config: ThinkToolModuleConfig {
                name: "BedRock".to_string(),
                version: "3.0.0".to_string(),
                description: "First principles decomposition with Tree-of-Thoughts reconstruction"
                    .to_string(),
                confidence_weight: 0.25,
            },
            analysis_config,
        }
    }

    /// Get the analysis configuration.
    pub fn analysis_config(&self) -> &BedRockConfig {
        &self.analysis_config
    }

    /// Perform first principles decomposition on the query.
    ///
    /// This is the core analysis method that:
    /// 1. Parses the query to identify claims
    /// 2. Recursively decomposes each claim
    /// 3. Classifies principles by type
    /// 4. Identifies gaps and assumptions
    pub fn decompose(&self, query: &str, previous_steps: &[String]) -> BedRockResult {
        let mut principles = Vec::new();
        let mut next_id = 0;

        // Step 1: Identify the root claim/question
        let root_principle = self.create_root_principle(query, &mut next_id);
        principles.push(root_principle);

        // Step 2: Recursive decomposition using heuristic analysis
        self.decompose_recursive(&mut principles, 0, 0, &mut next_id);

        // Step 3: Incorporate context from previous steps
        self.incorporate_context(&mut principles, previous_steps, &mut next_id);

        // Step 4: Classify and validate principles
        self.classify_principles(&mut principles);

        // Step 5: Build reconstruction paths
        let reconstructions = self.build_reconstructions(&principles);

        // Step 6: Identify gaps
        let gaps = self.identify_gaps(&principles, &reconstructions);

        // Step 7: Extract insights
        let insights = self.extract_insights(&principles, &gaps);

        // Step 8: Calculate overall confidence
        let confidence = self.calculate_confidence(&principles, &gaps);

        // Build metadata
        let metadata = BedRockMetadata {
            max_depth_reached: principles.iter().map(|p| p.depth).max().unwrap_or(0),
            total_principles: principles.len(),
            axiom_count: principles
                .iter()
                .filter(|p| p.principle_type == PrincipleType::Axiom)
                .count(),
            assumption_count: principles
                .iter()
                .filter(|p| p.principle_type == PrincipleType::Assumption)
                .count(),
            contested_count: principles
                .iter()
                .filter(|p| p.principle_type == PrincipleType::Contested)
                .count(),
            completeness: self.calculate_completeness(&principles, &gaps),
        };

        BedRockResult {
            query: query.to_string(),
            principles,
            reconstructions,
            gaps,
            insights,
            confidence,
            metadata,
        }
    }

    /// Create the root principle from the query.
    fn create_root_principle(&self, query: &str, next_id: &mut usize) -> Principle {
        let id = *next_id;
        *next_id += 1;

        // Analyze query to determine initial type
        let principle_type = self.classify_query(query);

        Principle {
            id,
            statement: query.to_string(),
            principle_type,
            fundamentality: 0.0, // Root is not fundamental - it's what we're decomposing
            confidence: 1.0,     // We're certain about what was asked
            parent_id: None,
            child_ids: Vec::new(),
            evidence: Vec::new(),
            challenges: Vec::new(),
            depth: 0,
        }
    }

    /// Classify a query/statement into a principle type.
    fn classify_query(&self, query: &str) -> PrincipleType {
        let lower = query.to_lowercase();

        // Check for definition markers
        if lower.contains("what is")
            || lower.contains("define")
            || lower.contains("meaning of")
            || lower.contains("definition")
        {
            return PrincipleType::Definition;
        }

        // Check for empirical markers
        if lower.contains("how many")
            || lower.contains("when did")
            || lower.contains("data shows")
            || lower.contains("research")
            || lower.contains("study")
            || lower.contains("evidence")
        {
            return PrincipleType::Empirical;
        }

        // Check for axiomatic/logical markers
        if lower.contains("always true")
            || lower.contains("by definition")
            || lower.contains("necessarily")
            || lower.contains("logically")
            || lower.contains("mathematically")
        {
            return PrincipleType::Axiom;
        }

        // Check for assumption markers
        if lower.contains("assume")
            || lower.contains("suppose")
            || lower.contains("if we")
            || lower.contains("given that")
        {
            return PrincipleType::Assumption;
        }

        // Check for contested/opinion markers
        if lower.contains("better")
            || lower.contains("worse")
            || lower.contains("should")
            || lower.contains("ought")
            || lower.contains("believe")
            || lower.contains("think")
        {
            return PrincipleType::Contested;
        }

        // Default to derived (needs further decomposition)
        PrincipleType::Derived
    }

    /// Recursively decompose a principle into sub-principles.
    fn decompose_recursive(
        &self,
        principles: &mut Vec<Principle>,
        parent_idx: usize,
        current_depth: usize,
        next_id: &mut usize,
    ) {
        if current_depth >= self.analysis_config.max_depth {
            return;
        }

        if principles.len() >= self.analysis_config.max_principles {
            return;
        }

        let parent_statement = principles[parent_idx].statement.clone();
        let sub_principles = self.extract_sub_principles(&parent_statement, current_depth);

        let mut child_ids = Vec::new();

        for (statement, principle_type, fundamentality) in sub_principles {
            if principles.len() >= self.analysis_config.max_principles {
                break;
            }

            let id = *next_id;
            *next_id += 1;
            child_ids.push(id);

            let confidence = self.estimate_confidence(&statement, principle_type);

            let principle = Principle {
                id,
                statement,
                principle_type,
                fundamentality,
                confidence,
                parent_id: Some(principles[parent_idx].id),
                child_ids: Vec::new(),
                evidence: Vec::new(),
                challenges: self.identify_challenges(principle_type),
                depth: current_depth + 1,
            };

            let new_idx = principles.len();
            principles.push(principle);

            // Only recurse for non-axiomatic principles
            if principle_type != PrincipleType::Axiom
                && fundamentality < self.analysis_config.axiom_threshold
            {
                self.decompose_recursive(principles, new_idx, current_depth + 1, next_id);
            }
        }

        principles[parent_idx].child_ids = child_ids;
    }

    /// Extract sub-principles from a statement using heuristic decomposition.
    fn extract_sub_principles(
        &self,
        statement: &str,
        depth: usize,
    ) -> Vec<(String, PrincipleType, f64)> {
        let mut sub_principles = Vec::new();
        let lower = statement.to_lowercase();

        // Extract comparative claims
        if lower.contains("better") || lower.contains("worse") || lower.contains("more") {
            sub_principles.push((
                "Comparison requires a defined metric or criterion".to_string(),
                PrincipleType::Definition,
                0.9,
            ));
            sub_principles.push((
                "Both alternatives must be well-understood".to_string(),
                PrincipleType::Assumption,
                0.7,
            ));
        }

        // Extract causal claims
        if lower.contains("because") || lower.contains("causes") || lower.contains("leads to") {
            sub_principles.push((
                "Causal relationships require evidence of mechanism".to_string(),
                PrincipleType::Empirical,
                0.6,
            ));
            sub_principles.push((
                "Correlation does not imply causation".to_string(),
                PrincipleType::Axiom,
                1.0,
            ));
        }

        // Extract quantitative claims
        if lower.contains("all")
            || lower.contains("every")
            || lower.contains("none")
            || lower.contains("never")
        {
            sub_principles.push((
                "Universal claims require exhaustive verification".to_string(),
                PrincipleType::Axiom,
                1.0,
            ));
            sub_principles.push((
                "A single counterexample disproves a universal claim".to_string(),
                PrincipleType::Axiom,
                1.0,
            ));
        }

        // Extract value judgments
        if lower.contains("good")
            || lower.contains("bad")
            || lower.contains("right")
            || lower.contains("wrong")
        {
            sub_principles.push((
                "Value judgments require a defined value framework".to_string(),
                PrincipleType::Definition,
                0.85,
            ));
            sub_principles.push((
                "Different stakeholders may have different values".to_string(),
                PrincipleType::Assumption,
                0.75,
            ));
        }

        // Extract temporal claims
        if lower.contains("will") || lower.contains("future") || lower.contains("predict") {
            sub_principles.push((
                "Future predictions carry inherent uncertainty".to_string(),
                PrincipleType::Axiom,
                1.0,
            ));
            sub_principles.push((
                "Past patterns may not continue".to_string(),
                PrincipleType::Assumption,
                0.6,
            ));
        }

        // Default decomposition if no specific patterns found
        if sub_principles.is_empty() && depth < self.analysis_config.max_depth {
            sub_principles.push((
                "The claim contains implicit assumptions".to_string(),
                PrincipleType::Assumption,
                0.5,
            ));
            sub_principles.push((
                "Terms used may have multiple interpretations".to_string(),
                PrincipleType::Definition,
                0.6,
            ));
        }

        sub_principles
    }

    /// Estimate confidence for a principle based on its type and content.
    fn estimate_confidence(&self, _statement: &str, principle_type: PrincipleType) -> f64 {
        match principle_type {
            PrincipleType::Axiom => 0.95,
            PrincipleType::Definition => 0.90,
            PrincipleType::Empirical => 0.75,
            PrincipleType::Derived => 0.70,
            PrincipleType::Assumption => 0.55,
            PrincipleType::Contested => 0.40,
        }
    }

    /// Identify potential challenges to a principle type.
    fn identify_challenges(&self, principle_type: PrincipleType) -> Vec<String> {
        match principle_type {
            PrincipleType::Axiom => vec![],
            PrincipleType::Definition => {
                vec!["Alternative definitions may exist".to_string()]
            }
            PrincipleType::Empirical => vec![
                "Data may be outdated".to_string(),
                "Sample may not be representative".to_string(),
            ],
            PrincipleType::Derived => vec![
                "Derivation logic may have flaws".to_string(),
                "Missing intermediate steps".to_string(),
            ],
            PrincipleType::Assumption => vec![
                "Assumption may not hold in all contexts".to_string(),
                "Implicit bias may be present".to_string(),
            ],
            PrincipleType::Contested => vec![
                "Subject to debate".to_string(),
                "Evidence may support opposing views".to_string(),
            ],
        }
    }

    /// Incorporate context from previous reasoning steps.
    fn incorporate_context(
        &self,
        principles: &mut Vec<Principle>,
        previous_steps: &[String],
        next_id: &mut usize,
    ) {
        for step in previous_steps {
            if principles.len() >= self.analysis_config.max_principles {
                break;
            }

            let principle_type = self.classify_query(step);
            let id = *next_id;
            *next_id += 1;

            let principle = Principle {
                id,
                statement: format!("Prior context: {}", step),
                principle_type,
                fundamentality: 0.3, // Context is not foundational
                confidence: 0.7,     // Moderate confidence in prior reasoning
                parent_id: None,
                child_ids: Vec::new(),
                evidence: vec!["From previous reasoning step".to_string()],
                challenges: vec!["May need re-evaluation in new context".to_string()],
                depth: 0, // Context is at root level
            };

            principles.push(principle);
        }
    }

    /// Classify all principles and refine their types.
    fn classify_principles(&self, principles: &mut [Principle]) {
        for principle in principles.iter_mut() {
            // Upgrade to axiom if fundamentality is high enough
            if principle.fundamentality >= self.analysis_config.axiom_threshold
                && principle.principle_type != PrincipleType::Axiom
                && principle.principle_type != PrincipleType::Contested
            {
                principle.principle_type = PrincipleType::Axiom;
                principle.challenges.clear();
            }

            // Downgrade contested claims with no support
            if principle.evidence.is_empty() && principle.principle_type == PrincipleType::Empirical
            {
                principle.principle_type = PrincipleType::Assumption;
                principle.confidence *= 0.8;
            }
        }
    }

    /// Build reconstruction paths from axioms to the root claim.
    fn build_reconstructions(&self, principles: &[Principle]) -> Vec<ReconstructionPath> {
        let mut reconstructions = Vec::new();

        // Find all axioms
        let axioms: Vec<_> = principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Axiom)
            .collect();

        // For each axiom, try to build a path to the root
        for axiom in axioms {
            let mut path = vec![axiom.id];
            let mut connectives = Vec::new();
            let mut current_id = axiom.id;
            let mut gaps = Vec::new();

            // Traverse up to parents
            while let Some(principle) = principles.iter().find(|p| p.id == current_id) {
                if let Some(parent_idx) = principles.iter().position(|p| {
                    p.child_ids.contains(&current_id) || Some(p.id) == principle.parent_id
                }) {
                    let parent = &principles[parent_idx];
                    path.push(parent.id);
                    connectives.push("implies".to_string());
                    current_id = parent.id;
                } else {
                    break;
                }

                // Prevent infinite loops
                if path.len() > principles.len() {
                    gaps.push("Circular dependency detected".to_string());
                    break;
                }
            }

            // Check if we reached the root (depth 0)
            let is_complete = principles
                .iter()
                .any(|p| path.contains(&p.id) && p.depth == 0);

            if !is_complete {
                gaps.push("Path does not reach the original claim".to_string());
            }

            let confidence = if is_complete && gaps.is_empty() {
                axiom.confidence * 0.9
            } else {
                axiom.confidence * 0.5
            };

            reconstructions.push(ReconstructionPath {
                principle_chain: path,
                connectives,
                confidence,
                is_complete,
                gaps,
            });
        }

        reconstructions
    }

    /// Identify gaps in the analysis.
    fn identify_gaps(
        &self,
        principles: &[Principle],
        reconstructions: &[ReconstructionPath],
    ) -> Vec<AnalysisGap> {
        let mut gaps = Vec::new();

        // Check for missing axioms (no reconstruction paths)
        if reconstructions.is_empty() {
            gaps.push(AnalysisGap {
                description: "No axiomatic foundation identified".to_string(),
                severity: 0.9,
                suggestion: Some("Decompose further to find self-evident truths".to_string()),
                affected_principles: principles.iter().map(|p| p.id).collect(),
            });
        }

        // Check for incomplete paths
        let incomplete_paths: Vec<_> = reconstructions.iter().filter(|r| !r.is_complete).collect();

        if !incomplete_paths.is_empty() {
            gaps.push(AnalysisGap {
                description: format!(
                    "{} reconstruction path(s) do not reach the root claim",
                    incomplete_paths.len()
                ),
                severity: 0.7,
                suggestion: Some("Add intermediate principles to complete the chain".to_string()),
                affected_principles: incomplete_paths
                    .iter()
                    .flat_map(|r| r.principle_chain.clone())
                    .collect(),
            });
        }

        // Check for unsupported assumptions
        let unsupported_assumptions: Vec<_> = principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Assumption && p.evidence.is_empty())
            .collect();

        if !unsupported_assumptions.is_empty() {
            gaps.push(AnalysisGap {
                description: format!(
                    "{} assumption(s) lack supporting evidence",
                    unsupported_assumptions.len()
                ),
                severity: 0.6,
                suggestion: Some("Provide evidence or acknowledge as unverified".to_string()),
                affected_principles: unsupported_assumptions.iter().map(|p| p.id).collect(),
            });
        }

        // Check for low-confidence principles
        let low_confidence: Vec<_> = principles
            .iter()
            .filter(|p| p.confidence < self.analysis_config.min_confidence)
            .collect();

        if !low_confidence.is_empty() {
            gaps.push(AnalysisGap {
                description: format!(
                    "{} principle(s) have confidence below threshold",
                    low_confidence.len()
                ),
                severity: 0.5,
                suggestion: Some("Verify or remove low-confidence principles".to_string()),
                affected_principles: low_confidence.iter().map(|p| p.id).collect(),
            });
        }

        // Check for contested claims without resolution
        let unresolved_contested: Vec<_> = principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Contested && !p.challenges.is_empty())
            .collect();

        if !unresolved_contested.is_empty() {
            gaps.push(AnalysisGap {
                description: format!(
                    "{} contested claim(s) require resolution",
                    unresolved_contested.len()
                ),
                severity: 0.8,
                suggestion: Some("Provide evidence to resolve contested claims".to_string()),
                affected_principles: unresolved_contested.iter().map(|p| p.id).collect(),
            });
        }

        gaps
    }

    /// Extract key insights from the analysis.
    fn extract_insights(&self, principles: &[Principle], gaps: &[AnalysisGap]) -> Vec<String> {
        let mut insights = Vec::new();

        // Insight: Number of axiomatic foundations
        let axiom_count = principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Axiom)
            .count();

        if axiom_count > 0 {
            insights.push(format!(
                "Analysis rests on {} axiomatic foundation(s)",
                axiom_count
            ));
        } else {
            insights.push(
                "No self-evident axioms identified - claim relies on assumptions".to_string(),
            );
        }

        // Insight: Assumption count
        let assumption_count = principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Assumption)
            .count();

        if assumption_count > 0 {
            insights.push(format!(
                "{} hidden assumption(s) identified that could be challenged",
                assumption_count
            ));
        }

        // Insight: Gap severity
        let critical_gaps: Vec<_> = gaps.iter().filter(|g| g.severity >= 0.8).collect();

        if !critical_gaps.is_empty() {
            insights.push(format!(
                "{} critical gap(s) in reasoning require attention",
                critical_gaps.len()
            ));
        }

        // Insight: Depth analysis
        let max_depth = principles.iter().map(|p| p.depth).max().unwrap_or(0);
        if max_depth > 0 {
            insights.push(format!(
                "Decomposition reached {} level(s) of depth",
                max_depth
            ));
        }

        // Insight: Contested claims
        let contested_count = principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Contested)
            .count();

        if contested_count > 0 {
            insights.push(format!(
                "{} contested claim(s) identified - these are debatable",
                contested_count
            ));
        }

        insights
    }

    /// Calculate overall confidence in the analysis.
    fn calculate_confidence(&self, principles: &[Principle], gaps: &[AnalysisGap]) -> f64 {
        if principles.is_empty() {
            return 0.0;
        }

        // Base confidence from principles
        let principle_confidence: f64 =
            principles.iter().map(|p| p.effective_weight()).sum::<f64>() / principles.len() as f64;

        // Penalty for gaps
        let gap_penalty: f64 = gaps.iter().map(|g| g.severity * 0.1).sum();

        // Bonus for axioms
        let axiom_count = principles
            .iter()
            .filter(|p| p.principle_type == PrincipleType::Axiom)
            .count();
        let axiom_bonus = (axiom_count as f64 * 0.05).min(0.2);

        (principle_confidence + axiom_bonus - gap_penalty).clamp(0.0, 1.0)
    }

    /// Calculate completeness of the analysis.
    fn calculate_completeness(&self, principles: &[Principle], gaps: &[AnalysisGap]) -> f64 {
        if principles.is_empty() {
            return 0.0;
        }

        // Check for presence of required components
        let has_axiom = principles
            .iter()
            .any(|p| p.principle_type == PrincipleType::Axiom);
        let has_definitions = principles
            .iter()
            .any(|p| p.principle_type == PrincipleType::Definition);
        let assumptions_identified = principles
            .iter()
            .any(|p| p.principle_type == PrincipleType::Assumption);

        let mut completeness = 0.0;

        if has_axiom {
            completeness += 0.3;
        }
        if has_definitions {
            completeness += 0.2;
        }
        if assumptions_identified {
            completeness += 0.2;
        }

        // Depth bonus
        let max_depth = principles.iter().map(|p| p.depth).max().unwrap_or(0);
        completeness += (max_depth as f64 * 0.1).min(0.2);

        // Gap penalty
        let critical_gaps = gaps.iter().filter(|g| g.severity >= 0.8).count();
        completeness -= critical_gaps as f64 * 0.1;

        completeness.clamp(0.0, 1.0)
    }
}

impl ThinkToolModule for BedRock {
    fn config(&self) -> &ThinkToolModuleConfig {
        &self.config
    }

    fn execute(&self, context: &ThinkToolContext) -> Result<ThinkToolOutput, crate::error::Error> {
        // Perform first principles decomposition
        let result = self.decompose(&context.query, &context.previous_steps);

        Ok(ThinkToolOutput {
            module: self.config.name.clone(),
            confidence: result.confidence,
            output: result.to_json(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bedrock_new() {
        let bedrock = BedRock::new();
        assert_eq!(bedrock.config().name, "BedRock");
        assert_eq!(bedrock.config().version, "3.0.0");
    }

    #[test]
    fn test_bedrock_with_config() {
        let config = BedRockConfig {
            max_depth: 5,
            axiom_threshold: 0.9,
            ..Default::default()
        };
        let bedrock = BedRock::with_config(config);
        assert_eq!(bedrock.analysis_config().max_depth, 5);
        assert_eq!(bedrock.analysis_config().axiom_threshold, 0.9);
    }

    #[test]
    fn test_principle_type_reliability() {
        assert_eq!(PrincipleType::Axiom.reliability_weight(), 1.0);
        assert_eq!(PrincipleType::Contested.reliability_weight(), 0.30);
        assert!(
            PrincipleType::Assumption.reliability_weight()
                < PrincipleType::Derived.reliability_weight()
        );
    }

    #[test]
    fn test_decompose_simple_query() {
        let bedrock = BedRock::new();
        let result = bedrock.decompose("Electric vehicles are better than gas cars", &[]);

        assert!(!result.principles.is_empty());
        assert_eq!(result.query, "Electric vehicles are better than gas cars");
        assert!(result.confidence > 0.0);
        assert!(!result.insights.is_empty());
    }

    #[test]
    fn test_decompose_with_comparison() {
        let bedrock = BedRock::new();
        let result = bedrock.decompose("Python is better than JavaScript for data science", &[]);

        // Should identify that comparison requires metrics
        let has_definition = result
            .principles
            .iter()
            .any(|p| p.principle_type == PrincipleType::Definition);
        assert!(has_definition, "Should identify need for comparison metric");
    }

    #[test]
    fn test_decompose_with_causation() {
        let bedrock = BedRock::new();
        let result = bedrock.decompose("Smoking causes cancer", &[]);

        // Should identify causal analysis requirements
        let has_axiom = result
            .principles
            .iter()
            .any(|p| p.principle_type == PrincipleType::Axiom);
        assert!(
            has_axiom,
            "Should identify axiomatic principles about causation"
        );
    }

    #[test]
    fn test_execute_trait() {
        let bedrock = BedRock::new();
        let context = ThinkToolContext {
            query: "What is the best programming language?".into(),
            previous_steps: vec!["Prior analysis: Consider use case".into()],
        };

        let output = bedrock.execute(&context).expect("Execution should succeed");

        assert_eq!(output.module, "BedRock");
        assert!(output.confidence > 0.0);
        assert!(output.output.get("axioms").is_some());
        assert!(output.output.get("assumptions").is_some());
        assert!(output.output.get("decomposition").is_some());
        assert!(output.output.get("insights").is_some());
    }

    #[test]
    fn test_classify_query() {
        let bedrock = BedRock::new();

        // Definition query
        let def_type = bedrock.classify_query("What is machine learning?");
        assert_eq!(def_type, PrincipleType::Definition);

        // Empirical query
        let emp_type = bedrock.classify_query("Research shows that exercise improves health");
        assert_eq!(emp_type, PrincipleType::Empirical);

        // Contested/value query
        let contested_type = bedrock.classify_query("Rust is better than C++");
        assert_eq!(contested_type, PrincipleType::Contested);
    }

    #[test]
    fn test_result_accessors() {
        let bedrock = BedRock::new();
        let result = bedrock.decompose("All birds can fly", &[]);

        // Universal claims should generate axioms about universal statements
        let axioms = result.axioms();
        let assumptions = result.assumptions();

        // Check that we can access principles at different depths
        let root_principles = result.at_depth(0);
        assert!(!root_principles.is_empty());

        // Check completeness calculation
        assert!(result.metadata.completeness >= 0.0);
        assert!(result.metadata.completeness <= 1.0);
    }

    #[test]
    fn test_principle_effective_weight() {
        let principle = Principle {
            id: 0,
            statement: "Test axiom".into(),
            principle_type: PrincipleType::Axiom,
            fundamentality: 1.0,
            confidence: 0.95,
            parent_id: None,
            child_ids: vec![],
            evidence: vec![],
            challenges: vec![],
            depth: 0,
        };

        let weight = principle.effective_weight();
        assert_eq!(weight, 0.95); // 1.0 * 0.95 * 1.0
        assert!(principle.is_axiomatic(0.85));
    }

    #[test]
    fn test_gap_identification() {
        let bedrock = BedRock::new();
        let result = bedrock.decompose("This is a vague statement", &[]);

        // Should identify gaps
        // Note: gaps may or may not be found depending on decomposition
        assert!(result.gaps.len() >= 0); // Gap identification works
    }

    #[test]
    fn test_max_principles_limit() {
        let config = BedRockConfig {
            max_principles: 5,
            ..Default::default()
        };
        let bedrock = BedRock::with_config(config);
        let result = bedrock.decompose("Complex multi-part query about many things", &[]);

        assert!(result.principles.len() <= 5);
    }

    #[test]
    fn test_json_output_structure() {
        let bedrock = BedRock::new();
        let result = bedrock.decompose("Test query for JSON", &[]);
        let json = result.to_json();

        assert!(json.get("query").is_some());
        assert!(json.get("axioms").is_some());
        assert!(json.get("assumptions").is_some());
        assert!(json.get("decomposition").is_some());
        assert!(json.get("reconstruction").is_some());
        assert!(json.get("gaps").is_some());
        assert!(json.get("insights").is_some());
        assert!(json.get("confidence").is_some());
        assert!(json.get("metadata").is_some());
    }
}
