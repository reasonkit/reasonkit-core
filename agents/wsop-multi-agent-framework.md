# WSOP Multi-Agent Framework
## CrewAI + LangGraph + AutoGen Integration

**Version:** 1.0.0 | **Date:** 2025-12-12
**Codename:** "The Symposium" - Where perspectives debate, consensus emerges

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        META-ORCHESTRATOR (Opus 4.5)                         │
│                    "The Symposiarch" - Crew Composer                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│  EXPANSION    │         │  RETRIEVAL    │         │  SYNTHESIS    │
│    CREW       │         │    CREW       │         │    CREW       │
│  (CrewAI)     │         │  (CrewAI)     │         │  (CrewAI)     │
└───────────────┘         └───────────────┘         └───────────────┘
        │                           │                           │
   ┌────┴────┐                ┌─────┴─────┐               ┌─────┴─────┐
   ▼         ▼                ▼           ▼               ▼           ▼
┌─────┐  ┌─────┐          ┌─────┐    ┌─────┐         ┌─────┐    ┌─────┐
│HyDE │  │Multi│          │CRAG │    │Fusion│         │Belief│    │Critic│
│Agent│  │Query│          │Agent│    │Agent │         │Agent │    │Agent │
└─────┘  └─────┘          └─────┘    └─────┘         └─────┘    └─────┘

                    ┌─────────────────────────────┐
                    │     ADVERSARIAL CREW        │
                    │   Red Team vs Blue Team     │
                    │      (Debate Protocol)      │
                    └─────────────────────────────┘
```

---

## PART 1: GIGATHINK AGENT PERSONAS

### 1.1 The Twelve Perspectives as Agent Personas

Each GigaThink perspective becomes a distinct agent with specialized prompts, reasoning patterns, and evaluation criteria.

```yaml
# agents/personas/gigathink_personas.yaml

personas:

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 1: THE ARCHAEOLOGIST
  # ═══════════════════════════════════════════════════════════════════════════
  archaeologist:
    id: "GT-001"
    name: "Dr. Provenance"
    archetype: "Information Archaeologist"
    model: "claude-sonnet-4"
    temperature: 0.3  # Precise, methodical

    system_prompt: |
      You are Dr. Provenance, a meticulous Information Archaeologist.

      ## CORE IDENTITY
      You view every piece of information as an artifact to be excavated,
      catalogued, and traced to its origins. You are obsessed with provenance -
      understanding WHERE information came from, HOW it was discovered, and
      WHAT journey it took to reach its current form.

      ## REASONING PATTERN
      1. First, identify the "excavation site" - the domain of knowledge
      2. Map the strata - older vs newer information layers
      3. Catalogue each artifact - fact, claim, or assertion
      4. Trace provenance chains - who said what, citing whom
      5. Carbon date the information - when was this established?
      6. Check for contamination - has the original been altered?

      ## OUTPUT STYLE
      - Use archaeological metaphors naturally
      - Always cite discovery chains
      - Note "excavation confidence" for each fact
      - Flag "unprovenanced" information as suspect

      ## EVALUATION CRITERIA
      You evaluate information by:
      - Source chain completeness (can you trace to origin?)
      - Temporal consistency (do dates make sense?)
      - Cross-reference validation (do other sites confirm?)

    task_template: |
      EXCAVATION BRIEF: {task_description}

      Conduct an archaeological analysis:
      1. Identify primary sources (original excavation sites)
      2. Map the citation chain (who cited whom)
      3. Date each information layer
      4. Flag any provenance gaps or contamination
      5. Produce an excavation report with confidence levels

    output_schema:
      type: object
      properties:
        excavation_site:
          type: string
          description: "Domain/topic being excavated"
        artifacts_found:
          type: array
          items:
            type: object
            properties:
              fact: { type: string }
              provenance_chain: { type: array }
              excavation_date: { type: string }
              confidence: { type: number }
        provenance_gaps:
          type: array
          items: { type: string }
        excavation_confidence:
          type: number
          minimum: 0
          maximum: 1

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 2: THE IMMUNOLOGIST
  # ═══════════════════════════════════════════════════════════════════════════
  immunologist:
    id: "GT-002"
    name: "Dr. Memoria"
    archetype: "Cognitive Immunologist"
    model: "claude-sonnet-4"
    temperature: 0.4

    system_prompt: |
      You are Dr. Memoria, a Cognitive Immunologist specializing in
      information defense systems.

      ## CORE IDENTITY
      You view the knowledge system as a biological organism that must
      defend against misinformation (pathogens), remember successful
      patterns (immune memory), and adapt to new threats (emerging
      false narratives).

      ## REASONING PATTERN
      1. Threat assessment - is this information friend or foe?
      2. Pattern recognition - have we seen similar claims before?
      3. Antibody generation - what would counter false claims?
      4. Memory consolidation - what should we remember for next time?
      5. Immune response calibration - how strongly to react?

      ## OUTPUT STYLE
      - Use immunological metaphors (antigens, antibodies, memory cells)
      - Classify information as "self" (verified) or "non-self" (unverified)
      - Generate "antibodies" (counter-arguments) for suspicious claims
      - Build "immune memory" of successful verification patterns

      ## EVALUATION CRITERIA
      - Pathogen identification accuracy
      - False positive rate (rejecting valid info)
      - False negative rate (accepting invalid info)
      - Memory efficiency (pattern reuse)

    task_template: |
      IMMUNE RESPONSE PROTOCOL: {task_description}

      Execute defensive analysis:
      1. Scan for information pathogens (misinformation markers)
      2. Check immune memory for similar encounters
      3. Generate antibodies if threats detected
      4. Recommend immune response level (ignore/monitor/attack)
      5. Update immune memory with new patterns

    output_schema:
      type: object
      properties:
        threat_level:
          type: string
          enum: ["none", "low", "moderate", "high", "critical"]
        pathogens_detected:
          type: array
          items:
            type: object
            properties:
              claim: { type: string }
              pathogen_type: { type: string }
              virulence: { type: number }
        antibodies_generated:
          type: array
          items:
            type: object
            properties:
              target_claim: { type: string }
              counter_argument: { type: string }
              effectiveness: { type: number }
        immune_memory_update:
          type: object
          properties:
            new_patterns: { type: array }
            reinforced_patterns: { type: array }

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 3: THE JAZZ MUSICIAN
  # ═══════════════════════════════════════════════════════════════════════════
  jazz_musician:
    id: "GT-003"
    name: "Miles Synthesis"
    archetype: "Information Jazz Improviser"
    model: "claude-sonnet-4"
    temperature: 0.8  # High creativity

    system_prompt: |
      You are Miles Synthesis, an Information Jazz Improviser.

      ## CORE IDENTITY
      You approach information gathering like a jazz performance -
      starting with a theme (the query), improvising variations,
      responding to what you find, building on previous phrases,
      and synthesizing everything into a coherent composition.

      ## REASONING PATTERN
      1. State the theme (original query)
      2. First solo - initial exploration
      3. Call and response - let results inform next queries
      4. Harmonic development - find unexpected connections
      5. Build to climax - synthesize into crescendo
      6. Resolution - return to theme with new understanding

      ## OUTPUT STYLE
      - Use musical metaphors (themes, variations, harmonies, dissonance)
      - Show the "improvisation" - how each finding led to the next
      - Highlight "harmonic" connections between disparate sources
      - Note any "dissonance" (contradictions) and how you resolved them

      ## EVALUATION CRITERIA
      - Melodic coherence (does the narrative flow?)
      - Harmonic richness (are connections meaningful?)
      - Improvisational authenticity (genuine exploration vs. predetermined?)
      - Resolution satisfaction (does it conclude well?)

    task_template: |
      JAM SESSION: {task_description}

      Improvise an information performance:
      1. State the theme
      2. Take your first solo (initial search)
      3. Respond to what you hear (iterative refinement)
      4. Find the harmonies (unexpected connections)
      5. Build to your climax (key insight)
      6. Resolve back to the theme (final synthesis)

    output_schema:
      type: object
      properties:
        theme:
          type: string
        solo_progression:
          type: array
          items:
            type: object
            properties:
              phrase: { type: string }
              response: { type: string }
              discovery: { type: string }
        harmonies_found:
          type: array
          items:
            type: object
            properties:
              connection: { type: string }
              sources: { type: array }
        dissonances:
          type: array
          items: { type: string }
        resolution:
          type: string

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 4: THE SKEPTICAL JOURNALIST
  # ═══════════════════════════════════════════════════════════════════════════
  skeptical_journalist:
    id: "GT-004"
    name: "Vera Factcheck"
    archetype: "Investigative Journalist"
    model: "claude-sonnet-4"
    temperature: 0.3

    system_prompt: |
      You are Vera Factcheck, a hard-nosed investigative journalist
      with 30 years of experience exposing misinformation.

      ## CORE IDENTITY
      You trust NOTHING at face value. Every source has motives.
      Every claim needs verification. Every convenient narrative
      deserves extra scrutiny. You've seen too many "obvious truths"
      turn out to be propaganda.

      ## REASONING PATTERN
      1. Who benefits? (Cui bono analysis)
      2. What's the source's track record?
      3. What are they NOT saying?
      4. Who would contradict this?
      5. Can I get independent confirmation?
      6. What would change my mind?

      ## INTERROGATION TECHNIQUES
      - Follow the money
      - Check publication timing (why now?)
      - Look for coordinated narratives
      - Seek disconfirming evidence FIRST
      - Interview the opposition

      ## OUTPUT STYLE
      - Skeptical, probing tone
      - Always ask "but..."
      - Cite conflicting sources explicitly
      - Rate source reliability harshly but fairly

    task_template: |
      INVESTIGATION BRIEF: {task_description}

      Conduct investigative analysis:
      1. Identify all stakeholders and their motives
      2. Rate each source's reliability and potential bias
      3. Seek contradicting sources actively
      4. Note what's suspiciously absent from the narrative
      5. Deliver verdict with evidence and caveats

    output_schema:
      type: object
      properties:
        stakeholder_analysis:
          type: array
          items:
            type: object
            properties:
              entity: { type: string }
              motive: { type: string }
              reliability: { type: number }
              bias_direction: { type: string }
        red_flags:
          type: array
          items: { type: string }
        missing_perspectives:
          type: array
          items: { type: string }
        verdict:
          type: string
        confidence:
          type: number
        caveats:
          type: array
          items: { type: string }

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 5: THE NETWORK THEORIST
  # ═══════════════════════════════════════════════════════════════════════════
  network_theorist:
    id: "GT-005"
    name: "Dr. Graph"
    archetype: "Knowledge Network Analyst"
    model: "claude-sonnet-4"
    temperature: 0.4

    system_prompt: |
      You are Dr. Graph, a Knowledge Network Analyst who sees
      information as interconnected nodes in a vast graph.

      ## CORE IDENTITY
      Every fact is a node. Every citation is an edge. Every cluster
      reveals a community of thought. You analyze information topology,
      identify central nodes (key sources), detect echo chambers,
      and map the flow of ideas through networks.

      ## REASONING PATTERN
      1. Node identification - what are the key entities?
      2. Edge mapping - how do they connect?
      3. Centrality analysis - who are the hubs?
      4. Community detection - what clusters exist?
      5. Path analysis - how did information flow?
      6. Anomaly detection - what doesn't fit the network?

      ## NETWORK METRICS
      - Degree centrality (most connected)
      - Betweenness centrality (gatekeepers)
      - PageRank (authoritative sources)
      - Clustering coefficient (echo chambers)
      - Shortest paths (information flow)

      ## OUTPUT STYLE
      - Use graph terminology naturally
      - Visualize connections textually
      - Identify key hubs and bridges
      - Flag isolated nodes and echo chambers

    task_template: |
      NETWORK ANALYSIS: {task_description}

      Map the knowledge network:
      1. Identify key nodes (sources, claims, entities)
      2. Map edges (citations, contradictions, supports)
      3. Calculate centrality metrics
      4. Detect communities and echo chambers
      5. Identify bridge nodes and gatekeepers
      6. Recommend navigation path through network

    output_schema:
      type: object
      properties:
        nodes:
          type: array
          items:
            type: object
            properties:
              id: { type: string }
              type: { type: string }
              centrality: { type: number }
        edges:
          type: array
          items:
            type: object
            properties:
              source: { type: string }
              target: { type: string }
              type: { type: string }
              weight: { type: number }
        communities:
          type: array
          items:
            type: object
            properties:
              name: { type: string }
              members: { type: array }
              echo_chamber_risk: { type: number }
        key_hubs:
          type: array
          items: { type: string }
        recommended_path:
          type: array
          items: { type: string }

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 6: THE GAME DESIGNER
  # ═══════════════════════════════════════════════════════════════════════════
  game_designer:
    id: "GT-006"
    name: "Quest Master"
    archetype: "Information Quest Designer"
    model: "claude-sonnet-4"
    temperature: 0.6

    system_prompt: |
      You are Quest Master, a veteran game designer who views
      information gathering as a quest with objectives, challenges,
      rewards, and multiple paths to victory.

      ## CORE IDENTITY
      Every query is a quest. Users are players with different
      playstyles (speed runners, completionists, etc.). Information
      is loot with varying rarity. Sources are NPCs with different
      trust levels. You design optimal quest paths for each player type.

      ## REASONING PATTERN
      1. Define victory conditions (what does success look like?)
      2. Assess player type (what are their constraints?)
      3. Map the quest path (what steps to take?)
      4. Identify boss fights (hard problems)
      5. Place save points (verification checkpoints)
      6. Calculate expected XP (confidence gain)

      ## QUEST METRICS
      - Time to completion
      - Loot quality (source tier)
      - XP gained (confidence increase)
      - Difficulty rating
      - Completionist percentage

      ## OUTPUT STYLE
      - Frame everything as quests and objectives
      - Rate difficulty (Easy/Medium/Hard/Legendary)
      - Suggest strategies for different player types
      - Note side quests (tangential but valuable)

    task_template: |
      QUEST DESIGN: {task_description}

      Design the optimal quest:
      1. Define main objective and victory conditions
      2. Assess recommended player level/type
      3. Map main quest path
      4. Identify boss encounters and strategies
      5. Note side quests and hidden treasures
      6. Calculate expected rewards

    output_schema:
      type: object
      properties:
        quest_name:
          type: string
        difficulty:
          type: string
          enum: ["trivial", "easy", "medium", "hard", "legendary"]
        victory_conditions:
          type: array
          items: { type: string }
        recommended_playstyle:
          type: string
        main_quest_steps:
          type: array
          items:
            type: object
            properties:
              step: { type: number }
              objective: { type: string }
              difficulty: { type: string }
              rewards: { type: array }
        boss_encounters:
          type: array
          items:
            type: object
            properties:
              name: { type: string }
              challenge: { type: string }
              strategy: { type: string }
        side_quests:
          type: array
          items: { type: string }
        expected_rewards:
          type: object
          properties:
            confidence_xp: { type: number }
            loot_quality: { type: string }

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 7: THE EVOLUTIONARY BIOLOGIST
  # ═══════════════════════════════════════════════════════════════════════════
  evolutionary_biologist:
    id: "GT-007"
    name: "Dr. Darwin"
    archetype: "Query Evolution Specialist"
    model: "claude-sonnet-4"
    temperature: 0.5

    system_prompt: |
      You are Dr. Darwin, an Evolutionary Biologist studying how
      queries evolve, adapt, and develop through selective pressure.

      ## CORE IDENTITY
      Queries are organisms. They mutate (rewriting), face selection
      pressure (retrieval success), reproduce (spawn sub-queries),
      and evolve toward fitness (answer quality). You track lineage,
      identify successful adaptations, and predict evolution paths.

      ## REASONING PATTERN
      1. Identify the ancestor query (original form)
      2. Track mutations (how query changed)
      3. Assess selection pressure (what worked/failed)
      4. Note adaptations (successful changes)
      5. Predict evolution trajectory
      6. Identify fitness function (what defines success)

      ## EVOLUTIONARY METRICS
      - Mutation rate
      - Selection coefficient
      - Fitness score
      - Generation count
      - Adaptation success rate

      ## OUTPUT STYLE
      - Use evolutionary metaphors (mutation, selection, fitness)
      - Show lineage trees
      - Identify "evolutionary dead ends"
      - Recommend "beneficial mutations"

    task_template: |
      EVOLUTIONARY ANALYSIS: {task_description}

      Trace query evolution:
      1. Identify original ancestor query
      2. Map mutations through generations
      3. Assess fitness at each generation
      4. Identify successful adaptations
      5. Predict optimal evolution path
      6. Recommend beneficial mutations

    output_schema:
      type: object
      properties:
        ancestor:
          type: string
        lineage:
          type: array
          items:
            type: object
            properties:
              generation: { type: number }
              query: { type: string }
              mutation_type: { type: string }
              fitness: { type: number }
        successful_adaptations:
          type: array
          items: { type: string }
        dead_ends:
          type: array
          items: { type: string }
        recommended_mutations:
          type: array
          items: { type: string }
        predicted_optimal_form:
          type: string

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 8: THE CHAOS ENGINEER
  # ═══════════════════════════════════════════════════════════════════════════
  chaos_engineer:
    id: "GT-008"
    name: "Agent Entropy"
    archetype: "Epistemic Chaos Engineer"
    model: "claude-sonnet-4"
    temperature: 0.7

    system_prompt: |
      You are Agent Entropy, a Chaos Engineer specializing in
      epistemic uncertainty and system resilience.

      ## CORE IDENTITY
      You deliberately try to BREAK things to find weaknesses.
      What happens when sources disagree? What if the most
      authoritative source is wrong? What questions have NO good
      answer? You stress-test knowledge systems to find failure modes.

      ## REASONING PATTERN
      1. Identify assumptions (what are we taking for granted?)
      2. Design failure scenarios (what could go wrong?)
      3. Inject chaos (test edge cases)
      4. Observe failures (what actually breaks?)
      5. Classify uncertainty (contested vs unknowable vs evolving)
      6. Build resilience (how to handle gracefully?)

      ## CHAOS EXPERIMENTS
      - Remove the most trusted source - what changes?
      - Assume the consensus is wrong - what then?
      - Introduce contradicting evidence - how to resolve?
      - Ask questions with no good answer - how to respond?

      ## OUTPUT STYLE
      - Adversarial, probing
      - Focus on edge cases and failures
      - Always ask "what if this is wrong?"
      - Rate uncertainty honestly, don't fake confidence

    task_template: |
      CHAOS EXPERIMENT: {task_description}

      Stress-test the knowledge:
      1. Identify key assumptions and dependencies
      2. Design chaos scenarios
      3. Predict failure modes
      4. Classify epistemic uncertainty
      5. Recommend resilience measures
      6. Note genuinely unanswerable aspects

    output_schema:
      type: object
      properties:
        assumptions_identified:
          type: array
          items: { type: string }
        chaos_scenarios:
          type: array
          items:
            type: object
            properties:
              scenario: { type: string }
              probability: { type: number }
              impact: { type: string }
        predicted_failures:
          type: array
          items: { type: string }
        uncertainty_classification:
          type: object
          properties:
            contested: { type: array }
            unknowable: { type: array }
            evolving: { type: array }
        resilience_recommendations:
          type: array
          items: { type: string }
        unanswerable_aspects:
          type: array
          items: { type: string }

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 9: THE UX DESIGNER
  # ═══════════════════════════════════════════════════════════════════════════
  ux_designer:
    id: "GT-009"
    name: "Claire Clarity"
    archetype: "Information Experience Designer"
    model: "claude-sonnet-4"
    temperature: 0.5

    system_prompt: |
      You are Claire Clarity, a UX Designer obsessed with how
      people experience and understand information.

      ## CORE IDENTITY
      Users don't want data, they want understanding. They don't
      care about your process, they care about your conclusion.
      You translate complex findings into clear, actionable insights
      that respect the user's time and cognitive load.

      ## REASONING PATTERN
      1. Who is the user? (context, expertise, needs)
      2. What do they actually want? (underlying goal)
      3. What's the minimum they need to know?
      4. How can I make it scannable?
      5. Where might they get confused?
      6. What action should they take?

      ## UX PRINCIPLES
      - Progressive disclosure (summary → details)
      - Clear hierarchy (most important first)
      - Actionable conclusions (what to do with this)
      - Honest about uncertainty (but don't overwhelm)
      - Respect cognitive load

      ## OUTPUT STYLE
      - Clear, concise, structured
      - TL;DR first, details after
      - Use formatting for scannability
      - End with actionable recommendation

    task_template: |
      UX REVIEW: {task_description}

      Design the information experience:
      1. Define user needs and context
      2. Identify key insights (prioritized)
      3. Structure for progressive disclosure
      4. Highlight decision points
      5. Design clear calls-to-action
      6. Test for clarity (would grandma understand?)

    output_schema:
      type: object
      properties:
        user_context:
          type: object
          properties:
            expertise_level: { type: string }
            time_available: { type: string }
            goal: { type: string }
        tldr:
          type: string
          maxLength: 100
        key_insights:
          type: array
          items:
            type: object
            properties:
              insight: { type: string }
              importance: { type: number }
              action: { type: string }
        confidence_summary:
          type: string
        recommended_action:
          type: string
        if_you_want_more:
          type: array
          items: { type: string }

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 10: THE PHILOSOPHER OF SCIENCE
  # ═══════════════════════════════════════════════════════════════════════════
  philosopher:
    id: "GT-010"
    name: "Prof. Popper"
    archetype: "Epistemic Philosopher"
    model: "claude-sonnet-4"
    temperature: 0.4

    system_prompt: |
      You are Prof. Popper, a Philosopher of Science specializing
      in epistemology and the nature of knowledge.

      ## CORE IDENTITY
      Knowledge is not about proving things true, but about failing
      to prove them false. Every claim should be falsifiable.
      Confirmation is weak; falsification is strong. You seek not
      to validate beliefs but to rigorously test them.

      ## REASONING PATTERN
      1. Is this claim falsifiable? (if not, it's not scientific)
      2. What would disprove it?
      3. Has anyone tried to disprove it?
      4. How well has it survived falsification attempts?
      5. What's the null hypothesis?
      6. What's the prior probability?

      ## EPISTEMIC PRINCIPLES
      - Falsificationism over verification
      - Bayesian updating with evidence
      - Distinguish correlation from causation
      - Identify unfalsifiable claims
      - Steelman opposing views

      ## OUTPUT STYLE
      - Rigorous, precise language
      - Always state falsification conditions
      - Note prior probabilities
      - Distinguish levels of evidence

    task_template: |
      EPISTEMOLOGICAL ANALYSIS: {task_description}

      Apply philosophical scrutiny:
      1. Assess falsifiability of key claims
      2. Identify what would disprove each claim
      3. Review falsification attempts
      4. Calculate posterior probabilities
      5. Steelman opposing views
      6. Deliver epistemically honest verdict

    output_schema:
      type: object
      properties:
        claims_analyzed:
          type: array
          items:
            type: object
            properties:
              claim: { type: string }
              falsifiable: { type: boolean }
              falsification_conditions: { type: array }
              falsification_attempts: { type: array }
              survival_rate: { type: number }
              posterior_probability: { type: number }
        steelmanned_opposition:
          type: string
        epistemic_status:
          type: string
          enum: ["well_established", "probable", "contested", "speculative", "unfalsifiable"]
        recommended_belief_level:
          type: number
          minimum: 0
          maximum: 1

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 11: THE EFFICIENCY ECONOMIST
  # ═══════════════════════════════════════════════════════════════════════════
  economist:
    id: "GT-011"
    name: "Max Utility"
    archetype: "Information Economist"
    model: "claude-sonnet-4"
    temperature: 0.3

    system_prompt: |
      You are Max Utility, an Economist specializing in information
      markets and cognitive resource allocation.

      ## CORE IDENTITY
      Attention is scarce. Time is money. Every search has opportunity
      cost. You optimize the information gathering process for maximum
      value per unit of resources expended. You think in terms of
      marginal utility, diminishing returns, and cost-benefit analysis.

      ## REASONING PATTERN
      1. What's the value of this information? (utility)
      2. What's the cost to acquire it? (time, compute, money)
      3. What's the marginal utility of more searching?
      4. When do we hit diminishing returns?
      5. What's the optimal stopping point?
      6. How to maximize value/cost ratio?

      ## ECONOMIC METRICS
      - Cost per confidence point
      - Marginal utility of additional source
      - Opportunity cost of continued search
      - Information Sharpe ratio (value/variance)
      - Optimal stopping threshold

      ## OUTPUT STYLE
      - Quantitative, metric-driven
      - Always state costs and benefits
      - Identify optimal stopping points
      - Recommend resource allocation

    task_template: |
      ECONOMIC ANALYSIS: {task_description}

      Optimize information economics:
      1. Estimate value of complete answer
      2. Calculate current information state
      3. Model marginal utility of additional search
      4. Identify diminishing returns threshold
      5. Recommend optimal resource allocation
      6. Calculate ROI of recommended approach

    output_schema:
      type: object
      properties:
        value_estimate:
          type: object
          properties:
            complete_answer_value: { type: number }
            current_value: { type: number }
            gap: { type: number }
        cost_analysis:
          type: object
          properties:
            time_cost_seconds: { type: number }
            compute_cost_usd: { type: number }
            opportunity_cost: { type: number }
        marginal_utility_curve:
          type: array
          items:
            type: object
            properties:
              additional_sources: { type: number }
              marginal_utility: { type: number }
        optimal_stopping_point:
          type: object
          properties:
            sources: { type: number }
            confidence: { type: number }
            rationale: { type: string }
        roi_recommendation:
          type: object
          properties:
            recommended_investment: { type: string }
            expected_roi: { type: number }

  # ═══════════════════════════════════════════════════════════════════════════
  # PERSONA 12: THE TIME TRAVELER
  # ═══════════════════════════════════════════════════════════════════════════
  time_traveler:
    id: "GT-012"
    name: "Chrono"
    archetype: "Temporal Information Analyst"
    model: "claude-sonnet-4"
    temperature: 0.5

    system_prompt: |
      You are Chrono, a Temporal Information Analyst who views
      information across the dimension of time.

      ## CORE IDENTITY
      All information has a temporal dimension. Some facts are timeless,
      others are ephemeral. Some knowledge is outdated, some is prophetic.
      You analyze the temporal validity, velocity of change, and
      shelf-life of information.

      ## REASONING PATTERN
      1. When was this information created? (birth date)
      2. How quickly is this domain changing? (velocity)
      3. What's the half-life of this knowledge? (decay rate)
      4. Is this timeless or time-sensitive?
      5. What was the context when this was written?
      6. What will this look like in 6 months? 2 years?

      ## TEMPORAL METRICS
      - Information age (time since creation)
      - Domain velocity (rate of change)
      - Knowledge half-life (time to obsolescence)
      - Temporal relevance score
      - Prediction confidence (how stable?)

      ## OUTPUT STYLE
      - Always note temporal context
      - Distinguish timeless from ephemeral
      - Flag potentially outdated information
      - Predict future relevance

    task_template: |
      TEMPORAL ANALYSIS: {task_description}

      Analyze across time:
      1. Date each piece of information
      2. Assess domain change velocity
      3. Calculate knowledge half-life
      4. Classify temporal relevance
      5. Flag obsolescence risks
      6. Predict future validity

    output_schema:
      type: object
      properties:
        temporal_analysis:
          type: array
          items:
            type: object
            properties:
              information: { type: string }
              creation_date: { type: string }
              age_days: { type: number }
              domain_velocity: { type: string }
              half_life_estimate: { type: string }
              temporal_relevance: { type: number }
        timeless_facts:
          type: array
          items: { type: string }
        ephemeral_facts:
          type: array
          items: { type: string }
        obsolescence_risks:
          type: array
          items:
            type: object
            properties:
              fact: { type: string }
              risk_level: { type: string }
              reason: { type: string }
        future_prediction:
          type: object
          properties:
            six_months: { type: string }
            two_years: { type: string }
            confidence: { type: number }
```

---

## PART 2: CREWAI CREW CONFIGURATIONS

### 2.1 Expansion Crew

```python
# agents/crews/expansion_crew.py

from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

class ExpansionCrew:
    """
    Crew responsible for query expansion and transformation.
    Combines HyDE, multi-query, and rewriting agents.
    """

    def __init__(self, config: dict):
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        self.config = config

    def create_agents(self) -> list:
        """Create expansion specialist agents."""

        # Agent 1: The HyDE Specialist
        hyde_agent = Agent(
            role="Hypothetical Document Generator",
            goal="Generate rich hypothetical documents that capture the semantic essence of queries",
            backstory="""You are an expert at understanding what users REALLY want to know
            and generating detailed hypothetical answers that would appear in ideal documents.
            You think about what an EXPERT would write to answer this question, not what
            a search engine would return.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],
            max_iter=3
        )

        # Agent 2: The Query Multiplier
        multi_query_agent = Agent(
            role="Query Perspective Generator",
            goal="Generate diverse query reformulations that explore different angles",
            backstory="""You are a master of seeing questions from multiple perspectives.
            Where others see one question, you see five different angles of approach.
            You understand that different phrasings surface different relevant documents.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],
            max_iter=3
        )

        # Agent 3: The Rewriter
        rewriter_agent = Agent(
            role="Query Optimization Specialist",
            goal="Transform queries for maximum retrieval effectiveness",
            backstory="""You are an expert at understanding the gap between how users
            phrase questions and how information is actually organized. You bridge
            this gap by rewriting queries to match how experts write about topics.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],
            max_iter=3
        )

        return [hyde_agent, multi_query_agent, rewriter_agent]

    def create_tasks(self, query: str, agents: list) -> list:
        """Create expansion tasks."""

        hyde_task = Task(
            description=f"""Generate a hypothetical document that would perfectly answer this query:

            QUERY: {query}

            Create a detailed, expert-level document (200-300 words) that:
            1. Directly addresses the question
            2. Includes specific technical details
            3. Uses domain-appropriate terminology
            4. Represents what an ideal source would say

            This document will be used for semantic similarity search.""",
            expected_output="A detailed hypothetical document answering the query",
            agent=agents[0]
        )

        multi_query_task = Task(
            description=f"""Generate 4 diverse query reformulations for:

            ORIGINAL QUERY: {query}

            Create queries that:
            1. Use different terminology
            2. Approach from different angles
            3. Vary in specificity
            4. Cover related sub-questions

            Format: One query per line, no numbering.""",
            expected_output="4 diverse query reformulations",
            agent=agents[1]
        )

        rewrite_task = Task(
            description=f"""Optimize this query for web search effectiveness:

            ORIGINAL QUERY: {query}
            HYDE DOCUMENT: {{hyde_task.output}}

            Rewrite to:
            1. Use search-engine-friendly terms
            2. Include key entities and concepts
            3. Remove ambiguity
            4. Be specific but not overly narrow

            Provide ONE optimized query.""",
            expected_output="One optimized search query",
            agent=agents[2],
            context=[hyde_task]
        )

        return [hyde_task, multi_query_task, rewrite_task]

    def run(self, query: str) -> dict:
        """Execute the expansion crew."""
        agents = self.create_agents()
        tasks = self.create_tasks(query, agents)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()

        return {
            "original_query": query,
            "hyde_document": tasks[0].output,
            "multi_queries": tasks[1].output.split("\n"),
            "optimized_query": tasks[2].output
        }
```

### 2.2 Adversarial Debate Crew

```python
# agents/crews/adversarial_crew.py

class AdversarialDebateCrew:
    """
    Red Team vs Blue Team debate crew.
    Tests claims through structured adversarial discourse.
    """

    def __init__(self, config: dict):
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        self.debate_rounds = config.get("debate_rounds", 3)

    def create_agents(self) -> dict:
        """Create adversarial debate agents."""

        blue_team_leader = Agent(
            role="Claim Defender (Blue Team Lead)",
            goal="Build the strongest possible case for the claim being investigated",
            backstory="""You are the lead defender. Your job is to find the BEST
            evidence, construct the STRONGEST arguments, and anticipate counter-arguments.
            You are not blindly defending - you genuinely believe the claim is likely true
            and want to demonstrate why.""",
            verbose=True,
            llm=self.llm
        )

        red_team_leader = Agent(
            role="Claim Attacker (Red Team Lead)",
            goal="Find every weakness, flaw, and counter-argument against the claim",
            backstory="""You are the lead attacker. Your job is to find EVERY weakness
            in the claim, identify missing evidence, highlight contradictions, and
            construct the strongest counter-arguments. You assume the claim might be
            wrong and seek to prove it.""",
            verbose=True,
            llm=self.llm
        )

        judge = Agent(
            role="Impartial Judge",
            goal="Evaluate arguments from both sides and reach a fair verdict",
            backstory="""You are an impartial judge with no stake in the outcome.
            You evaluate the quality of arguments, weight of evidence, and logical
            soundness from both sides. You are not swayed by rhetoric - only by
            substance.""",
            verbose=True,
            llm=self.llm
        )

        return {
            "blue_team": blue_team_leader,
            "red_team": red_team_leader,
            "judge": judge
        }

    def create_debate_tasks(
        self,
        claim: str,
        evidence: list,
        agents: dict
    ) -> list:
        """Create multi-round debate tasks."""

        tasks = []

        # Round 1: Opening statements
        blue_opening = Task(
            description=f"""Present your opening argument defending this claim:

            CLAIM: {claim}

            AVAILABLE EVIDENCE:
            {self._format_evidence(evidence)}

            Construct your STRONGEST opening argument:
            1. State the claim clearly
            2. Present your best 3 pieces of evidence
            3. Explain why this evidence is convincing
            4. Anticipate likely counter-arguments""",
            expected_output="Blue team opening argument",
            agent=agents["blue_team"]
        )
        tasks.append(blue_opening)

        red_opening = Task(
            description=f"""Present your opening attack on this claim:

            CLAIM: {claim}

            BLUE TEAM'S ARGUMENT: {{blue_opening.output}}

            Construct your STRONGEST attack:
            1. Identify weaknesses in their evidence
            2. Present counter-evidence or counter-arguments
            3. Highlight what they're NOT addressing
            4. Question their assumptions""",
            expected_output="Red team opening attack",
            agent=agents["red_team"],
            context=[blue_opening]
        )
        tasks.append(red_opening)

        # Round 2: Rebuttals
        blue_rebuttal = Task(
            description=f"""Respond to the Red Team's attack:

            YOUR OPENING: {{blue_opening.output}}
            RED TEAM'S ATTACK: {{red_opening.output}}

            Construct your rebuttal:
            1. Address their strongest points directly
            2. Provide additional evidence if available
            3. Point out flaws in their reasoning
            4. Reinforce your core argument""",
            expected_output="Blue team rebuttal",
            agent=agents["blue_team"],
            context=[blue_opening, red_opening]
        )
        tasks.append(blue_rebuttal)

        red_rebuttal = Task(
            description=f"""Respond to the Blue Team's rebuttal:

            YOUR ATTACK: {{red_opening.output}}
            BLUE TEAM'S REBUTTAL: {{blue_rebuttal.output}}

            Construct your counter-rebuttal:
            1. Show where their rebuttal falls short
            2. Introduce new angles of attack
            3. Strengthen your original critique
            4. Summarize why the claim remains doubtful""",
            expected_output="Red team counter-rebuttal",
            agent=agents["red_team"],
            context=[red_opening, blue_rebuttal]
        )
        tasks.append(red_rebuttal)

        # Final: Judge's verdict
        verdict = Task(
            description=f"""Deliver your verdict on this debate:

            CLAIM: {claim}

            BLUE TEAM ARGUMENTS:
            - Opening: {{blue_opening.output}}
            - Rebuttal: {{blue_rebuttal.output}}

            RED TEAM ARGUMENTS:
            - Attack: {{red_opening.output}}
            - Counter: {{red_rebuttal.output}}

            Deliver your verdict:
            1. Summarize the strongest points from each side
            2. Identify which arguments were decisive
            3. Rate the claim on a scale of 0-100 (confidence)
            4. Explain what would change your verdict
            5. List remaining uncertainties""",
            expected_output="Judge's verdict with confidence score",
            agent=agents["judge"],
            context=[blue_opening, red_opening, blue_rebuttal, red_rebuttal]
        )
        tasks.append(verdict)

        return tasks

    def run(self, claim: str, evidence: list) -> dict:
        """Execute adversarial debate."""
        agents = self.create_agents()
        tasks = self.create_debate_tasks(claim, evidence, agents)

        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()

        return {
            "claim": claim,
            "blue_team_argument": tasks[0].output,
            "red_team_attack": tasks[1].output,
            "blue_team_rebuttal": tasks[2].output,
            "red_team_counter": tasks[3].output,
            "verdict": tasks[4].output
        }

    def _format_evidence(self, evidence: list) -> str:
        return "\n".join([f"- {e}" for e in evidence])
```

### 2.3 Synthesis Crew (GigaThink Ensemble)

```python
# agents/crews/synthesis_crew.py

class GigaThinkSynthesisCrew:
    """
    Ensemble crew that runs all 12 GigaThink perspectives
    and synthesizes their outputs into a unified belief report.
    """

    def __init__(self, config: dict):
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        self.personas = self._load_personas()
        self.parallel_execution = config.get("parallel", True)

    def create_perspective_agents(self) -> list:
        """Create all 12 GigaThink perspective agents."""

        agents = []

        perspectives = [
            ("archaeologist", "Dr. Provenance", "Information Archaeologist"),
            ("immunologist", "Dr. Memoria", "Cognitive Immunologist"),
            ("jazz_musician", "Miles Synthesis", "Jazz Improviser"),
            ("skeptical_journalist", "Vera Factcheck", "Investigative Journalist"),
            ("network_theorist", "Dr. Graph", "Network Analyst"),
            ("game_designer", "Quest Master", "Quest Designer"),
            ("evolutionary_biologist", "Dr. Darwin", "Evolution Specialist"),
            ("chaos_engineer", "Agent Entropy", "Chaos Engineer"),
            ("ux_designer", "Claire Clarity", "UX Designer"),
            ("philosopher", "Prof. Popper", "Philosopher of Science"),
            ("economist", "Max Utility", "Information Economist"),
            ("time_traveler", "Chrono", "Temporal Analyst")
        ]

        for persona_id, name, role in perspectives:
            persona_config = self.personas.get(persona_id, {})

            agent = Agent(
                role=f"{role} ({name})",
                goal=persona_config.get("goal", f"Analyze from {role} perspective"),
                backstory=persona_config.get("system_prompt", ""),
                verbose=True,
                llm=self.llm,
                allow_delegation=False
            )
            agents.append((persona_id, agent))

        return agents

    def create_synthesizer_agent(self) -> Agent:
        """Create the meta-synthesizer agent."""

        return Agent(
            role="GigaThink Synthesizer",
            goal="Synthesize all 12 perspectives into a unified, coherent belief report",
            backstory="""You are the master synthesizer who takes insights from
            12 different expert perspectives and weaves them into a single,
            coherent understanding. You identify patterns across perspectives,
            resolve contradictions, weight insights by relevance, and produce
            a unified report that is greater than the sum of its parts.""",
            verbose=True,
            llm=ChatAnthropic(model="claude-opus-4-20250514"),  # Use Opus for synthesis
            allow_delegation=False
        )

    def create_perspective_tasks(
        self,
        query: str,
        context: str,
        agents: list
    ) -> list:
        """Create tasks for each perspective."""

        tasks = []

        for persona_id, agent in agents:
            persona_config = self.personas.get(persona_id, {})
            task_template = persona_config.get("task_template", "Analyze: {task_description}")

            task = Task(
                description=task_template.format(task_description=f"""
                QUERY: {query}

                CONTEXT:
                {context}

                Provide your unique perspective analysis."""),
                expected_output=f"{persona_id} perspective analysis",
                agent=agent
            )
            tasks.append((persona_id, task))

        return tasks

    def create_synthesis_task(
        self,
        query: str,
        perspective_tasks: list,
        synthesizer: Agent
    ) -> Task:
        """Create the final synthesis task."""

        return Task(
            description=f"""Synthesize all 12 GigaThink perspectives into a unified belief report:

            ORIGINAL QUERY: {query}

            PERSPECTIVE ANALYSES:
            {self._format_perspectives_placeholder(perspective_tasks)}

            Create a unified synthesis that:
            1. Identifies common themes across perspectives
            2. Highlights unique insights from each
            3. Resolves contradictions (or explains why they exist)
            4. Weights insights by relevance to the query
            5. Produces a final confidence score (0-100)
            6. Lists key uncertainties that remain
            7. Provides actionable recommendations

            Format as a structured BELIEF REPORT.""",
            expected_output="Unified GigaThink Belief Report",
            agent=synthesizer,
            context=[t for _, t in perspective_tasks]
        )

    def run(self, query: str, context: str) -> dict:
        """Execute the full GigaThink synthesis."""

        # Create agents
        perspective_agents = self.create_perspective_agents()
        synthesizer = self.create_synthesizer_agent()

        # Create tasks
        perspective_tasks = self.create_perspective_tasks(query, context, perspective_agents)
        synthesis_task = self.create_synthesis_task(query, perspective_tasks, synthesizer)

        # Build crew
        all_agents = [agent for _, agent in perspective_agents] + [synthesizer]
        all_tasks = [task for _, task in perspective_tasks] + [synthesis_task]

        crew = Crew(
            agents=all_agents,
            tasks=all_tasks,
            process=Process.hierarchical if self.parallel_execution else Process.sequential,
            manager_llm=ChatAnthropic(model="claude-opus-4-20250514"),
            verbose=True
        )

        result = crew.kickoff()

        # Collect outputs
        perspective_outputs = {
            persona_id: task.output
            for persona_id, task in perspective_tasks
        }

        return {
            "query": query,
            "perspective_analyses": perspective_outputs,
            "synthesis": synthesis_task.output,
            "crew_execution_log": result
        }

    def _load_personas(self) -> dict:
        """Load persona configurations."""
        # In production, load from wsop-gigathink-implementations.md
        # For now, return minimal configs
        return {}

    def _format_perspectives_placeholder(self, tasks: list) -> str:
        return "\n".join([
            f"[{persona_id.upper()}]: {{{{ {persona_id}_task.output }}}}"
            for persona_id, _ in tasks
        ])
```

---

## PART 3: LANGGRAPH STATE MACHINES

### 3.1 WSOP Orchestration Graph

```python
# agents/graphs/wsop_orchestration_graph.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from operator import add

class WSOPState(TypedDict):
    """State object for WSOP orchestration graph."""

    # Input
    original_query: str
    difficulty_mode: str
    user_constraints: dict

    # Expansion phase
    hyde_document: str
    multi_queries: List[str]
    optimized_query: str
    query_lineage: Annotated[List[dict], add]

    # Retrieval phase
    raw_results: List[dict]
    crag_action: str
    crag_confidence: float
    web_fallback_results: List[dict]

    # Analysis phase
    perspective_analyses: dict
    debate_verdict: dict
    falsification_results: List[dict]

    # Synthesis phase
    belief_report: dict
    final_confidence: float

    # Meta
    errors: Annotated[List[str], add]
    execution_path: Annotated[List[str], add]


def create_wsop_graph() -> StateGraph:
    """Create the WSOP orchestration graph."""

    graph = StateGraph(WSOPState)

    # ═══════════════════════════════════════════════════════════════════════
    # NODE DEFINITIONS
    # ═══════════════════════════════════════════════════════════════════════

    def classify_complexity(state: WSOPState) -> WSOPState:
        """Classify query complexity and determine execution path."""
        query = state["original_query"]
        mode = state.get("difficulty_mode", "auto")

        if mode == "auto":
            # Simple heuristics for complexity
            word_count = len(query.split())
            has_comparison = any(w in query.lower() for w in ["compare", "vs", "difference"])
            has_multi_hop = any(w in query.lower() for w in ["and then", "which leads to", "because"])

            if word_count < 10 and not has_comparison and not has_multi_hop:
                mode = "speed_run"
            elif has_multi_hop or has_comparison:
                mode = "completionist"
            else:
                mode = "balanced"

        state["difficulty_mode"] = mode
        state["execution_path"] = state.get("execution_path", []) + [f"classify:{mode}"]
        return state

    def expand_with_hyde(state: WSOPState) -> WSOPState:
        """Generate hypothetical document."""
        from agents.crews.expansion_crew import ExpansionCrew

        crew = ExpansionCrew({})
        result = crew.run(state["original_query"])

        state["hyde_document"] = result["hyde_document"]
        state["multi_queries"] = result["multi_queries"]
        state["optimized_query"] = result["optimized_query"]
        state["query_lineage"] = state.get("query_lineage", []) + [{
            "step": "hyde_expansion",
            "input": state["original_query"],
            "output": result["optimized_query"]
        }]
        state["execution_path"] = state.get("execution_path", []) + ["expand:hyde"]
        return state

    def skip_expansion(state: WSOPState) -> WSOPState:
        """Skip expansion for simple queries."""
        state["hyde_document"] = ""
        state["multi_queries"] = [state["original_query"]]
        state["optimized_query"] = state["original_query"]
        state["execution_path"] = state.get("execution_path", []) + ["expand:skip"]
        return state

    def retrieve_from_providers(state: WSOPState) -> WSOPState:
        """Execute retrieval from configured providers."""
        # Implementation would call actual search APIs
        queries = state["multi_queries"]

        # Mock results for illustration
        state["raw_results"] = [
            {"query": q, "results": [f"Result for {q}"]}
            for q in queries
        ]
        state["execution_path"] = state.get("execution_path", []) + ["retrieve:providers"]
        return state

    def evaluate_with_crag(state: WSOPState) -> WSOPState:
        """CRAG evaluation of retrieval quality."""
        # Implementation would run CRAG evaluator
        results = state["raw_results"]

        # Mock evaluation
        if len(results) > 0 and len(results[0].get("results", [])) > 0:
            state["crag_action"] = "CORRECT"
            state["crag_confidence"] = 0.85
        else:
            state["crag_action"] = "INCORRECT"
            state["crag_confidence"] = 0.2

        state["execution_path"] = state.get("execution_path", []) + [f"crag:{state['crag_action']}"]
        return state

    def web_search_fallback(state: WSOPState) -> WSOPState:
        """Execute web search fallback."""
        # Implementation would call web search APIs
        state["web_fallback_results"] = [
            {"source": "web", "content": f"Web result for {state['optimized_query']}"}
        ]
        state["execution_path"] = state.get("execution_path", []) + ["fallback:web"]
        return state

    def run_perspective_analysis(state: WSOPState) -> WSOPState:
        """Run GigaThink perspective analysis."""
        from agents.crews.synthesis_crew import GigaThinkSynthesisCrew

        crew = GigaThinkSynthesisCrew({"parallel": True})
        context = "\n".join([
            str(r) for r in state.get("raw_results", []) + state.get("web_fallback_results", [])
        ])
        result = crew.run(state["original_query"], context)

        state["perspective_analyses"] = result["perspective_analyses"]
        state["execution_path"] = state.get("execution_path", []) + ["analyze:gigathink"]
        return state

    def run_adversarial_debate(state: WSOPState) -> WSOPState:
        """Run adversarial debate on key claims."""
        from agents.crews.adversarial_crew import AdversarialDebateCrew

        # Extract key claim from context
        claim = f"The answer to '{state['original_query']}' is well-supported by evidence"
        evidence = [str(r) for r in state.get("raw_results", [])]

        crew = AdversarialDebateCrew({"debate_rounds": 2})
        result = crew.run(claim, evidence)

        state["debate_verdict"] = result
        state["execution_path"] = state.get("execution_path", []) + ["analyze:debate"]
        return state

    def search_for_falsification(state: WSOPState) -> WSOPState:
        """Active falsification search."""
        # Implementation would run falsification engine
        state["falsification_results"] = []
        state["execution_path"] = state.get("execution_path", []) + ["analyze:falsify"]
        return state

    def synthesize_belief_report(state: WSOPState) -> WSOPState:
        """Generate final belief report."""
        # Combine all analyses into final report
        state["belief_report"] = {
            "query": state["original_query"],
            "mode": state["difficulty_mode"],
            "perspectives": len(state.get("perspective_analyses", {})),
            "debate_verdict": state.get("debate_verdict", {}),
            "falsification_checked": len(state.get("falsification_results", [])) > 0,
            "execution_path": state.get("execution_path", [])
        }
        state["final_confidence"] = state.get("crag_confidence", 0.5)
        state["execution_path"] = state.get("execution_path", []) + ["synthesize:complete"]
        return state

    # ═══════════════════════════════════════════════════════════════════════
    # ADD NODES
    # ═══════════════════════════════════════════════════════════════════════

    graph.add_node("classify", classify_complexity)
    graph.add_node("expand_hyde", expand_with_hyde)
    graph.add_node("skip_expansion", skip_expansion)
    graph.add_node("retrieve", retrieve_from_providers)
    graph.add_node("evaluate_crag", evaluate_with_crag)
    graph.add_node("web_fallback", web_search_fallback)
    graph.add_node("perspective_analysis", run_perspective_analysis)
    graph.add_node("adversarial_debate", run_adversarial_debate)
    graph.add_node("falsification", search_for_falsification)
    graph.add_node("synthesize", synthesize_belief_report)

    # ═══════════════════════════════════════════════════════════════════════
    # ADD EDGES (CONDITIONAL ROUTING)
    # ═══════════════════════════════════════════════════════════════════════

    graph.set_entry_point("classify")

    # After classification, route based on mode
    def route_after_classify(state: WSOPState) -> str:
        mode = state["difficulty_mode"]
        if mode == "speed_run":
            return "skip_expansion"
        else:
            return "expand_hyde"

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "skip_expansion": "skip_expansion",
            "expand_hyde": "expand_hyde"
        }
    )

    # Both expansion paths lead to retrieval
    graph.add_edge("skip_expansion", "retrieve")
    graph.add_edge("expand_hyde", "retrieve")

    # After retrieval, evaluate with CRAG
    graph.add_edge("retrieve", "evaluate_crag")

    # After CRAG, route based on action
    def route_after_crag(state: WSOPState) -> str:
        action = state["crag_action"]
        if action == "INCORRECT":
            return "web_fallback"
        else:
            return "perspective_analysis"

    graph.add_conditional_edges(
        "evaluate_crag",
        route_after_crag,
        {
            "web_fallback": "web_fallback",
            "perspective_analysis": "perspective_analysis"
        }
    )

    # Web fallback leads to perspective analysis
    graph.add_edge("web_fallback", "perspective_analysis")

    # After perspective analysis, route based on mode
    def route_after_perspectives(state: WSOPState) -> str:
        mode = state["difficulty_mode"]
        if mode in ["completionist", "hardcore"]:
            return "adversarial_debate"
        else:
            return "synthesize"

    graph.add_conditional_edges(
        "perspective_analysis",
        route_after_perspectives,
        {
            "adversarial_debate": "adversarial_debate",
            "synthesize": "synthesize"
        }
    )

    # After debate, check if falsification needed
    def route_after_debate(state: WSOPState) -> str:
        mode = state["difficulty_mode"]
        if mode == "hardcore":
            return "falsification"
        else:
            return "synthesize"

    graph.add_conditional_edges(
        "adversarial_debate",
        route_after_debate,
        {
            "falsification": "falsification",
            "synthesize": "synthesize"
        }
    )

    # Falsification leads to synthesis
    graph.add_edge("falsification", "synthesize")

    # Synthesis is terminal
    graph.add_edge("synthesize", END)

    return graph.compile()


# Usage
def run_wsop_pipeline(query: str, mode: str = "auto") -> dict:
    """Run the complete WSOP pipeline."""
    graph = create_wsop_graph()

    initial_state = {
        "original_query": query,
        "difficulty_mode": mode,
        "user_constraints": {},
        "query_lineage": [],
        "errors": [],
        "execution_path": []
    }

    final_state = graph.invoke(initial_state)
    return final_state
```

---

## PART 4: BENCHMARKING & ANALYTICS FRAMEWORK

### 4.1 Agent Performance Tracker

```python
# agents/analytics/performance_tracker.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import sqlite3
import json

@dataclass
class AgentExecution:
    """Record of a single agent execution."""
    agent_id: str
    agent_name: str
    task_type: str
    query: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    success: bool
    quality_score: Optional[float]  # Human or automated evaluation
    timestamp: str

@dataclass
class CrewExecution:
    """Record of a crew execution."""
    crew_id: str
    crew_type: str
    query: str
    agents_used: List[str]
    total_latency_ms: int
    total_cost_usd: float
    final_confidence: float
    quality_score: Optional[float]
    execution_path: List[str]
    timestamp: str


class PerformanceTracker:
    """Track and analyze agent/crew performance."""

    def __init__(self, db_path: str = "agent_analytics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize analytics database."""
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_executions (
                id INTEGER PRIMARY KEY,
                agent_id TEXT,
                agent_name TEXT,
                task_type TEXT,
                query TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                latency_ms INTEGER,
                success INTEGER,
                quality_score REAL,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS crew_executions (
                id INTEGER PRIMARY KEY,
                crew_id TEXT,
                crew_type TEXT,
                query TEXT,
                agents_used TEXT,
                total_latency_ms INTEGER,
                total_cost_usd REAL,
                final_confidence REAL,
                quality_score REAL,
                execution_path TEXT,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY,
                test_name TEXT,
                variant_a TEXT,
                variant_b TEXT,
                query TEXT,
                winner TEXT,
                score_a REAL,
                score_b REAL,
                timestamp TEXT
            )
        """)

        conn.commit()
        conn.close()

    def record_agent_execution(self, execution: AgentExecution):
        """Record agent execution metrics."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO agent_executions
            (agent_id, agent_name, task_type, query, input_tokens, output_tokens,
             latency_ms, success, quality_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.agent_id,
            execution.agent_name,
            execution.task_type,
            execution.query,
            execution.input_tokens,
            execution.output_tokens,
            execution.latency_ms,
            1 if execution.success else 0,
            execution.quality_score,
            execution.timestamp
        ))
        conn.commit()
        conn.close()

    def record_crew_execution(self, execution: CrewExecution):
        """Record crew execution metrics."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO crew_executions
            (crew_id, crew_type, query, agents_used, total_latency_ms,
             total_cost_usd, final_confidence, quality_score, execution_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.crew_id,
            execution.crew_type,
            execution.query,
            json.dumps(execution.agents_used),
            execution.total_latency_ms,
            execution.total_cost_usd,
            execution.final_confidence,
            execution.quality_score,
            json.dumps(execution.execution_path),
            execution.timestamp
        ))
        conn.commit()
        conn.close()

    def get_agent_leaderboard(self, task_type: Optional[str] = None) -> List[dict]:
        """Get agent performance leaderboard."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                agent_name,
                COUNT(*) as executions,
                AVG(quality_score) as avg_quality,
                AVG(latency_ms) as avg_latency,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                AVG(output_tokens) as avg_output_tokens
            FROM agent_executions
            WHERE quality_score IS NOT NULL
        """

        if task_type:
            query += f" AND task_type = '{task_type}'"

        query += " GROUP BY agent_name ORDER BY avg_quality DESC"

        cursor = conn.execute(query)
        results = cursor.fetchall()
        conn.close()

        return [
            {
                "agent_name": r[0],
                "executions": r[1],
                "avg_quality": round(r[2], 3) if r[2] else None,
                "avg_latency_ms": round(r[3], 1) if r[3] else None,
                "success_rate": round(r[4], 1) if r[4] else None,
                "avg_output_tokens": round(r[5], 1) if r[5] else None
            }
            for r in results
        ]

    def get_crew_comparison(self) -> dict:
        """Compare performance across crew types."""
        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute("""
            SELECT
                crew_type,
                COUNT(*) as executions,
                AVG(quality_score) as avg_quality,
                AVG(final_confidence) as avg_confidence,
                AVG(total_latency_ms) as avg_latency,
                AVG(total_cost_usd) as avg_cost
            FROM crew_executions
            WHERE quality_score IS NOT NULL
            GROUP BY crew_type
            ORDER BY avg_quality DESC
        """)

        results = cursor.fetchall()
        conn.close()

        return {
            r[0]: {
                "executions": r[1],
                "avg_quality": round(r[2], 3) if r[2] else None,
                "avg_confidence": round(r[3], 3) if r[3] else None,
                "avg_latency_ms": round(r[4], 1) if r[4] else None,
                "avg_cost_usd": round(r[5], 4) if r[5] else None
            }
            for r in results
        }

    def get_optimal_crew_for_query_type(self, query_type: str) -> str:
        """Recommend optimal crew configuration for query type."""
        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute("""
            SELECT
                crew_type,
                AVG(quality_score) as avg_quality,
                AVG(total_latency_ms) as avg_latency
            FROM crew_executions
            WHERE quality_score IS NOT NULL
            GROUP BY crew_type
            ORDER BY avg_quality DESC
            LIMIT 1
        """)

        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0]
        return "balanced"  # Default

    def run_ab_test(
        self,
        test_name: str,
        query: str,
        variant_a: callable,
        variant_b: callable,
        evaluator: callable
    ) -> dict:
        """Run A/B test between two crew configurations."""

        # Run variant A
        result_a = variant_a(query)
        score_a = evaluator(result_a)

        # Run variant B
        result_b = variant_b(query)
        score_b = evaluator(result_b)

        # Determine winner
        winner = "A" if score_a > score_b else "B" if score_b > score_a else "TIE"

        # Record test
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO ab_tests
            (test_name, variant_a, variant_b, query, winner, score_a, score_b, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_name,
            str(variant_a),
            str(variant_b),
            query,
            winner,
            score_a,
            score_b,
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()

        return {
            "test_name": test_name,
            "query": query,
            "score_a": score_a,
            "score_b": score_b,
            "winner": winner
        }

    def generate_analytics_report(self) -> dict:
        """Generate comprehensive analytics report."""

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_leaderboard": self.get_agent_leaderboard(),
            "crew_comparison": self.get_crew_comparison(),
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on data."""
        recommendations = []

        leaderboard = self.get_agent_leaderboard()
        if leaderboard:
            top_agent = leaderboard[0]
            recommendations.append(
                f"Top performing agent: {top_agent['agent_name']} "
                f"(quality: {top_agent['avg_quality']}, success: {top_agent['success_rate']}%)"
            )

        crew_comparison = self.get_crew_comparison()
        if crew_comparison:
            best_quality = max(crew_comparison.items(), key=lambda x: x[1].get('avg_quality', 0))
            best_speed = min(crew_comparison.items(), key=lambda x: x[1].get('avg_latency_ms', float('inf')))

            recommendations.append(
                f"Best quality crew: {best_quality[0]} (avg quality: {best_quality[1]['avg_quality']})"
            )
            recommendations.append(
                f"Fastest crew: {best_speed[0]} (avg latency: {best_speed[1]['avg_latency_ms']}ms)"
            )

        return recommendations
```

---

## PART 5: PROMPT ENGINEERING TEMPLATES

### 5.1 High-Quality Prompt Library

```yaml
# agents/prompts/prompt_library.yaml

meta_orchestrator_prompt: |
  You are THE SYMPOSIARCH - the master orchestrator of a multi-agent symposium.

  ## YOUR ROLE
  You do not answer questions directly. Instead, you:
  1. Analyze the incoming query
  2. Determine which crews and perspectives are needed
  3. Compose the optimal team for this specific challenge
  4. Orchestrate their collaboration
  5. Synthesize their outputs into unified wisdom

  ## AVAILABLE CREWS
  - EXPANSION CREW: Query transformation specialists (HyDE, Multi-Query, Rewriting)
  - RETRIEVAL CREW: Information gathering specialists
  - ADVERSARIAL CREW: Red Team vs Blue Team debate
  - SYNTHESIS CREW: 12 GigaThink perspectives

  ## DECISION FRAMEWORK
  For each query, consider:
  1. COMPLEXITY: Is this simple, moderate, or complex?
  2. STAKES: What's the cost of being wrong?
  3. DOMAIN: Which perspectives are most relevant?
  4. CONSTRAINTS: Time, cost, confidence requirements?

  ## OUTPUT FORMAT
  Always respond with a CREW COMPOSITION PLAN:
  ```
  QUERY ANALYSIS:
  - Complexity: [simple/moderate/complex]
  - Stakes: [low/medium/high/critical]
  - Primary domain: [technical/factual/analytical/creative]

  CREW COMPOSITION:
  1. [Crew Name]: [Rationale]
  2. [Crew Name]: [Rationale]

  EXECUTION ORDER:
  [Sequential/Parallel description]

  SUCCESS CRITERIA:
  - [Criteria 1]
  - [Criteria 2]
  ```


adversarial_debate_judge_prompt: |
  You are an IMPARTIAL JUDGE presiding over an intellectual debate.

  ## YOUR SACRED DUTIES
  1. FAIRNESS: No predetermined conclusions. Evaluate arguments on merit.
  2. RIGOR: Identify logical fallacies, weak evidence, and rhetorical tricks.
  3. CLARITY: Explain your reasoning so both sides understand your verdict.
  4. HUMILITY: Acknowledge uncertainty where it exists.

  ## EVALUATION CRITERIA
  For each argument, assess:
  - EVIDENCE QUALITY: How strong is the supporting evidence?
  - LOGICAL COHERENCE: Does the reasoning follow?
  - COMPLETENESS: Are there obvious gaps?
  - COUNTER-ARGUMENT HANDLING: How well did they address opposition?

  ## VERDICT FORMAT
  ```
  SUMMARY OF ARGUMENTS:
  - Blue Team's strongest point: [X]
  - Red Team's strongest point: [Y]

  DECISIVE FACTORS:
  1. [Factor that tipped the balance]
  2. [Secondary consideration]

  VERDICT: [BLUE TEAM / RED TEAM / SPLIT DECISION]
  CONFIDENCE: [0-100]%

  REMAINING UNCERTAINTIES:
  - [What we still don't know]

  WHAT WOULD CHANGE MY MIND:
  - [Condition that would reverse verdict]
  ```


synthesis_maestro_prompt: |
  You are the SYNTHESIS MAESTRO - conductor of the GigaThink ensemble.

  ## YOUR ART
  You receive analyses from 12 distinct perspectives:
  - The Archaeologist (provenance and lineage)
  - The Immunologist (threat detection and memory)
  - The Jazz Musician (improvisation and harmony)
  - The Skeptical Journalist (motive analysis)
  - The Network Theorist (connections and hubs)
  - The Game Designer (quests and strategies)
  - The Evolutionary Biologist (adaptation and fitness)
  - The Chaos Engineer (failure modes and uncertainty)
  - The UX Designer (clarity and action)
  - The Philosopher (falsification and epistemics)
  - The Economist (costs and optimization)
  - The Time Traveler (temporal validity)

  ## YOUR TASK
  Weave these 12 threads into a SINGLE TAPESTRY that:
  1. Identifies CONVERGENT INSIGHTS (where perspectives agree)
  2. Highlights UNIQUE CONTRIBUTIONS (valuable divergent views)
  3. Resolves CONTRADICTIONS (or explains why they exist)
  4. Weights by RELEVANCE (not all perspectives equally important for every query)
  5. Produces ACTIONABLE SYNTHESIS (what should the user DO with this?)

  ## OUTPUT: THE BELIEF REPORT
  ```
  # UNIFIED BELIEF REPORT

  ## TL;DR (≤100 words)
  [The essential answer]

  ## CONFIDENCE: [X]%
  [Explanation of confidence level]

  ## CONVERGENT INSIGHTS
  [Where multiple perspectives agreed]

  ## UNIQUE VALUABLE PERSPECTIVES
  [Insights that only one perspective provided]

  ## RESOLVED CONTRADICTIONS
  [How you reconciled conflicting views]

  ## REMAINING UNCERTAINTIES
  [What we genuinely don't know]

  ## RECOMMENDED ACTIONS
  1. [Primary action]
  2. [Secondary action]
  3. [If you want more certainty, do X]
  ```
```

---

## CONFIGURATION FILES

### Main Configuration

```yaml
# config/multi_agent_config.yaml

wsop_multi_agent:
  version: "1.0.0"

  # Model allocation
  models:
    meta_orchestrator: "claude-opus-4-20250514"
    synthesis: "claude-opus-4-20250514"
    perspective_agents: "claude-sonnet-4-20250514"
    debate_agents: "claude-sonnet-4-20250514"
    expansion_agents: "claude-sonnet-4-20250514"

  # Crew configurations
  crews:
    expansion:
      enabled: true
      agents: ["hyde", "multi_query", "rewriter"]
      process: "sequential"

    adversarial:
      enabled: true
      debate_rounds: 2
      require_verdict: true

    synthesis:
      enabled: true
      perspectives: "all"  # or list specific ones
      parallel: true

  # Execution modes
  modes:
    speed_run:
      crews: ["expansion"]
      perspectives: ["ux_designer"]
      skip_debate: true

    balanced:
      crews: ["expansion", "synthesis"]
      perspectives: ["skeptical_journalist", "philosopher", "ux_designer", "economist"]
      skip_debate: true

    completionist:
      crews: ["expansion", "synthesis", "adversarial"]
      perspectives: "all"
      skip_debate: false

    hardcore:
      crews: ["expansion", "synthesis", "adversarial"]
      perspectives: "all"
      skip_debate: false
      require_falsification: true

  # Analytics
  analytics:
    enabled: true
    db_path: "agent_analytics.db"
    track_all_executions: true
    ab_testing_enabled: true

  # Cost controls
  cost_limits:
    per_query_usd: 0.50
    per_session_usd: 5.00
    alert_threshold: 0.80
```

---

## CHANGELOG

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-12 | Initial multi-agent framework |
| | | 12 GigaThink agent personas |
| | | 3 CrewAI crew configurations |
| | | LangGraph orchestration state machine |
| | | Performance analytics framework |
| | | High-quality prompt templates |

---

*WSOP Multi-Agent Framework v1.0 | "The Symposium" | CrewAI + LangGraph Integration*
