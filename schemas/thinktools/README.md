# ThinkTools JSON Schemas

This directory contains formal JSON Schema definitions for all ThinkTools module outputs.

## Schema Files

| Schema                      | Module        | Description                                          |
| --------------------------- | ------------- | ---------------------------------------------------- |
| `gigathink_output.json`     | GigaThink     | Expansive creative thinking with 10+ perspectives    |
| `laserlogic_output.json`    | LaserLogic    | Precision deductive reasoning with fallacy detection |
| `bedrock_output.json`       | BedRock       | First principles decomposition and axiom rebuilding  |
| `proofguard_output.json`    | ProofGuard    | Multi-source verification with triangulation         |
| `brutalhonesty_output.json` | BrutalHonesty | Adversarial self-critique and flaw detection         |
| `synthesis_output.json`     | Synthesis     | Final merged output from all modules                 |

## Schema Structure

All schemas follow this common pattern:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://reasonkit.sh/schemas/thinktools/<module>_output.json",
  "title": "<Module> Output Schema",
  "version": "2.0.0",
  "type": "object",
  "required": [
    "module",
    "version",
    "timestamp",
    "query",
    "confidence",
    "thinking_trace"
  ],
  "properties": {
    "module": { "type": "string", "const": "<module_name>" },
    "version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" },
    "timestamp": { "type": "string", "format": "date-time" },
    "query": { "type": "string" },
    "confidence": {
      "type": "object",
      "required": ["overall", "factors", "breakdown"],
      "properties": {
        "overall": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "factors": { "type": "object" },
        "breakdown": { "type": "string" }
      }
    },
    "thinking_trace": { "type": "array" }
  }
}
```

## Validation

### Rust

```rust
use serde_json::Value;
use jsonschema::JSONSchema;

fn validate_output(module: &str, output: &Value) -> Result<(), String> {
    let schema_path = format!("schemas/thinktools/{}_output.json", module);
    let schema: Value = serde_json::from_str(&std::fs::read_to_string(schema_path)?)?;
    let compiled = JSONSchema::compile(&schema).map_err(|e| e.to_string())?;

    compiled.validate(output)
        .map_err(|errors| {
            errors.map(|e| e.to_string()).collect::<Vec<_>>().join(", ")
        })
}
```

### Python

```python
import json
from jsonschema import validate

def validate_output(module: str, output: dict):
    with open(f"schemas/thinktools/{module}_output.json") as f:
        schema = json.load(f)
    validate(instance=output, schema=schema)
```

### JavaScript/TypeScript

```typescript
import Ajv from "ajv";
import fs from "fs";

function validateOutput(module: string, output: any): boolean {
  const schema = JSON.parse(
    fs.readFileSync(`schemas/thinktools/${module}_output.json`, "utf-8"),
  );
  const ajv = new Ajv();
  const validate = ajv.compile(schema);
  return validate(output);
}
```

## Schema Highlights

### GigaThink Output

**Key Fields:**

- `perspectives`: Array of 10-25 diverse viewpoints
- `emergent_insights`: Insights from combining perspectives
- `cross_domain_analogies`: Analogies from other domains
- `semantic_clusters`: Perspectives grouped by similarity

**Confidence Factors:**

- perspective_diversity (0.25 weight)
- insight_novelty (0.20 weight)
- domain_coverage (0.15 weight)
- contradiction_detection (0.20 weight)
- actionability (0.20 weight)

### LaserLogic Output

**Key Fields:**

- `premises`: Extracted premises with validity assessment
- `deductive_chains`: Logical reasoning chains
- `fallacies_detected`: Array of 18 fallacy types
- `conclusion`: Final conclusion with strength score

**Confidence Factors:**

- premise_validity (0.30 weight)
- chain_coherence (0.25 weight)
- fallacy_absence (0.25 weight)
- conclusion_strength (0.20 weight)

### BedRock Output

**Key Fields:**

- `decomposition_layers`: 5 levels from surface to axioms
- `axioms`: Fundamental axioms identified
- `assumptions_surfaced`: Hidden assumptions discovered
- `reconstruction`: Rebuilding from axioms upward

**Confidence Factors:**

- axiom_soundness (0.35 weight)
- decomposition_completeness (0.25 weight)
- assumption_identification (0.20 weight)
- rebuild_coherence (0.20 weight)

### ProofGuard Output

**Key Fields:**

- `claims_extracted`: Factual claims to verify
- `verification_results`: Verification status per claim
- `contradictions`: Contradictions found across sources
- `triangulation_table`: 3-source verification table

**Confidence Factors:**

- source_count (0.20 weight)
- source_diversity (0.25 weight)
- tier_quality (0.20 weight)
- contradiction_absence (0.25 weight)
- verification_completeness (0.10 weight)

### BrutalHonesty Output

**Key Fields:**

- `critiques`: Array of adversarial critiques
- `edge_cases`: Edge cases that could break reasoning
- `biases_detected`: Cognitive biases found
- `failure_modes`: Predicted failure modes
- `overall_assessment`: Verdict and recommended action

**Confidence Factors:**

- critique_depth (0.25 weight)
- fatal_flaw_detection (0.30 weight)
- edge_case_coverage (0.20 weight)
- bias_identification (0.15 weight)
- remediation_quality (0.10 weight)

### Synthesis Output

**Key Fields:**

- `profile`: Reasoning profile used
- `modules_executed`: Execution order and timing
- `module_outputs`: Raw outputs from each module
- `cross_module_analysis`: Contradictions, agreements, insights
- `final_synthesis`: Summary, findings, decision matrix
- `confidence`: Overall confidence with detailed calculation
- `recommendation`: Final action recommendation

## Schema Evolution

### Versioning

Schemas follow semantic versioning:

- **Major**: Breaking changes to required fields
- **Minor**: New optional fields
- **Patch**: Documentation, examples, clarifications

Current version: **2.0.0**

### Backward Compatibility

When updating schemas:

1. Never remove required fields
2. Make new fields optional when possible
3. Use `additionalProperties: false` carefully
4. Document migration path for breaking changes

## Testing

### Schema Validation Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gigathink_schema() {
        let output = serde_json::json!({
            "module": "gigathink",
            "version": "2.0.0",
            "timestamp": "2025-12-22T10:00:00Z",
            "query": "Test query",
            "perspectives": [/* ... */],
            "confidence": {
                "overall": 0.85,
                "factors": {},
                "breakdown": "Test"
            },
            "thinking_trace": []
        });

        validate_output("gigathink", &output).unwrap();
    }
}
```

## Contributing

To add or modify schemas:

1. Fork ReasonKit-core
2. Edit schema in `schemas/thinktools/`
3. Update version if breaking change
4. Add validation tests
5. Update this README
6. Submit PR

## Resources

- **Full Protocol:** `protocols/thinktools.yaml`
- **User Guide:** `docs/THINKTOOLS_GUIDE.md`
- **API Docs:** <https://docs.rs/reasonkit-core>

---

_ThinkTools Schemas | Apache 2.0 | <https://reasonkit.sh>_
