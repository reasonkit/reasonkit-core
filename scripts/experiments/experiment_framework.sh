#!/bin/bash
# =============================================================================
# ReasonKit Experiment Framework
# =============================================================================
#
# Shared utilities for running reproducible experiments.
# Source this file from individual experiment scripts.
#
# Usage:
#   source ./experiment_framework.sh
#   init_experiment "H3" "Cross-Encoder Reranking"
#   log_metric "mrr_at_10" "0.42"
#   finalize_experiment
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

FRAMEWORK_VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_ROOT="${PROJECT_ROOT}/data/experiments"
RESULTS_ROOT="${PROJECT_ROOT}/results/experiments"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Experiment state
EXPERIMENT_ID=""
EXPERIMENT_NAME=""
EXPERIMENT_DIR=""
EXPERIMENT_START=""
METRICS_FILE=""
LOG_FILE=""

# =============================================================================
# Initialization
# =============================================================================

init_experiment() {
    local hypothesis_id="$1"
    local experiment_name="$2"
    local run_id="${3:-$(date +%Y%m%d_%H%M%S)}"

    EXPERIMENT_ID="${hypothesis_id}"
    EXPERIMENT_NAME="${experiment_name}"
    EXPERIMENT_DIR="${RESULTS_ROOT}/${hypothesis_id}/${run_id}"
    EXPERIMENT_START=$(date +%s)
    METRICS_FILE="${EXPERIMENT_DIR}/metrics.json"
    LOG_FILE="${EXPERIMENT_DIR}/experiment.log"

    # Create directories
    mkdir -p "${EXPERIMENT_DIR}"
    mkdir -p "${DATA_ROOT}/${hypothesis_id}"

    # Initialize metrics file
    cat > "${METRICS_FILE}" << EOF
{
    "experiment_id": "${hypothesis_id}",
    "experiment_name": "${experiment_name}",
    "run_id": "${run_id}",
    "framework_version": "${FRAMEWORK_VERSION}",
    "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "system": {
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "kernel": "$(uname -r)",
        "arch": "$(uname -m)",
        "rust_version": "$(rustc --version 2>/dev/null || echo 'N/A')"
    },
    "metrics": {},
    "status": "running"
}
EOF

    # Log header
    {
        echo "=============================================="
        echo "ReasonKit Experiment: ${hypothesis_id}"
        echo "${experiment_name}"
        echo "=============================================="
        echo "Run ID: ${run_id}"
        echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Directory: ${EXPERIMENT_DIR}"
        echo "=============================================="
        echo ""
    } > "${LOG_FILE}"

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  Experiment: ${hypothesis_id} - ${experiment_name}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Run ID:${NC} ${run_id}"
    echo -e "${BLUE}Output:${NC} ${EXPERIMENT_DIR}"
    echo ""
}

# =============================================================================
# Logging
# =============================================================================

log_info() {
    local message="$1"
    echo -e "${BLUE}[INFO]${NC} ${message}"
    echo "[$(date +%H:%M:%S)] [INFO] ${message}" >> "${LOG_FILE}"
}

log_success() {
    local message="$1"
    echo -e "${GREEN}[PASS]${NC} ${message}"
    echo "[$(date +%H:%M:%S)] [PASS] ${message}" >> "${LOG_FILE}"
}

log_warning() {
    local message="$1"
    echo -e "${YELLOW}[WARN]${NC} ${message}"
    echo "[$(date +%H:%M:%S)] [WARN] ${message}" >> "${LOG_FILE}"
}

log_error() {
    local message="$1"
    echo -e "${RED}[FAIL]${NC} ${message}"
    echo "[$(date +%H:%M:%S)] [FAIL] ${message}" >> "${LOG_FILE}"
}

log_phase() {
    local phase="$1"
    echo ""
    echo -e "${MAGENTA}▶ Phase: ${phase}${NC}"
    echo "" >> "${LOG_FILE}"
    echo "▶ Phase: ${phase}" >> "${LOG_FILE}"
}

# =============================================================================
# Metrics
# =============================================================================

log_metric() {
    local name="$1"
    local value="$2"
    local unit="${3:-}"

    echo -e "  ${GREEN}✓${NC} ${name}: ${CYAN}${value}${NC} ${unit}"
    echo "[$(date +%H:%M:%S)] [METRIC] ${name}=${value} ${unit}" >> "${LOG_FILE}"

    # Update JSON metrics file using Python (more reliable than jq for nested updates)
    python3 << EOF
import json
with open('${METRICS_FILE}', 'r') as f:
    data = json.load(f)
data['metrics']['${name}'] = {'value': ${value}, 'unit': '${unit}'}
with open('${METRICS_FILE}', 'w') as f:
    json.dump(data, f, indent=2)
EOF
}

log_metric_string() {
    local name="$1"
    local value="$2"

    echo -e "  ${GREEN}✓${NC} ${name}: ${CYAN}${value}${NC}"
    echo "[$(date +%H:%M:%S)] [METRIC] ${name}=${value}" >> "${LOG_FILE}"

    python3 << EOF
import json
with open('${METRICS_FILE}', 'r') as f:
    data = json.load(f)
data['metrics']['${name}'] = '${value}'
with open('${METRICS_FILE}', 'w') as f:
    json.dump(data, f, indent=2)
EOF
}

# =============================================================================
# Benchmark Helpers
# =============================================================================

run_cargo_bench() {
    local bench_name="$1"
    local output_file="${EXPERIMENT_DIR}/${bench_name}_output.txt"

    log_info "Running benchmark: ${bench_name}"

    cd "${PROJECT_ROOT}"

    if cargo bench --bench "${bench_name}" 2>&1 | tee "${output_file}"; then
        log_success "Benchmark ${bench_name} completed"
        return 0
    else
        log_error "Benchmark ${bench_name} failed"
        return 1
    fi
}

run_cargo_test() {
    local test_filter="$1"
    local output_file="${EXPERIMENT_DIR}/test_${test_filter}_output.txt"

    log_info "Running tests: ${test_filter}"

    cd "${PROJECT_ROOT}"

    if cargo test --release "${test_filter}" 2>&1 | tee "${output_file}"; then
        log_success "Tests ${test_filter} passed"
        return 0
    else
        log_error "Tests ${test_filter} failed"
        return 1
    fi
}

# =============================================================================
# Data Management
# =============================================================================

download_dataset() {
    local name="$1"
    local url="$2"
    local dest="${DATA_ROOT}/${EXPERIMENT_ID}/${name}"

    if [ -f "${dest}" ]; then
        log_info "Dataset ${name} already exists"
        return 0
    fi

    log_info "Downloading dataset: ${name}"

    if curl -fsSL -o "${dest}" "${url}"; then
        log_success "Downloaded ${name}"
        return 0
    else
        log_error "Failed to download ${name}"
        return 1
    fi
}

# =============================================================================
# Statistical Helpers
# =============================================================================

compute_mean() {
    local values="$1"
    python3 -c "import statistics; print(statistics.mean([${values}]))"
}

compute_stdev() {
    local values="$1"
    python3 -c "import statistics; print(statistics.stdev([${values}]))"
}

compute_percentile() {
    local values="$1"
    local percentile="$2"
    python3 -c "
import statistics
vals = sorted([${values}])
k = (len(vals) - 1) * ${percentile} / 100
f = int(k)
c = f + 1 if f + 1 < len(vals) else f
print(vals[f] + (vals[c] - vals[f]) * (k - f))
"
}

# =============================================================================
# Finalization
# =============================================================================

finalize_experiment() {
    local status="${1:-completed}"
    local experiment_end=$(date +%s)
    local duration=$((experiment_end - EXPERIMENT_START))

    # Update metrics file with final status
    python3 << EOF
import json
with open('${METRICS_FILE}', 'r') as f:
    data = json.load(f)
data['status'] = '${status}'
data['completed_at'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
data['duration_seconds'] = ${duration}
with open('${METRICS_FILE}', 'w') as f:
    json.dump(data, f, indent=2)
EOF

    # Log footer
    {
        echo ""
        echo "=============================================="
        echo "Experiment Complete"
        echo "Status: ${status}"
        echo "Duration: ${duration}s"
        echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "=============================================="
    } >> "${LOG_FILE}"

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  Experiment Complete${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Status:${NC} ${status}"
    echo -e "${BLUE}Duration:${NC} ${duration}s"
    echo -e "${BLUE}Results:${NC} ${EXPERIMENT_DIR}"
    echo -e "${BLUE}Metrics:${NC} ${METRICS_FILE}"
    echo ""
}

# =============================================================================
# Report Generation
# =============================================================================

generate_markdown_report() {
    local report_file="${EXPERIMENT_DIR}/REPORT.md"

    log_info "Generating markdown report..."

    python3 << EOF
import json
from datetime import datetime

with open('${METRICS_FILE}', 'r') as f:
    data = json.load(f)

report = f"""# Experiment Report: {data['experiment_id']}

## {data['experiment_name']}

**Run ID:** {data['run_id']}
**Status:** {data['status']}
**Started:** {data['started_at']}
**Completed:** {data.get('completed_at', 'N/A')}
**Duration:** {data.get('duration_seconds', 0)}s

---

## System Information

| Property | Value |
|----------|-------|
| Hostname | {data['system']['hostname']} |
| OS | {data['system']['os']} |
| Kernel | {data['system']['kernel']} |
| Architecture | {data['system']['arch']} |
| Rust Version | {data['system']['rust_version']} |

---

## Metrics

| Metric | Value | Unit |
|--------|-------|------|
"""

for name, value in data.get('metrics', {}).items():
    if isinstance(value, dict):
        report += f"| {name} | {value.get('value', 'N/A')} | {value.get('unit', '')} |\n"
    else:
        report += f"| {name} | {value} | |\n"

report += """
---

## Reproducibility

To reproduce this experiment:

\`\`\`bash
cd reasonkit-core
./scripts/experiments/{experiment_id}.sh
\`\`\`

---

*Generated by ReasonKit Experiment Framework v{version}*
""".format(experiment_id=data['experiment_id'].lower(), version='${FRAMEWORK_VERSION}')

with open('${report_file}', 'w') as f:
    f.write(report)
EOF

    log_success "Report generated: ${report_file}"
}
