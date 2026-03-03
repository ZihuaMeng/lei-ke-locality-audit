#!/usr/bin/env bash
# run_audit.sh — M0.5 smoke test runner
# Usage: bash scripts/run_audit.sh [NO_EDIT|ROME|MEMIT]

set -euo pipefail

MODE="${1:-NO_EDIT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== KE Locality Audit ==="
echo "Mode     : $MODE"
echo "Repo root: $REPO_ROOT"
echo "Time     : $(date)"
echo "========================="

# Activate conda env
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ke_audit

mkdir -p logs outputs

python src/audit_suite.py \
    --mode "$MODE" \
    --prompt_dir data/prompts \
    --output_dir outputs \
    2>&1 | tee "logs/run_$(date +%Y%m%d_%H%M%S).log"

python src/report.py \
    --results outputs/results.json \
    --output_dir outputs

echo ""
echo "=== Done ==="
echo "results.json  -> outputs/results.json"
echo "audit_report  -> outputs/audit_report.md"
echo "logs          -> logs/"
