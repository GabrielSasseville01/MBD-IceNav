#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --job-name=tune-shared_trajopt-A
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --array=0-767
#SBATCH --output=slurm_logs/tune_shared_trajopt_A_%A_%a.out
#SBATCH --error=slurm_logs/tune_shared_trajopt_A_%A_%a.err

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="${HOME}/.local/bin:${PATH}"

# Add bundled nvidia CUDA libs so JAX can find them (no-op on CPU-only nodes)
NVIDIA_LIBS=$(find .venv/lib -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"

# ── Array decode ──────────────────────────────────────────────────────────
SCENARIOS=(A_conc05_thick A_conc05_bimodal A_conc10_thick A_conc10_bimodal A_conc20_thick A_conc20_bimodal A_conc30_thick A_conc30_bimodal A_conc40_thick A_conc40_bimodal A_conc50_thick A_conc50_bimodal)
N_SCENARIOS=${#SCENARIOS[@]}
N_GRID=64

TASK_ID=$SLURM_ARRAY_TASK_ID
SCENARIO_IDX=$((TASK_ID % N_SCENARIOS))
GRID_IDX=$((TASK_ID / N_SCENARIOS))
SCENARIO=${SCENARIOS[$SCENARIO_IDX]}
GRID_IDX_PADDED=$(printf '%04d' $GRID_IDX)

# ── Load grid overrides ───────────────────────────────────────────────────
GRID_JSON="outputs/tuning/shared_trajopt/grid.json"
OVERRIDES_JSON=$(uv run python -c "
import json
g = json.load(open('$GRID_JSON'))
print(json.dumps(g[str($GRID_IDX)]))
")

OUTPUT_SUBPATH="tuning/shared_trajopt/grid_${GRID_IDX_PADDED}"

echo "─── Task $TASK_ID: block=A scenario=$SCENARIO grid=$GRID_IDX ───"
echo "Overrides: $OVERRIDES_JSON"


# ── Run all (planner × seed) trials for this (scenario, grid) ──────────────
for PLANNER in mbd mppi rss ilqr; do
    for SEED in 0 1; do
        OUT_JSON="outputs/${OUTPUT_SUBPATH}/A/${SCENARIO}/${PLANNER}/seed_${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            echo "SKIP $OUT_JSON (exists)"
            continue
        fi
        echo "▸ planner=$PLANNER seed=$SEED"
        uv run python run_experiment_trial.py \
            --block A \
            --scenario "$SCENARIO" \
            --planner "$PLANNER" \
            --seed "$SEED" \
            --output-dir outputs \
            --output-subpath "$OUTPUT_SUBPATH" \
            --overrides-json "$OVERRIDES_JSON"
    done
done

