#!/bin/bash
#SBATCH --job-name=icenav-A-A_conc05_thin-mbd-s0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/icenav_A_A_conc05_thin_mbd_s0_%j.out
#SBATCH --error=slurm_logs/icenav_A_A_conc05_thin_mbd_s0_%j.err

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
export PATH="${HOME}/.local/bin:${PATH}"

# Add bundled nvidia CUDA libs so JAX can find them (CPU-only for now, no-op on CPU nodes)
NVIDIA_LIBS=$(find .venv/lib -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"

uv run python run_experiment_trial.py \
    --block A \
    --scenario A_conc05_thin \
    --planner mbd \
    --seed 0 \
    --output-dir outputs/experiments
