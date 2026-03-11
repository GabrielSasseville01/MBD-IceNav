# SLURM job scripts

Submit from project root:
```bash
sbatch scripts/run_full_map_lattice.slurm
sbatch scripts/run_full_map_diffusion.slurm
```

**Customize for your cluster:**
- `--partition`: Replace `defq` with your partition (e.g. `standard`, `normal`, `cpu`)
- `--time`: 24h requested; adjust if limits differ
- `--mem`: 16G default; increase if OOM
- `--cpus-per-task`: 4 threads for parallel animation workers

**Optional:** Load environment before `uv run` if needed:
```bash
module load python/3.11  # or equivalent
# or: source venv/bin/activate
```

**Output:** Logs in `outputs/slurm_<jobid>.out` and `outputs/slurm_<jobid>.err`; videos in `outputs/default/`.
