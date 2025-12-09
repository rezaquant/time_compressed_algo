# Time-Compressed Algorithms

Tensor-network experiments focused on PEPO/PEPS compression and time-evolution optimizations. Scripts implement cooling/optimization routines and utilities; notebooks capture exploratory runs and plots.

## Layout
- `src/tcompress/`: Python package for reusable code.
  - `algo_cooling.py`: cooling and tensor-network routines used across experiments.
  - `p_pepo.py`, `p_pepo_su.py`: PEPO optimization flows (standard and SU variants).
  - `Uoptimize.py`, `gate_arb.py`: optimization helpers and arbitrary-geometry tensor utilities.
  - `quf.py`, `register_.py`: shared utilities and registrations.
- `notebooks/pepo/`: PEPO/PEPS-focused experiments and trotter reductions.
- `notebooks/mps/`: MPS simulations.
- `notebooks/plotting/`: plotting/visualization explorations.
- `notebooks/analysis/`: analysis/scratch runs.
- `data/outputs/`: cached artifacts and reference data (ignored).
- `requirements.txt`: Python dependencies for experiments.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Option A: editable install
pip install -e .
# Option B: point Python to the package when running notebooks/scripts
# export PYTHONPATH=\"$PWD/src:$PYTHONPATH\"
# For GPU JAX/PyTorch, install vendor wheels as needed.
```

## Notes
- `.gitattributes` marks notebooks as binary to avoid noisy diffs; use `nbdiff` or screenshots for review.
- `.gitignore` excludes checkpoints, caches, and `data/outputs/`. Keep transient data there or outside the repo.
- Large generated data should stay out of version control or use Git LFS if required.
