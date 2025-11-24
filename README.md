# Time-Compressed Algorithms

Tensor-network experiments focused on PEPO/PEPS compression and time-evolution optimizations. Scripts implement cooling/optimization routines and utilities; notebooks capture exploratory runs and plots.

## Layout
- `algo_cooling.py`: cooling and tensor-network routines used across experiments.
- `p_pepo.py`, `p_pepo_su.py`: PEPO optimization flows (standard and SU variants).
- `Uoptimize.py`, `gate_arb.py`: optimization helpers and arbitrary-geometry tensor utilities.
- `quf.py`, `register_.py`: shared utilities and registrations.
- Notebooks (`*_pepo*.ipynb`, `draw_tn.ipynb`, `trotter_*.ipynb`, `plot*.ipynb`, etc.): experiments and visualizations.
- `cash/`: local cache/artifacts (ignored); `store/`, `store_state/`: reference data.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# For GPU JAX/PyTorch, install vendor wheels as needed.
```

## Notes
- `.gitattributes` marks notebooks as binary to avoid noisy diffs; use `nbdiff` or screenshots for review.
- `.gitignore` excludes checkpoints, caches, `cash/`, and `nohup.out`. Keep transient data there or outside the repo.
- Large generated data should stay out of version control or use Git LFS if required.
