# Environment setup

The project uses a lockfile for reproducible installs. On any machine (including a second server synced from the same repo):

1. **Install [uv](https://docs.astral.sh/uv/)** if needed (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`).

2. **From the repo root:**
   ```bash
   uv sync
   ```
   This creates a `.venv` and installs the project and all dependencies from `uv.lock`.

3. **Activate the environment and run:**
   ```bash
   source .venv/bin/activate   # Linux/macOS
   python -c "import jax; print('ok')"
   ```

To get only the main dependencies (no dev/notebook extras), use:
```bash
uv sync --no-dev
```

**GPU (NVIDIA CUDA):** Default `uv sync` installs JAX with CPU-only jaxlib. On a machine with NVIDIA drivers (≥ 525 for CUDA 12), use:
```bash
uv sync --extra cuda
```
This installs `jax[cuda12]` so JAX can use the GPU. For CUDA 13, edit `pyproject.toml` optional-dependencies `cuda` to use `jax[cuda13]` and run `uv sync --extra cuda`.

**Notebooks / plotting extras:** For jupyter, seaborn, and dev tools:
```bash
uv sync --extra notebooks
# or
uv sync --extra all
```

Main dependencies already include matplotlib and umap-learn for scripts like `experiments/visualize_umap.py` and `experiments/run_umap_sweep.sh`.

If you still use the conda `metabool` env on a machine, `conda activate metabool` continues to work; `uv sync` is an alternative that stays in sync via the shared `uv.lock` in the repo.
