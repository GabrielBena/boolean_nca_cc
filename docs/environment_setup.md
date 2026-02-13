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

If you still use the conda `metabool` env on a machine, `conda activate metabool` continues to work; `uv sync` is an alternative that stays in sync via the shared `uv.lock` in the repo.
