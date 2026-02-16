#!/usr/bin/env bash
#
# UMAP parameter sweep: run visualize_umap.py across n_neighbors, min_dist, and metric.
# Saves full-size PNGs and _mini.png thumbnails per metric folder and generates index.html.
#
# Usage:
#   ./experiments/run_umap_sweep.sh [RESULTS_DIR [SOLUTIONS]]
#
# Optional env vars:
#   CUDA_VISIBLE_DEVICES=6     use GPU 6 (set before running)
#   SHOW_EDGES=1              pass --show-edges to visualize_umap.py
#   EDGE_ALPHA=0.01           pass --edge-alpha (default 0.08 when not set)
#
# Examples:
#   ./experiments/run_umap_sweep.sh
#   ./experiments/run_umap_sweep.sh exploration_results/DFS_10_4_ROOT_SA_8zzudzmv_20260211_085457 100
#   CUDA_VISIBLE_DEVICES=6 SHOW_EDGES=1 EDGE_ALPHA=0.01 ./experiments/run_umap_sweep.sh exploration_results/DFS_10_4_ROOT_SA_8zzudzmv_20260211_085457 50000
#
# Output: exploration_results/umap_sweep_<name>_<N>_solutions/
#   index.html
#   metric_euclidean/   n005_mindist0.0.png, n005_mindist0.0_mini.png, ...
#   metric_hamming/     ...
#   metric_cosine/      ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${1:-exploration_results/DFS_10_4_ROOT_SA_8zzudzmv_20260211_085457}"
SOLUTIONS="${2:-}"   # optional: e.g. 100 for checkpoint_100_solutions

# Parameter grids
N_NEIGHBORS=(5 15 50 200 500)
MIN_DIST=(0.0 0.1 0.5 0.9)
METRICS=(euclidean hamming cosine)

# Thumbnail width for mini PNGs
MINI_WIDTH=400

cd "$PROJECT_ROOT"

# Use project venv (uv sync) and headless matplotlib for batch
export MPLBACKEND=Agg
# shellcheck source=/dev/null
source "$PROJECT_ROOT/.venv/bin/activate"

# Resolve paths and get solution count for sweep dir name
RESULTS_DIR_ABS="$(cd "$RESULTS_DIR" && pwd)"
EXPLORATION_NAME="$(basename "$RESULTS_DIR_ABS")"

get_solution_count() {
  python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from pathlib import Path
from experiments.visualize_umap import resolve_results_path
from experiments.explore_degenerate_solutions import load_exploration_results
results_dir = Path('$RESULTS_DIR_ABS')
solutions = None if not '$SOLUTIONS' else int('$SOLUTIONS')
load_path = resolve_results_path(results_dir, solutions)
results = load_exploration_results(load_path)
print(len(results['unique_solutions']))
"
}

NUM_SOLUTIONS="$(get_solution_count | tail -n1)"
SWEEP_ROOT="$RESULTS_DIR_ABS/../umap_sweep_${EXPLORATION_NAME}_${NUM_SOLUTIONS}_solutions"
mkdir -p "$SWEEP_ROOT"

echo "=============================================="
echo "UMAP parameter sweep"
echo "  Results dir:   $RESULTS_DIR_ABS"
echo "  Solutions:     $NUM_SOLUTIONS"
echo "  Sweep output:  $SWEEP_ROOT"
echo "  Grid: n_neighbors=${N_NEIGHBORS[*]} min_dist=${MIN_DIST[*]} metric=${METRICS[*]}"
echo "=============================================="

# Build optional --solutions arg for the Python script
SOLUTIONS_ARG=()
[[ -n "$SOLUTIONS" ]] && SOLUTIONS_ARG=(--solutions "$SOLUTIONS")

# Optional visualization args (edges)
EXTRA_VIS_ARGS=()
[[ -n "${SHOW_EDGES:-}" ]] && EXTRA_VIS_ARGS+=(--show-edges)
[[ -n "${EDGE_ALPHA:-}" ]] && EXTRA_VIS_ARGS+=(--edge-alpha "${EDGE_ALPHA}")

# Create mini PNG from full PNG (same dir, base_mini.png)
make_mini_png() {
  local full="$1"
  local base="${full%.png}"
  local mini="${base}_mini.png"
  if command -v convert &>/dev/null; then
    convert "$full" -resize "${MINI_WIDTH}x${MINI_WIDTH}>" "$mini"
  else
    python -c "
from PIL import Image
from pathlib import Path
p = Path('$full')
img = Image.open(p)
w, h = img.size
if w > $MINI_WIDTH or h > $MINI_WIDTH:
    ratio = min($MINI_WIDTH / w, $MINI_WIDTH / h)
    img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
img.save('$mini')
"
  fi
  echo "  mini: $mini"
}

# Zero-pad n_neighbors for sortable filenames (n005, n015, n050, n200, n500)
n_str() { printf "n%03d" "$1"; }

# Run UMAP for one parameter set and save full + mini PNG
run_one() {
  local metric="$1"
  local n="$2"
  local mindist="$3"
  local n_pad
  n_pad="$(n_str "$n")"
  local metric_dir="$SWEEP_ROOT/metric_${metric}"
  local fname="${n_pad}_mindist${mindist}.png"
  local outpath="$metric_dir/$fname"
  mkdir -p "$metric_dir"
  echo "--- $metric n_neighbors=$n min_dist=$mindist -> $outpath"
  python "$SCRIPT_DIR/visualize_umap.py" \
    --results-dir "$RESULTS_DIR_ABS" \
    "${SOLUTIONS_ARG[@]}" \
    --output-file "$outpath" \
    --n-neighbors "$n" \
    --min-dist "$mindist" \
    --metric "$metric" \
    "${EXTRA_VIS_ARGS[@]}"
  make_mini_png "$outpath"
}

total=$(( ${#METRICS[@]} * ${#N_NEIGHBORS[@]} * ${#MIN_DIST[@]} ))
count=0
for metric in "${METRICS[@]}"; do
  for n in "${N_NEIGHBORS[@]}"; do
    for mindist in "${MIN_DIST[@]}"; do
      count=$(( count + 1 ))
      echo "[$count/$total]"
      run_one "$metric" "$n" "$mindist"
    done
  done
done

# Generate index.html
echo "Writing index.html ..."
INDEX="$SWEEP_ROOT/index.html"
cat > "$INDEX" << 'INDEXHEAD'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>UMAP parameter sweep</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem 2rem; background: #1a1a1a; color: #e0e0e0; }
    h1 { font-size: 1.4rem; }
    h2 { font-size: 1.1rem; margin-top: 1.5rem; color: #b0d0ff; }
    table { border-collapse: collapse; margin-bottom: 1rem; }
    th, td { border: 1px solid #444; padding: 4px 8px; text-align: center; }
    th { background: #333; }
    td a { display: block; }
    td img { max-width: 100%; height: auto; max-height: 200px; display: block; }
    td a:hover img { outline: 2px solid #6af; }
    .meta { color: #888; font-size: 0.9rem; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <h1>UMAP parameter sweep</h1>
  <p class="meta">Rows: min_dist (cluster density). Columns: n_neighbors (local vs global). Click thumbnail for full-size.</p>
INDEXHEAD

for metric in "${METRICS[@]}"; do
  metric_dir="metric_${metric}"
  echo "  <h2>Metric: ${metric}</h2>" >> "$INDEX"
  echo "  <table>" >> "$INDEX"
  # Header row: n_neighbors
  echo "    <tr><th>min_dist \\ n_neighbors</th>" >> "$INDEX"
  for n in "${N_NEIGHBORS[@]}"; do
    echo "    <th>${n}</th>" >> "$INDEX"
  done
  echo "    </tr>" >> "$INDEX"
  for mindist in "${MIN_DIST[@]}"; do
    echo "    <tr><th>${mindist}</th>" >> "$INDEX"
    for n in "${N_NEIGHBORS[@]}"; do
      n_pad="$(n_str "$n")"
      fname="${n_pad}_mindist${mindist}.png"
      mini_name="${n_pad}_mindist${mindist}_mini.png"
      echo "    <td><a href=\"${metric_dir}/${fname}\" target=\"_blank\"><img src=\"${metric_dir}/${mini_name}\" alt=\"${fname}\"></a></td>" >> "$INDEX"
    done
    echo "    </tr>" >> "$INDEX"
  done
  echo "  </table>" >> "$INDEX"
done

echo "</body>" >> "$INDEX"
echo "</html>" >> "$INDEX"

echo "Done. Open in browser: file://$INDEX"
echo "Sweep output: $SWEEP_ROOT"
