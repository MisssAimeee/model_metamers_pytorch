"""
Automated sweep runner for AV metamer experiments.

This script runs `run_audio_visual_42_metamers.py` over a parameter grid,
then builds a single summary table + plot package.

Example:
    python run_av_metamer_grid.py \
        --out-root model_metamers_pytorch/results/av_grid_sweeps \
        --stimuli-dir model_metamers_pytorch/stimuli/MUSHRA_42_NATURAL

For a quick smoke test:
    python run_av_metamer_grid.py --limit-stim 2 --max-runs 2
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = REPO_ROOT / "run_audio_visual_42_metamers.py"
DEFAULT_OUT_ROOT = REPO_ROOT / "model_metamers_pytorch" / "results" / "av_grid_sweeps"
DEFAULT_STIMULI_DIR = REPO_ROOT / "model_metamers_pytorch" / "stimuli" / "MUSHRA_42_NATURAL"
DEFAULT_WEIGHTS = "/orcd/data/jhm/001/urops/aimee_yu/deep_avsr_weights/DeepAVSR_Weights/audio-visual.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid runner for AV metamer sweeps")
    p.add_argument("--base-script", type=Path, default=BASE_SCRIPT)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--stimuli-dir", type=Path, default=DEFAULT_STIMULI_DIR)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--target-layer", default="audioEncoder")
    p.add_argument("--limit-stim", type=int, default=0, help="Pass through to base script")
    p.add_argument("--max-runs", type=int, default=0, help="Optional cap for quick checks")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed runs and resume from incomplete ones",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--grid-json", type=Path, default=None, help="Optional JSON file with grid lists")
    return p.parse_args()


def default_grid() -> Dict[str, List]:
    return {
        "n_iters": [500, 1500],
        "step_size": [0.003, 0.01],
        "init": ["noise"],
        "gl_iters": [32, 64],
        "seed": [0, 1, 2],
        "eps": [1000.0],
    }


def load_grid(grid_json: Path | None) -> Dict[str, List]:
    if grid_json is None:
        return default_grid()
    data = json.loads(grid_json.read_text())
    required = ["n_iters", "step_size", "init", "gl_iters", "seed", "eps"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Grid JSON missing keys: {missing}")
    return data


def build_combos(grid: Dict[str, List]) -> List[Dict[str, object]]:
    keys = ["n_iters", "step_size", "init", "gl_iters", "seed", "eps"]
    combos = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, vals))
        run_name = (
            f"iter{combo['n_iters']}"
            f"_step{combo['step_size']}"
            f"_init{combo['init']}"
            f"_gl{combo['gl_iters']}"
            f"_seed{combo['seed']}"
            f"_eps{combo['eps']}"
        )
        combo["run_name"] = run_name.replace(".", "p")
        combos.append(combo)
    return combos


def category_from_stim(stim_folder: str) -> str:
    s = stim_folder.lower()
    speech_keys = ["speaking", "spanish", "french", "italian", "german", "hindi", "russian", "vocal"]
    music_keys = [
        "bluegrass", "violin", "cello", "jazz", "saxophone", "orchestra", "piano",
        "contemporary_rb", "band_music", "latin_music", "country_song", "rap_music",
        "acoustic_guitar",
    ]
    if any(k in s for k in speech_keys):
        return "speech"
    if any(k in s for k in music_keys):
        return "music"
    return "environment"


def _is_empty_text(s: str) -> bool:
    return s is None or str(s).strip() == ""


def summarize_run(csv_path: Path) -> Dict[str, float]:
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["match_distance"] = float(row["match_distance"])
            row["category"] = category_from_stim(row["stim_folder"])
            rows.append(row)

    result: Dict[str, float] = {}
    vals_all = [r["match_distance"] for r in rows]
    result["n_rows"] = len(rows)
    result["overall_mean"] = statistics.mean(vals_all)
    result["overall_median"] = statistics.median(vals_all)
    result["overall_std"] = statistics.pstdev(vals_all) if len(vals_all) > 1 else 0.0

    for cat in ["environment", "music", "speech"]:
        c_rows = [r for r in rows if r["category"] == cat]
        c_vals = [r["match_distance"] for r in c_rows]
        if c_vals:
            result[f"{cat}_n"] = len(c_vals)
            result[f"{cat}_mean"] = statistics.mean(c_vals)
            result[f"{cat}_median"] = statistics.median(c_vals)
            result[f"{cat}_std"] = statistics.pstdev(c_vals) if len(c_vals) > 1 else 0.0
            result[f"{cat}_metamer_empty_rate"] = sum(_is_empty_text(r["pred_metamer"]) for r in c_rows) / len(c_rows)
        else:
            result[f"{cat}_n"] = 0
            result[f"{cat}_mean"] = math.nan
            result[f"{cat}_median"] = math.nan
            result[f"{cat}_std"] = math.nan
            result[f"{cat}_metamer_empty_rate"] = math.nan

    result["env_minus_speech_mean"] = result["environment_mean"] - result["speech_mean"]
    return result


def write_summary_tables(out_root: Path, summary_rows: List[Dict[str, object]]) -> None:
    if not summary_rows:
        return
    summary_csv = out_root / "summary_runs.csv"
    fields = list(summary_rows[0].keys())
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_plots(out_root: Path, summary_rows: List[Dict[str, object]]) -> None:
    if not summary_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib not available, skipping plots: {exc}")
        return

    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Overall mean distance by run
    sorted_rows = sorted(summary_rows, key=lambda r: r["overall_mean"])
    x = list(range(len(sorted_rows)))
    y = [r["overall_mean"] for r in sorted_rows]
    labels = [r["run_name"] for r in sorted_rows]
    plt.figure(figsize=(12, 6))
    plt.bar(x, y)
    plt.xticks(x, labels, rotation=75, ha="right", fontsize=8)
    plt.ylabel("Overall mean match distance")
    plt.title("AV Metamer Sweep: Overall Mean Distance by Run")
    plt.tight_layout()
    plt.savefig(plots_dir / "overall_mean_distance_by_run.png", dpi=150)
    plt.close()

    # 2) Category means per run
    plt.figure(figsize=(12, 6))
    runs = [r["run_name"] for r in summary_rows]
    env = [r["environment_mean"] for r in summary_rows]
    music = [r["music_mean"] for r in summary_rows]
    speech = [r["speech_mean"] for r in summary_rows]
    plt.plot(runs, env, marker="o", label="environment")
    plt.plot(runs, music, marker="o", label="music")
    plt.plot(runs, speech, marker="o", label="speech")
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.ylabel("Mean match distance")
    plt.title("Category Mean Match Distance per Run")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "category_mean_distance_per_run.png", dpi=150)
    plt.close()

    # 3) Environment-minus-speech gap
    plt.figure(figsize=(12, 6))
    gap = [r["env_minus_speech_mean"] for r in summary_rows]
    plt.bar(runs, gap)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.ylabel("Environment mean - Speech mean")
    plt.title("Distance Gap (Environment - Speech) by Run")
    plt.tight_layout()
    plt.savefig(plots_dir / "env_minus_speech_gap.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    grid = load_grid(args.grid_json)
    combos = build_combos(grid)
    if args.max_runs > 0:
        combos = combos[: args.max_runs]

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "grid_spec.json").write_text(json.dumps(grid, indent=2))

    summary_rows: List[Dict[str, object]] = []
    print(f"[grid] planned runs: {len(combos)}")

    for i, combo in enumerate(combos, start=1):
        run_name = combo["run_name"]
        run_out = args.out_root / run_name
        comparison_csv = run_out / "comparison.csv"

        if args.resume and comparison_csv.exists():
            print(f"\n[run {i}/{len(combos)}] {run_name} (already complete, skipping)")
            summary = summarize_run(comparison_csv)
            summary_row = {**combo, **summary}
            summary_rows.append(summary_row)
            continue

        if args.resume and run_out.exists() and not comparison_csv.exists():
            # Interrupted run: clear partial outputs, then rerun cleanly.
            print(f"\n[run {i}/{len(combos)}] {run_name} (partial detected, resetting)")
            shutil.rmtree(run_out)

        cmd = [
            sys.executable,
            str(args.base_script),
            "--weights", args.weights,
            "--stimuli-dir", str(args.stimuli_dir),
            "--out-dir", str(run_out),
            "--target-layer", args.target_layer,
            "--n-iters", str(combo["n_iters"]),
            "--step-size", str(combo["step_size"]),
            "--eps", str(combo["eps"]),
            "--init", str(combo["init"]),
            "--gl-iters", str(combo["gl_iters"]),
            "--seed", str(combo["seed"]),
        ]
        if args.limit_stim > 0:
            cmd += ["--limit-stim", str(args.limit_stim)]

        print(f"\n[run {i}/{len(combos)}] {run_name}")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if completed.returncode != 0:
            print(f"[warn] run failed (exit={completed.returncode}): {run_name}")
            continue

        if not comparison_csv.exists():
            print(f"[warn] missing comparison.csv for {run_name}")
            continue

        summary = summarize_run(comparison_csv)
        summary_row = {**combo, **summary}
        summary_rows.append(summary_row)

    write_summary_tables(args.out_root, summary_rows)
    write_plots(args.out_root, summary_rows)
    print(f"\n[done] completed summaries for {len(summary_rows)} runs")
    print(f"[done] output package: {args.out_root}")


if __name__ == "__main__":
    main()

