"""
Generate all static assets for the Deep AVSR Metamers demo website.

Reads:
  model_metamers_pytorch/results/av_metamers_42_natural/comparison.csv
  model_metamers_pytorch/results/layer_sweep/<layer>/comparison.csv  (optional)
  model_metamers_pytorch/results/null_validation/null_stats.json     (optional)
  model_metamers_pytorch/stimuli/MUSHRA_42_NATURAL/**/*.wav

Writes to --out-dir (default: website/):
  index.html          (copied from this repo)
  data.json           (all metadata the JS app needs)
  audio/originals/    symlinks to reference wavs
  audio/metamers/     symlinks to metamer wavs (one folder per layer)
  spectrograms/       PNG spectrograms for every wav

Usage:
    cd /home/aimeeyu/mms_project
    python generate_website_assets.py
    # Then serve locally:
    python -m http.server 8080 --directory website
"""

import sys, os, csv, json, math, argparse, shutil
from pathlib import Path

_MMS_PY = "/home/aimeeyu/.conda/envs/mms/bin/python"
if Path(_MMS_PY).exists() and Path(sys.executable).resolve() != Path(_MMS_PY).resolve():
    print(f"[bootstrap] re-execing under {_MMS_PY}")
    os.execv(_MMS_PY, [_MMS_PY, *sys.argv])

import numpy as np
from scipy.io import wavfile
import scipy.signal as scisig

REPO_ROOT = Path(__file__).parent
if (REPO_ROOT / "deep_avsr").exists():
    _RESULTS_BASE = REPO_ROOT / "results"
    _STIMULI_BASE = REPO_ROOT / "stimuli"
else:
    _RESULTS_BASE = REPO_ROOT / "model_metamers_pytorch" / "results"
    _STIMULI_BASE = REPO_ROOT / "model_metamers_pytorch" / "stimuli"
TARGET_SR = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Category helper
# ─────────────────────────────────────────────────────────────────────────────

def category_from_stim(stim_folder: str) -> str:
    s = stim_folder.lower()
    speech_keys = ["speaking", "spanish", "french", "italian", "german", "hindi", "russian", "vocal"]
    music_keys  = ["bluegrass", "violin", "cello", "jazz", "saxophone", "orchestra", "piano",
                   "contemporary_rb", "band_music", "latin_music", "country_song", "rap_music",
                   "acoustic_guitar"]
    if any(k in s for k in speech_keys):
        return "speech"
    if any(k in s for k in music_keys):
        return "music"
    return "environment"


CATEGORY_COLORS = {
    "environment": "#4CAF50",
    "music":       "#2196F3",
    "speech":      "#FF5722",
}


def pretty_name(stim_folder: str) -> str:
    """Turn '14_stim33_bluegrass_orig_splits' → 'Bluegrass (stim 33)'."""
    parts = stim_folder.replace("_orig_splits", "").split("_")
    # Remove leading index and 'stim<num>'
    name_parts = []
    skip_next = False
    for p in parts:
        if skip_next:
            skip_next = False
            continue
        if p.isdigit():
            continue
        if p.startswith("stim") and p[4:].isdigit():
            continue
        name_parts.append(p)
    return " ".join(name_parts).title() or stim_folder


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram generation
# ─────────────────────────────────────────────────────────────────────────────

def make_spectrogram(wav_path: Path, out_png: Path, title: str = ""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sr, data = wavfile.read(str(wav_path))
        data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        maxval = np.abs(data).max()
        if maxval > 0:
            data = data / maxval

        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.specgram(data, Fs=sr, cmap="inferno", vmin=-80, vmax=-20, NFFT=512, noverlap=384)
        ax.set_ylim(0, min(8000, sr // 2))
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Hz", fontsize=8)
        ax.tick_params(labelsize=7)
        if title:
            ax.set_title(title, fontsize=8, pad=2)
        plt.tight_layout(pad=0.3)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_png), dpi=100, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  [warn] spectrogram failed for {wav_path.name}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Symlink helper
# ─────────────────────────────────────────────────────────────────────────────

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(str(src), str(dst))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-results-dir", type=Path,
                   default=_RESULTS_BASE / "av_metamers_42_natural")
    p.add_argument("--layer-sweep-dir", type=Path,
                   default=_RESULTS_BASE / "layer_sweep")
    p.add_argument("--null-stats", type=Path,
                   default=_RESULTS_BASE / "null_validation" / "null_stats.json")
    p.add_argument("--stimuli-dir", type=Path,
                   default=_STIMULI_BASE / "MUSHRA_42_NATURAL")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "website")
    p.add_argument("--no-spectrograms", action="store_true",
                   help="Skip spectrogram generation (faster rebuild)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    (out / "spectrograms").mkdir(exist_ok=True)

    # ── Load base comparison.csv ──────────────────────────────────────────────
    base_csv = args.base_results_dir / "comparison.csv"
    entries  = []
    with base_csv.open(newline="") as f:
        entries = list(csv.DictReader(f))
    print(f"Loaded {len(entries)} entries from {base_csv}")

    # ── Discover available layers (layer sweep) ───────────────────────────────
    layer_dirs = {}   # layer_name → Path
    layer_dirs["audioEncoder"] = args.base_results_dir   # baseline layer
    if args.layer_sweep_dir.exists():
        for d in sorted(args.layer_sweep_dir.iterdir()):
            if d.is_dir() and (d / "comparison.csv").exists():
                layer_dirs[d.name] = d
        print(f"Found {len(layer_dirs)-1} additional layer sweep directories")
    else:
        print("[info] No layer sweep directory found — only audioEncoder layer available")

    # Load layer sweep CSVs
    layer_entries = {}   # layer_name → {(stim_folder, wav_file): dist}
    for layer_name, layer_dir in layer_dirs.items():
        lcsv = layer_dir / "comparison.csv"
        if not lcsv.exists():
            continue
        ld = {}
        with lcsv.open(newline="") as f:
            for row in csv.DictReader(f):
                ld[(row["stim_folder"], row["wav_file"])] = {
                    "distance":    float(row["match_distance"]),
                    "metamer_wav": row["metamer_wav"],
                }
        layer_entries[layer_name] = ld

    # ── Load null stats if available ──────────────────────────────────────────
    null_stats = None
    if args.null_stats.exists():
        null_stats = json.loads(args.null_stats.read_text())
        print(f"Loaded null stats from {args.null_stats}")

    # ── Build data structure ──────────────────────────────────────────────────
    # Group entries by stim_folder
    by_stim = {}
    for e in entries:
        by_stim.setdefault(e["stim_folder"], []).append(e)

    sounds_data = []
    sound_idx   = 0

    for stim_folder, clips in sorted(by_stim.items(), key=lambda x: x[0]):
        category = category_from_stim(stim_folder)
        name     = pretty_name(stim_folder)

        clips_data = []
        for clip_entry in clips:
            wav_file = clip_entry["wav_file"]
            clip_id  = Path(wav_file).stem

            # Original wav symlink
            orig_src = args.stimuli_dir / stim_folder / wav_file
            if not orig_src.exists():
                candidates = list(args.stimuli_dir.glob(f"**/{wav_file}"))
                orig_src   = candidates[0] if candidates else None
            orig_audio_rel = None
            orig_spec_rel  = None
            if orig_src and orig_src.exists():
                dst_audio = out / "audio" / "originals" / stim_folder / wav_file
                safe_symlink(orig_src, dst_audio)
                orig_audio_rel = str(dst_audio.relative_to(out))

                spec_path = out / "spectrograms" / "originals" / stim_folder / f"{clip_id}.png"
                if not args.no_spectrograms and not spec_path.exists():
                    make_spectrogram(orig_src, spec_path, title=f"{name} — original")
                orig_spec_rel = str(spec_path.relative_to(out)) if spec_path.exists() else None

            # Metamer wavs per layer
            layers_clip = {}
            for layer_name, ld in layer_entries.items():
                key  = (stim_folder, wav_file)
                info = ld.get(key)
                if info is None:
                    continue
                met_src = Path(info["metamer_wav"])
                if not met_src.exists():
                    continue
                dst_met = out / "audio" / "metamers" / layer_name / stim_folder / wav_file
                safe_symlink(met_src, dst_met)
                met_audio_rel = str(dst_met.relative_to(out))

                spec_met = out / "spectrograms" / "metamers" / layer_name / stim_folder / f"{clip_id}.png"
                if not args.no_spectrograms and not spec_met.exists():
                    make_spectrogram(met_src, spec_met, title=f"{name} — {layer_name}")
                spec_met_rel = str(spec_met.relative_to(out)) if spec_met.exists() else None

                layers_clip[layer_name] = {
                    "audio":    met_audio_rel,
                    "spectrogram": spec_met_rel,
                    "distance": info["distance"],
                }

            clips_data.append({
                "id":           clip_id,
                "wav_file":     wav_file,
                "orig_audio":   orig_audio_rel,
                "orig_spec":    orig_spec_rel,
                "layers":       layers_clip,
            })

        sounds_data.append({
            "id":       stim_folder,
            "name":     name,
            "category": category,
            "color":    CATEGORY_COLORS[category],
            "clips":    clips_data,
        })
        sound_idx += 1
        print(f"  [{sound_idx}] {name} ({category})  {len(clips_data)} clips")

    # ── Null validation stats ─────────────────────────────────────────────────
    null_info = None
    if null_stats:
        null_info = {
            "within_mean":    null_stats.get("within_mean"),
            "cross_mean":     null_stats.get("cross_mean"),
            "cohens_d":       null_stats.get("cohens_d"),
            "p_value":        null_stats.get("mann_whitney_p"),
            "significant":    null_stats.get("significant_p005"),
            "per_category":   null_stats.get("per_category", {}),
        }

    # ── Category summary ──────────────────────────────────────────────────────
    cat_summary = {}
    for cat in ["environment", "music", "speech"]:
        cat_sounds = [s for s in sounds_data if s["category"] == cat]
        all_dists  = []
        for s in cat_sounds:
            for clip in s["clips"]:
                d = (clip["layers"].get("audioEncoder") or {}).get("distance")
                if d is not None:
                    all_dists.append(d)
        cat_summary[cat] = {
            "n_sounds": len(cat_sounds),
            "n_clips":  sum(len(s["clips"]) for s in cat_sounds),
            "mean_dist": float(np.mean(all_dists)) if all_dists else None,
            "std_dist":  float(np.std(all_dists))  if all_dists else None,
        }

    # ── Write data.json ───────────────────────────────────────────────────────
    data = {
        "title":        "Deep AVSR Audio Metamers",
        "subtitle":     "Metamer synthesis via the deep_avsr transformer backbone",
        "layers":       list(layer_entries.keys()),
        "categories":   ["environment", "music", "speech"],
        "cat_summary":  cat_summary,
        "null_validation": null_info,
        "sounds":       sounds_data,
    }
    data_path = out / "data.json"
    data_path.write_text(json.dumps(data, indent=2))
    print(f"\ndata.json → {data_path}")

    # Copy index.html if it exists in repo root
    html_src = REPO_ROOT / "website" / "index.html"
    if html_src.exists() and html_src.resolve() != (out / "index.html").resolve():
        shutil.copy2(str(html_src), str(out / "index.html"))

    print(f"\nAssets ready in {out}")
    print(f"Serve with:  python -m http.server 8080 --directory {out}")


if __name__ == "__main__":
    main()
