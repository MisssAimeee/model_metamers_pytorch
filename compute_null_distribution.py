"""
Null-distribution validation for AV metamer results.

Compares:
  Within-pair  : D(activations(metamer_i), activations(ref_i))
                 The final match-distance from synthesis — already in comparison.csv.
  Cross-pair   : D(activations(metamer_i), activations(ref_j)), i ≠ j
                 A shuffled-label baseline.  If matching is meaningful the
                 within-pair distribution must be significantly lower.

Outputs to --out-dir:
  null_stats.json         mean / std / Cohen's d / Mann-Whitney U p-value
  null_histogram.png      overlapping histogram with KDE
  null_pairwise.csv       every cross-pair distance (i, j, distance)

Usage:
    cd /home/aimeeyu/mms_project
    python compute_null_distribution.py \\
        --results-dir model_metamers_pytorch/results/av_metamers_42_natural \\
        --stimuli-dir model_metamers_pytorch/stimuli/MUSHRA_42_NATURAL \\
        --out-dir model_metamers_pytorch/results/null_validation
"""

import sys, os
from pathlib import Path

# ── Conda bootstrap ───────────────────────────────────────────────────────────
_MMS_PY = "/home/aimeeyu/.conda/envs/mms/bin/python"
if Path(_MMS_PY).exists() and Path(sys.executable).resolve() != Path(_MMS_PY).resolve():
    print(f"[bootstrap] re-execing under {_MMS_PY}")
    os.execv(_MMS_PY, [_MMS_PY, *sys.argv])

import argparse, csv, json, math, statistics
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
import scipy.signal as scisig

REPO_ROOT = Path(__file__).parent
# Support running from inside model_metamers_pytorch/ OR from the parent dir
if (REPO_ROOT / "deep_avsr").exists():
    AV_DIR       = REPO_ROOT / "deep_avsr" / "audio_visual"
    _RESULTS_BASE = REPO_ROOT / "results"
    _STIMULI_BASE = REPO_ROOT / "stimuli"
else:
    AV_DIR        = REPO_ROOT / "model_metamers_pytorch" / "deep_avsr" / "audio_visual"
    _RESULTS_BASE = REPO_ROOT / "model_metamers_pytorch" / "results"
    _STIMULI_BASE = REPO_ROOT / "model_metamers_pytorch" / "stimuli"
sys.path.insert(0, str(AV_DIR))
from config import args as _cfg
from models.av_net import AVNet

AV_WEIGHTS = "/orcd/data/jhm/001/urops/aimee_yu/deep_avsr_weights/DeepAVSR_Weights/audio-visual.pt"
TARGET_SR  = 16_000
TARGET_LAYER = "audioEncoder"   # layer used in original synthesis


# ─────────────────────────────────────────────────────────────────────────────
# Layer activation extractor  (shared with run_layer_sweep.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_activations(net: AVNet, stft_batch: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Run the AVNet audio-only forward path up to *layer_name*.

    stft_batch : (T, B, F)  — same shape accepted by AVNetWithLatent
    Returns    : activation tensor at the requested layer
    """
    x = stft_batch.transpose(0, 1).transpose(1, 2)   # (B, F, T)
    x = net.audioConv(x)                               # (B, dModel, T/4)
    x = x.transpose(1, 2).transpose(0, 1)              # (T/4, B, dModel)
    x = net.positionalEncoding(x)

    if layer_name == "audioConv_out":
        return x

    n_enc = len(net.audioEncoder.layers)
    for i, enc_layer in enumerate(net.audioEncoder.layers):
        x = enc_layer(x)
        if layer_name == f"audioEncoder_L{i}":
            return x

    # Full encoder output (all 6 layers) — canonical "audioEncoder"
    if layer_name == "audioEncoder":
        return x

    # Joint decoder (audio-only path — no jointConv needed)
    for j, dec_layer in enumerate(net.jointDecoder.layers):
        x = dec_layer(x)
        if layer_name == f"jointDecoder_L{j}":
            return x

    if layer_name == "jointDecoder_out":
        return x

    # Final logits
    x = x.transpose(0, 1).transpose(1, 2)
    x = net.outputConv(x)
    x = x.transpose(1, 2).transpose(0, 1)
    x = F.log_softmax(x, dim=2)
    if layer_name == "logits":
        return x

    raise ValueError(
        f"Unknown layer '{layer_name}'. Valid: audioConv_out, "
        f"audioEncoder_L{{0-{n_enc-1}}}, audioEncoder, "
        f"jointDecoder_out, logits"
    )


def scale_invariant_l2(target: torch.Tensor, current: torch.Tensor) -> float:
    """Scale-invariant L2 distance (Feather et al. eq. 1), scalar."""
    num = (target - current).flatten().norm()
    den = target.flatten().norm().clamp_min(1e-8)
    return (num / den).item()


# ─────────────────────────────────────────────────────────────────────────────
# STFT helpers (mirrored from run_audio_visual_42_metamers.py)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_stft(wav_path: Path, device) -> torch.Tensor:
    """Return (T, 1, F) STFT magnitude tensor on *device*."""
    sr, data = wavfile.read(str(wav_path))
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        data = scisig.resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
        sr = TARGET_SR

    win_s, ovl_s = _cfg["STFT_WIN_LENGTH"], _cfg["STFT_OVERLAP"]
    min_len = int(sr * (win_s + 3 * (win_s - ovl_s)))
    if len(data) < min_len:
        data = np.pad(data, (0, min_len - len(data)), "constant")
    maxval = np.max(np.abs(data))
    if maxval > 0:
        data = data / maxval

    nperseg  = int(sr * win_s)
    noverlap = int(sr * ovl_s)
    _, _, stftVals = scisig.stft(
        data, sr,
        window=_cfg["STFT_WINDOW"],
        nperseg=nperseg, noverlap=noverlap,
        boundary=None, padded=False,
    )
    audInp = np.abs(stftVals).T   # (T_stft, F)

    reqInpLen = _cfg["MAIN_REQ_INPUT_LENGTH"]
    inpLen = int(np.ceil(len(audInp) / 4))
    lp = int(np.floor((4 * inpLen - len(audInp)) / 2))
    rp = int(np.ceil( (4 * inpLen - len(audInp)) / 2))
    audInp = np.pad(audInp, ((lp, rp), (0, 0)), "constant")
    if inpLen < reqInpLen:
        lp2 = int(np.floor((reqInpLen - inpLen) / 2))
        rp2 = int(np.ceil( (reqInpLen - inpLen) / 2))
        audInp = np.pad(audInp, ((4 * lp2, 4 * rp2), (0, 0)), "constant")

    return torch.from_numpy(audInp).unsqueeze(1).float().to(device)   # (T, 1, F)


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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Null-distribution validation")
    p.add_argument("--results-dir", type=Path,
                   default=_RESULTS_BASE / "av_metamers_42_natural",
                   help="Directory containing comparison.csv and stimuli sub-folders")
    p.add_argument("--stimuli-dir", type=Path,
                   default=_STIMULI_BASE / "MUSHRA_42_NATURAL")
    p.add_argument("--weights", default=AV_WEIGHTS)
    p.add_argument("--target-layer", default=TARGET_LAYER)
    p.add_argument("--out-dir", type=Path,
                   default=_RESULTS_BASE / "null_validation")
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib (headless)")
    p.add_argument("--save-pairwise", action="store_true",
                   help="Save full N×(N-1) cross-pair CSV (can be large)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    net = AVNet(
        _cfg["TX_NUM_FEATURES"], _cfg["TX_ATTENTION_HEADS"], _cfg["TX_NUM_LAYERS"],
        _cfg["PE_MAX_LENGTH"], _cfg["AUDIO_FEATURE_SIZE"],
        _cfg["TX_FEEDFORWARD_DIM"], _cfg["TX_DROPOUT"], _cfg["NUM_CLASSES"],
    )
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.eval().to(device)
    print(f"Loaded AVNet from {args.weights}")
    print(f"Target layer: {args.target_layer}\n")

    # ── Read comparison.csv ───────────────────────────────────────────────────
    csv_path = args.results_dir / "comparison.csv"
    entries = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            entries.append({
                "stim_folder": row["stim_folder"],
                "wav_file":    row["wav_file"],
                "within_dist": float(row["match_distance"]),
                "category":    category_from_stim(row["stim_folder"]),
                "metamer_wav": row["metamer_wav"],
            })
    N = len(entries)
    print(f"Loaded {N} entries from comparison.csv")

    # ── Compute activations for every reference wav ───────────────────────────
    print("\nComputing reference activations …")
    ref_acts  = []   # list of (D,) float32 numpy vectors
    met_acts  = []   # list of (D,) float32 numpy vectors

    for idx, e in enumerate(entries):
        # reference wav path
        ref_wav = args.stimuli_dir / e["stim_folder"] / e["wav_file"]
        if not ref_wav.exists():
            # try direct search in stimuli_dir for partial matches
            candidates = list(args.stimuli_dir.glob(f"**/{e['wav_file']}"))
            ref_wav = candidates[0] if candidates else None
        if ref_wav is None or not ref_wav.exists():
            print(f"  [warn] reference wav not found: {e['wav_file']} — skipping")
            ref_acts.append(None)
            met_acts.append(None)
            continue

        with torch.no_grad():
            stft = prepare_stft(ref_wav, device)
            act  = get_activations(net, stft, args.target_layer)  # (T', 1, D)
            ref_acts.append(act.flatten().cpu().numpy())

        # metamer stft — prefer saved tensor, fall back to wav
        met_wav = Path(e["metamer_wav"])
        stim_stft = (
            args.results_dir
            / e["stim_folder"]
            / Path(e["wav_file"]).stem
            / "metamer_stft.pt"
        )
        if stim_stft.exists():
            with torch.no_grad():
                met_stft = torch.load(stim_stft, map_location=device)
                act = get_activations(net, met_stft, args.target_layer)
                met_acts.append(act.flatten().cpu().numpy())
        elif met_wav.exists():
            with torch.no_grad():
                stft = prepare_stft(met_wav, device)
                act  = get_activations(net, stft, args.target_layer)
                met_acts.append(act.flatten().cpu().numpy())
        else:
            print(f"  [warn] metamer not found for {e['wav_file']} — skipping")
            met_acts.append(None)
            ref_acts[-1] = None   # discard paired ref too

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{N} done")

    # Filter entries where both ref and met activations were computed
    valid = [(i, e) for i, e in enumerate(entries)
             if ref_acts[i] is not None and met_acts[i] is not None]
    print(f"\n{len(valid)}/{N} valid pairs")
    if len(valid) < 2:
        print("Not enough valid pairs — aborting.")
        return

    idxs     = [v[0] for v in valid]
    entries_ = [v[1] for v in valid]
    ref_vecs = np.stack([ref_acts[i] for i in idxs])   # (M, D)
    met_vecs = np.stack([met_acts[i] for i in idxs])   # (M, D)
    M = len(idxs)

    # Within-pair distances (already in CSV — fast path)
    within_dists = np.array([e["within_dist"] for e in entries_])

    # ── Cross-pair distances (vectorised) ─────────────────────────────────────
    print("\nComputing cross-pair distances (vectorised) …")
    from scipy.spatial.distance import cdist
    raw_l2   = cdist(met_vecs, ref_vecs, "euclidean")     # (M, M)
    ref_norms = np.linalg.norm(ref_vecs, axis=1)           # (M,)
    cross_mat = raw_l2 / ref_norms[None, :]                # (M, M), broadcast
    # Exclude diagonal (within-pair)
    off_mask      = ~np.eye(M, dtype=bool)
    cross_dists   = cross_mat[off_mask]                    # (M*(M-1),)

    # ── Statistics ─────────────────────────────────────────────────────────────
    from scipy import stats as sp_stats

    w_mean, w_std = float(within_dists.mean()), float(within_dists.std())
    c_mean, c_std = float(cross_dists.mean()),  float(cross_dists.std())
    pooled_std    = math.sqrt((w_std**2 + c_std**2) / 2)
    cohens_d      = (c_mean - w_mean) / (pooled_std + 1e-12)
    u_stat, p_val = sp_stats.mannwhitneyu(within_dists, cross_dists, alternative="less")

    print(f"\n{'─'*50}")
    print(f"  Within-pair   mean={w_mean:.4f}  std={w_std:.4f}  n={M}")
    print(f"  Cross-pair    mean={c_mean:.4f}  std={c_std:.4f}  n={len(cross_dists)}")
    print(f"  Cohen's d     = {cohens_d:.3f}  (>0.8 = large effect)")
    print(f"  Mann-Whitney U = {u_stat:.0f},  p = {p_val:.2e}")
    print(f"{'─'*50}")

    # Per-category breakdown
    cat_stats = {}
    for cat in ["environment", "music", "speech"]:
        cat_idx = [i for i, e in enumerate(entries_) if e["category"] == cat]
        if not cat_idx:
            continue
        w_cat = within_dists[cat_idx]
        # Cross-pair for this category: met_i vs ref_j where j is same cat, i != j
        c_idx_mat = np.ix_(cat_idx, cat_idx)
        sub_mat   = cross_mat[c_idx_mat]
        off       = ~np.eye(len(cat_idx), dtype=bool)
        c_cat     = sub_mat[off]
        cat_stats[cat] = {
            "n": len(cat_idx),
            "within_mean": float(w_cat.mean()),
            "within_std":  float(w_cat.std()),
            "cross_mean":  float(c_cat.mean()) if len(c_cat) else None,
            "cross_std":   float(c_cat.std())  if len(c_cat) else None,
        }
        print(f"  [{cat}] within={w_cat.mean():.4f}, cross={c_cat.mean():.4f}")

    # ── Save stats ────────────────────────────────────────────────────────────
    results_dict = {
        "target_layer":        args.target_layer,
        "n_valid_pairs":       M,
        "within_mean":         w_mean,
        "within_std":          w_std,
        "cross_mean":          c_mean,
        "cross_std":           c_std,
        "cohens_d":            cohens_d,
        "mann_whitney_u":      float(u_stat),
        "mann_whitney_p":      float(p_val),
        "significant_p005":    bool(p_val < 0.05),
        "per_category":        cat_stats,
    }
    stats_path = args.out_dir / "null_stats.json"
    stats_path.write_text(json.dumps(results_dict, indent=2))
    print(f"\nStats saved → {stats_path}")

    # ── Save pairwise CSV (optional) ──────────────────────────────────────────
    if args.save_pairwise:
        pairwise_path = args.out_dir / "null_pairwise.csv"
        with pairwise_path.open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["met_idx", "ref_idx", "met_stim", "ref_stim", "distance", "pair_type"])
            for i in range(M):
                # within-pair
                wr.writerow([i, i, entries_[i]["stim_folder"], entries_[i]["stim_folder"],
                             f"{within_dists[i]:.6f}", "within"])
                # cross-pair
                for j in range(M):
                    if i != j:
                        wr.writerow([i, j, entries_[i]["stim_folder"], entries_[j]["stim_folder"],
                                     f"{cross_mat[i,j]:.6f}", "cross"])
        print(f"Pairwise CSV saved → {pairwise_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy.stats import gaussian_kde

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"Null-distribution validation  |  layer: {args.target_layer}  |  "
                f"n={M} pairs\n"
                f"Cohen's d = {cohens_d:.2f}   p = {p_val:.2e}",
                fontsize=12,
            )

            # ── Subplot 1: histogram + KDE ────────────────────────────────────
            ax = axes[0]
            bins = np.linspace(0, max(within_dists.max(), np.percentile(cross_dists, 99)) * 1.05, 50)
            ax.hist(within_dists, bins=bins, alpha=0.55, color="#2196F3", label="Within-pair (metamer_i vs ref_i)")
            ax.hist(cross_dists,  bins=bins, alpha=0.40, color="#FF5722", label="Cross-pair  (metamer_i vs ref_j, i≠j)")
            # KDE overlays
            for vals, col in [(within_dists, "#0D47A1"), (cross_dists, "#BF360C")]:
                if len(vals) > 1:
                    kde = gaussian_kde(vals, bw_method=0.3)
                    xs  = np.linspace(bins[0], bins[-1], 300)
                    ax.plot(xs, kde(xs) * len(vals) * (bins[1] - bins[0]), color=col, lw=2)
            ax.axvline(w_mean, color="#0D47A1", lw=1.5, ls="--", label=f"Within mean {w_mean:.3f}")
            ax.axvline(c_mean, color="#BF360C", lw=1.5, ls="--", label=f"Cross mean  {c_mean:.3f}")
            ax.set_xlabel("Scale-invariant L2 distance")
            ax.set_ylabel("Count")
            ax.set_title("Overall distribution")
            ax.legend(fontsize=8)

            # ── Subplot 2: per-category within vs cross ───────────────────────
            ax2 = axes[1]
            cats      = list(cat_stats.keys())
            w_means   = [cat_stats[c]["within_mean"] for c in cats]
            c_means   = [cat_stats[c]["cross_mean"]  for c in cats]
            w_stds    = [cat_stats[c]["within_std"]  for c in cats]
            c_stds    = [cat_stats[c]["cross_std"]   for c in cats]
            x_pos     = np.arange(len(cats))
            width     = 0.35
            ax2.bar(x_pos - width/2, w_means, width, yerr=w_stds, capsize=4,
                    color="#2196F3", alpha=0.8, label="Within-pair")
            ax2.bar(x_pos + width/2, c_means, width, yerr=c_stds, capsize=4,
                    color="#FF5722", alpha=0.8, label="Cross-pair")
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(cats)
            ax2.set_ylabel("Mean distance")
            ax2.set_title("Per-category comparison")
            ax2.legend()

            plt.tight_layout()
            plot_path = args.out_dir / "null_histogram.png"
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Plot saved → {plot_path}")
        except Exception as exc:
            print(f"[warn] plot failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
