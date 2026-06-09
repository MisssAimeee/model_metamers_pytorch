"""
Layer-sweep metamer generation for AV metamers.

Generates metamers targeting each intermediate representation of AVNet:
  audioConv_out        post-Conv1d + PositionalEncoding (shallow embedding)
  audioEncoder_L0..L5  after each individual TransformerEncoder layer
  audioEncoder         full 6-layer audio encoder output  (default)
  jointDecoder_out     after the 6-layer joint decoder
  logits               final log-softmax output

Usage (all layers, 42 stimuli):
    cd /home/aimeeyu/mms_project
    python run_layer_sweep.py --layers all

Quick smoke-test (2 stimuli, shallow layers only):
    python run_layer_sweep.py --layers audioConv_out,audioEncoder_L0 --limit-stim 2 --n-iters 100

Output structure:
    <out-root>/
      audioConv_out/
        comparison.csv
        <stim>/<clip>/metamer_audio.wav   metamer_stft.pt
      audioEncoder_L0/ …
      …
      layer_sweep_summary.csv   (layer × category × mean-distance)
"""

import sys, os, csv, math, argparse, json, random
from pathlib import Path

_MMS_PY = "/home/aimeeyu/.conda/envs/mms/bin/python"
if Path(_MMS_PY).exists() and Path(sys.executable).resolve() != Path(_MMS_PY).resolve():
    print(f"[bootstrap] re-execing under {_MMS_PY}")
    os.execv(_MMS_PY, [_MMS_PY, *sys.argv])

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile
import scipy.signal as scisig

REPO_ROOT = Path(__file__).parent
if (REPO_ROOT / "deep_avsr").exists():
    AV_DIR        = REPO_ROOT / "deep_avsr" / "audio_visual"
    _RESULTS_BASE = REPO_ROOT / "results"
    _STIMULI_BASE = REPO_ROOT / "stimuli"
else:
    AV_DIR        = REPO_ROOT / "model_metamers_pytorch" / "deep_avsr" / "audio_visual"
    _RESULTS_BASE = REPO_ROOT / "model_metamers_pytorch" / "results"
    _STIMULI_BASE = REPO_ROOT / "model_metamers_pytorch" / "stimuli"
sys.path.insert(0, str(AV_DIR))
from config import args as _cfg
from models.av_net import AVNet
from utils.decoders import ctc_greedy_decode

AV_WEIGHTS  = "/orcd/data/jhm/001/urops/aimee_yu/deep_avsr_weights/DeepAVSR_Weights/audio-visual.pt"
STIMULI_DIR = _STIMULI_BASE / "MUSHRA_42_NATURAL"
OUT_ROOT    = _RESULTS_BASE / "layer_sweep"
TARGET_SR   = 16_000

ALL_LAYERS  = [
    "audioConv_out",
    "audioEncoder_L0", "audioEncoder_L1", "audioEncoder_L2",
    "audioEncoder_L3", "audioEncoder_L4", "audioEncoder_L5",
    "jointDecoder_out",
    "logits",
]


# ─────────────────────────────────────────────────────────────────────────────
# Layer activation extractor
# ─────────────────────────────────────────────────────────────────────────────

def get_activations(net: AVNet, stft_batch: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Run the AVNet audio-only branch to *layer_name* and return activations.

    stft_batch : (T, B, F)  — STFT magnitude batch
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

    if layer_name == "audioEncoder":
        return x   # full encoder = L5 alias

    n_dec = len(net.jointDecoder.layers)
    for j, dec_layer in enumerate(net.jointDecoder.layers):
        x = dec_layer(x)
        if layer_name == f"jointDecoder_L{j}":
            return x

    if layer_name == "jointDecoder_out":
        return x

    # logits
    x = x.transpose(0, 1).transpose(1, 2)
    x = net.outputConv(x)
    x = x.transpose(1, 2).transpose(0, 1)
    x = F.log_softmax(x, dim=2)
    if layer_name == "logits":
        return x

    valid = (["audioConv_out"] + [f"audioEncoder_L{i}" for i in range(n_enc)]
             + ["audioEncoder", "jointDecoder_out", "logits"])
    raise ValueError(f"Unknown layer '{layer_name}'. Valid: {valid}")


def metamer_loss(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    num = torch.norm((target - current).flatten(1), dim=1)
    den = torch.norm(target.flatten(1), dim=1).clamp_min(1e-8)
    return (num / den).mean()


# ─────────────────────────────────────────────────────────────────────────────
# STFT helpers
# ─────────────────────────────────────────────────────────────────────────────

def prepare_stft(wav_path: Path, device) -> torch.Tensor:
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
    audInp = np.abs(stftVals).T
    reqInpLen = _cfg["MAIN_REQ_INPUT_LENGTH"]
    inpLen = int(np.ceil(len(audInp) / 4))
    lp = int(np.floor((4 * inpLen - len(audInp)) / 2))
    rp = int(np.ceil( (4 * inpLen - len(audInp)) / 2))
    audInp = np.pad(audInp, ((lp, rp), (0, 0)), "constant")
    if inpLen < reqInpLen:
        lp2 = int(np.floor((reqInpLen - inpLen) / 2))
        rp2 = int(np.ceil( (reqInpLen - inpLen) / 2))
        audInp = np.pad(audInp, ((4 * lp2, 4 * rp2), (0, 0)), "constant")
    return torch.from_numpy(audInp).unsqueeze(1).float().to(device)


def stft_to_wav(stft_mag: torch.Tensor, out_path: Path, gl_iters: int = 64):
    nperseg  = int(TARGET_SR * _cfg["STFT_WIN_LENGTH"])
    noverlap = int(TARGET_SR * _cfg["STFT_OVERLAP"])
    hop_len  = nperseg - noverlap
    mag = stft_mag.detach().cpu().float()
    if mag.dim() == 3:
        mag = mag.squeeze(1)
    mag = mag.T.unsqueeze(0)   # (1, F, T)
    gl = torchaudio.transforms.GriffinLim(
        n_fft=nperseg, hop_length=hop_len, win_length=nperseg,
        window_fn=torch.hann_window, n_iter=gl_iters, power=1.0,
    )
    wav = gl(mag).squeeze(0)
    wav = wav / wav.abs().max().clamp_min(1e-8)
    wav_int16 = (wav.clamp(-1, 1) * 32767).short().numpy()
    from scipy.io import wavfile as _wf
    _wf.write(str(out_path), TARGET_SR, wav_int16)


# ─────────────────────────────────────────────────────────────────────────────
# PGD synthesis (layer-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_metamer(
    net: AVNet,
    ref_stft: torch.Tensor,
    layer_name: str,
    n_iters: int,
    step_size: float,
    eps: float,
    init: str,
    seed_offset: int = 0,
) -> tuple:
    """PGD in STFT-magnitude space matching *layer_name* activations of ref."""
    with torch.no_grad():
        A_ref = get_activations(net, ref_stft, layer_name).detach()

    if init == "ref":
        x = ref_stft.clone()
    else:
        rms = ref_stft.pow(2).mean().sqrt().clamp_min(1e-4)
        x   = torch.randn_like(ref_stft).abs() * rms

    history = []
    for it in range(n_iters):
        x = x.detach().requires_grad_(True)
        acts = get_activations(net, x, layer_name)
        loss = metamer_loss(A_ref, acts)
        loss.backward()

        with torch.no_grad():
            g      = x.grad
            g_norm = g / (g.flatten().norm() + 1e-12)
            x_new  = x - step_size * g_norm
            delta  = x_new - ref_stft
            d_norm = delta.flatten().norm()
            if d_norm > eps:
                delta = delta * (eps / d_norm)
            x = (ref_stft + delta).clamp_min(0.0)

        if it % 100 == 0:
            history.append(loss.item())
            print(f"    iter {it:4d}  loss {loss.item():.5f}")

    with torch.no_grad():
        final = get_activations(net, x, layer_name)
        dist  = metamer_loss(A_ref, final).item()

    return x, history, dist


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
    p = argparse.ArgumentParser(description="Layer-sweep metamer generation")
    p.add_argument("--weights",     default=AV_WEIGHTS)
    p.add_argument("--stimuli-dir", type=Path, default=STIMULI_DIR)
    p.add_argument("--out-root",    type=Path, default=OUT_ROOT)
    p.add_argument("--layers",       default="all",
                   help="Comma-separated list of layers, or 'all'")
    p.add_argument("--n-iters",      type=int,   default=500)
    p.add_argument("--step-size",    type=float, default=0.01)
    p.add_argument("--eps",          type=float, default=1000.0)
    p.add_argument("--init",         choices=["noise", "ref"], default="noise")
    p.add_argument("--gl-iters",     type=int,   default=32)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--limit-stim",   type=int,   default=0,
                   help="Limit number of stimulus folders (0 = all)")
    p.add_argument("--resume",       action="store_true",
                   help="Skip already-completed per-layer comparisons")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Run one layer
# ─────────────────────────────────────────────────────────────────────────────

def run_one_layer(
    net: AVNet,
    layer_name: str,
    stim_dirs: list,
    out_layer_dir: Path,
    args,
    device,
) -> list:
    """Synthesise metamers for all stim_dirs targeting layer_name.

    Returns list of dicts (one per wav) for summary aggregation.
    """
    out_layer_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = out_layer_dir / "comparison.csv"
    fieldnames = ["stim_folder", "wav_file", "layer", "match_distance",
                  "category", "metamer_wav"]

    completed_wavs = set()
    if args.resume and csv_path.exists():
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                completed_wavs.add((row["stim_folder"], row["wav_file"]))
        print(f"  [resume] found {len(completed_wavs)} completed wavs")

    total_wavs = sum(len(list(d.glob("*.wav"))) for d in stim_dirs)
    done_count  = 0
    rows        = []

    with csv_path.open("a" if args.resume else "w", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
        if not args.resume:
            writer.writeheader()

        for stim_dir in stim_dirs:
            for wav_path in sorted(stim_dir.glob("*.wav")):
                done_count += 1
                key = (stim_dir.name, wav_path.name)
                if key in completed_wavs:
                    print(f"  [skip] {stim_dir.name}/{wav_path.name}")
                    continue

                print(f"  [{done_count}/{total_wavs}] {stim_dir.name}/{wav_path.name}")

                local_seed = args.seed + done_count
                random.seed(local_seed)
                np.random.seed(local_seed)
                torch.manual_seed(local_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(local_seed)

                ref_stft = prepare_stft(wav_path, device)
                print(f"    synthesising ({args.n_iters} iters, layer={layer_name}) …")
                met_stft, history, dist = synthesize_metamer(
                    net, ref_stft, layer_name,
                    n_iters=args.n_iters, step_size=args.step_size,
                    eps=args.eps, init=args.init,
                )

                stim_out = out_layer_dir / stim_dir.name / wav_path.stem
                stim_out.mkdir(parents=True, exist_ok=True)
                met_wav = stim_out / "metamer_audio.wav"
                stft_to_wav(met_stft, met_wav, args.gl_iters)
                torch.save(met_stft.cpu(), stim_out / "metamer_stft.pt")
                (stim_out / "match_distance.txt").write_text(
                    f"{layer_name}\t{dist:.6f}\n"
                )
                print(f"    match distance: {dist:.4f}  →  {met_wav}")

                row = {
                    "stim_folder":    stim_dir.name,
                    "wav_file":       wav_path.name,
                    "layer":          layer_name,
                    "match_distance": f"{dist:.6f}",
                    "category":       category_from_stim(stim_dir.name),
                    "metamer_wav":    str(met_wav),
                }
                writer.writerow(row)
                csv_fh.flush()
                rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.layers.strip().lower() == "all":
        layers = ALL_LAYERS
    else:
        layers = [l.strip() for l in args.layers.split(",")]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Layers : {layers}\n")

    net = AVNet(
        _cfg["TX_NUM_FEATURES"], _cfg["TX_ATTENTION_HEADS"], _cfg["TX_NUM_LAYERS"],
        _cfg["PE_MAX_LENGTH"], _cfg["AUDIO_FEATURE_SIZE"],
        _cfg["TX_FEEDFORWARD_DIM"], _cfg["TX_DROPOUT"], _cfg["NUM_CLASSES"],
    )
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.eval().to(device)
    print(f"Loaded AVNet from {args.weights}\n")

    stim_dirs = sorted(
        [d for d in args.stimuli_dir.iterdir() if d.is_dir()],
        key=lambda d: int(d.name.split("_")[0]),
    )
    if args.limit_stim > 0:
        stim_dirs = stim_dirs[:args.limit_stim]
    print(f"Stimuli: {len(stim_dirs)} folders\n")

    args.out_root.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_cfg = {
        "layers": layers, "n_iters": args.n_iters, "step_size": args.step_size,
        "eps": args.eps, "init": args.init, "gl_iters": args.gl_iters, "seed": args.seed,
        "limit_stim": args.limit_stim,
    }
    (args.out_root / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    all_rows = []
    for layer_name in layers:
        print(f"\n{'═'*60}")
        print(f"  LAYER: {layer_name}")
        print(f"{'═'*60}")
        out_layer_dir = args.out_root / layer_name
        rows = run_one_layer(net, layer_name, stim_dirs, out_layer_dir, args, device)
        all_rows.extend(rows)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_path = args.out_root / "layer_sweep_summary.csv"
    summary_fields = ["layer", "category", "n", "mean_distance", "std_distance"]
    import statistics as _stat
    summary_rows = []
    for layer_name in layers:
        layer_rows = [r for r in all_rows if r["layer"] == layer_name]
        for cat in ["environment", "music", "speech", "ALL"]:
            subset = layer_rows if cat == "ALL" else [r for r in layer_rows if r["category"] == cat]
            if not subset:
                continue
            dists = [float(r["match_distance"]) for r in subset]
            summary_rows.append({
                "layer":         layer_name,
                "category":      cat,
                "n":             len(dists),
                "mean_distance": f"{_stat.mean(dists):.6f}",
                "std_distance":  f"{_stat.pstdev(dists):.6f}",
            })

    with summary_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=summary_fields)
        wr.writeheader()
        wr.writerows(summary_rows)

    print(f"\n{'═'*60}")
    print(f"Layer sweep complete.  Summary → {summary_path}")
    print(f"{'═'*60}")

    # Print quick ASCII table
    print(f"\n{'Layer':<25}  {'Environment':>13}  {'Music':>8}  {'Speech':>8}  {'ALL':>8}")
    print("─" * 70)
    layer_mean = {}
    for row in summary_rows:
        layer_mean.setdefault(row["layer"], {})[row["category"]] = float(row["mean_distance"])
    for layer_name in layers:
        lm = layer_mean.get(layer_name, {})
        print(f"{layer_name:<25}  "
              f"{lm.get('environment', float('nan')):13.4f}  "
              f"{lm.get('music', float('nan')):8.4f}  "
              f"{lm.get('speech', float('nan')):8.4f}  "
              f"{lm.get('ALL', float('nan')):8.4f}")


if __name__ == "__main__":
    main()
