"""
Generate metamers for the 42 MUSHRA natural sounds using the *trained*
AVNet (deep_avsr, audio-visual.pt) as the synthesis model — no random-init
scaffold, no separate av_tmseq2seq.

Pipeline per stimulus .wav:
  1. Compute reference STFT magnitude (exactly what AVNet takes as input).
  2. Capture target activations at AVNet's audioEncoder output (the 6-layer
     transformer stack, before the joint decoder).
  3. PGD in STFT-magnitude space to find a STFT that matches those activations.
  4. Reconstruct audio with Griffin-Lim and save as metamer_audio.wav.
  5. Score both the original and metamer with AVNet CTC greedy decode.
  6. Write a comparison row to comparison.csv.

Usage:
    cd /home/aimeeyu/mms_project
    python3 run_audio_visual_42_metamers.py
"""

import sys, os, csv, math, argparse, random, json
from pathlib import Path

# ── Conda env bootstrap ──────────────────────────────────────────────────────
# All heavy deps live in the `mms` env; re-exec silently if launched elsewhere.
_MMS_PY = "/home/aimeeyu/.conda/envs/mms/bin/python"
if Path(_MMS_PY).exists() and Path(sys.executable).resolve() != Path(_MMS_PY).resolve():
    print(f"[bootstrap] re-execing under {_MMS_PY}")
    os.execv(_MMS_PY, [_MMS_PY, *sys.argv])

import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile
import scipy.signal as scisig
import torchaudio

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
AV_DIR    = REPO_ROOT / "model_metamers_pytorch" / "deep_avsr" / "audio_visual"
sys.path.insert(0, str(AV_DIR))   # config, models.av_net, utils.decoders

# ── Config ────────────────────────────────────────────────────────────────────
AV_WEIGHTS  = "/orcd/data/jhm/001/urops/aimee_yu/deep_avsr_weights/DeepAVSR_Weights/audio-visual.pt"
STIMULI_DIR = REPO_ROOT / "model_metamers_pytorch" / "stimuli" / "MUSHRA_42_NATURAL"
OUT_DIR     = REPO_ROOT / "model_metamers_pytorch" / "results" / "av_metamers_42_natural"

TARGET_LAYER = "audioEncoder"   # AVNet's 6-layer audio transformer stack output
N_ITERS      = 500              # PGD iterations; bump to 3000 for pub-quality
STEP_SIZE    = 1e-2             # L2-normalised gradient step in STFT space
EPS          = 1e3              # L2 epsilon (effectively unbounded for metamers)
INIT         = "noise"          # "noise" = Gaussian noise at ref RMS (correct metamer init)
                                # "ref"   = trivially 0 loss from iter 0, not useful
GL_ITERS     = 64               # Griffin-Lim iterations for wav reconstruction

TARGET_SR = 16_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AV metamers for MUSHRA-42 natural sounds."
    )
    parser.add_argument("--weights", default=AV_WEIGHTS, help="Path to AVNet checkpoint")
    parser.add_argument("--stimuli-dir", default=str(STIMULI_DIR), help="Stimulus root directory")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    parser.add_argument("--target-layer", default=TARGET_LAYER, help="Activation layer to match")
    parser.add_argument("--n-iters", type=int, default=N_ITERS, help="PGD iterations")
    parser.add_argument("--step-size", type=float, default=STEP_SIZE, help="L2-normalized step size")
    parser.add_argument("--eps", type=float, default=EPS, help="L2 epsilon bound")
    parser.add_argument("--init", choices=["noise", "ref"], default=INIT, help="Metamer initialization")
    parser.add_argument("--gl-iters", type=int, default=GL_ITERS, help="Griffin-Lim iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--limit-stim", type=int, default=0, help="Optional number of stimulus folders")
    return parser.parse_args()

# ── Imports that need sys.path set first ──────────────────────────────────────
from config import args as _cfg
from models.av_net import AVNet
from utils.decoders import ctc_greedy_decode


# ═══════════════════════════════════════════════════════════════════════════════
# Scale-invariant L2 match loss (Feather et al. eq. 1)
# ═══════════════════════════════════════════════════════════════════════════════

def metamer_loss(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    num = torch.norm((target - current).flatten(1), dim=1)
    den = torch.norm(target.flatten(1), dim=1).clamp_min(1e-8)
    return (num / den).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# AVNet wrapper — exposes audioEncoder activations
# ═══════════════════════════════════════════════════════════════════════════════

class AVNetWithLatent(torch.nn.Module):
    """Wraps the trained AVNet to optionally return audioEncoder activations.

    Accepts audBatch (T, B, F) — the same shape produced by prepare_stft().
    Only the audio-only path is used (video=None for MUSHRA stimuli).
    """

    def __init__(self, avnet: AVNet):
        super().__init__()
        self.net = avnet

    def forward(self, audBatch: torch.Tensor, with_latent: bool = False):
        # (T, B, F) → (B, F, T) for Conv1d
        x = audBatch.transpose(0, 1).transpose(1, 2)
        x = self.net.audioConv(x)                       # (B, dModel, T/4)
        x = x.transpose(1, 2).transpose(0, 1)           # (T/4, B, dModel)
        x = self.net.positionalEncoding(x)
        encoded = self.net.audioEncoder(x)              # (T/4, B, dModel) ← target

        # Audio-only: skip video branch, go straight to joint decoder
        joint = self.net.jointDecoder(encoded)
        joint = joint.transpose(0, 1).transpose(1, 2)
        joint = self.net.outputConv(joint)
        joint = joint.transpose(1, 2).transpose(0, 1)
        output = F.log_softmax(joint, dim=2)

        if with_latent:
            return output, {TARGET_LAYER: encoded}
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# STFT helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _stft_params():
    """Return (nperseg, hop_length) from the AVNet config."""
    nperseg  = int(TARGET_SR * _cfg["STFT_WIN_LENGTH"])
    noverlap = int(TARGET_SR * _cfg["STFT_OVERLAP"])
    return nperseg, nperseg - noverlap   # (win_length, hop_length)


def prepare_stft(wav_path: Path, device) -> tuple:
    """Read WAV and compute the STFT magnitude exactly as AVNet expects.

    The model requires a minimum sequence length (MAIN_REQ_INPUT_LENGTH), so
    short clips are zero-padded symmetrically.  We record the STFT frame range
    occupied by the *real* (unpadded) signal so the reconstructed metamer can be
    cropped back to the original duration instead of leaving silence/noise tails.

    Returns:
        audBatch : (T, 1, F) float32, on device
        lenBatch : (1,)      int, on device
        meta     : dict with valid_start (frame), n_real (frames),
                   target_samples (exact output length to reconstruct to)
    """
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

    target_samples = len(data)   # exact length the metamer wav should match

    maxval = np.max(np.abs(data))
    if maxval > 0:
        data = data / maxval

    nperseg  = int(sr * win_s)
    noverlap = int(sr * ovl_s)
    _, _, stftVals = scipy_signal.stft(
        data, sr,
        window=_cfg["STFT_WINDOW"],
        nperseg=nperseg, noverlap=noverlap,
        boundary=None, padded=False,
    )
    audInp = np.abs(stftVals).T   # (T_stft, F)
    n_real = audInp.shape[0]      # STFT frames of the real signal

    reqInpLen = _cfg["MAIN_REQ_INPUT_LENGTH"]
    inpLen = int(np.ceil(len(audInp) / 4))
    lp = int(np.floor((4 * inpLen - len(audInp)) / 2))
    rp = int(np.ceil( (4 * inpLen - len(audInp)) / 2))
    audInp = np.pad(audInp, ((lp, rp), (0, 0)), "constant")
    valid_start = lp
    if inpLen < reqInpLen:
        lp2 = int(np.floor((reqInpLen - inpLen) / 2))
        rp2 = int(np.ceil( (reqInpLen - inpLen) / 2))
        audInp = np.pad(audInp, ((4 * lp2, 4 * rp2), (0, 0)), "constant")
        valid_start += 4 * lp2

    inpLen = int(len(audInp) / 4)
    audBatch = torch.from_numpy(audInp).unsqueeze(1).float().to(device)   # (T, 1, F)
    lenBatch = torch.tensor([inpLen]).int().to(device)
    meta = {"valid_start": valid_start, "n_real": n_real, "target_samples": target_samples}
    return audBatch, lenBatch, meta


def stft_to_wav(stft_mag: torch.Tensor, out_path: Path,
                crop: tuple = None, target_samples: int = None):
    """Reconstruct a wav from a STFT magnitude tensor using Griffin-Lim.

    Args:
        stft_mag       : (T, 1, F) float, on any device
        crop           : optional (start_frame, n_frames) to keep only the
                         real-signal region of a zero-padded STFT
        target_samples : optional exact output length (trim/pad to match input)
    """
    win_len, hop_len = _stft_params()
    mag = stft_mag.detach().cpu().float()
    if mag.dim() == 3:
        mag = mag.squeeze(1)       # (T, F)
    if crop is not None:
        start, n = crop
        mag = mag[start:start + n]  # drop zero-padded frames at both ends
    mag = mag.T.unsqueeze(0)       # (1, F, T) — GriffinLim expects (*, F, T)

    gl = torchaudio.transforms.GriffinLim(
        n_fft=win_len,
        hop_length=hop_len,
        win_length=win_len,
        window_fn=torch.hamming_window,   # match analysis window (STFT_WINDOW)
        n_iter=GL_ITERS,
        power=1.0,   # magnitude (not power) spectrogram
    )
    wav = gl(mag).squeeze(0)                            # (T_audio,)
    if target_samples is not None:
        if wav.numel() >= target_samples:
            wav = wav[:target_samples]
        else:
            wav = F.pad(wav, (0, target_samples - wav.numel()))
    wav = wav / wav.abs().max().clamp_min(1e-8)         # normalise to [-1, 1]
    wav_int16 = (wav.clamp(-1, 1) * 32767).short().numpy()
    from scipy.io import wavfile as _wf
    _wf.write(str(out_path), TARGET_SR, wav_int16)


# ═══════════════════════════════════════════════════════════════════════════════
# PGD in STFT-magnitude space
# ═══════════════════════════════════════════════════════════════════════════════

def synthesize_stft_metamer(
    model: AVNetWithLatent,
    ref_stft: torch.Tensor,
) -> tuple:
    """PGD on STFT magnitude space to match audioEncoder activations of ref.

    Returns:
        met_stft : (T, 1, F) optimised STFT magnitude
        history  : list of loss values logged every 100 iters
        dist     : final match distance (scalar)
    """
    with torch.no_grad():
        _, ref_acts = model(ref_stft, with_latent=True)
    A_ref = ref_acts[TARGET_LAYER].detach()   # (T', 1, dModel)

    if INIT == "ref":
        x = ref_stft.clone()
    else:
        # Start from non-negative Gaussian noise scaled to the reference RMS
        rms = ref_stft.pow(2).mean().sqrt().clamp_min(1e-4)
        x = torch.randn_like(ref_stft).abs() * rms

    history = []
    for it in range(N_ITERS):
        x = x.detach().requires_grad_(True)
        _, acts = model(x, with_latent=True)
        loss = metamer_loss(A_ref, acts[TARGET_LAYER])
        loss.backward()

        with torch.no_grad():
            g = x.grad
            g_norm = g / (g.flatten().norm() + 1e-12)
            x_new = x - STEP_SIZE * g_norm
            # L2 projection onto eps-ball around reference (inactive at EPS=1e3)
            delta = x_new - ref_stft
            d_norm = delta.flatten().norm()
            if d_norm > EPS:
                delta = delta * (EPS / d_norm)
            x = (ref_stft + delta).clamp_min(0.0)   # STFT magnitude ≥ 0

        if it % 100 == 0:
            history.append(loss.item())
            print(f"    iter {it:4d}  loss {loss.item():.4f}")

    with torch.no_grad():
        _, final_acts = model(x, with_latent=True)
        dist = metamer_loss(A_ref, final_acts[TARGET_LAYER]).item()

    return x, history, dist


# ═══════════════════════════════════════════════════════════════════════════════
# CTC prediction from a wav file path
# ═══════════════════════════════════════════════════════════════════════════════

def predict_wav(model: AVNetWithLatent, wav_path: Path, device) -> str:
    audBatch, lenBatch, _ = prepare_stft(wav_path, device)
    with torch.no_grad():
        outputBatch = model(audBatch)
    predBatch, _ = ctc_greedy_decode(
        outputBatch, lenBatch, _cfg["CHAR_TO_INDEX"]["<EOS>"]
    )
    pred = predBatch[:][:-1]
    return "".join([_cfg["INDEX_TO_CHAR"][ix] for ix in pred.tolist()])


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    global AV_WEIGHTS, STIMULI_DIR, OUT_DIR
    global TARGET_LAYER, N_ITERS, STEP_SIZE, EPS, INIT, GL_ITERS

    AV_WEIGHTS = args.weights
    STIMULI_DIR = Path(args.stimuli_dir)
    OUT_DIR = Path(args.out_dir)
    TARGET_LAYER = args.target_layer
    N_ITERS = args.n_iters
    STEP_SIZE = args.step_size
    EPS = args.eps
    INIT = args.init
    GL_ITERS = args.gl_iters

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Target layer : {TARGET_LAYER}  (trained AVNet audioEncoder)")
    print(f"Iters        : {N_ITERS}")
    print(f"Init         : {INIT}")
    print(f"Step size    : {STEP_SIZE}")
    print(f"GL iters     : {GL_ITERS}\n")
    print(f"Seed         : {args.seed}")
    print(f"Stimuli dir  : {STIMULI_DIR}")
    print(f"Output dir   : {OUT_DIR}\n")

    avnet = AVNet(
        _cfg["TX_NUM_FEATURES"],
        _cfg["TX_ATTENTION_HEADS"],
        _cfg["TX_NUM_LAYERS"],
        _cfg["PE_MAX_LENGTH"],
        _cfg["AUDIO_FEATURE_SIZE"],
        _cfg["TX_FEEDFORWARD_DIM"],
        _cfg["TX_DROPOUT"],
        _cfg["NUM_CLASSES"],
    )
    avnet.load_state_dict(torch.load(AV_WEIGHTS, map_location=device))
    avnet.eval()
    avnet.to(device)
    print(f"Loaded AVNet from {AV_WEIGHTS}\n")

    model = AVNetWithLatent(avnet)

    stim_dirs = sorted(
        [d for d in STIMULI_DIR.iterdir() if d.is_dir()],
        key=lambda d: int(d.name.split("_")[0]),
    )
    if args.limit_stim > 0:
        stim_dirs = stim_dirs[:args.limit_stim]
    print(f"Found {len(stim_dirs)} stimulus folders\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(
        json.dumps(
            {
                "weights": AV_WEIGHTS,
                "stimuli_dir": str(STIMULI_DIR),
                "out_dir": str(OUT_DIR),
                "target_layer": TARGET_LAYER,
                "n_iters": N_ITERS,
                "step_size": STEP_SIZE,
                "eps": EPS,
                "init": INIT,
                "gl_iters": GL_ITERS,
                "seed": args.seed,
                "limit_stim": args.limit_stim,
            },
            indent=2,
        )
    )
    csv_path = OUT_DIR / "comparison.csv"
    fieldnames = [
        "stim_folder", "wav_file",
        "pred_original", "pred_metamer",
        "match_distance", "metamer_wav",
    ]

    rows = []
    total_wavs = sum(len(list(d.glob("*.wav"))) for d in stim_dirs)
    done = 0

    for stim_dir in stim_dirs:
        for wav_path in sorted(stim_dir.glob("*.wav")):
            done += 1
            print(f"[{done}/{total_wavs}] {stim_dir.name}/{wav_path.name}")

            ref_stft, _, meta = prepare_stft(wav_path, device)
            # Make each stimulus deterministic but unique per run seed.
            local_seed = args.seed + done
            random.seed(local_seed)
            np.random.seed(local_seed)
            torch.manual_seed(local_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(local_seed)

            print(f"  synthesising ({N_ITERS} iters, init={INIT}) …")
            met_stft, history, dist = synthesize_stft_metamer(model, ref_stft)

            stim_out = OUT_DIR / stim_dir.name / wav_path.stem
            stim_out.mkdir(parents=True, exist_ok=True)

            metamer_wav = stim_out / "metamer_audio.wav"
            stft_to_wav(
                met_stft, metamer_wav,
                crop=(meta["valid_start"], meta["n_real"]),
                target_samples=meta["target_samples"],
            )
            (stim_out / "match_distance.txt").write_text(
                f"{TARGET_LAYER}\t{dist:.6f}\n"
            )
            torch.save(met_stft.cpu(), stim_out / "metamer_stft.pt")
            print(f"  match distance: {dist:.4f}  →  {metamer_wav}")

            try:
                pred_orig = predict_wav(model, wav_path, device)
            except Exception as e:
                pred_orig = f"ERROR:{e}"
            try:
                pred_met = predict_wav(model, metamer_wav, device)
            except Exception as e:
                pred_met = f"ERROR:{e}"

            print(f"  original → {pred_orig!r}")
            print(f"  metamer  → {pred_met!r}\n")

            rows.append({
                "stim_folder":    stim_dir.name,
                "wav_file":       wav_path.name,
                "pred_original":  pred_orig,
                "pred_metamer":   pred_met,
                "match_distance": f"{dist:.6f}",
                "metamer_wav":    str(metamer_wav),
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} pairs written to:")
    print(f"  CSV    : {csv_path}")
    print(f"  Wavs   : {OUT_DIR}/<stim>/<stem>/metamer_audio.wav")
    print(f"  STFTs  : {OUT_DIR}/<stim>/<stem>/metamer_stft.pt")


if __name__ == "__main__":
    main()
