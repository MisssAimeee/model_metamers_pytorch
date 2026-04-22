"""Driver for AV metamer synthesis.

This is the AV analog of Feather et al.'s `generate_auditory_metamers.py`
script in `model_metamers_pytorch/analysis_scripts/`.

Usage (once integrated into the repo):

    python -m analysis_scripts.generate_av_metamers \
        --architecture av_tmseq2seq \
        --checkpoint /path/to/ckpt.pt \
        --layer fusion.cross_attn.2 \
        --mode joint \
        --ref-audio samples/nat.wav \
        --ref-video samples/nat.mp4 \
        --out-dir out/metamers/fusion_cross_attn_2/

Three modes:
    single-audio : freeze video = ref, optimize audio to match target layer.
    single-video : freeze audio = ref, optimize video.
    joint        : optimize both audio and video.

Supports an optional semantic-consistency regularizer (ImageBind or CAV-MAE
embedding distance) to test question 3.b from the research report.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import torch
import torch.nn.functional as F

# In the real repo these are:
#   from robustness.build_network import build_network_av
#   from robustness.datasets import AVSpeech
#   from robustness.attacker import AttackerModel
from model_metamers_pytorch.robustness.AY_build_network_av import build_network_av       # scaffold path

# Clip length fed to the model (2 s @ 16 kHz audio, 50 frames @ 25 fps video).
_DURATION_SECS: float = 2.0
_AUDIO_SR: int = 16_000
_VIDEO_FPS: int = 25
_VIDEO_SIZE: int = 112


# ---------------------------------------------------------------------------
# Metamer loss (Feather eq. 1)
# ---------------------------------------------------------------------------

def metamer_loss(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """||A - A'||_2 / ||A||_2, a scale-invariant match loss."""
    num = torch.norm((target - current).flatten(1), dim=1)
    den = torch.norm(target.flatten(1), dim=1).clamp_min(1e-8)
    return (num / den).mean()


# ---------------------------------------------------------------------------
# Semantic-consistency regularizer (optional)
# ---------------------------------------------------------------------------

def semantic_consistency(av_features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Cosine distance between pooled audio and video features.

    In a real run we would swap this for ImageBind or CAV-MAE embeddings so
    that the distance reflects SEMANTIC consistency, not mere feature overlap.
    """
    a = av_features["audio_pool"]; v = av_features["video_pool"]
    return 1.0 - F.cosine_similarity(a, v, dim=-1).mean()


# ---------------------------------------------------------------------------
# Dict-valued PGD step (hook for a generalized attacker.py)
# ---------------------------------------------------------------------------

def pgd_step(
    model,
    x: Dict[str, torch.Tensor],
    target_layer: str,
    target_activation: torch.Tensor,
    mode: str,                       # 'single-audio' | 'single-video' | 'joint'
    step_sizes: Dict[str, float],
    eps: Dict[str, float],
    consistency_weight: float = 0.0,
):
    x = {k: v.clone().detach().requires_grad_(k in _active_keys(mode)) for k, v in x.items()}

    _, outs = model(x, with_latent=True, fake_relu=True)
    loss = metamer_loss(target_activation, outs[target_layer])

    if consistency_weight > 0:
        loss = loss + consistency_weight * semantic_consistency({
            "audio_pool": outs["audio.transformer.5"].mean(-1),
            "video_pool": outs["video.transformer.5"].mean(-1),
        })

    grads = torch.autograd.grad(loss, [x[k] for k in _active_keys(mode)])

    out = dict(x)
    for k, g in zip(_active_keys(mode), grads):
        # L2 step, then L2 projection onto the epsilon-ball around zero perturbation
        g_norm = g / (g.flatten(1).norm(dim=1).view(-1, *([1] * (g.dim() - 1))) + 1e-12)
        out[k] = x[k] - step_sizes[k] * g_norm
        # Project
        delta = out[k] - x[k]  # NOTE: around the current iterate, not ref; see loop
        delta_norm = delta.flatten(1).norm(dim=1).view(-1, *([1] * (delta.dim() - 1)))
        delta = delta * (eps[k] / delta_norm.clamp_min(eps[k]))
        out[k] = (x[k] + delta).detach()
    return out, loss.item()


def _active_keys(mode: str):
    return {
        "single-audio": ["audio"],
        "single-video": ["video"],
        "joint": ["audio", "video"],
    }[mode]


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------

def synthesize_metamer(model, ref: Dict[str, torch.Tensor], target_layer: str,
                       mode: str = "joint", n_iters: int = 3000,
                       eps=None, step_sizes=None, consistency_weight=0.0,
                       init: str = "noise"):
    # Capture target activation from the natural reference
    with torch.no_grad():
        _, ref_outs = model(ref, with_latent=True)
    A_ref = ref_outs[target_layer].detach()

    # Initialize
    x = {}
    for k, v in ref.items():
        if k in _active_keys(mode):
            if init == "noise":
                # Bandlimited / perceptually-flat noise would be better; Gaussian for scaffold.
                x[k] = torch.randn_like(v) * (0.3 if k == "audio" else 0.1)
            elif init == "ref":
                x[k] = v.clone()
            else:
                raise ValueError(init)
        else:
            x[k] = v.clone()

    eps = eps or {"audio": 1e3, "video": 1e3}      # effectively unbounded for metamers
    step_sizes = step_sizes or {"audio": 1e-2, "video": 1e-2}

    history = []
    for it in range(n_iters):
        x, loss = pgd_step(model, x, target_layer, A_ref, mode,
                            step_sizes, eps, consistency_weight)
        if it % 100 == 0:
            history.append(loss)
            print(f"  iter {it:4d}  loss {loss:.4f}")

    return x, history


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_audio(audio_path: Path) -> torch.Tensor:
    """Read a WAV file, mono-mix, resample to _AUDIO_SR, clip/pad to _DURATION_SECS.

    Returns (1, 1, T) float32 tensor with values in roughly [-1, 1].
    """
    sr, data = wavfile.read(str(audio_path))
    data = data.astype(np.float32)
    # Normalise int PCM to [-1, 1]
    if data.dtype != np.float32 or data.max() > 1.0:
        if data.max() > 1.0:
            data /= np.iinfo(np.int16).max if data.max() < 2**15 else np.iinfo(np.int32).max
    # Mono mix
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Resample if needed
    if sr != _AUDIO_SR:
        g = math.gcd(sr, _AUDIO_SR)
        data = scipy.signal.resample_poly(data, _AUDIO_SR // g, sr // g).astype(np.float32)
    # Clip / pad
    n = int(_AUDIO_SR * _DURATION_SECS)
    if len(data) >= n:
        data = data[:n]
    else:
        data = np.pad(data, (0, n - len(data)))
    return torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1, 1, T)


def _load_video(video_path: Path) -> torch.Tensor:
    """Read an MP4, sample _VIDEO_FPS frames/s, resize to _VIDEO_SIZE, return (1,3,T,H,W).

    Values in [0, 1] float32.  Pads with the last frame if the clip is short.
    """
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or _VIDEO_FPS
    n_frames = int(_VIDEO_FPS * _DURATION_SECS)  # 50

    # Compute which source frame indices to sample so we get _VIDEO_FPS output fps
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_duration = total_src / src_fps if total_src > 0 else _DURATION_SECS
    target_times = np.linspace(0, _DURATION_SECS, n_frames, endpoint=False)
    src_indices = np.clip((target_times * src_fps).astype(int), 0, max(total_src - 1, 0))

    frame_cache: dict[int, np.ndarray] = {}
    frames = []
    for idx in src_indices:
        if idx not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   (_VIDEO_SIZE, _VIDEO_SIZE))
                frame_cache[idx] = frame
        if idx in frame_cache:
            frames.append(frame_cache[idx])
        elif frames:
            frames.append(frames[-1])       # pad with last good frame
        else:
            frames.append(np.zeros((_VIDEO_SIZE, _VIDEO_SIZE, 3), dtype=np.uint8))
    cap.release()

    arr = np.stack(frames).astype(np.float32) / 255.0   # (T, H, W, C)
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)


def load_reference(audio_path: Path, video_path: Path) -> Dict[str, torch.Tensor]:
    """Load a real AV sample from disk.  Falls back to random tensors for /dev/null paths."""
    use_stub = str(audio_path) in ("/dev/null", "") or not Path(audio_path).exists()
    if use_stub:
        print("[load_reference] using random stub tensors (pass real --ref-audio/--ref-video for real data)")
        n_audio = int(_AUDIO_SR * _DURATION_SECS)
        n_video = int(_VIDEO_FPS * _DURATION_SECS)
        return {
            "audio": torch.randn(1, 1, n_audio),
            "video": torch.randn(1, 3, n_video, _VIDEO_SIZE, _VIDEO_SIZE),
        }
    audio = _load_audio(audio_path)
    video = _load_video(video_path)
    print(f"[load_reference] audio {tuple(audio.shape)}  video {tuple(video.shape)}")
    return {"audio": audio, "video": video}


def save_metamer(x: Dict[str, torch.Tensor], out_dir: Path):
    """Save metamer tensors as .pt, .wav, and per-frame .png files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw tensors
    torch.save(x, out_dir / "metamer.pt")

    # Audio → WAV
    wav = x["audio"].squeeze().cpu().numpy()   # (T,)
    wav = np.clip(wav, -1.0, 1.0)
    wav_int16 = (wav * 32767).astype(np.int16)
    wavfile.write(str(out_dir / "metamer_audio.wav"), _AUDIO_SR, wav_int16)

    # Video → PNG frames
    frames_dir = out_dir / "metamer_frames"
    frames_dir.mkdir(exist_ok=True)
    vid = x["video"].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
    vid = (np.clip(vid, 0.0, 1.0) * 255).astype(np.uint8)
    for i, frame in enumerate(vid):
        cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"saved -> {out_dir}/metamer.pt  |  metamer_audio.wav  |  metamer_frames/ ({len(vid)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", default="av_tmseq2seq")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--layer", required=True,
                    help="One of ALLOWED_METAMER_LAYERS, e.g. 'fusion.cross_attn.2'")
    p.add_argument("--mode", default="joint",
                    choices=["single-audio", "single-video", "joint"])
    p.add_argument("--ref-audio", type=Path, required=True)
    p.add_argument("--ref-video", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-iters", type=int, default=3000)
    p.add_argument("--consistency-weight", type=float, default=0.0)
    p.add_argument("--init", default="noise", choices=["noise", "ref"])
    return p.parse_args()


def main():
    args = parse_args()
    model, entry = build_network_av(args.architecture, checkpoint_path=args.checkpoint)
    model.eval()
    if args.layer not in entry["allowed_metamer_layers"]:
        raise ValueError(f"layer '{args.layer}' not in ALLOWED_METAMER_LAYERS")

    ref = load_reference(args.ref_audio, args.ref_video)
    metamer, history = synthesize_metamer(
        model, ref, args.layer,
        mode=args.mode, n_iters=args.n_iters,
        consistency_weight=args.consistency_weight,
        init=args.init,
    )
    save_metamer(metamer, args.out_dir)

    # Write final activation-distance for the null-distribution analysis
    with torch.no_grad():
        _, outs_m = model(metamer, with_latent=True)
        _, outs_r = model(ref, with_latent=True)
        d = metamer_loss(outs_r[args.layer], outs_m[args.layer]).item()
    (args.out_dir / "match_distance.txt").write_text(f"{args.layer}\t{d:.6f}\n")
    print(f"final match distance at {args.layer}: {d:.6f}")


if __name__ == "__main__":
    main()
