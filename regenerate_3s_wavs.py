"""
Regenerate metamer WAVs at the original input duration (fix length bug).

Background
----------
`prepare_stft` zero-pads every clip up to MAIN_REQ_INPUT_LENGTH (≈5.8 s) so the
trained AVNet sees a sequence at least as long as its training minimum.  The
old `stft_to_wav` reconstructed the *entire* padded magnitude, producing ~5.8 s
metamers with ~1.4 s of silence/noise at both ends.

The synthesis (PGD) result is stored in `metamer_stft.pt` (the full padded
magnitude).  Match distances were computed on activations of that padded tensor,
so they are **unchanged** by this fix.  This utility simply re-runs the final
STFT→audio step on the *valid* (unpadded) frame range, yielding metamers whose
length matches the original input — without repeating the expensive optimization.

It also switches Griffin-Lim to a Hamming window (matching the analysis window
`STFT_WINDOW="hamming"`) for cleaner reconstruction; the old code used a Hann
window.

Usage:
    cd /home/aimeeyu/mms_project/model_metamers_pytorch
    python regenerate_3s_wavs.py                 # base + all layer-sweep sets
    python regenerate_3s_wavs.py --dry-run       # report only, write nothing
"""

import sys, os, math, argparse, json
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

REPO_ROOT = Path(__file__).resolve().parent
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

TARGET_SR  = 16_000
STIMULI_DIR = _STIMULI_BASE / "MUSHRA_42_NATURAL"

LAYER_NAMES = [
    "audioConv_out",
    "audioEncoder_L0", "audioEncoder_L1", "audioEncoder_L2",
    "audioEncoder_L3", "audioEncoder_L4", "audioEncoder_L5",
    "jointDecoder_out", "logits",
]


def compute_meta(wav_path: Path) -> dict:
    """Replicate prepare_stft padding math to find the real-signal frame range."""
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
    target_samples = len(data)

    maxval = np.max(np.abs(data))
    if maxval > 0:
        data = data / maxval

    nperseg  = int(sr * win_s)
    noverlap = int(sr * ovl_s)
    _, _, stftVals = scisig.stft(
        data, sr, window=_cfg["STFT_WINDOW"],
        nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False,
    )
    audInp = np.abs(stftVals).T
    n_real = audInp.shape[0]
    reqInpLen = _cfg["MAIN_REQ_INPUT_LENGTH"]
    inpLen = int(np.ceil(n_real / 4))
    lp = int(np.floor((4 * inpLen - n_real) / 2))
    valid_start = lp
    if inpLen < reqInpLen:
        lp2 = int(np.floor((reqInpLen - inpLen) / 2))
        valid_start += 4 * lp2
    return {"valid_start": valid_start, "n_real": n_real, "target_samples": target_samples}


def stft_pt_to_wav(pt_path: Path, out_path: Path, meta: dict, gl_iters: int):
    nperseg  = int(TARGET_SR * _cfg["STFT_WIN_LENGTH"])
    noverlap = int(TARGET_SR * _cfg["STFT_OVERLAP"])
    hop_len  = nperseg - noverlap

    mag = torch.load(pt_path, map_location="cpu").float()
    if mag.dim() == 3:
        mag = mag.squeeze(1)                       # (T, F)
    start, n = meta["valid_start"], meta["n_real"]
    mag = mag[start:start + n]                     # crop padded frames
    mag = mag.T.unsqueeze(0)                       # (1, F, T)

    gl = torchaudio.transforms.GriffinLim(
        n_fft=nperseg, hop_length=hop_len, win_length=nperseg,
        window_fn=torch.hamming_window, n_iter=gl_iters, power=1.0,
    )
    wav = gl(mag).squeeze(0)
    tgt = meta["target_samples"]
    if wav.numel() >= tgt:
        wav = wav[:tgt]
    else:
        wav = F.pad(wav, (0, tgt - wav.numel()))
    wav = wav / wav.abs().max().clamp_min(1e-8)
    wav_int16 = (wav.clamp(-1, 1) * 32767).short().numpy()
    wavfile.write(str(out_path), TARGET_SR, wav_int16)
    return wav.numel()


def gl_iters_for(result_dir: Path, default: int = 64) -> int:
    cfg = result_dir / "run_config.json"
    if cfg.exists():
        try:
            return int(json.loads(cfg.read_text()).get("gl_iters", default))
        except Exception:
            pass
    return default


def find_original(stim_folder: str, clip_stem: str) -> Path | None:
    cand = STIMULI_DIR / stim_folder / f"{clip_stem}.wav"
    if cand.exists():
        return cand
    matches = list(STIMULI_DIR.glob(f"**/{clip_stem}.wav"))
    return matches[0] if matches else None


def regenerate_set(result_dir: Path, dry_run: bool) -> tuple:
    if not result_dir.exists():
        print(f"  [skip] missing: {result_dir.name}")
        return 0, 0
    gl_iters = gl_iters_for(result_dir)
    done, missing = 0, 0
    meta_cache = {}
    for stim_dir in sorted(p for p in result_dir.iterdir() if p.is_dir()):
        for clip_dir in sorted(p for p in stim_dir.iterdir() if p.is_dir()):
            pt = clip_dir / "metamer_stft.pt"
            if not pt.exists():
                continue
            orig = find_original(stim_dir.name, clip_dir.name)
            if orig is None:
                print(f"    [warn] no original for {clip_dir.name}")
                missing += 1
                continue
            if orig not in meta_cache:
                meta_cache[orig] = compute_meta(orig)
            meta = meta_cache[orig]
            out_wav = clip_dir / "metamer_audio.wav"
            if dry_run:
                done += 1
                continue
            n = stft_pt_to_wav(pt, out_wav, meta, gl_iters)
            done += 1
            if done % 40 == 0:
                print(f"    {result_dir.name}: {done} clips ({n} samples, "
                      f"{n/TARGET_SR:.2f}s, gl_iters={gl_iters})")
    print(f"  [{result_dir.name}] regenerated {done} wavs"
          + (f", {missing} missing originals" if missing else ""))
    return done, missing


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate metamer wavs at input length")
    p.add_argument("--result-dirs", nargs="*", default=None,
                   help="Specific result dirs (default: base + all layer-sweep sets)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.result_dirs:
        dirs = [Path(d) for d in args.result_dirs]
    else:
        dirs = [_RESULTS_BASE / "av_metamers_42_natural"]
        dirs += [_RESULTS_BASE / "layer_sweep" / l for l in LAYER_NAMES]

    print(f"Stimuli dir : {STIMULI_DIR}")
    print(f"Result sets : {len(dirs)}")
    if args.dry_run:
        print("(dry-run — no files written)\n")

    total_done, total_missing = 0, 0
    for d in dirs:
        done, missing = regenerate_set(d, args.dry_run)
        total_done += done
        total_missing += missing

    print(f"\nDone. {total_done} wavs regenerated"
          + (f", {total_missing} missing originals" if total_missing else ""))


if __name__ == "__main__":
    main()
