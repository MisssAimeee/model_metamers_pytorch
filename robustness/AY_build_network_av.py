"""Additions to `robustness/build_network.py` for an AV architecture.

Copy the `AV_ARCH_ENTRIES` and `normalize_av_input` helpers into Feather's
existing `build_network.py`, then add `AV_ARCH_ENTRIES` to the master
`ARCH_TABLE` dict that `analysis_scripts/*.py` consults.

The shape of this file mirrors Feather's existing per-architecture entries
(e.g. for `cochcnn`, `cochresnet50`) so reviewers can see what changed at a glance.
"""
from __future__ import annotations

from typing import Dict

import torch

from .audio_models import (
    AVMetamerModel,
    AVModelSpec,
    build_av_tmseq2seq,
)


# ---------------------------------------------------------------------------
# Per-branch normalization (mean/std), applied inside `AttackerModel`
# ---------------------------------------------------------------------------

AUDIO_MEAN = 0.0
AUDIO_STD = 0.1              # rough estimate for normalized waveform
VIDEO_MEAN = (0.485, 0.456, 0.406)  # ImageNet
VIDEO_STD = (0.229, 0.224, 0.225)


def normalize_av_input(x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize a dict-valued AV input in-place (functional)."""
    a = (x["audio"] - AUDIO_MEAN) / AUDIO_STD
    v = x["video"].clone()
    mean = torch.tensor(VIDEO_MEAN, device=v.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(VIDEO_STD, device=v.device).view(1, 3, 1, 1, 1)
    v = (v - mean) / std
    return {"audio": a, "video": v}


# ---------------------------------------------------------------------------
# ARCH_TABLE entry
# ---------------------------------------------------------------------------

def _instantiate_av_tmseq2seq(checkpoint_path=None, **kwargs):
    spec = AVModelSpec(
        d_model=kwargs.get("d_model", 512),
        n_audio_layers=kwargs.get("n_audio_layers", 6),
        n_video_layers=kwargs.get("n_video_layers", 6),
        n_fusion_layers=kwargs.get("n_fusion_layers", 6),
        n_classes=kwargs.get("n_classes", 1000),
    )
    return build_av_tmseq2seq(checkpoint_path, spec=spec)


AV_ARCH_ENTRIES = {
    "av_tmseq2seq": {
        "constructor": _instantiate_av_tmseq2seq,
        # Feather's framework keys by "allowed_metamer_layers"
        "allowed_metamer_layers": AVMetamerModel(AVModelSpec()).spec.allowed_metamer_layers,
        "input_spec": AVModelSpec().input_spec,
        # dict-valued inputs: signal to attacker.py that PGD must loop over keys
        "multimodal_input": True,
        "normalization": normalize_av_input,
        # For dataset handler:
        "dataset_handler": "avspeech",
        # Default adversarial bounds, per branch (L2 unless noted):
        "adversarial": {
            "audio": {"norm": "l2", "eps": 0.5, "step_size": 0.05, "steps": 40,
                       "target": "waveform"},   # or "latent"
            "video": {"norm": "l2", "eps": 2.0, "step_size": 0.2, "steps": 40,
                       "target": "pixel"},
        },
    },
    # Slot to add AV-HuBERT, CAV-MAE, etc. later, by reusing the contract above.
}


# ---------------------------------------------------------------------------
# Loader used by analysis_scripts
# ---------------------------------------------------------------------------

def build_network_av(architecture_name: str, checkpoint_path: str = None, **kwargs):
    if architecture_name not in AV_ARCH_ENTRIES:
        raise KeyError(f"Unknown AV architecture: {architecture_name}. "
                       f"Known: {list(AV_ARCH_ENTRIES)}")
    entry = AV_ARCH_ENTRIES[architecture_name]
    model = entry["constructor"](checkpoint_path=checkpoint_path, **kwargs)
    return model, entry
