"""AV model for metamer synthesis, adapted from Afouras & Shah (deep_avsr).

Contract expected by `robustness/AY_build_network_av.py` and
`analysis_scripts/AY_generate_av_metamers.py`:

    model = build_av_tmseq2seq(checkpoint_path=None, spec=AVModelSpec())
    logits, activations = model({"audio": (B,1,T), "video": (B,3,T,H,W)},
                                 with_latent=True, fake_relu=True)

`activations` is a dict keyed by strings in `spec.allowed_metamer_layers`.
Required keys include `audio.transformer.5`, `video.transformer.5`, and
`fusion.cross_attn.{0..5}` (targeted directly by the scaffold).

Architecture sketch (deep_avsr-inspired, not weight-compatible):
    audio: (B,1,32000) -> STFT mag (B,Ta,321) -> Linear -> PosEnc
        -> stack of TransformerEncoderLayer x n_audio_layers
    video: (B,3,50,112,112) -> RGB->gray -> 3D conv stem -> per-frame 2D blocks
        -> Linear -> PosEnc -> TransformerEncoderLayer x n_video_layers
    fusion: bidirectional cross-attention blocks x n_fusion_layers
    head:   concat(mean-pool audio, mean-pool video) -> Linear -> logits
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass
class AVModelSpec:
    d_model: int = 512
    n_audio_layers: int = 6
    n_video_layers: int = 6
    n_fusion_layers: int = 6
    n_classes: int = 1000

    n_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1

    audio_sr: int = 16_000
    audio_duration_s: float = 2.0
    video_fps: int = 25
    video_size: int = 112

    stft_n_fft: int = 640
    stft_hop: int = 160
    stft_win: int = 640

    @property
    def audio_feature_size(self) -> int:
        return self.stft_n_fft // 2 + 1

    @property
    def audio_samples(self) -> int:
        return int(self.audio_sr * self.audio_duration_s)

    @property
    def video_frames(self) -> int:
        return int(self.video_fps * self.audio_duration_s)

    @property
    def allowed_metamer_layers(self) -> List[str]:
        layers = ["audio.embed", "video.embed"]
        layers += [f"audio.transformer.{i}" for i in range(self.n_audio_layers)]
        layers += [f"video.transformer.{i}" for i in range(self.n_video_layers)]
        layers += [f"fusion.cross_attn.{i}" for i in range(self.n_fusion_layers)]
        layers.append("logits")
        return layers

    @property
    def input_spec(self) -> Dict[str, tuple]:
        return {
            "audio": (1, 1, self.audio_samples),
            "video": (1, 3, self.video_frames, self.video_size, self.video_size),
        }


# ---------------------------------------------------------------------------
# Audio preprocessing: differentiable STFT magnitude
# ---------------------------------------------------------------------------

class AudioSTFT(nn.Module):
    def __init__(self, spec: AVModelSpec):
        super().__init__()
        self.n_fft = spec.stft_n_fft
        self.hop = spec.stft_hop
        self.win = spec.stft_win
        self.register_buffer("window", torch.hamming_window(self.win))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            return_complex=True,
            center=False,
        )
        return spec.abs().transpose(1, 2)  # (B, n_frames, n_freq)


# ---------------------------------------------------------------------------
# Visual frontend: (B,3,T,H,W) RGB -> (B,T,d_model)
# ---------------------------------------------------------------------------

def _conv_bn_relu(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class VisualFrontend(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.register_buffer(
            "rgb_to_gray",
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1, 1),
        )
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                      padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.blocks = nn.Sequential(
            _conv_bn_relu(64, 128, stride=2),
            _conv_bn_relu(128, 256, stride=2),
            _conv_bn_relu(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, d_model)

    def forward(self, video_rgb: torch.Tensor) -> torch.Tensor:
        gray = (video_rgb * self.rgb_to_gray).sum(dim=1, keepdim=True)
        x = self.stem(gray)                                 # (B,64,T,h,w)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)                         # (B*T, 512)
        x = self.proj(x).view(B, T, -1)                     # (B, T, d_model)
        return x


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal, broadcast over batch)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                              * -(math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Fusion: bidirectional cross-attention block
# ---------------------------------------------------------------------------

class CrossAttentionFusionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.a2v = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.v2a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_v1 = nn.LayerNorm(d_model)
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm_a2 = nn.LayerNorm(d_model)
        self.norm_v2 = nn.LayerNorm(d_model)

    def forward(self, a: torch.Tensor, v: torch.Tensor):
        a2, _ = self.a2v(query=a, key=v, value=v)
        a = self.norm_a1(a + a2)
        v2, _ = self.v2a(query=v, key=a, value=a)
        v = self.norm_v1(v + v2)
        a = self.norm_a2(a + self.ffn_a(a))
        v = self.norm_v2(v + self.ffn_v(v))
        return a, v


# ---------------------------------------------------------------------------
# AV metamer model
# ---------------------------------------------------------------------------

class AVMetamerModel(nn.Module):
    def __init__(self, spec: Optional[AVModelSpec] = None):
        super().__init__()
        self.spec = spec or AVModelSpec()
        s = self.spec

        self.audio_stft = AudioSTFT(s)
        self.audio_embed = nn.Linear(s.audio_feature_size, s.d_model)
        self.audio_pe = PositionalEncoding(s.d_model, max_len=4096)
        audio_layer = nn.TransformerEncoderLayer(
            d_model=s.d_model, nhead=s.n_heads, dim_feedforward=s.ff_dim,
            dropout=s.dropout, batch_first=True, activation="relu",
        )
        self.audio_transformer = nn.ModuleList(
            [copy.deepcopy(audio_layer) for _ in range(s.n_audio_layers)]
        )

        self.visual_frontend = VisualFrontend(d_model=s.d_model)
        self.video_pe = PositionalEncoding(s.d_model, max_len=1024)
        video_layer = nn.TransformerEncoderLayer(
            d_model=s.d_model, nhead=s.n_heads, dim_feedforward=s.ff_dim,
            dropout=s.dropout, batch_first=True, activation="relu",
        )
        self.video_transformer = nn.ModuleList(
            [copy.deepcopy(video_layer) for _ in range(s.n_video_layers)]
        )

        self.fusion_cross_attn = nn.ModuleList([
            CrossAttentionFusionLayer(s.d_model, s.n_heads, s.ff_dim, s.dropout)
            for _ in range(s.n_fusion_layers)
        ])

        self.classifier = nn.Linear(2 * s.d_model, s.n_classes)

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        with_latent: bool = False,
        fake_relu: bool = False,  # accepted for Feather-API compatibility; no-op here
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        activations: Dict[str, torch.Tensor] = {}

        audio_feats = self.audio_stft(x["audio"])           # (B, Ta, 321)
        a = self.audio_embed(audio_feats)                   # (B, Ta, d)
        a = self.audio_pe(a)
        if with_latent:
            activations["audio.embed"] = a
        for i, layer in enumerate(self.audio_transformer):
            a = layer(a)
            if with_latent:
                activations[f"audio.transformer.{i}"] = a

        v = self.visual_frontend(x["video"])                # (B, Tv, d)
        v = self.video_pe(v)
        if with_latent:
            activations["video.embed"] = v
        for i, layer in enumerate(self.video_transformer):
            v = layer(v)
            if with_latent:
                activations[f"video.transformer.{i}"] = v

        for i, layer in enumerate(self.fusion_cross_attn):
            a, v = layer(a, v)
            if with_latent:
                activations[f"fusion.cross_attn.{i}"] = torch.cat([a, v], dim=1)

        logits = self.classifier(torch.cat([a.mean(dim=1), v.mean(dim=1)], dim=-1))
        if with_latent:
            activations["logits"] = logits
        return logits, activations


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_av_tmseq2seq(
    checkpoint_path: Optional[str] = None,
    spec: Optional[AVModelSpec] = None,
) -> AVMetamerModel:
    model = AVMetamerModel(spec)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    return model
