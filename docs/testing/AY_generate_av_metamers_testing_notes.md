# AV Metamer Pipeline — Architecture Notes

Companion doc for `AY_generate_av_metamers.py`. Covers (1) what the current
pipeline does end-to-end, and (2) how a CNN vs Transformer backbone changes
what the resulting metamers look like and mean.

---

## 1. What `AY_generate_av_metamers.py` currently does

The script is explicitly labeled a **scaffold**. It implements the
Feather-style metamer synthesis loop over a dict-valued AV input
`{audio, video}`, wrapped around a transformer AV model.

### 1.1 Model

Built via `build_network_av("av_tmseq2seq", checkpoint_path=...)` →
`AVMetamerModel` in `robustness/audio_models/av_model.py`. It is a
**transformer stack**:

- **Audio encoder**: STFT-magnitude → Linear → positional encoding →
  6× `TransformerEncoderLayer`.
- **Video encoder**: 3D conv stem → per-frame 2D blocks → Linear →
  positional encoding → 6× `TransformerEncoderLayer`.
- **Fusion**: 6 bidirectional cross-attention blocks
  (`fusion.cross_attn.{0..5}`).
- **Head**: concat(mean-pool audio, mean-pool video) → Linear → logits.

Allowed metamer-target layers (from `AVModelSpec.allowed_metamer_layers`):

```
audio.embed, video.embed,
audio.transformer.{0..5}, video.transformer.{0..5},
fusion.cross_attn.{0..5},
logits
```

### 1.2 Reference loading

- Audio: WAV → mono-mix → resample to 16 kHz → clip/pad to 2 s →
  tensor shape `(1, 1, 32000)`.
- Video: MP4 → sample at 25 fps → resize to 112×112 → 50 frames →
  tensor shape `(1, 3, 50, 112, 112)`.

### 1.3 Target activation

```python
with torch.no_grad():
    _, ref_outs = model(ref, with_latent=True)
A_ref = ref_outs[args.layer].detach()
```

### 1.4 Metamer initialization

Controlled by `--init`:
- `noise` (default): Gaussian, scaled `0.3` for audio and `0.1` for video.
- `ref`: start from a copy of the reference.

### 1.5 PGD inner loop (`n_iters=3000` by default)

Each step:

1. Forward pass → `outs = model(x, with_latent=True, fake_relu=True)`.
2. Loss (scale-invariant L2 match):

    ```
    L = ||A_ref − A_cur||₂ / ||A_ref||₂
    ```

3. Optional semantic-consistency regularizer: cosine distance between
   mean-pooled audio and video tokens at layer 5.
4. Backprop w.r.t. whichever modality is "active" (`_active_keys(mode)`):
   - `single-audio` → only audio.
   - `single-video` → only video.
   - `joint` → both.
5. Gradient step (L2-normalized) and projection onto an ε-ball.
   Default ε is `1e3` per branch — effectively unbounded, which is what
   you want for **metamers** (not adversarial examples).

### 1.6 Output

Saves to `--out-dir`:
- `metamer.pt` — raw tensor dict.
- `metamer_audio.wav` — 16 kHz int16.
- `metamer_frames/frame_%04d.png` — one per frame.
- `match_distance.txt` — final value of the match loss at the target
  layer.

### 1.7 Known scaffold issues to audit before trusting results

- Audio frontend is **STFT magnitude**, not a cochleagram. Swapping in
  a cochleagram is an explicit planned adaptation.
- `fake_relu=True` is passed through, but transformers use GELU /
  softmax / LayerNorm. Feather's fake-ReLU gradient trick doesn't apply
  cleanly here — audit what `AVMetamerModel.forward` actually does with
  that flag.
- Defaults (`step_size=1e-2` both branches, noise init, 3000 iters)
  are CNN-era defaults. Transformer inversion typically needs tuning
  (see §2.5).

---

## 2. CNN vs Transformer — how the metamers differ

### 2.1 Structural contrast

| Property                | CNN (Kell / CochCNN, ResNets)                           | Transformer (your AV model, AV-HuBERT, ViT)                     |
|-------------------------|---------------------------------------------------------|-----------------------------------------------------------------|
| Receptive field         | Grows gradually with depth; each unit sees a local patch | **Global from layer 1** — every token attends everywhere        |
| Per-layer computation   | Fixed conv kernels, weight-sharing → translation-equivariant | Input-dependent attention weights → no fixed spatial/temporal prior |
| Base unit               | Pixel / waveform sample                                 | Token (STFT bin, frame patch) — coarser                         |
| Cross-modal mixing      | Typically only at late pooling / fc                     | Happens every fusion block; audio gradients flow into video reps |

### 2.2 Progression with depth

Feather's central result — metamers from **late CNN** layers diverge
increasingly from human-recognizable stimuli while **early
(cochleagram / conv1) metamers remain recognizable** — relies on the
receptive-field hierarchy. Transformers do not have that hierarchy;
every layer is already global. Expect the "recognizability falls with
depth" curve to be **flatter and less monotonic** for a transformer.

### 2.3 Character of early-layer metamers

- **CNN conv1 metamers** preserve local spectro-temporal texture;
  humans can still identify the word / face.
- **Transformer layer-1 metamers** preserve *global*, attention-weighted
  summaries — they can look / sound less locally textured but more
  holistically structured than a comparable-depth CNN layer.

### 2.4 Character of late-layer metamers

Late-layer transformer metamers may be **more** human-recognizable, not
less, than CNN late-layer metamers. Recent ViT-metamer work suggests
attention-based invariances sometimes align better with human-relevant
global structure than CNN "texture" invariances. This is a hypothesis
to test head-on vs. CochCNN on matched stimuli.

### 2.5 Cross-modal leakage in joint mode

In `--mode joint`, because `fusion.cross_attn.*` mixes modalities at
every block, an audio gradient changes the video activation and vice
versa. A CNN fusion model with late concat does not do this. So
transformer joint metamers probe a genuinely **bimodal representation**
— the scientifically interesting contrast.

In `single-audio` / `single-video` mode on a transformer you are
isolating one modality's contribution to a **jointly-computed**
activation — still different from doing the same operation on a
late-fusion CNN.

### 2.6 Optimization difficulty

Transformer loss landscapes for activation inversion are bumpier than
CNN ones. Expect to:

- Tune `step_sizes` per modality (the `1e-2` defaults are a starting
  point, not a final choice).
- Use a cosine or linear LR schedule rather than a constant step.
- Initialize from `ref + small noise` rather than pure Gaussian noise
  for layers deep in the stack.
- Run more iterations than the 3000-iter CNN default.

### 2.7 Why the cochleagram frontend matters

The Feather story requires a **biologized lowest stage** — a
representation at which humans and models are known to agree
(the cochlea). Metamer divergence is then attributable to **learned
layers**, not the input transform.

An STFT-magnitude frontend is not biologized: it is a linear
time-frequency representation with uniform frequency resolution. A
cochleagram (ERB-spaced gammatone filterbank, half-wave rectification,
compression) matches peripheral auditory physiology and provides the
correct baseline against which to measure divergence at learned layers.
Swapping STFT → cochleagram is therefore not a nicety — it's what makes
the "where does the model diverge from humans?" question well-posed.

---

## 3. What the artifacts should be in a real run

Assumptions for a *real* run (as distinct from the smoke tests in
`smoke_test/` and `smoke_test_long/`): a trained checkpoint is loaded,
`--ref-audio` / `--ref-video` point at a real `(wav, mp4)` pair,
`--init ref` (or `noise` if you want a stricter test), GPU execution,
`--n-iters ≥ 3000`.

### 3.1 `metamer.pt`

- `{"audio": (1,1,32000) float32, "video": (1,3,50,112,112) float32}` —
  same shape as the smoke test, **but value distributions change**.
- **Audio**: roughly matches the reference RMS. Because the model
  normalizes by `AUDIO_STD=0.1`, the optimizer steers toward inputs
  that produce the right *normalized* activations; raw values end up
  roughly in `[-1, 1]`.
- **Video**: should stay in `[0, 1]`. If you see negatives or values
  > 1 in a real run, the optimizer drifted off the valid pixel
  manifold — that's a bug, not a metamer. (The smoke test's
  `[0, ~128]` PNG range is the flagged save-path issue from
  `SMOKE_TEST_NOTES.md` §4 — it goes away with real `[0, 1]` video
  input.)

### 3.2 `metamer_audio.wav`

What it should *sound like* depends entirely on the **target layer**:

| Target layer             | Expected percept                                                                                          |
|--------------------------|-----------------------------------------------------------------------------------------------------------|
| `audio.embed` (early)    | Near-indistinguishable from the reference — essentially STFT inversion                                     |
| `audio.transformer.0–2`  | Still speech-like; prosody and rough word shape preserved; fine phonetic detail may drift                  |
| `audio.transformer.3–5`  | Speech-like but potentially unintelligible; "someone talking" percept without the original words           |
| `fusion.cross_attn.*`    | Mixed — depends on whether the information the fusion block needs from audio is word-content, envelope, or both |
| `logits`                 | Can be almost any sound that elicits the same class distribution — often noise-like, potentially nothing like speech |

For a CNN (CochCNN / CochResNet) you'd see the same trend (early ≈ ref,
late → textural noise). For a transformer the trend is **flatter** —
because every layer is already global, late-layer audio may sound more
holistically speech-like than a late CNN metamer, not less.

### 3.3 `metamer_frames/`

- 50 PNGs that played at 25 fps give 2 s of video.
- **Early layers** → near-copy of the reference face / mouth.
- **Mid layers** → correct face layout and mouth-open / mouth-closed
  gestures but with drift in identity, lighting, background detail.
- **Late fusion layers** → may preserve only *the information the audio
  branch needed*, e.g. lip-aperture sequence but scrambled identity.

To view as video:

```
ffmpeg -r 25 -i metamer_frames/frame_%04d.png -c:v libx264 metamer.mp4
```

To mux with the matching audio metamer:

```
ffmpeg -r 25 -i metamer_frames/frame_%04d.png -i metamer_audio.wav \
       -c:v libx264 -c:a aac -shortest metamer_av.mp4
```

### 3.4 `match_distance.txt`

**This is the single most important diagnostic.** It is the final value
of `||A_ref − A_metamer||₂ / ||A_ref||₂` at the target layer. For a
successful metamer it should be **small** — Feather's convention is
"within the null distribution of natural-input pairs at that same
layer" (see §4.2).

Rough rules of thumb:

- `< 0.01` at early layers (`*.embed`) → pipeline working.
- `< 0.05` at mid layers → good metamer.
- `< 0.1` at late / fusion layers → acceptable.

The smoke-test values (`0.216` at 20 iters, `0.136` at 1000 iters,
both at `fusion.cross_attn.2`) are **not interpretable** — no
checkpoint, no real target. On real data with a trained model, 3000
iters should drive this well below 0.1.

---

## 4. How to verify the pipeline actually works

Run these in order — each one isolates a different failure mode.

### 4.1 Check 1 — Early-layer inversion sanity (must pass)

```
--layer audio.embed --mode single-audio --init ref --n-iters 500
```

Expected: `match_distance` drops to ≈ 1e-4 or lower. If it plateaus
high, the gradient path through `AudioSTFT` → Linear is broken.

Same for video: `--layer video.embed --mode single-video --init ref`.

This is the fastest "is anything broken?" test — under 30 s on GPU.

### 4.2 Check 2 — Null-distribution baseline

Generate 10 pairs of metamers from **different** reference pairs at the
same target layer. Compute pairwise match distances between `metamer_i`
activations and `ref_j` activations for `i ≠ j`. Call this distribution
`D_null`. A successful metamer must have

```
match_distance(metamer_i, ref_i) << percentile(D_null, 5)
```

`analysis_scripts/make_null_distributions.py` is the Feather-era
template — adapt it for the AV dict-valued input.

### 4.3 Check 3 — Loss curve shape

Dump `history` from `synthesize_metamer` to a PNG every run. Eyeball:

- **Monotonic descent, tapering to near-zero by the last 10% of iters**
  → healthy.
- **Early plateau that never descends** → step size too small, init
  wrong, or dead gradient (check `fake_relu` behavior).
- **Descends then spikes** → step size too large, or ε-ball projection
  kicking in (it shouldn't at `eps=1e3`).
- **Oscillates** → reduce step, add cosine schedule.

### 4.4 Check 4 — Activation identity (not just loss)

After the run, re-forward both ref and metamer and compare the **full
activation tensors** at the target layer — not just the scalar match
distance:

```python
d = (A_ref - A_met)
d.abs().max(), d.abs().mean()
```

Scalar match loss can be small while a few units diverge wildly. Both
max and mean should be small.

### 4.5 Check 5 — Human recognizability (the actual scientific point)

The payoff of a metamer is: "do humans perceive this the same way the
model does?" For audio, can a listener transcribe the word? For video,
identify the speaker / mouth shape? For AV, get the McGurk percept
right?

This is what `AuditoryBehavioralExperiments/` and
`VisionBehavioralExperiments/` exist for — they're the Feather-lab
psychophysics harness. A pipeline that passes Checks 1–4 but whose
metamers are unintelligible to humans at a layer where CochCNN
metamers are intelligible is **scientifically meaningful** (that's the
finding), not a bug.

### 4.6 Check 6 — Cross-layer sanity progression

Generate metamers at `audio.transformer.0, 2, 5` with the same
reference. Match distance should be achievable at all three; but
intelligibility / recognizability to the listener should degrade with
depth — or behave differently than for a CNN. That difference is the
hypothesis you are testing.

### 4.7 What each failure rules out

| Check fails              | Likely cause                                                                     |
|--------------------------|----------------------------------------------------------------------------------|
| 1 (early inversion)      | Broken gradient path, `fake_relu` misapplied, wrong normalization                 |
| 2 (null distribution)    | Metamer isn't actually layer-specific — distance matches a random pair            |
| 3 (loss curve)           | Optimizer hyperparameters wrong for transformer landscape                         |
| 4 (activation max-abs)   | Local minimum that matches mean but not tails                                     |
| 5 (behavior)             | Scientific result, not a bug (assuming 1–4 passed)                                |
| 6 (progression)          | No usable depth-dependent signal — try more iters or different loss weighting     |

**Bottom line**: `match_distance.txt` + a loss-curve plot catch ~80 %
of pipeline failures. Check 1 with `--init ref` on an `*.embed` layer
is the single fastest end-to-end correctness test.

---

## 5. Scaffold audit items (before any full-scale run)

1. **Early-layer inversion should be near-perfect.** Generate metamers
   targeting `audio.embed` / `video.embed` with `--init ref` — the
   match distance should fall to ~0 quickly. If not, the gradient path
   is broken somewhere (likely the STFT → Linear block or the 3D conv
   stem).
2. **Plot `history`.** With noise init + 3000 iters on
   `fusion.cross_attn.*`, confirm the loss actually converges rather
   than plateauing high.
3. **Audit `fake_relu` in `AVMetamerModel.forward`.** Confirm either
   that it is applied to the right nonlinearities, or that it's a
   no-op for a transformer (in which case the flag is dead code, not
   silently wrong).
4. **Verify the ε-ball is large enough to be inactive.** With
   `eps=1e3`, projection should essentially never bind; if it does, the
   optimizer is being clipped and you are not generating a true
   metamer.
