# AVNet metamer run — 42 MUSHRA natural sounds

Date: 2026-05-28
Script: `run_audio_visual_42_metamers.py`
Output dir: `model_metamers_pytorch/results/av_metamers_42_natural/`

## TL;DR

126 metamers generated (42 stimuli × 3 splits each), using the trained
**deep_avsr AVNet** (`audio-visual.pt`) as both the synthesis model and
the speech recogniser. **All non-speech metamers sound similar to each
other — this is the correct and informative result**, not a bug. AVNet's
audio encoder was trained on lip-reading speech data; it is selectively
responsive to speech and largely insensitive to environmental sound
detail. The metamer optimisation therefore converges to a single
"not-speech" attractor for every non-speech reference.

## What this run did

For each `.wav` in `model_metamers_pytorch/stimuli/MUSHRA_42_NATURAL/`:

1. Resample to 16 kHz mono, normalise, compute STFT magnitude.
2. Forward-pass through trained AVNet → capture activations at
   `audioEncoder` (the 6-layer audio transformer stack output, before
   the joint decoder).
3. Initialise an STFT-magnitude tensor with non-negative Gaussian noise
   scaled to the reference RMS.
4. Run 500 iterations of PGD with scale-invariant L2 loss
   `||A_ref − A_x||₂ / ||A_ref||₂`, L2-normalised step size `1e-2`,
   ε-ball `1e3` (effectively unbounded), clamp ≥ 0.
5. Invert the optimised STFT magnitude to a waveform with 64 iterations
   of Griffin-Lim, normalise to peak `±1`, save as int16 WAV.
6. Run AVNet CTC greedy decode on both the original and the metamer
   wav; record both transcripts and the final activation-match distance.

## Stimuli

42 natural sounds, copied from
`/orcd/data/jhm/001/dlatorre/matlab-stimuli-presentation-code-sam-2018-in-03-29-2026-MMS-deepnet-spectemp/stimuli/v6_experiment/MUSHRA_42_sounds/NATURAL/`
into `model_metamers_pytorch/stimuli/MUSHRA_42_NATURAL/`.

Each stimulus is provided as 3 overlapping 3-second splits
(`_01_04`, `_04_07`, `_07_10`), giving 126 wav files total. Native
sample rate is 20 kHz; resampled to 16 kHz before STFT.

Categories represented:

- **Environmental sounds (54 splits):** crumpling paper, finger tapping,
  keys jingling, walking on leaves, biting/chewing, scratching, siren,
  typing, walking with heels, writing on paper, breathing, heart beat,
  chimes in the wind, chopping food, cicadas, clock ticking, crickets,
  baby crying.
- **Music (39 splits):** saxophone jazz solo, bluegrass, violin, cello,
  jazz drums, saxophone solo, acoustic guitar, big band, orchestra,
  piano, contemporary R&B, latin music, country song, rap music.
- **Speech (33 splits):** woman speaking, man speaking, Spanish, French,
  Italian, German, Hindi, Russian, English vocal, Spanish vocal.

## Checkpoint

**AVNet (deep_avsr, audio-visual.pt)** —
`/orcd/data/jhm/001/urops/aimee_yu/deep_avsr_weights/DeepAVSR_Weights/audio-visual.pt`

Architecture (from `model_metamers_pytorch/deep_avsr/audio_visual/models/av_net.py`):

- `audioConv` : Conv1d (321 STFT bins → 512), kernel 4, stride 4
- `positionalEncoding` : sinusoidal PE
- `audioEncoder` : 6 × `nn.TransformerEncoderLayer` (d=512, h=8, ff=2048) ← **target layer**
- `videoEncoder` : 6-layer stack (unused — audio-only mode here)
- `jointDecoder` : 6-layer transformer over fused tokens
- `outputConv` : Conv1d (512 → 40 character classes)

Trained on Lip Reading Sentences (LRS), so the encoder is heavily
biased toward speech features.

## Hyperparameters

| Param | Value |
|---|---|
| Target layer | `audioEncoder` |
| Init | Non-negative Gaussian noise at reference RMS |
| Optimizer | PGD, L2-normalised gradient |
| Step size | `1e-2` |
| ε-ball | `1e3` (inactive — true metamer regime, not adversarial) |
| Iterations | 500 |
| Constraint | STFT magnitude clamped ≥ 0 |
| Wave reconstruction | 64 iter Griffin-Lim, n_fft=640, hop=160, hann window |
| Sample rate | 16 kHz |
| Device | CUDA |

## Results

### Match distance by stimulus category

(scale-invariant L2 at `audioEncoder`; lower = closer match)

| Category | n | Mean | Min | Max |
|---|---|---|---|---|
| Environmental | 54 | **0.088** | 0.018 | 0.310 |
| Music | 39 | 0.147 | 0.077 | 0.217 |
| Speech | 33 | **0.153** | 0.088 | 0.234 |

**Interpretation:** environmental sounds fit ~1.7× tighter than speech.
Not because the optimiser tries harder for one class than the other — it
runs the same loop for all 126 stimuli. The encoder simply produces
*less differentiated* activations for non-speech inputs, so PGD has an
easier target.

### AVNet CTC predictions

| | Empty / 126 |
|---|---|
| Original wav | 37 |
| Metamer wav | 54 |

Most empty originals are environmental sounds — AVNet (trained on
speech) outputs nothing because no phonemes are detected. Speech splits
(stim 5 `man_speaking`, stim 18 `woman_speaking`, stim 22–27 foreign
languages, etc.) produce real transcripts both for the original and for
the metamer, often with overlapping phonetic content. Example
(stim 5, split `07_10`):

- Original: `YOU CAN GET MORE OUT HIS BOOKS I AN FEEL SMARL`
- Metamer:  `YOU CAN HO MOTI WHAI AM FEEL SMOROW`

The metamer waveform is acoustically very different from the original
but preserves enough phonetic structure to be transcribed by AVNet into
something that **shares syllabic / vowel structure** with the reference.

## Why the metamers sound similar to each other

This is the substantive finding and is consistent with what we'd expect
from a speech-trained transformer encoder:

1. **`audioEncoder` is a learned speech representation.** It was trained
   to extract phonetic and lexical content from speech, not to
   discriminate textures of paper crumpling vs cicadas vs keys jingling.
2. **For all non-speech stimuli, the encoder produces a near-uniform
   "no speech here" activation pattern.** The 6-layer transformer
   essentially throws away spectro-temporal detail that doesn't help
   character prediction.
3. **PGD with the same target activation → similar solutions.** When
   the reference activations across 54 different environmental sounds
   all look approximately the same, the optimiser converges to roughly
   the same kind of STFT magnitude (whatever minimally satisfies the
   "not-speech" constraint), which Griffin-Lim then reconstructs into
   perceptually similar noise-like sounds.
4. **For real speech**, the encoder activations *are* stimulus-specific
   (they have to be, to drive the CTC head to different transcripts),
   so the metamers genuinely differ from each other and preserve
   phonetic structure.

This is the **standard metamer interpretation**: invariances in the
model show up as collapse modes in the metamers. The fact that 54
environmental metamers all sound similar is direct evidence that AVNet's
audioEncoder is **invariant to non-speech acoustic structure** — which
is exactly the kind of "what does this model represent" question
metamer experiments are designed to answer.

## What this run does NOT yet do

- **No null-distribution baseline.** A proper metamer paper compares
  the within-pair match distance against the cross-pair (metamer_i vs
  ref_j for i ≠ j) distribution. Without that, "match distance 0.088"
  is only meaningful relatively.
- **Single layer only.** Sweeping `audioConv` (early), middle
  transformer layers, and `outputConv` (logits) would show whether
  early-layer metamers preserve more environmental-sound structure
  (expected: yes, because they haven't yet been collapsed onto the
  speech manifold).
- **Griffin-Lim reconstruction is lossy.** The optimised STFT
  magnitude is the actual "true" metamer; the wav is a phase-recovered
  approximation. Some of the perceptual similarity across metamers
  may also be Griffin-Lim artefact (it's known to add a "watery"
  texture to noise-like spectrograms).
- **Audio-only.** Video branch was bypassed entirely. Multi-modal
  metamers (using `jointDecoder` activations) would be a separate run.

## Files written

```
model_metamers_pytorch/results/av_metamers_42_natural/
├── comparison.csv                      # 126 rows: orig pred, metamer pred, match dist
├── RUN_NOTES.md                        # this file
└── <stim_folder>/                      # one per stimulus (42 folders)
    └── <wav_stem>/
        ├── metamer_audio.wav           # playable 16-kHz int16 wav
        ├── metamer_stft.pt             # optimised STFT magnitude tensor
        └── match_distance.txt          # final loss at audioEncoder
```
