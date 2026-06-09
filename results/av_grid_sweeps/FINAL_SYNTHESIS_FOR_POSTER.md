# Final Synthesis (Poster-Ready)

Source: `results/av_grid_sweeps/summary_runs.csv`  
Sweep size: 24 runs (complete)

## 1) What one run name means

Example run: `iter1500_step0p01_initnoise_gl32_seed2_eps1000p0`

This run name decodes to:

- `iter1500`: optimize each metamer for 1500 gradient steps
- `step0p01`: step size = 0.01
- `initnoise`: initialize optimization from noise (not reference)
- `gl32`: Griffin-Lim reconstruction uses 32 iterations
- `seed2`: random seed = 2
- `eps1000p0`: L2 epsilon bound = 1000.0 (effectively unconstrained for this setup)

What this run *represents* experimentally:

- A full pass over all 126 clips (`42 stimuli x 3 splits`) under one fixed hyperparameter setting.
- For each clip, metamer optimization is run with the same setting, then summarized into category means and an overall mean.

## 2) Full list of all 24 runs and parameters

Each line shows: run name, tweaked parameters, and overall mean match distance.

1. `iter1500_step0p003_initnoise_gl32_seed0_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.003`, `init=noise`, `gl_iters=32`, `seed=0`, `eps=1000.0`
  - overall mean: `0.106698` | env/music/speech: `0.074360 / 0.126660 / 0.136958`
2. `iter1500_step0p003_initnoise_gl32_seed1_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.003`, `init=noise`, `gl_iters=32`, `seed=1`, `eps=1000.0`
  - overall mean: `0.104323` | env/music/speech: `0.075268 / 0.121293 / 0.132863`
3. `iter1500_step0p003_initnoise_gl32_seed2_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.003`, `init=noise`, `gl_iters=32`, `seed=2`, `eps=1000.0`
  - overall mean: `0.105054` | env/music/speech: `0.073365 / 0.124521 / 0.134841`
4. `iter1500_step0p003_initnoise_gl64_seed0_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.003`, `init=noise`, `gl_iters=64`, `seed=0`, `eps=1000.0`
  - overall mean: `0.106698` | env/music/speech: `0.074360 / 0.126660 / 0.136958`
5. `iter1500_step0p003_initnoise_gl64_seed1_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.003`, `init=noise`, `gl_iters=64`, `seed=1`, `eps=1000.0`
  - overall mean: `0.104323` | env/music/speech: `0.075268 / 0.121293 / 0.132863`
6. `iter1500_step0p003_initnoise_gl64_seed2_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.003`, `init=noise`, `gl_iters=64`, `seed=2`, `eps=1000.0`
  - overall mean: `0.105054` | env/music/speech: `0.073365 / 0.124521 / 0.134841`
7. `iter1500_step0p01_initnoise_gl32_seed0_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.01`, `init=noise`, `gl_iters=32`, `seed=0`, `eps=1000.0`
  - overall mean: `0.075083` | env/music/speech: `0.056577 / 0.088948 / 0.088984`
8. `iter1500_step0p01_initnoise_gl32_seed1_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.01`, `init=noise`, `gl_iters=32`, `seed=1`, `eps=1000.0`
  - overall mean: `0.074705` | env/music/speech: `0.056290 / 0.088268 / 0.088864`
9. `iter1500_step0p01_initnoise_gl32_seed2_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.01`, `init=noise`, `gl_iters=32`, `seed=2`, `eps=1000.0`
  - overall mean: `0.074377` | env/music/speech: `0.054756 / 0.089070 / 0.089124`
10. `iter1500_step0p01_initnoise_gl64_seed0_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.01`, `init=noise`, `gl_iters=64`, `seed=0`, `eps=1000.0`
  - overall mean: `0.075083` | env/music/speech: `0.056577 / 0.088948 / 0.088984`
11. `iter1500_step0p01_initnoise_gl64_seed1_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.01`, `init=noise`, `gl_iters=64`, `seed=1`, `eps=1000.0`
  - overall mean: `0.074705` | env/music/speech: `0.056290 / 0.088268 / 0.088864`
12. `iter1500_step0p01_initnoise_gl64_seed2_eps1000p0`
  - params: `n_iters=1500`, `step_size=0.01`, `init=noise`, `gl_iters=64`, `seed=2`, `eps=1000.0`
  - overall mean: `0.074377` | env/music/speech: `0.054756 / 0.089070 / 0.089124`
13. `iter500_step0p003_initnoise_gl32_seed0_eps1000p0`
  - params: `n_iters=500`, `step_size=0.003`, `init=noise`, `gl_iters=32`, `seed=0`, `eps=1000.0`
  - overall mean: `0.214384` | env/music/speech: `0.141286 / 0.248590 / 0.298074`
14. `iter500_step0p003_initnoise_gl32_seed1_eps1000p0`
  - params: `n_iters=500`, `step_size=0.003`, `init=noise`, `gl_iters=32`, `seed=1`, `eps=1000.0`
  - overall mean: `0.212206` | env/music/speech: `0.143684 / 0.241390 / 0.294688`
15. `iter500_step0p003_initnoise_gl32_seed2_eps1000p0`
  - params: `n_iters=500`, `step_size=0.003`, `init=noise`, `gl_iters=32`, `seed=2`, `eps=1000.0`
  - overall mean: `0.211686` | env/music/speech: `0.140868 / 0.244158 / 0.293699`
16. `iter500_step0p003_initnoise_gl64_seed0_eps1000p0`
  - params: `n_iters=500`, `step_size=0.003`, `init=noise`, `gl_iters=64`, `seed=0`, `eps=1000.0`
  - overall mean: `0.214384` | env/music/speech: `0.141286 / 0.248590 / 0.298074`
17. `iter500_step0p003_initnoise_gl64_seed1_eps1000p0`
  - params: `n_iters=500`, `step_size=0.003`, `init=noise`, `gl_iters=64`, `seed=1`, `eps=1000.0`
  - overall mean: `0.212206` | env/music/speech: `0.143684 / 0.241390 / 0.294688`
18. `iter500_step0p003_initnoise_gl64_seed2_eps1000p0`
  - params: `n_iters=500`, `step_size=0.003`, `init=noise`, `gl_iters=64`, `seed=2`, `eps=1000.0`
  - overall mean: `0.211686` | env/music/speech: `0.140868 / 0.244158 / 0.293699`
19. `iter500_step0p01_initnoise_gl32_seed0_eps1000p0`
  - params: `n_iters=500`, `step_size=0.01`, `init=noise`, `gl_iters=32`, `seed=0`, `eps=1000.0`
  - overall mean: `0.123906` | env/music/speech: `0.088180 / 0.145131 / 0.158499`
20. `iter500_step0p01_initnoise_gl32_seed1_eps1000p0`
  - params: `n_iters=500`, `step_size=0.01`, `init=noise`, `gl_iters=32`, `seed=1`, `eps=1000.0`
  - overall mean: `0.124186` | env/music/speech: `0.088503 / 0.144854 / 0.159481`
21. `iter500_step0p01_initnoise_gl32_seed2_eps1000p0`
  - params: `n_iters=500`, `step_size=0.01`, `init=noise`, `gl_iters=32`, `seed=2`, `eps=1000.0`
  - overall mean: `0.123306` | env/music/speech: `0.087097 / 0.144807 / 0.158379`
22. `iter500_step0p01_initnoise_gl64_seed0_eps1000p0`
  - params: `n_iters=500`, `step_size=0.01`, `init=noise`, `gl_iters=64`, `seed=0`, `eps=1000.0`
  - overall mean: `0.123906` | env/music/speech: `0.088180 / 0.145131 / 0.158499`
23. `iter500_step0p01_initnoise_gl64_seed1_eps1000p0`
  - params: `n_iters=500`, `step_size=0.01`, `init=noise`, `gl_iters=64`, `seed=1`, `eps=1000.0`
  - overall mean: `0.124186` | env/music/speech: `0.088503 / 0.144854 / 0.159481`
24. `iter500_step0p01_initnoise_gl64_seed2_eps1000p0`
  - params: `n_iters=500`, `step_size=0.01`, `init=noise`, `gl_iters=64`, `seed=2`, `eps=1000.0`
  - overall mean: `0.123306` | env/music/speech: `0.087097 / 0.144807 / 0.158379`

## 3) Best/Worst settings

### Best overall (lowest mean match distance)

1. `iter1500_step0p01_initnoise_gl32_seed2_eps1000p0`
  - overall: `0.074377`  
  - env/music/speech: `0.054756 / 0.089070 / 0.089124`
2. `iter1500_step0p01_initnoise_gl64_seed2_eps1000p0`
  - overall: `0.074377`  
  - env/music/speech: `0.054756 / 0.089070 / 0.089124`
3. `iter1500_step0p01_initnoise_gl32_seed1_eps1000p0`
  - overall: `0.074705`  
  - env/music/speech: `0.056290 / 0.088268 / 0.088864`
4. `iter1500_step0p01_initnoise_gl64_seed1_eps1000p0`
  - overall: `0.074705`  
  - env/music/speech: `0.056290 / 0.088268 / 0.088864`
5. `iter1500_step0p01_initnoise_gl32_seed0_eps1000p0`
  - overall: `0.075083`  
  - env/music/speech: `0.056577 / 0.088948 / 0.088984`

### Worst overall (highest mean match distance)

1. `iter500_step0p003_initnoise_gl64_seed0_eps1000p0`
  - overall: `0.214384`  
  - env/music/speech: `0.141286 / 0.248590 / 0.298074`
2. `iter500_step0p003_initnoise_gl32_seed0_eps1000p0`
  - overall: `0.214384`  
  - env/music/speech: `0.141286 / 0.248590 / 0.298074`
3. `iter500_step0p003_initnoise_gl64_seed1_eps1000p0`
  - overall: `0.212206`  
  - env/music/speech: `0.143684 / 0.241390 / 0.294688`
4. `iter500_step0p003_initnoise_gl32_seed1_eps1000p0`
  - overall: `0.212206`  
  - env/music/speech: `0.143684 / 0.241390 / 0.294688`
5. `iter500_step0p003_initnoise_gl64_seed2_eps1000p0`
  - overall: `0.211686`  
  - env/music/speech: `0.140868 / 0.244158 / 0.293699`

### End-to-end spread

- Best vs worst relative improvement: **65.3% lower** overall mean match distance.

## 4) Main category trends

Across all 24 runs:

- Environment is consistently easiest to match.
- Music and speech are consistently harder than environment.

Global mean by category (averaged across runs):

- Environment: ~`0.0900`
- Music: ~`0.1506`
- Speech: ~`0.1695`

Environment-minus-speech gap:

- Mean gap: `-0.0795` (environment < speech in every run)
- Range: `-0.1568` to `-0.0324`

## 5) Hyperparameter effects

### Iterations (`500` -> `1500`)

Overall mean:

- `0.16828` -> `0.09004` (**46.5% reduction**)

By category:

- Environment: **43.4% reduction**
- Music: **45.4% reduction**
- Speech: **50.7% reduction**

### Step size (`0.003` -> `0.01`)

Overall mean:

- `0.15906` -> `0.09926` (**37.6% reduction**)

By category:

- Environment: **33.5% reduction**
- Music: **36.6% reduction**
- Speech: **42.4% reduction**

### Griffin-Lim iterations (`32` vs `64`)

- No measurable difference in `summary_runs.csv` metrics for this sweep.
- Interpretation: current summary metrics are activation-distance based, so GL iterations do not shift these aggregate numbers.

## 6) Stability across seeds

Within each fixed setting (`n_iters`, `step_size`, `gl_iters`, `eps`, `init`), seed variance is low:

- `(500, 0.003, 32, 1000.0, 'noise')` -> overall mean std across seeds: `0.001169`
- `(500, 0.003, 64, 1000.0, 'noise')` -> overall mean std across seeds: `0.001169`
- `(500, 0.01, 32, 1000.0, 'noise')` -> overall mean std across seeds: `0.000367`
- `(500, 0.01, 64, 1000.0, 'noise')` -> overall mean std across seeds: `0.000367`
- `(1500, 0.003, 32, 1000.0, 'noise')` -> overall mean std across seeds: `0.000993`
- `(1500, 0.003, 64, 1000.0, 'noise')` -> overall mean std across seeds: `0.000993`
- `(1500, 0.01, 32, 1000.0, 'noise')` -> overall mean std across seeds: `0.000289`
- `(1500, 0.01, 64, 1000.0, 'noise')` -> overall mean std across seeds: `0.000289`
- Std range across settings: `0.000289` to `0.001169`

## 7) Poster-ready takeaway block

1. **Robust trend:** The `environment < music < speech` ordering is stable across all 24 runs.
2. **Optimization matters:** Increasing both optimization depth (`1500` iters) and step size (`0.01`) substantially improves metamer matching.
3. **Best recipe in this grid:** `n_iters=1500`, `step_size=0.01`, `init=noise`, `eps=1000` (seed choice has minor effect).
4. **Interpretation:** Even in the best settings, environment remains easier than speech/music, consistent with speech-focused representational bias in this backbone.

## 8) Compact appendix: run groups by sweep axis

This section groups the 24 runs by each axis so the sweep design is easy to scan.

### By `n_iters`

- `n_iters=500` (12 runs): all combinations of `step_size in {0.003, 0.01}`, `gl_iters in {32, 64}`, `seed in {0,1,2}`.
- `n_iters=1500` (12 runs): all combinations of `step_size in {0.003, 0.01}`, `gl_iters in {32, 64}`, `seed in {0,1,2}`.

### By `step_size`

- `step_size=0.003` (12 runs): all combinations of `n_iters in {500,1500}`, `gl_iters in {32,64}`, `seed in {0,1,2}`.
- `step_size=0.01` (12 runs): all combinations of `n_iters in {500,1500}`, `gl_iters in {32,64}`, `seed in {0,1,2}`.

### By `gl_iters`

- `gl_iters=32` (12 runs): all combinations of `n_iters in {500,1500}`, `step_size in {0.003,0.01}`, `seed in {0,1,2}`.
- `gl_iters=64` (12 runs): all combinations of `n_iters in {500,1500}`, `step_size in {0.003,0.01}`, `seed in {0,1,2}`.

### By `seed`

- `seed=0` (8 runs): all combinations of `n_iters in {500,1500}`, `step_size in {0.003,0.01}`, `gl_iters in {32,64}`.
- `seed=1` (8 runs): same grid structure.
- `seed=2` (8 runs): same grid structure.

### Constants across all runs

- `init=noise` for all 24 runs.
- `eps=1000.0` for all 24 runs.
- Dataset per run: 126 clips (`42 stimuli x 3 splits`).

## 9) Mentor-ready plain-language summary

You can send this directly to your mentor:

> I completed a full 24-condition metamer sweep on the deep AVSR backbone, where each run uses one fixed optimization setting and covers all 126 clips (42 stimuli x 3 splits). I systematically varied optimization depth (`500` vs `1500` steps), step size (`0.003` vs `0.01`), Griffin-Lim iterations (`32` vs `64`), and random seed (`0/1/2`). The main result is stable across all conditions: environmental sounds are consistently easier to match than music and speech, while stronger optimization settings substantially improve overall matching quality. The best settings were `1500` iterations with `0.01` step size, and seed effects were small, suggesting the pattern is robust rather than a random artifact.

