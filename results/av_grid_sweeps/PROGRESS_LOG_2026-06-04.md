# AV Grid Sweep Progress Log (2026-06-04 17:40 local)

## Current status

- Grid spec: `av_metamer_grid_default.json`
- Expected total runs: `24`
- Completed runs (`comparison.csv` present): `11`
- Remaining runs: `13`
- Active partial run:
  - `iter500_step0p01_initnoise_gl64_seed2_eps1000p0` (`52/126` metamers written at snapshot time)

## Remaining runs (in order)

1. `iter500_step0p01_initnoise_gl64_seed2_eps1000p0` (partial, currently running)
2. `iter1500_step0p003_initnoise_gl32_seed0_eps1000p0`
3. `iter1500_step0p003_initnoise_gl32_seed1_eps1000p0`
4. `iter1500_step0p003_initnoise_gl32_seed2_eps1000p0`
5. `iter1500_step0p003_initnoise_gl64_seed0_eps1000p0`
6. `iter1500_step0p003_initnoise_gl64_seed1_eps1000p0`
7. `iter1500_step0p003_initnoise_gl64_seed2_eps1000p0`
8. `iter1500_step0p01_initnoise_gl32_seed0_eps1000p0`
9. `iter1500_step0p01_initnoise_gl32_seed1_eps1000p0`
10. `iter1500_step0p01_initnoise_gl32_seed2_eps1000p0`
11. `iter1500_step0p01_initnoise_gl64_seed0_eps1000p0`
12. `iter1500_step0p01_initnoise_gl64_seed1_eps1000p0`
13. `iter1500_step0p01_initnoise_gl64_seed2_eps1000p0`

## Resume commands for tomorrow

From repo root (`/orcd/home/002/aimeeyu/mms_project`):

```bash
PYTHONUNBUFFERED=1 python -u run_av_metamer_grid.py --resume
```

If you want it in a detached session:

```bash
nohup PYTHONUNBUFFERED=1 python -u run_av_metamer_grid.py --resume > sweep_resume.log 2>&1 &
```

## Output locations

- Per-run outputs:
  - `model_metamers_pytorch/results/av_grid_sweeps/<run_name>/`
- Aggregate outputs (auto-updated as runs finish):
  - `model_metamers_pytorch/results/av_grid_sweeps/summary_runs.csv`
  - `model_metamers_pytorch/results/av_grid_sweeps/plots/`

## Notes

- `--resume` skips completed runs and resets only interrupted partial runs.
- The nested-tensor PyTorch warning in `deep_avsr` is expected and does not invalidate results.
