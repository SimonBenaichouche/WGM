# Figure 3 — Reproduction

Evaluates three gradient estimators (**Surrogate**, **WGM**, **Sobolev**)
on five benchmark functions across three sample sizes (N = 500, 2 000, 10 000).

## Files

```
reproduce_fig3.py   — single script: runs the experiment and saves figures
README.md           — this file
```

Results (CSV) and figures are written to `results/` and `figures/`
(created automatically).

## Dependencies

```
torch numpy pandas matplotlib
```

The script also requires the bench library located at `../bench_collect/1k/`
relative to this folder. Override with `--lib`:

```bash
python reproduce_fig3.py --lib /path/to/bench_collect/1k
```

## Protocol

Four RBF basis configurations are defined explicitly in `EVAL_CONFIGS` at
the top of the script.  For **each** configuration:

1. **Hyperparameter selection** (Phase 1) — at N = 2 000, a sequential
   grid search selects λ_surr, λ_wgm, and γ independently for each
   benchmark function by minimising the respective validation MSE.
   Each candidate is evaluated over 5 random seeds and the scores are
   averaged for stability.

2. **Statistical evaluation** (Phase 2) — the selected hyperparameters
   are frozen.  20 independent runs are performed at each (function, N)
   pair.  Error bars in the figures represent the standard error of the
   mean (SEM = std / √20).

Only the number of training samples varies across N values; the basis,
regularisation, and mixing coefficient are identical for N = 500 and
N = 10 000.  This is required for the consistency argument: any
improvement with N reflects more data, not better tuning.

## Output

One 2 × 3 figure per configuration:

- **Top row** — gradient MSE normalised by the Surrogate MSE (lower is better).
  The dashed line at 1.0 marks parity with the Surrogate.
- **Bottom row** — cosine similarity between the estimated and true gradient
  (higher is better).

## Key parameters

| Parameter | Value | Role |
|---|---|---|
| `N_REF` | 2 000 | N used for hyperparameter selection |
| `N_STAT_RUNS` | 20 | independent runs per (function, N) |
| `N_SEL_SEEDS` | 5 | seeds averaged during selection |
| `LAMBDA_SURR` | [1e-5, 1e-4, 1e-3] | search grid for surrogate regularisation |
| `LAMBDA_WGM` | [1e-3, 5e-3, 1e-2] | search grid for WGM regularisation |
| `GAMMA_VALUES` | [0.05 … 50] | search grid for Sobolev mixing coefficient |
