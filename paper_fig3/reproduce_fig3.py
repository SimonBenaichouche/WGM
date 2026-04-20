"""
Reproduction script for Figure 3.

Evaluates three gradient estimators — Surrogate, WGM, Sobolev — on five
benchmark functions across three sample sizes (N = 500, 2 000, 10 000).

Protocol
--------
Four RBF basis configurations are defined explicitly below (EVAL_CONFIGS).
For each configuration:
  1. Regularisation hyperparameters (λ_surr, λ_wgm, γ) are selected once
     at N_REF = 2 000 by minimising the respective validation MSE, averaged
     over N_SEL_SEEDS random seeds for stability.
  2. The selected hyperparameters are frozen and the same configuration is
     used unchanged for N = 500 and N = 10 000.  Only the number of
     training samples varies across columns.
  3. N_STAT_RUNS = 20 independent runs are performed at each (function, N)
     to estimate means and standard errors.

One figure (MSE top, cosine similarity bottom) is saved per configuration.

Usage
-----
    python reproduce_fig3.py [--lib PATH_TO_BENCH_LIB]

The bench library path defaults to the relative location below; override
with --lib if your layout differs.
"""

import argparse
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Library path
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--lib', default='../bench_collect/1k', help='Path to bench library')
args, _ = parser.parse_known_args()

sys.path.insert(0, os.path.abspath(args.lib))
from plot_and_stats_tmlr import TARGETS, run_evaluation   # noqa: E402

# ---------------------------------------------------------------------------
# Configurations — fixed upfront, not adapted per function
# ---------------------------------------------------------------------------
EVAL_CONFIGS = {
    "Simple_10x10_s2.0":    [(10, 2.0)],
    "Simple_12x12_s1.5":    [(12, 1.5)],
    "Multi_10c6f_s2.0-0.8": [(10, 2.0), (6, 0.8)],
    "Simple_10x10_s1.0":    [(10, 1.0)],
}

PAPER_FUNCS    = ['StyblinskiTang', 'SixHumpCamel', 'Himmelblau', 'Ackley', 'Rosenbrock']
FUNC_LABELS    = ['Styblinski\nTang', 'Six Hump\nCamel', 'Himmelblau', 'Ackley', 'Rosenbrock']
N_SAMPLES_LIST = [500, 2000, 10000]
N_REF          = 2000      # N used for hyperparameter selection
N_STAT_RUNS    = 20        # independent runs for statistics
N_SEL_SEEDS    = 5         # seeds averaged during selection for stability
SAMPLING_SCALE = 1.0

LAMBDA_SURR  = [1e-5, 1e-4, 1e-3]
LAMBDA_WGM   = [1e-3, 5e-3, 1e-2]
GAMMA_VALUES = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 50.0]

COLOR_SURR = '#1f77b4'
COLOR_WGM  = '#ff7f0e'
COLOR_MIX  = '#2ca02c'

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def make_cfg(func_key, n_samples, base_name, scale_configs, lam_surr, lam_wgm, gamma):
    return {
        'function_key':    func_key,
        'base_config_key': base_name,
        'scale_configs':   scale_configs,
        'n_samples':       n_samples,
        'noise_level':     0.01,
        'sampling_scale':  SAMPLING_SCALE,
        'lambda_surr':     lam_surr,
        'lambda_wgm':      lam_wgm,
        'lambda_mixed':    lam_surr,
        'gamma_mixed':     gamma,
    }


def averaged_eval(cfg):
    """Evaluate cfg over N_SEL_SEEDS seeds and return the mean of each metric."""
    results = []
    for s in range(N_SEL_SEEDS):
        try:
            torch.manual_seed(s * 777)
            results.append(run_evaluation(cfg, return_fields=False))
        except Exception:
            results.append((1e8, 1e8, 1e8, 0., 0., 0.))
    return tuple(float(np.mean([r[i] for r in results])) for i in range(6))


def select_hyperparams(func_key, base_name, scale_configs):
    """
    Sequential selection at N_REF:
      A — λ_surr : minimise Surrogate MSE
      B — λ_wgm  : minimise WGM MSE        (λ_surr fixed)
      C — γ      : minimise Sobolev MSE    (λ_surr, λ_wgm fixed)
    """
    res = [averaged_eval(make_cfg(func_key, N_REF, base_name, scale_configs, ls, 1e-3, 1.0))
           for ls in LAMBDA_SURR]
    best_ls = LAMBDA_SURR[int(np.argmin([r[0] for r in res]))]

    res = [averaged_eval(make_cfg(func_key, N_REF, base_name, scale_configs, best_ls, lw, 1.0))
           for lw in LAMBDA_WGM]
    best_lw = LAMBDA_WGM[int(np.argmin([r[1] for r in res]))]

    res = [averaged_eval(make_cfg(func_key, N_REF, base_name, scale_configs, best_ls, best_lw, g))
           for g in GAMMA_VALUES]
    best_g = GAMMA_VALUES[int(np.argmin([r[2] for r in res]))]

    return best_ls, best_lw, best_g


def run_stat(func_key, n_samples, base_name, scale_configs, lam_surr, lam_wgm, gamma):
    """N_STAT_RUNS evaluations with a fixed configuration."""
    cfg = make_cfg(func_key, n_samples, base_name, scale_configs, lam_surr, lam_wgm, gamma)
    s_ms, s_cs, w_mw, w_cw, m_mm, m_cm = [], [], [], [], [], []
    for j in range(N_STAT_RUNS):
        torch.manual_seed(j * 1000)
        ms, mw, mm, cs, cw, cm = run_evaluation(cfg, return_fields=False)
        s_ms.append(ms); s_cs.append(cs)
        w_mw.append(mw); w_cw.append(cw)
        m_mm.append(mm); m_cm.append(cm)
    return {
        'Function': func_key, 'N': n_samples,
        'Surr_MSE_mean':    np.mean(s_ms), 'Surr_MSE_std':    np.std(s_ms),
        'Surr_CosSim_mean': np.mean(s_cs), 'Surr_CosSim_std': np.std(s_cs),
        'WGM_MSE_mean':     np.mean(w_mw), 'WGM_MSE_std':     np.std(w_mw),
        'WGM_CosSim_mean':  np.mean(w_cw), 'WGM_CosSim_std':  np.std(w_cw),
        'Mix_MSE_mean':     np.mean(m_mm), 'Mix_MSE_std':     np.std(m_mm),
        'Mix_CosSim_mean':  np.mean(m_cm), 'Mix_CosSim_std':  np.std(m_cm),
        'lambda_surr': lam_surr, 'lambda_wgm': lam_wgm, 'gamma': gamma,
    }


def make_figure(df, config_name, out_path):
    x, width, sqN = np.arange(len(PAPER_FUNCS)), 0.25, np.sqrt(N_STAT_RUNS)

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle(
        f"RBF config: {config_name}  —  "
        "MSE relative to surrogate (top) · cosine similarity with true gradient (bottom)",
        fontsize=13, fontweight='bold'
    )

    for ci, n_samples in enumerate(N_SAMPLES_LIST):
        df_n = df[df['N'] == n_samples]

        def col(f):
            return np.array([df_n[df_n['Function'] == fn][f].values[0] for fn in PAPER_FUNCS])

        sm = col('Surr_MSE_mean')

        ax = axes[0][ci]
        ax.bar(x-width, sm/sm,             width, label='Surrogate',
               yerr=col('Surr_MSE_std')/sm/sqN, capsize=4, color=COLOR_SURR, alpha=0.85)
        ax.bar(x,       col('WGM_MSE_mean')/sm, width, label='WGM',
               yerr=col('WGM_MSE_std')/sm/sqN,  capsize=4, color=COLOR_WGM,  alpha=0.85)
        ax.bar(x+width, col('Mix_MSE_mean')/sm, width, label='Sobolev',
               yerr=col('Mix_MSE_std')/sm/sqN,  capsize=4, color=COLOR_MIX,  alpha=0.85)
        ax.axhline(1.0, color='black', linestyle='--', lw=1.0, alpha=0.6)
        ax.set_title(f"Relative MSE — N={n_samples}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(FUNC_LABELS, rotation=0, ha='center', fontsize=14)
        ax.set_ylabel("Relative MSE (lower is better)", fontsize=11)
        ax.set_ylim(0, 1.8)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.4, color='gray')
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color('#cccccc')

        ax = axes[1][ci]
        ax.bar(x-width, col('Surr_CosSim_mean'), width, label='Surrogate',
               yerr=col('Surr_CosSim_std')/sqN, capsize=4, color=COLOR_SURR, alpha=0.85)
        ax.bar(x,       col('WGM_CosSim_mean'),  width, label='WGM',
               yerr=col('WGM_CosSim_std')/sqN,  capsize=4, color=COLOR_WGM,  alpha=0.85)
        ax.bar(x+width, col('Mix_CosSim_mean'),  width, label='Sobolev',
               yerr=col('Mix_CosSim_std')/sqN,  capsize=4, color=COLOR_MIX,  alpha=0.85)
        ax.set_title(f"Gradient Cosine Similarity — N={n_samples}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(FUNC_LABELS, rotation=0, ha='center', fontsize=14)
        ax.set_ylabel("Cosine Similarity (higher is better)", fontsize=11)
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.4, color='gray')
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color('#cccccc')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

for config_name, scale_configs in EVAL_CONFIGS.items():
    K = sum(n ** 2 for n, _ in scale_configs)
    print(f"\n{'='*60}", flush=True)
    print(f"  Config: {config_name}  (K={K} basis functions)", flush=True)
    print(f"{'='*60}", flush=True)

    # --- Phase 1: hyperparameter selection at N_REF ---
    print(f"  Selecting hyperparameters at N={N_REF} ...", flush=True)
    hparams = {}
    for func_key in PAPER_FUNCS:
        print(f"    {func_key} ", end='', flush=True)
        ls, lw, g = select_hyperparams(func_key, config_name, scale_configs)
        hparams[func_key] = (ls, lw, g)
        print(f"→ λ_surr={ls:.0e}  λ_wgm={lw:.0e}  γ={g}", flush=True)

    # --- Phase 2: statistical evaluation with frozen hyperparams ---
    print(f"\n  Statistical evaluation ({N_STAT_RUNS} runs × {len(PAPER_FUNCS)} functions × {len(N_SAMPLES_LIST)} N values) ...", flush=True)
    rows = []
    for n_samples in N_SAMPLES_LIST:
        print(f"    N={n_samples}", flush=True)
        for func_key in PAPER_FUNCS:
            ls, lw, g = hparams[func_key]
            row = run_stat(func_key, n_samples, config_name, scale_configs, ls, lw, g)
            rows.append(row)
            sm = row['Surr_MSE_mean']
            print(f"      {func_key:<18}  WGM/S={row['WGM_MSE_mean']/sm:.3f}"
                  f"  Sob/S={row['Mix_MSE_mean']/sm:.3f}"
                  f"  CosSim: Surr={row['Surr_CosSim_mean']:.3f}"
                  f"  WGM={row['WGM_CosSim_mean']:.3f}"
                  f"  Sob={row['Mix_CosSim_mean']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/{config_name}_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Results saved: {csv_path}", flush=True)

    make_figure(df, config_name, f"figures/{config_name}.png")

print("\nDone.", flush=True)
