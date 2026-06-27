"""
Track 4 #2 — Block bootstrap confidence intervals for V4.3 CV metrics.

Wraps every claimed metric (RMSE, RMSE-vs-naive %, DM stat, PT stat, sign-DA,
EV/trade, annualised Sharpe*) with a 90% CI from a circular block bootstrap
on the stored (y_true, y_pred) arrays. Block size = h+1 to absorb the
overlap-induced autocorrelation in h-step-ahead forecast errors.

Why:
  V4.3 reports point estimates and a single significance p-value per cell.
  Bootstrap CIs turn each claim into an interval — same cost-to-compute, much
  more honest reporting and the basis for any later "is improvement X
  meaningful?" question.

API:
  bootstrap_all(preds, sign_results, B=1000, alpha=0.10, seed=0)
    → prints two tables, returns dict {(market, h, layer or 'sign'): {...CI fields...}}
"""
from __future__ import annotations

import numpy as np

from K2E_Model_V4_2 import dm_test, pt_test


# ─────────────────────────────────────────────────────────────────────────────
# Circular block bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n indices via the circular block bootstrap."""
    if block_size < 1:
        block_size = 1
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_size)
    idx = (starts[:, None] + offsets[None, :]) % n
    return idx.ravel()[:n]


def _bootstrap_stat(arrays: tuple[np.ndarray, ...],
                    stat_fn,
                    block_size: int,
                    B: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Resample aligned arrays with the circular block bootstrap, evaluate stat_fn each draw.

    arrays: tuple of 1D arrays of equal length n. Each resample uses the same
            block indices across all arrays, preserving cross-series alignment.
    stat_fn: callable(*arrays) → float (np.nan tolerated).
    Returns array of length B with finite values only.
    """
    n = len(arrays[0])
    out = np.empty(B, dtype=float)
    for b in range(B):
        idx = _block_indices(n, block_size, rng)
        resampled = tuple(a[idx] for a in arrays)
        try:
            out[b] = float(stat_fn(*resampled))
        except Exception:
            out[b] = np.nan
    return out[np.isfinite(out)]


def _ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    if samples.size == 0:
        return (np.nan, np.nan)
    lo = float(np.percentile(samples, 100 * (alpha / 2)))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell bootstrap routines
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_rmse_cell(y_true: np.ndarray, y_pred: np.ndarray,
                        h: int, B: int, alpha: float,
                        rng: np.random.Generator) -> dict:
    """CIs for RMSE, naive RMSE, RMSE-vs-naive %, DM stat (model vs naive)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    bs = max(2, h + 1)

    err_model = y_true - y_pred
    err_naive = y_true                          # naive forecast = 0

    def _rmse(e):
        return float(np.sqrt(np.mean(e ** 2)))

    def _vs_naive(em, en):
        rm = np.sqrt(np.mean(em ** 2))
        rn = np.sqrt(np.mean(en ** 2))
        if rn <= 0:
            return np.nan
        return (rm - rn) / rn * 100.0

    def _dm(em, en):
        s, _ = dm_test(em, en, h=h)
        return s

    rmse_b   = _bootstrap_stat((err_model,),         lambda em:    _rmse(em),         bs, B, rng)
    naive_b  = _bootstrap_stat((err_naive,),         lambda en:    _rmse(en),         bs, B, rng)
    vs_b     = _bootstrap_stat((err_model, err_naive), _vs_naive,                     bs, B, rng)
    dm_b     = _bootstrap_stat((err_model, err_naive), _dm,                           bs, B, rng)

    rmse_lo, rmse_hi = _ci(rmse_b, alpha)
    vs_lo, vs_hi     = _ci(vs_b, alpha)
    dm_lo, dm_hi     = _ci(dm_b, alpha)

    return {
        'n': n, 'h': h, 'block_size': bs, 'B_eff_rmse': len(rmse_b), 'B_eff_dm': len(dm_b),
        'rmse_point':  _rmse(err_model),
        'rmse_ci':     (rmse_lo, rmse_hi),
        'vs_naive_point': _vs_naive(err_model, err_naive),
        'vs_naive_ci': (vs_lo, vs_hi),
        'dm_point':    _dm(err_model, err_naive),
        'dm_ci':       (dm_lo, dm_hi),
    }


def bootstrap_sign_cell(y_true: np.ndarray, sign_pred: np.ndarray,
                        h: int, B: int, alpha: float,
                        rng: np.random.Generator) -> dict:
    """CIs for DA, PT stat, mean EV (hard PnL), annualised Sharpe*."""
    y_true    = np.asarray(y_true, dtype=float)
    sign_pred = np.asarray(sign_pred, dtype=float)
    mask = np.sign(y_true) != 0
    y_m  = y_true[mask]
    sp_m = sign_pred[mask]
    n = len(y_m)
    bs = max(2, h + 1)

    pnl = sp_m * y_m

    def _da(y, sp):
        return float((np.sign(y) == sp).mean() * 100.0)

    def _pt(y, sp):
        s, _ = pt_test(y, sp)
        return s

    def _ev(p):
        return float(np.mean(p))

    def _sharpe(p):
        sd = float(np.std(p, ddof=0))
        if sd <= 0:
            return np.nan
        return float(np.mean(p) / sd * np.sqrt(252.0 / max(h, 1)))

    da_b     = _bootstrap_stat((y_m, sp_m), _da,     bs, B, rng)
    pt_b     = _bootstrap_stat((y_m, sp_m), _pt,     bs, B, rng)
    ev_b     = _bootstrap_stat((pnl,),     lambda p: _ev(p),     bs, B, rng)
    sharpe_b = _bootstrap_stat((pnl,),     lambda p: _sharpe(p), bs, B, rng)

    return {
        'n': n, 'h': h, 'block_size': bs,
        'da_point':     _da(y_m, sp_m),     'da_ci':     _ci(da_b, alpha),
        'pt_point':     _pt(y_m, sp_m),     'pt_ci':     _ci(pt_b, alpha),
        'ev_point':     _ev(pnl),           'ev_ci':     _ci(ev_b, alpha),
        'sharpe_point': _sharpe(pnl),       'sharpe_ci': _ci(sharpe_b, alpha),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level driver + printers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ci(ci, fmt):
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return '[   n/a   ,    n/a   ]'
    return f'[{lo:{fmt}}, {hi:{fmt}}]'


def _print_rmse_table(rmse_results, alpha):
    pct = int(round((1 - alpha) * 100))
    print(f"\n\n{'=' * 110}")
    print(f"TRACK 4 — BOOTSTRAP {pct}% CIs ON CV RMSE & DM (block bootstrap, block size = h+1)")
    print('=' * 110)
    hdr = (f"{'Market':6}{'h':>3} {'Layer':>5}   "
           f"{'RMSE':>9}  {'RMSE CI':>22}   "
           f"{'vs_naive%':>10}  {'vs_naive CI':>22}   "
           f"{'DM':>7}  {'DM CI':>18}")
    print(hdr)
    print('-' * len(hdr))
    for (market, h, layer), r in rmse_results.items():
        print(f"{market:6}{h:>3} {layer:>5}   "
              f"{r['rmse_point']:9.5f}  {_fmt_ci(r['rmse_ci'], '9.5f'):>22}   "
              f"{r['vs_naive_point']:+9.1f}%  {_fmt_ci(r['vs_naive_ci'], '+8.1f'):>22}   "
              f"{r['dm_point']:+7.3f}  {_fmt_ci(r['dm_ci'], '+7.3f'):>18}")


def _print_sign_table(sign_ci_results, alpha):
    pct = int(round((1 - alpha) * 100))
    print(f"\n\n{'=' * 110}")
    print(f"TRACK 4 — BOOTSTRAP {pct}% CIs ON SIGN CLASSIFIER (DA, PT, EV, Sharpe*)")
    print('=' * 110)
    hdr = (f"{'Market':6}{'h':>3}   "
           f"{'DA%':>6}  {'DA CI':>16}   "
           f"{'PT':>7}  {'PT CI':>18}   "
           f"{'EV':>9}  {'EV CI':>22}   "
           f"{'Sh*':>7}  {'Sh* CI':>18}")
    print(hdr)
    print('-' * len(hdr))
    for (market, h), r in sign_ci_results.items():
        print(f"{market:6}{h:>3}   "
              f"{r['da_point']:5.1f}  {_fmt_ci(r['da_ci'], '5.1f'):>16}   "
              f"{r['pt_point']:+7.3f}  {_fmt_ci(r['pt_ci'], '+7.3f'):>18}   "
              f"{r['ev_point']:+9.5f}  {_fmt_ci(r['ev_ci'], '+9.5f'):>22}   "
              f"{r['sharpe_point']:+7.2f}  {_fmt_ci(r['sharpe_ci'], '+7.2f'):>18}")
    print(f"\n  CI excludes 0 → metric significantly different from 0 at the {pct}% level.")
    print( "  Sh* is crude annualised; ignores costs / overlap penalty for h>1.")


def bootstrap_all(preds: dict, sign_results: dict,
                  B: int = 1000, alpha: float = 0.10, seed: int = 0) -> dict:
    """Compute CIs for every cell in `preds` (L1 + L3) and `sign_results`."""
    rng = np.random.default_rng(seed)
    rmse_out = {}
    for (market, h, layer), (y_true, y_pred) in preds.items():
        rmse_out[(market, h, layer)] = bootstrap_rmse_cell(
            y_true, y_pred, h=h, B=B, alpha=alpha, rng=rng)

    sign_out = {}
    for (market, h), entry in sign_results.items():
        cv = entry['cv']
        sign_out[(market, h)] = bootstrap_sign_cell(
            cv['y_true'], cv['sign_pred'], h=h, B=B, alpha=alpha, rng=rng)

    _print_rmse_table(rmse_out, alpha)
    _print_sign_table(sign_out, alpha)
    return {'rmse': rmse_out, 'sign': sign_out, 'B': B, 'alpha': alpha}
