"""
Track 4 #3 — Per-regime L3 diagnostics.

For each (market, h) where the L3 final fit succeeded, slice the walk-forward
CV residuals by inferred regime and report per-state RMSE / DA / occupancy.

Method:
  1. Pull smoothed state probabilities from the full-history L3 fit
     (fitted[(market,h,'L3')]['smooth_probs']). State columns are already
     ordered low-vol → high-vol by _order_states.
  2. Hard-label each observation by argmax(smooth_probs, axis=1).
  3. Recompute the CV test-window indices via get_cv_folds(n, 3) — the same
     splits cv_layer3_single uses — so we know which rows in smooth_probs the
     CV preds correspond to.
  4. For each state k, compute RMSE / DA / count over the CV test rows whose
     regime label is k.

Caveat (logged in the printout):
  Smoothed probabilities are full-history hindsight assignments. They are NOT
  the per-fold inferred state at forecast time. This is a *diagnostic* slice
  — "in periods the trained model identifies as regime X, how good were the
  CV forecasts that fold-time models produced?" — not a deployable per-state
  edge claim.
"""
from __future__ import annotations

import numpy as np

from K2E_Model_V4_2 import make_X_y, get_cv_folds


def _state_label_per_row(smooth_probs: np.ndarray) -> np.ndarray:
    return np.argmax(smooth_probs, axis=1)


def _cv_test_indices(n: int, n_folds: int = 3) -> np.ndarray:
    folds = get_cv_folds(n, n_folds)
    parts = [np.arange(ts, te) for ts, te in folds]
    return np.concatenate(parts) if parts else np.array([], dtype=int)


def _per_state_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       state_labels: np.ndarray, n_states: int) -> list[dict]:
    out = []
    for k in range(n_states):
        mask = state_labels == k
        nk = int(mask.sum())
        if nk == 0:
            out.append({'state': k, 'n': 0, 'rmse': np.nan, 'da': np.nan,
                        'mean_y2': np.nan, 'naive_rmse': np.nan, 'vs_naive_pct': np.nan})
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        naive_rmse = float(np.sqrt(np.mean(yt ** 2)))
        da = float((np.sign(yt) == np.sign(yp)).mean() * 100.0)
        vs_naive = ((rmse - naive_rmse) / naive_rmse * 100.0) if naive_rmse > 0 else np.nan
        out.append({'state': k, 'n': nk, 'rmse': rmse, 'da': da,
                    'mean_y2': float(np.mean(yt ** 2)),
                    'naive_rmse': naive_rmse, 'vs_naive_pct': vs_naive})
    return out


def regime_diag_all(d, fitted: dict, preds: dict) -> dict:
    """For every L3 cell with a successful final fit, print per-regime CV slice."""
    print(f"\n\n{'=' * 110}")
    print("TRACK 4 — PER-REGIME L3 DIAGNOSTICS (full-history smoothed states; CV residuals sliced)")
    print('=' * 110)
    print("  Note: state labels come from the full-data fit (hindsight). Read as a *diagnostic* —")
    print("        which regimes carry the L3 win and which drag it. Not a deployable per-state edge.")

    out = {}
    for (market, h, layer), fit in fitted.items():
        if layer != 'L3':
            continue
        if (market, h, 'L3') not in preds:
            continue

        smooth = fit['smooth_probs']
        n_states = fit['n_states']
        sv = fit['switching_variance']
        state_var = fit['state_variance']
        trans = fit['trans_mat']

        X, _ = make_X_y(d, market, h)
        n = len(X)
        if smooth.shape[0] != n:
            print(f"\n  [{market} h={h}] smooth_probs length {smooth.shape[0]} ≠ X length {n} — skipping")
            continue

        labels = _state_label_per_row(smooth)
        test_idx = _cv_test_indices(n, n_folds=3)

        y_true_full, y_pred_full = preds[(market, h, 'L3')]
        y_true_full = np.asarray(y_true_full, dtype=float)
        y_pred_full = np.asarray(y_pred_full, dtype=float)

        if len(y_true_full) != len(test_idx):
            print(f"\n  [{market} h={h}] CV pred length {len(y_true_full)} ≠ test_idx length "
                  f"{len(test_idx)} — index alignment broken; skipping")
            continue

        test_labels = labels[test_idx]
        full_overall_pct = (np.bincount(labels, minlength=n_states) / n * 100.0)
        cv_overall_pct = (np.bincount(test_labels, minlength=n_states) / len(test_labels) * 100.0)

        rows = _per_state_metrics(y_true_full, y_pred_full, test_labels, n_states)
        sv_str = 'SV' if sv else 'FV'
        sd_strs = []
        for k in range(n_states):
            v = state_var.get(k, np.nan)
            sd = np.sqrt(v) if (np.isfinite(v) and v > 0) else np.nan
            sd_strs.append(f"σ_{k}={sd:.4f}" if np.isfinite(sd) else f"σ_{k}=n/a")
        diag_p = [trans[k, k] for k in range(n_states)]
        diag_strs = [f"p({k}→{k})={p:.2f}" for k, p in enumerate(diag_p)]

        print(f"\n  {market} h={h}  (k={fit['k_feats']} s={n_states} {sv_str}; "
              f"{', '.join(sd_strs)}; {', '.join(diag_strs)})")
        print(f"    Full-sample regime occupancy %: " +
              ", ".join(f"S{k}={full_overall_pct[k]:5.1f}" for k in range(n_states)))
        print(f"    CV-window regime occupancy %  : " +
              ", ".join(f"S{k}={cv_overall_pct[k]:5.1f}" for k in range(n_states)))
        print(f"    {'state':>5}  {'n':>5}  {'RMSE':>9}  {'naive RMSE':>11}  "
              f"{'vs_naive':>10}  {'DA':>6}")
        for r in rows:
            if r['n'] == 0:
                print(f"    {r['state']:>5}  {r['n']:>5}  {'n/a':>9}  {'n/a':>11}  {'n/a':>10}  {'n/a':>6}")
                continue
            print(f"    {r['state']:>5}  {r['n']:>5}  {r['rmse']:9.5f}  "
                  f"{r['naive_rmse']:11.5f}  {r['vs_naive_pct']:+9.1f}%  {r['da']:5.1f}%")

        out[(market, h)] = {
            'rows': rows, 'state_var': state_var, 'trans_diag': diag_p,
            'full_pct': full_overall_pct.tolist(), 'cv_pct': cv_overall_pct.tolist(),
        }

    if not out:
        print("\n  No L3 cells with a successful final fit — nothing to slice.")
    return out
