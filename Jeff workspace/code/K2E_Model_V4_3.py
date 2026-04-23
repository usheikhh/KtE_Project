"""
K2 Energy / Fordham University Collaboration
Phase 1 v4.3: V4.2 + Track 1 best-of-cell policy + Track 3 directional classifier.

Changes vs V4.2
---------------
Track 1 — Failure-cell policy fixes (data + discipline, not model class)
  * Best-of-cell policy picks {naive, L1, L3} per cell.
      - L3 wins only if it beats both L1 and naive AND is not DM-significantly
        worse than naive (p<0.10, stat>0). Fixes POWER h=5 (which was +10.1%
        worse than naive with DM p=0.001).
      - L1 wins only if it beats naive by ≥1% or DM p<0.10 with negative stat.
        Routes EUA h=3 (L1 ties naive at −0.1%) to the naive fallback instead
        of running a spurious L3.
  * L3 CV predictions preserved even when the final full-history fit fails.
    Fixes POWER h=1 where the sweep found a valid config (RMSE 0.03899) but
    the final fit crashed and the cell dropped off the results table.

Track 3 — Directional sign classifier (second deliverable, parallel to RMSE)
  * Binary logistic regression (L2) on sign(y); same feature blocks as L1.
  * Walk-forward CV with C tuned by log-loss.
  * Metrics: DA, PT p, EV/trade (hard & soft), crude Sharpe*.
  * Targeted at POWER h=3 (PT p=0.009, RMSE flat) and EUA h=5 L1
    (PT p=0.000, RMSE only −0.7%) — cells where direction matters more
    than magnitude.

Layers inherited from V4.2 (unchanged):
  Layer 1 — Ridge-ARX (α grid 1e-4..1e4)
  Layer 3 — Markov-switching ARX with HP sweep + degeneracy safeguards
  Significance tests — Diebold-Mariano (NW+HLN) & Pesaran-Timmermann
"""

import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

import K2E_Model_V4_2 as v42
from K2E_Model_V4_2 import (
    load_clean, engineer_features, make_X_y,
    get_cv_folds, fit_layer1, sweep_layer3, fit_layer3,
    dm_test, pt_test,
)


def _safe_fit_layer3(d, market, h, l1, best):
    """fit_layer3 with extraction-side exceptions converted to None.

    V4.2 fit_layer3 only returns None when the EM fit itself fails; any
    exception raised while unpacking params / state ordering / transition
    matrix propagates and crashes the whole pipeline. This wrapper makes the
    extraction path symmetric with the EM path so the CV-only fallback can
    always kick in.
    """
    try:
        return fit_layer3(d, market, h, l1, best)
    except Exception as e:
        print(f"    L3: final-fit extraction raised ({type(e).__name__}: {e}) — CV metrics retained")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# TRACK 1 — NAIVE BASELINE + BEST-OF-CELL POLICY
# ═══════════════════════════════════════════════════════════════════════════

def _naive_cv(y_true):
    y_true = np.asarray(y_true, dtype=float)
    return {
        'y_pred':   np.zeros_like(y_true),
        'y_true':   y_true,
        'cv_rmse':  float(np.sqrt(np.mean(y_true ** 2))),
        'cv_da':    50.0,
    }


def policy_pick(cv_L1, cv_L3, h, min_improve=0.01, dm_alpha=0.10):
    """Pick winner from {naive, L1, L3} by CV RMSE with DM-rigor veto.

    Rules
      1. L3 wins iff:
           L3_rmse < L1_rmse AND L3_rmse < naive_rmse AND
           NOT (DM L3 vs naive says L3 significantly WORSE, p<dm_alpha stat>0)
      2. Else L1 wins iff:
           L1 beats naive by ≥ min_improve (relative)  OR
           DM L1 vs naive significant with negative stat
      3. Else naive
    """
    naive_L1 = _naive_cv(cv_L1['y_true'])
    naive_rmse_L1 = naive_L1['cv_rmse']

    e_L1 = cv_L1['y_true'] - cv_L1['y_pred']
    e_naive_L1 = cv_L1['y_true']
    dm_L1, dm_p_L1 = dm_test(e_L1, e_naive_L1, h=h)
    L1_rel = ((naive_rmse_L1 - cv_L1['cv_rmse']) / naive_rmse_L1) if naive_rmse_L1 > 0 else 0.0
    L1_beats = (
        L1_rel >= min_improve
        or (np.isfinite(dm_L1) and dm_L1 < 0 and dm_p_L1 < dm_alpha)
    )
    vs_naive_L1 = ((cv_L1['cv_rmse'] - naive_rmse_L1) / naive_rmse_L1) * 100.0

    if cv_L3 is not None:
        y_L3 = np.asarray(cv_L3['y_true'], dtype=float)
        p_L3 = np.asarray(cv_L3['y_pred'], dtype=float)
        naive_rmse_L3 = float(np.sqrt(np.mean(y_L3 ** 2)))
        e_L3 = y_L3 - p_L3
        e_naive_L3 = y_L3
        dm_L3, dm_p_L3 = dm_test(e_L3, e_naive_L3, h=h)
        L3_sig_worse = np.isfinite(dm_L3) and dm_L3 > 0 and dm_p_L3 < dm_alpha
        L3_beats_L1 = cv_L3['cv_rmse'] < cv_L1['cv_rmse']
        L3_beats_naive = cv_L3['cv_rmse'] < naive_rmse_L3 and not L3_sig_worse
        vs_naive_L3 = ((cv_L3['cv_rmse'] - naive_rmse_L3) / naive_rmse_L3) * 100.0

        if L3_beats_L1 and L3_beats_naive:
            return {
                'policy':   'L3',
                'cv_rmse':  cv_L3['cv_rmse'],
                'cv_da':    cv_L3['cv_da'],
                'vs_naive': vs_naive_L3,
                'reason':   'L3 best (beats L1 and naive)',
            }
        if L3_sig_worse:
            veto = 'L3 vetoed: DM p<{:.2f} WORSE than naive'.format(dm_alpha)
        elif not L3_beats_L1:
            veto = 'L3 ≥ L1 RMSE'
        elif not L3_beats_naive:
            veto = 'L3 ≥ naive RMSE'
        else:
            veto = ''
    else:
        veto = 'no L3 config'

    if L1_beats:
        return {
            'policy':   'L1',
            'cv_rmse':  cv_L1['cv_rmse'],
            'cv_da':    cv_L1['cv_da'],
            'vs_naive': vs_naive_L1,
            'reason':   f'L1 beats naive ({L1_rel*100:+.1f}% / DM p={dm_p_L1:.3f}); {veto}',
        }
    return {
        'policy':   'naive',
        'cv_rmse':  naive_rmse_L1,
        'cv_da':    50.0,
        'vs_naive': 0.0,
        'reason':   f'fallback to naive (L1 {L1_rel*100:+.1f}% vs naive); {veto}',
    }


# ═══════════════════════════════════════════════════════════════════════════
# TRACK 3 — DIRECTIONAL SIGN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

LOGIT_CS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _prepare_sign_train(X_raw, y_raw):
    """Drop ties (y==0) and map sign → binary {0,1}. Returns (X, y_bin)."""
    mask = np.asarray(y_raw) != 0
    return X_raw[mask], (np.sign(y_raw[mask]) > 0).astype(int)


def tune_logit_C(X, y, n_folds=5, C_grid=LOGIT_CS):
    n = len(X)
    folds = get_cv_folds(n, n_folds)
    C_scores = {c: [] for c in C_grid}
    for ts, te in folds:
        Xtr_raw, ytr_raw = X.iloc[:ts], y.iloc[:ts]
        Xtr, ytr_bin = _prepare_sign_train(Xtr_raw, ytr_raw)
        Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
        if len(np.unique(ytr_bin)) < 2 or len(Xtr) < 20:
            continue
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        yte_bin = (np.sign(yte) > 0).astype(int)
        for c in C_grid:
            mdl = LogisticRegression(penalty='l2', C=c, max_iter=500, solver='lbfgs')
            mdl.fit(Xtr_s, ytr_bin)
            prob = mdl.predict_proba(Xte_s)[:, 1]
            eps = 1e-9
            ll = -(yte_bin * np.log(prob + eps) + (1 - yte_bin) * np.log(1 - prob + eps)).mean()
            C_scores[c].append(ll)
    means = {c: float(np.mean(s)) if s else np.inf for c, s in C_scores.items()}
    return min(means, key=means.get) if np.isfinite(min(means.values())) else 1.0


def cv_sign_classifier(X, y, n_folds=5):
    C_best = tune_logit_C(X, y, n_folds=n_folds)
    n = len(X)
    folds = get_cv_folds(n, n_folds)
    all_prob, all_y = [], []
    for ts, te in folds:
        Xtr_raw, ytr_raw = X.iloc[:ts], y.iloc[:ts]
        Xtr, ytr_bin = _prepare_sign_train(Xtr_raw, ytr_raw)
        Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
        if len(np.unique(ytr_bin)) < 2 or len(Xtr) < 20:
            prob = np.full(len(Xte), 0.5)
        else:
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            mdl = LogisticRegression(penalty='l2', C=C_best,
                                      max_iter=500, solver='lbfgs')
            mdl.fit(Xtr_s, ytr_bin)
            prob = mdl.predict_proba(Xte_s)[:, 1]
        all_prob.extend(prob)
        all_y.extend(yte.values)
    prob = np.asarray(all_prob, dtype=float)
    y_true = np.asarray(all_y, dtype=float)
    sign_pred = np.where(prob > 0.5, 1.0, -1.0)
    signal = 2.0 * prob - 1.0
    return {'C': C_best, 'prob': prob, 'signal': signal,
            'sign_pred': sign_pred, 'y_true': y_true}


def evaluate_sign(cv, h):
    y = cv['y_true']
    sp = cv['sign_pred']
    sig = cv['signal']
    mask = np.sign(y) != 0
    if mask.sum() < 10:
        return {'da': np.nan, 'pt_stat': np.nan, 'pt_p': np.nan,
                'ev': np.nan, 'ev_sig': np.nan, 'sharpe': np.nan,
                'n': int(mask.sum())}
    da = (np.sign(y[mask]) == sp[mask]).mean() * 100.0
    pt_stat, pt_p = pt_test(y[mask], sp[mask])
    pnl_hard = sp * y
    pnl_soft = sig * y
    ev = float(pnl_hard.mean())
    ev_sig = float(pnl_soft.mean())
    sd = pnl_hard.std(ddof=0)
    sharpe = (ev / (sd + 1e-12)) * np.sqrt(252.0 / max(h, 1))
    return {'da': da, 'pt_stat': pt_stat, 'pt_p': pt_p,
            'ev': ev, 'ev_sig': ev_sig, 'sharpe': float(sharpe),
            'n': int(mask.sum())}


def fit_sign_all(d):
    print("\n\n" + "=" * 88)
    print("TRACK 3 — DIRECTIONAL SIGN CLASSIFIER (logistic regression, L2)")
    print("=" * 88)
    print(f"{'Market':6}{'h':>3}   {'DA':>7}  {'PT_stat':>8}{'PT_p':>8}"
          f"   {'EV_hard':>10}{'EV_soft':>10}   {'Sharpe*':>8}   {'C':>6}  {'n':>5}")
    print("-" * 88)
    sign_results = {}
    for market in ['TTF', 'POWER', 'EUA']:
        for h in [1, 3, 5]:
            X, y = make_X_y(d, market, h)
            cv = cv_sign_classifier(X, y)
            ev = evaluate_sign(cv, h=h)
            sign_results[(market, h)] = {'cv': cv, 'metrics': ev}
            pt_s = f"{ev['pt_stat']:+.3f}" if np.isfinite(ev['pt_stat']) else '   n/a'
            pt_p = f"{ev['pt_p']:.3f}"    if np.isfinite(ev['pt_p'])    else ' n/a'
            print(f"{market:6}{h:>3}   {ev['da']:6.1f}%  {pt_s}{pt_p:>8}"
                  f"   {ev['ev']:+10.5f}{ev['ev_sig']:+10.5f}"
                  f"   {ev['sharpe']:+7.2f}   {cv['C']:>6.2f}  {ev['n']:>5}")
    print("\n  Legend: EV_hard = sign(pred)·y ; EV_soft = (2·prob-1)·y")
    print("          Sharpe* = crude daily-scaled (ignores costs, overlap for h>1, rebalancing)")
    return sign_results


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED V4.3 PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def _print_policy(rows):
    print(f"\n\n{'='*96}")
    print("TRACK 1 — BEST-OF-CELL POLICY (pick winner from {naive, L1, L3})")
    print('=' * 96)
    print(f"{'Market':6}{'h':>3}   {'Policy':>7}   {'CV_RMSE':>9}"
          f"   {'vs_naive':>9}   {'CV_DA':>7}   Reason")
    print('-' * 96)
    for r in rows:
        print(f"{r['market']:6}{r['h']:>3}   {r['policy']:>7}   "
              f"{r['cv_rmse']:9.5f}   {r['vs_naive']:+8.1f}%   "
              f"{r['cv_da']:6.1f}%   {r['reason']}")


def train_all_v43(train_path, sweep_l3=True):
    print("Loading and engineering features (V4.3)...")
    df_train = load_clean(train_path)
    d = engineer_features(df_train)

    fitted = {}
    results = []
    policy_rows = []
    sweep_log = {}
    preds = {}

    for market in ['TTF', 'POWER', 'EUA']:
        for h in [1, 3, 5]:
            print(f"\n{'─'*60}")
            print(f"  {market}  h={h}")
            print(f"{'─'*60}")

            # Layer 1
            print(f"    L1: Ridge-ARX ...", end=' ', flush=True)
            l1 = fit_layer1(d, market, h)
            fitted[(market, h, 'L1')] = l1
            cv1 = l1['cv']
            preds[(market, h, 'L1')] = (cv1['y_true'], cv1['y_pred'])
            vs1 = (cv1['cv_rmse'] - cv1['naive_rmse']) / cv1['naive_rmse'] * 100
            results.append({
                'market': market, 'h': h, 'layer': 'L1',
                'cv_rmse': cv1['cv_rmse'], 'cv_da': cv1['cv_da'],
                'naive_rmse': cv1['naive_rmse'], 'vs_naive': vs1,
                'note': f"α={l1['alpha']:.4f}",
            })
            print(f"RMSE={cv1['cv_rmse']:.5f}  DA={cv1['cv_da']:.1f}%  "
                  f"(α={l1['alpha']})")

            # Layer 3 — sweep always records CV preds even if final fit fails
            best_l3_cv = None
            if sweep_l3:
                print(f"    L3: Markov-switching HP sweep ...")
                best, all_res = sweep_layer3(d, market, h, l1, verbose=True)
                sweep_log[(market, h)] = all_res
                if best is not None:
                    best_l3_cv = best
                    preds[(market, h, 'L3')] = (best['y_true'], best['y_pred'])
                    l3 = _safe_fit_layer3(d, market, h, l1, best)
                    cv_flag = ''
                    if l3 is not None:
                        fitted[(market, h, 'L3')] = l3
                    else:
                        cv_flag = ' [CV-only]'
                        print(f"    L3: final fit failed — CV metrics retained")
                    vs3 = (best['cv_rmse'] - cv1['naive_rmse']) / cv1['naive_rmse'] * 100
                    sv_str = 'SV' if best['switching_variance'] else 'FV'
                    results.append({
                        'market': market, 'h': h, 'layer': 'L3',
                        'cv_rmse': best['cv_rmse'], 'cv_da': best['cv_da'],
                        'naive_rmse': cv1['naive_rmse'], 'vs_naive': vs3,
                        'note': f"k={best['k_feats']} s={best['n_states']} {sv_str}{cv_flag}",
                    })
                    print(f"    L3 BEST: RMSE={best['cv_rmse']:.5f}  "
                          f"DA={best['cv_da']:.1f}%  "
                          f"(k={best['k_feats']} s={best['n_states']} {sv_str}){cv_flag}")
                else:
                    print(f"    L3: no config converged")

            # Track 1 — best-of-cell policy
            pol = policy_pick(cv1, best_l3_cv, h=h)
            policy_rows.append({
                'market':   market, 'h': h,
                'policy':   pol['policy'],
                'cv_rmse':  pol['cv_rmse'],
                'cv_da':    pol['cv_da'],
                'vs_naive': pol['vs_naive'],
                'reason':   pol['reason'],
            })
            print(f"    POLICY: {pol['policy']:>5}  RMSE={pol['cv_rmse']:.5f}  "
                  f"vs_naive={pol['vs_naive']:+.1f}%")

    # Reports — reuse V4.2 printers + add policy block
    v42._print_comparison(results, fitted)
    v42._print_sweep_summary(sweep_log)
    v42._print_significance(preds)
    _print_policy(policy_rows)

    # Track 3 — sign classifier (parallel deliverable)
    sign_results = fit_sign_all(d)

    return fitted, d, results, sweep_log, preds, policy_rows, sign_results


if __name__ == '__main__':
    train_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--train' and i + 1 < len(sys.argv):
            train_path = sys.argv[i + 1]
    if train_path is None:
        train_path = ('/Users/jefflu/Library/CloudStorage/OneDrive-Personal/'
                      'Documents/MSQF Fordham/ML Internship/k2_clean_v3.csv')
    out = train_all_v43(train_path, sweep_l3=True)
