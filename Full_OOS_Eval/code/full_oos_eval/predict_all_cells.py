import os
"""Fit L1 (Ridge-ARX) AND L3 (Markov-switching, HP-swept) on ALL 9 cells,
ignoring the deployed policy gate, and run them out of sample over the full
18-month window. Answers: was the naive fallback the right call OOS, or would
a policy-discarded model have added value?
"""
import sys, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
from K2E_Model_V4_2 import (
    load_clean, engineer_features, make_X_y, FEAT_FNS, TARGET_P,
    fit_layer1, sweep_layer3, fit_layer3, _smoothed_probs_array,
)

MARKETS = ['TTF', 'POWER', 'EUA']
HORIZONS = [1, 3, 5]
TRAIN = 'data/k2_clean_v4.csv'


def filtered_seed(l3):
    res = l3['result']
    filt = np.asarray(res.filtered_marginal_probabilities, dtype=float)
    if filt.ndim == 1:
        filt = filt.reshape(-1, 1)
    return filt[-1][l3['state_order']]


def l3_predict(l3, Xs):
    n = l3['n_states']
    betas = np.vstack([l3['state_betas'][k] for k in range(n)])
    Xc = np.column_stack([np.ones(len(Xs)), Xs])
    return (Xc @ betas.T) @ l3['seed'][:n]


print('Loading + engineering training features...')
df_train = load_clean(TRAIN)
d_train = engineer_features(df_train)

fits = {}
for m in MARKETS:
    for h in HORIZONS:
        t0 = time.time()
        l1 = fit_layer1(d_train, m, h)
        best, _ = sweep_layer3(d_train, m, h, l1, verbose=False)
        l3 = None
        if best is not None:
            try:
                l3 = fit_layer3(d_train, m, h, l1, best)
                if l3 is not None:
                    l3['seed'] = filtered_seed(l3)
            except Exception as e:
                l3 = None
        fits[(m, h)] = dict(l1=l1, l3=l3, l3cfg=best)
        cfg = (f"k={best['k_feats']} s={best['n_states']} "
               f"{'SV' if best['switching_variance'] else 'FV'}") if best else 'none'
        print(f'  {m:5} h={h}: L1 a={l1["alpha"]:.4g}  L3 {cfg}'
              f'{" [final-fit failed]" if (best and l3 is None) else ""}'
              f'  ({time.time()-t0:.0f}s)')

# ---- OOS feature construction ----
full = pd.read_csv('data/full_parsed.csv', parse_dates=['Date'])
train_end = pd.to_datetime(df_train['Date']).max()
oos_raw = full[full['Date'] > train_end].reset_index(drop=True)
combo = pd.concat([df_train.tail(80), oos_raw], ignore_index=True, sort=False)
combo = combo.sort_values('Date').reset_index(drop=True)
d = engineer_features(combo)
oos_mask = d['Date'].isin(pd.to_datetime(oos_raw['Date']).values).values
oos_dates = d.loc[oos_mask, 'Date'].values
print(f'\nOOS rows: {oos_mask.sum()}  ({pd.Timestamp(oos_dates.min()).date()} -> {pd.Timestamp(oos_dates.max()).date()})')

# actuals indexed by date for scoring
act = full.set_index('Date')[[TARGET_P[m] for m in MARKETS]].rename(
    columns={TARGET_P[m]: m for m in MARKETS}).sort_index()
dates = list(act.index); idx_of = {dt: i for i, dt in enumerate(dates)}

rows = []
for m in MARKETS:
    feats_all = pd.DataFrame(FEAT_FNS[m](d), index=d.index)
    close0 = d.loc[oos_mask, TARGET_P[m]].values
    for h in HORIZONS:
        f = fits[(m, h)]
        # --- generate the 3 forecasts (price level) at each origin ---
        lr_naive = np.zeros(oos_mask.sum())
        # L1
        X1 = feats_all.loc[oos_mask, f['l1']['feature_names']]
        Xs1 = np.nan_to_num(f['l1']['scaler'].transform(X1.values), nan=0.0)
        lr_l1 = f['l1']['model'].predict(Xs1)
        # L3
        if f['l3'] is not None:
            X3 = feats_all.loc[oos_mask, f['l3']['top_feats']]
            Xs3 = np.nan_to_num(f['l3']['scaler'].transform(X3.values), nan=0.0)
            lr_l3 = l3_predict(f['l3'], Xs3)
        else:
            lr_l3 = None

        preds = {'naive': close0 * np.exp(lr_naive),
                 'L1':    close0 * np.exp(lr_l1)}
        if lr_l3 is not None:
            preds['L3'] = close0 * np.exp(lr_l3)

        # --- score each over the full OOS (h-step ahead) ---
        oo = pd.to_datetime(oos_dates)
        for layer, price in preds.items():
            em_list, en_list, adir, fdir = [], [], [], []
            for k, od in enumerate(oo):
                i = idx_of[od]; j = i + h
                if j >= len(dates):
                    continue
                a_t, a_o = act.iloc[j][m], act.iloc[i][m]
                em_list.append(price[k] - a_t)
                en_list.append(a_o - a_t)
                adir.append(np.sign(a_t - a_o))
                fdir.append(np.sign(price[k] - a_o))
            em = np.array(em_list); en = np.array(en_list)
            adir = np.array(adir); fdir = np.array(fdir)
            mae_m, mae_n = np.abs(em).mean(), np.abs(en).mean()
            r2 = 1 - (em**2).sum() / (en**2).sum()
            nz = adir != 0
            da = (adir[nz] == fdir[nz]).mean() * 100 if nz.any() else np.nan
            rows.append(dict(Market=m, Horizon=f'D+{h}', Layer=layer, N=len(em),
                             MAE=mae_m, MAE_vs_naive_pct=100*(mae_n-mae_m)/mae_n,
                             RMSE=np.sqrt((em**2).mean()), R2_vs_naive=r2, DA_pct=da))

res = pd.DataFrame(rows)
res.to_csv('data/metrics_all_layers_oos.csv', index=False)

# ---- report ----
pd.set_option('display.width', 220, 'display.max_rows', 200)
deployed = {('TTF',1):'L3',('TTF',5):'L3',('POWER',1):'L3'}  # rest naive
print('\n' + '='*108)
print('ALL-LAYER OOS COMPARISON (naive vs L1 vs L3) — full 18 months')
print('  * = deployed policy choice   |   best = lowest OOS RMSE in the cell')
print('='*108)
for m in MARKETS:
    for h in HORIZONS:
        sub = res[(res.Market==m) & (res.Horizon==f'D+{h}')].copy()
        best_layer = sub.loc[sub.RMSE.idxmin(), 'Layer']
        dep = deployed.get((m,h), 'naive')
        print(f'\n{m} D+{h}:')
        for _, r in sub.iterrows():
            tag = ' *deployed' if r.Layer==dep else ''
            tag += ' <BEST' if r.Layer==best_layer else ''
            print(f'   {r.Layer:5} MAE={r.MAE:7.3f}  vs_naive={r.MAE_vs_naive_pct:+6.1f}%  '
                  f'R2={r.R2_vs_naive:+6.3f}  DA={r.DA_pct:5.1f}%{tag}')

# Summary: did the policy pick the OOS-best layer?
print('\n' + '='*108)
print('VERDICT: would a policy-discarded model have beaten naive OOS?')
print('='*108)
for m in MARKETS:
    for h in HORIZONS:
        sub = res[(res.Market==m) & (res.Horizon==f'D+{h}')]
        best = sub.loc[sub.RMSE.idxmin()]
        naive_rmse = sub[sub.Layer=='naive'].RMSE.values[0]
        improved = best.Layer != 'naive' and best.RMSE < naive_rmse
        print(f'  {m:5} D+{h}: OOS-best = {best.Layer:5} '
              f'({"beats naive by %.1f%%"%(100*(naive_rmse-best.RMSE)/naive_rmse) if improved else "naive is best / nothing beats it"})')
