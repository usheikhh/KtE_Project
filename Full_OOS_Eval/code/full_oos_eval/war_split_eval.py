"""Split the OOS evaluation around the start of the Israel-Iran war
(2025-06-13) to test whether the model had predictive power in the calmer
pre-war regime. Buckets each forecast by its full h-day window:
  PRE      : target date  <  WAR_START          (window entirely pre-war)
  STRADDLE : origin < WAR_START <= target        (window crosses the shock)
  POST     : origin date  >= WAR_START           (window entirely post-war)
"""
import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
from K2E_Model_V4_2 import pt_test

WAR_START = pd.Timestamp('2025-06-13')

ACT = {'TTF': 'BM_TTF_M1_CLOSE_EUR_MWH',
       'POWER': 'BM_GERMANY_POWER_M1_CLOSE_EUR_MWH',
       'EUA': 'BM_EUA_CO2_CAL1_PRICE_EUR_TON'}

full = pd.read_csv('data/full_parsed.csv', parse_dates=['Date'])
pred = pd.read_csv('data/full_oos_predictions.csv', parse_dates=['Date'])
act = full.set_index('Date')[list(ACT.values())].rename(
    columns={v: k for k, v in ACT.items()}).sort_index()
dates = list(act.index); idx_of = {d: i for i, d in enumerate(dates)}


def bucket(origin, target):
    if target < WAR_START:
        return 'PRE'
    if origin < WAR_START <= target:
        return 'STRADDLE'
    return 'POST'


# ---------- 1) deployed-model PRICE metrics, active cells, by regime ----------
ACTIVE = [('TTF', 1), ('TTF', 5), ('POWER', 1)]
print('=' * 96)
print(f'DEPLOYED PRICE MODEL vs naive — split at Iran-war start {WAR_START.date()}')
print('=' * 96)
print(f'{"Cell":10}{"Regime":10}{"N":>5}{"MAE_model":>11}{"MAE_naive":>11}'
      f'{"MAEvsNv%":>10}{"R2_vs_nv":>10}{"Dir.acc%":>10}')
for (m, h) in ACTIVE:
    rows = []
    for _, r in pred.iterrows():
        i = idx_of[r['Date']]; j = i + h
        if j >= len(dates):
            continue
        o, t = r['Date'], dates[j]
        rows.append(dict(reg=bucket(o, t), f=r[f'{m}_h{h}_price'],
                         a=act.loc[t, m], a0=act.loc[o, m]))
    df = pd.DataFrame(rows)
    for reg in ['PRE', 'STRADDLE', 'POST']:
        s = df[df.reg == reg]
        if len(s) == 0:
            continue
        em = s.f - s.a; en = s.a0 - s.a
        mae_m, mae_n = em.abs().mean(), en.abs().mean()
        r2 = 1 - (em**2).sum() / (en**2).sum()
        ad = np.sign(s.a - s.a0); fd = np.sign(s.f - s.a0); nz = ad != 0
        da = (ad[nz] == fd[nz]).mean() * 100 if nz.any() else np.nan
        print(f'{m+" D+"+str(h):10}{reg:10}{len(s):>5}{mae_m:>11.3f}{mae_n:>11.3f}'
              f'{100*(mae_n-mae_m)/mae_n:>+10.1f}{r2:>+10.3f}{da:>10.1f}')


# ---------- 2) walk-forward directional, by regime ----------
det = pd.read_csv('data/walkforward_directional_detail.csv', parse_dates=['Date'])
H = {'D+1': 1, 'D+3': 3, 'D+5': 5}
print('\n' + '=' * 96)
print(f'WALK-FORWARD DIRECTIONAL SIGNAL — split at {WAR_START.date()}')
print('=' * 96)
print(f'{"Cell":10}{"Regime":10}{"N":>5}{"Dir.acc%":>10}{"PT_p":>8}'
      f'{"EV_bps":>9}{"Sharpe*":>9}{"predUP%":>9}')
out = []
for (m, hd), g in det.groupby(['Market', 'Horizon']):
    h = H[hd]
    g = g.copy()
    # target date via trading calendar
    tgt = []
    for o in g['Date']:
        i = idx_of.get(o)
        tgt.append(dates[i + h] if (i is not None and i + h < len(dates)) else pd.NaT)
    g['target'] = tgt
    g['reg'] = [bucket(o, t) if pd.notna(t) else 'POST' for o, t in zip(g['Date'], g['target'])]
    for reg in ['PRE', 'STRADDLE', 'POST']:
        s = g[g.reg == reg]
        yv = s['y'].values; sp = s['sign_pred'].values; nz = np.sign(yv) != 0
        if nz.sum() < 5:
            continue
        da = (np.sign(yv[nz]) == sp[nz]).mean() * 100
        _, ptp = pt_test(yv[nz], sp[nz])
        pnl = sp * yv; ev = pnl.mean()
        sh = ev / (pnl.std(ddof=0) + 1e-12) * np.sqrt(252.0 / h)
        up = (sp > 0).mean() * 100
        out.append(dict(Market=m, Horizon=hd, Regime=reg, N=int(nz.sum()),
                        DA=da, PT_p=ptp, EV_bps=ev*1e4, Sharpe=sh, predUP=up))
        print(f'{m+" "+hd:10}{reg:10}{int(nz.sum()):>5}{da:>10.1f}{ptp:>8.3f}'
              f'{ev*1e4:>+9.1f}{sh:>+9.2f}{up:>9.0f}')
pd.DataFrame(out).to_csv('data/war_split_directional.csv', index=False)

# ---------- 3) pre-war summary: any genuine edge? ----------
o = pd.DataFrame(out)
pre = o[(o.Regime == 'PRE')]
print('\n' + '=' * 96)
print('PRE-WAR directional cells with PT p<0.10:')
hit = pre[pre.PT_p < 0.10]
if len(hit):
    for _, r in hit.iterrows():
        print(f'  {r.Market} {r.Horizon}: DA={r.DA:.1f}%  PT_p={r.PT_p:.3f}  '
              f'Sharpe*={r.Sharpe:+.2f}  (N={r.N})')
else:
    print('  NONE')
print(f'\nPRE-war mean directional accuracy across 9 cells: {pre.DA.mean():.1f}%')
print(f'POST-war mean directional accuracy across 9 cells: {o[o.Regime=="POST"].DA.mean():.1f}%')
