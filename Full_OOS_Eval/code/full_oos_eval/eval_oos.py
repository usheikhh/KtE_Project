"""Full out-of-sample evaluation (Jan 2025 - Jun 2026, ~18 months) of the
deployed K2E V4.4 model, replicating the original walkthrough metric logic
(MAE / RMSE / R2-vs-naive / directional accuracy) over the full OOS window.
Also reports a Jan-2025-only slice for comparison with the original deck.
"""
import numpy as np
import pandas as pd

ACT_MAP = {
    'TTF':   'BM_TTF_M1_CLOSE_EUR_MWH',
    'POWER': 'BM_GERMANY_POWER_M1_CLOSE_EUR_MWH',
    'EUA':   'BM_EUA_CO2_CAL1_PRICE_EUR_TON',
}

full   = pd.read_csv('data/full_parsed.csv', parse_dates=['Date'])
pred   = pd.read_csv('data/full_oos_predictions.csv', parse_dates=['Date'])

act = full.set_index('Date')[list(ACT_MAP.values())].rename(
    columns={v: k for k, v in ACT_MAP.items()}).sort_index()
dates = list(act.index)
idx_of = {d: i for i, d in enumerate(dates)}


def build_forecasts(pred_df):
    F = {}
    for m in ['TTF', 'POWER', 'EUA']:
        for h in [1, 3, 5]:
            rows = []
            for _, r in pred_df.iterrows():
                i = idx_of[r['Date']]
                j = i + h
                if j >= len(dates):
                    continue
                tgt = dates[j]
                rows.append(dict(origin=r['Date'], target=tgt,
                                 forecast=r[f'{m}_h{h}_price'],
                                 actual=act.loc[tgt, m],
                                 actual_origin=act.loc[r['Date'], m]))
            F[(m, h)] = pd.DataFrame(rows)
    return F


def metrics_table(F, label):
    rows = []
    for (m, h), df in F.items():
        em = df['forecast'] - df['actual']
        en = df['actual_origin'] - df['actual']
        mae_m, mae_n = em.abs().mean(), en.abs().mean()
        rmse_m, rmse_n = np.sqrt((em**2).mean()), np.sqrt((en**2).mean())
        sse_n = (en**2).sum()
        r2 = 1 - (em**2).sum() / sse_n if sse_n > 0 else np.nan
        ad = np.sign(df['actual'] - df['actual_origin'])
        fd = np.sign(df['forecast'] - df['actual_origin'])
        nz = ad != 0
        da = (ad[nz] == fd[nz]).mean() if nz.any() else np.nan
        rows.append(dict(Period=label, Market=m, Horizon=f'D+{h}', N=len(df),
                         MAE_model=mae_m, MAE_naive=mae_n,
                         MAE_impr_pct=100*(mae_n-mae_m)/mae_n if mae_n else 0,
                         RMSE_model=rmse_m, RMSE_naive=rmse_n, R2_vs_naive=r2,
                         DA_pct=100*da if not np.isnan(da) else np.nan))
    return pd.DataFrame(rows)


def eua_direction(pred_df, label):
    rows = []
    for _, r in pred_df.iterrows():
        i = idx_of[r['Date']]; j = i + 3
        if j >= len(dates):
            continue
        a0, a3 = act.iloc[i]['EUA'], act.iloc[j]['EUA']
        rows.append(dict(pred=r['EUA_h3_direction'],
                         actual='UP' if a3 > a0 else 'DOWN'))
    e = pd.DataFrame(rows)
    e['correct'] = e.pred == e.actual
    n, c = len(e), int(e.correct.sum())
    return dict(Period=label, N=n, Correct=c, DA_pct=100*c/n,
                pred_UP_pct=100*(e.pred=='UP').mean(),
                actual_UP_pct=100*(e.actual=='UP').mean())


# Full OOS and Jan-2025 slice
slices = {'Full OOS (2025-01..2026-06)': pred,
          'Jan-2025 only': pred[pred.Date < '2025-02-01'],
          '2025': pred[(pred.Date >= '2025-01-01') & (pred.Date < '2026-01-01')],
          '2026 YTD': pred[pred.Date >= '2026-01-01']}

all_metrics = []
for lab, pdf in slices.items():
    F = build_forecasts(pdf)
    all_metrics.append(metrics_table(F, lab))
metrics = pd.concat(all_metrics, ignore_index=True)
metrics.to_csv('data/metrics_full_oos.csv', index=False)

pd.set_option('display.width', 200, 'display.max_columns', 20)
print('=' * 100)
print('FULL OOS PRICE-FORECAST METRICS (deployed V4.4 model vs naive random walk)')
print('=' * 100)
full_m = metrics[metrics.Period.str.startswith('Full')]
with pd.option_context('display.float_format', lambda x: f'{x:.3f}'):
    print(full_m[['Market','Horizon','N','MAE_model','MAE_naive','MAE_impr_pct',
                  'R2_vs_naive','DA_pct']].to_string(index=False))

print('\n' + '=' * 100)
print('ACTIVE (non-naive) CELLS — model vs naive, by period')
print('=' * 100)
active = [('TTF','D+1'),('TTF','D+5'),('POWER','D+1')]
for lab in slices:
    sub = metrics[(metrics.Period==lab)]
    sub = sub[sub.apply(lambda r:(r.Market,r.Horizon) in active, axis=1)]
    print(f'\n-- {lab} --')
    with pd.option_context('display.float_format', lambda x: f'{x:.3f}'):
        print(sub[['Market','Horizon','N','MAE_impr_pct','R2_vs_naive','DA_pct']].to_string(index=False))

print('\n' + '=' * 100)
print('EUA D+3 DIRECTIONAL SIGNAL (reconstructed logistic classifier)')
print('=' * 100)
for lab, pdf in slices.items():
    d = eua_direction(pdf, lab)
    print(f"  {lab:32} {d['Correct']:>3}/{d['N']:<3} = {d['DA_pct']:5.1f}%   "
          f"(pred UP {d['pred_UP_pct']:.0f}%, actual UP {d['actual_UP_pct']:.0f}%)")
