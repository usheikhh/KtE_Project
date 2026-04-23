"""
Merge GIE AGSI columns into k2_clean.csv → k2_clean_v2.csv
Prefix all new columns with GIE_ to distinguish from existing k2_clean fields.
"""

import pandas as pd
from pathlib import Path

WORKSPACE = Path(__file__).parent
ML_INTERN = WORKSPACE.parent
GIE_DIR = WORKSPACE / "gie_data"

CLEAN_IN  = ML_INTERN / "k2_clean.csv"
CLEAN_OUT = ML_INTERN / "k2_clean_v2.csv"


def load_and_rename(path, prefix, keep_cols):
    df = pd.read_csv(path)
    df['gasDayStart'] = pd.to_datetime(df['gasDayStart'])
    df = df[['gasDayStart'] + keep_cols].copy()
    rename = {c: f"{prefix}_{c.upper()}" for c in keep_cols}
    df = df.rename(columns=rename)
    df = df.rename(columns={'gasDayStart': 'DATE'})
    return df


def main():
    clean = pd.read_csv(CLEAN_IN)
    clean['DATE'] = pd.to_datetime(clean['DATE'])
    print(f"k2_clean: {len(clean)} rows, {clean['DATE'].min()} → {clean['DATE'].max()}")

    storage_cols = ['full', 'injection', 'withdrawal', 'netWithdrawal',
                    'gasInStorage', 'consumptionFull', 'trend']

    eu = load_and_rename(GIE_DIR / "gie_eu_storage.csv", "GIE_EU", storage_cols)
    de = load_and_rename(GIE_DIR / "gie_de_storage.csv", "GIE_DE", storage_cols)

    out = pd.read_csv(GIE_DIR / "gie_de_outages_daily.csv")
    out['date'] = pd.to_datetime(out['date'])
    out = out.rename(columns={'date': 'DATE',
                              'OUT_PLANNED_GWH_D':   'GIE_DE_OUT_PLANNED_GWH_D',
                              'OUT_UNPLANNED_GWH_D': 'GIE_DE_OUT_UNPLANNED_GWH_D',
                              'OUT_N_EVENTS':        'GIE_DE_OUT_N_EVENTS'})

    # Left-join on DATE: keep k2_clean rows (trading days), pull in GIE values.
    # Where k2_clean has a trading day but GIE gas day is the same date, rows align.
    merged = clean.merge(eu, on='DATE', how='left') \
                  .merge(de, on='DATE', how='left') \
                  .merge(out, on='DATE', how='left')

    # Fill NaNs for outage counts with 0 (GIE outage data is complete, NaN = no events)
    merged['GIE_DE_OUT_PLANNED_GWH_D']   = merged['GIE_DE_OUT_PLANNED_GWH_D'].fillna(0)
    merged['GIE_DE_OUT_UNPLANNED_GWH_D'] = merged['GIE_DE_OUT_UNPLANNED_GWH_D'].fillna(0)
    merged['GIE_DE_OUT_N_EVENTS']        = merged['GIE_DE_OUT_N_EVENTS'].fillna(0)

    # Forward-fill storage on weekends (GIE publishes daily but trading days have gaps)
    gie_cols = [c for c in merged.columns if c.startswith('GIE_EU_') or c.startswith('GIE_DE_')]
    storage_fill_cols = [c for c in gie_cols if 'OUT_' not in c]
    merged[storage_fill_cols] = merged[storage_fill_cols].ffill(limit=3)

    # Causal seasonality: expanding-DOY mean of EU_FULL (no leakage)
    # For each trading day, compute mean of EU_FULL for same DOY in PRIOR years only
    merged_sorted = merged.sort_values('DATE').reset_index(drop=True)
    merged_sorted['DOY'] = merged_sorted['DATE'].dt.dayofyear
    merged_sorted['GIE_EU_FULL_DOY_EXPAND_MEAN'] = (
        merged_sorted.groupby('DOY')['GIE_EU_FULL']
                     .transform(lambda s: s.expanding().mean().shift(1)))
    merged_sorted['GIE_EU_FULL_DOY_DEV'] = (
        merged_sorted['GIE_EU_FULL'] - merged_sorted['GIE_EU_FULL_DOY_EXPAND_MEAN'])
    merged_sorted['GIE_DE_FULL_DOY_EXPAND_MEAN'] = (
        merged_sorted.groupby('DOY')['GIE_DE_FULL']
                     .transform(lambda s: s.expanding().mean().shift(1)))
    merged_sorted['GIE_DE_FULL_DOY_DEV'] = (
        merged_sorted['GIE_DE_FULL'] - merged_sorted['GIE_DE_FULL_DOY_EXPAND_MEAN'])
    merged_sorted = merged_sorted.drop(columns=['DOY'])

    # Coverage check
    new_cols = [c for c in merged_sorted.columns if c.startswith('GIE_')]
    coverage = merged_sorted[new_cols].notna().mean() * 100
    print(f"\nAdded {len(new_cols)} GIE columns. Coverage %:")
    for c, pct in coverage.items():
        print(f"  {c:40s}  {pct:5.1f}%")

    merged_sorted.to_csv(CLEAN_OUT, index=False)
    print(f"\n✓ wrote {CLEAN_OUT}")
    print(f"  {len(merged_sorted)} rows, {len(merged_sorted.columns)} cols "
          f"(+{len(merged_sorted.columns)-len(clean.columns)} new)")


if __name__ == "__main__":
    main()
