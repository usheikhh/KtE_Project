"""
GIE AGSI Data Fetcher
─────────────────────
Pulls gas storage data from GIE AGSI API and writes to CSV for merge into k2_clean.

Endpoints used:
  /api?type=eu                   → EU aggregate daily storage
  /api?country=DE                → Germany country-aggregate daily storage
  /api/unavailability?country=DE → Germany storage outage events

Outputs (Workspace/gie_data/):
  gie_eu_storage.csv
  gie_de_storage.csv
  gie_de_outages_daily.csv       (events collapsed to daily GWh/d lost)

Authentication: requires AGSI_API_KEY in .env or env var.
"""

import os
import time
import sys
import json
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

BASE = "https://agsi.gie.eu/api"
OUT_DIR = Path(__file__).parent / "gie_data"
OUT_DIR.mkdir(exist_ok=True)

# Match k2_clean range plus a buffer for lag/diff features
DATE_FROM = "2020-06-01"
DATE_TO   = "2024-12-31"


def _load_key():
    # Prefer env var; fallback to .env in same dir
    key = os.environ.get("AGSI_API_KEY")
    if key:
        return key
    env_path = Path(__file__).parent / "KtE_Project" / "Jeff workspace" / ".env"
    if not env_path.exists():
        env_path = Path(__file__).parent / "KtE_Project" / "Jeff workspace" / ".env.example"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("AGSI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("AGSI_API_KEY not found in env or .env file")


def _get(params, key, max_retries=3):
    headers = {"x-key": key}
    for attempt in range(max_retries):
        try:
            r = requests.get(BASE + params.pop("_path", ""), params=params,
                             headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_storage(key, scope, code):
    """scope='type' or 'country'; code='eu' or 'DE' etc."""
    print(f"[storage] pulling {scope}={code} from {DATE_FROM} to {DATE_TO} ...")
    all_rows = []
    page = 1
    while True:
        params = {scope: code, "from": DATE_FROM, "to": DATE_TO,
                  "size": 300, "page": page}
        j = _get(params, key)
        rows = j.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        if page >= j.get("last_page", 1):
            break
        page += 1
        time.sleep(0.15)  # be polite
    print(f"  got {len(all_rows)} rows")
    df = pd.DataFrame(all_rows)
    # numeric columns to coerce
    num_cols = ['gasInStorage', 'consumption', 'consumptionFull', 'injection',
                'withdrawal', 'netWithdrawal', 'workingGasVolume',
                'injectionCapacity', 'withdrawalCapacity', 'trend', 'full']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['gasDayStart'] = pd.to_datetime(df['gasDayStart'], errors='coerce')
    df = df.sort_values('gasDayStart').reset_index(drop=True)
    return df


def fetch_outages(key, country):
    """Pull unavailability events for a country."""
    print(f"[outages] pulling country={country} ...")
    all_rows = []
    page = 1
    while True:
        params = {"_path": "/unavailability", "country": country,
                  "from": DATE_FROM, "to": DATE_TO,
                  "size": 300, "page": page}
        j = _get(params, key)
        rows = j.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        if page >= j.get("last_page", 1):
            break
        page += 1
        time.sleep(0.15)
    print(f"  got {len(all_rows)} events")
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df['start']  = pd.to_datetime(df['start'],  errors='coerce')
    df['end']    = pd.to_datetime(df['end'],    errors='coerce')
    df['volume']     = pd.to_numeric(df.get('volume'),     errors='coerce')
    df['injection']  = pd.to_numeric(df.get('injection'),  errors='coerce')
    df['withdrawal'] = pd.to_numeric(df.get('withdrawal'), errors='coerce')
    return df


def outages_to_daily(outages_df, date_from=DATE_FROM, date_to=DATE_TO):
    """Collapse event-based outages to daily: for each calendar day, sum the
    injection+withdrawal capacity lost by events ACTIVE on that day.
    Event is active if start <= day <= end.
    Separate planned vs unplanned."""
    if outages_df.empty:
        dates = pd.date_range(date_from, date_to, freq='D')
        return pd.DataFrame({'date': dates,
                             'OUT_PLANNED_GWH_D': 0.0,
                             'OUT_UNPLANNED_GWH_D': 0.0,
                             'OUT_N_EVENTS': 0})

    dates = pd.date_range(date_from, date_to, freq='D')
    records = []
    # Precompute for efficiency: for each day, find active events
    starts = outages_df['start'].fillna(pd.Timestamp.min).dt.normalize().values
    ends   = outages_df['end'].fillna(pd.Timestamp.max).dt.normalize().values
    inj  = outages_df['injection'].fillna(0).values
    wd   = outages_df['withdrawal'].fillna(0).values
    typ  = outages_df['type'].fillna('Unknown').values

    for day in dates:
        d64 = day.to_datetime64()
        mask = (starts <= d64) & (ends >= d64)
        if not mask.any():
            records.append((day, 0.0, 0.0, 0))
            continue
        cap_lost = (inj[mask] + wd[mask])  # GWh/d total capacity unavailable
        planned_mask   = (typ[mask] == 'Planned')
        unplanned_mask = ~planned_mask
        records.append((day,
                        float(cap_lost[planned_mask].sum()),
                        float(cap_lost[unplanned_mask].sum()),
                        int(mask.sum())))
    return pd.DataFrame(records, columns=['date', 'OUT_PLANNED_GWH_D',
                                           'OUT_UNPLANNED_GWH_D', 'OUT_N_EVENTS'])


def main():
    key = _load_key()
    print(f"Using API key: {key[:6]}...{key[-4:]}")

    # EU aggregate
    eu = fetch_storage(key, "type", "eu")
    eu_out = OUT_DIR / "gie_eu_storage.csv"
    eu.to_csv(eu_out, index=False)
    print(f"  wrote {eu_out} ({len(eu)} rows)")

    # Germany country-aggregate: filter to name='Germany' to drop facility detail
    de = fetch_storage(key, "country", "DE")
    de_agg = de[de['name'].str.lower() == 'germany'].copy()
    de_out = OUT_DIR / "gie_de_storage.csv"
    de_agg.to_csv(de_out, index=False)
    print(f"  wrote {de_out} ({len(de_agg)} rows Germany-aggregate of {len(de)} total)")

    # Germany outages
    out = fetch_outages(key, "DE")
    out_raw = OUT_DIR / "gie_de_outages_raw.csv"
    if not out.empty:
        out_flat = out.copy()
        # Flatten nested cols
        for nested in ['country', 'company', 'facility']:
            if nested in out_flat.columns:
                out_flat[f'{nested}_name'] = out_flat[nested].apply(
                    lambda x: x.get('name') if isinstance(x, dict) else None)
                out_flat = out_flat.drop(columns=[nested])
        out_flat.to_csv(out_raw, index=False)
        print(f"  wrote {out_raw} ({len(out_flat)} events)")

    daily_out = outages_to_daily(out)
    daily_csv = OUT_DIR / "gie_de_outages_daily.csv"
    daily_out.to_csv(daily_csv, index=False)
    print(f"  wrote {daily_csv} ({len(daily_out)} days)")

    print("\nDone. Sample of each:")
    print("\nEU storage head:")
    print(eu[['gasDayStart', 'full', 'injection', 'withdrawal',
              'netWithdrawal']].head().to_string())
    print("\nDE storage head:")
    print(de_agg[['gasDayStart', 'full', 'injection', 'withdrawal',
                  'netWithdrawal']].head().to_string())
    print("\nDaily outages head (first active day):")
    nonzero = daily_out[daily_out['OUT_N_EVENTS'] > 0]
    if not nonzero.empty:
        print(nonzero.head().to_string())


if __name__ == "__main__":
    main()
