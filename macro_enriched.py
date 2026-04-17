"""
Macro Factor Rotation Tracker — Data Enrichment
=================================================
Pulls missing FRED series, engineers macro regimes,
tags event dates, computes sector returns at T+1/T+5/T+20.

Run: python macro_enriched.py
Outputs (in data/):
  macro_events.csv       — each Fed/CPI/GDP/Jobs event with date + value
  sector_event_returns.csv — sector returns T+1/T+5/T+20 after each event
  regime_daily.csv       — daily data tagged with macro regime label
  macro_full.csv         — all macro series merged, ready for Power BI
"""

import os, logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
FRED_KEY = os.getenv("FRED_API_KEY", "")

END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

SECTORS = ["XLE","XLF","XLK","XLV","XLI","XLB","XLU","XLP","XLY","SPY"]


# ── 1. FETCH ADDITIONAL FRED SERIES ─────────────────────────────────────────

def fetch_extra_fred() -> pd.DataFrame:
    if not FRED_KEY:
        log.warning("FRED_API_KEY not set — skipping extra FRED series.")
        return pd.DataFrame()
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_KEY)
    except ImportError:
        log.warning("pip install fredapi")
        return pd.DataFrame()

    series = {
        ##"FEDFUNDS":         "fed_funds_rate",
        "DFF": "fed_funds_rate",
        "A191RL1Q225SBEA":  "gdp_growth_qoq",
        "PAYEMS":           "nonfarm_payrolls",
        "UNRATE":           "unemployment_rate",
        "DGS10":            "treasury_10y",
        "DGS2":             "treasury_2y",
    }

    frames = []
    for sid, col in series.items():
        try:
            s = fred.get_series(sid, observation_start=START_DATE, observation_end=END_DATE)
            frames.append(s.rename(col))
            log.info(f"  FRED {sid}: {len(s)} obs")
        except Exception as e:
            log.error(f"  FRED {sid}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).ffill().reset_index()
    df.columns = ["date"] + list(df.columns[1:])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Compute spread if both available
    if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
        df["yield_spread_10y2y"] = df["treasury_10y"] - df["treasury_2y"]

    return df


# ── 2. LOAD EXISTING DATA ────────────────────────────────────────────────────

def load_existing() -> tuple:
    equities = pd.read_csv(DATA_DIR / "equities_daily.csv")
    equities["date"] = pd.to_datetime(equities["date"]).dt.date

    macro = pd.read_csv(DATA_DIR / "macro_daily.csv")
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    fx = pd.read_csv(DATA_DIR / "fx_commodities.csv")
    fx["date"] = pd.to_datetime(fx["date"]).dt.date

    log.info(f"Loaded equities: {len(equities)} rows")
    log.info(f"Loaded macro: {len(macro)} rows")
    log.info(f"Loaded fx: {len(fx)} rows")
    return equities, macro, fx


# ── 3. BUILD MACRO FULL TABLE ────────────────────────────────────────────────

def build_macro_full(macro, fx, extra) -> pd.DataFrame:
    df = macro.copy()

    if not fx.empty:
        df = df.merge(fx, on="date", how="outer")

    if not extra.empty:
        extra["date"] = pd.to_datetime(extra["date"]).dt.date
        df = df.merge(extra, on="date", how="outer")

    df = df.sort_values("date").ffill().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Rate environment label
    if "fed_funds_rate" in df.columns:
        df["fed_rate_chg"] = df["fed_funds_rate"].diff()
        df["rate_regime"] = "Hold"
        df.loc[df["fed_rate_chg"] > 0.1,  "rate_regime"] = "Hiking"
        df.loc[df["fed_rate_chg"] < -0.1, "rate_regime"] = "Cutting"
        # Forward fill regime label so every day is tagged
        df["rate_regime"] = df["rate_regime"].replace("Hold", np.nan).ffill().fillna("Hold")

    # CPI regime
    if "CPI_US" in df.columns:
        df["cpi_yoy"] = df["CPI_US"].pct_change(12) * 100
        df["inflation_regime"] = pd.cut(
            df["cpi_yoy"],
            bins=[-np.inf, 2, 3.5, 5, np.inf],
            labels=["Low (<2%)", "Moderate (2-3.5%)", "Elevated (3.5-5%)", "High (>5%)"]
        )

    # Yield curve regime
    if "yield_spread_10y2y" in df.columns:
        df["curve_regime"] = np.where(df["yield_spread_10y2y"] < 0, "Inverted", "Normal")

    log.info(f"Macro full table: {len(df)} rows x {len(df.columns)} cols")
    return df


# ── 4. DETECT MACRO EVENTS ───────────────────────────────────────────────────

def detect_events(macro_full: pd.DataFrame) -> pd.DataFrame:
    """
    Tag discrete macro events:
    - Fed rate change (any month where fed_funds_rate changes)
    - CPI release (monthly, flag if surprise vs prior month)
    - GDP release (quarterly)
    - Jobs report (monthly nonfarm payrolls)
    - Yield curve inversion start/end
    """
    events = []

    df = macro_full.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Fed rate changes
    if "fed_funds_rate" in df.columns:
        fed = df[["date","fed_funds_rate"]].dropna().copy()
        fed["chg"] = fed["fed_funds_rate"].diff()
        rate_changes = fed[fed["chg"].abs() > 0.1].copy()
        for _, row in rate_changes.iterrows():
            bps = round(row["chg"] * 100)
            bps = round(bps / 25) * 25  # snap to nearest 25bps increment
            direction = "Hike" if row["chg"] > 0 else "Cut"
            events.append({
                "date":       row["date"],
                "event_type": "Fed Rate Change",
                "value":      round(row["fed_funds_rate"], 2),
                "change":     round(row["chg"], 2),
                "label":      f"Fed {direction} {abs(bps):.0f}bps",
            })

    # CPI monthly releases
    if "CPI_US" in df.columns:
        cpi = df[["date","CPI_US"]].dropna().copy()
        cpi = cpi.resample("ME", on="date").last().reset_index()
        cpi["mom_chg"] = cpi["CPI_US"].pct_change() * 100
        cpi["yoy"] = cpi["CPI_US"].pct_change(12) * 100
        for _, row in cpi.dropna().iterrows():
            events.append({
                "date":       row["date"],
                "event_type": "CPI Release",
                "value":      round(row["yoy"], 2),
                "change":     round(row["mom_chg"], 3),
                "label":      f"CPI {row['yoy']:.1f}% YoY",
            })

    # GDP quarterly
    if "gdp_growth_qoq" in df.columns:
        gdp = df[["date","gdp_growth_qoq"]].dropna().copy()
        gdp = gdp.resample("QE", on="date").last().reset_index()
        gdp["chg"] = gdp["gdp_growth_qoq"].diff()
        for _, row in gdp.dropna().iterrows():
            events.append({
                "date":       row["date"],
                "event_type": "GDP Release",
                "value":      round(row["gdp_growth_qoq"], 2),
                "change":     round(row["chg"], 2),
                "label":      f"GDP {row['gdp_growth_qoq']:.1f}% QoQ",
            })

    # Jobs report monthly
    if "nonfarm_payrolls" in df.columns:
        jobs = df[["date","nonfarm_payrolls"]].dropna().copy()
        jobs = jobs.resample("ME", on="date").last().reset_index()
        jobs["added"] = jobs["nonfarm_payrolls"].diff()
        for _, row in jobs.dropna().iterrows():
            added_k = row["added"]
            sign = "+" if added_k >= 0 else ""
            events.append({
                "date":       row["date"],
                "event_type": "Jobs Report",
                "value":      round(added_k, 0),
                "change":     round(added_k, 0),
                "label":      f"Payrolls {sign}{added_k:.0f}K",
            })

    # Yield curve inversion events
    if "yield_spread_10y2y" in df.columns:
        curve = df[["date","yield_spread_10y2y"]].dropna().copy()
        curve["inverted"] = curve["yield_spread_10y2y"] < 0
        curve["inv_change"] = curve["inverted"].astype(int).diff()
        inversions = curve[curve["inv_change"] != 0]
        for _, row in inversions.iterrows():
            direction = "Inversion Start" if row["inv_change"] == 1 else "Inversion End"
            events.append({
                "date":       row["date"],
                "event_type": "Yield Curve",
                "value":      round(row["yield_spread_10y2y"], 3),
                "change":     round(row["inv_change"], 0),
                "label":      f"Yield Curve {direction}",
            })

    events_df = pd.DataFrame(events).sort_values("date").reset_index(drop=True)
    log.info(f"Detected {len(events_df)} macro events")
    log.info(events_df["event_type"].value_counts().to_string())
    return events_df


# ── 5. COMPUTE SECTOR RETURNS AROUND EVENTS ──────────────────────────────────

def compute_event_returns(events: pd.DataFrame, equities: pd.DataFrame) -> pd.DataFrame:
    """
    For each macro event, compute sector returns at:
    T+1 (next day), T+5 (1 week), T+20 (1 month)
    """
    # Pivot equities to wide: date x ticker = close price
    prices = equities.pivot_table(index="date", columns="ticker", values="close")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    trading_dates = prices.index.tolist()

    rows = []
    for _, event in events.iterrows():
        event_date = pd.Timestamp(event["date"])

        # Find nearest trading date on or after event
        future = [d for d in trading_dates if d >= event_date]
        if not future:
            continue
        t0_date = future[0]
        t0_idx  = trading_dates.index(t0_date)

        for ticker in SECTORS:
            if ticker not in prices.columns:
                continue
            t0_price = prices[ticker].iloc[t0_idx] if t0_idx < len(trading_dates) else np.nan

            def pct_return(offset):
                idx = t0_idx + offset
                if idx >= len(trading_dates) or np.isnan(t0_price) or t0_price == 0:
                    return np.nan
                return round((prices[ticker].iloc[idx] / t0_price - 1) * 100, 3)

            rows.append({
                "event_date":  event_date.date(),
                "event_type":  event["event_type"],
                "event_label": event["label"],
                "event_value": event["value"],
                "event_change":event["change"],
                "ticker":      ticker,
                "return_T1":   pct_return(1),
                "return_T5":   pct_return(5),
                "return_T20":  pct_return(20),
            })

    df = pd.DataFrame(rows)
    log.info(f"Sector event returns: {len(df)} rows")
    return df


# ── 6. BUILD REGIME DAILY TABLE ─────────────────────────────────────────────

def build_regime_daily(macro_full: pd.DataFrame, equities: pd.DataFrame) -> pd.DataFrame:
    """
    One row per trading day per sector with regime labels attached.
    This powers the regime line charts in Power BI.
    """
    macro_slim = macro_full[[
        c for c in macro_full.columns
        if c in ["date","fed_funds_rate","cpi_yoy","yield_spread_10y2y",
                 "VIX","DXY","rate_regime","inflation_regime","curve_regime",
                 "Gold_USD","WTI_Oil","USD_INR"]
    ]].copy()
    macro_slim["date"] = pd.to_datetime(macro_slim["date"]).dt.date

    eq = equities[["date","ticker","sector","close","daily_return_pct"]].copy()
    eq["date"] = pd.to_datetime(eq["date"]).dt.date

    df = eq.merge(macro_slim, on="date", how="left")
    df["date"] = pd.to_datetime(df["date"])

    # Rolling 20-day return for momentum view
    df = df.sort_values(["ticker","date"])
    df["return_20d"] = df.groupby("ticker")["close"].transform(
        lambda x: x.pct_change(20) * 100
    ).round(3)

    log.info(f"Regime daily: {len(df)} rows")
    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Macro Factor Rotation — Enrichment ===")

    # Load existing
    equities, macro, fx = load_existing()

    # Fetch extra FRED series
    log.info("Fetching extra FRED series...")
    extra = fetch_extra_fred()

    # Build full macro table
    log.info("Building macro full table...")
    macro_full = build_macro_full(macro, fx, extra)
    macro_full.to_csv(DATA_DIR / "macro_full.csv", index=False)
    log.info(f"Saved macro_full.csv")

    # Detect events
    log.info("Detecting macro events...")
    events = detect_events(macro_full)
    events.to_csv(DATA_DIR / "macro_events.csv", index=False)
    log.info(f"Saved macro_events.csv — {len(events)} events")

    # Compute sector returns around each event
    log.info("Computing sector returns around events...")
    event_returns = compute_event_returns(events, equities)
    event_returns.to_csv(DATA_DIR / "sector_event_returns.csv", index=False)
    log.info(f"Saved sector_event_returns.csv")

    # Build regime daily
    log.info("Building regime daily table...")
    regime_daily = build_regime_daily(macro_full, equities)
    regime_daily.to_csv(DATA_DIR / "regime_daily.csv", index=False)
    log.info(f"Saved regime_daily.csv")

    # Summary
    log.info("\n=== Done ===")
    for f in ["macro_full.csv","macro_events.csv","sector_event_returns.csv","regime_daily.csv"]:
        df = pd.read_csv(DATA_DIR / f)
        log.info(f"  {f:<35} {len(df):>5} rows x {len(df.columns):>2} cols")

    log.info("\nPower BI files ready in data/")
    log.info("Next: open Power BI and load these 4 CSVs")

if __name__ == "__main__":
    main()
