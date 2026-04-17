"""
Macro factor Rotation — Free Data Ingestion
Sources: yFinance, FRED (free key)
"""

import os, time, logging
from datetime import datetime, timedelta
from pathlib import Path

import requests, feedparser, pandas as pd, yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR   = Path("data"); DATA_DIR.mkdir(exist_ok=True)
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
FRED_KEY   = os.getenv("FRED_API_KEY", "")

SECTOR_ETFS = {
    "SPY":"S&P 500","XLE":"Energy","XLF":"Financials","XLK":"Technology",
    "XLV":"Healthcare","XLI":"Industrials","XLB":"Materials",
    "XLU":"Utilities","XLP":"Consumer Staples","XLY":"Consumer Discretionary",
}
FX_TICKERS = {"GC=F":"Gold_USD","CL=F":"WTI_Oil","USDINR=X":"USD_INR","DX-Y.NYB":"DXY"}

FRED_SERIES = {
    "T10Y2Y":"yield_curve","VIXCLS":"VIX",
    "CPIAUCSL":"CPI_US","DTWEXBGS":"DXY_FRED",
}



# ── 1. EQUITIES ─────────────────────────────────────────────────────────────

def fetch_equities() -> pd.DataFrame:
    log.info("[2/4] Sector ETFs: yFinance")
    symbols = list(SECTOR_ETFS.keys())
    raw = yf.download(symbols, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    frames = []
    for sym in symbols:
        try:
            df = raw.xs(sym, axis=1, level=1)[["Close","Volume"]].copy()
            df.columns = ["close","volume"]
            df["ticker"] = sym
            df["sector"] = SECTOR_ETFS[sym]
            df["daily_return_pct"] = df["close"].pct_change() * 100
            df = df.reset_index().rename(columns={"Date":"date"})
            df["date"] = pd.to_datetime(df["date"]).dt.date
            frames.append(df)
        except Exception as e:
            log.error(f"  {sym}: {e}")
    df_out = pd.concat(frames, ignore_index=True)
    log.info(f"Equities: {len(df_out)} rows, {df_out['ticker'].nunique()} tickers")
    return df_out


# ── 2. FRED MACRO ────────────────────────────────────────────────────────────

def fetch_fred() -> pd.DataFrame:
    log.info("[3/4] Macro: FRED API")
    if not FRED_KEY:
        log.warning("  FRED_API_KEY not set. Get free key: fred.stlouisfed.org")
        return pd.DataFrame()
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_KEY)
    except ImportError:
        log.warning("  Run: pip install fredapi")
        return pd.DataFrame()
    frames = []
    for sid, col in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=START_DATE, observation_end=END_DATE)
            frames.append(s.rename(col))
            log.info(f"  {sid}: {len(s)} obs")
        except Exception as e:
            log.error(f"  {sid}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).ffill().reset_index()
    df.columns = ["date"] + list(df.columns[1:])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ── 3. FX & COMMODITIES ──────────────────────────────────────────────────────

def fetch_fx() -> pd.DataFrame:
    log.info("[4/4] FX & Commodities: yFinance")
    frames = []
    for sym, label in FX_TICKERS.items():
        try:
            raw = yf.download(sym, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
            s = raw["Close"].squeeze().rename(label)
            s.index = pd.to_datetime(s.index).date
            frames.append(s)
            log.info(f"  {sym}: {len(s)} rows")
        except Exception as e:
            log.error(f"  {sym}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).reset_index().rename(columns={"index":"date"})


# ── 4. MASTER TABLE ──────────────────────────────────────────────────────────

def build_master(equities, macro, fx) -> pd.DataFrame:
    if equities.empty:
        return pd.DataFrame()
    pivot = equities.pivot_table(index="date", columns="ticker", values="daily_return_pct").reset_index()
    pivot.columns = ["date"] + [f"ret_{c}" for c in pivot.columns[1:]]
    spy = equities[equities["ticker"]=="SPY"][["date","close"]].rename(columns={"close":"SPY_close"})
    master = pivot.merge(spy, on="date", how="left")
    for df in [macro, fx]:
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            master = master.merge(df, on="date", how="left")
    master["date"] = pd.to_datetime(master["date"])
    master["year"]    = master["date"].dt.year
    master["quarter"] = master["date"].dt.quarter
    master["month"]   = master["date"].dt.month
    return master.sort_values("date").reset_index(drop=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Date range: {START_DATE} to {END_DATE}")
    

    equities = fetch_equities()
    equities.to_csv(DATA_DIR/"equities_daily.csv", index=False)

    macro = fetch_fred()
    if not macro.empty:
        macro.to_csv(DATA_DIR/"macro_daily.csv", index=False)

    fx = fetch_fx()
    if not fx.empty:
        fx.to_csv(DATA_DIR/"fx_commodities.csv", index=False)

    master = build_master(equities, macro, fx)
    master.to_csv(DATA_DIR/"master_daily.csv", index=False)

    log.info("=== Done ===")
    for f in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        log.info(f"  {f.name:<28} {len(df):>5} rows x {len(df.columns):>2} cols")
    log.info("Next: python llm_tagger.py")

if __name__ == "__main__":
    main()
