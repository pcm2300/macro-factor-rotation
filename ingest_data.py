"""
Geopolitical Risk Dashboard — Free Data Ingestion
Sources: GDELT, RSS feeds, yFinance, FRED (free key)
Run: python ingest_data.py
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

RSS_FEEDS = [
    ("AP News",          "https://rsshub.app/apnews/topics/ap-top-news"),
    ("The Hindu",        "https://www.thehindu.com/news/international/?service=rss"),
    ("Economic Times",   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Al Jazeera Economy", "https://www.aljazeera.com/xml/rss/economy.xml"),
    ("Guardian World",     "https://www.theguardian.com/world/rss"),
]


GEO_KEYWORDS = [
    "war","conflict","sanction","missile","attack","military","troops",
    "tariff","trade","OPEC","oil","crude","embargo","ceasefire","nuclear",
    "inflation","Fed","rate","dollar","rupee","gold","recession","debt",
    "Taiwan","Ukraine","Russia","Gaza","Iran","China","NATO","India",
    "election","crisis","coup","protest","riot","collapse","default",
    "supply chain","energy","gas","pipeline","export","import","ban",
    "ceasefire","troops","nuclear","regime","bilateral","diplomatic",
    "exports","imports","currency","market","surge","slump","rally",
    "RBI","SEBI","Nifty","crude","commodity","geopolit",
]


# ── 1. NEWS ─────────────────────────────────────────────────────────────────

def fetch_wikipedia_events() -> pd.DataFrame:
    import re
    rows = []

    WIKI_CATEGORIES = [
        "Armed conflicts", "Disasters", "International relations",
        "Law and crime", "Politics", "Economics", "Military",
    ]

    for i in range(6):
        dt = datetime.today().replace(day=1) - timedelta(days=i*30)
        month_str = dt.strftime("%B_%Y")
        url = f"https://en.wikipedia.org/wiki/Portal:Current_events/{month_str}"
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()

            # Extract only the current-events-content divs
            content_blocks = re.findall(
                r'<div class="current-events-content description">(.*?)</div>\s*</div>\s*</div>',
                r.text, re.DOTALL
            )

            # Extract date for each block
            date_blocks = re.findall(
                r'<span class="bday dtstart published updated itvstart">(\d{4}-\d{2}-\d{2})</span>.*?'
                r'<div class="current-events-content description">(.*?)</div>\s*</div>\s*</div>',
                r.text, re.DOTALL
            )

            count = 0
            for date_str, block in date_blocks:
                # Get current category (bold heading above the li)
                current_cat = "General"
                # Split block by bold headings and li items
                segments = re.split(r'<b>(.*?)</b>', block)
                for j, seg in enumerate(segments):
                    # Odd indices are category names
                    if j % 2 == 1:
                        current_cat = re.sub(r'<[^>]+>', '', seg).strip()
                        continue
                    # Skip categories we don't care about
                    if not any(c.lower() in current_cat.lower() for c in WIKI_CATEGORIES):
                        continue
                    # Extract all <li> items
                    lis = re.findall(r'<li>(.*?)</li>', seg, re.DOTALL)
                    for li in lis:
                        clean = re.sub(r'<[^>]+>', '', li).strip()
                        clean = re.sub(r'\s+', ' ', clean)
                        # Only deepest level items have actual event descriptions
                        if len(clean) < 60 or len(clean) > 500:
                            continue
                        rows.append({
                            "published_at": date_str,
                            "source":       "Wikipedia Current Events",
                            "title":        clean,
                            "query":        f"wikipedia_{current_cat}",
                        })
                        count += 1

            log.info(f"  Wikipedia [{month_str}]: {count} events")
        except Exception as e:
            log.error(f"  Wikipedia [{month_str}]: {e}")

    df = pd.DataFrame(rows).drop_duplicates("title")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    log.info(f"Wikipedia total: {len(df)} events")
    return df

def fetch_gdelt_bulk(days_back=60) -> pd.DataFrame:
    import zipfile, io
    base = "http://data.gdeltproject.org/events/"
    rows = []

    GEO_CODES = {14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 2, 3}

    for i in range(2, days_back + 2):
        dt = datetime.today() - timedelta(days=i)
        filename = dt.strftime("%Y%m%d") + ".export.CSV.zip"
        url = base + filename
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(
                z.open(z.namelist()[0]), sep="\t", header=None,
                usecols=[1, 6, 26, 30, 57],
                names=["date", "actor", "eventcode", "goldstein", "sourceurl"],
                on_bad_lines="skip",
                dtype={"date": str, "actor": str, "sourceurl": str}
            )
            df["goldstein"] = pd.to_numeric(df["goldstein"], errors="coerce").fillna(0)
            df = df[df["goldstein"].abs() >= 7]

            df["eventcode"] = pd.to_numeric(df["eventcode"], errors="coerce").fillna(0).astype(int)
            df["code_prefix"] = df["eventcode"] // 10
            df = df[df["code_prefix"].isin(GEO_CODES)]

            df = df.sort_values("goldstein", ascending=False)
            df = df.drop_duplicates(subset=["date", "actor"])

            df["title"]        = df["actor"].fillna("Unknown") + " — Goldstein: " + df["goldstein"].round(1).astype(str)
            df["source"]       = df["sourceurl"].str.extract(r'https?://(?:www\.)?([^/]+)').fillna("")
            df["published_at"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            df["query"]        = "gdelt_bulk"

            # Drop rows outside our 2-year window
            df = df[df["published_at"] >= pd.Timestamp("2024-01-01")]

            rows.append(df[["published_at", "source", "title", "query"]])
            log.info(f"  GDELT {dt.strftime('%Y-%m-%d')}: {len(df)} events")
        except Exception as e:
            log.error(f"  GDELT {filename}: {e}")

    if not rows:
        return pd.DataFrame(columns=["published_at", "source", "title", "query"])
    return pd.concat(rows, ignore_index=True)


def fetch_gdelt(days_back=90) -> pd.DataFrame:
    """GDELT Doc 2.0 API — free, no key, 250 results per call."""
    queries = [
        "geopolitical conflict war sanctions",
        "oil OPEC crude supply",
        "Federal Reserve inflation dollar",
        "trade war tariff China US",
    ]
    rows = []
    since = (datetime.today() - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")
    for q in queries:
        url = (
            f"https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={requests.utils.quote(q)}&mode=artlist"
            f"&maxrecords=250&startdatetime={since}&format=csv"
        )
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            for _, row in df.iterrows():
                rows.append({
                    "source":       row.get("domain", ""),
                    "title":        row.get("title", ""),
                    "published_at": str(row.get("seendate", ""))[:8],
                    "url":          row.get("url", ""),
                    "query":        q,
                })
            log.info(f"  GDELT [{q[:35]}] -> {len(df)} articles")
            time.sleep(1)
        except Exception as e:
            log.error(f"  GDELT error: {e}")
    df_out = pd.DataFrame(rows).drop_duplicates("title")
    df_out["published_at"] = pd.to_datetime(df_out["published_at"], format="%Y%m%d", errors="coerce")
    log.info(f"GDELT total: {len(df_out)} articles")
    return df_out


def fetch_rss() -> pd.DataFrame:
    BLOCKED_TITLES = [
        "52-week", "IPO", "shareholding", "quarterly", "Q4", "Q3",
        "mutual fund", "smallcap", "largecap", "buyback", "chimpanzee",
        "gambling", "motorbike", "tourist", "cricket", "football",
        "Sensex", "Nifty points", "stock split", "dividend", "results season",
    ]

    rows = []
    for name, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                title = e.get("title", "")
                if not any(k.lower() in title.lower() for k in GEO_KEYWORDS):
                    continue
                if any(b.lower() in title.lower() for b in BLOCKED_TITLES):
                    continue
                rows.append({
                    "source":       name,
                    "title":        title,
                    "published_at": e.get("published", "")[:10],
                    "url":          e.get("link", ""),
                    "query":        "rss",
                })
            log.info(f"  RSS [{name}] -> {len(feed.entries)} entries scanned")
            time.sleep(0.3)
        except Exception as e:
            log.error(f"  RSS [{name}]: {e}")

    df = pd.DataFrame(rows).drop_duplicates("title")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    log.info(f"RSS total: {len(df)} relevant articles")
    return df

def fetch_news() -> pd.DataFrame:
    log.info("[1/4] News: GDELT + RSS + Wikipedia")
    df = pd.concat([
        fetch_gdelt_bulk(days_back=60),
        fetch_rss(),
        fetch_wikipedia_events(),
    ], ignore_index=True).drop_duplicates("title")
    df = df[df["title"].notna()].sort_values("published_at")
    log.info(f"News combined: {len(df)} unique headlines")
    return df

"""def fetch_news() -> pd.DataFrame:
    log.info("[1/4] News: GDELT bulk + RSS")
    df = pd.concat([fetch_gdelt_bulk(days_back=60), fetch_rss()], ignore_index=True)
    df = df.drop_duplicates("title").dropna(subset=["title"])
    df = df.sort_values("published_at")
    log.info(f"News combined: {len(df)} unique headlines")
    return df"""


# ── 2. EQUITIES ─────────────────────────────────────────────────────────────

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


# ── 3. FRED MACRO ────────────────────────────────────────────────────────────

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


# ── 4. FX & COMMODITIES ──────────────────────────────────────────────────────

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


# ── 5. MASTER TABLE ──────────────────────────────────────────────────────────

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
    news     = fetch_news()
    news.to_csv(DATA_DIR/"news_raw.csv", index=False)

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
