# ================================================================
# FAST Portfolio Analysis with yfinance + ipywidgets
# UPDATED: Now uses monthly snapshot Parquet files with pre-calculated metrics
# + Lazy loading: only fetch symbols user selects via filters
# + Caching: reuse downloaded data across queries
# + Time-window performance (5Y/YTD/6M/3M/30D) with \$100 investment per period
# + Benchmarks (SPY, IXN) comparison
# + Value-investing conservative DCF panel with margin of safety verdicts
# ================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import logging, io, urllib.request
from datetime import date, timedelta
from ipywidgets import widgets, Layout, VBox, HBox, SelectMultiple, Dropdown, Button, Text, Output, IntSlider, FloatSlider, FloatText, ToggleButtons
from IPython.display import display, clear_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ================================================================
# GLOBAL CACHES
# ================================================================
PRICE_CACHE = {}  # key: (tuple(sorted_syms), str(start), str(end)) -> DataFrame
RECO_CACHE = {}
FUNDAMENTAL_CACHE = {}

# ================================================================
# Parquet Parsing (Monthly Snapshots)
# ================================================================
def parse_tickers_from_parquet_github(parquet_url):
    """
    Load data from monthly snapshot parquet file.
    Uses YahooSymbolClean column for tickers.
    Snapshot includes pre-calculated metrics: monthly, 3mo, 6mo, YTD, 1yr, 5yr performance.
    """
    with urllib.request.urlopen(parquet_url) as resp:
        data = resp.read()
    df = pd.read_parquet(io.BytesIO(data))

    logging.info(f"Parquet columns detected: {list(df.columns)}")
    display(df.head(5))

    # Map parquet columns to expected structure
    # Snapshot has: symbol, YahooSymbolClean, companyName, country, industry, Main Industry, Sub Industry
    # Plus: monthly_open, monthly_close, pct_change_3mo, pct_change_6mo, etc.

    records = []
    for _, row in df.iterrows():
        # Use YahooSymbolClean as primary ticker (cleaned for yfinance)
        sym = row.get("YahooSymbolClean") or row.get("symbol")
        if pd.isna(sym):
            continue
        sym = str(sym).strip()
        if not sym:
            continue

        rec = {
            "YahooSymbol": sym,
            "Company Name": str(row.get("companyName", "")).strip(),
            "Index": str(row.get("exchange_code", "")).strip(),
            "Country": str(row.get("country", "")).strip(),
            "Industry": str(row.get("Main Industry") or row.get("industry", "")).strip(),
            # Include pre-calculated metrics from snapshot
            "Monthly % Change": row.get("monthly_pct_change"),
            "3M % Change": row.get("pct_change_3mo"),
            "6M % Change": row.get("pct_change_6mo"),
            "YTD % Change": row.get("pct_change_ytd"),
            "1Y % Change": row.get("pct_change_1yr"),
            "5Y % Change": row.get("pct_change_5yr"),
        }
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    out = out.drop_duplicates(subset=["YahooSymbol"], keep="first")
    logging.info(f"Parsed {len(out)} unique symbols from parquet snapshot.")
    return out

def build_portfolio_from_df(meta_df):
    """
    Portfolio keyed by Yahoo symbol, carrying metadata.
    No shares/buy_price yet—those will be computed dynamically per period.
    """
    port = {}
    for _, r in meta_df.iterrows():
        yfs = r["YahooSymbol"]
        port[yfs] = {
            "yahoo_symbol": yfs,
            "name": yfs,
            "Company Name": r.get("Company Name", ""),
            "Index": r.get("Index", ""),
            "Country": r.get("Country", ""),
            "Industry": r.get("Industry", ""),
            "Monthly % Change": r.get("Monthly % Change"),
            "3M % Change": r.get("3M % Change"),
            "6M % Change": r.get("6M % Change"),
            "YTD % Change": r.get("YTD % Change"),
            "1Y % Change": r.get("1Y % Change"),
            "5Y % Change": r.get("5Y % Change"),
        }
    return port

# ================================================================
# Cached Data Fetching
# ================================================================
def _download_hist_for_period(symbols, start, end):
    """
    Download historical data with caching. Only fetches Close prices to save memory.
    """
    key = (tuple(sorted(symbols)), str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
    if key in PRICE_CACHE:
        logging.info(f"Cache hit for {len(symbols)} symbols from {start.date()} to {end.date()}")
        return PRICE_CACHE[key]

    logging.info(f"Downloading {len(symbols)} symbols from {start.date()} to {end.date()}...")
    df = yf.download(list(symbols), start=start, end=end, group_by="ticker", progress=False, threads=False)

    # Normalize datetime index to date-only format (removes timezone and time)
    df.index = df.index.normalize()

    # Keep only Close to reduce memory
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [(sym, "Close") for sym in symbols if (sym, "Close") in df.columns]
        if close_cols:
            df = df[close_cols]
    elif "Close" in df.columns:
        df = df[["Close"]]

    PRICE_CACHE[key] = df
    return df

def _period_start_end(label):
    today = date.today()
    if label == "30D":
        start = today - timedelta(days=30)
        return pd.Timestamp(start), pd.Timestamp(today)
    if label == "3M":
        start = pd.Timestamp(today) - pd.DateOffset(months=3)
        return pd.Timestamp(start), pd.Timestamp(today)
    if label == "6M":
        start = pd.Timestamp(today) - pd.DateOffset(months=6)
        return pd.Timestamp(start), pd.Timestamp(today)
    if label == "YTD":
        start = pd.Timestamp(date(today.year, 1, 1))
        return start, pd.Timestamp(today)
    if label == "5Y":
        start = pd.Timestamp(today) - pd.DateOffset(years=5)
        return pd.Timestamp(start), pd.Timestamp(today)
    return pd.Timestamp(today) - pd.DateOffset(months=6), pd.Timestamp(today)

def _pick_close_at(hist_df, yf_symbol, when):
    """Pick the closest available close price to a given timestamp."""
    if isinstance(hist_df.columns, pd.MultiIndex):
        if (yf_symbol, "Close") not in hist_df.columns:
            return None
        series = hist_df[(yf_symbol, "Close")].dropna()
    else:
        series = hist_df["Close"].dropna() if "Close" in hist_df.columns else pd.Series(dtype=float)
    if series.empty:
        return None
    idx = series.index
    # Normalize the comparison timestamp to match the normalized index
    pos = np.argmin(np.abs(idx - pd.Timestamp(when).normalize()))
    return float(series.iloc[pos])

# ================================================================
# Indicators
# ================================================================
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1: return None
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else None

def compute_indicators(h):
    h = h.copy()
    h["SMA20"] = h["Close"].rolling(20).mean()
    h["SMA50"] = h["Close"].rolling(50).mean()
    rsi = calculate_rsi(h["Close"], 14)
    close = float(h["Close"].iloc[-1])
    sma20 = float(h["SMA20"].iloc[-1]) if pd.notna(h["SMA20"].iloc[-1]) else None
    sma50 = float(h["SMA50"].iloc[-1]) if pd.notna(h["SMA50"].iloc[-1]) else None
    return close, sma20, sma50, rsi

# ================================================================
# Recommendations
# ================================================================
def get_recommendation(yf_symbol, hist_df=None, rsi=None, price=None, sma20=None, sma50=None):
    if yf_symbol in RECO_CACHE: return RECO_CACHE[yf_symbol]
    try:
        info = yf.Ticker(yf_symbol).info
        key = info.get("recommendationKey")
        if key:
            rec = key.replace("_", " ").title()
            RECO_CACHE[yf_symbol] = rec
            return rec
    except Exception:
        pass
    try:
        if hist_df is not None and (sma20 is None or sma50 is None or price is None or rsi is None):
            price, sma20, sma50, rsi = compute_indicators(hist_df)
    except Exception:
        pass
    rec = "Hold"
    if rsi and sma20 and sma50 and price:
        bull = (price > sma20 > sma50)
        bear = (price < sma20 < sma50)
        if rsi <= 30 and bull: rec = "Buy"
        elif rsi >= 70 and bear: rec = "Sell"
    RECO_CACHE[yf_symbol] = rec
    return rec

# ================================================================
# Current Snapshot (for Sort/Export Dashboard)
# ================================================================
def try_hist(sym, period="6mo"):
    try:
        return yf.Ticker(sym).history(period=period)
    except:
        return pd.DataFrame()

def assemble_current_snapshot(port, symbols_to_fetch):
    """
    Build a current snapshot DataFrame for sorting/export dashboard.
    Only fetches the symbols provided in symbols_to_fetch.
    Includes pre-calculated metrics from snapshot.
    """
    if not symbols_to_fetch:
        return []

    # Download recent history for snapshot
    hist_all = yf.download(symbols_to_fetch, period="6mo", group_by="ticker", progress=False, threads=False)

    rows = []
    multi = isinstance(hist_all.columns, pd.MultiIndex)

    for yf_sym in symbols_to_fetch:
        if yf_sym not in port:
            continue
        info = port[yf_sym]

        if multi:
            if yf_sym in hist_all.columns.get_level_values(0):
                h = hist_all[yf_sym].dropna()
            else:
                h = pd.DataFrame()
        else:
            h = hist_all.dropna() if not hist_all.empty else pd.DataFrame()

        if h.empty:
            h = try_hist(yf_sym)
            if h is None or h.empty:
                continue

        if len(h) < 50:
            price = float(h["Close"].iloc[-1]); sma20 = sma50 = rsi = None
        else:
            price, sma20, sma50, rsi = compute_indicators(h)

        # Notional \$100 position for display
        shares = 100.0 / price
        val = price * shares

        rows.append({
            "Company Name": info.get("Company Name", ""),
            "Index": info.get("Index", ""),
            "Country": info.get("Country", ""),
            "Industry": info.get("Industry", ""),
            "Ticker": yf_sym,
            "Yahoo Symbol": yf_sym,
            "Name": info["name"],
            "Price": price,
            "Shares (notional)": shares,
            "Position Value": val,
            "RSI": rsi,
            "SMA20": sma20,
            "SMA50": sma50,
            "Recommendation": get_recommendation(yf_sym, h, rsi, price, sma20, sma50),
            # Add pre-calculated metrics from snapshot
            "Monthly %": info.get("Monthly % Change"),
            "3M %": info.get("3M % Change"),
            "6M %": info.get("6M % Change"),
            "YTD %": info.get("YTD % Change"),
            "1Y %": info.get("1Y % Change"),
            "5Y %": info.get("5Y % Change"),
        })
    return rows

# ================================================================
# UI: Sort + Export Dashboard
# ================================================================
def sort_dataframe(df, col, asc=True):
    if col not in df.columns: return df
    return df.sort_values(by=col, ascending=asc, kind="mergesort")

class PortfolioTableUI:
    def __init__(self, meta_df, port, currency="USD"):
        self.meta = meta_df
        self.port = port
        self.df = None
        self.currency = currency
        self.output = Output()

        # Filters
        countries = ["(All)"] + sorted([c for c in self.meta["Country"].dropna().unique().tolist() if c])
        industries = ["(All)"] + sorted([i for i in self.meta["Industry"].dropna().unique().tolist() if i])
        indices = ["(All)"] + sorted([idx for idx in self.meta["Index"].dropna().unique().tolist() if idx])

        self.country_filter = Dropdown(options=countries, value="(All)", description="Country:")
        self.industry_filter = Dropdown(options=industries, value="(All)", description="Industry:")
        self.index_filter = Dropdown(options=indices, value="(All)", description="Index:")
        self.symbol_filter = SelectMultiple(
            options=sorted(self.meta["YahooSymbol"].unique().tolist()),
            value=(),
            description="Symbols:",
            layout=Layout(width="50%", height="150px")
        )

        self.btn_load = Button(description="Load Filtered Stocks", button_style="success")
        self.btn_load.on_click(self.on_load)

        # Sort controls - updated with new columns
        sortables = ["Company Name","Index","Country","Ticker","Industry","Name",
                     "Price","Shares (notional)","Position Value","RSI","SMA20","SMA50","Recommendation","Weight %","Yahoo Symbol",
                     "Monthly %","3M %","6M %","YTD %","1Y %","5Y %"]
        self.sort_col = Dropdown(options=sortables, value="Position Value", description="Sort by:")
        self.sort_order = ToggleButtons(options=[("Descending", False), ("Ascending", True)], value=False)
        self.btn_sort = Button(description="Apply Sort", button_style="primary")
        self.btn_show = Button(description="Show Full DataFrame", button_style="info")

        # Export controls
        self.csv_name = Text(value="portfolio.csv", description="CSV file:")
        self.json_name = Text(value="portfolio.json", description="JSON file:")
        self.btn_csv = Button(description="Export CSV", button_style="info")
        self.btn_json = Button(description="Export JSON", button_style="info")

        self.btn_sort.on_click(self.on_sort)
        self.btn_show.on_click(self.on_show)
        self.btn_csv.on_click(self.on_csv)
        self.btn_json.on_click(self.on_json)

    def filtered_symbols(self):
        df = self.meta.copy()
        if self.country_filter.value != "(All)":
            df = df[df["Country"] == self.country_filter.value]
        if self.industry_filter.value != "(All)":
            df = df[df["Industry"] == self.industry_filter.value]
        if self.index_filter.value != "(All)":
            df = df[df["Index"] == self.index_filter.value]
        selected = set(self.symbol_filter.value)
        if selected:
            df = df[df["YahooSymbol"].isin(selected)]
        return df["YahooSymbol"].unique().tolist()

    def on_load(self, b):
        with self.output:
            clear_output()
            syms = self.filtered_symbols()
            if not syms:
                print("No symbols match the filters. Please adjust your selection.")
                return
            print(f"Loading {len(syms)} symbols...")
            rows = assemble_current_snapshot(self.port, syms)
            if not rows:
                print("No data available for selected symbols.")
                return
            self.df = pd.DataFrame(rows).sort_values("Position Value", ascending=False)
            self.df["Weight %"] = self.df["Position Value"] / self.df["Position Value"].sum() * 100
            print(f"Loaded {len(self.df)} stocks. Total Notional Value: {self.currency} {self.df['Position Value'].sum():,.2f}")
            display(self.df.head(20))

    def on_sort(self, b):
        with self.output:
            clear_output()
            if self.df is None:
                print("Please load data first.")
                return
            self.df = sort_dataframe(self.df, self.sort_col.value, self.sort_order.value)
            display(self.df.head(20))

    def on_show(self, b):
        with self.output:
            clear_output()
            if self.df is None:
                print("Please load data first.")
                return
            display(self.df)

    def on_csv(self, b):
        with self.output:
            clear_output()
            if self.df is None:
                print("Please load data first.")
                return
            fn = self.csv_name.value
            self.df.to_csv(fn, index=False)
            print(f"Exported {fn}")

    def on_json(self, b):
        with self.output:
            clear_output()
            if self.df is None:
                print("Please load data first.")
                return
            fn = self.json_name.value
            self.df.to_json(fn, orient="records", indent=2)
            print(f"Exported {fn}")

    def display(self):
        ui = VBox([
            widgets.HTML("<h3>Filter & Load Stocks (with Pre-calculated Metrics)</h3>"),
            HBox([self.country_filter, self.industry_filter, self.index_filter]),
            self.symbol_filter,
            self.btn_load,
            widgets.HTML("<h3>Sort & Export</h3>"),
            HBox([self.sort_col, self.sort_order, self.btn_sort, self.btn_show]),
            HBox([self.csv_name, self.btn_csv]),
            HBox([self.json_name, self.btn_json]),
            self.output
        ])
        display(ui)

# ================================================================
# Performance Panel (Lazy + Cached)
# ================================================================
PERIOD_OPTIONS = ["5Y", "YTD", "6M", "3M", "30D"]
DEFAULT_BENCHMARKS = ["SPY", "IXN"]

def _bench_return(bench_hist, bench_symbol):
    """Calculate benchmark return from downloaded history."""
    if isinstance(bench_hist.columns, pd.MultiIndex):
        if (bench_symbol, "Close") not in bench_hist.columns:
            return np.nan
        s = bench_hist[(bench_symbol, "Close")].dropna()
    else:
        s = bench_hist["Close"].dropna() if "Close" in bench_hist.columns else pd.Series(dtype=float)
    if s.empty or len(s) < 2:
        return np.nan
    start_px = s.iloc[0]
    end_px = s.iloc[-1]
    return float((end_px - start_px) / start_px) if start_px > 0 else np.nan

def compute_portfolio_returns_dynamic(symbols, hist_df, start, end, investment_per_stock=100.0):
    """
    For each stock, compute shares as $investment_per_stock / start_price,
    then compute end value and return.
    """
    records = []
    multi = isinstance(hist_df.columns, pd.MultiIndex)

    for yfs in symbols:
        if multi and (yfs, "Close") not in hist_df.columns:
            continue

        start_close = _pick_close_at(hist_df, yfs, start)
        end_close   = _pick_close_at(hist_df, yfs, end)
        if start_close is None or end_close is None or start_close <= 0:
            continue

        shares = investment_per_stock / start_close
        start_value = investment_per_stock
        end_value   = shares * end_close

        rec = {
            "Ticker": yfs,
            "Start Price": start_close,
            "End Price": end_close,
            "Shares": shares,
            "Start Value": start_value,
            "End Value": end_value,
            "Return (%)": (end_value - start_value) / start_value * 100 if start_value > 0 else np.nan,
            "P/L ($)": end_value - start_value,
        }
        records.append(rec)

    if not records:
        return pd.DataFrame(), np.nan, 0.0, 0.0

    tmp = pd.DataFrame(records)

    # Aggregate portfolio
    total_start = tmp["Start Value"].sum()
    total_end = tmp["End Value"].sum()
    overall_return = (total_end - total_start) / total_start if total_start > 0 else np.nan
    total_pl = total_end - total_start

    return tmp, overall_return, total_start, total_end

class FastPerformancePanel:
    def __init__(self, meta_df, port, benchmarks=None):
        self.meta = meta_df
        self.port = port
        self.period = Dropdown(options=PERIOD_OPTIONS, value="6M", description="Period:")

        # Filters
        countries = ["(All)"] + sorted([c for c in self.meta["Country"].dropna().unique().tolist() if c])
        industries = ["(All)"] + sorted([i for i in self.meta["Industry"].dropna().unique().tolist() if i])
        indices = ["(All)"] + sorted([idx for idx in self.meta["Index"].dropna().unique().tolist() if idx])

        self.country_filter = Dropdown(options=countries, value="(All)", description="Country:")
        self.industry_filter = Dropdown(options=industries, value="(All)", description="Industry:")
        self.index_filter = Dropdown(options=indices, value="(All)", description="Index:")
        self.symbol_filter = SelectMultiple(
            options=sorted(self.meta["YahooSymbol"].unique().tolist()),
            value=(),
            description="Symbols:",
            layout=Layout(width="50%", height="150px")
        )

        self.investment = FloatText(value=100.0, description="$ per stock:")
        self.benchmarks_input = Text(
            value=",".join(benchmarks if benchmarks else DEFAULT_BENCHMARKS),
            description="Benchmarks:"
        )
        self.btn = Button(description="Compute Performance", button_style="primary")
        self.output = Output()
        self.btn.on_click(self.on_compute)

    def filtered_symbols(self):
        df = self.meta.copy()
        if self.country_filter.value != "(All)":
            df = df[df["Country"] == self.country_filter.value]
        if self.industry_filter.value != "(All)":
            df = df[df["Industry"] == self.industry_filter.value]
        if self.index_filter.value != "(All)":
            df = df[df["Index"] == self.index_filter.value]
        selected = set(self.symbol_filter.value)
        if selected:
            df = df[df["YahooSymbol"].isin(selected)]
        return df["YahooSymbol"].unique().tolist()

    def on_compute(self, _):
        with self.output:
            clear_output()
            syms = self.filtered_symbols()
            if not syms:
                print("No symbols match the filters/selection.")
                return

            start, end = _period_start_end(self.period.value)
            inv = float(self.investment.value)

            print(f"Fetching data for {len(syms)} symbols from {start.date()} to {end.date()}...")
            hist = _download_hist_for_period(syms, start, end)

            result, overall, total_start, total_end = compute_portfolio_returns_dynamic(
                syms, hist, start, end, investment_per_stock=inv
            )

            if result.empty:
                print("No data available after filtering.")
                return

            print(f"\nPeriod: {self.period.value} | Investment: ${inv:.2f} per stock")
            print(f"Total Invested: ${total_start:,.2f} | Total End Value: ${total_end:,.2f}")
            print(f"Aggregate Portfolio Return: {overall*100:.2f}% | P/L: ${total_end - total_start:,.2f}\n")

            # Benchmarks
            bench_list = [s.strip() for s in self.benchmarks_input.value.split(",") if s.strip()]
            if bench_list:
                print("Benchmarks:")
                bench_hist = _download_hist_for_period(bench_list, start, end)
                for b in bench_list:
                    br = _bench_return(bench_hist, b)
                    if pd.notna(br):
                        bench_end = inv * (1 + br)
                        bench_pl = bench_end - inv
                        print(f"  {b}: {br*100:.2f}% | ${inv:.2f} → ${bench_end:.2f} (P/L: ${bench_pl:.2f})")
                print("")

            display(result.sort_values("Return (%)", ascending=False).reset_index(drop=True))

    def display(self):
        box = VBox([
            widgets.HTML("<h3>Performance Analysis (Filtered)</h3>"),
            HBox([self.period, self.investment]),
            HBox([self.country_filter, self.industry_filter, self.index_filter]),
            self.symbol_filter,
            self.benchmarks_input,
            self.btn,
            self.output
        ])
        display(box)

# ================================================================
# Value Investing: Fundamentals + Intrinsic Value (Conservative DCF)
# ================================================================
def _safe_get(dct, key, default=None):
    try:
        return dct.get(key, default)
    except Exception:
        return default

def fetch_fundamentals(yf_symbol):
    if yf_symbol in FUNDAMENTAL_CACHE:
        return FUNDAMENTAL_CACHE[yf_symbol]

    t = yf.Ticker(yf_symbol)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    shares_out = _safe_get(info, "sharesOutstanding")
    long_name = _safe_get(info, "longName") or _safe_get(info, "shortName")
    industry = _safe_get(info, "industry")
    fcf_series = None
    try:
        cf = t.cashflow
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            if "Free Cash Flow" in cf.index:
                fcf_series = cf.loc["Free Cash Flow"].dropna().astype(float)
            else:
                cfo = cf.loc["Total Cash From Operating Activities"].dropna().astype(float) if "Total Cash From Operating Activities" in cf.index else None
                capex = cf.loc["Capital Expenditures"].dropna().astype(float) if "Capital Expenditures" in cf.index else None
                if cfo is not None and capex is not None:
                    fcf_series = (cfo + capex).dropna()
    except Exception:
        pass

    result = {
        "info": info,
        "shares_out": shares_out,
        "long_name": long_name,
        "industry": industry,
        "fcf_series": fcf_series
    }
    FUNDAMENTAL_CACHE[yf_symbol] = result
    return result

def conservative_dcf_intrinsic(fcf_series, shares_out, base_growth=0.03, years=10, discount_rate=0.10, terminal_growth=0.02):
    if fcf_series is None or len(fcf_series) == 0 or shares_out is None or shares_out <= 0:
        return None, None
    vals = pd.Series(fcf_series).dropna().astype(float).values
    start_fcf = float(np.mean(vals[:3])) if len(vals) >= 3 else float(vals[0])
    if start_fcf <= 0:
        return None, None
    fcf_list = [start_fcf * ((1 + base_growth) ** t) for t in range(1, years + 1)]
    discounts = [(1 + discount_rate) ** t for t in range(1, years + 1)]
    pv_fcf = sum(f / d for f, d in zip(fcf_list, discounts))
    tv = fcf_list[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth) if discount_rate > terminal_growth else 0.0
    pv_tv = tv / ((1 + discount_rate)
