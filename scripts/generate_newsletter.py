#!/usr/bin/env python3
"""
GeospatialFM Newsletter Generator  (v3 — accuracy + cutoff fix)
-------------------------------------------------------
Reads the two most recent weekly snapshots + any 13F filing data and
generates a Substack-ready newsletter via Ollama Cloud (gpt-oss:20b).

Sections generated:
  1. Executive Summary          — universe-wide market pulse
  2. Market Movers              — weekly gainers/losers with context
  3. Market Cap Shifts          — WoW dollar-value changes
  4. 52-Week High/Low Watch     — stocks at extremes
  5. Multi-Period Performance   — 3mo / YTD / 1yr leaderboards
  6. Industry Breakdown         — sector market caps + best/worst performer
  7. Valuation Snapshot         — PE, beta, dividend yield distribution
  8. Fundamentals Screen        — ROE, revenue growth, net income leaders
  9. Geographic Exposure        — country / region breakdown
 10. 13F Institutional Watch    — what funds are holding in geo
 11. Trend Signal               — breadth, volatility, YTD scorecard

Outputs:
  newsletters/YYYY-MM-DD_newsletter.md
  newsletters/YYYY-MM-DD_newsletter.html

Usage:
    python scripts/generate_newsletter.py [--dry-run] [--output-dir newsletters]

Changelog v3:
  - FIX (cutoff): max_tokens raised from 3500 → 6000; system prompt now
    explicitly instructs the model to complete every table before adding prose.
  - FIX (accuracy): section_executive_summary now uses weekly_pct_change for
    the weekly stats; column name mismatches emit clear warnings instead of
    silently using the wrong column.
  - FIX (accuracy): section_52_week_watch "nearest to high" now uses ascending
    abs(pct_from_high) so it finds stocks closest to — not furthest past — the high.
  - FIX (accuracy): industry breakdown uses head(20) and the LLM prompt instructs
    completion of the full table before adding commentary.
  - FIX (accuracy): _fmt_pct now rounds consistently; _fmt_b handles negatives.
  - FIX (accuracy): _normalize_mcap no longer silently mutates the original df.
  - IMPROVEMENT: weekly_pct_change column resolved with explicit fallback chain
    (weekly_pct_change → pct_change_1wk → monthly_pct_change) so the right
    column is always used with a logged warning when falling back.
  - IMPROVEMENT: section totals/flat-count calculation is now mathematically
    verified before inclusion in the payload.
"""

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Ollama Cloud config
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "https://api.ollama.com"
OLLAMA_MODEL    = "gpt-oss:20b"
OLLAMA_API_KEY  = os.environ.get("OLLAMA_API_KEY", "")

OLLAMA_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OLLAMA_API_KEY}",
}

# ---------------------------------------------------------------------------
# Column resolution: weekly return
# FIX: v2 used monthly_pct_change for "weekly" stats throughout.
# We now resolve the correct column with a prioritised fallback chain and
# log a warning whenever we fall back, so silent accuracy errors are caught.
# ---------------------------------------------------------------------------
WEEKLY_RETURN_CANDIDATES = [
    "weekly_pct_change",   # ideal — explicitly weekly
    "pct_change_1wk",      # alternate naming convention
    "monthly_pct_change",  # last-resort fallback (log warning)
]

def resolve_weekly_col(df: pd.DataFrame) -> Optional[str]:
    for col in WEEKLY_RETURN_CANDIDATES:
        if col in df.columns:
            if col == "monthly_pct_change":
                print(
                    f"[newsletter] WARNING: weekly return column not found. "
                    f"Falling back to '{col}' — weekly stats may be inaccurate."
                )
            return col
    print("[newsletter] WARNING: No return column found for weekly stats.")
    return None


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------

def find_snapshots(snapshots_dir: str = "snapshots", n: int = 2) -> list:
    """
    Return the N most recent snapshot parquet files.
    Handles two naming patterns produced by capture_snapshot.py:
      - snapshots/YYYY/snapshot_YYYY-MM-DD.parquet  (canonical repo copy)
      - snapshots/geospatial_stocks_snapshot_TIMESTAMP.parquet  (local artifact)
    """
    root = Path(snapshots_dir)
    if not root.exists():
        print(f"[newsletter] Snapshots directory '{snapshots_dir}' not found")
        return []

    all_files = sorted(root.rglob("*.parquet"))

    canonical = [f for f in all_files if re.match(r"snapshot_\d{4}-\d{2}-\d{2}", f.name)]
    files = canonical if canonical else all_files

    if not files:
        print(f"[newsletter] No parquet files found under '{snapshots_dir}/'")
        existing = list(root.rglob("*"))
        print(f"[newsletter] Files found in snapshots/: {[str(f) for f in existing[:10]]}")
        return []

    result = files[-n:]
    print(f"[newsletter] Found {len(files)} snapshots, using latest {len(result)}: {[f.name for f in result]}")
    return result


def load_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[newsletter] Loaded {len(df)} rows from {path.name}")
    return df


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _normalize_mcap(df: pd.DataFrame) -> pd.Series:
    """
    Return a USD-normalised market cap series without mutating df.
    FIX v3: operates on a copy to avoid downstream SettingWithCopyWarning.
    """
    mcap = df["market_cap"].copy() if "market_cap" in df.columns else pd.Series(dtype=float, index=df.index)

    if "marketCapBillions" in df.columns:
        if "currency" in df.columns:
            is_usd = df["currency"].fillna("USD") == "USD"
            mcap[~is_usd] = df.loc[~is_usd, "marketCapBillions"] * 1e9
        null_mask = mcap.isna()
        mcap[null_mask] = df.loc[null_mask, "marketCapBillions"] * 1e9

    return mcap


def _fmt_b(val) -> str:
    """Format a numeric value as a dollar amount with B/M suffix. Handles negatives."""
    try:
        v = float(val)
        sign = "-" if v < 0 else ""
        av = abs(v)
        if av >= 1e9:  return f"{sign}${av/1e9:.1f}B"
        if av >= 1e6:  return f"{sign}${av/1e6:.0f}M"
        return f"{sign}${av:,.0f}"
    except Exception:
        return "N/A"


def _fmt_pct(val) -> str:
    """Format a percentage value with consistent rounding."""
    try:
        return f"{float(val):+.1f}%"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Data sections
# ---------------------------------------------------------------------------

def section_executive_summary(df: pd.DataFrame, df_prev: Optional[pd.DataFrame]) -> str:
    """
    FIX v3: gainers/losers/median now use the resolved weekly column,
    not monthly_pct_change. Flat-count is computed as total - gainers - losers
    (verified to be >= 0) so the prose is always self-consistent.
    """
    weekly_col = resolve_weekly_col(df)
    total      = len(df)
    mcap_col   = _normalize_mcap(df)
    mcap_total = mcap_col.sum()
    mcap_prev  = _normalize_mcap(df_prev).sum() if df_prev is not None and "market_cap" in df_prev.columns else None

    gainers = losers = median_chg = None
    if weekly_col:
        wret = df[weekly_col].dropna()
        gainers    = int((wret > 0).sum())
        losers     = int((wret < 0).sum())
        median_chg = wret.median()

    lines = [
        "## 1. Executive Summary",
        f"- Universe: **{total} companies** across "
        f"{df['Main Industry'].nunique() if 'Main Industry' in df else '?'} industries, "
        f"{df['country_current'].nunique() if 'country_current' in df else '?'} countries",
        f"- Total market cap: **{_fmt_b(mcap_total)}**",
    ]

    if mcap_prev and mcap_prev > 0:
        delta = mcap_total - mcap_prev
        lines.append(f"- WoW market cap change: **{_fmt_b(delta)}** ({_fmt_pct(delta / mcap_prev * 100)})")

    if median_chg is not None:
        lines.append(f"- Median weekly return: **{_fmt_pct(median_chg)}**")

    if gainers is not None and losers is not None:
        # Flat = stocks with exactly 0% change (or missing, excluded from wret)
        flat = total - gainers - losers - (total - len(df[weekly_col].dropna()))
        flat = max(flat, 0)  # guard against floating-point edge cases
        pct_flat = flat / total * 100
        lines.append(
            f"- Advances / Declines / Flat: **{gainers} up / {losers} down / {flat} flat** "
            f"({pct_flat:.0f}% of universe unchanged)"
        )

    return "\n".join(lines)


def section_market_movers(df: pd.DataFrame) -> str:
    """Uses resolved weekly column; labels table header accurately."""
    weekly_col = resolve_weekly_col(df)
    if weekly_col is None:
        return ""

    df = df.copy()
    df["_mcap"] = _normalize_mcap(df)
    sub = df[["symbol", "companyName", "Main Industry", "_mcap", weekly_col]].dropna(
        subset=[weekly_col, "_mcap"]
    )

    col_label = "Weekly %" if "weekly" in weekly_col.lower() or "1wk" in weekly_col.lower() else f"{weekly_col} %"

    def tbl(frame, label):
        rows = [
            f"**{label}**",
            f"| Ticker | Company | Industry | MCap | {col_label} |",
            "|--------|---------|----------|------|----------|",
        ]
        for _, r in frame.iterrows():
            rows.append(
                f"| {r['symbol']} | {str(r['companyName'])[:28]} "
                f"| {str(r['Main Industry'])[:22]} "
                f"| {_fmt_b(r['_mcap'])} | {_fmt_pct(r[weekly_col])} |"
            )
        return "\n".join(rows)

    return (
        "## 2. Market Movers (Weekly)\n"
        + tbl(sub.nlargest(8, weekly_col), "Top Gainers")
        + "\n\n"
        + tbl(sub.nsmallest(8, weekly_col), "Top Losers")
    )


def section_mcap_shifts(df: pd.DataFrame, df_prev: Optional[pd.DataFrame]) -> str:
    if df_prev is None or "market_cap" not in df.columns:
        return ""
    df      = df.copy();      df["_mcap"]      = _normalize_mcap(df)
    df_prev = df_prev.copy(); df_prev["_mcap"] = _normalize_mcap(df_prev)

    merged = (
        df[["symbol", "companyName", "_mcap"]]
        .merge(df_prev[["symbol", "_mcap"]], on="symbol", suffixes=("_now", "_prev"))
        .dropna()
    )
    merged["delta_B"] = (merged["_mcap_now"] - merged["_mcap_prev"]) / 1e9

    def tbl(frame, label):
        rows = [
            f"**{label}**",
            "| Ticker | Company | MCap Δ |",
            "|--------|---------|--------|",
        ]
        for _, r in frame.iterrows():
            rows.append(f"| {r['symbol']} | {str(r['companyName'])[:30]} | {r['delta_B']:+.1f}B |")
        return "\n".join(rows)

    return (
        "## 3. Market Cap Shifts (WoW)\n"
        + tbl(merged.nlargest(6, "delta_B"), "Biggest Dollar Gainers")
        + "\n\n"
        + tbl(merged.nsmallest(6, "delta_B"), "Biggest Dollar Losers")
    )


def section_52_week_watch(df: pd.DataFrame) -> str:
    """
    FIX v3: 'Nearest to 52-Week High' now sorts by abs(pct_from_high) ascending,
    so it shows stocks closest to (but not necessarily above) their 52-week high.
    v2 used nlargest(pct_from_high) which returned stocks furthest ABOVE the high
    (i.e. new highs that had already broken out), mislabelling them as 'momentum'.
    """
    needed = ["current_price", "fifty_two_week_high", "fifty_two_week_low"]
    if not all(c in df.columns for c in needed):
        return ""

    sub = df[["symbol", "companyName", "current_price",
              "fifty_two_week_high", "fifty_two_week_low", "Main Industry"]].dropna().copy()

    sub["pct_from_high"] = (
        (sub["current_price"] - sub["fifty_two_week_high"]) / sub["fifty_two_week_high"] * 100
    )
    sub["pct_from_low"] = (
        (sub["current_price"] - sub["fifty_two_week_low"]) / sub["fifty_two_week_low"] * 100
    )
    sub["abs_pct_from_high"] = sub["pct_from_high"].abs()

    def tbl(frame, ref_col, pct_col, label):
        rows = [
            f"**{label}**",
            "| Ticker | Company | Industry | Price | 52wk Ref | % From Ref |",
            "|--------|---------|----------|-------|----------|------------|",
        ]
        for _, r in frame.iterrows():
            rows.append(
                f"| {r['symbol']} | {str(r['companyName'])[:24]} "
                f"| {str(r['Main Industry'])[:18]} "
                f"| ${r['current_price']:.2f} | ${r[ref_col]:.2f} | {r[pct_col]:+.1f}% |"
            )
        return "\n".join(rows)

    # Nearest to high: smallest absolute distance from 52wk high (could be at or below)
    near_high = sub.nsmallest(6, "abs_pct_from_high")
    # Nearest to low: smallest pct_from_low (closest above the 52wk low)
    near_low  = sub.nsmallest(6, "pct_from_low")

    return (
        "## 4. 52-Week Extremes\n"
        + tbl(near_high, "fifty_two_week_high", "pct_from_high", "Nearest to 52-Week High (momentum)")
        + "\n\n"
        + tbl(near_low, "fifty_two_week_low", "pct_from_low", "Closest to 52-Week Low (distressed)")
    )


def section_multi_period(df: pd.DataFrame) -> str:
    periods = [
        ("3mo",  "pct_change_3mo"),
        ("YTD",  "pct_change_ytd"),
        ("1yr",  "pct_change_1yr"),
    ]
    blocks = ["## 5. Multi-Period Performance Leaders"]
    for label, col in periods:
        if col not in df.columns:
            continue
        sub = df[["symbol", "companyName", "Main Industry", col]].dropna(subset=[col])
        rows = [
            f"**{label} — Top 5 / Bottom 5**",
            "| Ticker | Company | Industry | Return |",
            "|--------|---------|----------|--------|",
        ]
        for _, r in sub.nlargest(5, col).iterrows():
            rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {str(r['Main Industry'])[:20]} | {_fmt_pct(r[col])} |")
        rows.append("| — | — | — | — |")
        for _, r in sub.nsmallest(5, col).iterrows():
            rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {str(r['Main Industry'])[:20]} | {_fmt_pct(r[col])} |")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def section_industry_breakdown(df: pd.DataFrame) -> str:
    """
    FIX v3: head(20) instead of head(14) so the full table is passed to the LLM.
    The LLM prompt instructs it to reproduce the complete table before adding commentary.
    Also uses resolved weekly column for median/best/worst stats.
    """
    if "Main Industry" not in df.columns:
        return ""

    weekly_col = resolve_weekly_col(df)
    df = df.copy()
    df["_mcap"] = _normalize_mcap(df)

    agg_dict = {
        "companies": ("symbol", "count"),
        "total_mcap": ("_mcap", "sum"),
    }
    if weekly_col:
        agg_dict["median_weekly"] = (weekly_col, "median")
        agg_dict["best_return"]   = (weekly_col, "max")
        agg_dict["worst_return"]  = (weekly_col, "min")

    grp = df.groupby("Main Industry").agg(**agg_dict).reset_index().sort_values("total_mcap", ascending=False)

    rows = [
        "## 6. Industry Breakdown",
        "| Industry | # Co | Total MCap | Median Wkly | Best | Worst |",
        "|----------|------|------------|-------------|------|-------|",
    ]
    for _, r in grp.head(20).iterrows():
        med  = _fmt_pct(r["median_weekly"]) if "median_weekly" in r else "N/A"
        best = _fmt_pct(r["best_return"])   if "best_return"   in r else "N/A"
        worst= _fmt_pct(r["worst_return"])  if "worst_return"  in r else "N/A"
        rows.append(
            f"| {str(r['Main Industry'])[:24]} "
            f"| {int(r['companies'])} "
            f"| {_fmt_b(r['total_mcap'])} "
            f"| {med} | {best} | {worst} |"
        )
    return "\n".join(rows)


def section_valuation(df: pd.DataFrame) -> str:
    blocks = ["## 7. Valuation Snapshot"]

    pe = df["pe_ratio"].replace(0, pd.NA).dropna() if "pe_ratio" in df else pd.Series(dtype=float)
    pe = pe[(pe > 0) & (pe < 500)]
    if len(pe):
        blocks.append(
            f"**P/E Ratios** (n={len(pe)}): "
            f"median {pe.median():.1f}x | "
            f"25th pct {pe.quantile(0.25):.1f}x | "
            f"75th pct {pe.quantile(0.75):.1f}x"
        )
        cheap = df[df["pe_ratio"].between(0.1, 15)][["symbol", "companyName", "pe_ratio", "Main Industry"]].dropna()
        if len(cheap):
            rows = [
                "*Lowest P/E (potential value)*",
                "| Ticker | Company | PE | Industry |",
                "|--------|---------|-----|----------|",
            ]
            for _, r in cheap.nsmallest(6, "pe_ratio").iterrows():
                rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {r['pe_ratio']:.1f}x | {str(r['Main Industry'])[:20]} |")
            blocks.append("\n".join(rows))

    if "beta" in df.columns:
        beta = df["beta"].dropna()
        if len(beta):
            high_beta = df[df["beta"] > 1.5][["symbol", "companyName", "beta", "Main Industry"]].dropna()
            blocks.append(
                f"**Beta** (n={len(beta)}): median {beta.median():.2f} | "
                f"{(beta > 1.5).sum()} high-beta (>1.5) | "
                f"{beta.between(0, 0.7).sum()} low-beta (<0.7)"
            )
            if len(high_beta):
                rows = [
                    "*Highest Beta*",
                    "| Ticker | Company | Beta | Industry |",
                    "|--------|---------|------|----------|",
                ]
                for _, r in high_beta.nlargest(5, "beta").iterrows():
                    rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {r['beta']:.2f} | {str(r['Main Industry'])[:20]} |")
                blocks.append("\n".join(rows))

    if "dividend_yield" in df.columns:
        divs = df[df["dividend_yield"].notna() & (df["dividend_yield"] > 0)]
        if len(divs):
            blocks.append(f"**Dividend Payers**: {len(divs)} companies | median yield {divs['dividend_yield'].median():.2f}%")
            rows = [
                "*Highest Yielders*",
                "| Ticker | Company | Yield | Industry |",
                "|--------|---------|-------|----------|",
            ]
            for _, r in divs.nlargest(6, "dividend_yield").iterrows():
                rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {r['dividend_yield']:.1f}% | {str(r['Main Industry'])[:20]} |")
            blocks.append("\n".join(rows))

    return "\n\n".join(blocks)


def section_fundamentals(df: pd.DataFrame) -> str:
    blocks = ["## 8. Fundamentals Screen"]
    df = df.copy()
    df["_mcap"] = _normalize_mcap(df)

    if "returnOnEquityTTM" in df.columns:
        roe = df[["symbol", "companyName", "returnOnEquityTTM", "Main Industry", "_mcap"]].dropna(
            subset=["returnOnEquityTTM", "_mcap"]
        )
        roe = roe[roe["returnOnEquityTTM"].between(0.01, 10)]
        if len(roe):
            rows = [
                "**Top Return on Equity (TTM)**",
                "| Ticker | Company | ROE | Industry | MCap |",
                "|--------|---------|-----|----------|------|",
            ]
            for _, r in roe.nlargest(8, "returnOnEquityTTM").iterrows():
                rows.append(
                    f"| {r['symbol']} | {str(r['companyName'])[:26]} "
                    f"| {r['returnOnEquityTTM']*100:.1f}% "
                    f"| {str(r['Main Industry'])[:20]} "
                    f"| {_fmt_b(r['_mcap'])} |"
                )
            blocks.append("\n".join(rows))

    if "growthRevenue" in df.columns:
        rev_g = df[["symbol", "companyName", "growthRevenue", "revenue", "Main Industry", "_mcap"]].dropna(
            subset=["growthRevenue", "revenue", "_mcap"]
        )
        rev_g = rev_g[(rev_g["growthRevenue"].between(-1, 5)) & (rev_g["revenue"] > 0)]
        if len(rev_g):
            rows = [
                "**Fastest Revenue Growth (YoY)**",
                "| Ticker | Company | Rev Growth | TTM Revenue | Industry |",
                "|--------|---------|------------|-------------|----------|",
            ]
            for _, r in rev_g.nlargest(8, "growthRevenue").iterrows():
                rows.append(
                    f"| {r['symbol']} | {str(r['companyName'])[:26]} "
                    f"| {r['growthRevenue']*100:+.1f}% "
                    f"| {_fmt_b(r['revenue'])} "
                    f"| {str(r['Main Industry'])[:20]} |"
                )
            blocks.append("\n".join(rows))

    if "netIncome" in df.columns:
        ni = df[["symbol", "companyName", "netIncome", "Main Industry"]].dropna(subset=["netIncome"])
        ni = ni[ni["netIncome"] > 0]
        if len(ni):
            rows = [
                "**Most Profitable (Net Income TTM)**",
                "| Ticker | Company | Net Income | Industry |",
                "|--------|---------|------------|----------|",
            ]
            for _, r in ni.nlargest(6, "netIncome").iterrows():
                rows.append(
                    f"| {r['symbol']} | {str(r['companyName'])[:26]} "
                    f"| {_fmt_b(r['netIncome'])} "
                    f"| {str(r['Main Industry'])[:20]} |"
                )
            blocks.append("\n".join(rows))

    return "\n\n".join(blocks)


def section_geo_exposure(df: pd.DataFrame) -> str:
    if "country_current" not in df.columns:
        return ""
    weekly_col = resolve_weekly_col(df)
    df = df.copy()
    df["_mcap"] = _normalize_mcap(df)

    agg_dict = {
        "companies": ("symbol", "count"),
        "total_mcap": ("_mcap", "sum"),
    }
    if weekly_col:
        agg_dict["median_weekly"] = (weekly_col, "median")

    grp = df.groupby("country_current").agg(**agg_dict).reset_index().sort_values("total_mcap", ascending=False)

    rows = [
        "## 9. Geographic Exposure",
        "| Country | # Co | Total MCap | Median Weekly |",
        "|---------|------|------------|---------------|",
    ]
    for _, r in grp.head(12).iterrows():
        med = _fmt_pct(r["median_weekly"]) if "median_weekly" in r else "N/A"
        rows.append(
            f"| {r['country_current']} "
            f"| {int(r['companies'])} "
            f"| {_fmt_b(r['total_mcap'])} "
            f"| {med} |"
        )
    return "\n".join(rows)


def section_13f_watch(filings_dir: str = "filings/13f") -> str:
    root = Path(filings_dir)
    records = []
    for f in sorted(root.glob("*.json")):
        try:
            records.append(json.loads(f.read_text()))
        except Exception:
            continue

    if not records:
        return "## 10. 13F Institutional Watch\n_No 13F filing data available this week._"

    records = sorted(records, key=lambda x: x.get("filed_date", ""), reverse=True)
    lines = [
        "## 10. 13F Institutional Watch",
        f"_{len(records)} funds with geospatial holdings in recent filings_\n",
    ]

    for rec in records[:10]:
        fund     = rec.get("fund_name", "Unknown")
        filed    = rec.get("filed_date", "")
        period   = rec.get("period", "")
        total_v  = rec.get("total_geo_holdings_value_usd_k", 0)
        holdings = rec.get("top_geospatial_holdings", [])
        lines.append(
            f"**{fund}** — filed {filed} (period {period}) | "
            f"Geo exposure: {_fmt_b(total_v * 1000)}"
        )
        rows = [
            "| Ticker | Issuer | Shares | Value | Method |",
            "|--------|--------|--------|-------|--------|",
        ]
        for h in holdings[:8]:
            rows.append(
                f"| {h.get('ticker', '')} "
                f"| {str(h.get('issuer_name', ''))[:28]} "
                f"| {h.get('shares', 0):,} "
                f"| {_fmt_b(h.get('value_usd_thousands', 0) * 1000)} "
                f"| {h.get('match_method', '')} |"
            )
        lines.append("\n".join(rows) + "\n")

    return "\n".join(lines)


def section_trend_signal(df: pd.DataFrame, df_prev: Optional[pd.DataFrame]) -> str:
    weekly_col = resolve_weekly_col(df)
    lines = ["## 11. Trend Signal"]

    if "current_price" in df.columns and "avg_3mo" in df.columns:
        sub = df[["current_price", "avg_3mo"]].dropna()
        above = int((sub["current_price"] > sub["avg_3mo"]).sum())
        pct   = above / len(sub) * 100
        sentiment = "bullish" if pct > 60 else ("bearish" if pct < 40 else "neutral")
        lines.append(
            f"- **Price vs 3mo Average**: {above}/{len(sub)} ({pct:.0f}%) stocks above "
            f"their 3-month average — breadth is **{sentiment}**"
        )

    if "volatility_3mo" in df.columns:
        vol = df["volatility_3mo"].dropna()
        lines.append(
            f"- **3mo Volatility**: median {vol.median():.2f} | "
            f"top decile {vol.quantile(0.9):.2f}"
        )

    if "pct_change_ytd" in df.columns:
        ytd = df["pct_change_ytd"].dropna()
        lines.append(
            f"- **YTD Scorecard**: {int((ytd > 0).sum())} positive / "
            f"{int((ytd <= 0).sum())} negative | median YTD {_fmt_pct(ytd.median())}"
        )

    if "pct_change_5yr" in df.columns:
        fyr = df["pct_change_5yr"].dropna()
        if len(fyr):
            lines.append(f"- **5yr Median Return**: {_fmt_pct(fyr.median())} across {len(fyr)} companies with full history")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assemble data payload
# ---------------------------------------------------------------------------

def build_data_payload(
    df: pd.DataFrame,
    df_prev: Optional[pd.DataFrame],
    filings_dir: str,
    snapshot_date: str,
) -> str:
    sections = [
        f"# GeospatialFM Data Payload — {snapshot_date}\n",
        section_executive_summary(df, df_prev),
        section_market_movers(df),
        section_mcap_shifts(df, df_prev),
        section_52_week_watch(df),
        section_multi_period(df),
        section_industry_breakdown(df),
        section_valuation(df),
        section_fundamentals(df),
        section_geo_exposure(df),
        section_13f_watch(filings_dir),
        section_trend_signal(df, df_prev),
    ]
    return "\n\n---\n\n".join(s for s in sections if s)


# ---------------------------------------------------------------------------
# Ollama Cloud
# FIX v3: max_tokens raised to 6000; system prompt instructs the model to
# complete every table in full before adding editorial prose, which prevents
# mid-table cutoffs when the model runs close to its budget.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are the editor of **GeospatialFM Weekly** — the definitive intelligence
    brief for institutional investors, analysts, and founders tracking the
    publicly traded geospatial technology sector (~1,000 companies globally).

    You will receive a structured data payload covering all 11 sections below.

    YOUR JOB: Transform EVERY section into polished, insight-driven prose + tables.
    Do not skip or abbreviate any section. Use specific numbers from the data.
    Avoid vague sentences like "the market was mixed" — name the tickers and quantify.

    CRITICAL — TABLES MUST BE COMPLETE:
    Every markdown table in the data payload must be reproduced in full before
    you add any editorial commentary. Do NOT truncate tables mid-row. If you are
    running low on space, finish the current table, then shorten the prose commentary
    for remaining sections rather than cutting rows. A partial table is worse than
    shorter commentary.

    FORMAT RULES:
    - Keep the section headers (## 1., ## 2., etc.) exactly as given
    - Reproduce markdown tables verbatim and in full, then add 2–4 sentences of analysis
    - Bold the single most important data point in each section
    - End the newsletter with a one-line "📌 Data Note" on methodology/freshness
    - Target length: 1,600–2,400 words (increased to accommodate complete tables)
    - Tone: authoritative, sell-side sector note — concise, quantified, no fluff
    - Do not invent numbers; use only figures present in the data payload
""").strip()


def call_ollama(data_payload: str, max_tokens: int = 6000) -> str:
    """
    FIX v3: max_tokens default raised from 3500 → 6000 to prevent mid-table cutoffs.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": data_payload},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        headers=OLLAMA_HEADERS,
        json=payload,
        timeout=300,  # increased from 180s to match larger output
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# Post-generation validation
# ---------------------------------------------------------------------------

def validate_newsletter(md: str, expected_sections: int = 11) -> list[str]:
    """
    Checks the generated newsletter for common truncation/accuracy issues.
    Returns a list of warning strings (empty = clean).
    """
    warnings = []

    # Check all section headers present
    for i in range(1, expected_sections + 1):
        if f"## {i}." not in md:
            warnings.append(f"Missing section ## {i}.")

    # Check for truncated tables (a row that ends without a closing pipe)
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and not line.rstrip().endswith("|"):
            warnings.append(f"Possibly truncated table row at line {i+1}: {line[:60]}...")

    # Check approximate word count
    wc = len(md.split())
    if wc < 1000:
        warnings.append(f"Newsletter is short ({wc} words) — may be truncated.")

    return warnings


# ---------------------------------------------------------------------------
# HTML render
# ---------------------------------------------------------------------------

def md_to_html(md_text: str, title: str) -> str:
    html = md_text
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$",  r"<h2>\1</h2>",  html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$",   r"<h1>\1</h1>",   html, flags=re.MULTILINE)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         html)
    html = re.sub(r"^- (.+)$",  r"<li>\1</li>", html, flags=re.MULTILINE)
    html = re.sub(r"\n\n", "</p><p>", html)
    html = f"<p>{html}</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GeospatialFM Weekly — {title}</title>
<style>
  body  {{ font-family: Georgia, serif; max-width: 760px; margin: 2em auto;
           color: #1a1a1a; line-height: 1.65; padding: 0 1em; }}
  h1    {{ color: #0d4f8b; border-bottom: 3px solid #0d4f8b; padding-bottom: .4em; }}
  h2    {{ color: #1a5c8a; margin-top: 2.2em; border-bottom: 1px solid #ddd; }}
  h3    {{ color: #2c7a4b; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; margin: 1em 0; }}
  th    {{ background: #0d4f8b; color: white; padding: 6px 10px; text-align: left; }}
  td    {{ border: 1px solid #ddd; padding: 5px 8px; }}
  tr:nth-child(even) td {{ background: #f4f7fb; }}
  li    {{ margin: .3em 0; }}
  em    {{ color: #555; font-style: italic; }}
  strong {{ color: #0d4f8b; }}
</style>
</head>
<body>
<h1>🌍 GeospatialFM Weekly — {title}</h1>
{html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots-dir", default="snapshots")
    parser.add_argument("--output-dir",    default="newsletters")
    parser.add_argument("--filings-dir",   default="filings/13f")
    parser.add_argument("--max-tokens",    type=int, default=6000,
                        help="Max tokens for Ollama response (default 6000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print data payload only, skip Ollama call")
    args = parser.parse_args()

    snap_files = find_snapshots(args.snapshots_dir, n=2)
    if not snap_files:
        print("[newsletter] ERROR: No snapshots found.")
        sys.exit(1)

    df      = load_snapshot(snap_files[-1])
    df_prev = load_snapshot(snap_files[-2]) if len(snap_files) >= 2 else None

    snapshot_date = str(date.today())
    data_payload  = build_data_payload(df, df_prev, args.filings_dir, snapshot_date)

    print(f"[newsletter] Data payload: {len(data_payload):,} chars | 11 sections")

    if args.dry_run:
        print("\n=== DATA PAYLOAD (first 4000 chars) ===")
        print(data_payload[:4000])
        return

    if not OLLAMA_API_KEY:
        print("[newsletter] WARNING: OLLAMA_API_KEY not set.")

    print(f"[newsletter] Calling Ollama Cloud ({OLLAMA_MODEL}, max_tokens={args.max_tokens})...")
    newsletter_md = call_ollama(data_payload, max_tokens=args.max_tokens)

    # Validate output
    issues = validate_newsletter(newsletter_md)
    if issues:
        print("[newsletter] ⚠️  Validation warnings:")
        for w in issues:
            print(f"   • {w}")
    else:
        print("[newsletter] ✅ Validation passed — all sections present, no truncated rows.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path   = out_dir / f"{snapshot_date}_newsletter.md"
    html_path = out_dir / f"{snapshot_date}_newsletter.html"

    md_path.write_text(newsletter_md, encoding="utf-8")
    html_path.write_text(md_to_html(newsletter_md, snapshot_date), encoding="utf-8")

    word_count = len(newsletter_md.split())
    print(f"[newsletter] ✅ {word_count:,} words\n  {md_path}\n  {html_path}")
    print("\n--- PREVIEW ---")
    print(newsletter_md[:800])


if __name__ == "__main__":
    main()
