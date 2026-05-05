#!/usr/bin/env python3
"""
GeospatialFM Newsletter Generator  (v2 — rich edition)
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
# Snapshot loading
# ---------------------------------------------------------------------------

def find_snapshots(snapshots_dir: str = "snapshots", n: int = 2) -> list:
    root = Path(snapshots_dir)
    files = sorted(root.rglob("snapshot_*.parquet"))
    return files[-n:] if len(files) >= n else files


def load_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[newsletter] Loaded {len(df)} rows from {path.name}")
    return df


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_b(val) -> str:
    try:
        v = float(val)
        if v >= 1e9:  return f"${v/1e9:.1f}B"
        if v >= 1e6:  return f"${v/1e6:.0f}M"
        return f"${v:,.0f}"
    except Exception:
        return "N/A"


def _fmt_pct(val) -> str:
    try:   return f"{float(val):+.1f}%"
    except: return "N/A"


# ---------------------------------------------------------------------------
# Data sections
# ---------------------------------------------------------------------------

def section_executive_summary(df: pd.DataFrame, df_prev: Optional[pd.DataFrame]) -> str:
    total      = len(df)
    mcap_total = df["market_cap"].sum() if "market_cap" in df else 0
    mcap_prev  = df_prev["market_cap"].sum() if df_prev is not None and "market_cap" in df_prev else None

    gainers = int((df["monthly_pct_change"] > 0).sum()) if "monthly_pct_change" in df else "?"
    losers  = int((df["monthly_pct_change"] < 0).sum()) if "monthly_pct_change" in df else "?"
    median_chg = df["monthly_pct_change"].median() if "monthly_pct_change" in df else None

    lines = [
        "## 1. Executive Summary",
        f"- Universe: **{total} companies** across "
        f"{df['Main Industry'].nunique()} industries, "
        f"{df['country_current'].nunique() if 'country_current' in df else '?'} countries",
        f"- Total market cap: **{_fmt_b(mcap_total)}**",
    ]
    if mcap_prev and mcap_prev > 0:
        delta = mcap_total - mcap_prev
        lines.append(f"- WoW market cap change: **{_fmt_b(delta)}** ({_fmt_pct(delta/mcap_prev*100)})")
    if median_chg is not None:
        lines.append(f"- Median weekly return: **{_fmt_pct(median_chg)}**")
    if isinstance(gainers, int):
        flat = total - gainers - losers
        lines.append(f"- Advances / Declines: **{gainers} up / {losers} down** ({flat} flat)")
    return "\n".join(lines)


def section_market_movers(df: pd.DataFrame) -> str:
    col = "monthly_pct_change"
    if col not in df.columns:
        return ""
    sub = df[["symbol","companyName","Main Industry","market_cap",col]].dropna(subset=[col])

    def tbl(frame, label):
        rows = [f"**{label}**",
                "| Ticker | Company | Industry | MCap | Weekly % |",
                "|--------|---------|----------|------|----------|"]
        for _, r in frame.iterrows():
            rows.append(
                f"| {r['symbol']} | {str(r['companyName'])[:28]} "
                f"| {str(r['Main Industry'])[:22]} "
                f"| {_fmt_b(r['market_cap'])} | {_fmt_pct(r[col])} |"
            )
        return "\n".join(rows)

    return (
        "## 2. Market Movers (Weekly)\n"
        + tbl(sub.nlargest(8, col), "Top Gainers")
        + "\n\n"
        + tbl(sub.nsmallest(8, col), "Top Losers")
    )


def section_mcap_shifts(df: pd.DataFrame, df_prev: Optional[pd.DataFrame]) -> str:
    if df_prev is None or "market_cap" not in df.columns:
        return ""
    merged = (
        df[["symbol","companyName","market_cap"]]
        .merge(df_prev[["symbol","market_cap"]], on="symbol", suffixes=("_now","_prev"))
        .dropna()
    )
    merged["delta_B"] = (merged["market_cap_now"] - merged["market_cap_prev"]) / 1e9

    def tbl(frame, label):
        rows = [f"**{label}**",
                "| Ticker | Company | MCap Δ |",
                "|--------|---------|--------|"]
        for _, r in frame.iterrows():
            rows.append(f"| {r['symbol']} | {str(r['companyName'])[:30]} | {r['delta_B']:+.1f}B |")
        return "\n".join(rows)

    return (
        "## 3. Market Cap Shifts (WoW)\n"
        + tbl(merged.nlargest(6,"delta_B"), "Biggest Dollar Gainers")
        + "\n\n"
        + tbl(merged.nsmallest(6,"delta_B"), "Biggest Dollar Losers")
    )


def section_52_week_watch(df: pd.DataFrame) -> str:
    needed = ["current_price","fifty_two_week_high","fifty_two_week_low"]
    if not all(c in df.columns for c in needed):
        return ""
    sub = df[["symbol","companyName","current_price",
              "fifty_two_week_high","fifty_two_week_low","Main Industry"]].dropna()
    sub = sub.copy()
    sub["pct_from_high"] = (sub["current_price"] - sub["fifty_two_week_high"]) / sub["fifty_two_week_high"] * 100
    sub["pct_from_low"]  = (sub["current_price"] - sub["fifty_two_week_low"])  / sub["fifty_two_week_low"]  * 100

    def tbl(frame, ref_col, pct_col, label):
        rows = [f"**{label}**",
                "| Ticker | Company | Industry | Price | 52wk Ref | % From Ref |",
                "|--------|---------|----------|-------|----------|------------|"]
        for _, r in frame.iterrows():
            rows.append(
                f"| {r['symbol']} | {str(r['companyName'])[:24]} "
                f"| {str(r['Main Industry'])[:18]} "
                f"| ${r['current_price']:.2f} | ${r[ref_col]:.2f} | {r[pct_col]:+.1f}% |"
            )
        return "\n".join(rows)

    return (
        "## 4. 52-Week Extremes\n"
        + tbl(sub.nlargest(6,"pct_from_high"), "fifty_two_week_high", "pct_from_high",
              "Nearest to 52-Week High (momentum)")
        + "\n\n"
        + tbl(sub.nlargest(6,"pct_from_low"), "fifty_two_week_low", "pct_from_low",
              "Furthest Above 52-Week Low (recovery)")
    )


def section_multi_period(df: pd.DataFrame) -> str:
    periods = [("3mo","pct_change_3mo"), ("YTD","pct_change_ytd"), ("1yr","pct_change_1yr")]
    blocks = ["## 5. Multi-Period Performance Leaders"]
    for label, col in periods:
        if col not in df.columns:
            continue
        sub = df[["symbol","companyName","Main Industry",col]].dropna(subset=[col])
        rows = [f"**{label} — Top 5 / Bottom 5**",
                "| Ticker | Company | Industry | Return |",
                "|--------|---------|----------|--------|"]
        for _, r in sub.nlargest(5,col).iterrows():
            rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {str(r['Main Industry'])[:20]} | {_fmt_pct(r[col])} |")
        rows.append("| — | — | — | — |")
        for _, r in sub.nsmallest(5,col).iterrows():
            rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {str(r['Main Industry'])[:20]} | {_fmt_pct(r[col])} |")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def section_industry_breakdown(df: pd.DataFrame) -> str:
    if "Main Industry" not in df.columns:
        return ""

    def safe_best(x):
        sub2 = df.loc[x.index].dropna(subset=["monthly_pct_change"])
        if len(sub2) == 0: return ""
        return sub2.nlargest(1,"monthly_pct_change")["symbol"].values[0]

    grp = df.groupby("Main Industry").agg(
        companies=("symbol","count"),
        total_mcap=("market_cap","sum"),
        median_weekly=("monthly_pct_change","median"),
        best_return=("monthly_pct_change","max"),
        worst_return=("monthly_pct_change","min"),
    ).reset_index().sort_values("total_mcap", ascending=False)

    rows = ["## 6. Industry Breakdown",
            "| Industry | # Co | Total MCap | Median Wkly | Best | Worst |",
            "|----------|------|------------|-------------|------|-------|"]
    for _, r in grp.head(14).iterrows():
        rows.append(
            f"| {str(r['Main Industry'])[:24]} "
            f"| {int(r['companies'])} "
            f"| {_fmt_b(r['total_mcap'])} "
            f"| {_fmt_pct(r['median_weekly'])} "
            f"| {_fmt_pct(r['best_return'])} "
            f"| {_fmt_pct(r['worst_return'])} |"
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
        cheap = df[df["pe_ratio"].between(0.1,15)][["symbol","companyName","pe_ratio","Main Industry"]].dropna()
        if len(cheap):
            rows = ["*Lowest P/E (potential value)*",
                    "| Ticker | Company | PE | Industry |",
                    "|--------|---------|-----|----------|"]
            for _, r in cheap.nsmallest(6,"pe_ratio").iterrows():
                rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {r['pe_ratio']:.1f}x | {str(r['Main Industry'])[:20]} |")
            blocks.append("\n".join(rows))

    if "beta" in df.columns:
        beta = df["beta"].dropna()
        if len(beta):
            high_beta = df[df["beta"] > 1.5][["symbol","companyName","beta","Main Industry"]].dropna()
            blocks.append(
                f"**Beta** (n={len(beta)}): median {beta.median():.2f} | "
                f"{(beta > 1.5).sum()} high-beta (>1.5) | "
                f"{beta.between(0,0.7).sum()} low-beta (<0.7)"
            )
            if len(high_beta):
                rows = ["*Highest Beta*",
                        "| Ticker | Company | Beta | Industry |",
                        "|--------|---------|------|----------|"]
                for _, r in high_beta.nlargest(5,"beta").iterrows():
                    rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {r['beta']:.2f} | {str(r['Main Industry'])[:20]} |")
                blocks.append("\n".join(rows))

    if "dividend_yield" in df.columns:
        divs = df[df["dividend_yield"].notna() & (df["dividend_yield"] > 0)]
        if len(divs):
            blocks.append(f"**Dividend Payers**: {len(divs)} companies | median yield {divs['dividend_yield'].median():.2f}%")
            rows = ["*Highest Yielders*",
                    "| Ticker | Company | Yield | Industry |",
                    "|--------|---------|-------|----------|"]
            for _, r in divs.nlargest(6,"dividend_yield").iterrows():
                rows.append(f"| {r['symbol']} | {str(r['companyName'])[:26]} | {r['dividend_yield']:.1f}% | {str(r['Main Industry'])[:20]} |")
            blocks.append("\n".join(rows))

    return "\n\n".join(blocks)


def section_fundamentals(df: pd.DataFrame) -> str:
    blocks = ["## 8. Fundamentals Screen"]

    if "returnOnEquityTTM" in df.columns:
        roe = df[["symbol","companyName","returnOnEquityTTM","Main Industry","market_cap"]].dropna(subset=["returnOnEquityTTM"])
        roe = roe[roe["returnOnEquityTTM"].between(0.01, 10)]
        if len(roe):
            rows = ["**Top Return on Equity (TTM)**",
                    "| Ticker | Company | ROE | Industry | MCap |",
                    "|--------|---------|-----|----------|------|"]
            for _, r in roe.nlargest(8,"returnOnEquityTTM").iterrows():
                rows.append(
                    f"| {r['symbol']} | {str(r['companyName'])[:26]} "
                    f"| {r['returnOnEquityTTM']*100:.1f}% "
                    f"| {str(r['Main Industry'])[:20]} "
                    f"| {_fmt_b(r['market_cap'])} |"
                )
            blocks.append("\n".join(rows))

    if "growthRevenue" in df.columns:
        rev_g = df[["symbol","companyName","growthRevenue","revenue","Main Industry"]].dropna(subset=["growthRevenue"])
        rev_g = rev_g[rev_g["growthRevenue"].between(-1, 5)]
        if len(rev_g):
            rows = ["**Fastest Revenue Growth (YoY)**",
                    "| Ticker | Company | Rev Growth | TTM Revenue | Industry |",
                    "|--------|---------|------------|-------------|----------|"]
            for _, r in rev_g.nlargest(8,"growthRevenue").iterrows():
                rows.append(
                    f"| {r['symbol']} | {str(r['companyName'])[:26]} "
                    f"| {r['growthRevenue']*100:+.1f}% "
                    f"| {_fmt_b(r['revenue'])} "
                    f"| {str(r['Main Industry'])[:20]} |"
                )
            blocks.append("\n".join(rows))

    if "netIncome" in df.columns:
        ni = df[["symbol","companyName","netIncome","Main Industry"]].dropna(subset=["netIncome"])
        ni = ni[ni["netIncome"] > 0]
        if len(ni):
            rows = ["**Most Profitable (Net Income TTM)**",
                    "| Ticker | Company | Net Income | Industry |",
                    "|--------|---------|------------|----------|"]
            for _, r in ni.nlargest(6,"netIncome").iterrows():
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
    grp = df.groupby("country_current").agg(
        companies=("symbol","count"),
        total_mcap=("market_cap","sum"),
        median_weekly=("monthly_pct_change","median"),
    ).reset_index().sort_values("total_mcap", ascending=False)

    rows = ["## 9. Geographic Exposure",
            "| Country | # Co | Total MCap | Median Weekly |",
            "|---------|------|------------|---------------|"]
    for _, r in grp.head(12).iterrows():
        rows.append(
            f"| {r['country_current']} "
            f"| {int(r['companies'])} "
            f"| {_fmt_b(r['total_mcap'])} "
            f"| {_fmt_pct(r['median_weekly'])} |"
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

    records = sorted(records, key=lambda x: x.get("filed_date",""), reverse=True)
    lines = ["## 10. 13F Institutional Watch",
             f"_{len(records)} funds with geospatial holdings in recent filings_\n"]

    for rec in records[:10]:
        fund    = rec.get("fund_name","Unknown")
        filed   = rec.get("filed_date","")
        period  = rec.get("period","")
        total_v = rec.get("total_geo_holdings_value_usd_k", 0)
        holdings = rec.get("top_geospatial_holdings", [])
        lines.append(
            f"**{fund}** — filed {filed} (period {period}) | "
            f"Geo exposure: {_fmt_b(total_v*1000)}"
        )
        rows = ["| Ticker | Issuer | Shares | Value | Method |",
                "|--------|--------|--------|-------|--------|"]
        for h in holdings[:8]:
            rows.append(
                f"| {h.get('ticker','')} "
                f"| {str(h.get('issuer_name',''))[:28]} "
                f"| {h.get('shares',0):,} "
                f"| {_fmt_b(h.get('value_usd_thousands',0)*1000)} "
                f"| {h.get('match_method','')} |"
            )
        lines.append("\n".join(rows) + "\n")

    return "\n".join(lines)


def section_trend_signal(df: pd.DataFrame, df_prev: Optional[pd.DataFrame]) -> str:
    lines = ["## 11. Trend Signal"]

    if "current_price" in df.columns and "avg_3mo" in df.columns:
        sub = df[["current_price","avg_3mo"]].dropna()
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
            f"- **YTD Scorecard**: {int((ytd>0).sum())} positive / "
            f"{int((ytd<=0).sum())} negative | median YTD {_fmt_pct(ytd.median())}"
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
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are the editor of **GeospatialFM Weekly** — the definitive intelligence
    brief for institutional investors, analysts, and founders tracking the
    publicly traded geospatial technology sector (~1,000 companies globally).

    You will receive a structured data payload covering all 11 sections below.
    YOUR JOB: Transform EVERY section into polished, insight-driven prose + tables.
    Do not skip or abbreviate any section. Use specific numbers from the data.
    Avoid vague sentences like "the market was mixed" — name the tickers and quantify.

    FORMAT RULES:
    - Keep the section headers (## 1., ## 2., etc.) exactly as given
    - Reproduce markdown tables verbatim, then add 2-4 sentences of editorial analysis
    - Bold the single most important data point in each section
    - End the newsletter with a one-line "📌 Data Note" on methodology/freshness
    - Target length: 1,400–2,000 words
    - Tone: authoritative, sell-side sector note — concise, quantified, no fluff
""").strip()


def call_ollama(data_payload: str, max_tokens: int = 3500) -> str:
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
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


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

    print(f"[newsletter] Calling Ollama Cloud ({OLLAMA_MODEL}, max_tokens=3500)...")
    newsletter_md = call_ollama(data_payload)

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
