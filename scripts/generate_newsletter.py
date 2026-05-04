#!/usr/bin/env python3
"""
GeospatialFM Newsletter Generator
----------------------------------
Reads the latest weekly snapshot + 13F filing data and generates
a Substack-ready newsletter via Ollama Cloud (gpt-oss:20b).

Usage:
    python scripts/generate_newsletter.py [--snapshot PATH] [--output PATH]

Outputs:
    - newsletters/YYYY-MM-DD_newsletter.md   (Markdown body)
    - newsletters/YYYY-MM-DD_newsletter.html (HTML version for email)

Requires:
    OLLAMA_API_KEY env var (or set in .env)
"""

import os
import sys
import json
import argparse
import textwrap
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Ollama Cloud config  (matches your existing stack: gpt-oss:20b)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "https://api.ollama.com"
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OLLAMA_API_KEY}",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_snapshot_path(snapshots_dir: str = "snapshots") -> Optional[Path]:
    """Return the most recently dated snapshot parquet file."""
    root = Path(snapshots_dir)
    files = sorted(root.rglob("snapshot_*.parquet"))
    return files[-1] if files else None


def load_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[newsletter] Loaded {len(df)} rows from {path.name}")
    return df


def build_market_summary(df: pd.DataFrame) -> dict:
    """
    Distill the snapshot dataframe into a compact summary dict
    suitable for injection into the LLM prompt.
    """
    summary = {}

    # --- Top performers (1-week % change) ---
    perf_col = next(
        (c for c in df.columns if "pct_change" in c.lower() and "1" not in c),
        None,
    )
    if perf_col and perf_col in df.columns:
        top5 = (
            df[["symbol", "companyName", perf_col, "Main Industry", "marketCap"]]
            .dropna(subset=[perf_col])
            .nlargest(5, perf_col)
        )
        bot5 = (
            df[["symbol", "companyName", perf_col, "Main Industry", "marketCap"]]
            .dropna(subset=[perf_col])
            .nsmallest(5, perf_col)
        )
        summary["top_5_gainers"] = top5.to_dict(orient="records")
        summary["top_5_losers"] = bot5.to_dict(orient="records")
        summary["perf_col"] = perf_col

    # --- Industry breakdown ---
    if "Main Industry" in df.columns and "marketCap" in df.columns:
        industry_mcap = (
            df.groupby("Main Industry")["marketCap"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .apply(lambda x: f"${x/1e9:.1f}B")
            .to_dict()
        )
        summary["industry_market_caps"] = industry_mcap

    # --- Universe stats ---
    summary["total_companies"] = int(len(df))
    if "marketCap" in df.columns:
        summary["total_market_cap_B"] = round(df["marketCap"].sum() / 1e9, 1)
        summary["median_market_cap_B"] = round(df["marketCap"].median() / 1e9, 3)

    # --- PE ratios ---
    pe_col = next((c for c in df.columns if "pe" in c.lower() or "trailing" in c.lower()), None)
    if pe_col:
        valid_pe = df[pe_col].replace(0, pd.NA).dropna()
        if len(valid_pe) > 0:
            summary["median_pe"] = round(valid_pe.median(), 1)

    return summary


def load_13f_data(filing_dir: str = "filings/13f") -> list:
    """Load any pre-fetched 13F JSON summaries from the filings dir."""
    root = Path(filing_dir)
    records = []
    for f in sorted(root.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            records.append(data)
        except Exception as e:
            print(f"[newsletter] Warning: could not load {f}: {e}")
    return records


def format_13f_block(filings: list) -> str:
    if not filings:
        return "No 13F filing data available this week."
    lines = []
    for rec in filings[:5]:  # cap at 5 most recent
        name = rec.get("fund_name", "Unknown Fund")
        filed = rec.get("filed_date", "")
        top_geo = rec.get("top_geospatial_holdings", [])
        lines.append(f"**{name}** (filed {filed})")
        if top_geo:
            for h in top_geo[:3]:
                ticker = h.get("ticker", "")
                shares = h.get("shares", "")
                value_k = h.get("value_usd_thousands", "")
                lines.append(f"  - {ticker}: {shares:,} shares / ${value_k:,}K")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama Cloud call
# ---------------------------------------------------------------------------

def call_ollama(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
    """
    Hit Ollama Cloud /api/chat with the native ollama payload format.
    Supports streaming=False for simplicity in GitHub Actions.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    url = f"{OLLAMA_BASE_URL}/api/chat"
    try:
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Ollama Cloud returns {"message": {"content": "..."}}
        return data.get("message", {}).get("content", "").strip()
    except requests.exceptions.HTTPError as e:
        print(f"[newsletter] Ollama HTTP error: {e} — {resp.text[:300]}")
        raise
    except Exception as e:
        print(f"[newsletter] Ollama error: {e}")
        raise


# ---------------------------------------------------------------------------
# Newsletter builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are the editor of GeospatialFM Weekly, a sharp, data-driven newsletter
    for institutional investors and analysts tracking the publicly traded
    geospatial technology sector.

    Tone: authoritative, concise, analytical. Avoid hype. Use specific numbers.
    Structure every newsletter with these exact sections:
    1. **🌍 The Week in Geo** — 2-3 sentence executive summary
    2. **📈 Market Movers** — top gainers and losers with brief analysis
    3. **🏭 Industry Pulse** — sector-level market cap trends
    4. **🗂️ 13F Watch** — institutional holdings changes (or note if no new filings)
    5. **🔭 Forward Look** — 2-3 sentences on macro tailwinds/headwinds for geo tech
    6. **📌 Data Note** — one sentence on methodology / data freshness

    Output valid Markdown only. Use tables where appropriate. No fluff.
""").strip()


def build_user_prompt(summary: dict, filing_block: str, snapshot_date: str) -> str:
    gainers = summary.get("top_5_gainers", [])
    losers = summary.get("top_5_losers", [])
    industries = summary.get("industry_market_caps", {})
    perf_col = summary.get("perf_col", "weekly_pct_change")

    gainer_lines = "\n".join(
        f"- {r['symbol']} ({r.get('companyName','')[:30]}): "
        f"{r.get(perf_col, 0):+.1f}% | MCap ${r.get('marketCap',0)/1e9:.1f}B | {r.get('Main Industry','')}"
        for r in gainers
    )
    loser_lines = "\n".join(
        f"- {r['symbol']} ({r.get('companyName','')[:30]}): "
        f"{r.get(perf_col, 0):+.1f}% | MCap ${r.get('marketCap',0)/1e9:.1f}B | {r.get('Main Industry','')}"
        for r in losers
    )
    industry_lines = "\n".join(
        f"- {ind}: {mcap}" for ind, mcap in industries.items()
    )

    return textwrap.dedent(f"""
        Snapshot date: {snapshot_date}
        Universe: {summary.get('total_companies')} geospatial companies
        Total market cap: ${summary.get('total_market_cap_B')}B
        Median market cap: ${summary.get('median_market_cap_B')}B
        Median P/E: {summary.get('median_pe', 'N/A')}

        TOP GAINERS (weekly):
        {gainer_lines or '(no data)'}

        TOP LOSERS (weekly):
        {loser_lines or '(no data)'}

        INDUSTRY MARKET CAPS (top 8):
        {industry_lines or '(no data)'}

        13F FILING HIGHLIGHTS:
        {filing_block}

        Write the full GeospatialFM Weekly newsletter for this week.
    """).strip()


def md_to_html(md_text: str, title: str) -> str:
    """Very lightweight Markdown → HTML (no external deps needed for basic formatting)."""
    import re

    html = md_text
    # Headers
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    # Bold
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    # List items
    html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
    # Paragraphs
    html = re.sub(r"\n\n", r"</p><p>", html)
    html = f"<p>{html}</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 700px; margin: 2em auto;
         color: #1a1a1a; line-height: 1.6; }}
  h1 {{ color: #0d6c3e; }} h2 {{ color: #1a5c8a; border-bottom: 1px solid #ddd; }}
  li {{ margin: 0.3em 0; }} table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ccc; padding: 6px 10px; }}
</style>
</head>
<body>
{html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate GeospatialFM weekly newsletter")
    parser.add_argument("--snapshot", default=None, help="Path to snapshot parquet (auto-detect if omitted)")
    parser.add_argument("--output-dir", default="newsletters", help="Output directory")
    parser.add_argument("--filings-dir", default="filings/13f", help="Directory with 13F JSON summaries")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt only, skip Ollama call")
    args = parser.parse_args()

    # Resolve snapshot
    snap_path = Path(args.snapshot) if args.snapshot else latest_snapshot_path()
    if snap_path is None or not snap_path.exists():
        print("[newsletter] ERROR: No snapshot found. Run capture_snapshot.py first.")
        sys.exit(1)

    df = load_snapshot(snap_path)
    snapshot_date = str(date.today())

    # Build inputs
    summary = build_market_summary(df)
    filings = load_13f_data(args.filings_dir)
    filing_block = format_13f_block(filings)
    user_prompt = build_user_prompt(summary, filing_block, snapshot_date)

    if args.dry_run:
        print("=== SYSTEM PROMPT ===\n", SYSTEM_PROMPT)
        print("\n=== USER PROMPT ===\n", user_prompt)
        return

    if not OLLAMA_API_KEY:
        print("[newsletter] WARNING: OLLAMA_API_KEY not set — Ollama call will likely fail.")

    print(f"[newsletter] Calling Ollama Cloud ({OLLAMA_MODEL})...")
    newsletter_md = call_ollama(SYSTEM_PROMPT, user_prompt, max_tokens=2200)

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{snapshot_date}_newsletter.md"
    html_path = out_dir / f"{snapshot_date}_newsletter.html"

    md_path.write_text(newsletter_md, encoding="utf-8")
    html_path.write_text(
        md_to_html(newsletter_md, f"GeospatialFM Weekly — {snapshot_date}"),
        encoding="utf-8",
    )

    print(f"[newsletter] ✅ Written:\n  {md_path}\n  {html_path}")
    print("\n--- PREVIEW (first 600 chars) ---")
    print(newsletter_md[:600])


if __name__ == "__main__":
    main()
