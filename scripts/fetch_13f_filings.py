#!/usr/bin/env python3
"""
GeospatialFM — SEC EDGAR 13F Filing Fetcher
---------------------------------------------
Searches SEC EDGAR full-text search + the EDGAR company facts API for
Form 13-F institutional holdings that mention geospatial companies
tracked in the GeospatialFM universe.

What it does:
  1. Loads your ticker universe from the cleaned parquet/CSV
  2. Queries EDGAR EFTS (full-text search) for recent 13F-HR filings
  3. For each filing, pulls the holding summary JSON and extracts rows
     matching your geo tickers
  4. Saves per-filing JSON summaries to filings/13f/
  5. Writes a consolidated CSV: filings/13f/all_13f_holdings.csv

Usage:
    python scripts/fetch_13f_filings.py [--top N] [--lookback-days 90]

EDGAR APIs used (all free, no API key required — rate limit: ~10 req/s):
  - https://efts.sec.gov/LATEST/search-index?q=...&dateRange=custom&...
  - https://data.sec.gov/submissions/{CIK}.json
  - https://www.sec.gov/Archives/edgar/...

Respects EDGAR rate limits with a 0.12 s inter-request sleep.
"""

import json
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_EFTS = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/{cik}.json"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/full-index"

HEADERS = {
    "User-Agent": "GeospatialFM research-bot ryan@iqspatial.com",  # EDGAR requires this
    "Accept-Encoding": "gzip, deflate",
}

SLEEP_BETWEEN_REQUESTS = 0.15  # stay well under EDGAR 10 req/s limit
FORM_TYPE = "13F-HR"


# ---------------------------------------------------------------------------
# Load universe
# ---------------------------------------------------------------------------

def load_universe(path: str = "geospatial_companies_cleaned.parquet") -> set:
    """Return a set of uppercase ticker symbols from the geo universe."""
    p = Path(path)
    if not p.exists():
        # fallback to CSV
        csv_p = p.with_suffix(".csv")
        if csv_p.exists():
            df = pd.read_csv(csv_p, usecols=["YahooSymbolClean"])
        else:
            raise FileNotFoundError(f"Cannot find universe file at {p} or {csv_p}")
    else:
        df = pd.read_parquet(p, columns=["YahooSymbolClean"])

    tickers = set(df["YahooSymbolClean"].dropna().str.upper().tolist())
    print(f"[13f] Universe loaded: {len(tickers)} tickers")
    return tickers


# ---------------------------------------------------------------------------
# EDGAR EFTS full-text search for 13F-HR filings
# ---------------------------------------------------------------------------

def search_recent_13f_filings(lookback_days: int = 90, max_hits: int = 200) -> list:
    """
    Search EDGAR EFTS for 13F-HR filings in the last N days.
    Returns a list of filing dicts with accession number, CIK, filed date, etc.
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)

    params = {
        "q": '""',          # blank query = all filings (filtered by form type below)
        "dateRange": "custom",
        "startdt": str(start_date),
        "enddt": str(end_date),
        "forms": FORM_TYPE,
        "_source": "file_date,period_of_report,entity_name,file_num,period_of_report",
        "from": 0,
        "size": min(max_hits, 200),
    }

    print(f"[13f] Searching EDGAR for {FORM_TYPE} filings {start_date} → {end_date}...")
    resp = requests.get(EDGAR_EFTS, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("hits", {}).get("hits", [])
    print(f"[13f] Found {len(hits)} filing hits")

    filings = []
    for h in hits:
        src = h.get("_source", {})
        filings.append({
            "accession_number": h.get("_id", "").replace(":", "-"),
            "cik": src.get("file_num", ""),
            "entity_name": src.get("entity_name", ""),
            "filed_date": src.get("file_date", ""),
            "period": src.get("period_of_report", ""),
        })

    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return filings


# ---------------------------------------------------------------------------
# Fetch holding detail from EDGAR submissions API
# ---------------------------------------------------------------------------

def _cik_from_entity(entity_name: str) -> Optional[str]:
    """Search EDGAR company search to resolve a CIK from entity name."""
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": f'"{entity_name}"',
        "forms": FORM_TYPE,
        "dateRange": "custom",
        "startdt": "2024-01-01",
        "enddt": datetime.utcnow().date().isoformat(),
        "_source": "file_num,entity_name",
        "size": 1,
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"].get("file_num", "")
    except Exception:
        pass
    return None


def fetch_13f_xml_holdings(accession: str, cik: str) -> list:
    """
    Pull the primary document from a 13F-HR filing index and parse holdings.
    Returns a list of holding dicts: {issuer_name, class, cusip, value_usd_k, shares}.
    
    EDGAR provides an XML holding report for modern filings (2013+).
    """
    # Normalise CIK to 10-digit zero-padded
    cik_clean = str(cik).lstrip("0").zfill(10)
    acc_clean = accession.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/{cik_clean}/{acc_clean}/{accession}-index.htm"
    )
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    try:
        r = requests.get(index_url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  [13f] Could not fetch index {accession}: {e}")
        return []

    # Find the infotable XML href
    import re
    xml_match = re.search(r'href="([^"]+infotable[^"]*\.xml)"', r.text, re.IGNORECASE)
    if not xml_match:
        xml_match = re.search(r'href="([^"]+\.xml)"', r.text, re.IGNORECASE)
    if not xml_match:
        return []

    xml_href = xml_match.group(1)
    if not xml_href.startswith("http"):
        xml_href = "https://www.sec.gov" + xml_href

    time.sleep(SLEEP_BETWEEN_REQUESTS)
    try:
        xr = requests.get(xml_href, headers=HEADERS, timeout=30)
        xr.raise_for_status()
    except Exception as e:
        print(f"  [13f] Could not fetch XML {xml_href}: {e}")
        return []

    return parse_13f_xml(xr.text)


def parse_13f_xml(xml_text: str) -> list:
    """Parse 13F-HR infotable XML into a list of holding dicts."""
    import xml.etree.ElementTree as ET

    # Strip namespace for simpler parsing
    import re
    xml_clean = re.sub(r' xmlns[^"]*"[^"]*"', "", xml_text)
    xml_clean = re.sub(r"<\?xml[^>]*\?>", "", xml_clean).strip()

    holdings = []
    try:
        root = ET.fromstring(xml_clean)
    except ET.ParseError as e:
        print(f"  [13f] XML parse error: {e}")
        return holdings

    ns_map = {"": ""}
    for entry in root.iter("infoTable"):
        def g(tag):
            el = entry.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        holdings.append({
            "issuer_name": g("nameOfIssuer"),
            "class": g("titleOfClass"),
            "cusip": g("cusip"),
            "value_usd_thousands": _safe_int(g("value")),
            "shares": _safe_int(g("sshPrnamt")),
            "share_type": g("sshPrnamtType"),
            "investment_discretion": g("investmentDiscretion"),
        })

    return holdings


def _safe_int(s: str) -> int:
    try:
        return int(s.replace(",", "").strip())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Match holdings to geo universe
# ---------------------------------------------------------------------------

def match_to_universe(holdings: list, universe_tickers: set) -> list:
    """
    Attempt fuzzy match of issuer names / CUSIP prefix to our geo universe.
    This is best-effort — CUSIP→ticker mapping would require a paid feed.
    We match on issuer name keywords for the open-source version.
    """
    geo_names = {t.upper() for t in universe_tickers}

    # Build a rough name→ticker lookup from issuer words
    matched = []
    for h in holdings:
        issuer = h["issuer_name"].upper()
        # Direct ticker match in issuer name
        for tick in geo_names:
            if tick in issuer.split() or tick == issuer[:len(tick)]:
                matched.append({**h, "matched_ticker": tick})
                break

    return matched


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_filing_summary(filing: dict, matched_holdings: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = filing["accession_number"].replace("/", "_")
    out_path = out_dir / f"{filing['filed_date']}_{acc[:20]}.json"

    doc = {
        "fund_name": filing.get("entity_name", ""),
        "filed_date": filing.get("filed_date", ""),
        "period": filing.get("period", ""),
        "accession": filing.get("accession_number", ""),
        "top_geospatial_holdings": [
            {
                "ticker": h.get("matched_ticker", ""),
                "issuer_name": h["issuer_name"],
                "shares": h["shares"],
                "value_usd_thousands": h["value_usd_thousands"],
                "cusip": h["cusip"],
            }
            for h in sorted(matched_holdings, key=lambda x: -x["value_usd_thousands"])[:20]
        ],
        "total_geo_holdings_value_usd_k": sum(
            h["value_usd_thousands"] for h in matched_holdings
        ),
        "total_holdings_in_filing": len(matched_holdings),
    }
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"  [13f] Saved → {out_path.name}  ({len(matched_holdings)} geo holdings)")
    return doc


def save_consolidated_csv(all_records: list, out_dir: Path):
    if not all_records:
        print("[13f] No geo holdings found — CSV not written.")
        return
    rows = []
    for rec in all_records:
        for h in rec.get("top_geospatial_holdings", []):
            rows.append(
                {
                    "fund_name": rec["fund_name"],
                    "filed_date": rec["filed_date"],
                    "period": rec["period"],
                    "ticker": h["ticker"],
                    "issuer_name": h["issuer_name"],
                    "shares": h["shares"],
                    "value_usd_thousands": h["value_usd_thousands"],
                    "cusip": h["cusip"],
                }
            )
    df = pd.DataFrame(rows)
    csv_path = out_dir / "all_13f_holdings.csv"
    df.to_csv(csv_path, index=False)
    print(f"[13f] Consolidated CSV → {csv_path}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch 13F filings for geo universe")
    parser.add_argument("--universe", default="geospatial_companies_cleaned.parquet")
    parser.add_argument("--output-dir", default="filings/13f")
    parser.add_argument("--lookback-days", type=int, default=90,
                        help="How many days back to search for 13F filings")
    parser.add_argument("--max-filings", type=int, default=50,
                        help="Max number of filings to process (rate-limit safety)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    universe = load_universe(args.universe)

    filings = search_recent_13f_filings(
        lookback_days=args.lookback_days,
        max_hits=args.max_filings,
    )

    all_records = []
    for i, filing in enumerate(filings[:args.max_filings]):
        print(f"[13f] Processing {i+1}/{len(filings)}: {filing['entity_name']} ({filing['filed_date']})")
        cik = filing.get("cik", "")
        if not cik:
            print("  [13f] Skipping — no CIK")
            continue

        holdings = fetch_13f_xml_holdings(filing["accession_number"], cik)
        if not holdings:
            print("  [13f] No holdings parsed")
            continue

        matched = match_to_universe(holdings, universe)
        if not matched:
            print(f"  [13f] 0 geo matches in {len(holdings)} total holdings — skipping")
            continue

        rec = save_filing_summary(filing, matched, out_dir)
        all_records.append(rec)

    save_consolidated_csv(all_records, out_dir)
    print(f"\n[13f] ✅ Done. Processed {len(all_records)} filings with geo holdings.")


if __name__ == "__main__":
    main()
