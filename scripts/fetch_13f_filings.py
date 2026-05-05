#!/usr/bin/env python3
"""
GeospatialFM — SEC EDGAR 13F Filing Fetcher
---------------------------------------------
Searches SEC EDGAR full-text search for Form 13-F institutional holdings
that mention geospatial companies tracked in the GeospatialFM universe.

What it does:
  1. Loads your ticker universe from the cleaned parquet/CSV
  2. Builds a CUSIP→ticker map via OpenFIGI API (free, no key needed)
     and caches it to filings/cusip_cache.json
  3. Queries EDGAR EFTS for recent 13F-HR filings
  4. For each filing, fetches + parses the XML infotable
  5. Matches holdings by CUSIP (primary) then issuer name (fallback)
  6. Saves per-filing JSON summaries to filings/13f/
  7. Writes a consolidated CSV: filings/13f/all_13f_holdings.csv

APIs used (all free):
  - https://api.openfigi.com/v3/mapping  (CUSIP→ticker, 250 req/min unauthed)
  - https://efts.sec.gov/LATEST/search-index
  - https://www.sec.gov/Archives/edgar/...

Usage:
    python scripts/fetch_13f_filings.py [--lookback-days 90] [--max-filings 50]
"""

import json
import re
import sys
import time
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_EFTS   = "https://efts.sec.gov/LATEST/search-index"
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

EDGAR_HEADERS = {
    "User-Agent": "GeospatialFM research-bot ryan@iqspatial.com",
    "Accept-Encoding": "gzip, deflate",
}
FIGI_HEADERS = {
    "Content-Type": "application/json",
    # Add "X-OPENFIGI-APIKEY": os.environ.get("OPENFIGI_KEY","") for higher limits
}

SLEEP_EDGAR  = 0.15   # EDGAR: 10 req/s max
SLEEP_FIGI   = 0.25   # OpenFIGI: 250 req/min unauthed → ~4/s
FIGI_BATCH   = 100    # OpenFIGI max batch size
FORM_TYPE    = "13F-HR"
CUSIP_CACHE  = Path("filings/cusip_cache.json")


# ---------------------------------------------------------------------------
# Load universe
# ---------------------------------------------------------------------------

def load_universe(path: str = "geospatial_companies_cleaned.parquet") -> tuple[set, dict]:
    """
    Return:
      tickers  — set of uppercase Yahoo ticker symbols
      name_map — dict {NORMALIZED_COMPANY_NAME: ticker} for fallback matching
    """
    p = Path(path)
    cols = ["YahooSymbolClean", "companyName"]
    if not p.exists():
        csv_p = p.with_suffix(".csv")
        if csv_p.exists():
            df = pd.read_csv(csv_p, usecols=cols)
        else:
            raise FileNotFoundError(f"Cannot find universe file at {p} or {csv_p}")
    else:
        df = pd.read_parquet(p, columns=cols)

    df = df.dropna(subset=["YahooSymbolClean"])
    tickers = set(df["YahooSymbolClean"].str.upper().tolist())

    # Build name→ticker for fuzzy fallback (strip legal suffixes, uppercase)
    _strip = re.compile(
        r"\b(inc|corp|ltd|llc|plc|co|group|holdings|technologies|technology"
        r"|solutions|systems|international|global|holdings|sa|nv|ag|se)\b\.?",
        re.IGNORECASE,
    )
    name_map = {}
    for _, row in df.iterrows():
        raw = str(row.get("companyName", ""))
        normed = _strip.sub("", raw).strip().upper()
        normed = re.sub(r"\s+", " ", normed)
        if normed:
            name_map[normed] = row["YahooSymbolClean"].upper()

    print(f"[13f] Universe: {len(tickers)} tickers, {len(name_map)} name entries")
    return tickers, name_map


# ---------------------------------------------------------------------------
# OpenFIGI — CUSIP → ticker resolution
# ---------------------------------------------------------------------------

def load_cusip_cache() -> dict:
    """Load the persistent CUSIP→ticker cache from disk."""
    if CUSIP_CACHE.exists():
        try:
            return json.loads(CUSIP_CACHE.read_text())
        except Exception:
            pass
    return {}


def save_cusip_cache(cache: dict):
    CUSIP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CUSIP_CACHE.write_text(json.dumps(cache, indent=2))


def resolve_cusips_via_figi(cusips: list[str], cache: dict) -> dict:
    """
    Resolve a list of CUSIPs to tickers using the OpenFIGI batch mapping API.
    Returns updated cache dict {cusip: ticker_or_None}.

    OpenFIGI limits:
      - Unauthenticated: 250 requests/min, 10 jobs/request
      - With free API key: 250 req/min, 100 jobs/request
    We use batches of 100 and sleep between calls.
    """
    to_resolve = [c for c in cusips if c and c not in cache]
    if not to_resolve:
        return cache

    print(f"[figi] Resolving {len(to_resolve)} new CUSIPs via OpenFIGI...")
    resolved = 0

    for i in range(0, len(to_resolve), FIGI_BATCH):
        batch = to_resolve[i : i + FIGI_BATCH]
        payload = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]
        try:
            resp = requests.post(
                OPENFIGI_URL,
                headers=FIGI_HEADERS,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 429:
                print("[figi] Rate limited — sleeping 60s")
                time.sleep(60)
                resp = requests.post(OPENFIGI_URL, headers=FIGI_HEADERS, json=payload, timeout=30)
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            print(f"[figi] Batch error: {e}")
            for c in batch:
                cache[c] = None
            time.sleep(SLEEP_FIGI)
            continue

        for cusip, result in zip(batch, results):
            ticker = None
            if "data" in result and result["data"]:
                # Prefer exchange-traded equity; take first hit
                for item in result["data"]:
                    if item.get("securityType", "") in ("Common Stock", "ETP", "ETF"):
                        ticker = item.get("ticker", "").upper() or None
                        if ticker:
                            break
                if not ticker:
                    ticker = result["data"][0].get("ticker", "").upper() or None
            cache[cusip] = ticker
            if ticker:
                resolved += 1

        time.sleep(SLEEP_FIGI)

    save_cusip_cache(cache)
    print(f"[figi] Resolved {resolved}/{len(to_resolve)} CUSIPs to tickers")
    return cache


# ---------------------------------------------------------------------------
# EDGAR full-text search for 13F-HR filings
# ---------------------------------------------------------------------------

# The EFTS search-index endpoint requires a non-empty query string and is
# unreliable for form-type-only searches.  The stable alternative is the
# EDGAR full-text search API at efts.sec.gov/LATEST/search-index with a
# category filter, OR the simpler EDGAR company-search endpoint.
# We use the EDGAR EFTS search endpoint with category=form-type which
# accepts a wildcard query correctly.

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FULL_SEARCH = "https://efts.sec.gov/LATEST/search-index"

def search_recent_13f_filings(lookback_days: int = 90, max_hits: int = 200) -> list:
    """
    Fetch recent 13F-HR filings from EDGAR using the full-text search API.
    Falls back to the EDGAR company submissions index if the search fails.
    Returns a list of filing dicts: {accession_number, cik, entity_name, filed_date, period}.
    """
    end_date   = datetime.now().date()  # UTC-equivalent for scheduling
    start_date = end_date - timedelta(days=lookback_days)
    print(f"[13f] Searching EDGAR for {FORM_TYPE} filings {start_date} → {end_date}...")

    filings = _search_via_efts(start_date, end_date, max_hits)
    if not filings:
        print("[13f] EFTS returned 0 results — trying EDGAR full-index fallback...")
        filings = _search_via_full_index(start_date, end_date, max_hits)

    print(f"[13f] Found {len(filings)} filings")
    return filings


def _search_via_efts(start_date, end_date, max_hits: int) -> list:
    """
    Primary: EDGAR EFTS search with dateRange + forms filter.
    Uses a single-space query (EFTS requires non-empty q).
    """
    # EFTS needs a real query token; "13F" appears in all 13F filing text
    params = {
        "q": "13F",
        "dateRange": "custom",
        "startdt": str(start_date),
        "enddt":   str(end_date),
        "forms":   FORM_TYPE,
        "from":    0,
        "size":    min(max_hits, 100),
    }
    try:
        resp = requests.get(EDGAR_SEARCH, params=params, headers=EDGAR_HEADERS, timeout=30)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        time.sleep(SLEEP_EDGAR)
        return [_parse_efts_hit(h) for h in hits]
    except Exception as e:
        print(f"[13f] EFTS search error: {e}")
        return []


def _parse_efts_hit(h: dict) -> dict:
    src = h.get("_source", {})
    # EFTS _id is formatted as  CIK:accession  e.g.  0001234567:0001234567-24-001234
    raw_id = h.get("_id", "")
    parts  = raw_id.split(":", 1)
    cik    = parts[0].lstrip("0") if parts else ""
    acc    = parts[1].replace(":", "-") if len(parts) > 1 else raw_id.replace(":", "-")
    return {
        "accession_number": acc,
        "cik":              cik,
        "entity_name":      src.get("entity_name", src.get("display_names", [""])[0] if src.get("display_names") else ""),
        "filed_date":       src.get("file_date",   src.get("period_of_report", "")),
        "period":           src.get("period_of_report", ""),
    }


def _search_via_full_index(start_date, end_date, max_hits: int) -> list:
    """
    Fallback: scrape the EDGAR quarterly full-index company.idx files.
    Covers Q filings up to the current quarter.
    """
    from io import StringIO

    filings = []
    seen_quarters = set()

    # Determine which year/quarter combos to fetch
    current = start_date.replace(day=1)
    while current <= end_date:
        q = (current.month - 1) // 3 + 1
        seen_quarters.add((current.year, q))
        # advance by ~3 months
        month = current.month + 3
        year  = current.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        current = current.replace(year=year, month=month)

    for (year, q) in sorted(seen_quarters):
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/company.idx"
        try:
            r = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
            r.raise_for_status()
            time.sleep(SLEEP_EDGAR)
        except Exception as e:
            print(f"  [13f] full-index {year}/QTR{q} error: {e}")
            continue

        # Parse fixed-width company.idx
        # Format: Company Name | Form Type | CIK | Date Filed | Filename
        lines = r.text.splitlines()
        for line in lines:
            if FORM_TYPE not in line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                # Last field is the filename: edgar/data/CIK/accession.txt
                filename   = parts[-1]
                filed_date = parts[-2]
                cik        = parts[-3]
                # accession embedded in filename path
                acc_raw    = filename.split("/")[-1].replace(".txt","").replace(".htm","")
                acc        = acc_raw  # already formatted as XXXXXXXXXX-YY-ZZZZZZ
                # entity name is everything before the form-type token
                ft_idx     = line.find(FORM_TYPE)
                entity     = line[:ft_idx].strip()

                if filed_date < str(start_date) or filed_date > str(end_date):
                    continue

                filings.append({
                    "accession_number": acc,
                    "cik":              cik.lstrip("0"),
                    "entity_name":      entity,
                    "filed_date":       filed_date,
                    "period":           "",
                })
                if len(filings) >= max_hits:
                    return filings
            except Exception:
                continue

    return filings


# ---------------------------------------------------------------------------
# Fetch holding detail from EDGAR archives
# ---------------------------------------------------------------------------

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
    time.sleep(SLEEP_EDGAR)

    try:
        r = requests.get(index_url, headers=EDGAR_HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"  [13f] Could not fetch index {accession}: {e}")
        return []

    # Find the infotable XML href
    xml_match = re.search(r'href="([^"]+infotable[^"]*\.xml)"', r.text, re.IGNORECASE)
    if not xml_match:
        xml_match = re.search(r'href="([^"]+\.xml)"', r.text, re.IGNORECASE)
    if not xml_match:
        return []

    xml_href = xml_match.group(1)
    if not xml_href.startswith("http"):
        xml_href = "https://www.sec.gov" + xml_href

    time.sleep(SLEEP_EDGAR)
    try:
        xr = requests.get(xml_href, headers=EDGAR_HEADERS, timeout=30)
        xr.raise_for_status()
    except Exception as e:
        print(f"  [13f] Could not fetch XML {xml_href}: {e}")
        return []

    return parse_13f_xml(xr.text)


def parse_13f_xml(xml_text: str) -> list:
    """Parse 13F-HR infotable XML into a list of holding dicts."""
    # Strip namespaces for simpler element access
    xml_clean = re.sub(r' xmlns[^"]*"[^"]*"', "", xml_text)
    xml_clean = re.sub(r"<\?xml[^>]*\?>", "", xml_clean).strip()

    holdings = []
    try:
        root = ET.fromstring(xml_clean)
    except ET.ParseError as e:
        print(f"  [13f] XML parse error: {e}")
        return holdings

    for entry in root.iter("infoTable"):
        def g(tag):
            el = entry.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        # sshPrnamt is nested under shrsOrPrnAmt in modern 13F XML
        shr_block = entry.find("shrsOrPrnAmt")
        if shr_block is not None:
            shares_el = shr_block.find("sshPrnamt")
            type_el   = shr_block.find("sshPrnamtType")
            shares_raw    = shares_el.text.strip() if shares_el is not None and shares_el.text else ""
            share_type    = type_el.text.strip()   if type_el   is not None and type_el.text   else ""
        else:
            shares_raw = g("sshPrnamt")
            share_type = g("sshPrnamtType")
        holdings.append({
            "issuer_name": g("nameOfIssuer"),
            "class": g("titleOfClass"),
            "cusip": g("cusip"),
            "value_usd_thousands": _safe_int(g("value")),
            "shares": _safe_int(shares_raw),
            "share_type": share_type,
            "investment_discretion": g("investmentDiscretion"),
        })

    return holdings


def _safe_int(s: str) -> int:
    try:
        return int(s.replace(",", "").strip())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Match holdings to geo universe  (CUSIP-first, name fallback)
# ---------------------------------------------------------------------------

_STRIP_LEGAL = re.compile(
    r"\b(inc|corp|ltd|llc|plc|co|group|holdings|technologies|technology"
    r"|solutions|systems|international|global|sa|nv|ag|se)\b\.?",
    re.IGNORECASE,
)


def _normalize(name: str) -> str:
    normed = _STRIP_LEGAL.sub("", name).strip().upper()
    return re.sub(r"\s+", " ", normed)


def match_to_universe(
    holdings: list,
    universe_tickers: set,
    name_map: dict,
    cusip_map: dict,
) -> list:
    """
    Match each holding to the geo universe using two strategies:
      1. CUSIP lookup via OpenFIGI-resolved cache (high precision)
      2. Normalized company name substring match (fallback)

    Returns matched holdings with an added `matched_ticker` field
    and `match_method` ('cusip' | 'name').
    """
    matched = []
    seen = set()  # deduplicate by (cusip, ticker)

    for h in holdings:
        ticker = None
        method = None

        # --- Strategy 1: CUSIP → ticker via OpenFIGI cache ---
        cusip = h.get("cusip", "")
        if cusip and cusip in cusip_map and cusip_map[cusip]:
            candidate = cusip_map[cusip].upper()
            if candidate in universe_tickers:
                ticker = candidate
                method = "cusip"

        # --- Strategy 2: normalized issuer name substring ---
        if not ticker:
            issuer_norm = _normalize(h.get("issuer_name", ""))
            # Strategy 2a: universe name is a substring of the issuer
            for name, tick in name_map.items():
                if len(name) >= 4 and name in issuer_norm:
                    ticker = tick
                    method = "name"
                    break
            # Strategy 2b: issuer norm is a substring of a universe name
            if not ticker:
                for name, tick in name_map.items():
                    if len(issuer_norm) >= 4 and issuer_norm in name:
                        ticker = tick
                        method = "name"
                        break
            # Strategy 2c: first significant word of issuer matches first word of universe name
            if not ticker:
                issuer_first = issuer_norm.split()[0] if issuer_norm else ""
                if len(issuer_first) >= 4:
                    for name, tick in name_map.items():
                        name_first = name.split()[0] if name else ""
                        if issuer_first == name_first and len(name_first) >= 4:
                            ticker = tick
                            method = "name-first-word"
                            break

        if ticker:
            key = (cusip, ticker)
            if key not in seen:
                seen.add(key)
                matched.append({**h, "matched_ticker": ticker, "match_method": method})

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
                "match_method": h.get("match_method", ""),
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
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--max-filings", type=int, default=50)
    parser.add_argument("--skip-figi", action="store_true",
                        help="Skip OpenFIGI resolution (name-only matching)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    universe, name_map = load_universe(args.universe)

    # --- Build / refresh CUSIP cache via OpenFIGI ---
    cusip_cache = load_cusip_cache()

    filings = search_recent_13f_filings(
        lookback_days=args.lookback_days,
        max_hits=args.max_filings,
    )

    # First pass: collect all CUSIPs across all filings so we can batch-resolve
    if not args.skip_figi:
        print("[13f] Pre-fetching holding XML to collect CUSIPs for batch FIGI lookup...")
        all_holdings_raw = {}
        for filing in filings[:args.max_filings]:
            cik = filing.get("cik", "")
            if not cik:
                continue
            holdings = fetch_13f_xml_holdings(filing["accession_number"], cik)
            all_holdings_raw[filing["accession_number"]] = holdings

        all_cusips = list({h["cusip"] for hl in all_holdings_raw.values() for h in hl if h.get("cusip")})
        cusip_cache = resolve_cusips_via_figi(all_cusips, cusip_cache)
    else:
        all_holdings_raw = {}

    # Second pass: match and save
    all_records = []
    for i, filing in enumerate(filings[:args.max_filings]):
        print(f"[13f] {i+1}/{len(filings)}: {filing['entity_name']} ({filing['filed_date']})")
        cik = filing.get("cik", "")
        if not cik:
            continue

        # Reuse pre-fetched holdings if available
        holdings = all_holdings_raw.get(filing["accession_number"])
        if holdings is None:
            holdings = fetch_13f_xml_holdings(filing["accession_number"], cik)

        if not holdings:
            print("  [13f] No holdings parsed")
            continue

        matched = match_to_universe(holdings, universe, name_map, cusip_cache)
        cusip_hits = sum(1 for h in matched if h.get("match_method") == "cusip")
        name_hits  = sum(1 for h in matched if h.get("match_method") == "name")

        if not matched:
            print(f"  [13f] 0 geo matches in {len(holdings)} holdings — skipping")
            continue

        print(f"  [13f] {len(matched)} geo matches ({cusip_hits} CUSIP, {name_hits} name)")
        rec = save_filing_summary(filing, matched, out_dir)
        all_records.append(rec)

    save_consolidated_csv(all_records, out_dir)
    print(f"\n[13f] ✅ Done. {len(all_records)} filings with geo holdings saved.")


if __name__ == "__main__":
    main()
