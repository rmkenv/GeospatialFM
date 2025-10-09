# GeospatialFM

A curated dataset and tooling for analyzing publicly traded geospatial companies, with monthly portfolio snapshots, value-investing utilities, and fast columnar storage.

This repo hosts:
- A cleaned master list of geospatial-related public companies (standardized tickers, industries, and HQ geocodes)
- Python scripts to fetch monthly market snapshots via yfinance
- A GitHub Actions workflow to automate end-of-month captures
- Output data stored as partitioned Parquet for fast analytics

Useful for screens, dashboards, and value-investing analyses across the “geo” universe.

## Features

- Cleaned master dataset
  - Yahoo Finance-ready tickers (`YahooSymbolClean`)
  - Normalized `Main Industry` and `Sub Industry`
  - Geocoded HQ latitude/longitude
- Fast storage
  - Master list in Parquet for high-speed I/O
  - Monthly snapshots saved as partitioned Parquet by year/month
- Automated monthly capture
  - GitHub Actions runs daily; script only saves on the last business day of each month
- Extensible analysis
  - Works with `pandas`, `pyarrow`, and `yfinance`
  - Easy to use in notebooks and pipelines

## Repository Structure

```
data/
  geospatial_companies_cleaned.parquet      # master company list (cleaned)
  snapshots/
    year=YYYY/
      month=MM/
        snapshot_YYYY-MM-DD.parquet         # last business day snapshot
  latest/
    latest_snapshot.parquet                 # most recent snapshot (convenience)
scripts/
  capture_monthly_snapshot.py               # fetch prices/metrics and write snapshot
.github/workflows/
  monthly_snapshot.yml                      # GitHub Actions schedule
requirements.txt                            # Python dependencies
```

Note: Folder contents may evolve.

## Quickstart

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies:
- pandas>=2.0.0
- yfinance>=0.2.28
- pyarrow>=12.0.0
- numpy>=1.24.0

### 2) Master Data

If you only have CSV, convert to Parquet:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/rmkenv/GeospatialFM/main/geospatial_companies_cleaned.csv"
df = pd.read_csv(url)
df.to_parquet("data/geospatial_companies_cleaned.parquet", index=False, compression="snappy")
```

### 3) Run a Snapshot Locally

```bash
python scripts/capture_monthly_snapshot.py
```

This will:
- Load companies from `data/geospatial_companies_cleaned.parquet`
- Pull recent prices + fundamentals via Yahoo Finance
- Save a partitioned snapshot under `data/snapshots/year=YYYY/month=MM/`
- Update `data/latest/latest_snapshot.parquet`

## Output Data Model (snapshots)

Each row includes:
- `symbol` — Yahoo-ready ticker (e.g., MSFT, 0700.HK, RIO.L)
- `price` — latest close captured
- Fundamentals (best-effort from yfinance’s `info`), e.g.:
  - `marketCap`, `enterpriseValue`, `trailingPE`, `forwardPE`
  - `priceToBook`, `priceToSales`, `revenue`, `margins`
  - `returnOnAssets`, `returnOnEquity`, `freeCashflow`, etc.
- Enriched metadata (merged from master):
  - `companyName`, `Main Industry`, `Sub Industry`, `country`, `HQ_Lat`, `HQ_Lon`
- `snapshot_date` — the effective date (last business day of the month)

Note: Yahoo fields can be sparse/subject to change; captured “as available.”

## Automation (GitHub Actions)

Workflow: `.github/workflows/monthly_snapshot.yml`

- Scheduled daily at a fixed UTC time
- Python script checks if “today is the last business day” (pandas BMonthEnd)
- Commits a new Parquet snapshot only on the last business day

Cron example (daily at 21:00 UTC):
```yaml
on:
  schedule:
    - cron: "0 21 * * *"
  workflow_dispatch: {}
```

Why daily? GitHub cron doesn’t support “last day” directly. The Python script enforces the last business day rule.

## Example Analysis

```python
import pandas as pd
from pathlib import Path

# Load a specific month
df_m = pd.read_parquet("data/snapshots/year=2025/month=01/snapshot_2025-01-31.parquet")

# Load all snapshots
frames = [pd.read_parquet(p) for p in Path("data/snapshots").rglob("*.parquet")]
df_all = pd.concat(frames, ignore_index=True)

# Industry market cap by month
out = (
    df_all
    .groupby(["snapshot_date", "Main Industry"])["marketCap"]
    .sum()
    .reset_index()
)
print(out.head())
```

## Best Practices

- Partitioned Parquet per month:
  - Efficient diffs, fast partial reads, append-only
- Keep `data/latest/latest_snapshot.parquet` for quick access
- If files grow large, consider Git LFS:
  ```bash
  git lfs track "data/snapshots/**/*.parquet"
  ```

## Development Notes

- Rate limits: `yfinance` calls are batched with minimal history to reduce load
- Time zones: actions run in UTC; snapshot date is computed with calendar logic
- Reproducibility: pin dependency versions when stabilizing

## Roadmap

- Retry/backoff and better telemetry
- Optional premium data sources
- ADR mappings; country-of-listing vs HQ
- Notebook(s) for value-analysis and DCF screens

## Contributing

Issues and PRs welcome. Please keep Parquet schema/partitions consistent.

## License

Choose a license (e.g., MIT) and add a LICENSE file.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance)
- [Apache Arrow / PyArrow](https://arrow.apache.org/)
