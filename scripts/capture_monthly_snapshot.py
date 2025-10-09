#!/usr/bin/env python3
"""
Monthly Geospatial Company Stock Snapshot Capture Script

This script:
1. Reads the geospatial companies data from parquet file
2. Fetches historical stock data using yfinance
3. Calculates monthly activity metrics (open, close, high, low, volume, % change)
4. Calculates multi-period performance metrics (3mo, 6mo, YTD, 1yr, 5yr)
   - Percentage change, high, low, average, volume, volatility for each period
5. Creates a timestamped snapshot with comprehensive metrics
6. Saves snapshot to GitHub repository in year-based folder structure
7. Uploads the snapshot to pCloud public upload link
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf
import requests
import json
from typing import Dict, List, Optional


def load_company_data(parquet_path: str = "geospatial_companies_cleaned.parquet") -> pd.DataFrame:
    """Load the geospatial companies data from parquet file."""
    print(f"Loading company data from {parquet_path}...")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data file not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} companies")
    return df


def calculate_period_metrics(hist_data: pd.DataFrame, period_name: str) -> Dict:
    """
    Calculate performance metrics for a given period of historical data.
    
    Args:
        hist_data: DataFrame with historical stock data
        period_name: Name of the period (e.g., '3mo', '6mo', 'ytd', '1yr', '5yr')
    
    Returns:
        Dictionary with calculated metrics for the period
    """
    if hist_data.empty or len(hist_data) < 2:
        return {
            f'pct_change_{period_name}': None,
            f'high_{period_name}': None,
            f'low_{period_name}': None,
            f'avg_{period_name}': None,
            f'volume_{period_name}': None,
            f'volatility_{period_name}': None,
        }
    
    try:
        start_price = float(hist_data.iloc[0]['Close'])
        end_price = float(hist_data.iloc[-1]['Close'])
        
        # Calculate percentage change
        if start_price > 0:
            pct_change = ((end_price - start_price) / start_price) * 100
        else:
            pct_change = 0.0
        
        return {
            f'pct_change_{period_name}': round(pct_change, 2),
            f'high_{period_name}': round(float(hist_data['High'].max()), 2),
            f'low_{period_name}': round(float(hist_data['Low'].min()), 2),
            f'avg_{period_name}': round(float(hist_data['Close'].mean()), 2),
            f'volume_{period_name}': int(hist_data['Volume'].sum()),
            f'volatility_{period_name}': round(float(hist_data['Close'].std()), 2),
        }
    except Exception as e:
        print(f"    Warning: Error calculating {period_name} metrics: {e}")
        return {
            f'pct_change_{period_name}': None,
            f'high_{period_name}': None,
            f'low_{period_name}': None,
            f'avg_{period_name}': None,
            f'volume_{period_name}': None,
            f'volatility_{period_name}': None,
        }


def fetch_monthly_stock_data(tickers: List[str]) -> Dict[str, Dict]:
    """
    Fetch monthly stock data for given tickers using yfinance.
    Calculates monthly activity metrics including:
    - Opening price (first trading day of month)
    - Closing price (last trading day of month)
    - Monthly high and low
    - Total trading volume
    - Percentage change over the month
    
    Also calculates multi-period performance metrics:
    - 3 months, 6 months, YTD, 1 year, 5 years
    """
    print(f"\nFetching stock data with multi-period metrics for {len(tickers)} tickers...")
    
    stock_data = {}
    failed_tickers = []
    
    # Calculate date ranges for different periods
    end_date = datetime.now()
    
    # For monthly data
    monthly_start = end_date - timedelta(days=35)
    
    # For multi-period data (fetch 5+ years to cover all periods)
    multi_period_start = end_date - timedelta(days=365 * 6)  # 6 years to be safe
    
    # Calculate YTD start date (January 1st of current year)
    ytd_start = datetime(end_date.year, 1, 1)
    
    print(f"Fetching historical data from {multi_period_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"  [{i}/{len(tickers)}] Fetching {ticker}...", end=" ")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Fetch extended historical data for multi-period analysis
            hist_full = stock.history(start=multi_period_start, end=end_date)
            
            if hist_full.empty:
                print("❌ No data")
                failed_tickers.append(ticker)
                continue
            
            # Convert timezone-aware index to timezone-naive for comparisons
            # yfinance returns timezone-aware datetimes (America/New_York)
            if hist_full.index.tz is not None:
                hist_full.index = hist_full.index.tz_localize(None)
            
            # Get monthly data (last month)
            month_data = hist_full[
                (hist_full.index.month == end_date.month) & 
                (hist_full.index.year == end_date.year)
            ]
            
            # If current month has less than 5 trading days, use previous month
            if len(month_data) < 5:
                prev_month_date = end_date.replace(day=1) - timedelta(days=1)
                month_data = hist_full[
                    (hist_full.index.month == prev_month_date.month) & 
                    (hist_full.index.year == prev_month_date.year)
                ]
            
            if month_data.empty:
                print("❌ No monthly data")
                failed_tickers.append(ticker)
                continue
            
            # Calculate monthly metrics
            monthly_open = float(month_data.iloc[0]['Open'])
            monthly_close = float(month_data.iloc[-1]['Close'])
            monthly_high = float(month_data['High'].max())
            monthly_low = float(month_data['Low'].min())
            monthly_volume = int(month_data['Volume'].sum())
            
            # Calculate percentage change
            if monthly_open > 0:
                monthly_pct_change = ((monthly_close - monthly_open) / monthly_open) * 100
            else:
                monthly_pct_change = 0.0
            
            # Get current/latest data
            latest = hist_full.iloc[-1]
            
            # Calculate additional monthly metrics
            monthly_avg_price = float(month_data['Close'].mean())
            monthly_volatility = float(month_data['Close'].std())
            trading_days = len(month_data)
            
            # Calculate multi-period metrics
            # Define period start dates
            periods = {
                '3mo': end_date - timedelta(days=90),
                '6mo': end_date - timedelta(days=180),
                'ytd': ytd_start,
                '1yr': end_date - timedelta(days=365),
                '5yr': end_date - timedelta(days=365 * 5),
            }
            
            # Calculate metrics for each period
            period_metrics = {}
            for period_name, period_start in periods.items():
                period_data = hist_full[hist_full.index >= period_start]
                metrics = calculate_period_metrics(period_data, period_name)
                period_metrics.update(metrics)
            
            stock_data[ticker] = {
                'ticker': ticker,
                'company_name': info.get('longName', info.get('shortName', 'N/A')),
                
                # Current/Latest data
                'current_price': float(latest['Close']),
                'current_volume': int(latest['Volume']),
                
                # Monthly activity metrics
                'monthly_open': monthly_open,
                'monthly_close': monthly_close,
                'monthly_high': monthly_high,
                'monthly_low': monthly_low,
                'monthly_volume': monthly_volume,
                'monthly_pct_change': round(monthly_pct_change, 2),
                'monthly_avg_price': round(monthly_avg_price, 2),
                'monthly_volatility': round(monthly_volatility, 2),
                'trading_days_in_month': trading_days,
                
                # Multi-period performance metrics
                **period_metrics,
                
                # Company information
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange'),
                
                # Additional metrics
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                
                # Metadata
                'snapshot_date': datetime.now().isoformat(),
                'data_start_date': month_data.index[0].isoformat(),
                'data_end_date': month_data.index[-1].isoformat(),
            }
            print("✓")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            failed_tickers.append(ticker)
            continue
    
    print(f"\nSuccessfully fetched data for {len(stock_data)}/{len(tickers)} tickers")
    if failed_tickers:
        print(f"Failed tickers: {', '.join(failed_tickers)}")
    
    return stock_data


def create_snapshot(company_df: pd.DataFrame, stock_data: Dict[str, Dict]) -> pd.DataFrame:
    """Combine company data with stock data to create snapshot."""
    print("\nCreating snapshot...")
    
    # Convert stock data to DataFrame
    stock_df = pd.DataFrame.from_dict(stock_data, orient='index')
    
    # Determine ticker column name (use YahooSymbolClean, fallback to symbol)
    ticker_col = None
    if 'YahooSymbolClean' in company_df.columns:
        ticker_col = 'YahooSymbolClean'
    elif 'symbol' in company_df.columns:
        ticker_col = 'symbol'
    elif 'ticker' in company_df.columns:
        ticker_col = 'ticker'
    
    # Merge with company data if ticker column exists
    if ticker_col:
        # Create a temporary column in stock_df to match company_df ticker column
        stock_df_copy = stock_df.copy()
        stock_df_copy[ticker_col] = stock_df_copy['ticker']
        
        snapshot_df = company_df.merge(
            stock_df_copy, 
            on=ticker_col, 
            how='left',
            suffixes=('_original', '_current')
        )
    else:
        # If no ticker column, just use stock data
        snapshot_df = stock_df
    
    print(f"Snapshot created with {len(snapshot_df)} records")
    return snapshot_df


def save_snapshot_to_github(snapshot_df: pd.DataFrame) -> str:
    """
    Save snapshot to GitHub repository in year-based folder structure.
    Creates folder like: snapshots/2025/snapshot_2025-10-08.parquet
    """
    # Get current date
    now = datetime.now()
    year = now.strftime("%Y")
    date_str = now.strftime("%Y-%m-%d")
    
    # Create year-based directory structure
    snapshots_dir = Path("snapshots") / year
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with date
    filename = f"snapshot_{date_str}.parquet"
    filepath = snapshots_dir / filename
    
    # Save to parquet
    snapshot_df.to_parquet(filepath, index=False)
    print(f"\nSnapshot saved to GitHub path: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")
    
    return str(filepath)


def save_snapshot_local(snapshot_df: pd.DataFrame) -> str:
    """Save snapshot to local snapshots directory with timestamp (for pCloud upload)."""
    # Create snapshots directory
    snapshots_dir = Path("snapshots")
    snapshots_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"geospatial_stocks_snapshot_{timestamp}.parquet"
    filepath = snapshots_dir / filename
    
    # Save to parquet
    snapshot_df.to_parquet(filepath, index=False)
    print(f"\nLocal snapshot saved to: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")
    
    return str(filepath)


def commit_and_push_to_github(filepath: str) -> bool:
    """
    Commit and push the snapshot file to GitHub repository.
    Uses git commands with GITHUB_TOKEN for authentication.
    """
    print(f"\nCommitting and pushing to GitHub...")
    
    try:
        # Configure git user (required for commits)
        subprocess.run(
            ["git", "config", "user.name", "Monthly Snapshot Bot"],
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.email", "snapshot-bot@geospatialfm.com"],
            check=True,
            capture_output=True
        )
        
        # Add the snapshot file
        print(f"  Adding {filepath}...")
        subprocess.run(
            ["git", "add", filepath],
            check=True,
            capture_output=True
        )
        
        # Commit the changes
        commit_message = f"Add monthly snapshot for {datetime.now().strftime('%Y-%m-%d')}"
        print(f"  Committing: {commit_message}")
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            check=True,
            capture_output=True
        )
        
        # Push to GitHub
        # Use GITHUB_TOKEN if available, otherwise use default authentication
        github_token = os.environ.get('GITHUB_TOKEN')
        if github_token:
            print("  Pushing to GitHub using GITHUB_TOKEN...")
            # Get the current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True
            )
            current_branch = result.stdout.strip()
            
            # Get the remote URL and modify it to include token
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                check=True,
                capture_output=True,
                text=True
            )
            remote_url = result.stdout.strip()
            
            # Replace https:// with https://token@
            if remote_url.startswith("https://"):
                authenticated_url = remote_url.replace(
                    "https://",
                    f"https://{github_token}@"
                )
                subprocess.run(
                    ["git", "push", authenticated_url, current_branch],
                    check=True,
                    capture_output=True
                )
            else:
                # Fallback to regular push
                subprocess.run(
                    ["git", "push", "origin", current_branch],
                    check=True,
                    capture_output=True
                )
        else:
            print("  Pushing to GitHub using default authentication...")
            subprocess.run(
                ["git", "push"],
                check=True,
                capture_output=True
            )
        
        print("  ✓ Successfully committed and pushed to GitHub!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Git operation failed: {e}")
        print(f"  stdout: {e.stdout.decode() if e.stdout else 'N/A'}")
        print(f"  stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def upload_to_pcloud(filepath: str, upload_url: str) -> bool:
    """Upload snapshot file to pCloud public upload link."""
    print(f"\nUploading to pCloud...")
    
    try:
        # Extract the upload code from the URL
        if "code=" in upload_url:
            code = upload_url.split("code=")[1].split("&")[0]
        else:
            raise ValueError("Invalid pCloud upload URL format")
        
        # pCloud public upload API endpoint
        api_url = f"https://api.pcloud.com/uploadtolink"
        
        # Prepare the file
        filename = Path(filepath).name
        
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f, 'application/octet-stream')}
            params = {'code': code}
            
            print(f"  Uploading {filename}...")
            response = requests.post(api_url, params=params, files=files, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('result') == 0:
                    print("  ✓ Upload successful!")
                    print(f"  File ID: {result.get('metadata', [{}])[0].get('fileid', 'N/A')}")
                    return True
                else:
                    print(f"  ❌ Upload failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"  ❌ HTTP Error {response.status_code}: {response.text}")
                return False
                
    except Exception as e:
        print(f"  ❌ Upload error: {str(e)}")
        return False


def main():
    """Main execution function."""
    print("=" * 60)
    print("Geospatial Company Stock Snapshot Capture")
    print("With Monthly Activity & Multi-Period Performance Metrics")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    try:
        # 1. Load company data
        company_df = load_company_data()
        
        # 2. Extract tickers (use YahooSymbolClean, fallback to symbol, then ticker)
        ticker_col = None
        if 'YahooSymbolClean' in company_df.columns:
            ticker_col = 'YahooSymbolClean'
        elif 'symbol' in company_df.columns:
            ticker_col = 'symbol'
        elif 'ticker' in company_df.columns:
            ticker_col = 'ticker'
        elif 'Ticker' in company_df.columns:
            ticker_col = 'Ticker'
        else:
            # Try to find any column that might contain tickers
            print("\nAvailable columns:", company_df.columns.tolist())
            raise ValueError("Could not find ticker column in data")
        
        print(f"Using ticker column: {ticker_col}")
        tickers = company_df[ticker_col].dropna().unique().tolist()
        print(f"Found {len(tickers)} unique tickers")
        
        # 3. Fetch monthly stock data with activity metrics
        stock_data = fetch_monthly_stock_data(tickers)
        
        if not stock_data:
            print("\n❌ No stock data fetched. Exiting.")
            sys.exit(1)
        
        # 4. Create snapshot
        snapshot_df = create_snapshot(company_df, stock_data)
        
        # 5. Save snapshot to GitHub repository
        github_filepath = save_snapshot_to_github(snapshot_df)
        
        # 6. Commit and push to GitHub
        github_success = commit_and_push_to_github(github_filepath)
        
        # 7. Save local copy for pCloud upload
        local_filepath = save_snapshot_local(snapshot_df)
        
        # 8. Upload to pCloud
        pcloud_success = False
        upload_url = os.environ.get('PCLOUD_UPLOAD_URL')
        if upload_url:
            pcloud_success = upload_to_pcloud(local_filepath, upload_url)
        else:
            print("\n⚠ PCLOUD_UPLOAD_URL not set, skipping pCloud upload")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"✓ Snapshot created with {len(snapshot_df)} records")
        print(f"✓ Monthly metrics calculated for {len(stock_data)} stocks")
        print(f"{'✓' if github_success else '❌'} GitHub storage: {github_filepath}")
        if upload_url:
            print(f"{'✓' if pcloud_success else '❌'} pCloud upload: {local_filepath}")
        print("=" * 60)
        
        if github_success:
            print("\n✓ Snapshot captured and stored successfully!")
        else:
            print("\n⚠ Snapshot captured but GitHub storage failed")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
