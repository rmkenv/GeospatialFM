
#!/usr/bin/env python3
"""
Monthly Geospatial Company Stock Snapshot Capture Script

This script:
1. Reads the geospatial companies data from parquet file
2. Fetches current stock data using yfinance
3. Creates a timestamped snapshot
4. Uploads the snapshot to pCloud public upload link
"""

import os
import sys
from datetime import datetime
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


def fetch_stock_data(tickers: List[str]) -> Dict[str, Dict]:
    """Fetch current stock data for given tickers using yfinance."""
    print(f"\nFetching stock data for {len(tickers)} tickers...")
    
    stock_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"  [{i}/{len(tickers)}] Fetching {ticker}...", end=" ")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="5d")
            
            if hist.empty:
                print("❌ No data")
                failed_tickers.append(ticker)
                continue
            
            latest = hist.iloc[-1]
            
            stock_data[ticker] = {
                'ticker': ticker,
                'company_name': info.get('longName', info.get('shortName', 'N/A')),
                'current_price': float(latest['Close']),
                'volume': int(latest['Volume']),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'snapshot_date': datetime.now().isoformat(),
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


def save_snapshot(snapshot_df: pd.DataFrame) -> str:
    """Save snapshot to parquet file with timestamp."""
    # Create snapshots directory
    snapshots_dir = Path("snapshots")
    snapshots_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"geospatial_stocks_snapshot_{timestamp}.parquet"
    filepath = snapshots_dir / filename
    
    # Save to parquet
    snapshot_df.to_parquet(filepath, index=False)
    print(f"\nSnapshot saved to: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")
    
    return str(filepath)


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
        
        # 3. Fetch stock data
        stock_data = fetch_stock_data(tickers)
        
        if not stock_data:
            print("\n❌ No stock data fetched. Exiting.")
            sys.exit(1)
        
        # 4. Create snapshot
        snapshot_df = create_snapshot(company_df, stock_data)
        
        # 5. Save snapshot
        filepath = save_snapshot(snapshot_df)
        
        # 6. Upload to pCloud
        upload_url = os.environ.get('PCLOUD_UPLOAD_URL')
        if upload_url:
            success = upload_to_pcloud(filepath, upload_url)
            if success:
                print("\n✓ Snapshot captured and uploaded successfully!")
            else:
                print("\n⚠ Snapshot captured but upload failed")
                sys.exit(1)
        else:
            print("\n⚠ PCLOUD_UPLOAD_URL not set, skipping upload")
            print("✓ Snapshot captured successfully!")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
