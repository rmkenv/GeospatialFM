import pandas as pd
import sys

def convert_cleaned_csv_to_parquet():
    url = "https://raw.githubusercontent.com/rmkenv/GeospatialFM/main/geospatial_companies_cleaned.csv"
    output = "geospatial_companies_cleaned.parquet"
    
    try:
        print(f"Reading CSV from: {url}")
        df = pd.read_csv(url)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\nWriting Parquet to: {output}")
        df.to_parquet(output, index=False, compression="snappy")
        
        # Verify file size
        import os
        size_mb = os.path.getsize(output) / (1024 * 1024)
        print(f"✓ Success! File size: {size_mb:.2f} MB")
        
        # Quick comparison
        df_test = pd.read_parquet(output)
        print(f"✓ Verified: read back {len(df_test)} rows")
        
        # Show preview
        print("\nPreview of data:")
        print(df.head(3))
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    convert_cleaned_csv_to_parquet()
