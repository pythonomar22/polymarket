# process_klines.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import timedelta # Import timedelta here

# Import configuration variables and helper
import config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_daily_kline(filepath: Path) -> pd.DataFrame | None:
    """Loads a single daily kline CSV, performs initial cleaning."""
    try:
        df = pd.read_csv(filepath, header=None, names=config.KLINE_RAW_COLUMNS)

        # Basic validation
        if df.empty:
            logger.warning(f"Empty file: {filepath}")
            return None
        if len(df.columns) != len(config.KLINE_RAW_COLUMNS):
             logger.error(f"Unexpected column count in {filepath}. Skipping.")
             return None

        # Convert open_time to datetime (UTC initially)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)

        # Select and rename useful columns
        df = df[list(config.KLINE_USEFUL_COLUMNS.keys())].rename(columns=config.KLINE_USEFUL_COLUMNS)

        # Set index
        df = df.set_index('open_time')

        # Convert numeric columns to float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_vol']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

        # Drop rows with NaNs that might result from conversion errors
        df.dropna(subset=numeric_cols, inplace=True)

        return df

    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        return None

def process_symbol_klines(symbol: str):
    """Loads, concatenates, cleans, and saves kline data for a single symbol."""
    logger.info(f"--- Processing Kline data for symbol: {symbol} ---")
    symbol_kline_dir = config.KLINE_INPUT_DIR / symbol
    output_file = config.PROCESSED_DATA_DIR / f"{symbol}_{config.INTERVAL}_ET.parquet"

    if output_file.exists():
        logger.info(f"Processed file already exists: {output_file}, skipping.")
        return

    all_dfs = []
    dates_to_process = list(config.daterange(config.START_DATE, config.END_DATE)) # Use helper from config

    for target_date in tqdm(dates_to_process, desc=f"{symbol} Klines", unit="day"):
        filename = f"{symbol}-{config.INTERVAL}-{target_date.strftime('%Y-%m-%d')}.csv"
        filepath = symbol_kline_dir / filename

        daily_df = load_daily_kline(filepath)
        if daily_df is not None:
            all_dfs.append(daily_df)

    if not all_dfs:
        logger.warning(f"No valid data found for symbol {symbol}. Cannot proceed.")
        return

    # Concatenate all daily dataframes
    logger.info(f"Concatenating {len(all_dfs)} daily files for {symbol}...")
    full_df = pd.concat(all_dfs)

    # Sort by time (essential!)
    full_df.sort_index(inplace=True)

    # Check for duplicate indices (can happen with overlapping data or errors)
    duplicates = full_df.index.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate timestamps for {symbol}. Keeping first occurrence.")
        full_df = full_df[~full_df.index.duplicated(keep='first')]

    # Convert index to target timezone (US/Eastern)
    logger.info(f"Converting index to {config.TARGET_TZ} for {symbol}...")
    try:
        # Ensure index is timezone-aware (should be UTC from loading)
        if full_df.index.tz is None:
             logger.warning(f"Index for {symbol} was not timezone-aware. Assuming UTC before conversion.")
             full_df.index = full_df.index.tz_localize('UTC')
        full_df.index = full_df.index.tz_convert(config.TARGET_TZ)
    except Exception as e:
        logger.error(f"Error converting timezone for {symbol}: {e}")
        return # Stop processing this symbol if timezone conversion fails

    # Check for time gaps (optional but good practice)
    time_diffs = full_df.index.to_series().diff().value_counts()
    expected_diff = pd.Timedelta(minutes=1)
    if (time_diffs.index != expected_diff).any():
        logger.warning(f"Potential time gaps found for {symbol}. Most common diffs:\n{time_diffs.head()}")

    # Save the processed data
    logger.info(f"Saving processed data for {symbol} to {output_file}...")
    try:
        full_df.to_parquet(output_file, engine='pyarrow') # Or 'fastparquet'
        logger.info(f"Successfully saved {output_file}")
    except Exception as e:
        logger.error(f"Failed to save {output_file}: {e}")


if __name__ == "__main__":
    logger.info("Starting Kline data processing script...")
    for symbol in config.SYMBOLS:
        process_symbol_klines(symbol)
    logger.info("--- Kline data processing finished. ---")