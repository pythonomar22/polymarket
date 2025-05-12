# config.py
import os
from pathlib import Path
from datetime import date, time, timedelta

# --- Essential Paths ---
# Assumes your scripts are in a directory adjacent to 'binance_market_data'
# Adjust BASE_DIR if your structure is different
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR_BASE = BASE_DIR / "binance_market_data" / "spot" / "daily"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

# --- Data Sources ---
KLINE_INPUT_DIR = RAW_DATA_DIR_BASE / "klines_1m"
AGGTRADE_INPUT_DIR = RAW_DATA_DIR_BASE / "aggTrades" # Added for potential future use

# --- Data Parameters ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "1m"
START_DATE = date(2021, 5, 9) # Match your downloaded data
END_DATE = date(2025, 5, 9)   # Match your downloaded data

# --- Contract & Timezone ---
TARGET_TZ = 'US/Eastern' # Polymarket resolution timezone
RESOLUTION_TIME = time(12, 0) # Noon ET

# --- Kline Column Definitions (Based on Binance API Docs for Spot Klines) ---
# These are the standard columns provided in the downloaded CSVs
KLINE_RAW_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

# Columns we intend to keep and potentially rename/use
KLINE_USEFUL_COLUMNS = {
    'open_time': 'open_time', # Keep for indexing and timezone conversion
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
    'taker_buy_base_asset_volume': 'taker_buy_vol' # Renamed for brevity
}

# --- AggTrade Column Definitions (Based on Binance API Docs) ---
AGGTRADE_COLUMNS = [
    'agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id',
    'timestamp', 'is_buyer_maker', 'is_best_match'
]

# --- Feature Engineering Parameters ---
FEATURE_ROLLING_WINDOWS = ['1H', '4H', '12H', '24H'] # Pandas offset strings

# --- Create Processed Data Directory ---
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Helper for date iteration
def daterange(start_date, end_date):
    """Generates a sequence of dates from start_date to end_date, inclusive."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)