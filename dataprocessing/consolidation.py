# consolidation_with_aggtrades.py
import pandas as pd
import glob
import os
import logging
from datetime import datetime
import pytz # For robust timezone handling
from tqdm import tqdm

# --- Configuration ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] # Add more if you downloaded them

# Input Directories
BASE_INPUT_DIR_KLINES = "/Users/omarabul-hassan/Desktop/projects/trading/binance_market_data/spot/daily/klines_1m"
BASE_INPUT_DIR_AGGTRADES = "/Users/omarabul-hassan/Desktop/projects/trading/binance_market_data/spot/daily/aggTrades"

# Output Directories (for consolidated 1-minute data)
BASE_OUTPUT_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/binance_market_data/spot/consolidated"
# Subdirectories will be created: e.g., consolidated/klines_1m_aggfeatures/

# Kline Column Names
KLINE_COLUMN_NAMES = [
    "kline_open_time", "open", "high", "low", "close", "volume",
    "kline_close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
]
KLINE_NUMERIC_COLS = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                      'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

# AggTrades Column Names (as per Binance documentation)
AGGTRADE_COLUMN_NAMES = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "timestamp", "is_buyer_maker", "is_best_match"
]
AGGTRADE_DTYPES = {
    "agg_trade_id": "int64",
    "price": "float64",
    "quantity": "float64",
    "first_trade_id": "int64",
    "last_trade_id": "int64",
    "timestamp": "int64", # Will be converted to datetime
    "is_buyer_maker": "bool",
    "is_best_match": "bool"
}
AGGTRADE_NUMERIC_COLS_FOR_AGG = ['price', 'quantity'] # For aggregation

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def convert_timestamp_series(ts_series_input, series_name, symbol_for_log):
    """
    Converts a series of timestamps (ms or us) to UTC datetime objects.
    Handles potential NaNs and logs errors.
    """
    numeric_series = pd.to_numeric(ts_series_input, errors='coerce')
    converted_times = pd.Series([pd.NaT] * len(numeric_series), index=numeric_series.index, dtype='datetime64[ns, UTC]')

    is_microsecond_scale = (numeric_series >= 900_000_000_000_000) 

    if is_microsecond_scale.any():
        converted_times.loc[is_microsecond_scale] = pd.to_datetime(
            numeric_series[is_microsecond_scale], unit='us', utc=True, errors='coerce'
        )

    is_still_nat = converted_times.isnull()
    if is_still_nat.any():
        converted_times.loc[is_still_nat] = pd.to_datetime(
            numeric_series[is_still_nat], unit='ms', utc=True, errors='coerce'
        )

    num_failed = converted_times.isnull().sum()
    if num_failed > 0:
        logger.warning(f"{num_failed} timestamps in '{series_name}' for symbol '{symbol_for_log}' could not be converted and are NaT.")
    return converted_times

# --- Main Processing Logic ---
if __name__ == "__main__":
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
        logger.info(f"Created base output directory: {BASE_OUTPUT_DIR}")

    output_dir_consolidated_klines = os.path.join(BASE_OUTPUT_DIR, "klines_1m")
    output_dir_consolidated_aggfeatures = os.path.join(BASE_OUTPUT_DIR, "aggfeatures_1m")

    for d in [output_dir_consolidated_klines, output_dir_consolidated_aggfeatures]:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Created output subdirectory: {d}")


    for symbol in SYMBOLS:
        logger.info(f"\n" + "="*30 + f" Processing Symbol: {symbol} " + "="*30)

        # --- 1. Consolidate Klines ---
        logger.info(f"--- Consolidating KLINE data for {symbol} ---")
        kline_symbol_input_path = os.path.join(BASE_INPUT_DIR_KLINES, symbol)
        kline_output_file_path = os.path.join(output_dir_consolidated_klines, f"{symbol}_1m_klines_consolidated.parquet")

        if os.path.exists(kline_output_file_path):
            logger.info(f"Kline consolidated file already exists, skipping: {kline_output_file_path}")
        elif not os.path.isdir(kline_symbol_input_path):
            logger.warning(f"Kline directory not found for symbol {symbol} at {kline_symbol_input_path}. Skipping kline consolidation.")
        else:
            all_kline_csv_files = sorted(glob.glob(os.path.join(kline_symbol_input_path, f"{symbol}-1m-*.csv")))
            if not all_kline_csv_files:
                logger.warning(f"No kline CSV files found for {symbol} in {kline_symbol_input_path}")
            else:
                df_kline_list = []
                for f_path in tqdm(all_kline_csv_files, desc=f"Reading Klines {symbol}", unit="file"):
                    if os.path.getsize(f_path) == 0: continue
                    try:
                        daily_df = pd.read_csv(f_path, header=None, names=KLINE_COLUMN_NAMES)
                        df_kline_list.append(daily_df)
                    except Exception as e: logger.error(f"Error reading kline CSV {f_path}: {e}")

                if not df_kline_list:
                    logger.warning(f"No kline dataframes created for {symbol}.")
                else:
                    consolidated_klines_df = pd.concat(df_kline_list, ignore_index=True)
                    logger.info(f"Consolidated {len(consolidated_klines_df)} kline rows for {symbol}.")

                    consolidated_klines_df["timestamp_utc"] = convert_timestamp_series(
                        consolidated_klines_df["kline_open_time"], "kline_open_time", symbol
                    )
                    initial_rows = len(consolidated_klines_df)
                    consolidated_klines_df.dropna(subset=["timestamp_utc"], inplace=True)
                    if initial_rows - len(consolidated_klines_df) > 0:
                        logger.warning(f"Dropped {initial_rows - len(consolidated_klines_df)} kline rows due to NaT timestamps for {symbol}.")

                    if consolidated_klines_df.empty:
                        logger.error(f"Kline DataFrame for {symbol} empty after NaT drop. Skipping save.")
                    else:
                        for col in KLINE_NUMERIC_COLS:
                            consolidated_klines_df[col] = pd.to_numeric(consolidated_klines_df[col], errors='coerce')
                        consolidated_klines_df.set_index("timestamp_utc", inplace=True)
                        consolidated_klines_df.sort_index(inplace=True)
                        consolidated_klines_df = consolidated_klines_df[~consolidated_klines_df.index.duplicated(keep='first')]

                        consolidated_klines_df.to_parquet(kline_output_file_path)
                        logger.info(f"Consolidated kline data for {symbol} saved to {kline_output_file_path}")

        # --- 2. Consolidate and Aggregate AggTrades to 1-minute intervals ---
        logger.info(f"--- Consolidating and Aggregating AGGTRADE data for {symbol} to 1-minute ---")
        aggtrade_symbol_input_path = os.path.join(BASE_INPUT_DIR_AGGTRADES, symbol)
        aggfeatures_output_file_path = os.path.join(output_dir_consolidated_aggfeatures, f"{symbol}_1m_aggfeatures_consolidated.parquet")

        if os.path.exists(aggfeatures_output_file_path):
            logger.info(f"Aggregated aggTrade feature file already exists, skipping: {aggfeatures_output_file_path}")
        elif not os.path.isdir(aggtrade_symbol_input_path):
            logger.warning(f"AggTrade directory not found for symbol {symbol} at {aggtrade_symbol_input_path}. Skipping aggTrade processing.")
        else:
            all_aggtrade_csv_files = sorted(glob.glob(os.path.join(aggtrade_symbol_input_path, f"{symbol}-aggTrades-*.csv")))
            if not all_aggtrade_csv_files:
                logger.warning(f"No aggTrade CSV files found for {symbol} in {aggtrade_symbol_input_path}")
            else:
                df_aggtrade_minute_list = []
                for f_path in tqdm(all_aggtrade_csv_files, desc=f"Processing AggTrades {symbol}", unit="day_file"):
                    if os.path.getsize(f_path) == 0:
                        logger.warning(f"Empty aggTrade file skipped: {f_path}")
                        continue
                    try:
                        daily_agg_df = pd.read_csv(f_path, header=None, names=AGGTRADE_COLUMN_NAMES, dtype=AGGTRADE_DTYPES)
                        if daily_agg_df.empty:
                            logger.warning(f"Empty DataFrame after reading aggTrade file: {f_path}")
                            continue

                        daily_agg_df["timestamp_utc"] = convert_timestamp_series(
                            daily_agg_df["timestamp"], "aggtrade_timestamp", symbol
                        )
                        daily_agg_df.dropna(subset=["timestamp_utc"], inplace=True)
                        if daily_agg_df.empty:
                            logger.warning(f"AggTrade data empty after timestamp conversion for file: {f_path}")
                            continue

                        daily_agg_df.set_index("timestamp_utc", inplace=True)
                        
                        def custom_agg(group):
                            res = {}
                            res['agg_taker_buy_qty_1m'] = group.loc[~group["is_buyer_maker"], "quantity"].sum()
                            res['agg_taker_sell_qty_1m'] = group.loc[group["is_buyer_maker"], "quantity"].sum()
                            res['agg_taker_buy_count_1m'] = group.loc[~group["is_buyer_maker"]].shape[0]
                            res['agg_taker_sell_count_1m'] = group.loc[group["is_buyer_maker"]].shape[0]
                            res['agg_total_qty_1m'] = group["quantity"].sum()
                            res['agg_total_trades_1m'] = group.shape[0]
                            res['agg_price_mean_1m'] = group["price"].mean()
                            res['agg_price_std_1m'] = group["price"].std()
                            res['agg_best_match_true_count_1m'] = group["is_best_match"].sum() 
                            return pd.Series(res)

                        minute_agg_df = daily_agg_df.resample('1min', label='left', closed='left').apply(custom_agg)
                        
                        cols_to_fill_zero = [
                            'agg_taker_buy_qty_1m', 'agg_taker_sell_qty_1m',
                            'agg_taker_buy_count_1m', 'agg_taker_sell_count_1m',
                            'agg_total_qty_1m', 'agg_total_trades_1m', 'agg_best_match_true_count_1m'
                        ]
                        for col in cols_to_fill_zero:
                            if col in minute_agg_df.columns:
                                minute_agg_df[col] = minute_agg_df[col].fillna(0)
                        
                        df_aggtrade_minute_list.append(minute_agg_df)

                    except pd.errors.EmptyDataError:
                        logger.warning(f"Empty aggTrade CSV (EmptyDataError): {f_path}")
                    except Exception as e:
                        logger.error(f"Error processing aggTrade file {f_path}: {e}", exc_info=True)

                if not df_aggtrade_minute_list:
                    logger.warning(f"No 1-minute aggregated aggTrade dataframes created for {symbol}.")
                else:
                    consolidated_aggfeatures_df = pd.concat(df_aggtrade_minute_list)
                    logger.info(f"Consolidated {len(consolidated_aggfeatures_df)} 1-minute aggTrade feature rows for {symbol}.")

                    consolidated_aggfeatures_df['agg_net_taker_qty_1m'] = consolidated_aggfeatures_df['agg_taker_buy_qty_1m'] - consolidated_aggfeatures_df['agg_taker_sell_qty_1m']
                    consolidated_aggfeatures_df['agg_taker_buy_sell_ratio_qty_1m'] = \
                        consolidated_aggfeatures_df['agg_taker_buy_qty_1m'] / (consolidated_aggfeatures_df['agg_taker_sell_qty_1m'] + 1e-9)
                    consolidated_aggfeatures_df['agg_taker_buy_sell_ratio_count_1m'] = \
                        consolidated_aggfeatures_df['agg_taker_buy_count_1m'] / (consolidated_aggfeatures_df['agg_taker_sell_count_1m'] + 1e-9)
                    consolidated_aggfeatures_df['agg_avg_taker_buy_size_1m'] = \
                        consolidated_aggfeatures_df['agg_taker_buy_qty_1m'] / (consolidated_aggfeatures_df['agg_taker_buy_count_1m'] + 1e-9)
                    consolidated_aggfeatures_df['agg_avg_taker_sell_size_1m'] = \
                        consolidated_aggfeatures_df['agg_taker_sell_qty_1m'] / (consolidated_aggfeatures_df['agg_taker_sell_count_1m'] + 1e-9)
                    if 'agg_total_trades_1m' in consolidated_aggfeatures_df.columns and 'agg_best_match_true_count_1m' in consolidated_aggfeatures_df.columns:
                        consolidated_aggfeatures_df['agg_pct_best_match_1m'] = \
                            consolidated_aggfeatures_df['agg_best_match_true_count_1m'] / (consolidated_aggfeatures_df['agg_total_trades_1m'] + 1e-9)
                    
                    consolidated_aggfeatures_df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
                    
                    consolidated_aggfeatures_df.sort_index(inplace=True)
                    consolidated_aggfeatures_df = consolidated_aggfeatures_df[~consolidated_aggfeatures_df.index.duplicated(keep='first')]

                    consolidated_aggfeatures_df.to_parquet(aggfeatures_output_file_path)
                    logger.info(f"Consolidated 1-minute aggTrade features for {symbol} saved to {aggfeatures_output_file_path}")

    logger.info("--- Data consolidation (Klines and AggTrades) script finished. ---")