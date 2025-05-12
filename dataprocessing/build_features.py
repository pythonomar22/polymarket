# build_features.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings

# Import configuration variables and helper
import config

# Ignore SettingWithCopyWarning, use .loc appropriately
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore warnings from rolling apply

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(symbol: str) -> pd.DataFrame | None:
    """Loads the processed Parquet file for a symbol."""
    filepath = config.PROCESSED_DATA_DIR / f"{symbol}_{config.INTERVAL}_ET.parquet"
    if not filepath.exists():
        logger.error(f"Processed data file not found: {filepath}. Run process_klines.py first.")
        return None
    try:
        logger.info(f"Loading processed data from {filepath}...")
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded data for {symbol} with shape {df.shape}")
        # Basic check
        if not isinstance(df.index, pd.DatetimeIndex):
             logger.error("Index is not DatetimeIndex after loading.")
             return None
        if df.index.tz is None or str(df.index.tz) != config.TARGET_TZ:
             logger.error(f"Index timezone is incorrect ({df.index.tz}) after loading. Expected {config.TARGET_TZ}")
             return None
        return df
    except Exception as e:
        logger.error(f"Error loading processed data {filepath}: {e}")
        return None

def calculate_target(df: pd.DataFrame, symbol: str) -> pd.Series | None:
    """Calculates the noon-to-noon target variable."""
    logger.info(f"Calculating target variable for {symbol}...")
    try:
        # Filter for rows exactly at noon ET
        noon_candles = df[df.index.time == config.RESOLUTION_TIME].copy()
        if noon_candles.empty:
            logger.warning(f"No noon candles found for {symbol}. Cannot calculate target.")
            return None

        noon_closes = noon_candles['close']

        # Calculate difference from the previous noon's close
        price_diff = noon_closes.diff() # Difference between noon[D] and noon[D-1]

        # Define target: 1 if price went up, 0 if down or equal
        # Polymarket resolves 50/50 on equality, but for binary models,
        # we usually map to 0 or 1. Mapping equal to 'down' (0) here.
        target = (price_diff > 0).astype(int)

        # Handle the first entry which will have NaN difference
        target = target.iloc[1:] # Drop the first NaN value

        target.name = f"{symbol}_target_up"
        logger.info(f"Target calculated for {symbol}. Shape: {target.shape}")
        return target

    except Exception as e:
        logger.error(f"Error calculating target for {symbol}: {e}")
        return None


def calculate_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Calculates features based on kline data."""
    logger.info(f"Calculating features for {symbol}...")
    try:
        features = pd.DataFrame(index=df.index) # Start with an empty DataFrame with the same index

        # --- Log Returns ---
        # Use fillna(0) for the first return, common practice
        features['log_ret_1m'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

        # --- Rolling Features ---
        for window in config.FEATURE_ROLLING_WINDOWS:
            # Use closed='left' to ensure data used is strictly before the current timestamp
            roll_obj = df['close'].rolling(window, closed='left')
            roll_ret_obj = features['log_ret_1m'].rolling(window, closed='left')
            roll_vol_obj = df['volume'].rolling(window, closed='left')
            roll_tbuy_obj = df['taker_buy_vol'].rolling(window, closed='left')

            # Momentum (Rolling Return Sum)
            features[f'ret_{window}'] = roll_ret_obj.sum()

            # Volatility (Rolling Std Dev of Log Returns)
            features[f'vol_{window}'] = roll_ret_obj.std()

            # Price vs Moving Average (using simple mean for MA)
            features[f'price_ma_ratio_{window}'] = df['close'] / roll_obj.mean()

            # Volume Sum
            features[f'vol_sum_{window}'] = roll_vol_obj.sum()

             # Taker Buy Volume Ratio (handle potential division by zero)
            vol_sum = roll_vol_obj.sum()
            tbuy_sum = roll_tbuy_obj.sum()
            features[f'tbuy_ratio_{window}'] = (tbuy_sum / vol_sum).replace([np.inf, -np.inf], np.nan).fillna(0.5) # Fill NaN with 0.5? Or 0?

        # --- Time Features ---
        features['dayofweek'] = df.index.dayofweek
        features['hourofday'] = df.index.hour # Hour in ET

        # Drop the initial rows that have NaNs due to rolling windows
        # Find the length of the longest window (e.g., '24H' = 1440 minutes)
        max_window_minutes = max(int(pd.Timedelta(w).total_seconds() / 60) for w in config.FEATURE_ROLLING_WINDOWS)
        features = features.iloc[max_window_minutes:]

        logger.info(f"Features calculated for {symbol}. Shape before noon selection: {features.shape}")
        return features

    except Exception as e:
        logger.error(f"Error calculating features for {symbol}: {e}")
        return None


def create_modeling_dataset(symbol: str, features_df: pd.DataFrame, target_series: pd.Series) -> pd.DataFrame | None:
    """Aligns features and target, creates the final dataset."""
    logger.info(f"Aligning features and target for {symbol}...")
    try:
        # --- Critical Alignment Step ---
        # We want features available *before* the start of the period the target measures.
        # Target for Day D (noon D-1 to noon D) is indexed at noon D.
        # We need features calculated using data *up to* noon D-1.

        # 1. Select feature rows corresponding to noon ET timestamps
        noon_features = features_df[features_df.index.time == config.RESOLUTION_TIME].copy()

        # 2. Shift these features forward by one day.
        # Now, the features calculated at noon D-1 are indexed at noon D.
        aligned_features = noon_features.shift(1, freq='D') # Shift index by 1 day

        # 3. Merge with the target Series. Target is already indexed at noon D.
        # Use an inner join to ensure we only have timestamps where both exist.
        modeling_df = pd.merge(aligned_features, target_series, left_index=True, right_index=True, how='inner')

        # Drop any remaining NaNs (e.g., from the shift or feature calculation)
        modeling_df.dropna(inplace=True)

        logger.info(f"Final modeling dataset created for {symbol}. Shape: {modeling_df.shape}")

        if modeling_df.empty:
             logger.warning(f"Modeling dataset for {symbol} is empty after alignment and NaN drop.")
             return None

        return modeling_df

    except Exception as e:
        logger.error(f"Error creating modeling dataset for {symbol}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Feature Engineering script...")

    all_processed_data = {}
    for symbol in config.SYMBOLS:
        df = load_processed_data(symbol)
        if df is not None:
            all_processed_data[symbol] = df

    if not all_processed_data:
        logger.critical("No processed data loaded. Exiting.")
        exit()

    # --- Process individual symbols ---
    for symbol, df in all_processed_data.items():
        target = calculate_target(df, symbol)
        if target is None:
            continue # Skip if target calculation failed

        features = calculate_features(df, symbol)
        if features is None:
            continue # Skip if feature calculation failed

        modeling_df = create_modeling_dataset(symbol, features, target)

        if modeling_df is not None:
            output_path = config.PROCESSED_DATA_DIR / f"{symbol}_modeling_data.parquet"
            try:
                modeling_df.to_parquet(output_path)
                logger.info(f"Successfully saved modeling data to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save modeling data for {symbol}: {e}")

    # --- Process ETH/BTC Ratio ---
    logger.info("--- Processing ETH/BTC Ratio ---")
    if "ETHUSDT" in all_processed_data and "BTCUSDT" in all_processed_data:
        eth_df = all_processed_data["ETHUSDT"]
        btc_df = all_processed_data["BTCUSDT"]

        # Align ETH and BTC data by index (inner join handles missing times)
        logger.info("Aligning ETH and BTC data for ratio calculation...")
        merged_df = pd.merge(eth_df[['close']].rename(columns={'close':'eth_close'}),
                             btc_df[['close']].rename(columns={'close':'btc_close'}),
                             left_index=True, right_index=True, how='inner')

        # Calculate ETH/BTC ratio
        merged_df['eth_btc_ratio'] = merged_df['eth_close'] / merged_df['btc_close']

        # Create a pseudo-DataFrame for ratio feature calculation
        ratio_df_for_features = pd.DataFrame({
            'close': merged_df['eth_btc_ratio'],
            # Add 'volume' etc. if needed for ratio features (e.g., use sum of ETH+BTC volume?)
            # For now, only use price-based features for the ratio
            'volume': eth_df.reindex(merged_df.index)['volume'].fillna(0) + btc_df.reindex(merged_df.index)['volume'].fillna(0), # Example: sum volumes
            'taker_buy_vol': eth_df.reindex(merged_df.index)['taker_buy_vol'].fillna(0) + btc_df.reindex(merged_df.index)['taker_buy_vol'].fillna(0) # Example: sum taker buys

        })
        ratio_df_for_features = ratio_df_for_features.dropna(subset=['close']) # Ensure ratio is valid

        # Calculate target for the ratio
        ratio_target = calculate_target(ratio_df_for_features.rename(columns={'close': 'close'}), "ETHBTC") # Pass 'close' column name
        if ratio_target is not None:
            # Calculate features for the ratio
            ratio_features = calculate_features(ratio_df_for_features, "ETHBTC") # Using the pseudo-df
            if ratio_features is not None:
                 # Align and create modeling dataset for the ratio
                 ratio_modeling_df = create_modeling_dataset("ETHBTC", ratio_features, ratio_target)

                 if ratio_modeling_df is not None:
                     output_path = config.PROCESSED_DATA_DIR / "ETHBTC_modeling_data.parquet"
                     try:
                         ratio_modeling_df.to_parquet(output_path)
                         logger.info(f"Successfully saved modeling data to {output_path}")
                     except Exception as e:
                          logger.error(f"Failed to save modeling data for ETHBTC: {e}")
            else:
                 logger.warning("Skipped ETHBTC modeling dataset creation due to feature calculation error.")
        else:
             logger.warning("Skipped ETHBTC modeling dataset creation due to target calculation error.")
    else:
        logger.warning("ETHUSDT or BTCUSDT processed data not available. Skipping ETH/BTC ratio.")


    # --- (Optional) Add AggTrades processing here if needed ---
    # Follow a similar pattern: load, consolidate, convert time, aggregate features (e.g., order flow imbalance per minute),
    # save processed aggtrade features, then merge them into the modeling datasets based on timestamp.
    # This is significantly more complex and memory-intensive.

    logger.info("--- Feature Engineering script finished. ---")
    logger.info(f"Final modeling datasets saved in: {config.PROCESSED_DATA_DIR}")