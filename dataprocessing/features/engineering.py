# engineering_aggtrades.py
import pandas as pd
import numpy as np
import os
import logging
import pytz # For timezone handling
from datetime import time as dt_time, datetime, timedelta
from tqdm import tqdm

# --- Configuration ---
SYMBOLS_TO_PROCESS = ["BTCUSDT"] # Start with BTCUSDT as you have aggTrades for it
# SYMBOLS_TO_PROCESS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] # Later, include others if you process their aggtrades

# Consolidated Data Directories
CONSOLIDATED_BASE_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/binance_market_data/spot/consolidated"
CONSOLIDATED_KLINES_DIR = os.path.join(CONSOLIDATED_BASE_DIR, "klines_1m")
CONSOLIDATED_AGGFEATURES_DIR = os.path.join(CONSOLIDATED_BASE_DIR, "aggfeatures_1m")

# Output directory for features and labels
FEATURE_OUTPUT_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/binance_market_data/features_labels_v3" # New version

EASTERN_TZ = pytz.timezone('America/New_York')
CONTRACT_TIME_ET = dt_time(12, 0, 0) # Noon ET

TARGET_RETURN_THRESHOLD = 0.000 # Price change threshold for "Up"

ROLLING_WINDOWS_MINUTES = [
    60, 240, 720, 1440, 1440 * 3, # 1h, 4h, 12h, 1d, 3d
]
SHORT_ROLLING_WINDOWS_MINUTES = [5, 15, 30] # For short-term return snapshots

# Technical Indicator Parameters
ATR_PERIOD = 14
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
STOCH_K_PERIOD = 14; STOCH_K_SMOOTH = 3; STOCH_D_SMOOTH = 3
AUTOCORR_LAGS = [1, 2, 3, 5, 10, 30] # Lags for 1-min log return autocorrelation

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions (can be moved to a utils.py later) ---
def calculate_log_returns(series, periods=1):
    s_numeric = pd.to_numeric(series, errors='coerce')
    s_shifted = s_numeric.shift(periods)
    # Ensure positivity for log calculation, replace 0 or negative with NaN
    s_numeric = s_numeric.where(s_numeric > 0, np.nan)
    s_shifted = s_shifted.where(s_shifted > 0, np.nan)
    return np.log(s_numeric / s_shifted)

def calculate_rolling_stat(series, window, stat_func_name, min_p_factor=0.5, **kwargs):
    min_p = max(1, int(window * min_p_factor)) # Ensure min_periods is at least 1
    roll_obj = series.rolling(window=window, min_periods=min_p)
    if hasattr(roll_obj, stat_func_name):
        return getattr(roll_obj, stat_func_name)(**kwargs)
    logger.warning(f"Stat func '{stat_func_name}' not found for rolling object on series '{series.name if series.name else 'Unnamed'}'.")
    return pd.Series(np.nan, index=series.index)

def calculate_rsi(series, window=14, min_p_factor=0.5):
    min_p = max(1, int(window * min_p_factor))
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=min_p).mean()
    loss = delta.clip(upper=0).abs().rolling(window=window, min_periods=min_p).mean()
    rs = gain / (loss + 1e-9) # Add epsilon to prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # Fill initial NaNs with 50 (neutral RSI)

def calculate_ema(series, span, min_p_factor=0.5):
    min_p = max(1, int(span * min_p_factor))
    return series.ewm(span=span, adjust=False, min_periods=min_p).mean()

def calculate_macd(series, fast_p, slow_p, signal_p):
    ema_fast = calculate_ema(series, span=fast_p)
    ema_slow = calculate_ema(series, span=slow_p)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, span=signal_p)
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def calculate_stochastic_oscillator(high_s, low_s, close_s, k_p, k_smooth_p, d_smooth_p):
    min_p_k = max(1, k_p // 2) # Ensure min_periods are reasonable
    min_p_smooth = max(1, k_smooth_p // 2)
    min_p_d = max(1, d_smooth_p // 2)

    lowest_low = low_s.rolling(window=k_p, min_periods=min_p_k).min()
    highest_high = high_s.rolling(window=k_p, min_periods=min_p_k).max()
    pK_raw = 100 * ((close_s - lowest_low) / (highest_high - lowest_low + 1e-9))
    pK_raw.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero if highest_high == lowest_low
    pK = pK_raw.rolling(window=k_smooth_p, min_periods=min_p_smooth).mean()
    pD = pK.rolling(window=d_smooth_p, min_periods=min_p_d).mean()
    return pK.fillna(50), pD.fillna(50) # Fill initial NaNs with 50

def calculate_atr(high_s, low_s, close_s, period):
    pc = close_s.shift(1)
    tr_df = pd.DataFrame({'tr1': high_s - low_s, 'tr2': np.abs(high_s - pc), 'tr3': np.abs(low_s - pc)})
    tr = tr_df.max(axis=1).fillna(0) # fillna(0) for first TR if prev_close is NaN
    min_p_atr = max(1, period // 2)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=min_p_atr).mean()

def calculate_bollinger_bands_width(series, window, num_std=2, min_p_factor=0.5):
    sma = calculate_rolling_stat(series, window, 'mean', min_p_factor)
    roll_std_price = calculate_rolling_stat(series, window, 'std', min_p_factor)
    upper = sma + (roll_std_price * num_std)
    lower = sma - (roll_std_price * num_std)
    return (upper - lower) / (sma + 1e-9)

def calculate_rolling_vwap(price_s, volume_s, window, min_p_factor=0.5):
    pv = price_s * volume_s
    roll_sum_pv = calculate_rolling_stat(pv, window, 'sum', min_p_factor)
    roll_sum_vol = calculate_rolling_stat(volume_s, window, 'sum', min_p_factor)
    return roll_sum_pv / (roll_sum_vol + 1e-9)

def calculate_obv(close_s, volume_s):
    direction = np.sign(close_s.diff()).fillna(0) # Fill first diff NaN with 0
    return (direction * volume_s).cumsum()

def get_utc_timestamp_for_et_time(date_et, time_et, tzinfo_et):
    try:
        dt_naive = datetime.combine(date_et, time_et)
        dt_aware = tzinfo_et.localize(dt_naive, is_dst=None) # is_dst=None for Ambiguous/NonExistent
    except pytz.exceptions.AmbiguousTimeError:
        logger.warning(f"Ambiguous time for {date_et} {time_et}, using is_dst=False.")
        dt_aware = tzinfo_et.localize(dt_naive, is_dst=False)
    except pytz.exceptions.NonExistentTimeError:
        logger.warning(f"Non-existent time for {date_et} {time_et}. This could be due to DST transition. Skipping this instance.")
        return pd.NaT # Return NaT to be handled by caller
    return dt_aware.astimezone(pytz.utc)

# --- Main Logic ---
if __name__ == "__main__":
    if not os.path.exists(FEATURE_OUTPUT_DIR):
        os.makedirs(FEATURE_OUTPUT_DIR)
        logger.info(f"Created output dir: {FEATURE_OUTPUT_DIR}")

    for symbol in SYMBOLS_TO_PROCESS:
        logger.info(f"\n" + "="*50 + f" Processing Symbol: {symbol} " + "="*50)

        kline_consolidated_path = os.path.join(CONSOLIDATED_KLINES_DIR, f"{symbol}_1m_klines_consolidated.parquet")
        aggfeatures_consolidated_path = os.path.join(CONSOLIDATED_AGGFEATURES_DIR, f"{symbol}_1m_aggfeatures_consolidated.parquet")
        feature_label_output_path = os.path.join(FEATURE_OUTPUT_DIR, f"{symbol}_features_labels_v3.parquet")

        # --- Load Kline Data ---
        logger.info(f"--- Loading consolidated KLINE data for {symbol} from {kline_consolidated_path} ---")
        if not os.path.exists(kline_consolidated_path):
            logger.error(f"Kline file not found: {kline_consolidated_path}. Skipping symbol {symbol}.")
            continue
        try:
            df_klines = pd.read_parquet(kline_consolidated_path)
            logger.info(f"Loaded {len(df_klines)} 1-minute klines for {symbol}.")
            df_klines.sort_index(inplace=True) # Ensure sorted
        except Exception as e:
            logger.error(f"Error loading klines for {symbol}: {e}. Skipping symbol.")
            continue
        
        # Basic kline data validation
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in ohlcv_cols if col not in df_klines.columns]
        if missing_cols:
            logger.error(f"Essential OHLCV columns missing in klines: {missing_cols}. Skipping {symbol}.")
            continue
        for col in ohlcv_cols: # Ensure numeric
            df_klines[col] = pd.to_numeric(df_klines[col], errors='coerce')
        df_klines.dropna(subset=ohlcv_cols, inplace=True) # Drop rows if essential data is NaN
        if df_klines.empty:
            logger.error(f"Kline DataFrame for {symbol} empty after OHLCV NaN drop. Skipping.")
            continue

        # --- Load Aggregated AggTrade Features Data ---
        df_aggfeatures = None
        if not os.path.exists(aggfeatures_consolidated_path):
            logger.warning(f"Aggregated features file not found: {aggfeatures_consolidated_path}. Proceeding with klines only for {symbol}.")
        else:
            logger.info(f"--- Loading consolidated AGGREGATED FEATURES data for {symbol} from {aggfeatures_consolidated_path} ---")
            try:
                df_aggfeatures = pd.read_parquet(aggfeatures_consolidated_path)
                logger.info(f"Loaded {len(df_aggfeatures)} 1-minute aggregated features for {symbol}.")
                df_aggfeatures.sort_index(inplace=True) # Ensure sorted
            except Exception as e:
                logger.error(f"Error loading aggfeatures for {symbol}: {e}. Proceeding with klines only.")
                df_aggfeatures = None # Reset to None if loading fails

        # --- Merge Klines and AggFeatures ---
        if df_aggfeatures is not None:
            logger.info(f"Merging klines ({len(df_klines)}) and aggfeatures ({len(df_aggfeatures)}) for {symbol}...")
            # Use inner join to keep only timestamps present in both
            df_merged = pd.merge(df_klines, df_aggfeatures, left_index=True, right_index=True, how='inner')
            logger.info(f"Merged DataFrame for {symbol} has {len(df_merged)} rows.")
            if df_merged.empty:
                logger.error(f"Merged DataFrame is empty for {symbol}. This could be due to misaligned timestamps or no common data. Skipping.")
                continue
        else:
            logger.info(f"Proceeding with kline data only for {symbol} as aggfeatures were not loaded/found.")
            df_merged = df_klines.copy() # Use klines df if no aggfeatures

        # Store initial columns from the merged DataFrame to distinguish later
        initial_df_columns = list(df_merged.columns)
        
        # Dictionary to hold new feature Series, using df_merged's index
        new_features = {}

        logger.info("Starting V3 feature engineering...")
        
        # --- Calculate features based on KLINE data (as before, but on df_merged) ---
        new_features['log_return_1m'] = calculate_log_returns(df_merged['close'])
        new_features['delta_close_1m'] = df_merged['close'] - df_merged['open'] # Renamed for clarity
        new_features['high_low_spread_1m'] = df_merged['high'] - df_merged['low']
        # Calculate wick sizes
        df_merged_max_open_close = pd.concat([df_merged['open'], df_merged['close']], axis=1).max(axis=1)
        df_merged_min_open_close = pd.concat([df_merged['open'], df_merged['close']], axis=1).min(axis=1)
        new_features['wick_top_1m'] = df_merged['high'] - df_merged_max_open_close
        new_features['wick_bottom_1m'] = df_merged_min_open_close - df_merged['low']

        new_features['atr_14m'] = calculate_atr(df_merged['high'], df_merged['low'], df_merged['close'], ATR_PERIOD)
        macd_l, macd_s, macd_h = calculate_macd(df_merged['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        new_features['macd_line_12_26'] = macd_l # More descriptive names
        new_features['macd_signal_12_26_9'] = macd_s
        new_features['macd_hist_12_26_9'] = macd_h
        
        st_k, st_d = calculate_stochastic_oscillator(
            df_merged['high'], df_merged['low'], df_merged['close'],
            STOCH_K_PERIOD, STOCH_K_SMOOTH, STOCH_D_SMOOTH
        )
        new_features[f'stoch_k_{STOCH_K_PERIOD}_{STOCH_K_SMOOTH}'] = st_k
        new_features[f'stoch_d_{STOCH_K_PERIOD}_{STOCH_K_SMOOTH}_{STOCH_D_SMOOTH}'] = st_d
        
        # OBV is cumulative, calculate it on df_merged then use its diff for slope.
        # Ensure 'obv_temp' is not already a column if re-running cells in a notebook
        if 'obv_temp' in df_merged.columns: df_merged.drop(columns=['obv_temp'], inplace=True)
        df_merged['obv_temp'] = calculate_obv(df_merged['close'], df_merged['volume'])
        new_features['obv'] = df_merged['obv_temp'] # Store the cumulative OBV itself

        # --- Calculate Rolling Features (Kline-based and potentially AggTrade-based) ---
        for window in tqdm(ROLLING_WINDOWS_MINUTES, desc=f"  RollFeat {symbol}", unit="win", leave=False):
            min_p_f = 0.5 # min_periods factor
            log_ret_1m = new_features.get('log_return_1m', calculate_log_returns(df_merged['close']))

            # Kline-based rolling features
            new_features[f'log_return_{window}m_sum'] = calculate_rolling_stat(log_ret_1m, window, 'sum', min_p_f)
            new_features[f'volatility_{window}m'] = calculate_rolling_stat(log_ret_1m, window, 'std', min_p_f)
            
            sma_c = calculate_rolling_stat(df_merged['close'], window, 'mean', min_p_f)
            vol_c = calculate_rolling_stat(df_merged['close'], window, 'std', min_p_f)
            new_features[f'price_zscore_{window}m'] = (df_merged['close'] - sma_c) / (vol_c + 1e-9)
            new_features[f'rsi_{window}m'] = calculate_rsi(df_merged['close'], window, min_p_factor=min_p_f)
            new_features[f'price_vs_sma_{window}m'] = (df_merged['close'] - sma_c) / (sma_c + 1e-9)
            ema_c = calculate_ema(df_merged['close'], span=window, min_p_factor=min_p_f)
            new_features[f'price_vs_ema_{window}m'] = (df_merged['close'] - ema_c) / (ema_c + 1e-9)
            new_features[f'bb_width_{window}m'] = calculate_bollinger_bands_width(df_merged['close'], window, num_std=2, min_p_factor=min_p_f)
            
            vol_m = calculate_rolling_stat(df_merged['volume'], window, 'mean', min_p_f)
            vol_s = calculate_rolling_stat(df_merged['volume'], window, 'std', min_p_f)
            new_features[f'volume_{window}m_mean'] = vol_m
            new_features[f'volume_zscore_{window}m'] = (df_merged['volume'] - vol_m) / (vol_s + 1e-9)
            
            if 'taker_buy_base_asset_volume' in df_merged.columns:
                # Note: 'taker_buy_base_asset_volume' is from klines. We also have aggTrade taker volumes.
                new_features[f'kline_taker_buy_vol_sum_{window}m'] = calculate_rolling_stat(df_merged['taker_buy_base_asset_volume'], window, 'sum', min_p_f)
                new_features[f'kline_total_vol_sum_{window}m'] = calculate_rolling_stat(df_merged['volume'], window, 'sum', min_p_f) # Renamed for clarity
                new_features[f'kline_taker_buy_ratio_{window}m'] = new_features[f'kline_taker_buy_vol_sum_{window}m'] / (new_features[f'kline_total_vol_sum_{window}m'] + 1e-9)
            
            if 'number_of_trades' in df_merged.columns: # From klines
                new_features[f'kline_trades_{window}m_mean'] = calculate_rolling_stat(df_merged['number_of_trades'], window, 'mean', min_p_f)
            
            vwap = calculate_rolling_vwap(df_merged['close'], df_merged['volume'], window, min_p_factor=min_p_f)
            new_features[f'vwap_{window}m'] = vwap
            new_features[f'price_vs_vwap_{window}m'] = (df_merged['close'] - vwap) / (vwap + 1e-9)
            new_features[f'obv_slope_{window}m'] = calculate_rolling_stat(df_merged['obv_temp'].diff(), window, 'mean', min_p_f)

            # --- START: New Rolling Features from AggTrade Data ---
            # These features are calculated IF aggTrade data was successfully merged
            aggtrade_cols_to_roll = [
                'agg_net_taker_qty_1m', 'agg_taker_buy_sell_ratio_qty_1m',
                'agg_taker_buy_sell_ratio_count_1m', 'agg_avg_taker_buy_size_1m',
                'agg_avg_taker_sell_size_1m', 'agg_total_trades_1m', 'agg_pct_best_match_1m'
            ]
            for agg_col in aggtrade_cols_to_roll:
                if agg_col in df_merged.columns:
                    # Sum for quantities, mean for ratios/averages/percentages
                    stat_type = 'sum' if 'qty' in agg_col else 'mean'
                    
                    new_features[f'{agg_col}_roll_{window}m_{stat_type}'] = calculate_rolling_stat(
                        df_merged[agg_col], window, stat_type, min_p_f
                    )
                    # Optional: Calculate Z-score for some of these aggTrade rolling features
                    if stat_type == 'mean': # Z-score for means
                         rolling_mean = new_features[f'{agg_col}_roll_{window}m_{stat_type}']
                         rolling_std = calculate_rolling_stat(df_merged[agg_col], window, 'std', min_p_f)
                         new_features[f'{agg_col}_roll_{window}m_zscore'] = (df_merged[agg_col] - rolling_mean) / (rolling_std + 1e-9)
            # --- END: New Rolling Features from AggTrade Data ---

        # Snapshot sum of 1-minute log returns for short windows
        for swindow in SHORT_ROLLING_WINDOWS_MINUTES:
             new_features[f'log_return_{swindow}m_snapshot_sum'] = calculate_rolling_stat(
                 new_features['log_return_1m'], swindow, 'sum', min_p_factor=0.5 # Use higher min_p_factor for snapshots
             )
        
        logger.info(f"Calculating autocorrelations for {symbol}...")
        log_ret_1m_series = new_features.get('log_return_1m', calculate_log_returns(df_merged['close']))
        for lag in tqdm(AUTOCORR_LAGS, desc=f"  Autocorr {symbol}", unit="lag", leave=False):
            # Dynamic rolling window for autocorrelation based on longest kline rolling window
            ac_roll_win = max(ROLLING_WINDOWS_MINUTES)//10 if ROLLING_WINDOWS_MINUTES else 144 # Default if no kline windows
            if ac_roll_win < lag * 5: ac_roll_win = lag * 5 # Ensure window is large enough for the lag
            if ac_roll_win < 30: ac_roll_win = 30 # Minimum practical window
            min_p_ac = max(lag + 5, ac_roll_win // 2)

            if len(log_ret_1m_series.dropna()) >= min_p_ac : # Check on non-NaN length
                shifted_log_ret = log_ret_1m_series.shift(lag)
                # Calculate rolling correlation
                new_features[f'log_return_1m_autocorr_{lag}'] = log_ret_1m_series.rolling(
                    window=ac_roll_win, min_periods=min_p_ac
                ).corr(shifted_log_ret)
            else:
                logger.warning(f"Not enough data points for autocorrelation lag {lag} with window {ac_roll_win} for {symbol}.")
                new_features[f'log_return_1m_autocorr_{lag}'] = pd.Series(np.nan, index=df_merged.index)
        logger.info(f"Finished autocorrelations for {symbol}.")

        # Time-based features (cyclic)
        new_features['hour_sin'] = np.sin(2 * np.pi * df_merged.index.hour / 24.0)
        new_features['hour_cos'] = np.cos(2 * np.pi * df_merged.index.hour / 24.0)
        new_features['dayofweek_sin'] = np.sin(2 * np.pi * df_merged.index.dayofweek / 7.0)
        new_features['dayofweek_cos'] = np.cos(2 * np.pi * df_merged.index.dayofweek / 7.0)
        # Consider adding month_sin/cos as well

        # Consolidate new features into the DataFrame
        logger.info(f"Consolidating {len(new_features)} new feature sets into DataFrame for {symbol}...")
        features_df_to_concat = pd.DataFrame(new_features, index=df_merged.index)
        
        # Concatenate df_merged (which contains klines and raw aggfeatures) with newly engineered features
        df_engineered = pd.concat([df_merged, features_df_to_concat], axis=1)
        
        # Clean up temporary columns
        if 'obv_temp' in df_engineered.columns:
            df_engineered.drop(columns=['obv_temp'], inplace=True)

        logger.info(f"Finished V3 feature engineering for {symbol}. Total columns in engineered df: {len(df_engineered.columns)}")
        
        # --- Label Creation (operates on df_engineered) ---
        logger.info(f"Starting label creation for {symbol}...")
        all_prediction_instances = []
        # Ensure index is timezone-aware for tz_convert
        if df_engineered.index.tz is None:
            logger.warning(f"Index for {symbol} is timezone-naive. Attempting to localize to UTC.")
            try:
                df_engineered.index = df_engineered.index.tz_localize('UTC')
            except Exception as e_tz:
                logger.error(f"Failed to localize index to UTC for {symbol}: {e_tz}. Skipping label creation.")
                continue
        
        unique_et_dates = sorted(list(df_engineered.index.tz_convert(EASTERN_TZ).normalize().unique()))

        for i in tqdm(range(len(unique_et_dates) - 1), desc=f"  Labels {symbol}", unit="day", leave=False):
            day_d_et_normalized = unique_et_dates[i]
            day_d_plus_1_et_normalized = unique_et_dates[i+1]
            
            ts_ref_utc = get_utc_timestamp_for_et_time(day_d_et_normalized.date(), CONTRACT_TIME_ET, EASTERN_TZ)
            ts_res_utc = get_utc_timestamp_for_et_time(day_d_plus_1_et_normalized.date(), CONTRACT_TIME_ET, EASTERN_TZ)

            if pd.isna(ts_ref_utc) or pd.isna(ts_res_utc):
                logger.warning(f"Could not determine reference or resolution time for {day_d_et_normalized.date()}, possibly DST. Skipping.")
                continue
            try:
                # Use asof for robustly finding the nearest available timestamp
                idx_ref = df_engineered.index.asof(ts_ref_utc)
                # Check if the found timestamp is reasonably close (e.g., within 5 minutes)
                if pd.isna(idx_ref) or abs((ts_ref_utc - idx_ref).total_seconds()) > 300:
                    logger.debug(f"Reference kline not found or too far for {ts_ref_utc} (found {idx_ref}). Skipping.")
                    continue
                
                # Snapshot all features available at the reference time
                kline_ref_snapshot_features = df_engineered.loc[idx_ref].copy()
                p_ref = kline_ref_snapshot_features['close'] # Price at reference time

                idx_res = df_engineered.index.asof(ts_res_utc)
                if pd.isna(idx_res) or abs((ts_res_utc - idx_res).total_seconds()) > 300:
                    logger.debug(f"Resolution kline not found or too far for {ts_res_utc} (found {idx_res}). Skipping.")
                    continue
                p_res = df_engineered.loc[idx_res, 'close'] # Price at resolution time

            except KeyError as e_key:
                logger.warning(f"KeyError during feature/label lookup for {ts_ref_utc} or {ts_res_utc}: {e_key}. Skipping.")
                continue
            except Exception as e_lookup:
                logger.error(f"Unexpected error during feature/label lookup: {e_lookup}", exc_info=True)
                continue

            if pd.isna(p_ref) or pd.isna(p_res) or p_ref == 0:
                logger.debug(f"NaN reference or resolution price, or p_ref is zero for {idx_ref} / {idx_res}. Skipping.")
                continue

            p_change_pct = (p_res - p_ref) / p_ref
            target_label = 1 if p_change_pct > TARGET_RETURN_THRESHOLD else 0 # As per Polymarket rules
            
            instance_dict = kline_ref_snapshot_features.to_dict()
            instance_dict['target_up_down'] = target_label
            instance_dict['event_ref_time_utc_actual'] = idx_ref # Actual timestamp of features
            instance_dict['event_res_time_utc_actual'] = idx_res # Actual timestamp of resolution price
            instance_dict['contract_day_d_ET'] = day_d_et_normalized.strftime('%Y-%m-%d')
            instance_dict['contract_day_d_plus_1_ET'] = day_d_plus_1_et_normalized.strftime('%Y-%m-%d')
            all_prediction_instances.append(instance_dict)

        if not all_prediction_instances:
            logger.warning(f"No prediction instances generated for {symbol}. Check data alignment and time ranges.")
            continue
            
        feature_label_df = pd.DataFrame(all_prediction_instances)
        if feature_label_df.empty:
            logger.error(f"Final feature_label DataFrame is empty for {symbol}. Skipping save.")
            continue
            
        # Set index for the final DataFrame
        if 'event_ref_time_utc_actual' in feature_label_df.columns:
            feature_label_df = feature_label_df.set_index('event_ref_time_utc_actual', drop=False)
            feature_label_df.index.name = 'timestamp_utc' # Standardize index name
        else:
            logger.error("Critical column 'event_ref_time_utc_actual' missing. Cannot set index. Skipping save.")
            continue
        
        logger.info(f"Generated {len(feature_label_df)} feature/label instances for {symbol}.")
        
        # Identify newly engineered features (excluding original kline/aggfeature columns and metadata/target)
        metadata_target_cols = [
            'target_up_down', 'event_ref_time_utc_actual', 'event_res_time_utc_actual',
            'contract_day_d_ET', 'contract_day_d_plus_1_ET'
        ]
        # initial_df_columns now contains columns from both df_klines and df_aggfeatures (if merged)
        truly_engineered_feature_names = [
            col for col in feature_label_df.columns 
            if col not in initial_df_columns and col not in metadata_target_cols
        ]
        
        if truly_engineered_feature_names:
            nan_in_eng_features = feature_label_df[truly_engineered_feature_names].isnull().sum()
            nan_summary = nan_in_eng_features[nan_in_eng_features > 0].sort_values(ascending=False)
            if not nan_summary.empty:
                logger.info(f"NaNs in V3 engineered features for {symbol} (Top 20 or all if fewer):\n{nan_summary.head(20).to_string()}")
            else:
                logger.info(f"No NaNs found in newly engineered V3 features for {symbol}.")
        else:
            logger.info(f"No distinct new V3 engineered features identified for NaN check in {symbol}.")

        feature_label_df.to_parquet(feature_label_output_path)
        logger.info(f"V3 Features/labels for {symbol} saved to {feature_label_output_path} ({len(feature_label_df)} rows, {len(feature_label_df.columns)} columns).")

    logger.info("--- V3 Feature engineering and label creation script finished. ---")