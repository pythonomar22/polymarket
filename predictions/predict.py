# predict.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime, time
import sys

# Add the parent directory to sys.path to be able to import from dataprocessing
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Import configuration and feature building functions
from dataprocessing import config
# We need the feature calculation logic, but applied only to the latest data
from dataprocessing.build_features import load_processed_data, calculate_features # Reuse feature calculation

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
MODEL_DIR = config.BASE_DIR / "models"
PREDICTION_OUTPUT_DIR = config.BASE_DIR / "predictions"
PREDICTION_OUTPUT_DIR.mkdir(exist_ok=True)

# --- Helper Functions ---
def get_latest_noon_timestamp(df: pd.DataFrame) -> pd.Timestamp | None:
    """Find the timestamp of the most recent 12:00 PM ET in the data."""
    noon_times = df[df.index.time == config.RESOLUTION_TIME]
    if noon_times.empty:
        logger.error("No noon timestamps found in the processed data.")
        return None
    latest_noon = noon_times.index.max()
    logger.info(f"Latest noon timestamp found in data: {latest_noon}")
    return latest_noon

def load_model(symbol: str) -> object | None:
    """Loads a saved model."""
    model_path = MODEL_DIR / f"{symbol}_lgbm_model.joblib"
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None

# --- Main Prediction Logic ---
if __name__ == "__main__":
    logger.info("Starting Prediction script...")

    # Determine the target prediction date (the day *after* the latest data)
    # This assumes the script is run shortly after noon ET on the last day of data.
    # Example: If latest data noon is May 12th 12:00 ET, we predict for May 13th.
    prediction_target_date_str = "2025-05-13" # Manually set for this example
    prediction_target_dt = pd.Timestamp(prediction_target_date_str, tz=config.TARGET_TZ).replace(hour=config.RESOLUTION_TIME.hour, minute=config.RESOLUTION_TIME.minute)
    logger.info(f"Generating predictions for target resolution date: {prediction_target_dt}")


    predictions = {}
    symbols_to_predict = config.SYMBOLS + ["ETHBTC"]

    # --- Load necessary data for feature calculation ---
    all_processed_data = {}
    latest_noon_overall = None
    required_symbols = list(set(config.SYMBOLS + ["BTCUSDT", "ETHUSDT"])) # Ensure BTC/ETH loaded for ratio

    for symbol in required_symbols:
        df = load_processed_data(symbol) # From build_features
        if df is None:
            logger.critical(f"Failed to load processed data for essential symbol {symbol}. Cannot proceed.")
            exit()
        all_processed_data[symbol] = df
        # Find the latest noon across all loaded data
        current_latest_noon = get_latest_noon_timestamp(df)
        if current_latest_noon:
             if latest_noon_overall is None or current_latest_noon > latest_noon_overall:
                 latest_noon_overall = current_latest_noon

    if latest_noon_overall is None:
         logger.critical("Could not determine the latest noon timestamp from data. Exiting.")
         exit()

    logger.info(f"Latest noon timestamp across all relevant data: {latest_noon_overall}")
    # We need features calculated *at* this latest noon timestamp to predict the *next* day.

    # --- Generate predictions for each symbol ---
    for symbol in symbols_to_predict:
        logger.info(f"===== Predicting for Symbol: {symbol} =====")

        # 1. Load Model
        model = load_model(symbol)
        if model is None:
            logger.warning(f"Skipping prediction for {symbol} due to model loading error.")
            continue

        # 2. Prepare Features for the Latest Timestamp
        features_for_prediction = None
        if symbol == "ETHBTC":
            # Calculate ratio features for the latest timestamp
            if "ETHUSDT" in all_processed_data and "BTCUSDT" in all_processed_data:
                eth_df = all_processed_data["ETHUSDT"]
                btc_df = all_processed_data["BTCUSDT"]

                # Align ETH and BTC data
                merged_df = pd.merge(eth_df[['close', 'volume', 'taker_buy_vol']].rename(columns={'close': 'close_eth', 'volume': 'volume_eth', 'taker_buy_vol': 'taker_buy_vol_eth'}),
                                     btc_df[['close', 'volume', 'taker_buy_vol']].rename(columns={'close': 'close_btc', 'volume': 'volume_btc', 'taker_buy_vol': 'taker_buy_vol_btc'}),
                                     left_index=True, right_index=True, how='inner')

                # Calculate ETH/BTC ratio
                merged_df['eth_btc_ratio'] = merged_df['close_eth'] / merged_df['close_btc']

                # Create pseudo-DataFrame for feature calculation
                ratio_df_for_features = pd.DataFrame({
                    'close': merged_df['eth_btc_ratio'],
                    'volume': merged_df['volume_eth'] + merged_df['volume_btc'],
                    'taker_buy_vol': merged_df['taker_buy_vol_eth'] + merged_df['taker_buy_vol_btc']
                 }).dropna(subset=['close'])

                # Calculate features using the full history of the ratio
                all_ratio_features = calculate_features(ratio_df_for_features, symbol)
                if all_ratio_features is not None:
                    # Select features exactly at the latest noon timestamp
                    if latest_noon_overall in all_ratio_features.index:
                        features_for_prediction = all_ratio_features.loc[[latest_noon_overall]]
                    else:
                        logger.warning(f"Latest noon timestamp {latest_noon_overall} not found in {symbol} features index.")

            else:
                logger.warning(f"Missing ETH or BTC data for ETHBTC feature calculation.")
        else:
            # Calculate features for the base symbol
            if symbol in all_processed_data:
                df = all_processed_data[symbol]
                all_symbol_features = calculate_features(df, symbol) # Calculate features on the whole history
                if all_symbol_features is not None:
                     # Select features exactly at the latest noon timestamp
                    if latest_noon_overall in all_symbol_features.index:
                         features_for_prediction = all_symbol_features.loc[[latest_noon_overall]]
                    else:
                        logger.warning(f"Latest noon timestamp {latest_noon_overall} not found in {symbol} features index.")
            else:
                 logger.warning(f"Processed data for {symbol} not loaded.")

        # Check if we got the features
        if features_for_prediction is None or features_for_prediction.empty:
            logger.error(f"Could not extract features for {symbol} at {latest_noon_overall}. Skipping prediction.")
            continue

        # Ensure feature columns match model's expected features
        # Note: Model object often stores feature names it was trained on.
        try:
            # LightGBM stores feature names if trained on pandas DataFrame
            model_features = model.feature_name_
            features_for_prediction = features_for_prediction[model_features] # Reorder/select columns
        except AttributeError:
            logger.warning(f"Could not get feature names from model for {symbol}. Assuming column order matches training.")
            # Check column count as a basic sanity check
            if model.n_features_ != len(features_for_prediction.columns):
                 logger.error(f"Feature count mismatch for {symbol}. Model expects {model.n_features_}, got {len(features_for_prediction.columns)}. Skipping.")
                 continue
        except Exception as e:
             logger.error(f"Error aligning features for prediction for {symbol}: {e}. Skipping.")
             continue

        # 3. Make Prediction
        try:
            pred_proba = model.predict_proba(features_for_prediction)[:, 1] # Probability of Up (class 1)
            prediction_value = pred_proba[0] # Get the single value
            predictions[symbol] = prediction_value
            logger.info(f"Prediction for {symbol} (Prob Up): {prediction_value:.4f}")
        except Exception as e:
            logger.error(f"Error during prediction for {symbol}: {e}")


    # 4. Save Predictions
    if predictions:
        pred_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['predicted_prob_up'])
        pred_df.index.name = 'symbol'
        output_filename = PREDICTION_OUTPUT_DIR / f"predictions_{prediction_target_date_str}.csv"
        try:
            pred_df.to_csv(output_filename)
            logger.info(f"Predictions saved to {output_filename}")
            print("\n--- Predictions for Market Resolution on {} ---".format(prediction_target_date_str))
            print(pred_df)
            print("-------------------------------------------------")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
    else:
        logger.warning("No predictions were generated.")


    logger.info("--- Prediction script finished. ---")