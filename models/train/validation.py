# File: /models/train/validation.py
import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler # Needed to load the scaler

# --- Configuration ---
SYMBOL_TO_MODEL = "BTCUSDT"
FEATURE_DATA_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/binance_market_data/features_labels_v3"
MODEL_OUTPUT_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/models/trained_models"
FEATURE_FILE_PATH = os.path.join(FEATURE_DATA_DIR, f"{SYMBOL_TO_MODEL}_features_labels_v3.parquet")
TARGET_COLUMN = 'target_up_down'

# --- Columns to drop (MUST MATCH TRAINING SCRIPTS) ---
# Combine all drop lists from training scripts
ORIGINAL_KLINE_COLS_TO_DROP = [
    'kline_open_time', 'open', 'high', 'low', 'close', 'volume',
    'kline_close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]
ORIGINAL_AGGTRADE_FEATURES_TO_DROP = [
    'agg_taker_buy_qty_1m', 'agg_taker_sell_qty_1m', 'agg_taker_buy_count_1m',
    'agg_taker_sell_count_1m', 'agg_total_qty_1m', 'agg_total_trades_1m',
    'agg_price_mean_1m', 'agg_price_std_1m', 'agg_best_match_true_count_1m',
    'agg_net_taker_qty_1m', 'agg_taker_buy_sell_ratio_qty_1m',
    'agg_taker_buy_sell_ratio_count_1m', 'agg_avg_taker_buy_size_1m',
    'agg_avg_taker_sell_size_1m', 'agg_pct_best_match_1m'
]
METADATA_COLS_TO_DROP = [
    'event_ref_time_utc_actual', 'event_res_time_utc_actual',
    'contract_day_d_ET', 'contract_day_d_plus_1_ET'
]
COLUMNS_TO_DROP_PRE_PROCESSING = (
    ORIGINAL_KLINE_COLS_TO_DROP +
    ORIGINAL_AGGTRADE_FEATURES_TO_DROP +
    METADATA_COLS_TO_DROP
)
N_SPLITS_CV = 5 # Must match the N_SPLITS_CV used during training

# --- Models to Validate ---
# Define the models and their corresponding saved files and scaler files
MODELS_TO_VALIDATE = {
    # From linreg.py
    "LinReg_Default_L2": {
        "model_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_Default_L2.joblib"),
        "scaler_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_scaler_final.joblib"),
        "suffix": "_v3_LinReg_NoOptuna" # Used for consistency if needed later
    },
    "LinReg_L1_C1.0": {
        "model_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_L1_C1.0.joblib"),
        "scaler_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_scaler_final.joblib"),
        "suffix": "_v3_LinReg_NoOptuna"
    },
     "LinReg_L2_C100": {
        "model_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_L2_C100.joblib"),
        "scaler_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_scaler_final.joblib"),
        "suffix": "_v3_LinReg_NoOptuna"
    },
    "LinReg_Saga_L1_C0.1": {
        "model_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_Saga_L1_C0.1.joblib"),
        "scaler_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_scaler_final.joblib"),
        "suffix": "_v3_LinReg_NoOptuna"
    },
     "LinReg_Saga_ElasticNet_C0.1_L1_0.5": {
        "model_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_Saga_ElasticNet_C0.1_L1_0.5.joblib"),
        "scaler_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_LinReg_NoOptuna_scaler_final.joblib"),
        "suffix": "_v3_LinReg_NoOptuna"
    },
    # From catboosttrain.py (using its specific suffix and setup name)
    "CatBoost_Defaults": {
        "model_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_CatBoost_Defaults_Default_CatBoost.joblib"),
        "scaler_path": os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_v3_CatBoost_Defaults_scaler_final.joblib"),
        "suffix": "_v3_CatBoost_Defaults"
    }
}


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function ---
def load_and_preprocess_data(file_path, target_col, drop_cols):
    """Loads data, checks index, drops columns, handles NaNs."""
    logger.info(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"FATAL: File not found: {file_path}. Exiting.")
        return None, None
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} instances.")
    except Exception as e:
        logger.error(f"FATAL: Error loading Parquet file {file_path}: {e}. Exiting.")
        return None, None

    # Basic Index Check (optional refinement: ensure datetime)
    if 'timestamp_utc' not in df.index.name and 'timestamp_utc' in df.columns:
         logger.warning("Setting index to 'timestamp_utc'. Ensure it's datetime.")
         df = df.set_index('timestamp_utc')

    if target_col not in df.columns:
        logger.error(f"FATAL: Target column '{target_col}' not found. Exiting.")
        return None, None

    y = df[target_col]
    feature_candidates = [col for col in df.columns if col not in [target_col] + drop_cols]
    X = df[feature_candidates].copy()
    logger.info(f"Initial features: {len(X.columns)}")

    # Drop all-NaN columns
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        logger.warning(f"Dropping all-NaN columns: {all_nan_cols}")
        X.drop(columns=all_nan_cols, inplace=True)

    # Fill remaining NaNs with median (consistent with training)
    logger.info("Filling remaining NaNs with column medians...")
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            fill_value = median_val if not pd.isna(median_val) else 0
            X[col] = X[col].fillna(fill_value)
            if pd.isna(median_val):
                 logger.warning(f"Median for '{col}' was NaN, filled with 0.")

    if X.isnull().sum().sum() > 0:
        logger.error("NaNs still present after fill. Exiting.")
        return None, None

    # Drop object columns
    obj_cols = X.select_dtypes(include='object').columns
    if not obj_cols.empty:
        logger.warning(f"Dropping object columns: {list(obj_cols)}.")
        X.drop(columns=obj_cols, inplace=True)

    logger.info(f"Final features after preproc: {len(X.columns)}")
    if len(X) != len(y):
         logger.error("Length mismatch X vs y after preproc. Exiting.")
         return None, None
    if X.empty or X.shape[1] == 0:
         logger.error("Feature set X is empty after preproc. Exiting.")
         return None, None

    return X, y

def calculate_metrics(y_true, y_pred_proba, model_name):
    """Calculates and returns a dictionary of metrics."""
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred_class)
    metrics['Precision'] = precision_score(y_true, y_pred_class, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred_class, zero_division=0)
    metrics['F1 Score'] = f1_score(y_true, y_pred_class, zero_division=0)
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics['AUC'] = np.nan # Handle cases with only one class in y_true
        logger.warning(f"Only one class present in y_true for {model_name}. AUC is NaN.")
    metrics['LogLoss'] = log_loss(y_true, y_pred_proba)
    # Optional: add confusion matrix if needed later
    # metrics['ConfusionMatrix'] = confusion_matrix(y_true, y_pred_class).tolist()
    return metrics


# --- Main Validation Logic ---
if __name__ == "__main__":
    logger.info(f"--- Starting Model Validation for Symbol: {SYMBOL_TO_MODEL} ---")

    X_full, y_full = load_and_preprocess_data(FEATURE_FILE_PATH, TARGET_COLUMN, COLUMNS_TO_DROP_PRE_PROCESSING)

    if X_full is None:
        exit()

    # --- Get the FINAL validation set indices (same as in training) ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    try:
        splits = list(tscv.split(X_full, y_full))
        if len(splits) < N_SPLITS_CV:
             raise ValueError(f"TimeSeriesSplit generated {len(splits)} splits, expected {N_SPLITS_CV}.")
        train_idx, val_idx = splits[-1] # Get indices for the last split
    except Exception as e:
         logger.error(f"FATAL: Error getting TimeSeriesSplit indices: {e}. Exiting.")
         exit()

    if len(val_idx) == 0:
         logger.error(f"FATAL: The final validation split is empty. Exiting.")
         exit()

    # We only need the validation data for this script
    X_val = X_full.iloc[val_idx]
    y_val = y_full.iloc[val_idx]
    logger.info(f"Using final validation set of size: {len(X_val)}")

    results = []

    # --- Iterate through models, load, predict, evaluate ---
    for model_name, paths in MODELS_TO_VALIDATE.items():
        logger.info(f"\n--- Validating: {model_name} ---")
        model_path = paths["model_path"]
        scaler_path = paths["scaler_path"]

        # Load Scaler
        if not os.path.exists(scaler_path):
            logger.error(f"Scaler file not found: {scaler_path}. Skipping {model_name}.")
            continue
        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler: {scaler_path}")
        except Exception as e:
            logger.error(f"Error loading scaler {scaler_path}: {e}. Skipping {model_name}.")
            continue

        # Load Model
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}. Skipping {model_name}.")
            continue
        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}. Skipping {model_name}.")
            continue

        # --- Prepare Validation Data ---
        # Ensure validation features match features model was trained on
        # This requires aligning columns before scaling if features changed between runs
        # For simplicity, assuming feature set is consistent as defined by drop_cols
        if set(X_val.columns) != set(scaler.feature_names_in_):
             logger.warning(f"Feature mismatch between validation data ({len(X_val.columns)}) and scaler ({len(scaler.feature_names_in_)}). Attempting to align.")
             common_features = list(set(X_val.columns) & set(scaler.feature_names_in_))
             missing_in_val = list(set(scaler.feature_names_in_) - set(X_val.columns))
             extra_in_val = list(set(X_val.columns) - set(scaler.feature_names_in_))

             if not common_features:
                 logger.error(f"No common features between validation data and scaler for {model_name}. Skipping.")
                 continue

             logger.info(f"Keeping {len(common_features)} common features.")
             if extra_in_val: logger.warning(f"Dropping extra features in validation data: {extra_in_val}")
             X_val_aligned = X_val[common_features].copy()

             if missing_in_val:
                 logger.warning(f"Adding missing features to validation data (filled with 0): {missing_in_val}")
                 for col in missing_in_val:
                     X_val_aligned[col] = 0 # Add missing columns filled with 0
                 X_val_aligned = X_val_aligned[list(scaler.feature_names_in_)] # Reorder
        else:
            X_val_aligned = X_val # Features already match


        # Scale validation data
        try:
            X_val_scaled = scaler.transform(X_val_aligned)
             # For CatBoost, convert back to DataFrame if needed (though predict_proba often works with numpy)
            if "CatBoost" in model_name:
                 X_val_scaled_df = pd.DataFrame(X_val_scaled, index=X_val_aligned.index, columns=X_val_aligned.columns)
                 predict_data = X_val_scaled_df
            else:
                 predict_data = X_val_scaled
        except Exception as e:
             logger.error(f"Error scaling validation data for {model_name}: {e}. Skipping.")
             continue

        # Predict probabilities
        try:
            y_pred_proba = model.predict_proba(predict_data)[:, 1]
        except Exception as e:
            logger.error(f"Error predicting probabilities for {model_name}: {e}. Skipping.")
            continue

        # Calculate metrics
        model_metrics = calculate_metrics(y_val, y_pred_proba, model_name)
        model_metrics['Model'] = model_name # Add model name for reporting
        results.append(model_metrics)

        # Log metrics for this model
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in model_metrics.items() if k != 'Model'])
        logger.info(f"Metrics for {model_name}: {metrics_str}")


    # --- Display Comparison ---
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index('Model') # Set Model name as index
        # Define column order for better readability
        col_order = ['AUC', 'LogLoss', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        # Ensure all expected columns exist, add NaN if not
        for col in col_order:
             if col not in results_df.columns:
                 results_df[col] = np.nan
        results_df = results_df[col_order] # Reorder columns

        logger.info("\n" + "="*80)
        logger.info("          Model Validation Comparison (on Final Split)")
        logger.info("="*80)
        logger.info("\n" + results_df.to_string(float_format="%.4f"))
        logger.info("="*80)
    else:
        logger.warning("No models were successfully validated.")

    logger.info(f"--- Model Validation Script Finished ---")