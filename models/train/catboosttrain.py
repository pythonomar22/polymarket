# File: /models/train/catboost.py
# train_catboost_v3_defaults.py

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.model_selection import TimeSeriesSplit # Keep using TimeSeriesSplit for validation strategy
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
SYMBOL_TO_MODEL = "BTCUSDT"
FEATURE_DATA_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/binance_market_data/features_labels_v3"
MODEL_OUTPUT_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/models/trained_models"
REPORTS_OUTPUT_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/models/reports"
FEATURE_FILE_PATH = os.path.join(FEATURE_DATA_DIR, f"{SYMBOL_TO_MODEL}_features_labels_v3.parquet")
TARGET_COLUMN = 'target_up_down'

# Columns to drop (same as before)
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
    'event_ref_time_utc_actual',
    'event_res_time_utc_actual',
    'contract_day_d_ET',
    'contract_day_d_plus_1_ET'
]
COLUMNS_TO_DROP_PRE_PROCESSING = (
    ORIGINAL_KLINE_COLS_TO_DROP +
    ORIGINAL_AGGTRADE_FEATURES_TO_DROP +
    METADATA_COLS_TO_DROP
)

N_SPLITS_CV = 5 # For TimeSeriesSplit

# --- Logging Setup ---
# Suppress verbose logging from matplotlib and PIL
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for logger

# --- Main Script ---
if __name__ == "__main__":
    script_version_suffix = "_v3_CatBoost_Defaults" # Updated suffix
    logger.info(f"--- Starting CatBoost (Default Params) for Symbol: {SYMBOL_TO_MODEL} (Features V3) ---")

    # Create output directories if they don't exist
    for dir_path in [MODEL_OUTPUT_DIR, REPORTS_OUTPUT_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created: {dir_path}")

    # --- Data Loading ---
    logger.info(f"Loading V3 data: {FEATURE_FILE_PATH}")
    if not os.path.exists(FEATURE_FILE_PATH):
        logger.error(f"FATAL: Feature file not found: {FEATURE_FILE_PATH}. Exiting.")
        exit()
    try:
        df = pd.read_parquet(FEATURE_FILE_PATH)
        logger.info(f"Loaded {len(df)} V3 instances for {SYMBOL_TO_MODEL}.")
    except Exception as e:
        logger.error(f"FATAL: Error loading Parquet file {FEATURE_FILE_PATH}: {e}. Exiting.")
        exit()

    # --- Index Validation ---
    if df.index.name != 'timestamp_utc' or not isinstance(df.index, pd.DatetimeIndex):
        # Attempt to set index if it's a column, otherwise fail
        if 'timestamp_utc' in df.columns:
             logger.warning("Index not set to 'timestamp_utc'. Setting it now.")
             df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce', utc=True)
             df = df.set_index('timestamp_utc')
             if df.index.isnull().any():
                 logger.error("FATAL: Null values found in 'timestamp_utc' after conversion. Exiting.")
                 exit()
             logger.info("Index successfully set to 'timestamp_utc'.")
        else:
            logger.error(f"Index is not 'timestamp_utc', not DatetimeIndex, and column not found. Current index: {df.index.name}. Exiting.")
            exit()
    df.sort_index(inplace=True) # Ensure data is sorted by time

    # --- Feature and Target Preparation ---
    if TARGET_COLUMN not in df.columns:
        logger.error(f"FATAL: Target column '{TARGET_COLUMN}' not found in DataFrame. Exiting.")
        exit()
    y_full = df[TARGET_COLUMN]

    # Identify feature candidates (all columns except target and those explicitly dropped)
    feature_candidates = [col for col in df.columns if col not in [TARGET_COLUMN] + COLUMNS_TO_DROP_PRE_PROCESSING]
    X_all_features_orig = df[feature_candidates].copy()
    logger.info(f"Initial number of candidate features: {len(X_all_features_orig.columns)}")

    # --- Preprocessing ---
    # Drop columns that are entirely NaN
    all_nan_cols = X_all_features_orig.columns[X_all_features_orig.isnull().all()].tolist()
    if all_nan_cols:
        logger.warning(f"Dropping columns from X that are all NaN: {all_nan_cols}")
        X_all_features_orig.drop(columns=all_nan_cols, inplace=True)

    # Fill remaining NaNs with column medians (keeping this strategy consistent for now)
    logger.info("Filling remaining NaNs in feature set with column medians...")
    cols_filled = 0
    for col in X_all_features_orig.columns:
        if X_all_features_orig[col].isnull().any():
            median_val = X_all_features_orig[col].median()
            if pd.isna(median_val):
                # If median is NaN (e.g., column has many NaNs or weird data), fill with 0
                logger.warning(f"Median for column '{col}' is NaN. Filling NaNs with 0.")
                X_all_features_orig[col] = X_all_features_orig[col].fillna(0)
            else:
                X_all_features_orig[col] = X_all_features_orig[col].fillna(median_val)
            cols_filled += 1
    logger.info(f"Filled NaNs in {cols_filled} columns using median (or 0 if median was NaN).")


    # Check for any remaining NaNs after filling
    if X_all_features_orig.isnull().sum().sum() > 0:
        remaining_nan_cols = X_all_features_orig.columns[X_all_features_orig.isnull().any()].tolist()
        logger.error(f"NaNs still present after attempting to fill in columns: {remaining_nan_cols}. Exiting.")
        exit()

    # Drop any object columns that might have slipped through (CatBoost can handle them, but we expect numerics here)
    obj_cols = X_all_features_orig.select_dtypes(include='object').columns
    if not obj_cols.empty:
        logger.warning(f"Object columns found in final features: {list(obj_cols)}. Dropping them.")
        X_all_features_orig.drop(columns=obj_cols, inplace=True)

    logger.info(f"X_all_features shape after pre-processing: {X_all_features_orig.shape}, y_full shape: {y_full.shape}")
    if X_all_features_orig.empty or X_all_features_orig.shape[1] == 0:
        logger.error("Feature set X is empty after preprocessing. Exiting.")
        exit()
    if len(X_all_features_orig) != len(y_full):
         logger.error(f"FATAL: Length mismatch between X ({len(X_all_features_orig)}) and y ({len(y_full)}). Exiting.")
         exit()


    X_to_use = X_all_features_orig

    # --- TimeSeriesSplit and Evaluation ---
    logger.info(f"Evaluating CatBoost on the last CV split using TimeSeriesSplit (n_splits={N_SPLITS_CV})...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

    # Get all splits and use the last one for final evaluation (same strategy as before)
    try:
        splits = list(tscv.split(X_to_use, y_full)) # Get all (train_indices, val_indices) pairs
        if not splits:
             raise ValueError("TimeSeriesSplit generated no splits.")
        final_train_idx, final_val_idx = splits[-1] # Unpack the last tuple
    except Exception as e:
        logger.error(f"FATAL: Error during TimeSeriesSplit: {e}. Check data length and n_splits. Exiting.")
        exit()


    # Check if splits are valid
    if len(final_train_idx) == 0 or len(final_val_idx) == 0:
        logger.error(f"FATAL: TimeSeriesSplit resulted in empty train ({len(final_train_idx)}) or validation ({len(final_val_idx)}) set. Exiting.")
        exit()

    X_train_final = X_to_use.iloc[final_train_idx]
    y_train_final = y_full.iloc[final_train_idx]
    X_val_final = X_to_use.iloc[final_val_idx]
    y_val_final = y_full.iloc[final_val_idx]

    logger.info(f"Final train set size: {len(X_train_final)}, Final validation set size: {len(X_val_final)}")

    # --- Scaling (kept for consistency, though less critical for CatBoost) ---
    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_final)
    X_val_final_scaled = scaler_final.transform(X_val_final)

    # Convert scaled arrays back to DataFrames with original columns for CatBoost (helps with potential categorical features later)
    X_train_final_scaled_df = pd.DataFrame(X_train_final_scaled, index=X_train_final.index, columns=X_train_final.columns)
    X_val_final_scaled_df = pd.DataFrame(X_val_final_scaled, index=X_val_final.index, columns=X_val_final.columns)


    # --- Model Setup (Simplified for CatBoost defaults) ---
    # Define a single default CatBoost setup
    # Common parameters: iterations, learning_rate, depth, l2_leaf_reg, loss_function, eval_metric
    model_setups_to_try = {
        "Default_CatBoost": CatBoostClassifier(
            iterations=1000, # Number of trees
            learning_rate=0.05, # Typical learning rate
            depth=6, # Tree depth
            l2_leaf_reg=3, # L2 regularization
            loss_function='Logloss', # For binary classification probability
            eval_metric='AUC', # Metric to potentially watch during training (though not used for early stopping here)
            auto_class_weights='Balanced', # Handles class imbalance like class_weight='balanced'
            random_state=42,
            verbose=False, # Suppress verbose output during fit/predict
            # Optional: consider task_type='GPU' if you have a compatible GPU and large dataset
        )
    }

    # --- Reporting Setup ---
    full_report_content = f"CatBoost Report (Defaults) for V3: {SYMBOL_TO_MODEL}\n"
    full_report_content += f"Data: {FEATURE_FILE_PATH}\nTotal instances in loaded df: {len(df)}\n"
    full_report_content += f"Features used for Model: {X_to_use.shape[1]} features.\n"
    full_report_content += f"Evaluation on last of {N_SPLITS_CV} TimeSeriesSplits.\n"
    full_report_content += f"Train size: {len(X_train_final)}, Validation size: {len(X_val_final)}\n"
    full_report_content += "="*70 + "\n"

    # --- Train and Evaluate ---
    # Loop through the defined setups (only one here)
    for setup_name, model_instance in model_setups_to_try.items():
        logger.info(f"\n--- Evaluating Setup: {setup_name} ---")
        full_report_content += f"\n--- Setup: {setup_name} ---\n"

        final_model = model_instance # Assign the pre-configured model instance
        full_report_content += f"Params: {final_model.get_params()}\n"

        try:
            # Train the model
            # Note: CatBoost can use eval_set for early stopping, but we're keeping it simple like linreg.py
            final_model.fit(X_train_final_scaled_df, y_train_final) # Using scaled data
        except Exception as e:
            logger.error(f"Error fitting model for setup {setup_name}: {e}", exc_info=True)
            full_report_content += f"Error fitting model: {e}\n"
            continue # Skip evaluation if fitting fails

        # --- Predict and Evaluate ---
        try:
            y_pred_proba_final = final_model.predict_proba(X_val_final_scaled_df)[:, 1]
            y_pred_class_final = (y_pred_proba_final >= 0.5).astype(int) # Standard 0.5 threshold
        except Exception as e:
            logger.error(f"Error during prediction for setup {setup_name}: {e}", exc_info=True)
            full_report_content += f"Error during prediction: {e}\n"
            continue # Skip metrics if prediction fails

        # Calculate metrics
        final_acc = accuracy_score(y_val_final, y_pred_class_final)
        final_prec = precision_score(y_val_final, y_pred_class_final, zero_division=0)
        final_rec = recall_score(y_val_final, y_pred_class_final, zero_division=0)
        final_f1 = f1_score(y_val_final, y_pred_class_final, zero_division=0)
        try:
            # Check if validation target has both classes before calculating AUC
            if len(np.unique(y_val_final)) > 1:
                 final_auc = roc_auc_score(y_val_final, y_pred_proba_final)
            else:
                 logger.warning(f"Only one class present in y_val_final for setup {setup_name}. AUC set to 0.5.")
                 final_auc = 0.5 # Assign neutral AUC if only one class exists
        except ValueError as e_auc:
             logger.error(f"ValueError calculating AUC for {setup_name}: {e_auc}. Setting AUC to NaN.")
             final_auc = np.nan # Error during calculation
        final_lloss = log_loss(y_val_final, y_pred_proba_final)
        final_cm = confusion_matrix(y_val_final, y_pred_class_final)

        log_msg = f"Metrics for {setup_name}: Acc={final_acc:.4f}, P={final_prec:.4f}, R={final_rec:.4f}, F1={final_f1:.4f}, AUC={final_auc:.4f}, LogLoss={final_lloss:.4f}"
        logger.info(log_msg)
        logger.info(f"CM for {setup_name}:\n{final_cm}")

        full_report_content += log_msg + "\n"
        full_report_content += f"CM:\n{final_cm}\n"

        # --- Save Model ---
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}{script_version_suffix}_{setup_name}.joblib")
        try:
            joblib.dump(final_model, model_path)
            logger.info(f"Model for {setup_name} saved: {model_path}")
        except Exception as e:
             logger.error(f"Error saving model {model_path}: {e}", exc_info=True)
             full_report_content += f"Error saving model: {e}\n"


        # --- Feature Importance (CatBoost specific) ---
        try:
            if hasattr(final_model, 'get_feature_importance') and X_train_final_scaled_df.shape[1] > 0:
                fi_values = final_model.get_feature_importance()
                if len(fi_values) == X_train_final_scaled_df.shape[1]:
                    fi_df = pd.DataFrame({'feature': X_train_final_scaled_df.columns,
                                            'importance': fi_values})
                    fi_df = fi_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
                    top_features_str = fi_df.head(20).to_string() # Show top 20
                    logger.info(f"Top Features for {setup_name} (by importance):\n{top_features_str}")
                    full_report_content += f"Top 20 Features (Importance):\n{top_features_str}\n"

                    # Plot feature importances
                    plt.figure(figsize=(12, max(8, min(20, len(fi_df.head(40))) // 1.5 ))) # Dynamic height
                    plot_data = fi_df.head(min(40, len(fi_df))) # Plot top 40 or fewer
                    sns.barplot(x='importance', y='feature', data=plot_data, palette="viridis") # Use importance
                    plt.title(f'Feature Importance - {SYMBOL_TO_MODEL} - {setup_name}')
                    plt.tight_layout()
                    fi_plot_path = os.path.join(REPORTS_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_FeatImp{script_version_suffix}_{setup_name}.png")
                    plt.savefig(fi_plot_path)
                    plt.close()
                    logger.info(f"Feature importance plot saved to {fi_plot_path}")

                else:
                    logger.warning(f"Feature importance length ({len(fi_values)}) mismatch with feature count ({X_train_final_scaled_df.shape[1]}) for {setup_name}. Skipping FI.")
                    full_report_content += "Feature importance length mismatch. Skipping FI.\n"
            else:
                 logger.warning(f"Could not get feature importance for {setup_name}.")
                 full_report_content += "Could not get feature importance.\n"
        except Exception as e:
            logger.error(f"Error generating feature importance for {setup_name}: {e}", exc_info=True)
            full_report_content += f"Error generating feature importance: {e}\n"

        full_report_content += "="*70 + "\n"


    # --- Save Report and Scaler ---
    report_path = os.path.join(REPORTS_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}{script_version_suffix}_report.txt")
    try:
        with open(report_path, 'w') as f:
            f.write(full_report_content)
        logger.info(f"CatBoost report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error writing report file {report_path}: {e}")


    scaler_path = os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}{script_version_suffix}_scaler_final.joblib")
    try:
        joblib.dump(scaler_final, scaler_path)
        logger.info(f"Final scaler saved: {scaler_path}")
    except Exception as e:
        logger.error(f"Error saving scaler {scaler_path}: {e}", exc_info=True)

    logger.info(f"--- V3 CatBoost (Defaults) Script Finished for {SYMBOL_TO_MODEL} ---")