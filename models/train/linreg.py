# train_linreg_v3_no_optuna.py
import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
SYMBOL_TO_MODEL = "BTCUSDT"

FEATURE_DATA_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/binance_market_data/features_labels_v3"
MODEL_OUTPUT_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/models/trained_models"
REPORTS_OUTPUT_DIR = "/Users/omarabul-hassan/Desktop/projects/trading/models/reports"

FEATURE_FILE_PATH = os.path.join(FEATURE_DATA_DIR, f"{SYMBOL_TO_MODEL}_features_labels_v3.parquet")

TARGET_COLUMN = 'target_up_down'

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

N_SPLITS_CV = 5

# --- Logging Setup ---
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


# --- Main Script ---
if __name__ == "__main__":
    script_version_suffix = "_v3_LinReg_NoOptuna"
    logger.info(f"--- Starting Logistic Regression (No Optuna) for Symbol: {SYMBOL_TO_MODEL} (Features V3) ---")

    for dir_path in [MODEL_OUTPUT_DIR, REPORTS_OUTPUT_DIR]:
        if not os.path.exists(dir_path): os.makedirs(dir_path); logger.info(f"Created: {dir_path}")

    logger.info(f"Loading V3 data: {FEATURE_FILE_PATH}")
    if not os.path.exists(FEATURE_FILE_PATH):
        logger.error(f"FATAL: Feature file not found: {FEATURE_FILE_PATH}. Exiting."); exit()
    df = pd.read_parquet(FEATURE_FILE_PATH)
    logger.info(f"Loaded {len(df)} V3 instances for {SYMBOL_TO_MODEL}.")

    if df.index.name != 'timestamp_utc' or not isinstance(df.index, pd.DatetimeIndex):
        logger.error(f"Index is not 'timestamp_utc' or not DatetimeIndex. Exiting.")
        exit()
    
    y_full = df[TARGET_COLUMN]
    feature_candidates = [col for col in df.columns if col not in [TARGET_COLUMN] + COLUMNS_TO_DROP_PRE_PROCESSING]
    X_all_features_orig = df[feature_candidates].copy()
    logger.info(f"Initial number of candidate features: {len(X_all_features_orig.columns)}")

    all_nan_cols = X_all_features_orig.columns[X_all_features_orig.isnull().all()].tolist()
    if all_nan_cols:
        logger.warning(f"Dropping columns from X that are all NaN: {all_nan_cols}")
        X_all_features_orig.drop(columns=all_nan_cols, inplace=True)

    logger.info("Filling remaining NaNs in feature set with column medians...")
    for col in X_all_features_orig.columns:
        if X_all_features_orig[col].isnull().any():
            median_val = X_all_features_orig[col].median()
            if pd.isna(median_val):
                logger.warning(f"Median for column '{col}' is NaN. Filling with 0.")
                X_all_features_orig[col] = X_all_features_orig[col].fillna(0)
            else:
                X_all_features_orig[col] = X_all_features_orig[col].fillna(median_val)
    
    if X_all_features_orig.isnull().sum().sum() > 0:
        logger.error("NaNs still present after attempting to fill. Exiting.")
        exit()

    obj_cols = X_all_features_orig.select_dtypes(include='object').columns
    if not obj_cols.empty:
        logger.warning(f"Object columns found in final features: {list(obj_cols)}. Dropping them.")
        X_all_features_orig.drop(columns=obj_cols, inplace=True)
    
    logger.info(f"X_all_features shape after pre-processing: {X_all_features_orig.shape}, y_full shape: {y_full.shape}")
    if X_all_features_orig.empty or X_all_features_orig.shape[1] == 0:
        logger.error("Feature set X is empty. Exiting."); exit()

    X_to_use = X_all_features_orig

    logger.info("Training and evaluating Logistic Regression on the last CV split (default parameters)...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    
    # --- CORRECTED PART ---
    splits = list(tscv.split(X_to_use, y_full)) # Get all (train_indices, val_indices) pairs
    
    # Using the last split for train/validation
    final_train_idx, final_val_idx = splits[-1] # Unpack the last tuple in the list
    # --- END CORRECTED PART ---

    X_train_final = X_to_use.iloc[final_train_idx]
    y_train_final = y_full.iloc[final_train_idx]
    X_val_final = X_to_use.iloc[final_val_idx]
    y_val_final = y_full.iloc[final_val_idx]

    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_final)
    X_val_final_scaled = scaler_final.transform(X_val_final)
    
    model_setups_to_try = {
        "Default_L2": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "L1_C1.0": LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000),
        "L2_C100": LogisticRegression(penalty='l2', C=100.0, random_state=42, class_weight='balanced', max_iter=2000),
        "Saga_L1_C0.1": LogisticRegression(penalty='l1', C=0.1, solver='saga', random_state=42, class_weight='balanced', max_iter=2000),
        "Saga_ElasticNet_C0.1_L1_0.5": LogisticRegression(penalty='elasticnet', C=0.1, solver='saga', l1_ratio=0.5, random_state=42, class_weight='balanced', max_iter=3000),
    }
    
    full_report_content = f"Manual Logistic Regression Report for V3: {SYMBOL_TO_MODEL}\n"
    full_report_content += f"Data: {FEATURE_FILE_PATH}\nTotal instances in loaded df: {len(df)}\n"
    full_report_content += f"Features used for Model: {X_to_use.shape[1]} features.\n"
    full_report_content += f"Evaluation on last of {N_SPLITS_CV} TimeSeriesSplits.\n"
    full_report_content += "="*70 + "\n"

    for setup_name, model_params_or_instance in model_setups_to_try.items():
        logger.info(f"\n--- Evaluating Setup: {setup_name} ---")
        full_report_content += f"\n--- Setup: {setup_name} ---\n"
        if isinstance(model_params_or_instance, dict): # Should not happen with current setup
             final_model = LogisticRegression(**model_params_or_instance)
             full_report_content += f"Params: {model_params_or_instance}\n"
        else: # It's already an instance
            final_model = model_params_or_instance # This will assign the pre-configured model instance
            full_report_content += f"Params: {final_model.get_params()}\n"

        try:
            final_model.fit(X_train_final_scaled, y_train_final)
        except Exception as e:
            logger.error(f"Error fitting model for setup {setup_name}: {e}")
            full_report_content += f"Error fitting model: {e}\n"
            continue

        y_pred_proba_final = final_model.predict_proba(X_val_final_scaled)[:, 1]
        y_pred_class_final = (y_pred_proba_final >= 0.5).astype(int)

        final_acc = accuracy_score(y_val_final, y_pred_class_final)
        final_prec = precision_score(y_val_final, y_pred_class_final, zero_division=0)
        final_rec = recall_score(y_val_final, y_pred_class_final, zero_division=0)
        final_f1 = f1_score(y_val_final, y_pred_class_final, zero_division=0)
        try: final_auc = roc_auc_score(y_val_final, y_pred_proba_final)
        except ValueError: final_auc = 0.5 if len(np.unique(y_val_final)) <=1 else np.nan
        final_lloss = log_loss(y_val_final, y_pred_proba_final)
        final_cm = confusion_matrix(y_val_final, y_pred_class_final)

        log_msg = f"Metrics for {setup_name}: Acc={final_acc:.4f}, P={final_prec:.4f}, R={final_rec:.4f}, F1={final_f1:.4f}, AUC={final_auc:.4f}, LogLoss={final_lloss:.4f}"
        logger.info(log_msg)
        logger.info(f"CM for {setup_name}:\n{final_cm}")
        
        full_report_content += log_msg + "\n"
        full_report_content += f"CM:\n{final_cm}\n"

        model_path = os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}{script_version_suffix}_{setup_name}.joblib")
        joblib.dump(final_model, model_path)
        logger.info(f"Model for {setup_name} saved: {model_path}")

        if hasattr(final_model, 'coef_') and X_train_final.shape[1] > 0:
            fi_values = final_model.coef_[0]
            if len(fi_values) == X_train_final.shape[1]:
                fi_df = pd.DataFrame({'feature': X_train_final.columns,
                                        'coefficient': fi_values,
                                        'abs_coefficient': np.abs(fi_values)})
                fi_df = fi_df.sort_values(by='abs_coefficient', ascending=False).reset_index(drop=True)
                top_coeffs_str = fi_df[['feature', 'coefficient']].head(20).to_string()
                logger.info(f"Top Coefficients for {setup_name} (by abs value):\n{top_coeffs_str}")
                full_report_content += f"Top 20 Coefficients:\n{top_coeffs_str}\n"
                
                plt.figure(figsize=(12, max(8, min(20, len(fi_df.head(40)) )//1.5 )))
                plot_data = fi_df.head(min(40, len(fi_df))).sort_values(by='coefficient', ascending=False)
                sns.barplot(x='coefficient', y='feature', data=plot_data, palette="vlag")
                plt.title(f'Coefficients - {SYMBOL_TO_MODEL} - {setup_name}'); plt.tight_layout()
                plt.savefig(os.path.join(REPORTS_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}_Coeffs{script_version_suffix}_{setup_name}.png")); plt.close()
            else:
                logger.warning(f"Coefficient/feature count mismatch for {setup_name}. Skipping FI.")
                full_report_content += "Coefficient/feature count mismatch. Skipping FI.\n"
        full_report_content += "="*70 + "\n"

    report_path = os.path.join(REPORTS_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}{script_version_suffix}_manual_setups_report.txt")
    with open(report_path, 'w') as f:
        f.write(full_report_content)
    logger.info(f"Manual setups report saved to {report_path}")
    
    scaler_path = os.path.join(MODEL_OUTPUT_DIR, f"{SYMBOL_TO_MODEL}{script_version_suffix}_scaler_final.joblib")
    joblib.dump(scaler_final, scaler_path)
    logger.info(f"Final scaler saved: {scaler_path}")

    logger.info(f"--- Manual V3 Logistic Regression Tests Finished for {SYMBOL_TO_MODEL} ---")