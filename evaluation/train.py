# train.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split # Can use this for splitting AFTER time sort
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
import joblib # For saving the model
from pathlib import Path
import logging
import matplotlib.pyplot as plt # For feature importance plot
import sys

# Add the parent directory to sys.path to be able to import from dataprocessing
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Import configuration variables
from dataprocessing import config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Define the split point for time-series validation
# Example: Use data up to end of 2024 for training, 2025 data for validation
# Adjust VALIDATION_START_DATE based on your data range and desired validation period
VALIDATION_START_DATE = pd.Timestamp('2024-06-01', tz=config.TARGET_TZ)
# VALIDATION_START_DATE = pd.Timestamp('2025-01-01', tz=config.TARGET_TZ)

# Define a threshold for simulated trading decisions
# Only "trade" if the model's predicted probability deviates from 0.5 by this much
# Example: If model predicts > 0.55 or < 0.45
CONFIDENCE_THRESHOLD = 0.05 # Corresponds to 55% / 45% probability thresholds

MODEL_OUTPUT_DIR = config.BASE_DIR / "models"
MODEL_OUTPUT_DIR.mkdir(exist_ok=True) # Create directory if it doesn't exist

def load_modeling_data(symbol: str) -> pd.DataFrame | None:
    """Loads the final modeling data Parquet file for a symbol."""
    filepath = config.PROCESSED_DATA_DIR / f"{symbol}_modeling_data.parquet"
    if not filepath.exists():
        logger.error(f"Modeling data file not found: {filepath}. Run build_features.py first.")
        return None
    try:
        logger.info(f"Loading modeling data from {filepath}...")
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded modeling data for {symbol} with shape {df.shape}")
        if df.empty:
            logger.warning(f"Modeling data for {symbol} is empty.")
            return None
        # Ensure index is sorted (should be from build_features, but double-check)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error loading modeling data {filepath}: {e}")
        return None

def time_series_split(df: pd.DataFrame, target_col: str, validation_start_date: pd.Timestamp):
    """Performs a chronological train/validation split."""
    logger.info(f"Splitting data at {validation_start_date}...")

    train_df = df[df.index < validation_start_date]
    val_df = df[df.index >= validation_start_date]

    if train_df.empty or val_df.empty:
        logger.error("Train or Validation set is empty after split. Check VALIDATION_START_DATE and data range.")
        return None, None, None, None

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    logger.info(f"Train set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
    logger.info(f"Train period: {train_df.index.min()} to {train_df.index.max()}")
    logger.info(f"Validation period: {val_df.index.min()} to {val_df.index.max()}")

    # Check target distribution
    logger.info(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    logger.info(f"Validation target distribution:\n{y_val.value_counts(normalize=True)}")


    return X_train, y_train, X_val, y_val


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
    """Trains a LightGBM classifier with early stopping."""
    logger.info("Training LightGBM model...")

    # Define LGBM parameters (tune these further based on validation results)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss', # Matches Log Loss metric, good for probability calibration
        'boosting_type': 'gbdt',
        'n_estimators': 1000, # Start with a large number, early stopping will find the best
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': -1, # No limit
        'seed': 42,
        'n_jobs': -1, # Use all available cores
        'verbose': -1, # Suppress verbose LightGBM output
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1, # L1 regularization
        'reg_lambda': 0.1, # L2 regularization
    }

    model = lgb.LGBMClassifier(**params)

    # Use early stopping to prevent overfitting and find optimal n_estimators
    # Monitors logloss on the validation set
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    eval_set = [(X_val, y_val)]

    model.fit(X_train, y_train,
              eval_set=eval_set,
              eval_metric='logloss', # Ensure this matches the metric in params
              callbacks=callbacks)

    logger.info(f"Model training complete. Best iteration: {model.best_iteration_}")
    return model

def evaluate_model(model: lgb.LGBMClassifier, X_val: pd.DataFrame, y_val: pd.DataFrame, confidence_threshold: float):
    """Evaluates the model on the validation set with relevant metrics."""
    logger.info("Evaluating model on validation set...")

    # Predict probabilities (important for Polymarket)
    y_pred_proba = model.predict_proba(X_val)[:, 1] # Probability of the positive class (Up=1)

    # Predict binary outcome using 0.5 threshold for standard metrics
    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    # --- Standard Classification Metrics ---
    accuracy = accuracy_score(y_val, y_pred_binary)
    auc = roc_auc_score(y_val, y_pred_proba)
    brier = brier_score_loss(y_val, y_pred_proba)
    logloss = log_loss(y_val, y_pred_proba)

    logger.info(f"--- Validation Metrics ---")
    logger.info(f"Accuracy (0.5 thresh): {accuracy:.4f}")
    logger.info(f"AUC ROC:               {auc:.4f}")
    logger.info(f"Brier Score Loss:      {brier:.4f} (Lower is better)")
    logger.info(f"Log Loss:              {logloss:.4f} (Lower is better)")

    # --- Simulated Trading Performance ---
    # Simulate trades only when confidence threshold is met
    # Assume market price is 0.5 (fair odds) for this simulation
    # Trade 'Up' if P(Up) > 0.5 + threshold
    # Trade 'Down' if P(Up) < 0.5 - threshold (i.e., P(Down) > 0.5 + threshold)

    trades_made = 0
    correct_trades = 0
    simulated_pnl = 0.0 # Assuming $1 bet per trade

    # Create a DataFrame for easier analysis
    results_df = pd.DataFrame({
        'y_true': y_val,
        'y_pred_proba': y_pred_proba
    })

    # Decide trade direction based on probability and threshold
    results_df['trade_direction'] = 0 # 0: No trade, 1: Buy Up, -1: Buy Down
    results_df.loc[results_df['y_pred_proba'] > (0.5 + confidence_threshold), 'trade_direction'] = 1
    results_df.loc[results_df['y_pred_proba'] < (0.5 - confidence_threshold), 'trade_direction'] = -1

    # Filter trades where a decision was made
    active_trades = results_df[results_df['trade_direction'] != 0]
    trades_made = len(active_trades)

    if trades_made > 0:
        # Check if the trade was correct
        # Correct 'Up' trade: trade_direction=1 and y_true=1
        # Correct 'Down' trade: trade_direction=-1 and y_true=0
        active_trades['correct_trade'] = ((active_trades['trade_direction'] == 1) & (active_trades['y_true'] == 1)) | \
                                         ((active_trades['trade_direction'] == -1) & (active_trades['y_true'] == 0))
        correct_trades = active_trades['correct_trade'].sum()
        trade_accuracy = correct_trades / trades_made

        # Calculate PnL assuming $1 bets placed at 0.5 odds (payout $1 if correct, lose $0.5 if wrong - net $0.5 profit)
        # Simplified PnL: +1 for correct, -1 for incorrect
        # More realistic (Polymarket): If bought 'Up' at $p, PnL = (1-p) if win, -p if loss.
        # Let's use simplified first:
        simplified_pnl = correct_trades - (trades_made - correct_trades)

        logger.info(f"--- Simulated Trading (Threshold: {confidence_threshold:.2f}, Assumed Market Price: 0.50) ---")
        logger.info(f"Total Validation Days: {len(y_val)}")
        logger.info(f"Days with Trade Signal: {trades_made}")
        logger.info(f"Correct Trades:        {correct_trades}")
        logger.info(f"Trade Accuracy:        {trade_accuracy:.4f}" if trades_made > 0 else "N/A")
        logger.info(f"Simplified PnL:        {simplified_pnl}" if trades_made > 0 else "N/A")
    else:
        logger.info(f"--- Simulated Trading (Threshold: {confidence_threshold:.2f}) ---")
        logger.info("No trades met the confidence threshold in the validation period.")

    return y_pred_proba # Return probabilities for potential further analysis


def plot_feature_importance(model: lgb.LGBMClassifier, X_train: pd.DataFrame, symbol: str):
    """Plots and saves feature importance."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        lgb.plot_importance(model, ax=ax, max_num_features=25, importance_type='gain') # 'gain' often more informative
        ax.set_title(f'Feature Importance (Gain) - {symbol}')
        plt.tight_layout()
        plot_path = MODEL_OUTPUT_DIR / f"{symbol}_feature_importance.png"
        plt.savefig(plot_path)
        logger.info(f"Feature importance plot saved to {plot_path}")
        plt.close(fig) # Close the plot to free memory
    except Exception as e:
        logger.error(f"Failed to plot feature importance for {symbol}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Model Training script...")

    # Define which symbols to train models for
    symbols_to_train = config.SYMBOLS + ["ETHBTC"] # Add the ratio symbol

    for symbol in symbols_to_train:
        logger.info(f"===== Processing Symbol: {symbol} =====")

        # 1. Load Data
        modeling_df = load_modeling_data(symbol)
        if modeling_df is None:
            logger.warning(f"Skipping {symbol} due to data loading error.")
            continue

        target_col = f"{symbol}_target_up"
        if target_col not in modeling_df.columns:
            logger.error(f"Target column '{target_col}' not found in data for {symbol}. Columns: {modeling_df.columns}")
            continue

        # 2. Split Data
        X_train, y_train, X_val, y_val = time_series_split(modeling_df, target_col, VALIDATION_START_DATE)
        if X_train is None:
            logger.warning(f"Skipping {symbol} due to data splitting error.")
            continue

        # 3. Train Model
        model = train_model(X_train, y_train, X_val, y_val)

        # 4. Evaluate Model
        evaluate_model(model, X_val, y_val, CONFIDENCE_THRESHOLD)

        # 5. Save Model
        model_filename = MODEL_OUTPUT_DIR / f"{symbol}_lgbm_model.joblib"
        try:
            joblib.dump(model, model_filename)
            logger.info(f"Model saved to {model_filename}")
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")

        # 6. Feature Importance
        plot_feature_importance(model, X_train, symbol)

    logger.info("--- Model Training script finished. ---")