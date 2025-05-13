# train.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna # Import Optuna
from sklearn.model_selection import TimeSeriesSplit # Use this for more robust HPO validation later if needed
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
import joblib # For saving the model
from pathlib import Path
import logging
import matplotlib.pyplot as plt # For feature importance plot
import sys
import copy # To copy params

# Add the parent directory to sys.path to be able to import from dataprocessing
# This assumes train.py is in evaluation/ and config.py is in dataprocessing/
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Import configuration variables
from dataprocessing import config

# --- Logging Setup ---
# Reduce Optuna's default logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
VALIDATION_START_DATE = pd.Timestamp('2024-06-01', tz=config.TARGET_TZ)
CONFIDENCE_THRESHOLD = 0.05

# --- Optuna Settings ---
N_TRIALS = 50 # Number of optimization trials to run (adjust as needed)
OPTUNA_TIMEOUT = 3600 # Optional: Timeout for Optuna study in seconds (e.g., 1 hour)

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
    logger.info(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    logger.info(f"Validation target distribution:\n{y_val.value_counts(normalize=True)}")

    return X_train, y_train, X_val, y_val


# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame) -> float:
    """Optuna objective function to maximize AUC ROC on validation set."""

    # Define hyperparameter search space
    # Reduced ranges initially based on weak previous results - smaller learning rate, less complex trees maybe
    params = {
        'objective': 'binary',
        'metric': 'auc', # Optimize directly for AUC since it measures discrimination
        'boosting_type': 'gbdt',
        'n_estimators': 1000, # Keep high, use early stopping
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 5, 50), # Smaller range
        'max_depth': trial.suggest_int('max_depth', 3, 10), # Limit depth
        'subsample': trial.suggest_float('subsample', 0.6, 0.95), # aka bagging_fraction
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95), # aka feature_fraction
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # L2
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    model = lgb.LGBMClassifier(**params)

    # Train with early stopping based on validation AUC
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    eval_set = [(X_val, y_val)]

    try:
        model.fit(X_train, y_train,
                  eval_set=eval_set,
                  eval_metric='auc', # Evaluate on AUC
                  callbacks=callbacks)

        # Predict on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)

        # Store the best iteration found by early stopping (optional but useful info)
        trial.set_user_attr('best_iteration', model.best_iteration_)

        return auc_score # Optuna maximizes this value

    except Exception as e:
        logger.warning(f"Optuna trial failed: {e}")
        # Report a very low score for failed trials
        return 0.0 # Return low score if trial fails


def train_final_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, best_params: dict):
    """Trains the final LightGBM model using best hyperparameters and early stopping."""
    logger.info("Training final LightGBM model with best hyperparameters...")

    # Make a copy to avoid modifying the original best_params dict
    final_params = copy.deepcopy(best_params)

    # Set fixed parameters and ensure metric is appropriate for final eval
    final_params['objective'] = 'binary'
    final_params['metric'] = 'binary_logloss' # Use logloss for final eval/calibration check
    final_params['n_estimators'] = 10000 # Set very high, rely on early stopping
    final_params['seed'] = 42
    final_params['n_jobs'] = -1
    final_params['verbose'] = -1

    model = lgb.LGBMClassifier(**final_params)

    # Use early stopping based on validation logloss for the final model
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)] # Longer patience for final model
    eval_set = [(X_val, y_val)]

    model.fit(X_train, y_train,
              eval_set=eval_set,
              eval_metric='logloss',
              callbacks=callbacks)

    logger.info(f"Final model training complete. Best iteration: {model.best_iteration_}")
    # Set n_estimators to the best iteration found for the saved model
    model.n_estimators = model.best_iteration_
    return model


# --- Evaluation Function (mostly unchanged, added return dict) ---
def evaluate_model(model: lgb.LGBMClassifier, X_val: pd.DataFrame, y_val: pd.DataFrame, confidence_threshold: float) -> dict:
    """Evaluates the model on the validation set and returns a dictionary of metrics."""
    logger.info("Evaluating final model on validation set...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_val, y_pred_binary)
    metrics['auc'] = roc_auc_score(y_val, y_pred_proba)
    metrics['brier'] = brier_score_loss(y_val, y_pred_proba)
    metrics['logloss'] = log_loss(y_val, y_pred_proba)

    logger.info(f"--- Final Validation Metrics ---")
    logger.info(f"Accuracy (0.5 thresh): {metrics['accuracy']:.4f}")
    logger.info(f"AUC ROC:               {metrics['auc']:.4f}")
    logger.info(f"Brier Score Loss:      {metrics['brier']:.4f} (Lower is better)")
    logger.info(f"Log Loss:              {metrics['logloss']:.4f} (Lower is better)")

    # Simulated Trading Performance
    results_df = pd.DataFrame({'y_true': y_val, 'y_pred_proba': y_pred_proba})
    results_df['trade_direction'] = 0
    results_df.loc[results_df['y_pred_proba'] > (0.5 + confidence_threshold), 'trade_direction'] = 1
    results_df.loc[results_df['y_pred_proba'] < (0.5 - confidence_threshold), 'trade_direction'] = -1
    active_trades = results_df[results_df['trade_direction'] != 0]
    trades_made = len(active_trades)

    metrics['trades_made'] = trades_made
    metrics['trade_accuracy'] = None
    metrics['simplified_pnl'] = None

    if trades_made > 0:
        active_trades['correct_trade'] = ((active_trades['trade_direction'] == 1) & (active_trades['y_true'] == 1)) | \
                                         ((active_trades['trade_direction'] == -1) & (active_trades['y_true'] == 0))
        correct_trades = active_trades['correct_trade'].sum()
        metrics['trade_accuracy'] = correct_trades / trades_made
        metrics['simplified_pnl'] = correct_trades - (trades_made - correct_trades)

        logger.info(f"--- Simulated Trading (Threshold: {confidence_threshold:.2f}, Assumed Market Price: 0.50) ---")
        logger.info(f"Total Validation Days: {len(y_val)}")
        logger.info(f"Days with Trade Signal: {metrics['trades_made']}")
        logger.info(f"Correct Trades:        {correct_trades}")
        logger.info(f"Trade Accuracy:        {metrics['trade_accuracy']:.4f}")
        logger.info(f"Simplified PnL:        {metrics['simplified_pnl']}")
    else:
        logger.info(f"--- Simulated Trading (Threshold: {confidence_threshold:.2f}) ---")
        logger.info("No trades met the confidence threshold in the validation period.")

    return metrics

def plot_feature_importance(model: lgb.LGBMClassifier, feature_names: list, symbol: str):
    """Plots and saves feature importance using the final trained model."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        # Use feature_names from the training data passed to the model
        lgb.plot_importance(model, ax=ax, max_num_features=25, importance_type='gain', feature_name=feature_names)
        ax.set_title(f'Feature Importance (Gain) - {symbol}')
        plt.tight_layout()
        plot_path = MODEL_OUTPUT_DIR / f"{symbol}_feature_importance.png"
        plt.savefig(plot_path)
        logger.info(f"Feature importance plot saved to {plot_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot feature importance for {symbol}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Model Training script with Hyperparameter Optimization...")

    symbols_to_train = config.SYMBOLS + ["ETHBTC"]

    for symbol in symbols_to_train:
        logger.info(f"===== Processing Symbol: {symbol} =====")

        # 1. Load Data
        modeling_df = load_modeling_data(symbol)
        if modeling_df is None: continue

        target_col = f"{symbol}_target_up"
        if target_col not in modeling_df.columns:
            logger.error(f"Target column '{target_col}' not found for {symbol}.")
            continue

        # 2. Split Data
        X_train, y_train, X_val, y_val = time_series_split(modeling_df, target_col, VALIDATION_START_DATE)
        if X_train is None: continue

        # Store feature names for plotting later
        feature_names = list(X_train.columns)

        # 3. Hyperparameter Optimization with Optuna
        logger.info(f"Starting Optuna optimization for {symbol}...")
        study = optuna.create_study(direction='maximize', study_name=f'{symbol}_optimization') # Maximize AUC

        try:
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                n_trials=N_TRIALS,
                timeout=OPTUNA_TIMEOUT, # Optional timeout
                show_progress_bar=True # Show progress
            )
        except Exception as e:
            logger.error(f"Optuna optimization failed for {symbol}: {e}")
            continue

        logger.info(f"Optuna study finished for {symbol}.")
        logger.info(f"Best trial AUC: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters found: {study.best_params}")
        best_params_from_study = study.best_params

        # 4. Train Final Model with Best Hyperparameters
        final_model = train_final_model(X_train, y_train, X_val, y_val, best_params_from_study)

        # 5. Evaluate Final Model
        evaluation_metrics = evaluate_model(final_model, X_val, y_val, CONFIDENCE_THRESHOLD)
        # Maybe log evaluation_metrics here or save them to a file

        # 6. Save Final Model
        model_filename = MODEL_OUTPUT_DIR / f"{symbol}_lgbm_model_optimized.joblib" # Indicate optimized model
        try:
            joblib.dump(final_model, model_filename)
            logger.info(f"Optimized model saved to {model_filename}")
        except Exception as e:
            logger.error(f"Failed to save optimized model for {symbol}: {e}")

        # 7. Feature Importance Plot
        # Pass the feature names explicitly
        plot_feature_importance(final_model, feature_names, symbol)

    logger.info("--- Model Training script finished. ---")