#!/usr/bin/env python3
"""
ROBUST TRAINING SCRIPT - Production Grade
- Comprehensive error handling
- NaN/Null checks at every step
- Memory-efficient processing
- Proper early stopping
- Fallback to XGBoost-only if LightGBM fails
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import joblib
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from src.config import Config
from src.utils import calculate_smape


class TrainingLogger:
    """Robust logging with file and console output"""
    def __init__(self, log_file='training_robust.log'):
        self.log_file = log_file
        
    def log(self, message, level='INFO'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')


def validate_data(X, y, name='data'):
    """Validate data for NaN, Inf, and other issues"""
    logger = TrainingLogger()
    
    # Check X if provided
    if X is not None:
        nan_count_X = np.isnan(X).sum()
        if nan_count_X > 0:
            logger.log(f" {name} X contains {nan_count_X} NaN values!", 'ERROR')
            raise ValueError(f"{name} contains NaN values")
        
        # Check for Inf
        inf_count_X = np.isinf(X).sum()
        if inf_count_X > 0:
            logger.log(f" {name} X contains {inf_count_X} Inf values!", 'ERROR')
            raise ValueError(f"{name} contains Inf values")
    
    # Check y if provided
    if y is not None:
        nan_count_y = np.isnan(y).sum()
        inf_count_y = np.isinf(y).sum()
        
        if nan_count_y > 0:
            logger.log(f" {name} y contains {nan_count_y} NaN values!", 'ERROR')
            raise ValueError(f"{name} target contains NaN values")
        
        if inf_count_y > 0:
            logger.log(f" {name} y contains {inf_count_y} Inf values!", 'ERROR')
            raise ValueError(f"{name} target contains Inf values")
    
    logger.log(f"{name} validation passed - no NaN/Inf values", 'SUCCESS')
    return True


def safe_expm1(values, clip_max=50):
    """Safely apply expm1 with clipping to avoid overflow"""
    # Clip values to prevent overflow in expm1
    clipped = np.clip(values, -10, clip_max)
    result = np.expm1(clipped)
    
    # Replace any remaining inf/nan with reasonable values
    result = np.nan_to_num(result, nan=0.0, posinf=np.exp(clip_max), neginf=0.01)
    
    return result


def load_features(config, dataset='train'):
    """Load all features with validation"""
    logger = TrainingLogger()
    logger.log(f"Loading {dataset} features...")
    
    try:
        if dataset == 'train':
            text_feat = pd.read_csv(config.TRAIN_TEXT_FEATURES)
            text_emb = np.load(config.TRAIN_TEXT_EMBEDDINGS)
            img_feat = np.load(config.TRAIN_IMAGE_FEATURES)
            img_stats = np.load(config.FEATURES_DIR / 'train_image_stats.npy')
            csv_path = config.TRAIN_CSV
        else:
            text_feat = pd.read_csv(config.TEST_TEXT_FEATURES)
            text_emb = np.load(config.TEST_TEXT_EMBEDDINGS)
            img_feat = np.load(config.TEST_IMAGE_FEATURES)
            img_stats = np.load(config.FEATURES_DIR / 'test_image_stats.npy')
            csv_path = config.TEST_CSV
        
        sample_ids = text_feat['sample_id'].values
        text_manual = text_feat.drop('sample_id', axis=1).values
        
        # Combine features
        X = np.hstack([text_manual, text_emb, img_feat, img_stats])
        logger.log(f"Features shape: {X.shape}")
        
        # Validate features
        validate_data(X, None, f'{dataset} features')
        
        # Load target if train
        y = None
        if dataset == 'train':
            df = pd.read_csv(csv_path)
            y = df['price'].values
            
            # Validate target
            if np.any(y <= 0):
                logger.log(f"  Found {(y <= 0).sum()} non-positive prices, clipping to 0.01", 'WARNING')
                y = np.maximum(y, 0.01)
            
            # Apply log transformation to target
            y_log = np.log1p(y)
            
            logger.log(f"Target stats - min: {y.min():.2f}, max: {y.max():.2f}, median: {np.median(y):.2f}")
            logger.log(f"Log target stats - min: {y_log.min():.2f}, max: {y_log.max():.2f}, median: {np.median(y_log):.2f}")
            
            validate_data(None, y_log, f'{dataset} target')
            y = y_log
        
        return X, y, sample_ids
        
    except Exception as e:
        logger.log(f" Error loading features: {e}", 'ERROR')
        raise


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with robust parameters"""
    logger = TrainingLogger()
    logger.log("="*80)
    logger.log("TRAINING XGBOOST")
    logger.log("="*80)
    
    try:
        params = {
            'n_estimators': 1500,
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'gamma': 0.1,
            'random_state': 42,
            'n_jobs': 8,
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
            'max_bin': 256
        }
        
        model = XGBRegressor(**params)
        
        logger.log("Starting XGBoost training...")
        start_time = time.time()
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        logger.log(f"XGBoost trained in {training_time:.1f}s")
        
        # Predict with safety
        val_pred_log = model.predict(X_val)
        validate_data(None, val_pred_log, 'XGBoost predictions')
        
        val_pred = safe_expm1(val_pred_log)
        y_val_original = safe_expm1(y_val)
        
        smape = calculate_smape(y_val_original, val_pred)
        logger.log(f"XGBoost SMAPE: {smape:.2f}%", 'SUCCESS')
        
        return model, smape
        
    except Exception as e:
        logger.log(f" XGBoost training failed: {e}", 'ERROR')
        raise


def train_lightgbm(X_train, y_train, X_val, y_val, timeout=300):
    """Train LightGBM with timeout and robust parameters"""
    logger = TrainingLogger()
    logger.log("="*80)
    logger.log("TRAINING LIGHTGBM")
    logger.log("="*80)
    
    try:
        params = {
            'n_estimators': 500,  # Reduced to avoid getting stuck
            'learning_rate': 0.08,
            'max_depth': 5,
            'num_leaves': 31,
            'min_child_samples': 30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 2.0,
            'reg_lambda': 3.0,
            'min_split_gain': 0.1,
            'random_state': 42,
            'n_jobs': 6,
            'force_col_wise': True,
            'objective': 'regression',
            'verbosity': -1,
            'min_data_in_leaf': 20,
            'max_bin': 128
        }
        
        model = LGBMRegressor(**params)
        
        logger.log("Starting LightGBM training (with timeout protection)...")
        start_time = time.time()
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        training_time = time.time() - start_time
        logger.log(f"LightGBM trained in {training_time:.1f}s")
        
        # Check if it took too long
        if training_time > timeout:
            logger.log(f"  LightGBM training exceeded {timeout}s timeout", 'WARNING')
            return None, None
        
        # Predict with safety
        val_pred_log = model.predict(X_val)
        validate_data(None, val_pred_log, 'LightGBM predictions')
        
        val_pred = safe_expm1(val_pred_log)
        y_val_original = safe_expm1(y_val)
        
        smape = calculate_smape(y_val_original, val_pred)
        logger.log(f"LightGBM SMAPE: {smape:.2f}%", 'SUCCESS')
        
        return model, smape
        
    except Exception as e:
        logger.log(f"  LightGBM training failed: {e}", 'WARNING')
        return None, None


def main():
    logger = TrainingLogger()
    
    logger.log("="*80)
    logger.log("ROBUST TRAINING - STARTING")
    logger.log("="*80)
    
    config = Config()
    
    try:
        # Load features
        X_full, y_full, _ = load_features(config, 'train')
        
        # Split data
        logger.log("Splitting data (80/20)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )
        logger.log(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
        
        # Train XGBoost
        xgb_model, xgb_smape = train_xgboost(X_train, y_train, X_val, y_val)
        
        # Try LightGBM (with fallback)
        lgb_model, lgb_smape = train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Determine best approach
        use_ensemble = lgb_model is not None
        
        if use_ensemble:
            logger.log("="*80)
            logger.log("OPTIMIZING ENSEMBLE")
            logger.log("="*80)
            
            # Get predictions
            xgb_val_pred = safe_expm1(xgb_model.predict(X_val))
            lgb_val_pred = safe_expm1(lgb_model.predict(X_val))
            y_val_original = safe_expm1(y_val)
            
            # Optimize weights
            best_smape = float('inf')
            best_weight = 0.5
            
            for w in np.arange(0, 1.05, 0.05):
                ensemble_pred = w * xgb_val_pred + (1-w) * lgb_val_pred
                smape = calculate_smape(y_val_original, ensemble_pred)
                if smape < best_smape:
                    best_smape = smape
                    best_weight = w
            
            logger.log(f"Optimal XGBoost weight: {best_weight:.2f}")
            logger.log(f"Optimal LightGBM weight: {1-best_weight:.2f}")
            logger.log(f"Ensemble SMAPE: {best_smape:.2f}%", 'SUCCESS')
            
            ensemble_weights = np.array([best_weight, 1-best_weight])
        else:
            logger.log("  Using XGBoost only (LightGBM failed)", 'WARNING')
            best_smape = xgb_smape
            ensemble_weights = np.array([1.0, 0.0])
        
        # Retrain on full data
        logger.log("="*80)
        logger.log("RETRAINING ON FULL DATA")
        logger.log("="*80)
        
        logger.log("Training final XGBoost...")
        start_time = time.time()
        xgb_model.fit(X_full, y_full, verbose=False)
        logger.log(f"XGBoost completed in {time.time()-start_time:.1f}s")
        
        if use_ensemble:
            logger.log("Training final LightGBM (with timeout)...")
            start_time = time.time()
            try:
                lgb_model.fit(X_full, y_full)
                elapsed = time.time() - start_time
                if elapsed < 600:  # 10 minute timeout
                    logger.log(f"LightGBM completed in {elapsed:.1f}s")
                else:
                    logger.log(f"  LightGBM timeout, using XGBoost only", 'WARNING')
                    use_ensemble = False
                    ensemble_weights = np.array([1.0, 0.0])
            except Exception as e:
                logger.log(f"  LightGBM retraining failed: {e}, using XGBoost only", 'WARNING')
                use_ensemble = False
                ensemble_weights = np.array([1.0, 0.0])
        
        # Save models
        joblib.dump(xgb_model, config.XGB_MODEL_PATH)
        logger.log(f"Saved XGBoost model")
        
        if use_ensemble:
            joblib.dump(lgb_model, config.LGB_MODEL_PATH)
            logger.log(f"Saved LightGBM model")
        
        joblib.dump(ensemble_weights, config.WEIGHTS_PATH)
        logger.log(f"Saved ensemble weights")
        
        # Generate predictions
        logger.log("="*80)
        logger.log("GENERATING TEST PREDICTIONS")
        logger.log("="*80)
        
        X_test, _, sample_ids = load_features(config, 'test')
        
        xgb_pred_log = xgb_model.predict(X_test)
        xgb_pred = safe_expm1(xgb_pred_log)
        
        if use_ensemble:
            lgb_pred_log = lgb_model.predict(X_test)
            lgb_pred = safe_expm1(lgb_pred_log)
            final_pred = ensemble_weights[0] * xgb_pred + ensemble_weights[1] * lgb_pred
        else:
            final_pred = xgb_pred
        
        # Ensure positive predictions
        final_pred = np.maximum(final_pred, 0.01)
        
        # Validate predictions
        validate_data(None, final_pred, 'Final predictions')
        
        # Create submission
        submission = pd.DataFrame({
            'sample_id': sample_ids,
            'price': final_pred
        })
        
        submission.to_csv(config.TEST_OUT_CSV, index=False)
        
        logger.log(f"Predictions saved to test_out.csv")
        logger.log(f"   Mean: ${final_pred.mean():.2f}")
        logger.log(f"   Median: ${np.median(final_pred):.2f}")
        logger.log(f"   Min: ${final_pred.min():.2f}")
        logger.log(f"   Max: ${final_pred.max():.2f}")
        
        logger.log("="*80)
        logger.log("TRAINING COMPLETED SUCCESSFULLY!")
        logger.log("="*80)
        logger.log(f"Final Validation SMAPE: {best_smape:.2f}%")
        logger.log(f"Model: {'Ensemble' if use_ensemble else 'XGBoost Only'}")
        
    except Exception as e:
        logger.log(f" CRITICAL ERROR: {e}", 'ERROR')
        import traceback
        traceback.print_exc()
        logger.log(traceback.format_exc(), 'ERROR')
        sys.exit(1)


if __name__ == "__main__":
    main()
