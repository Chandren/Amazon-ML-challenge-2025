#!/usr/bin/env python3


import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import joblib

sys.path.append(str(Path(__file__).parent))
from src.config import Config
from src.utils import calculate_smape

def load_features(config, dataset='train'):
    """Load all features"""
    print(f"\n Loading {dataset} features...")
    
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
    
    print(f"   Shape: {X.shape}")
    
    # Load target if train
    y = None
    if dataset == 'train':
        df = pd.read_csv(csv_path)
        y = df['price'].values
        
        # Apply log transformation to target
        y_log = np.log1p(y)
        print(f"   Target: min={y.min():.2f}, max={y.max():.2f}, median={np.median(y):.2f}")
        print(f"   Log target: min={y_log.min():.2f}, max={y_log.max():.2f}, median={np.median(y_log):.2f}")
        y = y_log
    
    return X, y, sample_ids

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         IMPROVED MODEL TRAINING                              ║
    ║         Target: SMAPE < 40%                                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    config = Config()
    
    # Load features
    X_full, y_full, _ = load_features(config, 'train')
    
    # Split data
    print("\n Splitting data (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    
    # Improved XGBoost parameters
    print("\n" + "="*80)
    print("TRAINING XGBOOST (IMPROVED)")
    print("="*80)
    
    xgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': 10,
        'tree_method': 'hist',
        'objective': 'reg:squarederror'
    }
    
    xgb_model = XGBRegressor(**xgb_params)
    
    print(" Training XGBoost...")
    eval_set = [(X_val, y_val)]
    xgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=100
    )
    
    # Predict and inverse transform
    xgb_val_pred_log = xgb_model.predict(X_val)
    xgb_val_pred = np.expm1(xgb_val_pred_log)
    y_val_original = np.expm1(y_val)
    
    xgb_smape = calculate_smape(y_val_original, xgb_val_pred)
    print(f"\n XGBoost SMAPE: {xgb_smape:.2f}%")
    
    # Improved LightGBM parameters
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM (IMPROVED)")
    print("="*80)
    
    lgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'max_depth': 8,
        'num_leaves': 128,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'min_split_gain': 0.01,
        'random_state': 42,
        'n_jobs': 10,
        'force_col_wise': True,
        'objective': 'regression'
    }
    
    lgb_model = LGBMRegressor(**lgb_params)
    
    print("Training LightGBM...")
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )
    
    # Predict and inverse transform
    lgb_val_pred_log = lgb_model.predict(X_val)
    lgb_val_pred = np.expm1(lgb_val_pred_log)
    
    lgb_smape = calculate_smape(y_val_original, lgb_val_pred)
    print(f"\n LightGBM SMAPE: {lgb_smape:.2f}%")
    
    # Ensemble
    print("\n" + "="*80)
    print("OPTIMIZING ENSEMBLE")
    print("="*80)
    
    # Optimize weights
    best_smape = float('inf')
    best_weight = 0.5
    
    for w in np.arange(0, 1.05, 0.05):
        ensemble_pred = w * xgb_val_pred + (1-w) * lgb_val_pred
        smape = calculate_smape(y_val_original, ensemble_pred)
        if smape < best_smape:
            best_smape = smape
            best_weight = w
    
    print(f"Optimal XGBoost weight: {best_weight:.2f}")
    print(f"Optimal LightGBM weight: {1-best_weight:.2f}")
    
    ensemble_weights = np.array([best_weight, 1-best_weight])
    
    print(f"\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"XGBoost SMAPE:   {xgb_smape:.2f}%")
    print(f"LightGBM SMAPE:  {lgb_smape:.2f}%")
    print(f"Ensemble SMAPE:  {best_smape:.2f}% ⭐")
    
    if best_smape < 40:
        print(f"\n SMAPE < 40%")
    elif best_smape < 50:
        print(f"\n Close to target.")
    else:
        print(f"\n Still needs improvement")
    
    # Retrain on full data
    print("\n" + "="*80)
    print("RETRAINING ON FULL DATA")
    print("="*80)
    
    print("🔄 Training final XGBoost...")
    xgb_model.fit(X_full, y_full, verbose=100)
    
    print("\n🔄 Training final LightGBM...")
    lgb_model.fit(X_full, y_full)
    
    # Save models
    joblib.dump(xgb_model, config.XGB_MODEL_PATH)
    joblib.dump(lgb_model, config.LGB_MODEL_PATH)
    joblib.dump(ensemble_weights, config.WEIGHTS_PATH)
    
    print("\n Models saved!")
    
    # Generate predictions
    print("\n" + "="*80)
    print("GENERATING TEST PREDICTIONS")
    print("="*80)
    
    X_test, _, sample_ids = load_features(config, 'test')
    
    print(" Predicting...")
    xgb_pred_log = xgb_model.predict(X_test)
    lgb_pred_log = lgb_model.predict(X_test)
    
    # Inverse transform
    xgb_pred = np.expm1(xgb_pred_log)
    lgb_pred = np.expm1(lgb_pred_log)
    
    # Ensemble
    ensemble_pred = ensemble_weights[0] * xgb_pred + ensemble_weights[1] * lgb_pred
    
    # Ensure positive
    ensemble_pred = np.maximum(ensemble_pred, 0.01)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': sample_ids,
        'price': ensemble_pred
    })
    
    submission.to_csv(config.TEST_OUT_CSV, index=False)
    
    print(f"\n Predictions saved to test_out.csv")
    print(f"   Mean: ${ensemble_pred.mean():.2f}")
    print(f"   Median: ${np.median(ensemble_pred):.2f}")
    print(f"   Range: ${ensemble_pred.min():.2f} - ${ensemble_pred.max():.2f}")
    
    print("\n" + "="*80)
    print(" TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Validation SMAPE: {best_smape:.2f}%")
    print("\n📁 Files ready for submission:")
    print("   - test_out.csv")
    print("   - models/ (xgb_model.pkl, lgb_model.pkl, ensemble_weights.pkl)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
