import numpy as np
import pandas as pd
import logging
from typing import List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from nutpred.metrics import r2_manual, smape

logger = logging.getLogger(__name__)

USING_XGB = True
try:
    from xgboost import XGBRegressor
    logger.info("XGBoost available for training")
except Exception:
    USING_XGB = False
    from sklearn.ensemble import RandomForestRegressor
    logger.warning("XGBoost not available, using RandomForest")

def _get_estimator_and_grid(force_rf: bool=False):
    if USING_XGB and not force_rf:
        est = XGBRegressor(tree_method="hist", random_state=42)
        grid = {
            "n_estimators": [200, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        model_name = "XGBoost"
        logger.info("Using XGBoost with hyperparameter grid")
    else:
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid = {
            "n_estimators": [300, 600],
            "max_depth": [None, 10, 20],
            "max_features": ["sqrt", 0.5],
            "min_samples_leaf": [1, 2],
        }
        model_name = "RandomForest"
        logger.info("Using RandomForest with hyperparameter grid")
    return est, grid, model_name

def train_eval_sets(
    df: pd.DataFrame,
    targets: List[str],
    feature_sets: dict,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 3,
    force_rf: bool = False
) -> pd.DataFrame:
    """
    Evaluate multiple ML feature sets using cross-validation.
    """
    logger.info(f"Starting ML training with {len(targets)} targets and {len(feature_sets)} feature sets")
    logger.info(f"Parameters: test_size={test_size}, cv={cv}, force_rf={force_rf}")
    
    # Create first_mapped filter
    def is_first_mapped(mapped_list):
        if not isinstance(mapped_list, list) or len(mapped_list) == 0:
            return False
        # Check if the first ingredient is mapped (not empty string)
        return mapped_list[0].strip() != ""
    
    first_mapped_mask = df['mapped_list'].apply(is_first_mapped)
    logger.info(f"First mapped samples: {first_mapped_mask.sum()}/{len(df)}")
    
    # Filter to only first_mapped samples
    df_filtered = df[first_mapped_mask].reset_index(drop=True)
    logger.info(f"Using filtered dataset: {len(df_filtered)} samples")
    
    idx = np.arange(len(df_filtered))
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state)
    logger.info(f"Train/test split: {len(tr)} train, {len(te)} test samples")
    
    est_base, param_grid, model_name = _get_estimator_and_grid(force_rf)

    results = []
    for nutrient in targets:
        logger.info(f"Training models for nutrient: {nutrient}")
        y = df_filtered[nutrient].to_numpy(float)
        y_tr, y_te = y[tr], y[te]
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y_tr)
        if not np.any(valid_mask):
            logger.warning(f"No valid training data for {nutrient}")
            continue
            
        y_tr_valid = y_tr[valid_mask]
        tr_valid = tr[valid_mask]
        
        logger.info(f"Valid training samples for {nutrient}: {len(y_tr_valid)}")

        for feature_set_name, feature_cols in feature_sets.items():
            logger.info(f"Training ML model with {feature_set_name} features for {nutrient}")
            
            # Check if all feature columns exist
            missing_cols = [col for col in feature_cols if col not in df_filtered.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {feature_set_name}: {missing_cols}")
                continue
            
            Xtr = df_filtered.loc[tr_valid, feature_cols].to_numpy(float)
            Xte = df_filtered.loc[te, feature_cols].to_numpy(float)
            
            # Check for NaN values in features
            if np.any(np.isnan(Xtr)) or np.any(np.isnan(Xte)):
                logger.warning(f"NaN values found in features for {feature_set_name}, skipping")
                continue
            
            logger.debug(f"Feature matrix shapes: Xtr={Xtr.shape}, Xte={Xte.shape}")
            
            try:
                est = est_base.__class__(**est_base.get_params())
                grid = GridSearchCV(est, param_grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
                grid.fit(Xtr, y_tr_valid)
                model = grid.best_estimator_
                preds = model.predict(Xte)
                
                # Filter test predictions for valid ground truth
                test_valid_mask = ~np.isnan(y_te)
                if not np.any(test_valid_mask):
                    logger.warning(f"No valid test data for {nutrient}")
                    continue
                    
                y_te_valid = y_te[test_valid_mask]
                preds_valid = preds[test_valid_mask]

                r2 = r2_manual(y_te_valid, preds_valid)
                rmse = float(np.sqrt(mean_squared_error(y_te_valid, preds_valid)))
                mae = float(mean_absolute_error(y_te_valid, preds_valid))
                mape = float(mean_absolute_percentage_error(y_te_valid + 1e-6, preds_valid))
                s = smape(y_te_valid, preds_valid)

                logger.info(f"{nutrient} - {feature_set_name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, SMAPE={s:.4f}%")

                results.append({
                    "Nutrient": nutrient, 
                    "FeatureSet": feature_set_name, 
                    "Model": model.__class__.__name__,
                    "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "SMAPE": s
                })
                
            except Exception as e:
                logger.error(f"Error training {feature_set_name} for {nutrient}: {str(e)}")
                continue

    result_df = pd.DataFrame(results)
    logger.info(f"ML training complete. Generated {len(result_df)} results")
    return result_df
