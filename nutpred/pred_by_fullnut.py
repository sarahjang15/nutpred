import numpy as np
import pandas as pd
import logging
from typing import List, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from nutpred.metrics import r2_manual, smape
from nutpred.preprocess import is_first_mapped
import warnings
import os

logger = logging.getLogger(__name__)

USING_XGB = True
try:
    from xgboost import XGBRegressor
    logger.info("XGBoost available for training")

    warnings.filterwarnings('ignore', module='xgboost')
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
            "objective": ["reg:tweedie", "reg:squarederror"]
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

def _handle_nan_features(X: np.ndarray, feature_set_name: str = "") -> np.ndarray:
    """Helper function to handle NaN values in feature matrices consistently."""
    if np.any(pd.isna(X)):
        if feature_set_name:
            logger.warning(f"NaN values found in features for {feature_set_name}, using median imputation") 
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    return X

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, nutrient: str, feature_set_name: str, group_name: str, sample_type: str) -> dict:
    """Helper function to calculate metrics consistently."""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true + 1e-6, y_pred))
    s = smape(y_true, y_pred)
    
    return {
        "Nutrient": nutrient, 
        "FeatureSet": feature_set_name, 
        "Model": "XGBoost",  # This will be updated by caller
        "Group": group_name, "SampleType": sample_type,
        "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape, "SMAPE": s
    }

def train_tree_models(
    df: pd.DataFrame,  
    feature_sets: dict,
    targets: List[str],
    test_indices: List[int],
    random_state: int = 42,
    cv: int = 3,
    group_name: str = None,
    force_rf: bool = False,
    outdir: str = None,
):
    """
    Train ML models and evaluate on test set + add predictions of all samples.
    """
    # Apply failed filter to get final strict samples for ML
    #success_mask = (df["failed"] == False)
    #success_indices = df[success_mask].index.tolist()
   
    #final_df = df[success_mask]
    final_df = df
    logger.info(f"   Samples used for ML training:: {len(final_df)}")

    test_indices = final_df[final_df.index.isin(test_indices)].index.tolist()
    train_indices = ~np.isin(final_df.index, test_indices)
    #logger.info(f"   Test indices after filtering out failed samples: {len(test_indices)}")

    logger.info(f"Starting ML training with {len(targets)} targets and {len(feature_sets)} feature sets")
    logger.info(f"Parameters: test_size={len(test_indices)}, cv={cv}, force_rf={force_rf}")
    
    
    # Check if we have enough samples for train/test split
    if len(final_df) < 10:
        logger.error(f"Not enough samples for ML training: {len(final_df)} samples")
        raise ValueError(f"Not enough samples for ML training: {len(final_df)} samples. Need at least 10 samples.")

    results = []
    models_dict = {}  # Store trained models for SHAP analysis
    
    for nutrient in targets:
        logger.info(f"Training models for nutrient: {nutrient}")
        y = final_df[nutrient]
        y_tr, y_te = y[train_indices].to_numpy(float), y[test_indices].to_numpy(float)
        
        logger.info(f"Valid training samples for {nutrient}: {len(y_tr)}")

        for feature_set_name, feature_cols in feature_sets.items():
            # Initialize models dict for this feature set if not exists
            if feature_set_name not in models_dict:
                models_dict[feature_set_name] = {}
            logger.info(f"Training ML model with {feature_set_name} features for {nutrient}")
            
            # Check if all feature columns exist
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {feature_set_name}: {missing_cols}")
                continue
            
            Xtr = final_df.loc[train_indices, feature_cols].to_numpy(float)
            Xte = final_df.loc[test_indices, feature_cols].to_numpy(float)
            
            # Handle NaN values consistently
            Xtr = _handle_nan_features(Xtr, feature_set_name)
            Xte = _handle_nan_features(Xte, feature_set_name)
           
            logger.debug(f"Feature matrix shapes: Xtr={Xtr.shape}, Xte={Xte.shape}")
            
            try:
                # Train the model with grid search
                est, param_grid, model_name = _get_estimator_and_grid(force_rf)
                grid = GridSearchCV(est, param_grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
                grid.fit(Xtr, y_tr) # train on training set
                
                # Get the best estimator
                model = grid.best_estimator_
                # save best estimator to config.yaml
                with open(os.path.join(outdir, "config.yaml"), "a") as f:
                    f.write(f"{feature_set_name}_{nutrient}: {model}\n")

                #logger.info(f"Best estimator: {model}")
                
                # Store the trained model for SHAP analysis
                models_dict[feature_set_name][nutrient] = model
                
                # Get predictions on test set with the best estimator
                preds = model.predict(Xte) # predict on test set

                # if below 0, cut to 0
                preds = np.maximum(preds, 0)
                
                # Evaluate on test set (strict_df test samples)
                metrics_result = _calculate_metrics(y_te, preds, nutrient, feature_set_name, group_name, "test")
                metrics_result["Model"] = model.__class__.__name__
                #logger.info(f"Metrics for {feature_set_name} for {nutrient}: {metrics_result}")
                results.append(metrics_result)

                # Get predictions on all samples with the best estimator
                preds_all = model.predict(final_df[feature_cols].to_numpy(float))
                preds_all = np.maximum(preds_all, 0) # if below 0, cut to 0
                final_df[f'{nutrient}_xgb_{group_name}_{feature_set_name}'] = preds_all
                logger.info(f"Added prediction column: {nutrient}_xgb_{group_name}_{feature_set_name}")

            except Exception as e:
                logger.error(f"Error training {feature_set_name} for {nutrient}: {str(e)}")
                continue

    result_df = pd.DataFrame(results)
    id_cols = ["id_col"] + final_df.columns[:3].tolist()
    pred_cols = [col for col in final_df.columns if '_xgb_' in col]
    
    output_df = final_df[id_cols + pred_cols]
    logger.info(f"ML training complete. Generated {len(result_df)} results (test set and remaining strict sample evaluations)")
    
    return result_df, models_dict, output_df

