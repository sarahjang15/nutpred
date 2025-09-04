import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from nutpred.preprocess import is_first_mapped

logger = logging.getLogger(__name__)

def r2_manual(y_true, y_pred):
    """
    Manual R^2 computed as 1 - (SSE / SST).
    SSE = Σ(y - ŷ)^2,  SST = Σ(y - ȳ)^2
    Returns NaN if SST ~ 0.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    logger.debug(f"Calculating R² for {len(y_true)} samples")
    
    if y_true.size == 0:
        logger.warning("Empty input arrays for R² calculation")
        return np.nan
    
    y_bar = np.mean(y_true)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_bar) ** 2)
    
    if sst <= 1e-12:
        logger.warning("SST too small for R² calculation, returning NaN")
        return np.nan
    
    r2_value = float(1 - (sse / sst))
    logger.debug(f"R² calculation complete: {r2_value:.4f}")
    return r2_value

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    logger.debug(f"Calculating SMAPE for {len(y_true)} samples")
    
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.zeros_like(y_true, dtype=float)
    nz = denom != 0
    
    if not np.any(nz):
        logger.warning("All denominators are zero for SMAPE calculation")
        return 0.0
    
    out[nz] = np.abs(y_true[nz] - y_pred[nz]) / denom[nz]
    smape_value = float(100 * np.mean(out))
    
    logger.debug(f"SMAPE calculation complete: {smape_value:.4f}%")
    return smape_value

def calculate_optimization_metrics(snack_df: pd.DataFrame, test_indices: List[int] = None) -> pd.DataFrame:
    """Calculate metrics for optimization model (ing_pred)"""
    return 

def calculate_ml_metrics(snack_df: pd.DataFrame, feature_sets: Dict[str, List[str]], targets: List[str], 
                        test_size: float = 0.2, cv: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    return

def calculate_all_metrics(snack_df: pd.DataFrame, test_indices: List[int], 
                         feature_sets: Dict[str, List[str]], targets: List[str]) -> pd.DataFrame:
    """Calculate metrics for all models and combine results"""
