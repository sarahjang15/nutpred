import numpy as np
import logging

logger = logging.getLogger(__name__)

def r2_manual(y_true, y_pred):
    """
    Manual R^2 computed as SSR / SST.
    SSR = Σ(ŷ - ȳ)^2,  SST = Σ(y - ȳ)^2
    Returns NaN if SST ~ 0.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    logger.debug(f"Calculating R² for {len(y_true)} samples")
    
    if y_true.size == 0:
        logger.warning("Empty input arrays for R² calculation")
        return np.nan
    
    y_bar = np.mean(y_true)
    ssr = np.sum((y_pred - y_bar) ** 2)
    sst = np.sum((y_true - y_bar) ** 2)
    
    if sst <= 1e-12:
        logger.warning("SST too small for R² calculation, returning NaN")
        return np.nan
    
    r2_value = float(ssr / sst)
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
