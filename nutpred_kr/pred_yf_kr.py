#!/usr/bin/env python3
"""
Korean Ingredient Yield Factor Prediction using CVXPY Optimization

This module predicts yield factors for Korean ingredients based on:
- 8 nutrients of food products (per 100g)
- Ingredient list for each food product  
- 8 nutrients of each ingredient (ratio * per 100g)

The optimization solves: sum(ing_nut_i / yf_i) = food_nut_j
Where yf_i are the predicted yield factors (0 < yf_i <= 1.0)

Output: Yield factors for each ingredient in each food product.
"""

import re
import numpy as np
import pandas as pd
import cvxpy as cp
from numpy.linalg import norm
from typing import List, Dict, Tuple, Optional
import logging
import traceback
from tqdm import tqdm
from nutpred.metrics import r2_manual, smape
from nutpred.preprocess import is_first_mapped
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import nutpred.constants as C

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Scaling functions
# ---------------------------------------------------------------------
def make_residual_weights(X_full, Y_mat, mode="std"):
    """
    Returns weights w (length p) so the loss is || diag(w) (X_j b - y_j) ||^2.
    
    Modes: none, std, pminmax, logiqr
    """
    EPS = 1e-9
    A = np.vstack([X_full, Y_mat])  # shape (K+n, p)
    
    if mode == "none":
        # No scaling - return uniform weights of 1.0
        w = np.ones(A.shape[1])
    elif mode == "std":
        s = np.nanstd(A, axis=0, ddof=1)
        w = 1.0 / np.clip(s, EPS, None)
        
    elif mode == "pminmax":
        low = np.nanpercentile(A, 1.0, axis=0)
        high = np.nanpercentile(A, 99.0, axis=0)
        s = high - low
        w = 1.0 / np.clip(s, EPS, None)
        
    elif mode == "logiqr":
        A_pos = np.where(A > 0, A, 0.0)
        A_log = np.log1p(A_pos)
        q1, q3 = np.nanpercentile(A_log, [25.0, 75.0], axis=0)
        s = q3 - q1
        w = 1.0 / np.clip(s, EPS, None)
    else:
        raise ValueError("scale must be one of: none, std, pminmax, logiqr")
    
    w = np.nan_to_num(w, nan=1.0, posinf=1e6, neginf=1e-6)
    w = np.clip(w, 1e-6, 1e6)
    mu = np.mean(w) if np.isfinite(np.mean(w)) and np.mean(w) > 0 else 1.0
    w = w / mu
    return w.astype(float)

# ---------------------------------------------------------------------
# Core predictor
# ---------------------------------------------------------------------
def predict_yield_factors(food_df: pd.DataFrame, korean_ingnut_df: pd.DataFrame, nut8_cols: List[str], 
                         korean_ingnut_cols: List[str], resolver: str = "rule",
                         ridge: float = 0.0, robust: bool = False, solver_name: str = "osqp", 
                         scale: str = "std", group_name: str = None, error_type: str = "l2", 
                         max_yield_factor: float = 1.0) -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
    """
    Predict yield factors for Korean ingredients using optimization.
    
    Args:
        food_df: DataFrame with food data (products with nutrition and ingredient lists)
        korean_ingnut_df: DataFrame with Korean ingredient-nutrient data (with yield factors)
        nut8_cols: List of nutrient column names in food_df
        korean_ingnut_cols: List of nutrient column names in korean_ingnut_df
        resolver: Variant resolution method ("rule" or "first")
        ridge: Ridge regularization parameter
        robust: Use robust loss function
        solver_name: Optimization solver name
        scale: Scaling method for residuals
        group_name: Group name for output columns
        error_type: Error type for optimization ("l2" for squared error, "l1" for absolute error)
        max_yield_factor: Maximum allowed yield factor (constraint: 0 < yf <= max_yield_factor)
    
    Returns:
        food_df: DataFrame with added yield factor prediction columns
        yf_preds: Yield factor matrix (N, K) - yield factors for each ingredient in each food product
        failed: List of failed optimization indices
    """
    logger.info("Starting Korean ingredient yield factor optimization prediction")
    logger.info(f"Optimization parameters: resolver={resolver}, ridge={ridge}, robust={robust}, solver={solver_name}, scale={scale}")
    
    # Mapping between food_df nutrient columns and korean_ingnut_df columns
    ingnut_mapping = {
        "Energy(kcal)": "Energy",
        "Carbohydrate(g)": "Carbohydrate, by difference",
        "Total fat(g)": "Total lipid (fat)",
        "Protein(g)": "Protein",
        "Sodium(mg)": "Sodium, Na",    
        "Total sugar(g)": "Sugars, total",
        "Saturated fatty acids(g)": "Fatty acids, total saturated",
        "Cholesterol(mg)": "Cholesterol",
        'Calcium(mg)': 'Calcium, Ca',
        'Fiber(g)': 'Fiber, total dietary', 
        'Iron(mg)': 'Iron, Fe'
    }

    try:
        # Prepare data matrices using column mapping
        # X_all: (K, D) nutrient matrix from Korean ingredient data (per 100g)
        X_all = korean_ingnut_df[[ingnut_mapping[c] for c in nut8_cols]].values
        
        # Y_mat: (N, D) target matrix from food product data (per 100g)
        Y_mat = food_df[nut8_cols].values
        
        logger.info(f"Data shapes: X_all={X_all.shape}, Y_mat={Y_mat.shape}, K={len(korean_ingnut_df)} Korean ingredients")
        
        # Map ingredients to indices using all ingredients from korean_ingnut_df
        logger.info("Mapping ingredients to indices using all ingredients from database...")
        ingredient_names = korean_ingnut_df['ing'].tolist()
        mapped_tokens = []  # list of lists, each list is a list of indices of the ingredients

        for j, (_, row) in enumerate(food_df.iterrows()):
            tokens = row.get("ing_list", [])
            if not isinstance(tokens, list):
                tokens = []
            
            row_tokens = []
            for token in tokens:
                try:
                    idx = ingredient_names.index(token)
                    row_tokens.append(idx)
                except ValueError:
                    logger.warning(f"Ingredient {token} not found in ingredient database")
                    continue
            
            mapped_tokens.append(row_tokens)

        logger.info(f"Mapping complete: {len(mapped_tokens)} samples processed using {len(ingredient_names)} ingredients")
        
        logger.info("Running optimization on all samples...")
        preds_w, failed = _run_yf_optimization(
            mapped_tokens, X_all, Y_mat, ridge=ridge, 
            robust=robust, solver_name=solver_name, scale=scale, ingredient_names=ingredient_names, 
            food_df=food_df, error_type=error_type, max_yield_factor=max_yield_factor
        )
        
        logger.info(f"Optimization complete. Weight matrix: {preds_w.shape}")
        logger.info(f"Failed optimizations: {len(failed)}")
        
        # Generate yield factor predictions for all Korean ingredients
        logger.info("Generating yield factor predictions for all Korean ingredients...")
        
        # Convert weights to yield factors (yf = 1/weight) and store in DataFrame
        yf_preds = np.zeros_like(preds_w)  # Initialize yield factor matrix
        yf_preds[preds_w > 0] = 1.0 / preds_w[preds_w > 0]  # Convert weights to yield factors
        yf_preds[preds_w == 0] = 0.0  # Keep zero weights as zero yield factors
        
        # Cap yield factors at max_yield_factor and round to 3 decimal places
        yf_preds = np.clip(yf_preds, 0.0, max_yield_factor)  # Cap at max_yield_factor
        yf_preds = np.round(yf_preds, 3)  # Round to 3 decimal places
        
        # Get yield factors from Korean ingredient database for comparison
        if 'yield_factor' in korean_ingnut_df.columns:
            yf_true = korean_ingnut_df['yield_factor'].values  # (K,) array
            yf_pred = yf_preds @ yf_true  # (N,) array - predicted yield factors for each food product
            food_df[f"yield_factor_pred_{group_name}"] = yf_pred
            logger.info(f"Added prediction column: yield_factor_pred_{group_name}")
        else:
            logger.warning("yield_factor column not found in korean_ingnut_df")
        
        # Add per-row list of yield factors aligned to ing_list
        try:
            opt_pred_yf = []
            for i in range(len(food_df)):
                tokens = food_df.iloc[i].get("ing_list", [])
                if not isinstance(tokens, list):
                    tokens = []
                row_yf = []
                for tok in tokens:
                    try:
                        idx = ingredient_names.index(tok)
                        yf_val = float(yf_preds[i, idx])
                        row_yf.append(yf_val)
                    except ValueError:
                        row_yf.append(0.0)
                opt_pred_yf.append(row_yf)
            food_df[f"opt_pred_yield_factors_{group_name}"] = opt_pred_yf
            logger.info("Added column with per-food yield factors aligned to ing_list")
        except Exception as e:
            logger.warning(f"Failed to add opt_pred_yield_factors column: {e}")

        return food_df, yf_preds, failed
        
    except Exception as e:
        logger.error(f"Error in predict_yield_factors: {e}")
        logger.error(traceback.format_exc())
        raise

def _run_yf_optimization(mapped_tokens: List[List[int]], X_all: np.ndarray, Y_mat: np.ndarray,
                        ridge: float = 0.0, robust: bool = False,
                        solver_name: str = "osqp", scale: str = "std", ingredient_names: List[str] = None, 
                        food_df: pd.DataFrame = None, error_type: str = "l2", max_yield_factor: float = 1.0) -> Tuple[np.ndarray, List[int]]:
    """
    Run optimization for yield factor prediction.
    
    This is similar to _run_optimization in pred_by_ingnut.py but adapted for yield factor prediction.
    
    Args:
        mapped_tokens: List of ingredient indices for each food product
        X_all: (K, D) Korean ingredient nutrient matrix
        Y_mat: (N, D) Food product nutrient matrix  
        ridge: Ridge regularization parameter
        robust: Use robust loss function
        solver_name: Optimization solver name
        scale: Scaling method for residuals
        ingredient_names: List of ingredient names
        food_df: Food DataFrame for debugging
    
    Returns:
        preds_w: (N, K) weight matrix (weights = 1/yf)
        failed: List of failed optimization indices
    """
    N, D = Y_mat.shape
    K = X_all.shape[0]
    
    logger.info(f"Running yield factor optimization: N={N} products, K={K} ingredients, D={D} nutrients")
    
    # Calculate residual weights for scaling
    w_res = make_residual_weights(X_all, Y_mat, mode=scale)
    logger.info(f"Residual weights calculated using {scale} scaling")
    
    # Initialize results
    preds = np.zeros((N, K))
    failed = []
    
    # Set up solver - prefer ECOS for L1 problems
    if error_type == "l1" and solver_name == "auto":
        solver_name = "ecos"  # Prefer ECOS for L1 problems
    
    if solver_name == "osqp":
        solver = cp.OSQP
        solver_kwargs = {"verbose": False, "eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 10000}
    elif solver_name == "clarabel":
        solver = cp.CLARABEL
        solver_kwargs = {"verbose": False, "max_iter": 10000}
    elif solver_name == "scs":
        solver = cp.SCS
        solver_kwargs = {"verbose": False, "max_iters": 10000}
    elif solver_name == "ecos":
        solver = cp.ECOS
        solver_kwargs = {"verbose": False, "max_iters": 10000}
    elif solver_name == "piqp":
        solver = cp.PIQP
        solver_kwargs = {"verbose": False, "max_iter": 10000}
    elif solver_name == "auto":
        # Auto-select based on error type
        if error_type == "l1":
            solver = cp.ECOS
            solver_kwargs = {"verbose": False, "max_iters": 10000}
            solver_name = "ecos"
        else:
            solver = cp.OSQP
            solver_kwargs = {"verbose": False, "eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 10000}
            solver_name = "osqp"
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
    
    logger.info(f"Using {error_type.upper()} optimization with solver: {solver_name}")
    
    # Process each food product
    for j in tqdm(range(N), desc="Optimizing yield factors"):
        try:
            tokens = mapped_tokens[j]
            if len(tokens) == 0:
                logger.warning(f"No ingredients found for product {j}")
                failed.append(j)
                continue
            
            # Extract relevant data for this product
            X_j = X_all[tokens, :]  # (n_ingredients, D) nutrient matrix for this product's ingredients
            y_j = Y_mat[j, :]       # (D,) target nutrition for this product
            
            n_ingredients = len(tokens)
            
            # Optimization variable: ingredient weights
            b = cp.Variable(n_ingredients)
            
            # Objective: minimize || diag(w_res) * (X_j @ b - y_j) ||^2 or ||^1
            residuals = cp.multiply(w_res, X_j.T @ b - y_j)
            
            if error_type == "l1":
                # L1 loss: minimize sum of absolute deviations
                objective = cp.sum(cp.abs(residuals))
            elif error_type == "l2":
                if robust:
                    # Use Huber loss for robustness
                    objective = cp.sum(cp.huber(residuals, M=1.0))
                else:
                    # Standard L2 loss
                    objective = cp.sum_squares(residuals)
            else:
                raise ValueError(f"Unknown error_type: {error_type}. Use 'l1' or 'l2'")
            
            # Add ridge regularization
            if ridge > 0:
                objective += ridge * cp.sum_squares(b)
            
            # Constraints for yield factors
            # weight_i = 1/yf_i, where 0 < yf_i <= max_yield_factor, so weight_i >= 1/max_yield_factor
            min_weight = 1.0 / max_yield_factor
            constraints = [b >= min_weight]  # This ensures 0 < yf_i <= max_yield_factor
            
            # Solve optimization problem
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver=solver, **solver_kwargs)
            
            if problem.status in ["infeasible", "unbounded"]:
                logger.warning(f"Optimization failed for product {j}: {problem.status}")
                failed.append(j)
                continue
            
            # Extract solution
            weights = b.value
            if weights is None or np.any(np.isnan(weights)):
                logger.warning(f"Invalid weights for product {j}")
                failed.append(j)
                continue
            
            # Store weights in the full weight matrix (weights = 1/yf)
            preds[j, tokens] = weights
            
        except Exception as e:
            logger.error(f"Error optimizing product {j}: {e}")
            failed.append(j)
            continue
    
    logger.info(f"Yield factor optimization complete. {len(failed)} samples failed")
    
    return preds, failed

# ---------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------
def eval_yield_factor_predictions(df: pd.DataFrame, group_name: str = None) -> pd.DataFrame:
    """Evaluate yield factor predictions against known values."""
    logger.info("Calculating metrics for yield factor predictions")
    
    rows = []
    
    # Check if we have yield factor predictions and true values
    pred_col = f"yield_factor_pred_{group_name}"
    true_col = "yield_factor"  # Assuming this exists in the data
    
    if pred_col in df.columns and true_col in df.columns:
        y_true = df[true_col].to_numpy(float)
        y_pred = df[pred_col].to_numpy(float)
        
        # Filter out NaN values
        eval_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_eval = y_true[eval_mask]
        y_pred_eval = y_pred[eval_mask]
        
        if len(y_eval) > 0:
            r2_eval = r2_score(y_eval, y_pred_eval)
            rmse_eval = float(mean_squared_error(y_eval, y_pred_eval, squared=False))
            mae_eval = float(mean_absolute_error(y_eval, y_pred_eval))
            mape_eval = float(mean_absolute_percentage_error(y_eval + 1e-6, y_pred_eval))
            s_eval = smape(y_eval + 1e-6, y_pred_eval)
            
            logger.info(f"Yield Factor Prediction Metrics:")
            logger.info(f"  R²={r2_eval:.4f}, RMSE={rmse_eval:.4f}, MAE={mae_eval:.4f}, SMAPE={s_eval:.4f}%")
            
            rows.append({
                "Metric": "Yield Factor", 
                "FeatureSet": "yf_pred", 
                "Model": "Optimization", 
                "Group": group_name, 
                "SampleType": "all", 
                "R2": r2_eval, 
                "RMSE": rmse_eval, 
                "MAE": mae_eval, 
                "MAPE": mape_eval, 
                "SMAPE": s_eval
            })
        else:
            logger.warning("No valid data for yield factor evaluation")
    else:
        logger.warning(f"Missing columns for evaluation: pred={pred_col in df.columns}, true={true_col in df.columns}")
    
    result_df = pd.DataFrame(rows)
    logger.info("Yield factor metrics calculation complete.")
    return result_df

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # Example usage
    logger.info("Korean Ingredient Yield Factor Prediction Module")
    logger.info("This module predicts yield factors for Korean ingredients using CVXPY optimization")
