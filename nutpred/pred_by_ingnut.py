import re
import numpy as np
import pandas as pd
import cvxpy as cp
from numpy.linalg import norm
from typing import List, Dict, Tuple, Optional
import logging
import traceback
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
    
    Modes: std, pminmax, logiqr
    """
    EPS = 1e-9
    A = np.vstack([X_full.T, Y_mat])  # shape (K+n, p)
    
    if mode == "std":
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
        raise ValueError("scale must be one of: std, pminmax, logiqr")
    
    w = np.nan_to_num(w, nan=1.0, posinf=1e6, neginf=1e-6)
    w = np.clip(w, 1e-6, 1e6)
    mu = np.mean(w) if np.isfinite(np.mean(w)) and np.mean(w) > 0 else 1.0
    w = w / mu
    return w.astype(float)

# ---------------------------------------------------------------------
# Core predictor
# ---------------------------------------------------------------------
def predict_ingnut_weights_and_targets(food_df: pd.DataFrame, ingnut_df: pd.DataFrame, nut8_cols: List[str], 
                                     ingnut_cols: List[str], resolver: str = "rule", constraint: str = "nnls_mono",
                                     ridge: float = 0.0, robust: bool = False, solver_name: str = "osqp", 
                                     scale: str = "std", top_list: List[str] = None, group_name: str = None) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[int]]:
    """
    Predict ingredient weights and target nutrients using optimization.
    
    Args:
        food_df: DataFrame with food data
        ingnut_df: DataFrame with ingredient-nutrient data
        nut8_cols: List of nutrient column names in food_df
        ingnut_cols: List of nutrient column names in ingnut_df
        resolver: Variant resolution method ("rule" or "first")
        constraint: Optimization constraint type
        ridge: Ridge regularization parameter
        robust: Use robust loss function
        solver_name: Optimization solver name
        scale: Scaling method for residuals
        test_indices: List of test sample indices (if None, use all samples for training)
    
    Returns:
        snack: DataFrame with added prediction columns
        preds_w: Weight matrix (N, K)
        failed: List of failed optimization indices
    """
    logger.info("Starting ingredient-nutrient optimization prediction")
    logger.info(f"Optimization parameters: resolver={resolver}, constraint={constraint}, ridge={ridge}, robust={robust}, solver={solver_name}, scale={scale}")
    

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
        X_all = ingnut_df[[ingnut_mapping[c] for c in nut8_cols]].values # (K, D) nutrient matrix from ingnut data
        Y_mat = food_df[nut8_cols].values        # (N, D) target matrix from snack data
        
        logger.info(f"Data shapes: X_all={X_all.shape}, Y_mat={Y_mat.shape}, K={len(ingnut_df)} ingredients")
        
        # Map ingredients to ingredients
        logger.info("Mapping ingredients to indices...")
        mapped_tokens = [] # list of lists, each list is a list of indices of the ingredients in the top_list

        for j, (_, row) in enumerate(food_df.iterrows()):
            tokens = row.get("mapped_list_topk_only", [])
            if not isinstance(tokens, list):
                tokens = []
            
            row_tokens = []
            for token in tokens:
                idx = top_list.index(token)
                if idx is not None:
                    row_tokens.append(idx)
            
            mapped_tokens.append(row_tokens)

        
        logger.info(f"Mapping complete: {len(mapped_tokens)} samples processed")
        
        logger.info("Running optimization on all samples...")
        preds_w, failed = _run_optimization(
            mapped_tokens, X_all, Y_mat, constraint=constraint, ridge=ridge, 
            robust=robust, solver_name=solver_name, scale=scale, top_list=top_list, food_df=food_df
        )
        
        logger.info(f"Optimization complete. Weight matrix: {preds_w.shape}")
        logger.info(f"Failed optimizations: {len(failed)}")
        
        # Generate predictions for all samples
        logger.info("Generating predictions for all samples...")
        Y_pred = preds_w @ X_all  # (N, D) prediction matrix for nut8 nutrients
        
        # Add prediction columns for nut8 nutrients
        for j, c in enumerate(nut8_cols):
            food_df[f"{c}_opt_{group_name}"] = Y_pred[:, j]
            logger.info(f"Added prediction column: {c}_opt_{group_name}")
        
        # Now use the same weights to predict target nutrients
        # We need to get the target nutrient values from ingnut_df
        logger.info("Generating predictions for target nutrients using same weights...")
        
        for target, ingnut_col in ingnut_mapping.items():
            if ingnut_col in nut8_cols:
                continue
            if ingnut_col in ingnut_df.columns:
                # Get the target nutrient values from ingnut_df for all variants
                target_values = ingnut_df[ingnut_col].values  # (K,) array
                
                # Use the same weights to predict target nutrients
                target_pred = preds_w @ target_values  # (N,) array
                food_df[f"{target}_opt_{group_name}"] = target_pred
                logger.info(f"Added prediction column: opt_{target} using {ingnut_col}")
            else:
                logger.warning(f"Target nutrient {ingnut_col} not found in ingnut_df, skipping {target}")      
             
        # Add per-row list of weights aligned to mapped_list_topk_only
        try:
            opt_pred_weights = []
            for i in range(len(food_df)):
                tokens = food_df.iloc[i].get("mapped_list_topk_only", [])
                if not isinstance(tokens, list):
                    tokens = []
                row_weights = []
                for tok in tokens:
                    try:
                        idx = top_list.index(tok)
                        row_weights.append(float(preds_w[i, idx]))
                    except ValueError:
                        row_weights.append(0.0)
                opt_pred_weights.append(row_weights)
            food_df[f"opt_pred_weights_{group_name}"] = opt_pred_weights
            logger.info("Added column with per-food weights aligned to mapped_list_topk_only")
        except Exception as e:
            logger.warning(f"Failed to add opt_pred_weights column: {e}")

        # Return the weights matrix that was used for predictions
        return food_df, preds_w, failed
        
    except Exception as e:
        logger.error(f"Error in predict_ingnut_weights_and_targets: {e}")
        logger.error(traceback.format_exc())
        raise

def _run_optimization(mapped_tokens: List[List[int]], X_all: np.ndarray, Y_mat: np.ndarray,
                     constraint: str = "nnls_mono", ridge: float = 0.0, robust: bool = False,
                     solver_name: str = "osqp", scale: str = "std", top_list: List[str] = None, 
                     food_df: pd.DataFrame = None) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Run optimization for all samples and return weights matrix, variant universe, and failed indices.
    
    constraint: constraint mode for optimization
      - "nnls_only": b >= 0,
      - "nnls_mono": b >= 0, b[k] >= b[k+1]
      - "eq1":       b >= 0, sum(b) == 1
      - "eq1_mono":  b >= 0, sum(b) == 1,    b[k] >= b[k+1]
      - "le1":       b >= 0, sum(b) <= 1
      - "le1_mono":  b >= 0, sum(b) <= 1,    b[k] >= b[k+1]
    """
    
    # Choose solver
    solver_map = {"osqp": cp.OSQP, "clarabel": cp.CLARABEL, "scs": cp.SCS, "ecos": cp.ECOS, "piqp": cp.PIQP}
    if solver_name not in solver_map:
        logger.warning(f"Solver {solver_name} not available, falling back to OSQP")
        solver_name = "osqp"
    solver_enum = solver_map[solver_name]
    solver_kwargs = dict(eps_abs=1e-9, eps_rel=1e-9, verbose=False, max_iter=100000, polish=True)
    logger.info(f"Using solver: {solver_name}")
    
    # Prepare optimization matrices
    K = X_all.shape[0]  # Number of variants
    n_snack, p = Y_mat.shape  # Number of samples, number of nutrients
    X_full = X_all.T    # (D, K) - transpose for optimization
    
    
    # Calculate scaling weights
    logger.info(f"Calculating scaling weights using mode: {scale}")
    W_vec = make_residual_weights(X_full, Y_mat, mode=scale)  # length p
    W_mat = np.diag(W_vec)  # (p x p)
    logger.info(f"Scaling weights calculated.")
    
    preds = np.zeros((n_snack, K))
    failed = []
    
    def loss(expr, var):
        """Loss function with optional ridge regularization"""
        base = cp.sum_squares(expr) if not robust else cp.sum(cp.huber(expr, 1.0))
        return base + (ridge * cp.sum_squares(var) if ridge > 0 else 0)
    

    logger.info("Loading food yield factors...")
    import pickle
    food_yield_factors = pickle.load(open("data/yf_by_food.pkl", "rb"))
    
    # Solve per snack
    logger.info("Starting optimization for each sample...")
    
    # jth sample is optimized with the ingredients in the jth list of mapped_tokens
    for j, Sj in enumerate(mapped_tokens):
        if j % 100 == 0:  # Progress update every 100 samples
            logger.info(f"Optimization progress: {j}/{n_snack} samples completed, {len(failed)} failed")
            
        if not Sj:
            logger.debug(f"Sample {j}: No ingredients to optimize")
            continue
            
        Kj = len(Sj)
        Xj = X_full[:, Sj]
        fdc_id = int(food_df.iloc[j]['fdc_id'])
        yf_full = food_yield_factors.get(fdc_id, np.ones(K))
        yf_j = yf_full[Sj]
        Xj = Xj / yf_j[np.newaxis, :]
        var = cp.Variable(Kj, pos=True) # positive
        
        # Set up constraints based on constraint mode
        cons = []
        #cons.append(var >= 0.001) # positive constraint
        if constraint in ("eq1", "eq1_mono"):
            cons.append(cp.sum(var) == 1)
        elif constraint in ("le1", "le1_mono"):
            cons.append(cp.sum(var) <= 1)
        elif constraint in ("nnls_only", "nnls_mono"):
            pass  # Only non-negativity constraint (handled by nonneg=True)
        else:
            raise ValueError(f"Unknown constraint mode: {constraint}")
        
        # Add monotonicity constraints 
        if constraint.endswith("_mono") and Kj > 1:
            cons += [var[k] >= var[k+1] for k in range(Kj - 1)]
        
        # Set up residual and solve
        resid = (W_mat @ Xj) @ var - (W_mat @ Y_mat[j])
        prob = cp.Problem(cp.Minimize(loss(resid, var)), cons)
        
        try:
            prob.solve(solver=solver_enum, **solver_kwargs)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                logger.warning(f"Sample {j}: Solver status '{prob.status}', using fallback")
                failed.append(j)
                wj = np.ones(Kj) / Kj
            else: 
                wj = var.value
                #logger.info(f"Sample {j}: Optimization successful with {Kj} ingredients")
        except Exception as e:
            logger.warning(f"Sample {j}: Optimization failed ({str(e)}), using uniform weights")
            failed.append(j)
            wj = np.ones(Kj) / Kj
        
        # Store results
        full = np.zeros(K)
        full[Sj] = wj
        preds[j] = full
    
    logger.info(f"Optimization complete. {len(failed)} samples failed, using fallback weights")
    
    # Use provided top_list for ingredient names, or create generic names if not provided
    if top_list is not None:
        logger.info(f"Using provided top_list for ingredient names: {len(top_list)} ingredients")
    else:
        # Fallback to generic variant names
        raise ValueError(f"top_list not found")
    
    return preds, failed

# ---------------------------------------------------------------------
# Evaluate ing_pred
# ---------------------------------------------------------------------
def eval_ing_pred(df: pd.DataFrame, group_name: str = None) -> pd.DataFrame:
    """Metrics for ing_pred method (direct predictions from optimizer)."""
    logger.info("Calculating metrics for ingredient prediction method")
    
    logger.info(f"Evaluating metrics for all nutrients: {C.NUT11}")
    
    rows = []
    
    # Evaluate only the nutrients that have predictions
    for nutrient in C.NUT11:
        pred_col = f"{nutrient}_opt_{group_name}"
        if nutrient in df.columns and pred_col in df.columns:
            y = df[nutrient].to_numpy(float)
            yhat = df[pred_col].to_numpy(float)

            eval_mask = (df["failed"] == 0) & ~(pd.isna(y) | pd.isna(yhat))
            
            y_eval = y[eval_mask]
            yhat_eval = yhat[eval_mask]
                    
            r2_eval = r2_score(y_eval, yhat_eval)
            rmse_eval = float(mean_squared_error(y_eval, yhat_eval, squared=False))
            mae_eval = float(mean_absolute_error(y_eval, yhat_eval))
            mape_eval = float(mean_absolute_percentage_error(y_eval + 1e-6, yhat_eval))
            s_eval = smape(y_eval + 1e-6, yhat_eval)
                    
            #logger.info(f"  {nutrient}: R²={r2_eval:.4f}, RMSE={rmse_eval:.4f}, MAE={mae_eval:.4f}, SMAPE={s_eval:.4f}%")
                    
            rows.append({"Nutrient": nutrient, "FeatureSet": "ing_pred", "Model": "Optimization", "Group": group_name, 
                    "SampleType": "all", "R2": r2_eval, "RMSE": rmse_eval, "MAE": mae_eval, "MAPE": mape_eval, "SMAPE": s_eval})
                
        else:
            logger.warning(f"Missing columns for {nutrient}: truth={nutrient in df.columns}, pred={pred_col in df.columns}")
    
    result_df = pd.DataFrame(rows)
    logger.info(f"Metrics calculation complete.")
    return result_df
